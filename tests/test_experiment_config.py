"""Verify configuration and records without loading models or dataset images."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from industrial_defect_diffusion.utils.experiment_config import (
    ConfigError,
    load_config,
    validate_config,
)
from industrial_defect_diffusion.utils.run_records import (
    make_run_record,
    seeded_rng,
    write_run_record,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/cpu-smoke.yaml")


def test_examples_and_json_round_trip(config, tmp_path):
    assert load_config(ROOT / "configs/baseline.yaml")["status"] == "proposed"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_config(path) == config
    assert seeded_rng(42).sample(range(100), 8) == seeded_rng(42).sample(range(100), 8)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("split", "ratios", [0.6, 0.2, 0.1], "sum to one"),
        ("split", "group_duplicates", False, "must be true"),
        ("paths", "manifest", "../unsafe.csv", "without traversal"),
        ("diffusion", "revision", None, "immutable"),
        ("diffusion", "learning_rate", float("nan"), "finite number"),
        ("classification", "arms", ["real_only"], "canonical arms"),
        ("classification", "batch_size", 3, "whole counts"),
        ("classification", "epochs", 2, "exactly one"),
        ("classes", "crazing", False, "canonical NEU"),
        ("evaluation", "device", "gpu", "expected cpu"),
    ],
)
def test_invalid_config(config, section, key, value, message):
    config[section][key] = value
    with pytest.raises(ConfigError, match=message):
        validate_config(config)


@pytest.mark.parametrize("content", ["split: [", "status: smoke\nstatus: approved\n"])
def test_cli_invalid_yaml(tmp_path, content):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(content, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_experiment_config.py"), str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "error"
    assert not result.stderr


def test_run_record_hashes_provenance_and_exclusive_output(config, tmp_path):
    manifest = tmp_path / config["paths"]["manifest"]
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"synthetic test inventory")
    instant = datetime(2026, 11, 13, tzinfo=UTC)
    kwargs = dict(
        stage="classification",
        seed=42,
        git_commit="a" * 40,
        started_at=instant,
        ended_at=instant,
        root=tmp_path,
        arm="real_only",
    )
    record = make_run_record(config, **kwargs)
    assert record == make_run_record(config, **kwargs)
    assert record["metrics"] is None
    assert record["split_manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
    assert record["model"]["id"] == "resnet18"
    output = tmp_path / "run.json"
    write_run_record(output, record)
    assert json.loads(output.read_text()) == record
    with pytest.raises(FileExistsError):
        write_run_record(output, record)
    manifest.write_bytes(b"changed")
    assert (
        make_run_record(config, **kwargs)["split_manifest_sha256"]
        != record["split_manifest_sha256"]
    )
    with pytest.raises(ConfigError, match="require a reason"):
        make_run_record(config, status="failed", **kwargs)
