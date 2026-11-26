"""CPU fixtures for launcher boundaries, review gates, selection and verified backup."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from industrial_defect_diffusion.utils.experiment_config import load_config
from industrial_defect_diffusion.utils.lab import (
    backup_workspace,
    check_pilot,
    check_quota,
    freeze_selection,
    prepare_config,
    sha256_file,
    workspace_path,
)

ROOT = Path(__file__).resolve().parents[1]
SHELL = shutil.which("powershell") if os.name == "nt" else shutil.which("pwsh")


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("artifacts/\n.cache/\n", encoding="utf-8")
    shutil.copyfile(ROOT / "run_lab.ps1", tmp_path / "run_lab.ps1")
    git(tmp_path, "add", ".gitignore", "run_lab.ps1")
    git(
        tmp_path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "authored launcher fixture",
    )
    return tmp_path


def run_launcher(root, *args, input_text=None):
    if not SHELL:
        pytest.skip("PowerShell is unavailable; Python helper tests still run")
    return subprocess.run(
        [
            SHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "run_lab.ps1"),
            *args,
        ],
        cwd=root.parent,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=45,
    )


def session(root, extra):
    path = root / "artifacts/session.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, **extra}), encoding="utf-8")
    return "artifacts/session.json"


def approval(path, relative):
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "reviewer": "Fixture reviewer",
        "note": "Synthetic fixture approval only",
        "approved_at": "2026-11-18T00:00:00Z",
    }


@pytest.mark.parametrize(
    ("action", "option", "expected"),
    [
        ("plan", "", "scripts/evaluate_classifier.py"),
        ("lora", "resume-full", "'--resume'"),
        ("generate", "resume-full", "'--resume'"),
        ("review", "export", "'--secondary'"),
        ("quality", "kid", "'--approved-weights-sha256'"),
        ("backup", "include-data-and-models", "'--include-assets'"),
    ],
)
def test_dry_run_does_not_execute_or_write(checkout, action, option, expected):
    before = {p.relative_to(checkout): p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    args = ["-DryRun", "-NonInteractive", "-Action", action]
    if option:
        args += ["-Option", option]
    result = run_launcher(checkout, *args)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    after = {p.relative_to(checkout): p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    assert after == before


def test_command_failure_stops_stage_and_is_logged(checkout):
    tools = checkout / "tools"
    tools.mkdir()
    name = "python probe.cmd" if os.name == "nt" else "python-probe"
    probe = tools / name
    probe.write_text(
        "@echo off\necho authored failure\nexit /b 9\n"
        if os.name == "nt"
        else "#!/bin/sh\necho 'authored failure'\nexit 9\n",
        encoding="ascii",
    )
    probe.chmod(0o755)
    path = session(checkout, {"python": probe.relative_to(checkout).as_posix()})
    result = run_launcher(
        checkout, "-Action", "environment", "-SessionFile", path, "-NonInteractive"
    )
    assert result.returncode == 2
    assert "exit 9" in result.stderr
    events = [
        json.loads(line)
        for log in (checkout / "artifacts/lab-launcher").glob("*.jsonl")
        for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["event"] == "exit" and row["details"]["code"] == 9 for row in events)
    assert not any("check_ml_environment.py" in str(row) for row in events)
    assert any(row["event"] == "error" for row in events)


def test_stale_approval_blocks_training_before_python(checkout):
    archive = checkout / "archive.rar"
    archive.write_bytes(b"fixture archive")
    record = approval(archive, "archive.rar")
    archive.write_bytes(b"changed fixture archive")
    path = session(checkout, {"archive": "archive.rar", "approvals": {"dataset-use": record}})
    result = run_launcher(checkout, "-Action", "baseline", "-SessionFile", path, "-NonInteractive")
    assert result.returncode == 2
    assert "stale dataset-use approval" in result.stderr
    assert "scripts/train_real_baseline.py" not in result.stdout


@pytest.mark.parametrize("path", ["../escape.json", "/outside.json"])
def test_session_path_escape_rejected_without_writes(checkout, path):
    result = run_launcher(checkout, "-DryRun", "-Action", "plan", "-SessionFile", path)
    assert result.returncode == 2
    assert not (checkout / "artifacts").exists()


def test_missing_executable_is_failure_not_stale_success(checkout):
    path = session(checkout, {"python": "missing/python.exe"})
    result = run_launcher(
        checkout, "-Action", "environment", "-SessionFile", path, "-NonInteractive"
    )
    assert result.returncode == 2
    assert "Command failed" in result.stderr


def test_templates_preserve_science_and_require_human_approval(tmp_path):
    shutil.copy(ROOT / "configs/baseline.yaml", tmp_path / "base.yaml")
    prepare_config(
        tmp_path, "base.yaml", "artifacts/pilot.yaml", "lora-pilot", "images", "split/manifest.csv"
    )
    config = load_config(tmp_path / "artifacts/pilot.yaml")
    source = load_config(tmp_path / "base.yaml")
    assert config["status"] == "proposed"
    assert config["diffusion"]["max_steps"] == 100
    assert config["diffusion"]["checkpoint_interval"] == 25
    assert config["classification"] == source["classification"]
    assert config["classes"] == source["classes"]
    with pytest.raises(FileExistsError):
        prepare_config(
            tmp_path,
            "base.yaml",
            "artifacts/pilot.yaml",
            "lora-pilot",
            "images",
            "split/manifest.csv",
        )
    with pytest.raises(ValueError):
        workspace_path(tmp_path, "../outside")


def test_selection_preserves_explicit_failed_runs_and_rejects_duplicates(frozen_images, tmp_path):
    _, manifest, metadata = frozen_images
    digest = json.loads(metadata.read_bytes())["manifest_sha256"]
    config = load_config(ROOT / "configs/cpu-smoke.yaml")
    config["paths"]["manifest"] = manifest.relative_to(tmp_path).as_posix()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    record = {
        "schema_version": 1,
        "stage": "classification",
        "arm": "real_only",
        "seed": 42,
        "status": "failed",
        "resolved_config": config,
        "split_manifest_sha256": digest,
    }
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    result = freeze_selection(tmp_path, "config.json", digest, ["run.json"], "selection.json")
    selection = json.loads((tmp_path / "selection.json").read_bytes())
    assert result["selected_pairs"] == 1
    assert selection["runs"][0]["sha256"] == sha256_file(tmp_path / "run.json")
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_selection(tmp_path, "config.json", digest, ["run.json"] * 2, "other.json")
    with pytest.raises(ValueError, match="approved"):
        freeze_selection(tmp_path, "config.json", "a" * 64, ["run.json"], "wrong.json")
    with pytest.raises(FileExistsError):
        freeze_selection(tmp_path, "config.json", digest, ["run.json"], "selection.json")


def test_accepted_quota_is_explicit(tmp_path):
    config = load_config(ROOT / "configs/cpu-smoke.yaml")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    review = tmp_path / "review"
    review.mkdir()
    rows = [{"class_name": name, "quality_status": "accepted"} for name in config["classes"]]
    payload = "\n".join(json.dumps(row) for row in rows)
    (review / "approved.jsonl").write_text(payload, encoding="utf-8")
    assert check_quota(tmp_path, "config.json", "review")["minimum_accepted"] == 6
    (review / "approved.jsonl").write_text("\n".join(payload.splitlines()[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum"):
        check_quota(tmp_path, "config.json", "review")


def test_backup_verifies_copies_and_refuses_overwrite(checkout):
    artifacts = checkout / "artifacts"
    artifacts.mkdir()
    (artifacts / "run.json").write_text('{"fixture": true}', encoding="utf-8")
    (checkout / ".env").write_text("untracked fixture credential", encoding="utf-8")
    destination = checkout.parent / (checkout.name + "-backup")
    result = backup_workspace(checkout, destination)
    index = json.loads((destination / "backup-index.json").read_bytes())
    assert result["files_verified"] == len(index["files"])
    assert not (destination / ".env").exists()
    for item in index["files"]:
        assert sha256_file(destination / item["path"]) == item["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        backup_workspace(checkout, destination)
    with pytest.raises(ValueError, match="outside"):
        backup_workspace(checkout, checkout / "nested-backup")


def test_corrupted_backup_keeps_sources_and_has_no_complete_index(checkout, monkeypatch):
    import industrial_defect_diffusion.utils.lab as lab

    original = (checkout / "run_lab.ps1").read_bytes()
    copy = shutil.copy2

    def corrupt(source, target):
        copy(source, target)
        Path(target).write_bytes(b"damaged copy")

    monkeypatch.setattr(lab.shutil, "copy2", corrupt)
    destination = checkout.parent / (checkout.name + "-bad-copy")
    with pytest.raises(ValueError, match="copy failed"):
        backup_workspace(checkout, destination)
    assert not (destination / "backup-index.json").exists()
    assert (checkout / "run_lab.ps1").read_bytes() == original


def test_helper_errors_are_machine_readable(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lab_support.py"), "config", "../outside.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "error"


def test_cpu_device_uses_the_existing_resolver(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lab_support.py"), "device", "--requested", "cpu"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"device": "cpu"}


def test_manifest_helper_verifies_frozen_metadata(frozen_images, tmp_path):
    _, manifest, metadata = frozen_images
    arguments = [
        sys.executable,
        str(ROOT / "scripts/lab_support.py"),
        "manifest",
        "--csv",
        manifest.relative_to(tmp_path).as_posix(),
    ]
    good = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr


def test_model_receipt_detects_changed_component_without_loading_weights(tmp_path):
    from industrial_defect_diffusion.diffusion.local_model import snapshot_fingerprints

    snapshot = tmp_path / "model"
    for name in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        folder = snapshot / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
    for folder, filename in (
        ("unet", "diffusion_pytorch_model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
        ("text_encoder", "model.safetensors"),
    ):
        (snapshot / folder / filename).write_bytes(b"authored hash fixture; never loaded")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(snapshot_fingerprints(snapshot, None)), encoding="utf-8")
    arguments = [
        sys.executable,
        str(ROOT / "scripts/lab_support.py"),
        "models",
        "--snapshot",
        "model",
        "--approved-receipt",
        "receipt.json",
    ]
    good = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    (snapshot / "unet/diffusion_pytorch_model.safetensors").write_bytes(b"changed")
    bad = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode == 2
    assert "reviewed fingerprint" in json.loads(bad.stdout)["message"]


@pytest.mark.parametrize("change", ["none", "failed", "cpu", "wrong-base", "partial", "artifact"])
def test_full_run_gate_checks_saved_pilot_evidence(tmp_path, change):
    config = load_config(ROOT / "configs/baseline.yaml")
    config["status"] = "approved"
    config["diffusion"]["max_steps"] = 100
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "receipt.json").write_text('{"component": "fixture-hash"}', encoding="utf-8")
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"authored fixture; never loaded")
    record = {
        "schema_version": 1,
        "stage": "diffusion",
        "status": "completed",
        "resolved_config": config,
        "split_manifest_sha256": "a" * 64,
        "base_model_id": config["diffusion"]["model_id"],
        "base_revision": config["diffusion"]["revision"],
        "environment": {
            "details": {
                "device": "cuda:0",
                "gpu": "authored fixture descriptor; no GPU used",
                "base_files": {"component": "fixture-hash"},
            }
        },
        "metrics": {"updates": [{"update": 100, "loss": 0.5}]},
        "adapter_sha256": sha256_file(adapter),
        "artifacts": [{"path": adapter.name, "sha256": sha256_file(adapter)}],
    }
    if change == "failed":
        record["status"] = "failed"
    elif change == "cpu":
        record["environment"]["details"]["device"] = "cpu"
    elif change == "wrong-base":
        record["base_revision"] = "b" * 40
    elif change == "partial":
        record["metrics"]["updates"][0]["update"] = 25
    elif change == "artifact":
        adapter.write_bytes(b"changed fixture")
    (tmp_path / "pilot.json").write_text(json.dumps(record), encoding="utf-8")
    if change == "none":
        assert (
            check_pilot(tmp_path, "pilot.json", "config.json", "a" * 64, "receipt.json")["status"]
            == "verified_pilot_artifacts"
        )
    else:
        with pytest.raises(ValueError):
            check_pilot(tmp_path, "pilot.json", "config.json", "a" * 64, "receipt.json")
