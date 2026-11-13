"""Verify the three classifier arms on small, local CPU fixtures."""

import json
import subprocess
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random

import pytest
import torch
from PIL import Image

from industrial_defect_diffusion.classification.arms import make_arm_loader
from industrial_defect_diffusion.classification.baseline import load_checkpoint
from industrial_defect_diffusion.data.preprocessing import NEUManifestDataset, read_frozen_manifest
from industrial_defect_diffusion.data.synthetic import ApprovedSyntheticDataset
from industrial_defect_diffusion.utils.experiment_config import ARMS, CLASS_MAPPING, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def comparison(tmp_path, frozen_images):
    real, manifest, metadata = frozen_images
    rows, details = read_frozen_manifest(manifest, metadata)
    config = load_config(ROOT / "configs/cpu-smoke.yaml")
    config["preprocessing"]["classifier_resolution"] = 32
    config["classification"].update(batch_size=4, mixed_synthetic_fraction=0.25)
    config["paths"].update(
        data_root=real.relative_to(tmp_path).as_posix(),
        manifest=manifest.relative_to(tmp_path).as_posix(),
        artifact_root="artifacts",
    )
    pool = tmp_path / "generated"
    pool.mkdir()
    rng = Random(14)
    approved = []
    for label, class_id in CLASS_MAPPING.items():
        name = f"{class_id}.png"
        path = pool / name
        Image.frombytes("L", (40, 40), rng.randbytes(1600)).save(path)
        approved.append(
            {
                "schema_version": 1,
                "image_id": f"fixture-{label}",
                "relative_path": name,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "class_id": class_id,
                "class_name": label,
                "prompt": f"fixture {label}",
                "seed": 42,
                "run_id": "synthetic-fixture",
                "scheduler": "fixture-scheduler",
                "inference_steps": 1,
                "width": 40,
                "height": 40,
                "reviewer_id": "fixture-reviewer",
                "base_revision": config["diffusion"]["revision"],
                "adapter_sha256": "b" * 64,
                "real_manifest_sha256": details["manifest_sha256"],
                "quality_status": "accepted",
                "parent_ids": [],
                "mask_id": None,
            }
        )
    return config, rows, details, pool, approved


def jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def synthetic(comparison):
    config, rows, details, pool, approved = comparison
    return ApprovedSyntheticDataset(
        pool,
        jsonl(approved),
        rows,
        details["manifest_sha256"],
        base_revision=config["diffusion"]["revision"],
        resolution=32,
        flip_probability=0,
    )


def test_sampler_source_membership_class_balance_and_reproducibility(comparison, frozen_images):
    config, rows, _, _, approved = comparison
    real = NEUManifestDataset(
        frozen_images[0],
        rows,
        split="train",
        target="classifier",
        resolution=32,
    )
    generated = synthetic(comparison)
    settings = {**config["classification"], "max_steps": 6}
    expected = (
        {"real": 4},
        {"synthetic": 4},
        {"real": 3, "synthetic": 1},
    )
    allowed = {
        "real": {row["image_id"] for row in rows if row["split"] == "train"},
        "synthetic": {row["image_id"] for row in approved},
    }
    for arm, quotas in zip(ARMS, expected, strict=True):
        loader = make_arm_loader(arm, real, generated, settings, 42)
        assert list(loader.batch_sampler) == list(loader.batch_sampler)
        totals = {source: Counter() for source in quotas}
        for batch in loader:
            assert Counter(batch["source"]) == quotas
            for source, label, image_id in zip(
                batch["source"],
                batch["label"].tolist(),
                batch["image_id"],
                strict=True,
            ):
                assert image_id in allowed[source]
                totals[source][label] += 1
        for source, count in quotas.items():
            assert totals[source] == {label: count for label in range(6)}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("class_id", 7, "class label"),
        ("quality_status", "pending", "only accepted"),
        ("quality_status", "rejected", "only accepted"),
        ("adapter_sha256", None, "provenance"),
        ("reviewer_id", "", "provenance"),
        ("real_manifest_sha256", "f" * 64, "different frozen"),
        ("relative_path", "../outside.png", "traversal"),
        ("split", "test", "training-only"),
    ],
)
def test_invalid_synthetic_record(comparison, field, value, message):
    comparison[4][0][field] = value
    with pytest.raises(ValueError, match=message):
        synthetic(comparison)


def test_held_out_ancestry_and_exact_overlap_are_rejected(comparison):
    row = next(row for row in comparison[1] if row["split"] == "test")
    candidate = comparison[4][0]
    candidate["parent_ids"] = [row["image_id"]]
    with pytest.raises(ValueError, match="only from the frozen real train"):
        synthetic(comparison)
    candidate.update(parent_ids=[], sha256=row["sha256"])
    with pytest.raises(ValueError, match="overlaps real"):
        synthetic(comparison)


def test_changed_synthetic_bytes_rejected_before_sampling(comparison):
    path = comparison[3] / comparison[4][-1]["relative_path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        synthetic(comparison)


def test_cli_trains_all_three_with_common_validation_and_separate_checkpoints(
    comparison,
    frozen_images,
    tmp_path,
):
    config, rows, _, pool, approved = comparison
    for row in rows:
        if row["split"] == "test":
            (frozen_images[0] / row["relative_path"]).unlink()
    (tmp_path / "approved.jsonl").write_bytes(jsonl(approved))
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "synthetic fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_real_baseline.py"),
            "config.json",
            "--seed",
            "42",
            "--arm",
            "all",
            "--synthetic-root",
            str(pool),
            "--synthetic-manifest",
            "approved.jsonl",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    runs = json.loads(result.stdout)["runs"]
    assert list(runs) == list(ARMS)
    revisions = set()
    checkpoints = set()
    expected = ({"real": 4}, {"synthetic": 4}, {"real": 3, "synthetic": 1})
    for arm, counts in zip(ARMS, expected, strict=True):
        path = tmp_path / runs[arm]["record"]
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["arm"] == arm and record["status"] == "completed"
        assert record["seed"] == 42 and record["metrics"]["updates"] == 1
        assert record["metrics"]["validation_image_ids"] == [
            row["image_id"] for row in rows if row["split"] == "val"
        ]
        assert {
            source: item["exposures"]
            for source, item in record["metrics"]["training_by_source"].items()
        } == counts
        assert bool(record["synthetic_manifest"]) == (arm != "real_only")
        assert record["metrics"]["repeated_training_exposures"] == 0
        events = [
            json.loads(line)["event"]
            for line in path.with_name("training.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert events == ["started", "validation", "completed"]
        checkpoints.add(path.with_name("best.pt"))
        revisions.add(record["model"]["revision"])
        with torch.inference_mode():
            assert load_checkpoint(path.with_name("best.pt"))(torch.zeros(1, 3, 32, 32)).shape == (
                1,
                6,
            )
    assert len(checkpoints) == len(revisions) == 3
