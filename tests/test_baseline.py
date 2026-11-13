"""CPU verification of the real-only training and checkpoint entry points."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from industrial_defect_diffusion.classification.baseline import (
    build_model,
    load_checkpoint,
    seed_training,
)
from industrial_defect_diffusion.data.preprocessing import (
    NEUManifestDataset,
    make_loader,
    read_frozen_manifest,
)
from industrial_defect_diffusion.utils.experiment_config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_six_logits_and_backward_on_tiny_fixture():
    seed_training(42)
    model = build_model("none")
    data = torch.stack((torch.zeros(3, 32, 32), torch.ones(3, 32, 32)))
    labels = torch.tensor([0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scores = model(data)
    assert scores.shape == (2, 6)
    initial = nn.functional.cross_entropy(scores, labels).item()
    for _ in range(8):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(data), labels)
        loss.backward()
        optimizer.step()
    final_scores = model(data)
    assert nn.functional.cross_entropy(final_scores, labels).item() < initial / 2
    assert torch.equal(final_scores.argmax(1), labels)


def test_cli_records_and_reloads_without_opening_test_images(frozen_images, tmp_path):
    images, manifest, metadata = frozen_images
    rows, _ = read_frozen_manifest(manifest, metadata)
    for row in rows:
        if row["split"] == "test":
            (images / row["relative_path"]).unlink()

    config = load_config(ROOT / "configs/cpu-smoke.yaml")
    config["paths"].update(
        data_root=images.relative_to(tmp_path).as_posix(),
        manifest=manifest.relative_to(tmp_path).as_posix(),
        artifact_root="artifacts",
    )
    config["classification"]["batch_size"] = 6
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
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
            str(config_path),
            "--seed",
            "42",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record_path = tmp_path / json.loads(result.stdout)["record"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["arm"] == "real_only" and record["status"] == "completed"
    assert record["metrics"]["updates"] == 1
    assert record["metrics"]["training_exposures"] == 6

    checkpoint = record_path.with_name("best.pt")
    validation = NEUManifestDataset(images, rows, split="val", target="classifier")
    batch = next(iter(make_loader(validation, batch_size=6, seed=42)))
    model = load_checkpoint(checkpoint)
    with torch.inference_mode():
        logits = model(batch["image"])
        assert torch.equal(logits, load_checkpoint(checkpoint)(batch["image"]))
    accuracy = (logits.argmax(1) == batch["label"]).sum().item() / 6
    assert accuracy == record["metrics"]["best"]["validation"]["accuracy"]


def test_pretrained_initialization_requires_local_file():
    with pytest.raises(ValueError, match="approved local weights"):
        build_model("IMAGENET1K_V1")
