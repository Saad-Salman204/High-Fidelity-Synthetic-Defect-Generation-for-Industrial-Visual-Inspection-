"""Exercise candidate generation without data or downloaded model weights."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random

import pytest
from PIL import Image

from industrial_defect_diffusion.diffusion.training import PROMPTS, TARGETS
from industrial_defect_diffusion.generation.candidates import candidate_plan, generate_candidates
from industrial_defect_diffusion.generation.sampling import (
    MODEL_ID,
    REVISION,
    verify_checkpoint,
)
from industrial_defect_diffusion.utils.experiment_config import CLASS_MAPPING, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    settings = load_config(ROOT / "configs/cpu-smoke.yaml")
    settings["preprocessing"]["diffusion_resolution"] = 8
    settings["generation"].update(
        batch_size=2, candidates_per_class=dict.fromkeys(CLASS_MAPPING, 1)
    )
    return settings


@pytest.fixture
def source():
    return {
        "base_model_id": MODEL_ID,
        "base_revision": REVISION,
        "adapter_sha256": "a" * 64,
        "adapter_checkpoint_sha256": "b" * 64,
        "adapter_checkpoint": "step-000100",
        "real_manifest_sha256": "c" * 64,
        "dataset_archive_sha256": "d" * 64,
    }


def sample(rows):
    return [
        Image.frombytes(
            "RGB",
            (row["width"], row["height"]),
            Random(row["seed"]).randbytes(row["width"] * row["height"] * 3),
        )
        for row in rows
    ]


def test_candidates_reproduce_all_classes_and_pending_provenance(config, source, tmp_path):
    plan = candidate_plan(config, source, {"device": "cpu"})
    left = generate_candidates(tmp_path / "left", plan, sample)
    right = generate_candidates(tmp_path / "right", plan, sample)
    assert left.read_bytes() == right.read_bytes()
    rows = [json.loads(line) for line in left.read_text().splitlines()]
    assert [row["prompt"] for row in rows] == list(PROMPTS)
    assert len(set(PROMPTS)) == 6
    assert [row["seed"] for row in rows] == list(range(42, 48))
    assert len({row["image_id"] for row in rows}) == 6
    for row in rows:
        assert row["class_id"] == CLASS_MAPPING[row["class_name"]]
        assert row["quality_status"] == "pending" and row["reviewer_id"] is None
        assert row["parent_ids"] == [] and row["mask_id"] is None
        assert all(row[key] == value for key, value in source.items())
        assert row["inference_steps"] == config["generation"]["inference_steps"]
        assert row["guidance_scale"] == config["generation"]["guidance_scale"]
        image = left.parent / row["relative_path"]
        assert image.stem == row["image_id"]
        assert sha256(image.read_bytes()).hexdigest() == row["sha256"]
        assert image.read_bytes() == (right.parent / row["relative_path"]).read_bytes()


def test_resume_preserves_batches_and_rejects_corruption(config, source, tmp_path):
    plan = candidate_plan(config, source, {"device": "cpu"})
    output = tmp_path / "resumed"
    seen = []

    def interrupted(rows):
        if seen:
            raise RuntimeError("simulated interruption")
        seen.extend(row["image_id"] for row in rows)
        return sample(rows)

    with pytest.raises(RuntimeError, match="simulated"):
        generate_candidates(output, plan, interrupted)
    assert not (output / "candidates.jsonl").exists()

    def remaining(rows):
        assert not set(seen).intersection(row["image_id"] for row in rows)
        return sample(rows)

    result = generate_candidates(output, plan, remaining, resume=True)
    expected = generate_candidates(tmp_path / "fresh", plan, sample)
    assert result.read_bytes() == expected.read_bytes()
    assert generate_candidates(output, plan, pytest.fail, resume=True) == result
    altered = deepcopy(plan)
    altered["rows"][0]["seed"] = 99
    with pytest.raises(ValueError, match="plan differs"):
        generate_candidates(output, altered, sample, resume=True)
    record = output / "batch-000000" / "batch.json"
    batch = json.loads(record.read_bytes())
    batch[0]["class_id"] = 5
    record.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match="Corrupted candidate provenance"):
        generate_candidates(output, plan, sample, resume=True)


def test_rejects_duplicate_ids_and_invalid_class(config, source, tmp_path):
    plan = candidate_plan(config, source, {})
    plan["rows"][1]["image_id"] = plan["rows"][0]["image_id"]
    with pytest.raises(ValueError, match="Duplicate candidate"):
        generate_candidates(tmp_path / "duplicate", plan, sample)
    config["generation"]["candidates_per_class"]["unknown"] = 1
    with pytest.raises(ValueError, match="candidates_per_class"):
        candidate_plan(config, source, {})


def test_checkpoint_provenance_and_missing_adapter(config, tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    metadata = {
        "manifest_sha256": "c" * 64,
        "dataset_archive_sha256": "d" * 64,
    }
    training_config = deepcopy(config)
    training_config["status"] = "approved"
    training_config["classification"]["weights"] = "IMAGENET1K_V1"
    files = {"unet/model": "base fingerprint"}
    monkeypatch.setattr(
        "industrial_defect_diffusion.generation.sampling.snapshot_fingerprints",
        lambda *_args: files,
    )
    weights = b"fixture adapter"
    checkpoint = {
        "schema_version": 1,
        "target_modules": list(TARGETS),
        "prompts": list(PROMPTS),
        "update": 100,
        "file_sha256": {"adapter.safetensors": sha256(weights).hexdigest()},
        "provenance": {
            "config": training_config,
            "variant": None,
            "base_files": files,
            "git_commit": "e" * 40,
            **metadata,
        },
    }
    payload = json.dumps(checkpoint).encode()
    (adapter / "checkpoint.json").write_bytes(payload)
    approval = sha256(payload).hexdigest()
    with pytest.raises(FileNotFoundError, match="adapter.safetensors"):
        verify_checkpoint(tmp_path, adapter, config, metadata, approval)
    (adapter / "adapter.safetensors").write_bytes(weights)
    result = verify_checkpoint(tmp_path, adapter, config, metadata, approval)
    assert result["adapter_checkpoint"] == "step-000100"
    with pytest.raises(ValueError, match="approved SHA"):
        verify_checkpoint(tmp_path, adapter, config, metadata, "f" * 64)
    config["diffusion"]["revision"] = "f" * 40
    with pytest.raises(ValueError, match="pinned SD v1.5"):
        verify_checkpoint(tmp_path, adapter, config, metadata, approval)


def test_tiny_local_pipeline_samples_with_real_peft_adapter(config, tmp_path):
    import torch
    from test_lora_training import CPU, components

    from industrial_defect_diffusion.diffusion.checkpoints import save_checkpoint
    from industrial_defect_diffusion.diffusion.training import make_optimizer, prepare_adapter
    from industrial_defect_diffusion.generation.sampling import make_sampler

    parts = components()
    model = tmp_path / "model"
    for name in ("unet", "vae", "text_encoder", "scheduler"):
        getattr(parts, name).save_pretrained(model / name)
    tokenizer = model / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.json").write_text(
        json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1}),
        encoding="utf-8",
    )
    (tokenizer / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 8}),
        encoding="utf-8",
    )
    settings = config["diffusion"]
    prepare_adapter(parts, settings, CPU)
    adapter = tmp_path / "adapter"
    save_checkpoint(
        adapter,
        parts,
        make_optimizer(parts, settings),
        torch.amp.GradScaler("cuda", enabled=False),
        CPU,
        {},
        (1, 0, 1),
    )
    sampler = make_sampler(
        model, adapter, {"variant": None, "training_settings": settings}, CPU, "fp32"
    )
    rows = [
        {
            "prompt": PROMPTS[i],
            "negative_prompt": "",
            "seed": i,
            "height": 8,
            "width": 8,
            "inference_steps": 1,
            "guidance_scale": 2.0,
        }
        for i in range(2)
    ]
    first = [image.tobytes() for image in sampler(rows)]
    assert first == [image.tobytes() for image in sampler(rows)]
    assert first[0] != first[1]
