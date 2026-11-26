"""Plan reproducible candidates and persist resumable, verified batches."""

import io
import json
import os
import tempfile
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from PIL import Image

from ..diffusion.local_model import file_sha256
from ..diffusion.training import PROMPTS
from ..utils.experiment_config import CLASS_MAPPING, validate_config
from .sampling import SCHEDULER


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def candidate_plan(config: dict, source: dict, runtime: dict) -> dict:
    """Create stable class IDs, seed assignments, and an immutable run identity."""
    validate_config(config)
    settings = config["generation"]
    size = config["preprocessing"]["diffusion_resolution"]
    if not 8 <= size <= 2048:
        raise ValueError("Generation resolution must be between 8 and 2048 pixels")
    counts = settings.get("candidates_per_class")
    if counts is None:
        per_class, remainder = divmod(settings["maximum_candidates"], 6)
        counts = {label: per_class + (index < remainder) for label, index in CLASS_MAPPING.items()}
    if min(counts.values()) < 1:
        raise ValueError("Candidate budget must include all six classes")
    identity = {
        "schema_version": 1,
        "config": config,
        "source": source,
        "runtime": runtime,
        "counts": counts,
        "prompts": list(PROMPTS),
        "scheduler": SCHEDULER,
        "eta": 0.0,
        "scheduler_overrides": {"clip_sample": False, "steps_offset": 1},
    }
    run_id = sha256(_json(identity)).hexdigest()
    rows = []
    for label, class_id in sorted(CLASS_MAPPING.items(), key=lambda item: item[1]):
        for index in range(counts[label]):
            seed = (settings["seed"] + len(rows)) % 2**32
            image_id = f"{run_id}-{class_id}-{index:05d}"
            rows.append(
                {
                    "schema_version": 1,
                    "image_id": image_id,
                    "run_id": run_id,
                    "class_name": label,
                    "class_id": class_id,
                    "seed": seed,
                    "prompt": PROMPTS[class_id],
                    "negative_prompt": settings.get("negative_prompt", ""),
                    "width": size,
                    "height": size,
                    "scheduler": SCHEDULER,
                    "eta": 0.0,
                    "inference_steps": settings["inference_steps"],
                    "guidance_scale": settings["guidance_scale"],
                    "batch_size": settings["batch_size"],
                    "precision": settings.get("precision", "fp32"),
                    "parent_ids": [],
                    "mask_id": None,
                    "split": "train",
                    "quality_status": "pending",
                    "rejection_reason": None,
                    "reviewer_id": None,
                    **{
                        key: source[key]
                        for key in (
                            "base_model_id",
                            "base_revision",
                            "adapter_sha256",
                            "adapter_checkpoint",
                            "adapter_checkpoint_sha256",
                            "real_manifest_sha256",
                            "dataset_source_sha256",
                        )
                    },
                }
            )
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate identifiers in plan")
    return {**identity, "run_id": run_id, "rows": rows}


def _verified_batch(directory: Path, expected: list[dict]) -> list[dict]:
    if directory.is_symlink():
        raise ValueError("Batch directory may not be a symbolic link")
    records = json.loads((directory / "batch.json").read_bytes())
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError(f"Corrupted batch record: {directory.name}")
    for row, original in zip(records, expected, strict=True):
        if (
            not isinstance(row, dict)
            or row.keys() != original.keys() | {"sha256"}
            or any(row[key] != value for key, value in original.items())
        ):
            raise ValueError(f"Corrupted candidate provenance in {directory.name}")
        image = directory / Path(original["relative_path"]).name
        if image.is_symlink() or file_sha256(image) != row["sha256"]:
            raise ValueError(f"Candidate image hash mismatch: {original['image_id']}")
    return records


def _publish_manifest(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError("Existing candidate manifest differs from verified batches")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        # Hard-link publication fails rather than replacing an existing manifest.
        os.link(temporary, path)
    finally:
        temporary.unlink()


def generate_candidates(
    output: Path,
    plan: dict,
    sampler: Callable[[list[dict]], list[Image.Image]],
    *,
    resume: bool = False,
) -> Path:
    """Persist PNGs and candidate JSONL, preserving completed batches on interruption."""
    if len({row["image_id"] for row in plan["rows"]}) != len(plan["rows"]):
        raise ValueError("Duplicate candidate identifiers in plan")
    if output.exists() and not resume:
        raise ValueError("Output exists. Use --resume with the identical generation inputs")
    output.mkdir(parents=True, exist_ok=resume)
    lock = output / ".generation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("Active generation. Remove only after verifying the process stopped.\n")
    try:
        return _generate(output, plan, sampler)
    finally:
        lock.unlink()


def _generate(output: Path, plan: dict, sampler) -> Path:
    plan_path = output / "plan.json"
    plan_bytes = _json(plan)
    if plan_path.exists():
        if plan_path.is_symlink() or plan_path.read_bytes() != plan_bytes:
            raise ValueError("Existing generation plan differs from the requested inputs")
    else:
        if any(path.name != ".generation.lock" for path in output.iterdir()):
            raise ValueError("Output contains files without a generation plan")
        with plan_path.open("xb") as stream:
            stream.write(plan_bytes)
    rows = plan["rows"]
    batch_size = plan["config"]["generation"]["batch_size"]
    batches = []
    for offset in range(0, len(rows), batch_size):
        batch_name = f"batch-{offset // batch_size:06d}"
        expected = [
            {**row, "relative_path": f"{batch_name}/{row['image_id']}.png"}
            for row in rows[offset : offset + batch_size]
        ]
        batches.append((output / batch_name, expected))
    finished = {
        directory: _verified_batch(directory, expected)
        for directory, expected in batches
        if directory.exists()
    }
    manifest = output / "candidates.jsonl"
    if manifest.exists() and len(finished) != len(batches):
        raise ValueError("Candidate manifest exists but a committed batch is missing")
    records = []
    for directory, expected in batches:
        if directory in finished:
            records.extend(finished[directory])
            continue
        images = sampler(expected)
        if len(images) != len(expected):
            raise ValueError("Sampler returned the wrong number of images")
        # A failed batch is discarded; only the directory rename commits it.
        with tempfile.TemporaryDirectory(dir=output, prefix=".batch-") as temporary:
            staging = Path(temporary) / "ready"
            staging.mkdir()
            batch = []
            for row, image in zip(expected, images, strict=True):
                if not isinstance(image, Image.Image) or image.size != (
                    row["width"],
                    row["height"],
                ):
                    raise ValueError("Sampler returned an invalid image or dimensions")
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
                payload = buffer.getvalue()
                (staging / Path(row["relative_path"]).name).write_bytes(payload)
                batch.append({**row, "sha256": sha256(payload).hexdigest()})
            (staging / "batch.json").write_bytes(_json(batch))
            staging.rename(directory)
        records.extend(batch)
    _publish_manifest(manifest, b"".join(_json(row) for row in records))
    return manifest
