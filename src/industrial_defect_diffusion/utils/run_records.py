"""Create immutable JSON experiment records with input and artifact fingerprints."""

import json
import platform
import random
import re
from datetime import UTC, datetime
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .experiment_config import ARMS, STAGES, ConfigError, relative_path, validate_config


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _file_reference(root: Path, name: str) -> dict[str, str]:
    relative_path(name, "file reference")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ConfigError(f"File reference escapes workspace: {name}")
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    return {"path": name, "sha256": digest}


def seeded_rng(seed: int) -> random.Random:
    """Return an isolated Python RNG without mutating global state."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ConfigError("seed: expected an unsigned 32-bit integer")
    return random.Random(seed)


def make_run_record(
    config: dict,
    *,
    stage: str,
    seed: int,
    git_commit: str,
    started_at: datetime,
    ended_at: datetime,
    root: Path,
    arm: str | None = None,
    status: str = "completed",
    metrics: dict | None = None,
    artifacts: tuple[str, ...] = (),
    environment: dict | None = None,
    failures: tuple[str, ...] = (),
    classifier_revision: str | None = None,
    dataset_source_sha256: str | None = None,
    synthetic_manifest: str | None = None,
    adapter_sha256: str | None = None,
) -> dict:
    """Build a JSON-ready record from real run inputs and caller-supplied observations."""
    validate_config(config)
    seeded_rng(seed)
    if stage not in STAGES:
        raise ConfigError(f"Unknown stage: {stage}")
    if status not in ("completed", "failed", "interrupted"):
        raise ConfigError("Invalid run status")
    if status != "completed" and not failures:
        raise ConfigError("Failed or interrupted runs require a reason")
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ConfigError("git_commit: expected a full commit SHA")
    if started_at.utcoffset() is None or ended_at.utcoffset() is None or ended_at < started_at:
        raise ConfigError("Run timestamps must be timezone-aware and chronological")
    section = config[stage]
    if stage == "classification":
        if arm not in ARMS or seed not in section["seeds"]:
            raise ConfigError("Classifier arm and seed must match the shared configuration")
    elif arm is not None or seed != section["seed"]:
        raise ConfigError("Run seed must match its stage and arm applies only to classification")
    if config["status"] == "proposed":
        raise ConfigError("Review and approve proposed settings before recording a real run")
    started_at = started_at.astimezone(UTC)
    ended_at = ended_at.astimezone(UTC)
    for field, digest in (
        ("dataset_source_sha256", dataset_source_sha256),
        ("adapter_sha256", adapter_sha256),
    ):
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ConfigError(f"{field}: expected a SHA-256 digest")

    manifest = None if stage == "split" else _file_reference(root, config["paths"]["manifest"])
    diffusion = config["diffusion"]
    classifier = config["classification"]
    if (
        stage in ("classification", "evaluation")
        and config["status"] == "approved"
        and (status == "completed" or classifier_revision is not None)
    ):
        if not isinstance(classifier_revision, str) or not re.fullmatch(
            r"[0-9a-f]{64}",
            classifier_revision,
        ):
            raise ConfigError("classifier_revision: expected the checkpoint SHA-256")
    model = None
    if stage in ("diffusion", "generation"):
        model = {"id": diffusion["model_id"], "revision": diffusion["revision"]}
    elif stage in ("classification", "evaluation"):
        model = {"id": classifier["architecture"], "revision": classifier_revision}

    record = {
        "schema_version": 1,
        "status": status,
        "run_id": f"{started_at:%Y%m%dT%H%M%SZ}_{arm or stage}_{seed}_{git_commit[:8]}",
        "stage": stage,
        "arm": arm,
        "seed": seed,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
        "git_commit": git_commit,
        "resolved_config": config,
        "config_sha256": sha256(_json(config)).hexdigest(),
        "class_mapping_sha256": sha256(_json(config["classes"])).hexdigest(),
        "dataset_manifest": manifest,
        "dataset_source_sha256": dataset_source_sha256,
        "synthetic_manifest": (
            _file_reference(root, synthetic_manifest) if synthetic_manifest is not None else None
        ),
        "adapter_sha256": adapter_sha256,
        "split_manifest_sha256": manifest["sha256"] if manifest else None,
        "model": model,
        "base_model_id": diffusion["model_id"] if stage in ("diffusion", "generation") else None,
        "base_revision": diffusion["revision"] if stage in ("diffusion", "generation") else None,
        "classifier_weights": classifier["weights"]
        if model and model["id"] == "resnet18"
        else None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pyyaml": version("PyYAML"),
            "details": environment or {},
        },
        "metrics": metrics,
        "artifacts": [_file_reference(root, name) for name in artifacts],
        "failures": list(failures),
    }
    # Detach caller-owned objects and reject values that cannot be represented in JSON.
    return json.loads(_json(record))


def write_run_record(path: Path, record: dict) -> None:
    """Write a new run record, refusing to overwrite an earlier record."""
    payload = _json(record)
    with path.open("xb") as stream:
        stream.write(payload)
