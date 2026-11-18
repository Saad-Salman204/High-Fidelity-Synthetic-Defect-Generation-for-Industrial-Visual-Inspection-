"""Small file operations for the lab launcher, without model or dataset downloads."""

import json
import math
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from hashlib import file_digest
from pathlib import Path

import yaml

from .experiment_config import ARMS, load_config, relative_path, validate_config


def sha256_file(path: Path) -> str:
    """Fingerprint file bytes without loading the entire artifact."""
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def workspace_path(root: Path, value: str) -> Path:
    """Resolve a portable path and reject traversal or links outside the workspace."""
    path = (root / relative_path(value, "lab path")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Lab path escapes the workspace")
    return path


def write_json(path: Path, value: dict) -> None:
    """Create an immutable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare_config(root: Path, source: str, output: str, kind: str, data: str, manifest: str):
    """Copy a validated config into a proposed lab template for human editing."""
    config = load_config(workspace_path(root, source))
    config["status"] = "proposed"
    workspace_path(root, data)
    workspace_path(root, manifest)
    config["paths"].update(data_root=data, manifest=manifest)
    if kind == "classifier-pilot":
        config["classification"].update(max_steps=10, epochs=None, validation_interval=5)
    elif kind == "lora-pilot":
        config["diffusion"].update(max_steps=100, epochs=None, checkpoint_interval=25)
    elif kind == "generation-pilot":
        config["generation"].update(
            candidates_per_class={label: 1 for label in config["classes"]},
            minimum_accepted=6,
            target_accepted=6,
            maximum_candidates=6,
        )
    elif kind != "base":
        raise ValueError("Unknown lab template kind")
    validate_config(config)
    path = workspace_path(root, output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return {"path": output, "status": "proposed", "sha256": sha256_file(path)}


def freeze_selection(
    root: Path, config_path: str, manifest_hash: str, records: list[str], output: str
):
    """Record explicit human run choices without predicting or reading test pixels."""
    config = load_config(workspace_path(root, config_path))
    from ..data.preprocessing import read_frozen_manifest

    manifest = workspace_path(root, config["paths"]["manifest"])
    _, metadata = read_frozen_manifest(manifest, manifest.with_name("split-metadata.json"))
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("Selection requires the approved real manifest hash")
    selected, seen = [], set()
    for value in records:
        path = workspace_path(root, value)
        payload = json.loads(path.read_bytes())
        pair = payload.get("arm"), payload.get("seed")
        if (
            payload.get("schema_version") != 1
            or payload.get("stage") != "classification"
            or pair[0] not in ARMS
            or type(pair[1]) is not int
            or pair[1] not in config["classification"]["seeds"]
            or pair in seen
            or payload.get("resolved_config") != config
            or payload.get("split_manifest_sha256") != manifest_hash
            or payload.get("status") not in ("completed", "failed", "interrupted")
        ):
            raise ValueError(f"Duplicate or incompatible selected run record: {value}")
        seen.add(pair)
        selected.append(
            {"arm": pair[0], "seed": pair[1], "path": value, "sha256": sha256_file(path)}
        )
    result = {"schema_version": 1, "manifest_sha256": manifest_hash, "runs": selected}
    path = workspace_path(root, output)
    write_json(path, result)
    return {"path": output, "sha256": sha256_file(path), "selected_pairs": len(selected)}


def check_quota(root: Path, config_path: str, review: str):
    """Check accepted counts against the configured minimum, retaining target shortfalls."""
    config = load_config(workspace_path(root, config_path))
    directory = workspace_path(root, review)
    rows = [json.loads(line) for line in (directory / "approved.jsonl").read_bytes().splitlines()]
    counts = Counter(row["class_name"] for row in rows if row["quality_status"] == "accepted")
    minimum = config["generation"]["minimum_accepted"]
    if (
        len(rows) != sum(counts.values())
        or set(counts) != set(config["classes"])
        or min(counts.values()) < (minimum + 5) // 6
    ):
        raise ValueError("Accepted pool misses the approved balanced minimum; review the protocol")
    return {"accepted_counts": dict(counts), "minimum_accepted": minimum}


def check_pilot(root: Path, record_path: str, config_path: str, manifest_hash: str, receipt: str):
    """Check saved pilot evidence before requesting a full adaptation run."""
    record = json.loads(workspace_path(root, record_path).read_bytes())
    config = load_config(workspace_path(root, config_path))
    pilot_config = validate_config(record["resolved_config"])
    environment = record["environment"]["details"]
    history = record["metrics"]["updates"]
    if (
        record.get("schema_version") != 1
        or record.get("stage") != "diffusion"
        or record.get("status") != "completed"
        or pilot_config["status"] != "approved"
        or record.get("split_manifest_sha256") != manifest_hash
        or record.get("base_model_id") != config["diffusion"]["model_id"]
        or record.get("base_revision") != config["diffusion"]["revision"]
        or not str(environment.get("device", "")).startswith("cuda")
        or not environment.get("gpu")
        or environment.get("base_files") != json.loads(workspace_path(root, receipt).read_bytes())
        or not history
        or any(not math.isfinite(float(item["loss"])) for item in history)
        or not record.get("adapter_sha256")
    ):
        raise ValueError(
            "Pilot must be completed CUDA training with matching data/base and finite loss"
        )
    budget = pilot_config["diffusion"]["max_steps"]
    if budget is not None and history[-1]["update"] != budget:
        raise ValueError("Pilot did not complete its reviewed update budget")
    for artifact in record["artifacts"]:
        if sha256_file(workspace_path(root, artifact["path"])) != artifact["sha256"]:
            raise ValueError("Pilot artifact changed after its run record was written")
    return {"status": "verified_pilot_artifacts", "record": record_path}


def backup_workspace(root: Path, destination: Path, include_assets: bool = False):
    """Copy code and artifacts to a fresh directory and verify every copied file."""
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("Backup must be outside and separate from the workspace")
    if destination.exists():
        raise ValueError("Backup destination already exists; choose a fresh directory")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    paths = {root / value for value in tracked if value}
    folders = ["artifacts"] + (["data", "models"] if include_assets else [])
    for name in folders:
        folder = root / name
        if _linked(folder):
            raise ValueError("Backup sources must not be links")
        if folder.exists():
            for path in folder.rglob("*"):
                if _linked(path):
                    raise ValueError(f"Backup source contains a link: {path.relative_to(root)}")
                if path.is_file():
                    paths.add(path)
    for path in paths:
        if _linked(path) or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Backup source is missing or linked outside the workspace")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
    destination.mkdir(parents=True, exist_ok=False)
    index = []
    for source in sorted(paths):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256_file(source)
        shutil.copy2(source, target)
        if sha256_file(target) != before or sha256_file(source) != before:
            raise ValueError(f"Source changed or backup copy failed: {relative}")
        index.append({"path": relative.as_posix(), "sha256": before})
    report = {
        "schema_version": 1,
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "created_at": datetime.now(UTC).isoformat(),
        "include_assets": include_assets,
        "files": index,
    }
    # A partial backup has no completion index and must never be reported as complete.
    write_json(destination / "backup-index.json", report)
    return {"destination": str(destination), "files_verified": len(index), "git_commit": commit}


def _linked(path: Path) -> bool:
    """Include Windows junctions and other reparse points in the link guard."""
    if not path.exists() and not path.is_symlink():
        return False
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
