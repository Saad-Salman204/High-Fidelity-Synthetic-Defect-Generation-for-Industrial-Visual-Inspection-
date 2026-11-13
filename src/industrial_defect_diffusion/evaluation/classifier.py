"""Read frozen comparison inputs and evaluate ResNet-18 checkpoints."""

import io
import json
from hashlib import file_digest, sha256
from pathlib import Path
from statistics import mean, stdev

import torch

from ..classification.baseline import load_checkpoint, resolve_device
from ..data.preprocessing import NEUManifestDataset, make_loader, read_frozen_manifest
from ..utils.experiment_config import ARMS, relative_path, validate_config
from .metrics import SCORES, classification_metrics


def fingerprint(payload: bytes) -> str:
    """Compute a SHA-256 digest for an immutable input."""
    return sha256(payload).hexdigest()


def config_fingerprint(value: dict) -> str:
    """Match the canonical JSON encoding used by training records."""
    return fingerprint(
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )


def _path(root: Path, value: str) -> Path:
    path = (root / relative_path(value, "evaluation input path")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evaluation input path escapes the workspace")
    return path


def _verified_file(root: Path, reference: dict) -> Path:
    path = _path(root, reference["path"])
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != reference["sha256"]:
            raise ValueError(f"Artifact SHA-256 mismatch: {reference['path']}")
    return path


def _record(root: Path, config: dict, entry: dict, metadata: dict) -> dict:
    path = _path(root, entry["path"])
    payload = path.read_bytes()
    if fingerprint(payload) != entry["sha256"]:
        raise ValueError(f"Run record SHA-256 mismatch: {entry['path']}")
    record = json.loads(payload)
    if (
        record.get("schema_version"),
        record.get("stage"),
        record.get("arm"),
        record.get("seed"),
    ) != (
        1,
        "classification",
        entry["arm"],
        entry["seed"],
    ):
        raise ValueError("Frozen run has an incompatible schema, stage, arm or seed")
    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        raise ValueError("Frozen run is missing its run ID")
    if (
        record.get("split_manifest_sha256") != metadata["manifest_sha256"]
        or record.get("dataset_manifest", {}).get("sha256") != metadata["manifest_sha256"]
        or record.get("dataset_archive_sha256") != metadata["dataset_archive_sha256"]
    ):
        raise ValueError("Run does not use the same frozen real split and archive")
    resolved = validate_config(record["resolved_config"])
    if record.get("class_mapping_sha256") != config_fingerprint(config["classes"]):
        raise ValueError("Run class mapping fingerprint does not match")
    if record.get("config_sha256") != config_fingerprint(resolved):
        raise ValueError("Run configuration fingerprint does not match")
    if resolved != config:
        raise ValueError("All arms must use the supplied frozen experiment configuration")
    return record


def _checkpoint(root: Path, record: dict, config: dict, validation_ids: list) -> tuple[Path, str]:
    if record["metrics"]["validation_image_ids"] != validation_ids:
        raise ValueError("Run did not use the frozen real validation IDs")
    if record["metrics"]["updates"] != config["classification"]["max_steps"]:
        raise ValueError("Run did not complete the shared update budget")
    if record.get("classifier_weights") != config["classification"]["weights"]:
        raise ValueError("Run initialization does not match the frozen configuration")
    model = record["model"]
    if model["id"] != "resnet18":
        raise ValueError("Evaluation requires a ResNet-18 checkpoint")
    references = [
        item
        for item in record["artifacts"]
        if Path(item["path"]).name == "best.pt" and item["sha256"] == model["revision"]
    ]
    if len(references) != 1:
        raise ValueError("Run must identify exactly one frozen best checkpoint")
    return _verified_file(root, references[0]), model["revision"]


@torch.inference_mode()
def predict_test(model, loader, device, run_id: str) -> list[dict]:
    """Predict once per test sample without gradients or training-time behavior."""
    if getattr(loader.dataset, "split", None) != "test":
        raise ValueError("Final evaluation requires the real test partition")
    model.eval()
    predictions = []
    for batch in loader:
        probabilities = model(batch["image"].to(device)).softmax(dim=1).cpu()
        if probabilities.shape != (len(batch["image_id"]), 6):
            raise ValueError("Checkpoint must produce six logits per image")
        for image_id, label, scores in zip(
            batch["image_id"],
            batch["label"].tolist(),
            probabilities.tolist(),
            strict=True,
        ):
            predictions.append(
                {
                    "run_id": run_id,
                    "image_id": image_id,
                    "true_class": label,
                    "predicted_class": max(range(6), key=scores.__getitem__),
                    **{f"p{i}": score for i, score in enumerate(scores)},
                }
            )
    return predictions


def _comparison(runs: list[dict]) -> dict:
    index = {(run["arm"], run["seed"]): run["metrics"] for run in runs}
    paired = sorted({seed for _, seed in index if all((arm, seed) in index for arm in ARMS)})
    aggregates = {}
    for arm in ARMS:
        aggregates[arm] = {}
        for score in SCORES:
            values = [index[arm, seed][score] for seed in paired]
            aggregates[arm][score] = {
                "mean": mean(values) if values else None,
                "sample_sd": stdev(values) if len(values) > 1 else None,
            }
    return {
        "paired_seeds": paired,
        "aggregates": aggregates,
        "paired_macro_f1_differences": [
            {
                "seed": seed,
                "arm": arm,
                "minus_real_only": index[arm, seed]["macro_f1"]
                - index["real_only", seed]["macro_f1"],
            }
            for seed in paired
            for arm in ARMS[1:]
        ],
    }


def evaluate(
    root: Path,
    config: dict,
    selection: dict,
    approved_manifest_sha256: str,
) -> tuple[dict, list[dict]]:
    """Validate the frozen inputs, score eligible runs and report missing arms."""
    validate_config(config)
    if config["status"] not in ("approved", "smoke"):
        raise ValueError("Evaluation requires approved settings or CPU smoke settings")
    if config["status"] == "smoke" and config["evaluation"]["device"] != "cpu":
        raise ValueError("Smoke evaluation requires CPU")
    if selection.get("schema_version") != 1 or not isinstance(selection.get("runs"), list):
        raise ValueError("Expected a version-1 frozen run index")
    manifest = _path(root, config["paths"]["manifest"])
    metadata_path = manifest.with_name("split-metadata.json")
    metadata_sha = fingerprint(metadata_path.read_bytes())
    rows, metadata = read_frozen_manifest(manifest, metadata_path)
    manifest_sha = metadata["manifest_sha256"]
    if manifest_sha != approved_manifest_sha256 or selection.get("manifest_sha256") != manifest_sha:
        raise ValueError("Evaluation requires the reviewed SHA-256 of the frozen real split")
    if metadata["class_mapping"] != config["classes"]:
        raise ValueError("Split metadata class mapping does not match")
    if (
        metadata["seed"] != config["split"]["seed"]
        or [metadata["ratios"][split] for split in ("train", "val", "test")]
        != config["split"]["ratios"]
    ):
        raise ValueError("Frozen split settings do not match the configuration")
    validation_ids = [row["image_id"] for row in rows if row["split"] == "val"]
    expected = {row["image_id"]: row["class_id"] for row in rows if row["split"] == "test"}
    entries = {}
    for entry in selection["runs"]:
        key = entry["arm"], entry["seed"]
        if (
            key in entries
            or key[0] not in ARMS
            or type(key[1]) is not int
            or (key[1] not in config["classification"]["seeds"])
        ):
            raise ValueError(f"Unexpected or duplicate frozen arm/seed: {key}")
        entries[key] = entry
    ready = []
    incomplete = []
    run_ids = set()
    synthetic_hashes = set()
    commits = set()
    initial_weights = set()
    for seed in config["classification"]["seeds"]:
        for arm in ARMS:
            entry = entries.get((arm, seed))
            if entry is None:
                incomplete.append({"arm": arm, "seed": seed, "status": "missing"})
                continue
            record = _record(root, config, entry, metadata)
            if record["run_id"] in run_ids:
                raise ValueError("Run ID is reused across frozen entries")
            run_ids.add(record["run_id"])
            if record["status"] in ("failed", "interrupted"):
                incomplete.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "status": record["status"],
                        "failures": record.get("failures", []),
                    }
                )
                continue
            if record["status"] != "completed":
                raise ValueError("Unsupported training run status")
            commits.add(record["git_commit"])
            weight_hash = record["environment"]["details"].get("initial_weights_sha256")
            if config["status"] == "approved" and not weight_hash:
                raise ValueError("Approved run lacks its initial weights fingerprint")
            initial_weights.add(weight_hash)
            source = record.get("synthetic_manifest")
            if arm == "real_only" and source is not None:
                raise ValueError("Real-only run unexpectedly references synthetic data")
            if arm != "real_only":
                if source is None:
                    raise ValueError("Synthetic training run lacks its source manifest")
                _verified_file(root, source)
                synthetic_hashes.add(source["sha256"])
            checkpoint, digest = _checkpoint(root, record, config, validation_ids)
            ready.append((arm, seed, entry, record, checkpoint, digest))
    if any(len(values) > 1 for values in (synthetic_hashes, commits, initial_weights)):
        raise ValueError(
            "Frozen runs disagree on synthetic pool, training commit or initial weights"
        )

    device = resolve_device(config["evaluation"]["device"])
    dataset = NEUManifestDataset(
        _path(root, config["paths"]["data_root"]),
        rows,
        split="test",
        target="classifier",
        resolution=config["preprocessing"]["classifier_resolution"],
    )
    loader = make_loader(
        dataset,
        batch_size=config["evaluation"]["batch_size"],
        seed=config["evaluation"]["seed"],
    )
    runs = []
    outputs = []
    for arm, seed, entry, record, checkpoint, digest in ready:
        payload = checkpoint.read_bytes()
        if fingerprint(payload) != digest:
            raise ValueError("Checkpoint changed after preflight")
        model = load_checkpoint(io.BytesIO(payload), device)
        del payload
        predictions = predict_test(model, loader, device, record["run_id"])
        del model
        metrics = classification_metrics(expected, predictions)
        provenance = {
            "arm": arm,
            "seed": seed,
            "run_id": record["run_id"],
            "checkpoint_sha256": digest,
            "training_record": entry,
            "training_commit": record["git_commit"],
            "metrics": metrics,
        }
        runs.append(provenance)
        outputs.append({**provenance, "predictions": predictions})
    if (
        fingerprint(manifest.read_bytes()) != manifest_sha
        or fingerprint(metadata_path.read_bytes()) != metadata_sha
    ):
        raise ValueError("Frozen split changed during evaluation")
    return {
        "schema_version": 1,
        "status": "incomplete" if incomplete else "completed",
        "manifest_sha256": manifest_sha,
        "split_metadata_sha256": metadata_sha,
        "dataset_archive_sha256": metadata["dataset_archive_sha256"],
        "selection_sha256": config_fingerprint(selection),
        "test_membership_sha256": config_fingerprint(expected),
        "test_count": len(expected),
        "device": str(device),
        "torch": str(torch.__version__),
        "config_sha256": config_fingerprint(config),
        "resolved_config": config,
        "runs": runs,
        "incomplete": incomplete,
        **_comparison(runs),
    }, outputs
