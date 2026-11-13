"""Train one or all classifier arms from approved local inputs."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from hashlib import file_digest, sha256
from pathlib import Path

import torch
import torchvision

from industrial_defect_diffusion.classification.arms import make_arm_loader
from industrial_defect_diffusion.classification.baseline import (
    build_model,
    fit_baseline,
    resolve_device,
    seed_training,
)
from industrial_defect_diffusion.data.preprocessing import (
    NEUManifestDataset,
    make_loader,
    read_frozen_manifest,
)
from industrial_defect_diffusion.data.synthetic import ApprovedSyntheticDataset
from industrial_defect_diffusion.utils.experiment_config import ARMS, load_config
from industrial_defect_diffusion.utils.run_records import make_run_record, write_run_record


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def train(config: dict, args: argparse.Namespace, root: Path, arm: str) -> dict:
    """Resolve one arm's inputs and record its independent training run."""
    settings = config["classification"]
    if config["status"] not in ("approved", "smoke"):
        raise ValueError("Use approved experiment settings or the synthetic smoke configuration")
    if args.seed not in settings["seeds"]:
        raise ValueError("Requested seed is absent from classification.seeds")
    if config["status"] == "smoke" and settings["weights"] != "none":
        raise ValueError("Smoke runs require randomly initialized weights")
    if config["status"] == "smoke" and settings["device"] != "cpu":
        raise ValueError("Smoke runs require an explicit CPU device")

    paths = {key: (root / value).resolve() for key, value in config["paths"].items()}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("Configured paths must resolve inside the workspace")
    if paths["artifact_root"].is_relative_to(paths["data_root"]):
        raise ValueError("Artifacts must be written outside the source data directory")
    manifest = paths["manifest"]
    rows, metadata = read_frozen_manifest(manifest, manifest.with_name("split-metadata.json"))
    if (
        config["status"] == "approved"
        and args.approved_manifest_sha256 != metadata["manifest_sha256"]
    ):
        raise ValueError("Supply --approved-manifest-sha256 after peer review of the frozen split")
    if (
        metadata["seed"] != config["split"]["seed"]
        or [metadata["ratios"][name] for name in ("train", "val", "test")]
        != config["split"]["ratios"]
    ):
        raise ValueError("Frozen split seed and ratios must match the configuration")

    device = resolve_device(settings["device"])
    commit = _git_commit(root)
    seed_training(args.seed)
    prep = config["preprocessing"]
    transform = {
        "target": "classifier",
        "resolution": prep["classifier_resolution"],
        "train_flip_probability": prep["train_flip_probability"],
    }
    validation = NEUManifestDataset(paths["data_root"], rows, split="val", **transform)
    validation_loader = make_loader(validation, batch_size=settings["batch_size"], seed=args.seed)
    real = (
        None
        if arm == "synthetic_only"
        else NEUManifestDataset(
            paths["data_root"],
            rows,
            split="train",
            **transform,
        )
    )
    synthetic = synthetic_reference = synthetic_digest = None
    if arm != "real_only":
        if args.synthetic_root is None or args.synthetic_manifest is None:
            raise ValueError("This arm requires --synthetic-root and --synthetic-manifest")
        synthetic_path = args.synthetic_manifest.resolve()
        synthetic_root = args.synthetic_root.resolve()
        if not synthetic_path.is_relative_to(root):
            raise ValueError("Synthetic manifest must be inside the workspace")
        if paths["artifact_root"].is_relative_to(synthetic_root):
            raise ValueError("Artifacts must be outside the synthetic image root")
        payload = synthetic_path.read_bytes()
        synthetic_reference = synthetic_path.relative_to(root).as_posix()
        synthetic_digest = sha256(payload).hexdigest()
        if (
            config["status"] == "approved"
            and args.approved_synthetic_manifest_sha256 != synthetic_digest
        ):
            raise ValueError("Supply --approved-synthetic-manifest-sha256 after peer review")
        synthetic = ApprovedSyntheticDataset(
            synthetic_root,
            payload,
            rows,
            metadata["manifest_sha256"],
            base_revision=config["diffusion"]["revision"],
            resolution=prep["classifier_resolution"],
            flip_probability=prep["train_flip_probability"],
        )
    if arm == "real_only" and settings["max_steps"] is None:
        train_loader = make_loader(
            real,
            batch_size=settings["batch_size"],
            seed=args.seed,
            balanced=True,
        )
    else:
        train_loader = make_arm_loader(arm, real, synthetic, settings, args.seed)

    started = datetime.now(UTC)
    run_name = f"{started:%Y%m%dT%H%M%SZ}_{arm}_{args.seed}_{commit[:8]}"
    output = paths["artifact_root"] / "classification" / run_name
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best.pt"
    environment = {
        "torch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    result = None
    status = "completed"
    failures = ()
    log = output / "training.jsonl"
    log.write_text(
        json.dumps({"event": "started", "arm": arm, "seed": args.seed}) + "\n", encoding="utf-8"
    )

    def log_update(entry: dict) -> None:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "validation", **entry}, allow_nan=False) + "\n")

    try:
        if args.local_weights is not None:
            environment["initial_weights_sha256"] = _digest(args.local_weights)
        model = build_model(settings["weights"], args.local_weights)
        result = fit_baseline(
            model,
            train_loader,
            validation_loader,
            settings,
            device,
            checkpoint,
            on_validation=log_update,
        )
        result["validation_image_ids"] = [row["image_id"] for row in validation.rows]
        if _digest(manifest) != metadata["manifest_sha256"] or (
            synthetic_reference and _digest(root / synthetic_reference) != synthetic_digest
        ):
            raise ValueError("A manifest changed during the run. Discard the run output")
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        failures = (f"{type(exc).__name__}: {exc}",)
        raise
    finally:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": status, "failures": failures}) + "\n")
        has_checkpoint = checkpoint.is_file() and checkpoint.stat().st_size > 0
        artifacts = (log, checkpoint) if has_checkpoint else (log,)
        record = make_run_record(
            config,
            stage="classification",
            arm=arm,
            seed=args.seed,
            git_commit=commit,
            started_at=started,
            ended_at=datetime.now(UTC),
            root=root,
            status=status,
            failures=failures,
            metrics=result,
            environment=environment,
            dataset_archive_sha256=metadata["dataset_archive_sha256"],
            classifier_revision=_digest(checkpoint) if has_checkpoint else None,
            synthetic_manifest=synthetic_reference,
            artifacts=tuple(path.relative_to(root).as_posix() for path in artifacts),
        )
        record["dataset_manifest"]["sha256"] = metadata["manifest_sha256"]
        record["split_manifest_sha256"] = metadata["manifest_sha256"]
        if synthetic_reference:
            record["synthetic_manifest"]["sha256"] = synthetic_digest
        write_run_record(output / "run.json", record)
    return {"status": status, "record": (output / "run.json").relative_to(root).as_posix()}


def main() -> int:
    """Run the selected arms for one configured seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--local-weights", type=Path, help="Approved local ImageNet state dict")
    parser.add_argument("--approved-manifest-sha256", help="Peer-reviewed split manifest SHA-256")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="real_only")
    parser.add_argument("--synthetic-root", type=Path)
    parser.add_argument("--synthetic-manifest", type=Path, help="Accepted-only JSONL file")
    parser.add_argument("--approved-synthetic-manifest-sha256", help="Peer-reviewed JSONL SHA-256")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        root = Path.cwd().resolve()
        if args.arm == "all":
            if config["classification"]["max_steps"] is None:
                raise ValueError("Three-arm comparison requires a shared max_steps budget")
            if args.synthetic_root is None or args.synthetic_manifest is None:
                raise ValueError("--arm all requires --synthetic-root and --synthetic-manifest")
            result = {
                "status": "completed",
                "runs": {arm: train(config, args, root, arm) for arm in ARMS},
            }
        else:
            result = train(config, args, root, args.arm)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
