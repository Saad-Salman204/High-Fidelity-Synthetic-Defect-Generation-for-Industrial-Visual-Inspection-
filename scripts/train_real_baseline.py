"""Train the real-only classifier from approved local inputs."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from hashlib import file_digest
from pathlib import Path

import torch
import torchvision

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
from industrial_defect_diffusion.utils.experiment_config import load_config
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


def train(config: dict, args: argparse.Namespace, root: Path) -> dict:
    """Resolve inputs, run the baseline, and record the outcome."""
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
    loaders = {}
    for split in ("train", "val"):
        dataset = NEUManifestDataset(
            paths["data_root"],
            rows,
            split=split,
            target="classifier",
            resolution=prep["classifier_resolution"],
            train_flip_probability=prep["train_flip_probability"],
        )
        loaders[split] = make_loader(
            dataset,
            batch_size=settings["batch_size"],
            seed=args.seed,
            balanced=split == "train",
        )

    started = datetime.now(UTC)
    run_name = f"{started:%Y%m%dT%H%M%SZ}_real_only_{args.seed}_{commit[:8]}"
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
    try:
        if args.local_weights is not None:
            environment["initial_weights_sha256"] = _digest(args.local_weights)
        model = build_model(settings["weights"], args.local_weights)
        result = fit_baseline(model, loaders["train"], loaders["val"], settings, device, checkpoint)
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        failures = (f"{type(exc).__name__}: {exc}",)
        raise
    finally:
        if _digest(manifest) != metadata["manifest_sha256"]:
            raise ValueError("Manifest changed during the run. Discard the run output")
        has_checkpoint = checkpoint.is_file() and checkpoint.stat().st_size > 0
        record = make_run_record(
            config,
            stage="classification",
            arm="real_only",
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
            artifacts=(checkpoint.relative_to(root).as_posix(),) if has_checkpoint else (),
        )
        write_run_record(output / "run.json", record)
    return {"status": status, "record": (output / "run.json").relative_to(root).as_posix()}


def main() -> int:
    """Run one configured seed and report the local run record path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--local-weights", type=Path, help="Approved local ImageNet state dict")
    parser.add_argument("--approved-manifest-sha256", help="Peer-reviewed split manifest SHA-256")
    args = parser.parse_args()
    try:
        result = train(load_config(args.config), args, Path.cwd().resolve())
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
