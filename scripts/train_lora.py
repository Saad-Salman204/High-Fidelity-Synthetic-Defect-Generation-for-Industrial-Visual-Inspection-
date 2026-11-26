"""Run the minimal SD v1 LoRA path on approved local inputs."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from industrial_defect_diffusion.data.preprocessing import NEUManifestDataset, read_frozen_manifest
from industrial_defect_diffusion.diffusion.checkpoints import resume_checkpoint, save_checkpoint
from industrial_defect_diffusion.diffusion.local_model import (
    file_sha256,
    load_local_components,
    snapshot_fingerprints,
)
from industrial_defect_diffusion.diffusion.training import (
    make_optimizer,
    prepare_adapter,
    train_adapter,
    training_device,
)
from industrial_defect_diffusion.utils.experiment_config import load_config
from industrial_defect_diffusion.utils.run_records import make_run_record, write_run_record

SUPPORTED_MODELS = {
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"),
    ("CompVis/stable-diffusion-v1-4", "133a221b8aa7292a167afc5127cb63fb5005638b"),
}


def run(config: dict, args: argparse.Namespace, root: Path) -> Path:
    """Verify local inputs and write a record for each training attempt."""
    if config["status"] != "approved":
        raise ValueError("Review the protocol and set status: approved for real LoRA training")
    settings = config["diffusion"]
    for key, default in {
        "optimizer": "AdamW",
        "checkpoint_interval": 250,
        "gradient_checkpointing": True,
    }.items():
        settings.setdefault(key, default)
    if (settings["model_id"], settings["revision"]) not in SUPPORTED_MODELS:
        raise ValueError("Use a pinned SD v1 model from the issue #6 decision")
    paths = {name: (root / value).resolve() for name, value in config["paths"].items()}
    if any(not path.is_relative_to(root) for name, path in paths.items() if name != "data_root"):
        raise ValueError("Configured paths must resolve inside the workspace")
    configured_data_root = paths["data_root"]
    data_root = (
        args.dataset_root.resolve() if args.dataset_root is not None else configured_data_root
    )
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    if args.dataset_root is None and not data_root.is_relative_to(root):
        raise ValueError("Configured data_root must resolve inside the workspace")
    model_root = args.model_dir.resolve()
    if paths["artifact_root"].is_relative_to(data_root) or (
        paths["artifact_root"].is_relative_to(model_root)
    ):
        raise ValueError("Artifacts must be outside the dataset and model directories")
    manifest = paths["manifest"]
    rows, meta = read_frozen_manifest(manifest, manifest.with_name("split-metadata.json"))
    if meta["manifest_sha256"] != args.approved_manifest_sha256:
        raise ValueError("Supply the peer-approved SHA-256 of the frozen manifest")
    if (
        meta["seed"] != config["split"]["seed"]
        or [meta["ratios"][name] for name in ("train", "val", "test")] != config["split"]["ratios"]
    ):
        raise ValueError("Frozen split seed and ratios must match the configuration")
    device = training_device(settings["device"], settings["precision"])
    files = snapshot_fingerprints(model_root, args.variant)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    environment = {
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "diffusers",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
            )
        },
    }
    provenance = {
        "config": config,
        "git_commit": commit,
        "manifest_sha256": meta["manifest_sha256"],
        "dataset_source_sha256": meta["dataset_source_sha256"],
        "base_files": files,
        "variant": args.variant,
        "dataset_root": str(data_root),
        "environment": environment,
    }
    if args.resume is not None:
        provenance_file = args.resume / "checkpoint.json"
        environment["resumed_from_checkpoint_sha256"] = file_sha256(provenance_file)
        provenance["environment"] = {
            key: value
            for key, value in environment.items()
            if key != "resumed_from_checkpoint_sha256"
        }

    prep = config["preprocessing"]
    dataset = NEUManifestDataset(
        data_root,
        rows,
        split="train",
        target="diffusion",
        resolution=prep["diffusion_resolution"],
        train_flip_probability=prep["train_flip_probability"],
    )
    started = datetime.now(UTC)
    seed = settings["seed"]
    name = f"{started:%Y%m%dT%H%M%SZ}_diffusion_{seed}_{commit[:8]}"
    output = paths["artifact_root"] / "diffusion" / name
    output.mkdir(parents=True, exist_ok=False)
    resolved = output / "resolved-config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    status, failures = "completed", ()
    history = []
    artifacts = [resolved]
    adapter = None

    def on_update(update, epoch, next_batch, loss, checkpoint):
        nonlocal adapter
        history.append({"update": update, "loss": loss})
        if checkpoint:
            path = output / f"step-{update:06d}"
            save_checkpoint(
                path, parts, optimizer, scaler, device, provenance, (update, epoch, next_batch)
            )
            adapter = path / "adapter.safetensors"
            artifacts.extend((adapter, path / "training-state.pt", path / "checkpoint.json"))

    try:
        torch.manual_seed(seed)
        parts = load_local_components(model_root, args.variant)
        prepare_adapter(parts, settings, device)
        optimizer = make_optimizer(parts, settings)
        scaler = torch.amp.GradScaler("cuda", enabled=settings["precision"] == "fp16")
        cursor = (0, 0, 0)
        if args.resume is not None:
            cursor = resume_checkpoint(args.resume, parts, optimizer, scaler, device, provenance)
        train_adapter(
            parts,
            dataset,
            settings,
            device,
            optimizer,
            scaler,
            update=cursor[0],
            epoch=cursor[1],
            next_batch=cursor[2],
            on_update=on_update,
        )
        if file_sha256(manifest) != meta["manifest_sha256"]:
            raise ValueError("Frozen manifest changed during training. Discard this run")
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        failures = (f"{type(exc).__name__}: {exc}",)
        raise
    finally:
        record = make_run_record(
            config,
            stage="diffusion",
            seed=seed,
            git_commit=commit,
            started_at=started,
            ended_at=datetime.now(UTC),
            root=root,
            status=status,
            failures=failures,
            metrics={"updates": history} if history else None,
            artifacts=tuple(path.relative_to(root).as_posix() for path in artifacts),
            environment={**environment, "base_files": files, "variant": args.variant},
            adapter_sha256=file_sha256(adapter) if adapter else None,
            dataset_source_sha256=meta["dataset_source_sha256"],
        )
        # Use the fingerprint of the manifest actually read at startup.
        record["dataset_manifest"]["sha256"] = meta["manifest_sha256"]
        record["split_manifest_sha256"] = meta["manifest_sha256"]
        write_run_record(output / "run.json", record)
    return output / "run.json"


def main() -> int:
    """Read the shared config and execute a local training attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True, help="Approved local snapshot")
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Explicit approved dataset root; manifest paths and per-file SHA-256 are still checked"
        ),
    )
    parser.add_argument("--variant", choices=("fp16",), help="Local safetensors variant")
    parser.add_argument("--resume", type=Path, help="Trusted local step checkpoint directory")
    args = parser.parse_args()
    try:
        path = run(load_config(args.config), args, Path.cwd().resolve())
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
    print(json.dumps({"status": "completed", "record": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
