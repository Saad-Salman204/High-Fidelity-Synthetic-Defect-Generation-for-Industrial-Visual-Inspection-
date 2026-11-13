"""Generate review-pending defect candidates from an approved local SD v1.5 adapter."""

import argparse
import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors import SafetensorError

from industrial_defect_diffusion.data.preprocessing import read_frozen_manifest
from industrial_defect_diffusion.diffusion.training import training_device
from industrial_defect_diffusion.generation.candidates import candidate_plan, generate_candidates
from industrial_defect_diffusion.generation.sampling import make_sampler, verify_checkpoint
from industrial_defect_diffusion.utils.experiment_config import load_config, relative_path


def run(args: argparse.Namespace, root: Path) -> Path:
    """Validate local inputs, record runtime identity and generate pending candidates."""
    config = load_config(args.config)
    if config["status"] != "approved":
        raise ValueError("Generation requires reviewed settings with status: approved")
    paths = {key: (root / value).resolve() for key, value in config["paths"].items()}
    output = (root / relative_path(args.output, "generation output")).resolve()
    model, adapter = args.model_dir.resolve(), args.adapter_dir.resolve()
    if (
        not output.is_relative_to(root)
        or not output.is_relative_to(paths["artifact_root"])
        or output == paths["artifact_root"]
        or any(output.is_relative_to(path) for path in (paths["data_root"], model, adapter))
    ):
        raise ValueError(
            "Output must be below the workspace artifact root and outside source inputs"
        )
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "--",
            (output / "candidates.jsonl").relative_to(root).as_posix(),
        ],
        cwd=root,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError("Generation output must be ignored by Git")
    for directory in (model, adapter):
        if (
            directory.is_relative_to(root)
            and subprocess.run(
                [
                    "git",
                    "check-ignore",
                    "-q",
                    "--",
                    (directory / "checkpoint.json").relative_to(root).as_posix(),
                ],
                cwd=root,
                check=False,
            ).returncode
            != 0
        ):
            raise ValueError("Keep local weights outside the repository or in an ignored directory")

    manifest = paths["manifest"]
    if not manifest.is_relative_to(root):
        raise ValueError("Frozen manifest must resolve inside the workspace")
    _, metadata = read_frozen_manifest(manifest, manifest.with_name("split-metadata.json"))
    if args.approved_manifest_sha256 != metadata["manifest_sha256"]:
        raise ValueError("Frozen manifest SHA-256 does not match peer approval")
    source = verify_checkpoint(model, adapter, config, metadata, args.approved_adapter_sha256)
    settings = config["generation"]
    precision = settings.get("precision", "fp32")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = training_device(settings["device"], precision)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "gpu_capability": list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "diffusers",
                "transformers",
                "peft",
                "safetensors",
                "Pillow",
            )
        },
    }
    plan = candidate_plan(config, source, runtime)
    sample_batch = None

    def sample(rows):
        nonlocal sample_batch
        if sample_batch is None:
            sample_batch = make_sampler(model, adapter, source, device, precision)
        return sample_batch(rows)

    return generate_candidates(output, plan, sample, resume=args.resume)


def main() -> int:
    """Generate candidates and print the JSONL location or a structured error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--approved-adapter-sha256", required=True, help="SHA-256 of checkpoint.json"
    )
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        manifest = run(args, Path.cwd().resolve())
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        AttributeError,
        SafetensorError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    print(json.dumps({"status": "pending_review", "manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
