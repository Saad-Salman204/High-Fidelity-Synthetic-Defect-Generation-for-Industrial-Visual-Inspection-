"""Evaluate frozen ResNet-18 runs on the shared real test split."""

import argparse
import json
import pickle
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torchvision

from industrial_defect_diffusion.evaluation.classifier import evaluate, fingerprint
from industrial_defect_diffusion.evaluation.reporting import write_results
from industrial_defect_diffusion.utils.experiment_config import load_config, relative_path


def main() -> int:
    """Verify the peer-reviewed index, run inference, and write separate results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--approved-selection-sha256", required=True)
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument(
        "--real-root", type=Path, help="External root for the frozen real dataset images"
    )
    parser.add_argument(
        "--output", required=True, help="New directory under configured artifact root"
    )
    args = parser.parse_args()
    try:
        root = Path.cwd().resolve()
        config = load_config(args.config)
        output = (root / relative_path(args.output, "output")).resolve()
        artifact_root = (root / config["paths"]["artifact_root"]).resolve()
        data_root = (root / config["paths"]["data_root"]).resolve()
        real_root = args.real_root.resolve() if args.real_root is not None else data_root
        if (
            not output.is_relative_to(root)
            or not output.is_relative_to(artifact_root)
            or output == artifact_root
            or output.is_relative_to(data_root)
            or output.is_relative_to(real_root)
        ):
            raise ValueError(
                "Output must be below the workspace artifact root and outside the data"
            )
        if output.exists():
            raise ValueError("Output directory already exists")
        selection = args.selection.read_bytes()
        selection_digest = fingerprint(selection)
        if selection_digest != args.approved_selection_sha256:
            raise ValueError("Frozen run index does not match the approved selection SHA-256")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        started = datetime.now(UTC)
        report, runs = evaluate(
            root,
            config,
            json.loads(selection),
            args.approved_manifest_sha256,
            real_root=real_root,
        )
        report.update(
            selection_file_sha256=selection_digest,
            git_commit=commit,
            started_at=started.isoformat(),
            ended_at=datetime.now(UTC).isoformat(),
            python=platform.python_version(),
            platform=platform.platform(),
            torchvision=torchvision.__version__,
        )
        write_results(output, report, runs)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        AttributeError,
        pickle.UnpicklingError,
        EOFError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    print(json.dumps({"status": report["status"], "output": args.output}))
    return 1 if report["incomplete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
