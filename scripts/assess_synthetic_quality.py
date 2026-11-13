"""Score reviewed synthetic images against real training data only."""

import argparse
import json
import subprocess
from pathlib import Path
from pickle import UnpicklingError

from PIL import Image

from industrial_defect_diffusion.evaluation.inception import local_extractor
from industrial_defect_diffusion.evaluation.quality import assess_quality


def main() -> int:
    """Write CPU quality artifacts or a structured error."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generation", "real-root", "manifest", "real-audit", "review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--approved-weights-sha256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class-size", type=int, default=40)
    parser.add_argument("--overall-size", type=int, default=200)
    parser.add_argument("--subsets", type=int, default=100)
    args = parser.parse_args()
    try:
        root = Path.cwd().resolve()
        output = args.output.resolve()
        if (
            not output.is_relative_to(root)
            or subprocess.run(
                [
                    "git",
                    "check-ignore",
                    "-q",
                    "--",
                    (output / "quality.json").relative_to(root).as_posix(),
                ],
                check=False,
            ).returncode
        ):
            raise ValueError("Use an ignored output directory inside the workspace")
        extractor, metadata = local_extractor(args.weights, args.approved_weights_sha256)
        report = assess_quality(
            args.generation,
            args.real_root,
            args.manifest,
            args.real_audit,
            args.review,
            args.approved_manifest_sha256,
            output,
            extractor=extractor,
            extractor_metadata=metadata,
            seed=args.seed,
            per_class_size=args.per_class_size,
            overall_size=args.overall_size,
            subsets=args.subsets,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        AttributeError,
        EOFError,
        UnpicklingError,
        Image.DecompressionBombError,
    ) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    print(json.dumps({"status": report["kid"]["overall"]["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
