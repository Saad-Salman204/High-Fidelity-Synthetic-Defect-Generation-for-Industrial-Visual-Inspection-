"""Audit candidates, request second review, or export accepted training rows."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from industrial_defect_diffusion.generation.quality import audit_candidates, write_contact_sheets
from industrial_defect_diffusion.generation.review import (
    json_bytes,
    review_candidates,
    review_template,
)


def main() -> int:
    """Use a new ignored directory for each workflow stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "second-review", "export"))
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-audit", type=Path, required=True)
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        root = Path.cwd().resolve()
        if not output.is_relative_to(root) or output.is_relative_to(args.generation.resolve()):
            raise ValueError(
                "Output must be inside the workspace and outside the generation directory"
            )
        for name in ("audit.json", "approved.jsonl", "crazing-001.png"):
            target = (output / name).relative_to(root).as_posix()
            if subprocess.run(["git", "check-ignore", "-q", "--", target], check=False).returncode:
                raise ValueError("Review output must be ignored by Git")
        if output.exists():
            raise ValueError("Review output already exists")
        audit = audit_candidates(
            args.generation,
            args.manifest,
            args.real_audit,
            args.approved_manifest_sha256,
            threshold=args.threshold,
        )
        if args.stage == "audit":
            output.mkdir(parents=True)
            write_contact_sheets(args.generation, audit, output)
            (output / "audit.json").write_bytes(json_bytes(audit))
            (output / "primary.json").write_bytes(json_bytes(review_template(audit)))
        else:
            if not args.primary or (args.stage == "export" and not args.secondary):
                raise ValueError(
                    "Supply primary decisions and, for export, second-review decisions"
                )
            primary = json.loads(args.primary.read_bytes())
            secondary = json.loads(args.secondary.read_bytes()) if args.stage == "export" else None
            result = review_candidates(audit, primary, secondary)
            output.mkdir(parents=True)
            if args.stage == "second-review":
                (output / "secondary.json").write_bytes(json_bytes(result["secondary_template"]))
            else:
                (output / "approved.jsonl").write_bytes(result["accepted_payload"])
                (output / "reviewed.jsonl").write_bytes(
                    b"".join(json_bytes(row) for row in result["ledger"])
                )
                (output / "audit.json").write_bytes(json_bytes(audit))
                (output / "review-summary.json").write_bytes(json_bytes(result["report"]))
                (output / "primary.json").write_bytes(json_bytes(primary))
                (output / "secondary.json").write_bytes(json_bytes(secondary))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        Image.DecompressionBombError,
    ) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    print(json.dumps({"status": "completed", "stage": args.stage, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
