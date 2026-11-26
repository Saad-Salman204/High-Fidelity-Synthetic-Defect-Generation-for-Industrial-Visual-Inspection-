"""Create a training pool from audit-eligible candidates without human approval claims."""

import argparse
import json
from pathlib import Path

from industrial_defect_diffusion.data.automated_selection import create_automated_selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = create_automated_selection(
            proposal_path=args.proposal,
            audit_path=args.audit,
            generation_manifest=args.generation_manifest,
            generation_root=args.generation_root,
            output=args.output,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "selection_manifest": (args.output / "selected.jsonl").as_posix(),
                "selection_record": (args.output / "selection.json").as_posix(),
                "selected_candidate_count": result["selected_candidate_count"],
                "selected_counts_by_class": result["selected_counts_by_class"],
                "excluded_integrity_failures": result["excluded_integrity_failures"],
                "selection_manifest_sha256": result["selection_manifest_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
