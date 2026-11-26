"""Freeze or verify a reviewed NEU-CLS split."""

import argparse
import json
from pathlib import Path

from industrial_defect_diffusion.data.splits import audit_digest, freeze_splits


def main() -> int:
    """Read audit and review files and emit JSON status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-digest", action="store_true", help="Print the review fingerprint")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-unresolved-dhash",
        action="store_true",
        help=(
            "Apply a documented amendment; unresolved candidates remain unverified and ungrouped"
        ),
    )
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    args = parser.parse_args()
    try:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        if args.audit_digest:
            print(audit_digest(audit))
            return 0
        if not all((args.root, args.review, args.output)):
            raise ValueError("Freeze requires --root, --review and --output")
        review = json.loads(args.review.read_text(encoding="utf-8"))
        inventory = (
            json.loads(args.inventory.read_text(encoding="utf-8")) if args.inventory else None
        )
        if args.inventory and (
            not isinstance(inventory, list) or not all(isinstance(path, str) for path in inventory)
        ):
            raise ValueError("Inventory must be an array of relative paths")
        metadata = freeze_splits(
            args.root,
            audit,
            review,
            args.output,
            inventory=inventory,
            seed=args.seed,
            ratios=tuple(args.ratios),
            allow_unresolved_dhash=args.allow_unresolved_dhash,
        )
        print(json.dumps({"status": "verified", "manifest_sha256": metadata["manifest_sha256"]}))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
