"""Audit a local NEU-CLS directory without changing its source files."""

import argparse
import json
from pathlib import Path

from industrial_defect_diffusion.data.validation import format_summary, validate_dataset


def main() -> int:
    """Write a new JSON report and print a short summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, help="JSON array of expected relative paths")
    parser.add_argument("--near-threshold", type=int, default=5)
    parser.add_argument("--json-errors", action="store_true", help="Write input errors as JSON")
    args = parser.parse_args()
    try:
        if args.report.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("Report must be outside the source directory")
        inventory = None
        if args.inventory:
            inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
            if not isinstance(inventory, list) or not all(isinstance(p, str) for p in inventory):
                raise ValueError("Inventory must be a JSON array of relative path strings")
        report = validate_dataset(
            args.root, expected_paths=inventory, near_threshold=args.near_threshold
        )
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        if args.json_errors:
            print(json.dumps({"status": "error", "message": str(exc)}))
            return 2
        parser.exit(2, f"Audit could not complete: {exc}\n")
    print(format_summary(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
