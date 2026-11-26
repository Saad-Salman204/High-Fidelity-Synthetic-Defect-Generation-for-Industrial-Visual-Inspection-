"""Bundle compact Phase 1 outputs and explicitly pending demo samples."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from presentation.backend.asset_packaging import package_presentation_assets  # noqa: E402


def main() -> int:
    """Verify and publish the fixed local presentation asset bundle."""
    result = package_presentation_assets(
        ROOT,
        ROOT / "artifacts" / "evaluation" / "phase1-final-20261126",
        ROOT / "artifacts" / "synthetic" / "neu-sd15-step-001000",
        ROOT / "artifacts" / "synthetic" / "review-contact-sheet-seed42" / "review-selection.json",
        ROOT / "presentation" / "web" / "public",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
