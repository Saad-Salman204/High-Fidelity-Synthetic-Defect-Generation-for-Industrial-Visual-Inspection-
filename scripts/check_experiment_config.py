"""Validate experiment settings and print a machine-readable result."""

import argparse
import json

from industrial_defect_diffusion.utils.experiment_config import load_config


def main() -> int:
    """Print the validated config or a JSON error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    print(json.dumps({"status": "valid", "config": config}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
