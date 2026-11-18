"""Local configuration, verification, selection and backup operations for run_lab.ps1."""

import argparse
import json
import subprocess
from pathlib import Path

from industrial_defect_diffusion.utils.experiment_config import load_config
from industrial_defect_diffusion.utils.lab import (
    backup_workspace,
    check_pilot,
    check_quota,
    freeze_selection,
    prepare_config,
    workspace_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    config = sub.add_parser("config")
    config.add_argument("path")
    device = sub.add_parser("device")
    device.add_argument("--requested", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--kind",
        required=True,
        choices=("base", "classifier-pilot", "lora-pilot", "generation-pilot"),
    )
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--manifest", required=True)
    models = sub.add_parser("models")
    models.add_argument("--snapshot", required=True)
    models.add_argument("--variant", choices=("fp16",))
    models.add_argument("--approved-receipt")
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--csv", required=True)
    manifest.add_argument("--archive-sha256", required=True)
    select = sub.add_parser("selection")
    select.add_argument("--config", required=True)
    select.add_argument("--manifest-sha256", required=True)
    select.add_argument("--record", action="append", default=[])
    select.add_argument("--output", required=True)
    quota = sub.add_parser("quota")
    quota.add_argument("--config", required=True)
    quota.add_argument("--review", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--record", required=True)
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--manifest-sha256", required=True)
    pilot.add_argument("--receipt", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--destination", type=Path, required=True)
    backup.add_argument("--include-assets", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    try:
        if args.action == "config":
            result = load_config(workspace_path(root, args.path))
        elif args.action == "device":
            from industrial_defect_diffusion.classification.baseline import resolve_device

            result = {"device": str(resolve_device(args.requested))}
        elif args.action == "prepare":
            result = prepare_config(
                root, args.source, args.output, args.kind, args.data, args.manifest
            )
        elif args.action == "models":
            from industrial_defect_diffusion.diffusion.local_model import snapshot_fingerprints

            result = snapshot_fingerprints(workspace_path(root, args.snapshot), args.variant)
            if args.approved_receipt:
                approved = json.loads(workspace_path(root, args.approved_receipt).read_bytes())
                if approved != result:
                    raise ValueError(
                        "Base model files differ from the reviewed fingerprint receipt"
                    )
        elif args.action == "manifest":
            from industrial_defect_diffusion.data.preprocessing import read_frozen_manifest

            path = workspace_path(root, args.csv)
            _, result = read_frozen_manifest(path, path.with_name("split-metadata.json"))
            if result["dataset_archive_sha256"] != args.archive_sha256:
                raise ValueError("Frozen split does not reference the approved archive")
        elif args.action == "selection":
            result = freeze_selection(
                root, args.config, args.manifest_sha256, args.record, args.output
            )
        elif args.action == "quota":
            result = check_quota(root, args.config, args.review)
        elif args.action == "pilot":
            result = check_pilot(root, args.record, args.config, args.manifest_sha256, args.receipt)
        else:
            result = backup_workspace(root, args.destination, args.include_assets)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
