"""Create a separate training pool from audit-eligible synthetic candidates."""

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..utils.experiment_config import CLASS_MAPPING, relative_path

POLICY_ID = "audit-eligible-no-flags-v1"
SELECTION_STATUS = "automatically_selected_by_audit_policy"


def _read_jsonl(path: Path) -> tuple[list[dict], bytes]:
    payload = path.read_bytes()
    rows = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
    return rows, payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_image(source_root: Path, relative: str) -> Path:
    relative_path(relative, "candidate image path")
    path = (source_root / relative).resolve()
    if not path.is_relative_to(source_root):
        raise ValueError(f"Image path escapes generation root: {relative}")
    return path


def create_automated_selection(
    *,
    proposal_path: Path,
    audit_path: Path,
    generation_manifest: Path,
    generation_root: Path,
    output: Path,
) -> dict:
    """Materialize every no-blocker/no-warning candidate and record full provenance.

    Near-candidate relationships are retained as metadata and never affect eligibility.
    Integrity failures are listed explicitly in selection.json and are not silently dropped.
    """
    proposal_path = proposal_path.resolve()
    audit_path = audit_path.resolve()
    generation_manifest = generation_manifest.resolve()
    generation_root = generation_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Selection output already exists: {output}")
    audit_payload = audit_path.read_bytes()
    audit = json.loads(audit_payload)
    audit_digest = hashlib.sha256(audit_payload).hexdigest()
    source_rows, source_payload = _read_jsonl(generation_manifest)
    source_digest = hashlib.sha256(source_payload).hexdigest()
    source_by_id = {row["image_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Generation manifest contains duplicate image IDs")
    audit_by_id = {item["candidate"]["image_id"]: item for item in audit["findings"]}
    if len(audit_by_id) != len(audit["findings"]):
        raise ValueError("Audit contains duplicate candidate IDs")

    proposal_rows, proposal_payload = _read_jsonl(proposal_path)
    proposal_by_id = {}
    for item in proposal_rows:
        candidate = item.get("candidate")
        if (
            item.get("proposal_status") != "eligible_for_human_review_not_accepted"
            or not isinstance(candidate, dict)
            or candidate.get("image_id") in proposal_by_id
        ):
            raise ValueError("Proposal has invalid status, candidate, or duplicate ID")
        proposal_by_id[candidate["image_id"]] = item
        if item.get("audit_sha256") != audit.get("real_audit_sha256"):
            raise ValueError("Proposal does not refer to this real-data audit")

    eligible_ids = {
        image_id
        for image_id, item in audit_by_id.items()
        if not item.get("blockers") and not item.get("warnings")
    }
    if eligible_ids != set(proposal_by_id):
        raise ValueError("Eligibility proposal does not exactly match the no-flags audit policy")
    if set(source_by_id) != set(audit_by_id):
        raise ValueError("Audit candidate IDs do not match the generation manifest")

    output.mkdir(parents=True)
    selected, exclusions = [], []
    for image_id in sorted(
        eligible_ids, key=lambda value: (source_by_id[value]["class_id"], value)
    ):
        proposal = proposal_by_id[image_id]
        candidate = proposal["candidate"]
        source = source_by_id[image_id]
        finding = audit_by_id[image_id]
        if candidate != source or finding["candidate"] != source:
            raise ValueError(f"Candidate provenance differs across source files: {image_id}")
        if proposal.get("audit_sha256") != audit.get("real_audit_sha256"):
            raise ValueError(f"Proposal audit provenance mismatch: {image_id}")
        reason = None
        label = source.get("class_name")
        class_id = source.get("class_id")
        if label not in CLASS_MAPPING or CLASS_MAPPING[label] != class_id:
            reason = "class_id_name_mismatch"
        elif source.get("quality_status") != "pending" or source.get("reviewer_id") is not None:
            reason = "source_candidate_is_not_pending"
        elif finding.get("observed_sha256") != source.get("sha256"):
            reason = "audit_source_image_hash_disagreement"
        try:
            source_image = _safe_image(generation_root, source["relative_path"])
            if reason is None and not source_image.is_file():
                reason = "missing_image"
            if reason is None and _sha256(source_image) != source["sha256"]:
                reason = "sha256_mismatch"
            if reason is None:
                with Image.open(source_image) as image:
                    image.verify()
                with Image.open(source_image) as image:
                    if image.size != (source["width"], source["height"]):
                        reason = "dimension_mismatch"
                    elif getattr(image, "n_frames", 1) != 1:
                        reason = "multiple_frames"
                    elif image.format not in ("PNG", "JPEG", "BMP", "TIFF"):
                        reason = "unsupported_image_format"
        except (OSError, ValueError, UnidentifiedImageError):
            reason = reason or "image_decode_failure"
        except KeyError as exc:
            raise ValueError(f"Candidate missing required source field: {exc}") from exc

        if reason:
            exclusions.append({"image_id": image_id, "class_name": label, "reason": reason})
            continue
        destination_relative = f"{label}/{image_id}.png"
        destination = output / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_image, destination)
        if _sha256(destination) != source["sha256"]:
            raise ValueError(f"Copied image hash mismatch: {image_id}")
        selected_row = source.copy()
        selected_row.update(
            relative_path=destination_relative,
            source_relative_path=source["relative_path"],
            selection_status=SELECTION_STATUS,
            selection_policy_id=POLICY_ID,
            near_candidate_count=len(finding.get("near_candidates", [])),
        )
        selected.append(selected_row)

    counts = Counter(row["class_name"] for row in selected)
    if set(counts) != set(CLASS_MAPPING):
        raise ValueError("Integrity exclusions left one or more classes without training examples")
    selected_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected).encode()
    manifest_path = output / "selected.jsonl"
    manifest_path.write_bytes(selected_payload)
    report = {
        "schema_version": 1,
        "selection_id": POLICY_ID,
        "selection_method": "deterministic_full_eligible_set_sorted_by_class_id_then_image_id",
        "human_review_status": "not_manually_reviewed_or_human_approved",
        "eligibility_policy": {
            "blockers": "exclude if one or more",
            "warnings": "exclude if one or more",
            "near_candidates": (
                "retain when blockers and warnings are empty; no automatic rejection"
            ),
        },
        "source_audit_sha256": audit_digest,
        "source_audit_real_data_fingerprint": audit.get("real_audit_sha256"),
        "source_generation_manifest_sha256": source_digest,
        "proposal_sha256": hashlib.sha256(proposal_payload).hexdigest(),
        "selection_manifest_sha256": hashlib.sha256(selected_payload).hexdigest(),
        "eligible_candidate_count": len(eligible_ids),
        "selected_candidate_count": len(selected),
        "selected_counts_by_class": {label: counts.get(label, 0) for label in CLASS_MAPPING},
        "excluded_by_eligibility_policy": [
            {
                "image_id": image_id,
                "class_name": audit_by_id[image_id]["candidate"]["class_name"],
                "blockers": audit_by_id[image_id].get("blockers", []),
                "warnings": audit_by_id[image_id].get("warnings", []),
            }
            for image_id in sorted(set(audit_by_id) - eligible_ids)
        ],
        "excluded_integrity_failures": exclusions,
        "selected_images": [
            {
                "image_id": row["image_id"],
                "class_name": row["class_name"],
                "relative_path": row["relative_path"],
                "source_relative_path": row["source_relative_path"],
                "sha256": row["sha256"],
                "near_candidate_count": row["near_candidate_count"],
            }
            for row in selected
        ],
    }
    (output / "selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
