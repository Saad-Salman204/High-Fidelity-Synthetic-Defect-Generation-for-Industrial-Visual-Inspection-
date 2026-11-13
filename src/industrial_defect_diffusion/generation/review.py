"""Apply explicit review decisions and build the accepted classifier handoff."""

import json
from collections import Counter
from datetime import datetime
from hashlib import sha256
from random import Random

from ..utils.experiment_config import CLASS_MAPPING


def json_bytes(value: object) -> bytes:
    """Serialize stable, portable review artifacts."""
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def review_template(audit: dict) -> dict:
    """Bind a blank primary review to the exact audit under review."""
    return {
        "schema_version": 1,
        "audit_sha256": sha256(json_bytes(audit)).hexdigest(),
        "decisions": [
            {
                "image_id": item["candidate"]["image_id"],
                "decision": "pending",
                "reviewer_id": None,
                "reviewed_at": None,
                "reason": None,
                "disputed": False,
            }
            for item in audit["findings"]
        ],
    }


def _decision(item: dict) -> None:
    reviewer = item.get("reviewer_id")
    timestamp = item.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Every decision needs a reviewer ID")
    if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp).utcoffset() is None:
        raise ValueError("Every decision needs a timezone-aware review timestamp")
    if item.get("decision") not in ("accepted", "rejected"):
        raise ValueError("Every candidate needs an explicit accept or reject decision")
    reason = item.get("reason")
    if item["decision"] == "rejected" and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("Every rejection needs a reason")


def review_candidates(audit: dict, primary: dict, secondary: dict | None = None) -> dict:
    """Validate decisions, select second-review samples, and finalize eligible rows."""
    digest = sha256(json_bytes(audit)).hexdigest()
    if primary.get("schema_version") != 1 or primary.get("audit_sha256") != digest:
        raise ValueError("Primary review refers to a different audit")
    findings = {item["candidate"]["image_id"]: item for item in audit["findings"]}
    decisions = {}
    for decision in primary["decisions"]:
        image_id = decision["image_id"]
        if image_id not in findings or image_id in decisions:
            raise ValueError("Unknown or duplicate primary review ID")
        _decision(decision)
        if type(decision.get("disputed")) is not bool:
            raise ValueError("Primary review must specify whether the label was disputed")
        decisions[image_id] = decision
    if decisions.keys() != findings.keys():
        raise ValueError("Primary review is missing decisions")
    ledger = []
    accepted = []
    hashes = set()
    disputes = set()
    for image_id, finding in findings.items():
        decision = decisions[image_id]
        if decision["disputed"]:
            disputes.add(image_id)
        row = {
            **finding["candidate"],
            "quality_status": decision["decision"],
            "reviewer_id": decision["reviewer_id"],
            "reviewed_at": decision["reviewed_at"],
            "rejection_reason": decision["reason"] if decision["decision"] == "rejected" else None,
        }
        ledger.append(row)
        if decision["decision"] == "accepted":
            if finding["blockers"]:
                raise ValueError(f"Cannot accept blocked candidate: {image_id}")
            if finding["warnings"] or finding["near_candidates"] or decision["disputed"]:
                reason = decision.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("Flagged or disputed acceptance requires a rationale")
            if row["sha256"] in hashes:
                raise ValueError("Cannot accept two exact duplicate candidates")
            hashes.add(row["sha256"])
            accepted.append(row)

    required = set(disputes)
    rng = Random(42)
    for label in CLASS_MAPPING:
        ids = sorted(row["image_id"] for row in accepted if row["class_name"] == label)
        required.update(rng.sample(ids, min(10, len(ids))))
    primary_digest = sha256(json_bytes(primary)).hexdigest()
    request = {
        "schema_version": 1,
        "audit_sha256": digest,
        "primary_sha256": primary_digest,
        "decisions": [
            {
                "image_id": key,
                "decision": "pending",
                "reviewer_id": None,
                "reviewed_at": None,
                "reason": None,
            }
            for key in sorted(required)
        ],
    }
    if secondary is None:
        return {"secondary_template": request}
    if any(
        secondary.get(key) != request[key]
        for key in (
            "schema_version",
            "audit_sha256",
            "primary_sha256",
        )
    ):
        raise ValueError("Second review refers to a different audit or primary review")
    seen = set()
    for decision in secondary["decisions"]:
        key = decision["image_id"]
        if key not in required or key in seen:
            raise ValueError("Unknown or duplicate second-review ID")
        _decision(decision)
        if (
            decision["reviewer_id"].strip().casefold()
            == decisions[key]["reviewer_id"].strip().casefold()
        ):
            raise ValueError("Second reviewer must be different from the primary reviewer")
        if decision["decision"] != decisions[key]["decision"]:
            raise ValueError("Review disagreement must be resolved before export")
        seen.add(key)
    if seen != required:
        raise ValueError("Second review is missing required decisions")
    counts = Counter(row["class_name"] for row in accepted)
    target, remainder = divmod(audit["target_accepted"], 6)
    payload = b"".join(json_bytes(row) for row in accepted)
    return {
        "accepted_payload": payload,
        "ledger": ledger,
        "report": {
            "schema_version": 1,
            "audit_sha256": digest,
            "primary_sha256": primary_digest,
            "secondary_sha256": sha256(json_bytes(secondary)).hexdigest(),
            "accepted_manifest_sha256": sha256(payload).hexdigest(),
            "accepted_counts": {label: counts[label] for label in CLASS_MAPPING},
            "shortfalls": {
                label: max(0, target + (idx < remainder) - counts[label])
                for label, idx in CLASS_MAPPING.items()
            },
            "rejections_by_reason": dict(
                Counter(
                    row["rejection_reason"] for row in ledger if row["quality_status"] == "rejected"
                )
            ),
            "second_review_count": len(seen),
            "has_all_six_classes": all(counts[label] for label in CLASS_MAPPING),
        },
    }
