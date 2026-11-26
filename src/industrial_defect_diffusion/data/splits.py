"""Create deterministic NEU-CLS splits from a reviewed version-1 audit."""

import csv
import io
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

from .validation import CLASSES, _expected_paths, validate_dataset

SPLITS = ("train", "val", "test")
FIELDS = (
    "schema_version",
    "image_id",
    "relative_path",
    "sha256",
    "class_id",
    "class_name",
    "width",
    "height",
    "mode",
    "group_id",
    "split",
    "source_version",
)
REVIEW_ONLY = {"exact_duplicate", "near_duplicate_candidate"}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(value: Any) -> str:
    return sha256(_json_bytes(value)).hexdigest()


def audit_digest(audit: dict) -> str:
    """Fingerprint the audit independently of JSON formatting."""
    return _digest(audit)


def dataset_source_digest(root: Path, audit: dict) -> str:
    """Fingerprint sorted source paths and their verified file SHA-256 values."""
    expected = {
        row["relative_path"]: row["sha256"]
        for row in audit["files"]
        if row["sha256"] and Path(row["relative_path"]).name.lower() != "thumbs.db"
    }
    actual = {}
    for relative in sorted(expected):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Source file missing: {relative}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != expected[relative]:
            raise ValueError(f"Source file hash differs from audit: {relative}")
        actual[relative] = digest
    return _digest([[path, actual[path]] for path in sorted(actual)])


def _require_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _reviewed_groups(
    audit: dict, review: dict, *, allow_unresolved_dhash: bool = False
) -> tuple[list[list[dict]], dict, list[list[str]]]:
    if audit["schema_version"] != 1 or audit["class_mapping"] != CLASSES:
        raise ValueError("Expected a version-1 NEU-CLS audit with the canonical class mapping")
    if review.get("schema_version") != 1 or review.get("audit_sha256") != audit_digest(audit):
        raise ValueError("Review must reference this audit SHA-256 and use schema version 1")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Review must name a reviewer")
    paths = _expected_paths(row["relative_path"] for row in audit["files"])
    rows = {row["relative_path"]: row for row in audit["files"]}
    exclusions = review.get("exclusions", {})
    if not isinstance(exclusions, dict) or not exclusions.keys() <= paths:
        raise ValueError("Exclusions must map known image paths to reasons")
    if any(not isinstance(reason, str) or not reason.strip() for reason in exclusions.values()):
        raise ValueError("Every exclusion needs a reason")
    if not audit["count_matches"] or audit["missing_paths"] or audit["structure_findings"]:
        note = review.get("inventory_note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("Unresolved inventory findings require a reviewed inventory_note")

    parent = {path: path for path in paths}

    def representative(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def join(left: str, right: str) -> None:
        first, second = sorted((representative(left), representative(right)))
        parent[second] = first

    by_hash = {}
    for path, row in rows.items():
        digest = row["sha256"]
        if digest is not None:
            _require_hash(digest, path)
            join(path, by_hash.setdefault(digest, path))
        if path in exclusions:
            continue
        if row["disposition"] == "proposed_exclusion" or set(row["problems"]) - REVIEW_ONLY:
            raise ValueError(f"{path}: unresolved audit finding requires exclusion or a new audit")
        label = row["class_name"]
        if label not in CLASSES or row["class_id"] != CLASSES[label]:
            raise ValueError(f"{path}: missing or invalid class label")
        if digest is None or (row["mode"], row["width"], row["height"]) != ("L", 200, 200):
            raise ValueError(f"{path}: missing hash or invalid image metadata")

    candidates = {tuple(sorted(pair["paths"])) for pair in audit["near_duplicate_pairs"]}
    decisions = {}
    for item in review.get("near_duplicates", []):
        pair = tuple(sorted(item["paths"]))
        if pair not in candidates or pair in decisions:
            raise ValueError("Near-duplicate review contains an unknown or repeated pair")
        decision = item.get("decision")
        if decision not in ("same", "different"):
            raise ValueError("Near-duplicate decision must be same or different")
        decisions[pair] = decision
        if decision == "same":
            join(*pair)
    unresolved = candidates - decisions.keys()
    amendment = review.get("protocol_amendment")
    if unresolved and allow_unresolved_dhash:
        required = {"amendment_id", "rationale", "reviewer", "date", "acknowledgment"}
        if not isinstance(amendment, dict) or amendment.keys() != required:
            raise ValueError(
                "Unresolved dHash candidates require a complete protocol_amendment record"
            )
        if any(
            not isinstance(amendment[key], str) or not amendment[key].strip() for key in required
        ):
            raise ValueError("Protocol amendment fields must be non-empty strings")
        try:
            date.fromisoformat(amendment["date"])
        except ValueError as exc:
            raise ValueError("Protocol amendment date must use ISO format YYYY-MM-DD") from exc
        acknowledgment = amendment["acknowledgment"].lower()
        if not all(token in acknowledgment for token in ("unverified", "dhash", "ungrouped")):
            raise ValueError(
                "Acknowledgment must state unresolved dHash candidates remain unverified "
                "and ungrouped"
            )
    if unresolved and not allow_unresolved_dhash:
        left, right = sorted(unresolved)[0]
        raise ValueError(f"Unresolved near-duplicate pair: {left}, {right}")
    for group in review.get("source_groups", []):
        if not isinstance(group, list) or len(group) < 2 or not set(group) <= paths:
            raise ValueError("Each source group must list at least two known paths")
        for path in group[1:]:
            join(group[0], path)

    grouped = defaultdict(list)
    for path in sorted(paths - exclusions.keys()):
        grouped[representative(path)].append(rows[path])
    if any(len({row["class_name"] for row in group}) != 1 for group in grouped.values()):
        raise ValueError("A confirmed group has conflicting labels. Resolve them before splitting")
    return list(grouped.values()), exclusions, [list(pair) for pair in sorted(unresolved)]


def validate_manifest(rows: list[dict]) -> None:
    """Reject invalid assignments, leakage and missing class coverage."""
    locations: dict[tuple[str, str], str] = {}
    seen = set()
    coverage = set()
    for row in rows:
        split = row["split"]
        label = row["class_name"]
        if split not in SPLITS or label not in CLASSES or row["class_id"] != CLASSES[label]:
            raise ValueError("Invalid split or class mapping")
        for field in ("image_id", "relative_path"):
            key = (field, row[field])
            if key in seen:
                raise ValueError(f"Repeated {field}: {row[field]}")
            seen.add(key)
        for field in ("sha256", "group_id"):
            key = (field, row[field])
            if locations.setdefault(key, split) != split:
                raise ValueError(f"{field} crosses split boundaries: {row[field]}")
        coverage.add((label, split))
    if coverage != {(label, split) for label in CLASSES for split in SPLITS}:
        raise ValueError("Each split must contain all six classes")


def create_splits(
    audit: dict,
    review: dict,
    *,
    source_version: str,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    allow_unresolved_dhash: bool = False,
) -> tuple[list[dict], dict]:
    """Allocate reviewed image groups and return manifest rows and metadata."""
    _require_hash(source_version, "Dataset-source SHA-256")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer from 0 to 4294967295")
    if (
        len(ratios) != 3
        or any(type(r) not in (int, float) or not math.isfinite(r) or r <= 0 for r in ratios)
        or not math.isclose(sum(ratios), 1, abs_tol=1e-9)
    ):
        raise ValueError("Split ratios must be three positive numbers totaling 1")
    groups, exclusions, unresolved_pairs = _reviewed_groups(
        audit, review, allow_unresolved_dhash=allow_unresolved_dhash
    )
    manifest = []
    counts = {}
    for label in CLASSES:
        class_groups = [group for group in groups if group[0]["class_name"] == label]
        if len(class_groups) < 3:
            raise ValueError(f"{label}: need at least three independent groups")
        ordered = sorted(
            class_groups,
            key=lambda group: (-len(group), _digest([seed, group[0]["relative_path"]])),
        )
        total = sum(map(len, ordered))
        assigned = dict.fromkeys(SPLITS, 0)
        targets = dict(zip(SPLITS, (total * ratio for ratio in ratios), strict=True))
        for index, group in enumerate(ordered):
            empty = [split for split in SPLITS if not assigned[split]]
            choices = empty if len(ordered) - index == len(empty) else SPLITS
            split = max(choices, key=lambda choice: targets[choice] - assigned[choice])
            assigned[split] += len(group)
            group_id = _digest([item["relative_path"] for item in group])
            for item in group:
                row = {field: item[field] for field in FIELDS if field in item}
                row.update(
                    schema_version=1,
                    image_id=_digest([source_version, item["relative_path"], item["sha256"]]),
                    group_id=group_id,
                    split=split,
                    source_version=source_version,
                )
                manifest.append(row)
        counts[label] = assigned

    manifest.sort(key=lambda row: row["relative_path"])
    validate_manifest(manifest)
    return manifest, {
        "schema_version": 1,
        "algorithm": "largest-group-first-class-deficit-v1",
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios, strict=True)),
        "dataset_source_sha256": source_version,
        "audit_sha256": audit_digest(audit),
        "review_sha256": _digest(review),
        "class_mapping": CLASSES,
        "class_mapping_sha256": _digest(CLASSES),
        "counts_by_class": counts,
        "counts_by_split": dict(Counter(row["split"] for row in manifest)),
        "group_count": len(groups),
        "largest_group_size": max(map(len, groups)),
        "review": review,
        "unresolved_dhash_pairs": unresolved_pairs,
        "exclusions": [
            {
                "relative_path": row["relative_path"],
                "reason": exclusions[row["relative_path"]],
                "audit_problems": row["problems"],
                "sha256": row["sha256"],
            }
            for row in audit["files"]
            if row["relative_path"] in exclusions
        ],
        "test_freeze_approval": "pending_peer_review",
    }


def freeze_splits(
    root: Path,
    audit: dict,
    review: dict,
    output: Path,
    *,
    source_version: str | None = None,
    inventory: list[str] | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    allow_unresolved_dhash: bool = False,
) -> dict:
    """Recheck source inputs and write a new immutable CSV and metadata pair."""
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output directory must be outside the source dataset")
    current = validate_dataset(
        root,
        expected_paths=inventory,
        near_threshold=audit["perceptual_hash"]["max_hamming_distance"],
    )
    if current != audit:
        raise ValueError("Source audit changed. Re-run the audit and review the new report")
    source_version = dataset_source_digest(root, audit)
    rows, metadata = create_splits(
        audit,
        review,
        source_version=source_version,
        seed=seed,
        ratios=ratios,
        allow_unresolved_dhash=allow_unresolved_dhash,
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode()
    metadata["manifest_sha256"] = sha256(csv_bytes).hexdigest()
    payloads = {"manifest.csv": csv_bytes, "split-metadata.json": _json_bytes(metadata)}
    try:
        output.mkdir()
    except FileExistsError:
        if any(
            not (output / name).is_file() or (output / name).read_bytes() != content
            for name, content in payloads.items()
        ):
            raise ValueError(
                "Existing freeze is incomplete or differs. Use a new output directory"
            ) from None
        return metadata
    for name, content in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    return metadata
