"""Audit generated candidates against their plan and frozen real fingerprints."""

import io
import json
import re
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw

from ..data.preprocessing import read_frozen_manifest
from ..data.splits import audit_digest
from ..data.validation import difference_hash
from ..utils.experiment_config import CLASS_MAPPING, relative_path
from .candidates import candidate_plan
from .sampling import MODEL_ID, REVISION

MAX_BYTES = 32 * 1024 * 1024


def _read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected nonempty JSONL object records")
    return rows


def _real_index(manifest: Path, audit_path: Path, approved_hash: str) -> tuple[list[dict], dict]:
    rows, metadata = read_frozen_manifest(manifest, manifest.with_name("split-metadata.json"))
    if metadata["manifest_sha256"] != approved_hash:
        raise ValueError("Real manifest does not match the approved SHA-256")
    audit = json.loads(audit_path.read_bytes())
    if (
        audit_digest(audit) != metadata["audit_sha256"]
        or audit["perceptual_hash"]["algorithm"] != "horizontal_dhash_64"
        or audit["class_mapping"] != CLASS_MAPPING
    ):
        raise ValueError("Real audit does not match the frozen split and hash algorithm")
    indexed = {row["relative_path"]: row for row in audit["files"]}
    for row in rows:
        original = indexed.get(row["relative_path"], {})
        digest = original.get("dhash")
        if (
            original.get("sha256") != row["sha256"]
            or original.get("class_id") != row["class_id"]
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{16}", digest)
        ):
            raise ValueError("A frozen real row lacks a verified matching perceptual fingerprint")
        row["dhash"] = int(digest, 16)
    return rows, metadata


def _expected(root: Path, metadata: dict) -> tuple[dict, dict]:
    plan = json.loads((root / "plan.json").read_bytes())
    source = plan["source"]
    if (
        source["real_manifest_sha256"] != metadata["manifest_sha256"]
        or source["dataset_source_sha256"] != metadata["dataset_source_sha256"]
        or (source["base_model_id"], source["base_revision"]) != (MODEL_ID, REVISION)
        or not isinstance(source["adapter_checkpoint"], str)
        or not re.fullmatch(r"step-[0-9]{6,}", source["adapter_checkpoint"])
        or int(source["adapter_checkpoint"].split("-")[1]) < 1
    ):
        raise ValueError("Generation source provenance does not match the frozen dataset or model")
    for key in ("adapter_sha256", "adapter_checkpoint_sha256"):
        if not isinstance(source[key], str) or not re.fullmatch(r"[0-9a-f]{64}", source[key]):
            raise ValueError(f"Invalid generation source {key}")
    rebuilt = candidate_plan(plan["config"], source, plan["runtime"])
    if rebuilt != plan:
        raise ValueError("Generation plan is inconsistent with the class catalog or configuration")
    batch = plan["config"]["generation"]["batch_size"]
    expected = {
        row["image_id"]: {
            **row,
            "relative_path": f"batch-{index // batch:06d}/{row['image_id']}.png",
        }
        for index, row in enumerate(plan["rows"])
    }
    return plan, expected


def _image(root: Path, row: dict) -> tuple[Image.Image, str]:
    path = (root / relative_path(row["relative_path"], "candidate path")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Candidate path escapes generation directory")
    with path.open("rb") as stream:
        payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Candidate image exceeds size limit")
    digest = sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload), formats=["PNG"]) as image:
        if (
            image.mode != "RGB"
            or image.size != (row["width"], row["height"])
            or (getattr(image, "n_frames", 1) != 1)
        ):
            raise ValueError("Expected one RGB PNG frame with the planned dimensions")
        image.load()
        return image.copy(), digest


def _compare(
    findings: list[dict], hashes: dict[str, int], real: list[dict], threshold: int
) -> None:
    real_bytes = {row["sha256"] for row in real}
    real_ids = {row["image_id"] for row in real}
    by_bytes = defaultdict(list)
    for item in findings:
        row = item["candidate"]
        if row["image_id"] in real_ids or item.get("observed_sha256") in real_bytes:
            item["blockers"].append("real_exact_overlap")
        if item.get("observed_sha256"):
            by_bytes[item["observed_sha256"]].append(item)
        if row["image_id"] not in hashes:
            continue
        value = hashes[row["image_id"]]
        nearest = min(
            ((value ^ entry["dhash"]).bit_count(), entry["image_id"])
            for entry in real
            if entry["split"] == "train"
        )
        item["nearest_train"] = {"distance": nearest[0], "image_id": nearest[1]}
        if nearest[0] <= threshold:
            item["warnings"].append("similar_to_real_train")
        # Never expose held-out identities, paths, scores, or pixels.
        if any(
            entry["split"] != "train" and (value ^ entry["dhash"]).bit_count() <= threshold
            for entry in real
        ):
            item["blockers"].append("held_out_similarity")
    for group in by_bytes.values():
        if len(group) > 1:
            for item in group:
                item["warnings"].append("exact_candidate_duplicate")
    for left, right in combinations(findings, 2):
        a, b = (item["candidate"]["image_id"] for item in (left, right))
        if (
            a in hashes
            and b in hashes
            and left.get("observed_sha256") != right.get("observed_sha256")
            and (hashes[a] ^ hashes[b]).bit_count() <= threshold
        ):
            left["near_candidates"].append(b)
            right["near_candidates"].append(a)


def audit_candidates(
    root: Path,
    manifest: Path,
    audit_path: Path,
    approved_hash: str,
    *,
    threshold: int = 5,
) -> dict:
    """Audit a complete #11 run without opening any real images."""
    if type(threshold) is not int or not 0 <= threshold <= 64:
        raise ValueError("Near-duplicate threshold must be an integer from 0 to 64")
    real, metadata = _real_index(manifest, audit_path, approved_hash)
    plan, expected = _expected(root, metadata)
    rows = _read_jsonl(root / "candidates.jsonl")
    ids = [row.get("image_id") for row in rows]
    if any(not isinstance(key, str) for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate candidate IDs")
    if set(ids) != expected.keys():
        raise ValueError("Candidate ledger is incomplete or differs from the generation plan")
    by_id = dict(zip(ids, rows, strict=True))
    findings = []
    hashes = {}
    for image_id, original in expected.items():
        row = by_id[image_id]
        item = {
            "candidate": row,
            "blockers": [],
            "warnings": [],
            "near_candidates": [],
            "nearest_train": None,
        }
        findings.append(item)
        if row.keys() != original.keys() | {"sha256"} or any(
            type(row.get(key)) is not type(value) or row[key] != value
            for key, value in original.items()
        ):
            item["blockers"].append("invalid_generation_provenance")
            continue
        try:
            image, digest = _image(root, row)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            item["blockers"].append("invalid_image")
            item["image_error"] = str(exc)
            continue
        item["observed_sha256"] = digest
        if digest != row["sha256"]:
            item["blockers"].append("image_sha256_mismatch")
        hashes[image_id] = difference_hash(image.convert("L"))
    _compare(findings, hashes, real, threshold)
    return {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "manifest_sha256": metadata["manifest_sha256"],
        "real_audit_sha256": metadata["audit_sha256"],
        "threshold": threshold,
        "algorithm": "horizontal_dhash_64",
        "target_accepted": plan["config"]["generation"]["target_accepted"],
        "candidate_counts": dict(Counter(row["class_name"] for row in expected.values())),
        "findings": findings,
    }


def write_contact_sheets(root: Path, audit: dict, output: Path) -> None:
    """Create paginated class sheets with short identifiers and blocked labels."""
    by_class = defaultdict(list)
    for index, item in enumerate(audit["findings"], 1):
        row = item["candidate"]
        if isinstance(row.get("class_name"), str) and row["class_name"] in CLASS_MAPPING:
            by_class[row["class_name"]].append((index, item))
    for label, items in by_class.items():
        for offset in range(0, len(items), 24):
            page = items[offset : offset + 24]
            sheet = Image.new("RGB", (768, 38 + ((len(page) + 5) // 6) * 126), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), label, fill="black")
            for cell, (index, item) in enumerate(page):
                x, y = cell % 6 * 128, 38 + cell // 6 * 126
                if item.get("observed_sha256") and not item["blockers"]:
                    thumbnail, digest = _image(root, item["candidate"])
                    if digest != item["observed_sha256"]:
                        raise ValueError("Candidate changed after audit")
                    thumbnail.thumbnail((112, 100))
                    sheet.paste(thumbnail, (x + 8, y))
                state = (
                    "blocked"
                    if item["blockers"]
                    else ("flagged" if item["warnings"] or item["near_candidates"] else "pending")
                )
                draw.text((x + 8, y + 104), f"#{index} {state}", fill="black")
            sheet.save(output / f"{label}-{offset // 24 + 1:03d}.png")
