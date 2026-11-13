"""Evaluate a reviewed synthetic pool against frozen real training images."""

import io
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor

from ..data.preprocessing import read_frozen_manifest
from ..generation.quality import audit_candidates
from ..generation.review import json_bytes, review_candidates
from ..utils.experiment_config import CLASS_MAPPING, relative_path
from .kid import Extractor, score_pools


def _image(root: Path, row: dict) -> Image.Image:
    path = (root / relative_path(row["relative_path"], "image path")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Image path escapes its source directory")
    with path.open("rb") as stream:
        payload = stream.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024 or sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"Image changed or exceeds size limit: {row['image_id']}")
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != (int(row["width"]), int(row["height"])):
            raise ValueError(f"Image dimensions differ from manifest: {row['image_id']}")
        return image.convert("RGB")


def _reviewed_inputs(
    generation: Path,
    manifest: Path,
    real_audit: Path,
    review: Path,
    approved_hash: str,
) -> tuple[dict, dict, dict]:
    saved = json.loads((review / "audit.json").read_bytes())
    fresh = audit_candidates(
        generation, manifest, real_audit, approved_hash, threshold=saved["threshold"]
    )
    if saved != fresh:
        raise ValueError("Candidate audit no longer matches the reviewed inputs")
    primary = json.loads((review / "primary.json").read_bytes())
    secondary = json.loads((review / "secondary.json").read_bytes())
    result = review_candidates(fresh, primary, secondary)
    if (
        (review / "approved.jsonl").read_bytes() != result["accepted_payload"]
        or (review / "reviewed.jsonl").read_bytes()
        != b"".join(json_bytes(row) for row in result["ledger"])
        or json.loads((review / "review-summary.json").read_bytes()) != result["report"]
    ):
        raise ValueError("Review export does not match the explicit review decisions")
    return fresh, result, {"primary": primary, "secondary": secondary}


def _samples(rows: list[dict], label: str, count: int, seed: int) -> list[dict]:
    pool = sorted(
        (row for row in rows if row.get("class_name") == label), key=lambda row: row["image_id"]
    )
    return Random(seed).sample(pool, min(count, len(pool)))


def _grids(
    roots: dict[str, Path],
    real: list[dict],
    accepted: list[dict],
    rejected: list[dict],
    nearest: dict[str, str],
    output: Path,
    seed: int,
) -> dict:
    selection = {}
    real_by_id = {row["image_id"]: row for row in real}
    for label in CLASS_MAPPING:
        groups = (
            ("real", "Real train", _samples(real, label, 10, seed)),
            ("synthetic", "Accepted synthetic", _samples(accepted, label, 10, seed)),
            ("synthetic", "Rejected synthetic", _samples(rejected, label, 10, seed)),
        )
        sheet = Image.new("RGB", (960, 657), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), label, fill="black")
        selection[label] = {}
        grid_rows = (
            *groups,
            *(
                (
                    "real",
                    f"Closest real train matches for {title.lower()} (same column)",
                    [real_by_id.get(nearest.get(row["image_id"])) for row in rows],
                )
                for _, title, rows in groups[1:]
            ),
        )
        for index, (source, title, rows) in enumerate(grid_rows):
            y = 25 + index * 126
            draw.text((8, y), f"{title} ({len(rows)} shown)", fill="black")
            selection[label][title] = [row["image_id"] if row else None for row in rows]
            for offset, row in enumerate(rows):
                if row is None:
                    continue
                tile = _image(roots[source], row)
                tile.thumbnail((88, 88))
                x = offset * 96 + 4
                sheet.paste(tile, (x, y + 19))
                draw.text((x, y + 108), str(offset + 1), fill="black")
        sheet.save(output / f"{label}.png")
    return selection


def assess_quality(
    generation: Path,
    real_root: Path,
    manifest: Path,
    real_audit: Path,
    review: Path,
    approved_hash: str,
    output: Path,
    *,
    extractor: Extractor | None,
    extractor_metadata: dict,
    seed: int = 42,
    per_class_size: int = 40,
    overall_size: int = 200,
    subsets: int = 100,
) -> dict:
    """Produce exploratory scores and grids without altering the acceptance ledger."""
    if any(
        output.resolve().is_relative_to(path.resolve()) for path in (generation, real_root, review)
    ):
        raise ValueError("Assessment output must be outside source and review directories")
    if output.exists():
        raise ValueError("Assessment output already exists")
    rows, details = read_frozen_manifest(manifest, manifest.with_name("split-metadata.json"))
    audit, reviewed, decisions = _reviewed_inputs(
        generation, manifest, real_audit, review, approved_hash
    )
    if details["manifest_sha256"] != audit["manifest_sha256"]:
        raise ValueError("Frozen manifest changed during input validation")
    real = [row for row in rows if row["split"] == "train"]
    accepted = [row for row in reviewed["ledger"] if row["quality_status"] == "accepted"]
    roots = {"real": real_root, "synthetic": generation}
    safe_ids = {item["candidate"]["image_id"] for item in audit["findings"] if not item["blockers"]}
    nearest = {
        item["candidate"]["image_id"]: item["nearest_train"]["image_id"]
        for item in audit["findings"]
        if item["nearest_train"] is not None
    }
    rejected = [
        row
        for row in reviewed["ledger"]
        if row["quality_status"] == "rejected" and row["image_id"] in safe_ids
    ]

    @torch.inference_mode()
    def features(source: str, row: dict) -> torch.Tensor:
        assert extractor is not None
        image = _image(roots[source], row).convert("L").convert("RGB")
        batch = pil_to_tensor(image).unsqueeze(0)
        result = extractor(batch)
        if result.ndim != 2 or result.shape[0] != 1 or result.shape[1] < 1:
            raise ValueError("Extractor must return a single nonempty feature vector")
        return result[0].detach().cpu()

    scores = score_pools(
        real,
        accepted,
        features,
        extractor_available=extractor is not None,
        seed=seed,
        per_class_size=per_class_size,
        overall_size=overall_size,
        subsets=subsets,
    )
    output.mkdir(parents=True)
    selected = _grids(roots, real, accepted, rejected, nearest, output, seed)
    report = {
        "schema_version": 1,
        "reference_split": "train",
        "synthetic_pool": "accepted_only",
        "run_id": audit["run_id"],
        "real_manifest_sha256": details["manifest_sha256"],
        "dataset_archive_sha256": details["dataset_archive_sha256"],
        "real_audit_sha256": audit["real_audit_sha256"],
        "review": reviewed["report"],
        "extractor": extractor_metadata,
        "preprocessing": "Pillow grayscale L replicated to RGB; uint8 CHW; no augmentations",
        "kernel": {"degree": 3, "gamma": "1 / feature dimension", "coefficient": 1},
        "sampling": "independent uniform subsets without replacement within each draw",
        "overall_weighting": "pooled observed class proportions",
        "grid_seed": seed,
        "grid_image_ids": selected,
        "reviewer_observations": [
            {
                "image_id": row["image_id"],
                "reviewer_id": row["reviewer_id"],
                "decision": row["decision"],
                "reviewed_at": row["reviewed_at"],
                "observation": row["reason"],
                "review_round": review_round,
            }
            for review_round, decision_file in decisions.items()
            for row in decision_file["decisions"]
            if row.get("reason")
        ],
        "blocked_examples_omitted_from_grids": sum(
            bool(item["blockers"]) for item in audit["findings"]
        ),
        "duplicate_findings": dict(
            Counter(warning for item in audit["findings"] for warning in item["warnings"])
        ),
        "near_candidate_pairs": sum(len(item["near_candidates"]) for item in audit["findings"])
        // 2,
        "kid": scores,
    }
    (output / "quality.json").write_bytes(json_bytes(report))
    lines = [
        "# Exploratory generation quality",
        "",
        "Reference: frozen real training images. Synthetic pool: accepted images only.",
        "",
        "| Pool | Real | Synthetic | Subset | Draws scored | Mean KID | Subset SD | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in scores.items():
        values = [
            f"{row[key]:.6g}" if row[key] is not None else "—" for key in ("mean", "subset_sd")
        ]
        status = ", ".join(row["reasons"]) or row["status"]
        lines.append(
            f"| {label} | {row['real_count']} | {row['synthetic_count']} | "
            f"{row['subset_size']} | {row['subsets_scored']} | {' | '.join(values)} | {status} |"
        )
    lines.extend(
        (
            "",
            f"Feature extractor: {extractor_metadata['name']}. Seed: {seed}.",
            "Subset SD is descriptive and is not a confidence interval.",
            "Unbiased estimates can be negative. Class proportions affect pooled scores.",
            "Inception features reflect natural images and may miss industrial defects.",
            "The training reference is in-sample. KID cannot establish class fidelity, "
            "generalization, or suitability for training.",
            "",
            "## Review observations",
            "",
        )
    )
    for reason, count in sorted(reviewed["report"]["rejections_by_reason"].items()):
        lines.append(f"- {count} rejected: {reason.replace(chr(10), ' ')}")
    if not reviewed["report"]["rejections_by_reason"]:
        lines.append("- No rejections recorded for this run.")
    lines.extend(("", "See the class grids and quality.json for individual reviewer observations."))
    (output / "quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
