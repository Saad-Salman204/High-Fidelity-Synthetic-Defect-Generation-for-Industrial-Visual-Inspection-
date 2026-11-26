"""Prepare a small, provenance-bound public presentation asset bundle."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from PIL import Image

from industrial_defect_diffusion.utils.experiment_config import CLASS_MAPPING

FIGURE_STEMS = (
    "01_experimental_design",
    "02_accuracy_by_seed",
    "03_macro_f1_by_seed",
    "04_precision_recall",
    "05_mean_sd_all_metrics",
    "06_confusion_real_only",
    "06_confusion_synthetic_only",
    "06_confusion_real_plus_synthetic",
    "09_per_class_f1",
    "10_pairwise_macro_f1_differences",
)
RESEARCH_FILES = (
    "evaluation.json",
    "comparative-analysis.json",
    "comparison.csv",
    "comparison.md",
    "per-class-aggregates.csv",
    "per-class-f1-differences.csv",
    "pairwise-seed-differences.csv",
    "research-report.md",
    "executive-summary.md",
    "jury-brief.md",
    "presentation-outline.md",
    "performance.svg",
)


def _sha256(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _publish_once(path: Path, payload: bytes) -> None:
    """Create a public asset without replacing a differing existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(
                f"Refusing to overwrite a changed presentation asset: {path}"
            ) from None


def package_presentation_assets(
    root: Path,
    evaluation_root: Path,
    generation_root: Path,
    review_selection_path: Path,
    public_root: Path,
    *,
    images_per_class: int = 2,
) -> dict:
    """Package sampled pending images and aggregate results, without predictions."""
    if type(images_per_class) is not int or not 1 <= images_per_class <= 12:
        raise ValueError("images_per_class must be an integer between 1 and 12")
    evaluation_path = evaluation_root / "evaluation.json"
    evaluation_bytes = evaluation_path.read_bytes()
    evaluation = json.loads(evaluation_bytes)
    if (
        evaluation.get("status") != "completed"
        or len(evaluation.get("runs", [])) != 9
        or evaluation.get("test_count") != 270
    ):
        raise ValueError("Only the completed, nine-run Phase 1 evaluation can be packaged")
    selection = json.loads(review_selection_path.read_text(encoding="utf-8"))
    generation_manifest = generation_root / "candidates.jsonl"
    candidate_bytes = generation_manifest.read_bytes()
    candidates = {
        item["image_id"]: item
        for item in (
            json.loads(line) for line in candidate_bytes.decode("utf-8").splitlines() if line
        )
    }
    selection_bytes = review_selection_path.read_bytes()
    selected_by_class = {name: [] for name in CLASS_MAPPING}
    for item in selection.get("candidates", []):
        source = candidates.get(item.get("image_id"))
        if not source or any(
            item.get(key) != source.get(key) for key in ("class_name", "relative_path", "sha256")
        ):
            raise ValueError(f"Contact-sheet selection mismatches generation manifest: {item}")
        label = item["class_name"]
        if label not in selected_by_class:
            raise ValueError(f"Unexpected class in contact-sheet selection: {label}")
        if len(selected_by_class[label]) < images_per_class:
            selected_by_class[label].append((item, source))
    if any(len(rows) != images_per_class for rows in selected_by_class.values()):
        raise ValueError("Contact-sheet selection does not provide the requested class samples")

    bundled_candidates = []
    for label in sorted(CLASS_MAPPING, key=CLASS_MAPPING.__getitem__):
        for sample_number, (item, source) in enumerate(selected_by_class[label], start=1):
            original = (generation_root / source["relative_path"]).resolve()
            if not original.is_relative_to(generation_root.resolve()) or not original.is_file():
                raise ValueError(f"Missing selected demonstration image: {item['image_id']}")
            payload = original.read_bytes()
            digest = _sha256(payload)
            if digest != source["sha256"]:
                raise ValueError(f"Selected demonstration image hash mismatch: {item['image_id']}")
            try:
                with Image.open(original) as image:
                    image.verify()
            except OSError as exc:
                raise ValueError(
                    f"Selected demonstration image is corrupt: {item['image_id']}"
                ) from exc
            relative = f"{label}/sample-{sample_number:02d}.png"
            _publish_once(public_root / "demo-gallery" / relative, payload)
            bundled_candidates.append(
                {
                    "image_id": item["image_id"],
                    "class_name": label,
                    "class_id": CLASS_MAPPING[label],
                    "relative_path": relative,
                    "sha256": digest,
                    "human_review_status": (
                        "sampled_for_contact_sheet_only_pending_candidate_review"
                    ),
                }
            )

    gallery_selection = {
        "schema_version": 1,
        "selection_method": (
            f"First {images_per_class} entries per class, in the seed-42 contact-sheet "
            "selection order; no accept/reject decisions were made."
        ),
        "human_review_status": "exploratory_samples_only_not_acceptance_decisions",
        "source_generation_manifest_sha256": _sha256(candidate_bytes),
        "source_review_selection_sha256": _sha256(selection_bytes),
        "candidates": bundled_candidates,
    }
    gallery_payload = (json.dumps(gallery_selection, indent=2, sort_keys=True) + "\n").encode()
    _publish_once(public_root / "demo-gallery" / "selection.json", gallery_payload)

    copied = {}
    for name in RESEARCH_FILES:
        payload = (evaluation_root / name).read_bytes()
        _publish_once(public_root / "research" / name, payload)
        copied[f"research/{name}"] = _sha256(payload)
    for stem in FIGURE_STEMS:
        for suffix in ("png", "pdf"):
            name = f"figures/{stem}.{suffix}"
            payload = (evaluation_root / name).read_bytes()
            _publish_once(public_root / "research" / name, payload)
            copied[f"research/{name}"] = _sha256(payload)

    package_manifest = {
        "schema_version": 1,
        "contents": "Aggregate Phase 1 results, figures, reports, and 12 pending review samples.",
        "source_evaluation_sha256": _sha256(evaluation_bytes),
        "source_generation_manifest_sha256": _sha256(candidate_bytes),
        "source_review_selection_sha256": _sha256(selection_bytes),
        "frozen_real_manifest_sha256": evaluation["manifest_sha256"],
        "synthetic_human_review_status": evaluation["synthetic_selection"]["human_review_status"],
        "demo_gallery_status": gallery_selection["human_review_status"],
        "files": copied,
        "gallery_selection_sha256": _sha256(gallery_payload),
    }
    manifest_payload = (json.dumps(package_manifest, indent=2, sort_keys=True) + "\n").encode()
    _publish_once(public_root / "asset-provenance.json", manifest_payload)
    return {
        "demo_images": len(bundled_candidates),
        "per_class": {label: len(rows) for label, rows in selected_by_class.items()},
        "research_files": len(copied),
        "asset_provenance": "asset-provenance.json",
        "asset_provenance_sha256": _sha256(manifest_payload),
    }
