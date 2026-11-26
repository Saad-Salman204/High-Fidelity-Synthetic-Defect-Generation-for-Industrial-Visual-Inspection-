"""Tests for small presentation-only assets and their source provenance."""

import json
from hashlib import sha256

import pytest
from PIL import Image

from industrial_defect_diffusion.utils.experiment_config import CLASS_MAPPING
from presentation.backend import runtime as runtime_module
from presentation.backend.asset_packaging import (
    FIGURE_STEMS,
    RESEARCH_FILES,
    package_presentation_assets,
)


@pytest.fixture
def sources(tmp_path):
    evaluation = tmp_path / "evaluation"
    generation = tmp_path / "generation"
    selection_path = tmp_path / "contact-selection.json"
    public = tmp_path / "public"
    evaluation.mkdir()
    generation.mkdir()
    records = []
    sample_rows = []
    for class_id, class_name in enumerate(sorted(CLASS_MAPPING, key=CLASS_MAPPING.__getitem__)):
        for sample in range(2):
            image_id = f"image-{class_id}-{sample}"
            relative = f"batch-{class_id}-{sample}/{image_id}.png"
            image_path = generation / relative
            image_path.parent.mkdir(parents=True)
            Image.new("L", (20, 20), color=class_id * 20 + sample).save(image_path)
            digest = sha256(image_path.read_bytes()).hexdigest()
            records.append(
                {
                    "image_id": image_id,
                    "class_name": class_name,
                    "relative_path": relative,
                    "sha256": digest,
                }
            )
            sample_rows.append(records[-1].copy())
    (generation / "candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    selection_path.write_text(json.dumps({"candidates": sample_rows}), encoding="utf-8")
    (evaluation / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "runs": [{} for _ in range(9)],
                "test_count": 270,
                "manifest_sha256": "b" * 64,
                "synthetic_selection": {
                    "human_review_status": "not_manually_reviewed_or_human_approved"
                },
            }
        ),
        encoding="utf-8",
    )
    for name in RESEARCH_FILES:
        path = evaluation / name
        if name != "evaluation.json":
            path.write_text(f"fixture {name}\n", encoding="utf-8")
    for stem in FIGURE_STEMS:
        for suffix in ("png", "pdf"):
            path = evaluation / "figures" / f"{stem}.{suffix}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"fixture figure")
    return evaluation, generation, selection_path, public


def _package(sources):
    evaluation, generation, selection, public = sources
    return package_presentation_assets(
        evaluation.parent,
        evaluation,
        generation,
        selection,
        public,
        images_per_class=2,
    )


def test_packaging_is_balanced_and_records_pending_provenance(sources):
    result = _package(sources)
    assert result["demo_images"] == 12
    assert result["research_files"] == len(RESEARCH_FILES) + 2 * len(FIGURE_STEMS)
    assert set(result["per_class"].values()) == {2}
    _, _, _, public = sources
    assert (public / "research" / "comparative-analysis.json").is_file()
    selection = json.loads((public / "demo-gallery" / "selection.json").read_text())
    assert len(selection["candidates"]) == 12
    assert selection["human_review_status"] == "exploratory_samples_only_not_acceptance_decisions"
    counts = {label: 0 for label in CLASS_MAPPING}
    for row in selection["candidates"]:
        counts[row["class_name"]] += 1
        assert (
            row["relative_path"]
            == f"{row['class_name']}/sample-{counts[row['class_name']]:02d}.png"
        )
        assert len(row["relative_path"]) <= 40
        image_path = public / "demo-gallery" / row["relative_path"]
        assert image_path.is_file()
        assert sha256(image_path.read_bytes()).hexdigest() == row["sha256"]
    provenance = json.loads((public / "asset-provenance.json").read_text())
    assert provenance["synthetic_human_review_status"] == (
        "not_manually_reviewed_or_human_approved"
    )
    assert "predictions.csv" not in provenance["files"]


def test_runtime_resolves_compact_bundled_gallery_paths(sources, monkeypatch):
    _package(sources)
    _, _, _, public = sources
    gallery_root = public / "demo-gallery"
    monkeypatch.setattr(runtime_module, "PACKAGED_GALLERY_ROOT", gallery_root)
    selection = json.loads((gallery_root / "selection.json").read_text())
    row = {**selection["candidates"][0], "_bundled": True}

    image_path = runtime_module.ProjectRuntime._gallery_path(row)

    assert image_path == gallery_root / row["relative_path"]
    assert image_path.is_file()
    assert sha256(image_path.read_bytes()).hexdigest() == row["sha256"]


def test_packaging_is_idempotent_and_refuses_to_overwrite_different_assets(sources):
    _package(sources)
    _package(sources)
    _, _, _, public = sources
    target = public / "demo-gallery" / "selection.json"
    target.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _package(sources)


def test_packaging_rejects_source_image_hash_mismatch(sources):
    _, generation, _, _ = sources
    original = next(generation.glob("batch-0-0/*.png"))
    original.write_bytes(b"changed source image")
    with pytest.raises(ValueError, match="hash mismatch"):
        _package(sources)


def test_packaging_rejects_candidate_contact_selection_mismatch(sources):
    evaluation, generation, selection, public = sources
    value = json.loads(selection.read_text())
    value["candidates"][0]["sha256"] = "0" * 64
    selection.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatches generation manifest"):
        package_presentation_assets(
            evaluation.parent,
            evaluation,
            generation,
            selection,
            public,
            images_per_class=2,
        )
