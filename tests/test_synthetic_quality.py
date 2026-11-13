"""CPU tests for KID math, reviewed inputs, and exploratory artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from test_synthetic_review import candidates as review_candidates_fixture
from test_synthetic_review import fill

from industrial_defect_diffusion.evaluation.inception import local_extractor
from industrial_defect_diffusion.evaluation.kid import kid_subset, score_pools
from industrial_defect_diffusion.evaluation.quality import assess_quality
from industrial_defect_diffusion.generation.quality import audit_candidates
from industrial_defect_diffusion.generation.review import (
    json_bytes,
    review_candidates,
    review_template,
)

ROOT = Path(__file__).resolve().parents[1]
candidates = review_candidates_fixture


@pytest.fixture
def reviewed_inputs(candidates, frozen_images, tmp_path, monkeypatch):
    generation, manifest, real_audit, digest, rows = candidates[:5]
    real_root = frozen_images[0]
    audit = audit_candidates(generation, manifest, real_audit, digest, threshold=0)
    primary = fill(review_template(audit), "reviewer-a")
    primary["decisions"][0].update(decision="rejected", reason="Unclear defect")
    secondary = fill(review_candidates(audit, primary)["secondary_template"], "reviewer-b")
    result = review_candidates(audit, primary, secondary)
    review = tmp_path / "review"
    review.mkdir()
    for filename, payload in {
        "audit.json": json_bytes(audit),
        "primary.json": json_bytes(primary),
        "secondary.json": json_bytes(secondary),
        "reviewed.jsonl": b"".join(json_bytes(row) for row in result["ledger"]),
        "approved.jsonl": result["accepted_payload"],
        "review-summary.json": json_bytes(result["report"]),
    }.items():
        (review / filename).write_bytes(payload)

    monkeypatch.undo()
    open_file = Path.open
    held_out = {
        (real_root / row["relative_path"]).resolve() for row in rows if row["split"] != "train"
    }

    def guard(path, *args, **kwargs):
        assert path.resolve() not in held_out, "Assessment must not read held-out pixels"
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    return generation, real_root, manifest, real_audit, review, digest


def test_unbiased_kid_matches_hand_calculation():
    assert kid_subset(torch.zeros((2, 1)), torch.ones((2, 1))) == 7.0
    mixed = torch.tensor([[0.0], [1.0]])
    assert kid_subset(mixed, mixed) == -3.5
    with pytest.raises(ValueError, match="at least two"):
        kid_subset(mixed[:1], mixed)
    with pytest.raises(ValueError, match="finite"):
        kid_subset(mixed, mixed * float("nan"))


def test_seeded_scores_aggregate_per_class_and_ignore_input_order():
    rows = [
        {"image_id": str(i), "class_name": "crazing" if i < 3 else "inclusion"} for i in range(6)
    ]
    calls = []

    def feature(source, row):
        calls.append((source, row["image_id"]))
        return torch.tensor([float(row["image_id"]) + (source == "synthetic")])

    def score(pool, seed):
        return score_pools(
            pool,
            pool,
            feature,
            extractor_available=True,
            overall_size=3,
            per_class_size=2,
            subsets=4,
            seed=seed,
        )

    first = score(rows, 42)
    assert len(calls) == len(set(calls))
    assert first == score(rows[::-1], 42)
    assert first["overall"]["mean"] != score(rows, 43)["overall"]["mean"]
    assert first["crazing"]["subsets_scored"] == first["inclusion"]["subsets_scored"] == 4
    assert first["scratches"]["reasons"] == ["insufficient_samples"]
    with pytest.raises(ValueError, match="overall_size"):
        score_pools(rows, rows, feature, extractor_available=True, overall_size=1)


def test_quality_report_train_only_and_reproducible(reviewed_inputs, tmp_path):
    def extractor(batch):
        assert batch.dtype == torch.uint8
        assert batch.shape == (1, 3, 200, 200)
        assert torch.equal(batch[:, 0], batch[:, 1])
        return batch.float().mean((2, 3))

    kwargs = dict(
        extractor=extractor,
        extractor_metadata={"name": "fixture extractor"},
        overall_size=2,
        per_class_size=2,
        subsets=3,
    )
    out = tmp_path / "assessment"
    report = assess_quality(*reviewed_inputs, out, **kwargs)
    assert report == assess_quality(*reviewed_inputs, tmp_path / "repeat", **kwargs)
    assert report["reference_split"] == "train"
    assert report["kid"]["overall"]["status"] == "measured"
    assert report["kid"]["overall"]["real_count"] == 6
    assert report["kid"]["overall"]["synthetic_count"] == 5
    assert report["kid"]["crazing"]["reasons"] == ["insufficient_samples"]
    assert report["reviewer_observations"][0]["observation"] == "Unclear defect"
    assert len(list(out.glob("*.png"))) == 6
    assert any(
        grid["Closest real train matches for rejected synthetic (same column)"]
        for grid in report["grid_image_ids"].values()
    )
    with Image.open(out / "crazing.png") as sheet:
        sheet.verify()
    assert json.loads((out / "quality.json").read_bytes()) == report
    assert "Unclear defect" in (out / "quality.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("tamper", ["approved", "image", "manifest"])
def test_changed_inputs_are_rejected(reviewed_inputs, tmp_path, tamper):
    generation, _, manifest, _, review, _ = reviewed_inputs
    if tamper == "approved":
        (review / "approved.jsonl").write_bytes(b"{}")
        error = "Review export"
    elif tamper == "image":
        next(generation.rglob("*.png")).write_bytes(b"bad image")
        error = "Candidate audit"
    else:
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        error = "Manifest SHA-256"
    output = tmp_path / "assessment"
    with pytest.raises(ValueError, match=error):
        assess_quality(
            *reviewed_inputs,
            output,
            extractor=None,
            extractor_metadata={"name": "fixture"},
        )
    assert not output.exists()


def test_cli_without_weights_still_writes_qualitative_report(reviewed_inputs, tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("reports/\n", encoding="utf-8")
    generation, real_root, manifest, audit, review, digest = reviewed_inputs
    output = tmp_path / "reports" / "quality"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/assess_synthetic_quality.py"),
            "--generation",
            str(generation),
            "--real-root",
            str(real_root),
            "--manifest",
            str(manifest),
            "--real-audit",
            str(audit),
            "--review",
            str(review),
            "--approved-manifest-sha256",
            digest,
            "--overall-size",
            "2",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "unavailable"
    report = json.loads((output / "quality.json").read_bytes())
    assert report["extractor"]["status"] == "missing_local_weights"
    assert report["kid"]["overall"]["mean"] is None
    assert report["kid"]["overall"]["reasons"] == ["feature_weights_unavailable"]
    assert len(list(output.glob("*.png"))) == 6


def test_local_weights_must_match_explicit_digest(tmp_path):
    path = tmp_path / "weights.pth"
    assert local_extractor(path, None)[0] is None
    path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="approved digest"):
        local_extractor(path, "a" * 64)
