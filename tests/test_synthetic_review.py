"""Synthetic-fixture audit, review, and classifier handoff tests."""

import io
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from random import Random

import pytest
from PIL import Image

from industrial_defect_diffusion.data.preprocessing import read_frozen_manifest
from industrial_defect_diffusion.data.synthetic import ApprovedSyntheticDataset
from industrial_defect_diffusion.data.validation import validate_dataset
from industrial_defect_diffusion.generation.candidates import candidate_plan, generate_candidates
from industrial_defect_diffusion.generation.quality import audit_candidates, write_contact_sheets
from industrial_defect_diffusion.generation.review import (
    json_bytes,
    review_candidates,
    review_template,
)
from industrial_defect_diffusion.generation.sampling import MODEL_ID, REVISION
from industrial_defect_diffusion.utils.experiment_config import load_config

ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-11-13T12:00:00Z"


@pytest.fixture
def candidates(tmp_path, frozen_images, monkeypatch):
    real_root, manifest, metadata = frozen_images
    rows, details = read_frozen_manifest(manifest, metadata)
    held_out = next(row for row in rows if row["split"] == "test")
    with Image.open(io.BytesIO((real_root / held_out["relative_path"]).read_bytes())) as original:
        held_image = original.convert("RGB")
    real_audit = tmp_path / "real-audit.json"
    real_audit.write_bytes(json_bytes(validate_dataset(real_root, near_threshold=0)))
    open_file = Path.open

    def guard(path, *args, **kwargs):
        assert not path.is_relative_to(real_root), "The review must never read real pixels"
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    config = load_config(ROOT / "configs/cpu-smoke.yaml")
    config["preprocessing"]["diffusion_resolution"] = 200
    source = {
        "base_model_id": MODEL_ID,
        "base_revision": REVISION,
        "adapter_sha256": "a" * 64,
        "adapter_checkpoint_sha256": "b" * 64,
        "adapter_checkpoint": "step-000100",
        "real_manifest_sha256": details["manifest_sha256"],
        "dataset_source_sha256": details["dataset_source_sha256"],
    }
    plan = candidate_plan(config, source, {})
    output = tmp_path / "candidates"

    def sample(batch):
        return [
            Image.frombytes("L", (9, 8), Random(row["seed"]).randbytes(72))
            .resize((200, 200))
            .convert("RGB")
            for row in batch
        ]

    ledger = generate_candidates(output, plan, sample)
    return output, manifest, real_audit, details["manifest_sha256"], rows, held_image, ledger


def audit(fixture):
    return audit_candidates(*fixture[:4], threshold=0)


def change_row(fixture, callback):
    ledger = fixture[-1]
    rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    callback(rows)
    ledger.write_bytes(b"".join(json_bytes(row) for row in rows))


def replace_image(fixture, row, image, *, png_metadata=None):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=png_metadata)
    payload = buffer.getvalue()
    (fixture[0] / row["relative_path"]).write_bytes(payload)
    row["sha256"] = sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ("class", "invalid_generation_provenance"),
        ("prompt", "invalid_generation_provenance"),
        ("provenance", "invalid_generation_provenance"),
        ("corrupt", "invalid_image"),
        ("channels", "invalid_image"),
        ("hash", "image_sha256_mismatch"),
    ],
)
def test_candidate_technical_failures(candidates, change, blocker):
    def mutate(rows):
        row = rows[0]
        if change == "class":
            row["class_id"] = 4
        elif change == "prompt":
            row["prompt"] = "wrong defect"
        elif change == "provenance":
            row.pop("adapter_sha256")
        elif change == "corrupt":
            (candidates[0] / row["relative_path"]).write_bytes(b"bad PNG")
        elif change == "channels":
            replace_image(candidates, row, Image.new("L", (200, 200)))
        else:
            row["sha256"] = "f" * 64

    change_row(candidates, mutate)
    assert blocker in audit(candidates)["findings"][0]["blockers"]


def test_exact_near_and_held_out_similarity(candidates):
    from PIL.PngImagePlugin import PngInfo

    def mutate(rows):
        original = candidates[0] / rows[0]["relative_path"]
        (candidates[0] / rows[1]["relative_path"]).write_bytes(original.read_bytes())
        rows[1]["sha256"] = rows[0]["sha256"]
        tag = PngInfo()
        tag.add_text("fixture", "different encoding")
        with Image.open(original) as image:
            replace_image(candidates, rows[2], image, png_metadata=tag)
        replace_image(candidates, rows[3], candidates[5])

    change_row(candidates, mutate)
    findings = audit(candidates)["findings"]
    assert "exact_candidate_duplicate" in findings[0]["warnings"]
    assert findings[2]["candidate"]["image_id"] in findings[0]["near_candidates"]
    assert "held_out_similarity" in findings[3]["blockers"]
    assert "held_out_similarity" in json.dumps(findings[3])
    for row in candidates[4]:
        if row["split"] != "train":
            assert row["image_id"] not in json.dumps(findings)


def fill(template, reviewer, decision="accepted", reason=None):
    for row in template["decisions"]:
        row.update(decision=decision, reviewer_id=reviewer, reviewed_at=STAMP, reason=reason)
    return template


def test_review_export_loads_with_classifier(candidates, tmp_path):
    report = audit(candidates)
    write_contact_sheets(candidates[0], report, tmp_path)
    assert len(list(tmp_path.glob("*-001.png"))) == 6
    assert not any(item["blockers"] for item in report["findings"])
    primary = fill(review_template(report), "reviewer-a")
    second = review_candidates(report, primary)["secondary_template"]
    assert len(second["decisions"]) == 6
    result = review_candidates(report, primary, fill(second, "reviewer-b"))
    assert result["report"]["has_all_six_classes"]
    assert (
        result["report"]["accepted_manifest_sha256"]
        == sha256(result["accepted_payload"]).hexdigest()
    )
    dataset = ApprovedSyntheticDataset(
        candidates[0],
        result["accepted_payload"],
        candidates[4],
        candidates[3],
        base_revision=REVISION,
        resolution=32,
        flip_probability=0,
    )
    assert sorted(dataset[index]["label"] for index in range(len(dataset))) == list(range(6))
    assert dataset[0]["image"].shape == (3, 32, 32)
    assert json.loads(result["accepted_payload"].splitlines()[0])["reviewed_at"] == STAMP
    assert b'"pending"' in candidates[-1].read_bytes()


def test_pending_rejection_and_dispute_workflow(candidates):
    report = audit(candidates)
    primary = review_template(report)
    with pytest.raises(ValueError, match="reviewer ID"):
        review_candidates(report, primary)
    fill(primary, "reviewer-a")
    primary["decisions"][0].update(decision="rejected", reason="wrong defect", disputed=True)
    second = review_candidates(report, primary)["secondary_template"]
    fill(second, "reviewer-b")
    with pytest.raises(ValueError, match="disagreement"):
        review_candidates(report, primary, second)
    rejected = primary["decisions"][0]["image_id"]
    for row in second["decisions"]:
        if row["image_id"] == rejected:
            row.update(decision="rejected", reason="wrong defect")
    with pytest.raises(ValueError, match="missing required"):
        review_candidates(report, primary, {**second, "decisions": second["decisions"][:-1]})
    result = review_candidates(report, primary, second)
    assert result["report"]["rejections_by_reason"] == {"wrong defect": 1}
    assert result["report"]["shortfalls"]["crazing"] == 1
    assert rejected.encode() not in result["accepted_payload"]


def test_blocked_candidate_cannot_be_approved(candidates):
    change_row(candidates, lambda rows: rows[0].update(seed=99))
    report = audit(candidates)
    with pytest.raises(ValueError, match="Cannot accept blocked"):
        review_candidates(report, fill(review_template(report), "reviewer-a"))


def test_cli_audit_and_export(candidates, tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("review/\n", encoding="utf-8")
    root, manifest, real_audit, digest = candidates[:4]
    inputs = [
        "--generation",
        str(root),
        "--manifest",
        str(manifest),
        "--real-audit",
        str(real_audit),
        "--approved-manifest-sha256",
        digest,
        "--threshold",
        "0",
    ]

    def cli(stage, output, *extra):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/review_synthetic.py"),
                stage,
                *inputs,
                "--output",
                output,
                *extra,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    cli("audit", "review/first")
    primary = tmp_path / "review/first/primary.json"
    primary.write_bytes(json_bytes(fill(json.loads(primary.read_bytes()), "reviewer-a")))
    cli("second-review", "review/second", "--primary", str(primary))
    secondary = tmp_path / "review/second/secondary.json"
    secondary.write_bytes(json_bytes(fill(json.loads(secondary.read_bytes()), "reviewer-b")))
    cli("export", "review/final", "--primary", str(primary), "--secondary", str(secondary))
    result = tmp_path / "review/final"
    assert len((result / "approved.jsonl").read_bytes().splitlines()) == 6
    assert json.loads((result / "review-summary.json").read_bytes())["has_all_six_classes"]
