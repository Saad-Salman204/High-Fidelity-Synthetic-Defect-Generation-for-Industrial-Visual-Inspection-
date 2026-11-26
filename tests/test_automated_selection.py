"""Test the separately named audit-policy synthetic selection workflow."""

import json
from hashlib import sha256

import pytest
from PIL import Image

from industrial_defect_diffusion.data.automated_selection import (
    POLICY_ID,
    create_automated_selection,
)
from industrial_defect_diffusion.data.preprocessing import read_frozen_manifest
from industrial_defect_diffusion.data.synthetic import AutomatedSelectionDataset
from industrial_defect_diffusion.utils.experiment_config import CLASS_MAPPING

REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def _fixture(tmp_path, frozen_images, *, copies=2, corrupt=False):
    generation = tmp_path / "generation"
    generation.mkdir()
    audit_key = "f" * 64
    _, _, real_metadata = frozen_images
    real_manifest_sha256 = json.loads(real_metadata.read_text(encoding="utf-8"))["manifest_sha256"]
    candidates, findings, proposals = [], [], []
    image_number = 0
    for label, class_id in CLASS_MAPPING.items():
        for index in range(copies):
            image_id = f"candidate-{class_id}-{index}"
            relative = f"batch-{image_number:04d}/{image_id}.png"
            path = generation / relative
            path.parent.mkdir()
            if corrupt and label == "crazing" and index == 0:
                payload = b"not-an-image"
                path.write_bytes(payload)
            else:
                Image.new("L", (16, 16), class_id * 20 + index).save(path)
                payload = path.read_bytes()
            digest = sha256(payload).hexdigest()
            row = {
                "schema_version": 1,
                "image_id": image_id,
                "relative_path": relative,
                "sha256": digest,
                "class_id": class_id,
                "class_name": label,
                "prompt": f"fixture {label}",
                "seed": 42 + image_number,
                "run_id": "fixture-run",
                "scheduler": "DDIMScheduler",
                "inference_steps": 2,
                "width": 16,
                "height": 16,
                "reviewer_id": None,
                "base_revision": REVISION,
                "adapter_sha256": "a" * 64,
                "real_manifest_sha256": real_manifest_sha256,
                "quality_status": "pending",
                "rejection_reason": None,
                "parent_ids": [],
                "mask_id": None,
                "split": "train",
            }
            candidates.append(row)
            finding = {
                "candidate": row,
                "blockers": [],
                "warnings": [],
                "near_candidates": ["nearby-id"] if index else [],
                "observed_sha256": digest,
            }
            findings.append(finding)
            proposals.append(
                {
                    "candidate": row,
                    "audit_flags": {
                        "blockers": [],
                        "warnings": [],
                        "near_candidate_count": len(finding["near_candidates"]),
                    },
                    "audit_sha256": audit_key,
                    "generation_run_id": "fixture-run",
                    "proposal_status": "eligible_for_human_review_not_accepted",
                }
            )
            image_number += 1
    candidates_path = generation / "candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    audit = {
        "findings": findings,
        "real_audit_sha256": audit_key,
        "manifest_sha256": real_manifest_sha256,
        "run_id": "fixture-run",
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    proposal_path = tmp_path / "proposal.jsonl"
    proposal_path.write_text("".join(json.dumps(row) + "\n" for row in proposals), encoding="utf-8")
    return generation, candidates_path, audit_path, proposal_path


def _select(tmp_path, files, name="selection"):
    generation, candidates_path, audit_path, proposal_path = files
    return create_automated_selection(
        proposal_path=proposal_path,
        audit_path=audit_path,
        generation_manifest=candidates_path,
        generation_root=generation,
        output=tmp_path / name,
    )


def test_selection_is_complete_reproducible_and_loadable(tmp_path, frozen_images):
    files = _fixture(tmp_path, frozen_images)
    first = _select(tmp_path, files, "selected-a")
    second = _select(tmp_path, files, "selected-b")
    assert first["selection_id"] == POLICY_ID
    assert first["selected_candidate_count"] == 12
    assert first["selected_counts_by_class"] == {label: 2 for label in CLASS_MAPPING}
    assert first["source_generation_manifest_sha256"] == second["source_generation_manifest_sha256"]
    assert first["selection_manifest_sha256"] == second["selection_manifest_sha256"]
    assert first["human_review_status"] == "not_manually_reviewed_or_human_approved"
    assert len(first["selected_images"]) == 12
    assert all(item["sha256"] for item in first["selected_images"])
    payload = (tmp_path / "selected-a/selected.jsonl").read_bytes()
    rows, details = read_frozen_manifest(frozen_images[1], frozen_images[2])
    dataset = AutomatedSelectionDataset(
        tmp_path / "selected-a",
        payload,
        rows,
        details["manifest_sha256"],
        base_revision=REVISION,
        resolution=32,
        flip_probability=0,
    )
    assert len(dataset) == 12
    assert sorted(dataset[index]["label"] for index in range(len(dataset))) == [
        class_id for class_id in range(6) for _ in range(2)
    ]
    assert all(
        row["quality_status"] == "pending" and row["reviewer_id"] is None for row in dataset.rows
    )
    assert any(row["near_candidate_count"] for row in dataset.rows)


def test_integrity_failures_are_explicit_and_other_class_examples_remain(tmp_path, frozen_images):
    files = _fixture(tmp_path, frozen_images, corrupt=True)
    result = _select(tmp_path, files)
    assert result["eligible_candidate_count"] == 12
    assert result["selected_candidate_count"] == 11
    assert result["excluded_integrity_failures"] == [
        {"image_id": "candidate-0-0", "class_name": "crazing", "reason": "image_decode_failure"}
    ]
    assert result["selected_counts_by_class"]["crazing"] == 1
    assert len(result["selected_images"]) == 11


def test_proposal_must_match_audit_policy_exactly(tmp_path, frozen_images):
    files = _fixture(tmp_path, frozen_images)
    files[3].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid status"):
        _select(tmp_path, files)


def test_policy_exclusions_are_recorded_and_near_matches_remain_eligible(tmp_path, frozen_images):
    generation, candidates_path, audit_path, proposal_path = _fixture(tmp_path, frozen_images)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    flagged_id = audit["findings"][0]["candidate"]["image_id"]
    audit["findings"][0]["warnings"] = ["fixture_warning"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    proposal = [
        row
        for row in map(json.loads, proposal_path.read_text(encoding="utf-8").splitlines())
        if row["candidate"]["image_id"] != flagged_id
    ]
    proposal_path.write_text("".join(json.dumps(row) + "\n" for row in proposal), encoding="utf-8")
    result = create_automated_selection(
        proposal_path=proposal_path,
        audit_path=audit_path,
        generation_manifest=candidates_path,
        generation_root=generation,
        output=tmp_path / "selection",
    )
    assert result["selected_candidate_count"] == 11
    assert result["excluded_by_eligibility_policy"] == [
        {
            "image_id": flagged_id,
            "class_name": "crazing",
            "blockers": [],
            "warnings": ["fixture_warning"],
        }
    ]
    assert result["selected_counts_by_class"]["crazing"] == 1
