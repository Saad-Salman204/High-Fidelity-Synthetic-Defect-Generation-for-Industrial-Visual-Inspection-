"""Exercise splitting against the actual audit format with authored images."""

from random import Random

import pytest
from PIL import Image

from industrial_defect_diffusion.data.splits import (
    audit_digest,
    create_splits,
    dataset_source_digest,
    freeze_splits,
    validate_manifest,
)
from industrial_defect_diffusion.data.validation import CLASSES, validate_dataset


@pytest.fixture
def reviewed(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rng = Random(7)
    for label in CLASSES:
        for index in range(10):
            image = Image.frombytes("L", (9, 8), rng.randbytes(72)).resize((200, 200))
            image.save(root / f"{label}_{index}.bmp")
    first = root / "crazing_0.bmp"
    (root / "crazing_10.bmp").write_bytes(first.read_bytes())
    with Image.open(first) as image:
        image.save(root / "crazing_11.png")
    audit = validate_dataset(root, near_threshold=0)
    review = {
        "schema_version": 1,
        "audit_sha256": audit_digest(audit),
        "reviewer": "Synthetic fixture reviewer",
        "inventory_note": "Reduced synthetic fixture",
        "near_duplicates": [
            {"paths": pair["paths"], "decision": "same"} for pair in audit["near_duplicate_pairs"]
        ],
    }
    return root, audit, review


def test_split_coverage_and_transitive_groups(reviewed):
    _, audit, review = reviewed
    rows, metadata = create_splits(audit, review, source_version="a" * 64)
    matching = [
        row
        for row in rows
        if row["relative_path"]
        in (
            "crazing_0.bmp",
            "crazing_10.bmp",
            "crazing_11.png",
        )
    ]
    assert len(matching) == 3
    assert len({(row["group_id"], row["split"]) for row in matching}) == 1
    assert metadata["seed"] == 42
    assert metadata["counts_by_class"]["inclusion"] == {"train": 7, "val": 2, "test": 1}
    assert create_splits(audit, review, source_version="a" * 64) == (rows, metadata)
    matching[0]["split"] = "val" if matching[1]["split"] == "train" else "train"
    with pytest.raises(ValueError, match="crosses split"):
        validate_manifest(rows)


def test_unresolved_pair_and_too_few_groups(reviewed):
    _, audit, review = reviewed
    with pytest.raises(ValueError, match="Unresolved near-duplicate"):
        create_splits(audit, {**review, "near_duplicates": []}, source_version="a" * 64)
    amended = {
        **review,
        "near_duplicates": [],
        "protocol_amendment": {
            "amendment_id": "AMEND-1",
            "rationale": "Candidates cannot be conclusively adjudicated",
            "reviewer": "Reviewer",
            "date": "2026-11-25",
            "acknowledgment": "Unresolved dHash candidates remain unverified and ungrouped.",
        },
    }
    rows, metadata = create_splits(
        audit, amended, source_version="a" * 64, allow_unresolved_dhash=True
    )
    assert metadata["unresolved_dhash_pairs"]
    with pytest.raises(ValueError, match="complete protocol_amendment"):
        create_splits(
            audit,
            {**amended, "protocol_amendment": {}},
            source_version="a" * 64,
            allow_unresolved_dhash=True,
        )
    review["exclusions"] = {
        row["relative_path"]: "Synthetic exclusion"
        for row in audit["files"]
        if row["class_name"] == "scratches"
    }
    with pytest.raises(ValueError, match="three independent groups"):
        create_splits(audit, review, source_version="a" * 64)


def test_conflicts_require_explicit_exclusion(reviewed):
    root, _, review = reviewed
    conflict = root / "inclusion_99.bmp"
    conflict.write_bytes((root / "crazing_0.bmp").read_bytes())
    audit = validate_dataset(root, near_threshold=0)
    review["audit_sha256"] = audit_digest(audit)
    with pytest.raises(ValueError, match="unresolved audit finding"):
        create_splits(audit, review, source_version="a" * 64)


def test_freeze_reuse_and_stale_inputs(reviewed, tmp_path):
    root, audit, review = reviewed
    output = tmp_path / "freeze"
    kwargs = {"source_version": "a" * 64}
    first = freeze_splits(root, audit, review, output, **kwargs)
    second = freeze_splits(root, audit, review, tmp_path / "other", **kwargs)
    assert first == second == freeze_splits(root, audit, review, output, **kwargs)
    assert (output / "manifest.csv").read_bytes() == (
        tmp_path / "other" / "manifest.csv"
    ).read_bytes()
    with pytest.raises(ValueError, match="differs"):
        freeze_splits(root, audit, review, output, seed=43, **kwargs)
    (root / "crazing_1.bmp").unlink()
    with pytest.raises(ValueError, match="Source audit changed"):
        freeze_splits(root, audit, review, output, **kwargs)


def test_source_fingerprint_is_deterministic_and_uses_actual_files(reviewed):
    root, audit, _ = reviewed
    first = dataset_source_digest(root, audit)
    assert first == dataset_source_digest(root, audit)
    (root / "Thumbs.db").write_bytes(b"windows metadata")
    assert first == dataset_source_digest(root, audit)
    image = root / "crazing_1.bmp"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs from audit"):
        dataset_source_digest(root, audit)


def test_source_fingerprint_rejects_audit_hash_mismatch(reviewed):
    root, audit, _ = reviewed
    altered = {**audit, "files": [dict(row) for row in audit["files"]]}
    altered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from audit"):
        dataset_source_digest(root, altered)


def test_frozen_metadata_uses_source_fingerprint_and_separate_audit_digest(reviewed, tmp_path):
    root, audit, review = reviewed
    fingerprint = dataset_source_digest(root, audit)
    output = tmp_path / "freeze"
    metadata = freeze_splits(root, audit, review, output)
    assert metadata["dataset_source_sha256"] == fingerprint
    assert metadata["audit_sha256"] == audit_digest(audit)
    assert "dataset_archive_sha256" not in metadata
