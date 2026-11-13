"""Exercise splitting against the actual audit format with authored images."""

from random import Random

import pytest
from PIL import Image

from industrial_defect_diffusion.data.splits import (
    audit_digest,
    create_splits,
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
