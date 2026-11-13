"""Small image collections for CPU integration tests."""

from random import Random

import pytest
from PIL import Image

from industrial_defect_diffusion.data.splits import audit_digest, freeze_splits
from industrial_defect_diffusion.data.validation import CLASSES, validate_dataset


@pytest.fixture
def frozen_images(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rng = Random(5)
    for label in CLASSES:
        for index in range(3):
            image = Image.frombytes("L", (20, 20), rng.randbytes(400))
            image.resize((200, 200)).save(root / f"{label}_{index}.bmp")
    audit = validate_dataset(root, near_threshold=0)
    review = {
        "schema_version": 1,
        "reviewer": "Synthetic fixture",
        "audit_sha256": audit_digest(audit),
        "inventory_note": "Small synthetic fixture",
    }
    output = tmp_path / "freeze"
    freeze_splits(root, audit, review, output, source_version="a" * 64)
    return root, output / "manifest.csv", output / "split-metadata.json"
