"""Exercise preprocessing with synthetic images and the frozen split contract."""

import pytest
import torch
from PIL import Image

from industrial_defect_diffusion.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NEUManifestDataset,
    make_loader,
    preprocess_image,
    read_frozen_manifest,
)


def test_grayscale_bicubic_and_normalization():
    image = Image.linear_gradient("L").resize((200, 200))
    diffusion = preprocess_image(image, "diffusion")
    assert diffusion.shape == (3, 512, 512)
    assert diffusion.dtype == torch.float32
    assert diffusion.min() == -1 and diffusion.max() == 1
    assert torch.equal(diffusion[0], diffusion[2])
    classifier = preprocess_image(image, "classifier")
    assert classifier.shape == (3, 224, 224)
    restored = classifier * torch.tensor(IMAGENET_STD)[:, None, None]
    restored += torch.tensor(IMAGENET_MEAN)[:, None, None]
    expected = image.resize((224, 224), Image.Resampling.BICUBIC)
    assert torch.allclose(
        restored[0],
        torch.tensor(list(expected.get_flattened_data())).reshape(224, 224) / 255,
        atol=1e-6,
    )
    assert torch.allclose(restored[0], restored[2], atol=1e-6)
    assert torch.equal(classifier, preprocess_image(image.convert("RGB"), "classifier"))


def test_partition_order_and_training_only_flips(frozen_images):
    root, manifest, metadata = frozen_images
    rows, _ = read_frozen_manifest(manifest, metadata)
    validation = NEUManifestDataset(
        root,
        rows,
        split="val",
        target="classifier",
        train_flip_probability=1,
    )
    train = NEUManifestDataset(
        root,
        rows,
        split="train",
        target="classifier",
        train_flip_probability=1,
    )
    plain_train = NEUManifestDataset(
        root,
        rows,
        split="train",
        target="classifier",
        train_flip_probability=0,
    )
    assert torch.equal(train[0]["image"], plain_train[0]["image"].flip((-1, -2)))
    assert torch.equal(validation[0]["image"], validation[0]["image"])
    batch = next(iter(make_loader(validation, batch_size=6, seed=42)))
    assert batch["image_id"] == [row["image_id"] for row in rows if row["split"] == "val"]
    assert sorted(batch["label"].tolist()) == list(range(6))
    with pytest.raises(ValueError, match="only allowed on training"):
        make_loader(validation, batch_size=2, seed=42, balanced=True)


def test_changed_csv_and_source_are_rejected(frozen_images):
    root, manifest, metadata = frozen_images
    rows, _ = read_frozen_manifest(manifest, metadata)
    dataset = NEUManifestDataset(root, rows, split="train", target="classifier")
    path = root / dataset.rows[0]["relative_path"]
    Image.new("L", (200, 200), 20).save(path)
    with pytest.raises(ValueError, match="Source image changed"):
        dataset[0]
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Manifest SHA-256"):
        read_frozen_manifest(manifest, metadata)
