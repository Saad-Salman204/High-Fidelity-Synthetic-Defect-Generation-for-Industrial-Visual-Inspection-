"""Read frozen NEU manifests and prepare images for the two model families."""

import csv
import io
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as F

from ..utils.experiment_config import relative_path
from .splits import FIELDS, SPLITS, validate_manifest

Target = Literal["classifier", "diffusion"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_frozen_manifest(manifest: Path, metadata: Path) -> tuple[list[dict], dict]:
    """Verify the frozen CSV and validate its rows without opening any images."""
    payload = manifest.read_bytes()
    details = json.loads(metadata.read_text(encoding="utf-8"))
    if details.get("schema_version") != 1:
        raise ValueError("Unsupported split metadata schema")
    if sha256(payload).hexdigest() != details.get("manifest_sha256"):
        raise ValueError("Manifest SHA-256 does not match split metadata")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    if reader.fieldnames != list(FIELDS):
        raise ValueError("Manifest columns do not match the version-1 contract")
    rows = []
    for row in reader:
        if None in row or any(not value for value in row.values()):
            raise ValueError("Manifest contains a missing or extra field")
        relative_path(row["relative_path"], "image path")
        if (row["schema_version"], row["mode"], row["width"], row["height"]) != (
            "1",
            "L",
            "200",
            "200",
        ):
            raise ValueError(f"Invalid original image metadata: {row['relative_path']}")
        if row["source_version"] != details.get("dataset_source_sha256"):
            raise ValueError("Manifest source version does not match split metadata")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", row[field])
            for field in ("source_version", "image_id", "sha256", "group_id")
        ):
            raise ValueError("Manifest contains an invalid fingerprint")
        row["class_id"] = int(row["class_id"])
        rows.append(row)
    validate_manifest(rows)
    return rows, details


def preprocess_image(
    image: Image.Image,
    target: Target,
    *,
    resolution: int | None = None,
    flip_probability: float = 0,
) -> torch.Tensor:
    """Resize the whole grayscale image, replicate RGB channels and normalize."""
    if target not in ("classifier", "diffusion"):
        raise ValueError("Unknown preprocessing target")
    if resolution is None:
        resolution = 224 if target == "classifier" else 512
    if type(resolution) is not int or resolution <= 0 or not 0 <= flip_probability <= 1:
        raise ValueError("Invalid resolution or flip probability")
    image = image.convert("L").resize((resolution, resolution), Image.Resampling.BICUBIC)
    if flip_probability:
        if torch.rand(()) < flip_probability:
            image = F.hflip(image)
        if torch.rand(()) < flip_probability:
            image = F.vflip(image)
    tensor = F.pil_to_tensor(image.convert("RGB")).to(torch.float32).div_(255)
    if target == "diffusion":
        return tensor.mul_(2).sub_(1)
    return F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class NEUManifestDataset(Dataset):
    """Expose a single frozen partition, checking source bytes as each image is read."""

    def __init__(
        self,
        root: Path,
        rows: list[dict],
        *,
        split: str,
        target: Target,
        resolution: int | None = None,
        train_flip_probability: float = 0.5,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Invalid dataset partition: {split}")
        validate_manifest(rows)
        if not root.is_dir():
            raise NotADirectoryError(root)
        self.root = root.resolve()
        self.split = split
        self.target = target
        self.resolution = resolution
        self.flip_probability = train_flip_probability if split == "train" else 0
        self.rows = tuple(row.copy() for row in rows if row["split"] == split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        relative = relative_path(row["relative_path"], "image path")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Image path escapes the dataset root: {relative}")
        with path.open("rb") as stream:
            payload = stream.read(8 * 1024 * 1024 + 1)
        if sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"Source image changed since the audit: {relative}")
        with Image.open(io.BytesIO(payload)) as image:
            if image.mode != "L" or image.size != (200, 200):
                raise ValueError(f"Expected a 200×200 grayscale image: {relative}")
            pixels = preprocess_image(
                image,
                self.target,
                resolution=self.resolution,
                flip_probability=self.flip_probability,
            )
        return {"image": pixels, "label": row["class_id"], "image_id": row["image_id"]}


def make_loader(
    dataset: NEUManifestDataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    balanced: bool = False,
) -> DataLoader:
    """Build a seeded loader. Evaluation order is stable and never resampled."""
    training = dataset.split == "train"
    if balanced and not training:
        raise ValueError("Class-balanced sampling is only allowed on training data")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balanced:
        labels = [row["class_id"] for row in dataset.rows]
        counts = {label: labels.count(label) for label in set(labels)}
        sampler = WeightedRandomSampler(
            [1 / counts[label] for label in labels],
            len(labels),
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=num_workers,
    )
