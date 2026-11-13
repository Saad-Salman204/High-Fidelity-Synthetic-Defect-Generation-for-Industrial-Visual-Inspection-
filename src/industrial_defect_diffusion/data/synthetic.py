"""Read the approved synthetic JSONL handoff without performing quality review."""

import io
import json
import re
from hashlib import file_digest, sha256
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from ..utils.experiment_config import CLASS_MAPPING, relative_path
from .preprocessing import preprocess_image


class ApprovedSyntheticDataset(Dataset):
    """Expose reviewed synthetic images using the classifier's training transform."""

    split = "train"

    def __init__(
        self,
        root: Path,
        payload: bytes,
        real_rows: list[dict],
        real_manifest_sha256: str,
        *,
        base_revision: str,
        resolution: int,
        flip_probability: float,
    ) -> None:
        self.root = root.resolve()
        self.resolution = resolution
        self.flip_probability = flip_probability
        real_ids = {row["image_id"] for row in real_rows}
        real_hashes = {row["sha256"] for row in real_rows}
        train_ids = {row["image_id"] for row in real_rows if row["split"] == "train"}
        seen_ids, seen_paths, seen_hashes = set(), set(), set()
        self.rows = []
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                self._validate(row, real_manifest_sha256, base_revision, train_ids)
                image_id, path, digest = row["image_id"], row["relative_path"], row["sha256"]
                if image_id in real_ids or digest in real_hashes:
                    raise ValueError("Synthetic record overlaps real image IDs or bytes")
                if image_id in seen_ids or path.casefold() in seen_paths or digest in seen_hashes:
                    raise ValueError("Duplicate synthetic ID, path or image bytes")
                seen_ids.add(image_id)
                seen_paths.add(path.casefold())
                seen_hashes.add(digest)
                image_path = self._path(path)
                with image_path.open("rb") as stream:
                    if file_digest(stream, "sha256").hexdigest() != digest:
                        raise ValueError(f"Synthetic image hash mismatch: {path}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Synthetic manifest line {line_number}: {exc}") from exc
            self.rows.append(row)
        if {row["class_name"] for row in self.rows} != set(CLASS_MAPPING):
            raise ValueError("Approved synthetic pool must contain all six NEU classes")
        self.rows.sort(key=lambda row: row["image_id"])

    @staticmethod
    def _validate(row, real_manifest_sha256, base_revision, train_ids) -> None:
        if (
            not isinstance(row, dict)
            or type(row.get("schema_version")) is not int
            or (row["schema_version"] != 1)
        ):
            raise ValueError("Expected a version-1 synthetic record")
        if row.get("quality_status") != "accepted" or row.get("rejection_reason") is not None:
            raise ValueError("Synthetic training manifest may contain only accepted images")
        label = row.get("class_name")
        if (
            not isinstance(label, str)
            or label not in CLASS_MAPPING
            or (type(row.get("class_id")) is not int or row["class_id"] != CLASS_MAPPING[label])
        ):
            raise ValueError("Invalid synthetic class label")
        for key in ("image_id", "reviewer_id", "prompt", "run_id", "scheduler"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing synthetic provenance: {key}")
        for key in ("sha256", "adapter_sha256", "real_manifest_sha256"):
            if not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError(f"Invalid synthetic provenance: {key}")
        if row["real_manifest_sha256"] != real_manifest_sha256:
            raise ValueError("Synthetic provenance refers to a different frozen real manifest")
        if row.get("base_revision") != base_revision:
            raise ValueError("Synthetic base revision does not match the experiment configuration")
        if type(row.get("seed")) is not int or not 0 <= row["seed"] < 2**32:
            raise ValueError("Invalid synthetic generation seed")
        if type(row.get("inference_steps")) is not int or row["inference_steps"] < 1:
            raise ValueError("Invalid synthetic inference steps")
        if any(
            type(row.get(key)) is not int or not 0 < row[key] <= 2048 for key in ("width", "height")
        ):
            raise ValueError("Invalid synthetic image dimensions")
        parents = row.get("parent_ids")
        if not isinstance(parents, list) or any(
            not isinstance(parent, str) or parent not in train_ids for parent in parents
        ):
            raise ValueError("Synthetic parents must come only from the frozen real train split")
        if "mask_id" not in row or row["mask_id"] is not None or parents:
            raise ValueError("This handoff supports text-to-image only (no parents or mask)")
        if "split" in row and row["split"] != "train":
            raise ValueError("Synthetic records must be training-only")
        relative_path(row.get("relative_path"), "synthetic image path")

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Synthetic image path escapes the source root: {relative}")
        return path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = self._path(row["relative_path"])
        with path.open("rb") as stream:
            payload = stream.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024 or sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"Synthetic image hash mismatch or size limit exceeded: {path.name}")
        with Image.open(io.BytesIO(payload)) as image:
            if image.size != (row["width"], row["height"]) or getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Synthetic image dimensions or frame count mismatch: {path.name}")
            tensor = preprocess_image(
                image,
                "classifier",
                resolution=self.resolution,
                flip_probability=self.flip_probability,
            )
        return {"image": tensor, "label": row["class_id"], "image_id": row["image_id"]}
