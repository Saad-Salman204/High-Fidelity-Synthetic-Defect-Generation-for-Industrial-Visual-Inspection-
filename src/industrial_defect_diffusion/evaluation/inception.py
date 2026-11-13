"""Load an explicitly approved local Inception extractor without network access."""

import os
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch

from .kid import Extractor


def local_extractor(path: Path | None, approved_hash: str | None) -> tuple[Extractor | None, dict]:
    """Return unavailable for absent weights, and reject unverified supplied files."""
    metadata = {
        "name": "torch-fidelity Inception-v3",
        "feature_layer": "2048",
        "implementation_version": "0.3.0",
        "preprocessing": "299×299 TensorFlow-compatible bilinear resize, then (x - 128) / 128",
        "torch_version": str(torch.__version__),
        "weights_sha256": None,
    }
    if path is None or not path.is_file():
        return None, {**metadata, "status": "missing_local_weights"}
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if digest != approved_hash:
        raise ValueError("Feature weights SHA-256 does not match the approved digest")
    try:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    except ImportError:
        return None, {**metadata, "weights_sha256": digest, "status": "missing_torch_fidelity"}
    if version("torch-fidelity") != "0.3.0":
        raise ValueError("Install the pinned torch-fidelity==0.3.0")
    # v0.3.0 delegates local loading to torch.load. Require the restricted loader mode.
    with patch.dict(os.environ, {"TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}):
        model = (
            FeatureExtractorInceptionV3(
                "inception-v3-compat",
                ["2048"],
                feature_extractor_weights_path=str(path),
            )
            .cpu()
            .eval()
        )

    @torch.inference_mode()
    def extract(batch: torch.Tensor) -> torch.Tensor:
        return model(batch)[0]

    return extract, {**metadata, "weights_sha256": digest, "status": "loaded"}
