"""Save and reload adapter-only weights and optimizer-boundary resume state."""

import json
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from .local_model import file_sha256
from .training import PROMPTS, TARGETS, Components


def save_checkpoint(
    path: Path,
    parts: Components,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    provenance: dict,
    cursor: tuple[int, int, int],
) -> None:
    """Save adapter weights separately from the local optimizer and RNG state."""
    path.mkdir(exist_ok=False)
    tensors = get_peft_model_state_dict(parts.unet)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        path / "adapter.safetensors",
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        },
        path / "training-state.pt",
    )
    update, epoch, next_batch = cursor
    metadata = {
        "schema_version": 1,
        "provenance": provenance,
        "target_modules": list(TARGETS),
        "prompts": list(PROMPTS),
        "update": update,
        "epoch": epoch,
        "next_batch": next_batch,
        "file_sha256": {
            name: file_sha256(path / name) for name in ("adapter.safetensors", "training-state.pt")
        },
    }
    (path / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_adapter(path: Path, parts: Components) -> None:
    """Load saved tensors into a compatible, already configured UNet adapter."""
    metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("target_modules") != list(TARGETS):
        raise ValueError("Unsupported adapter checkpoint")
    if file_sha256(path / "adapter.safetensors") != metadata["file_sha256"]["adapter.safetensors"]:
        raise ValueError("Adapter SHA-256 does not match checkpoint metadata")
    state = load_file(path / "adapter.safetensors")
    expected = get_peft_model_state_dict(parts.unet)
    if state.keys() != expected.keys() or any(
        value.shape != expected[key].shape for key, value in state.items()
    ):
        raise ValueError("Checkpoint adapter keys or tensor shapes do not match this UNet")
    result = set_peft_model_state_dict(parts.unet, state)
    if result.unexpected_keys or any(".lora_" in key for key in result.missing_keys):
        raise ValueError("Checkpoint did not load every expected adapter tensor")


def resume_checkpoint(
    path: Path,
    parts: Components,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    provenance: dict,
) -> tuple[int, int, int]:
    """Restore compatible training state; reject different inputs or runtime settings."""
    metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("provenance") != provenance
        or metadata.get("target_modules") != list(TARGETS)
        or metadata.get("prompts") != list(PROMPTS)
    ):
        raise ValueError("Checkpoint provenance differs from the current training inputs")
    cursor = tuple(metadata[key] for key in ("update", "epoch", "next_batch"))
    if any(type(item) is not int or item < 0 for item in cursor):
        raise ValueError("Invalid checkpoint training cursor")
    if file_sha256(path / "training-state.pt") != metadata["file_sha256"]["training-state.pt"]:
        raise ValueError("Resume state SHA-256 does not match checkpoint metadata")
    state = torch.load(path / "training-state.pt", map_location="cpu", weights_only=True)
    load_adapter(path, parts)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    return cursor
