"""Offline Stable Diffusion sampling with the reviewed UNet adapter."""

import json
import re
from hashlib import sha256
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from ..diffusion.checkpoints import load_adapter
from ..diffusion.local_model import file_sha256, load_local_components, snapshot_fingerprints
from ..diffusion.training import PROMPTS, TARGETS, prepare_adapter
from ..utils.experiment_config import validate_config

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
SCHEDULER = "DDIMScheduler"


def verify_checkpoint(
    model_dir: Path,
    adapter: Path,
    config: dict,
    manifest: dict,
    approved_metadata_hash: str,
) -> dict:
    """Bind the selected adapter to approved metadata, base files and frozen training data."""
    if (config["diffusion"]["model_id"], config["diffusion"]["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("Generation requires the pinned SD v1.5 model ID and revision")
    payload = (adapter / "checkpoint.json").read_bytes()
    if sha256(payload).hexdigest() != approved_metadata_hash:
        raise ValueError("Adapter metadata does not match the approved SHA-256")
    checkpoint = json.loads(payload)
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("target_modules") != list(TARGETS)
        or checkpoint.get("prompts") != list(PROMPTS)
        or type(checkpoint.get("update")) is not int
        or checkpoint["update"] <= 0
    ):
        raise ValueError("Incompatible LoRA checkpoint metadata or prompt catalog")
    provenance = checkpoint["provenance"]
    training = validate_config(provenance["config"])
    if (
        training["status"] != "approved"
        or training["classes"] != config["classes"]
        or (training["diffusion"]["model_id"], training["diffusion"]["revision"])
        != (MODEL_ID, REVISION)
        or provenance["manifest_sha256"] != manifest["manifest_sha256"]
        or provenance["dataset_source_sha256"] != manifest["dataset_source_sha256"]
        or not re.fullmatch(r"[0-9a-f]{40}", provenance.get("git_commit", ""))
    ):
        raise ValueError("Checkpoint training provenance is incompatible with the approved inputs")
    variant = provenance["variant"]
    if variant not in (None, "fp16"):
        raise ValueError("Unsupported model file variant")
    if snapshot_fingerprints(model_dir, variant) != provenance["base_files"]:
        raise ValueError("Local base model fingerprints differ from the training checkpoint")
    digest = file_sha256(adapter / "adapter.safetensors")
    if digest != checkpoint["file_sha256"]["adapter.safetensors"]:
        raise ValueError("Adapter SHA-256 does not match checkpoint metadata")
    return {
        "base_model_id": MODEL_ID,
        "base_revision": REVISION,
        "adapter_sha256": digest,
        "adapter_checkpoint_sha256": approved_metadata_hash,
        "adapter_checkpoint": f"step-{checkpoint['update']:06d}",
        "real_manifest_sha256": manifest["manifest_sha256"],
        "dataset_source_sha256": manifest["dataset_source_sha256"],
        "base_files": provenance["base_files"],
        "variant": variant,
        "training_commit": provenance["git_commit"],
        "training_settings": training["diffusion"],
    }


def make_sampler(
    model_dir: Path, adapter: Path, source: dict, device: torch.device, precision: str
):
    """Build a local pipeline and return a batch sampler with independent image RNGs."""
    parts = load_local_components(model_dir, source["variant"])
    prepare_adapter(
        parts,
        {**source["training_settings"], "precision": precision, "gradient_checkpointing": False},
        device,
    )
    load_adapter(adapter, parts)
    parts.unet.requires_grad_(False).eval()
    pipeline = StableDiffusionPipeline(
        vae=parts.vae,
        text_encoder=parts.text_encoder,
        tokenizer=parts.tokenizer,
        unet=parts.unet,
        scheduler=DDIMScheduler.from_config(
            parts.scheduler.config,
            clip_sample=False,
            steps_offset=1,
        ),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def sample(rows: list[dict]) -> list[Image.Image]:
        first = rows[0]
        return pipeline(
            prompt=[row["prompt"] for row in rows],
            negative_prompt=[row["negative_prompt"] for row in rows],
            generator=[torch.Generator(device="cpu").manual_seed(row["seed"]) for row in rows],
            height=first["height"],
            width=first["width"],
            num_inference_steps=first["inference_steps"],
            guidance_scale=first["guidance_scale"],
            eta=0.0,
            output_type="pil",
        ).images

    return sample
