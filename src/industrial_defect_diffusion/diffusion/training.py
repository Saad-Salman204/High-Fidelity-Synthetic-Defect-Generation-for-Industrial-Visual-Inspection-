"""Train one attention adapter on frozen real NEU training images."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from ..data.preprocessing import NEUManifestDataset, make_loader
from ..utils.experiment_config import CLASS_MAPPING

TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
DESCRIPTIONS = {
    "crazing": "fine interconnected cracks",
    "inclusion": "embedded irregular inclusions",
    "patches": "irregular patches",
    "pitted_surface": "small surface pits",
    "rolled_in_scale": "rolled-in mill scale",
    "scratches": "long narrow scratches",
}
PROMPTS = tuple(
    f"grayscale close-up inspection image of a hot-rolled steel surface with {DESCRIPTIONS[label]}"
    for label in sorted(CLASS_MAPPING, key=CLASS_MAPPING.__getitem__)
)


@dataclass
class Components:
    """Stable Diffusion components required for the training objective."""

    unet: UNet2DConditionModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    scheduler: DDPMScheduler


def training_device(name: str, precision: str) -> torch.device:
    """Resolve the device and reject unsupported precision combinations."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {name} is unavailable. Verify the university GPU setup")
        with torch.cuda.device(index):
            if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("This GPU does not support native bf16. Choose fp16 or fp32")
        return torch.device("cuda", index)
    if device.type != "cpu" or precision != "fp32":
        raise ValueError("CPU training requires fp32")
    return device


def prepare_adapter(parts: Components, settings: dict, device: torch.device) -> None:
    """Freeze the base components and enable only UNet attention LoRA weights."""
    if parts.scheduler.config.prediction_type != "epsilon":
        raise ValueError("The SD v1 LoRA path requires an epsilon-prediction scheduler")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        settings["precision"]
    ]
    for component in (parts.unet, parts.vae, parts.text_encoder):
        component.requires_grad_(False).to(device=device, dtype=dtype).eval()
    parts.unet.add_adapter(
        LoraConfig(
            r=settings["lora_rank"],
            lora_alpha=settings["lora_alpha"],
            target_modules=list(TARGETS),
            init_lora_weights="gaussian",
        )
    )
    for parameter in adapter_parameters(parts.unet):
        parameter.data = parameter.data.float()
    if settings.get("gradient_checkpointing", True):
        parts.unet.enable_gradient_checkpointing()
    parts.unet.train()


def adapter_parameters(unet: UNet2DConditionModel) -> list[torch.nn.Parameter]:
    """Return the trainable parameters after checking the expected adapter names."""
    named = [
        (name, parameter) for name, parameter in unet.named_parameters() if parameter.requires_grad
    ]
    if not named or any(
        not any(
            name.endswith(f".{target}.lora_{side}.default.weight")
            for target in TARGETS
            for side in ("A", "B")
        )
        for name, _ in named
    ):
        raise ValueError("Expected only UNet attention LoRA A/B matrices to be trainable")
    return [parameter for _, parameter in named]


def encode_prompts(parts: Components, device: torch.device) -> torch.Tensor:
    """Encode the six fixed class prompts once using the frozen text encoder."""
    tokens = parts.tokenizer(
        list(PROMPTS),
        padding="max_length",
        truncation=True,
        max_length=parts.tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        return parts.text_encoder(tokens.input_ids.to(device))[0]


def noise_prediction_loss(
    parts: Components,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    prompt_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Encode clean latents, add scheduled noise, and predict that noise."""
    if pixels.ndim != 4 or pixels.shape[1] != 3 or not pixels.shape[0]:
        raise ValueError("Expected a nonempty batch of NCHW RGB images")
    if (
        labels.shape != (len(pixels),)
        or labels.dtype != torch.long
        or ((labels < 0) | (labels >= len(PROMPTS))).any()
    ):
        raise ValueError("Expected one valid NEU class ID per image")
    device = prompt_embeddings.device
    with torch.no_grad():
        distribution = parts.vae.encode(pixels.to(device=device, dtype=parts.vae.dtype))
        latents = distribution.latent_dist.sample() * parts.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            parts.scheduler.config.num_train_timesteps,
            (len(latents),),
            device=device,
        )
        noisy = parts.scheduler.add_noise(latents, noise, timesteps).to(parts.unet.dtype)
    prediction = parts.unet(noisy, timesteps, prompt_embeddings[labels.to(device)]).sample
    if prediction.shape != noise.shape:
        raise ValueError("UNet output must match the latent noise shape")
    return F.mse_loss(prediction.float(), noise.float())


def make_optimizer(parts: Components, settings: dict) -> torch.optim.Optimizer:
    """Build the configured optimizer over the adapter only."""
    name = settings.get("optimizer", "AdamW")
    if name not in ("AdamW", "Adam"):
        raise ValueError("Expected AdamW or Adam")
    optimizer = torch.optim.AdamW if name == "AdamW" else torch.optim.Adam
    return optimizer(adapter_parameters(parts.unet), lr=settings["learning_rate"])


def train_adapter(
    parts: Components,
    dataset: NEUManifestDataset,
    settings: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    update: int = 0,
    epoch: int = 0,
    next_batch: int = 0,
    on_update: Callable[[int, int, int, float, bool], None],
) -> None:
    """Run optimizer updates, reporting an exact batch cursor at checkpoint boundaries."""
    if (
        not isinstance(dataset, NEUManifestDataset)
        or dataset.split != "train"
        or (dataset.target != "diffusion" or not len(dataset))
    ):
        raise ValueError("LoRA requires a nonempty frozen real train dataset for diffusion")
    steps = settings["max_steps"]
    epochs = settings["epochs"]
    interval = settings.get("checkpoint_interval", 250)
    accumulation = settings["gradient_accumulation"]
    batch_count = (len(dataset) + settings["batch_size"] - 1) // settings["batch_size"]
    updates_per_epoch = (batch_count + accumulation - 1) // accumulation
    total = steps if steps is not None else epochs * updates_per_epoch
    if not 0 <= update < total or not 0 <= next_batch < batch_count or epoch < 0:
        raise ValueError("Checkpoint cursor is outside the configured training budget")
    embeddings = encode_prompts(parts, device)
    precision = settings["precision"]
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    while update < total:
        loader = make_loader(
            dataset, batch_size=settings["batch_size"], seed=settings["seed"] + epoch
        )
        iterator = iter(loader)
        rng = torch.get_rng_state()
        for _ in islice(iterator, next_batch):
            pass
        torch.set_rng_state(rng)
        while next_batch < batch_count and update < total:
            batches = list(islice(iterator, accumulation))
            examples = sum(len(batch["label"]) for batch in batches)
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for batch in batches:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=precision != "fp32"):
                    loss = noise_prediction_loss(parts, batch["image"], batch["label"], embeddings)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite LoRA loss. Check images and try a safer precision")
                weight = len(batch["label"]) / examples
                scaler.scale(loss * weight).backward()
                mean_loss += loss.detach().item() * weight
            scale = scaler.get_scale()
            if not scaler.is_enabled():
                torch.nn.utils.clip_grad_norm_(
                    adapter_parameters(parts.unet),
                    float("inf"),
                    error_if_nonfinite=True,
                )
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:
                raise ValueError("FP16 gradient overflow. Retry with bf16 or fp32")
            update += 1
            next_batch += len(batches)
            finished_epoch = next_batch == batch_count
            cursor = (epoch + 1, 0) if finished_epoch else (epoch, next_batch)
            on_update(update, *cursor, mean_loss, update % interval == 0 or update == total)
        epoch += 1
        next_batch = 0
