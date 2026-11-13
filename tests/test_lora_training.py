"""Exercise real Diffusers and PEFT components at tiny CPU-only sizes."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import get_peft_model_state_dict
from transformers import CLIPTextConfig, CLIPTextModel

from industrial_defect_diffusion.data.preprocessing import NEUManifestDataset, read_frozen_manifest
from industrial_defect_diffusion.diffusion.checkpoints import resume_checkpoint, save_checkpoint
from industrial_defect_diffusion.diffusion.training import (
    PROMPTS,
    Components,
    encode_prompts,
    make_optimizer,
    noise_prediction_loss,
    prepare_adapter,
    train_adapter,
    training_device,
)
from industrial_defect_diffusion.utils.experiment_config import load_config

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


class TinyTokenizer:
    model_max_length = 8

    def __call__(self, prompts, **_kwargs):
        assert tuple(prompts) == PROMPTS and len(set(prompts)) == 6
        return SimpleNamespace(input_ids=torch.arange(6)[:, None].repeat(1, 8))


def components():
    torch.manual_seed(8)
    return Components(
        unet=UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D",),
            up_block_types=("CrossAttnUpBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            cross_attention_dim=8,
            attention_head_dim=2,
            norm_num_groups=4,
        ),
        vae=AutoencoderKL(
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=4,
        ),
        text_encoder=CLIPTextModel(
            CLIPTextConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=8,
            )
        ),
        tokenizer=TinyTokenizer(),
        scheduler=DDPMScheduler(num_train_timesteps=10),
    )


@pytest.fixture
def settings():
    config = load_config(ROOT / "configs/cpu-smoke.yaml")["diffusion"]
    return {**config, "max_steps": 2, "batch_size": 2, "gradient_accumulation": 2}


def test_loss_shapes_and_adapter_only_gradients(settings):
    parts = components()
    prepare_adapter(parts, {**settings, "gradient_checkpointing": True}, CPU)
    embeddings = encode_prompts(parts, CPU)
    assert embeddings.shape == (6, 8, 8)
    shapes = []
    handle = parts.unet.register_forward_pre_hook(
        lambda _module, inputs: shapes.append(tuple(item.shape for item in inputs))
    )
    loss = noise_prediction_loss(parts, torch.zeros(2, 3, 8, 8), torch.tensor([0, 5]), embeddings)
    handle.remove()
    assert shapes == [(torch.Size((2, 4, 8, 8)), torch.Size((2,)), torch.Size((2, 8, 8)))]
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
    for component in (parts.vae, parts.text_encoder):
        assert all(not p.requires_grad and p.grad is None for p in component.parameters())
    assert all(p.grad is None for p in parts.unet.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parts.unet.parameters())
    with pytest.raises(ValueError, match="valid NEU class"):
        noise_prediction_loss(parts, torch.zeros(1, 3, 8, 8), torch.tensor([6]), embeddings)
    with pytest.raises(ValueError, match="NCHW"):
        noise_prediction_loss(parts, torch.zeros(1, 1, 8, 8), torch.tensor([0]), embeddings)


def test_resume_restores_optimizer_rng_and_batch_cursor(settings, frozen_images, tmp_path):
    root, manifest, meta = frozen_images
    rows, _ = read_frozen_manifest(manifest, meta)
    for row in rows:
        if row["split"] != "train":
            (root / row["relative_path"]).unlink()
    dataset = NEUManifestDataset(root, rows, split="train", target="diffusion", resolution=8)
    checkpoint = tmp_path / "checkpoint"
    provenance = {"config": settings, "manifest": "synthetic"}

    def setup():
        parts = components()
        prepare_adapter(parts, settings, CPU)
        return parts, make_optimizer(parts, settings), torch.amp.GradScaler("cuda", enabled=False)

    first, optimizer, scaler = setup()
    uninterrupted = []

    def capture(update, epoch, next_batch, loss, save):
        uninterrupted.append(loss)
        if update == 1:
            assert save and (epoch, next_batch) == (0, 2)
            save_checkpoint(
                checkpoint, first, optimizer, scaler, CPU, provenance, (update, epoch, next_batch)
            )

    train_adapter(first, dataset, settings, CPU, optimizer, scaler, on_update=capture)
    resumed, optimizer, scaler = setup()
    with pytest.raises(ValueError, match="provenance"):
        resume_checkpoint(checkpoint, resumed, optimizer, scaler, CPU, {})
    update, epoch, next_batch = resume_checkpoint(
        checkpoint,
        resumed,
        optimizer,
        scaler,
        CPU,
        provenance,
    )
    losses = []
    train_adapter(
        resumed,
        dataset,
        settings,
        CPU,
        optimizer,
        scaler,
        update=update,
        epoch=epoch,
        next_batch=next_batch,
        on_update=lambda _u, _e, _b, loss, _save: losses.append(loss),
    )
    assert losses == uninterrupted[1:]
    expected = get_peft_model_state_dict(first.unet)
    actual = get_peft_model_state_dict(resumed.unet)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    with (checkpoint / "adapter.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Adapter SHA-256"):
        resume_checkpoint(checkpoint, resumed, optimizer, scaler, CPU, provenance)


def test_train_guard_and_device_errors(settings, frozen_images, monkeypatch):
    root, manifest, meta = frozen_images
    rows, _ = read_frozen_manifest(manifest, meta)
    wrong = NEUManifestDataset(root, rows, split="val", target="diffusion", resolution=8)
    parts = components()
    prepare_adapter(parts, settings, CPU)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(ValueError, match="frozen real train"):
        train_adapter(
            parts,
            wrong,
            settings,
            CPU,
            make_optimizer(parts, settings),
            scaler,
            on_update=lambda *args: None,
        )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        training_device("cuda", "fp16")
    with pytest.raises(ValueError, match="CPU training requires fp32"):
        training_device("auto", "fp16")


def test_nonfinite_loss_stops_before_update(settings, frozen_images, monkeypatch):
    root, manifest, meta = frozen_images
    rows, _ = read_frozen_manifest(manifest, meta)
    dataset = NEUManifestDataset(root, rows, split="train", target="diffusion", resolution=8)
    parts = components()
    prepare_adapter(parts, settings, CPU)
    optimizer = make_optimizer(parts, settings)
    monkeypatch.setattr(
        "industrial_defect_diffusion.diffusion.training.noise_prediction_loss",
        lambda *_args: torch.tensor(float("nan")),
    )
    with pytest.raises(ValueError, match="Non-finite LoRA loss"):
        train_adapter(
            parts,
            dataset,
            settings,
            CPU,
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            on_update=lambda *args: None,
        )
    assert not optimizer.state


def test_local_snapshot_loading_and_missing_files(tmp_path):
    import json

    from industrial_defect_diffusion.diffusion.local_model import (
        load_local_components,
        snapshot_fingerprints,
    )

    parts = components()
    with pytest.raises(ValueError, match="missing unet"):
        snapshot_fingerprints(tmp_path, None)
    for name in ("unet", "vae", "text_encoder", "scheduler"):
        getattr(parts, name).save_pretrained(tmp_path / name)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.json").write_text(
        json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1}),
        encoding="utf-8",
    )
    (tokenizer / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 8}),
        encoding="utf-8",
    )
    fingerprints = snapshot_fingerprints(tmp_path, None)
    assert "unet/diffusion_pytorch_model.safetensors" in fingerprints
    loaded = load_local_components(tmp_path, None)
    assert loaded.unet.config.in_channels == 4
    assert encode_prompts(loaded, CPU).shape == (6, 8, 8)
    with pytest.raises(ValueError, match="selected variant"):
        snapshot_fingerprints(tmp_path, "fp16")
