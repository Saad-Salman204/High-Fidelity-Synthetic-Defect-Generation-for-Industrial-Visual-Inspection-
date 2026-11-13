"""Load only the local Stable Diffusion components needed by LoRA training."""

from hashlib import file_digest
from pathlib import Path

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .training import Components

FOLDERS = ("unet", "vae", "text_encoder", "tokenizer", "scheduler")
WEIGHT_NAMES = {
    "unet": "diffusion_pytorch_model",
    "vae": "diffusion_pytorch_model",
    "text_encoder": "model",
}


def file_sha256(path: Path) -> str:
    """Hash a local file without loading it all into memory."""
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def snapshot_fingerprints(root: Path, variant: str | None) -> dict[str, str]:
    """Identify selected component files before loading the approved local snapshot."""
    files = {}
    suffix = f".{variant}.safetensors" if variant else ".safetensors"
    for folder in FOLDERS:
        directory = root / folder
        if not directory.is_dir():
            raise ValueError(f"Local model snapshot is missing {folder}/")
        if folder in WEIGHT_NAMES:
            weight = directory / f"{WEIGHT_NAMES[folder]}{suffix}"
            if not weight.is_file():
                raise ValueError(f"Missing local safetensors for {folder}/ and selected variant")
            files[weight.relative_to(root).as_posix()] = file_sha256(weight)
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in (".json", ".txt"):
                files[path.relative_to(root).as_posix()] = file_sha256(path)
    return files


def load_local_components(root: Path, variant: str | None) -> Components:
    """Load the selected local safetensors and configs without any Hub lookup."""
    options = {"local_files_only": True, "use_safetensors": True, "variant": variant}
    return Components(
        unet=UNet2DConditionModel.from_pretrained(str(root / "unet"), **options),
        vae=AutoencoderKL.from_pretrained(str(root / "vae"), **options),
        text_encoder=CLIPTextModel.from_pretrained(str(root / "text_encoder"), **options),
        tokenizer=CLIPTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True),
        scheduler=DDPMScheduler.from_pretrained(str(root / "scheduler"), local_files_only=True),
    )
