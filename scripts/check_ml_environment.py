"""Check the optional ML environment with a small tensor and no downloads."""

import argparse
import platform
from importlib.metadata import version

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler
from peft import LoraConfig
from torchvision.models import resnet18
from transformers import CLIPTextConfig


def main() -> None:
    """Report the installed stack and verify a forward pass on the requested device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    print(f"{platform.platform()} | Python {platform.python_version()}")
    for package in ("torch", "torchvision", "diffusers", "peft", "accelerate", "transformers"):
        print(f"{package}=={version(package)}")
    print(f"PyTorch CUDA build: {torch.version.cuda}; CUDA available: {torch.cuda.is_available()}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.exit(1, "CUDA requested but unavailable. Check the driver and PyTorch wheel.\n")

    device = torch.device(args.device)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(f"GPU: {properties.name}; compute capability: {properties.major}.{properties.minor}")
        print(f"VRAM free/total: {free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB")

    Accelerator(cpu=device.type == "cpu")
    DDPMScheduler()
    LoraConfig()
    CLIPTextConfig()
    model = resnet18(weights=None, num_classes=6).to(device).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 32, 32, device=device))
    if output.shape != (1, 6) or not torch.isfinite(output).all():
        raise RuntimeError("Expected six finite scores from the sample input.")
    print(f"{device.type} smoke check passed. No data or pretrained weights were loaded.")


if __name__ == "__main__":
    main()
