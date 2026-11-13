# Environment and setup plan

Foundation targets Python 3.11. Historical foundation checks are recorded in [verification](../../reports/foundation-verification.md). This page records the separate ML environment check for #2.

## Foundation

Create a Python 3.11 virtual environment and install `.[dev]` as shown in [README](../../README.md). The current development pins and build backend are in [pyproject.toml](../../pyproject.toml). GitHub Actions installs the foundation only. The ML stack below uses a separate environment.

Run foundation validator and Ruff commands from CONTRIBUTING. Hooks are optional and use the active environment. The foundation validator checks local Markdown file targets (not remote URLs or heading anchors), JSON/YAML/TOML syntax, issue dependency DAG, planned configuration invariants and package scaffolding. This is repository verification, not an ML test suite. Markdown style linting, type checking, pytest and vulnerability scanning are deferred until their applicable configuration/logic exists; I19 enables meaningful tests, I02 reviews dependency security.

## CPU ML environment (#2)

Run from the repository root with Python 3.11 on Windows x64. The CPU package index is the one listed in the [official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/).

```powershell
python -m venv .cache/issue2-cpu
$mlPython = '.\.cache\issue2-cpu\Scripts\python.exe'
& $mlPython -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
& $mlPython -m pip install -r requirements/ml-cpu-win-py311.txt
& $mlPython -m pip check
& $mlPython scripts/check_ml_environment.py --device cpu
```

The check uses a small tensor and a randomly initialized ResNet-18. It checks that the libraries work together well enough to return six finite scores. Package wheels are downloaded during installation. The check itself loads no data or pretrained weights. Training and performance are outside this check.

| Local verification | Versions |
| --- | --- |
| Windows x64 CPU, Python 3.11.9 | PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu |
| Companion packages | Diffusers 0.43.0, PEFT 0.21.3, Accelerate 1.15.0, Transformers 5.22.2 |
| Separate fresh foundation install | Ruff 0.17.3, PyYAML 6.0.3, pre-commit 4.0.1, setuptools build backend 84.2.1 |

The [Windows CPU dependency snapshot](../../requirements/ml-cpu-win-py311.txt) records the full installed set. Keep the CPU wheels from the official index when recreating it.

## University GPU verification

**Pending on the university PC.** Record the operating system, Python version, GPU model, NVIDIA driver, total/free VRAM, and actual booking or session limit. In PowerShell, run:

```powershell
python --version
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

1. Check the driver against [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) and select a supported CUDA wheel using the [official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/). Match PyTorch and torchvision versions and the wheel's CUDA build. The CUDA number shown by `nvidia-smi` describes driver support. Record the runtime used by PyTorch separately from that number.
2. Create a fresh Python 3.11 GPU environment outside the CPU environment. Use the official command for the chosen wheel index. For example, if the driver, GPU and OS support CUDA 12.6, use `python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126` inside that environment. Use the tested companion library versions recorded in the CPU section as initial pins. The Windows CPU lock can contain platform-specific packages and should not be applied wholesale to the GPU environment.
3. In the GPU environment, run `python -m pip check` and `python scripts/check_ml_environment.py --device cuda`. Confirm CUDA is available, the expected GPU and VRAM are reported, and the forward pass succeeds. The command exits with an error if CUDA is unavailable.
4. Save the output, `python -m pip freeze --all`, and the confirmed GPU allocation in the issue's verification record. Investigate a driver mismatch or unsupported GPU with the university administrator before selecting another wheel. A passing small-tensor check still leaves full training memory and runtime to the later GPU pilot.

## Local artifact storage

Use ignored `data/local`, `artifacts`, `.cache` and optional `.env` paths. Credentials belong in OS stores/environment; `.env.example` has no secrets. Copy approved artifacts and manifest/checksum records to team storage before ephemeral GPU sessions expire. No cloud deployment or paid experiment service required.
