# Environment and setup plan

Foundation targets Python 3.11. Actual local verification and hosted CI results are recorded in [verification](../../reports/foundation-verification.md). The proposed ML stack is not yet installed/tested and no training command exists.

## Foundation

Create a virtual environment and install `.[dev]` as shown in [README](../../README.md). Pinned development tools: Ruff 0.8.6, PyYAML 6.0.2, pre-commit 4.0.1; build backend setuptools 75.6.0. These are deliberate reproducible versions, not a claim of latest releases. No runtime dependencies are installed by the empty package. Dependabot proposes updates for peer review.

Run foundation validator and Ruff commands from CONTRIBUTING. Hooks are optional and use the active environment. The foundation validator checks local Markdown file targets (not remote URLs or heading anchors), JSON/YAML/TOML syntax, issue dependency DAG, planned configuration invariants and package scaffolding. This is repository verification, not an ML test suite. Markdown style linting, type checking, pytest and vulnerability scanning are deferred until their applicable configuration/logic exists; I19 enables meaningful tests, I02 reviews dependency security.

## ML environment gate (I02)

Record available GPU/VRAM/session limits first. Select compatible PyTorch/torchvision CUDA wheels from official sources; pin Diffusers, PEFT, Accelerate, Transformers, NumPy, Pillow, pandas, scikit-learn and matplotlib after testing. Add TorchMetrics/torch-fidelity only for quality assessment. Avoid speculative universal CUDA pins and unnecessary bitsandbytes/xformers dependencies.

Record Python/OS/CUDA/driver/GPU, package lock or freeze, base-model revision and artifact hashes. Prove imports and one CPU input/output check, then separately a real 100-step GPU pilot. Never install/download model weights in PR CI. ML dependency resolution and training verification remain implementation tasks.

## Local artifact storage

Use ignored `data/local`, `artifacts`, `.cache` and optional `.env` paths. Credentials belong in OS stores/environment; `.env.example` has no secrets. Copy approved artifacts and manifest/checksum records to team storage before ephemeral GPU sessions expire. No cloud deployment or paid experiment service required.
