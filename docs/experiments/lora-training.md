# Train the SD v1 UNet LoRA adapter

Resume restores the adapter, optimizer, gradient scaler, CPU/CUDA RNG states, and batch cursor. A resume may increase only `diffusion.max_steps` above the saved budget; every other provenance/configuration field must match. The current diffusion training loop has no learning-rate scheduler, so there is no scheduler state to restore. The budget is an absolute global update target: resuming at update 100 with `max_steps: 1000` stops at update 1,000.

Issue #9 implements the training path. The university GPU pilot and real NEU
verification belong to #10. The CPU suite uses tiny randomly initialized
Diffusers and PEFT models.

1. Verify CUDA on the university PC using the [environment guide](../planning/setup.md).
   Install the tested LoRA packages with `python -m pip install -e ".[audit,lora]"`
   after installing the compatible PyTorch and torchvision build.
2. Obtain dataset permission and peer approval of the frozen real split.
   Record its manifest SHA-256. Keep `split-metadata.json` next to the CSV.
3. Obtain an approved local Diffusers snapshot from the pinned revision in the
   [model decision](../research/diffusion-selection.md). The snapshot needs
   `unet`, `vae`, `text_encoder`, `tokenizer`, and `scheduler` folders.
   Keep these outside Git. The loader uses local safetensors only. Add
   `--variant fp16` only if all three model components have that file variant.
   Precision controls training dtype independently of the on-disk variant.
4. Copy `configs/baseline.yaml` to a local pilot config, set `status: approved`
   after review, and fill in workspace-relative data, manifest, and artifact
   paths. Set `diffusion.max_steps: 100` and `diffusion.epochs: null` for the
   #10 pilot. Choose device and precision after checking the machine.
5. Run from the repository root:

```powershell
python scripts/train_lora.py configs/approved-pilot.yaml --model-dir "path/to/local-snapshot" --approved-manifest-sha256 "<reviewed-manifest-hash>"
```

The loader opens only real training images, using the existing 512×512
grayscale-to-RGB transform by default. The six prompts are encoded once.
For each image the VAE produces scaled latents, the DDPM scheduler adds noise
at a sampled timestep, and the UNet learns to predict that noise with mean
squared error. The VAE, text encoder and base UNet are frozen. PEFT updates
only the attention projection adapters. Batch size, learning rate, rank,
alpha, optimizer, seed, accumulation, precision, and checkpoint interval
come from the shared diffusion configuration. Gradient checkpointing is
enabled by default.

The run directory holds resolved settings and the #8 JSON run record. It
records losses, device, package versions, model and data fingerprints, and
output hashes. Checkpoint directories hold `adapter.safetensors` in PEFT
UNet format, `checkpoint.json` with provenance and progress, and
`training-state.pt` with optimizer, scaler and PyTorch RNG state.
`load_adapter` loads the adapter into an already prepared compatible UNet.

To resume an interrupted run, repeat the command with
`--resume "path/to/step-000025"` pointing to a completed intermediate
checkpoint from the same configuration, source manifest, local weights,
code commit and environment. Resume writes to a new run directory. Only
use trusted local optimizer checkpoints. The CPU test verifies an identical
next update with a saved cursor. CUDA kernels may introduce differences.
Full-size GPU memory, mixed precision, throughput and image quality still
need the #10 pilot.
