# Real-only ResNet-18 baseline

Use the [environment guide](../planning/setup.md) to verify PyTorch and CUDA
on the university PC. Obtain permission for NEU and peer approval of its
frozen split before training. Start with a copy of
[baseline.yaml](../../configs/baseline.yaml), review the actual device and
training budget, set `status: approved`, and set the data and manifest paths
relative to the workspace. Keep `split-metadata.json` next to the manifest.

Run from the repository root with the manifest fingerprint from the peer
review and an approved local torchvision ResNet-18 ImageNet V1 state dict:

```powershell
python scripts/train_real_baseline.py configs/approved-baseline.yaml --seed 42 --approved-manifest-sha256 <reviewed-sha256> --local-weights <approved-local-file>
```

The command uses only the train and validation partitions. Training samples
with replacement to balance exposure across the six classes. AdamW and cosine
decay follow the shared configuration. The budget may be expressed as epochs
for a CPU smoke run or as optimizer updates for approved comparisons.
Validation runs at the configured update interval and at the end. The earliest
checkpoint with the highest validation macro F1 is kept. The held-out test set
has no scoring path in this command.

Each run gets its own output directory. `run.json` records losses, validation
metrics, update and exposure counts, input and checkpoint hashes, device and
software versions. `best.pt` contains the model state and can be opened with
`classification.baseline.load_checkpoint`. This is a model checkpoint for
inference, without optimizer state for resuming training.

The CPU integration suite uses synthetic images and random initialization.
University device verification, approved weight acquisition and the actual
NEU baseline are pending. Synthetic smoke scores are only test evidence.
