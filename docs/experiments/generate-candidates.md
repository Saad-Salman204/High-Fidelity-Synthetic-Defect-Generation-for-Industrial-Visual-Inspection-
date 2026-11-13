# Generate review-pending defect candidates

Use the GPU setup and [LoRA pilot guide](lora-training.md) first. Review the
finished adapter checkpoint from the pilot or bounded training run. Use its
approved local SD v1.5 snapshot and unchanged frozen source manifest.
No network model lookup is used. The sampler supports the local safetensors
variant recorded at training time.

In the reviewed shared YAML, set `status: approved` and configure
`generation.seed`, `device`, `precision`, `batch_size`, `inference_steps`,
`guidance_scale`, and `candidates_per_class`. The baseline requests 166
candidates per class, 996 total, within the cap of 1,000. The accepted
target of 600 is a review goal. Size comes from
`preprocessing.diffusion_resolution`. The six prompts come from the same
catalog used by the trainer.

From the repository root:

```powershell
python scripts/generate_defects.py configs/approved-generation.yaml --model-dir "models/sd-v1-5" --adapter-dir "artifacts/diffusion/RUN/step-001000" --approved-adapter-sha256 "<reviewed-checkpoint.json-hash>" --approved-manifest-sha256 "<reviewed-real-manifest-hash>" --output artifacts/generation/neu-v1
```

Get the checkpoint metadata digest with
`(Get-FileHash -Algorithm SHA256 "path/to/checkpoint.json").Hash.ToLowerInvariant()`.
The CLI checks the adapter hash, pinned revision, training source and base
component fingerprints against that metadata. Keep the model and adapter in
ignored directories or outside the repository. The output must be ignored
by Git.

DDIM uses a separate seeded CPU generator for each image and a fixed batch
plan. The plan records the code commit, configuration, package versions and
device. Reproducibility is scoped to matching software, hardware and batch
settings. If deterministic operations are unavailable, the run can fail
with an actionable runtime error. Full-sized CUDA inference is pending.

Repeat the command with `--resume` after an interruption. The plan and all
committed batch records and image hashes must match. If a process was killed
and left `.generation.lock`, confirm that it has stopped before removing
the lock and resuming. Incomplete temporary batches are ignored. A completed
manifest is never overwritten.

`candidates.jsonl` holds image IDs, paths relative to the output directory,
class IDs and names, prompts, seeds, dimensions, sampling parameters, base
revision, adapter and source fingerprints, and output SHA-256. All rows are
`pending` with an empty reviewer field. #12 must verify defect labels,
quality and near-duplicates against the permitted sources. Preserve the
candidate ledger. Write a separate accepted-only export with reviewer IDs
and the original provenance fields for #14. Use the generation output
directory as the image root when consuming that export.

Use the [synthetic review guide](synthetic-review.md) to run the CPU audit and record both reviews.
