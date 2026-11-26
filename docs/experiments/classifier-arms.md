# Three classifier training arms

After freezing the completed comparison, use the [test evaluation guide](classifier-evaluation.md).

Use the same reviewed configuration, seed, frozen real manifest and approved
local ResNet-18 weights for all arms. From the repository root:

```powershell
python scripts/train_real_baseline.py configs/approved-baseline.yaml --seed 42 --arm all --approved-manifest-sha256 "<real-manifest-hash>" --local-weights "<approved-local-weights>" --synthetic-root "<local-image-directory>" --synthetic-manifest "<workspace-relative-accepted.jsonl>" --approved-synthetic-manifest-sha256 "<accepted-jsonl-hash>"
```

Use `--arm real_only`, `--arm synthetic_only` or `--arm real_plus_synthetic` to
run one arm. The original real-only command still works. Comparison runs use
the same `classification.max_steps` and batch size. Each arm starts a new model
and optimizer from the same seed and initialization. Every full mixed batch
has the configured `mixed_synthetic_fraction` of synthetic images, default
50%. Choose a fraction that gives a whole number of images from each source.
Within each source, shuffled cycles expose each class once per six draws.
Images are sampled with replacement. All arms use the same real validation
rows and choose the earliest best validation macro F1 checkpoint.

## Approved synthetic export

After #12 review, export only accepted records from the generation ledger.
Keep the full ledger separately. Each JSONL line needs:

- `schema_version: 1`, unique `image_id`, relative `relative_path`, and image `sha256`.
- Canonical `class_id` and matching `class_name`, with all six classes in the pool.
- `quality_status: "accepted"`, reviewer ID in `reviewer_id`, and no rejection reason.
- `run_id`, `prompt`, `seed`, `scheduler`, `inference_steps`, `width`, and `height`.
- The configured `base_revision`, actual `adapter_sha256`, and
  `real_manifest_sha256` from the source training run.
- `parent_ids: []` and `mask_id: null` for text-to-image.

The loader checks the submitted hashes and rejects real-image ID or byte
overlap, including held-out images, using manifest metadata. It never opens
test images. It also rejects path escapes, duplicate synthetic entries and
nonempty ancestry. Peer review of the export must establish the provenance
from the actual run records and check image quality and near-duplicates.
The loader checks the declared provenance and image bytes.

Each arm writes its own `best.pt`, `training.jsonl` and `run.json`. The record contains validation
IDs and scores, loss history, manifest and checkpoint hashes, and exposure
counts overall and by source and class. Repeated exposures count draws beyond
the number of unique images seen. The CPU tests use synthetic fixtures.
Real generated images and university GPU comparison results remain pending.

## Automated audit-policy selection

The separate automated policy workflow and its preparation command are
documented in [automated-synthetic-selection.md](automated-synthetic-selection.md).
It does not create `accepted` records or reviewer fields. For classifier runs,
use `--automated-selection-manifest` and `--automated-selection-sha256`; the
human-reviewed `--synthetic-manifest` path remains unchanged.
