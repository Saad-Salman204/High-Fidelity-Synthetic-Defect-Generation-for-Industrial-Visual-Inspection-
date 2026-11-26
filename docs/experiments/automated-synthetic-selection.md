# Automated synthetic selection

This opt-in data-preparation workflow is separate from the human review/export
path in `scripts/review_synthetic.py`. It implements the requested fixed policy:
exclude candidates with any audit blocker or warning; retain candidates with
near-candidate relationships when they have neither. This is an automated
policy selection, not manual visual review or human acceptance. Candidate rows
remain `quality_status: pending` and keep `reviewer_id: null`.

The selector checks that the proposal exactly matches the audit's no-blocker,
no-warning set and that proposal, audit, and generation manifest describe the
same candidate rows. It verifies source image presence, SHA-256, decode, frame
count, dimensions, and class ID/name agreement, then copies unchanged image
bytes into a separate class-organized directory. Integrity failures are listed
with candidate ID, class, and reason in `selection.json`; they are never
silently removed. All six classes must retain at least one candidate. The
selection record contains the raw source audit and generation-manifest hashes,
proposal hash, output-manifest hash, policy, counts, audit-policy exclusions,
integrity exclusions, and the SHA-256 and source path of each selected image.

From the repository root, create a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts/select_synthetic_eligible.py `
  --proposal artifacts/review/audit-v1/eligible-proposal.jsonl `
  --audit artifacts/review/audit-v1/audit.json `
  --generation-manifest artifacts/synthetic/neu-sd15-step-001000/candidates.jsonl `
  --generation-root artifacts/synthetic/neu-sd15-step-001000 `
  --output artifacts/synthetic/automated-selection-audit-v1
```

The output contains `selected.jsonl`, `selection.json`, and exact byte copies
of selected images. The selector does not modify the proposal, audit, candidate
ledger, or source images.

`scripts/train_real_baseline.py` consumes this pool only through the distinct
`--automated-selection-manifest` option and requires the exact
`--automated-selection-sha256`. It verifies the matching `selection.json`,
preserves the pending status, and records the method and source hashes in each
classifier run record. The existing `--synthetic-manifest` path continues to
require human-reviewed accepted records and its separate peer-review hash.

For the approved experiment, all arms use the frozen split and shared real
validation set. Supply `--real-root` when raw NEU-CLS images are outside the
repository; the frozen manifest and its approved hash remain unchanged. Use the
same local ResNet weights, device, seed, budget, and preprocessing for each arm.
