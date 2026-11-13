# Review synthetic candidates (#12)

Run on CPU with the same Python environment as #11. Keep the original
generation folder, including `plan.json` and `candidates.jsonl`. Use the
frozen real CSV, its adjacent split metadata, and the original #3 audit JSON.
The audit must match the fingerprint in split metadata and contain valid
perceptual fingerprints for every real split row. If those are missing,
stop and resolve the source audit before reviewing candidates.

Choose and record the perceptual distance threshold before review. Default
is 5 bits, using #3's 64-bit difference hash. Steel textures can look similar
at this scale, so a flag is evidence for review. Exact real matches and
potential held-out matches are blocked. The process uses the pre-freeze
fingerprints and opens no real images. Held-out matches provide no held-out
IDs, paths, pixels, or distances. Keep the reference audit with the data
custodian. Run one frozen screen and do not use held-out flags to tune
prompts, models, thresholds, or resampling.

From the repository root in PowerShell:

```powershell
$inputs = @('--generation', 'artifacts/generation/neu-v1', '--manifest', 'artifacts/neu-split-v1/manifest.csv', '--real-audit', 'artifacts/neu-audit.json', '--approved-manifest-sha256', '<reviewed-real-manifest-hash>')
python scripts/review_synthetic.py audit @inputs --output artifacts/review/first
python scripts/review_synthetic.py second-review @inputs --primary artifacts/review/first/primary.json --output artifacts/review/second
python scripts/review_synthetic.py export @inputs --primary artifacts/review/first/primary.json --secondary artifacts/review/second/secondary.json --output artifacts/review/final
```

Between commands:

1. Open the class contact sheets. The numbered tiles follow `audit.json`
   findings and the primary review template order. Blocked tiles are blank.
   Examine each eligible candidate for the intended defect, credible steel
   texture, blank regions, obvious artifacts, and near-copies. Use only
   training images for real-image visual comparison. For a training match,
   the report provides the closest training ID and hash distance.
2. Fill every primary decision with `accepted` or `rejected`, reviewer ID,
   and a timezone-aware ISO timestamp, for example `2026-11-13T12:00:00Z`.
   Record a reason for rejection or for accepting a flagged candidate.
   Mark a disputed class label with `disputed: true`. Reject ambiguous
   labels instead of changing them. Blocked candidates cannot be accepted.
   Keep at most one copy of an exact duplicate.
3. A different reviewer fills the second template. It lists up to ten
   seeded accepted samples per class and all disputed labels. Resolve
   disagreements and regenerate the second template in a fresh output
   directory if the primary ledger changes.

The export writes `reviewed.jsonl`, `approved.jsonl`, the original findings,
both decision files, and `review-summary.json` with accepted counts, shortfalls,
rejection reasons, and the approved export SHA-256. Pass the generation
directory as #14's image root and `approved.jsonl` as its manifest. If a class
has no accepted samples, the summary flags it and #14 will reject the pool.
Review outputs must be ignored by Git. Source manifests and images remain
available for audit. Real candidate fidelity and reviewer agreement remain
pending until the university generates images.
