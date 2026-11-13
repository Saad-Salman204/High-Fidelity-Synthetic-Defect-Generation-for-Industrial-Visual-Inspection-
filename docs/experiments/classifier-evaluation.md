# Evaluate frozen classifier runs

Issue #16 reads the saved #14 training records after the #15 comparison has
finished. Review the real split, training settings, and checkpoint choices
before opening any test scores. Use the exact approved config used for
training. Every evaluated arm must reference the same real manifest, class
mapping, training commit and initialization. The synthetic arms must use
the same approved synthetic export.

Write a local `selection.json` listing the records you chose before
evaluation. Paths are relative to the repository root. Record the SHA-256
of each exact `run.json` file. The following illustrates the format:

```json
{
  "schema_version": 1,
  "manifest_sha256": "<reviewed-real-manifest-sha256>",
  "runs": [
    {"arm": "real_only", "seed": 42, "path": "artifacts/real/run.json", "sha256": "<record-hash>"},
    {"arm": "synthetic_only", "seed": 42, "path": "artifacts/synthetic/run.json", "sha256": "<record-hash>"},
    {"arm": "real_plus_synthetic", "seed": 42, "path": "artifacts/mixed/run.json", "sha256": "<record-hash>"}
  ]
}
```

List every configured arm and seed with an available run record, including
failed attempts. Omitted pairs are reported as missing. Obtain peer review
of the selection and its SHA-256. On Windows, use
`(Get-FileHash -Algorithm SHA256 selection.json).Hash.ToLowerInvariant()`.
From the repository root:

```powershell
python scripts/evaluate_classifier.py configs/approved-baseline.yaml --selection selection.json --approved-selection-sha256 "<selection-file-hash>" --approved-manifest-sha256 "<real-manifest-hash>" --output artifacts/evaluation/frozen-comparison
```

The command uses the existing classifier transform in test mode. No
training or validation pixels are loaded. It checks all referenced hashes
and scores each test image once per completed checkpoint. Reusing an
output directory is an error. Exit 0 means all expected runs completed,
1 means at least one run is missing or failed, and 2 means an input or
artifact error. An invalid completed checkpoint stops evaluation.

Each arm and seed gets predictions, per-class counts and scores, full
metrics JSON, and an SVG confusion matrix. Rows are true classes and
columns are predictions. Each cell shows its count and share of its row.
Precision and recall use zero when the denominator is zero. The report
flags absent true and predicted classes. Macro scores give equal weight
to all six classes.

`comparison.csv`, `comparison.md`, and `performance.svg` summarize the
results. Matched-seed averages and paired macro F1 differences use only
seeds with all three completed arms. Sample SD requires at least two
matched seeds and describes run-to-run variability. `evaluation.json`
records input fingerprints, configuration, runtime, and per-run provenance.
The CPU fixtures verify the pipeline. Actual NEU results require approved
real data, the completed #15 checkpoints and a sealed run selection.
