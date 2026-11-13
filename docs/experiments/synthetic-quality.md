# Exploratory generation quality (#13)

Run this after generating candidates and finishing the [#12 review](synthetic-review.md).
Use the original candidate directory, final review export, original audit, and frozen split.
The command verifies the artifacts and replays the existing review to check that it
reproduces the saved export. Only real training pixels are opened, and only accepted
synthetic images contribute to KID. The assessment leaves review decisions untouched.

From the repository root, use the existing #11 CPU environment. For KID,
install the pinned extractor with `python -m pip install -e '.[quality]'`.
Use the separately provisioned torch-fidelity Inception-v3 compatibility
weights and the independently verified SHA-256 for that file.

```powershell
python scripts/assess_synthetic_quality.py `
  --generation artifacts/generation/neu-v1 --real-root data/neu `
  --manifest artifacts/neu-split-v1/manifest.csv --real-audit artifacts/neu-audit.json `
  --approved-manifest-sha256 '<reviewed-real-manifest-hash>' `
  --review artifacts/review/final --output artifacts/quality/neu-v1 `
  --weights models/inception-v3-compat.pth --approved-weights-sha256 '<verified-weight-hash>'
```

The CLI uses local weights only. Omit the last line and the preceding
continuation character for a qualitative-only run. Use a fresh ignored output
directory for each run. The defaults are seed 42, 100 draws, 40 images per
class from each pool and 200 per pool overall. Each draw samples without
replacement. If either pool is too small for the requested subset, that
score is unavailable. `--per-class-size`, `--overall-size`, `--subsets`, and
`--seed` are recorded overrides for separately disclosed exploratory runs.

`quality.json` records source hashes, sample counts, preprocessing, the
extractor and weight hash, mean KID, subset standard deviation, and any
unavailable reasons. `quality.md` summarizes the scores and rejection reasons.
Each class grid shows seeded real training, accepted synthetic, and eligible
rejected synthetic examples, plus their closest real training matches from
the review audit. Review notes are carried over from the decision ledgers.
Use the grids to discuss defect visibility, texture, artifacts, and possible
copies in the final report.

This is an in-sample comparison using natural-image features. Negative
estimates are possible, subset spread is descriptive, and the pooled score
depends on class balance. KID cannot establish defect fidelity, rule out
memorization, or decide which images belong in classifier training.
Real results remain pending until reviewed samples and verified feature
weights are available.
