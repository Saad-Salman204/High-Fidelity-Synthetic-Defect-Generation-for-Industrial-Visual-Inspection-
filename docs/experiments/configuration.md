# Experiment settings and run records (#8)

[baseline.yaml](../../configs/baseline.yaml) holds the proposed university GPU
settings, including the SD v1.5 revision from [#6](../research/diffusion-selection.md).
[cpu-smoke.yaml](../../configs/cpu-smoke.yaml) is for checking settings and run-record
plumbing on CPU with synthetic inputs. Neither command below starts a model:

```text
python scripts/check_experiment_config.py configs/cpu-smoke.yaml
python scripts/check_experiment_config.py configs/baseline.yaml
```

Use a fresh environment with `pip install -e ".[dev,audit]"` for the CPU suite.
The CLI prints JSON and returns 2 for malformed input. Stage code should call
`load_config` from `industrial_defect_diffusion.utils.experiment_config` for the same
validation. Unknown keys and duplicate YAML keys are errors. Paths are relative
to the caller's workspace root. Device values are `cpu`, `cuda`, `cuda:<index>` or
`auto`. Hardware availability is checked by the eventual caller.

| Section | Consumer |
| --- | --- |
| `split` | #4 uses the seed and train/validation/test ratios with grouping enabled. |
| `preprocessing` | #5 reads model input sizes and training-only flip probability. |
| `classification` | #7 and #14 share one model, optimizer, batch size, update budget, and seed list across the three arms. |
| `diffusion` | #9 uses the pinned model, LoRA settings, precision, device and budget. |
| `generation` | #11 reads its seed, batch size, sampling steps, guidance and candidate limits. |
| `evaluation` | Evaluation jobs use the configured real manifest, batch size, seed and device. |

Choose one of `max_steps` or `epochs` for a training stage. Approved classifier
comparisons use `max_steps` so different dataset sizes cannot silently change
training budgets. Review proposed settings against the experiment protocol,
real split and GPU results before marking a config `approved`. `smoke` is for
local test inputs. Config validation does not authorize a GPU run or download.

After a run attempt, call `make_run_record` in
`industrial_defect_diffusion.utils.run_records` with the validated config, stage,
actual seed, full Git commit, timezone-aware timestamps and workspace root.
Pass the arm for classification and the actual checkpoint SHA-256 for approved
classification/evaluation. Record package and hardware details using
`environment`, and provide real metrics only if measured. Include relative
artifact paths, optional source/archive and adapter hashes, and a reason for
failed or interrupted runs. The function embeds the resolved config, computes
the manifest and artifact fingerprints from disk, and keeps missing results
as JSON null. `write_run_record` creates an exclusive UTF-8 JSON file.
Use a fresh run directory for each attempt. The helper `seeded_rng` creates an
isolated Python RNG. Later model code must separately seed its own NumPy,
PyTorch and worker RNGs and record device-specific settings.
