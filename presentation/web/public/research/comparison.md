# Frozen test comparison

| Arm | Seed | Status | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| real_only | 42 | completed | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| synthetic_only | 42 | completed | 0.3074 | 0.2831 | 0.3074 | 0.2489 |
| real_plus_synthetic | 42 | completed | 0.9926 | 0.9929 | 0.9926 | 0.9926 |
| real_only | 43 | completed | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| synthetic_only | 43 | completed | 0.4259 | 0.3576 | 0.4259 | 0.3634 |
| real_plus_synthetic | 43 | completed | 0.9963 | 0.9964 | 0.9963 | 0.9963 |
| real_only | 44 | completed | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| synthetic_only | 44 | completed | 0.2889 | 0.3198 | 0.2889 | 0.2260 |
| real_plus_synthetic | 44 | completed | 0.9963 | 0.9964 | 0.9963 | 0.9963 |

## Matched seed summaries

| Arm | Metric | Mean | Sample SD |
| --- | --- | ---: | ---: |
| real_only | accuracy | 1.0000 | 0.0000 |
| real_only | macro_precision | 1.0000 | 0.0000 |
| real_only | macro_recall | 1.0000 | 0.0000 |
| real_only | macro_f1 | 1.0000 | 0.0000 |
| synthetic_only | accuracy | 0.3407 | 0.0744 |
| synthetic_only | macro_precision | 0.3202 | 0.0372 |
| synthetic_only | macro_recall | 0.3407 | 0.0744 |
| synthetic_only | macro_f1 | 0.2794 | 0.0736 |
| real_plus_synthetic | accuracy | 0.9951 | 0.0021 |
| real_plus_synthetic | macro_precision | 0.9952 | 0.0020 |
| real_plus_synthetic | macro_recall | 0.9951 | 0.0021 |
| real_plus_synthetic | macro_f1 | 0.9951 | 0.0021 |

Only seeds completed in every arm enter aggregate comparisons. Sample SD describes variability across seeds and requires at least two.
Inspect the configuration status in evaluation.json to identify fixture runs.
