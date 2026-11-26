# Executive summary

All 9 selected classifier runs were evaluated on the same 270 real test images; evaluation status: **completed**.

## Results

| Arm | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Real-only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Synthetic-only | 0.3407 ± 0.0744 | 0.3202 ± 0.0372 | 0.3407 ± 0.0744 | 0.2794 ± 0.0736 |
| Real + synthetic | 0.9951 ± 0.0021 | 0.9952 ± 0.0020 | 0.9951 ± 0.0021 | 0.9951 ± 0.0021 |

## Main measured comparisons

- Across matched seeds, the real + synthetic arm had lower accuracy than real-only by -0.0049 on average (difference in 0–1 score units).
- Across matched seeds, the real + synthetic arm had lower macro f1 than real-only by -0.0049 on average (difference in 0–1 score units).
- The largest mean class-level F1 increase for real + synthetic versus real-only was rolled_in_scale (+0.0000); the largest decrease was inclusion (-0.0110).

## Interpretation limits

The synthetic pool was selected by automated blocker/warning filtering and was not individually human-approved. Near-match candidates were retained under that policy. Results use one fixed real split and three seeds; no significance test was performed.
