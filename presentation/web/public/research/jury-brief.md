# Jury brief

## Study in one minute

We fine-tuned Stable Diffusion 1.5 with LoRA on the frozen real training partition, generated defect candidates, filtered them with a documented automated audit policy, and compared three ResNet-18 data-source arms. All arms use the same real validation/test partitions and three seeds.

## Strongest measured evidence

- Across matched seeds, the real + synthetic arm had lower accuracy than real-only by -0.0049 on average (difference in 0–1 score units).
- Across matched seeds, the real + synthetic arm had lower macro f1 than real-only by -0.0049 on average (difference in 0–1 score units).
- The largest mean class-level F1 increase for real + synthetic versus real-only was rolled_in_scale (+0.0000); the largest decrease was inclusion (-0.0110).

## Likely jury questions and accurate answers

- **Were all selected synthetic images approved by a person?** No. The 783-image pool was selected by automated eligibility rules. It was not individually human-approved.
- **Were near-duplicates removed?** No. Eligible candidates with near-match relationships were retained as the documented policy specified; they are disclosed in selection provenance.
- **Did the test set guide model selection?** The training implementation selected checkpoints from real validation macro F1. The evaluation stage used the held-out test split for final scoring only.
- **Does a gain prove statistical significance?** No. These are descriptive results over three seeds; no significance test was performed.
- **Was the ResNet backbone frozen?** No. The implementation optimizes the model parameter set after replacing the final classifier layer, so the full backbone is fine-tuned.
- **Does classifier performance prove image realism?** No. It measures transfer to this held-out real split under this setup, not industrial acceptance or generalization to new sites.

## Limitations to state openly

One frozen split, three seeds, an automated but not human-approved synthetic pool, and no external industrial test set. The earlier visual review sampled 72 candidates and cannot be generalized to the complete pool.
