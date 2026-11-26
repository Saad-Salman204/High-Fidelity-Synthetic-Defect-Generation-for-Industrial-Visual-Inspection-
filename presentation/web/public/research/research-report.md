# Synthetic defect augmentation for industrial visual inspection

## Abstract

This report evaluates the existing Stable Diffusion 1.5 + LoRA synthetic-defect workflow through three ResNet-18 arms: real-only, synthetic-only, and real + synthetic. Each arm has three seeds and all models are scored on the same frozen real test set of 270 images. The evaluator completed 9 run evaluations with status **completed**. The synthetic pool was selected by the documented automated audit policy; its candidates were not individually human-approved. Measured arm-level results and paired-seed differences are shown below. No significance test was performed.

## Research problem and motivation

Industrial surface-defect datasets are limited and class distributions can constrain supervised model training. This project tests whether generated NEU-CLS-like defect images can provide a useful additional training source for a six-class ResNet-18 classifier. The central comparison is whether mixing the selected synthetic pool with real training images changes performance on held-out real data relative to real-only training.

## Objectives

1. Train and compare the three prespecified data-source arms with matched seeds and a shared training budget.
2. Evaluate every completed run on the same frozen real test partition.
3. Report aggregate and per-class behavior, including degradations and variability across seeds.

## Dataset and frozen split

The real data are the six NEU-CLS classes with canonical IDs from the approved experiment configuration. Split provenance: manifest SHA-256 `b8ab71f7381cac36cd45c85381cb48807ef155c5fa3d66e4545e2eaf51fa49ef`, source fingerprint `bf8e6820092dc973be3ed40ae20003f37bed11e6a04809549838df7cb2d7c288`, seed 42, ratios 70/15/15. The evaluator checked the manifest and metadata before and after inference, and each test image was loaded through the frozen manifest with its recorded file hash.

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| crazing | 210 | 45 | 45 |
| inclusion | 210 | 45 | 45 |
| patches | 210 | 45 | 45 |
| pitted_surface | 210 | 45 | 45 |
| rolled_in_scale | 210 | 45 | 45 |
| scratches | 210 | 45 | 45 |

Total partition sizes are train=1260, validation=270, and test=270.

## Stable Diffusion 1.5 and LoRA methodology

The recorded base model is `stable-diffusion-v1-5/stable-diffusion-v1-5` at revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. The adapter configuration uses rank 4 and alpha 4, with 1000 optimizer updates. Generator training was completed before the classifier comparison; this phase launched no diffusion or classifier training.

## Synthetic generation, audit, and selection

The generation run produced 996 candidates (166 per class) using the saved generation plan. The classifier pool has 783 candidates, selected under `audit-eligible-no-flags-v1`. The policy excludes any candidate with an audit blocker or warning and retains near-match relationships when neither is present. The selection has 186 candidates with near-match links; these were retained, not automatically rejected. Integrity/decode exclusions recorded in the selection report: 0. This selection was automated eligibility filtering, not individual visual acceptance or human approval.

Selection manifest SHA-256: `be0bb91d1a0f6d10f84ab002d6e4039bfa6ee5eedf4826aaebac0568c185343a`. Source audit SHA-256: `07aae97c14ada7f7dd0ae8efd6db3140dd1a785fa6e3f4d6ae3429796f530147`. Source generation-manifest SHA-256: `681221d419977de810a016b82c1c22472866eefaf6e1dc40230096bb748dc515`.

| Class | Selected synthetic candidates |
| --- | ---: |
| crazing | 148 |
| inclusion | 96 |
| patches | 134 |
| pitted_surface | 132 |
| rolled_in_scale | 146 |
| scratches | 127 |

## ResNet-18 architecture and training

Each arm used torchvision ResNet-18 initialized from ImageNet-1K V1 weights. The original 1,000-class final layer was replaced by a six-class linear layer. **The full backbone and replacement classifier were trainable**: model parameters are passed to AdamW without freezing any layers, and the implementation does not set `requires_grad=False`. The initialization file hash recorded by all runs was `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

Training used AdamW, cosine learning-rate scheduling, batch size 24, 1000 optimizer updates, and validation every 100 updates. The real + synthetic arm used a configured synthetic fraction of 0.50. The earliest checkpoint with the best validation macro F1 was saved. Validation—not test—metrics governed checkpoint selection. Training code exposed only real train rows (when applicable) and synthetic train rows to optimizers; validation rows were used for model selection. The test split has no path in the training loader.

## Experimental arms and seed design

The run selection contains all nine configured arm/seed combinations: seeds 42, 43, 44; each record reports 1000 updates and the same frozen configuration hash `8b1cf6d22cb4e5aa8496ef481e3018ae9858d77ae477d2aee8c76a9b59e8efd1`. All arms used the same frozen validation IDs, test manifest, model initialization hash, and training commit. The synthetic arms reference the same selected-manifest hash.

## Evaluation metrics

For each run, the evaluator generated one prediction per frozen test image and computed accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1, support counts, and a 6×6 confusion matrix in canonical class order. Macro scores weight the six classes equally. Aggregates use the three matched seeds; standard deviations are sample SDs.

## Experimental results

### Aggregate scores across seeds

| Arm | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Real-only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Synthetic-only | 0.3407 ± 0.0744 | 0.3202 ± 0.0372 | 0.3407 ± 0.0744 | 0.2794 ± 0.0736 |
| Real + synthetic | 0.9951 ± 0.0021 | 0.9952 ± 0.0020 | 0.9951 ± 0.0021 | 0.9951 ± 0.0021 |

### Per-class F1 across seeds

| Class | Real-only F1 | Synthetic-only F1 | Real + synthetic F1 |
| --- | ---: | ---: | ---: |
| crazing | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9963 ± 0.0063 |
| inclusion | 1.0000 ± 0.0000 | 0.4309 ± 0.0761 | 0.9890 ± 0.0109 |
| patches | 1.0000 ± 0.0000 | 0.1019 ± 0.1764 | 0.9963 ± 0.0065 |
| pitted_surface | 1.0000 ± 0.0000 | 0.0418 ± 0.0369 | 0.9963 ± 0.0063 |
| rolled_in_scale | 1.0000 ± 0.0000 | 0.6228 ± 0.1783 | 1.0000 ± 0.0000 |
| scratches | 1.0000 ± 0.0000 | 0.4794 ± 0.0800 | 0.9924 ± 0.0131 |

### Paired comparisons

- Across matched seeds, the real + synthetic arm had lower accuracy than real-only by -0.0049 on average (difference in 0–1 score units).
- Across matched seeds, the real + synthetic arm had lower macro f1 than real-only by -0.0049 on average (difference in 0–1 score units).
- The largest mean class-level F1 increase for real + synthetic versus real-only was rolled_in_scale (+0.0000); the largest decrease was inclusion (-0.0110).

All pairwise per-seed metric differences are saved in `pairwise-seed-differences.csv`; class-level summaries and real-only deltas are in `per-class-aggregates.csv` and `per-class-f1-differences.csv`. Differences are descriptive. Three seeds provide limited precision, and results do not establish statistical significance or causal generalization beyond this split.

## Discussion

Interpret the mixed-vs-real-only difference as the primary estimate of synthetic augmentation under this specific training recipe and automated candidate policy. Synthetic-only performance measures transfer from the selected generated pool to real held-out images. Per-class tables and confusion matrices should be considered alongside macro averages: a higher aggregate can conceal class-specific losses. Automated eligibility flags do not establish visual defect fidelity; prior contact-sheet review sampled only 72 images and was not an acceptance review of these 783 candidates.

## Limitations and threats to validity

- Only one frozen 70/15/15 partition and three random seeds were evaluated.
- The synthetic pool was not individually human-approved; the eligibility procedure is automated and near-match relationships remain.
- Earlier visual review covered a 72-image sample, not the full candidate pool.
- Classifier results do not independently validate synthetic image realism or industrial usefulness.
- The analysis is descriptive; no inferential significance tests were run.
- Results are specific to this ResNet-18 initialization, preprocessing, training budget, sampling design, and test set.
- No external production-line dataset or cross-site generalization test was included.

## Conclusion and future work

The measured answer to whether augmentation helped is the paired real + synthetic minus real-only comparison above, with its three per-seed values and per-class differences. Treat this result as evidence for this protocol only. Future work should preregister any revised candidate policy, obtain a documented visual review for image-quality claims, and evaluate on an independent acquisition source before making deployment claims.

## Provenance

Evaluation status: completed. Test membership fingerprint: `858aa320a975b419968e4be2b3bf36b5da32df3d3aa622b7516f309e2efcd431`. Frozen run selection fingerprint: `48c5c620e838eaa2e0798c36c3ebfade4796178659b5b96a9243cf551cff28fe`. Evaluation configuration fingerprint: `8b1cf6d22cb4e5aa8496ef481e3018ae9858d77ae477d2aee8c76a9b59e8efd1`. The complete per-run predictions, metrics, confusion matrices, training-record references, software/hardware details, and input hashes are in the adjacent machine-readable evaluation files.
