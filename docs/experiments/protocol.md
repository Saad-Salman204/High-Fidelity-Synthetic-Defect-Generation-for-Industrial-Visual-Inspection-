# Experiment protocol v1 — proposed, freeze before training

Primary question: does adding generated imagery improve macro F1 on held-out NEU real images relative to real-only training? Negative or null effects are valid outcomes. Team review of this protocol precedes implementation runs. Any change gets a dated rationale and revised configuration before test evaluation.

## Provenance and split freeze

Acquire NEU-CLS from the official source after checking terms. Record URL, access date, archive filename/hash, extracted checksums, citations and exclusions. No assumed semantic dataset version: use archive SHA256 as version. Inspect corrupt files and conflicting labels; do not silently repair them.

Group exact duplicates, reviewed perceptual near-duplicates and any known source/capture siblings before splitting. Use seed 42, stratified group-aware 70/15/15 allocation; target 210/45/45 per class (1,260/270/270 total) only if grouping permits. Publish actual counts and residual grouping uncertainty. Quarantine conflicts before freeze. Neither crops nor augmentations may cross partitions.

Seal test manifest/hash after split audit. Only real train may train the generator or supply inpainting pixels/masks/prompts derived from examples. Validation may select settings but never train the generator. Test pixels, labels and generated derivatives must not influence prompts, filtering, training, early stopping or selection. All classifier arms use the identical real test IDs.

## Preprocessing and generator plan

Preserve raw 200 px originals. L→RGB replication; full-image bicubic 512 for generator, normalized [-1,1]. One SD v1.5 UNet attention LoRA, frozen text encoder/VAE; exact base revision and environment must be pinned in the GPU pilot. Suggested rank 4, alpha 4, LR 1e-4, batch 1, accumulation 4, 1,000 updates, seed 42. Confirm actual trainable module set. Save adapter/config/environment and fixed-prompt grids every 250 updates. One small validation-informed revision maximum; record both attempts.

Start with a 100-step feasibility pilot, not an assumed VRAM/runtime promise. Target 600 accepted synthetic images (100/class); minimum 300 (50/class) for the reduced core run. Candidate cap 1,000; allocate initially 166/class, leaving four as a documented reserve. Generate by a fixed seed sequence; never count repeats as unique examples. If a class cannot meet quota, report shortfall, reduce all class quotas equally within budget, or declare generation gate failed. Do not manufacture labels to fill quota.

Fixed prompt template: `grayscale close-up inspection image of a hot-rolled steel surface with {class_description}`. Describe each class using training-only examples. Record full prompt, negative prompt, scheduler, 30 inference steps and guidance 7.5 as proposed defaults; freeze after pilot. Human review rejects unreadable, wrong/ambiguous-class, blank, highly artifacted and duplicate candidates. A second reviewer audits 10 randomly selected accepted images/class plus all disputed labels. Retain reasons/counts for every candidate. Do not filter by final test performance.

## Fair classifier comparison

| Arm | Training pool | Source exposure |
| --- | --- | --- |
| Real-only | Entire real training split | Real 100% |
| Synthetic-only | Accepted synthetic pool | Synthetic 100% |
| Real + synthetic | Same real and same synthetic pools as above | Fixed 50:50 per batch |

Use a uniform six-class sampler in every arm. Sampling with replacement is permitted and logged. The mixed pool may have a different unique-size ratio than its sampling ratio; report both. Same ImageNet ResNet-18 initialization policy, six-class head, 224 px L→RGB inputs and normalization; no color jitter. Same training-only flips if approved by inspection. Synthetic-only still uses a generator trained on real data and real validation for checkpoint selection; disclose this explicitly.

Proposed common training budget: batch 24 (mixed 12 real/12 synthetic), 1,000 optimizer updates, AdamW LR 1e-3, weight decay 1e-4, fixed cosine schedule, validation every 100 steps. All layers fine-tuned equally across arms. Select highest real-validation macro F1, ties choose earliest checkpoint. No arm-specific tuning; choose common settings in the real-only pilot before comparison. Equal epochs would give unequal compute, so equal updates are the primary control. Record sample exposures, unique images and wall time. This estimates utility under a fixed compute budget, not pure dataset-size-matched causality.

Seeds 42, 43, 44 for paired classifier repeats if quota allows. At minimum run seed 42 in all arms; reduce repeats symmetrically before starting final runs and disclose the reduced design. Use same fixed split across repeats. Record Python/NumPy/PyTorch/worker seeds and deterministic settings; CUDA determinism is best effort and may be slower.

## Freeze and evaluation

Freeze manifests, hyperparameters, quality decisions and selected model hashes before opening test metrics. Evaluate each frozen checkpoint once on the same real test manifest. Technical reruns must reproduce identical artifacts or document the correction; do not tune after seeing results.

Primary metric macro F1; also accuracy, macro precision/recall, per-class precision/recall/F1/support. Undefined precision/recall uses `zero_division=0`, with absent predicted/true classes explicitly flagged. Confusion rows=true, columns=predicted; publish integer counts and row-normalized values in fixed class order. Include raw per-example predictions.

Report each seed and mean ± sample standard deviation across three seeds; one seed has no across-seed SD. Optional paired bootstrap on test IDs (2,000 replicates, seed 42) can estimate a conditional interval for macro-F1 differences; disclose dependence and small-sample limits, and do not treat repeated seeds as independent test samples. Report all arms and failed runs, not only the best seed.

## Quantitative generation quality (recommendation)

The proposal's “Basic quantitative check using” is incomplete. Recommend exploratory KID plus the human class-fidelity/rejection audit. Compute against real validation, never test, after generation settings are frozen; label the comparison validation-based and not independent of development selection.

Use pinned TorchMetrics/torch-fidelity Inception-v3 2048 features, uint8 L→RGB, library 299 px resize, polynomial degree 3, gamma 1/2048, coefficient 1; record feature weight checksum. Use 100 seeded subsets of size 40 per class when both sides have at least 40 samples, and size 200 pooled when both have at least 200. If fewer samples, report unavailable or a separately disclosed smaller size ≥2; never silently use the library default 1,000. Report raw KID mean and subset SD (not confidence interval); unbiased estimates can be negative. Record all sizes and parameters. Do not compare scores with different preprocessing as if equivalent.

Natural-image feature mismatch and only 45 validation images/class limit conclusions. KID does not prove correct defects or lack of memorization. Add nearest-training-image visual review, exact/perceptual duplicate counts, deterministic 10-image/class grids with seed 42 and rejected examples. Do not use unpaired pixel SSIM/PSNR as distribution-quality evidence. If feature weights are unavailable, record KID as unmeasured; class-review acceptance rates are a simpler quantitative audit, not a silently substituted distribution metric.

## Records and reporting

Track commit, resolved config/hash, manifest hashes, prompt catalog, model/adapter hashes, seeds, library lock/freeze, Python/CUDA/driver/GPU/VRAM, wall time, failures, artifact location and licenses. Use [record template](../../configs/experiment-record.yaml), [contracts](../architecture/contracts.md), and [results template](../../reports/results-template.md). No data, weights or predictions in Git by default. Publish approved aggregate tables/figures with sufficient provenance to reproduce them.
