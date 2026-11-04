# Technical feasibility and decision review

Reviewed 2026-11-04. The supplied PDF was read in full. No experiments have been run.

| Classification | Content |
| --- | --- |
| Explicit proposal | NEU, pretrained Stable Diffusion + LoRA, modest synthetic set, classifier, three settings, shared real test, classification metrics, visual review |
| Missing specification | The generation-quality item ends at “Basic quantitative check using”; no metric is specified |
| Recommendation | SD v1.5, UNet LoRA, ResNet-18, grouped 70/15/15 split, local tracking, KID, equal update budgets |
| Needs confirmation | GPU/time quota, start date, contributor handles, dataset terms, exact model revision, proposed protocol approval |
| Stretch | NEU-SEG/inpainting, EfficientNet-B0, scarcity ablation, extra synthetic ratios |

## Research distinction

[SynSur v2](https://arxiv.org/html/2604.26633v2), by Paul Julius Kühn, Mika Pommeranz, Arjan Kuijper and Saptarshi Neil Sinha (April 2026; revised July 2026), uses FLUX.1-dev LoRA, VLM prompts, masks, filtering and automatic annotations for detection on BSData/MSD. Our manual class prompts and NEU classification retain only the adaptation/downstream-comparison idea. Its gains do not establish ours.

[Rombach et al.](https://arxiv.org/html/2112.10752v2), CVPR 2022 (preprint December 2021), motivate latent-space diffusion and cross-attention conditioning. This foundational work is not a LoRA recipe. We reuse a pretrained autoencoder rather than learn a latent representation from scratch.

## Technical decisions

| Topic | Proposal and rationale | Validation gate |
| --- | --- | --- |
| Acquisition | Official NEU-CLS, not a mirror's predefined split | Inspect terms; record archive SHA256 and original filenames |
| Grayscale | Decode L, replicate channels for pretrained RGB interfaces | Verify channel equality; synthetic classifier inputs also become L then RGB |
| Resolution | Preserve raw files; full-image bicubic 512 for diffusion, 224 for classifier | Upsampling invents no physical detail; inspect fine defects |
| Split | Duplicate/source-group-aware stratification before augmentation | Unknown original source grouping remains a limitation |
| Model | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Pin revision/hash and inspect complete terms; model-card license label is OpenRAIL-M |
| Alternatives | SDXL/FLUX are heavier than needed for this first study | Do not copy SynSur's compute requirements into this project |
| LoRA | One six-class adapter; frozen VAE/text encoder/base UNet; attention projections only | Assert trainable parameter names/shapes in chosen library version |
| Compute | Batch 1, accumulation 4, rank/alpha 4/4, FP16 when supported, gradient checkpointing | 12–16 GB GPU is a target, not a measured minimum; benchmark 100 steps |
| Duration | Start 1,000 optimizer steps, fixed-seed visual grids every 250 | Measure seconds/step, peak VRAM, save overhead and available quota |
| Conditioning | Manual fixed class prompts, LR 1e-4/dropout 0 starting hypotheses | Inspect six classes; intended prompt class is not guaranteed image truth |
| Generation | 600 accepted target, 1,000 candidates maximum | Record prompts, seeds, checkpoint, failures and rejection rates |
| Inpainting | Optional separate feasibility gate | NEU-CLS supplies no segmentation masks or guaranteed normal backgrounds; verify NEU-SEG terms/overlap |
| Classifier | ImageNet ResNet-18 with six logits | Common weight enum, preprocessing, optimizer and selection rule |
| Tracking | JSON/CSV/YAML with Git and SHA256 | No service, deployment stack or paid tracker |

Budget one small validation-driven configuration revision before freezing. Avoid optional CUDA extensions until the basic stack works. CPU supports data/tests, not a practical full diffusion-training fallback.

## Quality and scope limits

Recommend exploratory KID with fixed features/preprocessing/subset parameters plus visual class fidelity and duplicate inspection. Its unbiased estimator does not remove high variance or natural-image feature mismatch on grayscale textures. FID is not the preferred primary quality metric for this sample size. Full details are in the [protocol](../experiments/protocol.md).

The official dataset is balanced (300/class). Do not claim rare-class gains without a separately agreed scarcity protocol. Filtering may create imbalance; report counts and balance training exposure without altering the test set.

## Go/no-go gates

Day 5: environment and resource readiness; complete the 100-step GPU pilot by day 7. Day 10: reloadable adapter and usable samples per class. If memory fails, use batch 1/checkpointing, then fewer steps or a separately recorded 256 px pilot. By day 12, drop inpainting and extra architectures if core integration is incomplete.

Minimum full deliverable: real baseline, trained LoRA, modest validated synthetic set, all three arms, required metrics and honest limitations. If LoRA is impossible, deliver baseline plus reproducible failure evidence; a frozen-model comparison requires explicit team approval and is labeled a partial fallback, not successful completion of the approved methodology.
