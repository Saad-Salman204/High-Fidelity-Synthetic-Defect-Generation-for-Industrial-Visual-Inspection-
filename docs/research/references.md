# Source ledger

Accessed 2026-11-04; full-text HTML for both primary papers was accessible. No data or model weights were downloaded.

| Source | Verified use | Limits |
| --- | --- | --- |
| [SynSur v2](https://arxiv.org/abs/2604.26633v2), [full text](https://arxiv.org/html/2604.26633v2) | Kühn, Pommeranz, Kuijper, Sinha (2026); FLUX generation and detector evaluation | Preprint, not NEU classifier evidence |
| [LDM v2](https://arxiv.org/abs/2112.10752v2), [full text](https://arxiv.org/html/2112.10752v2) | Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer; CVPR 2022 | Foundation, not adaptation recipe |
| [Official NEU page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/) | 1,800 images, 300/class, 200 × 200; CLS/DET/SEG distinction | Explicit reuse license absent from reviewed page; archive uninspected |
| [SD v1.5 card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | Candidate, 512 px training, OpenRAIL-M label | Full license fetch unsuccessful; revision not pinned |
| [Diffusers LoRA](https://huggingface.co/docs/diffusers/training/lora) | Lightweight adaptation tooling | Rolling docs; compatibility must be tested |
| [KID paper](https://arxiv.org/abs/1801.01401), [authors' code](https://github.com/mbinkowski/MMD-GAN) | Demystifying MMD GANs, ICLR 2018; KID motivation | Not a domain-specific NEU metric validation |
| [TorchMetrics KID](https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html) | RGB range, subset controls, feature-weight dependency | Feature weights excluded from CI |
| [GitHub protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) | Review/check rules and plan-dependent support | Verify actual repository permissions |

Dataset citation: K. Song and Y. Yan, “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,” Applied Surface Science 285, 858–864 (2013).

Our parameter choices are engineering recommendations, not copied experimental findings. No literature score is presented as a project result.
