# Provisional diffusion selection for NEU-CLS

Reviewed 2026-11-13 for #6. The proposal calls for Stable Diffusion with LoRA and a comparison on held-out real NEU photos. Use one text-to-image adapter for the six classes. Mask-guided inpainting remains a separate optional decision under ADR-008.

| Candidate | Architecture, images and tooling | Compute and selection |
| --- | --- | --- |
| **Primary: SD v1.5** | 512 × 512 latent text-to-image model with one CLIP text encoder, a VAE and a UNet. Use the Diffusers Stable Diffusion LoRA recipe with PyTorch, PEFT and Accelerate. | Best first fit for this project. The smaller model and 512 px target are expected to use less memory and time than SDXL. The v1.5 card records more 512 px pretraining steps than v1.4. That does not establish better NEU results. |
| **Fallback: SD v1.4** | Same 512 px model family and Diffusers pipeline. Keep the training procedure and compare only if primary weights cannot be acquired or their provenance cannot be approved. | Expect roughly the same resource cost as v1.5 at the same settings. This is a source fallback. If GPU memory is the problem, first reduce v1.5 resolution and measure again. |
| SDXL base 1.0 | Common 1024 px target, roughly three times larger UNet and a second text encoder. Diffusers provides a separate SDXL LoRA recipe. | Expect higher memory and longer training and sampling at its usual resolution. The 1024 px target has four times the pixel area of 512 px. The original NEU images are 200 px, so upsampling adds no new physical detail. No SDXL pilot is planned. |

For either v1 model, resize the whole training image and replicate grayscale into three equal channels. Use the agreed normalization and class prompts. The checkpoints have not been validated on industrial textures. Class fidelity and memorization checks remain necessary.

Sources: [SD v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [CompVis SD v1.4 model card](https://huggingface.co/CompVis/stable-diffusion-v1-4), [SDXL model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [SDXL paper](https://arxiv.org/abs/2307.01952), [Diffusers LoRA guide](https://huggingface.co/docs/diffusers/training/lora).

## Revisions and terms

| Model ID | Repository revision | License |
| --- | --- | --- |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` ([Hub metadata](https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5)) | CreativeML Open RAIL-M. This repository identifies itself as a community mirror of the earlier RunwayML release. |
| `CompVis/stable-diffusion-v1-4` | [`133a221b8aa7292a167afc5127cb63fb5005638b`](https://huggingface.co/CompVis/stable-diffusion-v1-4/commit/133a221b8aa7292a167afc5127cb63fb5005638b) | CreativeML Open RAIL-M, per the CompVis card. |
| `stabilityai/stable-diffusion-xl-base-1.0` | [`462165984030d82259a11f4367a4eed129e94a7b`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/commit/462165984030d82259a11f4367a4eed129e94a7b) | [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md). |

The [full v1 license](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE) permits lawful use and fine-tuning subject to the restrictions in section 5 and Attachment A. Section 4 requires retaining applicable notices, marking modified files, supplying the license and carrying the use restrictions forward when distributing covered model derivatives. Treat adapters as covered for release review. Section 6 leaves responsibility for outputs with the user. SDXL has its own restrictions and redistribution terms. The repository MIT license and these model licenses do not grant permission to use or share NEU images. Keep weights, adapters and generated samples in approved storage until the applicable terms are reviewed.

These are repository commits, not weight checksums. When authorized to acquire the primary model, record the exact Diffusers components and precision variant, confirm safetensors availability and hash the actual files. Fetch only the chosen component set. The v1.5 Hub metadata lists roughly 50 GB for all variants and formats combined. For planning, reserve 10 GiB free disk for a single v1 pipeline plus cache and pilot artifacts, then check actual selected file sizes before transfer. The SDXL repo lists a 6.94 GB base safetensors checkpoint alone. No model files were downloaded for this decision. Exact component load behavior and the approved machine's package versions remain unverified.

## University GPU pilot for #10

**Planning estimate:** request a 12–16 GiB GPU for SD v1 LoRA at 512 px. This is a booking target, not a measured minimum. Activations, precision, driver, allocator and other workloads affect usable memory. SD v1.4 has a similar estimate. SDXL is expected to need more under comparable settings and has no booking target in this plan.

1. Complete [#2's university check](../planning/setup.md), confirm #1 permissions, and use only the approved real training list. Record the GPU, free and total VRAM, session limit, software versions, base revision and selected weight checksums. Confirm the #9 implementation and agreed #8 settings before the pilot.
2. Run a batch-one smoke step with one UNet attention LoRA, rank/alpha 4, frozen base UNet weights, VAE and text encoder, accumulation 4 and seed 42. Use FP16 where supported and gradient checkpointing. Record actual precision and settings. Check that only adapter parameters update and that the loss stays finite.
3. Run **100 optimizer updates** and save an adapter/checkpoint. Measure host wall time and device usage. Synchronize CUDA around timed segments, reset peak statistics before the measured interval, and record peak allocated and reserved memory plus device-wide usage from `nvidia-smi`. Include startup and warm-up separately. Resume from a saved checkpoint and reload the adapter. Time a small fixed-prompt sample at the planned 30 inference steps and inspect examples across all six classes without opening the test set.
4. Project total time from measured seconds per update × 1,000 and seconds per image × up to 1,000 candidates. Add startup, save, reload and a 20% time buffer. **Go** if training and reload succeed, there are no non-finite losses or out-of-memory errors, device-wide peak leaves at least 1 GiB free, the projection fits the booked GPU time, and examples show recognizable steel textures and requested defects across classes. Record the sample failures as well as successes.
5. **No-go** if the checks fail or the time projection exceeds quota. For a memory failure, retry once at 256 px with the same model and record the changed setting and its effect on defect visibility. Confirm this protocol change before full runs. For a source/access failure, try the pinned v1.4 fallback using the same pilot gates. Stop and report a partial outcome if the revised run cannot meet the gates.

ADR-002 remains proposed until hardware, model acquisition, image fidelity and GPU time are confirmed. The CPU library smoke check from #2 is useful for setup and does not measure these outcomes.
