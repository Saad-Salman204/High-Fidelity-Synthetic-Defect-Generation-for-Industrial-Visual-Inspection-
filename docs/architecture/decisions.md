# Architectural decision records

Accepted here means mandated by the request, not experimentally validated. Proposed choices require the named gate before use. Amend an ADR by adding date/rationale; preserve superseded decisions.

| ID / status | Decision and alternatives | Consequences / acceptance gate |
| --- | --- | --- |
| ADR-001 Accepted | Python/PyTorch ecosystem; avoid a multi-framework stack | Matches requested preference; exact environment tested in I02 |
| ADR-002 Proposed | SD v1.5 512 rather than SDXL/FLUX | Lower expected resource cost; model revision/license and GPU pilot I06/I10 decide |
| ADR-003 Proposed | One UNet attention LoRA, frozen VAE/text encoder; not six adapters/full fine-tuning | Shares statistical strength but may confuse classes; I09/I10 inspect trainable parameters and fidelity |
| ADR-004 Proposed | ResNet-18 instead of simultaneous EfficientNet study | Less comparison complexity; I07 baseline verifies input and compute |
| ADR-005 Proposed | Group-aware stratified 70/15/15; not random augmented-image split | Exact counts may vary; I04 peer audit freezes hash |
| ADR-006 Accepted | Simple local experiment tracking rather than hosted MLOps | Must retain immutable metadata and hashes; I08 defines records |
| ADR-007 Accepted | No datasets/checkpoints in Git; controlled artifact storage | Team must back up manifests/records with artifacts; no Git LFS required |
| ADR-008 Proposed | Text-to-image core, inpainting only after I20 gate | No guaranteed masks/normal backgrounds in NEU-CLS; optional work cannot delay comparison |
| ADR-009 Proposed | Python 3.11, pinned dev dependencies; tested GPU-specific ML constraints later | Avoid pretending a CUDA stack is verified; I02 records compatible versions |
| ADR-010 Accepted (2026-11-04) | MIT for original code, user authorized; asset terms separate | Dataset/model/generated artifact distribution still gated by license ledger |
| ADR-011 Proposed | KID + visual fidelity audit; not FID alone | Small-sample/domain caveats; I13 tests subset sizing and records features |
| ADR-012 Proposed | Equal optimizer updates, uniform class exposure, 50:50 mixed-source sampling | Data-size and sample-repeat effects remain; I08 freezes protocol before I15 |

Issue IDs are stable local identifiers; actual GitHub number mapping is in the [backlog](../planning/backlog.md). No unresolved model/license/compute decision is marked Accepted.
