# Detailed implementation plan — no stages implemented

Every stage uses the versioned contracts in [architecture/contracts](../architecture/contracts.md). The table defines purpose, input/output boundaries, libraries, artifact validation, failures, recovery and active effort. Dependencies are issue IDs resolved in the [backlog](backlog.md). Training/compute wait time is additional.

| Stage / purpose | Inputs → outputs and artifacts | Dependencies / proposed libraries | Validation / failure / recovery | Effort / issues |
| --- | --- | --- | --- | --- |
| Acquisition / lawful reproducible source | Official source/terms → archive hash and provenance ledger | None; stdlib hashing, manual official download | Check terms/variant; unclear terms or inaccessible archive → seek provider clarification, no silent mirror | 4h I01 |
| Environment / executable reproducibility | Hardware quota + package candidates → tested constraints, environment record | None; Python/PyTorch tooling | Import/device compatibility; mismatch → known compatible pinned stack | 5h I02 |
| Validation / trustworthy data | Raw archive → validated real manifest, exclusions and group ledger | I01,I02; Pillow, NumPy | Decode/count/class/hash checks; conflict → quarantine and reviewer decision | 6h I03 |
| Splitting / independent evaluation | Validated groups → immutable train/val/test CSV + metadata | I03; scikit-learn/NumPy | Zero group overlap, stable hashes; impossible balance → report actual grouped counts | 6h I04 |
| Preprocessing / compatible tensors | Split manifest + transform config → loaders/cache, source hashes retained | I04; Pillow, torchvision | Shape/channel/range checks; fine-detail damage → inspect train-only examples, revise before freeze | 6h I05 |
| Model preparation / valid base | Candidate card/terms + environment → pinned model specification | I02; Diffusers/Transformers | Revision/license/config match; unsupported input → re-evaluate candidate ADR | 4h I06 |
| Baseline / early classifier result | Real train/val + config → reloadable classifier and validation record | I02,I05; PyTorch/torchvision | Tiny-fixture overfit, reload equality; failure → debug preprocessing/labels before synthetic work | 10h I07 |
| Configuration / shared contracts | Splits/model decision → approved protocol, validated configs/run schema | I04,I06; YAML, stdlib | Missing hashes or path traversal fail; drift → reject run and version schema | 5h I08 |
| LoRA / adapt generator | Real train only + frozen base → adapter, config, checkpoints | I05,I06,I08; Diffusers, PEFT, Accelerate | Parameter freeze, train-only guard; incompatible projections → inspect pinned API | 10h I09 |
| Compute gate / bounded training | LoRA interface → 100-step pilot then trained adapter and resource log | I09; CUDA/PyTorch | Measure VRAM/time and resume; OOM/quota → checkpointing/smaller approved budget | 6h I10 |
| Generation / labeled candidates | Adapter + prompts/seeds → PNGs and synthetic JSONL | I10; Diffusers/Pillow | Unique IDs, all six labels, resumable ledger; collapse → log and one bounded review | 7h I11 |
| Synthetic validation / accepted set | Candidate ledger → accepted manifest, rejection/reviewer records | I11; Pillow/hash utilities | Fidelity/duplicates/ancestry; poor class → documented shortfall or no-go | 7h I12 |
| Generation evaluation / inspect quality | Accepted set + real val → KID record and fixed grids | I12; TorchMetrics/torch-fidelity/matplotlib | Subset bounds and feature hash; insufficient counts → explicit unavailable metric | 6h I13 |
| Arm integration / fair training | Real + accepted synthetic manifests → three shared training paths | I07,I08,I12; PyTorch | Membership/source ratio/budgets; mismatch → block full runs | 7h I14 |
| Comparative training / research runs | Frozen shared config → paired-seed checkpoints/run inventory | I14,I13; PyTorch | Same updates/selection policy; quota → symmetric predeclared seed reduction | 6h I15 |
| Final evaluation / defensible evidence | Frozen checkpoints + sealed test → predictions/metrics/confusions | I15; scikit-learn/matplotlib | Exact test IDs/counts, hand-checked metrics; invalid output → fix technical bug with disclosure | 6h I16 |
| Reporting / explain findings | Run artifacts + quality analysis → report/tables/limitations | I13,I16; Markdown/matplotlib | Every claim traceable; missing evidence → mark unmeasured | 6h I17 |
| Handoff / reproducibility | Report + environment/artifact inventory → independent audit | I17; fresh environment | Replay checks/known limitations; missing artifact → restore approved backup | 5h I18 |
| CPU CI / prevent regressions | Implemented data/config logic → meaningful tests/workflow | I05,I08; pytest/Ruff | No large downloads; empty/mocked tests → clearly separate real validation | 4h I19 |
| Optional masks / feasibility only | Core accepted synthetic set + NEU-SEG evidence → go/no-go ADR | I12; manual audit/Pillow | Terms/alignment/ancestry/backgrounds; missing prerequisite → drop optional work | 4h I20 |

Configuration flows from configs to explicit functions, never global machine paths. Runtime entry points must be introduced and documented by their implementation issues. Data, generator and classifier owners can build against the fixed CSV/JSON contracts independently after the first-week handshake. No actual ML functions are included in the foundation.
