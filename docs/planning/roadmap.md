# Three-week roadmap and responsibility matrix

Relative project days are used: actual kickoff date and GPU quota are unconfirmed. Estimates total 120 active hours (116 excluding optional I20), about 40 per contributor across three weeks before review/contingency. Reserve roughly 20% additional time for review/integration; confirm team availability at kickoff. GPU waiting is additional, and one shared GPU must be scheduled serially.

## Schedule and gates

| Window | A — Data | B — Generative ML | C — Classification/evaluation | Integration |
| --- | --- | --- | --- | --- |
| Days 1–2 | I01 provenance; begin I03 after environment | I02 environment with C; I06 model review | Review contracts and help I02; baseline design only | Agree mapping, paths, ownership; no unavailable-data training |
| Days 3–5 | I03/I04 freeze splits; I05 transforms | I08 with C; prepare I09 only after contracts | I07 real-only baseline after I05 | Day 4 manifest handshake; day 5 CPU interface + resource readiness |
| Days 6–7 | Finish loader review; I19 fixtures | I09 implementation; 100-step I10 pilot as soon as ready | Complete I07 and shared budget review | Week 1 exit: baseline validation, frozen split, GPU feasibility; escalate missed pilot |
| Days 8–10 | Review provenance/quality rubric | I10 bounded LoRA then I11 generation | I19 and I14 loader scaffolding against authored fixtures | Day 10 adapter reload + six-class candidate inspection |
| Days 11–14 | I12 candidate audit | Finish I11; help I12; I20 only if ahead | I13 quality metric then I14 integration | Day 12 small synthetic classifier run; day 14 freeze accepted set and configs |
| Days 15–18 | Audit counts/manifests, support reporting | Support reproducibility and generation figures | I15 paired arms then I16 final evaluation | Freeze checkpoints before opening test; no late tuning |
| Days 19–21 | I17 data/limitations section | I17 generation section; I18 handoff support | I17 comparison; I18 independent replay | Buffer, faculty report and final audit |

Critical path: I01/I02 → I03 → I04 → I05 → I09 (also I06/I08) → I10 → I11 → I12 → I13/I14 (also I07) → I15 → I16 → I17 → I18. Baseline I07 runs in parallel with generator work after preprocessing. Integration occurs before Week 3, not on the final day.

GPU slots: one short classifier pilot in Week 1, then LoRA pilot/training in early Week 2, generation later Week 2, final classifiers in Week 3. CPU validation/docs proceed while GPU is busy. Checkpoint before session expiry and copy approved artifacts to team storage.

## Responsibility matrix (suggestions, not person assignments)

| Tasks | Suggested owner | Dependencies | Deliverables | Week | Reviewer |
| --- | --- | --- | --- | --- | --- |
| I01/I03/I04/I05 | A | Environment/source chain | Provenance, groups, split and transforms | 1 | C for splits, B for diffusion inputs |
| I02 | B with C | None | Tested environment | 1 | A |
| I06/I09/I10/I11 | B | Model/config/data contracts | Adapter, candidates and run records | 1–2 | A for provenance; C for interface |
| I07 | C | I02/I05 | Real baseline | 1 | B |
| I08 | C with A/B approval | I04/I06 | Frozen protocol/config schema | 1 | A/B |
| I12 | A with B | I11 | Accepted manifest and audit | 2 | C second-review sample |
| I13/I14/I15/I16 | C | Quality/baseline chain | Three arms and metrics | 2–3 | A on counts, B on controls |
| I19 | A with C | I05/I08 | CPU CI fixtures/contracts | 2 | B |
| I17 | Shared, A coordinates | I13/I16 | Final report | 3 | Each reviews another section |
| I18 | A runs replay, B supports | I17 | Independent handoff | 3 | C |
| I20 | B, optional | I12 | Inpainting go/no-go | 2 | A/C |

No individual names or GitHub handles are assigned. C's workload is highest: A takes metric-fixture preparation and report assembly; B helps checkpoint/record audit once generation ends. One contributor edits shared protocol/config files at a time; consumers review interface PRs before merging.

## Scope cuts and fallback

Drop I20/inpainting first, then EfficientNet, extra ratios and scarcity ablations (not backlog commitments). Reduce seeds symmetrically to one only before final runs, and reduce accepted synthetic quota uniformly to 50/class within the candidate budget if needed. Do not remove a required comparison arm or leak test data to make the schedule.

If no trainable LoRA by day 10, record the exact failure and discuss the reduced methodology immediately; CPU training is not a credible substitute under this deadline. Baseline plus failure analysis is an honest partial deliverable. Frozen-model experiments need explicit team approval and must not be described as fine-tuned results.
