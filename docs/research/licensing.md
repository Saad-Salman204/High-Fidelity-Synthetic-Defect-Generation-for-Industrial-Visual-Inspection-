# Asset licensing ledger

| Asset | Status | Required action |
| --- | --- | --- |
| Original source code | MIT authorized by requester on 2026-11-04 | Preserve root LICENSE notices |
| Planning documents | Original project documentation; cited third-party material retains its rights | Keep attribution |
| Existing proposal PDF | User supplied and already tracked; contains student identifiers | Team reviews before any future public release; do not rewrite history automatically |
| NEU-CLS | Download page public, explicit reuse/redistribution terms unresolved | Inspect archive terms and seek provider clarification if absent before use/sharing |
| NEU-SEG | Optional; identity, terms, overlap unverified | Separate provenance and leakage review |
| SD v1.5 | Model card identifies CreativeML OpenRAIL-M | Record full terms and revision before use; MIT does not override restrictions |
| LoRA adapters | Derived from model and dataset | Review obligations before release; do not automatically apply MIT |
| Synthetic images | Dependent on model/data terms and applicable rights | Record provenance; no automatic ownership/redistribution guarantee |
| Classifier/feature weights | Weight-specific terms pending | Record originating model license and exact enum/revision |
| Dependencies | Each package has its own license | Record resolved environment and required notices |

Unknown terms are pending, not assumed permissive. Keep data/weights in controlled local or team storage. Git LFS is not needed for the core plan; consider it only for small approved demonstration assets after terms/quota review. This ledger is an engineering release gate, not legal advice.
