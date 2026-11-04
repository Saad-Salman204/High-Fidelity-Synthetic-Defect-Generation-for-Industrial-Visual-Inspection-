# Risk register

Owners are suggested workstreams. Likelihood/impact are qualitative planning judgments, not measured probabilities.

| Risk | Likelihood / impact | Early warning | Mitigation | Contingency | Owner |
| --- | --- | --- | --- | --- | --- |
| Insufficient VRAM | High / high | Pilot OOM | Batch 1, mixed precision, checkpointing; measure early | Smaller approved resolution/budget; document failure if infeasible | B |
| Excessive duration / quota loss | High / high | Pilot projection exceeds session | Budget 100-step pilot, periodic saves, shared GPU calendar | Shorter approved training; symmetric classifier seed reduction | B/C |
| Poor synthetic fidelity | High / high | Artifacts or no clear defect | Fixed prompts, train-only grids, bounded pilot revision | Reduce claims/quota; report negative utility | B/A |
| Incorrect class conditioning | Medium / high | Reviewer disagreement | Six-class rubric and second review | Reject ambiguous outputs; report shortage, never relabel to fill quota | A/B |
| Dataset leakage | Medium / critical | Hash/group overlap, test-derived prompt | Pre-transform grouping, sealed test and ancestry guards | Invalidate affected runs and rerun within quota; disclose | A/C |
| Limited labeled data | High / medium | Seed-sensitive scores | Paired repeats, simple model and uncertainty | One-seed limitation explicit; avoid broad generalization | C |
| Filtering imbalance | Medium / medium | Unequal accepted counts | Report per-class counts and uniform training exposure | Lower quota uniformly or declare class gate failed | A/C |
| Overfitting / memorization | High / high | Train loss falls while validation degrades | Checkpoint selection, nearest-train review, no long searches | Stop early; reject duplicates; report limits | B/C |
| Inconsistent preprocessing | Medium / high | Unexpected ranges or domains | One shared classifier transform and versioned cache keys | Invalidate mismatched artifacts and rebuild | A |
| Licensing restrictions | Medium / high | No explicit archive terms | Source ledger before acquisition/release | Seek clarification; do not redistribute unresolved assets | Shared |
| Dependency incompatibility | High / medium | Import/device/API failure | Tested pinned environment and exact model revision | Known compatible stack; omit optional CUDA extensions | B |
| Integration delays | Medium / high | Contract divergence | Day 4 manifest handshake, day 12 synthetic batch | Freeze interfaces, smaller focused PRs, move reviewers | Shared |
| Three-week deadline | Medium / high | Week 2 adapter or synthetic gate missed | Early baseline, bounded searches, buffer | Drop stretch tasks; deliver honest partial study if core impossible | Shared |
| KID instability/domain mismatch | High / medium | Large subset SD or insufficient images | Fixed features/size, visual review, caveats | Report unavailable/uncertain metric; no false fidelity conclusion | C |
| Missing masks/normal backgrounds | High / low for core | SEG terms/alignment unknown | Keep inpainting outside critical path | No-go I20; text-to-image core unchanged | B |
| Lost ephemeral GPU artifacts | Medium / high | Session expiry, local-only saves | Checkpoint and copy to approved storage with hashes | Resume last verified checkpoint and record lost work | B/C |

Minimum full research deliverable: provenance and frozen splits; real baseline; trained LoRA; modest class-reviewed synthetic set; all three classifier arms; required metrics, generation assessment and honest report. No production deployment is needed.

If LoRA cannot run, baseline plus failure analysis is a partial fallback. A frozen-model augmentation experiment needs team approval and distinct labeling; it does not fulfill the original fine-tuning requirement. Inpainting can be abandoned without compromising the core proposal. See [roadmap](roadmap.md) for dated gates and scope cuts.
