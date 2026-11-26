# Suggested presentation structure

1. **Problem and motivation** — NEU-CLS class scarcity and the synthetic augmentation question.
2. **Dataset and leakage controls** — class mapping, frozen 70/15/15 split, manifest hash, shared test set.
3. **Diffusion + LoRA pipeline** — base revision, train-only ancestry, rank/alpha, 1,000 updates.
4. **Candidate audit and selection** — 996 candidates; 783 selected under the automated no-blocker/no-warning policy; disclose no individual human approval and retained near matches.
5. **Classifier and experiment design** — fully fine-tuned ResNet-18, three arms, seeds 42/43/44, equal 1,000-step budget.
6. **Aggregate test results** — accuracy and macro F1 by arm with seed variability.
7. **Class-wise behavior** — per-class F1 and confusion matrices; highlight both gains and regressions.
8. **Interpretation and limitations** — descriptive n=3, one split, automated quality screen, no external dataset.
9. **Conclusion and next experiment** — state the measured mixed-vs-real result and propose preregistered follow-up.
