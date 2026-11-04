# Implementation backlog — 20 open GitHub issues

All tasks remain planned. Suggested workstreams are not individual assignments. Estimates exclude GPU queue time. Dependencies below use actual GitHub numbers.

| Local ID | GitHub issue | Week | Priority | Hours | Dependencies |
| --- | --- | --- | --- | --- | --- |
| [I01](issues/I01.md) | [#1: Confirm NEU provenance and permitted use](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/1) | 1 | P0 | 4 | None |
| [I02](issues/I02.md) | [#2: Validate and lock the development and GPU environment](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/2) | 1 | P0 | 5 | None |
| [I03](issues/I03.md) | [#3: Implement dataset integrity and duplicate audit](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/3) | 1 | P0 | 6 | #1, #2 |
| [I04](issues/I04.md) | [#4: Freeze reproducible group-aware real splits](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/4) | 1 | P0 | 6 | #3 |
| [I05](issues/I05.md) | [#5: Implement shared grayscale preprocessing and loaders](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/5) | 1 | P0 | 6 | #4 |
| [I06](issues/I06.md) | [#6: Select and record the pretrained diffusion candidate](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/6) | 1 | P0 | 4 | #2 |
| [I07](issues/I07.md) | [#7: Build and verify the real-only classifier baseline](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/7) | 1 | P0 | 10 | #2, #5 |
| [I08](issues/I08.md) | [#8: Freeze experiment configuration and run-record interfaces](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/8) | 1 | P0 | 5 | #4, #6 |
| [I09](issues/I09.md) | [#9: Implement the minimal UNet LoRA training path](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/9) | 2 | P0 | 10 | #5, #6, #8 |
| [I10](issues/I10.md) | [#10: Run the GPU pilot and bounded LoRA adaptation](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/10) | 2 | P0 | 6 | #9 |
| [I11](issues/I11.md) | [#11: Implement seeded class-conditioned generation and provenance](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/11) | 2 | P0 | 7 | #10 |
| [I12](issues/I12.md) | [#12: Validate synthetic quality and class labels](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/12) | 2 | P0 | 7 | #11 |
| [I13](issues/I13.md) | [#13: Add exploratory KID and qualitative generation assessment](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/13) | 2 | P1 | 6 | #12 |
| [I14](issues/I14.md) | [#14: Integrate the three classifier training arms](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/14) | 2 | P0 | 7 | #7, #8, #12 |
| [I15](issues/I15.md) | [#15: Run the frozen three-arm classifier comparison](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/15) | 3 | P0 | 6 | #14, #13 |
| [I16](issues/I16.md) | [#16: Evaluate frozen models on the shared real test set](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/16) | 3 | P0 | 6 | #15 |
| [I17](issues/I17.md) | [#17: Produce the comparative report and limitations discussion](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/17) | 3 | P0 | 6 | #13, #16 |
| [I18](issues/I18.md) | [#18: Audit reproducibility and complete the research handoff](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/18) | 3 | P0 | 5 | #17 |
| [I19](issues/I19.md) | [#19: Expand CPU CI for implemented contracts and failure cases](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/19) | 2 | P1 | 4 | #5, #8 |
| [I20](issues/I20.md) | [#20: Assess optional mask-guided inpainting feasibility](https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-/issues/20) | 2 | P2 | 4 | #12 |

Recommended order follows dependencies, not numerical assignment alone. I19 can run alongside Week 2 generation; I20 is optional.
