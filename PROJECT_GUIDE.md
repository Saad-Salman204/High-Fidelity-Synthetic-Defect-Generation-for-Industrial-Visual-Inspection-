# Understand our steel defect project

For junior developers, teammates and viva preparation. Basic Python is enough to start reading. The [lab manual](IMPLEMENTATION.md) has the exact commands, required assets, review pauses and recovery steps.

## Chapter 1 — Why are we doing this?

Imagine a camera photographing a steel sheet. One photo shows a long scratch. Another shows tiny pits. Another shows fine cracks. **Industrial visual inspection** means checking pictures for problems. Our classifier predicts the type of defect in a small image. A full factory system would also need evidence about healthy surfaces, different cameras and lighting, and finding the location of defects.

Real defect images can be costly to collect and label. **Data scarcity** means too few examples. **Class imbalance** means some categories have many more examples than others. NEU-CLS starts balanced. The publisher describes 1,800 grayscale images at 200×200 pixels, 300 each of crazing, inclusion, patches, pitted surface, rolled-in scale and scratches. Our main experiment studies a small balanced benchmark. [Official NEU description](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/).

**Our question is whether adding reviewed generated images helps a classifier recognize defects in unseen real photos.** We compare real-only, synthetic-only and mixed training. The main result is mixed minus real-only macro F1 on the shared real test set. Improvement, no clear difference and a decline are all possible outcomes.

## Chapter 2 — The complete system

```text
Approved NEU archive
       |
Audit files, labels and duplicates
       |
Human grouping decisions
       |
Frozen real split table
       |
       +-- TRAIN --> 224 px --> real training pool ---------------------+
       |       |                                                      |
       |       +--> 512 px --> Stable Diffusion LoRA                   |
       |                             |                                |
       |                        saved adapter                         |
       |                             |                                |
       |                       seeded generation                      |
       |                             |                                |
       |                  candidate PNGs + provenance                 |
       |                             |                                |
       |                  audit + two human reviews                   |
       |                             |                                |
       |                   approved synthetic pool -------------------+
       |                             |                                |
       +-- TRAIN reference ------ KID + class grids                    |
                                                                      v
       +-- VALIDATION ------------------------------> three classifiers
                                                     real / synthetic / mixed
                                                                      |
                                                            frozen checkpoints
                                                                      |
       +-- TEST (sealed until final scoring) --------------------------+
                                                                      |
                                                   metrics, charts and report
```

A **manifest** is a table linking an image to its path, ID, label, fingerprint and partition. **Train** changes model weights. **Validation** chooses a classifier checkpoint. **Test** measures the already frozen choices. Related duplicate images stay in one partition. Only real train teaches the generator. Synthetic candidates need review before joining a classifier training pool.

CPU work includes data checks, splits, fixture tests, reviews and reports. KID and classifier inference can run on CPU with the required local assets. The lab GPU is the practical choice for full-size LoRA, image generation and classifier experiments. **Implemented** means code exists. **Fixture-tested** means it works with small controlled input. **Experimentally verified** means the real assets were used and actual outputs measured.

## Chapter 3 — Important AI concepts

### How a classifier learns

**Artificial intelligence** is the broad area of building systems that perform tasks requiring useful judgment. **Machine learning** learns patterns from examples. **Deep learning** uses neural networks with multiple processing layers. A **neural network** applies mathematical operations with adjustable numbers called weights.

A **convolutional neural network**, CNN, slides small learned filters across an image. A filter can detect a thin line wherever it appears. **ResNet-18** is a CNN with skip connections, allowing blocks to learn corrections to information already present. Our last layer produces six **logits**, or raw class scores. The largest is the prediction. Softmax turns scores into numbers that sum to one.

**Transfer learning** starts from existing learned weights. We load approved local ImageNet weights, replace the original output layer with six outputs, and fine-tune the classifier. Imagine a student who already recognizes edges and textures and now learns steel categories. [Torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

1. A **batch** of images goes through a **forward pass**.
2. The **loss function** measures error. Cross-entropy penalizes low probability for the correct class.
3. **Backpropagation** calculates **gradients**, which describe how changing a weight affects loss.
4. The **optimizer**, AdamW here, updates weights. The learning rate controls update size. A cosine schedule gradually lowers it in classifier training.
5. Repeat. An **epoch** usually means one pass through the training pool. Our comparison controls optimizer **updates**, because pools differ in size and training can draw the same image repeatedly.

During **inference**, weights stay fixed and the model makes predictions. Validation and final testing are inference.

### How diffusion learns and generates

**Generative AI** makes new examples. Imagine a photo covered with static. During training we add a known amount of noise and ask the model to predict the noise. During generation we start from noise and repeatedly move toward an image.

| Part | Simple job | Mechanism here |
| --- | --- | --- |
| **VAE** | Compress and reconstruct | Encodes pixels into a smaller array, the **latent**. Decodes the final latent into pixels. |
| **Tokenizer and text encoder** | Turn words into vectors | Encode the class prompt for the UNet. |
| **UNet** | Estimate noise | Reads the noisy latent, timestep and text encoding, and predicts noise. |
| **Scheduler** | Define noise levels and updates | DDPM adds noise during training. DDIM controls the denoising steps during generation. |

**Forward diffusion** adds noise at a sampled timestep. We compare the predicted and actual noise with mean squared error. **Reverse denoising** uses successive UNet predictions to update the starting noise. **Latent diffusion** does the main work in compressed space. Stable Diffusion v1.5 is our pretrained base. [Latent Diffusion paper](https://arxiv.org/abs/2112.10752).

**Fine-tuning** adapts a pretrained model. **LoRA** represents a weight adjustment using two small matrices. Think of adding correction notes to a textbook. We freeze the VAE, text encoder and original UNet weights. We train adapter matrices on UNet attention projections `to_q`, `to_k`, `to_v` and `to_out.0`. One adapter covers six class prompts. This reduces trainable parameters and checkpoint size. The GPU still holds the base model and intermediate calculations. [LoRA paper](https://arxiv.org/abs/2106.09685).

A **random seed** starts a repeatable random sequence. We record seeds together with code, model, data, configuration, software and hardware. **KID**, Kernel Inception Distance, compares two image collections using Inception feature vectors and a mathematical similarity function. It measures distribution difference. Human review establishes the visible intended label, while the classifier experiment measures utility. [KID paper](https://arxiv.org/abs/1801.01401).

## Chapter 4 — All 20 issues

The table below preserves the original planning-time issue snapshot; any pending markers are historical. Current implementation and experiment evidence are summarized in the README and Phase 1 research report. An open or merged GitHub label alone does not establish real experimental completion. Paths in the package columns are relative to `src/industrial_defect_diffusion/`.

| Issue and prerequisites | Purpose, code and handoff | Verification and remaining work |
| --- | --- | --- |
| **#1 Confirm NEU provenance and permitted use** · none | `data/README.md` and licensing ledger identify the official collection and terms. Source evidence → provenance ledger. | Publisher description checked. Actual archive, hash and permitted use still need evidence. Feeds #3. |
| **#2 Validate and lock the development and GPU environment** · none | Environment probe, package pins and Windows CPU snapshot. Installed stack → version record and tiny six-score forward pass. | CPU tested. Lab driver, CUDA, GPU memory and quota pending. Feeds #3, #6, #7. |
| **#3 Implement dataset integrity and duplicate audit** · #1, #2 | `scripts/audit_dataset.py` calls `data/validation.py`. Images/inventory → counts, bad files, labels, exact groups and near pairs. | Tests cover corrupt images, wrong shapes, conflicts, unsafe paths and missing inventory. Real audit and human grouping feed #4. |
| **#4 Freeze reproducible group-aware real splits** · #3 | `scripts/freeze_splits.py` and `data/splits.py`. Audit/review and verified dataset-source fingerprint → immutable CSV and metadata. | Tests cover stable allocation, groups, unresolved pairs and stale files. Real split approval feeds #5 and #8. |
| **#5 Implement shared grayscale preprocessing and loaders** · #4 | `data/preprocessing.py`. Image/frozen row → model-sized tensor, label and ID. | Tests check channels, ranges, deterministic evaluation, train-only transforms and hashes. Real sample inspection pending. Feeds #7, #9. |
| **#6 Select and record the pretrained diffusion candidate** · #2 | Diffusion selection document and ADR-002 pin v1.5. Candidate research → revision and pilot gates. | Hardware, acquired weights and fidelity unverified. v1.4 is a source fallback in training; generation/review currently expect v1.5. Feeds #8, #9. |
| **#7 Build and verify the real-only classifier baseline** · #2, #5 | Baseline script and `classification/baseline.py`. Real train/val/config/weights → best checkpoint and records. | Tests cover six logits, backward, reload and held-out protection. Actual baseline pending. Feeds #14. |
| **#8 Freeze experiment configuration and run-record interfaces** · #4, #6 | `configs/`, config validator and run records. Settings/run inputs → validated config and hashed evidence. | Tests cover malformed settings, paths, serialization and exclusive outputs. Team must approve actual settings. Feeds #9, #14. |
| **#9 Implement the minimal UNet LoRA training path** · #5, #6, #8 | LoRA script and `diffusion/`. Local base/real train/config → adapter and resume state. | Tiny model tests verify loss, trainable adapter parameters and recovery. Full-size device evidence belongs to #10. |
| **#10 Run the GPU pilot and bounded LoRA adaptation** · #9 | Use LoRA CLI and pilot guide to measure memory/time and save a trained adapter. | Real experiment pending. Memory instrumentation, sample inspection and go/no-go need lab work. Feeds #11. |
| **#11 Implement seeded class-conditioned generation and provenance** · #10 | Generation script, `generation/sampling.py` and `candidates.py`. Approved model/adapter → plan, PNGs and pending ledger. | Tests cover six prompts, stable bookkeeping, tiny sampling and resume. Real samples and speed pending. Feeds #12. |
| **#12 Validate synthetic quality and class labels** · #11 | Review script and `generation/{quality,review}.py`. Candidates/fingerprints/two reviewers → approved export and full ledger. | Tests cover corruption, duplicates, held-out flags, missing/disputed decisions and classifier loading. Real review pending. Feeds #13, #14 and optional #20. |
| **#13 Add exploratory KID and qualitative generation assessment** · #12 | Quality CLI and `evaluation/{inception,kid,quality}.py`. Real train/accepted synthetic → KID status, grids and reports. | Tests cover known calculations, seed repeatability, missing weights and isolation. Actual feature extraction and scores pending. Implementation uses train reference per current protocol. Feeds #15, #17. |
| **#14 Integrate the three classifier training arms** · #7, #8, #12 | `classification/arms.py`, `data/synthetic.py` and shared trainer. Approved pools → separate arm checkpoints. | CPU fixtures check membership, quotas, classes, overlap and shared validation. Real runs belong to #15. |
| **#15 Run the frozen three-arm classifier comparison** · #14, #13 | Arm guide and protocol define paired seeds and common budget. Approved pools/config → run inventory and checkpoint selection. | Actual comparable runs and hashes pending. Feeds #16. |
| **#16 Evaluate frozen models on the shared real test set** · #15 | Evaluator CLI and `evaluation/{classifier,metrics,reporting}.py`. Sealed run selection → predictions, metrics and plots. | CPU tests cover known scores, invalid checkpoints/records, determinism and missing pairs. Real test results await #15. Feeds #17. |
| **#17 Produce the comparative report and limitations discussion** · #13, #16 | `reports/results-template.md` guides reporting. Run outputs → reviewed evidence-based conclusions. | Real report pending. Check every number against its artifact. Feeds #18. |
| **#18 Audit reproducibility and complete the research handoff** · #17 | Setup and handoff guides. Approved code/assets/records → independent reproduction record. | Independent scientific handoff still needs another teammate's replay and checks. |
| **#19 Expand CPU CI for implemented contracts and failure cases** · #5, #8 | GitHub workflow and CPU contract tests. Small fixtures → CI status. | Meaningful regression checks implemented. Real GPU verification is separate. |
| **#20 Assess optional mask-guided inpainting feasibility** · #12 | ADR-008 records a possible extension. Masks/backgrounds/terms/overlap → go/no-go decision. | Optional and unimplemented. Text-to-image core can proceed independently. |

## Chapter 5 — Walk through the files

```text
README.md                       entry point
IMPLEMENTATION.md                lab commands
PROJECT_GUIDE.md                 this guide
AGENTS.md, CLAUDE.md              engineering rules
CONTRIBUTING.md, LICENSE         collaboration and code terms
GenAI Project Proposal.pdf       motivation
pyproject.toml, requirements/    packages and versions
.github/workflows/quality.yml    CPU checks
configs/                        baseline.yaml, cpu-smoke.yaml, experiment-record.yaml
data/README.md                  source and permissions
scripts/                        command entry points
src/industrial_defect_diffusion/
  data/                         validation, splits, preprocessing, synthetic loaders
  classification/               baseline and arm sampling
  diffusion/                    local model, training, checkpoints
  generation/                   sampling, candidates, quality, review
  evaluation/                   KID, quality, final classifier scoring and reporting
  utils/                        configurations and records
tests/                          fixtures, module and integration checks
docs/architecture/              overview, contracts, decisions
docs/experiments/               protocol and stage usage
docs/planning/, docs/research/   backlog, setup, model, terms and risks
reports/                        verification and results template
```

Other root files include formatting settings, changelog, security and conduct policies. GitHub has PR/issue templates, CODEOWNERS and dependency update settings. Data, weights and run folders are ignored local storage. The archive, logs and provenance need approved backup alongside the code.

Running `python scripts/train_lora.py ...` enters the file's `main()`. The CLI parses the flags, reads configuration and calls imported functions. The editable package install points imports at `src/`. That keeps reusable mathematics and validation separate from command orchestration. PyTorch handles tensors, torchvision supplies ResNet, Pillow decodes images, Diffusers/Transformers/PEFT supply diffusion components, and optional torch-fidelity supplies the KID features.

### Command entry points

| File under `scripts/` | Responsibility |
| --- | --- |
| `check_foundation.py` | Check local links, source/config syntax and planned dependencies. |
| `check_ml_environment.py` | Print actual packages/device information and verify a tiny forward pass. |
| `check_experiment_config.py` | Validate settings and print structured status. |
| `audit_dataset.py` | Open real files and write the JSON audit. |
| `freeze_splits.py` | Print an audit digest or freeze reviewed rows. |
| `train_real_baseline.py` | Run one or all classifier arms for one configured seed. |
| `train_lora.py` | Run or resume local train-only LoRA. |
| `generate_defects.py` | Verify the adapter and sample pending candidates. |
| `review_synthetic.py` | Audit, request second review, or export accepted images. |
| `assess_synthetic_quality.py` | Write exploratory KID status and qualitative grids. |
| `evaluate_classifier.py` | Verify selected runs and produce frozen real test predictions/reports. |

### Reusable modules

Paths below start at `src/industrial_defect_diffusion/`.

| Module | Important functions, inputs, outputs and failure behavior |
| --- | --- |
| `data/validation.py` | Audit/freeze call `validate_dataset(root, expected_paths=..., near_threshold=...)`. It scans layouts, decodes, assigns class IDs, checks properties and returns a report. `difference_hash(image)` returns a coarse visual fingerprint. `format_summary(report)` returns text. Bad images are recorded and invalid inputs can stop the stage. |
| `data/splits.py` | `audit_digest(audit)` fingerprints the canonical report. `create_splits(audit, review, source_version=..., seed=..., ratios=...)` applies exclusions, joins related groups, then puts larger groups where their class has the largest shortfall. It returns rows/metadata. `validate_manifest(rows)` checks isolation. `freeze_splits(...)` audits again and writes/verifies output. Unresolved groups, conflicts or changed inputs stop it. |
| `data/preprocessing.py` | `read_frozen_manifest(manifest, metadata)` validates hashes/schema and returns rows/metadata. `NEUManifestDataset(root, rows, split=..., target=...)` exposes a partition. Indexing checks bytes and returns tensor/label/ID. `preprocess_image(image, target, ...)` performs full-image bicubic resize, grayscale replication, train-only flips and model normalization. `make_loader(...)` returns batches with deterministic evaluation order. Both model families call it. Changed images fail loading. |
| `data/synthetic.py` | `ApprovedSyntheticDataset(...)` verifies accepted-only rows, canonical labels, source hashes, reviewer and ancestry. It checks overlap and uses the shared classifier transform. Pending rows, unsafe paths and changed image bytes fail. |
| `utils/experiment_config.py` | `load_config(path)` reads YAML/JSON and `validate_config(config)` checks keys, values and budgets. `relative_path(value, field)` rejects invalid paths. All run CLIs use this common system. Errors name the problem. |
| `utils/run_records.py` | `make_run_record(config, stage=..., seed=..., git_commit=..., ...)` combines settings, timestamps, environment, artifact hashes and measured results. `write_run_record(path, record)` creates an exclusive JSON file. Missing failure reasons or invalid metadata are errors. |
| `classification/baseline.py` | `build_model(weights, local_weights)` creates ResNet and six outputs. `fit_baseline(model, train, validation, settings, device, checkpoint, ...)` performs cross-entropy/backward/AdamW updates and cosine scheduling. `validate_epoch(...)` produces validation scores. The earliest highest macro F1 checkpoint wins. `load_checkpoint(path, device)` restores fixed weights. Wrong partitions and non-finite losses fail. |
| `classification/arms.py` | `ArmDataset(pools)` tags selected training sources. `ArmBatchSampler(dataset, quotas, steps, seed)` cycles class orders, draws images with replacement and shuffles batches. `make_arm_loader(arm, real, synthetic, settings, seed)` returns a correctly sourced loader. Missing sources/classes fail. |
| `diffusion/local_model.py` | `snapshot_fingerprints(root, variant)` hashes selected component files. `load_local_components(root, variant)` returns UNet/VAE/text encoder/tokenizer/scheduler. Training and generation call it. Missing local safetensors or components cause errors. |
| `diffusion/training.py` | `prepare_adapter(parts, settings, device)` freezes base weights and inserts LoRA. `adapter_parameters(unet)` verifies trainable names. `encode_prompts(...)` returns six embeddings. `noise_prediction_loss(parts, pixels, labels, prompt_embeddings)` encodes images, samples noise/timesteps, predicts noise and returns scalar MSE. `train_adapter(...)` accumulates gradients, updates and invokes checkpoints. Bad partitions/shapes/loss/device fail. |
| `diffusion/checkpoints.py` | `save_checkpoint(..., provenance, cursor)` saves adapter, optimizer/scaler/RNG and progress. `load_adapter(path, parts)` checks tensors/hashes. `resume_checkpoint(...)` restores compatible state. Changed provenance blocks resume. |
| `generation/sampling.py` | `verify_checkpoint(model_dir, adapter, config, manifest, approved_metadata_hash)` binds the local assets to the approved source. `make_sampler(...)` returns a callable mapping seed/prompt rows to Pillow images. Wrong revisions/base files/adapter hashes stop sampling. |
| `generation/candidates.py` | `candidate_plan(config, source, runtime)` creates counts, IDs, prompts and seeds. `generate_candidates(output, plan, sampler, resume=...)` writes batch PNGs and metadata and returns the ledger path. Existing batches are checked on resume. Changed plans and corrupt committed output fail. |
| `generation/quality.py` | `audit_candidates(generation, manifest, real_audit, approved_hash, threshold=...)` reconstructs the plan and checks provenance, image properties and duplicate/leakage fingerprints. `write_contact_sheets(...)` makes review pages. Held-out pixels are never opened here. |
| `generation/review.py` | `review_template(audit)` creates a blank form. `review_candidates(audit, primary, secondary=None)` validates every decision, requests seeded second reviews, or returns the ledger, accepted bytes and counts. Missing/disagreeing/blocked decisions fail export. |
| `evaluation/inception.py` | `local_extractor(path, approved_hash)` returns the CPU extractor and metadata or a missing-weight status. Incorrect supplied hashes fail. |
| `evaluation/kid.py` | `kid_subset(real, synthetic)` computes the cubic-kernel estimate from feature matrices. `score_pools(...)` makes seeded per-class/pooled draws, caches features and returns statistics or unavailable reasons. Invalid/non-finite features fail. |
| `evaluation/quality.py` | `assess_quality(...)` verifies review output, opens real train only, scores accepted synthetic rows and writes grids/JSON/Markdown. Changed artifacts and reused destinations fail. |
| `evaluation/classifier.py` | `evaluate(root, config, selection, approved_manifest_sha256)` verifies records/checkpoints and returns per-run and aggregate reports. `predict_test(model, loader, device, run_id)` produces real test probabilities in inference mode. Bad checkpoints fail and incomplete experiments remain visible. |
| `evaluation/metrics.py` and `reporting.py` | `classification_metrics(expected, predictions)` checks ID coverage, classes and probabilities, then calculates metrics. `write_results(output, report, runs)` writes predictions, per-class tables, confusion SVGs and comparisons. Duplicate/missing IDs, corrupt probabilities and existing output are errors. |

The baseline config contains proposed lab defaults. The smoke config defines tiny CPU settings, without supplying a real dataset. The record template explains what runs save. Test files mirror the modules. `conftest.py` creates controlled images and frozen rows. Audit/split tests check bad files and grouping, preprocessing tests check numeric transforms, training tests check gradients and reload, and review/evaluation tests check boundaries and outputs. CI installs packages, then runs dependency, foundation, lint, format and CPU tests. Real assets are outside those tests.

## Chapter 6 — Follow one hypothetical image

Take a teaching example called `Sc_017.bmp`. We have not inspected a real file with this name.

1. The audit recognizes a 200×200 grayscale scratch, class ID 5. It records byte and perceptual hashes and a reviewer resolves possible duplicates.
2. Suppose the whole group lands in train. The frozen row stores its relative path, hash, ID, group and label. Related photos stay with it.
3. For classification, loading verifies bytes, resizes to 224, replicates three channels and normalizes. Real-only or mixed can draw it. Raw bytes stay unchanged.
4. For LoRA, loading makes a 512 image in `[-1,1]`. The VAE encodes it, noise is added, and the UNet predicts that noise under the scratch prompt.
5. Later generation starts from new seeded noise. The new PNG has a generated path, ID, hash and source manifest/adapter provenance. The core text-to-image record has empty direct parent IDs.
6. A reviewer accepts or rejects the candidate. The approved export preserves the generated file's identity and adds review information. Synthetic-only and mixed can draw it.
7. A different real scratch held in test is scored by every frozen classifier. The real training photograph and the synthetic PNG remain training examples.

## Chapter 7 — Explain the experiment

| Arm | Pool | Question |
| --- | --- | --- |
| A | Real train | How strong is the baseline? |
| B | Approved synthetic | How much can generated samples teach the classifier? |
| C | Both | Does added synthetic data help under our fixed budget? |

#14 supplies the machinery, #15 runs it on the actual pools, and #16 grades the sealed choices. All arms share architecture, initial weight policy, transforms, optimizer budget, seeds and real validation. The proposed mixed batch has 12 real and 12 synthetic images at batch 24. Sampling with replacement means unique file counts differ from exposure counts.

Every arm takes the same real test exam. Synthetic-only still depends on real generator learning and real validation selection. Compare matched seeds and inspect each class. Improvement in scratches may come with worse pitting. Better-looking images can still be repetitive or teach an unhelpful texture. Interpret negative or inconclusive results with the recorded limitations.

## Chapter 8 — Metrics you can calculate

This invented two-class example is for teaching.

| Actual / predicted | Scratch | Pit |
| --- | ---: | ---: |
| Scratch | 3 | 1 |
| Pit | 2 | 4 |

For scratch, true positives are 3, false positives 2, false negatives 1 and true negatives 4. Rows are true classes and columns predictions.

- **Accuracy** is all correct over all tested, `7/10 = 70%`.
- **Precision** asks how many called scratch really were scratches, `3/5 = 60%`.
- **Recall** asks how many actual scratches were found, `3/4 = 75%`.
- **F1** combines the two, `2 × 0.60 × 0.75 / (0.60 + 0.75) ≈ 66.7%`.
- Pit F1 is `8/11 ≈ 72.7%`. **Macro F1** averages the class F1 values, about 69.7% here. Our real metric averages six classes equally.
- **Support** is the true sample count for a class. Undefined scores use zero and absent classes are flagged.

Suppose an arm gets fictional F1 values 0.60, 0.65 and 0.70 across three seeds. Mean is 0.65 and sample SD is 0.05. Compute mixed minus real-only within each seed for a paired difference. Aggregate only the completed matched seeds. One seed has no across-seed SD.

KID uses defaults of 100 seeded draws with 40 examples per side per class and 200 per side pooled. A small negative estimate can occur. Missing weights or too few images produce unavailable status. KID subset SD describes variation over image draws. Classifier seed SD describes variation over training runs.

## Chapter 9 — Running and demonstrating

At home run the CPU environment probe, config validator and pytest commands in the manual. Small fixtures exercise the data paths, three arms and evaluator.

At university verify CUDA, audit the permitted data and freeze the split. Run the real baseline and LoRA pilots. Measure memory/time and inspect samples. Run bounded adaptation, generation, technical validation and both human reviews. Make the quality report, train all planned arm/seed pairs and seal the selection. Only then run final testing. [Exact steps and expected outputs](IMPLEMENTATION.md).

Show an examiner one manifest row, an example rejection, a class grid and a run record. Then calculate recall from a confusion row and explain the paired comparison. Show measured charts when available and label teaching examples. If the GPU runs out of memory, preserve the failure and log, then review the hardware/settings and a bounded smaller pilot. The manual explains which jobs resume.

## Chapter 10 — Scientific honesty

CPU tests establish software behavior on controlled inputs. The GPU pilot measures full-model feasibility and quality. Human review checks visible labels. KID uses natural-image features and can miss subtle industrial distinctions. It cannot establish utility or lack of memorization. The small single-source dataset limits conclusions about new factories and cameras.

Report all planned runs, class shortfalls and failures. A professional negative conclusion can say that under the frozen protocol, mixed training showed no consistent gain across completed seeds, with the per-class evidence explaining the differences. Actual accuracy and improvement claims need real predictions and traceable records.

## Chapter 11 — Teaching and viva

### A two-minute explanation

> A steel factory needs to recognize scratches, pits and other defects. Real images can be difficult to collect. We test whether generated examples help a classifier recognize those defects in unseen real photographs.
>
> We audit NEU, group duplicates and freeze train, validation and test lists. Train teaches weights, validation chooses checkpoints and test stays sealed.
>
> We adapt Stable Diffusion with LoRA. It learns to predict noise added to compressed real training images. Generation starts from seeded noise with a class prompt. Every candidate has a record of its source. Two reviewers check labels and quality.
>
> We train the same ResNet-18 using real images, synthetic images and a mix. They share the training budget and validation rules. We test on identical real photos and compare macro F1 and per-class scores. The software has fixture tests. Real claims require the lab experiment.

### Ten-minute walkthrough split across three teammates

| Time | Person | Explain and show |
| --- | --- | --- |
| 0–3 minutes | Data owner | Problem, classes, audit finding, grouped split and sealed test rule. |
| 3–6.5 minutes | Generation owner | VAE/noise/prompt/UNet, LoRA, candidate record, rejection and quality grid. |
| 6.5–9 minutes | Classification owner | Three arms, equal updates, common validation, one confusion matrix and comparison. |
| 9–10 minutes | All | Implementation evidence, actual measurements and remaining limits. |

### Viva questions

| # | Question | Answer |
| ---: | --- | --- |
| 1 | What is your research question? | Does reviewed synthetic data improve real-test macro F1? |
| 2 | What is a label? | The image's defect category, such as scratches ID 5. |
| 3 | Is NEU imbalanced? | It starts with 300 examples per category. |
| 4 | Does your model locate defects? | Our classifier gives a whole-image category. Localization is a separate task. |
| 5 | Why archive hashes? | To identify the exact acquired bytes. |
| 6 | Why two duplicate checks? | Byte hashes find exact copies and perceptual hashes flag visual similarities. |
| 7 | Why three splits? | Train learns, validation selects, test scores frozen choices. |
| 8 | Why grouped splits? | Related copies in train and test could inflate results. |
| 9 | Why three channels for grayscale? | Pretrained components expect three-channel input. |
| 10 | Does resizing create physical detail? | It interpolates pixels from the original information. |
| 11 | What is a logit? | A raw class score. |
| 12 | What is transfer learning? | Adapting existing learned weights. |
| 13 | Why ResNet-18? | It provides an established, manageable common classifier. |
| 14 | What is a gradient? | A measure of how changing a weight affects loss. |
| 15 | Why equal updates? | Unequal pool sizes make equal epochs different workloads. |
| 16 | What is the VAE? | An image encoder/decoder using compact latents. |
| 17 | What does the UNet predict? | Noise conditioned on latent, timestep and text. |
| 18 | What is the text encoder? | It converts the class prompt into vectors. |
| 19 | What is the scheduler? | It controls noise levels and denoising updates. |
| 20 | Why LoRA? | Small trainable additions reduce update and checkpoint size. |
| 21 | What is gradient accumulation? | Combining small-batch gradients before one optimizer update. |
| 22 | Is generation a transformed photograph? | It starts from noise and learned patterns. Review still checks for memorized-looking copies. |
| 23 | Why store revisions and seeds? | They help identify and reproduce the exact recipe. |
| 24 | Why another reviewer? | To check a fixed sample and disputed labels independently. |
| 25 | Is synthetic-only real-data-free? | Its generator learned from real train and selection uses real validation. |
| 26 | Why macro F1? | It combines precision/recall while weighting classes equally. |
| 27 | Can KID be negative? | The finite-sample unbiased estimate can be negative. |
| 28 | Does low KID imply useful labels? | Label fidelity and classifier utility need their own evidence. |
| 29 | What can one seed show? | One paired comparison, without across-seed SD. |
| 30 | What if the method fails? | Preserve the runs and explain measured limits and negative results. |

### Common mix-ups and glossary

Passing pytest establishes a controlled software example. Prompts express intended labels and reviewers inspect the actual output. A seed must be paired with a matching recipe and environment. Checkpoint selection uses validation; test comes after the choice. KID measures image features, while accuracy counts correct classifier predictions.

| Term | Meaning |
| --- | --- |
| Artifact | A saved output such as a checkpoint, manifest or chart. |
| Manifest | A table linking IDs to file paths and metadata. |
| Provenance | Recorded origin and processing history. |
| SHA-256 | Exact-byte fingerprint. |
| Leakage | Held-out information influencing learning or selection. |
| Tensor | Numerical array, such as batch × channels × height × width. |
| Normalization | Scaling values to a model's expected range. |
| Latent | Compact image representation. |
| Adapter | Small learned additions to base weights. |
| Checkpoint | Saved weights, with separate recovery state where supported. |
| VRAM | GPU memory. |
| Fixture | Small authored test input. |
| Arm | An experimental training condition. |
| Support | True sample count per class. |
| Matched seed | Seed completed in every compared arm. |
