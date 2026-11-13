# University lab implementation manual

Verified against `main` at `88c0d13`, which contains KID PR #43. Final classifier evaluation PR #40 merged into the #14 feature branch after PR #39 merged that branch into `main`. Section 9 gives the verified #16 command and its integration gate. GitHub issues #1–#20 were open at inspection. This manual describes runnable code, fixture evidence, and lab work still to do.

## 1. Set up the lab PC

Use Git, Python 3.11, a compatible NVIDIA GPU and driver for full diffusion runs, sufficient disk for local snapshots and checkpoints, and approved team storage. The project requests **12–16 GiB VRAM** as a booking estimate for 512 px LoRA and **10 GiB free disk** as an initial allowance for one base pipeline and pilot artifacts. Measure actual requirements. Record OS, GPU, driver, free and total VRAM, session length and storage quota. The CPU environment is suitable for tests, dataset preparation and review.

Run PowerShell from the repository root. If starting fresh, clone from the team's remote and enter the checkout. Record the exact reviewed revision for scientific runs.

```powershell
git clone https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-.git industrial-defect-diffusion
Set-Location industrial-defect-diffusion
git switch main
git pull --ff-only
git rev-parse HEAD
git status --short
py -3.11 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py --version
```

For CPU, install the pair used in CI. On the lab GPU, first run the diagnostics below and choose the **matching** torch and torchvision CUDA build listed by PyTorch for the OS, GPU and driver. The last install line is conditional on verified CUDA 12.6 compatibility. Run only the install for your environment. The CUDA value displayed by the driver and the CUDA build inside PyTorch are separate records. [PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/), [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/).

```powershell
# CPU environment
& $py -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu

# GPU diagnostics and conditional CUDA 12.6 example
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
& $py -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

```powershell
& $py -m pip install -e '.[dev,audit,lora]'
& $py -m pip check
& $py scripts/check_ml_environment.py --device cpu
# On the lab GPU environment use --device cuda instead.
New-Item -ItemType Directory -Force artifacts | Out-Null
& $py -m pip freeze --all | Set-Content artifacts/environment-packages.txt
& $py scripts/check_foundation.py
& $py -m ruff check .
& $py -m ruff format --check .
$env:OMP_NUM_THREADS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $py -m pytest -q --strict-config --strict-markers
```

Linux uses `python3.11 -m venv .venv` and `.venv/bin/python`. Use the same Python arguments, `export NAME=value` for environment variables and `sha256sum FILE` for file hashes. The Windows CPU package snapshot is in [requirements](requirements/ml-cpu-win-py311.txt). The environment probe imports the ML stack and checks for six finite scores from a tiny ResNet input. **Proceed to a lab pilot only after the CUDA check passes and the session and hardware are recorded.** See the [environment guide](docs/planning/setup.md).

## 2. Acquire, audit and freeze the dataset

Use the official NEU **classification** collection. The publisher describes 1,800 grayscale 200×200 images, 300 per class. The source ledger still records unresolved reuse terms and no verified local archive. Confirm intended academic use with the provider if the archive terms do not cover it. Record source URL, access date, archive name, checksum, terms and citation. Preserve the original archive and use a read-only extracted image root. A mirror's train/test folders do not define this study's split. [Official NEU page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/), [project source ledger](data/README.md).

The image root can have flat files such as `Sc_1.bmp` or one level of class folders. Point the command at the actual image directory, inside any wrapper. IDs are crazing 0, inclusion 1, patches 2, pitted_surface 3, rolled_in_scale 4, scratches 5. Source aliases are Cr, In, Pa, PS, RS and Sc.

```powershell
$archive = 'data/local/raw/NEU-CLS.rar'
$real = 'data/local/raw/NEU-CLS'
$audit = 'artifacts/neu-audit.json'
$split = 'artifacts/neu-split-v1'
$manifest = "$split/manifest.csv"
$archiveHash = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
& $py scripts/audit_dataset.py --root $real --report $audit --json-errors
& $py scripts/freeze_splits.py --audit $audit --audit-digest
```

The audit exits 0 for clean checks, 1 for findings requiring review, 2 for input/execution errors. If it returns 2, resolve the error before reading the digest. The JSON contains per-file decode, mode, size, label and digest findings. SHA-256 finds identical bytes. The 64-bit difference hash flags similar brightness patterns for manual review. If a trusted source inventory exists, pass `--inventory artifacts/expected-paths.json` to both audit and freeze. The inventory is a JSON array of relative paths. Otherwise a shortage gives a count without identifying the missing filename. If you change the audit threshold, regenerate the report and review together.

Prepare `artifacts/neu-review.json` using the canonical audit digest printed above. Every flagged pair needs a `same` or `different` decision. Exact duplicates stay grouped. Explicitly resolve label conflicts, invalid files, capture siblings and inventory findings. Use relative paths copied from the audit. The empty collections below are only a schema example.

```json
{
  "schema_version": 1,
  "audit_sha256": "<canonical audit digest>",
  "reviewer": "<peer reviewer>",
  "inventory_note": "<explain applicable count, structure and inventory findings>",
  "exclusions": {},
  "near_duplicates": [],
  "source_groups": []
}
```

Exclusions map paths to reasons. Near-duplicate entries contain `paths` and `decision`. A source group is an array of paths. Retain the reviewed audit and exclusions, then run:

```powershell
& $py scripts/freeze_splits.py --root $real --audit $audit --review artifacts/neu-review.json --archive-sha256 $archiveHash --output $split
$manifestHash = (Get-FileHash -Algorithm SHA256 $manifest).Hash.ToLowerInvariant()
Get-Content "$split/split-metadata.json"
```

Outputs are `manifest.csv` and adjacent `split-metadata.json`. The default seed is 42 with approximately 70/15/15 per class. Whole duplicate groups move together, so verify actual counts. The freeze reruns the audit and an identical invocation checks the saved output byte for byte. Verify archive, audit and manifest fingerprints, and get peer approval of the real split before training. Keep the test partition sealed. Record a new reviewed version if any inputs change.

## 3. Prepare local model assets and configuration

| Asset | What to provision and verify |
| --- | --- |
| Classifier | Approved torchvision ResNet-18 `IMAGENET1K_V1` local state dict, with source, applicable terms and SHA-256. Training replaces the original output head with six classes. Smoke tests initialize from scratch. [Official weights documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). |
| Base diffusion | `stable-diffusion-v1-5/stable-diffusion-v1-5` revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. Provision Diffusers components `unet/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/` and their config/tokenizer files. The three weighted components need matching local safetensors. Review CreativeML Open RAIL-M terms and retain component hashes. [Model decision](docs/research/diffusion-selection.md). |
| LoRA adapter | Produced in section 4. Retain the whole step directory. Approve the selected checkpoint JSON and verify the adapter hash recorded inside. |
| Inception | Optional local torch-fidelity compatibility file `weights-inception-2015-12-05-6726825d.pth`. Verify source, terms and SHA-256 before providing it. Ordinary torchvision Inception weights use a different path. [Pinned extractor source](https://raw.githubusercontent.com/toshas/torch-fidelity/v0.3.0/torch_fidelity/feature_extractor_inceptionv3.py). |

Use ignored `models/` and `artifacts/` directories. The repository MIT license covers original code. Review data, model, adapter, generated image and reporting rights separately. SD v1.4 is a documented source fallback accepted by the trainer. Generation and review currently pin v1.5, so v1.4 needs a reviewed compatibility change for an end-to-end run.

Copy `configs/baseline.yaml` into ignored storage. Set `paths.data_root` to the image root, `paths.manifest` to the frozen CSV, and `paths.artifact_root` to an ignored root within the checkout. Review preprocessing, device and budgets. Set `status: approved` only after peer review. A smoke configuration uses tiny fixture data, CPU and no pretrained classifier weights.

```powershell
Copy-Item configs/baseline.yaml artifacts/approved-baseline.yaml
& $py scripts/check_experiment_config.py configs/cpu-smoke.yaml
& $py scripts/check_experiment_config.py artifacts/approved-baseline.yaml
```

The validator reports configuration errors and starts no training. Config, split hash and asset approval are separate checks. See [configuration and records](docs/experiments/configuration.md).

## 4. Train the real baseline and pilot LoRA

For a baseline pilot, use a reviewed local copy of the config with a lab-sized budget. Use the frozen real validation split for checkpoint selection. Section 8 reruns or uses a baseline produced with the exact final comparison configuration.

```powershell
$resnet = 'models/resnet18-f37072fd.pth'
& $py scripts/train_real_baseline.py artifacts/approved-baseline.yaml --seed 42 --approved-manifest-sha256 $manifestHash --local-weights $resnet
```

Verify the arm directory under `artifacts/classification/` has `best.pt`, `training.jsonl` and `run.json`. Check the validation IDs, class counts, initialization digest, completed update count and earliest best validation macro F1. The classifier checkpoint is for inference and has no optimizer-resume state.

For LoRA, copy the approved config to `artifacts/approved-pilot.yaml`. Set `diffusion.max_steps: 100`, `epochs: null`, and a checkpoint interval shorter than the pilot, for example 25. A one-update smoke attempt can use its own reviewed config. Start with the booked device, batch one, accumulation four, checkpointing and reviewed precision. Set `--variant fp16` only when all three weight files use that variant. File variant and compute precision are distinct.

```powershell
$sd = 'models/sd-v1-5'
& $py scripts/check_experiment_config.py artifacts/approved-pilot.yaml
& $py scripts/train_lora.py artifacts/approved-pilot.yaml --model-dir $sd --approved-manifest-sha256 $manifestHash
```

In a separate terminal, sample device-wide usage and keep the log. Stop polling with Ctrl+C after the run.

```powershell
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free --format=csv -l 1 | Tee-Object artifacts/pilot-gpu.csv
```

Expect `resolved-config.yaml`, `run.json`, and step folders containing `adapter.safetensors`, `checkpoint.json` and `training-state.pt`. Confirm finite losses, correct device, completed updates, adapter reload and resume from an intermediate checkpoint. Measure seconds per update and per generated image. The CLI records losses and environment. PyTorch peak allocated/reserved VRAM and synchronized timing require lab instrumentation, which the current CLI does not export. Periodic prompt grids are also a separate lab step using section 5 on selected checkpoints. The [pilot decision](docs/research/diffusion-selection.md) calls for at least 1 GiB device-wide headroom, acceptable samples across classes and a time projection with a 20% buffer.

After the go decision, use a fresh reviewed full config with the bounded budget, proposed at 1,000 updates, and periodic checkpoints, proposed every 250. The same command runs with the full config path. Resume is for the **same** configuration and environment:

```powershell
# After reviewing artifacts/approved-lora-full.yaml with the final budget:
& $py scripts/train_lora.py artifacts/approved-lora-full.yaml --model-dir $sd --approved-manifest-sha256 $manifestHash

# To resume the 100-update pilot after interruption:
$intermediate = 'artifacts/diffusion/REPLACE-WITH-RUN/step-000025'
& $py scripts/train_lora.py artifacts/approved-pilot.yaml --model-dir $sd --approved-manifest-sha256 $manifestHash --resume $intermediate
```

Resume uses trusted optimizer state and checks code commit, settings, manifest, variant, device and base files. It creates a new run directory. A 100-step pilot checkpoint does not resume into a changed 1,000-step configuration. See the [LoRA guide](docs/experiments/lora-training.md).

## 5. Generate candidates

Select a completed adapter step after review. Use the same approved v1.5 snapshot, variant, and frozen split. In a reviewed generation YAML, choose precision, batch size, seed, counts, steps, guidance and resolution. Baseline defaults are 166 per class (996 total), 30 inference steps, guidance 7.5, seed 42. For a pilot contact sheet, use a separate reviewed small count plan covering all six classes. Each class uses the training prompt catalog, and each image receives a deterministic seed and ID.

```powershell
$adapter = 'artifacts/diffusion/REPLACE-WITH-RUN/step-001000'
$adapterApproval = (Get-FileHash -Algorithm SHA256 "$adapter/checkpoint.json").Hash.ToLowerInvariant()
$generated = 'artifacts/generation/neu-v1'
& $py scripts/generate_defects.py artifacts/approved-generation.yaml --model-dir $sd --adapter-dir $adapter --approved-adapter-sha256 $adapterApproval --approved-manifest-sha256 $manifestHash --output $generated
```

The hash argument covers the checkpoint JSON. The program verifies the adapter bytes and base component fingerprints against it. Outputs are `plan.json`, numbered batch folders with PNGs and batch metadata, and `candidates.jsonl` with prompt, seed, hashes, intended class and pending review. Check counts, unique IDs and source hashes. Repeat the identical command with `--resume` after interruption. Existing committed batches must pass byte verification. If a killed process leaves a lock, confirm it has stopped before removing the stale lock. Use a new output directory for a changed plan. See the [generation guide](docs/experiments/generate-candidates.md).

## 6. Audit and review generated images

Keep the original real audit and candidate artifacts. Choose the perceptual threshold before review. The CPU screen verifies PNGs, dimensions, channels, IDs, hashes, prompts and ancestry. It checks near matches against frozen real fingerprints. Held-out similarity blocks eligibility and the audit exposes no held-out pixels, IDs or distances. Keep those flags with the data custodian and avoid using them to tune the generator.

```powershell
$reviewInputs = @('--generation', $generated, '--manifest', $manifest, '--real-audit', $audit, '--approved-manifest-sha256', $manifestHash)
& $py scripts/review_synthetic.py audit @reviewInputs --output artifacts/review/first
& $py scripts/review_synthetic.py second-review @reviewInputs --primary artifacts/review/first/primary.json --output artifacts/review/second
& $py scripts/review_synthetic.py export @reviewInputs --primary artifacts/review/first/primary.json --secondary artifacts/review/second/secondary.json --output artifacts/review/final
```

Pause between commands. First inspect class sheets and record every decision in `primary.json`, with a reviewer ID, timezone-aware timestamp and reasons for rejections or accepted flags. Flag disputed labels. Review texture, label fidelity, duplicates and artifacts. Keep at most one exact copy. Next, a different reviewer fills `secondary.json` for the seeded sample of up to ten accepted per class and all disputes. Resolve disagreement before export. If the primary changes, regenerate the second template in a fresh directory. Use the same threshold on every step.

The final directory contains `approved.jsonl`, `reviewed.jsonl`, `review-summary.json`, both decisions and the audit. Verify accepted counts, rejection reasons, class coverage and recorded shortfalls. Review the accepted export hash. The downstream classifier needs all six classes, and the team must separately enforce the approved scientific quota. The proposed target is 600 accepted and the reduced core target is 300, balanced by class. [Review workflow](docs/experiments/synthetic-review.md).

## 7. Measure exploratory quality

For KID, install the pinned optional package and provide verified local weights. Without the weight arguments the command still writes qualitative outputs. Use a separate output path for each assessment.

```powershell
& $py -m pip install -e '.[quality]'
$inception = 'models/weights-inception-2015-12-05-6726825d.pth'
$inceptionHash = '<independently approved SHA-256>'
& $py scripts/assess_synthetic_quality.py --generation $generated --real-root $real --manifest $manifest --real-audit $audit --review artifacts/review/final --approved-manifest-sha256 $manifestHash --output artifacts/quality/neu-v1 --weights $inception --approved-weights-sha256 $inceptionHash
```

Outputs are `quality.json`, `quality.md` and six class grids. Confirm the reference is frozen real **train**, the synthetic pool is accepted-only, and the output lists counts, seed, feature settings, hashes and unavailable reasons. Defaults use 100 draws, seed 42, subset size 40 per class and 200 pooled. A pool smaller than the required subset or missing feature weights yields unavailable statistics. Inspect reviewer observations, nearest training examples and rejections. KID measures feature-distribution difference and can be negative as a finite-sample estimate. Its subset SD is descriptive. It gives no label verdict or classifier utility result. [Quality workflow](docs/experiments/synthetic-quality.md).

## 8. Run the three classifier arms

Freeze the shared config and its seed list before the comparison. Use the same ImageNet starting weights, real validation rows, transform, optimizer, update budget and checkpoint rule for all three arms. The proposed design uses 1,000 updates, batch 24, 50:50 mixed exposure, uniform class exposure within each source, and seeds 42/43/44. A one-seed reduced study needs the same seed across all arms and advance agreement. Real-only sees real train, synthetic-only sees accepted synthetic, and mixed sees both. Sampling with replacement means exposure counts and unique files differ.

```powershell
$accepted = 'artifacts/review/final/approved.jsonl'
$acceptedHash = (Get-FileHash -Algorithm SHA256 $accepted).Hash.ToLowerInvariant()
$comparison = 'artifacts/approved-comparison.yaml'
# Prepare this reviewed copy from baseline.yaml with the final seeds and budget.
& $py scripts/check_experiment_config.py $comparison
foreach ($seed in @(42,43,44)) {
    & $py scripts/train_real_baseline.py $comparison --seed $seed --arm all --approved-manifest-sha256 $manifestHash --local-weights $resnet --synthetic-root $generated --synthetic-manifest $accepted --approved-synthetic-manifest-sha256 $acceptedHash
    if ($LASTEXITCODE -ne 0) { throw "Comparison failed for seed $seed. Preserve its records." }
}
```

Adjust the loop to the approved seed list. A single arm can be selected with `--arm real_only`, `synthetic_only` or `real_plus_synthetic`. Use the **generation directory** as synthetic image root. Verify separate `best.pt`, `training.jsonl` and `run.json` per arm and seed. Compare train source counts, real validation IDs, update count, initialization and hashes. Record failures and all planned arm/seed pairs in the #15 inventory. The repo supplies orchestration for one seed per invocation. Selection inventory and quota approval are lab tasks. [Three-arm guide](docs/experiments/classifier-arms.md).

## 9. Evaluate frozen checkpoints

**Integration gate.** PR #40 contains the evaluator and its fixture tests, but `main` at this audit lacks them. For code inspection, use the command below. For scientific execution, wait for reviewed integration onto current main or a team-approved compatible checkout containing the evaluator and upstream stages. Preserve the training commit in the run records.

```powershell
git show origin/feat/16-classifier-evaluation:docs/experiments/classifier-evaluation.md
```

After #15, build `artifacts/selection.json` with `schema_version: 1`, `manifest_sha256`, and a `runs` array. Each entry has `arm`, `seed`, workspace-relative `path` to the exact `run.json` and that file's `sha256`. Include available failed records. Missing pairs are reported. Have a peer approve the selection hash before anyone views test metrics.

```json
{
  "schema_version": 1,
  "manifest_sha256": "<approved real manifest hash>",
  "runs": [
    {
      "arm": "real_only",
      "seed": 42,
      "path": "artifacts/classification/REPLACE-WITH-RUN/run.json",
      "sha256": "<exact run.json file hash>"
    }
  ]
}
```

The example shows one entry. Add the remaining configured arm/seed entries from the reviewed run inventory.

```powershell
$selection = 'artifacts/selection.json'
$selectionHash = (Get-FileHash -Algorithm SHA256 $selection).Hash.ToLowerInvariant()
& $py scripts/evaluate_classifier.py $comparison --selection $selection --approved-selection-sha256 $selectionHash --approved-manifest-sha256 $manifestHash --output artifacts/evaluation/frozen-comparison
```

The evaluator verifies common configuration, provenance, validation IDs, checkpoint compatibility and the frozen test manifest. Every completed checkpoint predicts the same real test IDs once. It writes per-arm/seed predictions, per-class metrics, confusion SVGs, `comparison.csv`, `comparison.md`, `performance.svg` and `evaluation.json`. Exit 0 means all expected runs are complete, 1 means missing or failed runs, and 2 indicates invalid inputs or artifacts. Check that confusion counts sum to test support and macro F1 is reported alongside individual class results. Use matched seeds for comparisons and sample SD only when at least two matched seeds exist. Use the [results template](reports/results-template.md) for #17. Final report numbers must point to run artifacts.

## 10. Recover and reproduce

| Symptom | Action and verification |
| --- | --- |
| CUDA unavailable | Compare driver, wheel build and device index with the lab record. Rerun the CUDA probe. A successful CPU check establishes CPU behavior. |
| GPU out of memory | Save the failed record and memory log. Ask the lab to free unrelated workloads or book a larger device. Verify batch one, checkpointing and compatible precision. The proposed fallback is a reviewed 256 px pilot with a fresh config and sample inspection. Recalculate runtime and headroom. |
| Missing weights or wrong variant | Confirm the selected safetensors and configurations are present locally. Verify source and digest. A missing Inception file still allows a qualitative report. |
| Dataset or manifest mismatch | Compare archive, canonical audit, manifest and per-file SHA-256 values with the approved copies. Restore the correct source or repeat audit, review and split freeze with a new version. |
| Interrupted work | Resume LoRA from a compatible completed step. Resume generation with its identical plan. Classifier restarts are new attempts. Preserve failed records. |
| Different results on repeat | Compare code commit, data/asset hashes, config, seeds, package lock, device, precision, batch and deterministic settings. Record reproducibility limits across machines. |

Before handing off, have another teammate recreate the environment, run the CPU tests, verify the hashes, and replay affordable frozen outputs. Preserve the raw archive, audit, split approval, generation plan, both reviews, configuration and logs in controlled storage. Separate fixture results, real pilot findings and final measurements in the report. Optional inpainting is outside this manual's runnable path. Use the [protocol](docs/experiments/protocol.md) and [license ledger](docs/research/licensing.md) when approving deviations or sharing artifacts.
