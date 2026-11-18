# Run the project at the university lab

For a menu that calls the commands below, start with [LAB_QUICKSTART.md](LAB_QUICKSTART.md) and `./run_lab.ps1`. This manual remains the detailed reference for assets, review decisions and experiment settings.

**Use this file at the GPU PC.** Your juniors can start with [PROJECT_GUIDE.md](PROJECT_GUIDE.md). You can follow this guide with PowerShell, a text editor, and a browser. It includes the commands for training, review, and final testing.

**Read each step before pasting its commands.** Run one block at a time. If a block contains `notepad`, paste through that line, save the file, then paste the remaining lines. If a command prints an error, stop at that step and use the help at the bottom. Paths assume you are inside the cloned repository. Replace `RUN-FOLDER` with the actual folder name printed by the trainer.

## 1. Clone and check the lab PC

1. Before going to the lab, verify that PR #44 has been merged and you can access the repository from the lab. Arrange a GPU booking, disk space, and approved storage for the files you will take away. Ask the lab for permission to install Python packages, or arrange the environment in advance.
2. On the lab PC, open **PowerShell** in a writable folder. Run:

```powershell
git --version
py -3.11 --version
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

You should see a Git version, Python 3.11, and the NVIDIA GPU and its memory. If a command is missing, ask the lab administrator to install or locate it. The project uses **12–16 GiB VRAM** as a booking estimate for 512 px LoRA. Actual use and speed are still unverified. Reserve at least 10 GiB initially for the selected base snapshot and pilot files, plus space for packages, data, and additional run artifacts.

3. Clone and enter the repository. Check that the guides and final evaluation script are present.

```powershell
git clone https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-.git industrial-defect-diffusion
Set-Location industrial-defect-diffusion
git rev-parse HEAD
Get-Item IMPLEMENTATION.md, PROJECT_GUIDE.md, scripts/evaluate_classifier.py
```

If access is denied, sign in to GitHub through the lab's approved Git authentication method. If the files are missing, check the cloned branch and whether the PR merged. Record the commit printed by Git. Keep that checkout fixed for the experiment.

## 2. Install and test Python

Create a new virtual environment. This makes an isolated set of Python packages.

```powershell
py -3.11 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py --version
```

**At the GPU lab**, use the CUDA wheel index that matches the lab's driver and GPU. The next command is the CUDA 12.6 example for the pinned Torch pair. Confirm the index with the lab before running it. The number in `nvidia-smi` describes driver support, and the Python probe below reports the runtime build. [PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/), [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/).

```powershell
& $py -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

**At home on CPU**, use this install instead. Keep the CPU and GPU environments separate.

```powershell
& $py -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

On either PC:

```powershell
& $py -m pip install -e '.[dev,audit,lora]'
& $py -m pip check
New-Item -ItemType Directory -Force artifacts, models, data/local/raw | Out-Null
& $py -m pip freeze --all | Set-Content artifacts/environment-packages.txt
```

At the lab, run the GPU probe. Proceed to GPU training only when it prints `cuda smoke check passed`.

```powershell
& $py scripts/check_ml_environment.py --device cuda
```

Run the software tests before using real images. These use small local fixtures. You can use the same block at home. On Windows, the fixture suite exercises Git as well as Python.

```powershell
$previousGpuVisibility = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $py scripts/check_ml_environment.py --device cpu
& $py scripts/check_foundation.py
& $py -m ruff check .
& $py -m ruff format --check .
& $py -m pytest -q --strict-config --strict-markers
# Restore normal GPU visibility after the CPU tests.
if ($null -eq $previousGpuVisibility) {
    Remove-Item Env:CUDA_VISIBLE_DEVICES
} else {
    $env:CUDA_VISIBLE_DEVICES = $previousGpuVisibility
}
```

The tests do not need NEU or pretrained weights. On Linux, use `python3.11 -m venv .venv` and `.venv/bin/python`; these examples otherwise use PowerShell.

## 3. Put the real images and model files in the right places

**Permission first.** The source ledger records unresolved terms for NEU. Confirm written permission for the intended use, inspect the archive terms, and record the source, acquisition date, terms and citation. Use **NEU-CLS**, the image classification archive. Get it from the link on the official source page once permitted. [Official NEU source](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/), [project data ledger](data/README.md).

Use File Explorer or the lab's archive tool to place the original archive and extract a copy. Keep the originals read-only. Open the extracted folder and find the directory that directly contains the images or one level of class folders. Set `$real` to that directory if its path differs from the example below. Names such as `Sc_1.bmp` and class folders such as `scratches` are supported. Existing mirror train/test folders and extra wrappers need layout review.

| Put it here | What you need |
| --- | --- |
| `data/local/raw/NEU-CLS.rar` | Original approved archive. |
| `data/local/raw/NEU-CLS/` | Extracted image root. Expect 1,800 images, 300 per class, before any reviewed exclusions. |
| `models/resnet18-f37072fd.pth` | Approved local torchvision ResNet-18 `IMAGENET1K_V1` state dict. Record the full hash and terms. [Torchvision weights documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). |
| `models/sd-v1-5/` | Approved Diffusers snapshot of `stable-diffusion-v1-5/stable-diffusion-v1-5` revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. It must contain `unet`, `vae`, `text_encoder`, `tokenizer` and `scheduler` directories with their local configs and files. [Model decision and terms](docs/research/diffusion-selection.md). |
| `models/weights-inception-2015-12-05-6726825d.pth` | Optional local torch-fidelity feature weights for KID, acquired and verified separately. [Pinned extractor source and weight URL](https://raw.githubusercontent.com/toshas/torch-fidelity/v0.3.0/torch_fidelity/feature_extractor_inceptionv3.py). |

The training and generation commands load local model files. Obtain the assets through approved lab storage or source and verify their terms separately from the repository's MIT license. Stable Diffusion needs safetensors for the UNet, VAE and text encoder. See the variant check in step 6. LoRA adapter files are **created** in step 6.

### Optional online asset preparation

Skip this subsection if the lab has already provided approved files. Run it only on a machine with permitted internet access after reviewing the sources and terms. These commands transfer large files. Complete preparation before the booked training session. The guide author has not downloaded or verified the real model assets.

Download the approved ImageNet V1 state dict from torchvision's own weight URL:

```powershell
@'
from pathlib import Path
import torch
from torchvision.models import ResNet18_Weights

destination = Path("models/resnet18-f37072fd.pth")
if destination.exists():
    raise FileExistsError("Keep the existing file and check its source and hash")
torch.hub.download_url_to_file(
    ResNet18_Weights.IMAGENET1K_V1.url,
    str(destination),
    hash_prefix="f37072fd",
)
'@ | & $py -
if ($LASTEXITCODE -ne 0) { throw 'Classifier weight preparation failed.' }
Get-FileHash -Algorithm SHA256 models/resnet18-f37072fd.pth
```

For SD, this selects one pinned local component set with default safetensors. It excludes other weight formats and variants. If access is gated, complete the provider's access procedure through its approved tools and keep credentials out of the repository.

```powershell
$env:HF_HUB_OFFLINE = '0'
@'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    local_dir="models/sd-v1-5",
    allow_patterns=[
        "model_index.json", "README.md", "LICENSE*",
        "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json", "text_encoder/model.safetensors",
        "tokenizer/*.json", "tokenizer/*.txt", "scheduler/*.json",
    ],
)
'@ | & $py -
$assetExitCode = $LASTEXITCODE
$env:HF_HUB_OFFLINE = '1'
if ($assetExitCode -ne 0) { throw 'SD preparation failed. Resolve access or missing files first.' }
Get-ChildItem models/sd-v1-5 -Recurse -Filter '*.safetensors' | Get-FileHash -Algorithm SHA256
```

Save the source URLs, pinned revision and actual hashes with the asset approval. For optional Inception, open the pinned extractor source linked in the table and use its `URL_INCEPTION_V3` release file. Save it under the exact table filename. A peer must verify that source and the full downloaded SHA-256 before KID. You can complete the qualitative report without it.

Run this block again after reopening PowerShell from the repository root. These are convenience names for paths.

```powershell
$py = '.\.venv\Scripts\python.exe'
$archive = 'data/local/raw/NEU-CLS.rar'
$real = 'data/local/raw/NEU-CLS'
$audit = 'artifacts/neu-audit.json'
$split = 'artifacts/neu-split-v1'
$manifest = "$split/manifest.csv"
$resnet = 'models/resnet18-f37072fd.pth'
$sd = 'models/sd-v1-5'
$baseCfg = 'artifacts/lab-base.yaml'
$pilotCfg = 'artifacts/lab-lora-pilot.yaml'
$fullCfg = 'artifacts/lab-lora-full.yaml'
$compareCfg = 'artifacts/lab-comparison.yaml'
$generated = 'artifacts/generation/neu-v1'
$reviewFinal = 'artifacts/review/final'
$accepted = "$reviewFinal/approved.jsonl"
```

Check your files and record these fingerprints in the team notes. A computed hash identifies the file. Asset approval comes from your review of the source and terms.

```powershell
Get-Item $archive, $real, $resnet, $sd
Get-FileHash -Algorithm SHA256 $archive, $resnet
Get-ChildItem $real | Select-Object -First 12 Name
```

## 4. Audit the images and freeze the three lists

1. Run the audit. It writes a new JSON file. Exit 0 means checks are clean, 1 means findings need review, and 2 means the command failed.

```powershell
& $py scripts/audit_dataset.py --root $real --report $audit --json-errors
$LASTEXITCODE
```

2. Open `artifacts/neu-audit.json`. Check missing files or class shortages, unreadable images, wrong dimensions and conflicting labels. SHA-256 finds identical files. The difference hash flags visually similar pairs. Review **every** suggested pair, even when one file will be excluded. A pair is `same` if it is a confirmed duplicate and `different` otherwise. Exact copies are grouped automatically. If you have a trusted inventory of expected paths, pass the same `--inventory FILE` to audit and freeze.

3. Print the audit fingerprint, then create the review file in a text editor:

```powershell
& $py scripts/freeze_splits.py --audit $audit --audit-digest
notepad artifacts/neu-review.json
```

Paste this structure. Replace the digest and reviewer, fill an inventory note if applicable, and add only real decisions from your audit. Exclusions map paths to reasons. A near-pair entry looks like `{"paths":["first/path.bmp","other/path.bmp"],"decision":"same"}`. A source group is an array of two or more known related paths. Save as JSON.

```json
{
  "schema_version": 1,
  "audit_sha256": "PASTE THE AUDIT DIGEST",
  "reviewer": "ENTER REVIEWER NAME",
  "inventory_note": "",
  "exclusions": {},
  "near_duplicates": [],
  "source_groups": []
}
```

4. Freeze, then inspect the counts and fingerprints. The command rereads and verifies the audit. The target is about 70% train, 15% validation and 15% test. Whole groups stay together, so exact percentages can differ.

```powershell
$archiveHash = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
& $py scripts/freeze_splits.py --root $real --audit $audit --review artifacts/neu-review.json --archive-sha256 $archiveHash --output $split
Get-Content "$split/split-metadata.json"
(Get-FileHash -Algorithm SHA256 $manifest).Hash.ToLowerInvariant()
```

**Stop here for peer review.** Ask a teammate to check the archive hash, group decisions, actual class counts and that the test set is sealed. Save the approved **manifest CSV** hash in your lab notes. The manifest hash and archive hash are different. Before a training, generation, review or testing session, paste the approved hash when prompted:

```powershell
$manifestHash = (Read-Host 'Paste the peer-approved manifest.csv SHA-256').Trim().ToLowerInvariant()
if ((Get-FileHash -Algorithm SHA256 $manifest).Hash.ToLowerInvariant() -ne $manifestHash) {
    throw 'Manifest changed. Restore the approved copy or review a new freeze.'
}
```

## 5. Make the base config and run a real-only pilot

Copy the config once. In Notepad, set `paths.data_root` to the actual `$real` value, `paths.manifest` to `artifacts/neu-split-v1/manifest.csv`, and `paths.artifact_root` to `artifacts`. Use forward slashes in YAML paths. Review the settings with your team and change the top-level `status: proposed` to `status: approved`. The validator checks the file structure. It does not train.

```powershell
Copy-Item configs/baseline.yaml $baseCfg
notepad $baseCfg
& $py scripts/check_experiment_config.py $baseCfg
```

For a short classifier pilot, make the copy below. Under **classification**, use `max_steps: 10`, `validation_interval: 5`, and `epochs: null`. Keep `weights: IMAGENET1K_V1`. Save and run. This pilot teaches the workflow and checks the real loader and checkpoint. It is separate from the final comparison.

```powershell
Copy-Item $baseCfg artifacts/lab-real-pilot.yaml
notepad artifacts/lab-real-pilot.yaml
& $py scripts/check_experiment_config.py artifacts/lab-real-pilot.yaml
& $py scripts/train_real_baseline.py artifacts/lab-real-pilot.yaml --seed 42 --arm real_only --approved-manifest-sha256 $manifestHash --local-weights $resnet
```

Open the printed run folder. Expect `training.jsonl`, `run.json` and `best.pt`. Check that the run completed and that the checkpoint was chosen on real validation macro F1. The real test set remains sealed.

## 6. Pilot LoRA, then run bounded adaptation

1. Check the GPU probe again. Inspect the SD snapshot:

```powershell
Get-ChildItem $sd -Directory
Get-ChildItem "$sd/unet", "$sd/vae", "$sd/text_encoder" -Filter '*.safetensors' | Get-FileHash -Algorithm SHA256
```

The default uses `unet/diffusion_pytorch_model.safetensors`, `vae/diffusion_pytorch_model.safetensors`, and `text_encoder/model.safetensors`. If all three are the `.fp16.safetensors` variant instead, set `$variant = @('--variant', 'fp16')`. Otherwise keep the empty default below. This flag describes filenames. The YAML precision field describes training arithmetic. The end-to-end generation workflow uses SD v1.5.

2. In the pilot config's **diffusion** section set `max_steps: 100`, `epochs: null`, `checkpoint_interval: 25`, and confirm `batch_size: 1`, `gradient_accumulation: 4`, `gradient_checkpointing: true`, `device: cuda`, and reviewed precision. Start with `fp16` if supported. Use a separate config for any one-update smoke attempt. Choose exactly one variant line in the block.

```powershell
$variant = @()
# If the three component files all use .fp16.safetensors, run this instead:
# $variant = @('--variant', 'fp16')
Copy-Item $baseCfg $pilotCfg
notepad $pilotCfg
& $py scripts/check_experiment_config.py $pilotCfg
```

3. In a **second PowerShell window**, enter the same checkout and start the memory log. Use Ctrl+C in that window to stop it after the pilot. In the **first window**, run the training command.

```powershell
# Second window
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free --format=csv -l 1 | Tee-Object artifacts/pilot-gpu.csv
```

```powershell
# First window
$pilotStart = Get-Date
& $py scripts/train_lora.py $pilotCfg --model-dir $sd --approved-manifest-sha256 $manifestHash @variant
if ($LASTEXITCODE -ne 0) { throw 'Pilot failed. Save the error and the GPU log.' }
$pilotSeconds = ((Get-Date) - $pilotStart).TotalSeconds
"Total pilot seconds: $pilotSeconds"
"Approximate seconds per update including setup and saves: $($pilotSeconds / 100)"
```

Look under `artifacts/diffusion/`. The printed run record should show completed training and finite loss values. A step directory has `adapter.safetensors`, `checkpoint.json`, and `training-state.pt`. Measure wall time, seconds per update, and free memory during the run. The GPU log samples device-wide use. The CLI does not export PyTorch peak allocated/reserved memory. Use the lab's instrumentation if collecting those additional measurements.

4. Verify recovery using an **intermediate** step. Replace the example path with the actual **pilot** run path. The resume creates a new run directory and uses exactly the same config, code, weights and environment.

```powershell
$intermediate = 'artifacts/diffusion/RUN-FOLDER/step-000025'
& $py scripts/train_lora.py $pilotCfg --model-dir $sd --approved-manifest-sha256 $manifestHash @variant --resume $intermediate
```

5. Use step 7 once with the pilot's `step-000100` to check that the adapter reloads and samples across the six classes. The pilot guide asks for at least 1 GiB of device-wide memory headroom and a runtime estimate that fits the booking with a 20% buffer. Inspect the pilot images, including failures. [Pilot criteria](docs/research/diffusion-selection.md).

6. After the team agrees to proceed, copy the base config to the full run config. Confirm the approved budget and checkpoint interval. The proposed full values are 1,000 updates and a checkpoint every 250. Start a new full run. A 100-step pilot checkpoint cannot resume into a changed 1,000-step config.

```powershell
Copy-Item $baseCfg $fullCfg
notepad $fullCfg
& $py scripts/check_experiment_config.py $fullCfg
& $py scripts/train_lora.py $fullCfg --model-dir $sd --approved-manifest-sha256 $manifestHash @variant
```

## 7. Make images from the saved adapter

**For the pilot**, copy the base config to `artifacts/lab-generation-pilot.yaml`. Under **generation**, set each of the six `candidates_per_class` values to 1 and set `minimum_accepted`, `target_accepted` and `maximum_candidates` to 6. Use the pilot adapter path and the fresh output `artifacts/generation/pilot`. This is a small sample for testing and timing. Skip the full-run preparation block below and use this pilot block, then run the shared command under the hash prompt.

```powershell
$generationConfig = 'artifacts/lab-generation-pilot.yaml'
Copy-Item $baseCfg $generationConfig
notepad $generationConfig
$adapter = 'artifacts/diffusion/RUN-FOLDER/step-000100'
$generationOut = 'artifacts/generation/pilot'
Get-Item "$adapter/checkpoint.json", "$adapter/adapter.safetensors"
(Get-FileHash -Algorithm SHA256 "$adapter/checkpoint.json").Hash.ToLowerInvariant()
```

**For the full run**, use a reviewed generation config with the final seed, precision, batch size, inference steps, guidance and counts. The baseline defaults to 166 samples per class, 996 total, 30 steps and guidance 7.5. Use the selected full adapter and `$generated`. Do not place pilot outputs in the final pool.

```powershell
# Full run
Copy-Item $baseCfg artifacts/lab-generation.yaml
notepad artifacts/lab-generation.yaml
$generationConfig = 'artifacts/lab-generation.yaml'
$adapter = 'artifacts/diffusion/RUN-FOLDER/step-001000'
$generationOut = $generated
Get-Item "$adapter/checkpoint.json", "$adapter/adapter.safetensors"
(Get-FileHash -Algorithm SHA256 "$adapter/checkpoint.json").Hash.ToLowerInvariant()
```

Have a teammate review the checkpoint and its hash. The next prompt expects the hash of **checkpoint.json**. The CLI checks the adapter tensors separately.

```powershell
$adapterHash = (Read-Host 'Paste the approved checkpoint.json SHA-256').Trim().ToLowerInvariant()
& $py scripts/check_experiment_config.py $generationConfig
$generationStart = Get-Date
& $py scripts/generate_defects.py $generationConfig --model-dir $sd --adapter-dir $adapter --approved-adapter-sha256 $adapterHash --approved-manifest-sha256 $manifestHash --output $generationOut
if ($LASTEXITCODE -ne 0) { throw 'Generation stopped. Keep the plan and batches for resume.' }
"Total generation seconds: $(((Get-Date) - $generationStart).TotalSeconds)"
```

Expect `plan.json`, numbered batch folders with PNGs, and `candidates.jsonl`. Every row should have an intended class, prompt, seed, image hash and source metadata. Every image starts pending review. After interruption repeat the **identical** command with `--resume`. If a killed job leaves `.generation.lock`, confirm that no generation process is running before removing that one stale lock. A changed plan needs a new output path.

## 8. Review the generated images

Run this on CPU after final generation. Use fresh output directories. Keep the audit, original candidate folder and both review files. The default perceptual threshold is 5. If the team approves a different threshold before the review, pass the same `--threshold` to all three commands.

1. Run the audit and open the PNG contact sheets in `artifacts/review/first`. The numbered tiles follow the order in `audit.json` and `primary.json`.

```powershell
$reviewArgs = @('--generation', $generated, '--manifest', $manifest, '--real-audit', $audit, '--approved-manifest-sha256', $manifestHash)
& $py scripts/review_synthetic.py audit @reviewArgs --output artifacts/review/first
Invoke-Item artifacts/review/first
notepad artifacts/review/first/primary.json
```

2. For every row in `primary.json`, replace `pending` with `accepted` or `rejected`, add your reviewer ID and a timezone-aware timestamp such as `2026-11-18T12:00:00+05:00`. Record reasons for all rejections and flagged acceptances. Mark disputed labels with `"disputed": true`. Reject unreadable, blocked, wrong-class or ambiguous examples. Keep at most one exact copy. Compare visually with **real train** only. The automated leakage screen compares saved fingerprints and blocks suspicious held-out matches. Preserve those flags and keep them out of generator tuning.

3. Ask a different reviewer to complete the second file. It lists a seeded sample of up to ten accepted images per class plus all disputed labels. If reviewers disagree, resolve it before export. After a primary edit, generate the second template again in a new folder.

```powershell
& $py scripts/review_synthetic.py second-review @reviewArgs --primary artifacts/review/first/primary.json --output artifacts/review/second
notepad artifacts/review/second/secondary.json
```

4. Export accepted images and check the report.

```powershell
& $py scripts/review_synthetic.py export @reviewArgs --primary artifacts/review/first/primary.json --secondary artifacts/review/second/secondary.json --output $reviewFinal
Get-Content "$reviewFinal/review-summary.json"
(Get-FileHash -Algorithm SHA256 $accepted).Hash.ToLowerInvariant()
```

The export writes `approved.jsonl` for training, `reviewed.jsonl` with all decisions, and `review-summary.json` with counts and shortfalls. Have a teammate review and save the approved export hash. Confirm all six classes and your approved quotas before opening the comparison. The protocol proposes 600 accepted images and a balanced reduced target of 300. If a quota is missed, record it and agree on how to proceed before classifier training.

## 9. Make the quality report

The qualitative report works without extra weights. It shows accepted and eligible rejected images, their closest training matches, and reviewer notes. KID compares real train with accepted synthetic images. If samples or weights are missing, the report marks the score unavailable.

```powershell
& $py scripts/assess_synthetic_quality.py --generation $generated --real-root $real --manifest $manifest --real-audit $audit --review $reviewFinal --approved-manifest-sha256 $manifestHash --output artifacts/quality/visual-v1
```

For actual KID, use the separately approved Inception file from step 3:

```powershell
& $py -m pip install -e '.[quality]'
$weights = 'models/weights-inception-2015-12-05-6726825d.pth'
$weightsHash = (Read-Host 'Paste the independently verified Inception weight SHA-256').Trim().ToLowerInvariant()
& $py scripts/assess_synthetic_quality.py --generation $generated --real-root $real --manifest $manifest --real-audit $audit --review $reviewFinal --approved-manifest-sha256 $manifestHash --output artifacts/quality/kid-v1 --weights $weights --approved-weights-sha256 $weightsHash
```

Open `quality.md`, `quality.json` and the six PNGs in the chosen directory. Check counts, unavailable reasons, and reviewer observations. Defaults use 100 seeded draws, 40 images per side per class and 200 per side overall. KID measures a difference in image features. Human decisions and classifier usefulness are evaluated separately.

## 10. Train the three final classifiers

1. Copy the base config to the final config. Review the common budget, weights, validation interval and device. The proposal uses 1,000 updates, batch 24 and a 50:50 mix. Set the **classification** seed list to the approved matched seeds, normally 42, 43 and 44. If the team approves a reduced one-seed experiment, use `[42]` in the YAML and change the loop below to `@(42)`. If you want final inference on CPU, set `evaluation.device: cpu` **now**. Keep this exact config for final evaluation.

```powershell
Copy-Item $baseCfg $compareCfg
notepad $compareCfg
& $py scripts/check_experiment_config.py $compareCfg
Get-FileHash -Algorithm SHA256 $compareCfg
```

2. Use the approved synthetic hash from step 8. Train all three arms for every planned seed with the **same** config and initial weights. The loop stops on an error so you can inspect the run record.

```powershell
$acceptedHash = (Read-Host 'Paste the peer-approved approved.jsonl SHA-256').Trim().ToLowerInvariant()
if ((Get-FileHash -Algorithm SHA256 $accepted).Hash.ToLowerInvariant() -ne $acceptedHash) {
    throw 'Approved synthetic export changed.'
}
foreach ($seed in @(42,43,44)) {
    & $py scripts/train_real_baseline.py $compareCfg --seed $seed --arm all --approved-manifest-sha256 $manifestHash --local-weights $resnet --synthetic-root $generated --synthetic-manifest $accepted --approved-synthetic-manifest-sha256 $acceptedHash
    if ($LASTEXITCODE -ne 0) { throw "Training stopped at seed $seed. Inspect its records." }
}
```

Each completed arm writes `best.pt`, `training.jsonl` and `run.json` in its own folder. Save the printed record paths. Check matching validation IDs, update counts and initial weight hash. Check the source counts in each record. Real-only uses real train, synthetic-only uses approved generated images and mixed uses both. Every arm chooses its checkpoint on real validation. A classifier checkpoint has no optimizer resume state. If one run stops, keep its records and use the same command with an individual `--arm` and `--seed` for an approved retry. Record the retry in the run inventory.

## 11. Test the frozen classifiers on real test images

1. Finish checkpoint selection **before looking at test results**. List the saved run records, including any failed attempts. Agree on one record per configured arm and seed using the protocol and validation evidence only. Keep the full inventory separately. For three seeds you expect nine arm/seed pairs. Missing or failed pairs are reported as incomplete.

```powershell
Get-ChildItem artifacts/classification -Recurse -Filter run.json | Select-Object FullName
```

2. Run the following helper. For each prompt, paste the chosen path beginning with `artifacts/` and ending in `/run.json`, with forward slashes. Press Enter on an empty line for a missing record. The helper reads the record metadata and computes hashes. It does not open test images or choose checkpoints.

```powershell
$selection = 'artifacts/selection.json'
$runs = @()
do {
    $path = (Read-Host 'Paste a selected relative run.json path, or Enter when done').Trim()
    if ($path) {
        if ($path -notmatch '^artifacts/classification/[^/]+/run\.json$' -or
            -not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw 'Use a saved relative path in artifacts/classification with forward slashes.'
        }
        $record = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
        $runs += @{ arm = $record.arm; seed = $record.seed; path = $path
                    sha256 = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant() }
    }
} while ($path)
if (Test-Path $selection) { throw 'Selection already exists. Preserve the sealed copy.' }
@{ schema_version = 1; manifest_sha256 = $manifestHash; runs = $runs } |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding ascii $selection
Get-Content $selection
(Get-FileHash -Algorithm SHA256 $selection).Hash.ToLowerInvariant()
```

3. Ask a peer to verify the selection and record its exact SHA-256. Run the evaluator with the exact comparison config used for training. The output path must be new.

```powershell
$selectionHash = (Read-Host 'Paste the peer-approved selection.json SHA-256').Trim().ToLowerInvariant()
& $py scripts/evaluate_classifier.py $compareCfg --selection $selection --approved-selection-sha256 $selectionHash --approved-manifest-sha256 $manifestHash --output artifacts/evaluation/frozen-comparison
$LASTEXITCODE
```

Exit 0 means all expected runs completed. Exit 1 means the report includes failed or missing runs. Exit 2 means an input or artifact error. Open `comparison.md` first. Then open `performance.svg`, each `confusion.svg`, and `evaluation.json`. Per-run folders include predictions, per-class metrics and supports. Verify that each confusion matrix sums to the shared real test count. Report every planned arm and seed. Use macro F1 as the primary metric, and compare mixed minus real-only using matched seeds. One seed has no across-seed standard deviation.

```powershell
notepad artifacts/evaluation/frozen-comparison/comparison.md
Invoke-Item artifacts/evaluation/frozen-comparison/performance.svg
Invoke-Item artifacts/evaluation/frozen-comparison
```

## 12. Leave the lab with these files

- Save the exact Git commit, package list, GPU and driver details, booking notes and measured pilot memory/time.
- Keep the original archive and permitted-use record, dataset audit, peer decisions and frozen manifest pair with their hashes.
- Back up resolved configs, LoRA adapter/checkpoint folders, generation plan and batches, both image reviews, and the accepted export.
- Preserve all classifier attempts, selected run hashes, test outputs and the final report. Only share assets permitted by their terms. Use the [results template](reports/results-template.md) to tie each number to its run.
- Have someone else replay the CPU checks and verify the asset hashes. A real accuracy claim requires the actual experiment. The checked-in tests use fixtures.

### If something fails

| What you see | What to do next |
| --- | --- |
| PowerShell says `$py` is empty | Enter the cloned repository and rerun the variables block in step 3. If required, repeat the approved manifest hash prompt in step 4. |
| CUDA unavailable | Confirm the GPU and wheel in steps 1–2. Ask the lab administrator to check the driver. Repeat the CUDA probe. |
| Out of GPU memory | Save the failed run and memory log. Check other jobs, batch size 1, gradient checkpointing and supported precision. Discuss a reviewed 256 px pilot or larger GPU. Recheck memory and image detail. |
| File missing or wrong variant | Check the model folders and filenames in step 6. Confirm you are in the checkout and using the approved local files. |
| Hash mismatch | Find the file named in the error. Compare it with the approved copy. Restore the correct artifact or repeat the applicable review. Treat a changed hash as a changed input. |
| Output already exists | Keep completed output. Use a new directory for a new attempt. Use `--resume` only for the documented LoRA or generation cases. |
| Bad YAML or JSON | Save the file as plain UTF-8, check indentation or commas, and rerun the relevant validator. |
| Fixture test says filename too long | Use a short writable temporary directory outside the checkout, for example `--basetemp C:/lab-temp/test-01`, with a new directory name for each run. Do not point it at existing data or artifacts. Avoid `.cache` and `.tmp*` path components because the foundation validator deliberately ignores them. |
| Unavailable KID or missing classifier runs | Report the reason shown. Qualitative review remains available. Keep missing or failed classifier runs visible in the comparison. |
