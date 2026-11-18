# Run the project using the lab menu

Use this file at the university PC. Keep [IMPLEMENTATION.md](IMPLEMENTATION.md) open for the dataset review schema, model preparation and scientific rules. Your juniors can read [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## 1. Open the launcher

Clone the reviewed repository and open PowerShell **inside its folder**. The menu runs from the script's own folder even if your current directory differs. Windows PowerShell 5.1 or PowerShell 7 on Windows is supported.

First preview the workflow. This command writes no files, installs nothing, and starts no Python, Git, GPU or training processes.

```powershell
.\run_lab.ps1 -DryRun -Action plan
```

Then open the real menu:

```powershell
.\run_lab.ps1
```

If your approved local script is blocked by execution policy, ask the lab administrator which method is permitted. When process-only execution is allowed, use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_lab.ps1
```

The launcher prints a numbered list. Enter the number beside the step you want. Enter **0** to exit. Complete one step and inspect its output before starting the dependent step. A red error or a line starting with **STOP** means you need to resolve that problem. The menu stays available after a failed step.

## 2. Set your paths once

Choose **configure**. Press Enter to keep a displayed value, or type `-` to clear an optional value such as `inventory` or `variant`. Use forward slashes for paths inside this checkout. The session saves in ignored `artifacts/lab-session.json` and is loaded next time.

| Field | What to enter |
| --- | --- |
| `python`, `venv_dir` | Default `.venv/Scripts/python.exe` and `.venv`. These must match for package setup. |
| `device`, `wheel_index` | Start with `cpu` and `cpu` at home. At university use `cuda` and the lab-approved index, such as `cu126`, after checking the driver. |
| `archive`, `data_root` | The original approved NEU archive and the folder directly containing the real images/class folders. |
| `inventory` | Keep empty unless you have a trusted JSON inventory of expected paths. |
| `snapshot`, `classifier_weights` | The approved local SD v1.5 component folder and ImageNet ResNet-18 weight file. |
| `variant` | Keep empty for default safetensors filenames. Enter `fp16` only if all three weighted SD components use that filename variant. |
| `pilot_record`, `pilot_adapter_dir` | Update after the pilot with its actual run record and completed step directory. |
| `adapter_dir`, `resume_checkpoint` | Update with the selected full-run adapter or a complete intermediate recovery checkpoint. |
| Output/config/review paths | Defaults follow the manual. If an output already exists, preserve it and enter a fresh path. |
| `backup_destination` | A new absolute directory outside the checkout on approved storage, for example a new folder on your team drive. |

The session contains paths and review receipts. The existing YAML experiment files contain the scientific settings. Edit those files in Notepad or your editor. The launcher validates them through the shared Python configuration validator.

## 3. Set up the environment and approved assets

1. Choose **setup**. Read the proposed installation command and type `INSTALL` when ready. It creates a Python 3.11 virtual environment if needed, installs the pinned CPU or reviewed CUDA Torch pair and existing development/LoRA packages, then checks dependencies. Package downloads need internet.
2. Choose **environment**. Confirm the package versions and requested device probe succeed. On CUDA it also prints GPU, driver and free/total memory.
3. Choose **tests** and type `CHECK`. These are CPU fixture tests. Keep Git installed so temporary repository tests work.
4. Obtain the permitted dataset and approved local model assets using IMPLEMENTATION.md step 3. The launcher operates on local assets. Weight/model acquisition remains a deliberate preparation step in that guide.
5. Choose **models**. It hashes the local ResNet weights and SD component files and records the model review receipt. This checks local fingerprints and required files. The lab pilot still needs to establish that the full model loads and runs successfully.

For each approval prompt, a teammate must have reviewed the named artifact and its purpose. Paste the SHA-256 from that independent review and enter the reviewer name and evidence note. The receipt is tied to the current file bytes. If the file changes, its previous approval stops applying. A script cannot establish dataset terms, inspect class fidelity or make an independent peer judgment.

## 4. Prepare configs, audit and freeze the real split

Choose **templates**, then `base`. The launcher copies the existing baseline YAML and fills the session data and manifest paths. Open the printed file path in an editor. Review settings with the team. Leave it proposed while editing, and set `status: approved` after review.

Choose **audit**. Exit 1 from the audit means findings need review, and the launcher prints the canonical audit digest. Use the audit to create the explicit duplicate/exclusion review JSON described in IMPLEMENTATION.md step 4. Keep all source images unchanged.

Choose **freeze** after reviewing that JSON. The launcher passes the same inventory, archive identity and the base configuration's split seed/ratios to the existing split command. An identical existing freeze is verified. A changed or incomplete freeze fails.

Inspect `manifest.csv` and `split-metadata.json` with a peer. Record its reviewed hash, class counts, grouping decisions and sealed test membership. Training, review and evaluation steps ask for that manifest approval and verify that its archive identity matches the approved archive. Only the training partition teaches models.

## 5. Run pilots before the full experiment

1. In **templates**, create `classifier-pilot`, `lora-pilot`, and `generation-pilot`. Each starts as proposed and needs review. The classifier pilot is 10 updates; LoRA pilot is 100 with checkpoints every 25. The generation pilot plans one candidate per class. Other settings come from the base.
2. Use **baseline** for the real-only pilot. It uses the first seed in its reviewed config and real validation for checkpoint selection. Type `TRAIN` to start.
3. Use **lora**, then `pilot`. It checks the config, manifest and approved SD receipt, probes the configured device and asks you to type `TRAIN`.
4. Check the printed run folder and loss history. Update `pilot_record` and `pilot_adapter_dir` through **configure**.
5. Use **generate**, then `pilot`, to reload the pilot adapter and inspect six new candidates. Type `GENERATE` when ready. This pool remains separate from the final images.
6. Complete the pilot memory, timing, sample and recovery checks from the manual. Review whether the projected full run fits the booking and has memory headroom.

CUDA training/generation starts a hidden `nvidia-smi` sampler and writes a CSV beside the launcher log. It stops that sampler when the job returns or fails. Wall seconds are recorded per job. Device-wide sampled memory and wall time include other GPU users/startup/save costs. PyTorch peak allocated/reserved memory still requires the additional lab instrumentation described in the manual.

## 6. Full adaptation, generation and image review

Create `full` and `generation` templates. Review their budgets and precision. A **full** LoRA action also needs the reviewed pilot evidence. It starts a new run with the approved full config.

After full training, update `adapter_dir` with the selected complete step. Use **generate → full** to create the planned final candidate pool. Every candidate remains pending review.

Use **review** three times with a human pause between:

1. **audit** writes technical findings, class contact sheets and the primary review template.
2. Fill every primary decision in the JSON file with accepted/rejected, reviewer, timestamp and required reasons. The manual explains the fields. Then run **second-review**.
3. A different reviewer completes the second template and resolves disputes. Then run **export**.

The existing Python review code rejects blocked approvals, duplicate acceptances, incomplete decisions and unresolved disagreement. Reviewers see training comparisons, and held-out similarity is screened using saved fingerprints. Preserve the full ledger and rejection reasons. Review the final accepted manifest and counts before training.

## 7. Quality report, classifiers and final test

1. Use **quality → visual** for class grids and a report with unavailable KID if weights are absent.
2. For KID, first use **quality → setup-quality**, then provide the independently verified local Inception file and select **kid**. KID uses real train and the accepted synthetic pool.
3. Create and review the **comparison** template. Set the final shared update budget, seeds and desired evaluation device before running.
4. Use **classifiers → all**. The launcher reads seeds from that exact config and runs the three arms for every configured seed. It checks the accepted export against the configured balanced minimum. A missed quota needs a reviewed protocol/config decision.
5. Keep every attempt and inspect the printed records. Use an individual arm retry only for an approved recorded failure, keeping the same config and seed.
6. Use **selection** and enter one chosen relative run.json path for each arm/seed, including failed records where applicable. End with an empty line. Selection uses protocol and validation evidence. The helper writes a new sealed index and rejects duplicate or incompatible entries.
7. Have a peer review the exact selection and its hash. Use **evaluate**, then type `TEST`. The existing evaluator checks checkpoint/config/split compatibility and scores the same real test IDs in inference mode.

Exit 1 from final evaluation means the comparison is incomplete. Inspect and report missing/failed pairs. Open the comparison Markdown, per-class metrics, confusion SVGs and performance SVG in the evaluation directory. Final metrics must not be used to change training or checkpoint choices.

## 8. Recover safely and back up

| Situation | Menu action |
| --- | --- |
| Interrupted LoRA | Update `resume_checkpoint` to a completed intermediate step; choose `resume-pilot` or `resume-full`. The existing Python code checks config, code, environment, source and optimizer state. Resume writes a new run directory. |
| Interrupted generation | Choose the matching `resume-pilot` or `resume-full` with the same plan and output. Confirm the previous process stopped. Existing batches are verified. |
| Generation lock remains | Read the manual's stale-lock instructions. Confirm the old process stopped before manually removing that one lock. |
| Interrupted classifier | Preserve its record and start an approved individual arm/seed retry. Classifier checkpoints contain inference weights and provide no optimizer resume. |
| Hash/config error | Find the named artifact and compare with the approved source. Restore it or review a new version. |
| Existing output | Choose a fresh output path. Completed outputs and frozen selections are preserved. |

After jobs stop, choose **backup**. Default `code-and-artifacts` copies tracked code and experiment artifacts. Choose `include-data-and-models` to also copy local data and models when permissions and space allow. Use a **fresh** directory outside the checkout. The backup verifies every copied file against its source and writes `backup-index.json` only after all files match. A partial copy has no completion index and needs a new attempt/destination. The index records the code commit and whether there were uncommitted changes.

The default backup omits ignored raw data and base models. Keep those assets in approved team storage too, or select the full asset backup. File copies preserve artifact identity; reproducibility also needs the environment, approvals and actual input assets.

## Logs and preview commands

Real launcher sessions write `artifacts/lab-launcher/*.jsonl` with commands, output, exit codes, approval references, errors and timing. Python run records remain beside their model artifacts. Keep these logs with the session JSON in the backup. Keep credentials out of session fields and configs.

You can preview any submenu without input or files:

```powershell
.\run_lab.ps1 -DryRun -Action lora -Option resume-full
.\run_lab.ps1 -DryRun -Action review -Option export
.\run_lab.ps1 -DryRun -Action quality -Option kid
.\run_lab.ps1 -DryRun -Action backup -Option include-data-and-models
```

The full plan displays the first alternative in each submenu and lists available alternatives. Its displayed default seed/budget examples are previews. Real execution reads your approved configs. `-NonInteractive` is useful for previews and environment probes; actions needing a new approval or typed execution confirmation stop.

The session file is locked while the launcher is open, so two copies cannot change the same session concurrently. Use the running menu or exit it before another invocation with that session. Dry-run needs no lock and writes no logs. Menu actions preserve jobs and artifacts on errors and ask you to resolve the failure before dependent work.

## What remains to verify in the lab

CPU fixtures verify launcher command dispatch, error handling, approvals, file preservation, selection and backup logic. Actual CUDA installation, NVIDIA sampling, full SD loading, precision, memory use, training recovery, generation quality, human labels and final real-test scores need the approved lab assets and hardware.
