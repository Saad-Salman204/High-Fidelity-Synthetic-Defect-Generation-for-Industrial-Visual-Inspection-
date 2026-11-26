# Local research presentation dashboard

The dashboard is a localhost-only React + Tailwind application with a FastAPI
backend. It presents the verified Phase 1 artifacts and gives a live, serialized
inference demo using the existing local SD 1.5 + LoRA adapter and ResNet-18
checkpoints. It does not train models or change the candidate review ledger.

## Requirements

- Windows 10/11, Python 3.11, and the project's configured `.venv`.
- The existing CUDA-enabled PyTorch environment and NVIDIA GPU for live model
  inference. Research results remain readable without loading a model.
- Node.js 20.19+ or 22.12+ and npm. The currently tested local Node version was
  26.13.0.
- Existing local project artifacts: `artifacts/neu-lora-resume-1000.yaml`, the
  frozen split files, `artifacts/classification/*`, the completed diffusion run,
  `models/sd-v1-5`, the synthetic run, and
  `artifacts/evaluation/phase1-final-20261126`.

The real dataset image root is not needed for this dashboard: upload classification
uses the existing shared image preprocessor, generation uses the already recorded
frozen manifest and verified adapter provenance, and results are loaded from Phase
1 artifacts. No service binds to the LAN or cloud.

## First-time setup

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[presentation]"
Push-Location presentation\web
npm ci
Pop-Location
```

These install the small API and local frontend dependencies. They do not download
model weights or datasets. The UI uses the repository's installed ML environment.

## Start and stop

Start both local services in hidden windows and write logs under ignored local
artifacts:

```powershell
.\scripts\start_presentation.ps1
```

Open **http://127.0.0.1:5173** in the browser on the university PC. This localhost
address also works while that desktop is being controlled through AnyDesk.

For separate foreground terminals, run:

```powershell
.\scripts\start_presentation_backend.ps1
```

```powershell
.\scripts\start_presentation_frontend.ps1
```

Stop the background services after any active GPU operation finishes:

```powershell
.\scripts\stop_presentation.ps1
```

Backend API documentation is at **http://127.0.0.1:8000/docs**. Service logs and
the local PID record are saved in `artifacts/presentation/` (ignored by Git).

## Features and research status

- **Overview:** methodology, frozen 70/15/15 split, model flow, selection policy,
  and central measured result.
- **Live demo:** select one of six labels and a deterministic seed to generate one
  512 × 512 sample from the pinned SD 1.5 revision and the verified 1,000-step
  LoRA checkpoint; classify generated or uploaded PNG/JPEG/WebP images using any
  of the three arms and seeds 42/43/44. The displayed classifier is the exact
  selected run. Inputs are never added to training data.
- **Results:** actual nine-run metrics, seed-wise values, per-class metrics,
  confusion matrices, and Phase 1 PNG figures. Figures and JSON/CSV data are read
  from the verified evaluation output; the dashboard does not recalculate scores.
- **Gallery:** deterministic review-contact-sheet examples. They remain marked as
  pending and are not presented as accepted training examples.

The app holds one inference model family at a time. Concurrent generation and
classification requests wait behind a GPU lock. Models load only after a live
request, and out-of-memory errors release loaded models and return a useful error.
Demo-generated images and sidecar provenance are written under the separate,
ignored directory `artifacts/demo/presentation/`; the 996-candidate run is never
modified.

The selected 783 synthetic images were filtered by the automated no-blocker/no-
warning policy, not individually human-approved. The 186 retained near-match
relationships are disclosed. The 72-image contact-sheet sample is not a review of
the full candidate set. Real-only scored 100% on the frozen test set across all
three seeds; this ceiling limits possible measured gains and deserves independent
methodology scrutiny. No statistical significance test was performed.

## Troubleshooting

- **Frontend says backend unavailable:** check `artifacts/presentation/logs/`;
  verify port 8000 is free and start the backend from the repository root.
- **Missing frontend packages:** run `npm ci` from `presentation/web`.
- **Missing FastAPI:** run the first-time pip command in the same `.venv` used for
  training.
- **Generator/checkpoint unavailable:** confirm the ignored model/checkpoint
  directories named above exist. The app does not download replacements.
- **CUDA out of memory:** wait for the current task, then retry. The runtime
  serializes GPU jobs and releases model memory after an OOM.
- **Port already in use:** stop the old local service or change the port in the
  PowerShell start scripts and Vite proxy together.

## API surface

`GET /api/health`, `GET /api/models`, `GET /api/project/overview`,
`GET /api/research/results`, `GET /api/research/metrics`,
`GET /api/research/experiments`, `GET /api/gallery`, and the image/figure routes
serve local, verified files. `POST /api/generate`, `POST /api/classify`, and
`POST /api/classify-generated/{image_id}` execute inference in worker threads.
Uploads are limited to 10 MB and 40 megapixels, and decode-checked as PNG, JPEG,
or WebP. File serving uses fixed ID/filename allowlists rather than arbitrary
filesystem paths.
