# Contributing

All ML functionality is planned. Start with the [backlog](docs/planning/backlog.md), [contracts](docs/architecture/contracts.md) and [protocol](docs/experiments/protocol.md). Confirm ownership with teammates; A/B/C are suggested streams, not people.

## Setup

Clone the existing repository after access is granted:

```powershell
git clone https://github.com/Saad-Salman204/High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-.git
cd High-Fidelity-Synthetic-Defect-Generation-for-Industrial-Visual-Inspection-
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Use `.venv/bin/python` on Linux/macOS. See [setup](docs/planning/setup.md) for tested versus pending commands and the later GPU environment gate. Credentials are never entered into tracked files.

## Work on an issue

Pick a dependency-ready task, agree assignment and clarify acceptance criteria. Use an actual issue number in branch names:

```text
git fetch origin
git switch main
git pull --ff-only
git switch -c feat/<issue-number>-short-description
```

Replace angle-bracket placeholders; these are workflow examples, not commands executed during setup. Use `fix/` or `docs/` when appropriate. Keep branches short-lived and avoid a long-lived develop branch.

## Check and submit

```text
python scripts/check_foundation.py
python -m ruff check .
python -m ruff format --check .
```

Run in the development environment. `python -m pre_commit install` optionally installs hooks after dependencies are installed. No pytest suite exists in the foundation; I19 enables it when meaningful implemented logic exists. PR checks must stay CPU-only and avoid dataset/model downloads.

Use conventional commits, e.g. `feat(data): add dataset validation pipeline` or `docs: document experiment methodology`. Mention the issue in the commit/PR. Stage specific intended files, inspect staged diff and confirm no credentials/data/weights. Push the feature branch, open a focused PR using the template, and request one teammate review. Do not merge until acceptance criteria, required CI and peer review are satisfied. Record limitations and distinguish mock tests from real model verification.

## Staying current and resolving conflicts

Fetch and merge `origin/main` into your feature branch; do not force-push a shared branch. Inspect conflicting files with their owner, preserve both intended changes, stage the resolution and rerun relevant checks. Avoid concurrent edits to shared protocol/config files by agreeing one editor. Review changes to interfaces with downstream consumers.

Project board statuses: Backlog → Ready once dependencies are merged → In Progress → In Review → Done after reviewed merge. Blocked requires a linked prerequisite and reason. Actual board configuration is recorded in [GitHub setup](docs/planning/github-setup.md). Follow [Definition of Done](docs/planning/definition-of-done.md) and [AGENTS.md](AGENTS.md) for AI-assisted work.
