# Canonical engineering policy for coding agents

This is a three-week academic research repository. The foundation is planning-only; no ML implementation exists yet. Future implementation requires an assigned issue and its acceptance criteria. CLAUDE.md references this policy rather than duplicating it.

## Before editing

1. Read README, docs/architecture/overview.md, contracts.md, relevant ADRs and docs/experiments/protocol.md.
2. Read the assigned issue, dependencies, acceptance criteria, relevant code and current Git diff. Identify blocked prerequisites and do independent work only when safe.
3. State a small plan for complex work. Ask only about material ambiguity or required approval; routine choices follow documented contracts.
4. Respect data/diffusion/generation/classification/evaluation/utils boundaries. Keep CLI orchestration separate from library logic. Avoid monolithic notebooks, generic frameworks, unrelated refactors and unexplained dependencies.

## Research integrity: nonnegotiable checks

- Never train generator/classifier on test data; never use test images for prompts, masks, backgrounds, filtering, early stopping or tuning.
- Inspect immutable split/group hashes before runs. Only real train can supply generator learning or example-derived conditioning. Synthetic data stays out of real validation/test.
- Preserve class mapping, comparable preprocessing and shared real test IDs across all arms. Validate labels, dimensions and schema before training.
- Record seeds, dataset/archive versions, split hashes, model revisions, adapter/checkpoint hashes, resolved parameters and environment/hardware.
- Record synthetic ancestry and quality decisions. A prompt class is an intended label, not proof of fidelity.
- Follow equal-budget comparison rules. Do not silently alter ratios, selection criteria or seed counts. Record and obtain peer approval for protocol changes before test results are viewed.
- Report class-wise metrics, macro F1, failures, limitations and all preregistered arms/seeds. Never fabricate data, outputs, test results or improvements. Never cherry-pick without disclosure.
- Determinism is best effort: seed Python/NumPy/PyTorch/workers and record nondeterministic kernels and hardware differences.

## Implementation standards

- Follow PEP 8, Ruff and readable names. Prefer focused functions, appropriate type hints and public-interface docstrings.
- Use meaningful exceptions and actionable logging; do not swallow failures. Explicitly handle CPU/GPU availability and dtypes.
- Keep experiment parameters in versioned config, not machine-specific paths. Reject traversal and invalid manifests/configs.
- Justify dependencies and pin tested compatible versions. Keep reusable logic out of notebooks and avoid duplication.
- Preserve raw data and immutable completed artifacts; new runs get new IDs. Record interrupted/resumed state.
- Significant interface or research decisions require ADR/protocol updates and affected-owner review.

## Verification and completion

- Add meaningful unit tests for implemented logic and integration tests for boundaries. Cover wrong dimensions, missing/invalid labels, schema errors, leakage, invalid configs and model I/O.
- Use small authored fixtures in CPU CI; no pretrained downloads, external data or GPU training in normal PR checks.
- Distinguish mocks from real GPU/ML verification. Enable pytest in CI when meaningful tests exist; never add dummy passing tests or report unexecuted checks as passing.
- Run `python scripts/check_foundation.py`, `python -m ruff check .`, and `python -m ruff format --check .` in the configured environment, plus tests relevant to the change. Disclose unavailable checks.
- Check acceptance criteria against evidence, not intent. Update docs/config examples and keep README commands executable and status accurate.

## Git and review

- Use short-lived `feat/<issue>-description`, `fix/<issue>-description` or `docs/<issue>-description` branches. Never commit implementation directly to protected main.
- Focused conventional commits reference the issue. Open focused PRs with purpose, verification and limitations.
- Require one peer review and required CI before merge where supported. Do not bypass agreed reviews. CODEOWNERS rules apply once valid handles are configured.
- Never force-push shared branches or overwrite another contributor's work. Resolve conflicts with the affected owner and rerun checks.
- Do not merge unrelated changes. Do not assign actual people without confirmation. Do not close implementation issues for planning alone.

## Security and artifacts

- No committed secrets, tokens, private paths, datasets, checkpoints or generated datasets. Use ignored local storage and environment variables for credentials.
- Treat downloads as untrusted: verify source/checksum/license, inspect archives, avoid blind script execution and unsafe pickle loading where possible. Prefer safetensors for compatible model artifacts.
- Never use privileged operations without need. Respect package/model/dataset terms separately from code MIT.
- Review approved aggregate outputs before committing. Raw predictions/images remain ignored unless explicitly approved with licensing/provenance.

## Mandatory task completion report

Every future coding task ends with: (1) assigned issue, (2) changes made, (3) files modified, (4) exact tests executed, (5) actual outcomes, (6) known limitations/incomplete work, (7) docs updated, (8) commit/PR references, (9) decisions needing human approval. State blockers plainly. Never claim completion without verification.
