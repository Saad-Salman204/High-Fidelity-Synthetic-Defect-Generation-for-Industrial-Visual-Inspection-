# CPU tests

From a fresh Python 3.11 environment:

```text
python -m pip install -e ".[dev,audit]"
python -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[lora]"
python -m pip check
python scripts/check_foundation.py
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --strict-config --strict-markers
```

The `foundation` job runs these steps on Ubuntu. Only installation needs a package
index. The test suite uses temporary files and synthetic images, with no dataset,
model, GPU or credentials. The tests cover dataset audit and foundation contracts
on the base branch. Split and experiment config PRs add their own suites, which
pytest will pick up after merge. Preprocessing remains a planned interface.

`scripts/audit_dataset.py --json-errors` emits a JSON error on stdout and returns
2 for input or file errors. A completed audit with findings still writes a report
and returns 1. The default input error message remains human-readable.

Use small authored image/manifest fixtures for corrupt decode, grouping, split overlap, normalization, sampler membership, config validation and hand-computed metrics. Verify failure paths. No downloaded NEU images or pretrained weights in ordinary CI. Separate mocked sampler tests, CPU integration and real GPU pilot evidence. Never add dummy passing tests merely to obtain a green badge.
