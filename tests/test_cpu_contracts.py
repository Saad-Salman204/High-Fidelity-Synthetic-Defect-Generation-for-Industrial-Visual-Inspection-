"""Failure-path and deterministic contract checks for the CPU foundation."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def run_script(path, *args):
    return subprocess.run(
        [sys.executable, str(path), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("inventory_text", "message"),
    [
        ("[", "Expecting"),
        ('{"paths": []}', "must be a JSON array"),
        ('["../Cr_1.bmp"]', "Invalid inventory path"),
    ],
)
def test_audit_cli_json_error_has_no_report(tmp_path, inventory_text, message):
    dataset = tmp_path / "data"
    dataset.mkdir()
    inventory = tmp_path / "inventory.json"
    inventory.write_text(inventory_text, encoding="utf-8")
    output = tmp_path / "report.json"
    result = run_script(
        ROOT / "scripts/audit_dataset.py",
        "--root",
        dataset,
        "--report",
        output,
        "--inventory",
        inventory,
        "--json-errors",
    )
    assert result.returncode == 2
    error = json.loads(result.stdout)
    assert error["status"] == "error" and message in error["message"]
    assert not output.exists()


def test_audit_cli_byte_stable_and_keeps_conflicts(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    original = dataset / "Cr_1.bmp"
    Image.new("L", (200, 200), 42).save(original)
    payload = original.read_bytes()
    (dataset / "In_1.bmp").write_bytes(payload)
    (dataset / "Pa_1.bmp").write_bytes(payload[: len(payload) // 2])

    reports = [tmp_path / f"report-{index}.json" for index in range(2)]
    for report in reports:
        result = run_script(
            ROOT / "scripts/audit_dataset.py",
            "--root",
            dataset,
            "--report",
            report,
        )
        assert result.returncode == 1, result.stderr
    assert reports[0].read_bytes() == reports[1].read_bytes()
    audit = json.loads(reports[0].read_bytes())
    assert audit["image_file_count"] == 3
    assert audit["count_matches"] is False
    assert audit["exact_duplicate_groups"][0]["conflicting_labels"] is True
    assert "unreadable_image" in audit["files"][2]["problems"]
    assert original.read_bytes() == payload


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("malformed_yaml", "broken.yaml"),
        ("invalid_ratios", "Split ratios do not sum to 1"),
        ("cycle", "Dependency cycle"),
    ],
)
def test_foundation_rejects_invalid_contracts(tmp_path, change, message):
    for name in ("scripts", "configs", "docs/planning"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    for name in ("scripts/check_foundation.py", "CLAUDE.md", "configs/baseline.yaml"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    (tmp_path / "docs/planning/issues.json").write_text("[]", encoding="utf-8")

    if change == "malformed_yaml":
        (tmp_path / "broken.yaml").write_text("invalid: [", encoding="utf-8")
    elif change == "invalid_ratios":
        (tmp_path / "configs/baseline.yaml").write_text(
            "split: {ratios: [0.5, 0.2, 0.2]}\nclasses: {a: 0, b: 1, c: 2, d: 3, e: 4, f: 5}\n",
            encoding="utf-8",
        )
    else:
        (tmp_path / "issue.md").write_text("", encoding="utf-8")
        items = [
            {"id": "X", "dependencies": ["X"], "body_path": "issue.md", "week": 1, "priority": "P0"}
        ]
        (tmp_path / "docs/planning/issues.json").write_text(json.dumps(items), encoding="utf-8")

    result = run_script(tmp_path / "scripts/check_foundation.py")
    assert result.returncode == 1
    assert message in result.stderr
    assert "Traceback" not in result.stderr
