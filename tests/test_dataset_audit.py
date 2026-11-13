"""Exercise the read-only audit with locally authored images."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from industrial_defect_diffusion.data.validation import format_summary, validate_dataset


def write_image(path: Path, mode: str = "L", size: tuple[int, int] = (200, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, 60).save(path)
    return path


def test_counts_decode_and_proposed_exclusions(tmp_path):
    valid = write_image(tmp_path / "Cr_1.bmp")
    broken = tmp_path / "In_2.bmp"
    broken.write_bytes(b"invalid image")
    write_image(tmp_path / "Pa_1.png", "RGB")
    write_image(tmp_path / "PS_1.png", size=(201, 200))
    write_image(tmp_path / "unknown_1.png")
    write_image(tmp_path / "train" / "scratches" / "Sc_1.bmp")
    (tmp_path / "notes.txt").write_text("unexpected", encoding="utf-8")
    before = valid.read_bytes()

    report = validate_dataset(tmp_path)
    files = {row["relative_path"]: row for row in report["files"]}
    assert len(files) == 7
    assert files["Cr_1.bmp"]["class_id"] == 0
    assert files["Cr_1.bmp"]["sha256"] == sha256(before).hexdigest()
    assert "unreadable_image" in files["In_2.bmp"]["problems"]
    assert "expected_grayscale_L" in files["Pa_1.png"]["problems"]
    assert "expected_200x200" in files["PS_1.png"]["problems"]
    assert "unknown_label" in files["unknown_1.png"]["problems"]
    assert files["notes.txt"]["disposition"] == "proposed_exclusion"
    assert report["structure_findings"]
    assert report["missing_count_by_class"]["inclusion"] == 299
    assert report["status"] == "review_required"
    assert valid.read_bytes() == before and broken.exists()
    assert "Images found: 6 / 1800" in format_summary(report)


def test_duplicates_and_conflicting_labels(tmp_path):
    original = write_image(tmp_path / "crazing" / "Cr_1.bmp")
    copy = tmp_path / "In_1.bmp"
    copy.write_bytes(original.read_bytes())
    write_image(tmp_path / "Cr_2.png")
    write_image(tmp_path / "patches" / "Sc_4.png", size=(201, 200))

    report = validate_dataset(tmp_path)
    group = report["exact_duplicate_groups"][0]
    assert group["conflicting_labels"] and group["review_status"] == "pending"
    assert set(group["paths"]) == {"In_1.bmp", "crazing/Cr_1.bmp"}
    assert any(
        pair["distance"] == 0 and set(pair["paths"]) == {"Cr_2.png", "crazing/Cr_1.bmp"}
        for pair in report["near_duplicate_pairs"]
    )
    by_path = {row["relative_path"]: row for row in report["files"]}
    assert "conflicting_label" in by_path["patches/Sc_4.png"]["problems"]
    assert by_path["Cr_2.png"]["disposition"] == "needs_review"


@pytest.mark.parametrize("path", ["../outside.bmp", "C:/absolute.bmp", "a\\b.bmp"])
def test_inventory_rejects_unsafe_paths(tmp_path, path):
    with pytest.raises(ValueError, match="Invalid inventory path"):
        validate_dataset(tmp_path, expected_paths=[path])


def test_missing_inventory_entries_and_missing_root(tmp_path):
    write_image(tmp_path / "Cr_1.bmp")
    report = validate_dataset(tmp_path, expected_paths=["In_1.bmp"])
    assert report["missing_paths"] == ["In_1.bmp"]
    assert report["files"][0]["disposition"] == "proposed_exclusion"
    with pytest.raises(NotADirectoryError):
        validate_dataset(tmp_path / "missing")


def test_cli_report_and_no_overwrite(tmp_path):
    root = tmp_path / "dataset"
    write_image(root / "Cr_1.bmp")
    report = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_dataset.py"
    command = [sys.executable, str(script), "--root", str(root), "--report", str(report)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "review_required" in result.stdout
    saved = report.read_bytes()
    assert json.loads(saved)["files"][0]["class_name"] == "crazing"
    rerun = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rerun.returncode == 2
    assert report.read_bytes() == saved
