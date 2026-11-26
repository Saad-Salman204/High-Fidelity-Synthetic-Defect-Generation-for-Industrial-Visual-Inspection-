"""CPU tests for frozen evaluation, hand-computed metrics and report outputs."""

import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import pytest
import torch

from industrial_defect_diffusion.classification.baseline import build_model
from industrial_defect_diffusion.data.preprocessing import read_frozen_manifest
from industrial_defect_diffusion.evaluation.analysis import comparative_analysis
from industrial_defect_diffusion.evaluation.classifier import _comparison, evaluate, fingerprint
from industrial_defect_diffusion.evaluation.metrics import classification_metrics
from industrial_defect_diffusion.utils.experiment_config import ARMS, load_config
from industrial_defect_diffusion.utils.run_records import make_run_record, write_run_record

ROOT = Path(__file__).resolve().parents[1]


def prediction(image, actual, predicted):
    return {
        "image_id": image,
        "true_class": actual,
        "predicted_class": predicted,
        **{f"p{i}": float(i == predicted) for i in range(6)},
    }


def test_hand_calculated_scores_and_absent_classes():
    expected = {"a": 0, "b": 0, "c": 1, "d": 1}
    rows = [
        prediction(*item)
        for item in (
            ("a", 0, 0),
            ("b", 0, 1),
            ("c", 1, 1),
            ("d", 1, 1),
        )
    ]
    result = classification_metrics(expected, rows)
    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"][0][:2] == [1, 1]
    assert result["confusion_matrix"][1][:2] == [0, 2]
    assert result["confusion_row_normalized"][0][:2] == [0.5, 0.5]
    assert result["macro_precision"] == pytest.approx((1 + 2 / 3) / 6)
    assert result["macro_recall"] == pytest.approx(1.5 / 6)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 6)
    assert result["per_class"][2]["absent_true_class"]
    assert result["per_class"][2]["absent_predicted_class"]
    assert result["per_class"][2]["f1"] == 0


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_label", "nan", "sum", "argmax"])
def test_invalid_predictions(problem):
    rows = [prediction("a", 0, 0), prediction("b", 1, 1)]
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[1] = rows[0]
    elif problem == "wrong_label":
        rows[1]["true_class"] = 0
    elif problem == "nan":
        rows[1]["p0"] = float("nan")
    elif problem == "sum":
        rows[1]["p1"] = 0.9
    else:
        rows[1]["predicted_class"] = 0
    with pytest.raises(ValueError):
        classification_metrics({"a": 0, "b": 1}, rows)


@pytest.fixture
def comparison(tmp_path, frozen_images):
    images, manifest, metadata = frozen_images
    rows, details = read_frozen_manifest(manifest, metadata)
    config = load_config(ROOT / "configs/cpu-smoke.yaml")
    config["preprocessing"]["classifier_resolution"] = 32
    config["paths"].update(
        manifest=manifest.relative_to(tmp_path).as_posix(),
        data_root=images.relative_to(tmp_path).as_posix(),
        artifact_root="artifacts",
    )
    instant = datetime(2026, 11, 13, tzinfo=UTC)
    (tmp_path / "synthetic.jsonl").write_text("fixture provenance\n", encoding="utf-8")
    selection = {"schema_version": 1, "manifest_sha256": details["manifest_sha256"], "runs": []}
    for index, arm in enumerate(ARMS):
        directory = tmp_path / arm
        directory.mkdir()
        checkpoint = directory / "best.pt"
        torch.manual_seed(index)
        model = build_model("none")
        torch.save(
            {
                "schema_version": 1,
                "architecture": "resnet18",
                "update": 1,
                "model_state": model.state_dict(),
            },
            checkpoint,
        )
        record = make_run_record(
            config,
            stage="classification",
            arm=arm,
            seed=42,
            git_commit="a" * 40,
            started_at=instant,
            ended_at=instant,
            root=tmp_path,
            artifacts=(checkpoint.relative_to(tmp_path).as_posix(),),
            classifier_revision=fingerprint(checkpoint.read_bytes()),
            dataset_source_sha256=details["dataset_source_sha256"],
            synthetic_manifest="synthetic.jsonl" if index else None,
            metrics={
                "updates": 1,
                "validation_image_ids": [row["image_id"] for row in rows if row["split"] == "val"],
            },
        )
        path = directory / "run.json"
        write_run_record(path, record)
        selection["runs"].append(
            {
                "arm": arm,
                "seed": 42,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": fingerprint(path.read_bytes()),
            }
        )
    for row in rows:
        if row["split"] != "test":
            (images / row["relative_path"]).unlink()
    return config, selection


def test_deterministic_three_arm_evaluation_and_cli_outputs(comparison, tmp_path):
    config, selection = comparison
    approved = selection["manifest_sha256"]
    report, outputs = evaluate(tmp_path, config, selection, approved)
    assert evaluate(tmp_path, config, selection, approved) == (report, outputs)
    assert [run["arm"] for run in outputs] == list(ARMS)
    assert report["test_count"] == 6 and report["paired_seeds"] == [42]
    assert report["aggregates"]["real_only"]["macro_f1"]["sample_sd"] is None
    test_ids = [row["image_id"] for row in outputs[0]["predictions"]]
    assert len(set(test_ids)) == 6
    for run in outputs:
        assert [row["image_id"] for row in run["predictions"]] == test_ids

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "evaluation fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    selected = json.dumps(selection).encode()
    (tmp_path / "selection.json").write_bytes(selected)
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_classifier.py"),
        "config.json",
        "--selection",
        "selection.json",
        "--approved-selection-sha256",
        fingerprint(selected),
        "--approved-manifest-sha256",
        approved,
        "--real-root",
        str(tmp_path / config["paths"]["data_root"]),
        "--output",
        "artifacts/frozen-evaluation",
    ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / json.loads(result.stdout)["output"]
    assert json.loads((output / "evaluation.json").read_text())["runs"] == report["runs"]
    for arm in ARMS:
        directory = output / arm / "seed-42"
        assert (directory / "metrics.json").exists()
        for name in ("predictions.csv", "per-class.csv"):
            with (directory / name).open(newline="", encoding="utf-8") as stream:
                assert len(list(csv.DictReader(stream))) == 6
        ElementTree.parse(directory / "confusion.svg")
    ElementTree.parse(output / "performance.svg")
    assert "Macro F1" in (output / "comparison.md").read_text(encoding="utf-8")
    assert (output / "comparison.csv").is_file()
    assert (output / "comparative-analysis.json").is_file()
    assert (output / "per-class-aggregates.csv").is_file()
    assert (output / "research-report.md").is_file()
    assert (output / "executive-summary.md").is_file()
    assert (output / "jury-brief.md").is_file()
    assert (output / "presentation-outline.md").is_file()
    figures = output / "figures"
    assert (figures / "01_experimental_design.png").is_file()
    assert (figures / "09_per_class_f1.pdf").is_file()


def test_comparative_analysis_reports_paired_differences_and_class_means():
    runs = []
    for seed in (42, 43, 44):
        for arm_index, arm in enumerate(ARMS):
            value = 0.5 + 0.05 * arm_index + (seed - 42) * 0.01
            per_class = [
                {
                    "class_id": index,
                    "precision": value,
                    "recall": value - 0.02,
                    "f1": value - 0.01,
                }
                for index in range(6)
            ]
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "metrics": {
                        "accuracy": value,
                        "macro_precision": value,
                        "macro_recall": value - 0.02,
                        "macro_f1": value - 0.01,
                        "per_class": per_class,
                    },
                }
            )
    result = comparative_analysis(runs, [42, 43, 44])
    assert result["paired_seeds"] == [42, 43, 44]
    mixed_f1 = next(
        row
        for row in result["pairwise_metric_differences"]
        if row["baseline_arm"] == "real_only"
        and row["candidate_arm"] == "real_plus_synthetic"
        and row["metric"] == "macro_f1"
    )
    assert mixed_f1["candidate_minus_baseline"]["mean"] == pytest.approx(0.1)
    assert mixed_f1["candidate_minus_baseline"]["sample_sd"] == pytest.approx(0)
    assert len(result["per_class_seed_aggregates"]) == 6 * 3 * 3
    assert "no significance test" in result["interpretation"]


@pytest.mark.parametrize(
    "problem",
    [
        "approval",
        "different_split",
        "mapping",
        "validation",
        "missing",
        "corrupt",
        "incompatible",
    ],
)
def test_incompatible_or_missing_inputs(comparison, tmp_path, problem):
    config, selection = comparison
    approved = selection["manifest_sha256"]
    entry = selection["runs"][0]
    record_path = tmp_path / entry["path"]
    record = json.loads(record_path.read_text())
    checkpoint = record_path.with_name("best.pt")
    messages = {
        "approval": "reviewed SHA",
        "different_split": "same frozen real split",
        "mapping": "class mapping",
        "validation": "validation IDs",
        "missing": "best.pt",
        "corrupt": "Artifact SHA",
        "incompatible": "Unsupported baseline checkpoint",
    }
    if problem == "approval":
        approved = "0" * 64
    elif problem == "different_split":
        record["split_manifest_sha256"] = "0" * 64
    elif problem == "mapping":
        record["class_mapping_sha256"] = "0" * 64
    elif problem == "validation":
        record["metrics"]["validation_image_ids"] = []
    elif problem == "missing":
        checkpoint.unlink()
    elif problem == "corrupt":
        checkpoint.write_bytes(b"corrupt")
    else:
        torch.save({"schema_version": 1, "architecture": "wrong"}, checkpoint)
        digest = fingerprint(checkpoint.read_bytes())
        record["model"]["revision"] = digest
        record["artifacts"][0]["sha256"] = digest
    record_path.write_text(json.dumps(record), encoding="utf-8")
    entry["sha256"] = fingerprint(record_path.read_bytes())
    with pytest.raises((ValueError, FileNotFoundError), match=messages[problem]):
        evaluate(tmp_path, config, selection, approved)


def test_incomplete_runs_do_not_produce_paired_averages(comparison, tmp_path):
    config, selection = comparison
    selection["runs"].pop()
    entry = selection["runs"][1]
    path = tmp_path / entry["path"]
    record = json.loads(path.read_text())
    record.update(status="failed", failures=["fixture failure"])
    path.write_text(json.dumps(record), encoding="utf-8")
    entry["sha256"] = fingerprint(path.read_bytes())
    report, runs = evaluate(tmp_path, config, selection, selection["manifest_sha256"])
    assert len(runs) == 1
    assert report["status"] == "incomplete"
    assert [item["status"] for item in report["incomplete"]] == ["failed", "missing"]
    assert report["paired_seeds"] == []
    assert report["aggregates"]["real_only"]["accuracy"]["mean"] is None


def test_matched_seed_sample_sd_and_paired_delta():
    runs = [
        {
            "arm": arm,
            "seed": seed,
            "metrics": dict.fromkeys(
                ("accuracy", "macro_precision", "macro_recall", "macro_f1"),
                score,
            ),
        }
        for arm, seed, score in (
            ("real_only", 42, 0.25),
            ("synthetic_only", 42, 0.5),
            ("real_plus_synthetic", 42, 0.75),
            ("real_only", 43, 0.75),
            ("synthetic_only", 43, 0.5),
            ("real_plus_synthetic", 43, 0.75),
            ("real_only", 44, 1.0),
        )
    ]
    report = _comparison(runs)
    assert report["paired_seeds"] == [42, 43]
    aggregate = report["aggregates"]["real_only"]["macro_f1"]
    assert aggregate["mean"] == 0.5
    assert aggregate["sample_sd"] == pytest.approx(0.5 / 2**0.5)
    assert report["paired_macro_f1_differences"][0]["minus_real_only"] == 0.25
