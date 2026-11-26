"""CPU-only API contract tests; model inference is mocked at the GPU boundary."""

import csv
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from presentation.backend import main as api
from presentation.backend import runtime as runtime_module


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def evaluation_files(tmp_path, monkeypatch):
    root = tmp_path / "phase1"
    root.mkdir()
    metrics = {key: 0.8 for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    runs = [
        {
            "arm": arm,
            "seed": seed,
            "run_id": f"{arm}-{seed}",
            "checkpoint_sha256": "a" * 64,
            "metrics": metrics | {"per_class": []},
        }
        for arm in ("real_only", "synthetic_only", "real_plus_synthetic")
        for seed in (42, 43, 44)
    ]
    aggregates = {
        arm: {metric: {"mean": 0.8, "sample_sd": 0.01, "n": 3} for metric in metrics}
        for arm in ("real_only", "synthetic_only", "real_plus_synthetic")
    }
    selection = {
        "selected_candidate_count": 783,
        "near_match_selected_count": 186,
        "human_review_status": "not_manually_reviewed_or_human_approved",
    }
    report = {
        "status": "completed",
        "manifest_sha256": "b" * 64,
        "test_membership_sha256": "c" * 64,
        "test_count": 270,
        "split_counts": {"train": 1260, "val": 270, "test": 270},
        "class_counts_by_split": {"test": {"crazing": 45}},
        "device": "cpu",
        "torch": "test",
        "aggregates": aggregates,
        "runs": runs,
        "synthetic_selection": selection,
    }
    (root / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    with (root / "per-class-aggregates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["class_name", "class_id", "metric", "arm", "mean", "sample_sd", "n"],
        )
        writer.writeheader()
        for class_id, class_name in enumerate(
            ("crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches")
        ):
            for arm in ("real_only", "synthetic_only", "real_plus_synthetic"):
                writer.writerow(
                    {
                        "class_name": class_name,
                        "class_id": class_id,
                        "metric": "f1",
                        "arm": arm,
                        "mean": 0.8,
                        "sample_sd": 0.01,
                        "n": 3,
                    }
                )
    for filename, fields, rows in (
        (
            "pairwise-seed-differences.csv",
            ["baseline_arm", "candidate_arm", "metric", "seed", "difference"],
            [
                {
                    "baseline_arm": "real_only",
                    "candidate_arm": "synthetic_only",
                    "metric": "macro_f1",
                    "seed": 42,
                    "difference": 0,
                }
            ],
        ),
        (
            "comparison.csv",
            ["arm", "seed", "status", *metrics],
            [{"arm": "real_only", "seed": 42, "status": "completed", **metrics}],
        ),
    ):
        with (root / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    monkeypatch.setattr(runtime_module, "EVALUATION_ROOT", root)
    return report


def test_local_health_endpoint_does_not_load_models(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["busy"] is False


def test_research_results_are_loaded_from_structured_artifacts(client, evaluation_files):
    response = client.get("/api/research/results")
    assert response.status_code == 200
    body = response.json()
    assert len(body["runs"]) == 9
    assert body["aggregates"]["real_only"]["macro_f1"]["mean"] == 0.8
    assert body["synthetic_selection"]["near_match_selected_count"] == 186
    assert body["synthetic_selection"]["human_review_status"] == (
        "not_manually_reviewed_or_human_approved"
    )
    assert len(client.get("/api/research/experiments").json()) == 9
    assert client.get("/api/research/metrics").json()["real_only"]["accuracy"]["mean"] == 0.8


def test_incomplete_evaluation_is_not_presented(client, evaluation_files):
    evaluation_files["status"] = "incomplete"
    (runtime_module.EVALUATION_ROOT / "evaluation.json").write_text(
        json.dumps(evaluation_files), encoding="utf-8"
    )
    response = client.get("/api/research/results")
    assert response.status_code == 503


def test_overview_discloses_automated_selection_and_ceiling(client, evaluation_files):
    body = client.get("/api/project/overview").json()
    assert body["selected_synthetic"] == 783
    assert body["selected_near_matches"] == 186
    assert "not individually human-approved" in body["limitations"][1]
    assert "ceiling" in body["limitations"][-1]


def test_generation_endpoint_calls_existing_runtime_and_returns_output(client, monkeypatch):
    called = []

    def generate(class_name, seed):
        called.append((class_name, seed))
        return {"image_id": "d" * 32, "class_name": class_name, "seed": seed, "image_url": "/demo"}

    monkeypatch.setattr(api.runtime, "generate", generate)
    response = client.post("/api/generate", json={"class_name": "patches", "seed": 17})
    assert response.status_code == 200
    assert called == [("patches", 17)]
    assert response.json()["image_id"] == "d" * 32


def test_generation_rejects_malformed_request(client):
    assert client.post("/api/generate", json={"class_name": 4, "seed": "x"}).status_code == 422


def test_upload_classification_validates_image_and_routes_to_runtime(client, monkeypatch):
    called = []

    def classify(image, arm, seed):
        called.append((image.size, arm, seed))
        return {"predicted_class": "scratches", "confidence": 0.9, "arm": arm, "seed": seed}

    monkeypatch.setattr(api.runtime, "classify", classify)
    buffer = __import__("io").BytesIO()
    Image.new("RGB", (24, 24), "gray").save(buffer, format="PNG")
    response = client.post(
        "/api/classify?arm=synthetic_only&seed=43",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "scratches"
    assert called == [((24, 24), "synthetic_only", 43)]


def test_upload_classification_rejects_invalid_media(client):
    response = client.post(
        "/api/classify",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_classification_rejects_oversized_payload(client):
    from presentation.backend.runtime import MAX_UPLOAD_BYTES

    response = client.post(
        "/api/classify",
        files={"file": ("large.png", b"0" * (MAX_UPLOAD_BYTES + 1), "image/png")},
    )
    assert response.status_code == 413


def test_classify_generated_rejects_path_traversal(client):
    response = client.post("/api/classify-generated/..%2F..%2Fsecret")
    assert response.status_code == 404


def test_gallery_contract_keeps_samples_pending(client, monkeypatch):
    monkeypatch.setattr(
        api.runtime,
        "gallery",
        lambda _limit: [{"image_id": "a" * 64, "human_review_status": "pending"}],
    )
    body = client.get("/api/gallery?limit_per_class=1").json()
    assert body["total_returned"] == 1
    assert body["human_review_status"] == "exploratory_samples_only_not_acceptance_decisions"
