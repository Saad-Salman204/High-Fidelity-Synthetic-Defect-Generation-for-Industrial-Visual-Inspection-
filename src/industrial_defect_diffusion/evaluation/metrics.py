"""Compute fixed-order classifier metrics from validated per-image predictions."""

import math

from ..utils.experiment_config import CLASS_MAPPING

LABELS = tuple(sorted(CLASS_MAPPING, key=CLASS_MAPPING.__getitem__))
SCORES = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def classification_metrics(expected: dict[str, int], predictions: list[dict]) -> dict:
    """Require exactly one valid prediction for every expected test image."""
    size = len(LABELS)
    confusion = [[0] * size for _ in LABELS]
    seen = set()
    for row in predictions:
        image_id = row["image_id"]
        if image_id not in expected or image_id in seen:
            raise ValueError(f"Unexpected or duplicate test image ID: {image_id}")
        seen.add(image_id)
        actual, predicted = row["true_class"], row["predicted_class"]
        if (
            type(actual) is not int
            or type(predicted) is not int
            or not 0 <= actual < size
            or not 0 <= predicted < size
            or actual != expected[image_id]
        ):
            raise ValueError(f"Invalid class ID or test label: {image_id}")
        probabilities = [row[f"p{i}"] for i in range(size)]
        if (
            any(
                type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                for p in probabilities
            )
            or not math.isclose(sum(probabilities), 1, abs_tol=1e-6)
            or max(range(size), key=probabilities.__getitem__) != predicted
        ):
            raise ValueError(f"Invalid probabilities or prediction: {image_id}")
        confusion[actual][predicted] += 1
    if not expected or seen != expected.keys():
        raise ValueError("Predictions must cover every frozen test image")
    per_class = []
    normalized = []
    for index, label in enumerate(LABELS):
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        correct = confusion[index][index]
        per_class.append(
            {
                "class_id": index,
                "class_name": label,
                "support": support,
                "predicted_count": predicted_count,
                "precision": correct / predicted_count if predicted_count else 0.0,
                "recall": correct / support if support else 0.0,
                "f1": 2 * correct / (support + predicted_count)
                if support + predicted_count
                else 0.0,
                "absent_true_class": support == 0,
                "absent_predicted_class": predicted_count == 0,
            }
        )
        normalized.append([count / support if support else 0.0 for count in confusion[index]])
    return {
        "schema_version": 1,
        "class_order": list(LABELS),
        "test_count": len(expected),
        "accuracy": sum(confusion[i][i] for i in range(size)) / len(expected),
        **{
            f"macro_{key}": sum(row[key] for row in per_class) / size
            for key in ("precision", "recall", "f1")
        },
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_row_normalized": normalized,
    }
