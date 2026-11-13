"""Exploratory KID from locally extracted features and seeded subsets."""

from collections.abc import Callable
from random import Random
from statistics import mean, pstdev

import torch

from ..utils.experiment_config import CLASS_MAPPING

Extractor = Callable[[torch.Tensor], torch.Tensor]


def kid_subset(real: torch.Tensor, synthetic: torch.Tensor) -> float:
    """Compute unbiased polynomial MMD squared, excluding within-pool diagonals."""
    x, y = real.to(torch.float64), synthetic.to(torch.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or not x.shape[1]
        or min(len(x), len(y)) < 2
    ):
        raise ValueError("KID requires compatible feature matrices with at least two rows")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("KID features must be finite")
    xx = (x @ x.T / x.shape[1] + 1).pow(3)
    yy = (y @ y.T / x.shape[1] + 1).pow(3)
    xy = (x @ y.T / x.shape[1] + 1).pow(3)
    value = (
        (xx.sum() - xx.diagonal().sum()) / (len(x) * (len(x) - 1))
        + (yy.sum() - yy.diagonal().sum()) / (len(y) * (len(y) - 1))
        - 2 * xy.mean()
    )
    if not torch.isfinite(value):
        raise ValueError("KID calculation overflowed")
    return value.item()


def score_pools(
    real: list[dict],
    synthetic: list[dict],
    load_feature: Callable[[str, dict], torch.Tensor],
    *,
    extractor_available: bool,
    per_class_size: int = 40,
    overall_size: int = 200,
    subsets: int = 100,
    seed: int = 42,
) -> dict:
    """Sample independent subsets without replacement and cache features per image."""
    for name, value, minimum in (
        ("per_class_size", per_class_size, 2),
        ("overall_size", overall_size, 2),
        ("subsets", subsets, 1),
        ("seed", seed, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    cache = {}
    for rows in (real, synthetic):
        if len({row["image_id"] for row in rows}) != len(rows):
            raise ValueError("Feature pools must have unique image IDs")

    def get(source: str, row: dict) -> torch.Tensor:
        key = source, row["image_id"]
        if key not in cache:
            cache[key] = load_feature(source, row)
        return cache[key]

    result = {}
    for label in ("overall", *CLASS_MAPPING):
        x = sorted(
            (r for r in real if label == "overall" or r["class_name"] == label),
            key=lambda row: row["image_id"],
        )
        y = sorted(
            (r for r in synthetic if label == "overall" or r["class_name"] == label),
            key=lambda row: row["image_id"],
        )
        size = overall_size if label == "overall" else per_class_size
        reasons = []
        if min(len(x), len(y)) < size:
            reasons.append("insufficient_samples")
        if not extractor_available:
            reasons.append("feature_weights_unavailable")
        record = {
            "status": "unavailable" if reasons else "measured",
            "reasons": reasons,
            "real_count": len(x),
            "synthetic_count": len(y),
            "subset_size": size,
            "subsets_requested": subsets,
            "subsets_scored": 0,
            "seed": seed,
            "mean": None,
            "subset_sd": None,
        }
        if not reasons:
            rng = Random(seed)
            scores = []
            for _ in range(subsets):
                xs, ys = rng.sample(x, size), rng.sample(y, size)
                scores.append(
                    kid_subset(
                        torch.stack([get("real", row) for row in xs]),
                        torch.stack([get("synthetic", row) for row in ys]),
                    )
                )
            record.update(mean=mean(scores), subset_sd=pstdev(scores), subsets_scored=len(scores))
        result[label] = record
    return result
