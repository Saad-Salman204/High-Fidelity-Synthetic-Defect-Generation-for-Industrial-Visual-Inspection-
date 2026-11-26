"""Seed-level and class-level comparisons for the frozen classifier evaluation."""

from itertools import combinations
from statistics import mean, stdev

from ..utils.experiment_config import ARMS, CLASS_MAPPING
from .metrics import SCORES


def _summary(values: list[float]) -> dict:
    return {
        "mean": mean(values) if values else None,
        "sample_sd": stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def comparative_analysis(runs: list[dict], paired_seeds: list[int]) -> dict:
    """Summarize observed seed differences without claiming significance."""
    indexed = {(run["arm"], run["seed"]): run["metrics"] for run in runs}
    paired = sorted(seed for seed in paired_seeds if all((arm, seed) in indexed for arm in ARMS))
    pairwise = []
    for baseline, candidate in combinations(ARMS, 2):
        seeds = sorted(seed for seed in paired if (baseline, seed) in indexed)
        for score in SCORES:
            differences = [
                indexed[candidate, seed][score] - indexed[baseline, seed][score] for seed in seeds
            ]
            pairwise.append(
                {
                    "baseline_arm": baseline,
                    "candidate_arm": candidate,
                    "metric": score,
                    "candidate_minus_baseline": _summary(differences),
                    "per_seed": [
                        {"seed": seed, "difference": difference}
                        for seed, difference in zip(seeds, differences, strict=True)
                    ],
                }
            )

    class_summary = []
    class_differences = []
    for class_name, class_id in sorted(CLASS_MAPPING.items(), key=lambda item: item[1]):
        for metric in ("precision", "recall", "f1"):
            for arm in ARMS:
                values = []
                for seed in paired:
                    per_class = indexed[arm, seed]["per_class"]
                    values.append(
                        next(row[metric] for row in per_class if row["class_id"] == class_id)
                    )
                class_summary.append(
                    {
                        "class_name": class_name,
                        "class_id": class_id,
                        "metric": metric,
                        "arm": arm,
                        **_summary(values),
                    }
                )
            for candidate in ARMS[1:]:
                differences = []
                seeds = []
                for seed in paired:
                    baseline_rows = indexed[ARMS[0], seed]["per_class"]
                    candidate_rows = indexed[candidate, seed]["per_class"]
                    baseline_value = next(
                        row[metric] for row in baseline_rows if row["class_id"] == class_id
                    )
                    candidate_value = next(
                        row[metric] for row in candidate_rows if row["class_id"] == class_id
                    )
                    differences.append(candidate_value - baseline_value)
                    seeds.append(seed)
                class_differences.append(
                    {
                        "class_name": class_name,
                        "class_id": class_id,
                        "metric": metric,
                        "baseline_arm": ARMS[0],
                        "candidate_arm": candidate,
                        "candidate_minus_baseline": _summary(differences),
                        "per_seed": [
                            {"seed": seed, "difference": difference}
                            for seed, difference in zip(seeds, differences, strict=True)
                        ],
                    }
                )
    return {
        "paired_seeds": paired,
        "pairwise_metric_differences": pairwise,
        "per_class_seed_aggregates": class_summary,
        "per_class_f1_differences_vs_real_only": [
            row for row in class_differences if row["metric"] == "f1"
        ],
        "interpretation": "Descriptive seed differences only; no significance test was performed.",
    }
