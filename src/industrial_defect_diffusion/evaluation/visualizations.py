"""Publication-style Matplotlib figures for measured classifier results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..utils.experiment_config import ARMS, CLASS_MAPPING
from .metrics import SCORES

COLORS = {"real_only": "#35618d", "synthetic_only": "#d58b2a", "real_plus_synthetic": "#3c8876"}
LABELS = {
    "real_only": "Real-only",
    "synthetic_only": "Synthetic-only",
    "real_plus_synthetic": "Real + synthetic",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_f1": "Macro F1",
}
CLASS_LABELS = tuple(sorted(CLASS_MAPPING, key=CLASS_MAPPING.__getitem__))


def _save(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _seed_score_fig(output: Path, runs: list[dict], score: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for index, arm in enumerate(ARMS):
        selected = sorted((run for run in runs if run["arm"] == arm), key=lambda r: r["seed"])
        values = [run["metrics"][score] for run in selected]
        seeds = [run["seed"] for run in selected]
        if values:
            ax.errorbar(
                [index] * len(values),
                values,
                yerr=None,
                fmt="o",
                color=COLORS[arm],
                markersize=8,
                label=f"{LABELS[arm]} seeds",
                alpha=0.9,
            )
            offset = {
                seed: delta
                for seed, delta in zip(
                    seeds,
                    np.linspace(-0.08, 0.08, len(seeds)),
                    strict=True,
                )
            }
            for seed, value in zip(seeds, values, strict=True):
                ax.scatter(index + offset[seed], value, color=COLORS[arm], zorder=4)
                ax.annotate(
                    str(seed),
                    (index + offset[seed], value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            if len(values) > 1:
                ax.errorbar(
                    index,
                    np.mean(values),
                    yerr=np.std(values, ddof=1),
                    fmt="_",
                    color="#222222",
                    capsize=5,
                    markersize=22,
                    linewidth=1.5,
                    label="Mean ± sample SD" if index == 0 else None,
                    zorder=5,
                )
    ax.set_xticks(range(len(ARMS)), [LABELS[arm] for arm in ARMS])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel(METRIC_LABELS[score])
    ax.set_title(f"{METRIC_LABELS[score]} by experimental arm and seed")
    ax.grid(axis="y", color="#d9e0e8", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, output, stem)


def _metric_means_fig(output: Path, runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharey=True, layout="constrained")
    for ax, score in zip(axes.flat, SCORES, strict=True):
        for index, arm in enumerate(ARMS):
            values = [run["metrics"][score] for run in runs if run["arm"] == arm]
            if values:
                ax.errorbar(
                    index,
                    np.mean(values),
                    yerr=np.std(values, ddof=1) if len(values) > 1 else 0,
                    fmt="o",
                    color=COLORS[arm],
                    capsize=5,
                    markersize=7,
                )
        ax.set_title(METRIC_LABELS[score])
        ax.set_xticks(range(len(ARMS)), [LABELS[arm] for arm in ARMS], rotation=15, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", color="#d9e0e8", linewidth=0.8)
    fig.suptitle("Mean ± sample standard deviation across seeds")
    _save(fig, output, "05_mean_sd_all_metrics")


def _precision_recall_fig(output: Path, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    offsets = {"macro_precision": -0.12, "macro_recall": 0.12}
    markers = {"macro_precision": "o", "macro_recall": "s"}
    for score in offsets:
        for index, arm in enumerate(ARMS):
            values = [run["metrics"][score] for run in runs if run["arm"] == arm]
            if not values:
                continue
            ax.errorbar(
                index + offsets[score],
                np.mean(values),
                yerr=np.std(values, ddof=1) if len(values) > 1 else 0,
                fmt=markers[score],
                color=COLORS[arm],
                capsize=4,
                label=f"{METRIC_LABELS[score]} · {LABELS[arm]}" if index == 0 else None,
            )
    ax.set_xticks(range(len(ARMS)), [LABELS[arm] for arm in ARMS])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Macro precision and recall, mean ± sample SD")
    ax.grid(axis="y", color="#d9e0e8", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, output, "04_precision_recall")


def _confusion_fig(output: Path, runs: list[dict], arm: str) -> None:
    selected = [run for run in runs if run["arm"] == arm]
    if not selected:
        return
    matrix = np.sum(
        [np.asarray(run["metrics"]["confusion_matrix"], dtype=float) for run in selected], axis=0
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
    fig, ax = plt.subplots(figsize=(7.3, 6.2), layout="constrained")
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}\n{normalized[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.58 else "#182433",
                fontsize=9,
            )
    ax.set_xticks(range(6), CLASS_LABELS, rotation=35, ha="right")
    ax.set_yticks(range(6), CLASS_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{LABELS[arm]} · pooled counts across {len(selected)} seeds")
    fig.colorbar(image, ax=ax, label="Share of true-class images")
    _save(fig, output, f"06_confusion_{arm}")


def _per_class_f1_fig(output: Path, analysis: dict) -> None:
    rows = [row for row in analysis["per_class_seed_aggregates"] if row["metric"] == "f1"]
    fig, ax = plt.subplots(figsize=(10.2, 5.3), layout="constrained")
    x = np.arange(len(CLASS_LABELS))
    width = 0.24
    for index, arm in enumerate(ARMS):
        lookup = {row["class_name"]: row for row in rows if row["arm"] == arm}
        means = [lookup[label]["mean"] or 0 for label in CLASS_LABELS]
        errors = [lookup[label]["sample_sd"] or 0 for label in CLASS_LABELS]
        ax.bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=COLORS[arm],
            label=LABELS[arm],
        )
    ax.set_xticks(x, CLASS_LABELS, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 · mean ± sample SD across seeds")
    ax.grid(axis="y", color="#d9e0e8", linewidth=0.8)
    ax.legend(frameon=False)
    _save(fig, output, "09_per_class_f1")


def _design_fig(output: Path, report: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.4), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, width, height, text, color):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012",
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)

    box(0.03, 0.68, 0.26, 0.18, "Frozen NEU-CLS training split\nResNet-18 real-only", "#e8f0f7")
    box(
        0.37,
        0.68,
        0.26,
        0.18,
        "783 automated-policy candidates\nResNet-18 synthetic-only",
        "#f9eedb",
    )
    box(0.71, 0.68, 0.26, 0.18, "Real train + synthetic pool\nResNet-18 mixed arm", "#e4f1ed")
    box(
        0.37,
        0.37,
        0.26,
        0.16,
        "Shared real validation split\ncheckpoint selection by macro F1",
        "#eef0f8",
    )
    box(
        0.29,
        0.07,
        0.42,
        0.18,
        f"Shared frozen real test split · {report['test_count']} images\n"
        "opened only for final evaluation",
        "#f5e8e8",
    )
    for start_x in (0.16, 0.50, 0.84):
        ax.add_patch(
            FancyArrowPatch((start_x, 0.67), (0.50, 0.54), arrowstyle="-|>", mutation_scale=12)
        )
    ax.add_patch(FancyArrowPatch((0.50, 0.36), (0.50, 0.26), arrowstyle="-|>", mutation_scale=12))
    ax.text(
        0.5,
        0.93,
        "Three-arm ResNet-18 comparison · same split, initialization, preprocessing and budget",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.5,
        0.01,
        "Synthetic pool selected by automated blocker/warning policy; "
        "not individually human-approved.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#7a341e",
    )
    _save(fig, output, "01_experimental_design")


def _pairwise_fig(output: Path, analysis: dict) -> None:
    rows = [
        row
        for row in analysis["pairwise_metric_differences"]
        if row["metric"] == "macro_f1" and row["candidate_minus_baseline"]["mean"] is not None
    ]
    if not rows:
        return
    labels = [f"{LABELS[row['candidate_arm']]} − {LABELS[row['baseline_arm']]}" for row in rows]
    means = [row["candidate_minus_baseline"]["mean"] for row in rows]
    errors = [row["candidate_minus_baseline"]["sample_sd"] or 0 for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
    colors = [COLORS[row["candidate_arm"]] for row in rows]
    y = np.arange(len(rows))
    ax.barh(y, means, xerr=errors, color=colors, alpha=0.9, capsize=4)
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired macro-F1 difference (candidate minus baseline)")
    ax.set_title("Descriptive paired-seed macro-F1 differences")
    ax.grid(axis="x", color="#d9e0e8", linewidth=0.8)
    _save(fig, output, "10_pairwise_macro_f1_differences")


def write_visualizations(output: Path, report: dict, analysis: dict, runs: list[dict]) -> None:
    """Write measured PNG and PDF figures under the evaluation output directory."""
    figures = output / "figures"
    figures.mkdir()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.facecolor": "white",
        }
    )
    _design_fig(figures, report)
    _seed_score_fig(figures, runs, "accuracy", "02_accuracy_by_seed")
    _seed_score_fig(figures, runs, "macro_f1", "03_macro_f1_by_seed")
    _precision_recall_fig(figures, runs)
    _metric_means_fig(figures, runs)
    for arm in ARMS:
        _confusion_fig(figures, runs, arm)
    _per_class_f1_fig(figures, analysis)
    _pairwise_fig(figures, analysis)
