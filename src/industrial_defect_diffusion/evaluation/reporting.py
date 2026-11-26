"""Export deterministic comparison tables and standalone SVG figures."""

import csv
import json
from html import escape
from pathlib import Path

from ..utils.experiment_config import ARMS
from .analysis import comparative_analysis
from .metrics import LABELS, SCORES
from .research_report import write_research_outputs


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _text(x: int, y: int, label: str, *, anchor: str = "start", color: str = "#162536") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}">{escape(label)}</text>'


def _svg(path: Path, width: int, height: int, parts: list[str]) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<g font-family="sans-serif" font-size="13">{"".join(parts)}</g></svg>\n',
        encoding="utf-8",
    )


def _confusion(path: Path, run: dict) -> None:
    metrics = run["metrics"]
    parts = [
        _text(16, 26, f"{run['arm']} · seed {run['seed']}"),
        _text(158, 54, "Predicted class → (ID in key)"),
        _text(16, 80, "True class ↓"),
    ]
    for i, label in enumerate(LABELS):
        x = 158 + i * 67
        y = 90 + i * 49
        parts.extend((_text(x + 32, 80, str(i), anchor="middle"), _text(16, y + 30, label)))
        for j, count in enumerate(metrics["confusion_matrix"][i]):
            opacity = metrics["confusion_row_normalized"][i][j]
            x = 158 + j * 67
            parts.append(
                f'<rect x="{x}" y="{y}" width="65" height="47" '
                f'fill="#18568b" fill-opacity="{opacity:.4f}" stroke="#d8e0e8"/>'
            )
            parts.append(
                _text(
                    x + 32,
                    y + 20,
                    str(count),
                    anchor="middle",
                    color="white" if opacity > 0.65 else "#162536",
                )
            )
            parts.append(
                _text(
                    x + 32,
                    y + 37,
                    f"{opacity:.0%}",
                    anchor="middle",
                    color="white" if opacity > 0.65 else "#162536",
                )
            )
    for i, label in enumerate(LABELS):
        parts.append(_text(16 + (i % 3) * 185, 414 + (i // 3) * 22, f"{i} · {label}"))
    _svg(path, 580, 453, parts)


def _performance(path: Path, report: dict) -> None:
    parts = [_text(16, 25, "Mean scores on seeds completed by all three arms")]
    if not report["paired_seeds"]:
        parts.append(_text(16, 54, "No complete matched seed. See comparison.md for run status."))
        _svg(path, 700, 75, parts)
        return
    parts.append(_text(16, 51, f"Paired seeds: {', '.join(map(str, report['paired_seeds']))}"))
    for tick in range(5):
        x = 176 + tick * 100
        parts.append(f'<path d="M {x} 76 V 365" stroke="#d8e0e8"/>')
        parts.append(_text(x, 385, f"{tick / 4:.2f}", anchor="middle"))
    colors = ("#225b91", "#9e5019", "#287366")
    for index, score in enumerate(SCORES):
        y = 81 + index * 71
        parts.append(_text(16, y + 28, score))
        for offset, (arm, color) in enumerate(zip(ARMS, colors, strict=True)):
            value = report["aggregates"][arm][score]["mean"]
            bar_y = y + offset * 18
            parts.append(
                f'<rect x="176" y="{bar_y}" width="{400 * value:.3f}" height="14" fill="{color}"/>'
            )
            parts.append(_text(183 + round(400 * value), bar_y + 12, f"{value:.3f}"))
    for index, (arm, color) in enumerate(zip(ARMS, colors, strict=True)):
        parts.append(_text(16 + index * 205, 416, arm, color=color))
    _svg(path, 700, 440, parts)


def write_results(output: Path, report: dict, runs: list[dict]) -> None:
    """Write per-run outputs and a comparison into a new directory."""
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    for run in runs:
        directory = output / run["arm"] / f"seed-{run['seed']}"
        directory.mkdir(parents=True)
        metrics = run["metrics"]
        _write_json(
            directory / "metrics.json",
            {key: value for key, value in run.items() if key != "predictions"},
        )
        _write_csv(
            directory / "predictions.csv",
            run["predictions"],
            ["run_id", "image_id", "true_class", "predicted_class", *(f"p{i}" for i in range(6))],
        )
        _write_csv(directory / "per-class.csv", metrics["per_class"], list(metrics["per_class"][0]))
        _confusion(directory / "confusion.svg", run)
        summary.append(
            {
                "arm": run["arm"],
                "seed": run["seed"],
                "status": "completed",
                **{score: metrics[score] for score in SCORES},
            }
        )
    summary.extend(
        {key: item[key] for key in ("arm", "seed", "status")} for item in report["incomplete"]
    )
    _write_csv(output / "comparison.csv", summary, ["arm", "seed", "status", *SCORES])
    analysis = comparative_analysis(runs, report["paired_seeds"])
    _write_json(output / "comparative-analysis.json", analysis)
    pair_rows = []
    for comparison in analysis["pairwise_metric_differences"]:
        for result in comparison["per_seed"]:
            pair_rows.append(
                {
                    "baseline_arm": comparison["baseline_arm"],
                    "candidate_arm": comparison["candidate_arm"],
                    "metric": comparison["metric"],
                    **result,
                }
            )
    _write_csv(
        output / "pairwise-seed-differences.csv",
        pair_rows,
        ["baseline_arm", "candidate_arm", "metric", "seed", "difference"],
    )
    _write_csv(
        output / "per-class-aggregates.csv",
        analysis["per_class_seed_aggregates"],
        ["class_name", "class_id", "metric", "arm", "mean", "sample_sd", "n"],
    )
    _write_csv(
        output / "per-class-f1-differences.csv",
        [
            {
                "class_name": row["class_name"],
                "baseline_arm": row["baseline_arm"],
                "candidate_arm": row["candidate_arm"],
                "mean_difference": row["candidate_minus_baseline"]["mean"],
                "sample_sd": row["candidate_minus_baseline"]["sample_sd"],
                "n": row["candidate_minus_baseline"]["n"],
            }
            for row in analysis["per_class_f1_differences_vs_real_only"]
        ],
        [
            "class_name",
            "baseline_arm",
            "candidate_arm",
            "mean_difference",
            "sample_sd",
            "n",
        ],
    )
    table = [
        "# Frozen test comparison",
        "",
        "| Arm | Seed | Status | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        scores = [f"{row[key]:.4f}" if key in row else "—" for key in SCORES]
        table.append(
            "| " + " | ".join(map(str, (row["arm"], row["seed"], row["status"], *scores))) + " |"
        )
    table.extend(
        (
            "",
            "## Matched seed summaries",
            "",
            "| Arm | Metric | Mean | Sample SD |",
            "| --- | --- | ---: | ---: |",
        )
    )
    for arm in ARMS:
        for score, values in report["aggregates"][arm].items():
            formatted = [
                f"{values[key]:.4f}" if values[key] is not None else "—"
                for key in ("mean", "sample_sd")
            ]
            table.append(f"| {arm} | {score} | {' | '.join(formatted)} |")
    table.extend(
        (
            "",
            "Only seeds completed in every arm enter aggregate comparisons. "
            "Sample SD describes variability across seeds and requires at least two.",
            "Inspect the configuration status in evaluation.json to identify fixture runs.",
        )
    )
    (output / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    _performance(output / "performance.svg", report)
    from .visualizations import write_visualizations

    write_visualizations(output, report, analysis, runs)
    _write_json(output / "evaluation.json", report)
    write_research_outputs(output, report, analysis)
