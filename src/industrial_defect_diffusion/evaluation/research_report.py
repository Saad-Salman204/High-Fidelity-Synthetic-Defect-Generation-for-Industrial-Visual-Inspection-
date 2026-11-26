"""Render research and jury-facing Markdown from verified evaluation metrics."""

# E501 is disabled here because the module embeds long-form report prose in
# Markdown templates; the rendered document retains readable paragraph flow.
# ruff: noqa: E501

from pathlib import Path

from ..utils.experiment_config import ARMS

ARM_LABELS = {
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


def _mean_sd(value: dict) -> str:
    if value["mean"] is None:
        return "—"
    if value["sample_sd"] is None:
        return f"{value['mean']:.4f} (n={value['n']})"
    return f"{value['mean']:.4f} ± {value['sample_sd']:.4f}"


def _tables(report: dict, analysis: dict) -> tuple[str, str]:
    aggregates = report["aggregates"]
    results = [
        "| Arm | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        cells = [_mean_sd(aggregates[arm][score]) for score in METRIC_LABELS]
        results.append(f"| {ARM_LABELS[arm]} | " + " | ".join(cells) + " |")
    per_class = [
        "| Class | Real-only F1 | Synthetic-only F1 | Real + synthetic F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    f1_rows = {
        (row["class_name"], row["arm"]): row
        for row in analysis["per_class_seed_aggregates"]
        if row["metric"] == "f1"
    }
    for class_name in sorted(
        {row["class_name"] for row in analysis["per_class_seed_aggregates"]},
        key=lambda name: next(
            row["class_id"]
            for row in analysis["per_class_seed_aggregates"]
            if row["class_name"] == name
        ),
    ):
        cells = [_mean_sd(f1_rows[class_name, arm]) for arm in ARMS]
        per_class.append(f"| {class_name} | " + " | ".join(cells) + " |")
    return "\n".join(results), "\n".join(per_class)


def _findings(report: dict, analysis: dict) -> list[str]:
    comparisons = {
        (item["baseline_arm"], item["candidate_arm"], item["metric"]): item
        for item in analysis["pairwise_metric_differences"]
    }
    sentences = []
    for metric in ("accuracy", "macro_f1"):
        item = comparisons.get(("real_only", "real_plus_synthetic", metric))
        if item is None or item["candidate_minus_baseline"]["mean"] is None:
            continue
        change = item["candidate_minus_baseline"]["mean"]
        direction = "higher" if change > 0 else "lower" if change < 0 else "unchanged"
        sentences.append(
            f"Across matched seeds, the real + synthetic arm had {direction} {METRIC_LABELS[metric].lower()} "
            f"than real-only by {change:+.4f} on average (difference in 0–1 score units)."
        )
    class_diffs = analysis["per_class_f1_differences_vs_real_only"]
    mixed = [row for row in class_diffs if row["candidate_arm"] == "real_plus_synthetic"]
    if mixed:
        largest_up = max(mixed, key=lambda row: row["candidate_minus_baseline"]["mean"])
        largest_down = min(mixed, key=lambda row: row["candidate_minus_baseline"]["mean"])
        sentences.append(
            "The largest mean class-level F1 increase for real + synthetic versus real-only was "
            f"{largest_up['class_name']} ({largest_up['candidate_minus_baseline']['mean']:+.4f}); "
            "the largest decrease was "
            f"{largest_down['class_name']} ({largest_down['candidate_minus_baseline']['mean']:+.4f})."
        )
    if not sentences:
        sentences.append("No complete matched-seed comparison was available for these findings.")
    return sentences


def write_research_outputs(output: Path, report: dict, analysis: dict) -> None:
    """Write a detailed research report and concise presentation-facing briefs."""
    table, per_class_table = _tables(report, analysis)
    selection = report.get("synthetic_selection") or {}
    config = report["resolved_config"]
    findings = _findings(report, analysis)
    class_rows = report.get("class_counts_by_split", {})
    class_counts_text = "\n".join(
        "| "
        + class_name
        + " | "
        + " | ".join(
            str(class_rows.get(split, {}).get(class_name, 0)) for split in ("train", "val", "test")
        )
        + " |"
        for class_name in config["classes"]
    )
    result_state = "completed" if report["status"] == "completed" else "incomplete"
    research = (
        f"""# Synthetic defect augmentation for industrial visual inspection

## Abstract

This report evaluates the existing Stable Diffusion 1.5 + LoRA synthetic-defect workflow through three ResNet-18 arms: real-only, synthetic-only, and real + synthetic. Each arm has three seeds and all models are scored on the same frozen real test set of {report["test_count"]} images. The evaluator completed {len(report["runs"])} run evaluations with status **{result_state}**. The synthetic pool was selected by the documented automated audit policy; its candidates were not individually human-approved. Measured arm-level results and paired-seed differences are shown below. No significance test was performed.

## Research problem and motivation

Industrial surface-defect datasets are limited and class distributions can constrain supervised model training. This project tests whether generated NEU-CLS-like defect images can provide a useful additional training source for a six-class ResNet-18 classifier. The central comparison is whether mixing the selected synthetic pool with real training images changes performance on held-out real data relative to real-only training.

## Objectives

1. Train and compare the three prespecified data-source arms with matched seeds and a shared training budget.
2. Evaluate every completed run on the same frozen real test partition.
3. Report aggregate and per-class behavior, including degradations and variability across seeds.

## Dataset and frozen split

The real data are the six NEU-CLS classes with canonical IDs from the approved experiment configuration. Split provenance: manifest SHA-256 `{report["manifest_sha256"]}`, source fingerprint `{report["dataset_source_sha256"]}`, seed {config["split"]["seed"]}, ratios 70/15/15. The evaluator checked the manifest and metadata before and after inference, and each test image was loaded through the frozen manifest with its recorded file hash.

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
{class_counts_text}

Total partition sizes are train={report["split_counts"]["train"]}, validation={report["split_counts"]["val"]}, and test={report["split_counts"]["test"]}.

## Stable Diffusion 1.5 and LoRA methodology

The recorded base model is `{config["diffusion"]["model_id"]}` at revision `{config["diffusion"]["revision"]}`. The adapter configuration uses rank {config["diffusion"]["lora_rank"]} and alpha {config["diffusion"]["lora_alpha"]}, with {config["diffusion"]["max_steps"]} optimizer updates. Generator training was completed before the classifier comparison; this phase launched no diffusion or classifier training.

## Synthetic generation, audit, and selection

The generation run produced 996 candidates (166 per class) using the saved generation plan. The classifier pool has {selection.get("selected_candidate_count", "not recorded")} candidates, selected under `{selection.get("selection_id", "not recorded")}`. The policy excludes any candidate with an audit blocker or warning and retains near-match relationships when neither is present. The selection has {selection.get("near_match_selected_count", 0)} candidates with near-match links; these were retained, not automatically rejected. Integrity/decode exclusions recorded in the selection report: {len(selection.get("excluded_integrity_failures", []))}. This selection was automated eligibility filtering, not individual visual acceptance or human approval.

Selection manifest SHA-256: `{selection.get("selection_manifest_sha256", "not recorded")}`. Source audit SHA-256: `{selection.get("source_audit_sha256", "not recorded")}`. Source generation-manifest SHA-256: `{selection.get("source_generation_manifest_sha256", "not recorded")}`.

| Class | Selected synthetic candidates |
| --- | ---: |
"""
        + "\n".join(
            f"| {label} | {count} |"
            for label, count in selection.get("selected_counts_by_class", {}).items()
        )
        + f"""

## ResNet-18 architecture and training

Each arm used torchvision ResNet-18 initialized from ImageNet-1K V1 weights. The original 1,000-class final layer was replaced by a six-class linear layer. **The full backbone and replacement classifier were trainable**: model parameters are passed to AdamW without freezing any layers, and the implementation does not set `requires_grad=False`. The initialization file hash recorded by all runs was `{report.get("initial_weights_sha256", "recorded identically in every run")}`.

Training used AdamW, cosine learning-rate scheduling, batch size {config["classification"]["batch_size"]}, {config["classification"]["max_steps"]} optimizer updates, and validation every {config["classification"]["validation_interval"]} updates. The real + synthetic arm used a configured synthetic fraction of {config["classification"]["mixed_synthetic_fraction"]:.2f}. The earliest checkpoint with the best validation macro F1 was saved. Validation—not test—metrics governed checkpoint selection. Training code exposed only real train rows (when applicable) and synthetic train rows to optimizers; validation rows were used for model selection. The test split has no path in the training loader.

## Experimental arms and seed design

The run selection contains all nine configured arm/seed combinations: seeds {", ".join(map(str, config["classification"]["seeds"]))}; each record reports {config["classification"]["max_steps"]} updates and the same frozen configuration hash `{report["config_sha256"]}`. All arms used the same frozen validation IDs, test manifest, model initialization hash, and training commit. The synthetic arms reference the same selected-manifest hash.

## Evaluation metrics

For each run, the evaluator generated one prediction per frozen test image and computed accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1, support counts, and a 6×6 confusion matrix in canonical class order. Macro scores weight the six classes equally. Aggregates use the three matched seeds; standard deviations are sample SDs.

## Experimental results

### Aggregate scores across seeds

{table}

### Per-class F1 across seeds

{per_class_table}

### Paired comparisons

{chr(10).join(f"- {finding}" for finding in findings)}

All pairwise per-seed metric differences are saved in `pairwise-seed-differences.csv`; class-level summaries and real-only deltas are in `per-class-aggregates.csv` and `per-class-f1-differences.csv`. Differences are descriptive. Three seeds provide limited precision, and results do not establish statistical significance or causal generalization beyond this split.

## Discussion

Interpret the mixed-vs-real-only difference as the primary estimate of synthetic augmentation under this specific training recipe and automated candidate policy. Synthetic-only performance measures transfer from the selected generated pool to real held-out images. Per-class tables and confusion matrices should be considered alongside macro averages: a higher aggregate can conceal class-specific losses. Automated eligibility flags do not establish visual defect fidelity; prior contact-sheet review sampled only 72 images and was not an acceptance review of these 783 candidates.

## Limitations and threats to validity

- Only one frozen 70/15/15 partition and three random seeds were evaluated.
- The synthetic pool was not individually human-approved; the eligibility procedure is automated and near-match relationships remain.
- Earlier visual review covered a 72-image sample, not the full candidate pool.
- Classifier results do not independently validate synthetic image realism or industrial usefulness.
- The analysis is descriptive; no inferential significance tests were run.
- Results are specific to this ResNet-18 initialization, preprocessing, training budget, sampling design, and test set.
- No external production-line dataset or cross-site generalization test was included.

## Conclusion and future work

The measured answer to whether augmentation helped is the paired real + synthetic minus real-only comparison above, with its three per-seed values and per-class differences. Treat this result as evidence for this protocol only. Future work should preregister any revised candidate policy, obtain a documented visual review for image-quality claims, and evaluate on an independent acquisition source before making deployment claims.

## Provenance

Evaluation status: {report["status"]}. Test membership fingerprint: `{report["test_membership_sha256"]}`. Frozen run selection fingerprint: `{report["selection_file_sha256"]}`. Evaluation configuration fingerprint: `{report["config_sha256"]}`. The complete per-run predictions, metrics, confusion matrices, training-record references, software/hardware details, and input hashes are in the adjacent machine-readable evaluation files.
"""
    )
    (output / "research-report.md").write_text(research, encoding="utf-8")

    executive = "\n".join(
        [
            "# Executive summary",
            "",
            f"All {len(report['runs'])} selected classifier runs were evaluated on the same {report['test_count']} real test images; evaluation status: **{report['status']}**.",
            "",
            "## Results",
            "",
            table,
            "",
            "## Main measured comparisons",
            "",
            *[f"- {finding}" for finding in findings],
            "",
            "## Interpretation limits",
            "",
            "The synthetic pool was selected by automated blocker/warning filtering and was not individually human-approved. Near-match candidates were retained under that policy. Results use one fixed real split and three seeds; no significance test was performed.",
        ]
    )
    (output / "executive-summary.md").write_text(executive + "\n", encoding="utf-8")

    jury = "\n".join(
        [
            "# Jury brief",
            "",
            "## Study in one minute",
            "",
            "We fine-tuned Stable Diffusion 1.5 with LoRA on the frozen real training partition, generated defect candidates, filtered them with a documented automated audit policy, and compared three ResNet-18 data-source arms. All arms use the same real validation/test partitions and three seeds.",
            "",
            "## Strongest measured evidence",
            "",
            *[f"- {finding}" for finding in findings],
            "",
            "## Likely jury questions and accurate answers",
            "",
            "- **Were all selected synthetic images approved by a person?** No. The 783-image pool was selected by automated eligibility rules. It was not individually human-approved.",
            "- **Were near-duplicates removed?** No. Eligible candidates with near-match relationships were retained as the documented policy specified; they are disclosed in selection provenance.",
            "- **Did the test set guide model selection?** The training implementation selected checkpoints from real validation macro F1. The evaluation stage used the held-out test split for final scoring only.",
            "- **Does a gain prove statistical significance?** No. These are descriptive results over three seeds; no significance test was performed.",
            "- **Was the ResNet backbone frozen?** No. The implementation optimizes the model parameter set after replacing the final classifier layer, so the full backbone is fine-tuned.",
            "- **Does classifier performance prove image realism?** No. It measures transfer to this held-out real split under this setup, not industrial acceptance or generalization to new sites.",
            "",
            "## Limitations to state openly",
            "",
            "One frozen split, three seeds, an automated but not human-approved synthetic pool, and no external industrial test set. The earlier visual review sampled 72 candidates and cannot be generalized to the complete pool.",
        ]
    )
    (output / "jury-brief.md").write_text(jury + "\n", encoding="utf-8")

    outline = """# Suggested presentation structure

1. **Problem and motivation** — NEU-CLS class scarcity and the synthetic augmentation question.
2. **Dataset and leakage controls** — class mapping, frozen 70/15/15 split, manifest hash, shared test set.
3. **Diffusion + LoRA pipeline** — base revision, train-only ancestry, rank/alpha, 1,000 updates.
4. **Candidate audit and selection** — 996 candidates; 783 selected under the automated no-blocker/no-warning policy; disclose no individual human approval and retained near matches.
5. **Classifier and experiment design** — fully fine-tuned ResNet-18, three arms, seeds 42/43/44, equal 1,000-step budget.
6. **Aggregate test results** — accuracy and macro F1 by arm with seed variability.
7. **Class-wise behavior** — per-class F1 and confusion matrices; highlight both gains and regressions.
8. **Interpretation and limitations** — descriptive n=3, one split, automated quality screen, no external dataset.
9. **Conclusion and next experiment** — state the measured mixed-vs-real result and propose preregistered follow-up.
"""
    (output / "presentation-outline.md").write_text(outline, encoding="utf-8")
