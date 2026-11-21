"""day13_evaluation.py — Deterministic S1–S4 Hybrid Gate Evaluation on Development Benchmark.

Implements Day 13 evaluation requirements:
- Strictly guarded development partition evaluation (159 programs, 53 families).
- Zero access to frozen test partition (protected by TestPartitionAccessError).
- Evaluates four distinct evaluation modes:
    * S1: Static-Only (deterministic AST/CFG/STL rules; abstains on inconclusive)
    * S2: ML-Only (Day 11 winning Random Forest classifier OOF predictions)
    * S3: Naive Hybrid (static-first; un-gated fallback on any inconclusive static result)
    * S4: Hybrid with Verification and Abstention (reason-aware fallback, score >= 0.55)
- Strict program_id alignment verification between static analysis and ML OOF predictions.
- Full performance metrics (Accuracy, Macro-F1, per-class, confusion matrices, abstention metrics).
- Detailed decision bucket breakdown for S4 (Static, ML Fallback, Abstained).
- 1000-replicate family-cluster bootstrap (seed=42) for 95% CIs.
- S4 threshold sensitivity curve across tau in [0.0, 1.0].
- Output stored strictly in gitignored artifacts/day13/.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from complexity_lens.complexity.models import AnalysisResult
from complexity_lens.evaluation.cv_runner import (
    CrossValidationRunner,
    OutOfFoldPrediction,
    compute_fold_assignment_hash,
    get_git_info,
)
from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    EvaluationRecord,
    load_development_dataset,
)
from complexity_lens.evaluation.rigorous import (
    CANONICAL_CLASS_TO_INDEX,
    compute_abstention_metrics,
    compute_accuracy,
    compute_confusion_matrix,
    compute_family_cluster_bootstrap,
    compute_macro_f1,
    compute_per_class_metrics,
    evaluate_static_only_engine,
)
from complexity_lens.fusion import (
    DEFAULT_ML_THRESHOLD,
    EvaluationMode,
    FusionResult,
    MLPrediction,
    decide_fusion,
)

STATIC_DISCLOSURE: str = "Static results were developed with development data visible."
RF_SELECTION_DISCLOSURE: str = (
    "Random Forest candidate model selected via 3-fold grouped family cross-validation "
    "on development partition."
)
DEFAULT_DAY13_OUTPUT_DIR: str = "artifacts/day13"


def align_and_verify_program_ids(
    records: Sequence[EvaluationRecord],
    oof_predictions: Sequence[OutOfFoldPrediction],
) -> None:
    """Strictly verify 1-to-1 program_id alignment between dataset records and ML predictions."""
    if len(records) != len(oof_predictions):
        raise ValueError(
            f"Record count mismatch: dataset has {len(records)} records, "
            f"ML OOF has {len(oof_predictions)} predictions."
        )

    for idx, (rec, oof) in enumerate(zip(records, oof_predictions)):
        if rec.program_id != oof.program_id:
            raise ValueError(
                f"Program ID mismatch at index {idx}: dataset record '{rec.program_id}' "
                f"!= ML OOF prediction '{oof.program_id}'."
            )


def build_ml_prediction_containers(
    oof_predictions: Sequence[OutOfFoldPrediction],
) -> list[MLPrediction]:
    """Convert raw OutOfFoldPrediction objects into strictly validated MLPrediction containers."""
    containers: list[MLPrediction] = []
    for oof in oof_predictions:
        probs_dict = {
            cls_name: float(oof.probabilities[idx])
            for idx, cls_name in enumerate(CANONICAL_CLASS_ORDER)
        }
        pred_idx = CANONICAL_CLASS_TO_INDEX.get(oof.predicted_label, -1)
        top_prob = (
            float(oof.probabilities[pred_idx])
            if pred_idx >= 0
            else float(max(oof.probabilities))
        )
        containers.append(
            MLPrediction(
                predicted_label=oof.predicted_label,
                probability=top_prob,
                probabilities=probs_dict,
                model_id="random_forest_oof",
            )
        )
    return containers


def evaluate_mode(
    records: Sequence[EvaluationRecord],
    static_results: Sequence[AnalysisResult | None],
    ml_predictions: Sequence[MLPrediction],
    mode: EvaluationMode,
    threshold: float = DEFAULT_ML_THRESHOLD,
) -> tuple[list[str], list[FusionResult], dict[str, Any]]:
    """Evaluate a specific regime (S1, S2, S3, or S4) across all programs."""
    y_true = [r.label for r in records]
    preds: list[str] = []
    fusion_results: list[FusionResult] = []

    for s_res, ml_pred in zip(static_results, ml_predictions):
        res = decide_fusion(s_res, ml_pred, mode=mode, ml_threshold=threshold)
        fusion_results.append(res)
        preds.append(res.primary_label)

    acc = compute_accuracy(y_true, preds)
    per_cls = compute_per_class_metrics(y_true, preds)
    macro_f1 = compute_macro_f1(per_cls)
    cm = compute_confusion_matrix(y_true, preds)
    abstention = compute_abstention_metrics(y_true, preds)

    summary: dict[str, Any] = {
        "mode": mode.value,
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "coverage": round(abstention.coverage, 6),
        "selective_accuracy": round(abstention.selective_accuracy, 6),
        "selective_risk": round(abstention.selective_risk, 6),
        "unknown_recall": round(abstention.unknown_recall, 6)
        if abstention.unknown_recall is not None
        else None,
        "unknown_precision": round(abstention.unknown_precision, 6)
        if abstention.unknown_precision is not None
        else None,
        "confusion_matrix": cm,
        "per_class": {k: v.to_dict() for k, v in per_cls.items()},
    }

    # Add S4-specific bucket breakdown
    if mode == EvaluationMode.S4:
        static_indices = [i for i, r in enumerate(fusion_results) if r.source == "STATIC"]
        fallback_indices = [i for i, r in enumerate(fusion_results) if r.source == "ML_ESTIMATE"]
        abstained_indices = [i for i, r in enumerate(fusion_results) if r.source == "UNKNOWN"]

        static_correct = sum(1 for i in static_indices if preds[i] == y_true[i])
        fallback_correct = sum(1 for i in fallback_indices if preds[i] == y_true[i])
        abstained_correct = sum(1 for i in abstained_indices if y_true[i] == "UNKNOWN")

        summary["buckets"] = {
            "static_decisions": {
                "count": len(static_indices),
                "correct": static_correct,
                "accuracy": round(static_correct / len(static_indices), 6)
                if static_indices
                else 0.0,
            },
            "ml_fallback_decisions": {
                "count": len(fallback_indices),
                "correct": fallback_correct,
                "accuracy": round(fallback_correct / len(fallback_indices), 6)
                if fallback_indices
                else 0.0,
            },
            "abstained_decisions": {
                "count": len(abstained_indices),
                "gold_unknown_count": abstained_correct,
                "accuracy_as_abstention": round(abstained_correct / len(abstained_indices), 6)
                if abstained_indices
                else 0.0,
            },
        }

    return preds, fusion_results, summary


def compute_s4_sensitivity_curve(
    records: Sequence[EvaluationRecord],
    static_results: Sequence[AnalysisResult | None],
    ml_predictions: Sequence[MLPrediction],
    threshold_grid: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Compute S4 decision curve across various ML confidence thresholds."""
    if threshold_grid is None:
        threshold_grid = [round(float(t), 2) for t in np.arange(0.0, 1.05, 0.05)]

    y_true = [r.label for r in records]
    curve: list[dict[str, Any]] = []

    for th in threshold_grid:
        preds, f_results, _ = evaluate_mode(
            records,
            static_results,
            ml_predictions,
            mode=EvaluationMode.S4,
            threshold=th,
        )
        acc = compute_accuracy(y_true, preds)
        per_cls = compute_per_class_metrics(y_true, preds)
        macro_f1 = compute_macro_f1(per_cls)
        abstention = compute_abstention_metrics(y_true, preds)

        fallback_count = sum(1 for r in f_results if r.source == "ML_ESTIMATE")
        static_count = sum(1 for r in f_results if r.source == "STATIC")
        abstained_count = sum(1 for r in f_results if r.source == "UNKNOWN")

        curve.append(
            {
                "threshold": th,
                "accuracy": round(acc, 6),
                "macro_f1": round(macro_f1, 6),
                "coverage": round(abstention.coverage, 6),
                "selective_accuracy": round(abstention.selective_accuracy, 6),
                "selective_risk": round(abstention.selective_risk, 6),
                "static_count": static_count,
                "fallback_count": fallback_count,
                "abstained_count": abstained_count,
            }
        )

    return curve


def run_day13_evaluation(
    dataset_dir: str = "datasets",
    manifest_path: str = "data/splits/split_manifest_v1.0.frozen.json",
    output_dir: str = DEFAULT_DAY13_OUTPUT_DIR,
    threshold: float = DEFAULT_ML_THRESHOLD,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute complete Day 13 S1–S4 evaluation pipeline and write artifacts."""
    print("=" * 70)
    print("ComplexityLens — Day 13: S1–S4 Hybrid Gate & Verification Evaluation")
    print("=" * 70)

    # 1. Load development dataset safely
    print("\n[1/6] Loading development dataset (frozen test access strictly blocked)...")
    dev_dataset = load_development_dataset(manifest_path, dataset_dir)
    n_programs = len(dev_dataset.records)
    n_families = len(set(r.family_id for r in dev_dataset.records))
    print(f"  Loaded {n_programs} programs across {n_families} families.")

    # 2. Evaluate static-only engine
    print("\n[2/6] Evaluating static-only complexity engine on development partition...")
    _, raw_static_results = evaluate_static_only_engine(dev_dataset)
    print(f"  Evaluated {len(raw_static_results)} static AST/CFG programs.")

    # 3. Obtain Day 11 Random Forest out-of-fold predictions
    print("\n[3/6] Generating Day 11 Random Forest 3-fold CV out-of-fold predictions...")
    runner = CrossValidationRunner(
        manifest_path=manifest_path,
        dataset_root=dataset_dir,
        dataset=dev_dataset,
    )
    rf_cv_result = runner.evaluate_model("random_forest")
    rf_oof = rf_cv_result.oof_predictions
    print(f"  Obtained {len(rf_oof)} OOF predictions.")

    # 4. Strict alignment check
    print("\n[4/6] Verifying strict program_id alignment...")
    oof_by_id = {oof.program_id: oof for oof in rf_oof}
    aligned_oof = [oof_by_id[r.program_id] for r in dev_dataset.records]
    align_and_verify_program_ids(dev_dataset.records, aligned_oof)
    print("  Program ID alignment verified: 100% match across 159 records.")

    ml_containers = build_ml_prediction_containers(aligned_oof)

    # 5. Evaluate S1, S2, S3, S4
    print("\n[5/6] Evaluating four regimes (S1, S2, S3, S4)...")
    modes = [
        EvaluationMode.S1,
        EvaluationMode.S2,
        EvaluationMode.S3,
        EvaluationMode.S4,
    ]
    mode_predictions: dict[str, list[str]] = {}
    mode_summaries: dict[str, Any] = {}

    for m in modes:
        preds, _, summary = evaluate_mode(
            dev_dataset.records,
            raw_static_results,
            ml_containers,
            mode=m,
            threshold=threshold,
        )
        mode_predictions[m.value] = preds
        mode_summaries[m.value] = summary
        print(
            f"  {m.value:2s} -> Acc: {summary['accuracy']:.4f} | "
            f"Macro-F1: {summary['macro_f1']:.4f} | "
            f"Coverage: {summary['coverage']:.4f} | "
            f"Selective Acc: {summary['selective_accuracy']:.4f}"
        )

    # Bootstrap 95% CIs
    print(f"\n  Running family-cluster bootstrap ({bootstrap_replicates} reps, seed={seed})...")
    bootstrap_results = compute_family_cluster_bootstrap(
        dev_dataset.records,
        mode_predictions,
        seed=seed,
        n_replicates=bootstrap_replicates,
    )

    for m_name in mode_summaries:
        b_res = bootstrap_results[m_name]
        mode_summaries[m_name]["bootstrap_ci_95"] = {
            "accuracy": b_res["accuracy"].to_dict(),
            "macro_f1": b_res["macro_f1"].to_dict(),
        }

    # Sensitivity curve for S4
    print("  Computing S4 threshold sensitivity curve...")
    sensitivity_curve = compute_s4_sensitivity_curve(
        dev_dataset.records,
        raw_static_results,
        ml_containers,
    )

    # 6. Format and save results
    print(f"\n[6/6] Writing evaluation artifacts to {output_dir}...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info()
    fold_hash = compute_fold_assignment_hash(dev_dataset)

    evaluation_payload: dict[str, Any] = {
        "benchmark": {
            "dataset_root": str(dataset_dir),
            "fold_assignment_hash": fold_hash,
            "manifest_path": str(manifest_path),
            "manifest_sha256": dev_dataset.manifest_sha256,
            "num_families": n_families,
            "num_programs": n_programs,
        },
        "disclosures": {
            "random_forest_selection_bias": RF_SELECTION_DISCLOSURE,
            "static_engine": STATIC_DISCLOSURE,
        },
        "git_info": git_info,
        "modes": mode_summaries,
        "primary_threshold": threshold,
        "s4_sensitivity_curve": sensitivity_curve,
    }

    json_file = out_path / "day13_evaluation_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2, sort_keys=True)
    print(f"  Saved JSON results: {json_file}")

    # Generate Markdown summary table
    md_file = out_path / "day13_evaluation_summary.md"
    s4_buckets = mode_summaries["S4"]["buckets"]
    stat_b = s4_buckets["static_decisions"]
    fb_b = s4_buckets["ml_fallback_decisions"]
    abs_b = s4_buckets["abstained_decisions"]

    def _fmt_row(mode_key: str, desc: str, bold: bool = False) -> str:
        acc = mode_summaries[mode_key]["accuracy"]
        f1 = mode_summaries[mode_key]["macro_f1"]
        cov = mode_summaries[mode_key]["coverage"]
        sel_acc = mode_summaries[mode_key]["selective_accuracy"]
        sel_risk = mode_summaries[mode_key]["selective_risk"]
        b = mode_summaries[mode_key]["bootstrap_ci_95"]
        acc_ci = f"[{b['accuracy']['ci_lower']:.4f}, {b['accuracy']['ci_upper']:.4f}]"
        f1_ci = f"[{b['macro_f1']['ci_lower']:.4f}, {b['macro_f1']['ci_upper']:.4f}]"
        if bold:
            return (
                f"| **{mode_key}** | {desc} | **{acc:.4f}** {acc_ci} | "
                f"**{f1:.4f}** {f1_ci} | {cov:.4f} | **{sel_acc:.4f}** | **{sel_risk:.4f}** |"
            )
        return (
            f"| **{mode_key}** | {desc} | {acc:.4f} {acc_ci} | "
            f"{f1:.4f} {f1_ci} | {cov:.4f} | {sel_acc:.4f} | {sel_risk:.4f} |"
        )

    summary_lines = [
        "# ComplexityLens — Day 13 S1–S4 Hybrid Evaluation Summary",
        "",
        "Evaluated on 159 development programs (53 families) using 3-fold grouped CV.",
        "",
        "## Primary Regime Comparison",
        "",
        "| Regime | Description | Accuracy (95% CI) | Macro-F1 (95% CI) | Coverage | Selective Acc | Selective Risk |",  # noqa: E501
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
        _fmt_row("S1", "Static-Only"),
        _fmt_row("S2", "ML-Only (RF)"),
        _fmt_row("S3", "Naive Hybrid"),
        _fmt_row("S4", "Hybrid + Verification", bold=True),
        "",
        f"## S4 Decision Bucket Breakdown (Threshold = {threshold})",
        "",
        (
            f"- **Static Decisions**: {stat_b['count']} programs "
            f"({stat_b['correct']} correct, accuracy: {stat_b['accuracy']:.4f})"
        ),
        (
            f"- **ML Fallback Decisions**: {fb_b['count']} programs "
            f"({fb_b['correct']} correct, accuracy: {fb_b['accuracy']:.4f})"
        ),
        (
            f"- **Abstained Decisions**: {abs_b['count']} programs "
            f"({abs_b['gold_unknown_count']} true UNKNOWNs, "
            f"accuracy as abstention: {abs_b['accuracy_as_abstention']:.4f})"
        ),
        "",
        "## Mandatory Disclosures",
        "",
        f"- **Static Engine**: {STATIC_DISCLOSURE}",
        f"- **ML Model**: {RF_SELECTION_DISCLOSURE}",
        "",
    ]
    summary_md = "\n".join(summary_lines)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"  Saved Markdown summary: {md_file}")

    print("\nDay 13 evaluation completed successfully.")
    return evaluation_payload


if __name__ == "__main__":
    run_day13_evaluation()
