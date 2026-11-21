"""
rigorous.py — Rigorous Model Evaluation, Abstention, Calibration, and Bootstrap for ComplexityLens.

Implements Day 12 of the ComplexityLens evaluation roadmap:
- Strict evaluation on the development partition (159 programs, 53 families).
- Zero access to frozen test partition (guarded by TestPartitionAccessError).
- Evaluates:
    * Static-only engine (with required disclosure)
    * Baseline ML models: majority_class, logistic_regression, decision_tree
    * Candidate ML models: random_forest, xgboost (or hist_gradient_boosting fallback)
    * Hybrid system using pure decide_fusion at provisional threshold 0.55
- Complete performance metrics: Accuracy, Macro-F1, per-class Precision/Recall/F1/Support,
  and canonical 8x8 confusion matrices.
- Abstention metrics: Coverage, Selective Accuracy, UNKNOWN Precision/Recall,
  false non-UNKNOWN assignments on gold UNKNOWN, and raw counts.
- Risk-coverage analysis for scored ML models and static operating point.
- Raw calibration metrics: 5-bin ECE, bin counts, multiclass Brier score, multiclass log-loss.
  (Post-hoc calibration: "not performed (sample size)").
- Seeded family-cluster bootstrap (resampling families with replacement) for 95% CIs.
- Pairwise model disagreement counts across families.
- Hybrid threshold sensitivity analysis.
- Deterministic JSON serialization excluding timestamp/runtime from reproducible hashes.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from complexity_lens.complexity.engine import analyze_complexity
from complexity_lens.complexity.models import AnalysisResult, TaxonomyLabel
from complexity_lens.evaluation.candidates import SELECTION_BIASED_LABEL
from complexity_lens.evaluation.cv_runner import (
    CrossValidationRunner,
    OutOfFoldPrediction,
    compute_fold_assignment_hash,
    get_git_info,
)
from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    DevelopmentDataset,
    EvaluationRecord,
    load_development_dataset,
)
from complexity_lens.fusion import (
    DEFAULT_ML_THRESHOLD,
    FusionResult,
    MLPrediction,
    decide_fusion,
)
from complexity_lens.ir.builder import build_ir_from_source

STATIC_DISCLOSURE: str = "rules developed with dev data visible; not an independent estimate"
CALIBRATION_STATUS: str = "not performed (sample size)"
DEFAULT_RESULTS_DIR: str = "artifacts/results"
CANONICAL_CLASS_TO_INDEX: dict[str, int] = {
    cls_name: idx for idx, cls_name in enumerate(CANONICAL_CLASS_ORDER)
}


class RigorousEvaluationError(Exception):
    """Base exception for Day 12 rigorous evaluation pipeline errors."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PerClassRecord:
    """Metrics for a single class in canonical order."""

    precision: float | None
    recall: float | None
    f1: float | None
    support: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "f1": round(self.f1, 6) if self.f1 is not None else None,
            "precision": round(self.precision, 6) if self.precision is not None else None,
            "recall": round(self.recall, 6) if self.recall is not None else None,
            "support": self.support,
        }


@dataclasses.dataclass(frozen=True)
class AbstentionMetrics:
    """Abstention and selective prediction metrics."""

    coverage: float
    selective_accuracy: float
    selective_risk: float
    unknown_precision: float | None
    unknown_recall: float | None
    gold_unknown_falsely_accepted: int
    predicted_unknown_count: int
    raw_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": round(self.coverage, 6),
            "gold_unknown_falsely_accepted": self.gold_unknown_falsely_accepted,
            "predicted_unknown_count": self.predicted_unknown_count,
            "raw_counts": dict(self.raw_counts),
            "selective_accuracy": round(self.selective_accuracy, 6),
            "selective_risk": round(self.selective_risk, 6),
            "unknown_precision": round(self.unknown_precision, 6)
            if self.unknown_precision is not None
            else None,
            "unknown_recall": round(self.unknown_recall, 6)
            if self.unknown_recall is not None
            else None,
        }


@dataclasses.dataclass(frozen=True)
class RiskCoveragePoint:
    """A single operating point on the risk-coverage curve."""

    threshold: float
    coverage: float
    selective_risk: float
    selective_accuracy: float
    accepted_count: int
    rejected_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_count": self.accepted_count,
            "coverage": round(self.coverage, 6),
            "rejected_count": self.rejected_count,
            "selective_accuracy": round(self.selective_accuracy, 6),
            "selective_risk": round(self.selective_risk, 6),
            "threshold": round(self.threshold, 4),
        }


@dataclasses.dataclass(frozen=True)
class CalibrationMetrics:
    """Probability calibration diagnostics."""

    status: str
    ece: float | None
    bin_edges: list[float]
    bin_counts: list[int]
    bin_accuracies: list[float]
    bin_confidences: list[float]
    brier_score: float | None
    log_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_accuracies": [round(a, 6) for a in self.bin_accuracies],
            "bin_confidences": [round(c, 6) for c in self.bin_confidences],
            "bin_counts": list(self.bin_counts),
            "bin_edges": [round(e, 4) for e in self.bin_edges],
            "brier_score": round(self.brier_score, 6) if self.brier_score is not None else None,
            "ece": round(self.ece, 6) if self.ece is not None else None,
            "log_loss": round(self.log_loss, 6) if self.log_loss is not None else None,
            "status": self.status,
        }


@dataclasses.dataclass(frozen=True)
class BootstrapInterval:
    """Bootstrap confidence interval."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_replicates: int
    seed: int
    procedure: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "confidence_level": self.confidence_level,
            "n_replicates": self.n_replicates,
            "point_estimate": round(self.point_estimate, 6),
            "procedure": self.procedure,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_order: Sequence[str] = CANONICAL_CLASS_ORDER,
) -> list[list[int]]:
    """Compute 8x8 confusion matrix where rows are true classes and columns are predictions."""
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    n = len(class_order)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            matrix[class_to_idx[t]][class_to_idx[p]] += 1
    return matrix


def compute_per_class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_order: Sequence[str] = CANONICAL_CLASS_ORDER,
) -> dict[str, PerClassRecord]:
    """Compute per-class precision, recall, f1, and support following documented zero-division."""
    results: dict[str, PerClassRecord] = {}
    for c in class_order:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)

        if support == 0:
            precision: float | None = None
            recall: float | None = None
            f1: float | None = None
        elif tp + fp == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            denom = precision + recall
            f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0

        results[c] = PerClassRecord(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )
    return results


def compute_macro_f1(per_class: dict[str, PerClassRecord]) -> float:
    """Compute macro-F1 averaged strictly over classes with support > 0."""
    valid_f1s = [rec.f1 for rec in per_class.values() if rec.support > 0 and rec.f1 is not None]
    if not valid_f1s:
        return 0.0
    return float(np.mean(valid_f1s))


def compute_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute simple accuracy over all examples."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def compute_abstention_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> AbstentionMetrics:
    """Compute abstention, coverage, and selective accuracy metrics.

    Definitions:
    - Coverage: fraction of predictions that are not UNKNOWN.
    - Selective accuracy: accuracy evaluated strictly among non-UNKNOWN predictions.
    - Selective risk: 1.0 - selective accuracy.
    - UNKNOWN precision/recall: computed with UNKNOWN as the positive class.
    - Gold UNKNOWN false assignments: count of records where true is UNKNOWN but
      a non-UNKNOWN label was predicted.
    """
    total = len(y_true)
    if total == 0:
        return AbstentionMetrics(
            coverage=0.0,
            selective_accuracy=0.0,
            selective_risk=0.0,
            unknown_precision=None,
            unknown_recall=None,
            gold_unknown_falsely_accepted=0,
            predicted_unknown_count=0,
            raw_counts={},
        )

    accepted_preds = [p for p in y_pred if p != "UNKNOWN"]
    accepted_count = len(accepted_preds)
    rejected_count = total - accepted_count
    coverage = accepted_count / total

    correct_accepted = sum(1 for t, p in zip(y_true, y_pred) if p != "UNKNOWN" and t == p)
    selective_accuracy = (correct_accepted / accepted_count) if accepted_count > 0 else 0.0
    selective_risk = (1.0 - selective_accuracy) if accepted_count > 0 else 0.0

    # UNKNOWN as positive class
    tp_unk = sum(1 for t, p in zip(y_true, y_pred) if t == "UNKNOWN" and p == "UNKNOWN")
    fp_unk = sum(1 for t, p in zip(y_true, y_pred) if t != "UNKNOWN" and p == "UNKNOWN")
    fn_unk = sum(1 for t, p in zip(y_true, y_pred) if t == "UNKNOWN" and p != "UNKNOWN")
    tn_unk = sum(1 for t, p in zip(y_true, y_pred) if t != "UNKNOWN" and p != "UNKNOWN")

    gold_unknown_total = sum(1 for t in y_true if t == "UNKNOWN")
    gold_unknown_falsely_accepted = fn_unk

    unk_precision = (tp_unk / (tp_unk + fp_unk)) if (tp_unk + fp_unk) > 0 else 0.0
    unk_recall = (tp_unk / (tp_unk + fn_unk)) if (tp_unk + fn_unk) > 0 else 0.0

    raw_counts = {
        "correct_accepted": correct_accepted,
        "false_negatives_unknown": fn_unk,
        "false_positives_unknown": fp_unk,
        "gold_non_unknown_count": total - gold_unknown_total,
        "gold_unknown_count": gold_unknown_total,
        "gold_unknown_falsely_accepted_non_unknown": gold_unknown_falsely_accepted,
        "incorrect_accepted": accepted_count - correct_accepted,
        "n_accepted_predictions": accepted_count,
        "n_rejected_predictions": rejected_count,
        "total_predictions": total,
        "true_negatives_unknown": tn_unk,
        "true_positives_unknown": tp_unk,
    }

    return AbstentionMetrics(
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        selective_risk=selective_risk,
        unknown_precision=unk_precision,
        unknown_recall=unk_recall,
        gold_unknown_falsely_accepted=gold_unknown_falsely_accepted,
        predicted_unknown_count=rejected_count,
        raw_counts=raw_counts,
    )


def compute_risk_coverage_curve(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> list[RiskCoveragePoint]:
    """Generate risk-coverage curve by sweeping the confidence acceptance threshold."""
    total = len(y_true)
    if total == 0:
        return []

    if thresholds is None:
        fixed_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
        unique_confs = sorted(set(confidences))
        all_th = sorted(set(fixed_grid + [float(round(c, 4)) for c in unique_confs]))
    else:
        all_th = sorted(thresholds)

    curve: list[RiskCoveragePoint] = []
    for th in all_th:
        accepted_indices = [i for i, c in enumerate(confidences) if c >= th]
        acc_count = len(accepted_indices)
        rej_count = total - acc_count
        cov = acc_count / total

        if acc_count > 0:
            correct = sum(1 for i in accepted_indices if y_pred[i] == y_true[i])
            sel_acc = correct / acc_count
            sel_risk = 1.0 - sel_acc
        else:
            sel_acc = 0.0
            sel_risk = 0.0

        curve.append(
            RiskCoveragePoint(
                threshold=th,
                coverage=cov,
                selective_risk=sel_risk,
                selective_accuracy=sel_acc,
                accepted_count=acc_count,
                rejected_count=rej_count,
            )
        )
    return curve


def compute_calibration_metrics(
    y_true: Sequence[str],
    probabilities: Sequence[Sequence[float]],
    num_bins: int = 5,
) -> CalibrationMetrics:
    """Calculate raw probability calibration metrics (ECE, bin counts, Brier, log-loss).

    Guarantees:
    - 5 equal-width bins: [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0].
    - Reports counts in every bin, including empty bins.
    - Multiclass Brier score over all 8 canonical classes.
    - Safe log-loss with 1e-15 epsilon clamping for zero probabilities.
    - Post-hoc calibration status explicitly set to 'not performed (sample size)'.
    """
    total = len(y_true)
    if total == 0:
        return CalibrationMetrics(
            status=CALIBRATION_STATUS,
            ece=None,
            bin_edges=[i / num_bins for i in range(num_bins + 1)],
            bin_counts=[0] * num_bins,
            bin_accuracies=[0.0] * num_bins,
            bin_confidences=[0.0] * num_bins,
            brier_score=None,
            log_loss=None,
        )

    probs_arr = np.array(probabilities, dtype=float)
    confidences = np.max(probs_arr, axis=1)
    pred_indices = np.argmax(probs_arr, axis=1)
    pred_labels = [CANONICAL_CLASS_ORDER[idx] for idx in pred_indices]

    # Equal width bin edges: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    bin_edges = [float(i / num_bins) for i in range(num_bins + 1)]
    bin_counts: list[int] = []
    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []

    ece = 0.0
    for b in range(num_bins):
        low = bin_edges[b]
        high = bin_edges[b + 1]
        if b < num_bins - 1:
            mask = (confidences >= low) & (confidences < high)
        else:
            mask = (confidences >= low) & (confidences <= high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            indices = np.where(mask)[0]
            acc = sum(1 for i in indices if pred_labels[i] == y_true[i]) / count
            conf_mean = float(np.mean(confidences[mask]))
            bin_accuracies.append(acc)
            bin_confidences.append(conf_mean)
            ece += (count / total) * abs(acc - conf_mean)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    # Multiclass Brier score: 1/N * sum_i sum_k (p_ik - y_ik)^2
    brier_sum = 0.0
    for i, t in enumerate(y_true):
        t_idx = CANONICAL_CLASS_TO_INDEX.get(t, -1)
        for k in range(len(CANONICAL_CLASS_ORDER)):
            y_ik = 1.0 if k == t_idx else 0.0
            p_ik = probs_arr[i, k]
            brier_sum += (p_ik - y_ik) ** 2
    brier_score = brier_sum / total

    # Multiclass log-loss: -1/N * sum_i log(max(p_{i, true}, 1e-15))
    eps = 1e-15
    log_loss_sum = 0.0
    for i, t in enumerate(y_true):
        t_idx = CANONICAL_CLASS_TO_INDEX.get(t, -1)
        if t_idx >= 0:
            p_true = max(float(probs_arr[i, t_idx]), eps)
        else:
            p_true = eps
        log_loss_sum += -math.log(p_true)
    log_loss = log_loss_sum / total

    return CalibrationMetrics(
        status=CALIBRATION_STATUS,
        ece=ece,
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        brier_score=brier_score,
        log_loss=log_loss,
    )


def compute_family_cluster_bootstrap(
    records: Sequence[EvaluationRecord],
    predictions_by_model: dict[str, Sequence[str]],
    seed: int = 42,
    n_replicates: int = 1000,
) -> dict[str, dict[str, BootstrapInterval]]:
    """Compute 95% family-cluster bootstrap intervals for Accuracy and Macro-F1.

    Methodology:
    - Clusters: Resamples algorithm families (family_id), NOT individual programs.
    - Paired comparisons: Identical bootstrap family draws are applied across all models.
    - Confidence interval: 2.5th and 97.5th percentiles of replicate distributions.
    """
    unique_fams = sorted(set(r.family_id for r in records))
    n_fams = len(unique_fams)
    rng = np.random.RandomState(seed)

    # Group record indices by family_id
    fam_to_indices: dict[str, list[int]] = {f: [] for f in unique_fams}
    for idx, r in enumerate(records):
        fam_to_indices[r.family_id].append(idx)

    # Compute bootstrap replicates
    model_replicates: dict[str, dict[str, list[float]]] = {
        m: {"accuracy": [], "macro_f1": []} for m in predictions_by_model
    }

    y_true_all = [r.label for r in records]

    for _ in range(n_replicates):
        sampled_fams = rng.choice(unique_fams, size=n_fams, replace=True)
        sample_indices: list[int] = []
        for fam in sampled_fams:
            sample_indices.extend(fam_to_indices[fam])

        sample_y_true = [y_true_all[i] for i in sample_indices]

        for m_id, preds in predictions_by_model.items():
            sample_preds = [preds[i] for i in sample_indices]
            acc = compute_accuracy(sample_y_true, sample_preds)
            per_cls = compute_per_class_metrics(sample_y_true, sample_preds)
            f1 = compute_macro_f1(per_cls)
            model_replicates[m_id]["accuracy"].append(acc)
            model_replicates[m_id]["macro_f1"].append(f1)

    intervals: dict[str, dict[str, BootstrapInterval]] = {}
    procedure_desc = (
        f"Seeded family-cluster bootstrap (resampling {n_fams} families with replacement, "
        f"{n_replicates} replicates, seed={seed})"
    )

    for m_id, preds in predictions_by_model.items():
        base_acc = compute_accuracy(y_true_all, preds)
        base_per_cls = compute_per_class_metrics(y_true_all, preds)
        base_f1 = compute_macro_f1(base_per_cls)

        acc_dist = model_replicates[m_id]["accuracy"]
        f1_dist = model_replicates[m_id]["macro_f1"]

        intervals[m_id] = {
            "accuracy": BootstrapInterval(
                point_estimate=base_acc,
                ci_lower=float(np.percentile(acc_dist, 2.5)),
                ci_upper=float(np.percentile(acc_dist, 97.5)),
                confidence_level=0.95,
                n_replicates=n_replicates,
                seed=seed,
                procedure=procedure_desc,
            ),
            "macro_f1": BootstrapInterval(
                point_estimate=base_f1,
                ci_lower=float(np.percentile(f1_dist, 2.5)),
                ci_upper=float(np.percentile(f1_dist, 97.5)),
                confidence_level=0.95,
                n_replicates=n_replicates,
                seed=seed,
                procedure=procedure_desc,
            ),
        }

    return intervals


def compute_pairwise_model_disagreements(
    records: Sequence[EvaluationRecord],
    predictions_by_model: dict[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    """Compute number of benchmark families on which each pair of models differs.

    Definition: A family is counted as differing between models A and B if at least
    one program in that family receives different predictions from A and B.
    """
    unique_fams = sorted(set(r.family_id for r in records))
    total_fams = len(unique_fams)
    fam_to_indices: dict[str, list[int]] = {f: [] for f in unique_fams}
    for idx, r in enumerate(records):
        fam_to_indices[r.family_id].append(idx)

    model_names = sorted(predictions_by_model.keys())
    disagreements: dict[str, dict[str, Any]] = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1 = model_names[i]
            m2 = model_names[j]
            pair_key = f"{m1}_vs_{m2}"

            diff_fams = 0
            for fam, indices in fam_to_indices.items():
                if any(
                    predictions_by_model[m1][idx] != predictions_by_model[m2][idx]
                    for idx in indices
                ):
                    diff_fams += 1

            disagreements[pair_key] = {
                "differing_families": diff_fams,
                "disagreement_rate": round(diff_fams / total_fams, 6) if total_fams > 0 else 0.0,
                "model_1": m1,
                "model_2": m2,
                "total_families": total_fams,
            }

    return disagreements


# ---------------------------------------------------------------------------
# Evaluator Implementations
# ---------------------------------------------------------------------------


def evaluate_static_only_engine(
    dataset: DevelopmentDataset,
) -> tuple[list[str], list[AnalysisResult | None]]:
    """Evaluate static complexity engine on all development records.

    Uses the caller function if ambiguous function selection occurs.
    """
    preds: list[str] = []
    raw_results: list[AnalysisResult | None] = []
    total = len(dataset.records)

    for idx, r in enumerate(dataset.records):
        if (idx + 1) % 40 == 0 or idx == total - 1:
            print(f"  [Static Engine] Evaluated {idx + 1}/{total} programs...", flush=True)
        try:
            ir = build_ir_from_source(r.source, filename=f"{r.program_id}.cpp")
            fn: str | None = None
            if len(ir.functions) > 1 and not any(f.name == "main" for f in ir.functions):
                fn = ir.functions[-1].name
            res = analyze_complexity(ir, function_name=fn)
            raw_results.append(res)
            preds.append(res.taxonomy_label)
        except Exception:
            raw_results.append(None)
            preds.append(TaxonomyLabel.UNKNOWN.value)

    return preds, raw_results


def evaluate_hybrid_fusion(
    static_results: Sequence[AnalysisResult | None],
    ml_oof_preds: Sequence[OutOfFoldPrediction],
    threshold: float = DEFAULT_ML_THRESHOLD,
) -> tuple[list[str], list[FusionResult]]:
    """Run pure deterministic fusion gate combining static results and ML predictions."""
    if len(static_results) != len(ml_oof_preds):
        raise RigorousEvaluationError(
            "Mismatched counts between static results and ML OOF predictions."
        )

    preds: list[str] = []
    fusion_results: list[FusionResult] = []

    for s_res, ml_pred in zip(static_results, ml_oof_preds):
        # Create MLPrediction container
        probs_dict = {
            cls_name: float(ml_pred.probabilities[idx])
            for idx, cls_name in enumerate(CANONICAL_CLASS_ORDER)
        }
        pred_idx = CANONICAL_CLASS_TO_INDEX.get(ml_pred.predicted_label, -1)
        top_prob = (
            float(ml_pred.probabilities[pred_idx])
            if pred_idx >= 0
            else float(max(ml_pred.probabilities))
        )

        ml_container = MLPrediction(
            predicted_label=ml_pred.predicted_label,
            probability=top_prob,
            probabilities=probs_dict,
            model_id="ml_oof",
        )

        fusion_res = decide_fusion(s_res, ml_container, ml_threshold=threshold)
        fusion_results.append(fusion_res)
        preds.append(fusion_res.primary_label)

    return preds, fusion_results


def compute_hybrid_sensitivity_curve(
    y_true: Sequence[str],
    static_results: Sequence[AnalysisResult | None],
    ml_oof_preds: Sequence[OutOfFoldPrediction],
    threshold_grid: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Compute sensitivity curve across a grid of ML acceptance thresholds."""
    if threshold_grid is None:
        threshold_grid = [round(float(t), 2) for t in np.arange(0.0, 1.05, 0.05)]

    curve: list[dict[str, Any]] = []
    for th in threshold_grid:
        preds, _ = evaluate_hybrid_fusion(static_results, ml_oof_preds, threshold=th)
        acc = compute_accuracy(y_true, preds)
        per_cls = compute_per_class_metrics(y_true, preds)
        macro_f1 = compute_macro_f1(per_cls)
        abstention = compute_abstention_metrics(y_true, preds)

        curve.append({
            "accuracy": round(acc, 6),
            "coverage": round(abstention.coverage, 6),
            "gold_unknown_falsely_accepted": abstention.gold_unknown_falsely_accepted,
            "macro_f1": round(macro_f1, 6),
            "selective_accuracy": round(abstention.selective_accuracy, 6),
            "threshold": round(th, 2),
            "unknown_precision": round(abstention.unknown_precision, 6)
            if abstention.unknown_precision is not None
            else None,
            "unknown_recall": round(abstention.unknown_recall, 6)
            if abstention.unknown_recall is not None
            else None,
        })
    return curve


# ---------------------------------------------------------------------------
# Master Day 12 Orchestration
# ---------------------------------------------------------------------------


def run_rigorous_evaluation(
    manifest_path: str | Path = "data/splits/split_manifest_v1.0.frozen.json",
    dataset_dir: str | Path = "datasets",
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    """Execute complete Day 12 rigorous evaluation pipeline.

    Evaluates:
    1. Static-only engine (with required disclosure)
    2. ML candidate models:
       - majority_class
       - logistic_regression
       - decision_tree
       - random_forest (reproduces Day 11 winner)
       - xgboost (reproduces Day 11 winner)
    3. Hybrid system at provisional threshold 0.55 and sensitivity curve
    4. Bootstrap 95% intervals and pairwise family disagreement counts
    5. Saves deterministic JSON output to output_dir
    """
    import time
    start_time = time.time()

    # 1. Load development dataset strictly through guarded loader
    print("[1/5] Loading development dataset (159 records across 53 families)...", flush=True)
    dev_dataset = load_development_dataset(manifest_path, dataset_dir)
    records = dev_dataset.records
    y_true = [r.label for r in records]
    manifest_sha256 = dev_dataset.manifest_sha256
    fold_hash = compute_fold_assignment_hash(dev_dataset)
    git_commit, git_dirty = get_git_info()

    # 2. Evaluate static-only engine
    print("[2/5] Evaluating static-only engine on development partition...", flush=True)
    static_preds, raw_static_results = evaluate_static_only_engine(dev_dataset)
    static_acc = compute_accuracy(y_true, static_preds)
    static_per_cls = compute_per_class_metrics(y_true, static_preds)
    static_macro_f1 = compute_macro_f1(static_per_cls)
    static_cm = compute_confusion_matrix(y_true, static_preds)
    static_abstention = compute_abstention_metrics(y_true, static_preds)

    # Static operating point for risk-coverage
    static_risk_coverage = [
        RiskCoveragePoint(
            threshold=1.0,
            coverage=static_abstention.coverage,
            selective_risk=static_abstention.selective_risk,
            selective_accuracy=static_abstention.selective_accuracy,
            accepted_count=static_abstention.raw_counts["n_accepted_predictions"],
            rejected_count=static_abstention.raw_counts["n_rejected_predictions"],
        ).to_dict()
    ]

    # 3. Evaluate ML candidate models via 3-fold cross validation
    print("[3/5] Evaluating ML candidate models via 3-fold cross validation...", flush=True)
    runner = CrossValidationRunner(
        manifest_path=manifest_path,
        dataset_root=dataset_dir,
        dataset=dev_dataset,
    )

    ml_model_ids = [
        "majority_class",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    ]

    ml_eval_results: dict[str, Any] = {}
    model_predictions: dict[str, list[str]] = {"static_only": static_preds}
    ml_oof_preds_map: dict[str, list[OutOfFoldPrediction]] = {}

    for m_id in ml_model_ids:
        print(f"  [ML CV] Evaluating {m_id} in-memory across 3 folds...", flush=True)
        m_res = runner.evaluate_model(m_id)
        oof_preds = m_res.oof_predictions
        ml_oof_preds_map[m_id] = oof_preds
        preds = [p.predicted_label for p in oof_preds]
        model_predictions[m_id] = preds

        probs = [p.probabilities for p in oof_preds]
        confs = [float(max(p)) for p in probs]

        acc = compute_accuracy(y_true, preds)
        per_cls = compute_per_class_metrics(y_true, preds)
        macro_f1 = compute_macro_f1(per_cls)
        cm = compute_confusion_matrix(y_true, preds)
        abstention = compute_abstention_metrics(y_true, preds)
        rc_curve = compute_risk_coverage_curve(y_true, preds, confs)
        cal = compute_calibration_metrics(y_true, probs)

        bias_label = (
            SELECTION_BIASED_LABEL
            if m_id in ("random_forest", "xgboost")
            else "unbiased CV estimate (pre-fixed hyperparameters)"
        )

        ml_eval_results[m_id] = {
            "abstention": abstention.to_dict(),
            "accuracy": round(acc, 6),
            "bias_label": bias_label,
            "calibration": cal.to_dict(),
            "confusion_matrix": cm,
            "fixed_hyperparameters": m_res.fixed_hyperparameters,
            "macro_f1": round(macro_f1, 6),
            "model_id": m_id,
            "per_class": {k: v.to_dict() for k, v in per_cls.items()},
            "risk_coverage_curve": [p.to_dict() for p in rc_curve],
        }

    # 4. Evaluate Hybrid System (Static + RF ML OOF at provisional threshold 0.55)
    print("[4/5] Evaluating hybrid fusion (Th=0.55) and sensitivity curve...", flush=True)
    rf_oof = ml_oof_preds_map["random_forest"]
    hybrid_preds, _ = evaluate_hybrid_fusion(
        raw_static_results, rf_oof, threshold=DEFAULT_ML_THRESHOLD
    )
    model_predictions["hybrid_provisional_0.55"] = hybrid_preds

    hybrid_acc = compute_accuracy(y_true, hybrid_preds)
    hybrid_per_cls = compute_per_class_metrics(y_true, hybrid_preds)
    hybrid_macro_f1 = compute_macro_f1(hybrid_per_cls)
    hybrid_cm = compute_confusion_matrix(y_true, hybrid_preds)
    hybrid_abstention = compute_abstention_metrics(y_true, hybrid_preds)
    hybrid_sensitivity = compute_hybrid_sensitivity_curve(y_true, raw_static_results, rf_oof)

    hybrid_eval_result = {
        "abstention": hybrid_abstention.to_dict(),
        "accuracy": round(hybrid_acc, 6),
        "bias_label": "provisional threshold 0.55",
        "confusion_matrix": hybrid_cm,
        "macro_f1": round(hybrid_macro_f1, 6),
        "model_id": "hybrid_fusion",
        "per_class": {k: v.to_dict() for k, v in hybrid_per_cls.items()},
        "provisional_threshold": DEFAULT_ML_THRESHOLD,
        "sensitivity_curve": hybrid_sensitivity,
    }

    # 5. Bootstrap 95% Confidence Intervals (seeded family cluster)
    print("[5/5] Running family-cluster bootstrap (1000 replicates, seed=42)...", flush=True)
    bootstrap_intervals = compute_family_cluster_bootstrap(
        records,
        model_predictions,
        seed=42,
        n_replicates=1000,
    )

    # 6. Pairwise Model Disagreement across Families
    disagreements = compute_pairwise_model_disagreements(records, model_predictions)

    # 7. Assemble Structured Output Document
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    results_payload: dict[str, Any] = {
        "calibration_status": CALIBRATION_STATUS,
        "canonical_class_order": list(CANONICAL_CLASS_ORDER),
        "dataset_manifest_sha256": manifest_sha256,
        "disagreements_pairwise": disagreements,
        "family_cluster_bootstrap": {
            m: {metric: inter.to_dict() for metric, inter in m_dict.items()}
            for m, m_dict in bootstrap_intervals.items()
        },
        "fold_assignment_hash": fold_hash,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hybrid_system": hybrid_eval_result,
        "ml_candidates": ml_eval_results,
        "protocol": {
            "cv_scheme": "3-fold grouped CV by family_id",
            "development_families": 53,
            "development_programs": 159,
            "provisional_hybrid_threshold": DEFAULT_ML_THRESHOLD,
            "test_partition_isolation": "guarded by TestPartitionAccessError (zero access)",
        },
        "reproduction_verification": {
            "day11_rf_target_accuracy": 0.534591,
            "day11_rf_target_macro_f1": 0.549887,
            "day11_xgb_target_accuracy": 0.528302,
            "day11_xgb_target_macro_f1": 0.535573,
            "reproduced_rf_accuracy": ml_eval_results["random_forest"]["accuracy"],
            "reproduced_rf_macro_f1": ml_eval_results["random_forest"]["macro_f1"],
            "reproduced_xgb_accuracy": ml_eval_results["xgboost"]["accuracy"],
            "reproduced_xgb_macro_f1": ml_eval_results["xgboost"]["macro_f1"],
        },
        "schema_version": "1.0",
        "small_gaps_disclaimer": "Small gaps are not meaningful.",
        "static_only_engine": {
            "abstention": static_abstention.to_dict(),
            "accuracy": round(static_acc, 6),
            "bias_label": STATIC_DISCLOSURE,
            "calibration": {
                "brier_score": None,
                "ece": None,
                "log_loss": None,
                "note": "Static engine does not emit continuous probabilities; calibration N/A",
                "status": "N/A",
            },
            "confusion_matrix": static_cm,
            "disclosure": STATIC_DISCLOSURE,
            "macro_f1": round(static_macro_f1, 6),
            "model_id": "static_only_engine",
            "per_class": {k: v.to_dict() for k, v in static_per_cls.items()},
            "risk_coverage_curve": static_risk_coverage,
        },
    }

    # Separate metadata field for timestamp and runtime so reproducibility check can ignore them
    runtime_sec = time.time() - start_time
    metadata = {
        "execution_runtime_seconds": round(runtime_sec, 2),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results_payload["metadata"] = metadata

    # 8. Deterministic Write to disk
    json_path = out_dir_path / "day12_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, sort_keys=True)

    # Also mirror to artifacts/evaluation for convention compatibility
    mirror_dir = Path("artifacts/evaluation")
    mirror_dir.mkdir(parents=True, exist_ok=True)
    with open(mirror_dir / "day12_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, sort_keys=True)

    return results_payload
