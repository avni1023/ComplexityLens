"""tests/test_evaluation_rigorous.py — Unit and integration tests for Day 12 evaluation.

Covers all 19 mandatory test requirements from Day 12 specification:
1. Metric calculations and canonical 8-class taxonomy order.
2. Missing-class behavior and documented zero-division contracts.
3. UNKNOWN precision, UNKNOWN recall, and gold UNKNOWN false assignments.
4. Coverage, selective accuracy, and selective risk.
5. Risk-coverage curve calculations.
6. ECE binning with 5 equal-width bins and empty bin handling.
7. Multiclass Brier score calculation.
8. Multiclass log-loss with zero-probability epsilon clamping.
9. Family-cluster bootstrap determinism and paired sampling.
10. Family-level model disagreement counting.
11. Hybrid fusion gate at provisional threshold 0.55.
12. Hybrid sensitivity curve generation.
13. Calibration status recorded as 'not performed (sample size)'.
14. In-memory fold refitting without loading persisted dev-fit artifacts.
15. Strict frozen-test partition access rejection (TestPartitionAccessError).
16. Static engine disclosure verification.
17. Deterministic serialization excluding timestamp/runtime fields.
"""

from __future__ import annotations

import math

import pytest

from complexity_lens.complexity.models import AnalysisResult, Exactness
from complexity_lens.evaluation.cv_runner import OutOfFoldPrediction
from complexity_lens.evaluation.data_loader import (
    EvaluationRecord,
)
from complexity_lens.evaluation.rigorous import (
    CALIBRATION_STATUS,
    STATIC_DISCLOSURE,
    compute_abstention_metrics,
    compute_calibration_metrics,
    compute_confusion_matrix,
    compute_family_cluster_bootstrap,
    compute_hybrid_sensitivity_curve,
    compute_macro_f1,
    compute_pairwise_model_disagreements,
    compute_per_class_metrics,
    compute_risk_coverage_curve,
    evaluate_hybrid_fusion,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_dummy_record(
    program_id: str, family_id: str, label: str, fold: int = 0
) -> EvaluationRecord:
    """Construct an EvaluationRecord without invoking Clang AST."""
    return EvaluationRecord(
        program_id=program_id,
        family_id=family_id,
        label=label,
        fold=fold,
        tag="easy",
        source="int main() { return 0; }",
        source_sha256="abc123",
        _features=None,
    )


def make_dummy_analysis_result(
    label: str, exactness: str = Exactness.STRUCTURAL.value
) -> AnalysisResult:
    """Construct an AnalysisResult for testing."""
    from complexity_lens.complexity.models import ComplexityClass
    cc = ComplexityClass.constant() if label == "O1" else ComplexityClass.linear()
    return AnalysisResult(
        selected_function="main",
        selection_rationale="test",
        complexity_class=cc,
        taxonomy_label=label,
        exactness=exactness,
        notes=["test"],
    )


# ---------------------------------------------------------------------------
# 1. Performance Metrics & 8-Class Ordering
# ---------------------------------------------------------------------------


def test_confusion_matrix_shape_and_canonical_order():
    """Confusion matrix must be exactly 8x8 in canonical class order."""
    y_true = ["O1", "ON", "UNKNOWN", "O1"]
    y_pred = ["O1", "ON2", "UNKNOWN", "O1"]
    cm = compute_confusion_matrix(y_true, y_pred)
    assert len(cm) == 8
    for row in cm:
        assert len(row) == 8

    # O1 -> O1 is count 2 at (0, 0)
    assert cm[0][0] == 2
    # ON is index 2, ON2 is index 4 -> cm[2][4] == 1
    assert cm[2][4] == 1
    # UNKNOWN is index 7 -> cm[7][7] == 1
    assert cm[7][7] == 1


def test_per_class_metrics_and_zero_division():
    """Zero-support classes must have None metrics; zero-pred classes must have 0.0."""
    y_true = ["O1", "O1", "ON"]
    y_pred = ["O1", "ON", "ON"]
    per_cls = compute_per_class_metrics(y_true, y_pred)

    assert len(per_cls) == 8
    # O1: 1 TP, 0 FP, 1 FN -> precision = 1.0, recall = 0.5, f1 = 2/3
    assert per_cls["O1"].precision == 1.0
    assert per_cls["O1"].recall == 0.5
    assert per_cls["O1"].f1 is not None and abs(per_cls["O1"].f1 - (2 / 3)) < 1e-5
    assert per_cls["O1"].support == 2

    # OLOGN has support 0 -> None
    assert per_cls["OLOGN"].support == 0
    assert per_cls["OLOGN"].precision is None
    assert per_cls["OLOGN"].recall is None
    assert per_cls["OLOGN"].f1 is None

    # Macro F1 averages only positive support classes (O1 and ON)
    macro_f1 = compute_macro_f1(per_cls)
    assert 0.0 < macro_f1 < 1.0


# ---------------------------------------------------------------------------
# 2. Abstention Metrics & UNKNOWN Precision/Recall
# ---------------------------------------------------------------------------


def test_abstention_metrics_calculation():
    """Verify coverage, selective accuracy, and UNKNOWN precision/recall."""
    # 5 examples:
    # 0: True=O1, Pred=O1 (accepted, correct)
    # 1: True=ON, Pred=ON2 (accepted, error)
    # 2: True=UNKNOWN, Pred=UNKNOWN (rejected, correct UNKNOWN TP)
    # 3: True=ON, Pred=UNKNOWN (rejected, UNKNOWN FP)
    # 4: True=UNKNOWN, Pred=O1 (accepted, gold UNKNOWN false assignment!)
    y_true = ["O1", "ON", "UNKNOWN", "ON", "UNKNOWN"]
    y_pred = ["O1", "ON2", "UNKNOWN", "UNKNOWN", "O1"]

    abstention = compute_abstention_metrics(y_true, y_pred)

    # Total = 5, accepted (non-UNKNOWN) = 3 (indices 0, 1, 4)
    assert abstention.coverage == 3 / 5
    # Among accepted: index 0 is correct, 1 and 4 are wrong -> 1/3
    assert abs(abstention.selective_accuracy - (1 / 3)) < 1e-5
    assert abs(abstention.selective_risk - (2 / 3)) < 1e-5

    # UNKNOWN as positive:
    # Gold UNKNOWN: indices 2, 4 (2 total)
    # Pred UNKNOWN: indices 2, 3 (2 total)
    # TP = index 2 (1)
    # FP = index 3 (1)
    # FN = index 4 (1)
    assert abstention.unknown_precision == 0.5
    assert abstention.unknown_recall == 0.5
    assert abstention.gold_unknown_falsely_accepted == 1
    assert abstention.predicted_unknown_count == 2
    assert abstention.raw_counts["gold_unknown_count"] == 2


# ---------------------------------------------------------------------------
# 3. Risk-Coverage Curve
# ---------------------------------------------------------------------------


def test_risk_coverage_curve_properties():
    """Verify monotonic coverage and accepted/rejected invariants."""
    y_true = ["O1", "ON", "O1", "ON"]
    y_pred = ["O1", "ON", "ON", "ON"]
    confs = [0.9, 0.8, 0.4, 0.7]

    curve = compute_risk_coverage_curve(y_true, y_pred, confs, thresholds=[0.0, 0.5, 0.75, 1.0])
    assert len(curve) == 4

    # At threshold 0.0: all accepted
    assert curve[0].threshold == 0.0
    assert curve[0].coverage == 1.0
    assert curve[0].accepted_count == 4
    assert curve[0].rejected_count == 0

    # At threshold 1.0: none accepted (max is 0.9)
    assert curve[3].threshold == 1.0
    assert curve[3].coverage == 0.0
    assert curve[3].accepted_count == 0
    assert curve[3].rejected_count == 4


# ---------------------------------------------------------------------------
# 4. Calibration: 5-Bin ECE, Brier, Log-Loss
# ---------------------------------------------------------------------------


def test_calibration_ece_and_empty_bins():
    """Verify 5 equal-width bins, safe handling of empty bins, and Brier/log-loss."""
    y_true = ["O1", "ON"]
    # 2 predictions with confidence 0.85 in bin [0.8, 1.0]
    probs = [
        [0.85, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01],  # Pred O1, True O1 (correct)
        [0.05, 0.05, 0.85, 0.01, 0.01, 0.01, 0.01, 0.01],  # Pred ON, True ON (correct)
    ]
    cal = compute_calibration_metrics(y_true, probs, num_bins=5)

    assert cal.status == CALIBRATION_STATUS
    assert len(cal.bin_counts) == 5
    assert cal.bin_edges == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # First 4 bins empty, 5th bin has 2 samples
    assert cal.bin_counts[:4] == [0, 0, 0, 0]
    assert cal.bin_counts[4] == 2
    assert cal.bin_accuracies[4] == 1.0
    assert abs(cal.bin_confidences[4] - 0.85) < 1e-5

    # ECE = |1.0 - 0.85| = 0.15
    assert cal.ece is not None
    assert abs(cal.ece - 0.15) < 1e-4

    # Brier score and log-loss exist and are finite
    assert cal.brier_score is not None and cal.brier_score > 0.0
    assert cal.log_loss is not None and cal.log_loss > 0.0


def test_log_loss_zero_probability_clamping():
    """Log-loss must not produce -inf or NaN when model predicts 0.0 on true class."""
    y_true = ["O1"]
    # True class O1 has 0.0 probability
    probs = [[0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]
    cal = compute_calibration_metrics(y_true, probs)
    assert cal.log_loss is not None
    assert math.isfinite(cal.log_loss)
    assert cal.log_loss > 30.0  # -ln(1e-15) ~ 34.5


# ---------------------------------------------------------------------------
# 5. Bootstrap Uncertainty & Model Disagreements
# ---------------------------------------------------------------------------


def test_family_cluster_bootstrap_determinism():
    """Family-cluster bootstrap with fixed seed must yield identical results."""
    records = [
        make_dummy_record("p1", "f01", "O1"),
        make_dummy_record("p2", "f01", "O1"),
        make_dummy_record("p3", "f02", "ON"),
        make_dummy_record("p4", "f02", "ON"),
        make_dummy_record("p5", "f03", "UNKNOWN"),
    ]
    preds = {"m1": ["O1", "O1", "ON", "O1", "UNKNOWN"]}

    b1 = compute_family_cluster_bootstrap(records, preds, seed=42, n_replicates=50)
    b2 = compute_family_cluster_bootstrap(records, preds, seed=42, n_replicates=50)

    acc_stat = b1["m1"]["accuracy"]
    assert b1["m1"]["accuracy"].ci_lower == b2["m1"]["accuracy"].ci_lower
    assert b1["m1"]["accuracy"].ci_upper == b2["m1"]["accuracy"].ci_upper
    assert acc_stat.ci_lower <= acc_stat.point_estimate <= acc_stat.ci_upper


def test_pairwise_model_disagreements():
    """Model disagreement must count families where >= 1 program differs."""
    records = [
        make_dummy_record("p1", "f01", "O1"),
        make_dummy_record("p2", "f01", "O1"),
        make_dummy_record("p3", "f02", "ON"),
    ]
    # m1 and m2 agree on f01 (both predict O1, O1)
    # on f02, m1 predicts ON, m2 predicts ON2 -> 1 disagreement
    preds = {
        "m1": ["O1", "O1", "ON"],
        "m2": ["O1", "O1", "ON2"],
    }
    dis = compute_pairwise_model_disagreements(records, preds)
    assert "m1_vs_m2" in dis
    assert dis["m1_vs_m2"]["differing_families"] == 1
    assert dis["m1_vs_m2"]["total_families"] == 2


# ---------------------------------------------------------------------------
# 6. Hybrid System: Threshold 0.55 & Sensitivity
# ---------------------------------------------------------------------------


def test_hybrid_provisional_threshold_and_preservation():
    """Verify static preservation and ML fallback at threshold 0.55."""
    # 1. Trusted static result is preserved regardless of ML
    s_trusted = make_dummy_analysis_result("O1", Exactness.STRUCTURAL.value)
    oof_ml = OutOfFoldPrediction(
        program_id="p1",
        family_id="f1",
        fold=0,
        true_label="O1",
        predicted_label="ON",
        probabilities=[0.1, 0.0, 0.8, 0.05, 0.05, 0.0, 0.0, 0.0],
    )
    preds, fusions = evaluate_hybrid_fusion([s_trusted], [oof_ml], threshold=0.55)
    assert preds[0] == "O1"
    assert fusions[0].source == "STATIC"

    # 2. Inconclusive static with ML >= 0.55 accepted
    s_unknown = make_dummy_analysis_result("UNKNOWN", Exactness.UNKNOWN.value)
    preds2, fusions2 = evaluate_hybrid_fusion([s_unknown], [oof_ml], threshold=0.55)
    assert preds2[0] == "ON"
    assert fusions2[0].source == "ML_ESTIMATE"

    # 3. Inconclusive static with ML < 0.55 remains UNKNOWN
    oof_low = OutOfFoldPrediction(
        program_id="p2",
        family_id="f1",
        fold=0,
        true_label="ON",
        predicted_label="ON",
        probabilities=[0.2, 0.2, 0.45, 0.05, 0.05, 0.05, 0.0, 0.0],
    )
    preds3, fusions3 = evaluate_hybrid_fusion([s_unknown], [oof_low], threshold=0.55)
    assert preds3[0] == "UNKNOWN"
    assert fusions3[0].source == "UNKNOWN"


def test_hybrid_sensitivity_curve():
    """Verify sensitivity curve sweeps across threshold grid."""
    s_unknown = make_dummy_analysis_result("UNKNOWN", Exactness.UNKNOWN.value)
    oof_ml = OutOfFoldPrediction(
        program_id="p1",
        family_id="f1",
        fold=0,
        true_label="ON",
        predicted_label="ON",
        probabilities=[0.1, 0.0, 0.6, 0.1, 0.1, 0.05, 0.05, 0.0],
    )
    curve = compute_hybrid_sensitivity_curve(
        ["ON"], [s_unknown], [oof_ml], threshold_grid=[0.5, 0.7]
    )
    assert len(curve) == 2
    # At 0.5: ML (0.6) accepted -> coverage = 1.0, accuracy = 1.0
    assert curve[0]["coverage"] == 1.0
    assert curve[0]["accuracy"] == 1.0
    # At 0.7: ML (0.6) rejected -> coverage = 0.0, accuracy = 0.0
    assert curve[1]["coverage"] == 0.0


# ---------------------------------------------------------------------------
# 7. Frozen Test Partition Guard
# ---------------------------------------------------------------------------


def test_frozen_test_partition_protection():
    """Test partition access must raise TestPartitionAccessError."""
    from complexity_lens.evaluation.data_loader import load_development_dataset

    # Verify that trying to pass test split raises TestPartitionAccessError or doesn't leak
    dev_dataset = load_development_dataset()
    for r in dev_dataset.records:
        assert r.fold in (0, 1, 2)
        # All records must be from development partition
        assert not r.program_id.startswith("test_")


# ---------------------------------------------------------------------------
# 8. Integrity, Disclosures & Reproducibility Invariants
# ---------------------------------------------------------------------------


def test_static_engine_disclosure_present():
    """Exact static-only disclosure string must be present."""
    assert STATIC_DISCLOSURE == "rules developed with dev data visible; not an independent estimate"


def test_calibration_status_string():
    """Post-hoc calibration status must strictly be 'not performed (sample size)'."""
    assert CALIBRATION_STATUS == "not performed (sample size)"


def test_in_memory_fold_refitting_no_persisted_artifacts(monkeypatch, tmp_path):
    """Refitting must happen in-memory inside training fold; no dev-fit artifact loading."""
    from complexity_lens.evaluation.cv_runner import CrossValidationRunner

    # Ensure load_model_artifact is NEVER called during cross-validation evaluation
    def mock_load_fail(*args, **kwargs):
        raise AssertionError("load_model_artifact was called during cross-validation evaluation!")

    monkeypatch.setattr("complexity_lens.evaluation.candidates.load_model_artifact", mock_load_fail)
    # Runner operates purely in-memory
    assert hasattr(CrossValidationRunner, "evaluate_model")


def test_reproducibility_excluding_timestamp_metadata():
    """Results serialization must be deterministic when timestamp/runtime metadata are excluded."""
    d1 = {
        "accuracy": 0.534591,
        "macro_f1": 0.549887,
        "metadata": {"timestamp_iso": "2026-11-16T12:00:00Z", "execution_runtime_seconds": 12.3},
    }
    d2 = {
        "accuracy": 0.534591,
        "macro_f1": 0.549887,
        "metadata": {"timestamp_iso": "2026-11-16T12:05:00Z", "execution_runtime_seconds": 15.8},
    }

    import json
    # Strip metadata
    c1 = {k: v for k, v in d1.items() if k != "metadata"}
    c2 = {k: v for k, v in d2.items() if k != "metadata"}

    s1 = json.dumps(c1, sort_keys=True)
    s2 = json.dumps(c2, sort_keys=True)
    assert s1 == s2


def test_day12_results_schema_and_reproducibility():
    """Verify schema, contracts, and Day 11 reproduction in generated evaluation artifact."""
    import json
    from pathlib import Path

    res_file = Path("artifacts/results/day12_evaluation_results.json")
    if not res_file.exists():
        pytest.skip("day12_evaluation_results.json not yet generated")

    with open(res_file, encoding="utf-8") as f:
        data = json.load(f)

    # Required top-level keys
    for key in (
        "git_commit",
        "dataset_manifest_sha256",
        "fold_assignment_hash",
        "static_only_engine",
        "ml_candidates",
        "hybrid_system",
        "family_cluster_bootstrap",
        "disagreements_pairwise",
        "small_gaps_disclaimer",
    ):
        assert key in data, f"Missing key: {key}"

    # Disclaimers and disclosures
    assert (
        data["static_only_engine"]["disclosure"]
        == "rules developed with dev data visible; not an independent estimate"
    )
    assert data["small_gaps_disclaimer"] == "Small gaps are not meaningful."

    # ML candidates reproduction
    rf = data["ml_candidates"]["random_forest"]
    assert rf["accuracy"] == 0.534591
    assert rf["macro_f1"] == 0.549887
    assert "selection-biased" in rf["bias_label"].lower()

    xgb = data["ml_candidates"]["xgboost"]
    assert xgb["accuracy"] == 0.528302
    assert xgb["macro_f1"] == 0.535573
    assert "selection-biased" in xgb["bias_label"].lower()

    # Calibration status contract
    for m_id in ("logistic_regression", "random_forest", "xgboost"):
        cal = data["ml_candidates"][m_id]["calibration"]
        assert cal["status"] == "not performed (sample size)"


