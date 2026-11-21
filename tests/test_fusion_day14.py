"""tests/test_fusion_day14.py — Unit tests for Day 14 Reason-Coded Abstention & Disagreement Policy.

Verifies:
1. All 10 conditions in the taxonomy-aware strong disagreement test matrix.
2. Canonical rejection categories mapping (LOW_CONFIDENCE, INVALID_OR_UNAVAILABLE_ML,
   ML_PREDICTED_UNKNOWN, UNSUPPORTED_STRUCTURE, UNRESOLVED_STATIC_ANALYSIS, STRONG_DISAGREEMENT).
3. Deterministic precedence of multiple rejection conditions.
4. Backward-compatible preservation of FusionResult fields, to_dict(), and decision_reason.
"""

from __future__ import annotations

from complexity_lens.complexity.models import AnalysisResult, ComplexityClass
from complexity_lens.fusion import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_REJECTION_CATEGORIES,
    REJECTION_CATEGORY_INVALID_OR_UNAVAILABLE_ML,
    REJECTION_CATEGORY_LOW_CONFIDENCE,
    REJECTION_CATEGORY_ML_PREDICTED_UNKNOWN,
    REJECTION_CATEGORY_STRONG_DISAGREEMENT,
    REJECTION_CATEGORY_UNRESOLVED_STATIC_ANALYSIS,
    REJECTION_CATEGORY_UNSUPPORTED_STRUCTURE,
    EvaluationMode,
    FusionDecisionReason,
    MLPrediction,
    VerificationStatus,
    decide_fusion,
)


def _make_prob_vector(pred_class: str, prob: float) -> dict[str, float]:
    """Helper to build a valid probability vector summing to 1.0."""
    remaining = (1.0 - prob) / (len(CANONICAL_CLASS_ORDER) - 1)
    return {c: (prob if c == pred_class else remaining) for c in CANONICAL_CLASS_ORDER}


def _make_static_result(
    label: str = "ON",
    exactness: str = "STRUCTURAL",
    reason_codes: list[str] | None = None,
    notes: list[str] | None = None,
) -> AnalysisResult:
    """Helper to build an AnalysisResult."""
    if label == "ON":
        c_class = ComplexityClass.linear()
    elif label == "ON2":
        c_class = ComplexityClass(p=2, q=0)
    elif label == "O1":
        c_class = ComplexityClass.constant()
    else:
        c_class = ComplexityClass.unknown()

    return AnalysisResult(
        selected_function="test_fn",
        selection_rationale="Synthetic test function",
        complexity_class=c_class,
        taxonomy_label=label,
        exactness=exactness,
        reason_codes=reason_codes or ["STATIC_RESOLVED"],
        notes=notes or ["Static analysis complete."],
    )


def _make_ml_prediction(label: str = "ON", prob: float = 0.8) -> MLPrediction:
    """Helper to build an MLPrediction."""
    return MLPrediction(
        predicted_label=label,
        probability=prob,
        probabilities=_make_prob_vector(label, prob),
        model_id="test_rf",
    )


# ===========================================================================
# 1. Taxonomy-Aware Disagreement Test Matrix (Conditions 1 to 10)
# ===========================================================================


class TestTaxonomyAwareDisagreementMatrix:
    """Exhaustively test the 10 conditions from the Day 14 disagreement matrix."""

    def test_condition_1_upper_bound_compatible_tighter_linear(self) -> None:
        """Condition 1: Static ON2 UPPER_BOUND vs ML ON (s=0.80 >= 0.55) -> Compatible."""
        static_res = _make_static_result("ON2", exactness="UPPER_BOUND")
        ml_pred = _make_ml_prediction("ON", 0.80)

        # Standard S4 (default)
        res_s4 = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=False)
        assert res_s4.primary_label == "ON2"
        assert res_s4.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert res_s4.rejection_category is None

        # Opt-in S4_STRONG_DISAGREEMENT
        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "ON2"
        assert res_opt.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert res_opt.rejection_category is None

    def test_condition_2_upper_bound_compatible_tighter_nlogn(self) -> None:
        """Condition 2: Static ON2 UPPER_BOUND vs ML ONLOGN (s=0.70 >= 0.55) -> Compatible."""
        static_res = _make_static_result("ON2", exactness="UPPER_BOUND")
        ml_pred = _make_ml_prediction("ONLOGN", 0.70)

        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "ON2"
        assert res_opt.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert res_opt.rejection_category is None

    def test_condition_3_upper_bound_contradiction_ml_exceeds(self) -> None:
        """Condition 3: Static ON UPPER_BOUND vs ML ON2 (s=0.85 >= 0.55) -> contradiction."""
        static_res = _make_static_result("ON", exactness="UPPER_BOUND")
        ml_pred = _make_ml_prediction("ON2", 0.85)

        # Standard S4: static preserved
        res_s4 = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=False)
        assert res_s4.primary_label == "ON"
        assert res_s4.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert res_s4.rejection_category is None

        # Opt-in S4_STRONG_DISAGREEMENT: abstains
        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "UNKNOWN"
        assert res_opt.decision_reason == FusionDecisionReason.STRONG_DISAGREEMENT.value
        assert res_opt.rejection_category == REJECTION_CATEGORY_STRONG_DISAGREEMENT
        assert res_opt.verification_status == VerificationStatus.ABSTAINED.value

    def test_condition_4_upper_bound_score_below_threshold_non_binding(self) -> None:
        """Condition 4: Static ON UPPER_BOUND vs ML ON2 (s=0.40 < 0.55) -> Non-binding."""
        static_res = _make_static_result("ON", exactness="UPPER_BOUND")
        ml_pred = _make_ml_prediction("ON2", 0.40)

        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "ON"
        assert res_opt.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert res_opt.rejection_category is None

    def test_condition_5_structural_compatible_identical(self) -> None:
        """Condition 5: Static ON STRUCTURAL vs ML ON (s=0.90 >= 0.55) -> Compatible."""
        static_res = _make_static_result("ON", exactness="STRUCTURAL")
        ml_pred = _make_ml_prediction("ON", 0.90)

        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "ON"
        assert res_opt.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert res_opt.agreement is True
        assert res_opt.rejection_category is None

    def test_condition_6_structural_contradiction_higher_class(self) -> None:
        """Condition 6: Static ON STRUCTURAL vs ML ON2 (s=0.80 >= 0.55) -> Genuine contradiction."""
        static_res = _make_static_result("ON", exactness="STRUCTURAL")
        ml_pred = _make_ml_prediction("ON2", 0.80)

        # Standard S4 preserves static
        res_s4 = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=False)
        assert res_s4.primary_label == "ON"
        assert res_s4.verification_status == VerificationStatus.STATIC_VERIFIED.value

        # Opt-in abstains
        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "UNKNOWN"
        assert res_opt.decision_reason == FusionDecisionReason.STRONG_DISAGREEMENT.value
        assert res_opt.rejection_category == REJECTION_CATEGORY_STRONG_DISAGREEMENT

    def test_condition_7_structural_contradiction_lower_class(self) -> None:
        """Condition 7: Static ON2 STRUCTURAL vs ML ON (s=0.75 >= 0.55) -> Genuine contradiction."""
        static_res = _make_static_result("ON2", exactness="STRUCTURAL")
        ml_pred = _make_ml_prediction("ON", 0.75)

        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "UNKNOWN"
        assert res_opt.decision_reason == FusionDecisionReason.STRONG_DISAGREEMENT.value
        assert res_opt.rejection_category == REJECTION_CATEGORY_STRONG_DISAGREEMENT

    def test_condition_8_structural_incomparable_ml_unknown(self) -> None:
        """Condition 8: Static ON STRUCTURAL vs ML UNKNOWN (s=0.60 >= 0.55) -> Incomparable."""
        static_res = _make_static_result("ON", exactness="STRUCTURAL")
        ml_pred = _make_ml_prediction("UNKNOWN", 0.60)

        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "ON"
        assert res_opt.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert res_opt.rejection_category is None

    def test_condition_9_static_inconclusive_fallback(self) -> None:
        """Condition 9: Static UNKNOWN vs ML ON (s=0.80 >= 0.55) -> Incomparable, fallback."""
        static_res = _make_static_result(
            "UNKNOWN", exactness="UNKNOWN", reason_codes=["unsupported_construct"]
        )
        ml_pred = _make_ml_prediction("ON", 0.80)

        res_opt = decide_fusion(static_res, ml_pred, reject_on_strong_disagreement=True)
        assert res_opt.primary_label == "ON"
        assert res_opt.decision_reason == FusionDecisionReason.ML_FALLBACK_ACCEPTED.value
        assert res_opt.rejection_category is None

    def test_condition_10_invalid_or_unavailable_ml(self) -> None:
        """Condition 10: Static ON STRUCTURAL vs Invalid / Unavailable ML -> Static preserved."""
        static_res = _make_static_result("ON", exactness="STRUCTURAL")

        res_none = decide_fusion(static_res, None, reject_on_strong_disagreement=True)
        assert res_none.primary_label == "ON"
        assert res_none.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert res_none.rejection_category is None

        # Non-finite probability ML
        bad_ml = MLPrediction(
            predicted_label="ON2",
            probability=float("nan"),
            probabilities={"ON2": float("nan")},
            model_id="bad",
        )
        res_bad = decide_fusion(static_res, bad_ml, reject_on_strong_disagreement=True)
        assert res_bad.primary_label == "ON"
        assert res_bad.verification_status == VerificationStatus.STATIC_VERIFIED.value


# ===========================================================================
# 2. Canonical Rejection Categories Verification
# ===========================================================================


class TestCanonicalRejectionCategories:
    """Verify that all six canonical rejection categories are mapped correctly."""

    def test_canonical_rejection_categories_tuple(self) -> None:
        assert len(CANONICAL_REJECTION_CATEGORIES) == 6
        assert REJECTION_CATEGORY_LOW_CONFIDENCE in CANONICAL_REJECTION_CATEGORIES
        assert REJECTION_CATEGORY_INVALID_OR_UNAVAILABLE_ML in CANONICAL_REJECTION_CATEGORIES
        assert REJECTION_CATEGORY_ML_PREDICTED_UNKNOWN in CANONICAL_REJECTION_CATEGORIES
        assert REJECTION_CATEGORY_UNSUPPORTED_STRUCTURE in CANONICAL_REJECTION_CATEGORIES
        assert REJECTION_CATEGORY_UNRESOLVED_STATIC_ANALYSIS in CANONICAL_REJECTION_CATEGORIES
        assert REJECTION_CATEGORY_STRONG_DISAGREEMENT in CANONICAL_REJECTION_CATEGORIES

    def test_low_confidence_rejection_category(self) -> None:
        """Score below threshold maps to LOW_CONFIDENCE."""
        static_res = _make_static_result(
            "UNKNOWN", exactness="UNKNOWN", reason_codes=["unsupported_construct"]
        )
        ml_pred = _make_ml_prediction("ON", 0.45)
        res = decide_fusion(static_res, ml_pred, ml_threshold=0.55)
        assert res.primary_label == "UNKNOWN"
        assert res.decision_reason == FusionDecisionReason.ML_FALLBACK_BELOW_THRESHOLD.value
        assert res.rejection_category == REJECTION_CATEGORY_LOW_CONFIDENCE

    def test_invalid_probability_rejection_category(self) -> None:
        """Invalid probability vector maps to INVALID_OR_UNAVAILABLE_ML."""
        static_res = _make_static_result(
            "UNKNOWN", exactness="UNKNOWN", reason_codes=["unsupported_construct"]
        )
        ml_pred = MLPrediction(
            predicted_label="ON",
            probability=float("nan"),
            probabilities={"ON": float("nan")},
            model_id="test",
        )
        res = decide_fusion(static_res, ml_pred)
        assert res.primary_label == "UNKNOWN"
        assert res.decision_reason == FusionDecisionReason.ML_FALLBACK_INVALID_PROBABILITY.value
        assert res.rejection_category == REJECTION_CATEGORY_INVALID_OR_UNAVAILABLE_ML

    def test_ml_predicted_unknown_category_s2_and_s4(self) -> None:
        """ML predicted UNKNOWN maps to ML_PREDICTED_UNKNOWN in S2 and S4."""
        ml_pred = _make_ml_prediction("UNKNOWN", 0.80)

        # S2
        res_s2 = decide_fusion(None, ml_pred, mode=EvaluationMode.S2)
        assert res_s2.primary_label == "UNKNOWN"
        assert res_s2.decision_reason == FusionDecisionReason.ML_PREDICTED_UNKNOWN.value
        assert res_s2.rejection_category == REJECTION_CATEGORY_ML_PREDICTED_UNKNOWN

        # S4
        static_res = _make_static_result("UNKNOWN", exactness="UNKNOWN")
        res_s4 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S4)
        assert res_s4.primary_label == "UNKNOWN"
        assert res_s4.decision_reason == FusionDecisionReason.ML_PREDICTED_UNKNOWN.value
        assert res_s4.rejection_category == REJECTION_CATEGORY_ML_PREDICTED_UNKNOWN

    def test_unsupported_structure_category(self) -> None:
        """Static unsupported construct without accepted ML maps to UNSUPPORTED_STRUCTURE."""
        static_res = _make_static_result(
            "UNKNOWN", exactness="UNKNOWN", reason_codes=["UNSUPPORTED_CONSTRUCT"]
        )
        # S1
        res_s1 = decide_fusion(static_res, None, mode=EvaluationMode.S1)
        assert res_s1.primary_label == "UNKNOWN"
        assert res_s1.rejection_category == REJECTION_CATEGORY_UNSUPPORTED_STRUCTURE

        # S4 with no ML
        res_s4 = decide_fusion(
            static_res,
            None,
            mode=EvaluationMode.S4,
            eligible_unknown_reasons=frozenset(),  # Force ineligible
        )
        assert res_s4.primary_label == "UNKNOWN"
        assert res_s4.rejection_category == REJECTION_CATEGORY_UNSUPPORTED_STRUCTURE

    def test_unresolved_static_analysis_category(self) -> None:
        """Static semantic uncertainty maps to UNRESOLVED_STATIC_ANALYSIS."""
        static_res = _make_static_result(
            "UNKNOWN", exactness="UNKNOWN", reason_codes=["DEPENDENT_BOUND"]
        )
        ml_pred = _make_ml_prediction("ON", 0.85)

        # Ineligible reason in S4 -> UNRESOLVED_STATIC_ANALYSIS
        res_s4 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S4)
        assert res_s4.primary_label == "UNKNOWN"
        assert res_s4.decision_reason == FusionDecisionReason.INELIGIBLE_UNKNOWN_REASON.value
        assert res_s4.rejection_category == REJECTION_CATEGORY_UNRESOLVED_STATIC_ANALYSIS


# ===========================================================================
# 3. Backward Compatibility of FusionResult Interface & to_dict()
# ===========================================================================


class TestFusionResultBackwardCompatibility:
    """Verify that FusionResult fields, field count, and serialization remain intact."""

    def test_fusion_result_fields_count_and_rejection_category(self) -> None:
        static_res = _make_static_result("ON", exactness="STRUCTURAL")
        res = decide_fusion(static_res, None)
        assert hasattr(res, "rejection_category")
        assert res.rejection_category is None

        # Check to_dict()
        d = res.to_dict()
        assert "rejection_category" in d
        assert d["rejection_category"] is None
        assert "decision_reason" in d
        assert d["decision_reason"] == FusionDecisionReason.STATIC_TRUSTED.value
        assert "primary_label" in d
        assert d["primary_label"] == "ON"
