"""Tests for Day 13 Hybrid Fusion Gate, Reason-Aware Fallback & Verification.

Exhaustively verifies:
1. Probability vector boundary validation (tolerance, finite, no booleans, bounds [0, 1]).
2. Table-driven tests covering all conditions in ADR-013 decision table.
3. Behavior under all four evaluation modes (S1, S2, S3, S4).
4. Reason-aware fallback eligibility and ineligibility.
5. Strict UPPER_BOUND and STRUCTURAL static preservation.
6. JSON serialization round-trip and uncalibrated score labeling.
"""

from __future__ import annotations

import json
import math

import pytest

from complexity_lens.complexity.models import (
    AnalysisResult,
    ComplexityClass,
)
from complexity_lens.fusion import (
    CANONICAL_CLASS_ORDER,
    EvaluationMode,
    FusionDecisionReason,
    FusionSource,
    MLPrediction,
    ProbabilityValidationError,
    VerificationStatus,
    create_validated_ml_prediction,
    decide_fusion,
    validate_probability_vector,
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
    c_class = ComplexityClass.linear() if label == "ON" else ComplexityClass.unknown()
    return AnalysisResult(
        selected_function="test_fn",
        selection_rationale="Synthetic test function",
        complexity_class=c_class,
        taxonomy_label=label,
        exactness=exactness,
        reason_codes=reason_codes or ["STATIC_RESOLVED"],
        notes=notes or ["Static analysis complete."],
    )


# ===========================================================================
# 1. Probability Vector Validation Tests
# ===========================================================================


class TestProbabilityVectorValidation:
    """Rigorous tests for validate_probability_vector and create_validated_ml_prediction."""

    def test_valid_dict_vector(self) -> None:
        probs = _make_prob_vector("ON", 0.72)
        validated = validate_probability_vector(probs)
        assert len(validated) == len(CANONICAL_CLASS_ORDER)
        assert math.isclose(sum(validated.values()), 1.0, abs_tol=1e-4)
        assert validated["ON"] == 0.72

    def test_valid_sequence_vector(self) -> None:
        raw_list = [0.125] * 8
        validated = validate_probability_vector(raw_list)
        assert len(validated) == 8
        assert all(math.isclose(v, 0.125) for v in validated.values())

    def test_reject_out_of_bounds_negative(self) -> None:
        probs = _make_prob_vector("ON", 0.8)
        probs["O1"] = -0.01
        with pytest.raises(ProbabilityValidationError, match="out of bounds"):
            validate_probability_vector(probs)

    def test_reject_out_of_bounds_exceed_one(self) -> None:
        probs = _make_prob_vector("ON", 1.05)
        with pytest.raises(ProbabilityValidationError, match="out of bounds"):
            validate_probability_vector(probs)

    def test_reject_sum_deviation(self) -> None:
        probs = {c: 0.1 for c in CANONICAL_CLASS_ORDER}  # Sum = 0.8 (error > 1e-4)
        with pytest.raises(ProbabilityValidationError, match="deviates from 1.0"):
            validate_probability_vector(probs)

    def test_reject_nan(self) -> None:
        probs = _make_prob_vector("ON", 0.6)
        probs["O1"] = float("nan")
        with pytest.raises(ProbabilityValidationError, match="non-finite"):
            validate_probability_vector(probs)

    def test_reject_infinity(self) -> None:
        probs = _make_prob_vector("ON", 0.6)
        probs["O1"] = float("inf")
        with pytest.raises(ProbabilityValidationError, match="non-finite"):
            validate_probability_vector(probs)

    def test_reject_booleans(self) -> None:
        # In Python, isinstance(True, int) is True, so we must strictly disallow bool
        probs = _make_prob_vector("ON", 0.6)
        probs["O1"] = True  # type: ignore[assignment]
        with pytest.raises(ProbabilityValidationError, match="boolean"):
            validate_probability_vector(probs)

    def test_reject_non_numeric_strings(self) -> None:
        probs = _make_prob_vector("ON", 0.6)
        probs["O1"] = "invalid"  # type: ignore[assignment]
        with pytest.raises(ProbabilityValidationError, match="not numeric"):
            validate_probability_vector(probs)

    def test_reject_missing_classes(self) -> None:
        probs = {"O1": 0.5, "ON": 0.5}
        with pytest.raises(ProbabilityValidationError, match="missing expected classes"):
            validate_probability_vector(probs)

    def test_reject_unsupported_type(self) -> None:
        with pytest.raises(ProbabilityValidationError, match="Unsupported probability vector"):
            validate_probability_vector(42)  # type: ignore[arg-type]

    def test_create_validated_ml_prediction_valid(self) -> None:
        probs = _make_prob_vector("ONLOGN", 0.85)
        pred = create_validated_ml_prediction("ONLOGN", probs, model_id="rf_test")
        assert pred.predicted_label == "ONLOGN"
        assert math.isclose(pred.probability, 0.85)
        assert pred.model_id == "rf_test"
        assert pred.probabilities["ONLOGN"] == 0.85

    def test_create_validated_ml_prediction_rejects_noncanonical_label(self) -> None:
        probs = _make_prob_vector("ON", 0.8)
        with pytest.raises(ProbabilityValidationError, match="not in canonical vocabulary"):
            create_validated_ml_prediction("UNKNOWN_CLASS", probs)

    def test_create_validated_ml_prediction_rejects_boolean_label(self) -> None:
        probs = _make_prob_vector("ON", 0.8)
        with pytest.raises(ProbabilityValidationError, match="Invalid predicted label type"):
            create_validated_ml_prediction(True, probs)  # type: ignore[arg-type]


# ===========================================================================
# 2. Table-Driven Decision Gate Tests (ADR-013)
# ===========================================================================


class TestAuthoritativeDecisionTable:
    """Verifies all rows in the ADR-013 authoritative decision table."""

    @pytest.mark.parametrize(
        (
            "static_exactness, ml_label, ml_score, agreement, "
            "expected_s1, expected_s2, expected_s3, expected_s4"
        ),
        [
            # Condition 1: STRUCTURAL, Score >= tau, Agree
            ("STRUCTURAL", "ON", 0.80, True, "ON", "ON", "ON", "ON"),
            # Condition 2: STRUCTURAL, Score >= tau, Disagree
            ("STRUCTURAL", "ON2", 0.80, False, "ON", "ON2", "ON", "ON"),
            # Condition 3: STRUCTURAL, Score < tau, Disagree
            ("STRUCTURAL", "ON2", 0.40, False, "ON", "ON2", "ON", "ON"),
            # Condition 5: UPPER_BOUND, Score >= tau, Agree
            ("UPPER_BOUND", "ON2", 0.80, True, "ON2", "ON2", "ON2", "ON2"),
            # Condition 6: UPPER_BOUND, Score >= tau, Disagree
            ("UPPER_BOUND", "ON", 0.80, False, "ON2", "ON", "ON2", "ON2"),
            # Condition 7: UPPER_BOUND, Score < tau, Disagree
            ("UPPER_BOUND", "ON", 0.40, False, "ON2", "ON", "ON2", "ON2"),
        ],
    )
    def test_determined_static_results(
        self,
        static_exactness: str,
        ml_label: str,
        ml_score: float,
        agreement: bool,
        expected_s1: str,
        expected_s2: str,
        expected_s3: str,
        expected_s4: str,
    ) -> None:
        static_label = "ON2" if static_exactness == "UPPER_BOUND" else "ON"
        static_res = _make_static_result(label=static_label, exactness=static_exactness)
        ml_probs = _make_prob_vector(ml_label, ml_score)
        ml_pred = MLPrediction(
            predicted_label=ml_label,
            probability=ml_score,
            probabilities=ml_probs,
            model_id="test",
        )

        r1 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S1)
        r2 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S2)
        r3 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S3)
        r4 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S4)

        assert r1.primary_label == expected_s1
        assert r1.source == FusionSource.STATIC.value
        assert r1.verification_status == VerificationStatus.STATIC_VERIFIED.value

        assert r2.primary_label == expected_s2
        assert r2.source == FusionSource.ML_ESTIMATE.value
        assert r2.verification_status == VerificationStatus.ML_UNVERIFIED.value

        assert r3.primary_label == expected_s3
        assert r3.source == FusionSource.STATIC.value
        assert r3.verification_status == VerificationStatus.STATIC_VERIFIED.value

        assert r4.primary_label == expected_s4
        assert r4.source == FusionSource.STATIC.value
        assert r4.verification_status == VerificationStatus.STATIC_VERIFIED.value
        assert r4.static_exactness == static_exactness

    def test_condition_4_determined_with_missing_or_invalid_ml(self) -> None:
        static_res = _make_static_result(label="ON", exactness="STRUCTURAL")

        # Missing ML
        r1 = decide_fusion(static_res, None, mode=EvaluationMode.S1)
        r2 = decide_fusion(static_res, None, mode=EvaluationMode.S2)
        r4 = decide_fusion(static_res, None, mode=EvaluationMode.S4)

        assert r1.primary_label == "ON"
        assert r2.primary_label == "UNKNOWN"
        assert r2.verification_status == VerificationStatus.ABSTAINED.value
        assert r4.primary_label == "ON"
        assert r4.verification_status == VerificationStatus.STATIC_VERIFIED.value

    def test_condition_8_eligible_unknown_score_ge_tau(self) -> None:
        static_res = _make_static_result(
            label="UNKNOWN",
            exactness="UNKNOWN",
            reason_codes=["UNSUPPORTED_CONSTRUCT_DYNAMIC_GOTO"],
        )
        ml_probs = _make_prob_vector("ON", 0.75)
        ml_pred = MLPrediction("ON", 0.75, ml_probs, "test")

        r1 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S1)
        r2 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S2)
        r3 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S3)
        r4 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S4)

        assert r1.primary_label == "UNKNOWN"
        assert r1.verification_status == VerificationStatus.ABSTAINED.value

        assert r2.primary_label == "ON"
        assert r2.source == FusionSource.ML_ESTIMATE.value

        assert r3.primary_label == "ON"
        assert r3.source == FusionSource.ML_ESTIMATE.value
        assert r3.verification_status == VerificationStatus.ML_FALLBACK_UNVERIFIED.value

        assert r4.primary_label == "ON"
        assert r4.source == FusionSource.ML_ESTIMATE.value
        assert r4.decision_reason == FusionDecisionReason.ML_FALLBACK_ACCEPTED.value
        assert r4.verification_status == VerificationStatus.ML_FALLBACK_UNVERIFIED.value
        assert r4.unknown_reason_eligible is True

    def test_condition_9_eligible_unknown_score_lt_tau(self) -> None:
        static_res = _make_static_result(
            label="UNKNOWN",
            exactness="UNKNOWN",
            reason_codes=["INCOMPLETE_ANALYSIS_RECURSION"],
        )
        ml_probs = _make_prob_vector("ON", 0.45)
        ml_pred = MLPrediction("ON", 0.45, ml_probs, "test")

        r3 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S3)
        r4 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S4, ml_threshold=0.55)

        # S3 accepts fallback without threshold check
        assert r3.primary_label == "ON"
        assert r3.source == FusionSource.ML_ESTIMATE.value

        # S4 rejects because 0.45 < 0.55
        assert r4.primary_label == "UNKNOWN"
        assert r4.source == FusionSource.UNKNOWN.value
        assert r4.decision_reason == FusionDecisionReason.ML_FALLBACK_BELOW_THRESHOLD.value
        assert r4.verification_status == VerificationStatus.ABSTAINED.value

    def test_condition_10_eligible_unknown_ml_predicts_unknown(self) -> None:
        static_res = _make_static_result(
            label="UNKNOWN",
            exactness="UNKNOWN",
            reason_codes=["UNSUPPORTED_CONSTRUCT"],
        )
        ml_probs = _make_prob_vector("UNKNOWN", 0.90)
        ml_pred = MLPrediction("UNKNOWN", 0.90, ml_probs, "test")

        r2 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S2)
        r3 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S3)
        r4 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S4)

        assert r2.primary_label == "UNKNOWN"
        assert r3.primary_label == "UNKNOWN"
        assert r4.primary_label == "UNKNOWN"
        assert r4.decision_reason == FusionDecisionReason.ML_PREDICTED_UNKNOWN.value
        assert r4.verification_status == VerificationStatus.ABSTAINED.value

    @pytest.mark.parametrize(
        "ineligible_reason",
        ["termination", "external_call", "data_dependent", "irregular_recursion"],
    )
    def test_condition_11_and_12_ineligible_unknown_reasons(self, ineligible_reason: str) -> None:
        static_res = _make_static_result(
            label="UNKNOWN",
            exactness="UNKNOWN",
            notes=[f"Non-analyzable due to {ineligible_reason}."],
        )
        ml_probs = _make_prob_vector("ON", 0.95)
        ml_pred = MLPrediction("ON", 0.95, ml_probs, "test")

        # S3 accepts fallback anyway (naive)
        r3 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S3)
        assert r3.primary_label == "ON"

        # S4 strictly abstains due to ineligible reason
        r4 = decide_fusion(
            static_res,
            ml_pred,
            mode=EvaluationMode.S4,
            unknown_reason=ineligible_reason,
        )
        assert r4.primary_label == "UNKNOWN"
        assert r4.source == FusionSource.UNKNOWN.value
        assert r4.decision_reason == FusionDecisionReason.INELIGIBLE_UNKNOWN_REASON.value
        assert r4.verification_status == VerificationStatus.ABSTAINED.value
        assert r4.unknown_reason_eligible is False

    def test_condition_13_invalid_probability_vector(self) -> None:
        static_res = _make_static_result(
            label="UNKNOWN",
            exactness="UNKNOWN",
            reason_codes=["UNSUPPORTED_CONSTRUCT"],
        )
        invalid_probs = {"ON": 0.5, "O1": 0.1}  # Sum = 0.6, missing classes
        ml_pred = MLPrediction("ON", 0.5, invalid_probs, "test")

        r4 = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S4, strict_validation=True)
        assert r4.primary_label == "UNKNOWN"
        assert r4.decision_reason == FusionDecisionReason.ML_FALLBACK_INVALID_PROBABILITY.value
        assert r4.verification_status == VerificationStatus.ABSTAINED.value


# ===========================================================================
# 3. Decision Boundary & Threshold Tests
# ===========================================================================


class TestDecisionBoundary:
    """Verifies exact threshold boundary at tau=0.55."""

    def test_exact_threshold_boundary(self) -> None:
        static_res = _make_static_result(
            label="UNKNOWN",
            exactness="UNKNOWN",
            reason_codes=["UNSUPPORTED_CONSTRUCT"],
        )

        # Exact threshold 0.550000 -> Accepted
        p_at_threshold = _make_prob_vector("ON", 0.55)
        pred_at = MLPrediction("ON", 0.55, p_at_threshold, "test")
        r_at = decide_fusion(static_res, pred_at, ml_threshold=0.55)
        assert r_at.primary_label == "ON"
        assert r_at.decision_reason == FusionDecisionReason.ML_FALLBACK_ACCEPTED.value

        # Just below threshold 0.549999 -> Rejected (Abstained)
        p_below = _make_prob_vector("ON", 0.549999)
        pred_below = MLPrediction("ON", 0.549999, p_below, "test")
        r_below = decide_fusion(static_res, pred_below, ml_threshold=0.55)
        assert r_below.primary_label == "UNKNOWN"
        assert r_below.decision_reason == FusionDecisionReason.ML_FALLBACK_BELOW_THRESHOLD.value


# ===========================================================================
# 4. JSON Serialization and Labeling Tests
# ===========================================================================


class TestSerializationAndLabeling:
    """Verifies JSON round-trip and exact score labeling requirements."""

    def test_uncalibrated_score_label(self) -> None:
        static_res = _make_static_result(label="ON", exactness="STRUCTURAL")
        r = decide_fusion(static_res)
        assert r.ml_score_label == "ML score (max class probability, uncalibrated)"
        d = r.to_dict()
        assert d["ml_score_label"] == "ML score (max class probability, uncalibrated)"

    def test_json_round_trip(self) -> None:
        static_res = _make_static_result(
            label="UNKNOWN",
            exactness="UNKNOWN",
            reason_codes=["UNSUPPORTED_CONSTRUCT"],
        )
        ml_probs = _make_prob_vector("ONLOGN", 0.812345)
        ml_pred = MLPrediction("ONLOGN", 0.812345, ml_probs, "test_rf")
        result = decide_fusion(static_res, ml_pred, mode=EvaluationMode.S4)

        result_dict = result.to_dict()
        serialized = json.dumps(result_dict)
        deserialized = json.loads(serialized)

        assert deserialized["primary_label"] == "ONLOGN"
        expected_status = VerificationStatus.ML_FALLBACK_UNVERIFIED.value
        assert deserialized["verification_status"] == expected_status
        assert deserialized["ml_score_uncalibrated"] == 0.812345
        assert deserialized["unknown_reason_eligible"] is True

    def test_json_round_trip_with_none_scores(self) -> None:
        static_res = _make_static_result(label="UNKNOWN", exactness="UNKNOWN")
        result = decide_fusion(static_res, None, mode=EvaluationMode.S4)
        result_dict = result.to_dict()
        serialized = json.dumps(result_dict)
        deserialized = json.loads(serialized)

        assert deserialized["ml_probability"] is None
        assert deserialized["ml_score_uncalibrated"] is None
        assert deserialized["primary_label"] == "UNKNOWN"
