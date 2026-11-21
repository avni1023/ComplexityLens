"""
fusion.py — Deterministic hybrid fusion gate for ComplexityLens.

Combines static analysis results with machine learning estimates using a
pure, deterministic decision gate:
- Trusted static results (STRUCTURAL or UPPER_BOUND) are always preserved.
- UPPER_BOUND exactness is preserved without allowing ML to tighten or alter the bound.
- When static analysis is UNKNOWN or inconclusive, an ML estimate may be
  supplied in S4 if and only if the static UNKNOWN reason is eligible, the ML predicted
  label is valid, and its uncalibrated score meets the decision threshold (default 0.55).
- Ineligible UNKNOWN reasons (termination, external_call, data_dependent, irregular_recursion)
  strictly remain UNKNOWN in S4.
- Evaluates four distinct modes: S1 (Static-only), S2 (ML-only), S3 (Naive hybrid),
  and S4 (Hybrid with verification and abstention).
- Pure function: no I/O, no side effects, no state mutation.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from enum import Enum
from typing import Any

from complexity_lens.complexity.models import AnalysisResult, Exactness, TaxonomyLabel

DEFAULT_ML_THRESHOLD: float = 0.55
DEFAULT_PROBABILITY_TOLERANCE: float = 1e-4
ML_SCORE_LABEL: str = "ML score (max class probability, uncalibrated)"

CANONICAL_CLASS_ORDER: tuple[str, ...] = (
    "O1",
    "OLOGN",
    "ON",
    "ONLOGN",
    "ON2",
    "ON3",
    "OEXP",
    "UNKNOWN",
)

# Default reason eligibility sets for S4 reason-aware fallback
DEFAULT_ELIGIBLE_UNKNOWN_REASONS: frozenset[str] = frozenset({
    "unsupported_construct",
    "incomplete_analysis",
    "incomplete_ir",
    "partial_ir",
    "UNSUPPORTED_CONSTRUCT",
    "INCOMPLETE_IR",
    "PARTIAL_IR",
    "UNKNOWN_BOUND",  # Included for backward-compatibility with prior tests
})

DEFAULT_INELIGIBLE_UNKNOWN_REASONS: frozenset[str] = frozenset({
    "termination",
    "external_call",
    "data_dependent",
    "irregular_recursion",
    "DEPENDENT_BOUND",
})


class EvaluationMode(str, Enum):
    """Evaluation and fusion execution modes."""

    S1 = "S1"  # Static-only
    S2 = "S2"  # ML-only
    S3 = "S3"  # Naive hybrid (static-first, ML fallback for any non-supported, no gate)
    S4 = "S4"  # Hybrid with verification and abstention (reason-aware, threshold-gated)


class FusionSource(str, Enum):
    """Origin of the final complexity label."""

    STATIC = "STATIC"
    ML_ESTIMATE = "ML_ESTIMATE"
    UNKNOWN = "UNKNOWN"


class FusionDecisionReason(str, Enum):
    """Machine-readable reason for the fusion decision."""

    STATIC_TRUSTED = "STATIC_TRUSTED"
    ML_FALLBACK_ACCEPTED = "ML_FALLBACK_ACCEPTED"
    ML_FALLBACK_BELOW_THRESHOLD = "ML_FALLBACK_BELOW_THRESHOLD"
    ML_FALLBACK_INVALID_PROBABILITY = "ML_FALLBACK_INVALID_PROBABILITY"
    ML_PREDICTED_UNKNOWN = "ML_PREDICTED_UNKNOWN"
    INCONCLUSIVE_NO_ML = "INCONCLUSIVE_NO_ML"
    INCONCLUSIVE_ALL = "INCONCLUSIVE_ALL"
    INELIGIBLE_UNKNOWN_REASON = "INELIGIBLE_UNKNOWN_REASON"
    S1_STATIC_ONLY = "S1_STATIC_ONLY"
    S2_ML_ONLY = "S2_ML_ONLY"
    STRONG_DISAGREEMENT = "STRONG_DISAGREEMENT"


CLASS_DOMINANCE_RANK: dict[str, int] = {
    "O1": 1,
    "OLOGN": 2,
    "ON": 3,
    "ONLOGN": 4,
    "ON2": 5,
    "ON3": 6,
    "OEXP": 7,
}

REJECTION_CATEGORY_LOW_CONFIDENCE: str = "LOW_CONFIDENCE"
REJECTION_CATEGORY_INVALID_OR_UNAVAILABLE_ML: str = "INVALID_OR_UNAVAILABLE_ML"
REJECTION_CATEGORY_ML_PREDICTED_UNKNOWN: str = "ML_PREDICTED_UNKNOWN"
REJECTION_CATEGORY_UNSUPPORTED_STRUCTURE: str = "UNSUPPORTED_STRUCTURE"
REJECTION_CATEGORY_UNRESOLVED_STATIC_ANALYSIS: str = "UNRESOLVED_STATIC_ANALYSIS"
REJECTION_CATEGORY_STRONG_DISAGREEMENT: str = "STRONG_DISAGREEMENT"

CANONICAL_REJECTION_CATEGORIES: tuple[str, ...] = (
    REJECTION_CATEGORY_LOW_CONFIDENCE,
    REJECTION_CATEGORY_INVALID_OR_UNAVAILABLE_ML,
    REJECTION_CATEGORY_ML_PREDICTED_UNKNOWN,
    REJECTION_CATEGORY_UNSUPPORTED_STRUCTURE,
    REJECTION_CATEGORY_UNRESOLVED_STATIC_ANALYSIS,
    REJECTION_CATEGORY_STRONG_DISAGREEMENT,
)


def _classify_static_rejection(reason: str | None) -> str:
    """Classify an inconclusive static analysis reason into a canonical rejection category."""
    if reason and any(k in reason.lower() for k in ("unsupported", "incomplete", "partial")):
        return REJECTION_CATEGORY_UNSUPPORTED_STRUCTURE
    return REJECTION_CATEGORY_UNRESOLVED_STATIC_ANALYSIS


class VerificationStatus(str, Enum):
    """Diagnostic verification status of the fused decision."""

    STATIC_VERIFIED = "STATIC_VERIFIED"
    ML_FALLBACK_UNVERIFIED = "ML_FALLBACK_UNVERIFIED"
    ML_UNVERIFIED = "ML_UNVERIFIED"
    ABSTAINED = "ABSTAINED"


class ProbabilityValidationError(ValueError):
    """Raised when an ML probability vector is structurally invalid or non-finite."""


def validate_probability_vector(
    probabilities: Any,
    *,
    expected_classes: Sequence[str] = CANONICAL_CLASS_ORDER,
    tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
) -> dict[str, float]:
    """Validate that a probability distribution is complete, finite, bounded, and sums to 1.0.

    Parameters
    ----------
    probabilities:
        Dictionary mapping class label to probability, sequence of probabilities matching
        expected_classes order, or 1D array.
    expected_classes:
        Ordered canonical class vocabulary (default: CANONICAL_CLASS_ORDER, 8 classes).
    tolerance:
        Maximum allowed absolute deviation of probability sum from 1.0 (default: 1e-4).

    Returns
    -------
    dict[str, float]
        Validated dictionary mapping canonical class label to float probability.

    Raises
    ------
    ProbabilityValidationError
        If vector is missing, contains non-finite values, contains values outside [0, 1],
        contains boolean values, has incorrect length, or does not sum to 1.0 ± tolerance.
    """
    if probabilities is None:
        raise ProbabilityValidationError("Probabilities cannot be None.")

    if isinstance(probabilities, bool):
        raise ProbabilityValidationError("Probabilities cannot be a boolean.")

    result_dict: dict[str, float] = {}

    if isinstance(probabilities, dict):
        missing_classes = [c for c in expected_classes if c not in probabilities]
        if missing_classes:
            raise ProbabilityValidationError(
                f"Probability dictionary missing expected classes: {missing_classes}"
            )
        for c in expected_classes:
            val = probabilities[c]
            if isinstance(val, bool):
                raise ProbabilityValidationError(f"Probability for class '{c}' is a boolean: {val}")
            try:
                f_val = float(val)
            except (TypeError, ValueError) as err:
                raise ProbabilityValidationError(
                    f"Probability for class '{c}' is not numeric: {val!r}"
                ) from err

            if not math.isfinite(f_val):
                raise ProbabilityValidationError(
                    f"Probability for class '{c}' is non-finite: {f_val}"
                )
            if f_val < 0.0 or f_val > 1.0:
                raise ProbabilityValidationError(
                    f"Probability for class '{c}' is out of bounds [0, 1]: {f_val}"
                )
            result_dict[c] = f_val
    elif isinstance(probabilities, (list, tuple)) or hasattr(probabilities, "__len__"):
        if len(probabilities) != len(expected_classes):
            raise ProbabilityValidationError(
                f"Expected probability sequence of length {len(expected_classes)}, "
                f"got {len(probabilities)}."
            )
        for c, val in zip(expected_classes, probabilities):
            if isinstance(val, bool):
                raise ProbabilityValidationError(f"Probability for class '{c}' is a boolean: {val}")
            try:
                f_val = float(val)
            except (TypeError, ValueError) as err:
                raise ProbabilityValidationError(
                    f"Probability for class '{c}' is not numeric: {val!r}"
                ) from err

            if not math.isfinite(f_val):
                raise ProbabilityValidationError(
                    f"Probability for class '{c}' is non-finite: {f_val}"
                )
            if f_val < 0.0 or f_val > 1.0:
                raise ProbabilityValidationError(
                    f"Probability for class '{c}' is out of bounds [0, 1]: {f_val}"
                )
            result_dict[c] = f_val
    else:
        raise ProbabilityValidationError(
            f"Unsupported probability vector type: {type(probabilities).__name__}"
        )

    prob_sum = sum(result_dict.values())
    if abs(prob_sum - 1.0) > tolerance:
        raise ProbabilityValidationError(
            f"Probability vector sum ({prob_sum:.6f}) deviates from 1.0 by more than "
            f"numerical tolerance {tolerance}."
        )

    return result_dict


def create_validated_ml_prediction(
    predicted_label: str,
    probabilities: Any,
    model_id: str = "random_forest",
    *,
    expected_classes: Sequence[str] = CANONICAL_CLASS_ORDER,
    tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
) -> MLPrediction:
    """Construct an MLPrediction with strict boundary validation."""
    if isinstance(predicted_label, bool) or not isinstance(predicted_label, str):
        raise ProbabilityValidationError(f"Invalid predicted label type: {type(predicted_label)}")

    if predicted_label not in expected_classes:
        raise ProbabilityValidationError(
            f"Predicted label '{predicted_label}' not in canonical vocabulary {expected_classes}."
        )

    validated_probs = validate_probability_vector(
        probabilities, expected_classes=expected_classes, tolerance=tolerance
    )
    score = max(validated_probs.values())

    return MLPrediction(
        predicted_label=predicted_label,
        probability=score,
        probabilities=validated_probs,
        model_id=model_id,
    )


@dataclasses.dataclass(frozen=True)
class MLPrediction:
    """Structured machine learning prediction container."""

    predicted_label: str
    probability: float
    probabilities: dict[str, float]
    model_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "predicted_label": self.predicted_label,
            "probabilities": dict(self.probabilities) if self.probabilities is not None else None,
            "probability": round(self.probability, 6) if self.probability is not None else None,
        }


@dataclasses.dataclass(frozen=True)
class FusionResult:
    """Deterministic result produced by the hybrid fusion gate."""

    primary_label: str
    source: str
    decision_reason: str
    rationale: str
    static_label: str | None
    static_exactness: str | None
    ml_label: str | None
    ml_probability: float | None
    ml_probabilities: dict[str, float] | None
    agreement: bool | None
    threshold_used: float
    is_provisional_threshold: bool = True
    mode_used: str = "S4"
    verification_status: str = "STATIC_VERIFIED"
    abstention_reason: str | None = None
    unknown_reason: str | None = None
    unknown_reason_eligible: bool | None = None
    ml_score_uncalibrated: float | None = None
    ml_score_label: str = ML_SCORE_LABEL
    rejection_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstention_reason": self.abstention_reason,
            "agreement": self.agreement,
            "decision_reason": self.decision_reason,
            "is_provisional_threshold": self.is_provisional_threshold,
            "ml_label": self.ml_label,
            "ml_probabilities": dict(self.ml_probabilities) if self.ml_probabilities else None,
            "ml_probability": round(self.ml_probability, 6)
            if self.ml_probability is not None
            else None,
            "ml_score_label": self.ml_score_label,
            "ml_score_uncalibrated": round(self.ml_score_uncalibrated, 6)
            if self.ml_score_uncalibrated is not None
            else None,
            "mode_used": self.mode_used,
            "primary_label": self.primary_label,
            "rationale": self.rationale,
            "rejection_category": self.rejection_category,
            "source": self.source,
            "static_exactness": self.static_exactness,
            "static_label": self.static_label,
            "threshold_used": self.threshold_used,
            "unknown_reason": self.unknown_reason,
            "unknown_reason_eligible": self.unknown_reason_eligible,
            "verification_status": self.verification_status,
        }


def _extract_unknown_reason(static_result: AnalysisResult | None) -> str | None:
    """Extract an explicit UNKNOWN reason from AnalysisResult reason codes or notes."""
    if static_result is None:
        return None

    # Check reason codes for known reason keywords
    for code in static_result.reason_codes:
        c_lower = code.lower()
        if "unsupported_construct" in c_lower:
            return "unsupported_construct"
        if "incomplete_ir" in c_lower or "partial_ir" in c_lower or "incomplete" in c_lower:
            return "incomplete_analysis"
        if "dependent_bound" in c_lower or "data_dependent" in c_lower:
            return "data_dependent"
        if "termination" in c_lower:
            return "termination"
        if "external" in c_lower:
            return "external_call"
        if "irregular" in c_lower:
            return "irregular_recursion"

    for note in static_result.notes:
        n_lower = note.lower()
        if "unsupported" in n_lower:
            return "unsupported_construct"
        if "incomplete" in n_lower or "partial" in n_lower:
            return "incomplete_analysis"
        if "data-dependent" in n_lower or "data dependent" in n_lower:
            return "data_dependent"
        if "termination" in n_lower:
            return "termination"
        if "external" in n_lower:
            return "external_call"

    return None


def decide_fusion(
    static_result: AnalysisResult | None,
    ml_prediction: MLPrediction | None = None,
    *,
    ml_threshold: float = DEFAULT_ML_THRESHOLD,
    mode: str | EvaluationMode = EvaluationMode.S4,
    eligible_unknown_reasons: set[str] | Sequence[str] | None = None,
    unknown_reason: str | None = None,
    strict_validation: bool = False,
    reject_on_strong_disagreement: bool = False,
) -> FusionResult:
    """Pure deterministic decision gate combining static and ML results.

    Supports four distinct evaluation modes:
    - S1: Static-only (preserves static outcomes; no ML replacement).
    - S2: ML-only (uses ML predictions exclusively; abstains on missing/unknown).
    - S3: Naive hybrid (static-first; any valid ML prediction provides fallback).
    - S4: Hybrid with verification and abstention (reason-aware fallback, score >= threshold).

    Parameters
    ----------
    static_result:
        Output of static complexity engine, or None if analysis was not run / failed.
    ml_prediction:
        Optional ML prediction container, or None if ML was not run / unavailable.
    ml_threshold:
        Provisional decision threshold for ML fallback acceptance (default: 0.55).
    mode:
        Evaluation mode ('S1', 'S2', 'S3', 'S4'). Defaults to 'S4'.
    eligible_unknown_reasons:
        Set of UNKNOWN reason identifiers eligible for ML fallback in S4.
        Defaults to DEFAULT_ELIGIBLE_UNKNOWN_REASONS.
    unknown_reason:
        Explicit ground-truth or static UNKNOWN reason string, if known.
    strict_validation:
        If True, strictly enforces probability vector sum == 1.0 ± 1e-4 and full class vocabulary.
    reject_on_strong_disagreement:
        If True, enables the opt-in disagreement-rejection policy (S4_STRONG_DISAGREEMENT),
        abstaining to UNKNOWN if a supported static result is contradicted by ML.

    Returns
    -------
    FusionResult
        Structured fusion decision with authoritative source, agreement, rationale, and metadata.
    """
    mode_str = mode.value if isinstance(mode, EvaluationMode) else str(mode).upper()
    if mode_str not in ("S1", "S2", "S3", "S4"):
        raise ValueError(f"Unknown evaluation mode: {mode!r}. Expected 'S1', 'S2', 'S3', or 'S4'.")

    # Resolve eligible reason set
    if eligible_unknown_reasons is None:
        eligible_set = set(DEFAULT_ELIGIBLE_UNKNOWN_REASONS)
    else:
        eligible_set = set(eligible_unknown_reasons)

    # Determine effective unknown reason
    effective_reason = unknown_reason or _extract_unknown_reason(static_result)

    # Check whether the reason is eligible
    is_reason_eligible: bool | None = None
    if effective_reason is not None:
        is_reason_eligible = (
            effective_reason in eligible_set
            or effective_reason.lower() in {r.lower() for r in eligible_set}
        )
        if effective_reason.lower() in {r.lower() for r in DEFAULT_INELIGIBLE_UNKNOWN_REASONS}:
            is_reason_eligible = False

    # Extract static information
    static_label = static_result.taxonomy_label if static_result is not None else None
    static_exactness = static_result.exactness if static_result is not None else None

    # Supported static result definition
    static_supported = (
        static_result is not None
        and static_label is not None
        and static_label not in (TaxonomyLabel.UNKNOWN.value, "UNKNOWN")
        and static_exactness in (Exactness.STRUCTURAL.value, Exactness.UPPER_BOUND.value)
    )

    # Extract ML information
    ml_label = ml_prediction.predicted_label if ml_prediction is not None else None
    ml_prob = ml_prediction.probability if ml_prediction is not None else None
    ml_probs = ml_prediction.probabilities if ml_prediction is not None else None

    # Probability and prediction validity checking
    is_valid_ml = False
    validation_err_msg: str | None = None

    if ml_prediction is not None and ml_label is not None and ml_prob is not None:
        if isinstance(ml_prob, bool) or not isinstance(ml_prob, (int, float)):
            validation_err_msg = f"Probability must be numeric, got {type(ml_prob).__name__}"
        elif not math.isfinite(ml_prob):
            validation_err_msg = f"Probability must be finite, got {ml_prob}"
        elif ml_prob < 0.0 or ml_prob > 1.0:
            validation_err_msg = f"Probability must be within [0, 1], got {ml_prob}"
        elif ml_label not in CANONICAL_CLASS_ORDER and ml_label not in [
            e.value for e in TaxonomyLabel
        ]:
            validation_err_msg = f"Predicted label '{ml_label}' is noncanonical"
        else:
            is_valid_ml = True

        # Check full vector if strict validation or if probabilities is present
        if is_valid_ml and ml_probs is not None:
            if strict_validation:
                try:
                    validate_probability_vector(ml_probs)
                except ProbabilityValidationError as pve:
                    is_valid_ml = False
                    validation_err_msg = str(pve)
            else:
                # Basic sanity check on dictionary values (no booleans, finite, bounds)
                for k, v in ml_probs.items():
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        is_valid_ml = False
                        validation_err_msg = f"Probability for '{k}' must be numeric"
                        break
                    if not math.isfinite(v) or v < 0.0 or v > 1.0:
                        is_valid_ml = False
                        validation_err_msg = f"Probability for '{k}' out of range: {v}"
                        break

    ml_score_uncalibrated = ml_prob if is_valid_ml else None

    # Agreement calculation
    agreement: bool | None = None
    if (
        static_label is not None
        and ml_label is not None
        and static_label != "UNKNOWN"
        and ml_label != "UNKNOWN"
    ):
        agreement = static_label == ml_label

    # =======================================================================
    # MODE S1: Static-Only
    # =======================================================================
    if mode_str == "S1":
        if static_supported and static_label is not None:
            return FusionResult(
                primary_label=static_label,
                source=FusionSource.STATIC.value,
                decision_reason=FusionDecisionReason.STATIC_TRUSTED.value,
                rationale=(
                    f"S1 Static-Only: Static analysis established a verified {static_exactness} "
                    f"bound of {static_label}."
                ),
                static_label=static_label,
                static_exactness=static_exactness,
                ml_label=ml_label,
                ml_probability=ml_score_uncalibrated,
                ml_probabilities=ml_probs if is_valid_ml else None,
                agreement=agreement,
                threshold_used=ml_threshold,
                is_provisional_threshold=True,
                mode_used="S1",
                verification_status=VerificationStatus.STATIC_VERIFIED.value,
                abstention_reason=None,
                unknown_reason=effective_reason,
                unknown_reason_eligible=is_reason_eligible,
                ml_score_uncalibrated=ml_score_uncalibrated,
                rejection_category=None,
            )
        return FusionResult(
            primary_label=TaxonomyLabel.UNKNOWN.value,
            source=FusionSource.UNKNOWN.value,
            decision_reason=FusionDecisionReason.INCONCLUSIVE_ALL.value,
            rationale="S1 Static-Only: Static analysis was inconclusive (UNKNOWN).",
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=ml_label,
            ml_probability=ml_score_uncalibrated,
            ml_probabilities=ml_probs if is_valid_ml else None,
            agreement=agreement,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used="S1",
            verification_status=VerificationStatus.ABSTAINED.value,
            abstention_reason="Static analysis was inconclusive in S1 mode.",
            unknown_reason=effective_reason,
            unknown_reason_eligible=is_reason_eligible,
            ml_score_uncalibrated=ml_score_uncalibrated,
            rejection_category=_classify_static_rejection(effective_reason),
        )

    # =======================================================================
    # MODE S2: ML-Only
    # =======================================================================
    if mode_str == "S2":
        if not is_valid_ml or ml_label is None:
            return FusionResult(
                primary_label=TaxonomyLabel.UNKNOWN.value,
                source=FusionSource.UNKNOWN.value,
                decision_reason=(
                    FusionDecisionReason.ML_FALLBACK_INVALID_PROBABILITY.value
                    if validation_err_msg
                    else FusionDecisionReason.INCONCLUSIVE_NO_ML.value
                ),
                rationale=(
                    f"S2 ML-Only: ML prediction was unavailable or invalid: "
                    f"{validation_err_msg or 'None'}."
                ),
                static_label=static_label,
                static_exactness=static_exactness,
                ml_label=ml_label,
                ml_probability=None,
                ml_probabilities=None,
                agreement=None,
                threshold_used=ml_threshold,
                is_provisional_threshold=True,
                mode_used="S2",
                verification_status=VerificationStatus.ABSTAINED.value,
                abstention_reason="ML prediction was unavailable or invalid in S2 mode.",
                unknown_reason=effective_reason,
                unknown_reason_eligible=is_reason_eligible,
                ml_score_uncalibrated=None,
                rejection_category=REJECTION_CATEGORY_INVALID_OR_UNAVAILABLE_ML,
            )

        if ml_label in (TaxonomyLabel.UNKNOWN.value, "UNKNOWN"):
            return FusionResult(
                primary_label=TaxonomyLabel.UNKNOWN.value,
                source=FusionSource.UNKNOWN.value,
                decision_reason=FusionDecisionReason.ML_PREDICTED_UNKNOWN.value,
                rationale="S2 ML-Only: ML model predicted UNKNOWN.",
                static_label=static_label,
                static_exactness=static_exactness,
                ml_label=ml_label,
                ml_probability=ml_score_uncalibrated,
                ml_probabilities=ml_probs,
                agreement=None,
                threshold_used=ml_threshold,
                is_provisional_threshold=True,
                mode_used="S2",
                verification_status=VerificationStatus.ABSTAINED.value,
                abstention_reason="ML model predicted UNKNOWN in S2 mode.",
                unknown_reason=effective_reason,
                unknown_reason_eligible=is_reason_eligible,
                ml_score_uncalibrated=ml_score_uncalibrated,
                rejection_category=REJECTION_CATEGORY_ML_PREDICTED_UNKNOWN,
            )

        return FusionResult(
            primary_label=ml_label,
            source=FusionSource.ML_ESTIMATE.value,
            decision_reason=FusionDecisionReason.ML_FALLBACK_ACCEPTED.value,
            rationale=(
                f"S2 ML-Only: Using standalone ML prediction {ml_label} "
                f"with score {ml_score_uncalibrated:.4f}."
            ),
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=ml_label,
            ml_probability=ml_score_uncalibrated,
            ml_probabilities=ml_probs,
            agreement=agreement,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used="S2",
            verification_status=VerificationStatus.ML_UNVERIFIED.value,
            abstention_reason=None,
            unknown_reason=effective_reason,
            unknown_reason_eligible=is_reason_eligible,
            ml_score_uncalibrated=ml_score_uncalibrated,
            rejection_category=None,
        )

    # =======================================================================
    # SHARED RULE: Supported Static Results Always Win (S3 & S4)
    # =======================================================================
    if static_supported and static_label is not None:
        # Check for genuine contradiction under opt-in disagreement policy
        if reject_on_strong_disagreement:
            is_contradiction = False
            if (
                is_valid_ml
                and ml_label is not None
                and ml_label not in (TaxonomyLabel.UNKNOWN.value, "UNKNOWN")
                and ml_score_uncalibrated is not None
                and ml_score_uncalibrated >= ml_threshold
            ):
                if static_exactness == Exactness.STRUCTURAL.value:
                    if ml_label != static_label:
                        is_contradiction = True
                elif static_exactness == Exactness.UPPER_BOUND.value:
                    r_static = CLASS_DOMINANCE_RANK.get(static_label)
                    r_ml = CLASS_DOMINANCE_RANK.get(ml_label)
                    if r_static is not None and r_ml is not None and r_ml > r_static:
                        is_contradiction = True

            if is_contradiction:
                return FusionResult(
                    primary_label=TaxonomyLabel.UNKNOWN.value,
                    source=FusionSource.UNKNOWN.value,
                    decision_reason=FusionDecisionReason.STRONG_DISAGREEMENT.value,
                    rationale=(
                        f"Strong disagreement policy rejected prediction: static analysis "
                        f"established {static_exactness} bound of {static_label}, but "
                        f"high-confidence ML predicted contradictory class {ml_label} "
                        f"(score {ml_score_uncalibrated:.4f} >= {ml_threshold:.2f}); "
                        "abstained to UNKNOWN."
                    ),
                    static_label=static_label,
                    static_exactness=static_exactness,
                    ml_label=ml_label,
                    ml_probability=ml_score_uncalibrated,
                    ml_probabilities=ml_probs if is_valid_ml else None,
                    agreement=False,
                    threshold_used=ml_threshold,
                    is_provisional_threshold=True,
                    mode_used=mode_str,
                    verification_status=VerificationStatus.ABSTAINED.value,
                    abstention_reason=(
                        f"Strong disagreement: verified {static_exactness} bound {static_label} "
                        f"contradicted by ML {ml_label} (score {ml_score_uncalibrated:.4f})."
                    ),
                    unknown_reason=effective_reason,
                    unknown_reason_eligible=is_reason_eligible,
                    ml_score_uncalibrated=ml_score_uncalibrated,
                    rejection_category=REJECTION_CATEGORY_STRONG_DISAGREEMENT,
                )

        # Special case: UPPER_BOUND is strictly preserved
        if static_exactness == Exactness.UPPER_BOUND.value:
            if ml_label is not None:
                agree_str = "agrees" if agreement else "disagrees"
                rationale = (
                    f"Static analysis established a verified UPPER_BOUND of {static_label}. "
                    f"ML model predicted {ml_label} ({agree_str}); "
                    "verified static upper bound is preserved."
                )
            else:
                rationale = (
                    f"Static analysis established a verified UPPER_BOUND of {static_label}."
                )
        else:
            if ml_label is not None:
                agree_str = "agrees" if agreement else "disagrees"
                rationale = (
                    f"Static analysis established a verified {static_exactness} bound of "
                    f"{static_label}. ML model predicted {ml_label} ({agree_str}); "
                    f"verified static result is preserved."
                )
            else:
                rationale = (
                    f"Static analysis established a verified {static_exactness} bound of "
                    f"{static_label}."
                )

        return FusionResult(
            primary_label=static_label,
            source=FusionSource.STATIC.value,
            decision_reason=FusionDecisionReason.STATIC_TRUSTED.value,
            rationale=rationale,
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=ml_label,
            ml_probability=ml_score_uncalibrated,
            ml_probabilities=ml_probs if is_valid_ml else None,
            agreement=agreement,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used=mode_str,
            verification_status=VerificationStatus.STATIC_VERIFIED.value,
            abstention_reason=None,
            unknown_reason=effective_reason,
            unknown_reason_eligible=is_reason_eligible,
            ml_score_uncalibrated=ml_score_uncalibrated,
            rejection_category=None,
        )

    # =======================================================================
    # STATIC INCONCLUSIVE / UNKNOWN (None, UNKNOWN label, or UNKNOWN exactness)
    # =======================================================================

    # Check ML availability
    if ml_prediction is None or ml_label is None:
        return FusionResult(
            primary_label=TaxonomyLabel.UNKNOWN.value,
            source=FusionSource.UNKNOWN.value,
            decision_reason=FusionDecisionReason.INCONCLUSIVE_NO_ML.value,
            rationale=(
                "Static analysis was inconclusive (UNKNOWN) and no ML prediction was available."
            ),
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=None,
            ml_probability=None,
            ml_probabilities=None,
            agreement=None,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used=mode_str,
            verification_status=VerificationStatus.ABSTAINED.value,
            abstention_reason="Static analysis was inconclusive and ML prediction was unavailable.",
            unknown_reason=effective_reason,
            unknown_reason_eligible=is_reason_eligible,
            ml_score_uncalibrated=None,
            rejection_category=(
                REJECTION_CATEGORY_INVALID_OR_UNAVAILABLE_ML
                if is_reason_eligible
                else _classify_static_rejection(effective_reason)
            ),
        )

    # Check ML probability validity
    if not is_valid_ml or ml_score_uncalibrated is None:
        return FusionResult(
            primary_label=TaxonomyLabel.UNKNOWN.value,
            source=FusionSource.UNKNOWN.value,
            decision_reason=FusionDecisionReason.ML_FALLBACK_INVALID_PROBABILITY.value,
            rationale=(
                f"Static analysis was inconclusive ({static_label or 'UNKNOWN'}). "
                f"ML prediction was rejected due to invalid/non-finite probability: "
                f"{validation_err_msg or ml_prob!r}."
            ),
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=ml_label,
            ml_probability=None,
            ml_probabilities=None,
            agreement=agreement,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used=mode_str,
            verification_status=VerificationStatus.ABSTAINED.value,
            abstention_reason=(
                f"ML prediction rejected: {validation_err_msg or 'invalid probability'}"
            ),
            unknown_reason=effective_reason,
            unknown_reason_eligible=is_reason_eligible,
            ml_score_uncalibrated=None,
            rejection_category=REJECTION_CATEGORY_INVALID_OR_UNAVAILABLE_ML,
        )

    # Check if ML predicted UNKNOWN
    if ml_label in (TaxonomyLabel.UNKNOWN.value, "UNKNOWN"):
        return FusionResult(
            primary_label=TaxonomyLabel.UNKNOWN.value,
            source=FusionSource.UNKNOWN.value,
            decision_reason=FusionDecisionReason.ML_PREDICTED_UNKNOWN.value,
            rationale="Both static analysis and ML model concluded UNKNOWN.",
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=ml_label,
            ml_probability=ml_score_uncalibrated,
            ml_probabilities=ml_probs,
            agreement=True if static_label == "UNKNOWN" else None,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used=mode_str,
            verification_status=VerificationStatus.ABSTAINED.value,
            abstention_reason="ML model predicted UNKNOWN.",
            unknown_reason=effective_reason,
            unknown_reason_eligible=is_reason_eligible,
            ml_score_uncalibrated=ml_score_uncalibrated,
            rejection_category=REJECTION_CATEGORY_ML_PREDICTED_UNKNOWN,
        )

    # =======================================================================
    # MODE S3: Naive Hybrid (Static-first, fallback on ANY non-supported)
    # =======================================================================
    if mode_str == "S3":
        return FusionResult(
            primary_label=ml_label,
            source=FusionSource.ML_ESTIMATE.value,
            decision_reason=FusionDecisionReason.ML_FALLBACK_ACCEPTED.value,
            rationale=(
                f"S3 Naive Hybrid: Static analysis inconclusive ({static_label or 'UNKNOWN'}). "
                f"Using un-gated ML fallback estimate {ml_label} with score "
                f"{ml_score_uncalibrated:.4f}."
            ),
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=ml_label,
            ml_probability=ml_score_uncalibrated,
            ml_probabilities=ml_probs,
            agreement=agreement,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used="S3",
            verification_status=VerificationStatus.ML_FALLBACK_UNVERIFIED.value,
            abstention_reason=None,
            unknown_reason=effective_reason,
            unknown_reason_eligible=is_reason_eligible,
            ml_score_uncalibrated=ml_score_uncalibrated,
            rejection_category=None,
        )

    # =======================================================================
    # MODE S4: Hybrid with Verification and Abstention
    # =======================================================================
    # Check reason-aware eligibility
    if is_reason_eligible is False:
        # Ineligible UNKNOWN reason: ML cannot provide fallback; remains UNKNOWN
        return FusionResult(
            primary_label=TaxonomyLabel.UNKNOWN.value,
            source=FusionSource.UNKNOWN.value,
            decision_reason=FusionDecisionReason.INELIGIBLE_UNKNOWN_REASON.value,
            rationale=(
                f"Static analysis abstained due to ineligible UNKNOWN reason '{effective_reason}'. "
                f"ML model predicted {ml_label} with score {ml_score_uncalibrated:.4f} "
                "(non-binding); retained UNKNOWN."
            ),
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=ml_label,
            ml_probability=ml_score_uncalibrated,
            ml_probabilities=ml_probs,
            agreement=agreement,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used="S4",
            verification_status=VerificationStatus.ABSTAINED.value,
            abstention_reason=(
                f"Static analysis abstained due to ineligible UNKNOWN reason '{effective_reason}'; "
                f"ML fallback prohibited."
            ),
            unknown_reason=effective_reason,
            unknown_reason_eligible=False,
            ml_score_uncalibrated=ml_score_uncalibrated,
            rejection_category=_classify_static_rejection(effective_reason),
        )

    # For eligible reasons (or unspecified reasons in legacy mode): check threshold
    if ml_score_uncalibrated >= ml_threshold:
        return FusionResult(
            primary_label=ml_label,
            source=FusionSource.ML_ESTIMATE.value,
            decision_reason=FusionDecisionReason.ML_FALLBACK_ACCEPTED.value,
            rationale=(
                f"Static analysis returned UNKNOWN (reason: {effective_reason or 'unspecified'}). "
                f"Model provided provisional ML estimate {ml_label} with score "
                f"{ml_score_uncalibrated:.4f} (meeting provisional threshold {ml_threshold:.2f})."
            ),
            static_label=static_label,
            static_exactness=static_exactness,
            ml_label=ml_label,
            ml_probability=ml_score_uncalibrated,
            ml_probabilities=ml_probs,
            agreement=agreement,
            threshold_used=ml_threshold,
            is_provisional_threshold=True,
            mode_used="S4",
            verification_status=VerificationStatus.ML_FALLBACK_UNVERIFIED.value,
            abstention_reason=None,
            unknown_reason=effective_reason,
            unknown_reason_eligible=is_reason_eligible,
            ml_score_uncalibrated=ml_score_uncalibrated,
            rejection_category=None,
        )

    # Eligible reason but score below threshold
    return FusionResult(
        primary_label=TaxonomyLabel.UNKNOWN.value,
        source=FusionSource.UNKNOWN.value,
        decision_reason=FusionDecisionReason.ML_FALLBACK_BELOW_THRESHOLD.value,
        rationale=(
            f"Static analysis was inconclusive ({static_label or 'UNKNOWN'}). "
            f"ML model predicted {ml_label} with score {ml_score_uncalibrated:.4f}, which does not "
            f"meet the provisional decision threshold of {ml_threshold:.2f}; retained UNKNOWN."
        ),
        static_label=static_label,
        static_exactness=static_exactness,
        ml_label=ml_label,
        ml_probability=ml_score_uncalibrated,
        ml_probabilities=ml_probs,
        agreement=agreement,
        threshold_used=ml_threshold,
        is_provisional_threshold=True,
        mode_used="S4",
        verification_status=VerificationStatus.ABSTAINED.value,
        abstention_reason=(
            f"ML score {ml_score_uncalibrated:.4f} is below decision threshold {ml_threshold:.2f}."
        ),
        unknown_reason=effective_reason,
        unknown_reason_eligible=is_reason_eligible,
        ml_score_uncalibrated=ml_score_uncalibrated,
        rejection_category=REJECTION_CATEGORY_LOW_CONFIDENCE,
    )
