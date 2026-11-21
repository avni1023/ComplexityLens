"""
pipeline.py — Unified boundary analysis pipeline for ComplexityLens.

Provides the single, authoritative entry point for user submissions and UI:
1. Validates the source at the application boundary using `validate_source()`.
2. Builds `ProgramIR` with libclang only if validation succeeds.
3. Executes static time-complexity analysis via `ComplexityEngine`.
4. Extracts authoritative 26-feature vector from valid IR.
5. Performs optional ML inference using pre-verified development artifacts.
6. Combines static bounds and ML estimates through the deterministic fusion gate.
7. Produces an evidence-backed explanation and recommendation report.

Safety & Architectural Guarantees:
- Never compiles or executes submitted C++ source code.
- Strict input validation prevents parser abuse before libclang invocation.
- Optional ML failures or missing artifacts never prevent static analysis.
- Parser execution is in-process; no explicit timeout is enforced.
- Space complexity is explicitly NOT computed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from complexity_lens.complexity.engine import analyze_complexity
from complexity_lens.complexity.models import AnalysisResult
from complexity_lens.evaluation.candidates import (
    DEFAULT_ARTIFACTS_DIR,
    ArtifactIntegrityError,
    ArtifactSecurityError,
    load_model_artifact,
)
from complexity_lens.evaluation.data_loader import CANONICAL_CLASS_ORDER
from complexity_lens.explanation import ExplanationReport, generate_explanation
from complexity_lens.features.extractor import extract_features
from complexity_lens.features.schema import FeatureVector
from complexity_lens.fusion import (
    DEFAULT_ML_THRESHOLD,
    FusionResult,
    MLPrediction,
    decide_fusion,
    validate_probability_vector,
)
from complexity_lens.ir.builder import build_ir_from_source
from complexity_lens.ir.model import ProgramIR
from complexity_lens.validation import ValidationError, validate_source


@dataclasses.dataclass(frozen=True)
class PipelineResult:
    """Complete, end-to-end result returned by the ComplexityLens analysis pipeline."""

    success: bool
    validation_error: str | None
    validation_error_line: int | None
    program_ir: ProgramIR | None
    static_result: AnalysisResult | None
    feature_vector: FeatureVector | None
    ml_prediction: MLPrediction | None
    ml_error: str | None
    fusion_result: FusionResult | None
    explanation: ExplanationReport | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "feature_vector": self.feature_vector.to_dict() if self.feature_vector else None,
            "fusion_result": self.fusion_result.to_dict() if self.fusion_result else None,
            "ml_error": self.ml_error,
            "ml_prediction": self.ml_prediction.to_dict() if self.ml_prediction else None,
            "static_result": self.static_result.to_dict() if self.static_result else None,
            "success": self.success,
            "validation_error": self.validation_error,
            "validation_error_line": self.validation_error_line,
        }


def safe_load_model(
    model_id: str = "random_forest",
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> tuple[Any, dict[str, Any]] | None:
    """Safely load and cryptographically verify a persisted model artifact.

    Parameters
    ----------
    model_id:
        Identifier of model to load: 'random_forest' (default) or 'xgboost'.
    artifacts_dir:
        Directory containing the model artifact and metadata JSON.

    Returns
    -------
    tuple[Any, dict[str, Any]] | None
        Fitted model and verified metadata dict, or None if unavailable/corrupted.
    """
    artifacts_root = Path(artifacts_dir)
    if model_id == "random_forest":
        artifact_path = artifacts_root / "random_forest_dev_fit.joblib"
        metadata_path = artifacts_root / "random_forest_metadata.json"
    elif model_id == "xgboost":
        artifact_path = artifacts_root / "xgboost_dev_fit.json"
        metadata_path = artifacts_root / "xgboost_metadata.json"
    else:
        return None

    if not artifact_path.exists() or not metadata_path.exists():
        return None

    try:
        return load_model_artifact(
            model_id=model_id,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            allowed_dir=artifacts_root,
        )
    except (ArtifactIntegrityError, ArtifactSecurityError, Exception):
        return None


def run_ml_inference(
    model: Any,
    feature_vector: FeatureVector,
    model_id: str = "random_forest",
) -> MLPrediction:
    """Execute ML inference using a verified fitted model and 26-feature vector.

    Guarantees:
    - Strictly preserves canonical 8-class probability mapping.
    - Validates probability finiteness and bounds.
    - Does NOT label probabilities as calibrated probabilities.
    """
    fitted_model = model[0] if isinstance(model, tuple) else model
    X = [feature_vector.to_list()]
    preds = fitted_model.predict(X)
    probs_matrix = fitted_model.predict_proba(X)

    expected_shape = (1, len(CANONICAL_CLASS_ORDER))
    if not isinstance(probs_matrix, np.ndarray) or probs_matrix.shape != expected_shape:
        raise ValueError(
            f"Expected probability matrix of shape (1, {len(CANONICAL_CLASS_ORDER)}), "
            f"got {getattr(probs_matrix, 'shape', None)}"
        )

    # Validate that all probabilities are finite numbers
    if not np.all(np.isfinite(probs_matrix)):
        raise ValueError("Model predicted non-finite probability values.")

    predicted_label = str(preds[0])
    raw_probs = probs_matrix[0]

    # Validate complete probability vector at adapter boundary
    prob_dict = validate_probability_vector(raw_probs, tolerance=1e-4)

    if predicted_label in CANONICAL_CLASS_ORDER:
        top_prob = float(prob_dict[predicted_label])
    else:
        top_prob = float(max(prob_dict.values()))

    return MLPrediction(
        predicted_label=predicted_label,
        probability=top_prob,
        probabilities=prob_dict,
        model_id=model_id,
    )


def analyze_source_pipeline(
    source: str,
    *,
    function_name: str | None = None,
    filename: str = "submission.cpp",
    cost_model: str = "expected_hash",
    model: Any | None = None,
    model_id: str = "random_forest",
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    ml_threshold: float = DEFAULT_ML_THRESHOLD,
    enable_ml: bool = True,
) -> PipelineResult:
    """Authoritative, reusable boundary analysis pipeline for C++ complexity analysis.

    Execution order:
    1. Validate source (size, lines, includes, braces, NUL characters).
    2. Build ProgramIR only if validation succeeds.
    3. Run static time-complexity analysis.
    4. Extract 26-feature vector only from valid IR.
    5. Run ML inference only if model is verified and available.
    6. Pass static result and ML prediction to fusion gate.
    7. Generate evidence-grounded explanation and conservative recommendations.

    Parameters
    ----------
    source:
        Unvalidated C++ source string submitted by user.
    function_name:
        Optional target function name. If None, resolves via standard entry precedence.
    filename:
        Virtual filename used by the parser.
    cost_model:
        Hash cost model: 'expected_hash' (default) or 'worst_case_hash'.
    model:
        Pre-loaded, verified model instance. If None and enable_ml=True, loads
        from artifacts_dir.
    model_id:
        Model identifier ('random_forest' or 'xgboost').
    artifacts_dir:
        Directory storing development model artifacts.
    ml_threshold:
        Provisional decision threshold for ML fallback acceptance (default 0.55).
    enable_ml:
        Whether to attempt ML inference fallback.

    Returns
    -------
    PipelineResult
        Full structured analysis result containing static analysis, ML estimates,
        fusion gate decision, and evidence-backed explanations.
    """
    # -----------------------------------------------------------------------
    # Step 1: Validate source at application boundary
    # -----------------------------------------------------------------------
    try:
        validated_source = validate_source(source)
    except ValidationError as ve:
        # Source rejected before parser invocation
        return PipelineResult(
            success=False,
            validation_error=str(ve),
            validation_error_line=ve.line,
            program_ir=None,
            static_result=None,
            feature_vector=None,
            ml_prediction=None,
            ml_error=None,
            fusion_result=None,
            explanation=None,
        )

    # -----------------------------------------------------------------------
    # Step 2: Build IR only if validation succeeded
    # -----------------------------------------------------------------------
    try:
        program_ir = build_ir_from_source(validated_source, filename=filename)
    except Exception as e:
        return PipelineResult(
            success=False,
            validation_error=f"Parser failure during IR construction: {e}",
            validation_error_line=None,
            program_ir=None,
            static_result=None,
            feature_vector=None,
            ml_prediction=None,
            ml_error=None,
            fusion_result=None,
            explanation=None,
        )

    # -----------------------------------------------------------------------
    # Step 3: Run static time-complexity analysis
    # -----------------------------------------------------------------------
    try:
        static_result = analyze_complexity(
            program_ir,
            function_name=function_name,
            cost_model=cost_model,
        )
    except Exception:
        static_result = None

    # -----------------------------------------------------------------------
    # Step 4: Extract features only when IR is available
    # -----------------------------------------------------------------------
    feature_vector: FeatureVector | None = None
    feature_error: str | None = None
    try:
        feature_vector = extract_features(program_ir, entry_function_name=function_name)
    except Exception as e:
        feature_error = f"Feature extraction failed: {e}"

    # -----------------------------------------------------------------------
    # Step 5: Run ML inference only when verified model is available
    # -----------------------------------------------------------------------
    ml_prediction: MLPrediction | None = None
    ml_error: str | None = None

    if enable_ml and feature_vector is not None:
        active_model = model[0] if isinstance(model, tuple) else model
        if active_model is None:
            loaded_tuple = safe_load_model(model_id=model_id, artifacts_dir=artifacts_dir)
            if loaded_tuple is not None:
                active_model = loaded_tuple[0]
            else:
                ml_error = f"Verified ML artifact '{model_id}' unavailable in '{artifacts_dir}'."

        if active_model is not None:
            try:
                ml_prediction = run_ml_inference(
                    active_model,
                    feature_vector,
                    model_id=model_id,
                )
            except Exception as e:
                ml_error = f"ML inference failed: {e}"
    elif not enable_ml:
        ml_error = "ML inference disabled by configuration."
    elif feature_error:
        ml_error = feature_error

    # -----------------------------------------------------------------------
    # Step 6: Pass results to fusion gate
    # -----------------------------------------------------------------------
    fusion_result = decide_fusion(
        static_result,
        ml_prediction,
        ml_threshold=ml_threshold,
    )

    # -----------------------------------------------------------------------
    # Step 7: Generate evidence-backed explanation
    # -----------------------------------------------------------------------
    explanation = generate_explanation(
        program=program_ir,
        analysis_result=static_result,
        fusion_result=fusion_result,
    )

    return PipelineResult(
        success=True,
        validation_error=None,
        validation_error_line=None,
        program_ir=program_ir,
        static_result=static_result,
        feature_vector=feature_vector,
        ml_prediction=ml_prediction,
        ml_error=ml_error,
        fusion_result=fusion_result,
        explanation=explanation,
    )
