"""
test_fusion_and_pipeline.py — Unit and integration tests for fusion, explanation, and pipeline.

Validates:
1. Fusion Gate Decision Rules:
   - Trusted static result overrides conflicting ML prediction.
   - Static UNKNOWN + high valid ML probability produces ML_ESTIMATE.
   - Static UNKNOWN + low ML probability remains UNKNOWN.
   - Missing ML model preserves static result or returns UNKNOWN as appropriate.
   - Invalid or non-finite probabilities cannot produce a trusted ML estimate.
   - Agreement and disagreement are recorded correctly.
2. Boundary Pipeline Integration:
   - Invalid source is rejected before IR construction.
   - Valid source produces a real static result.
   - Missing ML artifact does not prevent static analysis.
   - Unsupported constructs and diagnostics are surfaced honestly.
   - No space-complexity value is fabricated.
3. Explanation and Recommendations:
   - Recognized loops and nested loops reflect actual IR evidence.
   - Conservative recommendations only when evidence supports it.
   - Explicit limitations (absence of space complexity and parser timeout).
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from complexity_lens.complexity.models import (
    AnalysisResult,
    ComplexityClass,
    Exactness,
    RuleTraceEntry,
    TaxonomyLabel,
)
from complexity_lens.fusion import (
    FusionDecisionReason,
    FusionSource,
    MLPrediction,
    decide_fusion,
)
from complexity_lens.pipeline import analyze_source_pipeline, run_ml_inference, safe_load_model

# ---------------------------------------------------------------------------
# Test Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_static_result(
    taxonomy_label: str = "ON",
    exactness: str = Exactness.STRUCTURAL.value,
    selected_function: str = "solve",
    reason_codes: list[str] | None = None,
    rule_trace: list[RuleTraceEntry] | None = None,
) -> AnalysisResult:
    """Create a synthetic AnalysisResult for pure fusion tests."""
    c_class = ComplexityClass.linear() if taxonomy_label == "ON" else ComplexityClass.unknown()
    return AnalysisResult(
        selected_function=selected_function,
        selection_rationale="Synthetic test function",
        complexity_class=c_class,
        taxonomy_label=taxonomy_label,
        exactness=exactness,
        reason_codes=reason_codes or ["LINEAR_LOOP"],
        rule_trace=rule_trace or [],
    )


def _make_ml_prediction(
    predicted_label: str = "ON",
    probability: float = 0.85,
    model_id: str = "random_forest",
) -> MLPrediction:
    """Create a synthetic MLPrediction container for pure fusion tests."""
    classes = ["O1", "OLOGN", "ON", "ONLOGN", "ON2", "ON3", "OEXP", "UNKNOWN"]
    probs = {c: 0.0 for c in classes}
    if predicted_label in probs:
        probs[predicted_label] = probability
    return MLPrediction(
        predicted_label=predicted_label,
        probability=probability,
        probabilities=probs,
        model_id=model_id,
    )


# ===========================================================================
# 1. Fusion Gate Unit Tests
# ===========================================================================


def test_fusion_trusted_static_overrides_conflicting_ml() -> None:
    """Case 1: Trusted static result overrides a conflicting ML prediction."""
    static_res = _make_static_result(taxonomy_label="ON", exactness=Exactness.STRUCTURAL.value)
    ml_pred = _make_ml_prediction(predicted_label="ON2", probability=0.92)

    fused = decide_fusion(static_res, ml_pred, ml_threshold=0.55)

    assert fused.primary_label == "ON"
    assert fused.source == FusionSource.STATIC.value
    assert fused.decision_reason == FusionDecisionReason.STATIC_TRUSTED.value
    assert fused.static_label == "ON"
    assert fused.ml_label == "ON2"
    assert fused.ml_probability == pytest.approx(0.92)
    assert fused.agreement is False
    assert "verified STRUCTURAL bound of ON" in fused.rationale
    assert "disagrees" in fused.rationale


def test_fusion_static_unknown_plus_high_ml_produces_ml_estimate() -> None:
    """Case 2: Static UNKNOWN plus sufficiently high valid ML probability produces ML_ESTIMATE."""
    static_res = _make_static_result(
        taxonomy_label=TaxonomyLabel.UNKNOWN.value,
        exactness=Exactness.UNKNOWN.value,
        reason_codes=["UNKNOWN_BOUND"],
    )
    ml_pred = _make_ml_prediction(predicted_label="ONLOGN", probability=0.78)

    fused = decide_fusion(static_res, ml_pred, ml_threshold=0.55)

    assert fused.primary_label == "ONLOGN"
    assert fused.source == FusionSource.ML_ESTIMATE.value
    assert fused.decision_reason == FusionDecisionReason.ML_FALLBACK_ACCEPTED.value
    assert fused.static_label == "UNKNOWN"
    assert fused.ml_label == "ONLOGN"
    assert fused.ml_probability == pytest.approx(0.78)
    assert fused.threshold_used == 0.55
    assert fused.is_provisional_threshold is True
    assert "provisional ML estimate" in fused.rationale


def test_fusion_static_unknown_plus_low_ml_remains_unknown() -> None:
    """Case 3: Static UNKNOWN plus low ML probability remains UNKNOWN."""
    static_res = _make_static_result(
        taxonomy_label=TaxonomyLabel.UNKNOWN.value,
        exactness=Exactness.UNKNOWN.value,
    )
    ml_pred = _make_ml_prediction(predicted_label="ON", probability=0.42)

    fused = decide_fusion(static_res, ml_pred, ml_threshold=0.55)

    assert fused.primary_label == "UNKNOWN"
    assert fused.source == FusionSource.UNKNOWN.value
    assert fused.decision_reason == FusionDecisionReason.ML_FALLBACK_BELOW_THRESHOLD.value
    assert fused.static_label == "UNKNOWN"
    assert fused.ml_label == "ON"
    assert fused.ml_probability == pytest.approx(0.42)
    assert "does not meet the provisional decision threshold" in fused.rationale


def test_fusion_missing_ml_preserves_static_or_returns_unknown() -> None:
    """Case 4: Missing ML model preserves static result or returns UNKNOWN."""
    # When static is trusted and ML is None
    static_trusted = _make_static_result(taxonomy_label="O1", exactness=Exactness.UPPER_BOUND.value)
    fused_trusted = decide_fusion(static_trusted, None)
    assert fused_trusted.primary_label == "O1"
    assert fused_trusted.source == FusionSource.STATIC.value
    assert fused_trusted.decision_reason == FusionDecisionReason.STATIC_TRUSTED.value
    assert fused_trusted.agreement is None

    # When static is UNKNOWN and ML is None
    static_unknown = _make_static_result(
        taxonomy_label="UNKNOWN", exactness=Exactness.UNKNOWN.value
    )
    fused_unknown = decide_fusion(static_unknown, None)
    assert fused_unknown.primary_label == "UNKNOWN"
    assert fused_unknown.source == FusionSource.UNKNOWN.value
    assert fused_unknown.decision_reason == FusionDecisionReason.INCONCLUSIVE_NO_ML.value
    assert fused_unknown.agreement is None


def test_fusion_invalid_or_nonfinite_probabilities_rejected() -> None:
    """Case 5: Invalid or non-finite probabilities cannot produce a trusted ML estimate."""
    static_unknown = _make_static_result(
        taxonomy_label="UNKNOWN", exactness=Exactness.UNKNOWN.value
    )

    invalid_probabilities = [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.5,
        1.5,
    ]

    for bad_prob in invalid_probabilities:
        ml_bad = _make_ml_prediction(predicted_label="ON", probability=bad_prob)
        fused = decide_fusion(static_unknown, ml_bad)
        assert fused.primary_label == "UNKNOWN"
        assert fused.source == FusionSource.UNKNOWN.value
        assert fused.decision_reason == FusionDecisionReason.ML_FALLBACK_INVALID_PROBABILITY.value
        assert fused.ml_probability is None


def test_fusion_agreement_representation() -> None:
    """Case 6: Agreement and disagreement are represented correctly."""
    static_linear = _make_static_result(taxonomy_label="ON", exactness=Exactness.STRUCTURAL.value)

    # Agreement
    ml_agree = _make_ml_prediction(predicted_label="ON", probability=0.90)
    res_agree = decide_fusion(static_linear, ml_agree)
    assert res_agree.agreement is True
    assert "agrees" in res_agree.rationale

    # Disagreement
    ml_disagree = _make_ml_prediction(predicted_label="ON2", probability=0.88)
    res_disagree = decide_fusion(static_linear, ml_disagree)
    assert res_disagree.agreement is False
    assert "disagrees" in res_disagree.rationale

    # Static UNKNOWN + ML UNKNOWN
    static_unknown = _make_static_result(
        taxonomy_label="UNKNOWN", exactness=Exactness.UNKNOWN.value
    )
    ml_unknown = _make_ml_prediction(predicted_label="UNKNOWN", probability=0.60)
    res_both_unknown = decide_fusion(static_unknown, ml_unknown)
    assert res_both_unknown.agreement is True
    assert res_both_unknown.decision_reason == FusionDecisionReason.ML_PREDICTED_UNKNOWN.value


# ===========================================================================
# 2. Pipeline Integration Tests
# ===========================================================================


def test_pipeline_invalid_source_rejected_before_ir() -> None:
    """Case 1: Invalid source is rejected before IR construction."""
    # Disallowed quoted include: must fail validation
    invalid_src = '#include "secret.h"\nint main() { return 0; }\n'

    # Ensure build_ir_from_source is NEVER called
    with patch("complexity_lens.pipeline.build_ir_from_source") as mock_build_ir:
        result = analyze_source_pipeline(invalid_src)
        assert not mock_build_ir.called

    assert result.success is False
    assert result.validation_error is not None
    assert (
        "directive" in result.validation_error.lower()
        or "include" in result.validation_error.lower()
    )
    assert result.program_ir is None
    assert result.static_result is None
    assert result.feature_vector is None


def test_pipeline_valid_source_produces_real_static_result() -> None:
    """Case 2: Valid source produces a real static result."""
    valid_src = (
        "int compute(int n) {\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < n; ++i) {\n"
        "        s += i;\n"
        "    }\n"
        "    return s;\n"
        "}\n"
    )

    result = analyze_source_pipeline(valid_src, function_name="compute", enable_ml=False)

    assert result.success is True
    assert result.validation_error is None
    assert result.program_ir is not None
    assert result.static_result is not None
    assert result.static_result.taxonomy_label == "ON"
    assert result.static_result.exactness == Exactness.STRUCTURAL.value
    assert result.fusion_result is not None
    assert result.fusion_result.primary_label == "ON"
    assert result.fusion_result.source == FusionSource.STATIC.value
    assert result.feature_vector is not None
    assert result.feature_vector["n_loops"] == 1
    assert result.feature_vector["max_loop_depth"] == 1


def test_pipeline_missing_ml_artifact_does_not_prevent_static_analysis(tmp_path: Path) -> None:
    """Case 3: Missing ML artifact does not prevent static analysis."""
    src = (
        "int main() {\n"
        "    int sum = 0;\n"
        "    for (int i = 0; i < 10; ++i) sum += i;\n"
        "    return sum;\n"
        "}\n"
    )

    # Point to an empty directory containing no artifacts
    empty_artifacts_dir = tmp_path / "no_models"
    empty_artifacts_dir.mkdir()

    result = analyze_source_pipeline(
        src,
        artifacts_dir=empty_artifacts_dir,
        enable_ml=True,
    )

    assert result.success is True
    assert result.static_result is not None
    assert result.static_result.taxonomy_label == "O1"
    assert result.ml_prediction is None
    assert result.ml_error is not None
    assert "unavailable" in result.ml_error.lower()
    assert result.fusion_result is not None
    assert result.fusion_result.primary_label == "O1"
    assert result.fusion_result.source == FusionSource.STATIC.value


def test_pipeline_unsupported_constructs_surfaced_honestly() -> None:
    """Case 4: Unsupported constructs and diagnostics are surfaced honestly."""
    src_with_asm = 'int main() {\n    __asm__("nop");\n    return 0;\n}\n'

    result = analyze_source_pipeline(src_with_asm, enable_ml=False)

    assert result.success is True
    assert result.static_result is not None
    assert result.static_result.taxonomy_label == "UNKNOWN"
    assert result.explanation is not None
    assert len(result.explanation.unsupported_constructs) > 0
    assert result.explanation.unknown_reason is not None
    assert "unsupported" in result.explanation.unknown_reason.lower()


def test_pipeline_no_space_complexity_fabricated() -> None:
    """Case 5: No space-complexity value is fabricated across all pipeline data structures."""
    src = "int main() { return 0; }\n"
    result = analyze_source_pipeline(src, enable_ml=False)

    # PipelineResult attributes
    assert not hasattr(result, "space_complexity")
    assert not hasattr(result, "auxiliary_space")

    # Static AnalysisResult
    assert not hasattr(result.static_result, "space_complexity")
    assert not hasattr(result.static_result, "auxiliary_space")

    # FusionResult
    assert not hasattr(result.fusion_result, "space_complexity")

    # ExplanationReport explicitly acknowledges absence
    assert result.explanation is not None
    limitations_text = " ".join(result.explanation.limitations).lower()
    assert "space complexity is not" in limitations_text


def test_pipeline_with_verified_dev_model_artifact() -> None:
    """Integration check: loading real development artifact safely and running inference."""
    # Check if existing development model exists in artifacts/models
    rf_artifact = Path("artifacts/models/random_forest_dev_fit.joblib")
    rf_metadata = Path("artifacts/models/random_forest_metadata.json")

    if not rf_artifact.exists() or not rf_metadata.exists():
        pytest.skip("Development model artifact not present on disk.")

    loaded = safe_load_model(model_id="random_forest", artifacts_dir="artifacts/models")
    assert loaded is not None
    model, metadata = loaded
    assert metadata["model_id"] == "random_forest"
    assert len(metadata["canonical_feature_names"]) == 26
    assert len(metadata["canonical_class_order"]) == 8

    src = (
        "int linear_scan(int n) {\n"
        "    int sum = 0;\n"
        "    for (int i = 0; i < n; ++i) sum += i;\n"
        "    return sum;\n"
        "}\n"
    )

    res = analyze_source_pipeline(
        src,
        function_name="linear_scan",
        model=model,
        model_id="random_forest",
        enable_ml=True,
    )

    assert res.success is True
    assert res.static_result is not None
    assert res.static_result.taxonomy_label == "ON"
    assert res.ml_prediction is not None
    assert res.ml_prediction.predicted_label in metadata["canonical_class_order"]
    assert math.isfinite(res.ml_prediction.probability)
    assert len(res.ml_prediction.probabilities) == 8
    # Static is trusted ON, so fusion must preserve ON
    assert res.fusion_result is not None
    assert res.fusion_result.primary_label == "ON"
    assert res.fusion_result.source == FusionSource.STATIC.value


# ===========================================================================
# 3. Explanation and Recommendation Unit Tests
# ===========================================================================


def test_explanation_loop_bound_and_nesting_evidence() -> None:
    """Verify recognized loops and nesting reflect actual IR evidence."""
    src = (
        "int nested_loops(int n) {\n"
        "    int sum = 0;\n"
        "    for (int i = 0; i < n; ++i) {\n"
        "        for (int j = 0; j < n; ++j) {\n"
        "            sum += i * j;\n"
        "        }\n"
        "    }\n"
        "    return sum;\n"
        "}\n"
    )

    res = analyze_source_pipeline(src, function_name="nested_loops", enable_ml=False)
    assert res.success is True
    assert res.explanation is not None

    detected_loops = [c for c in res.explanation.detected_constructs if c.kind == "loop"]
    assert len(detected_loops) == 2
    # Check depth / nesting details
    outer = detected_loops[0]
    inner = detected_loops[1]
    assert outer.details["depth"] == 1
    assert inner.details["depth"] == 2
    assert inner.details["is_nested"] is True

    # Rule reasoning contains nested loop derivation
    assert any("nested loop" in r.lower() for r in res.explanation.evidence_reasoning)


def test_explanation_recommendation_evidence_supported_and_empty_otherwise() -> None:
    """Verify recommendations are empty by default, and generated only when evidence exists."""
    # Source with no special pattern: recommendations should be empty
    plain_src = (
        "int simple(int n) {\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < n; ++i) s += i;\n"
        "    return s;\n"
        "}\n"
    )
    plain_res = analyze_source_pipeline(plain_src, function_name="simple", enable_ml=False)
    assert plain_res.success is True
    assert plain_res.explanation is not None
    assert plain_res.explanation.recommendations == []

    # Source with linear search inside loop
    search_src = (
        "int find(int x);\n"
        "int find_in_loop(int n) {\n"
        "    int matches = 0;\n"
        "    for (int i = 0; i < n; ++i) {\n"
        "        matches += find(i);\n"
        "    }\n"
        "    return matches;\n"
        "}\n"
    )
    search_res = analyze_source_pipeline(search_src, function_name="find_in_loop", enable_ml=False)
    assert search_res.success is True
    assert search_res.explanation is not None
    assert len(search_res.explanation.recommendations) > 0
    assert any("unordered_set" in rec for rec in search_res.explanation.recommendations)


def test_pipeline_and_inference_model_tuple_unpacking_regression():
    """Regression test: verify pipeline and run_ml_inference unpack model tuples safely.

    Prevents runtime error: ML inference failed: 'tuple' object has no attribute 'predict'.
    Verifies both Random Forest and XGBoost with both tuple and unpacked model inputs.
    """
    sample_src = (
        "int loop(int n) {\n"
        "    int sum = 0;\n"
        "    for (int i = 0; i < n; ++i) sum += i;\n"
        "    return sum;\n"
        "}\n"
    )

    # 1. Random Forest as tuple
    rf_loaded = safe_load_model("random_forest")
    assert isinstance(rf_loaded, tuple)
    res_rf_tuple = analyze_source_pipeline(sample_src, model=rf_loaded, model_id="random_forest")
    assert res_rf_tuple.success is True
    assert res_rf_tuple.ml_error is None
    assert res_rf_tuple.ml_prediction is not None
    assert res_rf_tuple.ml_prediction.predicted_label == "ON"

    # 2. Random Forest as unpacked model
    res_rf_unpacked = analyze_source_pipeline(
        sample_src, model=rf_loaded[0], model_id="random_forest"
    )
    assert res_rf_unpacked.success is True
    assert res_rf_unpacked.ml_error is None
    assert res_rf_unpacked.ml_prediction is not None
    assert res_rf_unpacked.ml_prediction.predicted_label == "ON"

    # 3. XGBoost as tuple
    xgb_loaded = safe_load_model("xgboost")
    assert isinstance(xgb_loaded, tuple)
    res_xgb_tuple = analyze_source_pipeline(sample_src, model=xgb_loaded, model_id="xgboost")
    assert res_xgb_tuple.success is True
    assert res_xgb_tuple.ml_error is None
    assert res_xgb_tuple.ml_prediction is not None
    assert res_xgb_tuple.ml_prediction.predicted_label == "ON"

    # 4. XGBoost as unpacked model
    res_xgb_unpacked = analyze_source_pipeline(sample_src, model=xgb_loaded[0], model_id="xgboost")
    assert res_xgb_unpacked.success is True
    assert res_xgb_unpacked.ml_error is None
    assert res_xgb_unpacked.ml_prediction is not None
    assert res_xgb_unpacked.ml_prediction.predicted_label == "ON"

    # 5. Direct run_ml_inference call with tuple
    assert res_rf_tuple.feature_vector is not None
    pred_direct = run_ml_inference(rf_loaded, res_rf_tuple.feature_vector, model_id="random_forest")
    assert pred_direct.predicted_label == "ON"
    assert pred_direct.probability > 0.0


def test_loop_bound_symbolic_n_regressions() -> None:
    """Regression tests verifying symbolic input-scale handling and loop bounds.

    Covers:
    1. Exact linear loop program in main() with local n = 100 -> O(n).
    2. Simple loop with i < n, i++ -> O(n).
    3. Sequential loops bounded by n -> O(n), not O(n^2).
    4. Two nested loops independently bounded by n -> O(n^2).
    5. Loop with recognized halving update -> O(log n).
    6. Non-symbolic constant bound (literal 100, local limit = 10) -> O(1).
    """
    model_bundle = safe_load_model("random_forest")
    model = model_bundle[0] if model_bundle else None

    # Case 1: Exact linear loop program from user defect report
    exact_prog = (
        "int main() {\n"
        "    int n = 100;\n"
        "\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        int x = i;\n"
        "    }\n"
        "\n"
        "    return 0;\n"
        "}\n"
    )
    res_exact = analyze_source_pipeline(exact_prog, model=model)
    assert res_exact.success is True
    assert res_exact.static_result is not None
    assert res_exact.static_result.taxonomy_label == "ON"
    assert res_exact.fusion_result.primary_label == "ON"
    assert res_exact.fusion_result.source == FusionSource.STATIC
    assert "RULE_LINEAR_LOOP" in [r.rule_id for r in res_exact.static_result.rule_trace]

    # Case 2: Simple loop with i < n, i++
    simple_loop_func = (
        "void solve(int n) {\n    for (int i = 0; i < n; i++) {\n        int x = i;\n    }\n}\n"
    )
    res_simple = analyze_source_pipeline(simple_loop_func, model=model)
    assert res_simple.success is True
    assert res_simple.static_result.taxonomy_label == "ON"
    assert res_simple.fusion_result.primary_label == "ON"
    assert "RULE_LINEAR_LOOP" in [r.rule_id for r in res_simple.static_result.rule_trace]

    # Case 3: Two sequential loops bounded by n -> O(n), not O(n^2)
    seq_loops_func = (
        "void solve(int n) {\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        int x = i;\n"
        "    }\n"
        "    for (int j = 0; j < n; j++) {\n"
        "        int y = j;\n"
        "    }\n"
        "}\n"
    )
    res_seq = analyze_source_pipeline(seq_loops_func, model=model)
    assert res_seq.success is True
    assert res_seq.static_result.taxonomy_label == "ON"
    assert res_seq.fusion_result.primary_label == "ON"
    assert "RULE_SEQUENTIAL_COMPOSITION" in [r.rule_id for r in res_seq.static_result.rule_trace]

    # Sequential loops in main() with n = 100 also compose to O(n)
    seq_loops_main = (
        "int main() {\n"
        "    int n = 100;\n"
        "    for (int i = 0; i < n; i++) {}\n"
        "    for (int j = 0; j < n; j++) {}\n"
        "    return 0;\n"
        "}\n"
    )
    res_seq_main = analyze_source_pipeline(seq_loops_main, model=model)
    assert res_seq_main.success is True
    assert res_seq_main.static_result.taxonomy_label == "ON"
    assert res_seq_main.fusion_result.primary_label == "ON"

    # Case 4: Two nested loops independently bounded by n -> O(n^2)
    nested_loops_func = (
        "void solve(int n) {\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        for (int j = 0; j < n; j++) {\n"
        "            int y = j;\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    res_nested = analyze_source_pipeline(nested_loops_func, model=model)
    assert res_nested.success is True
    assert res_nested.static_result.taxonomy_label == "ON2"
    assert res_nested.fusion_result.primary_label == "ON2"
    assert "RULE_NESTED_LOOP" in [r.rule_id for r in res_nested.static_result.rule_trace]

    # Nested loops in main() with n = 100 also compose to O(n^2)
    nested_loops_main = (
        "int main() {\n"
        "    int n = 100;\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        for (int j = 0; j < n; j++) {}\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    res_nested_main = analyze_source_pipeline(nested_loops_main, model=model)
    assert res_nested_main.success is True
    assert res_nested_main.static_result.taxonomy_label == "ON2"
    assert res_nested_main.fusion_result.primary_label == "ON2"

    # Case 5: Loop with a recognized halving update -> O(log n)
    halving_func = "void solve(int n) {\n    while (n > 0) {\n        n /= 2;\n    }\n}\n"
    res_halving = analyze_source_pipeline(halving_func, model=model)
    assert res_halving.success is True
    assert res_halving.static_result.taxonomy_label == "OLOGN"
    assert res_halving.fusion_result.primary_label == "OLOGN"
    assert "RULE_LOGARITHMIC_LOOP" in [r.rule_id for r in res_halving.static_result.rule_trace]

    # Halving in main() with n = 100
    halving_main = (
        "int main() {\n"
        "    int n = 100;\n"
        "    while (n > 0) {\n"
        "        n /= 2;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    res_halving_main = analyze_source_pipeline(halving_main, model=model)
    assert res_halving_main.success is True
    assert res_halving_main.static_result.taxonomy_label == "OLOGN"
    assert res_halving_main.fusion_result.primary_label == "OLOGN"

    # Preserved constant bounds: literal constants and local limits stay O(1)
    literal_const_code = (
        "int main() {\n"
        "    int sum = 0;\n"
        "    for (int i = 0; i < 100; ++i) {\n"
        "        sum += i;\n"
        "    }\n"
        "    return sum;\n"
        "}\n"
    )
    res_lit = analyze_source_pipeline(literal_const_code, model=model)
    assert res_lit.success is True
    assert res_lit.static_result.taxonomy_label == "O1"
    assert res_lit.fusion_result.primary_label == "O1"

    local_limit_code = (
        "int main() {\n"
        "    int limit = 10;\n"
        "    int sum = 0;\n"
        "    for (int i = 0; i < limit; ++i) {\n"
        "        sum += i;\n"
        "    }\n"
        "    return sum;\n"
        "}\n"
    )
    res_limit = analyze_source_pipeline(local_limit_code, model=model)
    assert res_limit.success is True
    assert res_limit.static_result.taxonomy_label == "O1"
    assert res_limit.fusion_result.primary_label == "O1"
