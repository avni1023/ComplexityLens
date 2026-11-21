"""Tests for ComplexityLens Streamlit Web Interface (app.py).

Verifies UI structure, demo snippets, safe rendering, static-analysis safety
regression (no subprocess/compilation), model limitation notes, and evaluation metrics.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import app
from complexity_lens.fusion import FusionSource
from complexity_lens.pipeline import analyze_source_pipeline


def test_app_imports_and_constants():
    """Verify app imports cleanly and core mappings are well-formed."""
    assert hasattr(app, "DEMO_SNIPPETS")
    assert hasattr(app, "LABEL_MATH_NOTATION")
    assert hasattr(app, "MODEL_LIMITATION_NOTE")

    canonical_classes = {
        "O1": "O(1)",
        "OLOGN": "O(log n)",
        "ON": "O(n)",
        "ONLOGN": "O(n log n)",
        "ON2": "O(n²)",
        "ON3": "O(n³)",
        "OEXP": "O(2ⁿ)",
        "UNKNOWN": "Unknown",
    }
    for cls_name, expected_disp in canonical_classes.items():
        assert cls_name in app.LABEL_MATH_NOTATION
        assert app.LABEL_MATH_NOTATION[cls_name] == expected_disp


def test_demo_snippets_integrity_and_execution():
    """Verify all built-in demonstration snippets parse and analyze safely."""
    snippets = app.DEMO_SNIPPETS
    assert len(snippets) >= 5

    required_keys = {"description", "expected", "code"}
    for name, item in snippets.items():
        for key in required_keys:
            assert key in item, f"Snippet {name} missing key {key}"
        assert len(item["code"].strip()) > 0

    # Test Linear snippet
    linear_res = analyze_source_pipeline(snippets["Linear Loop O(n)"]["code"])
    assert linear_res.success is True
    assert linear_res.fusion_result is not None
    assert linear_res.fusion_result.primary_label == "ON"
    assert linear_res.fusion_result.source == FusionSource.STATIC

    # Test Constant snippet
    const_res = analyze_source_pipeline(snippets["Constant Time O(1)"]["code"])
    assert const_res.success is True
    assert const_res.fusion_result is not None
    assert const_res.fusion_result.primary_label == "O1"
    assert const_res.fusion_result.source == FusionSource.STATIC

    # Test Nested snippet
    nested_res = analyze_source_pipeline(snippets["Nested Loops O(n²)"]["code"])
    assert nested_res.success is True
    assert nested_res.fusion_result is not None
    assert nested_res.fusion_result.primary_label == "ON2"
    assert nested_res.fusion_result.source == FusionSource.STATIC

    # Test Logarithmic snippet
    log_res = analyze_source_pipeline(snippets["Logarithmic Loop O(log n)"]["code"])
    assert log_res.success is True
    assert log_res.fusion_result is not None
    assert log_res.fusion_result.primary_label == "OLOGN"
    assert log_res.fusion_result.source == FusionSource.STATIC

    # Test Collatz inconclusive snippet
    collatz_res = analyze_source_pipeline(snippets["Inconclusive Collatz Loop"]["code"])
    assert collatz_res.success is True
    assert collatz_res.static_result is not None
    assert collatz_res.static_result.taxonomy_label == "UNKNOWN"
    assert collatz_res.fusion_result is not None
    assert collatz_res.fusion_result.primary_label == "UNKNOWN"
    assert collatz_res.fusion_result.source == FusionSource.UNKNOWN


def test_empty_or_whitespace_input():
    """Verify that empty or whitespace-only code fails validation safely."""
    res_empty = analyze_source_pipeline("")
    assert res_empty.success is False
    assert res_empty.validation_error is not None

    res_ws = analyze_source_pipeline("   \n\t  ")
    assert res_ws.success is False
    assert res_ws.validation_error is not None


def test_disclosures_present_in_app_source():
    """Verify that critical technical disclosures and limitation notes are present."""
    app_source = inspect.getsource(app)

    # Must disclose lack of auxiliary space complexity
    assert "Space Complexity" in app_source
    assert "NOT IMPLEMENTED" in app_source

    # Must disclose no parser timeout
    assert "timeout" in app_source.lower()

    # Must disclose uncalibrated ML probabilities
    assert "not calibrated" in app_source.lower()

    # Must include mandatory model limitation note
    assert "A prediction for an individual input is not evidence of model accuracy." in app_source


def test_safe_rendering_compliance():
    """Verify that app.py never interpolates user-derived or dynamic text into unsafe HTML."""
    app_source = inspect.getsource(app)

    # Count occurrences of unsafe_allow_html
    unsafe_calls = [
        line.strip() for line in app_source.splitlines() if "unsafe_allow_html=True" in line
    ]

    # Only static developer CSS should use unsafe_allow_html
    assert len(unsafe_calls) <= 1, "Only inject_custom_styles may use unsafe_allow_html"

    # Ensure no f-string markdown with unsafe_allow_html
    for line in app_source.splitlines():
        if "unsafe_allow_html=True" in line:
            # Check surrounding lines in inject_custom_styles
            assert "f'''" not in line and 'f"""' not in line


def test_static_analysis_safety_regression_no_compilation():
    """Safety regression test: verifying submitted C++ is NEVER compiled or executed.

    Asserts that analyze_source_pipeline does not invoke subprocess.run, Popen, call,
    or os.system during analysis.
    """
    mock_run = MagicMock()
    mock_popen = MagicMock()
    mock_call = MagicMock()
    mock_system = MagicMock()

    with (
        patch.object(subprocess, "run", mock_run),
        patch.object(subprocess, "Popen", mock_popen),
        patch.object(subprocess, "call", mock_call),
        patch.object(os, "system", mock_system),
    ):
        for _name, item in app.DEMO_SNIPPETS.items():
            result = analyze_source_pipeline(item["code"])
            assert result.success is True

    # Assert none of the execution primitives were called
    mock_run.assert_not_called()
    mock_popen.assert_not_called()
    mock_call.assert_not_called()
    mock_system.assert_not_called()


def test_ml_disabled_pipeline_execution():
    """Verify UI backend runs cleanly when ML is explicitly disabled."""
    code = app.DEMO_SNIPPETS["Linear Loop O(n)"]["code"]
    res = analyze_source_pipeline(code, enable_ml=False)
    assert res.success is True
    assert res.ml_prediction is None
    assert res.static_result is not None
    assert res.static_result.taxonomy_label == "ON"
    assert res.fusion_result is not None
    assert res.fusion_result.source == FusionSource.STATIC


def test_cached_model_loader():
    """Verify get_cached_model loads the Random Forest model."""
    loaded = app.get_cached_model("random_forest")
    assert loaded is not None
    model_obj = loaded[0] if isinstance(loaded, tuple) else loaded
    assert hasattr(model_obj, "predict_proba")


def test_load_dev_evaluation_metrics():
    """Verify development CV evaluation metrics are loaded honestly without fabricating."""
    metrics = app.load_dev_evaluation_metrics()
    assert metrics is not None
    assert "random_forest" in metrics
    assert "xgboost" in metrics

    rf = metrics["random_forest"]
    assert rf["metric_name"] == "pooled_oof_macro_f1"
    assert "selection-biased" in rf["score_label"].lower()
    assert 0.50 <= rf["macro_f1"] <= 0.60
    assert 0.50 <= rf["accuracy"] <= 0.60


def test_missing_ml_artifacts_simulation():
    """Test I: Verify static analysis continues when ML artifact directory is missing."""
    code = app.DEMO_SNIPPETS["Linear Loop O(n)"]["code"]
    res = analyze_source_pipeline(code, artifacts_dir="nonexistent_artifacts_dir")
    assert res.success is True
    assert res.static_result is not None
    assert res.static_result.taxonomy_label == "ON"
    assert res.ml_prediction is None
    assert res.ml_error is not None
    assert res.fusion_result is not None
    assert res.fusion_result.source == "STATIC"


def test_render_functions_smoke_regression():
    """Regression test ensuring render_verdict_card and render_tabs execute without exceptions."""
    code = app.DEMO_SNIPPETS["Constant Time O(1)"]["code"]
    res = analyze_source_pipeline(code)
    assert res.success is True
    # Test that calling rendering functions does not raise AttributeError on fusion.source
    # (e.g. 'str' object has no attribute 'value')
    app.render_verdict_card(res)
    app.render_tabs(res)


def test_ui_model_caching_and_inference_flow_regression():
    """Regression test ensuring UI caching does not fail with tuple attribute error."""
    for model_id in ["random_forest", "xgboost"]:
        cached = app.get_cached_model(model_id)
        assert cached is not None
        model_obj = cached[0] if isinstance(cached, tuple) else cached

        code = app.DEMO_SNIPPETS["Linear Loop O(n)"]["code"]
        # Test passing unpacked model
        res1 = analyze_source_pipeline(code, model=model_obj, model_id=model_id)
        assert res1.success is True
        assert res1.ml_error is None
        assert res1.ml_prediction is not None
        assert res1.ml_prediction.predicted_label == "ON"

        # Test passing raw cached tuple (defensive unpack test)
        res2 = analyze_source_pipeline(code, model=cached, model_id=model_id)
        assert res2.success is True
        assert res2.ml_error is None
        assert res2.ml_prediction is not None
        assert res2.ml_prediction.predicted_label == "ON"


def test_render_verdict_card_displays_big_o_notation():
    """Verify render_verdict_card formats internal labels to standard Big-O notation."""
    code = app.DEMO_SNIPPETS["Linear Loop O(n)"]["code"]
    res = analyze_source_pipeline(code)
    assert res.success is True

    with (
        patch("streamlit.subheader") as mock_subheader,
        patch("streamlit.metric") as mock_metric,
        patch("streamlit.write") as mock_write,
        patch("streamlit.warning"),
        patch("streamlit.caption"),
        patch("streamlit.columns", return_value=[MagicMock(), MagicMock(), MagicMock()]),
        patch("streamlit.markdown"),
        patch("streamlit.success"),
    ):
        app.render_verdict_card(res)

        # Subheader should read "Time Complexity: O(n)"
        mock_subheader.assert_called_with("Time Complexity: O(n)")

        # Metric should have label "Time Complexity" and value "O(n)"
        metric_calls = mock_metric.call_args_list
        found_primary_metric = False
        for call in metric_calls:
            kwargs = call.kwargs
            if kwargs.get("label") == "Time Complexity" and kwargs.get("value") == "O(n)":
                found_primary_metric = True
                break
        assert found_primary_metric, f"Primary metric not found in: {metric_calls}"

        # Write calls should format rationale with O(n) rather than ON
        rationale_call = [c for c in mock_write.call_args_list if "Decision Rationale:" in str(c)]
        assert len(rationale_call) > 0
        rationale_text = str(rationale_call[0])
        assert "O(n)" in rationale_text
        assert " ON " not in rationale_text and " ON." not in rationale_text


def test_render_detected_construct_cards_no_raw_dataclass_repr():
    """Verify detected construct cards do not display raw Python dataclass representations."""
    test_snippets = [
        "Linear Loop O(n)",
        "Nested Loops O(n²)",
        "Exponential Recursion O(2ⁿ)",
    ]

    for name in test_snippets:
        code = app.DEMO_SNIPPETS[name]["code"]
        res = analyze_source_pipeline(code)
        assert res.success is True
        assert res.explanation is not None
        assert len(res.explanation.detected_constructs) > 0

        recorded_texts: list[str] = []

        def record_text(*args: Any, **kwargs: Any) -> None:
            for a in args:
                recorded_texts.append(str(a))
            for v in kwargs.values():
                recorded_texts.append(str(v))

        mock_container = MagicMock()
        mock_container.__enter__.return_value = mock_container
        mock_container.__exit__.return_value = None

        mock_expander = MagicMock()
        mock_expander.__enter__.return_value = mock_expander
        mock_expander.__exit__.return_value = None

        mock_cols = lambda spec: [  # noqa: E731
            MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
        ]
        tabs = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        with (
            patch("streamlit.container", return_value=mock_container),
            patch("streamlit.columns", side_effect=mock_cols),
            patch("streamlit.markdown", side_effect=record_text),
            patch("streamlit.write", side_effect=record_text),
            patch("streamlit.caption", side_effect=record_text),
            patch("streamlit.json") as mock_json,
            patch("streamlit.expander", return_value=mock_expander),
            patch("streamlit.tabs", return_value=tabs),
            patch("streamlit.subheader"),
            patch("streamlit.dataframe"),
            patch("streamlit.bar_chart"),
            patch("streamlit.info"),
            patch("streamlit.warning"),
        ):
            app.render_tabs(res)

            # Raw dataclass representations must NEVER appear in rendered text
            all_rendered = " ".join(recorded_texts)
            assert "DetectedConstruct(" not in all_rendered, (
                f"Raw DetectedConstruct found in UI: {all_rendered}"
            )
            assert "SourceSpanInfo(" not in all_rendered, (
                f"Raw SourceSpanInfo found in UI: {all_rendered}"
            )

            # Technical metadata must be routed to expander / st.json
            assert mock_json.call_count >= 1
            json_args = mock_json.call_args[0][0]
            assert isinstance(json_args, dict)
            assert "kind" in json_args

