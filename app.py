"""ComplexityLens Streamlit Web Application.

Interactive prototype for analyzing C++ algorithmic time complexity using
hybrid static inference rules and machine learning estimation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from complexity_lens.pipeline import (
    PipelineResult,
    analyze_source_pipeline,
    safe_load_model,
)
from complexity_lens.presentation import (
    COMPLEXITY_DISPLAY_MAP,
    format_complexity_label,
    format_construct_display,
    format_explanation_text,
)

# Canonical complexity label mathematical representations (standard Big-O display notation)
LABEL_MATH_NOTATION: dict[str, str] = COMPLEXITY_DISPLAY_MAP

# Hand-crafted demonstration snippets safe for AST parsing
DEMO_SNIPPETS: dict[str, dict[str, str]] = {
    "Linear Loop O(n)": {
        "description": "Single for-loop iterating up to n with constant-time loop body.",
        "expected": "O(n)",
        "code": """// Linear loop iterating n times
int compute_linear(int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += i;
    }
    return sum;
}
""",
    },
    "Constant Time O(1)": {
        "description": "Straight-line arithmetic computation without loops or recursion.",
        "expected": "O(1)",
        "code": """// Constant-time arithmetic operations
int compute_constant(int a, int b) {
    int x = a + b;
    int y = x * 2;
    return y;
}
""",
    },
    "Nested Loops O(n²)": {
        "description": "Two nested loops both bound by parameter n.",
        "expected": "O(n²)",
        "code": """// Nested quadratic loops
int compute_nested(int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sum += i * j;
        }
    }
    return sum;
}
""",
    },
    "Logarithmic Loop O(log n)": {
        "description": "Multiplicative step loop doubling index at each iteration.",
        "expected": "O(log n)",
        "code": """// Logarithmic loop doubling step
int compute_log(int n) {
    int count = 0;
    for (int i = 1; i < n; i *= 2) {
        count++;
    }
    return count;
}
""",
    },
    "Exponential Recursion O(2ⁿ)": {
        "description": "Binary recursive tree with constant reduction per level.",
        "expected": "O(2ⁿ)",
        "code": """// Binary tree recursive branching
int compute_exp(int n) {
    if (n <= 1) {
        return 1;
    }
    return compute_exp(n - 1) + compute_exp(n - 2);
}
""",
    },
    "Static UNKNOWN with ML Fallback": {
        "description": "Inline assembly construct triggering static UNKNOWN, resolved via ML.",
        "expected": "ML Estimate",
        "code": """// Inline assembly unsupported by static rules
int compute_assembly_fallback(int n) {
    __asm__("nop");
    return n;
}
""",
    },
    "Inconclusive Collatz Loop": {
        "description": "Data-dependent while-loop convergence legitimately yielding UNKNOWN.",
        "expected": "UNKNOWN",
        "code": """// Collatz 3n+1 loop with data-dependent convergence
int compute_collatz(int n) {
    int steps = 0;
    while (n > 1) {
        if (n % 2 == 0) {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        steps++;
    }
    return steps;
}
""",
    },
}

MODEL_LIMITATION_NOTE: str = (
    "The ML model was developed using development-partition programs. "
    "A prediction for an individual input is not evidence of model accuracy."
)


@st.cache_resource(show_spinner=False)
def get_cached_model(model_id: str) -> Any:
    """Load and cache the verified ML model artifact."""
    return safe_load_model(model_id=model_id, artifacts_dir="artifacts/models")


@st.cache_data(show_spinner=False)
def load_dev_evaluation_metrics() -> dict[str, Any] | None:
    """Load development cross-validation metrics from verified artifact metadata."""
    artifacts_dir = Path("artifacts/models")
    rf_meta_path = artifacts_dir / "random_forest_metadata.json"
    xgb_meta_path = artifacts_dir / "xgboost_metadata.json"

    if not rf_meta_path.is_file() or not xgb_meta_path.is_file():
        return None

    try:
        with rf_meta_path.open("r", encoding="utf-8") as f:
            rf_meta = json.load(f)
        with xgb_meta_path.open("r", encoding="utf-8") as f:
            xgb_meta = json.load(f)

        return {
            "random_forest": {
                "model_id": "random_forest",
                "score_label": rf_meta.get("winning_score_label", "Development CV estimate"),
                "macro_f1": rf_meta.get("winning_score_value"),
                "accuracy": rf_meta.get("winning_accuracy"),
                "metric_name": rf_meta.get("selection_metric", "pooled_oof_macro_f1"),
                "candidates": rf_meta.get("selection_evaluations", []),
            },
            "xgboost": {
                "model_id": "xgboost",
                "score_label": xgb_meta.get("winning_score_label", "Development CV estimate"),
                "macro_f1": xgb_meta.get("winning_score_value"),
                "accuracy": xgb_meta.get("winning_accuracy"),
                "metric_name": xgb_meta.get("selection_metric", "pooled_oof_macro_f1"),
                "candidates": xgb_meta.get("selection_evaluations", []),
            },
        }
    except Exception:
        return None


def inject_custom_styles() -> None:
    """Inject static developer-authored CSS styles.

    Never interpolates user-derived text, code, or diagnostics.
    """
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        div[data-baseweb="textarea"] textarea {
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
            font-size: 0.88rem !important;
            line-height: 1.5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render top page header and subtitle."""
    st.title("ComplexityLens")
    st.caption(
        "Hybrid Static-Analysis & Machine-Learning Asymptotic Complexity Verification for C++"
    )


def render_sidebar() -> dict[str, Any]:
    """Render sidebar configuration and technical disclosures."""
    with st.sidebar:
        st.subheader("Analysis Configuration")

        model_id = st.selectbox(
            "ML Model Artifact",
            options=["random_forest", "xgboost"],
            index=0,
            help=(
                "Random Forest is the audited default prototype model with higher development CV. "
                "XGBoost is available as an optional comparator."
            ),
        )

        enable_ml = st.checkbox(
            "Enable ML Fallback",
            value=True,
            help="Allow machine-learning inference when static analysis is inconclusive.",
        )

        ml_threshold = st.slider(
            "ML Fallback Threshold",
            min_value=0.40,
            max_value=0.90,
            value=0.55,
            step=0.05,
            help=(
                "Minimum required predicted class probability before an ML estimate is accepted "
                "when static analysis yields UNKNOWN. Default provisional threshold is 0.55."
            ),
        )

        cost_model = st.selectbox(
            "STL Cost Model",
            options=["expected_hash", "worst_case_tree"],
            index=0,
            help="Cost semantics for standard library operations.",
        )

        fn_target = st.text_input(
            "Target Function (optional)",
            value="",
            help="Specify a function name to analyze. If empty, the engine resolves automatically.",
        )

        st.divider()
        st.subheader("Technical Disclosures")
        st.warning(
            "**Space Complexity**: NOT IMPLEMENTED. The current prototype analyzes "
            "time complexity only."
        )
        st.info(
            "**Parser Execution**: The AST parser uses `libclang` deterministically and does "
            "not enforce an execution timeout."
        )
        st.caption(
            "**ML Probabilities**: Probability values reflect raw classifier heuristic "
            "estimates and are NOT calibrated."
        )
        st.caption(
            "**Execution Safety**: ComplexityLens NEVER compiles or executes C++ code; "
            "all analysis is purely static and AST-based."
        )

    return {
        "model_id": model_id,
        "enable_ml": enable_ml,
        "ml_threshold": ml_threshold,
        "cost_model": cost_model,
        "function_name": fn_target.strip() or None,
    }


def render_verdict_card(result: PipelineResult) -> None:
    """Render the primary fused verdict safely using standard Streamlit components."""
    fusion = result.fusion_result
    static = result.static_result
    ml = result.ml_prediction

    if not fusion:
        return

    primary_label = fusion.primary_label
    math_label = format_complexity_label(primary_label)
    source_str = fusion.source.value if hasattr(fusion.source, "value") else str(fusion.source)

    if source_str == "STATIC":
        source_desc = "STATIC ANALYSIS (VERIFIED)"
    elif source_str == "ML_ESTIMATE":
        source_desc = "ML ESTIMATE (EMPIRICAL FALLBACK)"
    else:
        source_desc = "INCONCLUSIVE (UNKNOWN)"

    exactness_label = static.exactness if static else "UNKNOWN"

    st.subheader(f"Time Complexity: {math_label}")

    v_col1, v_col2, v_col3 = st.columns([2, 1.5, 1.5])
    with v_col1:
        st.metric(
            label="Time Complexity",
            value=math_label,
            delta=f"Source: {source_desc}",
        )
    with v_col2:
        st.metric(label="Static Exactness", value=exactness_label)
    with v_col3:
        st.metric(label="Decision Reason", value=fusion.decision_reason)

    v_status = getattr(fusion, "verification_status", "STATIC_VERIFIED")
    st.write(f"**Verification Status:** `{v_status}`")
    st.write(f"**Decision Rationale:** {format_explanation_text(fusion.rationale)}")

    if getattr(fusion, "abstention_reason", None):
        st.info(f"**Abstention Reason:** {fusion.abstention_reason}")

    # Mandatory System Notices rendered safely
    st.warning(
        "⚠️ **Prototype Disclosures:**\n"
        "- **Auxiliary Space Complexity**: Not implemented in the current prototype. "
        "Only time complexity is analyzed.\n"
        "- **Parser Execution**: No execution timeout is enforced during AST parsing."
    )

    # Three-Column Metric Comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Static Engine")
        stat_lbl = static.taxonomy_label if static else "N/A"
        stat_math = format_complexity_label(stat_lbl) if stat_lbl != "N/A" else "N/A"
        st.write(f"**Time Complexity:** `{stat_math}`")
        st.write(f"**Exactness:** `{static.exactness if static else 'N/A'}`")
        stat_codes = ", ".join(static.reason_codes) if (static and static.reason_codes) else "None"
        st.write(f"**Reason Codes:** `{stat_codes}`")

    with col2:
        st.markdown("#### Machine Learning")
        if ml:
            ml_math = format_complexity_label(ml.predicted_label)
            prob_val = getattr(fusion, "ml_score_uncalibrated", ml.probability)
            prob_score = f"{prob_val:.4f}" if prob_val is not None else "N/A"
            st.write(f"**Time Complexity:** `{ml_math}`")
            st.write(f"**ML score (max class probability, uncalibrated):** `{prob_score}`")
            st.write(f"**Model Artifact:** `{ml.model_id}`")
        else:
            st.write("**Status:** Unavailable")
            st.caption(result.ml_error or "ML fallback disabled or model missing.")

    with col3:
        st.markdown("#### Fusion Gate")
        if fusion.agreement is True:
            st.success("Engine Agreement: Verified Match")
        elif fusion.agreement is False:
            st.warning("Engine Disagreement: Static Precedence")
        else:
            st.info("Engine Agreement: Inconclusive / N/A")
        st.write(f"**Active Source:** `{source_str}`")
        st.write(f"**Verification:** `{getattr(fusion, 'verification_status', 'N/A')}`")
        st.write(f"**Rule Code:** `{fusion.decision_reason}`")

    # Visible model limitation note as strictly required
    st.caption(f"ℹ️ {MODEL_LIMITATION_NOTE}")


def render_detected_construct_card(construct: Any) -> None:
    """Render a single detected code construct as a structured, human-readable card."""
    card = format_construct_display(construct)

    try:
        container = st.container(border=True)
    except TypeError:
        container = st.container()

    with container:
        hdr_col1, hdr_col2 = st.columns([3, 1])
        with hdr_col1:
            st.markdown(f"**{card['title']}**")
        with hdr_col2:
            st.caption(f"📍 `{card['line_range']}`")

        st.write(card["description"])

        if card["primary_details"]:
            cols = st.columns(min(len(card["primary_details"]), 4))
            for col, (label, val) in zip(cols, card["primary_details"]):
                col.caption(f"**{label}**")
                col.write(val)

        if card["technical_details"]:
            with st.expander("Technical Construct Metadata", expanded=False):
                st.json(card["technical_details"])


def render_tabs(result: PipelineResult) -> None:
    """Render structured tab views using safe components."""
    dev_metrics = load_dev_evaluation_metrics()
    tab_titles = [
        "Explanation & Derivation",
        "Evidence & Code Spans",
        "Recommendations",
        "AST Diagnostics & Limitations",
        "ML Features & Probabilities",
    ]
    if dev_metrics is not None:
        tab_titles.append("Evaluation Metrics (Dev CV)")

    tabs = st.tabs(tab_titles)

    expl = result.explanation
    static = result.static_result

    with tabs[0]:
        st.subheader("Explanation & Derivation")
        if expl:
            st.write(f"**Summary:** {format_explanation_text(expl.summary)}")

            if expl.detected_constructs:
                st.write("**Detected Code Constructs:**")
                for construct in expl.detected_constructs:
                    render_detected_construct_card(construct)

            if expl.evidence_reasoning:
                st.write("**Derivation Reasoning:**")
                for step in expl.evidence_reasoning:
                    st.write(f"- {format_explanation_text(step)}")

        if static and static.rule_trace:
            st.write("**Static Rule Application Trace:**")
            trace_items = []
            for idx, step in enumerate(static.rule_trace, 1):
                rule_id = getattr(step, "rule_id", getattr(step, "rule_name", "UNKNOWN_RULE"))
                out_cls = getattr(step, "output_class", None)
                if out_cls is not None:
                    out_lbl = getattr(out_cls, "taxonomy_label", "N/A")
                    out_lbl_str = out_lbl.value if hasattr(out_lbl, "value") else str(out_lbl)
                    bound_str = format_complexity_label(out_lbl_str)
                    exact_val = getattr(out_cls, "exactness", "N/A")
                    exact_str = exact_val.value if hasattr(exact_val, "value") else str(exact_val)
                else:
                    raw_bound = getattr(step, "resulting_bound", "N/A")
                    bound_str = format_complexity_label(raw_bound)
                    exact_str = "N/A"

                nodes_str = ", ".join(getattr(step, "node_ids", [])) or "None"
                trace_items.append(
                    {
                        "Step": idx,
                        "Rule": rule_id,
                        "Resulting Bound": bound_str,
                        "Exactness": exact_str,
                        "Evidence Nodes": nodes_str,
                    }
                )
            st.dataframe(pd.DataFrame(trace_items), use_container_width=True)
        elif not expl:
            st.info("No detailed explanation available for this analysis.")

    with tabs[1]:
        st.subheader("Source Evidence & Relevant Code Spans")
        spans_to_show = []
        if expl and expl.relevant_spans:
            for s in expl.relevant_spans:
                descr = getattr(s, "description", "") or "Relevant AST construct"
                start_l = getattr(s, "start_line", 0)
                end_l = getattr(s, "end_line", 0)
                start_c = getattr(s, "start_col", 0)
                end_c = getattr(s, "end_col", 0)
                spans_to_show.append(
                    {
                        "Lines": f"{start_l} - {end_l}",
                        "Columns": f"{start_c} - {end_c}",
                        "Description": descr,
                    }
                )
        elif static and static.source_spans:
            for span in static.source_spans:
                start_l = getattr(span, "start_line", 0)
                end_l = getattr(span, "end_line", 0)
                start_c = getattr(span, "start_col", 0)
                end_c = getattr(span, "end_col", 0)
                file_name = getattr(span, "file", "Source Range")
                spans_to_show.append(
                    {
                        "Lines": f"{start_l} - {end_l}",
                        "Columns": f"{start_c} - {end_c}",
                        "Description": f"Location in {file_name}",
                    }
                )

        if spans_to_show:
            st.dataframe(pd.DataFrame(spans_to_show), use_container_width=True)
        else:
            st.info("No specific localized code spans reported by the analysis engine.")

    with tabs[2]:
        st.subheader("Evidence-Backed Recommendations")
        if expl and expl.recommendations:
            st.write(
                "The following conservative recommendations were derived from verified "
                "AST patterns:"
            )
            for rec in expl.recommendations:
                st.write(f"- {rec}")
        else:
            st.info(
                "No specific optimization recommendations identified from the verified AST "
                "evidence. The system does not fabricate speculative advice."
            )

    with tabs[3]:
        st.subheader("AST Diagnostics & Limitations")

        if expl and expl.unsupported_constructs:
            st.warning("Unsupported Constructs Encountered:")
            for uc in expl.unsupported_constructs:
                st.write(
                    f"- Kind: `{uc.kind}` (Line {uc.line or 'unknown'}): "
                    f"{uc.description or 'No description'}"
                )

        if expl and expl.unknown_reason:
            st.write(
                f"**Inconclusive Analysis Reason:** {format_explanation_text(expl.unknown_reason)}"
            )

        if expl and expl.limitations:
            st.write("**Engine Limitations:**")
            for lim in expl.limitations:
                st.write(f"- {lim}")

        if result.program_ir and result.program_ir.diagnostics:
            st.write("**Parser Diagnostics (`libclang`):**")
            diag_data = [
                {
                    "Severity": (
                        d.severity.name if hasattr(d.severity, "name") else str(d.severity)
                    ),
                    "Message": d.message,
                    "Line": d.location.line if d.location else "N/A",
                    "Column": d.location.column if d.location else "N/A",
                }
                for d in result.program_ir.diagnostics
            ]
            st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
        else:
            st.caption("No syntax errors or parser warnings emitted by libclang.")

    with tabs[4]:
        st.subheader("ML Features & Probability Distribution")
        st.caption(f"Notice: {MODEL_LIMITATION_NOTE}")

        if result.ml_prediction and result.ml_prediction.probabilities:
            st.write("**Canonical Class Probability Distribution:**")
            probs = result.ml_prediction.probabilities
            df_probs = pd.DataFrame(
                [
                    {
                        "Class": cls_name,
                        "Notation": format_complexity_label(cls_name),
                        "Probability": float(probs.get(cls_name, 0.0)),
                    }
                    for cls_name in [
                        "O1",
                        "OLOGN",
                        "ON",
                        "ONLOGN",
                        "ON2",
                        "ON3",
                        "OEXP",
                        "UNKNOWN",
                    ]
                ]
            )
            st.bar_chart(df_probs.set_index("Notation")["Probability"])
            st.dataframe(
                df_probs.style.format({"Probability": "{:.3%}"}),
                use_container_width=True,
            )
            st.caption("Note: Classifier probabilities are heuristic and not calibrated.")
        elif result.ml_error:
            st.warning(f"ML probability distribution unavailable: {result.ml_error}")

        if result.feature_vector:
            st.write("**26-Feature Canonical Vector (AST Extracted):**")
            feat_dict = result.feature_vector.to_dict()
            df_features = pd.DataFrame(
                [{"Feature Name": k, "Value": v} for k, v in feat_dict.items()]
            )
            st.dataframe(df_features, use_container_width=True)

    if dev_metrics is not None and len(tabs) > 5:
        with tabs[5]:
            st.subheader("Development Cross-Validation Metrics")
            st.info(
                "**Dataset Note**: The metrics below represent development cross-validation "
                "estimates across development partition folds. "
                "**The frozen test set has NOT been evaluated.**"
            )

            metric_rows = []
            for m_key, m_info in dev_metrics.items():
                macro_val = m_info["macro_f1"]
                acc_val = m_info["accuracy"]
                metric_rows.append(
                    {
                        "Model": m_info["model_id"],
                        "Selection Metric": m_info["metric_name"],
                        "Winning Score Label": m_info["score_label"],
                        "Macro-F1 (Pooled OOF)": (
                            f"{macro_val:.4f}" if macro_val is not None else "N/A"
                        ),
                        "Accuracy (Pooled OOF)": (
                            f"{acc_val:.4f}" if acc_val is not None else "N/A"
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)
            st.caption(
                "Label: Development CV — selection-biased estimate. "
                "Winning configurations were selected using pooled out-of-fold macro-F1."
            )


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="ComplexityLens",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_styles()
    render_header()
    config = render_sidebar()

    # Session State Initialization
    if "code_area_key" not in st.session_state:
        st.session_state["code_area_key"] = DEMO_SNIPPETS["Linear Loop O(n)"]["code"]
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    st.subheader("C++ Source Input")

    col_selector, col_meta = st.columns([2, 3])
    snippet_names = ["Custom / Blank"] + list(DEMO_SNIPPETS.keys())

    with col_selector:

        def on_snippet_select() -> None:
            chosen = st.session_state["snippet_choice"]
            if chosen in DEMO_SNIPPETS:
                st.session_state["code_area_key"] = DEMO_SNIPPETS[chosen]["code"]
            elif chosen == "Custom / Blank":
                st.session_state["code_area_key"] = ""
            st.session_state["last_result"] = None

        st.selectbox(
            "Load Demonstration Example:",
            options=snippet_names,
            key="snippet_choice",
            on_change=on_snippet_select,
            help=(
                "Select a hand-crafted sample program designed to test specific complexity classes."
            ),
        )

    with col_meta:
        current_choice = st.session_state.get("snippet_choice", snippet_names[0])
        if current_choice in DEMO_SNIPPETS:
            meta = DEMO_SNIPPETS[current_choice]
            st.info(
                f"**Expected**: `{meta['expected']}` — {meta['description']}",
                icon="ℹ️",
            )
        else:
            st.caption(
                "Paste or type custom C++ functions. No system header includes needed for "
                "standard loops."
            )

    code = st.text_area(
        "C++ Source Code:",
        height=260,
        key="code_area_key",
        help="Paste standalone C++ code. The engine analyzes AST structure without compilation.",
    )

    col_btn_run, col_btn_clear, _ = st.columns([1.5, 1, 6])

    with col_btn_run:
        analyze_clicked = st.button("Analyze Complexity", type="primary", use_container_width=True)

    with col_btn_clear:
        if st.button("Clear Code", use_container_width=True):
            st.session_state["code_area_key"] = ""
            st.session_state["last_result"] = None
            st.session_state["snippet_choice"] = "Custom / Blank"
            st.rerun()

    if analyze_clicked:
        clean_source = (code or "").strip()
        if not clean_source:
            st.error("Validation Error: Source code cannot be empty. Please enter C++ source code.")
            st.session_state["last_result"] = None
        else:
            cached_model = None
            if config["enable_ml"]:
                try:
                    loaded = get_cached_model(config["model_id"])
                    cached_model = loaded[0] if isinstance(loaded, tuple) else loaded
                except Exception:
                    cached_model = None

            with st.spinner("Analyzing C++ Abstract Syntax Tree..."):
                result = analyze_source_pipeline(
                    source=code,
                    function_name=config["function_name"],
                    cost_model=config["cost_model"],
                    model=cached_model,
                    model_id=config["model_id"],
                    artifacts_dir="artifacts/models",
                    ml_threshold=config["ml_threshold"],
                    enable_ml=config["enable_ml"],
                )
            st.session_state["last_result"] = result

    # Render results if available
    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()
        if not result.success:
            err_line = (
                f" (Line {result.validation_error_line})" if result.validation_error_line else ""
            )
            st.error(
                f"Source Validation / Parser Failure: "
                f"{result.validation_error or 'Unknown error'}{err_line}"
            )
            if result.program_ir and result.program_ir.diagnostics:
                with st.expander("Clang Parse Diagnostics", expanded=True):
                    diag_data = [
                        {
                            "Severity": (
                                d.severity.name if hasattr(d.severity, "name") else str(d.severity)
                            ),
                            "Message": d.message,
                            "Line": d.location.line if d.location else "N/A",
                            "Column": d.location.column if d.location else "N/A",
                        }
                        for d in result.program_ir.diagnostics
                    ]
                    st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
        else:
            render_verdict_card(result)
            render_tabs(result)


if __name__ == "__main__":
    main()
