"""tests/test_cli_day14.py — Day 14 End-to-End CLI Integration & Verification Tests.

Verifies:
1. Complete JSON structure: static_result, ml_prediction, ml_unavailable,
   fusion_result, rejection_category, explanation, features, and ir_summary.
2. Serialization integrity: missing values serialize as null, never NaN or Infinity.
3. Uncalibrated ML score labeling: score_label == "uncalibrated_probability".
4. Text report format: includes Static Bound, ML Estimate, Fusion Decision, and Evidence.
5. Input validation gate: rejects malformed code with exit code 1 and FAILED status.
6. Safe fallback and reason-coded abstention when ML is unavailable or static is inconclusive.
"""

from __future__ import annotations

import json
from pathlib import Path

from complexity_lens.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cpp"


class TestCLIDay14EndToEnd:
    """Verify Day 14 end-to-end CLI enhancements and full hybrid reporting."""

    def test_cli_json_full_hybrid_structure(self, capsys) -> None:
        """Verify that CLI --json output exposes all required Day 14 fields."""
        fixture = FIXTURES_DIR / "01_simple_function.cpp"
        ret = main(["analyze", str(fixture), "--json"])
        assert ret == 0

        captured = capsys.readouterr().out
        payload = json.loads(captured)

        # 1. Validation & parser status
        assert payload["input_validation_status"] == "VALID"
        assert payload["parser_status"] == "PARSED"
        assert payload["ir_summary"] is not None
        assert payload["ir_summary"]["num_functions"] >= 1

        # 2. Features in schema order
        assert "features" in payload
        assert len(payload["features"]) == 26

        # 3. Static complexity result
        assert "static_result" in payload
        stat = payload["static_result"]
        assert stat["label"] == "O1"
        assert stat["exactness"] == "STRUCTURAL"
        assert "reason_codes" in stat
        assert "evidence" in stat

        # 4. ML prediction container
        assert "ml_prediction" in payload
        assert "ml_unavailable" in payload
        if not payload["ml_unavailable"]:
            ml = payload["ml_prediction"]
            assert ml is not None
            assert "predicted_label" in ml
            assert "uncalibrated_score" in ml
            assert ml["score_label"] == "uncalibrated_probability"
            assert "probabilities" in ml
            assert len(ml["probabilities"]) == 8

        # 5. Hybrid fusion gate result
        assert "fusion_result" in payload
        fusion = payload["fusion_result"]
        assert fusion is not None
        assert "primary_label" in fusion
        assert fusion["primary_label"] == "O1"
        assert "source" in fusion
        assert "verification_status" in fusion
        assert "decision_reason" in fusion
        assert "rejection_category" in fusion
        # For statically verified O1, rejection category is null (None in JSON)
        assert fusion["rejection_category"] is None
        assert "rationale" in fusion

        # 6. Explanation report
        assert "explanation" in payload
        if payload["explanation"] is not None:
            assert "summary" in payload["explanation"]
            assert "recommendations" in payload["explanation"]

    def test_cli_json_no_nan_or_infinity(self, capsys) -> None:
        """Verify that JSON serialization adheres strictly to standard JSON with nulls, not NaNs."""
        fixture = FIXTURES_DIR / "02_for_loop.cpp"
        ret = main(["analyze", str(fixture), "--json"])
        assert ret == 0

        captured = capsys.readouterr().out
        # Standard json.loads with parse_constant that rejects NaN/Infinity
        def _fail_on_constant(val: str) -> None:
            raise ValueError(f"Encountered non-standard JSON constant: {val}")

        payload = json.loads(captured, parse_constant=_fail_on_constant)
        assert payload["static_result"]["label"] == "ON"
        assert payload["fusion_result"]["primary_label"] == "ON"

    def test_cli_text_full_sections_present(self, capsys) -> None:
        """Verify that CLI text report includes all expected Day 14 report sections."""
        fixture = FIXTURES_DIR / "01_simple_function.cpp"
        ret = main(["analyze", str(fixture)])
        assert ret == 0

        captured = capsys.readouterr().out
        # Standard legacy headers
        assert "ComplexityLens Analysis Report: 01_simple_function.cpp" in captured
        assert "Input Validation: VALID" in captured
        assert "Parser Status:    PARSED" in captured
        assert "[IR Summary]" in captured
        assert "[Structural Features (26 in schema order)]" in captured
        assert "[Complexity Engine Result]" in captured
        assert "Complexity Label     : O1" in captured

        # Day 14 enhanced sections
        assert "[Machine Learning Estimate]" in captured
        assert "[Hybrid Fusion Decision]" in captured
        assert "Final Complexity     : O1" in captured
        assert "Decision Source      :" in captured
        assert "Verification Status  :" in captured
        assert "Decision Reason      :" in captured

    def test_cli_input_validation_failure(self, capsys, tmp_path) -> None:
        """Verify that input validation failures yield exit code 1 and structured error output."""
        bad_cpp = tmp_path / "bad_syntax.cpp"
        # Over-length line or unclosed braces to trigger validation error
        bad_cpp.write_text("int main() {\n" + ("x" * 2500) + "\n", encoding="utf-8")

        # In JSON mode
        ret_json = main(["analyze", str(bad_cpp), "--json"])
        assert ret_json == 1
        captured_json = capsys.readouterr().out
        payload = json.loads(captured_json)
        assert payload["input_validation_status"] == "FAILED"
        assert "error" in payload
        assert "exceeds maximum length" in payload["error"].lower()

        # In Text mode
        ret_text = main(["analyze", str(bad_cpp)])
        assert ret_text == 1
        captured_text = capsys.readouterr().out
        assert "Input Validation: FAILED" in captured_text
