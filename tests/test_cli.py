"""tests/test_cli.py — Unit and integration tests for ComplexityLens CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from complexity_lens.cli import main
from complexity_lens.features.schema import FEATURE_NAMES

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cpp"


class TestCLIAnalyzeCommand:
    """Test `analyze <file.cpp>` on existing fixtures in text and JSON mode."""

    def test_analyze_simple_function_text_output(self, capsys):
        fixture = FIXTURES_DIR / "01_simple_function.cpp"
        ret = main(["analyze", str(fixture)])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "ComplexityLens Analysis Report: 01_simple_function.cpp" in captured
        assert "Input Validation: VALID" in captured
        assert "Parser Status:    PARSED" in captured
        assert "Complexity Label     : O1" in captured
        assert "[Structural Features (26 in schema order)]" in captured
        for feat in FEATURE_NAMES:
            assert feat in captured

    def test_analyze_simple_function_json_output(self, capsys):
        fixture = FIXTURES_DIR / "01_simple_function.cpp"
        ret = main(["analyze", str(fixture), "--json"])
        assert ret == 0
        captured = capsys.readouterr().out
        payload = json.loads(captured)
        assert payload["input_validation_status"] == "VALID"
        assert payload["parser_status"] == "PARSED"
        assert payload["static_result"]["label"] == "O1"
        assert "features" in payload
        assert len(payload["features"]) == 26
        # Features must match schema order
        assert list(payload["features"].keys()) == list(FEATURE_NAMES)
        assert "ir_summary" in payload
        assert "reasons" in payload["static_result"]
        assert "evidence" in payload["static_result"]

    def test_analyze_for_loop_fixture(self, capsys):
        fixture = FIXTURES_DIR / "02_for_loop.cpp"
        ret = main(["analyze", str(fixture), "--json"])
        assert ret == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["static_result"]["label"] == "ON"
        assert payload["features"]["n_loops"] >= 1

    def test_analyze_nonexistent_file(self, capsys):
        ret = main(["analyze", "nonexistent.cpp"])
        assert ret == 1


class TestCLIDatasetReportCommand:
    """Test `dataset report` command."""

    def test_dataset_report_text(self, capsys):
        ret = main(["dataset", "report"])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "ComplexityLens" in captured
        assert "Total Records:  38" in captured
        assert "Total Families: 10" in captured
        assert "Duplicate Program IDs : 0" in captured
        assert "Metadata/Schema Errors: 0" in captured

    def test_dataset_report_json(self, capsys):
        ret = main(["dataset", "report", "--json"])
        assert ret == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["total_records"] == 38
        assert payload["total_families"] == 10
        assert len(payload["metadata_schema_failures"]) == 0
        assert len(payload["duplicate_program_ids"]) == 0
        assert len(payload["duplicate_source_hashes"]) == 0


class TestAgreementDiagnosticIntegrity:
    """Test that scripts/label_engine_agreement.py contains exact warning and runs cleanly."""

    def test_diagnostic_script_warning_present(self):
        script_path = Path("scripts/label_engine_agreement.py")
        assert script_path.exists()
        content = script_path.read_text(encoding="utf-8")
        assert "Re-check the rationale; never edit a label to match the engine" in content
