"""tests/test_cli.py — Unit and integration tests for ComplexityLens CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from complexity_lens.cli import main
from complexity_lens.dataset import load_dataset
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
        records, families, _ = load_dataset()
        ret = main(["dataset", "report"])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "ComplexityLens" in captured
        assert f"Total Records:  {len(records)}" in captured
        assert f"Total Families: {len(families)}" in captured
        assert "Duplicate Program IDs : 0" in captured
        assert "Metadata/Schema Errors: 0" in captured

    def test_dataset_report_json(self, capsys):
        records, families, _ = load_dataset()
        ret = main(["dataset", "report", "--json"])
        assert ret == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["total_records"] == len(records)
        assert payload["total_families"] == len(families)
        assert len(payload["metadata_schema_failures"]) == 0
        assert len(payload["duplicate_program_ids"]) == 0
        assert len(payload["duplicate_source_hashes"]) == 0

    def test_run_dataset_report_keyword_argument_regression(self, capsys, monkeypatch):
        """Regression test for Fix 1: verify check_parser argument and output schema."""
        from unittest.mock import MagicMock

        from complexity_lens import cli

        mock_report = {
            "total_records": 10,
            "total_families": 3,
            "counts_by_label": {"O1": 5, "ON": 5},
            "counts_by_family": {"f01": 5, "f02": 5},
            "counts_by_origin": {"hand_written": 10},
            "unknown_reasons_breakdown": {},
            "metadata_schema_failures": [],
            "duplicate_program_ids": [],
            "duplicate_source_hashes": [],
            "parser_validity": {
                "status": "PARSER_AVAILABLE",
                "total_tested": 10,
                "parsed_count": 10,
                "failed_count": 0,
            },
        }

        mock_gen = MagicMock(return_value=mock_report)
        monkeypatch.setattr(cli, "generate_dataset_report", mock_gen)

        ret = cli.run_dataset_report("datasets", as_json=False)
        assert ret == 0
        mock_gen.assert_called_once_with(dataset_root=Path("datasets"), check_parser=True)
        captured = capsys.readouterr().out
        assert "Total Records:  10" in captured
        assert "Total Families: 3" in captured

        # JSON mode regression
        ret_json = cli.run_dataset_report("datasets", as_json=True)
        assert ret_json == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["total_records"] == 10
        assert payload["total_families"] == 3


class TestAgreementDiagnosticIntegrity:
    """Test that scripts/label_engine_agreement.py contains exact warning and runs cleanly."""

    def test_diagnostic_script_warning_present(self):
        script_path = Path("scripts/label_engine_agreement.py")
        assert script_path.exists()
        content = script_path.read_text(encoding="utf-8")
        assert "Re-check the rationale; never edit a label to match the engine" in content
