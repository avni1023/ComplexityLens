"""cli.py — Command Line Interface for ComplexityLens.

Provides user-facing commands:
- analyze <file.cpp> [--json]: Full analysis pipeline (validation, parser status, IR summary,
  26 features in schema order, static complexity result with evidence and reasons).
- dataset report [--json]: Evaluates dataset records, counting labels, families, UNKNOWN reasons,
  duplicate hashes, metadata failures, and parser-validity status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from complexity_lens.complexity.engine import analyze_file
from complexity_lens.dataset import generate_dataset_report
from complexity_lens.features.extractor import extract_features
from complexity_lens.features.schema import FEATURE_NAMES
from complexity_lens.ir.builder import build_ir_from_file
from complexity_lens.ir.model import LoopIR, ProgramIR
from complexity_lens.parsers import ClangParserError
from complexity_lens.validation import ValidationError, validate_source


def _summarize_ir(program: ProgramIR) -> dict[str, Any]:
    """Derive a summary of IR statistics from ProgramIR."""
    num_loops = sum(
        1
        for f in program.functions
        for item in f.body
        if isinstance(item, LoopIR)
    )
    entry = None
    try:
        from complexity_lens.ir.entry_point import resolve_entry_function
        entry_func, _ = resolve_entry_function(program)
        entry = entry_func.name if entry_func else None
    except Exception:
        entry = program.functions[0].name if program.functions else None

    return {
        "num_functions": len(program.functions),
        "num_loops": num_loops,
        "entry_function": entry,
        "analysis_complete": program.analysis_complete,
        "diagnostics_count": len(program.diagnostics),
    }


def run_analyze(
    file_path: str | Path,
    as_json: bool = False,
    function_name: str | None = None,
) -> int:
    """Execute analysis for a single C++ source file and print text or JSON output."""
    path = Path(file_path)
    if not path.exists():
        err_msg = f"File not found: {file_path}"
        if as_json:
            print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        err_msg = f"Failed to read file: {e}"
        if as_json:
            print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    # 1. Validation phase
    try:
        validate_source(source)
        validation_status = "VALID"
        validation_error = None
    except ValidationError as e:
        validation_status = "FAILED"
        validation_error = str(e)
        if as_json:
            print(
                json.dumps(
                    {
                        "file": str(path),
                        "input_validation_status": validation_status,
                        "error": validation_error,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Input Validation: {validation_status} ({validation_error})")
        return 1

    # 2. Parsing and IR construction phase
    try:
        program = build_ir_from_file(path)
        parser_status = "PARSED"
        parser_error = None
    except ClangParserError as e:
        parser_status = "PARSER_UNAVAILABLE"
        parser_error = str(e)
        if as_json:
            print(
                json.dumps(
                    {
                        "file": str(path),
                        "input_validation_status": validation_status,
                        "parser_status": parser_status,
                        "error": parser_error,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Parser Status: {parser_status} ({parser_error})")
        return 1
    except Exception as e:
        parser_status = "FAILED"
        parser_error = str(e)
        if as_json:
            print(
                json.dumps(
                    {
                        "file": str(path),
                        "input_validation_status": validation_status,
                        "parser_status": parser_status,
                        "error": parser_error,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Parser Status: {parser_status} ({parser_error})")
        return 1

    # 3. IR summary
    ir_summary = _summarize_ir(program)

    # 4. Feature extraction (all 26 features in schema order)
    feat_vec = extract_features(program, entry_function_name=function_name)
    features_ordered = feat_vec.to_dict()

    # 5. Static complexity analysis
    result = analyze_file(path, function_name=function_name)

    evidence_dict: dict[str, Any] = {
        "rule_trace": [t.to_dict() for t in result.rule_trace],
        "evidence_node_ids": list(result.evidence_node_ids),
        "source_spans": [span.to_dict() for span in result.source_spans],
    }

    reasons_list = list(result.reason_codes)

    if as_json:
        payload = {
            "file": str(path),
            "input_validation_status": validation_status,
            "parser_status": parser_status,
            "ir_summary": ir_summary,
            "features": features_ordered,
            "static_result": {
                "label": result.taxonomy_label,
                "complexity_class": result.complexity_class.to_dict(),
                "exactness": result.exactness,
                "selected_function": result.selected_function,
                "explanation": result.explanation,
                "reasons": reasons_list,
                "evidence": evidence_dict,
            },
        }
        print(json.dumps(payload, indent=2))
        return 0

    # Human-readable text output
    print("=" * 80)
    print(f"ComplexityLens Analysis Report: {path.name}")
    print("=" * 80)
    print(f"Input Validation: {validation_status}")
    print(f"Parser Status:    {parser_status}")
    print(f"Target Function:  {result.selected_function or ir_summary.get('entry_function')}")

    print("\n[IR Summary]")
    for k, v in ir_summary.items():
        print(f"  {k:<22}: {v}")

    print("\n[Structural Features (26 in schema order)]")
    for fname in FEATURE_NAMES:
        val = features_ordered.get(fname, 0)
        print(f"  {fname:<30}: {val}")

    print("\n[Static Complexity Result]")
    print(f"  Complexity Label     : {result.taxonomy_label}")
    print(f"  Asymptotic Class     : {str(result.complexity_class)}")
    print(f"  Exactness            : {result.exactness}")
    print(f"  Explanation          : {result.explanation}")

    print("\n[Supporting Evidence & Reasons]")
    if reasons_list:
        print(f"  Reason Codes         : {', '.join(reasons_list)}")
    else:
        print("  Reason Codes         : None")

    if result.rule_trace:
        print("  Rule Trace:")
        for t in result.rule_trace:
            inputs = ", ".join(str(c) for c in t.input_classes) if t.input_classes else "none"
            print(f"    - [{t.rule_id}] input: ({inputs}) -> output: {str(t.output_class)}")

    if result.source_spans:
        print("  Evidence Spans:")
        for span in result.source_spans:
            print(f"    - {span}")

    print("=" * 80)
    return 0


def run_dataset_report(dataset_dir: str | Path = "datasets", as_json: bool = False) -> int:
    """Execute dataset report over the ground-truth benchmark and print results."""
    report = generate_dataset_report(dataset_dir, check_parser=True)

    if as_json:
        print(json.dumps(report, indent=2))
        return 0 if not report["metadata_schema_failures"] else 1

    print("=" * 80)
    print("ComplexityLens — Dataset Verification and Integrity Report")
    print("=" * 80)

    print(f"\nTotal Records:  {report['total_records']}")
    print(f"Total Families: {report['total_families']}")

    print("\n[Counts by Expected Label]")
    for lbl, count in sorted(report["counts_by_label"].items()):
        print(f"  {lbl:<12}: {count}")

    print("\n[Counts by Family]")
    for fam, count in sorted(report["counts_by_family"].items()):
        print(f"  {fam:<36}: {count}")

    if "counts_by_origin" in report:
        print("\n[Counts by Origin]")
        for orig, count in sorted(report["counts_by_origin"].items()):
            print(f"  {orig:<14}: {count}")

    print("\n[UNKNOWN Counts and Reasons Breakdown]")
    if report["unknown_reasons_breakdown"]:
        for reason, count in sorted(report["unknown_reasons_breakdown"].items()):
            print(f"  {reason:<26}: {count}")
    else:
        print("  None")

    print("\n[Integrity & Duplicate Checks]")
    dup_ids = report["duplicate_program_ids"]
    dup_hashes = report["duplicate_source_hashes"]
    print(
        f"  Duplicate Program IDs : {len(dup_ids)}"
        + (f" ({dup_ids})" if dup_ids else "")
    )
    print(
        f"  Duplicate Hashes      : {len(dup_hashes)}"
        + (f" ({dup_hashes})" if dup_hashes else "")
    )

    schema_failures = report["metadata_schema_failures"]
    print(f"  Metadata/Schema Errors: {len(schema_failures)}")
    for sf in schema_failures:
        print(f"    - {sf}")

    print("\n[Parser-Validity Status]")
    parser_info = report["parser_validity"]
    status = parser_info.get("status", "UNKNOWN")
    print(f"  Status        : {status}")
    if status == "PARSER_UNAVAILABLE":
        print(f"  Message       : {parser_info.get('message')}")
    else:
        print(f"  Total Tested  : {parser_info.get('total_tested', 0)}")
        print(f"  Passed Parsing: {parser_info.get('parsed_count', 0)}")
        print(f"  Failed Parsing: {parser_info.get('failed_count', 0)}")

    print("=" * 80)
    return 0 if not schema_failures else 1


def main(args: list[str] | None = None) -> int:
    """CLI entry point dispatching to appropriate command."""
    parser = argparse.ArgumentParser(
        prog="complexity-lens",
        description="ComplexityLens: Hybrid C++ static complexity analysis and dataset tooling.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single C++ file")
    analyze_parser.add_argument("file", help="Path to C++ source file")
    analyze_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    analyze_parser.add_argument("--function", help="Target function name to analyze")

    # dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(
        dest="dataset_command", help="Dataset subcommands"
    )

    report_parser = dataset_subparsers.add_parser("report", help="Generate dataset status report")
    report_parser.add_argument("--dataset-dir", default="datasets", help="Path to dataset root")
    report_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "analyze":
        return run_analyze(
            file_path=parsed_args.file,
            as_json=parsed_args.json,
            function_name=parsed_args.function,
        )
    elif parsed_args.command == "dataset":
        if parsed_args.dataset_command == "report":
            return run_dataset_report(
                dataset_dir=parsed_args.dataset_dir,
                as_json=parsed_args.json,
            )
        else:
            dataset_parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
