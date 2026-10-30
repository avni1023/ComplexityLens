"""cli.py — Command Line Interface for ComplexityLens.

Provides user-facing commands:
- analyze <file.cpp> [--json]: Full analysis pipeline (validation, parser status, IR summary,
  26 features in schema order, static complexity result with evidence and reasons).
- dataset report [--json]: Evaluates dataset records, counting labels, families, UNKNOWN reasons,
  duplicate hashes, metadata failures, and parser-validity status.
- dataset split {propose, validate, freeze, verify}: Manages dataset partitioning lifecycle.
- evaluate cv [--manifest <path>] [--dataset-dir <dir>] [--out <path>] [--json]: Runs 3-fold
  cross-validation on development partition across baseline models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from complexity_lens.complexity.engine import analyze_file
from complexity_lens.dataset import generate_dataset_report
from complexity_lens.evaluation import (
    BaselineError,
    CrossValidationExperimentResult,
    CrossValidationRunnerError,
    DataIsolationError,
    EmptyInputError,
    EvaluationDataLoaderError,
    FoldConfigurationError,
    ManifestVerificationError,
    MetadataValidationError,
    MetricsError,
    SerializationError,
    TestPartitionAccessError,
    run_cross_validation,
    to_deterministic_json,
)
from complexity_lens.features.extractor import extract_features
from complexity_lens.features.schema import FEATURE_NAMES
from complexity_lens.ir.builder import build_ir_from_file
from complexity_lens.ir.model import LoopIR, ProgramIR
from complexity_lens.parsers import ClangParserError
from complexity_lens.split import (
    DEFAULT_SPLIT_SEED,
    FreezeGateError,
    SplitError,
    SplitValidationError,
    freeze_manifest,
    generate_proposal_manifest,
    validate_manifest,
    verify_frozen_manifest,
)
from complexity_lens.validation import ValidationError, validate_source


def _summarize_ir(program: ProgramIR) -> dict[str, Any]:
    """Derive a summary of IR statistics from ProgramIR."""
    num_loops = sum(1 for f in program.functions for item in f.body if isinstance(item, LoopIR))
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

    # 2. Parser phase & IR Summary
    try:
        program_ir = build_ir_from_file(path)
        parser_status = "PARSED"
        parser_error = None
        ir_summary = _summarize_ir(program_ir)
    except ClangParserError as e:
        parser_status = "FAILED"
        parser_error = str(e)
        ir_summary = None
    except Exception as e:
        parser_status = "FAILED"
        parser_error = f"IR builder error: {e}"
        ir_summary = None

    # 3. Feature Extraction
    features_dict = None
    if parser_status == "PARSED" and program_ir is not None:
        try:
            fv = extract_features(program_ir)
            features_dict = fv.to_dict()
        except Exception:
            features_dict = None

    # 4. Static Complexity Result
    try:
        result = analyze_file(path, function_name=function_name)
        result_dict = result.to_dict()
    except Exception as e:
        result_dict = {
            "predicted_class": "UNKNOWN",
            "exactness": "EXACT",
            "evidence": [],
            "reason_codes": ["PARSER_ERROR"],
            "error": str(e),
        }

    if as_json:
        out = {
            "file": str(path),
            "input_validation_status": validation_status,
            "parser_status": parser_status,
            "ir_summary": ir_summary,
            "features": features_dict,
            "result": result_dict,
        }
        if validation_error:
            out["validation_error"] = validation_error
        if parser_error:
            out["parser_error"] = parser_error
        print(json.dumps(out, indent=2))
        return 0

    # Text output format
    print("=" * 80)
    print(f"ComplexityLens Static Analysis: {path.name}")
    print("=" * 80)
    print(f"File Path            : {path}")
    print(f"Input Validation     : {validation_status}")
    print(f"Parser Status        : {parser_status}")
    if parser_error:
        print(f"Parser Error         : {parser_error}")

    if ir_summary:
        print("\n[IR Summary]")
        print(f"  Entry Function     : {ir_summary['entry_function']}")
        print(f"  Total Functions    : {ir_summary['num_functions']}")
        print(f"  Total Loops        : {ir_summary['num_loops']}")
        print(f"  Analysis Complete  : {ir_summary['analysis_complete']}")

    if features_dict:
        print("\n[Features (26 Total)]")
        for fn in FEATURE_NAMES:
            val = features_dict.get(fn, "N/A")
            print(f"  {fn:<30}: {val}")

    print("\n[Complexity Engine Result]")
    print(f"  Predicted Class    : {result_dict.get('predicted_class')}")
    print(f"  Exactness          : {result_dict.get('exactness')}")
    rcs = result_dict.get("reason_codes", [])
    print(f"  Reason Codes       : {', '.join(rcs) if rcs else 'None'}")

    evs = result_dict.get("evidence", [])
    print(f"  Evidence Items ({len(evs)}):")
    for ev in evs:
        print(f"    - [{ev.get('source_entity')}] {ev.get('description')}")

    trace = result_dict.get("rule_trace", [])
    if trace:
        print(f"\n[Rule Trace ({len(trace)} Steps)]")
        for st in trace:
            applied = "APPLIED" if st.get("applied") else "SKIPPED"
            print(f"  - [{applied}] {st.get('rule_id')}: {st.get('explanation')}")

    print("=" * 80)
    return 0


def run_dataset_report(
    dataset_dir: str | Path = "datasets",
    as_json: bool = False,
) -> int:
    """Generate and display dataset validation report."""
    root = Path(dataset_dir)
    if not root.exists():
        err_msg = f"Dataset directory not found: {dataset_dir}"
        if as_json:
            print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    report = generate_dataset_report(dataset_root=root, check_parsing=True)

    if as_json:
        print(json.dumps(report, indent=2))
        return 0 if not report["metadata_schema_failures"] else 1

    # Text report
    print("=" * 80)
    print(f"ComplexityLens Dataset Validation Report: {root}")
    print("=" * 80)
    print(f"Total Programs Listed   : {report['total_programs']}")
    print(f"Total Unique Families   : {report['unique_families']}")

    print("\n[Ground-Truth Label Distribution]")
    for label, count in sorted(report["label_distribution"].items()):
        print(f"  {label:<10}: {count}")

    print("\n[Tag Distribution]")
    for tag, count in sorted(report["tag_distribution"].items()):
        print(f"  {tag:<10}: {count}")

    print("\n[Origin Distribution]")
    if report.get("counts_by_origin"):
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
    print(f"  Duplicate Program IDs : {len(dup_ids)}" + (f" ({dup_ids})" if dup_ids else ""))
    print(
        f"  Duplicate Hashes      : {len(dup_hashes)}" + (f" ({dup_hashes})" if dup_hashes else "")
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


def run_dataset_split_propose(
    dataset_dir: str | Path = "datasets",
    split_groups: str | Path | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
    out: str | Path | None = None,
    as_json: bool = False,
) -> int:
    """Generate and display or save a proposed dataset partition."""
    try:
        manifest = generate_proposal_manifest(
            dataset_root=dataset_dir,
            split_groups_path=split_groups,
            seed=seed,
        )
    except SplitError as e:
        err_msg = f"Split proposal error: {e}"
        if as_json:
            print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    if out:
        out_p = Path(out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(manifest.to_json(), encoding="utf-8")
        if not as_json:
            print(f"Proposal manifest written to: {out_p}")

    if as_json:
        print(manifest.to_json())
        return 0

    print("=" * 80)
    print("ComplexityLens Dataset Partition Proposal")
    print("=" * 80)
    print(f"Seed                   : {manifest.seed}")
    print(f"Status                 : {manifest.status}")
    print(f"Git Commit             : {manifest.git_commit or 'N/A'}")
    base_c = manifest.base_dataset_commit or manifest.git_commit or "N/A"
    print(f"Base Dataset Commit    : {base_c}")
    print(f"Working Tree Dirty     : {manifest.git_dirty}")

    print("\n[Partitions]")
    for part, pinfo in manifest.partitions.items():
        print(f"  {part.upper()} Partition:")
        print(f"    Families ({pinfo['family_count']})  : {', '.join(pinfo['families'])}")
        print(f"    Programs Count     : {pinfo['program_count']}")
        print(f"    Classes            : {pinfo['class_distribution']}")
        print(f"    Tags               : {pinfo['tag_distribution']}")
        print(f"    UNKNOWN Reasons    : {pinfo['unknown_reasons']}")

    print("\n[Cross-Validation Folds in Development]")
    for fold in manifest.cv_folds:
        print(f"  Fold {fold['fold']}:")
        print(f"    Families ({fold['family_count']})  : {', '.join(fold['families'])}")
        print(f"    Programs Count     : {fold['program_count']}")
        print(f"    Classes            : {fold['class_distribution']}")
        print(f"    Tags               : {fold['tag_distribution']}")
        print(f"    UNKNOWN Reasons    : {fold['unknown_reasons']}")

    q_fams = ", ".join(manifest.quarantined_families)
    print(f"\nQuarantined Families ({len(manifest.quarantined_families)}): {q_fams}")
    print("=" * 80)
    return 0


def run_dataset_split_validate(
    manifest_path: str | Path,
    dataset_dir: str | Path = "datasets",
    as_json: bool = False,
) -> int:
    """Validate a split manifest file against dataset integrity constraints."""
    path = Path(manifest_path)
    if not path.exists():
        err_msg = f"Manifest file not found: {manifest_path}"
        if as_json:
            print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    errors = validate_manifest(path, dataset_root=dataset_dir)
    is_valid = len(errors) == 0

    if as_json:
        print(
            json.dumps(
                {
                    "manifest": str(path),
                    "status": "VALID" if is_valid else "INVALID",
                    "errors": errors,
                },
                indent=2,
            )
        )
        return 0 if is_valid else 1

    print("=" * 80)
    print(f"Split Manifest Validation: {path.name}")
    print("=" * 80)
    if is_valid:
        print("Status: VALID (all partition, fold, and coverage constraints satisfied)")
    else:
        print(f"Status: INVALID ({len(errors)} errors found)")
        for err in errors:
            print(f"  - {err}")
    print("=" * 80)
    return 0 if is_valid else 1


def run_dataset_split_freeze(
    proposal_path: str | Path,
    out_path: str | Path,
    dataset_dir: str | Path = "datasets",
    as_json: bool = False,
) -> int:
    """Freeze a proposal manifest after verifying clean git state and writing sidecar hash."""
    try:
        frozen_file = freeze_manifest(
            proposal_path_or_manifest=proposal_path,
            frozen_output_path=out_path,
            dataset_root=dataset_dir,
        )
    except (FreezeGateError, SplitValidationError, SplitError) as e:
        err_msg = f"Freeze gate error: {e}"
        if as_json:
            print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    sidecar = frozen_file.with_name(f"{frozen_file.name}.sha256")
    if as_json:
        print(
            json.dumps(
                {
                    "status": "FROZEN",
                    "frozen_manifest": str(frozen_file),
                    "sidecar_file": str(sidecar),
                },
                indent=2,
            )
        )
        return 0

    print("=" * 80)
    print("Split Manifest Frozen Successfully")
    print("=" * 80)
    print(f"Frozen Manifest : {frozen_file}")
    print(f"Sidecar Checksum: {sidecar}")
    print("=" * 80)
    return 0


def run_dataset_split_verify(
    frozen_path: str | Path,
    dataset_dir: str | Path = "datasets",
    as_json: bool = False,
) -> int:
    """Verify frozen manifest against sidecar checksum and dataset integrity."""
    try:
        res = verify_frozen_manifest(frozen_manifest_path=frozen_path, dataset_root=dataset_dir)
    except (SplitValidationError, SplitError) as e:
        err_msg = f"Verification failed: {e}"
        if as_json:
            print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(res, indent=2))
        return 0

    print("=" * 80)
    print(f"Frozen Split Manifest Verified: {Path(frozen_path).name}")
    print("=" * 80)
    print(f"Status          : {res['status']}")
    print(f"SHA-256 Checksum: {res['sha256']}")
    print("Checksum and all partition/fold integrity constraints verified successfully.")
    print("=" * 80)
    return 0


def _print_cv_summary(
    result: CrossValidationExperimentResult,
    out_path: str | Path | None = None,
) -> None:
    """Print structured, human-readable summary of cross-validation results."""
    print("=" * 80)
    print("ComplexityLens Evaluation: Development Cross-Validation Summary")
    print("=" * 80)
    print(f"Experiment ID        : {result.experiment_id}")
    print(f"Manifest SHA-256     : {result.manifest_sha256}")
    print("Partition            : Development (Frozen Test Partition Strictly Isolated)")
    summary = result.dataset_summary
    prog_count = summary.get("program_count", 159)
    fam_count = summary.get("family_count", 53)
    print(f"Programs Evaluated   : {prog_count} across {fam_count} families")

    folds_info = summary.get("per_fold_counts", {})
    if folds_info:
        fold_str = ", ".join(
            f"F{k.split('_')[-1]}: {v.get('val_programs')}" for k, v in sorted(folds_info.items())
        )
        print(f"Folds                : {len(folds_info)} ({fold_str})")
    else:
        print("Folds                : 3")

    model_names = list(result.models.keys())
    print(f"Models Evaluated     : {len(model_names)} ({', '.join(model_names)})")
    print()
    print("[Development Out-of-Fold (OOF) Metrics]")
    all_warnings: list[str] = []
    for model_id, model_res in sorted(result.models.items()):
        metrics = model_res.combined_oof_metrics
        acc = metrics.accuracy
        f1 = metrics.macro_f1
        print(f"  {model_id}:")
        print(f"    Accuracy         : {acc:.4f}")
        print(f"    Macro F1         : {f1:.4f}")
        if model_res.warnings:
            all_warnings.extend(f"[{model_id}] {w}" for w in model_res.warnings)

    print()
    print(f"Output File          : {out_path if out_path else 'None (in-memory execution)'}")
    if all_warnings:
        print(f"Warnings ({len(all_warnings)})         :")
        for w in all_warnings:
            print(f"  - {w}")
    else:
        print("Warnings (0)         : None")
    print("=" * 80)
    print(
        "* Note: Metrics reflect development cross-validation only, "
        "NOT final test generalization."
    )
    print("=" * 80)


def run_evaluate_cv(
    manifest_path: str | Path = "data/splits/split_manifest_v1.0.frozen.json",
    dataset_dir: str | Path = "datasets",
    out_path: str | Path | None = None,
    as_json: bool = False,
) -> int:
    """Execute 3-fold cross-validation on the development partition and report results."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        err_msg = f"Manifest file not found: {manifest_path}"
        if as_json:
            print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
        else:
            print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    if out_path is not None:
        destination = Path(out_path)
        if not destination.parent.exists():
            err_msg = (
                f"Destination parent directory does not exist: {destination.parent}. "
                "Parent directories must be created prior to execution."
            )
            if as_json:
                print(json.dumps({"status": "ERROR", "error": err_msg}, indent=2))
            else:
                print(f"Error: {err_msg}", file=sys.stderr)
            return 1

    try:
        result = run_cross_validation(
            manifest_path=m_path,
            dataset_root=dataset_dir,
            destination=out_path,
            ensure_parents=False,
        )
    except (
        ManifestVerificationError,
        DataIsolationError,
        TestPartitionAccessError,
        FoldConfigurationError,
        SerializationError,
        FileNotFoundError,
        CrossValidationRunnerError,
        EvaluationDataLoaderError,
        MetricsError,
        MetadataValidationError,
        EmptyInputError,
        BaselineError,
    ) as e:
        if as_json:
            print(json.dumps({"status": "ERROR", "error": str(e)}, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    if as_json:
        print(to_deterministic_json(result.to_dict()).rstrip("\n"))
        return 0

    _print_cv_summary(result, out_path=out_path)
    return 0


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

    # dataset report
    report_parser = dataset_subparsers.add_parser("report", help="Generate dataset status report")
    report_parser.add_argument("--dataset-dir", default="datasets", help="Path to dataset root")
    report_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # dataset split
    split_parser = dataset_subparsers.add_parser(
        "split", help="Dataset partitioning and split management"
    )
    split_subparsers = split_parser.add_subparsers(dest="split_command", help="Split subcommands")

    # dataset split propose
    propose_parser = split_subparsers.add_parser(
        "propose", help="Propose reproducible dataset split"
    )
    propose_parser.add_argument("--dataset-dir", default="datasets", help="Path to dataset root")
    propose_parser.add_argument("--split-groups", default=None, help="Path to split_groups.yaml")
    propose_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED, help="Random seed")
    propose_parser.add_argument("--out", default=None, help="Output file path for proposal JSON")
    propose_parser.add_argument("--json", action="store_true", help="Output JSON to stdout")

    # dataset split validate
    validate_parser = split_subparsers.add_parser("validate", help="Validate a split manifest")
    validate_parser.add_argument("manifest", help="Path to split manifest JSON")
    validate_parser.add_argument("--dataset-dir", default="datasets", help="Path to dataset root")
    validate_parser.add_argument("--json", action="store_true", help="Output JSON")

    # dataset split freeze
    freeze_parser = split_subparsers.add_parser("freeze", help="Freeze a proposal manifest")
    freeze_parser.add_argument("proposal", help="Path to proposal manifest JSON")
    freeze_parser.add_argument("--out", required=True, help="Path to output frozen manifest JSON")
    freeze_parser.add_argument("--dataset-dir", default="datasets", help="Path to dataset root")
    freeze_parser.add_argument("--json", action="store_true", help="Output JSON")

    # dataset split verify
    verify_parser = split_subparsers.add_parser("verify", help="Verify frozen manifest and sidecar")
    verify_parser.add_argument("manifest", help="Path to frozen manifest JSON")
    verify_parser.add_argument("--dataset-dir", default="datasets", help="Path to dataset root")
    verify_parser.add_argument("--json", action="store_true", help="Output JSON")

    # evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluation and benchmarking operations"
    )
    evaluate_subparsers = evaluate_parser.add_subparsers(
        dest="evaluate_command", help="Evaluation subcommands"
    )

    # evaluate cv
    cv_parser = evaluate_subparsers.add_parser(
        "cv", help="Run 3-fold cross-validation on development partition"
    )
    cv_parser.add_argument(
        "--manifest",
        default="data/splits/split_manifest_v1.0.frozen.json",
        help="Path to frozen split manifest JSON",
    )
    cv_parser.add_argument(
        "--dataset-dir",
        default="datasets",
        help="Path to dataset root directory",
    )
    cv_parser.add_argument(
        "--out",
        default=None,
        help="Explicit destination file path for deterministic JSON results",
    )
    cv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON to stdout",
    )

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
        elif parsed_args.dataset_command == "split":
            if parsed_args.split_command == "propose":
                return run_dataset_split_propose(
                    dataset_dir=parsed_args.dataset_dir,
                    split_groups=parsed_args.split_groups,
                    seed=parsed_args.seed,
                    out=parsed_args.out,
                    as_json=parsed_args.json,
                )
            elif parsed_args.split_command == "validate":
                return run_dataset_split_validate(
                    manifest_path=parsed_args.manifest,
                    dataset_dir=parsed_args.dataset_dir,
                    as_json=parsed_args.json,
                )
            elif parsed_args.split_command == "freeze":
                return run_dataset_split_freeze(
                    proposal_path=parsed_args.proposal,
                    out_path=parsed_args.out,
                    dataset_dir=parsed_args.dataset_dir,
                    as_json=parsed_args.json,
                )
            elif parsed_args.split_command == "verify":
                return run_dataset_split_verify(
                    frozen_path=parsed_args.manifest,
                    dataset_dir=parsed_args.dataset_dir,
                    as_json=parsed_args.json,
                )
            else:
                split_parser.print_help()
                return 1
        else:
            dataset_parser.print_help()
            return 1
    elif parsed_args.command == "evaluate":
        if parsed_args.evaluate_command == "cv":
            return run_evaluate_cv(
                manifest_path=parsed_args.manifest,
                dataset_dir=parsed_args.dataset_dir,
                out_path=parsed_args.out,
                as_json=parsed_args.json,
            )
        else:
            evaluate_parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
