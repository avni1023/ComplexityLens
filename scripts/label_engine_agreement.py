#!/usr/bin/env python3
"""scripts/label_engine_agreement.py — Read-only engine agreement diagnostic.

This diagnostic tool compares independently authored ground-truth dataset labels against
the predictions produced by the ComplexityLens deterministic static analysis engine.

MANDATORY INTEGRITY REQUIREMENT:
Re-check the rationale; never edit a label to match the engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src directory to module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from complexity_lens.complexity.engine import analyze_source
from complexity_lens.dataset import load_dataset

EXACT_WARNING: str = "Re-check the rationale; never edit a label to match the engine"


def run_agreement_diagnostic(dataset_root: str | Path = "datasets") -> int:
    """Run read-only comparison of ground truth labels vs engine results.

    Prints diagnostic table and summary statistics.
    Never modifies source files, metadata, labels, or commits.
    """
    print("=" * 80)
    print("ComplexityLens — Read-Only Ground Truth vs Engine Agreement Diagnostic")
    print("=" * 80)
    print(f"\nWARNING: {EXACT_WARNING}\n")

    records, families, errors = load_dataset(dataset_root)
    if errors:
        print("Dataset Schema/Integrity Errors:")
        for err in errors:
            print(f"  - {err}")
        print("\nAborting diagnostic due to dataset errors.")
        return 1

    if not records:
        print("No program records found in dataset.")
        return 1

    header = (
        f"{'Program ID':<34} {'Family ID':<32} {'Expected':<10} "
        f"{'Engine':<10} {'Agreement':<10} {'Diagnostics'}"
    )
    print(header)
    print("-" * 120)

    matches = 0
    disagreements = 0

    for r in records:
        engine_label = "ERROR"
        diagnostics = ""
        try:
            try:
                res = analyze_source(r.source, filename=f"{r.program_id}.cpp")
            except Exception as fe:
                # If ambiguous function selection, attempt with the last defined function (caller)
                if "Ambiguous function selection" in str(fe):
                    from complexity_lens.ir.builder import build_ir_from_source
                    ir_prog = build_ir_from_source(r.source, filename=f"{r.program_id}.cpp")
                    if ir_prog.functions:
                        res = analyze_source(
                            r.source,
                            filename=f"{r.program_id}.cpp",
                            function_name=ir_prog.functions[-1].name,
                        )
                    else:
                        raise fe
                else:
                    raise fe

            engine_label = res.taxonomy_label
            if res.reason_codes:
                diagnostics = ",".join(res.reason_codes[:3])
            elif res.notes:
                diagnostics = ",".join(res.notes[:2])
        except Exception as e:
            engine_label = "ERROR"
            diagnostics = str(e)[:45]

        agrees = (r.expected_label == engine_label)
        if agrees:
            matches += 1
            status = "MATCH"
        else:
            disagreements += 1
            status = "DISAGREE"

        line = (
            f"{r.program_id:<34} {r.family_id:<32} {r.expected_label:<10} "
            f"{engine_label:<10} {status:<10} {diagnostics}"
        )
        print(line)

    print("-" * 120)
    total = len(records)
    pct = (matches / total) * 100 if total > 0 else 0.0
    print("\nSummary:")
    print(f"  Total records evaluated:  {total}")
    print(f"  Agreements (matches):     {matches} ({pct:.1f}%)")
    print(f"  Disagreements:            {disagreements} ({100.0 - pct:.1f}%)")
    print(f"\nREMINDER: {EXACT_WARNING}\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only diagnostic comparing dataset ground truth to ComplexityLens engine output."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets",
        help="Path to dataset root directory (default: datasets)",
    )
    args = parser.parse_args()
    exit_code = run_agreement_diagnostic(args.dataset_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
