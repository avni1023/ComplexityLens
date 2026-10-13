"""
complexity package — Deterministic Static Complexity Engine for ComplexityLens.
"""

from complexity_lens.complexity.engine import (
    ComplexityEngine,
    analyze_complexity,
    analyze_file,
    analyze_source,
    select_function,
)
from complexity_lens.complexity.models import (
    AnalysisResult,
    ComplexityClass,
    Exactness,
    FunctionSelectionError,
    ReasonCode,
    RuleTraceEntry,
    TaxonomyLabel,
)
from complexity_lens.complexity.rules import (
    classify_loop,
    compare_dominance,
    dominant,
    is_call_allowlisted,
    multiply,
)

__all__ = [
    "AnalysisResult",
    "ComplexityClass",
    "ComplexityEngine",
    "Exactness",
    "FunctionSelectionError",
    "ReasonCode",
    "RuleTraceEntry",
    "TaxonomyLabel",
    "analyze_complexity",
    "analyze_file",
    "analyze_source",
    "classify_loop",
    "compare_dominance",
    "dominant",
    "is_call_allowlisted",
    "multiply",
    "select_function",
]
