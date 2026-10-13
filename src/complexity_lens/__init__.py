"""ComplexityLens: Explainable Hybrid Static Analysis and ML System for C++ Complexity Analysis."""

from complexity_lens.complexity import (
    AnalysisResult,
    ComplexityClass,
    Exactness,
    FunctionSelectionError,
    ReasonCode,
    RuleTraceEntry,
    TaxonomyLabel,
    analyze_complexity,
    analyze_file,
    analyze_source,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "ComplexityClass",
    "Exactness",
    "FunctionSelectionError",
    "ReasonCode",
    "RuleTraceEntry",
    "TaxonomyLabel",
    "analyze_complexity",
    "analyze_file",
    "analyze_source",
]
