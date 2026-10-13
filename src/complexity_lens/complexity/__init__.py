"""
complexity package — Deterministic Static Complexity Engine for ComplexityLens.
"""

from complexity_lens.complexity.call_graph import (
    MAX_CALL_DEPTH,
    ArgMapping,
    build_call_graph,
    compute_sccs,
    map_argument,
    substitute_complexity,
)
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
from complexity_lens.complexity.stl_rules import (
    HASH_EXPECTED,
    HASH_STRICT,
    classify_stl_operation,
    validate_cost_model,
)

__all__ = [
    # Engine
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
    # STL rules (Stage 1)
    "HASH_EXPECTED",
    "HASH_STRICT",
    "classify_stl_operation",
    "validate_cost_model",
    # Call graph (Stage 2)
    "ArgMapping",
    "MAX_CALL_DEPTH",
    "build_call_graph",
    "compute_sccs",
    "map_argument",
    "substitute_complexity",
]
