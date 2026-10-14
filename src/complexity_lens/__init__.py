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
from complexity_lens.dataset import (
    FamilyMetadata,
    ProgramRecord,
    compute_source_sha256,
    generate_dataset_report,
    load_dataset,
)
from complexity_lens.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureVector,
    extract_features,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "ComplexityClass",
    "Exactness",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FamilyMetadata",
    "FeatureVector",
    "FunctionSelectionError",
    "ProgramRecord",
    "ReasonCode",
    "RuleTraceEntry",
    "TaxonomyLabel",
    "analyze_complexity",
    "analyze_file",
    "analyze_source",
    "compute_source_sha256",
    "extract_features",
    "generate_dataset_report",
    "load_dataset",
]
