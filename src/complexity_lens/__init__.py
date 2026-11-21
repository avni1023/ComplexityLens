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
    ALLOWED_ORIGINS,
    FamilyMetadata,
    ProgramRecord,
    compute_source_sha256,
    generate_dataset_report,
    load_dataset,
)
from complexity_lens.explanation import (
    DetectedConstruct,
    ExplanationReport,
    SourceSpanInfo,
    generate_explanation,
)
from complexity_lens.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureVector,
    extract_features,
)
from complexity_lens.fusion import (
    DEFAULT_ML_THRESHOLD,
    FusionDecisionReason,
    FusionResult,
    FusionSource,
    MLPrediction,
    decide_fusion,
)
from complexity_lens.presentation import (
    COMPLEXITY_DISPLAY_MAP,
    format_complexity_label,
    format_construct_display,
    format_explanation_text,
)
from complexity_lens.split import (
    ALLOWED_PARTITIONS,
    DEFAULT_SPLIT_SEED,
    STRATIFICATION_DEFICIT_WARNING,
    AtomicUnit,
    CardinalityInfeasibilityError,
    FreezeGateError,
    InfeasibleSplitConstraintsError,
    SplitError,
    SplitManifest,
    SplitProgramRecord,
    SplitValidationError,
    StratificationDeficitWarning,
    allocate_development_folds,
    canonical_json_dumps,
    format_stratification_deficit_warning,
    freeze_manifest,
    generate_proposal_manifest,
    load_atomic_units,
    optimize_test_partition,
    validate_manifest,
    verify_cross_partition_similarity,
    verify_frozen_manifest,
)

__version__ = "0.1.0"

__all__ = [
    "ALLOWED_ORIGINS",
    "ALLOWED_PARTITIONS",
    "AnalysisResult",
    "AtomicUnit",
    "COMPLEXITY_DISPLAY_MAP",
    "CardinalityInfeasibilityError",
    "ComplexityClass",
    "DEFAULT_ML_THRESHOLD",
    "DEFAULT_SPLIT_SEED",
    "DetectedConstruct",
    "Exactness",
    "ExplanationReport",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FamilyMetadata",
    "FeatureVector",
    "FreezeGateError",
    "FunctionSelectionError",
    "FusionDecisionReason",
    "FusionResult",
    "FusionSource",
    "InfeasibleSplitConstraintsError",
    "MLPrediction",
    "PipelineResult",
    "ProgramRecord",
    "ReasonCode",
    "RuleTraceEntry",
    "SourceSpanInfo",
    "SplitError",
    "SplitManifest",
    "SplitProgramRecord",
    "SplitValidationError",
    "STRATIFICATION_DEFICIT_WARNING",
    "StratificationDeficitWarning",
    "TaxonomyLabel",
    "allocate_development_folds",
    "analyze_complexity",
    "analyze_file",
    "analyze_source",
    "analyze_source_pipeline",
    "canonical_json_dumps",
    "compute_source_sha256",
    "decide_fusion",
    "extract_features",
    "format_complexity_label",
    "format_construct_display",
    "format_explanation_text",
    "format_stratification_deficit_warning",
    "freeze_manifest",
    "generate_dataset_report",
    "generate_explanation",
    "generate_proposal_manifest",
    "load_atomic_units",
    "load_dataset",
    "optimize_test_partition",
    "run_ml_inference",
    "safe_load_model",
    "validate_manifest",
    "verify_cross_partition_similarity",
    "verify_frozen_manifest",
]


def __getattr__(name: str) -> object:
    if name in (
        "PipelineResult",
        "analyze_source_pipeline",
        "run_ml_inference",
        "safe_load_model",
    ):
        import complexity_lens.pipeline as _pipe

        return getattr(_pipe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
