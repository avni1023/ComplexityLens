"""
metrics.py — Evaluation metrics engine and deterministic serialization for ComplexityLens.

Provides:
- Strict canonical 8-class classification metrics:
  * Accuracy across all supplied examples.
  * Macro precision, macro recall, and macro F1 averaged strictly over classes
    with positive true support (support > 0).
  * Per-class precision, recall, F1, and support across all eight canonical classes.
  * Canonical 8x8 confusion matrix where rows are true classes and columns are
    predicted classes in exact canonical taxonomy order.
- Documented Zero-Division Policy:
  * Zero true support (support == 0): per-class precision, recall, and F1 are
    defined as None (serialized to JSON null) and excluded from macro averages.
  * Positive true support (support > 0) with zero predictions (TP + FP == 0):
    per-class precision is defined as 0.0, recall is 0.0, and F1 is 0.0.
- Strict input validation:
  * Rejection of empty input sequences.
  * Rejection of mismatched input lengths.
  * Rejection of unknown labels not in the canonical 8-class taxonomy.
- UNKNOWN class treated as an ordinary canonical label (index 7), not an abstention state.
- Deterministic JSON serialization:
  * Consistent sorted dictionary keys.
  * Stable indentation and trailing newline policy.
  * Strict rejection of non-standard JSON values (NaN, +Infinity, -Infinity).
  * Serialization of None as JSON null.
  * Byte-identical output across repeated invocations.
- Typed EvaluationMetadata contract with structural and invariant validation.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
)

CANONICAL_CLASS_SET: frozenset[str] = frozenset(CANONICAL_CLASS_ORDER)
CANONICAL_CLASS_TO_INDEX: dict[str, int] = {
    cls_name: idx for idx, cls_name in enumerate(CANONICAL_CLASS_ORDER)
}
NUM_CANONICAL_CLASSES: int = len(CANONICAL_CLASS_ORDER)  # 8


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------


class MetricsError(Exception):
    """Base exception for evaluation metrics and serialization errors."""


class InvalidLabelsError(MetricsError, ValueError):
    """Raised when label sequences contain unknown classes not in the canonical taxonomy."""


class EmptyInputError(MetricsError, ValueError):
    """Raised when label sequences are empty."""


class MismatchedLengthsError(MetricsError, ValueError):
    """Raised when true and predicted label sequences have unequal lengths."""


class SerializationError(MetricsError, ValueError):
    """Raised when deterministic JSON serialization encounters invalid data (e.g. NaN/Inf)."""


class MetadataValidationError(MetricsError, ValueError):
    """Raised when evaluation metadata violates schema or invariant requirements."""


# ---------------------------------------------------------------------------
# Metrics Data Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PerClassMetrics:
    """Classification metrics for a single canonical complexity class.

    Zero-Division Contract:
    - If support == 0: precision, recall, and F1 are None (JSON null).
    - If support > 0 and predicted count == 0: precision is 0.0, recall is 0.0, F1 is 0.0.
    """

    class_name: str
    precision: float | None
    recall: float | None
    f1: float | None
    support: int

    def to_dict(self) -> dict[str, Any]:
        """Convert per-class metrics to a JSON-compatible dictionary."""
        return {
            "class_name": self.class_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclasses.dataclass(frozen=True)
class EvaluationMetrics:
    """Comprehensive multi-class evaluation metrics in canonical order."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: list[list[int]]
    canonical_classes: tuple[str, ...] = CANONICAL_CLASS_ORDER

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-compatible dictionary with canonical structure."""
        return {
            "accuracy": self.accuracy,
            "canonical_classes": list(self.canonical_classes),
            "confusion_matrix": self.confusion_matrix,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
        }


# ---------------------------------------------------------------------------
# Metrics Computation Engine
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    zero_division: float = 0.0,
) -> EvaluationMetrics:
    """Compute classification metrics and canonical 8x8 confusion matrix.

    Parameters
    ----------
    y_true:
        Sequence of ground-truth canonical class labels.
    y_pred:
        Sequence of predicted canonical class labels.
    zero_division:
        Value assigned to precision/F1 when a class has positive true support
        (support > 0) but zero predictions (TP + FP == 0). Defaults to 0.0.

    Returns
    -------
    EvaluationMetrics
        Structured evaluation metrics container.

    Raises
    ------
    EmptyInputError
        If y_true or y_pred is empty.
    MismatchedLengthsError
        If len(y_true) != len(y_pred).
    InvalidLabelsError
        If any label in y_true or y_pred is not a canonical class.
    """
    n_true = len(y_true)
    n_pred = len(y_pred)

    if n_true == 0:
        raise EmptyInputError("Cannot compute evaluation metrics on empty label sequences.")
    if n_true != n_pred:
        raise MismatchedLengthsError(
            f"Mismatched label sequence lengths: len(y_true)={n_true} != len(y_pred)={n_pred}."
        )

    # Validate label vocabulary against canonical taxonomy
    invalid_true = sorted({str(lbl) for lbl in y_true if lbl not in CANONICAL_CLASS_SET})
    invalid_pred = sorted({str(lbl) for lbl in y_pred if lbl not in CANONICAL_CLASS_SET})
    if invalid_true or invalid_pred:
        err_parts: list[str] = []
        if invalid_true:
            err_parts.append(f"invalid true labels: {invalid_true}")
        if invalid_pred:
            err_parts.append(f"invalid predicted labels: {invalid_pred}")
        raise InvalidLabelsError(
            f"Encountered unknown labels not in canonical class order: {'; '.join(err_parts)}."
        )

    # Construct canonical 8x8 confusion matrix: matrix[true_idx][pred_idx]
    matrix: list[list[int]] = [[0] * NUM_CANONICAL_CLASSES for _ in range(NUM_CANONICAL_CLASSES)]
    for yt, yp in zip(y_true, y_pred):
        row = CANONICAL_CLASS_TO_INDEX[yt]
        col = CANONICAL_CLASS_TO_INDEX[yp]
        matrix[row][col] += 1

    # Accuracy over all supplied examples
    total_samples = n_true
    correct_samples = sum(matrix[i][i] for i in range(NUM_CANONICAL_CLASSES))
    accuracy = correct_samples / total_samples

    # Per-class calculations
    per_class_results: dict[str, PerClassMetrics] = {}
    classes_with_support: list[str] = []

    for idx, class_name in enumerate(CANONICAL_CLASS_ORDER):
        # Support = sum across row (all true instances of class_name)
        support = sum(matrix[idx][col] for col in range(NUM_CANONICAL_CLASSES))
        # Total predicted = sum across column (all predictions of class_name)
        pred_count = sum(matrix[row][idx] for row in range(NUM_CANONICAL_CLASSES))
        true_positives = matrix[idx][idx]

        if support == 0:
            # Documented policy: zero true support yields None (JSON null)
            prec: float | None = None
            rec: float | None = None
            f1: float | None = None
        else:
            classes_with_support.append(class_name)
            rec = true_positives / support

            if pred_count == 0:
                # Documented policy: positive support but 0 predictions -> zero_division (0.0)
                prec = float(zero_division)
                f1 = 0.0
            else:
                prec = true_positives / pred_count
                if prec + rec > 0.0:
                    f1 = (2.0 * prec * rec) / (prec + rec)
                else:
                    f1 = 0.0

        per_class_results[class_name] = PerClassMetrics(
            class_name=class_name,
            precision=prec,
            recall=rec,
            f1=f1,
            support=support,
        )

    # Macro averages: strictly average classes with positive true support
    if classes_with_support:
        prec_sum = sum(
            per_class_results[c].precision  # type: ignore[arg-type]
            for c in classes_with_support
        )
        rec_sum = sum(
            per_class_results[c].recall  # type: ignore[arg-type]
            for c in classes_with_support
        )
        f1_sum = sum(
            per_class_results[c].f1  # type: ignore[arg-type]
            for c in classes_with_support
        )
        num_supported = len(classes_with_support)
        macro_precision = prec_sum / num_supported
        macro_recall = rec_sum / num_supported
        macro_f1 = f1_sum / num_supported
    else:  # pragma: no cover — impossible when total_samples > 0 and labels are valid
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    return EvaluationMetrics(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=per_class_results,
        confusion_matrix=matrix,
        canonical_classes=CANONICAL_CLASS_ORDER,
    )


# Convenience aliases
compute_metrics = compute_classification_metrics
calculate_metrics = compute_classification_metrics


# ---------------------------------------------------------------------------
# Deterministic Serialization
# ---------------------------------------------------------------------------


def _to_json_compatible(obj: Any) -> Any:
    """Recursively convert structured results into standard JSON-compatible Python types.

    Rejects non-compliant float values (NaN, +Infinity, -Infinity).
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise SerializationError(
                f"Float value {obj} is not compliant with standard JSON (NaN/Inf rejected)."
            )
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _to_json_compatible(obj.to_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_json_compatible(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        sorted_pairs = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return {str(k): _to_json_compatible(v) for k, v in sorted_pairs}
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(item) for item in obj]
    if isinstance(obj, Path):
        # Normalize paths to relative POSIX style; avoid embedding local drive letters
        return str(obj).replace("\\", "/")

    # Optional numpy scalar conversions if numpy is in sys.modules
    if "numpy" in sys.modules:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                raise SerializationError(
                    f"NumPy float {obj} is not compliant with standard JSON (NaN/Inf rejected)."
                )
            return val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [_to_json_compatible(x) for x in obj.tolist()]

    raise SerializationError(
        f"Unsupported object type for deterministic serialization: {type(obj).__name__}."
    )


def to_deterministic_json(obj: Any, *, indent: int = 2) -> str:
    """Serialize *obj* to a deterministic UTF-8 JSON string.

    Guarantees:
    - Object keys are sorted consistently at all nesting levels.
    - Stable indentation and trailing newline (LF).
    - None serializes to standard JSON null.
    - Strict rejection of NaN, +Infinity, and -Infinity.
    - Repeated serialization of identical logical input produces byte-identical output.
    """
    clean_obj = _to_json_compatible(obj)
    try:
        serialized = json.dumps(
            clean_obj,
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (ValueError, TypeError) as e:
        raise SerializationError(f"Deterministic JSON serialization failed: {e}") from e
    return serialized + "\n"


def write_deterministic_json(
    obj: Any,
    destination: str | Path,
    *,
    indent: int = 2,
    ensure_parents: bool = False,
) -> Path:
    """Write *obj* to *destination* file as deterministic UTF-8 JSON.

    Parameters
    ----------
    obj:
        The object to serialize.
    destination:
        Explicit target file path. No implicit caches or default directories are created.
    indent:
        Indentation spaces (default: 2).
    ensure_parents:
        If True, parent directories are created if missing. Defaults to False
        to prevent unintended directory creation.

    Returns
    -------
    Path
        The destination Path.
    """
    path = Path(destination)
    if ensure_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise FileNotFoundError(
            f"Destination parent directory does not exist: {path.parent}. "
            "Pass ensure_parents=True if directory creation is intended."
        )

    json_str = to_deterministic_json(obj, indent=indent)
    path.write_bytes(json_str.encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# Metadata Contract
# ---------------------------------------------------------------------------

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclasses.dataclass(frozen=True)
class EvaluationMetadata:
    """Documented, machine-readable metadata contract for evaluation runs."""

    schema_version: str
    model_id: str
    fixed_hyperparameters: dict[str, Any]
    canonical_feature_schema: list[str]
    canonical_class_ordering: list[str]
    manifest_sha256: str
    fold_assignment_hash: str
    git_commit: str
    git_dirty: bool
    library_versions: dict[str, str]
    sample_count: int
    family_count: int
    training_class_distribution: dict[str, int]
    missing_class_details: dict[str, Any]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return {
            "canonical_class_ordering": list(self.canonical_class_ordering),
            "canonical_feature_schema": list(self.canonical_feature_schema),
            "family_count": self.family_count,
            "fixed_hyperparameters": dict(self.fixed_hyperparameters),
            "fold_assignment_hash": self.fold_assignment_hash,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "library_versions": dict(self.library_versions),
            "limitations": list(self.limitations),
            "manifest_sha256": self.manifest_sha256,
            "missing_class_details": dict(self.missing_class_details),
            "model_id": self.model_id,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "training_class_distribution": dict(self.training_class_distribution),
        }

    def validate(self) -> None:
        """Validate metadata invariants, raising MetadataValidationError if violated."""
        validate_evaluation_metadata(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationMetadata:
        """Validate and construct an EvaluationMetadata instance from a dictionary."""
        validated = validate_evaluation_metadata(data)
        return cls(
            schema_version=validated["schema_version"],
            model_id=validated["model_id"],
            fixed_hyperparameters=validated["fixed_hyperparameters"],
            canonical_feature_schema=validated["canonical_feature_schema"],
            canonical_class_ordering=validated["canonical_class_ordering"],
            manifest_sha256=validated["manifest_sha256"],
            fold_assignment_hash=validated["fold_assignment_hash"],
            git_commit=validated["git_commit"],
            git_dirty=validated["git_dirty"],
            library_versions=validated["library_versions"],
            sample_count=validated["sample_count"],
            family_count=validated["family_count"],
            training_class_distribution=validated["training_class_distribution"],
            missing_class_details=validated["missing_class_details"],
            limitations=validated["limitations"],
        )


def validate_evaluation_metadata(data: dict[str, Any] | EvaluationMetadata) -> dict[str, Any]:
    """Validate that *data* satisfies all required evaluation metadata contracts.

    Returns the validated dictionary if valid; raises MetadataValidationError otherwise.
    """
    if isinstance(data, EvaluationMetadata):
        d = data.to_dict()
    elif isinstance(data, dict):
        d = dict(data)
    else:
        raise MetadataValidationError(
            f"Evaluation metadata must be a dict or EvaluationMetadata, got {type(data).__name__}."
        )

    required_fields = (
        "schema_version",
        "model_id",
        "fixed_hyperparameters",
        "canonical_feature_schema",
        "canonical_class_ordering",
        "manifest_sha256",
        "fold_assignment_hash",
        "git_commit",
        "git_dirty",
        "library_versions",
        "sample_count",
        "family_count",
        "training_class_distribution",
        "missing_class_details",
        "limitations",
    )
    missing = [f for f in required_fields if f not in d]
    if missing:
        raise MetadataValidationError(f"Evaluation metadata missing required fields: {missing}.")

    # schema_version
    if not isinstance(d["schema_version"], str) or not d["schema_version"].strip():
        raise MetadataValidationError("Metadata 'schema_version' must be a non-empty string.")

    # model_id
    if not isinstance(d["model_id"], str) or not d["model_id"].strip():
        raise MetadataValidationError("Metadata 'model_id' must be a non-empty string.")

    # fixed_hyperparameters
    if not isinstance(d["fixed_hyperparameters"], dict):
        raise MetadataValidationError("Metadata 'fixed_hyperparameters' must be a dict.")

    # canonical_feature_schema
    feat_schema = d["canonical_feature_schema"]
    if not isinstance(feat_schema, (list, tuple)) or (
        list(feat_schema) != list(CANONICAL_FEATURE_NAMES)
    ):
        raise MetadataValidationError(
            "Metadata 'canonical_feature_schema' must match the authoritative 26-feature schema."
        )

    # canonical_class_ordering
    class_order = d["canonical_class_ordering"]
    if not isinstance(class_order, (list, tuple)) or (
        list(class_order) != list(CANONICAL_CLASS_ORDER)
    ):
        raise MetadataValidationError(
            f"Metadata 'canonical_class_ordering' must match canonical order "
            f"{list(CANONICAL_CLASS_ORDER)}."
        )

    # manifest_sha256
    manifest_hash = str(d["manifest_sha256"]).strip()
    if not _SHA256_HEX_RE.match(manifest_hash):
        raise MetadataValidationError(
            f"Metadata 'manifest_sha256' must be a valid 64-character hex SHA-256, "
            f"got {manifest_hash!r}."
        )

    # fold_assignment_hash
    fold_hash = str(d["fold_assignment_hash"]).strip()
    if not fold_hash:
        raise MetadataValidationError("Metadata 'fold_assignment_hash' must be a non-empty string.")

    # git_commit
    commit = str(d["git_commit"]).strip()
    if not commit:
        raise MetadataValidationError("Metadata 'git_commit' must be a non-empty string.")

    # git_dirty
    if type(d["git_dirty"]) is not bool:
        raise MetadataValidationError("Metadata 'git_dirty' must be a boolean.")

    # library_versions
    lib_vers = d["library_versions"]
    if not isinstance(lib_vers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in lib_vers.items()
    ):
        raise MetadataValidationError("Metadata 'library_versions' must be a dict[str, str].")

    # sample_count & family_count
    sample_count = d["sample_count"]
    if type(sample_count) is not int or sample_count < 0:
        raise MetadataValidationError("Metadata 'sample_count' must be a non-negative integer.")

    family_count = d["family_count"]
    if type(family_count) is not int or family_count < 0:
        raise MetadataValidationError("Metadata 'family_count' must be a non-negative integer.")

    # training_class_distribution
    class_dist = d["training_class_distribution"]
    if not isinstance(class_dist, dict):
        raise MetadataValidationError("Metadata 'training_class_distribution' must be a dict.")
    for k, v in class_dist.items():
        if k not in CANONICAL_CLASS_SET or type(v) is not int or v < 0:
            raise MetadataValidationError(
                f"Invalid training class distribution entry {k!r}: {v!r}. "
                "Must be canonical class -> int >= 0."
            )

    # missing_class_details
    if not isinstance(d["missing_class_details"], dict):
        raise MetadataValidationError("Metadata 'missing_class_details' must be a dict.")

    # limitations
    limitations = d["limitations"]
    if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
        raise MetadataValidationError("Metadata 'limitations' must be a list of strings.")

    return d
