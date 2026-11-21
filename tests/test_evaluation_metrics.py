"""
tests/test_evaluation_metrics.py — Tests for evaluation metrics and serialization.

Covers:
1. Accuracy and macro metrics on a known example.
2. Correct canonical 8x8 confusion matrix.
3. Correct per-class support.
4. Missing true classes appear in per-class results with None metrics.
5. Macro averages exclude classes with zero true support.
6. A class with positive support but no predictions follows the documented zero-division policy.
7. Unknown true or predicted labels are rejected.
8. Mismatched input lengths are rejected.
9. Empty inputs are rejected.
10. UNKNOWN remains an ordinary canonical class, not an abstention state.
11. Repeated serialization produces byte-identical output.
12. JSON keys are deterministically ordered.
13. None becomes JSON null.
14. NaN and Infinity are rejected.
15. Metadata validation accepts valid metadata and rejects missing or invalid required fields.
16. No serializer call implicitly creates an output directory or persistent cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
)
from complexity_lens.evaluation.metrics import (
    CANONICAL_CLASS_TO_INDEX,
    NUM_CANONICAL_CLASSES,
    EmptyInputError,
    EvaluationMetadata,
    EvaluationMetrics,
    InvalidLabelsError,
    MetadataValidationError,
    MismatchedLengthsError,
    SerializationError,
    compute_classification_metrics,
    to_deterministic_json,
    validate_evaluation_metadata,
    write_deterministic_json,
)

# ---------------------------------------------------------------------------
# Test Fixtures & Sample Data
# ---------------------------------------------------------------------------


def _valid_metadata_dict() -> dict:
    """Return a completely valid metadata dictionary satisfying all contracts."""
    return {
        "schema_version": "1.0",
        "model_id": "test_baseline_model_v1",
        "fixed_hyperparameters": {"random_state": 42, "max_depth": 5},
        "canonical_feature_schema": list(CANONICAL_FEATURE_NAMES),
        "canonical_class_ordering": list(CANONICAL_CLASS_ORDER),
        "manifest_sha256": "0b382938897af0f43de4ddd29b0fdf0306d2f33fbe73ad1c7876d7410d973cd4",
        "fold_assignment_hash": "a" * 64,
        "git_commit": "a1e2ae0bb410ee1ae537155bbf71f98f5b2255b4",
        "git_dirty": False,
        "library_versions": {"scikit-learn": "1.9.1", "python": "3.13.14"},
        "sample_count": 159,
        "family_count": 53,
        "training_class_distribution": {cls_name: 20 for cls_name in CANONICAL_CLASS_ORDER},
        "missing_class_details": {},
        "limitations": ["Development partition only; zero test exposure."],
    }


# ---------------------------------------------------------------------------
# 1. Known Example Accuracy and Macro Metrics
# ---------------------------------------------------------------------------


def test_known_example_accuracy_and_macro_metrics():
    """Verify metrics on a precisely computed synthetic example.

    y_true: ["O1", "O1", "ON", "ONLOGN"]
    y_pred: ["O1", "ON", "ON", "OEXP"]

    Class analysis:
    - O1: support=2, TP=1, FP=0, pred=1 -> P=1.0, R=0.5, F1=2/3
    - ON: support=1, TP=1, FP=1, pred=2 -> P=0.5, R=1.0, F1=2/3
    - ONLOGN: support=1, TP=0, FP=0, pred=0 -> P=0.0 (zero-div), R=0.0, F1=0.0
    - Other 5 classes: support=0 -> P=None, R=None, F1=None

    Accuracy: 2/4 = 0.5
    Classes with positive support: O1, ON, ONLOGN (3 classes)
    macro_precision = (1.0 + 0.5 + 0.0) / 3 = 1.5 / 3 = 0.5
    macro_recall = (0.5 + 1.0 + 0.0) / 3 = 1.5 / 3 = 0.5
    macro_f1 = (2/3 + 2/3 + 0.0) / 3 = 4/9 ≈ 0.4444444444444444
    """
    y_true = ["O1", "O1", "ON", "ONLOGN"]
    y_pred = ["O1", "ON", "ON", "OEXP"]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_precision == pytest.approx(0.5)
    assert metrics.macro_recall == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(4.0 / 9.0)


# ---------------------------------------------------------------------------
# 2. Canonical 8x8 Confusion Matrix
# ---------------------------------------------------------------------------


def test_canonical_8x8_confusion_matrix_shape_and_values():
    """Verify confusion matrix is always exactly 8x8 in canonical class order."""
    y_true = ["O1", "O1", "ON", "ONLOGN", "UNKNOWN"]
    y_pred = ["O1", "ON", "ON", "OEXP", "UNKNOWN"]

    metrics = compute_classification_metrics(y_true, y_pred)
    cm = metrics.confusion_matrix

    assert len(cm) == NUM_CANONICAL_CLASSES
    for row in cm:
        assert len(row) == NUM_CANONICAL_CLASSES
        assert all(isinstance(val, int) for val in row)

    # Row 0 (O1): 1 predicted O1 (col 0), 1 predicted ON (col 2)
    assert cm[0][0] == 1
    assert cm[0][2] == 1
    assert sum(cm[0]) == 2

    # Row 2 (ON): 1 predicted ON (col 2)
    assert cm[2][2] == 1
    assert sum(cm[2]) == 1

    # Row 3 (ONLOGN): 1 predicted OEXP (col 6)
    assert cm[3][6] == 1
    assert sum(cm[3]) == 1

    # Row 7 (UNKNOWN): 1 predicted UNKNOWN (col 7)
    assert cm[7][7] == 1
    assert sum(cm[7]) == 1

    # Rows with zero support must have all zeros
    for absent_idx in [1, 4, 5, 6]:
        assert sum(cm[absent_idx]) == 0


# ---------------------------------------------------------------------------
# 3. Correct Per-Class Support
# ---------------------------------------------------------------------------


def test_correct_per_class_support():
    """Verify support matches exact frequency in y_true for every canonical class."""
    y_true = [
        "O1", "O1", "O1",
        "OLOGN",
        "ON", "ON",
        "ONLOGN",
        "ON2",
        "ON3", "ON3", "ON3", "ON3",
        "OEXP",
        "UNKNOWN", "UNKNOWN",
    ]
    # Prediction same as true to focus on support
    y_pred = list(y_true)

    metrics = compute_classification_metrics(y_true, y_pred)

    expected_support = {
        "O1": 3,
        "OLOGN": 1,
        "ON": 2,
        "ONLOGN": 1,
        "ON2": 1,
        "ON3": 4,
        "OEXP": 1,
        "UNKNOWN": 2,
    }
    for class_name, expected_cnt in expected_support.items():
        assert metrics.per_class[class_name].support == expected_cnt


# ---------------------------------------------------------------------------
# 4 & 5. Missing True Classes Yield None and Excluded from Macro
# ---------------------------------------------------------------------------


def test_missing_true_classes_yield_none_metrics_and_excluded_from_macro():
    """Verify classes with zero support have None metrics and do not dilute macro averages."""
    # Only O1 and ON have true instances; other 6 classes have support == 0
    y_true = ["O1", "O1", "ON"]
    y_pred = ["O1", "O1", "ON"]

    metrics = compute_classification_metrics(y_true, y_pred)

    # O1 and ON have perfect precision/recall/f1
    assert metrics.per_class["O1"].precision == pytest.approx(1.0)
    assert metrics.per_class["O1"].recall == pytest.approx(1.0)
    assert metrics.per_class["O1"].f1 == pytest.approx(1.0)
    assert metrics.per_class["O1"].support == 2

    assert metrics.per_class["ON"].precision == pytest.approx(1.0)
    assert metrics.per_class["ON"].recall == pytest.approx(1.0)
    assert metrics.per_class["ON"].f1 == pytest.approx(1.0)
    assert metrics.per_class["ON"].support == 1

    # Absent classes must have None for metrics and 0 for support
    for cls_name in ["OLOGN", "ONLOGN", "ON2", "ON3", "OEXP", "UNKNOWN"]:
        m = metrics.per_class[cls_name]
        assert m.support == 0
        assert m.precision is None
        assert m.recall is None
        assert m.f1 is None

    # Macro averages must be exactly 1.0 (average of O1 and ON only, not divided by 8)
    assert metrics.macro_precision == pytest.approx(1.0)
    assert metrics.macro_recall == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Zero-Division Policy: Positive Support with Zero Predictions
# ---------------------------------------------------------------------------


def test_zero_division_policy_positive_support_zero_predictions():
    """Verify class with support > 0 but 0 predictions gets precision=0.0, recall=0.0, f1=0.0."""
    y_true = ["O1", "O1", "ON"]
    # All predicted as O1 -> ON has positive support (1), but predicted count == 0
    y_pred = ["O1", "O1", "O1"]

    metrics = compute_classification_metrics(y_true, y_pred)

    on_metrics = metrics.per_class["ON"]
    assert on_metrics.support == 1
    assert on_metrics.precision == 0.0
    assert on_metrics.recall == 0.0
    assert on_metrics.f1 == 0.0

    # Macro metrics must include ON
    # O1: P = 2/3, R = 1.0, F1 = 2 * (2/3) * 1 / (5/3) = 4/5 = 0.8
    # ON: P = 0.0, R = 0.0, F1 = 0.0
    # Macro P = (2/3 + 0.0) / 2 = 1/3
    # Macro R = (1.0 + 0.0) / 2 = 0.5
    # Macro F1 = (0.8 + 0.0) / 2 = 0.4
    assert metrics.macro_precision == pytest.approx(1.0 / 3.0)
    assert metrics.macro_recall == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(0.4)


def test_custom_zero_division_parameter():
    """Verify custom zero_division parameter is respected when provided."""
    y_true = ["O1", "ON"]
    y_pred = ["O1", "O1"]  # ON has support=1, pred=0

    metrics = compute_classification_metrics(y_true, y_pred, zero_division=0.5)
    assert metrics.per_class["ON"].precision == 0.5


# ---------------------------------------------------------------------------
# 7. Rejection of Unknown Labels
# ---------------------------------------------------------------------------


def test_unknown_labels_rejected():
    """Verify unknown labels in either y_true or y_pred raise InvalidLabelsError."""
    with pytest.raises(InvalidLabelsError, match="Encountered unknown labels"):
        compute_classification_metrics(["O1", "INVALID_CLASS"], ["O1", "O1"])

    with pytest.raises(InvalidLabelsError, match="Encountered unknown labels"):
        compute_classification_metrics(["O1", "O1"], ["O1", "NON_CANONICAL"])


# ---------------------------------------------------------------------------
# 8. Rejection of Mismatched Lengths
# ---------------------------------------------------------------------------


def test_mismatched_lengths_rejected():
    """Verify mismatched input lengths raise MismatchedLengthsError."""
    with pytest.raises(MismatchedLengthsError, match="Mismatched label sequence lengths"):
        compute_classification_metrics(["O1", "ON"], ["O1"])


# ---------------------------------------------------------------------------
# 9. Rejection of Empty Inputs
# ---------------------------------------------------------------------------


def test_empty_inputs_rejected():
    """Verify empty input sequences raise EmptyInputError."""
    with pytest.raises(EmptyInputError, match="empty label sequences"):
        compute_classification_metrics([], [])


# ---------------------------------------------------------------------------
# 10. UNKNOWN is a Canonical Class, Not Abstention
# ---------------------------------------------------------------------------


def test_unknown_class_is_treated_as_canonical_class():
    """Verify UNKNOWN has its own row/col (index 7), support, and metrics like any other class."""
    y_true = ["UNKNOWN", "UNKNOWN", "O1"]
    y_pred = ["UNKNOWN", "O1", "O1"]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert CANONICAL_CLASS_TO_INDEX["UNKNOWN"] == 7
    un_metrics = metrics.per_class["UNKNOWN"]
    assert un_metrics.class_name == "UNKNOWN"
    assert un_metrics.support == 2
    # UNKNOWN: 1 TP, 1 pred -> P = 1.0, R = 1/2 = 0.5, F1 = 2/3
    assert un_metrics.precision == pytest.approx(1.0)
    assert un_metrics.recall == pytest.approx(0.5)
    assert un_metrics.f1 == pytest.approx(2.0 / 3.0)

    # UNKNOWN appears in confusion matrix at row 7, col 7
    assert metrics.confusion_matrix[7][7] == 1
    assert metrics.confusion_matrix[7][0] == 1  # 1 true UNKNOWN predicted as O1


# ---------------------------------------------------------------------------
# 11 & 12. Deterministic Serialization & Sorted Keys
# ---------------------------------------------------------------------------


def test_repeated_serialization_is_byte_identical_with_sorted_keys():
    """Verify serialization is deterministic, sorted by key, and byte-identical."""
    y_true = ["O1", "ON", "UNKNOWN"]
    y_pred = ["O1", "ON2", "UNKNOWN"]

    metrics = compute_classification_metrics(y_true, y_pred)

    json_str_1 = to_deterministic_json(metrics)
    json_str_2 = to_deterministic_json(metrics)

    assert json_str_1 == json_str_2
    assert json_str_1.endswith("\n")

    # Verify keys are sorted alphabetically at top level
    parsed = json.loads(json_str_1)
    keys = list(parsed.keys())
    assert keys == sorted(keys)

    # Verify per-class keys are sorted
    per_class_keys = list(parsed["per_class"].keys())
    assert per_class_keys == sorted(per_class_keys)


# ---------------------------------------------------------------------------
# 13. None Serializes to JSON null
# ---------------------------------------------------------------------------


def test_none_serializes_to_json_null():
    """Verify that None values in metrics serialize exactly to JSON null."""
    y_true = ["O1"]
    y_pred = ["O1"]

    metrics = compute_classification_metrics(y_true, y_pred)
    json_str = to_deterministic_json(metrics)
    parsed = json.loads(json_str)

    # ON was absent in y_true, so its precision/recall/f1 must be JSON null
    assert parsed["per_class"]["ON"]["precision"] is None
    assert parsed["per_class"]["ON"]["recall"] is None
    assert parsed["per_class"]["ON"]["f1"] is None
    assert parsed["per_class"]["ON"]["support"] == 0

    # Raw substring check for literal null
    assert '"precision": null' in json_str


# ---------------------------------------------------------------------------
# 14. Rejection of NaN and Infinity
# ---------------------------------------------------------------------------


def test_nan_and_infinity_rejected_by_serializer():
    """Verify that NaN and Infinity raise SerializationError."""
    with pytest.raises(SerializationError, match="not compliant with standard JSON"):
        to_deterministic_json({"score": float("nan")})

    with pytest.raises(SerializationError, match="not compliant with standard JSON"):
        to_deterministic_json({"score": float("inf")})

    with pytest.raises(SerializationError, match="not compliant with standard JSON"):
        to_deterministic_json({"score": float("-inf")})


# ---------------------------------------------------------------------------
# 15. Metadata Contract Validation
# ---------------------------------------------------------------------------


def test_valid_evaluation_metadata_passes_validation():
    """Verify valid EvaluationMetadata passes validation and supports serialization."""
    raw = _valid_metadata_dict()
    meta = EvaluationMetadata.from_dict(raw)
    meta.validate()

    assert meta.model_id == "test_baseline_model_v1"
    assert meta.sample_count == 159
    assert meta.family_count == 53
    assert meta.git_dirty is False

    json_str = to_deterministic_json(meta)
    reloaded = json.loads(json_str)
    assert reloaded["model_id"] == "test_baseline_model_v1"
    assert reloaded["manifest_sha256"] == (
        "0b382938897af0f43de4ddd29b0fdf0306d2f33fbe73ad1c7876d7410d973cd4"
    )


@pytest.mark.parametrize(
    ("corrupt_field", "corrupt_value", "error_match"),
    [
        ("schema_version", "", "must be a non-empty string"),
        ("model_id", "   ", "must be a non-empty string"),
        ("fixed_hyperparameters", ["not", "a", "dict"], "must be a dict"),
        (
            "canonical_feature_schema",
            ["feat1", "feat2"],
            "must match the authoritative 26-feature schema",
        ),
        ("canonical_class_ordering", ["O1", "ON"], "must match canonical order"),
        ("manifest_sha256", "not_a_hex_hash", "valid 64-character hex SHA-256"),
        ("manifest_sha256", "0" * 63, "valid 64-character hex SHA-256"),
        ("fold_assignment_hash", "", "must be a non-empty string"),
        ("git_commit", "", "must be a non-empty string"),
        ("git_dirty", "False", "must be a boolean"),
        ("git_dirty", 0, "must be a boolean"),
        ("library_versions", {"scikit-learn": 1.9}, "must be a dict\\[str, str\\]"),
        ("sample_count", -1, "must be a non-negative integer"),
        ("sample_count", True, "must be a non-negative integer"),
        ("family_count", -5, "must be a non-negative integer"),
        ("family_count", 3.14, "must be a non-negative integer"),
        (
            "training_class_distribution",
            {"INVALID_CLASS": 10},
            "Invalid training class distribution",
        ),
        ("training_class_distribution", {"O1": -1}, "Invalid training class distribution"),
        ("missing_class_details", "not_a_dict", "must be a dict"),
        ("limitations", "not_a_list", "must be a list of strings"),
        ("limitations", [123], "must be a list of strings"),
    ],
)
def test_metadata_validation_rejects_invalid_fields(corrupt_field, corrupt_value, error_match):
    """Verify validation strictly rejects invalid values across all metadata fields."""
    data = _valid_metadata_dict()
    data[corrupt_field] = corrupt_value
    with pytest.raises(MetadataValidationError, match=error_match):
        validate_evaluation_metadata(data)


def test_metadata_validation_rejects_missing_required_fields():
    """Verify missing required fields raise MetadataValidationError."""
    data = _valid_metadata_dict()
    del data["manifest_sha256"]
    with pytest.raises(MetadataValidationError, match="missing required fields"):
        validate_evaluation_metadata(data)


# ---------------------------------------------------------------------------
# 16. Serializer Does Not Implicitly Create Directories or Caches
# ---------------------------------------------------------------------------


def test_write_deterministic_json_does_not_implicitly_create_parent_directory(tmp_path: Path):
    """Verify write_deterministic_json fails if destination parent does not exist."""
    non_existent_dir = tmp_path / "implicit_dir" / "nested"
    target_file = non_existent_dir / "metrics.json"

    assert not non_existent_dir.exists()

    with pytest.raises(FileNotFoundError, match="parent directory does not exist"):
        write_deterministic_json({"status": "ok"}, target_file, ensure_parents=False)

    # Directory was NOT created
    assert not non_existent_dir.exists()


def test_write_deterministic_json_success_with_explicit_path(tmp_path: Path):
    """Verify write_deterministic_json successfully writes UTF-8 JSON when path exists."""
    target_file = tmp_path / "metrics_output.json"
    obj = {"accuracy": 0.95, "name": "test"}

    written_path = write_deterministic_json(obj, target_file)
    assert written_path == target_file
    assert target_file.exists()

    content = target_file.read_text(encoding="utf-8")
    assert content == '{\n  "accuracy": 0.95,\n  "name": "test"\n}\n'


# ---------------------------------------------------------------------------
# Edge Cases: Perfect Predictions, All Wrong, and Single Class
# ---------------------------------------------------------------------------


def test_perfect_predictions_across_all_classes():
    """Verify metrics when every canonical class has perfect predictions."""
    y_true = list(CANONICAL_CLASS_ORDER)
    y_pred = list(CANONICAL_CLASS_ORDER)

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.macro_f1 == 1.0
    for cls_name in CANONICAL_CLASS_ORDER:
        m = metrics.per_class[cls_name]
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.support == 1


def test_all_predictions_wrong():
    """Verify metrics when every prediction is incorrect."""
    # Shift each prediction by 1 modulo 8
    y_true = list(CANONICAL_CLASS_ORDER)
    y_pred = [
        CANONICAL_CLASS_ORDER[(i + 1) % NUM_CANONICAL_CLASSES]
        for i in range(NUM_CANONICAL_CLASSES)
    ]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 0.0
    assert metrics.macro_precision == 0.0
    assert metrics.macro_recall == 0.0
    assert metrics.macro_f1 == 0.0
    for cls_name in CANONICAL_CLASS_ORDER:
        m = metrics.per_class[cls_name]
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.support == 1
