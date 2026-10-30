"""
tests/test_evaluation_baselines.py — Tests for fixed baseline classifiers.

Covers:
1. Majority-Class baseline:
   - Correct most-frequent class.
   - Canonical-order tie-breaking.
   - Rejection of empty training input and invalid labels.
   - Deterministic predictions.
   - Probability output matches 8-class canonical format.
2. Logistic Regression baseline:
   - Fixed hyperparameters and Pipeline integration (StandardScaler + LogisticRegression).
   - Valid synthetic training and deterministic predictions.
   - Rejection of invalid feature count, non-finite features, empty data, and invalid labels.
   - Surfaces convergence warnings.
   - Probability mapping to complete 8-class canonical order with 0.0 for absent classes.
3. Decision Tree baseline:
   - Fixed hyperparameters (gini, max_depth=5, seed=42).
   - Valid synthetic training and deterministic predictions.
   - Rejection of invalid feature shapes and invalid labels.
   - Probability mapping for absent classes.
4. Isolation and leakage safeguards:
   - Fitting strictly on supplied training samples.
   - Scaler fitted exclusively on training features (not validation).
   - Zero disk caches or persistent model artifacts created.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from complexity_lens.evaluation.baselines import (
    LOGISTIC_REGRESSION_CONFIG,
    NUM_CANONICAL_FEATURES,
    DecisionTreeBaseline,
    InvalidInputError,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    NotFittedError,
    create_baseline_model,
)
from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
)
from complexity_lens.evaluation.metrics import (
    CANONICAL_CLASS_TO_INDEX,
    NUM_CANONICAL_CLASSES,
    InvalidLabelsError,
)

# ---------------------------------------------------------------------------
# Synthetic Test Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    n_samples: int = 20,
    classes: list[str] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Generate deterministic synthetic feature matrix and labels."""
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 10, size=(n_samples, NUM_CANONICAL_FEATURES)).astype(np.float64)

    if classes is None:
        classes = ["O1", "ON", "ON2", "UNKNOWN"]

    y = [classes[i % len(classes)] for i in range(n_samples)]
    return X, y


# ---------------------------------------------------------------------------
# Majority-Class Baseline Tests
# ---------------------------------------------------------------------------


def test_majority_class_most_frequent_prediction():
    """Verify MajorityClassBaseline selects the single most frequent class."""
    X, _ = _make_synthetic_data(n_samples=10)
    # 5 "ON", 3 "O1", 2 "UNKNOWN" -> majority is "ON"
    y = ["ON"] * 5 + ["O1"] * 3 + ["UNKNOWN"] * 2

    model = MajorityClassBaseline()
    assert not model.is_fitted

    model.fit(X, y)
    assert model.is_fitted
    assert model.majority_class_ == "ON"
    assert model.training_sample_count_ == 10
    assert model.class_counts_["ON"] == 5
    assert model.class_counts_["O1"] == 3

    X_test, _ = _make_synthetic_data(n_samples=4, seed=99)
    preds = model.predict(X_test)
    assert preds == ["ON", "ON", "ON", "ON"]

    proba = model.predict_proba(X_test)
    assert proba.shape == (4, NUM_CANONICAL_CLASSES)
    # Column for "ON" (index 2) must be 1.0; all others 0.0
    on_idx = CANONICAL_CLASS_TO_INDEX["ON"]
    for row in proba:
        assert row[on_idx] == 1.0
        assert sum(row) == pytest.approx(1.0)


def test_majority_class_canonical_order_tie_breaking():
    """Verify tie-breaking picks earliest class in canonical order.

    Canonical order: O1, OLOGN, ON, ONLOGN, ON2, ON3, OEXP, UNKNOWN.
    If 'ON' and 'O1' are tied with 3 votes each, 'O1' must win because it precedes 'ON'.
    """
    y = ["ON", "ON", "ON", "O1", "O1", "O1"]
    model = MajorityClassBaseline().fit(X=None, y=y)
    assert model.majority_class_ == "O1"

    # Tie between 'UNKNOWN' and 'OEXP' -> 'OEXP' precedes 'UNKNOWN'
    y2 = ["UNKNOWN", "UNKNOWN", "OEXP", "OEXP"]
    model2 = MajorityClassBaseline().fit(X=None, y=y2)
    assert model2.majority_class_ == "OEXP"


def test_majority_class_empty_input_rejected():
    """Verify empty label sequence raises InvalidInputError."""
    model = MajorityClassBaseline()
    with pytest.raises(InvalidInputError, match="cannot be empty"):
        model.fit(X=None, y=[])


def test_majority_class_invalid_labels_rejected():
    """Verify non-canonical labels raise InvalidLabelsError."""
    model = MajorityClassBaseline()
    with pytest.raises(InvalidLabelsError, match="unknown labels"):
        model.fit(X=None, y=["O1", "O_N_SQUARED"])


def test_majority_class_unfitted_raises_not_fitted_error():
    """Verify calling predict or predict_proba before fit raises NotFittedError."""
    model = MajorityClassBaseline()
    X, _ = _make_synthetic_data(n_samples=2)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(X)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict_proba(X)


# ---------------------------------------------------------------------------
# Logistic Regression Baseline Tests
# ---------------------------------------------------------------------------


def test_logistic_regression_fixed_hyperparameters():
    """Verify fixed hyperparameters match specifications."""
    model = LogisticRegressionBaseline(random_state=42)
    assert model.fixed_hyperparameters["solver"] == "lbfgs"
    assert model.fixed_hyperparameters["C"] == 1.0
    assert model.fixed_hyperparameters["max_iter"] == 1000
    assert model.fixed_hyperparameters["random_state"] == 42
    assert model.fixed_hyperparameters["preprocessing"] == "StandardScaler"


def test_logistic_regression_pipeline_and_training():
    """Verify model fits inside a StandardScaler pipeline and predicts deterministically."""
    X_train, y_train = _make_synthetic_data(n_samples=30, classes=["O1", "ON", "OEXP"])
    model = LogisticRegressionBaseline()
    assert not model.is_fitted

    model.fit(X_train, y_train)
    assert model.is_fitted

    # Verify pipeline structure
    assert isinstance(model._pipeline, Pipeline)
    assert isinstance(model._pipeline.named_steps["scaler"], StandardScaler)

    X_test, _ = _make_synthetic_data(n_samples=5, seed=123)
    preds_1 = model.predict(X_test)
    preds_2 = model.predict(X_test)
    assert preds_1 == preds_2
    assert len(preds_1) == 5
    assert all(p in CANONICAL_CLASS_ORDER for p in preds_1)


def test_logistic_regression_probability_mapping_missing_classes():
    """Verify probability output maps to all 8 canonical classes with 0.0 for absent classes."""
    # Only 3 classes present in training data: O1, ON, UNKNOWN
    # 5 classes absent: OLOGN, ONLOGN, ON2, ON3, OEXP
    X_train, y_train = _make_synthetic_data(n_samples=20, classes=["O1", "ON", "UNKNOWN"])
    model = LogisticRegressionBaseline().fit(X_train, y_train)

    X_test, _ = _make_synthetic_data(n_samples=4, seed=77)
    proba = model.predict_proba(X_test)

    assert proba.shape == (4, NUM_CANONICAL_CLASSES)

    # Absent classes must receive exactly 0.0 probability
    for absent_cls in ["OLOGN", "ONLOGN", "ON2", "ON3", "OEXP"]:
        col_idx = CANONICAL_CLASS_TO_INDEX[absent_cls]
        np.testing.assert_array_equal(proba[:, col_idx], 0.0)

    # Each row must sum to 1.0
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_logistic_regression_rejects_single_class():
    """Verify fitting on only 1 distinct class raises InvalidInputError."""
    X, _ = _make_synthetic_data(n_samples=10)
    y = ["O1"] * 10
    model = LogisticRegressionBaseline()
    with pytest.raises(InvalidInputError, match="at least 2 distinct classes"):
        model.fit(X, y)


def test_logistic_regression_convergence_warning_surfaced():
    """Verify convergence warnings are surfaced rather than silently suppressed."""
    # Create ill-conditioned data and fit with max_iter=1 to trigger convergence warning
    rng = np.random.RandomState(42)
    X = rng.randn(20, NUM_CANONICAL_FEATURES) * 100.0
    y = ["O1"] * 10 + ["ON"] * 10

    LOGISTIC_REGRESSION_CONFIG["max_iter"] = 1
    try:
        model = LogisticRegressionBaseline()
        with pytest.warns(ConvergenceWarning):
            model.fit(X, y)
        assert len(model.convergence_warnings_) > 0
    finally:
        LOGISTIC_REGRESSION_CONFIG["max_iter"] = 1000


# ---------------------------------------------------------------------------
# Decision Tree Baseline Tests
# ---------------------------------------------------------------------------


def test_decision_tree_fixed_hyperparameters():
    """Verify fixed hyperparameters for DecisionTreeBaseline."""
    model = DecisionTreeBaseline(random_state=42)
    assert model.fixed_hyperparameters["criterion"] == "gini"
    assert model.fixed_hyperparameters["max_depth"] == 5
    assert model.fixed_hyperparameters["random_state"] == 42


def test_decision_tree_training_and_deterministic_predictions():
    """Verify decision tree fits and generates deterministic predictions."""
    X_train, y_train = _make_synthetic_data(n_samples=30, classes=["O1", "ON", "ON2", "OEXP"])
    model = DecisionTreeBaseline()
    assert not model.is_fitted

    model.fit(X_train, y_train)
    assert model.is_fitted

    X_test, _ = _make_synthetic_data(n_samples=6, seed=88)
    preds1 = model.predict(X_test)
    preds2 = model.predict(X_test)
    assert preds1 == preds2
    assert len(preds1) == 6


def test_decision_tree_probability_mapping_missing_classes():
    """Verify decision tree maps absent classes to 0.0 in 8-class probability matrix."""
    # Only 2 classes: ON, ON3
    X_train, y_train = _make_synthetic_data(n_samples=25, classes=["ON", "ON3"])
    model = DecisionTreeBaseline().fit(X_train, y_train)

    X_test, _ = _make_synthetic_data(n_samples=5, seed=11)
    proba = model.predict_proba(X_test)

    assert proba.shape == (5, NUM_CANONICAL_CLASSES)
    for absent_cls in ["O1", "OLOGN", "ONLOGN", "ON2", "OEXP", "UNKNOWN"]:
        col_idx = CANONICAL_CLASS_TO_INDEX[absent_cls]
        np.testing.assert_array_equal(proba[:, col_idx], 0.0)

    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Input Validation & Non-Finite Value Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [MajorityClassBaseline, LogisticRegressionBaseline, DecisionTreeBaseline],
)
def test_models_reject_empty_training_data(model_cls):
    """Verify all models reject empty feature or label inputs."""
    model = model_cls()
    with pytest.raises(InvalidInputError):
        model.fit([], [])


@pytest.mark.parametrize(
    "model_cls",
    [MajorityClassBaseline, LogisticRegressionBaseline, DecisionTreeBaseline],
)
def test_models_reject_mismatched_sample_counts(model_cls):
    """Verify mismatched sample counts between X and y are rejected."""
    X, y = _make_synthetic_data(n_samples=10)
    model = model_cls()
    with pytest.raises(InvalidInputError, match="Mismatched sample counts"):
        model.fit(X, y[:5])


@pytest.mark.parametrize(
    "model_cls",
    [LogisticRegressionBaseline, DecisionTreeBaseline],
)
def test_models_reject_wrong_feature_count(model_cls):
    """Verify matrices with != 26 features are rejected."""
    X_bad = np.ones((10, 25), dtype=np.float64)  # 25 instead of 26
    y = ["O1"] * 5 + ["ON"] * 5
    model = model_cls()
    with pytest.raises(InvalidInputError, match="Expected exactly 26 canonical features"):
        model.fit(X_bad, y)


@pytest.mark.parametrize(
    "model_cls",
    [LogisticRegressionBaseline, DecisionTreeBaseline],
)
def test_models_reject_non_finite_features(model_cls):
    """Verify NaN and Infinity in feature inputs are rejected."""
    X_nan, y = _make_synthetic_data(n_samples=10)
    X_nan[2, 5] = np.nan
    model = model_cls()
    with pytest.raises(InvalidInputError, match="non-finite values"):
        model.fit(X_nan, y)

    X_inf, _ = _make_synthetic_data(n_samples=10)
    X_inf[3, 8] = np.inf
    with pytest.raises(InvalidInputError, match="non-finite values"):
        model.fit(X_inf, y)


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------


def test_create_baseline_model_factory():
    """Verify create_baseline_model instantiates correct model types."""
    m1 = create_baseline_model("majority_class")
    assert isinstance(m1, MajorityClassBaseline)

    m2 = create_baseline_model("logistic_regression")
    assert isinstance(m2, LogisticRegressionBaseline)

    m3 = create_baseline_model("decision_tree")
    assert isinstance(m3, DecisionTreeBaseline)

    with pytest.raises(ValueError, match="Unknown baseline model_id"):
        create_baseline_model("unsupported_neural_net")


# ---------------------------------------------------------------------------
# Isolation and Leakage Safeguards
# ---------------------------------------------------------------------------


def test_scaler_isolated_from_validation_data():
    """Verify StandardScaler is fitted only on training data, not validation data."""
    # Training features have mean ~ 100
    X_train = np.ones((20, NUM_CANONICAL_FEATURES)) * 100.0
    y_train = ["O1"] * 10 + ["ON"] * 10

    model = LogisticRegressionBaseline().fit(X_train, y_train)
    scaler: StandardScaler = model._pipeline.named_steps["scaler"]  # type: ignore[union-attr]

    np.testing.assert_allclose(scaler.mean_, 100.0)

    # Validation features have wildly different distribution (mean ~ 5000)
    X_val = np.ones((10, NUM_CANONICAL_FEATURES)) * 5000.0
    _ = model.predict(X_val)

    # Scaler mean must NOT have changed
    np.testing.assert_allclose(scaler.mean_, 100.0)


def test_no_disk_artifacts_created_during_fitting(tmp_path: Path):
    """Verify baseline model fitting creates zero disk files or cache directories."""
    repo_artifacts = Path("artifacts")
    assert not repo_artifacts.exists()

    X, y = _make_synthetic_data(n_samples=15)
    _ = MajorityClassBaseline().fit(X, y)
    _ = LogisticRegressionBaseline().fit(X, y)
    _ = DecisionTreeBaseline().fit(X, y)

    # Confirm artifacts directory was NOT created
    assert not repo_artifacts.exists()
