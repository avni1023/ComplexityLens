"""
baselines.py — Fixed baseline classifiers for ComplexityLens Stage 2C evaluation.

Provides three deterministic baseline classifiers:
1. MajorityClassBaseline:
   - Predicts the most frequent canonical class in the training labels.
   - Deterministic tie-breaking using the canonical class ordering (selects earliest tied class).
   - Zero reliance on validation or test data.
2. LogisticRegressionBaseline:
   - Scikit-learn Pipeline with StandardScaler and LogisticRegression.
   - Fixed hyperparameters: solver="lbfgs", C=1.0, max_iter=1000, random_state=42.
   - Fits scaler exclusively on training fold features.
   - Maps probability output into the complete 8-class canonical order with 0.0 for absent classes.
   - Surfaces convergence warnings.
3. DecisionTreeBaseline:
   - Scikit-learn DecisionTreeClassifier.
   - Fixed hyperparameters: criterion="gini", max_depth=5, random_state=42.
   - Enforces the 26-feature canonical schema and canonical label vocabulary.
   - Maps probability output into the complete 8-class canonical order with 0.0 for absent classes.

All models strictly validate input dimensionality, non-finite values, and label vocabularies,
and preserve complete isolation from holdout/test partitions.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE: bool = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False

from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
)
from complexity_lens.evaluation.metrics import (
    CANONICAL_CLASS_SET,
    CANONICAL_CLASS_TO_INDEX,
    NUM_CANONICAL_CLASSES,
    InvalidLabelsError,
)

NUM_CANONICAL_FEATURES: int = len(CANONICAL_FEATURE_NAMES)  # 26
DEFAULT_RANDOM_STATE: int = 42

LOGISTIC_REGRESSION_CONFIG: dict[str, Any] = {
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 1000,
    "random_state": DEFAULT_RANDOM_STATE,
}

DECISION_TREE_CONFIG: dict[str, Any] = {
    "criterion": "gini",
    "max_depth": 5,
    "random_state": DEFAULT_RANDOM_STATE,
}

RANDOM_FOREST_DEFAULT_CONFIG: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": 1,
}

XGBOOST_DEFAULT_CONFIG: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 2,
    "learning_rate": 0.1,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": 1,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BaselineError(Exception):
    """Base exception for baseline model errors."""


class NotFittedError(BaselineError, RuntimeError):
    """Raised when predict or predict_proba is called on an unfitted model."""


class InvalidInputError(BaselineError, ValueError):
    """Raised when feature or label inputs violate schema or numeric validity."""


# ---------------------------------------------------------------------------
# Input Validation Helpers
# ---------------------------------------------------------------------------


def validate_features(X: Any, *, allow_empty: bool = False) -> np.ndarray:
    """Validate that X is a 2D numeric array of shape (n_samples, 26) with finite values.

    Parameters
    ----------
    X:
        Feature matrix as list of sequences, array-like, or numpy ndarray.
    allow_empty:
        If False (default), raises InvalidInputError if n_samples == 0.

    Returns
    -------
    np.ndarray
        Validated 2D float64 array of shape (n_samples, 26).
    """
    if X is None:
        raise InvalidInputError("Feature input X cannot be None.")

    try:
        arr = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise InvalidInputError(f"Features must be convertible to numeric floats: {e}") from e

    if arr.ndim != 2:
        raise InvalidInputError(
            f"Features must be a 2D matrix of shape (n_samples, 26), got ndim={arr.ndim}."
        )

    if arr.shape[1] != NUM_CANONICAL_FEATURES:
        raise InvalidInputError(
            f"Expected exactly {NUM_CANONICAL_FEATURES} canonical features, got {arr.shape[1]}."
        )

    if not allow_empty and arr.shape[0] == 0:
        raise InvalidInputError("Feature matrix cannot be empty.")

    if not np.all(np.isfinite(arr)):
        raise InvalidInputError("Features contain non-finite values (NaN, +Inf, or -Inf).")

    return arr


def validate_labels(y: Sequence[str], *, allow_empty: bool = False) -> list[str]:
    """Validate that y contains canonical class labels.

    Parameters
    ----------
    y:
        Sequence of label strings.
    allow_empty:
        If False (default), raises InvalidInputError if len(y) == 0.

    Returns
    -------
    list[str]
        List of verified canonical labels.
    """
    if y is None:
        raise InvalidInputError("Labels input y cannot be None.")

    y_list = [str(lbl) for lbl in y]
    if not allow_empty and len(y_list) == 0:
        raise InvalidInputError("Labels input y cannot be empty.")

    invalid = sorted({lbl for lbl in y_list if lbl not in CANONICAL_CLASS_SET})
    if invalid:
        raise InvalidLabelsError(
            f"Encountered unknown labels not in canonical class order: {invalid}."
        )

    return y_list


def map_probabilities_to_canonical_order(
    model_classes: Sequence[str],
    raw_probabilities: np.ndarray,
    canonical_classes: Sequence[str] = CANONICAL_CLASS_ORDER,
) -> np.ndarray:
    """Map model predicted probability columns into the full canonical class order.

    Canonical classes absent from model_classes receive probability 0.0.

    Parameters
    ----------
    model_classes:
        Class labels corresponding to columns of raw_probabilities.
    raw_probabilities:
        2D array of shape (n_samples, len(model_classes)).
    canonical_classes:
        Target canonical class ordering (default: 8 canonical classes).

    Returns
    -------
    np.ndarray
        2D float64 array of shape (n_samples, len(canonical_classes)).
    """
    n_samples = raw_probabilities.shape[0]
    n_canon = len(canonical_classes)
    mapped = np.zeros((n_samples, n_canon), dtype=np.float64)

    for model_idx, cls_name in enumerate(model_classes):
        if cls_name in CANONICAL_CLASS_TO_INDEX:
            canon_idx = CANONICAL_CLASS_TO_INDEX[cls_name]
            mapped[:, canon_idx] = raw_probabilities[:, model_idx]

    return mapped


# ---------------------------------------------------------------------------
# Baseline A: Majority-Class Classifier
# ---------------------------------------------------------------------------


class MajorityClassBaseline:
    """Deterministic majority-class classifier with canonical tie-breaking.

    - Fits exclusively on training labels y.
    - Resolves ties using canonical class order (earliest tied class).
    - Features X are validated if provided but not used for fitting.
    """

    model_id: str = "majority_class"

    def __init__(self) -> None:
        self.majority_class_: str | None = None
        self.class_counts_: dict[str, int] = {}
        self.training_sample_count_: int = 0
        self.fixed_hyperparameters: dict[str, Any] = {
            "strategy": "most_frequent",
            "tie_breaker": "canonical_order",
        }

    @property
    def is_fitted(self) -> bool:
        """True if the model has been fitted on training labels."""
        return self.majority_class_ is not None

    def fit(self, X: Any, y: Sequence[str]) -> MajorityClassBaseline:
        """Fit the majority class on training labels.

        Parameters
        ----------
        X:
            Feature matrix (validated if provided; shape matching y enforced).
        y:
            Sequence of ground-truth canonical labels.
        """
        y_validated = validate_labels(y, allow_empty=False)

        if X is not None:
            X_arr = validate_features(X, allow_empty=False)
            if len(X_arr) != len(y_validated):
                raise InvalidInputError(
                    f"Mismatched sample counts: len(X)={len(X_arr)} != len(y)={len(y_validated)}."
                )

        counts = Counter(y_validated)
        self.class_counts_ = {
            cls_name: counts.get(cls_name, 0) for cls_name in CANONICAL_CLASS_ORDER
        }
        self.training_sample_count_ = len(y_validated)

        # Tie-breaker: iterate in canonical class order to select earliest tied class
        max_count = max(counts.values())
        tied_classes = [c for c in CANONICAL_CLASS_ORDER if counts.get(c, 0) == max_count]
        self.majority_class_ = tied_classes[0]

        return self

    def predict(self, X: Any) -> list[str]:
        """Predict the majority class for all samples in X."""
        if not self.is_fitted:
            raise NotFittedError("MajorityClassBaseline is not fitted. Call fit() first.")

        X_arr = validate_features(X, allow_empty=False)
        return [self.majority_class_] * len(X_arr)  # type: ignore[list-item]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return 8-class probabilities with 1.0 assigned to the majority class."""
        if not self.is_fitted:
            raise NotFittedError("MajorityClassBaseline is not fitted. Call fit() first.")

        X_arr = validate_features(X, allow_empty=False)
        n_samples = len(X_arr)
        proba = np.zeros((n_samples, NUM_CANONICAL_CLASSES), dtype=np.float64)
        maj_idx = CANONICAL_CLASS_TO_INDEX[self.majority_class_]  # type: ignore[index]
        proba[:, maj_idx] = 1.0
        return proba


# ---------------------------------------------------------------------------
# Baseline B: Logistic Regression (StandardScaler + LogisticRegression)
# ---------------------------------------------------------------------------


class LogisticRegressionBaseline:
    """Logistic Regression baseline with integrated StandardScaler pipeline.

    Fixed Hyperparameters:
    - solver: 'lbfgs'
    - C: 1.0
    - max_iter: 1000
    - random_state: 42
    - Preprocessing: StandardScaler fitted strictly inside Pipeline on training features.
    """

    model_id: str = "logistic_regression"

    def __init__(self, *, random_state: int = DEFAULT_RANDOM_STATE) -> None:
        self.random_state = random_state
        self.fixed_hyperparameters: dict[str, Any] = {
            "solver": LOGISTIC_REGRESSION_CONFIG["solver"],
            "C": LOGISTIC_REGRESSION_CONFIG["C"],
            "max_iter": LOGISTIC_REGRESSION_CONFIG["max_iter"],
            "random_state": self.random_state,
            "preprocessing": "StandardScaler",
        }
        self.convergence_warnings_: list[str] = []
        self._classes_: list[str] = []
        self._pipeline: Pipeline | None = None

    @property
    def is_fitted(self) -> bool:
        """True if the pipeline has been fitted."""
        return self._pipeline is not None and len(self._classes_) > 0

    def fit(self, X: Any, y: Sequence[str]) -> LogisticRegressionBaseline:
        """Fit the StandardScaler and LogisticRegression model on training data.

        Scaler statistics are derived strictly from X.
        """
        X_arr = validate_features(X, allow_empty=False)
        y_validated = validate_labels(y, allow_empty=False)

        if len(X_arr) != len(y_validated):
            raise InvalidInputError(
                f"Mismatched sample counts: len(X)={len(X_arr)} != len(y)={len(y_validated)}."
            )

        unique_classes = sorted(set(y_validated))
        if len(unique_classes) < 2:
            raise InvalidInputError(
                f"LogisticRegression requires at least 2 distinct classes in training data, "
                f"got {len(unique_classes)}: {unique_classes}."
            )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver=LOGISTIC_REGRESSION_CONFIG["solver"],
                    C=LOGISTIC_REGRESSION_CONFIG["C"],
                    max_iter=LOGISTIC_REGRESSION_CONFIG["max_iter"],
                    random_state=self.random_state,
                ),
            ),
        ])

        self.convergence_warnings_ = []
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(X_arr, y_validated)

        # Re-emit captured warnings so they surface to callers and test runners
        for w in captured:
            if issubclass(w.category, ConvergenceWarning):
                self.convergence_warnings_.append(str(w.message))
                warnings.warn(str(w.message), ConvergenceWarning, stacklevel=2)
            else:
                warnings.warn(str(w.message), w.category, stacklevel=2)

        self._pipeline = pipeline
        clf: LogisticRegression = pipeline.named_steps["classifier"]
        self._classes_ = [str(c) for c in clf.classes_]

        return self

    def predict(self, X: Any) -> list[str]:
        """Predict canonical class labels for feature matrix X."""
        if not self.is_fitted:
            raise NotFittedError("LogisticRegressionBaseline is not fitted. Call fit() first.")

        X_arr = validate_features(X, allow_empty=False)
        preds = self._pipeline.predict(X_arr)  # type: ignore[union-attr]
        return [str(p) for p in preds]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities mapped to the complete 8-class canonical order."""
        if not self.is_fitted:
            raise NotFittedError("LogisticRegressionBaseline is not fitted. Call fit() first.")

        X_arr = validate_features(X, allow_empty=False)
        raw_proba = self._pipeline.predict_proba(X_arr)  # type: ignore[union-attr]

        return map_probabilities_to_canonical_order(
            model_classes=self._classes_,
            raw_probabilities=raw_proba,
            canonical_classes=CANONICAL_CLASS_ORDER,
        )


# ---------------------------------------------------------------------------
# Baseline C: Decision Tree Classifier
# ---------------------------------------------------------------------------


class DecisionTreeBaseline:
    """Decision Tree baseline with fixed depth and gini criterion.

    Fixed Hyperparameters:
    - criterion: 'gini'
    - max_depth: 5
    - random_state: 42
    """

    model_id: str = "decision_tree"

    def __init__(self, *, random_state: int = DEFAULT_RANDOM_STATE) -> None:
        self.random_state = random_state
        self.fixed_hyperparameters: dict[str, Any] = {
            "criterion": DECISION_TREE_CONFIG["criterion"],
            "max_depth": DECISION_TREE_CONFIG["max_depth"],
            "random_state": self.random_state,
        }
        self._classes_: list[str] = []
        self._classifier: DecisionTreeClassifier | None = None

    @property
    def is_fitted(self) -> bool:
        """True if the tree has been fitted."""
        return self._classifier is not None and len(self._classes_) > 0

    def fit(self, X: Any, y: Sequence[str]) -> DecisionTreeBaseline:
        """Fit the DecisionTreeClassifier on training data."""
        X_arr = validate_features(X, allow_empty=False)
        y_validated = validate_labels(y, allow_empty=False)

        if len(X_arr) != len(y_validated):
            raise InvalidInputError(
                f"Mismatched sample counts: len(X)={len(X_arr)} != len(y)={len(y_validated)}."
            )

        clf = DecisionTreeClassifier(
            criterion=DECISION_TREE_CONFIG["criterion"],
            max_depth=DECISION_TREE_CONFIG["max_depth"],
            random_state=self.random_state,
        )
        clf.fit(X_arr, y_validated)

        self._classifier = clf
        self._classes_ = [str(c) for c in clf.classes_]

        return self

    def predict(self, X: Any) -> list[str]:
        """Predict canonical class labels for feature matrix X."""
        if not self.is_fitted:
            raise NotFittedError("DecisionTreeBaseline is not fitted. Call fit() first.")

        X_arr = validate_features(X, allow_empty=False)
        preds = self._classifier.predict(X_arr)  # type: ignore[union-attr]
        return [str(p) for p in preds]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities mapped to the complete 8-class canonical order."""
        if not self.is_fitted:
            raise NotFittedError("DecisionTreeBaseline is not fitted. Call fit() first.")

        X_arr = validate_features(X, allow_empty=False)
        raw_proba = self._classifier.predict_proba(X_arr)  # type: ignore[union-attr]

        return map_probabilities_to_canonical_order(
            model_classes=self._classes_,
            raw_probabilities=raw_proba,
            canonical_classes=CANONICAL_CLASS_ORDER,
        )


# ---------------------------------------------------------------------------
# Candidate D: Random Forest Classifier
# ---------------------------------------------------------------------------


class RandomForestBaseline:
    """Random Forest classifier adhering to canonical 8-class evaluation schema.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest (default: 200).
    max_depth:
        Maximum depth of trees (default: None; or integer like 6).
    min_samples_leaf:
        Minimum number of samples required at a leaf node (default: 1; or 2).
    random_state:
        Seed for reproducibility (default: 42).
    n_jobs:
        Worker threads (default: 1 for determinism).
    """

    model_id: str = "random_forest"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth is not None else None
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.fixed_hyperparameters: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        self._classes_: list[str] = []
        self._classifier: RandomForestClassifier | None = None

    @property
    def is_fitted(self) -> bool:
        """True if the random forest has been fitted."""
        return self._classifier is not None and len(self._classes_) > 0

    @property
    def feature_importances_(self) -> np.ndarray:
        """Normalized feature importances across the 26 canonical features."""
        if not self.is_fitted or self._classifier is None:
            raise NotFittedError("RandomForestBaseline is not fitted. Call fit() first.")
        return np.asarray(self._classifier.feature_importances_, dtype=np.float64)

    def fit(self, X: Any, y: Sequence[str]) -> RandomForestBaseline:
        """Fit the random forest on training feature matrix and labels."""
        X_arr = validate_features(X, allow_empty=False)
        y_validated = validate_labels(y, allow_empty=False)

        if len(X_arr) != len(y_validated):
            raise InvalidInputError(
                f"Mismatched sample counts: len(X)={len(X_arr)} != len(y)={len(y_validated)}."
            )

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        clf.fit(X_arr, y_validated)
        self._classifier = clf
        self._classes_ = [str(c) for c in clf.classes_]
        return self

    def predict(self, X: Any) -> list[str]:
        """Predict canonical class labels for feature matrix X."""
        if not self.is_fitted or self._classifier is None:
            raise NotFittedError("RandomForestBaseline is not fitted. Call fit() first.")
        X_arr = validate_features(X, allow_empty=False)
        preds = self._classifier.predict(X_arr)
        return [str(p) for p in preds]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities mapped to the complete 8-class canonical order."""
        if not self.is_fitted or self._classifier is None:
            raise NotFittedError("RandomForestBaseline is not fitted. Call fit() first.")
        X_arr = validate_features(X, allow_empty=False)
        raw_proba = self._classifier.predict_proba(X_arr)
        return map_probabilities_to_canonical_order(
            model_classes=self._classes_,
            raw_probabilities=raw_proba,
            canonical_classes=CANONICAL_CLASS_ORDER,
        )


# ---------------------------------------------------------------------------
# Candidate E: XGBoost Classifier
# ---------------------------------------------------------------------------


class XGBoostBaseline:
    """XGBoost classifier with contiguous label encoding and canonical probability mapping.

    Requirements:
    - Encodes training fold classes into contiguous integer codes [0 .. K-1].
    - Trains using only encoded labels present in the fold.
    - Sets n_jobs=1, fixed random seed, eval_metric='mlogloss', tree_method='hist'.
    - Predicts probabilities for the K trained classes.
    - Maps predicted probabilities into canonical 8-class order with 0.0 for absent classes.
    - Supports native JSON serialization and deserialization.
    """

    model_id: str = "xgboost"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 2,
        learning_rate: float = 0.1,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = 1,
    ) -> None:
        if not XGBOOST_AVAILABLE:
            raise BaselineError(
                "xgboost package is not installed or importable. "
                "Install with prebuilt wheels: pip install 'xgboost>=2.0.0'"
            )
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.fixed_hyperparameters: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
        }
        self._classes_: list[str] = []
        self._class_to_code: dict[str, int] = {}
        self._classifier: Any = None

    @property
    def is_fitted(self) -> bool:
        """True if the model has been fitted."""
        return self._classifier is not None and len(self._classes_) > 0

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances across the 26 canonical features."""
        if not self.is_fitted:
            raise NotFittedError("XGBoostBaseline is not fitted. Call fit() first.")
        if self._classifier == "single_class":
            return np.zeros(NUM_CANONICAL_FEATURES, dtype=np.float64)
        return np.asarray(self._classifier.feature_importances_, dtype=np.float64)

    def fit(self, X: Any, y: Sequence[str]) -> XGBoostBaseline:
        """Fit the XGBoost classifier on training features and labels."""
        X_arr = validate_features(X, allow_empty=False)
        y_validated = validate_labels(y, allow_empty=False)

        if len(X_arr) != len(y_validated):
            raise InvalidInputError(
                f"Mismatched sample counts: len(X)={len(X_arr)} != len(y)={len(y_validated)}."
            )

        # Identify classes present in this fold's training labels, ordered canonically
        present_classes = sorted(
            list(set(y_validated)),
            key=lambda c: CANONICAL_CLASS_TO_INDEX.get(c, 999),
        )
        self._classes_ = present_classes
        self._class_to_code = {cls_name: idx for idx, cls_name in enumerate(present_classes)}

        if len(present_classes) == 1:
            self._classifier = "single_class"
            return self

        y_encoded = np.array([self._class_to_code[c] for c in y_validated], dtype=np.int32)
        eval_metric = "mlogloss" if len(present_classes) > 2 else "logloss"

        clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            eval_metric=eval_metric,
        )
        clf.fit(X_arr, y_encoded)
        self._classifier = clf
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities mapped to the complete 8-class canonical order."""
        if not self.is_fitted:
            raise NotFittedError("XGBoostBaseline is not fitted. Call fit() first.")
        X_arr = validate_features(X, allow_empty=False)
        n_samples = len(X_arr)

        if self._classifier == "single_class":
            raw_proba = np.ones((n_samples, 1), dtype=np.float64)
        else:
            raw_proba = self._classifier.predict_proba(X_arr)

        return map_probabilities_to_canonical_order(
            model_classes=self._classes_,
            raw_probabilities=raw_proba,
            canonical_classes=CANONICAL_CLASS_ORDER,
        )

    def predict(self, X: Any) -> list[str]:
        """Predict canonical class labels for feature matrix X."""
        if not self.is_fitted:
            raise NotFittedError("XGBoostBaseline is not fitted. Call fit() first.")
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return [CANONICAL_CLASS_ORDER[idx] for idx in pred_indices]

    def save_model(self, file_path: str | Path) -> None:
        """Save fitted XGBoost model in native JSON format."""
        if not self.is_fitted:
            raise NotFittedError("Cannot save unfitted XGBoost model.")
        if self._classifier == "single_class":
            raise BaselineError("Cannot persist degenerate single-class model via native XGBoost.")
        self._classifier.save_model(str(file_path))

    def load_model(self, file_path: str | Path, classes: Sequence[str]) -> None:
        """Load fitted XGBoost model from native JSON format."""
        if not XGBOOST_AVAILABLE:
            raise BaselineError("xgboost package is not installed or importable.")
        clf = xgb.XGBClassifier()
        clf.load_model(str(file_path))
        self._classifier = clf
        self._classes_ = list(classes)
        self._class_to_code = {c: i for i, c in enumerate(self._classes_)}


# ---------------------------------------------------------------------------
# Designated Fallback: HistGradientBoosting Classifier
# ---------------------------------------------------------------------------


class HistGradientBoostingBaseline:
    """Designated fallback classifier using scikit-learn HistGradientBoostingClassifier."""

    model_id: str = "hist_gradient_boosting"

    def __init__(
        self,
        *,
        max_iter: int = 100,
        max_depth: int = 2,
        learning_rate: float = 0.1,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.max_iter = int(max_iter)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.fixed_hyperparameters: dict[str, Any] = {
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }
        self._classes_: list[str] = []
        self._classifier: HistGradientBoostingClassifier | None = None

    @property
    def is_fitted(self) -> bool:
        """True if the classifier has been fitted."""
        return self._classifier is not None and len(self._classes_) > 0

    def fit(self, X: Any, y: Sequence[str]) -> HistGradientBoostingBaseline:
        """Fit the gradient boosting classifier on training data."""
        X_arr = validate_features(X, allow_empty=False)
        y_validated = validate_labels(y, allow_empty=False)

        if len(X_arr) != len(y_validated):
            raise InvalidInputError(
                f"Mismatched sample counts: len(X)={len(X_arr)} != len(y)={len(y_validated)}."
            )

        clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        clf.fit(X_arr, y_validated)
        self._classifier = clf
        self._classes_ = [str(c) for c in clf.classes_]
        return self

    def predict(self, X: Any) -> list[str]:
        """Predict canonical class labels for feature matrix X."""
        if not self.is_fitted or self._classifier is None:
            raise NotFittedError("HistGradientBoostingBaseline is not fitted. Call fit() first.")
        X_arr = validate_features(X, allow_empty=False)
        preds = self._classifier.predict(X_arr)
        return [str(p) for p in preds]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities mapped to the complete 8-class canonical order."""
        if not self.is_fitted or self._classifier is None:
            raise NotFittedError("HistGradientBoostingBaseline is not fitted. Call fit() first.")
        X_arr = validate_features(X, allow_empty=False)
        raw_proba = self._classifier.predict_proba(X_arr)
        return map_probabilities_to_canonical_order(
            model_classes=self._classes_,
            raw_probabilities=raw_proba,
            canonical_classes=CANONICAL_CLASS_ORDER,
        )


# ---------------------------------------------------------------------------
# Model Registry / Factory
# ---------------------------------------------------------------------------


def create_baseline_model(model_id: str, **kwargs: Any) -> Any:
    """Instantiate a model classifier by its canonical identifier.

    Supported model_ids:
    - 'majority_class'
    - 'logistic_regression'
    - 'decision_tree'
    - 'random_forest'
    - 'xgboost'
    - 'hist_gradient_boosting'
    """
    normalized_id = str(model_id).strip().lower()
    if normalized_id == "majority_class":
        return MajorityClassBaseline()
    if normalized_id == "logistic_regression":
        return LogisticRegressionBaseline(**kwargs)
    if normalized_id == "decision_tree":
        return DecisionTreeBaseline(**kwargs)
    if normalized_id == "random_forest":
        return RandomForestBaseline(**kwargs)
    if normalized_id == "xgboost":
        return XGBoostBaseline(**kwargs)
    if normalized_id == "hist_gradient_boosting":
        return HistGradientBoostingBaseline(**kwargs)

    raise ValueError(
        f"Unknown baseline model_id: {model_id!r}. Supported models: "
        f"['majority_class', 'logistic_regression', 'decision_tree', "
        f"'random_forest', 'xgboost', 'hist_gradient_boosting']."
    )
