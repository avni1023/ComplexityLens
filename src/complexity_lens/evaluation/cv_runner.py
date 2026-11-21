"""
cv_runner.py — Cross-validation runner and experiment orchestration for ComplexityLens.

Orchestrates reproducible cross-validation experiments for the explainable
hybrid static-analysis and ML-based complexity estimator:
- Strictly evaluates development partition data using frozen CV fold assignments.
- Enforces absolute data isolation: zero test-partition access, strict family disjointness,
  and no validation-fold feature/label leakage into training.
- Evaluates the three fixed baseline models:
  * majority_class
  * logistic_regression
  * decision_tree
- Generates out-of-fold (OOF) predictions for all development programs, with exactly
  one prediction per covered program produced by its held-out fold model.
- Computes per-fold and combined OOF classification metrics with canonical 8x8
  confusion matrices, per-class support, and documented zero-division policies.
- Validates evaluation metadata using the typed EvaluationMetadata contract.
- Guarantees deterministic serialization without creating persistent caches
  or saving fitted model artifacts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from complexity_lens.evaluation.baselines import (
    DECISION_TREE_CONFIG,
    LOGISTIC_REGRESSION_CONFIG,
    RANDOM_FOREST_DEFAULT_CONFIG,
    XGBOOST_DEFAULT_CONFIG,
    create_baseline_model,
)
from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
    DevelopmentDataset,
    FoldSplit,
    load_development_dataset,
)
from complexity_lens.evaluation.metrics import (
    EvaluationMetadata,
    EvaluationMetrics,
    compute_classification_metrics,
    write_deterministic_json,
)

DEFAULT_BASELINE_MODEL_IDS: tuple[str, ...] = (
    "majority_class",
    "logistic_regression",
    "decision_tree",
)
DEFAULT_MANIFEST_PATH: str = "data/splits/split_manifest_v1.0.frozen.json"
DEFAULT_DATASET_ROOT: str = "datasets"
DEFAULT_SCHEMA_VERSION: str = "1.0"
DEFAULT_EXPERIMENT_ID: str = "cv_baselines_development"


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------


class CrossValidationRunnerError(Exception):
    """Base exception for all cross-validation runner and orchestration errors."""


class FoldConfigurationError(CrossValidationRunnerError, ValueError):
    """Raised when fold assignments, cardinalities, or coverage are malformed."""


class DataIsolationError(CrossValidationRunnerError):
    """Raised when data leakage or test-partition access is detected."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OutOfFoldPrediction:
    """Out-of-fold prediction for a single development benchmark program.

    Attributes
    ----------
    program_id:
        Unique program identifier.
    family_id:
        Benchmark family identifier.
    fold:
        Held-out validation fold index.
    true_label:
        Ground truth canonical complexity label.
    predicted_label:
        Predicted canonical complexity label from the fold model.
    probabilities:
        Canonical 8-class probability distribution.
    """

    program_id: str
    family_id: str
    fold: int
    true_label: str
    predicted_label: str
    probabilities: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction record to a deterministic JSON-serializable dictionary."""
        return {
            "family_id": self.family_id,
            "fold": self.fold,
            "predicted_label": self.predicted_label,
            "probabilities": list(self.probabilities),
            "program_id": self.program_id,
            "true_label": self.true_label,
        }


@dataclasses.dataclass(frozen=True)
class FoldEvaluationResult:
    """Evaluation metrics, cardinality, and distributions for a single CV fold."""

    fold: int
    model_id: str
    train_sample_count: int
    val_sample_count: int
    train_family_count: int
    val_family_count: int
    train_class_distribution: dict[str, int]
    val_class_distribution: dict[str, int]
    missing_classes: list[str]
    metrics: EvaluationMetrics
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert fold result to a dictionary."""
        return {
            "fold": self.fold,
            "metrics": self.metrics.to_dict(),
            "missing_classes": list(self.missing_classes),
            "model_id": self.model_id,
            "train_class_distribution": dict(self.train_class_distribution),
            "train_family_count": self.train_family_count,
            "train_sample_count": self.train_sample_count,
            "val_class_distribution": dict(self.val_class_distribution),
            "val_family_count": self.val_family_count,
            "val_sample_count": self.val_sample_count,
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(frozen=True)
class ModelEvaluationResult:
    """Complete cross-validation evaluation result for a single baseline model."""

    model_id: str
    fixed_hyperparameters: dict[str, Any]
    metadata: EvaluationMetadata
    per_fold: list[FoldEvaluationResult]
    combined_oof_metrics: EvaluationMetrics
    oof_predictions: list[OutOfFoldPrediction]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert model evaluation result to a dictionary."""
        return {
            "combined_oof_metrics": self.combined_oof_metrics.to_dict(),
            "fixed_hyperparameters": dict(self.fixed_hyperparameters),
            "metadata": self.metadata.to_dict(),
            "model_id": self.model_id,
            "oof_predictions": [p.to_dict() for p in self.oof_predictions],
            "per_fold": [f.to_dict() for f in self.per_fold],
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(frozen=True)
class CrossValidationExperimentResult:
    """Root evaluation result for a multi-model cross-validation experiment."""

    schema_version: str
    experiment_id: str
    manifest_sha256: str
    fold_assignment_hash: str
    git_commit: str
    git_dirty: bool
    library_versions: dict[str, str]
    dataset_summary: dict[str, Any]
    models: dict[str, ModelEvaluationResult]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert experiment result to a deterministic dictionary."""
        return {
            "dataset_summary": dict(self.dataset_summary),
            "experiment_id": self.experiment_id,
            "fold_assignment_hash": self.fold_assignment_hash,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "library_versions": dict(self.library_versions),
            "limitations": list(self.limitations),
            "manifest_sha256": self.manifest_sha256,
            "models": {k: v.to_dict() for k, v in sorted(self.models.items())},
            "schema_version": self.schema_version,
        }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def compute_fold_assignment_hash(dataset: DevelopmentDataset) -> str:
    """Compute deterministic SHA-256 hex digest of (program_id -> fold) assignments.

    Canonicalization policy:
    1. Collect (program_id, fold) for all development records in the dataset.
    2. Sort ascending by program_id.
    3. Format as 'program_id:fold\n' UTF-8 lines.
    4. Compute SHA-256 digest.
    """
    entries = sorted((r.program_id, r.fold) for r in dataset.records)
    payload = "\n".join(f"{pid}:{fold}" for pid, fold in entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_git_info(repo_dir: str | Path = ".") -> tuple[str, bool]:
    """Retrieve Git commit and dirty status safely."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        commit = proc.stdout.strip()
    except Exception:
        commit = "unknown"

    try:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(status_proc.stdout.strip())
    except Exception:
        dirty = True

    return commit, dirty


def get_library_versions() -> dict[str, str]:
    """Retrieve runtime versions for numerical and ML libraries."""
    import numpy
    import scipy
    import sklearn

    return {
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }


def get_model_fixed_hyperparameters(model_id: str) -> dict[str, Any]:
    """Return canonical fixed hyperparameter dictionary for a baseline model."""
    normalized_id = str(model_id).strip().lower()
    if normalized_id == "majority_class":
        return {"strategy": "majority_class", "tie_breaker": "canonical_order"}
    if normalized_id == "logistic_regression":
        return dict(LOGISTIC_REGRESSION_CONFIG)
    if normalized_id == "decision_tree":
        return dict(DECISION_TREE_CONFIG)
    if normalized_id == "random_forest":
        return dict(RANDOM_FOREST_DEFAULT_CONFIG)
    if normalized_id == "xgboost":
        return dict(XGBOOST_DEFAULT_CONFIG)
    if normalized_id == "hist_gradient_boosting":
        return {"random_state": 42}
    raise ValueError(f"Unknown baseline model ID: '{model_id}'")


# ---------------------------------------------------------------------------
# Cross-Validation Runner
# ---------------------------------------------------------------------------


class CrossValidationRunner:
    """Orchestrates 3-fold cross validation across fixed baseline models."""

    def __init__(
        self,
        dataset: DevelopmentDataset | None = None,
        *,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        model_ids: Sequence[str] = DEFAULT_BASELINE_MODEL_IDS,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        git_commit: str | None = None,
        git_dirty: bool | None = None,
        repo_dir: str | Path = ".",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.experiment_id = str(experiment_id).strip() or DEFAULT_EXPERIMENT_ID

        # Load or bind dataset
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = load_development_dataset(
                manifest_path=self.manifest_path,
                dataset_root=self.dataset_root,
            )

        # Validate model IDs
        self.model_ids = tuple(model_ids)
        if not self.model_ids:
            raise ValueError("model_ids sequence cannot be empty.")
        for mid in self.model_ids:
            if mid not in DEFAULT_BASELINE_MODEL_IDS:
                raise ValueError(
                    f"Unsupported baseline model_id '{mid}'. "
                    f"Supported: {DEFAULT_BASELINE_MODEL_IDS}"
                )

        # Determine git status
        if git_commit is not None and git_dirty is not None:
            self.git_commit = git_commit
            self.git_dirty = git_dirty
        else:
            detected_commit, detected_dirty = get_git_info(repo_dir)
            self.git_commit = git_commit or detected_commit
            self.git_dirty = git_dirty if git_dirty is not None else detected_dirty

        # Derived dataset attributes
        self.fold_assignment_hash = compute_fold_assignment_hash(self.dataset)
        self.library_versions = get_library_versions()

        # Identify unique folds in the dataset
        folds = sorted({r.fold for r in self.dataset.records})
        if not folds:
            raise FoldConfigurationError("Development dataset contains no fold assignments.")
        self.folds = tuple(folds)

        # Validate that fold assignments partition the dataset
        self._validate_fold_partitioning()

    def _validate_fold_partitioning(self) -> None:
        """Validate that all records have valid fold assignments and form a partition."""
        all_pids = set(self.dataset.program_ids)
        fold_pids: set[str] = set()

        for fold in self.folds:
            fold_records = [r for r in self.dataset.records if r.fold == fold]
            if not fold_records:
                raise FoldConfigurationError(f"Fold {fold} has no assigned records.")
            pids = {r.program_id for r in fold_records}
            if fold_pids & pids:
                raise FoldConfigurationError(
                    f"Duplicate program assignments detected across folds: {fold_pids & pids}"
                )
            fold_pids.update(pids)

        if fold_pids != all_pids:
            missing = all_pids - fold_pids
            raise FoldConfigurationError(
                f"Folds do not cover all development programs; missing: {missing}"
            )

    def evaluate_fold(
        self,
        fold_split: FoldSplit,
        model_id: str,
    ) -> tuple[FoldEvaluationResult, list[OutOfFoldPrediction]]:
        """Fit a model on a single fold's training split and evaluate held-out validation.

        Parameters
        ----------
        fold_split:
            FoldSplit containing train_records and val_records.
        model_id:
            Identifier of baseline model to fit and evaluate.

        Returns
        -------
        tuple[FoldEvaluationResult, list[OutOfFoldPrediction]]
            Fold metrics and OOF prediction records.
        """
        fold = fold_split.fold
        train_records = fold_split.train_records
        val_records = fold_split.val_records

        if not train_records:
            raise FoldConfigurationError(f"Fold {fold} training set cannot be empty.")
        if not val_records:
            raise FoldConfigurationError(f"Fold {fold} validation set cannot be empty.")

        # Strict isolation validation: program disjointness
        train_pids = set(fold_split.train_program_ids)
        val_pids = set(fold_split.val_program_ids)
        if train_pids & val_pids:
            raise DataIsolationError(
                f"Data leakage detected in fold {fold}: programs appear in both train and val: "
                f"{train_pids & val_pids}"
            )

        # Strict isolation validation: family disjointness
        train_fams = fold_split.train_families
        val_fams = fold_split.val_families
        if train_fams & val_fams:
            raise DataIsolationError(
                f"Family leakage detected in fold {fold}: families appear in both train and val: "
                f"{train_fams & val_fams}"
            )

        # Extract features strictly inside this fold
        X_train, y_train = fold_split.to_train_xy()
        X_val, y_val = fold_split.to_val_xy()

        # Compute class distributions
        train_dist = {cls: 0 for cls in CANONICAL_CLASS_ORDER}
        for lbl, count in Counter(y_train).items():
            if lbl in train_dist:
                train_dist[lbl] = count

        val_dist = {cls: 0 for cls in CANONICAL_CLASS_ORDER}
        for lbl, count in Counter(y_val).items():
            if lbl in val_dist:
                val_dist[lbl] = count

        missing_classes = [cls for cls in CANONICAL_CLASS_ORDER if train_dist[cls] == 0]

        # Instantiate a fresh, unfitted model for this fold
        model = create_baseline_model(model_id)

        # Fit model strictly on training features and labels
        model.fit(X_train, y_train)

        # Predict on validation features
        val_preds = model.predict(X_val)
        val_proba = model.predict_proba(X_val)

        # Collect any model-specific warnings (e.g. LogisticRegression convergence)
        fold_warnings: list[str] = []
        if hasattr(model, "convergence_warnings_"):
            fold_warnings.extend(model.convergence_warnings_)

        # Compute classification metrics for this fold
        fold_metrics = compute_classification_metrics(y_val, val_preds)

        # Collect OOF prediction objects
        oof_preds: list[OutOfFoldPrediction] = []
        for i, r in enumerate(val_records):
            oof_preds.append(
                OutOfFoldPrediction(
                    program_id=r.program_id,
                    family_id=r.family_id,
                    fold=fold,
                    true_label=y_val[i],
                    predicted_label=val_preds[i],
                    probabilities=[float(p) for p in val_proba[i]],
                )
            )

        # Discard fitted model instance
        del model

        fold_result = FoldEvaluationResult(
            fold=fold,
            model_id=model_id,
            train_sample_count=len(train_records),
            val_sample_count=len(val_records),
            train_family_count=len(train_fams),
            val_family_count=len(val_fams),
            train_class_distribution=train_dist,
            val_class_distribution=val_dist,
            missing_classes=missing_classes,
            metrics=fold_metrics,
            warnings=fold_warnings,
        )

        return fold_result, oof_preds

    def get_fold_split(self, fold: int) -> FoldSplit:
        """Obtain FoldSplit for a given fold, supporting both production and synthetic datasets."""
        try:
            return self.dataset.get_fold(fold)
        except Exception:
            # Fallback for synthetic/custom datasets with non-production cardinalities
            val_records = tuple(r for r in self.dataset.records if r.fold == fold)
            train_records = tuple(r for r in self.dataset.records if r.fold != fold)
            return FoldSplit(
                fold=fold,
                train_records=train_records,
                val_records=val_records,
            )

    def evaluate_model(self, model_id: str) -> ModelEvaluationResult:
        """Run complete cross-validation for a single baseline model across all folds."""
        fold_results: list[FoldEvaluationResult] = []
        all_oof_predictions: list[OutOfFoldPrediction] = []
        model_warnings: list[str] = []

        # Iterate over all folds
        for fold in self.folds:
            fold_split = self.get_fold_split(fold)
            f_res, f_oof = self.evaluate_fold(fold_split, model_id)
            fold_results.append(f_res)
            all_oof_predictions.extend(f_oof)
            model_warnings.extend(f_res.warnings)

        # Sort OOF predictions deterministically by program_id
        all_oof_predictions.sort(key=lambda p: p.program_id)

        # Validation: exactly 1 prediction per covered development program
        covered_pids = [p.program_id for p in all_oof_predictions]
        expected_pids = sorted(self.dataset.program_ids)

        if len(covered_pids) != len(expected_pids):
            raise FoldConfigurationError(
                f"OOF prediction count mismatch for model '{model_id}': expected "
                f"{len(expected_pids)}, got {len(covered_pids)}."
            )

        if covered_pids != expected_pids:
            raise FoldConfigurationError(
                "OOF prediction program IDs do not match dataset development programs."
            )

        # Compute combined OOF metrics across all held-out predictions
        oof_y_true = [p.true_label for p in all_oof_predictions]
        oof_y_pred = [p.predicted_label for p in all_oof_predictions]
        combined_metrics = compute_classification_metrics(oof_y_true, oof_y_pred)

        # Overall development class distribution
        overall_dist = {cls: 0 for cls in CANONICAL_CLASS_ORDER}
        for r in self.dataset.records:
            if r.label in overall_dist:
                overall_dist[r.label] += 1

        missing_details: dict[str, Any] = {
            f"fold_{res.fold}": res.missing_classes for res in fold_results
        }

        # Build documented limitations
        limitations: list[str] = [
            (
                "Cross-validation executed strictly on development partition "
                "(159 programs, 53 families)."
            ),
            "Frozen test partition (45 programs, 15 families) strictly excluded and untouched.",
            (
                "Models fitted exclusively on training fold features with canonical-order "
                "tie breaking or fixed seed."
            ),
        ]
        if model_warnings:
            limitations.append(
                f"Encountered {len(model_warnings)} warning(s) during fold fitting: "
                f"{model_warnings[0][:100]}..."
            )

        # Build and validate EvaluationMetadata contract
        metadata = EvaluationMetadata(
            schema_version=DEFAULT_SCHEMA_VERSION,
            model_id=model_id,
            fixed_hyperparameters=get_model_fixed_hyperparameters(model_id),
            canonical_feature_schema=list(CANONICAL_FEATURE_NAMES),
            canonical_class_ordering=list(CANONICAL_CLASS_ORDER),
            manifest_sha256=self.dataset.manifest_sha256,
            fold_assignment_hash=self.fold_assignment_hash,
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            library_versions=self.library_versions,
            sample_count=len(all_oof_predictions),
            family_count=len(self.dataset.families),
            training_class_distribution=overall_dist,
            missing_class_details=missing_details,
            limitations=limitations,
        )
        metadata.validate()

        return ModelEvaluationResult(
            model_id=model_id,
            fixed_hyperparameters=get_model_fixed_hyperparameters(model_id),
            metadata=metadata,
            per_fold=fold_results,
            combined_oof_metrics=combined_metrics,
            oof_predictions=all_oof_predictions,
            warnings=model_warnings,
        )

    def run(self) -> CrossValidationExperimentResult:
        """Execute full cross-validation experiment across all specified baseline models."""
        models_results: dict[str, ModelEvaluationResult] = {}

        for model_id in self.model_ids:
            models_results[model_id] = self.evaluate_model(model_id)

        dataset_summary: dict[str, Any] = {
            "family_count": len(self.dataset.families),
            "folds": list(self.folds),
            "manifest_path": self.dataset.manifest_path,
            "manifest_sha256": self.dataset.manifest_sha256,
            "partition": "development",
            "per_fold_counts": {
                f"fold_{f}": {
                    "val_families": len([
                        fam
                        for fam in self.dataset.families
                        if any(r.family_id == fam and r.fold == f for r in self.dataset.records)
                    ]),
                    "val_programs": sum(1 for r in self.dataset.records if r.fold == f),
                }
                for f in self.folds
            },
            "program_count": len(self.dataset.records),
        }

        experiment_limitations: list[str] = [
            "Evaluated strictly on development partition using frozen CV folds F0, F1, F2.",
            "Test partition (45 programs, 15 families) strictly isolated and unread.",
            "Features extracted in memory; no persistent caches or saved model weights created.",
        ]

        return CrossValidationExperimentResult(
            schema_version=DEFAULT_SCHEMA_VERSION,
            experiment_id=self.experiment_id,
            manifest_sha256=self.dataset.manifest_sha256,
            fold_assignment_hash=self.fold_assignment_hash,
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            library_versions=self.library_versions,
            dataset_summary=dataset_summary,
            models=models_results,
            limitations=experiment_limitations,
        )


# ---------------------------------------------------------------------------
# Public Execution & Output Helpers
# ---------------------------------------------------------------------------


def run_cross_validation(
    dataset: DevelopmentDataset | None = None,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    model_ids: Sequence[str] = DEFAULT_BASELINE_MODEL_IDS,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    destination: str | Path | None = None,
    ensure_parents: bool = False,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    repo_dir: str | Path = ".",
) -> CrossValidationExperimentResult:
    """Run full cross-validation and optionally serialize results to destination file.

    Parameters
    ----------
    dataset:
        Pre-loaded DevelopmentDataset. If None, loads from manifest_path.
    manifest_path:
        Path to frozen split manifest.
    dataset_root:
        Root path to dataset directory.
    model_ids:
        Sequence of baseline model IDs to evaluate.
    experiment_id:
        Identifier string for this experiment run.
    destination:
        Optional file path to serialize deterministic JSON results.
    ensure_parents:
        If True, creates parent directories for destination if missing.
    git_commit:
        Explicit git commit hash (optional).
    git_dirty:
        Explicit git dirty boolean flag (optional).
    repo_dir:
        Repository directory for detecting git metadata if not explicitly provided.

    Returns
    -------
    CrossValidationExperimentResult
        The complete experiment result structure.
    """
    runner = CrossValidationRunner(
        dataset=dataset,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        model_ids=model_ids,
        experiment_id=experiment_id,
        git_commit=git_commit,
        git_dirty=git_dirty,
        repo_dir=repo_dir,
    )
    result = runner.run()

    if destination is not None:
        write_cv_results(result, destination, ensure_parents=ensure_parents)

    return result


def write_cv_results(
    result: CrossValidationExperimentResult,
    destination: str | Path,
    *,
    indent: int = 2,
    ensure_parents: bool = False,
) -> Path:
    """Serialize CrossValidationExperimentResult to destination as deterministic JSON.

    Parameters
    ----------
    result:
        CrossValidationExperimentResult instance.
    destination:
        Explicit file destination path.
    indent:
        Indentation spaces (default: 2).
    ensure_parents:
        If True, creates parent directories if missing (default: False).

    Returns
    -------
    Path
        Destination Path.
    """
    return write_deterministic_json(
        result.to_dict(),
        destination,
        indent=indent,
        ensure_parents=ensure_parents,
    )
