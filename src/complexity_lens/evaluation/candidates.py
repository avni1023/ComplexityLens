"""
candidates.py — Candidate model selection, persistence, and evaluation for Day 11.

Implements Random Forest and XGBoost candidate evaluations for ComplexityLens:
- Fixed candidate parameter spaces (<= 4 configurations per model).
- Documented simplicity orderings and 4-tier deterministic tie-breaking.
- Pooled out-of-fold cross-validation on the development partition (159 programs, 53 families).
- Labeling winning candidate score as "Selection-biased CV estimate".
- Refitting winning models exclusively on the full development partition.
- Native artifact persistence (joblib for Random Forest, native JSON for XGBoost).
- Cryptographic SHA-256 verification and path-traversal / symlink protection in safe loader.
- Out-of-fold per-family comparison against the strongest Day 10 baseline (logistic_regression).
- Feature importance recording as descriptive metadata.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

from complexity_lens.evaluation.baselines import (
    DEFAULT_RANDOM_STATE,
    XGBoostBaseline,
    create_baseline_model,
)
from complexity_lens.evaluation.cv_runner import (
    compute_fold_assignment_hash,
    get_library_versions,
)
from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
    DevelopmentDataset,
    load_development_dataset,
)
from complexity_lens.evaluation.metrics import (
    compute_classification_metrics,
    write_deterministic_json,
)

# ---------------------------------------------------------------------------
# Prescribed Candidate Spaces and Simplicity Rankings
# ---------------------------------------------------------------------------

# Random Forest: 4 configurations
# Simplicity ordering rationale:
# 1. Constrained max_depth (6) is simpler than unconstrained depth (None).
# 2. More regularized leaf constraint (min_samples_leaf=2) is simpler than less regularized (1).
RF_CANDIDATE_CONFIGS: tuple[dict[str, Any], ...] = (
    {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 2, "simplicity_rank": 1},
    {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 1, "simplicity_rank": 2},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "simplicity_rank": 3},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "simplicity_rank": 4},
)

# XGBoost: 4 configurations
# Simplicity ordering rationale:
# 1. Shallower trees (max_depth=2) are simpler than deeper trees (max_depth=3).
# 2. Fewer boosting rounds (n_estimators=100) are simpler than more rounds (200).
XGB_CANDIDATE_CONFIGS: tuple[dict[str, Any], ...] = (
    {"learning_rate": 0.1, "max_depth": 2, "n_estimators": 100, "simplicity_rank": 1},
    {"learning_rate": 0.1, "max_depth": 2, "n_estimators": 200, "simplicity_rank": 2},
    {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100, "simplicity_rank": 3},
    {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200, "simplicity_rank": 4},
)

SELECTION_BIASED_LABEL: str = "Selection-biased CV estimate"
DEFAULT_ARTIFACTS_DIR: str = "artifacts/models"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CandidateError(Exception):
    """Base exception for candidate model selection and persistence."""


class ArtifactSecurityError(CandidateError, ValueError):
    """Raised when an artifact or metadata path violates filesystem boundaries."""


class ArtifactIntegrityError(CandidateError, ValueError):
    """Raised when an artifact fails SHA-256 verification or metadata validation."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CandidateEvaluationResult:
    """Evaluation result for a single candidate hyperparameter configuration."""

    model_id: str
    config_id: str
    hyperparameters: dict[str, Any]
    simplicity_rank: int
    pooled_oof_accuracy: float
    pooled_oof_macro_f1: float
    oof_predictions: tuple[tuple[str, str, str, bool], ...]  # (pid, true_lbl, pred_lbl, correct)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "hyperparameters": dict(self.hyperparameters),
            "model_id": self.model_id,
            "pooled_oof_accuracy": round(self.pooled_oof_accuracy, 6),
            "pooled_oof_macro_f1": round(self.pooled_oof_macro_f1, 6),
            "simplicity_rank": self.simplicity_rank,
        }


@dataclasses.dataclass(frozen=True)
class CandidateSelectionSummary:
    """Complete summary of candidate evaluation and winner selection for a model family."""

    model_id: str
    candidates: tuple[CandidateEvaluationResult, ...]
    winner: CandidateEvaluationResult
    winning_score_label: str = SELECTION_BIASED_LABEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "model_id": self.model_id,
            "winner": self.winner.to_dict(),
            "winning_score_label": self.winning_score_label,
            "winning_score_value": round(self.winner.pooled_oof_macro_f1, 6),
        }


# ---------------------------------------------------------------------------
# Model Selection and Cross-Validation
# ---------------------------------------------------------------------------


def evaluate_single_candidate(
    model_id: str,
    config: dict[str, Any],
    dataset: DevelopmentDataset,
    config_id: str = "",
) -> CandidateEvaluationResult:
    """Evaluate a single hyperparameter configuration using 3-fold grouped cross-validation."""
    clean_params = {k: v for k, v in config.items() if k != "simplicity_rank"}
    simplicity_rank = int(config.get("simplicity_rank", 999))

    folds = [dataset.get_fold(f) for f in (0, 1, 2)]
    oof_records: list[tuple[str, str, str, bool]] = []
    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for fold_split in folds:
        X_train, y_train = fold_split.to_train_xy()
        X_val, y_val = fold_split.to_val_xy()

        model = create_baseline_model(model_id, **clean_params)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)

        for rec, true_lbl, pred_lbl in zip(fold_split.val_records, y_val, val_preds, strict=True):
            oof_records.append((rec.program_id, true_lbl, pred_lbl, true_lbl == pred_lbl))
            y_true_all.append(true_lbl)
            y_pred_all.append(pred_lbl)

    # Sort OOF records deterministically by program_id
    oof_records.sort(key=lambda r: r[0])

    metrics = compute_classification_metrics(y_true_all, y_pred_all)

    return CandidateEvaluationResult(
        model_id=model_id,
        config_id=config_id or f"{model_id}_rank{simplicity_rank}",
        hyperparameters=clean_params,
        simplicity_rank=simplicity_rank,
        pooled_oof_accuracy=float(metrics.accuracy),
        pooled_oof_macro_f1=float(metrics.macro_f1),
        oof_predictions=tuple(oof_records),
    )


def select_best_candidate(
    candidates: Sequence[CandidateEvaluationResult],
) -> CandidateEvaluationResult:
    """Apply the 4-tier fixed tie-breaking rule to select the winning configuration.

    Tie-breaking hierarchy:
    1. Higher pooled out-of-fold macro-F1 (over classes with support > 0).
    2. Higher pooled out-of-fold accuracy.
    3. Simpler configuration (lower simplicity rank).
    4. Lexicographically first configuration (by JSON-formatted hyperparameters).
    """
    if not candidates:
        raise ValueError("Candidates sequence cannot be empty.")

    def sort_key(c: CandidateEvaluationResult) -> tuple[float, float, int, str]:
        # Sort descending by macro_f1 and accuracy, ascending by simplicity_rank and lex
        lex_key = json.dumps(c.hyperparameters, sort_keys=True)
        return (
            -round(c.pooled_oof_macro_f1, 6),
            -round(c.pooled_oof_accuracy, 6),
            c.simplicity_rank,
            lex_key,
        )

    sorted_candidates = sorted(candidates, key=sort_key)
    return sorted_candidates[0]


def run_candidate_selection(
    model_id: str,
    dataset: DevelopmentDataset | None = None,
    candidate_configs: Sequence[dict[str, Any]] | None = None,
) -> CandidateSelectionSummary:
    """Evaluate all prescribed candidate configurations and select the winning configuration."""
    if dataset is None:
        dataset = load_development_dataset()

    if candidate_configs is None:
        if model_id == "random_forest":
            candidate_configs = RF_CANDIDATE_CONFIGS
        elif model_id == "xgboost":
            candidate_configs = XGB_CANDIDATE_CONFIGS
        else:
            raise ValueError(f"No prescribed candidate space for model_id: '{model_id}'")

    if len(candidate_configs) > 4:
        raise ValueError(
            f"Candidate space exceeds the maximum limit of 4 configurations: "
            f"got {len(candidate_configs)}."
        )

    evaluated: list[CandidateEvaluationResult] = []
    for idx, cfg in enumerate(candidate_configs, 1):
        cid = f"{model_id}_candidate_{idx}"
        res = evaluate_single_candidate(model_id, cfg, dataset, config_id=cid)
        evaluated.append(res)

    winner = select_best_candidate(evaluated)
    return CandidateSelectionSummary(
        model_id=model_id,
        candidates=tuple(evaluated),
        winner=winner,
    )


# ---------------------------------------------------------------------------
# Final Development-Partition Refitting
# ---------------------------------------------------------------------------


def refit_on_full_development_partition(
    model_id: str,
    hyperparameters: dict[str, Any],
    dataset: DevelopmentDataset | None = None,
) -> tuple[Any, dict[str, float]]:
    """Refit the selected model configuration on all 159 development programs.

    Returns
    -------
    tuple[Any, dict[str, float]]
        (fitted_model, feature_importances_dict)
    """
    if dataset is None:
        dataset = load_development_dataset()

    X_dev = [r.features.to_list() for r in dataset.records]
    y_dev = [r.label for r in dataset.records]

    model = create_baseline_model(model_id, **hyperparameters)
    model.fit(X_dev, y_dev)

    # Extract feature importances
    importances_raw = getattr(model, "feature_importances_", None)
    if importances_raw is not None and len(importances_raw) == len(CANONICAL_FEATURE_NAMES):
        importances = {
            name: float(round(val, 6))
            for name, val in zip(CANONICAL_FEATURE_NAMES, importances_raw, strict=True)
        }
    else:
        importances = {name: 0.0 for name in CANONICAL_FEATURE_NAMES}

    return model, importances


# ---------------------------------------------------------------------------
# Artifact Persistence and Safe Loading
# ---------------------------------------------------------------------------


def persist_development_model(
    model: Any,
    selection_summary: CandidateSelectionSummary,
    feature_importances: dict[str, float],
    dataset: DevelopmentDataset,
    output_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> tuple[Path, Path, str]:
    """Persist the final development model and its rich metadata safely.

    Artifact Formats:
    - Random Forest: joblib (.joblib)
    - XGBoost: native JSON (.json)

    Returns
    -------
    tuple[Path, Path, str]
        (model_path, metadata_path, artifact_sha256)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_id = selection_summary.model_id
    if model_id == "random_forest":
        artifact_filename = "random_forest_dev_fit.joblib"
        artifact_format = "joblib"
        artifact_file = out_path / artifact_filename
        joblib.dump(model, str(artifact_file))
    elif model_id == "xgboost":
        artifact_filename = "xgboost_dev_fit.json"
        artifact_format = "native_xgboost_json"
        artifact_file = out_path / artifact_filename
        model.save_model(str(artifact_file))
    else:
        raise ValueError(f"Unsupported model_id for persistence: '{model_id}'")

    # Compute SHA-256 of the model artifact file
    artifact_bytes = artifact_file.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()

    metadata_filename = f"{model_id}_metadata.json"
    metadata_file = out_path / metadata_filename

    metadata = {
        "artifact_file": artifact_filename,
        "artifact_format": artifact_format,
        "artifact_sha256": artifact_sha256,
        "canonical_class_order": list(CANONICAL_CLASS_ORDER),
        "canonical_feature_names": list(CANONICAL_FEATURE_NAMES),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dependency_versions": get_library_versions(),
        "feature_importances": dict(feature_importances),
        "final_development_fit_status": "fitted_full_development_partition",
        "fold_assignment_hash": compute_fold_assignment_hash(dataset),
        "model_class": model.__class__.__name__,
        "model_id": model_id,
        "random_seeds_and_determinism": {
            "n_jobs": 1,
            "random_state": DEFAULT_RANDOM_STATE,
        },
        "selected_hyperparameters": dict(selection_summary.winner.hyperparameters),
        "selection_evaluations": [c.to_dict() for c in selection_summary.candidates],
        "selection_metric": "pooled_oof_macro_f1",
        "training_class_mapping": list(model._classes_),
        "winning_accuracy": round(selection_summary.winner.pooled_oof_accuracy, 6),
        "winning_score_label": SELECTION_BIASED_LABEL,
        "winning_score_value": round(selection_summary.winner.pooled_oof_macro_f1, 6),
    }

    write_deterministic_json(metadata, metadata_file)
    return artifact_file, metadata_file, artifact_sha256


def load_model_artifact(
    model_id: str,
    artifact_path: str | Path,
    metadata_path: str | Path,
    allowed_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> tuple[Any, dict[str, Any]]:
    """Safely load a persisted model artifact and verify metadata integrity.

    Security & Integrity Guarantees:
    - Resolves all paths and asserts containment within allowed_dir.
    - Rejects path traversal and symlinks escaping allowed_dir.
    - Verifies artifact SHA-256 against metadata artifact_sha256.
    - Validates feature schema, class order, and model identity.
    - Loads using native format loaders.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        (loaded_model, metadata_dict)
    """
    allowed_root = Path(allowed_dir).resolve()
    resolved_artifact = Path(artifact_path).resolve()
    resolved_metadata = Path(metadata_path).resolve()

    # Path traversal & containment checks
    try:
        resolved_artifact.relative_to(allowed_root)
        resolved_metadata.relative_to(allowed_root)
    except ValueError as e:
        raise ArtifactSecurityError(
            f"Path traversal or escaping path rejected: path is not inside "
            f"allowed directory '{allowed_root}'."
        ) from e

    # Symlink escape verification: ensure the target is also within allowed_root
    if resolved_artifact.is_symlink() or resolved_metadata.is_symlink():
        raise ArtifactSecurityError("Symlinked artifacts are rejected for security.")

    if not resolved_artifact.exists():
        raise ArtifactIntegrityError(f"Artifact file not found: {resolved_artifact}")
    if not resolved_metadata.exists():
        raise ArtifactIntegrityError(f"Metadata file not found: {resolved_metadata}")

    # Load and validate metadata
    try:
        metadata = json.loads(resolved_metadata.read_text(encoding="utf-8"))
    except Exception as e:
        raise ArtifactIntegrityError(f"Malformed metadata JSON: {e}") from e

    expected_sha = metadata.get("artifact_sha256")
    if not expected_sha:
        raise ArtifactIntegrityError("Metadata is missing 'artifact_sha256' field.")

    actual_sha = hashlib.sha256(resolved_artifact.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ArtifactIntegrityError(
            f"Artifact SHA-256 mismatch: expected {expected_sha}, got {actual_sha}."
        )

    # Validate model identity and schema
    if metadata.get("model_id") != model_id:
        raise ArtifactIntegrityError(
            f"Model ID mismatch: expected '{model_id}', metadata says '{metadata.get('model_id')}'."
        )
    if metadata.get("canonical_feature_names") != list(CANONICAL_FEATURE_NAMES):
        raise ArtifactIntegrityError(
            "Metadata feature schema does not match canonical 26 features."
        )
    if metadata.get("canonical_class_order") != list(CANONICAL_CLASS_ORDER):
        raise ArtifactIntegrityError("Metadata class order does not match canonical 8 classes.")

    # Native loading
    if model_id == "random_forest":
        loaded_model = joblib.load(str(resolved_artifact))
    elif model_id == "xgboost":
        training_classes = metadata.get("training_class_mapping", list(CANONICAL_CLASS_ORDER))
        loaded_model = XGBoostBaseline(**metadata.get("selected_hyperparameters", {}))
        loaded_model.load_model(str(resolved_artifact), classes=training_classes)
    else:
        raise ValueError(f"Unsupported model_id for loading: '{model_id}'")

    return loaded_model, metadata


# ---------------------------------------------------------------------------
# Per-Family Comparison Against Day 10 Baseline
# ---------------------------------------------------------------------------


def generate_per_family_comparison(
    candidate_summary: CandidateSelectionSummary,
    dataset: DevelopmentDataset,
    baseline_id: str = "logistic_regression",
) -> dict[str, Any]:
    """Compare candidate model OOF predictions vs best Day 10 baseline per family."""
    # Run Day 10 baseline 3-fold CV to collect out-of-fold predictions
    folds = [dataset.get_fold(f) for f in (0, 1, 2)]
    baseline_oof: dict[str, str] = {}  # pid -> pred_label

    for fold_split in folds:
        X_train, y_train = fold_split.to_train_xy()
        X_val, _ = fold_split.to_val_xy()
        model = create_baseline_model(baseline_id)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        for rec, pred_lbl in zip(fold_split.val_records, preds, strict=True):
            baseline_oof[rec.program_id] = pred_lbl

    # Map candidate OOF predictions: pid -> (true_lbl, pred_lbl, correct)
    candidate_oof: dict[str, tuple[str, str, bool]] = {
        pid: (true_lbl, pred_lbl, correct)
        for pid, true_lbl, pred_lbl, correct in candidate_summary.winner.oof_predictions
    }

    family_records = dataset._records_by_family
    per_family: list[dict[str, Any]] = []

    counts = {"better": 0, "worse": 0, "tied": 0}

    for fam_id in sorted(family_records.keys()):
        recs = family_records[fam_id]
        prog_count = len(recs)
        base_correct = sum(1 for r in recs if baseline_oof.get(r.program_id) == r.label)
        cand_correct = sum(
            1 for r in recs if candidate_oof.get(r.program_id, (None, None, False))[2]
        )

        if cand_correct > base_correct:
            status = "better"
            counts["better"] += 1
        elif cand_correct < base_correct:
            status = "worse"
            counts["worse"] += 1
        else:
            status = "tied"
            counts["tied"] += 1

        disagreements = []
        for r in recs:
            b_pred = baseline_oof.get(r.program_id)
            c_pred = candidate_oof.get(r.program_id, (None, None, False))[1]
            if b_pred != c_pred:
                disagreements.append(
                    {
                        "baseline_pred": b_pred,
                        "candidate_pred": c_pred,
                        "program_id": r.program_id,
                        "true_label": r.label,
                    }
                )

        per_family.append(
            {
                "baseline_correct": base_correct,
                "candidate_correct": cand_correct,
                "disagreements": disagreements,
                "family_id": fam_id,
                "program_count": prog_count,
                "status": status,
            }
        )

    return {
        "baseline_id": baseline_id,
        "candidate_model_id": candidate_summary.model_id,
        "families": per_family,
        "summary": {
            "better": counts["better"],
            "tied": counts["tied"],
            "total_families": len(per_family),
            "worse": counts["worse"],
        },
        "winner_hyperparameters": dict(candidate_summary.winner.hyperparameters),
    }


# ---------------------------------------------------------------------------
# Master Day 11 Pipeline Runner
# ---------------------------------------------------------------------------


def run_day11_pipeline(
    output_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    dataset: DevelopmentDataset | None = None,
) -> dict[str, Any]:
    """Execute complete Day 11 candidate selection, refit, and comparison."""
    if dataset is None:
        dataset = load_development_dataset()

    # 1. Run Candidate Selection for Random Forest
    rf_summary_1 = run_candidate_selection("random_forest", dataset)
    rf_summary_2 = run_candidate_selection("random_forest", dataset)
    # Programmatic assertion of determinism across repeat runs
    assert rf_summary_1.winner.hyperparameters == rf_summary_2.winner.hyperparameters
    assert rf_summary_1.winner.pooled_oof_macro_f1 == rf_summary_2.winner.pooled_oof_macro_f1
    assert rf_summary_1.winner.pooled_oof_accuracy == rf_summary_2.winner.pooled_oof_accuracy

    # 2. Run Candidate Selection for XGBoost
    xgb_summary_1 = run_candidate_selection("xgboost", dataset)
    xgb_summary_2 = run_candidate_selection("xgboost", dataset)
    # Programmatic assertion of determinism across repeat runs
    assert xgb_summary_1.winner.hyperparameters == xgb_summary_2.winner.hyperparameters
    assert xgb_summary_1.winner.pooled_oof_macro_f1 == xgb_summary_2.winner.pooled_oof_macro_f1
    assert xgb_summary_1.winner.pooled_oof_accuracy == xgb_summary_2.winner.pooled_oof_accuracy

    # 3. Refit winning models on full development partition
    rf_model, rf_importances = refit_on_full_development_partition(
        "random_forest", rf_summary_1.winner.hyperparameters, dataset
    )
    xgb_model, xgb_importances = refit_on_full_development_partition(
        "xgboost", xgb_summary_1.winner.hyperparameters, dataset
    )

    # 4. Persist development model artifacts and metadata
    rf_art, rf_meta, rf_sha = persist_development_model(
        rf_model, rf_summary_1, rf_importances, dataset, output_dir=output_dir
    )
    xgb_art, xgb_meta, xgb_sha = persist_development_model(
        xgb_model, xgb_summary_1, xgb_importances, dataset, output_dir=output_dir
    )

    # 5. Verify roundtrip loading with safe artifact loader
    rf_loaded, rf_meta_dict = load_model_artifact("random_forest", rf_art, rf_meta, output_dir)
    xgb_loaded, xgb_meta_dict = load_model_artifact("xgboost", xgb_art, xgb_meta, output_dir)

    # 6. Generate per-family comparisons vs Day 10 logistic_regression
    rf_family_comp = generate_per_family_comparison(rf_summary_1, dataset, "logistic_regression")
    xgb_family_comp = generate_per_family_comparison(xgb_summary_1, dataset, "logistic_regression")

    return {
        "per_family_comparisons": {
            "random_forest": rf_family_comp,
            "xgboost": xgb_family_comp,
        },
        "persisted_artifacts": {
            "random_forest": {
                "artifact_path": str(rf_art),
                "metadata_path": str(rf_meta),
                "sha256": rf_sha,
            },
            "xgboost": {
                "artifact_path": str(xgb_art),
                "metadata_path": str(xgb_meta),
                "sha256": xgb_sha,
            },
        },
        "random_forest": rf_summary_1.to_dict(),
        "xgboost": xgb_summary_1.to_dict(),
    }
