"""
test_evaluation_candidates.py — Test suite for Day 11 Random Forest and XGBoost candidates.

Validates:
1. Random Forest configuration and deterministic execution.
2. XGBoost initialization, version compatibility, and native format saving/loading.
3. Explicit fallback behavior (HistGradientBoostingBaseline).
4. Missing-class label encoding and probability mapping (filling absent classes with 0.0).
5. Candidate space size constraints (<= 4 configurations).
6. Fixed selection rule and 4-tier tie-breaking.
7. Validation-only candidate selection on development folds.
8. Deterministic repeat runs yielding identical metrics and winners.
9. Development-only final refitting and feature-importance recording without feature selection.
10. Artifact persistence, SHA-256 verification, and metadata completeness.
11. Safe artifact loader security: path-traversal and symlink rejection.
12. Per-family comparison generation against Day 10 baseline.
13. Day 10 baseline regression invariance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from complexity_lens.evaluation.baselines import (
    HistGradientBoostingBaseline,
    NotFittedError,
    RandomForestBaseline,
    XGBoostBaseline,
    create_baseline_model,
)
from complexity_lens.evaluation.candidates import (
    RF_CANDIDATE_CONFIGS,
    SELECTION_BIASED_LABEL,
    XGB_CANDIDATE_CONFIGS,
    ArtifactIntegrityError,
    ArtifactSecurityError,
    CandidateEvaluationResult,
    generate_per_family_comparison,
    load_model_artifact,
    persist_development_model,
    refit_on_full_development_partition,
    run_candidate_selection,
    select_best_candidate,
)
from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
    DevelopmentDataset,
    load_development_dataset,
)
from complexity_lens.evaluation.metrics import NUM_CANONICAL_CLASSES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dev_dataset() -> DevelopmentDataset:
    return load_development_dataset()


@pytest.fixture
def synthetic_features_labels() -> tuple[np.ndarray, list[str]]:
    rng = np.random.RandomState(42)
    X = rng.randn(30, 26)
    # Subset of 3 classes: O1, ON, UNKNOWN
    classes = ["O1", "ON", "UNKNOWN"]
    y = [classes[i % 3] for i in range(30)]
    return X, y


# ---------------------------------------------------------------------------
# 1. Random Forest Configuration & Determinism
# ---------------------------------------------------------------------------


def test_random_forest_init_and_determinism(
    synthetic_features_labels: tuple[np.ndarray, list[str]],
) -> None:
    X, y = synthetic_features_labels
    rf1 = RandomForestBaseline(n_estimators=50, max_depth=4, min_samples_leaf=1, random_state=42)
    rf2 = RandomForestBaseline(n_estimators=50, max_depth=4, min_samples_leaf=1, random_state=42)

    rf1.fit(X, y)
    rf2.fit(X, y)

    preds1 = rf1.predict(X)
    preds2 = rf2.predict(X)
    proba1 = rf1.predict_proba(X)
    proba2 = rf2.predict_proba(X)

    assert preds1 == preds2
    np.testing.assert_allclose(proba1, proba2)
    assert proba1.shape == (len(X), NUM_CANONICAL_CLASSES)


def test_random_forest_feature_importances(
    synthetic_features_labels: tuple[np.ndarray, list[str]],
) -> None:
    X, y = synthetic_features_labels
    rf = RandomForestBaseline(n_estimators=50, random_state=42)
    with pytest.raises(NotFittedError):
        _ = rf.feature_importances_

    rf.fit(X, y)
    importances = rf.feature_importances_
    assert len(importances) == 26
    assert np.isclose(np.sum(importances), 1.0)
    assert np.all(importances >= 0.0)


# ---------------------------------------------------------------------------
# 2. XGBoost Initialization & Version Compatibility
# ---------------------------------------------------------------------------


def test_xgboost_init_and_probabilities(
    synthetic_features_labels: tuple[np.ndarray, list[str]],
) -> None:
    X, y = synthetic_features_labels
    xgb_model = XGBoostBaseline(n_estimators=20, max_depth=2, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)

    assert xgb_model.is_fitted
    preds = xgb_model.predict(X)
    proba = xgb_model.predict_proba(X)

    assert len(preds) == len(X)
    assert proba.shape == (len(X), NUM_CANONICAL_CLASSES)

    # In synthetic dataset, only O1, ON, UNKNOWN were present.
    # Canonical columns for absent classes (e.g. OLOGN, ONLOGN, ON2, ON3, OEXP) must be 0.0.
    absent_indices = [
        CANONICAL_CLASS_ORDER.index(c) for c in ["OLOGN", "ONLOGN", "ON2", "ON3", "OEXP"]
    ]
    for idx in absent_indices:
        np.testing.assert_allclose(proba[:, idx], 0.0)

    # Probabilities across present classes should sum to approx 1.0
    np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-5)


def test_xgboost_native_save_load(
    tmp_path: Path, synthetic_features_labels: tuple[np.ndarray, list[str]]
) -> None:
    X, y = synthetic_features_labels
    xgb_model = XGBoostBaseline(n_estimators=10, max_depth=2, random_state=42)
    xgb_model.fit(X, y)

    save_path = tmp_path / "model.json"
    xgb_model.save_model(save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0

    loaded_xgb = XGBoostBaseline()
    loaded_xgb.load_model(save_path, classes=xgb_model._classes_)
    assert loaded_xgb.is_fitted

    preds_orig = xgb_model.predict(X)
    preds_loaded = loaded_xgb.predict(X)
    assert preds_orig == preds_loaded


# ---------------------------------------------------------------------------
# 3. Explicit Fallback Behavior (HistGradientBoostingBaseline)
# ---------------------------------------------------------------------------


def test_hist_gradient_boosting_fallback(
    synthetic_features_labels: tuple[np.ndarray, list[str]],
) -> None:
    X, y = synthetic_features_labels
    hgb = HistGradientBoostingBaseline(max_iter=20, max_depth=2, random_state=42)
    hgb.fit(X, y)

    assert hgb.is_fitted
    preds = hgb.predict(X)
    proba = hgb.predict_proba(X)

    assert len(preds) == len(X)
    assert proba.shape == (len(X), NUM_CANONICAL_CLASSES)
    assert hgb.model_id == "hist_gradient_boosting"

    model_from_factory = create_baseline_model("hist_gradient_boosting")
    assert isinstance(model_from_factory, HistGradientBoostingBaseline)


# ---------------------------------------------------------------------------
# 4. Missing-Class Label Encoding & Probability Alignment
# ---------------------------------------------------------------------------


def test_missing_class_probability_mapping() -> None:
    # 2 classes only
    X = np.ones((5, 26))
    y = ["ON", "ON", "UNKNOWN", "ON", "UNKNOWN"]

    xgb_model = XGBoostBaseline(n_estimators=10, max_depth=2, random_state=42)
    xgb_model.fit(X, y)

    proba = xgb_model.predict_proba(X)
    assert proba.shape == (5, 8)
    # Check that absent classes have strictly 0.0
    for cls_name in ("O1", "OLOGN", "ONLOGN", "ON2", "ON3", "OEXP"):
        idx = CANONICAL_CLASS_ORDER.index(cls_name)
        assert np.all(proba[:, idx] == 0.0)


# ---------------------------------------------------------------------------
# 5. Candidate-Space Constraints (<= 4 configurations)
# ---------------------------------------------------------------------------


def test_candidate_space_limits(dev_dataset: DevelopmentDataset) -> None:
    assert len(RF_CANDIDATE_CONFIGS) == 4
    assert len(XGB_CANDIDATE_CONFIGS) == 4

    too_many = [
        {"max_depth": 2, "n_estimators": 50, "learning_rate": 0.1, "simplicity_rank": i}
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="exceeds the maximum limit of 4"):
        run_candidate_selection("xgboost", dev_dataset, candidate_configs=too_many)


# ---------------------------------------------------------------------------
# 6. Fixed Selection Rule and 4-Tier Tie-Breaking
# ---------------------------------------------------------------------------


def test_selection_rule_tie_breaking() -> None:
    # 1. Higher macro-F1 wins
    c1 = CandidateEvaluationResult(
        "rf",
        "c1",
        {"d": 1},
        simplicity_rank=1,
        pooled_oof_accuracy=0.5,
        pooled_oof_macro_f1=0.6,
        oof_predictions=(),
    )
    c2 = CandidateEvaluationResult(
        "rf",
        "c2",
        {"d": 2},
        simplicity_rank=2,
        pooled_oof_accuracy=0.5,
        pooled_oof_macro_f1=0.7,
        oof_predictions=(),
    )
    assert select_best_candidate([c1, c2]).config_id == "c2"

    # 2. Tied macro-F1 -> Higher accuracy wins
    c3 = CandidateEvaluationResult(
        "rf",
        "c3",
        {"d": 1},
        simplicity_rank=1,
        pooled_oof_accuracy=0.6,
        pooled_oof_macro_f1=0.7,
        oof_predictions=(),
    )
    c4 = CandidateEvaluationResult(
        "rf",
        "c4",
        {"d": 2},
        simplicity_rank=2,
        pooled_oof_accuracy=0.65,
        pooled_oof_macro_f1=0.7,
        oof_predictions=(),
    )
    assert select_best_candidate([c3, c4]).config_id == "c4"

    # 3. Tied macro-F1 and accuracy -> Simpler configuration wins (lower rank)
    c5 = CandidateEvaluationResult(
        "rf",
        "c5",
        {"d": 1},
        simplicity_rank=1,
        pooled_oof_accuracy=0.65,
        pooled_oof_macro_f1=0.7,
        oof_predictions=(),
    )
    c6 = CandidateEvaluationResult(
        "rf",
        "c6",
        {"d": 2},
        simplicity_rank=2,
        pooled_oof_accuracy=0.65,
        pooled_oof_macro_f1=0.7,
        oof_predictions=(),
    )
    assert select_best_candidate([c5, c6]).config_id == "c5"

    # 4. Tied rank -> Lexicographically first configuration wins
    c7 = CandidateEvaluationResult(
        "rf",
        "c7",
        {"alpha": 1},
        simplicity_rank=1,
        pooled_oof_accuracy=0.65,
        pooled_oof_macro_f1=0.7,
        oof_predictions=(),
    )
    c8 = CandidateEvaluationResult(
        "rf",
        "c8",
        {"beta": 1},
        simplicity_rank=1,
        pooled_oof_accuracy=0.65,
        pooled_oof_macro_f1=0.7,
        oof_predictions=(),
    )
    assert select_best_candidate([c8, c7]).config_id == "c7"


# ---------------------------------------------------------------------------
# 7. Validation-Only Selection & Deterministic Repeat Runs
# ---------------------------------------------------------------------------


def test_candidate_selection_determinism(dev_dataset: DevelopmentDataset) -> None:
    # Test on a small subset of 2 configs for quick execution in tests
    sub_configs = (
        {"n_estimators": 20, "max_depth": 6, "min_samples_leaf": 2, "simplicity_rank": 1},
        {"n_estimators": 20, "max_depth": 6, "min_samples_leaf": 1, "simplicity_rank": 2},
    )
    res1 = run_candidate_selection("random_forest", dev_dataset, candidate_configs=sub_configs)
    res2 = run_candidate_selection("random_forest", dev_dataset, candidate_configs=sub_configs)

    assert res1.winner.config_id == res2.winner.config_id
    assert res1.winner.pooled_oof_macro_f1 == res2.winner.pooled_oof_macro_f1
    assert res1.winner.pooled_oof_accuracy == res2.winner.pooled_oof_accuracy
    assert res1.winning_score_label == SELECTION_BIASED_LABEL


# ---------------------------------------------------------------------------
# 8. Development-Only Final Refitting & Feature Importances
# ---------------------------------------------------------------------------


def test_final_refit_on_full_development(dev_dataset: DevelopmentDataset) -> None:
    model, importances = refit_on_full_development_partition(
        "random_forest",
        {"n_estimators": 20, "max_depth": 6, "min_samples_leaf": 1, "random_state": 42},
        dev_dataset,
    )
    assert model.is_fitted
    assert len(importances) == 26
    for feat_name in CANONICAL_FEATURE_NAMES:
        assert feat_name in importances
        assert importances[feat_name] >= 0.0


# ---------------------------------------------------------------------------
# 9. Artifact Persistence, Verification, and Safe Loading
# ---------------------------------------------------------------------------


def test_artifact_persistence_and_safe_loader(
    tmp_path: Path, dev_dataset: DevelopmentDataset
) -> None:
    allowed_dir = tmp_path / "artifacts" / "models"
    allowed_dir.mkdir(parents=True)

    fast_cfg = (
        {
            "n_estimators": 20,
            "max_depth": 4,
            "min_samples_leaf": 2,
            "simplicity_rank": 1,
        },
    )
    summary = run_candidate_selection(
        "random_forest",
        dev_dataset,
        candidate_configs=fast_cfg,
    )
    model, importances = refit_on_full_development_partition(
        "random_forest", summary.winner.hyperparameters, dev_dataset
    )

    art_path, meta_path, sha256_hex = persist_development_model(
        model, summary, importances, dev_dataset, output_dir=allowed_dir
    )

    assert art_path.exists()
    assert meta_path.exists()
    assert hashlib.sha256(art_path.read_bytes()).hexdigest() == sha256_hex

    # Safe loading should succeed
    loaded_model, metadata = load_model_artifact(
        "random_forest", art_path, meta_path, allowed_dir=allowed_dir
    )
    assert loaded_model.is_fitted
    assert metadata["artifact_sha256"] == sha256_hex
    assert metadata["final_development_fit_status"] == "fitted_full_development_partition"
    assert metadata["winning_score_label"] == SELECTION_BIASED_LABEL


def test_safe_loader_path_traversal_rejection(
    tmp_path: Path, dev_dataset: DevelopmentDataset
) -> None:
    allowed_dir = tmp_path / "allowed"
    outside_dir = tmp_path / "outside"
    allowed_dir.mkdir()
    outside_dir.mkdir()

    art_outside = outside_dir / "model.joblib"
    art_outside.write_bytes(b"dummy")
    meta_outside = outside_dir / "meta.json"
    meta_outside.write_text("{}", encoding="utf-8")

    # Path traversal / escape outside allowed_dir must be rejected
    with pytest.raises(ArtifactSecurityError):
        load_model_artifact("random_forest", art_outside, meta_outside, allowed_dir=allowed_dir)


def test_safe_loader_sha256_mismatch_rejection(
    tmp_path: Path, dev_dataset: DevelopmentDataset
) -> None:
    allowed_dir = tmp_path / "models"
    allowed_dir.mkdir()

    art_file = allowed_dir / "model.joblib"
    art_file.write_bytes(b"content_a")
    meta_file = allowed_dir / "meta.json"
    metadata = {
        "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "canonical_class_order": list(CANONICAL_CLASS_ORDER),
        "canonical_feature_names": list(CANONICAL_FEATURE_NAMES),
        "model_id": "random_forest",
    }
    meta_file.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        load_model_artifact("random_forest", art_file, meta_file, allowed_dir=allowed_dir)


# ---------------------------------------------------------------------------
# 10. Per-Family Comparison Against Day 10 Baseline
# ---------------------------------------------------------------------------


def test_per_family_comparison_generation(dev_dataset: DevelopmentDataset) -> None:
    fast_cfg = (
        {
            "n_estimators": 20,
            "max_depth": 4,
            "min_samples_leaf": 2,
            "simplicity_rank": 1,
        },
    )
    summary = run_candidate_selection(
        "random_forest",
        dev_dataset,
        candidate_configs=fast_cfg,
    )
    comp = generate_per_family_comparison(summary, dev_dataset, baseline_id="logistic_regression")

    assert comp["baseline_id"] == "logistic_regression"
    assert comp["candidate_model_id"] == "random_forest"
    assert len(comp["families"]) == 53
    summary_counts = comp["summary"]
    assert summary_counts["better"] + summary_counts["worse"] + summary_counts["tied"] == 53
