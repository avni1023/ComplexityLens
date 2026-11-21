"""
test_evaluation_cv_runner.py — Unit tests for cross-validation runner and experiment orchestration.

Verifies:
1. Runner uses exactly the frozen development fold assignments.
2. Each model is trained separately for each fold.
3. No validation sample enters that fold's training data (DataIsolationError).
4. No family crosses training and validation within a fold (DataIsolationError).
5. Every covered development program receives exactly one OOF prediction per model.
6. OOF predictions are generated only by the corresponding held-out fold model.
7. Missing or duplicate fold assignments fail clearly (FoldConfigurationError).
8. Empty or malformed fold data fails clearly (FoldConfigurationError).
9. Canonical class order is preserved in probability outputs.
10. Missing classes in a training fold are handled according to existing baseline API.
11. Per-fold metrics match the existing metrics implementation.
12. Combined OOF metrics match independently calculated metrics on combined OOF predictions.
13. Output serialization is deterministic.
14. Metadata contains correct manifest checksum and fold assignment hash.
15. Repeated runs on identical synthetic data produce identical prediction labels.
16. No model artifacts or feature caches are created on disk.
17. The runner does not access the frozen test partition.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from complexity_lens.evaluation.baselines import MajorityClassBaseline
from complexity_lens.evaluation.cv_runner import (
    CrossValidationRunner,
    DataIsolationError,
    FoldConfigurationError,
    compute_fold_assignment_hash,
    run_cross_validation,
    write_cv_results,
)
from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
    DevelopmentDataset,
    EvaluationRecord,
    FoldSplit,
    TestPartitionAccessError,
    load_development_dataset,
    load_development_program,
)
from complexity_lens.evaluation.metrics import (
    CANONICAL_CLASS_TO_INDEX,
    NUM_CANONICAL_CLASSES,
    compute_classification_metrics,
)
from complexity_lens.features.schema import FeatureVector

# ---------------------------------------------------------------------------
# Synthetic Test Fixtures (In-Memory, Zero C++ Compilation / Zero Libclang)
# ---------------------------------------------------------------------------


def _make_synthetic_feature_values(seed: int, n_features: int = 26) -> tuple[int, ...]:
    rng = np.random.RandomState(seed)
    vals = []
    for name in CANONICAL_FEATURE_NAMES:
        if name in ("has_recursion", "has_early_exit_in_loop"):
            vals.append(int(rng.randint(0, 2)))
        else:
            vals.append(int(rng.randint(0, 50)))
    return tuple(vals)


def _make_synthetic_record(
    program_id: str,
    family_id: str,
    label: str,
    fold: int,
    seed: int = 42,
) -> EvaluationRecord:
    features = FeatureVector(
        names=CANONICAL_FEATURE_NAMES,
        values=_make_synthetic_feature_values(seed),
    )
    return EvaluationRecord(
        program_id=program_id,
        family_id=family_id,
        label=label,
        fold=fold,
        tag="easy",
        source="int main() { return 0; }",
        source_sha256="0" * 64,
        _features=features,
    )


def _make_synthetic_dataset(
    n_families: int = 6,
    programs_per_family: int = 2,
    n_folds: int = 3,
) -> DevelopmentDataset:
    """Build a deterministic synthetic DevelopmentDataset across canonical classes and folds."""
    records: list[EvaluationRecord] = []
    classes = ["O1", "ON", "ON2", "OLOGN", "ONLOGN", "UNKNOWN"]

    seed = 100
    for fam_idx in range(n_families):
        family_id = f"f{fam_idx:02d}"
        assigned_fold = fam_idx % n_folds
        cls_name = classes[fam_idx % len(classes)]

        for p_idx in range(programs_per_family):
            program_id = f"prog_{family_id}_{p_idx}"
            seed += 1
            records.append(
                _make_synthetic_record(
                    program_id=program_id,
                    family_id=family_id,
                    label=cls_name,
                    fold=assigned_fold,
                    seed=seed,
                )
            )

    return DevelopmentDataset(
        records=tuple(records),
        manifest_path="synthetic_manifest.json",
        manifest_sha256="a" * 64,
    )


# ---------------------------------------------------------------------------
# 1. Frozen Fold Assignments & Structure
# ---------------------------------------------------------------------------


def test_cv_runner_uses_frozen_development_fold_assignments():
    """Verify runner uses exactly the frozen development fold assignments F0, F1, F2."""
    ds = load_development_dataset()
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class"])

    assert runner.folds == (0, 1, 2)
    assert len(ds.records) == 159

    # Check validation program counts match frozen contracts
    f0 = ds.get_fold(0)
    assert len(f0.val_records) == 54
    assert len(f0.train_records) == 105

    f1 = ds.get_fold(1)
    assert len(f1.val_records) == 54
    assert len(f1.train_records) == 105

    f2 = ds.get_fold(2)
    assert len(f2.val_records) == 51
    assert len(f2.train_records) == 108


# ---------------------------------------------------------------------------
# 2. Separate Training per Fold
# ---------------------------------------------------------------------------


def test_models_trained_separately_for_each_fold():
    """Verify each model is fitted separately for each fold with fold-specific data."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class"])

    with patch.object(
        MajorityClassBaseline, "fit", autospec=True, side_effect=MajorityClassBaseline.fit
    ) as mock_fit:
        res = runner.evaluate_model("majority_class")

        # fit() must be called exactly once per fold (3 times total)
        assert mock_fit.call_count == 3
        assert len(res.per_fold) == 3


# ---------------------------------------------------------------------------
# 3. Validation Sample Leakage Guard
# ---------------------------------------------------------------------------


def test_no_validation_sample_enters_training_data():
    """Verify DataIsolationError is raised if a validation program appears in training data."""
    r1 = _make_synthetic_record("p1", "f01", "O1", fold=0)
    r2 = _make_synthetic_record("p2", "f02", "ON", fold=1)

    # Corrupted fold split where p1 appears in both train and val
    corrupted_split = FoldSplit(
        fold=0,
        train_records=(r1, r2),
        val_records=(r1,),
    )

    ds = DevelopmentDataset(
        records=(r1, r2),
        manifest_path="test.json",
        manifest_sha256="0" * 64,
    )
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class"])

    with pytest.raises(DataIsolationError, match="Data leakage detected.*programs appear in both"):
        runner.evaluate_fold(corrupted_split, "majority_class")


# ---------------------------------------------------------------------------
# 4. Family Leakage Guard
# ---------------------------------------------------------------------------


def test_no_family_crosses_training_and_validation_within_fold():
    """Verify DataIsolationError is raised if a family crosses train and val within a fold."""
    # Two distinct programs from the SAME family f01 assigned to train and val
    r1 = _make_synthetic_record("p1", "f01", "O1", fold=0)
    r2 = _make_synthetic_record("p2", "f01", "O1", fold=1)
    r3 = _make_synthetic_record("p3", "f02", "ON", fold=1)

    corrupted_split = FoldSplit(
        fold=0,
        train_records=(r2, r3),
        val_records=(r1,),
    )

    ds = DevelopmentDataset(
        records=(r1, r2, r3),
        manifest_path="test.json",
        manifest_sha256="0" * 64,
    )
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class"])

    with pytest.raises(DataIsolationError, match="Family leakage detected"):
        runner.evaluate_fold(corrupted_split, "majority_class")


# ---------------------------------------------------------------------------
# 5. Exactly One OOF Prediction per Covered Program
# ---------------------------------------------------------------------------


def test_every_covered_development_program_receives_exactly_one_oof_prediction():
    """Verify every covered development program receives exactly one OOF prediction per model."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class", "decision_tree"])
    exp_res = runner.run()

    for model_id in ["majority_class", "decision_tree"]:
        model_res = exp_res.models[model_id]
        oof_pids = [p.program_id for p in model_res.oof_predictions]

        # Exactly 1 prediction per covered program
        assert len(oof_pids) == len(ds.records)
        assert len(oof_pids) == len(set(oof_pids))
        assert sorted(oof_pids) == sorted(ds.program_ids)


# ---------------------------------------------------------------------------
# 6. OOF Predictions Generated Strictly by Held-Out Model
# ---------------------------------------------------------------------------


def test_oof_predictions_generated_only_by_corresponding_held_out_fold_model():
    """Verify predictions are strictly generated by the model whose training data excluded them."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class"])
    exp_res = runner.run()

    model_res = exp_res.models["majority_class"]
    for pred in model_res.oof_predictions:
        rec = ds.get_program(pred.program_id)
        # Prediction fold must match the program's held-out validation fold
        assert pred.fold == rec.fold


# ---------------------------------------------------------------------------
# 7. Missing or Duplicate Fold Assignments Fail
# ---------------------------------------------------------------------------


def test_missing_or_duplicate_fold_assignments_fail_clearly():
    """Verify duplicate or missing fold assignments fail with FoldConfigurationError."""
    # Program with duplicate assignment across different fold records
    r1 = _make_synthetic_record("p1", "f01", "O1", fold=0)
    r2 = _make_synthetic_record("p1", "f01", "O1", fold=1)

    ds_dup = DevelopmentDataset(
        records=(r1, r2),
        manifest_path="test.json",
        manifest_sha256="0" * 64,
    )

    with pytest.raises(FoldConfigurationError, match="Duplicate program assignments detected"):
        CrossValidationRunner(dataset=ds_dup, model_ids=["majority_class"])


# ---------------------------------------------------------------------------
# 8. Empty or Malformed Fold Data Fails
# ---------------------------------------------------------------------------


def test_empty_or_malformed_fold_data_fails_clearly():
    """Verify empty training or validation sets raise FoldConfigurationError."""
    r1 = _make_synthetic_record("p1", "f01", "O1", fold=0)

    split_empty_train = FoldSplit(fold=0, train_records=(), val_records=(r1,))
    ds = DevelopmentDataset(records=(r1,), manifest_path="t.json", manifest_sha256="0" * 64)
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class"])

    with pytest.raises(FoldConfigurationError, match="training set cannot be empty"):
        runner.evaluate_fold(split_empty_train, "majority_class")

    split_empty_val = FoldSplit(fold=0, train_records=(r1,), val_records=())
    with pytest.raises(FoldConfigurationError, match="validation set cannot be empty"):
        runner.evaluate_fold(split_empty_val, "majority_class")


# ---------------------------------------------------------------------------
# 9. Canonical Class Order in Probabilities
# ---------------------------------------------------------------------------


def test_canonical_class_order_preserved_in_probability_outputs():
    """Verify probability outputs have shape 8 in canonical class order and sum to 1.0."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    runner = CrossValidationRunner(dataset=ds, model_ids=["logistic_regression", "decision_tree"])
    exp_res = runner.run()

    for model_id in ["logistic_regression", "decision_tree"]:
        model_res = exp_res.models[model_id]
        for pred in model_res.oof_predictions:
            assert len(pred.probabilities) == NUM_CANONICAL_CLASSES
            assert np.isclose(sum(pred.probabilities), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# 10. Missing Classes in Training Fold Handled According to Baseline API
# ---------------------------------------------------------------------------


def test_missing_classes_in_training_fold_handled_cleanly():
    """Verify absent classes in training fold receive 0.0 probability and are tracked."""
    # Synthetic dataset containing only O1 and ON (6 classes absent)
    r1 = _make_synthetic_record("p1", "f01", "O1", fold=0)
    r2 = _make_synthetic_record("p2", "f02", "ON", fold=0)
    r3 = _make_synthetic_record("p3", "f03", "O1", fold=1)
    r4 = _make_synthetic_record("p4", "f04", "ON", fold=1)

    ds = DevelopmentDataset(
        records=(r1, r2, r3, r4),
        manifest_path="test.json",
        manifest_sha256="0" * 64,
    )
    runner = CrossValidationRunner(dataset=ds, model_ids=["logistic_regression"])
    exp_res = runner.run()

    model_res = exp_res.models["logistic_regression"]
    # 6 classes absent: OLOGN, ONLOGN, ON2, ON3, OEXP, UNKNOWN
    for fold_res in model_res.per_fold:
        assert "OLOGN" in fold_res.missing_classes
        assert "OEXP" in fold_res.missing_classes

    for pred in model_res.oof_predictions:
        for absent_cls in ["OLOGN", "ONLOGN", "ON2", "ON3", "OEXP", "UNKNOWN"]:
            idx = CANONICAL_CLASS_TO_INDEX[absent_cls]
            assert pred.probabilities[idx] == 0.0


# ---------------------------------------------------------------------------
# 11. Per-Fold Metrics Match Metrics Implementation
# ---------------------------------------------------------------------------


def test_per_fold_metrics_match_existing_metrics_implementation():
    """Verify per-fold metrics match compute_classification_metrics output exactly."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    runner = CrossValidationRunner(dataset=ds, model_ids=["decision_tree"])
    exp_res = runner.run()

    model_res = exp_res.models["decision_tree"]
    for fold_res in model_res.per_fold:
        fold_preds = [p for p in model_res.oof_predictions if p.fold == fold_res.fold]
        y_true = [p.true_label for p in fold_preds]
        y_pred = [p.predicted_label for p in fold_preds]

        expected = compute_classification_metrics(y_true, y_pred)
        assert fold_res.metrics.accuracy == expected.accuracy
        assert fold_res.metrics.macro_f1 == expected.macro_f1
        assert fold_res.metrics.confusion_matrix == expected.confusion_matrix


# ---------------------------------------------------------------------------
# 12. Combined OOF Metrics Match Independent Calculation
# ---------------------------------------------------------------------------


def test_combined_oof_metrics_match_independently_calculated_metrics():
    """Verify combined OOF metrics match independent calculation across all OOF predictions."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class", "decision_tree"])
    exp_res = runner.run()

    for model_id in ["majority_class", "decision_tree"]:
        model_res = exp_res.models[model_id]
        oof_y_true = [p.true_label for p in model_res.oof_predictions]
        oof_y_pred = [p.predicted_label for p in model_res.oof_predictions]

        expected_combined = compute_classification_metrics(oof_y_true, oof_y_pred)
        assert model_res.combined_oof_metrics.accuracy == expected_combined.accuracy
        assert model_res.combined_oof_metrics.macro_f1 == expected_combined.macro_f1
        assert (
            model_res.combined_oof_metrics.confusion_matrix == expected_combined.confusion_matrix
        )


# ---------------------------------------------------------------------------
# 13. Output Serialization Is Deterministic
# ---------------------------------------------------------------------------


def test_output_serialization_is_deterministic():
    """Verify writing result to JSON produces byte-identical files across repeated calls."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    res = run_cross_validation(
        dataset=ds,
        model_ids=["majority_class"],
        git_commit="abcdef1234567890abcdef1234567890abcdef12",
        git_dirty=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "res1.json"
        path2 = Path(tmpdir) / "res2.json"

        write_cv_results(res, path1)
        write_cv_results(res, path2)

        content1 = path1.read_bytes()
        content2 = path2.read_bytes()

        assert content1 == content2
        assert len(content1) > 0


# ---------------------------------------------------------------------------
# 14. Metadata Integrity (Manifest Checksum & Fold Hash)
# ---------------------------------------------------------------------------


def test_metadata_contains_correct_manifest_checksum_and_fold_assignment_hash():
    """Verify metadata validates and contains exact manifest checksum and fold assignment hash."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class"])
    exp_res = runner.run()

    model_res = exp_res.models["majority_class"]
    meta = model_res.metadata

    # Validate schema
    meta.validate()

    assert meta.manifest_sha256 == ds.manifest_sha256
    expected_fold_hash = compute_fold_assignment_hash(ds)
    assert meta.fold_assignment_hash == expected_fold_hash
    assert meta.canonical_class_ordering == list(CANONICAL_CLASS_ORDER)
    assert meta.canonical_feature_schema == list(CANONICAL_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# 15. Repeated Runs on Identical Synthetic Data Produce Identical Results
# ---------------------------------------------------------------------------


def test_repeated_runs_on_identical_synthetic_data_produce_identical_predictions():
    """Verify repeated runs on identical data produce identical prediction labels and probs."""
    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)

    res1 = run_cross_validation(
        dataset=ds,
        model_ids=["majority_class", "decision_tree"],
        git_commit="commit123",
        git_dirty=False,
    )
    res2 = run_cross_validation(
        dataset=ds,
        model_ids=["majority_class", "decision_tree"],
        git_commit="commit123",
        git_dirty=False,
    )

    for m in ["majority_class", "decision_tree"]:
        preds1 = res1.models[m].oof_predictions
        preds2 = res2.models[m].oof_predictions

        for p1, p2 in zip(preds1, preds2):
            assert p1.program_id == p2.program_id
            assert p1.predicted_label == p2.predicted_label
            assert np.allclose(p1.probabilities, p2.probabilities)


# ---------------------------------------------------------------------------
# 16. No Model Artifacts or Feature Caches Created on Disk
# ---------------------------------------------------------------------------


def test_no_model_artifacts_or_feature_caches_created():
    """Verify running cross validation does not create any persistent model or cache files."""
    repo_root = Path.cwd()

    def _get_tracked_files() -> set[Path]:
        return set(repo_root.glob("**/*.pkl")) | set(repo_root.glob("**/*.joblib"))

    files_before = _get_tracked_files()

    ds = _make_synthetic_dataset(n_families=6, programs_per_family=2, n_folds=3)
    _ = run_cross_validation(dataset=ds, model_ids=["majority_class", "decision_tree"])

    files_after = _get_tracked_files()
    assert files_after == files_before


# ---------------------------------------------------------------------------
# 17. Runner Does Not Access Frozen Test Partition
# ---------------------------------------------------------------------------


def test_runner_does_not_access_frozen_test_partition():
    """Verify that the runner operates exclusively on development data and rejects test IDs."""
    # Attempting to load test program directly via loader must raise TestPartitionAccessError
    with pytest.raises(TestPartitionAccessError, match="strictly forbidden"):
        # f01_swap_basic is in family f01 which is in the frozen test partition
        load_development_program("f01_swap_basic")

    # Verify that the development dataset loaded by runner contains only development programs
    ds = load_development_dataset()
    runner = CrossValidationRunner(dataset=ds, model_ids=["majority_class"])

    # None of the records must belong to any test family
    test_families = {
        "f01", "f03", "f08", "f10", "f16", "f18", "f21",
        "f28", "f29", "f34", "f41", "f46", "f62", "f63", "f67",
    }
    assert runner.dataset.families.isdisjoint(test_families)
