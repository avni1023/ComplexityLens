"""
test_cli_evaluate.py — Unit and integration tests for the ComplexityLens evaluation CLI.

Verifies:
1. Command is registered and discoverable through CLI help (`evaluate`, `evaluate cv`).
2. Command invokes existing cross-validation runner (`run_cross_validation`).
3. Default command evaluates all three fixed baseline models.
4. Successful run displays experiment ID, development-only metric labels, and model summaries.
5. Explicit output destination produces valid deterministic JSON.
6. No output file is created when no destination is requested.
7. Invalid output parent paths produce clear error and nonzero exit code.
8. Invalid manifest paths produce clear error and nonzero exit code.
9. Manifest integrity failures stop evaluation with clean error message.
10. Command does not invoke frozen test-partition evaluation.
11. Command does not create model artifacts or persistent caches.
12. Existing runner's results are surfaced correctly without recalculating metrics in CLI.
13. Repeated runs with identical controlled inputs produce consistent summaries and JSON.
14. JSON stdout mode emits valid machine-readable JSON.
15. Model warnings are cleanly displayed in the summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from complexity_lens.cli import main
from complexity_lens.evaluation import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
    CrossValidationExperimentResult,
    EvaluationMetadata,
    EvaluationMetrics,
    FoldEvaluationResult,
    ManifestVerificationError,
    ModelEvaluationResult,
    OutOfFoldPrediction,
    PerClassMetrics,
    TestPartitionAccessError,
    write_cv_results,
)

# ---------------------------------------------------------------------------
# Synthetic Test Helpers
# ---------------------------------------------------------------------------


def _make_dummy_metrics(accuracy: float = 0.50, macro_f1: float = 0.45) -> EvaluationMetrics:
    per_class = {
        cls_name: PerClassMetrics(
            class_name=cls_name,
            precision=accuracy,
            recall=accuracy,
            f1=macro_f1,
            support=10,
        )
        for cls_name in CANONICAL_CLASS_ORDER
    }
    matrix = [[0] * 8 for _ in range(8)]
    for i in range(8):
        matrix[i][i] = 10
    return EvaluationMetrics(
        accuracy=accuracy,
        macro_precision=accuracy,
        macro_recall=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=matrix,
    )


def _make_dummy_cv_result(
    warnings_map: dict[str, list[str]] | None = None,
) -> CrossValidationExperimentResult:
    """Build a deterministic, valid CrossValidationExperimentResult for CLI tests."""
    warnings_map = warnings_map or {}
    model_ids = ["majority_class", "logistic_regression", "decision_tree"]
    models: dict[str, ModelEvaluationResult] = {}

    for mid in model_ids:
        acc = 0.25 if mid == "majority_class" else (0.55 if mid == "logistic_regression" else 0.65)
        f1 = 0.05 if mid == "majority_class" else (0.45 if mid == "logistic_regression" else 0.55)
        metrics = _make_dummy_metrics(accuracy=acc, macro_f1=f1)
        fold_results = [
            FoldEvaluationResult(
                fold=f,
                model_id=mid,
                train_sample_count=100,
                val_sample_count=50,
                train_family_count=35,
                val_family_count=18,
                train_class_distribution={c: 12 for c in CANONICAL_CLASS_ORDER},
                val_class_distribution={c: 6 for c in CANONICAL_CLASS_ORDER},
                missing_classes=[],
                metrics=metrics,
                warnings=warnings_map.get(mid, []),
            )
            for f in range(3)
        ]
        oof_preds = [
            OutOfFoldPrediction(
                program_id=f"p_{i:03d}",
                family_id=f"f_{i // 3:02d}",
                fold=i % 3,
                true_label="O1",
                predicted_label="O1",
                probabilities=[1.0] + [0.0] * 7,
            )
            for i in range(159)
        ]
        meta = EvaluationMetadata(
            schema_version="1.0",
            model_id=mid,
            fixed_hyperparameters={"test": True},
            canonical_feature_schema=list(CANONICAL_FEATURE_NAMES),
            canonical_class_ordering=list(CANONICAL_CLASS_ORDER),
            manifest_sha256="0b382938897af0f43de4ddd29b0fdf0306d2f33fbe73ad1c7876d7410d973cd4",
            fold_assignment_hash="f7c9a424393682abae83a2d8f5e879cca5aa2838ed94aae63c15b8573b9a3c46",
            git_commit="a1e2ae0bb410ee1ae537155bbf71f98f5b2255b4",
            git_dirty=False,
            library_versions={"numpy": "1.0", "python": "3.13"},
            sample_count=159,
            family_count=53,
            training_class_distribution={c: 20 for c in CANONICAL_CLASS_ORDER},
            missing_class_details={},
            limitations=["Test limitations"],
        )
        models[mid] = ModelEvaluationResult(
            model_id=mid,
            fixed_hyperparameters={"test": True},
            metadata=meta,
            per_fold=fold_results,
            combined_oof_metrics=metrics,
            oof_predictions=oof_preds,
            warnings=warnings_map.get(mid, []),
        )

    summary = {
        "family_count": 53,
        "folds": [0, 1, 2],
        "manifest_path": "data/splits/split_manifest_v1.0.frozen.json",
        "manifest_sha256": "0b382938897af0f43de4ddd29b0fdf0306d2f33fbe73ad1c7876d7410d973cd4",
        "partition": "development",
        "per_fold_counts": {
            "fold_0": {"val_families": 18, "val_programs": 54},
            "fold_1": {"val_families": 18, "val_programs": 54},
            "fold_2": {"val_families": 17, "val_programs": 51},
        },
        "program_count": 159,
    }

    return CrossValidationExperimentResult(
        schema_version="1.0",
        experiment_id="cv_baselines_development",
        manifest_sha256="0b382938897af0f43de4ddd29b0fdf0306d2f33fbe73ad1c7876d7410d973cd4",
        fold_assignment_hash="f7c9a424393682abae83a2d8f5e879cca5aa2838ed94aae63c15b8573b9a3c46",
        git_commit="a1e2ae0bb410ee1ae537155bbf71f98f5b2255b4",
        git_dirty=False,
        library_versions={"numpy": "1.0", "python": "3.13"},
        dataset_summary=summary,
        models=models,
        limitations=["Evaluated strictly on development partition using frozen CV folds."],
    )


# ---------------------------------------------------------------------------
# CLI Command Registration & Help Tests
# ---------------------------------------------------------------------------


def test_evaluate_command_registered_in_top_level_help(capsys):
    """Verify 'evaluate' is listed in main CLI --help output."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr().out
    assert "evaluate" in captured


def test_evaluate_subcommand_help_discoverable(capsys):
    """Verify 'evaluate --help' and 'evaluate cv --help' display clean usage."""
    with pytest.raises(SystemExit) as exc_info:
        main(["evaluate", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "cv" in out

    with pytest.raises(SystemExit) as exc_info:
        main(["evaluate", "cv", "--help"])
    assert exc_info.value.code == 0
    cv_out = capsys.readouterr().out
    assert "--manifest" in cv_out
    assert "--dataset-dir" in cv_out
    assert "--out" in cv_out
    assert "--json" in cv_out


def test_evaluate_without_subcommand_prints_help(capsys):
    """Verify invoking 'evaluate' without a subcommand prints help and returns 1."""
    ret = main(["evaluate"])
    assert ret == 1
    out = capsys.readouterr().out
    assert "usage:" in out or "help" in out


# ---------------------------------------------------------------------------
# Runner Invocation and Baselines Tests
# ---------------------------------------------------------------------------


def test_evaluate_cv_invokes_cross_validation_runner(capsys):
    """Verify 'evaluate cv' delegates directly to run_cross_validation."""
    dummy_result = _make_dummy_cv_result()

    with patch(
        "complexity_lens.cli.run_cross_validation", return_value=dummy_result
    ) as mock_runner:
        ret = main(["evaluate", "cv"])
        assert ret == 0
        mock_runner.assert_called_once()
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["manifest_path"] == Path("data/splits/split_manifest_v1.0.frozen.json")
        assert kwargs["dataset_root"] == "datasets"
        assert kwargs["destination"] is None
        assert kwargs["ensure_parents"] is False


def test_evaluate_cv_evaluates_all_three_baselines(capsys):
    """Verify default CV command displays all three fixed baseline models."""
    dummy_result = _make_dummy_cv_result()

    with patch("complexity_lens.cli.run_cross_validation", return_value=dummy_result):
        ret = main(["evaluate", "cv"])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "majority_class:" in captured
        assert "logistic_regression:" in captured
        assert "decision_tree:" in captured


def test_evaluate_cv_output_summary_format(capsys):
    """Verify output contains experiment ID, development-only labels, and isolation footer."""
    dummy_result = _make_dummy_cv_result()

    with patch("complexity_lens.cli.run_cross_validation", return_value=dummy_result):
        ret = main(["evaluate", "cv"])
        assert ret == 0
        captured = capsys.readouterr().out

        assert "ComplexityLens Evaluation: Development Cross-Validation Summary" in captured
        assert f"Experiment ID        : {dummy_result.experiment_id}" in captured
        assert f"Manifest SHA-256     : {dummy_result.manifest_sha256}" in captured
        assert "Partition            : Development (Frozen Test Partition Strictly Isolated)" in (
            captured
        )
        assert "Programs Evaluated   : 159 across 53 families" in captured
        assert "Folds                : 3 (F0: 54, F1: 54, F2: 51)" in captured
        assert "[Development Out-of-Fold (OOF) Metrics]" in captured
        assert (
            "* Note: Metrics reflect development cross-validation only, "
            "NOT final test generalization."
        ) in captured


# ---------------------------------------------------------------------------
# Output File & Path Handling Tests
# ---------------------------------------------------------------------------


def test_evaluate_cv_explicit_out_writes_deterministic_json(capsys, tmp_path):
    """Verify passing --out writes valid deterministic JSON to the specified destination."""
    out_file = tmp_path / "results.json"
    dummy_result = _make_dummy_cv_result()

    def mock_run_cv(**kwargs):
        if kwargs.get("destination"):
            write_cv_results(dummy_result, kwargs["destination"])
        return dummy_result

    with patch("complexity_lens.cli.run_cross_validation", side_effect=mock_run_cv):
        ret = main(["evaluate", "cv", "--out", str(out_file)])
        assert ret == 0
        assert out_file.exists()

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["experiment_id"] == "cv_baselines_development"
        assert "majority_class" in data["models"]
        assert "logistic_regression" in data["models"]
        assert "decision_tree" in data["models"]

        captured = capsys.readouterr().out
        assert f"Output File          : {out_file}" in captured


def test_evaluate_cv_no_out_does_not_create_file(capsys, tmp_path):
    """Verify running without --out does not write any file to the current working directory."""
    dummy_result = _make_dummy_cv_result()
    files_before = set(Path.cwd().glob("*.json"))

    with patch("complexity_lens.cli.run_cross_validation", return_value=dummy_result):
        ret = main(["evaluate", "cv"])
        assert ret == 0

    files_after = set(Path.cwd().glob("*.json"))
    assert files_after == files_before


def test_evaluate_cv_invalid_out_parent_dir_fails(capsys):
    """Verify passing --out with a nonexistent parent directory prints an error and returns 1."""
    nonexistent_dest = "nonexistent_dir/nested/cv_results.json"
    ret = main(["evaluate", "cv", "--out", nonexistent_dest])
    assert ret == 1
    err = capsys.readouterr().err
    assert "Destination parent directory does not exist" in err


def test_evaluate_cv_invalid_manifest_path_fails(capsys):
    """Verify passing a nonexistent manifest path reports a clear error and returns 1."""
    ret = main(["evaluate", "cv", "--manifest", "nonexistent_manifest.json"])
    assert ret == 1
    err = capsys.readouterr().err
    assert "Manifest file not found" in err


def test_evaluate_cv_manifest_integrity_failure_stops_evaluation(capsys):
    """Verify manifest integrity failures stop evaluation with clean error and exit code 1."""
    with patch(
        "complexity_lens.cli.run_cross_validation",
        side_effect=ManifestVerificationError("Checksum mismatch in frozen manifest"),
    ):
        ret = main(["evaluate", "cv"])
        assert ret == 1
        err = capsys.readouterr().err
        assert "Checksum mismatch in frozen manifest" in err


# ---------------------------------------------------------------------------
# Isolation, Warnings, and Metrics Consistency Tests
# ---------------------------------------------------------------------------


def test_evaluate_cv_does_not_access_test_partition(capsys):
    """Verify that TestPartitionAccessError is handled cleanly and stops evaluation."""
    with patch(
        "complexity_lens.cli.run_cross_validation",
        side_effect=TestPartitionAccessError("Test partition program access attempted!"),
    ):
        ret = main(["evaluate", "cv"])
        assert ret == 1
        err = capsys.readouterr().err
        assert "Test partition program access attempted!" in err


def test_evaluate_cv_does_not_create_model_artifacts_or_caches():
    """Verify CLI evaluation creates zero .pkl or .joblib model artifacts."""
    repo_root = Path.cwd()
    artifacts_before = set(repo_root.glob("**/*.pkl")) | set(repo_root.glob("**/*.joblib"))

    dummy_result = _make_dummy_cv_result()
    with patch("complexity_lens.cli.run_cross_validation", return_value=dummy_result):
        ret = main(["evaluate", "cv"])
        assert ret == 0

    artifacts_after = set(repo_root.glob("**/*.pkl")) | set(repo_root.glob("**/*.joblib"))
    assert artifacts_after == artifacts_before


def test_evaluate_cv_surfaces_existing_metrics_without_recalculating(capsys):
    """Verify metric values computed by runner are displayed directly without modification."""
    import dataclasses

    dummy_result = _make_dummy_cv_result()
    # Force distinct test values on frozen dataclass
    new_model = dataclasses.replace(
        dummy_result.models["majority_class"],
        combined_oof_metrics=_make_dummy_metrics(0.1234, 0.4321),
    )
    new_models = dict(dummy_result.models)
    new_models["majority_class"] = new_model
    dummy_result = dataclasses.replace(dummy_result, models=new_models)

    with patch("complexity_lens.cli.run_cross_validation", return_value=dummy_result):
        ret = main(["evaluate", "cv"])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "Accuracy         : 0.1234" in captured
        assert "Macro F1         : 0.4321" in captured


def test_evaluate_cv_json_stdout_mode(capsys):
    """Verify '--json' emits clean, deterministic JSON to stdout."""
    dummy_result = _make_dummy_cv_result()

    with patch("complexity_lens.cli.run_cross_validation", return_value=dummy_result):
        ret = main(["evaluate", "cv", "--json"])
        assert ret == 0
        captured = capsys.readouterr().out
        payload = json.loads(captured)
        assert payload["schema_version"] == "1.0"
        assert payload["experiment_id"] == "cv_baselines_development"
        assert "majority_class" in payload["models"]


def test_evaluate_cv_surfaces_warnings_properly(capsys):
    """Verify convergence and runtime warnings are surfaced in the summary box."""
    dummy_result = _make_dummy_cv_result(
        warnings_map={"logistic_regression": ["ConvergenceWarning: lbfgs reached limit"]}
    )

    with patch("complexity_lens.cli.run_cross_validation", return_value=dummy_result):
        ret = main(["evaluate", "cv"])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "Warnings (1)         :" in captured
        assert "[logistic_regression] ConvergenceWarning: lbfgs reached limit" in captured


def test_evaluate_cv_repeated_runs_deterministic(capsys, tmp_path):
    """Verify repeated runs with identical inputs produce identical summary output."""
    dummy_result = _make_dummy_cv_result()

    with patch("complexity_lens.cli.run_cross_validation", return_value=dummy_result):
        main(["evaluate", "cv"])
        out1 = capsys.readouterr().out

        main(["evaluate", "cv"])
        out2 = capsys.readouterr().out

        assert out1 == out2
