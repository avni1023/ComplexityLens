"""Tests for Day 13 S1–S4 Evaluation Pipeline and Alignment Verification."""

from __future__ import annotations

import pytest

from complexity_lens.evaluation.cv_runner import OutOfFoldPrediction
from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    EvaluationRecord,
    load_development_dataset,
)
from complexity_lens.evaluation.day13_evaluation import (
    align_and_verify_program_ids,
    build_ml_prediction_containers,
)


def _make_dummy_record(program_id: str) -> EvaluationRecord:
    return EvaluationRecord(
        program_id=program_id,
        family_id="f01",
        label="ON",
        fold=0,
        tag="easy",
        source="int main() {}",
        source_sha256="abc123sha",
    )


def _make_dummy_oof(program_id: str) -> OutOfFoldPrediction:
    return OutOfFoldPrediction(
        program_id=program_id,
        family_id="f01",
        fold=0,
        true_label="ON",
        predicted_label="ON",
        probabilities=[0.125] * 8,
    )


class TestDay13EvaluationAlignment:
    """Verifies strict 1-to-1 program_id alignment between dataset and ML OOF."""

    def test_alignment_success(self) -> None:
        records = [_make_dummy_record("prog_01"), _make_dummy_record("prog_02")]
        oof_preds = [_make_dummy_oof("prog_01"), _make_dummy_oof("prog_02")]

        # Must not raise
        align_and_verify_program_ids(records, oof_preds)

    def test_alignment_length_mismatch(self) -> None:
        records = [_make_dummy_record("prog_01")]
        oof_preds = [_make_dummy_oof("prog_01"), _make_dummy_oof("prog_02")]

        with pytest.raises(ValueError, match="Record count mismatch"):
            align_and_verify_program_ids(records, oof_preds)

    def test_alignment_id_mismatch(self) -> None:
        records = [_make_dummy_record("prog_01"), _make_dummy_record("prog_02")]
        oof_preds = [_make_dummy_oof("prog_01"), _make_dummy_oof("prog_99")]

        with pytest.raises(ValueError, match="Program ID mismatch at index 1"):
            align_and_verify_program_ids(records, oof_preds)

    def test_build_ml_prediction_containers(self) -> None:
        oof_preds = [_make_dummy_oof("prog_01")]
        containers = build_ml_prediction_containers(oof_preds)

        assert len(containers) == 1
        c = containers[0]
        assert c.predicted_label == "ON"
        assert c.probability == 0.125
        assert len(c.probabilities) == len(CANONICAL_CLASS_ORDER)
        assert c.model_id == "random_forest_oof"


class TestDay13LiveAlignment:
    """Live verification of alignment on the development partition."""

    def test_s1_s4_program_id_alignment(self) -> None:
        dev_dataset = load_development_dataset(
            manifest_path="data/splits/split_manifest_v1.0.frozen.json",
            dataset_root="datasets",
        )
        assert len(dev_dataset.records) == 159

        # Verify that dev dataset program IDs match expected sorted structure
        dev_ids = [r.program_id for r in dev_dataset.records]
        assert len(dev_ids) == 159
        assert len(set(dev_ids)) == 159
        assert dev_ids == sorted(dev_ids)
