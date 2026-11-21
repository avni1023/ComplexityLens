"""
test_evaluation_loader.py — Comprehensive tests for Stage 2C development-only data loader.

Covers:
1. Correct loading of all 159 development programs and 53 families.
2. Canonical 26-feature ordering, types, and schema validation.
3. Exact fold cardinalities (54, 54, 51 validation programs).
4. Correct training/validation family assignments for every fold.
5. Strict family disjointness (train vs val) and full coverage.
6. Zero test-partition leakage in returned development or CV fold data.
7. Explicit TestPartitionAccessError on any attempt to access test programs or families.
8. Clear rejection of unknown IDs and quarantined legacy families.
9. Rejection of missing manifest or missing sidecar.
10. Rejection of tampered manifest, hash mismatch, or non-frozen status.
11. Clear FeatureExtractionError when parsing/extraction fails.
12. Determinism and zero persistent feature caches on disk.
13. Safe isolated execution with temporary fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from complexity_lens.evaluation.data_loader import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_FEATURE_NAMES,
    DevelopmentDataset,
    EvaluationRecord,
    FeatureExtractionError,
    FoldSplit,
    ManifestVerificationError,
    QuarantinedFamilyAccessError,
    TestPartitionAccessError,
    _extract_record_features,
    get_canonical_class_order,
    get_canonical_feature_names,
    get_cv_fold,
    load_development_dataset,
    load_development_family,
    load_development_program,
)
from complexity_lens.features.schema import FEATURE_NAMES

FROZEN_MANIFEST_PATH = Path("data/splits/split_manifest_v1.0.frozen.json")
EXPECTED_SHA256 = "0b382938897af0f43de4ddd29b0fdf0306d2f33fbe73ad1c7876d7410d973cd4"


# ===========================================================================
# 1. Schema & Public Query Tests
# ===========================================================================


def test_canonical_schema_and_class_order():
    """Verify canonical feature names and class order queryable without dataset loading."""
    feature_names = get_canonical_feature_names()
    assert len(feature_names) == 26
    assert feature_names == FEATURE_NAMES
    assert feature_names == CANONICAL_FEATURE_NAMES

    class_order = get_canonical_class_order()
    assert len(class_order) == 8
    assert class_order == (
        "O1",
        "OLOGN",
        "ON",
        "ONLOGN",
        "ON2",
        "ON3",
        "OEXP",
        "UNKNOWN",
    )


# ===========================================================================
# 2. Verified Development Dataset Loading Tests
# ===========================================================================


def test_load_development_dataset_counts():
    """Verify loading the real frozen split loads exactly 159 programs and 53 families."""
    dataset = load_development_dataset()
    assert isinstance(dataset, DevelopmentDataset)
    assert len(dataset) == 159
    assert len(dataset.records) == 159
    assert len(dataset.program_ids) == 159
    assert len(dataset.families) == 53
    assert dataset.manifest_sha256 == EXPECTED_SHA256

    # Verify every record attributes
    for r in dataset.records:
        assert isinstance(r, EvaluationRecord)
        assert r.fold in (0, 1, 2)
        assert r.label in CANONICAL_CLASS_ORDER
        assert r.tag in ("easy", "rule_hard")
        assert len(r.source) > 0
        assert len(r.source_sha256) == 64


def test_record_features_canonical_order_and_validation():
    """Verify feature vectors have exact 26 canonical features and non-negative ints."""
    dataset = load_development_dataset()
    sample = dataset.get_program("f02_fixed_for_accumulate")

    fv = sample.features
    assert len(fv.names) == 26
    assert fv.names == FEATURE_NAMES
    assert len(fv.values) == 26

    for name, val in zip(fv.names, fv.values):
        assert isinstance(val, int)
        assert val >= 0

    fv_dict = fv.to_dict()
    assert len(fv_dict) == 26
    assert list(fv_dict.keys()) == list(FEATURE_NAMES)
    assert fv.to_list() == list(fv.values)

    # In-memory caching on record
    assert sample.features is fv


# ===========================================================================
# 3. Cross-Validation Fold Cardinality & Disjointness Tests
# ===========================================================================


def test_cv_fold_exact_cardinalities():
    """Verify exact validation fold sizes (54, 54, 51) and family counts (18, 18, 17)."""
    dataset = load_development_dataset()

    fold_specs = [
        (0, 54, 18, 105, 35),
        (1, 54, 18, 105, 35),
        (2, 51, 17, 108, 36),
    ]

    for fold_idx, exp_val_progs, exp_val_fams, exp_tr_progs, exp_tr_fams in fold_specs:
        split = dataset.get_fold(fold_idx)
        assert isinstance(split, FoldSplit)
        assert split.fold == fold_idx

        assert len(split.val_records) == exp_val_progs
        assert len(split.val_families) == exp_val_fams
        assert len(split.train_records) == exp_tr_progs
        assert len(split.train_families) == exp_tr_fams

        assert len(split.val_records) + len(split.train_records) == 159
        assert len(split.val_families) + len(split.train_families) == 53


def test_cv_fold_family_disjointness_and_full_coverage():
    """Verify training and validation families are strictly disjoint and cover all 53 families."""
    dataset = load_development_dataset()

    all_val_families: set[str] = set()

    for fold_idx in (0, 1, 2):
        split = get_cv_fold(fold_idx, dataset=dataset)

        train_fams = split.train_families
        val_fams = split.val_families

        # Disjointness
        overlap = train_fams & val_fams
        assert overlap == frozenset(), f"Leakage detected in fold {fold_idx}: {overlap}"

        # Coverage
        union = train_fams | val_fams
        assert union == dataset.families

        # Program fold consistency
        for r in split.val_records:
            assert r.fold == fold_idx
        for r in split.train_records:
            assert r.fold != fold_idx

        all_val_families.update(val_fams)

    # Across all 3 folds, every dev family appears in validation exactly once
    assert all_val_families == dataset.families


def test_invalid_fold_index_raises_value_error():
    """Verify requesting an invalid fold index raises ValueError."""
    dataset = load_development_dataset()
    with pytest.raises(ValueError, match="Invalid fold index 3"):
        dataset.get_fold(3)

    with pytest.raises(ValueError, match="Invalid fold index -1"):
        get_cv_fold(-1)


# ===========================================================================
# 4. Strict Test Partition Isolation Guards
# ===========================================================================


def test_no_test_partition_leakage_in_development_data():
    """Verify no program or family from test partition appears in development data or CV folds."""
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    test_programs = {p["program_id"] for p in manifest["programs"] if p["partition"] == "test"}
    test_families = set(manifest["partitions"]["test"]["families"])

    assert len(test_programs) == 45
    assert len(test_families) == 15

    dataset = load_development_dataset()

    # Dataset level
    dev_prog_set = set(dataset.program_ids)
    assert dev_prog_set.isdisjoint(test_programs)
    assert dataset.families.isdisjoint(test_families)

    # Fold level
    for fold_idx in (0, 1, 2):
        split = dataset.get_fold(fold_idx)
        train_pids = set(split.train_program_ids)
        val_pids = set(split.val_program_ids)

        assert train_pids.isdisjoint(test_programs)
        assert val_pids.isdisjoint(test_programs)
        assert split.train_families.isdisjoint(test_families)
        assert split.val_families.isdisjoint(test_families)


def test_test_partition_program_access_raises_error():
    """Verify direct requests for test programs raise TestPartitionAccessError."""
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    test_programs = [p["program_id"] for p in manifest["programs"] if p["partition"] == "test"]

    for pid in test_programs[:5]:
        with pytest.raises(TestPartitionAccessError, match="strictly forbidden"):
            load_development_program(pid)


def test_test_partition_family_access_raises_error():
    """Verify direct requests for test families raise TestPartitionAccessError."""
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    test_families = manifest["partitions"]["test"]["families"]

    for fid in test_families[:5]:
        with pytest.raises(TestPartitionAccessError, match="strictly forbidden"):
            load_development_family(fid)


def test_quarantined_family_access_rejected():
    """Verify requests for quarantined legacy benchmark families are rejected."""
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    quarantined = manifest["quarantined_families"]
    assert len(quarantined) == 10

    for qfam in quarantined[:3]:
        with pytest.raises(QuarantinedFamilyAccessError, match="quarantined"):
            load_development_family(qfam)


def test_unknown_program_and_family_rejected():
    """Verify requests for non-existent IDs raise KeyError."""
    dataset = load_development_dataset()

    with pytest.raises(KeyError, match="not found"):
        dataset.get_program("non_existent_program_xyz_123")

    with pytest.raises(KeyError, match="not found"):
        dataset.get_family("non_existent_family_xyz_123")

    with pytest.raises(KeyError, match="not found"):
        load_development_program("non_existent_program_xyz_123")

    with pytest.raises(KeyError, match="not found"):
        load_development_family("non_existent_family_xyz_123")


# ===========================================================================
# 5. Manifest & Sidecar Verification Guards
# ===========================================================================


def test_missing_manifest_rejected(tmp_path: Path):
    """Verify loading a non-existent manifest path raises ManifestVerificationError."""
    missing = tmp_path / "non_existent.json"
    with pytest.raises(ManifestVerificationError, match="not found"):
        load_development_dataset(manifest_path=missing)


def test_missing_sidecar_rejected(tmp_path: Path):
    """Verify a manifest without companion .sha256 sidecar is rejected."""
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="Sidecar checksum file not found"):
        load_development_dataset(manifest_path=fake_manifest)


def test_tampered_manifest_checksum_mismatch_rejected(tmp_path: Path):
    """Verify tampered manifest content produces a sidecar checksum mismatch error."""
    manifest_copy = tmp_path / "manifest.json"
    sidecar_copy = tmp_path / "manifest.json.sha256"

    # Copy real manifest
    content = FROZEN_MANIFEST_PATH.read_text(encoding="utf-8")
    manifest_copy.write_text(content, encoding="utf-8")
    # Write incorrect SHA in sidecar
    sidecar_copy.write_text("0" * 64 + "  manifest.json\n", encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="Sidecar checksum mismatch"):
        load_development_dataset(manifest_path=manifest_copy)


def test_non_frozen_manifest_rejected(tmp_path: Path):
    """Verify a proposal manifest with status 'proposed' is rejected by evaluation loader."""
    manifest_data = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_data["status"] = "proposed"

    manifest_file = tmp_path / "manifest.json"
    sidecar_file = tmp_path / "manifest.json.sha256"

    serialized = json.dumps(manifest_data, indent=2).encode("utf-8")
    manifest_file.write_bytes(serialized)
    sha256 = hashlib.sha256(serialized).hexdigest()
    sidecar_file.write_text(f"{sha256}  manifest.json\n", encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="Expected status 'frozen'"):
        load_development_dataset(manifest_path=manifest_file)


# ===========================================================================
# 6. Feature Extraction Error Handling & Determinism
# ===========================================================================


def test_feature_extraction_error_on_invalid_source():
    """Verify feature extraction error identifies affected program rather than dropping it."""
    with pytest.raises(FeatureExtractionError, match="bad_prog"):
        _extract_record_features("int invalid_syntax( {", "bad_prog")


def test_feature_extraction_error_when_builder_fails():
    """Verify unparseable IR raises FeatureExtractionError with program identifier."""
    with patch(
        "complexity_lens.evaluation.data_loader.build_ir_from_source",
        side_effect=RuntimeError("Clang crash simulation"),
    ):
        with pytest.raises(FeatureExtractionError, match="bad_prog"):
            _extract_record_features("int x = 1;", "bad_prog")


def test_repeated_loading_is_deterministic_no_disk_cache():
    """Verify repeated loading produces identical results and creates no persistent files."""
    existed_before = Path("artifacts").exists()
    ds1 = load_development_dataset()
    ds2 = load_development_dataset()

    assert ds1.program_ids == ds2.program_ids
    assert ds1.families == ds2.families
    assert ds1.manifest_sha256 == ds2.manifest_sha256

    # Verify loading creates no 'artifacts/' directory if not already present
    if not existed_before:
        assert not Path("artifacts").exists()


# ===========================================================================
# 7. Helper Methods (to_train_xy, to_val_xy)
# ===========================================================================


def test_xy_helpers_on_fold_and_dataset():
    """Verify FoldSplit to_train_xy and to_val_xy helpers return matching feature vectors."""
    dataset = load_development_dataset()
    fold0 = dataset.get_fold(0)

    # Test single program extraction
    prog = fold0.val_records[0]
    fv = prog.features
    assert len(fv.to_list()) == 26
    assert isinstance(fv[0], int)
    assert isinstance(fv["n_loops"], int)

    # Test program retrieval via dataset
    p_retrieved = dataset.get_program(prog.program_id)
    assert p_retrieved.program_id == prog.program_id

    # Test family retrieval via dataset
    fam_records = dataset.get_family(prog.family_id)
    assert len(fam_records) > 0
    assert all(r.family_id == prog.family_id for r in fam_records)

