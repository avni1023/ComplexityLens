"""
data_loader.py — Secure, deterministic evaluation data loader with test-isolation guard.

Enforces strict test-partition isolation for Stage 2C evaluation of ComplexityLens:
- Exposes development-only feature data and cross-validation folds.
- Guarantees zero leakage: rejects any access to test programs or test families
  with explicit TestPartitionAccessError.
- Verifies frozen split manifest and SHA-256 sidecar checksum before loading.
- Rejects quarantined legacy families and unknown IDs.
- Validates the 26 canonical features in schema order.
- Extracts features strictly in memory; creates no persistent cache.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from complexity_lens.dataset import compute_source_sha256
from complexity_lens.features.extractor import extract_features
from complexity_lens.features.schema import FEATURE_NAMES, FeatureVector
from complexity_lens.ir.builder import build_ir_from_source
from complexity_lens.split import (
    SplitError,
    SplitValidationError,
    verify_frozen_manifest,
)

# Canonical 8-class taxonomy order for model targets and metrics evaluation
CANONICAL_CLASS_ORDER: tuple[str, ...] = (
    "O1",
    "OLOGN",
    "ON",
    "ONLOGN",
    "ON2",
    "ON3",
    "OEXP",
    "UNKNOWN",
)

# Authoritative 26-feature schema names
CANONICAL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES

# Expected CV fold cardinality in development partition (53 families total)
CV_FOLD_VALIDATION_PROGRAM_COUNTS: dict[int, int] = {0: 54, 1: 54, 2: 51}
CV_FOLD_VALIDATION_FAMILY_COUNTS: dict[int, int] = {0: 18, 1: 18, 2: 17}


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------


class EvaluationDataLoaderError(Exception):
    """Base exception for all evaluation data loading and isolation errors."""


class TestPartitionAccessError(EvaluationDataLoaderError):
    """Raised when an attempt is made to access test partition data or programs.

    The test partition is strictly frozen for final holdout benchmarking and must
    never be accessed during model fitting, feature scaling, tuning, or CV.
    """

    __test__ = False  # Prevent pytest from collecting this class as a test suite


class ManifestVerificationError(EvaluationDataLoaderError):
    """Raised when the split manifest fails integrity checks or sidecar verification."""


class QuarantinedFamilyAccessError(EvaluationDataLoaderError, ValueError):
    """Raised when an attempt is made to access quarantined legacy benchmark families."""


class FeatureExtractionError(EvaluationDataLoaderError):
    """Raised when structural feature extraction fails for a development program."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EvaluationRecord:
    """Immutable evaluation record for a single benchmark program.

    Attributes
    ----------
    program_id:
        Unique program identifier.
    family_id:
        Benchmark family identifier (e.g. 'f02').
    label:
        Ground truth complexity label from canonical taxonomy.
    fold:
        Assigned CV fold index (0, 1, or 2).
    tag:
        Benchmark difficulty tag ('easy' or 'rule_hard').
    source:
        Original C++ source code text.
    source_sha256:
        SHA-256 hex digest of normalized source text.
    unknown_reason:
        Sub-reason identifier if label is 'UNKNOWN', otherwise None.
    """

    program_id: str
    family_id: str
    label: str
    fold: int
    tag: str
    source: str
    source_sha256: str
    unknown_reason: str | None = None
    _features: FeatureVector | None = dataclasses.field(default=None, repr=False)

    @property
    def features(self) -> FeatureVector:
        """Canonical 26-feature vector extracted in memory.

        Computed on first access and cached in-memory on the record.
        """
        if self._features is not None:
            return self._features
        fv = _extract_record_features(self.source, self.program_id)
        object.__setattr__(self, "_features", fv)
        return fv


@dataclasses.dataclass(frozen=True)
class FoldSplit:
    """Immutable training and validation partition for a single CV fold."""

    fold: int
    train_records: tuple[EvaluationRecord, ...]
    val_records: tuple[EvaluationRecord, ...]

    @property
    def train_families(self) -> frozenset[str]:
        """Set of family IDs present in training records."""
        return frozenset(r.family_id for r in self.train_records)

    @property
    def val_families(self) -> frozenset[str]:
        """Set of family IDs present in validation records."""
        return frozenset(r.family_id for r in self.val_records)

    @property
    def train_program_ids(self) -> tuple[str, ...]:
        """Tuple of program IDs present in training records."""
        return tuple(r.program_id for r in self.train_records)

    @property
    def val_program_ids(self) -> tuple[str, ...]:
        """Tuple of program IDs present in validation records."""
        return tuple(r.program_id for r in self.val_records)

    def to_train_xy(self) -> tuple[list[list[int]], list[str]]:
        """Return (X_train, y_train) feature vectors and target labels."""
        X = [r.features.to_list() for r in self.train_records]
        y = [r.label for r in self.train_records]
        return X, y

    def to_val_xy(self) -> tuple[list[list[int]], list[str]]:
        """Return (X_val, y_val) feature vectors and target labels."""
        X = [r.features.to_list() for r in self.val_records]
        y = [r.label for r in self.val_records]
        return X, y


@dataclasses.dataclass(frozen=True)
class DevelopmentDataset:
    """Immutable container of verified development benchmark records.

    Provides safe accessors for cross-validation folds, individual development
    programs, and development families. Never contains or exposes test data.
    """

    records: tuple[EvaluationRecord, ...]
    manifest_path: str
    manifest_sha256: str
    _records_by_id: dict[str, EvaluationRecord] = dataclasses.field(
        default_factory=dict, repr=False
    )
    _records_by_family: dict[str, list[EvaluationRecord]] = dataclasses.field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if not self._records_by_id:
            by_id = {r.program_id: r for r in self.records}
            object.__setattr__(self, "_records_by_id", by_id)
        if not self._records_by_family:
            by_fam: dict[str, list[EvaluationRecord]] = {}
            for r in self.records:
                by_fam.setdefault(r.family_id, []).append(r)
            object.__setattr__(
                self,
                "_records_by_family",
                {k: tuple(v) for k, v in by_fam.items()},
            )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def program_ids(self) -> tuple[str, ...]:
        """Tuple of all 159 development program IDs in canonical order."""
        return tuple(r.program_id for r in self.records)

    @property
    def families(self) -> frozenset[str]:
        """Set of all 53 development family IDs."""
        return frozenset(r.family_id for r in self.records)

    def get_program(self, program_id: str) -> EvaluationRecord:
        """Retrieve a specific development program record by ID."""
        if program_id not in self._records_by_id:
            raise KeyError(
                f"Program '{program_id}' not found in development dataset."
            )
        return self._records_by_id[program_id]

    def get_family(self, family_id: str) -> tuple[EvaluationRecord, ...]:
        """Retrieve all development program records belonging to a family."""
        if family_id not in self._records_by_family:
            raise KeyError(
                f"Family '{family_id}' not found in development dataset."
            )
        return self._records_by_family[family_id]

    def get_fold(self, fold: int) -> FoldSplit:
        """Obtain train/validation split for one of the three frozen CV folds (0, 1, 2)."""
        return get_cv_fold(fold, dataset=self)

    def to_features_and_labels(self) -> tuple[list[list[int]], list[str]]:
        """Return (X, y) containing all development feature vectors and labels."""
        X = [r.features.to_list() for r in self.records]
        y = [r.label for r in self.records]
        return X, y

    def extract_all_features(self) -> None:
        """Force in-memory feature extraction for all records in this dataset."""
        for r in self.records:
            _ = r.features


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _extract_record_features(source: str, program_id: str) -> FeatureVector:
    """Parse C++ source and extract canonical 26-feature vector in memory."""
    try:
        ir = build_ir_from_source(source, filename=f"{program_id}.cpp")
        if ir.has_syntax_errors or not ir.analysis_complete:
            err_details = "; ".join(d.message for d in ir.diagnostics if d.is_error())
            raise FeatureExtractionError(
                f"Parsing failed with syntax errors for development program '{program_id}': "
                f"{err_details}"
            )
        fv = extract_features(ir)
    except FeatureExtractionError:
        raise
    except Exception as e:
        raise FeatureExtractionError(
            f"Feature extraction failed for development program '{program_id}': {e}"
        ) from e

    # Schema integrity assertions
    if len(fv.names) != 26 or fv.names != FEATURE_NAMES:
        raise FeatureExtractionError(
            f"Invalid feature schema for program '{program_id}': "
            "names do not match canonical schema."
        )
    if len(fv.values) != 26:
        raise FeatureExtractionError(
            f"Invalid feature vector length for program '{program_id}': "
            f"expected 26 values, got {len(fv.values)}."
        )
    for name, val in zip(fv.names, fv.values):
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise FeatureExtractionError(
                f"Invalid feature value {val!r} for '{name}' in program '{program_id}': "
                "must be non-negative int."
            )
    return fv


def _verify_manifest_data(
    manifest_path: str | Path,
    dataset_root: str | Path = "datasets",
) -> tuple[dict[str, Any], str]:
    """Verify frozen manifest against sidecar and return parsed manifest content with sha256."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise ManifestVerificationError(f"Split manifest file not found: {m_path}")

    try:
        res = verify_frozen_manifest(m_path, dataset_root=dataset_root)
    except (SplitValidationError, SplitError, FileNotFoundError, json.JSONDecodeError) as e:
        raise ManifestVerificationError(f"Frozen manifest verification failed: {e}") from e

    if res.get("status") != "VALID":
        raise ManifestVerificationError(
            f"Manifest verification returned invalid status: {res.get('status')}"
        )

    try:
        manifest_data = json.loads(m_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ManifestVerificationError(f"Failed to read verified manifest JSON: {e}") from e

    if manifest_data.get("status") != "frozen":
        raise ManifestVerificationError(
            f"Expected manifest status 'frozen', found '{manifest_data.get('status')}'"
        )

    actual_hash = str(res.get("sha256"))
    return manifest_data, actual_hash


# ---------------------------------------------------------------------------
# Public Query Functions (No Source Loading Required)
# ---------------------------------------------------------------------------


def get_canonical_feature_names() -> tuple[str, ...]:
    """Return the canonical 26 feature names in schema order without loading dataset."""
    return CANONICAL_FEATURE_NAMES


def get_canonical_class_order() -> tuple[str, ...]:
    """Return canonical 8 complexity class labels in evaluation order without loading dataset."""
    return CANONICAL_CLASS_ORDER


# ---------------------------------------------------------------------------
# Public Data Loaders
# ---------------------------------------------------------------------------


def load_development_dataset(
    manifest_path: str | Path = "data/splits/split_manifest_v1.0.frozen.json",
    dataset_root: str | Path = "datasets",
    eager_features: bool = False,
) -> DevelopmentDataset:
    """Load and verify the development partition from the frozen split manifest.

    Parameters
    ----------
    manifest_path:
        Path to the frozen split manifest JSON.
    dataset_root:
        Root path to dataset directory containing 'programs/' and 'families/'.
    eager_features:
        If True, extracts all 159 feature vectors into memory immediately upon loading.
        If False (default), features are lazily extracted in memory on first access.

    Returns
    -------
    DevelopmentDataset:
        Immutable dataset containing all 159 development program records.

    Raises
    ------
    ManifestVerificationError:
        If manifest file, sidecar, or dataset integrity check fails.
    FeatureExtractionError:
        If eager_features is True and feature extraction fails on any record.
    """
    manifest_data, actual_hash = _verify_manifest_data(manifest_path, dataset_root)
    root = Path(dataset_root)

    # Partition definitions from verified manifest
    partitions = manifest_data.get("partitions", {})
    dev_info = partitions.get("development", {})
    dev_families = frozenset(dev_info.get("families", []))

    programs = manifest_data.get("programs", [])
    dev_records: list[EvaluationRecord] = []

    for p in programs:
        if p.get("partition") != "development":
            # Strictly skip all test partition programs — never open or extract them
            continue

        pid = p["program_id"]
        fid = p["family_id"]
        expected_label = p["expected_label"]
        fold = p["fold"]
        tag = p["tag"]
        unknown_reason = p.get("unknown_reason")
        expected_sha = p["source_sha256"]

        if fid not in dev_families:
            raise ManifestVerificationError(
                f"Integrity violation: program '{pid}' claims family '{fid}' "
                "not in development partition."
            )

        src_path = root / "programs" / fid / f"{pid}.cpp"
        if not src_path.exists():
            raise ManifestVerificationError(
                f"Source file for development program '{pid}' not found: {src_path}"
            )

        source_text = src_path.read_text(encoding="utf-8")
        actual_sha = compute_source_sha256(source_text)
        if actual_sha != expected_sha:
            raise ManifestVerificationError(
                f"Source hash mismatch for program '{pid}': "
                f"manifest={expected_sha}, disk={actual_sha}"
            )

        record = EvaluationRecord(
            program_id=pid,
            family_id=fid,
            label=expected_label,
            fold=fold,
            tag=tag,
            source=source_text,
            source_sha256=expected_sha,
            unknown_reason=unknown_reason,
        )
        if eager_features:
            _ = record.features
        dev_records.append(record)

    # Sanity checks on loaded development set
    if len(dev_records) != 159:
        raise ManifestVerificationError(
            f"Expected 159 development programs, loaded {len(dev_records)}"
        )
    loaded_fams = frozenset(r.family_id for r in dev_records)
    if len(loaded_fams) != 53:
        raise ManifestVerificationError(
            f"Expected 53 development families, loaded {len(loaded_fams)}"
        )

    return DevelopmentDataset(
        records=tuple(dev_records),
        manifest_path=str(manifest_path),
        manifest_sha256=actual_hash,
    )


def get_cv_fold(
    fold: int,
    dataset: DevelopmentDataset | None = None,
    manifest_path: str | Path = "data/splits/split_manifest_v1.0.frozen.json",
    dataset_root: str | Path = "datasets",
) -> FoldSplit:
    """Retrieve train and validation split for one of the three frozen CV folds.

    Parameters
    ----------
    fold:
        Fold index (0, 1, or 2).
    dataset:
        Pre-loaded DevelopmentDataset. If None, loads from manifest_path.
    manifest_path:
        Path to frozen split manifest JSON (used if dataset is None).
    dataset_root:
        Path to dataset root directory (used if dataset is None).

    Returns
    -------
    FoldSplit:
        Tuple containing train_records and val_records for the fold.

    Raises
    ------
    ValueError:
        If fold is not 0, 1, or 2.
    EvaluationDataLoaderError:
        If family leakage, cardinality violation, or incomplete coverage is detected.
    """
    if fold not in (0, 1, 2):
        raise ValueError(f"Invalid fold index {fold}; expected 0, 1, or 2.")

    if dataset is None:
        dataset = load_development_dataset(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
        )

    val_records = tuple(r for r in dataset.records if r.fold == fold)
    train_records = tuple(r for r in dataset.records if r.fold != fold)

    val_families = frozenset(r.family_id for r in val_records)
    train_families = frozenset(r.family_id for r in train_records)

    # Isolation guarantee: disjointness
    overlap = train_families & val_families
    if overlap:
        raise EvaluationDataLoaderError(
            f"Evaluation isolation breach: family overlap between train and val in fold {fold}: "
            f"{overlap}"
        )

    # Exact cardinality verification
    expected_val_programs = CV_FOLD_VALIDATION_PROGRAM_COUNTS[fold]
    if len(val_records) != expected_val_programs:
        raise EvaluationDataLoaderError(
            f"Fold {fold} validation program count mismatch: expected {expected_val_programs}, "
            f"got {len(val_records)}"
        )

    expected_val_fams = CV_FOLD_VALIDATION_FAMILY_COUNTS[fold]
    if len(val_families) != expected_val_fams:
        raise EvaluationDataLoaderError(
            f"Fold {fold} validation family count mismatch: expected {expected_val_fams}, "
            f"got {len(val_families)}"
        )

    # Full coverage of development partition families
    if (train_families | val_families) != dataset.families:
        raise EvaluationDataLoaderError(
            f"Fold {fold} train and val families do not fully partition development families."
        )

    return FoldSplit(
        fold=fold,
        train_records=train_records,
        val_records=val_records,
    )


def load_development_program(
    program_id: str,
    manifest_path: str | Path = "data/splits/split_manifest_v1.0.frozen.json",
    dataset_root: str | Path = "datasets",
) -> EvaluationRecord:
    """Load a specific program record by ID, enforcing test and quarantine isolation guards.

    Raises
    ------
    TestPartitionAccessError:
        If program_id belongs to the test partition.
    QuarantinedFamilyAccessError:
        If program_id belongs to a quarantined legacy benchmark family.
    KeyError:
        If program_id is unknown or not in the development partition.
    """
    manifest_data, _ = _verify_manifest_data(manifest_path, dataset_root)

    test_programs: set[str] = set()
    dev_programs: dict[str, dict[str, Any]] = {}
    quarantined_fams = frozenset(manifest_data.get("quarantined_families", []))

    for p in manifest_data.get("programs", []):
        pid = p["program_id"]
        part = p.get("partition")
        fid = p.get("family_id")
        if fid in quarantined_fams:
            continue
        if part == "test":
            test_programs.add(pid)
        elif part == "development":
            dev_programs[pid] = p

    if program_id in test_programs:
        raise TestPartitionAccessError(
            f"Access to test partition program '{program_id}' is strictly forbidden "
            "by evaluation protocol."
        )

    if program_id not in dev_programs:
        # Check if program belongs to a quarantined family on disk
        root = Path(dataset_root)
        for qfam in quarantined_fams:
            if (root / "programs" / qfam / f"{program_id}.cpp").exists():
                raise QuarantinedFamilyAccessError(
                    f"Program '{program_id}' belongs to quarantined family '{qfam}' "
                    "and is excluded from evaluation."
                )
        raise KeyError(
            f"Program '{program_id}' not found in development partition of verified manifest."
        )

    p_info = dev_programs[program_id]
    fid = p_info["family_id"]
    src_path = Path(dataset_root) / "programs" / fid / f"{program_id}.cpp"
    source_text = src_path.read_text(encoding="utf-8")
    actual_sha = compute_source_sha256(source_text)
    expected_sha = p_info["source_sha256"]
    if actual_sha != expected_sha:
        raise ManifestVerificationError(
            f"Source hash mismatch for program '{program_id}': "
            f"manifest={expected_sha}, disk={actual_sha}"
        )

    return EvaluationRecord(
        program_id=program_id,
        family_id=fid,
        label=p_info["expected_label"],
        fold=p_info["fold"],
        tag=p_info["tag"],
        source=source_text,
        source_sha256=expected_sha,
        unknown_reason=p_info.get("unknown_reason"),
    )


def load_development_family(
    family_id: str,
    manifest_path: str | Path = "data/splits/split_manifest_v1.0.frozen.json",
    dataset_root: str | Path = "datasets",
) -> tuple[EvaluationRecord, ...]:
    """Load all program records for a development family, enforcing isolation guards.

    Raises
    ------
    TestPartitionAccessError:
        If family_id belongs to the test partition.
    QuarantinedFamilyAccessError:
        If family_id is a quarantined legacy benchmark family.
    KeyError:
        If family_id is unknown or not in the development partition.
    """
    manifest_data, _ = _verify_manifest_data(manifest_path, dataset_root)

    partitions = manifest_data.get("partitions", {})
    test_families = frozenset(partitions.get("test", {}).get("families", []))
    dev_families = frozenset(partitions.get("development", {}).get("families", []))
    quarantined_fams = frozenset(manifest_data.get("quarantined_families", []))

    if family_id in test_families:
        raise TestPartitionAccessError(
            f"Access to test partition family '{family_id}' is strictly forbidden "
            "by evaluation protocol."
        )
    if family_id in quarantined_fams:
        raise QuarantinedFamilyAccessError(
            f"Family '{family_id}' is quarantined and excluded from evaluation."
        )
    if family_id not in dev_families:
        raise KeyError(
            f"Family '{family_id}' not found in development partition of verified manifest."
        )

    # Load all records for this family
    dev_dataset = load_development_dataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    return dev_dataset.get_family(family_id)

