"""test_split.py — Tests for dataset partitioning, optimizer, CV folds, and lifecycle.

All git lifecycle tests use temporary isolated repositories under tmp_path with
explicit local git configuration to guarantee zero side-effects on the host repository.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import pytest

from complexity_lens.dataset import (
    ALLOWED_LABELS,
    FamilyMetadata,
)
from complexity_lens.split import (
    NORMALIZATION_VOCABULARY,
    QUARANTINED_FAMILIES,
    STRATIFICATION_DEFICIT_WARNING,
    AtomicUnit,
    CardinalityInfeasibilityError,
    FreezeGateError,
    InfeasibleSplitConstraintsError,
    SplitManifest,
    SplitValidationError,
    StratificationDeficitWarning,
    allocate_development_folds,
    canonical_json_dumps,
    compute_min_rank_sum_dp,
    compute_unit_rank,
    freeze_manifest,
    generate_proposal_manifest,
    get_git_info,
    jaccard_similarity,
    load_atomic_units,
    normalize_tokens,
    optimize_test_partition,
    tokenize_cpp_lexical,
    validate_manifest,
    verify_frozen_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures and Synthetic Generators
# ---------------------------------------------------------------------------


def run_git(
    repo: Path,
    args: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute git CLI command in an isolated repository."""
    return subprocess.run(
        ["git"] + args,
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def create_synthetic_family(
    family_id: str,
    label: str,
    tag: str = "easy",
    unknown_reason: str | None = None,
    num_programs: int = 3,
) -> FamilyMetadata:
    """Create a synthetic FamilyMetadata instance for isolated testing."""
    prog_ids = tuple(f"{family_id}_prog_{i}.cpp" for i in range(num_programs))
    annotations = {}
    if label == "UNKNOWN":
        reason = unknown_reason or "unsupported_construct"
        for pid in prog_ids:
            annotations[pid] = {
                "unknown_reason": reason,
                "unknown_explanation": f"Synthetic reason {reason}",
            }

    return FamilyMetadata(
        family_id=family_id,
        family_idea=f"Synthetic family {family_id}",
        label=label,
        label_rationale="Synthetic rationale",
        tag=tag,
        expected_parse="valid",
        programs=prog_ids,
        assumptions=("None",),
        program_annotations=annotations,
    )


def create_mock_dataset(
    tmp_path: Path,
    num_fams: int = 68,
) -> Path:
    """Create a synthetic mock dataset layout matching real dataset schema."""
    root = tmp_path / "mock_dataset"
    fams_dir = root / "families"
    progs_dir = root / "programs"
    fams_dir.mkdir(parents=True)
    progs_dir.mkdir(parents=True)

    # 68 families with distribution matching Phase 2:
    # O1: 8, OLOGN: 8, ON: 10, ONLOGN: 10, ON2: 9, ON3: 4, OEXP: 4, UNKNOWN: 15
    class_specs = [
        ("O1", 8, "easy", None),
        ("OLOGN", 8, "easy", None),
        ("ON", 10, "easy", None),
        ("ONLOGN", 10, "easy", None),
        ("ON2", 9, "easy", None),
        ("ON3", 4, "easy", None),
        ("OEXP", 4, "easy", None),
        ("UNKNOWN", 15, "rule_hard", "data_dependent"),
    ]

    reasons_cycle = [
        "data_dependent",
        "termination",
        "unsupported_construct",
        "external_call",
        "irregular_recursion",
    ]

    fam_num = 1
    for label, count, default_tag, default_reason in class_specs:
        for i in range(count):
            while f"f{fam_num:02d}" in QUARANTINED_FAMILIES:
                fam_num += 1
            fid = f"f{fam_num:02d}"
            fam_num += 1
            tag = "rule_hard" if label == "UNKNOWN" else default_tag
            reason = reasons_cycle[i % len(reasons_cycle)] if label == "UNKNOWN" else None

            p1 = f"{fid}_p1"
            p2 = f"{fid}_p2"
            p3 = f"{fid}_p3"

            yaml_lines = [
                f"family_id: {fid}",
                f"family_idea: Test family {fid}",
                f"label: {label}",
                'label_rationale: "Synthetic"',
                f"tag: {tag}",
                "expected_parse: valid",
                "programs:",
                f"  - {p1}",
                f"  - {p2}",
                f"  - {p3}",
                "assumptions:",
                '  - "None"',
            ]
            if label == "UNKNOWN":
                yaml_lines.extend(
                    [
                        "program_annotations:",
                        f"  {p1}:",
                        f'    unknown_reason: "{reason}"',
                        '    explanation: "Synthetic"',
                        f"  {p2}:",
                        f'    unknown_reason: "{reason}"',
                        '    explanation: "Synthetic"',
                        f"  {p3}:",
                        f'    unknown_reason: "{reason}"',
                        '    explanation: "Synthetic"',
                    ]
                )

            (fams_dir / f"{fid}.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

            # Write program C++ files under programs/<family_id>/<program_id>.cpp
            fam_prog_dir = progs_dir / fid
            fam_prog_dir.mkdir(parents=True, exist_ok=True)
            for p in (p1, p2, p3):
                code = (
                    f"// File {p}\n"
                    f"int main() {{ int a_{fid}_{p} = {fam_num}; return a_{fid}_{p}; }}\n"
                )
                (fam_prog_dir / f"{p}.cpp").write_text(code, encoding="utf-8")

    # Also create the 10 quarantined families
    for q_fid in sorted(QUARANTINED_FAMILIES):
        qp1, qp2, qp3 = f"{q_fid}_p1", f"{q_fid}_p2", f"{q_fid}_p3"
        q_lines = [
            f"family_id: {q_fid}",
            f"family_idea: Quarantined family {q_fid}",
            "label: ON",
            'label_rationale: "Quarantine"',
            "tag: easy",
            "expected_parse: valid",
            "programs:",
            f"  - {qp1}",
            f"  - {qp2}",
            f"  - {qp3}",
            "assumptions:",
            '  - "None"',
        ]
        (fams_dir / f"{q_fid}.yaml").write_text("\n".join(q_lines) + "\n", encoding="utf-8")
        q_prog_dir = progs_dir / q_fid
        q_prog_dir.mkdir(parents=True, exist_ok=True)
        for qp in (qp1, qp2, qp3):
            (q_prog_dir / f"{qp}.cpp").write_text(
                f"int main() {{ int q_{q_fid}_{qp} = 0; return q_{q_fid}_{qp}; }}\n",
                encoding="utf-8",
            )

    return root


# ---------------------------------------------------------------------------
# Unit Tests: Atomic Units and Ranks
# ---------------------------------------------------------------------------


class TestAtomicUnits:
    """Tests for atomic unit construction, ranking, and split group handling."""

    def test_compute_unit_rank_determinism(self) -> None:
        rank1 = compute_unit_rank(42, "f01")
        rank2 = compute_unit_rank(42, "f01")
        assert rank1 == rank2
        assert isinstance(rank1, int)
        assert rank1 > 0

        # Different seed or key changes rank
        rank_other_seed = compute_unit_rank(43, "f01")
        rank_other_key = compute_unit_rank(42, "f02")
        assert rank1 != rank_other_seed
        assert rank1 != rank_other_key

    def test_singletons_created_by_default(self) -> None:
        f1 = create_synthetic_family("f01", "O1")
        f2 = create_synthetic_family("f02", "ON")
        units = load_atomic_units([f1, f2], split_groups_path=None, seed=42)
        assert len(units) == 2
        assert units[0].weight == 1
        assert units[1].weight == 1

    def test_quarantined_families_excluded(self) -> None:
        f_normal = create_synthetic_family("f01", "O1")
        f_quarantine = create_synthetic_family("fam_constant_ops", "O1")
        units = load_atomic_units([f_normal, f_quarantine], seed=42)
        unit_fams = {fid for u in units for fid in u.family_ids}
        assert "f01" in unit_fams
        assert "fam_constant_ops" not in unit_fams

    def test_split_groups_yaml_grouping(self, tmp_path: Path) -> None:
        f1 = create_synthetic_family("f01", "O1")
        f2 = create_synthetic_family("f02", "ON")
        f3 = create_synthetic_family("f03", "ON2")

        yaml_content = "groups:\n  - [f01, f02]\n"
        sg_file = tmp_path / "split_groups.yaml"
        sg_file.write_text(yaml_content, encoding="utf-8")

        units = load_atomic_units([f1, f2, f3], split_groups_path=sg_file, seed=42)
        assert len(units) == 2  # (f01, f02) group and f03 singleton
        grp_unit = next(u for u in units if len(u.family_ids) == 2)
        assert grp_unit.family_ids == ("f01", "f02")
        assert grp_unit.weight == 2


# ---------------------------------------------------------------------------
# Unit Tests: Min Rank-Sum DP and Test Optimizer
# ---------------------------------------------------------------------------


class TestTestOptimizer:
    """Tests for exact branch-and-bound test partition optimization."""

    def test_min_rank_sum_dp_exactness(self) -> None:
        # Create small test units with known weights and ranks
        u1 = AtomicUnit(
            "u1", ("f1",), weight=1, rank=10, classes=("O1",), rule_hard_count=0, unknown_reasons=()
        )
        u2 = AtomicUnit(
            "u2", ("f2",), weight=2, rank=15, classes=("ON",), rule_hard_count=0, unknown_reasons=()
        )
        u3 = AtomicUnit(
            "u3",
            ("f3",),
            weight=3,
            rank=20,
            classes=("ON2",),
            rule_hard_count=0,
            unknown_reasons=(),
        )

        dp = compute_min_rank_sum_dp([u1, u2, u3], max_weight=6)
        assert dp[0] == 0
        assert dp[1] == 10  # u1
        assert dp[2] == 15  # u2
        assert dp[3] == 20  # u3 (min(20, 10+15=25) -> 20)
        assert dp[4] == 30  # u1 + u3 = 30
        assert dp[5] == 35  # u2 + u3 = 35
        assert dp[6] == 45  # u1 + u2 + u3 = 45

    def test_test_optimizer_hard_constraints_and_bounds(self) -> None:
        # Build 68 synthetic families with exact Phase 2 class profile
        fams: list[FamilyMetadata] = []
        fid = 1
        class_counts = [
            ("O1", 8),
            ("OLOGN", 8),
            ("ON", 10),
            ("ONLOGN", 10),
            ("ON2", 9),
            ("ON3", 4),
            ("OEXP", 4),
            ("UNKNOWN", 15),
        ]
        reasons = [
            "data_dependent",
            "termination",
            "unsupported_construct",
            "external_call",
            "irregular_recursion",
        ]
        for c, count in class_counts:
            for i in range(count):
                f_name = f"f{fid:02d}"
                tag = "rule_hard" if c == "UNKNOWN" else "easy"
                r = reasons[i % len(reasons)] if c == "UNKNOWN" else None
                fams.append(create_synthetic_family(f_name, c, tag=tag, unknown_reason=r))
                fid += 1

        units = load_atomic_units(fams, seed=42)
        test_units = optimize_test_partition(units, target_weight=15)

        assert len(test_units) == 15
        classes = [u.classes[0] for u in test_units]
        assert set(classes) == ALLOWED_LABELS
        assert classes.count("ON3") == 1
        assert classes.count("OEXP") == 1
        assert classes.count("UNKNOWN") == 3

        # Locked family f65 must NOT be in test
        test_fams = {fid for u in test_units for fid in u.family_ids}
        assert "f65" not in test_fams

        # Max 3 distinct unknown reasons achieved
        test_reasons = {r for u in test_units for r in u.unknown_reasons}
        assert len(test_reasons) == 3

        # Exactly 3 rule_hard
        total_rh = sum(u.rule_hard_count for u in test_units)
        assert total_rh == 3

    def test_infeasible_constraints_raises_error(self) -> None:
        # Not enough UNKNOWN units to satisfy TARGET_TEST_UNKNOWN (3)
        fams = [
            create_synthetic_family("f01", "O1"),
            create_synthetic_family("f02", "ON"),
        ]
        units = load_atomic_units(fams, seed=42)
        with pytest.raises(InfeasibleSplitConstraintsError):
            optimize_test_partition(units, target_weight=2)


# ---------------------------------------------------------------------------
# Unit Tests: Fold Allocation inside Development
# ---------------------------------------------------------------------------


class TestFoldAllocation:
    """Tests for cross-validation fold balancing and feasibility."""

    def test_fold_allocation_case_a(self) -> None:
        # Create 53 development units:
        # 3 ON3, 3 OEXP, 12 UNKNOWN, 6 O1, 6 OLOGN, 8 ON, 8 ONLOGN, 7 ON2 = 53
        fams: list[FamilyMetadata] = []
        fid = 1
        dev_profile = [
            ("ON3", 3),
            ("OEXP", 3),
            ("UNKNOWN", 12),
            ("O1", 6),
            ("OLOGN", 6),
            ("ON", 8),
            ("ONLOGN", 8),
            ("ON2", 7),
        ]
        for c, count in dev_profile:
            for _ in range(count):
                f_name = f"dev_f{fid:02d}"
                tag = "rule_hard" if c == "UNKNOWN" else "easy"
                fams.append(
                    create_synthetic_family(
                        f_name,
                        c,
                        tag=tag,
                        unknown_reason="data_dependent" if c == "UNKNOWN" else None,
                    )
                )
                fid += 1

        dev_units = load_atomic_units(fams, seed=42)
        folds, case = allocate_development_folds(dev_units, fold_sizes=(18, 18, 17))

        assert case == "Case A"
        assert len(folds) == 3
        assert sum(u.weight for u in folds[0]) == 18
        assert sum(u.weight for u in folds[1]) == 18
        assert sum(u.weight for u in folds[2]) == 17

        # Exactly 1 ON3 and 1 OEXP in each fold
        for f in range(3):
            fold_classes = [u.classes[0] for u in folds[f]]
            assert fold_classes.count("ON3") == 1
            assert fold_classes.count("OEXP") == 1
            # 8 classes present in every fold
            assert set(fold_classes) == ALLOWED_LABELS

        # Symmetry breaking: rank_sum(fold 0) <= rank_sum(fold 1)
        r0 = sum(u.rank for u in folds[0])
        r1 = sum(u.rank for u in folds[1])
        assert r0 <= r1

    def test_fold_allocation_cardinality_mismatch(self) -> None:
        u1 = AtomicUnit(
            "u1", ("f1",), weight=10, rank=1, classes=("O1",), rule_hard_count=0, unknown_reasons=()
        )
        with pytest.raises(CardinalityInfeasibilityError):
            allocate_development_folds([u1], fold_sizes=(18, 18, 17))

    def test_multi_on3_unit_raises_infeasible(self) -> None:
        """Atomic unit containing multiple ON3 families cannot be split; must fail."""
        # 1 unit with 2 ON3 families, 1 pure ON3 unit (total = 3 ON3 families)
        # 3 pure OEXP units, and remaining units summing to 53 total weight
        units: list[AtomicUnit] = [
            AtomicUnit(
                unit_id="u_multi_on3",
                family_ids=("f_on3_1", "f_on3_2"),
                weight=2,
                rank=100,
                classes=("ON3", "ON3"),
                rule_hard_count=0,
                unknown_reasons=(),
            ),
            AtomicUnit(
                unit_id="u_pure_on3",
                family_ids=("f_on3_3",),
                weight=1,
                rank=101,
                classes=("ON3",),
                rule_hard_count=0,
                unknown_reasons=(),
            ),
        ]
        # 3 pure OEXP units
        for i in range(3):
            units.append(
                AtomicUnit(
                    unit_id=f"u_oexp_{i}",
                    family_ids=(f"f_oexp_{i}",),
                    weight=1,
                    rank=200 + i,
                    classes=("OEXP",),
                    rule_hard_count=0,
                    unknown_reasons=(),
                )
            )
        # Remaining 47 units (all O1 singletons, weight 1)
        for i in range(47):
            units.append(
                AtomicUnit(
                    unit_id=f"u_other_{i}",
                    family_ids=(f"f_o1_{i}",),
                    weight=1,
                    rank=300 + i,
                    classes=("O1",),
                    rule_hard_count=0,
                    unknown_reasons=(),
                )
            )

        assert sum(u.weight for u in units) == 53
        assert sum(u.classes.count("ON3") for u in units) == 3
        with pytest.raises(InfeasibleSplitConstraintsError, match="cannot divide atomic unit"):
            allocate_development_folds(units, fold_sizes=(18, 18, 17))

    def test_multi_oexp_unit_raises_infeasible(self) -> None:
        """Atomic unit containing multiple OEXP families cannot be split; must fail."""
        units: list[AtomicUnit] = [
            AtomicUnit(
                unit_id="u_multi_oexp",
                family_ids=("f_oexp_1", "f_oexp_2"),
                weight=2,
                rank=100,
                classes=("OEXP", "OEXP"),
                rule_hard_count=0,
                unknown_reasons=(),
            ),
            AtomicUnit(
                unit_id="u_pure_oexp",
                family_ids=("f_oexp_3",),
                weight=1,
                rank=101,
                classes=("OEXP",),
                rule_hard_count=0,
                unknown_reasons=(),
            ),
        ]
        for i in range(3):
            units.append(
                AtomicUnit(
                    unit_id=f"u_on3_{i}",
                    family_ids=(f"f_on3_{i}",),
                    weight=1,
                    rank=200 + i,
                    classes=("ON3",),
                    rule_hard_count=0,
                    unknown_reasons=(),
                )
            )
        for i in range(47):
            units.append(
                AtomicUnit(
                    unit_id=f"u_other_{i}",
                    family_ids=(f"f_o1_{i}",),
                    weight=1,
                    rank=300 + i,
                    classes=("O1",),
                    rule_hard_count=0,
                    unknown_reasons=(),
                )
            )

        assert sum(u.weight for u in units) == 53
        assert sum(u.classes.count("OEXP") for u in units) == 3
        with pytest.raises(InfeasibleSplitConstraintsError, match="cannot divide atomic unit"):
            allocate_development_folds(units, fold_sizes=(18, 18, 17))

    def test_joint_on3_oexp_unit_assigned_to_fold(self) -> None:
        """An atomic unit containing both 1 ON3 and 1 OEXP family is valid and satisfies both."""
        units: list[AtomicUnit] = [
            AtomicUnit(
                unit_id="u_joint_on3_oexp",
                family_ids=("f_on3_1", "f_oexp_1"),
                weight=2,
                rank=50,
                classes=("ON3", "OEXP"),
                rule_hard_count=0,
                unknown_reasons=(),
            ),
            # 2 pure ON3 units
            AtomicUnit("u_on3_2", ("f_on3_2",), 1, 51, ("ON3",), 0, ()),
            AtomicUnit("u_on3_3", ("f_on3_3",), 1, 52, ("ON3",), 0, ()),
            # 2 pure OEXP units
            AtomicUnit("u_oexp_2", ("f_oexp_2",), 1, 53, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_3", ("f_oexp_3",), 1, 54, ("OEXP",), 0, ()),
        ]
        # 12 UNKNOWN units
        for i in range(12):
            units.append(
                AtomicUnit(
                    f"u_unk_{i}",
                    (f"f_unk_{i}",),
                    1,
                    100 + i,
                    ("UNKNOWN",),
                    1,
                    ("data_dependent",),
                )
            )
        # 35 remaining units: 6 O1, 6 OLOGN, 8 ON, 8 ONLOGN, 7 ON2
        remaining_classes = [("O1", 6), ("OLOGN", 6), ("ON", 8), ("ONLOGN", 8), ("ON2", 7)]
        uid = 0
        for c, count in remaining_classes:
            for _ in range(count):
                units.append(
                    AtomicUnit(f"u_other_{uid}", (f"f_{c}_{uid}",), 1, 200 + uid, (c,), 0, ())
                )
                uid += 1

        assert sum(u.weight for u in units) == 53
        assert sum(u.classes.count("ON3") for u in units) == 3
        assert sum(u.classes.count("OEXP") for u in units) == 3

        folds, case = allocate_development_folds(units, fold_sizes=(18, 18, 17))
        assert len(folds) == 3
        assert sum(u.weight for u in folds[0]) == 18
        assert sum(u.weight for u in folds[1]) == 18
        assert sum(u.weight for u in folds[2]) == 17

        # Exactly 1 ON3 and 1 OEXP family in each fold
        for f_idx, fold in enumerate(folds):
            on3_cnt = sum(u.classes.count("ON3") for u in fold)
            oexp_cnt = sum(u.classes.count("OEXP") for u in fold)
            assert on3_cnt == 1, f"Fold {f_idx} has {on3_cnt} ON3 families"
            assert oexp_cnt == 1, f"Fold {f_idx} has {oexp_cnt} OEXP families"

        # The joint unit must be intact in exactly one fold
        joint_in_folds = [
            f_idx
            for f_idx, fold in enumerate(folds)
            if any(u.unit_id == "u_joint_on3_oexp" for u in fold)
        ]
        assert len(joint_in_folds) == 1

    def test_impossible_group_configuration_raises_infeasible(self) -> None:
        """Atomic unit with weight 19 cannot fit in folds (18, 18, 17); must fail."""
        # Oversized atomic unit of weight 19
        oversized_fams = tuple(f"f_large_{i}" for i in range(19))
        oversized_classes = tuple(["O1"] * 19)
        units: list[AtomicUnit] = [
            AtomicUnit(
                unit_id="u_oversized",
                family_ids=oversized_fams,
                weight=19,
                rank=1,
                classes=oversized_classes,
                rule_hard_count=0,
                unknown_reasons=(),
            ),
            # 3 ON3, 3 OEXP
            AtomicUnit("u_on3_1", ("f_on3_1",), 1, 10, ("ON3",), 0, ()),
            AtomicUnit("u_on3_2", ("f_on3_2",), 1, 11, ("ON3",), 0, ()),
            AtomicUnit("u_on3_3", ("f_on3_3",), 1, 12, ("ON3",), 0, ()),
            AtomicUnit("u_oexp_1", ("f_oexp_1",), 1, 20, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_2", ("f_oexp_2",), 1, 21, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_3", ("f_oexp_3",), 1, 22, ("OEXP",), 0, ()),
        ]
        # 19 + 6 = 25; remaining 28 units of weight 1
        for i in range(28):
            units.append(AtomicUnit(f"u_rem_{i}", (f"f_rem_{i}",), 1, 100 + i, ("ON",), 0, ()))

        assert sum(u.weight for u in units) == 53
        with pytest.raises(InfeasibleSplitConstraintsError):
            allocate_development_folds(units, fold_sizes=(18, 18, 17))

    def test_greedy_misallocation_recovered_by_backtracking(self) -> None:
        """Greedy placement misallocates weights; backtracking recovers exact packing."""
        # 3 ON3 singletons, 3 OEXP singletons
        units: list[AtomicUnit] = [
            AtomicUnit("u_on3_0", ("f_on3_0",), 1, 10, ("ON3",), 0, ()),
            AtomicUnit("u_on3_1", ("f_on3_1",), 1, 11, ("ON3",), 0, ()),
            AtomicUnit("u_on3_2", ("f_on3_2",), 1, 12, ("ON3",), 0, ()),
            AtomicUnit("u_oexp_0", ("f_oexp_0",), 1, 20, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_1", ("f_oexp_1",), 1, 21, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_2", ("f_oexp_2",), 1, 22, ("OEXP",), 0, ()),
        ]
        # Multi-family units: four units of weight 8, three units of weight 5
        # Total weight: 6 + 32 + 15 = 53
        # Valid packing: Fold 0 (18) = bundle(2) + 8 + 8
        #                Fold 1 (18) = bundle(2) + 8 + 8
        #                Fold 2 (17) = bundle(2) + 5 + 5 + 5
        for i in range(4):
            fids = tuple(f"f_w8_{i}_{j}" for j in range(8))
            units.append(
                AtomicUnit(
                    unit_id=f"u_w8_{i}",
                    family_ids=fids,
                    weight=8,
                    rank=30 + i,
                    classes=("ON",) * 8,
                    rule_hard_count=0,
                    unknown_reasons=(),
                )
            )
        for i in range(3):
            fids = tuple(f"f_w5_{i}_{j}" for j in range(5))
            units.append(
                AtomicUnit(
                    unit_id=f"u_w5_{i}",
                    family_ids=fids,
                    weight=5,
                    rank=40 + i,
                    classes=("O1",) * 5,
                    rule_hard_count=0,
                    unknown_reasons=(),
                )
            )

        assert sum(u.weight for u in units) == 53
        folds, case = allocate_development_folds(units, fold_sizes=(18, 18, 17))

        assert len(folds) == 3
        assert sum(u.weight for u in folds[0]) == 18
        assert sum(u.weight for u in folds[1]) == 18
        assert sum(u.weight for u in folds[2]) == 17

        # Unit completeness & uniqueness
        all_uids = [u.unit_id for f in folds for u in f]
        assert len(all_uids) == len(units)
        assert set(all_uids) == {u.unit_id for u in units}

        # Multi-family units remain intact
        for u in units:
            matching = [
                f_idx for f_idx, f in enumerate(folds) if any(x.unit_id == u.unit_id for x in f)
            ]
            assert len(matching) == 1

        # Exactly 1 ON3 and 1 OEXP per fold
        for f in folds:
            assert sum(u.classes.count("ON3") for u in f) == 1
            assert sum(u.classes.count("OEXP") for u in f) == 1

    def test_genuinely_infeasible_configuration_proves_infeasible(self) -> None:
        """Units summing to 53 that mathematically cannot partition into (18, 18, 17)."""
        units: list[AtomicUnit] = [
            AtomicUnit("u_on3_0", ("f_on3_0",), 1, 10, ("ON3",), 0, ()),
            AtomicUnit("u_on3_1", ("f_on3_1",), 1, 11, ("ON3",), 0, ()),
            AtomicUnit("u_on3_2", ("f_on3_2",), 1, 12, ("ON3",), 0, ()),
            AtomicUnit("u_oexp_0", ("f_oexp_0",), 1, 20, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_1", ("f_oexp_1",), 1, 21, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_2", ("f_oexp_2",), 1, 22, ("OEXP",), 0, ()),
        ]
        # Remaining 47 weight made of 4 units of weight 10, and 1 unit of weight 7
        # Subsets of {10, 10, 10, 10, 7} cannot sum to 15 (needed for residual capacity of fold 2)
        for i in range(4):
            fids = tuple(f"f_w10_{i}_{j}" for j in range(10))
            units.append(
                AtomicUnit(
                    unit_id=f"u_w10_{i}",
                    family_ids=fids,
                    weight=10,
                    rank=30 + i,
                    classes=("ON",) * 10,
                    rule_hard_count=0,
                    unknown_reasons=(),
                )
            )
        fids_7 = tuple(f"f_w7_{j}" for j in range(7))
        units.append(
            AtomicUnit(
                unit_id="u_w7_0",
                family_ids=fids_7,
                weight=7,
                rank=40,
                classes=("O1",) * 7,
                rule_hard_count=0,
                unknown_reasons=(),
            )
        )
        assert sum(u.weight for u in units) == 53
        with pytest.raises(
            InfeasibleSplitConstraintsError,
            match="exhaustive search proved no valid assignment exists",
        ):
            allocate_development_folds(units, fold_sizes=(18, 18, 17))

    def test_case_b_warning_details(self) -> None:
        """Synthetic configuration: exact capacities feasible, 8-class coverage impossible."""
        # 3 ON3, 3 OEXP, 12 UNKNOWN, 1 O1, 1 OLOGN, 11 ON, 11 ONLOGN, 11 ON2 = 53 units
        # Because there is only 1 O1 family, at least 2 folds must be missing O1.
        # Because there is only 1 OLOGN family, at least 2 folds must be missing OLOGN.
        # Exact fold sizes (18, 18, 17) are trivially satisfied.
        units: list[AtomicUnit] = []
        uid = 0
        profile = [
            ("ON3", 3),
            ("OEXP", 3),
            ("UNKNOWN", 12),
            ("O1", 1),
            ("OLOGN", 1),
            ("ON", 11),
            ("ONLOGN", 11),
            ("ON2", 11),
        ]
        for c, count in profile:
            for _ in range(count):
                units.append(
                    AtomicUnit(
                        unit_id=f"u_b_{uid}",
                        family_ids=(f"f_b_{uid}",),
                        weight=1,
                        rank=100 + uid,
                        classes=(c,),
                        rule_hard_count=1 if c == "UNKNOWN" else 0,
                        unknown_reasons=("data_dependent",) if c == "UNKNOWN" else (),
                    )
                )
                uid += 1

        assert len(units) == 53
        assert sum(u.weight for u in units) == 53

        with pytest.warns(StratificationDeficitWarning) as warning_records:
            folds, case = allocate_development_folds(units, fold_sizes=(18, 18, 17))

        assert case == "Case B"
        assert len(folds) == 3
        # Invariant 1: Exact capacities preserved
        assert sum(u.weight for u in folds[0]) == 18
        assert sum(u.weight for u in folds[1]) == 18
        assert sum(u.weight for u in folds[2]) == 17

        # Invariant 2: Atomic unit integrity
        assigned_uids = [u.unit_id for f in folds for u in f]
        assert len(assigned_uids) == 53
        assert len(set(assigned_uids)) == 53

        # Invariant 3: Hard constraints on ON3 and OEXP seeds preserved
        for f in folds:
            assert sum(u.classes.count("ON3") for u in f) == 1
            assert sum(u.classes.count("OEXP") for u in f) == 1

        # Invariant 4: Warning format and contents
        assert len(warning_records) == 1
        w_text = str(warning_records[0].message)
        assert STRATIFICATION_DEFICIT_WARNING in w_text
        assert "total_coverage_deficit=4" in w_text
        assert "not a mathematically proven minimal deficit" in w_text

        # Invariant 5: Each affected fold has missing labels and counts reported
        assert "Fold 0" in w_text
        assert "Fold 1" in w_text
        assert "Fold 2" in w_text
        assert "missing=" in w_text
        assert "counts=" in w_text

        # Verify exact missing fold/class combinations sum to total deficit (4)
        total_missing = 0
        for f in folds:
            f_classes = {c for u in f for c in u.classes}
            missing_in_f = ALLOWED_LABELS - f_classes
            total_missing += len(missing_in_f)
        assert total_missing == 4

    def test_case_a_remains_preferred(self) -> None:
        """When full class coverage is feasible, Case A is returned with no warning."""
        units: list[AtomicUnit] = []
        uid = 0
        profile = [
            ("ON3", 3),
            ("OEXP", 3),
            ("UNKNOWN", 12),
            ("O1", 6),
            ("OLOGN", 6),
            ("ON", 8),
            ("ONLOGN", 8),
            ("ON2", 7),
        ]
        for c, count in profile:
            for _ in range(count):
                units.append(
                    AtomicUnit(
                        unit_id=f"u_a_{uid}",
                        family_ids=(f"f_a_{uid}",),
                        weight=1,
                        rank=100 + uid,
                        classes=(c,),
                        rule_hard_count=1 if c == "UNKNOWN" else 0,
                        unknown_reasons=("data_dependent",) if c == "UNKNOWN" else (),
                    )
                )
                uid += 1

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            folds, case = allocate_development_folds(units, fold_sizes=(18, 18, 17))

        assert case == "Case A"
        assert len(recorded_warnings) == 0
        for f in folds:
            f_classes = {c for u in f for c in u.classes}
            assert f_classes == ALLOWED_LABELS

    def test_case_c_remains_distinct_and_raises_cardinality_infeasibility(self) -> None:
        """When exact fold sizes cannot be satisfied, CardinalityInfeasibilityError is raised."""
        units: list[AtomicUnit] = [
            AtomicUnit("u_on3_0", ("f_on3_0",), 1, 10, ("ON3",), 0, ()),
            AtomicUnit("u_on3_1", ("f_on3_1",), 1, 11, ("ON3",), 0, ()),
            AtomicUnit("u_on3_2", ("f_on3_2",), 1, 12, ("ON3",), 0, ()),
            AtomicUnit("u_oexp_0", ("f_oexp_0",), 1, 20, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_1", ("f_oexp_1",), 1, 21, ("OEXP",), 0, ()),
            AtomicUnit("u_oexp_2", ("f_oexp_2",), 1, 22, ("OEXP",), 0, ()),
        ]
        for i in range(4):
            units.append(
                AtomicUnit(
                    unit_id=f"u_w10_{i}",
                    family_ids=tuple(f"f_w10_{i}_{j}" for j in range(10)),
                    weight=10,
                    rank=30 + i,
                    classes=("ON",) * 10,
                    rule_hard_count=0,
                    unknown_reasons=(),
                )
            )
        units.append(
            AtomicUnit(
                unit_id="u_w7_0",
                family_ids=tuple(f"f_w7_{j}" for j in range(7)),
                weight=7,
                rank=40,
                classes=("O1",) * 7,
                rule_hard_count=0,
                unknown_reasons=(),
            )
        )
        assert sum(u.weight for u in units) == 53
        with pytest.raises(CardinalityInfeasibilityError):
            allocate_development_folds(units, fold_sizes=(18, 18, 17))

    def test_case_b_deterministic_reporting(self) -> None:
        """Repeated runs of the same Case B configuration produce identical folds and warnings."""
        units: list[AtomicUnit] = []
        uid = 0
        profile = [
            ("ON3", 3),
            ("OEXP", 3),
            ("UNKNOWN", 12),
            ("O1", 1),
            ("OLOGN", 1),
            ("ON", 11),
            ("ONLOGN", 11),
            ("ON2", 11),
        ]
        for c, count in profile:
            for _ in range(count):
                units.append(
                    AtomicUnit(
                        unit_id=f"u_det_{uid}",
                        family_ids=(f"f_det_{uid}",),
                        weight=1,
                        rank=100 + uid,
                        classes=(c,),
                        rule_hard_count=1 if c == "UNKNOWN" else 0,
                        unknown_reasons=("data_dependent",) if c == "UNKNOWN" else (),
                    )
                )
                uid += 1

        results = []
        warning_messages = []
        for _ in range(3):
            with pytest.warns(StratificationDeficitWarning) as rec:
                folds, case = allocate_development_folds(units, fold_sizes=(18, 18, 17))
            fold_uids = [[u.unit_id for u in f] for f in folds]
            results.append((fold_uids, case))
            warning_messages.append(str(rec[0].message))

        assert results[0] == results[1] == results[2]
        assert results[0][1] == "Case B"
        assert warning_messages[0] == warning_messages[1] == warning_messages[2]


# ---------------------------------------------------------------------------
# Unit Tests: Tokenization and Similarity
# ---------------------------------------------------------------------------


class TestLexicalSimilarity:
    """Tests for tokenization, normalization, and Jaccard similarity."""

    def test_vocabulary_size(self) -> None:
        assert len(NORMALIZATION_VOCABULARY) == 37

    def test_tokenization_and_normalization(self) -> None:
        code = """
        // Loop example
        int main() {
            int customVar = 0;
            for (size_t i = 0; i < 10; ++i) {
                customVar += i;
            }
            return customVar;
        }
        """
        toks = tokenize_cpp_lexical(code)
        assert "// Loop example" not in toks
        assert "for" in toks
        assert "size_t" in toks
        assert "customVar" in toks

        norm = normalize_tokens(toks)
        assert "ID" in norm  # customVar mapped to ID
        assert "for" in norm  # keyword preserved
        assert "size_t" in norm  # stdlib identifier preserved

    def test_jaccard_similarity(self) -> None:
        s1 = {("int", "main", "("), ("main", "(", ")")}
        s2 = {("int", "main", "("), ("main", "(", ")")}
        s3 = {("void", "test", "("), ("test", "(", ")")}
        assert jaccard_similarity(s1, s2) == 1.0
        assert jaccard_similarity(s1, s3) == 0.0


# ---------------------------------------------------------------------------
# Unit Tests: Frozen-File Overwrite Protection
# ---------------------------------------------------------------------------


class TestFreezeOverwriteProtection:
    """Tests guaranteeing that existing frozen manifests or sidecars are never overwritten."""

    @pytest.fixture
    def clean_env(self, tmp_path: Path) -> tuple[Path, SplitManifest, Path]:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        assert str(repo_dir).startswith(str(tmp_path))

        env = {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(tmp_path / "isolated_gitconfig"),
        }
        run_git(repo_dir, ["init"], env=env)
        run_git(repo_dir, ["config", "user.name", "Test User"], env=env)
        run_git(repo_dir, ["config", "user.email", "test@complexitylens.org"], env=env)
        run_git(repo_dir, ["config", "core.autocrlf", "false"], env=env)

        mock_ds = create_mock_dataset(repo_dir)
        run_git(repo_dir, ["add", "."], env=env)
        run_git(repo_dir, ["commit", "-m", "Commit dataset"], env=env)

        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)
        return repo_dir, manifest, mock_ds

    def test_existing_frozen_manifest_blocks_freezing(
        self, clean_env: tuple[Path, SplitManifest, Path], tmp_path: Path
    ) -> None:
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        frozen_out.parent.mkdir(parents=True, exist_ok=True)
        frozen_out.write_text("existing content\n", encoding="utf-8")

        with pytest.raises(FreezeGateError, match="Frozen manifest already exists"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

    def test_existing_sidecar_blocks_freezing(
        self, clean_env: tuple[Path, SplitManifest, Path], tmp_path: Path
    ) -> None:
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        sidecar = frozen_out.with_name(f"{frozen_out.name}.sha256")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("dummy_hash  split_manifest_v1.0.frozen.json\n", encoding="utf-8")

        assert not frozen_out.exists()
        with pytest.raises(FreezeGateError, match="Sidecar checksum file already exists"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        # Assert no partial new output was created
        assert not frozen_out.exists()

    def test_existing_frozen_manifest_remains_byte_identical(
        self, clean_env: tuple[Path, SplitManifest, Path], tmp_path: Path
    ) -> None:
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        frozen_out.parent.mkdir(parents=True, exist_ok=True)
        sentinel_bytes = b"EXACT_SENTINEL_FROZEN_BYTES_PRESERVED_12345\n"
        frozen_out.write_bytes(sentinel_bytes)

        with pytest.raises(FreezeGateError):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        assert frozen_out.read_bytes() == sentinel_bytes

    def test_existing_sidecar_remains_byte_identical(
        self, clean_env: tuple[Path, SplitManifest, Path], tmp_path: Path
    ) -> None:
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        sidecar = frozen_out.with_name(f"{frozen_out.name}.sha256")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sentinel_bytes = b"EXACT_SENTINEL_SIDECAR_BYTES_PRESERVED_67890\n"
        sidecar.write_bytes(sentinel_bytes)

        with pytest.raises(FreezeGateError):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        assert sidecar.read_bytes() == sentinel_bytes

    def test_no_partial_new_output_left_behind(
        self, clean_env: tuple[Path, SplitManifest, Path], tmp_path: Path
    ) -> None:
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        sidecar = frozen_out.with_name(f"{frozen_out.name}.sha256")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("existing sidecar\n", encoding="utf-8")

        assert not frozen_out.exists()
        with pytest.raises(FreezeGateError):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        assert not frozen_out.exists()

    def test_destination_created_between_preflight_and_exclusive_creation_blocked(
        self, clean_env: tuple[Path, SplitManifest, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exclusive creation mode ('xb') prevents overwrite if file appears concurrently."""
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"

        orig_open = builtins.open

        def mock_open(path, mode="r", *args, **kwargs):
            if str(path) == str(frozen_out) and "x" in mode:
                raise FileExistsError("Concurrent process created file")
            return orig_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(FreezeGateError, match="already exists"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

    def test_simulated_sidecar_creation_failure_cleans_up_manifest(
        self, clean_env: tuple[Path, SplitManifest, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If sidecar write fails, manifest is unlinked so no partial artifacts remain."""
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        sidecar = frozen_out.with_name(f"{frozen_out.name}.sha256")

        orig_open = builtins.open

        def mock_open(path, mode="r", *args, **kwargs):
            if str(path) == str(sidecar) and "x" in mode:
                raise OSError("Simulated disk I/O failure writing sidecar")
            return orig_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        assert not frozen_out.exists()
        assert not sidecar.exists()

        with pytest.raises(FreezeGateError, match="Failed to create sidecar checksum file"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        # Crucial verification: manifest created during the call must be rolled back
        assert not frozen_out.exists()
        assert not sidecar.exists()

    def test_cleanup_does_not_delete_preexisting_sidecar(
        self, clean_env: tuple[Path, SplitManifest, Path]
    ) -> None:
        """Cleanup logic must never delete a pre-existing file on disk."""
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        sidecar = frozen_out.with_name(f"{frozen_out.name}.sha256")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sentinel = b"PREEXISTING_SIDECAR_CONTENT_DO_NOT_DELETE\n"
        sidecar.write_bytes(sentinel)

        with pytest.raises(FreezeGateError):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        # Pre-existing sidecar remains untouched
        assert sidecar.exists()
        assert sidecar.read_bytes() == sentinel

    def test_both_files_produced_on_success_and_checksum_matches(
        self, clean_env: tuple[Path, SplitManifest, Path]
    ) -> None:
        """Successful freezing produces canonical manifest and exactly matching SHA-256 sidecar."""
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        sidecar = frozen_out.with_name(f"{frozen_out.name}.sha256")

        res_path = freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)
        assert res_path == frozen_out
        assert frozen_out.exists()
        assert sidecar.exists()

        manifest_bytes = frozen_out.read_bytes()
        expected_hash = hashlib.sha256(manifest_bytes).hexdigest()
        sidecar_text = sidecar.read_text(encoding="utf-8")
        assert sidecar_text == f"{expected_hash}  {frozen_out.name}\n"

    def test_manifest_write_failure_cleans_up_partial_manifest(
        self, clean_env: tuple[Path, SplitManifest, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If writing manifest fails after creation, partial manifest is unlinked."""
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        sidecar = frozen_out.with_name(f"{frozen_out.name}.sha256")

        orig_open = builtins.open

        def mock_open(path, mode="r", *args, **kwargs):
            f = orig_open(path, mode, *args, **kwargs)
            if str(path) == str(frozen_out) and "x" in mode:
                class FailingWriter:
                    def __init__(self, real_f: Any) -> None:
                        self._real_f = real_f

                    def write(self, data: bytes) -> int:
                        raise OSError("Simulated disk full during manifest write")

                    def flush(self) -> None:
                        self._real_f.flush()

                    def close(self) -> None:
                        self._real_f.close()

                    def __enter__(self) -> FailingWriter:
                        return self

                    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                        self._real_f.close()

                return FailingWriter(f)
            return f

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(FreezeGateError, match="Failed to write frozen manifest"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        # Incomplete manifest must not remain on disk
        assert not frozen_out.exists()
        assert not sidecar.exists()

    def test_sidecar_write_failure_cleans_up_both_manifest_and_sidecar(
        self, clean_env: tuple[Path, SplitManifest, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If writing sidecar fails after creation, both manifest and sidecar are unlinked."""
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        sidecar = frozen_out.with_name(f"{frozen_out.name}.sha256")

        orig_open = builtins.open

        def mock_open(path, mode="r", *args, **kwargs):
            f = orig_open(path, mode, *args, **kwargs)
            if str(path) == str(sidecar) and "x" in mode:
                class FailingWriter:
                    def __init__(self, real_f: Any) -> None:
                        self._real_f = real_f

                    def write(self, data: bytes) -> int:
                        raise OSError("Simulated disk full during sidecar write")

                    def flush(self) -> None:
                        self._real_f.flush()

                    def close(self) -> None:
                        self._real_f.close()

                    def __enter__(self) -> FailingWriter:
                        return self

                    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                        self._real_f.close()

                return FailingWriter(f)
            return f

        monkeypatch.setattr(builtins, "open", mock_open)

        with pytest.raises(FreezeGateError, match="Failed to write sidecar checksum file"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        # Neither file created during this invocation should remain
        assert not frozen_out.exists()
        assert not sidecar.exists()

    def test_cleanup_failure_is_observable_and_preserves_primary_error(
        self, clean_env: tuple[Path, SplitManifest, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If rollback cleanup itself fails, error is observable alongside primary error."""
        repo_dir, manifest, mock_ds = clean_env
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"

        orig_open = builtins.open
        orig_unlink = Path.unlink

        def mock_open(path, mode="r", *args, **kwargs):
            if str(path).endswith(".sha256") and "x" in mode:
                raise OSError("Primary I/O error creating sidecar")
            return orig_open(path, mode, *args, **kwargs)

        def mock_unlink(self, *args, **kwargs):
            if str(self) == str(frozen_out):
                raise PermissionError("Simulated locked file: access denied during rollback")
            return orig_unlink(self, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)
        monkeypatch.setattr(Path, "unlink", mock_unlink)

        with pytest.raises(FreezeGateError) as exc_info:
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        err_str = str(exc_info.value)
        assert "Primary I/O error creating sidecar" in err_str
        assert "rollback cleanup failed" in err_str
        assert "PermissionError" in err_str
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Unit Tests: Explicit Git Ancestry Gate
# ---------------------------------------------------------------------------


class TestGitAncestryGate:
    """Tests enforcing explicit git ancestry from base_dataset_commit and integrity gates."""

    @pytest.fixture
    def git_env(self, tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
        repo_dir = tmp_path / "git_ancestry_repo"
        repo_dir.mkdir()
        assert str(repo_dir).startswith(str(tmp_path))

        env = {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(tmp_path / "isolated_gitconfig"),
        }
        run_git(repo_dir, ["init"], env=env)
        run_git(repo_dir, ["config", "user.name", "Test User"], env=env)
        run_git(repo_dir, ["config", "user.email", "test@complexitylens.org"], env=env)
        run_git(repo_dir, ["config", "core.autocrlf", "false"], env=env)

        mock_ds = create_mock_dataset(repo_dir)
        run_git(repo_dir, ["add", "."], env=env)
        run_git(repo_dir, ["commit", "-m", "Base dataset commit"], env=env)

        return repo_dir, mock_ds, env

    def test_valid_descendant_head_succeeds(
        self, git_env: tuple[Path, Path, dict[str, str]]
    ) -> None:
        repo_dir, mock_ds, env = git_env
        base_commit, _ = get_git_info(repo_dir)
        assert base_commit is not None

        # Generate proposal at base commit
        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)
        assert manifest.base_dataset_commit == base_commit

        # Make another commit (descendant HEAD) adding a harmless doc file
        (repo_dir / "NOTES.md").write_text("# Notes\n", encoding="utf-8")
        run_git(repo_dir, ["add", "NOTES.md"], env=env)
        run_git(repo_dir, ["commit", "-m", "Add notes"], env=env)

        head_commit, _ = get_git_info(repo_dir)
        assert head_commit != base_commit

        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        res_path = freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)
        assert res_path.exists()

    def test_head_equal_to_base_commit_succeeds(
        self, git_env: tuple[Path, Path, dict[str, str]]
    ) -> None:
        repo_dir, mock_ds, env = git_env
        base_commit, _ = get_git_info(repo_dir)

        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        res_path = freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)
        assert res_path.exists()

    def test_unrelated_histories_rejected(
        self, git_env: tuple[Path, Path, dict[str, str]]
    ) -> None:
        repo_dir, mock_ds, env = git_env
        base_commit, _ = get_git_info(repo_dir)
        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)

        # Create an unrelated orphan branch with its own root commit
        run_git(repo_dir, ["checkout", "--orphan", "unrelated_branch"], env=env)
        (repo_dir / "orphan.txt").write_text("orphan\n", encoding="utf-8")
        run_git(repo_dir, ["add", "orphan.txt"], env=env)
        run_git(repo_dir, ["commit", "-m", "Unrelated root"], env=env)

        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        with pytest.raises(FreezeGateError, match="not a descendant"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

    def test_nonexistent_base_commit_rejected(
        self, git_env: tuple[Path, Path, dict[str, str]]
    ) -> None:
        repo_dir, mock_ds, env = git_env
        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)
        manifest_data = manifest.to_dict()
        manifest_data["base_dataset_commit"] = "0" * 40

        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        with pytest.raises(FreezeGateError, match="does not exist in repository"):
            freeze_manifest(manifest_data, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

    def test_proposal_content_differing_from_head_rejected(
        self, git_env: tuple[Path, Path, dict[str, str]]
    ) -> None:
        repo_dir, mock_ds, env = git_env
        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)
        prop_path = repo_dir / "data" / "splits" / "split_manifest_v1.0.proposed.json"
        prop_path.parent.mkdir(parents=True, exist_ok=True)
        prop_path.write_text(manifest.to_json(), encoding="utf-8")

        run_git(repo_dir, ["add", "."], env=env)
        run_git(repo_dir, ["commit", "-m", "Commit proposal"], env=env)

        # Caller attempts to freeze a tampered proposal differing from committed HEAD
        tampered_manifest = manifest.to_dict()
        tampered_manifest["seed"] = 9999

        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        with pytest.raises(FreezeGateError, match="differs from the version committed in HEAD"):
            freeze_manifest(tampered_manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

    def test_dataset_governance_change_after_base_commit_rejected(
        self, git_env: tuple[Path, Path, dict[str, str]]
    ) -> None:
        repo_dir, mock_ds, env = git_env
        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)

        # Modify a program inside mock dataset and commit it
        any_prog = next(mock_ds.glob("programs/*/*.cpp"))
        any_prog.write_text("// Modified after base commit\nint x = 0;\n", encoding="utf-8")
        run_git(repo_dir, ["add", "."], env=env)
        run_git(repo_dir, ["commit", "-m", "Mutate dataset"], env=env)

        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        with pytest.raises(FreezeGateError, match="modified after base commit"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

    def test_dirty_working_tree_rejected(
        self, git_env: tuple[Path, Path, dict[str, str]]
    ) -> None:
        repo_dir, mock_ds, env = git_env
        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)

        (repo_dir / "untracked_dirty.txt").write_text("dirty\n", encoding="utf-8")
        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        with pytest.raises(FreezeGateError, match="Working tree is dirty"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

    def test_successful_freezing_records_proposal_commit(
        self, git_env: tuple[Path, Path, dict[str, str]]
    ) -> None:
        repo_dir, mock_ds, env = git_env
        base_commit, _ = get_git_info(repo_dir)
        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=repo_dir)

        frozen_out = repo_dir / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        res_path = freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=repo_dir)

        frozen_data = json.loads(res_path.read_text(encoding="utf-8"))
        head_commit, _ = get_git_info(repo_dir)
        assert frozen_data["proposal_commit"] == head_commit
        assert frozen_data["base_dataset_commit"] == base_commit
        assert frozen_data["status"] == "frozen"
        assert not frozen_data["git_dirty"]


# ---------------------------------------------------------------------------
# Git Lifecycle and Manifest Verification (Isolated in tmp_path)
# ---------------------------------------------------------------------------


class TestManifestLifecycleAndFreeze:
    """Tests proposal generation, validation, freeze gate, and verification."""

    @pytest.fixture
    def isolated_git_repo(self, tmp_path: Path) -> Path:
        """Create a clean isolated temporary git repository."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        assert str(repo_dir).startswith(str(tmp_path))

        env = {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(tmp_path / "isolated_gitconfig"),
        }
        run_git(repo_dir, ["init"], env=env)
        run_git(repo_dir, ["config", "user.name", "Test User"], env=env)
        run_git(repo_dir, ["config", "user.email", "test@complexitylens.org"], env=env)
        run_git(repo_dir, ["config", "core.autocrlf", "false"], env=env)

        (repo_dir / "README.md").write_text("# Test Repo\n", encoding="utf-8")
        run_git(repo_dir, ["add", "README.md"], env=env)
        run_git(repo_dir, ["commit", "-m", "Initial commit"], env=env)

        return repo_dir

    def test_proposal_and_validation(self, tmp_path: Path) -> None:
        mock_ds = create_mock_dataset(tmp_path)
        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=mock_ds)

        assert manifest.status == "proposed"
        assert manifest.partitions["test"]["family_count"] == 15
        assert manifest.partitions["development"]["family_count"] == 53
        assert len(manifest.cv_folds) == 3

        # Validate
        errors = validate_manifest(manifest, dataset_root=mock_ds)
        assert errors == []

    def test_freeze_gate_blocks_on_dirty_working_tree(
        self, isolated_git_repo: Path, tmp_path: Path
    ) -> None:
        mock_ds = create_mock_dataset(isolated_git_repo)
        run_git(isolated_git_repo, ["add", "."])
        run_git(isolated_git_repo, ["commit", "-m", "Add dataset"])

        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=isolated_git_repo)

        (isolated_git_repo / "dirty_file.txt").write_text("dirty\n", encoding="utf-8")

        frozen_out = isolated_git_repo / "data" / "splits" / "split_manifest_v1.0.frozen.json"
        with pytest.raises(FreezeGateError, match="Working tree is dirty"):
            freeze_manifest(manifest, frozen_out, dataset_root=mock_ds, repo_root=isolated_git_repo)

    def test_freeze_and_verify_success_in_clean_repo(self, isolated_git_repo: Path) -> None:
        mock_ds = create_mock_dataset(isolated_git_repo)
        run_git(isolated_git_repo, ["add", "."])
        run_git(isolated_git_repo, ["commit", "-m", "Add dataset"])

        manifest = generate_proposal_manifest(dataset_root=mock_ds, repo_root=isolated_git_repo)
        frozen_out = isolated_git_repo / "data" / "splits" / "split_manifest_v1.0.frozen.json"

        res_path = freeze_manifest(
            manifest, frozen_out, dataset_root=mock_ds, repo_root=isolated_git_repo
        )
        assert res_path.exists()

        sidecar = res_path.with_name(f"{res_path.name}.sha256")
        assert sidecar.exists()

        # Verify sidecar hash
        v_res = verify_frozen_manifest(res_path, dataset_root=mock_ds, repo_root=isolated_git_repo)
        assert v_res["status"] == "VALID"

        # Tampering with frozen manifest fails verification
        tampered_data = json.loads(res_path.read_text(encoding="utf-8"))
        tampered_data["seed"] = 9999
        res_path.write_text(canonical_json_dumps(tampered_data), encoding="utf-8")

        with pytest.raises(SplitValidationError, match="checksum mismatch"):
            verify_frozen_manifest(res_path, dataset_root=mock_ds, repo_root=isolated_git_repo)
