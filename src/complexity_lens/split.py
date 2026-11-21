"""split.py — Dataset partitioning, test optimization, cross-validation folds, and split lifecycle.

Provides reproducible, family-aware development/test partitioning and cross-validation
fold allocation for ComplexityLens.

Terminology:
- Exactly two partitions: 'development' and 'test'. There is NO 'train' partition.
- Cross-validation folds (0, 1, 2) exist strictly inside 'development'.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import itertools
import json
import re
import subprocess
import warnings
from pathlib import Path
from typing import Any

from complexity_lens.dataset import (
    ALLOWED_LABELS,
    FamilyMetadata,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Constants and Policy Configurations
# ---------------------------------------------------------------------------

ALLOWED_PARTITIONS: frozenset[str] = frozenset({"development", "test"})

TARGET_TEST_FAMILIES: int = 15
TARGET_DEV_FAMILIES: int = 53
TARGET_TEST_RULE_HARD: int = 3
TARGET_TEST_UNKNOWN: int = 3
TARGET_TEST_ON3: int = 1
TARGET_TEST_OEXP: int = 1

CV_FOLD_SIZES: tuple[int, int, int] = (18, 18, 17)
NUM_CV_FOLDS: int = 3

DEFAULT_SPLIT_SEED: int = 42

LOCKED_DEV_FAMILIES: frozenset[str] = frozenset({"f65"})

QUARANTINED_FAMILIES: frozenset[str] = frozenset(
    {
        "fam_binary_search",
        "fam_constant_ops",
        "fam_exponential_recurrence",
        "fam_linear_scan",
        "fam_quadratic_nested",
        "fam_rule_hard_divide_conquer",
        "fam_rule_hard_linear",
        "fam_rule_hard_nested",
        "fam_unknown_data_dependent",
        "fam_unknown_external_unsupported",
    }
)

SPLIT_SCHEMA_VERSION: str = "1.0"
SPLIT_DATASET_VERSION: str = "1.0"

# Normalization vocabulary: 29 language keywords + 8 stdlib identifiers = 37 tokens
NORMALIZATION_VOCABULARY: frozenset[str] = frozenset(
    {
        # 29 C++ Keywords
        "auto",
        "bool",
        "break",
        "case",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "false",
        "float",
        "for",
        "goto",
        "if",
        "int",
        "long",
        "nullptr",
        "return",
        "short",
        "sizeof",
        "static",
        "struct",
        "switch",
        "true",
        "void",
        "while",
        # 8 Standard Library Identifiers
        "std",
        "vector",
        "string",
        "size_t",
        "unordered_map",
        "unordered_set",
        "set",
        "map",
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SplitError(Exception):
    """Base exception for dataset partitioning and split operations."""


class InfeasibleSplitConstraintsError(SplitError):
    """Raised when hard constraints for test or fold allocation cannot be satisfied."""


class CardinalityInfeasibilityError(InfeasibleSplitConstraintsError):
    """Raised when target partition or fold cardinalities cannot be satisfied."""


class StratificationDeficitWarning(UserWarning):
    """Warned when perfect stratification is mathematically impossible."""


STRATIFICATION_DEFICIT_WARNING = "STRATIFICATION_DEFICIT_WARNING"


class SplitValidationError(SplitError):
    """Raised when a proposed or frozen split manifest fails validation checks."""


class FreezeGateError(SplitError):
    """Raised when conditions for freezing a split manifest are not met."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AtomicUnit:
    """An atomic grouping of one or more families that cannot be divided across partitions."""

    unit_id: str
    family_ids: tuple[str, ...]
    weight: int
    rank: int
    classes: tuple[str, ...]
    rule_hard_count: int
    unknown_reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SplitProgramRecord:
    """Assignment record for a single program in the partitioned dataset."""

    program_id: str
    family_id: str
    partition: str
    fold: int | None
    expected_label: str
    tag: str
    unknown_reason: str | None
    git_blob_sha256: str | None
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return {
            "program_id": self.program_id,
            "family_id": self.family_id,
            "partition": self.partition,
            "fold": self.fold,
            "expected_label": self.expected_label,
            "tag": self.tag,
            "unknown_reason": self.unknown_reason,
            "git_blob_sha256": self.git_blob_sha256,
            "source_sha256": self.source_sha256,
        }


@dataclasses.dataclass
class SplitManifest:
    """Top-level dataset split manifest containing partitions and program records."""

    schema_version: str
    dataset_version: str
    created_at: str
    seed: int
    status: str
    git_commit: str | None
    base_dataset_commit: str | None = None
    proposal_commit: str | None = None
    git_dirty: bool = False
    partitions: dict[str, Any] = dataclasses.field(default_factory=dict)
    cv_folds: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    programs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    quarantined_families: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "seed": self.seed,
            "status": self.status,
            "git_commit": self.git_commit,
            "base_dataset_commit": self.base_dataset_commit,
            "proposal_commit": self.proposal_commit,
            "git_dirty": self.git_dirty,
            "partitions": self.partitions,
            "cv_folds": self.cv_folds,
            "programs": self.programs,
            "quarantined_families": self.quarantined_families,
        }

    def to_json(self) -> str:
        """Produce deterministic canonical JSON string."""
        return canonical_json_dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Canonical JSON and Utility Helpers
# ---------------------------------------------------------------------------


def canonical_json_dumps(data: Any) -> str:
    """Format an object as deterministic canonical JSON with sorted keys and trailing newline."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def compute_unit_rank(seed: int, unit_key: str) -> int:
    """Compute deterministic 64-bit unsigned integer rank for an atomic unit.

    Formula: int(sha256(f"{seed}:{unit_key}")[:16], 16)
    """
    key = f"{seed}:{unit_key}".encode("utf-8")
    hex_prefix = hashlib.sha256(key).hexdigest()[:16]
    return int(hex_prefix, 16)


# ---------------------------------------------------------------------------
# Atomic Unit Loading
# ---------------------------------------------------------------------------


def _parse_split_groups_yaml(path: Path) -> list[list[str]]:
    """Parse split groups YAML using built-in scanner without PyYAML dependency."""
    lines = path.read_text(encoding="utf-8").splitlines()
    groups: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("groups:"):
            continue
        if "[" in stripped and "]" in stripped:
            inside = stripped[stripped.index("[") + 1 : stripped.index("]")]
            fams = [item.strip().strip("\"'") for item in inside.split(",") if item.strip()]
            if fams:
                groups.append(fams)
        elif stripped.startswith("- "):
            val = stripped[2:].strip().strip("\"'")
            if val:
                groups.append([val])
    return groups


def load_atomic_units(
    families: list[FamilyMetadata],
    split_groups_path: Path | str | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
    quarantined: frozenset[str] = QUARANTINED_FAMILIES,
) -> list[AtomicUnit]:
    """Assemble atomic units from family metadata and optional split groups YAML.

    Quarantined families are excluded from unit creation.
    If no split groups file exists, every non-quarantined family becomes a singleton atomic unit.
    """
    family_map = {f.family_id: f for f in families if f.family_id not in quarantined}
    assigned_families: set[str] = set()
    units: list[AtomicUnit] = []

    groups_data: list[list[str]] = []
    if split_groups_path is not None:
        p = Path(split_groups_path)
        if p.exists() and p.is_file():
            groups_data = _parse_split_groups_yaml(p)

    # Process explicit groups
    for grp in groups_data:
        active_fams = [fid for fid in grp if fid in family_map]
        if not active_fams:
            continue
        for fid in active_fams:
            if fid in assigned_families:
                raise SplitError(f"Family {fid} is assigned to multiple split groups.")
            assigned_families.add(fid)

        active_fams.sort()
        unit_key = ",".join(active_fams)
        f_objs = [family_map[fid] for fid in active_fams]
        classes = tuple(f.label for f in f_objs)
        rh_count = sum(1 for f in f_objs if f.tag == "rule_hard")
        reasons = tuple(
            f.program_annotations.get(f.programs[0], {}).get("unknown_reason")
            for f in f_objs
            if f.label == "UNKNOWN" and f.programs
        )
        clean_reasons = tuple(r for r in reasons if r is not None)

        units.append(
            AtomicUnit(
                unit_id=f"group_{unit_key}",
                family_ids=tuple(active_fams),
                weight=len(active_fams),
                rank=compute_unit_rank(seed, unit_key),
                classes=classes,
                rule_hard_count=rh_count,
                unknown_reasons=clean_reasons,
            )
        )

    # Remaining families become singletons
    for fid in sorted(family_map.keys()):
        if fid in assigned_families:
            continue
        f_obj = family_map[fid]
        unit_key = fid
        reasons: list[str] = []
        if f_obj.label == "UNKNOWN":
            for pid in f_obj.programs:
                ann = f_obj.program_annotations.get(pid, {})
                r = ann.get("unknown_reason")
                if r and r not in reasons:
                    reasons.append(r)

        units.append(
            AtomicUnit(
                unit_id=f"fam_{fid}",
                family_ids=(fid,),
                weight=1,
                rank=compute_unit_rank(seed, unit_key),
                classes=(f_obj.label,),
                rule_hard_count=1 if f_obj.tag == "rule_hard" else 0,
                unknown_reasons=tuple(reasons),
            )
        )

    # Sort units deterministically by rank then unit_id
    units.sort(key=lambda u: (u.rank, u.unit_id))
    return units


# ---------------------------------------------------------------------------
# Test Set Optimizer (Branch-and-Bound with Exact DP Rank-Sum Lower Bound)
# ---------------------------------------------------------------------------

INF_RANK: int = 10**30


def compute_min_rank_sum_dp(remaining_units: list[AtomicUnit], max_weight: int) -> list[int]:
    """Compute 0-1 knapsack min-cost subset-sum table for exact weights 0..max_weight.

    dp[w] is the minimum rank sum to achieve exact total weight w using a subset of remaining_units.
    """
    dp = [INF_RANK] * (max_weight + 1)
    dp[0] = 0

    for u in remaining_units:
        w_u = u.weight
        r_u = u.rank
        for w in range(max_weight, w_u - 1, -1):
            if dp[w - w_u] != INF_RANK:
                cost = dp[w - w_u] + r_u
                if cost < dp[w]:
                    dp[w] = cost

    return dp


def optimize_test_partition(
    units: list[AtomicUnit],
    target_weight: int = TARGET_TEST_FAMILIES,
    locked_dev_families: frozenset[str] = LOCKED_DEV_FAMILIES,
) -> list[AtomicUnit]:
    """Find the lexicographically optimal test partition via branch-and-bound.

    Hard constraints:
    1. Total weight == target_weight (15).
    2. All 8 complexity classes present.
    3. Exactly 1 ON3 unit.
    4. Exactly 1 OEXP unit.
    5. Exactly 3 UNKNOWN units.
    6. No unit contains a family in locked_dev_families.
    7. rule_hard count <= 3 (which forces exactly 3 since all 3 UNKNOWN are rule_hard).

    Lexicographic objectives (in order of priority):
    1. Maximize distinct UNKNOWN reasons (up to 3).
    2. Minimize abs(rule_hard_count - 3) (0 is optimal).
    3. Minimize total unit rank sum.
    4. Minimize canonical sorted family-ID tuple.
    """
    # Filter candidates: exclude any unit containing locked families
    candidates = [u for u in units if not any(fid in locked_dev_families for fid in u.family_ids)]

    # Pre-sort candidates deterministically by rank then unit_id
    candidates.sort(key=lambda u: (u.rank, u.unit_id))
    n = len(candidates)

    # Precompute suffix min-rank-sum DP tables for admissible lower bounding
    dp_tables: list[list[int]] = []
    for i in range(n):
        dp_tables.append(compute_min_rank_sum_dp(candidates[i:], target_weight))
    dp_tables.append([0] + [INF_RANK] * target_weight)

    # Track best solution: (score, tuple_ids, list_of_units)
    # score: (distinct_unknown_reasons, -abs(rh - 3), -rank_sum)
    best_score: tuple[int, int, int] | None = None
    best_tuple: tuple[str, ...] | None = None
    best_solution: list[AtomicUnit] | None = None

    def search(
        idx: int,
        current_weight: int,
        current_rank_sum: int,
        current_classes: set[str],
        current_on3: int,
        current_oexp: int,
        current_unknown: int,
        current_rh: int,
        current_reasons: set[str],
        selected: list[AtomicUnit],
    ) -> None:
        nonlocal best_score, best_tuple, best_solution

        # 1. Check weight feasibility using DP table
        remaining_weight = target_weight - current_weight
        if remaining_weight < 0:
            return

        min_additional_rank = dp_tables[idx][remaining_weight]
        if min_additional_rank >= INF_RANK:
            return

        rank_lower_bound = current_rank_sum + min_additional_rank

        # 2. Hard constraint pruning
        if current_on3 > TARGET_TEST_ON3:
            return
        if current_oexp > TARGET_TEST_OEXP:
            return
        if current_unknown > TARGET_TEST_UNKNOWN:
            return
        if current_rh > TARGET_TEST_RULE_HARD:
            return

        # Check remaining capacity for exact class requirements
        remaining_units_count = n - idx
        if current_on3 + remaining_units_count < TARGET_TEST_ON3:
            return
        if current_oexp + remaining_units_count < TARGET_TEST_OEXP:
            return
        if current_unknown + remaining_units_count < TARGET_TEST_UNKNOWN:
            return

        # 3. Base case: target weight reached
        if current_weight == target_weight:
            if current_on3 != TARGET_TEST_ON3:
                return
            if current_oexp != TARGET_TEST_OEXP:
                return
            if current_unknown != TARGET_TEST_UNKNOWN:
                return
            if len(current_classes) < len(ALLOWED_LABELS):
                return
            if current_rh > TARGET_TEST_RULE_HARD:
                return

            distinct_reasons = len(current_reasons)
            rh_dev = -abs(current_rh - TARGET_TEST_RULE_HARD)
            neg_rank_sum = -current_rank_sum
            candidate_score = (distinct_reasons, rh_dev, neg_rank_sum)

            all_fids: list[str] = []
            for u in selected:
                all_fids.extend(u.family_ids)
            cand_tuple = tuple(sorted(all_fids))

            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_tuple = cand_tuple
                best_solution = list(selected)
            elif (
                candidate_score == best_score and best_tuple is not None and cand_tuple < best_tuple
            ):
                best_tuple = cand_tuple
                best_solution = list(selected)
            return

        if idx >= n:
            return

        # 4. Lexicographic bounding and pruning
        if best_score is not None:
            # Upper bound on distinct UNKNOWN reasons
            avail_reasons: set[str] = set()
            for u in candidates[idx:]:
                avail_reasons.update(u.unknown_reasons)
            max_possible_reasons = min(
                TARGET_TEST_UNKNOWN,
                len(current_reasons | avail_reasons),
            )

            # If upper bound on reasons is strictly worse than incumbent, prune
            if max_possible_reasons < best_score[0]:
                return

            if max_possible_reasons == best_score[0]:
                # Best possible rh_dev is 0
                max_possible_rh_dev = 0
                if max_possible_rh_dev < best_score[1]:
                    return

                if max_possible_rh_dev == best_score[1]:
                    # Admissible rank sum check
                    incumbent_rank_sum = -best_score[2]
                    if rank_lower_bound > incumbent_rank_sum:
                        return

        # 5. Branching: Try include candidates[idx] first, then exclude
        u = candidates[idx]

        new_on3 = current_on3 + u.classes.count("ON3")
        new_oexp = current_oexp + u.classes.count("OEXP")
        new_unknown = current_unknown + u.classes.count("UNKNOWN")
        new_rh = current_rh + u.rule_hard_count
        new_weight = current_weight + u.weight

        if (
            new_weight <= target_weight
            and new_on3 <= TARGET_TEST_ON3
            and new_oexp <= TARGET_TEST_OEXP
            and new_unknown <= TARGET_TEST_UNKNOWN
            and new_rh <= TARGET_TEST_RULE_HARD
        ):
            new_classes = current_classes | set(u.classes)
            new_reasons = current_reasons | set(u.unknown_reasons)
            selected.append(u)
            search(
                idx + 1,
                new_weight,
                current_rank_sum + u.rank,
                new_classes,
                new_on3,
                new_oexp,
                new_unknown,
                new_rh,
                new_reasons,
                selected,
            )
            selected.pop()

        # Branch: Exclude candidates[idx]
        search(
            idx + 1,
            current_weight,
            current_rank_sum,
            current_classes,
            current_on3,
            current_oexp,
            current_unknown,
            current_rh,
            current_reasons,
            selected,
        )

    # Initial search invocation
    search(
        idx=0,
        current_weight=0,
        current_rank_sum=0,
        current_classes=set(),
        current_on3=0,
        current_oexp=0,
        current_unknown=0,
        current_rh=0,
        current_reasons=set(),
        selected=[],
    )

    if best_solution is None:
        raise InfeasibleSplitConstraintsError(
            f"Failed to find any feasible test set of weight {target_weight}."
        )

    return sorted(best_solution, key=lambda u: u.family_ids[0])


# ---------------------------------------------------------------------------
# Cross-Validation Fold Allocator inside Development Partition
# ---------------------------------------------------------------------------


def _finish_singletons(
    singletons: list[AtomicUnit],
    folds_state: list[list[AtomicUnit]],
    caps_state: list[int],
) -> tuple[list[list[AtomicUnit]], str] | None:
    """Deterministically assign weight-1 singletons to satisfy missing classes and capacities."""
    if sum(caps_state) != sum(u.weight for u in singletons):
        return None
    if any(u.weight != 1 for u in singletons):
        return None

    f_res = [list(f) for f in folds_state]
    caps = list(caps_state)

    by_class: dict[str, list[AtomicUnit]] = {}
    for u in singletons:
        c = u.classes[0] if u.classes else ""
        by_class.setdefault(c, []).append(u)

    for c in by_class:
        by_class[c].sort(key=lambda u: (u.rank, u.unit_id))

    # Phase 1: Give missing classes to folds that need them
    for c in sorted(ALLOWED_LABELS):
        if c not in by_class or not by_class[c]:
            continue
        for f_idx in range(len(f_res)):
            if caps[f_idx] <= 0:
                continue
            f_classes = {cls for u in f_res[f_idx] for cls in u.classes}
            if c not in f_classes and by_class[c]:
                u = by_class[c].pop(0)
                f_res[f_idx].append(u)
                caps[f_idx] -= 1

    # Phase 2: Place any remaining singletons into folds with remaining capacity
    leftovers: list[AtomicUnit] = []
    for c_units in by_class.values():
        leftovers.extend(c_units)
    leftovers.sort(key=lambda u: (u.rank, u.unit_id))

    for u in leftovers:
        placed = False
        for f_idx in range(len(f_res)):
            if caps[f_idx] > 0:
                f_res[f_idx].append(u)
                caps[f_idx] -= 1
                placed = True
                break
        if not placed:
            return None

    if any(c != 0 for c in caps):
        return None

    all_covered = all(
        len({cls for u in f for cls in u.classes}) == len(ALLOWED_LABELS) for f in f_res
    )
    case_str = "Case A" if all_covered else "Case B"
    return f_res, case_str


def _backtrack_pack_folds(
    bundle_assignments: list[list[list[AtomicUnit]]],
    all_dev_units: list[AtomicUnit],
    fold_sizes: tuple[int, int, int],
) -> tuple[list[list[AtomicUnit]], str] | None:
    """Exhaustive search over bundle assignments and multi-family units from clean capacities.

    Guarantees:
    1. Preserves atomic units (never split across folds).
    2. Exact fold capacities (18, 18, 17).
    3. Exactly 1 ON3 and 1 OEXP family per fold (via bundle assignments).
    4. Explores alternative packings to maximize class coverage (Case A if achievable, else Case B).
    5. Returns None only when exhaustive search proves no valid packing exists.
    """
    best_case_b: tuple[list[list[AtomicUnit]], str] | None = None

    for b_assignment in bundle_assignments:
        initial_folds = [list(f) for f in b_assignment]
        caps = [
            fold_sizes[i] - sum(u.weight for u in initial_folds[i])
            for i in range(len(fold_sizes))
        ]
        if any(c < 0 for c in caps):
            continue

        assigned_ids = {u.unit_id for f in initial_folds for u in f}
        unplaced = [u for u in all_dev_units if u.unit_id not in assigned_ids]
        if sum(u.weight for u in unplaced) != sum(caps):
            continue

        # Sort multi-family units (>1) first descending by weight, then singletons
        multi_units = [u for u in unplaced if u.weight > 1]
        singletons = [u for u in unplaced if u.weight == 1]
        multi_units.sort(key=lambda u: (u.weight, u.rank, u.unit_id), reverse=True)
        singletons.sort(key=lambda u: (u.rank, u.unit_id))

        ordered_unplaced = multi_units + singletons

        def search(
            idx: int,
            folds_state: list[list[AtomicUnit]],
            caps_state: list[int],
        ) -> tuple[list[list[AtomicUnit]], str] | None:
            nonlocal best_case_b

            # If all remaining units are singletons (or all placed)
            if idx == len(ordered_unplaced) or ordered_unplaced[idx].weight == 1:
                rem_singletons = ordered_unplaced[idx:]
                res = _finish_singletons(rem_singletons, folds_state, caps_state)
                if res is not None:
                    sol_folds, sol_case = res
                    if sol_case == "Case A":
                        return sol_folds, sol_case
                    elif best_case_b is None:
                        best_case_b = (sol_folds, sol_case)
                return None

            u = ordered_unplaced[idx]
            rem_weight = sum(item.weight for item in ordered_unplaced[idx:])
            if rem_weight != sum(caps_state):
                return None

            for f_idx in range(len(caps_state)):
                if caps_state[f_idx] >= u.weight:
                    caps_state[f_idx] -= u.weight
                    folds_state[f_idx].append(u)
                    sub_res = search(idx + 1, folds_state, caps_state)
                    if sub_res is not None and sub_res[1] == "Case A":
                        return sub_res
                    folds_state[f_idx].pop()
                    caps_state[f_idx] += u.weight

            return None

        case_a_res = search(0, initial_folds, caps)
        if case_a_res is not None:
            return case_a_res

    if best_case_b is not None:
        return best_case_b

    return None


def format_stratification_deficit_warning(
    folds: list[list[AtomicUnit]],
    fold_sizes: tuple[int, ...] = CV_FOLD_SIZES,
) -> tuple[str, int, dict[int, list[str]], dict[int, dict[str, int]]]:
    """Calculate and format detailed stratification deficits across cross-validation folds.

    Returns:
        (warning_message, total_coverage_deficit, missing_per_fold, counts_per_fold)
    """
    missing_per_fold: dict[int, list[str]] = {}
    counts_per_fold: dict[int, dict[str, int]] = {}
    total_coverage_deficit = 0

    for f_idx, f in enumerate(folds):
        present_classes = {c for u in f for c in u.classes}
        missing = sorted(c for c in ALLOWED_LABELS if c not in present_classes)
        missing_per_fold[f_idx] = missing
        total_coverage_deficit += len(missing)
        counts_per_fold[f_idx] = {
            c: sum(u.classes.count(c) for u in f) for c in sorted(ALLOWED_LABELS)
        }

    lines = [
        f"{STRATIFICATION_DEFICIT_WARNING}: Fold allocation could not achieve "
        f"complete 8-class coverage (total_coverage_deficit={total_coverage_deficit}).",
        "Note: total_coverage_deficit is the observed deficit of the deterministic allocator "
        "(not a mathematically proven minimal deficit).",
    ]
    for f_idx in range(len(folds)):
        actual_weight = sum(u.weight for u in folds[f_idx])
        target_cap = fold_sizes[f_idx] if f_idx < len(fold_sizes) else actual_weight
        lines.append(
            f"  Fold {f_idx} (target_capacity={target_cap}, actual_weight={actual_weight}): "
            f"missing={missing_per_fold[f_idx]}, counts={counts_per_fold[f_idx]}"
        )

    warning_msg = "\n".join(lines)
    return warning_msg, total_coverage_deficit, missing_per_fold, counts_per_fold


def allocate_development_folds(
    dev_units: list[AtomicUnit],
    fold_sizes: tuple[int, int, int] = CV_FOLD_SIZES,
) -> tuple[list[list[AtomicUnit]], str]:
    """Partition development units into 3 cross-validation folds.

    Hard constraints:
    1. Fold weights match fold_sizes (18, 18, 17).
    2. Exactly 1 ON3 family in each fold (counted across member families).
    3. Exactly 1 OEXP family in each fold (counted across member families).
    4. Atomic units remain indivisible.

    Feasibility Classification:
    - Case A: Exact sizes feasible and all 8 classes present in every fold.
    - Case B: Exact sizes feasible, but 8-class coverage impossible in all folds.
    """
    total_dev_weight = sum(u.weight for u in dev_units)
    if total_dev_weight != sum(fold_sizes):
        raise CardinalityInfeasibilityError(
            f"Total dev units weight ({total_dev_weight}) != sum of fold sizes ({sum(fold_sizes)})."
        )

    # Validate family-level ON3 and OEXP counts across all atomic units
    total_on3 = sum(u.classes.count("ON3") for u in dev_units)
    total_oexp = sum(u.classes.count("OEXP") for u in dev_units)
    if total_on3 != NUM_CV_FOLDS or total_oexp != NUM_CV_FOLDS:
        raise InfeasibleSplitConstraintsError(
            f"Expected exactly {NUM_CV_FOLDS} ON3 and {NUM_CV_FOLDS} OEXP families in dev, "
            f"found {total_on3} ON3 and {total_oexp} OEXP."
        )

    # Indivisible unit check: Any unit containing >1 ON3 or >1 OEXP cannot satisfy
    # exactly 1 family per fold across separate folds.
    for u in dev_units:
        if u.classes.count("ON3") > 1:
            raise InfeasibleSplitConstraintsError(
                f"Atomic unit '{u.unit_id}' contains {u.classes.count('ON3')} ON3 families; "
                "cannot divide atomic unit to achieve exactly 1 ON3 family per fold."
            )
        if u.classes.count("OEXP") > 1:
            raise InfeasibleSplitConstraintsError(
                f"Atomic unit '{u.unit_id}' contains {u.classes.count('OEXP')} OEXP families; "
                "cannot divide atomic unit to achieve exactly 1 OEXP family per fold."
            )

    # Separate joint units (1 ON3 and 1 OEXP) and pure units
    joint_units = [
        u for u in dev_units if u.classes.count("ON3") == 1 and u.classes.count("OEXP") == 1
    ]
    pure_on3 = [
        u for u in dev_units if u.classes.count("ON3") == 1 and u.classes.count("OEXP") == 0
    ]
    pure_oexp = [
        u for u in dev_units if u.classes.count("ON3") == 0 and u.classes.count("OEXP") == 1
    ]

    joint_units.sort(key=lambda u: (u.rank, u.unit_id))
    pure_on3.sort(key=lambda u: (u.rank, u.unit_id))
    pure_oexp.sort(key=lambda u: (u.rank, u.unit_id))

    k = len(joint_units)
    bundles: list[list[AtomicUnit]] = []
    for i in range(k):
        bundles.append([joint_units[i]])
    for i in range(NUM_CV_FOLDS - k):
        bundles.append([pure_on3[i], pure_oexp[i]])

    # Assign bundles to folds respecting fold capacities
    valid_assignments: list[list[list[AtomicUnit]]] = []
    for perm in itertools.permutations(bundles):
        if all(sum(u.weight for u in perm[f]) <= fold_sizes[f] for f in range(NUM_CV_FOLDS)):
            valid_assignments.append([list(perm[f]) for f in range(NUM_CV_FOLDS)])

    if not valid_assignments:
        raise InfeasibleSplitConstraintsError(
            "Cannot assign ON3/OEXP units to folds without violating fold capacities."
        )

    def assignment_key(assignment: list[list[AtomicUnit]]) -> Any:
        util = max(
            sum(u.weight for u in assignment[f]) / fold_sizes[f] for f in range(NUM_CV_FOLDS)
        )
        ranks = sum(u.rank for f in assignment for u in f)
        fids = tuple(fid for f in assignment for u in f for fid in u.family_ids)
        return (util, ranks, fids)

    valid_assignments.sort(key=assignment_key)
    folds = valid_assignments[0]

    assigned_unit_ids = {u.unit_id for f in folds for u in f}
    remaining_units = [u for u in dev_units if u.unit_id not in assigned_unit_ids]

    # 2. Assign UNKNOWN units: balance unknown reasons across folds
    unk_units = [u for u in remaining_units if "UNKNOWN" in u.classes]
    by_reason: dict[str, list[AtomicUnit]] = {}
    for u in unk_units:
        r = u.unknown_reasons[0] if u.unknown_reasons else "none"
        by_reason.setdefault(r, []).append(u)

    for r in sorted(by_reason.keys()):
        for u in sorted(by_reason[r], key=lambda x: (x.rank, x.unit_id)):
            valid_folds = [
                f
                for f in range(NUM_CV_FOLDS)
                if sum(x.weight for x in folds[f]) + u.weight <= fold_sizes[f]
            ]
            if not valid_folds:
                best_f = min(range(NUM_CV_FOLDS), key=lambda f: sum(x.weight for x in folds[f]))
            else:
                best_f = min(
                    valid_folds,
                    key=lambda f: (
                        sum(1 for x in folds[f] if "UNKNOWN" in x.classes),
                        sum(x.weight for x in folds[f]) / fold_sizes[f],
                        sum(x.rank for x in folds[f]),
                    ),
                )
            folds[best_f].append(u)

    # 3. Remaining classes: O1, ONLOGN, OLOGN, ON2, ON
    assigned_unit_ids = {u.unit_id for f in folds for u in f}
    other_classes = ["O1", "ONLOGN", "OLOGN", "ON2", "ON"]
    by_class: dict[str, list[AtomicUnit]] = {
        c: [u for u in dev_units if u.unit_id not in assigned_unit_ids and c in u.classes]
        for c in other_classes
    }

    # Step 3a: Ensure at least 1 unit of each remaining class in each fold if possible
    for c in other_classes:
        c_units = sorted(by_class[c], key=lambda x: (x.rank, x.unit_id))
        for f in range(NUM_CV_FOLDS):
            if c_units and sum(x.weight for x in folds[f]) + c_units[0].weight <= fold_sizes[f]:
                folds[f].append(c_units.pop(0))
        by_class[c] = c_units

    # Step 3b: Distribute remaining unplaced units
    assigned_unit_ids = {u.unit_id for f in folds for u in f}
    unplaced_units = [u for u in dev_units if u.unit_id not in assigned_unit_ids]
    unplaced_units.sort(key=lambda u: (u.weight, u.rank, u.unit_id), reverse=True)

    for u in unplaced_units:
        c = u.classes[0] if u.classes else ""
        valid_folds = [
            f
            for f in range(NUM_CV_FOLDS)
            if sum(x.weight for x in folds[f]) + u.weight <= fold_sizes[f]
        ]
        if not valid_folds:
            best_f = min(range(NUM_CV_FOLDS), key=lambda f: sum(x.weight for x in folds[f]))
        else:
            best_f = min(
                valid_folds,
                key=lambda f: (
                    sum(1 for x in folds[f] if c in x.classes),
                    sum(x.weight for x in folds[f]) / fold_sizes[f],
                    sum(x.rank for x in folds[f]),
                ),
            )
        folds[best_f].append(u)

    # Check if greedy achieved exact sizes, placed all units, and achieved Case A
    actual_sizes = tuple(sum(u.weight for u in f) for f in folds)
    assigned_count = len({u.unit_id for f in folds for u in f})
    is_greedy_case_a = (
        actual_sizes == fold_sizes
        and assigned_count == len(dev_units)
        and all(len({c for u in f for c in u.classes}) == len(ALLOWED_LABELS) for f in folds)
    )

    if is_greedy_case_a:
        case_classification = "Case A"
    else:
        # Fallback to exact exhaustive backtracking from clean bundle assignments
        backtrack_res = _backtrack_pack_folds(valid_assignments, dev_units, fold_sizes)
        if backtrack_res is not None:
            folds, case_classification = backtrack_res
        elif actual_sizes == fold_sizes and assigned_count == len(dev_units):
            # Greedy found an exact packing that is Case B
            case_classification = "Case B"
        else:
            raise CardinalityInfeasibilityError(
                f"Could not satisfy exact fold capacities {fold_sizes} with indivisible "
                "atomic units: exhaustive search proved no valid assignment exists."
            )

    # Final independent verification of all hard constraints
    final_sizes = tuple(sum(u.weight for u in f) for f in folds)
    if final_sizes != fold_sizes:
        raise CardinalityInfeasibilityError(
            f"Could not satisfy exact fold capacities with indivisible units: "
            f"actual {final_sizes} != expected {fold_sizes}."
        )

    # Unit uniqueness and completeness
    all_assigned_ids = [u.unit_id for f in folds for u in f]
    expected_ids = [u.unit_id for u in dev_units]
    if len(all_assigned_ids) != len(expected_ids) or set(all_assigned_ids) != set(expected_ids):
        raise InfeasibleSplitConstraintsError(
            "Unit uniqueness or completeness violated in fold allocation."
        )

    # Enforce exactly 1 ON3 and 1 OEXP family in every fold
    for f_idx, f in enumerate(folds):
        on3_in_fold = sum(u.classes.count("ON3") for u in f)
        oexp_in_fold = sum(u.classes.count("OEXP") for u in f)
        if on3_in_fold != 1 or oexp_in_fold != 1:
            raise InfeasibleSplitConstraintsError(
                f"Fold {f_idx} constraint violated: {on3_in_fold} ON3 and "
                f"{oexp_in_fold} OEXP families (expected 1 each)."
            )

    # Final case classification and warning
    all_covered = all(
        len({c for u in f for c in u.classes}) == len(ALLOWED_LABELS) for f in folds
    )
    if all_covered:
        case_classification = "Case A"
    else:
        case_classification = "Case B"
        warning_msg, _, _, _ = format_stratification_deficit_warning(folds, fold_sizes)
        warnings.warn(
            warning_msg,
            StratificationDeficitWarning,
            stacklevel=2,
        )

    # Symmetry breaking for equal-sized folds (Fold 0 and Fold 1 both have capacity 18)
    rs0 = sum(u.rank for u in folds[0])
    rs1 = sum(u.rank for u in folds[1])
    t0 = tuple(sorted(fid for u in folds[0] for fid in u.family_ids))
    t1 = tuple(sorted(fid for u in folds[1] for fid in u.family_ids))

    if rs0 > rs1 or (rs0 == rs1 and t0 > t1):
        folds[0], folds[1] = folds[1], folds[0]

    # Sort each fold deterministically by family ID
    sorted_folds = [sorted(f, key=lambda u: u.family_ids[0]) for f in folds]
    return sorted_folds, case_classification


# ---------------------------------------------------------------------------
# Lexical Token Normalization and Similarity Analysis
# ---------------------------------------------------------------------------


def tokenize_cpp_lexical(source: str) -> list[str]:
    """Perform deterministic character-level lexical tokenization of C++ source.

    Strips C (/* */) and C++ (//) comments and string literals, then tokenizes
    alphanumeric identifiers and punctuation.
    """
    cleaned: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        # Line comment
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            continue
        # Block comment
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i + 1 < n:
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            continue
        # String literal
        if ch == '"':
            i += 1
            while i < n:
                c = source[i]
                i += 1
                if c == "\\" and i < n:
                    i += 1
                elif c == '"':
                    break
            continue
        # Character literal
        if ch == "'":
            i += 1
            while i < n:
                c = source[i]
                i += 1
                if c == "\\" and i < n:
                    i += 1
                elif c == "'":
                    break
            continue

        cleaned.append(ch)
        i += 1

    clean_text = "".join(cleaned)
    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|&&|\|\||\+\+|--|::|[^\s\w]", clean_text
    )
    return tokens


def normalize_tokens(tokens: list[str]) -> list[str]:
    """Normalize identifiers not in the 37-token vocabulary to a generic 'ID' placeholder."""
    normalized: list[str] = []
    for tok in tokens:
        if tok.isidentifier() and tok not in NORMALIZATION_VOCABULARY:
            normalized.append("ID")
        else:
            normalized.append(tok)
    return normalized


def compute_token_shingles(tokens: list[str], k: int = 3) -> set[tuple[str, ...]]:
    """Compute k-token shingles (k-grams)."""
    if len(tokens) < k:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard_similarity(s1: set[Any], s2: set[Any]) -> float:
    """Compute Jaccard similarity coefficient between two sets."""
    if not s1 and not s2:
        return 1.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    if union == 0:
        return 0.0
    return intersection / union


def verify_cross_partition_similarity(
    records: list[SplitProgramRecord],
    program_sources: dict[str, str],
    threshold: float = 0.85,
) -> list[tuple[str, str, float, float]]:
    """Verify that no cross-partition program pairs exceed similarity threshold.

    Returns list of offending pairs: (dev_prog_id, test_prog_id, primary_jaccard, norm_jaccard).
    """
    dev_progs = [r for r in records if r.partition == "development"]
    test_progs = [r for r in records if r.partition == "test"]

    dev_shingles = {}
    dev_norm_shingles = {}
    for r in dev_progs:
        src = program_sources[r.program_id]
        toks = tokenize_cpp_lexical(src)
        norm_toks = normalize_tokens(toks)
        dev_shingles[r.program_id] = compute_token_shingles(toks, 3)
        dev_norm_shingles[r.program_id] = compute_token_shingles(norm_toks, 3)

    test_shingles = {}
    test_norm_shingles = {}
    for r in test_progs:
        src = program_sources[r.program_id]
        toks = tokenize_cpp_lexical(src)
        norm_toks = normalize_tokens(toks)
        test_shingles[r.program_id] = compute_token_shingles(toks, 3)
        test_norm_shingles[r.program_id] = compute_token_shingles(norm_toks, 3)

    violations = []
    for dp in dev_progs:
        s1 = dev_shingles[dp.program_id]
        s1_norm = dev_norm_shingles[dp.program_id]
        for tp in test_progs:
            s2 = test_shingles[tp.program_id]
            s2_norm = test_norm_shingles[tp.program_id]
            j_prim = jaccard_similarity(s1, s2)
            j_norm = jaccard_similarity(s1_norm, s2_norm)
            if j_prim > threshold or j_norm > threshold:
                violations.append((dp.program_id, tp.program_id, j_prim, j_norm))

    return violations


# ---------------------------------------------------------------------------
# Git Metadata Inspection
# ---------------------------------------------------------------------------


def get_git_info(repo_root: Path | str = ".") -> tuple[str | None, bool]:
    """Retrieve HEAD commit SHA and working tree dirty state from git CLI."""
    try:
        res_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res_commit.returncode != 0:
            return None, True
        commit = res_commit.stdout.strip()

        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        is_dirty = bool(res_status.stdout.strip())
        return commit, is_dirty
    except Exception:
        return None, True


def get_git_blob_sha256(file_path: Path | str, repo_root: Path | str = ".") -> str | None:
    """Compute SHA-256 of git blob content for a given file at HEAD."""
    try:
        rel_path = Path(file_path).relative_to(repo_root).as_posix()
        res = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=str(repo_root),
            capture_output=True,
            check=False,
        )
        if res.returncode == 0:
            return hashlib.sha256(res.stdout).hexdigest()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Split Proposal and Freeze Lifecycle
# ---------------------------------------------------------------------------


def generate_proposal_manifest(
    dataset_root: Path | str = "datasets",
    split_groups_path: Path | str | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
    repo_root: Path | str = ".",
) -> SplitManifest:
    """Generate a proposal SplitManifest using exact optimization and fold allocation.

    Does not modify any repository files. Feature extraction is deferred (no persistent cache).
    """
    records, families, errors = load_dataset(dataset_root)
    if errors:
        raise SplitError(f"Dataset integrity errors: {errors}")

    # Build atomic units
    atomic_units = load_atomic_units(
        families=families,
        split_groups_path=split_groups_path,
        seed=seed,
        quarantined=QUARANTINED_FAMILIES,
    )

    # 1. Optimize test partition
    test_units = optimize_test_partition(atomic_units, target_weight=TARGET_TEST_FAMILIES)
    test_unit_ids = {u.unit_id for u in test_units}
    test_family_ids = {fid for u in test_units for fid in u.family_ids}

    # 2. Allocate development folds
    dev_units = [u for u in atomic_units if u.unit_id not in test_unit_ids]
    cv_folds, _ = allocate_development_folds(dev_units, fold_sizes=CV_FOLD_SIZES)

    family_to_fold: dict[str, int] = {}
    for f_idx, fold in enumerate(cv_folds):
        for u in fold:
            for fid in u.family_ids:
                family_to_fold[fid] = f_idx

    # 3. Assemble program records
    family_map = {f.family_id: f for f in families}
    prog_records: list[SplitProgramRecord] = []
    quarantined_fams = sorted(list(QUARANTINED_FAMILIES))

    dataset_path = Path(dataset_root)
    programs_dir = dataset_path / "programs"

    for r in records:
        if r.family_id in QUARANTINED_FAMILIES:
            continue
        fam = family_map[r.family_id]
        if r.family_id in test_family_ids:
            partition = "test"
            fold = None
        else:
            partition = "development"
            fold = family_to_fold[r.family_id]

        file_path = programs_dir / r.family_id / f"{r.program_id}.cpp"
        blob_sha = get_git_blob_sha256(file_path, repo_root=repo_root)

        prog_records.append(
            SplitProgramRecord(
                program_id=r.program_id,
                family_id=r.family_id,
                partition=partition,
                fold=fold,
                expected_label=r.expected_label,
                tag=fam.tag,
                unknown_reason=r.unknown_reason,
                git_blob_sha256=blob_sha,
                source_sha256=r.source_sha256,
            )
        )

    # Sort programs deterministically by program_id
    prog_records.sort(key=lambda p: p.program_id)

    # Partition summary statistics
    partitions_summary: dict[str, Any] = {}
    for part in ("development", "test"):
        p_recs = [p for p in prog_records if p.partition == part]
        p_fams = sorted(list({p.family_id for p in p_recs}))
        classes: dict[str, int] = {}
        for c in sorted(ALLOWED_LABELS):
            count = sum(1 for p in p_recs if p.expected_label == c)
            if count > 0:
                classes[c] = count
        tags: dict[str, int] = {}
        for t in sorted(list({fam.tag for fid, fam in family_map.items() if fid in p_fams})):
            tags[t] = sum(1 for fid in p_fams if family_map[fid].tag == t)
        reasons: dict[str, int] = {}
        for p in p_recs:
            if p.unknown_reason:
                reasons[p.unknown_reason] = reasons.get(p.unknown_reason, 0) + 1

        partitions_summary[part] = {
            "family_count": len(p_fams),
            "program_count": len(p_recs),
            "families": p_fams,
            "class_distribution": classes,
            "tag_distribution": tags,
            "unknown_reasons": reasons,
        }

    # CV Folds summary statistics
    cv_summary: list[dict[str, Any]] = []
    for f_idx in range(NUM_CV_FOLDS):
        f_recs = [p for p in prog_records if p.fold == f_idx]
        f_fams = sorted(list({p.family_id for p in f_recs}))
        classes = {}
        for c in sorted(ALLOWED_LABELS):
            count = sum(1 for p in f_recs if p.expected_label == c)
            if count > 0:
                classes[c] = count
        tags = {}
        for t in sorted(list({fam.tag for fid, fam in family_map.items() if fid in f_fams})):
            tags[t] = sum(1 for fid in f_fams if family_map[fid].tag == t)
        reasons = {}
        for p in f_recs:
            if p.unknown_reason:
                reasons[p.unknown_reason] = reasons.get(p.unknown_reason, 0) + 1

        cv_summary.append(
            {
                "fold": f_idx,
                "family_count": len(f_fams),
                "program_count": len(f_recs),
                "families": f_fams,
                "class_distribution": classes,
                "tag_distribution": tags,
                "unknown_reasons": reasons,
            }
        )

    git_commit, git_dirty = get_git_info(repo_root)

    return SplitManifest(
        schema_version=SPLIT_SCHEMA_VERSION,
        dataset_version=SPLIT_DATASET_VERSION,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed=seed,
        status="proposed",
        git_commit=git_commit,
        base_dataset_commit=git_commit,
        proposal_commit=None,
        git_dirty=git_dirty,
        partitions=partitions_summary,
        cv_folds=cv_summary,
        programs=[p.to_dict() for p in prog_records],
        quarantined_families=quarantined_fams,
    )


def validate_manifest(
    manifest: SplitManifest | dict[str, Any] | str | Path,
    dataset_root: Path | str = "datasets",
) -> list[str]:
    """Validate a split manifest against all integrity, partition, and coverage constraints.

    Returns list of error strings (empty if validation passed).
    """
    if isinstance(manifest, (str, Path)):
        p = Path(manifest)
        if not p.exists():
            return [f"Manifest file not found: {p}"]
        data = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(manifest, SplitManifest):
        data = manifest.to_dict()
    else:
        data = manifest

    errors: list[str] = []

    # 1. Schema fields
    for field in (
        "schema_version",
        "dataset_version",
        "seed",
        "status",
        "partitions",
        "cv_folds",
        "programs",
        "quarantined_families",
    ):
        if field not in data:
            errors.append(f"Missing required manifest field: '{field}'")

    if errors:
        return errors

    # 2. Load dataset to cross-check records
    records, families, ds_errors = load_dataset(dataset_root)
    if ds_errors:
        errors.extend(ds_errors)
        return errors

    family_map = {f.family_id: f for f in families}
    record_map = {r.program_id: r for r in records}

    # 3. Check partition definitions
    parts = data.get("partitions", {})
    if set(parts.keys()) != {"development", "test"}:
        errors.append(
            f"Partitions must be exactly {{'development', 'test'}}, found {set(parts.keys())}"
        )

    dev_info = parts.get("development", {})
    test_info = parts.get("test", {})

    dev_fams = set(dev_info.get("families", []))
    test_fams = set(test_info.get("families", []))

    # Test cardinality and dev cardinality
    if len(test_fams) != TARGET_TEST_FAMILIES:
        errors.append(
            f"Test partition must have {TARGET_TEST_FAMILIES} families, found {len(test_fams)}"
        )

    if len(dev_fams) != TARGET_DEV_FAMILIES:
        errors.append(
            f"Dev partition must have {TARGET_DEV_FAMILIES} families, found {len(dev_fams)}"
        )

    # Overlap check
    overlap = dev_fams & test_fams
    if overlap:
        errors.append(f"Family leakage detected between development and test: {overlap}")

    # Locked families check
    locked_in_test = test_fams & LOCKED_DEV_FAMILIES
    if locked_in_test:
        errors.append(f"Locked development families present in test partition: {locked_in_test}")

    # Check test class coverage
    test_classes = set(test_info.get("class_distribution", {}).keys())
    missing_test_classes = ALLOWED_LABELS - test_classes
    if missing_test_classes:
        errors.append(f"Test partition missing complexity classes: {missing_test_classes}")

    test_fam_labels = [family_map[fid].label for fid in test_fams if fid in family_map]
    if test_fam_labels.count("ON3") != TARGET_TEST_ON3:
        errors.append(
            f"Test must have {TARGET_TEST_ON3} ON3 family, found {test_fam_labels.count('ON3')}"
        )
    if test_fam_labels.count("OEXP") != TARGET_TEST_OEXP:
        errors.append(
            f"Test must have {TARGET_TEST_OEXP} OEXP family, found {test_fam_labels.count('OEXP')}"
        )
    if test_fam_labels.count("UNKNOWN") != TARGET_TEST_UNKNOWN:
        errors.append(
            f"Test must have {TARGET_TEST_UNKNOWN} UNKNOWN: {test_fam_labels.count('UNKNOWN')}"
        )

    # Rule hard count in test
    test_rh = sum(
        1 for fid in test_fams if fid in family_map and family_map[fid].tag == "rule_hard"
    )
    if test_rh > TARGET_TEST_RULE_HARD:
        errors.append(
            f"Test set rule_hard count exceeds maximum of {TARGET_TEST_RULE_HARD}: {test_rh}"
        )

    # 4. Check CV Folds
    folds = data.get("cv_folds", [])
    if len(folds) != NUM_CV_FOLDS:
        errors.append(f"Expected {NUM_CV_FOLDS} CV folds, found {len(folds)}")

    fold_family_union: set[str] = set()
    for f in folds:
        f_idx = f.get("fold")
        f_fams = set(f.get("families", []))
        expected_size = CV_FOLD_SIZES[f_idx] if f_idx < len(CV_FOLD_SIZES) else None
        if len(f_fams) != expected_size:
            errors.append(f"Fold {f_idx} expected {expected_size} families, found {len(f_fams)}")

        fold_overlap = fold_family_union & f_fams
        if fold_overlap:
            errors.append(f"Cross-fold leakage detected in fold {f_idx}: {fold_overlap}")
        fold_family_union.update(f_fams)

        fold_fam_labels = [family_map[fid].label for fid in f_fams if fid in family_map]
        if fold_fam_labels.count("ON3") != 1:
            errors.append(
                f"Fold {f_idx} must have 1 ON3 family, found {fold_fam_labels.count('ON3')}"
            )
        if fold_fam_labels.count("OEXP") != 1:
            errors.append(
                f"Fold {f_idx} must have 1 OEXP family, found {fold_fam_labels.count('OEXP')}"
            )

    if fold_family_union != dev_fams:
        errors.append("Union of CV fold families does not match development partition families.")

    # 5. Program records checks
    progs = data.get("programs", [])
    seen_prog_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for p in progs:
        pid = p.get("program_id")
        if pid in seen_prog_ids:
            errors.append(f"Duplicate program_id in manifest: {pid}")
        seen_prog_ids.add(pid)

        src_hash = p.get("source_sha256")
        if src_hash in seen_hashes:
            errors.append(f"Duplicate source_sha256 in manifest: {src_hash} ({pid})")
        seen_hashes.add(src_hash)

        if pid not in record_map:
            errors.append(f"Manifest program {pid} not found in dataset")
        else:
            rec = record_map[pid]
            if rec.source_sha256 != src_hash:
                errors.append(
                    f"Source SHA mismatch for {pid}: ds={rec.source_sha256}, m={src_hash}"
                )
            if rec.family_id != p.get("family_id"):
                errors.append(f"Family ID mismatch for {pid}")

    return errors


def freeze_manifest(
    proposal_path_or_manifest: Path | str | SplitManifest,
    frozen_output_path: Path | str,
    dataset_root: Path | str = "datasets",
    repo_root: Path | str = ".",
) -> Path:
    """Freeze a proposal manifest after verifying clean git state, ancestry, and validation gates.

    Writes the frozen manifest and a companion SHA-256 sidecar file: <output_path>.sha256.
    """
    out_path = Path(frozen_output_path)
    sidecar_path = out_path.with_name(f"{out_path.name}.sha256")

    # 1. Enforce Frozen-File Overwrite Protection Gate
    # Pre-flight existence checks must strictly prevent overwriting existing frozen manifests.
    if out_path.exists():
        raise FreezeGateError(
            f"Frozen manifest already exists at '{out_path}'. Overwrite is strictly forbidden."
        )
    if sidecar_path.exists():
        raise FreezeGateError(
            f"Sidecar checksum file already exists at '{sidecar_path}'. "
            "Overwrite is strictly forbidden."
        )

    # 2. Enforce git cleanliness gate
    commit, is_dirty = get_git_info(repo_root)
    if is_dirty or not commit:
        raise FreezeGateError(
            "Working tree is dirty; freeze requires a clean working tree."
        )

    # 3. Load proposal data
    proposal_file_path: Path | None = None
    if isinstance(proposal_path_or_manifest, (str, Path)):
        p = Path(proposal_path_or_manifest)
        if not p.exists():
            raise FreezeGateError(f"Proposal manifest not found: {p}")
        proposal_file_path = p
        manifest_data = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(proposal_path_or_manifest, SplitManifest):
        manifest_data = proposal_path_or_manifest.to_dict()
    else:
        manifest_data = copy.deepcopy(proposal_path_or_manifest)

    # 4. Resolve and Validate base_dataset_commit
    base_commit = manifest_data.get("base_dataset_commit") or manifest_data.get("git_commit")
    if not base_commit:
        raise FreezeGateError("Proposal manifest is missing 'base_dataset_commit'.")

    # Verify base commit exists in git
    res_cat = subprocess.run(
        ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
        cwd=str(repo_root),
        capture_output=True,
        check=False,
    )
    if res_cat.returncode != 0:
        raise FreezeGateError(
            f"Base dataset commit '{base_commit}' does not exist in repository."
        )

    # Verify Git Ancestry: HEAD must descend from base_dataset_commit
    res_anc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_commit, "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        check=False,
    )
    if res_anc.returncode != 0:
        raise FreezeGateError(
            f"Current HEAD commit ({commit}) is not a descendant of "
            f"base_dataset_commit ({base_commit})."
        )

    # 5. Dataset / Governance Change Gate: no dataset/governance modifications after base commit
    diff_paths = ["docs/FAMILY_PLAN.md", "docs/audits"]
    try:
        ds_rel = Path(dataset_root).resolve().relative_to(Path(repo_root).resolve()).as_posix()
        diff_paths.append(ds_rel)
    except ValueError:
        diff_paths.append("datasets")

    res_diff = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD", "--"] + diff_paths,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if res_diff.returncode != 0:
        err = res_diff.stderr.strip()
        raise FreezeGateError(
            f"Failed to check git diff between base commit '{base_commit}' and HEAD: {err}"
        )
    changed_gov = [line.strip() for line in res_diff.stdout.splitlines() if line.strip()]
    if changed_gov:
        raise FreezeGateError(
            f"Dataset or governance files were modified after base commit '{base_commit}': "
            f"{changed_gov}"
        )

    # 6. Verify proposal byte-identity against committed version in HEAD (if committed in HEAD)
    check_paths: list[str] = []
    if proposal_file_path is not None:
        try:
            rel = proposal_file_path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
            check_paths.append(rel)
        except ValueError:
            pass
    default_proposed = "data/splits/split_manifest_v1.0.proposed.json"
    if default_proposed not in check_paths:
        check_paths.append(default_proposed)

    for cand_rel in check_paths:
        res_show = subprocess.run(
            ["git", "show", f"HEAD:{cand_rel}"],
            cwd=str(repo_root),
            capture_output=True,
            check=False,
        )
        if res_show.returncode == 0:
            try:
                head_manifest = json.loads(res_show.stdout.decode("utf-8"))
            except Exception as e:
                raise FreezeGateError(
                    f"Committed proposal in HEAD at '{cand_rel}' is not valid JSON: {e}"
                ) from e

            # Compare comparable dictionary contents (excluding dynamic timestamp)
            def _strip_volatile(d: dict[str, Any]) -> dict[str, Any]:
                c = copy.deepcopy(d)
                c.pop("created_at", None)
                return c

            if _strip_volatile(manifest_data) != _strip_volatile(head_manifest):
                raise FreezeGateError(
                    f"Proposal content differs from the version committed in HEAD at '{cand_rel}'."
                )
            break

    # 7. Validate proposal manifest
    errors = validate_manifest(manifest_data, dataset_root=dataset_root)
    if errors:
        raise SplitValidationError(f"Cannot freeze invalid manifest: {errors}")

    # 8. Update status and git metadata
    manifest_data["status"] = "frozen"
    manifest_data["git_commit"] = commit
    manifest_data["proposal_commit"] = commit
    manifest_data["base_dataset_commit"] = base_commit
    manifest_data["git_dirty"] = False
    manifest_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 9. Exclusive Creation and Rollback Cleanup
    # Note on crash consistency: While userland cleanup safely handles application-level
    # exceptions and unlinks newly created artifacts if writing the sidecar fails, standard OS
    # filesystems do not offer multi-file transactional atomicity against power loss or SIGKILL
    # between two distinct file creations.
    if sidecar_path.exists():
        raise FreezeGateError(
            f"Sidecar checksum file already exists at '{sidecar_path}'. "
            "Overwrite is strictly forbidden."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_bytes = canonical_json_dumps(manifest_data).encode("utf-8")
    sha256_hash = hashlib.sha256(json_bytes).hexdigest()
    sidecar_bytes = f"{sha256_hash}  {out_path.name}\n".encode("utf-8")

    manifest_created_by_us = False
    sidecar_created_by_us = False

    try:
        # Step 1: Exclusively create frozen manifest in 'xb' mode
        try:
            f_manifest = open(out_path, "xb")
        except FileExistsError as exc:
            raise FreezeGateError(
                f"Frozen manifest already exists at '{out_path}'. "
                "Overwrite is strictly forbidden."
            ) from exc
        except Exception as exc:
            raise FreezeGateError(
                f"Failed to create frozen manifest at '{out_path}': {exc}"
            ) from exc

        manifest_created_by_us = True
        try:
            with f_manifest:
                f_manifest.write(json_bytes)
                f_manifest.flush()
        except Exception as exc:
            raise FreezeGateError(
                f"Failed to write frozen manifest at '{out_path}': {exc}"
            ) from exc

        # Step 2: Exclusively create sidecar checksum file in 'xb' mode
        try:
            f_sidecar = open(sidecar_path, "xb")
        except FileExistsError as exc:
            raise FreezeGateError(
                f"Sidecar checksum file already exists at '{sidecar_path}'. "
                "Overwrite is strictly forbidden."
            ) from exc
        except Exception as exc:
            raise FreezeGateError(
                f"Failed to create sidecar checksum file at '{sidecar_path}': {exc}"
            ) from exc

        sidecar_created_by_us = True
        try:
            with f_sidecar:
                f_sidecar.write(sidecar_bytes)
                f_sidecar.flush()
        except Exception as exc:
            raise FreezeGateError(
                f"Failed to write sidecar checksum file at '{sidecar_path}': {exc}"
            ) from exc

    except Exception as primary_exc:
        # Cleanup newly created artifacts if writing fails; never delete pre-existing files
        cleanup_errors: list[Exception] = []
        if sidecar_created_by_us:
            try:
                sidecar_path.unlink(missing_ok=True)
            except Exception as cl_exc:
                cleanup_errors.append(cl_exc)
        if manifest_created_by_us:
            try:
                out_path.unlink(missing_ok=True)
            except Exception as cl_exc:
                cleanup_errors.append(cl_exc)

        if cleanup_errors:
            err_details = "; ".join(f"{type(e).__name__}: {e}" for e in cleanup_errors)
            raise FreezeGateError(
                f"{primary_exc}; additionally, rollback cleanup failed: {err_details}"
            ) from primary_exc
        raise

    return out_path


def verify_frozen_manifest(
    frozen_manifest_path: Path | str,
    dataset_root: Path | str = "datasets",
    repo_root: Path | str = ".",
) -> dict[str, Any]:
    """Verify integrity of a frozen manifest against its SHA-256 sidecar and the dataset."""
    m_path = Path(frozen_manifest_path)
    if not m_path.exists():
        raise SplitValidationError(f"Frozen manifest file not found: {m_path}")

    # 1. Verify sidecar
    sidecar_path = m_path.with_name(f"{m_path.name}.sha256")
    if not sidecar_path.exists():
        raise SplitValidationError(f"Sidecar checksum file not found: {sidecar_path}")

    sidecar_line = sidecar_path.read_text(encoding="utf-8").strip()
    expected_hash = sidecar_line.split()[0]

    actual_hash = hashlib.sha256(m_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise SplitValidationError(
            f"Sidecar checksum mismatch: expected {expected_hash}, calculated {actual_hash}"
        )

    # 2. Parse and validate manifest contents
    manifest_data = json.loads(m_path.read_text(encoding="utf-8"))
    if manifest_data.get("status") != "frozen":
        raise SplitValidationError(
            f"Expected status 'frozen', found '{manifest_data.get('status')}'"
        )

    errors = validate_manifest(manifest_data, dataset_root=dataset_root)
    if errors:
        raise SplitValidationError(f"Frozen manifest validation failed: {errors}")

    return {
        "status": "VALID",
        "manifest_path": str(m_path),
        "sha256": actual_hash,
        "partitions": manifest_data.get("partitions"),
        "cv_folds": manifest_data.get("cv_folds"),
    }
