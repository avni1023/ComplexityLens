"""
schema.py — Authoritative 26-feature schema definition for ComplexityLens.

Defines:
- FEATURE_SCHEMA_VERSION = "1.0"
- FEATURE_NAMES: The exact 26 feature names in documented canonical order.
- FeatureVector: Immutable, ordered named structure with to_list() and to_dict().
"""

from __future__ import annotations

import dataclasses
from typing import Any

FEATURE_SCHEMA_VERSION = "1.0"

FEATURE_NAMES: tuple[str, ...] = (
    # Loops — 11 features
    "n_loops",
    "max_loop_depth",
    "n_sequential_top_loops",
    "n_nested_loop_pairs",
    "n_for",
    "n_while",
    "n_loops_linear_bound",
    "n_loops_const_bound",
    "n_loops_log_update",
    "n_dependent_bound_loops",
    "n_loops_unknown_bound",
    # Recursion — 5 features
    "has_recursion",
    "n_recursive_call_sites",
    "shrink_minus_const",
    "shrink_divide",
    "shrink_other",
    # Calls and operations — 6 features
    "n_user_calls",
    "n_sort_calls",
    "n_linear_search_calls",
    "n_binary_search_calls",
    "n_hash_ops",
    "n_ordered_container_ops",
    # Other — 4 features
    "n_container_ops_in_loops",
    "has_early_exit_in_loop",
    "n_input_dependent_ops",
    "n_unsupported_constructs",
)

_FEATURE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
_BOOLEAN_FEATURES: set[str] = {"has_recursion", "has_early_exit_in_loop"}


@dataclasses.dataclass(frozen=True)
class FeatureVector:
    """Deterministic structural feature vector representing a C++ translation unit.

    Attributes
    ----------
    names:
        Canonical tuple of exactly 26 feature names.
    values:
        Canonical tuple of exactly 26 non-negative integer values.
    """

    names: tuple[str, ...]
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.names, tuple):
            object.__setattr__(self, "names", tuple(self.names))
        if not isinstance(self.values, tuple):
            object.__setattr__(self, "values", tuple(self.values))

        if len(self.names) != 26:
            raise ValueError(f"Feature names tuple must have length 26, got {len(self.names)}")
        if len(self.values) != 26:
            raise ValueError(f"Feature values tuple must have length 26, got {len(self.values)}")

        for name, val in zip(self.names, self.values):
            if not isinstance(val, int) or isinstance(val, bool):
                msg = f"Feature '{name}' value must be an int, got {type(val).__name__}: {val!r}"
                raise TypeError(msg)
            if val < 0:
                raise ValueError(f"Feature '{name}' must be non-negative, got {val}")
            if name in _BOOLEAN_FEATURES and val not in (0, 1):
                raise ValueError(f"Boolean feature '{name}' must be exactly 0 or 1, got {val}")

    def to_list(self) -> list[int]:
        """Return ordered list of feature values."""
        return list(self.values)

    def to_dict(self) -> dict[str, int]:
        """Return dictionary mapping feature names to values preserving schema order."""
        return dict(zip(self.names, self.values))

    def __getitem__(self, item: int | str) -> int:
        if isinstance(item, int):
            return self.values[item]
        if isinstance(item, str):
            if item in _FEATURE_INDEX:
                return self.values[_FEATURE_INDEX[item]]
            raise KeyError(f"Unknown feature name: {item!r}")
        raise TypeError(f"Index must be int or str, got {type(item).__name__}")

    def __getattr__(self, name: str) -> Any:
        if name in _FEATURE_INDEX:
            return self.values[_FEATURE_INDEX[name]]
        raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")
