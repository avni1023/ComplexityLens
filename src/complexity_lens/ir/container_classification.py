"""
container_classification.py — Container-kind classification helpers for ComplexityLens.

Determines whether containers are hash-based (unordered_map, unordered_set)
or ordered trees (map, set), or random-access (vector, array, deque).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from complexity_lens.ir.model import FunctionIR


def detect_container_kind(type_spelling: str | None) -> str | None:
    """Classify a declared type spelling into a canonical container kind."""
    if not type_spelling:
        return None
    low = type_spelling.lower()

    if "unordered_map" in low:
        return "unordered_map"
    if "unordered_set" in low:
        return "unordered_set"
    if re.search(r"\bmap\b", low) or "std::map" in low:
        return "map"
    if re.search(r"\bset\b", low) or "std::set" in low:
        return "set"
    if "vector" in low:
        return "vector"
    if "deque" in low:
        return "deque"
    if "array" in low:
        return "array"

    return None


def get_container_kind(
    container_name: str | None,
    func: FunctionIR | None = None,
) -> str | None:
    """Retrieve the container kind for a variable name within a function context."""
    if not container_name:
        return None

    # 1. Check FunctionIR.container_types if present
    if func is not None:
        c_types = getattr(func, "container_types", None)
        if c_types and container_name in c_types:
            return c_types[container_name]

        # 2. Check parameters
        for p_type, p_name in func.parameters:
            if p_name == container_name:
                kind = detect_container_kind(p_type)
                if kind:
                    return kind

    # 3. Best-effort fallback on the variable identifier itself
    low = container_name.lower()
    if "unordered_map" in low:
        return "unordered_map"
    if "unordered_set" in low:
        return "unordered_set"
    if "map" in low and "unordered" not in low:
        return "map"
    if "set" in low and "unordered" not in low:
        return "set"
    if "vec" in low:
        return "vector"

    return None


def is_hash_container(container_name: str | None, func: FunctionIR | None = None) -> bool:
    """Return True if container is an unordered_map or unordered_set."""
    kind = get_container_kind(container_name, func)
    return kind in ("unordered_map", "unordered_set")


def is_ordered_container(container_name: str | None, func: FunctionIR | None = None) -> bool:
    """Return True if container is a map or set (and not unordered)."""
    kind = get_container_kind(container_name, func)
    return kind in ("map", "set")
