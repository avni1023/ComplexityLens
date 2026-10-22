"""
stl_rules.py — STL and container cost-model rules for ComplexityLens.

Implements deterministic classification for:
- Hash-container operations (insert/find/erase/operator[]) with configurable
  cost_model: ``"expected_hash"`` (default) or ``"strict_worst_case"``.
- Ordered-container operations (map/set) → O(log n).
- Vector push_back/emplace_back (amortized O(1)), insert/erase.
- Standard algorithms: std::sort, std::find, std::count, std::accumulate,
  std::lower_bound, std::upper_bound, std::binary_search on random-access ranges.

All functions are pure / stateless.
"""

from __future__ import annotations

from complexity_lens.complexity.models import (
    ComplexityClass,
    Exactness,
    ReasonCode,
)
from complexity_lens.ir.model import STLOperationIR, STLOperationKind

# ---------------------------------------------------------------------------
# Supported cost_model values
# ---------------------------------------------------------------------------
HASH_EXPECTED = "expected_hash"
HASH_STRICT = "strict_worst_case"
_VALID_COST_MODELS = {HASH_EXPECTED, HASH_STRICT}

# Operations covered by the hash-container cost model
_HASH_OPS = {
    STLOperationKind.HASH_LOOKUP,
}

# Operations covered by the ordered-container cost model
_ORDERED_OPS = {
    STLOperationKind.ORDERED_LOOKUP,
}

# Supported random-access container kinds (from STLOperationIR.container or callee context)
_RANDOM_ACCESS_KINDS = {"vector", "array", "deque"}

# Supported explicit comparator argument texts (empty means default ordering)
_SUPPORTED_COMPARATORS = {None, "", "std::less<>", "std::less<", "std::greater<>", "std::greater<"}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def validate_cost_model(cost_model: str) -> None:
    """Raise ValueError for unsupported cost_model values.

    Parameters
    ----------
    cost_model:
        Must be one of ``"expected_hash"`` or ``"strict_worst_case"``.

    Raises
    ------
    ValueError
        If ``cost_model`` is not one of the supported values.
    """
    if cost_model not in _VALID_COST_MODELS:
        raise ValueError(
            f"Unsupported hash container cost_model: {cost_model!r}. "
            f"Supported values: {sorted(_VALID_COST_MODELS)}"
        )


def classify_stl_operation(
    op: STLOperationIR,
    cost_model: str,
    param_names: set[str],
    cin_vars: set[str],
    named_constants: set[str],
    local_constants: dict[str, int],
) -> tuple[ComplexityClass, Exactness, list[str], list[str]]:
    """Classify a single STLOperationIR node into a ComplexityClass.

    Parameters
    ----------
    op:
        The STL operation IR node to classify.
    cost_model:
        Hash-container cost model; must be ``"expected_hash"`` or ``"strict_worst_case"``.
    param_names:
        Set of function parameter names that are considered input-size symbols.
    cin_vars:
        Variables populated from ``std::cin`` (treated as input-size symbols).
    named_constants:
        Named compile-time constant identifiers treated as input-size symbols.
    local_constants:
        Mapping of local variable names to their literal integer values.

    Returns
    -------
    (complexity_class, exactness, reason_codes, assumptions)
    """
    validate_cost_model(cost_model)

    kind = op.operation
    args = op.arguments or []
    container = op.container
    container_kind_source = op.container_kind_source

    # Collect source-text provenance assumption if relevant
    provenance_assumptions: list[str] = []
    if container_kind_source == "source_text":
        provenance_assumptions = [ReasonCode.CONTAINER_KIND_FROM_SOURCE_TEXT.value]

    # ----------------------------------------------------------------
    # 1. Hash-container operations
    # ----------------------------------------------------------------
    if kind == STLOperationKind.HASH_LOOKUP:
        if cost_model == HASH_EXPECTED:
            return (
                ComplexityClass.constant(),
                Exactness.STRUCTURAL,
                [ReasonCode.STL_HASH_OP.value] + provenance_assumptions,
                [ReasonCode.HASH_AVERAGE_CASE.value] + provenance_assumptions,
            )
        else:  # strict_worst_case
            return (
                ComplexityClass.linear(),
                Exactness.UPPER_BOUND,
                [ReasonCode.STL_HASH_OP.value] + provenance_assumptions,
                provenance_assumptions,
            )

    # ----------------------------------------------------------------
    # 2. Ordered-container operations (map/set member find/count/lower_bound)
    # ----------------------------------------------------------------
    if kind == STLOperationKind.ORDERED_LOOKUP:
        return (
            ComplexityClass.logarithmic(),
            Exactness.STRUCTURAL,
            [ReasonCode.STL_ORDERED_OP.value] + provenance_assumptions,
            provenance_assumptions,
        )

    # ----------------------------------------------------------------
    # 3. Vector push_back / emplace_back — amortized O(1)
    # ----------------------------------------------------------------
    if kind == STLOperationKind.PUSH_BACK:
        return (
            ComplexityClass.constant(),
            Exactness.STRUCTURAL,
            [ReasonCode.AMORTIZED_CONSTANT_CALL.value, ReasonCode.AMORTIZED.value]
            + provenance_assumptions,
            [ReasonCode.AMORTIZED.value] + provenance_assumptions,
        )

    # ----------------------------------------------------------------
    # 4. std::sort
    # ----------------------------------------------------------------
    if kind == STLOperationKind.SORT:
        comparator = args[2] if len(args) >= 3 else None
        if not _is_supported_comparator(comparator):
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [
                    ReasonCode.UNSUPPORTED_COMPARATOR.value,
                    ReasonCode.UNSUPPORTED_CALL.value,
                ]
                + provenance_assumptions,
                provenance_assumptions,
            )
        # Validate range form
        range_ok, range_sym = _validate_random_access_range(
            args, container, param_names, cin_vars, named_constants, local_constants
        )
        if not range_ok:
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [
                    ReasonCode.UNSUPPORTED_RANGE.value,
                    ReasonCode.UNSUPPORTED_CALL.value,
                ]
                + provenance_assumptions,
                provenance_assumptions,
            )
        assumptions = list(provenance_assumptions)
        if range_sym is None:
            return (
                ComplexityClass.constant(),
                Exactness.STRUCTURAL,
                [ReasonCode.CONSTANT_WORK.value] + provenance_assumptions,
                assumptions,
            )
        return (
            ComplexityClass.linearithmic(),
            Exactness.STRUCTURAL,
            [ReasonCode.STL_SORT.value] + provenance_assumptions,
            assumptions,
        )

    # ----------------------------------------------------------------
    # 5. std::find / std::count / std::accumulate — O(n)
    # ----------------------------------------------------------------
    if kind == STLOperationKind.FIND:
        range_ok, range_sym = _validate_random_access_range(
            args, container, param_names, cin_vars, named_constants, local_constants
        )
        if not range_ok:
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNSUPPORTED_RANGE.value] + provenance_assumptions,
                provenance_assumptions,
            )
        if range_sym is None:
            return (
                ComplexityClass.constant(),
                Exactness.STRUCTURAL,
                [ReasonCode.CONSTANT_WORK.value] + provenance_assumptions,
                list(provenance_assumptions),
            )
        return (
            ComplexityClass.linear(),
            Exactness.STRUCTURAL,
            [ReasonCode.STL_FIND.value] + provenance_assumptions,
            list(provenance_assumptions),
        )

    if kind == STLOperationKind.COUNT:
        range_ok, range_sym = _validate_random_access_range(
            args, container, param_names, cin_vars, named_constants, local_constants
        )
        if not range_ok:
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNSUPPORTED_RANGE.value] + provenance_assumptions,
                provenance_assumptions,
            )
        if range_sym is None:
            return (
                ComplexityClass.constant(),
                Exactness.STRUCTURAL,
                [ReasonCode.CONSTANT_WORK.value] + provenance_assumptions,
                list(provenance_assumptions),
            )
        return (
            ComplexityClass.linear(),
            Exactness.STRUCTURAL,
            [ReasonCode.STL_FIND.value] + provenance_assumptions,
            list(provenance_assumptions),
        )

    # ----------------------------------------------------------------
    # 6. std::lower_bound / std::upper_bound / std::binary_search — O(log n)
    #    Only for random-access ranges.
    # ----------------------------------------------------------------
    if kind == STLOperationKind.LOWER_BOUND:
        # Check if the container is random-access; set/map iterators are NOT random-access
        if not _is_random_access_container(container, args):
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNSUPPORTED_ITERATOR.value] + provenance_assumptions,
                provenance_assumptions,
            )
        range_ok, range_sym = _validate_random_access_range(
            args, container, param_names, cin_vars, named_constants, local_constants
        )
        if not range_ok:
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNSUPPORTED_RANGE.value] + provenance_assumptions,
                provenance_assumptions,
            )
        if range_sym is None:
            return (
                ComplexityClass.constant(),
                Exactness.STRUCTURAL,
                [ReasonCode.CONSTANT_WORK.value] + provenance_assumptions,
                list(provenance_assumptions),
            )
        return (
            ComplexityClass.logarithmic(),
            Exactness.STRUCTURAL,
            [ReasonCode.STL_BINARY_SEARCH.value] + provenance_assumptions,
            list(provenance_assumptions),
        )

    if kind == STLOperationKind.BINARY_SEARCH:
        if not _is_random_access_container(container, args):
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNSUPPORTED_ITERATOR.value] + provenance_assumptions,
                provenance_assumptions,
            )
        range_ok, range_sym = _validate_random_access_range(
            args, container, param_names, cin_vars, named_constants, local_constants
        )
        if not range_ok:
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNSUPPORTED_RANGE.value] + provenance_assumptions,
                provenance_assumptions,
            )
        if range_sym is None:
            return (
                ComplexityClass.constant(),
                Exactness.STRUCTURAL,
                [ReasonCode.CONSTANT_WORK.value] + provenance_assumptions,
                list(provenance_assumptions),
            )
        return (
            ComplexityClass.logarithmic(),
            Exactness.STRUCTURAL,
            [ReasonCode.STL_BINARY_SEARCH.value] + provenance_assumptions,
            list(provenance_assumptions),
        )

    # ----------------------------------------------------------------
    # 7. vector::insert / vector::erase
    # ----------------------------------------------------------------
    if kind == STLOperationKind.INSERT:
        return _classify_vector_insert_erase(
            op, args, container, param_names, cin_vars, named_constants,
            local_constants, provenance_assumptions
        )

    if kind == STLOperationKind.ERASE:
        return _classify_vector_insert_erase(
            op, args, container, param_names, cin_vars, named_constants,
            local_constants, provenance_assumptions
        )

    # ----------------------------------------------------------------
    # Fallback: unrecognized or unsupported operation kind
    # ----------------------------------------------------------------
    return (
        ComplexityClass.unknown(),
        Exactness.UNKNOWN,
        [ReasonCode.UNSUPPORTED_CALL.value] + provenance_assumptions,
        provenance_assumptions,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_supported_comparator(comparator: str | None) -> bool:
    """Return True if comparator is supported (None/empty or std::less/greater)."""
    if comparator is None or comparator.strip() == "":
        return True
    stripped = comparator.strip()
    # Accept std::less<> or std::greater<> (with or without the closing >)
    for prefix in ("std::less<", "std::greater<"):
        if stripped.startswith(prefix):
            return True
    return False


def _is_input_symbol(expr: str, param_names: set[str], cin_vars: set[str],
                     named_constants: set[str]) -> bool:
    """Return True if expr is a known input-size symbol."""
    return (
        expr in param_names
        or expr in cin_vars
        or expr in named_constants
    )


def _is_literal_or_local_const(expr: str, local_constants: dict[str, int]) -> bool:
    """Return True if expr is a literal integer or a local constant."""
    return expr.isdigit() or expr in local_constants


def _is_random_access_container(container: str | None, args: list[str]) -> bool:
    """Determine whether the container is random-access.

    Supported: vector, array, deque, and C-style array names.
    Non-random-access: set, unordered_set, map, unordered_map iterators.
    """
    if container is not None:
        low = container.lower()
        # map/set containers are explicitly non-random-access
        for non_ra in ("set", "map"):
            if non_ra in low:
                return False

    # Check for C-style array / pointer range: (a, a+n)
    if len(args) >= 2:
        first = args[0].strip().replace(" ", "")
        second = args[1].strip().replace(" ", "")
        if "+" in second:
            base = second.split("+", 1)[0].strip()
            if base == first:
                return True

    if container is None:
        return False

    low = container.lower()
    for kind in _RANDOM_ACCESS_KINDS:
        if kind in low:
            return True

    # Default: unknown, be conservative
    return False



def _validate_random_access_range(
    args: list[str],
    container: str | None,
    param_names: set[str],
    cin_vars: set[str],
    named_constants: set[str],
    local_constants: dict[str, int],
) -> tuple[bool, str | None]:
    """Check that the range arguments are a supported form.

    Supported forms:
    - ``(c.begin(), c.end())`` where c is a known container variable.
    - ``(a, a + n)`` where n is a recognized input-size symbol or literal.

    Returns
    -------
    (ok, input_symbol_or_None)
    """
    if len(args) < 2:
        return False, None

    first = args[0].strip().replace(" ", "")
    second = args[1].strip().replace(" ", "")

    # Form 1: c.begin(), c.end() — container must be known and recognized
    if first.endswith(".begin()") and second.endswith(".end()"):
        prefix_first = first[: -len(".begin()")]
        prefix_second = second[: -len(".end()")]
        if prefix_first and prefix_first == prefix_second:
            return True, prefix_first
        if not prefix_first and not prefix_second and container:
            return True, container
        return False, None

    if first in ("begin()", "begin") and second in ("end()", "end") and container:
        return True, container

    # Form 2: a, a+n
    if "+" in second:
        parts = second.split("+", 1)
        base = parts[0].strip()
        n_part = parts[1].strip().strip("()") if len(parts) > 1 else ""
        if base == first:
            if _is_input_symbol(n_part, param_names, cin_vars, named_constants):
                return True, n_part
            if _is_literal_or_local_const(n_part, local_constants):
                if not _is_input_symbol(n_part, param_names, cin_vars, named_constants):
                    return True, None  # constant range
        return False, None

    # If container is known and recognized as random-access, and args don't
    # match the two forms above, reject
    return False, None


def _classify_vector_insert_erase(
    op: STLOperationIR,
    args: list[str],
    container: str | None,
    param_names: set[str],
    cin_vars: set[str],
    named_constants: set[str],
    local_constants: dict[str, int],
    provenance_assumptions: list[str],
) -> tuple[ComplexityClass, Exactness, list[str], list[str]]:
    """Classify vector insert/erase at end (amortized O(1)) or arbitrary (O(n)).

    Returns UNKNOWN when position or size cannot be determined.
    """
    # If we can prove insertion is at end, it's amortized O(1)
    # Position argument: args[0] typically is the iterator position
    # Without deep semantic resolution, conservatively use O(n) with UPPER_BOUND
    # if the container has a recognized input-size relationship; UNKNOWN otherwise.
    if container is None:
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_CALL.value] + provenance_assumptions,
            provenance_assumptions,
        )

    # Check if the container is related to an input symbol (size is n)
    container_is_input = (
        container in param_names
        or container in cin_vars
        or container in named_constants
    )

    if container_is_input:
        return (
            ComplexityClass.linear(),
            Exactness.UPPER_BOUND,
            [ReasonCode.VECTOR_INSERT_ERASE.value] + provenance_assumptions,
            [ReasonCode.AMORTIZED.value] + provenance_assumptions,
        )

    return (
        ComplexityClass.unknown(),
        Exactness.UNKNOWN,
        [ReasonCode.UNSUPPORTED_CALL.value] + provenance_assumptions,
        provenance_assumptions,
    )
