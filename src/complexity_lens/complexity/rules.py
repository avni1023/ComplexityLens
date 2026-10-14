"""
rules.py — Deterministic Static Complexity Rules for ComplexityLens.

Implements:
- Dominance ordering and asymptotic max composition.
- Multiplication for nested constructs.
- Constant-cost allowlist checking.
- Loop classification rules (linear, doubling, halving, range-for, early-exit).
- Strict UNKNOWN propagation following a documented conservative policy.
"""

from __future__ import annotations

from complexity_lens.complexity.models import (
    ComplexityClass,
    Exactness,
    ReasonCode,
)
from complexity_lens.ir.loop_classification import (
    LoopBoundCategory,
    classify_loop_structural,
)
from complexity_lens.ir.model import LoopIR

# Allowlisted call names that are known O(1)
_ALLOWLISTED_CALLS = {
    "cin",
    "cout",
    "operator<<",
    "operator>>",
    "operator[]",
    "size",
    "empty",
    "min",
    "max",
    "swap",
    "abs",
    "std::min",
    "std::max",
    "std::swap",
    "std::abs",
}

_AMORTIZED_CALLS = {
    "push_back",
    "emplace_back",
}

_CONTAINER_CONSTRUCTORS = {
    "vector",
    "string",
    "array",
    "pair",
    "tuple",
}


def dominant(c1: ComplexityClass, c2: ComplexityClass) -> ComplexityClass:
    """Return the asymptotically dominant complexity class between c1 and c2.

    Follows a documented conservative UNKNOWN propagation policy:
    - If either is UNKNOWN, the dominant result is UNKNOWN.
    - If either is EXP, EXP dominates any polynomial/logarithmic class.
    - Otherwise, compares polynomial degree first, then logarithmic degree.

    Satisfies commutativity: dominant(a, b) == dominant(b, a).
    Satisfies associativity: dominant(dominant(a, b), c) == dominant(a, dominant(b, c)).
    """
    if c1.is_unknown or c2.is_unknown:
        return ComplexityClass.unknown()
    if c1.is_exp or c2.is_exp:
        return ComplexityClass.exponential()

    p1, q1 = c1.p or 0, c1.q or 0
    p2, q2 = c2.p or 0, c2.q or 0

    if p1 > p2:
        return c1
    elif p2 > p1:
        return c2
    elif q1 > q2:
        return c1
    elif q2 > q1:
        return c2
    else:
        return c1


def multiply(c1: ComplexityClass, c2: ComplexityClass) -> ComplexityClass:
    """Multiply two complexity classes for nested independent iteration.

    Follows conservative UNKNOWN policy:
    - If either is UNKNOWN, the result is UNKNOWN.
    - If either is EXP, the result is EXP.
    - For ordinary classes, adds polynomial degrees and logarithmic degrees.

    Satisfies commutativity and associativity.
    """
    if c1.is_unknown or c2.is_unknown:
        return ComplexityClass.unknown()
    if c1.is_exp or c2.is_exp:
        return ComplexityClass.exponential()

    p1, q1 = c1.p or 0, c1.q or 0
    p2, q2 = c2.p or 0, c2.q or 0

    return ComplexityClass(p=p1 + p2, q=q1 + q2)


def compare_dominance(c1: ComplexityClass, c2: ComplexityClass) -> int:
    """Compare two complexity classes in asymptotic dominance ordering.

    Returns:
         1 if c1 strictly dominates c2
        -1 if c2 strictly dominates c1
         0 if c1 and c2 have the same asymptotic order

    Raises ValueError if either class is UNKNOWN.
    """
    if c1.is_unknown or c2.is_unknown:
        raise ValueError("Cannot order UNKNOWN complexity class")

    if c1.is_exp and c2.is_exp:
        return 0
    if c1.is_exp:
        return 1
    if c2.is_exp:
        return -1

    p1, q1 = c1.p or 0, c1.q or 0
    p2, q2 = c2.p or 0, c2.q or 0

    if p1 > p2:
        return 1
    elif p2 > p1:
        return -1
    elif q1 > q2:
        return 1
    elif q2 > q1:
        return -1
    return 0


def is_call_allowlisted(
    callee_name: str,
    arguments: list[str] | None = None,
) -> tuple[bool, bool]:
    """Check if a callee name is on the constant-cost allowlist.

    Returns
    -------
    (is_allowlisted, is_amortized)
    """
    clean = callee_name.strip()
    if clean.startswith("std::"):
        clean_stem = clean[5:]
    else:
        clean_stem = clean

    if clean in _AMORTIZED_CALLS or clean_stem in _AMORTIZED_CALLS:
        return True, True

    if clean in _ALLOWLISTED_CALLS or clean_stem in _ALLOWLISTED_CALLS:
        return True, False

    if clean in _CONTAINER_CONSTRUCTORS or clean_stem in _CONTAINER_CONSTRUCTORS:
        # Genuine default constructor with 0 arguments is O(1)
        if not arguments:
            return True, False
        return False, False

    return False, False


def classify_loop(
    loop: LoopIR,
    local_constants: dict[str, int],
    named_constants: set[str],
    cin_vars: set[str],
    param_names: set[str],
) -> tuple[ComplexityClass, Exactness, list[str], str | None]:
    """Classify the iteration complexity of a single loop construct.

    Returns
    -------
    (iteration_class, exactness, reason_codes, input_symbol)
    """
    struct_res = classify_loop_structural(
        loop,
        local_constants=local_constants,
        named_constants=named_constants,
        cin_vars=cin_vars,
        param_names=param_names,
    )

    reason = struct_res.reason
    input_sym = struct_res.input_symbol
    is_named_c = struct_res.is_named_constant

    if struct_res.category == LoopBoundCategory.LINEAR_BOUND:
        reasons = [ReasonCode.LINEAR_LOOP.value]
        if is_named_c:
            reasons.append(ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value)
        return (
            ComplexityClass.linear(),
            Exactness.STRUCTURAL,
            reasons,
            input_sym,
        )

    if struct_res.category == LoopBoundCategory.CONST_BOUND:
        return (
            ComplexityClass.constant(),
            Exactness.STRUCTURAL,
            [ReasonCode.CONSTANT_BOUND_LOOP.value],
            None,
        )

    if struct_res.category == LoopBoundCategory.LOG_UPDATE:
        if reason == ReasonCode.CONSTANT_BOUND_LOOP.value:
            return (
                ComplexityClass.constant(),
                Exactness.STRUCTURAL,
                [ReasonCode.CONSTANT_BOUND_LOOP.value],
                None,
            )
        reasons = [ReasonCode.LOGARITHMIC_UPDATE.value]
        if is_named_c:
            reasons.append(ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value)
        return (
            ComplexityClass.logarithmic(),
            Exactness.STRUCTURAL,
            reasons,
            input_sym,
        )

    if struct_res.category == LoopBoundCategory.DEPENDENT_BOUND:
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.DEPENDENT_BOUND.value],
            None,
        )

    # UNKNOWN_BOUND
    reason_code = (
        ReasonCode.LOOP_VAR_MODIFIED.value
        if reason == ReasonCode.LOOP_VAR_MODIFIED.value
        else ReasonCode.UNKNOWN_BOUND.value
    )
    return (
        ComplexityClass.unknown(),
        Exactness.UNKNOWN,
        [reason_code],
        None,
    )
