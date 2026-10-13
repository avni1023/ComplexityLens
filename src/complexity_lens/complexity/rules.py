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
from complexity_lens.ir.model import BoundKind, LoopIR, LoopKind, UpdateKind

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
    "vector",
    "std::vector",
    "string",
    "std::string",
    "array",
    "std::array",
    "pair",
    "std::pair",
}

_AMORTIZED_CALLS = {
    "push_back",
    "emplace_back",
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


def is_call_allowlisted(callee_name: str) -> tuple[bool, bool]:
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
    # 1. Loop variable or bound variable modified in loop body -> UNKNOWN
    if loop.loop_var_modified:
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.LOOP_VAR_MODIFIED.value],
            None,
        )

    # 2. Triangular / Dependent bound -> UNKNOWN with DEPENDENT_BOUND
    if loop.is_dependent_bound or loop.bound.kind == BoundKind.DEPENDENT:
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.DEPENDENT_BOUND.value],
            None,
        )

    # 3. Unsupported comparison operator ('!=') -> UNKNOWN
    if loop.bound.comparison_op == "!=":
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNKNOWN_BOUND.value],
            None,
        )

    # 4. Range-for loops
    if loop.kind == LoopKind.RANGE_FOR:
        if loop.is_fixed_size_array:
            return (
                ComplexityClass.constant(),
                Exactness.STRUCTURAL,
                [ReasonCode.CONSTANT_BOUND_LOOP.value],
                None,
            )
        if loop.container_name and (
            loop.container_name in param_names or loop.container_name in cin_vars
        ):
            return (
                ComplexityClass.linear(),
                Exactness.STRUCTURAL,
                [ReasonCode.LINEAR_LOOP.value],
                loop.container_name,
            )
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNKNOWN_BOUND.value],
            None,
        )

    # 5. Doubling loops (i *= c, i = i * c, i += i with literal c >= 2)
    is_mult = (loop.update.kind == UpdateKind.MULTIPLICATIVE)
    is_doubling_step = (loop.update.step_op in ("*=", ">>=")) or (
        loop.update.step_expr and "*" in loop.update.step_expr
    )
    is_halving_step = (loop.update.step_op == "/=") or (
        loop.update.step_expr and "/" in loop.update.step_expr
    )

    if is_mult and is_doubling_step:
        # Require supported condition (< or <=)
        if loop.bound.comparison_op not in ("<", "<="):
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNKNOWN_BOUND.value],
                None,
            )
        # Require initialization provably at least 1
        if loop.init_val is None:
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNKNOWN_BOUND.value],
                None,
            )
        if loop.init_val == 0:
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNKNOWN_BOUND.value],
                None,
            )

        # Check bound
        b_expr = loop.bound.bound_expr
        if b_expr:
            if b_expr.isdigit() or (b_expr in local_constants):
                return (
                    ComplexityClass.constant(),
                    Exactness.STRUCTURAL,
                    [ReasonCode.CONSTANT_BOUND_LOOP.value],
                    None,
                )
            if b_expr in named_constants:
                return (
                    ComplexityClass.logarithmic(),
                    Exactness.STRUCTURAL,
                    [
                        ReasonCode.LOGARITHMIC_UPDATE.value,
                        ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value,
                    ],
                    b_expr,
                )
            if (
                b_expr in param_names
                or b_expr in cin_vars
                or loop.bound.kind == BoundKind.LINEAR_INPUT
            ):
                return (
                    ComplexityClass.logarithmic(),
                    Exactness.STRUCTURAL,
                    [ReasonCode.LOGARITHMIC_UPDATE.value],
                    b_expr,
                )

        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNKNOWN_BOUND.value],
            None,
        )

    # 6. Halving loops (n /= c, i /= c with literal c >= 2)
    if is_mult and is_halving_step:
        # Require supported termination comparison (> 0, >= 1, > 1)
        comp = loop.bound.comparison_op
        if comp not in (">", ">="):
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNKNOWN_BOUND.value],
                None,
            )
        target = loop.bound.variable or loop.update.variable
        if target:
            if target in param_names or target in cin_vars:
                return (
                    ComplexityClass.logarithmic(),
                    Exactness.STRUCTURAL,
                    [ReasonCode.LOGARITHMIC_UPDATE.value],
                    target,
                )
            if target in named_constants:
                return (
                    ComplexityClass.logarithmic(),
                    Exactness.STRUCTURAL,
                    [
                        ReasonCode.LOGARITHMIC_UPDATE.value,
                        ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value,
                    ],
                    target,
                )
            if target in local_constants:
                return (
                    ComplexityClass.constant(),
                    Exactness.STRUCTURAL,
                    [ReasonCode.CONSTANT_BOUND_LOOP.value],
                    None,
                )

        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNKNOWN_BOUND.value],
            None,
        )

    # 7. Linear loops (++i, --i, += c, -= c with literal nonzero c)
    if loop.update.kind == UpdateKind.ADDITIVE:
        if loop.bound.comparison_op not in ("<", "<=", ">", ">="):
            return (
                ComplexityClass.unknown(),
                Exactness.UNKNOWN,
                [ReasonCode.UNKNOWN_BOUND.value],
                None,
            )

        b_expr = loop.bound.bound_expr
        if b_expr:
            if b_expr.isdigit() or (b_expr in local_constants):
                return (
                    ComplexityClass.constant(),
                    Exactness.STRUCTURAL,
                    [ReasonCode.CONSTANT_BOUND_LOOP.value],
                    None,
                )
            if b_expr in named_constants:
                return (
                    ComplexityClass.linear(),
                    Exactness.STRUCTURAL,
                    [
                        ReasonCode.LINEAR_LOOP.value,
                        ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value,
                    ],
                    b_expr,
                )
            if (
                b_expr in param_names
                or b_expr in cin_vars
                or loop.bound.kind == BoundKind.LINEAR_INPUT
            ):
                return (
                    ComplexityClass.linear(),
                    Exactness.STRUCTURAL,
                    [ReasonCode.LINEAR_LOOP.value],
                    b_expr,
                )

        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNKNOWN_BOUND.value],
            None,
        )

    return (
        ComplexityClass.unknown(),
        Exactness.UNKNOWN,
        [ReasonCode.UNKNOWN_BOUND.value],
        None,
    )
