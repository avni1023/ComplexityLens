"""
recursion.py — Recursion template classification for ComplexityLens (Stage 3).

Implements the four supported recursion templates T1–T4:

T1 — One decrementing recursive call (p-c, c>=1):
    O(1) work → O(n)
    O(n) work → O(n²)
    Other work → UNKNOWN

T2 — One dividing recursive call (p/b, b>=2):
    O(1) work → O(log n)
    O(n) work → O(n)
    Other work → UNKNOWN
    NOTE: exclusive if/else branches count as ONE call per path.

T3 — Two half-size recursive calls (p/2) with linear non-recursive work:
    Result: O(n log n)

T4 — Two or more decrementing recursive calls (p-c) with O(1) non-recursive work:
    Result: EXP

Explicit unsupported patterns return UNKNOWN with descriptive reason codes.

Design principles:
- Base-case guard is required; if missing → UNKNOWN (BASE_CASE_MISSING).
- Recursive calls inside loops → UNKNOWN (RECURSION_IN_LOOP).
- Range-style recursion (lo, hi, mid) → UNKNOWN (RANGE_RECURSION_UNSUPPORTED).
- Mutual recursion → UNKNOWN (MUTUAL_RECURSION).
- Any unmatched pattern → UNKNOWN (UNSUPPORTED_REDUCTION).
- Calls in exclusive if/else branches take the maximum, not the sum (for T2).
- Sequential recursive calls add toward template classification.
"""

from __future__ import annotations

from typing import Any

from complexity_lens.complexity.models import (
    ComplexityClass,
    Exactness,
    ReasonCode,
)
from complexity_lens.ir.model import (
    BranchIR,
    CallIR,
    FunctionIR,
    LoopIR,
)

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def classify_recursion(
    func: FunctionIR,
    non_recursive_class: ComplexityClass,
) -> tuple[ComplexityClass, Exactness, list[str]]:
    """Classify a recursive function using the four supported templates T1–T4.

    Parameters
    ----------
    func:
        The FunctionIR for the function being analyzed. Must have
        ``is_recursive=True``.
    non_recursive_class:
        The complexity class of the function's non-recursive work (all body items
        that are NOT recursive calls, computed by the engine).

    Returns
    -------
    (complexity_class, exactness, reason_codes)
    """
    if not func.is_recursive:
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_REDUCTION.value],
        )

    # 1. Validate parameter: must have exactly one integer-typed parameter
    int_param = _find_size_param(func)
    if int_param is None:
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_REDUCTION.value],
        )

    # 2. Detect range-style recursion (lo, hi, mid pattern in parameter list)
    if _is_range_recursion(func):
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.RANGE_RECURSION_UNSUPPORTED.value],
        )

    # 3. Check for base-case guard
    if not _has_base_case(func, int_param):
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.BASE_CASE_MISSING.value],
        )

    # 4. Check for recursive calls inside loops
    if _has_recursion_in_loop(func.body):
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.RECURSION_IN_LOOP.value],
        )

    # 5. Analyse recursive call structure
    rec_info = _analyse_recursive_calls(func.body, func.name, int_param)
    if rec_info is None:
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_REDUCTION.value],
        )

    call_count, reduction_kind = rec_info
    # reduction_kind: "minus" | "divide" | "unsupported"

    if reduction_kind == "unsupported":
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_REDUCTION.value],
        )

    # 6. Match templates
    work = non_recursive_class

    # T4: Two or more decrementing recursive calls, O(1) work
    if call_count >= 2 and reduction_kind == "minus":
        if work == ComplexityClass.constant():
            return (
                ComplexityClass.exponential(),
                Exactness.STRUCTURAL,
                [ReasonCode.RECURSION_TEMPLATE_T4.value],
            )
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_REDUCTION.value],
        )

    # T3: Exactly two dividing recursive calls on p/2 with linear work
    if call_count == 2 and reduction_kind == "divide_2":
        if work == ComplexityClass.linear():
            return (
                ComplexityClass.linearithmic(),
                Exactness.STRUCTURAL,
                [ReasonCode.RECURSION_TEMPLATE_T3.value],
            )
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_REDUCTION.value],
        )

    # T1: Exactly one decrementing call
    if call_count == 1 and reduction_kind == "minus":
        if work == ComplexityClass.constant():
            return (
                ComplexityClass.linear(),
                Exactness.STRUCTURAL,
                [ReasonCode.RECURSION_TEMPLATE_T1.value],
            )
        if work == ComplexityClass.linear():
            return (
                ComplexityClass.quadratic(),
                Exactness.STRUCTURAL,
                [ReasonCode.RECURSION_TEMPLATE_T1.value],
            )
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_REDUCTION.value],
        )

    # T2: Exactly one dividing call (p/b where b >= 2)
    if call_count == 1 and reduction_kind in ("divide", "divide_2"):
        if work == ComplexityClass.constant():
            return (
                ComplexityClass.logarithmic(),
                Exactness.STRUCTURAL,
                [ReasonCode.RECURSION_TEMPLATE_T2.value],
            )
        if work == ComplexityClass.linear():
            return (
                ComplexityClass.linear(),
                Exactness.STRUCTURAL,
                [ReasonCode.RECURSION_TEMPLATE_T2.value],
            )
        return (
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNSUPPORTED_REDUCTION.value],
        )

    return (
        ComplexityClass.unknown(),
        Exactness.UNKNOWN,
        [ReasonCode.UNSUPPORTED_REDUCTION.value],
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_size_param(func: FunctionIR) -> str | None:
    """Return the first integer parameter name, or None if none found."""
    int_types = {"int", "long", "size_t", "long long", "unsigned int",
                 "unsigned long", "unsigned"}
    for ptype, pname in func.parameters:
        if any(t in ptype for t in int_types):
            return pname
    return None


def _is_range_recursion(func: FunctionIR) -> bool:
    """Return True if the function has lo/hi/mid range-style parameters."""
    range_names = {"lo", "hi", "low", "high", "mid", "left", "right"}
    param_names_lower = {p[1].lower() for p in func.parameters}
    return len(param_names_lower & range_names) >= 2


def _has_base_case(func: FunctionIR, param: str) -> bool:
    """Return True if the function has a detectable base-case guard.

    A base-case guard is a BranchIR at the top level whose condition mentions
    the size parameter and whose then/else body returns without recursing.
    """
    for item in func.body:
        if isinstance(item, BranchIR):
            # Check condition mentions the parameter
            cond = item.condition or ""
            if param in cond:
                # Check that at least one branch has no recursive calls
                then_has_rec = _body_has_recursion(item.then_body, func.name)
                else_has_rec = _body_has_recursion(item.else_body, func.name)
                if not then_has_rec or not else_has_rec:
                    return True
    return False


def _body_has_recursion(body: list[Any], func_name: str) -> bool:
    """Return True if any item in body (recursively) is a recursive call."""
    for item in body:
        if isinstance(item, CallIR) and item.callee_name == func_name and item.is_recursion:
            return True
        if isinstance(item, LoopIR) and _body_has_recursion(item.body, func_name):
            return True
        if isinstance(item, BranchIR):
            if _body_has_recursion(item.then_body, func_name):
                return True
            if _body_has_recursion(item.else_body, func_name):
                return True
    return False


def _has_recursion_in_loop(body: list[Any]) -> bool:
    """Return True if any LoopIR contains a recursive call anywhere inside it."""
    for item in body:
        if isinstance(item, LoopIR):
            if _any_recursive_call(item.body):
                return True
            if _has_recursion_in_loop(item.body):
                return True
        elif isinstance(item, BranchIR):
            if _has_recursion_in_loop(item.then_body):
                return True
            if _has_recursion_in_loop(item.else_body):
                return True
    return False


def _any_recursive_call(body: list[Any]) -> bool:
    """Return True if any item in body is a recursive CallIR."""
    for item in body:
        if isinstance(item, CallIR) and item.is_recursion:
            return True
        if isinstance(item, (LoopIR, BranchIR)):
            sub_body: list[Any] = []
            if isinstance(item, LoopIR):
                sub_body = item.body
            else:
                sub_body = item.then_body + item.else_body
            if _any_recursive_call(sub_body):
                return True
    return False


def _analyse_recursive_calls(
    body: list[Any],
    func_name: str,
    param: str,
) -> tuple[int, str] | None:
    """Analyse the recursive call structure in the function body.

    Returns (call_count_per_path, reduction_kind) or None for unsupported.

    call_count_per_path: max number of recursive calls per execution path.
    reduction_kind: "minus" | "divide" | "unsupported"

    Rules:
    - Sequential recursive calls: count adds.
    - Exclusive if/else recursive calls: take the max per path (not sum).
    - Recursive call inside a loop → caller reports RECURSION_IN_LOOP
      (already handled before this function is called).
    """
    count, kind = _scan_body(body, func_name, param)
    if kind is None and count == 0:
        return (0, "unsupported")
    if kind is None:
        return None
    return (count, kind)


def _scan_body(
    body: list[Any],
    func_name: str,
    param: str,
) -> tuple[int, str | None]:
    """Scan a body list, accumulating recursive call count and reduction kind.

    Returns (total_count, kind) where kind may be None on conflict/unsupported.
    """
    total_count = 0
    combined_kind: str | None = None

    for item in body:
        if isinstance(item, CallIR) and item.callee_name == func_name and item.is_recursion:
            arg_kind = _call_reduction_kind(item, param)
            if arg_kind is None:
                return total_count, None  # unsupported reduction
            if combined_kind is None:
                combined_kind = arg_kind
            elif combined_kind != arg_kind:
                return total_count, None  # conflicting kinds
            total_count += 1

        elif isinstance(item, BranchIR):
            then_count, then_kind = _scan_body(item.then_body, func_name, param)
            else_count, else_kind = _scan_body(item.else_body, func_name, param)

            # Exclusive branches: take max per path
            branch_count = max(then_count, else_count)
            # Reconcile kinds
            branch_kind: str | None = None
            if then_count > 0 and else_count > 0:
                if then_kind == else_kind:
                    branch_kind = then_kind
                else:
                    branch_kind = None  # conflicting
            elif then_count > 0:
                branch_kind = then_kind
            elif else_count > 0:
                branch_kind = else_kind

            if branch_kind is None and branch_count > 0:
                return total_count, None

            if combined_kind is None:
                combined_kind = branch_kind
            elif branch_kind is not None and combined_kind != branch_kind:
                return total_count, None

            total_count += branch_count

        elif isinstance(item, LoopIR):
            # Recursion inside loops should be caught earlier
            pass

    return total_count, combined_kind


def _call_reduction_kind(call: CallIR, param: str) -> str | None:
    """Return the reduction kind for a recursive call argument.

    Returns "minus" if argument is param-c (literal c>=1),
    "divide_2" if argument is param/2,
    "divide" if argument is param/b (literal b>2),
    None if unsupported.
    """
    candidates: list[str] = []
    for a in call.arguments:
        a_clean = a.strip()
        if a_clean == call.callee_name:
            continue
        candidates.append(a_clean)
    if not candidates and call.arguments:
        candidates = [call.arguments[0].strip()]

    for arg in candidates:
        # p-c (c is a positive literal >= 1)
        if "-" in arg:
            parts = arg.split("-", 1)
            base = parts[0].strip()
            c_part = parts[1].strip()
            if base == param and c_part.isdigit() and int(c_part) >= 1:
                return "minus"

        # p/b (b >= 2)
        if "/" in arg:
            parts = arg.split("/", 1)
            base = parts[0].strip()
            b_part = parts[1].strip()
            if base == param and b_part.isdigit():
                b_val = int(b_part)
                if b_val == 2:
                    return "divide_2"
                elif b_val > 2:
                    return "divide"

    # Fallback to shrink_kind if present
    if call.shrink_kind == "minus_const":
        return "minus"
    if call.shrink_kind == "divide":
        for a in candidates:
            if "2" in a:
                return "divide_2"
        return "divide_2"

    return None

