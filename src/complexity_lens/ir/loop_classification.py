"""
loop_classification.py — Shared deterministic structural loop bound classification.

Classifies a LoopIR into one of 5 mutually exclusive structural categories:
- LINEAR_BOUND: Unit/constant-step progression to an input-size symbol or container size.
- CONST_BOUND: Literal constant bound.
- LOG_UPDATE: Doubling or halving update.
- DEPENDENT_BOUND: Bound depends on an enclosing loop variable.
- UNKNOWN_BOUND: Unresolved or unsupported bound kind.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

from complexity_lens.ir.model import BoundKind, LoopIR, LoopKind, UpdateKind

if TYPE_CHECKING:
    pass


class LoopBoundCategory(str, Enum):
    LINEAR_BOUND = "linear_bound"
    CONST_BOUND = "const_bound"
    LOG_UPDATE = "log_update"
    DEPENDENT_BOUND = "dependent_bound"
    UNKNOWN_BOUND = "unknown_bound"


def classify_loop_structural(
    loop: LoopIR,
    local_constants: dict[str, int] | None = None,
    named_constants: set[str] | None = None,
    cin_vars: set[str] | None = None,
    param_names: set[str] | None = None,
) -> LoopBoundCategory:
    """Classify a single loop into a structural bound category without using engine models."""
    local_constants = local_constants or {}
    named_constants = named_constants or set()
    cin_vars = cin_vars or set()
    param_names = param_names or set()

    # 1. Loop variable modified in body -> UNKNOWN
    if loop.loop_var_modified:
        return LoopBoundCategory.UNKNOWN_BOUND

    # 2. Dependent bound on enclosing loop variable
    if loop.is_dependent_bound or loop.bound.kind == BoundKind.DEPENDENT:
        return LoopBoundCategory.DEPENDENT_BOUND

    # 3. Unsupported comparison operator ('!=') -> UNKNOWN
    if loop.bound.comparison_op == "!=":
        return LoopBoundCategory.UNKNOWN_BOUND

    # 4. Range-for loops
    if loop.kind == LoopKind.RANGE_FOR:
        if loop.is_fixed_size_array:
            return LoopBoundCategory.CONST_BOUND
        if loop.container_name and (
            loop.container_name in param_names or loop.container_name in cin_vars
        ):
            return LoopBoundCategory.LINEAR_BOUND
        return LoopBoundCategory.UNKNOWN_BOUND

    # 5. Multiplicative updates (doubling / halving)
    is_mult = loop.update.kind == UpdateKind.MULTIPLICATIVE
    is_doubling_step = (loop.update.step_op in ("*=", ">>=")) or (
        bool(loop.update.step_expr and "*" in loop.update.step_expr)
    )
    is_halving_step = (loop.update.step_op == "/=") or (
        bool(loop.update.step_expr and "/" in loop.update.step_expr)
    )

    if is_mult and is_doubling_step:
        if loop.bound.comparison_op not in ("<", "<="):
            return LoopBoundCategory.UNKNOWN_BOUND
        if loop.init_val is None or loop.init_val == 0:
            return LoopBoundCategory.UNKNOWN_BOUND
        return LoopBoundCategory.LOG_UPDATE

    if is_mult and is_halving_step:
        comp = loop.bound.comparison_op
        if comp not in (">", ">="):
            return LoopBoundCategory.UNKNOWN_BOUND
        return LoopBoundCategory.LOG_UPDATE

    # 6. Additive linear loops (++i, --i, += c, -= c)
    if loop.update.kind == UpdateKind.ADDITIVE:
        is_inc = loop.update.step_op in ("++", "+=")
        is_dec = loop.update.step_op in ("--", "-=")

        if not is_inc and not is_dec:
            return LoopBoundCategory.UNKNOWN_BOUND

        comp = loop.bound.comparison_op
        if comp not in ("<", "<=", ">", ">="):
            return LoopBoundCategory.UNKNOWN_BOUND

        b_expr = loop.bound.bound_expr or ""
        init_expr = loop.init_expr or ""

        def _is_const(expr: str, val: int | None) -> bool:
            if val is not None:
                return True
            if not expr:
                return False
            return expr.isdigit() or (expr in local_constants)

        def _find_symbol(expr: str) -> str | None:
            if not expr:
                return None
            if expr in param_names or expr in cin_vars:
                return expr
            for p in param_names:
                if re.search(rf"\b{re.escape(p)}\b", expr):
                    return p
            for c in cin_vars:
                if re.search(rf"\b{re.escape(c)}\b", expr):
                    return c
            return None

        def _find_named_const(expr: str) -> str | None:
            if not expr:
                return None
            if expr in named_constants:
                return expr
            for nc in named_constants:
                if re.search(rf"\b{re.escape(nc)}\b", expr):
                    return nc
            return None

        # Increasing loops
        if is_inc:
            if comp in (">", ">="):
                # Incompatible direction
                if _is_const(b_expr, None) and _is_const(init_expr, loop.init_val):
                    if loop.init_val is not None:
                        init_v = loop.init_val
                    elif init_expr.isdigit():
                        init_v = int(init_expr)
                    else:
                        init_v = local_constants.get(init_expr)

                    bound_v = (
                        int(b_expr) if b_expr.isdigit() else local_constants.get(b_expr)
                    )
                    if init_v is not None and bound_v is not None:
                        gt_cond = comp == ">" and init_v <= bound_v
                        gte_cond = comp == ">=" and init_v < bound_v
                        if gt_cond or gte_cond:
                            return LoopBoundCategory.CONST_BOUND
                return LoopBoundCategory.UNKNOWN_BOUND

            if _find_symbol(b_expr) or _find_named_const(b_expr):
                return LoopBoundCategory.LINEAR_BOUND

            if _is_const(b_expr, None):
                if _find_symbol(init_expr):
                    return LoopBoundCategory.LINEAR_BOUND
                return LoopBoundCategory.CONST_BOUND

            if loop.bound.kind == BoundKind.LINEAR_INPUT:
                return LoopBoundCategory.LINEAR_BOUND

            return LoopBoundCategory.UNKNOWN_BOUND

        # Decreasing loops
        if is_dec:
            if comp in ("<", "<="):
                # Incompatible direction
                if _is_const(b_expr, None) and _is_const(init_expr, loop.init_val):
                    if loop.init_val is not None:
                        init_v = loop.init_val
                    elif init_expr.isdigit():
                        init_v = int(init_expr)
                    else:
                        init_v = local_constants.get(init_expr)

                    bound_v = (
                        int(b_expr) if b_expr.isdigit() else local_constants.get(b_expr)
                    )
                    if init_v is not None and bound_v is not None:
                        lt_cond = comp == "<" and init_v >= bound_v
                        lte_cond = comp == "<=" and init_v > bound_v
                        if lt_cond or lte_cond:
                            return LoopBoundCategory.CONST_BOUND
                return LoopBoundCategory.UNKNOWN_BOUND

            start_expr = init_expr or loop.bound.variable or loop.update.variable or ""
            if _find_symbol(start_expr) or _find_named_const(start_expr) or _find_symbol(b_expr):
                return LoopBoundCategory.LINEAR_BOUND

            if _is_const(start_expr, loop.init_val) and _is_const(b_expr, None):
                return LoopBoundCategory.CONST_BOUND

            return LoopBoundCategory.UNKNOWN_BOUND

    return LoopBoundCategory.UNKNOWN_BOUND
