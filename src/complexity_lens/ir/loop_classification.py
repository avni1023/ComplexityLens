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

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class StructuralLoopClassification:
    """Detailed structural loop classification result."""

    category: LoopBoundCategory
    reason: str
    input_symbol: str | None = None
    is_named_constant: bool = False


def classify_loop_structural(
    loop: LoopIR,
    local_constants: dict[str, int] | None = None,
    named_constants: set[str] | None = None,
    cin_vars: set[str] | None = None,
    param_names: set[str] | None = None,
) -> StructuralLoopClassification:
    """Classify a single loop into a structural bound category without using engine models."""
    local_constants = local_constants or {}
    named_constants = named_constants or set()
    cin_vars = cin_vars or set()
    param_names = param_names or set()

    # 1. Loop variable modified in body -> UNKNOWN
    if loop.loop_var_modified:
        return StructuralLoopClassification(
            category=LoopBoundCategory.UNKNOWN_BOUND,
            reason="LOOP_VAR_MODIFIED",
        )

    # 2. Dependent bound on enclosing loop variable
    if loop.is_dependent_bound or loop.bound.kind == BoundKind.DEPENDENT:
        return StructuralLoopClassification(
            category=LoopBoundCategory.DEPENDENT_BOUND,
            reason="DEPENDENT_BOUND",
        )

    # 3. Unsupported comparison operator ('!=') -> UNKNOWN
    if loop.bound.comparison_op == "!=":
        return StructuralLoopClassification(
            category=LoopBoundCategory.UNKNOWN_BOUND,
            reason="UNKNOWN_BOUND",
        )

    # 4. Range-for loops
    if loop.kind == LoopKind.RANGE_FOR:
        if loop.is_fixed_size_array:
            return StructuralLoopClassification(
                category=LoopBoundCategory.CONST_BOUND,
                reason="CONSTANT_BOUND_LOOP",
            )
        if loop.container_name and (
            loop.container_name in param_names or loop.container_name in cin_vars
        ):
            return StructuralLoopClassification(
                category=LoopBoundCategory.LINEAR_BOUND,
                reason="LINEAR_LOOP",
                input_symbol=loop.container_name,
            )
        return StructuralLoopClassification(
            category=LoopBoundCategory.UNKNOWN_BOUND,
            reason="UNKNOWN_BOUND",
        )

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
            return StructuralLoopClassification(
                category=LoopBoundCategory.UNKNOWN_BOUND,
                reason="UNKNOWN_BOUND",
            )
        if loop.init_val is None or loop.init_val == 0:
            return StructuralLoopClassification(
                category=LoopBoundCategory.UNKNOWN_BOUND,
                reason="UNKNOWN_BOUND",
            )
        b_expr = loop.bound.bound_expr
        if b_expr:
            if b_expr.isdigit() or (b_expr in local_constants):
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LOG_UPDATE,
                    reason="CONSTANT_BOUND_LOOP",
                )
            if b_expr in named_constants:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LOG_UPDATE,
                    reason="LOGARITHMIC_UPDATE",
                    input_symbol=b_expr,
                    is_named_constant=True,
                )
            if (
                b_expr in param_names
                or b_expr in cin_vars
                or loop.bound.kind == BoundKind.LINEAR_INPUT
            ):
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LOG_UPDATE,
                    reason="LOGARITHMIC_UPDATE",
                    input_symbol=b_expr,
                )
        return StructuralLoopClassification(
            category=LoopBoundCategory.UNKNOWN_BOUND,
            reason="UNKNOWN_BOUND",
        )

    if is_mult and is_halving_step:
        comp = loop.bound.comparison_op
        if comp not in (">", ">="):
            return StructuralLoopClassification(
                category=LoopBoundCategory.UNKNOWN_BOUND,
                reason="UNKNOWN_BOUND",
            )
        target = loop.bound.variable or loop.update.variable
        if target:
            if target in param_names or target in cin_vars:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LOG_UPDATE,
                    reason="LOGARITHMIC_UPDATE",
                    input_symbol=target,
                )
            if target in named_constants:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LOG_UPDATE,
                    reason="LOGARITHMIC_UPDATE",
                    input_symbol=target,
                    is_named_constant=True,
                )
            if target in local_constants:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LOG_UPDATE,
                    reason="CONSTANT_BOUND_LOOP",
                )
        return StructuralLoopClassification(
            category=LoopBoundCategory.UNKNOWN_BOUND,
            reason="UNKNOWN_BOUND",
        )

    # 6. Additive linear loops (++i, --i, += c, -= c)
    if loop.update.kind == UpdateKind.ADDITIVE:
        is_inc = loop.update.step_op in ("++", "+=")
        is_dec = loop.update.step_op in ("--", "-=")

        if not is_inc and not is_dec:
            return StructuralLoopClassification(
                category=LoopBoundCategory.UNKNOWN_BOUND,
                reason="UNKNOWN_BOUND",
            )

        comp = loop.bound.comparison_op
        if comp not in ("<", "<=", ">", ">="):
            return StructuralLoopClassification(
                category=LoopBoundCategory.UNKNOWN_BOUND,
                reason="UNKNOWN_BOUND",
            )

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
                            return StructuralLoopClassification(
                                category=LoopBoundCategory.CONST_BOUND,
                                reason="CONSTANT_BOUND_LOOP",
                            )
                return StructuralLoopClassification(
                    category=LoopBoundCategory.UNKNOWN_BOUND,
                    reason="UNKNOWN_BOUND",
                )

            sym = _find_symbol(b_expr)
            if sym:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LINEAR_BOUND,
                    reason="LINEAR_LOOP",
                    input_symbol=sym,
                )

            named_c = _find_named_const(b_expr)
            if named_c:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LINEAR_BOUND,
                    reason="LINEAR_LOOP",
                    input_symbol=named_c,
                    is_named_constant=True,
                )

            if _is_const(b_expr, None):
                init_sym = _find_symbol(init_expr)
                if init_sym:
                    return StructuralLoopClassification(
                        category=LoopBoundCategory.LINEAR_BOUND,
                        reason="LINEAR_LOOP",
                        input_symbol=init_sym,
                    )
                return StructuralLoopClassification(
                    category=LoopBoundCategory.CONST_BOUND,
                    reason="CONSTANT_BOUND_LOOP",
                )

            if loop.bound.kind == BoundKind.LINEAR_INPUT:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LINEAR_BOUND,
                    reason="LINEAR_LOOP",
                    input_symbol=b_expr or None,
                )

            return StructuralLoopClassification(
                category=LoopBoundCategory.UNKNOWN_BOUND,
                reason="UNKNOWN_BOUND",
            )

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
                            return StructuralLoopClassification(
                                category=LoopBoundCategory.CONST_BOUND,
                                reason="CONSTANT_BOUND_LOOP",
                            )
                return StructuralLoopClassification(
                    category=LoopBoundCategory.UNKNOWN_BOUND,
                    reason="UNKNOWN_BOUND",
                )

            start_expr = init_expr or loop.bound.variable or loop.update.variable or ""
            start_sym = _find_symbol(start_expr)
            if start_sym:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LINEAR_BOUND,
                    reason="LINEAR_LOOP",
                    input_symbol=start_sym,
                )

            start_named = _find_named_const(start_expr)
            if start_named:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LINEAR_BOUND,
                    reason="LINEAR_LOOP",
                    input_symbol=start_named,
                    is_named_constant=True,
                )

            bound_sym = _find_symbol(b_expr)
            if bound_sym:
                return StructuralLoopClassification(
                    category=LoopBoundCategory.LINEAR_BOUND,
                    reason="LINEAR_LOOP",
                    input_symbol=bound_sym,
                )

            if _is_const(start_expr, loop.init_val) and _is_const(b_expr, None):
                return StructuralLoopClassification(
                    category=LoopBoundCategory.CONST_BOUND,
                    reason="CONSTANT_BOUND_LOOP",
                )

            return StructuralLoopClassification(
                category=LoopBoundCategory.UNKNOWN_BOUND,
                reason="UNKNOWN_BOUND",
            )

    return StructuralLoopClassification(
        category=LoopBoundCategory.UNKNOWN_BOUND,
        reason="UNKNOWN_BOUND",
    )
