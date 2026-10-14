"""
builder.py — Construct typed Intermediate Representation from ParseResult.

This module consumes the parser output (ParseResult / ASTNode) and transforms
it into the typed ComplexityLens IR (ProgramIR, FunctionIR, LoopIR, CallIR,
STLOperationIR, UnsupportedIR).

Parsing is kept strictly separate from IR construction.

All parser calls are centralized through ``parse_file`` / ``parse_source`` in
``complexity_lens.parsers.cpp_libclang``.  No subprocess is launched; libclang
runs in-process.  Process isolation and configurable timeouts are future
hardening planned for the API milestone and are not implemented here.

Input validation
----------------
``build_ir_from_source`` accepts only ``str``.  Call
``complexity_lens.validation.validate_source(source)`` before invoking it when
accepting untrusted input; the validation check is *not* repeated here to avoid
double-rejection and to keep the parser boundary clean.

Container-type provenance
-------------------------
``STLOperationIR.container_kind_source`` is set to:

* ``"resolved"``    — when libclang resolves the declaration type of the
  container (currently not implemented; reserved for future semantic analysis).
* ``"source_text"`` — when the container name is inferred from token / spelling
  source text (best-effort; default for all Day-2/3 detections).
* ``None``          — when no container could be determined.

Fatal failures
--------------
A fatal parse failure never produces a misleading, valid-looking empty IR.
``ProgramIR.has_syntax_errors`` is set to ``True`` and
``ProgramIR.analysis_complete`` is set to ``False``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from complexity_lens.ir.container_classification import detect_container_kind
from complexity_lens.ir.model import (
    BoundKind,
    BranchIR,
    CallIR,
    FunctionIR,
    IRDiagnostic,
    LoopBound,
    LoopIR,
    LoopKind,
    LoopUpdate,
    ProgramIR,
    SourceSpan,
    STLOperationIR,
    STLOperationKind,
    UnsupportedIR,
    UnsupportedKind,
    UpdateKind,
)
from complexity_lens.parsers.cpp_libclang import ASTNode, ParseResult

# ---------------------------------------------------------------------------
# Recognized STL Functions and Containers
# ---------------------------------------------------------------------------

_STL_FUNCTION_MAP: dict[str, STLOperationKind] = {
    "sort": STLOperationKind.SORT,
    "stable_sort": STLOperationKind.SORT,
    "find": STLOperationKind.FIND,
    "find_if": STLOperationKind.FIND,
    "count": STLOperationKind.COUNT,
    "binary_search": STLOperationKind.BINARY_SEARCH,
    "lower_bound": STLOperationKind.LOWER_BOUND,
    "upper_bound": STLOperationKind.LOWER_BOUND,
}

_STL_MEMBER_MAP: dict[str, STLOperationKind] = {
    "push_back": STLOperationKind.PUSH_BACK,
    "emplace_back": STLOperationKind.PUSH_BACK,
    "insert": STLOperationKind.INSERT,
    "erase": STLOperationKind.ERASE,
}


# ---------------------------------------------------------------------------
# Span extraction
# ---------------------------------------------------------------------------


def _span(node: ASTNode) -> SourceSpan:
    return SourceSpan(
        file=node.loc.file,
        start_line=node.extent_start.line,
        start_col=node.extent_start.col,
        end_line=node.extent_end.line,
        end_col=node.extent_end.col,
    )


# ---------------------------------------------------------------------------
# Loop Bound Analysis
# ---------------------------------------------------------------------------


def _extract_loop_bound(
    cond_node: ASTNode | None,
    loop_var: str | None,
    outer_vars: set[str],
    param_names: set[str],
    local_constants: dict[str, int] | None = None,
    named_constants: set[str] | None = None,
    multiply_assigned: set[str] | None = None,
    cin_vars: set[str] | None = None,
) -> LoopBound:
    """Analyze a condition expression to classify the loop termination bound."""
    if not cond_node:
        return LoopBound(kind=BoundKind.UNKNOWN)

    # If the condition node is a BINARY_OPERATOR (e.g. i < n, j < i, i < 10)
    tokens = cond_node.tokens
    comp_ops = {"<", "<=", ">", ">=", "!=", "=="}

    comparison_op: str | None = None
    var_name = loop_var
    bound_expr: str | None = None

    if tokens:
        # Find comparison operator in tokens
        for i, tok in enumerate(tokens):
            if tok in comp_ops:
                comparison_op = tok
                left_tokens = tokens[:i]
                right_tokens = tokens[i + 1:]
                if not var_name and left_tokens:
                    var_name = left_tokens[-1]
                if right_tokens:
                    bound_expr = " ".join(right_tokens)
                break

    # If tokens were not available or did not give comparison_op, inspect children
    if not comparison_op and cond_node.kind == "BINARY_OPERATOR":
        children = cond_node.children
        if len(children) >= 2:
            left_child = children[0]
            right_child = children[1]
            if not var_name:
                var_name = left_child.spelling
            bound_expr = right_child.spelling or "".join(right_child.tokens)

    if not bound_expr:
        # Could not extract bound
        return LoopBound(
            kind=BoundKind.UNKNOWN,
            variable=var_name,
            comparison_op=comparison_op,
        )

    bound_str = bound_expr.strip()

    # Rule: comparison with '!=' yields UNKNOWN
    if comparison_op == "!=":
        return LoopBound(
            kind=BoundKind.UNKNOWN,
            variable=var_name,
            bound_expr=bound_str,
            comparison_op=comparison_op,
        )

    # Classification logic
    # 1. Constant integer literal
    if bound_str.isdigit():
        return LoopBound(
            kind=BoundKind.CONSTANT,
            variable=var_name,
            bound_expr=bound_str,
            comparison_op=comparison_op,
        )

    # 2. Dependent on outer loop variable
    if bound_str in outer_vars:
        return LoopBound(
            kind=BoundKind.DEPENDENT,
            variable=var_name,
            bound_expr=bound_str,
            comparison_op=comparison_op,
        )

    # 3. Local variable with single literal assignment is constant
    if local_constants and bound_str in local_constants:
        return LoopBound(
            kind=BoundKind.CONSTANT,
            variable=var_name,
            bound_expr=bound_str,
            comparison_op=comparison_op,
        )

    # 4. Local variable with multiple assignments -> UNKNOWN
    if multiply_assigned and bound_str in multiply_assigned:
        return LoopBound(
            kind=BoundKind.UNKNOWN,
            variable=var_name,
            bound_expr=bound_str,
            comparison_op=comparison_op,
        )

    # 5. Named compile-time constant (#define N, const int N, constexpr int N)
    if named_constants and bound_str in named_constants:
        return LoopBound(
            kind=BoundKind.LINEAR_INPUT,
            variable=var_name,
            bound_expr=bound_str,
            comparison_op=comparison_op,
        )

    # 6. Linear input: function parameter, cin assigned, or container size() call
    clean_id = bound_str.split(".")[0].strip()
    if (
        clean_id in param_names
        or (cin_vars and clean_id in cin_vars)
        or "size" in bound_str
        or "length" in bound_str
    ):
        return LoopBound(
            kind=BoundKind.LINEAR_INPUT,
            variable=var_name,
            bound_expr=bound_str,
            comparison_op=comparison_op,
        )

    # If it's a simple identifier that isn't outer_var, treat as LINEAR_INPUT symbol
    if bound_str.isidentifier():
        return LoopBound(
            kind=BoundKind.LINEAR_INPUT,
            variable=var_name,
            bound_expr=bound_str,
            comparison_op=comparison_op,
        )

    # Otherwise, explicit UNKNOWN
    return LoopBound(
        kind=BoundKind.UNKNOWN,
        variable=var_name,
        bound_expr=bound_str,
        comparison_op=comparison_op,
    )


# ---------------------------------------------------------------------------
# Loop Update Analysis
# ---------------------------------------------------------------------------


def _extract_loop_update(
    update_node: ASTNode | None,
    loop_var: str | None,
    body_node: ASTNode | None = None,
) -> LoopUpdate:
    """Analyze a loop update expression or body to classify step modification."""
    # First inspect explicit update_node (standard in for-loops)
    if update_node:
        tokens = update_node.tokens
        text = " ".join(tokens) if tokens else update_node.spelling

        # Unary operators: ++i, i++, --i, i--
        if any(op in tokens for op in ("++", "--")) or update_node.kind == "UNARY_OPERATOR":
            step_op = "++" if "++" in text or "++" in tokens else "--"
            return LoopUpdate(
                kind=UpdateKind.ADDITIVE,
                variable=loop_var,
                step_expr=text or step_op,
                step_op=step_op,
            )

        # Compound assignments: +=, -=
        if "+=" in tokens or "-=" in tokens or update_node.kind == "COMPOUND_ASSIGNMENT_OPERATOR":
            op_str = "+=" if "+=" in tokens else "-="
            # Check for doubling pattern i += i
            if op_str == "+=" and loop_var and tokens:
                try:
                    idx = tokens.index("+=")
                    rhs_toks = tokens[idx + 1:]
                    if rhs_toks and rhs_toks[0] == loop_var:
                        return LoopUpdate(
                            kind=UpdateKind.MULTIPLICATIVE,
                            variable=loop_var,
                            step_expr=text,
                            step_op="*=",
                        )
                except ValueError:
                    pass

            if op_str in tokens:
                try:
                    idx = tokens.index(op_str)
                    rhs_toks = tokens[idx + 1:]
                    rhs_str = "".join(rhs_toks).strip()
                    # Nonzero literal constant step
                    if rhs_str.isdigit() and int(rhs_str) != 0:
                        return LoopUpdate(
                            kind=UpdateKind.ADDITIVE,
                            variable=loop_var,
                            step_expr=text,
                            step_op=op_str,
                        )
                    else:
                        # Variable step or zero -> UNKNOWN
                        return LoopUpdate(
                            kind=UpdateKind.UNKNOWN,
                            variable=loop_var,
                            step_expr=text,
                            step_op=op_str,
                        )
                except ValueError:
                    pass

            if "*=" in tokens or "/=" in tokens or "<<=" in tokens or ">>=" in tokens:
                mul_op = "*=" if "*=" in tokens else "/="
                try:
                    op_find = next(op for op in ("*=", "/=", ">>=", "<<=") if op in tokens)
                    idx = tokens.index(op_find)
                    rhs_toks = tokens[idx + 1:]
                    rhs_str = "".join(rhs_toks).strip()
                    if rhs_str.isdigit() and int(rhs_str) >= (1 if op_find == ">>=" else 2):
                        return LoopUpdate(
                            kind=UpdateKind.MULTIPLICATIVE,
                            variable=loop_var,
                            step_expr=text,
                            step_op=mul_op,
                        )
                    else:
                        return LoopUpdate(
                            kind=UpdateKind.UNKNOWN,
                            variable=loop_var,
                            step_expr=text,
                            step_op=mul_op,
                        )
                except ValueError:
                    pass

        # Check binary expressions like i = i * 2 or i = i + 1
        if any(op in tokens for op in ("*", "/")):
            try:
                op_char = "*" if "*" in tokens else "/"
                idx = tokens.index(op_char)
                rhs_toks = tokens[idx + 1:]
                rhs_str = "".join(rhs_toks).strip()
                if rhs_str.isdigit() and int(rhs_str) >= 2:
                    return LoopUpdate(
                        kind=UpdateKind.MULTIPLICATIVE,
                        variable=loop_var,
                        step_expr=text,
                        step_op="*=" if op_char == "*" else "/=",
                    )
                else:
                    return LoopUpdate(
                        kind=UpdateKind.UNKNOWN,
                        variable=loop_var,
                        step_expr=text,
                        step_op="*=" if op_char == "*" else "/=",
                    )
            except ValueError:
                pass

        if any(op in tokens for op in ("+", "-")):
            try:
                op_char = "+" if "+" in tokens else "-"
                idx = tokens.index(op_char)
                rhs_toks = tokens[idx + 1:]
                rhs_str = "".join(rhs_toks).strip()
                if rhs_str.isdigit() and int(rhs_str) != 0:
                    return LoopUpdate(
                        kind=UpdateKind.ADDITIVE,
                        variable=loop_var,
                        step_expr=text,
                        step_op="+=" if op_char == "+" else "-=",
                    )
                else:
                    return LoopUpdate(
                        kind=UpdateKind.UNKNOWN,
                        variable=loop_var,
                        step_expr=text,
                        step_op="+=" if op_char == "+" else "-=",
                    )
            except ValueError:
                pass

    # For while-loops, the update is inside the body
    if body_node and loop_var:
        for node in body_node.all_nodes():
            if node.kind in ("UNARY_OPERATOR", "COMPOUND_ASSIGNMENT_OPERATOR", "BINARY_OPERATOR"):
                toks = node.tokens
                if not toks:
                    continue
                if loop_var in toks or loop_var == node.spelling:
                    text = " ".join(toks)
                    if any(op in toks for op in ("++", "--")):
                        return LoopUpdate(
                            kind=UpdateKind.ADDITIVE,
                            variable=loop_var,
                            step_expr=text,
                            step_op="++" if "++" in toks else "--",
                        )
                    if "+=" in toks or "-=" in toks:
                        op_str = "+=" if "+=" in toks else "-="
                        idx = toks.index(op_str)
                        lhs_toks = toks[:idx]
                        if loop_var in lhs_toks or loop_var == node.spelling:
                            rhs_str = "".join(toks[idx + 1:]).strip()
                            if rhs_str.isdigit() and int(rhs_str) != 0:
                                return LoopUpdate(
                                    kind=UpdateKind.ADDITIVE,
                                    variable=loop_var,
                                    step_expr=text,
                                    step_op=op_str,
                                )
                            elif rhs_str == loop_var and op_str == "+=":
                                return LoopUpdate(
                                    kind=UpdateKind.MULTIPLICATIVE,
                                    variable=loop_var,
                                    step_expr=text,
                                    step_op="*=",
                                )
                            else:
                                return LoopUpdate(
                                    kind=UpdateKind.UNKNOWN,
                                    variable=loop_var,
                                    step_expr=text,
                                    step_op=op_str,
                                )
                    if any(op in toks for op in ("*=", "/=", "*", "/")):
                        op_char = next(op for op in ("*=", "/=", "*", "/") if op in toks)
                        idx = toks.index(op_char)
                        lhs_toks = toks[:idx]
                        if loop_var in lhs_toks or loop_var == node.spelling:
                            rhs_str = "".join(toks[idx + 1:]).strip()
                            if rhs_str.isdigit() and int(rhs_str) >= 2:
                                return LoopUpdate(
                                    kind=UpdateKind.MULTIPLICATIVE,
                                    variable=loop_var,
                                    step_expr=text,
                                    step_op="*=" if "*" in op_char else "/=",
                                )
                            else:
                                return LoopUpdate(
                                    kind=UpdateKind.UNKNOWN,
                                    variable=loop_var,
                                    step_expr=text,
                                    step_op="*=" if "*" in op_char else "/=",
                                )

    return LoopUpdate(
        kind=UpdateKind.UNKNOWN,
        variable=loop_var,
        step_expr=" ".join(update_node.tokens) if (update_node and update_node.tokens) else None,
    )


# ---------------------------------------------------------------------------
# Early exit detection
# ---------------------------------------------------------------------------


def _has_early_exit(body_node: ASTNode | None) -> bool:
    """Return True if loop body contains a break or return statement."""
    if not body_node:
        return False
    LOOP_KINDS = {"FOR_STMT", "WHILE_STMT", "DO_STMT", "CXX_FOR_RANGE_STMT"}

    for n in body_node.all_nodes():
        if n == body_node:
            continue
        # If it's a break statement not inside a nested loop
        if n.kind == "BREAK_STMT":
            return True
        if n.kind == "RETURN_STMT":
            return True
        # Do not descend into nested loops when checking for break
        if n.kind in LOOP_KINDS:
            continue
    return False


# ---------------------------------------------------------------------------
# Call, Recursion & STL Op Analysis
# ---------------------------------------------------------------------------


def _analyze_call(
    call_node: ASTNode,
    enclosing_func_name: str,
    node_id: str,
) -> CallIR | STLOperationIR:
    """Analyze a CALL_EXPR node into either STLOperationIR or CallIR."""
    callee = call_node.spelling
    tokens = call_node.tokens
    span = _span(call_node)

    clean_callee = callee.replace("std::", "").strip()

    # 1. Recognized standalone STL functions: std::sort, std::find, std::count, etc.
    if clean_callee in _STL_FUNCTION_MAP:
        op_kind = _STL_FUNCTION_MAP[clean_callee]
        # Extract container from arguments like v.begin(), v.end()
        container = None
        for child in call_node.children:
            if child.children:
                first_ref = next((c for c in child.all_nodes() if c.kind == "DECL_REF_EXPR"), None)
                if first_ref and first_ref.spelling and first_ref.spelling != callee:
                    container = first_ref.spelling
                    break
        if not container:
            for tok in tokens:
                if (
                    tok.isidentifier()
                    and tok != callee
                    and tok not in (
                        "std",
                        "sort",
                        "find",
                        "count",
                        "begin",
                        "end",
                        "binary_search",
                        "lower_bound",
                        "upper_bound",
                    )
                ):
                    container = tok
                    break

        full_callee = f"std::{clean_callee}"
        return STLOperationIR(
            id=node_id,
            operation=op_kind,
            callee_name=full_callee,
            container=container,
            arguments=[c.spelling for c in call_node.children if c.spelling],
            span=span,
            container_kind_source="source_text" if container else None,
        )

    # 2. Container member methods: push_back, insert, erase
    if clean_callee in _STL_MEMBER_MAP:
        op_kind = _STL_MEMBER_MAP[clean_callee]
        container = None
        first_ref = next((c for c in call_node.all_nodes() if c.kind == "DECL_REF_EXPR"), None)
        if first_ref and first_ref.spelling != callee:
            container = first_ref.spelling
        return STLOperationIR(
            id=node_id,
            operation=op_kind,
            callee_name=clean_callee,
            container=container,
            arguments=[c.spelling for c in call_node.children if c.spelling],
            span=span,
            container_kind_source="source_text" if container else None,
        )

    # 3. Recursion check
    is_rec = (callee == enclosing_func_name)
    shrink_kind = None
    if is_rec:
        # Determine shrink kind from arguments
        arg_tokens = tokens[tokens.index("(") + 1:] if "(" in tokens else tokens
        arg_text = " ".join(arg_tokens)
        if "-" in arg_text or "--" in arg_text:
            shrink_kind = "minus_const"
        elif "/" in arg_text or ">>" in arg_text:
            shrink_kind = "divide"
        else:
            shrink_kind = "other"

    args: list[str] = []
    for c in call_node.children:
        text = (c.spelling or "".join(c.tokens)).strip()
        if text:
            args.append(text)

    return CallIR(
        id=node_id,
        callee_name=callee,
        arguments=args,
        span=span,
        is_recursion=is_rec,
        shrink_kind=shrink_kind,
    )


# ---------------------------------------------------------------------------
# AST to IR Body Walker
# ---------------------------------------------------------------------------


def _detect_loop_var_modification(
    body_node: ASTNode | None,
    target_vars: set[str],
    is_while: bool = False,
) -> bool:
    """Return True if loop variable or bound variable is modified in body."""
    if not body_node or not target_vars:
        return False
    mod_count = 0
    for n in body_node.all_nodes():
        if n == body_node:
            continue
        toks = n.tokens
        if not toks:
            continue
        # Unary operators: ++i, i++, --i, i--
        if n.kind == "UNARY_OPERATOR" and any(op in toks for op in ("++", "--")):
            if any(v in toks for v in target_vars):
                mod_count += 1
        # Compound assignment
        elif n.kind == "COMPOUND_ASSIGNMENT_OPERATOR":
            for op in ("+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="):
                if op in toks:
                    idx = toks.index(op)
                    lhs = toks[:idx]
                    if any(v in lhs for v in target_vars):
                        mod_count += 1
                    break
        # Assignment
        elif (
            n.kind == "BINARY_OPERATOR"
            and "=" in toks
            and not any(cmp_op in toks for cmp_op in ("==", "<=", ">=", "!="))
        ):
            try:
                idx = toks.index("=")
                lhs = toks[:idx]
                if any(v in lhs for v in target_vars):
                    mod_count += 1
            except ValueError:
                pass
        # Streaming from cin
        elif "cin" in toks and ">>" in toks:
            try:
                idx = toks.index(">>")
                rhs = toks[idx + 1:]
                if any(v in rhs for v in target_vars):
                    mod_count += 1
            except ValueError:
                pass

    if is_while:
        # In while loops, exactly 1 modification to the loop var is expected as its step
        return mod_count > 1
    return mod_count > 0


class _IRBuilderContext:
    def __init__(
        self,
        enclosing_func_name: str,
        param_names: set[str],
        local_constants: dict[str, int] | None = None,
        named_constants: set[str] | None = None,
        multiply_assigned: set[str] | None = None,
        cin_vars: set[str] | None = None,
        initial_values: dict[str, int] | None = None,
    ) -> None:
        self.enclosing_func_name = enclosing_func_name
        self.param_names = param_names
        self.local_constants = local_constants or {}
        self.named_constants = named_constants or set()
        self.multiply_assigned = multiply_assigned or set()
        self.cin_vars = cin_vars or set()
        self.initial_values = initial_values or {}
        self.counter = 0

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"


def _build_body_items(
    node: ASTNode,
    ctx: _IRBuilderContext,
    current_depth: int,
    outer_vars: set[str],
) -> list[Any]:
    """Recursively transform AST statement nodes into typed IR body items."""
    items: list[Any] = []
    LOOP_KINDS = {
        "FOR_STMT": LoopKind.FOR,
        "WHILE_STMT": LoopKind.WHILE,
        "DO_STMT": LoopKind.DO_WHILE,
        "CXX_FOR_RANGE_STMT": LoopKind.RANGE_FOR,
    }

    for child in node.children:
        # 1. Loop construct
        if child.kind in LOOP_KINDS:
            loop_kind = LOOP_KINDS[child.kind]
            loop_depth = current_depth + 1
            loop_span = _span(child)
            loop_id = ctx.next_id("loop")

            loop_var: str | None = None
            cond_node: ASTNode | None = None
            update_node: ASTNode | None = None
            body_node: ASTNode | None = None
            init_expr: str | None = None
            init_val: int | None = None
            init_var: str | None = None
            container_name: str | None = None
            is_fixed_array: bool = False

            if loop_kind == LoopKind.FOR:
                c_list = child.children
                _init_node = None
                if len(c_list) >= 4:
                    _init_node, cond_node, update_node, body_node = (
                        c_list[0],
                        c_list[1],
                        c_list[2],
                        c_list[3],
                    )
                elif len(c_list) == 3:
                    _init_node, cond_node, body_node = c_list[0], c_list[1], c_list[2]
                elif len(c_list) == 1:
                    body_node = c_list[0]

                # Identify loop variable and initializer from init
                if _init_node:
                    var_decl = next(
                        (n for n in _init_node.all_nodes() if n.kind == "VAR_DECL"),
                        None,
                    )
                    if var_decl:
                        loop_var = var_decl.spelling
                        init_var = loop_var
                    itoks = _init_node.tokens
                    if "=" in itoks:
                        try:
                            eq_idx = itoks.index("=")
                            if eq_idx + 1 < len(itoks):
                                rhs_toks = itoks[eq_idx + 1:]
                                val_str = "".join(rhs_toks).strip().rstrip(";")
                                init_expr = val_str
                                if val_str.isdigit():
                                    init_val = int(val_str)
                                elif val_str in ctx.local_constants:
                                    init_val = ctx.local_constants[val_str]
                            if eq_idx > 0 and not init_var:
                                init_var = itoks[eq_idx - 1]
                        except ValueError:
                            pass

            elif loop_kind == LoopKind.WHILE:
                c_list = child.children
                if len(c_list) >= 2:
                    cond_node, body_node = c_list[0], c_list[1]
                elif len(c_list) == 1:
                    body_node = c_list[0]

                # Guess loop variable from condition
                if cond_node:
                    first_ref = next(
                        (n for n in cond_node.all_nodes() if n.kind == "DECL_REF_EXPR"),
                        None,
                    )
                    if first_ref:
                        loop_var = first_ref.spelling
                if loop_var:
                    init_var = loop_var
                    if loop_var in ctx.local_constants:
                        init_val = ctx.local_constants[loop_var]
                        init_expr = str(init_val)
                    elif loop_var in ctx.initial_values:
                        init_val = ctx.initial_values[loop_var]
                        init_expr = str(init_val)

            elif loop_kind == LoopKind.DO_WHILE:
                c_list = child.children
                if len(c_list) >= 2:
                    body_node, cond_node = c_list[0], c_list[1]
                elif len(c_list) == 1:
                    body_node = c_list[0]

                # Guess loop variable from condition
                if cond_node:
                    first_ref = next(
                        (n for n in cond_node.all_nodes() if n.kind == "DECL_REF_EXPR"),
                        None,
                    )
                    if first_ref:
                        loop_var = first_ref.spelling
                if loop_var:
                    init_var = loop_var
                    if loop_var in ctx.local_constants:
                        init_val = ctx.local_constants[loop_var]
                        init_expr = str(init_val)
                    elif loop_var in ctx.initial_values:
                        init_val = ctx.initial_values[loop_var]
                        init_expr = str(init_val)

            elif loop_kind == LoopKind.RANGE_FOR:
                c_list = child.children
                if len(c_list) >= 2:
                    body_node = c_list[-1]
                    var_decl = next(
                        (n for n in c_list[0].all_nodes() if n.kind == "VAR_DECL"),
                        None,
                    )
                    if var_decl:
                        loop_var = var_decl.spelling

                # Find container name and fixed-array check
                c_toks = child.tokens
                if ":" in c_toks and ")" in c_toks:
                    try:
                        c_idx = c_toks.index(":")
                        close_idx = c_toks.index(")")
                        range_toks = c_toks[c_idx + 1:close_idx]
                        range_id = "".join(range_toks).strip()
                        if range_id:
                            container_name = range_id
                    except ValueError:
                        pass
                for n in child.all_nodes():
                    if (
                        "[" in n.tokens
                        or "[" in n.type_spelling
                        or "array" in n.type_spelling
                    ):
                        is_fixed_array = True

            # Extract bound and update
            bound = _extract_loop_bound(
                cond_node,
                loop_var,
                outer_vars,
                ctx.param_names,
                local_constants=ctx.local_constants,
                named_constants=ctx.named_constants,
                multiply_assigned=ctx.multiply_assigned,
                cin_vars=ctx.cin_vars,
            )
            update = _extract_loop_update(update_node, loop_var, body_node)
            early_exit = _has_early_exit(body_node)

            # Check loop variable or bound variable modification in body
            target_vars = set()
            if loop_var:
                target_vars.add(loop_var)
            if bound.bound_expr and bound.bound_expr.isidentifier():
                target_vars.add(bound.bound_expr)
            var_modified = _detect_loop_var_modification(
                body_node, target_vars, is_while=(loop_kind in (LoopKind.WHILE, LoopKind.DO_WHILE))
            )
            is_dep = (bound.kind == BoundKind.DEPENDENT)

            # Recurse into loop body
            new_outer = set(outer_vars)
            if loop_var:
                new_outer.add(loop_var)

            inner_items = []
            if body_node:
                inner_items = _build_body_items(body_node, ctx, loop_depth, new_outer)

            items.append(
                LoopIR(
                    id=loop_id,
                    kind=loop_kind,
                    depth=loop_depth,
                    bound=bound,
                    update=update,
                    span=loop_span,
                    has_early_exit=early_exit,
                    body=inner_items,
                    init_expr=init_expr,
                    init_val=init_val,
                    init_var=init_var,
                    loop_var_modified=var_modified,
                    is_dependent_bound=is_dep,
                    container_name=container_name,
                    is_fixed_size_array=is_fixed_array,
                )
            )

        # 2. Conditional branch
        elif child.kind == "IF_STMT":
            branch_id = ctx.next_id("branch")
            c_list = child.children
            cond_expr = "condition"
            then_body: list[Any] = []
            else_body: list[Any] = []
            if c_list:
                cond_node = c_list[0]
                cond_expr = (
                    " ".join(cond_node.tokens)
                    if cond_node.tokens
                    else cond_node.spelling or "condition"
                )
                if len(c_list) >= 2:
                    then_node = c_list[1]
                    if then_node.kind == "COMPOUND_STMT":
                        then_body = _build_body_items(
                            then_node, ctx, current_depth, outer_vars
                        )
                    else:
                        then_body = _build_body_items(
                            ASTNode(
                                kind="COMPOUND_STMT",
                                spelling="",
                                loc=then_node.loc,
                                extent_start=then_node.extent_start,
                                extent_end=then_node.extent_end,
                                depth=then_node.depth,
                                children=[then_node],
                            ),
                            ctx,
                            current_depth,
                            outer_vars,
                        )
                if len(c_list) >= 3:
                    else_node = c_list[2]
                    if else_node.kind == "COMPOUND_STMT":
                        else_body = _build_body_items(
                            else_node, ctx, current_depth, outer_vars
                        )
                    else:
                        else_body = _build_body_items(
                            ASTNode(
                                kind="COMPOUND_STMT",
                                spelling="",
                                loc=else_node.loc,
                                extent_start=else_node.extent_start,
                                extent_end=else_node.extent_end,
                                depth=else_node.depth,
                                children=[else_node],
                            ),
                            ctx,
                            current_depth,
                            outer_vars,
                        )
            items.append(
                BranchIR(
                    id=branch_id,
                    condition=cond_expr,
                    then_body=then_body,
                    else_body=else_body,
                    span=_span(child),
                )
            )

        # 3. Inline Assembly or Unsupported construct
        elif child.kind == "ASM_STMT":
            items.append(
                UnsupportedIR(
                    id=ctx.next_id("unsupported"),
                    kind=UnsupportedKind.ASM_STMT,
                    raw_spelling=" ".join(child.tokens) if child.tokens else "__asm__",
                    reason="Inline assembly is opaque to static analysis",
                    span=_span(child),
                )
            )

        # 4. Call Expression
        elif child.kind == "CALL_EXPR":
            # Filter out trivial iterator calls (.begin, .end) and internal operator overloads
            if (
                child.spelling not in ("begin", "end", "cbegin", "cend")
                and not child.spelling.startswith("operator")
            ):
                call_ir = _analyze_call(child, ctx.enclosing_func_name, ctx.next_id("call"))
                items.append(call_ir)
            # Still visit children for nested calls
            items.extend(_build_body_items(child, ctx, current_depth, outer_vars))

        # 5. Other statements (compound, return, binary expr): recurse
        else:
            items.extend(_build_body_items(child, ctx, current_depth, outer_vars))

    return items


# ---------------------------------------------------------------------------
# Public IR Builder
# ---------------------------------------------------------------------------


def build_ir(parse_result: ParseResult) -> ProgramIR:
    """Construct a typed ProgramIR from a ParseResult.

    Separates parsing completely from IR construction: consumes AST nodes and
    diagnostics, identifies user functions, loop nests, recursion sites,
    and unsupported constructs.

    A fatal parsing failure never produces a misleading, valid-looking empty IR:
    ``has_syntax_errors`` is set to ``True`` and ``analysis_complete`` to
    ``False``.
    """
    source_stem = Path(parse_result.source_path).name

    # ------------------------------------------------------------------
    # Convert raw parser diagnostics to IRDiagnostic objects
    # ------------------------------------------------------------------
    user_diags = [
        d
        for d in parse_result.diagnostics
        if (
            not d.loc
            or not d.loc.file
            or d.loc.file in (parse_result.source_path, "<string>.cpp")
            or Path(d.loc.file).name == source_stem
        )
    ]

    ir_diagnostics: list[IRDiagnostic] = []
    for d in user_diags:
        span = None
        if d.loc:
            span = SourceSpan(
                file=d.loc.file or parse_result.source_path,
                start_line=d.loc.line,
                start_col=d.loc.col,
                end_line=d.loc.line,
                end_col=d.loc.col,
            )
        ir_diagnostics.append(IRDiagnostic(severity=d.severity, message=d.message, span=span))

    has_syntax_errors = parse_result.has_fatal_error or any(
        d.is_error() for d in ir_diagnostics
    )

    program = ProgramIR(
        source_path=parse_result.source_path,
        has_syntax_errors=has_syntax_errors,
        diagnostics=ir_diagnostics,
        analysis_complete=True,  # will be set False below if needed
    )

    # ------------------------------------------------------------------
    # Error diagnostics make the analysis incomplete
    # ------------------------------------------------------------------
    if has_syntax_errors:
        program.analysis_complete = False

    # ------------------------------------------------------------------
    # Identify user-defined functions belonging to the source file
    # ------------------------------------------------------------------
    user_functions: list[ASTNode] = []
    for func in parse_result.functions():
        loc_file = func.loc.file
        is_src = (
            loc_file in (parse_result.source_path, "<string>.cpp")
            or Path(loc_file).name == source_stem
        )
        if is_src:
            # Must have a body (COMPOUND_STMT) to be a definition, not just a declaration
            has_body = any(c.kind == "COMPOUND_STMT" for c in func.children)
            if has_body or not func.children:
                user_functions.append(func)

    # ------------------------------------------------------------------
    # Top-level named constants (#define N, const int N, constexpr int N)
    # ------------------------------------------------------------------
    top_level_named_constants: set[str] = set()
    for n in parse_result.top_level_nodes:
        if n.kind == "VAR_DECL" and n.spelling:
            toks = n.tokens
            if any(k in toks for k in ("const", "constexpr")):
                top_level_named_constants.add(n.spelling)

    if parse_result.source_path and Path(parse_result.source_path).is_file():
        try:
            src_text = Path(parse_result.source_path).read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"#define\s+([A-Za-z_]\w*)\s+(\d+)", src_text):
                top_level_named_constants.add(m.group(1))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Process each user function
    # ------------------------------------------------------------------
    func_counter = 0
    for f_node in user_functions:
        func_counter += 1
        f_name = f_node.spelling
        ret_type = f_node.type_spelling or "void"

        # Parameters
        parameters: list[tuple[str, str]] = []
        param_names: set[str] = set()
        container_types: dict[str, str] = {}
        for p in f_node.children:
            if p.kind == "PARM_DECL":
                p_type = p.type_spelling or "auto"
                p_name = p.spelling
                parameters.append((p_type, p_name))
                if p_name:
                    param_names.add(p_name)
                    ckind = detect_container_kind(p_type)
                    if ckind:
                        container_types[p_name] = ckind

        # Function-level variable analysis
        local_named_constants = set(top_level_named_constants)
        local_constants: dict[str, int] = {}
        initial_values: dict[str, int] = {}
        multiply_assigned: set[str] = set()
        cin_vars: set[str] = set()
        assigned_counts: dict[str, int] = {}

        for n in f_node.all_nodes():
            if n == f_node:
                continue
            toks = n.tokens
            if not toks:
                continue
            if "cin" in toks and ">>" in toks:
                try:
                    idx = toks.index(">>")
                    for tok in toks[idx + 1:]:
                        if tok.isidentifier() and tok not in ("cin", "std"):
                            cin_vars.add(tok)
                            break
                except ValueError:
                    pass
            if n.kind == "VAR_DECL" and n.spelling:
                var_name = n.spelling
                ckind = detect_container_kind(n.type_spelling or " ".join(toks))
                if ckind:
                    container_types[var_name] = ckind
                if any(k in toks for k in ("const", "constexpr")):
                    local_named_constants.add(var_name)
                elif "=" in toks:
                    eq_idx = toks.index("=")
                    if eq_idx + 1 < len(toks) and toks[eq_idx + 1].isdigit():
                        val = int(toks[eq_idx + 1])
                        local_constants[var_name] = val
                        initial_values[var_name] = val
                        assigned_counts[var_name] = assigned_counts.get(var_name, 0) + 1
                else:
                    assigned_counts[var_name] = assigned_counts.get(var_name, 0)
            elif n.kind in ("COMPOUND_ASSIGNMENT_OPERATOR", "UNARY_OPERATOR", "BINARY_OPERATOR"):
                if any(op in toks for op in ("++", "--", "+=", "-=", "*=", "/=", "%=", "=")):
                    for var in list(local_constants.keys()):
                        if var in toks:
                            assigned_counts[var] = assigned_counts.get(var, 0) + 1

        for var, cnt in assigned_counts.items():
            if cnt > 1:
                multiply_assigned.add(var)
                local_constants.pop(var, None)

        ctx = _IRBuilderContext(
            enclosing_func_name=f_name,
            param_names=param_names,
            local_constants=local_constants,
            named_constants=local_named_constants,
            multiply_assigned=multiply_assigned,
            cin_vars=cin_vars,
            initial_values=initial_values,
        )

        # Walk body
        body_items = _build_body_items(f_node, ctx, current_depth=0, outer_vars=set())

        # Collect recursion call sites
        all_calls: list[CallIR] = []
        for item in body_items:
            if isinstance(item, CallIR):
                all_calls.append(item)
            elif isinstance(item, LoopIR):
                all_calls.extend(item.all_calls())
            elif isinstance(item, BranchIR):
                all_calls.extend(item.all_calls())

        rec_sites = [c for c in all_calls if c.is_recursion]
        is_rec = len(rec_sites) > 0 or parse_result.is_recursive(f_node)

        function_ir = FunctionIR(
            id=f"func_{func_counter}",
            name=f_name,
            return_type=ret_type,
            parameters=parameters,
            span=_span(f_node),
            body=body_items,
            is_recursive=is_rec,
            recursive_call_sites=rec_sites,
            local_constants=local_constants,
            named_constants=local_named_constants,
            cin_vars=cin_vars,
            multiply_assigned_vars=multiply_assigned,
            container_types=container_types,
        )
        program.functions.append(function_ir)

        # Any UnsupportedIR node anywhere makes analysis incomplete
        if not program.analysis_complete:
            pass  # already False
        elif function_ir.all_unsupported():
            program.analysis_complete = False

    # ------------------------------------------------------------------
    # Populate top-level syntax-error UnsupportedIR nodes
    # ------------------------------------------------------------------
    for diag in ir_diagnostics:
        if diag.is_error():
            span = diag.span or SourceSpan(
                file=parse_result.source_path,
                start_line=1,
                start_col=1,
                end_line=1,
                end_col=1,
            )
            program.unsupported_constructs.append(
                UnsupportedIR(
                    id=f"syntax_error_{len(program.unsupported_constructs) + 1}",
                    kind=UnsupportedKind.UNKNOWN_SYNTAX,
                    raw_spelling=diag.message,
                    reason=f"Parser syntax error: {diag.message}",
                    span=span,
                )
            )

    # Any top-level unsupported constructs also make analysis incomplete
    if program.unsupported_constructs and program.analysis_complete:
        program.analysis_complete = False

    return program


def build_ir_from_file(
    path: str | Path,
    extra_args: list[str] | None = None,
) -> ProgramIR:
    """Convenience: parse C++ file and build ProgramIR."""
    from complexity_lens.parsers.cpp_libclang import parse_file

    return build_ir(parse_file(path, extra_args=extra_args))


def build_ir_from_source(
    source: str,
    filename: str = "<string>.cpp",
    extra_args: list[str] | None = None,
) -> ProgramIR:
    """Convenience: parse in-memory C++ source string and build ProgramIR.

    Parameters
    ----------
    source:
        Must be a ``str``.  Call ``validate_source(source)`` before this
        function when accepting untrusted input.
    """
    from complexity_lens.parsers.cpp_libclang import parse_source

    return build_ir(parse_source(source, filename=filename, extra_args=extra_args))
