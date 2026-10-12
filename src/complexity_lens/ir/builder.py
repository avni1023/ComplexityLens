"""
builder.py — Construct typed Intermediate Representation from ParseResult.

This module consumes the parser output (ParseResult / ASTNode) and transforms
it into the typed ComplexityLens IR (ProgramIR, FunctionIR, LoopIR, CallIR,
STLOperationIR, UnsupportedIR).

Parsing is kept strictly separate from IR construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from complexity_lens.ir.model import (
    BoundKind,
    CallIR,
    FunctionIR,
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

    # 3. Linear input: function parameter or container size() call
    clean_id = bound_str.split(".")[0].strip()
    if clean_id in param_names or "size" in bound_str or "length" in bound_str:
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
            if "+=" in tokens or "-=" in tokens:
                step_op = "+=" if "+=" in tokens else "-="
                return LoopUpdate(
                    kind=UpdateKind.ADDITIVE,
                    variable=loop_var,
                    step_expr=text,
                    step_op=step_op,
                )
            if "*=" in tokens or "/=" in tokens or "<<=" in tokens or ">>=" in tokens:
                step_op = "*=" if "*=" in tokens else "/="
                return LoopUpdate(
                    kind=UpdateKind.MULTIPLICATIVE,
                    variable=loop_var,
                    step_expr=text,
                    step_op=step_op,
                )

        # Check binary expressions like i = i * 2 or i = i + 1
        if any(op in tokens for op in ("*", "/")):
            return LoopUpdate(
                kind=UpdateKind.MULTIPLICATIVE,
                variable=loop_var,
                step_expr=text,
                step_op="*=",
            )
        if any(op in tokens for op in ("+", "-")):
            return LoopUpdate(
                kind=UpdateKind.ADDITIVE,
                variable=loop_var,
                step_expr=text,
                step_op="+=",
            )

    # For while-loops, the update is inside the body
    if body_node and loop_var:
        for node in body_node.all_nodes():
            if node.kind in ("UNARY_OPERATOR", "COMPOUND_ASSIGNMENT_OPERATOR", "BINARY_OPERATOR"):
                toks = node.tokens
                if loop_var in toks or loop_var == node.spelling:
                    text = " ".join(toks)
                    if any(op in toks for op in ("++", "--", "+=", "-=")):
                        return LoopUpdate(
                            kind=UpdateKind.ADDITIVE,
                            variable=loop_var,
                            step_expr=text,
                            step_op="++" if "++" in toks else "+=",
                        )
                    if any(op in toks for op in ("*=", "/=", "*", "/")):
                        return LoopUpdate(
                            kind=UpdateKind.MULTIPLICATIVE,
                            variable=loop_var,
                            step_expr=text,
                            step_op="*=" if "*=" in toks else "/=",
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

    return CallIR(
        id=node_id,
        callee_name=callee,
        arguments=[c.spelling for c in call_node.children if c.spelling],
        span=span,
        is_recursion=is_rec,
        shrink_kind=shrink_kind,
    )


# ---------------------------------------------------------------------------
# AST to IR Body Walker
# ---------------------------------------------------------------------------


class _IRBuilderContext:
    def __init__(self, enclosing_func_name: str, param_names: set[str]) -> None:
        self.enclosing_func_name = enclosing_func_name
        self.param_names = param_names
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

            if loop_kind == LoopKind.FOR:
                # Typically [init, cond, update, body] or subset
                c_list = child.children
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

                # Identify loop variable from init
                if c_list:
                    var_decl = next(
                        (n for n in c_list[0].all_nodes() if n.kind == "VAR_DECL"),
                        None,
                    )
                    if var_decl:
                        loop_var = var_decl.spelling

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

            # Extract bound and update
            bound = _extract_loop_bound(cond_node, loop_var, outer_vars, ctx.param_names)
            update = _extract_loop_update(update_node, loop_var, body_node)
            early_exit = _has_early_exit(body_node)

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
                )
            )

        # 2. Inline Assembly or Unsupported construct
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

        # 3. Call Expression
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

        # 4. Other statements (compound, if, return, binary expr): recurse
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
    """
    program = ProgramIR(
        source_path=parse_result.source_path,
        has_syntax_errors=parse_result.has_fatal_error,
        diagnostics=[
            d.message
            for d in parse_result.diagnostics
            if d.severity in ("error", "fatal")
        ],
    )

    source_stem = Path(parse_result.source_path).name

    # Identify user-defined functions belonging to the source file
    user_functions: list[ASTNode] = []
    for func in parse_result.functions():
        loc_file = func.loc.file
        # Ignore functions from standard headers
        is_src = (
            loc_file in (parse_result.source_path, "<string>.cpp")
            or Path(loc_file).name == source_stem
        )
        if is_src:
            # Must have a body (COMPOUND_STMT) to be a definition, not just a declaration
            has_body = any(c.kind == "COMPOUND_STMT" for c in func.children)
            if has_body or not func.children:
                user_functions.append(func)

    # Process each user function
    func_counter = 0
    for f_node in user_functions:
        func_counter += 1
        f_name = f_node.spelling
        ret_type = f_node.type_spelling or "void"

        # Parameters
        parameters: list[tuple[str, str]] = []
        param_names: set[str] = set()
        for p in f_node.children:
            if p.kind == "PARM_DECL":
                p_type = p.type_spelling or "auto"
                p_name = p.spelling
                parameters.append((p_type, p_name))
                if p_name:
                    param_names.add(p_name)

        ctx = _IRBuilderContext(enclosing_func_name=f_name, param_names=param_names)

        # Walk body
        body_items = _build_body_items(f_node, ctx, current_depth=0, outer_vars=set())

        # Collect recursion call sites
        all_calls: list[CallIR] = []
        for item in body_items:
            if isinstance(item, CallIR):
                all_calls.append(item)
            elif isinstance(item, LoopIR):
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
        )
        program.functions.append(function_ir)

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
    has_syntax_errors = parse_result.has_fatal_error or any(
        d.severity in ("error", "fatal") for d in user_diags
    )
    if has_syntax_errors:
        program.has_syntax_errors = True
        for diag in user_diags:
            if diag.severity in ("error", "fatal"):
                span = SourceSpan(
                    file=parse_result.source_path,
                    start_line=diag.loc.line if diag.loc else 1,
                    start_col=diag.loc.col if diag.loc else 1,
                    end_line=diag.loc.line if diag.loc else 1,
                    end_col=diag.loc.col if diag.loc else 1,
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
    """Convenience: parse in-memory C++ source string and build ProgramIR."""
    from complexity_lens.parsers.cpp_libclang import parse_source

    return build_ir(parse_source(source, filename=filename, extra_args=extra_args))
