"""
extractor.py — Deterministic structural feature extraction for ComplexityLens.

Extracts the authoritative 26-feature vector from a ProgramIR without reading
complexity engine predictions, models, confidence, or exactness.
"""

from __future__ import annotations

import re
from typing import Any

from complexity_lens.complexity.call_graph import build_call_graph, compute_sccs
from complexity_lens.features.schema import FEATURE_NAMES, FeatureVector
from complexity_lens.ir.container_classification import (
    is_hash_container,
    is_ordered_container,
)
from complexity_lens.ir.entry_point import resolve_entry_function
from complexity_lens.ir.loop_classification import (
    LoopBoundCategory,
    classify_loop_structural,
)
from complexity_lens.ir.model import (
    BranchIR,
    CallIR,
    LoopIR,
    LoopKind,
    ProgramIR,
    STLOperationIR,
    STLOperationKind,
)


def _collect_descendant_loops(loop: LoopIR) -> list[LoopIR]:
    """Collect all descendant loops strictly nested inside a loop's body."""
    descendants: list[LoopIR] = []
    for item in loop.body:
        if isinstance(item, LoopIR):
            descendants.append(item)
            descendants.extend(_collect_descendant_loops(item))
        elif isinstance(item, BranchIR):
            descendants.extend(_collect_branch_loops(item))
    return descendants


def _collect_branch_loops(branch: BranchIR) -> list[LoopIR]:
    """Collect loops inside both branches of a conditional construct."""
    loops: list[LoopIR] = []
    for item in branch.then_body + branch.else_body:
        if isinstance(item, LoopIR):
            loops.append(item)
            loops.extend(_collect_descendant_loops(item))
        elif isinstance(item, BranchIR):
            loops.extend(_collect_branch_loops(item))
    return loops


def _collect_nodes_with_depth(
    items: list[Any],
    current_loop_depth: int = 0,
) -> list[tuple[Any, int]]:
    """Recursively collect AST items along with their enclosing loop nesting depth."""
    result: list[tuple[Any, int]] = []
    for item in items:
        result.append((item, current_loop_depth))
        if isinstance(item, LoopIR):
            result.extend(_collect_nodes_with_depth(item.body, current_loop_depth + 1))
        elif isinstance(item, BranchIR):
            result.extend(_collect_nodes_with_depth(item.then_body, current_loop_depth))
            result.extend(_collect_nodes_with_depth(item.else_body, current_loop_depth))
    return result


def _classify_shrink(call: CallIR, param_name: str | None) -> str:
    """Classify recursive call site into minus_const, divide, or other."""
    if call.shrink_kind == "minus_const":
        return "minus_const"
    if call.shrink_kind == "divide":
        return "divide"

    args = call.arguments or []
    for arg in args:
        clean = arg.strip()
        if "-" in clean:
            parts = clean.split("-", 1)
            base = parts[0].strip()
            c_part = parts[1].strip()
            if param_name and base == param_name and c_part.isdigit() and int(c_part) >= 1:
                return "minus_const"
            if c_part.isdigit() and int(c_part) >= 1:
                return "minus_const"
        if "/" in clean:
            parts = clean.split("/", 1)
            base = parts[0].strip()
            b_part = parts[1].strip()
            if param_name and base == param_name and b_part.isdigit() and int(b_part) >= 2:
                return "divide"
            if b_part.isdigit() and int(b_part) >= 2:
                return "divide"
        if ">>" in clean:
            return "divide"

    if call.shrink_kind == "other":
        return "other"
    return "other"


def extract_features(
    program: ProgramIR,
    entry_function_name: str | None = None,
) -> FeatureVector:
    """Extract authoritative 26-feature vector from a ProgramIR.

    Parameters
    ----------
    program:
        ProgramIR of the translation unit. Must be an instance of ProgramIR.
    entry_function_name:
        Optional explicit function name to treat as entry. If None, resolves
        via standard precedence (main -> single function -> all functions).

    Returns
    -------
    FeatureVector with exactly 26 features.

    Raises
    ------
    TypeError:
        If program is not a ProgramIR or entry_function_name is not str | None.
    """
    if not isinstance(program, ProgramIR):
        raise TypeError(f"Expected ProgramIR, got {type(program).__name__}")
    if entry_function_name is not None and not isinstance(entry_function_name, str):
        tname = type(entry_function_name).__name__
        raise TypeError(f"Expected str or None for entry_function_name, got {tname}")

    # 1. Entry function resolution and reachability
    entry_func, _ = resolve_entry_function(program, entry_function_name)

    graph = build_call_graph(program)
    user_func_map = {f.name: f for f in program.functions}
    user_func_names = set(user_func_map.keys())

    if entry_func is not None:
        # Reachable functions via BFS
        reachable_names: set[str] = set()
        queue = [entry_func.name]
        while queue:
            curr = queue.pop(0)
            if curr not in reachable_names:
                reachable_names.add(curr)
                for callee in graph.get(curr, set()):
                    if callee not in reachable_names and callee in user_func_names:
                        queue.append(callee)
        reachable_functions = [f for f in program.functions if f.name in reachable_names]
    else:
        # No entry function: aggregate over all functions
        reachable_functions = list(program.functions)
        reachable_names = set(user_func_names)

    # 2. Call Graph & Recursion Analysis
    sccs = compute_sccs(graph)
    scc_by_name: dict[str, list[str]] = {}
    for scc in sccs:
        for fname in scc:
            scc_by_name[fname] = scc

    recursive_funcs: set[str] = set()
    for f in program.functions:
        is_rec = f.is_recursive or any(c.is_recursion for c in f.all_calls())
        # Multi-node SCC indicates mutual recursion
        scc = scc_by_name.get(f.name, [f.name])
        if is_rec or len(scc) > 1:
            recursive_funcs.add(f.name)

    # Recursion among reachable functions
    reachable_recursive = [f for f in reachable_functions if f.name in recursive_funcs]
    has_recursion = 1 if len(reachable_recursive) > 0 else 0

    # 3. Interprocedural Loop Depth Calculation
    # Intrinsic loop depth for each function
    intrinsic_depths: dict[str, int] = {}
    for f in program.functions:
        loops = f.all_loops()
        intrinsic_depths[f.name] = max((lp.depth for lp in loops), default=0)

    # func_max_depth memoization with cycle protection
    memo_depths: dict[str, int] = {}

    def get_func_max_depth(func_name: str, call_stack: set[str]) -> int:
        if func_name in call_stack or func_name in recursive_funcs:
            # Deterministic conservative policy for recursive cycles:
            # Do not traverse recursive calls to expand loop depth
            return intrinsic_depths.get(func_name, 0)
        if func_name in memo_depths:
            return memo_depths[func_name]

        f = user_func_map.get(func_name)
        if f is None:
            return 0

        best = intrinsic_depths.get(func_name, 0)
        new_stack = call_stack | {func_name}

        # Inspect all calls in f and their loop nesting depths
        nodes_with_depth = _collect_nodes_with_depth(f.body, current_loop_depth=0)
        for node, d in nodes_with_depth:
            if isinstance(node, CallIR) and node.callee_name in user_func_names:
                if node.callee_name not in recursive_funcs and node.callee_name not in call_stack:
                    callee_k = get_func_max_depth(node.callee_name, new_stack)
                    candidate = d + callee_k
                    if candidate > best:
                        best = candidate

        memo_depths[func_name] = best
        return best

    # Overall max_loop_depth
    candidate_depths: list[int] = [0]
    if entry_func is not None:
        candidate_depths.append(get_func_max_depth(entry_func.name, set()))
    else:
        for f in reachable_functions:
            candidate_depths.append(get_func_max_depth(f.name, set()))
    max_loop_depth = max(candidate_depths)

    # 4. Loop Features Aggregation
    n_loops = 0
    n_for = 0
    n_while = 0
    n_sequential_top_loops = 0
    n_nested_loop_pairs = 0
    n_loops_linear_bound = 0
    n_loops_const_bound = 0
    n_loops_log_update = 0
    n_dependent_bound_loops = 0
    n_loops_unknown_bound = 0
    has_early_exit_in_loop = 0

    for f in reachable_functions:
        f_param_names = {p[1] for p in f.parameters if p[1]}
        f_loops = f.all_loops()
        n_loops += len(f_loops)

        # Sequential top loops (nesting depth 1 in function)
        for lp in f_loops:
            if lp.depth == 1:
                n_sequential_top_loops += 1

            if lp.kind in (LoopKind.FOR, LoopKind.RANGE_FOR):
                n_for += 1
            elif lp.kind in (LoopKind.WHILE, LoopKind.DO_WHILE):
                n_while += 1

            # Early exit inside loop
            if lp.has_early_exit:
                has_early_exit_in_loop = 1

            # Ancestor-descendant loop pairs
            descendants = _collect_descendant_loops(lp)
            n_nested_loop_pairs += len(descendants)

            # Bound classification
            cat = classify_loop_structural(
                lp,
                local_constants=f.local_constants,
                named_constants=f.named_constants,
                cin_vars=f.cin_vars,
                param_names=f_param_names,
            )
            if cat == LoopBoundCategory.LINEAR_BOUND:
                n_loops_linear_bound += 1
            elif cat == LoopBoundCategory.CONST_BOUND:
                n_loops_const_bound += 1
            elif cat == LoopBoundCategory.LOG_UPDATE:
                n_loops_log_update += 1
            elif cat == LoopBoundCategory.DEPENDENT_BOUND:
                n_dependent_bound_loops += 1
            else:
                n_loops_unknown_bound += 1

    # 5. Recursion Site Features
    n_recursive_call_sites = 0
    shrink_minus_const = 0
    shrink_divide = 0
    shrink_other = 0

    for f in reachable_functions:
        int_types = ("int", "long", "size_t", "unsigned")
        f_int_param = next(
            (p[1] for p in f.parameters if p[1] and any(t in p[0] for t in int_types)),
            None,
        )
        scc = scc_by_name.get(f.name, [f.name])
        for call in f.all_calls():
            is_rec_call = (
                call.is_recursion
                or call.callee_name == f.name
                or (call.callee_name in recursive_funcs and call.callee_name in scc)
            )
            if is_rec_call:
                n_recursive_call_sites += 1
                sh_kind = _classify_shrink(call, f_int_param)
                if sh_kind == "minus_const":
                    shrink_minus_const += 1
                elif sh_kind == "divide":
                    shrink_divide += 1
                else:
                    shrink_other += 1

    # 6. Calls and Operations
    n_user_calls = 0
    n_sort_calls = 0
    n_linear_search_calls = 0
    n_binary_search_calls = 0
    n_hash_ops = 0
    n_ordered_container_ops = 0
    loop_container_ops: set[str] = set()
    n_input_dependent_ops = 0

    for f in reachable_functions:
        f_param_names = {p[1] for p in f.parameters if p[1]}
        f_input_symbols = f_param_names | set(f.cin_vars) | set(f.named_constants)

        nodes_with_depth = _collect_nodes_with_depth(f.body, current_loop_depth=0)

        for node, loop_depth in nodes_with_depth:
            # Container ops in loops
            if loop_depth > 0:
                if isinstance(node, STLOperationIR):
                    if node.operation in (
                        STLOperationKind.PUSH_BACK,
                        STLOperationKind.INSERT,
                        STLOperationKind.ERASE,
                    ) or node.callee_name in ("push_back", "emplace_back", "insert", "erase"):
                        loop_container_ops.add(node.id)
                elif isinstance(node, CallIR):
                    clean_c = node.callee_name.replace("std::", "").strip()
                    if clean_c in ("push_back", "emplace_back", "insert", "erase"):
                        loop_container_ops.add(node.id)

            # STL operations
            if isinstance(node, STLOperationIR):
                # Sort calls
                if node.operation == STLOperationKind.SORT or "sort" in node.callee_name:
                    n_sort_calls += 1

                # Linear search calls
                clean_name = node.callee_name.replace("std::", "").strip()
                if clean_name in ("find", "count", "find_if") or node.operation in (
                    STLOperationKind.FIND,
                    STLOperationKind.COUNT,
                ):
                    n_linear_search_calls += 1

                # Binary search calls
                if node.operation in (
                    STLOperationKind.BINARY_SEARCH,
                    STLOperationKind.LOWER_BOUND,
                ) or clean_name in ("binary_search", "lower_bound", "upper_bound"):
                    n_binary_search_calls += 1

                # Hash ops and ordered container ops
                if node.operation == STLOperationKind.HASH_LOOKUP:
                    n_hash_ops += 1
                elif node.operation == STLOperationKind.ORDERED_LOOKUP:
                    n_ordered_container_ops += 1
                elif node.container:
                    if is_hash_container(node.container, f):
                        n_hash_ops += 1
                    elif is_ordered_container(node.container, f):
                        n_ordered_container_ops += 1

                # Input-dependent operations
                is_input_dep = False
                for arg in node.arguments:
                    if ".size()" in arg or ".length()" in arg:
                        is_input_dep = True
                        break
                    for sym in f_input_symbols:
                        if re.search(rf"\b{re.escape(sym)}\b", arg):
                            is_input_dep = True
                            break
                    if is_input_dep:
                        break
                if is_input_dep:
                    n_input_dependent_ops += 1

            # User calls
            elif isinstance(node, CallIR):
                if node.callee_name in user_func_names:
                    scc = scc_by_name.get(f.name, [f.name])
                    is_rec = (
                        node.is_recursion
                        or node.callee_name == f.name
                        or (node.callee_name in recursive_funcs and node.callee_name in scc)
                    )
                    if not is_rec and node.callee_name not in recursive_funcs:
                        n_user_calls += 1

                        # Input-dependent user call
                        is_input_dep = False
                        for arg in node.arguments:
                            if ".size()" in arg or ".length()" in arg:
                                is_input_dep = True
                                break
                            for sym in f_input_symbols:
                                if re.search(rf"\b{re.escape(sym)}\b", arg):
                                    is_input_dep = True
                                    break
                            if is_input_dep:
                                break
                        if is_input_dep:
                            n_input_dependent_ops += 1

    n_container_ops_in_loops = len(loop_container_ops)

    # 7. Unsupported Constructs
    n_unsupported_constructs = sum(len(f.all_unsupported()) for f in reachable_functions)
    if entry_func is None:
        n_unsupported_constructs += len(program.unsupported_constructs)

    # Assemble values tuple in exact canonical order
    values = (
        n_loops,
        max_loop_depth,
        n_sequential_top_loops,
        n_nested_loop_pairs,
        n_for,
        n_while,
        n_loops_linear_bound,
        n_loops_const_bound,
        n_loops_log_update,
        n_dependent_bound_loops,
        n_loops_unknown_bound,
        has_recursion,
        n_recursive_call_sites,
        shrink_minus_const,
        shrink_divide,
        shrink_other,
        n_user_calls,
        n_sort_calls,
        n_linear_search_calls,
        n_binary_search_calls,
        n_hash_ops,
        n_ordered_container_ops,
        n_container_ops_in_loops,
        has_early_exit_in_loop,
        n_input_dependent_ops,
        n_unsupported_constructs,
    )

    return FeatureVector(names=FEATURE_NAMES, values=values)
