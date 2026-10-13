"""
engine.py — Deterministic Static Complexity Engine for ComplexityLens.

Implements the top-level static analysis pipeline:
- Function entry selection according to the 4-stage precedence rules.
- Scoped diagnostic and UnsupportedIR handling (preventing external constructs
  from poisoning well-formed functions while strictly honoring local failures).
- Recursive structural AST/IR traversal over loops, branches, calls, and sequences.
- Structured rule trace recording (rule_id, node_ids, input_classes, output_class).
- Asymptotic class derivation: O(n^p (log n)^q), EXP, and UNKNOWN.
- Exactness derivation (STRUCTURAL, UPPER_BOUND, UNKNOWN).
- Deterministic explanation string and reason code generation.
- Stage 1: STL/container cost model with configurable hash cost_model.
- Stage 2: Safe non-recursive user-function substitution with call-depth cap.
- Stage 3: Conservative recursion template analysis (T1–T4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from complexity_lens.complexity.call_graph import (
    MAX_CALL_DEPTH,
    build_call_graph,
    compute_sccs,
    safe_substitute_call,
)
from complexity_lens.complexity.models import (
    AnalysisResult,
    ComplexityClass,
    Exactness,
    FunctionSelectionError,
    ReasonCode,
    RuleTraceEntry,
    TaxonomyLabel,
)
from complexity_lens.complexity.recursion import classify_recursion
from complexity_lens.complexity.rules import (
    classify_loop,
    dominant,
    is_call_allowlisted,
    multiply,
)
from complexity_lens.complexity.stl_rules import (
    classify_stl_operation,
    validate_cost_model,
)
from complexity_lens.ir.builder import build_ir_from_file, build_ir_from_source
from complexity_lens.ir.model import (
    BranchIR,
    CallIR,
    FunctionIR,
    LoopIR,
    ProgramIR,
    SourceSpan,
    STLOperationIR,
    UnsupportedIR,
)


def _span_encloses(container: SourceSpan, target: SourceSpan) -> bool:
    """Return True if `target` span is wholly contained within `container` span."""
    if container.file and target.file and container.file != target.file:
        return False
    if container.start_line < target.start_line:
        start_ok = True
    elif container.start_line == target.start_line:
        start_ok = container.start_col <= target.start_col
    else:
        start_ok = False

    if container.end_line > target.end_line:
        end_ok = True
    elif container.end_line == target.end_line:
        end_ok = container.end_col >= target.end_col
    else:
        end_ok = False

    return start_ok and end_ok


def select_function(
    program: ProgramIR,
    function_name: str | None = None,
) -> tuple[FunctionIR, str]:
    """Select the function to analyze following the 4-stage precedence rules:

    1. If caller explicitly names a function, analyze that function.
    2. Otherwise, select 'main' when present.
    3. Otherwise, select the only function when exactly one exists.
    4. If there are multiple functions and no 'main', raise FunctionSelectionError.

    Returns
    -------
    (selected_function, selection_rationale)
    """
    avail = [f.name for f in program.functions]

    # 1. Explicit function name requested
    if function_name is not None:
        for f in program.functions:
            if f.name == function_name:
                return f, f"Explicitly selected function '{function_name}'."
        raise FunctionSelectionError(
            f"Requested function '{function_name}' not found in program IR.",
            available_functions=avail,
        )

    # 2. No explicit name: check for 'main'
    for f in program.functions:
        if f.name == "main":
            return f, "Selected entry point 'main'."

    # 3. Exactly one function exists
    if len(program.functions) == 1:
        f = program.functions[0]
        return f, f"Selected sole function '{f.name}'."

    # 4. Multiple functions and no 'main'
    if len(program.functions) > 1:
        raise FunctionSelectionError(
            "Ambiguous function selection: multiple functions present and no 'main' entry point. "
            f"Please specify a function name. Available: {avail}",
            available_functions=avail,
        )

    # 0 functions
    raise FunctionSelectionError(
        "No functions found in program IR to analyze.",
        available_functions=[],
    )


class _AnalysisContext:
    """Internal mutable tracking context for an analysis run."""

    def __init__(self, func: FunctionIR) -> None:
        self.func = func
        self.param_names = {p[1] for p in func.parameters}
        self.rule_trace: list[RuleTraceEntry] = []
        self.reason_codes: list[str] = []
        self.assumptions: list[str] = []
        self.evidence_node_ids: list[str] = []
        self.source_spans: list[SourceSpan] = []
        self.notes: list[str] = []
        self.input_symbols: set[str] = set()

    def add_rule_trace(
        self,
        rule_id: str,
        node_ids: list[str],
        input_classes: list[ComplexityClass],
        output_class: ComplexityClass,
    ) -> None:
        self.rule_trace.append(
            RuleTraceEntry(
                rule_id=rule_id,
                node_ids=list(node_ids),
                input_classes=list(input_classes),
                output_class=output_class,
            )
        )

    def add_reason(self, code: str) -> None:
        if code not in self.reason_codes:
            self.reason_codes.append(code)

    def add_assumption(self, assumption: str) -> None:
        if assumption not in self.assumptions:
            self.assumptions.append(assumption)

    def add_evidence(self, node_id: str, span: SourceSpan | None) -> None:
        if node_id not in self.evidence_node_ids:
            self.evidence_node_ids.append(node_id)
        if span and span not in self.source_spans:
            self.source_spans.append(span)


class ComplexityEngine:
    """Deterministic static complexity analyzer."""

    def __init__(self, cost_model: str = "expected_hash") -> None:
        """Initialize the engine with a hash-container cost model.

        Parameters
        ----------
        cost_model:
            ``"expected_hash"`` (default) — hash operations are O(1) average-case.
            ``"strict_worst_case"`` — hash operations are O(n) worst-case.

        Raises
        ------
        ValueError
            If ``cost_model`` is not a supported value.
        """
        validate_cost_model(cost_model)
        self._cost_model = cost_model

    def analyze_program(
        self,
        program: ProgramIR,
        function_name: str | None = None,
    ) -> AnalysisResult:
        """Analyze a program IR, selecting a function according to entry rules."""
        func, rationale = select_function(program, function_name)
        return self.analyze_function(func, program=program, selection_rationale=rationale)

    def analyze_function(
        self,
        func: FunctionIR,
        program: ProgramIR | None = None,
        selection_rationale: str = "",
        _subst_depth: int = 0,
        _subst_visited: set[str] | None = None,
        _subst_sccs: dict[str, list[str]] | None = None,
    ) -> AnalysisResult:
        """Analyze a single selected FunctionIR."""
        if not selection_rationale:
            selection_rationale = f"Analyzed function '{func.name}'."

        ctx = _AnalysisContext(func)

        # Pre-compute call graph / SCCs if we have a program and are at top level
        subst_sccs: dict[str, list[str]] = {}
        if program is not None and _subst_sccs is None and _subst_depth == 0:
            call_graph = build_call_graph(program)
            sccs = compute_sccs(call_graph)
            for scc in sccs:
                for name in scc:
                    subst_sccs[name] = scc
        elif _subst_sccs is not None:
            subst_sccs = _subst_sccs

        subst_visited: set[str] = _subst_visited or set()

        # ------------------------------------------------------------------
        # 1. Scoped Diagnostic and Unsupported IR Checks
        # ------------------------------------------------------------------
        # A. Check for error-severity diagnostics in program
        if program and program.diagnostics:
            for diag in program.diagnostics:
                if diag.is_error():
                    if diag.span is None:
                        # Cannot establish where error belongs -> UNKNOWN
                        return self._make_unknown_result(
                            func=func,
                            rationale=selection_rationale,
                            reason_code=ReasonCode.INCOMPLETE_IR.value,
                            rule_id="RULE_INCOMPLETE_IR",
                            summary="IR contains error diagnostic with unlocatable source span.",
                            ctx=ctx,
                            notes=[
                                "Error diagnostic present without location; cannot safely continue."
                            ],
                        )
                    if _span_encloses(func.span, diag.span):
                        # Error is inside the analyzed function -> UNKNOWN
                        ctx.source_spans.append(diag.span)
                        return self._make_unknown_result(
                            func=func,
                            rationale=selection_rationale,
                            reason_code=ReasonCode.INCOMPLETE_IR.value,
                            rule_id="RULE_INCOMPLETE_IR",
                            summary=(
                                f"IR error inside '{func.name}': {diag.message}"
                            ),
                            ctx=ctx,
                        )
                    else:
                        # Error is outside the analyzed function -> do not poison, add note
                        ctx.notes.append(
                            f"PARTIAL_IR: Error diagnostic outside function at "
                            f"line {diag.span.start_line}: {diag.message}"
                        )
                        ctx.add_reason(ReasonCode.PARTIAL_IR.value)

        # B. Check for UnsupportedIR nodes inside this function
        unsupported_inside = func.all_unsupported()
        if unsupported_inside:
            node_ids = [u.id for u in unsupported_inside]
            for u in unsupported_inside:
                ctx.add_evidence(u.id, u.span)
            return self._make_unknown_result(
                func=func,
                rationale=selection_rationale,
                reason_code=ReasonCode.UNSUPPORTED_CONSTRUCT.value,
                rule_id="RULE_UNSUPPORTED_CONSTRUCT",
                summary=(
                    f"Function '{func.name}' contains unsupported construct: "
                    f"{unsupported_inside[0].reason}"
                ),
                ctx=ctx,
                evidence_node_ids=node_ids,
            )

        # C. Check for UnsupportedIR nodes outside this function
        if program:
            all_unsupported = program.all_unsupported()
            outside_unsupported = [
                u for u in all_unsupported if not _span_encloses(func.span, u.span)
            ]
            if outside_unsupported:
                ctx.notes.append(
                    f"PARTIAL_IR: {len(outside_unsupported)} unsupported construct(s) "
                    f"detected outside function '{func.name}'."
                )
                ctx.add_reason(ReasonCode.PARTIAL_IR.value)

        # ------------------------------------------------------------------
        # 2. Recursion template analysis (Stage 3)
        # ------------------------------------------------------------------
        if func.is_recursive:
            return self._analyze_recursive_function(
                func, program, selection_rationale, ctx, subst_sccs, subst_visited
            )

        # ------------------------------------------------------------------
        # 3. Structural Traversal of Function Body (non-recursive)
        # ------------------------------------------------------------------
        c_class, exactness = self._analyze_block(
            func.body, ctx, program, subst_sccs, subst_visited, _subst_depth
        )

        # If analysis produced an ordinary class, ensure exactness is set properly
        if c_class.is_unknown:
            exactness = Exactness.UNKNOWN

        # Check multi-symbol input size convention
        if len(ctx.input_symbols) > 1:
            ctx.add_reason(ReasonCode.SINGLE_INPUT_SYMBOL.value)

        # Build deterministic explanation
        explanation = self._format_explanation(
            func=func,
            c_class=c_class,
            exactness=exactness,
            ctx=ctx,
        )

        return AnalysisResult(
            selected_function=func.name,
            selection_rationale=selection_rationale,
            complexity_class=c_class,
            taxonomy_label=c_class.to_taxonomy_label(),
            exactness=exactness.value,
            evidence_node_ids=list(ctx.evidence_node_ids),
            source_spans=list(ctx.source_spans),
            reason_codes=list(ctx.reason_codes),
            assumptions=list(ctx.assumptions),
            explanation=explanation,
            notes=list(ctx.notes),
            rule_trace=list(ctx.rule_trace),
        )

    def _analyze_recursive_function(
        self,
        func: FunctionIR,
        program: ProgramIR | None,
        rationale: str,
        ctx: _AnalysisContext,
        subst_sccs: dict[str, list[str]],
        subst_visited: set[str],
    ) -> AnalysisResult:
        """Classify a recursive function via template matching."""
        ctx.add_evidence(func.id, func.span)

        # Compute non-recursive work (all body items that are NOT recursive calls)
        non_rec_body = _filter_non_recursive(func.body, func.name)
        if non_rec_body:
            non_rec_class, _ = self._analyze_block(
                non_rec_body, ctx, program, subst_sccs, subst_visited, 0
            )
        else:
            non_rec_class = ComplexityClass.constant()

        rec_class, rec_exact, rec_reasons = classify_recursion(func, non_rec_class)

        for r in rec_reasons:
            ctx.add_reason(r)

        ctx.add_rule_trace(
            rule_id=f"RULE_RECURSION_{rec_reasons[0] if rec_reasons else 'UNKNOWN'}",
            node_ids=[func.id],
            input_classes=[non_rec_class],
            output_class=rec_class,
        )

        if rec_class.is_unknown:
            rec_exact = Exactness.UNKNOWN

        explanation = self._format_explanation(
            func=func,
            c_class=rec_class,
            exactness=rec_exact,
            ctx=ctx,
        )

        return AnalysisResult(
            selected_function=func.name,
            selection_rationale=rationale,
            complexity_class=rec_class,
            taxonomy_label=rec_class.to_taxonomy_label(),
            exactness=rec_exact.value,
            evidence_node_ids=list(ctx.evidence_node_ids),
            source_spans=list(ctx.source_spans),
            reason_codes=list(ctx.reason_codes),
            assumptions=list(ctx.assumptions),
            explanation=explanation,
            notes=list(ctx.notes),
            rule_trace=list(ctx.rule_trace),
        )

    # ------------------------------------------------------------------
    # Block & Item Traversal
    # ------------------------------------------------------------------

    def _analyze_block(
        self,
        items: list[Any],
        ctx: _AnalysisContext,
        program: ProgramIR | None,
        subst_sccs: dict[str, list[str]],
        subst_visited: set[str],
        depth: int,
    ) -> tuple[ComplexityClass, Exactness]:
        """Analyze a list of body items sequentially."""
        if not items:
            # Straight-line code with no loops or calls -> O(1)
            ctx.add_reason(ReasonCode.CONSTANT_WORK.value)
            ctx.add_evidence(ctx.func.id, ctx.func.span)
            ctx.add_rule_trace(
                rule_id="RULE_CONSTANT_WORK",
                node_ids=[ctx.func.id],
                input_classes=[],
                output_class=ComplexityClass.constant(),
            )
            return ComplexityClass.constant(), Exactness.STRUCTURAL

        item_results: list[tuple[ComplexityClass, Exactness, str]] = []

        for item in items:
            c, e, nid = self._analyze_single_item(
                item, ctx, program, subst_sccs, subst_visited, depth
            )
            item_results.append((c, e, nid))

        # Compose items sequentially using asymptotic dominance
        current_class, current_exactness, current_id = item_results[0]

        for next_class, next_exactness, next_id in item_results[1:]:
            composed_class = dominant(current_class, next_class)
            ctx.add_rule_trace(
                rule_id="RULE_SEQUENTIAL_COMPOSITION",
                node_ids=[current_id, next_id],
                input_classes=[current_class, next_class],
                output_class=composed_class,
            )
            ctx.add_reason(ReasonCode.SEQUENTIAL_COMPOSITION.value)

            if composed_class.is_unknown:
                current_exactness = Exactness.UNKNOWN
            elif (
                current_exactness == Exactness.UPPER_BOUND
                or next_exactness == Exactness.UPPER_BOUND
            ):
                current_exactness = Exactness.UPPER_BOUND
            else:
                current_exactness = Exactness.STRUCTURAL

            current_class = composed_class
            current_id = next_id

        return current_class, current_exactness

    def _analyze_single_item(
        self,
        item: Any,
        ctx: _AnalysisContext,
        program: ProgramIR | None,
        subst_sccs: dict[str, list[str]],
        subst_visited: set[str],
        depth: int,
    ) -> tuple[ComplexityClass, Exactness, str]:
        """Analyze a single AST/IR statement construct."""
        # 1. Loop construct
        if isinstance(item, LoopIR):
            ctx.add_evidence(item.id, item.span)
            iter_class, iter_exact, iter_reasons, input_sym = classify_loop(
                item,
                local_constants=ctx.func.local_constants,
                named_constants=ctx.func.named_constants,
                cin_vars=ctx.func.cin_vars,
                param_names=ctx.param_names,
            )
            for r in iter_reasons:
                ctx.add_reason(r)
            if input_sym:
                ctx.input_symbols.add(input_sym)

            # Determine iteration rule trace ID
            if iter_class.is_unknown:
                rule_id = "RULE_UNKNOWN_LOOP"
            elif iter_class == ComplexityClass.linear():
                rule_id = "RULE_LINEAR_LOOP"
            elif iter_class == ComplexityClass.logarithmic():
                rule_id = "RULE_LOGARITHMIC_LOOP"
            elif iter_class == ComplexityClass.constant():
                rule_id = "RULE_CONSTANT_BOUND_LOOP"
            else:
                rule_id = "RULE_LOOP_ITERATION"

            ctx.add_rule_trace(
                rule_id=rule_id,
                node_ids=[item.id],
                input_classes=[],
                output_class=iter_class,
            )

            # Recurse into loop body
            if item.body:
                body_class, body_exactness = self._analyze_block(
                    item.body, ctx, program, subst_sccs, subst_visited, depth
                )
            else:
                body_class = ComplexityClass.constant()
                body_exactness = Exactness.STRUCTURAL

            # Multiply iteration count by body cost
            if body_class == ComplexityClass.constant():
                loop_class = iter_class
            else:
                loop_class = multiply(iter_class, body_class)
                ctx.add_reason(ReasonCode.NESTED_LOOP.value)
                ctx.add_rule_trace(
                    rule_id="RULE_NESTED_LOOP",
                    node_ids=[item.id],
                    input_classes=[iter_class, body_class],
                    output_class=loop_class,
                )

            # Exactness
            if loop_class.is_unknown:
                loop_exactness = Exactness.UNKNOWN
            elif item.has_early_exit or body_exactness == Exactness.UPPER_BOUND:
                loop_exactness = Exactness.UPPER_BOUND
            else:
                loop_exactness = Exactness.STRUCTURAL

            return loop_class, loop_exactness, item.id

        # 2. Conditional branch
        if isinstance(item, BranchIR):
            ctx.add_evidence(item.id, item.span)
            then_class, then_exactness = (
                self._analyze_block(item.then_body, ctx, program, subst_sccs, subst_visited, depth)
                if item.then_body
                else (ComplexityClass.constant(), Exactness.STRUCTURAL)
            )
            else_class, else_exactness = (
                self._analyze_block(item.else_body, ctx, program, subst_sccs, subst_visited, depth)
                if item.else_body
                else (ComplexityClass.constant(), Exactness.STRUCTURAL)
            )

            branch_class = dominant(then_class, else_class)
            ctx.add_reason(ReasonCode.IF_ELSE_WORST_CASE.value)
            ctx.add_rule_trace(
                rule_id="RULE_IF_ELSE_WORST_CASE",
                node_ids=[item.id],
                input_classes=[then_class, else_class],
                output_class=branch_class,
            )

            if branch_class.is_unknown:
                branch_exact = Exactness.UNKNOWN
            elif (
                then_exactness == Exactness.UPPER_BOUND
                or else_exactness == Exactness.UPPER_BOUND
            ):
                branch_exact = Exactness.UPPER_BOUND
            else:
                branch_exact = Exactness.STRUCTURAL

            return branch_class, branch_exact, item.id

        # 3. CallIR
        if isinstance(item, CallIR):
            ctx.add_evidence(item.id, item.span)

            # Recursive call — handled at function level (not here)
            if item.is_recursion:
                c = ComplexityClass.constant()
                ctx.add_reason(ReasonCode.CONSTANT_WORK.value)
                ctx.add_rule_trace(
                    rule_id="RULE_RECURSIVE_CALL_PLACEHOLDER",
                    node_ids=[item.id],
                    input_classes=[],
                    output_class=c,
                )
                return c, Exactness.STRUCTURAL, item.id

            # Container constructor handling
            clean_callee = item.callee_name.replace("std::", "").strip()
            if clean_callee in ("vector", "string", "array", "pair", "tuple"):
                return self._classify_constructor(item, clean_callee, ctx)

            # Standard allowlist (O(1)/amortized) check
            is_allow, is_amort = is_call_allowlisted(item.callee_name, item.arguments)
            if is_allow:
                c = ComplexityClass.constant()
                if is_amort:
                    ctx.add_reason(ReasonCode.AMORTIZED_CONSTANT_CALL.value)
                    ctx.add_assumption(ReasonCode.AMORTIZED.value)
                else:
                    ctx.add_reason(ReasonCode.CONSTANT_WORK.value)
                ctx.add_rule_trace(
                    rule_id="RULE_ALLOWLISTED_CALL",
                    node_ids=[item.id],
                    input_classes=[],
                    output_class=c,
                )
                return c, Exactness.STRUCTURAL, item.id

            # Non-recursive user-function substitution (Stage 2)
            if program is not None and depth < MAX_CALL_DEPTH:
                user_func_names = {f.name for f in program.functions}
                if item.callee_name in user_func_names:
                    sub_result = safe_substitute_call(
                        callee_name=item.callee_name,
                        arguments=item.arguments,
                        program=program,
                        engine_analyze_fn=self.analyze_function,
                        depth=depth,
                        visited=subst_visited,
                        sccs_by_name=subst_sccs,
                    )
                    for r in sub_result.reason_codes:
                        ctx.add_reason(r)
                    for a in sub_result.assumptions:
                        ctx.add_assumption(a)
                    for n in sub_result.notes:
                        ctx.notes.append(n)
                    c = sub_result.complexity_class
                    exact = sub_result.exactness
                    rule = (
                        "RULE_FUNCTION_SUBSTITUTED"
                        if sub_result.succeeded
                        else "RULE_UNSUPPORTED_CALL"
                    )
                    ctx.add_rule_trace(
                        rule_id=rule,
                        node_ids=[item.id],
                        input_classes=[],
                        output_class=c,
                    )
                    return c, exact, item.id

            # Unknown call
            c = ComplexityClass.unknown()
            ctx.add_reason(ReasonCode.UNSUPPORTED_CALL.value)
            ctx.add_rule_trace(
                rule_id="RULE_UNSUPPORTED_CALL",
                node_ids=[item.id],
                input_classes=[],
                output_class=c,
            )
            return c, Exactness.UNKNOWN, item.id

        # 4. STLOperationIR (Stage 1)
        if isinstance(item, STLOperationIR):
            ctx.add_evidence(item.id, item.span)
            stl_class, stl_exact, stl_reasons, stl_assumptions = classify_stl_operation(
                op=item,
                cost_model=self._cost_model,
                param_names=ctx.param_names,
                cin_vars=ctx.func.cin_vars,
                named_constants=ctx.func.named_constants,
                local_constants=ctx.func.local_constants,
            )
            for r in stl_reasons:
                ctx.add_reason(r)
            for a in stl_assumptions:
                ctx.add_assumption(a)
            # Record the rule trace for this STL operation
            rule_id = _stl_rule_id(item)
            ctx.add_rule_trace(
                rule_id=rule_id,
                node_ids=[item.id],
                input_classes=[],
                output_class=stl_class,
            )
            return stl_class, stl_exact, item.id

        # 5. UnsupportedIR
        if isinstance(item, UnsupportedIR):
            ctx.add_evidence(item.id, item.span)
            c = ComplexityClass.unknown()
            ctx.add_reason(ReasonCode.UNSUPPORTED_CONSTRUCT.value)
            ctx.add_rule_trace(
                rule_id="RULE_UNSUPPORTED_CONSTRUCT",
                node_ids=[item.id],
                input_classes=[],
                output_class=c,
            )
            return c, Exactness.UNKNOWN, item.id

        # Default fallback
        item_id = getattr(item, "id", ctx.func.id)
        ctx.add_evidence(item_id, getattr(item, "span", None))
        c = ComplexityClass.constant()
        return c, Exactness.STRUCTURAL, item_id

    def _classify_constructor(
        self,
        item: CallIR,
        clean_callee: str,
        ctx: _AnalysisContext,
    ) -> tuple[ComplexityClass, Exactness, str]:
        """Classify a container constructor call."""
        if not item.arguments:
            # Genuinely constant-cost default construction: vector<int> v;
            c = ComplexityClass.constant()
            ctx.add_reason(ReasonCode.CONSTANT_WORK.value)
            ctx.add_rule_trace(
                rule_id="RULE_ALLOWLISTED_CALL",
                node_ids=[item.id],
                input_classes=[],
                output_class=c,
            )
            return c, Exactness.STRUCTURAL, item.id

        if clean_callee in ("vector", "string"):
            size_arg = item.arguments[0]
            if size_arg.isdigit() or (size_arg in ctx.func.local_constants):
                # Literal-sized container: e.g. std::vector<int> v(10); -> O(1)
                c = ComplexityClass.constant()
                ctx.add_reason(ReasonCode.CONSTANT_WORK.value)
                ctx.add_rule_trace(
                    rule_id="RULE_ALLOWLISTED_CALL",
                    node_ids=[item.id],
                    input_classes=[],
                    output_class=c,
                )
                return c, Exactness.STRUCTURAL, item.id
            if size_arg in ctx.param_names or size_arg in ctx.func.cin_vars:
                # Input-sized container: e.g. std::vector<int> v(n); -> O(n)
                c = ComplexityClass.linear()
                ctx.input_symbols.add(size_arg)
                ctx.add_reason(ReasonCode.LINEAR_LOOP.value)
                ctx.add_rule_trace(
                    rule_id="RULE_CONTAINER_CONSTRUCTOR",
                    node_ids=[item.id],
                    input_classes=[],
                    output_class=c,
                )
                return c, Exactness.STRUCTURAL, item.id
            if size_arg in ctx.func.named_constants:
                # Named-constant sized container: e.g. std::vector<int> v(N); -> O(n)
                c = ComplexityClass.linear()
                ctx.input_symbols.add(size_arg)
                ctx.add_reason(ReasonCode.LINEAR_LOOP.value)
                ctx.add_reason(ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value)
                ctx.add_rule_trace(
                    rule_id="RULE_CONTAINER_CONSTRUCTOR",
                    node_ids=[item.id],
                    input_classes=[],
                    output_class=c,
                )
                return c, Exactness.STRUCTURAL, item.id
            # Range constructor or unanalyzable argument -> UNKNOWN
            c = ComplexityClass.unknown()
            ctx.add_reason(ReasonCode.UNSUPPORTED_CALL.value)
            ctx.add_rule_trace(
                rule_id="RULE_UNSUPPORTED_CALL",
                node_ids=[item.id],
                input_classes=[],
                output_class=c,
            )
            return c, Exactness.UNKNOWN, item.id
        else:
            # Other container with arguments -> UNKNOWN
            c = ComplexityClass.unknown()
            ctx.add_reason(ReasonCode.UNSUPPORTED_CALL.value)
            ctx.add_rule_trace(
                rule_id="RULE_UNSUPPORTED_CALL",
                node_ids=[item.id],
                input_classes=[],
                output_class=c,
            )
            return c, Exactness.UNKNOWN, item.id

    # ------------------------------------------------------------------
    # Formatting Helpers
    # ------------------------------------------------------------------

    def _make_unknown_result(
        self,
        func: FunctionIR,
        rationale: str,
        reason_code: str,
        rule_id: str,
        summary: str,
        ctx: _AnalysisContext,
        evidence_node_ids: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> AnalysisResult:
        """Create a deterministic UNKNOWN analysis result."""
        ctx.add_reason(reason_code)
        if notes:
            ctx.notes.extend(notes)

        nodes = evidence_node_ids or ([func.id] if func else [])
        ctx.add_rule_trace(
            rule_id=rule_id,
            node_ids=nodes,
            input_classes=[],
            output_class=ComplexityClass.unknown(),
        )

        explanation = self._format_explanation(
            func=func,
            c_class=ComplexityClass.unknown(),
            exactness=Exactness.UNKNOWN,
            ctx=ctx,
            summary_override=summary,
        )

        return AnalysisResult(
            selected_function=func.name,
            selection_rationale=rationale,
            complexity_class=ComplexityClass.unknown(),
            taxonomy_label=TaxonomyLabel.UNKNOWN.value,
            exactness=Exactness.UNKNOWN.value,
            evidence_node_ids=list(ctx.evidence_node_ids or nodes),
            source_spans=list(ctx.source_spans),
            reason_codes=list(ctx.reason_codes),
            assumptions=list(ctx.assumptions),
            explanation=explanation,
            notes=list(ctx.notes),
            rule_trace=list(ctx.rule_trace),
        )

    def _format_explanation(
        self,
        func: FunctionIR,
        c_class: ComplexityClass,
        exactness: Exactness,
        ctx: _AnalysisContext,
        summary_override: str | None = None,
    ) -> str:
        """Construct a deterministic human-readable explanation."""
        lines = [
            f"Static complexity analysis for function '{func.name}':",
            f"- Complexity Class: {c_class} ({c_class.to_taxonomy_label()})",
            f"- Exactness: {exactness.value}",
            f"- Reason Codes: {', '.join(ctx.reason_codes) if ctx.reason_codes else 'NONE'}",
        ]

        if summary_override:
            lines.append(f"- Summary: {summary_override}")
        else:
            if c_class.is_unknown:
                lines.append(
                    "- Summary: Complexity cannot be safely proven by static rules."
                )
            elif c_class == ComplexityClass.constant():
                lines.append(
                    "- Summary: Function contains straight-line code and/or constant-bounded "
                    "loops running in O(1) time."
                )
            else:
                lines.append(
                    f"- Summary: Function derives asymptotic bound {c_class} via structured "
                    "loop and dominance composition."
                )

        if ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value in ctx.reason_codes:
            lines.append(
                "- Assumption: Compile-time named constant(s) treated as input size symbol "
                "(NAMED_CONSTANT_AS_INPUT_SIZE)."
            )
        if ReasonCode.AMORTIZED_CONSTANT_CALL.value in ctx.reason_codes:
            lines.append(
                "- Assumption: Container push_back/emplace_back operation(s) assumed amortized "
                "O(1) constant time (AMORTIZED)."
            )
        if ReasonCode.HASH_AVERAGE_CASE.value in ctx.assumptions:
            lines.append(
                f"- Assumption: Hash-container operations evaluated under expected average-case "
                f"cost model (HASH_AVERAGE_CASE); cost_model='{self._cost_model}'."
            )
        if ReasonCode.CONTAINER_KIND_FROM_SOURCE_TEXT.value in ctx.assumptions:
            lines.append(
                "- Assumption: Container kind inferred from source-text spelling "
                "(CONTAINER_KIND_FROM_SOURCE_TEXT); this is a best-effort inference, not a "
                "resolved semantic type."
            )
        if ReasonCode.AMORTIZED.value in ctx.assumptions and (
            ReasonCode.AMORTIZED_CONSTANT_CALL.value not in ctx.reason_codes
        ):
            lines.append(
                "- Assumption: Vector insert/erase operation(s) use amortized cost model "
                "(AMORTIZED)."
            )
        if ReasonCode.SINGLE_INPUT_SYMBOL.value in ctx.reason_codes:
            syms = ", ".join(sorted(ctx.input_symbols))
            lines.append(
                f"- Assumption: Multiple distinct input symbols ({syms}) affect iteration counts; "
                "evaluated under SINGLE_INPUT_SYMBOL assumption (dominant single-variable scaling)."
            )
        if ReasonCode.FUNCTION_SUBSTITUTED.value in ctx.reason_codes:
            lines.append(
                "- Note: One or more user-defined function(s) were safely substituted "
                "(FUNCTION_SUBSTITUTED)."
            )

        if ctx.notes:
            lines.append(f"- Notes: {' | '.join(ctx.notes)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _filter_non_recursive(body: list[Any], func_name: str) -> list[Any]:
    """Return body items that are NOT direct recursive calls."""
    result = []
    for item in body:
        if isinstance(item, CallIR) and item.callee_name == func_name and item.is_recursion:
            continue  # skip recursive calls
        result.append(item)
    return result


def _stl_rule_id(op: STLOperationIR) -> str:
    """Return a rule trace ID for an STL operation."""
    from complexity_lens.ir.model import STLOperationKind
    mapping = {
        STLOperationKind.SORT: "RULE_STL_SORT",
        STLOperationKind.FIND: "RULE_STL_FIND",
        STLOperationKind.COUNT: "RULE_STL_COUNT",
        STLOperationKind.BINARY_SEARCH: "RULE_STL_BINARY_SEARCH",
        STLOperationKind.LOWER_BOUND: "RULE_STL_LOWER_BOUND",
        STLOperationKind.PUSH_BACK: "RULE_VECTOR_PUSH_BACK",
        STLOperationKind.INSERT: "RULE_VECTOR_INSERT",
        STLOperationKind.ERASE: "RULE_VECTOR_ERASE",
        STLOperationKind.HASH_LOOKUP: "RULE_HASH_LOOKUP",
        STLOperationKind.ORDERED_LOOKUP: "RULE_ORDERED_LOOKUP",
    }
    return mapping.get(op.operation, "RULE_UNSUPPORTED_STL")


# ---------------------------------------------------------------------------
# Public Convenience APIs
# ---------------------------------------------------------------------------


def analyze_complexity(
    program: ProgramIR,
    function_name: str | None = None,
    cost_model: str = "expected_hash",
) -> AnalysisResult:
    """Analyze the time complexity of a program IR."""
    engine = ComplexityEngine(cost_model=cost_model)
    return engine.analyze_program(program, function_name=function_name)


def analyze_source(
    source: str,
    function_name: str | None = None,
    filename: str = "main.cpp",
    cost_model: str = "expected_hash",
) -> AnalysisResult:
    """Build IR from C++ source string and analyze time complexity."""
    program = build_ir_from_source(source, filename=filename)
    return analyze_complexity(program, function_name=function_name, cost_model=cost_model)


def analyze_file(
    file_path: str | Path,
    function_name: str | None = None,
    cost_model: str = "expected_hash",
) -> AnalysisResult:
    """Build IR from C++ file and analyze time complexity."""
    program = build_ir_from_file(file_path)
    return analyze_complexity(program, function_name=function_name, cost_model=cost_model)
