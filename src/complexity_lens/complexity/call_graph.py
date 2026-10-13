"""
call_graph.py — Call graph, SCC detection, and safe non-recursive function
substitution for ComplexityLens (Stage 2).

Implements:
- Build a call graph from a ProgramIR (user-defined function calls only).
- Tarjan's SCC algorithm for strongly-connected component detection (deterministic).
- Conservative argument mapping for callee parameters.
- Non-recursive function substitution with depth cap.

Design principles:
- Direct self-recursion is NOT handled here; it is delegated to Stage 3
  recursion-template analysis in recursion.py.
- Mutual recursion (multi-function SCC) → UNKNOWN (MUTUAL_RECURSION).
- Call depth cap: MAX_CALL_DEPTH = 20.
- Argument mapping supports only the six documented forms.
- Preserves UPPER_BOUND exactness through the substitution chain.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

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
    ProgramIR,
)

if TYPE_CHECKING:
    pass

MAX_CALL_DEPTH = 20

# ---------------------------------------------------------------------------
# Argument Mapping
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArgMapping:
    """Describes how a caller argument maps to a callee parameter."""

    kind: str  # "input_symbol", "constant", "size_of", "scaled", "shifted", "unsupported"
    symbol: str | None = None    # resolved input symbol name (if kind in supported types)
    value: int | None = None     # resolved constant value (if kind == "constant")
    scale: int | None = None     # constant multiplier/divisor (kind == "scaled")
    shift: int | None = None     # constant shift offset (kind == "shifted")


def map_argument(
    arg: str,
    param_names: set[str],
    cin_vars: set[str],
    named_constants: set[str],
    local_constants: dict[str, int],
) -> ArgMapping:
    """Map a single caller argument text to an ArgMapping.

    Supported forms:
    - ``n``            → input_symbol(n)
    - ``n + c``        → shifted(n, +c)
    - ``n - c``        → shifted(n, -c)
    - ``c * n``        → scaled(n, c)
    - ``n / c``        → scaled(n, 1/c — represented via ArgMapping.scale as denominator)
    - ``c.size()``     → input_symbol(c)  [mapped to n]
    - literal int      → constant(value)

    Anything else → ArgMapping(kind="unsupported").
    """
    arg = arg.strip()

    # Literal integer
    if arg.isdigit():
        return ArgMapping(kind="constant", value=int(arg))
    if arg in local_constants:
        return ArgMapping(kind="constant", value=local_constants[arg])

    # Direct input symbol
    if arg in param_names or arg in cin_vars or arg in named_constants:
        return ArgMapping(kind="input_symbol", symbol=arg)

    # .size() expression: c.size() → treat as input symbol c
    if arg.endswith(".size()"):
        base = arg[: -len(".size()")].strip()
        if base in param_names or base in cin_vars or base in named_constants:
            return ArgMapping(kind="input_symbol", symbol=base)
        return ArgMapping(kind="unsupported")

    # n + c or n - c
    for op in ("+", "-"):
        if op in arg:
            parts = arg.split(op, 1)
            left = parts[0].strip()
            right = parts[1].strip()
            if (
                left in param_names or left in cin_vars or left in named_constants
            ) and (right.isdigit() or right in local_constants):
                shift_val = int(right) if right.isdigit() else local_constants[right]
                if op == "-":
                    shift_val = -shift_val
                return ArgMapping(kind="shifted", symbol=left, shift=shift_val)

    # c * n
    if "*" in arg:
        parts = arg.split("*", 1)
        left = parts[0].strip()
        right = parts[1].strip()
        for sym_part, const_part in ((left, right), (right, left)):
            if (
                sym_part in param_names
                or sym_part in cin_vars
                or sym_part in named_constants
            ) and (const_part.isdigit() or const_part in local_constants):
                c = int(const_part) if const_part.isdigit() else local_constants[const_part]
                return ArgMapping(kind="scaled", symbol=sym_part, scale=c)

    # n / c
    if "/" in arg:
        parts = arg.split("/", 1)
        left = parts[0].strip()
        right = parts[1].strip()
        if (
            left in param_names or left in cin_vars or left in named_constants
        ) and (right.isdigit() or right in local_constants):
            c = int(right) if right.isdigit() else local_constants[right]
            if c >= 2:
                return ArgMapping(kind="scaled", symbol=left, scale=-c)  # negative = divisor
            return ArgMapping(kind="unsupported")

    return ArgMapping(kind="unsupported")


def substitute_complexity(
    callee_class: ComplexityClass,
    mapping: ArgMapping,
) -> ComplexityClass:
    """Apply argument mapping to a callee's complexity class.

    - If mapping is ``constant``, the callee's input-dependent terms vanish:
      O(n) → O(1), O(n²) → O(1), O(log n) → O(1).
    - If mapping is ``input_symbol``, ``shifted``, or ``scaled`` by positive c ≥ 1,
      the complexity is unchanged in big-O terms.
    - If mapping is ``scaled`` by division (scale < 0, denominator ≥ 2) and callee is O(n),
      result is still O(n) (n/c is O(n)).
    - If mapping is ``unsupported``, return UNKNOWN.
    """
    if mapping.kind == "unsupported":
        return ComplexityClass.unknown()

    if mapping.kind == "constant":
        # Constant argument: any input-dependent term becomes O(1)
        if callee_class.is_unknown or callee_class.is_exp:
            return callee_class
        # All polynomial/log terms reduce to O(1) for constant argument
        return ComplexityClass.constant()

    # input_symbol / shifted / scaled — big-O class is preserved
    return callee_class


# ---------------------------------------------------------------------------
# Call Graph & SCC
# ---------------------------------------------------------------------------


def build_call_graph(program: ProgramIR) -> dict[str, set[str]]:
    """Build a directed call graph among user-defined functions.

    Returns a mapping: function_name → set of callee function_names (user-defined only).
    """
    func_names = {f.name for f in program.functions}
    graph: dict[str, set[str]] = {f.name: set() for f in program.functions}

    for func in program.functions:
        callees = _collect_callees(func.body)
        for callee in callees:
            if callee in func_names and callee != func.name:
                graph[func.name].add(callee)
            elif callee == func.name:
                # Self-recursion: mark but do not add to graph edge
                # (handled by recursion templates in Stage 3)
                pass

    return graph


def _collect_callees(body: list[Any]) -> list[str]:
    """Collect all callee names reachable from a body list."""
    result = []
    for item in body:
        if isinstance(item, CallIR):
            result.append(item.callee_name)
        elif isinstance(item, LoopIR):
            result.extend(_collect_callees(item.body))
        elif isinstance(item, BranchIR):
            result.extend(_collect_callees(item.then_body))
            result.extend(_collect_callees(item.else_body))
    return result


def compute_sccs(graph: dict[str, set[str]]) -> list[list[str]]:
    """Compute strongly connected components using Tarjan's algorithm.

    Returns a list of SCCs in reverse topological order (each SCC is a list of
    function names). The algorithm is deterministic because function names are
    processed in sorted order.

    SCCs are returned in bottom-up order (callees before callers).
    """
    index_counter = [0]
    stack: list[str] = []
    on_stack: dict[str, bool] = {}
    index_map: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index_map[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in sorted(graph.get(v, set())):
            if w not in index_map:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index_map[w])

        if lowlink[v] == index_map[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for node in sorted(graph.keys()):
        if node not in index_map:
            strongconnect(node)

    return sccs


def is_mutual_recursion(scc: list[str], func_name: str) -> bool:
    """Return True if an SCC represents mutual recursion (multi-function cycle)."""
    return len(scc) > 1 and func_name in scc


# ---------------------------------------------------------------------------
# Safe substitution interface
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SubstitutionResult:
    """Result of attempting non-recursive function substitution."""

    complexity_class: ComplexityClass
    exactness: Exactness
    reason_codes: list[str]
    assumptions: list[str]
    notes: list[str]
    succeeded: bool


def safe_substitute_call(
    callee_name: str,
    arguments: list[str],
    program: ProgramIR,
    engine_analyze_fn: Any,  # callable: (FunctionIR, program, ...) -> AnalysisResult
    depth: int,
    visited: set[str],
    sccs_by_name: dict[str, list[str]],
) -> SubstitutionResult:
    """Safely substitute a non-recursive user-function call.

    Parameters
    ----------
    callee_name:
        Name of the function being called.
    arguments:
        Argument texts from the call site.
    program:
        Full program IR for callee lookup.
    engine_analyze_fn:
        The engine's ``analyze_function`` callable (avoids circular import).
    depth:
        Current substitution depth (starts at 0, capped at MAX_CALL_DEPTH).
    visited:
        Set of function names currently on the analysis stack (for cycle detection).
    sccs_by_name:
        Pre-computed mapping from function name to its SCC list.

    Returns
    -------
    SubstitutionResult
    """
    # Depth cap
    if depth >= MAX_CALL_DEPTH:
        return SubstitutionResult(
            complexity_class=ComplexityClass.unknown(),
            exactness=Exactness.UNKNOWN,
            reason_codes=[ReasonCode.CALL_DEPTH_EXCEEDED.value],
            assumptions=[],
            notes=[f"Call depth cap ({MAX_CALL_DEPTH}) exceeded at '{callee_name}'."],
            succeeded=False,
        )

    # Mutual recursion detection
    scc = sccs_by_name.get(callee_name, [callee_name])
    if len(scc) > 1:
        return SubstitutionResult(
            complexity_class=ComplexityClass.unknown(),
            exactness=Exactness.UNKNOWN,
            reason_codes=[ReasonCode.MUTUAL_RECURSION.value],
            assumptions=[],
            notes=[f"Mutual recursion detected involving '{callee_name}'."],
            succeeded=False,
        )

    # Already on the analysis stack (back-edge for non-self recursion)
    if callee_name in visited:
        return SubstitutionResult(
            complexity_class=ComplexityClass.unknown(),
            exactness=Exactness.UNKNOWN,
            reason_codes=[ReasonCode.MUTUAL_RECURSION.value],
            assumptions=[],
            notes=[f"Recursive call cycle detected at '{callee_name}'."],
            succeeded=False,
        )

    # Find callee function IR
    callee_func: FunctionIR | None = None
    for f in program.functions:
        if f.name == callee_name:
            callee_func = f
            break

    if callee_func is None:
        return SubstitutionResult(
            complexity_class=ComplexityClass.unknown(),
            exactness=Exactness.UNKNOWN,
            reason_codes=[ReasonCode.UNSUPPORTED_CALL.value],
            assumptions=[],
            notes=[f"Callee '{callee_name}' not found in program IR."],
            succeeded=False,
        )

    # Self-recursive function: delegate to recursion template (not handled here)
    if callee_func.is_recursive:
        return SubstitutionResult(
            complexity_class=ComplexityClass.unknown(),
            exactness=Exactness.UNKNOWN,
            reason_codes=[ReasonCode.UNSUPPORTED_CALL.value],
            assumptions=[],
            notes=[
                f"Callee '{callee_name}' is recursive; substitution deferred to "
                "recursion-template analysis."
            ],
            succeeded=False,
        )

    # Build argument mappings
    callee_params = [p[1] for p in callee_func.parameters]
    call_args = list(arguments)
    if call_args and call_args[0] == callee_name and len(call_args) == len(callee_params) + 1:
        call_args = call_args[1:]

    if len(call_args) != len(callee_params) and callee_params:
        return SubstitutionResult(
            complexity_class=ComplexityClass.unknown(),
            exactness=Exactness.UNKNOWN,
            reason_codes=[
                ReasonCode.ARGUMENT_MAPPING_UNSUPPORTED.value,
                ReasonCode.UNSUPPORTED_CALL.value,
            ],
            assumptions=[],
            notes=[
                f"Argument count mismatch for '{callee_name}': "
                f"{len(call_args)} provided, {len(callee_params)} expected."
            ],
            succeeded=False,
        )

    # Analyze the callee
    new_visited = visited | {callee_name}
    callee_result = engine_analyze_fn(
        callee_func,
        program=program,
        selection_rationale=f"Substitution of '{callee_name}'.",
        _subst_depth=depth + 1,
        _subst_visited=new_visited,
        _subst_sccs=sccs_by_name,
    )
    callee_class = callee_result.complexity_class
    callee_exact = Exactness(callee_result.exactness)

    # Apply argument substitution to first parameter mapping (single-param simplification)
    # For multi-param functions: apply worst-case over all parameters
    if callee_params and call_args:
        final_class = callee_class
        for arg_text, _param in zip(call_args, callee_params):
            mapping = map_argument(
                arg_text,
                param_names=set(p[1] for p in callee_func.parameters),
                cin_vars=callee_func.cin_vars,
                named_constants=callee_func.named_constants,
                local_constants=callee_func.local_constants,
            )
            if mapping.kind == "unsupported":
                return SubstitutionResult(
                    complexity_class=ComplexityClass.unknown(),
                    exactness=Exactness.UNKNOWN,
                    reason_codes=[
                        ReasonCode.ARGUMENT_MAPPING_UNSUPPORTED.value,
                        ReasonCode.UNSUPPORTED_CALL.value,
                    ],
                    assumptions=[],
                    notes=[
                        f"Argument '{arg_text}' for parameter '{_param}' of "
                        f"'{callee_name}' has unsupported mapping form."
                    ],
                    succeeded=False,
                )
            final_class = substitute_complexity(callee_class, mapping)
    else:
        final_class = callee_class

    notes = list(callee_result.notes)
    notes.append(f"Function '{callee_name}' successfully substituted.")

    return SubstitutionResult(
        complexity_class=final_class,
        exactness=callee_exact,
        reason_codes=[ReasonCode.FUNCTION_SUBSTITUTED.value] + list(callee_result.reason_codes),
        assumptions=list(getattr(callee_result, "assumptions", [])),
        notes=notes,
        succeeded=True,
    )
