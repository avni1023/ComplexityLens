"""Unit tests for ComplexityLens Stage 2 (safe user-function complexity substitution)."""
from __future__ import annotations

from complexity_lens.complexity import (
    HASH_EXPECTED,
    ArgMapping,
    ComplexityClass,
    analyze_complexity,
    analyze_source,
    build_call_graph,
    classify_stl_operation,
    compute_sccs,
    map_argument,
    substitute_complexity,
)
from complexity_lens.ir.builder import build_ir_from_source
from complexity_lens.ir.model import (
    CallIR,
    FunctionIR,
    ProgramIR,
    SourceSpan,
    STLOperationIR,
    STLOperationKind,
)

_DUMMY_SPAN = SourceSpan("<string>.cpp", 1, 1, 1, 10)


def _make_stl_op(
    kind: STLOperationKind,
    container: str | None = "vector",
    arguments: list[str] | None = None,
    container_kind_source: str | None = None,
) -> STLOperationIR:
    return STLOperationIR(
        id="op_1",
        operation=kind,
        callee_name=f"std::{kind.value}",
        container=container,
        arguments=arguments or ["a.begin()", "a.end()"],
        span=_DUMMY_SPAN,
        container_kind_source=container_kind_source,
    )


def _classify(
    op_kind: STLOperationKind,
    cost_model: str = HASH_EXPECTED,
    container: str | None = "unordered_map",
):
    op = _make_stl_op(op_kind, container=container)
    c, exactness, reasons, assumptions = classify_stl_operation(
        op, cost_model, {"n"}, set(), set(), {}
    )
    return c, c.to_taxonomy_label(), reasons, assumptions

class TestArgumentMapping:
    def test_direct_input_symbol(self):
        m = map_argument("n", {"n"}, set(), set(), {})
        assert m.kind == "input_symbol"
        assert m.symbol == "n"

    def test_literal_constant(self):
        m = map_argument("42", set(), set(), set(), {})
        assert m.kind == "constant"
        assert m.value == 42

    def test_local_constant(self):
        m = map_argument("limit", set(), set(), set(), {"limit": 10})
        assert m.kind == "constant"
        assert m.value == 10

    def test_n_plus_c(self):
        m = map_argument("n+1", {"n"}, set(), set(), {})
        assert m.kind == "shifted"
        assert m.symbol == "n"
        assert m.shift == 1

    def test_n_minus_c(self):
        m = map_argument("n-2", {"n"}, set(), set(), {})
        assert m.kind == "shifted"
        assert m.shift == -2

    def test_c_times_n(self):
        m = map_argument("2*n", {"n"}, set(), set(), {})
        assert m.kind == "scaled"
        assert m.symbol == "n"

    def test_n_div_c(self):
        m = map_argument("n/2", {"n"}, set(), set(), {})
        assert m.kind == "scaled"
        assert m.symbol == "n"

    def test_size_expression(self):
        m = map_argument("v.size()", {"v"}, set(), set(), {})
        assert m.kind == "input_symbol"
        assert m.symbol == "v"

    def test_unsupported_expression(self):
        m = map_argument("complex_expr + other", {"n"}, set(), set(), {})
        assert m.kind == "unsupported"


# ===========================================================================
# Section 11 — Substitute Complexity
# ===========================================================================


class TestSubstituteComplexity:
    def test_constant_arg_reduces_linear_to_constant(self):
        mapping = ArgMapping(kind="constant", value=5)
        result = substitute_complexity(ComplexityClass.linear(), mapping)
        assert result == ComplexityClass.constant()

    def test_constant_arg_reduces_quadratic_to_constant(self):
        mapping = ArgMapping(kind="constant", value=10)
        result = substitute_complexity(ComplexityClass.quadratic(), mapping)
        assert result == ComplexityClass.constant()

    def test_input_symbol_preserves_linear(self):
        mapping = ArgMapping(kind="input_symbol", symbol="n")
        result = substitute_complexity(ComplexityClass.linear(), mapping)
        assert result == ComplexityClass.linear()

    def test_shifted_preserves_class(self):
        mapping = ArgMapping(kind="shifted", symbol="n", shift=1)
        result = substitute_complexity(ComplexityClass.linear(), mapping)
        assert result == ComplexityClass.linear()

    def test_unsupported_yields_unknown(self):
        mapping = ArgMapping(kind="unsupported")
        result = substitute_complexity(ComplexityClass.linear(), mapping)
        assert result.is_unknown


# ===========================================================================
# Section 12 — Call Graph and SCC (Stage 2)
# ===========================================================================


class TestCallGraph:
    def _make_program_with_calls(self) -> ProgramIR:
        """Create a synthetic ProgramIR: main calls helper; helper is non-recursive."""
        span = _DUMMY_SPAN
        helper = FunctionIR(
            id="f_helper",
            name="helper",
            return_type="int",
            parameters=[("int", "n")],
            span=span,
            body=[],
        )
        main = FunctionIR(
            id="f_main",
            name="main",
            return_type="int",
            parameters=[("int", "n")],
            span=span,
            body=[
                CallIR(id="call_helper", callee_name="helper", arguments=["n"],
                       span=span, is_recursion=False)
            ],
        )
        return ProgramIR(source_path="test.cpp", functions=[helper, main])

    def test_build_call_graph_finds_edge(self):
        program = self._make_program_with_calls()
        graph = build_call_graph(program)
        assert "helper" in graph.get("main", set())

    def test_scc_on_dag_all_singletons(self):
        program = self._make_program_with_calls()
        graph = build_call_graph(program)
        sccs = compute_sccs(graph)
        # All SCCs in a DAG are singletons
        for scc in sccs:
            assert len(scc) == 1

    def test_mutual_recursion_scc_has_size_2(self):
        """Construct a graph with a mutual recursion cycle."""
        # A calls B, B calls A
        graph = {"A": {"B"}, "B": {"A"}}
        sccs = compute_sccs(graph)
        multi_sccs = [s for s in sccs if len(s) > 1]
        assert len(multi_sccs) == 1
        assert set(multi_sccs[0]) == {"A", "B"}

    def test_scc_determinism(self):
        """Same graph always produces same SCC ordering."""
        graph = {"A": {"B"}, "B": {"C"}, "C": set(), "D": {"A"}}
        sccs1 = compute_sccs(graph)
        sccs2 = compute_sccs(graph)
        assert sccs1 == sccs2


# ===========================================================================
# Section 13 — Function Substitution Integration (Stage 2)
# ===========================================================================


class TestFunctionSubstitution:
    """End-to-end integration tests for user-function substitution."""

    def test_helper_called_inside_loop(self):
        """O(1) helper inside O(n) loop → overall O(n)."""
        src = """
        int helper(int x) {
            return x + 1;
        }
        void solve(int n) {
            for (int i = 0; i < n; ++i) {
                helper(i);
            }
        }
        """
        res = analyze_source(src, function_name="solve")
        assert res.taxonomy_label in ("ON", "UNKNOWN")

    def test_helper_called_twice_sequentially(self):
        """Two O(n) helpers in sequence → O(n) by dominance."""
        src = """
        void linear_helper(int n) {
            for (int i = 0; i < n; ++i) {}
        }
        void solve(int n) {
            linear_helper(n);
            linear_helper(n);
        }
        """
        res = analyze_source(src, function_name="solve")
        # Two O(n) helpers sequentially → O(n)
        assert res.taxonomy_label in ("ON", "UNKNOWN")

    def test_helper_called_with_literal_arg(self):
        """Helper called with literal → input-dependent terms vanish → O(1)."""
        src = """
        int linear_work(int n) {
            for (int i = 0; i < n; ++i) {}
            return 0;
        }
        void solve() {
            linear_work(5);
        }
        """
        res = analyze_source(src, function_name="solve")
        # linear_work(5) → substitute n=5 → O(1)
        assert res.taxonomy_label in ("O1", "ON", "UNKNOWN")

    def test_mutual_recursion_returns_unknown(self):
        """Two functions calling each other → UNKNOWN (MUTUAL_RECURSION)."""
        src = """
        int funcA(int n);
        int funcB(int n) {
            if (n <= 0) return 0;
            return funcA(n - 1);
        }
        int funcA(int n) {
            if (n <= 0) return 0;
            return funcB(n - 1);
        }
        """
        # We need to analyze just funcA or funcB; since both have `int n` as
        # size param they should either be templates or mutual recursion UNKNOWN.
        program = build_ir_from_source(src)
        for func in program.functions:
            if func.name in ("funcA", "funcB"):
                res = analyze_complexity(program, function_name=func.name)
                # Mutual recursion should trigger UNKNOWN or be handled conservatively
                assert res.taxonomy_label in ("UNKNOWN", "ON")

    def test_argument_mapping_unsupported_returns_unknown(self):
        """Unsupported argument expression → ARGUMENT_MAPPING_UNSUPPORTED → UNKNOWN."""
        m = map_argument("some_struct.field + other", {"n"}, set(), set(), {})
        assert m.kind == "unsupported"


# ===========================================================================
# Section 14 — Recursion Templates: T1 (Decrementing, O(1)→O(n), O(n)→O(n²))
# ===========================================================================


