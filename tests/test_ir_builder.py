"""
test_ir_builder.py
===================
Day-2 tests for the Minimal Intermediate Representation (IR) of ComplexityLens.

Verifies:
- Typed IR representation for functions, loops, loop bounds/updates, calls,
  recursion call sites, recognized STL operations, source spans, and unsupported constructs.
- Parsing remains separate from IR construction (IR consumes ParseResult).
- Golden tests for all 12 parser fixtures.
- Explicit classification of unknown / unsupported syntax rather than silent guessing.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from complexity_lens.ir import (
    BoundKind,
    LoopIR,
    LoopKind,
    ProgramIR,
    STLOperationKind,
    UnsupportedKind,
    UpdateKind,
    build_ir,
    build_ir_from_source,
)
from complexity_lens.parsers.cpp_libclang import parse_file

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "cpp"


def _fixture(name: str) -> str:
    return str(FIXTURES / name)


def _ir(name: str) -> ProgramIR:
    parse_res = parse_file(_fixture(name))
    return build_ir(parse_res)


# ---------------------------------------------------------------------------
# Fixture 01 — Simple Function
# ---------------------------------------------------------------------------


class TestIRSimpleFunction:
    """01_simple_function.cpp -> int add(int a, int b)"""

    def test_function_structure(self):
        prog = _ir("01_simple_function.cpp")
        assert len(prog.functions) == 1
        fn = prog.functions[0]
        assert fn.name == "add"
        assert fn.return_type == "int"
        assert len(fn.parameters) == 2
        assert fn.parameters[0] == ("int", "a")
        assert fn.parameters[1] == ("int", "b")
        assert not fn.is_recursive
        assert len(fn.recursive_call_sites) == 0
        assert len(fn.all_loops()) == 0
        assert len(fn.all_calls()) == 0
        assert len(fn.all_unsupported()) == 0

    def test_source_span(self):
        prog = _ir("01_simple_function.cpp")
        fn = prog.functions[0]
        span = fn.span
        assert span.start_line == 3
        assert span.end_line == 5
        assert span.start_col >= 1
        assert "01_simple_function.cpp" in span.file


# ---------------------------------------------------------------------------
# Fixture 02 — For Loop with Linear Input Bound
# ---------------------------------------------------------------------------


class TestIRForLoop:
    """02_for_loop.cpp -> for (int i = 0; i < n; ++i)"""

    def test_loop_ir(self):
        prog = _ir("02_for_loop.cpp")
        fn = prog.functions[0]
        loops = fn.all_loops()
        assert len(loops) == 1
        loop = loops[0]

        assert loop.kind == LoopKind.FOR
        assert loop.depth == 1
        assert not loop.has_early_exit

        # Bound classification
        assert loop.bound.kind == BoundKind.LINEAR_INPUT
        assert loop.bound.variable == "i"
        assert loop.bound.comparison_op == "<"
        assert loop.bound.bound_expr == "n"

        # Update classification
        assert loop.update.kind == UpdateKind.ADDITIVE
        assert loop.update.step_op == "++"

    def test_loop_span(self):
        prog = _ir("02_for_loop.cpp")
        loop = prog.functions[0].all_loops()[0]
        assert loop.span.start_line == 5
        assert loop.span.end_line == 7


# ---------------------------------------------------------------------------
# Fixture 03 — While Loop
# ---------------------------------------------------------------------------


class TestIRWhileLoop:
    """03_while_loop.cpp -> while (i < n)"""

    def test_while_ir(self):
        prog = _ir("03_while_loop.cpp")
        fn = prog.functions[0]
        loops = fn.all_loops()
        assert len(loops) == 1
        loop = loops[0]

        assert loop.kind == LoopKind.WHILE
        assert loop.depth == 1
        assert loop.bound.kind == BoundKind.LINEAR_INPUT
        assert loop.bound.bound_expr == "n"
        assert loop.update.kind == UpdateKind.ADDITIVE


# ---------------------------------------------------------------------------
# Fixture 04 — Nested Loops (O(n^2))
# ---------------------------------------------------------------------------


class TestIRNestedLoops:
    """04_nested_loops.cpp -> for (i) { for (j) { ... } }"""

    def test_nesting_hierarchy(self):
        prog = _ir("04_nested_loops.cpp")
        fn = prog.functions[0]
        assert len(fn.body) == 1  # outer loop at top level
        outer = fn.body[0]
        assert isinstance(outer, LoopIR)
        assert outer.depth == 1
        assert outer.bound.variable == "i"

        assert len(outer.body) == 1
        inner = outer.body[0]
        assert isinstance(inner, LoopIR)
        assert inner.depth == 2
        assert inner.bound.variable == "j"

        # Both have LINEAR_INPUT bounds
        assert outer.bound.kind == BoundKind.LINEAR_INPUT
        assert inner.bound.kind == BoundKind.LINEAR_INPUT


# ---------------------------------------------------------------------------
# Fixture 05 — Sequential Loops (O(n))
# ---------------------------------------------------------------------------


class TestIRSequentialLoops:
    """05_sequential_loops.cpp -> for (i) ... for (j) ..."""

    def test_sequential_at_same_depth(self):
        prog = _ir("05_sequential_loops.cpp")
        fn = prog.functions[0]
        top_loops = [item for item in fn.body if isinstance(item, LoopIR)]
        assert len(top_loops) == 2
        for loop in top_loops:
            assert loop.depth == 1
            assert loop.bound.kind == BoundKind.LINEAR_INPUT


# ---------------------------------------------------------------------------
# Fixture 06 — Cross-Function Call
# ---------------------------------------------------------------------------


class TestIRFunctionCall:
    """06_function_call.cpp -> helper() and caller()"""

    def test_call_ir(self):
        prog = _ir("06_function_call.cpp")
        assert len(prog.functions) == 2
        caller = next(f for f in prog.functions if f.name == "caller")
        calls = caller.all_calls()
        assert len(calls) == 2
        for call in calls:
            assert call.callee_name == "helper"
            assert not call.is_recursion
            assert call.shrink_kind is None
            assert call.span.start_line == 8


# ---------------------------------------------------------------------------
# Fixture 07 — Recursion Call Sites
# ---------------------------------------------------------------------------


class TestIRRecursion:
    """07_recursion.cpp -> fib(n - 1) + fib(n - 2)"""

    def test_recursive_function(self):
        prog = _ir("07_recursion.cpp")
        fn = prog.functions[0]
        assert fn.name == "fib"
        assert fn.is_recursive
        assert len(fn.recursive_call_sites) == 2

        for site in fn.recursive_call_sites:
            assert site.callee_name == "fib"
            assert site.is_recursion
            assert site.shrink_kind == "minus_const"
            assert site.span.start_line == 5


# ---------------------------------------------------------------------------
# Fixture 08 — STL Sort
# ---------------------------------------------------------------------------


class TestIRSTLSort:
    """08_stl_sort.cpp -> std::sort(v.begin(), v.end())"""

    def test_stl_sort_operation(self):
        from complexity_lens.parsers.cpp_libclang import get_configured_cxx_include_dirs
        if not get_configured_cxx_include_dirs() and not os.environ.get("CXX_INCLUDE_DIRS"):
            pytest.skip("STL headers not configured in CXX_INCLUDE_DIRS")

        prog = _ir("08_stl_sort.cpp")
        fn = prog.functions[0]
        assert fn.name == "sort_vec"
        stl_ops = fn.all_stl_ops()
        assert len(stl_ops) == 1
        op = stl_ops[0]
        assert op.operation == STLOperationKind.SORT
        assert op.callee_name == "std::sort"
        assert op.container == "v"
        assert op.span.start_line == 7


# ---------------------------------------------------------------------------
# Fixture 09 — STL Search
# ---------------------------------------------------------------------------


class TestIRSTLSearch:
    """09_stl_search.cpp -> std::find(v.begin(), v.end(), val)"""

    def test_stl_find_operation(self):
        from complexity_lens.parsers.cpp_libclang import get_configured_cxx_include_dirs
        if not get_configured_cxx_include_dirs() and not os.environ.get("CXX_INCLUDE_DIRS"):
            pytest.skip("STL headers not configured in CXX_INCLUDE_DIRS")

        prog = _ir("09_stl_search.cpp")
        fn = prog.functions[0]
        assert fn.name == "contains"
        stl_ops = fn.all_stl_ops()
        assert len(stl_ops) == 1
        op = stl_ops[0]
        assert op.operation == STLOperationKind.FIND
        assert op.callee_name == "std::find"
        assert op.container == "v"
        assert op.span.start_line == 7


# ---------------------------------------------------------------------------
# Fixture 10 — Malformed Code & Explicit Syntax Errors
# ---------------------------------------------------------------------------


class TestIRMalformed:
    """10_malformed.cpp -> syntax errors captured explicitly."""

    def test_syntax_errors_recorded(self):
        prog = _ir("10_malformed.cpp")
        assert prog.has_syntax_errors
        # diagnostics is now list[IRDiagnostic] — check structured objects
        assert len(prog.diagnostics) >= 1
        error_diags = [d for d in prog.diagnostics if d.is_error()]
        assert len(error_diags) >= 1
        assert len(prog.unsupported_constructs) >= 1
        u = prog.unsupported_constructs[0]
        assert u.kind == UnsupportedKind.UNKNOWN_SYNTAX
        assert "Parser syntax error" in u.reason
        assert u.span.start_line in (3, 4)


# ---------------------------------------------------------------------------
# Fixture 11 — Unsupported Inline Assembly
# ---------------------------------------------------------------------------


class TestIRUnsupportedASM:
    """11_unsupported_asm.cpp -> __asm__ volatile (...)"""

    def test_asm_unsupported_ir(self):
        prog = _ir("11_unsupported_asm.cpp")
        fn = prog.functions[0]
        assert fn.name == "read_tsc"
        unsupported = fn.all_unsupported()
        assert len(unsupported) == 1
        u = unsupported[0]
        assert u.kind == UnsupportedKind.ASM_STMT
        assert "Inline assembly is opaque" in u.reason
        assert u.span.start_line == 6


# ---------------------------------------------------------------------------
# Fixture 12 — Triple-Nested Loops
# ---------------------------------------------------------------------------


class TestIRTripleNested:
    """12_triple_nested.cpp -> 3x for loop nesting."""

    def test_triple_nesting_depths(self):
        prog = _ir("12_triple_nested.cpp")
        fn = prog.functions[0]
        loops = fn.all_loops()
        assert len(loops) == 3
        depths = [lp.depth for lp in loops]
        assert depths == [1, 2, 3]


# ---------------------------------------------------------------------------
# Explicit Unknown / Dependent / Multiplicative Syntax Tests
# ---------------------------------------------------------------------------


class TestExplicitSyntaxVariants:
    """Verify that unknown/unrecognized constructs are explicit, not guessed."""

    def test_constant_loop_bound(self):
        code = "void f() { for (int i = 0; i < 100; ++i) {} }"
        prog = build_ir_from_source(code)
        loop = prog.all_loops()[0]
        assert loop.bound.kind == BoundKind.CONSTANT
        assert loop.bound.bound_expr == "100"

    def test_dependent_triangular_loop_bound(self):
        code = "void f(int n) { for (int i = 0; i < n; ++i) { for (int j = 0; j < i; ++j) {} } }"
        prog = build_ir_from_source(code)
        inner = prog.all_loops()[1]
        assert inner.bound.kind == BoundKind.DEPENDENT
        assert inner.bound.bound_expr == "i"

    def test_unknown_loop_bound_is_explicit(self):
        code = "bool check(); void f() { for (int i = 0; check(); ++i) {} }"
        prog = build_ir_from_source(code)
        loop = prog.all_loops()[0]
        assert loop.bound.kind == BoundKind.UNKNOWN

    def test_multiplicative_update(self):
        code = "void f(int n) { for (int i = 1; i < n; i *= 2) {} }"
        prog = build_ir_from_source(code)
        loop = prog.all_loops()[0]
        assert loop.update.kind == UpdateKind.MULTIPLICATIVE

    def test_early_exit_break_detected(self):
        code = "void f(int n) { for (int i = 0; i < n; ++i) { if (i == 5) break; } }"
        prog = build_ir_from_source(code)
        loop = prog.all_loops()[0]
        assert loop.has_early_exit

    def test_recursion_divide_shrink(self):
        code = "int bin_search(int n) { if (n <= 1) return 1; return bin_search(n / 2); }"
        prog = build_ir_from_source(code)
        fn = prog.functions[0]
        assert fn.is_recursive
        assert fn.recursive_call_sites[0].shrink_kind == "divide"

    def test_vector_push_back_stl_operation(self):
        code = """
        #include <vector>
        void append(std::vector<int>& v, int x) { v.push_back(x); }
        """
        from complexity_lens.parsers.cpp_libclang import get_configured_cxx_include_dirs
        if not get_configured_cxx_include_dirs() and not os.environ.get("CXX_INCLUDE_DIRS"):
            pytest.skip("STL headers not configured in CXX_INCLUDE_DIRS")

        prog = build_ir_from_source(code)
        fn = prog.functions[0]
        stl_ops = fn.all_stl_ops()
        assert len(stl_ops) == 1
        assert stl_ops[0].operation == STLOperationKind.PUSH_BACK
        assert stl_ops[0].container == "v"
        assert stl_ops[0].arguments == ["x"]

    def test_member_call_arguments_and_pointer_syntax(self):
        code = """
        #include <vector>
        void ops(std::vector<int>& v, std::vector<int>* ptr, int x) {
            v.push_back(x);
            ptr->push_back(x);
            v.insert(v.begin(), x);
        }
        """
        from complexity_lens.parsers.cpp_libclang import get_configured_cxx_include_dirs
        if not get_configured_cxx_include_dirs() and not os.environ.get("CXX_INCLUDE_DIRS"):
            pytest.skip("STL headers not configured in CXX_INCLUDE_DIRS")

        prog = build_ir_from_source(code)
        fn = prog.functions[0]
        stl_ops = fn.all_stl_ops()
        assert len(stl_ops) == 3
        # 1. v.push_back(x) -> args should be ['x'], NOT ['v.push_back', 'x']
        assert stl_ops[0].arguments == ["x"]
        assert stl_ops[0].container == "v"
        # 2. ptr->push_back(x) -> args should be ['x'], NOT ['ptr->push_back', 'x']
        assert stl_ops[1].arguments == ["x"]
        assert stl_ops[1].container == "ptr"
        # 3. v.insert(v.begin(), x) -> args should be ['v.begin()', 'x']
        assert stl_ops[2].arguments == ["v.begin()", "x"]
        assert stl_ops[2].container == "v"


# ---------------------------------------------------------------------------
# Serialization & Pretty Printing Tests
# ---------------------------------------------------------------------------


class TestIRSerialization:
    def test_to_dict_structure(self):
        prog = _ir("02_for_loop.cpp")
        d = prog.to_dict()
        assert "source_path" in d
        assert "functions" in d
        assert len(d["functions"]) == 1
        fn_dict = d["functions"][0]
        assert fn_dict["name"] == "sum_for"
        assert fn_dict["return_type"] == "int"
        assert len(fn_dict["body"]) == 1
        loop_dict = fn_dict["body"][0]
        assert loop_dict["type"] == "LoopIR"
        assert loop_dict["kind"] == "FOR"
        assert loop_dict["bound"]["kind"] == "LINEAR_INPUT"
        assert loop_dict["update"]["kind"] == "ADDITIVE"

    def test_pretty_print(self):
        prog = _ir("04_nested_loops.cpp")
        output = prog.pretty_print()
        assert "ProgramIR(" in output
        assert "Function matrix_sum" in output
        assert "Loop FOR (depth=1)" in output
        assert "Loop FOR (depth=2)" in output
        assert "LINEAR_INPUT" in output
