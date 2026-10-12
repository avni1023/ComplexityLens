"""
test_cpp_libclang_spike.py
==========================
Day-1 feasibility spike tests for the libclang C++ parser.

Each test targets one or more of the 12 fixture files under
tests/fixtures/cpp/ and verifies that the correct AST node kinds
are produced along with valid source locations.

Run:
    pytest tests/test_cpp_libclang_spike.py -v
"""

from __future__ import annotations

import os
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "cpp"


def fixture_path(name: str) -> str:
    return str(FIXTURES / name)


def _parse(name: str):
    """Parse one fixture file; import deferred so collection succeeds without libclang."""
    from complexity_lens.parsers.cpp_libclang import parse_file  # noqa: PLC0415

    return parse_file(fixture_path(name))


# ---------------------------------------------------------------------------
# STL skip helper
# ---------------------------------------------------------------------------
_STL_SKIP_REASON = (
    "STL system headers (<algorithm>, <vector>) are not available with the "
    "bundled libclang==18.1.1 wheel on Windows.  These tests document a known "
    "limitation; they will pass once a system LLVM with C++ headers is installed.  "
    "Workaround: install LLVM from https://releases.llvm.org/ and set LIBCLANG_PATH."
)


def _stl_skip_if_no_headers(result) -> None:
    """Skip gracefully when system C++ headers are absent."""
    if result.has_fatal_error:
        pytest.skip(_STL_SKIP_REASON)


# ---------------------------------------------------------------------------
# Module-level cached fixtures (avoids class-scoped instance-method deprecation)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r01():
    return _parse("01_simple_function.cpp")


@pytest.fixture(scope="module")
def r02():
    return _parse("02_for_loop.cpp")


@pytest.fixture(scope="module")
def r03():
    return _parse("03_while_loop.cpp")


@pytest.fixture(scope="module")
def r04():
    return _parse("04_nested_loops.cpp")


@pytest.fixture(scope="module")
def r05():
    return _parse("05_sequential_loops.cpp")


@pytest.fixture(scope="module")
def r06():
    return _parse("06_function_call.cpp")


@pytest.fixture(scope="module")
def r07():
    return _parse("07_recursion.cpp")


@pytest.fixture(scope="module")
def r08():
    return _parse("08_stl_sort.cpp")


@pytest.fixture(scope="module")
def r09():
    return _parse("09_stl_search.cpp")


@pytest.fixture(scope="module")
def r10():
    return _parse("10_malformed.cpp")


@pytest.fixture(scope="module")
def r11():
    return _parse("11_unsupported_asm.cpp")


@pytest.fixture(scope="module")
def r12():
    return _parse("12_triple_nested.cpp")


# ---------------------------------------------------------------------------
# Fixture 01 — simple function
# ---------------------------------------------------------------------------


class TestSimpleFunction:
    """01_simple_function.cpp — int add(int a, int b)."""

    def test_no_fatal_error(self, r01):
        assert not r01.has_fatal_error

    def test_function_decl_found(self, r01):
        names = [f.spelling for f in r01.functions()]
        assert "add" in names, f"Expected 'add' in {names}"

    def test_source_location_valid(self, r01):
        add_fn = next(f for f in r01.functions() if f.spelling == "add")
        assert add_fn.loc.line >= 1
        assert add_fn.loc.col >= 1
        assert "01_simple_function" in add_fn.loc.file

    def test_parameters_present(self, r01):
        parms = r01.nodes_of_kind("PARM_DECL")
        names = [p.spelling for p in parms]
        assert "a" in names and "b" in names

    def test_return_stmt_present(self, r01):
        assert r01.nodes_of_kind("RETURN_STMT"), "Expected at least one RETURN_STMT"


# ---------------------------------------------------------------------------
# Fixture 02 — for loop
# ---------------------------------------------------------------------------


class TestForLoop:
    """02_for_loop.cpp — for-loop with accumulator."""

    def test_no_fatal_error(self, r02):
        assert not r02.has_fatal_error

    def test_for_stmt_found(self, r02):
        assert r02.for_stmts(), "Expected at least one FOR_STMT"

    def test_for_stmt_has_location(self, r02):
        assert r02.for_stmts()[0].loc.line >= 1

    def test_loop_nesting_depth_is_one(self, r02):
        assert r02.max_loop_nesting_depth() == 1


# ---------------------------------------------------------------------------
# Fixture 03 — while loop
# ---------------------------------------------------------------------------


class TestWhileLoop:
    """03_while_loop.cpp — while-loop variant."""

    def test_no_fatal_error(self, r03):
        assert not r03.has_fatal_error

    def test_while_stmt_found(self, r03):
        assert r03.while_stmts(), "Expected at least one WHILE_STMT"

    def test_while_stmt_has_location(self, r03):
        assert r03.while_stmts()[0].loc.line >= 1

    def test_no_for_stmt(self, r03):
        assert not r03.for_stmts(), "Unexpected FOR_STMT in while fixture"


# ---------------------------------------------------------------------------
# Fixture 04 — nested loops
# ---------------------------------------------------------------------------


class TestNestedLoops:
    """04_nested_loops.cpp — O(n^2) double-nested for."""

    def test_no_fatal_error(self, r04):
        assert not r04.has_fatal_error

    def test_two_for_stmts(self, r04):
        assert len(r04.for_stmts()) >= 2

    def test_nesting_depth_is_two(self, r04):
        assert r04.max_loop_nesting_depth() >= 2

    def test_inner_loop_location_after_outer(self, r04):
        stmts = r04.for_stmts()
        depths = sorted(set(s.depth for s in stmts))
        assert len(depths) >= 2, f"Expected loops at different depths, got {depths}"


# ---------------------------------------------------------------------------
# Fixture 05 — sequential loops
# ---------------------------------------------------------------------------


class TestSequentialLoops:
    """05_sequential_loops.cpp — two for-loops at the same level (O(n), not O(n^2))."""

    def test_no_fatal_error(self, r05):
        assert not r05.has_fatal_error

    def test_two_for_stmts(self, r05):
        assert len(r05.for_stmts()) >= 2

    def test_nesting_depth_is_one(self, r05):
        assert r05.max_loop_nesting_depth() == 1

    def test_loops_at_same_depth(self, r05):
        depths = [s.depth for s in r05.for_stmts()]
        assert len(set(depths)) == 1, f"Sequential loops should share depth, got {depths}"


# ---------------------------------------------------------------------------
# Fixture 06 — function call
# ---------------------------------------------------------------------------


class TestFunctionCall:
    """06_function_call.cpp — caller() invokes helper()."""

    def test_no_fatal_error(self, r06):
        assert not r06.has_fatal_error

    def test_two_function_decls(self, r06):
        names = [f.spelling for f in r06.functions()]
        assert "helper" in names and "caller" in names

    def test_call_expr_present(self, r06):
        assert r06.call_exprs(), "Expected CALL_EXPR nodes"

    def test_call_expr_spelling_is_helper(self, r06):
        spellings = [c.spelling for c in r06.call_exprs()]
        assert "helper" in spellings, f"Expected 'helper' in {spellings}"

    def test_call_expr_has_location(self, r06):
        assert r06.call_exprs()[0].loc.line >= 1


# ---------------------------------------------------------------------------
# Fixture 07 — recursion
# ---------------------------------------------------------------------------


class TestRecursion:
    """07_recursion.cpp — fib() calls itself."""

    def test_no_fatal_error(self, r07):
        assert not r07.has_fatal_error

    def test_fib_decl_found(self, r07):
        names = [f.spelling for f in r07.functions()]
        assert "fib" in names

    def test_is_recursive(self, r07):
        fib_node = next(f for f in r07.functions() if f.spelling == "fib")
        assert r07.is_recursive(fib_node), "fib() should be detected as recursive"

    def test_call_expr_to_fib(self, r07):
        calls = [c for c in r07.call_exprs() if c.spelling == "fib"]
        assert len(calls) >= 2, f"Expected >=2 recursive calls to fib, found {len(calls)}"


# ---------------------------------------------------------------------------
# Fixture 08 — STL sort (skipped when system headers absent)
# ---------------------------------------------------------------------------


class TestSTLSort:
    """08_stl_sort.cpp — std::sort via #include <algorithm>."""

    def test_no_fatal_error_or_known_skip(self, r08):
        """Fatal error is acceptable only when system headers are absent."""
        if r08.has_fatal_error:
            msgs = [d.message for d in r08.diagnostics]
            if any("file not found" in m or "not found" in m for m in msgs):
                pytest.skip(_STL_SKIP_REASON)
            else:
                pytest.fail(f"Unexpected fatal error: {msgs}")

    def test_sort_vec_function_found(self, r08):
        _stl_skip_if_no_headers(r08)
        names = [f.spelling for f in r08.functions()]
        assert "sort_vec" in names

    def test_call_expr_present(self, r08):
        _stl_skip_if_no_headers(r08)
        assert r08.call_exprs(), "Expected CALL_EXPR from std::sort"

    def test_sort_call_has_location(self, r08):
        _stl_skip_if_no_headers(r08)
        located = [c for c in r08.call_exprs() if c.loc.line > 0]
        assert located, "CallExprs should have line numbers"


# ---------------------------------------------------------------------------
# Fixture 09 — STL search (skipped when system headers absent)
# ---------------------------------------------------------------------------


class TestSTLSearch:
    """09_stl_search.cpp — std::find (linear search)."""

    def test_no_fatal_error_or_known_skip(self, r09):
        if r09.has_fatal_error:
            msgs = [d.message for d in r09.diagnostics]
            if any("file not found" in m or "not found" in m for m in msgs):
                pytest.skip(_STL_SKIP_REASON)
            else:
                pytest.fail(f"Unexpected fatal error: {msgs}")

    def test_contains_function_found(self, r09):
        _stl_skip_if_no_headers(r09)
        names = [f.spelling for f in r09.functions()]
        assert "contains" in names

    def test_call_expr_present(self, r09):
        _stl_skip_if_no_headers(r09)
        assert r09.call_exprs(), "Expected CALL_EXPR from std::find"


# ---------------------------------------------------------------------------
# Fixture 10 — malformed input
# ---------------------------------------------------------------------------


class TestMalformedInput:
    """10_malformed.cpp — missing closing brace and semicolon."""

    def test_diagnostics_emitted(self, r10):
        assert r10.diagnostics, "Malformed source must produce diagnostics"

    def test_diagnostics_have_locations(self, r10):
        with_loc = [d for d in r10.diagnostics if d.loc and d.loc.line > 0]
        assert with_loc, "Diagnostics should carry source locations"

    def test_partial_ast_still_produced(self, r10):
        assert r10.top_level_nodes, "Partial AST expected even for malformed input"

    def test_errors_are_error_or_fatal(self, r10):
        severities = {d.severity for d in r10.diagnostics}
        assert severities & {"error", "fatal"}, f"Expected error/fatal, got {severities}"


# ---------------------------------------------------------------------------
# Fixture 11 — unsupported construct (inline assembly)
# ---------------------------------------------------------------------------


class TestUnsupportedASM:
    """11_unsupported_asm.cpp — inline __asm__ (opaque to complexity analysis)."""

    def test_no_fatal_error(self, r11):
        assert not r11.has_fatal_error

    def test_function_decl_found(self, r11):
        names = [f.spelling for f in r11.functions()]
        assert "read_tsc" in names

    def test_asm_stmt_detected(self, r11):
        assert r11.asm_stmts(), (
            "ASM_STMT must be detected so the analyser can flag it as UNSUPPORTED"
        )

    def test_asm_stmt_has_location(self, r11):
        assert r11.asm_stmts()[0].loc.line >= 1


# ---------------------------------------------------------------------------
# Fixture 12 — triple nested loops
# ---------------------------------------------------------------------------


class TestTripleNested:
    """12_triple_nested.cpp — O(n^3) triple-nested for loops."""

    def test_no_fatal_error(self, r12):
        assert not r12.has_fatal_error

    def test_three_for_stmts(self, r12):
        assert len(r12.for_stmts()) >= 3

    def test_nesting_depth_is_three(self, r12):
        assert r12.max_loop_nesting_depth() >= 3

    def test_loops_at_three_distinct_depths(self, r12):
        depths = sorted(set(s.depth for s in r12.for_stmts()))
        assert len(depths) >= 3, f"Expected 3 distinct depths, got {depths}"


# ---------------------------------------------------------------------------
# Cross-cutting: source-location sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", [
    "01_simple_function.cpp",
    "02_for_loop.cpp",
    "03_while_loop.cpp",
    "04_nested_loops.cpp",
    "05_sequential_loops.cpp",
    "06_function_call.cpp",
    "07_recursion.cpp",
    "12_triple_nested.cpp",
])
def test_all_nodes_have_valid_locations(fixture_name):
    """Verify SourceLoc.line/col are non-zero for the vast majority of nodes."""
    result = _parse(fixture_name)
    bad = [n for n in result.all_nodes() if n.loc.line == 0 and n.loc.col == 0]
    total = sum(1 for _ in result.all_nodes())
    assert len(bad) / max(total, 1) < 0.20, (
        f"{fixture_name}: too many zero-location nodes ({len(bad)}/{total})"
    )


# ---------------------------------------------------------------------------
# CXX_INCLUDE_DIRS configuration handling
# ---------------------------------------------------------------------------


class TestCXXIncludeDirsConfig:
    """Verify CXX_INCLUDE_DIRS environment variable inspection."""

    def test_cxx_include_dirs_unset(self, monkeypatch):
        from complexity_lens.parsers.cpp_libclang import (
            _get_cxx_include_args,
            get_configured_cxx_include_dirs,
        )

        monkeypatch.delenv("CXX_INCLUDE_DIRS", raising=False)
        assert get_configured_cxx_include_dirs() == []
        assert _get_cxx_include_args() == []

    def test_cxx_include_dirs_filters_nonexistent(self, monkeypatch, tmp_path):
        from complexity_lens.parsers.cpp_libclang import (
            _get_cxx_include_args,
            get_configured_cxx_include_dirs,
        )

        real_dir = tmp_path / "valid_include"
        real_dir.mkdir()
        bogus_dir = tmp_path / "does_not_exist"

        raw = f"{real_dir}{os.pathsep}{bogus_dir}"
        monkeypatch.setenv("CXX_INCLUDE_DIRS", raw)

        dirs = get_configured_cxx_include_dirs()
        assert len(dirs) == 2
        args = _get_cxx_include_args()
        assert args == ["-isystem", str(real_dir)]

