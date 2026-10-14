"""
test_features.py — Comprehensive unit tests for ComplexityLens Day 6 Feature Extraction.

Validates:
1. Exactly 26 feature names and documented canonical order.
2. Snapshot test locking down the full names tuple.
3. Stable output across repeated extraction.
4. to_list() and to_dict() order and value consistency.
5. Every value is a non-negative int; flags are only 0 or 1.
6. Missing constructs produce documented defaults.
7. Nested loops versus sequential top-level loops.
8. A three-deep chain contributes three nested-loop pairs.
9. Each loop kind and each supported loop-bound category.
10. Direct recursion and mutual recursion.
11. Recursive shrink-minus-constant, divide, and other call sites.
12. Sequential calls and calls inside loops, with functions counted once.
13. Helper-with-loop called from inside a loop increases max_loop_depth correctly.
14. Recursive cycles cannot cause infinite traversal.
15. STL call categories and hash/ordered container operations.
16. Container operations inside nested loops are counted once per call site.
17. Early exits inside and outside loops.
18. Input-dependent operations using an input symbol or .size().
19. Unsupported constructs in nested structures and unreachable functions.
20. Incomplete IR returns a vector and leaves completeness metadata outside schema.
21. Wrong input types raise TypeError; valid incomplete IR does not.
22. Identifier renaming does not change the vector.
23. Comments and whitespace changes do not change the vector.
24. Rewriting classic for as while changes only n_for and n_while.
25. Leakage checks for labels, family IDs, filenames, paths, sample IDs, and split metadata.
26. Import-based check forbidding dependence on top-level engine analysis.
27. Import-based check forbidding ML package dependencies.
28. Hand-computed expected vectors for at least 12 small fixtures.
"""

from __future__ import annotations

import sys

import pytest

from complexity_lens.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureVector,
    extract_features,
)
from complexity_lens.ir.builder import build_ir_from_source
from complexity_lens.ir.model import (
    FunctionIR,
    ProgramIR,
    SourceSpan,
    UnsupportedIR,
    UnsupportedKind,
)

_CANONICAL_26_NAMES = (
    "n_loops",
    "max_loop_depth",
    "n_sequential_top_loops",
    "n_nested_loop_pairs",
    "n_for",
    "n_while",
    "n_loops_linear_bound",
    "n_loops_const_bound",
    "n_loops_log_update",
    "n_dependent_bound_loops",
    "n_loops_unknown_bound",
    "has_recursion",
    "n_recursive_call_sites",
    "shrink_minus_const",
    "shrink_divide",
    "shrink_other",
    "n_user_calls",
    "n_sort_calls",
    "n_linear_search_calls",
    "n_binary_search_calls",
    "n_hash_ops",
    "n_ordered_container_ops",
    "n_container_ops_in_loops",
    "has_early_exit_in_loop",
    "n_input_dependent_ops",
    "n_unsupported_constructs",
)

_DUMMY_SPAN = SourceSpan("<test>", 1, 1, 1, 1)


# ===========================================================================
# 1 & 2: Names and Order Snapshot
# ===========================================================================

def test_01_feature_names_count_and_order():
    """Verify exactly 26 feature names in documented order."""
    assert len(FEATURE_NAMES) == 26
    assert FEATURE_NAMES == _CANONICAL_26_NAMES


def test_02_feature_names_snapshot():
    """Snapshot test locking down the full names tuple and schema version."""
    assert FEATURE_SCHEMA_VERSION == "1.0"
    assert FEATURE_NAMES == (
        "n_loops",
        "max_loop_depth",
        "n_sequential_top_loops",
        "n_nested_loop_pairs",
        "n_for",
        "n_while",
        "n_loops_linear_bound",
        "n_loops_const_bound",
        "n_loops_log_update",
        "n_dependent_bound_loops",
        "n_loops_unknown_bound",
        "has_recursion",
        "n_recursive_call_sites",
        "shrink_minus_const",
        "shrink_divide",
        "shrink_other",
        "n_user_calls",
        "n_sort_calls",
        "n_linear_search_calls",
        "n_binary_search_calls",
        "n_hash_ops",
        "n_ordered_container_ops",
        "n_container_ops_in_loops",
        "has_early_exit_in_loop",
        "n_input_dependent_ops",
        "n_unsupported_constructs",
    )


# ===========================================================================
# 3: Determinism / Stability
# ===========================================================================

def test_03_stable_output_across_repeated_extraction():
    """Repeated extraction from same IR must yield identical values."""
    code = "void f(int n) { for (int i = 0; i < n; ++i) {} }"
    prog = build_ir_from_source(code)
    v1 = extract_features(prog)
    v2 = extract_features(prog)
    assert v1.values == v2.values
    assert v1.to_dict() == v2.to_dict()


# ===========================================================================
# 4 & 5: Structure and Types
# ===========================================================================

def test_04_to_list_and_to_dict_consistency():
    """to_list() and to_dict() must preserve canonical schema order and values."""
    code = "void f(int n) { for (int i = 0; i < n; ++i) {} }"
    prog = build_ir_from_source(code)
    feat = extract_features(prog)

    val_list = feat.to_list()
    assert len(val_list) == 26
    assert tuple(val_list) == feat.values

    val_dict = feat.to_dict()
    assert list(val_dict.keys()) == list(FEATURE_NAMES)
    assert list(val_dict.values()) == val_list


def test_05_types_and_binary_flags():
    """Every value must be an int >= 0; boolean flags must be strictly 0 or 1."""
    code = "void f(int n) { for (int i = 0; i < n; ++i) {} }"
    prog = build_ir_from_source(code)
    feat = extract_features(prog)

    for name, val in feat.to_dict().items():
        assert isinstance(val, int) and not isinstance(val, bool), f"{name} must be int"
        assert val >= 0, f"{name} must be >= 0"

    assert feat.has_recursion in (0, 1)
    assert feat.has_early_exit_in_loop in (0, 1)


# ===========================================================================
# 6: Missing Constructs Produce Defaults
# ===========================================================================

def test_06_missing_constructs_produce_defaults():
    """Empty function produces natural zero values for all features."""
    prog = build_ir_from_source("void empty() {}")
    feat = extract_features(prog)
    for name, val in feat.to_dict().items():
        assert val == 0, f"Expected 0 for {name}, got {val}"


# ===========================================================================
# 7 & 8: Loop Nesting and Pairs
# ===========================================================================

def test_07_nested_vs_sequential_top_loops():
    """Sequential top loops have depth 1; nested loops increase depth."""
    code = """
    void f(int n) {
        for (int i = 0; i < n; ++i) {}
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {}
        }
    }
    """
    prog = build_ir_from_source(code)
    feat = extract_features(prog)
    assert feat.n_loops == 3
    assert feat.max_loop_depth == 2
    assert feat.n_sequential_top_loops == 2
    assert feat.n_nested_loop_pairs == 1


def test_08_three_deep_chain_contributes_three_pairs():
    """A three-deep loop chain contributes exactly 3 ancestor-descendant pairs."""
    code = """
    void f(int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {}
            }
        }
    }
    """
    prog = build_ir_from_source(code)
    feat = extract_features(prog)
    assert feat.n_loops == 3
    assert feat.max_loop_depth == 3
    assert feat.n_sequential_top_loops == 1
    assert feat.n_nested_loop_pairs == 3


# ===========================================================================
# 9: Loop Kinds and Bound Categories
# ===========================================================================

def test_09_each_loop_kind_and_bound_category():
    """Test classic for, while, linear, const, log update, dependent, and unknown bound."""
    # 1. Linear bound for-loop
    f1 = extract_features(build_ir_from_source("void f(int n) { for (int i = 0; i < n; ++i) {} }"))
    assert f1.n_for == 1 and f1.n_while == 0 and f1.n_loops_linear_bound == 1

    # 2. Constant bound while-loop
    code_c = "void f() { int i = 0; while (i < 10) { ++i; } }"
    f2 = extract_features(build_ir_from_source(code_c))
    assert f2.n_for == 0 and f2.n_while == 1 and f2.n_loops_const_bound == 1

    # 3. Log update while-loop
    code_log = "void f(int n) { int i = 1; while (i < n) { i *= 2; } }"
    f3 = extract_features(build_ir_from_source(code_log))
    assert f3.n_loops_log_update == 1

    # 4. Dependent bound inner loop
    code_dep = (
        "void f(int n) { for (int i = 0; i < n; ++i) { for (int j = 0; j < i; ++j) {} } }"
    )
    f4 = extract_features(build_ir_from_source(code_dep))
    assert f4.n_dependent_bound_loops == 1

    # 5. Unknown bound (modified inside loop)
    code_unk = "void f(int n) { for (int i = 0; i < n; ++i) { i += 2; } }"
    f5 = extract_features(build_ir_from_source(code_unk))
    assert f5.n_loops_unknown_bound == 1


# ===========================================================================
# 10 & 11: Recursion & Shrink Kinds
# ===========================================================================

def test_10_direct_and_mutual_recursion():
    """Verify direct self-recursion and multi-function mutual recursion."""
    direct_code = """
    int fib(int n) {
        if (n <= 1) return 1;
        return fib(n - 1) + fib(n - 2);
    }
    """
    f_dir = extract_features(build_ir_from_source(direct_code))
    assert f_dir.has_recursion == 1
    assert f_dir.n_recursive_call_sites == 2

    # Mutual recursion: A calls B and B calls A
    mutual_code = """
    int foo(int n);
    int bar(int n) {
        if (n <= 0) return 0;
        return foo(n - 1);
    }
    int foo(int n) {
        if (n <= 0) return 0;
        return bar(n - 1);
    }
    """
    f_mut = extract_features(build_ir_from_source(mutual_code))
    assert f_mut.has_recursion == 1
    assert f_mut.n_recursive_call_sites >= 2


def test_11_recursive_shrink_categories():
    """Test shrink_minus_const, shrink_divide, and shrink_other."""
    minus_code = "int f(int n) { if (n <= 1) return 1; return f(n - 1); }"
    f_minus = extract_features(build_ir_from_source(minus_code))
    assert f_minus.shrink_minus_const == 1
    assert f_minus.shrink_divide == 0

    div_code = "int f(int n) { if (n <= 1) return 1; return f(n / 2); }"
    f_div = extract_features(build_ir_from_source(div_code))
    assert f_div.shrink_divide == 1
    assert f_div.shrink_minus_const == 0

    other_code = "int f(int n) { if (n <= 1) return 1; return f(n); }"
    f_other = extract_features(build_ir_from_source(other_code))
    assert f_other.shrink_other == 1


# ===========================================================================
# 12 & 13 & 14: User Calls & Interprocedural Loop Depth
# ===========================================================================

def test_12_sequential_calls_and_functions_counted_once():
    """Reachable function is counted once even if called multiple times."""
    code = """
    void helper(int n) {
        for (int i = 0; i < n; ++i) {}
    }
    void main_func(int n) {
        helper(n);
        helper(n);
    }
    """
    prog = build_ir_from_source(code)
    feat = extract_features(prog)
    assert feat.n_user_calls == 2
    assert feat.n_loops == 1  # helper's loop counted once in program


def test_13_helper_with_loop_increases_max_loop_depth():
    """Loop at depth 1 calling helper with loop at depth 1 yields max_loop_depth=2."""
    code = """
    void helper(int n) {
        for (int i = 0; i < n; ++i) {}
    }
    void main_func(int n) {
        for (int j = 0; j < n; ++j) {
            helper(n);
        }
    }
    """
    prog = build_ir_from_source(code)
    feat = extract_features(prog)
    assert feat.max_loop_depth == 2


def test_14_recursive_cycles_cannot_cause_infinite_traversal():
    """Mutually recursive functions with loops do not cause infinite recursion."""
    code = """
    void foo(int n);
    void bar(int n) {
        for (int i = 0; i < n; ++i) {}
        foo(n);
    }
    void foo(int n) {
        for (int j = 0; j < n; ++j) {}
        bar(n);
    }
    """
    prog = build_ir_from_source(code)
    feat = extract_features(prog)
    assert feat.has_recursion == 1
    assert feat.max_loop_depth >= 1


# ===========================================================================
# 15 & 16: STL Operations and Containers
# ===========================================================================

def test_15_stl_call_categories_and_container_ops():
    """Test recognized STL sort, linear search, binary search, and container kinds."""
    code = """
    #include <vector>
    #include <algorithm>
    #include <map>
    void test_stl(std::vector<int>& v, std::map<int,int>& m) {
        std::sort(v.begin(), v.end());
        std::find(v.begin(), v.end(), 42);
        std::binary_search(v.begin(), v.end(), 42);
        m.insert({1, 2});
    }
    """
    prog = build_ir_from_source(code)
    feat = extract_features(prog)
    assert feat.n_sort_calls == 1
    assert feat.n_linear_search_calls == 1
    assert feat.n_binary_search_calls == 1
    assert feat.n_ordered_container_ops >= 1


def test_16_container_ops_in_nested_loops_counted_once():
    """push_back inside nested loop is counted once per source call site."""
    code = """
    #include <vector>
    void f(std::vector<int>& v, int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                v.push_back(i + j);
            }
        }
    }
    """
    prog = build_ir_from_source(code)
    feat = extract_features(prog)
    assert feat.n_container_ops_in_loops == 1


# ===========================================================================
# 17: Early Exits
# ===========================================================================

def test_17_early_exits_inside_and_outside_loops():
    """Only break or return inside loops triggers has_early_exit_in_loop."""
    code_outside = "int f(int n) { for (int i = 0; i < n; ++i) {} return 0; }"
    f_out = extract_features(build_ir_from_source(code_outside))
    assert f_out.has_early_exit_in_loop == 0

    code_inside = "void f(int n) { for (int i = 0; i < n; ++i) { if (i == 5) break; } }"
    f_in = extract_features(build_ir_from_source(code_inside))
    assert f_in.has_early_exit_in_loop == 1


# ===========================================================================
# 18: Input Dependent Operations
# ===========================================================================

def test_18_input_dependent_operations():
    """User calls and STL operations taking input symbols or .size() are counted."""
    code = """
    #include <vector>
    void helper(int x) {}
    void test_fn(std::vector<int>& v, int n) {
        helper(n);
        helper(10);
    }
    """
    prog = build_ir_from_source(code)
    feat = extract_features(prog)
    assert feat.n_user_calls == 2
    assert feat.n_input_dependent_ops == 1  # helper(n) is input dependent, helper(10) is not


# ===========================================================================
# 19 & 20: Unsupported Constructs & Incomplete IR
# ===========================================================================

def test_19_unsupported_constructs_scoped_to_reachable():
    """Unsupported nodes in unreachable functions are not counted when entry exists."""
    unreachable_fn = FunctionIR(
        id="func_unreach",
        name="unreachable_func",
        return_type="void",
        parameters=[],
        span=_DUMMY_SPAN,
        body=[
            UnsupportedIR(
                id="unsupp_1",
                kind=UnsupportedKind.ASM_STMT,
                raw_spelling="__asm__",
                reason="asm",
                span=_DUMMY_SPAN,
            )
        ],
    )
    main_fn = FunctionIR(
        id="func_main",
        name="main",
        return_type="int",
        parameters=[],
        span=_DUMMY_SPAN,
        body=[],
    )
    prog = ProgramIR(
        source_path="<test>",
        functions=[main_fn, unreachable_fn],
        analysis_complete=False,
    )
    feat = extract_features(prog)
    assert feat.n_unsupported_constructs == 0


def test_20_incomplete_ir_returns_vector_without_poisoning_schema():
    """Incomplete IR returns a valid FeatureVector; completeness metadata stays separate."""
    prog = ProgramIR(
        source_path="<test>",
        functions=[
            FunctionIR(
                id="f1",
                name="main",
                return_type="int",
                parameters=[],
                span=_DUMMY_SPAN,
                body=[],
            )
        ],
        analysis_complete=False,
    )
    feat = extract_features(prog)
    assert isinstance(feat, FeatureVector)
    assert "analysis_complete" not in feat.names
    assert "analysis_complete" not in feat.to_dict()


# ===========================================================================
# 21: Wrong Input Types Raise TypeError
# ===========================================================================

def test_21_wrong_input_types_raise_type_error():
    """Only wrong input types raise TypeError; valid ProgramIR does not raise."""
    with pytest.raises(TypeError):
        extract_features("not a program ir")  # type: ignore

    with pytest.raises(TypeError):
        extract_features(123)  # type: ignore

    # Valid empty ProgramIR does not raise
    prog = ProgramIR(source_path="<test>")
    feat = extract_features(prog)
    assert feat.n_loops == 0


# ===========================================================================
# 22 & 23 & 24: Invariance & Rewriting Checks
# ===========================================================================

def test_22_identifier_renaming_invariance():
    """Renaming identifiers preserves all 26 feature values."""
    code1 = "void solve(int n) { for (int i = 0; i < n; ++i) {} }"
    code2 = "void algorithm(int total) { for (int idx = 0; idx < total; ++idx) {} }"
    feat1 = extract_features(build_ir_from_source(code1))
    feat2 = extract_features(build_ir_from_source(code2))
    assert feat1.values == feat2.values


def test_23_comments_and_whitespace_invariance():
    """Comments and whitespace modifications do not alter feature values."""
    code1 = "void f(int n) { for (int i = 0; i < n; ++i) {} }"
    code2 = """
    // Leading comment
    void f(int n) {
        /* inner comment */
        for   ( int i = 0 ; i < n ; ++i )  {
            // body comment
        }
    }
    """
    feat1 = extract_features(build_ir_from_source(code1))
    feat2 = extract_features(build_ir_from_source(code2))
    assert feat1.values == feat2.values


def test_24_for_to_while_rewriting_changes_only_n_for_and_n_while():
    """Rewriting classic for as while changes only n_for and n_while."""
    code_for = "void f(int n) { for (int i = 0; i < n; ++i) {} }"
    code_while = "void f(int n) { int i = 0; while (i < n) { ++i; } }"

    feat_for = extract_features(build_ir_from_source(code_for))
    feat_while = extract_features(build_ir_from_source(code_while))

    dict_for = feat_for.to_dict()
    dict_while = feat_while.to_dict()

    assert dict_for["n_for"] == 1
    assert dict_for["n_while"] == 0
    assert dict_while["n_for"] == 0
    assert dict_while["n_while"] == 1

    # All other 24 features must be strictly identical
    for k in FEATURE_NAMES:
        if k not in ("n_for", "n_while"):
            assert dict_for[k] == dict_while[k], (
                f"Mismatch for {k}: {dict_for[k]} != {dict_while[k]}"
            )


# ===========================================================================
# 25: Leakage Checks
# ===========================================================================

def test_25_leakage_checks():
    """Ensure labels, family IDs, paths, sample IDs, and split metadata never leak."""
    forbidden = {
        "label",
        "family",
        "family_id",
        "path",
        "filename",
        "source_path",
        "sample_id",
        "split",
        "target",
        "prediction",
        "confidence",
        "analysis_complete",
    }
    for name in FEATURE_NAMES:
        assert name.lower() not in forbidden
        for f in forbidden:
            assert f != name.lower()

    prog = build_ir_from_source("void f() {}")
    feat = extract_features(prog)
    for k in feat.to_dict().keys():
        assert k not in forbidden


# ===========================================================================
# 26 & 27: Architecture and Dependency Bounds
# ===========================================================================

def test_26_extractor_does_not_import_engine_analysis_entry_point():
    """Feature extraction must not import top-level complexity analysis entry points."""
    import complexity_lens.features.extractor as ext_mod

    # Inspect module globals and imported symbols
    assert not hasattr(ext_mod, "ComplexityEngine")
    assert not hasattr(ext_mod, "analyze_source")
    assert not hasattr(ext_mod, "analyze_file")
    assert not hasattr(ext_mod, "ComplexityClass")
    assert not hasattr(ext_mod, "Exactness")
    assert not hasattr(ext_mod, "AnalysisResult")


def test_27_extractor_does_not_import_ml_packages():
    """The feature-extraction module dependency graph must not import any ML packages."""
    ml_packages = ("sklearn", "scipy", "numpy", "pandas", "torch", "xgboost", "lightgbm")
    for pkg in ml_packages:
        assert pkg not in sys.modules, f"ML package {pkg} was imported!"


# ===========================================================================
# 28: Hand-computed Expected Vectors for 12 Fixtures
# ===========================================================================

def _expected_vector(**kwargs) -> tuple[int, ...]:
    """Helper to produce 26-tuple with defaults of 0 for specified kwargs."""
    mapping = {k: 0 for k in FEATURE_NAMES}
    for k, v in kwargs.items():
        if k not in mapping:
            raise KeyError(f"Unknown feature {k}")
        mapping[k] = v
    return tuple(mapping[k] for k in FEATURE_NAMES)


def test_28_hand_computed_12_fixtures():
    """Verify 12 distinct fixtures against hand-computed expected feature vectors."""
    fixtures = [
        # 1. Empty program
        (
            "void f() {}",
            _expected_vector(),
        ),
        # 2. Single linear for loop
        (
            "void f(int n) { for (int i = 0; i < n; ++i) {} }",
            _expected_vector(
                n_loops=1,
                max_loop_depth=1,
                n_sequential_top_loops=1,
                n_for=1,
                n_loops_linear_bound=1,
            ),
        ),
        # 3. Triple nested loops
        (
            """
            void f(int n) {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        for (int k = 0; k < n; ++k) {}
                    }
                }
            }
            """,
            _expected_vector(
                n_loops=3,
                max_loop_depth=3,
                n_sequential_top_loops=1,
                n_nested_loop_pairs=3,
                n_for=3,
                n_loops_linear_bound=3,
            ),
        ),
        # 4. Sequential top loops with linear and constant bounds
        (
            """
            void f(int n) {
                for (int i = 0; i < n; ++i) {}
                for (int j = 0; j < 10; ++j) {}
            }
            """,
            _expected_vector(
                n_loops=2,
                max_loop_depth=1,
                n_sequential_top_loops=2,
                n_for=2,
                n_loops_linear_bound=1,
                n_loops_const_bound=1,
            ),
        ),
        # 5. While loop with logarithmic update
        (
            """
            void f(int n) {
                int i = 1;
                while (i < n) {
                    i *= 2;
                }
            }
            """,
            _expected_vector(
                n_loops=1,
                max_loop_depth=1,
                n_sequential_top_loops=1,
                n_while=1,
                n_loops_log_update=1,
            ),
        ),
        # 6. Triangular dependent bound loop
        (
            """
            void f(int n) {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < i; ++j) {}
                }
            }
            """,
            _expected_vector(
                n_loops=2,
                max_loop_depth=2,
                n_sequential_top_loops=1,
                n_nested_loop_pairs=1,
                n_for=2,
                n_loops_linear_bound=1,
                n_dependent_bound_loops=1,
            ),
        ),
        # 7. Linear recursion with shrink minus const
        (
            """
            int fact(int n) {
                if (n <= 1) return 1;
                return n * fact(n - 1);
            }
            """,
            _expected_vector(
                has_recursion=1,
                n_recursive_call_sites=1,
                shrink_minus_const=1,
            ),
        ),
        # 8. Divide recursion
        (
            """
            int bsearch_rec(int n) {
                if (n <= 1) return 1;
                return bsearch_rec(n / 2);
            }
            """,
            _expected_vector(
                has_recursion=1,
                n_recursive_call_sites=1,
                shrink_divide=1,
            ),
        ),
        # 9. Tree recursion (Fibonacci)
        (
            """
            int fib(int n) {
                if (n <= 1) return 1;
                return fib(n - 1) + fib(n - 2);
            }
            """,
            _expected_vector(
                has_recursion=1,
                n_recursive_call_sites=2,
                shrink_minus_const=2,
            ),
        ),
        # 10. Interprocedural loop depth propagation
        (
            """
            void helper(int n) {
                for (int i = 0; i < n; ++i) {}
            }
            void main_func(int n) {
                for (int j = 0; j < n; ++j) {
                    helper(n);
                }
            }
            """,
            _expected_vector(
                n_loops=2,
                max_loop_depth=2,
                n_sequential_top_loops=2,
                n_for=2,
                n_loops_linear_bound=2,
                n_user_calls=1,
                n_input_dependent_ops=1,
            ),
        ),
        # 11. Early exit inside loop
        (
            """
            void f(int n) {
                for (int i = 0; i < n; ++i) {
                    if (i == 5) break;
                }
            }
            """,
            _expected_vector(
                n_loops=1,
                max_loop_depth=1,
                n_sequential_top_loops=1,
                n_for=1,
                n_loops_linear_bound=1,
                has_early_exit_in_loop=1,
            ),
        ),
        # 12. STL sort and container operation inside loop
        (
            """
            #include <vector>
            #include <algorithm>
            void sort_and_fill(std::vector<int>& v) {
                std::sort(v.begin(), v.end());
                for (int i = 0; i < 10; ++i) {
                    v.push_back(i);
                }
            }
            """,
            _expected_vector(
                n_loops=1,
                max_loop_depth=1,
                n_sequential_top_loops=1,
                n_for=1,
                n_loops_const_bound=1,
                n_sort_calls=1,
                n_container_ops_in_loops=1,
            ),
        ),
    ]

    for idx, (source, expected) in enumerate(fixtures, start=1):
        prog = build_ir_from_source(source)
        actual = extract_features(prog)
        assert actual.values == expected, (
            f"Fixture {idx} mismatch:\n"
            f"Expected: {dict(zip(FEATURE_NAMES, expected))}\n"
            f"Actual:   {actual.to_dict()}"
        )
