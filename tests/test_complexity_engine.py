"""
test_complexity_engine.py — Comprehensive unit tests for the Static Complexity Engine.

Tests all requirements specified in Day 4 Specification Addendum:
- Section 1: Input-size convention, named constants, cin variables, single/multi symbols.
- Section 2: Structured result, rule trace, deterministic output across repeated runs.
- Section 3: Function selection precedence (explicit, main, single, ambiguous error).
- Section 4: Loop classification rules (linear, doubling, halving, range-for,
             variable modification, early exit break/return, continue, dependent/triangular).
             Proof that loops NEVER produce EXP.
- Section 5: Composition, dominance ordering, branches, algebraic properties.
- Section 6: Constant-cost call allowlist, unsupported calls, user-defined calls.
- Section 7: Diagnostics and unsupported IR scoping (inside vs outside selected function).
- Section 8: Taxonomy mapping (every label: O1, OLOGN, ON, ONLOGN, ON2, ON3, OEXP, etc.).
- Section 9: Exactness semantics (STRUCTURAL, UPPER_BOUND, UNKNOWN) and stable reason codes.
"""

import pytest

from complexity_lens.complexity import (
    ComplexityClass,
    Exactness,
    FunctionSelectionError,
    ReasonCode,
    TaxonomyLabel,
    analyze_complexity,
    analyze_source,
    classify_loop,
    compare_dominance,
    dominant,
    multiply,
)
from complexity_lens.ir.builder import build_ir_from_source
from complexity_lens.ir.model import (
    BoundKind,
    DiagnosticSeverity,
    IRDiagnostic,
    LoopBound,
    LoopIR,
    LoopKind,
    LoopUpdate,
    SourceSpan,
    UnsupportedIR,
    UnsupportedKind,
    UpdateKind,
)

# ===========================================================================
# 1. Algebraic Properties & Dominance Ordering (Section 5)
# ===========================================================================


def test_algebraic_dominance_commutativity():
    """dominant(a, b) == dominant(b, a) for all tested classes."""
    classes = [
        ComplexityClass.constant(),
        ComplexityClass.logarithmic(),
        ComplexityClass.linear(),
        ComplexityClass.linearithmic(),
        ComplexityClass.quadratic(),
        ComplexityClass.cubic(),
        ComplexityClass.exponential(),
        ComplexityClass.unknown(),
    ]
    for c1 in classes:
        for c2 in classes:
            r1 = dominant(c1, c2)
            r2 = dominant(c2, c1)
            assert r1 == r2, f"Commutativity failed for {c1} and {c2}"


def test_algebraic_dominance_associativity():
    """dominant(dominant(a, b), c) == dominant(a, dominant(b, c))."""
    classes = [
        ComplexityClass.constant(),
        ComplexityClass.logarithmic(),
        ComplexityClass.linear(),
        ComplexityClass.quadratic(),
        ComplexityClass.unknown(),
    ]
    for c1 in classes:
        for c2 in classes:
            for c3 in classes:
                r1 = dominant(dominant(c1, c2), c3)
                r2 = dominant(c1, dominant(c2, c3))
                assert r1 == r2, f"Associativity failed for {c1}, {c2}, {c3}"


def test_algebraic_multiplication_properties():
    """multiply(a, b) commutativity, associativity, and degree addition."""
    c_lin = ComplexityClass.linear()
    c_log = ComplexityClass.logarithmic()
    c_quad = ComplexityClass.quadratic()

    prod = multiply(c_lin, c_lin)
    assert prod == c_quad
    assert prod.p == 2 and prod.q == 0

    prod_nlogn = multiply(c_lin, c_log)
    assert prod_nlogn == ComplexityClass.linearithmic()
    assert prod_nlogn.p == 1 and prod_nlogn.q == 1

    # Commutativity
    assert multiply(c_lin, c_log) == multiply(c_log, c_lin)

    # Unknown propagation
    assert multiply(c_lin, ComplexityClass.unknown()).is_unknown


def test_dominance_ordering_comparison():
    """compare_dominance verifies standard asymptotic order."""
    c_1 = ComplexityClass.constant()
    c_logn = ComplexityClass.logarithmic()
    c_n = ComplexityClass.linear()
    c_nlogn = ComplexityClass.linearithmic()
    c_n2 = ComplexityClass.quadratic()
    c_n3 = ComplexityClass.cubic()
    c_exp = ComplexityClass.exponential()

    chain = [c_1, c_logn, c_n, c_nlogn, c_n2, c_n3, c_exp]
    for i in range(len(chain) - 1):
        assert compare_dominance(chain[i], chain[i + 1]) == -1
        assert compare_dominance(chain[i + 1], chain[i]) == 1
        assert compare_dominance(chain[i], chain[i]) == 0

    # Comparing UNKNOWN raises ValueError
    with pytest.raises(ValueError):
        compare_dominance(c_1, ComplexityClass.unknown())


# ===========================================================================
# 2. Taxonomy Mapping (Section 8)
# ===========================================================================


def test_taxonomy_label_all_mappings():
    """Test every taxonomy label maps to the expected string."""
    assert ComplexityClass.constant().to_taxonomy_label() == TaxonomyLabel.O1.value
    assert ComplexityClass.logarithmic().to_taxonomy_label() == TaxonomyLabel.OLOGN.value
    assert ComplexityClass.linear().to_taxonomy_label() == TaxonomyLabel.ON.value
    assert ComplexityClass.linearithmic().to_taxonomy_label() == TaxonomyLabel.ONLOGN.value
    assert ComplexityClass.quadratic().to_taxonomy_label() == TaxonomyLabel.ON2.value
    assert ComplexityClass.cubic().to_taxonomy_label() == TaxonomyLabel.ON3.value
    assert ComplexityClass.exponential().to_taxonomy_label() == TaxonomyLabel.OEXP.value
    assert ComplexityClass.unknown().to_taxonomy_label() == TaxonomyLabel.UNKNOWN.value

    # Degree > 3 -> OUT_OF_TAXONOMY
    c_n4 = ComplexityClass(p=4, q=0)
    assert c_n4.to_taxonomy_label() == TaxonomyLabel.OUT_OF_TAXONOMY.value


# ===========================================================================
# 3. Straight-line Code O(1) & Call Allowlist (Sections 4, 6)
# ===========================================================================


def test_straight_line_constant():
    """Straight-line arithmetic code is O(1) with exactness STRUCTURAL."""
    code = """
    int main() {
        int a = 10;
        int b = 20;
        int c = a + b * 2;
        return c;
    }
    """
    res = analyze_source(code)
    assert res.taxonomy_label == "O1"
    assert res.exactness == "STRUCTURAL"
    assert ReasonCode.CONSTANT_WORK.value in res.reason_codes
    assert len(res.rule_trace) >= 1
    assert res.rule_trace[0].rule_id == "RULE_CONSTANT_WORK"


def test_call_allowlist_cin_cout_and_operations():
    """Allowlisted calls (cin, cout, min, max, abs) remain O(1)."""
    code = """
    #include <iostream>
    #include <algorithm>
    #include <cmath>

    int main() {
        int x;
        std::cin >> x;
        int y = std::abs(x);
        int z = std::max(x, y);
        std::cout << z;
        return 0;
    }
    """
    res = analyze_source(code)
    assert res.taxonomy_label == "O1"
    assert res.exactness == "STRUCTURAL"


def test_call_allowlist_push_back_amortized():
    """v.push_back() is allowlisted as amortized constant with explicit reason code."""
    code = """
    #include <vector>

    int main() {
        std::vector<int> v;
        v.push_back(42);
        v.emplace_back(99);
        return 0;
    }
    """
    res = analyze_source(code)
    assert res.taxonomy_label == "O1"
    assert res.exactness == "STRUCTURAL"
    assert ReasonCode.AMORTIZED_CONSTANT_CALL.value in res.reason_codes
    assert "amortized" in res.explanation.lower()


def test_unsupported_calls_yield_unknown():
    """Calls to unsupported functions and non-allowlisted STL yield UNKNOWN."""
    # User-defined function call with unsupported argument mapping
    code_user = """
    void helper(int x) {}
    int main(int n) {
        helper(n * n);
        return 0;
    }
    """
    res = analyze_source(code_user)
    assert res.taxonomy_label == "UNKNOWN"
    assert res.exactness == "UNKNOWN"
    assert ReasonCode.UNSUPPORTED_CALL.value in res.reason_codes

    # Supported STL algorithm: std::sort on random access container range
    code_sort = """
    #include <vector>
    #include <algorithm>
    int main() {
        std::vector<int> v;
        std::sort(v.begin(), v.end());
        return 0;
    }
    """
    res_sort = analyze_source(code_sort)
    assert res_sort.taxonomy_label == "ONLOGN"
    assert res_sort.exactness == "STRUCTURAL"
    assert ReasonCode.STL_SORT.value in res_sort.reason_codes

    # Truly unsupported / non-allowlisted STL algorithm: std::reverse
    code_reverse = """
    #include <vector>
    #include <algorithm>
    int main() {
        std::vector<int> v;
        std::reverse(v.begin(), v.end());
        return 0;
    }
    """
    res_reverse = analyze_source(code_reverse)
    assert res_reverse.taxonomy_label == "UNKNOWN"
    assert res_reverse.exactness == "UNKNOWN"
    assert ReasonCode.UNSUPPORTED_CALL.value in res_reverse.reason_codes


# ===========================================================================
# 4. Loop Classification Rules (Section 4)
# ===========================================================================


def test_linear_loop_input_size():
    """for (int i = 0; i < n; ++i) is O(n) STRUCTURAL."""
    code = """
    void solve(int n) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += i;
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "ON"
    assert res.exactness == "STRUCTURAL"
    assert ReasonCode.LINEAR_LOOP.value in res.reason_codes


def test_constant_bound_loop():
    """for (int i = 0; i < 100; ++i) is O(1) STRUCTURAL."""
    code = """
    int main() {
        int sum = 0;
        for (int i = 0; i < 100; ++i) {
            sum += i;
        }
        return sum;
    }
    """
    res = analyze_source(code)
    assert res.taxonomy_label == "O1"
    assert res.exactness == "STRUCTURAL"
    assert ReasonCode.CONSTANT_BOUND_LOOP.value in res.reason_codes


def test_doubling_loop():
    """for (int i = 1; i < n; i *= 2) is O(log n)."""
    code = """
    void solve(int n) {
        for (int i = 1; i < n; i *= 2) {
            // work
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "OLOGN"
    assert res.exactness == "STRUCTURAL"
    assert ReasonCode.LOGARITHMIC_UPDATE.value in res.reason_codes


def test_doubling_loop_zero_initialization_yields_unknown():
    """Multiplicative loop starting at 0 yields UNKNOWN."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; i *= 2) {
            // infinite or unknown
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "UNKNOWN"
    assert ReasonCode.UNKNOWN_BOUND.value in res.reason_codes


def test_halving_loop():
    """while (n > 0) { n /= 2; } is O(log n)."""
    code = """
    void solve(int n) {
        while (n > 0) {
            n /= 2;
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "OLOGN"
    assert res.exactness == "STRUCTURAL"
    assert ReasonCode.LOGARITHMIC_UPDATE.value in res.reason_codes


def test_unsupported_not_equal_condition():
    """Loops with '!=' condition yield UNKNOWN."""
    code = """
    void solve(int n) {
        for (int i = 0; i != n; ++i) {}
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "UNKNOWN"
    assert ReasonCode.UNKNOWN_BOUND.value in res.reason_codes


def test_variable_step_size_yields_unknown():
    """Loops with variable step (e.g. i += step) yield UNKNOWN."""
    code = """
    void solve(int n, int step) {
        for (int i = 0; i < n; i += step) {}
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "UNKNOWN"
    assert ReasonCode.UNKNOWN_BOUND.value in res.reason_codes


def test_range_for_container_vs_fixed_array():
    """Range-for over input container is O(n); over fixed array is O(1)."""
    code_container = """
    #include <vector>
    void process(std::vector<int> v) {
        for (auto x : v) {}
    }
    """
    res_cont = analyze_source(code_container, function_name="process")
    assert res_cont.taxonomy_label == "ON"

    code_arr = """
    void process() {
        int arr[10];
        for (auto x : arr) {}
    }
    """
    res_arr = analyze_source(code_arr, function_name="process")
    assert res_arr.taxonomy_label == "O1"


def test_loop_var_modified_in_body():
    """Modifying loop counter or bound inside body yields UNKNOWN with LOOP_VAR_MODIFIED."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {
            if (i % 2 == 0) {
                i += 3;
            }
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "UNKNOWN"
    assert ReasonCode.LOOP_VAR_MODIFIED.value in res.reason_codes


def test_break_and_return_yield_upper_bound():
    """Early exit inside a loop (break/return) changes exactness to UPPER_BOUND."""
    code_break = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {
            if (i == 42) break;
        }
    }
    """
    res = analyze_source(code_break, function_name="solve")
    assert res.taxonomy_label == "ON"
    assert res.exactness == "UPPER_BOUND"

    code_return = """
    int find_val(int n) {
        for (int i = 0; i < n; ++i) {
            if (i == 42) return i;
        }
        return -1;
    }
    """
    res_ret = analyze_source(code_return, function_name="find_val")
    assert res_ret.taxonomy_label == "ON"
    assert res_ret.exactness == "UPPER_BOUND"


def test_continue_preserves_structural_exactness():
    """A 'continue' statement in loop body does not change exactness."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {
            if (i % 2 == 0) continue;
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "ON"
    assert res.exactness == "STRUCTURAL"


def test_triangular_dependent_bound_yields_unknown():
    """Inner loop bound depending on outer loop variable yields UNKNOWN with DEPENDENT_BOUND."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                // triangular
            }
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "UNKNOWN"
    assert ReasonCode.DEPENDENT_BOUND.value in res.reason_codes


def test_loops_never_produce_exp():
    """Explicitly verify loop classification never produces EXP."""
    # Build a loop with arbitrarily large step or bound
    loop = LoopIR(
        id="loop_test",
        kind=LoopKind.FOR,
        depth=1,
        bound=LoopBound(kind=BoundKind.LINEAR_INPUT, bound_expr="n", comparison_op="<"),
        update=LoopUpdate(kind=UpdateKind.ADDITIVE, variable="i", step_op="++"),
        span=SourceSpan("test.cpp", 1, 1, 5, 1),
    )
    c, e, reasons, sym = classify_loop(loop, {}, set(), set(), {"n"})
    assert not c.is_exp
    assert c.to_taxonomy_label() != TaxonomyLabel.OEXP.value


# ===========================================================================
# 5. Composition and Dominance (Section 5)
# ===========================================================================


def test_sequential_linear_loops_are_on_not_on2():
    """Two sequential linear loops compose to O(n), not O(n^2)."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {}
        for (int j = 0; j < n; ++j) {}
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "ON"
    assert res.complexity_class == ComplexityClass.linear()
    assert ReasonCode.SEQUENTIAL_COMPOSITION.value in res.reason_codes


def test_nested_linear_and_constant_inner_loop():
    """A linear loop containing an inner constant loop is O(n)."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 10; ++j) {}
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "ON"
    assert res.complexity_class == ComplexityClass.linear()


def test_nested_independent_linear_loops():
    """Independent nested loops: O(n) * O(n) = O(n^2)."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {}
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "ON2"
    assert res.complexity_class == ComplexityClass.quadratic()
    assert ReasonCode.NESTED_LOOP.value in res.reason_codes


def test_nested_three_loops():
    """Three independent nested loops: O(n^3)."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {}
            }
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "ON3"
    assert res.complexity_class == ComplexityClass.cubic()


def test_branch_worst_case_dominance():
    """if-else branches take the worst-case dominant branch."""
    code = """
    void solve(int n, bool flag) {
        if (flag) {
            for (int i = 0; i < n; ++i) {}
        } else {
            for (int i = 0; i < 10; ++i) {}
        }
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "ON"
    assert ReasonCode.IF_ELSE_WORST_CASE.value in res.reason_codes


# ===========================================================================
# 6. Function Entry Selection (Section 3)
# ===========================================================================


def test_function_selection_precedence():
    """Test function selection precedence: explicit name, main, single, ambiguous error."""
    # 1. Explicit name requested
    code_multi = """
    void f1() {}
    void f2() {}
    """
    res_f2 = analyze_source(code_multi, function_name="f2")
    assert res_f2.selected_function == "f2"
    assert "Explicitly selected" in res_f2.selection_rationale

    # 2. Main selected automatically
    code_with_main = """
    void helper() {}
    int main() { return 0; }
    """
    res_main = analyze_source(code_with_main)
    assert res_main.selected_function == "main"
    assert "Selected entry point 'main'" in res_main.selection_rationale

    # 3. Single function selected automatically
    code_single = """
    void only_one() {}
    """
    res_single = analyze_source(code_single)
    assert res_single.selected_function == "only_one"
    assert "Selected sole function" in res_single.selection_rationale

    # 4. Ambiguous (multiple functions, no main, no name provided) -> error
    with pytest.raises(FunctionSelectionError) as exc_info:
        analyze_source(code_multi)
    assert "Ambiguous" in str(exc_info.value)
    assert "f1" in exc_info.value.available_functions
    assert "f2" in exc_info.value.available_functions

    # 5. Non-existent requested name -> error
    with pytest.raises(FunctionSelectionError) as exc_info:
        analyze_source(code_single, function_name="does_not_exist")
    assert "not found" in str(exc_info.value)


# ===========================================================================
# 7. Input-Size Convention & Symbols (Section 1)
# ===========================================================================


def test_named_constant_as_input_size():
    """Compile-time named constants (#define, const int) are input-size symbols under assumption."""
    code_const = """
    const int N = 1000;
    int main() {
        for (int i = 0; i < N; ++i) {}
        return 0;
    }
    """
    res = analyze_source(code_const)
    assert res.taxonomy_label == "ON"
    assert ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value in res.reason_codes
    assert "NAMED_CONSTANT_AS_INPUT_SIZE" in res.explanation


def test_cin_variable_as_input_size():
    """Variables assigned from cin are treated as input size symbols."""
    code_cin = """
    #include <iostream>
    int main() {
        int n;
        std::cin >> n;
        for (int i = 0; i < n; ++i) {}
        return 0;
    }
    """
    res = analyze_source(code_cin)
    assert res.taxonomy_label == "ON"
    assert res.exactness == "STRUCTURAL"


def test_multiple_input_symbols_trigger_single_input_symbol_assumption():
    """Multiple distinct input symbols record the SINGLE_INPUT_SYMBOL assumption."""
    code = """
    void solve(int n, int m) {
        for (int i = 0; i < n; ++i) {}
        for (int j = 0; j < m; ++j) {}
    }
    """
    res = analyze_source(code, function_name="solve")
    assert res.taxonomy_label == "ON"
    assert ReasonCode.SINGLE_INPUT_SYMBOL.value in res.reason_codes
    assert "SINGLE_INPUT_SYMBOL" in res.explanation


# ===========================================================================
# 8. Diagnostics and Unsupported IR Scoping (Section 7)
# ===========================================================================


def test_error_diagnostic_inside_selected_function_yields_incomplete_ir():
    """Error diagnostic inside the selected function makes result UNKNOWN with INCOMPLETE_IR."""
    ir = build_ir_from_source("void test() {}")
    fn = ir.functions[0]
    # Inject error diagnostic located inside test()'s span
    ir.diagnostics.append(
        IRDiagnostic(
            severity=DiagnosticSeverity.ERROR.value,
            message="Type error in test",
            span=fn.span,
        )
    )
    res = analyze_complexity(ir, function_name="test")
    assert res.taxonomy_label == "UNKNOWN"
    assert ReasonCode.INCOMPLETE_IR.value in res.reason_codes


def test_error_diagnostic_outside_selected_function_does_not_poison():
    """Error diagnostic outside selected function adds PARTIAL_IR note and does not poison."""
    code = """
    void bad_func() {}
    void good_func() {
        for (int i = 0; i < 10; ++i) {}
    }
    """
    ir = build_ir_from_source(code)
    bad_fn = next(f for f in ir.functions if f.name == "bad_func")
    # Add error diagnostic located on bad_func
    ir.diagnostics.append(
        IRDiagnostic(
            severity=DiagnosticSeverity.ERROR.value,
            message="Syntax problem in bad_func",
            span=bad_fn.span,
        )
    )

    res = analyze_complexity(ir, function_name="good_func")
    assert res.taxonomy_label == "O1"
    assert res.exactness == "STRUCTURAL"
    assert any("PARTIAL_IR" in note for note in res.notes)
    assert ReasonCode.PARTIAL_IR.value in res.reason_codes


def test_unsupported_node_inside_function_yields_unsupported_construct():
    """UnsupportedIR node inside function yields UNKNOWN with UNSUPPORTED_CONSTRUCT."""
    ir = build_ir_from_source("void test() {}")
    fn = ir.functions[0]
    fn.body.append(
        UnsupportedIR(
            id="unsup_1",
            kind=UnsupportedKind.ASM_STMT,
            raw_spelling="__asm__",
            reason="Inline assembly is unsupported",
            span=fn.span,
        )
    )
    res = analyze_complexity(ir, function_name="test")
    assert res.taxonomy_label == "UNKNOWN"
    assert ReasonCode.UNSUPPORTED_CONSTRUCT.value in res.reason_codes
    assert "unsup_1" in res.evidence_node_ids


def test_unsupported_node_outside_function_does_not_poison():
    """UnsupportedIR node outside function adds PARTIAL_IR note without poisoning."""
    code = """
    void good_func() {
        for (int i = 0; i < 5; ++i) {}
    }
    """
    ir = build_ir_from_source(code)
    # Add unsupported construct to program level outside good_func
    ir.unsupported_constructs.append(
        UnsupportedIR(
            id="global_unsup",
            kind=UnsupportedKind.COMPLEX_POINTER,
            raw_spelling="void* p",
            reason="Pointer aliasing",
            span=SourceSpan("main.cpp", 100, 1, 100, 10),
        )
    )
    res = analyze_complexity(ir, function_name="good_func")
    assert res.taxonomy_label == "O1"
    assert any("PARTIAL_IR" in note for note in res.notes)


# ===========================================================================
# 9. Rule Trace & Determinism (Section 2)
# ===========================================================================


def test_rule_trace_fields_and_determinism():
    """Rule trace contains rule_id, node_ids, input_classes, output_class, deterministic output."""
    code = """
    void solve(int n) {
        for (int i = 0; i < n; ++i) {}
        for (int j = 0; j < n; ++j) {}
    }
    """
    res1 = analyze_source(code, function_name="solve")
    res2 = analyze_source(code, function_name="solve")

    # Trace inspection
    assert len(res1.rule_trace) >= 2
    for entry in res1.rule_trace:
        assert entry.rule_id is not None
        assert isinstance(entry.node_ids, list)
        assert isinstance(entry.input_classes, list)
        assert isinstance(entry.output_class, ComplexityClass)

    # Identical deterministic output across runs
    assert res1.to_dict() == res2.to_dict()
    assert res1.explanation == res2.explanation


# ===========================================================================
# 10. Audit Regression Tests (Decreasing Loops, Directions, Constructors)
# ===========================================================================


def test_decreasing_loop_regression():
    """Decreasing loops from input symbol down to 0 are O(n); literal starts are O(1)."""
    code_input = """
    void solve(int n) {
        for (int i = n; i > 0; --i) {}
    }
    """
    res_in = analyze_source(code_input, function_name="solve")
    assert res_in.taxonomy_label == "ON"
    assert res_in.exactness == "STRUCTURAL"
    assert ReasonCode.LINEAR_LOOP.value in res_in.reason_codes

    code_ge = """
    void solve(int n) {
        for (int i = n; i >= 1; --i) {}
    }
    """
    res_ge = analyze_source(code_ge, function_name="solve")
    assert res_ge.taxonomy_label == "ON"
    assert res_ge.exactness == "STRUCTURAL"

    code_while = """
    void solve(int n) {
        while (n > 0) {
            --n;
        }
    }
    """
    res_while = analyze_source(code_while, function_name="solve")
    assert res_while.taxonomy_label == "ON"

    code_literal = """
    void solve() {
        for (int i = 100; i > 0; --i) {}
    }
    """
    res_lit = analyze_source(code_literal, function_name="solve")
    assert res_lit.taxonomy_label == "O1"
    assert res_lit.exactness == "STRUCTURAL"
    assert ReasonCode.CONSTANT_BOUND_LOOP.value in res_lit.reason_codes


def test_direction_compatibility_regression():
    """Incompatible update and comparison directions yield UNKNOWN or O(1) for 0 executions."""
    # Incompatible increasing with >
    code_incomp_inc = """
    void solve(int n) {
        for (int i = 0; i > n; ++i) {}
    }
    """
    res_inc = analyze_source(code_incomp_inc, function_name="solve")
    assert res_inc.taxonomy_label == "UNKNOWN"
    assert res_inc.exactness == "UNKNOWN"
    assert ReasonCode.UNKNOWN_BOUND.value in res_inc.reason_codes

    # Incompatible decreasing with <
    code_incomp_dec = """
    void solve(int n) {
        for (int i = 0; i < n; --i) {}
    }
    """
    res_dec = analyze_source(code_incomp_dec, function_name="solve")
    assert res_dec.taxonomy_label == "UNKNOWN"
    assert res_dec.exactness == "UNKNOWN"
    assert ReasonCode.UNKNOWN_BOUND.value in res_dec.reason_codes

    # Literal zero-execution loops
    code_zero_inc = """
    void solve() {
        for (int i = 0; i > 10; ++i) {}
    }
    """
    res_zero_inc = analyze_source(code_zero_inc, function_name="solve")
    assert res_zero_inc.taxonomy_label == "O1"

    code_zero_dec = """
    void solve() {
        for (int i = 10; i < 5; --i) {}
    }
    """
    res_zero_dec = analyze_source(code_zero_dec, function_name="solve")
    assert res_zero_dec.taxonomy_label == "O1"


def test_container_constructor_regression():
    """Default is O(1), input-sized is O(n), literal-sized is O(1), range is UNKNOWN."""
    # Default construction
    code_default = """
    #include <vector>
    void solve() {
        std::vector<int> v;
    }
    """
    res_def = analyze_source(code_default, function_name="solve")
    assert res_def.taxonomy_label == "O1"
    assert ReasonCode.CONSTANT_WORK.value in res_def.reason_codes

    # Input-sized construction
    code_sized = """
    #include <vector>
    void solve(int n) {
        std::vector<int> v(n);
    }
    """
    res_sized = analyze_source(code_sized, function_name="solve")
    assert res_sized.taxonomy_label == "ON"
    assert res_sized.exactness == "STRUCTURAL"
    assert ReasonCode.LINEAR_LOOP.value in res_sized.reason_codes
    assert any(entry.rule_id == "RULE_CONTAINER_CONSTRUCTOR" for entry in res_sized.rule_trace)

    # Literal-sized construction
    code_lit = """
    #include <vector>
    void solve() {
        std::vector<int> v(10);
    }
    """
    res_lit = analyze_source(code_lit, function_name="solve")
    assert res_lit.taxonomy_label == "O1"

    # Range construction yields UNKNOWN
    code_range = """
    #include <vector>
    void solve(std::vector<int> a) {
        std::vector<int> v(a.begin(), a.end());
    }
    """
    res_range = analyze_source(code_range, function_name="solve")
    assert res_range.taxonomy_label == "UNKNOWN"
    assert ReasonCode.UNSUPPORTED_CALL.value in res_range.reason_codes


def test_unified_loop_classifier_consistency():
    """Verify consistency across loop categories between engine and feature extractor."""
    from complexity_lens.features import extract_features
    from complexity_lens.ir.loop_classification import (
        LoopBoundCategory,
        classify_loop_structural,
    )

    cases = [
        # 1. Linear loop
        (
            "void f(int n) { for (int i = 0; i < n; ++i) {} }",
            LoopBoundCategory.LINEAR_BOUND,
            ComplexityClass.linear(),
            Exactness.STRUCTURAL,
            [ReasonCode.LINEAR_LOOP.value],
            "n",
            "ON",
            "linear",
        ),
        # 2. Constant loop
        (
            "void f() { for (int i = 0; i < 10; ++i) {} }",
            LoopBoundCategory.CONST_BOUND,
            ComplexityClass.constant(),
            Exactness.STRUCTURAL,
            [ReasonCode.CONSTANT_BOUND_LOOP.value],
            None,
            "O1",
            "const",
        ),
        # 3. Logarithmic update (input-sized)
        (
            "void f(int n) { for (int i = 1; i < n; i *= 2) {} }",
            LoopBoundCategory.LOG_UPDATE,
            ComplexityClass.logarithmic(),
            Exactness.STRUCTURAL,
            [ReasonCode.LOGARITHMIC_UPDATE.value],
            "n",
            "OLOGN",
            "log",
        ),
        # 4. Literal-bound logarithmic update
        (
            "void f() { for (int i = 1; i < 100; i *= 2) {} }",
            LoopBoundCategory.LOG_UPDATE,
            ComplexityClass.constant(),
            Exactness.STRUCTURAL,
            [ReasonCode.CONSTANT_BOUND_LOOP.value],
            None,
            "O1",
            "log",
        ),
        # 5. Named-constant logarithmic update
        (
            "const int N = 1000; void f() { for (int i = 1; i < N; i *= 2) {} }",
            LoopBoundCategory.LOG_UPDATE,
            ComplexityClass.logarithmic(),
            Exactness.STRUCTURAL,
            [
                ReasonCode.LOGARITHMIC_UPDATE.value,
                ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value,
            ],
            "N",
            "OLOGN",
            "log",
        ),
        # 6. Named-constant linear loop
        (
            "const int N = 1000; void f() { for (int i = 0; i < N; ++i) {} }",
            LoopBoundCategory.LINEAR_BOUND,
            ComplexityClass.linear(),
            Exactness.STRUCTURAL,
            [
                ReasonCode.LINEAR_LOOP.value,
                ReasonCode.NAMED_CONSTANT_AS_INPUT_SIZE.value,
            ],
            "N",
            "ON",
            "linear",
        ),
        # 7. Nested dependent bound
        (
            "void f(int n) { for (int i = 0; i < n; ++i) { for (int j = 0; j < i; ++j) {} } }",
            LoopBoundCategory.DEPENDENT_BOUND,
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.DEPENDENT_BOUND.value],
            None,
            "UNKNOWN",
            "dependent",
        ),
        # 8. Unsupported != comparison
        (
            "void f(int n) { for (int i = 0; i != n; ++i) {} }",
            LoopBoundCategory.UNKNOWN_BOUND,
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNKNOWN_BOUND.value],
            None,
            "UNKNOWN",
            "unknown",
        ),
        # 9. Reverse input-sized loop
        (
            "void f(int n) { for (int i = n; i > 0; --i) {} }",
            LoopBoundCategory.LINEAR_BOUND,
            ComplexityClass.linear(),
            Exactness.STRUCTURAL,
            [ReasonCode.LINEAR_LOOP.value],
            "n",
            "ON",
            "linear",
        ),
        # 10. Incompatible direction
        (
            "void f(int n) { for (int i = 0; i > n; ++i) {} }",
            LoopBoundCategory.UNKNOWN_BOUND,
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.UNKNOWN_BOUND.value],
            None,
            "UNKNOWN",
            "unknown",
        ),
        # 11. Loop variable modified inside body
        (
            "void f(int n) { for (int i = 0; i < n; ++i) { i += 2; } }",
            LoopBoundCategory.UNKNOWN_BOUND,
            ComplexityClass.unknown(),
            Exactness.UNKNOWN,
            [ReasonCode.LOOP_VAR_MODIFIED.value],
            None,
            "UNKNOWN",
            "unknown",
        ),
        # 12. Range-for container loop
        (
            "#include <vector>\nvoid f(std::vector<int>& v) { for (auto x : v) {} }",
            LoopBoundCategory.LINEAR_BOUND,
            ComplexityClass.linear(),
            Exactness.STRUCTURAL,
            [ReasonCode.LINEAR_LOOP.value],
            "v",
            "ON",
            "linear",
        ),
        # 13. Range-for fixed array loop
        (
            "#include <array>\nvoid f() { std::array<int, 5> arr; for (auto x : arr) {} }",
            LoopBoundCategory.CONST_BOUND,
            ComplexityClass.constant(),
            Exactness.STRUCTURAL,
            [ReasonCode.CONSTANT_BOUND_LOOP.value],
            None,
            "O1",
            "const",
        ),
    ]

    for (
        code,
        exp_cat,
        exp_class,
        exp_exact,
        exp_reasons,
        exp_sym,
        exp_tax,
        exp_feat_kind,
    ) in cases:
        prog = build_ir_from_source(code)
        fn = prog.functions[0]
        lp = fn.all_loops()[-1]
        param_names = {p[1] for p in fn.parameters if p[1]}

        # 1. Structural classifier (explicit category comparison)
        struct_res = classify_loop_structural(
            lp,
            local_constants=fn.local_constants,
            named_constants=fn.named_constants,
            cin_vars=fn.cin_vars,
            param_names=param_names,
        )
        assert struct_res.category == exp_cat

        # 2. Engine classify_loop contract
        iter_class, exactness, reasons, sym = classify_loop(
            lp,
            local_constants=fn.local_constants,
            named_constants=fn.named_constants,
            cin_vars=fn.cin_vars,
            param_names=param_names,
        )
        assert iter_class == exp_class
        assert exactness == exp_exact
        assert reasons == exp_reasons
        assert sym == exp_sym

        # 3. Full engine analysis result
        res = analyze_source(code)
        assert res.taxonomy_label == exp_tax

        # 4. Feature extractor
        feat = extract_features(prog)
        if exp_feat_kind == "linear":
            assert feat.n_loops_linear_bound >= 1
        elif exp_feat_kind == "const":
            assert feat.n_loops_const_bound >= 1
        elif exp_feat_kind == "log":
            assert feat.n_loops_log_update >= 1
        elif exp_feat_kind == "dependent":
            assert feat.n_dependent_bound_loops >= 1
        elif exp_feat_kind == "unknown":
            assert feat.n_loops_unknown_bound >= 1

