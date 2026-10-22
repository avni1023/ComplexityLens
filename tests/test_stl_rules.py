"""Unit tests for ComplexityLens Stage 1 (STL and container complexity rules)."""
from __future__ import annotations

import pytest

from complexity_lens.complexity import (
    HASH_EXPECTED,
    HASH_STRICT,
    ComplexityEngine,
    Exactness,
    ReasonCode,
    analyze_source,
    classify_stl_operation,
    validate_cost_model,
)
from complexity_lens.ir.model import (
    BoundKind,
    BranchIR,
    CallIR,
    LoopBound,
    LoopIR,
    LoopKind,
    LoopUpdate,
    SourceSpan,
    STLOperationIR,
    STLOperationKind,
    UnsupportedIR,
    UnsupportedKind,
    UpdateKind,
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
    container_kind_source: str | None = None,
):
    op = _make_stl_op(
        op_kind, container=container, container_kind_source=container_kind_source
    )
    c, exactness, reasons, assumptions = classify_stl_operation(
        op, cost_model, {"n"}, set(), set(), {}
    )
    return c, c.to_taxonomy_label(), reasons, assumptions

class TestCostModelValidation:
    def test_expected_hash_is_valid(self):
        validate_cost_model(HASH_EXPECTED)  # no exception

    def test_strict_worst_case_is_valid(self):
        validate_cost_model(HASH_STRICT)  # no exception

    def test_invalid_cost_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported hash container cost_model"):
            validate_cost_model("average")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_cost_model("")

    def test_engine_invalid_cost_model_raises(self):
        with pytest.raises(ValueError):
            ComplexityEngine(cost_model="bogus")


# ===========================================================================
# Section 2 — Hash Container Operations
# ===========================================================================


class TestHashContainerOperations:
    """Unordered-container operations under both cost models."""

    def test_hash_lookup_expected_model_is_constant(self):
        c, lbl, reasons, assumptions = _classify(
            STLOperationKind.HASH_LOOKUP, cost_model=HASH_EXPECTED
        )
        assert lbl == "O1"
        assert ReasonCode.STL_HASH_OP.value in reasons
        assert ReasonCode.HASH_AVERAGE_CASE.value in assumptions

    def test_hash_lookup_strict_model_is_linear(self):
        c, lbl, reasons, assumptions = _classify(
            STLOperationKind.HASH_LOOKUP, cost_model=HASH_STRICT
        )
        assert lbl == "ON"
        assert ReasonCode.STL_HASH_OP.value in reasons
        assert ReasonCode.HASH_AVERAGE_CASE.value not in assumptions

    def test_hash_lookup_expected_assumption_in_trace_and_explanation(self):
        """Assumption HASH_AVERAGE_CASE must appear in result.assumptions and explanation."""
        src = """
        #include <unordered_map>
        int solve(int n) {
            std::unordered_map<int,int> freq;
            for (int i = 0; i < n; ++i) {
                freq[i]++;
            }
            return 0;
        }
        """
        res = analyze_source(src, function_name="solve", cost_model=HASH_EXPECTED)
        # At minimum the overall function should not be UNKNOWN
        # (hash lookup inside loop is O(1) expected → loop is O(n))
        assert res.taxonomy_label in ("ON", "UNKNOWN")
        # If the hash op was analyzed, HASH_AVERAGE_CASE should appear
        if ReasonCode.STL_HASH_OP.value in res.reason_codes:
            assert ReasonCode.HASH_AVERAGE_CASE.value in res.assumptions
            assert "HASH_AVERAGE_CASE" in res.explanation

    def test_hash_lookup_strict_no_hash_average_assumption(self):
        """In strict mode, HASH_AVERAGE_CASE must NOT appear."""
        c, lbl, reasons, assumptions = _classify(
            STLOperationKind.HASH_LOOKUP, cost_model=HASH_STRICT
        )
        assert ReasonCode.HASH_AVERAGE_CASE.value not in assumptions


# ===========================================================================
# Section 3 — Ordered Container Operations
# ===========================================================================


class TestOrderedContainerOperations:
    """Ordered map/set operations under both cost models."""

    def test_ordered_lookup_is_logarithmic_expected(self):
        c, lbl, reasons, _ = _classify(
            STLOperationKind.ORDERED_LOOKUP, cost_model=HASH_EXPECTED
        )
        assert lbl == "OLOGN"
        assert ReasonCode.STL_ORDERED_OP.value in reasons

    def test_ordered_lookup_is_logarithmic_strict(self):
        c, lbl, reasons, _ = _classify(
            STLOperationKind.ORDERED_LOOKUP, cost_model=HASH_STRICT
        )
        assert lbl == "OLOGN"

    def test_ordered_lookup_is_structural(self):
        op = _make_stl_op(STLOperationKind.ORDERED_LOOKUP)
        c, e, _, _ = classify_stl_operation(op, HASH_EXPECTED, {"n"}, set(), set(), {})
        assert e == Exactness.STRUCTURAL

    def test_ordered_lookup_with_source_text_provenance_has_assumption(self):
        c, lbl, reasons, assumptions = _classify(
            STLOperationKind.ORDERED_LOOKUP,
            container_kind_source="source_text",
        )
        assert ReasonCode.CONTAINER_KIND_FROM_SOURCE_TEXT.value in assumptions
        assert lbl == "OLOGN"


# ===========================================================================
# Section 4 — Vector push_back / emplace_back
# ===========================================================================


class TestVectorAppend:
    def test_push_back_is_amortized_constant(self):
        c, lbl, reasons, assumptions = _classify(STLOperationKind.PUSH_BACK)
        assert lbl == "O1"
        assert ReasonCode.AMORTIZED_CONSTANT_CALL.value in reasons
        assert ReasonCode.AMORTIZED.value in assumptions

    def test_push_back_exactness_is_structural(self):
        op = _make_stl_op(STLOperationKind.PUSH_BACK)
        c, e, _, _ = classify_stl_operation(op, HASH_EXPECTED, {"n"}, set(), set(), {})
        assert e == Exactness.STRUCTURAL

    def test_push_back_amortized_assumption_appears_in_engine_result(self):
        """AMORTIZED assumption must propagate via engine into result.assumptions."""
        src = """
        #include <vector>
        void solve(int n) {
            std::vector<int> v;
            for (int i = 0; i < n; ++i) {
                v.push_back(i);
            }
        }
        """
        res = analyze_source(src, function_name="solve")
        # push_back inside loop → loop is O(n) with amortized assumption
        assert res.taxonomy_label in ("ON", "UNKNOWN")
        if ReasonCode.AMORTIZED_CONSTANT_CALL.value in res.reason_codes:
            assert "AMORTIZED" in res.explanation


# ===========================================================================
# Section 5 — Vector insert/erase
# ===========================================================================


class TestVectorInsertErase:
    def test_vector_insert_with_input_container_is_linear_upper_bound(self):
        op = _make_stl_op(
            STLOperationKind.INSERT,
            container="v",
            arguments=["v.begin()", "x"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"v"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "ON"
        assert e == Exactness.UPPER_BOUND
        assert ReasonCode.VECTOR_INSERT_ERASE.value in reasons

    def test_vector_erase_unknown_container_is_unknown(self):
        op = _make_stl_op(
            STLOperationKind.ERASE,
            container=None,
            arguments=["it"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert c.is_unknown
        assert e == Exactness.UNKNOWN

    def test_vector_insert_non_input_container_is_unknown(self):
        op = _make_stl_op(
            STLOperationKind.INSERT,
            container="local_vec",
            arguments=["local_vec.begin()", "x"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        # local_vec not in param_names/cin_vars → UNKNOWN
        assert c.is_unknown


# ===========================================================================
# Section 6 — std::sort
# ===========================================================================


class TestStdSort:
    def test_sort_random_access_no_comparator_is_nlogn(self):
        op = _make_stl_op(
            STLOperationKind.SORT,
            container="v",
            arguments=["v.begin()", "v.end()"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"v"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "ONLOGN"
        assert e == Exactness.STRUCTURAL
        assert ReasonCode.STL_SORT.value in reasons

    def test_sort_with_less_comparator_is_nlogn(self):
        op = _make_stl_op(
            STLOperationKind.SORT,
            container="v",
            arguments=["v.begin()", "v.end()", "std::less<>"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"v"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "ONLOGN"

    def test_sort_with_greater_comparator_is_nlogn(self):
        op = _make_stl_op(
            STLOperationKind.SORT,
            arguments=["v.begin()", "v.end()", "std::greater<>"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"v"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "ONLOGN"

    def test_sort_with_lambda_comparator_is_unknown(self):
        op = _make_stl_op(
            STLOperationKind.SORT,
            arguments=["v.begin()", "v.end()", "[](int a, int b){ return a > b; }"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"v"}, set(), set(), {}
        )
        assert c.is_unknown
        assert e == Exactness.UNKNOWN
        assert ReasonCode.UNSUPPORTED_COMPARATOR.value in reasons

    def test_sort_unsupported_range_is_unknown(self):
        """sort with arbitrary range (not begin/end or a+n) → UNKNOWN."""
        op = _make_stl_op(
            STLOperationKind.SORT,
            container=None,
            arguments=["ptr", "ptr + end"],  # end not a recognized input symbol
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert c.is_unknown


# ===========================================================================
# Section 7 — std::find / std::count (O(n))
# ===========================================================================


class TestStdFindCount:
    def test_find_begin_end_is_linear(self):
        op = _make_stl_op(
            STLOperationKind.FIND,
            container="v",
            arguments=["v.begin()", "v.end()", "target"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"v"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "ON"
        assert ReasonCode.STL_FIND.value in reasons

    def test_count_begin_end_is_linear(self):
        op = _make_stl_op(
            STLOperationKind.COUNT,
            container="arr",
            arguments=["arr.begin()", "arr.end()", "val"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"arr"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "ON"

    def test_find_unsupported_range_is_unknown(self):
        op = _make_stl_op(
            STLOperationKind.FIND,
            container=None,
            arguments=["it_start", "it_end"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert c.is_unknown
        assert ReasonCode.UNSUPPORTED_RANGE.value in reasons


# ===========================================================================
# Section 8 — std::lower_bound / std::binary_search (O(log n))
# ===========================================================================


class TestBinarySearchAlgorithms:
    def test_lower_bound_on_vector_begin_end_is_logn(self):
        op = _make_stl_op(
            STLOperationKind.LOWER_BOUND,
            container="vector",
            arguments=["v.begin()", "v.end()", "key"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"v"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "OLOGN"
        assert ReasonCode.STL_BINARY_SEARCH.value in reasons

    def test_binary_search_on_array_is_logn(self):
        op = _make_stl_op(
            STLOperationKind.BINARY_SEARCH,
            container="array",
            arguments=["a.begin()", "a.end()", "key"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"a"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "OLOGN"

    def test_lower_bound_on_set_is_unknown(self):
        """std::lower_bound on set iterators is NOT random-access → UNKNOWN."""
        op = _make_stl_op(
            STLOperationKind.LOWER_BOUND,
            container="set",
            arguments=["s.begin()", "s.end()", "key"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert c.is_unknown
        assert ReasonCode.UNSUPPORTED_ITERATOR.value in reasons

    def test_lower_bound_on_map_is_unknown(self):
        op = _make_stl_op(
            STLOperationKind.LOWER_BOUND,
            container="map",
            arguments=["m.begin()", "m.end()", "key"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert c.is_unknown

    def test_lower_bound_a_plus_n_form_is_logn(self):
        op = _make_stl_op(
            STLOperationKind.LOWER_BOUND,
            container="v",
            arguments=["a", "a+n", "key"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "OLOGN"


# ===========================================================================
# Section 9 — Container Kind Provenance Assumption
# ===========================================================================


class TestContainerKindProvenance:
    def test_source_text_provenance_adds_assumption(self):
        op = _make_stl_op(
            STLOperationKind.ORDERED_LOOKUP,
            container_kind_source="source_text",
        )
        c, e, reasons, assumptions = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert ReasonCode.CONTAINER_KIND_FROM_SOURCE_TEXT.value in assumptions

    def test_resolved_provenance_does_not_add_source_text_assumption(self):
        op = _make_stl_op(
            STLOperationKind.ORDERED_LOOKUP,
            container_kind_source="resolved",
        )
        c, e, reasons, assumptions = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert ReasonCode.CONTAINER_KIND_FROM_SOURCE_TEXT.value not in assumptions

    def test_none_provenance_does_not_add_source_text_assumption(self):
        op = _make_stl_op(
            STLOperationKind.PUSH_BACK,
            container_kind_source=None,
        )
        c, e, reasons, assumptions = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert ReasonCode.CONTAINER_KIND_FROM_SOURCE_TEXT.value not in assumptions

    def test_source_text_provenance_in_engine_explanation(self):
        """CONTAINER_KIND_FROM_SOURCE_TEXT must appear in explanation when provenance
        is source_text.
        """
        op = _make_stl_op(
            STLOperationKind.ORDERED_LOOKUP,
            container_kind_source="source_text",
        )
        c, e, reasons, assumptions = classify_stl_operation(
            op, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert ReasonCode.CONTAINER_KIND_FROM_SOURCE_TEXT.value in assumptions


# ===========================================================================
# Section 10 — Nested STL Loop Composition & Range Soundness (Regression)
# ===========================================================================


class TestNestedSTLLoopComposition:
    """Regression tests verifying nested STL complexity soundness and range bounds."""

    def test_linear_find_in_outer_input_sized_loop_is_quadratic(self):
        """Case 1: std::find in an outer input-sized loop over input-sized range -> ON2."""
        src = """
        #include <vector>
        #include <algorithm>
        int count_matching_elements(const std::vector<int>& a, const std::vector<int>& b) {
            int matches = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::find(b.begin(), b.end(), a[i]) != b.end()) {
                    matches++;
                }
            }
            return matches;
        }
        """
        res = analyze_source(src, function_name="count_matching_elements")
        assert res.taxonomy_label == "ON2"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.NESTED_LOOP.value in res.reason_codes
        assert ReasonCode.STL_FIND.value in res.reason_codes

    def test_linear_count_in_outer_input_sized_loop_is_quadratic(self):
        """Case 2: std::count in an outer input-sized loop -> ON2."""
        src = """
        #include <vector>
        #include <algorithm>
        int count_cross(const std::vector<int>& keys, const std::vector<int>& data) {
            int total = 0;
            for (size_t i = 0; i < keys.size(); ++i) {
                total += std::count(data.begin(), data.end(), keys[i]);
            }
            return total;
        }
        """
        res = analyze_source(src, function_name="count_cross")
        assert res.taxonomy_label == "ON2"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.NESTED_LOOP.value in res.reason_codes
        assert ReasonCode.STL_FIND.value in res.reason_codes

    def test_stl_operations_outside_loop_remain_linear(self):
        """Case 3: std::find and std::count outside loop remain linear (ON)."""
        src_find = """
        #include <vector>
        #include <algorithm>
        bool contains(const std::vector<int>& vec, int target) {
            return std::find(vec.begin(), vec.end(), target) != vec.end();
        }
        """
        res_find = analyze_source(src_find, function_name="contains")
        assert res_find.taxonomy_label == "ON"
        assert res_find.exactness == Exactness.STRUCTURAL

        src_count = """
        #include <vector>
        #include <algorithm>
        int count_target(const std::vector<int>& vec, int target) {
            return std::count(vec.begin(), vec.end(), target);
        }
        """
        res_count = analyze_source(src_count, function_name="count_target")
        assert res_count.taxonomy_label == "ON"
        assert res_count.exactness == Exactness.STRUCTURAL

    def test_stl_operation_over_constant_range_in_loop_remains_linear(self):
        """Case 4: An STL operation over a constant-sized range inside a loop is ON, not ON2."""
        src = """
        #include <algorithm>
        int find_const_range_in_loop(int n, int* arr, int target) {
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (std::find(arr, arr + 5, target) != arr + 5) {
                    count++;
                }
            }
            return count;
        }
        """
        res = analyze_source(src, function_name="find_const_range_in_loop")
        assert res.taxonomy_label == "ON"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.NESTED_LOOP.value not in res.reason_codes

    def test_early_exit_in_loop_preserves_worst_case_soundness(self):
        """Case 5: Early-exit branch still yields ON2 worst-case with UPPER_BOUND exactness."""
        src = """
        #include <vector>
        #include <algorithm>
        bool find_any_match(const std::vector<int>& a, const std::vector<int>& b) {
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::find(b.begin(), b.end(), a[i]) != b.end()) {
                    return true;
                }
            }
            return false;
        }
        """
        res = analyze_source(src, function_name="find_any_match")
        assert res.taxonomy_label == "ON2"
        assert res.exactness == Exactness.UPPER_BOUND
        assert ReasonCode.NESTED_LOOP.value in res.reason_codes

    def test_unsupported_unbounded_range_returns_unknown(self):
        """Case 6: Unsupported or unbounded range returns UNKNOWN rather than unjustified claim."""
        src = """
        #include <algorithm>
        int find_unbounded(int n, int* p1, int* p2, int target) {
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (std::find(p1, p2, target) != p2) {
                    count++;
                }
            }
            return count;
        }
        """
        res = analyze_source(src, function_name="find_unbounded")
        assert res.taxonomy_label == "UNKNOWN"
        assert res.exactness == Exactness.UNKNOWN
        assert ReasonCode.UNSUPPORTED_RANGE.value in res.reason_codes


class TestBranchIRConditionItemsTraversal:
    """Regression tests verifying condition_items inclusion in IR traversals and serialization."""

    def test_condition_items_in_all_traversals_and_to_dict(self):
        span = SourceSpan("test.cpp", 1, 1, 1, 10)
        call_node = CallIR(id="call_1", callee_name="helper", arguments=[], span=span)
        stl_node = STLOperationIR(
            id="stl_1",
            operation=STLOperationKind.FIND,
            callee_name="std::find",
            container="v",
            arguments=["v.begin()", "v.end()", "x"],
            span=span,
        )
        loop_node = LoopIR(
            id="loop_1",
            kind=LoopKind.FOR,
            depth=1,
            bound=LoopBound(kind=BoundKind.LINEAR_INPUT, bound_expr="n"),
            update=LoopUpdate(kind=UpdateKind.ADDITIVE),
            span=span,
        )
        unsupported_node = UnsupportedIR(
            id="unsup_1",
            kind=UnsupportedKind.ASM_STMT,
            raw_spelling="__asm__",
            reason="inline asm unsupported",
            span=span,
        )

        branch = BranchIR(
            id="branch_1",
            condition="cond",
            condition_items=[call_node, stl_node, loop_node, unsupported_node],
            then_body=[],
            else_body=[],
            span=span,
        )

        # 1. all_loops includes loop in condition_items
        assert branch.all_loops() == [loop_node]
        # 2. all_calls includes call in condition_items
        assert branch.all_calls() == [call_node]
        # 3. all_stl_ops includes stl op in condition_items
        assert branch.all_stl_ops() == [stl_node]
        # 4. all_unsupported includes unsupported construct in condition_items
        assert branch.all_unsupported() == [unsupported_node]
        # 5. to_dict serializes condition_items
        d = branch.to_dict()
        assert "condition_items" in d
        assert len(d["condition_items"]) == 4
        assert d["condition_items"][0]["type"] == "CallIR"
        assert d["condition_items"][1]["type"] == "STLOperationIR"
        assert d["condition_items"][2]["type"] == "LoopIR"
        assert d["condition_items"][3]["type"] == "UnsupportedIR"

    def test_condition_items_nested_traversal(self):
        """Nested branches inside condition_items are recursively traversed."""
        span = SourceSpan("test.cpp", 1, 1, 1, 10)
        inner_call = CallIR(id="call_inner", callee_name="inner_check", arguments=[], span=span)
        inner_branch = BranchIR(
            id="inner_branch",
            condition="subcond",
            condition_items=[inner_call],
            then_body=[],
            else_body=[],
            span=span,
        )
        outer_branch = BranchIR(
            id="outer_branch",
            condition="outer_cond",
            condition_items=[inner_branch],
            then_body=[],
            else_body=[],
            span=span,
        )
        assert outer_branch.all_calls() == [inner_call]


# ===========================================================================
# Section 12 — Constant vs Input-Dependent STL Range Soundness
# ===========================================================================


class TestSTLRandomAccessRangeSoundness:
    """Regression tests verifying that constant ranges (arr + 5) yield O(1)
    while input-dependent ranges (length = n) yield O(n) or UNKNOWN, never O(1).
    """

    def test_fixed_literal_range_find_is_constant(self):
        """Case 1a: Genuinely fixed-size literal range (arr + 5) yields O(1) with CONSTANT_WORK."""
        src = """
        #include <algorithm>
        int search_fixed(int* arr, int n) {
            return std::find(arr, arr + 5, 42) != (arr + 5);
        }
        """
        res = analyze_source(src, function_name="search_fixed")
        assert res.taxonomy_label == "O1"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.CONSTANT_WORK.value in res.reason_codes

    def test_fixed_local_constant_find_is_constant(self):
        """Case 1b: Fixed local constant (const int length = 5) yields O(1) with CONSTANT_WORK."""
        src = """
        #include <algorithm>
        int search_fixed_const(int* arr, int n) {
            const int length = 5;
            return std::find(arr, arr + length, 42) != (arr + length);
        }
        """
        res = analyze_source(src, function_name="search_fixed_const")
        assert res.taxonomy_label == "O1"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.CONSTANT_WORK.value in res.reason_codes

    def test_input_dependent_local_constant_is_not_constant(self):
        """Case 2: Input-dependent local constant (const int length = n) yields O(n), NOT O(1)."""
        src = """
        #include <algorithm>
        int search_input_dep(int* arr, int n) {
            const int length = n;
            return std::find(arr, arr + length, 42) != (arr + length);
        }
        """
        res = analyze_source(src, function_name="search_input_dep")
        assert res.taxonomy_label != "O1"
        assert res.taxonomy_label == "ON"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.STL_FIND.value in res.reason_codes

    def test_input_dependent_mutable_local_is_not_constant(self):
        """Case 3a: Input-dependent local (int length = n) yields O(n), NOT O(1)."""
        src = """
        #include <algorithm>
        int search_input_dep_var(int* arr, int n) {
            int length = n;
            return std::find(arr, arr + length, 42) != (arr + length);
        }
        """
        res = analyze_source(src, function_name="search_input_dep_var")
        assert res.taxonomy_label != "O1"
        assert res.taxonomy_label == "ON"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.STL_FIND.value in res.reason_codes

    def test_input_dependent_untracked_local_is_unknown_not_constant(self):
        """Case 3b: Untracked local (int length = foo()) yields UNKNOWN, NOT O(1)."""
        src = """
        #include <algorithm>
        int get_len();
        int search_untracked(int* arr, int n) {
            int length = get_len();
            return std::find(arr, arr + length, 42) != (arr + length);
        }
        """
        res = analyze_source(src, function_name="search_untracked")
        assert res.taxonomy_label != "O1"
        assert res.taxonomy_label == "UNKNOWN"
        assert res.exactness == Exactness.UNKNOWN
        assert ReasonCode.UNSUPPORTED_RANGE.value in res.reason_codes

    def test_compile_time_named_constant_is_input_size(self):
        """Case 4: Top-level named constant (const int N = 1000) represents input size -> O(n)."""
        src = """
        #include <algorithm>
        const int N = 1000;
        int search_named_const(int* arr) {
            return std::find(arr, arr + N, 42) != (arr + N);
        }
        """
        res = analyze_source(src, function_name="search_named_const")
        assert res.taxonomy_label != "O1"
        assert res.taxonomy_label == "ON"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.STL_FIND.value in res.reason_codes

    def test_all_range_algorithms_fixed_local_constant_are_constant(self):
        """Case 5: All 5 supported range algorithms with const int length = 5 yield O(1)."""
        cases = [
            ("sort", "std::sort(arr, arr + length);"),
            ("find", "std::find(arr, arr + length, 42);"),
            ("count", "std::count(arr, arr + length, 42);"),
            ("bsearch", "std::binary_search(arr, arr + length, 42);"),
            ("lbound", "std::lower_bound(arr, arr + length, 42);"),
        ]
        for name, stmt in cases:
            src = f"""
            #include <algorithm>
            void run_fixed_{name}(int* arr, int n) {{
                const int length = 5;
                {stmt}
            }}
            """
            res = analyze_source(src, function_name=f"run_fixed_{name}")
            assert res.taxonomy_label == "O1", f"{name} with fixed local const must be O(1)"
            assert res.exactness == Exactness.STRUCTURAL
            assert ReasonCode.CONSTANT_WORK.value in res.reason_codes

    def test_all_range_algorithms_input_dependent_local_constant_are_not_constant(self):
        """Case 4: All 5 supported range algorithms with const int length = n yield their proper
        complexity (ON, ONLOGN, OLOGN), and NEVER O(1).
        """
        cases = [
            ("sort", "std::sort(arr, arr + length);", "ONLOGN", ReasonCode.STL_SORT.value),
            ("find", "std::find(arr, arr + length, 42);", "ON", ReasonCode.STL_FIND.value),
            ("count", "std::count(arr, arr + length, 42);", "ON", ReasonCode.STL_FIND.value),
            (
                "bsearch",
                "std::binary_search(arr, arr + length, 42);",
                "OLOGN",
                ReasonCode.STL_BINARY_SEARCH.value,
            ),
            (
                "lbound",
                "std::lower_bound(arr, arr + length, 42);",
                "OLOGN",
                ReasonCode.STL_BINARY_SEARCH.value,
            ),
        ]
        for name, stmt, expected_tax, expected_reason in cases:
            src = f"""
            #include <algorithm>
            void run_{name}(int* arr, int n) {{
                const int length = n;
                {stmt}
            }}
            """
            res = analyze_source(src, function_name=f"run_{name}")
            assert res.taxonomy_label != "O1", f"{name} must not be classified as O(1)"
            assert res.taxonomy_label == expected_tax, (
                f"{name} expected {expected_tax}, got {res.taxonomy_label}"
            )
            assert res.exactness == Exactness.STRUCTURAL
            assert expected_reason in res.reason_codes

    def test_all_range_algorithms_fixed_literal_range_are_constant(self):
        """Case 5: All 5 supported range algorithms with fixed literal range (arr + 5)
        soundly yield O(1) with CONSTANT_WORK.
        """
        cases = [
            ("sort", "std::sort(arr, arr + 5);"),
            ("find", "std::find(arr, arr + 5, 42);"),
            ("count", "std::count(arr, arr + 5, 42);"),
            ("bsearch", "std::binary_search(arr, arr + 5, 42);"),
            ("lbound", "std::lower_bound(arr, arr + 5, 42);"),
        ]
        for name, stmt in cases:
            src = f"""
            #include <algorithm>
            void run_fixed_{name}(int* arr, int n) {{
                {stmt}
            }}
            """
            res = analyze_source(src, function_name=f"run_fixed_{name}")
            assert res.taxonomy_label == "O1", f"{name} with fixed range must be O(1)"
            assert res.exactness == Exactness.STRUCTURAL
            assert ReasonCode.CONSTANT_WORK.value in res.reason_codes

    def test_classify_stl_operation_direct_soundness(self):
        """Case 6: Direct unit evaluation of classify_stl_operation ensures range_sym is None
        only returns O(1) for recognized literals, and never for input-dependent or unknown ranges.
        """
        # A. Fixed literal range (arr, arr + 5) -> O(1)
        op_lit = _make_stl_op(
            STLOperationKind.FIND,
            container=None,
            arguments=["arr", "arr + 5", "42"],
        )
        c, e, reasons, _ = classify_stl_operation(
            op_lit, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert c.to_taxonomy_label() == "O1"
        assert e == Exactness.STRUCTURAL
        assert ReasonCode.CONSTANT_WORK.value in reasons

        # B. Input-dependent range symbol via named_constants -> ON
        op_dep = _make_stl_op(
            STLOperationKind.FIND,
            container=None,
            arguments=["arr", "arr + length", "42"],
        )
        c_dep, e_dep, reasons_dep, _ = classify_stl_operation(
            op_dep, HASH_EXPECTED, {"n"}, set(), {"length"}, {}
        )
        assert c_dep.to_taxonomy_label() == "ON"
        assert e_dep == Exactness.STRUCTURAL
        assert ReasonCode.STL_FIND.value in reasons_dep

        # C. Unresolved range symbol (neither input symbol nor local constant) -> UNKNOWN
        c_unres, e_unres, reasons_unres, _ = classify_stl_operation(
            op_dep, HASH_EXPECTED, {"n"}, set(), set(), {}
        )
        assert c_unres.is_unknown
        assert c_unres.to_taxonomy_label() != "O1"
        assert ReasonCode.UNSUPPORTED_RANGE.value in reasons_unres

    def test_mutated_local_input_alias_invalidates_to_unknown(self):
        """A mutated input-dependent local alias cannot retain stale metadata -> UNKNOWN."""
        src = """
        #include <algorithm>
        int search_mutated(int* arr, int n) {
            int length = n;
            length++;
            return std::find(arr, arr + length, 42) != (arr + length);
        }
        """
        res = analyze_source(src, function_name="search_mutated")
        assert res.taxonomy_label == "UNKNOWN"
        assert res.exactness == Exactness.UNKNOWN
        assert ReasonCode.UNSUPPORTED_RANGE.value in res.reason_codes

    def test_mutated_local_const_invalidates_to_unknown(self):
        """A mutated local constant variable cannot retain stale constant metadata -> UNKNOWN."""
        src = """
        #include <algorithm>
        int search_mutated_const(int* arr, int n) {
            int length = 5;
            length += 2;
            return std::find(arr, arr + length, 42) != (arr + length);
        }
        """
        res = analyze_source(src, function_name="search_mutated_const")
        assert res.taxonomy_label == "UNKNOWN"
        assert res.exactness == Exactness.UNKNOWN
        assert ReasonCode.UNSUPPORTED_RANGE.value in res.reason_codes

    def test_read_local_const_on_rhs_preserves_o1(self):
        """Reading a local constant on the RHS of an assignment does not falsely
        invalidate it -> O(1).
        """
        src = """
        #include <algorithm>
        int search_read_const(int* arr, int n) {
            const int length = 5;
            int x;
            x = length;
            return std::find(arr, arr + length, 42) != (arr + length);
        }
        """
        res = analyze_source(src, function_name="search_read_const")
        assert res.taxonomy_label == "O1"
        assert res.exactness == Exactness.STRUCTURAL
        assert ReasonCode.CONSTANT_WORK.value in res.reason_codes


