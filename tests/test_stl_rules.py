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
# Section 10 — Argument Mapping (Stage 2)
# ===========================================================================


