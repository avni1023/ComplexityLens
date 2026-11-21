"""tests/test_presentation.py — Unit tests for presentation and complexity label formatting.

Validates:
1. Exact mapping of every canonical internal label to proper Big-O notation.
2. Case-insensitivity and whitespace tolerance.
3. Enum handling (TaxonomyLabel instances).
4. UNKNOWN, None, and empty input handling.
5. Graceful pass-through of unexpected input values.
6. Preservation of already formatted Big-O notation.
7. Explanation and narrative text formatting for rationales and summaries.
"""

from __future__ import annotations

import pytest

from complexity_lens.complexity.models import TaxonomyLabel
from complexity_lens.explanation import DetectedConstruct, SourceSpanInfo
from complexity_lens.presentation import (
    COMPLEXITY_DISPLAY_MAP,
    format_complexity_label,
    format_construct_display,
    format_explanation_text,
)


@pytest.mark.parametrize(
    ("internal_label", "expected_display"),
    [
        ("O1", "O(1)"),
        ("OLOGN", "O(log n)"),
        ("ON", "O(n)"),
        ("ONLOGN", "O(n log n)"),
        ("ON2", "O(n²)"),
        ("ON3", "O(n³)"),
        ("OEXP", "O(2ⁿ)"),
        ("UNKNOWN", "Unknown"),
    ],
)
def test_format_complexity_label_exact_canonical_mappings(
    internal_label: str, expected_display: str
) -> None:
    """Every internal complexity label maps to its exact required UI representation."""
    assert format_complexity_label(internal_label) == expected_display
    # Also verify mapping table directly
    assert COMPLEXITY_DISPLAY_MAP[internal_label] == expected_display


@pytest.mark.parametrize(
    ("internal_label", "expected_display"),
    [
        ("o1", "O(1)"),
        ("ologn", "O(log n)"),
        ("on", "O(n)"),
        ("onlogn", "O(n log n)"),
        ("on2", "O(n²)"),
        ("on3", "O(n³)"),
        ("oexp", "O(2ⁿ)"),
        ("unknown", "Unknown"),
        ("  ON  ", "O(n)"),
        ("  O1\n", "O(1)"),
    ],
)
def test_format_complexity_label_case_and_whitespace_insensitivity(
    internal_label: str, expected_display: str
) -> None:
    """Formatting is robust to casing variations and surrounding whitespace."""
    assert format_complexity_label(internal_label) == expected_display


def test_format_complexity_label_taxonomy_label_enums() -> None:
    """TaxonomyLabel enum instances resolve to standard Big-O notation."""
    assert format_complexity_label(TaxonomyLabel.O1) == "O(1)"
    assert format_complexity_label(TaxonomyLabel.OLOGN) == "O(log n)"
    assert format_complexity_label(TaxonomyLabel.ON) == "O(n)"
    assert format_complexity_label(TaxonomyLabel.ONLOGN) == "O(n log n)"
    assert format_complexity_label(TaxonomyLabel.ON2) == "O(n²)"
    assert format_complexity_label(TaxonomyLabel.ON3) == "O(n³)"
    assert format_complexity_label(TaxonomyLabel.OEXP) == "O(2ⁿ)"
    assert format_complexity_label(TaxonomyLabel.UNKNOWN) == "Unknown"


@pytest.mark.parametrize("empty_input", [None, "", "   ", "\t\n"])
def test_format_complexity_label_empty_and_none_yield_unknown(empty_input: object) -> None:
    """Missing or empty inputs map safely to 'Unknown'."""
    assert format_complexity_label(empty_input) == "Unknown"


@pytest.mark.parametrize(
    ("already_formatted", "expected"),
    [
        ("O(1)", "O(1)"),
        ("O(log n)", "O(log n)"),
        ("O(n)", "O(n)"),
        ("O(n log n)", "O(n log n)"),
        ("O(n²)", "O(n²)"),
        ("O(n³)", "O(n³)"),
        ("O(2ⁿ)", "O(2ⁿ)"),
        ("Unknown", "Unknown"),
    ],
)
def test_format_complexity_label_already_formatted_preserved(
    already_formatted: str, expected: str
) -> None:
    """Already formatted Big-O strings pass through unchanged."""
    assert format_complexity_label(already_formatted) == expected


@pytest.mark.parametrize(
    "unexpected_input",
    [
        "N/A",
        "CUSTOM_LABEL",
        "O(n^4)",
        12345,
        99.9,
    ],
)
def test_format_complexity_label_unexpected_inputs_fallback_safely(
    unexpected_input: object,
) -> None:
    """Unexpected or unknown inputs fall back cleanly to str(input)."""
    assert format_complexity_label(unexpected_input) == str(unexpected_input)


def test_format_explanation_text_rationales() -> None:
    """Replaces internal complexity labels within narrative explanations."""
    raw_rationale = (
        "Static analysis established a verified STRUCTURAL bound of O1. "
        "ML model predicted O1 (agrees); verified static result is preserved."
    )
    formatted = format_explanation_text(raw_rationale)
    assert (
        formatted == "Static analysis established a verified STRUCTURAL bound of O(1). "
        "ML model predicted O(1) (agrees); verified static result is preserved."
    )

    raw_summary = "Overall Complexity: ON (Source: STATIC)"
    assert format_explanation_text(raw_summary) == "Overall Complexity: O(n) (Source: STATIC)"

    raw_multi = "Compared ON2 against ONLOGN and OEXP, inconclusive (UNKNOWN)."
    assert (
        format_explanation_text(raw_multi)
        == "Compared O(n²) against O(n log n) and O(2ⁿ), inconclusive (Unknown)."
    )


def test_format_explanation_text_empty_and_none() -> None:
    """None and empty text format safely to empty string."""
    assert format_explanation_text(None) == ""
    assert format_explanation_text("") == ""


def test_format_construct_display_loop() -> None:
    """Format a loop construct into card presentation fields."""
    span = SourceSpanInfo(
        file="test.cpp",
        start_line=4,
        start_col=5,
        end_line=7,
        end_col=6,
        description="For loop",
    )
    construct = DetectedConstruct(
        kind="loop",
        description="For loop with symbolic ON bound on variable 'i'",
        span=span,
        details={
            "bound_kind": "SYMBOLIC",
            "depth": 2,
            "is_nested": True,
            "loop_id": "loop_1",
            "loop_kind": "FOR",
            "variable": "i",
        },
    )

    card = format_construct_display(construct)
    assert card["kind"] == "loop"
    assert card["category"] == "Loop"
    assert card["title"] == "For Loop"
    assert card["line_range"] == "Lines 4–7"
    assert "O(n)" in card["description"]
    assert "ON" not in card["description"]

    detail_dict = dict(card["primary_details"])
    assert detail_dict["Bound Kind"] == "Symbolic"
    assert detail_dict["Variable"] == "`i`"
    assert "Depth 2 (Nested)" in detail_dict["Nesting"]

    assert "loop_id" in card["technical_details"]
    assert card["technical_details"]["source_span"]["start_line"] == 4


def test_format_construct_display_recursion() -> None:
    """Format a recursive function construct into card presentation fields."""
    span = SourceSpanInfo(
        file="rec.cpp",
        start_line=10,
        start_col=1,
        end_line=15,
        end_col=2,
    )
    construct = DetectedConstruct(
        kind="recursion",
        description="Recursive branching with OEXP complexity",
        span=span,
        details={
            "call_site_count": 2,
            "shrink_kinds": ["n - 1", "n - 2"],
        },
    )

    card = format_construct_display(construct)
    assert card["kind"] == "recursion"
    assert card["title"] == "Recursive Function Call"
    assert card["line_range"] == "Lines 10–15"
    assert "O(2ⁿ)" in card["description"]

    detail_dict = dict(card["primary_details"])
    assert detail_dict["Recursive Sites"] == "2"
    assert "n - 1, n - 2" in detail_dict["Argument Reduction"]


def test_format_construct_display_stl_operation() -> None:
    """Format an STL operation construct into card presentation fields."""
    span = SourceSpanInfo(
        file="stl.cpp",
        start_line=20,
        start_col=5,
        end_line=20,
        end_col=25,
    )
    construct = DetectedConstruct(
        kind="stl_operation",
        description="STL operation 'sort' on container 'vec'",
        span=span,
        details={
            "container": "vec",
            "container_kind_source": "std::vector<int>",
            "operation": "sort",
        },
    )

    card = format_construct_display(construct)
    assert card["kind"] == "stl_operation"
    assert card["title"] == "STL Operation: sort"
    assert card["line_range"] == "Line 20"

    detail_dict = dict(card["primary_details"])
    assert detail_dict["Operation"] == "`sort`"
    assert detail_dict["Container"] == "`vec`"
    assert detail_dict["Type Source"] == "std::vector<int>"


def test_format_construct_display_unsupported_and_fallback() -> None:
    """Format unsupported construct and generic dictionary into card fields."""
    u_construct = DetectedConstruct(
        kind="unsupported",
        description="Encountered inline assembly",
        span=None,
        details={"kind": "asm", "raw_spelling": "__asm__"},
    )
    card_u = format_construct_display(u_construct)
    assert card_u["kind"] == "unsupported"
    assert card_u["title"] == "Unsupported Language Construct"
    assert card_u["line_range"] == "Line range not available"
    detail_dict_u = dict(card_u["primary_details"])
    assert detail_dict_u["Construct Kind"] == "asm"
    assert detail_dict_u["Raw Token"] == "`__asm__`"

    # Test raw dict input
    dict_construct = {
        "kind": "custom_gate",
        "description": "Custom barrier",
        "details": {"threshold": 42},
    }
    card_d = format_construct_display(dict_construct)
    assert card_d["kind"] == "custom_gate"
    assert card_d["title"] == "Custom Gate Construct"
    detail_dict_d = dict(card_d["primary_details"])
    assert detail_dict_d["Threshold"] == "42"
