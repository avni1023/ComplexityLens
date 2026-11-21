"""presentation.py — User-facing display and formatting helpers for ComplexityLens.

Translates internal machine-readable complexity labels into standard mathematical
Big-O notation across all presentation surfaces without mutating backend models
or serialized analysis results.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical display mapping from internal taxonomy labels to standard Big-O notation
COMPLEXITY_DISPLAY_MAP: dict[str, str] = {
    "O1": "O(1)",
    "OLOGN": "O(log n)",
    "ON": "O(n)",
    "ONLOGN": "O(n log n)",
    "ON2": "O(n²)",
    "ON3": "O(n³)",
    "OEXP": "O(2ⁿ)",
    "UNKNOWN": "Unknown",
}

# Regex to safely match internal complexity labels at word boundaries in text
_INTERNAL_LABEL_PATTERN = re.compile(r"\b(ONLOGN|OLOGN|OEXP|ON2|ON3|O1|ON|UNKNOWN)\b")


def format_complexity_label(label: Any) -> str:
    """Format an internal complexity label into standard Big-O notation.

    Parameters
    ----------
    label:
        Internal label string, TaxonomyLabel enum, or None.

    Returns
    -------
    str
        Standard Big-O notation (e.g. 'O(1)', 'O(n)', 'Unknown'), or str(label)
        for unrecognized non-empty inputs.
    """
    if label is None:
        return COMPLEXITY_DISPLAY_MAP["UNKNOWN"]

    # Extract string from enum if applicable
    if hasattr(label, "value"):
        label = label.value

    label_str = str(label).strip()
    if not label_str:
        return COMPLEXITY_DISPLAY_MAP["UNKNOWN"]

    # Direct match against canonical keys
    if label_str in COMPLEXITY_DISPLAY_MAP:
        return COMPLEXITY_DISPLAY_MAP[label_str]

    # Case-insensitive match
    upper_label = label_str.upper()
    if upper_label in COMPLEXITY_DISPLAY_MAP:
        return COMPLEXITY_DISPLAY_MAP[upper_label]

    # If already formatted as Big-O notation, return as-is
    if label_str in COMPLEXITY_DISPLAY_MAP.values():
        return label_str

    # Safe fallback for unexpected input values
    return label_str


def format_explanation_text(text: str | None) -> str:
    """Replace internal complexity labels in narrative text with Big-O notation.

    Parameters
    ----------
    text:
        Raw explanation, rationale, or summary string containing internal labels.

    Returns
    -------
    str
        Text with internal labels replaced by formatted Big-O notation.
    """
    if not text:
        return ""

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        return COMPLEXITY_DISPLAY_MAP.get(token, token)

    return _INTERNAL_LABEL_PATTERN.sub(_replace, text)


def format_construct_display(construct: Any) -> dict[str, Any]:
    """Format a detected code construct into human-readable presentation fields.

    Parameters
    ----------
    construct:
        A DetectedConstruct instance, dictionary, or duck-typed construct object.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "kind": Normalized category identifier (e.g. "loop", "recursion", "stl_operation")
        - "category": Display category name (e.g. "Loop", "Recursion", "STL Operation")
        - "title": Concise heading for the card (e.g. "For Loop", "Recursive Branching")
        - "description": Narrative description with Big-O notation applied
        - "line_range": Human-readable line range (e.g. "Lines 4–8" or "Line 5")
        - "primary_details": List of tuples (label, formatted_value) for key badges
        - "technical_details": Dict of metadata safe for JSON inspection in an expander
    """
    span = getattr(construct, "span", None)
    if span is None and isinstance(construct, dict):
        span = construct.get("span")

    line_range = "Line range not available"
    span_dict: dict[str, Any] = {}
    if span is not None:
        if isinstance(span, dict):
            s_line = span.get("start_line")
            e_line = span.get("end_line")
            s_col = span.get("start_col")
            e_col = span.get("end_col")
            span_file = span.get("file")
        else:
            s_line = getattr(span, "start_line", None)
            e_line = getattr(span, "end_line", None)
            s_col = getattr(span, "start_col", None)
            e_col = getattr(span, "end_col", None)
            span_file = getattr(span, "file", None)

        span_dict = {
            "file": span_file,
            "start_col": s_col,
            "start_line": s_line,
            "end_col": e_col,
            "end_line": e_line,
        }

        if s_line is not None and e_line is not None and s_line > 0 and e_line > 0:
            if s_line == e_line:
                line_range = f"Line {s_line}"
            else:
                line_range = f"Lines {s_line}–{e_line}"
        elif s_line is not None and s_line > 0:
            line_range = f"Line {s_line}"

    raw_desc = getattr(construct, "description", None)
    if raw_desc is None and isinstance(construct, dict):
        raw_desc = construct.get("description")
    description = format_explanation_text(
        str(raw_desc or "Construct detected during AST analysis.")
    )

    details_obj = getattr(construct, "details", None)
    if details_obj is None and isinstance(construct, dict):
        details_obj = construct.get("details")
    details = dict(details_obj) if isinstance(details_obj, dict) else {}

    raw_kind = getattr(construct, "kind", None)
    if raw_kind is None and isinstance(construct, dict):
        raw_kind = construct.get("kind")
    kind = str(raw_kind or "unknown").strip().lower()

    primary_details: list[tuple[str, str]] = []

    if kind == "loop":
        category = "Loop"
        loop_kind = str(details.get("loop_kind", "Loop")).upper()
        if loop_kind in ("FOR", "WHILE", "DO_WHILE", "DO"):
            title = f"{loop_kind.replace('_', '-').title()} Loop"
        else:
            title = "Loop Construct"

        bound_kind = str(details.get("bound_kind", "UNKNOWN")).replace("_", " ").title()
        primary_details.append(("Bound Kind", bound_kind))

        var = details.get("variable")
        if var:
            primary_details.append(("Variable", f"`{var}`"))

        depth = details.get("depth", 1)
        is_nested = details.get("is_nested", False) or (isinstance(depth, int) and depth > 1)
        nesting_label = f"Depth {depth} ({'Nested' if is_nested else 'Outer'})"
        primary_details.append(("Nesting", nesting_label))

    elif kind == "recursion":
        category = "Recursion"
        title = "Recursive Function Call"
        call_count = details.get("call_site_count", 1)
        primary_details.append(("Recursive Sites", str(call_count)))

        shrink_kinds = details.get("shrink_kinds")
        if shrink_kinds and isinstance(shrink_kinds, (list, tuple)):
            shrink_str = ", ".join(str(s) for s in shrink_kinds if s)
            if shrink_str:
                primary_details.append(("Argument Reduction", shrink_str))

    elif kind == "stl_operation":
        category = "STL Operation"
        op_name = details.get("operation", "call")
        title = f"STL Operation: {op_name}"
        primary_details.append(("Operation", f"`{op_name}`"))

        container = details.get("container")
        if container:
            primary_details.append(("Container", f"`{container}`"))

        source = details.get("container_kind_source")
        if source:
            primary_details.append(("Type Source", str(source)))

    elif kind == "unsupported":
        category = "Unsupported"
        title = "Unsupported Language Construct"
        u_kind = details.get("kind")
        if u_kind:
            primary_details.append(("Construct Kind", str(u_kind)))
        raw_tok = details.get("raw_spelling")
        if raw_tok:
            primary_details.append(("Raw Token", f"`{raw_tok}`"))

    else:
        category = kind.replace("_", " ").title()
        title = f"{category} Construct"
        for k, v in details.items():
            if not isinstance(v, (dict, list, tuple)):
                primary_details.append((k.replace("_", " ").title(), str(v)))

    def _sanitize(val: Any) -> Any:
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        if isinstance(val, dict):
            return {str(k): _sanitize(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_sanitize(v) for v in val]
        return str(val)

    sanitized_tech: dict[str, Any] = {
        "kind": kind,
        **{str(k): _sanitize(v) for k, v in details.items()},
    }
    if span_dict:
        sanitized_tech["source_span"] = {str(k): _sanitize(v) for k, v in span_dict.items()}

    return {
        "category": category,
        "description": description,
        "kind": kind,
        "line_range": line_range,
        "primary_details": primary_details,
        "technical_details": sanitized_tech,
        "title": title,
    }
