"""
validation.py — Centralized input validation for ComplexityLens.

All validation runs *before* the parser is invoked.  The parser is never called
with input that fails validation.

Validation rules
----------------
* Input type: must be ``str``.  Bytes and all other types are rejected.
* Non-empty: empty or whitespace-only source is rejected.
* No NUL characters: ``\\0`` is always rejected.
* UTF-8 note: the validator accepts ``str`` (already decoded).  Any upstream
  byte decoding MUST use ``bytes.decode("utf-8", errors="strict")`` so that
  invalid UTF-8 sequences raise ``UnicodeDecodeError`` before reaching here.
* Maximum source size: 20 KiB (20,480 UTF-8 bytes).
* Maximum source lines: 200 (inclusive).
* Maximum line length: 1,000 characters (excluding line terminator).
* Maximum brace nesting depth: 50.  Braces inside C/C++ line comments
  (``// …``), block comments (``/* … */``), string literals, and character
  literals are ignored during depth counting.
* No silent truncation: the validator never modifies input.

Include / directive safety
--------------------------
Only ``#include <header>`` directives whose header name appears in
:data:`ALLOWED_HEADERS` are permitted.  All other include forms and
prohibited preprocessor directives cause a :class:`ValidationError`.

Specifically rejected:
* ``#include "…"`` (quoted includes)
* ``#include <path/with/slash>`` (path-style system includes)
* ``#include_next``
* ``#embed``
* ``#pragma`` (any pragma)
* ``__has_include`` appearing anywhere in the source text

Disallowed directive behaviour: the line is rejected and a
:class:`ValidationError` is raised.  Source-line mapping is preserved for
compliant input.

The whitelist is kept compatible with the existing Day-1 / Day-2 fixtures.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum source size in UTF-8 bytes.
MAX_SOURCE_BYTES: int = 20_480  # 20 KiB

#: Maximum number of source lines (inclusive).
MAX_SOURCE_LINES: int = 200

#: Maximum line length in characters (excluding line terminator).
MAX_LINE_LENGTH: int = 1_000

#: Maximum brace nesting depth.
MAX_BRACE_DEPTH: int = 50

# ---------------------------------------------------------------------------
# Allowed standard C++ headers
# ---------------------------------------------------------------------------

#: Whitelist of permitted ``#include <header>`` names.
#:
#: Compatible with:
#: - Day-1 fixtures (vector, algorithm, unordered_map)
#: - Day-2 IR tests (vector)
#: - Day-3 validation tests
ALLOWED_HEADERS: frozenset[str] = frozenset(
    {
        # Containers
        "array",
        "deque",
        "forward_list",
        "list",
        "map",
        "queue",
        "set",
        "stack",
        "unordered_map",
        "unordered_set",
        "vector",
        # Algorithms & numeric
        "algorithm",
        "numeric",
        "functional",
        "iterator",
        # Strings
        "string",
        "string_view",
        # I/O (read-only context; not for execution)
        "iostream",
        "istream",
        "ostream",
        "sstream",
        "fstream",
        # Utilities
        "utility",
        "tuple",
        "optional",
        "variant",
        "any",
        "memory",
        "chrono",
        "type_traits",
        "limits",
        "climits",
        "cstddef",
        "cstdint",
        "cassert",
        "cmath",
        "cstring",
        "cstdlib",
        "cstdio",
        "cerrno",
        # C++17 extras
        "execution",
    }
)

# ---------------------------------------------------------------------------
# Prohibited preprocessor pattern
# ---------------------------------------------------------------------------

# Matches any #include directive on a line.
_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*(.+)", re.MULTILINE)

# Matches a system include: #include <header>
_SYSTEM_INCLUDE_RE = re.compile(r"^<([^>]*)>$")

# Prohibited directive keywords that must not appear as top-level directives.
_PROHIBITED_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*("
    r"include_next"
    r"|embed"
    r"|pragma"
    r")\b",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """Raised when submitted C++ source fails any validation rule.

    Parameters
    ----------
    message:
        Human-readable explanation of the rule violation.
    line:
        1-based source line where the problem was detected, or ``None``.
    """

    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        detail = f" (line {line})" if line is not None else ""
        super().__init__(f"{message}{detail}")


# ---------------------------------------------------------------------------
# Brace-depth scanner
# ---------------------------------------------------------------------------


def _max_brace_depth(source: str) -> int:
    """Return the maximum ``{`` nesting depth in *source*.

    Braces inside line comments (``//``), block comments (``/* … */``),
    string literals (``"…"``), and character literals (``'…'``) are
    ignored.  Escaped characters inside literals are handled correctly.
    """
    depth = 0
    max_depth = 0
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        # ── line comment ──────────────────────────────────────────────────
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            continue

        # ── block comment ─────────────────────────────────────────────────
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i + 1 < n:
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            else:
                i = n  # unterminated block comment — skip to end
            continue

        # ── string literal ────────────────────────────────────────────────
        if ch == '"':
            i += 1
            while i < n:
                c2 = source[i]
                if c2 == "\\":
                    i += 2  # skip escaped char
                elif c2 == '"':
                    i += 1
                    break
                else:
                    i += 1
            continue

        # ── character literal ─────────────────────────────────────────────
        if ch == "'":
            i += 1
            while i < n:
                c2 = source[i]
                if c2 == "\\":
                    i += 2
                elif c2 == "'":
                    i += 1
                    break
                else:
                    i += 1
            continue

        # ── braces ────────────────────────────────────────────────────────
        if ch == "{":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch == "}":
            if depth > 0:
                depth -= 1

        i += 1

    return max_depth


# ---------------------------------------------------------------------------
# Directive validator
# ---------------------------------------------------------------------------


def _check_directives(source: str) -> None:
    """Validate all preprocessor directives in *source*.

    Raises :class:`ValidationError` on the first violation.
    """
    # Reject ``__has_include`` anywhere in source text.
    if "__has_include" in source:
        line_no = source[: source.index("__has_include")].count("\n") + 1
        raise ValidationError(
            "__has_include is not permitted in submitted source", line=line_no
        )

    lines = source.splitlines(keepends=True)

    for lineno, line in enumerate(lines, start=1):
        stripped = line.rstrip("\r\n")

        # ── #include_next / #embed / #pragma ─────────────────────────────
        m = _PROHIBITED_DIRECTIVE_RE.match(stripped)
        if m:
            keyword = m.group(1)
            raise ValidationError(
                f"#{keyword} is not permitted in submitted source", line=lineno
            )

        # ── #include ──────────────────────────────────────────────────────
        m2 = re.match(r"^\s*#\s*include\s+(.+)", stripped)
        if m2:
            rest = m2.group(1).strip()

            # Quoted include: #include "…"
            if rest.startswith('"'):
                raise ValidationError(
                    "Quoted #include directives are not permitted; "
                    "use only allowed system headers (<header>)",
                    line=lineno,
                )

            # System include with optional trailing whitespace and comments
            m3 = re.match(r"^<([^>]+)>(?:\s*(?://.*|/\*.*?\*/\s*)?)?$", rest)
            if not m3:
                raise ValidationError(
                    f"Unrecognized #include format: {rest!r}", line=lineno
                )

            header_name = m3.group(1).strip()

            # Path-style: contains '/' or '\\'
            if "/" in header_name or "\\" in header_name:
                raise ValidationError(
                    f"Path-style #include <{header_name}> is not permitted; "
                    "use only bare header names",
                    line=lineno,
                )

            # Whitelist check
            if header_name not in ALLOWED_HEADERS:
                raise ValidationError(
                    f"#include <{header_name}> is not on the permitted "
                    "header whitelist",
                    line=lineno,
                )


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------


def validate_source(source: object) -> str:
    """Validate *source* and return it unchanged if all checks pass.

    Parameters
    ----------
    source:
        The C++ source to validate.  Must be a ``str`` (already decoded from
        UTF-8).  ``bytes`` and all other types are rejected.

    Returns
    -------
    str
        The original *source* string, unchanged.

    Raises
    ------
    ValidationError
        On any rule violation.  The error message names the violated rule.
        The validator never silently truncates input.

    Notes
    -----
    Upstream byte decoding must use ``data.decode("utf-8", errors="strict")``
    to ensure invalid UTF-8 sequences raise ``UnicodeDecodeError`` before
    this function is called.
    """
    # ── Type check ────────────────────────────────────────────────────────
    if not isinstance(source, str):
        raise ValidationError(
            f"Source must be str, got {type(source).__name__!r}"
        )

    # ── NUL characters ────────────────────────────────────────────────────
    if "\x00" in source:
        line_no = source[: source.index("\x00")].count("\n") + 1
        raise ValidationError("NUL (\\x00) characters are not permitted", line=line_no)

    # ── Non-empty ─────────────────────────────────────────────────────────
    if not source.strip():
        raise ValidationError("Source must not be empty or whitespace-only")

    # ── Size limit ────────────────────────────────────────────────────────
    utf8_size = len(source.encode("utf-8"))
    if utf8_size > MAX_SOURCE_BYTES:
        raise ValidationError(
            f"Source exceeds maximum size: {utf8_size} UTF-8 bytes "
            f"(limit {MAX_SOURCE_BYTES})"
        )

    # ── Line count ────────────────────────────────────────────────────────
    lines = source.splitlines()
    if len(lines) > MAX_SOURCE_LINES:
        raise ValidationError(
            f"Source exceeds maximum line count: {len(lines)} lines "
            f"(limit {MAX_SOURCE_LINES})"
        )

    # ── Line length ───────────────────────────────────────────────────────
    for lineno, line in enumerate(lines, start=1):
        if len(line) > MAX_LINE_LENGTH:
            raise ValidationError(
                f"Line {lineno} exceeds maximum length: {len(line)} characters "
                f"(limit {MAX_LINE_LENGTH})",
                line=lineno,
            )

    # ── Brace nesting depth ───────────────────────────────────────────────
    depth = _max_brace_depth(source)
    if depth > MAX_BRACE_DEPTH:
        raise ValidationError(
            f"Source exceeds maximum brace nesting depth: {depth} "
            f"(limit {MAX_BRACE_DEPTH})"
        )

    # ── Directive safety ──────────────────────────────────────────────────
    _check_directives(source)

    return source
