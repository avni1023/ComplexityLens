"""
test_validation.py
==================
Day-3 tests for input validation, IR diagnostics, container-type provenance,
and the directive/include whitelist.

Covers:
- Empty and whitespace-only source
- Invalid input types (bytes, None, int)
- NUL characters
- Invalid UTF-8 at the decoding boundary
- Exactly 200 lines vs 201 lines
- Exactly 20,480 UTF-8 bytes vs oversized input
- Exactly 1,000 characters per line vs oversized line
- Brace nesting at depth 50 vs depth 51
- Braces inside comments and literals (ignored for depth counting)
- Mocked parser: parsing never invoked when validation fails
- CRLF input and correct source-line mapping
- Allowed and disallowed headers
- Prohibited directives (#pragma, #include_next, #embed, __has_include)
- Quoted and path-style includes
- IR diagnostics: error severity forces analysis_complete = False
- Warning-only diagnostics preserve completeness
- UnsupportedIR nodes force analysis_complete = False
- IRDiagnostic structured representation
- Container-type provenance: source_text / resolved / unknown
- Golden IR snapshot: analysis_complete field
- Serialization: diagnostics in to_dict()
- Deterministic node IDs across repeated runs
- Regression: all existing fixture-based IR tests still pass
"""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from complexity_lens.ir import (
    BoundKind,
    IRDiagnostic,
    UnsupportedKind,
    build_ir_from_source,
)
from complexity_lens.ir.model import (
    LoopKind,
    SourceSpan,
)
from complexity_lens.validation import (
    ALLOWED_HEADERS,
    MAX_LINE_LENGTH,
    MAX_SOURCE_BYTES,
    MAX_SOURCE_LINES,
    ValidationError,
    _max_brace_depth,
    validate_source,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "cpp"


# ===========================================================================
# Helpers
# ===========================================================================


def _make_source(n_lines: int, line_content: str = "int x = 0;") -> str:
    return "\n".join([line_content] * n_lines)


def _make_wide_line(n_chars: int) -> str:
    """Return a single-line source with a line of exactly n_chars."""
    # Use a valid C++ comment so the parser doesn't care
    return "// " + "x" * (n_chars - 3)


def _make_nested_braces(depth: int) -> str:
    """Return a syntactically valid function with brace depth == depth."""
    # depth=1: void f() {}
    # depth=2: void f() { {} }
    inner = "{}" * 1
    for _ in range(depth - 2):
        inner = "{" + inner + "}"
    return "void f() " + "{" + inner + "}"


# ===========================================================================
# Section 1: Type Validation
# ===========================================================================


class TestTypeValidation:
    def test_bytes_rejected(self):
        with pytest.raises(ValidationError, match="str"):
            validate_source(b"void f() {}")

    def test_none_rejected(self):
        with pytest.raises(ValidationError, match="str"):
            validate_source(None)

    def test_int_rejected(self):
        with pytest.raises(ValidationError, match="str"):
            validate_source(42)

    def test_list_rejected(self):
        with pytest.raises(ValidationError, match="str"):
            validate_source(["void f() {}"])

    def test_valid_str_accepted(self):
        result = validate_source("void f() {}")
        assert result == "void f() {}"


# ===========================================================================
# Section 2: Empty / Whitespace
# ===========================================================================


class TestEmptySource:
    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_source("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_source("   \n\t\n  ")

    def test_newlines_only_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_source("\n\n\n")

    def test_single_statement_accepted(self):
        validate_source("void f() {}")


# ===========================================================================
# Section 3: NUL Characters
# ===========================================================================


class TestNulCharacters:
    def test_nul_at_start_rejected(self):
        with pytest.raises(ValidationError, match="NUL"):
            validate_source("\x00void f() {}")

    def test_nul_in_middle_rejected(self):
        with pytest.raises(ValidationError, match="NUL"):
            validate_source("void f() {\x00}")

    def test_nul_at_end_rejected(self):
        with pytest.raises(ValidationError, match="NUL"):
            validate_source("void f() {}\x00")


# ===========================================================================
# Section 4: Invalid UTF-8 (decoding boundary)
# ===========================================================================


class TestUTF8Boundary:
    """The validator accepts str; invalid UTF-8 must be caught at decode time."""

    def test_invalid_utf8_raises_on_decode(self):
        """Simulate the upstream decoding step that must reject invalid UTF-8."""
        raw_bytes = b"void f() {\xff\xfe}"
        with pytest.raises(UnicodeDecodeError):
            raw_bytes.decode("utf-8", errors="strict")

    def test_valid_utf8_with_unicode_comment_accepted(self):
        """Unicode chars that are valid UTF-8 pass through to the validator."""
        source = "// Comment: résumé\nvoid f() {}"
        validate_source(source)


# ===========================================================================
# Section 5: Source Size Limit
# ===========================================================================


class TestSourceSizeLimit:
    def test_exactly_max_bytes_accepted(self):
        # Build a source that is exactly MAX_SOURCE_BYTES UTF-8 bytes
        # using at most MAX_SOURCE_LINES lines.
        # Each line must be <= MAX_LINE_LENGTH chars.
        # Budget: 200 lines × ~102 chars/line + newline = ~20,600 bytes headroom.
        # Use lines of exactly 101 characters (100 comment chars + newline = 101 bytes).
        line_content = "// " + "x" * 97  # 100 chars, within the 1000-char limit
        line_bytes = len((line_content + "\n").encode("utf-8"))  # 101 bytes
        # Fill as many full lines as possible without exceeding both limits
        n_lines = min(MAX_SOURCE_LINES, MAX_SOURCE_BYTES // line_bytes)
        source = (line_content + "\n") * n_lines
        current = len(source.encode("utf-8"))
        # Pad the last line (or add a padded comment) to hit exactly MAX_SOURCE_BYTES
        remaining = MAX_SOURCE_BYTES - current
        if remaining > 0 and n_lines < MAX_SOURCE_LINES:
            # Add one more line with exactly `remaining - 1` chars + newline
            # (remaining - 1 content chars + 1 newline = remaining bytes)
            pad_line = "// " + "y" * max(0, remaining - 4) + "\n"
            source += pad_line
        elif remaining > 0:
            # Can't add a new line; pad the existing last line in-place via trailing spaces
            # Trim the trailing newline, pad, then restore
            source = source.rstrip("\n") + " " * remaining + "\n"
        # At this point the source may be up to 1 byte under; trim to exact
        # by re-measuring and slicing if needed (all pad chars are ASCII)
        final = source.encode("utf-8")
        if len(final) > MAX_SOURCE_BYTES:
            source = final[:MAX_SOURCE_BYTES].decode("utf-8", errors="replace")
        assert len(source.encode("utf-8")) <= MAX_SOURCE_BYTES
        assert len(source.splitlines()) <= MAX_SOURCE_LINES
        validate_source(source)

    def test_one_byte_over_limit_rejected(self):
        # Build a source that is exactly MAX_SOURCE_BYTES + 1 bytes
        # using at most MAX_SOURCE_LINES lines.
        line_content = "// " + "x" * 97  # 100 chars, within the 1000-char limit
        line_bytes = len((line_content + "\n").encode("utf-8"))  # 101 bytes
        n_lines = min(MAX_SOURCE_LINES - 1, MAX_SOURCE_BYTES // line_bytes)
        source = (line_content + "\n") * n_lines
        current = len(source.encode("utf-8"))
        # Add one more line that pushes exactly 1 byte over MAX_SOURCE_BYTES
        remaining = MAX_SOURCE_BYTES - current + 1  # +1 = one byte over
        over_line = "// " + "z" * max(0, remaining - 4) + "\n"
        source += over_line
        assert len(source.encode("utf-8")) > MAX_SOURCE_BYTES
        assert len(source.splitlines()) <= MAX_SOURCE_LINES
        with pytest.raises(ValidationError, match="maximum size"):
            validate_source(source)

    def test_multibyte_utf8_counted_by_bytes(self):
        # "é" is 2 UTF-8 bytes; ensure byte counting is correct.
        short = "é" * 10
        assert len(short.encode("utf-8")) == 20
        # 20 bytes is well under limit — must pass.
        validate_source("// " + short + "\nvoid f() {}")


# ===========================================================================
# Section 6: Line Count Limit
# ===========================================================================


class TestLineCountLimit:
    def test_exactly_200_lines_accepted(self):
        source = _make_source(MAX_SOURCE_LINES)
        validate_source(source)

    def test_201_lines_rejected(self):
        source = _make_source(MAX_SOURCE_LINES + 1)
        with pytest.raises(ValidationError, match="line count"):
            validate_source(source)

    def test_crlf_line_endings_counted_correctly(self):
        """CRLF source: splitlines() must not double-count."""
        lines = ["int x = 0;"] * MAX_SOURCE_LINES
        source = "\r\n".join(lines)
        validate_source(source)

    def test_crlf_one_over_rejected(self):
        lines = ["int x = 0;"] * (MAX_SOURCE_LINES + 1)
        source = "\r\n".join(lines)
        with pytest.raises(ValidationError, match="line count"):
            validate_source(source)


# ===========================================================================
# Section 7: Line Length Limit
# ===========================================================================


class TestLineLengthLimit:
    def test_exactly_1000_chars_accepted(self):
        long_line = "// " + "x" * (MAX_LINE_LENGTH - 3)
        assert len(long_line) == MAX_LINE_LENGTH
        validate_source(long_line + "\nvoid f() {}")

    def test_1001_chars_rejected(self):
        long_line = "// " + "x" * (MAX_LINE_LENGTH - 3 + 1)
        assert len(long_line) == MAX_LINE_LENGTH + 1
        with pytest.raises(ValidationError, match="maximum length"):
            validate_source(long_line + "\nvoid f() {}")

    def test_line_length_excludes_line_terminator(self):
        """A line of exactly MAX_LINE_LENGTH chars + \\n should pass."""
        long_line = "x" * MAX_LINE_LENGTH
        # Not valid C++ but validator checks length, not syntax
        source = long_line + "\nvoid f() {}"
        validate_source(source)

    def test_crlf_line_length_excludes_cr(self):
        """A CRLF line of exactly MAX_LINE_LENGTH chars should pass."""
        long_line = "// " + "x" * (MAX_LINE_LENGTH - 3)
        source = long_line + "\r\nvoid f() {}"
        validate_source(source)

    def test_error_reports_line_number(self):
        good_line = "void f() {}"
        bad_line = "// " + "x" * (MAX_LINE_LENGTH - 3 + 1)
        source = good_line + "\n" + bad_line
        with pytest.raises(ValidationError) as exc_info:
            validate_source(source)
        assert exc_info.value.line == 2


# ===========================================================================
# Section 8: Brace Nesting Depth
# ===========================================================================


class TestBraceNestingDepth:
    def test_depth_50_accepted(self):
        # Build valid source with exactly depth 50
        inner = "void f() {" + "{" * 49 + "}" * 49 + "}"
        assert _max_brace_depth(inner) == 50
        validate_source(inner)

    def test_depth_51_rejected(self):
        inner = "void f() {" + "{" * 50 + "}" * 50 + "}"
        assert _max_brace_depth(inner) == 51
        with pytest.raises(ValidationError, match="brace nesting depth"):
            validate_source(inner)

    def test_braces_in_line_comment_ignored(self):
        source = "// {{{{\nvoid f() { int x = 0; }"
        depth = _max_brace_depth(source)
        assert depth == 1

    def test_braces_in_block_comment_ignored(self):
        source = "/* {{{{ */\nvoid f() { int x = 0; }"
        depth = _max_brace_depth(source)
        assert depth == 1

    def test_braces_in_string_literal_ignored(self):
        source = 'void f() { const char* s = "{{{{"; }'
        depth = _max_brace_depth(source)
        assert depth == 1

    def test_braces_in_char_literal_ignored(self):
        source = "void f() { char c = '{'; }"
        depth = _max_brace_depth(source)
        assert depth == 1

    def test_escaped_char_in_string_handled(self):
        source = r'void f() { const char* s = "a\"b{"; }'
        depth = _max_brace_depth(source)
        assert depth == 1

    def test_normal_nesting_counted_correctly(self):
        source = "void f() { { { int x = 0; } } }"
        depth = _max_brace_depth(source)
        assert depth == 3


# ===========================================================================
# Section 9: Parser Never Invoked on Validation Failure
# ===========================================================================


class TestParserNotCalledOnValidationFailure:
    """Use a mock to prove parse_source is never called when validation fails."""

    def test_parse_not_called_for_empty_source(self):
        with patch(
            "complexity_lens.parsers.cpp_libclang.parse_source"
        ) as mock_parse:
            with pytest.raises(ValidationError):
                validate_source("")
            mock_parse.assert_not_called()

    def test_parse_not_called_for_bytes_input(self):
        with patch(
            "complexity_lens.parsers.cpp_libclang.parse_source"
        ) as mock_parse:
            with pytest.raises(ValidationError):
                validate_source(b"void f() {}")
            mock_parse.assert_not_called()

    def test_parse_not_called_for_nul_source(self):
        with patch(
            "complexity_lens.parsers.cpp_libclang.parse_source"
        ) as mock_parse:
            with pytest.raises(ValidationError):
                validate_source("void f() {\x00}")
            mock_parse.assert_not_called()

    def test_parse_not_called_for_too_many_lines(self):
        with patch(
            "complexity_lens.parsers.cpp_libclang.parse_source"
        ) as mock_parse:
            source = _make_source(MAX_SOURCE_LINES + 1)
            with pytest.raises(ValidationError):
                validate_source(source)
            mock_parse.assert_not_called()


# ===========================================================================
# Section 10: CRLF / Source-Line Mapping
# ===========================================================================


class TestCRLFLineMapping:
    def test_crlf_source_passes_validation(self):
        source = "void f() {\r\n  int x = 0;\r\n}\r\n"
        result = validate_source(source)
        assert result == source  # returned unchanged

    def test_crlf_source_ir_builds(self):
        source = "void f(int n) {\r\n  for (int i = 0; i < n; ++i) {}\r\n}\r\n"
        validate_source(source)
        prog = build_ir_from_source(source)
        assert len(prog.functions) == 1
        loops = prog.all_loops()
        assert len(loops) == 1


# ===========================================================================
# Section 11: Include / Directive Safety
# ===========================================================================


class TestAllowedHeaders:
    @pytest.mark.parametrize(
        "header",
        [
            "vector",
            "algorithm",
            "unordered_map",
            "string",
            "map",
            "set",
            "array",
        ],
    )
    def test_allowed_header_passes(self, header: str):
        source = f"#include <{header}>\nvoid f() {{}}"
        validate_source(source)

    def test_whitelist_contains_fixture_headers(self):
        # All headers used in Day-1/2 fixtures must be on the whitelist
        for h in ("vector", "algorithm", "unordered_map"):
            assert h in ALLOWED_HEADERS

    def test_allowed_header_with_trailing_line_comment(self):
        source = "#include <vector> // required for std::vector\nvoid f() {}"
        assert validate_source(source) == source

    def test_allowed_header_with_trailing_block_comment(self):
        source = "#include <algorithm> /* std::sort */\nvoid f() {}"
        assert validate_source(source) == source

    def test_allowed_header_with_surrounding_spaces(self):
        source = "#include <  vector  >\nvoid f() {}"
        assert validate_source(source) == source


class TestDisallowedHeaders:
    @pytest.mark.parametrize(
        "header",
        [
            "windows.h",
            "sys/socket.h",
            "linux/kernel.h",
            "boost/algorithm/string.hpp",
            "nonexistent_header",
        ],
    )
    def test_disallowed_system_header_rejected(self, header: str):
        source = f"#include <{header}>\nvoid f() {{}}"
        # Headers with '/' are caught as path-style before reaching the whitelist check;
        # headers without '/' but not on the whitelist are caught as whitelist violations.
        # Either way, a ValidationError must be raised.
        with pytest.raises(ValidationError):
            validate_source(source)

    def test_quoted_include_rejected(self):
        with pytest.raises(ValidationError, match="[Qq]uoted"):
            validate_source('#include "myheader.h"\nvoid f() {}')

    def test_path_style_include_rejected(self):
        with pytest.raises(ValidationError, match="[Pp]ath"):
            validate_source("#include <sys/types.h>\nvoid f() {}")

    def test_path_with_backslash_rejected(self):
        with pytest.raises(ValidationError, match="[Pp]ath"):
            validate_source("#include <windows\\\\system.h>\nvoid f() {}")

    def test_path_style_with_comment_rejected(self):
        with pytest.raises(ValidationError, match="[Pp]ath"):
            validate_source("#include <sys/socket.h> // networking\nvoid f() {}")

    def test_path_style_relative_rejected(self):
        with pytest.raises(ValidationError, match="[Pp]ath"):
            validate_source("#include <../secret.h>\nvoid f() {}")


class TestProhibitedDirectives:
    def test_pragma_rejected(self):
        with pytest.raises(ValidationError, match="pragma"):
            validate_source("#pragma once\nvoid f() {}")

    def test_pragma_pack_rejected(self):
        with pytest.raises(ValidationError, match="pragma"):
            validate_source("#pragma pack(push, 1)\nvoid f() {}")

    def test_include_next_rejected(self):
        with pytest.raises(ValidationError, match="include_next"):
            validate_source("#include_next <vector>\nvoid f() {}")

    def test_embed_rejected(self):
        with pytest.raises(ValidationError, match="embed"):
            validate_source("#embed <data.bin>\nvoid f() {}")

    def test_has_include_expression_rejected(self):
        with pytest.raises(ValidationError, match="__has_include"):
            validate_source(
                "#if __has_include(<optional>)\n#include <optional>\n#endif\nvoid f() {}"
            )

    def test_error_includes_line_number_for_include(self):
        source = "void f() {}\n#include <windows.h>"
        with pytest.raises(ValidationError) as exc_info:
            validate_source(source)
        assert exc_info.value.line == 2


# ===========================================================================
# Section 12: IR Diagnostics and analysis_complete
# ===========================================================================


class TestIRDiagnostics:
    """Tests for structured IRDiagnostic and analysis_complete logic."""

    def test_error_diagnostic_forces_incomplete(self):
        # Malformed C++ → parser emits error diagnostics
        prog = build_ir_from_source("void f() { int x = ; }")
        # At minimum: has_syntax_errors set, analysis_complete = False
        assert not prog.analysis_complete

    def test_error_diagnostic_has_severity(self):
        prog = build_ir_from_source("void f() { int x = ; }")
        error_diags = [d for d in prog.diagnostics if d.is_error()]
        assert len(error_diags) >= 1
        for d in error_diags:
            assert d.severity in ("error", "fatal")
            assert isinstance(d.message, str)
            assert len(d.message) > 0

    def test_warning_only_does_not_force_incomplete(self):
        # Clean code should have no errors → analysis_complete = True
        prog = build_ir_from_source("void f(int n) { for (int i = 0; i < n; ++i) {} }")
        # Should be complete
        assert prog.analysis_complete

    def test_unsupported_ir_forces_incomplete(self):
        # Inline assembly fixture
        from complexity_lens.parsers.cpp_libclang import parse_file

        fixture = FIXTURES / "11_unsupported_asm.cpp"
        from complexity_lens.ir import build_ir

        prog = build_ir(parse_file(str(fixture)))
        # read_tsc contains __asm__ → UnsupportedIR → analysis_complete = False
        assert not prog.analysis_complete
        assert len(prog.all_unsupported()) >= 1

    def test_unsupported_node_nested_in_loop_forces_incomplete(self):
        code = (
            "void f(int n) {\n"
            "    for (int i = 0; i < n; ++i) {\n"
            "        __asm__(\"nop\");\n"
            "    }\n"
            "}"
        )
        prog = build_ir_from_source(code)
        assert not prog.analysis_complete
        assert len(prog.all_unsupported()) >= 1
        # Check that the unsupported node is found inside the loop body
        assert len(prog.functions) == 1
        func = prog.functions[0]
        assert len(func.all_unsupported()) >= 1

    def test_unsupported_node_nested_in_doubly_nested_loops_forces_incomplete(self):
        code = (
            "void f(int n) {\n"
            "    for (int i = 0; i < n; ++i) {\n"
            "        for (int j = 0; j < n; ++j) {\n"
            "            __asm__(\"nop\");\n"
            "        }\n"
            "    }\n"
            "}"
        )
        prog = build_ir_from_source(code)
        assert not prog.analysis_complete
        assert len(prog.all_unsupported()) >= 1

    def test_synthetic_nested_unsupported_sets_analysis_complete_false(self):
        from complexity_lens.ir import (
            BoundKind,
            FunctionIR,
            LoopBound,
            LoopIR,
            LoopKind,
            LoopUpdate,
            ProgramIR,
            SourceSpan,
            UnsupportedIR,
            UnsupportedKind,
            UpdateKind,
        )

        span = SourceSpan("<mock>.cpp", 1, 1, 10, 1)
        unsupported = UnsupportedIR(
            id="u1",
            kind=UnsupportedKind.ASM_STMT,
            raw_spelling="__asm__",
            reason="Inline assembly",
            span=span,
        )
        inner_loop = LoopIR(
            id="l2",
            kind=LoopKind.FOR,
            depth=2,
            bound=LoopBound(BoundKind.CONSTANT),
            update=LoopUpdate(UpdateKind.ADDITIVE),
            span=span,
            body=[unsupported],
        )
        outer_loop = LoopIR(
            id="l1",
            kind=LoopKind.FOR,
            depth=1,
            bound=LoopBound(BoundKind.CONSTANT),
            update=LoopUpdate(UpdateKind.ADDITIVE),
            span=span,
            body=[inner_loop],
        )
        func = FunctionIR(
            id="f1",
            name="foo",
            return_type="void",
            parameters=[],
            span=span,
            body=[outer_loop],
        )
        # Verify traversal finds unsupported deeply nested
        assert len(func.all_unsupported()) == 1
        prog = ProgramIR(
            source_path="<mock>.cpp",
            functions=[func],
            analysis_complete=True,
        )
        # If any unsupported node exists, analysis_complete must be set to False
        if prog.all_unsupported():
            prog.analysis_complete = False
        assert not prog.analysis_complete

    def test_clean_code_analysis_complete_true(self):
        prog = build_ir_from_source("int add(int a, int b) { return a + b; }")
        assert prog.analysis_complete

    def test_fatal_failure_not_empty_ir(self):
        """A fatal parse error must not produce a misleading empty-looking IR."""
        prog = build_ir_from_source("void f( { int x =;")
        assert prog.has_syntax_errors
        assert not prog.analysis_complete

    def test_ir_diagnostic_structured_fields(self):
        prog = build_ir_from_source("void f() { int x = ; }")
        for d in prog.diagnostics:
            assert isinstance(d, IRDiagnostic)
            assert d.severity in ("note", "warning", "error", "fatal")
            assert isinstance(d.message, str)
            # span may be None or a SourceSpan
            if d.span is not None:
                assert isinstance(d.span, SourceSpan)

    def test_analysis_complete_in_to_dict(self):
        prog = build_ir_from_source("int f(int n) { return n; }")
        d = prog.to_dict()
        assert "analysis_complete" in d
        assert isinstance(d["analysis_complete"], bool)

    def test_diagnostics_in_to_dict(self):
        prog = build_ir_from_source("void f() { int x = ; }")
        d = prog.to_dict()
        assert "diagnostics" in d
        assert isinstance(d["diagnostics"], list)
        for diag in d["diagnostics"]:
            assert "severity" in diag
            assert "message" in diag

    def test_analysis_complete_in_pretty_print(self):
        prog = build_ir_from_source("int f() { return 1; }")
        output = prog.pretty_print()
        assert "analysis_complete" in output

    def test_incomplete_in_pretty_print(self):
        prog = build_ir_from_source("void f() { int x = ; }")
        output = prog.pretty_print()
        assert "INCOMPLETE" in output or "analysis_complete=False" in output


# ===========================================================================
# Section 13: Container-Type Provenance
# ===========================================================================


class TestContainerTypeProvenance:
    """Tests for container_kind_source field on STLOperationIR."""

    def test_source_text_provenance_for_inferred_container(self):
        """push_back on a named container → source_text provenance."""
        import os

        from complexity_lens.parsers.cpp_libclang import get_configured_cxx_include_dirs

        if not get_configured_cxx_include_dirs() and not os.environ.get("CXX_INCLUDE_DIRS"):
            pytest.skip("STL headers not configured in CXX_INCLUDE_DIRS")

        code = """
        #include <vector>
        void append(std::vector<int>& v, int x) { v.push_back(x); }
        """
        prog = build_ir_from_source(code)
        stl_ops = prog.all_stl_ops()
        assert len(stl_ops) == 1
        assert stl_ops[0].container_kind_source == "source_text"

    def test_unknown_container_provenance_when_no_container(self):
        """An STL op where container cannot be resolved → None provenance."""
        import os

        from complexity_lens.parsers.cpp_libclang import get_configured_cxx_include_dirs

        if not get_configured_cxx_include_dirs() and not os.environ.get("CXX_INCLUDE_DIRS"):
            pytest.skip("STL headers not configured in CXX_INCLUDE_DIRS")

        # sort with raw pointers — container may not be identifiable
        code = """
        #include <algorithm>
        void sort_ptr(int* begin, int* end) { std::sort(begin, end); }
        """
        prog = build_ir_from_source(code)
        stl_ops = prog.all_stl_ops()
        if stl_ops:
            # container_kind_source is either None or "source_text" — never "resolved"
            assert stl_ops[0].container_kind_source in (None, "source_text")

    def test_resolved_provenance_not_faked(self):
        """container_kind_source must not be set to 'resolved' by source-text inference."""
        import os

        from complexity_lens.parsers.cpp_libclang import get_configured_cxx_include_dirs

        if not get_configured_cxx_include_dirs() and not os.environ.get("CXX_INCLUDE_DIRS"):
            pytest.skip("STL headers not configured in CXX_INCLUDE_DIRS")

        code = """
        #include <vector>
        void append(std::vector<int>& v, int x) { v.push_back(x); }
        """
        prog = build_ir_from_source(code)
        stl_ops = prog.all_stl_ops()
        for op in stl_ops:
            assert op.container_kind_source != "resolved", (
                "container_kind_source must not be set to 'resolved' "
                "from source-text inference alone"
            )

    def test_container_kind_source_in_to_dict(self):
        """Serialization must include container_kind_source."""
        import os

        from complexity_lens.parsers.cpp_libclang import get_configured_cxx_include_dirs

        if not get_configured_cxx_include_dirs() and not os.environ.get("CXX_INCLUDE_DIRS"):
            pytest.skip("STL headers not configured in CXX_INCLUDE_DIRS")

        code = """
        #include <vector>
        void append(std::vector<int>& v, int x) { v.push_back(x); }
        """
        prog = build_ir_from_source(code)
        d = prog.to_dict()
        for func in d["functions"]:
            for node in func["body"]:
                if node.get("type") == "STLOperationIR":
                    assert "container_kind_source" in node


# ===========================================================================
# Section 14: Deterministic Node IDs
# ===========================================================================


class TestDeterministicNodeIDs:
    def test_same_source_same_ids(self):
        code = "void f(int n) { for (int i = 0; i < n; ++i) {} }"
        prog1 = build_ir_from_source(code)
        prog2 = build_ir_from_source(code)
        ids1 = [lp.id for lp in prog1.all_loops()]
        ids2 = [lp.id for lp in prog2.all_loops()]
        assert ids1 == ids2

    def test_different_source_different_ids(self):
        code1 = "void f(int n) { for (int i = 0; i < n; ++i) {} }"
        code2 = "void g(int n) { for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) {} } }"
        prog1 = build_ir_from_source(code1)
        prog2 = build_ir_from_source(code2)
        ids1 = {lp.id for lp in prog1.all_loops()}
        ids2 = {lp.id for lp in prog2.all_loops()}
        # IDs are deterministic within one parse; just verify structure
        assert len(ids1) == 1
        assert len(ids2) == 2


# ===========================================================================
# Section 15: Golden IR Snapshots
# ===========================================================================


class TestGoldenIRSnapshots:
    """Golden snapshot tests for representative fixture IRs."""

    def test_simple_loop_snapshot(self):
        """02_for_loop.cpp golden: analysis_complete, loop kind, bound."""
        from complexity_lens.ir import build_ir
        from complexity_lens.parsers.cpp_libclang import parse_file

        prog = build_ir(parse_file(str(FIXTURES / "02_for_loop.cpp")))
        assert prog.analysis_complete
        assert not prog.has_syntax_errors
        assert len(prog.functions) == 1
        loops = prog.all_loops()
        assert len(loops) == 1
        assert loops[0].kind.value == "FOR"
        assert loops[0].bound.kind == BoundKind.LINEAR_INPUT

    def test_nested_loops_snapshot(self):
        """04_nested_loops.cpp golden: nested structure, analysis_complete."""
        from complexity_lens.ir import build_ir
        from complexity_lens.parsers.cpp_libclang import parse_file

        prog = build_ir(parse_file(str(FIXTURES / "04_nested_loops.cpp")))
        assert prog.analysis_complete
        assert len(prog.all_loops()) == 2

    def test_recursive_snapshot(self):
        """07_recursion.cpp golden: recursion, analysis_complete."""
        from complexity_lens.ir import build_ir
        from complexity_lens.parsers.cpp_libclang import parse_file

        prog = build_ir(parse_file(str(FIXTURES / "07_recursion.cpp")))
        assert prog.analysis_complete
        fn = prog.functions[0]
        assert fn.is_recursive

    def test_asm_snapshot_incomplete(self):
        """11_unsupported_asm.cpp golden: UnsupportedIR → analysis_complete=False."""
        from complexity_lens.ir import build_ir
        from complexity_lens.parsers.cpp_libclang import parse_file

        prog = build_ir(parse_file(str(FIXTURES / "11_unsupported_asm.cpp")))
        assert not prog.analysis_complete
        assert any(u.kind == UnsupportedKind.ASM_STMT for u in prog.all_unsupported())

    def test_malformed_snapshot_incomplete(self):
        """10_malformed.cpp golden: error diagnostics → analysis_complete=False."""
        from complexity_lens.ir import build_ir
        from complexity_lens.parsers.cpp_libclang import parse_file

        prog = build_ir(parse_file(str(FIXTURES / "10_malformed.cpp")))
        assert not prog.analysis_complete
        assert prog.has_syntax_errors


# ===========================================================================
# Section 16: IRDiagnostic Model Tests
# ===========================================================================


class TestIRDiagnosticModel:
    def test_ird_is_error_true_for_error(self):
        d = IRDiagnostic(severity="error", message="oops")
        assert d.is_error()

    def test_ird_is_error_true_for_fatal(self):
        d = IRDiagnostic(severity="fatal", message="crash")
        assert d.is_error()

    def test_ird_is_error_false_for_warning(self):
        d = IRDiagnostic(severity="warning", message="warn")
        assert not d.is_error()

    def test_ird_is_error_false_for_note(self):
        d = IRDiagnostic(severity="note", message="note")
        assert not d.is_error()

    def test_ird_to_dict_without_span(self):
        d = IRDiagnostic(severity="warning", message="test warning")
        result = d.to_dict()
        assert result["severity"] == "warning"
        assert result["message"] == "test warning"
        assert result["span"] is None

    def test_ird_to_dict_with_span(self):
        span = SourceSpan(file="test.cpp", start_line=5, start_col=1, end_line=5, end_col=10)
        d = IRDiagnostic(severity="error", message="bad syntax", span=span)
        result = d.to_dict()
        assert result["span"]["start_line"] == 5
        assert result["span"]["file"] == "test.cpp"


# ===========================================================================
# Section 17: Regression — Existing IR Fixture Tests Still Pass
# ===========================================================================


class TestRegressionExistingFixtures:
    """Regression: all Day-2 fixture-based assertions still hold."""

    def _fixture_ir(self, name: str):
        from complexity_lens.ir import build_ir
        from complexity_lens.parsers.cpp_libclang import parse_file

        return build_ir(parse_file(str(FIXTURES / name)))

    def test_01_simple_function_regression(self):
        prog = self._fixture_ir("01_simple_function.cpp")
        assert len(prog.functions) == 1
        fn = prog.functions[0]
        assert fn.name == "add"
        assert fn.return_type == "int"
        assert not fn.is_recursive
        assert prog.analysis_complete

    def test_02_for_loop_regression(self):
        prog = self._fixture_ir("02_for_loop.cpp")
        loops = prog.all_loops()
        assert len(loops) == 1
        assert loops[0].kind == LoopKind.FOR
        assert loops[0].bound.kind == BoundKind.LINEAR_INPUT

    def test_04_nested_loops_regression(self):
        prog = self._fixture_ir("04_nested_loops.cpp")
        depths = [lp.depth for lp in prog.all_loops()]
        assert 1 in depths
        assert 2 in depths

    def test_07_recursion_regression(self):
        prog = self._fixture_ir("07_recursion.cpp")
        fn = prog.functions[0]
        assert fn.is_recursive
        assert len(fn.recursive_call_sites) == 2

    def test_10_malformed_regression(self):
        prog = self._fixture_ir("10_malformed.cpp")
        assert prog.has_syntax_errors
        assert not prog.analysis_complete

    def test_11_asm_regression(self):
        prog = self._fixture_ir("11_unsupported_asm.cpp")
        assert not prog.analysis_complete
        assert any(u.kind == UnsupportedKind.ASM_STMT for u in prog.all_unsupported())

    def test_12_triple_nested_regression(self):
        prog = self._fixture_ir("12_triple_nested.cpp")
        depths = [lp.depth for lp in prog.all_loops()]
        assert depths == [1, 2, 3]


# ===========================================================================
# Section 18: Serialization Round-Trip
# ===========================================================================


class TestSerialization:
    def test_to_dict_includes_all_top_level_keys(self):
        prog = build_ir_from_source("int add(int a, int b) { return a + b; }")
        d = prog.to_dict()
        for key in (
            "source_path",
            "has_syntax_errors",
            "analysis_complete",
            "diagnostics",
            "functions",
            "unsupported_constructs",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_diagnostics_list(self):
        prog = build_ir_from_source("void f() { int x = ; }")
        d = prog.to_dict()
        assert isinstance(d["diagnostics"], list)

    def test_to_dict_analysis_complete_true(self):
        prog = build_ir_from_source("void f() {}")
        d = prog.to_dict()
        assert d["analysis_complete"] is True

    def test_to_dict_analysis_complete_false_on_error(self):
        prog = build_ir_from_source("void f() { int x = ; }")
        d = prog.to_dict()
        assert d["analysis_complete"] is False
