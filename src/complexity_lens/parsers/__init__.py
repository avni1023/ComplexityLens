"""parsers sub-package for ComplexityLens."""

from complexity_lens.parsers.cpp_libclang import (
    ASTNode,
    Diagnostic,
    ParseResult,
    SourceLoc,
    parse_file,
    parse_source,
)


class ClangParserError(Exception):
    """Raised when the libclang parser fails or is unavailable."""


__all__ = [
    "ASTNode",
    "ClangParserError",
    "Diagnostic",
    "ParseResult",
    "SourceLoc",
    "parse_file",
    "parse_source",
]
