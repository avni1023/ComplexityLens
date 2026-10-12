"""
cpp_libclang.py — libclang-backed C++ AST extractor (feasibility spike).

This module is intentionally minimal: it proves that libclang can parse every
construct covered by the Day-1 fixtures and surfaces enough information for a
downstream complexity analyser.  It does NOT build a full IR or execute code.

Public surface
--------------
parse_file(path, extra_args=None) -> ParseResult
parse_source(source, filename="<string>.cpp", extra_args=None) -> ParseResult
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# libclang bootstrap
# ---------------------------------------------------------------------------
try:
    import clang.cindex as _ci
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "libclang Python bindings not found.  "
        "Install with: pip install libclang==18.1.1"
    ) from exc

# The libclang wheel ships its own libclang.dll/so under
# site-packages/clang/native/  — the package registers it automatically when
# imported, so no manual Config.set_library_file() is needed unless the user
# has a system LLVM they prefer.  We honour LIBCLANG_PATH if set.
_LIBCLANG_PATH = os.environ.get("LIBCLANG_PATH")
if _LIBCLANG_PATH:
    _ci.Config.set_library_file(_LIBCLANG_PATH)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SourceLoc:
    """A (file, line, column) triple."""

    file: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclasses.dataclass
class ASTNode:
    """Lightweight, serialisable snapshot of one libclang cursor."""

    kind: str           # CursorKind.name
    spelling: str       # identifier / operator text
    loc: SourceLoc
    extent_start: SourceLoc
    extent_end: SourceLoc
    depth: int          # 0 = translation-unit children
    children: list["ASTNode"] = dataclasses.field(default_factory=list)
    type_spelling: str = ""
    tokens: list[str] = dataclasses.field(default_factory=list)

    def all_nodes(self) -> Generator["ASTNode", None, None]:
        """Depth-first iteration over self and all descendants."""
        yield self
        for child in self.children:
            yield from child.all_nodes()

    def nodes_of_kind(self, kind_name: str) -> list["ASTNode"]:
        return [n for n in self.all_nodes() if n.kind == kind_name]


@dataclasses.dataclass
class Diagnostic:
    severity: str       # "note" | "warning" | "error" | "fatal"
    message: str
    loc: SourceLoc | None


@dataclasses.dataclass
class ParseResult:
    """Everything the spike needs from one translation unit."""

    source_path: str
    top_level_nodes: list[ASTNode]
    diagnostics: list[Diagnostic]
    has_fatal_error: bool

    # Convenience -----------------------------------------------------------

    def all_nodes(self) -> Generator[ASTNode, None, None]:
        for node in self.top_level_nodes:
            yield from node.all_nodes()

    def nodes_of_kind(self, kind_name: str) -> list[ASTNode]:
        return [n for n in self.all_nodes() if n.kind == kind_name]

    def functions(self) -> list[ASTNode]:
        return self.nodes_of_kind("FUNCTION_DECL")

    def for_stmts(self) -> list[ASTNode]:
        return self.nodes_of_kind("FOR_STMT")

    def while_stmts(self) -> list[ASTNode]:
        return self.nodes_of_kind("WHILE_STMT")

    def call_exprs(self) -> list[ASTNode]:
        return self.nodes_of_kind("CALL_EXPR")

    def asm_stmts(self) -> list[ASTNode]:
        return self.nodes_of_kind("ASM_STMT")

    def max_loop_nesting_depth(self) -> int:
        """Return the deepest loop-nest level found in this TU."""
        LOOP_KINDS = {"FOR_STMT", "WHILE_STMT", "DO_STMT", "CXX_FOR_RANGE_STMT"}

        def _depth(node: ASTNode, current: int) -> int:
            d = current
            for child in node.children:
                d = max(d, _depth(child, current + (1 if child.kind in LOOP_KINDS else 0)))
            return d

        best = 0
        for n in self.top_level_nodes:
            best = max(best, _depth(n, 1 if n.kind in LOOP_KINDS else 0))
        return best

    def is_recursive(self, func_node: ASTNode) -> bool:
        """True if *func_node* contains a CallExpr that calls itself."""
        name = func_node.spelling
        return any(
            c.spelling == name
            for c in func_node.all_nodes()
            if c.kind == "CALL_EXPR"
        )

    # String summary --------------------------------------------------------

    def summary(self) -> str:
        lines = [f"=== ParseResult: {self.source_path} ==="]
        if self.diagnostics:
            lines.append(f"  diagnostics ({len(self.diagnostics)}):")
            for d in self.diagnostics:
                lines.append(f"    [{d.severity}] {d.loc}: {d.message}")
        kinds: dict[str, int] = {}
        for n in self.all_nodes():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        lines.append(f"  AST node kinds: {dict(sorted(kinds.items()))}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEVERITY_MAP = {
    _ci.Diagnostic.Note: "note",
    _ci.Diagnostic.Warning: "warning",
    _ci.Diagnostic.Error: "error",
    _ci.Diagnostic.Fatal: "fatal",
}


def _loc(clang_loc) -> SourceLoc:
    """Convert a libclang SourceLocation to our SourceLoc."""
    f = clang_loc.file
    return SourceLoc(
        file=f.name if f else "<unknown>",
        line=clang_loc.line,
        col=clang_loc.column,
    )


def _build_node(cursor, depth: int, source_path: str = "") -> ASTNode:
    """Recursively convert a libclang cursor tree into ASTNode trees."""
    loc = _loc(cursor.location)
    start = _loc(cursor.extent.start)
    end = _loc(cursor.extent.end)

    type_spelling = ""
    try:
        if cursor.kind == _ci.CursorKind.FUNCTION_DECL:
            type_spelling = cursor.result_type.spelling or ""
        elif cursor.type:
            type_spelling = cursor.type.spelling or ""
    except Exception:
        pass

    tokens: list[str] = []
    try:
        is_in_source = (
            (not source_path)
            or (loc.file == source_path)
            or (Path(loc.file).name == Path(source_path).name)
        )
        if is_in_source and (
            cursor.kind.is_expression()
            or cursor.kind.is_statement()
            or cursor.kind.is_declaration()
        ):
            if end.line - start.line <= 10:
                tokens = [t.spelling for t in cursor.get_tokens()]
    except Exception:
        pass

    node = ASTNode(
        kind=cursor.kind.name,
        spelling=cursor.spelling,
        loc=loc,
        extent_start=start,
        extent_end=end,
        depth=depth,
        children=[],
        type_spelling=type_spelling,
        tokens=tokens,
    )
    for child in cursor.get_children():
        node.children.append(_build_node(child, depth + 1, source_path))
    return node


def _diag(clang_diag) -> Diagnostic:
    loc = None
    try:
        loc = _loc(clang_diag.location)
    except Exception:  # pragma: no cover
        pass
    return Diagnostic(
        severity=_SEVERITY_MAP.get(clang_diag.severity, "unknown"),
        message=clang_diag.spelling,
        loc=loc,
    )


# Base arguments passed to libclang.
_BASE_ARGS = [
    "-std=c++17",
    "-x", "c++",
]

# MSVC compatibility flags for standalone parsing without MinGW headers.
_MS_COMPAT_ARGS = [
    "-fms-extensions",
    "-fms-compatibility",
    "-fms-compatibility-version=19",
]


def get_configured_cxx_include_dirs() -> list[Path]:
    """Return the list of directories configured in CXX_INCLUDE_DIRS.

    Splits on os.pathsep (';' on Windows, ':' on POSIX).
    Does NOT hardcode machine-specific paths, invoke external compiler executables,
    or use silent legacy toolchain fallbacks.
    """
    raw = os.environ.get("CXX_INCLUDE_DIRS", "").strip()
    if not raw:
        return []
    dirs: list[Path] = []
    for item in raw.split(os.pathsep):
        cleaned = item.strip().strip('"').strip("'")
        if cleaned:
            dirs.append(Path(cleaned))
    return dirs


def _get_cxx_include_args() -> list[str]:
    """Generate libclang -isystem arguments from CXX_INCLUDE_DIRS.

    Only directories that actually exist on disk are passed as -isystem flags.
    """
    args: list[str] = []
    for d in get_configured_cxx_include_dirs():
        if d.is_dir():
            args.extend(["-isystem", str(d)])
    return args


def _effective_args(extra_args: list[str] | None = None) -> list[str]:
    """Assemble compiler arguments for libclang.

    If CXX_INCLUDE_DIRS contains MinGW / UCRT include directories, -target x86_64-w64-mingw32
    is passed to align standard library macro expansions with GNU ABI. Otherwise, default
    MSVC compatibility flags are used for standalone parsing.
    """
    configured_dirs = get_configured_cxx_include_dirs()
    is_mingw = any(
        "mingw" in str(d).lower() or "ucrt" in str(d).lower()
        for d in configured_dirs
    )

    args = list(_BASE_ARGS)
    if is_mingw:
        args.extend(["-target", "x86_64-w64-mingw32"])
    else:
        args.extend(_MS_COMPAT_ARGS)

    args.extend(_get_cxx_include_args())
    if extra_args:
        args.extend(extra_args)
    return args


def parse_file(
    path: str | Path,
    extra_args: list[str] | None = None,
) -> ParseResult:
    """Parse a C++ source file and return a :class:`ParseResult`."""
    path = str(path)
    args = _effective_args(extra_args)
    index = _ci.Index.create()
    tu = index.parse(
        path,
        args=args,
        options=(
            _ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            | _ci.TranslationUnit.PARSE_INCOMPLETE
        ),
    )
    return _build_result(path, tu)


def parse_source(
    source: str,
    filename: str = "<string>.cpp",
    extra_args: list[str] | None = None,
) -> ParseResult:
    """Parse a C++ *source* string in-memory and return a :class:`ParseResult`."""
    args = _effective_args(extra_args)
    index = _ci.Index.create()
    tu = index.parse(
        filename,
        args=args,
        unsaved_files=[(filename, source)],
        options=(
            _ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            | _ci.TranslationUnit.PARSE_INCOMPLETE
        ),
    )
    return _build_result(filename, tu)


def _build_result(source_path: str, tu) -> ParseResult:
    diags = [_diag(d) for d in tu.diagnostics]
    has_fatal = any(d.severity == "fatal" for d in diags)

    top_level: list[ASTNode] = []
    for cursor in tu.cursor.get_children():
        top_level.append(_build_node(cursor, depth=0, source_path=source_path))

    return ParseResult(
        source_path=source_path,
        top_level_nodes=top_level,
        diagnostics=diags,
        has_fatal_error=has_fatal,
    )
