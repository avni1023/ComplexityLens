"""
model.py — Small typed Intermediate Representation (IR) for ComplexityLens.

This module defines the pure, strongly-typed AST/IR data structures for
functions, loops, bounds/updates, function calls, recursion sites, recognized
STL operations, source spans, and unsupported constructs.

Parsing logic is kept completely separate from these models.

``analysis_complete``
---------------------
``ProgramIR.analysis_complete = True`` means no known completeness failures
were encountered while constructing the supported IR for this translation unit.
It does **not** guarantee that any future complexity prediction is
mathematically exact.  The flag is set to ``False`` by:

* Any error-severity (``"error"`` or ``"fatal"``) diagnostic.
* Any ``UnsupportedIR`` node at any nesting level.

Warning-only diagnostics do not automatically make the IR incomplete.

``container_kind_source``
-------------------------
``STLOperationIR.container_kind_source`` records how the container kind was
determined:

* ``"resolved"``  — the parser successfully resolved the container's declared
  type; this is a genuine semantic fact.
* ``"source_text"`` — the container kind was inferred from declaration source
  text (e.g. a ``vector<…>`` spelling); this is a best-effort guess and MUST
  NOT be presented as a resolved semantic type.
* ``None``  — the container kind is unknown or could not be determined.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any


class LoopKind(str, Enum):
    FOR = "FOR"
    WHILE = "WHILE"
    DO_WHILE = "DO_WHILE"
    RANGE_FOR = "RANGE_FOR"


class BoundKind(str, Enum):
    CONSTANT = "CONSTANT"           # e.g., i < 100
    LINEAR_INPUT = "LINEAR_INPUT"   # e.g., i < n, i < v.size()
    DEPENDENT = "DEPENDENT"         # e.g., j < i (depends on outer loop variable)
    UNKNOWN = "UNKNOWN"             # Not recognized, explicit UNKNOWN


class UpdateKind(str, Enum):
    ADDITIVE = "ADDITIVE"             # i++, i += c, i--, i -= c
    MULTIPLICATIVE = "MULTIPLICATIVE" # i *= c, i /= c, i >>= 1
    UNKNOWN = "UNKNOWN"               # Not recognized, explicit UNKNOWN


class STLOperationKind(str, Enum):
    SORT = "SORT"                     # std::sort, std::stable_sort
    FIND = "FIND"                     # std::find
    COUNT = "COUNT"                   # std::count
    BINARY_SEARCH = "BINARY_SEARCH"   # std::binary_search
    LOWER_BOUND = "LOWER_BOUND"       # std::lower_bound, std::upper_bound
    PUSH_BACK = "PUSH_BACK"           # v.push_back(), v.emplace_back()
    INSERT = "INSERT"                 # container.insert()
    ERASE = "ERASE"                   # container.erase()
    HASH_LOOKUP = "HASH_LOOKUP"       # unordered_map / unordered_set lookup
    ORDERED_LOOKUP = "ORDERED_LOOKUP" # map / set lookup


class UnsupportedKind(str, Enum):
    ASM_STMT = "ASM_STMT"                 # Inline assembly (__asm__)
    UNKNOWN_SYNTAX = "UNKNOWN_SYNTAX"     # Unrecognized / syntax error construct
    COMPLEX_POINTER = "COMPLEX_POINTER"   # Pointer aliasing / cast
    OPAQUE_CONSTRUCT = "OPAQUE_CONSTRUCT" # Other unsupported constructs


class DiagnosticSeverity(str, Enum):
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclasses.dataclass(frozen=True)
class SourceSpan:
    """Accurate source span representing start and end location in a file."""

    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def __str__(self) -> str:
        return f"{self.file}:{self.start_line}:{self.start_col}-{self.end_line}:{self.end_col}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "start_line": self.start_line,
            "start_col": self.start_col,
            "end_line": self.end_line,
            "end_col": self.end_col,
        }


@dataclasses.dataclass
class IRDiagnostic:
    """A structured diagnostic associated with a ProgramIR.

    Parameters
    ----------
    severity:
        One of ``"note"``, ``"warning"``, ``"error"``, ``"fatal"``.
    message:
        Human-readable description of the diagnostic.
    span:
        Source location, if available.
    """

    severity: str  # matches DiagnosticSeverity values
    message: str
    span: SourceSpan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "span": self.span.to_dict() if self.span else None,
        }

    def is_error(self) -> bool:
        """Return True for error- or fatal-severity diagnostics."""
        return self.severity in ("error", "fatal")


@dataclasses.dataclass
class LoopBound:
    """Loop termination condition bound classification."""

    kind: BoundKind
    variable: str | None = None       # e.g., "i"
    bound_expr: str | None = None     # e.g., "n", "10", "i"
    comparison_op: str | None = None  # e.g., "<", "<=", "!="

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "variable": self.variable,
            "bound_expr": self.bound_expr,
            "comparison_op": self.comparison_op,
        }


@dataclasses.dataclass
class LoopUpdate:
    """Loop step update classification."""

    kind: UpdateKind
    variable: str | None = None   # e.g., "i"
    step_expr: str | None = None  # e.g., "++", "+= 1", "*= 2"
    step_op: str | None = None    # e.g., "++", "+=", "*="

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "variable": self.variable,
            "step_expr": self.step_expr,
            "step_op": self.step_op,
        }


@dataclasses.dataclass
class CallIR:
    """A function call expression."""

    id: str
    callee_name: str
    arguments: list[str]
    span: SourceSpan
    is_recursion: bool = False
    shrink_kind: str | None = None  # "minus_const", "divide", "other", or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "CallIR",
            "callee_name": self.callee_name,
            "arguments": self.arguments,
            "span": self.span.to_dict(),
            "is_recursion": self.is_recursion,
            "shrink_kind": self.shrink_kind,
        }


@dataclasses.dataclass
class STLOperationIR:
    """A recognized C++ Standard Template Library operation.

    Parameters
    ----------
    container_kind_source:
        How the container kind was inferred.  One of:

        * ``"resolved"``    — parser resolved the declared type.
        * ``"source_text"`` — inferred from source-text spelling; best-effort.
        * ``None``          — unknown / not determined.
    """

    id: str
    operation: STLOperationKind
    callee_name: str
    container: str | None
    arguments: list[str]
    span: SourceSpan
    container_kind_source: str | None = None  # "resolved", "source_text", or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "STLOperationIR",
            "operation": self.operation.value,
            "callee_name": self.callee_name,
            "container": self.container,
            "container_kind_source": self.container_kind_source,
            "arguments": self.arguments,
            "span": self.span.to_dict(),
        }


@dataclasses.dataclass
class UnsupportedIR:
    """Explicit representation of syntax or construct unsupported by static rules."""

    id: str
    kind: UnsupportedKind
    raw_spelling: str
    reason: str
    span: SourceSpan

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "UnsupportedIR",
            "kind": self.kind.value,
            "raw_spelling": self.raw_spelling,
            "reason": self.reason,
            "span": self.span.to_dict(),
        }


@dataclasses.dataclass
class LoopIR:
    """A loop construct (for, while, do-while, range-for)."""

    id: str
    kind: LoopKind
    depth: int
    bound: LoopBound
    update: LoopUpdate
    span: SourceSpan
    has_early_exit: bool = False
    body: list[Any] = dataclasses.field(default_factory=list)

    def all_loops(self) -> list[LoopIR]:
        res = [self]
        for item in self.body:
            if isinstance(item, LoopIR):
                res.extend(item.all_loops())
        return res

    def all_calls(self) -> list[CallIR]:
        res: list[CallIR] = []
        for item in self.body:
            if isinstance(item, CallIR):
                res.append(item)
            elif isinstance(item, LoopIR):
                res.extend(item.all_calls())
        return res

    def all_stl_ops(self) -> list[STLOperationIR]:
        res: list[STLOperationIR] = []
        for item in self.body:
            if isinstance(item, STLOperationIR):
                res.append(item)
            elif isinstance(item, LoopIR):
                res.extend(item.all_stl_ops())
        return res

    def all_unsupported(self) -> list[UnsupportedIR]:
        res: list[UnsupportedIR] = []
        for item in self.body:
            if isinstance(item, UnsupportedIR):
                res.append(item)
            elif isinstance(item, LoopIR):
                res.extend(item.all_unsupported())
        return res

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "LoopIR",
            "kind": self.kind.value,
            "depth": self.depth,
            "bound": self.bound.to_dict(),
            "update": self.update.to_dict(),
            "has_early_exit": self.has_early_exit,
            "span": self.span.to_dict(),
            "body": [node.to_dict() for node in self.body if hasattr(node, "to_dict")],
        }


@dataclasses.dataclass
class FunctionIR:
    """A top-level function definition."""

    id: str
    name: str
    return_type: str
    parameters: list[tuple[str, str]]  # [(type, name), ...]
    span: SourceSpan
    body: list[Any] = dataclasses.field(default_factory=list)
    is_recursive: bool = False
    recursive_call_sites: list[CallIR] = dataclasses.field(default_factory=list)

    def all_loops(self) -> list[LoopIR]:
        res: list[LoopIR] = []
        for item in self.body:
            if isinstance(item, LoopIR):
                res.extend(item.all_loops())
        return res

    def all_calls(self) -> list[CallIR]:
        res: list[CallIR] = []
        for item in self.body:
            if isinstance(item, CallIR):
                res.append(item)
            elif isinstance(item, LoopIR):
                res.extend(item.all_calls())
        return res

    def all_stl_ops(self) -> list[STLOperationIR]:
        res: list[STLOperationIR] = []
        for item in self.body:
            if isinstance(item, STLOperationIR):
                res.append(item)
            elif isinstance(item, LoopIR):
                res.extend(item.all_stl_ops())
        return res

    def all_unsupported(self) -> list[UnsupportedIR]:
        res: list[UnsupportedIR] = []
        for item in self.body:
            if isinstance(item, UnsupportedIR):
                res.append(item)
            elif isinstance(item, LoopIR):
                res.extend(item.all_unsupported())
        return res

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "FunctionIR",
            "name": self.name,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "span": self.span.to_dict(),
            "is_recursive": self.is_recursive,
            "recursive_call_sites": [c.to_dict() for c in self.recursive_call_sites],
            "body": [node.to_dict() for node in self.body if hasattr(node, "to_dict")],
        }


@dataclasses.dataclass
class ProgramIR:
    """The root Intermediate Representation of a C++ translation unit.

    Attributes
    ----------
    analysis_complete:
        ``True`` iff no completeness failures were detected during IR
        construction.  Set to ``False`` by any error/fatal diagnostic or any
        ``UnsupportedIR`` node.  Does **not** guarantee mathematical exactness
        of future complexity predictions.
    diagnostics:
        Structured list of :class:`IRDiagnostic` objects with severity,
        message, and optional span.
    """

    source_path: str
    functions: list[FunctionIR] = dataclasses.field(default_factory=list)
    unsupported_constructs: list[UnsupportedIR] = dataclasses.field(default_factory=list)
    has_syntax_errors: bool = False
    diagnostics: list[IRDiagnostic] = dataclasses.field(default_factory=list)
    analysis_complete: bool = True

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def all_loops(self) -> list[LoopIR]:
        res: list[LoopIR] = []
        for func in self.functions:
            res.extend(func.all_loops())
        return res

    def all_calls(self) -> list[CallIR]:
        res: list[CallIR] = []
        for func in self.functions:
            res.extend(func.all_calls())
        return res

    def all_stl_ops(self) -> list[STLOperationIR]:
        res: list[STLOperationIR] = []
        for func in self.functions:
            res.extend(func.all_stl_ops())
        return res

    def all_unsupported(self) -> list[UnsupportedIR]:
        res = list(self.unsupported_constructs)
        for func in self.functions:
            res.extend(func.all_unsupported())
        return res

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "has_syntax_errors": self.has_syntax_errors,
            "analysis_complete": self.analysis_complete,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "functions": [f.to_dict() for f in self.functions],
            "unsupported_constructs": [u.to_dict() for u in self.unsupported_constructs],
        }

    def pretty_print(self) -> str:
        """Return a structured, readable indented view of the IR."""
        lines: list[str] = [f"ProgramIR({self.source_path})"]

        complete_tag = "COMPLETE" if self.analysis_complete else "INCOMPLETE"
        lines.append(f"  analysis_complete={self.analysis_complete} [{complete_tag}]")

        if self.diagnostics:
            err_diags = [d for d in self.diagnostics if d.is_error()]
            warn_diags = [d for d in self.diagnostics if not d.is_error()]
            if err_diags:
                lines.append(f"  [ERRORS] ({len(err_diags)} error/fatal diagnostics)")
                for d in err_diags:
                    loc = f" @ {d.span}" if d.span else ""
                    lines.append(f"    ! [{d.severity}] {d.message}{loc}")
            if warn_diags:
                lines.append(f"  [WARNINGS] ({len(warn_diags)} diagnostics)")
                for d in warn_diags:
                    loc = f" @ {d.span}" if d.span else ""
                    lines.append(f"    ~ [{d.severity}] {d.message}{loc}")

        if self.unsupported_constructs:
            lines.append("  [Top-Level Unsupported Constructs]")
            for u in self.unsupported_constructs:
                lines.append(f"    * {u.kind.value}: {u.reason} @ {u.span}")

        for f in self.functions:
            rec_tag = (
                f" [RECURSIVE, sites={len(f.recursive_call_sites)}]"
                if f.is_recursive
                else ""
            )
            params_str = ", ".join(f"{t} {n}" for t, n in f.parameters)
            lines.append(
                f"  Function {f.name}({params_str}) -> {f.return_type}{rec_tag} @ {f.span}"
            )

            def _print_body(items: list[Any], indent: int) -> None:
                ind = "  " * indent
                for item in items:
                    if isinstance(item, LoopIR):
                        bound_desc = (
                            f"{item.bound.kind.value}(var={item.bound.variable}, "
                            f"op='{item.bound.comparison_op}', bound='{item.bound.bound_expr}')"
                        )
                        update_desc = f"{item.update.kind.value}(step='{item.update.step_expr}')"
                        early = " [EARLY_EXIT]" if item.has_early_exit else ""
                        loop_head = f"{ind}- Loop {item.kind.value} (depth={item.depth}){early}"
                        lines.append(f"{loop_head} @ {item.span}")
                        lines.append(f"{ind}    Bound: {bound_desc}")
                        lines.append(f"{ind}    Update: {update_desc}")
                        _print_body(item.body, indent + 2)
                    elif isinstance(item, STLOperationIR):
                        cont_desc = f" on {item.container}" if item.container else ""
                        kind_src = (
                            f" [{item.container_kind_source}]"
                            if item.container_kind_source
                            else ""
                        )
                        args_s = ", ".join(item.arguments)
                        lines.append(
                            f"{ind}- STL {item.operation.value}{cont_desc}{kind_src}: "
                            f"{item.callee_name}({args_s}) @ {item.span}"
                        )
                    elif isinstance(item, CallIR):
                        rec = (
                            f" [RECURSION: shrink={item.shrink_kind}]"
                            if item.is_recursion
                            else ""
                        )
                        args_s = ", ".join(item.arguments)
                        lines.append(
                            f"{ind}- Call: {item.callee_name}({args_s}){rec} @ {item.span}"
                        )
                    elif isinstance(item, UnsupportedIR):
                        lines.append(
                            f"{ind}- Unsupported: {item.kind.value} ({item.reason}) @ {item.span}"
                        )

            _print_body(f.body, 2)
        return "\n".join(lines)
