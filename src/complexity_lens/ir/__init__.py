"""
complexity_lens.ir — Typed Intermediate Representation package.

Provides pure typed data classes for representing functions, loops,
loop bounds/updates, calls, recursion sites, STL operations, source spans,
and unsupported constructs.

Public Surface:
    - build_ir(parse_result) -> ProgramIR
    - ProgramIR, FunctionIR, LoopIR, CallIR, STLOperationIR, UnsupportedIR
    - IRDiagnostic
    - LoopBound, LoopUpdate, SourceSpan
    - BoundKind, UpdateKind, LoopKind, STLOperationKind, UnsupportedKind
    - DiagnosticSeverity
"""

from __future__ import annotations

from complexity_lens.ir.builder import (
    build_ir,
    build_ir_from_file,
    build_ir_from_source,
)
from complexity_lens.ir.model import (
    BoundKind,
    BranchIR,
    CallIR,
    DiagnosticSeverity,
    FunctionIR,
    IRDiagnostic,
    LoopBound,
    LoopIR,
    LoopKind,
    LoopUpdate,
    ProgramIR,
    SourceSpan,
    STLOperationIR,
    STLOperationKind,
    UnsupportedIR,
    UnsupportedKind,
    UpdateKind,
)

__all__ = [
    "BoundKind",
    "BranchIR",
    "CallIR",
    "DiagnosticSeverity",
    "FunctionIR",
    "IRDiagnostic",
    "LoopBound",
    "LoopIR",
    "LoopKind",
    "LoopUpdate",
    "ProgramIR",
    "STLOperationIR",
    "STLOperationKind",
    "SourceSpan",
    "UnsupportedIR",
    "UnsupportedKind",
    "UpdateKind",
    "build_ir",
    "build_ir_from_file",
    "build_ir_from_source",
]
