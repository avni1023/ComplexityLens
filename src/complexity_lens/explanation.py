"""
explanation.py — Evidence-backed explanation and recommendation module for ComplexityLens.

Reads existing ProgramIR, AnalysisResult, SourceSpans, and RuleTraces to produce
structured, evidence-backed explanations for user inspection:
- Summarizes detected structural constructs (loops, recursion, STL calls, branches).
- Details step-by-step derivation reasoning directly from engine rule traces.
- Preserves accurate source spans and line mappings.
- Honest unknown-reason reporting and explicit system limitations (e.g., absence
  of space complexity, absence of parser timeout).
- Conservative, evidence-based recommendations only when supported by observed patterns.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from complexity_lens.complexity.models import AnalysisResult, ReasonCode
from complexity_lens.fusion import FusionResult, FusionSource
from complexity_lens.ir.model import (
    CallIR,
    FunctionIR,
    ProgramIR,
    SourceSpan,
    STLOperationIR,
    STLOperationKind,
)


@dataclasses.dataclass(frozen=True)
class SourceSpanInfo:
    """Serializable source code span reference."""

    file: str | None
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    description: str = ""

    @classmethod
    def from_source_span(cls, span: SourceSpan, description: str = "") -> SourceSpanInfo:
        return cls(
            file=span.file,
            start_line=span.start_line,
            start_col=span.start_col,
            end_line=span.end_line,
            end_col=span.end_col,
            description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "end_col": self.end_col,
            "end_line": self.end_line,
            "file": self.file,
            "start_col": self.start_col,
            "start_line": self.start_line,
        }


@dataclasses.dataclass(frozen=True)
class DetectedConstruct:
    """A verified structural construct detected in the Program IR."""

    kind: str
    description: str
    span: SourceSpanInfo | None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "details": dict(self.details),
            "kind": self.kind,
            "span": self.span.to_dict() if self.span else None,
        }


@dataclasses.dataclass(frozen=True)
class ExplanationReport:
    """Structured, evidence-grounded explanation for static and hybrid analysis."""

    selected_function: str
    summary: str
    detected_constructs: list[DetectedConstruct]
    evidence_reasoning: list[str]
    relevant_spans: list[SourceSpanInfo]
    unsupported_constructs: list[str]
    unknown_reason: str | None
    limitations: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_constructs": [c.to_dict() for c in self.detected_constructs],
            "evidence_reasoning": list(self.evidence_reasoning),
            "limitations": list(self.limitations),
            "recommendations": list(self.recommendations),
            "relevant_spans": [s.to_dict() for s in self.relevant_spans],
            "selected_function": self.selected_function,
            "summary": self.summary,
            "unknown_reason": self.unknown_reason,
            "unsupported_constructs": list(self.unsupported_constructs),
        }


def _find_function(program: ProgramIR, name: str) -> FunctionIR | None:
    """Locate a FunctionIR by name in the program."""
    for func in program.functions:
        if func.name == name:
            return func
    return None


def generate_explanation(
    program: ProgramIR | None,
    analysis_result: AnalysisResult | None,
    fusion_result: FusionResult | None = None,
) -> ExplanationReport:
    """Generate structured evidence-backed explanation and conservative recommendations.

    Parameters
    ----------
    program:
        ProgramIR constructed from C++ source, or None if parsing/validation failed.
    analysis_result:
        AnalysisResult from the static complexity engine, or None.
    fusion_result:
        FusionResult from the hybrid fusion gate, or None.

    Returns
    -------
    ExplanationReport
        Detailed, evidence-backed breakdown with verified spans and recommendations.
    """
    func_name = analysis_result.selected_function if analysis_result else "<unknown>"
    func_ir = _find_function(program, func_name) if program and func_name != "<unknown>" else None

    detected: list[DetectedConstruct] = []
    spans: list[SourceSpanInfo] = []
    reasoning: list[str] = []
    unsupported: list[str] = []
    recommendations: list[str] = []

    # 1. Detected structural constructs
    if func_ir is not None:
        # Loops
        for loop in func_ir.all_loops():
            bound_kind = loop.bound.kind.value if loop.bound else "UNKNOWN"
            desc = f"{loop.kind.value.capitalize()} loop with {bound_kind.lower()} bound"
            loop_var = loop.bound.variable if loop.bound else loop.init_var
            if loop_var:
                desc += f" on variable '{loop_var}'"
            span_info = SourceSpanInfo.from_source_span(loop.span, desc) if loop.span else None
            detected.append(
                DetectedConstruct(
                    kind="loop",
                    description=desc,
                    span=span_info,
                    details={
                        "bound_kind": bound_kind,
                        "depth": loop.depth,
                        "is_nested": loop.depth > 1,
                        "loop_id": loop.id,
                        "loop_kind": loop.kind.value,
                        "variable": loop_var,
                    },
                )
            )
            if span_info:
                spans.append(span_info)

        # Recursion
        if func_ir.is_recursive:
            call_count = len(func_ir.recursive_call_sites)
            rec_desc = (
                f"Recursive function '{func_ir.name}' with {call_count} recursive call site(s)"
            )
            rec_span = (
                SourceSpanInfo.from_source_span(func_ir.span, rec_desc) if func_ir.span else None
            )
            shrink_kinds = [c.shrink_kind for c in func_ir.recursive_call_sites if c.shrink_kind]
            detected.append(
                DetectedConstruct(
                    kind="recursion",
                    description=rec_desc,
                    span=rec_span,
                    details={
                        "call_site_count": len(func_ir.recursive_call_sites),
                        "shrink_kinds": shrink_kinds,
                    },
                )
            )
            if rec_span:
                spans.append(rec_span)

        # STL operations
        for stl_op in func_ir.all_stl_ops():
            op_desc = f"STL operation '{stl_op.operation.value}'"
            if stl_op.container:
                op_desc += f" on container '{stl_op.container}'"
            stl_span = (
                SourceSpanInfo.from_source_span(stl_op.span, op_desc) if stl_op.span else None
            )
            detected.append(
                DetectedConstruct(
                    kind="stl_operation",
                    description=op_desc,
                    span=stl_span,
                    details={
                        "container": stl_op.container,
                        "container_kind_source": stl_op.container_kind_source,
                        "operation": stl_op.operation.value,
                    },
                )
            )
            if stl_span:
                spans.append(stl_span)

        # Unsupported constructs in function
        for u in func_ir.all_unsupported():
            u_desc = f"Unsupported construct: {u.reason}"
            u_span = SourceSpanInfo.from_source_span(u.span, u_desc) if u.span else None
            unsupported.append(u_desc)
            detected.append(
                DetectedConstruct(
                    kind="unsupported",
                    description=u_desc,
                    span=u_span,
                    details={"kind": u.kind.value, "raw_spelling": u.raw_spelling},
                )
            )
            if u_span:
                spans.append(u_span)

    # Top-level program unsupported / syntax errors
    if program:
        for u in program.unsupported_constructs:
            u_desc = f"Top-level construct: {u.reason}"
            u_span = SourceSpanInfo.from_source_span(u.span, u_desc) if u.span else None
            unsupported.append(u_desc)
            if u_span and u_span not in spans:
                spans.append(u_span)

    # 2. Evidence reasoning from AnalysisResult rule traces and assumptions
    if analysis_result:
        for trace in analysis_result.rule_trace:
            rule_id = trace.rule_id
            nodes_str = f" (nodes: {', '.join(trace.node_ids)})" if trace.node_ids else ""
            if rule_id == "RULE_LINEAR_LOOP":
                reasoning.append(f"Linear loop bound derives O(n) iteration work{nodes_str}.")
            elif rule_id == "RULE_CONSTANT_BOUND_LOOP":
                reasoning.append(f"Constant-bounded loop derives O(1) iteration work{nodes_str}.")
            elif rule_id == "RULE_LOG_UPDATE_LOOP":
                reasoning.append(
                    f"Logarithmic update loop derives O(log n) iteration work{nodes_str}."
                )
            elif rule_id == "RULE_NESTED_LOOP":
                reasoning.append(f"Nested loop structure multiplies iteration bounds{nodes_str}.")
            elif rule_id == "RULE_SEQUENTIAL_COMPOSITION":
                reasoning.append(
                    f"Sequential composition selects dominant asymptotic term: "
                    f"{trace.output_class}{nodes_str}."
                )
            elif rule_id == "RULE_STL_SORT":
                reasoning.append(
                    f"Standard sort algorithm derives O(n log n) comparisons{nodes_str}."
                )
            elif rule_id == "RULE_STL_BINARY_SEARCH":
                reasoning.append(f"Binary search derives O(log n) comparisons{nodes_str}.")
            elif rule_id == "RULE_HASH_LOOKUP":
                reasoning.append(
                    f"Hash container lookup derives expected average O(1) time{nodes_str}."
                )
            elif rule_id.startswith("RULE_RECURSION"):
                reasoning.append(
                    f"Recursion template '{rule_id}' establishes asymptotic bound "
                    f"{trace.output_class}{nodes_str}."
                )
            else:
                reasoning.append(f"Rule '{rule_id}' derived {trace.output_class}{nodes_str}.")

        # Assumptions
        for asm in analysis_result.assumptions:
            reasoning.append(f"Assumption: {asm}")

        # Add spans from analysis result if not already included
        for span in analysis_result.source_spans:
            s_info = SourceSpanInfo.from_source_span(span, "Analysis evidence span")
            if s_info not in spans:
                spans.append(s_info)

    # 3. Unknown reason summary
    unknown_reason: str | None = None
    is_unknown = (analysis_result is not None and analysis_result.taxonomy_label == "UNKNOWN") or (
        fusion_result is not None and fusion_result.primary_label == "UNKNOWN"
    )
    if is_unknown:
        reasons: list[str] = []
        if analysis_result:
            for code in analysis_result.reason_codes:
                if code == ReasonCode.UNKNOWN_BOUND.value:
                    reasons.append(
                        "Loop condition has non-constant or data-dependent bound "
                        "that cannot be proven statically"
                    )
                elif code == ReasonCode.LOOP_VAR_MODIFIED.value:
                    reasons.append(
                        "Loop induction variable is modified inside the loop body, "
                        "invalidating static bound"
                    )
                elif code == ReasonCode.UNSUPPORTED_CALL.value:
                    reasons.append("Function invokes external or unsupported function calls")
                elif code == ReasonCode.UNSUPPORTED_CONSTRUCT.value:
                    reasons.append(
                        "Source contains unsupported language constructs "
                        "(e.g. goto, custom templates)"
                    )
                elif code == ReasonCode.INCOMPLETE_IR.value:
                    reasons.append(
                        "IR construction was incomplete due to syntax or parser diagnostics"
                    )
        if unsupported:
            reasons.append(f"Unsupported constructs detected: {len(unsupported)}")
        if fusion_result and fusion_result.source == FusionSource.UNKNOWN.value:
            if fusion_result.decision_reason == "ML_FALLBACK_BELOW_THRESHOLD":
                reasons.append(
                    "ML fallback probability fell below the provisional confidence threshold"
                )
            elif fusion_result.decision_reason == "INCONCLUSIVE_NO_ML":
                reasons.append("ML inference was unavailable to provide an estimate")

        unknown_reason = (
            "; ".join(reasons)
            if reasons
            else "Complexity bound could not be verified by static rules."
        )

    # 4. Limitations (strict compliance with repository principles)
    limitations: list[str] = [
        "Auxiliary space complexity is not implemented or supported in ComplexityLens.",
        (
            "Static time analysis relies on syntactic AST patterns and structural loop bounds; "
            "data-dependent recurrence relations or unmodeled external APIs "
            "cannot be proven statically."
        ),
        (
            "Parser execution runs via in-process libclang and does not enforce an explicit "
            "subprocess timeout."
        ),
    ]
    if fusion_result and fusion_result.source == FusionSource.ML_ESTIMATE.value:
        limitations.append(
            "Primary result was provided by a machine learning estimate under a provisional "
            "decision threshold (0.55). It is NOT statically verified."
        )

    # 5. Conservative, evidence-backed recommendations
    if func_ir is not None:
        # Check if an STL find or linear search is nested inside a loop
        has_find_in_loop = False
        has_vector_mod_in_loop = False

        for loop in func_ir.all_loops():
            for item in loop.body:
                if isinstance(item, STLOperationIR):
                    if item.operation in (STLOperationKind.FIND, STLOperationKind.COUNT):
                        has_find_in_loop = True
                    if item.operation in (STLOperationKind.INSERT, STLOperationKind.ERASE):
                        if item.container in ("vector", "std::vector"):
                            has_vector_mod_in_loop = True
                elif isinstance(item, CallIR):
                    if item.callee_name in ("find", "std::find"):
                        has_find_in_loop = True

        if has_find_in_loop:
            recommendations.append(
                "Linear search detected inside a loop. If performing repeated membership lookups, "
                "consider an associative container (e.g. std::unordered_set) for average "
                "O(1) lookups."
            )
        if has_vector_mod_in_loop:
            recommendations.append(
                "std::vector insert/erase detected inside a loop. If frequently inserting or "
                "removing elements at non-terminal positions, consider std::deque or std::list "
                "to avoid linear element shifting."
            )

    # 6. High-level summary
    if fusion_result:
        label = fusion_result.primary_label
        src = fusion_result.source
        summary = f"Overall Complexity: {label} (Source: {src})"
    elif analysis_result:
        summary = (
            f"Static Complexity: {analysis_result.taxonomy_label} "
            f"(Exactness: {analysis_result.exactness})"
        )
    else:
        summary = "No complexity analysis results available."

    return ExplanationReport(
        selected_function=func_name,
        summary=summary,
        detected_constructs=detected,
        evidence_reasoning=reasoning,
        relevant_spans=spans,
        unsupported_constructs=unsupported,
        unknown_reason=unknown_reason,
        limitations=limitations,
        recommendations=recommendations,
    )
