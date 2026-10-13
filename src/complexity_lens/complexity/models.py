"""
models.py — Structured complexity classes, reason codes, taxonomy, and analysis models.

Defines the core data structures for static time-complexity analysis:
- ComplexityClass: (p, q) ordinary polynomial/logarithmic class, plus EXP and UNKNOWN.
- Exactness: STRUCTURAL, UPPER_BOUND, UNKNOWN.
- TaxonomyLabel: O1, OLOGN, ON, ONLOGN, ON2, ON3, OEXP, OUT_OF_TAXONOMY, UNKNOWN.
- ReasonCode: stable reason codes describing analysis status.
- RuleTraceEntry: structured rule derivation trace entry.
- AnalysisResult: full structured analysis result for a selected function.
- FunctionSelectionError: structured error when function selection is ambiguous or missing.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

from complexity_lens.ir.model import SourceSpan


class Exactness(str, Enum):
    """Exactness status for an analysis result."""

    STRUCTURAL = "STRUCTURAL"
    UPPER_BOUND = "UPPER_BOUND"
    UNKNOWN = "UNKNOWN"


class TaxonomyLabel(str, Enum):
    """Discrete, human-readable taxonomy labels."""

    O1 = "O1"
    OLOGN = "OLOGN"
    ON = "ON"
    ONLOGN = "ONLOGN"
    ON2 = "ON2"
    ON3 = "ON3"
    OEXP = "OEXP"
    OUT_OF_TAXONOMY = "OUT_OF_TAXONOMY"
    UNKNOWN = "UNKNOWN"


class ReasonCode(str, Enum):
    """Stable reason codes explaining why an analysis has its status."""

    CONSTANT_WORK = "CONSTANT_WORK"
    LINEAR_LOOP = "LINEAR_LOOP"
    CONSTANT_BOUND_LOOP = "CONSTANT_BOUND_LOOP"
    LOGARITHMIC_UPDATE = "LOGARITHMIC_UPDATE"
    NESTED_LOOP = "NESTED_LOOP"
    SEQUENTIAL_COMPOSITION = "SEQUENTIAL_COMPOSITION"
    IF_ELSE_WORST_CASE = "IF_ELSE_WORST_CASE"
    UNKNOWN_BOUND = "UNKNOWN_BOUND"
    LOOP_VAR_MODIFIED = "LOOP_VAR_MODIFIED"
    UNSUPPORTED_CALL = "UNSUPPORTED_CALL"
    UNSUPPORTED_CONSTRUCT = "UNSUPPORTED_CONSTRUCT"
    INCOMPLETE_IR = "INCOMPLETE_IR"
    DEPENDENT_BOUND = "DEPENDENT_BOUND"
    SINGLE_INPUT_SYMBOL = "SINGLE_INPUT_SYMBOL"
    NAMED_CONSTANT_AS_INPUT_SIZE = "NAMED_CONSTANT_AS_INPUT_SIZE"
    AMORTIZED_CONSTANT_CALL = "AMORTIZED_CONSTANT_CALL"
    PARTIAL_IR = "PARTIAL_IR"


@dataclasses.dataclass(frozen=True)
class ComplexityClass:
    """Asymptotic complexity class: O(n^p (log n)^q), plus special EXP and UNKNOWN.

    Attributes
    ----------
    p:
        Non-negative integer polynomial degree, or None if special (EXP/UNKNOWN).
    q:
        Non-negative integer logarithmic degree, or None if special (EXP/UNKNOWN).
    is_exp:
        True if complexity is genuinely exponential O(2^n).
    is_unknown:
        True if complexity cannot be established.
    """

    p: int | None = 0
    q: int | None = 0
    is_exp: bool = False
    is_unknown: bool = False

    def __post_init__(self) -> None:
        if self.is_unknown:
            object.__setattr__(self, "p", None)
            object.__setattr__(self, "q", None)
            object.__setattr__(self, "is_exp", False)
        elif self.is_exp:
            object.__setattr__(self, "p", None)
            object.__setattr__(self, "q", None)
            object.__setattr__(self, "is_unknown", False)
        else:
            if self.p is None or self.p < 0:
                object.__setattr__(self, "p", 0)
            if self.q is None or self.q < 0:
                object.__setattr__(self, "q", 0)

    # ------------------------------------------------------------------
    # Canonical constructors
    # ------------------------------------------------------------------

    @classmethod
    def constant(cls) -> ComplexityClass:
        """O(1)"""
        return cls(p=0, q=0)

    @classmethod
    def logarithmic(cls) -> ComplexityClass:
        """O(log n)"""
        return cls(p=0, q=1)

    @classmethod
    def linear(cls) -> ComplexityClass:
        """O(n)"""
        return cls(p=1, q=0)

    @classmethod
    def linearithmic(cls) -> ComplexityClass:
        """O(n log n)"""
        return cls(p=1, q=1)

    @classmethod
    def quadratic(cls) -> ComplexityClass:
        """O(n^2)"""
        return cls(p=2, q=0)

    @classmethod
    def cubic(cls) -> ComplexityClass:
        """O(n^3)"""
        return cls(p=3, q=0)

    @classmethod
    def exponential(cls) -> ComplexityClass:
        """O(2^n)"""
        return cls(p=None, q=None, is_exp=True)

    @classmethod
    def unknown(cls) -> ComplexityClass:
        """UNKNOWN"""
        return cls(p=None, q=None, is_unknown=True)

    # ------------------------------------------------------------------
    # Taxonomy mapping
    # ------------------------------------------------------------------

    def to_taxonomy_label(self) -> str:
        """Map this complexity class into one of the discrete taxonomy labels."""
        if self.is_unknown:
            return TaxonomyLabel.UNKNOWN.value
        if self.is_exp:
            return TaxonomyLabel.OEXP.value

        p = self.p or 0
        q = self.q or 0

        if p == 0 and q == 0:
            return TaxonomyLabel.O1.value
        if p == 0 and q == 1:
            return TaxonomyLabel.OLOGN.value
        if p == 1 and q == 0:
            return TaxonomyLabel.ON.value
        if p == 1 and q == 1:
            return TaxonomyLabel.ONLOGN.value
        if p == 2 and q == 0:
            return TaxonomyLabel.ON2.value
        if p == 3 and q == 0:
            return TaxonomyLabel.ON3.value

        return TaxonomyLabel.OUT_OF_TAXONOMY.value

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if self.is_unknown:
            return "UNKNOWN"
        if self.is_exp:
            return "O(2^n)"

        p = self.p or 0
        q = self.q or 0

        if p == 0 and q == 0:
            return "O(1)"
        if p == 0 and q == 1:
            return "O(log n)"
        if p == 0 and q > 1:
            return f"O((log n)^{q})"
        if p == 1 and q == 0:
            return "O(n)"
        if p == 1 and q == 1:
            return "O(n log n)"
        if p == 1 and q > 1:
            return f"O(n (log n)^{q})"
        if p > 1 and q == 0:
            return f"O(n^{p})"
        if p > 1 and q == 1:
            return f"O(n^{p} log n)"
        return f"O(n^{p} (log n)^{q})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "p": self.p,
            "q": self.q,
            "is_exp": self.is_exp,
            "is_unknown": self.is_unknown,
            "label": self.to_taxonomy_label(),
            "formatted": str(self),
        }


@dataclasses.dataclass(frozen=True)
class RuleTraceEntry:
    """A single step in the rule trace recording how the engine derived a class.

    Attributes
    ----------
    rule_id:
        Identifier of the rule applied (e.g. 'RULE_LINEAR_LOOP', 'RULE_SEQUENTIAL_COMPOSITION').
    node_ids:
        Evidence IR node IDs associated with this step.
    input_classes:
        Complexity classes input to the rule.
    output_class:
        Complexity class produced by the rule.
    """

    rule_id: str
    node_ids: list[str]
    input_classes: list[ComplexityClass]
    output_class: ComplexityClass

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "node_ids": list(self.node_ids),
            "input_classes": [c.to_dict() for c in self.input_classes],
            "output_class": self.output_class.to_dict(),
        }


@dataclasses.dataclass
class AnalysisResult:
    """Complete, structured, deterministic result of static complexity analysis.

    Attributes
    ----------
    selected_function:
        Name of the function selected for analysis.
    selection_rationale:
        Deterministic rationale string explaining why this function was chosen.
    complexity_class:
        Derived ComplexityClass instance.
    taxonomy_label:
        Human-readable discrete label (O1, ON, etc.).
    exactness:
        STRUCTURAL, UPPER_BOUND, or UNKNOWN.
    evidence_node_ids:
        List of IR node IDs contributing evidence to the result.
    source_spans:
        Source code spans corresponding to the evidence.
    reason_codes:
        Stable reason codes summarizing why the result has its status.
    explanation:
        Deterministic explanation string detailing the derivation and assumptions.
    notes:
        Completeness or partial-IR notes where applicable.
    rule_trace:
        Structured trace of rule applications derivation steps.
    """

    selected_function: str
    selection_rationale: str
    complexity_class: ComplexityClass
    taxonomy_label: str
    exactness: str
    evidence_node_ids: list[str] = dataclasses.field(default_factory=list)
    source_spans: list[SourceSpan] = dataclasses.field(default_factory=list)
    reason_codes: list[str] = dataclasses.field(default_factory=list)
    explanation: str = ""
    notes: list[str] = dataclasses.field(default_factory=list)
    rule_trace: list[RuleTraceEntry] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_function": self.selected_function,
            "selection_rationale": self.selection_rationale,
            "complexity_class": self.complexity_class.to_dict(),
            "taxonomy_label": self.taxonomy_label,
            "exactness": self.exactness,
            "evidence_node_ids": list(self.evidence_node_ids),
            "source_spans": [s.to_dict() for s in self.source_spans],
            "reason_codes": list(self.reason_codes),
            "explanation": self.explanation,
            "notes": list(self.notes),
            "rule_trace": [entry.to_dict() for entry in self.rule_trace],
        }


class FunctionSelectionError(ValueError):
    """Raised when function selection is ambiguous or a requested function does not exist."""

    def __init__(self, message: str, available_functions: list[str] | None = None) -> None:
        super().__init__(message)
        self.available_functions = available_functions or []
