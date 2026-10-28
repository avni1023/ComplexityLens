"""dataset.py — Dataset schema, loader, source hashing, and validation for ComplexityLens.

This module provides the independent ground-truth data models and dataset integrity
tooling for the ComplexityLens benchmark.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

from complexity_lens.parsers import ClangParserError, parse_source
from complexity_lens.validation import validate_source

# ---------------------------------------------------------------------------
# Closed Vocabularies
# ---------------------------------------------------------------------------

ALLOWED_LABELS: frozenset[str] = frozenset(
    {"O1", "OLOGN", "ON", "ONLOGN", "ON2", "ON3", "OEXP", "UNKNOWN"}
)

ALLOWED_TAGS: frozenset[str] = frozenset({"easy", "rule_hard"})

ALLOWED_EXPECTED_PARSE: frozenset[str] = frozenset(
    {"valid", "has_unsupported_constructs"}
)

ALLOWED_UNKNOWN_REASONS: frozenset[str] = frozenset(
    {
        "data_dependent",
        "termination",
        "unsupported_construct",
        "external_call",
        "irregular_recursion",
    }
)

ALLOWED_ORIGINS: frozenset[str] = frozenset({"hand_written", "mutation", "unspecified"})


# ---------------------------------------------------------------------------
# Source Hashing Policy
# ---------------------------------------------------------------------------


def normalize_source_for_hashing(source: str) -> str:
    """Normalize source text according to the exact ComplexityLens hashing policy.

    Policy:
    1. Replace Windows CRLF (``\\r\\n``) line endings with Unix LF (``\\n``).
    2. Trim trailing whitespace from the end of the entire source (``source.rstrip()``).
       Trailing whitespace on intermediate lines is preserved.
    """
    return source.replace("\r\n", "\n").rstrip()


def compute_source_sha256(source: str) -> str:
    """Compute deterministic SHA-256 hex digest for normalized C++ source text."""
    normalized = normalize_source_for_hashing(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_comment_stripped_sha256(source: str) -> str:
    """Compute optional diagnostic SHA-256 after stripping comments using lexical scanner.

    Uses a character-level scanner (no regular expressions) consistent with
    ``validation.py`` lexical rules. Does not replace ``source_sha256``.
    """
    cleaned: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        # Line comment
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            continue
        # Block comment
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i + 1 < n:
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            continue
        # String literal (preserve content)
        if ch == '"':
            cleaned.append(ch)
            i += 1
            while i < n:
                c = source[i]
                cleaned.append(c)
                i += 1
                if c == "\\" and i < n:
                    cleaned.append(source[i])
                    i += 1
                elif c == '"':
                    break
            continue
        cleaned.append(ch)
        i += 1

    stripped = "".join(cleaned)
    normalized = normalize_source_for_hashing(stripped)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProgramRecord:
    """Assembled program record representing a single benchmark source file."""

    program_id: str
    family_id: str
    source: str
    expected_label: str
    label_rationale: str
    source_sha256: str
    unknown_reason: str | None = None
    unknown_explanation: str | None = None
    label_status: str = "author_labeled"
    verified: bool = False
    origin: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a stable dictionary representation."""
        return {
            "program_id": self.program_id,
            "family_id": self.family_id,
            "source": self.source,
            "expected_label": self.expected_label,
            "label_rationale": self.label_rationale,
            "unknown_reason": self.unknown_reason,
            "unknown_explanation": self.unknown_explanation,
            "label_status": self.label_status,
            "verified": self.verified,
            "origin": self.origin,
            "source_sha256": self.source_sha256,
        }


@dataclasses.dataclass(frozen=True)
class FamilyMetadata:
    """Metadata specification for a family of related programs."""

    family_id: str
    family_idea: str
    label: str
    label_rationale: str
    tag: str
    expected_parse: str
    programs: tuple[str, ...]
    assumptions: tuple[str, ...]
    program_annotations: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "family_idea": self.family_idea,
            "label": self.label,
            "label_rationale": self.label_rationale,
            "tag": self.tag,
            "expected_parse": self.expected_parse,
            "programs": list(self.programs),
            "assumptions": list(self.assumptions),
            "program_annotations": self.program_annotations,
        }


# ---------------------------------------------------------------------------
# Simple YAML Parser
# ---------------------------------------------------------------------------


def _parse_yaml_file(path: Path) -> dict[str, Any]:
    """Parse a family YAML file using PyYAML if available or robust built-in fallback."""
    try:
        import yaml  # type: ignore[import-untyped]
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except ImportError:
        pass

    # Built-in robust parser for our family YAML schema
    lines = path.read_text(encoding="utf-8").splitlines()
    data: dict[str, Any] = {}
    current_key: str | None = None
    in_subdict: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Level 1 indentation (2 spaces)
        if line.startswith("  ") and not line.startswith("    "):
            if stripped.startswith("- "):
                val = stripped[2:].strip().strip("\"'")
                if current_key and isinstance(data.get(current_key), list):
                    data[current_key].append(val)
            elif ":" in stripped:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if current_key:
                    if current_key not in data or not isinstance(data[current_key], dict):
                        data[current_key] = {}
                    if not v:
                        in_subdict = k
                        data[current_key][k] = {}
                    else:
                        data[current_key][k] = v
        # Level 2 indentation (4 spaces)
        elif line.startswith("    "):
            if ":" in stripped and current_key and in_subdict:
                k, v = stripped.split(":", 1)
                if in_subdict in data[current_key]:
                    data[current_key][in_subdict][k.strip()] = v.strip().strip("\"'")
        # Top-level keys (no leading indentation)
        else:
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                current_key = k.strip()
                v = v.strip().strip("\"'")
                in_subdict = None
                if not v:
                    if current_key in ("programs", "assumptions"):
                        data[current_key] = []
                    else:
                        data[current_key] = {}
                else:
                    data[current_key] = v

    return data


# ---------------------------------------------------------------------------
# Dataset Loader and Validator
# ---------------------------------------------------------------------------


class DatasetValidationError(ValueError):
    """Raised when dataset structure, metadata, or files fail integrity checks."""


def load_dataset(
    dataset_root: str | Path = "datasets",
) -> tuple[list[ProgramRecord], list[FamilyMetadata], list[str]]:
    """Assemble program records and family metadata from dataset directory layout.

    Returns
    -------
    records:
        List of assembled ProgramRecord instances.
    families:
        List of FamilyMetadata instances.
    errors:
        List of human-readable integrity error strings (empty if validation passed).
    """
    root = Path(dataset_root)
    families_dir = root / "families"
    programs_dir = root / "programs"

    errors: list[str] = []
    families: list[FamilyMetadata] = []
    records: list[ProgramRecord] = []

    if not families_dir.exists():
        errors.append(f"Families directory not found: {families_dir}")
        return records, families, errors

    if not programs_dir.exists():
        errors.append(f"Programs directory not found: {programs_dir}")
        return records, families, errors

    yaml_files = sorted(families_dir.glob("*.yaml"))
    if not yaml_files:
        errors.append(f"No family YAML files found in {families_dir}")
        return records, families, errors

    seen_prog_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    family_referenced_programs: set[Path] = set()

    for yf in yaml_files:
        fam_id = yf.stem
        try:
            raw = _parse_yaml_file(yf)
        except Exception as e:
            errors.append(f"Failed to parse {yf.name}: {e}")
            continue

        # Check required fields
        required_fields = [
            "family_idea",
            "label",
            "label_rationale",
            "tag",
            "expected_parse",
            "programs",
            "assumptions",
        ]
        missing = [f for f in required_fields if f not in raw]
        if missing:
            errors.append(f"Family {fam_id} missing required fields: {missing}")
            continue

        label = str(raw["label"])
        if label not in ALLOWED_LABELS:
            errors.append(f"Family {fam_id} invalid label '{label}' not in vocabulary")

        tag = str(raw["tag"])
        if tag not in ALLOWED_TAGS:
            errors.append(f"Family {fam_id} invalid tag '{tag}', must be 'easy' or 'rule_hard'")

        expected_parse = str(raw["expected_parse"])
        if expected_parse not in ALLOWED_EXPECTED_PARSE:
            errors.append(f"Family {fam_id} invalid expected_parse '{expected_parse}'")

        programs_list = raw.get("programs", [])
        if not isinstance(programs_list, list) or len(programs_list) == 0:
            errors.append(f"Family {fam_id} programs list must be non-empty")
            continue

        assumptions_list = raw.get("assumptions", [])
        annotations = raw.get("program_annotations", {})

        family_meta = FamilyMetadata(
            family_id=fam_id,
            family_idea=str(raw["family_idea"]),
            label=label,
            label_rationale=str(raw["label_rationale"]),
            tag=tag,
            expected_parse=expected_parse,
            programs=tuple(str(p) for p in programs_list),
            assumptions=(
                tuple(str(a) for a in assumptions_list)
                if isinstance(assumptions_list, list)
                else ()
            ),
            program_annotations=annotations if isinstance(annotations, dict) else {},
        )
        families.append(family_meta)

        # Assemble program records
        for pid in family_meta.programs:
            if pid in seen_prog_ids:
                errors.append(f"Duplicate program_id '{pid}' in family {fam_id}")
            seen_prog_ids.add(pid)

            cpp_path = programs_dir / fam_id / f"{pid}.cpp"
            family_referenced_programs.add(cpp_path.resolve())

            if not cpp_path.exists():
                errors.append(f"Missing program source file: {cpp_path}")
                continue

            try:
                source = cpp_path.read_text(encoding="utf-8")
            except Exception as e:
                errors.append(f"Error reading {cpp_path}: {e}")
                continue

            sha = compute_source_sha256(source)
            if sha in seen_hashes:
                errors.append(
                    f"Duplicate source hash {sha} between '{pid}' and '{seen_hashes[sha]}'"
                )
            else:
                seen_hashes[sha] = pid

            # Check origin
            p_ann = annotations.get(pid, {})
            origin = p_ann.get("origin", raw.get("origin", "unspecified"))
            if origin not in ALLOWED_ORIGINS:
                errors.append(
                    f"Program '{pid}' in family {fam_id} invalid origin '{origin}', "
                    f"must be one of {sorted(ALLOWED_ORIGINS)}"
                )

            # Check UNKNOWN annotations if applicable
            unknown_reason: str | None = None
            unknown_explanation: str | None = None
            if label == "UNKNOWN":
                reason = p_ann.get("unknown_reason")
                if not reason or reason not in ALLOWED_UNKNOWN_REASONS:
                    errors.append(
                        f"Program '{pid}' in UNKNOWN family {fam_id} "
                        f"missing valid unknown_reason (got: {reason})"
                    )
                else:
                    unknown_reason = str(reason)
                expl = p_ann.get("explanation")
                if not expl:
                    errors.append(
                        f"Program '{pid}' in UNKNOWN family {fam_id} "
                        "missing program-specific explanation"
                    )
                else:
                    unknown_explanation = str(expl)

            records.append(
                ProgramRecord(
                    program_id=pid,
                    family_id=fam_id,
                    source=source,
                    expected_label=label,
                    label_rationale=family_meta.label_rationale,
                    source_sha256=sha,
                    unknown_reason=unknown_reason,
                    unknown_explanation=unknown_explanation,
                    label_status="author_labeled",
                    verified=False,
                    origin=str(origin),
                )
            )

    # Check for extra unreferenced .cpp files
    for actual_cpp in sorted(programs_dir.glob("*/*.cpp")):
        if actual_cpp.resolve() not in family_referenced_programs:
            errors.append(f"Extra unreferenced source file: {actual_cpp}")

    return records, families, errors


# ---------------------------------------------------------------------------
# Parser Validity Checker
# ---------------------------------------------------------------------------


def check_parser_validity(records: list[ProgramRecord]) -> dict[str, Any]:
    """Test parser environment and parse validity for each dataset program.

    If parser or system headers are unavailable, returns status 'PARSER_UNAVAILABLE'.
    Does not compile or execute any program.
    """
    # Probe parser availability first with a trivial snippet
    probe_source = "int probe_validity() { return 0; }"
    try:
        probe_res = parse_source(probe_source, filename="probe.cpp")
        if probe_res.has_fatal_error or not probe_res.top_level_nodes:
            return {
                "status": "PARSER_UNAVAILABLE",
                "message": "libclang parser probe returned fatal error or no AST nodes",
                "total_tested": 0,
                "parsed_count": 0,
                "failed_count": 0,
                "per_program": {},
            }
    except ClangParserError as e:
        return {
            "status": "PARSER_UNAVAILABLE",
            "message": f"Parser dependency unavailable: {e}",
            "total_tested": 0,
            "parsed_count": 0,
            "failed_count": 0,
            "per_program": {},
        }
    except Exception as e:
        return {
            "status": "PARSER_UNAVAILABLE",
            "message": f"Unexpected parser probe error: {e}",
            "total_tested": 0,
            "parsed_count": 0,
            "failed_count": 0,
            "per_program": {},
        }

    # Parser is available; test each dataset program
    per_prog: dict[str, dict[str, Any]] = {}
    parsed_count = 0
    failed_count = 0

    for rec in records:
        try:
            # First run pre-parser validation
            validate_source(rec.source)
            parse_res = parse_source(rec.source, filename=f"{rec.program_id}.cpp")
            diag_msgs = [
                d.message
                for d in parse_res.diagnostics
                if d.severity in ("error", "fatal")
            ]
            is_valid = not parse_res.has_fatal_error and len(parse_res.top_level_nodes) > 0
            if is_valid:
                parsed_count += 1
            else:
                failed_count += 1

            per_prog[rec.program_id] = {
                "valid": is_valid,
                "has_fatal_error": parse_res.has_fatal_error,
                "diagnostics": diag_msgs,
            }
        except Exception as e:
            failed_count += 1
            per_prog[rec.program_id] = {
                "valid": False,
                "has_fatal_error": True,
                "diagnostics": [str(e)],
            }

    return {
        "status": "PARSER_AVAILABLE",
        "total_tested": len(records),
        "parsed_count": parsed_count,
        "failed_count": failed_count,
        "per_program": per_prog,
    }


# ---------------------------------------------------------------------------
# Comprehensive Dataset Report Generator
# ---------------------------------------------------------------------------


def generate_dataset_report(
    dataset_root: str | Path = "datasets",
    check_parser: bool = True,
) -> dict[str, Any]:
    """Generate comprehensive report covering schema, counts, hashes, and parser validity."""
    records, families, errors = load_dataset(dataset_root)

    # Label counts
    counts_by_label: dict[str, int] = {}
    for lbl in ALLOWED_LABELS:
        counts_by_label[lbl] = 0
    for r in records:
        counts_by_label[r.expected_label] = counts_by_label.get(r.expected_label, 0) + 1

    # Family counts
    counts_by_family: dict[str, int] = {}
    for f in families:
        counts_by_family[f.family_id] = len(f.programs)

    # UNKNOWN reasons breakdown
    unknown_reasons_breakdown: dict[str, int] = {}
    for r in records:
        if r.expected_label == "UNKNOWN" and r.unknown_reason:
            unknown_reasons_breakdown[r.unknown_reason] = (
                unknown_reasons_breakdown.get(r.unknown_reason, 0) + 1
            )

    # Hash and duplicate checks
    duplicate_program_ids: list[str] = [
        err for err in errors if "Duplicate program_id" in err
    ]
    duplicate_source_hashes: list[str] = [
        err for err in errors if "Duplicate source hash" in err
    ]

    # Parser validity check
    parser_result: dict[str, Any] = {}
    if check_parser and records:
        parser_result = check_parser_validity(records)
    else:
        parser_result = {"status": "SKIPPED", "message": "Parser check skipped or no records"}

    # Origin counts
    counts_by_origin: dict[str, int] = {}
    for orig in ALLOWED_ORIGINS:
        counts_by_origin[orig] = 0
    for r in records:
        counts_by_origin[r.origin] = counts_by_origin.get(r.origin, 0) + 1

    return {
        "total_records": len(records),
        "total_families": len(families),
        "counts_by_label": counts_by_label,
        "counts_by_family": counts_by_family,
        "counts_by_origin": counts_by_origin,
        "unknown_reasons_breakdown": unknown_reasons_breakdown,
        "metadata_schema_failures": errors,
        "duplicate_program_ids": duplicate_program_ids,
        "duplicate_source_hashes": duplicate_source_hashes,
        "parser_validity": parser_result,
    }
