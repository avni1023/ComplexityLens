"""
entry_point.py — Entry function selection helpers for ComplexityLens.

Shared between the static complexity engine and structural feature extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from complexity_lens.ir.model import FunctionIR, ProgramIR


def resolve_entry_function(
    program: ProgramIR,
    function_name: str | None = None,
) -> tuple[FunctionIR | None, str]:
    """Resolve the entry function following ComplexityLens precedence rules:

    1. If caller explicitly names a function, select that function if present.
    2. Otherwise, select 'main' when present.
    3. Otherwise, select the single function when exactly one exists.
    4. Otherwise (no functions, or multiple functions with no 'main' and no explicit name):
       return (None, rationale).

    Returns
    -------
    (selected_function, selection_rationale)
    """
    # 1. Explicit function name requested
    if function_name is not None:
        for f in program.functions:
            if f.name == function_name:
                return f, f"Explicitly selected function '{function_name}'."
        return None, f"Requested function '{function_name}' not found in program IR."

    # 2. No explicit name: check for 'main'
    for f in program.functions:
        if f.name == "main":
            return f, "Selected entry point 'main'."

    # 3. Exactly one function exists
    if len(program.functions) == 1:
        f = program.functions[0]
        return f, f"Selected sole function '{f.name}'."

    # 4. Multiple functions and no 'main'
    if len(program.functions) > 1:
        avail = [f.name for f in program.functions]
        return None, f"Multiple functions present without 'main': {avail}."

    # 0 functions
    return None, "No functions found in program IR."
