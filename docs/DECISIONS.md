# Project Decisions & Architecture Records (ADR)

## ADR-001: Minimal Python Project Structure & Tooling (Day 0)
* **Date**: 2026-10-11
* **Status**: Accepted
* **Context**: ComplexityLens requires a deterministic, lightweight Python development environment without extraneous frameworks or unnecessary overhead.
* **Decision**:
  - Use `pyproject.toml` with `setuptools` build backend.
  - Standardize source code under `src/complexity_lens/` to avoid top-level import ambiguity.
  - Use `pytest` for testing and `ruff` for linting.
  - Provide both `Makefile` (cross-platform / CI) and `run_checks.ps1` (native Windows PowerShell).
  - Scope out all deep learning, databases, Docker, and academic paper tools.
* **Consequences**: Fast startup, zero global pip pollution, fully reproducible local development.

---

## ADR-002: C++ Parser — libclang Feasibility Spike (Day 1)

* **Date**: 2026-10-12
* **Status**: Accepted (with one documented limitation)

### Context

ComplexityLens must recognise C++ constructs (loops, calls, recursion, STL, malformed
code, inline assembly) and extract source locations for a downstream complexity analyser.
Day 1 was a time-boxed spike to prove or disprove that libclang can serve as the parser
backbone before any IR or execution work is attempted.

### Environment Inspection (commands run)

```powershell
# No system LLVM found:
where clang          # empty
where clang++        # empty
where llvm-config    # empty
Get-ChildItem "C:\Program Files\LLVM" -ErrorAction SilentlyContinue   # empty
Get-ChildItem "C:\" -Recurse -Filter "libclang.dll" -Depth 6          # empty

# Python venv state before changes:
.venv\Scripts\python.exe -m pip list
# → no clang / llvm / tree-sitter packages present

# libclang available on PyPI as a self-contained wheel:
.venv\Scripts\python.exe -m pip install libclang==18.1.1
# → Downloaded libclang-18.1.1-py2.py3-none-win_amd64.whl (26.4 MB)
# → Successfully installed libclang-18.1.1
# The wheel bundles libclang.dll under site-packages/clang/native/
# — no separate LLVM installation required.

# Smoke-test the binding:
.venv\Scripts\python.exe -c "import clang.cindex as ci; ci.Index.create(); print('OK')"
# → OK
```

### Decision

Use **`libclang==18.1.1`** (PyPI wheel, bundled DLL) as the primary C++ parser for
ComplexityLens. Binding is via the official `clang.cindex` Python module.

* Parser module: `src/complexity_lens/parsers/cpp_libclang.py`
* Public API: `parse_file(path)` / `parse_source(source)` → `ParseResult`
* `ParseResult` exposes: `functions()`, `for_stmts()`, `while_stmts()`,
  `call_exprs()`, `asm_stmts()`, `max_loop_nesting_depth()`, `is_recursive()`.

### Fixtures Created (12 total)

| # | File | Construct | AST Nodes Verified |
|---|------|-----------|--------------------|
| 01 | `01_simple_function.cpp` | Function | `FUNCTION_DECL`, `PARM_DECL`, `RETURN_STMT` |
| 02 | `02_for_loop.cpp` | For loop | `FOR_STMT`, depth=1 |
| 03 | `03_while_loop.cpp` | While loop | `WHILE_STMT`, no `FOR_STMT` |
| 04 | `04_nested_loops.cpp` | Nested for (O(n²)) | 2× `FOR_STMT` at distinct depths, depth≥2 |
| 05 | `05_sequential_loops.cpp` | Sequential for (O(n)) | 2× `FOR_STMT` at same depth, depth=1 |
| 06 | `06_function_call.cpp` | Cross-function call | `CALL_EXPR` spelling = "helper" |
| 07 | `07_recursion.cpp` | Recursive fibonacci | `CALL_EXPR` to same `FUNCTION_DECL`, ≥2 self-calls |
| 08 | `08_stl_sort.cpp` | `std::sort` | SKIPPED (see failure below) |
| 09 | `09_stl_search.cpp` | `std::find` | SKIPPED (see failure below) |
| 10 | `10_malformed.cpp` | Missing brace/semi | Diagnostics with line:col, partial AST |
| 11 | `11_unsupported_asm.cpp` | Inline `__asm__` | `ASM_STMT` detected with location |
| 12 | `12_triple_nested.cpp` | Triple nested (O(n³)) | 3× `FOR_STMT` at 3 distinct depths, depth≥3 |

### Test Results

```
pytest tests/test_cpp_libclang_spike.py -v   (Python 3.13.14 / pytest 9.1.1)

52 passed, 7 skipped, 0 warnings in 0.82s
```

All non-STL fixtures: **100 % pass**.  
STL fixtures (08, 09): **skipped** — see failure section.

### Known Failure: STL Headers Not Found

**Symptom**: `fatal error: 'algorithm' file not found` / `'vector' file not found`

**Root cause**: The `libclang==18.1.1` PyPI wheel ships only `libclang.dll`; it does
**not** bundle the MSVC or GCC C++ standard-library headers (`<algorithm>`, `<vector>`,
etc.).  When libclang tries to resolve `#include <algorithm>`, it finds no system
include path because no LLVM or MSVC toolchain is installed globally.

**Impact assessment**: LOW for the spike scope.  All non-STL constructs (loops, calls,
recursion, malformed code, inline ASM) parse correctly.  For production use, the analyser
can:
  1. Install LLVM for Windows from https://releases.llvm.org/ (adds system headers).
  2. Pass `-isystem "C:\path\to\llvm\lib\clang\18\include"` via `extra_args`.
  3. Fall back to tree-sitter (see below) for header-dependent files.

**Workaround (optional)**:
```powershell
# Install LLVM 18 then point libclang at its DLL + headers:
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin\libclang.dll"
# The libclang wheel's _DEFAULT_ARGS already pass -std=c++17 -fms-extensions.
# Add -isystem to pick up LLVM's bundled headers:
parse_file("08_stl_sort.cpp", extra_args=["-isystem", r"C:\Program Files\LLVM\lib\clang\18\include"])
```

### Planned Tree-Sitter Fallback

If libclang proves unworkable for a given target (e.g., heavy STL or CUDA code with no
system toolchain), the fallback is:

* **Package**: `tree-sitter==0.21.*` + `tree-sitter-cpp` grammar
* **Advantages**: pure-Python, no native headers required, grammar can be updated
  independently of LLVM releases, lower cold-start overhead.
* **Disadvantages**: grammar-level AST (no type resolution), no semantic CallExpr
  targets, harder to distinguish `std::sort` from any other `sort` call.
* **Decision gate**: if STL header tests cannot pass within one more day via LLVM
  installer, switch to tree-sitter for the full IR prototype and keep libclang only for
  environments with a working toolchain.

### Consequences

* `libclang==18.1.1` added to project dependencies (see `pyproject.toml` update needed).
* `LIBCLANG_PATH` env var honoured for custom LLVM installs.
* STL-dependent tests are skipped automatically (not failed) to keep CI green.
* Source locations (file, line, col) confirmed accurate on all 10 passing fixture
  classes — sufficient for Day 2 complexity feature extraction.

---

### Addendum: STL Parser Compatibility Check Findings

* **Date**: 2026-10-12
* **Objective**: Evaluate whether existing local Windows headers can resolve `<vector>` and `<algorithm>` under `libclang==18.1.1`.
* **Findings**:
  1. **Wheel Limitations**: The PyPI `libclang==18.1.1` wheel provides `libclang.dll` only. It omits both C++ standard-library headers (`<vector>`, `<algorithm>`) and Clang's internal resource headers (`lib/clang/18/include/`).
  2. **System Toolchain Absence**: No modern MSVC toolchain, Windows SDK UCRT includes, or system LLVM toolchain are installed on this machine.
  3. **MinGW GCC 4.9.2 Incompatibility**: A legacy Dev-Cpp MinGW installation (`C:\Program Files (x86)\Dev-Cpp\MinGW64`) exists. However, pointing libclang at its include tree triggers fatal compiler intrinsic errors: GCC 4.9.2 headers rely on GCC-private builtins (`__builtin_ia32_bsrsi`, `__builtin_ia32_addss`, etc.) inside `ia32intrin.h`/`xmmintrin.h` which Clang 18 does not implement.
  4. **Safety & Robustness**: Masking compiler intrinsics via stub headers is fragile, non-portable, and masks legitimate parser diagnostics.
* **Chosen Path**: **libclang with limited support** (standalone C++ constructs) for the immediate spike, maintaining **tree-sitter fallback** as the strategic path for zero-dependency AST extraction.
* **Test Verification**: 52 passed, 7 skipped (all 7 STL tests cleanly skipped; zero failures; `run_checks.ps1` exit code 0).

---

### Addendum: Outcome 1 Resolution — MSYS2 UCRT64 & CXX_INCLUDE_DIRS Verification

* **Date**: 2026-10-12
* **Objective**: Resolve STL header parsing via verified MSYS2 MinGW-w64 UCRT64 headers.
* **Environment Verification**:
  - MSYS2 UCRT64 headers verified at:
    - `C:\msys64\ucrt64\include\c++\16.2.0`
    - `C:\msys64\ucrt64\include\c++\16.2.0\x86_64-w64-mingw32`
    - `C:\msys64\ucrt64\include`
    - `C:\msys64\ucrt64\lib\gcc\x86_64-w64-mingw32\16.2.0\include`
    - `C:\msys64\ucrt64\lib\gcc\x86_64-w64-mingw32\16.2.0\include-fixed`
  - Required headers verified: `<vector>`, `<algorithm>`, `<unordered_map>`.
* **Parser Adapter Adaptation**:
  - `src/complexity_lens/parsers/cpp_libclang.py` inspects `CXX_INCLUDE_DIRS` (splitting on `os.pathsep`).
  - When MinGW/UCRT paths are detected, `-target x86_64-w64-mingw32` is aligned and MSVC compatibility macros are omitted, eliminating Windows/MSVC CRT declaration mismatches in `stdlib.h`.
* **Actual Test Results**:
  - Focused spike test suite: **59 passed, 0 skipped, 0 failed** in ~17s.
  - Complete test suite & linter (`.\run_checks.ps1`): **61 passed, 0 skipped, 0 failed** in ~21s.
  - All 7 previously skipped STL tests (`TestSTLSort`, `TestSTLSearch`) pass cleanly.
  - AST correctly extracts `std::sort`, `std::find`, `std::lower_bound`, `vector<vector<int>>`, and `unordered_map<int, int>` with precise source locations.
* **Remaining Limitations**:
  - GCC 16 `stl_iterator.h` emits a non-fatal diagnostic on a friend template trailing return type (`member access into incomplete type`), which Clang 18 tolerates while successfully generating the complete AST.
  - Standalone/zero-header environments remain protected: if `CXX_INCLUDE_DIRS` is unset, the parser gracefully skips STL tests and preserves all non-STL test passes (54 passed, 7 skipped).

---

## ADR-003: Minimal Typed Intermediate Representation (IR) (Day 2)

* **Date**: 2026-10-12
* **Status**: Accepted

### Context

ComplexityLens requires a deterministic, lightweight Intermediate Representation (IR) that
downstream complexity rules (Days 4–5) and feature extractors (Day 6) can consume.
The IR must:
1. Decouple parsing (`libclang` AST) from analysis data structures.
2. Formally represent functions, loop nesting hierarchies, bounds, step updates, calls,
   recursion sites, recognized STL operations, exact source spans, and unsupported constructs.
3. Make unknown, complex, or unparsable syntax explicit (`UNKNOWN`, `UNKNOWN_SYNTAX`)
   rather than silently guessing.
4. Not implement any complexity evaluation rules yet (scoped to Day 4).

### Decision

Implement the typed IR under package `src/complexity_lens/ir/`:
- **`model.py`**: Pure, parser-agnostic dataclasses and enumerations:
  - `SourceSpan`: `(file, start_line, start_col, end_line, end_col)`
  - `BoundKind`: `CONSTANT`, `LINEAR_INPUT`, `DEPENDENT` (triangular loop), `UNKNOWN`
  - `UpdateKind`: `ADDITIVE` (e.g. `++i`, `+= c`), `MULTIPLICATIVE` (e.g. `*= c`, `/= c`), `UNKNOWN`
  - `LoopKind`: `FOR`, `WHILE`, `DO_WHILE`, `RANGE_FOR`
  - `STLOperationKind`: `SORT`, `FIND`, `COUNT`, `BINARY_SEARCH`, `LOWER_BOUND`, `PUSH_BACK`, `INSERT`, `ERASE`, `HASH_LOOKUP`, `ORDERED_LOOKUP`
  - `UnsupportedKind`: `ASM_STMT`, `UNKNOWN_SYNTAX`, `COMPLEX_POINTER`, `OPAQUE_CONSTRUCT`
  - IR Node classes: `ProgramIR`, `FunctionIR`, `LoopIR`, `CallIR`, `STLOperationIR`, `UnsupportedIR`, `LoopBound`, `LoopUpdate`.
- **`builder.py`**: `build_ir(parse_result: ParseResult) -> ProgramIR`:
  - Consumes the parser's `ParseResult` and translates user functions (excluding standard headers).
  - Determines loop depth, loop variable, bound classification, update classification, and early exit (`has_early_exit` for `break`/`return`).
  - Detects self-calls as recursion with argument shrink analysis (`minus_const`, `divide`, `other`).
  - Maps `std::sort`, `std::find`, `push_back`, etc. into typed `STLOperationIR` nodes with target container resolution.
  - Catches inline assembly (`__asm__`) and parser syntax errors explicitly as `UnsupportedIR`.
  - Provides serialization (`to_dict()`) and structured human-readable views (`pretty_print()`).
- **`tests/test_ir_builder.py`**: Golden tests covering all 12 fixtures and edge cases.

### Consequences

- Strict separation of concerns achieved: downstream static complexity engine and ML feature extractor only interact with `ProgramIR`.
- All 12 baseline fixtures produce valid, strongly typed IR with exact source spans.
- Total test suite expanded to **84 tests**:
  - `.\run_checks.ps1`: **84 passed, 0 skipped, 0 failed** in ~71s.
- `ruff` checks 100% clean.

---

## ADR-004: Complete IR, Input Validation, and Directive Safety (Day 3)

* **Date**: 2026-10-13
* **Status**: Accepted

### Context

ComplexityLens must accept user-submitted C++ source safely before it is ever
forwarded to the libclang parser.  Two classes of risk must be mitigated:

1. **Malicious or pathological input**: oversized files, unbounded nesting,
   embedded NUL bytes, non-text types, or invalid UTF-8 can crash the parser or
   produce misleading IR silently.
2. **Dangerous preprocessor directives**: `#pragma`, `#include_next`, `#embed`,
   `__has_include`, quoted includes, or non-whitelisted headers could pull in
   unexpected code or expose system state.

The IR also lacked structured diagnostics; `ProgramIR.diagnostics` was `list[str]`
and there was no `analysis_complete` flag, making downstream consumers unable to
distinguish "clean IR" from "IR with suppressed errors".

### Decisions

#### Input Validation (`src/complexity_lens/validation.py`)

A single `validate_source(source)` function runs **before any parser call**.
Rules enforced:

| Rule | Limit |
|------|-------|
| Type | `str` only; `bytes` / other types rejected |
| Non-empty | Empty or whitespace-only source rejected |
| NUL chars | `\0` always rejected |
| UTF-8 | Upstream decode must use `errors="strict"` |
| Source size | ≤ 20,480 UTF-8 bytes |
| Line count | ≤ 200 (inclusive) |
| Line length | ≤ 1,000 chars (excl. terminator) |
| Brace depth | ≤ 50 (braces in comments/strings/chars ignored) |

The brace-depth scanner uses a character-level state machine — not a regex —
to correctly ignore braces inside line comments, block comments, string
literals, and character literals.

No input is silently truncated.  `ValidationError` is raised on any violation,
with a 1-based `line` attribute when applicable.

#### Include / Directive Safety

A documented whitelist (`ALLOWED_HEADERS`) of standard C++ header names
(bare names only, no paths) is enforced at the directive level.

Rejected:
- `#include "…"` (quoted includes)
- `#include <path/to/header.h>` (path-style)
- `#include_next`, `#embed`, `#pragma`
- `__has_include` anywhere in source text

Behaviour on rejection: `ValidationError` is raised; the parser is never
called.  Source-line mapping is preserved for compliant input.

The whitelist is checked line-by-line before parsing and is compatible with
all Day-1/Day-2 fixtures.

#### Structured IR Diagnostics (`IRDiagnostic`)

`ProgramIR.diagnostics` is now `list[IRDiagnostic]` instead of `list[str]`.
Each `IRDiagnostic` carries:
- `severity: str` — `"note" | "warning" | "error" | "fatal"`
- `message: str`
- `span: SourceSpan | None`
- `.is_error() -> bool` — True for `"error"` or `"fatal"` severity

#### `analysis_complete` Flag

`ProgramIR.analysis_complete: bool` is introduced.

- `True` — no known completeness failures during IR construction.
- Set to `False` by any error/fatal diagnostic **or** any `UnsupportedIR` node.
- Warning-only diagnostics do **not** set it to `False`.
- Fatal parse failures never produce a misleading empty-looking IR.
- Serialization (`to_dict()`) and `pretty_print()` always include this field.

`analysis_complete = True` does **not** guarantee mathematical exactness of
future complexity predictions.

#### Container-Type Provenance (`container_kind_source`)

`STLOperationIR.container_kind_source: str | None` records inference method:

- `"source_text"` — inferred from token/spelling; best-effort, Day 3 default.
- `"resolved"` — reserved for future semantic resolution; not yet set.
- `None` — container could not be determined.

Source-text inferences must never be presented as resolved semantic facts.

#### Parser Isolation

All parser calls pass through `parse_file` / `parse_source` in
`complexity_lens.parsers.cpp_libclang`.  No other module calls libclang
directly.  libclang runs **in-process** — no subprocess is spawned.
Submitted C++ is **never compiled or executed**.  Process isolation and
timeouts are planned for the API milestone.

#### DO_WHILE and RANGE_FOR

The IR builder now handles `do { … } while (cond)` and `for (auto x : c) { … }`
loop kinds in `_build_body_items`.  Range-for bounds are classified `UNKNOWN`
because the range expression is not a traditional bound.

### Consequences

- All validation failures raise before the parser is invoked (tested with mocks).
- Directive whitelist is compatible with all 12 Day-1/Day-2 fixture files.
- Downstream consumers can rely on `analysis_complete` and structured
  `IRDiagnostic` objects.
- Container-type provenance prevents future code from treating best-effort
  inferences as resolved semantic facts.

