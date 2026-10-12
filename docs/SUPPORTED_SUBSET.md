# Supported C++ Subset — ComplexityLens

*Last updated: Day 3 milestone*

This document describes what C++ constructs ComplexityLens can analyse, what
it explicitly rejects, and what it silently represents as unknown or
unsupported in the IR.

> [!IMPORTANT]
> ComplexityLens never compiles or executes submitted C++ programs.  All
> analysis is purely static, operating on libclang's AST.

---

## Input Requirements

All submitted source must:
- Be a **single C++ translation unit** (`.cpp`-style; no `.h`-only files).
- Contain no more than **200 lines** and **20,480 UTF-8 bytes**.
- Have no line exceeding **1,000 characters** (excluding the line terminator).
- Have brace nesting no deeper than **50 levels** (braces inside comments /
  string / char literals are not counted).
- Be valid `str` Python objects decoded from strict UTF-8.
- Contain no NUL characters (`\0`).

---

## Preprocessing / Include Directives

Only `#include <header>` directives whose header name appears on the documented
**whitelist** are accepted.  All other directives are rejected at validation
time, before the parser is invoked.

**Explicitly rejected:**
- `#include "…"` — quoted includes
- `#include <path/to/header.h>` — path-style system includes
- `#include_next` — not permitted
- `#embed` — not permitted
- `#pragma` — any pragma
- `__has_include` — not permitted in source expressions

**Permitted standard headers (representative list):**
`array`, `algorithm`, `deque`, `functional`, `iterator`, `list`, `map`,
`numeric`, `queue`, `set`, `stack`, `string`, `string_view`, `tuple`,
`unordered_map`, `unordered_set`, `utility`, `variant`, `vector`, `optional`,
`memory`, `chrono`, `type_traits`, `limits`, `climits`, `cstddef`, `cstdint`,
`cassert`, `cmath`, `cstring`, `cstdlib`, `cstdio`, `cerrno`, `iostream`,
`istream`, `ostream`, `sstream`, `fstream`, `execution`.

The full whitelist is defined in `src/complexity_lens/validation.py` as
`ALLOWED_HEADERS`.  The whitelist is compatible with all Day-1/Day-2 fixtures.

---

## Supported Constructs

### Functions

- **Top-level function definitions** with any return type and named parameters.
- Source span (file, start/end line, start/end column) is extracted for every
  function.
- Multiple functions per translation unit are supported.
- Forward declarations (no body) are excluded from the IR.

**Limitations:**
- Function templates are not instantiated by the IR builder; they appear only
  if libclang instantiates them during the parse.
- Methods inside class / struct bodies are treated as unsupported (they are
  not function-definition top-level nodes in the libclang output for the
  current filter).

---

### Loops

| Kind | Construct | Supported |
|------|-----------|-----------|
| `FOR` | `for (init; cond; update) { … }` | ✅ |
| `WHILE` | `while (cond) { … }` | ✅ |
| `DO_WHILE` | `do { … } while (cond);` | ✅ |
| `RANGE_FOR` | `for (auto x : container) { … }` | ✅ (body only; bound = UNKNOWN) |

Each loop node carries:
- **`kind`** — `LoopKind` enum value.
- **`depth`** — nesting depth (1 = outermost).
- **`bound`** — classified as `CONSTANT`, `LINEAR_INPUT`, `DEPENDENT`, or
  `UNKNOWN`.  See [Loop Bound Classification](#loop-bound-classification).
- **`update`** — classified as `ADDITIVE`, `MULTIPLICATIVE`, or `UNKNOWN`.
- **`has_early_exit`** — `True` if the loop body contains a `break` or
  `return` not inside a nested loop.
- **`span`** — source span of the entire loop statement.

#### Loop Bound Classification

| `BoundKind` | Detected when | Example |
|-------------|---------------|---------|
| `CONSTANT` | Bound expression is an integer literal | `i < 100` |
| `LINEAR_INPUT` | Bound is a parameter name or `size()`/`length()` call | `i < n`, `i < v.size()` |
| `DEPENDENT` | Bound is an outer loop variable | `j < i` |
| `UNKNOWN` | None of the above patterns match | `i < f()` |

**Limitation:** the bound classifier uses token-level analysis, not full
semantic analysis.  Complex expressions such as `i < (n - 1) / 2` will be
classified as `UNKNOWN`.

#### Loop Update Classification

| `UpdateKind` | Detected patterns |
|--------------|------------------|
| `ADDITIVE` | `++i`, `i++`, `--i`, `i--`, `+= c`, `-= c` |
| `MULTIPLICATIVE` | `*= c`, `/= c`, `>>= 1`, `<<= 1`, `i * 2` |
| `UNKNOWN` | None of the above recognized |

---

### Nested and Sequential Loops

- Nested loops are represented hierarchically: each `LoopIR.body` can contain
  further `LoopIR` nodes.
- Sequential loops at the same depth appear as siblings in the parent's
  `body` list.
- Maximum nesting is limited only by the brace-depth validator (50 levels).

---

### Function Calls

- Calls to user-defined functions are captured as `CallIR` nodes with the
  callee name, argument list, and source span.
- **Recursion detection:** if a function calls itself (directly), it is marked
  `is_recursive = True` and the call sites are recorded in
  `recursive_call_sites`.
- **Shrink kind** for recursive calls:
  - `"minus_const"` — argument decremented by a constant (`n - 1`, `n--`)
  - `"divide"` — argument halved / right-shifted (`n / 2`, `n >> 1`)
  - `"other"` — recursion detected but shrink pattern not recognized
- Trivial iterator calls (`.begin()`, `.end()`, `operator*`, etc.) are
  filtered out.

**Limitation:** mutual recursion (function A calling B calling A) is not
detected; only direct self-recursion is classified.

---

### Recognized STL Operations

| `STLOperationKind` | Recognized spellings |
|--------------------|----------------------|
| `SORT` | `std::sort`, `std::stable_sort` |
| `FIND` | `std::find` |
| `COUNT` | `std::count` |
| `BINARY_SEARCH` | `std::binary_search` |
| `LOWER_BOUND` | `std::lower_bound`, `std::upper_bound` |
| `PUSH_BACK` | `.push_back()`, `.emplace_back()` |
| `INSERT` | `.insert()` |
| `ERASE` | `.erase()` |
| `HASH_LOOKUP` | (planned; not yet emitted) |
| `ORDERED_LOOKUP` | (planned; not yet emitted) |

Each `STLOperationIR` records:
- **`container`** — name of the container if determinable.
- **`container_kind_source`** — provenance of the container inference:
  - `"source_text"` — inferred from token/spelling source text (best-effort).
  - `"resolved"` — reserved for future semantic type resolution (not yet used).
  - `None` — container could not be determined.

**Limitation:** STL recognition requires that STL headers are parsed.  If
`CXX_INCLUDE_DIRS` is not set, tests that include `<vector>` or `<algorithm>`
are skipped.

---

## Explicitly Unsupported Constructs

The following constructs produce an `UnsupportedIR` node in the IR and cause
`analysis_complete = False`:

| Construct | `UnsupportedKind` | Produces diagnostic? |
|-----------|------------------|---------------------|
| Inline assembly (`__asm__`, `__asm`) | `ASM_STMT` | No (AST node detected) |
| Parser syntax errors / unrecognized syntax | `UNKNOWN_SYNTAX` | Yes (error diagnostic) |
| Complex pointer aliasing / casts | `COMPLEX_POINTER` | No (future) |
| Other opaque constructs | `OPAQUE_CONSTRUCT` | No (future) |

### Lambda expressions

Lambda expressions are not recognized by the current IR builder.  A lambda
inside a function body will typically be walked without emitting an IR node;
its interior body may contain loop/call nodes that are captured.  **Lambda
closures are not tracked.**  This is a known limitation and does not produce an
`UnsupportedIR` node in the current implementation.

### `goto`

`goto` statements are not detected by the current IR builder.  They do not
produce an `UnsupportedIR` node.  Code using `goto` is walked without error,
but the resulting IR may be misleading for complexity analysis because the
control flow is not represented.  Future versions should emit `UnsupportedKind.OPAQUE_CONSTRUCT`.

### User-defined templates

Template function *definitions* are not analyzed; only instantiations that
appear in the libclang AST are processed.  Uninstantiated templates produce no
function IR.

### Function pointers

Function pointer calls are not recognized as meaningful call sites.  A call
through a function pointer appears in the IR as `CallIR` with whatever spelling
libclang assigns (often the pointer variable name), without semantic callee
resolution.

### Virtual calls

Virtual method calls are not resolved semantically.  They appear as `CallIR`
with the method's static spelling but the dynamic dispatch target is not known.

### `try`/`catch`

Exception handling constructs (`try`, `catch`, `throw`) are not explicitly
represented in the IR.  The body of a `try` block is walked normally; exception
paths are not captured.  This does not produce an `UnsupportedIR` node.

### Inline assembly

Inline assembly (`__asm__`, `__asm volatile`) is detected at AST level and
emitted as `UnsupportedIR` with `kind = UnsupportedKind.ASM_STMT`.  This sets
`analysis_complete = False`.

---

## IR Diagnostics and Completeness

```
ProgramIR.analysis_complete : bool
```

- `True` when no completeness failures were found during IR construction.
- `False` when:
  - Any `error`- or `fatal`-severity diagnostic is present.
  - Any `UnsupportedIR` node exists at any nesting level.

Warning-only diagnostics do **not** automatically set `analysis_complete = False`.

`analysis_complete = True` does **not** guarantee that any complexity
prediction derived from the IR is mathematically exact.  It means no known IR
construction failures occurred.

---

## Source Spans

Every IR node carries a `SourceSpan`:
- `file` — absolute path of the source file (or `<string>.cpp` for in-memory
  parsing).
- `start_line`, `start_col` — 1-based start of the node.
- `end_line`, `end_col` — 1-based end of the node.

---

## Parser Isolation

All parser calls go through `complexity_lens.parsers.cpp_libclang.parse_file`
or `parse_source`.  No other module invokes libclang directly.

**libclang runs in-process** — no subprocess is spawned.  Process isolation
(e.g., sandboxing with a timeout) is planned for the API milestone and is not
implemented in Day 3.

The submitted C++ source is **never compiled or executed**.  libclang performs
AST construction only.
