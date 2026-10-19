# Dataset Documentation and Integrity Protocol

## 1. Ground Truth Integrity and Labels-Only Commit

To establish an independent ground truth for the ComplexityLens benchmark, all dataset programs, family metadata, expected labels, and hand-derived rationales were authored and committed in a strictly separated **labels-only commit** (Phase A) prior to implementing dataset evaluation tooling and agreement diagnostics (Phase B).

- **Phase A Labels-Only Commit Hash:** `f23df075370cbadd5ac7dc7efbd254214fa0552f`
- **Provenance Note:** Because a Git commit's hash is computed over the tree containing all tracked file contents, a commit cannot contain its own final hash in a tracked file. The exact Phase A commit hash is therefore recorded in subsequent documentation updates.
- **Integrity Rule:** Re-check the rationale; never edit a label to match the engine. If a rationale or schema defect is discovered, it must be documented transparently in a subsequent commit rather than rewriting historical ground truth.

---

## 2. Label Conventions and Cost Model

- **Complexity Measure:** All labels represent **worst-case time complexity**.
- **Input-Size Symbol:** A single input-size symbol `n` is used. Named compile-time constants, including `#define N ...` and `const int N = ...`, are treated as input-size symbols when they represent the problem's input size.
- **Hash Table Cost Model (`HASH_AVERAGE_CASE`):** Unordered-container operations (`std::unordered_map`, `std::unordered_set`) have expected $O(1)$ cost under the default `HASH_AVERAGE_CASE` assumption. This assumption is recorded explicitly. Worst-case hash-table guarantees ($O(n)$ bucket collisions) are not silently substituted for the specified expected-cost model.
- **Closed Vocabulary:** Labels are strictly limited to the following 8 tokens:
  - `O1`: Constant time $O(1)$
  - `OLOGN`: Logarithmic time $O(\log n)$
  - `ON`: Linear time $O(n)$
  - `ONLOGN`: Linearithmic time $O(n \log n)$
  - `ON2`: Quadratic time $O(n^2)$
  - `ON3`: Cubic time $O(n^3)$
  - `OEXP`: Exponential time $O(2^n)$
  - `UNKNOWN`: Complexity cannot be derived statically within the supported model
- **Allowed UNKNOWN Reasons:** For programs labeled `UNKNOWN`, exactly one primary reason must be specified from this closed vocabulary:
  - `data_dependent`: Loop trip count or recursion depth depends on runtime data values or contents.
  - `termination`: Non-obvious termination properties (e.g. Collatz 3n+1 sequences) without known closed-form bounds.
  - `unsupported_construct`: Contains a syntactically valid C++ construct deliberately unsupported by the static IR (e.g. inline assembly `__asm__`).
  - `external_call`: Calls an external or opaque function whose source implementation and complexity are unavailable.
  - `irregular_recursion`: Recurrence relation branches irregularly (e.g. alternating subtractive and divisive steps) not fitting standard recurrence patterns.

---

## 3. Source Hashing Policy

Each program record computes a deterministic `source_sha256` digest over its C++ source text according to this exact normalization policy:

1. **Line Ending Normalization:** Replace all Windows CRLF (`\r\n`) line endings with Unix LF (`\n`).
2. **Trailing Whitespace Trimming:** Trim all trailing whitespace from the end of the entire source string (`source.rstrip()`). Trailing whitespace on individual intermediate lines is preserved.
3. **UTF-8 Encoding & SHA-256:** Compute the SHA-256 hex digest of the normalized UTF-8 byte stream.

An optional comment-stripped hash may be computed for diagnostics only using the validator's lexical comment scanner; it never replaces `source_sha256`.

---

## 4. Dataset Family Definitions and Structure

The initial benchmark comprises **38 C++ programs across 10 families** (3–4 programs per family). At least one third of the programs (18/38, or 47.4%) belong to rule-hard families.

Directory layout:
- `datasets/families/<family_id>.yaml`
- `datasets/programs/<family_id>/<program_id>.cpp`

### Family Summary Table

| Family ID | Label | Tag | Expected Parse | Programs Count | Core Algorithmic Idea |
|---|---|---|---|---|---|
| `fam_constant_ops` | `O1` | `easy` | `valid` | 4 | Straight-line primitive arithmetic, vector lookups, bounded loops. |
| `fam_binary_search` | `OLOGN` | `easy` | `valid` | 4 | Logarithmic domain reduction via division by 2 or `std::lower_bound`. |
| `fam_linear_scan` | `ON` | `easy` | `valid` | 4 | Single linear loop iterating up to $n$ or `std::find`/`std::count`. |
| `fam_quadratic_nested` | `ON2` | `easy` | `valid` | 4 | Doubly-nested loops iterating up to $n$ (matrix / triangular pairs). |
| `fam_rule_hard_divide_conquer` | `ONLOGN` | `rule_hard` | `valid` | 4 | Merge-sort divide-and-conquer recursion over ranges with linear merge. |
| `fam_rule_hard_linear` | `ON` | `rule_hard` | `valid` | 4 | Linear loops with intermediate bounds (`n/2`), constant step variables, early exit. |
| `fam_rule_hard_nested` | `ON2` | `rule_hard` | `valid` | 4 | Helper loops called inside loops, `std::find`/`std::count` called inside loops. |
| `fam_exponential_recurrence` | `OEXP` | `easy` | `valid` | 4 | Multi-branching recursive calls on decremented arguments (Fibonacci / subset-sum). |
| `fam_unknown_data_dependent` | `UNKNOWN` | `rule_hard` | `valid` | 3 | Collatz sequence, data-dependent array jumps, irregular recursion. |
| `fam_unknown_external_unsupported` | `UNKNOWN` | `rule_hard` | `has_unsupported_constructs` | 3 | External opaque function call, inline `__asm__`, Euclidean GCD. |

Total: **10 families, 38 programs**.

---

## 5. Schema Specifications

### Family Metadata YAML Schema (`datasets/families/<family_id>.yaml`)
- `family_idea`: (string) High-level algorithmic pattern.
- `label`: (string) One of the 8 vocabulary labels.
- `label_rationale`: (string) Hand-derived proof or reasoning.
- `tag`: (string) Exactly `easy` or `rule_hard`.
- `expected_parse`: (string) Exactly `valid` or `has_unsupported_constructs`.
- `programs`: (list of strings) Program IDs belonging to this family.
- `assumptions`: (list of strings) Stated operational assumptions (e.g. `HASH_AVERAGE_CASE`).
- `program_annotations`: (optional mapping) Program-specific overrides, including `unknown_reason` and `explanation`.

### Assembled Program Record Schema
- `program_id`: (string) Unique program identifier.
- `family_id`: (string) Family identifier.
- `source`: (string) Complete C++ source code text.
- `expected_label`: (string) Independent ground truth label.
- `label_rationale`: (string) Hand-derived justification.
- `unknown_reason`: (string or null) Specific allowed UNKNOWN reason identifier where applicable.
- `label_status`: (string) Exactly `author_labeled`.
- `verified`: (boolean) `false` until peer-reviewed.
- `source_sha256`: (string) 64-character hex SHA-256 digest under normalized policy.
