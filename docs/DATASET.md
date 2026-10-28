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

The complete benchmark encompasses **242 C++ programs across 78 families**:
- **Primary Phase 2 Benchmark:** 68 fine-grained families (`f01`–`f68`) comprising **204 hand-written programs** (`origin: hand_written`, 3 programs per family: 1 seed plus 2 structurally distinct variants). Documented in canonical [`docs/FAMILY_PLAN.md`](file:///C:/Users/Admin/OneDrive/Documents/ComplexityLens/docs/FAMILY_PLAN.md).
- **Historical Baseline Benchmark:** 10 macro-families (`fam_*`) comprising **38 programs** (`origin: unspecified`), preserved unaltered under Policy 2 of [`docs/audits/DATASET_GOVERNANCE_DECISIONS.md`](file:///C:/Users/Admin/OneDrive/Documents/ComplexityLens/docs/audits/DATASET_GOVERNANCE_DECISIONS.md).

Directory layout:
- `datasets/families/<family_id>.yaml`
- `datasets/programs/<family_id>/<program_id>.cpp`

### Benchmark Distribution (Day 8 Finalization)

| Complexity Class | Token | Programs | Percentage | Primary Phase 2 Families |
|---|---|---|---|---|
| Constant | `O1` | 21 | 8.7% | `f01`–`f06` + baseline |
| Logarithmic | `OLOGN` | 31 | 12.8% | `f07`–`f15` + baseline |
| Linear | `ON` | 56 | 23.1% | `f16`–`f23`, `f53`–`f55`, `f57`–`f58` + baseline |
| Linearithmic | `ONLOGN` | 25 | 10.3% | `f24`–`f29` + baseline |
| Quadratic | `ON2` | 44 | 18.2% | `f30`–`f41` + baseline |
| Cubic | `ON3` | 12 | 5.0% | `f42`–`f45` |
| Exponential | `OEXP` | 17 | 7.0% | `f46`–`f48` + baseline |
| Unknown / Abstain | `UNKNOWN` | 36 | 14.9% | `f59`–`f68` + baseline |

### UNKNOWN Reasons Breakdown

- `data_dependent`: 8 programs (dynamic array bounds, pointer chasing, values governing iterations)
- `external_call`: 9 programs (opaque library routines, declared-only prototypes, `rand()`)
- `irregular_recursion`: 3 programs (data-governed recurrence split points)
- `termination`: 7 programs (Collatz sequences, continuous floating-point epsilon convergence)
- `unsupported_construct`: 9 programs (inline `__asm__`, lambda comparator, function pointers, `goto` loops)

---

## 5. Schema Specifications

### Family Metadata YAML Schema (`datasets/families/<family_id>.yaml`)
- `family_idea`: (string) High-level algorithmic pattern.
- `label`: (string) One of the 8 vocabulary labels (`O1`, `OLOGN`, `ON`, `ONLOGN`, `ON2`, `ON3`, `OEXP`, `UNKNOWN`).
- `label_rationale`: (string) Hand-derived proof or reasoning.
- `tag`: (string) Exactly `easy` or `rule_hard`.
- `expected_parse`: (string) Exactly `valid` or `has_unsupported_constructs`.
- `origin`: (optional string) Program provenance; one of `hand_written`, `mutation`, or `unspecified` (defaults to `unspecified` if omitted for legacy families).
- `programs`: (list of strings) Program IDs belonging to this family.
- `assumptions`: (list of strings) Stated operational assumptions (e.g. `HASH_AVERAGE_CASE`, `SINGLE_INPUT_SYMBOL`).
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
- `origin`: (string) Provenance classification (`hand_written`, `mutation`, or `unspecified`).
- `source_sha256`: (string) 64-character hex SHA-256 digest under normalized policy.
