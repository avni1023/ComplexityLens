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

---

## 6. Dataset Partitioning and Split Protocol (Day 9 Stage 2B)

### 6.1 Two-Partition Model

The ComplexityLens benchmark strictly adheres to a **two-partition model**:
- **`development`**: 53 families (159 programs in Phase 2 inventory).
- **`test`**: 15 families (45 programs in Phase 2 inventory).

There is **NO** separate `train` partition. All cross-validation folds exist strictly inside the `development` partition.

### 6.2 Cross-Validation Folds inside Development

Cross-validation inside `development` uses **3 family-stratified folds** of deterministic sizes:
- Fold 0: 18 families (54 programs)
- Fold 1: 18 families (54 programs)
- Fold 2: 17 families (51 programs)

Each fold contains:
- Exactly 1 `ON3` family
- Exactly 1 `OEXP` family
- Complete coverage of all 8 complexity classes (`O1`, `OLOGN`, `ON`, `ONLOGN`, `ON2`, `ON3`, `OEXP`, `UNKNOWN`) under Case A feasibility.

#### Cross-Validation Allocation Outcomes:
- **Case A (Optimal Stratification):** Exact fold capacities `(18, 18, 17)` are satisfied, and all 8 complexity classes are present in every fold.
- **Case B (Stratification Deficit Fallback):** Exact fold capacities `(18, 18, 17)` and `ON3`/`OEXP` seeds are satisfied, but full 8-class coverage across all folds is mathematically impossible or unmet. Emits `STRATIFICATION_DEFICIT_WARNING` reporting per-fold missing classes, per-fold class counts, and `total_coverage_deficit` (total missing fold/class combinations). The reported deficit is the observed deficit of the deterministic greedy/backtracking allocator, not a proven minimal deficit.
- **Case C (Cardinality Infeasibility):** Exact fold capacities `(18, 18, 17)` cannot be satisfied under indivisible atomic unit constraints. Raises `CardinalityInfeasibilityError`.

#### Atomic-Unit CV Counting and Indivisibility Rules
When assigning complete atomic units (singletons or multi-family split groups) to development folds:
1. `ON3` and `OEXP` counts are evaluated strictly at the **family level** across the entire unit (`u.classes.count("ON3")` and `u.classes.count("OEXP")`).
2. Atomic units remain completely indivisible. Any atomic unit containing $>1$ `ON3` families or $>1$ `OEXP` families immediately raises `InfeasibleSplitConstraintsError`, as assigning it to any single fold would violate the exact 1-per-fold requirement.
3. A joint atomic unit containing 1 `ON3` and 1 `OEXP` family simultaneously fulfills both requirements for a single fold, leaving the remaining pure `ON3` and pure `OEXP` units to pair into the other two folds.
4. Fold capacity enforcement strictly requires (18, 18, 17) families, using greedy assignment with deterministic symmetry breaking ($\text{rank\_sum}(\text{fold}_0) \le \text{rank\_sum}(\text{fold}_1)$) and exact backtracking bin-packing fallback.

### 6.3 Hard Constraints for Test-Set Optimization

1. Exactly 15 families in `test`.
2. All 8 complexity classes present in `test`.
3. Exactly 1 `ON3` family in `test` (leaving 3 for development, 1 per fold).
4. Exactly 1 `OEXP` family in `test` (leaving 3 for development, 1 per fold).
5. Exactly 3 `UNKNOWN` families in `test` (leaving 12 for development, 4 per fold).
6. Family `f65` is locked to `development` (`f65` is UNKNOWN / data_dependent / rule_hard).
7. At most 3 `rule_hard` families in `test` (which forces exactly 3, matching the 3 UNKNOWN families).
8. Family atomicity: No family or approved atomic group is divided across partitions or folds.

### 6.4 Lexicographic Objectives

The test partition is chosen by exact branch-and-bound optimization with admissible 0-1 min-cost subset-sum dynamic programming:
1. Maximize distinct UNKNOWN reasons in `test` (optimal: 3 distinct reasons).
2. Minimize `abs(test_rule_hard_count - 3)` (optimal: 0).
3. Minimize total unit rank sum: $\sum_{u \in T} \text{rank}(u)$, where $\text{rank}(u) = \text{int}(\text{SHA256}(f"{seed}:{unit\_key}")[:16], 16)$.
4. Tie-break via canonical sorted family-ID tuple.

### 6.5 Token Normalization Vocabulary (37 Tokens)

To detect potential lexical leakage between development and test partitions without relying on variable names, lexical normalization uses a closed vocabulary of **37 tokens**:
- **29 C++ Language Keywords:** `auto`, `bool`, `break`, `case`, `char`, `class`, `const`, `continue`, `default`, `do`, `double`, `else`, `false`, `float`, `for`, `goto`, `if`, `int`, `long`, `nullptr`, `return`, `short`, `sizeof`, `static`, `struct`, `switch`, `true`, `void`, `while`.
- **8 Standard Library Identifiers:** `std`, `vector`, `string`, `size_t`, `unordered_map`, `unordered_set`, `set`, `map`.

Any identifier not in this vocabulary is replaced with `ID`. 3-token shingles are computed, and cross-partition set Jaccard similarity must not exceed 0.85.

### 6.6 Dual Hashing Policy

Program records in the split manifest contain dual hashes for verification:
- `source_sha256`: LF-normalized source text hash (used for semantic comparisons and dataset integrity).
- `git_blob_sha256`: Git blob hash at HEAD (used for repository commit integrity tracking).

### 6.7 Split Lifecycle and Freeze Gate

1. **`propose`**: Proposes a deterministic split manifest without modifying files. Records `base_dataset_commit` (current HEAD) and sets `status = "proposed"`.
2. **`validate`**: Validates manifest structure, cardinalities, coverage, and leakage constraints against the dataset on disk.
3. **`freeze`**: Freezes a validated proposal manifest subject to strict hardening gates:
   - **Frozen-File Overwrite Protection:** Pre-flight checks ensure that neither the destination frozen manifest JSON nor its `.sha256` sidecar exists on disk. If either file exists, execution halts immediately with `FreezeGateError` before any writes or computational steps. Existing files are preserved byte-identically, and no partial files are left behind.
   - **Clean Tree Requirement:** Working tree must be completely clean (`git status --porcelain` is empty).
   - **Git Ancestry Gate:** The `base_dataset_commit` must exist in the repository (`git cat-file -e`). Current HEAD must descend from or match `base_dataset_commit` (`git merge-base --is-ancestor <base> HEAD`).
   - **Dataset/Governance Mutex Gate:** No modifications to `datasets/`, `docs/FAMILY_PLAN.md`, or `docs/audits/` are permitted between `base_dataset_commit` and HEAD (`git diff --name-only <base> HEAD`).
   - **Committed Proposal Byte-Identity:** If the proposal JSON is committed in HEAD, its content must match the proposal being frozen.
   - **Sidecar Checksum:** Upon passing all gates, records `proposal_commit = HEAD`, sets `status = "frozen"`, and writes canonical frozen JSON with a companion SHA-256 sidecar file (`<manifest>.sha256`).
4. **`verify`**: Verifies that a frozen manifest matches its sidecar checksum and satisfies all dataset integrity rules.

### 6.8 CLI Usage

```bash
# Propose a split and print summary
complexity-lens dataset split propose

# Validate a split manifest
complexity-lens dataset split validate data/splits/split_manifest_v1.0.proposed.json

# Freeze a proposal manifest (requires clean git working tree)
complexity-lens dataset split freeze data/splits/split_manifest_v1.0.proposed.json --out data/splits/split_manifest_v1.0.frozen.json

# Verify a frozen manifest and checksum sidecar
complexity-lens dataset split verify data/splits/split_manifest_v1.0.frozen.json
```
