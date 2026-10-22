# Dataset Review Checklist

This checklist tracks human review of hand-derived complexity labels and rationales across all dataset families.

> [!NOTE]
> All reviewer initials and review dates below remain intentionally blank pending formal human sign-off. Automated audit findings and diagnostic comparisons must not be conflated with human verification.

| Family ID | Expected label | Rationale reviewed | Reviewer initials | Review date |
|---|---|---|---|---|
| fam_constant_ops | O1 | [ ] | | |
| fam_binary_search | OLOGN | [ ] | | |
| fam_linear_scan | ON | [ ] | | |
| fam_quadratic_nested | ON2 | [ ] | | |
| fam_rule_hard_divide_conquer | ONLOGN | [ ] | | |
| fam_rule_hard_linear | ON | [ ] | | |
| fam_rule_hard_nested | ON2 | [ ] | | |
| fam_exponential_recurrence | OEXP | [ ] | | |
| fam_unknown_data_dependent | UNKNOWN | [ ] | | |
| fam_unknown_external_unsupported | UNKNOWN | [ ] | | |

---

## Week 1 Ground-Truth Review and Disagreement Audit

### 1. Benchmark Baseline & Rule-Hard Count Reconciliation

- **Total Records:** 38 programs across 10 families.
- **Rule-Hard Family Distribution:**
  - `fam_rule_hard_divide_conquer`: 4 programs (`tag: rule_hard`)
  - `fam_rule_hard_linear`: 4 programs (`tag: rule_hard`)
  - `fam_rule_hard_nested`: 4 programs (`tag: rule_hard`)
  - `fam_unknown_data_dependent`: 3 programs (`tag: rule_hard`)
  - `fam_unknown_external_unsupported`: 3 programs (`tag: rule_hard`)
  - **Total Rule-Hard Programs:** **18 / 38 (47.4%)**
- **Reconciliation Note:** The metadata and `docs/DATASET.md` correctly establish 18 / 38 (47.4%) rule-hard programs, easily satisfying the benchmark requirement ($\ge 33\%$). A previous agent response mistakenly mentioned "16/38" in prose due to summing only the four-program families ($4 \times 4 = 16$), overlooking the two three-program UNKNOWN rule-hard families. The actual dataset files and validation code have always contained 18 rule-hard programs.
- **Agreement Diagnostic Baseline:**
  - **Evaluated:** 38
  - **Agreements (Matches):** 24 (63.2%)
  - **Disagreements:** 14 (36.8%)
  - **Integrity Rule:** `WARNING: Re-check the rationale; never edit a label to match the engine`

---

### 2. Comprehensive Review of the 14 Disagreements

Each disagreement was independently analyzed against the C++ source, family metadata, documented cost model (`HASH_AVERAGE_CASE`), IR/feature extraction evidence, and static analyzer rule traces.

| Program ID | Expected Label | Analyzer Result | Independent Complexity Derivation | Current Label & Rationale Correct? | Cause of Disagreement | Recommended Next Action |
|---|---|---|---|---|---|---|
| `binsearch_std_lower_bound` | `OLOGN` | `UNKNOWN` | `std::lower_bound` on random-access vector executes logarithmic comparisons $\lceil \log_2(n) \rceil + O(1) = O(\log n)$. | Yes, correct. | Analyzer limitation (`UNSUPPORTED_ITERATOR`; iterator range not resolved to random-access vector size). | Preserve ground-truth label `OLOGN`. Enhance iterator range classification in analyzer. |
| `binsearch_range_halving` | `OLOGN` | `UNKNOWN` | Classic iterative binary search over $[0, n-1]$; interval $(high - low)$ halves each step, bounded by $\lfloor \log_2(n) \rfloor + 1 = O(\log n)$. | Yes, correct. | Analyzer limitation (`LOOP_VAR_MODIFIED`, `UNKNOWN_BOUND`; while-loop classifier does not track two-pointer interval halving). | Preserve ground-truth label `OLOGN`. Add interval convergence pattern to loop classifier. |
| `exp_subset_sum_recursive` | `OEXP` | `UNKNOWN` | Worst-case branching evaluates both branches $T(n) = 2T(n-1) + O(1)$, generating recursion tree of depth $n$ with $2^n$ leaves: $\Theta(2^n) = O(2^n)$. | Yes, correct. | Analyzer limitation (`BASE_CASE_MISSING`; recursion rule fails on dual guard returns `target==0` / `n==0` combined with short-circuit `\|\|`). | Preserve ground-truth label `OEXP`. Expand recursion base-case parser for multi-guard returns. |
| `linear_std_find` | `ON` | `UNKNOWN` | Linear search over vector of size $n$ inspects up to $n$ elements: $O(n)$ worst-case comparisons. | Yes, correct. | Analyzer limitation (`UNSUPPORTED_RANGE`; range `vec.begin()` to `vec.end()` not bound to container size symbol). | Preserve ground-truth label `ON`. Map full-range container iterators to container size in `RULE_STL_FIND`. |
| `linear_vector_count` | `ON` | `UNKNOWN` | `std::count` over full vector range unconditionally traverses all $n$ elements: $\Theta(n) = O(n)$ steps. | Yes, correct. | Analyzer limitation (`UNSUPPORTED_RANGE`; `vec.begin()`..`vec.end()` not mapped to size symbol $n$). | Preserve ground-truth label `ON`. Enable container-range mapping in `RULE_STL_COUNT`. |
| `nested_bubble_pass` | `ON2` | `UNKNOWN` | Nested loops with outer bound $n$ and inner bound $n - 1$ execute $n(n - 1) = n^2 - n = \Theta(n^2) = O(n^2)$ iterations. | Yes, correct. | Analyzer limitation (`UNKNOWN_BOUND`; loop condition `j < n - 1` with binary arithmetic not recognized as linear bound). | Preserve ground-truth label `ON2`. Support shifted bounds `n - c` in loop condition normalizer. |
| `mergesort_range_recurse` | `ONLOGN` | `UNKNOWN` | Divide-and-conquer recurrence $T(n) = 2T(n/2) + O(n)$ on subarray $[l, r]$ resolves to $\Theta(n \log n) = O(n \log n)$ via Master theorem. | Yes, correct. | Analyzer limitation (`BASE_CASE_MISSING`; range recursion with condition `if (l < r)` not recognized by single-variable recurrence template). | Preserve ground-truth label `ONLOGN`. Add divide-and-conquer range recurrence template. |
| `mergesort_index_split` | `ONLOGN` | `UNKNOWN` | Half-open index partitioning $[start, end)$ divides into two subproblems of size $n/2$ plus $O(n)$ combine: $T(n) = O(n \log n)$. | Yes, correct. | Analyzer limitation (`BASE_CASE_MISSING`; recurrence on parameter difference $(end - start)$ not matched to recurrence rules). | Preserve ground-truth label `ONLOGN`. Add parameter-difference recurrence recognition. |
| `mergesort_subrange_vector` | `ONLOGN` | `UNKNOWN` | Subrange halving with linear merge pass: $T(n) = 2T(n/2) + O(n) = O(n \log n)$ worst-case time complexity. | Yes, correct. | Analyzer limitation (`RANGE_RECURSION_UNSUPPORTED`; explicit rule abstains on range-based recursion). | Preserve ground-truth label `ONLOGN`. Implement Master Theorem solver for range recursion. |
| `mergesort_halving_linear_merge` | `ONLOGN` | `UNKNOWN` | Classic binary divide-and-conquer mergesort on $[lo, hi]$ with linear merge pass: $T(n) = 2T(n/2) + O(n) = O(n \log n)$. | Yes, correct. | Analyzer limitation (`RANGE_RECURSION_UNSUPPORTED`; explicit abstention rule on interval divide-and-conquer). | Preserve ground-truth label `ONLOGN`. Implement Master Theorem solver for range recursion. |
| `hard_while_constant_step` | `ON` | `UNKNOWN` | Loop initialized at 0 incrementing by `const int step = 2` until $n$ executes $\lceil n/2 \rceil = \Theta(n) = O(n)$ steps. | Yes, correct. | Analyzer limitation (`UNKNOWN_BOUND`; while loop analyzer does not propagate local constant `const int step = 2` into loop increment). | Preserve ground-truth label `ON`. Add local constant propagation to while-loop step detector. |
| `hard_helper_loop_in_loop` | `ON2` | `UNKNOWN` | Outer loop runs $n$ times; each iteration invokes `helper_scan(vec, n)` which executes $n$ iterations: $n \times n = O(n^2)$. | Yes, correct. | Analyzer limitation (`ARGUMENT_MAPPING_UNSUPPORTED`; inter-procedural argument substitution fails on local variable `int n = vec.size()`). | Preserve ground-truth label `ON2`. Support local variable aliases in argument substitution. |
| `hard_stl_find_in_loop` | `ON2` | `ON` | Outer loop runs $|a| = n$ times; in each iteration, `std::find(b.begin(), b.end(), a[i])` scans up to $|b| = m$ elements. Under worst-case (no matches), work is $n \times m = O(n^2)$. | Yes, correct. | **Analyzer defect / unsoundness** (`LINEAR_LOOP` + `IF_ELSE_WORST_CASE: O(1)`; engine treated `std::find` in `if` condition as $O(1)$, underestimating quadratic complexity as linear). | **Preserve ground-truth label `ON2`.** Log priority engine bug: condition expressions in `if` statements containing linear STL calls must not be evaluated as $O(1)$. |
| `hard_stl_count_in_loop` | `ON2` | `UNKNOWN` | Outer loop runs $n$ times; inner call `std::count` traverses $m$ elements: $n \times m = O(n^2)$ worst-case time complexity. | Yes, correct. | Analyzer limitation (`UNSUPPORTED_RANGE`; inner `std::count` returned `UNKNOWN`, making `RULE_NESTED_LOOP` return `UNKNOWN`). | Preserve ground-truth label `ON2`. Fix container range support in `std::count` to compose quadratically. |

---

### 3. Critical Diagnostic Findings

1. **Zero Ground-Truth Label Defects:**
   All 14 hand-derived labels and rationales are mathematically sound under worst-case time complexity and documented assumptions. No labels require correction or modification.
2. **Engine False Negative / Underestimation:**
   `hard_stl_find_in_loop` exposes an analyzer unsoundness bug: the static analysis engine evaluated `std::find(b.begin(), b.end(), a[i])` located inside an `if (...)` condition as $O(1)$ constant work instead of $O(n)$, causing an $O(n^2)$ algorithm to be incorrectly classified as $O(n)$.
3. **Engine Abstentions:**
   The remaining 13 disagreements are conservative analyzer abstentions (`UNKNOWN`), primarily attributable to:
   - STL iterator ranges (`vec.begin()`..`vec.end()`) not mapped to container size symbols (4 programs).
   - Divide-and-conquer range-based recursion and Master theorem templates (4 programs).
   - Interval-based two-pointer while loops (1 program).
   - Multi-guard base-case recursion patterns (1 program).
   - Shifted bounds (`n - 1`) in for-loops (1 program).
   - Local constant propagation and local variable argument mapping (2 programs).

---

### 4. Post-Fix Agreement Diagnostic & STL Range Soundness Results

Following the Day 7 STL argument extraction fixes, branch condition item composition, and STL range sound classification:

- **Baseline Comparison:**
  - **Original Audit Baseline:** 24 matches / 14 disagreements (63.2% agreement).
  - **Post-Fix Agreement:** **29 matches / 9 disagreements (76.3% agreement)**.

- **Reconciled Disagreements (5 programs):**
  1. `linear_std_find`: Resolved from `UNKNOWN` to `ON` (MATCH). Full vector iterator range correctly maps to container size.
  2. `linear_vector_count`: Resolved from `UNKNOWN` to `ON` (MATCH). `std::count` iterator range correctly maps to linear container traversal.
  3. `hard_stl_find_in_loop`: Resolved from the unsound `ON` underestimation to `ON2` (MATCH). Branch condition expressions (`if (std::find(...) != ...)`) are now fully traversed as active operations and correctly compose quadratically within input-bounded outer loops.
  4. `hard_stl_count_in_loop`: Resolved from `UNKNOWN` to `ON2` (MATCH). Inner linear `std::count` soundly composes with outer loop to yield quadratic time complexity.
  5. `nested_bubble_pass`: Resolved from `UNKNOWN` to `ON2` (MATCH). Local variable alias (`int n = arr.size()`) and shifted bound (`n - 1`) soundly recognized under input-dependent variable binding.

- **Remaining 9 Disagreements:**
  All 9 remaining disagreements are verified conservative analyzer abstentions (`UNKNOWN`). No ground-truth labels require adjustment.

- **STL Range Constant Soundness Audit:**
  - **Fixed literal and local constants (`arr + 5`, `const int length = 5;`):** Soundly classified as $O(1)$ (`CONSTANT_WORK`) when the range offset resolves to an integer literal or proven non-reassigned local constant. When invoked inside input-bounded loops, these compose soundly as $O(n) \times O(1) = O(n)$.
  - **Input-dependent local constants (`const int length = n;`):** Accurately classified as $O(n)$ (`STL_FIND`) or conservative `UNKNOWN` if untracked. They are **never** classified as $O(1)$.
  - **Input-dependent mutable locals (`int length = n;`):** Resolved to the input dependency ($O(n)$) or conservative `UNKNOWN`. Never classified as $O(1)$.
  - **Compile-time named constants (`#define N ...`, top-level `const int N = ...`):** Correctly classified as input-size symbols under project conventions (`NAMED_CONSTANT_AS_INPUT_SIZE`), yielding input-dependent bounds ($O(n)$, $O(n \log n)$, $O(\log n)$).
  - **Alias Invalidation Audit:** Re-assignments (`=`), compound assignments (`+=`, `-=`, `*=`, etc.), increments/decrements (`++`, `--`), and stream extractions (`cin >>`) strictly increment LHS mutation counts. Any mutated variable is immediately purged from both `local_constants` and `local_named_constants`, guaranteeing that mutated identifiers conservatively yield `UNKNOWN` (`UNSUPPORTED_RANGE`) rather than retaining stale constant or input-dependent assumptions. Reads strictly on assignment RHS do not cause false invalidations.

> [!CAUTION]
> Reaching 29/38 agreement on the benchmark suite does **not** prove that zero soundness defects exist in the static analysis engine. Engine generalizations must continue to be audited against complex multi-parameter interactions and non-standard syntax patterns.
