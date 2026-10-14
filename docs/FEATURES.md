# ComplexityLens Feature Specification (Day 6)

## 1. Overview and Schema Version

```python
FEATURE_SCHEMA_VERSION = "1.0"
```

ComplexityLens extracts a deterministic, structural 26-feature vector from the Intermediate Representation (`ProgramIR`) of a C++ translation unit. The feature vector is extracted strictly using program structure, independent of static complexity labels, ML predictions, confidence scores, or ground truth.

Every feature value is a non-negative integer (`int >= 0`). Flags are strictly binary (`0` or `1`).

---

## 2. Authoritative 26-Feature Schema

| # | Exact Feature Name | Definition | Value Semantics | Reason for Inclusion |
|---:|---|---|---|---|
| 1 | `n_loops` | Number of all loops: `for`, range-`for`, `while`, `do-while`. | Non-negative integer (`>= 0`) | Core measure of iteration volume indicating potential non-constant runtime. |
| 2 | `max_loop_depth` | Maximum nesting depth, including depth contributed through non-recursive user calls: loop depth at call site + callee maximum depth. | Non-negative integer (`>= 0`) | Primary structural indicator of polynomial degree (e.g., degree $k$ for $k$-deep nested loops). |
| 3 | `n_sequential_top_loops` | Number of loops at nesting depth 1 within their own function. | Non-negative integer (`>= 0`) | Distinguishes additive sequential stages ($O(n) + O(n) = O(n)$) from multiplicative nesting. |
| 4 | `n_nested_loop_pairs` | Number of ancestor–descendant loop pairs within a function. A three-deep chain contributes 3. | Non-negative integer (`>= 0`) | Measures coupling intensity and super-linear iteration structures across nested loop hierarchies. |
| 5 | `n_for` | Number of classic `for` and range-`for` loops. | Non-negative integer (`>= 0`) | Captures structured index-based or range-based iterations commonly bounded by container or input size. |
| 6 | `n_while` | Number of `while` and `do-while` loops. | Non-negative integer (`>= 0`) | Captures condition-driven loops that frequently correspond to logarithmic reductions or search bounds. |
| 7 | `n_loops_linear_bound` | Number of loops with a unit/constant-step progression to an input-size symbol or container size. | Non-negative integer (`>= 0`) | Direct structural indicator of $O(n)$ iteration scaling with problem size. |
| 8 | `n_loops_const_bound` | Number of loops with a literal constant bound. | Non-negative integer (`>= 0`) | Identifies unrolled or fixed-capacity loops that contribute $O(1)$ asymptotic work. |
| 9 | `n_loops_log_update` | Number of loops with doubling or halving updates. | Non-negative integer (`>= 0`) | Direct structural indicator of logarithmic iteration behavior ($O(\log n)$). |
| 10 | `n_dependent_bound_loops` | Number of loops whose bound depends on an enclosing loop variable. | Non-negative integer (`>= 0`) | Identifies triangular loop patterns (such as $j < i$) characteristic of $O(n^2)$ comparison algorithms. |
| 11 | `n_loops_unknown_bound` | Number of loops whose bound kind is unresolved or unsupported. | Non-negative integer (`>= 0`) | Quantifies analytical uncertainty and triggers static or hybrid abstention policies. |
| 12 | `has_recursion` | `1` if direct or mutual recursion exists among reachable functions; otherwise `0`. | Binary integer (`0` or `1`) | Flags recursive control flow which requires recurrence relation analysis rather than loop counting. |
| 13 | `n_recursive_call_sites` | Number of recursive call sites. Count source call sites, not runtime executions. | Non-negative integer (`>= 0`) | Differentiates linear divide-and-conquer (single call) from branching tree recursion (multiple calls). |
| 14 | `shrink_minus_const` | Number of recursive call sites with a size argument of the form `p-c`. | Non-negative integer (`>= 0`) | Indicates decrementing recursion templates ($T(n) = T(n-1) + W$), yielding linear or exponential complexity. |
| 15 | `shrink_divide` | Number of recursive call sites with a size argument of the form `p/b`. | Non-negative integer (`>= 0`) | Indicates dividing recursion templates ($T(n) = aT(n/b) + W$), characteristic of divide-and-conquer ($O(\log n)$ or $O(n \log n)$). |
| 16 | `shrink_other` | Number of other recursive call sites. | Non-negative integer (`>= 0`) | Identifies complex, variable, or mutual recursive transitions that fall outside standard templates. |
| 17 | `n_user_calls` | Number of call sites to non-recursive user functions. | Non-negative integer (`>= 0`) | Measures interprocedural composition complexity and non-inlined sub-computations. |
| 18 | `n_sort_calls` | Number of `std::sort` and `std::stable_sort` call sites. | Non-negative integer (`>= 0`) | Direct structural trigger for $O(n \log n)$ comparison-sort cost components. |
| 19 | `n_linear_search_calls` | Number of `std::find`, `std::count`, and `std::find_if` call sites. | Non-negative integer (`>= 0`) | Identifies hidden linear scans through container ranges that multiply enclosing loop costs. |
| 20 | `n_binary_search_calls` | Number of `std::binary_search`, `std::lower_bound`, `std::upper_bound`, and member `lower_bound`/`upper_bound` call sites. | Non-negative integer (`>= 0`) | Identifies logarithmic search primitives on sorted ranges ($O(\log n)$ on random-access sequences). |
| 21 | `n_hash_ops` | Number of operations on `unordered_map` and `unordered_set`. | Non-negative integer (`>= 0`) | Distinguishes hash-table constant expected cost from logarithmic tree-based lookups. |
| 22 | `n_ordered_container_ops` | Number of operations on `map` and `set`. | Non-negative integer (`>= 0`) | Distinguishes balanced BST operations ($O(\log n)$ per operation) from hash operations. |
| 23 | `n_container_ops_in_loops` | Number of `push_back`, `emplace_back`, `insert`, and `erase` call sites inside any loop. | Non-negative integer (`>= 0`) | Detects dynamic accumulation or linear element shifts inside iterative loops. |
| 24 | `has_early_exit_in_loop` | `1` if a `break` or `return` occurs inside a loop; otherwise `0`. | Binary integer (`0` or `1`) | Indicates potential best-case divergence and distinguishes upper bounds from exact asymptotic classes. |
| 25 | `n_input_dependent_ops` | Number of STL-operation or user-call nodes whose size argument references an input symbol or container `.size()`. | Non-negative integer (`>= 0`) | Verifies that algorithm operations scale with primary problem size rather than fixed bounds. |
| 26 | `n_unsupported_constructs` | Number of `UnsupportedIR` nodes in reachable functions. | Non-negative integer (`>= 0`) | Quantifies opaque language features (assembly, pointer aliasing, syntax errors) requiring model abstention. |

---

## 3. Aggregation and Traversal Semantics

1. **Entry-Function Selection**:
   - Follows the identical 4-stage precedence rule as the static engine:
     1. If the caller explicitly specifies a function name, select that function.
     2. Otherwise, select `main` if present.
     3. Otherwise, select the single function if exactly one exists.
     4. If multiple functions exist without `main`, or 0 functions exist: there is no single entry function, so aggregate over **all functions**.
2. **Reachability**:
   - Starting from the selected entry function, compute the set of reachable user functions using directed reachability over the call graph.
   - Count each reachable function once for function-level aggregations, irrespective of runtime invocation counts.
   - If there is no entry function, aggregate over all functions in `program.functions`.
3. **Loop Hierarchy & Nesting Pairs**:
   - `n_sequential_top_loops`: Count of loops whose nesting depth within their defining function is `1`.
   - `n_nested_loop_pairs`: For each loop $L$ in a reachable function, count all descendant loops $D$ contained within $L$'s body (at any depth). An ancestor-descendant pair $(L, D)$ is counted once per function. A 3-deep chain ($L_1 \supset L_2 \supset L_3$) contributes 3 pairs: $(L_1, L_2)$, $(L_1, L_3)$, and $(L_2, L_3)$.
4. **Interprocedural Maximum Loop Depth (`max_loop_depth`)**:
   - For a single function $F$, its intrinsic loop depth is the maximum loop depth among loops defined in $F$ (or 0 if $F$ has no loops).
   - If a call to a non-recursive user function $G$ occurs at loop nesting depth $d$ inside $F$, it contributes a candidate depth of $d + \text{max\_depth}(G)$.
   - **Deterministic Policy for Recursive Cycles**: Recursive functions (self-recursive or member of a multi-function strongly-connected component) do NOT contribute interprocedural loop depth to their callers. Cycles in the call graph terminate traversal immediately with depth contribution 0, preventing infinite recursion.
5. **Recursion Aggregation**:
   - `has_recursion`: Set to `1` if any reachable function is self-recursive or part of a strongly connected component of size $> 1$; otherwise `0`.
   - `n_recursive_call_sites`: Sum of source call sites targeting recursive functions.
   - Recursive call sites are partitioned into `shrink_minus_const` ($p - c$), `shrink_divide` ($p / b$ or $p \gg b$), and `shrink_other`.
6. **Container Operations in Loops**:
   - Count qualifying operations (`push_back`, `emplace_back`, `insert`, `erase`) located inside at least one loop. A call site nested inside multiple loops is counted exactly once.
7. **Missing Constructs and Incomplete IR**:
   - Absent constructs yield natural defaults (`0`).
   - Incomplete IR (e.g. `analysis_complete == False`) produces a valid feature vector rather than raising an exception. Completeness metadata remains strictly separate from feature vectors.

---

## 4. Shared Structural Module Architecture

To avoid circular dependencies and preserve modular isolation:
- The extractor imports low-level structural helpers:
  - `src/complexity_lens/ir/entry_point.py`: Shared entry-function selection.
  - `src/complexity_lens/ir/loop_classification.py`: Shared per-loop structural bound-kind classification.
  - `src/complexity_lens/ir/container_classification.py`: Container kind classification.
  - `src/complexity_lens/complexity/call_graph.py`: Call graph and Tarjan SCC reachability.
- The extractor **never** imports `ComplexityEngine`, `analyze_source`, `analyze_file`, or static prediction classes (`ComplexityClass`, `Exactness`, `AnalysisResult`).
- The extractor has **zero** ML dependencies (no scikit-learn, numpy, pandas, torch, xgboost).

---

## 5. Human-Readable Example and Reproduction

### Sample Program (`tests/fixtures/cpp/04_nested_loops.cpp`)

```cpp
// Fixture 04: Nested for loops (O(n^2) pattern)
// Expected: 2x ForStmt (outer/inner nesting detectable by depth)
int matrix_sum(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            total += i * j;
        }
    }
    return total;
}
```

### Extracted Feature Vector

Ordered tuple of 26 non-negative integer values (`FEATURE_SCHEMA_VERSION = "1.0"`):

```text
(2, 2, 1, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
```

### Readable `to_dict()` Output

```json
{
  "n_loops": 2,
  "max_loop_depth": 2,
  "n_sequential_top_loops": 1,
  "n_nested_loop_pairs": 1,
  "n_for": 2,
  "n_while": 0,
  "n_loops_linear_bound": 2,
  "n_loops_const_bound": 0,
  "n_loops_log_update": 0,
  "n_dependent_bound_loops": 0,
  "n_loops_unknown_bound": 0,
  "has_recursion": 0,
  "n_recursive_call_sites": 0,
  "shrink_minus_const": 0,
  "shrink_divide": 0,
  "shrink_other": 0,
  "n_user_calls": 0,
  "n_sort_calls": 0,
  "n_linear_search_calls": 0,
  "n_binary_search_calls": 0,
  "n_hash_ops": 0,
  "n_ordered_container_ops": 0,
  "n_container_ops_in_loops": 0,
  "has_early_exit_in_loop": 0,
  "n_input_dependent_ops": 0,
  "n_unsupported_constructs": 0
}
```

### Reproduction Command

```bash
python -c "
import json
from pathlib import Path
from complexity_lens.ir.builder import build_ir_from_source
from complexity_lens.features import extract_features

code = Path('tests/fixtures/cpp/04_nested_loops.cpp').read_text(encoding='utf-8')
program = build_ir_from_source(code)
fv = extract_features(program)
print(json.dumps(fv.to_dict(), indent=2))
"
```
