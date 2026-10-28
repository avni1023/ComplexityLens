# ComplexityLens — Daily Learning Log & Viva Preparation Guide

> **Purpose of this document:**  
> A simple, easy-to-read explanation of what ComplexityLens is doing, what was built on each day, why each component is needed, and how each update improves the project. Use this document to review the codebase regularly, understand design decisions, and prepare confidently for exams and project vivas without feeling overwhelmed!

---

## High-Level Project Overview

### What is ComplexityLens?
**ComplexityLens** is an explainable static analysis system that predicts and proves the **asymptotic time complexity (Big-O)** of C++ programs (such as $O(1)$, $O(\log n)$, $O(n)$, $O(n \log n)$, $O(n^2)$, $O(n^3)$, or Exponential).

### Why do we need this? (Viva Concept)
1. **Static Analysis vs. Runtime Profiling:**  
   Runtime benchmarking (running code with timers like `std::chrono`) depends on CPU speed, RAM, cache, compiler optimizations, and requires running the code with actual inputs. If user code contains an infinite loop or security vulnerability, running it is dangerous.  
   **ComplexityLens never executes the C++ code.** It inspects the source code mathematically and structurally to derive the Big-O class.
2. **Explainability & Proofs:**  
   Instead of just giving a number, ComplexityLens gives **exact evidence nodes, source code lines, rule traces, and human-readable reasoning**.
3. **Worst-Case Guarantees:**  
   For algorithms and competitive programming, worst-case guarantees matter most. ComplexityLens models conservative worst-case semantics.

---

## Daily Work Breakdown

### Day 1 & Day 2 — Safe Input Validation & Preprocessing
* **What was built:**
  - Integrated `libclang` (the official Clang compiler library) to parse C++ AST without running code.
  - Strict input validation: maximum 200 lines, 20 KB UTF-8, no NUL characters, maximum brace depth of 50.
  - Header whitelist: only safe standard headers (e.g. `<vector>`, `<algorithm>`, `<iostream>`) are allowed; system headers like `<windows.h>` or `<sys/socket.h>` and malicious directives like `#pragma` or quoted includes `#include "..."` are rejected.
* **Why was it needed?**
  - Prevents denial-of-service (crashing the parser with a 100,000-line file) and code execution attacks.
  - Ensures deterministic behavior and isolates the parser from platform differences.

---

### Day 3 — Intermediate Representation (IR) Builder
* **What was built:**
  - Transformed Clang's raw, complex C++ AST into clean, high-level Python dataclasses:
    - `ProgramIR`: Contains translation unit metadata and all functions.
    - `FunctionIR`: Name, parameters, return type, body statements, source span.
    - `LoopIR`: Loops (`for`, `while`, `do-while`), bound kind (`CONSTANT`, `LINEAR_INPUT`, `DEPENDENT`), step update (`ADDITIVE`, `MULTIPLICATIVE`).
    - `BranchIR`: `if`/`else` conditionals.
    - `CallIR`: Function calls, recursion indicators, shrink direction (`minus_const`, `divide`).
    - `STLOperationIR`: Standard library operations (`sort`, `lower_bound`, `push_back`, etc.).
    - `UnsupportedIR`: Syntax outside our supported subset (e.g. `goto`, `asm`).
* **Why was it needed?**
  - Directly reasoning about Clang's raw AST is brittle and messy. Having a normalized, language-agnostic Intermediate Representation (IR) separates parsing from mathematical complexity analysis.

---

### Day 4 — Deterministic Static Complexity Engine
* **What was built:**
  - Implemented the deterministic complexity analysis engine using a mathematical representation:
    $$\text{Complexity} = (p, q) \iff O(n^p \cdot (\log n)^q)$$
    - $(0, 0) = O(1)$
    - $(0, 1) = O(\log n)$
    - $(1, 0) = O(n)$
    - $(1, 1) = O(n \log n)$
    - $(2, 0) = O(n^2)$
    - $(3, 0) = O(n^3)$
    - Special classes: `EXP` (Exponential) and `UNKNOWN`.
  - **Loop classification:**
    - Constant bounds (`for (int i = 0; i < 100; ++i)`) $\to O(1)$.
    - Input-size linear loops (`for (int i = 0; i < n; ++i)` or reverse `for (int i = n; i > 0; --i)`) $\to O(n)$.
    - Halving/doubling updates (`i *= 2`, `i /= 2`) $\to O(\log n)$.
  - **Composition rules:**
    - Sequential statements: take the dominant term ($\max(O(1), O(n)) = O(n)$).
    - Nested loops: multiply iteration counts ($O(n) \times O(n) = O(n^2)$).
    - Branches (`if/else`): take the worst-case (maximum of then-branch and else-branch).
* **Why was it needed?**
  - Forms the foundational core of the Big-O derivation system. Proves that straight-line code and standard bounded loops never produce exponential complexity.

---

### Day 5 — STL Containers, Function Substitution, and Recursion
Day 5 extended the engine across three major stages:

#### Stage 1: STL and Container Cost Tables (`stl_rules.py`)
* **What was built:**
  - **Configurable Hash Model:**
    - `expected_hash`: Hash table lookups/inserts (`unordered_map`, `unordered_set`) are $O(1)$ amortized, with the explicit assumption `HASH_AVERAGE_CASE`.
    - `strict_worst_case`: In the worst case (hash collisions / adversarial inputs), hash lookups degrade to $O(n)$.
  - **Ordered Containers:** `std::map` and `std::set` (balanced binary search trees) are always $O(\log n)$.
  - **Vector Operations:**
    - `push_back`/`emplace_back` are $O(1)$ amortized (`AMORTIZED` assumption).
    - `insert`/`erase` at the beginning or arbitrary position are $O(n)$ due to element shifting.
  - **Algorithms:**
    - `std::sort` on random-access ranges is $O(n \log n)$.
    - `std::find`, `std::count`, `std::accumulate` are $O(n)$.
    - `std::lower_bound`, `std::binary_search` are $O(\log n)$ **only on random-access ranges** (like `vector`, `array`, C-style arrays `a, a+n`). Calling `std::lower_bound` on a `std::set` returns `UNKNOWN` because bidirectional iterators take linear time!
* **Why was it needed?**
  - Real C++ programs use standard library containers and algorithms. Without cost models for STL, almost every modern program would be classified as `UNKNOWN`.

#### Stage 2: Safe User-Function Complexity Substitution (`call_graph.py`)
* **What was built:**
  - Built a call graph across functions in the translation unit.
  - Used **Tarjan’s Strongly Connected Components (SCC)** algorithm to detect cycles. Mutual recursion is identified and conservatively marked `UNKNOWN`.
  - Added a call depth recursion cap (`MAX_CALL_DEPTH = 20`) to prevent infinite analysis loops.
  - **Safe Argument Mapping:** Mapped caller arguments to callee parameters:
    - $n$, $n+c$, $n-c$, $c \cdot n$, $n/c$, $c\text{.size()}$, literal integers.
    - Unsupported argument expressions (e.g. $n \cdot n$ or complex arithmetic) safely return `UNKNOWN` with `ARGUMENT_MAPPING_UNSUPPORTED`.
  - Integrated substitution: sequential function calls sum their costs, calls inside loops multiply by loop trip count, and `UPPER_BOUND` exactness is preserved.
* **Why was it needed?**
  - Real code is modular and split into helper functions. Without substitution, any call to a helper function resulted in `UNKNOWN`.

#### Stage 3: Conservative Recursion Templates T1–T4 (`recursion.py`)
* **What was built:**
  - Defined four explicit templates in `docs/RECURSION_TEMPLATES.md`:
    - **T1 (One decrementing call):** $T(n) = T(n-1) + f(n)$.  
      - If $f(n) = O(1) \implies O(n)$ (e.g., Factorial).  
      - If $f(n) = O(n) \implies O(n^2)$.
    - **T2 (One dividing call):** $T(n) = T(n/b) + f(n)$ where $b \ge 2$.  
      - If $f(n) = O(1) \implies O(\log n)$ (e.g., Recursive Binary Search).  
      - If $f(n) = O(n) \implies O(n)$.  
      - Exclusive `if/else` recursive branches count as one call per path, avoiding false exponential classification.
    - **T3 (Two half-size calls + linear work):** $T(n) = 2T(n/2) + O(n) \implies O(n \log n)$ (e.g., Merge Sort recurrence).
    - **T4 (Multiple decrementing calls + $O(1)$ work):** $T(n) \ge 2T(n-1) + O(1) \implies O(\text{EXP})$ (e.g., Naive Fibonacci).
  - Explicit unsupported cases: range recursion with `(lo, hi, mid)`, recursive calls inside loops, and missing base cases return `UNKNOWN` with explicit reason codes.
* **Why was it needed?**
  - General recurrence solving is undecidable, but common competitive programming recurrences match these four patterns. This delivers deterministic, mathematically grounded results for recursive algorithms.

---

### Day 6 — Deterministic Structural Feature Extraction
* **What was built:**
  - Authoritative 26-feature schema in `docs/FEATURES.md` and `src/complexity_lens/features/`:
    - **11 Loop features:** `n_loops`, `max_loop_depth` (with interprocedural propagation through non-recursive user calls), `n_sequential_top_loops`, `n_nested_loop_pairs`, `n_for`, `n_while`, `n_loops_linear_bound`, `n_loops_const_bound`, `n_loops_log_update`, `n_dependent_bound_loops`, `n_loops_unknown_bound`.
    - **5 Recursion features:** `has_recursion`, `n_recursive_call_sites`, `shrink_minus_const`, `shrink_divide`, `shrink_other`.
    - **6 Calls and Operations features:** `n_user_calls`, `n_sort_calls`, `n_linear_search_calls`, `n_binary_search_calls`, `n_hash_ops`, `n_ordered_container_ops`.
    - **4 Other features:** `n_container_ops_in_loops`, `has_early_exit_in_loop`, `n_input_dependent_ops`, `n_unsupported_constructs`.
  - Canonical immutable `FeatureVector` with schema version `"1.0"`, preserving canonical order across `to_list()` and `to_dict()`.
  - Decoupled structural helpers: `loop_classification.py`, `entry_point.py`, and `container_classification.py` isolate structural inspection from engine predictions.
  - Leakage prevention: labels, family IDs, sample IDs, filenames, and split metadata are forbidden and excluded from the schema.
  - Zero ML library dependencies.
* **Why was it needed? (Viva Concept)**
  - Prepares the tabular representation needed by machine learning baselines (Days 10–12) and hybrid gates (Day 13).
  - Isolating feature extraction from engine predictions prevents data leakage and ensures features remain objective mathematical and structural measurements of the program AST.

---

### Day 7 — Dataset Tooling, Integrity, and Week 1 Gate
* **What was built:**
  - Independent ground truth architecture: historical Phase A labels-only commit (`f23df075...`) established gold labels and hand-derived rationales before implementing evaluation tooling.
  - Strict YAML schema and normalized `source_sha256` hashing policy in `docs/DATASET.md`.
  - Core dataset module in `src/complexity_lens/dataset.py` with dataset loader, schema validator, duplicate detection, and AST parser validity checks.
  - CLI commands `complexity-lens dataset report` and `complexity-lens dataset check-agreement` with machine-readable JSON exports.
  - Baseline agreement audit: 63.2% (24/38) agreement with static engine, with detailed justification for all 14 engine abstentions documented in `docs/DATASET_REVIEW.md`.
  - Integrity rule established: *never edit a gold label to match the engine*.
* **Why was it needed? (Viva Concept)**
  - Prevents evaluation circularity: benchmarks must have independently authored ground truth. Disagreements between static predictions and gold labels expose analyzer limitations, not label errors.

---

### Day 8 — Benchmark Expansion, Origin Tracking, and Governance Audit
* **What was built:**
  - Expanded dataset from 10 baseline families (38 programs) to 78 families (242 programs) across 8 complexity classes (`O1`, `OLOGN`, `ON`, `ONLOGN`, `ON2`, `ON3`, `OEXP`, `UNKNOWN`).
  - Implemented 68 Phase 2 families (`f01`–`f68`) comprising 204 hand-written programs (seed + 2 structurally distinct variants per family).
  - Added program provenance tracking (`origin: hand_written`, `mutation`, or `unspecified`) across schemas, data models, validators, and CLI reporting.
  - Conducted comprehensive Phase 2 dataset audits across Groups A through E, completing all 10 UNKNOWN families (`f59`–`f68`) across 5 audit batches.
  - Formally recorded and approved dataset governance policies in `docs/audits/DATASET_GOVERNANCE_DECISIONS.md`: canonical family plan designation, historical baseline preservation, and split isolation policy.
  - Full CLI dataset report verified: 0 schema errors, 0 duplicate IDs or hashes, and 242/242 programs successfully parsed.
* **Why was it needed? (Viva Concept)**
  - Provides a balanced, diverse, fine-grained benchmark necessary for training and evaluating machine learning baselines (Days 10–12) while preserving the legacy benchmark for out-of-distribution regression testing.
  - Origin tracking differentiates original hand-written code from future synthetic mutations, ensuring rigorous provenance control.

---

## Quick Viva Questions & Answers

**Q1: Why does ComplexityLens perform static analysis instead of just running the program and measuring time?**  
*Answer:* Dynamic timing depends on hardware specs, background tasks, and specific test cases. Crucially, executing untrusted code could trigger infinite loops or execute malicious operations. Static analysis analyzes code safely and derives asymptotic growth independent of machine hardware.

**Q2: What is the difference between `STRUCTURAL` and `UPPER_BOUND` exactness?**  
*Answer:* `STRUCTURAL` means the engine proved both upper and lower bounds structurally (e.g., a loop guaranteed to run $n$ times). `UPPER_BOUND` is used when early exits (`break`, `return`) or conservative assumptions (like vector position) mean the execution could take fewer steps, but will never exceed this bound.

**Q3: Why does `std::lower_bound` on a `std::set` return `UNKNOWN` instead of $O(\log n)$?**  
*Answer:* The member function `s.lower_bound(k)` runs in $O(\log n)$ using the tree's internal pointers. However, the standalone algorithm `std::lower_bound(s.begin(), s.end(), k)` operates on bidirectional iterators and must step through elements linearly, taking $O(n)$ iterator advances. ComplexityLens correctly distinguishes member lookups from standard range algorithms.

**Q4: How does ComplexityLens handle hash table lookups?**  
*Answer:* It provides a configurable cost model: in `expected_hash` mode (default), hash lookups are $O(1)$ with an explicit assumption `HASH_AVERAGE_CASE`. In `strict_worst_case` mode, it assumes adversarial collisions where all keys hash to the same bucket, classifying lookups as $O(n)$.

**Q5: How does the engine prevent infinite loops during function substitution?**  
*Answer:* We compute Strongly Connected Components (SCCs) using Tarjan's algorithm to detect cycles, and we enforce a strict recursion depth cap (`MAX_CALL_DEPTH = 20`). If a cycle or depth limit is encountered, the engine terminates and returns `UNKNOWN`.

**Q6: What is the 26-feature vector in Day 6 and why does it not include static engine predictions or ground truth labels?**  
*Answer:* The feature vector extracts 26 purely structural and deterministic properties directly from the IR (loops, nesting pairs, recursion sites, call kinds, container operations). Engine predictions and labels are strictly excluded to avoid data leakage; machine learning models trained on these features must learn patterns independently without circular self-fulfilling inputs.

**Q7: How is `max_loop_depth` computed when a loop in function A calls function B?**  
*Answer:* A loop at depth $d$ calling a non-recursive callee B with maximum intrinsic loop depth $k$ contributes a candidate depth of $d + k$. To prevent infinite traversal on recursive functions, recursive cycles contribute 0 additional depth across call sites and traversal does not follow cyclic calls.

**Q8: Why is ground truth labeled independently in a separate commit before running the static analyzer?**  
*Answer:* If labels were derived using the tool or modified to match its outputs, the benchmark would measure self-consistency rather than true algorithmic accuracy. Hand-derived mathematical proofs ensure ground truth reflects actual asymptotic complexity.

**Q9: What are the five allowed reasons for an `UNKNOWN` ground-truth label?**  
*Answer:* The closed vocabulary consists of: `data_dependent` (iteration count depends on array/pointer data values), `termination` (non-obvious mathematical halting like Collatz or floating-point convergence), `unsupported_construct` (syntax deliberately outside supported IR like inline assembly or unstructured `goto`), `external_call` (opaque library or unresolved functions), and `irregular_recursion` (dynamic or non-standard recurrence branching).

**Q10: Why does the project maintain both a 10-family legacy baseline and a 68-family Phase 2 benchmark?**  
*Answer:* The 10 legacy families (38 programs) represent the historical Week 1 baseline. Preserving them unaltered under governance policy prevents benchmark drift and allows regression testing, while the 68 new fine-grained families (204 programs) provide a structured, stratified dataset with seeds and structural variants for ML model training and evaluation.
