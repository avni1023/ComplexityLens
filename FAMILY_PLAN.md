Markdown · FAMILY_PLAN.md

# ComplexityLens — Final Family Plan (v1.0, 68 families)

Authored independently of the engine. **The author (you) must review every row before the agent writes any code.** If you disagree with a label, change it here first, with a reason. The agent must never change labels; it may only list "label concerns" in its report.

## Conventions (copy into docs/DATASET.md)

- Worst-case time; single input symbol `n` (integer parameter value, container size, or named constant such as `N`).
- `unordered_*` operations are O(1) under the default `expected_hash` cost model (assumption `HASH_AVERAGE_CASE`); `push_back` is amortized O(1).
- Labels: `O1, OLOGN, ON, ONLOGN, ON2, ON3, OEXP, UNKNOWN`.
- UNKNOWN reasons: `data_dependent` (bound depends on the program's own input/data values), `termination` (deterministic code whose termination is not provable), `unsupported_construct`, `external_call` (behavior depends on a call outside the analyzed code), `irregular_recursion`.
- UNKNOWN is the gold label when *static analysis of the code alone* cannot determine the class, even if a human could sometimes reason about it.

## Per-family recipe

- **Day 8 (hand-written):** 3 programs per family = a seed plus 2 structurally different variants (different loop style, helper-function split, container choice, or early-return style). About 204 programs. Each record has `origin: hand_written`.
- **Later (mutation tool, separate task):** 2 variants per family (rename identifiers, `for↔while`, helper wrapper), `origin: mutation`, bringing the total to about 340. Same label as the seed.
- Mutation variants test robustness; they add no algorithmic diversity. The 68 families are what count for generalization. Report hand-written and mutated counts separately.

## Families

| ID | Idea | Label | Tag | Derivation note |
| --- | --- | --- | --- | --- |
| f01 | swap two ints, return sum | O1 | easy | straight-line |
| f02 | `for(i<10)` accumulate | O1 | easy | fixed 10 iterations |
| f03 | 10×10 constant nested loops | O1 | easy | constant bounds |
| f04 | `a[0]+a[5]` fixed-index access | O1 | easy | constant work |
| f05 | return `n*(n+1)/2` | O1 | easy | arithmetic only |
| f06 | `v.front()+v.back()` on a non-empty vector (guard against empty) | O1 | easy | constant member calls |
| f07 | `while(n>1) n/=2` count steps | OLOGN | easy | halving |
| f08 | `for(i=1;i<n;i*=2)` | OLOGN | easy | doubling |
| f09 | iterative binary search on sorted vector | OLOGN | easy | interval halves each step |
| f10 | digit count `while(n>0) n/=10` | OLOGN | easy | divide by 10 (n = integer value) |
| f11 | power by squaring (iterative, exponent halves) | OLOGN | easy | exponent halves |
| f12 | `std::lower_bound` on sorted vector | OLOGN | easy | random-access binary search |
| f13 | recursion `f(n/2)` + O(1) | OLOGN | easy | T(n)=T(n/2)+O(1) |
| f14 | sum of array | ON | easy | one pass |
| f15 | max element | ON | easy | one pass |
| f16 | linear search with early return | ON | easy | worst case scans all n |
| f17 | count elements matching a condition | ON | easy | one pass |
| f18 | reverse array in place | ON | easy | n/2 swaps |
| f19 | two sequential loops over n | ON | easy | n + n |
| f20 | build prefix-sum array | ON | easy | one pass |
| f21 | frequency count with `unordered_map` | ON | easy | n O(1) expected ops (record HASH_AVERAGE_CASE) |
| f22 | factorial, recursive | ON | easy | T(n)=T(n−1)+O(1) |
| f23 | `std::sort` then sum | ONLOGN | easy | sort dominates |
| f24 | merge sort, recursive on (lo,hi) ranges | ONLOGN | rule_hard | 2T(n/2)+O(n) |
| f25 | loop to n with inner halving loop | ONLOGN | easy | n × log n |
| f26 | sort, then `lower_bound` for each element | ONLOGN | easy | n log n + n log n |
| f27 | insert n elements into `std::set` | ONLOGN | easy | n × O(log n) |
| f28 | sort then count unique via adjacent scan | ONLOGN | easy | n log n + n |
| f29 | sort, then two-pointer pair-sum scan | ONLOGN | easy | n log n + n |
| f30 | print n×n matrix, nested loops | ON2 | easy | n × n |
| f31 | triangular loops `j<i` | ON2 | easy | n(n−1)/2 |
| f32 | bubble sort | ON2 | easy | nested passes |
| f33 | selection sort | ON2 | easy | nested scans |
| f34 | insertion sort (worst case: reverse order) | ON2 | easy | worst-case shifts n²/2 |
| f35 | pair-sum brute force | ON2 | easy | all pairs |
| f36 | duplicate check by nested loops | ON2 | easy | all pairs |
| f37 | for each element of A, `std::find` in B (both size n) | ON2 | easy | n × O(n) |
| f38 | repeated `v.erase(v.begin())` until empty | ON2 | easy | each erase O(n) |
| f39 | matrix multiplication, 3 loops | ON3 | easy | n³ |
| f40 | triple-sum brute force | ON3 | easy | all triples |
| f41 | Floyd–Warshall, 3 loops over V | ON3 | easy | V³ |
| f42 | loop to n calling a helper with nested n×n loops | ON3 | rule_hard | n × n² via helper |
| f43 | naive Fibonacci | OEXP | easy | two calls on n−1, n−2 |
| f44 | count subsets by recursion (two calls on n−1) | OEXP | easy | 2^n |
| f45 | Tower of Hanoi | OEXP | easy | T(n)=2T(n−1)+1 |
| f46 | three recursive calls on n−1 | OEXP | easy | 3^n (coarse EXP) |
| f47 | bound via intermediate variable `m=n/2`, `for(i<m)` | ON | rule_hard | n/2 iterations |
| f48 | `for(i<v.size()-2)` bound | ON | rule_hard | size minus literal |
| f49 | while loop with constant step variable (`step=2`) | ON | rule_hard | n/2 iterations |
| f50 | helper with linear loop, called inside a linear loop | ON2 | rule_hard | n × n via helper |
| f51 | `std::count` inside a loop over n | ON2 | rule_hard | n × O(n) |
| f52 | nested loops with `break` on a rare condition (worst case no break) | ON2 | rule_hard | worst case n² |
| f53 | two pointers from both ends (`while(l<r)`) | ON | rule_hard | pointers meet in ≤ n steps |
| f54 | fixed-size sliding window sum, window k ≤ n | ON | rule_hard | n−k+1 steps, O(1) each |
| f55 | recursive binary search on (lo,hi) | OLOGN | rule_hard | interval halves |
| f56 | Euclid's gcd loop (`a % b`) | OLOGN | rule_hard | remainder shrinks geometrically |
| f57 | mutual recursion `isEven/isOdd` with n−1 | ON | rule_hard | n calls |
| f58 | divide-and-conquer array sum on (lo,hi) | ON | rule_hard | 2T(n/2)+O(1) |
| f59 | Collatz step counter | UNKNOWN | hard | `termination` |
| f60 | pointer chase `while(x!=target) x=next[x]` | UNKNOWN | hard | `data_dependent` |
| f61 | `for(i<a[0])` bound is a data value, not a size | UNKNOWN | hard | `data_dependent` |
| f62 | `std::sort` with a lambda comparator containing a loop | UNKNOWN | hard | `unsupported_construct` |
| f63 | calls a declared-only external function `process(n)` | UNKNOWN | hard | `external_call` |
| f64 | function pointer invoked in a loop | UNKNOWN | hard | `unsupported_construct` |
| f65 | recursion whose split point comes from data values | UNKNOWN | hard | `irregular_recursion` |
| f66 | `while(rand()%10 != 0)` (include `<cstdlib>`) | UNKNOWN | hard | `external_call` |
| f67 | floating-point convergence loop `while(fabs(x-y)>eps)` | UNKNOWN | hard | `termination` |
| f68 | loop built with `goto` | UNKNOWN | hard | `unsupported_construct` |

**Totals (68 families):** O1 6 · OLOGN 9 · ON 16 · ONLOGN 7 · ON2 12 · ON3 4 · OEXP 4 · UNKNOWN 10. Rule-hard: 14 families (f24, f42, f47–f58), about 21%. UNKNOWN: 10 families, about 15%.

## Parse expectations (`expected_parse` in family YAML)

- `has_unsupported_constructs`: f62 (lambda), f64 (function pointer), f68 (goto). Programs must be syntactically valid C++; the IR should mark them incomplete.
- `valid`: every other family, including f63 (declared-only function; the parser needs no definition) and f66 (needs `<cstdlib>` on the header whitelist).
- No syntax-error programs anywhere in the dataset.

## Split guidance (for the split day)

- Split by family, stratified by class. ON3 and OEXP have only 4 families each, and UNKNOWN has 10 (5 reason types, 2 families each at most), so stratify or the test set may get one or zero of a class.
- Suggested family counts: train about 40, validation about 10, test about 16–18. Aim for at least 1 family of every class and 2–3 UNKNOWN families in test.
- Run the cross-family similarity check before freezing the split.

## Mapping your existing 10 families

Map each onto the closest row above and keep them. Do not create duplicates. Existing labels, rationales and hashes stay unchanged.

## Traps for the labeler

- f06: guard against the empty vector so the program has no undefined behavior.
- f16, f34, f52: label the worst case, not typical behavior.
- f21: record the hash assumption.
- f10: `n` is the integer's value.
- f54: use a running sum (a nested loop that re-sums each window would be O(n·k)).
- f37, f41: treat both input sizes (or V) as the single symbol `n`; record `SINGLE_INPUT_SYMBOL`.
- f46: EXP is a coarse class; the rationale should say 3^n.
- f66: reason is `external_call`, not `data_dependent` (the exit comes from library `rand()`, outside the analyzed code, with no worst-case bound) and not `termination` (that is for deterministic code like Collatz, f59, and f67). f63 and f66 share a reason on purpose; they differ in structure.
- f59–f68: the question is whether static analysis alone can determine the class, not whether a human could.