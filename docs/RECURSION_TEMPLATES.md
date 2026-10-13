# Recursion Templates Reference

`complexity_lens.complexity.recursion` implements four conservative recursion templates T1–T4
for deterministic classification of recursive functions.

---

## Preconditions (All Templates)

Every supported template must satisfy **all** of the following:

1. **Single integer size parameter** `p` — the function must have exactly one
   integer-typed parameter (`int`, `long`, `size_t`, etc.) that serves as the
   input size.

2. **Base-case guard** — the function must contain a conditional branch whose
   condition mentions `p` and whose non-recursive branch path returns without
   making any further recursive calls.  
   Example: `if (p <= 1) return 1;`  
   If the base-case guard is absent, the engine returns `UNKNOWN` with reason
   code `BASE_CASE_MISSING`.

3. **Exact recursive-call reduction** — each recursive call must reduce `p`
   using the exact form required by the template: either `p - c` (literal
   constant c ≥ 1) or `p / b` (literal constant b ≥ 2).

4. **Recursive call counting rules**:
   - Calls in **exclusive** `if / else` branches take the **maximum** count,
     not the sum.
   - **Sequential** recursive calls add to the count.
   - A recursive call inside a **loop** → `UNKNOWN` with reason code
     `RECURSION_IN_LOOP`.

5. **No template inference from function names** — templates are matched on
   structural properties only.

---

## T1 — One Decrementing Recursive Call

**Structure**:
- Exactly one recursive call per execution path on `p - c`, where `c` is a
  literal integer and `c ≥ 1`.

**Results by non-recursive work `f`**:

| Non-recursive work `f` | Result |
| :--- | :--- |
| O(1) | O(n) |
| O(n) | O(n²) |
| Any other class | UNKNOWN |

**Reason code**: `RECURSION_TEMPLATE_T1`

**Example — factorial** (T1, O(1) work → O(n)):
```cpp
int factorial(int n) {
    if (n <= 1) return 1;            // base-case guard
    return n * factorial(n - 1);     // one call on p-1
}
// Result: O(n), STRUCTURAL
```

---

## T2 — One Dividing Recursive Call

**Structure**:
- Exactly one recursive call per execution path on `p / b`, where `b` is a
  literal integer and `b ≥ 2`.

**Important**: A pair of recursive calls in **mutually exclusive** `if / else`
branches counts as **one** call per execution path.  It must not be classified
as exponential merely because the function contains two call expressions.

**Results by non-recursive work `f`**:

| Non-recursive work `f` | Result |
| :--- | :--- |
| O(1) | O(log n) |
| O(n) | O(n) |
| Any other class | UNKNOWN |

**Reason code**: `RECURSION_TEMPLATE_T2`

**Example — binary search** (T2, O(1) work → O(log n)):
```cpp
int bsearch(int n) {
    if (n <= 0) return -1;            // base-case guard
    // ... O(1) work ...
    return bsearch(n / 2);            // one call on p/2
}
// Result: O(log n), STRUCTURAL
```

**Example — exclusive if/else** (T2, not EXP):
```cpp
int biSplit(int n) {
    if (n <= 1) return 0;             // base-case guard
    if (someCondition)
        return biSplit(n / 2);        // branch A: one call on p/2
    else
        return biSplit(n / 2) + 1;    // branch B: one call on p/2
    // max(1, 1) = 1 call per path → T2
}
// Result: O(log n), STRUCTURAL
```

---

## T3 — Two Half-Size Recursive Calls with Linear Work

**Structure**:
- Exactly **two** recursive calls on `p / 2`, where `2` is a literal divisor.
- **Linear** (O(n)) non-recursive work.

**Result**: O(n log n)

**Reason code**: `RECURSION_TEMPLATE_T3`

Any other reduction factor, number of recursive calls, or work class → `UNKNOWN`
unless another explicitly defined template matches.

**Example** (merge-sort style with 2 calls):
```cpp
void mergeStyle(int n) {
    if (n <= 1) return;               // base-case guard
    doLinearWork(n);                  // O(n) non-recursive work
    mergeStyle(n / 2);                // call 1
    mergeStyle(n / 2);                // call 2
}
// Result: O(n log n), STRUCTURAL
```

---

## T4 — Multiple Decrementing Recursive Calls

**Structure**:
- **Two or more** recursive calls per execution path.
- Every recursive call uses `p - c`, with literal `c ≥ 1`.
- O(1) non-recursive work.

**Result**: `EXP` (exponential)

**Reason code**: `RECURSION_TEMPLATE_T4`

> **Limitation**: `EXP` is the engine's coarse exponential category.  It does
> not imply a specific base; the actual growth rate depends on the exact number
> of recursive calls and the constant `c`.  The engine does not attempt to
> derive a precise exponential base.

**Example — naive Fibonacci** (T4):
```cpp
int fib(int n) {
    if (n <= 1) return n;             // base-case guard
    return fib(n - 1) + fib(n - 2);  // two calls on p-1 / p-2
}
// Result: EXP, STRUCTURAL
```

---

## Explicit Unsupported Patterns → UNKNOWN

The following patterns are **not** matched by T1–T4 and always return `UNKNOWN`:

| Pattern | Reason Code |
| :--- | :--- |
| Range-style recursion (`lo`, `hi`, `mid` parameters) | `RANGE_RECURSION_UNSUPPORTED` |
| Mutual recursion (multi-function cycle) | `MUTUAL_RECURSION` |
| Missing base-case guard | `BASE_CASE_MISSING` |
| Recursive call inside a loop body | `RECURSION_IN_LOOP` |
| Unsupported input-size reduction (e.g. `p - x` where `x` is not a literal) | `UNSUPPORTED_REDUCTION` |
| Unsupported non-recursive work class (anything other than O(1) or O(n)) | `UNSUPPORTED_REDUCTION` |
| Ambiguous or conflicting recursive call patterns | `UNSUPPORTED_REDUCTION` |
| Any recursive function not matching T1–T4 | `UNSUPPORTED_REDUCTION` |

---

## Summary Table

| Template | Recursive calls | Reduction | Non-recursive work | Result |
| :--- | :--- | :--- | :--- | :--- |
| T1 | 1 | `p - c` (c ≥ 1) | O(1) | O(n) |
| T1 | 1 | `p - c` (c ≥ 1) | O(n) | O(n²) |
| T2 | 1 (per path) | `p / b` (b ≥ 2) | O(1) | O(log n) |
| T2 | 1 (per path) | `p / b` (b ≥ 2) | O(n) | O(n) |
| T3 | 2 | `p / 2` | O(n) | O(n log n) |
| T4 | ≥ 2 | `p - c` (c ≥ 1) | O(1) | EXP |
