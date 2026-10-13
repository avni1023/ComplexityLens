# Static Complexity Engine Specification & Reference

`complexity_lens.complexity` implements the deterministic static time-complexity engine for ComplexityLens. This engine analyzes C++ AST and typed Intermediate Representation (IR) structures to derive asymptotic bounds, structured rule traces, reason codes, and human-readable explanations.

---

## 1. Input-Size Convention

The static complexity engine operates under a rigorous, explicit input-size convention:

1. **Input-Size Symbols**:
   - Integer function parameters (e.g. `int n`, `size_t len`).
   - Container size expressions (`.size()`).
   - Local variables populated from standard input streaming (`std::cin >> n`).
2. **Constant Bounds**:
   - Literal numeric bounds (e.g. `i < 100`) are compile-time constants. Loops bounded by literals have $O(1)$ iteration counts.
   - Local variables with **exactly one literal assignment** (e.g. `int limit = 10;`) are classified as constants.
   - Variables with **multiple assignments** are classified as `UNKNOWN` bounds unless a stronger property is provably established.
3. **Named Compile-Time Constants as Input Size**:
   - Named compile-time constants (`#define N 1000`, `const int N = 1000;`, `constexpr int N = 1000;`) are treated as input-size symbols under the explicit assumption `NAMED_CONSTANT_AS_INPUT_SIZE`. This reflects common competitive programming and algorithm benchmarks where problem scale $N$ is declared as a global constant.
4. **Single Input Symbol Assumption (`SINGLE_INPUT_SYMBOL`)**:
   - If more than one distinct input-size symbol affects the analysis (e.g. `n` and `m` in sequential or nested loops), the engine records the assumption `SINGLE_INPUT_SYMBOL`.
   - The explanation explicitly states that a single dominant scale represents the iteration cost and that multivariable relationships (e.g. $O(n \cdot m)$ vs $O(n^2)$) are not modeled as separate algebraic dimensions.

---

## 2. Asymptotic Class Model & Discrete Taxonomy

### Mathematical Representation
Asymptotic complexity classes are represented as:
$$O(n^p (\log n)^q)$$
where $p \ge 0$ is the non-negative polynomial degree, and $q \ge 0$ is the logarithmic degree. In addition, two special classes exist:
- **`EXP`**: Proven exponential complexity ($O(2^n)$).
- **`UNKNOWN`**: Complexity could not be proven by static rules.

### Discrete Taxonomy Labels
Every result maps deterministically via `to_taxonomy_label()` to exactly one of nine discrete labels:

| Taxonomy Label | Class Representation | Description |
| :--- | :--- | :--- |
| `O1` | $p=0, q=0$ | Constant time $O(1)$ |
| `OLOGN` | $p=0, q=1$ | Logarithmic time $O(\log n)$ |
| `ON` | $p=1, q=0$ | Linear time $O(n)$ |
| `ONLOGN` | $p=1, q=1$ | Linearithmic time $O(n \log n)$ |
| `ON2` | $p=2, q=0$ | Quadratic time $O(n^2)$ |
| `ON3` | $p=3, q=0$ | Cubic time $O(n^3)$ |
| `OEXP` | `is_exp=True` | Exponential time $O(2^n)$ |
| `OUT_OF_TAXONOMY` | $p > 3$ or $q > 1$ | Valid class outside discrete labels (e.g. $O(n^4)$) |
| `UNKNOWN` | `is_unknown=True` | Indeterminate or unanalyzable construct |

**Invariance**: Loop analysis never produces `EXP`. Genuinely established exponential complexity is reserved for recursive branch patterns.

---

## 3. Function Entry Selection Precedence

Analysis begins by selecting a function from the translation unit:

1. **Explicit Name**: If caller requests a function name, analyze that function. If missing, raise `FunctionSelectionError`.
2. **`main` Entry Point**: If no function is requested, select `main` when present.
3. **Sole Function**: If no `main` exists and exactly one function is present, select it.
4. **Ambiguous Selection**: If multiple functions exist and no `main` is present, raise `FunctionSelectionError` with available function names.
5. **Empty Program**: If no functions exist, raise `FunctionSelectionError`.

---

## 4. Loop Classification Rules

The engine classifies loops based on verifiable syntactic and semantic properties:

### A. Linear Loops ($O(n)$ or $O(1)$)
- **Supported Updates**: `++`, `--`, `+= c`, `-= c`, where $c \ge 1$ is a nonzero literal.
- **Supported Conditions**: `<`, `<=`, `>`, `>=`.
- **Unsupported**: Variable step sizes (`i += step`) and `!=` conditions yield `UNKNOWN` with `UNKNOWN_BOUND`.
- **Bounds**: Literal bounds yield $O(1)$; input-size bounds yield $O(n)$.

### B. Doubling Loops ($O(\log n)$)
- **Supported Updates**: `i *= c`, `i = i * c`, `i += i` ($c \ge 2$ literal).
- **Initialization**: Must be provably $\ge 1$. Initialization at $0$ yields `UNKNOWN` with `UNKNOWN_BOUND` (preventing infinite loops or misclassification).
- **Conditions**: `<` or `<=`.

### C. Halving Loops ($O(\log n)$)
- **Supported Updates**: `n /= c`, `i /= c` ($c \ge 2$ literal).
- **Conditions**: Termination condition such as `i > 0` or `i > 1`.
- Verified that the loop step operates directly on the condition variable.

### D. Range-For Loops
- Container with input size (e.g. `vector<int> v` parameter): $O(n)$.
- Fixed-size array (e.g. `int arr[10]`): $O(1)$.
- Unidentifiable container size: `UNKNOWN`.

### E. Variable Modification & Early Exits
- **`LOOP_VAR_MODIFIED`**: If the loop variable or bound variable is modified in the loop body, return `UNKNOWN`.
- **Early Exits (`break`, `return`)**: Inside a loop, changes exactness to `UPPER_BOUND`; asymptotic class is preserved.
- **`continue`**: Does not change asymptotic class or exactness.
- **Dependent Bounds (`DEPENDENT_BOUND`)**: Triangular bounds (e.g. `for (int j = 0; j < i; ++j)`) yield `UNKNOWN` with reason code `DEPENDENT_BOUND`.

---

## 5. Composition and Dominance

### Dominance Ordering & Sequential Composition
- Sequential constructs are composed by computing the asymptotic maximum:
  $$\max(C_1, C_2)$$
- Two sequential linear loops yield $O(n)$, not $O(n^2)$.
- **Ordering**: Compares polynomial degree $p$ first, then logarithmic degree $q$.
- **Algebraic Invariants**: Dominance is commutative ($\max(a, b) = \max(b, a)$) and associative ($\max(\max(a, b), c) = \max(a, \max(b, c))$).

### Nested Loop Multiplication
- For independent nested loops, iteration counts multiply:
  $$O(n^{p_1} (\log n)^{q_1}) \times O(n^{p_2} (\log n)^{q_2}) = O(n^{p_1 + p_2} (\log n)^{q_1 + q_2})$$
- Multiplication is commutative and associative.

### Conservative `UNKNOWN` Propagation
- If any component of a sequential block or product is `UNKNOWN`, the composed result is strictly `UNKNOWN`.
- `UNKNOWN` is never silently demoted to $O(1)$ or discarded.

---

## 6. Constant-Cost Call Allowlist

Every call inside analyzed code must be verified against the constant-cost allowlist:

| Construct | Cost | Notes / Assumption |
| :--- | :--- | :--- |
| `std::cin >>` | $O(1)$ | Input stream extraction |
| `std::cout <<` | $O(1)$ | Output stream insertion |
| `.size()` | $O(1)$ | Container size inspection |
| `.empty()` | $O(1)$ | Container emptiness check |
| `.push_back()`, `.emplace_back()` | $O(1)$ amortized | Assumption `AMORTIZED_CONSTANT_CALL` |
| `operator[]` | $O(1)$ | Direct indexing on vector/array |
| `std::min`, `std::max`, `std::abs`, `std::swap` | $O(1)$ | Standard value comparisons and swaps |
| `vector()`, `string()`, `array()`, `pair()` | $O(1)$ | Default container constructors |
| Arithmetic operators (`+`, `-`, `*`, `/`, `%`) | $O(1)$ | Primitive arithmetic operations |

**Strict Prohibition**:
Any function call outside this allowlist yields `UNKNOWN` with reason code `UNSUPPORTED_CALL`. This includes user-defined functions and non-allowlisted STL algorithms (`std::sort`, `std::find`, `std::lower_bound`).

---

## 7. Diagnostics and Unsupported IR Scoping

1. **Local Error Diagnostic**: Any error-severity IR diagnostic located inside the selected function span makes the function result `UNKNOWN` with reason `INCOMPLETE_IR`.
2. **Local Unsupported IR**: Any `UnsupportedIR` construct inside the selected function makes the result `UNKNOWN` with `UNSUPPORTED_CONSTRUCT`.
3. **External Diagnostics & Constructs**: Diagnostics or `UnsupportedIR` nodes located demonstrably outside the selected function do **not** poison the selected function's result. A `PARTIAL_IR` note is appended to the result notes.
4. **Unlocatable Diagnostics**: If a diagnostic has no source span and cannot be attributed, the engine conservatively returns `UNKNOWN` with `INCOMPLETE_IR`.

---

## 8. Exactness and Stable Reason Codes

### Exactness Values
- **`STRUCTURAL`**: Proven asymptotic bound with no loop early exits.
- **`UPPER_BOUND`**: Proven asymptotic bound affected by `break` or `return` inside loop bodies.
- **`UNKNOWN`**: Indeterminate or unsupported construct.

### Stable Reason Codes
- `CONSTANT_WORK`: Straight-line statements or $O(1)$ operations.
- `LINEAR_LOOP`: Linear loop iteration bounded by input symbol.
- `CONSTANT_BOUND_LOOP`: Loop bounded by literal or constant.
- `LOGARITHMIC_UPDATE`: Multiplicative doubling or division halving loop.
- `NESTED_LOOP`: Nested independent loop multiplication.
- `SEQUENTIAL_COMPOSITION`: Sequential dominance composition.
- `IF_ELSE_WORST_CASE`: Branching evaluated using dominant branch.
- `UNKNOWN_BOUND`: Loop bound condition or update unsupported.
- `LOOP_VAR_MODIFIED`: Loop counter modified within body.
- `UNSUPPORTED_CALL`: Callee not on constant-cost allowlist.
- `UNSUPPORTED_CONSTRUCT`: Unsupported syntax or construct in function.
- `INCOMPLETE_IR`: Translation unit or function has error diagnostic.
- `DEPENDENT_BOUND`: Triangular/dependent loop bound.
- `SINGLE_INPUT_SYMBOL`: Multiple input symbols evaluated under single-variable scale.
- `NAMED_CONSTANT_AS_INPUT_SIZE`: Compile-time named constant treated as input size.
- `AMORTIZED_CONSTANT_CALL`: Amortized constant time operation assumed.
- `PARTIAL_IR`: Construct outside function noted without poisoning result.
