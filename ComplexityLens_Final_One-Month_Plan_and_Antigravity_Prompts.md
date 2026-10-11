# ComplexityLens — One-Month MVRS Plan (v3)

**Title (frozen):** *ComplexityLens: An Explainable Hybrid Static Analysis and Machine Learning System for C++ Complexity Analysis*

> v3 **replaces the schedule and scope** of v1/v2 and keeps their research logic (hybrid, verification, abstention, family splits, evidence-based explanations). Anything not restated here is cut or optional. Your message ended mid-sentence at the Week 2 gate ("The system must produce…"), and Weeks 3–4 were not included. I completed the Week 2 gate and designed Weeks 3–4 myself (§9–§10). Check them against what you intended.

## 0. Honest feasibility statement

One month is feasible **only if** you (a) work about 4–5 focused hours per day, (b) let the Antigravity agent write most code while you review, test and author/verify the dataset, and (c) enforce the cut rules in §11 without negotiating. The real bottlenecks are (1) the parser environment and (2) authoring and verifying the labeled program families. Dataset authoring starts in Week 1 in parallel with the engine. No research paper or academic report is required; maintain only practical technical documentation.

## 1. The Minimum Viable Research System (MVRS)

**Frozen pipeline**

```
C++ code → validate → parse (libclang) → IR → features
   ├─ static rule engine → (class | UNKNOWN, evidence, reason)
   └─ XGBoost (calibrated) → (class, probability)
→ verification + abstention gate → final class OR UNKNOWN(reason)
→ evidence-based explanation → optimization suggestions → CLI / API / simple UI
```

**MVRS = exactly this (17 required items from your list)**

| # | Required item | MVRS definition (what "done" means) |
| --- | --- | --- |
| 1 | C++ input | Single `.cpp` text, ≤ 200 lines, one translation unit |
| 2–3 | Parsing, AST/IR | libclang → small own IR (functions, loops, calls, recursion info, container decls) |
| 4–5 | Static engine, classification | Rule-based over IR; outputs one of 7 labels or UNKNOWN |
| 6 | Features | 26 features (§4) |
| 7 | ML model | Majority, LogReg, DecisionTree, RandomForest, **XGBoost (primary)**, calibrated |
| 8 | Hybrid | Fusion of static + ML (§6) |
| 9 | Abstention | 4 triggers (§6), thresholds from validation only |
| 10 | Explanation | Evidence → claims → templates; small injected-fault test |
| 11 | Optimization | 4 rules (§7) with "validity could not be established" outcome |
| 12–13 | Dataset, split | \~300–350 programs, \~65–70 families, family-level train/val/test, hash-frozen test |
| 14–15 | Baselines, evaluation | S1 static-only, S2 ML-only, S3 naive hybrid, S4 hybrid+verification/abstention |
| 16 | API/UI | `POST /analyze` (FastAPI) + simple Streamlit page |
| 17 | Final results | E1–E6 tables/figures from one reproducible script |

**OPTIONAL / ONLY IF TIME REMAINS:** space-complexity analysis (a tiny descriptive version is in §11), human evaluation, Docker packaging (mandatory only if the parser needs it), SQLite history, React frontend, extra optimization rules, extra patterns, Tier-C real-world samples, LLM paraphrasing, risk–coverage CIs beyond simple bootstrap, calibration beyond one method.

**Note on space complexity:** your original problem statement mentions it, but the one-month MVRS is **time-complexity only**. State this in the project documentation as a scoped limitation; if a faculty member asks, you can show the lightweight descriptive space note from §11 (if built) and list full space analysis as future work.

## 2. Taxonomy (7 labels + UNKNOWN)

`O(1), O(log n), O(n), O(n log n), O(n²), O(n³), O(2ⁿ)`, plus `UNKNOWN`.

I recommend **adding O(n³)** instead of mapping it to UNKNOWN: triple-nested loops are the most common "next" case, the static rule is trivial (depth 3), and calling a fully determinable program UNKNOWN would corrupt the "UNKNOWN" meaning in your abstention evaluation. Fallback: if you end up with fewer than \~20 O(n³) programs, merge them out of training and keep only 1–2 test examples, or drop the class. Decide at Day 8 from the dataset counts.

## 3. Supported patterns (8) and the simplified static engine

**Patterns (MVRS):** (1) constant/straight-line, (2) single linear loop, (3) logarithmic loop / binary-search-like loop, (4) nested loops (independent and triangular), (5) sequential loops, (6) sorting/searching via `std::sort/find/count/binary_search/lower_bound`, (7) simple recursion (4 recurrence templates), (8) common STL ops (`unordered_*` average O(1), `map/set` O(log n), `push_back` amortized O(1), `vector::insert/erase` O(n)). Not required: DP, graphs, two pointers, sliding window, complex recursion, templates, pointer aliasing, concurrency.

**Engine design: no symbolic algebra.** A complexity is a pair **(p, l)** meaning `n^p · (log n)^l`, plus the special value `EXP` and `UNKNOWN`. Whole algebra in \~100 lines:

| Operation | Rule |
| --- | --- |
| Sequential statements | element-wise **max** by (p, then l) |
| Nested loop | **add** components: `(p₁+p₂, l₁+l₂)` |
| `if/else` | max of branches |
| Loop trip count | `i++/i+=c` to input-size bound → (1,0); `i*=c`/`i/=c` → (0,1); constant bound → (0,0); inner bound depends on outer index (`j<i`, `j=i..n`) → (1,0) (triangular, same class as linear) |
| `std::sort` → (1,1); `find/count/accumulate` → (1,0); `binary_search/lower_bound` → (0,1); `unordered_*` op → (0,0) \[assumption\]; `map/set` op → (0,1); `vector::insert/erase` → (1,0) |  |
| Call to user function | substitute callee's (p,l) (non-recursive only) |
| Recursion | only 4 templates: `T(n)=T(n−c)+f(n)` → f·n (f const → (1,0)); `T(n)=T(n/2)+f(n)` with f const → (0,1), with f linear → (1,0); `T(n)=2T(n/2)+f(n)` with f linear → (1,1); `≥2` self-calls on `n−c` with const work → EXP. Anything else → UNKNOWN |
| Early `break/return` | does not lower the worst case; mark `exactness=UPPER_BOUND` |
| Anything unresolved | UNKNOWN with a reason code |

**Single input symbol:** all input-size symbols are treated as one `n` (documented assumption). Two independent sizes (`n·m`) are reported as `O(n²)` with an assumption note. Mark this as a limitation.

Output of the engine: `StaticResult(label | UNKNOWN, exactness {EXACT, UPPER_BOUND}, evidence[IR node IDs + lines], reason_codes)`.

## 4. Feature set: 26 features

| Group | Features | Why it exists |
| --- | --- | --- |
| Loops (11) | `n_loops`, `max_loop_depth`, `n_sequential_top_loops`, `n_nested_loop_pairs`, `n_for`, `n_while`, `n_loops_linear_bound`, `n_loops_const_bound`, `n_loops_log_update` (mul/div), `n_dependent_bound_loops`, `n_loops_unknown_bound` | Polynomial degree, log factors, triangular loops, unresolved bounds |
| Recursion (5) | `has_recursion`, `n_recursive_call_sites`, `shrink_minus_const`, `shrink_divide`, `shrink_other` | Linear vs log vs exponential recursion; branching = call-site count |
| Calls/ops (6) | `n_user_calls`, `n_sort_calls`, `n_linear_search_calls`, `n_binary_search_calls`, `n_hash_ops`, `n_ordered_container_ops` | n log n, log, hidden linear operations |
| Other (4) | `n_container_ops_in_loops` (push_back/insert/erase), `has_early_exit_in_loop`, `n_input_dependent_ops`, `n_unsupported_constructs` | Hidden costs, upper-bound cases, and a direct signal for "be careful" |

Dropped as not worth the extraction effort: array-access patterns, loop-variable relationship graphs, memory allocation, number of variables, LOC, call-graph depth. **Leakage rule:** features come from IR only, never from engine outputs, identifiers, or labels (one test asserts this).

**Honest caveat to put in the report:** these features overlap with what the engine reads, so on programs the engine solves, ML largely re-learns the same rules. The ML adds value where rules abstain or are brittle. So the dataset **must contain "rule-hard but determinable" families** (\~10–12): bound computed through an intermediate variable (`m = n/2`), `while` with constant step variable, loop inside a helper called in a loop, `std::count` inside a loop, bound from `v.size()-k`, early-exit loops. Without them, RQ2 (does ML add coverage?) would be trivially "no".

## 5. Dataset: small, controlled, \~330 programs

| Item | Target |
| --- | --- |
| Families | **≈ 66** (minimum acceptable: 50) |
| Variants per family | ≈ 5 (seed + 4 mutations/hand variants) |
| Programs | ≈ 330 (minimum 250) |
| Class coverage (families) | O(1): 5 · log: 7 · n: 9 · n log n: 7 · n²: 9 · n³: 4 · exp: 4 → 45 determinable-easy; **+ \~11 rule-hard determinable** (spread over classes) ; **+ \~10 UNKNOWN** (≈15% of programs) |
| UNKNOWN reasons | data-dependent bound, non-obvious termination (Collatz-like), unsupported construct (lambda/function pointer hiding cost), unknown external call, irregular recursion |
| Splits (by family) | train ≈ 40 · validation ≈ 10 · **test ≈ 16** (≈ 80 programs) |

**Authoring workflow (the part that decides your schedule):**

1. You write a *family spec* (one-line idea + complexity + one-sentence derivation). The agent may draft code from your spec, but **you read every program and its label**.
2. Variants come from a small mutation script (rename identifiers, `for↔while`, helper wrapper, change constants/containers where safe) and from 1–2 hand-written variants per family.
3. **Labels never come from running our engine or model.** Each family has a `label_rationale`. If the engine disagrees later, you re-check your *rationale*; you change the label only if the rationale was wrong, and log it in a text file.
4. Ask one classmate to verify labels of \~20 random programs plus all UNKNOWN families (30–45 min of their time). Report agreement informally.
5. Similarity check: token-set Jaccard between families; merge into one split group if > 0.8.
6. Freeze: write `splits/test_families.txt`, store its SHA-256 in `splits/MANIFEST.txt`, commit. Rule development, feature choices, thresholds and model selection use train/validation only. One final test evaluation script.

**Be upfront about limits:** 16 test families means wide uncertainty. So also report **grouped cross-validation over all families** as supporting evidence, bootstrap over families for confidence intervals, and state that results are indicative, not conclusive.

## 6. Hybrid, verification, abstention (simple and complete)

**ML:** XGBoost primary; baselines majority, LogReg, DecisionTree, RandomForest; train on determinable programs (7 classes). UNKNOWN is produced by the gate, not predicted. Calibrate on validation (one method: isotonic if enough data, else Platt/temperature — pick on validation).

**Gate (policy P1), in priority order:**

| Condition | Output | Reason code |
| --- | --- | --- |
| Parse errors or unsupported structure in the part that matters | `UNKNOWN` | `UNSUPPORTED` |
| Static EXACT result and ML agrees | answer, confidence HIGH | — |
| Static EXACT result and ML differs by **1** class | answer = static, confidence MEDIUM | `ML_DISAGREES` |
| Static result and ML differs by **≥ 2** classes, or ML class exceeds static UPPER_BOUND | `UNKNOWN` | `CONFLICT` |
| Static UPPER_BOUND, ML class ≤ bound and `p ≥ τ` | ML class, MEDIUM, "within proven bound" | — |
| Static UNKNOWN, ML `p ≥ τ` | ML class labelled "ML estimate, not derived", LOW | — |
| Static UNKNOWN, ML `p < τ` | `UNKNOWN` | `LOW_CONFIDENCE` / `STATIC_UNDETERMINED` |

**Four triggers** = `LOW_CONFIDENCE`, `STATIC_UNDETERMINED`, `UNSUPPORTED`, `CONFLICT`. Single threshold τ, chosen on **validation** as the smallest value giving selective risk ≤ 5% (also report 2% and 10%). One simple variant for E4: **no gate** (S3).

**Evaluation systems:** S1 static-only; S2 ML-only (answers when `p ≥ τ`); S3 naive hybrid (static if available else ML top-1, no conflict check, no abstention); **S4 proposed hybrid + verification/abstention**. (Optional S0: `n^maxDepth` heuristic — 10 lines.)

**Metrics (kept small):** accuracy, macro-F1, confusion matrix, coverage, selective risk, risk–coverage curve, ECE (10 bins; say it is noisy at this size), abstention recall on gold-UNKNOWN programs, unnecessary-abstention rate.

## 7. Verification, explanations, optimization

**Verification:** checks the final label against structural evidence — e.g., claimed O(n²) requires two nested input-dependent loops in the evidence; claimed O(log n) requires a multiplicative/divisive update; claimed EXP requires ≥2 self-calls on shrinking arguments. A label that no evidence supports is rejected (→ UNKNOWN `CONFLICT`).

**Explanations (lightweight):** engine emits typed claims (`LOOP(id,lines,trips)`, `NESTED(inner,outer)`, `SEQUENTIAL(a,b)`, `RECURSION(form)`, `RESULT(label)`, `ASSUMES(text)`); templates turn claims into sentences; every sentence cites claim IDs. **Mini faithfulness test (E6):** take 40 generated explanations, inject 4 kinds of false claims (wrong line, wrong nesting, wrong result, invented loop) → \~160 corrupted cases; a small checker verifies claims against the IR; report detection rate and false-alarm rate on the clean ones. The checker re-walks the IR itself; it doesn't trust the engine's trace.

**Optimization (4 rules + one outcome):**

| Rule | Evidence → suggestion | Preconditions |
| --- | --- | --- |
| **R1** repeated linear search | `find/count`/manual `==` scan of container B inside a loop over A → build `unordered_set/map` of B | B unchanged in loop, equality lookup, order unused |
| **R2** unnecessary nested traversal | nested loops testing pair condition (`a[i]+a[j]==t`, duplicate check) → hash lookup or sort+two pointers | Marked *candidate*, MEDIUM at best |
| **R3** repeated sorting/computation | `sort` (or other invariant expensive call) inside a loop whose inputs aren't modified → hoist out | Input unchanged in loop |
| **R4** naive exponential recursion | ≥2 overlapping self-calls with pure scalar args → memoization | Function pure |

Each recommendation carries: evidence lines → detected pattern → inefficiency → suggested change → expected complexity effect → assumptions → confidence. If a precondition can't be established: *"Potential optimization identified, but validity could not be established."* E7: evaluate on \~30 programs you label for opportunities (precision, false-recommendation rate, "cannot-validate" count).

## 8. Backend and UI (deliberately tiny)

- **Backend:** one FastAPI file; `POST /analyze`, `GET /health`; Pydantic request/response; validator (size ≤ 20 KB/200 lines, include whitelist, no NUL bytes); parse in a **subprocess with a timeout**; **no DB, no auth, no Docker unless the parser needs it**. Never compile or run submitted code.
- **UI: Streamlit** (instead of React, a change from v1, to save \~3–4 days). One page: text area + Analyze button → panels for final result and status, static result, ML prediction (with probability), detected patterns, evidence (code with highlighted lines), explanation, optimization suggestions, UNKNOWN reason. If Streamlit can't highlight lines easily, render the code with line numbers and mark evidence lines in an HTML block.
- Calls the API (or imports the pipeline directly if the API slips; the demo works either way).

---

## 9. The 4-week plan

**Daily rhythm (suggested):** 1 h review/test the agent's output, 2 h next agent prompt + fixes, 1 h dataset authoring or validation, 30 min documentation/testing. **Dataset track runs every day in Weeks 1–2** (≈5–6 families/day ≈ 66 by Day 12).

### WEEK 1 — Foundation (Days 0–7)

| Day | Build | Dataset track | Done when |
| --- | --- | --- | --- |
| 0 | Repo, Python env, deps, Makefile, rules file, test skeleton | Write the family-spec template (CSV/YAML) | `make test` green |
| 1 | **Parser spike** (12 fixtures, G-criteria, decision record) | 5 family specs | libclang frozen **or** fallback chosen. **Hard stop: if not passing by end of Day 2, use fallback** |
| 2 | IR design + builder (loops, calls, recursion info, containers) | 5 families | IR dumps for fixtures |
| 3 | Finish IR; validator + subprocess/timeout wrapper | 6 families | IR golden tests pass |
| 4 | Static engine: (p,l) algebra, loops, sequential/nested, STL table | 6 families | ≥ 25 engine tests pass |
| 5 | Engine: recursion templates, UNKNOWN reasons, evidence output; engine tests | 6 families | ≥ 40 tests; UNKNOWN cases covered |
| 6 | Feature extraction (26) + leakage test | 6 families | Hand-checked vectors |
| 7 | Mutation script + dataset validate script (schema, parse check, dedupe) | 6 families; total ≥ 35 | **WEEK-1 GATE:** a CLI takes C++ → IR → features → static result with evidence |

*If the Week-1 gate fails:* stop all dataset authoring beyond what exists, freeze scope, fix the foundation on Days 8–9, and apply cut rules C1–C3 (§11).

### WEEK 2 — ML + hybrid (Days 8–14)

| Day | Build | Dataset / writing | Done when |
| --- | --- | --- | --- |
| 8 | Dataset finalization: finish families, mutate variants, validate, similarity check | Classmate verifies labels; decide the O(n³) class | ≥ 250 programs validate; counts printed |
| 9 | Family split (train/val/test), freeze test hash; extract features for all programs (cached) | Verify dataset labels, family membership, and split integrity | `splits/` committed |
| 10 | Baselines: majority, LogReg, DecisionTree (grouped CV) |  | Metrics JSON for each |
| 11 | RandomForest + XGBoost, simple tuning on validation, calibration |  | Selected model saved |
| 12 | Model evaluation on validation: accuracy, macro-F1, confusion, calibration; static-only vs ML-only on validation |  | Table V1 |
| 13 | Hybrid fusion gate (P1) + S3/S4 implementation |  | Unit tests for every gate row |
| 14 | Abstention thresholds (validation only), reason codes, CLI end-to-end |  | **WEEK-2 GATE (my completion of your cut-off sentence):** for any C++ input, the system outputs static result, ML prediction, final hybrid result or UNKNOWN with a reason, and a validation-set table comparing S1/S2/S3/S4 |

### WEEK 3 — Evidence, optimization, demo, results (Days 15–21)

| Day | Build | Done when |
| --- | --- | --- |
| 15 | Verification rules + claims + explanation templates | Explanations for all fixtures |
| 16 | Explanation checker + injected-fault experiment (E6) | Detection rate computed |
| 17 | Optimization rules R1–R4 + validator + "cannot establish validity" | Positive/negative/precondition tests pass |
| 18 | FastAPI `/analyze` + error handling + hostile-input tests | API tests pass |
| 19 | Streamlit UI | Demo works from a clean start |
| 20 | **Final experiments on the frozen test set** (one script: E1–E5), optimization eval (E7) | `results/` tables + figures generated reproducibly |
| 21 | **WEEK-3 GATE:** complete working system + final results v1. Freeze features, thresholds, models, rules | Tag `v0.9`. After this day: **no new features** |

### WEEK 4 — Integration, hardening, final evaluation, and demo (Days 22–30)

| Day | Build | Done when |
| --- | --- | --- |
| 22 | Full integration: parser → IR → static + ML → fusion → explanation → optimization | End-to-end CLI works on representative fixtures |
| 23 | Test hardening: coverage, engine/gate/validator tests, edge cases, hostile inputs | Full test suite green |
| 24 | Final dataset verification: labels, duplicates, family splits, frozen test manifest | Dataset validation passes with no unexplained issues |
| 25 | Reproducibility run from clean state; verify frozen models/configs/thresholds | Final run reproduces expected validation/test artifacts |
| 26 | Streamlit/API polish; improve evidence display, UNKNOWN messages, examples, and errors | Demo is clear and stable |
| 27 | Performance/stability testing; parser timing, API latency, repeated end-to-end runs | No blocking stability/performance issues |
| 28 | Project documentation: README, architecture, setup, dataset, experiments, limitations, usage | A new user can install, run, and understand the system |
| 29 | Final bug fixes, repository cleanup, remove dead code, final version/tag, demo dataset | Clean repository and final build |
| 30 | Demo/viva rehearsal, backup, final verification | Final demo works from a clean start |

**No research-paper or academic-report writing is part of the one-month implementation schedule.**
Research scope is demonstrated through the controlled dataset, baselines, hybrid comparison, abstention/coverage evaluation, verification experiment, optimization evaluation, ablations, and reproducible final results.


## Project documentation (not a research paper)

Maintain concise technical documentation only:
- README: purpose, setup, quickstart, demo, limitations.
- docs/ARCHITECTURE.md: pipeline and module responsibilities.
- docs/DATASET.md: dataset construction, labels, family splits, and frozen test protocol.
- docs/EXPERIMENTS.md: experiments, metrics, configurations, and final results.
- docs/LIMITATIONS.md: supported scope and known failure cases.
- docs/PROGRESS.md and docs/DECISIONS.md: implementation progress and architectural decisions.

Do not create a research paper, literature review, related-work chapter, academic report, citation list, or paper-formatting workflow unless the user explicitly requests one.


## Viva preparation (questions you must answer)

1. Why hybrid instead of pure ML or pure rules? (rules are exact but brittle; ML generalizes but can't prove; the gate combines them.)
2. Why does ML help if features come from the same AST? (it handles rule-hard cases; show the numbers; admit overlap.)
3. How did you avoid label leakage and data leakage? (human labels, family split, frozen test, similarity check.)
4. What does UNKNOWN mean and why is it good? (risk–coverage.)
5. How is the explanation verified? (checker + injected faults.)
6. Why XGBoost, not deep learning? (small tabular data, interpretability.)
7. What can't the system do? (space, templates, lambdas, multi-variable sizes, average/best cases.)
8. Is it safe? (no execution, validator, timeout, subprocess.)
9. What is novel? (the combination + evaluation framing; not Big-O prediction itself.)
10. What would you do with more time? (space analysis, more patterns, bigger dataset, other languages via the language-agnostic IR, optional LLM paraphrasing of verified claims.)

## Risk register (top 5)

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Parser environment fights you | Medium | Day-1 spike, 2-day cap, fallback C1 |
| Dataset authoring takes longer than coding | **High** | Start Day 1, 5–6 families/day, agent drafts code but you verify labels; C3 reduces to 50 families |
| ML shows no benefit over rules | Medium | Include rule-hard families; report honestly; the gate/abstention analysis still stands |
| Test-set temptation | Medium | Single `evaluate_final.py --final`; commit hash recorded |
| Integration left to the end | **High** | Keep end-to-end checks throughout Weeks 3–4; reserve Days 22–30 for hardening and demo |

## Final answer to "is this feasible?"

Yes, as the MVRS above. It is **not** feasible if you add DP/graph patterns, space analysis, React, SQLite, human study or Docker packaging before Day 21. The dataset and validation are your real critical path, not the code.

## 10. Antigravity prompts — copy and paste one day at a time

**How to use:** Open the project repository in Antigravity and paste only the prompt for the current day. Do not ask the agent to implement the whole month at once. Review the diff, run the tests yourself, and do not move forward until the day's completion check passes. If the repository already has code, inspect and extend it rather than replacing it.

### Day 0 — Repository and environment

```text
You are helping build ComplexityLens, a one-month C++ time-complexity analysis project. Follow the frozen scope in the project plan. First inspect the repository and report its current structure; do not overwrite existing work. Set up a minimal Python project, test framework, configuration, .gitignore, README skeleton, and Makefile or equivalent commands for tests/lint. Add a short dependency/environment note. Do not build the parser or UI today. Do not add a research paper, academic report, LLM, database, cloud deployment, or extra language. Run the available checks and show changed files, commands, results, and any blockers. Done when a clean test command runs successfully.
```

### Day 1 — Parser spike (time-boxed)

```text
Implement only a parser feasibility spike for C++ using the smallest reliable libclang setup available in this environment. Inspect installed compiler/libclang paths before changing dependencies. Create about 12 tiny C++ fixtures covering a function, for/while loops, nested/sequential loops, a call, recursion, STL sort/search, malformed input, and an unsupported construct. Extract enough AST information to prove whether these constructs can be recognized with source locations. Add tests and a short docs/DECISIONS.md entry recording setup, failures, and the chosen parser path. Do not build the full IR or run submitted code. If libclang is not working by the end of this spike, document the exact blocker and recommend the planned tree-sitter fallback; do not spend days debugging it. Report commands and test results.
```

### Day 2 — Minimal IR

```text
Based on the parser path that passed the spike, implement a small typed intermediate representation for functions, loops, loop bounds/updates, calls, recursion call sites, recognized STL operations, source spans, and unsupported constructs. Keep parsing separate from IR construction. Add golden tests for the parser fixtures and make unknown/unrecognized syntax explicit rather than silently guessing. Do not implement the complexity rules yet. Show sample IR output and run tests.
```

### Day 3 — Complete IR and safe input handling

```text
Finish the IR builder for the supported ComplexityLens subset. Add input validation (single C++ translation unit, maximum 200 lines, reasonable input-size limit), clear parse errors, and safe subprocess timeouts only if the parser tool requires a subprocess. Never execute the submitted C++ program or compile/run arbitrary user code. Add tests for malformed code, empty input, long input, unsupported syntax, and source-line mapping. Keep the parser adapter isolated so fallback changes are localized. Run all tests and show representative IR.
```

### Day 4 — Static complexity rules, first half

```text
Implement the deterministic static engine over the IR using the plan's (polynomial degree, logarithmic degree) representation plus EXP and UNKNOWN. Handle straight-line code, constant-bound loops, input-size linear loops, logarithmic updates, sequential statements, nested loops, and if/else worst-case combination. Return a structured result with label, exactness, evidence node IDs/source lines, and reason codes. Do not use ML or natural-language generation yet. Add unit tests for every rule and explain any unsupported case as UNKNOWN.
```

### Day 5 — Static rules, STL and recursion

```text
Extend the static engine only for the documented supported cases: recognized sort/search functions, common unordered and ordered containers, vector push_back/insert/erase, non-recursive user-function substitution, and the four recursion templates in the project plan. Preserve worst-case semantics and mark upper-bound results where appropriate. Anything outside the supported templates must return UNKNOWN with a reason, not a guessed class. Add positive, negative, boundary, and UNKNOWN tests; target at least 40 engine tests total. Do not broaden the supported language subset.
```

### Day 6 — Feature extraction

```text
Implement the 26 planned features from the IR in a stable, documented schema. Keep feature extraction deterministic and independent of labels and predictions. Add tests for feature names/order, missing constructs, nested and sequential loops, recursion, calls, and unsupported constructs. Add a leakage check proving labels, family IDs, filenames, and split metadata are not model features. Include a small human-readable feature-vector example. Do not train a model yet.
```

### Day 7 — Dataset tooling and Week 1 gate

```text
Create the dataset schema and tooling: family specification template, program record schema, validator, parser-validity check, duplicate/hash check, and a script that reports label/family counts. Dataset gold labels must be independently reasoned and recorded before running ComplexityLens on the examples; do not use the system's own predictions as ground truth. Add an initial set of at least 35 programs across several families, with source, family ID, expected label, short derivation, and UNKNOWN reason where applicable. Run the end-to-end CLI: C++ input → IR → features → static result with evidence. If this gate fails, fix the foundation before adding more scope.
```

### Day 8 — Dataset finalization

```text
Extend and validate the controlled benchmark toward the plan's target of roughly 300–350 programs across roughly 65–70 related families, but prioritize correct labels over quantity. Include all supported classes that can be labeled responsibly and UNKNOWN cases with explicit reason codes. Use independently written derivations and a second-person label review where possible. Print class and family counts, parser failures, duplicates, and unresolved labels. Do not invent examples just to balance class counts, and do not claim a target was reached unless the validation script confirms it.
```

### Day 9 — Family split and frozen test set

```text
Create train/validation/test splits by program family, not by individual file, to reduce near-duplicate leakage. Add similarity/deduplication checks, save split manifests, and hash the frozen test manifest and source files. Ensure test labels are not used for feature design, threshold tuning, model selection, or rule changes. Cache feature extraction reproducibly. Add tests proving no family overlaps between splits and document the split protocol in docs/DATASET.md. Print the final counts.
```

### Day 10 — Baseline models

```text
Implement the majority-class, logistic-regression, and decision-tree baselines using the fixed feature schema and family-based training data. Keep preprocessing inside training folds to avoid leakage. Use grouped cross-validation on training data where practical; use validation for model selection. Save configuration and metrics as machine-readable JSON, including accuracy, macro precision/recall/F1, per-class results, and confusion matrix. Do not inspect the frozen test set for tuning. Run tests and show the metrics file paths.
```

### Day 11 — Random Forest and XGBoost

```text
Add Random Forest and XGBoost candidates using the same split and feature pipeline as the baselines. Check whether XGBoost is installed before adding it; keep a clear, documented fallback if installation fails. Limit tuning to a small, reproducible validation-only search. Save model artifacts, preprocessing, class mapping, seeds, and configs together. Do not add deep learning or LightGBM. Run training from a clean command and report failures honestly.
```

### Day 12 — Model evaluation and calibration

```text
Build a repeatable validation evaluation script for all candidate models and the static-only engine. Report accuracy, macro-F1, per-class precision/recall, confusion matrix, and a calibration metric such as ECE when probabilities are available. Fit any calibration only using training/validation data, never the frozen test set. Compare static-only and ML-only results without claiming ML is better unless metrics support it. Save machine-readable outputs and a concise technical summary in docs/EXPERIMENTS.md.
```

### Day 13 — Hybrid fusion gate

```text
Implement the deterministic fusion gate combining StaticResult and ML prediction. Exact supported static results should not be downgraded merely because ML disagrees; keep analytical status separate from ML confidence. Handle static UNKNOWN, ML confidence, unsupported cases, and disagreement using explicit, unit-tested rules. Add the S1 static-only, S2 ML-only, S3 naive-hybrid, and S4 hybrid-with-verification/abstention evaluation modes. Do not hide disagreements or silently invent confidence values. Test every gate branch.
```

### Day 14 — Abstention and Week 2 gate

```text
Implement the planned abstention/reject option with explicit reason codes for low confidence, unsupported structure, unresolved static analysis, and strong disagreement where policy calls for it. Tune thresholds on validation data only. Report both coverage and selective risk so abstention cannot appear better simply by refusing more cases. Ensure any input returns static result, ML prediction when possible, and final class or UNKNOWN with a reason. Generate a validation comparison table for S1/S2/S3/S4. Do not build the frontend until the CLI gate passes.
```

### Day 15 — Evidence and explanation templates

```text
Create typed evidence records and explanation claims based only on actual IR/static-engine evidence. Generate template-based explanations that cite the relevant source lines and explain the derivation (for example, an input-dependent loop or nested loops). Do not use an LLM. If evidence is insufficient, say so instead of making a claim. Add fixtures covering each supported pattern and tests that every displayed claim points to valid evidence.
```

### Day 16 — Independent explanation checker

```text
Implement a small explanation verifier that checks typed claims against the IR/evidence records without simply calling the same explanation-generation function. Add a controlled injected-fault test that deliberately supplies false or mismatched claims and measures whether the checker catches them. Record the number of injected faults, detected faults, and failure modes in machine-readable output. Keep the verifier modest and explain its limits; do not claim formal proof of all natural-language text.
```

### Day 17 — Optimization recommendations

```text
Implement only four evidence-based optimization rules for clearly recognized cases. Each recommendation must include the triggering code evidence, preconditions, expected effect, and a validity status. If correctness or preconditions cannot be established, return 'validity could not be established' rather than suggesting a risky rewrite. Do not automatically rewrite user code. Add positive, negative, and missing-precondition tests, and evaluate recommendation precision/false-recommendation rate on a small labeled set.
```

### Day 18 — FastAPI endpoint

```text
Add a minimal FastAPI POST /analyze endpoint that calls the already-working analysis pipeline. Define validated request/response models and return the static result, ML result, final class or UNKNOWN reason, patterns, evidence, explanation, and recommendations. Add clear errors for invalid or oversized input and parser failures. Do not execute or compile submitted C++ programs. Add API tests for valid input, invalid input, unsupported code, and internal errors. Keep the CLI usable if the API fails.
```

### Day 19 — Streamlit interface

```text
Build a simple Streamlit UI for ComplexityLens: C++ text area, Analyze button, final complexity/status, static result, ML prediction/probability, detected patterns, evidence with source lines, explanation, optimization suggestions, and UNKNOWN reason. Keep the interface focused and readable. Call the existing pipeline/API rather than duplicating analysis logic. Do not add login, database, React, cloud deployment, or decorative features. Verify the app starts from the documented command and handles errors gracefully.
```

### Day 20 — Frozen-test experiments

```text
Create one reproducible command that evaluates the frozen test set exactly once under the preselected configurations. Produce S1 static-only, S2 ML-only, S3 naive hybrid, and S4 verified hybrid/abstention metrics; include classification metrics, coverage/selective risk, calibration where supported, explanation-verifier results, and optimization recommendation validity. Do not tune after viewing test results. Save the exact config, seed, test-manifest hash, and outputs. Make tables/plots from saved result files only.
```

### Day 21 — Freeze the engine

```text
Run the full test suite and the planned evaluation pipeline. Freeze feature definitions, labels, model artifacts, fusion rules, thresholds, and test manifest. Create a v0.9 tag or equivalent checkpoint and record any known limitations. Fix only correctness/blocking bugs; no new patterns or features after today. Verify that a clean checkout can run tests and the demo. Summarize the remaining Week 4 hardening tasks.
```

### Day 22 — Full integration

```text
Integrate parser, IR, static engine, feature extraction, saved ML model, fusion/abstention, evidence-based explanation, optimization rules, CLI, API, and Streamlit. Use one shared pipeline and avoid duplicated business logic. Add end-to-end tests for representative easy, nested-loop, logarithmic, sorting/searching, recursion, and UNKNOWN cases. Do not change frozen rules or thresholds just to make a demo example look better. Show the exact clean-start commands.
```

### Day 23 — Test hardening and hostile inputs

```text
Review test coverage and add edge cases for malformed C++, empty input, unsupported constructs, deeply nested input, oversized source, parser timeouts, missing model artifacts, and API errors. Confirm submitted code is never executed. Fix bugs without broadening the scope. Run all tests, report failures and coverage if available, and add regression tests for each bug fixed.
```

### Day 24 — Final dataset audit

```text
Run the final dataset audit: schema, independent labels/derivations, class counts, family counts, duplicate/similarity checks, split leakage, source hashes, and frozen test manifest. Investigate every validation warning; do not suppress warnings without explanation. Confirm that no test family appears in train or validation. Save a concise audit output and update docs/DATASET.md with the actual final counts, not planned counts.
```

### Day 25 — Reproducibility from a clean state

```text
Follow the documented setup in a clean environment or clean checkout. Run tests, feature generation, training only if the workflow requires it, and the evaluation command using frozen configs and seeds. Verify that the frozen test manifest hash and expected output files match. Fix missing setup instructions or hidden local-path dependencies. Do not retune models or thresholds from final test results. Record the exact reproducible commands.
```

### Day 26 — UI/API polish

```text
Improve only usability blockers in the Streamlit page and API: line-numbered evidence, clear distinction between static and ML predictions, confidence/status wording, UNKNOWN reason, actionable parser errors, and empty/loading/error states. Ensure the UI never presents a recommendation as guaranteed correct. Avoid new architecture or optional features. Test the main demo examples from the UI and endpoint.
```

### Day 27 — Performance and stability

```text
Run repeated end-to-end analyses on a small representative fixture set. Measure parser/analysis latency and API response time where practical; record environment and method. Check for crashes, state leakage between requests, unbounded input handling, and missing-file failures. Fix blocking stability problems only. Do not claim performance improvements without measurements or add cloud deployment.
```

### Day 28 — Technical documentation only

```text
Write concise implementation documentation, not a research paper or academic report. Update README.md with purpose, supported scope, setup, quickstart, demo examples, and limitations. Add/update docs/ARCHITECTURE.md, DATASET.md, EXPERIMENTS.md, LIMITATIONS.md, PROGRESS.md, and DECISIONS.md. Document actual results and commands; do not invent metrics. Do not create a literature review, related-work section, citation list, paper chapters, or paper formatting.
```

### Day 29 — Release cleanup

```text
Do a release-readiness pass: remove dead code and debug artifacts, check secrets are not committed, verify .gitignore, ensure model/data paths are portable, pin or document dependencies, and make the demo dataset available. Run lint/tests and a clean-start smoke test. Fix only release blockers. Add a short release checklist and tag the final version. Do not add new features.
```

### Day 30 — Demo and viva rehearsal

```text
Prepare the project for a live college demo. Verify the clean-start commands, choose 3–5 representative C++ examples including one UNKNOWN case, and ensure outputs show complexity, evidence, explanation, and any valid optimization suggestion. Create a short demo runbook and backup plan. Generate likely viva questions and concise answers about the parser, static analysis, ML model, hybrid gate, abstention, dataset split, metrics, limitations, and why the project is AI/ML. Do not write a research paper or academic report. Run the demo one final time and report pass/fail for every checklist item.
```

### If a day fails — universal recovery prompt

```text
Do not continue to the next day's feature. Inspect the failing command/test and identify the smallest root cause. Make the smallest targeted fix, add a regression test, rerun the relevant test and the full suite, and show the changed files and exact outputs. Do not rewrite working modules, silently relax tests, invent successful results, add dependencies without need, or expand scope. If the planned parser/model dependency is blocked, report the evidence and use the fallback already defined in the plan.
```

### Required response format from Antigravity after every prompt

Ask the agent to finish every daily task with:
1. Files created/changed.
2. Design decisions and assumptions.
3. Commands actually run and their real outputs.
4. Tests passed/failed (never claim tests were run if they were not).
5. Known limitations/blockers.
6. Exact completion-gate status and what must be shown before moving to the next day.

