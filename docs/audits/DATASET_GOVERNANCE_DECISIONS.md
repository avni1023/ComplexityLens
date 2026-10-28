# Dataset Governance Decisions

**Document Type:** Dataset Governance Record  
**Date:** 2026-10-27  
**Status:** Approved Policy Record  
**Branch Context:** `feature/day-7-dataset-tooling`  
**Governing Repository:** `ComplexityLens`  

> [!IMPORTANT]
> **Scope & Authority Notice:**  
> This record documents the user-approved dataset governance policies governing the repository's family plans, benchmark datasets, and evaluation scope. This document is a governance and audit record only. It **does not authorize** any modifications to dataset files, source code, gold labels, program metadata, or Git history. It **does not create, stage, or commit** any files, nor does it implement automated filtering or split generation.

---

## 1. User-Approved Policy Decisions

The project author has formally approved five governing policies:

### Policy 1: Canonical Family Plan Designation
* **Canonical Specification:** `docs/FAMILY_PLAN.md` is designated as the sole canonical family-plan document for the project.
* **Duplicate Status:** The repository root file `FAMILY_PLAN.md` is an exact duplicate of `docs/FAMILY_PLAN.md`.
* **Disposal Policy:** The root duplicate may be removed in a future, explicitly authorized documentation cleanup task. It is retained unchanged in the current working state.

### Policy 2: Historical Legacy Benchmark Preservation
* **Baseline Preservation:** The 10 legacy families (`datasets/families/fam_*.yaml`) and their 38 associated C++ programs (`datasets/programs/fam_*/*.cpp`) represent the historical Week 1 baseline benchmark committed in Phase A (commit `f23df075...`).
* **Non-Modification Policy:** All legacy YAML metadata files, program sources, SHA-256 hashes, rationales, and historical labels must be preserved without alteration or deletion.

### Policy 3: Primary Phase 2 Benchmark Designation
* **Primary Benchmark Scope:** The 68 new families (`datasets/families/f01.yaml` through `f68.yaml`) and their 204 associated C++ source programs (`datasets/programs/f01/` through `f68/`) constitute the primary Phase 2 benchmark dataset for ComplexityLens.
* **Taxonomic Independence:** The Phase 2 benchmark must remain logically distinct from the legacy baseline.

### Policy 4: Training and Evaluation Split Isolation
* **Default Isolation Policy:** The legacy baseline dataset (10 families, 38 programs) is excluded by default from the primary benchmark training, validation, and testing splits.
* **Rationale:** The legacy families and the 68 new families contain overlapping algorithmic concepts (e.g., merge sort, Collatz termination, logarithmic binary reduction). Pooling them without isolation risks train/test leakage and distorted class distributions.
* **Implementation Boundary:** This policy defines the default evaluation scope only. It does not implement any filtering flags, dataset partitioning tools, or split generation logic within this task.

### Policy 5: Euclidean GCD Historical Discrepancy Preservation
* **Preserved Historical Annotations:** Both historical labels for the Euclidean greatest common divisor algorithm are preserved without modification:
  1. **Legacy Baseline:** Program `unknown_euclidean_gcd` in `datasets/families/fam_unknown_external_unsupported.yaml` retains its historical label `UNKNOWN` with reason code `data_dependent` and explanation citing numerical magnitude dependence.
  2. **Phase 2 Benchmark:** Family `datasets/families/f56.yaml` (`datasets/programs/f56/*.cpp`) retains its ground-truth label `OLOGN` with tag `rule_hard` and rationale citing geometric shrinkage of the remainder sequence.
* **Unresolved Status:** The discrepancy between these two perspectives remains an unresolved historical difference.
* **Integrity Guard:** Neither label is chosen as the exclusive canonical definition at this time. No gold labels, rationales, or metadata fields may be altered to force agreement.

---

## 2. Supporting Evidence from Read-Only Investigation

The approved policies are grounded in empirical evidence gathered during the read-only repository inspection conducted on 2026-10-27:

### 1. Verification of the Family Plan Duplicate
* **Cryptographic Equivalence:**
  * `FAMILY_PLAN.md` (root): 9,326 bytes | SHA-256: `816E6501C9CD64D30584882C3F3770ECB0E3D4883553A8752A32BB6411E595AC`
  * `docs/FAMILY_PLAN.md`: 9,326 bytes | SHA-256: `816E6501C9CD64D30584882C3F3770ECB0E3D4883553A8752A32BB6411E595AC`
* **Architectural Precedent:** `README.md` lines 16–19 establish `docs/` as the exclusive home for project architecture, decision records, and technical documentation (`DECISIONS.md`, `DATASET.md`, `ENVIRONMENT.md`, etc.). Line 7 of `FAMILY_PLAN.md` instructs copying conventions into `docs/DATASET.md`.

### 2. Composition of the 78-Family Working Tree
* **File Counts:** 320 total files exist across `datasets/`:
  * 10 legacy YAMLs + 38 legacy `.cpp` programs (tracked in Git).
  * 68 new YAMLs + 204 new `.cpp` programs (untracked; backed up externally).
  * 0 unexpected files, binaries, or stray extensions.
* **Macro-Families vs. Fine-Grained Families:**
  * The 10 legacy families are *composite macro-families*. For example, `fam_binary_search` bundled four separate mechanisms (`while(n>1) n/=2`, `for(i*=2)`, iterative binary search, and `std::lower_bound`).
  * In the Phase 2 plan (`docs/FAMILY_PLAN.md`), these four mechanisms were separated into dedicated single-concept families: `f07`, `f08`, `f09`, and `f12`.
  * Similarly, `fam_quadratic_nested` spanned what became `f30`, `f31`, `f32`, and `f36`.
  * `fam_rule_hard_divide_conquer` and `f24` both represent recursive range-based merge sort ($T(n) = 2T(n/2) + O(n)$).
* **File-Level Uniqueness:** All 242 `.cpp` files in the repository have unique paths, names, and source contents. There is zero file-level duplication, even where algorithmic concepts overlap.

### 3. Euclidean GCD Implementation Comparison
* **Legacy Implementation (`datasets/programs/fam_unknown_external_unsupported/unknown_euclidean_gcd.cpp`):**
  ```cpp
  int compute_gcd(int a, int b) {
      while (b != 0) {
          int temp = b;
          b = a % b;
          a = temp;
      }
      return a;
  }
  ```
  * Annotated as `UNKNOWN` (`data_dependent`) based on the reasoning that step counts depend on the specific numerical values of `a` and `b`.
* **Phase 2 Implementation (`datasets/programs/f56/f56_gcd_euclid_while.cpp`):**
  ```cpp
  int gcd_euclid_while(int a, int b) {
      while (b != 0) {
          int tmp = b;
          b = a % b;
          a = tmp;
      }
      return a;
  }
  ```
  * Annotated as `OLOGN` (`rule_hard`) based on Gabriel Lamé's theorem that remainder values decrease by at least a factor of 2 every two iterations, establishing an asymptotic upper bound of $O(\log(\min(a, b)))$.

---

## 3. Unresolved Technical Questions

The following technical questions are explicitly identified and deferred:

1. **Theoretical Resolution of Euclidean GCD:**
   * Does the project consider input parameters `a` and `b` as numerical magnitudes ($n = \max(a, b)$) or as fixed bit-width integers?
   * How should the static engine's single-symbol input model ($n$) represent two-argument arithmetic reductions?
   * A future technical review must analyze this question without modifying existing historical annotations.
2. **Formal Role of Legacy Benchmark:**
   * Should the 10 legacy families serve exclusively as an invariant regression test suite for the static analyzer, or will they be used as an out-of-distribution transfer test set for the ML models?
3. **Program-Level Similarity Metrics:**
   * Before any future cross-dataset experiments, an automated token-based or AST-based similarity audit should quantify the exact distance between legacy programs and their `f01`–`f68` counterparts.

---

## 4. Future Actions Requiring Separate Authorization

The following actions are **out of scope** for this task and must not be performed without explicit future authorization:

* Removing the root duplicate `FAMILY_PLAN.md`.
* Modifying `docs/FAMILY_PLAN.md` or any existing documentation in `docs/`.
* Modifying any YAML family definition or C++ source program.
* Implementing dataset filtering CLI flags or split generation scripts.
* Modifying the test suite or tooling files.
* Staging, committing, or pushing any changes to Git.
