# ComplexityLens — Final Project Evaluation & Demonstration Checklist

**Project Title**: Explainable Hybrid Static-Analysis and ML Estimation of Time Complexity for C++ Programs  
**Repository**: `C:\Users\Admin\OneDrive\Documents\ComplexityLens`  
**Evaluation Scope**: Worst-case time complexity analysis only. Auxiliary-space complexity is not implemented.

---

## 1. Quick Launch Commands

To start the interactive Streamlit web dashboard:
```powershell
# From repository root
.\.venv\Scripts\streamlit.exe run app.py
```
Or via Python module:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To run the command-line analysis tool:
```powershell
# Text analysis report
.\.venv\Scripts\python.exe -m complexity_lens.cli analyze tests/fixtures/cpp/02_for_loop.cpp

# JSON output
.\.venv\Scripts\python.exe -m complexity_lens.cli analyze tests/fixtures/cpp/02_for_loop.cpp --json
```

---

## 2. Recommended Demonstration Sequence

During evaluation, present the following representative cases to showcase the hybrid architecture:

| # | Demo Case | Source Pattern | Expected Big-O | Architecture Mechanism Demonstrated |
|---|---|---|---|---|
| **1** | **Constant Arithmetic** | Multiple math operations, no loops | `O(1)` | AST basic block inspection; constant work rule. |
| **2** | **Linear Loop** | `for (int i = 0; i < n; ++i)` | `O(n)` | Structural loop bounds; `LINEAR_INPUT` bound detection. |
| **3** | **Logarithmic Halving** | `while (n > 0) { count++; n /= 2; }` | `O(log n)` | Multiplicative halving update recognition. |
| **4** | **Nested Loops** | `for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)` | `O(n²)` | Nested construct multiplication rule. |
| **5** | **Fixed-Range STL** | `const int length = 5; std::find(arr, arr + length, 42)` | `O(1)` | Local constant resolution; recognized STL call over fixed bound. |
| **6** | **Input-Scale STL** | `std::sort(arr, arr + n)` | `O(n log n)` | Input symbol range resolution; STL algorithm complexity classification. |
| **7** | **Honest Abstention** | Collatz sequence or data-dependent break | `Unknown` (`UNKNOWN`) | Static engine abstains; ML fallback detects unknown bounds; conservative abstention gate. |

---

## 3. Scope & Known Limitations

1. **Time Complexity Focus**: The system estimates worst-case asymptotic time complexity ($O(1)$ to $O(2^n)$). Space complexity analysis is explicitly out of scope for this prototype.
2. **Translation Unit Scope**: Single translation unit $\le 200$ LOC, adhering strictly to safe C++ subset rules. External libraries and OS system calls are rejected by validation.
3. **Safety Guarantee**: Submitted code is parsed strictly as data using libclang AST. Code is never compiled or executed.
4. **Machine Learning Calibration**: ML class probabilities are raw tree-based probabilities from Random Forest / XGBoost classifiers and are uncalibrated.
5. **Frozen Test Holdout**: Reported evaluation metrics represent 3-fold grouped cross-validation on the development partition (159 programs, 53 families). The frozen test partition (83 programs) remains untouched for final evaluation.

---

## 4. Troubleshooting & Recovery Steps

- **Environment Variable**: Ensure `CXX_INCLUDE_DIRS` is set if analyzing STL headers (e.g. MinGW headers in `C:\msys64\ucrt64\include\c++\16.2.0`).
- **Model Loader Failures**: If model loading fails, verify `artifacts/models/random_forest_dev_fit.joblib` and `artifacts/models/random_forest_metadata.json` exist. The pipeline gracefully falls back to static-only mode if model loading is disabled or fails integrity checks.
- **Port Conflict**: If port 8501 is busy, specify `--server.port=8502`:
  ```powershell
  .\.venv\Scripts\streamlit.exe run app.py --server.port=8502
  ```

