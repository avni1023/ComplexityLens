# ComplexityLens Prototype Web Interface (`app.py`)

## 1. Overview

ComplexityLens provides an interactive Streamlit-based web dashboard allowing users to submit C++ algorithmic source code, execute hybrid static and machine-learning time complexity verification, and inspect structured evidence, rule derivation traces, and confidence metrics.

The application connects directly to the core analysis boundary (`complexity_lens.pipeline.analyze_source_pipeline`) and employs a deterministic fusion gate (`complexity_lens.fusion.decide_fusion`).

---

## 2. Launch Instructions

To launch the prototype bound strictly to `localhost` with usage statistics disabled, execute the following command in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.headless=true --server.port=8501 --browser.gatherUsageStats=false
```

Once running, navigate to `http://127.0.0.1:8501` in your web browser.

### Configuration File (`.streamlit/config.toml`)
The repository includes a dedicated `.streamlit/config.toml` that enforces localhost-only operation and a clean light theme:
```toml
[theme]
primaryColor = "#1a73e8"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1e293b"
font = "sans serif"

[server]
headless = true
address = "127.0.0.1"
port = 8501

[browser]
gatherUsageStats = false
```

---

## 3. Security and Safe Rendering Architecture

1. **Localhost Isolation**: The server binds exclusively to `127.0.0.1`, preventing exposure to external networks.
2. **Safe Component Rendering**:
   - User-derived text (submitted C++ source, parser diagnostics, exception messages, and rule traces) is **never passed** to `st.markdown(..., unsafe_allow_html=True)`.
   - Source code is displayed via `st.code()`, structured data via `st.dataframe()`, and dynamic text via `st.write()` and `st.metric()`.
   - `unsafe_allow_html=True` is strictly restricted to static developer CSS containing zero dynamic interpolations.
3. **Execution Safety**:
   - The engine **never compiles or executes** submitted C++ source code.
   - An automated safety regression test (`test_static_analysis_safety_regression_no_compilation`) patches `subprocess.run`, `subprocess.Popen`, `subprocess.call`, and `os.system` during code analysis to guarantee that code execution primitives are never invoked.

---

## 4. Built-in Demonstration Snippets & Actual Engine Outputs

All demonstration snippets were written independently outside the frozen dataset and verified against the live engine:

| Demonstration Snippet | Static Label | Static Exactness | ML Estimate (RF) | Fusion Verdict | Active Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Loop O(n)** | `ON` | `STRUCTURAL` | `ON` (78.3%) | **O(n)** | `STATIC` |
| **Constant Time O(1)** | `O1` | `STRUCTURAL` | `O1` (92.5%) | **O(1)** | `STATIC` |
| **Nested Loops O(n²)** | `ON2` | `STRUCTURAL` | `ON2` (74.8%) | **O(n²)** | `STATIC` |
| **Logarithmic Loop O(log n)** | `OLOGN` | `STRUCTURAL` | `OLOGN` (65.2%) | **O(log n)** | `STATIC` |
| **Exponential Recursion O(2ⁿ)**| `OEXP` | `STRUCTURAL` | `OEXP` (70.4%) | **O(2ⁿ)** | `STATIC` |
| **Static UNKNOWN with ML Fallback** (`__asm__`) | `UNKNOWN` | `UNKNOWN` | `O1` (88.1%) | **O(1)** | `ML_ESTIMATE` |
| **Inconclusive Collatz Loop** | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` (45.5%)| **UNKNOWN** | `UNKNOWN` |

---

## 5. Technical Disclosures & Limitations

The prototype explicitly discloses the following system limitations in both the UI and documentation:

1. **Auxiliary Space Complexity**: **NOT IMPLEMENTED**. ComplexityLens currently analyzes **time complexity only**.
2. **Parser Execution Timeout**: Abstract Syntax Tree parsing through `libclang` does not enforce an execution timeout.
3. **Uncalibrated ML Confidence**: Classifier probability outputs reflect heuristic decision scores and are **not calibrated probabilities**.
4. **Model Limitation Notice**:
   > *"The ML model was developed using development-partition programs. A prediction for an individual input is not evidence of model accuracy."*
5. **Development CV Metrics**: Metrics shown in the Evaluation tab reflect cross-validation selection-biased estimates across development partition folds. The **frozen test set has NOT been evaluated**.

---

## 6. Verification and Testing

### Test Suite Execution
Run the complete test suite:
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_smoke.py tests/test_fusion_and_pipeline.py tests/test_ui.py -v
```
**Outcome**: 25 passed in 43.11s.

### Verification Categories
- **Automated Unit & Integration Tests**: Executed and verified (25 tests covering smoke, fusion rules, pipeline, UI components, safety regressions, and disclosures).
- **Application Startup Verification**: Verified CLI invocation (`streamlit config show` and process startup).
- **Browser Testing**: **Not claimed or performed**. No live headless or browser session was attached; verification relied on automated API and CLI process tests.

