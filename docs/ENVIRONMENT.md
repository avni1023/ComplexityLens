# Environment and Dependency Notes

## Target Runtime
- **Python**: 3.10+ (tested on Python 3.13.14 on Windows 64-bit)
- **Virtual Environment**: `.venv/` at workspace root

## Dependencies

### Core Development
- `pytest` (>= 8.0.0): Unit and golden test runner.
- `ruff` (>= 0.3.0): Fast Python linter and formatter.
- `setuptools` (>= 61.0.0): Build system for editable installs (`pip install -e .`).

### Planned Stack (Upcoming Days)
- **Parser (Days 1–3)**: `libclang` (Python bindings + LLVM Clang shared libraries).
  - *Fallback*: `tree-sitter` + `tree-sitter-cpp` if `libclang` C-bindings fail on Windows.
- **Machine Learning (Days 10–12)**: `numpy`, `scipy`, `scikit-learn`, `xgboost`.
- **API & UI (Days 18–19)**: `fastapi`, `uvicorn`, `streamlit`, `pydantic`.

## Commands Reference

### Windows (PowerShell)
```powershell
# Run lint and test suite
.\run_checks.ps1

# Direct pytest
.\.venv\Scripts\pytest.exe

# Direct ruff lint
.\.venv\Scripts\ruff.exe check .
```

### Linux / WSL / macOS
```bash
# Run lint and test suite
make check

# Direct pytest
make test

# Direct ruff lint
make lint
```
