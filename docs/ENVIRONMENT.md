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

## C++ Standard-Library Header Configuration (CXX_INCLUDE_DIRS)

`libclang==18.1.1` from PyPI ships only `libclang.dll` and does not bundle C++ standard-library headers (`<vector>`, `<algorithm>`, `<unordered_map>`) or Clang compiler resource headers.

To enable full STL parsing in libclang without hardcoded paths or toolchain executables, configure the `CXX_INCLUDE_DIRS` environment variable.

### MSYS2 MinGW-w64 / UCRT Setup Steps

1. **Install MSYS2**:
   Download and install MSYS2 from https://www.msys2.org/ (default: `C:\msys64`).

2. **Install UCRT64 Toolchain Headers**:
   In the MSYS2 UCRT64 terminal:
   ```bash
   pacman -Syu
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-headers
   ```

3. **Configure `CXX_INCLUDE_DIRS`**:
   Set `CXX_INCLUDE_DIRS` to the semicolon-separated list of verified MSYS2 UCRT64 include directories:
   ```powershell
   # In PowerShell (current session):
   $env:CXX_INCLUDE_DIRS = "C:\msys64\ucrt64\include\c++\16.2.0;C:\msys64\ucrt64\include\c++\16.2.0\x86_64-w64-mingw32;C:\msys64\ucrt64\include;C:\msys64\ucrt64\lib\gcc\x86_64-w64-mingw32\16.2.0\include;C:\msys64\ucrt64\lib\gcc\x86_64-w64-mingw32\16.2.0\include-fixed"

   # Or persist at User level (permanent):
   [System.Environment]::SetEnvironmentVariable('CXX_INCLUDE_DIRS', 'C:\msys64\ucrt64\include\c++\16.2.0;C:\msys64\ucrt64\include\c++\16.2.0\x86_64-w64-mingw32;C:\msys64\ucrt64\include;C:\msys64\ucrt64\lib\gcc\x86_64-w64-mingw32\16.2.0\include;C:\msys64\ucrt64\lib\gcc\x86_64-w64-mingw32\16.2.0\include-fixed', 'User')
   ```

4. **Running Tests**:
   - Automated check runner (automatically detects and sets `CXX_INCLUDE_DIRS` if MSYS2 is present):
     ```powershell
     .\run_checks.ps1
     ```
   - Direct focused pytest execution:
     ```powershell
     .\.venv\Scripts\python.exe -m pytest tests/test_cpp_libclang_spike.py -v
     ```

5. **Verification & Test Counts**:
   - When `CXX_INCLUDE_DIRS` is configured: **61 passed, 0 skipped, 0 failed**.
   - When `CXX_INCLUDE_DIRS` is unset: **54 passed, 7 skipped, 0 failed** (STL fixtures cleanly skip).
   - `cpp_libclang.py` detects MinGW/UCRT include paths and aligns the target triple (`-target x86_64-w64-mingw32`) to prevent MSVC runtime macro conflicts.


