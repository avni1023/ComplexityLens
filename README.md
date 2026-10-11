# ComplexityLens

*An Explainable Hybrid Static Analysis and Machine Learning System for C++ Complexity Analysis*

ComplexityLens analyzes small C++ programs ($\le 200$ lines, single translation unit) and determines their worst-case time complexity using a hybrid pipeline combining deterministic static rules and calibrated machine learning with verification and abstention.

## Supported Taxonomy
- Determinable classes: `O(1)`, `O(log n)`, `O(n)`, `O(n log n)`, `O(n²)`, `O(n³)`, `O(2ⁿ)`
- Abstention: `UNKNOWN` (with structured reason codes)

## Project Structure
```text
ComplexityLens/
├── data/
│   └── templates/           # Dataset authoring templates (YAML / CSV)
├── docs/
│   ├── DECISIONS.md         # Architecture decision records
│   ├── ENVIRONMENT.md       # Environment, tooling, and dependency notes
│   └── PROGRESS.md          # Implementation progress tracking
├── src/
│   └── complexity_lens/     # Core package
├── tests/                   # Pytest test suite
├── Makefile                 # POSIX test and lint runner
├── run_checks.ps1           # Windows PowerShell test and lint runner
├── pyproject.toml           # Project metadata and tool configuration
└── requirements-dev.txt     # Frozen development dependencies
```

## Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.13)

### Setup
```powershell
# Create virtual environment and install dev dependencies
python -m venv .venv
.\.venv\Scripts\pip.exe install -e .
.\.venv\Scripts\pip.exe install -r requirements-dev.txt
```

### Running Tests & Linting

**On Windows (PowerShell):**
```powershell
.\run_checks.ps1
```

**On Linux / macOS:**
```bash
make check
```

## Scope & Limitations
- **Time Complexity Only**: Space complexity is explicitly out of scope for the MVRS.
- **Input Scope**: Single `.cpp` source files, $\le 200$ LOC, single translation unit without external build systems.
- **Safety**: Analyzed code is parsed strictly as data and is never compiled or executed.
