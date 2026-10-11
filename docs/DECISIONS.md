# Project Decisions & Architecture Records (ADR)

## ADR-001: Minimal Python Project Structure & Tooling (Day 0)
* **Date**: 2026-10-11
* **Status**: Accepted
* **Context**: ComplexityLens requires a deterministic, lightweight Python development environment without extraneous frameworks or unnecessary overhead.
* **Decision**:
  - Use `pyproject.toml` with `setuptools` build backend.
  - Standardize source code under `src/complexity_lens/` to avoid top-level import ambiguity.
  - Use `pytest` for testing and `ruff` for linting.
  - Provide both `Makefile` (cross-platform / CI) and `run_checks.ps1` (native Windows PowerShell).
  - Scope out all deep learning, databases, Docker, and academic paper tools.
* **Consequences**: Fast startup, zero global pip pollution, fully reproducible local development.
