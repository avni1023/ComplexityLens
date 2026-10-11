.PHONY: test lint check clean help

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PYTEST ?= $(VENV)/bin/pytest
RUFF ?= $(VENV)/bin/ruff

help:
	@echo "Available commands:"
	@echo "  make test   - Run test suite via pytest"
	@echo "  make lint   - Run ruff linter"
	@echo "  make check  - Run both lint and tests"
	@echo "  make clean  - Remove build and cache files"

test:
	$(PYTEST)

lint:
	$(RUFF) check .

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info `find . -name '__pycache__'`
