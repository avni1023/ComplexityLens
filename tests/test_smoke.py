"""Smoke test verifying project package import and test runner setup."""

import complexity_lens


def test_package_import():
    """Verify complexity_lens package imports and exposes version."""
    assert hasattr(complexity_lens, "__version__")
    assert complexity_lens.__version__ == "0.1.0"


def test_environment_smoke():
    """Basic smoke assertion for test harness verification."""
    assert 1 + 1 == 2
