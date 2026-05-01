"""Pytest configuration for the Phase VII-B strategic-games test suite.

Registers the ``slow`` marker so the runner-smoke / reproducibility tests
that take more than ~1s of wall-clock can be filtered with
``pytest -m "not slow"``.
"""

from __future__ import annotations


def pytest_configure(config):  # type: ignore[no-untyped-def]
    """Register custom markers used by this test subdirectory."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests that exercise full runner / repro workflows "
        "(deselect with `-m \"not slow\"`).",
    )
    config.addinivalue_line(
        "markers",
        "smoke: marks falsifiable-prediction smoke tests that run a small "
        "training sweep (deselect with `-m \"not smoke\"` under tight CI "
        "budget; included in milestone-close sweeps).",
    )
