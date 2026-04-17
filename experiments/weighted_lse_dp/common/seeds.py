"""Seed policy for the weighted-LSE DP experimental program.

Implements the seed contract from the Phase I/III specs (§11.1):

* The canonical "main results" seed list is ``(11, 29, 47)``.
* A 10-seed cheap audit extends the main list to
  ``(11, 29, 47, 53, 67, 79, 83, 97, 101, 113)``. The audit is MANDATORY
  for the combinations in :data:`AUDIT_REQUIRED` (currently
  ``chain_base`` paired with ``QLearning`` or ``ExpectedSARSA``); it is
  best-effort for every other (task, algorithm) pair.

Only the stdlib ``random`` / ``os`` modules and ``numpy`` are used here.
Torch is intentionally NOT imported: downstream agents that need torch
seeding must do it themselves after calling :func:`seed_everything`.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = [
    "MAIN_SEEDS",
    "AUDIT_SEEDS",
    "AUDIT_REQUIRED",
    "get_seeds",
    "seed_everything",
    "validate_seed",
]


# ---------------------------------------------------------------------------
# Seed constants
# ---------------------------------------------------------------------------

#: Canonical three-seed list used for every main-result run.
MAIN_SEEDS: tuple[int, ...] = (11, 29, 47)

#: Ten-seed audit list. The first three entries MUST equal
#: :data:`MAIN_SEEDS` so that audit runs strictly extend the main runs.
AUDIT_SEEDS: tuple[int, ...] = (11, 29, 47, 53, 67, 79, 83, 97, 101, 113)

#: (task, algorithm) pairs where the 10-seed audit is mandatory. For every
#: other pair the audit is best-effort and callers fall back to
#: :data:`MAIN_SEEDS`.
AUDIT_REQUIRED: frozenset[tuple[str, str]] = frozenset(
    {
        ("chain_base", "QLearning"),
        ("chain_base", "ExpectedSARSA"),
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_seeds(
    task: str,
    algorithm: str,
    *,
    audit: bool = False,
) -> tuple[int, ...]:
    """Return the seed list to use for ``(task, algorithm)``.

    Parameters
    ----------
    task:
        Task identifier, e.g. ``"chain_base"`` or ``"grid_base"``. The
        identifier must match the task key used throughout the config
        directory.
    algorithm:
        Algorithm identifier, e.g. ``"QLearning"`` or ``"ExpectedSARSA"``.
    audit:
        When ``True`` and ``(task, algorithm)`` is listed in
        :data:`AUDIT_REQUIRED`, return :data:`AUDIT_SEEDS`. Otherwise
        return :data:`MAIN_SEEDS`. The default is ``False`` so that
        callers who never opt in always receive the canonical three
        seeds.

    Returns
    -------
    tuple[int, ...]
        Immutable tuple of seeds (non-negative ints), in the order they
        should be consumed.
    """
    if audit and (task, algorithm) in AUDIT_REQUIRED:
        return AUDIT_SEEDS
    return MAIN_SEEDS


def seed_everything(seed: int) -> None:
    """Seed Python ``random``, NumPy's legacy global RNG, and ``PYTHONHASHSEED``.

    This covers every RNG that might be touched implicitly by library
    code in this repo. Torch and Gymnasium seeding is intentionally
    NOT handled here: the caller (typically the runner) must seed those
    explicitly so that the seed it records in ``run.json`` matches the
    seed each simulator actually used.

    Parameters
    ----------
    seed:
        Non-negative integer seed. Validated via :func:`validate_seed`.

    Notes
    -----
    Setting ``PYTHONHASHSEED`` after the interpreter has started only
    affects *child* processes that are spawned later; the current
    interpreter's hash randomization is fixed at startup. We still set
    it so that subprocess-based workers inherit a deterministic hash
    seed.
    """
    validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def validate_seed(seed: int) -> None:
    """Raise ``ValueError`` if ``seed`` is not a non-negative integer.

    Booleans are rejected because silently accepting ``True``/``False``
    as ``1``/``0`` masks config-loading bugs.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"seed must be a non-negative int, got {type(seed).__name__}={seed!r}"
        )
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
