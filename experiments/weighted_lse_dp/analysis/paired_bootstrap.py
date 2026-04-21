#!/usr/bin/env python
"""Phase IV-B: shared paired bootstrap confidence interval helper.

Exported API
------------
paired_bootstrap_ci(a, b, n_bootstrap=10000, ci=0.95, seed=0)
    -> (lower, upper, mean_diff)

`a` and `b` are 1-D arrays of paired observations (same seed order).
The CI is on the mean of (a - b) computed via percentile bootstrap.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for mean(a - b) under paired observations.

    Parameters
    ----------
    a, b:
        1-D float arrays of equal length.  Observations are paired by index
        (same seed produces one element in each array).
    n_bootstrap:
        Number of bootstrap resamples.  Default: 10 000 (spec §Q10).
    ci:
        Coverage level.  Default: 0.95.
    seed:
        NumPy RNG seed for reproducibility.

    Returns
    -------
    (lower, upper, mean_diff)
        lower, upper: percentile bootstrap CI bounds at `ci` level.
        mean_diff: point estimate mean(a - b) on the original sample.

    Raises
    ------
    ValueError
        If a and b have different lengths or are empty.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            f"paired_bootstrap_ci: a and b must have equal shape; "
            f"got {a.shape} vs {b.shape}"
        )
    if a.size == 0:
        raise ValueError("paired_bootstrap_ci: empty input arrays")

    # Paired differences — same index means same seed (spec §Q7)
    diffs: np.ndarray = a - b
    mean_diff: float = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)

    # Resample with replacement; compute bootstrap means
    idxs = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means: np.ndarray = diffs[idxs].mean(axis=1)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return lower, upper, mean_diff
