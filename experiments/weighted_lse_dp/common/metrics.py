"""Scalar / curve aggregation utilities for the weighted-LSE DP program.

Implements the summary statistics required by the Phase I/II/III spec
(§9.1 "Reporting across seeds", §9.2 DP metrics, §9.3 RL metrics, §9.4
calibration metrics).

The module is intentionally small and dependency-light: ``numpy`` only.
No scipy, no matplotlib, no torch. All public functions are side-effect
free and suitable for direct use by the ``table-maker`` aggregator and
by ``plotter-analyst`` via ``results/processed/``.

Notes
-----
The bootstrap confidence interval is a plain percentile bootstrap over
the seed axis. We use a caller-supplied ``numpy.random.Generator`` so
that aggregation is deterministic given an upstream seed policy; if
``None`` is passed we construct a fresh default generator (still
deterministic per process because numpy's SeedSequence is fixed).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "aggregate",
    "curve_auc",
    "final_performance",
    "steps_to_threshold",
    "sweep_to_tolerance",
    "sup_norm",
    "margin_quantiles",
    "aligned_margin_freq",
    "success_rate",
]


# ---------------------------------------------------------------------------
# Seed-axis aggregation (spec §9.1)
# ---------------------------------------------------------------------------


def aggregate(
    values: np.ndarray,
    axis: int = 0,
    ci_level: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[str, float | np.ndarray]:
    """Summarise ``values`` across the seed axis.

    Computes the eight statistics required by the spec for every main
    (task, algorithm) pair: ``mean``, ``std`` (sample standard
    deviation, ``ddof=1``), ``median``, ``q25``, ``q75``, ``iqr``,
    ``ci_low``, ``ci_high``. ``ci_low``/``ci_high`` are the
    ``(1 - ci_level) / 2`` and ``(1 + ci_level) / 2`` percentiles of
    the bootstrap distribution of the mean.

    Parameters
    ----------
    values:
        Either a 1-D array of shape ``(n_seeds,)`` — one scalar per
        seed — or a 2-D array of shape ``(n_seeds, T)`` — one learning
        curve per seed. The aggregation is always along ``axis``.
    axis:
        Axis along which to aggregate. Defaults to ``0`` (the seed
        axis).
    ci_level:
        Two-sided confidence level, e.g. ``0.95`` for a 95% CI. Must
        be in ``(0, 1)``.
    n_bootstrap:
        Number of bootstrap resamples used for the CI on the mean.
    rng:
        NumPy ``Generator``. When ``None``, a fresh default generator
        is instantiated.

    Returns
    -------
    dict[str, float | numpy.ndarray]
        Keys ``mean``, ``std``, ``median``, ``q25``, ``q75``, ``iqr``,
        ``ci_low``, ``ci_high``. For 1-D inputs every value is a
        Python ``float``. For 2-D inputs every value is an array whose
        shape equals ``values.shape`` with ``axis`` removed.
    """
    if not isinstance(values, np.ndarray):
        values = np.asarray(values)
    if values.ndim not in (1, 2):
        raise ValueError(
            f"aggregate expects 1-D or 2-D input, got ndim={values.ndim}"
        )
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if rng is None:
        rng = np.random.default_rng()

    n = values.shape[axis]
    if n < 2:
        raise ValueError(
            f"aggregate needs at least 2 samples along axis={axis}, got {n}"
        )

    mean = np.mean(values, axis=axis)
    std = np.std(values, axis=axis, ddof=1)
    median = np.median(values, axis=axis)
    q25 = np.quantile(values, 0.25, axis=axis)
    q75 = np.quantile(values, 0.75, axis=axis)
    iqr = q75 - q25

    # Percentile bootstrap on the mean along ``axis``.
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    # np.take(values, idx, axis=axis) inserts the index shape (n_bootstrap, n)
    # at position ``axis``, giving:
    #   axis=0, ndim=1: (n_bootstrap, n)
    #   axis=0, ndim=2: (n_bootstrap, n, T)
    #   axis=1, ndim=2: (n_seeds, n_bootstrap, n)
    # The sample dimension (n) always sits at axis+1; taking the mean there
    # reduces it and leaves the bootstrap dimension at position ``axis``.
    resampled = np.take(values, idx, axis=axis)
    boot_means = np.mean(resampled, axis=axis + 1)
    # boot_means has n_bootstrap at position ``axis``:
    #   axis=0 -> (n_bootstrap,) or (n_bootstrap, T)  → quantile over axis=0
    #   axis=1 -> (n_seeds, n_bootstrap)              → quantile over axis=1
    alpha = (1.0 - ci_level) / 2.0
    ci_low = np.quantile(boot_means, alpha, axis=axis)
    ci_high = np.quantile(boot_means, 1.0 - alpha, axis=axis)

    out: dict[str, float | np.ndarray] = {
        "mean": _to_float_if_scalar(mean),
        "std": _to_float_if_scalar(std),
        "median": _to_float_if_scalar(median),
        "q25": _to_float_if_scalar(q25),
        "q75": _to_float_if_scalar(q75),
        "iqr": _to_float_if_scalar(iqr),
        "ci_low": _to_float_if_scalar(ci_low),
        "ci_high": _to_float_if_scalar(ci_high),
    }
    return out


def _to_float_if_scalar(x: np.ndarray) -> float | np.ndarray:
    """Return a Python ``float`` if ``x`` is 0-D, else the array unchanged."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return arr


# ---------------------------------------------------------------------------
# RL learning-curve metrics (spec §9.3)
# ---------------------------------------------------------------------------


def curve_auc(returns: np.ndarray, dx: float = 1.0) -> float:
    """Area under a learning curve via the trapezoidal rule.

    Parameters
    ----------
    returns:
        1-D array of shape ``(T,)``. Typically a single seed's curve
        or an already-averaged curve.
    dx:
        Uniform spacing between checkpoints on the x-axis. For
        evaluation-step curves, set ``dx`` to the number of environment
        steps between checkpoints.

    Returns
    -------
    float
        ``numpy.trapz(returns, dx=dx)``.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"curve_auc expects a 1-D array, got ndim={arr.ndim}")
    if arr.size < 2:
        raise ValueError(
            f"curve_auc needs at least 2 checkpoints, got T={arr.size}"
        )
    return float(np.trapezoid(arr, dx=dx))


def final_performance(returns: np.ndarray, frac: float = 0.10) -> float:
    """Mean of the last ``frac`` fraction of a learning curve.

    The number of points averaged is
    ``max(1, ceil(frac * T))`` so that ``frac = 0.10`` on ``T = 10``
    selects the final point, not an empty slice.

    Parameters
    ----------
    returns:
        1-D array of shape ``(T,)``.
    frac:
        Fraction of the tail to average, in ``(0, 1]``.

    Returns
    -------
    float
        Mean of ``returns[-k:]`` with ``k = max(1, ceil(frac * T))``.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"final_performance expects a 1-D array, got ndim={arr.ndim}"
        )
    if arr.size == 0:
        raise ValueError("final_performance received an empty array")
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    k = max(1, math.ceil(frac * arr.size))
    return float(np.mean(arr[-k:]))


def steps_to_threshold(
    returns: np.ndarray,
    threshold: float,
    checkpoints: np.ndarray,
) -> float | None:
    """First checkpoint at which the curve reaches ``threshold``.

    Parameters
    ----------
    returns:
        1-D array of shape ``(T,)`` giving the performance at each
        checkpoint.
    threshold:
        Scalar performance target.
    checkpoints:
        1-D array of shape ``(T,)`` giving the environment-step count
        (or any monotonically increasing x-axis) at each entry of
        ``returns``.

    Returns
    -------
    float | None
        The value from ``checkpoints`` at the first index ``i`` where
        ``returns[i] >= threshold``. Returns ``None`` if the threshold
        is never crossed. The return type is ``float`` even when
        ``checkpoints`` is integer-typed.
    """
    ret = np.asarray(returns)
    chk = np.asarray(checkpoints)
    if ret.ndim != 1 or chk.ndim != 1:
        raise ValueError(
            "steps_to_threshold expects 1-D returns and checkpoints, "
            f"got ndims {ret.ndim} and {chk.ndim}"
        )
    if ret.shape != chk.shape:
        raise ValueError(
            "returns and checkpoints must have the same shape, "
            f"got {ret.shape} vs {chk.shape}"
        )
    hits = np.flatnonzero(ret >= threshold)
    if hits.size == 0:
        return None
    return float(chk[hits[0]])


# ---------------------------------------------------------------------------
# DP planner metrics (spec §9.2)
# ---------------------------------------------------------------------------


def sweep_to_tolerance(residuals: np.ndarray, tol: float) -> int | None:
    """First sweep index where the Bellman residual falls to ``<= tol``.

    Parameters
    ----------
    residuals:
        1-D array of shape ``(n_sweeps,)`` of per-sweep Bellman
        residuals. Non-negative.
    tol:
        Tolerance. Must be non-negative.

    Returns
    -------
    int | None
        Zero-based sweep index, or ``None`` if the residual never
        reaches ``tol``.
    """
    arr = np.asarray(residuals)
    if arr.ndim != 1:
        raise ValueError(
            f"sweep_to_tolerance expects a 1-D array, got ndim={arr.ndim}"
        )
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    hits = np.flatnonzero(arr <= tol)
    if hits.size == 0:
        return None
    return int(hits[0])


def sup_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Sup-norm ``||a - b||_inf`` between two conformable arrays.

    Parameters
    ----------
    a, b:
        Arrays of identical shape.

    Returns
    -------
    float
        ``numpy.max(numpy.abs(a - b))`` as a Python float.
    """
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        raise ValueError(
            f"sup_norm requires identical shapes, got {aa.shape} vs {bb.shape}"
        )
    if aa.size == 0:
        raise ValueError("sup_norm received an empty array")
    return float(np.max(np.abs(aa - bb)))


# ---------------------------------------------------------------------------
# Calibration metrics (spec §9.4)
# ---------------------------------------------------------------------------


def margin_quantiles(
    margins: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> dict[str, float]:
    """Empirical quantiles of the per-transition margin distribution.

    Parameters
    ----------
    margins:
        1-D array of shape ``(N,)`` of ``margin_beta0`` values logged
        per transition.
    quantiles:
        Quantile levels in ``[0, 1]``. Each level ``q`` is reported
        under the key ``f"q{int(round(q * 100))}"``.

    Returns
    -------
    dict[str, float]
        Quantile dictionary. Order is preserved from ``quantiles``.
    """
    arr = np.asarray(margins, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"margin_quantiles expects a 1-D array, got ndim={arr.ndim}"
        )
    if arr.size == 0:
        raise ValueError("margin_quantiles received an empty array")
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile levels must be in [0, 1], got {q}")
    qs = np.quantile(arr, quantiles)
    return {f"q{int(round(q * 100))}": float(v) for q, v in zip(quantiles, qs)}


def aligned_margin_freq(margins: np.ndarray) -> dict[str, float]:
    """Sign frequencies of the margin distribution.

    Parameters
    ----------
    margins:
        1-D array of shape ``(N,)``.

    Returns
    -------
    dict[str, float]
        ``pos_freq`` (fraction ``> 0``), ``neg_freq`` (fraction
        ``< 0``), ``zero_freq`` (fraction ``== 0``). The three values
        sum to 1 up to floating-point error.
    """
    arr = np.asarray(margins, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"aligned_margin_freq expects a 1-D array, got ndim={arr.ndim}"
        )
    if arr.size == 0:
        raise ValueError("aligned_margin_freq received an empty array")
    n = arr.size
    return {
        "pos_freq": float(np.sum(arr > 0.0) / n),
        "neg_freq": float(np.sum(arr < 0.0) / n),
        "zero_freq": float(np.sum(arr == 0.0) / n),
    }


# ---------------------------------------------------------------------------
# Success-rate helper
# ---------------------------------------------------------------------------


def success_rate(successes: np.ndarray) -> float:
    """Fraction of successful episodes.

    Parameters
    ----------
    successes:
        1-D array of shape ``(n_episodes,)`` of booleans or 0/1 flags.

    Returns
    -------
    float
        ``mean(successes)`` as a Python float.
    """
    arr = np.asarray(successes)
    if arr.ndim != 1:
        raise ValueError(
            f"success_rate expects a 1-D array, got ndim={arr.ndim}"
        )
    if arr.size == 0:
        raise ValueError("success_rate received an empty array")
    return float(np.mean(arr.astype(float)))
