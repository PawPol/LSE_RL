"""Phase IV-C §4: Schedule smoothing for state-dependent schedulers."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hierarchical_shrinkage(
    u_design: NDArray[np.float64],
    u_stage: NDArray[np.float64],
    bin_counts: NDArray[np.int64],
    tau_bin: float = 100.0,
) -> NDArray[np.float64]:
    """Hierarchical-shrinkage backoff toward the stagewise schedule.

    Formula (spec §4.2):

        w_{t,b}   = n_{t,b} / (n_{t,b} + tau_bin)
        u_hb_{t,b} = u_stage_t + w_{t,b} * (u_design_{t,b} - u_stage_t)

    Parameters
    ----------
    u_design : NDArray[np.float64], shape (T, n_bins)
        Per-stage, per-bin design natural-shift targets.
    u_stage : NDArray[np.float64], shape (T,)
        Stagewise fallback values (scalar per stage).
    bin_counts : NDArray[np.int64], shape (T, n_bins)
        Number of observed transitions per (stage, bin).
    tau_bin : float
        Shrinkage bandwidth. Larger → stronger shrinkage toward u_stage.

    Returns
    -------
    NDArray[np.float64], shape (T, n_bins)
        Hierarchically backed-off per-(stage, bin) natural-shift values.
    """
    u_design = np.asarray(u_design, dtype=np.float64)
    u_stage = np.asarray(u_stage, dtype=np.float64)
    bin_counts = np.asarray(bin_counts, dtype=np.float64)
    tau_bin = float(tau_bin)

    # w_{t,b} = n / (n + tau)
    w = bin_counts / (bin_counts + tau_bin)  # (T, n_bins)

    # u_stage has shape (T,) → broadcast to (T, n_bins)
    u_stage_2d = u_stage[:, np.newaxis]
    return u_stage_2d + w * (u_design - u_stage_2d)


def smooth_schedule(
    raw_schedule: NDArray[np.float64],
    smoothing_method: str = "hierarchical_shrinkage",
    stagewise_fallback: NDArray[np.float64] | None = None,
    bin_counts: NDArray[np.int64] | None = None,
    tau_bin: float = 100.0,
) -> NDArray[np.float64]:
    """Apply smoothing to a raw state-dependent schedule.

    Parameters
    ----------
    raw_schedule : NDArray[np.float64], shape (T, n_bins)
        Raw per-(stage, bin) schedule values.
    smoothing_method : str
        Smoothing strategy:
        - ``"hierarchical_shrinkage"``: shrink toward stagewise fallback
          using bin visit counts (spec §4.2, default).
        - ``"moving_average"``: uniform moving average across bins.
        - ``"kernel"``: placeholder; falls back to ``"moving_average"``.
    stagewise_fallback : NDArray[np.float64], shape (T,), optional
        Stagewise fallback for ``"hierarchical_shrinkage"``. Required for
        that mode; if None, uses column mean as approximation.
    bin_counts : NDArray[np.int64], shape (T, n_bins), optional
        Visit counts per (stage, bin). Required for
        ``"hierarchical_shrinkage"``; if None, uses zeros (full shrinkage).
    tau_bin : float
        Shrinkage bandwidth for ``"hierarchical_shrinkage"``.

    Returns
    -------
    NDArray[np.float64], same shape as raw_schedule
        Smoothed schedule values.
    """
    raw_schedule = np.asarray(raw_schedule, dtype=np.float64)

    if smoothing_method == "hierarchical_shrinkage":
        T, n_bins = raw_schedule.shape if raw_schedule.ndim == 2 else (1, len(raw_schedule))
        raw_2d = raw_schedule.reshape(T, n_bins)
        if stagewise_fallback is None:
            fallback = np.mean(raw_2d, axis=1)
        else:
            fallback = np.asarray(stagewise_fallback, dtype=np.float64).ravel()[:T]
        if bin_counts is None:
            counts = np.zeros((T, n_bins), dtype=np.int64)
        else:
            counts = np.asarray(bin_counts, dtype=np.int64).reshape(T, n_bins)
        return hierarchical_shrinkage(raw_2d, fallback, counts, tau_bin=tau_bin).reshape(raw_schedule.shape)

    if smoothing_method in ("moving_average", "kernel"):
        # Simple per-row moving average with window=3
        if raw_schedule.ndim == 1:
            result = np.convolve(raw_schedule, np.ones(3) / 3.0, mode="same")
        else:
            result = np.array([
                np.convolve(row, np.ones(3) / 3.0, mode="same")
                for row in raw_schedule
            ])
        return result

    raise ValueError(f"Unknown smoothing method: {smoothing_method!r}. "
                     f"Supported: 'hierarchical_shrinkage', 'moving_average', 'kernel'.")
