"""Phase IV-A S8.2, S11.1, S13: Activation diagnostic metrics.

Aggregate diagnostics over transitions for the natural-shift activation layer.
Provides gate checks to determine whether operator activation is meaningful.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "compute_aggregate_diagnostics",
    "activation_gate_check",
    "compute_event_conditioned_diagnostics",
    "compute_stage_aggregate",
]


def _safe_quantile(arr: NDArray[np.float64], q: float) -> float:
    """Quantile that handles empty arrays."""
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def _compute_diagnostics_core(
    natural_shift: NDArray[np.float64],
    delta_d: NDArray[np.float64],
    target_gap: NDArray[np.float64],
    reward_bound: float,
    beta_used: NDArray[np.float64],
) -> dict[str, float]:
    """Core diagnostics computation on a flat array of transitions.

    Parameters
    ----------
    natural_shift : NDArray[np.float64], shape (N,)
        Natural shift u = beta * (r - v) per transition.
    delta_d : NDArray[np.float64], shape (N,)
        Discount gap d_t - gamma_base per transition.
    target_gap : NDArray[np.float64], shape (N,)
        Safe target minus classical target per transition.
    reward_bound : float
        Reward bound R_max for normalization.
    beta_used : NDArray[np.float64], shape (N,)
        Beta actually used per transition.

    Returns
    -------
    dict[str, float]
        All aggregate diagnostic statistics.
    """
    n = natural_shift.size
    if n == 0:
        return {
            "mean_abs_u": 0.0,
            "std_abs_u": 0.0,
            "q50_abs_u": 0.0,
            "q75_abs_u": 0.0,
            "q95_abs_u": 0.0,
            "frac_u_ge_5e3": 0.0,
            "frac_u_ge_1e2": 0.0,
            "mean_abs_delta_d": 0.0,
            "std_abs_delta_d": 0.0,
            "frac_delta_d_ge_1e3": 0.0,
            "frac_delta_d_ge_5e3": 0.0,
            "mean_abs_target_gap": 0.0,
            "normalized_mean_abs_target_gap": 0.0,
            "frac_target_gap_ge_5e3_rb": 0.0,
            "mean_beta_used": 0.0,
        }

    abs_u = np.abs(natural_shift)
    abs_dd = np.abs(delta_d)
    abs_tg = np.abs(target_gap)
    rb = max(reward_bound, 1e-8)

    return {
        "mean_abs_u": float(np.mean(abs_u)),
        "std_abs_u": float(np.std(abs_u)),
        "q50_abs_u": _safe_quantile(abs_u, 0.50),
        "q75_abs_u": _safe_quantile(abs_u, 0.75),
        "q95_abs_u": _safe_quantile(abs_u, 0.95),
        "frac_u_ge_5e3": float(np.mean(abs_u >= 5e-3)),
        "frac_u_ge_1e2": float(np.mean(abs_u >= 1e-2)),
        "mean_abs_delta_d": float(np.mean(abs_dd)),
        "std_abs_delta_d": float(np.std(abs_dd)),
        "frac_delta_d_ge_1e3": float(np.mean(abs_dd >= 1e-3)),
        "frac_delta_d_ge_5e3": float(np.mean(abs_dd >= 5e-3)),
        "mean_abs_target_gap": float(np.mean(abs_tg)),
        "normalized_mean_abs_target_gap": float(np.mean(abs_tg)) / rb,
        "frac_target_gap_ge_5e3_rb": float(np.mean(abs_tg >= 5e-3 * rb)),
        "mean_beta_used": float(np.mean(beta_used)),
    }


def compute_aggregate_diagnostics(
    natural_shift: NDArray[np.float64],
    delta_d: NDArray[np.float64],
    target_gap: NDArray[np.float64],
    reward_bound: float,
    beta_used: NDArray[np.float64],
    informative_mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Aggregate activation diagnostics over all transitions.

    Parameters
    ----------
    natural_shift : NDArray[np.float64], shape (N,) or (T, N_s)
        Natural shift u per transition. Flattened internally.
    delta_d : NDArray[np.float64], same shape
        Discount gap per transition.
    target_gap : NDArray[np.float64], same shape
        Target gap per transition.
    reward_bound : float
        Reward bound R_max.
    beta_used : NDArray[np.float64], same shape
        Beta actually used per transition.
    informative_mask : NDArray[np.bool_] or None, same shape
        If provided, also compute diagnostics restricted to masked transitions.

    Returns
    -------
    dict[str, float]
        Aggregate statistics. If informative_mask is provided, keys prefixed
        with "informative_" are added for the masked subset.
    """
    natural_shift = np.asarray(natural_shift, dtype=np.float64).ravel()
    delta_d = np.asarray(delta_d, dtype=np.float64).ravel()
    target_gap = np.asarray(target_gap, dtype=np.float64).ravel()
    beta_used = np.asarray(beta_used, dtype=np.float64).ravel()

    result = _compute_diagnostics_core(
        natural_shift, delta_d, target_gap, reward_bound, beta_used
    )

    if informative_mask is not None:
        mask = np.asarray(informative_mask, dtype=np.bool_).ravel()
        masked_diag = _compute_diagnostics_core(
            natural_shift[mask],
            delta_d[mask],
            target_gap[mask],
            reward_bound,
            beta_used[mask],
        )
        for k, v in masked_diag.items():
            result[f"informative_{k}"] = v

    return result


def activation_gate_check(
    diagnostics: dict[str, float],
    min_mean_abs_u: float = 5e-3,
    min_frac_active: float = 0.10,
    min_mean_abs_delta_d: float = 1e-3,
    min_normalized_target_gap: float = 5e-3,
) -> dict[str, bool | dict]:
    """Check activation gate thresholds (Phase IV-A S13).

    Parameters
    ----------
    diagnostics : dict[str, float]
        Output of compute_aggregate_diagnostics.
    min_mean_abs_u : float
        Minimum mean |u| for the operator to be active.
    min_frac_active : float
        Minimum fraction of transitions with |u| >= 5e-3.
    min_mean_abs_delta_d : float
        Minimum mean |delta_d| for measurable discount effect.
    min_normalized_target_gap : float
        Minimum normalized mean |target_gap| / R_max.

    Returns
    -------
    dict with keys:
        global_gate_pass : bool
            True if ALL checks pass.
        preferred_gate_pass : bool
            True if mean_abs_u and frac_active pass (minimum viable activation).
        individual_checks : dict[str, bool]
            Per-check pass/fail.
        values : dict[str, float]
            Actual values compared against thresholds.
    """
    checks = {
        "mean_abs_u": diagnostics.get("mean_abs_u", 0.0) >= min_mean_abs_u,
        "frac_active": diagnostics.get("frac_u_ge_5e3", 0.0) >= min_frac_active,
        "mean_abs_delta_d": diagnostics.get("mean_abs_delta_d", 0.0) >= min_mean_abs_delta_d,
        "normalized_target_gap": diagnostics.get("normalized_mean_abs_target_gap", 0.0) >= min_normalized_target_gap,
    }

    values = {
        "mean_abs_u": diagnostics.get("mean_abs_u", 0.0),
        "frac_active": diagnostics.get("frac_u_ge_5e3", 0.0),
        "mean_abs_delta_d": diagnostics.get("mean_abs_delta_d", 0.0),
        "normalized_target_gap": diagnostics.get("normalized_mean_abs_target_gap", 0.0),
    }

    return {
        "global_gate_pass": all(checks.values()),
        "preferred_gate_pass": checks["mean_abs_u"] and checks["frac_active"],
        "individual_checks": checks,
        "values": values,
    }


def compute_event_conditioned_diagnostics(
    natural_shift: NDArray[np.float64],
    delta_d: NDArray[np.float64],
    target_gap: NDArray[np.float64],
    reward_bound: float,
    event_mask: NDArray[np.bool_],
) -> dict[str, float]:
    """Compute aggregate diagnostics restricted to event_mask==True transitions.

    Parameters
    ----------
    natural_shift : NDArray[np.float64], shape (N,)
        Natural shift per transition.
    delta_d : NDArray[np.float64], shape (N,)
        Discount gap per transition.
    target_gap : NDArray[np.float64], shape (N,)
        Target gap per transition.
    reward_bound : float
        Reward bound R_max.
    event_mask : NDArray[np.bool_], shape (N,)
        Boolean mask selecting transitions of interest.

    Returns
    -------
    dict[str, float]
        Same keys as compute_aggregate_diagnostics, restricted to masked transitions.
    """
    natural_shift = np.asarray(natural_shift, dtype=np.float64).ravel()
    delta_d = np.asarray(delta_d, dtype=np.float64).ravel()
    target_gap = np.asarray(target_gap, dtype=np.float64).ravel()
    event_mask = np.asarray(event_mask, dtype=np.bool_).ravel()

    # Create a dummy beta_used (ones) since we don't have it as a parameter
    # For event-conditioned, beta_used is not the focus; use ones as placeholder
    beta_dummy = np.ones(np.sum(event_mask), dtype=np.float64)

    return _compute_diagnostics_core(
        natural_shift[event_mask],
        delta_d[event_mask],
        target_gap[event_mask],
        reward_bound,
        beta_dummy,
    )


def compute_stage_aggregate(
    natural_shift_by_stage: list[NDArray[np.float64]],
    delta_d_by_stage: list[NDArray[np.float64]],
    target_gap_by_stage: list[NDArray[np.float64]],
    reward_bound: float,
) -> list[dict[str, float]]:
    """Per-stage aggregate diagnostics.

    Parameters
    ----------
    natural_shift_by_stage : list of NDArray[np.float64]
        Per-stage arrays of natural shift values. Length T.
    delta_d_by_stage : list of NDArray[np.float64]
        Per-stage arrays of discount gap values. Length T.
    target_gap_by_stage : list of NDArray[np.float64]
        Per-stage arrays of target gap values. Length T.
    reward_bound : float
        Reward bound R_max.

    Returns
    -------
    list of dict[str, float]
        Per-stage aggregate diagnostics. Length T.
    """
    T = len(natural_shift_by_stage)
    results = []
    for t in range(T):
        ns = np.asarray(natural_shift_by_stage[t], dtype=np.float64).ravel()
        dd = np.asarray(delta_d_by_stage[t], dtype=np.float64).ravel()
        tg = np.asarray(target_gap_by_stage[t], dtype=np.float64).ravel()
        # Use ones for beta_used placeholder at stage level
        beta_dummy = np.ones_like(ns)
        results.append(_compute_diagnostics_core(ns, dd, tg, reward_bound, beta_dummy))
    return results
