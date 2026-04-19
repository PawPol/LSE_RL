"""Phase IV-A S6.6-6.7: Adaptive headroom for schedule calibration.

Computes the adaptive alpha/kappa/Bhat chain and the fixed-point iteration
for balancing activation targets against contraction certification.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "compute_informativeness",
    "compute_alpha_base",
    "compute_kappa",
    "compute_bhat_backward",
    "compute_a_t",
    "compute_theta_safe",
    "compute_u_safe_ref",
    "run_fixed_point",
]


def compute_informativeness(
    xi_ref_t: NDArray[np.float64],
    p_align_t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute informativeness score I_t = normalize(xi_ref * sqrt(p_align)).

    Parameters
    ----------
    xi_ref_t : NDArray[np.float64], shape (T,)
        Reference normalized margin per stage.
    p_align_t : NDArray[np.float64], shape (T,)
        Alignment probability per stage.

    Returns
    -------
    NDArray[np.float64], shape (T,)
        Informativeness I_t in [0, 1], normalized by max.
    """
    xi_ref_t = np.asarray(xi_ref_t, dtype=np.float64)
    p_align_t = np.asarray(p_align_t, dtype=np.float64)
    raw = xi_ref_t * np.sqrt(np.maximum(p_align_t, 0.0))
    max_val = np.max(np.abs(raw))
    if max_val < 1e-8:
        return np.zeros_like(raw)
    return raw / max_val


def compute_alpha_base(
    I_t: NDArray[np.float64],
    alpha_min: float = 0.05,
    alpha_max: float = 0.20,
) -> NDArray[np.float64]:
    """Compute base headroom alpha_base_t = alpha_min + (alpha_max - alpha_min) * I_t.

    Parameters
    ----------
    I_t : NDArray[np.float64], shape (T,)
        Informativeness score per stage, in [0, 1].
    alpha_min : float
        Minimum headroom (default 0.05).
    alpha_max : float
        Maximum headroom at full informativeness (default 0.20).

    Returns
    -------
    NDArray[np.float64], shape (T,)
        Base headroom alpha_base_t.
    """
    I_t = np.asarray(I_t, dtype=np.float64)
    return alpha_min + (alpha_max - alpha_min) * I_t


def compute_kappa(
    alpha_t: NDArray[np.float64],
    gamma_base: float,
) -> NDArray[np.float64]:
    """Compute contraction rate kappa_t = gamma_base + alpha_t * (1 - gamma_base).

    Parameters
    ----------
    alpha_t : NDArray[np.float64], shape (T,)
        Headroom parameter per stage.
    gamma_base : float
        Baseline discount factor.

    Returns
    -------
    NDArray[np.float64], shape (T,)
        Contraction rate kappa_t, strictly less than 1.

    Raises
    ------
    AssertionError
        If any kappa_t >= 1.0.
    """
    alpha_t = np.asarray(alpha_t, dtype=np.float64)
    kappa_t = gamma_base + alpha_t * (1.0 - gamma_base)
    assert np.all(kappa_t < 1.0), (
        f"kappa_t must be < 1.0 but max is {np.max(kappa_t):.6f}. "
        f"Check alpha_t (max={np.max(alpha_t):.6f}) and gamma_base={gamma_base}."
    )
    return kappa_t


def compute_bhat_backward(
    kappa_t: NDArray[np.float64],
    r_max: float,
    T: int,
) -> NDArray[np.float64]:
    """Backward recursion for certified radius Bhat.

    Bhat[T] = 0
    Bhat[t] = kappa_t[t] * (r_max + Bhat[t+1]) / (1 - kappa_t[t])

    Parameters
    ----------
    kappa_t : NDArray[np.float64], shape (T,)
        Contraction rate per stage.
    r_max : float
        Maximum reward magnitude.
    T : int
        Number of stages (horizon length).

    Returns
    -------
    NDArray[np.float64], shape (T+1,)
        Certified radius Bhat[0..T] with Bhat[T] = 0.
    """
    kappa_t = np.asarray(kappa_t, dtype=np.float64)
    bhat = np.zeros(T + 1, dtype=np.float64)
    for t in range(T - 1, -1, -1):
        bhat[t] = kappa_t[t] * (r_max + bhat[t + 1]) / (1.0 - kappa_t[t])
    return bhat


def compute_a_t(
    r_max: float,
    bhat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute normalization anchor A_t = r_max + Bhat_{t+1} for t=0..T-1.

    Parameters
    ----------
    r_max : float
        Maximum reward magnitude.
    bhat : NDArray[np.float64], shape (T+1,)
        Certified radius array from compute_bhat_backward.

    Returns
    -------
    NDArray[np.float64], shape (T,)
        Normalization anchors A_t.
    """
    bhat = np.asarray(bhat, dtype=np.float64)
    # A_t = r_max + Bhat_{t+1} for t = 0, ..., T-1
    return r_max + bhat[1:]


def compute_theta_safe(
    kappa_t: NDArray[np.float64],
    gamma_base: float,
) -> NDArray[np.float64]:
    """Compute safe theta cap: Theta_safe_t = log(kappa_t / (gamma_base * (1 + gamma_base - kappa_t))).

    Parameters
    ----------
    kappa_t : NDArray[np.float64], shape (T,)
        Contraction rate per stage.
    gamma_base : float
        Baseline discount factor.

    Returns
    -------
    NDArray[np.float64], shape (T,)
        Safe theta cap per stage.

    Raises
    ------
    AssertionError
        If denominator gamma_base * (1 + gamma_base - kappa_t) <= 0.
    """
    kappa_t = np.asarray(kappa_t, dtype=np.float64)
    denom = gamma_base * (1.0 + gamma_base - kappa_t)
    assert np.all(denom > 0), (
        f"Denominator must be > 0 but min is {np.min(denom):.6e}. "
        f"Need kappa_t < 1 + gamma_base."
    )
    return np.log(kappa_t / denom)


def compute_u_safe_ref(
    theta_safe_t: NDArray[np.float64],
    xi_ref_t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute reference safe natural shift U_safe_ref_t = Theta_safe_t * xi_ref_t.

    Parameters
    ----------
    theta_safe_t : NDArray[np.float64], shape (T,)
        Safe theta cap per stage.
    xi_ref_t : NDArray[np.float64], shape (T,)
        Reference normalized margin per stage.

    Returns
    -------
    NDArray[np.float64], shape (T,)
        Reference safe natural shift.
    """
    theta_safe_t = np.asarray(theta_safe_t, dtype=np.float64)
    xi_ref_t = np.asarray(xi_ref_t, dtype=np.float64)
    return theta_safe_t * xi_ref_t


def run_fixed_point(
    xi_ref_t: NDArray[np.float64],
    p_align_t: NDArray[np.float64],
    r_max: float,
    gamma_base: float,
    alpha_min: float = 0.05,
    alpha_max: float = 0.20,
    alpha_budget_max: float = 0.30,
    max_iters: int = 4,
) -> dict[str, NDArray[np.float64]]:
    """Fixed-point iteration for adaptive headroom (spec S6.7).

    Iterates:
    1. Compute informativeness I_t from xi_ref and p_align.
    2. Compute alpha_base from I_t.
    3. Compute kappa, Bhat, A_t, theta_safe, U_safe_ref.
    4. If u_target > U_safe_ref at some stage, increase alpha (clipped by
       alpha_budget_max) and repeat.

    Parameters
    ----------
    xi_ref_t : NDArray[np.float64], shape (T,)
        Reference normalized margin per stage.
    p_align_t : NDArray[np.float64], shape (T,)
        Alignment probability per stage.
    r_max : float
        Maximum reward magnitude.
    gamma_base : float
        Baseline discount factor.
    alpha_min : float
        Minimum headroom (default 0.05).
    alpha_max : float
        Maximum headroom at full informativeness (default 0.20).
    alpha_budget_max : float
        Hard ceiling on alpha to maintain contraction (default 0.30).
    max_iters : int
        Maximum fixed-point iterations (default 4).

    Returns
    -------
    dict with keys:
        alpha_t : NDArray[np.float64], shape (T,)
        kappa_t : NDArray[np.float64], shape (T,)
        bhat : NDArray[np.float64], shape (T+1,)
        A_t : NDArray[np.float64], shape (T,)
        theta_safe_t : NDArray[np.float64], shape (T,)
        U_safe_ref_t : NDArray[np.float64], shape (T,)
    """
    xi_ref_t = np.asarray(xi_ref_t, dtype=np.float64)
    p_align_t = np.asarray(p_align_t, dtype=np.float64)
    T = len(xi_ref_t)

    # Step 1: informativeness and base alpha
    I_t = compute_informativeness(xi_ref_t, p_align_t)
    alpha_t = compute_alpha_base(I_t, alpha_min, alpha_max)

    for _it in range(max_iters):
        # Clip alpha
        alpha_t = np.clip(alpha_t, 0.0, alpha_budget_max)

        # Compute the chain
        kappa_t = compute_kappa(alpha_t, gamma_base)
        bhat = compute_bhat_backward(kappa_t, r_max, T)
        a_t_arr = compute_a_t(r_max, bhat)
        theta_safe_t = compute_theta_safe(kappa_t, gamma_base)
        u_safe_ref_t = compute_u_safe_ref(theta_safe_t, xi_ref_t)

        # Compute u_target as theta_safe * xi_ref (the desired activation)
        # If u_target exceeds U_safe_ref at any stage, we need to increase alpha.
        # In the initial iteration, u_target == U_safe_ref by construction,
        # so we check if the constraint is tight or violated after adjusting.
        # The fixed-point adjusts alpha upward where needed.
        # For the first iteration, u_target is simply the desired level.
        # After first pass, check if the certified radius accommodates
        # the target. If Bhat grew (due to alpha increase), A_t changes,
        # which changes xi and thus U_safe_ref.

        # Recompute xi_ref with updated A_t
        # xi_ref_t is given and fixed (it's the empirical margin from data),
        # but A_t changes, so the effective u_target changes.
        # If |u_safe_ref| < desired activation, increase alpha.
        # We use a heuristic: stages where theta_safe is small relative
        # to what alpha_max would allow get a bump.
        kappa_max = compute_kappa(
            np.full(T, alpha_budget_max), gamma_base
        )
        theta_safe_max = compute_theta_safe(kappa_max, gamma_base)
        headroom_ratio = theta_safe_t / np.maximum(theta_safe_max, 1e-8)

        # Stages with low headroom_ratio get alpha increased
        needs_increase = headroom_ratio < 0.8
        if not np.any(needs_increase):
            break

        # Gentle increase: move alpha toward alpha_budget_max for needy stages
        alpha_t = np.where(
            needs_increase,
            np.minimum(alpha_t * 1.3, alpha_budget_max),
            alpha_t,
        )

    return {
        "alpha_t": alpha_t,
        "kappa_t": kappa_t,
        "bhat": bhat,
        "A_t": a_t_arr,
        "theta_safe_t": theta_safe_t,
        "U_safe_ref_t": u_safe_ref_t,
    }
