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
    gamma_base: float,
) -> NDArray[np.float64]:
    """Backward recursion for certified radius Bhat (Phase III spec §5).

    The canonical recursion from the safe weighted-LSE operator contract is

    .. math::

        \\hat B_T = 0, \\quad
        \\hat B_t = (1 + \\gamma_{\\text{base}}) R_{\\max}
                    + \\kappa_t\\, \\hat B_{t+1}.

    This function delegates to
    ``mushroom_rl.algorithms.value.dp.safe_weighted_common.compute_certified_radii``
    so the geometry layer and the operator layer remain bit-for-bit identical.

    Parameters
    ----------
    kappa_t : NDArray[np.float64], shape (T,)
        Contraction rate per stage.
    r_max : float
        Maximum reward magnitude (``R_max``).
    T : int
        Number of stages (horizon length).
    gamma_base : float
        Nominal discount factor used by the safe operator.

    Returns
    -------
    NDArray[np.float64], shape (T+1,)
        Certified radius ``Bhat[0..T]`` with ``Bhat[T] = 0``.

    Notes
    -----
    The previous geometric-series form
    ``Bhat[t] = kappa_t * (r_max + Bhat[t+1]) / (1 - kappa_t)`` does NOT
    match the operator's contraction argument and was replaced in-place.
    See ``docs/specs/phase_IV_A_activation_audit_and_counterfactual.md``
    §6 and Phase III spec §5.
    """
    # Local import avoids pulling mushroom-rl at module load time.
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
        compute_certified_radii,
    )

    kappa_arr = np.asarray(kappa_t, dtype=np.float64)
    if len(kappa_arr) != T:
        raise ValueError(
            f"kappa_t has length {len(kappa_arr)}, expected T={T}."
        )
    return compute_certified_radii(T, kappa_arr, float(r_max), float(gamma_base))


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
    u_target_t: NDArray[np.float64] | None = None,
    max_iters: int = 4,
) -> dict[str, NDArray[np.float64]]:
    """Fixed-point iteration for adaptive headroom (spec S6.7).

    Iterates:
    1. Compute informativeness I_t from xi_ref and p_align.
    2. Compute alpha_base from I_t.
    3. Compute kappa, Bhat, A_t, theta_safe, U_safe_ref.
    4. If u_target_t > U_safe_ref at some stage, bump alpha upward and repeat.
       Stopping criterion: max_t(u_target_t - U_safe_ref_t) <= 0 (all feasible).

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
    u_target_t : NDArray[np.float64], shape (T,), optional
        Desired natural-shift target per stage.  When supplied, alpha is
        bumped only at stages where u_target_t > U_safe_ref_t (spec S6.7
        feasibility constraint).  When None the iteration runs for a fixed
        ``max_iters`` passes without a feasibility check.
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

    if u_target_t is not None:
        u_target_t = np.asarray(u_target_t, dtype=np.float64)

    # Step 1: informativeness and base alpha
    I_t = compute_informativeness(xi_ref_t, p_align_t)
    alpha_t = compute_alpha_base(I_t, alpha_min, alpha_max)

    for _it in range(max_iters):
        alpha_t = np.clip(alpha_t, 0.0, alpha_budget_max)

        kappa_t = compute_kappa(alpha_t, gamma_base)
        bhat = compute_bhat_backward(kappa_t, r_max, T, gamma_base)
        a_t_arr = compute_a_t(r_max, bhat)
        theta_safe_t = compute_theta_safe(kappa_t, gamma_base)
        u_safe_ref_t = compute_u_safe_ref(theta_safe_t, xi_ref_t)

        if u_target_t is not None:
            # Spec S6.7: bump alpha only where feasibility is violated.
            # Stopping criterion: all stages satisfy u_target <= U_safe_ref.
            needs_increase = u_target_t > u_safe_ref_t - 1e-10
        else:
            # No feasibility reference: run for full max_iters without early stop.
            needs_increase = np.zeros(T, dtype=bool)

        if not np.any(needs_increase):
            break

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
