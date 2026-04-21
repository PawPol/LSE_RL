"""Phase IV-A S2: Natural-shift coordinate computations.

u = beta * (r - v) = theta * xi
where A_t = R_max + Bhat_{t+1}, xi = (r - v) / A_t, theta = beta * A_t.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "compute_natural_shift",
    "compute_normalized_coordinates",
    "compute_theta",
    "compute_aligned_margin",
    "small_signal_discount_gap",
    "small_signal_target_gap",
]


def compute_natural_shift(
    beta: float | NDArray[np.float64],
    reward: float | NDArray[np.float64],
    value_next: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute u = beta * (r - v).

    Parameters
    ----------
    beta : float or NDArray[np.float64], shape scalar or (N,) or (N, |A|)
        Operator temperature parameter. When zero, returns zeros.
    reward : float or NDArray[np.float64], shape scalar or (N,) or (N, |A|)
        One-step rewards.
    value_next : float or NDArray[np.float64], shape scalar or (N,) or (N, |A|)
        Next-state value estimates.

    Returns
    -------
    NDArray[np.float64], broadcast shape of inputs
        Natural shift u = beta * (r - v).
    """
    beta = np.asarray(beta, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)
    value_next = np.asarray(value_next, dtype=np.float64)
    return beta * (reward - value_next)


def compute_normalized_coordinates(
    reward: float | NDArray[np.float64],
    value_next: float | NDArray[np.float64],
    r_max: float,
    b_hat_next: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute (xi, A_t) where xi = (r - v) / A_t, A_t = R_max + Bhat_{t+1}.

    Parameters
    ----------
    reward : float or NDArray[np.float64], shape (N,)
        One-step rewards.
    value_next : float or NDArray[np.float64], shape (N,)
        Next-state value estimates.
    r_max : float
        Maximum reward magnitude for normalization.
    b_hat_next : float or NDArray[np.float64], shape (N,) or scalar
        Next-state certified radius Bhat_{t+1}.

    Returns
    -------
    xi : NDArray[np.float64], shape (N,)
        Normalized shift xi = (r - v) / A_t.
    a_t : NDArray[np.float64], shape (N,) or scalar
        Normalization anchor A_t = R_max + Bhat_{t+1}, clipped >= 1e-8.
    """
    reward = np.asarray(reward, dtype=np.float64)
    value_next = np.asarray(value_next, dtype=np.float64)
    b_hat_next = np.asarray(b_hat_next, dtype=np.float64)
    a_t = np.maximum(r_max + b_hat_next, 1e-8)
    xi = (reward - value_next) / a_t
    return xi, a_t


def compute_theta(
    beta: float | NDArray[np.float64],
    a_t: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute theta = beta * A_t.

    Parameters
    ----------
    beta : float or NDArray[np.float64], shape (T,) or scalar
        Operator temperature parameter.
    a_t : float or NDArray[np.float64], shape (T,) or scalar
        Normalization anchor A_t = R_max + Bhat_{t+1}.

    Returns
    -------
    NDArray[np.float64], broadcast shape
        Natural-coordinate scale theta = beta * A_t.
    """
    beta = np.asarray(beta, dtype=np.float64)
    a_t = np.asarray(a_t, dtype=np.float64)
    return beta * a_t


def compute_aligned_margin(
    reward: float | NDArray[np.float64],
    value_ref: float | NDArray[np.float64],
    a_t: float | NDArray[np.float64],
    sign_family: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute aligned normalized margin a = sign_family * (r - v_ref) / A_t.

    Parameters
    ----------
    reward : float or NDArray[np.float64], shape (N,)
        One-step rewards.
    value_ref : float or NDArray[np.float64], shape (N,)
        Reference value (e.g., V^* or current iterate).
    a_t : float or NDArray[np.float64], shape (N,) or scalar
        Normalization anchor A_t, must be > 0.
    sign_family : float or NDArray[np.float64], shape (N,) or scalar
        Sign of the operator family (+1 for optimistic, -1 for pessimistic).

    Returns
    -------
    NDArray[np.float64], shape (N,)
        Aligned normalized margin.
    """
    reward = np.asarray(reward, dtype=np.float64)
    value_ref = np.asarray(value_ref, dtype=np.float64)
    a_t = np.asarray(a_t, dtype=np.float64)
    sign_family = np.asarray(sign_family, dtype=np.float64)
    return sign_family * (reward - value_ref) / np.maximum(a_t, 1e-8)


def small_signal_discount_gap(
    beta: float | NDArray[np.float64],
    margin: float | NDArray[np.float64],
    gamma_base: float,
) -> NDArray[np.float64]:
    """First-order small-signal approximation to the discount gap.

    delta_d approx -(gamma_base / (1 + gamma_base)) * u
    where u = beta * margin. Diagnostic approximation only (spec S2.3).

    Parameters
    ----------
    beta : float or NDArray[np.float64], shape (N,)
        Operator temperature.
    margin : float or NDArray[np.float64], shape (N,)
        Reward-value margin (r - v).
    gamma_base : float
        Baseline discount factor.

    Returns
    -------
    NDArray[np.float64], shape (N,)
        Approximate discount gap. Returns zeros when beta == 0.
    """
    beta = np.asarray(beta, dtype=np.float64)
    margin = np.asarray(margin, dtype=np.float64)
    u = beta * margin
    coeff = -gamma_base / (1.0 + gamma_base)
    return coeff * u


def small_signal_target_gap(
    beta: float | NDArray[np.float64],
    margin: float | NDArray[np.float64],
    gamma_base: float,
) -> NDArray[np.float64]:
    """Second-order small-signal approximation to the target gap.

    gap approx (gamma_base / (2 * (1 + gamma_base))) * beta * margin^2.
    Diagnostic approximation only (spec S2.3).

    Parameters
    ----------
    beta : float or NDArray[np.float64], shape (N,)
        Operator temperature.
    margin : float or NDArray[np.float64], shape (N,)
        Reward-value margin (r - v).
    gamma_base : float
        Baseline discount factor.

    Returns
    -------
    NDArray[np.float64], shape (N,)
        Approximate target gap. Returns zeros when beta == 0.
    """
    beta = np.asarray(beta, dtype=np.float64)
    margin = np.asarray(margin, dtype=np.float64)
    coeff = gamma_base / (2.0 * (1.0 + gamma_base))
    return coeff * beta * margin**2
