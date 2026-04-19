"""Phase IV-A S6.5: Trust-region caps for natural-shift scheduling.

Trust region is defined in terms of KL divergence of Bernoulli responsibility:
    eta0 = log(1 / gamma_base)
    p0 = 1 / (1 + gamma_base)
    rho(u) = sigmoid(eta0 + u)
    eps_design = KL_Bern(rho(u_target) || p0)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "kl_bernoulli",
    "compute_eps_design",
    "compute_stagewise_confidence",
    "solve_u_tr_cap",
    "compute_trust_region_cap",
]

_EPS = 1e-10


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    result = np.empty_like(x)
    # For x >= 0: 1 / (1 + exp(-x))
    exp_neg = np.exp(-np.where(pos, x, 0.0))
    result[pos] = 1.0 / (1.0 + exp_neg[pos])
    # For x < 0: exp(x) / (1 + exp(x))
    exp_pos = np.exp(np.where(~pos, x, 0.0))
    result[~pos] = exp_pos[~pos] / (1.0 + exp_pos[~pos])
    return result


def kl_bernoulli(
    p: float | NDArray[np.float64],
    q: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """KL divergence KL(Bern(p) || Bern(q)).

    Parameters
    ----------
    p : float or NDArray[np.float64]
        Parameter of the first Bernoulli distribution.
    q : float or NDArray[np.float64]
        Parameter of the second Bernoulli distribution.

    Returns
    -------
    NDArray[np.float64]
        KL divergence, non-negative. Scalar if both inputs are scalar.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    q = np.clip(q, _EPS, 1.0 - _EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def _rho(u: float | NDArray[np.float64], gamma_base: float) -> NDArray[np.float64]:
    """Responsibility rho(u) = sigmoid(log(1/gamma_base) + u)."""
    eta0 = np.log(1.0 / gamma_base)
    return _sigmoid(np.asarray(eta0 + u, dtype=np.float64))


def compute_eps_design(
    u_target: float | NDArray[np.float64],
    gamma_base: float,
) -> NDArray[np.float64]:
    """Design-time KL budget: eps_design = KL_Bern(rho(u_target) || p0).

    Parameters
    ----------
    u_target : float or NDArray[np.float64]
        Target natural-shift magnitude.
    gamma_base : float
        Baseline discount factor in (0, 1).

    Returns
    -------
    NDArray[np.float64]
        Design KL divergence.
    """
    p0 = 1.0 / (1.0 + gamma_base)
    rho_val = _rho(u_target, gamma_base)
    return kl_bernoulli(rho_val, p0)


def compute_stagewise_confidence(
    n_t: float | NDArray[np.float64],
    p_align_t: float | NDArray[np.float64],
    tau_n: float = 200.0,
) -> NDArray[np.float64]:
    """Stagewise confidence from sample count and alignment probability.

    c_t = clip((n_t / (n_t + tau_n)) * sqrt(max(p_align_t, 0)), 0, 1)

    Parameters
    ----------
    n_t : float or NDArray[np.float64], shape (T,) or scalar
        Number of samples at stage t.
    p_align_t : float or NDArray[np.float64], shape (T,) or scalar
        Empirical alignment probability at stage t.
    tau_n : float
        Regularization constant for sample count (default 200).

    Returns
    -------
    NDArray[np.float64]
        Confidence in [0, 1].
    """
    n_t = np.asarray(n_t, dtype=np.float64)
    p_align_t = np.asarray(p_align_t, dtype=np.float64)
    raw = (n_t / (n_t + tau_n)) * np.sqrt(np.maximum(p_align_t, 0.0))
    return np.clip(raw, 0.0, 1.0)


def solve_u_tr_cap(
    eps_tr: float,
    gamma_base: float,
    u_init: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Find u_tr_cap >= 0 such that KL_Bern(rho(u_tr_cap) || p0) = eps_tr.

    Uses bisection on [0, 20].

    Parameters
    ----------
    eps_tr : float
        Target KL divergence for the trust region.
    gamma_base : float
        Baseline discount factor.
    u_init : float
        Unused (kept for API compatibility). Bisection starts from [0, 20].
    tol : float
        Convergence tolerance on KL difference.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    float
        Non-negative u_tr_cap. Returns 0.0 if eps_tr <= 0.
    """
    if eps_tr <= 0.0:
        return 0.0

    p0 = 1.0 / (1.0 + gamma_base)

    # KL_Bern(rho(u) || p0) is monotonically increasing for u >= 0
    # (rho moves away from p0 as u increases from 0).
    lo, hi = 0.0, 20.0

    # Check if eps_tr is beyond the range
    kl_hi = float(kl_bernoulli(float(_rho(hi, gamma_base)), p0))
    if eps_tr >= kl_hi:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rho_mid = float(_rho(mid, gamma_base))
        kl_mid = float(kl_bernoulli(rho_mid, p0))
        if abs(kl_mid - eps_tr) < tol:
            return mid
        if kl_mid < eps_tr:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def compute_trust_region_cap(
    u_target: float | NDArray[np.float64],
    p_align_t: float | NDArray[np.float64],
    n_t: float | NDArray[np.float64],
    gamma_base: float,
    tau_n: float = 200.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Full trust-region cap pipeline.

    1. eps_design = KL_Bern(rho(u_target) || p0)
    2. c_t = stagewise confidence
    3. eps_tr = c_t * eps_design
    4. u_tr_cap = solve bisection for KL = eps_tr

    Parameters
    ----------
    u_target : float or NDArray[np.float64], shape (T,) or scalar
        Target natural-shift magnitude (desired activation level).
    p_align_t : float or NDArray[np.float64], shape (T,) or scalar
        Empirical alignment probability at each stage.
    n_t : float or NDArray[np.float64], shape (T,) or scalar
        Sample count at each stage.
    gamma_base : float
        Baseline discount factor.
    tau_n : float
        Sample regularization constant.

    Returns
    -------
    u_tr_cap : NDArray[np.float64]
        Trust-region cap on natural shift.
    eps_design : NDArray[np.float64]
        Design KL budget.
    c_t : NDArray[np.float64]
        Stagewise confidence.
    eps_tr : NDArray[np.float64]
        Effective trust-region KL budget (c_t * eps_design).
    """
    u_target = np.asarray(u_target, dtype=np.float64)
    eps_design = compute_eps_design(u_target, gamma_base)
    c_t = compute_stagewise_confidence(n_t, p_align_t, tau_n)
    eps_tr = c_t * eps_design

    # Solve per-element
    eps_tr_flat = np.atleast_1d(eps_tr).ravel()
    u_cap_flat = np.array([
        solve_u_tr_cap(float(e), gamma_base) for e in eps_tr_flat
    ])

    # Reshape to match input shape
    if u_target.ndim == 0:
        u_tr_cap = np.float64(u_cap_flat[0])
    else:
        u_tr_cap = u_cap_flat.reshape(u_target.shape)

    return u_tr_cap, eps_design, c_t, eps_tr
