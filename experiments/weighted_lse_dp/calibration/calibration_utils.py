"""
Utility functions for the Phase III safe-schedule calibration pipeline.

Spec reference: docs/specs/phase_III_safe_weighted_lse_experiments.md S5.4--S5.10

All functions are deterministic (no random state).  Arrays follow the
convention: shape ``(T,)`` for per-stage quantities, ``(T+1,)`` for the
certified-radius sequence ``Bhat_t`` which includes a terminal entry.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_calibration_json(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a Phase I/II calibration JSON file and return its contents."""
    path = pathlib.Path(path)
    with open(path, "r") as f:
        return json.load(f)


def compute_calibration_hash(path: str | pathlib.Path) -> str:
    """SHA-256 hex digest of the calibration JSON file (raw bytes)."""
    path = pathlib.Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def get_authoritative_T(cal: dict[str, Any]) -> int:
    """Return the authoritative horizon length T from margin_quantiles."""
    return len(cal["margin_quantiles"]["stages"])


def extract_stagewise_arrays(cal: dict[str, Any], T: int) -> dict[str, np.ndarray]:
    """Extract per-stage arrays from the column-oriented stagewise dict.

    Returns dict with keys:
        aligned_margin_freq, aligned_positive_mean,
        margin_q50, margin_q05, margin_q95.
    Each value is ``np.ndarray`` of shape ``(T,)``.
    Only the first *T* entries are used (stagewise may have T+1 rows due
    to an off-by-one in the Phase II logger).
    """
    sw = cal["stagewise"]
    return {
        "aligned_margin_freq": np.asarray(sw["aligned_margin_freq_mean"][:T], dtype=np.float64),
        "aligned_positive_mean": np.asarray(sw["aligned_positive_mean_mean"][:T], dtype=np.float64),
        "margin_q50": np.asarray(sw["margin_q50_mean"][:T], dtype=np.float64),
        "margin_q05": np.asarray(sw["margin_q05_mean"][:T], dtype=np.float64),
        "margin_q95": np.asarray(sw["margin_q95_mean"][:T], dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Calibration computations  (S5.4 -- S5.8)
# ---------------------------------------------------------------------------

def compute_representative_margin(
    aligned_positive_mean: np.ndarray,
    aligned_margin_freq: np.ndarray,
    min_freq_threshold: float = 0.01,
) -> np.ndarray:
    """Representative aligned margin ``m_t^*`` per stage.

    Uses the mean of positive aligned margins as a proxy for Q75
    (the raw Q75 is not available in the Phase II summary JSON).

    Where ``aligned_margin_freq[t] < min_freq_threshold``, the margin is
    set to 0.0 and downstream code must fall back to ``beta_raw_t = 0``.

    Parameters
    ----------
    aligned_positive_mean : (T,)
    aligned_margin_freq   : (T,)
    min_freq_threshold    : float

    Returns
    -------
    m_star : np.ndarray, shape (T,)
    """
    m_star = aligned_positive_mean.copy()
    sparse_mask = aligned_margin_freq < min_freq_threshold
    m_star[sparse_mask] = 0.0
    return m_star


def compute_informativeness(
    aligned_positive_mean: np.ndarray,
    aligned_margin_freq: np.ndarray,
) -> np.ndarray:
    """Normalized informativeness score ``I_t`` in [0, 1].

    .. math::
        \\text{raw}_t = \\text{aligned\\_positive\\_mean}[t]
                        \\cdot \\sqrt{\\text{aligned\\_margin\\_freq}[t]}

    Then min-max normalize to [0, 1].  If the raw scores are flat
    (max == min), all stages receive ``I_t = 0.5``.

    Parameters
    ----------
    aligned_positive_mean : (T,)
    aligned_margin_freq   : (T,)

    Returns
    -------
    I_t : np.ndarray, shape (T,), values in [0, 1]
    """
    raw = aligned_positive_mean * np.sqrt(aligned_margin_freq)
    r_min = raw.min()
    r_max = raw.max()
    eps = 1e-10
    if r_max - r_min < eps:
        return np.full_like(raw, 0.5)
    I_t = (raw - r_min) / (r_max - r_min + eps)
    return np.clip(I_t, 0.0, 1.0)


def compute_derivative_targets(
    informativeness: np.ndarray,
    gamma: float,
    lambda_min: float = 0.10,
    lambda_max: float = 0.50,
) -> np.ndarray:
    """Desired local derivative target per stage (S5.6).

    .. math::
        d_t^{\\text{target}} = \\gamma (1 - \\lambda_t), \\quad
        \\lambda_t = \\lambda_{\\min} + (\\lambda_{\\max} - \\lambda_{\\min}) I_t.

    Parameters
    ----------
    informativeness : (T,)  values in [0, 1]
    gamma           : float
    lambda_min      : float
    lambda_max      : float

    Returns
    -------
    d_target : np.ndarray, shape (T,)
    """
    lambda_t = lambda_min + (lambda_max - lambda_min) * informativeness
    return gamma * (1.0 - lambda_t)


def compute_raw_beta(
    m_star: np.ndarray,
    d_target: np.ndarray,
    gamma: float,
    sign: int,
) -> np.ndarray:
    """Raw beta magnitude from the derivative target (S5.7).

    .. math::
        |\\beta_t^{\\text{raw}}| = \\frac{1}{m_t^*}
            \\log\\!\\left(\\frac{(1+\\gamma)\\gamma}{d_t^{\\text{target}}} - \\gamma\\right)

    Fallback to 0 when ``m_t^* == 0`` or the log argument is non-positive.

    Parameters
    ----------
    m_star   : (T,)  representative aligned margin (non-negative)
    d_target : (T,)  derivative target
    gamma    : float
    sign     : int, +1 or -1

    Returns
    -------
    beta_raw : np.ndarray, shape (T,)
    """
    T = len(m_star)
    beta_raw = np.zeros(T, dtype=np.float64)
    for t in range(T):
        if m_star[t] <= 0.0:
            continue
        arg = (1.0 + gamma) * gamma / d_target[t] - gamma
        if arg <= 0.0:
            continue
        beta_raw[t] = sign * np.log(arg) / m_star[t]
    return beta_raw


def compute_headroom_fractions(
    informativeness: np.ndarray,
    alpha_min: float = 0.02,
    alpha_max: float = 0.10,
) -> np.ndarray:
    """Headroom fractions ``alpha_t`` per stage (S5.8).

    .. math::
        \\alpha_t = \\alpha_{\\min} + (\\alpha_{\\max} - \\alpha_{\\min}) I_t

    Parameters
    ----------
    informativeness : (T,)  values in [0, 1]
    alpha_min       : float
    alpha_max       : float

    Returns
    -------
    alpha_t : np.ndarray, shape (T,)
    """
    return alpha_min + (alpha_max - alpha_min) * informativeness


# ---------------------------------------------------------------------------
# Certification  (S5.9 / S2.2)
# ---------------------------------------------------------------------------

def build_certification(
    alpha_t: np.ndarray,
    R_max: float,
    gamma: float,
) -> dict[str, np.ndarray]:
    """Compute certification levels, radii, and clip caps (spec S2.2 / S5.9).

    Parameters
    ----------
    alpha_t : (T,)   headroom fractions in [0, 1)
    R_max   : float  absolute reward bound
    gamma   : float  discount factor

    Returns
    -------
    dict with keys:
        kappa_t   : (T,)    certification contraction levels
        Bhat_t    : (T+1,)  certified value-function radii (Bhat_T = 0)
        beta_cap_t: (T,)    stagewise clip caps (positive)
    """
    T = len(alpha_t)
    kappa_t = gamma + alpha_t * (1.0 - gamma)          # (T,)

    # Backward recursion for Bhat
    Bhat = np.zeros(T + 1, dtype=np.float64)            # Bhat[T] = 0
    for t in range(T - 1, -1, -1):
        Bhat[t] = (1.0 + gamma) * R_max + kappa_t[t] * Bhat[t + 1]

    # Clip caps
    beta_cap = np.zeros(T, dtype=np.float64)
    for t in range(T):
        denom = R_max + Bhat[t + 1]
        numer_arg = kappa_t[t] / (gamma * (1.0 + gamma - kappa_t[t]))
        if numer_arg > 0.0 and denom > 0.0:
            beta_cap[t] = np.log(numer_arg) / denom
        else:
            beta_cap[t] = 0.0

    return {
        "kappa_t": kappa_t,
        "Bhat_t": Bhat,
        "beta_cap_t": beta_cap,
    }


def clip_beta(
    beta_raw: np.ndarray,
    beta_cap: np.ndarray,
) -> tuple[np.ndarray, list[bool]]:
    """Clip raw beta to the certified box and report clipping activity.

    Parameters
    ----------
    beta_raw : (T,)
    beta_cap : (T,)  positive caps

    Returns
    -------
    beta_used    : np.ndarray, shape (T,)
    clip_active  : list[bool], length T
    """
    beta_used = np.clip(beta_raw, -beta_cap, beta_cap)
    clip_active = (np.abs(beta_raw) > np.abs(beta_used) + 1e-15).tolist()
    return beta_used, clip_active
