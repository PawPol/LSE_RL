"""Single source of truth for the centered/scaled weighted-LSE Bellman kernel.

This module is the only location where the ``g_{β,γ}`` target, its
responsibility ``ρ_{β,γ}``, and the effective discount ``d_{β,γ}`` are
implemented. Both the Phase III–VI certified planner
(``mushroom_rl.algorithms.value.dp.safe_weighted_common.SafeWeightedCommon``)
and the Phase VII adaptive-β agent
(``experiments/adaptive_beta/agents.py``) import from here.

Spec authority: ``docs/specs/phase_VII_adaptive_beta.md`` §3.4 + §22.1
(option (b), locked 2026-04-26).

For β ≠ 0::

    g_{β,γ}(r, v) = (1 + γ) / β · [logaddexp(β·r, β·v + log γ) − log(1+γ)]
    ρ_{β,γ}(r, v) = sigmoid(β · (r − v) − log γ)
    d_{β,γ}(r, v) = (1 + γ) · (1 − ρ_{β,γ}(r, v))

For ``|β| <= _EPS_BETA`` the classical collapse branch is taken with no
``logaddexp`` call: ``g = r + γ·v``, ``ρ = 1/(1+γ)``, ``d = γ``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit as _sigmoid  # numerically stable sigmoid

# Mirrors safe_weighted_common._EPS_BETA so that classical-collapse semantics
# are byte-identical between the two callers.
_EPS_BETA: float = 1e-8

__all__ = [
    "_EPS_BETA",
    "_sigmoid",
    "g",
    "rho",
    "effective_discount",
    "g_batch",
    "rho_batch",
    "effective_discount_batch",
]


def _is_classical(beta: float) -> bool:
    return beta == 0.0 or abs(beta) <= _EPS_BETA


# ---------------------------------------------------------------------------
# Scalar kernel
# ---------------------------------------------------------------------------

def g(beta: float, gamma: float, r: float, v: float) -> float:
    r_f = float(r)
    v_f = float(v)
    b = float(beta)
    g_ = float(gamma)
    if _is_classical(b):
        return r_f + g_ * v_f
    one_plus_gamma = 1.0 + g_
    log_gamma = np.log(g_)
    log_1_plus_gamma = np.log(one_plus_gamma)
    log_sum = np.logaddexp(b * r_f, b * v_f + log_gamma)
    return float((one_plus_gamma / b) * (log_sum - log_1_plus_gamma))


def rho(beta: float, gamma: float, r: float, v: float) -> float:
    b = float(beta)
    g_ = float(gamma)
    one_plus_gamma = 1.0 + g_
    if _is_classical(b):
        return 1.0 / one_plus_gamma
    log_inv_gamma = -np.log(g_)
    arg = b * (float(r) - float(v)) + log_inv_gamma
    return float(_sigmoid(arg))


def effective_discount(beta: float, gamma: float, r: float, v: float) -> float:
    b = float(beta)
    g_ = float(gamma)
    if _is_classical(b):
        return g_
    one_plus_gamma = 1.0 + g_
    return one_plus_gamma * (1.0 - rho(b, g_, r, v))


# ---------------------------------------------------------------------------
# Batched kernel
# ---------------------------------------------------------------------------

def g_batch(
    beta: float,
    gamma: float,
    r: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    # r, v: any broadcastable shape; output matches the broadcast.
    b = float(beta)
    g_ = float(gamma)
    r_arr = np.asarray(r, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    one_plus_gamma = 1.0 + g_
    if _is_classical(b):
        return r_arr + g_ * v_arr
    log_gamma = np.log(g_)
    log_1_plus_gamma = np.log(one_plus_gamma)
    log_sum = np.logaddexp(b * r_arr, b * v_arr + log_gamma)
    return (one_plus_gamma / b) * (log_sum - log_1_plus_gamma)


def rho_batch(
    beta: float,
    gamma: float,
    r: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    # r, v: same broadcast contract as g_batch.
    b = float(beta)
    g_ = float(gamma)
    r_arr = np.asarray(r, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    one_plus_gamma = 1.0 + g_
    if _is_classical(b):
        return np.full_like(r_arr, 1.0 / one_plus_gamma)
    log_inv_gamma = -np.log(g_)
    arg = b * (r_arr - v_arr) + log_inv_gamma
    return _sigmoid(arg)


def effective_discount_batch(
    beta: float,
    gamma: float,
    r: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    b = float(beta)
    g_ = float(gamma)
    r_arr = np.asarray(r, dtype=np.float64)
    one_plus_gamma = 1.0 + g_
    if _is_classical(b):
        return np.full_like(r_arr, g_)
    return one_plus_gamma * (1.0 - rho_batch(b, g_, r_arr, np.asarray(v, dtype=np.float64)))
