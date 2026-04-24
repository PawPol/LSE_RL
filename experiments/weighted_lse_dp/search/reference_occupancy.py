"""Reference-occupancy helper for Phase V (WP1a).

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 3
("Reachability (fixed)") defining

    d_ref = 0.5 * d^{pi*_cl} + 0.5 * d^{pi*_safe}

as the reference state-occupancy used by every §6 candidate metric.

Conventions
-----------
* Occupancy is **undiscounted** per the Phase V spec -- the reference
  distribution represents *where the agent actually visits*, not a
  geometric-return weighting.  (Explicit deviation from the Phase
  III/IV usage of discounted visitation.)
* Arrays are dense ``(T, S_base)`` ndarrays, matching the
  ``V``/``Q`` arrays emitted by the geometry-priority DP planner.
* Policies may be deterministic (``int`` array of shape ``(T, S)``) or
  stochastic (``float`` array of shape ``(T, S, A)``).  Detection is
  by ``ndim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["compute_d_ref", "save_occupancy"]


def _policy_tensor(
    pi: np.ndarray,
    T: int,
    S: int,
    A: int,
) -> np.ndarray:
    """Return a stochastic policy tensor ``(T, S, A)`` from either shape."""
    pi = np.asarray(pi)
    if pi.ndim == 2:
        # Deterministic: pi[t, s] in {0, ..., A-1}.
        if pi.shape != (T, S):
            raise ValueError(
                f"deterministic policy shape must be ({T}, {S}); got {pi.shape}."
            )
        pi_int = pi.astype(np.int64, copy=False)
        if pi_int.min() < 0 or pi_int.max() >= A:
            raise ValueError(
                f"deterministic policy entries must lie in [0, {A-1}]; "
                f"range=[{pi_int.min()}, {pi_int.max()}]."
            )
        out = np.zeros((T, S, A), dtype=np.float64)
        t_idx = np.arange(T)[:, None]
        s_idx = np.arange(S)[None, :]
        out[t_idx, s_idx, pi_int] = 1.0
        return out
    if pi.ndim == 3:
        if pi.shape != (T, S, A):
            raise ValueError(
                f"stochastic policy shape must be ({T}, {S}, {A}); got {pi.shape}."
            )
        row_sums = pi.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(
                "stochastic policy rows must sum to 1 (atol=1e-8); "
                f"max deviation={np.max(np.abs(row_sums - 1.0)):.3e}."
            )
        return pi.astype(np.float64, copy=False)
    raise ValueError(
        f"policy ndim must be 2 (deterministic) or 3 (stochastic); got {pi.ndim}."
    )


def _forward_sweep(
    P: np.ndarray,
    pi: np.ndarray,
    mu_0: np.ndarray,
    T: int,
) -> np.ndarray:
    """Undiscounted forward occupancy sweep on a stationary finite MDP.

    Recurrence::

        d[0, s]       = mu_0[s]
        d[t+1, s']    = sum_{s, a} d[t, s] * pi[t, s, a] * P[s, a, s']

    Parameters
    ----------
    P : (S, A, S')
    pi : (T, S, A)  stochastic policy tensor
    mu_0 : (S,)
    T : int

    Returns
    -------
    d : (T, S)  occupancy per stage
    """
    S = P.shape[0]
    d = np.zeros((T, S), dtype=np.float64)
    d[0] = mu_0
    for t in range(T - 1):
        # joint p(s, a) at stage t:
        joint = d[t, :, None] * pi[t]              # (S, A)
        # next = sum_{s, a} joint[s, a] * P[s, a, :]
        d[t + 1] = np.einsum("ij,ijk->k", joint, P)
    return d


def compute_d_ref(
    mdp: Any,
    pi_classical: np.ndarray,
    pi_safe: np.ndarray,
    *,
    mu_0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact occupancy sweep for classical, safe, and reference policies.

    Parameters
    ----------
    mdp : MushroomRL ``FiniteMDP`` (stationary ``p``, finite ``horizon``).
    pi_classical, pi_safe : ndarray
        Either ``(T, S)`` deterministic int arrays or ``(T, S, A)``
        stochastic float arrays.
    mu_0 : ndarray | None
        Initial state distribution over ``S_base``.  Defaults to a point
        mass at ``mdp.initial_state`` when that attribute exists;
        otherwise falls back to ``mdp.mu`` (MushroomRL's optional initial
        distribution) or a point mass at state 0.

    Returns
    -------
    dict with keys:
        d_cl, d_safe, d_ref : (T, S) undiscounted occupancy arrays
        horizon : int
        gamma : float
    """
    P = np.asarray(mdp.p, dtype=np.float64)  # (S, A, S')
    S, A = P.shape[0], P.shape[1]
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)

    # Initial distribution.
    if mu_0 is None:
        init_s = getattr(mdp, "initial_state", None)
        if init_s is not None:
            mu_0 = np.zeros(S, dtype=np.float64)
            mu_0[int(init_s)] = 1.0
        elif getattr(mdp, "mu", None) is not None:
            mu_0 = np.asarray(mdp.mu, dtype=np.float64).reshape(-1)
        else:
            mu_0 = np.zeros(S, dtype=np.float64)
            mu_0[0] = 1.0
    else:
        mu_0 = np.asarray(mu_0, dtype=np.float64).reshape(-1)

    if mu_0.shape != (S,):
        raise ValueError(
            f"mu_0 shape must be ({S},); got {mu_0.shape}."
        )
    mass = float(mu_0.sum())
    if not np.isclose(mass, 1.0, atol=1e-8):
        raise ValueError(
            f"mu_0 must sum to 1; got sum={mass:.6g}."
        )

    pi_cl_t = _policy_tensor(pi_classical, T, S, A)
    pi_safe_t = _policy_tensor(pi_safe, T, S, A)

    d_cl = _forward_sweep(P, pi_cl_t, mu_0, T)
    d_safe = _forward_sweep(P, pi_safe_t, mu_0, T)
    d_ref = 0.5 * d_cl + 0.5 * d_safe

    return {
        "d_cl": d_cl,
        "d_safe": d_safe,
        "d_ref": d_ref,
        "horizon": T,
        "gamma": gamma,
    }


def save_occupancy(path: Path | str, d: dict[str, Any]) -> None:
    """Persist occupancy arrays to ``occupancy.npz`` (spec section 7 WP1).

    Keys written to disk: ``d_cl, d_safe, d_ref, time_augmented_shape,
    horizon, gamma``.  ``time_augmented_shape`` records ``(T, S_base)`` so
    downstream loaders can assert shape invariants without re-reading the
    MDP.
    """
    path = Path(path)
    d_cl = np.asarray(d["d_cl"], dtype=np.float64)
    d_safe = np.asarray(d["d_safe"], dtype=np.float64)
    d_ref = np.asarray(d["d_ref"], dtype=np.float64)
    horizon = int(d["horizon"])
    gamma = float(d["gamma"])
    time_augmented_shape = np.asarray(d_ref.shape, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        d_cl=d_cl,
        d_safe=d_safe,
        d_ref=d_ref,
        time_augmented_shape=time_augmented_shape,
        horizon=np.int64(horizon),
        gamma=np.float64(gamma),
    )
