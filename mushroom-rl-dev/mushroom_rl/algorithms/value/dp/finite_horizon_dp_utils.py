"""
Finite-horizon dynamic programming utilities.

This module provides lightweight, pure-numpy helpers for finite-horizon dynamic
programming over a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP`.

Conventions
-----------
- Transition tensor ``P`` has shape ``(S, A, S)``; ``P[s, a, s']`` is the
  probability of transitioning to ``s'`` given state ``s`` and action ``a``.
- Reward tensor ``R`` has shape ``(S, A, S)``; ``R[s, a, s']`` is the reward
  received on the transition ``(s, a, s')``.
- The horizon ``T`` is a positive integer. The stage-indexed canonical arrays
  are ``Q[t, s, a]`` and ``pi[t, s]`` for ``t in 0..T-1``, and
  ``V[t, s]`` for ``t in 0..T`` with terminal convention ``V[T, :] = 0``.
- Absorbing states (rows of ``P`` that are entirely zero) are permitted and
  are not forced to sum to one.

These helpers are intentionally framework-agnostic: they consume raw numpy
arrays plus scalars ``(horizon, gamma)`` and do not subclass
:class:`mushroom_rl.core.Agent`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


_PROB_ROW_TOL = 1e-6


def validate_finite_mdp(
    p: np.ndarray,
    r: np.ndarray,
    horizon: int,
    gamma: float,
) -> None:
    """
    Validate shapes and values for finite-horizon DP.

    Args:
        p: transition tensor of shape ``(S, A, S)``.
        r: reward tensor of shape ``(S, A, S)``.
        horizon: positive finite integer horizon.
        gamma: discount factor in ``(0, 1]``.

    Raises:
        ValueError: if any of the shape/value constraints is violated.
    """
    if not isinstance(p, np.ndarray) or p.ndim != 3:
        raise ValueError(
            f"p must be a 3-D np.ndarray with shape (S, A, S); got "
            f"type={type(p).__name__}, "
            f"ndim={getattr(p, 'ndim', None)}."
        )
    if p.shape[0] != p.shape[2]:
        raise ValueError(
            f"p must have shape (S, A, S) with p.shape[0] == p.shape[2]; "
            f"got shape {p.shape}."
        )
    if not isinstance(r, np.ndarray) or r.shape != p.shape:
        raise ValueError(
            f"r must be an np.ndarray with the same shape as p; "
            f"got p.shape={p.shape}, r.shape={getattr(r, 'shape', None)}."
        )

    # Horizon must be a finite positive integer. np.inf is explicitly rejected.
    if isinstance(horizon, float) and not np.isfinite(horizon):
        raise ValueError(
            f"horizon must be a finite positive integer; got {horizon!r}."
        )
    if not isinstance(horizon, (int, np.integer)):
        raise ValueError(
            f"horizon must be an int; got type={type(horizon).__name__}, "
            f"value={horizon!r}."
        )
    if horizon <= 0:
        raise ValueError(
            f"horizon must be a positive integer; got {horizon}."
        )

    if not isinstance(gamma, (int, float, np.floating, np.integer)):
        raise ValueError(
            f"gamma must be a real scalar; got type={type(gamma).__name__}."
        )
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError(
            f"gamma must lie in (0, 1]; got {gamma}."
        )

    # Row-stochastic check per (s, a), with absorbing all-zero rows allowed.
    row_sums = p.sum(axis=2)
    is_zero_row = np.all(p == 0.0, axis=2)
    bad = (~is_zero_row) & (np.abs(row_sums - 1.0) > _PROB_ROW_TOL)
    if np.any(bad):
        idx = np.argwhere(bad)
        s0, a0 = int(idx[0, 0]), int(idx[0, 1])
        raise ValueError(
            f"p rows must sum to 1 within tol={_PROB_ROW_TOL} "
            f"(absorbing all-zero rows allowed); "
            f"{bad.sum()} violating (s, a) pairs, first at "
            f"(s={s0}, a={a0}) with row_sum={row_sums[s0, a0]:.6g}."
        )


def extract_mdp_arrays(mdp) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Extract ``(p, r, horizon, gamma)`` from a MushroomRL ``FiniteMDP``.

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP`
            (or any object exposing ``p``, ``r``, and ``info.{gamma, horizon}``).

    Returns:
        Tuple ``(p, r, horizon, gamma)`` where ``p, r`` are cast to
        ``float64`` copies, ``horizon`` is an ``int``, and ``gamma`` is a
        ``float``.

    Raises:
        ValueError: if ``horizon`` is not finite or any of the shape/value
            checks in :func:`validate_finite_mdp` fails.
    """
    raw_horizon = mdp.info.horizon
    if isinstance(raw_horizon, float) and not np.isfinite(raw_horizon):
        raise ValueError(
            "extract_mdp_arrays requires a finite horizon; "
            f"got mdp.info.horizon={raw_horizon!r}."
        )
    horizon = int(raw_horizon)
    gamma = float(mdp.info.gamma)

    p = np.asarray(mdp.p, dtype=np.float64).copy()
    r = np.asarray(mdp.r, dtype=np.float64).copy()

    validate_finite_mdp(p, r, horizon, gamma)
    return p, r, horizon, gamma


def expected_reward(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute the ``(S, A)`` expected-reward matrix.

    ``R_bar[s, a] = sum_{s'} P[s, a, s'] * R[s, a, s']``.

    Args:
        p: transition tensor of shape ``(S, A, S)``.
        r: reward tensor of shape ``(S, A, S)``.

    Returns:
        ``R_bar`` of shape ``(S, A)`` as ``float64``.
    """
    # Elementwise weighted sum over s'.
    return np.einsum("ijk,ijk->ij", p, r).astype(np.float64, copy=False)


def allocate_value_tables(
    n_states: int,
    n_actions: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Allocate zeroed canonical stage-indexed arrays.

    Args:
        n_states: number of states ``S``.
        n_actions: number of actions ``A``.
        horizon: positive integer horizon ``T``.

    Returns:
        Tuple ``(Q, V, pi)`` with shapes
        ``Q: (T, S, A) float64``,
        ``V: (T + 1, S) float64`` (``V[T, :] = 0`` by construction),
        ``pi: (T, S) int64``.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError(
            f"horizon must be a positive integer; got {horizon!r}."
        )
    if not isinstance(n_states, (int, np.integer)) or n_states <= 0:
        raise ValueError(
            f"n_states must be a positive integer; got {n_states!r}."
        )
    if not isinstance(n_actions, (int, np.integer)) or n_actions <= 0:
        raise ValueError(
            f"n_actions must be a positive integer; got {n_actions!r}."
        )

    Q = np.zeros((int(horizon), int(n_states), int(n_actions)), dtype=np.float64)
    V = np.zeros((int(horizon) + 1, int(n_states)), dtype=np.float64)
    pi = np.zeros((int(horizon), int(n_states)), dtype=np.int64)
    return Q, V, pi


def bellman_q_backup(
    t: int,
    V: np.ndarray,
    r_bar: np.ndarray,
    p: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    One-step Bellman Q-backup at stage ``t``.

    ``Q[t, s, a] = R_bar[s, a] + gamma * sum_{s'} P[s, a, s'] * V[t + 1, s']``.

    Args:
        t: stage index in ``0..T - 1``.
        V: value table of shape ``(T + 1, S)``.
        r_bar: expected-reward matrix of shape ``(S, A)``.
        p: transition tensor of shape ``(S, A, S)``.
        gamma: discount factor.

    Returns:
        ``Q_t`` of shape ``(S, A)`` as ``float64``.
    """
    if not (0 <= t < V.shape[0] - 1):
        raise ValueError(
            f"t must satisfy 0 <= t < T (= V.shape[0] - 1 = {V.shape[0] - 1}); "
            f"got t={t}."
        )
    v_next = V[t + 1]                                   # (S,)
    continuation = np.einsum("ijk,k->ij", p, v_next)    # (S, A)
    return (r_bar + float(gamma) * continuation).astype(np.float64, copy=False)


def bellman_v_from_q(Q_t: np.ndarray) -> np.ndarray:
    """
    ``V[t, s] = max_a Q[t, s, a]``.

    Args:
        Q_t: Q-slice of shape ``(S, A)``.

    Returns:
        ``V_t`` of shape ``(S,)``.
    """
    if Q_t.ndim != 2:
        raise ValueError(
            f"Q_t must be 2-D with shape (S, A); got ndim={Q_t.ndim}."
        )
    return Q_t.max(axis=1)


def greedy_policy(Q_t: np.ndarray) -> np.ndarray:
    """
    ``pi[t, s] = argmax_a Q[t, s, a]``.

    Ties are broken by ``np.argmax`` (lowest action index wins).

    Args:
        Q_t: Q-slice of shape ``(S, A)``.

    Returns:
        ``pi_t`` of shape ``(S,)`` as ``int64``.
    """
    if Q_t.ndim != 2:
        raise ValueError(
            f"Q_t must be 2-D with shape (S, A); got ndim={Q_t.ndim}."
        )
    return Q_t.argmax(axis=1).astype(np.int64, copy=False)


def bellman_q_policy_backup(
    t: int,
    V: np.ndarray,
    r_bar: np.ndarray,
    p: np.ndarray,
    gamma: float,
    pi_t: np.ndarray,
) -> np.ndarray:
    """
    Fixed-policy Q-backup at stage ``t``.

    Fills only the columns selected by ``pi_t``:
    ``Q[t, s, pi_t[s]] = R_bar[s, pi_t[s]]
                        + gamma * sum_{s'} P[s, pi_t[s], s'] * V[t + 1, s']``
    and leaves the remaining columns as zero.

    Args:
        t: stage index in ``0..T - 1``.
        V: value table of shape ``(T + 1, S)``.
        r_bar: expected-reward matrix of shape ``(S, A)``.
        p: transition tensor of shape ``(S, A, S)``.
        gamma: discount factor.
        pi_t: deterministic action per state, shape ``(S,)`` integer.

    Returns:
        ``Q_t`` of shape ``(S, A)`` as ``float64``.
    """
    if not (0 <= t < V.shape[0] - 1):
        raise ValueError(
            f"t must satisfy 0 <= t < T (= V.shape[0] - 1 = {V.shape[0] - 1}); "
            f"got t={t}."
        )
    S, A = r_bar.shape
    if pi_t.shape != (S,):
        raise ValueError(
            f"pi_t must have shape ({S},); got {pi_t.shape}."
        )
    pi_int = np.asarray(pi_t, dtype=np.int64)
    if np.any(pi_int < 0) or np.any(pi_int >= A):
        raise ValueError(
            f"pi_t entries must be in [0, {A}); got "
            f"min={int(pi_int.min())}, max={int(pi_int.max())}."
        )

    v_next = V[t + 1]                                   # (S,)
    states = np.arange(S)
    # P[s, pi_t[s], :] → shape (S, S); dot with v_next → (S,)
    p_pi = p[states, pi_int, :]                         # (S, S)
    continuation = p_pi @ v_next                        # (S,)
    r_pi = r_bar[states, pi_int]                        # (S,)

    Q_t = np.zeros((S, A), dtype=np.float64)
    Q_t[states, pi_int] = r_pi + float(gamma) * continuation
    return Q_t


def sup_norm_residual(V_new: np.ndarray, V_old: np.ndarray) -> float:
    """
    ``||V_new - V_old||_inf``. Broadcasts over any shape.

    Args:
        V_new: updated value array.
        V_old: previous value array (same broadcastable shape).

    Returns:
        Scalar sup-norm of the difference as a Python ``float``.
    """
    return float(np.max(np.abs(V_new - V_old)))


def deterministic_policy_array(
    horizon: int,
    n_states: int,
    action_per_state: np.ndarray,
) -> np.ndarray:
    """
    Build a time-stationary deterministic policy array.

    ``pi[t, s] = action_per_state[s]`` for all ``t in 0..T - 1``.

    Args:
        horizon: positive integer horizon ``T``.
        n_states: number of states ``S``.
        action_per_state: 1-D integer array of shape ``(S,)``.

    Returns:
        ``pi`` of shape ``(T, S)`` as ``int64``.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError(
            f"horizon must be a positive integer; got {horizon!r}."
        )
    if not isinstance(n_states, (int, np.integer)) or n_states <= 0:
        raise ValueError(
            f"n_states must be a positive integer; got {n_states!r}."
        )
    aps = np.asarray(action_per_state, dtype=np.int64)
    if aps.shape != (int(n_states),):
        raise ValueError(
            f"action_per_state must have shape ({n_states},); "
            f"got {aps.shape}."
        )
    return np.broadcast_to(aps, (int(horizon), int(n_states))).astype(
        np.int64, copy=True
    )
