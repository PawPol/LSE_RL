"""Compute the transitions payload for ``transitions.npz``.

Formulae (spec S7.1)::

    margin_beta0     = reward - v_next_beta0           (NO gamma)
    td_target_beta0  = reward + gamma * v_next_beta0
    td_error_beta0   = td_target_beta0 - q_current_beta0

The margin drives allocation geometry in the responsibility operator
rho*(r, v) = sigma(beta*(r - v) + log(1/gamma)), which depends on
(r - v), NOT (r - gamma*v).  The TD target remains standard Bellman.

See :mod:`experiments.weighted_lse_dp.common.schemas` for the schema
contract (``TRANSITIONS_ARRAYS``, ``CALIBRATION_ARRAYS``,
``MARGIN_BETA0_FORMULA``).

**Per-stage calibration aggregator** (Task 33 / spec S7.2) consumes the
per-transition payload and produces summary statistics whose keys match
:data:`~experiments.weighted_lse_dp.common.schemas.CALIBRATION_ARRAYS`.
The aggregator is deterministic: given the same input arrays it always
produces byte-identical output (no wall-clock seeds, no unsorted
iteration).
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.schemas import (  # noqa: E402
    CALIBRATION_ARRAYS,
    MARGIN_BETA0_FORMULA,
    TRANSITIONS_ARRAYS,
)

__all__ = [
    "aggregate_calibration_stats",
    "aggregate_calibration_stats_from_file",
    "build_calibration_stats_from_dp_tables",
    "build_transitions_payload",
    "build_transitions_payload_from_lists",
]


def build_transitions_payload(
    *,
    episode_index: np.ndarray,   # (N,) int64
    t: np.ndarray,               # (N,) int64  -- stage within episode
    state: np.ndarray,           # (N,) int64  -- UN-augmented base state id
    action: np.ndarray,          # (N,) int64
    reward: np.ndarray,          # (N,) float64
    next_state: np.ndarray,      # (N,) int64
    absorbing: np.ndarray,       # (N,) bool
    last: np.ndarray,            # (N,) bool
    q_current_beta0: np.ndarray, # (N,) float64  -- Q[t, state, action]
    v_next_beta0: np.ndarray,    # (N,) float64  -- V[t+1, next_state]
    gamma: float,
) -> dict[str, np.ndarray]:
    """Build the full transitions payload dict from raw episode data.

    Every input array must be 1-D with the same length N > 0.  The
    returned dict contains all 13 keys from
    :data:`~schemas.TRANSITIONS_ARRAYS`.

    Parameters
    ----------
    episode_index, t, state, action, reward, next_state, absorbing, last:
        Raw per-transition data logged by the RL callback.
    q_current_beta0:
        Q(t, state, action) under the beta=0 (standard Bellman) policy.
    v_next_beta0:
        V(t+1, next_state) under the beta=0 policy.
    gamma:
        Discount factor.  Used in ``td_target_beta0`` but NOT in
        ``margin_beta0``.

    Returns
    -------
    dict[str, np.ndarray]
        Payload dict ready for :meth:`RunWriter.set_transitions`.

    Raises
    ------
    ValueError
        If input arrays have mismatched lengths or N == 0.
    """
    # -- validate shapes ------------------------------------------------
    arrays_by_name: dict[str, np.ndarray] = {
        "episode_index": np.asarray(episode_index),
        "t": np.asarray(t),
        "state": np.asarray(state),
        "action": np.asarray(action),
        "reward": np.asarray(reward),
        "next_state": np.asarray(next_state),
        "absorbing": np.asarray(absorbing),
        "last": np.asarray(last),
        "q_current_beta0": np.asarray(q_current_beta0),
        "v_next_beta0": np.asarray(v_next_beta0),
    }

    lengths = {name: arr.shape[0] if arr.ndim >= 1 else 0
               for name, arr in arrays_by_name.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(lengths.items()))
        raise ValueError(
            f"All input arrays must have the same length; got: {detail}"
        )

    (n,) = unique_lengths
    if n == 0:
        raise ValueError("Input arrays must have length N > 0 (got N=0)")

    for name, arr in arrays_by_name.items():
        if arr.ndim != 1:
            raise ValueError(
                f"Array '{name}' must be 1-D, got shape {arr.shape!r}"
            )

    # -- derived arrays -------------------------------------------------
    _reward = arrays_by_name["reward"].astype(np.float64, copy=False)
    _v_next = arrays_by_name["v_next_beta0"].astype(np.float64, copy=False)
    _q_curr = arrays_by_name["q_current_beta0"].astype(np.float64, copy=False)

    # margin_beta0 = reward - v_next_beta0   (NO gamma -- spec S7.1)
    margin_beta0 = _reward - _v_next

    # td_target_beta0 = reward + gamma * v_next_beta0  (standard Bellman)
    td_target_beta0 = _reward + gamma * _v_next

    # td_error_beta0 = td_target_beta0 - q_current_beta0
    td_error_beta0 = td_target_beta0 - _q_curr

    # -- assemble payload -----------------------------------------------
    payload: dict[str, np.ndarray] = {
        **arrays_by_name,
        "margin_beta0": margin_beta0,
        "td_target_beta0": td_target_beta0,
        "td_error_beta0": td_error_beta0,
    }

    # Sanity: every required key must be present.
    missing = [k for k in TRANSITIONS_ARRAYS if k not in payload]
    if missing:  # pragma: no cover — defensive; should never trigger
        raise RuntimeError(
            f"Internal error: assembled payload is missing keys {missing!r}"
        )

    return payload


def build_transitions_payload_from_lists(
    *,
    episode_index: Sequence[int],
    t: Sequence[int],
    state: Sequence[int],
    action: Sequence[int],
    reward: Sequence[float],
    next_state: Sequence[int],
    absorbing: Sequence[bool],
    last: Sequence[bool],
    q_current_beta0: Sequence[float],
    v_next_beta0: Sequence[float],
    gamma: float,
) -> dict[str, np.ndarray]:
    """Convenience wrapper: same as :func:`build_transitions_payload` but
    accepts Python lists (or any sequences) instead of numpy arrays.

    Useful for the RL callback that accumulates data row-by-row into
    plain Python lists before flushing.
    """
    return build_transitions_payload(
        episode_index=np.asarray(episode_index, dtype=np.int64),
        t=np.asarray(t, dtype=np.int64),
        state=np.asarray(state, dtype=np.int64),
        action=np.asarray(action, dtype=np.int64),
        reward=np.asarray(reward, dtype=np.float64),
        next_state=np.asarray(next_state, dtype=np.int64),
        absorbing=np.asarray(absorbing, dtype=bool),
        last=np.asarray(last, dtype=bool),
        q_current_beta0=np.asarray(q_current_beta0, dtype=np.float64),
        v_next_beta0=np.asarray(v_next_beta0, dtype=np.float64),
        gamma=gamma,
    )


# ---------------------------------------------------------------------------
# Per-stage calibration aggregator (Task 33 / spec S7.2)
# ---------------------------------------------------------------------------


def aggregate_calibration_stats(
    transitions: dict[str, np.ndarray],
    *,
    horizon: int,
    bellman_residuals: dict[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate per-transition data into per-stage calibration statistics.

    Parameters
    ----------
    transitions:
        Dict whose keys are a superset of
        :data:`~schemas.TRANSITIONS_ARRAYS`.  The arrays ``t``,
        ``reward``, ``v_next_beta0``, ``q_current_beta0``, and
        ``margin_beta0`` are consumed directly.
    horizon:
        Episode length *H*.  Output arrays have shape ``(H+1,)``
        covering stages ``0, 1, ..., H``.
    bellman_residuals:
        Optional mapping ``{stage_index: 1-D array_of_residuals}``.
        When ``None`` (or when a stage has no entry), the corresponding
        ``bellman_residual_mean`` / ``bellman_residual_std`` entries are
        ``NaN``.

    Returns
    -------
    dict[str, np.ndarray]
        Dict with all 36 keys from :data:`~schemas.CALIBRATION_ARRAYS`.
        Phase I per-stage fields have shape ``(H+1,)``; Phase II
        per-stage fields have shape ``(H+1,)`` (populated from
        transitions); Phase II scalar fields have shape ``(1,)`` and
        default to ``NaN`` (to be filled by Phase II runners).
    """
    # -- validate required transition keys ---------------------------------
    missing = [k for k in TRANSITIONS_ARRAYS if k not in transitions]
    if missing:
        raise KeyError(
            f"transitions dict missing required keys: {missing!r}"
        )

    n_stages = horizon + 1  # stages 0 .. H

    t_arr = np.asarray(transitions["t"], dtype=np.int64)
    reward = np.asarray(transitions["reward"], dtype=np.float64)
    v_next = np.asarray(transitions["v_next_beta0"], dtype=np.float64)
    q_curr = np.asarray(transitions["q_current_beta0"], dtype=np.float64)
    margin = np.asarray(transitions["margin_beta0"], dtype=np.float64)

    # -- allocate output arrays (NaN default for floats) -------------------
    stage = np.arange(n_stages, dtype=np.int64)
    count = np.zeros(n_stages, dtype=np.int64)
    reward_mean = np.full(n_stages, np.nan, dtype=np.float64)
    reward_std = np.full(n_stages, np.nan, dtype=np.float64)
    v_next_mean = np.full(n_stages, np.nan, dtype=np.float64)
    v_next_std = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q05 = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q25 = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q50 = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q75 = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q95 = np.full(n_stages, np.nan, dtype=np.float64)
    pos_margin_mean = np.full(n_stages, np.nan, dtype=np.float64)
    neg_margin_mean = np.full(n_stages, np.nan, dtype=np.float64)
    max_abs_v_next = np.full(n_stages, np.nan, dtype=np.float64)
    max_abs_q_current = np.full(n_stages, np.nan, dtype=np.float64)
    bellman_residual_mean = np.full(n_stages, np.nan, dtype=np.float64)
    bellman_residual_std = np.full(n_stages, np.nan, dtype=np.float64)
    aligned_margin_freq = np.full(n_stages, np.nan, dtype=np.float64)

    # Phase II per-stage fields (default NaN, populated from transitions)
    aligned_positive_mean = np.full(n_stages, np.nan, dtype=np.float64)
    aligned_negative_mean = np.full(n_stages, np.nan, dtype=np.float64)
    td_target_std = np.full(n_stages, np.nan, dtype=np.float64)
    td_error_std = np.full(n_stages, np.nan, dtype=np.float64)

    # -- per-stage aggregation ---------------------------------------------
    # Pre-extract td_target and td_error if available (may not be present
    # in older Phase I transition files, so guard with .get).
    _td_target_arr = transitions.get("td_target_beta0")
    _td_error_arr = transitions.get("td_error_beta0")
    if _td_target_arr is not None:
        _td_target_arr = np.asarray(_td_target_arr, dtype=np.float64)
    if _td_error_arr is not None:
        _td_error_arr = np.asarray(_td_error_arr, dtype=np.float64)

    for t in range(n_stages):
        mask = t_arr == t
        n_t = int(mask.sum())
        count[t] = n_t
        if n_t == 0:
            continue

        r_t = reward[mask]
        v_t = v_next[mask]
        q_t = q_curr[mask]
        m_t = margin[mask]

        reward_mean[t] = np.mean(r_t)
        reward_std[t] = np.std(r_t)          # ddof=0 (population std)
        v_next_mean[t] = np.mean(v_t)
        v_next_std[t] = np.std(v_t)

        margin_q05[t] = np.percentile(m_t, 5, method="linear")
        margin_q25[t] = np.percentile(m_t, 25, method="linear")
        margin_q50[t] = np.percentile(m_t, 50, method="linear")
        margin_q75[t] = np.percentile(m_t, 75, method="linear")
        margin_q95[t] = np.percentile(m_t, 95, method="linear")

        pos_margin_mean[t] = np.mean(np.maximum(m_t, 0.0))
        neg_margin_mean[t] = np.mean(np.maximum(-m_t, 0.0))

        max_abs_v_next[t] = np.max(np.abs(v_t))
        max_abs_q_current[t] = np.max(np.abs(q_t))

        # Aligned-margin frequency: fraction of transitions with margin_beta0 > 0
        aligned_margin_freq[t] = np.mean(m_t > 0.0)

        # Phase II per-stage fields
        aligned_positive_mean[t] = np.mean(np.maximum(m_t, 0.0))
        aligned_negative_mean[t] = np.mean(np.maximum(-m_t, 0.0))
        if _td_target_arr is not None:
            td_target_std[t] = np.std(_td_target_arr[mask])
        if _td_error_arr is not None:
            td_error_std[t] = np.std(_td_error_arr[mask])

        # Bellman residuals (optional, per-stage)
        if bellman_residuals is not None and t in bellman_residuals:
            br = np.asarray(bellman_residuals[t], dtype=np.float64)
            if br.size > 0:
                bellman_residual_mean[t] = np.mean(br)
                bellman_residual_std[t] = np.std(br)

    # -- Phase II scalar fields (shape (1,), default NaN) --------------------
    _nan1 = np.array([np.nan], dtype=np.float64)

    return {
        "stage": stage,
        "count": count,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "v_next_mean": v_next_mean,
        "v_next_std": v_next_std,
        "margin_q05": margin_q05,
        "margin_q25": margin_q25,
        "margin_q50": margin_q50,
        "margin_q75": margin_q75,
        "margin_q95": margin_q95,
        "pos_margin_mean": pos_margin_mean,
        "neg_margin_mean": neg_margin_mean,
        "max_abs_v_next": max_abs_v_next,
        "max_abs_q_current": max_abs_q_current,
        "bellman_residual_mean": bellman_residual_mean,
        "bellman_residual_std": bellman_residual_std,
        "aligned_margin_freq": aligned_margin_freq,
        # Phase II per-stage
        "aligned_positive_mean": aligned_positive_mean,
        "aligned_negative_mean": aligned_negative_mean,
        "td_target_std": td_target_std,
        "td_error_std": td_error_std,
        # Phase II event-level scalars
        "jackpot_event_rate": _nan1.copy(),
        "catastrophe_event_rate": _nan1.copy(),
        "regime_shift_episode": _nan1.copy(),
        "hazard_hit_rate": _nan1.copy(),
        # Phase II tail-risk scalars
        "return_cvar_5pct": _nan1.copy(),
        "return_cvar_10pct": _nan1.copy(),
        "return_top5pct_mean": _nan1.copy(),
        "event_rate": _nan1.copy(),
        "event_conditioned_return": _nan1.copy(),
        # Phase II adaptation scalars
        "adaptation_pre_change_auc": _nan1.copy(),
        "adaptation_post_change_auc": _nan1.copy(),
        "adaptation_lag_50pct": _nan1.copy(),
        "adaptation_lag_75pct": _nan1.copy(),
        "adaptation_lag_90pct": _nan1.copy(),
    }


# ---------------------------------------------------------------------------
# Convenience: load from file and aggregate
# ---------------------------------------------------------------------------


def build_calibration_stats_from_dp_tables(
    Q: np.ndarray,   # (H, S, A) float64 — same layout as planner.Q
    V: np.ndarray,   # (H+1, S) float64
    P: np.ndarray,   # (S, A, S) float64 — transition probabilities P[s,a,s']
    R: np.ndarray,   # (S, A, S) float64 — reward per (s,a,s')
    *,
    gamma: float,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Build per-stage calibration stats directly from exact DP tables.

    For exact DP runs the Q/V tables are already available in closed form,
    so we can bypass per-transition logging and compute the calibration
    statistics analytically.  The returned dict has the same 18 keys as
    :func:`aggregate_calibration_stats` (matching
    :data:`~schemas.CALIBRATION_ARRAYS`).

    Parameters
    ----------
    Q : shape ``(H, S, A)``
        Stage-indexed action-value table from the DP planner (stages
        ``t = 0..H-1`` only), matching :attr:`mushroom_rl.algorithms.value.dp`
        planners.
    V : shape ``(H+1, S)``
        State value table (``V[t, s] = max_a Q[t, s, a]`` for greedy;
        ``V[H, :] = 0``).
    P : shape ``(S, A, S)``
        Transition probability matrix ``P[s, a, s']``.
    R : shape ``(S, A, S)``
        Reward matrix ``R[s, a, s']``.
    gamma : float
        Discount factor.
    horizon : int
        Episode length *H*.

    Returns
    -------
    dict[str, np.ndarray]
        All 36 keys from :data:`~schemas.CALIBRATION_ARRAYS`.  Phase I
        per-stage fields have shape ``(H+1,)``; Phase II per-stage fields
        have shape ``(H+1,)``; Phase II scalar fields have shape ``(1,)``
        and default to ``NaN``.  Stage *H* (terminal) has ``count=0`` and
        ``NaN`` for all float fields.

    Notes
    -----
    **Provenance vs. RL empirical stats**: this function computes statistics
    over the analytic uniform distribution on ``(state, action)`` pairs (i.e.
    ``count[t] = S * A`` and quantiles are over the full model table).  RL
    runs instead compute empirical statistics from actual transition logs.
    The two modes share the same :data:`~schemas.CALIBRATION_ARRAYS` schema
    but differ in semantics.  The NPZ schema header written by
    :class:`~schemas.RunWriter` distinguishes them via the ``storage_mode``
    key: ``"dp_stagewise"`` for DP runs, ``"rl_online"`` for RL runs.
    Downstream consumers **must** check ``storage_mode`` before comparing
    ``count`` or margin statistics across run types.
    """
    Q = np.asarray(Q, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    H = horizon
    n_stages = H + 1
    S = P.shape[0]
    A = P.shape[1]

    # Validate shapes — Q has one row per decision stage (H rows), not H+1.
    if Q.shape != (H, S, A):
        raise ValueError(
            f"Q shape {Q.shape} != expected ({H}, {S}, {A})"
        )
    if V.shape != (n_stages, S):
        raise ValueError(
            f"V shape {V.shape} != expected ({n_stages}, {S})"
        )
    if P.shape != (S, A, S):
        raise ValueError(f"P shape {P.shape} != expected ({S}, {A}, {S})")
    if R.shape != (S, A, S):
        raise ValueError(f"R shape {R.shape} != expected ({S}, {A}, {S})")

    # Pre-compute expected reward and expected V_next for all (s, a)
    # reward_sa[s, a] = sum_{s'} P[s,a,s'] * R[s,a,s']
    reward_sa = (P * R).sum(axis=-1)  # (S, A)

    # -- allocate output arrays (NaN default for floats) -------------------
    stage = np.arange(n_stages, dtype=np.int64)
    count = np.zeros(n_stages, dtype=np.int64)
    reward_mean = np.full(n_stages, np.nan, dtype=np.float64)
    reward_std = np.full(n_stages, np.nan, dtype=np.float64)
    v_next_mean = np.full(n_stages, np.nan, dtype=np.float64)
    v_next_std = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q05 = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q25 = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q50 = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q75 = np.full(n_stages, np.nan, dtype=np.float64)
    margin_q95 = np.full(n_stages, np.nan, dtype=np.float64)
    pos_margin_mean = np.full(n_stages, np.nan, dtype=np.float64)
    neg_margin_mean = np.full(n_stages, np.nan, dtype=np.float64)
    max_abs_v_next = np.full(n_stages, np.nan, dtype=np.float64)
    max_abs_q_current = np.full(n_stages, np.nan, dtype=np.float64)
    bellman_residual_mean = np.full(n_stages, np.nan, dtype=np.float64)
    bellman_residual_std = np.full(n_stages, np.nan, dtype=np.float64)
    aligned_margin_freq = np.full(n_stages, np.nan, dtype=np.float64)

    # Phase II per-stage fields
    aligned_positive_mean = np.full(n_stages, np.nan, dtype=np.float64)
    aligned_negative_mean = np.full(n_stages, np.nan, dtype=np.float64)
    td_target_std = np.full(n_stages, np.nan, dtype=np.float64)
    td_error_std = np.full(n_stages, np.nan, dtype=np.float64)

    for t in range(H):
        # v_next_sa[s, a] = sum_{s'} P[s,a,s'] * V[t+1, s']
        v_next_sa = P @ V[t + 1]  # (S, A, S) @ (S,) -> (S, A)

        # Flatten to (S*A,) for stats computation
        r_flat = reward_sa.ravel()           # (S*A,)
        v_flat = v_next_sa.ravel()           # (S*A,)
        q_flat = Q[t].ravel()                # (S*A,)

        # margin_beta0 = reward - v_next_beta0  (NO gamma)
        margin_flat = r_flat - v_flat

        # td_target_beta0 = reward + gamma * v_next_beta0
        td_target_flat = r_flat + gamma * v_flat

        # td_error_beta0 = td_target - q_current
        td_error_flat = td_target_flat - q_flat

        n_t = S * A
        count[t] = n_t

        reward_mean[t] = np.mean(r_flat)
        reward_std[t] = np.std(r_flat)
        v_next_mean[t] = np.mean(v_flat)
        v_next_std[t] = np.std(v_flat)

        margin_q05[t] = np.percentile(margin_flat, 5, method="linear")
        margin_q25[t] = np.percentile(margin_flat, 25, method="linear")
        margin_q50[t] = np.percentile(margin_flat, 50, method="linear")
        margin_q75[t] = np.percentile(margin_flat, 75, method="linear")
        margin_q95[t] = np.percentile(margin_flat, 95, method="linear")

        pos_margin_mean[t] = np.mean(np.maximum(margin_flat, 0.0))
        neg_margin_mean[t] = np.mean(np.maximum(-margin_flat, 0.0))

        max_abs_v_next[t] = np.max(np.abs(v_flat))
        max_abs_q_current[t] = np.max(np.abs(q_flat))

        # Aligned-margin frequency: fraction of (s,a) pairs with margin_beta0 > 0
        aligned_margin_freq[t] = np.mean(margin_flat > 0.0)

        bellman_residual_mean[t] = np.mean(td_error_flat)
        bellman_residual_std[t] = np.std(td_error_flat)

        # Phase II per-stage fields
        aligned_positive_mean[t] = np.mean(np.maximum(margin_flat, 0.0))
        aligned_negative_mean[t] = np.mean(np.maximum(-margin_flat, 0.0))
        td_target_std[t] = np.std(td_target_flat)
        td_error_std[t] = np.std(td_error_flat)

    # Stage H: count=0, all float fields remain NaN (terminal).

    # -- Phase II scalar fields (shape (1,), default NaN) --------------------
    _nan1 = np.array([np.nan], dtype=np.float64)

    return {
        "stage": stage,
        "count": count,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "v_next_mean": v_next_mean,
        "v_next_std": v_next_std,
        "margin_q05": margin_q05,
        "margin_q25": margin_q25,
        "margin_q50": margin_q50,
        "margin_q75": margin_q75,
        "margin_q95": margin_q95,
        "pos_margin_mean": pos_margin_mean,
        "neg_margin_mean": neg_margin_mean,
        "max_abs_v_next": max_abs_v_next,
        "max_abs_q_current": max_abs_q_current,
        "bellman_residual_mean": bellman_residual_mean,
        "bellman_residual_std": bellman_residual_std,
        "aligned_margin_freq": aligned_margin_freq,
        # Phase II per-stage
        "aligned_positive_mean": aligned_positive_mean,
        "aligned_negative_mean": aligned_negative_mean,
        "td_target_std": td_target_std,
        "td_error_std": td_error_std,
        # Phase II event-level scalars
        "jackpot_event_rate": _nan1.copy(),
        "catastrophe_event_rate": _nan1.copy(),
        "regime_shift_episode": _nan1.copy(),
        "hazard_hit_rate": _nan1.copy(),
        # Phase II tail-risk scalars
        "return_cvar_5pct": _nan1.copy(),
        "return_cvar_10pct": _nan1.copy(),
        "return_top5pct_mean": _nan1.copy(),
        "event_rate": _nan1.copy(),
        "event_conditioned_return": _nan1.copy(),
        # Phase II adaptation scalars
        "adaptation_pre_change_auc": _nan1.copy(),
        "adaptation_post_change_auc": _nan1.copy(),
        "adaptation_lag_50pct": _nan1.copy(),
        "adaptation_lag_75pct": _nan1.copy(),
        "adaptation_lag_90pct": _nan1.copy(),
    }


# ---------------------------------------------------------------------------
# Convenience: load from file and aggregate
# ---------------------------------------------------------------------------


def aggregate_calibration_stats_from_file(
    path: str | Path,
    *,
    horizon: int,
    bellman_residuals: dict[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Load transitions from an ``.npz`` file and aggregate.

    Thin wrapper around :func:`aggregate_calibration_stats` that handles
    the ``np.load`` call and dict conversion.

    Parameters
    ----------
    path:
        Path to a ``transitions.npz`` file whose keys are a superset of
        :data:`~schemas.TRANSITIONS_ARRAYS`.
    horizon:
        Episode length *H*; forwarded to
        :func:`aggregate_calibration_stats`.
    bellman_residuals:
        Optional; forwarded to :func:`aggregate_calibration_stats`.

    Returns
    -------
    dict[str, np.ndarray]
        Same as :func:`aggregate_calibration_stats`.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        transitions = {k: npz[k] for k in npz.files if k != "_schema"}
    return aggregate_calibration_stats(
        transitions, horizon=horizon, bellman_residuals=bellman_residuals,
    )
