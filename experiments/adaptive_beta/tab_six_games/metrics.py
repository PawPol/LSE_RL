"""Phase VIII delta metrics (spec §7.2).

Pure-function module implementing the sixteen Phase VIII delta metrics
defined in ``docs/specs/phase_VIII_tab_six_games.md`` §7.2. Every
function here:

* takes ``numpy`` arrays (or array-like) and returns either a scalar or
  a ``numpy`` array;
* is empty-array safe (no exception on ``len(arr) == 0``);
* uses the uniform floor ``EPSILON = 1e-8`` for any logarithmic or
  ratio kernel;
* uses ``numpy.log`` directly — **never** ``numpy.expm1`` or
  ``numpy.log1p`` (lessons.md #27 documents the negative-tail underflow
  failure mode that bans these primitives in Phase III+ code);
* depends only on ``numpy``;
* does not import any Phase III–VII runtime (no ``tab_operator``,
  ``schedules``, ``agents``, etc.). The downstream test suite (W2.B,
  ``tests/adaptive_beta/strategic_games/test_phase_VIII_metrics.py``)
  exercises these on synthetic traces.

The constant :data:`EPSILON` is pinned to ``1e-8`` to match
``src/lse_rl/operator/tab_operator._EPS_BETA``. Test 13.5 verifies the
delta-metric definitions match the spec verbatim.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

# Uniform numerical floor for logarithms / ratios across all delta
# metrics. Mirrors ``tab_operator._EPS_BETA`` so the classical-collapse
# semantics are consistent across the codebase.
EPSILON: float = 1e-8

__all__ = [
    "EPSILON",
    "contraction_reward",
    "empirical_contraction_ratio",
    "log_residual_reduction",
    "ucb_arm_count",
    "ucb_arm_value",
    "beta_clip_count",
    "beta_clip_frequency",
    "recovery_time_after_shift",
    "beta_sign_correct",
    "beta_lag_to_oracle",
    "regret_vs_oracle",
    "catastrophic_episodes",
    "worst_window_return_percentile",
    "trap_entries",
    "constraint_violations",
    "overflow_count",
]


# ---------------------------------------------------------------------------
# Kernel helpers (private)
# ---------------------------------------------------------------------------

def _log_residual_diff_kernel(
    bellman_residual_arr: np.ndarray, eps: float
) -> np.ndarray:
    """Shared kernel for ``contraction_reward`` / ``log_residual_reduction``.

    Returns ``log(R_e + eps) - log(R_{e+1} + eps)`` over consecutive
    pairs. Output shape is ``(len(arr) - 1,)``; an empty float64 array
    is returned for inputs of length ``<= 1``.

    Uses ``np.log`` directly per lessons.md #27 (no ``log1p``).
    """
    R = np.asarray(bellman_residual_arr, dtype=np.float64)
    if R.ndim != 1:
        R = R.reshape(-1)
    if R.size <= 1:
        return np.zeros(0, dtype=np.float64)
    log_R = np.log(R + eps)
    return log_R[:-1] - log_R[1:]


# ---------------------------------------------------------------------------
# §7.2 metric (1): contraction_reward
# ---------------------------------------------------------------------------

def contraction_reward(
    bellman_residual_arr: np.ndarray, eps: float = EPSILON
) -> np.ndarray:
    """Per-step empirical contraction reward ``M_e(β)`` (spec §7.2).

    ``M_e = log(R_e + eps) - log(R_{e+1} + eps)``, where ``R`` is the
    per-episode Bellman residual (NOT episode return; see spec §7.2
    line ``contraction_reward`` definition).

    Parameters
    ----------
    bellman_residual_arr : np.ndarray, shape (E,)
        Per-episode Bellman residuals ``R_0, R_1, ..., R_{E-1}``. Must
        be non-negative (residual norms). Empty / length-1 inputs
        return an empty float64 array.
    eps : float, default ``EPSILON`` (= 1e-8)
        Logarithmic floor.

    Returns
    -------
    np.ndarray, shape (max(E - 1, 0),), dtype float64
    """
    return _log_residual_diff_kernel(bellman_residual_arr, eps)


# ---------------------------------------------------------------------------
# §7.2 metric (2): empirical_contraction_ratio
# ---------------------------------------------------------------------------

def empirical_contraction_ratio(
    bellman_residual_arr: np.ndarray, eps: float = EPSILON
) -> np.ndarray:
    """Empirical contraction ratio ``(R_{e+1} + eps) / (R_e + eps)``
    (spec §7.2).

    A value ``< 1`` indicates contraction across the step; ``>= 1``
    indicates the residual did not shrink between consecutive
    episodes.

    Parameters
    ----------
    bellman_residual_arr : np.ndarray, shape (E,)
    eps : float, default ``EPSILON``

    Returns
    -------
    np.ndarray, shape (max(E - 1, 0),), dtype float64
    """
    R = np.asarray(bellman_residual_arr, dtype=np.float64)
    if R.ndim != 1:
        R = R.reshape(-1)
    if R.size <= 1:
        return np.zeros(0, dtype=np.float64)
    return (R[1:] + eps) / (R[:-1] + eps)


# ---------------------------------------------------------------------------
# §7.2 metric (3): log_residual_reduction (alias of contraction_reward)
# ---------------------------------------------------------------------------

def log_residual_reduction(
    bellman_residual_arr: np.ndarray, eps: float = EPSILON
) -> np.ndarray:
    """Logarithmic residual reduction (spec §7.2).

    Defined identically to :func:`contraction_reward`:
    ``log(R_e + eps) - log(R_{e+1} + eps)``. Implemented as a
    distinct entry-point delegating to the same kernel for naming
    parity with the spec-§7.2 metric inventory.
    """
    return _log_residual_diff_kernel(bellman_residual_arr, eps)


# ---------------------------------------------------------------------------
# §7.2 metric (4): ucb_arm_count
# ---------------------------------------------------------------------------

def ucb_arm_count(
    arm_idx_history: np.ndarray, n_arms: int = 7
) -> np.ndarray:
    """Per-arm pull count over the episode history (spec §7.2).

    Parameters
    ----------
    arm_idx_history : np.ndarray, shape (E,)
        Integer arm indices in ``[0, n_arms)``. Empty input returns a
        zero vector of shape ``(n_arms,)``.
    n_arms : int, default 7
        Number of arms in the UCB schedule (spec §6 fixes 7 arms).

    Returns
    -------
    np.ndarray, shape (n_arms,), dtype int64

    Raises
    ------
    ValueError
        If any entry in ``arm_idx_history`` is outside ``[0, n_arms)``.
    """
    arms = np.asarray(arm_idx_history, dtype=np.int64).reshape(-1)
    counts = np.zeros(int(n_arms), dtype=np.int64)
    if arms.size == 0:
        return counts
    if arms.min() < 0 or arms.max() >= n_arms:
        raise ValueError(
            f"arm_idx_history entries must lie in [0, {n_arms}); "
            f"observed min={int(arms.min())}, max={int(arms.max())}."
        )
    # np.bincount returns shape (max+1,); pad up to n_arms.
    bc = np.bincount(arms, minlength=int(n_arms))
    counts[: bc.shape[0]] = bc[: int(n_arms)].astype(np.int64)
    return counts


# ---------------------------------------------------------------------------
# §7.2 metric (5): ucb_arm_value
# ---------------------------------------------------------------------------

def ucb_arm_value(
    reward_history: np.ndarray,
    arm_idx_history: np.ndarray,
    n_arms: int = 7,
) -> np.ndarray:
    """Per-arm running mean reward (spec §7.2).

    Computes the exact cumulative mean for each arm via a Welford-style
    accumulator (single pass; mathematically exact for the final mean).
    Arms that were never pulled return ``0.0``.

    Parameters
    ----------
    reward_history : np.ndarray, shape (E,)
        Per-episode UCB reward signal aligned with ``arm_idx_history``.
    arm_idx_history : np.ndarray, shape (E,)
        Integer arm indices in ``[0, n_arms)``.
    n_arms : int, default 7

    Returns
    -------
    np.ndarray, shape (n_arms,), dtype float64

    Raises
    ------
    ValueError
        If lengths mismatch or arm indices are out of range.
    """
    rewards = np.asarray(reward_history, dtype=np.float64).reshape(-1)
    arms = np.asarray(arm_idx_history, dtype=np.int64).reshape(-1)
    means = np.zeros(int(n_arms), dtype=np.float64)
    if rewards.shape[0] != arms.shape[0]:
        raise ValueError(
            f"reward_history (len={rewards.shape[0]}) and arm_idx_history "
            f"(len={arms.shape[0]}) must have equal length."
        )
    if rewards.size == 0:
        return means
    if arms.min() < 0 or arms.max() >= n_arms:
        raise ValueError(
            f"arm_idx_history entries must lie in [0, {n_arms}); "
            f"observed min={int(arms.min())}, max={int(arms.max())}."
        )
    counts = np.zeros(int(n_arms), dtype=np.int64)
    # Welford recursion: m_n = m_{n-1} + (x_n - m_{n-1}) / n.
    for arm_idx, r in zip(arms, rewards):
        a = int(arm_idx)
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]
    # Arms never pulled stay at 0.0 (counts == 0); explicit reset for
    # numerical safety in case of accumulator drift on length-0 paths.
    means[counts == 0] = 0.0
    return means


# ---------------------------------------------------------------------------
# §7.2 metric (6): beta_clip_count
# ---------------------------------------------------------------------------

def beta_clip_count(
    beta_raw_arr: np.ndarray,
    beta_used_arr: np.ndarray,
    atol: float = EPSILON,
) -> int:
    """Count episodes where β was clipped (spec §7.2).

    A clip event is defined as
    ``|beta_raw - beta_used| > atol``.

    Parameters
    ----------
    beta_raw_arr : np.ndarray, shape (E,)
        Pre-clip schedule output.
    beta_used_arr : np.ndarray, shape (E,)
        Post-clip β actually deployed.
    atol : float, default ``EPSILON``
        Absolute tolerance below which raw and used are considered
        identical (no clip).

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If lengths mismatch.
    """
    raw = np.asarray(beta_raw_arr, dtype=np.float64).reshape(-1)
    used = np.asarray(beta_used_arr, dtype=np.float64).reshape(-1)
    if raw.shape[0] != used.shape[0]:
        raise ValueError(
            f"beta_raw_arr (len={raw.shape[0]}) and beta_used_arr "
            f"(len={used.shape[0]}) must have equal length."
        )
    if raw.size == 0:
        return 0
    return int(np.sum(np.abs(raw - used) > atol))


# ---------------------------------------------------------------------------
# §7.2 metric (7): beta_clip_frequency
# ---------------------------------------------------------------------------

def beta_clip_frequency(
    beta_raw_arr: np.ndarray,
    beta_used_arr: np.ndarray,
    atol: float = EPSILON,
) -> float:
    """Fraction of episodes where β was clipped (spec §7.2).

    Returns ``beta_clip_count / len(beta_raw_arr)``; returns ``0.0``
    for empty input.
    """
    raw = np.asarray(beta_raw_arr, dtype=np.float64).reshape(-1)
    if raw.size == 0:
        return 0.0
    return float(beta_clip_count(beta_raw_arr, beta_used_arr, atol)) / float(
        raw.size
    )


# ---------------------------------------------------------------------------
# §7.2 metric (8): recovery_time_after_shift
# ---------------------------------------------------------------------------

def recovery_time_after_shift(
    return_smooth_arr: np.ndarray,
    shift_episode: int,
    threshold: float,
) -> int:
    """Episodes from a regime shift to first ``return_smooth >= threshold``
    (spec §7.2).

    Parameters
    ----------
    return_smooth_arr : np.ndarray, shape (E,)
        Smoothed (e.g. rolling-mean) return trajectory.
    shift_episode : int
        Episode index at which the regime shift occurred.
    threshold : float
        Recovery threshold ``θ``.

    Returns
    -------
    int
        ``e - shift_episode`` for the smallest ``e >= shift_episode``
        with ``return_smooth_arr[e] >= threshold``. Returns ``-1`` if
        the threshold is never reached, or if ``shift_episode`` is
        out of range.
    """
    arr = np.asarray(return_smooth_arr, dtype=np.float64).reshape(-1)
    s = int(shift_episode)
    if arr.size == 0 or s >= arr.size:
        return -1
    if s < 0:
        s = 0
    tail = arr[s:]
    hits = np.where(tail >= threshold)[0]
    if hits.size == 0:
        return -1
    return int(hits[0])


# ---------------------------------------------------------------------------
# §7.2 metric (9): beta_sign_correct
# ---------------------------------------------------------------------------

def beta_sign_correct(
    beta_used_arr: np.ndarray, oracle_beta_arr: np.ndarray
) -> np.ndarray:
    """Per-episode β-sign correctness vector (spec §7.2).

    ``True`` iff ``np.sign(beta_used) == np.sign(oracle_beta)``. When
    the oracle is exactly ``0`` (classical regime), only
    ``beta_used == 0`` is considered correct; any nonzero ``beta_used``
    is incorrect at that index.

    Parameters
    ----------
    beta_used_arr : np.ndarray, shape (E,)
    oracle_beta_arr : np.ndarray, shape (E,)

    Returns
    -------
    np.ndarray, shape (E,), dtype bool

    Raises
    ------
    ValueError
        If lengths mismatch.
    """
    used = np.asarray(beta_used_arr, dtype=np.float64).reshape(-1)
    oracle = np.asarray(oracle_beta_arr, dtype=np.float64).reshape(-1)
    if used.shape[0] != oracle.shape[0]:
        raise ValueError(
            f"beta_used_arr (len={used.shape[0]}) and oracle_beta_arr "
            f"(len={oracle.shape[0]}) must have equal length."
        )
    if used.size == 0:
        return np.zeros(0, dtype=bool)
    # np.sign returns 0 for +0/-0/NaN-free zeros; -1, 0, +1 for negatives,
    # zeros, positives. The equality below correctly captures the
    # classical-regime rule (oracle == 0 ⇒ used must equal 0).
    return np.sign(used) == np.sign(oracle)


# ---------------------------------------------------------------------------
# §7.2 metric (10): beta_lag_to_oracle
# ---------------------------------------------------------------------------

def beta_lag_to_oracle(
    beta_used_arr: np.ndarray, oracle_beta_arr: np.ndarray
) -> np.ndarray:
    """Per-episode lag of ``beta_used`` to ``oracle_beta`` (spec §7.2).

    For each episode ``e``, the lag is
    ``e - argmin_{e' <= e} |beta_used[e'] - oracle_beta[e]|`` —
    i.e. how many episodes ago our deployed β was closest to the
    *current* oracle target. Lag ``0`` means the current ``beta_used``
    is the historical best match.

    On ties (``argmin`` not unique), ``np.argmin`` returns the smallest
    matching index, so the reported lag is the largest gap among ties
    (the most pessimistic). This is consistent with the spec's "last
    matched" wording.

    Parameters
    ----------
    beta_used_arr : np.ndarray, shape (E,)
    oracle_beta_arr : np.ndarray, shape (E,)

    Returns
    -------
    np.ndarray, shape (E,), dtype int64

    Raises
    ------
    ValueError
        If lengths mismatch.
    """
    used = np.asarray(beta_used_arr, dtype=np.float64).reshape(-1)
    oracle = np.asarray(oracle_beta_arr, dtype=np.float64).reshape(-1)
    if used.shape[0] != oracle.shape[0]:
        raise ValueError(
            f"beta_used_arr (len={used.shape[0]}) and oracle_beta_arr "
            f"(len={oracle.shape[0]}) must have equal length."
        )
    n = used.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    lag = np.zeros(n, dtype=np.int64)
    for e in range(n):
        # Search the prefix [0, e]. argmin returns the first minimum,
        # which gives the most pessimistic (largest) lag on ties — this
        # is intentional: ties favour the older match.
        diffs = np.abs(used[: e + 1] - oracle[e])
        best = int(np.argmin(diffs))
        lag[e] = e - best
    return lag


# ---------------------------------------------------------------------------
# §7.2 metric (11): regret_vs_oracle
# ---------------------------------------------------------------------------

def regret_vs_oracle(
    oracle_return_arr: np.ndarray, method_return_arr: np.ndarray
) -> float:
    """Cumulative regret against the oracle (spec §7.2).

    ``sum(oracle_return - method_return)`` over equal-length per-episode
    return trajectories. Empty input returns ``0.0``.

    Parameters
    ----------
    oracle_return_arr : np.ndarray, shape (E,)
    method_return_arr : np.ndarray, shape (E,)

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the two arrays have different lengths.
    """
    oracle = np.asarray(oracle_return_arr, dtype=np.float64).reshape(-1)
    method = np.asarray(method_return_arr, dtype=np.float64).reshape(-1)
    if oracle.shape[0] != method.shape[0]:
        raise ValueError(
            f"oracle_return_arr (len={oracle.shape[0]}) and "
            f"method_return_arr (len={method.shape[0]}) must have equal "
            "length."
        )
    if oracle.size == 0:
        return 0.0
    return float(np.sum(oracle - method))


# ---------------------------------------------------------------------------
# §7.2 metric (12): catastrophic_episodes
# ---------------------------------------------------------------------------

def catastrophic_episodes(
    return_arr: np.ndarray, theta_low: float
) -> int:
    """Count episodes with return ``<= theta_low`` (spec §7.2).

    Parameters
    ----------
    return_arr : np.ndarray, shape (E,)
    theta_low : float
        Catastrophe threshold.

    Returns
    -------
    int
    """
    arr = np.asarray(return_arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0
    return int(np.sum(arr <= theta_low))


# ---------------------------------------------------------------------------
# §7.2 metric (13): worst_window_return_percentile
# ---------------------------------------------------------------------------

def worst_window_return_percentile(
    return_arr: np.ndarray, window: int = 100, pct: float = 5.0
) -> float:
    """Worst-case rolling-window mean return at percentile ``pct``
    (spec §7.2).

    Computes the mean return over each contiguous window of length
    ``window``, then returns the ``pct``-th percentile of those
    window-means. Provides a single-number summary of "how bad does the
    worst stretch get" useful for safety / catastrophe analysis.

    For ``len(return_arr) < window``, falls back to a single window
    spanning ``[0, len(return_arr))``. Empty input returns ``0.0``.

    Parameters
    ----------
    return_arr : np.ndarray, shape (E,)
    window : int, default 100
    pct : float, default 5.0
        Percentile in ``[0, 100]``.

    Returns
    -------
    float
    """
    arr = np.asarray(return_arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    w = int(window)
    if w <= 0:
        raise ValueError(f"window must be positive; got {w}.")
    if arr.size < w:
        # Fall back: single window spanning the full array.
        return float(np.mean(arr))
    # Rolling-mean via cumulative sum: O(E) and exact for window means.
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    window_means = (cs[w:] - cs[:-w]) / float(w)
    return float(np.percentile(window_means, pct))


# ---------------------------------------------------------------------------
# §7.2 metric (14): trap_entries
# ---------------------------------------------------------------------------

def trap_entries(
    state_traj: np.ndarray,
    trap_states: Union[Sequence[int], np.ndarray],
) -> int:
    """Count transitions *into* a trap state (spec §7.2).

    A "trap entry" at index ``t`` is registered iff
    ``state_traj[t] in trap_states`` AND
    (``t == 0`` OR ``state_traj[t-1] not in trap_states``). This counts
    the number of distinct visits to the trap region rather than the
    total time spent there.

    Parameters
    ----------
    state_traj : np.ndarray, shape (T,)
        Integer state trajectory.
    trap_states : sequence or np.ndarray of int
        Trap-state indices.

    Returns
    -------
    int
    """
    states = np.asarray(state_traj).reshape(-1)
    if states.size == 0:
        return 0
    trap_arr = np.asarray(list(trap_states) if not isinstance(trap_states, np.ndarray) else trap_states)
    if trap_arr.size == 0:
        return 0
    in_trap = np.isin(states, trap_arr)
    # An entry occurs at t when in_trap[t] is True and either t == 0
    # or in_trap[t-1] is False. Vectorised: prepend False.
    prev_in_trap = np.concatenate([[False], in_trap[:-1]])
    entries = in_trap & ~prev_in_trap
    return int(np.sum(entries))


# ---------------------------------------------------------------------------
# §7.2 metric (15): constraint_violations
# ---------------------------------------------------------------------------

def constraint_violations(constraint_log: np.ndarray) -> int:
    """Count nonzero entries in a per-episode constraint log (spec §7.2).

    The log is failure-mode-specific (e.g. clipping flags, safety-shield
    triggers); this metric just counts how many entries are nonzero.

    Parameters
    ----------
    constraint_log : np.ndarray, shape (E,)

    Returns
    -------
    int
    """
    arr = np.asarray(constraint_log).reshape(-1)
    if arr.size == 0:
        return 0
    return int(np.count_nonzero(arr))


# ---------------------------------------------------------------------------
# §7.2 metric (16): overflow_count
# ---------------------------------------------------------------------------

def overflow_count(
    q_abs_max_arr: np.ndarray, threshold: float = 1e6
) -> int:
    """Count episodes where ``|Q|_max >= threshold`` (spec §7.2).

    A coarse divergence indicator: any episode whose largest absolute
    Q-value reached or exceeded ``threshold`` is counted.

    Parameters
    ----------
    q_abs_max_arr : np.ndarray, shape (E,)
        Per-episode ``max |Q|``.
    threshold : float, default 1e6

    Returns
    -------
    int
    """
    arr = np.asarray(q_abs_max_arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0
    return int(np.sum(arr >= threshold))
