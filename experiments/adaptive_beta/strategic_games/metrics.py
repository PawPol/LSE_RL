"""Strategic-learning metrics for Phase VII-B.

Spec authority:
- ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md`` §9, §10, §13.
- Planner item VII-B-27 (minimum metric set).

This module is **pure NumPy**. No pandas, no MushroomRL, no internal RNG. All
public functions are deterministic given their inputs and handle empty arrays
gracefully (returning ``np.nan`` rather than raising).

Public API
----------
- :func:`rolling_policy_entropy`
- :func:`policy_total_variation`
- :func:`support_shift_count`
- :func:`external_regret`
- :func:`cycling_amplitude`
- :func:`event_aligned_window`
- :func:`search_phase_vs_stable_phase_return`
- :func:`miscoordination_rate`
- :func:`coordination_rate`
- :func:`empirical_best_response_value`

Notes on conventions
--------------------
- All entropies are in **nats** (natural log), consistent with
  ``StrategicAdversary._entropy``.
- Action arrays are 1-D ``np.ndarray`` of integer indices in
  ``[0, n_actions)``. Out-of-range entries are silently dropped from
  histogram counts (``np.bincount`` with ``minlength=n_actions``).
- ``event_aligned_window`` pads with ``np.nan`` (NOT zero) so that
  downstream plotting can mask correctly. Spec §10 plots are required to
  ignore boundary samples; zero-padding would silently bias the mean.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "rolling_policy_entropy",
    "policy_total_variation",
    "support_shift_count",
    "external_regret",
    "cycling_amplitude",
    "event_aligned_window",
    "search_phase_vs_stable_phase_return",
    "miscoordination_rate",
    "coordination_rate",
    "empirical_best_response_value",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empirical_distribution(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """Empirical action distribution.

    Parameters
    ----------
    actions
        Integer action indices, shape ``(T,)``.
    n_actions
        Action-space cardinality.

    Returns
    -------
    probs : np.ndarray, shape ``(n_actions,)``
        Normalised frequency vector. Entries outside ``[0, n_actions)`` are
        dropped from the count. If ``actions`` is empty or all entries are
        out-of-range, returns the uniform distribution over ``n_actions``.
    """
    if n_actions <= 0:
        return np.zeros(0, dtype=np.float64)
    a = np.asarray(actions).reshape(-1)
    if a.size == 0:
        return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
    a = a.astype(np.int64, copy=False)
    mask = (a >= 0) & (a < n_actions)
    if not np.any(mask):
        return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
    counts = np.bincount(a[mask], minlength=n_actions).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
    return counts / total  # shape (n_actions,)


def _shannon_entropy_nats(probs: np.ndarray) -> float:
    """Shannon entropy in nats; robust to zero entries."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)  # shape (k,)
    if p.size == 0:
        return float("nan")
    nz = p > 0.0
    if not np.any(nz):
        return 0.0
    return float(-np.sum(p[nz] * np.log(p[nz])))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rolling_policy_entropy(
    actions: np.ndarray,
    n_actions: int,
    window: int = 100,
) -> float:
    """Shannon entropy (nats) of the empirical action distribution over
    the trailing ``window`` of ``actions``.

    Parameters
    ----------
    actions : np.ndarray, shape ``(T,)``
        Integer action indices in chronological order.
    n_actions
        Action-space cardinality.
    window
        Trailing window length. ``window <= 0`` is treated as the full
        history.

    Returns
    -------
    entropy : float
        ``np.nan`` for empty input or non-positive ``n_actions``.
    """
    a = np.asarray(actions).reshape(-1)
    if a.size == 0 or n_actions <= 0:
        return float("nan")
    if window is None or window <= 0 or window >= a.size:
        window_slice = a
    else:
        window_slice = a[-int(window):]  # shape (<=window,)
    probs = _empirical_distribution(window_slice, n_actions)  # (n_actions,)
    return _shannon_entropy_nats(probs)


def policy_total_variation(
    actions_window_a: np.ndarray,
    actions_window_b: np.ndarray,
    n_actions: int,
) -> float:
    """Total-variation distance between two empirical action distributions.

    ``TV(p, q) = 0.5 * sum_i |p_i - q_i|`` in ``[0, 1]``.

    Parameters
    ----------
    actions_window_a, actions_window_b : np.ndarray, shape ``(T_a,)``, ``(T_b,)``
        Action-index arrays for the two windows.
    n_actions
        Action-space cardinality (shared).

    Returns
    -------
    tv : float
        ``np.nan`` if both windows are empty or ``n_actions <= 0``.
    """
    if n_actions <= 0:
        return float("nan")
    a = np.asarray(actions_window_a).reshape(-1)
    b = np.asarray(actions_window_b).reshape(-1)
    if a.size == 0 and b.size == 0:
        return float("nan")
    p = _empirical_distribution(a, n_actions)  # (n_actions,)
    q = _empirical_distribution(b, n_actions)  # (n_actions,)
    return float(0.5 * np.sum(np.abs(p - q)))


def support_shift_count(
    opponent_policy_trace: np.ndarray,
    threshold_tv: float = 0.1,
) -> int:
    """Number of step-to-step support shifts in an opponent-policy trace.

    A "shift" is a step where the L1 distance between successive policy
    rows exceeds ``2 * threshold_tv`` (equivalently, TV > ``threshold_tv``).

    Parameters
    ----------
    opponent_policy_trace : np.ndarray, shape ``(T, n_actions)``
        Per-step policy distributions over ``T`` time steps.
    threshold_tv
        TV-distance threshold. Defaults to ``0.1`` (spec default).

    Returns
    -------
    count : int
        Number of shift events. ``0`` for empty traces or single rows.
    """
    P = np.asarray(opponent_policy_trace, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] < 2:
        return 0
    # Successive-row L1 norms, shape (T-1,).
    diffs = np.abs(P[1:] - P[:-1]).sum(axis=1)
    tv = 0.5 * diffs  # shape (T-1,)
    return int(np.sum(tv > float(threshold_tv)))


def external_regret(
    agent_actions: np.ndarray,
    payoffs: np.ndarray,
    opponent_actions: np.ndarray,
) -> float:
    """External regret vs the best fixed pure agent action in hindsight.

    Definition (per spec §9.3):

    ``regret = (1/T) * (max_a sum_t payoffs[a, opp_t]) - (1/T) * sum_t payoffs[agent_t, opp_t]``

    Parameters
    ----------
    agent_actions : np.ndarray, shape ``(T,)``
        Agent actions per step.
    payoffs : np.ndarray, shape ``(n_a, n_o)``
        Agent payoff matrix; rows index agent action, columns index
        opponent action.
    opponent_actions : np.ndarray, shape ``(T,)``
        Opponent actions per step (must align with ``agent_actions``).

    Returns
    -------
    regret : float
        ``np.nan`` if any input is empty / mismatched, or payoffs is not 2-D.
    """
    a = np.asarray(agent_actions).reshape(-1).astype(np.int64, copy=False)
    o = np.asarray(opponent_actions).reshape(-1).astype(np.int64, copy=False)
    M = np.asarray(payoffs, dtype=np.float64)
    if a.size == 0 or o.size == 0 or M.ndim != 2:
        return float("nan")
    if a.size != o.size:
        return float("nan")
    n_a, n_o = M.shape
    # Defensive bounds — drop out-of-range rows symmetrically.
    valid = (a >= 0) & (a < n_a) & (o >= 0) & (o < n_o)
    if not np.any(valid):
        return float("nan")
    a_v = a[valid]
    o_v = o[valid]
    T = a_v.size
    # Sum of payoffs for each fixed agent action against the realised
    # opponent sequence: shape (n_a,) via column-aggregation.
    # opp_counts[j] = number of times opponent played j, shape (n_o,).
    opp_counts = np.bincount(o_v, minlength=n_o).astype(np.float64)
    # Per-row hindsight value: M @ opp_counts gives shape (n_a,).
    hindsight = M @ opp_counts  # shape (n_a,)
    best_fixed = float(np.max(hindsight)) / float(T)
    realised = float(np.sum(M[a_v, o_v])) / float(T)
    return best_fixed - realised


def cycling_amplitude(
    agent_actions: np.ndarray,
    opponent_actions: np.ndarray,
    window: int = 100,
) -> float:
    """Max - min of windowed empirical action frequencies, summed over both
    players.

    For each window-rolling step, compute the empirical action distributions
    of agent and opponent over the trailing ``window``. The "amplitude" is
    the peak-to-peak spread (max minus min over time) of any action-frequency
    coordinate, summed across players. This captures the magnitude of
    cycling behaviour in repeated games (Shapley, RPS).

    Parameters
    ----------
    agent_actions, opponent_actions : np.ndarray, shape ``(T,)``
        Per-step action sequences. Must be the same length.
    window
        Rolling window length.

    Returns
    -------
    amplitude : float
        ``np.nan`` if either array is empty, lengths differ, or ``T < window``.
    """
    a = np.asarray(agent_actions).reshape(-1).astype(np.int64, copy=False)
    o = np.asarray(opponent_actions).reshape(-1).astype(np.int64, copy=False)
    if a.size == 0 or o.size == 0 or a.size != o.size:
        return float("nan")
    T = a.size
    w = int(window)
    if w <= 0 or w > T:
        return float("nan")
    # Infer action-space cardinality.
    n_a = int(max(a.max(), 0)) + 1
    n_o = int(max(o.max(), 0)) + 1
    n_steps = T - w + 1  # number of windowed snapshots
    # freqs: shape (n_steps, n_a + n_o), per-step concatenated frequency vector.
    freqs = np.zeros((n_steps, n_a + n_o), dtype=np.float64)
    for i in range(n_steps):
        a_win = a[i : i + w]  # shape (w,)
        o_win = o[i : i + w]  # shape (w,)
        freqs[i, :n_a] = np.bincount(a_win, minlength=n_a).astype(np.float64) / w
        freqs[i, n_a:] = np.bincount(o_win, minlength=n_o).astype(np.float64) / w
    # Peak-to-peak spread per coordinate, summed across coordinates.
    spread = freqs.max(axis=0) - freqs.min(axis=0)  # shape (n_a + n_o,)
    return float(spread.sum())


def event_aligned_window(
    values: np.ndarray,
    event_indices: np.ndarray,
    half_window: int = 50,
) -> np.ndarray:
    """Stack windows of length ``2*half_window+1`` centered on each event.

    Parameters
    ----------
    values : np.ndarray, shape ``(T,)``
        Per-step scalar metric (e.g. return, beta).
    event_indices : np.ndarray, shape ``(K,)``
        Integer time-step indices of events; values outside ``[0, T)`` are
        kept (their windows pad fully with ``np.nan``).
    half_window
        Half-window radius. The output column count is ``2*half_window+1``;
        the column at index ``half_window`` is the event step itself.

    Returns
    -------
    panel : np.ndarray, shape ``(K, 2*half_window+1)``
        Each row is the windowed slice for one event. Boundary samples are
        padded with ``np.nan`` (per spec §10 / OQ-3 binding).

    Notes
    -----
    Returns shape ``(0, 2*half_window+1)`` if no events. NaN-padding (NOT
    zero) is mandatory: zero-padding would silently bias downstream
    averages.
    """
    h = int(half_window)
    if h < 0:
        raise ValueError(f"half_window must be >= 0, got {half_window}")
    width = 2 * h + 1
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    ev = np.asarray(event_indices).reshape(-1).astype(np.int64, copy=False)
    K = ev.size
    panel = np.full((K, width), np.nan, dtype=np.float64)
    if K == 0 or v.size == 0:
        return panel
    T = v.size
    for k in range(K):
        e = int(ev[k])
        # Source slice within values: [e - h, e + h] inclusive, clipped.
        src_lo = max(0, e - h)
        src_hi = min(T, e + h + 1)  # exclusive
        if src_hi <= src_lo:
            continue  # event entirely outside [0, T) — leave NaN row.
        # Destination indices in the panel row.
        dst_lo = src_lo - (e - h)
        dst_hi = dst_lo + (src_hi - src_lo)
        panel[k, dst_lo:dst_hi] = v[src_lo:src_hi]
    return panel


def search_phase_vs_stable_phase_return(
    returns: np.ndarray,
    search_phase_flags: np.ndarray,
) -> Tuple[float, float]:
    """Mean episode-return inside vs outside hypothesis-testing search phase.

    Parameters
    ----------
    returns : np.ndarray, shape ``(T,)``
        Per-episode returns.
    search_phase_flags : np.ndarray, shape ``(T,)``
        Boolean array; ``True`` where the adversary was in its search phase.

    Returns
    -------
    (search_mean, stable_mean) : tuple of float
        Mean return during search phase, mean return during stable
        (non-search) phase. Either is ``np.nan`` if its set is empty.
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    f = np.asarray(search_phase_flags).reshape(-1).astype(bool)
    if r.size == 0 or f.size == 0:
        return (float("nan"), float("nan"))
    if r.size != f.size:
        return (float("nan"), float("nan"))
    search_mask = f
    stable_mask = ~f
    search_mean = float(r[search_mask].mean()) if np.any(search_mask) else float("nan")
    stable_mean = float(r[stable_mask].mean()) if np.any(stable_mask) else float("nan")
    return (search_mean, stable_mean)


def miscoordination_rate(
    agent_actions: np.ndarray,
    opponent_actions: np.ndarray,
) -> float:
    """Fraction of steps where ``(agent, opponent)`` are NOT on the
    coordination diagonal.

    Treats the diagonal pairs ``(i, i)`` as the coordination set; this
    matches the symmetric coordination conventions (rules-of-the-road,
    asymmetric coordination Stage B2) where action-index alignment is the
    equilibrium.

    Parameters
    ----------
    agent_actions, opponent_actions : np.ndarray, shape ``(T,)``
        Per-step action sequences. Must be the same length.

    Returns
    -------
    rate : float
        In ``[0, 1]``. ``np.nan`` for empty / mismatched inputs.
    """
    a = np.asarray(agent_actions).reshape(-1)
    o = np.asarray(opponent_actions).reshape(-1)
    if a.size == 0 or o.size == 0 or a.size != o.size:
        return float("nan")
    return float(np.mean(a != o))


def coordination_rate(
    agent_actions: np.ndarray,
    opponent_actions: np.ndarray,
) -> float:
    """Complement of :func:`miscoordination_rate`.

    Returns
    -------
    rate : float
        ``1 - miscoordination_rate``; ``np.nan`` for empty / mismatched
        inputs.
    """
    mis = miscoordination_rate(agent_actions, opponent_actions)
    if np.isnan(mis):
        return float("nan")
    return float(1.0 - mis)


def empirical_best_response_value(
    opponent_action_freqs: np.ndarray,
    payoff_agent: np.ndarray,
) -> float:
    """Value of the best response to an empirical opponent distribution.

    Computes ``max_a sum_o payoff_agent[a, o] * opp_freqs[o]``.

    Parameters
    ----------
    opponent_action_freqs : np.ndarray, shape ``(n_o,)``
        Empirical opponent distribution. Need not be normalised; the
        function will renormalise if it sums to a positive value.
    payoff_agent : np.ndarray, shape ``(n_a, n_o)``
        Agent payoff matrix; rows index agent action, columns index
        opponent action.

    Returns
    -------
    value : float
        Best-response expected payoff. ``np.nan`` if shapes are
        incompatible or the frequency vector is degenerate.
    """
    q = np.asarray(opponent_action_freqs, dtype=np.float64).reshape(-1)
    M = np.asarray(payoff_agent, dtype=np.float64)
    if M.ndim != 2 or q.size == 0:
        return float("nan")
    if q.size != M.shape[1]:
        return float("nan")
    s = q.sum()
    if not np.isfinite(s) or s <= 0.0:
        return float("nan")
    q_norm = q / s  # shape (n_o,)
    # Per-action expected payoff: shape (n_a,).
    expected = M @ q_norm
    return float(np.max(expected))
