"""Per-candidate exact-planning metrics for Phase V search (WP1a).

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 6
("Required per-candidate metrics") and planner-resolution addendum
section 13.

Decision provenance (WP1a brief, 2026-04-23)
--------------------------------------------
* C1. ``V_next_ref = V_safe`` -- margin_pos is evaluated under the safe
  DP fixed point because ``beta_tilde_t`` is the safe temperature.
* C2. Pointwise form: ``margin_pos`` is averaged per transition
  ``(s, a, s')`` with reward ``R(s, a, s')`` and
  ``V_next_ref = V_safe[t+1, s']``, under
  ``d_ref(t, s) * pi*_safe(a | t, s) * P(s' | s, a)``.
* C3. Greedy ``pi*_safe`` policy under the safe Bellman target.
* D1. ``clip_fraction`` is per-transition: a transition ``(s, a, s', t)``
  is clipped iff the stagewise schedule entry satisfies
  ``|beta_raw_t| > beta_cap_t``.  (We consume the already-clipped
  schedule arrays rather than re-deriving ``beta_raw``; D1 is in
  practice a per-stage flag broadcast to transitions under d_ref.)
* D2. Three clipping metrics: fraction, inactive-complement,
  saturation.  ``saturation`` tracks ``|beta_used_t| == beta_cap_t``.
* I1. ``raw_convergence_status`` is always ``"not_evaluated"`` in WP1a;
  WP4 (safe-vs-raw) overwrites this sentinel.

Schedule arrays expected (dict, as produced by
``phase4_calibration_v3.build_schedule_v3``)
    * ``beta_used_t`` : (T,)  clipped beta applied in the safe operator
    * ``beta_cap_t``  : (T,)  positive stagewise clip caps
    * ``beta_raw_t``  : (T,)  optional; falls back to ``beta_used_t``
    * ``kappa_t``     : (T,)  optional certified contraction levels
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..calibration.calibration_utils import clip_beta as _clip_beta_certutil
from .family_spec import ContestState
from .reference_occupancy import absorbing_mask

__all__ = ["evaluate_candidate"]


# ---------------------------------------------------------------------------
# Exact classical and safe DP (self-contained)
# ---------------------------------------------------------------------------

def _extract_tensors(
    mdp: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, int]:
    """Return ``(P, R, r_bar, gamma, T, S, A)``."""
    P = np.asarray(mdp.p, dtype=np.float64)   # (S, A, S')
    R = np.asarray(mdp.r, dtype=np.float64)   # (S, A, S')
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)
    S, A = P.shape[0], P.shape[1]
    r_bar = np.einsum("ijk,ijk->ij", P, R)
    return P, R, r_bar, gamma, T, S, A


def _classical_qv(
    P: np.ndarray,
    r_bar: np.ndarray,
    gamma: float,
    T: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact classical finite-horizon DP: returns Q[T,S,A], V[T+1,S]."""
    S = P.shape[0]
    A = P.shape[1]
    V = np.zeros((T + 1, S), dtype=np.float64)
    Q = np.zeros((T, S, A), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        E_v = np.einsum("ijk,k->ij", P, V[t + 1])
        Q[t] = r_bar + gamma * E_v
        V[t] = np.max(Q[t], axis=1)
    return Q, V


def _safe_q_stage(
    beta: float,
    gamma: float,
    R: np.ndarray,            # (S, A, S')
    P: np.ndarray,            # (S, A, S')
    V_next: np.ndarray,       # (S,)
) -> np.ndarray:
    """Safe stage-Q[s, a] using the closed-form g_t^safe under logaddexp.

    ``g_t^safe(r, v) = (1+gamma)/beta * [log(exp(beta*r) + gamma*exp(beta*v))
                                          - log(1+gamma)]``

    The beta = 0 branch collapses to classical ``r + gamma * v``.
    """
    S, A, _ = R.shape
    if abs(beta) < 1e-12:
        # Classical.  E_{s'}[R + gamma V] under P.
        target = R + gamma * V_next[None, None, :]     # (S, A, S')
        return np.einsum("ijk,ijk->ij", P, target)     # (S, A)
    log_gamma = float(np.log(gamma))
    log_1pg = float(np.log(1.0 + gamma))
    # log( exp(beta*r) + gamma*exp(beta*v) )
    #   = logaddexp(beta*r, beta*v + log_gamma)
    beta_r = beta * R                                 # (S, A, S')
    beta_v = beta * V_next[None, None, :]             # (1, 1, S')
    log_sum = np.logaddexp(beta_r, beta_v + log_gamma)  # (S, A, S')
    g = (1.0 + gamma) / beta * (log_sum - log_1pg)     # (S, A, S')
    return np.einsum("ijk,ijk->ij", P, g)              # (S, A)


def _safe_qv(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    T: int,
    beta_used_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact safe finite-horizon DP under the deployed clipped schedule."""
    S, A = P.shape[0], P.shape[1]
    V = np.zeros((T + 1, S), dtype=np.float64)
    Q = np.zeros((T, S, A), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        Q[t] = _safe_q_stage(float(beta_used_t[t]), gamma, R, P, V[t + 1])
        V[t] = np.max(Q[t], axis=1)
    return Q, V


# ---------------------------------------------------------------------------
# Occupancy (self-contained to avoid circular imports from d_ref module)
# ---------------------------------------------------------------------------

def _greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy: argmax_a Q[t, s, a]; shape (T, S)."""
    return np.argmax(Q, axis=2).astype(np.int64)


def _deterministic_to_stochastic(
    pi: np.ndarray,
    A: int,
) -> np.ndarray:
    """Expand deterministic ``(T, S)`` -> stochastic ``(T, S, A)``."""
    T, S = pi.shape
    out = np.zeros((T, S, A), dtype=np.float64)
    t_idx = np.arange(T)[:, None]
    s_idx = np.arange(S)[None, :]
    out[t_idx, s_idx, pi] = 1.0
    return out


def _forward_occupancy(
    P: np.ndarray,
    pi_stoch: np.ndarray,    # (T, S, A)
    mu_0: np.ndarray,
    T: int,
) -> np.ndarray:
    S = P.shape[0]
    d = np.zeros((T, S), dtype=np.float64)
    d[0] = mu_0
    for t in range(T - 1):
        joint = d[t, :, None] * pi_stoch[t]
        d[t + 1] = np.einsum("ij,ijk->k", joint, P)
    return d


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _pull_schedule(
    schedule: dict[str, Any],
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(beta_used_t, beta_cap_t, beta_raw_t)`` as (T,) arrays."""
    beta_used_t = np.asarray(schedule.get("beta_used_t"), dtype=np.float64)
    beta_cap_t = np.asarray(schedule.get("beta_cap_t"), dtype=np.float64)
    beta_raw_t = schedule.get("beta_raw_t", None)
    if beta_used_t.shape != (T,):
        raise ValueError(
            f"schedule['beta_used_t'] must have shape ({T},); "
            f"got {beta_used_t.shape}."
        )
    if beta_cap_t.shape != (T,):
        raise ValueError(
            f"schedule['beta_cap_t'] must have shape ({T},); "
            f"got {beta_cap_t.shape}."
        )
    if beta_raw_t is None:
        beta_raw_arr = beta_used_t.copy()
    else:
        beta_raw_arr = np.asarray(beta_raw_t, dtype=np.float64)
        if beta_raw_arr.shape != (T,):
            raise ValueError(
                f"schedule['beta_raw_t'] must have shape ({T},); "
                f"got {beta_raw_arr.shape}."
            )
    return beta_used_t, beta_cap_t, beta_raw_arr


def evaluate_candidate(
    mdp: Any,
    schedule: dict[str, Any],
    *,
    contest_state: ContestState,
    reward_scale: float | None = None,
    mu_0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute every §6 metric for a single ``(lam, psi, schedule)`` candidate.

    Parameters
    ----------
    mdp : MushroomRL ``FiniteMDP`` built by ``family.build_mdp(lam, psi)``.
    schedule : dict
        Deployed clipped schedule (``beta_used_t``, ``beta_cap_t``,
        optional ``beta_raw_t``).  WP1c owns schedule construction; WP1a
        consumes it.
    contest_state : ContestState
        Designated near-tie contest state for this family.
    reward_scale : float | None
        Optional override; defaults to
        ``max(R_max, |V*_cl(0, s_0)|, 1e-8)`` per spec §3.
    mu_0 : ndarray | None
        Initial distribution; defaults to a point mass at
        ``mdp.initial_state`` when the attribute exists.

    Returns
    -------
    dict with all §6 keys plus ``raw_convergence_status='not_evaluated'``.
    """
    P, R, r_bar, gamma, T, S, A = _extract_tensors(mdp)
    beta_used_t, beta_cap_t, beta_raw_t = _pull_schedule(schedule, T)

    # Sanity check: verify the supplied beta_used is clip(beta_raw, -cap, cap).
    # (Non-fatal: WP1a does not overwrite the caller's schedule; we only log
    # the saturation bit from the caller's arrays.)
    _expected_used, _ = _clip_beta_certutil(beta_raw_t, beta_cap_t)
    # We intentionally do not raise if they disagree; some callers (raw
    # ablations in WP4) deliberately pass beta_used != clip(beta_raw, cap).

    # Initial distribution.
    if mu_0 is None:
        init_s = getattr(mdp, "initial_state", None)
        if init_s is not None:
            s0 = int(init_s)
        elif getattr(mdp, "mu", None) is not None:
            s0 = int(np.argmax(np.asarray(mdp.mu)))
        else:
            s0 = 0
        mu_0 = np.zeros(S, dtype=np.float64)
        mu_0[s0] = 1.0
    else:
        mu_0 = np.asarray(mu_0, dtype=np.float64).reshape(-1)
        if mu_0.shape != (S,):
            raise ValueError(f"mu_0 shape must be ({S},); got {mu_0.shape}.")
        s0 = int(np.argmax(mu_0))

    # Exact DP.
    Q_cl, V_cl = _classical_qv(P, r_bar, gamma, T)
    Q_safe, V_safe = _safe_qv(P, R, gamma, T, beta_used_t)

    return _evaluate_core(
        P=P, R=R, gamma=gamma, T=T, S=S, A=A,
        Q_cl=Q_cl, V_cl=V_cl, Q_safe=Q_safe, V_safe=V_safe,
        beta_used_t=beta_used_t,
        beta_cap_t=beta_cap_t,
        beta_raw_t=beta_raw_t,
        mu_0=mu_0, s0=s0,
        contest_state=contest_state,
        reward_scale=reward_scale,
    )


def _evaluate_core(
    *,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    T: int,
    S: int,
    A: int,
    Q_cl: np.ndarray,
    V_cl: np.ndarray,
    Q_safe: np.ndarray,
    V_safe: np.ndarray,
    beta_used_t: np.ndarray,
    beta_cap_t: np.ndarray,
    beta_raw_t: np.ndarray,
    mu_0: np.ndarray,
    s0: int,
    contest_state: ContestState,
    reward_scale: float | None,
) -> dict[str, Any]:
    """Core metric computation given pre-computed Q/V arrays.

    Exposed (as ``_evaluate_core``) for tests that need to inject Q arrays
    bypassing the DP path; production callers use :func:`evaluate_candidate`.
    """
    # Greedy policies (deterministic, broken by numpy argmax tie-rule).
    pi_cl = _greedy_policy(Q_cl)           # (T, S) int
    pi_safe = _greedy_policy(Q_safe)        # (T, S) int
    pi_cl_stoch = _deterministic_to_stochastic(pi_cl, A)
    pi_safe_stoch = _deterministic_to_stochastic(pi_safe, A)

    # Occupancy.
    d_cl = _forward_occupancy(P, pi_cl_stoch, mu_0, T)
    d_safe = _forward_occupancy(P, pi_safe_stoch, mu_0, T)
    d_ref = 0.5 * d_cl + 0.5 * d_safe

    # ---- Reward scale ------------------------------------------------
    R_max = float(np.max(np.abs(R))) if R.size else 0.0
    v_cl_0 = float(abs(V_cl[0, s0]))
    if reward_scale is None:
        reward_scale = float(max(R_max, v_cl_0, 1e-8))
    else:
        reward_scale = float(reward_scale)
    if reward_scale <= 0.0:
        reward_scale = 1e-8

    # ---- §6 metrics --------------------------------------------------

    # ------ policy_disagreement under d_ref (t, s) marginal ------
    # d_ref is already a (T, S) distribution over (t, s) pairs — each
    # stage sums to 1, so the (t, s) marginal is d_ref / T.  Per spec the
    # metric is a probability under d_ref treated as a state-occupancy;
    # following the Phase IV convention we normalize by T (the sum of
    # per-stage masses) so disagreement is reported on [0, 1].
    # At absorbing states reached under pi*_safe the operator has no
    # future mass to discount, so disagreement there is operationally
    # meaningless. Mask those cells out (spec §13 addendum + WP0
    # remediation 2026-04-24).
    absorb_sa = absorbing_mask(P)                              # (S, A) bool
    pi_safe_idx = pi_safe                                       # (T, S) int
    t_idx = np.arange(T)[:, None]
    s_idx = np.arange(S)[None, :]
    absorb_under_pi_safe = absorb_sa[s_idx, pi_safe_idx]        # (T, S) bool
    live_mask = (~absorb_under_pi_safe).astype(np.float64)      # (T, S)
    disagree_mask = (pi_safe != pi_cl).astype(np.float64)       # (T, S)
    d_ref_live = d_ref * live_mask
    d_ref_live_total = float(d_ref_live.sum())
    if d_ref_live_total > 0.0:
        policy_disagreement = float(
            (d_ref_live * disagree_mask).sum() / d_ref_live_total
        )
    else:
        policy_disagreement = 0.0
    # Retain the raw d_ref mass for backward-compatible metrics below
    # (margin_pos, delta_d, clip_fraction already mask absorbing cells
    # naturally via the P-weighted `weights` tensor).
    d_ref_mass_total = float(d_ref.sum())

    start_state_flip = int(pi_safe[0, s0] != pi_cl[0, s0])

    # ------ value_gap at start state ------
    value_gap = float(V_safe[0, s0] - V_cl[0, s0])
    value_gap_norm = float(value_gap / reward_scale)

    # ------ contest gap ------
    t_c = int(contest_state.t)
    s_c = int(contest_state.s)
    ap = contest_state.action_pair
    if ap is None:
        # Default: top-2 classical-Q actions at (t_c, s_c).
        order = np.argsort(Q_cl[t_c, s_c])[::-1]
        a1, a2 = int(order[0]), int(order[1])
    else:
        a1, a2 = int(ap[0]), int(ap[1])
    contest_gap = float(Q_cl[t_c, s_c, a1] - Q_cl[t_c, s_c, a2])
    contest_gap_abs = float(abs(contest_gap))
    contest_gap_norm = float(contest_gap_abs / reward_scale)
    # Per-stage marginal probability: the forward sweep normalizes each
    # stage's d_ref row to 1 (sum over s).  d_ref[t_c, s_c] is already the
    # "prob the agent occupies s_c at stage t_c under d_ref".  Reported as
    # the raw stage-marginal probability per spec §3.
    contest_occupancy_ref = float(d_ref[t_c, s_c])

    # ------ margin_pos (pointwise under d_ref * pi*_safe * P) ------
    # Per-transition contribution: beta_tilde_t * (r[s,a,s'] - V_safe[t+1,s'])
    # max(..., 0) ; averaged under p(t, s, a, s') = d_ref[t,s] * pi*_safe[t,s,a] * P[s,a,s']
    beta_3d = beta_used_t[:, None, None, None]                       # (T, 1, 1, 1)
    V_next_safe = V_safe[1:T + 1, :]                                 # (T, S')
    r_minus_v = R[None, :, :, :] - V_next_safe[:, None, None, :]     # (T, S, A, S')
    pointwise = np.maximum(beta_3d * r_minus_v, 0.0)                 # (T, S, A, S')
    weights = (
        d_ref[:, :, None, None] * pi_safe_stoch[:, :, :, None] * P[None, :, :, :]
    )  # (T, S, A, S')
    margin_pos = float((weights * pointwise).sum() / max(d_ref_mass_total, 1e-30))
    margin_pos_norm = float(margin_pos / reward_scale)

    # ------ delta_d / mass_delta_d ------
    # d_t^safe(s, a, s') = (1 + gamma) * (1 - rho_t(r, v))
    # rho_t = sigmoid(beta_tilde * (r - v) + log(1/gamma))
    # For beta=0 rho -> 1/(1+gamma) and d_t -> gamma exactly.
    safe_local_deriv = _compute_safe_local_deriv(R, V_next_safe, beta_used_t, gamma)
    # (T, S, A, S')
    abs_dev = np.abs(safe_local_deriv - gamma)                       # (T, S, A, S')
    delta_d = float((weights * abs_dev).sum() / max(d_ref_mass_total, 1e-30))
    mask_big = (abs_dev > 1e-3).astype(np.float64)
    mass_delta_d = float((weights * mask_big).sum() / max(d_ref_mass_total, 1e-30))

    # ------ clip metrics (per-transition, broadcast from per-stage flags) ------
    clip_bit_stage = (np.abs(beta_raw_t) > beta_cap_t + 1e-15).astype(np.float64)  # (T,)
    saturation_bit_stage = (
        np.abs(np.abs(beta_used_t) - beta_cap_t) <= 1e-12
    ).astype(np.float64)  # (T,)
    # Only count saturation when beta_cap is positive — beta_cap=0 with
    # beta_used=0 is the classical collapse, not a saturated clip.
    saturation_bit_stage = saturation_bit_stage * (beta_cap_t > 0.0).astype(np.float64)

    # Per-transition weight sum under d_ref * pi*_safe * P, grouped by stage.
    stage_mass = weights.sum(axis=(1, 2, 3))   # (T,)
    total_mass = float(stage_mass.sum())
    if total_mass > 0.0:
        clip_fraction = float(
            (stage_mass * clip_bit_stage).sum() / total_mass
        )
        clip_saturation_fraction = float(
            (stage_mass * saturation_bit_stage).sum() / total_mass
        )
    else:
        clip_fraction = 0.0
        clip_saturation_fraction = 0.0
    clip_inactive_fraction = float(1.0 - clip_fraction)

    # ------ raw local derivative stats (evaluated at beta_raw_t) ------
    # d_raw_t(r, v) = (1 + gamma) * (1 - sigmoid(beta_raw * (r - v) + log(1/gamma)))
    raw_deriv = _compute_safe_local_deriv(R, V_next_safe, beta_raw_t, gamma)
    flat_vals = raw_deriv.reshape(-1)
    flat_w = weights.reshape(-1)
    # Weighted mean, weighted quantiles (under d_ref weights).
    total_w = float(flat_w.sum())
    if total_w > 0.0:
        mean_val = float(np.sum(flat_vals * flat_w) / total_w)
        p50 = float(_weighted_quantile(flat_vals, flat_w, 0.50))
        p90 = float(_weighted_quantile(flat_vals, flat_w, 0.90))
        max_val = float(np.max(flat_vals[flat_w > 0.0])) if np.any(flat_w > 0.0) else 0.0
    else:
        mean_val = 0.0
        p50 = 0.0
        p90 = 0.0
        max_val = 0.0
    raw_local_deriv_stats = {
        "mean": mean_val,
        "p50": p50,
        "p90": p90,
        "max": max_val,
    }

    return {
        "margin_pos": margin_pos,
        "margin_pos_norm": margin_pos_norm,
        "delta_d": delta_d,
        "mass_delta_d": mass_delta_d,
        "policy_disagreement": policy_disagreement,
        "start_state_flip": start_state_flip,
        "value_gap": value_gap,
        "value_gap_norm": value_gap_norm,
        "contest_gap_abs": contest_gap_abs,
        "contest_gap_norm": contest_gap_norm,
        "contest_occupancy_ref": contest_occupancy_ref,
        "clip_fraction": clip_fraction,
        "clip_inactive_fraction": clip_inactive_fraction,
        "clip_saturation_fraction": clip_saturation_fraction,
        "raw_local_deriv_stats": raw_local_deriv_stats,
        "raw_convergence_status": "not_evaluated",
    }


# ---------------------------------------------------------------------------
# Helpers: safe local derivative and weighted quantile
# ---------------------------------------------------------------------------

def _compute_safe_local_deriv(
    R: np.ndarray,           # (S, A, S')
    V_next: np.ndarray,      # (T, S')
    beta_t: np.ndarray,      # (T,)
    gamma: float,
) -> np.ndarray:
    """Local continuation derivative ``d_t(r, v) = (1+gamma)(1-rho_t(r,v))``.

    rho_t = sigmoid(beta_t * (r - v) + log(1/gamma))
          = 1 / (1 + gamma * exp(-beta_t * (r - v)))

    For ``beta_t = 0`` this collapses to ``gamma`` exactly.  We use
    ``1 / (1 + gamma * exp(-x))`` and short-circuit the ``beta = 0`` stages
    to avoid any ``exp`` on the classical branch.
    """
    T = beta_t.shape[0]
    S, A, _ = R.shape
    r_expanded = R[None, :, :, :]                         # (1, S, A, S')
    v_expanded = V_next[:, None, None, :]                 # (T, 1, 1, S')
    diff = r_expanded - v_expanded                        # (T, S, A, S')
    out = np.empty((T, S, A, R.shape[2]), dtype=np.float64)
    for t in range(T):
        b = float(beta_t[t])
        if abs(b) < 1e-12:
            out[t] = gamma  # broadcast scalar
            continue
        # rho_t = 1 / (1 + gamma * exp(-b * diff))
        # Use a log-domain expression to avoid overflow when b*diff large.
        x = b * diff[t]                                   # (S, A, S')
        # gamma * exp(-x) = exp(log gamma - x); rho = 1/(1 + exp(log_gamma - x)).
        log_gamma = float(np.log(gamma))
        rho = 1.0 / (1.0 + np.exp(log_gamma - x))
        out[t] = (1.0 + gamma) * (1.0 - rho)
    return out


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """Weighted quantile; treats weights as probability mass."""
    mask = weights > 0.0
    if not np.any(mask):
        return 0.0
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted) / w_sorted.sum()
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(idx, v_sorted.size - 1)
    return float(v_sorted[idx])
