"""Phase IV-A S5: Activation search pipeline.

Selects operator-sensitive tasks using only classical pilot diagnostics
and closed-form schedule predictions. No safe-return performance data
is used for selection -- all scoring is ex-ante.

Pipeline:
  1. Run finite-horizon DP (VI) on each candidate to get V* (spec §S5.1:
     classical pilot — QL/ESARSA or exact DP). Use V*(s') for margins.
  2. Build a calibration-v3 schedule from the pilot margin data.
  3. Score each candidate on predicted activation diagnostics.
  4. Apply acceptance criteria and per-family caps to select the suite.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from experiments.weighted_lse_dp.geometry.phase4_calibration_v3 import (
    build_schedule_v3_from_pilot,
    select_sign,
)
from experiments.weighted_lse_dp.tasks.phase4_operator_suite import (
    build_phase4_task,
    get_search_grid,
)
from experiments.weighted_lse_dp.common.seeds import seed_everything

logger = logging.getLogger(__name__)

__all__ = [
    "run_classical_pilot",
    "compute_candidate_score",
    "select_activation_suite",
    "score_all_candidates",
]


# ---------------------------------------------------------------------------
# Helper: detect whether mdp_rl is time-augmented
# ---------------------------------------------------------------------------

def _is_time_augmented(mdp_rl: Any) -> bool:
    """Check whether the RL environment is a DiscreteTimeAugmentedEnv."""
    # Avoid importing at module level to keep the dependency graph light.
    try:
        from mushroom_rl.environments.time_augmented_env import (
            DiscreteTimeAugmentedEnv,
        )
        return isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
    except ImportError:
        return hasattr(mdp_rl, "n_base_states")


def _get_n_base(mdp_base: Any) -> int:
    """Return the number of base (un-augmented) states."""
    if hasattr(mdp_base, "p"):
        return int(mdp_base.p.shape[0])
    return int(mdp_base.info.observation_space.n)


def _get_n_actions(mdp_rl: Any) -> int:
    """Return the number of actions from the MDP info."""
    return int(mdp_rl.info.action_space.n)


def _compute_vstar(
    mdp_base: Any,
    gamma: float,
    horizon: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Finite-horizon backward VI returning V_table[T+1, S] and Q_star[S, A].

    V_table[t, s] = optimal value at stage t from state s, so the correct
    continuation for a transition at stage t is V_table[t+1, s'].
    V_table[horizon] = 0 (boundary condition).
    Q_star is the greedy Q at stage 0 (used for epsilon-greedy action selection).
    """
    if not (hasattr(mdp_base, "p") and hasattr(mdp_base, "r")):
        S = int(mdp_base.info.observation_space.n)
        A = int(mdp_base.info.action_space.n)
        return np.zeros((horizon + 1, S)), np.zeros((S, A))
    P = np.asarray(mdp_base.p, dtype=np.float64)   # (S, A, S')
    R_raw = np.asarray(mdp_base.r, dtype=np.float64)
    # MushroomRL stores R as (S, A, S') — compute expected reward r̄(s,a)
    if R_raw.ndim == 3:
        R = np.einsum("ijk,ijk->ij", P, R_raw)   # (S, A)
    else:
        R = R_raw   # already (S, A)
    S, A = R.shape[:2]
    V_table = np.zeros((horizon + 1, S), dtype=np.float64)  # [T+1, S]; V_table[horizon]=0
    for t in range(horizon - 1, -1, -1):
        Q = R + gamma * np.einsum("ijk,k->ij", P, V_table[t + 1])   # (S, A)
        V_table[t] = Q.max(axis=1)
    Q_star = R + gamma * np.einsum("ijk,k->ij", P, V_table[1])
    return V_table, Q_star


# ---------------------------------------------------------------------------
# 1. Classical pilot
# ---------------------------------------------------------------------------


def run_classical_pilot(
    cfg: dict[str, Any],
    seed: int = 42,
    n_episodes: int = 50,
    max_steps: int | None = None,
    prebuilt_env: tuple[Any, Any, dict[str, Any]] | None = None,
    collect_transitions: bool = False,
    sign_family: int = 1,
) -> dict[str, Any]:
    """Run a classical pilot using exact DP V* as the value proxy.

    Uses finite-horizon backward VI (spec §S5.1: classical DP pilot) to get
    V*(s) for every base state, then rolls out an epsilon-greedy policy and
    computes margins as ``r_t - V*(s_{t+1})`` (no gamma — per lessons.md).

    Parameters
    ----------
    cfg : dict
        Task configuration (must contain ``"family"`` key).
    seed : int
        Random seed for reproducibility.
    n_episodes : int
        Number of pilot episodes to run.
    max_steps : int or None
        Maximum steps per episode.  If None, uses the MDP horizon.
    prebuilt_env : tuple or None
        Optional ``(mdp_base, mdp_rl, resolved_cfg)`` to use instead of
        rebuilding via ``build_phase4_task``. When supplied, the caller is
        responsible for having seeded the environment construction; this
        function still calls ``seed_everything(seed)`` to make trajectory
        sampling reproducible. Used by the counterfactual replay runner
        to guarantee idempotent build across pilot and replay.
    collect_transitions : bool
        If True, additionally include a ``"transitions"`` key in the
        returned dict containing the per-step ``(r, v_next, stage)``
        tuples used to compute margins. Used by the counterfactual replay
        runner so the same trajectories drive both pilot stats and the
        replay diagnostics (no second rollout needed).
    sign_family : int
        Task family sign (+1 or -1).  Used to compute sign-aligned p_align:
        ``p_align_t = P(sign_family * margin > 0)``.  Default +1 is correct
        for positive-shock families; pass -1 for catastrophe/hazard families.

    Returns
    -------
    dict
        Pilot data with keys: margins_by_stage, p_align_by_stage,
        n_by_stage, episode_returns, event_rate, n_episodes, gamma,
        horizon, reward_bound. When ``collect_transitions`` is True,
        also includes ``transitions`` (list of ``(r, v_next, stage)``).
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    if prebuilt_env is not None:
        mdp_base, mdp_rl, resolved_cfg = prebuilt_env
    else:
        mdp_base, mdp_rl, resolved_cfg = build_phase4_task(cfg, seed=seed)

    gamma = float(resolved_cfg.get("gamma", mdp_rl.info.gamma))
    horizon = int(resolved_cfg.get("horizon", mdp_rl.info.horizon))
    reward_bound = float(resolved_cfg.get("reward_bound", 1.0))

    if max_steps is None:
        max_steps = horizon

    n_actions = _get_n_actions(mdp_rl)
    time_aug = _is_time_augmented(mdp_rl)
    n_base = _get_n_base(mdp_base)

    # ------------------------------------------------------------------
    # Compute V* and Q* via finite-horizon backward VI on the base MDP.
    # This gives a spec-compliant classical pilot (§S5.1) and avoids the
    # zero-margin problem of random-policy exploration on sparse tasks.
    # ------------------------------------------------------------------
    V_star, Q_star = _compute_vstar(mdp_base, gamma, horizon)

    # ------------------------------------------------------------------
    # Collect episodes using epsilon-greedy Q* policy (eps=0.1)
    # ------------------------------------------------------------------
    all_episode_rewards: list[list[float]] = []
    all_episode_stages: list[list[int]] = []
    all_episode_next_bases: list[list[int]] = []  # base-state index of s'
    episode_returns: list[float] = []
    EPS_GREEDY = 0.1

    for ep in range(n_episodes):
        state, _ = mdp_rl.reset()
        rewards: list[float] = []
        stages: list[int] = []
        next_bases: list[int] = []

        for step_idx in range(max_steps):
            # Extract stage and base state index
            if time_aug:
                state_idx = int(np.asarray(state).flat[0])
                stage = state_idx // n_base
                base_state = state_idx % n_base
            else:
                stage = step_idx
                base_state = int(np.asarray(state).flat[0])

            stages.append(stage)

            # Epsilon-greedy action from Q*
            if len(Q_star) > base_state and rng.random() > EPS_GREEDY:
                action = np.array([int(Q_star[base_state].argmax())])
            else:
                action = np.array([rng.integers(0, n_actions)])

            next_state, reward, absorbing, info = mdp_rl.step(action)
            rewards.append(float(reward))

            # Record next base state for margin computation
            if time_aug:
                ns_idx = int(np.asarray(next_state).flat[0])
                next_bases.append(ns_idx % n_base)
            else:
                next_bases.append(int(np.asarray(next_state).flat[0]))

            state = next_state
            if absorbing:
                break

        all_episode_rewards.append(rewards)
        all_episode_stages.append(stages)
        all_episode_next_bases.append(next_bases)
        episode_returns.append(float(np.sum(rewards)))

    # -----------------------------------------------------------------
    # Compute margin = r_t - V*(s_{t+1})  (no gamma — per lessons.md)
    # -----------------------------------------------------------------
    margins_by_stage: dict[int, list[float]] = defaultdict(list)
    transitions: list[tuple[float, float, int]] = []  # (r, v_next, stage)

    for ep_rewards, ep_stages, ep_next_bases in zip(
        all_episode_rewards, all_episode_stages, all_episode_next_bases
    ):
        T_ep = len(ep_rewards)
        if T_ep == 0:
            continue

        for i in range(T_ep):
            r_t = ep_rewards[i]
            nb = ep_next_bases[i]
            stage = ep_stages[i]
            # Finite-horizon: use V_table[stage+1, s'] so each stage's
            # continuation correctly reflects remaining time-to-go.
            next_stage = stage + 1
            V_shape0 = V_star.shape[0]  # horizon+1
            if V_shape0 > next_stage and nb < V_star.shape[1]:
                v_next = float(V_star[next_stage, nb])
            else:
                v_next = 0.0
            # absorbing terminal: override with 0
            if i == T_ep - 1 and len(ep_rewards) < max_steps:
                v_next = 0.0
            margin = r_t - v_next   # no gamma (lessons.md: margin formula)
            margins_by_stage[stage].append(margin)
            if collect_transitions:
                transitions.append((float(r_t), float(v_next), int(stage)))

    # -----------------------------------------------------------------
    # Aggregate per-stage statistics
    # -----------------------------------------------------------------
    all_stages = sorted(margins_by_stage.keys())
    if len(all_stages) == 0:
        # Edge case: no data collected
        all_stages = list(range(horizon))

    margins_by_stage_list: list[NDArray[np.float64]] = []
    p_align_by_stage_list: list[float] = []
    n_by_stage_list: list[int] = []

    for t in range(max(all_stages) + 1 if all_stages else horizon):
        margins_arr = np.asarray(
            margins_by_stage.get(t, [0.0]), dtype=np.float64
        )
        margins_by_stage_list.append(margins_arr)
        n_t = len(margins_arr)
        n_by_stage_list.append(n_t)
        if n_t > 0:
            # Sign-aligned: P(sign_family * margin > 0)
            p_align_by_stage_list.append(
                float(np.mean(sign_family * margins_arr > 0.0))
            )
        else:
            p_align_by_stage_list.append(0.0)

    # -----------------------------------------------------------------
    # Event rate: fraction of episodes with extreme reward
    # -----------------------------------------------------------------
    event_threshold = 0.5 * reward_bound
    event_count = 0
    for ep_rewards in all_episode_rewards:
        if any(
            r > event_threshold or r < -event_threshold
            for r in ep_rewards
        ):
            event_count += 1
    event_rate = event_count / max(n_episodes, 1)

    out: dict[str, Any] = {
        "margins_by_stage": margins_by_stage_list,
        "p_align_by_stage": p_align_by_stage_list,
        "n_by_stage": n_by_stage_list,
        "episode_returns": episode_returns,
        "event_rate": event_rate,
        "n_episodes": n_episodes,
        "gamma": gamma,
        "horizon": horizon,
        "reward_bound": reward_bound,
        "family": resolved_cfg.get("family", cfg.get("family", "unknown")),
        "resolved_cfg": resolved_cfg,
    }
    if collect_transitions:
        out["transitions"] = transitions
    return out


# ---------------------------------------------------------------------------
# 2. Candidate scoring
# ---------------------------------------------------------------------------


def compute_candidate_score(
    pilot_data: dict[str, Any],
    schedule: dict[str, Any],
    reward_bound: float,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a candidate on predicted activation diagnostics.

    Scores are designed to rank candidates likely to satisfy the Phase IV-A
    gate (check_gate.py C2a/C2b), which evaluates per-transition natural-shift
    magnitudes conditioned on informative stages.  These are proxies — they
    are computed from pilot margins and the deployed beta schedule rather than
    from replayed transitions through the frozen suite — so a candidate that
    passes the screening filter is a stronger candidate for passing the gate,
    not a guarantee.

    Primary metrics (gate-aligned proxies, drive ranking and acceptance):
      predicted_mean_abs_u_informative   — proxy for gate C2a (mean |u|)
      predicted_median_abs_u_informative — proxy for gate C2a (median |u|)
      predicted_frac_informative_ge_5e3  — proxy for gate C2b (frac >= 5e-3)

    Secondary / auxiliary metrics (retained for diagnostics):
      mean_abs_target_gap_norm           — spec §5.4 target-gap term
      informative_stage_frac             — prerequisite: stages must exist
      mean_abs_u_pred                    — legacy stage-level mean (diagnostic)
      frac_u_ge_5e3                      — legacy stage-level frac (diagnostic)

    Batch z-score standardization is applied in ``score_all_candidates``.

    Parameters
    ----------
    pilot_data : dict
        Output of ``run_classical_pilot`` (must contain margins_by_stage and
        p_align_by_stage).
    schedule : dict
        Output of ``build_schedule_v3_from_pilot`` (must contain beta_used_t,
        xi_ref_t, u_ref_used_t).
    reward_bound : float
        One-step reward bound R_max.
    weights : dict or None
        Score weights.  Defaults: w1=1.0 (mean_abs_u_info), w2=1.0
        (frac_info), w3=0.5 (target_gap), w4=0.25 (informative_stage_frac,
        auxiliary).  informative_stage_frac weight is deliberately small
        so the informative-transition metrics dominate.

    Returns
    -------
    dict
        Keys: raw_metrics (dict), score_components (dict), total_score (float),
        has_informative_stages (bool).
    """
    if weights is None:
        weights = {
            "w1_mean_abs_u_informative": 1.0,
            "w2_frac_informative_ge_5e3": 1.0,
            "w3_mean_abs_target_gap_norm": 0.5,
            "w4_informative_stage_frac": 0.25,  # auxiliary; low weight
        }

    u_ref_used = np.asarray(schedule["u_ref_used_t"], dtype=np.float64)
    beta_used = np.asarray(schedule["beta_used_t"], dtype=np.float64)
    xi_ref = np.asarray(schedule["xi_ref_t"], dtype=np.float64)
    gamma_base = float(schedule["gamma_base"])

    p_align = np.asarray(pilot_data["p_align_by_stage"], dtype=np.float64)
    margins_by_stage = pilot_data["margins_by_stage"]

    T = len(u_ref_used)
    T_min = min(T, len(p_align), len(xi_ref))

    # ------------------------------------------------------------------
    # Informative-stage mask (same criterion as check_gate.py / replay)
    # Stage t is informative iff xi_ref_t * sqrt(p_align_t) >= 0.05
    # ------------------------------------------------------------------
    _INFO_THRESH = 0.05
    informative_mask: list[bool] = []
    for t in range(T_min):
        score = float(xi_ref[t]) * float(np.sqrt(max(float(p_align[t]), 0.0)))
        informative_mask.append(score >= _INFO_THRESH)
    # Stages beyond T_min have no p_align data; treat as non-informative.
    informative_stage_frac = float(sum(informative_mask)) / max(T_min, 1)
    has_informative_stages = any(informative_mask)

    # ------------------------------------------------------------------
    # Primary metrics: gate-aligned informative-subset transition proxies
    # For each informative stage t, predicted per-transition |u_i| =
    #   |beta_used_t| * |m_i|  for m_i in margins_by_stage[t].
    # This mirrors the gate's frac_informative_u_ge_5e3 computation.
    # ------------------------------------------------------------------
    u_pred_info: list[float] = []
    for t in range(T_min):
        if not informative_mask[t]:
            continue
        b = abs(float(beta_used[t]))
        if t < len(margins_by_stage):
            m_arr = np.asarray(margins_by_stage[t], dtype=np.float64)
            if m_arr.size > 0:
                u_pred_info.extend((b * np.abs(m_arr)).tolist())

    if has_informative_stages and len(u_pred_info) > 0:
        u_info_arr = np.array(u_pred_info, dtype=np.float64)
        predicted_mean_abs_u_informative = float(np.mean(u_info_arr))
        predicted_median_abs_u_informative = float(np.median(u_info_arr))
        predicted_frac_informative_ge_5e3 = float(np.mean(u_info_arr >= 5e-3))
    else:
        # No informative stages or no margin data: metrics are undefined.
        # Set to 0 so they fail the acceptance filter explicitly.
        predicted_mean_abs_u_informative = 0.0
        predicted_median_abs_u_informative = 0.0
        predicted_frac_informative_ge_5e3 = 0.0

    # ------------------------------------------------------------------
    # Secondary metric: target-gap proxy (spec §5.4 w3 term)
    # Small-signal: gap ~ (gamma / (2*(1+gamma))) * |beta_t| * mean(m^2)
    # ------------------------------------------------------------------
    coeff_tg = gamma_base / (2.0 * (1.0 + gamma_base))
    target_gap_terms = []
    for t in range(min(T, len(margins_by_stage))):
        m_arr = np.asarray(margins_by_stage[t], dtype=np.float64)
        mean_m2 = float(np.mean(m_arr ** 2)) if m_arr.size > 0 else 0.0
        target_gap_terms.append(coeff_tg * abs(float(beta_used[t])) * mean_m2)
    mean_abs_target_gap = float(np.mean(target_gap_terms)) if target_gap_terms else 0.0
    rb = max(reward_bound, 1e-8)
    mean_abs_target_gap_norm = mean_abs_target_gap / rb

    # ------------------------------------------------------------------
    # Legacy stage-level diagnostics (retained for backwards-compat and
    # comparison; NOT used for acceptance or primary ranking)
    # ------------------------------------------------------------------
    mean_abs_u_pred = float(np.mean(np.abs(u_ref_used)))
    frac_u_ge_5e3 = float(np.mean(np.abs(u_ref_used) >= 5e-3))

    raw_metrics = {
        # Primary (gate-aligned informative-subset proxies)
        "predicted_mean_abs_u_informative": predicted_mean_abs_u_informative,
        "predicted_median_abs_u_informative": predicted_median_abs_u_informative,
        "predicted_frac_informative_ge_5e3": predicted_frac_informative_ge_5e3,
        # Secondary / auxiliary
        "mean_abs_target_gap_norm": mean_abs_target_gap_norm,
        "informative_stage_frac": informative_stage_frac,
        # Legacy diagnostics
        "mean_abs_u_pred": mean_abs_u_pred,
        "frac_u_ge_5e3": frac_u_ge_5e3,
    }

    w1 = weights.get("w1_mean_abs_u_informative", 1.0)
    w2 = weights.get("w2_frac_informative_ge_5e3", 1.0)
    w3 = weights.get("w3_mean_abs_target_gap_norm", 0.5)
    w4 = weights.get("w4_informative_stage_frac", 0.25)

    score_components = {
        "w1_x_mean_abs_u_informative": w1 * predicted_mean_abs_u_informative,
        "w2_x_frac_informative_ge_5e3": w2 * predicted_frac_informative_ge_5e3,
        "w3_x_mean_abs_target_gap_norm": w3 * mean_abs_target_gap_norm,
        "w4_x_informative_stage_frac": w4 * informative_stage_frac,
    }

    total_score = sum(score_components.values())

    return {
        "raw_metrics": raw_metrics,
        "score_components": score_components,
        "total_score": total_score,
        "has_informative_stages": has_informative_stages,
    }


# ---------------------------------------------------------------------------
# 3. Suite selection
# ---------------------------------------------------------------------------


def select_activation_suite(
    scored_candidates: list[dict[str, Any]],
    min_per_family: int = 1,
    max_per_family: int = 2,
) -> list[dict[str, Any]]:
    """Select the activation suite from scored candidates.

    Applies a gate-aligned screening filter before ranking.  Candidates must
    satisfy proxies for gate conditions C2a and C2b (check_gate.py):

      C2a proxy: predicted_mean_abs_u_informative >= 5e-3
                 OR predicted_median_abs_u_informative >= 5e-3
      C2b proxy: predicted_frac_informative_ge_5e3 >= 0.10

    Candidates with no informative stages fail acceptance explicitly
    (the proxies are undefined / zero, not a borderline value).

    Thresholds 5e-3 and 0.10 are taken directly from gate C2a/C2b; no new
    numbers are introduced.  The filter is a gate-aligned screening proxy,
    not the gate itself: pilot-based predictions carry noise relative to
    replayed transitions through the frozen suite.

    Parameters
    ----------
    scored_candidates : list of dict
        Each dict must have keys: family, scoring (output of
        compute_candidate_score), pilot_data, schedule, cfg.
    min_per_family : int
        Minimum number of tasks to select per family (best-effort).
    max_per_family : int
        Maximum number of tasks to select per family.

    Returns
    -------
    list of dict
        Selected candidates (subset of input), each augmented with
        "selected_reason" and "acceptance_status".
    """
    # Gate thresholds taken verbatim from check_gate.py C2a / C2b
    _C2A_THRESHOLD = 5e-3
    _C2B_THRESHOLD = 0.10

    # Group by family
    by_family: dict[str, list[dict]] = defaultdict(list)
    for cand in scored_candidates:
        family = cand.get("family", "unknown")
        by_family[family].append(cand)

    selected: list[dict] = []

    for family, candidates in sorted(by_family.items()):
        # Filter by gate-aligned acceptance proxies
        passing = []
        for c in candidates:
            # Explicit rejection when informative set is empty
            if not c["scoring"].get("has_informative_stages", False):
                continue
            metrics = c["scoring"]["raw_metrics"]
            c2a_proxy = (
                metrics["predicted_mean_abs_u_informative"] >= _C2A_THRESHOLD
                or metrics["predicted_median_abs_u_informative"] >= _C2A_THRESHOLD
            )
            c2b_proxy = (
                metrics["predicted_frac_informative_ge_5e3"] >= _C2B_THRESHOLD
            )
            if c2a_proxy and c2b_proxy:
                passing.append(c)

        if len(passing) == 0:
            logger.warning(
                "Family '%s': no candidates pass gate-aligned acceptance filter "
                "(C2a proxy >= %.0e, C2b proxy >= %.2f). "
                "%d candidates scored.",
                family, _C2A_THRESHOLD, _C2B_THRESHOLD,
                len(candidates),
            )
            continue

        # Sort by total_score descending
        passing.sort(key=lambda c: c["scoring"]["total_score"], reverse=True)

        # Select up to max_per_family
        n_select = min(max_per_family, len(passing))
        for c in passing[:n_select]:
            c_out = dict(c)
            c_out["acceptance_status"] = "accepted"
            c_out["selected_reason"] = (
                f"Rank {passing.index(c)+1}/{len(passing)} in family "
                f"'{family}' (score={c['scoring']['total_score']:.6f})"
            )
            selected.append(c_out)

        logger.info(
            "Family '%s': %d/%d pass acceptance, selected %d.",
            family, len(passing), len(candidates), n_select,
        )

    return selected


# ---------------------------------------------------------------------------
# 4. Full search pipeline
# ---------------------------------------------------------------------------


def _z_score_standardize(
    values: NDArray[np.float64],
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Z-score standardize an array, with epsilon for stability."""
    mu = np.mean(values)
    sigma = np.std(values)
    return (values - mu) / max(sigma, eps)


def score_all_candidates(
    search_grid: list[dict[str, Any]] | None = None,
    seed: int = 42,
    n_pilot_episodes: int = 50,
    score_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Run the full activation search pipeline over all candidates.

    This function MUST NOT import or read any Phase IV safe result files.

    Parameters
    ----------
    search_grid : list of dict or None
        Candidate configs. If None, uses ``get_search_grid()``.
    seed : int
        Random seed for pilots.
    n_pilot_episodes : int
        Number of pilot episodes per candidate.
    score_weights : dict or None
        Score weights for ``compute_candidate_score``.

    Returns
    -------
    list of dict
        Scored candidate dicts, each containing: cfg, family,
        pilot_data, schedule, scoring.
    """
    if search_grid is None:
        search_grid = get_search_grid()

    if score_weights is None:
        score_weights = {
            "w1_mean_abs_u": 1.0,
            "w2_mean_abs_delta_d": 1.0,
            "w3_mean_abs_target_gap_norm": 1.0,
            "w4_informative_stage_frac": 0.5,
        }

    candidates: list[dict[str, Any]] = []

    for idx, cfg in enumerate(search_grid):
        family = cfg.get("family", "unknown")
        logger.info(
            "Scoring candidate %d/%d: family=%s",
            idx + 1, len(search_grid), family,
        )

        try:
            # Step 1: Run classical pilot (raw margins; p_align uses sign=+1 initially)
            pilot_data = run_classical_pilot(
                cfg, seed=seed, n_episodes=n_pilot_episodes,
            )

            # Step 2: Determine sign from margin distribution, then
            # recompute sign-aligned p_align before building the schedule.
            gamma_val = float(cfg.get("gamma", 0.97))
            r_max = float(cfg.get("reward_bound", 1.0))
            resolved_sign = select_sign(pilot_data["margins_by_stage"], r_max)

            corrected_p_align: list[float] = []
            for margins_arr in pilot_data["margins_by_stage"]:
                m = np.asarray(margins_arr, dtype=np.float64)
                corrected_p_align.append(
                    float(np.mean(resolved_sign * m > 0.0)) if len(m) > 0 else 0.0
                )
            pilot_data = {**pilot_data, "p_align_by_stage": corrected_p_align}

            schedule = build_schedule_v3_from_pilot(
                pilot_data=pilot_data,
                r_max=r_max,
                gamma_base=gamma_val,
                gamma_eval=gamma_val,
                sign_family=resolved_sign,
                task_family=family,
                source_phase="pilot",
                notes=f"activation_search candidate {idx}",
            )

            # Step 3: Compute score
            scoring = compute_candidate_score(
                pilot_data=pilot_data,
                schedule=schedule,
                reward_bound=r_max,
                weights=score_weights,
            )

            candidates.append({
                "idx": idx,
                "cfg": cfg,
                "family": family,
                "pilot_data": pilot_data,
                "schedule": schedule,
                "scoring": scoring,
            })

        except Exception as exc:
            logger.error(
                "Candidate %d (family=%s) failed: %s", idx, family, exc
            )
            candidates.append({
                "idx": idx,
                "cfg": cfg,
                "family": family,
                "pilot_data": None,
                "schedule": None,
                "scoring": {
                    "raw_metrics": {
                        "predicted_mean_abs_u_informative": 0.0,
                        "predicted_median_abs_u_informative": 0.0,
                        "predicted_frac_informative_ge_5e3": 0.0,
                        "mean_abs_target_gap_norm": 0.0,
                        "informative_stage_frac": 0.0,
                        "mean_abs_u_pred": 0.0,
                        "frac_u_ge_5e3": 0.0,
                    },
                    "score_components": {},
                    "total_score": 0.0,
                    "has_informative_stages": False,
                },
                "error": str(exc),
            })

    # -----------------------------------------------------------------
    # Batch z-score standardization
    # Primary metrics drive ranking; informative_stage_frac is auxiliary.
    # -----------------------------------------------------------------
    metric_keys = [
        "predicted_mean_abs_u_informative",
        "predicted_frac_informative_ge_5e3",
        "mean_abs_target_gap_norm",
        "informative_stage_frac",
    ]
    weight_keys = [
        "w1_mean_abs_u_informative",
        "w2_frac_informative_ge_5e3",
        "w3_mean_abs_target_gap_norm",
        "w4_informative_stage_frac",
    ]

    # Only standardize if we have more than one valid candidate
    valid_mask = [c.get("error") is None for c in candidates]
    n_valid = sum(valid_mask)

    if n_valid > 1:
        # Extract raw metric arrays
        metric_arrays: dict[str, NDArray[np.float64]] = {}
        for mk in metric_keys:
            vals = np.array([
                c["scoring"]["raw_metrics"][mk]
                for c in candidates
            ], dtype=np.float64)
            metric_arrays[mk] = vals

        # Z-score and recompute total_score
        z_arrays: dict[str, NDArray[np.float64]] = {}
        for mk in metric_keys:
            z_arrays[mk] = _z_score_standardize(metric_arrays[mk])

        for i, c in enumerate(candidates):
            total = 0.0
            for mk, wk in zip(metric_keys, weight_keys):
                w = score_weights.get(wk, 1.0)
                z_val = float(z_arrays[mk][i])
                c["scoring"]["score_components"][f"z_{mk}"] = z_val
                total += w * z_val
            c["scoring"]["total_score"] = total

    return candidates
