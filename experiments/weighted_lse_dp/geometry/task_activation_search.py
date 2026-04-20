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
    """Finite-horizon backward VI to get V* and Q*. Returns (V_star, Q_star)."""
    if not (hasattr(mdp_base, "p") and hasattr(mdp_base, "r")):
        S = int(mdp_base.info.observation_space.n)
        A = int(mdp_base.info.action_space.n)
        return np.zeros(S), np.zeros((S, A))
    P = np.asarray(mdp_base.p, dtype=np.float64)   # (S, A, S')
    R_raw = np.asarray(mdp_base.r, dtype=np.float64)
    # MushroomRL stores R as (S, A, S') — compute expected reward r̄(s,a)
    if R_raw.ndim == 3:
        R = np.einsum("ijk,ijk->ij", P, R_raw)   # (S, A)
    else:
        R = R_raw   # already (S, A)
    S, A = R.shape[:2]
    V = np.zeros(S, dtype=np.float64)
    for _ in range(horizon):
        Q = R + gamma * np.einsum("ijk,k->ij", P, V)   # (S, A)
        V = Q.max(axis=1)
    Q_star = R + gamma * np.einsum("ijk,k->ij", P, V)
    return V, Q_star


# ---------------------------------------------------------------------------
# 1. Classical pilot
# ---------------------------------------------------------------------------


def run_classical_pilot(
    cfg: dict[str, Any],
    seed: int = 42,
    n_episodes: int = 50,
    max_steps: int | None = None,
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

    Returns
    -------
    dict
        Pilot data with keys: margins_by_stage, p_align_by_stage,
        n_by_stage, episode_returns, event_rate, n_episodes, gamma,
        horizon, reward_bound.
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)

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

    for ep_rewards, ep_stages, ep_next_bases in zip(
        all_episode_rewards, all_episode_stages, all_episode_next_bases
    ):
        T_ep = len(ep_rewards)
        if T_ep == 0:
            continue

        for i in range(T_ep):
            r_t = ep_rewards[i]
            nb = ep_next_bases[i]
            # V*(terminal) = 0; guard against out-of-bound index
            v_next = float(V_star[nb]) if (len(V_star) > 0 and nb < len(V_star)) else 0.0
            # terminal step: V*(s') = 0 (absorbing)
            if i == T_ep - 1 and len(ep_rewards) < max_steps:
                v_next = 0.0
            margin = r_t - v_next   # no gamma (lessons.md: margin formula)
            stage = ep_stages[i]
            margins_by_stage[stage].append(margin)

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
            p_align_by_stage_list.append(
                float(np.mean(margins_arr > 0.0))
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

    return {
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

    Per spec S5.4, computes four raw metrics from the schedule and pilot
    data, then returns both raw values and (for single-candidate use)
    the unweighted sum as a score placeholder. True z-score
    standardization is applied across the batch in ``score_all_candidates``.

    Parameters
    ----------
    pilot_data : dict
        Output of ``run_classical_pilot``.
    schedule : dict
        Output of ``build_schedule_v3_from_pilot``.
    reward_bound : float
        One-step reward bound R_max.
    weights : dict or None
        Score weights. Defaults to spec values:
        w1=1.0, w2=1.0, w3=1.0, w4=0.5.

    Returns
    -------
    dict
        Keys: raw_metrics (dict of float), score_components (dict),
        total_score (float).
    """
    if weights is None:
        weights = {
            "w1_mean_abs_u": 1.0,
            "w2_mean_abs_delta_d": 1.0,
            "w3_mean_abs_target_gap_norm": 1.0,
            "w4_informative_stage_frac": 0.5,
        }

    u_ref_used = np.asarray(schedule["u_ref_used_t"], dtype=np.float64)
    beta_used = np.asarray(schedule["beta_used_t"], dtype=np.float64)
    xi_ref = np.asarray(schedule["xi_ref_t"], dtype=np.float64)
    gamma_base = float(schedule["gamma_base"])

    p_align = np.asarray(pilot_data["p_align_by_stage"], dtype=np.float64)
    margins_by_stage = pilot_data["margins_by_stage"]

    T = len(u_ref_used)

    # --- Raw metric 1: mean |u_pred| ---
    mean_abs_u_pred = float(np.mean(np.abs(u_ref_used)))

    # --- Raw metric 2: mean |delta_d_pred| ---
    # Use small-signal approximation: delta_d ~ -(gamma / (1+gamma)) * u
    coeff_dd = gamma_base / (1.0 + gamma_base)
    delta_d_pred = coeff_dd * np.abs(u_ref_used)
    mean_abs_delta_d_pred = float(np.mean(delta_d_pred))

    # --- Raw metric 3: mean |target_gap| / R_max ---
    # Small-signal: gap ~ (gamma / (2*(1+gamma))) * beta * margin^2
    # Use mean margin^2 per stage * beta_used
    coeff_tg = gamma_base / (2.0 * (1.0 + gamma_base))
    target_gap_terms = []
    for t in range(min(T, len(margins_by_stage))):
        m_arr = np.asarray(margins_by_stage[t], dtype=np.float64)
        if m_arr.size > 0:
            mean_m2 = float(np.mean(m_arr**2))
        else:
            mean_m2 = 0.0
        target_gap_terms.append(coeff_tg * abs(float(beta_used[t])) * mean_m2)
    mean_abs_target_gap = float(np.mean(target_gap_terms)) if target_gap_terms else 0.0
    rb = max(reward_bound, 1e-8)
    mean_abs_target_gap_norm = mean_abs_target_gap / rb

    # --- Raw metric 4: informative stage fraction ---
    # Fraction of stages where xi_ref_t * sqrt(p_align_t) >= 0.05
    T_min = min(T, len(p_align))
    informative_count = 0
    for t in range(T_min):
        info_score = float(xi_ref[t]) * float(np.sqrt(max(p_align[t], 0.0)))
        if info_score >= 0.05:
            informative_count += 1
    informative_stage_frac = informative_count / max(T_min, 1)

    # --- Fraction of stages with |u| >= 5e-3 ---
    frac_u_ge_5e3 = float(np.mean(np.abs(u_ref_used) >= 5e-3))

    raw_metrics = {
        "mean_abs_u_pred": mean_abs_u_pred,
        "mean_abs_delta_d_pred": mean_abs_delta_d_pred,
        "mean_abs_target_gap_norm": mean_abs_target_gap_norm,
        "informative_stage_frac": informative_stage_frac,
        "frac_u_ge_5e3": frac_u_ge_5e3,
    }

    # For a single candidate, z-score is trivial (raw value).
    # The batch-level standardization is done in score_all_candidates.
    w1 = weights.get("w1_mean_abs_u", 1.0)
    w2 = weights.get("w2_mean_abs_delta_d", 1.0)
    w3 = weights.get("w3_mean_abs_target_gap_norm", 1.0)
    w4 = weights.get("w4_informative_stage_frac", 0.5)

    score_components = {
        "w1_x_mean_abs_u": w1 * mean_abs_u_pred,
        "w2_x_mean_abs_delta_d": w2 * mean_abs_delta_d_pred,
        "w3_x_mean_abs_target_gap_norm": w3 * mean_abs_target_gap_norm,
        "w4_x_informative_stage_frac": w4 * informative_stage_frac,
    }

    total_score = sum(score_components.values())

    return {
        "raw_metrics": raw_metrics,
        "score_components": score_components,
        "total_score": total_score,
    }


# ---------------------------------------------------------------------------
# 3. Suite selection
# ---------------------------------------------------------------------------


def select_activation_suite(
    scored_candidates: list[dict[str, Any]],
    min_per_family: int = 1,
    max_per_family: int = 2,
    min_mean_abs_u_pred: float = 2e-3,
    min_frac_active_stages: float = 0.05,
) -> list[dict[str, Any]]:
    """Select the activation suite from scored candidates.

    Per spec S5.3, applies minimum acceptance criteria and per-family caps.

    Parameters
    ----------
    scored_candidates : list of dict
        Each dict must have keys: family, scoring (output of
        compute_candidate_score), pilot_data, schedule, cfg.
    min_per_family : int
        Minimum number of tasks to select per family (best-effort).
    max_per_family : int
        Maximum number of tasks to select per family.
    min_mean_abs_u_pred : float
        Minimum predicted mean |u| for acceptance (2e-3 for search).
    min_frac_active_stages : float
        Minimum fraction of stages with |u| >= 5e-3 (0.05).

    Returns
    -------
    list of dict
        Selected candidates (subset of input), each augmented with
        "selected_reason" and "acceptance_status".
    """
    # Group by family
    by_family: dict[str, list[dict]] = defaultdict(list)
    for cand in scored_candidates:
        family = cand.get("family", "unknown")
        by_family[family].append(cand)

    selected: list[dict] = []

    for family, candidates in sorted(by_family.items()):
        # Filter by acceptance criteria
        passing = []
        for c in candidates:
            metrics = c["scoring"]["raw_metrics"]
            u_ok = metrics["mean_abs_u_pred"] >= min_mean_abs_u_pred
            frac_ok = metrics["frac_u_ge_5e3"] >= min_frac_active_stages
            if u_ok and frac_ok:
                passing.append(c)

        if len(passing) == 0:
            logger.warning(
                "Family '%s': no candidates pass acceptance criteria "
                "(min_mean_abs_u_pred=%.4f, min_frac_active=%.4f). "
                "%d candidates scored.",
                family, min_mean_abs_u_pred, min_frac_active_stages,
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
            # Step 1: Run classical pilot
            pilot_data = run_classical_pilot(
                cfg, seed=seed, n_episodes=n_pilot_episodes,
            )

            # Step 2: Build schedule v3
            gamma_val = float(cfg.get("gamma", 0.97))
            r_max = float(cfg.get("reward_bound", 1.0))

            schedule = build_schedule_v3_from_pilot(
                pilot_data=pilot_data,
                r_max=r_max,
                gamma_base=gamma_val,
                gamma_eval=gamma_val,
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
                        "mean_abs_u_pred": 0.0,
                        "mean_abs_delta_d_pred": 0.0,
                        "mean_abs_target_gap_norm": 0.0,
                        "informative_stage_frac": 0.0,
                        "frac_u_ge_5e3": 0.0,
                    },
                    "score_components": {},
                    "total_score": 0.0,
                },
                "error": str(exc),
            })

    # -----------------------------------------------------------------
    # Batch z-score standardization
    # -----------------------------------------------------------------
    metric_keys = [
        "mean_abs_u_pred",
        "mean_abs_delta_d_pred",
        "mean_abs_target_gap_norm",
        "informative_stage_frac",
    ]
    weight_keys = [
        "w1_mean_abs_u",
        "w2_mean_abs_delta_d",
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
