#!/usr/bin/env python
"""Phase IV-A: counterfactual target replay.

Freezes transitions from classical random-policy pilots and replays them
through the safe TAB formula to isolate operator effects from exploration
differences.  For each task in the activation suite, collects (r, v_next)
pairs and computes exact safe targets, discount shifts, and natural-shift
diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Ensure repo root and mushroom-rl-dev are on the path when run as a script.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MUSHROOM_DEV = str(Path(_REPO_ROOT) / "mushroom-rl-dev")
if _MUSHROOM_DEV not in sys.path:
    sys.path.insert(0, _MUSHROOM_DEV)

from experiments.weighted_lse_dp.common.seeds import seed_everything
from experiments.weighted_lse_dp.geometry.phase4_calibration_v3 import (
    build_schedule_v3_from_pilot,
)
from experiments.weighted_lse_dp.geometry.task_activation_search import (
    run_classical_pilot,
)
from experiments.weighted_lse_dp.geometry.trust_region import kl_bernoulli
from experiments.weighted_lse_dp.tasks.phase4_operator_suite import (
    build_phase4_task,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------

def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    result = np.empty_like(x)
    exp_neg = np.exp(-np.where(pos, x, 0.0))
    result[pos] = 1.0 / (1.0 + exp_neg[pos])
    exp_pos = np.exp(np.where(~pos, x, 0.0))
    result[~pos] = exp_pos[~pos] / (1.0 + exp_pos[~pos])
    return result


# --------------------------------------------------------------------------
# Pilot that also captures per-transition (r, v_next, stage)
# --------------------------------------------------------------------------

def _run_pilot_with_transitions(
    cfg: dict[str, Any],
    seed: int,
    n_episodes: int,
) -> tuple[dict[str, Any], list[tuple[float, float, int]]]:
    """Run classical pilot AND collect frozen (r, v_next, stage) transitions.

    Returns
    -------
    pilot_data : dict
        Standard pilot data from ``run_classical_pilot``.
    transitions : list of (r, v_next, stage)
        One entry per transition across all episodes.
    """
    # Run the standard pilot to get margins and schedule-building data
    pilot_data = run_classical_pilot(cfg, seed=seed, n_episodes=n_episodes)

    # Re-run to collect raw transitions with the value proxy
    # (the pilot already ran episodes; we replay its logic to extract
    #  per-transition v_next using the same seed)
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    mdp_base, mdp_rl, resolved_cfg = build_phase4_task(cfg, seed=seed)

    gamma = float(resolved_cfg.get("gamma", mdp_rl.info.gamma))
    horizon = int(resolved_cfg.get("horizon", mdp_rl.info.horizon))

    # Detect time-augmented env
    try:
        from mushroom_rl.environments.time_augmented_env import (
            DiscreteTimeAugmentedEnv,
        )
        time_aug = isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
    except ImportError:
        time_aug = hasattr(mdp_rl, "n_base_states")

    if hasattr(mdp_base, "p"):
        n_base = int(mdp_base.p.shape[0])
    else:
        n_base = int(mdp_base.info.observation_space.n)

    n_actions = int(mdp_rl.info.action_space.n)

    transitions: list[tuple[float, float, int]] = []

    for ep in range(n_episodes):
        state, _ = mdp_rl.reset()
        ep_rewards: list[float] = []
        ep_stages: list[int] = []

        for step_idx in range(horizon):
            if time_aug:
                state_idx = int(np.asarray(state).flat[0])
                stage = state_idx // n_base
            else:
                stage = step_idx

            ep_stages.append(stage)
            action = np.array([rng.integers(0, n_actions)])
            next_state, reward, absorbing, info = mdp_rl.step(action)
            ep_rewards.append(float(reward))
            state = next_state
            if absorbing:
                break

        T_ep = len(ep_rewards)
        if T_ep == 0:
            continue

        # Compute discounted returns from each step onward
        value_proxies = np.zeros(T_ep, dtype=np.float64)
        running = 0.0
        for i in range(T_ep - 1, -1, -1):
            running = ep_rewards[i] + gamma * running
            value_proxies[i] = running

        for i in range(T_ep):
            r_t = ep_rewards[i]
            if i + 1 < T_ep:
                v_next = value_proxies[i + 1]
            else:
                v_next = 0.0
            transitions.append((r_t, v_next, ep_stages[i]))

    return pilot_data, transitions


# --------------------------------------------------------------------------
# Core counterfactual replay for one task
# --------------------------------------------------------------------------

def _replay_task(
    task_entry: dict[str, Any],
    seed: int,
    n_episodes: int,
    output_dir: Path,
    task_idx: int,
) -> dict[str, Any]:
    """Run counterfactual replay for a single task and write results.

    Returns the replay summary dict.
    """
    cfg = task_entry["cfg"]
    family = task_entry["family"]
    tag = f"{family}_{task_idx}"

    logger.info("=== Replaying task %s (idx=%d) ===", family, task_idx)

    # Step 1-2: pilot + transitions
    pilot_data, transitions = _run_pilot_with_transitions(
        cfg, seed=seed, n_episodes=n_episodes,
    )

    gamma_val = float(cfg.get("gamma", 0.97))
    r_max = float(cfg.get("reward_bound", 1.0))

    # Step 3: build schedule
    schedule = build_schedule_v3_from_pilot(
        pilot_data=pilot_data,
        r_max=r_max,
        gamma_base=gamma_val,
        gamma_eval=gamma_val,
        task_family=family,
        source_phase="counterfactual_replay",
        notes=f"counterfactual replay task {tag}",
    )

    gamma_b = float(schedule["gamma_base"])
    log_gamma_b = np.log(gamma_b)
    log_1_plus_gamma = np.log(1.0 + gamma_b)
    one_plus_gamma = 1.0 + gamma_b

    beta_arr = np.asarray(schedule["beta_used_t"], dtype=np.float64)
    T_schedule = len(beta_arr)

    # Schedule arrays
    xi_ref_t = np.asarray(schedule["xi_ref_t"], dtype=np.float64)
    u_target_t = np.asarray(schedule["u_target_t"], dtype=np.float64)
    u_tr_cap_t = np.asarray(schedule["u_tr_cap_t"], dtype=np.float64)
    U_safe_ref_t = np.asarray(schedule["U_safe_ref_t"], dtype=np.float64)
    u_ref_used_t = np.asarray(schedule["u_ref_used_t"], dtype=np.float64)
    trust_clip_active_t = schedule["trust_clip_active_t"]
    safe_clip_active_t = schedule["safe_clip_active_t"]

    # A_t: from schedule or recompute
    A_t_list = schedule.get("A_t")
    Bhat_t_list = schedule.get("Bhat_t")

    N = len(transitions)
    if N == 0:
        logger.warning("Task %s: no transitions collected.", tag)
        empty_summary: dict[str, Any] = {"family": family, "tag": tag, "n_transitions": 0}
        task_dir = output_dir / tag
        task_dir.mkdir(parents=True, exist_ok=True)
        with open(task_dir / "replay_summary.json", "w") as f:
            json.dump(empty_summary, f, indent=2)
        return empty_summary

    # Pre-allocate arrays
    r_all = np.zeros(N, dtype=np.float64)
    v_next_all = np.zeros(N, dtype=np.float64)
    stage_all = np.zeros(N, dtype=np.int64)
    classical_target = np.zeros(N, dtype=np.float64)
    safe_target = np.zeros(N, dtype=np.float64)
    rho_all = np.zeros(N, dtype=np.float64)
    eff_discount = np.zeros(N, dtype=np.float64)
    delta_eff_discount = np.zeros(N, dtype=np.float64)
    natural_shift = np.zeros(N, dtype=np.float64)
    target_gap = np.zeros(N, dtype=np.float64)
    margin_all = np.zeros(N, dtype=np.float64)
    beta_used_all = np.zeros(N, dtype=np.float64)
    kl_to_prior_all = np.zeros(N, dtype=np.float64)
    A_t_all = np.zeros(N, dtype=np.float64)
    xi_ref_all = np.zeros(N, dtype=np.float64)
    u_target_all = np.zeros(N, dtype=np.float64)
    u_tr_cap_all = np.zeros(N, dtype=np.float64)
    U_safe_ref_all = np.zeros(N, dtype=np.float64)
    u_ref_used_all = np.zeros(N, dtype=np.float64)
    trust_clip_all = np.zeros(N, dtype=np.bool_)
    safe_clip_all = np.zeros(N, dtype=np.bool_)

    for i, (r, v_next, t) in enumerate(transitions):
        r_all[i] = r
        v_next_all[i] = v_next
        stage_all[i] = t

        # Clamp stage index to schedule length
        t_clamped = min(t, T_schedule - 1)
        beta = beta_arr[t_clamped]
        beta_used_all[i] = beta

        # Classical target (same gamma_base for fair comparison)
        classical_target[i] = r + gamma_b * v_next

        # Safe target
        if abs(beta) < 1e-12:
            safe_target[i] = r + gamma_b * v_next
        else:
            c = one_plus_gamma / beta
            safe_target[i] = c * (
                np.logaddexp(beta * r, beta * v_next + log_gamma_b)
                - log_1_plus_gamma
            )

        # rho = sigmoid(log(1/gamma_b) + beta*(r - v_next))
        arg = -log_gamma_b + beta * (r - v_next)
        rho_all[i] = 1.0 / (1.0 + np.exp(-arg)) if arg >= 0 else np.exp(arg) / (1.0 + np.exp(arg))

        eff_discount[i] = one_plus_gamma * (1.0 - rho_all[i])
        delta_eff_discount[i] = eff_discount[i] - gamma_b
        natural_shift[i] = beta * (r - v_next)
        target_gap[i] = safe_target[i] - classical_target[i]
        margin_all[i] = r - v_next

        # Schedule diagnostics at this stage
        if A_t_list is not None and t_clamped < len(A_t_list):
            A_t_all[i] = A_t_list[t_clamped]
        elif Bhat_t_list is not None and t_clamped + 1 < len(Bhat_t_list):
            A_t_all[i] = r_max + Bhat_t_list[t_clamped + 1]
        else:
            A_t_all[i] = r_max

        if t_clamped < len(xi_ref_t):
            xi_ref_all[i] = xi_ref_t[t_clamped]
        if t_clamped < len(u_target_t):
            u_target_all[i] = u_target_t[t_clamped]
        if t_clamped < len(u_tr_cap_t):
            u_tr_cap_all[i] = u_tr_cap_t[t_clamped]
        if t_clamped < len(U_safe_ref_t):
            U_safe_ref_all[i] = U_safe_ref_t[t_clamped]
        if t_clamped < len(u_ref_used_t):
            u_ref_used_all[i] = u_ref_used_t[t_clamped]
        if t_clamped < len(trust_clip_active_t):
            trust_clip_all[i] = trust_clip_active_t[t_clamped]
        if t_clamped < len(safe_clip_active_t):
            safe_clip_all[i] = safe_clip_active_t[t_clamped]

    # KL to prior (vectorized)
    p0 = 1.0 / one_plus_gamma
    kl_to_prior_all = np.asarray(
        kl_bernoulli(rho_all, np.full(N, p0, dtype=np.float64)),
        dtype=np.float64,
    ).ravel()

    # Write NPZ
    task_dir = output_dir / tag
    task_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        task_dir / "replay_diagnostics.npz",
        schema_version="1.0",
        r=r_all,
        v_next=v_next_all,
        stage=stage_all,
        classical_target=classical_target,
        safe_target=safe_target,
        rho=rho_all,
        effective_discount=eff_discount,
        delta_effective_discount=delta_eff_discount,
        natural_shift=natural_shift,
        target_gap_same_gamma_base=target_gap,
        margin=margin_all,
        beta_used=beta_used_all,
        KL_to_prior=kl_to_prior_all,
        A_t=A_t_all,
        xi_ref=xi_ref_all,
        u_target=u_target_all,
        u_tr_cap=u_tr_cap_all,
        U_safe_ref=U_safe_ref_all,
        u_ref_used=u_ref_used_all,
        trust_clip_active=trust_clip_all,
        safe_clip_active=safe_clip_all,
    )

    # Compute summary
    abs_u = np.abs(natural_shift)
    abs_dd = np.abs(delta_eff_discount)
    abs_tg = np.abs(target_gap)
    rb = max(r_max, 1e-8)

    summary: dict[str, Any] = {
        "family": family,
        "tag": tag,
        "n_transitions": N,
        "gamma_base": gamma_b,
        "r_max": r_max,
        "seed": seed,
        "n_episodes": n_episodes,
        # natural_shift
        "mean_natural_shift": float(np.mean(natural_shift)),
        "std_natural_shift": float(np.std(natural_shift)),
        "p25_natural_shift": float(np.percentile(natural_shift, 25)),
        "p75_natural_shift": float(np.percentile(natural_shift, 75)),
        "mean_abs_u": float(np.mean(abs_u)),
        # delta_effective_discount
        "mean_delta_effective_discount": float(np.mean(delta_eff_discount)),
        "std_delta_effective_discount": float(np.std(delta_eff_discount)),
        "p25_delta_effective_discount": float(np.percentile(delta_eff_discount, 25)),
        "p75_delta_effective_discount": float(np.percentile(delta_eff_discount, 75)),
        "mean_abs_delta_d": float(np.mean(abs_dd)),
        # target_gap_same_gamma_base
        "mean_target_gap_same_gamma_base": float(np.mean(target_gap)),
        "std_target_gap_same_gamma_base": float(np.std(target_gap)),
        "p25_target_gap_same_gamma_base": float(np.percentile(target_gap, 25)),
        "p75_target_gap_same_gamma_base": float(np.percentile(target_gap, 75)),
        "mean_abs_target_gap": float(np.mean(abs_tg)),
        # Fractions
        "frac_u_ge_5e3": float(np.mean(abs_u >= 5e-3)),
        "frac_delta_d_ge_1e3": float(np.mean(abs_dd >= 1e-3)),
        "frac_target_gap_ge_5e3_normed": float(np.mean(abs_tg / rb >= 5e-3)),
        # Other
        "mean_beta_used": float(np.mean(beta_used_all)),
        "mean_KL_to_prior": float(np.mean(kl_to_prior_all)),
    }

    with open(task_dir / "replay_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info(
        "Task %s: %d transitions | mean|u|=%.6f frac(|u|>=5e-3)=%.4f "
        "mean|delta_d|=%.6f frac(|delta_d|>=1e-3)=%.4f "
        "mean|tg|/r_max=%.6f frac(|tg/r_max|>=5e-3)=%.4f "
        "mean_beta=%.6f mean_KL=%.8f",
        tag, N,
        summary["mean_abs_u"], summary["frac_u_ge_5e3"],
        summary["mean_abs_delta_d"], summary["frac_delta_d_ge_1e3"],
        summary["mean_abs_target_gap"] / rb, summary["frac_target_gap_ge_5e3_normed"],
        summary["mean_beta_used"], summary["mean_KL_to_prior"],
    )

    return summary


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Run Phase IV-A counterfactual target replay."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    suite_path = Path(args.suite)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(suite_path) as f:
        suite = json.load(f)

    selected_tasks = suite["selected_tasks"]
    logger.info(
        "Loaded activation suite with %d tasks from %s",
        len(selected_tasks), suite_path,
    )

    seed = args.seed
    n_episodes = args.n_episodes

    all_summaries: list[dict[str, Any]] = []
    t_start = time.time()

    for idx, task_entry in enumerate(selected_tasks):
        summary = _replay_task(
            task_entry=task_entry,
            seed=seed,
            n_episodes=n_episodes,
            output_dir=output_dir,
            task_idx=idx,
        )
        all_summaries.append(summary)

    t_end = time.time()

    # Write top-level summary
    top_summary = {
        "n_tasks": len(selected_tasks),
        "seed": seed,
        "n_episodes": n_episodes,
        "elapsed_s": round(t_end - t_start, 2),
        "tasks": all_summaries,
    }
    with open(output_dir / "all_replay_summaries.json", "w") as f:
        json.dump(top_summary, f, indent=2)

    logger.info(
        "Counterfactual replay complete: %d tasks in %.1f s. "
        "Results in %s",
        len(selected_tasks), t_end - t_start, output_dir,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite", type=str, required=True,
        help="Path to activation_suite.json",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument(
        "--output-dir", type=str,
        default="results/weighted_lse_dp/phase4/counterfactual_replay/",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
