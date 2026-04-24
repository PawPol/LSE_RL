#!/usr/bin/env python
"""Phase V WP5 -- Stage-1 paired RL pilot.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` sections 7
(WP5) and 14.5.

Arms (per task, per seed):

* ``classical_q``       -- MushroomRL QLearning at native gamma.
* ``safe_zero``         -- SafeTargetQLearning with a zeroed calibrated schedule.
* ``safe_nonlinear``    -- SafeTargetQLearning with the deployed calibrated schedule.
* ``tuned_fixed_gamma`` -- QLearning at gamma_eff tuned over {0.85,0.90,0.95,0.97,0.99}.
* ``td_lambda``         -- Watkins Q(lambda) tuned over {0.3, 0.6, 0.9}.

Paired CRN (option b): the calibrated safe schedule is built ONCE per
(task, seed) BEFORE any arm-training loop runs.  Before each arm's training
loop we re-seed numpy/random/torch with ``seed`` fresh so every arm sees
the same starting RNG state.

CLI::

    python -m experiments.weighted_lse_dp.runners.run_phase_V_rl_pilot \\
        --shortlist results/search/shortlist.csv \\
        --tasks A_003 A_000 \\
        --seeds 42 123 456 789 1024 \\
        --output-root results/rl/ [--n-episodes 1000] [--eval-every 10]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Path bootstrapping so ``-m`` invocation works without editable install.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MUSHROOM = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.common.manifests import git_sha  # noqa: E402
from experiments.weighted_lse_dp.common.seeds import seed_everything  # noqa: E402
from experiments.weighted_lse_dp.runners.run_phase_V_search import (  # noqa: E402
    _calibrate_schedule,
)
from experiments.weighted_lse_dp.tasks.family_a_jackpot_vs_stream import (  # noqa: E402
    family_a,
)

logger = logging.getLogger("phaseV.rl_pilot")

__all__ = ["main", "run_pilot", "ARMS", "DEFAULT_SEEDS", "DEFAULT_TASKS"]


SCHEMA_VERSION: str = "phaseV.rl_pilot.v1"
ARMS: tuple[str, ...] = (
    "classical_q", "safe_zero", "safe_nonlinear",
    "tuned_fixed_gamma", "td_lambda",
)
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)
DEFAULT_TASKS: tuple[str, ...] = ("A_003", "A_000")
GAMMA_GRID: tuple[float, ...] = (0.85, 0.90, 0.95, 0.97, 0.99)
LAMBDA_GRID: tuple[float, ...] = (0.3, 0.6, 0.9)
SUBPILOT_SEEDS: tuple[int, ...] = (42, 123, 456)
SUBPILOT_EPISODES: int = 300
FINAL_WINDOW: int = 20

N_EPISODES_DEFAULT: int = 1000
EVAL_EVERY_DEFAULT: int = 10
EVAL_EPISODES: int = 50
LEARNING_RATE: float = 0.1
EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_FRAC: float = 0.70

# Safe-target target-network cadence.  Polyak is preferable on short
# horizons with few updates per episode (hard sync at sync_every=200 steps
# would fire ~40 times over a 1000-ep horizon-8 run; Polyak at tau=0.05
# averages continuously).
POLYAK_TAU: float = 0.05
SYNC_EVERY: int = 200

PILOT_CFG: dict[str, Any] = {"n_episodes": 30, "eps_greedy": 0.1}


# ---------------------------------------------------------------------------
# Seeding / shortlist / schedule helpers
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    """Fresh-seed every RNG that an arm could touch."""
    seed_everything(int(seed))
    try:
        import torch  # noqa: WPS433 -- optional dep
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _load_tasks_from_shortlist(
    shortlist_path: Path, task_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Resolve ``A_XXX`` task ids to (psi, lam, horizon, gamma, ...).

    Task id convention: ``{family}_{idx:03d}`` with ``idx`` the positional
    index within the family-filtered shortlist (matches
    ``run_phase_V_planning.run_planning``).
    """
    df = pd.read_csv(shortlist_path)
    if df.empty:
        raise ValueError(f"shortlist at {shortlist_path} is empty")
    want: list[dict[str, Any]] = []
    for task_id in task_ids:
        if "_" not in task_id:
            raise ValueError(f"Unrecognised task id format: {task_id!r}")
        family, idx_str = task_id.split("_", 1)
        idx = int(idx_str)
        fam_rows = df[df["family"] == family].reset_index(drop=True)
        if idx >= len(fam_rows):
            raise ValueError(
                f"task_id {task_id}: index {idx} out of range "
                f"(have {len(fam_rows)} rows for family {family})"
            )
        row = fam_rows.iloc[idx]
        want.append({
            "task_id": task_id, "family": family,
            "psi": json.loads(str(row["psi_json"])),
            "psi_json": str(row["psi_json"]),
            "lam": float(row["lam"]),
            "horizon": int(row["horizon"]),
            "gamma": float(row["gamma"]),
            "value_gap": float(row["value_gap"]),
            "value_gap_norm": float(row["value_gap_norm"]),
            "policy_disagreement": float(row["policy_disagreement"]),
        })
    return want


def _schedule_dict_from_calibration(
    sched: dict[str, Any], *, gamma: float, task_family: str,
    reward_bound: float, zero_out: bool,
) -> dict[str, Any]:
    """Convert ``_calibrate_schedule`` output into a BetaSchedule dict.

    Recomputes ``kappa_t / Bhat_t / beta_cap_t`` from ``alpha_t`` and the
    supplied ``reward_bound`` via ``build_certification`` so the
    BetaSchedule round-trip consistency check passes (the calibration
    output may have been computed with a slightly different R_max basis).
    """
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
        build_certification,
    )

    alpha_t = np.asarray(sched["alpha_t"], dtype=np.float64)
    beta_used_t = np.asarray(sched["beta_used_t"], dtype=np.float64).copy()
    cert = build_certification(
        alpha_t=alpha_t, R_max=float(reward_bound), gamma=float(gamma),
    )
    beta_cap_t = np.asarray(cert["beta_cap_t"], dtype=np.float64)
    kappa_t = np.asarray(cert["kappa_t"], dtype=np.float64)
    Bhat_t = np.asarray(cert["Bhat_t"], dtype=np.float64)

    if zero_out:
        beta_used_t = np.zeros_like(beta_used_t)
    beta_used_t = np.clip(beta_used_t, -beta_cap_t, beta_cap_t)
    beta_raw_t = beta_used_t.copy()

    T = len(beta_used_t)
    return {
        "task_family": str(task_family), "gamma": float(gamma),
        "sign": int(sched.get("sign_family", 1)),
        "source_phase": "phaseV_rl_pilot",
        "reward_bound": float(reward_bound),
        "alpha_t": alpha_t.tolist(), "kappa_t": kappa_t.tolist(),
        "Bhat_t": Bhat_t.tolist(),
        "beta_raw_t": beta_raw_t.tolist(), "beta_cap_t": beta_cap_t.tolist(),
        "beta_used_t": beta_used_t.tolist(),
        "clip_active_t": [False] * T, "informativeness_t": [0.0] * T,
        "d_target_t": [float(gamma)] * T,
        "calibration_source_path": "", "calibration_hash": "",
        "notes": (
            "Phase V WP5 pilot schedule (zero)" if zero_out
            else "Phase V WP5 pilot schedule (nonlinear deployed)"
        ),
    }


# ---------------------------------------------------------------------------
# Epsilon schedule + greedy helpers + evaluation
# ---------------------------------------------------------------------------


def _epsilon_at(episode: int, n_episodes: int) -> float:
    """Linear decay EPS_START -> EPS_END over first EPS_FRAC of episodes."""
    horizon_ep = max(1, int(EPS_FRAC * n_episodes))
    if episode >= horizon_ep:
        return EPS_END
    frac = float(episode) / float(horizon_ep)
    return EPS_START + frac * (EPS_END - EPS_START)


def _argmax_random_tiebreak(q: np.ndarray, rng: np.random.Generator) -> int:
    m = float(np.max(q))
    ties = np.flatnonzero(q == m)
    return int(rng.choice(ties))


def _eval_stationary(
    mdp: Any, Q: np.ndarray, horizon: int, gamma_eval: float,
    rng: np.random.Generator, n_episodes: int = EVAL_EPISODES,
) -> tuple[float, int]:
    """Greedy rollout using a stationary Q[s,a].  Returns (mean_return, mode_start_action)."""
    returns = np.zeros(n_episodes, dtype=np.float64)
    start_actions = np.zeros(n_episodes, dtype=np.int64)
    for ep in range(n_episodes):
        state, _ = mdp.reset()
        s = int(np.asarray(state).flat[0])
        disc, ep_ret = 1.0, 0.0
        for t in range(horizon):
            a = _argmax_random_tiebreak(Q[s], rng)
            if t == 0:
                start_actions[ep] = a
            next_state, reward, absorbing, _ = mdp.step(np.array([a]))
            ep_ret += disc * float(reward)
            disc *= gamma_eval
            s = int(np.asarray(next_state).flat[0])
            if absorbing:
                break
        returns[ep] = ep_ret
    vals, counts = np.unique(start_actions, return_counts=True)
    return float(np.mean(returns)), int(vals[int(np.argmax(counts))])


def _eval_stageful(
    mdp: Any, agent: Any, horizon: int, gamma_eval: float,
    rng: np.random.Generator, n_episodes: int = EVAL_EPISODES,
) -> tuple[float, int]:
    """Greedy rollout using a stage-indexed Q[t,s,a]."""
    returns = np.zeros(n_episodes, dtype=np.float64)
    start_actions = np.zeros(n_episodes, dtype=np.int64)
    for ep in range(n_episodes):
        state, _ = mdp.reset()
        s = int(np.asarray(state).flat[0])
        disc, ep_ret = 1.0, 0.0
        for t in range(horizon):
            t_safe = min(t, agent.T - 1)
            a = _argmax_random_tiebreak(agent.Q_online[t_safe, s], rng)
            if t == 0:
                start_actions[ep] = a
            next_state, reward, absorbing, _ = mdp.step(np.array([a]))
            ep_ret += disc * float(reward)
            disc *= gamma_eval
            s = int(np.asarray(next_state).flat[0])
            if absorbing:
                break
        returns[ep] = ep_ret
    vals, counts = np.unique(start_actions, return_counts=True)
    return float(np.mean(returns)), int(vals[int(np.argmax(counts))])


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


@dataclass
class EpisodeLog:
    mean_eval_return: float
    final_return: float
    d_t_mean: float
    d_t_p90: float
    clip_fraction: float
    start_action: int


def _train_q_stationary(
    mdp: Any, mdp_eval: Any, *, gamma_learn: float, gamma_eval: float,
    horizon: int, n_episodes: int, eval_every: int,
    rng_train: np.random.Generator, rng_eval: np.random.Generator,
    lambda_coeff: float = 0.0, trace_type: str = "replacing",
) -> tuple[list[EpisodeLog], np.ndarray]:
    """Tabular Q-learning or Watkins Q(lambda) with stationary Q[s,a].

    ``gamma_learn`` is the discount used in the TD target; ``gamma_eval`` is
    the discount used to report episode returns and eval returns (may differ
    for the ``tuned_fixed_gamma`` arm).  ``lambda_coeff == 0`` gives plain
    Q-learning.  Watkins' cut zeroes the trace on exploratory actions.
    """
    n_states = int(mdp.info.observation_space.n)
    n_actions = int(mdp.info.action_space.n)
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    E = np.zeros_like(Q) if lambda_coeff > 0.0 else None
    logs: list[EpisodeLog] = []

    for ep in range(n_episodes):
        if E is not None:
            E.fill(0.0)
        state, _ = mdp.reset()
        s = int(np.asarray(state).flat[0])
        ep_ret, disc = 0.0, 1.0
        eps = _epsilon_at(ep, n_episodes)
        for _t in range(horizon):
            if rng_train.random() < eps:
                a = int(rng_train.integers(0, n_actions))
                explored = True
            else:
                a = _argmax_random_tiebreak(Q[s], rng_train)
                explored = False
            next_state, reward, absorbing, _ = mdp.step(np.array([a]))
            s_next = int(np.asarray(next_state).flat[0])
            r = float(reward)
            ep_ret += disc * r
            disc *= gamma_eval
            q_next = 0.0 if absorbing else float(np.max(Q[s_next]))
            td = r + gamma_learn * q_next - float(Q[s, a])
            if E is None:
                Q[s, a] += LEARNING_RATE * td
            else:
                if trace_type == "replacing":
                    E[s] = 0.0
                    E[s, a] = 1.0
                else:
                    E[s, a] += 1.0
                Q += LEARNING_RATE * td * E
                if explored:
                    E.fill(0.0)
                else:
                    E *= gamma_learn * lambda_coeff
            s = s_next
            if absorbing:
                break
        if (ep + 1) % eval_every == 0:
            mean_ret, start_a = _eval_stationary(
                mdp_eval, Q, horizon, gamma_eval, rng_eval,
            )
            logs.append(EpisodeLog(
                mean_eval_return=mean_ret, final_return=ep_ret,
                d_t_mean=gamma_eval, d_t_p90=gamma_eval,
                clip_fraction=0.0, start_action=start_a,
            ))
    return logs, Q


def _train_safe_target(
    mdp: Any, mdp_eval: Any, *, gamma: float, horizon: int, schedule: Any,
    n_episodes: int, eval_every: int,
    rng_train: np.random.Generator, rng_eval: np.random.Generator,
    agent_seed: int,
) -> tuple[list[EpisodeLog], Any, dict[str, Any]]:
    """SafeTargetQLearning loop; stage is the in-episode step index."""
    from lse_rl.algorithms import SafeTargetQLearning

    n_states = int(mdp.info.observation_space.n)
    n_actions = int(mdp.info.action_space.n)
    agent = SafeTargetQLearning(
        n_states=n_states, n_actions=n_actions, schedule=schedule,
        learning_rate=LEARNING_RATE, gamma=gamma,
        sync_every=SYNC_EVERY, polyak_tau=POLYAK_TAU, seed=agent_seed,
    )
    logs: list[EpisodeLog] = []
    global_step = 0
    n_clip_updates = 0

    for ep in range(n_episodes):
        state, _ = mdp.reset()
        s = int(np.asarray(state).flat[0])
        eps = _epsilon_at(ep, n_episodes)
        ep_ret, disc = 0.0, 1.0
        d_buf: list[float] = []
        c_buf: list[int] = []
        for t in range(horizon):
            t_safe = min(t, agent.T - 1)
            if rng_train.random() < eps:
                a = int(rng_train.integers(0, n_actions))
            else:
                a = _argmax_random_tiebreak(agent.Q_online[t_safe, s], rng_train)
            next_state, reward, absorbing, _ = mdp.step(np.array([a]))
            s_next = int(np.asarray(next_state).flat[0])
            r = float(reward)
            ep_ret += disc * r
            disc *= gamma
            global_step += 1
            log = agent.update(
                state=s, action=a, reward=r, next_state=s_next,
                absorbing=bool(absorbing), stage=t_safe, global_step=global_step,
            )
            d_buf.append(float(log["effective_discount"]))
            clip_flag = bool(log["clip_active"])
            c_buf.append(1 if clip_flag else 0)
            if clip_flag:
                n_clip_updates += 1
            s = s_next
            if absorbing:
                break
        if (ep + 1) % eval_every == 0:
            mean_ret, start_a = _eval_stageful(
                mdp_eval, agent, horizon, gamma, rng_eval,
            )
            d_arr = (np.asarray(d_buf, dtype=np.float64) if d_buf
                     else np.asarray([gamma], dtype=np.float64))
            c_arr = (np.asarray(c_buf, dtype=np.float64) if c_buf
                     else np.asarray([0.0], dtype=np.float64))
            logs.append(EpisodeLog(
                mean_eval_return=mean_ret, final_return=ep_ret,
                d_t_mean=float(np.mean(d_arr)),
                d_t_p90=float(np.percentile(d_arr, 90)),
                clip_fraction=float(np.mean(c_arr)),
                start_action=start_a,
            ))
    return logs, agent, {"n_clip_updates": int(n_clip_updates)}


# ---------------------------------------------------------------------------
# Sub-pilots
# ---------------------------------------------------------------------------


def _subpilot_best_param(
    mdp: Any, mdp_eval: Any, *, gamma_native: float, horizon: int,
    grid: tuple[float, ...], seeds: tuple[int, ...],
    n_episodes: int, eval_every: int, mode: str,
) -> tuple[float, dict[float, float]]:
    """Grid-sweep a tuning param (gamma_eff or lambda); pick max final-20 mean."""
    scores: dict[float, float] = {}
    for p in grid:
        per_seed: list[float] = []
        for s in seeds:
            _seed_all(int(s))
            rng_train = np.random.default_rng(int(s))
            rng_eval = np.random.default_rng(int(s) + 10_000)
            if mode == "gamma_eff":
                logs, _ = _train_q_stationary(
                    mdp, mdp_eval, gamma_learn=float(p), gamma_eval=gamma_native,
                    horizon=horizon, n_episodes=n_episodes, eval_every=eval_every,
                    rng_train=rng_train, rng_eval=rng_eval, lambda_coeff=0.0,
                )
            elif mode == "lambda":
                logs, _ = _train_q_stationary(
                    mdp, mdp_eval, gamma_learn=gamma_native, gamma_eval=gamma_native,
                    horizon=horizon, n_episodes=n_episodes, eval_every=eval_every,
                    rng_train=rng_train, rng_eval=rng_eval,
                    lambda_coeff=float(p), trace_type="replacing",
                )
            else:
                raise ValueError(f"unknown sub-pilot mode: {mode}")
            tail = [float(lg.mean_eval_return) for lg in logs[-FINAL_WINDOW:]]
            per_seed.append(float(np.mean(tail)) if tail else float("nan"))
        scores[float(p)] = float(np.mean(per_seed))
    best = max(scores, key=lambda k: scores[k])
    return float(best), scores


# ---------------------------------------------------------------------------
# Paired-difference statistics
# ---------------------------------------------------------------------------


def _bca_mean_ci(
    diffs: np.ndarray, *, n_resamples: int = 10_000, seed: int = 0,
) -> tuple[float, float, float]:
    """BCa 95% CI for the mean of paired diffs (fallback: percentile)."""
    arr = np.asarray(diffs, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean_d = float(np.mean(arr))
    if arr.size < 2 or float(np.std(arr)) == 0.0:
        return mean_d, mean_d, mean_d
    try:
        from scipy.stats import bootstrap
        res = bootstrap(
            (arr,), np.mean,
            n_resamples=int(n_resamples), confidence_level=0.95,
            method="BCa", random_state=int(seed),
        )
        return (mean_d,
                float(res.confidence_interval.low),
                float(res.confidence_interval.high))
    except Exception:  # pragma: no cover
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
        means = arr[idx].mean(axis=1)
        return (mean_d,
                float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5)))


def _hedges_g(diffs: np.ndarray) -> float:
    """Hedges' g (small-sample-corrected one-sample effect size)."""
    arr = np.asarray(diffs, dtype=np.float64).ravel()
    n = arr.size
    if n < 2:
        return float("nan")
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return float("nan")
    J = 1.0 - 3.0 / (4.0 * n - 5.0)
    return float(J * float(np.mean(arr)) / sd)


# ---------------------------------------------------------------------------
# Single-arm dispatch
# ---------------------------------------------------------------------------


def _run_single_arm(
    *, arm: str, mdp: Any, mdp_eval: Any, gamma_native: float, horizon: int,
    seed: int, n_episodes: int, eval_every: int,
    sched_nonlinear: Any | None, sched_zero: Any | None,
    gamma_eff_chosen: float | None, lambda_chosen: float | None,
) -> tuple[list[EpisodeLog], dict[str, Any]]:
    _seed_all(int(seed))
    rng_t = np.random.default_rng(int(seed))
    rng_e = np.random.default_rng(int(seed) + 10_000)
    kw_stat = dict(
        horizon=horizon, n_episodes=n_episodes, eval_every=eval_every,
        rng_train=rng_t, rng_eval=rng_e,
    )
    if arm == "classical_q":
        logs, _ = _train_q_stationary(
            mdp, mdp_eval, gamma_learn=gamma_native, gamma_eval=gamma_native,
            lambda_coeff=0.0, **kw_stat,
        )
        return logs, {"n_clip_updates": 0}
    if arm == "tuned_fixed_gamma":
        if gamma_eff_chosen is None:
            raise ValueError("tuned_fixed_gamma requires gamma_eff_chosen")
        logs, _ = _train_q_stationary(
            mdp, mdp_eval, gamma_learn=float(gamma_eff_chosen),
            gamma_eval=gamma_native, lambda_coeff=0.0, **kw_stat,
        )
        return logs, {"n_clip_updates": 0}
    if arm == "td_lambda":
        if lambda_chosen is None:
            raise ValueError("td_lambda requires lambda_chosen")
        logs, _ = _train_q_stationary(
            mdp, mdp_eval, gamma_learn=gamma_native, gamma_eval=gamma_native,
            lambda_coeff=float(lambda_chosen), trace_type="replacing",
            **kw_stat,
        )
        return logs, {"n_clip_updates": 0}
    if arm in ("safe_zero", "safe_nonlinear"):
        sched = sched_zero if arm == "safe_zero" else sched_nonlinear
        if sched is None:
            raise ValueError(f"{arm} requires a schedule")
        logs, _, aux = _train_safe_target(
            mdp, mdp_eval, gamma=gamma_native, horizon=horizon, schedule=sched,
            n_episodes=n_episodes, eval_every=eval_every,
            rng_train=rng_t, rng_eval=rng_e, agent_seed=int(seed),
        )
        return logs, aux
    raise ValueError(f"unknown arm: {arm}")


# ---------------------------------------------------------------------------
# Pilot driver
# ---------------------------------------------------------------------------


def run_pilot(
    *, shortlist_path: Path, output_root: Path,
    task_ids: Iterable[str] = DEFAULT_TASKS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    n_episodes: int = N_EPISODES_DEFAULT, eval_every: int = EVAL_EVERY_DEFAULT,
    exact_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the Stage-1 paired RL pilot; write outputs; return summary."""
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule

    t_start = time.time()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    schedules_root = output_root / "schedules"
    schedules_root.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks_from_shortlist(shortlist_path, list(task_ids))
    seed_list = [int(s) for s in seeds]

    # --- Sub-pilots: tune gamma_eff and lambda once per task ----------------
    gamma_eff_per_task: dict[str, float] = {}
    lambda_per_task: dict[str, float] = {}
    for task in tasks:
        tid = task["task_id"]
        mdp = family_a.build_mdp(task["lam"], task["psi"])
        mdp_eval = copy.deepcopy(mdp)
        gamma_native = float(task["gamma"])
        horizon = int(mdp.info.horizon)
        logger.info("[subpilot] %s gamma_eff sweep", tid)
        g_best, g_scores = _subpilot_best_param(
            mdp, mdp_eval, gamma_native=gamma_native, horizon=horizon,
            grid=GAMMA_GRID, seeds=SUBPILOT_SEEDS,
            n_episodes=SUBPILOT_EPISODES, eval_every=eval_every,
            mode="gamma_eff",
        )
        gamma_eff_per_task[tid] = float(g_best)
        logger.info("[subpilot] %s gamma_eff=%.3f scores=%s",
                    tid, g_best, g_scores)
        logger.info("[subpilot] %s lambda sweep", tid)
        l_best, l_scores = _subpilot_best_param(
            mdp, mdp_eval, gamma_native=gamma_native, horizon=horizon,
            grid=LAMBDA_GRID, seeds=SUBPILOT_SEEDS,
            n_episodes=SUBPILOT_EPISODES, eval_every=eval_every, mode="lambda",
        )
        lambda_per_task[tid] = float(l_best)
        logger.info("[subpilot] %s lambda=%.3f scores=%s",
                    tid, l_best, l_scores)

    # --- Main pilot runs ----------------------------------------------------
    per_ep_rows: list[dict[str, Any]] = []
    per_sc_rows: list[dict[str, Any]] = []
    schedule_payloads: dict[str, dict[str, Any]] = {}

    for task in tasks:
        tid = task["task_id"]
        gamma_native = float(task["gamma"])
        # V*_safe threshold per dispatch: shortlist reports value_gap and
        # value_gap_norm (= value_gap / reward_scale); V*_classical ~
        # reward_scale; V*_safe = value_gap + V*_classical.
        reward_scale = (float(task["value_gap"] / task["value_gap_norm"])
                        if abs(task["value_gap_norm"]) > 0 else 1.0)
        vstar_safe = float(task["value_gap"]) + reward_scale
        threshold = 0.8 * vstar_safe

        for seed in seed_list:
            t0 = time.time()
            logger.info("[run] task=%s seed=%d", tid, seed)
            mdp = family_a.build_mdp(task["lam"], task["psi"])
            mdp_eval = copy.deepcopy(mdp)
            horizon = int(mdp.info.horizon)

            # Calibrate ONCE per (task, seed) BEFORE any arm runs.
            sched_dict_raw = _calibrate_schedule(
                mdp, family_label="A", pilot_cfg=PILOT_CFG, seed=int(seed),
            )
            reward_bound = (float(np.max(np.abs(np.asarray(mdp.r, dtype=np.float64))))
                            if mdp.r.size else 1.0)
            nonlinear_payload = _schedule_dict_from_calibration(
                sched_dict_raw, gamma=gamma_native,
                task_family=f"phaseV_family_A_{tid}",
                reward_bound=reward_bound, zero_out=False,
            )
            zero_payload = _schedule_dict_from_calibration(
                sched_dict_raw, gamma=gamma_native,
                task_family=f"phaseV_family_A_{tid}_zero",
                reward_bound=reward_bound, zero_out=True,
            )
            sched_nonlinear = BetaSchedule(nonlinear_payload)
            sched_zero = BetaSchedule(zero_payload)

            payload_path = schedules_root / tid / f"seed_{seed}_nonlinear.json"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            with open(payload_path, "w") as f:
                json.dump(nonlinear_payload, f, indent=2, sort_keys=True)
            schedule_payloads[f"{tid}_seed_{seed}"] = {
                "beta_used_t": nonlinear_payload["beta_used_t"],
                "beta_cap_t": nonlinear_payload["beta_cap_t"],
                "beta_raw_t": nonlinear_payload["beta_raw_t"],
                "alpha_t": nonlinear_payload["alpha_t"],
                "kappa_t": nonlinear_payload["kappa_t"],
                "path": str(payload_path),
            }

            for arm in ARMS:
                logs, aux = _run_single_arm(
                    arm=arm, mdp=mdp, mdp_eval=mdp_eval,
                    gamma_native=gamma_native, horizon=horizon,
                    seed=int(seed), n_episodes=n_episodes, eval_every=eval_every,
                    sched_nonlinear=sched_nonlinear, sched_zero=sched_zero,
                    gamma_eff_chosen=gamma_eff_per_task.get(tid),
                    lambda_chosen=lambda_per_task.get(tid),
                )
                for ep_idx, log in enumerate(logs):
                    per_ep_rows.append({
                        "task_id": tid, "seed": int(seed), "arm": arm,
                        "episode": (ep_idx + 1) * eval_every,
                        "mean_eval_return": float(log.mean_eval_return),
                        "final_return": float(log.final_return),
                        "d_t_mean": float(log.d_t_mean),
                        "d_t_p90": float(log.d_t_p90),
                        "clip_fraction": float(log.clip_fraction),
                        "start_action": int(log.start_action),
                    })
                eval_series = np.asarray(
                    [log.mean_eval_return for log in logs], dtype=np.float64,
                )
                window = (eval_series[-FINAL_WINDOW:]
                          if eval_series.size >= FINAL_WINDOW else eval_series)
                final_mean = float(np.mean(window)) if window.size else float("nan")
                final_std = (float(np.std(window, ddof=1))
                             if window.size >= 2 else 0.0)
                ep_axis = (np.arange(1, eval_series.size + 1, dtype=np.float64)
                           * float(eval_every))
                # np.trapz was removed in numpy 2.0; use np.trapezoid.
                _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
                auc = (float(_trap(eval_series, x=ep_axis))
                       if (eval_series.size >= 2 and _trap is not None) else 0.0)
                ttt = float("nan")
                hits = np.where(eval_series >= threshold)[0]
                if hits.size:
                    ttt = float((hits[0] + 1) * eval_every)
                per_sc_rows.append({
                    "task_id": tid, "seed": int(seed), "arm": arm,
                    "final_return_mean": final_mean, "final_return_std": final_std,
                    "auc_return": auc, "time_to_threshold": ttt,
                    "n_clip_updates": int(aux.get("n_clip_updates", 0)),
                    "gamma_eff_chosen": (
                        float(gamma_eff_per_task[tid])
                        if arm == "tuned_fixed_gamma" else float("nan")
                    ),
                    "lambda_chosen": (
                        float(lambda_per_task[tid])
                        if arm == "td_lambda" else float("nan")
                    ),
                    "vstar_safe_threshold": float(threshold),
                })
            logger.info("[run] task=%s seed=%d elapsed=%.1fs",
                        tid, seed, time.time() - t0)

    # --- Persist outputs ----------------------------------------------------
    per_ep_df = pd.DataFrame(per_ep_rows)
    per_sc_df = pd.DataFrame(per_sc_rows)
    per_ep_path = output_root / "pilot_runs.parquet"
    per_sc_path = output_root / "pilot_runs_scalars.parquet"
    per_ep_df.to_parquet(per_ep_path, index=False)
    per_sc_df.to_parquet(per_sc_path, index=False)

    # --- Paired differences -------------------------------------------------
    pairs: tuple[tuple[str, str], ...] = (
        ("safe_nonlinear", "classical_q"),
        ("safe_nonlinear", "safe_zero"),
        ("safe_nonlinear", "tuned_fixed_gamma"),
        ("safe_nonlinear", "td_lambda"),
    )
    paired_rows: list[dict[str, Any]] = []
    for task in tasks:
        tid = task["task_id"]
        sub = per_sc_df[per_sc_df["task_id"] == tid]
        for arm_a, arm_b in pairs:
            auc_a = sub[sub["arm"] == arm_a].set_index("seed")["auc_return"]
            auc_b = sub[sub["arm"] == arm_b].set_index("seed")["auc_return"]
            common = sorted(set(auc_a.index) & set(auc_b.index))
            diffs = np.asarray(
                [float(auc_a[s] - auc_b[s]) for s in common], dtype=np.float64,
            )
            mean_d, lo, hi = _bca_mean_ci(diffs, n_resamples=10_000, seed=0)
            paired_rows.append({
                "task_id": tid, "arm_a": arm_a, "arm_b": arm_b,
                "mean_diff": float(mean_d),
                "ci_low": float(lo), "ci_high": float(hi),
                "hedges_g": float(_hedges_g(diffs)),
                "n_seeds": int(diffs.size),
            })
    paired_df = pd.DataFrame(paired_rows)
    paired_path = output_root / "paired_differences.parquet"
    paired_df.to_parquet(paired_path, index=False)

    manifest_path = _write_manifest(
        shortlist_path=shortlist_path, output_root=output_root,
        tasks=[t["task_id"] for t in tasks], seeds=seed_list, arms=list(ARMS),
        gamma_eff_per_task=gamma_eff_per_task, lambda_per_task=lambda_per_task,
        schedule_payloads=schedule_payloads,
        exact_argv=list(exact_argv) if exact_argv is not None else [],
        runtime_sec=float(time.time() - t_start),
        per_ep_path=per_ep_path, per_sc_path=per_sc_path,
        paired_path=paired_path,
    )
    logger.info("[pilot] DONE in %.1fs manifest=%s",
                float(time.time() - t_start), manifest_path)
    return {
        "pilot_runs_path": str(per_ep_path),
        "pilot_runs_scalars_path": str(per_sc_path),
        "paired_differences_path": str(paired_path),
        "manifest_path": str(manifest_path),
        "gamma_eff_per_task": gamma_eff_per_task,
        "lambda_per_task": lambda_per_task,
        "runtime_sec": float(time.time() - t_start),
    }


# ---------------------------------------------------------------------------
# Manifest writer + CLI
# ---------------------------------------------------------------------------


def _write_manifest(
    *, shortlist_path: Path, output_root: Path,
    tasks: list[str], seeds: list[int], arms: list[str],
    gamma_eff_per_task: dict[str, float], lambda_per_task: dict[str, float],
    schedule_payloads: dict[str, dict[str, Any]],
    exact_argv: list[str], runtime_sec: float,
    per_ep_path: Path, per_sc_path: Path, paired_path: Path,
) -> Path:
    """Read-modify-write ``results/summaries/experiment_manifest.json``."""
    manifest_dir = _REPO_ROOT / "results" / "summaries"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "experiment_manifest.json"
    payload: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    payload.setdefault("schema_version", SCHEMA_VERSION)
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    payload["wp5_pilot_runs"] = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(), "host": host,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exact_argv": list(exact_argv),
        "shortlist_path": str(shortlist_path),
        "tasks": list(tasks), "seeds": list(seeds), "arms": list(arms),
        "gamma_eff_per_task": gamma_eff_per_task,
        "lambda_per_task": lambda_per_task,
        "schedule_per_task": schedule_payloads,
        "runtime_sec": float(runtime_sec),
        "output_paths": {
            "pilot_runs": str(per_ep_path),
            "pilot_runs_scalars": str(per_sc_path),
            "paired_differences": str(paired_path),
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
    return manifest_path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"unserialisable: {type(obj).__name__}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_phase_V_rl_pilot", description=__doc__,
    )
    p.add_argument("--shortlist", type=Path, required=True)
    p.add_argument("--tasks", type=str, nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--output-root", type=Path, default=Path("results/rl/"))
    p.add_argument("--n-episodes", type=int, default=N_EPISODES_DEFAULT)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY_DEFAULT)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    summary = run_pilot(
        shortlist_path=args.shortlist, output_root=args.output_root,
        task_ids=args.tasks, seeds=args.seeds,
        n_episodes=int(args.n_episodes), eval_every=int(args.eval_every),
        exact_argv=(argv if argv is not None else sys.argv),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
