"""Phase VI-D — RL retry with DP-Q initialization.

Hypothesis
----------
WP5 Stage-1 found identical AUC across all RL arms because tabular
Q-learning noise (~O(R_max * learning_rate) per update) exceeds the
safe-vs-classical DP Q-gap at s0 (1e-3 to 2e-2).  Under this hypothesis,
initializing the Q-table from the exact DP fixed point and fine-tuning
briefly at a small learning rate should preserve the DP-level start-state
argmax pattern.

Test
----
For each Family A task (the two WP5 Stage-1 tasks + the VI-B stronger
shortlist):

1. Compute exact DP Q-table for both classical and safe operators.
2. Initialize RL Q-table from each operator's DP optimum.
3. Fine-tune with on-policy Q-learning: 200 episodes, learning rate
   alpha=0.01 (10x smaller than WP5 baseline).
4. Evaluate every 10 episodes; record start-state argmax over 50 greedy
   rollouts.
5. Paired CRN across arms per (task, seed).

Expected outcome
----------------
If the DP-vs-RL-noise hypothesis holds, the classical arm preserves the
classical DP argmax at s0 and the safe arm preserves the safe DP argmax.
Paired AUC differences become non-zero in exact proportion to the DP
value gap.

If paired AUC differences remain zero even after DP-init, the RL null
is deeper than sample noise — likely an on-policy fine-tuning bias that
drives both arms toward the same local attractor.

CLI
---
.. code-block:: text

    python -m experiments.weighted_lse_dp.runners.run_phase_VI_rl_dp_init \\
      --shortlist results/search/shortlist_VI_B_A.csv \\
      --seeds 42 123 456 789 1024 \\
      --output-root results/rl_VI_D/ \\
      [--n-episodes 200] [--learning-rate 0.01]
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _REPO_ROOT / "mushroom-rl-dev"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    _classical_qv,
    _extract_tensors,
    _safe_qv,
)
from experiments.weighted_lse_dp.tasks.family_a_jackpot_vs_stream import (  # noqa: E402
    family_a,
)
from experiments.weighted_lse_dp.runners.run_phase_V_search import (  # noqa: E402
    _calibrate_schedule,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("phaseVI.rl_dp_init")


def _rollout_eval_return(
    Q: np.ndarray,               # (T, S, A) greedy Q-table
    P: np.ndarray,               # (S, A, S)
    R: np.ndarray,               # (S, A, S)
    gamma: float,
    T: int,
    s0: int,
    *,
    n_rollouts: int,
    rng: np.random.Generator,
) -> float:
    """Expected discounted return under the greedy policy from s0."""
    returns = np.zeros(n_rollouts, dtype=np.float64)
    for i in range(n_rollouts):
        s = s0
        disc = 1.0
        G = 0.0
        for t in range(T):
            a = int(np.argmax(Q[t, s]))
            # Sample next state from P[s, a, :].
            probs = P[s, a]
            if probs.sum() == 0.0:
                break  # absorbing
            s_next = int(rng.choice(len(probs), p=probs / probs.sum()))
            G += disc * float(R[s, a, s_next])
            disc *= gamma
            s = s_next
        returns[i] = G
    return float(returns.mean())


def _q_learning_finetune(
    Q_init: np.ndarray,          # (T, S, A) initial Q
    P: np.ndarray,               # (S, A, S)
    R: np.ndarray,               # (S, A, S)
    gamma: float,
    T: int,
    s0: int,
    *,
    n_episodes: int,
    alpha: float,
    eps: float,
    eval_every: int,
    n_eval_rollouts: int,
    rng_train: np.random.Generator,
    rng_eval: np.random.Generator,
) -> list[dict[str, Any]]:
    """Fine-tune Q starting from Q_init via eps-greedy Q-learning.

    Returns a list of per-eval-episode dicts with ``episode,
    mean_eval_return, start_action``.  start_action is argmax_a Q[0, s0, a].
    """
    Q = Q_init.copy()
    rows: list[dict[str, Any]] = []
    for ep in range(n_episodes):
        s = s0
        for t in range(T):
            # eps-greedy action.
            if rng_train.random() < eps:
                a = int(rng_train.integers(0, Q.shape[2]))
            else:
                a = int(np.argmax(Q[t, s]))
            probs = P[s, a]
            if probs.sum() == 0.0:
                break  # absorbing
            s_next = int(rng_train.choice(len(probs), p=probs / probs.sum()))
            r_sa = float(R[s, a, s_next])
            if t < T - 1:
                td_target = r_sa + gamma * float(np.max(Q[t + 1, s_next]))
            else:
                td_target = r_sa  # terminal
            Q[t, s, a] += alpha * (td_target - Q[t, s, a])
            s = s_next
        if (ep + 1) % eval_every == 0 or ep == 0:
            mean_ret = _rollout_eval_return(
                Q, P, R, gamma, T, s0,
                n_rollouts=n_eval_rollouts, rng=rng_eval,
            )
            start_action = int(np.argmax(Q[0, s0]))
            rows.append({
                "episode": int(ep + 1),
                "mean_eval_return": float(mean_ret),
                "start_action": start_action,
            })
    return rows


def run(
    shortlist_path: pathlib.Path,
    output_root: pathlib.Path,
    seeds: list[int],
    *,
    n_episodes: int,
    alpha: float,
    eps: float,
    eval_every: int,
    n_eval_rollouts: int,
) -> dict[str, Any]:
    shortlist = pd.read_csv(shortlist_path)
    A = shortlist[shortlist["family"] == "A"].reset_index(drop=True)
    logger.info("Phase VI-D: %d Family A tasks, %d seeds", len(A), len(seeds))

    all_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, row in A.iterrows():
        psi = json.loads(row["psi_json"])
        lam = float(row["lam"])
        mdp = family_a.build_mdp(lam, psi)
        P, R, r_bar, gamma, T, S, A_sz = _extract_tensors(mdp)
        s0 = int(getattr(mdp, "initial_state", 0))

        # Compute DP optima.
        Q_cl, V_cl = _classical_qv(P, r_bar, gamma, T)
        for seed in seeds:
            sched = _calibrate_schedule(
                mdp, family_label="A",
                pilot_cfg={"n_episodes": 30, "eps_greedy": 0.1},
                seed=int(seed),
            )
            beta_used_t = np.asarray(sched["beta_used_t"], dtype=np.float64)
            Q_safe, V_safe = _safe_qv(P, R, gamma, T, beta_used_t)

            task_id = f"A_{i:03d}"
            for arm, Q_init in [
                ("classical_dp_init", Q_cl),
                ("safe_nonlinear_dp_init", Q_safe),
            ]:
                rng_train = np.random.default_rng(int(seed) * 1000 + hash(arm) % 1000)
                rng_eval = np.random.default_rng(int(seed) * 7919 + hash(arm) % 7919)
                fine_rows = _q_learning_finetune(
                    Q_init, P, R, gamma, T, s0,
                    n_episodes=n_episodes, alpha=alpha, eps=eps,
                    eval_every=eval_every, n_eval_rollouts=n_eval_rollouts,
                    rng_train=rng_train, rng_eval=rng_eval,
                )
                for r in fine_rows:
                    r.update({
                        "task_id": task_id,
                        "psi_json": json.dumps(psi, default=str, sort_keys=True),
                        "lam": lam,
                        "seed": int(seed),
                        "arm": arm,
                        "v_star_cl_s0": float(V_cl[0, s0]),
                        "v_star_safe_s0": float(V_safe[0, s0]),
                        "value_gap_dp": float(V_safe[0, s0] - V_cl[0, s0]),
                    })
                all_rows.extend(fine_rows)

    elapsed = time.time() - t0

    df = pd.DataFrame(all_rows)
    out_root = pathlib.Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    runs_path = out_root / "dp_init_runs.parquet"
    df.to_parquet(runs_path, index=False)

    # Compute paired AUC differences.
    def _trap(x: pd.Series, y: pd.Series) -> float:
        import numpy as np
        return float(np.trapezoid(y.values, x.values))

    auc_records: list[dict[str, Any]] = []
    for (task, seed), group in df.groupby(["task_id", "seed"]):
        arm_aucs = {}
        for arm, sub in group.groupby("arm"):
            sub = sub.sort_values("episode")
            arm_aucs[arm] = _trap(sub["episode"], sub["mean_eval_return"])
        auc_records.append({
            "task_id": task,
            "seed": seed,
            "classical_auc": arm_aucs.get("classical_dp_init", float("nan")),
            "safe_auc": arm_aucs.get("safe_nonlinear_dp_init", float("nan")),
            "auc_diff": arm_aucs.get("safe_nonlinear_dp_init", float("nan"))
                - arm_aucs.get("classical_dp_init", float("nan")),
            "value_gap_dp": float(group["value_gap_dp"].iloc[0]),
        })
    auc_df = pd.DataFrame(auc_records)
    auc_path = out_root / "dp_init_auc.parquet"
    auc_df.to_parquet(auc_path, index=False)

    # Start-action preservation: fraction of evals where start_action
    # matches the DP argmax for each arm.
    preserve: list[dict[str, Any]] = []
    for (task, seed, arm), group in df.groupby(["task_id", "seed", "arm"]):
        # DP argmax for this arm.
        sub0 = group.iloc[0]
        psi = json.loads(sub0["psi_json"])
        lam = float(sub0["lam"])
        mdp = family_a.build_mdp(lam, psi)
        P, R, r_bar, gamma, T, S, A_sz = _extract_tensors(mdp)
        if arm.startswith("classical"):
            Q_dp, _ = _classical_qv(P, r_bar, gamma, T)
        else:
            sched = _calibrate_schedule(
                mdp, family_label="A",
                pilot_cfg={"n_episodes": 30, "eps_greedy": 0.1}, seed=int(seed),
            )
            beta_used_t = np.asarray(sched["beta_used_t"], dtype=np.float64)
            Q_dp, _ = _safe_qv(P, R, gamma, T, beta_used_t)
        s0 = int(getattr(mdp, "initial_state", 0))
        dp_action = int(np.argmax(Q_dp[0, s0]))
        final_action_matches = int(group.iloc[-1]["start_action"] == dp_action)
        preserve.append({
            "task_id": task,
            "seed": seed,
            "arm": arm,
            "dp_start_action": dp_action,
            "final_start_action": int(group.iloc[-1]["start_action"]),
            "preserved": final_action_matches,
        })
    preserve_df = pd.DataFrame(preserve)
    preserve_path = out_root / "dp_init_preservation.parquet"
    preserve_df.to_parquet(preserve_path, index=False)

    # Summary
    n_tasks = auc_df["task_id"].nunique()
    mean_auc_diff = float(auc_df["auc_diff"].mean())
    preserve_rate = float(preserve_df["preserved"].mean())
    preserve_by_arm = preserve_df.groupby("arm")["preserved"].mean().round(3).to_dict()
    logger.info(
        "Phase VI-D complete: n_tasks=%d seeds=%d mean_auc_diff=%.6f "
        "preserve_rate=%.2f per_arm=%s elapsed=%.2fs",
        n_tasks, len(seeds), mean_auc_diff,
        preserve_rate, preserve_by_arm, elapsed,
    )
    return {
        "n_tasks": n_tasks,
        "seeds": seeds,
        "mean_auc_diff": mean_auc_diff,
        "preserve_rate": preserve_rate,
        "preserve_by_arm": preserve_by_arm,
        "elapsed_sec": elapsed,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shortlist", type=pathlib.Path, required=True)
    p.add_argument("--output-root", type=pathlib.Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1024])
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--learning-rate", type=float, dest="alpha", default=0.01)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--n-eval-rollouts", type=int, default=50)
    return p.parse_args()


def _main() -> int:
    args = _parse_args()
    run(
        shortlist_path=args.shortlist,
        output_root=args.output_root,
        seeds=args.seeds,
        n_episodes=args.n_episodes,
        alpha=args.alpha,
        eps=args.eps,
        eval_every=args.eval_every,
        n_eval_rollouts=args.n_eval_rollouts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
