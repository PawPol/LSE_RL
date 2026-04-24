"""Phase VI-G — risk-sensitive policy evaluation.

Enumerates return distributions for pi*_classical vs pi*_safe on the
stochastic Family A shortlist via Monte-Carlo rollouts. Computes:

- mean return  E[G]                   (sanity; should match dual_eval V_cl)
- variance     Var[G]
- CVaR_alpha   E[G | G <= q_alpha(G)] for alpha in {0.05, 0.10, 0.25}
- exp utility  (1/beta) log E[exp(beta G)]  (entropic risk)

The safe weighted-LSE operator is the fixed point of a targeted local
entropic-risk correction.  The prediction is that the safe policy
dominates the classical policy on entropic risk / CVaR -- a claim the
paper can back with this runner's exact enumeration (up to Monte-Carlo
error).
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
logger = logging.getLogger("phaseVI.risk_sensitive")


def _rollout_batch(
    pi: np.ndarray,                  # (T, S) int
    P: np.ndarray,                   # (S, A, S)
    R: np.ndarray,                   # (S, A, S)
    gamma: float,
    T: int,
    s0: int,
    *,
    n_rollouts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_rollouts returns; shape (n_rollouts,)."""
    out = np.zeros(n_rollouts, dtype=np.float64)
    for i in range(n_rollouts):
        s = s0
        disc = 1.0
        G = 0.0
        for t in range(T):
            a = int(pi[t, s])
            probs = P[s, a]
            psum = probs.sum()
            if psum == 0.0:
                break  # absorbing
            s_next = int(rng.choice(len(probs), p=probs / psum))
            G += disc * float(R[s, a, s_next])
            disc *= gamma
            s = s_next
        out[i] = G
    return out


def _cvar(returns: np.ndarray, alpha: float) -> float:
    """CVaR at level alpha: mean of the worst alpha-fraction of returns."""
    q = float(np.quantile(returns, alpha))
    tail = returns[returns <= q]
    if tail.size == 0:
        return q
    return float(tail.mean())


def _exp_utility(returns: np.ndarray, beta: float) -> float:
    """Entropic risk: (1/beta) log E[exp(beta * G)].

    For beta > 0, rewards higher E[exp(beta G)] (risk-seeking wrt
    gains).  We use the risk-AVERSE variant: -E[-(1/beta) log E[exp(-beta G)]],
    i.e. beta > 0 and we compute -(1/beta) log E[exp(-beta G)] which
    penalises low returns.  Higher = better.
    """
    # Numerically stable: max -beta G, use log-sum-exp
    if beta <= 0:
        return float(returns.mean())
    m = float(np.max(-beta * returns))
    s = float(np.log(np.mean(np.exp(-beta * returns - m))) + m)
    return float(-(1.0 / beta) * s)


def run(
    shortlist_path: pathlib.Path,
    output_root: pathlib.Path,
    *,
    n_rollouts: int,
    alphas: list[float],
    betas: list[float],
    seed: int,
    pilot_seed: int,
) -> dict[str, Any]:
    shortlist = pd.read_csv(shortlist_path)
    A = shortlist[shortlist["family"] == "A"].reset_index(drop=True)
    logger.info("Phase VI-G: %d Family A tasks; %d rollouts per policy per task",
                len(A), n_rollouts)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    rng = np.random.default_rng(int(seed))

    for i, row in A.iterrows():
        psi = json.loads(row["psi_json"])
        lam = float(row["lam"])
        mdp = family_a.build_mdp(lam, psi)
        P, R, r_bar, gamma, T, S, A_sz = _extract_tensors(mdp)
        s0 = int(getattr(mdp, "initial_state", 0))

        Q_cl, _ = _classical_qv(P, r_bar, gamma, T)
        sched = _calibrate_schedule(
            mdp, family_label="A",
            pilot_cfg={"n_episodes": 30, "eps_greedy": 0.1},
            seed=int(pilot_seed),
        )
        beta_used_t = np.asarray(sched["beta_used_t"], dtype=np.float64)
        Q_safe, _ = _safe_qv(P, R, gamma, T, beta_used_t)
        pi_cl = np.argmax(Q_cl, axis=2).astype(np.int64)
        pi_safe = np.argmax(Q_safe, axis=2).astype(np.int64)

        for arm_name, pi in [("classical", pi_cl), ("safe", pi_safe)]:
            returns = _rollout_batch(
                pi, P, R, gamma, T, s0,
                n_rollouts=n_rollouts, rng=rng,
            )
            rec: dict[str, Any] = {
                "task_id": f"A_{i:03d}",
                "arm": arm_name,
                "psi_json": json.dumps(psi, default=str, sort_keys=True),
                "lam": lam,
                "n_rollouts": int(n_rollouts),
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std(ddof=1)),
                "variance_return": float(returns.var(ddof=1)),
            }
            for alpha in alphas:
                rec[f"cvar_{alpha:.2f}"] = _cvar(returns, alpha)
            for beta in betas:
                rec[f"exp_util_beta_{beta:.1f}"] = _exp_utility(returns, beta)
            rows.append(rec)

    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    out_root = pathlib.Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "risk_sensitive_eval.parquet"
    df.to_parquet(out_path, index=False)

    # Paired comparison per task
    pivot = df.pivot(index="task_id", columns="arm")
    diff_rows: list[dict[str, Any]] = []
    metric_cols = [c for c in df.columns
                   if c not in ("task_id", "arm", "psi_json", "lam", "n_rollouts")]
    for task_id in pivot.index:
        d: dict[str, Any] = {"task_id": task_id}
        for metric in metric_cols:
            safe_val = float(pivot.loc[task_id, (metric, "safe")])
            cl_val = float(pivot.loc[task_id, (metric, "classical")])
            d[f"{metric}_classical"] = cl_val
            d[f"{metric}_safe"] = safe_val
            d[f"{metric}_safe_minus_classical"] = safe_val - cl_val
        diff_rows.append(d)
    diff_df = pd.DataFrame(diff_rows)
    diff_path = out_root / "risk_sensitive_diffs.parquet"
    diff_df.to_parquet(diff_path, index=False)

    logger.info("Phase VI-G summary:")
    logger.info("  mean(mean_return_diff) = %.6f",
                float(diff_df["mean_return_safe_minus_classical"].mean()))
    for alpha in alphas:
        col = f"cvar_{alpha:.2f}_safe_minus_classical"
        logger.info("  mean(cvar_%.2f_diff) = %.6f",
                    alpha, float(diff_df[col].mean()))
    for beta in betas:
        col = f"exp_util_beta_{beta:.1f}_safe_minus_classical"
        logger.info("  mean(exp_util_beta=%.1f_diff) = %.6f",
                    beta, float(diff_df[col].mean()))
    logger.info("  elapsed %.2fs", elapsed)

    return {
        "n_tasks": int(len(A)),
        "elapsed_sec": elapsed,
        "output": str(out_path),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shortlist", type=pathlib.Path, required=True)
    p.add_argument("--output-root", type=pathlib.Path, required=True)
    p.add_argument("--n-rollouts", type=int, default=10000)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.05, 0.10, 0.25])
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pilot-seed", type=int, default=42)
    return p.parse_args()


def _main() -> int:
    args = _parse_args()
    run(
        shortlist_path=args.shortlist,
        output_root=args.output_root,
        n_rollouts=args.n_rollouts,
        alphas=args.alphas,
        betas=args.betas,
        seed=args.seed,
        pilot_seed=args.pilot_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
