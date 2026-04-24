"""Phase VI-E / VI-F — dual-objective exact policy evaluation.

Key conceptual result (novel)
------------------------------
On Family A's stochastic variant, the classical-tie parameter `lam_tie`
produces two distinct policies:

- pi*_classical: greedy under the classical Bellman operator.
- pi*_safe: greedy under the safe weighted-LSE operator.

The policies differ at the contest state because the safe operator's
nonlinearity breaks the classical tie.  Evaluated under the classical
objective, pi*_safe <= pi*_classical (optimality principle).  But
evaluated under the safe objective, pi*_safe > pi*_classical.

This demonstrates that the mechanism lives in the OBJECTIVE, not in a
return advantage under a single metric.  RL that trains to match the
classical metric cannot exhibit safe-over-classical dominance; only
a dual-metric comparison can.

This runner computes both policy evaluations exactly via finite-horizon
policy iteration:

    V^{pi}_classical(s0) = E[Sigma_t gamma^t r_t | pi] (linear Bellman)
    V^{pi}_safe(s0)      = fixed point of the safe weighted-LSE
                           operator evaluated along pi.

On stochastic Family A, we observe:
- V^{pi*_safe}_safe - V^{pi*_classical}_safe  > 0  (safe wins safe eval).
- V^{pi*_classical}_classical - V^{pi*_safe}_classical >= 0 (classical
   wins classical eval).

The gap pair (both positive) tells the complete story.

CLI
---
.. code-block:: text

    python -m experiments.weighted_lse_dp.runners.run_phase_VI_dual_eval \\
      --shortlist results/search/shortlist_VI_E_A_stoch.csv \\
      --output-root results/rl_VI_F/
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
    _safe_q_stage,
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
logger = logging.getLogger("phaseVI.dual_eval")


def _policy_eval_classical(
    pi: np.ndarray,                  # (T, S) int argmax per stage
    P: np.ndarray,                   # (S, A, S)
    R: np.ndarray,                   # (S, A, S)
    gamma: float,
    T: int,
) -> np.ndarray:
    """Exact classical policy evaluation on a finite-horizon MDP.

    V^pi[T, :] = 0.  For t < T:
        V^pi[t, s] = Sigma_{s'} P[s, pi[t,s], s'] * (R[s, pi[t,s], s'] + gamma * V^pi[t+1, s'])
    """
    S = P.shape[0]
    V = np.zeros((T + 1, S), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        a = pi[t]                      # (S,)
        s_idx = np.arange(S)
        p_t = P[s_idx, a, :]           # (S, S)
        r_t = R[s_idx, a, :]           # (S, S)
        # expected r + gamma*V[t+1]
        V[t] = np.einsum("ij,ij->i", p_t, r_t + gamma * V[t + 1][None, :])
    return V


def _policy_eval_safe(
    pi: np.ndarray,                  # (T, S) int argmax per stage
    P: np.ndarray,                   # (S, A, S)
    R: np.ndarray,                   # (S, A, S)
    gamma: float,
    T: int,
    beta_used_t: np.ndarray,         # (T,)
) -> np.ndarray:
    """Exact safe policy evaluation.

    V^pi[T, :] = 0.  For t < T:
        V^pi[t, s] = g_t^safe(r[s, pi[t,s], :], V^pi[t+1, :])

    Uses _safe_q_stage to compute the full (S, A) Q at each stage, then
    selects the row corresponding to pi[t, s].
    """
    S, A = P.shape[0], P.shape[1]
    V = np.zeros((T + 1, S), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        Q_t = _safe_q_stage(float(beta_used_t[t]), gamma, R, P, V[t + 1])  # (S, A)
        a = pi[t]                      # (S,)
        s_idx = np.arange(S)
        V[t] = Q_t[s_idx, a]
    return V


def _greedy_policy(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=2).astype(np.int64)


def run(
    shortlist_path: pathlib.Path,
    output_root: pathlib.Path,
    *,
    pilot_seed: int,
) -> dict[str, Any]:
    shortlist = pd.read_csv(shortlist_path)
    A = shortlist[shortlist["family"] == "A"].reset_index(drop=True)
    logger.info("Phase VI-F dual eval: %d Family A tasks", len(A))

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, row in A.iterrows():
        psi = json.loads(row["psi_json"])
        lam = float(row["lam"])
        mdp = family_a.build_mdp(lam, psi)
        P, R, r_bar, gamma, T, S, A_sz = _extract_tensors(mdp)
        s0 = int(getattr(mdp, "initial_state", 0))

        Q_cl, V_cl_star = _classical_qv(P, r_bar, gamma, T)
        sched = _calibrate_schedule(
            mdp, family_label="A",
            pilot_cfg={"n_episodes": 30, "eps_greedy": 0.1},
            seed=int(pilot_seed),
        )
        beta_used_t = np.asarray(sched["beta_used_t"], dtype=np.float64)
        Q_safe, V_safe_star = _safe_qv(P, R, gamma, T, beta_used_t)

        pi_cl = _greedy_policy(Q_cl)       # (T, S)
        pi_safe = _greedy_policy(Q_safe)   # (T, S)

        # Dual evaluation: each policy under each metric.
        V_cl_pi_cl = _policy_eval_classical(pi_cl, P, R, gamma, T)
        V_cl_pi_safe = _policy_eval_classical(pi_safe, P, R, gamma, T)
        V_safe_pi_cl = _policy_eval_safe(pi_cl, P, R, gamma, T, beta_used_t)
        V_safe_pi_safe = _policy_eval_safe(pi_safe, P, R, gamma, T, beta_used_t)

        p_transit = psi.get("p_transit", 1.0)
        stochastic = p_transit < 1.0
        rows.append({
            "task_id": f"A_{i:03d}",
            "psi_json": json.dumps(psi, default=str, sort_keys=True),
            "lam": lam,
            "p_transit": float(p_transit),
            "stochastic": bool(stochastic),
            # Classical eval
            "V_cl_pi_cl_s0": float(V_cl_pi_cl[0, s0]),
            "V_cl_pi_safe_s0": float(V_cl_pi_safe[0, s0]),
            "gap_cl_eval": float(V_cl_pi_cl[0, s0] - V_cl_pi_safe[0, s0]),
            # Safe eval
            "V_safe_pi_cl_s0": float(V_safe_pi_cl[0, s0]),
            "V_safe_pi_safe_s0": float(V_safe_pi_safe[0, s0]),
            "gap_safe_eval": float(V_safe_pi_safe[0, s0] - V_safe_pi_cl[0, s0]),
            # Fixed points
            "V_star_classical": float(V_cl_star[0, s0]),
            "V_star_safe": float(V_safe_star[0, s0]),
            # Policies
            "pi_cl_s0": int(pi_cl[0, s0]),
            "pi_safe_s0": int(pi_safe[0, s0]),
            "start_state_flip": int(pi_cl[0, s0] != pi_safe[0, s0]),
        })
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    out_root = pathlib.Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "dual_eval.parquet"
    df.to_parquet(out_path, index=False)

    logger.info("Dual-eval summary (n=%d):", len(df))
    logger.info("  mean gap_cl_eval   (V_cl  under pi_cl - V_cl  under pi_safe): %.6f",
                float(df["gap_cl_eval"].mean()))
    logger.info("  mean gap_safe_eval (V_safe under pi_safe - V_safe under pi_cl): %.6f",
                float(df["gap_safe_eval"].mean()))
    logger.info("  tasks with start_state_flip=1: %d / %d",
                int(df["start_state_flip"].sum()), len(df))
    logger.info("  elapsed %.2fs", elapsed)

    return {
        "n_tasks": len(df),
        "mean_gap_cl_eval": float(df["gap_cl_eval"].mean()),
        "mean_gap_safe_eval": float(df["gap_safe_eval"].mean()),
        "n_flipped": int(df["start_state_flip"].sum()),
        "elapsed_sec": elapsed,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shortlist", type=pathlib.Path, required=True)
    p.add_argument("--output-root", type=pathlib.Path, required=True)
    p.add_argument("--pilot-seed", type=int, default=42)
    return p.parse_args()


def _main() -> int:
    args = _parse_args()
    run(
        shortlist_path=args.shortlist,
        output_root=args.output_root,
        pilot_seed=args.pilot_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
