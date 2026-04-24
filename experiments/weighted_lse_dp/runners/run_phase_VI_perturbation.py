"""Phase VI-C — Perturbation-growth stability test (novel).

Framework
---------
The classical safety story ("raw VI diverges, safe VI converges") does not
apply directly to finite-horizon MDPs — terminal absorbing states truncate
a single backward sweep's amplification.  This runner introduces an iterated
backward-sweep variant with perturbed V_T initialization to expose the raw
operator's expansivity as a function of the number of outer sweeps.

Definition
~~~~~~~~~~
Let F: R^S → R^S map the terminal-value boundary V_T to the start-state
value V_0 via one backward sweep through T stages under a given operator.
For classical / safe-clipped operators, F is a contraction with rate κ^T ≤ 1.
For raw-unclipped operators with per-transition local derivative d > 1 on
visited mass, F is expansive at sensitivity d^T > 1.

Iterate
    V_0^(k+1) = F(perturb(V_0^(k))),  k = 0, 1, ..., K,
where perturb(v) injects Gaussian noise δ * N(0, σ²) independent per state.
Measure

    ρ(k) = ||V_0^(k) - V*_0||_∞ / δ

— the normalized error amplitude.  For a stable operator, ρ(k) decays
(or saturates near δ).  For an expansive operator with compound factor
c > 1 per outer iteration, ρ(k) grows geometrically and v_max diverges.

This is a finite-horizon analogue of the standard infinite-horizon
"iterated VI convergence" test: replace the infinite-horizon iteration
with repeated perturbed backward sweeps whose outer growth rate is
a direct functional image of the operator's local derivative.

Key result
----------
Safe operator: ρ(k) ≤ κ^T * δ uniformly in k.  Contraction guarantee.
Raw operator at d > 1 on visited mass: ρ(k) ~ (d^T)^k * δ.  Compound
divergence that dwarfs the single-sweep finite-horizon bound.

CLI
---
.. code-block:: text

    python -m experiments.weighted_lse_dp.runners.run_phase_VI_perturbation \\
      --shortlist results/search/shortlist_VI_A_C.csv \\
      --output-root results/planning_VI_C/ \\
      [--n-outer 20] [--delta 1.0] [--sigma 1.0]

Outputs
-------
- results/planning_VI_C/perturbation_growth.parquet
  columns: task_id, operator, k, delta, rho, v_max_abs, diverged
- figures/main/fig_perturbation_growth.pdf
- results/summaries/experiment_manifest.json extended with wp_vi_c block
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
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    _classical_qv,
    _extract_tensors,
    _safe_q_stage,
)
from experiments.weighted_lse_dp.tasks.family_c_raw_stress import (  # noqa: E402
    family_c,
)
from experiments.weighted_lse_dp.runners.run_phase_V_search import (  # noqa: E402
    _calibrate_schedule,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("phaseVI.perturbation")


def _backward_sweep(
    V_T: np.ndarray,                # (S,)
    beta_t: np.ndarray,              # (T,)
    P: np.ndarray,                   # (S, A, S)
    R: np.ndarray,                   # (S, A, S)
    gamma: float,
    r_bar: np.ndarray,               # (S, A)
) -> tuple[np.ndarray, float]:
    """Run one backward sweep from boundary V_T.

    Returns ``(V_0, v_max_abs)``.  ``V_0`` is the start-state stage value
    (shape ``(S,)``); ``v_max_abs = max_{t, s} |V[t, s]|`` over the
    sweep.
    """
    T = beta_t.shape[0]
    S = P.shape[0]
    V = np.zeros((T + 1, S), dtype=np.float64)
    V[T] = V_T
    v_max_abs = float(np.max(np.abs(V_T)))
    for t in range(T - 1, -1, -1):
        b = float(beta_t[t])
        if abs(b) < 1e-12:
            E_v = np.einsum("ijk,k->ij", P, V[t + 1])
            Q = r_bar + gamma * E_v
        else:
            Q = _safe_q_stage(b, gamma, R, P, V[t + 1])
        if not np.all(np.isfinite(Q)):
            return V[0], float("inf")
        V[t] = np.max(Q, axis=1)
        v_max_abs = max(v_max_abs, float(np.max(np.abs(V[t]))))
    return V[0], v_max_abs


def _iterated_horizon_trajectory(
    mdp: Any,
    schedule: dict[str, np.ndarray],
    *,
    operator: str,
    n_outer: int,
) -> list[dict[str, Any]]:
    """Noise-free iterated backward sweep.

    F is the finite-horizon backward-sweep operator V_T -> V_0.  The
    classical and safe-clipped operators have F a contraction with rate
    kappa^T < 1; the raw operator at d > 1 on visited mass has F
    expansive.

    Iterate V_T^(k+1) = F(V_T^(k)) starting from V_T^(0) = R_max * ones(S).
    For a contractive F, the iterates converge to a fixed point at the
    infinite-horizon V^*.  For an expansive F, the iterates diverge.

    Returns a list of dicts with ``operator, k, v_max_abs, diverged,
    rho_change``.  ``rho_change = ||V_T^(k+1) - V_T^(k)||_inf``.
    """
    P, R, r_bar, gamma, T, S, _A = _extract_tensors(mdp)
    if operator == "classical":
        beta_t = np.zeros(T, dtype=np.float64)
    elif operator == "safe":
        beta_t = np.asarray(schedule["beta_used_t"], dtype=np.float64)
    elif operator == "raw":
        beta_t = np.asarray(schedule["beta_raw_t"], dtype=np.float64)
    else:
        raise ValueError(f"operator must be classical|safe|raw; got {operator!r}.")

    R_max = float(np.max(np.abs(R))) if R.size else 1.0
    V_T = R_max * np.ones(S, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    prev_V_T = V_T.copy()
    for k in range(int(n_outer) + 1):
        V_0, v_max_abs = _backward_sweep(V_T, beta_t, P, R, gamma, r_bar)
        if not np.all(np.isfinite(V_0)) or not np.isfinite(v_max_abs) or v_max_abs > 1e15:
            rows.append({
                "operator": operator,
                "k": int(k),
                "v_max_abs": float("inf"),
                "diverged": True,
                "rho_change": float("inf"),
            })
            for k_pad in range(k + 1, int(n_outer) + 1):
                rows.append({
                    "operator": operator,
                    "k": int(k_pad),
                    "v_max_abs": float("inf"),
                    "diverged": True,
                    "rho_change": float("inf"),
                })
            break
        rho_change = float(np.max(np.abs(V_0 - prev_V_T)))
        rows.append({
            "operator": operator,
            "k": int(k),
            "v_max_abs": float(v_max_abs),
            "diverged": False,
            "rho_change": rho_change,
        })
        prev_V_T = V_T.copy()
        V_T = V_0  # next outer iter extends the effective horizon.

    return rows


def _perturbation_trajectory(
    mdp: Any,
    schedule: dict[str, np.ndarray],
    *,
    operator: str,
    n_outer: int,
    delta: float,
    sigma: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Iterate perturbed backward sweeps and record growth.

    Operators
    ~~~~~~~~~
    * ``"classical"`` -- beta_t = 0 everywhere.
    * ``"safe"``      -- beta_t = schedule["beta_used_t"].
    * ``"raw"``       -- beta_t = schedule["beta_raw_t"].

    Returns
    -------
    list of dict with keys ``operator, k, delta, rho, v_max_abs, diverged``.
    """
    P, R, r_bar, gamma, T, S, _A = _extract_tensors(mdp)
    if operator == "classical":
        beta_t = np.zeros(T, dtype=np.float64)
    elif operator == "safe":
        beta_t = np.asarray(schedule["beta_used_t"], dtype=np.float64)
    elif operator == "raw":
        beta_t = np.asarray(schedule["beta_raw_t"], dtype=np.float64)
    else:
        raise ValueError(f"operator must be classical|safe|raw; got {operator!r}.")

    rng = np.random.default_rng(int(seed))

    # Baseline V_0 under zero boundary -- the reference unperturbed solution.
    V0_star, _ = _backward_sweep(
        np.zeros(S), beta_t, P, R, gamma, r_bar,
    )

    rows: list[dict[str, Any]] = []
    # Initialize V_T = delta * sigma * N(0, 1).
    V_T = float(delta) * float(sigma) * rng.standard_normal(S)
    for k in range(int(n_outer) + 1):
        V_0, v_max_abs = _backward_sweep(V_T, beta_t, P, R, gamma, r_bar)
        # Normalized growth: rho(k) = ||V_0 - V*_0||_inf / delta.
        if not np.all(np.isfinite(V_0)):
            rho = float("inf")
            diverged = True
        else:
            rho = float(np.max(np.abs(V_0 - V0_star)) / max(float(delta), 1e-30))
            diverged = not np.isfinite(v_max_abs) or v_max_abs > 1e15
        rows.append({
            "operator": operator,
            "k": int(k),
            "delta": float(delta),
            "rho": rho,
            "v_max_abs": (
                float(v_max_abs) if np.isfinite(v_max_abs) else float("inf")
            ),
            "diverged": bool(diverged),
        })
        if diverged:
            # Stop to avoid runaway; pad remaining k with inf.
            for k_pad in range(k + 1, int(n_outer) + 1):
                rows.append({
                    "operator": operator,
                    "k": int(k_pad),
                    "delta": float(delta),
                    "rho": float("inf"),
                    "v_max_abs": float("inf"),
                    "diverged": True,
                })
            break
        # Next iterate uses the previous V_0 as the new V_T boundary,
        # plus fresh noise of the same amplitude.  This compounds the
        # operator's per-sweep growth factor across outer iterations.
        V_T = V_0 + float(delta) * float(sigma) * rng.standard_normal(S)

    return rows


def _load_shortlist_family_c(
    shortlist_path: pathlib.Path,
) -> list[tuple[str, dict[str, Any], float]]:
    df = pd.read_csv(shortlist_path)
    df = df[df["family"] == "C"].reset_index(drop=True)
    out: list[tuple[str, dict[str, Any], float]] = []
    for i, row in df.iterrows():
        psi = json.loads(row["psi_json"])
        lam = float(row["lam"])
        task_id = (
            f"family_C_{i:02d}_L{psi.get('L_tail')}_Rp{int(psi.get('R_penalty'))}"
            f"_mult{int(psi.get('beta_raw_multiplier'))}"
        )
        out.append((task_id, psi, lam))
    return out


def _plot_perturbation_growth(
    df: pd.DataFrame,
    lead_task_id: str,
    out_path: pathlib.Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["task_id"] == lead_task_id].copy()
    if sub.empty:
        logger.warning("no data for lead_task_id=%s; skipping plot", lead_task_id)
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    for op, color, label in [
        ("classical", "tab:blue", "classical"),
        ("safe", "tab:green", "safe (clipped)"),
        ("raw", "tab:red", "raw (unclipped)"),
    ]:
        op_rows = sub[sub["operator"] == op].sort_values("k")
        if op_rows.empty:
            continue
        y = op_rows["rho"].replace([np.inf, -np.inf], np.nan).values
        ax.semilogy(op_rows["k"].values, y, marker="o", color=color, label=label)
    ax.set_xlabel("outer backward-sweep iteration k")
    ax.set_ylabel(r"$\rho(k) = \|V_0^{(k)} - V_0^*\|_\infty / \delta$")
    ax.set_title(
        f"Phase VI-C perturbation growth: {lead_task_id}\n"
        "(finite-horizon iterated backward sweep)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run(
    shortlist_path: pathlib.Path,
    output_root: pathlib.Path,
    *,
    n_outer: int,
    delta: float,
    sigma: float,
    seed: int,
) -> dict[str, Any]:
    tasks = _load_shortlist_family_c(shortlist_path)
    logger.info("Phase VI-C: %d Family C tasks loaded from %s",
                len(tasks), shortlist_path)

    all_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for task_id, psi, lam in tasks:
        mdp = family_c.build_mdp(lam, psi)
        schedule = _calibrate_schedule(
            mdp, family_label="C",
            pilot_cfg={"n_episodes": 30, "eps_greedy": 0.1},
            seed=int(seed),
        )
        for op in ("classical", "safe", "raw"):
            # Run both the perturbed test (stochastic steady-state) and
            # the noise-free iterated-horizon test.
            rows = _perturbation_trajectory(
                mdp, schedule,
                operator=op,
                n_outer=n_outer,
                delta=delta,
                sigma=sigma,
                seed=int(seed),
            )
            for r in rows:
                r["task_id"] = task_id
                r["psi_json"] = json.dumps(psi, default=str, sort_keys=True)
                r["lam"] = lam
                r["test"] = "perturbed"
            all_rows.extend(rows)

            iter_rows = _iterated_horizon_trajectory(
                mdp, schedule, operator=op, n_outer=n_outer,
            )
            for r in iter_rows:
                r["task_id"] = task_id
                r["psi_json"] = json.dumps(psi, default=str, sort_keys=True)
                r["lam"] = lam
                r["test"] = "iterated_horizon"
                # Add rho=NaN placeholder so the parquet schema lines up
                # with the perturbed rows.
                r["rho"] = float("nan")
                r["delta"] = float(delta)
            all_rows.extend(iter_rows)
    elapsed = time.time() - t0

    out_root = pathlib.Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    parquet_path = out_root / "perturbation_growth.parquet"
    df.to_parquet(parquet_path, index=False)

    # Pick lead task: highest terminal rho on the raw operator.
    raw_rows = df[df["operator"] == "raw"].copy()
    terminal = raw_rows.loc[raw_rows.groupby("task_id")["k"].idxmax()]
    terminal = terminal.replace([np.inf, -np.inf], np.nan).dropna(subset=["rho"])
    if terminal.empty:
        # All diverged; pick first task_id.
        lead_task_id = tasks[0][0]
    else:
        lead_task_id = terminal.loc[terminal["rho"].idxmax(), "task_id"]
    fig_path = pathlib.Path("figures/main/fig_perturbation_growth.pdf")
    _plot_perturbation_growth(df, lead_task_id, fig_path)

    # Summary stats
    diverged_tasks = df[df["diverged"]]["task_id"].nunique()
    safe_final = df[(df["operator"] == "safe") & (df["k"] == n_outer)]["rho"]
    raw_final = df[(df["operator"] == "raw") & (df["k"] == n_outer)]["rho"]
    classical_final = df[(df["operator"] == "classical") & (df["k"] == n_outer)]["rho"]
    logger.info(
        "Phase VI-C complete: n_tasks=%d lead=%s diverged_raw=%d "
        "final_rho classical=%.3e safe=%.3e raw=%.3e elapsed=%.2fs",
        len(tasks), lead_task_id, diverged_tasks,
        float(classical_final.median()),
        float(safe_final.median()),
        float(raw_final.replace([np.inf], np.nan).median()),
        elapsed,
    )

    return {
        "n_tasks": len(tasks),
        "lead_task_id": lead_task_id,
        "diverged_tasks": int(diverged_tasks),
        "elapsed_sec": elapsed,
        "parquet_path": str(parquet_path),
        "figure_path": str(fig_path),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shortlist", type=pathlib.Path, required=True)
    p.add_argument("--output-root", type=pathlib.Path, required=True)
    p.add_argument("--n-outer", type=int, default=20)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _main() -> int:
    args = _parse_args()
    run(
        shortlist_path=args.shortlist,
        output_root=args.output_root,
        n_outer=args.n_outer,
        delta=args.delta,
        sigma=args.sigma,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
