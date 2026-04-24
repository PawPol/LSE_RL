#!/usr/bin/env python
"""Phase V WP4 -- safe-vs-raw stability on Family C shortlist.

Spec: ``docs/specs/phase_V_mechanism_experiments.md`` section 7 WP4 +
14.6 figure 4.

Family C is finite-horizon: one backward sweep per operator, then
classify by (max local continuation derivative, V_max_abs, nan/inf
guard).  Five labels:

* ``stable_linear``      -- V bounded, ``max d_t <= gamma + tol``.
* ``stable_nonlinear``   -- V bounded, ``gamma < max d_t <= kappa_t + tol``.
* ``expansive_bounded``  -- V bounded, ``max d_t > kappa_t + tol``.
* ``expansive_unbounded`` -- ``V_max_abs > R_box_bound``.
* ``nan_guarded``        -- NaN/inf detected after a stage; sweep
  aborted mid-stage (Q checked before reducing to V).

``R_box_bound = (1 + gamma) / (1 - gamma) * R_max + 1``.  The derivative
is evaluated under ``d_ref * pi*_safe * P`` (same visited measure as
the Phase V search ``p90`` computation).

Outputs: ``results/planning/safe_vs_raw_stability.parquet``,
``figures/main/fig_safe_vs_raw_stability.pdf``, and a read-extend-write
``wp4_runs`` block appended to ``results/summaries/experiment_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MUSHROOM = _REPO_ROOT / "mushroom-rl-dev"
if _MUSHROOM.exists() and str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

import pandas as pd  # noqa: E402

from experiments.weighted_lse_dp.common.manifests import git_sha  # noqa: E402
from experiments.weighted_lse_dp.runners.run_phase_V_search import (  # noqa: E402
    _calibrate_schedule,
)
from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    _classical_qv,
    _compute_safe_local_deriv,
    _deterministic_to_stochastic,
    _forward_occupancy,
    _greedy_policy,
    _safe_q_stage,
    _safe_qv,
)
from experiments.weighted_lse_dp.tasks.family_c_raw_stress import (  # noqa: E402
    family_c,
)

__all__ = [
    "main",
    "run_stability",
    "classify_sweep",
    "sweep_operator",
    "OperatorReport",
]

logger = logging.getLogger("phaseV.stability")

_SCHEMA_VERSION = "phaseV.wp4.v1"
_DEFAULT_SEED = 42
_DEFAULT_PILOT: dict[str, Any] = {"n_episodes": 30, "eps_greedy": 0.1}
_STATUS_STABLE: set[str] = {"stable_linear", "stable_nonlinear"}
_STATUS_ALL: set[str] = {
    "stable_linear", "stable_nonlinear",
    "expansive_bounded", "expansive_unbounded", "nan_guarded",
}

_PARQUET_COLUMNS: list[str] = [
    "task_id", "family", "psi_json", "lam", "operator",
    "v_start", "v_max_abs",
    "local_deriv_max", "local_deriv_p90",
    "local_deriv_exceeds_one_frac", "local_deriv_exceeds_kappa_frac",
    "status", "sweep_completed",
    "gamma", "horizon", "kappa_t_max", "R_max", "R_box_bound",
]


# ---------------------------------------------------------------------------
# Sweep primitives
# ---------------------------------------------------------------------------

@dataclass
class OperatorReport:
    """Result of one backward sweep for a single operator."""

    operator: str
    status: str
    sweep_completed: bool
    v_start: float
    v_max_abs: float
    local_deriv_max: float
    local_deriv_p90: float
    local_deriv_exceeds_one_frac: float
    local_deriv_exceeds_kappa_frac: float
    V: NDArray[np.float64]


def _weighted_quantile(
    values: NDArray[np.float64], weights: NDArray[np.float64], q: float,
) -> float:
    mask = weights > 0.0
    if not np.any(mask):
        return 0.0
    v, w = values[mask], weights[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    idx = min(int(np.searchsorted(cdf, q, side="left")), v.size - 1)
    return float(v[idx])


def classify_sweep(
    *,
    nan_aborted: bool,
    v_max_abs: float,
    R_box_bound: float,
    local_deriv_max: float,
    kappa_t_max: float,
    gamma: float,
    tol: float = 1e-6,
) -> str:
    """Classify one sweep into one of the five status labels."""
    if nan_aborted:
        return "nan_guarded"
    if not np.isfinite(v_max_abs) or v_max_abs > R_box_bound:
        return "expansive_unbounded"
    if local_deriv_max > kappa_t_max + tol:
        return "expansive_bounded"
    if local_deriv_max <= gamma + tol:
        return "stable_linear"
    return "stable_nonlinear"


def sweep_operator(
    *,
    operator: str,
    beta_t: NDArray[np.float64],       # (T,)
    kappa_t: NDArray[np.float64],      # (T,)
    P: NDArray[np.float64],            # (S, A, S')
    R: NDArray[np.float64],            # (S, A, S')
    gamma: float,
    T: int,
    weights: NDArray[np.float64],      # (T, S, A, S') visited measure
    R_box_bound: float,
) -> OperatorReport:
    """Run one backward sweep and classify.

    Per-stage NaN guard: after computing Q[t] via ``_safe_q_stage`` (and
    BEFORE the ``np.max`` reduction to V), check
    ``np.all(np.isfinite(Q[t]))``.  On failure, mark ``nan_guarded`` and
    abort.  A non-finite V[t+1] from a previous stage propagates through
    the same guard because logaddexp returns inf on inf inputs and
    ``np.all(np.isfinite(inf))`` is False.
    """
    S, A = P.shape[0], P.shape[1]
    V = np.full((T + 1, S), np.nan, dtype=np.float64)
    V[T] = 0.0
    Q = np.full((T, S, A), np.nan, dtype=np.float64)
    nan_aborted = False

    for t in range(T - 1, -1, -1):
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                Q[t] = _safe_q_stage(float(beta_t[t]), gamma, R, P, V[t + 1])
        except FloatingPointError:  # pragma: no cover -- defensive
            nan_aborted = True
            break
        if not np.all(np.isfinite(Q[t])):
            nan_aborted = True
            break
        V[t] = np.max(Q[t], axis=1)

    # Derivative under visited measure (using the safe-V estimate for
    # stages that completed; zero-fill aborted-stage V_next).
    V_next = V[1 : T + 1, :]
    V_next_clean = np.where(np.isfinite(V_next), V_next, 0.0)
    deriv = _compute_safe_local_deriv(R, V_next_clean, beta_t, gamma)

    flat_vals = deriv.reshape(-1)
    flat_w = weights.reshape(-1)
    total_w = float(flat_w.sum())
    if total_w > 0.0:
        max_val = float(np.max(flat_vals[flat_w > 0.0]))
        p90 = _weighted_quantile(flat_vals, flat_w, 0.90)
        exceeds_one = float(flat_w[flat_vals > 1.0 + 1e-6].sum() / total_w)
        kappa_bcast = np.broadcast_to(
            kappa_t[:, None, None, None], deriv.shape,
        ).reshape(-1)
        exceeds_kappa = float(
            flat_w[flat_vals > (kappa_bcast + 1e-6)].sum() / total_w
        )
    else:
        max_val = p90 = exceeds_one = exceeds_kappa = 0.0

    # Start-state value and V_max_abs over finite entries.
    if nan_aborted or not np.isfinite(V[0]).all():
        v_start = float("nan")
    else:
        v_start = float(V[0, 0])   # Family C always initial state s0 = 0.
    finite = np.isfinite(V)
    v_max_abs = float(np.max(np.abs(V[finite]))) if np.any(finite) else float("inf")

    status = classify_sweep(
        nan_aborted=nan_aborted,
        v_max_abs=v_max_abs,
        R_box_bound=R_box_bound,
        local_deriv_max=max_val,
        kappa_t_max=float(np.max(kappa_t)) if kappa_t.size else float(gamma),
        gamma=gamma,
    )
    return OperatorReport(
        operator=str(operator), status=status,
        sweep_completed=(not nan_aborted),
        v_start=v_start, v_max_abs=v_max_abs,
        local_deriv_max=max_val, local_deriv_p90=p90,
        local_deriv_exceeds_one_frac=exceeds_one,
        local_deriv_exceeds_kappa_frac=exceeds_kappa,
        V=V,
    )


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------

def _build_visited_weights(
    P: NDArray[np.float64], T: int,
    Q_cl: NDArray[np.float64], Q_safe: NDArray[np.float64], s0: int,
) -> NDArray[np.float64]:
    """Return ``d_ref * pi*_safe * P`` as a (T, S, A, S') tensor.

    ``d_ref = 0.5 * d^{pi*_cl} + 0.5 * d^{pi*_safe}`` (Phase V convention).
    """
    S, A = P.shape[0], P.shape[1]
    pi_cl_stoch = _deterministic_to_stochastic(_greedy_policy(Q_cl), A)
    pi_safe_stoch = _deterministic_to_stochastic(_greedy_policy(Q_safe), A)
    mu_0 = np.zeros(S, dtype=np.float64)
    mu_0[s0] = 1.0
    d_cl = _forward_occupancy(P, pi_cl_stoch, mu_0, T)
    d_safe = _forward_occupancy(P, pi_safe_stoch, mu_0, T)
    d_ref = 0.5 * d_cl + 0.5 * d_safe
    return (
        d_ref[:, :, None, None]
        * pi_safe_stoch[:, :, :, None]
        * P[None, :, :, :]
    )


@dataclass
class _TaskResult:
    task_id: str
    family: str
    psi_json: str
    lam: float
    gamma: float
    T: int
    kappa_t_max: float
    kappa_t: NDArray[np.float64]
    R: NDArray[np.float64]
    P: NDArray[np.float64]
    R_max: float
    R_box_bound: float
    beta_used_t: NDArray[np.float64]
    beta_raw_t: NDArray[np.float64]
    weights: NDArray[np.float64]
    classical: OperatorReport
    safe: OperatorReport
    raw: OperatorReport


def _task_id(row: pd.Series, task_index: int) -> str:
    if "task_id" in row and isinstance(row["task_id"], str) and row["task_id"]:
        return str(row["task_id"])
    psi = json.loads(row["psi_json"])
    L = int(psi.get("L_tail", 0))
    rp = float(psi.get("R_penalty", 0.0))
    mult = float(psi.get("beta_raw_multiplier", 0.0))
    return f"family_C_{task_index:02d}_L{L}_Rp{rp:g}_mult{mult:g}"


def _evaluate_task(
    row: pd.Series, *, seed: int, pilot_cfg: dict[str, Any], task_index: int,
) -> _TaskResult:
    """Rebuild one Family C task, recalibrate, run three sweeps."""
    psi = json.loads(row["psi_json"])
    lam = float(row.get("lam", 0.0))
    mdp = family_c.build_mdp(lam, psi)
    mdp._beta_raw_multiplier = float(  # type: ignore[attr-defined]
        psi.get("beta_raw_multiplier", 2.5)
    )

    sched = _calibrate_schedule(
        mdp, family_label="C", pilot_cfg=pilot_cfg, seed=int(seed),
    )
    beta_used_t = np.asarray(sched["beta_used_t"], dtype=np.float64)
    beta_raw_t = np.asarray(sched["beta_raw_t"], dtype=np.float64)
    kappa_t = np.asarray(sched["kappa_t"], dtype=np.float64)

    P = np.asarray(mdp.p, dtype=np.float64)
    R = np.asarray(mdp.r, dtype=np.float64)
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)
    s0 = int(getattr(mdp, "initial_state", 0))
    R_max = float(np.max(np.abs(R))) if R.size else 1.0
    if gamma < 1.0:
        R_box_bound = float((1.0 + gamma) / (1.0 - gamma) * R_max + 1.0)
    else:
        R_box_bound = float((T + 1) * R_max + 1.0)

    r_bar = np.einsum("ijk,ijk->ij", P, R)
    Q_cl, _ = _classical_qv(P, r_bar, gamma, T)
    Q_safe, _ = _safe_qv(P, R, gamma, T, beta_used_t)
    weights = _build_visited_weights(P, T, Q_cl, Q_safe, s0)

    sweep_kwargs = dict(
        kappa_t=kappa_t, P=P, R=R, gamma=gamma, T=T,
        weights=weights, R_box_bound=R_box_bound,
    )
    report_cl = sweep_operator(
        operator="classical", beta_t=np.zeros(T, dtype=np.float64),
        **sweep_kwargs,
    )
    report_safe = sweep_operator(
        operator="safe", beta_t=beta_used_t, **sweep_kwargs,
    )
    report_raw = sweep_operator(
        operator="raw", beta_t=beta_raw_t, **sweep_kwargs,
    )

    return _TaskResult(
        task_id=_task_id(row, task_index),
        family="C", psi_json=str(row["psi_json"]),
        lam=lam, gamma=gamma, T=T,
        kappa_t_max=float(np.max(kappa_t)), kappa_t=kappa_t,
        R=R, P=P, R_max=R_max, R_box_bound=R_box_bound,
        beta_used_t=beta_used_t, beta_raw_t=beta_raw_t,
        weights=weights,
        classical=report_cl, safe=report_safe, raw=report_raw,
    )


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def _build_parquet_df(task_results: list[_TaskResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tr in task_results:
        for rep in (tr.classical, tr.safe, tr.raw):
            rows.append({
                "task_id": tr.task_id, "family": tr.family,
                "psi_json": tr.psi_json, "lam": float(tr.lam),
                "operator": rep.operator,
                "v_start": float(rep.v_start), "v_max_abs": float(rep.v_max_abs),
                "local_deriv_max": float(rep.local_deriv_max),
                "local_deriv_p90": float(rep.local_deriv_p90),
                "local_deriv_exceeds_one_frac": float(rep.local_deriv_exceeds_one_frac),
                "local_deriv_exceeds_kappa_frac": float(rep.local_deriv_exceeds_kappa_frac),
                "status": rep.status,
                "sweep_completed": bool(rep.sweep_completed),
                "gamma": float(tr.gamma), "horizon": int(tr.T),
                "kappa_t_max": float(tr.kappa_t_max),
                "R_max": float(tr.R_max), "R_box_bound": float(tr.R_box_bound),
            })
    return pd.DataFrame(rows or [], columns=_PARQUET_COLUMNS)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _pick_lead_task(task_results: list[_TaskResult]) -> _TaskResult | None:
    if not task_results:
        return None
    return min(
        task_results,
        key=lambda tr: (-tr.raw.local_deriv_p90, tr.task_id),
    )


def _render_figure(lead: _TaskResult | None, output_path: Path) -> None:
    """Two-panel WP4 figure on the lead Family C task (or a placeholder
    when the shortlist has no Family C rows)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if lead is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no Family C tasks on shortlist",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(output_path, format="pdf", dpi=300)
        plt.close(fig)
        return

    gamma, T, kappa_max = lead.gamma, lead.T, lead.kappa_t_max
    stages = np.arange(T + 1)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=150)

    for label, rep, color in [
        ("classical", lead.classical, "#1f77b4"),
        ("safe",      lead.safe,      "#2ca02c"),
        ("raw",       lead.raw,       "#d62728"),
    ]:
        v = rep.V[:, 0]   # Family C initial state s0 = 0.
        fin = np.isfinite(v)
        axA.plot(stages[fin], v[fin], marker="o", markersize=3.5,
                 linewidth=1.6, color=color,
                 label=f"{label}: status={rep.status} | V[0,s0]={rep.v_start:.3g}")
    axA.set_yscale("symlog", linthresh=1.0)
    axA.set_xlabel("stage t")
    axA.set_ylabel(r"$V[t, s_0]$  (symlog)")
    axA.set_title(f"Panel A -- V per stage on lead task\n{lead.task_id}")
    axA.legend(fontsize=8, loc="best")
    axA.grid(True, alpha=0.35)

    V_next = np.where(np.isfinite(lead.safe.V[1 : T + 1, :]),
                      lead.safe.V[1 : T + 1, :], 0.0)
    deriv_safe = _compute_safe_local_deriv(
        lead.R, V_next, lead.beta_used_t, gamma).reshape(-1)
    deriv_raw = _compute_safe_local_deriv(
        lead.R, V_next, lead.beta_raw_t, gamma).reshape(-1)
    w = lead.weights.reshape(-1)
    mask = w > 0.0
    if np.any(mask):
        bins = np.linspace(0.0,
                           max(float(deriv_raw[mask].max()) + 0.01, 1.1), 48)
        axB.hist(deriv_safe[mask], bins=bins, weights=w[mask],
                 color="#2ca02c", alpha=0.55, label="safe-clipped")
        axB.hist(deriv_raw[mask], bins=bins, weights=w[mask],
                 color="#d62728", alpha=0.55, label="raw")
    axB.axvline(gamma, color="black", linestyle="--", linewidth=1.0,
                label=fr"$\gamma = {gamma:.3f}$")
    axB.axvline(kappa_max, color="#555555", linestyle="--", linewidth=1.0,
                label=fr"$\kappa_{{t,\max}} = {kappa_max:.3f}$")
    axB.axvline(1.0, color="#999999", linestyle=":", linewidth=1.0,
                label=r"$d_t = 1$")
    axB.set_xlabel(r"local continuation derivative $d_t(r, v)$")
    axB.set_ylabel(r"mass under $d_\mathrm{ref}\cdot\pi^*_\mathrm{safe}\cdot P$")
    axB.set_title("Panel B -- safe vs raw local continuation derivative")
    axB.legend(fontsize=8, loc="best")
    axB.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest extension + timestamp helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return ts[: -len("+00:00")] + "Z" if ts.endswith("+00:00") else ts


def _safe_hostname() -> str:
    try:
        return socket.gethostname() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def _extend_manifest(manifest_path: Path, *, wp4_block: dict[str, Any]) -> None:
    """Read-extend-write the ``wp4_runs`` block; preserve other keys."""
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            with open(manifest_path) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}
    existing["wp4_runs"] = wp4_block
    existing.setdefault("schema_version", _SCHEMA_VERSION)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stability(
    *,
    shortlist_path: Path,
    output_root: Path,
    figure_path: Path | None = None,
    seed: int = _DEFAULT_SEED,
    pilot_cfg: dict[str, Any] | None = None,
    exact_argv: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the Family C safe-vs-raw stability comparison end-to-end."""
    t_start = time.time()
    pilot_cfg = dict(pilot_cfg or _DEFAULT_PILOT)

    shortlist_df = pd.read_csv(shortlist_path)
    if "family" not in shortlist_df.columns:
        raise ValueError(
            f"shortlist at {shortlist_path} missing required 'family' column"
        )
    rows_c = shortlist_df[shortlist_df["family"] == "C"].reset_index(drop=True)
    if rows_c.empty:
        logger.warning("no Family C rows in shortlist %s", shortlist_path)

    task_results: list[_TaskResult] = [
        _evaluate_task(row, seed=seed, pilot_cfg=pilot_cfg, task_index=int(idx))
        for idx, row in rows_c.iterrows()
    ]

    df = _build_parquet_df(task_results)
    output_root.mkdir(parents=True, exist_ok=True)
    parquet_path = output_root / "safe_vs_raw_stability.parquet"
    df.to_parquet(parquet_path, index=False)

    lead = _pick_lead_task(task_results)
    if figure_path is None:
        figure_path = _REPO_ROOT / "figures" / "main" / "fig_safe_vs_raw_stability.pdf"
    _render_figure(lead, figure_path)

    manifest_path = _REPO_ROOT / "results" / "summaries" / "experiment_manifest.json"
    wp4_block = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "phase": "phaseV",
        "runner": "run_phase_V_stability",
        "exact_argv": list(exact_argv) if exact_argv is not None else list(sys.argv),
        "seed_list": [int(seed)],
        "shortlist_path": str(shortlist_path),
        "pilot_cfg": dict(pilot_cfg),
        "output_paths": {
            "safe_vs_raw_stability_parquet": str(parquet_path),
            "figure": str(figure_path),
        },
        "n_tasks": int(len(task_results)),
        "lead_task_id": str(lead.task_id) if lead is not None else None,
        "git_sha": git_sha(),
        "host": _safe_hostname(),
        "timestamp": _utc_now_iso(),
        "dry_run": bool(dry_run),
        "elapsed_sec": float(time.time() - t_start),
        "per_task_summary": [
            {
                "task_id": tr.task_id,
                "psi_json": tr.psi_json,
                "classical_status": tr.classical.status,
                "safe_status": tr.safe.status,
                "raw_status": tr.raw.status,
                "raw_local_deriv_p90": float(tr.raw.local_deriv_p90),
                "raw_v_max_abs": float(tr.raw.v_max_abs),
            }
            for tr in task_results
        ],
    }
    _extend_manifest(manifest_path, wp4_block=wp4_block)

    return {
        "n_tasks": int(len(task_results)),
        "lead_task_id": str(lead.task_id) if lead is not None else None,
        "parquet_path": str(parquet_path),
        "figure_path": str(figure_path),
        "manifest_path": str(manifest_path),
        "elapsed_sec": float(time.time() - t_start),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase V WP4 -- safe-vs-raw stability sweep on Family C.",
    )
    p.add_argument("--shortlist", type=Path,
                   default=_REPO_ROOT / "results" / "search" / "shortlist.csv")
    p.add_argument("--output-root", type=Path,
                   default=_REPO_ROOT / "results" / "planning")
    p.add_argument("--figure-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    r = run_stability(
        shortlist_path=args.shortlist,
        output_root=args.output_root,
        figure_path=args.figure_path,
        seed=int(args.seed),
        exact_argv=(sys.argv if argv is None else list(argv)),
        dry_run=bool(args.dry_run),
    )
    logger.info(
        "stability sweep complete: n_tasks=%d lead=%s elapsed=%.2fs",
        r["n_tasks"], r["lead_task_id"], r["elapsed_sec"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
