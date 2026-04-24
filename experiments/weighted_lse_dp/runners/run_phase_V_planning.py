#!/usr/bin/env python
"""Phase V WP3 -- limited-backup planning diagnostics on the frozen shortlist.

Spec: ``docs/specs/phase_V_mechanism_experiments.md`` §7 WP3 + §14.6. Runs
classical and safe finite-horizon DP under a limited backup budget ``k``
on every **Family A** shortlisted task and persists the six WP3 metrics.
Reuses ``candidate_metrics._classical_qv/_safe_qv/_safe_q_stage``,
``run_phase_V_search._calibrate_schedule``, and
``reference_occupancy.compute_d_ref`` -- never reimplements them.

Outputs: ``<output-root>/limited_backup_metrics.parquet`` (one row per
``(task, k, t, operator)``), ``<output-root>/limited_backup_scalars.parquet``
(per-k scalar metrics), ``results/summaries/experiment_manifest.json`` per
spec §9. Figures live in ``analysis/make_phaseV_wp3_figures.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MUSHROOM = _REPO_ROOT / "mushroom-rl-dev"
if _MUSHROOM.exists() and str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from experiments.weighted_lse_dp.common.manifests import git_sha  # noqa: E402
from experiments.weighted_lse_dp.runners.run_phase_V_search import (  # noqa: E402
    _calibrate_schedule,
)
from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    _classical_qv,
    _extract_tensors,
    _safe_q_stage,
    _safe_qv,
)
from experiments.weighted_lse_dp.search.reference_occupancy import (  # noqa: E402
    compute_d_ref,
)
from experiments.weighted_lse_dp.tasks.family_a_jackpot_vs_stream import (  # noqa: E402
    family_a,
)

logger = logging.getLogger("phaseV.planning")

__all__ = [
    "main",
    "run_planning",
    "limited_backup_sweep",
    "load_family_a_shortlist",
]

_FAMILY_BUILDERS: dict[str, Any] = {"A": family_a.build_mdp}

_METRIC_COLUMNS: list[str] = [
    "task_id", "family", "psi_json", "lam", "k", "t", "operator",
    "v_err_inf", "greedy_correct_fraction",
    "d_ref_weighted_err", "policy_disagreement",
]
_SCALAR_COLUMNS: list[str] = [
    "task_id", "family", "psi_json", "lam", "k", "operator",
    "earliest_stage_correct", "propagation_distance_t",
]


# ---------------------------------------------------------------------------
# Shortlist loading
# ---------------------------------------------------------------------------

def load_family_a_shortlist(
    shortlist_path: Path,
    families: list[str] | None = None,
) -> pd.DataFrame:
    """Load ``shortlist.csv`` and return rows whose ``family`` is in ``families``
    (default ``["A"]``).  Family C rows are skipped: that is WP4's story."""
    df = pd.read_csv(shortlist_path)
    if df.empty:
        return df
    fams = list(families) if families is not None else ["A"]
    keep = df["family"].astype(str).isin(fams)
    return df.loc[keep].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Limited-backup sweep
# ---------------------------------------------------------------------------

def limited_backup_sweep(
    mdp: Any,
    schedule: dict[str, np.ndarray],
    *,
    k_values: list[int],
) -> dict[str, np.ndarray]:
    """Finite-horizon DP under a limited backup budget ``k``.

    For each ``k`` in ``k_values`` and each operator, perform exactly ``k``
    stagewise backups (``t = T-1, T-2, ..., T-k``) from ``V[T] = 0``. Under
    ``k < T`` the terminal boundary is not yet propagated to stages
    ``0..T-k-1`` -- that is the WP3 diagnostic target (spec §7 WP3 item 4).

    Returns arrays stacked over k: V_cl/Q_cl/V_safe/Q_safe plus V*/Q*."""
    P, R, r_bar, gamma, T, S, A = _extract_tensors(mdp)
    beta_used_t = np.asarray(schedule["beta_used_t"], dtype=np.float64)
    if beta_used_t.shape != (T,):
        raise ValueError(
            f"schedule['beta_used_t'] must have shape ({T},); "
            f"got {beta_used_t.shape}."
        )

    Q_cl_star, V_cl_star = _classical_qv(P, r_bar, gamma, T)
    Q_safe_star, V_safe_star = _safe_qv(P, R, gamma, T, beta_used_t)

    K = len(k_values)
    V_cl = np.zeros((K, T + 1, S), dtype=np.float64)
    V_safe = np.zeros((K, T + 1, S), dtype=np.float64)
    Q_cl = np.zeros((K, T, S, A), dtype=np.float64)
    Q_safe = np.zeros((K, T, S, A), dtype=np.float64)

    for idx, k in enumerate(k_values):
        if not (1 <= k <= T):
            raise ValueError(f"k must lie in [1, T]; got k={k}, T={T}.")
        # Classical: sweep stages T-1, T-2, ..., T-k.
        for t in range(T - 1, T - 1 - k, -1):
            E_v = np.einsum("ijk,k->ij", P, V_cl[idx, t + 1])
            Q_cl[idx, t] = r_bar + gamma * E_v
            V_cl[idx, t] = np.max(Q_cl[idx, t], axis=1)
        for t in range(T - 1, T - 1 - k, -1):
            Q_safe[idx, t] = _safe_q_stage(
                float(beta_used_t[t]), gamma, R, P, V_safe[idx, t + 1]
            )
            V_safe[idx, t] = np.max(Q_safe[idx, t], axis=1)

    return {
        "V_cl": V_cl, "Q_cl": Q_cl,
        "V_safe": V_safe, "Q_safe": Q_safe,
        "V_cl_star": V_cl_star, "Q_cl_star": Q_cl_star,
        "V_safe_star": V_safe_star, "Q_safe_star": Q_safe_star,
        "gamma": gamma, "horizon": T,
    }


# ---------------------------------------------------------------------------
# Per-task metrics
# ---------------------------------------------------------------------------

def _argmax_policy(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=-1).astype(np.int64)


def _earliest_full_correct(frac_by_stage: np.ndarray) -> float:
    """Smallest stage ``t`` where greedy correctness hits 1.0; nan if never."""
    hits = np.where(frac_by_stage >= 1.0 - 1e-12)[0]
    return float(hits[0]) if hits.size else float("nan")


def _propagation_distance(
    V_k_stages: np.ndarray,   # (T, S)
    v_star_at_s0: float,
    s0: int,
) -> float:
    """Smallest ``t`` with ``V_k[t, s0]`` crossing ``0.5 * V*[0, s0]``.

    Sign-agnostic: if ``V*[0, s0] > 0`` we look for V_k > threshold; if
    ``< 0`` we look for ``V_k < threshold``; ``== 0`` returns ``nan``.
    Returns ``nan`` if no stage meets the criterion.
    """
    if not np.isfinite(v_star_at_s0) or v_star_at_s0 == 0.0:
        return float("nan")
    threshold = 0.5 * v_star_at_s0
    if v_star_at_s0 > 0:
        hits = np.where(V_k_stages[:, s0] > threshold)[0]
    else:
        hits = np.where(V_k_stages[:, s0] < threshold)[0]
    return float(hits[0]) if hits.size else float("nan")


def _compute_task_rows(
    *,
    task_id: str,
    family: str,
    psi_json: str,
    lam: float,
    sweep: dict[str, np.ndarray],
    d_ref: np.ndarray,
    s0: int,
    k_values: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-(k, t, operator) metric rows and per-k scalar rows."""
    T = int(sweep["horizon"])
    V_cl_star = sweep["V_cl_star"]
    V_safe_star = sweep["V_safe_star"]
    Q_cl_star = sweep["Q_cl_star"]
    Q_safe_star = sweep["Q_safe_star"]
    pi_cl_star = _argmax_policy(Q_cl_star)
    pi_safe_star = _argmax_policy(Q_safe_star)

    d_ref_sum = float(d_ref.sum())
    d_ref_norm = d_ref_sum if d_ref_sum > 0.0 else 1.0

    metric_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []

    for idx, k in enumerate(k_values):
        V_cl_k = sweep["V_cl"][idx]
        V_safe_k = sweep["V_safe"][idx]
        Q_cl_k = sweep["Q_cl"][idx]
        Q_safe_k = sweep["Q_safe"][idx]

        pi_cl_k = _argmax_policy(Q_cl_k)
        pi_safe_k = _argmax_policy(Q_safe_k)
        disagree_mask = (pi_cl_k != pi_safe_k).astype(np.float64)
        policy_disagreement_k = float(
            (d_ref * disagree_mask).sum() / d_ref_norm
        )

        for t in range(T):
            v_err_cl = float(np.max(np.abs(V_cl_k[t] - V_cl_star[t])))
            greedy_cl = float((pi_cl_k[t] == pi_cl_star[t]).mean())
            dref_err_cl = float(
                (d_ref[t] * np.abs(V_cl_k[t] - V_cl_star[t])).sum()
            )
            v_err_safe = float(np.max(np.abs(V_safe_k[t] - V_safe_star[t])))
            greedy_safe = float((pi_safe_k[t] == pi_safe_star[t]).mean())
            dref_err_safe = float(
                (d_ref[t] * np.abs(V_safe_k[t] - V_safe_star[t])).sum()
            )
            base = {
                "task_id": task_id, "family": family, "psi_json": psi_json,
                "lam": float(lam), "k": int(k), "t": int(t),
                "policy_disagreement": policy_disagreement_k,
            }
            metric_rows.append({
                **base, "operator": "classical",
                "v_err_inf": v_err_cl,
                "greedy_correct_fraction": greedy_cl,
                "d_ref_weighted_err": dref_err_cl,
            })
            metric_rows.append({
                **base, "operator": "safe",
                "v_err_inf": v_err_safe,
                "greedy_correct_fraction": greedy_safe,
                "d_ref_weighted_err": dref_err_safe,
            })

        cl_frac = (pi_cl_k == pi_cl_star).mean(axis=1).astype(np.float64)
        safe_frac = (pi_safe_k == pi_safe_star).mean(axis=1).astype(np.float64)
        earliest_cl = _earliest_full_correct(cl_frac)
        earliest_safe = _earliest_full_correct(safe_frac)
        prop_cl = _propagation_distance(V_cl_k[:T], V_cl_star[0, s0], s0)
        prop_safe = _propagation_distance(V_safe_k[:T], V_safe_star[0, s0], s0)

        scalar_rows.append({
            "task_id": task_id, "family": family, "psi_json": psi_json,
            "lam": float(lam), "k": int(k), "operator": "classical",
            "earliest_stage_correct": float(earliest_cl),
            "propagation_distance_t": float(prop_cl),
        })
        scalar_rows.append({
            "task_id": task_id, "family": family, "psi_json": psi_json,
            "lam": float(lam), "k": int(k), "operator": "safe",
            "earliest_stage_correct": float(earliest_safe),
            "propagation_distance_t": float(prop_safe),
        })
    return metric_rows, scalar_rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_planning(
    shortlist_path: Path,
    *,
    output_root: Path,
    k_max: int | None = None,
    families: list[str] | None = None,
    seed: int = 42,
    exact_argv: list[str] | None = None,
    dry_run: bool = False,
    emit_figures: bool = True,
) -> dict[str, Any]:
    """Execute the WP3 limited-backup diagnostic sweep.  Returns a summary."""
    t_start = time.time()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_family_a_shortlist(shortlist_path, families=families)
    if df.empty:
        logger.warning("No Family A tasks in %s; nothing to do.", shortlist_path)
        _emit_empty_outputs(output_root)
        manifest_path = _emit_manifest(
            output_root=output_root, shortlist_path=shortlist_path,
            n_tasks=0, k_max=k_max, seed=seed,
            families=families or ["A"],
            exact_argv=exact_argv, dry_run=dry_run,
            elapsed_sec=float(time.time() - t_start),
            lead_task_id=None,
        )
        return {
            "n_tasks": 0, "manifest_path": str(manifest_path),
            "elapsed_sec": float(time.time() - t_start),
            "lead_task_id": None,
        }

    metric_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    per_task_summaries: list[dict[str, Any]] = []
    pilot_cfg: dict[str, Any] = {"n_episodes": 30, "eps_greedy": 0.1}

    for idx, row in df.iterrows():
        fam = str(row["family"])
        if fam not in _FAMILY_BUILDERS:
            logger.warning("task %d: unsupported family %s; skipping", idx, fam)
            continue
        psi = json.loads(str(row["psi_json"]))
        lam = float(row["lam"])
        task_id = f"{fam}_{idx:03d}"
        logger.info(
            "task %s family=%s lam=%.4g psi=%s",
            task_id, fam, lam, json.dumps(psi, sort_keys=True),
        )
        mdp = _FAMILY_BUILDERS[fam](lam, psi)
        sched = _calibrate_schedule(
            mdp, family_label=fam, pilot_cfg=pilot_cfg, seed=int(seed),
        )
        T = int(mdp.info.horizon)
        k_up = int(T) if k_max is None else int(min(k_max, T))
        if k_up < 1:
            raise ValueError(f"k_max={k_max} yields 0 backups for T={T}")
        k_values = list(range(1, k_up + 1))

        sweep = limited_backup_sweep(mdp, sched, k_values=k_values)
        d = compute_d_ref(
            mdp, _argmax_policy(sweep["Q_cl_star"]),
            _argmax_policy(sweep["Q_safe_star"]),
        )
        d_ref = np.asarray(d["d_ref"], dtype=np.float64)
        s0 = int(getattr(mdp, "initial_state", 0))

        t_rows, s_rows = _compute_task_rows(
            task_id=task_id, family=fam,
            psi_json=str(row["psi_json"]), lam=lam,
            sweep=sweep, d_ref=d_ref, s0=s0, k_values=k_values,
        )
        metric_rows.extend(t_rows)
        scalar_rows.extend(s_rows)
        per_task_summaries.append({
            "task_id": task_id, "family": fam,
            "psi_json": str(row["psi_json"]), "lam": lam, "horizon": T,
            "k_values": k_values,
            "value_gap_norm": float(row.get("value_gap_norm", 0.0)),
            "sweep": sweep, "d_ref": d_ref, "s0": s0,
        })

    metrics_df = pd.DataFrame(metric_rows, columns=_METRIC_COLUMNS)
    scalars_df = pd.DataFrame(scalar_rows, columns=_SCALAR_COLUMNS)
    _write_parquet(metrics_df, output_root / "limited_backup_metrics.parquet")
    _write_parquet(scalars_df, output_root / "limited_backup_scalars.parquet")

    lead_task_id: str | None = None
    if emit_figures and per_task_summaries:
        lead = max(
            per_task_summaries,
            key=lambda d: abs(float(d.get("value_gap_norm", 0.0))),
        )
        lead_task_id = str(lead["task_id"])
        from experiments.weighted_lse_dp.analysis.make_phaseV_wp3_figures import (
            emit_wp3_figures,
        )
        fig_dir = _REPO_ROOT / "figures" / "main"
        fig_dir.mkdir(parents=True, exist_ok=True)
        emit_wp3_figures(
            lead=lead,
            metrics_df=metrics_df,
            scalars_df=scalars_df,
            fig_dir=fig_dir,
        )

    manifest_path = _emit_manifest(
        output_root=output_root, shortlist_path=shortlist_path,
        n_tasks=len(per_task_summaries), k_max=k_max, seed=seed,
        families=families or ["A"], exact_argv=exact_argv, dry_run=dry_run,
        elapsed_sec=float(time.time() - t_start),
        lead_task_id=lead_task_id,
    )
    return {
        "n_tasks": len(per_task_summaries),
        "manifest_path": str(manifest_path),
        "elapsed_sec": float(time.time() - t_start),
        "lead_task_id": lead_task_id,
    }


# ---------------------------------------------------------------------------
# Output writers / manifest
# ---------------------------------------------------------------------------

def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
    except ImportError:  # pragma: no cover
        try:
            df.to_parquet(path, engine="fastparquet", index=False)
        except ImportError as exc:
            raise RuntimeError(
                f"parquet engines unavailable; cannot write {path}."
            ) from exc


def _emit_empty_outputs(output_root: Path) -> None:
    _write_parquet(
        pd.DataFrame(columns=_METRIC_COLUMNS),
        output_root / "limited_backup_metrics.parquet",
    )
    _write_parquet(
        pd.DataFrame(columns=_SCALAR_COLUMNS),
        output_root / "limited_backup_scalars.parquet",
    )


def _emit_manifest(
    *,
    output_root: Path,
    shortlist_path: Path,
    n_tasks: int,
    k_max: int | None,
    seed: int,
    families: list[str],
    exact_argv: list[str] | None,
    dry_run: bool,
    elapsed_sec: float,
    lead_task_id: str | None,
) -> Path:
    summaries_dir = _REPO_ROOT / "results" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "phaseV.planning.v1",
        "created_at": _utc_now_iso(),
        "phase": "phaseV",
        "runner": "run_phase_V_planning",
        "exact_argv": list(exact_argv) if exact_argv is not None else list(sys.argv),
        "seed_list": [int(seed)],
        "families": list(families),
        "shortlist_path": str(shortlist_path),
        "output_paths": {
            "limited_backup_metrics": str(
                output_root / "limited_backup_metrics.parquet"
            ),
            "limited_backup_scalars": str(
                output_root / "limited_backup_scalars.parquet"
            ),
        },
        "k_max": int(k_max) if k_max is not None else None,
        "n_tasks": int(n_tasks),
        "lead_task_id": lead_task_id,
        "git_sha": git_sha(),
        "host": _safe_hostname(),
        "timestamp": _utc_now_iso(),
        "dry_run": bool(dry_run),
        "elapsed_sec": float(elapsed_sec),
    }
    path = summaries_dir / "experiment_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def _utc_now_iso() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if ts.endswith("+00:00"):
        ts = ts[: -len("+00:00")] + "Z"
    return ts


def _safe_hostname() -> str:
    try:
        return socket.gethostname() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase V WP3 -- limited-backup planning diagnostics.",
    )
    p.add_argument("--shortlist", type=Path, required=True,
                   help="Path to shortlist.csv produced by run_phase_V_search.")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Output directory for limited-backup parquet files.")
    p.add_argument("--families", nargs="+", default=None,
                   help="Families to include (default: A).")
    p.add_argument("--k-max", type=int, default=None,
                   help="Backup budget upper bound (default: task horizon T).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Execute the pipeline but skip figure emission.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    summary = run_planning(
        args.shortlist,
        output_root=args.output_root,
        k_max=args.k_max,
        families=args.families,
        seed=int(args.seed),
        exact_argv=sys.argv if argv is None else list(argv),
        dry_run=bool(args.dry_run),
        emit_figures=not bool(args.dry_run),
    )
    logger.info(
        "WP3 planning complete: n_tasks=%d lead=%s elapsed=%.2fs",
        summary["n_tasks"], summary.get("lead_task_id"), summary["elapsed_sec"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
