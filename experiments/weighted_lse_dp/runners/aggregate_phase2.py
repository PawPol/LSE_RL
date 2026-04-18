#!/usr/bin/env python
"""Phase II seed aggregation and calibration-stat extraction.

Reads all completed run directories under the Phase II raw result tree,
groups them by ``(suite, task, algorithm)``, aggregates scalar metrics
across seeds (with bootstrapped 95% CIs), averages stage-wise
calibration arrays, and writes:

1. Per-(task, algorithm) summary JSON files under
   ``<out-root>/phase2/aggregated/<task>/<algorithm>/summary.json``.
2. Per-task-family calibration JSON files under
   ``<out-root>/phase2/calibration/<task_family>.json`` for Phase III
   consumption (spec section 12).

Output layout::

    <out-root>/
        phase2/
            aggregated/
                <task>/
                    <algorithm>/
                        summary.json                -- per-(task, algo) seed aggregate
                    calibration_mean.npz            -- task-level calibration average
            calibration/
                <task_family>.json                  -- Phase III calibration input

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/aggregate_phase2.py \\
        [--out-root PATH]     # default: results/weighted_lse_dp
        [--task TASK]         # aggregate only this task (optional)
        [--dry-run]           # print discovery without writing
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.io import (  # noqa: E402
    load_json,
    load_npz,
    save_json,
    save_npz_with_schema,
    make_npz_schema,
)
from experiments.weighted_lse_dp.common.manifests import (  # noqa: E402
    find_run_dirs,
    load_run_json,
    load_metrics_json,
)
from experiments.weighted_lse_dp.common.metrics import aggregate  # noqa: E402

__all__ = [
    "discover_runs",
    "group_runs",
    "aggregate_group",
    "build_calibration_json",
    "write_outputs",
    "main",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUT_ROOT: str = "results/weighted_lse_dp"
"""Bare results root -- no phase/suite embedded (lesson 2026-04-17)."""

_SUITES = ("paper_suite", "smoke")
"""Suites to scan, in priority order."""

_CALIBRATION_SCHEMA_VERSION: str = "2.0.0"
"""Schema version for per-task-family calibration JSON (spec section 12)."""


# ---------------------------------------------------------------------------
# Scalar metric keys we attempt to aggregate from metrics.json
# ---------------------------------------------------------------------------

_SCALAR_METRIC_KEYS: tuple[str, ...] = (
    # DP metrics
    "final_bellman_residual",
    "n_sweeps",
    "n_iters",
    "wall_clock_s",
    # RL metrics
    "train_steps",
    "n_transitions",
    "final_disc_return_mean",
    "auc_disc_return",
    "steps_to_threshold",
    "success_rate",
)


# ---------------------------------------------------------------------------
# Per-stage calibration array names we average across seeds
# ---------------------------------------------------------------------------

_STAGEWISE_KEYS: tuple[str, ...] = (
    "stage",
    "count",
    "reward_mean",
    "margin_q50",
    "margin_q05",
    "margin_q95",
    "pos_margin_mean",
    "neg_margin_mean",
    "aligned_margin_freq",
    "aligned_positive_mean",
    "aligned_negative_mean",
    "td_target_std",
    "td_error_std",
)
"""Calibration array names that appear in the calibration JSON ``stagewise``
block.  These are per-stage arrays of shape ``(H+1,)`` in each seed's
``calibration_stats.npz``."""

# Scalar calibration keys (shape (1,)) read from calibration_stats.npz
_CALIB_SCALAR_TAIL_KEYS: tuple[str, ...] = (
    "return_cvar_5pct",
    "return_cvar_10pct",
    "event_rate",
    "event_conditioned_return",
)
_CALIB_SCALAR_ADAPTATION_KEYS: tuple[str, ...] = (
    "adaptation_pre_change_auc",
    "adaptation_post_change_auc",
    "adaptation_lag_50pct",
    "adaptation_lag_75pct",
    "adaptation_lag_90pct",
    "regime_shift_episode",  # R7-A2: carry change-point index into calibration JSON
)
_CALIB_SCALAR_EVENT_KEYS: tuple[str, ...] = (
    "jackpot_event_rate",
    "catastrophe_event_rate",
    "hazard_hit_rate",
    "shortcut_risky_path_fraction",  # R7-A4: fraction of episodes using risky path
)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def discover_runs(
    run_root: Path,
    *,
    task_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Find all completed Phase II run directories and parse their manifests.

    Each returned dict contains:
        - ``run_dir``: absolute Path to the seed directory
        - ``run_json``: the full run.json dict
        - ``suite``: which suite the run belongs to
        - ``task``: task name
        - ``algorithm``: algorithm name
        - ``seed``: integer seed

    Parameters
    ----------
    run_root:
        Root of the raw results tree (e.g.
        ``results/weighted_lse_dp``).
    task_filter:
        Optional task name to restrict discovery to a single task.

    Returns
    -------
    list[dict]
        One entry per discovered run, sorted by (suite, task, algo, seed).
    """
    records: list[dict[str, Any]] = []

    for suite in _SUITES:
        dirs = find_run_dirs(
            run_root,
            phase="phase2",
            suite=suite,
            task=task_filter,
        )
        for run_dir in dirs:
            try:
                rj = load_run_json(run_dir)
            except Exception as exc:
                print(
                    f"  [WARN] skipping {run_dir}: cannot load run.json: {exc}",
                    file=sys.stderr,
                )
                continue

            # R5-3 / R6-1: keep raw task label as the grouping key so
            # *_pre_shift and *_post_shift DP runs remain in separate groups
            # (their calibration stats must NOT be averaged together).
            # canonical_task_family is preserved as metadata for downstream
            # code that needs to map back to the parent family name.
            raw_task = rj.get("task", "unknown")
            run_cfg = rj.get("config", {})
            canonical_family = (
                run_cfg.get("canonical_task_family")
                if isinstance(run_cfg, dict)
                else None
            )
            records.append({
                "run_dir": run_dir,
                "run_json": rj,
                "suite": suite,
                "task": raw_task,                        # grouping key unchanged
                "canonical_task_family": canonical_family or raw_task,
                "algorithm": rj.get("algorithm", "unknown"),
                "seed": rj.get("seed", -1),
            })

    # Deterministic ordering for reproducibility.
    records.sort(
        key=lambda r: (r["suite"], r["task"], r["algorithm"], r["seed"]),
    )
    return records


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

GroupKey = tuple[str, str, str]
"""(suite, task, algorithm)."""


def _group_key(rec: dict[str, Any]) -> GroupKey:
    return (rec["suite"], rec["task"], rec["algorithm"])


def group_runs(
    records: list[dict[str, Any]],
) -> dict[GroupKey, list[dict[str, Any]]]:
    """Group run records by ``(suite, task, algorithm)``.

    Returns
    -------
    dict[GroupKey, list[dict]]
        Mapping from group key to the list of run records in that group.
    """
    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[_group_key(rec)].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Safe loaders
# ---------------------------------------------------------------------------


def _load_metrics_safe(run_dir: Path) -> dict[str, Any] | None:
    """Load metrics.json, returning None on failure."""
    try:
        return load_metrics_json(run_dir)
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(
            f"  [WARN] cannot load metrics.json from {run_dir}: {exc}",
            file=sys.stderr,
        )
        return None


def _load_calibration_safe(run_dir: Path) -> dict[str, np.ndarray] | None:
    """Load calibration_stats.npz, returning None on failure."""
    calib_path = run_dir / "calibration_stats.npz"
    if not calib_path.is_file():
        return None
    try:
        return load_npz(calib_path)
    except Exception as exc:
        print(
            f"  [WARN] cannot load calibration_stats.npz from {run_dir}: {exc}",
            file=sys.stderr,
        )
        return None


def _load_curves_safe(run_dir: Path) -> dict[str, np.ndarray] | None:
    """Load curves.npz, returning None on failure (R4-5)."""
    curves_path = run_dir / "curves.npz"
    if not curves_path.is_file():
        return None
    try:
        return load_npz(curves_path)
    except Exception as exc:
        print(
            f"  [WARN] cannot load curves.npz from {run_dir}: {exc}",
            file=sys.stderr,
        )
        return None


def _load_transitions_safe(run_dir: Path) -> dict[str, np.ndarray] | None:
    """Load transitions.npz, returning None on failure."""
    trans_path = run_dir / "transitions.npz"
    if not trans_path.is_file():
        return None
    try:
        return load_npz(trans_path)
    except Exception as exc:
        print(
            f"  [WARN] cannot load transitions.npz from {run_dir}: {exc}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Per-group aggregation
# ---------------------------------------------------------------------------


def aggregate_group(
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate scalar metrics and calibration arrays across seeds.

    Parameters
    ----------
    group
        List of run records sharing the same (suite, task, algorithm).

    Returns
    -------
    dict
        ``"n_seeds"``: int,
        ``"seeds"``: list of ints,
        ``"scalar_metrics"``: dict of metric_name -> aggregation dict,
        ``"calibration"``: dict of array_name -> ndarray (stage-wise means),
                           or None if no calibration data,
        ``"tail_risk"``: dict of tail-risk scalar means (from run.json or
                         calibration_stats.npz), or None,
        ``"adaptation"``: dict of adaptation scalar means, or None,
        ``"event_rates"``: dict of event-rate scalar means, or None.
    """
    seeds: list[int] = []
    per_seed_metrics: list[dict[str, Any]] = []
    per_seed_calib: list[dict[str, np.ndarray]] = []
    per_seed_run_json: list[dict[str, Any]] = []
    per_seed_transitions: list[dict[str, np.ndarray]] = []
    per_seed_curves: list[dict[str, np.ndarray]] = []

    for rec in group:
        seeds.append(rec["seed"])
        per_seed_run_json.append(rec["run_json"])
        m = _load_metrics_safe(rec["run_dir"])
        if m is not None:
            per_seed_metrics.append(m)
        c = _load_calibration_safe(rec["run_dir"])
        if c is not None:
            per_seed_calib.append(c)
        cv = _load_curves_safe(rec["run_dir"])
        if cv is not None:
            per_seed_curves.append(cv)
        tr = _load_transitions_safe(rec["run_dir"])
        if tr is not None:
            per_seed_transitions.append(tr)

    # -- Aggregate scalar metrics ------------------------------------------
    scalar_agg: dict[str, Any] = {}
    if len(per_seed_metrics) >= 2:
        for key in _SCALAR_METRIC_KEYS:
            values = []
            for m in per_seed_metrics:
                v = m.get(key)
                if v is not None and isinstance(v, (int, float)):
                    values.append(float(v))
            if len(values) >= 2:
                arr = np.array(values, dtype=np.float64)
                agg = aggregate(arr)
                scalar_agg[key] = {
                    k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in agg.items()
                }
    elif len(per_seed_metrics) == 1:
        m = per_seed_metrics[0]
        for key in _SCALAR_METRIC_KEYS:
            v = m.get(key)
            if v is not None and isinstance(v, (int, float)):
                scalar_agg[key] = {"mean": float(v), "n_seeds": 1}

    # -- Aggregate calibration arrays (stage-wise mean across seeds) -------
    calib_agg: dict[str, np.ndarray] | None = None
    if per_seed_calib:
        common_keys = set(per_seed_calib[0].keys())
        for c in per_seed_calib[1:]:
            common_keys &= set(c.keys())
        common_keys.discard("_schema")

        calib_agg = {}
        for arr_key in sorted(common_keys):
            arrays = []
            for c in per_seed_calib:
                a = c[arr_key]
                if a.ndim == 1:
                    arrays.append(a)
            if arrays:
                try:
                    stacked = np.stack(arrays, axis=0)
                    calib_agg[arr_key] = np.mean(stacked, axis=0)
                except ValueError:
                    pass

    # -- Aggregate tail-risk scalars from calibration npz ------------------
    tail_risk = _aggregate_scalar_block(per_seed_calib, _CALIB_SCALAR_TAIL_KEYS)
    # Also try run.json tail_risk_metrics as fallback/override
    tail_risk = _merge_run_json_scalars(
        tail_risk, per_seed_run_json, "tail_risk_metrics",
        _CALIB_SCALAR_TAIL_KEYS,
    )

    # -- Aggregate adaptation scalars from calibration npz -----------------
    adaptation = _aggregate_scalar_block(
        per_seed_calib, _CALIB_SCALAR_ADAPTATION_KEYS,
    )
    adaptation = _merge_run_json_scalars(
        adaptation, per_seed_run_json, "adaptation_metrics",
        _CALIB_SCALAR_ADAPTATION_KEYS,
    )

    # -- Aggregate event-rate scalars from calibration npz -----------------
    event_rates = _aggregate_scalar_block(
        per_seed_calib, _CALIB_SCALAR_EVENT_KEYS,
    )

    # -- Stacked per-seed margin arrays (not averaged) for quantile accuracy --
    _MARGIN_SEED_KEYS: tuple[str, ...] = (
        "aligned_positive_mean", "aligned_negative_mean",
    )
    calib_stacked: dict[str, np.ndarray] | None = None
    if per_seed_calib:
        _stk: dict[str, np.ndarray] = {}
        for mk in _MARGIN_SEED_KEYS:
            arrays = [
                c[mk] for c in per_seed_calib
                if mk in c and c[mk].ndim == 1
            ]
            if arrays:
                try:
                    _stk[mk] = np.stack(arrays, axis=0)  # (n_seeds, H+1)
                except ValueError:
                    pass
        if _stk:
            calib_stacked = _stk

    # -- Per-stage event-conditioned margins from transitions.npz ----------
    event_conditioned_stagewise: dict[str, Any] | None = None
    if per_seed_transitions:
        event_conditioned_stagewise = _compute_event_conditioned_stagewise(
            per_seed_transitions,
        )

    # -- Per-stage margin quantiles from raw per-transition data (R4-3) -----
    # Pool all margin_beta0 values per stage across seeds, then compute
    # quantiles from the pooled per-transition distribution rather than from
    # per-seed means.  This preserves the full tail structure that Phase III
    # needs to calibrate against.
    margin_quantiles_from_transitions: dict[str, Any] | None = (
        _compute_margin_quantiles_from_transitions(per_seed_transitions)
    )

    # -- True empirical_r_max from raw reward values (R4-4) -----------------
    # Use max(|reward|) over all observed transitions, not a moment heuristic.
    empirical_r_max_from_transitions: float | None = None
    for trans in per_seed_transitions:
        rwd = trans.get("reward")
        if rwd is not None and len(rwd) > 0:
            cand = float(np.nanmax(np.abs(rwd)))
            if empirical_r_max_from_transitions is None or cand > empirical_r_max_from_transitions:
                empirical_r_max_from_transitions = cand

    # -- Pool episode returns split by event flag (MAJOR R3-3) --------------
    # run_phase2_rl writes episode_returns_noevent / episode_returns_event to
    # metrics.json for RL runs.  Pool across seeds for the calibration JSON.
    # Also pool ALL eval episode returns (regardless of event) so the calibration
    # JSON can expose the full return distribution for figure 11.1.2.
    pooled_returns_noevent: list[float] = []
    pooled_returns_event: list[float] = []
    pooled_all_episode_returns: list[float] = []
    for m in per_seed_metrics:
        ne = m.get("episode_returns_noevent")
        ev = m.get("episode_returns_event")
        er = m.get("episode_returns")
        if isinstance(ne, list):
            pooled_returns_noevent.extend(ne)
        if isinstance(ev, list):
            pooled_returns_event.extend(ev)
        if isinstance(er, list):
            pooled_all_episode_returns.extend(er)

    # -- Aggregate learning curves from curves.npz (R4-5) -------------------
    # Pool actual checkpoint returns across seeds instead of synthesising them
    # from per-stage calibration means.
    curves_agg: dict[str, Any] | None = None
    if per_seed_curves:
        # Check which type of curves are present (RL vs DP).
        sample_curves = per_seed_curves[0]
        if "disc_return_mean" in sample_curves and "checkpoints" in sample_curves:
            # RL learning curves: mean over seeds at each checkpoint.
            ckpt_arrays = [c["checkpoints"] for c in per_seed_curves if "checkpoints" in c]
            ret_arrays = [c["disc_return_mean"] for c in per_seed_curves if "disc_return_mean" in c]
            std_arrays = [c.get("disc_return_std") for c in per_seed_curves if "disc_return_std" in c]
            success_arrays = [c.get("success_rate") for c in per_seed_curves if "success_rate" in c]

            # Align on the common checkpoint sequence (use first seed as reference).
            ref_ckpts = ckpt_arrays[0] if ckpt_arrays else np.array([], dtype=np.int64)
            mean_ret = np.nanmean(
                np.stack([r[:len(ref_ckpts)] for r in ret_arrays if len(r) >= len(ref_ckpts)], axis=0),
                axis=0,
            ) if ret_arrays else None
            std_ret = (
                np.nanmean(
                    np.stack([r[:len(ref_ckpts)] for r in std_arrays if r is not None and len(r) >= len(ref_ckpts)], axis=0),
                    axis=0,
                ) if any(r is not None for r in std_arrays) else None
            )
            mean_success = (
                np.nanmean(
                    np.stack([r[:len(ref_ckpts)] for r in success_arrays if r is not None and len(r) >= len(ref_ckpts)], axis=0),
                    axis=0,
                ) if success_arrays else None
            )

            if mean_ret is not None:
                curves_agg = {
                    "steps": ref_ckpts.tolist(),
                    "mean_return": _nan_safe_list(mean_ret),
                    "std_return": _nan_safe_list(std_ret) if std_ret is not None else [],
                    "success_rate": _nan_safe_list(mean_success) if mean_success is not None else [],
                    # alias for fig_adaptation_plots
                    "episode_returns": _nan_safe_list(mean_ret),
                }
        elif "bellman_residual" in sample_curves and "sweep_index" in sample_curves:
            # DP convergence curves: mean over seeds.
            sweep_arrays = [c["sweep_index"] for c in per_seed_curves if "sweep_index" in c]
            res_arrays = [c["bellman_residual"] for c in per_seed_curves if "bellman_residual" in c]
            sup_arrays = [c.get("supnorm_to_exact") for c in per_seed_curves if "supnorm_to_exact" in c]

            if res_arrays and sweep_arrays:
                min_len = min(len(a) for a in res_arrays)
                mean_res = np.nanmean(np.stack([a[:min_len] for a in res_arrays], axis=0), axis=0)
                mean_sup = None
                if any(a is not None for a in sup_arrays):
                    valid_sup = [a for a in sup_arrays if a is not None and len(a) >= min_len]
                    if valid_sup:
                        mean_sup = np.nanmean(np.stack([a[:min_len] for a in valid_sup], axis=0), axis=0)

                curves_agg = {
                    "steps": sweep_arrays[0][:min_len].tolist(),
                    "bellman_residual": _nan_safe_list(mean_res),
                    "supnorm_to_exact": _nan_safe_list(mean_sup) if mean_sup is not None else [],
                    # alias for generic plot scripts
                    "mean_return": _nan_safe_list(-mean_res),  # negated residual as return proxy
                    "episode_returns": _nan_safe_list(-mean_res),
                }

    # -- Per-seed episode returns (for adaptation figures — R7-1/R7-3 fix) ---
    # Collected from metrics.json so the adaptation figure can plot a
    # per-episode return curve in episode units (not checkpoint units).
    episode_returns_by_seed: dict[str, list[float]] = {}
    for i, m in enumerate(per_seed_metrics):
        er = m.get("episode_returns")
        if isinstance(er, list) and er:
            seed_key = str(seeds[i]) if i < len(seeds) else str(i)
            episode_returns_by_seed[seed_key] = er

    # -- Per-state visitation counts from transitions (R7-2 fix) --------------
    # Aggregated across seeds by summing visit counts per state.
    visitation_counts: list[int] | None = None
    if per_seed_transitions:
        state_arrays = [
            t["state"].astype(int)
            for t in per_seed_transitions
            if "state" in t and len(t["state"]) > 0
        ]
        if state_arrays:
            all_states = np.concatenate(state_arrays)
            n_states = int(all_states.max()) + 1
            counts = np.bincount(all_states, minlength=n_states)
            visitation_counts = counts.tolist()

    return {
        "n_seeds": len(seeds),
        "seeds": sorted(seeds),
        "scalar_metrics": scalar_agg,
        "calibration": calib_agg,
        "calibration_stacked": calib_stacked,
        "tail_risk": tail_risk,
        "adaptation": adaptation,
        "event_rates": event_rates,
        "event_conditioned_stagewise": event_conditioned_stagewise,
        "episode_returns_noevent": pooled_returns_noevent,
        "episode_returns_event": pooled_returns_event,
        "all_episode_returns": pooled_all_episode_returns,
        "margin_quantiles_from_transitions": margin_quantiles_from_transitions,
        "empirical_r_max_from_transitions": empirical_r_max_from_transitions,
        "curves_from_checkpoints": curves_agg,
        "episode_returns_by_seed": episode_returns_by_seed,
        "visitation_counts": visitation_counts,
    }


def _compute_event_conditioned_stagewise(
    per_seed_transitions: list[dict[str, np.ndarray]],
) -> dict[str, Any] | None:
    """Compute per-stage event-conditioned margin statistics from transitions.

    For each horizon stage ``t``, computes the mean and count of
    ``margin_beta0`` values conditioned on a stress event occurring at that
    transition.  Aggregates across seeds before returning.

    Returns ``None`` if no usable transition data is found.
    """
    # Determine which event key is present (only one per task family).
    event_keys = (
        "catastrophe_event", "jackpot_event", "hazard_cell_hit",
        "regime_post_change", "shortcut_action_taken",
    )

    # Collect per-stage event-conditioned margin samples across all seeds.
    # key: stage index → list of margin values (filtered by event flag)
    stage_event_margins: dict[int, list[float]] = {}
    found_event_key: str | None = None

    for trans in per_seed_transitions:
        margin = trans.get("margin_beta0")
        t_arr = trans.get("t")
        if margin is None or t_arr is None:
            continue

        # Find available event flag.
        event_flag: np.ndarray | None = None
        for ek in event_keys:
            ef = trans.get(ek)
            if ef is not None and ef.any():
                event_flag = ef.astype(bool)
                found_event_key = ek
                break

        if event_flag is None:
            continue

        for t_val, m_val, ev_val in zip(
            t_arr.astype(int), margin, event_flag,
        ):
            if ev_val:
                if t_val not in stage_event_margins:
                    stage_event_margins[t_val] = []
                stage_event_margins[t_val].append(float(m_val))

    if not stage_event_margins:
        return None

    stages_sorted = sorted(stage_event_margins.keys())
    margin_means: list[float | None] = []
    margin_counts: list[int] = []
    for t_val in stages_sorted:
        vals = stage_event_margins[t_val]
        margin_means.append(float(np.mean(vals)) if vals else None)
        margin_counts.append(len(vals))

    return {
        "stages": stages_sorted,
        "event_conditioned_margin_mean": margin_means,
        "event_conditioned_margin_count": margin_counts,
        "event_key": found_event_key,
    }


def _compute_margin_quantiles_from_transitions(
    per_seed_transitions: list[dict[str, np.ndarray]],
) -> dict[str, Any] | None:
    """Compute per-stage margin quantiles from raw per-transition margins (R4-3/R5-2).

    Pools all ``margin_beta0`` values at each stage ``t`` across seeds, then
    computes quantiles over the pooled per-transition distribution.  This
    preserves the full tail structure rather than computing percentiles of
    per-seed means.

    Additionally computes aligned-positive and aligned-negative quantiles from
    the true per-transition sign split (R5-2):

    - ``aligned_positive`` at transition i = max(margin_beta0[i], 0)
    - ``aligned_negative`` at transition i = max(-margin_beta0[i], 0)

    These are stored as ``pos_q{05,25,50,75,95}`` and ``neg_q{05,25,50,75,95}``
    in the returned dict so downstream code can emit faithful Phase III
    calibration quantiles without fabrication.

    Returns ``None`` if no transition data has ``margin_beta0`` and ``t``.
    """
    # stage_margins: stage_index → list of raw margin_beta0 values
    stage_margins: dict[int, list[float]] = {}

    for trans in per_seed_transitions:
        margin = trans.get("margin_beta0")
        t_arr = trans.get("t")
        if margin is None or t_arr is None:
            continue
        for t_val, m_val in zip(t_arr.astype(int), margin):
            if t_val not in stage_margins:
                stage_margins[t_val] = []
            stage_margins[t_val].append(float(m_val))

    if not stage_margins:
        return None

    stages_sorted = sorted(stage_margins.keys())
    q05, q25, q50, q75, q95 = [], [], [], [], []
    pos_q05, pos_q25, pos_q50, pos_q75, pos_q95 = [], [], [], [], []
    neg_q05, neg_q25, neg_q50, neg_q75, neg_q95 = [], [], [], [], []
    counts = []
    for t_val in stages_sorted:
        vals = np.array(stage_margins[t_val], dtype=np.float64)
        # Raw margin quantiles
        q05.append(float(np.nanpercentile(vals, 5)))
        q25.append(float(np.nanpercentile(vals, 25)))
        q50.append(float(np.nanpercentile(vals, 50)))
        q75.append(float(np.nanpercentile(vals, 75)))
        q95.append(float(np.nanpercentile(vals, 95)))
        # Aligned-positive: max(m, 0) per transition
        pos_vals = np.maximum(vals, 0.0)
        pos_q05.append(float(np.nanpercentile(pos_vals, 5)))
        pos_q25.append(float(np.nanpercentile(pos_vals, 25)))
        pos_q50.append(float(np.nanpercentile(pos_vals, 50)))
        pos_q75.append(float(np.nanpercentile(pos_vals, 75)))
        pos_q95.append(float(np.nanpercentile(pos_vals, 95)))
        # Aligned-negative: max(-m, 0) per transition
        neg_vals = np.maximum(-vals, 0.0)
        neg_q05.append(float(np.nanpercentile(neg_vals, 5)))
        neg_q25.append(float(np.nanpercentile(neg_vals, 25)))
        neg_q50.append(float(np.nanpercentile(neg_vals, 50)))
        neg_q75.append(float(np.nanpercentile(neg_vals, 75)))
        neg_q95.append(float(np.nanpercentile(neg_vals, 95)))
        counts.append(len(vals))

    return {
        "stages": stages_sorted,
        "q05": q05, "q25": q25, "q50": q50, "q75": q75, "q95": q95,
        "pos_q05": pos_q05, "pos_q25": pos_q25, "pos_q50": pos_q50,
        "pos_q75": pos_q75, "pos_q95": pos_q95,
        "neg_q05": neg_q05, "neg_q25": neg_q25, "neg_q50": neg_q50,
        "neg_q75": neg_q75, "neg_q95": neg_q95,
        "counts": counts,
    }


def _aggregate_scalar_block(
    per_seed_calib: list[dict[str, np.ndarray]],
    keys: tuple[str, ...],
) -> dict[str, float] | None:
    """Average scalar (shape (1,)) calibration fields across seeds.

    Returns None if no valid data is found for any key.
    """
    if not per_seed_calib:
        return None

    result: dict[str, float] = {}
    any_valid = False
    for key in keys:
        values: list[float] = []
        for c in per_seed_calib:
            arr = c.get(key)
            if arr is not None and arr.size >= 1:
                val = float(arr.flat[0])
                if not np.isnan(val):
                    values.append(val)
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            any_valid = True
        else:
            result[f"{key}_mean"] = None  # type: ignore[assignment]

    return result if any_valid else None


def _merge_run_json_scalars(
    block: dict[str, float] | None,
    run_jsons: list[dict[str, Any]],
    json_key: str,
    calib_keys: tuple[str, ...],
) -> dict[str, float] | None:
    """Merge scalar metrics from run.json into an existing block.

    If the calibration npz block already has non-null values, those are
    kept. Values from run.json serve as fallback for keys that are
    missing or null in the npz-derived block.
    """
    # Collect values from run.json
    per_seed_values: dict[str, list[float]] = defaultdict(list)
    for rj in run_jsons:
        sub = rj.get(json_key, {})
        if not isinstance(sub, dict):
            continue
        for key in calib_keys:
            v = sub.get(key)
            if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                per_seed_values[key].append(float(v))

    if not per_seed_values:
        return block

    if block is None:
        block = {}

    any_valid = False
    for key in calib_keys:
        mean_key = f"{key}_mean"
        # Only fill if the npz block has no valid value
        if block.get(mean_key) is None and key in per_seed_values:
            vals = per_seed_values[key]
            if vals:
                block[mean_key] = float(np.mean(vals))
                any_valid = True

        if block.get(mean_key) is not None:
            any_valid = True

    return block if any_valid else None


# ---------------------------------------------------------------------------
# Calibration JSON builder (Task 26, spec section 12)
# ---------------------------------------------------------------------------


def build_calibration_json(
    task_family: str,
    groups: dict[GroupKey, dict[str, Any]],
    task_config: dict[str, Any] | None = None,
    raw_tasks: set[str] | None = None,
) -> dict[str, Any]:
    """Build the calibration JSON document for a single task family.

    Averages calibration statistics across all algorithms for the given
    task to produce a single task-level calibration profile.

    Parameters
    ----------
    task_family:
        Canonical task name (e.g. ``"chain_regime_shift"``).  Used as the
        calibration document's ``task_family`` field and for sign lookup.
    groups:
        Already-aggregated groups keyed by ``(suite, task, algorithm)``.
        Only entries whose raw task label is in ``raw_tasks`` are consumed.
    task_config:
        Optional task configuration from ``paper_suite.json``. Used to
        extract ``gamma``, reward range, etc.
    raw_tasks:
        Set of raw task labels (group keys) to include.  Defaults to
        ``{task_family}`` for non-regime-shift families.  Pass all
        suffixed variants (e.g. ``{"chain_regime_shift",
        "chain_regime_shift_pre_shift", "chain_regime_shift_post_shift"}``)
        to merge pre/post DP groups into one calibration document (R7-A1).

    Returns
    -------
    dict
        The calibration JSON document per spec section 12.
    """
    if raw_tasks is None:
        raw_tasks = {task_family}

    # Filter groups for this task family (match any raw task label in raw_tasks)
    task_groups: list[dict[str, Any]] = []
    for (suite, task, algo), agg in groups.items():
        if task in raw_tasks:
            task_groups.append(agg)

    n_seeds_total = max((g["n_seeds"] for g in task_groups), default=0)

    # Extract nominal gamma and reward range from config
    gamma = 0.99  # default
    reward_range = [-1.0, 1.0]
    if task_config is not None:
        gamma = task_config.get("gamma", 0.99)
        # Infer reward range from config fields
        r_vals = [0.0]
        for rkey in (
            "jackpot_reward", "catastrophe_reward", "goal_reward",
            "hazard_reward", "bonus_reward",
        ):
            v = task_config.get(rkey)
            if v is not None:
                r_vals.append(float(v))
        reward_range = [min(r_vals), max(max(r_vals), 1.0)]

    # Collect calibration dicts for stagewise arrays.
    # User decision (MINOR R3-5): prefer exact DP algorithm groups
    # (VI / PI / MPI / AsyncVI / PE) so the Phase III calibration
    # reference is the best available planner, not an RL average.
    # Fall back to all groups if no DP group produced calibration data.
    _DP_ALGO_NAMES: frozenset[str] = frozenset(
        {"vi", "pi", "mpi", "asyncvi", "pe",
         "value_iteration", "policy_iteration",
         "modified_policy_iteration", "async_value_iteration",
         "policy_evaluation"}
    )
    calib_dicts_dp: list[dict[str, np.ndarray]] = []
    calib_dicts_all: list[dict[str, np.ndarray]] = []
    for (suite, task, algo), agg in groups.items():
        if task != task_family:
            continue
        c = agg.get("calibration")
        if c is None:
            continue
        calib_dicts_all.append(c)
        if algo.lower() in _DP_ALGO_NAMES:
            calib_dicts_dp.append(c)

    # Use DP-only dicts when available; fall back to all groups.
    calib_dicts: list[dict[str, np.ndarray]] = (
        calib_dicts_dp if calib_dicts_dp else calib_dicts_all
    )

    # Build stagewise block
    stagewise: dict[str, Any] | None = None
    empirical_r_max: float | None = None

    if calib_dicts:
        stagewise = {}
        for key in _STAGEWISE_KEYS:
            arrays = []
            for c in calib_dicts:
                a = c.get(key)
                if a is not None and a.ndim == 1:
                    arrays.append(a)
            if arrays:
                try:
                    stacked = np.stack(arrays, axis=0)
                    mean_arr = np.mean(stacked, axis=0)
                    # Use key + "_mean" for non-index fields
                    if key == "stage":
                        stagewise["stage"] = mean_arr.astype(int).tolist()
                    elif key == "count":
                        stagewise["count_mean"] = mean_arr.tolist()
                    else:
                        stagewise[f"{key}_mean"] = _nan_safe_list(mean_arr)
                except ValueError:
                    pass

        # -- Per-stage margin quantiles (spec §12) ----------------------------
        # Priority (R4-3): use raw per-transition margin_beta0 from
        # transitions.npz (stored in group["margin_quantiles_from_transitions"]).
        # These are quantiles of the actual margin distribution at each stage,
        # not of per-seed means.  Fall back to calibration_stacked (seed-level
        # means) only when no transition data is available.

        # Collect per-stage quantiles from transitions-derived data.
        trans_mq_list: list[dict[str, Any]] = [
            g["margin_quantiles_from_transitions"]
            for g in task_groups
            if g.get("margin_quantiles_from_transitions") is not None
        ]

        if trans_mq_list:
            # Pool per-stage quantiles across groups by averaging.
            # pos_* and neg_* keys are computed from true per-transition sign
            # splits (R5-2 fix): aligned_positive[i] = max(m[i], 0),
            # aligned_negative[i] = max(-m[i], 0).  These are available when
            # _compute_margin_quantiles_from_transitions produced R5-2 output.
            all_stages: list[int] = sorted(
                {s for mq in trans_mq_list for s in mq["stages"]}
            )

            def _pool_stage_key(key: str) -> list[float | None]:
                result: list[float | None] = []
                for s in all_stages:
                    vals = [
                        mq[key][mq["stages"].index(s)]
                        for mq in trans_mq_list
                        if s in mq["stages"] and key in mq
                    ]
                    result.append(float(np.mean(vals)) if vals else None)
                return result

            # Raw margin quantiles (full distribution, signed) — used as the
            # top-level margin_quantiles alias for Phase II figures (R6-2).
            stagewise["raw_margin_quantiles"] = {
                "q05": _pool_stage_key("q05"),
                "q25": _pool_stage_key("q25"),
                "q50": _pool_stage_key("q50"),
                "q75": _pool_stage_key("q75"),
                "q95": _pool_stage_key("q95"),
            }

            # Emit aligned pos/neg quantiles from true per-transition splits
            # (R5-2).  Fall through to the seed-level fallback only when the
            # per-transition keys are absent (legacy data pre-R5-2).
            has_aligned_split = any("pos_q05" in mq for mq in trans_mq_list)
            if has_aligned_split:
                stagewise["pos_margin_quantiles"] = {
                    "q05": _pool_stage_key("pos_q05"),
                    "q25": _pool_stage_key("pos_q25"),
                    "q50": _pool_stage_key("pos_q50"),
                    "q75": _pool_stage_key("pos_q75"),
                    "q95": _pool_stage_key("pos_q95"),
                }
                stagewise["neg_margin_quantiles"] = {
                    "q05": _pool_stage_key("neg_q05"),
                    "q25": _pool_stage_key("neg_q25"),
                    "q50": _pool_stage_key("neg_q50"),
                    "q75": _pool_stage_key("neg_q75"),
                    "q95": _pool_stage_key("neg_q95"),
                }
            else:
                # Legacy trans_mq_list without per-transition sign split.
                # Fall back to seed-level aligned_*_mean arrays for quantiles.
                for margin_key, out_key in (
                    ("aligned_positive_mean", "pos_margin_quantiles"),
                    ("aligned_negative_mean", "neg_margin_quantiles"),
                ):
                    seed_pools_legacy: list[np.ndarray] = []
                    for g in task_groups:
                        cs = g.get("calibration_stacked") or {}
                        arr = cs.get(margin_key)
                        if arr is not None and arr.ndim == 2:
                            seed_pools_legacy.append(arr)
                    if seed_pools_legacy:
                        try:
                            pooled_legacy = np.concatenate(seed_pools_legacy, axis=0)
                            stagewise[out_key] = {
                                "q05": _nan_safe_list(np.nanpercentile(pooled_legacy, 5, axis=0)),
                                "q25": _nan_safe_list(np.nanpercentile(pooled_legacy, 25, axis=0)),
                                "q50": _nan_safe_list(np.nanpercentile(pooled_legacy, 50, axis=0)),
                                "q75": _nan_safe_list(np.nanpercentile(pooled_legacy, 75, axis=0)),
                                "q95": _nan_safe_list(np.nanpercentile(pooled_legacy, 95, axis=0)),
                            }
                        except ValueError:
                            pass
        else:
            # Fallback: pool calibration_stacked seed-level arrays.
            for margin_key, out_key in (
                ("aligned_positive_mean", "pos_margin_quantiles"),
                ("aligned_negative_mean", "neg_margin_quantiles"),
            ):
                seed_pools: list[np.ndarray] = []
                for g in task_groups:
                    cs = g.get("calibration_stacked") or {}
                    arr = cs.get(margin_key)
                    if arr is not None and arr.ndim == 2:
                        seed_pools.append(arr)

                if seed_pools:
                    try:
                        pooled = np.concatenate(seed_pools, axis=0)
                        stagewise[out_key] = {
                            "q05": _nan_safe_list(np.nanpercentile(pooled, 5, axis=0)),
                            "q25": _nan_safe_list(np.nanpercentile(pooled, 25, axis=0)),
                            "q50": _nan_safe_list(np.nanpercentile(pooled, 50, axis=0)),
                            "q75": _nan_safe_list(np.nanpercentile(pooled, 75, axis=0)),
                            "q95": _nan_safe_list(np.nanpercentile(pooled, 95, axis=0)),
                        }
                    except ValueError:
                        pass

        # -- empirical_r_max from true observed rewards (R4-4) ----------------
        # Prefer max(|reward|) aggregated from raw transitions.npz over the
        # moment-based heuristic (mean ± 2*std), which understates rare spikes.
        for g in task_groups:
            rmax = g.get("empirical_r_max_from_transitions")
            if rmax is not None:
                if empirical_r_max is None or rmax > empirical_r_max:
                    empirical_r_max = rmax

        # Fallback moment heuristic (used only when no transition data exists).
        if empirical_r_max is None:
            for c in calib_dicts:
                r_mean = c.get("reward_mean")
                r_std = c.get("reward_std")
                if r_mean is not None:
                    abs_max = float(np.nanmax(np.abs(r_mean)))
                    if empirical_r_max is None or abs_max > empirical_r_max:
                        empirical_r_max = abs_max
                if r_std is not None and r_mean is not None:
                    upper = float(np.nanmax(r_mean + 2.0 * r_std))
                    lower = float(np.nanmin(r_mean - 2.0 * r_std))
                    candidate = max(abs(upper), abs(lower))
                    if empirical_r_max is None or candidate > empirical_r_max:
                        empirical_r_max = candidate

    # Aggregate tail_risk across algorithms
    tail_risk = _merge_algo_scalar_blocks(
        task_groups, "tail_risk",
    )

    # Aggregate adaptation across algorithms
    adaptation = _merge_algo_scalar_blocks(
        task_groups, "adaptation",
    )

    # Aggregate event_rates across algorithms
    event_rates = _merge_algo_scalar_blocks(
        task_groups, "event_rates",
    )

    # Determine recommended_task_sign from family semantics (spec §12).
    recommended_task_sign = _determine_task_sign(task_family, stagewise)

    # -- Event-conditioned margin block (spec §12) --------------------------
    # Pool per-stage event-conditioned margin stats from all algorithm groups.
    event_cond_return: float | None = None
    if tail_risk is not None:
        event_cond_return = tail_risk.get("event_conditioned_return_mean")

    # Collect per-stage data from groups (computed in aggregate_group via
    # transitions.npz loading).
    ecs_list: list[dict[str, Any]] = [
        g["event_conditioned_stagewise"]
        for g in task_groups
        if g.get("event_conditioned_stagewise") is not None
    ]
    ecs_merged: dict[str, Any] | None = None
    if ecs_list:
        # Use first available (all task groups share the same task so event
        # keys match); merge by averaging margin means per stage.
        all_stages: set[int] = set()
        for ecs in ecs_list:
            all_stages.update(ecs["stages"])
        stages_sorted = sorted(all_stages)

        merged_means: list[float | None] = []
        merged_counts: list[int] = []
        for t_val in stages_sorted:
            # Count-weighted merge (R4-6): weight each group's mean by its
            # event count so that high-event-count groups dominate sparse-
            # event groups, giving a less noisy estimate of the true margin.
            weighted_sum: float = 0.0
            total_cnt: int = 0
            for ecs in ecs_list:
                try:
                    idx = ecs["stages"].index(t_val)
                    m = ecs["event_conditioned_margin_mean"][idx]
                    c_ = ecs["event_conditioned_margin_count"][idx]
                    if m is not None and c_ > 0:
                        weighted_sum += float(m) * int(c_)
                        total_cnt += int(c_)
                except (ValueError, IndexError):
                    pass
            merged_means.append(weighted_sum / total_cnt if total_cnt > 0 else None)
            merged_counts.append(total_cnt)

        ecs_merged = {
            "stages": stages_sorted,
            "event_conditioned_margin_mean": merged_means,
            "event_conditioned_margin_count": merged_counts,
            "event_key": ecs_list[0].get("event_key"),
        }

    event_conditioned_margins: dict[str, Any] = {
        "event_conditioned_return": event_cond_return,
    }
    if ecs_merged is not None:
        event_conditioned_margins["stagewise"] = ecs_merged
    else:
        event_conditioned_margins["stagewise"] = None

    # -- Figure-compat top-level keys (BLOCKER-2 fix, MAJOR-R3-3 fix) -------
    # These allow make_phase2_figures.py to load return and margin data
    # without re-reading nested stagewise structures.

    # base_returns / stress_returns: per-episode eval returns from the same
    # eval protocol, pooled across seeds.
    #
    # base_returns  = episodes where NO stress event fired (noevent split).
    #                 Represents the "normal" return distribution.
    # stress_returns = ALL eval episode returns from the stress task.
    #                 Using all returns (not just event-conditioned ones)
    #                 shows the full distribution — rare jackpot / catastrophe
    #                 returns appear as a visible tail rather than being
    #                 absent because event_returns_event is near-empty when
    #                 events are rare (e.g., jackpot_prob=0.05 over 500 eval
    #                 episodes gives ~25 events vs 5036 base episodes).
    base_returns_pool: list[float] = []
    stress_returns_pool: list[float] = []
    for g in task_groups:
        ne = g.get("episode_returns_noevent")
        er = g.get("all_episode_returns")
        if isinstance(ne, list):
            base_returns_pool.extend(ne)
        if isinstance(er, list):
            stress_returns_pool.extend(er)

    base_returns_list: list[float] = [float(v) for v in base_returns_pool]
    stress_returns_list: list[float] = [float(v) for v in stress_returns_pool]

    # margin_quantiles: top-level alias for the RAW margin distribution (R6-2).
    # Must use the full margin_beta0 quantiles — NOT pos_margin_quantiles —
    # so the negative tail (catastrophe/hazard tasks) is preserved and Phase III
    # schedule calibration sees the correct q05/q95 ribbon.
    margin_quantiles_top: dict[str, Any] | None = None
    if stagewise is not None:
        stage_vals = stagewise.get("stage")
        # Prefer trans_mq_list-derived raw quantiles (q05..q95 of full margin_beta0).
        # These are stored in the first available trans_mq entry for this call context.
        # Fall back to pos_margin_quantiles only if raw quantiles are unavailable.
        raw_mq = stagewise.get("raw_margin_quantiles")
        fallback_q = stagewise.get("pos_margin_quantiles")
        chosen_q = raw_mq if raw_mq is not None else fallback_q
        if chosen_q is not None:
            n_q = len(chosen_q.get("q50", []))
            n_s = len(stage_vals) if isinstance(stage_vals, list) else n_q
            # Align stages and quantile arrays to the shorter of the two.
            # stagewise["stage"] includes the terminal step (len = horizon+1)
            # while raw_margin_quantiles has horizon entries → clip stages.
            # For grid tasks, _compute_margin_quantiles_from_transitions can
            # produce more quantile entries than stagewise stages → clip quantiles.
            n_align = min(n_q, n_s)
            stages_for_q = (
                stage_vals[:n_align] if isinstance(stage_vals, list)
                else list(range(n_align))
            )
            margin_quantiles_top = {
                "stages": stages_for_q,
                "q05": (chosen_q.get("q05") or [])[:n_align],
                "q50": (chosen_q.get("q50") or [])[:n_align],
                "q95": (chosen_q.get("q95") or [])[:n_align],
            }

    doc: dict[str, Any] = {
        "task_family": task_family,
        "schema_version": _CALIBRATION_SCHEMA_VERSION,
        "nominal_gamma": gamma,
        "reward_range": reward_range,
        "empirical_r_max": empirical_r_max,
        "n_seeds": n_seeds_total,
        "stagewise": stagewise,
        "tail_risk": tail_risk,
        "adaptation": adaptation,
        "event_rates": event_rates,
        "event_conditioned_margins": event_conditioned_margins,
        "recommended_task_sign": recommended_task_sign,
        # Figure-compat keys (loaded by make_phase2_figures.py):
        "base_returns": base_returns_list,
        "stress_returns": stress_returns_list,
        "margin_quantiles": margin_quantiles_top,
    }

    return doc


def _nan_safe_list(arr: np.ndarray) -> list[float | None]:
    """Convert numpy array to list, replacing NaN with None for JSON."""
    result: list[float | None] = []
    for v in arr.flat:
        if np.isnan(v):
            result.append(None)
        else:
            result.append(float(v))
    return result


def _merge_algo_scalar_blocks(
    task_groups: list[dict[str, Any]],
    block_key: str,
) -> dict[str, float | None] | None:
    """Average a scalar block (tail_risk, adaptation, event_rates) across
    algorithm groups for a single task family.
    """
    blocks: list[dict[str, float]] = []
    for g in task_groups:
        b = g.get(block_key)
        if b is not None:
            blocks.append(b)

    if not blocks:
        return None

    # Collect all keys
    all_keys: set[str] = set()
    for b in blocks:
        all_keys.update(b.keys())

    result: dict[str, float | None] = {}
    any_valid = False
    for key in sorted(all_keys):
        values = [
            b[key] for b in blocks
            if key in b and b[key] is not None
        ]
        if values:
            result[key] = float(np.mean(values))
            any_valid = True
        else:
            result[key] = None

    return result if any_valid else None


# Explicit per-family sign: +1 for jackpot/positive-shock families,
# -1 for catastrophe/hazard families.  Source of truth is family semantics,
# not data heuristics (spec §12).
_TASK_FAMILY_SIGNS: dict[str, int] = {
    "chain_sparse_long":  1,
    "chain_jackpot":      1,
    "chain_catastrophe": -1,
    "chain_regime_shift": 1,
    "grid_sparse_goal":   1,
    "grid_hazard":       -1,
    "grid_regime_shift":  1,
    "taxi_bonus_shock":   1,
}


def _determine_task_sign(
    task_family: str,
    stagewise: dict[str, Any] | None = None,  # kept for validation only
) -> int:
    """Return the calibration sign for *task_family*.

    The sign is derived from family semantics, not data heuristics.
    Raises ``ValueError`` for unknown families so misconfiguration is
    caught immediately rather than silently defaulting to +1.

    Spec §12: one integer sign per experiment family.
    ``stagewise`` is accepted but used only for optional validation logging.

    Regime-shift DP groups carry a ``_pre_shift`` / ``_post_shift`` suffix on
    the task label (R5-3: directory uniqueness).  Strip the suffix before
    lookup so both phases resolve to the canonical family sign.
    """
    canonical = task_family
    for suffix in ("_pre_shift", "_post_shift"):
        if task_family.endswith(suffix):
            canonical = task_family[: -len(suffix)]
            break
    sign = _TASK_FAMILY_SIGNS.get(canonical)
    if sign is None:
        raise ValueError(
            f"_determine_task_sign: unknown task_family={task_family!r} "
            f"(canonical={canonical!r}). "
            f"Add it to _TASK_FAMILY_SIGNS with the correct sign."
        )
    return sign


def _canonical_family_from_task(raw_task: str) -> str:
    """Strip ``_pre_shift`` / ``_post_shift`` suffix to get canonical family name.

    Regime-shift DP runs are stored under suffixed task labels for directory
    uniqueness (R5-3).  This function recovers the canonical name used in
    calibration JSON filenames and ``_TASK_FAMILY_SIGNS``.
    """
    for suffix in ("_pre_shift", "_post_shift"):
        if raw_task.endswith(suffix):
            return raw_task[: -len(suffix)]
    return raw_task


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python builtins for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return _nan_safe_list(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_outputs(
    groups: dict[GroupKey, dict[str, Any]],
    out_root: Path,
    task_configs: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    """Write aggregated outputs to disk.

    Parameters
    ----------
    groups
        Mapping from group key to the output of :func:`aggregate_group`.
    out_root
        Root destination directory (e.g. ``results/weighted_lse_dp``).
    task_configs
        Optional mapping of task name -> task config dict from
        ``paper_suite.json``.  Used to populate calibration JSON fields.

    Returns
    -------
    list[Path]
        Paths of all files written.
    """
    out_root = Path(out_root)
    written: list[Path] = []

    aggregated_root = out_root / "phase2" / "aggregated"
    calibration_root = out_root / "phase2" / "calibration"

    # 1. Write per-(task, algorithm) summary.json files.
    # Suite priority: paper_suite wins over smoke.  Pre-compute which
    # (task, algo) pairs are covered by the highest-priority suite so
    # lower-priority suites don't overwrite their summary.json files.
    _top_suite = _SUITES[0]  # "paper_suite"
    _top_suite_pairs: set[tuple[str, str]] = {
        (task, algo)
        for (suite, task, algo) in groups
        if suite == _top_suite
    }

    for (suite, task, algorithm), agg in sorted(groups.items()):
        # Skip smoke (or any lower-priority suite) when paper_suite data
        # exists for this (task, algo) pair — prevents smoke runs from
        # silently overwriting full paper-suite aggregates.
        if suite != _top_suite and (task, algorithm) in _top_suite_pairs:
            continue

        safe_task = task.replace("/", "_").replace(" ", "_")
        safe_algo = algorithm.replace("/", "_").replace(" ", "_")

        summary_dir = aggregated_root / safe_task / safe_algo
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Use real learning curves from curves.npz (R4-5 fix).
        # Fall back to a stage-normalised calibration proxy only when no
        # curves.npz data is available (DP-only or legacy runs).
        curves: dict[str, Any] = agg.get("curves_from_checkpoints") or {}
        if not curves:
            calib = agg.get("calibration")
            if calib is not None:
                stage_arr = calib.get("stage")
                reward_arr = calib.get("reward_mean")
                if (
                    stage_arr is not None
                    and reward_arr is not None
                    and len(stage_arr) > 0
                ):
                    max_stage = float(np.nanmax(np.abs(stage_arr))) or 1.0
                    curves = {
                        "steps": (stage_arr / max_stage).tolist(),
                        "mean_return": _nan_safe_list(reward_arr),
                        "std_return": [],
                        "episode_returns": _nan_safe_list(reward_arr),
                    }

        # Add change_point from adaptation metadata if available.
        adapt = agg.get("adaptation")
        change_at_episode: int | None = None
        if adapt is not None:
            cp = adapt.get("regime_shift_episode_mean")
            if cp is not None and cp > 0:
                curves = {**curves, "change_point": float(cp)}
                change_at_episode = int(cp)

        entry = {
            "suite": suite,
            "task": task,
            "algorithm": algorithm,
            "n_seeds": agg["n_seeds"],
            "seeds": agg["seeds"],
            "scalar_metrics": agg["scalar_metrics"],
            "tail_risk": agg.get("tail_risk"),
            "adaptation": agg.get("adaptation"),
            "event_rates": agg.get("event_rates"),
            "curves": curves,
            # -- Top-level aliases for standalone plotting scripts (R7-3 fix) --
            # plot_phase2_learning_curves.py expects top-level "checkpoints"
            "checkpoints": curves.get("steps"),
            # plot_phase2_adaptation.py expects {seed: [returns]} at top-level
            "episode_returns": agg.get("episode_returns_by_seed") or {},
            # plot_phase2_adaptation.py reads "change_at_episode" at top-level
            "change_at_episode": change_at_episode,
            # make_phase2_figures.fig_visitation_heatmaps reads this (R7-2 fix)
            "visitation_counts": agg.get("visitation_counts"),
        }
        summary_path = summary_dir / "summary.json"
        save_json(summary_path, _make_json_safe(entry))
        written.append(summary_path)

        # Write per-group calibration NPZ
        calib = agg.get("calibration")
        if calib:
            npz_path = aggregated_root / safe_task / f"{safe_algo}_calibration_mean.npz"
            schema = make_npz_schema(
                phase="phase2",
                task=task,
                algorithm=algorithm,
                seed=-1,  # aggregated across seeds
                storage_mode="calibration_aggregated",
                arrays=list(calib.keys()),
            )
            save_npz_with_schema(npz_path, schema, calib)
            written.append(npz_path)

    # 2. Write per-task-family calibration JSON files (R7-A1 fix).
    # Group raw task labels by canonical family name so that regime-shift
    # *_pre_shift / *_post_shift DP groups merge into one family-level JSON
    # (spec §12: one calibration file per stress-task family).
    #
    # Use only the highest-priority suite's groups for calibration so smoke
    # runs don't contaminate the paper-suite statistics.
    priority_groups: dict[GroupKey, list[dict[str, Any]]] = {
        k: v for k, v in groups.items()
        if k[0] == _top_suite or (k[1], k[2]) not in _top_suite_pairs
    }

    family_to_raw_tasks: dict[str, set[str]] = {}
    for (suite, task, algo) in priority_groups:
        canonical = _canonical_family_from_task(task)
        family_to_raw_tasks.setdefault(canonical, set()).add(task)

    for canonical_family in sorted(family_to_raw_tasks):
        raw_tasks = family_to_raw_tasks[canonical_family]
        tc = None
        if task_configs is not None:
            # Look up config under canonical name first, then any raw variant.
            tc = task_configs.get(canonical_family)
            if tc is None:
                for rt in raw_tasks:
                    tc = task_configs.get(rt)
                    if tc is not None:
                        break

        calib_doc = build_calibration_json(canonical_family, priority_groups, tc, raw_tasks)
        calib_doc_safe = _make_json_safe(calib_doc)

        calibration_root.mkdir(parents=True, exist_ok=True)
        safe_family = canonical_family.replace("/", "_").replace(" ", "_")
        calib_path = calibration_root / f"{safe_family}.json"
        save_json(calib_path, calib_doc_safe)
        written.append(calib_path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_paper_suite_config() -> dict[str, dict[str, Any]] | None:
    """Load the paper_suite.json config and return the tasks dict.

    Returns None if the file is not found or cannot be parsed.
    """
    config_path = _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs" / "phase2" / "paper_suite.json"
    if not config_path.is_file():
        return None
    try:
        data = load_json(config_path)
        return data.get("tasks")
    except Exception:
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase II aggregator: summarise raw run artifacts across seeds "
            "and emit calibration JSON for Phase III."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(_DEFAULT_OUT_ROOT),
        help=(
            "Root of the results tree "
            f"(default: {_DEFAULT_OUT_ROOT})."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Aggregate only this task (optional). "
            "Default: aggregate all tasks."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovery results without writing any files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    print("Phase II aggregator")
    print(f"  out-root: {args.out_root}")
    if args.task:
        print(f"  task filter: {args.task}")
    print()

    # 1. Discover runs.
    records = discover_runs(args.out_root, task_filter=args.task)
    print(f"Discovered {len(records)} run(s).")

    if not records:
        print("No completed Phase II runs found. Nothing to aggregate.")
        if not args.dry_run:
            # Still write an empty calibration directory so downstream
            # tools don't break.
            agg_root = args.out_root / "phase2" / "aggregated"
            agg_root.mkdir(parents=True, exist_ok=True)
            print(f"Created empty aggregated directory: {agg_root}")
        return

    # 2. Group runs.
    grouped = group_runs(records)
    print(f"Grouped into {len(grouped)} (suite, task, algorithm) group(s):")
    for (suite, task, algo), members in sorted(grouped.items()):
        seeds = sorted(m["seed"] for m in members)
        print(
            f"  {suite}/{task}/{algo}  "
            f"seeds={seeds}  (n={len(members)})"
        )
    print()

    if args.dry_run:
        print("DRY RUN -- no files written.")
        return

    # 3. Aggregate each group.
    aggregated: dict[GroupKey, dict[str, Any]] = {}
    for key, members in grouped.items():
        aggregated[key] = aggregate_group(members)

    # 4. Load task configs for calibration JSON enrichment.
    task_configs = _load_paper_suite_config()

    # 5. Write outputs.
    written = write_outputs(aggregated, args.out_root, task_configs)
    print(f"Wrote {len(written)} file(s):")
    for p in written:
        print(f"  {p}")
    print("\nPhase II aggregation complete.")


if __name__ == "__main__":
    main()
