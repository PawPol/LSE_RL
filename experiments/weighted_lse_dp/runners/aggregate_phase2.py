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
)
_CALIB_SCALAR_EVENT_KEYS: tuple[str, ...] = (
    "jackpot_event_rate",
    "catastrophe_event_rate",
    "hazard_hit_rate",
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

            records.append({
                "run_dir": run_dir,
                "run_json": rj,
                "suite": suite,
                "task": rj.get("task", "unknown"),
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

    for rec in group:
        seeds.append(rec["seed"])
        per_seed_run_json.append(rec["run_json"])
        m = _load_metrics_safe(rec["run_dir"])
        if m is not None:
            per_seed_metrics.append(m)
        c = _load_calibration_safe(rec["run_dir"])
        if c is not None:
            per_seed_calib.append(c)

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

    return {
        "n_seeds": len(seeds),
        "seeds": sorted(seeds),
        "scalar_metrics": scalar_agg,
        "calibration": calib_agg,
        "tail_risk": tail_risk,
        "adaptation": adaptation,
        "event_rates": event_rates,
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
) -> dict[str, Any]:
    """Build the calibration JSON document for a single task family.

    Averages calibration statistics across all algorithms for the given
    task to produce a single task-level calibration profile.

    Parameters
    ----------
    task_family:
        Task name (e.g. ``"chain_jackpot"``).
    groups:
        Already-aggregated groups keyed by ``(suite, task, algorithm)``.
        Only entries matching ``task_family`` are consumed.
    task_config:
        Optional task configuration from ``paper_suite.json``. Used to
        extract ``gamma``, reward range, etc.

    Returns
    -------
    dict
        The calibration JSON document per spec section 12.
    """
    # Filter groups for this task family
    task_groups: list[dict[str, Any]] = []
    for (suite, task, algo), agg in groups.items():
        if task == task_family:
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

    # Collect all per-group calibration dicts and average across algos
    calib_dicts: list[dict[str, np.ndarray]] = []
    for g in task_groups:
        c = g.get("calibration")
        if c is not None:
            calib_dicts.append(c)

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

        # -- Per-stage quantiles for pos_margin and neg_margin (spec §12) --
        # Quantiles are computed across algorithm-group calibration dicts
        # (each already averaged across seeds within its group).
        for margin_key, out_key in (
            ("aligned_positive_mean", "pos_margin_quantiles"),
            ("aligned_negative_mean", "neg_margin_quantiles"),
        ):
            margin_arrays: list[np.ndarray] = []
            for c in calib_dicts:
                a = c.get(margin_key)
                if a is not None and a.ndim == 1:
                    margin_arrays.append(a)
            if len(margin_arrays) >= 2:
                try:
                    stacked = np.stack(margin_arrays, axis=0)  # (n_groups, H+1)
                    stagewise[out_key] = {
                        "q05": _nan_safe_list(
                            np.nanpercentile(stacked, 5, axis=0),
                        ),
                        "q25": _nan_safe_list(
                            np.nanpercentile(stacked, 25, axis=0),
                        ),
                        "q50": _nan_safe_list(
                            np.nanpercentile(stacked, 50, axis=0),
                        ),
                        "q75": _nan_safe_list(
                            np.nanpercentile(stacked, 75, axis=0),
                        ),
                        "q95": _nan_safe_list(
                            np.nanpercentile(stacked, 95, axis=0),
                        ),
                    }
                except ValueError:
                    pass
            elif len(margin_arrays) == 1:
                # Single group: report the array as all quantiles (degenerate)
                arr = margin_arrays[0]
                safe = _nan_safe_list(arr)
                stagewise[out_key] = {
                    "q05": safe, "q25": safe, "q50": safe,
                    "q75": safe, "q95": safe,
                }

        # Compute empirical_r_max from reward_mean across stages
        for c in calib_dicts:
            r_mean = c.get("reward_mean")
            r_std = c.get("reward_std")
            if r_mean is not None:
                abs_max = float(np.nanmax(np.abs(r_mean)))
                if empirical_r_max is None or abs_max > empirical_r_max:
                    empirical_r_max = abs_max
            # Also check raw reward range if available
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

    # Determine recommended_task_sign
    recommended_task_sign = _determine_task_sign(stagewise)

    # -- Event-conditioned margin block (spec §12) --------------------------
    event_cond_return: float | None = None
    if tail_risk is not None:
        event_cond_return = tail_risk.get("event_conditioned_return_mean")

    event_conditioned_margins: dict[str, Any] = {
        "event_conditioned_return": event_cond_return,
        "note": (
            "event_conditioned_return is the mean episode return "
            "conditioned on at least one stress event occurring. "
            "Per-stage event-conditioned margin arrays require "
            "event-flagged transitions logged per stage, which "
            "is available in transitions.npz for future processing."
        ),
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


def _determine_task_sign(
    stagewise: dict[str, Any] | None,
) -> int:
    """Determine recommended task sign from stage-0 margin statistics.

    Returns +1 if pos_margin_mean >= neg_margin_mean at stage 0,
    -1 if neg_margin_mean > pos_margin_mean.  If the sign is ambiguous
    (no data available), defaults to +1 with a warning.

    Per spec section 12: one integer sign per experiment family.
    """
    if stagewise is None:
        print(
            "  [WARN] _determine_task_sign: no stagewise data; "
            "defaulting to +1",
            file=sys.stderr,
        )
        return 1

    pos = stagewise.get("pos_margin_mean_mean")
    neg = stagewise.get("neg_margin_mean_mean")

    if pos is None or neg is None:
        print(
            "  [WARN] _determine_task_sign: missing pos/neg margin means; "
            "defaulting to +1",
            file=sys.stderr,
        )
        return 1

    # Use stage 0 values
    if not isinstance(pos, list) or not isinstance(neg, list):
        print(
            "  [WARN] _determine_task_sign: margin data not list; "
            "defaulting to +1",
            file=sys.stderr,
        )
        return 1
    if len(pos) == 0 or len(neg) == 0:
        print(
            "  [WARN] _determine_task_sign: empty margin arrays; "
            "defaulting to +1",
            file=sys.stderr,
        )
        return 1

    pos_0 = pos[0]
    neg_0 = neg[0]

    if pos_0 is None or neg_0 is None:
        print(
            "  [WARN] _determine_task_sign: stage-0 margin is None; "
            "defaulting to +1",
            file=sys.stderr,
        )
        return 1

    if neg_0 > pos_0:
        return -1
    else:
        return 1


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

    # 1. Write per-(task, algorithm) summary.json files
    for (suite, task, algorithm), agg in sorted(groups.items()):
        safe_task = task.replace("/", "_").replace(" ", "_")
        safe_algo = algorithm.replace("/", "_").replace(" ", "_")

        summary_dir = aggregated_root / safe_task / safe_algo
        summary_dir.mkdir(parents=True, exist_ok=True)

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

    # 2. Write per-task-family calibration JSON files
    task_families: set[str] = set()
    for (suite, task, algo) in groups:
        task_families.add(task)

    for task_family in sorted(task_families):
        tc = None
        if task_configs is not None:
            tc = task_configs.get(task_family)

        calib_doc = build_calibration_json(task_family, groups, tc)
        calib_doc_safe = _make_json_safe(calib_doc)

        calibration_root.mkdir(parents=True, exist_ok=True)
        safe_family = task_family.replace("/", "_").replace(" ", "_")
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
