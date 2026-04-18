#!/usr/bin/env python3
"""Phase III aggregation: classical-vs-safe comparisons.

Scans ``results/weighted_lse_dp/phase3/paper_suite/<task>/<algorithm>/<seed>/``
for completed runs, groups by ``(task, algorithm)``, and produces:

    results/weighted_lse_dp/phase3/aggregated/<task>/<algorithm>/
        summary.json          -- scalar metrics, mean +/- std across seeds
        safe_stagewise.npz    -- per-stage quantiles from *pooled* raw data
        curves.npz            -- aggregated curve arrays

CRITICAL (Task 36 lesson, 2026-04-17):
    Quantiles for safe fields are computed from pooled raw per-transition
    arrays across ALL seeds, NOT from pre-computed per-seed means.  This
    avoids the "summary-of-summaries" anti-pattern that loses distributional
    information.

Usage::

    python -m experiments.weighted_lse_dp.runners.aggregate_phase3 \\
        [--out-root PATH] [--task TASK] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping -- ensure repo root is importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.io import (
    load_json,
    load_npz,
    save_json,
    save_npz_with_schema,
)
from experiments.weighted_lse_dp.common.manifests import find_run_dirs
from experiments.weighted_lse_dp.common.metrics import aggregate
from experiments.weighted_lse_dp.common.schemas import (
    CURVES_ARRAYS_DP,
    CURVES_ARRAYS_RL,
    SAFE_CALIBRATION_ARRAYS,
    SAFE_TRANSITIONS_ARRAYS,
    SCHEMA_VERSION,
    TRANSITIONS_ARRAYS,
    aggregate_safe_stats,
    validate_safe_transitions_npz,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe-specific scalar keys expected in metrics.json
# ---------------------------------------------------------------------------
_SAFE_SCALAR_KEYS: Tuple[str, ...] = (
    "safe_clip_fraction_mean",
    "safe_underdiscount_fraction",
    "safe_rho_overall_mean",
    "safe_effective_discount_mean",
)

# Quantile levels for per-stage safe aggregation
_SAFE_QUANTILES: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

# Fields from transitions.npz to aggregate with full quantile set
_SAFE_FULL_QUANTILE_FIELDS: Tuple[str, ...] = (
    "safe_rho",
    "safe_effective_discount",
    "safe_beta_used",
)

# Fields aggregated with (q05, q50, q95) only
_SAFE_REDUCED_QUANTILE_FIELDS: Tuple[str, ...] = (
    "safe_margin",
)

# Per-stage mean fields (from calibration arrays or computed)
_SAFE_MEAN_FIELDS: Tuple[str, ...] = (
    "safe_clip_fraction",
    "safe_underdiscount_fraction",
    "safe_bellman_residual",
)


# ===================================================================
# 1. Run discovery
# ===================================================================

def discover_runs(
    root: Path,
    *,
    task_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discover completed Phase III runs.

    Parameters
    ----------
    root : Path
        Base results directory (e.g. ``results/weighted_lse_dp``).
    task_filter : str, optional
        If set, only return runs whose task name matches.

    Returns
    -------
    list of dict
        Each dict has keys ``dir``, ``task``, ``algorithm``, ``seed``,
        ``run_json``, and optionally ``safe_provenance``.
    """
    suite_root = root / "phase3" / "paper_suite"
    run_dirs = find_run_dirs(root, phase="phase3", suite="paper_suite", task=task_filter)

    runs: List[Dict[str, Any]] = []
    for d in run_dirs:
        run_json = _load_run_json_safe(d)
        if run_json is None:
            continue

        # Infer task / algorithm / seed from path structure:
        #   .../phase3/paper_suite/<task>/<algorithm>/<seed>/
        parts = d.relative_to(suite_root).parts
        if len(parts) < 3:
            logger.warning(
                "Unexpected directory depth for run dir %s; skipping.", d
            )
            continue

        task = parts[0]
        algorithm = parts[1]
        seed_str = parts[2]

        if task_filter is not None and task != task_filter:
            continue

        try:
            seed = int(seed_str)
        except ValueError:
            seed = seed_str  # type: ignore[assignment]

        safe_prov = _load_safe_provenance(d)

        runs.append({
            "dir": d,
            "task": task,
            "algorithm": algorithm,
            "seed": seed,
            "run_json": run_json,
            "safe_provenance": safe_prov,
        })

    logger.info("Discovered %d Phase III run(s).", len(runs))
    return runs


def _load_run_json_safe(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ``run.json`` from *run_dir*, returning *None* on failure."""
    path = run_dir / "run.json"
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def _load_safe_provenance(run_dir: Path) -> Dict[str, Any]:
    """Load ``safe_provenance.json``, returning empty dict if absent."""
    path = run_dir / "safe_provenance.json"
    if not path.is_file():
        return {}
    try:
        return load_json(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return {}


# ===================================================================
# 2. Grouping
# ===================================================================

GroupKey = Tuple[str, str]  # (task, algorithm)


def group_runs(
    runs: List[Dict[str, Any]],
) -> Dict[GroupKey, List[Dict[str, Any]]]:
    """Group runs by ``(task, algorithm)``."""
    groups: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        key: GroupKey = (r["task"], r["algorithm"])
        groups[key].append(r)

    # Sort seeds within each group for determinism
    for key in groups:
        groups[key].sort(key=lambda r: r["seed"])

    return dict(groups)


# ===================================================================
# 3. Per-group aggregation
# ===================================================================

def aggregate_group(
    group_key: GroupKey,
    seed_runs: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    """Aggregate a single (task, algorithm) group and write outputs.

    Parameters
    ----------
    group_key : (task, algorithm)
    seed_runs : list of run dicts (one per seed)
    out_dir : Path
        Output directory for this group's aggregated files.

    Returns
    -------
    dict
        Summary dict (also written to ``summary.json``).
    """
    task, algorithm = group_key
    seeds = [r["seed"] for r in seed_runs]
    seed_dirs = [r["dir"] for r in seed_runs]

    logger.info(
        "Aggregating group (%s, %s) with %d seed(s): %s",
        task, algorithm, len(seeds), seeds,
    )

    # ----- scalar metrics -----
    per_seed_scalars = _collect_scalar_metrics(seed_dirs)
    agg_scalars = aggregate(per_seed_scalars) if per_seed_scalars else {}

    # ----- curves -----
    agg_curves = _aggregate_curves(seed_dirs, algorithm)

    # ----- safe per-stage quantiles (Task 36: raw pooling) -----
    # Resolve gamma from provenance or first seed's run.json.
    gamma_for_safe: float = 0.99  # fallback
    for r in seed_runs:
        rj = r.get("run_json", {})
        cfg = rj.get("config", {})
        if "gamma" in cfg:
            gamma_for_safe = float(cfg["gamma"])
            break
        tc = cfg.get("task_config", {})
        if "gamma" in tc:
            gamma_for_safe = float(tc["gamma"])
            break
    safe_stagewise = _aggregate_safe_stagewise_from_raw(seed_dirs, gamma=gamma_for_safe)

    # ----- provenance (from first seed with data) -----
    provenance = {}
    for r in seed_runs:
        if r.get("safe_provenance"):
            provenance = {
                "schedule_path": r["safe_provenance"].get("schedule_path", ""),
                "calibration_hash": r["safe_provenance"].get(
                    "calibration_hash", ""
                ),
                "source_phase": r["safe_provenance"].get("source_phase", ""),
            }
            break

    # ----- write outputs -----
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "phase": "III",
        "task": task,
        "algorithm": algorithm,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "metrics": agg_scalars,
        "provenance": provenance,
    }
    save_json(out_dir / "summary.json", summary)

    if agg_curves:
        from experiments.weighted_lse_dp.common.io import make_npz_schema
        curves_schema = make_npz_schema(
            phase="phase3",
            task=task,
            algorithm=algorithm,
            seed=-1,
            storage_mode="aggregated",
            arrays=list(agg_curves.keys()),
        )
        save_npz_with_schema(out_dir / "curves.npz", curves_schema, agg_curves)

    if safe_stagewise:
        from experiments.weighted_lse_dp.common.io import make_npz_schema
        stagewise_schema = make_npz_schema(
            phase="phase3",
            task=task,
            algorithm=algorithm,
            seed=-1,
            storage_mode="aggregated",
            arrays=list(safe_stagewise.keys()),
        )
        save_npz_with_schema(
            out_dir / "safe_stagewise.npz", stagewise_schema, safe_stagewise
        )

    logger.info("Wrote aggregated outputs to %s", out_dir)
    return summary


# -------------------------------------------------------------------
# 3a. Scalar metric collection
# -------------------------------------------------------------------

def _collect_scalar_metrics(
    seed_dirs: List[Path],
) -> List[Dict[str, float]]:
    """Load ``metrics.json`` from each seed dir."""
    results: List[Dict[str, float]] = []
    for d in seed_dirs:
        path = d / "metrics.json"
        if not path.is_file():
            logger.warning("Missing metrics.json in %s; skipping scalars.", d)
            continue
        try:
            data = load_json(path)
            # Flatten: metrics.json may have nested structure or flat scalars.
            scalars = _flatten_scalars(data)
            results.append(scalars)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error loading %s: %s", path, exc)
    return results


def _flatten_scalars(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract scalar float values from a potentially nested dict."""
    out: Dict[str, float] = {}
    for k, v in data.items():
        if k == "schema_version":
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    out[f"{k}_{k2}"] = float(v2)
    return out


# -------------------------------------------------------------------
# 3b. Curve aggregation
# -------------------------------------------------------------------

def _aggregate_curves(
    seed_dirs: List[Path],
    algorithm: str,
) -> Dict[str, np.ndarray]:
    """Aggregate curve arrays across seeds (mean/std)."""
    all_curves: List[Dict[str, np.ndarray]] = []
    for d in seed_dirs:
        path = d / "curves.npz"
        if not path.is_file():
            logger.warning("Missing curves.npz in %s; skipping curves.", d)
            continue
        try:
            all_curves.append(load_npz(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error loading %s: %s", path, exc)

    if not all_curves:
        return {}

    # Determine which keys to aggregate: use intersection across seeds
    common_keys = set(all_curves[0].keys())
    for c in all_curves[1:]:
        common_keys &= set(c.keys())
    common_keys -= {"schema_version"}

    result: Dict[str, np.ndarray] = {}
    for k in sorted(common_keys):
        arrays = []
        for c in all_curves:
            arr = c[k]
            if arr.ndim == 0:
                continue
            arrays.append(arr)
        if not arrays:
            continue
        # Truncate to common length (seeds may differ slightly)
        min_len = min(a.shape[0] for a in arrays)
        stacked = np.stack([a[:min_len] for a in arrays], axis=0)  # (S, L)
        result[f"{k}_mean"] = np.mean(stacked, axis=0)
        result[f"{k}_std"] = np.std(stacked, axis=0)

    return result


# -------------------------------------------------------------------
# 3c. Safe per-stage aggregation from RAW per-transition data
# -------------------------------------------------------------------
# TASK 36 IMPLEMENTATION
#
# LESSON (2026-04-17): Do NOT compute "mean of means" or "quantile of
# means".  Always POOL raw per-transition arrays across all seeds
# BEFORE computing quantiles.  This preserves distributional info.


def _aggregate_safe_stagewise_from_raw(
    seed_dirs: List[Path],
    *,
    gamma: float,
) -> Dict[str, np.ndarray]:
    """Compute per-stage aggregate safe stats by pooling raw transitions across seeds.

    This is the Task 36 implementation.  We:

    1. Load ``transitions.npz`` from every seed.
    2. Concatenate the raw per-transition arrays across all seeds into a
       single payload dict matching the ``aggregate_safe_stats`` contract.
    3. Call ``aggregate_safe_stats(payload, T, gamma)`` once to get all
       per-stage stats.

    This avoids the "summary-of-summaries" anti-pattern.

    Parameters
    ----------
    seed_dirs : list of Path
        Directories containing per-seed ``transitions.npz`` files.
    gamma : float
        Nominal discount factor (used for underdiscount fraction threshold).
    """
    # Keys required by aggregate_safe_stats payload
    _PAYLOAD_KEYS = (
        "safe_stage",
        "safe_rho",
        "safe_effective_discount",
        "safe_beta_used",
        "safe_clip_active",
    )

    # Collect raw arrays from all seeds
    raw_fields: Dict[str, List[np.ndarray]] = defaultdict(list)

    for d in seed_dirs:
        path = d / "transitions.npz"
        if not path.is_file():
            logger.warning(
                "Missing transitions.npz in %s; skipping safe stagewise.", d
            )
            continue

        try:
            npz = load_npz(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error loading %s: %s", path, exc)
            continue

        # Check for safe arrays
        if "safe_stage" not in npz:
            logger.warning(
                "No safe_stage in transitions.npz at %s; "
                "skipping (may be a DP run or incomplete).",
                d,
            )
            continue

        for key in _PAYLOAD_KEYS:
            if key in npz:
                raw_fields[key].append(npz[key])

    # Need at least safe_stage to proceed
    if "safe_stage" not in raw_fields or not raw_fields["safe_stage"]:
        return {}

    # Pool across seeds into a single payload dict
    payload: Dict[str, np.ndarray] = {}
    for key in _PAYLOAD_KEYS:
        if key in raw_fields:
            payload[key] = np.concatenate(raw_fields[key])

    # Determine T from pooled stages
    n_stages = int(payload["safe_stage"].max()) + 1

    # Call aggregate_safe_stats with the correct signature
    result: Dict[str, np.ndarray] = aggregate_safe_stats(
        payload, T=n_stages, gamma=gamma
    )

    # Add stage indices for downstream convenience
    result["stage_indices"] = np.arange(n_stages)

    return result


# ===================================================================
# 4. Write outputs (orchestration)
# ===================================================================

def _print_discovery_table(
    groups: Dict[GroupKey, List[Dict[str, Any]]],
) -> None:
    """Print a discovery summary table to stdout."""
    print(f"\n{'Task':<30} {'Algorithm':<25} {'Seeds':>6}")
    print("-" * 65)
    if not groups:
        print("  (no runs found)")
    for (task, algo), runs in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in runs)
        print(f"{task:<30} {algo:<25} {len(seeds):>6}  {seeds}")
    print()


# ===================================================================
# 5. Main entry point
# ===================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for Phase III aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate Phase III (safe weighted-LSE) experiment results."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/weighted_lse_dp"),
        help="Root results directory (default: results/weighted_lse_dp).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter to a specific task name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovery table without writing any output.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_root: Path = args.out_root

    # Step 1: discover
    runs = discover_runs(out_root, task_filter=args.task)

    # Step 2: group
    groups = group_runs(runs)

    # Dry-run: just print and exit
    if args.dry_run:
        print("Phase III aggregation -- dry run")
        _print_discovery_table(groups)
        print(f"Total groups: {len(groups)}, total runs: {len(runs)}")
        return 0

    if not groups:
        logger.warning(
            "No Phase III runs found under %s. Nothing to aggregate.",
            out_root / "phase3" / "paper_suite",
        )
        return 0

    # Step 3: aggregate each group
    agg_root = out_root / "phase3" / "aggregated"
    summaries: List[Dict[str, Any]] = []

    for group_key, seed_runs in sorted(groups.items()):
        task, algo = group_key
        group_out = agg_root / task / algo
        summary = aggregate_group(group_key, seed_runs, group_out)
        summaries.append(summary)

    logger.info(
        "Aggregation complete. %d group(s) written to %s",
        len(summaries),
        agg_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
