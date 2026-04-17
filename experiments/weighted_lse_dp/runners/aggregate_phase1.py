#!/usr/bin/env python
"""Phase I seed aggregation and calibration-stat extraction.

Reads all completed run directories under the Phase I raw result tree,
groups them by ``(suite, task, algorithm, [gamma_prime])``, aggregates
scalar metrics across seeds (with bootstrapped 95% CIs), averages
stage-wise calibration arrays, and writes processed summaries.

Output layout::

    <out-root>/
        summary.json                            -- all (task, algorithm) groups
        <task>_<algorithm>_calibration.npz       -- stage-wise mean calibration
        ablation_summary.json                   -- gamma' ablation (if present)

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/aggregate_phase1.py \\
        [--run-root PATH]     # default: results/weighted_lse_dp
        [--out-root PATH]     # default: results/weighted_lse_dp/processed/phase1
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
    "write_outputs",
    "main",
]


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

# Suites to scan, in priority order.
_SUITES = ("paper_suite", "smoke")
_ABLATION_SUITE = "ablation"


def discover_runs(
    run_root: Path,
    *,
    include_ablation: bool = True,
) -> list[dict[str, Any]]:
    """Find all completed run directories and parse their manifests.

    Each returned dict contains:
        - ``run_dir``: absolute Path to the seed directory
        - ``run_json``: the full run.json dict
        - ``suite``: which suite the run belongs to
        - ``task``: task name
        - ``algorithm``: algorithm name
        - ``seed``: integer seed
        - ``gamma_prime``: float or None (ablation override)

    Returns
    -------
    list[dict]
        One entry per discovered run, sorted by (suite, task, algo, seed).
    """
    records: list[dict[str, Any]] = []

    suites_to_scan = list(_SUITES)
    if include_ablation:
        suites_to_scan.append(_ABLATION_SUITE)

    for suite in suites_to_scan:
        dirs = find_run_dirs(run_root, phase="phase1", suite=suite)
        for run_dir in dirs:
            try:
                rj = load_run_json(run_dir)
            except Exception as exc:
                print(
                    f"  [WARN] skipping {run_dir}: cannot load run.json: {exc}",
                    file=sys.stderr,
                )
                continue

            config = rj.get("config", {})
            gamma_prime = config.get("gamma_prime_override", None)

            records.append({
                "run_dir": run_dir,
                "run_json": rj,
                "suite": rj.get("phase", "phase1") and suite,
                "task": rj.get("task", "unknown"),
                "algorithm": rj.get("algorithm", "unknown"),
                "seed": rj.get("seed", -1),
                "gamma_prime": gamma_prime,
            })

    # Deterministic ordering for reproducibility.
    records.sort(
        key=lambda r: (
            r["suite"],
            r["task"],
            r["algorithm"],
            r.get("gamma_prime") or 0.0,
            r["seed"],
        )
    )
    return records


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

GroupKey = tuple[str, str, str, float | None]
"""(suite, task, algorithm, gamma_prime)."""


def _group_key(rec: dict[str, Any]) -> GroupKey:
    return (rec["suite"], rec["task"], rec["algorithm"], rec.get("gamma_prime"))


def group_runs(
    records: list[dict[str, Any]],
) -> dict[GroupKey, list[dict[str, Any]]]:
    """Group run records by ``(suite, task, algorithm, gamma_prime)``.

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
# Per-group aggregation
# ---------------------------------------------------------------------------

# Scalar metric keys we attempt to aggregate from metrics.json.
_SCALAR_METRIC_KEYS: tuple[str, ...] = (
    "final_bellman_residual",
    "n_sweeps",
    "wall_clock_s",
    # RL runner metrics
    "train_steps",
    "n_transitions",
    "final_disc_return_mean",
    "auc_disc_return",
    "steps_to_threshold",
)


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


def aggregate_group(
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate scalar metrics and calibration arrays across seeds.

    Parameters
    ----------
    group
        List of run records sharing the same (suite, task, algorithm,
        gamma_prime).

    Returns
    -------
    dict
        ``"n_seeds"``: int,
        ``"seeds"``: list of ints,
        ``"scalar_metrics"``: dict of metric_name -> aggregation dict,
        ``"calibration"``: dict of array_name -> ndarray (stage-wise means),
                           or None if no calibration data.
    """
    seeds: list[int] = []
    per_seed_metrics: list[dict[str, Any]] = []
    per_seed_calib: list[dict[str, np.ndarray]] = []

    for rec in group:
        seeds.append(rec["seed"])
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
                # Convert numpy scalars to Python floats for JSON.
                scalar_agg[key] = {
                    k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in agg.items()
                }
    elif len(per_seed_metrics) == 1:
        # Single seed: report raw values, no CI.
        m = per_seed_metrics[0]
        for key in _SCALAR_METRIC_KEYS:
            v = m.get(key)
            if v is not None and isinstance(v, (int, float)):
                scalar_agg[key] = {"mean": float(v), "n_seeds": 1}

    # -- Aggregate calibration arrays (stage-wise mean across seeds) -------
    calib_agg: dict[str, np.ndarray] | None = None
    if per_seed_calib:
        # Determine which array keys to average (exclude _schema).
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
                # Stack along a new seed axis and compute mean.
                try:
                    stacked = np.stack(arrays, axis=0)  # (n_seeds, H+1)
                    calib_agg[arr_key] = np.mean(stacked, axis=0)
                except ValueError:
                    # Shape mismatch across seeds (should not happen).
                    pass

    return {
        "n_seeds": len(seeds),
        "seeds": sorted(seeds),
        "scalar_metrics": scalar_agg,
        "calibration": calib_agg,
    }


# ---------------------------------------------------------------------------
# Output writing
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
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_outputs(
    groups: dict[GroupKey, dict[str, Any]],
    out_root: Path,
) -> list[Path]:
    """Write aggregated outputs to disk.

    Parameters
    ----------
    groups
        Mapping from group key to the output of :func:`aggregate_group`.
    out_root
        Destination directory.

    Returns
    -------
    list[Path]
        Paths of all files written.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Partition into main and ablation groups.
    main_entries: list[dict[str, Any]] = []
    ablation_entries: list[dict[str, Any]] = []

    for (suite, task, algorithm, gamma_prime), agg in sorted(groups.items()):
        entry = {
            "suite": suite,
            "task": task,
            "algorithm": algorithm,
            "n_seeds": agg["n_seeds"],
            "seeds": agg["seeds"],
            "scalar_metrics": agg["scalar_metrics"],
        }
        if gamma_prime is not None:
            entry["gamma_prime"] = gamma_prime

        if suite == _ABLATION_SUITE or gamma_prime is not None:
            ablation_entries.append(entry)
        else:
            main_entries.append(entry)

        # Write per-group calibration NPZ if calibration data exists.
        calib = agg.get("calibration")
        if calib:
            safe_task = task.replace("/", "_").replace(" ", "_")
            safe_algo = algorithm.replace("/", "_").replace(" ", "_")
            suffix = ""
            if gamma_prime is not None:
                suffix = f"_gp{gamma_prime:.4f}".rstrip("0").rstrip(".")
            npz_name = f"{safe_task}_{safe_algo}{suffix}_calibration.npz"
            npz_path = out_root / npz_name

            schema = make_npz_schema(
                phase="phase1",
                task=task,
                algorithm=algorithm,
                seed=-1,  # aggregated across seeds
                storage_mode="calibration_aggregated",
                arrays=list(calib.keys()),
            )
            save_npz_with_schema(npz_path, schema, calib)
            written.append(npz_path)

    # Write summary.json.
    summary_path = out_root / "summary.json"
    save_json(summary_path, _make_json_safe({"groups": main_entries}))
    written.append(summary_path)

    # Write ablation_summary.json if there are ablation groups.
    if ablation_entries:
        abl_path = out_root / "ablation_summary.json"
        save_json(abl_path, _make_json_safe({"groups": ablation_entries}))
        written.append(abl_path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase I aggregator: summarise raw run artifacts across seeds."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("results/weighted_lse_dp"),
        help=(
            "Root of the raw results tree "
            "(default: results/weighted_lse_dp)."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/weighted_lse_dp/processed/phase1"),
        help=(
            "Destination for processed outputs "
            "(default: results/weighted_lse_dp/processed/phase1)."
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

    print("Phase I aggregator")
    print(f"  run-root: {args.run_root}")
    print(f"  out-root: {args.out_root}")
    print()

    # 1. Discover runs.
    records = discover_runs(args.run_root)
    print(f"Discovered {len(records)} run(s).")

    if not records:
        print("No completed runs found. Nothing to aggregate.")
        if not args.dry_run:
            # Still write an empty summary so downstream tools don't break.
            args.out_root.mkdir(parents=True, exist_ok=True)
            summary_path = args.out_root / "summary.json"
            save_json(summary_path, {"groups": []})
            print(f"Wrote empty summary: {summary_path}")
        return

    # 2. Group runs.
    grouped = group_runs(records)
    print(f"Grouped into {len(grouped)} (suite, task, algorithm) group(s):")
    for (suite, task, algo, gp), members in sorted(grouped.items()):
        gp_str = f"  gamma'={gp}" if gp is not None else ""
        seeds = sorted(m["seed"] for m in members)
        print(
            f"  {suite}/{task}/{algo}{gp_str}  "
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

    # 4. Write outputs.
    written = write_outputs(aggregated, args.out_root)
    print(f"Wrote {len(written)} file(s) to {args.out_root}:")
    for p in written:
        print(f"  {p}")
    print("\nAggregation complete.")


if __name__ == "__main__":
    main()
