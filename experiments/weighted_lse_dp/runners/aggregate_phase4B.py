#!/usr/bin/env python
"""Phase IV-B: aggregate translation experiment results.

Scans ``results/weighted_lse_dp/phase4/<suite>/<task_tag>/<algorithm>/seed_<N>/``
seed-level artifacts, groups them by ``(suite, task_tag, algorithm)``,
aggregates scalar metrics across seeds (mean, std, min, max, n_seeds), and
writes a per-(task, algorithm) ``summary.json`` under
``<out_root>/phase4/<suite>/aggregated/<task_tag>/<algorithm>/summary.json``.

A top-level ``<out_root>/phase4/<suite>/aggregated/index.json`` is also
written listing every (task, algorithm) pair with its seed count and
contributing run directories, for downstream consumption by
``plotter-analyst``.

CLI::

    python experiments/weighted_lse_dp/runners/aggregate_phase4B.py \\
        [--out-root PATH]          # default: results/weighted_lse_dp
        [--suite SUITE]            # e.g. translation_4a2; default: all suites
        [--task TASK_TAG]          # filter to one task_tag
        [--algorithm ALG]          # filter to one algorithm
        [--dry-run]                # print discovery without writing
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

from experiments.weighted_lse_dp.common.manifests import (  # noqa: E402
    load_metrics_json,
    load_run_json,
)

__all__ = [
    "discover_runs",
    "group_runs",
    "aggregate_group",
    "write_outputs",
    "main",
]


# ---------------------------------------------------------------------------
# Aggregation schema
# ---------------------------------------------------------------------------

_SCHEMA_VERSION: str = "1.0.0"

#: Scalar metric keys to aggregate from metrics.json.  Keys absent from a
#: particular run are skipped for that seed.
_SCALAR_METRIC_KEYS: tuple[str, ...] = (
    # RL outcome
    "train_steps",
    "n_transitions",
    "mean_return",
    "final_disc_return_mean",
    "final_disc_return_std",
    "final_success_rate",
    "final_10pct_success_rate",
    "steps_to_threshold",
    # DP outcome
    "final_bellman_residual",
    "n_sweeps",
    "wall_clock_s",
    "n_states",
    "sweeps_to_1e-2",
    "sweeps_to_1e-4",
    "sweeps_to_1e-6",
    # Safe-specific
    "schedule_sign",
    "beta_used_min",
    "beta_used_max",
    "n_stages_clipped",
    "clip_fraction",
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_runs(
    out_root: Path,
    *,
    suite: str | None = None,
    task: str | None = None,
    algorithm: str | None = None,
) -> list[Path]:
    """Return the list of seed-level run directories under the phase4 tree.

    ``suite=None`` scans every sub-suite under ``phase4/``.  ``task`` and
    ``algorithm`` are simple substring filters on their respective path
    components (exact match required).
    """
    root = Path(out_root) / "phase4"
    if not root.is_dir():
        return []

    suite_glob = suite if suite is not None else "*"
    task_glob = task if task is not None else "*"
    algo_glob = algorithm if algorithm is not None else "*"

    pattern = f"{suite_glob}/{task_glob}/{algo_glob}/seed_*"
    # Exclude the aggregation output dir (phase4/<suite>/aggregated/...).
    matches = [
        p
        for p in root.glob(pattern)
        if p.is_dir()
        and (p / "run.json").is_file()
        and "aggregated" not in p.relative_to(root).parts
    ]
    return sorted(matches)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_runs(
    run_dirs: list[Path],
) -> dict[tuple[str, str, str], list[Path]]:
    """Group seed dirs by ``(suite, task_tag, algorithm)``."""
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for rd in run_dirs:
        # Expected layout: .../phase4/<suite>/<task>/<algorithm>/seed_<N>
        parts = rd.parts
        try:
            i = parts.index("phase4")
        except ValueError:
            continue
        if i + 4 >= len(parts):
            continue
        suite = parts[i + 1]
        task = parts[i + 2]
        algorithm = parts[i + 3]
        groups[(suite, task, algorithm)].append(rd)
    return dict(groups)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _compute_safe_diagnostics(run_dir: Path) -> dict[str, float]:
    """Compute safe operator diagnostics from transitions.npz for safe variants."""
    trans_path = run_dir / "transitions.npz"
    if not trans_path.exists():
        return {}
    try:
        d = np.load(trans_path, allow_pickle=False)
    except Exception:
        return {}
    result: dict[str, float] = {}
    if "safe_beta_used" in d and "margin_beta0" in d:
        u = d["safe_beta_used"] * d["margin_beta0"]
        result["mean_abs_natural_shift"] = float(np.mean(np.abs(u)))
        result["frac_u_ge_5e3"] = float(np.mean(np.abs(u) >= 5e-3))
    if "safe_effective_discount" in d:
        delta_d = d["safe_effective_discount"] - d.get("safe_beta_used", np.zeros(1)) * 0.0
        # delta_effective_discount = safe_effective_discount - classical_discount
        # classical effective discount is just gamma, use safe_effective_discount directly
        result["mean_abs_delta_effective_discount"] = float(
            np.mean(np.abs(d["safe_effective_discount"] - 1.0))  # deviation from unity
        )
    if "safe_target" in d and "td_target_beta0" in d:
        target_gap = d["safe_target"] - d["td_target_beta0"]
        result["mean_abs_target_gap"] = float(np.mean(np.abs(target_gap)))
    return result


def aggregate_group(
    suite: str,
    task_tag: str,
    algorithm: str,
    run_dirs: list[Path],
) -> dict[str, Any]:
    """Aggregate metrics across seeds for one (suite, task_tag, algorithm)."""
    per_seed: list[dict[str, Any]] = []
    seed_values: dict[str, list[float]] = defaultdict(list)

    for rd in sorted(run_dirs):
        try:
            run = load_run_json(rd)
        except FileNotFoundError:
            continue
        try:
            metrics = load_metrics_json(rd)
        except FileNotFoundError:
            metrics = {}

        seed = int(run.get("seed", -1))
        entry: dict[str, Any] = {
            "seed": seed,
            "run_dir": str(rd),
            "status": run.get("status", "unknown"),
        }

        for key in _SCALAR_METRIC_KEYS:
            if key in metrics:
                v = _safe_float(metrics[key])
                if v is not None:
                    entry[key] = v
                    seed_values[key].append(v)

        # Compute safe diagnostics from transitions.npz
        if "safe" in algorithm:
            diag = _compute_safe_diagnostics(rd)
            for k, v in diag.items():
                entry[k] = v
                seed_values[k].append(v)

        per_seed.append(entry)

    # Aggregate stats for each key we saw.
    agg: dict[str, Any] = {}
    for key, values in seed_values.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        agg[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }

    return {
        "schema_version": _SCHEMA_VERSION,
        "suite": suite,
        "task_tag": task_tag,
        "algorithm": algorithm,
        "n_seeds": len(per_seed),
        "seeds": sorted({int(e["seed"]) for e in per_seed if e.get("seed", -1) >= 0}),
        "per_seed": per_seed,
        "metrics": agg,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(
    groups: dict[tuple[str, str, str], list[Path]],
    out_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write per-(task, algo) summaries + an index.json per suite."""
    per_suite_index: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for (suite, task, algorithm), run_dirs in sorted(groups.items()):
        summary = aggregate_group(suite, task, algorithm, run_dirs)

        out_dir = (
            Path(out_root) / "phase4" / suite / "aggregated" / task / algorithm
        )
        out_path = out_dir / "summary.json"

        if dry_run:
            print(
                f"  [DRY] suite={suite} task={task} algo={algorithm}  "
                f"n_seeds={summary['n_seeds']} -> {out_path}"
            )
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print(
                f"  [WRITE] suite={suite} task={task} algo={algorithm}  "
                f"n_seeds={summary['n_seeds']} -> {out_path}"
            )

        per_suite_index[suite].append({
            "task_tag": task,
            "algorithm": algorithm,
            "n_seeds": summary["n_seeds"],
            "seeds": summary["seeds"],
            "summary_path": str(out_path),
        })

    # Suite-level index.json
    for suite, entries in per_suite_index.items():
        index_payload: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "suite": suite,
            "n_groups": len(entries),
            "entries": sorted(
                entries,
                key=lambda e: (e["task_tag"], e["algorithm"]),
            ),
        }
        index_path = (
            Path(out_root) / "phase4" / suite / "aggregated" / "index.json"
        )
        if dry_run:
            print(f"  [DRY] suite={suite} index -> {index_path}")
        else:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w") as f:
                json.dump(index_payload, f, indent=2, sort_keys=True)
            print(f"  [WRITE] suite={suite} index -> {index_path}")

    return dict(per_suite_index)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="aggregate_phase4B",
        description=__doc__,
    )
    p.add_argument("--out-root", type=Path,
                   default=Path("results/weighted_lse_dp"),
                   help="Results root directory.")
    p.add_argument("--suite", type=str, default=None,
                   help="Restrict aggregation to a single suite.")
    p.add_argument("--task", type=str, default=None,
                   help="Restrict aggregation to a single task_tag.")
    p.add_argument("--algorithm", type=str, default=None,
                   help="Restrict aggregation to a single algorithm.")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover and print without writing summary files.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    run_dirs = discover_runs(
        args.out_root,
        suite=args.suite,
        task=args.task,
        algorithm=args.algorithm,
    )
    if not run_dirs:
        print(
            f"[aggregate_phase4B] No runs discovered under "
            f"{args.out_root}/phase4 (suite={args.suite}, task={args.task}, "
            f"algorithm={args.algorithm})."
        )
        return 1

    groups = group_runs(run_dirs)
    print(
        f"[aggregate_phase4B] Discovered {len(run_dirs)} seed run(s) "
        f"in {len(groups)} group(s)."
    )

    write_outputs(groups, args.out_root, dry_run=args.dry_run)
    print(f"[aggregate_phase4B] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
