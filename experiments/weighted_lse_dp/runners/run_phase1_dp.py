#!/usr/bin/env python
"""Phase I dynamic-programming runner: produces ground-truth value tables.

For each (task, dp_algorithm, seed) in the paper suite, this runner:

1. Loads the task via the appropriate ``make_*`` factory.
2. Runs the DP planner (PE, VI, PI, MPI, or AsyncVI).
3. Records planning curves via :class:`DPCurvesLogger`.
4. Generates calibration stats via :func:`build_calibration_stats_from_dp_tables`.
5. Flushes everything via :class:`RunWriter`.

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase1_dp.py \\
        [--config PATH]            # default: paper_suite.json
        [--task TASK]              # filter to one task
        [--algorithm ALG]          # filter to one DP algorithm
        [--seed SEED]              # filter to one seed
        [--out-root PATH]          # default: results/weighted_lse_dp/phase1/paper_suite
        [--dry-run]                # print what would run, no execution
        [--gamma-prime GAMMA]      # optional override for gamma (ablation mode)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping -- ensure repo root and mushroom-rl-dev are importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
from experiments.weighted_lse_dp.common.callbacks import DPCurvesLogger
from experiments.weighted_lse_dp.common.calibration import (
    build_calibration_stats_from_dp_tables,
)
from experiments.weighted_lse_dp.common.schemas import RunWriter
from experiments.weighted_lse_dp.common.seeds import seed_everything
from experiments.weighted_lse_dp.common.task_factories import (
    make_chain_base,
    make_grid_base,
    make_taxi_base,
)
from mushroom_rl.algorithms.value.dp import (
    ClassicalAsyncValueIteration,
    ClassicalModifiedPolicyIteration,
    ClassicalPolicyEvaluation,
    ClassicalPolicyIteration,
    ClassicalValueIteration,
    extract_mdp_arrays,
)


# ---------------------------------------------------------------------------
# Registry: task factories and DP planner constructors
# ---------------------------------------------------------------------------

TASK_FACTORIES: dict[str, Any] = {
    "chain_base": make_chain_base,
    "grid_base": make_grid_base,
    "taxi_base": make_taxi_base,
}

#: Short name -> planner class
DP_ALGORITHMS: dict[str, type] = {
    "PE": ClassicalPolicyEvaluation,
    "VI": ClassicalValueIteration,
    "PI": ClassicalPolicyIteration,
    "MPI": ClassicalModifiedPolicyIteration,
    "AsyncVI": ClassicalAsyncValueIteration,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = (
    _REPO_ROOT
    / "experiments"
    / "weighted_lse_dp"
    / "configs"
    / "phase1"
    / "paper_suite.json"
)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build the run matrix from config + CLI filters
# ---------------------------------------------------------------------------


def _build_run_list(
    config: dict,
    *,
    task_filter: str | None = None,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Return a list of ``{task, algorithm, seed}`` dicts to execute."""
    seeds = config.get("seeds", [11, 29, 47])
    tasks_cfg = config.get("tasks", {})

    runs: list[dict[str, Any]] = []

    for task_name, task_cfg in tasks_cfg.items():
        if task_filter is not None and task_name != task_filter:
            continue
        if task_name not in TASK_FACTORIES:
            continue

        dp_algos = task_cfg.get("dp_algorithms", [])
        for algo_name in dp_algos:
            if algorithm_filter is not None and algo_name != algorithm_filter:
                continue
            if algo_name not in DP_ALGORITHMS:
                continue

            for seed in seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue
                runs.append({
                    "task": task_name,
                    "algorithm": algo_name,
                    "seed": seed,
                    "task_cfg": task_cfg,
                })

    return runs


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------


def _make_planner(
    algo_name: str,
    mdp_base: Any,
    ref_pi: np.ndarray,
    *,
    gamma_prime: float | None = None,
) -> Any:
    """Construct a DP planner instance.

    Parameters
    ----------
    algo_name : str
        Short name from :data:`DP_ALGORITHMS`.
    mdp_base :
        The FiniteMDP environment.
    ref_pi :
        Reference policy array of shape ``(H, S)``.  Used only for PE.
    gamma_prime :
        Optional gamma override (ablation mode).  When provided, the
        MDP's ``info.gamma`` is temporarily overridden before planner
        construction.

    Returns
    -------
    planner
        A planner instance ready for ``.run()``.
    """
    # If gamma_prime is requested, temporarily patch the MDP's gamma so
    # the planner picks it up via extract_mdp_arrays.
    original_gamma = None
    if gamma_prime is not None:
        original_gamma = mdp_base.info.gamma
        mdp_base.info.gamma = gamma_prime

    try:
        if algo_name == "PE":
            planner = ClassicalPolicyEvaluation(mdp_base, pi=ref_pi)
        elif algo_name == "VI":
            planner = ClassicalValueIteration(mdp_base)
        elif algo_name == "PI":
            planner = ClassicalPolicyIteration(mdp_base)
        elif algo_name == "MPI":
            planner = ClassicalModifiedPolicyIteration(mdp_base)
        elif algo_name == "AsyncVI":
            planner = ClassicalAsyncValueIteration(mdp_base)
        else:
            raise ValueError(f"Unknown DP algorithm: {algo_name!r}")
    finally:
        if original_gamma is not None:
            mdp_base.info.gamma = original_gamma

    return planner


def _get_sweep_times(planner: Any) -> list[float]:
    """Extract per-sweep wall-clock times from any planner type."""
    if hasattr(planner, "sweep_times_s"):
        return list(planner.sweep_times_s)
    if hasattr(planner, "iter_times_s"):
        return list(planner.iter_times_s)
    return []


def _v_for_sweep_index(planner: Any, sweep_idx: int) -> Any:
    """Return the value table after sweep ``sweep_idx`` (for curve logging).

    Iterative planners populate :attr:`V_sweep_history` with one full
    ``V`` copy per logged residual; the DP runner replays these into
    :class:`~experiments.weighted_lse_dp.common.callbacks.DPCurvesLogger`
    so ``supnorm_to_exact`` and per-sweep snapshots reflect the table at
    that iteration, not only the final ``planner.V``.
    """
    hist = getattr(planner, "V_sweep_history", None)
    if hist is not None and sweep_idx < len(hist):
        return hist[sweep_idx]
    return planner.V


def _run_single(
    task_name: str,
    algo_name: str,
    seed: int,
    task_cfg: dict,
    *,
    out_root: Path,
    suite: str,
    gamma_prime: float | None = None,
    full_config: dict,
) -> None:
    """Execute one (task, algorithm, seed) DP run and flush artifacts."""
    print(f"  [RUN] task={task_name}  algo={algo_name}  seed={seed}")

    # 1. Seed and create task
    seed_everything(seed)
    factory = TASK_FACTORIES[task_name]
    mdp_base, _mdp_rl, cfg, ref_pi = factory(seed=seed, time_augment=False)

    # Resolve effective gamma
    effective_gamma = gamma_prime if gamma_prime is not None else cfg["gamma"]

    # Build resolved config for this run
    run_config = {
        **cfg,
        "suite_config_path": str(_DEFAULT_CONFIG),
        "gamma_effective": effective_gamma,
    }
    if gamma_prime is not None:
        run_config["gamma_prime_override"] = gamma_prime

    # 2. Create RunWriter
    rw = RunWriter.create(
        base=out_root,
        phase="phase1",
        suite=suite,
        task=task_name,
        algorithm=algo_name,
        seed=seed,
        config=run_config,
        storage_mode="dp_stagewise",
    )

    # 3. Construct the planner
    planner = _make_planner(
        algo_name, mdp_base, ref_pi, gamma_prime=gamma_prime,
    )

    # 4. Run the planner
    with rw.timer.phase("fit"):
        planner.run()

    # 5. Extract MDP arrays for calibration stats
    p, r, horizon, gamma_mdp = extract_mdp_arrays(mdp_base)
    gamma_for_calib = gamma_prime if gamma_prime is not None else gamma_mdp

    # 6. Record per-sweep curves via DPCurvesLogger
    #    The planners run internally and store residuals/sweep_times.
    #    We replay them into DPCurvesLogger after the fact.
    #    Pass planner.V as the exact reference so supnorm_to_exact is populated.
    dp_logger = DPCurvesLogger(run_writer=rw, v_exact=planner.V, task=task_name)

    sweep_times = _get_sweep_times(planner)
    residuals = list(planner.residuals)
    # VI/AsyncVI/PE use ``n_sweeps``; PI/MPI use ``n_iters`` — len(residuals)
    # is the portable count of logged outer iterations.
    n_sweeps_actual = len(residuals)

    cumulative_time = 0.0
    for i in range(n_sweeps_actual):
        sweep_wall = sweep_times[i] if i < len(sweep_times) else 0.0
        cumulative_time += sweep_wall
        residual = residuals[i] if i < len(residuals) else 0.0

        dp_logger.record_sweep(
            sweep_idx=i,
            bellman_residual=residual,
            wall_clock_s=cumulative_time,
            v_current=_v_for_sweep_index(planner, i),
        )

    summary = dp_logger.summary()

    # 7. Build calibration stats from exact DP tables
    calib_stats = build_calibration_stats_from_dp_tables(
        Q=planner.Q,
        V=planner.V,
        P=p,
        R=r,
        gamma=gamma_for_calib,
        horizon=horizon,
    )
    rw.set_calibration_stats(calib_stats)

    # 8. Flush everything
    final_residual = residuals[-1] if residuals else 0.0
    metrics: dict[str, Any] = {
        "final_bellman_residual": float(final_residual),
        "n_sweeps": int(n_sweeps_actual),
        "wall_clock_s": float(planner.wall_clock_s),
        "converged": bool(planner.converged) if hasattr(planner, "converged") else True,
    }
    # Add threshold-crossing sweep counts
    for key, val in summary.items():
        metrics[key] = val

    rw.flush(metrics=metrics)

    print(
        f"    -> done: {n_sweeps_actual} sweeps, "
        f"residual={final_residual:.2e}, "
        f"wall_clock={planner.wall_clock_s:.4f}s"
    )
    print(f"    -> artifacts: {rw.run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase I DP runner: exact planners for the paper suite.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to the suite config JSON (default: paper_suite.json).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=list(TASK_FACTORIES.keys()),
        help="Filter to a single task.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=list(DP_ALGORITHMS.keys()),
        help="Filter to a single DP algorithm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter to a single seed.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/weighted_lse_dp"),
        help="Root directory for result artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run, without executing.",
    )
    parser.add_argument(
        "--gamma-prime",
        type=float,
        default=None,
        help="Optional gamma override for ablation mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = _load_config(config_path)
    suite = config.get("suite", "paper_suite")

    # Build run list
    runs = _build_run_list(
        config,
        task_filter=args.task,
        algorithm_filter=args.algorithm,
        seed_filter=args.seed,
    )

    if not runs:
        print("No runs matched the given filters.")
        sys.exit(0)

    print(f"Phase I DP runner: {len(runs)} run(s) planned.")
    print(f"  config : {config_path}")
    print(f"  out-root: {args.out_root}")
    if args.gamma_prime is not None:
        print(f"  gamma' : {args.gamma_prime}")
    print()

    if args.dry_run:
        print("DRY RUN -- planned executions:")
        for i, run in enumerate(runs, 1):
            print(
                f"  {i:3d}. task={run['task']:<12s}  "
                f"algo={run['algorithm']:<8s}  "
                f"seed={run['seed']}"
            )
        print(f"\nTotal: {len(runs)} run(s). No artifacts written.")
        return

    # Execute
    t_start = time.perf_counter()
    n_ok = 0
    n_fail = 0

    for run in runs:
        try:
            _run_single(
                task_name=run["task"],
                algo_name=run["algorithm"],
                seed=run["seed"],
                task_cfg=run["task_cfg"],
                out_root=args.out_root,
                suite=suite,
                gamma_prime=args.gamma_prime,
                full_config=config,
            )
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(
                f"  [FAIL] task={run['task']}  algo={run['algorithm']}  "
                f"seed={run['seed']}: {exc}",
                file=sys.stderr,
            )
            import traceback
            traceback.print_exc(file=sys.stderr)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed, {elapsed:.1f}s total.")


if __name__ == "__main__":
    main()
