#!/usr/bin/env python
"""Phase II dynamic-programming runner: exact planners for stress tasks.

For each (task, dp_algorithm, seed) in the Phase II paper suite, this runner:

1. Loads the task via the appropriate stress-task factory.
2. Runs the DP planner (PE, VI, PI, MPI, or AsyncVI).
3. Records planning curves via :class:`DPCurvesLogger`.
4. Generates calibration stats via :func:`build_calibration_stats_from_dp_tables`.
5. Flushes everything via :class:`RunWriter`.

For regime-shift tasks (``warmstart_dp=True``), the runner:
- Runs planners on the pre-shift MDP first.
- Warm-starts planners on the post-shift MDP from the pre-shift solution.
- Logs pre- and post-shift runs as separate artifacts with suffixed task names.

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase2_dp.py \\
        [--config PATH]            # default: paper_suite.json
        [--task TASK | all]        # filter to one task or run all
        [--algorithm ALG]          # filter to one DP algorithm
        [--seed SEED]              # filter to one seed
        [--out-root PATH]          # default: results/weighted_lse_dp
        [--dry-run]                # print what would run, no execution
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
    get_task_sign,
)
from experiments.weighted_lse_dp.common.schemas import RunWriter
from experiments.weighted_lse_dp.common.seeds import seed_everything
from experiments.weighted_lse_dp.common.task_factories import build_ref_pi_for_task

# Phase II stress-task factories
from experiments.weighted_lse_dp.tasks.stress_families import (
    make_chain_sparse_long,
    make_chain_jackpot,
    make_chain_catastrophe,
    make_grid_sparse_goal,
    make_taxi_bonus_shock,
)
from experiments.weighted_lse_dp.tasks.nonstationary_wrappers import (
    make_chain_regime_shift,
    make_grid_regime_shift,
)
from experiments.weighted_lse_dp.tasks.hazard_wrappers import (
    make_grid_hazard,
)

from mushroom_rl.algorithms.value.dp import (
    ClassicalAsyncValueIteration,
    ClassicalModifiedPolicyIteration,
    ClassicalPolicyEvaluation,
    ClassicalPolicyIteration,
    ClassicalValueIteration,
    extract_mdp_arrays,
)
from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    deterministic_policy_array,
)


# ---------------------------------------------------------------------------
# Registry: task factories and DP planner constructors
# ---------------------------------------------------------------------------

#: Tasks whose factories return (wrapper, mdp_rl, cfg) with a regime-shift
#: wrapper that exposes ``._pre`` and ``._post`` FiniteMDP attributes.
_REGIME_SHIFT_TASKS: frozenset[str] = frozenset({
    "chain_regime_shift",
    "grid_regime_shift",
})

#: Tasks excluded from DP because their stress dynamics are injected at
#: runtime via wrapper.step() and cannot be encoded in the P/R kernel
#: without redesigning the factory.  These tasks are RL-only.
_RL_ONLY_TASKS: frozenset[str] = frozenset({
    "grid_hazard",
    "taxi_bonus_shock",
})


def _call_factory(
    task_name: str,
    task_cfg: dict[str, Any],
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Dispatch to the correct Phase II factory.

    Returns
    -------
    (mdp_or_wrapper, mdp_rl, resolved_cfg)
        For bare MDPs, ``mdp_or_wrapper`` is a FiniteMDP.
        For wrapper tasks, it is the wrapper object.
    """
    cfg_copy = dict(task_cfg)

    if task_name == "chain_sparse_long":
        return make_chain_sparse_long(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 60)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 120)),
        )
    elif task_name == "chain_jackpot":
        return make_chain_jackpot(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 25)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 60)),
            jackpot_state=int(task_cfg.get("jackpot_state", 20)),
            jackpot_prob=float(task_cfg.get("jackpot_prob", 0.05)),
            jackpot_reward=float(task_cfg.get("jackpot_reward", 10.0)),
            jackpot_terminates=bool(task_cfg.get("jackpot_terminates", True)),
        )
    elif task_name == "chain_catastrophe":
        return make_chain_catastrophe(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 25)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 60)),
            risky_state=int(task_cfg.get("risky_state", 15)),
            risky_prob=float(task_cfg.get("risky_prob", 0.05)),
            catastrophe_reward=float(task_cfg.get("catastrophe_reward", -10.0)),
            shortcut_jump=int(task_cfg.get("shortcut_jump", 5)),
        )
    elif task_name == "chain_regime_shift":
        wrapper, mdp_rl, resolved = make_chain_regime_shift(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 25)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 60)),
            change_at_episode=int(task_cfg.get("change_at_episode", 500)),
            shift_type=str(task_cfg.get("shift_type", "goal_flip")),
            post_prob=float(task_cfg.get("post_prob", 0.7)),
            time_augment=False,
        )
        return wrapper, mdp_rl, resolved
    elif task_name == "grid_sparse_goal":
        return make_grid_sparse_goal(
            cfg=cfg_copy,
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 80)),
            goal_reward=float(task_cfg.get("goal_reward", 1.0)),
            time_augment=False,
            seed=seed,
        )
    elif task_name == "grid_hazard":
        wrapper, _mdp_rl, resolved = make_grid_hazard(
            cfg=cfg_copy,
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 80)),
            hazard_states=task_cfg.get("hazard_states"),
            hazard_prob=float(task_cfg.get("hazard_prob", 1.0)),
            hazard_reward=float(task_cfg.get("hazard_reward", -5.0)),
            hazard_terminates=bool(task_cfg.get("hazard_terminates", True)),
            time_augment=False,
            seed=seed,
        )
        return wrapper, _mdp_rl, resolved
    elif task_name == "grid_regime_shift":
        wrapper, mdp_rl, resolved = make_grid_regime_shift(
            cfg=cfg_copy,
            change_at_episode=int(task_cfg.get("change_at_episode", 300)),
            shift_type=str(task_cfg.get("shift_type", "goal_move")),
            time_augment=False,
        )
        return wrapper, mdp_rl, resolved
    elif task_name == "taxi_bonus_shock":
        wrapper, _mdp_rl, resolved = make_taxi_bonus_shock(
            cfg=cfg_copy,
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 120)),
            bonus_prob=float(task_cfg.get("bonus_prob", 0.05)),
            bonus_reward=float(task_cfg.get("bonus_reward", 5.0)),
            time_augment=False,
            seed=seed,
        )
        return wrapper, _mdp_rl, resolved
    else:
        raise ValueError(f"Unknown Phase II task: {task_name!r}")


def _get_base_mdp(task_name: str, mdp_or_wrapper: Any) -> Any:
    """Extract the FiniteMDP used for DP planning from the factory output.

    For bare MDPs this is the identity.  Regime-shift wrappers are handled
    by the caller (pre/post split).  RL-only tasks (grid_hazard,
    taxi_bonus_shock) never reach this function -- they are filtered out
    before dispatch.
    """
    # For bare MDPs, return as-is.
    return mdp_or_wrapper


# ---------------------------------------------------------------------------
# All known Phase II task names (for CLI --task validation)
# ---------------------------------------------------------------------------
TASK_NAMES: list[str] = [
    "chain_sparse_long",
    "chain_jackpot",
    "chain_catastrophe",
    "chain_regime_shift",
    "grid_sparse_goal",
    "grid_hazard",
    "grid_regime_shift",
    "taxi_bonus_shock",
]

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
    / "phase2"
    / "paper_suite.json"
)

_DEFAULT_OUT_ROOT = "results/weighted_lse_dp"


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
    """Return a list of ``{task, algorithm, seed, task_cfg}`` dicts to execute."""
    seeds = config.get("seeds", [11, 29, 47])
    tasks_cfg = config.get("tasks", {})

    runs: list[dict[str, Any]] = []

    for task_name, task_cfg in tasks_cfg.items():
        if task_filter is not None and task_filter != "all" and task_name != task_filter:
            continue
        if task_name not in TASK_NAMES:
            continue
        # Skip RL-only tasks (runtime-injected stress, no DP encoding).
        if task_name in _RL_ONLY_TASKS:
            continue

        # Chain tasks use the extended seed list if available.
        if task_name.startswith("chain_"):
            task_seeds = config.get("chain_seeds", seeds)
        else:
            task_seeds = seeds

        dp_algos = task_cfg.get("dp_algorithms", [])
        for algo_name in dp_algos:
            if algorithm_filter is not None and algo_name != algorithm_filter:
                continue
            if algo_name not in DP_ALGORITHMS:
                continue

            for seed in task_seeds:
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
# Reference policy builder (needed for PE)
# ---------------------------------------------------------------------------


def _build_ref_pi(task_name: str, task_cfg: dict, mdp: Any) -> np.ndarray:
    """Build the reference policy for PE by dispatching on ``task_cfg['ref_policy']``.

    Delegates to :func:`build_ref_pi_for_task` which handles
    ``"always_right"`` (chains), ``"shortest_path"`` (grids), and
    ``"pickup_then_deliver"`` (taxi).

    Returns shape ``(H, S)`` int64 array.
    """
    return build_ref_pi_for_task(task_name, task_cfg, mdp)


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------


def _make_planner(
    algo_name: str,
    mdp: Any,
    ref_pi: np.ndarray,
    *,
    v_init: np.ndarray | None = None,
) -> Any:
    """Construct a DP planner instance.

    Parameters
    ----------
    algo_name : str
        Short name from :data:`DP_ALGORITHMS`.
    mdp :
        The FiniteMDP environment.
    ref_pi :
        Reference policy array of shape ``(H, S)``.  Used only for PE.
    v_init :
        Optional warm-start value table of shape ``(H+1, S)``.
        Passed to VI, PI, MPI, AsyncVI constructors.  Ignored for PE
        (PE does exact backward induction, no iterative warm-start).

    Returns
    -------
    planner
        A planner instance ready for ``.run()``.
    """
    if algo_name == "PE":
        planner = ClassicalPolicyEvaluation(mdp, pi=ref_pi)
    elif algo_name == "VI":
        planner = ClassicalValueIteration(mdp, v_init=v_init)
    elif algo_name == "PI":
        planner = ClassicalPolicyIteration(mdp, v_init=v_init)
    elif algo_name == "MPI":
        planner = ClassicalModifiedPolicyIteration(mdp, v_init=v_init)
    elif algo_name == "AsyncVI":
        planner = ClassicalAsyncValueIteration(mdp, v_init=v_init)
    else:
        raise ValueError(f"Unknown DP algorithm: {algo_name!r}")

    return planner


# ---------------------------------------------------------------------------
# Helpers (mirrored from Phase I runner)
# ---------------------------------------------------------------------------


def _get_sweep_times(planner: Any) -> list[float]:
    """Extract per-sweep wall-clock times from any planner type."""
    if hasattr(planner, "sweep_times_s"):
        return list(planner.sweep_times_s)
    if hasattr(planner, "iter_times_s"):
        return list(planner.iter_times_s)
    return []


def _v_for_sweep_index(planner: Any, sweep_idx: int) -> Any:
    """Return the value table after sweep ``sweep_idx``."""
    hist = getattr(planner, "V_sweep_history", None)
    if hist is not None and sweep_idx < len(hist):
        return hist[sweep_idx]
    return planner.V


def _run_dp_on_mdp(
    mdp: Any,
    algo_name: str,
    ref_pi: np.ndarray,
    *,
    v_init: np.ndarray | None = None,
    v_exact: np.ndarray | None = None,
    task_label: str,
    seed: int,
    task_cfg: dict[str, Any],
    out_root: Path,
    suite: str,
    resolved_cfg: dict[str, Any],
    full_config: dict,
    canonical_task_family: str | None = None,
    regime_phase: str | None = None,
) -> np.ndarray:
    """Run a single DP algorithm on a single MDP and flush artifacts.

    For regime-shift tasks the ``task_label`` carries a ``_pre_shift`` /
    ``_post_shift`` suffix so each phase lands in a separate directory.
    Pass ``canonical_task_family`` (the undecorated task name, e.g.
    ``"chain_regime_shift"``) and ``regime_phase`` (``"pre_shift"`` or
    ``"post_shift"``) so that aggregation can re-group the runs under the
    canonical family name (R5-3 fix).

    Returns the planner's final ``V`` table (for downstream warm-start use).
    """
    print(f"  [RUN] task={task_label}  algo={algo_name}  seed={seed}")

    # Build resolved config for this run
    run_config: dict[str, Any] = {
        **resolved_cfg,
        "suite_config_path": str(_DEFAULT_CONFIG),
    }
    # R5-3: persist canonical family name + phase so aggregation can group
    # pre/post regime-shift runs under the single canonical task family.
    if canonical_task_family is not None:
        run_config["canonical_task_family"] = canonical_task_family
    if regime_phase is not None:
        run_config["regime_phase"] = regime_phase

    # Create RunWriter
    rw = RunWriter.create(
        base=out_root,
        phase="phase2",
        suite=suite,
        task=task_label,
        algorithm=algo_name,
        seed=seed,
        config=run_config,
        storage_mode="dp_stagewise",
    )

    # Construct the planner
    planner = _make_planner(algo_name, mdp, ref_pi, v_init=v_init)

    # Run the planner
    with rw.timer.phase("fit"):
        planner.run()

    # Extract MDP arrays for calibration stats
    p, r, horizon, gamma_mdp = extract_mdp_arrays(mdp)

    # Compute V* reference for supnorm_to_exact (R4-1 fix).
    # For non-VI planners with no provided v_exact, the old code fell back to
    # planner.V (self-reference), making supnorm always collapse to 0.  Instead,
    # run VI once to get the true V* and pass it to every planner in the group.
    if v_exact is not None:
        _v_exact_ref: np.ndarray | None = v_exact
    elif algo_name == "VI":
        # VI is computing V* itself; use its own final table (self-reference
        # correctly tracks how far each sweep is from the final solution).
        _v_exact_ref = None  # resolved to planner.V below
    else:
        print(f"    [dp] Computing VI reference for supnorm_to_exact ({algo_name})...")
        _vi_ref = _make_planner("VI", mdp, ref_pi, v_init=None)
        _vi_ref.run()
        _v_exact_ref = _vi_ref.V

    dp_logger = DPCurvesLogger(
        run_writer=rw,
        v_exact=_v_exact_ref if _v_exact_ref is not None else planner.V,
        task=task_label,
    )

    sweep_times = _get_sweep_times(planner)
    residuals = list(planner.residuals)
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

    # Build calibration stats from exact DP tables
    calib_stats = build_calibration_stats_from_dp_tables(
        Q=planner.Q,
        V=planner.V,
        P=p,
        R=r,
        gamma=gamma_mdp,
        horizon=horizon,
        sign=get_task_sign(task_label),
    )
    rw.set_calibration_stats(calib_stats)

    # Flush everything
    final_residual = residuals[-1] if residuals else 0.0
    metrics: dict[str, Any] = {
        "final_bellman_residual": float(final_residual),
        "n_sweeps": int(n_sweeps_actual),
        "wall_clock_s": float(planner.wall_clock_s),
        "converged": bool(planner.converged) if hasattr(planner, "converged") else True,
    }
    for key, val in summary.items():
        metrics[key] = val

    rw.flush(metrics=metrics)

    print(
        f"    -> done: {n_sweeps_actual} sweeps, "
        f"residual={final_residual:.2e}, "
        f"wall_clock={planner.wall_clock_s:.4f}s"
    )
    print(f"    -> artifacts: {rw.run_dir}")

    return planner.V


# ---------------------------------------------------------------------------
# Single-run execution (with regime-shift warm-start logic)
# ---------------------------------------------------------------------------


def _run_single(
    task_name: str,
    algo_name: str,
    seed: int,
    task_cfg: dict[str, Any],
    *,
    out_root: Path,
    suite: str,
    full_config: dict,
) -> None:
    """Execute one (task, algorithm, seed) DP run and flush artifacts.

    For regime-shift tasks with ``warmstart_dp=True``, this runs the planner
    twice: once on the pre-shift MDP, once on the post-shift MDP with
    ``v_init`` from the pre-shift solution.

    Returns
    -------
    dict
        Summary dict with ``task``, ``algorithm``, ``seed``, ``passed``,
        and optionally ``skipped`` / ``wall_s``.
    """
    # BLOCKER B fix: RL-only tasks have runtime-injected stress that
    # cannot be represented in the P/R kernel; skip them for DP.
    if task_name in _RL_ONLY_TASKS:
        print(f"[phase2_dp] {task_name}: RL-only task (runtime-injected stress; no DP run).")
        return {
            "task": task_name,
            "algorithm": algo_name,
            "seed": seed,
            "passed": True,
            "skipped": "RL-only task",
            "wall_s": 0.0,
        }

    seed_everything(seed)

    mdp_or_wrapper, _mdp_rl, resolved_cfg = _call_factory(
        task_name, task_cfg, seed,
    )

    is_regime_shift = task_name in _REGIME_SHIFT_TASKS
    warmstart = bool(task_cfg.get("warmstart_dp", False))

    if is_regime_shift and warmstart:
        # --- Pre-shift run ---
        pre_mdp = mdp_or_wrapper._pre
        ref_pi_pre = _build_ref_pi(task_name, task_cfg, pre_mdp)

        # Run VI first to get exact V* for supnorm computation on other algos.
        vi_v_pre: np.ndarray | None = None
        if algo_name == "VI":
            # This is the VI run itself; v_exact = planner.V (self-reference).
            v_pre = _run_dp_on_mdp(
                mdp=pre_mdp,
                algo_name="VI",
                ref_pi=ref_pi_pre,
                v_exact=None,
                task_label=f"{task_name}_pre_shift",
                seed=seed,
                task_cfg=task_cfg,
                out_root=out_root,
                suite=suite,
                resolved_cfg=resolved_cfg,
                full_config=full_config,
                canonical_task_family=task_name,
                regime_phase="pre_shift",
            )
        else:
            v_pre = _run_dp_on_mdp(
                mdp=pre_mdp,
                algo_name=algo_name,
                ref_pi=ref_pi_pre,
                v_exact=None,
                task_label=f"{task_name}_pre_shift",
                seed=seed,
                task_cfg=task_cfg,
                out_root=out_root,
                suite=suite,
                resolved_cfg=resolved_cfg,
                full_config=full_config,
                canonical_task_family=task_name,
                regime_phase="pre_shift",
            )

        # --- Post-shift run (warm-started from pre-shift V) ---
        post_mdp = mdp_or_wrapper._post
        ref_pi_post = _build_ref_pi(task_name, task_cfg, post_mdp)

        # PE does not support v_init; pass None for PE.
        v_init_post = v_pre if algo_name != "PE" else None

        _run_dp_on_mdp(
            mdp=post_mdp,
            algo_name=algo_name,
            ref_pi=ref_pi_post,
            v_init=v_init_post,
            v_exact=None,
            task_label=f"{task_name}_post_shift",
            seed=seed,
            task_cfg=task_cfg,
            out_root=out_root,
            suite=suite,
            resolved_cfg=resolved_cfg,
            full_config=full_config,
            canonical_task_family=task_name,
            regime_phase="post_shift",
        )
    elif is_regime_shift:
        # --- Regime-shift without warmstart: run on the pre-shift MDP ---
        # The regime-shift wrapper exposes ._pre and ._post sub-MDPs.
        # Without warmstart we simply run on the pre-shift MDP (the baseline
        # regime); this is well-defined and avoids calling extract_mdp_arrays
        # on the wrapper itself (which does not expose p/r directly).
        pre_mdp = mdp_or_wrapper._pre
        ref_pi = _build_ref_pi(task_name, task_cfg, pre_mdp)

        _run_dp_on_mdp(
            mdp=pre_mdp,
            algo_name=algo_name,
            ref_pi=ref_pi,
            v_exact=None,
            task_label=f"{task_name}_pre_shift",
            seed=seed,
            task_cfg=task_cfg,
            out_root=out_root,
            suite=suite,
            resolved_cfg=resolved_cfg,
            full_config=full_config,
            canonical_task_family=task_name,
            regime_phase="pre_shift",
        )
    else:
        # --- Standard (non-regime-shift) run ---
        mdp = _get_base_mdp(task_name, mdp_or_wrapper)
        ref_pi = _build_ref_pi(task_name, task_cfg, mdp)

        _run_dp_on_mdp(
            mdp=mdp,
            algo_name=algo_name,
            ref_pi=ref_pi,
            v_exact=None,  # v_exact=planner.V used inside _run_dp_on_mdp
            task_label=task_name,
            seed=seed,
            task_cfg=task_cfg,
            out_root=out_root,
            suite=suite,
            resolved_cfg=resolved_cfg,
            full_config=full_config,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase II DP runner: exact planners for stress tasks.",
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
        help=(
            "Filter to a single task, or 'all' to run every task. "
            f"Choices: {', '.join(TASK_NAMES)}, all"
        ),
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
        default=Path(_DEFAULT_OUT_ROOT),
        help="Root directory for result artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run, without executing.",
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

    # Validate --task
    if args.task is not None and args.task != "all" and args.task not in TASK_NAMES:
        print(
            f"ERROR: unknown task {args.task!r}. "
            f"Choose from: {', '.join(TASK_NAMES)}, all",
            file=sys.stderr,
        )
        sys.exit(1)

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

    # Count regime-shift runs that will produce 2 sub-runs each.
    n_warmstart = sum(
        1 for r in runs
        if r["task"] in _REGIME_SHIFT_TASKS
        and r["task_cfg"].get("warmstart_dp", False)
    )
    n_artifacts = len(runs) + n_warmstart  # each warmstart produces pre+post

    print(f"Phase II DP runner: {len(runs)} run(s) planned "
          f"({n_artifacts} artifact sets including warm-start splits).")
    print(f"  config : {config_path}")
    print(f"  out-root: {args.out_root}")
    print()

    if args.dry_run:
        print("DRY RUN -- planned executions:")
        for i, run in enumerate(runs, 1):
            warmstart_flag = ""
            if (run["task"] in _REGIME_SHIFT_TASKS
                    and run["task_cfg"].get("warmstart_dp", False)):
                warmstart_flag = "  [warmstart: pre+post]"
            print(
                f"  {i:3d}. task={run['task']:<24s}  "
                f"algo={run['algorithm']:<8s}  "
                f"seed={run['seed']}{warmstart_flag}"
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


# Public alias expected by callers and import checks.
run_single = _run_single


if __name__ == "__main__":
    main()
