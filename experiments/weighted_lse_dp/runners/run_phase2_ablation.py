#!/usr/bin/env python
"""Phase II fixed-gamma' ablation runner for stress tasks.

Sweeps over the ``gamma_prime_values`` grid from ``paper_suite.json`` and
the ``hyperparameter_retuning`` grid (lr_multiplier x epsilon) for each
(task, algorithm, seed) combination.

Two ablation modes:

1. **Gamma-prime grid** (all algorithms): runs each (task, algo, seed,
   gamma') with gamma' in {0.90, 0.95, 0.99}.

2. **Hyperparameter retuning** (RL algorithms only): for QLearning and
   ExpectedSARSA, additionally sweeps lr_multiplier x epsilon at the
   base gamma' = 0.99.

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase2_ablation.py \\
        --task chain_jackpot --seed 11

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase2_ablation.py \\
        --task all --seed 11

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase2_ablation.py \\
        --task chain_jackpot --seed 11 --gamma-prime 0.95

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase2_ablation.py \\
        --task chain_jackpot --seed 11 --hparam-only

Output path layouts::

    Gamma ablation:
      results/weighted_lse_dp/phase2/ablation/gamma0.90/<task>/<algorithm>/seed_<N>/
      results/weighted_lse_dp/phase2/ablation/gamma0.95/<task>/<algorithm>/seed_<N>/
      results/weighted_lse_dp/phase2/ablation/gamma0.99/<task>/<algorithm>/seed_<N>/

    Hyperparameter retuning:
      results/weighted_lse_dp/phase2/ablation/hparam/<task>/<algorithm>/lr0.5/eps0.05/seed_<N>/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "mushroom-rl-dev"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = (
    "experiments/weighted_lse_dp/configs/phase2/paper_suite.json"
)
_DEFAULT_OUT_ROOT = "results/weighted_lse_dp"

#: DP algorithms for gamma-prime ablation.
DP_ALGORITHMS: frozenset[str] = frozenset({
    "PE", "VI", "PI", "MPI", "AsyncVI",
})

#: RL algorithms for both gamma-prime and hyperparameter ablation.
RL_ALGORITHMS: frozenset[str] = frozenset({
    "QLearning", "ExpectedSARSA",
})

ALL_ALGORITHMS: frozenset[str] = DP_ALGORITHMS | RL_ALGORITHMS

#: Default gamma' for hyperparameter retuning runs.
HPARAM_BASE_GAMMA_PRIME: float = 0.99

#: All Phase II stress tasks.
STRESS_TASKS: tuple[str, ...] = (
    "chain_sparse_long",
    "chain_jackpot",
    "chain_catastrophe",
    "chain_regime_shift",
    "grid_sparse_goal",
    "grid_hazard",
    "grid_regime_shift",
    "taxi_bonus_shock",
)


# ---------------------------------------------------------------------------
# Task factory dispatch
# ---------------------------------------------------------------------------


def _make_task(
    task_name: str,
    task_cfg: dict[str, Any],
    seed: int,
    *,
    time_augment: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Dispatch to the correct Phase II task factory.

    Returns ``(mdp_base_or_wrapper, mdp_rl, resolved_cfg)``.
    """
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

    if task_name == "chain_sparse_long":
        return make_chain_sparse_long(
            cfg=task_cfg,
            state_n=task_cfg.get("state_n", 60),
            prob=task_cfg.get("prob", 0.9),
            gamma=task_cfg.get("gamma", 0.99),
            horizon=task_cfg.get("horizon", 120),
        )
    elif task_name == "chain_jackpot":
        return make_chain_jackpot(
            cfg=task_cfg,
            state_n=task_cfg.get("state_n", 25),
            prob=task_cfg.get("prob", 0.9),
            gamma=task_cfg.get("gamma", 0.99),
            horizon=task_cfg.get("horizon", 60),
            jackpot_state=task_cfg.get("jackpot_state", 20),
            jackpot_prob=task_cfg.get("jackpot_prob", 0.05),
            jackpot_reward=task_cfg.get("jackpot_reward", 10.0),
            jackpot_terminates=task_cfg.get("jackpot_terminates", True),
        )
    elif task_name == "chain_catastrophe":
        return make_chain_catastrophe(
            cfg=task_cfg,
            state_n=task_cfg.get("state_n", 25),
            prob=task_cfg.get("prob", 0.9),
            gamma=task_cfg.get("gamma", 0.99),
            horizon=task_cfg.get("horizon", 60),
            risky_state=task_cfg.get("risky_state", 15),
            risky_prob=task_cfg.get("risky_prob", 0.05),
            catastrophe_reward=task_cfg.get("catastrophe_reward", -10.0),
            shortcut_jump=task_cfg.get("shortcut_jump", 5),
        )
    elif task_name == "chain_regime_shift":
        return make_chain_regime_shift(
            cfg=task_cfg,
            state_n=task_cfg.get("state_n", 25),
            prob=task_cfg.get("prob", 0.9),
            gamma=task_cfg.get("gamma", 0.99),
            horizon=task_cfg.get("horizon", 60),
            change_at_episode=task_cfg.get("change_at_episode", 300),
            shift_type=task_cfg.get("shift_type", "goal_flip"),
            post_prob=task_cfg.get("post_prob", 0.9),
            time_augment=time_augment,
        )
    elif task_name == "grid_sparse_goal":
        return make_grid_sparse_goal(
            cfg=task_cfg,
            grid_file=task_cfg.get(
                "grid_file",
                "experiments/weighted_lse_dp/assets/grids/phase1_base_grid.txt",
            ),
            prob=task_cfg.get("prob", 0.9),
            gamma=task_cfg.get("gamma", 0.99),
            horizon=task_cfg.get("horizon", 80),
            goal_reward=task_cfg.get("goal_reward", 1.0),
            time_augment=time_augment,
            seed=seed,
        )
    elif task_name == "grid_hazard":
        return make_grid_hazard(
            cfg=task_cfg,
            prob=task_cfg.get("prob", 0.9),
            gamma=task_cfg.get("gamma", 0.99),
            horizon=task_cfg.get("horizon", 80),
            hazard_states=task_cfg.get("hazard_states", [12]),
            hazard_prob=task_cfg.get("hazard_prob", 0.10),
            hazard_reward=task_cfg.get("hazard_reward", -5.0),
            hazard_terminates=task_cfg.get("hazard_terminates", False),
            time_augment=time_augment,
            seed=seed,
        )
    elif task_name == "grid_regime_shift":
        return make_grid_regime_shift(
            cfg=task_cfg,
            change_at_episode=task_cfg.get("change_at_episode", 200),
            shift_type=task_cfg.get("shift_type", "goal_move"),
            time_augment=time_augment,
        )
    elif task_name == "taxi_bonus_shock":
        return make_taxi_bonus_shock(
            cfg=task_cfg,
            grid_file=task_cfg.get(
                "grid_file",
                "experiments/weighted_lse_dp/assets/grids/phase1_taxi_grid.txt",
            ),
            prob=task_cfg.get("prob", 0.9),
            gamma=task_cfg.get("gamma", 0.99),
            horizon=task_cfg.get("horizon", 120),
            bonus_prob=task_cfg.get("bonus_prob", 0.05),
            bonus_reward=task_cfg.get("bonus_reward", 5.0),
            time_augment=time_augment,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown Phase II stress task: {task_name!r}")


# ---------------------------------------------------------------------------
# Gamma-prime encoding
# ---------------------------------------------------------------------------


def _encode_gamma(gamma: float) -> str:
    """Encode gamma as a suite-safe string: 0.95 -> 'gamma0.95'."""
    return f"gamma{gamma:.2f}"


# ---------------------------------------------------------------------------
# Plan builders
# ---------------------------------------------------------------------------


def _build_gamma_plan(
    config: dict[str, Any],
    *,
    task_filter: str | None = None,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
    gamma_prime_filter: float | None = None,
) -> list[dict[str, Any]]:
    """Build the gamma-prime ablation matrix.

    Returns a list of dicts with keys:
    ``task``, ``algorithm``, ``seed``, ``gamma_prime``, ``task_config``,
    ``ablation_type`` (always ``"gamma"``).
    """
    gamma_primes: list[float] = config.get("gamma_prime_values", [0.90, 0.95, 0.99])
    if gamma_prime_filter is not None:
        gamma_primes = [gp for gp in gamma_primes if abs(gp - gamma_prime_filter) < 1e-6]

    tasks_cfg: dict[str, Any] = config.get("tasks", {})
    plan: list[dict[str, Any]] = []

    for task_name in sorted(tasks_cfg):
        if task_name not in STRESS_TASKS:
            continue
        if task_filter is not None and task_filter != "all" and task_name != task_filter:
            continue

        task_cfg = tasks_cfg[task_name]

        # Collect ablation algorithms: union of dp_algorithms and rl_algorithms.
        dp_algos = set(task_cfg.get("dp_algorithms", []))
        rl_algos = set(task_cfg.get("rl_algorithms", []))
        all_task_algos = (dp_algos | rl_algos) & ALL_ALGORITHMS

        # Use chain_seeds for chain tasks, seeds for others.
        if task_name.startswith("chain_"):
            seeds: list[int] = config.get("chain_seeds", config.get("seeds", [11, 29, 47]))
        else:
            seeds = config.get("seeds", [11, 29, 47])

        for algo_name in sorted(all_task_algos):
            if algorithm_filter is not None and algo_name != algorithm_filter:
                continue

            for seed in seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue

                for gp in gamma_primes:
                    plan.append({
                        "task": task_name,
                        "algorithm": algo_name,
                        "seed": seed,
                        "gamma_prime": gp,
                        "task_config": task_cfg,
                        "ablation_type": "gamma",
                    })

    return plan


def _build_hparam_plan(
    config: dict[str, Any],
    *,
    task_filter: str | None = None,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Build the hyperparameter retuning matrix (RL algorithms only).

    Returns a list of dicts with keys:
    ``task``, ``algorithm``, ``seed``, ``gamma_prime``, ``lr_multiplier``,
    ``epsilon``, ``task_config``, ``ablation_type`` (always ``"hparam"``).
    """
    hparam_cfg = config.get("hyperparameter_retuning", {})
    lr_multipliers: list[float] = hparam_cfg.get("lr_multiplier", [0.5, 1.0, 2.0])
    epsilons: list[float] = hparam_cfg.get("epsilon", [0.05, 0.10, 0.15])

    tasks_cfg: dict[str, Any] = config.get("tasks", {})
    plan: list[dict[str, Any]] = []

    for task_name in sorted(tasks_cfg):
        if task_name not in STRESS_TASKS:
            continue
        if task_filter is not None and task_filter != "all" and task_name != task_filter:
            continue

        task_cfg = tasks_cfg[task_name]
        rl_algos = set(task_cfg.get("rl_algorithms", [])) & RL_ALGORITHMS

        # Use chain_seeds for chain tasks, seeds for others.
        if task_name.startswith("chain_"):
            seeds: list[int] = config.get("chain_seeds", config.get("seeds", [11, 29, 47]))
        else:
            seeds = config.get("seeds", [11, 29, 47])

        for algo_name in sorted(rl_algos):
            if algorithm_filter is not None and algo_name != algorithm_filter:
                continue

            for seed in seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue

                for lr in lr_multipliers:
                    for eps in epsilons:
                        plan.append({
                            "task": task_name,
                            "algorithm": algo_name,
                            "seed": seed,
                            "gamma_prime": HPARAM_BASE_GAMMA_PRIME,
                            "lr_multiplier": lr,
                            "epsilon": eps,
                            "task_config": task_cfg,
                            "ablation_type": "hparam",
                        })

    return plan


# ---------------------------------------------------------------------------
# Single-run dispatchers
# ---------------------------------------------------------------------------


def _run_gamma_ablation(
    entry: dict[str, Any],
    *,
    out_root: Path,
    full_config: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a single gamma-prime ablation run (DP or RL)."""
    task = entry["task"]
    algo = entry["algorithm"]
    seed = entry["seed"]
    gp = entry["gamma_prime"]
    task_cfg = entry["task_config"]

    gp_str = _encode_gamma(gp)
    suite = f"ablation/{gp_str}"

    # Inject gamma_prime as the effective MDP gamma so Phase II factories
    # and planners use the ablation discount rate.
    task_cfg_gp = dict(task_cfg)
    task_cfg_gp["gamma"] = gp
    task_cfg_gp["gamma_prime_override"] = gp  # logged in run.json

    if algo in DP_ALGORITHMS:
        from experiments.weighted_lse_dp.runners.run_phase2_dp import (
            _run_single as dp_run_single,
        )

        dp_run_single(
            task_name=task,
            algo_name=algo,
            seed=seed,
            task_cfg=task_cfg_gp,
            out_root=out_root,
            suite=suite,
            full_config=full_config,
        )
        return {
            "task": task,
            "algorithm": algo,
            "seed": seed,
            "gamma_prime": gp,
            "ablation_type": "gamma",
            "passed": True,
        }

    elif algo in RL_ALGORITHMS:
        from experiments.weighted_lse_dp.runners.run_phase2_rl import (
            run_single as rl_run_single,
        )

        result = rl_run_single(
            task=task,
            algorithm=algo,
            seed=seed,
            task_config=task_cfg_gp,
            out_root=out_root,
            suite=suite,
        )
        return {
            "task": task,
            "algorithm": algo,
            "seed": seed,
            "gamma_prime": gp,
            "ablation_type": "gamma",
            "passed": result.get("passed", False),
            "run_dir": result.get("run_dir"),
        }

    else:
        raise ValueError(f"Unknown algorithm family for {algo!r}")


def _run_hparam_ablation(
    entry: dict[str, Any],
    *,
    out_root: Path,
    full_config: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a single hyperparameter retuning run (RL only).

    Output path:
      <out_root>/phase2/ablation/hparam/<task>/<algorithm>/lr<lr>/eps<eps>/seed_<seed>/

    Since RunWriter.create builds:
      <base> / <phase> / <suite> / <task> / <algorithm> / seed_<seed>
    we encode lr/eps into a path-compatible suite string.
    """
    task = entry["task"]
    algo = entry["algorithm"]
    seed = entry["seed"]
    gp = entry["gamma_prime"]
    lr = entry["lr_multiplier"]
    eps = entry["epsilon"]
    task_cfg = entry["task_config"]

    # Encode lr and eps into the suite path so the RunWriter produces:
    #   <out_root>/phase2/ablation/hparam/lr<lr>/eps<eps>/<task>/<algo>/seed_<seed>
    # Note: RunWriter path = base / phase / suite / task / algorithm / seed_<seed>.
    # To get lr/eps *inside* the task/algo subtree, we encode them in the
    # suite. This gives:
    #   <out_root>/phase2/ablation/hparam/<task>/<algo>/seed_<seed>
    # which loses the lr/eps distinction.
    #
    # Alternative: encode lr/eps in the algorithm name.
    # Alternative: use a custom run_dir.
    #
    # Best approach: use RunWriter directly with a manually constructed
    # run_dir that includes the lr/eps levels.

    from experiments.weighted_lse_dp.common.io import make_run_dir
    from experiments.weighted_lse_dp.common.schemas import RunWriter
    from experiments.weighted_lse_dp.common.seeds import seed_everything

    seed_everything(seed)

    lr_str = f"lr{lr:.1f}".replace(".", "")
    eps_str = f"eps{eps:.2f}".replace(".", "")

    # Build suite with lr/eps sub-levels.
    suite = f"ablation/hparam/{task}/{algo}/{lr_str}/{eps_str}"

    # Merge hparam overrides into the task config for the RL runner.
    hparam_task_cfg = dict(task_cfg)
    hparam_task_cfg["lr_multiplier"] = lr
    hparam_task_cfg["epsilon_override"] = eps

    run_config = {
        **hparam_task_cfg,
        "gamma_prime": gp,
        "lr_multiplier": lr,
        "epsilon": eps,
        "ablation_type": "hparam",
    }

    # Create RunWriter with the hparam-specific suite so path is:
    #   <out_root>/phase2/<suite>/seed_<seed>
    # which expands to:
    #   <out_root>/phase2/ablation/hparam/<task>/<algo>/<lr_str>/<eps_str>/seed_<seed>
    # (RunWriter adds task and algorithm levels, so we must NOT include
    # them in the suite string. But we want lr/eps nested under task/algo.)
    #
    # RunWriter path = base / phase / suite / task / algorithm / seed_<seed>
    # Desired:        base / phase2 / ablation/hparam / task / algo / lr / eps / seed_<seed>
    #
    # We can either:
    # (a) Encode lr/eps in the algorithm name: algo_name = f"{algo}/lr{lr}/eps{eps}"
    # (b) Encode lr/eps in the seed dir: not possible with make_run_dir
    # (c) Build run_dir manually
    #
    # Option (c) is cleanest: build the full path, then pass it directly.

    run_dir = (
        Path(out_root)
        / "phase2"
        / "ablation"
        / "hparam"
        / task
        / algo
        / lr_str
        / eps_str
        / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    rw = RunWriter(
        run_dir=run_dir,
        phase="phase2",
        suite="ablation/hparam",
        task=task,
        algorithm=algo,
        seed=seed,
        config=run_config,
        storage_mode="rl_online",
    )

    # The actual RL training would be dispatched here. For now, we
    # record the run metadata and return. The actual training loop
    # reuses the same logic as run_phase1_rl.run_single but with
    # modified hyperparameters.
    #
    # In production, this calls into the RL training loop with the
    # overridden lr_multiplier and epsilon. We delegate to the RL
    # runner's run_single, passing the hparam overrides via task_config.

    from experiments.weighted_lse_dp.runners.run_phase2_rl import (
        run_single as rl_run_single,
    )

    # The RL runner reads epsilon from task_config; we override it.
    result = rl_run_single(
        task=task,
        algorithm=algo,
        seed=seed,
        task_config=hparam_task_cfg,
        out_root=out_root,
        suite=f"ablation/hparam/{lr_str}/{eps_str}",
    )

    return {
        "task": task,
        "algorithm": algo,
        "seed": seed,
        "gamma_prime": gp,
        "lr_multiplier": lr,
        "epsilon": eps,
        "ablation_type": "hparam",
        "passed": result.get("passed", False),
        "run_dir": result.get("run_dir"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser(
        prog="run_phase2_ablation",
        description=(
            "Phase II ablation runner: gamma-prime grid and "
            "hyperparameter retuning for stress tasks."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(_DEFAULT_CONFIG),
        help=f"Suite config JSON (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Filter to one task (e.g. 'chain_jackpot'), or 'all' for "
            "all stress tasks."
        ),
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Filter to one algorithm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter to one seed.",
    )
    parser.add_argument(
        "--gamma-prime",
        type=float,
        default=None,
        help="Filter gamma-prime ablation to one value (e.g. 0.95).",
    )
    parser.add_argument(
        "--hparam-only",
        action="store_true",
        default=False,
        help="Run only the hyperparameter retuning grid (skip gamma ablation).",
    )
    parser.add_argument(
        "--gamma-only",
        action="store_true",
        default=False,
        help="Run only the gamma-prime grid (skip hyperparameter retuning).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(_DEFAULT_OUT_ROOT),
        help=f"Output root directory (default: {_DEFAULT_OUT_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print planned runs without executing.",
    )
    args = parser.parse_args(argv)

    if args.hparam_only and args.gamma_only:
        print(
            "ERROR: --hparam-only and --gamma-only are mutually exclusive.",
            file=sys.stderr,
        )
        return 1

    # -- Load config --------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        return 1

    with open(config_path, "r") as f:
        config = json.load(f)

    # -- Build plans --------------------------------------------------------
    gamma_plan: list[dict[str, Any]] = []
    hparam_plan: list[dict[str, Any]] = []

    if not args.hparam_only:
        gamma_plan = _build_gamma_plan(
            config,
            task_filter=args.task,
            algorithm_filter=args.algorithm,
            seed_filter=args.seed,
            gamma_prime_filter=args.gamma_prime,
        )

    if not args.gamma_only:
        hparam_plan = _build_hparam_plan(
            config,
            task_filter=args.task,
            algorithm_filter=args.algorithm,
            seed_filter=args.seed,
        )

    full_plan = gamma_plan + hparam_plan

    if not full_plan:
        print("No ablation runs matched the filters.", file=sys.stderr)
        return 1

    out_root = Path(args.out_root)

    # -- Dry run ------------------------------------------------------------
    if args.dry_run:
        n_gamma = len(gamma_plan)
        n_hparam = len(hparam_plan)
        gamma_primes = sorted({e["gamma_prime"] for e in gamma_plan}) if gamma_plan else []
        print(f"[phase2-ablation] DRY RUN -- {len(full_plan)} run(s) planned")
        print(f"  config:              {config_path}")
        print(f"  out_root:            {out_root}")
        print(f"  gamma' runs:         {n_gamma}")
        print(f"  hparam runs:         {n_hparam}")
        if gamma_primes:
            print(f"  gamma' values:       {gamma_primes}")

        if gamma_plan:
            print(f"\n  --- Gamma-prime ablation ({n_gamma} runs) ---")
            for i, entry in enumerate(gamma_plan, 1):
                family = "DP" if entry["algorithm"] in DP_ALGORITHMS else "RL"
                print(
                    f"  [{i:>4d}] task={entry['task']:<24s} "
                    f"algo={entry['algorithm']:<16s} "
                    f"seed={entry['seed']:<5d} "
                    f"gamma'={entry['gamma_prime']:.2f}  "
                    f"({family})"
                )

        if hparam_plan:
            print(f"\n  --- Hyperparameter retuning ({n_hparam} runs) ---")
            for i, entry in enumerate(hparam_plan, 1):
                print(
                    f"  [{i:>4d}] task={entry['task']:<24s} "
                    f"algo={entry['algorithm']:<16s} "
                    f"seed={entry['seed']:<5d} "
                    f"lr={entry['lr_multiplier']:.1f}  "
                    f"eps={entry['epsilon']:.2f}"
                )

        print(f"\nTotal: {len(full_plan)} run(s). No artifacts written.")
        return 0

    # -- Execute runs -------------------------------------------------------
    print(
        f"[phase2-ablation] Executing {len(full_plan)} run(s) "
        f"(out_root={out_root})"
    )

    t_start = time.perf_counter()
    n_passed = 0
    n_failed = 0
    results: list[dict[str, Any]] = []

    for i, entry in enumerate(full_plan, 1):
        abl_type = entry["ablation_type"]

        if abl_type == "gamma":
            gp_str = _encode_gamma(entry["gamma_prime"])
            label = (
                f"{entry['task']}/{entry['algorithm']}/"
                f"{gp_str}/seed_{entry['seed']}"
            )
        else:
            lr_str = f"lr{entry['lr_multiplier']:.1f}".replace(".", "")
            eps_str = f"eps{entry['epsilon']:.2f}".replace(".", "")
            label = (
                f"{entry['task']}/{entry['algorithm']}/"
                f"{lr_str}/{eps_str}/seed_{entry['seed']}"
            )

        print(
            f"\n[phase2-ablation] === Run {i}/{len(full_plan)}: "
            f"[{abl_type}] {label} ==="
        )

        try:
            if abl_type == "gamma":
                result = _run_gamma_ablation(
                    entry, out_root=out_root, full_config=config,
                )
            else:
                result = _run_hparam_ablation(
                    entry, out_root=out_root, full_config=config,
                )

            results.append(result)
            if result.get("passed", False):
                n_passed += 1
            else:
                n_failed += 1

        except Exception as exc:
            n_failed += 1
            tb = traceback.format_exc()
            print(f"[phase2-ablation] FAILED: {exc!r}", file=sys.stderr)
            print(tb, file=sys.stderr)
            results.append({
                "task": entry["task"],
                "algorithm": entry["algorithm"],
                "seed": entry["seed"],
                "ablation_type": abl_type,
                "passed": False,
                "error": repr(exc),
                "traceback": tb,
            })

    elapsed = time.perf_counter() - t_start
    print(
        f"\n[phase2-ablation] Summary: {n_passed} passed, {n_failed} failed, "
        f"{elapsed:.1f}s total"
    )

    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        abl = r.get("ablation_type", "?")
        if abl == "gamma":
            detail = f"gamma'={r.get('gamma_prime', '?'):.2f}"
        else:
            detail = (
                f"lr={r.get('lr_multiplier', '?')}  "
                f"eps={r.get('epsilon', '?')}"
            )
        print(
            f"  [{abl:>6s}] {r['task']:<24s} {r['algorithm']:<16s} "
            f"seed={r['seed']:<5d} {detail}  {status}"
        )

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
