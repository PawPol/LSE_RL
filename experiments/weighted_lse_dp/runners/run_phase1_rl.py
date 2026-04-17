"""Phase I RL runner: online tabular RL runs (QLearning / ExpectedSARSA).

For each (task, algorithm, seed) in the paper suite, this runner:

1. Loads the time-augmented task via the ``make_*`` factory.
2. Creates the RL agent (QLearning or ExpectedSARSA) with fixed
   epsilon-greedy exploration (eps=0.1) and constant learning rate 0.1.
3. Attaches a :class:`TransitionLogger` callback_step to Core.
4. Runs ``Core.learn(...)`` in checkpoint segments, evaluating with
   :class:`RLEvaluator` at each checkpoint.
5. After training: builds transitions payload and calibration stats,
   flushes everything via :class:`RunWriter`.

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase1_rl.py \\
        [--config PATH]            # default: experiments/weighted_lse_dp/configs/phase1/paper_suite.json
        [--task TASK]              # filter to one task
        [--algorithm ALG]          # QLearning or ExpectedSARSA
        [--seed SEED]              # filter to one seed
        [--out-root PATH]          # default: results/weighted_lse_dp/phase1/paper_suite
        [--dry-run]                # print plan, no execution
        [--gamma-prime GAMMA]      # optional gamma override for ablation mode

Spec anchors: Phase I spec S5.1, S7.1-7.4, S9.3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure repo root and vendored MushroomRL are importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "mushroom-rl-dev"))

import numpy as np  # noqa: E402

from experiments.weighted_lse_dp.common.callbacks import (  # noqa: E402
    RLEvaluator,
    TransitionLogger,
)
from experiments.weighted_lse_dp.common.calibration import (  # noqa: E402
    aggregate_calibration_stats,
)
from experiments.weighted_lse_dp.common.schemas import RunWriter  # noqa: E402
from experiments.weighted_lse_dp.common.seeds import (  # noqa: E402
    get_seeds,
    seed_everything,
)
from experiments.weighted_lse_dp.common.task_factories import (  # noqa: E402
    make_chain_base,
    make_grid_base,
    make_taxi_base,
)

__all__ = ["main", "run_single", "build_plan"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default config path (relative to repo root).
_DEFAULT_CONFIG = (
    "experiments/weighted_lse_dp/configs/phase1/paper_suite.json"
)

#: Default output root (relative to repo root).
#: RunWriter.create appends phase/suite, so this must NOT pre-include them.
_DEFAULT_OUT_ROOT = "results/weighted_lse_dp"

#: Map task name -> factory function.
_TASK_FACTORIES: dict[str, Any] = {
    "chain_base": make_chain_base,
    "grid_base": make_grid_base,
    "taxi_base": make_taxi_base,
}

#: Map task name -> n_base (number of un-augmented base states).
_N_BASE: dict[str, int] = {
    "chain_base": 25,
    "grid_base": 25,
    "taxi_base": 44,
}

#: Supported RL algorithms.
_ALGORITHMS: dict[str, str] = {
    "QLearning": "QLearning",
    "ExpectedSARSA": "ExpectedSARSA",
}

#: Fixed training hyperparameters (spec S5.1).
_EPSILON = 0.1
_LEARNING_RATE = 0.1


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------


def build_plan(
    config: dict[str, Any],
    *,
    task_filter: str | None = None,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Build the list of (task, algorithm, seed) runs from a config dict.

    Parameters
    ----------
    config:
        Loaded paper_suite.json config.
    task_filter:
        When set, include only this task.
    algorithm_filter:
        When set, include only this algorithm.
    seed_filter:
        When set, include only this seed.

    Returns
    -------
    list[dict]
        Each entry has keys: ``task``, ``algorithm``, ``seed``,
        ``task_config`` (the per-task config block from the JSON).
    """
    tasks_cfg = config.get("tasks", {})
    seeds_from_config = tuple(config.get("seeds", [11, 29, 47]))

    plan: list[dict[str, Any]] = []

    for task_name, task_cfg in sorted(tasks_cfg.items()):
        if task_filter is not None and task_name != task_filter:
            continue
        if task_name not in _TASK_FACTORIES:
            continue

        rl_algorithms = task_cfg.get("rl_algorithms", [])

        for algo_name in rl_algorithms:
            if algorithm_filter is not None and algo_name != algorithm_filter:
                continue
            if algo_name not in _ALGORITHMS:
                continue

            seeds = seeds_from_config
            if seed_filter is not None:
                if seed_filter in seeds:
                    seeds = (seed_filter,)
                else:
                    # Allow running with a seed not in the config
                    # (useful for ad-hoc testing).
                    seeds = (seed_filter,)

            for seed in seeds:
                plan.append({
                    "task": task_name,
                    "algorithm": algo_name,
                    "seed": seed,
                    "task_config": task_cfg,
                })

    return plan


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------


def _make_agent(
    algo_name: str,
    mdp_info: Any,
    *,
    epsilon: float = _EPSILON,
    learning_rate: float = _LEARNING_RATE,
) -> Any:
    """Construct a MushroomRL tabular RL agent.

    Deferred import so the module can be imported without MushroomRL on
    sys.path (e.g. for ``--dry-run``).
    """
    from mushroom_rl.algorithms.value.td import ExpectedSARSA, QLearning
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import Parameter

    pi = EpsGreedy(epsilon=Parameter(value=epsilon))

    if algo_name == "QLearning":
        return QLearning(
            mdp_info, pi, learning_rate=Parameter(value=learning_rate)
        )
    elif algo_name == "ExpectedSARSA":
        return ExpectedSARSA(
            mdp_info, pi, learning_rate=Parameter(value=learning_rate)
        )
    else:
        raise ValueError(f"Unsupported RL algorithm: {algo_name!r}")


def run_single(
    task: str,
    algorithm: str,
    seed: int,
    task_config: dict[str, Any],
    *,
    out_root: Path,
    suite: str = "paper_suite",
    gamma_prime: float | None = None,
) -> dict[str, Any]:
    """Execute one (task, algorithm, seed) run.

    Parameters
    ----------
    task:
        Task identifier (``chain_base``, ``grid_base``, ``taxi_base``).
    algorithm:
        Algorithm identifier (``QLearning`` or ``ExpectedSARSA``).
    seed:
        Integer seed for reproducibility.
    task_config:
        Per-task config block from the suite JSON.
    out_root:
        Base output directory. RunWriter appends ``/<phase>/<suite>/…``.
    suite:
        Suite name for RunWriter path construction (default ``"paper_suite"``).
        Pass e.g. ``"ablation/gamma090"`` for gamma-ablation runs.
    gamma_prime:
        Optional gamma override for ablation mode. When ``None``,
        uses the task's native gamma.

    Returns
    -------
    dict
        Summary dict with keys: ``task``, ``algorithm``, ``seed``,
        ``passed``, ``wall_s``, ``run_dir``, ``summary``, and
        optionally ``error`` / ``traceback``.
    """
    from mushroom_rl.core import Core  # noqa: E402

    t_start = time.perf_counter()

    # -- Resolve parameters from task config --------------------------------
    train_steps = int(task_config["train_steps"])
    checkpoint_every = int(task_config["checkpoint_every"])
    eval_episodes_checkpoint = int(task_config.get("eval_episodes_checkpoint", 50))
    success_threshold = float(task_config["success_threshold"])
    gamma = float(task_config["gamma"])
    horizon = int(task_config["horizon"])
    n_base = _N_BASE[task]

    if gamma_prime is not None:
        gamma = float(gamma_prime)

    # -- Build resolved config for run.json ---------------------------------
    resolved_config: dict[str, Any] = {
        "task": task,
        "algorithm": algorithm,
        "seed": seed,
        "gamma": gamma,
        "gamma_prime": gamma_prime,
        "horizon": horizon,
        "n_base": n_base,
        "train_steps": train_steps,
        "checkpoint_every": checkpoint_every,
        "eval_episodes_checkpoint": eval_episodes_checkpoint,
        "success_threshold": success_threshold,
        "epsilon": _EPSILON,
        "learning_rate": _LEARNING_RATE,
        "task_config": task_config,
    }
    # Store gamma_prime_override so aggregate_phase1.discover_runs() can
    # group ablation runs by their effective discount (mirrors DP runner).
    if gamma_prime is not None:
        resolved_config["gamma_prime_override"] = gamma_prime

    # -- Seed everything ----------------------------------------------------
    seed_everything(seed)

    # -- Create environment -------------------------------------------------
    factory = _TASK_FACTORIES[task]
    _mdp_base, mdp_rl, cfg, _ref_pi = factory(seed=seed)

    # Propagate gamma (possibly overridden by gamma_prime) into the RL env
    # so that agent training, the Q-update TD target, and metadata all use
    # the same effective discount.
    mdp_rl.info.gamma = gamma

    # -- Create agent -------------------------------------------------------
    agent = _make_agent(algorithm, mdp_rl.info)

    # -- Create RunWriter ---------------------------------------------------
    rw = RunWriter.create(
        base=out_root,
        phase="phase1",
        suite=suite,
        task=task,
        algorithm=algorithm,
        seed=seed,
        config=resolved_config,
        storage_mode="rl_online",
    )

    # -- Create callbacks ---------------------------------------------------
    logger = TransitionLogger(agent, n_base=n_base, gamma=gamma)
    evaluator = RLEvaluator(
        agent=agent,
        env=mdp_rl,
        run_writer=rw,
        n_eval_episodes=eval_episodes_checkpoint,
        success_threshold=success_threshold,
        gamma=gamma,
    )

    # -- Create Core --------------------------------------------------------
    core = Core(agent, mdp_rl, callback_step=logger)

    # -- Training loop with checkpoints ------------------------------------
    print(
        f"[phase1_rl] {task}/{algorithm}/seed_{seed}: "
        f"train_steps={train_steps}, checkpoint_every={checkpoint_every}"
    )

    with rw.timer.phase("train"):
        steps_done = 0
        for ckpt_start in range(0, train_steps, checkpoint_every):
            n_steps_this_segment = min(checkpoint_every, train_steps - steps_done)
            core.learn(
                n_steps=n_steps_this_segment,
                n_steps_per_fit=1,
                quiet=True,
            )
            steps_done += n_steps_this_segment

            with rw.timer.phase("eval"):
                eval_result = evaluator.evaluate(steps=steps_done)

            print(
                f"  checkpoint {steps_done:>8d}/{train_steps}: "
                f"disc_return={eval_result['disc_return_mean']:.4f}, "
                f"success_rate={eval_result['success_rate']:.4f}"
            )

    # -- Post-training: build transitions payload ---------------------------
    print(
        f"[phase1_rl] {task}/{algorithm}/seed_{seed}: "
        f"transitions logged: {logger.n_transitions}"
    )

    with rw.timer.phase("post"):
        transitions_payload = logger.build_payload()
        rw.set_transitions(transitions_payload)

        # Build calibration stats from the transitions.
        calibration_stats = aggregate_calibration_stats(
            transitions_payload, horizon=horizon
        )
        rw.set_calibration_stats(calibration_stats)

    # -- Compute summary metrics --------------------------------------------
    eval_summary = evaluator.summary()

    metrics: dict[str, Any] = {
        "train_steps": train_steps,
        "n_transitions": logger.n_transitions,
        **{k: v for k, v in eval_summary.items()},
    }

    # -- Flush everything to disk -------------------------------------------
    rw.flush(
        metrics=metrics,
        step_count=train_steps,
        update_count=train_steps,  # n_steps_per_fit=1 => one update per step
    )

    wall_s = time.perf_counter() - t_start
    print(
        f"[phase1_rl] {task}/{algorithm}/seed_{seed}: "
        f"DONE in {wall_s:.1f}s -> {rw.run_dir}"
    )

    return {
        "task": task,
        "algorithm": algorithm,
        "seed": seed,
        "passed": True,
        "wall_s": wall_s,
        "run_dir": str(rw.run_dir),
        "summary": (
            f"steps_to_thr={eval_summary.get('steps_to_threshold')}, "
            f"final_sr={eval_summary.get('final_10pct_success_rate', 0):.3f}"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on full success, 1 if any run fails.

    Parameters
    ----------
    argv:
        Optional argv list for testing. When ``None``, ``sys.argv[1:]``
        is used.
    """
    parser = argparse.ArgumentParser(
        prog="run_phase1_rl",
        description=(
            "Phase I RL runner: online tabular RL runs for the "
            "weighted-LSE DP paper suite."
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
        choices=list(_TASK_FACTORIES.keys()),
        help="Filter to one task.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=list(_ALGORITHMS.keys()),
        help="Filter to one algorithm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter to one seed.",
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
    parser.add_argument(
        "--gamma-prime",
        type=float,
        default=None,
        help="Override gamma for ablation mode.",
    )
    args = parser.parse_args(argv)

    # -- Load config --------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        return 1

    with open(config_path, "r") as f:
        config = json.load(f)

    # -- Build plan ---------------------------------------------------------
    plan = build_plan(
        config,
        task_filter=args.task,
        algorithm_filter=args.algorithm,
        seed_filter=args.seed,
    )

    if not plan:
        print("No runs matched the filters.", file=sys.stderr)
        return 1

    out_root = Path(args.out_root)

    # -- Dry run: print plan and exit --------------------------------------
    if args.dry_run:
        print(f"[phase1_rl] DRY RUN -- {len(plan)} run(s) planned:")
        print(f"  config:    {config_path}")
        print(f"  out_root:  {out_root}")
        if args.gamma_prime is not None:
            print(f"  gamma':    {args.gamma_prime}")
        print()
        for i, entry in enumerate(plan, 1):
            tc = entry["task_config"]
            print(
                f"  [{i:>3d}] task={entry['task']:<12s} "
                f"algorithm={entry['algorithm']:<16s} "
                f"seed={entry['seed']:<5d} "
                f"train_steps={tc['train_steps']:>8d} "
                f"checkpoint_every={tc['checkpoint_every']:>6d}"
            )
        return 0

    # -- Execute runs -------------------------------------------------------
    print(
        f"[phase1_rl] Executing {len(plan)} run(s) "
        f"(out_root={out_root})"
    )

    results: list[dict[str, Any]] = []
    n_passed = 0

    for i, entry in enumerate(plan, 1):
        print(
            f"\n[phase1_rl] === Run {i}/{len(plan)}: "
            f"{entry['task']}/{entry['algorithm']}/seed_{entry['seed']} ==="
        )
        try:
            result = run_single(
                task=entry["task"],
                algorithm=entry["algorithm"],
                seed=entry["seed"],
                task_config=entry["task_config"],
                out_root=out_root,
                gamma_prime=args.gamma_prime,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[phase1_rl] FAILED: {exc!r}")
            print(tb)
            result = {
                "task": entry["task"],
                "algorithm": entry["algorithm"],
                "seed": entry["seed"],
                "passed": False,
                "wall_s": 0.0,
                "run_dir": None,
                "summary": "EXCEPTION",
                "error": repr(exc),
                "traceback": tb,
            }
        results.append(result)
        if result["passed"]:
            n_passed += 1

    # -- Summary ------------------------------------------------------------
    print(f"\n[phase1_rl] Summary: {n_passed}/{len(results)} runs passed")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['task']:<12s} {r['algorithm']:<16s} seed={r['seed']:<5d} "
            f"{status}  {r.get('summary', '')}"
        )

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
