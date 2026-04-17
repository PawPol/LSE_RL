"""Phase I smoke-run entry point: tiny configs to verify the pipeline end-to-end.

Spec anchor: Phase I spec §4.2 / tasks/todo.md item 18.

This module is the **permanent** smoke harness for the weighted-LSE DP
empirical program. It wraps the six Phase I reproductions
(``simple_chain``, ``grid_world``, ``double_chain``, ``taxi``,
``mountain_car``, ``puddle_world``) behind a single CLI and a uniform
logging contract:

    results/weighted_lse_dp/phase1/smoke/<task>/<algorithm>/seed_<seed>/
        stdout.log
        run.json

The ``run.json`` header is emitted by
:func:`experiments.weighted_lse_dp.common.manifests.write_run_json` and
therefore carries the canonical ``schema_version``, ``phase``, ``task``,
``algorithm``, ``seed``, ``git_sha``, ``host``, ``created_at`` fields
required by the aggregators. Each example writes its own ``run.json``;
this runner additionally writes
``results/weighted_lse_dp/phase1/smoke/smoke_summary.json`` with the
full per-example result list.

Design notes
------------
- Each of the six examples is a **self-contained function** (``run_*``)
  that owns its deferred ``mushroom_rl`` imports, its own reduced-budget
  config, and its non-degeneracy contract. They intentionally do NOT
  share state with the one-off wrappers under
  ``results/weighted_lse_dp/phase1/smoke/<name>/run_smoke.py``; those
  stay on disk as historical reproductions. Future edits should happen
  here, not there.
- Per-run wall-clock is tracked via
  :class:`experiments.weighted_lse_dp.common.timing.RunTimer` with the
  canonical ``step`` / ``fit`` / ``eval`` / ``other`` phases where
  applicable, and the timing dict is merged into the ``run.json``
  ``extra`` payload under the ``"timings"`` key.
- Exit code is 0 iff every requested example passed its non-degeneracy
  contract; 1 otherwise.

CLI usage::

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase1_smoke.py \\
        [--seed SEED] [--out-root PATH] [--only NAME]

``--only`` filters to a single example (valid values: ``simple_chain``,
``grid_world``, ``double_chain``, ``taxi``, ``mountain_car``,
``puddle_world``); omitted runs all six sequentially in the order
above.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

# Repo root and vendored MushroomRL must be importable regardless of cwd.
sys.path.insert(0, "/Users/liq/Documents/Claude/Projects/LSE_RL")
sys.path.insert(0, "/Users/liq/Documents/Claude/Projects/LSE_RL/mushroom-rl-dev")

import numpy as np  # noqa: E402

from experiments.weighted_lse_dp.common.io import (  # noqa: E402
    RESULT_ROOT,
    make_run_dir,
    save_json,
    stdout_to_log,
)
from experiments.weighted_lse_dp.common.manifests import (  # noqa: E402
    write_run_json,
)
from experiments.weighted_lse_dp.common.seeds import seed_everything  # noqa: E402
from experiments.weighted_lse_dp.common.timing import RunTimer  # noqa: E402

__all__ = [
    "EXAMPLE_NAMES",
    "run_simple_chain",
    "run_grid_world",
    "run_double_chain",
    "run_taxi",
    "run_mountain_car",
    "run_puddle_world",
    "main",
]


#: Canonical ordering of the six Phase I smoke reproductions. Used for
#: the ``--only`` validator, for the summary table, and as the iteration
#: order when ``--only`` is omitted.
EXAMPLE_NAMES: tuple[str, ...] = (
    "simple_chain",
    "grid_world",
    "double_chain",
    "taxi",
    "mountain_car",
    "puddle_world",
)


# ============================================================================
# Helpers
# ============================================================================


def _q_table_stats_plain(agent) -> dict:
    """Summarise a plain ``Table`` Q-table for non-degeneracy checks.

    Handles ``EnsembleTable`` (e.g. ``DoubleQLearning``) by averaging
    member tables so the stats remain shape-stable.
    """
    q = agent.Q
    if hasattr(q, "table"):
        # shape: (S, A)
        arr = np.asarray(q.table, dtype=np.float64)
    else:
        # EnsembleTable -> average members -> shape: (S, A)
        members = [np.asarray(m.table, dtype=np.float64) for m in q.model]
        arr = np.stack(members, axis=0).mean(axis=0)

    return {
        "shape": list(arr.shape),
        "nnz": int(np.count_nonzero(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "has_nan": bool(np.any(np.isnan(arr))),
        "has_inf": bool(np.any(np.isinf(arr))),
    }


def _weights_stats(agent) -> dict:
    """Summarise a linear-approximator weight vector for non-degeneracy."""
    w = np.asarray(agent.Q.get_weights(), dtype=np.float64)
    return {
        "shape": list(w.shape),
        "size": int(w.size),
        "nnz": int(np.count_nonzero(w)),
        "min": float(w.min()),
        "max": float(w.max()),
        "mean": float(w.mean()),
        "l2_norm": float(np.linalg.norm(w)),
        "has_nan": bool(np.any(np.isnan(w))),
        "has_inf": bool(np.any(np.isinf(w))),
    }


def _derive_rep_seeds(seed: int, n_experiment: int) -> list[int]:
    """Derive deterministic per-repetition seeds from the config seed."""
    return [seed * 100 + i for i in range(n_experiment)]


# ============================================================================
# Example 1: simple_chain (QLearning)
# ============================================================================


def run_simple_chain(seed: int, out_root: Path) -> dict:
    """Run the ``simple_chain_qlearning.py`` smoke reproduction.

    Single QLearning agent on ``generate_simple_chain`` with
    ``n_steps=2000`` train and ``n_steps=500`` eval (pre- and post-
    training). Non-degeneracy = Q-table has non-zero entries, no
    NaN/Inf, and final evaluation returns are non-zero.
    """
    task = "simple_chain"
    algorithm = "QLearning"
    run_dir = make_run_dir(
        base=out_root,
        phase="phase1",
        suite="smoke",
        task=task,
        algorithm=algorithm,
        seed=seed,
        exist_ok=True,
    )
    stdout_path = run_dir / "stdout.log"

    # ----- config (explicit, no hidden defaults) -----
    n_steps_train = 2000
    n_steps_per_fit = 1
    n_steps_eval = 500

    chain_state_n = 5
    chain_goal = [2]
    chain_prob = 0.8
    chain_rew = 1.0
    chain_gamma = 0.9
    epsilon_val = 0.15
    learning_rate_val = 0.2

    rt = RunTimer()
    non_degenerate = False
    J0 = float("nan")
    J1 = float("nan")
    q_stats: dict = {}

    t_total_0 = time.perf_counter()
    with stdout_to_log(stdout_path):
        print(f"[smoke:{task}] seed={seed}")
        seed_everything(seed)

        # Deferred imports so the tee captures mushroom_rl init warnings.
        from mushroom_rl.algorithms.value import QLearning
        from mushroom_rl.core import Core, Logger
        from mushroom_rl.environments import generate_simple_chain
        from mushroom_rl.policy import EpsGreedy
        from mushroom_rl.rl_utils.parameters import Parameter

        logger = Logger(QLearning.__name__, results_dir=None)
        logger.strong_line()
        logger.info("Smoke: simple_chain_qlearning")

        mdp = generate_simple_chain(
            state_n=chain_state_n,
            goal_states=chain_goal,
            prob=chain_prob,
            rew=chain_rew,
            gamma=chain_gamma,
        )
        pi = EpsGreedy(epsilon=Parameter(value=epsilon_val))
        agent = QLearning(
            mdp.info, pi, learning_rate=Parameter(value=learning_rate_val)
        )
        core = Core(agent, mdp)

        with rt.phase("eval"):
            dataset0 = core.evaluate(n_steps=n_steps_eval)
        J0_arr = np.asarray(dataset0.discounted_return, dtype=np.float64)
        J0 = float(np.mean(J0_arr))
        logger.info(f"J start: {J0}")

        with rt.phase("fit"):
            core.learn(
                n_steps=n_steps_train, n_steps_per_fit=n_steps_per_fit
            )

        with rt.phase("eval"):
            dataset1 = core.evaluate(n_steps=n_steps_eval)
        J1_arr = np.asarray(dataset1.discounted_return, dtype=np.float64)
        J1 = float(np.mean(J1_arr))
        logger.info(f"J final: {J1}")

        q_table = np.asarray(agent.Q.table, dtype=np.float64)
        q_stats = {
            "shape": list(q_table.shape),
            "min": float(q_table.min()),
            "max": float(q_table.max()),
            "nnz": int(np.count_nonzero(q_table)),
            "has_nan": bool(np.any(np.isnan(q_table))),
            "has_inf": bool(np.any(np.isinf(q_table))),
        }
        has_nonzero = q_stats["nnz"] > 0
        eval_nondegenerate = bool(np.any(J1_arr != 0.0))
        non_degenerate = (
            has_nonzero
            and (not q_stats["has_nan"])
            and (not q_stats["has_inf"])
            and eval_nondegenerate
        )
        print(
            f"[smoke:{task}] non_degeneracy: nnz={q_stats['nnz']}, "
            f"has_nan={q_stats['has_nan']}, has_inf={q_stats['has_inf']}, "
            f"eval_nondegenerate={eval_nondegenerate}, "
            f"J0={J0:.6f}, J1={J1:.6f}"
        )
        if not non_degenerate:
            print(f"[smoke:{task}] WARNING: non-degeneracy check FAILED")

    wall_s = time.perf_counter() - t_total_0
    timings = rt.to_dict()

    run_json_path = write_run_json(
        run_dir,
        config={
            "source": "mushroom-rl-dev/examples/simple_chain_qlearning.py",
            "seed": seed,
            "n_steps": n_steps_train,
            "n_steps_per_fit": n_steps_per_fit,
            "n_steps_eval": n_steps_eval,
            "mdp": {
                "state_n": chain_state_n,
                "goal_states": chain_goal,
                "prob": chain_prob,
                "rew": chain_rew,
                "gamma": chain_gamma,
            },
            "policy": {"kind": "EpsGreedy", "epsilon": epsilon_val},
            "agent": {
                "algorithm": "QLearning",
                "learning_rate": learning_rate_val,
            },
        },
        phase="phase1",
        task=task,
        algorithm=algorithm,
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": non_degenerate,
            "J_start": J0,
            "J_final": J1,
            "q_table_shape": q_stats.get("shape"),
            "q_table_min": q_stats.get("min"),
            "q_table_max": q_stats.get("max"),
            "q_table_nnz": q_stats.get("nnz"),
            "timings": timings,
            "total_wall_s": float(wall_s),
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke:{task}] wrote {run_json_path}")

    return {
        "name": task,
        "seed": seed,
        "passed": bool(non_degenerate),
        "J_final": float(J1) if np.isfinite(J1) else None,
        "wall_s": float(wall_s),
        "run_dir": str(run_dir),
        "summary": f"J_final={J1:.2f}" if np.isfinite(J1) else "J_final=NaN",
    }


# ============================================================================
# Example 2: grid_world (TD sweep: 5 algos)
# ============================================================================


def _grid_world_run_one(
    algorithm_class,
    exp: float,
    n_steps: int,
    rep_seed: int,
):
    """One repetition of the grid_world_td experiment body."""
    from mushroom_rl.core import Core
    from mushroom_rl.environments import GridWorldVanHasselt
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import DecayParameter
    from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ

    np.random.seed(rep_seed)
    mdp = GridWorldVanHasselt()

    epsilon = DecayParameter(
        value=1.0, exp=0.5, size=mdp.info.observation_space.size
    )
    pi = EpsGreedy(epsilon=epsilon)
    learning_rate = DecayParameter(value=1.0, exp=exp, size=mdp.info.size)
    agent = algorithm_class(mdp.info, pi, learning_rate=learning_rate)

    start = mdp.convert_to_int(mdp._start, mdp._width)
    collect_max_Q = CollectMaxQ(agent.Q, start)
    collect_dataset = CollectDataset()
    core = Core(agent, mdp, [collect_dataset, collect_max_Q])
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    ds = collect_dataset.get()
    rewards = np.asarray(
        getattr(ds, "reward", getattr(ds, "rewards", None)),
        dtype=np.float64,
    )
    max_Qs = np.asarray(collect_max_Q.get(), dtype=np.float64)
    return rewards, max_Qs, agent


def run_grid_world(seed: int, out_root: Path) -> dict:
    """Run the ``grid_world_td.py`` smoke reproduction (5-algorithm sweep).

    Sweeps QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning,
    and SARSA on ``GridWorldVanHasselt`` with ``n_steps=2000`` and
    ``n_experiment=2`` per algorithm. Algorithm tag ``TD_sweep`` is used
    since a single algorithm label cannot describe the sweep.
    """
    task = "grid_world"
    algorithm = "TD_sweep"
    run_dir = make_run_dir(
        base=out_root,
        phase="phase1",
        suite="smoke",
        task=task,
        algorithm=algorithm,
        seed=seed,
        exist_ok=True,
    )
    stdout_path = run_dir / "stdout.log"

    n_steps = 2000
    n_experiment = 2
    exp_value = 0.8

    grid_cfg = {
        "height": 3,
        "width": 3,
        "goal": [0, 2],
        "start": [2, 0],
        "gamma": 0.95,
        "horizon": "inf",
    }

    rt = RunTimer()
    algo_results: dict[str, dict] = {}
    overall_non_degenerate = True
    n_algos = 0

    t_total_0 = time.perf_counter()
    with stdout_to_log(stdout_path):
        print(f"[smoke:{task}] seed={seed}")
        seed_everything(seed)

        from mushroom_rl.algorithms.value import (
            DoubleQLearning,
            QLearning,
            SARSA,
            SpeedyQLearning,
            WeightedQLearning,
        )
        from mushroom_rl.core import Logger

        names = {
            QLearning: "QLearning",
            DoubleQLearning: "DoubleQLearning",
            WeightedQLearning: "WeightedQLearning",
            SpeedyQLearning: "SpeedyQLearning",
            SARSA: "SARSA",
        }
        algo_classes = [
            QLearning,
            DoubleQLearning,
            WeightedQLearning,
            SpeedyQLearning,
            SARSA,
        ]
        n_algos = len(algo_classes)

        logger = Logger("grid_world_td_sweep", results_dir=None)
        logger.strong_line()
        logger.info(
            "Smoke: grid_world_td "
            f"(exp={exp_value}, n_experiment={n_experiment}, n_steps={n_steps})"
        )

        rep_seeds = _derive_rep_seeds(seed, n_experiment)

        for algo in algo_classes:
            algo_name = names[algo]
            logger.info(f"Alg: {algo_name}")

            rewards_list, max_Qs_list, last_agent = [], [], None
            with rt.phase("fit"):
                for rs in rep_seeds:
                    rewards, max_Qs, agent = _grid_world_run_one(
                        algo, exp_value, n_steps, rs
                    )
                    rewards_list.append(rewards)
                    max_Qs_list.append(max_Qs)
                    last_agent = agent

            # shape: (n_experiment, n_steps)
            rewards_arr = np.stack(rewards_list, axis=0)
            max_Qs_arr = np.stack(max_Qs_list, axis=0)
            mean_reward = rewards_arr.mean(axis=0)
            mean_max_Q = max_Qs_arr.mean(axis=0)

            q_stats = _q_table_stats_plain(last_agent)
            reward_nonzero = bool(np.any(mean_reward != 0.0))
            q_nonzero = q_stats["nnz"] > 0
            no_nan_inf = (not q_stats["has_nan"]) and (not q_stats["has_inf"])
            algo_non_degenerate = reward_nonzero and q_nonzero and no_nan_inf
            overall_non_degenerate = (
                overall_non_degenerate and algo_non_degenerate
            )

            print(
                f"[smoke:{task}] {algo_name}: "
                f"reward_sum(mean-over-reps)={float(mean_reward.sum()):.4f}, "
                f"max_Q@start_final={float(mean_max_Q[-1]):.4f}, "
                f"Q.shape={q_stats['shape']}, nnz={q_stats['nnz']}, "
                f"non_degenerate={algo_non_degenerate}"
            )

            algo_results[algo_name] = {
                "reward_sum_mean": float(mean_reward.sum()),
                "max_Q_at_start_final": float(mean_max_Q[-1]),
                "max_Q_at_start_initial": float(mean_max_Q[0]),
                "q_table_shape": q_stats["shape"],
                "q_table_nnz": q_stats["nnz"],
                "q_table_min": q_stats["min"],
                "q_table_max": q_stats["max"],
                "q_table_mean": q_stats["mean"],
                "q_has_nan": q_stats["has_nan"],
                "q_has_inf": q_stats["has_inf"],
                "reward_nonzero": reward_nonzero,
                "non_degenerate": algo_non_degenerate,
            }

        if not overall_non_degenerate:
            print(f"[smoke:{task}] WARNING: non-degeneracy check FAILED")

    wall_s = time.perf_counter() - t_total_0
    timings = rt.to_dict()

    run_json_path = write_run_json(
        run_dir,
        config={
            "source": "mushroom-rl-dev/examples/grid_world_td.py",
            "seed": seed,
            "n_steps": n_steps,
            "n_steps_per_fit": 1,
            "n_experiment": n_experiment,
            "exp": exp_value,
            "mdp": grid_cfg,
            "policy": {
                "kind": "EpsGreedy",
                "epsilon_schedule": "DecayParameter(value=1.0, exp=0.5, size=obs_space.size)",
            },
            "agent": {
                "algorithms": [
                    "QLearning",
                    "DoubleQLearning",
                    "WeightedQLearning",
                    "SpeedyQLearning",
                    "SARSA",
                ],
                "learning_rate_schedule": "DecayParameter(value=1.0, exp=exp, size=mdp.info.size)",
            },
        },
        phase="phase1",
        task=task,
        algorithm=algorithm,
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": overall_non_degenerate,
            "algorithms_sweep": sorted(algo_results.keys()),
            "per_algorithm": algo_results,
            "timings": timings,
            "total_wall_s": float(wall_s),
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke:{task}] wrote {run_json_path}")

    return {
        "name": task,
        "seed": seed,
        "passed": bool(overall_non_degenerate),
        "J_final": None,  # sweep: no single J_final
        "wall_s": float(wall_s),
        "run_dir": str(run_dir),
        "summary": f"({n_algos} algos)",
    }


# ============================================================================
# Example 3: double_chain (4-algorithm Q-learning sweep)
# ============================================================================


_DOUBLE_CHAIN_ASSET_DIR = Path(
    "/Users/liq/Documents/Claude/Projects/LSE_RL"
    "/mushroom-rl-dev/examples/double_chain_q_learning/chain_structure"
)


def _double_chain_run_one(
    algorithm_class,
    exp: float,
    n_steps: int,
    rep_seed: int,
):
    """One repetition of the double_chain experiment body."""
    from mushroom_rl.core import Core
    from mushroom_rl.environments import FiniteMDP
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import DecayParameter, Parameter
    from mushroom_rl.utils.callbacks import CollectQ

    np.random.seed(rep_seed)
    p = np.load(_DOUBLE_CHAIN_ASSET_DIR / "p.npy")  # shape: (S, A, S)
    rew = np.load(_DOUBLE_CHAIN_ASSET_DIR / "rew.npy")  # shape: (S, A, S)
    mdp = FiniteMDP(p, rew, gamma=0.9)

    epsilon = Parameter(value=1.0)
    pi = EpsGreedy(epsilon=epsilon)
    learning_rate = DecayParameter(value=1.0, exp=exp, size=mdp.info.size)
    agent = algorithm_class(mdp.info, pi, learning_rate=learning_rate)

    collect_Q = CollectQ(agent.Q)
    core = Core(agent, mdp, [collect_Q])
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    # shape: (n_steps, S, A)
    Qs = np.asarray(collect_Q.get(), dtype=np.float64)
    return Qs, agent


def run_double_chain(seed: int, out_root: Path) -> dict:
    """Run the ``double_chain.py`` smoke reproduction (4-algorithm sweep).

    Sweeps QLearning, DoubleQLearning, WeightedQLearning,
    SpeedyQLearning on the Peters double-chain asset with
    ``n_steps=2000``, ``n_experiment=2``, and ``exp=1.0``.
    """
    task = "double_chain"
    algorithm = "TD_sweep"
    run_dir = make_run_dir(
        base=out_root,
        phase="phase1",
        suite="smoke",
        task=task,
        algorithm=algorithm,
        seed=seed,
        exist_ok=True,
    )
    stdout_path = run_dir / "stdout.log"

    n_steps = 2000
    n_experiment = 2
    exp_value = 1.0

    mdp_cfg = {
        "kind": "FiniteMDP",
        "asset": str(_DOUBLE_CHAIN_ASSET_DIR),
        "n_states": 9,
        "n_actions": 2,
        "gamma": 0.9,
        "horizon": "inf",
    }

    rt = RunTimer()
    algo_results: dict[str, dict] = {}
    overall_non_degenerate = True
    n_algos = 0

    t_total_0 = time.perf_counter()
    with stdout_to_log(stdout_path):
        print(f"[smoke:{task}] seed={seed}")
        seed_everything(seed)

        from mushroom_rl.algorithms.value import (
            DoubleQLearning,
            QLearning,
            SpeedyQLearning,
            WeightedQLearning,
        )
        from mushroom_rl.core import Logger

        names = {
            QLearning: "QLearning",
            DoubleQLearning: "DoubleQLearning",
            WeightedQLearning: "WeightedQLearning",
            SpeedyQLearning: "SpeedyQLearning",
        }
        algo_classes = [
            QLearning,
            DoubleQLearning,
            WeightedQLearning,
            SpeedyQLearning,
        ]
        n_algos = len(algo_classes)

        logger = Logger("double_chain_sweep", results_dir=None)
        logger.strong_line()
        logger.info(
            "Smoke: double_chain "
            f"(exp={exp_value}, n_experiment={n_experiment}, n_steps={n_steps})"
        )

        rep_seeds = _derive_rep_seeds(seed, n_experiment)

        for algo in algo_classes:
            algo_name = names[algo]
            logger.info(f"Alg: {algo_name}")

            Qs_list, last_agent = [], None
            with rt.phase("fit"):
                for rs in rep_seeds:
                    Qs, agent = _double_chain_run_one(
                        algo, exp_value, n_steps, rs
                    )
                    Qs_list.append(Qs)
                    last_agent = agent

            # shape: (n_experiment, n_steps, S, A)
            Qs_arr = np.stack(Qs_list, axis=0)
            mean_Qs = Qs_arr.mean(axis=0)
            q_s0a0_trace = mean_Qs[:, 0, 0]

            q_stats = _q_table_stats_plain(last_agent)
            traj_nonzero = bool(np.any(mean_Qs != 0.0))
            q_nonzero = q_stats["nnz"] > 0
            no_nan_inf = (not q_stats["has_nan"]) and (not q_stats["has_inf"])
            algo_non_degenerate = traj_nonzero and q_nonzero and no_nan_inf
            overall_non_degenerate = (
                overall_non_degenerate and algo_non_degenerate
            )

            print(
                f"[smoke:{task}] {algo_name}: "
                f"Q(0,0) {float(q_s0a0_trace[0]):.4f} -> "
                f"{float(q_s0a0_trace[-1]):.4f}, "
                f"traj max|Q|={float(np.max(np.abs(mean_Qs))):.4f}, "
                f"non_degenerate={algo_non_degenerate}"
            )

            algo_results[algo_name] = {
                "q_s0a0_initial": float(q_s0a0_trace[0]),
                "q_s0a0_final": float(q_s0a0_trace[-1]),
                "q_trajectory_abs_max": float(np.max(np.abs(mean_Qs))),
                "q_table_shape": q_stats["shape"],
                "q_table_nnz": q_stats["nnz"],
                "q_table_min": q_stats["min"],
                "q_table_max": q_stats["max"],
                "q_table_mean": q_stats["mean"],
                "q_has_nan": q_stats["has_nan"],
                "q_has_inf": q_stats["has_inf"],
                "trajectory_nonzero": traj_nonzero,
                "non_degenerate": algo_non_degenerate,
            }

        if not overall_non_degenerate:
            print(f"[smoke:{task}] WARNING: non-degeneracy check FAILED")

    wall_s = time.perf_counter() - t_total_0
    timings = rt.to_dict()

    run_json_path = write_run_json(
        run_dir,
        config={
            "source": "mushroom-rl-dev/examples/double_chain_q_learning/double_chain.py",
            "seed": seed,
            "n_steps": n_steps,
            "n_steps_per_fit": 1,
            "n_experiment": n_experiment,
            "exp": exp_value,
            "mdp": mdp_cfg,
            "policy": {
                "kind": "EpsGreedy",
                "epsilon_schedule": "Parameter(value=1.0)",
            },
            "agent": {
                "algorithms": [
                    "QLearning",
                    "DoubleQLearning",
                    "WeightedQLearning",
                    "SpeedyQLearning",
                ],
                "learning_rate_schedule": (
                    "DecayParameter(value=1.0, exp=exp, size=mdp.info.size)"
                ),
            },
        },
        phase="phase1",
        task=task,
        algorithm=algorithm,
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": overall_non_degenerate,
            "algorithms_sweep": sorted(algo_results.keys()),
            "per_algorithm": algo_results,
            "timings": timings,
            "total_wall_s": float(wall_s),
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke:{task}] wrote {run_json_path}")

    return {
        "name": task,
        "seed": seed,
        "passed": bool(overall_non_degenerate),
        "J_final": None,
        "wall_s": float(wall_s),
        "run_dir": str(run_dir),
        "summary": f"({n_algos} algos)",
    }


# ============================================================================
# Example 4: taxi (3-policy SARSA sweep)
# ============================================================================


_TAXI_GRID = Path(
    "/Users/liq/Documents/Claude/Projects/LSE_RL"
    "/mushroom-rl-dev/examples/taxi_mellow_sarsa/grid.txt"
)


def _taxi_run_one(policy_cls, param_value: float, n_steps: int, rep_seed: int):
    """One repetition of the taxi_mellow_sarsa experiment body."""
    from mushroom_rl.algorithms.value import SARSA
    from mushroom_rl.core import Core
    from mushroom_rl.environments.generators.taxi import generate_taxi
    from mushroom_rl.rl_utils.parameters import Parameter
    from mushroom_rl.utils.callbacks import CollectDataset

    np.random.seed(rep_seed)
    mdp = generate_taxi(str(_TAXI_GRID))
    pi = policy_cls(Parameter(value=param_value))
    learning_rate = Parameter(value=0.15)
    agent = SARSA(mdp.info, pi, learning_rate=learning_rate)

    collect_dataset = CollectDataset()
    core = Core(agent, mdp, [collect_dataset])
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    ds = collect_dataset.get()
    rewards = np.asarray(
        getattr(ds, "reward", getattr(ds, "rewards", None)),
        dtype=np.float64,
    )
    mean_reward = float(rewards.sum()) / float(n_steps)
    return mean_reward, agent, int(rewards.size)


def run_taxi(seed: int, out_root: Path) -> dict:
    """Run the ``taxi_mellow_sarsa.py`` smoke reproduction (3-policy sweep).

    Sweeps EpsGreedy (eps=0.05), Boltzmann (beta=0.5), Mellowmax
    (omega=0.5) with SARSA on the Asadi-Littman Taxi. ``n_steps=20_000``,
    ``n_experiment=2``.
    """
    task = "taxi"
    algorithm = "SARSA_policy_sweep"
    run_dir = make_run_dir(
        base=out_root,
        phase="phase1",
        suite="smoke",
        task=task,
        algorithm=algorithm,
        seed=seed,
        exist_ok=True,
    )
    stdout_path = run_dir / "stdout.log"

    n_steps = 20_000
    n_experiment = 2
    policy_values = {"epsilon": 0.05, "boltzmann": 0.5, "mellow": 0.5}

    mdp_cfg = {
        "kind": "FiniteMDP",
        "generator": "generate_taxi",
        "grid": str(_TAXI_GRID),
        "prob": 0.9,
        "rew": [0, 1, 3, 15],
        "gamma": 0.99,
        "horizon": "inf",
    }

    rt = RunTimer()
    policy_results: dict[str, dict] = {}
    overall_non_degenerate = True
    n_policies = 0

    t_total_0 = time.perf_counter()
    with stdout_to_log(stdout_path):
        print(f"[smoke:{task}] seed={seed}")
        seed_everything(seed)

        from mushroom_rl.core import Logger
        from mushroom_rl.policy import Boltzmann, EpsGreedy, Mellowmax

        algs = {EpsGreedy: "epsilon", Boltzmann: "boltzmann", Mellowmax: "mellow"}
        n_policies = len(algs)

        logger = Logger("taxi_mellow_sarsa_sweep", results_dir=None)
        logger.strong_line()
        logger.info(
            f"Smoke: taxi_mellow_sarsa (n_experiment={n_experiment}, n_steps={n_steps})"
        )

        rep_seeds = _derive_rep_seeds(seed, n_experiment)

        for policy_cls in (EpsGreedy, Boltzmann, Mellowmax):
            policy_name = algs[policy_cls]
            param_value = policy_values[policy_name]
            logger.info(f"Policy: {policy_name} (value={param_value})")

            mean_rewards: list[float] = []
            last_agent = None
            t0 = time.perf_counter()
            with rt.phase("fit"):
                for rs in rep_seeds:
                    mean_r, agent, n_rew = _taxi_run_one(
                        policy_cls, param_value, n_steps, rs
                    )
                    mean_rewards.append(mean_r)
                    last_agent = agent
                    print(
                        f"[smoke:{task}] {policy_name} rep_seed={rs}: "
                        f"mean_step_reward={mean_r:.6f}, n_rewards={n_rew}"
                    )
            policy_wall_s = time.perf_counter() - t0

            mean_J = float(np.mean(mean_rewards))
            q_stats = _q_table_stats_plain(last_agent)
            q_nonzero = q_stats["nnz"] > 0
            no_nan_inf = (not q_stats["has_nan"]) and (not q_stats["has_inf"])
            reward_nonzero = bool(np.any(np.asarray(mean_rewards) != 0.0))
            reward_finite = bool(np.all(np.isfinite(np.asarray(mean_rewards))))
            policy_non_degenerate = (
                q_nonzero and no_nan_inf and reward_nonzero and reward_finite
            )
            overall_non_degenerate = (
                overall_non_degenerate and policy_non_degenerate
            )

            print(
                f"[smoke:{task}] {policy_name}: "
                f"mean_J={mean_J:.6f}, Q.shape={q_stats['shape']}, "
                f"nnz={q_stats['nnz']}, wall_s={policy_wall_s:.2f}, "
                f"non_degenerate={policy_non_degenerate}"
            )

            policy_results[policy_name] = {
                "policy_class": policy_cls.__name__,
                "param_value": float(param_value),
                "per_rep_mean_reward": [float(x) for x in mean_rewards],
                "mean_J": mean_J,
                "q_table_shape": q_stats["shape"],
                "q_table_nnz": q_stats["nnz"],
                "q_table_min": q_stats["min"],
                "q_table_max": q_stats["max"],
                "q_table_mean": q_stats["mean"],
                "q_has_nan": q_stats["has_nan"],
                "q_has_inf": q_stats["has_inf"],
                "reward_nonzero": reward_nonzero,
                "reward_finite": reward_finite,
                "wall_s": float(policy_wall_s),
                "non_degenerate": policy_non_degenerate,
            }

        if not overall_non_degenerate:
            print(f"[smoke:{task}] WARNING: non-degeneracy check FAILED")

    wall_s = time.perf_counter() - t_total_0
    timings = rt.to_dict()

    run_json_path = write_run_json(
        run_dir,
        config={
            "source": "mushroom-rl-dev/examples/taxi_mellow_sarsa/taxi_mellow.py",
            "seed": seed,
            "n_steps": n_steps,
            "n_steps_per_fit": 1,
            "n_experiment": n_experiment,
            "mdp": mdp_cfg,
            "policy_values": policy_values,
            "agent": {
                "algorithm": "SARSA",
                "learning_rate": "Parameter(value=0.15)",
            },
        },
        phase="phase1",
        task=task,
        algorithm=algorithm,
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": overall_non_degenerate,
            "policies_sweep": sorted(policy_results.keys()),
            "per_policy": policy_results,
            "timings": timings,
            "total_wall_s": float(wall_s),
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke:{task}] wrote {run_json_path}")

    return {
        "name": task,
        "seed": seed,
        "passed": bool(overall_non_degenerate),
        "J_final": None,
        "wall_s": float(wall_s),
        "run_dir": str(run_dir),
        "summary": f"({n_policies} policies)",
    }


# ============================================================================
# Example 5: mountain_car (TrueOnlineSARSALambda + Gymnasium)
# ============================================================================


def run_mountain_car(seed: int, out_root: Path) -> dict:
    """Run the ``mountain_car_sarsa.py`` smoke reproduction.

    Single TrueOnlineSARSALambda + Tiles + LinearApproximator agent on
    Gymnasium ``MountainCar-v0``. ``n_steps=3000`` train, ``horizon=500``,
    2 greedy eval episodes.
    """
    task = "mountain_car"
    algorithm = "TrueOnlineSARSALambda"
    run_dir = make_run_dir(
        base=out_root,
        phase="phase1",
        suite="smoke",
        task=task,
        algorithm=algorithm,
        seed=seed,
        exist_ok=True,
    )
    stdout_path = run_dir / "stdout.log"

    n_steps = 3_000
    eval_n_episodes = 2
    eval_horizon = 500

    mdp_cfg = {
        "kind": "Gymnasium",
        "name": "MountainCar-v0",
        "horizon": eval_horizon,
        "gamma": 1.0,
        "headless": True,
    }
    features_cfg = {
        "kind": "Tiles",
        "n_tilings": 10,
        "per_tiling_shape": [10, 10],
    }
    agent_cfg = {
        "algorithm": "TrueOnlineSARSALambda",
        "approximator": "LinearApproximator",
        "lambda_coeff": 0.9,
        "alpha": 0.1,
        "learning_rate_formula": "alpha / n_tilings",
        "epsilon": 0.0,
    }

    rt = RunTimer()
    non_degenerate = False
    J_mean = float("nan")
    metrics: dict = {}

    t_total_0 = time.perf_counter()
    with stdout_to_log(stdout_path):
        print(f"[smoke:{task}] seed={seed}")
        seed_everything(seed)

        from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
        from mushroom_rl.core import Core, Logger
        from mushroom_rl.environments import Gymnasium
        from mushroom_rl.features import Features
        from mushroom_rl.features.tiles import Tiles
        from mushroom_rl.policy import EpsGreedy
        from mushroom_rl.rl_utils.parameters import Parameter

        logger = Logger(TrueOnlineSARSALambda.__name__, results_dir=None)
        logger.strong_line()
        logger.info(
            "Smoke: mountain_car_sarsa "
            f"(n_steps={n_steps}, horizon={eval_horizon}, gamma=1.0)"
        )

        mdp = Gymnasium(
            name=mdp_cfg["name"],
            horizon=mdp_cfg["horizon"],
            gamma=mdp_cfg["gamma"],
            headless=mdp_cfg["headless"],
        )
        # Gymnasium reset must receive the seed explicitly
        # (lessons.md 2026-04-16).
        mdp.seed(seed)

        epsilon = Parameter(value=agent_cfg["epsilon"])
        pi = EpsGreedy(epsilon=epsilon)
        n_tilings = features_cfg["n_tilings"]
        tilings = Tiles.generate(
            n_tilings,
            features_cfg["per_tiling_shape"],
            mdp.info.observation_space.low,
            mdp.info.observation_space.high,
        )
        features = Features(tilings=tilings)

        learning_rate = Parameter(agent_cfg["alpha"] / n_tilings)
        approximator_params = dict(
            input_shape=(features.size,),
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            phi=features,
        )
        agent = TrueOnlineSARSALambda(
            mdp.info,
            pi,
            approximator_params=approximator_params,
            learning_rate=learning_rate,
            lambda_coeff=agent_cfg["lambda_coeff"],
        )

        core = Core(agent, mdp)
        with rt.phase("fit"):
            t0 = time.perf_counter()
            core.learn(
                n_steps=n_steps, n_steps_per_fit=1, quiet=True, render=False
            )
            train_wall_s = time.perf_counter() - t0
        print(
            f"[smoke:{task}] learn(n_steps={n_steps}) done in "
            f"{train_wall_s:.2f}s"
        )

        mdp.seed(seed + 1)
        with rt.phase("eval"):
            t1 = time.perf_counter()
            eval_dataset = core.evaluate(
                n_episodes=eval_n_episodes, render=False, quiet=True
            )
            eval_wall_s = time.perf_counter() - t1

        J_eval = np.asarray(eval_dataset.undiscounted_return, dtype=np.float64)
        J_mean = float(J_eval.mean())
        print(
            f"[smoke:{task}] evaluate(n_episodes={eval_n_episodes}) in "
            f"{eval_wall_s:.2f}s; undiscounted_returns={J_eval.tolist()}"
        )

        w_stats = _weights_stats(agent)
        weights_nonzero = w_stats["nnz"] > 0
        no_nan_inf = (not w_stats["has_nan"]) and (not w_stats["has_inf"])
        returns_finite = bool(np.all(np.isfinite(J_eval)))
        non_degenerate = weights_nonzero and no_nan_inf and returns_finite

        print(
            f"[smoke:{task}] weights: shape={w_stats['shape']}, "
            f"nnz={w_stats['nnz']}, l2={w_stats['l2_norm']:.6f}, "
            f"J_mean={J_mean:.4f}, non_degenerate={non_degenerate}"
        )
        if not non_degenerate:
            print(f"[smoke:{task}] WARNING: non-degeneracy check FAILED")

        try:
            mdp.stop()
        except Exception as exc:
            print(f"[smoke:{task}] mdp.stop() raised (ignored): {exc!r}")

        metrics = {
            "train_n_steps": int(n_steps),
            "train_wall_s": float(train_wall_s),
            "eval_n_episodes": int(eval_n_episodes),
            "eval_horizon": int(eval_horizon),
            "eval_wall_s": float(eval_wall_s),
            "J_per_episode": [float(x) for x in J_eval],
            "J_mean": float(J_eval.mean()),
            "J_min": float(J_eval.min()),
            "J_max": float(J_eval.max()),
            "returns_finite": returns_finite,
            "weights_shape": w_stats["shape"],
            "weights_size": w_stats["size"],
            "weights_nnz": w_stats["nnz"],
            "weights_min": w_stats["min"],
            "weights_max": w_stats["max"],
            "weights_mean": w_stats["mean"],
            "weights_l2_norm": w_stats["l2_norm"],
            "weights_nonzero": weights_nonzero,
            "weights_has_nan": w_stats["has_nan"],
            "weights_has_inf": w_stats["has_inf"],
            "non_degenerate": non_degenerate,
        }

    wall_s = time.perf_counter() - t_total_0
    timings = rt.to_dict()

    run_json_path = write_run_json(
        run_dir,
        config={
            "source": "mushroom-rl-dev/examples/mountain_car_sarsa.py",
            "seed": seed,
            "n_steps": n_steps,
            "n_steps_per_fit": 1,
            "eval_n_episodes": eval_n_episodes,
            "mdp": mdp_cfg,
            "features": features_cfg,
            "agent": agent_cfg,
        },
        phase="phase1",
        task=task,
        algorithm=algorithm,
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": non_degenerate,
            "metrics": metrics,
            "timings": timings,
            "total_wall_s": float(wall_s),
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke:{task}] wrote {run_json_path}")

    return {
        "name": task,
        "seed": seed,
        "passed": bool(non_degenerate),
        "J_final": float(J_mean) if np.isfinite(J_mean) else None,
        "wall_s": float(wall_s),
        "run_dir": str(run_dir),
        "summary": f"J_final={J_mean:.1f}" if np.isfinite(J_mean) else "J_final=NaN",
    }


# ============================================================================
# Example 6: puddle_world (TrueOnlineSARSALambda + native PuddleWorld)
# ============================================================================


def run_puddle_world(seed: int, out_root: Path) -> dict:
    """Run the ``puddle_world_sarsa.py`` smoke reproduction.

    Single TrueOnlineSARSALambda + Tiles + LinearApproximator agent on
    native ``PuddleWorld``. ``n_steps=3000`` train, ``horizon=500``, 2
    greedy eval episodes (pre- and post-training).
    """
    task = "puddle_world"
    algorithm = "TrueOnlineSARSALambda"
    run_dir = make_run_dir(
        base=out_root,
        phase="phase1",
        suite="smoke",
        task=task,
        algorithm=algorithm,
        seed=seed,
        exist_ok=True,
    )
    stdout_path = run_dir / "stdout.log"

    n_steps = 3_000
    eval_n_episodes = 2
    env_horizon = 500

    mdp_cfg = {
        "kind": "PuddleWorld",
        "horizon": env_horizon,
        "gamma": 0.99,
        "start": [0.2, 0.4],
        "goal": [1.0, 1.0],
        "goal_threshold": 0.1,
        "noise_step": 0.025,
        "noise_reward": 0.0,
        "reward_goal": 0.0,
        "thrust": 0.05,
    }
    features_cfg = {
        "kind": "Tiles",
        "n_tilings": 10,
        "per_tiling_shape": [10, 10],
    }
    agent_cfg = {
        "algorithm": "TrueOnlineSARSALambda",
        "approximator": "LinearApproximator",
        "lambda_coeff": 0.9,
        "alpha": 0.1,
        "learning_rate_formula": "alpha / n_tilings",
        "epsilon": 0.1,
    }

    rt = RunTimer()
    non_degenerate = False
    J_disc_mean = float("nan")
    metrics: dict = {}

    t_total_0 = time.perf_counter()
    with stdout_to_log(stdout_path):
        print(f"[smoke:{task}] seed={seed}")
        seed_everything(seed)

        from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
        from mushroom_rl.core import Core, Logger
        from mushroom_rl.environments import PuddleWorld
        from mushroom_rl.features import Features
        from mushroom_rl.features.tiles import Tiles
        from mushroom_rl.policy import EpsGreedy
        from mushroom_rl.rl_utils.parameters import Parameter

        logger = Logger("PuddleWorld_TrueOnlineSARSALambda", results_dir=None)
        logger.strong_line()
        logger.info(
            "Smoke: puddle_world_sarsa "
            f"(n_steps={n_steps}, horizon={env_horizon}, gamma=0.99)"
        )

        mdp = PuddleWorld(horizon=env_horizon)
        # PuddleWorld stochasticity is numpy-global; seed_everything
        # already covers it (PuddleWorld.seed() is the base-class stub).

        epsilon = Parameter(value=agent_cfg["epsilon"])
        pi = EpsGreedy(epsilon=epsilon)
        n_tilings = features_cfg["n_tilings"]
        tilings = Tiles.generate(
            n_tilings,
            features_cfg["per_tiling_shape"],
            mdp.info.observation_space.low,
            mdp.info.observation_space.high,
        )
        features = Features(tilings=tilings)

        learning_rate = Parameter(agent_cfg["alpha"] / n_tilings)
        approximator_params = dict(
            input_shape=(features.size,),
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            phi=features,
        )
        agent = TrueOnlineSARSALambda(
            mdp.info,
            pi,
            approximator_params=approximator_params,
            learning_rate=learning_rate,
            lambda_coeff=agent_cfg["lambda_coeff"],
        )

        core = Core(agent, mdp)
        with rt.phase("eval"):
            t_eval0 = time.perf_counter()
            baseline_dataset = core.evaluate(
                n_episodes=eval_n_episodes, render=False, quiet=True
            )
            baseline_eval_wall_s = time.perf_counter() - t_eval0
        J_baseline = np.asarray(
            baseline_dataset.discounted_return, dtype=np.float64
        )
        print(
            f"[smoke:{task}] baseline J_disc mean={float(J_baseline.mean()):.4f}, "
            f"values={J_baseline.tolist()} "
            f"(wall={baseline_eval_wall_s:.2f}s)"
        )

        with rt.phase("fit"):
            t0 = time.perf_counter()
            core.learn(
                n_steps=n_steps, n_steps_per_fit=1, quiet=True, render=False
            )
            train_wall_s = time.perf_counter() - t0
        print(
            f"[smoke:{task}] learn(n_steps={n_steps}) done in "
            f"{train_wall_s:.2f}s"
        )

        with rt.phase("eval"):
            t1 = time.perf_counter()
            eval_dataset = core.evaluate(
                n_episodes=eval_n_episodes, render=False, quiet=True
            )
            eval_wall_s = time.perf_counter() - t1
        J_disc = np.asarray(eval_dataset.discounted_return, dtype=np.float64)
        J_undisc = np.asarray(
            eval_dataset.undiscounted_return, dtype=np.float64
        )
        J_disc_mean = float(J_disc.mean())
        print(
            f"[smoke:{task}] evaluate(n_episodes={eval_n_episodes}) in "
            f"{eval_wall_s:.2f}s; disc={J_disc.tolist()}, "
            f"undisc={J_undisc.tolist()}"
        )

        w_stats = _weights_stats(agent)
        weights_nonzero = w_stats["nnz"] > 0
        no_nan_inf = (not w_stats["has_nan"]) and (not w_stats["has_inf"])
        returns_finite = bool(
            np.all(np.isfinite(J_disc)) and np.all(np.isfinite(J_undisc))
        )
        non_degenerate = weights_nonzero and no_nan_inf and returns_finite

        print(
            f"[smoke:{task}] weights: shape={w_stats['shape']}, "
            f"nnz={w_stats['nnz']}, l2={w_stats['l2_norm']:.6f}, "
            f"J_disc_mean={J_disc_mean:.4f}, J_undisc_mean="
            f"{float(J_undisc.mean()):.4f}, non_degenerate={non_degenerate}"
        )
        if not non_degenerate:
            print(f"[smoke:{task}] WARNING: non-degeneracy check FAILED")

        try:
            mdp.stop()
        except Exception as exc:
            print(f"[smoke:{task}] mdp.stop() raised (ignored): {exc!r}")

        metrics = {
            "train_n_steps": int(n_steps),
            "train_wall_s": float(train_wall_s),
            "baseline_eval_n_episodes": int(eval_n_episodes),
            "baseline_eval_wall_s": float(baseline_eval_wall_s),
            "J_baseline_discounted_per_episode": [
                float(x) for x in J_baseline
            ],
            "J_baseline_discounted_mean": float(J_baseline.mean()),
            "eval_n_episodes": int(eval_n_episodes),
            "eval_horizon": int(env_horizon),
            "eval_wall_s": float(eval_wall_s),
            "J_discounted_per_episode": [float(x) for x in J_disc],
            "J_discounted_mean": float(J_disc.mean()),
            "J_discounted_min": float(J_disc.min()),
            "J_discounted_max": float(J_disc.max()),
            "J_undiscounted_per_episode": [float(x) for x in J_undisc],
            "J_undiscounted_mean": float(J_undisc.mean()),
            "J_undiscounted_min": float(J_undisc.min()),
            "J_undiscounted_max": float(J_undisc.max()),
            "returns_finite": returns_finite,
            "weights_shape": w_stats["shape"],
            "weights_size": w_stats["size"],
            "weights_nnz": w_stats["nnz"],
            "weights_min": w_stats["min"],
            "weights_max": w_stats["max"],
            "weights_mean": w_stats["mean"],
            "weights_l2_norm": w_stats["l2_norm"],
            "weights_nonzero": weights_nonzero,
            "weights_has_nan": w_stats["has_nan"],
            "weights_has_inf": w_stats["has_inf"],
            "non_degenerate": non_degenerate,
        }

    wall_s = time.perf_counter() - t_total_0
    timings = rt.to_dict()

    run_json_path = write_run_json(
        run_dir,
        config={
            "source": "mushroom-rl-dev/examples/puddle_world_sarsa.py",
            "seed": seed,
            "n_steps": n_steps,
            "n_steps_per_fit": 1,
            "eval_n_episodes": eval_n_episodes,
            "mdp": mdp_cfg,
            "features": features_cfg,
            "agent": agent_cfg,
        },
        phase="phase1",
        task=task,
        algorithm=algorithm,
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": non_degenerate,
            "metrics": metrics,
            "timings": timings,
            "total_wall_s": float(wall_s),
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke:{task}] wrote {run_json_path}")

    return {
        "name": task,
        "seed": seed,
        "passed": bool(non_degenerate),
        "J_final": float(J_disc_mean) if np.isfinite(J_disc_mean) else None,
        "wall_s": float(wall_s),
        "run_dir": str(run_dir),
        "summary": (
            f"J_final={J_disc_mean:.1f}"
            if np.isfinite(J_disc_mean)
            else "J_final=NaN"
        ),
    }


# ============================================================================
# Dispatch + CLI
# ============================================================================


_DISPATCH: dict[str, Callable[[int, Path], dict]] = {
    "simple_chain": run_simple_chain,
    "grid_world": run_grid_world,
    "double_chain": run_double_chain,
    "taxi": run_taxi,
    "mountain_car": run_mountain_car,
    "puddle_world": run_puddle_world,
}


def _print_summary(results: list[dict]) -> None:
    """Print the Phase I smoke summary table to stdout."""
    print("")
    print("== Phase I smoke summary ==")
    name_w = max((len(r["name"]) for r in results), default=12)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['name']:<{name_w}}  seed={r['seed']}  {status}  "
            f"{r['summary']:<18}  {r['wall_s']:.1f}s"
        )
    total_wall = sum(r["wall_s"] for r in results)
    n_pass = sum(1 for r in results if r["passed"])
    n = len(results)
    if n_pass == n:
        print(
            f"\nAll {n} smoke run{'s' if n != 1 else ''} PASSED. "
            f"Total wall: {total_wall:.1f}s"
        )
    else:
        print(
            f"\n{n_pass}/{n} smoke runs passed ({n - n_pass} failed). "
            f"Total wall: {total_wall:.1f}s"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on full success, 1 if any run fails.

    Parameters
    ----------
    argv:
        Optional argv list for testing. When ``None`` (default),
        :func:`sys.argv` is used.
    """
    parser = argparse.ArgumentParser(
        prog="run_phase1_smoke",
        description=(
            "Unified Phase I smoke harness. Wraps the six reproductions "
            "(simple_chain, grid_world, double_chain, taxi, mountain_car, "
            "puddle_world) behind a uniform logging + run.json contract."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Integer seed (default: 11). Must be non-negative.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(RESULT_ROOT),
        help=(
            "Base output directory (default: "
            "results/weighted_lse_dp). Artifacts land under "
            "<out_root>/phase1/smoke/<task>/<algorithm>/seed_<seed>/."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=list(EXAMPLE_NAMES),
        help=(
            "Run only one example by name. If omitted, run all six "
            "sequentially in the canonical order."
        ),
    )
    args = parser.parse_args(argv)

    seed: int = int(args.seed)
    out_root: Path = Path(args.out_root).resolve()

    selected: tuple[str, ...]
    if args.only is not None:
        selected = (args.only,)
    else:
        selected = EXAMPLE_NAMES

    print(
        f"[smoke] Phase I harness: seed={seed}, out_root={out_root}, "
        f"examples={list(selected)}"
    )

    results: list[dict] = []
    for name in selected:
        fn = _DISPATCH[name]
        print(f"\n[smoke] === running: {name} ===")
        t0 = time.perf_counter()
        try:
            result = fn(seed, out_root)
        except Exception as exc:
            wall_s = time.perf_counter() - t0
            tb = traceback.format_exc()
            # Failures must not crash the whole harness; record and move
            # on so subsequent examples still run and the summary still
            # reflects the full state of the suite.
            print(f"[smoke] {name}: EXCEPTION after {wall_s:.2f}s: {exc!r}")
            print(tb)
            # Best-effort run_dir for the failing example (not guaranteed
            # to exist if make_run_dir itself was what failed).
            result = {
                "name": name,
                "seed": seed,
                "passed": False,
                "J_final": None,
                "wall_s": float(wall_s),
                "run_dir": None,
                "summary": "EXCEPTION",
                "error": repr(exc),
                "traceback": tb,
            }
        results.append(result)

    # Write the aggregate summary. Path is fixed at
    # <out_root>/phase1/smoke/smoke_summary.json so aggregation scripts
    # can find it without knowing which examples were selected.
    summary_path = out_root / "phase1" / "smoke" / "smoke_summary.json"
    save_json(
        summary_path,
        {
            "seed": seed,
            "out_root": str(out_root),
            "examples_selected": list(selected),
            "n_examples": len(results),
            "n_passed": sum(1 for r in results if r["passed"]),
            "all_passed": all(r["passed"] for r in results),
            "total_wall_s": float(sum(r["wall_s"] for r in results)),
            "results": results,
        },
    )
    print(f"\n[smoke] wrote {summary_path}")

    _print_summary(results)

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
