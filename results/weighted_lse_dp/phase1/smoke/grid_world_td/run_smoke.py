"""Smoke reproduction of ``mushroom-rl-dev/examples/grid_world_td.py``.

Spec anchor: Phase I spec §4.2.2 / tasks/todo.md item 13.

This is a one-off wrapper, NOT a reusable runner. It exists to prove that the
vendored MushroomRL fork, the local ``.venv``, and the
``experiments.weighted_lse_dp.common`` helpers form a working stack end-to-end
for the Hasselt grid-world TD example. The real runner will live under
``experiments/weighted_lse_dp/run.py``.

Differences vs. the upstream example:
    - The upstream script calls ``np.random.seed()`` (wall-clock seeding) and
      launches ``joblib.Parallel`` across 10,000 experiments per algorithm,
      per decay exponent. Here we seed from the config (seed=11) via
      ``seed_everything`` and run a tiny sequential sweep so the smoke
      finishes in well under 30 s on the reference laptop.
    - Reduced budget: ``n_steps=2000`` train (vs 10,000), ``n_experiment=3``
      (vs 10,000), and only ``exp=0.8`` (vs both ``{1, 0.8}``).
    - All 5 TD variants (QLearning, DoubleQLearning, WeightedQLearning,
      SpeedyQLearning, SARSA) from the example are still exercised.
    - No matplotlib rendering, no ``.npy`` dumps in the cwd.
    - Captures stdout to ``stdout.log`` via ``stdout_to_log``.
    - Writes ``run.json`` via ``write_run_json`` with the smoke-suite header.

Non-degeneracy contract:
    For each algorithm the smoke asserts that (a) the Q-table has at least
    one non-zero entry, (b) no NaN/Inf, and (c) the averaged reward trace
    is not identically zero. Failures are flagged but run.json is still
    written so the failure is auditable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root and vendored MushroomRL must be importable regardless of cwd.
_REPO_ROOT = "/Users/liq/Documents/Claude/Projects/LSE_RL"
_MUSHROOM = "/Users/liq/Documents/Claude/Projects/LSE_RL/mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from experiments.weighted_lse_dp.common.io import stdout_to_log  # noqa: E402
from experiments.weighted_lse_dp.common.manifests import write_run_json  # noqa: E402
from experiments.weighted_lse_dp.common.seeds import seed_everything  # noqa: E402


def _run_one(
    algorithm_class,
    exp: float,
    n_steps: int,
    rep_seed: int,
):
    """Mirror the body of ``examples/grid_world_td.py::experiment``.

    The only semantic difference is that we re-seed ``numpy`` with
    ``rep_seed`` instead of calling the upstream's wall-clock
    ``np.random.seed()``. Everything else (env, policy, agent, callbacks,
    ``Core.learn`` call) is identical.
    """
    # Deferred imports so the stdout tee also captures mushroom_rl's own
    # warnings (pygame/cv2 dylib loads, etc.).
    from mushroom_rl.core import Core
    from mushroom_rl.environments import GridWorldVanHasselt
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import DecayParameter
    from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ

    # Re-seed per repetition so sequential repetitions are not identical.
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
    callbacks = [collect_dataset, collect_max_Q]
    core = Core(agent, mdp, callbacks)

    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    # shape: (n_steps,) float per-transition reward. Note: the upstream
    # example calls ``collect_dataset.get().rewards`` but the current
    # MushroomRL Dataset exposes the singular ``reward`` property; we
    # accommodate both for robustness.
    ds = collect_dataset.get()
    reward_prop = getattr(ds, "reward", None)
    if reward_prop is None:
        reward_prop = getattr(ds, "rewards")
    rewards = np.asarray(reward_prop, dtype=np.float64)
    # shape: (n_steps,) max_a Q(start, a) after each transition
    max_Qs = np.asarray(collect_max_Q.get(), dtype=np.float64)

    # Expose the agent so the caller can inspect the final Q-table.
    return rewards, max_Qs, agent


def _q_table_stats(agent) -> dict:
    """Return a minimal non-degeneracy summary of the agent's Q-table(s)."""
    # Algorithms like DoubleQLearning expose an EnsembleTable; average over
    # members when needed so we still get a scalar shape/stat report.
    q = agent.Q
    if hasattr(q, "table"):
        # shape: (n_states, n_actions) for a plain Table
        arr = np.asarray(q.table, dtype=np.float64)
    else:
        # EnsembleTable: iterate members and stack -> shape (k, S, A).
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


def main() -> int:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    run_dir = Path(
        "/Users/liq/Documents/Claude/Projects/LSE_RL"
        "/results/weighted_lse_dp/phase1/smoke/grid_world_td"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"

    # ------------------------------------------------------------------
    # Config (explicit, no hidden defaults).
    # ------------------------------------------------------------------
    seed = 11
    n_steps = 2000  # reduced from upstream 10,000
    n_experiment = 3  # reduced from upstream 10,000
    exp_value = 0.8  # subset of upstream {1.0, 0.8}

    # Grid env defaults match GridWorldVanHasselt() constructor defaults.
    grid_cfg = {
        "height": 3,
        "width": 3,
        "goal": [0, 2],
        "start": [2, 0],
        "gamma": 0.95,
        "horizon": "inf",
    }

    # ------------------------------------------------------------------
    # Body: all logging tee'd to stdout.log.
    # ------------------------------------------------------------------
    algo_results: dict[str, dict] = {}
    overall_non_degenerate = True

    with stdout_to_log(stdout_path):
        print(f"[smoke] seed={seed}")
        seed_everything(seed)

        from mushroom_rl.algorithms.value import (  # noqa: E402
            DoubleQLearning,
            QLearning,
            SARSA,
            SpeedyQLearning,
            WeightedQLearning,
        )
        from mushroom_rl.core import Logger  # noqa: E402

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

        logger = Logger(QLearning.__name__, results_dir=None)
        logger.strong_line()
        logger.info(
            "Smoke reproduction of examples/grid_world_td.py "
            f"(exp={exp_value}, n_experiment={n_experiment}, n_steps={n_steps})"
        )

        for algo in algo_classes:
            algo_name = names[algo]
            logger.info(f"Alg: {algo_name}")

            # Per-repetition seeds derived from the config seed so the whole
            # smoke is deterministic. shape: (n_experiment,)
            rep_seeds = [seed * 100 + i for i in range(n_experiment)]

            rewards_list = []
            max_Qs_list = []
            last_agent = None
            for rs in rep_seeds:
                rewards, max_Qs, agent = _run_one(
                    algo, exp_value, n_steps, rs
                )
                rewards_list.append(rewards)
                max_Qs_list.append(max_Qs)
                last_agent = agent

            # shape: (n_experiment, n_steps)
            rewards_arr = np.stack(rewards_list, axis=0)
            max_Qs_arr = np.stack(max_Qs_list, axis=0)
            # shape: (n_steps,)
            mean_reward = rewards_arr.mean(axis=0)
            mean_max_Q = max_Qs_arr.mean(axis=0)

            q_stats = _q_table_stats(last_agent)

            reward_nonzero = bool(np.any(mean_reward != 0.0))
            q_nonzero = q_stats["nnz"] > 0
            no_nan_inf = (not q_stats["has_nan"]) and (not q_stats["has_inf"])
            algo_non_degenerate = reward_nonzero and q_nonzero and no_nan_inf
            overall_non_degenerate = (
                overall_non_degenerate and algo_non_degenerate
            )

            print(
                f"[smoke] {algo_name}: "
                f"reward_sum_over_steps(mean over reps)={float(mean_reward.sum()):.4f}, "
                f"max_Q_at_start(final_step_mean)={float(mean_max_Q[-1]):.4f}, "
                f"Q-table shape={q_stats['shape']}, "
                f"nnz={q_stats['nnz']}, "
                f"min={q_stats['min']:.4f}, max={q_stats['max']:.4f}, "
                f"has_nan={q_stats['has_nan']}, has_inf={q_stats['has_inf']}, "
                f"non_degenerate={algo_non_degenerate}"
            )

            algo_results[algo_name] = {
                "reward_sum_mean": float(mean_reward.sum()),
                "reward_mean_over_reps_shape": list(rewards_arr.shape),
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
            print("[smoke] WARNING: overall non-degeneracy check FAILED")

    # --------------------------------------------------------------------
    # Persist run.json. Kept outside the stdout tee so the "[smoke] wrote ..."
    # line is written after the log file is closed.
    # --------------------------------------------------------------------
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
        task="grid_world_td",
        # Not a single algorithm — the example sweeps 5. Use the group tag
        # so the schema field still validates as a non-empty string.
        algorithm="TD_sweep",
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": overall_non_degenerate,
            "algorithms_sweep": sorted(algo_results.keys()),
            "per_algorithm": algo_results,
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke] wrote {run_json_path}")
    return 0 if overall_non_degenerate else 1


if __name__ == "__main__":
    raise SystemExit(main())
