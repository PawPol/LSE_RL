"""Smoke reproduction of ``mushroom-rl-dev/examples/double_chain_q_learning/double_chain.py``.

Spec anchor: Phase I spec §4.2.3 / tasks/todo.md item 14.

One-off wrapper (NOT a reusable runner). Proves that the vendored
MushroomRL ``FiniteMDP`` + the four tabular Q-learning variants from the
upstream example train end-to-end on the Peters double-chain asset under
the local ``.venv`` + ``experiments.weighted_lse_dp.common`` stack.

Differences vs. the upstream example
------------------------------------
- Seeded from config via :func:`seed_everything(11)` (upstream calls the
  wall-clock ``np.random.seed()`` inside each repetition).
- Reduced budget: ``n_steps=2000`` (vs 20,000), ``n_experiment=2`` (vs 5),
  single decay exponent ``exp=1.0`` (vs ``{1.0, 0.51}``).
- All 4 upstream algorithms (QLearning, DoubleQLearning, WeightedQLearning,
  SpeedyQLearning) are still exercised.
- No ``joblib.Parallel``; sequential execution so the stdout tee captures
  every per-repetition log line.
- No ``.npy`` dumps in ``log_path`` — only the canonical smoke tree
  (``stdout.log`` + ``run.json``) is produced.

Non-degeneracy contract
-----------------------
For each algorithm, the smoke asserts:
    (a) the Q-table has at least one non-zero entry,
    (b) no NaN / Inf in the Q-table,
    (c) the collected Q-table tensor over the training trajectory is not
        identically zero (i.e. at least one sampled-state update fired).
Failures are flagged but ``run.json`` is still written so the failure is
auditable.
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


# Upstream double-chain asset lives next to the example we reproduce.
_CHAIN_ASSET_DIR = Path(
    "/Users/liq/Documents/Claude/Projects/LSE_RL"
    "/mushroom-rl-dev/examples/double_chain_q_learning/chain_structure"
)


def _run_one(
    algorithm_class,
    exp: float,
    n_steps: int,
    rep_seed: int,
):
    """Mirror the body of ``examples/double_chain_q_learning/double_chain.py::experiment``.

    Only semantic difference: re-seed numpy with ``rep_seed`` instead of
    the upstream wall-clock ``np.random.seed()``. Env, policy, agent,
    callback, and ``Core.learn`` call are identical.
    """
    # Deferred imports so the stdout tee also captures any mushroom_rl
    # init-time warnings.
    from mushroom_rl.core import Core
    from mushroom_rl.environments import FiniteMDP
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import DecayParameter, Parameter
    from mushroom_rl.utils.callbacks import CollectQ

    # Re-seed per repetition so sequential reps are not identical.
    np.random.seed(rep_seed)

    # MDP from the vendored asset: (9, 2, 9) transition + reward tensors.
    p = np.load(_CHAIN_ASSET_DIR / "p.npy")  # shape: (S, A, S)
    rew = np.load(_CHAIN_ASSET_DIR / "rew.npy")  # shape: (S, A, S)
    mdp = FiniteMDP(p, rew, gamma=0.9)

    # Policy (epsilon=1 pure exploration, mirrors upstream).
    epsilon = Parameter(value=1.0)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent.
    learning_rate = DecayParameter(value=1.0, exp=exp, size=mdp.info.size)
    agent = algorithm_class(mdp.info, pi, learning_rate=learning_rate)

    # Callback collects the Q-table at every step. shape per entry:
    # (S, A) for plain Table, averaged ensemble for DoubleQLearning.
    collect_Q = CollectQ(agent.Q)
    core = Core(agent, mdp, [collect_Q])

    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    # shape: (n_steps, S, A) — length-n_steps list of Q-snapshots.
    Qs = np.asarray(collect_Q.get(), dtype=np.float64)

    return Qs, agent


def _q_table_stats(agent) -> dict:
    """Return a minimal non-degeneracy summary of the agent's Q-table(s)."""
    q = agent.Q
    if hasattr(q, "table"):
        # shape: (S, A) for a plain Table.
        arr = np.asarray(q.table, dtype=np.float64)
    else:
        # EnsembleTable (DoubleQLearning): mean over members -> (S, A).
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
        "/results/weighted_lse_dp/phase1/smoke/double_chain"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"

    # ------------------------------------------------------------------
    # Config (explicit, no hidden defaults).
    # ------------------------------------------------------------------
    seed = 11
    n_steps = 2000  # reduced from upstream 20,000
    n_experiment = 2  # reduced from upstream 5
    exp_value = 1.0  # single decay exponent (upstream sweeps {1.0, 0.51})

    mdp_cfg = {
        "kind": "FiniteMDP",
        "asset": str(_CHAIN_ASSET_DIR),
        "n_states": 9,
        "n_actions": 2,
        "gamma": 0.9,
        "horizon": "inf",
    }

    # ------------------------------------------------------------------
    # Body: tee to stdout.log.
    # ------------------------------------------------------------------
    algo_results: dict[str, dict] = {}
    overall_non_degenerate = True

    with stdout_to_log(stdout_path):
        print(f"[smoke] seed={seed}")
        seed_everything(seed)

        from mushroom_rl.algorithms.value import (  # noqa: E402
            DoubleQLearning,
            QLearning,
            SpeedyQLearning,
            WeightedQLearning,
        )
        from mushroom_rl.core import Logger  # noqa: E402

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

        logger = Logger(QLearning.__name__, results_dir=None)
        logger.strong_line()
        logger.info(
            "Smoke reproduction of examples/double_chain_q_learning/double_chain.py "
            f"(exp={exp_value}, n_experiment={n_experiment}, n_steps={n_steps})"
        )

        for algo in algo_classes:
            algo_name = names[algo]
            logger.info(f"Alg: {algo_name}")

            # Derive per-repetition seeds from the config seed.
            # shape: (n_experiment,)
            rep_seeds = [seed * 100 + i for i in range(n_experiment)]

            Qs_list = []
            last_agent = None
            for rs in rep_seeds:
                Qs, agent = _run_one(algo, exp_value, n_steps, rs)
                Qs_list.append(Qs)
                last_agent = agent

            # shape: (n_experiment, n_steps, S, A)
            Qs_arr = np.stack(Qs_list, axis=0)
            # shape: (n_steps, S, A) — mean over reps, mirrors upstream.
            mean_Qs = Qs_arr.mean(axis=0)
            # shape: (n_steps,) — tracks Q(state=0, action=0), the quantity
            # the upstream example ultimately saves.
            q_s0a0_trace = mean_Qs[:, 0, 0]

            q_stats = _q_table_stats(last_agent)

            traj_nonzero = bool(np.any(mean_Qs != 0.0))
            q_nonzero = q_stats["nnz"] > 0
            no_nan_inf = (not q_stats["has_nan"]) and (not q_stats["has_inf"])
            algo_non_degenerate = traj_nonzero and q_nonzero and no_nan_inf
            overall_non_degenerate = (
                overall_non_degenerate and algo_non_degenerate
            )

            print(
                f"[smoke] {algo_name}: "
                f"Q(s=0,a=0) start={float(q_s0a0_trace[0]):.4f}, "
                f"final={float(q_s0a0_trace[-1]):.4f}, "
                f"trajectory max|Q|={float(np.max(np.abs(mean_Qs))):.4f}, "
                f"Q-table shape={q_stats['shape']}, "
                f"nnz={q_stats['nnz']}, "
                f"min={q_stats['min']:.4f}, max={q_stats['max']:.4f}, "
                f"has_nan={q_stats['has_nan']}, has_inf={q_stats['has_inf']}, "
                f"non_degenerate={algo_non_degenerate}"
            )

            algo_results[algo_name] = {
                "q_s0a0_initial": float(q_s0a0_trace[0]),
                "q_s0a0_final": float(q_s0a0_trace[-1]),
                "q_trajectory_shape": list(mean_Qs.shape),
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
            print("[smoke] WARNING: overall non-degeneracy check FAILED")

    # --------------------------------------------------------------------
    # Persist run.json. Outside the stdout tee so the "[smoke] wrote ..."
    # line is written after the log file is closed.
    # --------------------------------------------------------------------
    run_json_path = write_run_json(
        run_dir,
        config={
            "source": (
                "mushroom-rl-dev/examples/double_chain_q_learning/double_chain.py"
            ),
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
        task="double_chain",
        # The example sweeps 4 tabular algorithms; group-tag the algorithm
        # field so the schema still validates as a non-empty string.
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
