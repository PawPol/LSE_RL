"""Smoke reproduction of ``mushroom-rl-dev/examples/taxi_mellow_sarsa/taxi_mellow.py``.

Spec anchor: Phase I spec §4.2.4 / tasks/todo.md item 15.

One-off wrapper (NOT a reusable runner). Proves that the vendored
MushroomRL ``generate_taxi`` + ``SARSA`` + the three upstream policies
(EpsGreedy, Boltzmann, Mellowmax) train end-to-end on the Asadi-Littman
Taxi asset under the local ``.venv`` + ``experiments.weighted_lse_dp.common``
stack.

Differences vs. the upstream example
------------------------------------
- Seeded from config via :func:`seed_everything(11)` (upstream calls the
  wall-clock ``np.random.seed()`` inside each repetition).
- Reduced budget: ``n_steps=20_000`` (vs 300,000), ``n_experiment=2``
  (vs 25), single parameter value per policy (vs 10-point sweep).
- All 3 upstream policies (EpsGreedy, Boltzmann, Mellowmax) are still
  exercised with the first value from each upstream range:
    * EpsGreedy:  epsilon = 0.05
    * Boltzmann:  beta    = 0.5
    * Mellowmax:  omega   = 0.5
- No ``joblib.Parallel``; sequential execution so the stdout tee captures
  every per-repetition log line.
- No ``.npy`` dumps in ``log_path`` — only the canonical smoke tree
  (``stdout.log`` + ``run.json``) is produced.

Non-degeneracy contract
-----------------------
For each policy, the smoke asserts:
    (a) the Q-table has at least one non-zero entry,
    (b) no NaN / Inf in the Q-table,
    (c) the mean per-step reward over training is finite and non-zero.
Failures are flagged but ``run.json`` is still written so the failure is
auditable.
"""

from __future__ import annotations

import sys
import time
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


# Upstream grid asset lives next to the example we reproduce.
_TAXI_GRID = Path(
    "/Users/liq/Documents/Claude/Projects/LSE_RL"
    "/mushroom-rl-dev/examples/taxi_mellow_sarsa/grid.txt"
)


def _run_one(policy_cls, param_value: float, n_steps: int, rep_seed: int):
    """Mirror the body of ``examples/taxi_mellow_sarsa/taxi_mellow.py::experiment``.

    Only semantic difference: re-seed numpy with ``rep_seed`` instead of
    the upstream wall-clock ``np.random.seed()``. MDP, policy, agent,
    callback, and ``Core.learn`` call are identical.

    Returns
    -------
    mean_reward : float
        Average per-step reward over the training trajectory (matches the
        upstream return value ``sum(rewards) / n_steps``).
    agent : SARSA
        The trained agent (for Q-table stats).
    n_rewards : int
        Number of per-step reward samples collected.
    """
    # Deferred imports so the stdout tee also captures any mushroom_rl
    # init-time warnings.
    from mushroom_rl.algorithms.value import SARSA
    from mushroom_rl.core import Core
    from mushroom_rl.environments.generators.taxi import generate_taxi
    from mushroom_rl.rl_utils.parameters import Parameter
    from mushroom_rl.utils.callbacks import CollectDataset

    # Re-seed per repetition so sequential reps are not identical.
    np.random.seed(rep_seed)

    # MDP
    mdp = generate_taxi(str(_TAXI_GRID))

    # Policy
    pi = policy_cls(Parameter(value=param_value))

    # Agent
    learning_rate = Parameter(value=0.15)
    agent = SARSA(mdp.info, pi, learning_rate=learning_rate)

    # Algorithm
    collect_dataset = CollectDataset()
    core = Core(agent, mdp, [collect_dataset])

    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    # Collected Dataset — use .reward (singular) per lessons.md 2026-04-16.
    # shape: (n_steps,)
    ds = collect_dataset.get()
    rewards = np.asarray(
        getattr(ds, "reward", getattr(ds, "rewards", None)),
        dtype=np.float64,
    )
    mean_reward = float(rewards.sum()) / float(n_steps)

    return mean_reward, agent, int(rewards.size)


def _q_table_stats(agent) -> dict:
    """Return a minimal non-degeneracy summary of the agent's Q-table."""
    q = agent.Q
    # SARSA uses a plain Table whose .table is (S, A).
    arr = np.asarray(q.table, dtype=np.float64)

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
        "/results/weighted_lse_dp/phase1/smoke/taxi_mellow_sarsa"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"

    # ------------------------------------------------------------------
    # Config (explicit, no hidden defaults).
    # ------------------------------------------------------------------
    seed = 11
    n_steps = 20_000  # reduced from upstream 300,000
    n_experiment = 2  # reduced from upstream 25
    # First entry of each upstream range: EpsGreedy[0]=.05, Boltzmann[0]=.5, Mellowmax[0]=.5.
    policy_values = {
        "epsilon": 0.05,
        "boltzmann": 0.5,
        "mellow": 0.5,
    }

    mdp_cfg = {
        "kind": "FiniteMDP",
        "generator": "generate_taxi",
        "grid": str(_TAXI_GRID),
        "prob": 0.9,
        "rew": [0, 1, 3, 15],
        "gamma": 0.99,
        "horizon": "inf",
    }

    # ------------------------------------------------------------------
    # Body: tee to stdout.log.
    # ------------------------------------------------------------------
    policy_results: dict[str, dict] = {}
    overall_non_degenerate = True

    with stdout_to_log(stdout_path):
        print(f"[smoke] seed={seed}")
        seed_everything(seed)

        from mushroom_rl.core import Logger  # noqa: E402
        from mushroom_rl.policy import Boltzmann, EpsGreedy, Mellowmax  # noqa: E402

        algs = {
            EpsGreedy: "epsilon",
            Boltzmann: "boltzmann",
            Mellowmax: "mellow",
        }

        logger = Logger("taxi_mellow_sarsa", results_dir=None)
        logger.strong_line()
        logger.info(
            "Smoke reproduction of examples/taxi_mellow_sarsa/taxi_mellow.py "
            f"(n_experiment={n_experiment}, n_steps={n_steps})"
        )

        for policy_cls in (EpsGreedy, Boltzmann, Mellowmax):
            policy_name = algs[policy_cls]
            param_value = policy_values[policy_name]
            logger.info(f"Policy: {policy_name} (value={param_value})")

            # Derive per-repetition seeds from the config seed.
            rep_seeds = [seed * 100 + i for i in range(n_experiment)]

            mean_rewards: list[float] = []
            last_agent = None
            t0 = time.perf_counter()
            for rs in rep_seeds:
                mean_r, agent, n_rew = _run_one(
                    policy_cls, param_value, n_steps, rs
                )
                mean_rewards.append(mean_r)
                last_agent = agent
                print(
                    f"[smoke] {policy_name} rep_seed={rs}: "
                    f"mean_step_reward={mean_r:.6f}, n_rewards={n_rew}"
                )
            wall_s = time.perf_counter() - t0

            # Aggregate over reps (mirrors upstream `np.mean(J)`).
            mean_J = float(np.mean(mean_rewards))
            q_stats = _q_table_stats(last_agent)

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
                f"[smoke] {policy_name}: "
                f"mean_J={mean_J:.6f}, "
                f"Q-table shape={q_stats['shape']}, "
                f"nnz={q_stats['nnz']}, "
                f"min={q_stats['min']:.4f}, max={q_stats['max']:.4f}, "
                f"has_nan={q_stats['has_nan']}, has_inf={q_stats['has_inf']}, "
                f"wall_s={wall_s:.2f}, "
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
                "wall_s": float(wall_s),
                "non_degenerate": policy_non_degenerate,
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
                "mushroom-rl-dev/examples/taxi_mellow_sarsa/taxi_mellow.py"
            ),
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
        task="taxi_mellow_sarsa",
        # The example sweeps three policies with the same SARSA core; tag
        # the algorithm field so the schema still validates.
        algorithm="SARSA_policy_sweep",
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": overall_non_degenerate,
            "policies_sweep": sorted(policy_results.keys()),
            "per_policy": policy_results,
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke] wrote {run_json_path}")
    return 0 if overall_non_degenerate else 1


if __name__ == "__main__":
    raise SystemExit(main())
