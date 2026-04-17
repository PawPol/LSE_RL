"""Smoke reproduction of ``mushroom-rl-dev/examples/simple_chain_qlearning.py``.

Spec anchor: Phase I spec §4.2.1 / tasks/todo.md item 12.

This is a one-off wrapper, NOT a reusable runner. It exists to prove that the
vendored MushroomRL fork, the local ``.venv``, and the ``experiments.weighted_lse_dp.common``
helpers form a working stack end-to-end. The real runner lives (or will live)
under ``experiments/weighted_lse_dp/run.py``.

Differences vs. the upstream example:
    - The upstream script calls ``np.random.seed()`` (wall-clock seeding).
      Here we seed from the config (seed=11) via ``seed_everything``.
    - Reduced budget: ``n_steps=2000`` train, ``n_steps=500`` eval so the
      smoke finishes in well under 30 s on the reference laptop.
    - Captures stdout to ``stdout.log`` via ``stdout_to_log``.
    - Writes ``run.json`` via ``write_run_json`` with the smoke-suite header.
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


def main() -> int:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    run_dir = Path(
        "/Users/liq/Documents/Claude/Projects/LSE_RL"
        "/results/weighted_lse_dp/phase1/smoke/simple_chain_qlearning"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"

    # ------------------------------------------------------------------
    # Config (explicit, no hidden defaults).
    # ------------------------------------------------------------------
    seed = 11
    n_steps_train = 2000
    n_steps_per_fit = 1
    n_steps_eval = 500  # reduced from upstream's 1000

    # MDP hyperparameters mirror the upstream example.
    chain_state_n = 5
    chain_goal = [2]
    chain_prob = 0.8
    chain_rew = 1.0
    chain_gamma = 0.9

    epsilon_val = 0.15
    learning_rate_val = 0.2

    # ------------------------------------------------------------------
    # Body: all logging tee'd to stdout.log.
    # ------------------------------------------------------------------
    with stdout_to_log(stdout_path):
        print(f"[smoke] seed={seed}")
        seed_everything(seed)

        # Deferred imports so the stdout tee captures any warnings from
        # mushroom_rl / pygame / cv2 dylib load.
        from mushroom_rl.algorithms.value import QLearning
        from mushroom_rl.core import Core, Logger
        from mushroom_rl.environments import generate_simple_chain
        from mushroom_rl.policy import EpsGreedy
        from mushroom_rl.rl_utils.parameters import Parameter

        logger = Logger(QLearning.__name__, results_dir=None)
        logger.strong_line()
        logger.info("Experiment Algorithm: " + QLearning.__name__)

        # MDP
        mdp = generate_simple_chain(
            state_n=chain_state_n,
            goal_states=chain_goal,
            prob=chain_prob,
            rew=chain_rew,
            gamma=chain_gamma,
        )

        # Policy
        pi = EpsGreedy(epsilon=Parameter(value=epsilon_val))

        # Agent
        agent = QLearning(
            mdp.info,
            pi,
            learning_rate=Parameter(value=learning_rate_val),
        )

        # Core
        core = Core(agent, mdp)

        # Initial policy evaluation
        dataset0 = core.evaluate(n_steps=n_steps_eval)
        # shape: (n_episodes_eval,)
        J0_arr = np.asarray(dataset0.discounted_return, dtype=np.float64)
        J0 = float(np.mean(J0_arr))
        logger.info(f"J start: {J0}")

        # Train
        core.learn(n_steps=n_steps_train, n_steps_per_fit=n_steps_per_fit)

        # Final policy evaluation
        dataset1 = core.evaluate(n_steps=n_steps_eval)
        J1_arr = np.asarray(dataset1.discounted_return, dtype=np.float64)
        J1 = float(np.mean(J1_arr))
        logger.info(f"J final: {J1}")

        # ------------------------------------------------------------------
        # Q-table summary + non-degeneracy checks.
        # ------------------------------------------------------------------
        # shape: (state_n, n_actions)  -- here (5, 2)
        q_table = np.asarray(agent.Q.table, dtype=np.float64)
        print("[smoke] Q-table (state_n x n_actions):")
        print(q_table)

        has_nonzero = bool(np.any(q_table != 0.0))
        has_nan = bool(np.any(np.isnan(q_table)))
        has_inf = bool(np.any(np.isinf(q_table)))
        eval_nondegenerate = bool(np.any(J1_arr != 0.0))

        print(
            "[smoke] non_degeneracy: "
            f"has_nonzero={has_nonzero}, has_nan={has_nan}, "
            f"has_inf={has_inf}, eval_nondegenerate={eval_nondegenerate}, "
            f"J0={J0:.6f}, J1={J1:.6f}"
        )

        non_degenerate = (
            has_nonzero
            and (not has_nan)
            and (not has_inf)
            and eval_nondegenerate
        )
        if not non_degenerate:
            # Loud, but still let the wrapper finish and write run.json so the
            # failure is auditable.
            print("[smoke] WARNING: non-degeneracy check FAILED")

    # --------------------------------------------------------------------
    # Persist run.json. Kept outside the stdout tee so the "[smoke] wrote ..."
    # line is written after the log file is closed.
    # --------------------------------------------------------------------
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
            "agent": {"algorithm": "QLearning", "learning_rate": learning_rate_val},
        },
        phase="phase1",
        task="simple_chain_qlearning",
        algorithm="QLearning",
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": non_degenerate,
            "J_start": J0,
            "J_final": J1,
            "q_table_shape": list(q_table.shape),
            "q_table_min": float(q_table.min()),
            "q_table_max": float(q_table.max()),
            "q_table_nnz": int(np.count_nonzero(q_table)),
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke] wrote {run_json_path}")
    return 0 if non_degenerate else 1


if __name__ == "__main__":
    raise SystemExit(main())
