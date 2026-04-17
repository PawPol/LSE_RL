"""Smoke reproduction of ``mushroom-rl-dev/examples/puddle_world_sarsa.py``.

Spec anchor: Phase I spec §4.2.6 / tasks/todo.md item 17.

One-off wrapper (NOT a reusable runner). Proves that the vendored
MushroomRL ``TrueOnlineSARSALambda`` + native ``PuddleWorld`` + ``Tiles``
+ ``LinearApproximator`` trains end-to-end under the local ``.venv`` +
``experiments.weighted_lse_dp.common`` stack.

Differences vs. the upstream example
------------------------------------
- Seeded explicitly from config. The upstream example calls
  ``np.random.seed()`` with no argument (wall-clock seed) inside each
  repetition; here we call :func:`seed_everything(11)`. ``PuddleWorld``
  does not expose a custom ``seed()`` — all its stochasticity goes
  through ``np.random.{uniform,randn}`` which ``seed_everything`` covers
  (lessons.md 2026-04-16 — Gymnasium reset must receive the seed
  explicitly, but PuddleWorld is a native Environment so that rule does
  not apply here).
- Drastically reduced budget: ``n_steps=3_000`` training (vs upstream
  10 epochs * 5000 steps = 50,000). Target < 30s wall-time on the dev
  laptop.
- ``render=False`` throughout so no viewer is opened in a smoke run.
- No ``joblib.Parallel``; sequential single experiment.
- No viewer evaluation pass. Non-degeneracy is probed directly from the
  linear approximator weights + a tiny (2-episode, short-horizon)
  greedy evaluation.

Non-degeneracy contract
-----------------------
The smoke asserts, for the trained agent:
    (a) the linear approximator has at least one non-zero weight,
    (b) no NaN / Inf in the weight vector,
    (c) the greedy evaluation return is finite.
Failures are flagged but ``run.json`` is still written so the failure is
auditable. Note: at 3,000 training steps with PuddleWorld the agent has
almost certainly NOT converged; the returns are expected to be strongly
negative (puddle cost accumulates until the episode times out). Non-
degeneracy just means updates are happening.
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


def _weights_stats(agent) -> dict:
    """Return a minimal non-degeneracy summary of the linear Q weights."""
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


def main() -> int:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    run_dir = Path(
        "/Users/liq/Documents/Claude/Projects/LSE_RL"
        "/results/weighted_lse_dp/phase1/smoke/puddle_world_sarsa"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"

    # ------------------------------------------------------------------
    # Config (explicit, no hidden defaults).
    # ------------------------------------------------------------------
    seed = 11
    # Training budget: 3,000 online steps (vs upstream 50,000). Target
    # < 30s wall-time.
    n_steps = 3_000
    # Eval budget: 2 greedy episodes with a short horizon cap so the
    # un-converged policy cannot chew up the whole wall-time on a single
    # timeout-length episode.
    eval_n_episodes = 2
    # Upstream PuddleWorld default horizon is 5000; shorten to 500 for
    # both training and smoke-eval to bound wall-time.
    env_horizon = 500

    mdp_cfg = {
        "kind": "PuddleWorld",
        "horizon": env_horizon,
        "gamma": 0.99,  # PuddleWorld default
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
        "epsilon": 0.1,  # upstream example value
    }

    # ------------------------------------------------------------------
    # Body: tee to stdout.log.
    # ------------------------------------------------------------------
    metrics: dict = {}
    non_degenerate = True
    t_total_0 = time.perf_counter()

    with stdout_to_log(stdout_path):
        print(f"[smoke] seed={seed}")
        seed_everything(seed)

        # Deferred imports so the stdout tee also captures any
        # mushroom_rl init-time warnings.
        from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
        from mushroom_rl.core import Core, Logger
        from mushroom_rl.environments import PuddleWorld
        from mushroom_rl.features import Features
        from mushroom_rl.features.tiles import Tiles
        from mushroom_rl.policy import EpsGreedy
        from mushroom_rl.rl_utils.parameters import Parameter

        logger = Logger(
            "PuddleWorld_TrueOnlineSARSALambda", results_dir=None
        )
        logger.strong_line()
        logger.info(
            "Smoke reproduction of examples/puddle_world_sarsa.py "
            f"(n_steps={n_steps}, horizon={env_horizon}, gamma=0.99)"
        )

        # --------------------------------------------------------------
        # MDP. Upstream uses horizon=1000 (example) / default 5000.
        # Smoke uses the short cap so both training and evaluation are
        # bounded.
        # --------------------------------------------------------------
        mdp = PuddleWorld(horizon=env_horizon)
        # NOTE: PuddleWorld.seed() is the base-class warning stub; all
        # env stochasticity goes through global numpy RNG, which
        # seed_everything has already pinned.

        # --------------------------------------------------------------
        # Policy: epsilon=0.1 (upstream example).
        # --------------------------------------------------------------
        epsilon = Parameter(value=agent_cfg["epsilon"])
        pi = EpsGreedy(epsilon=epsilon)

        # --------------------------------------------------------------
        # Features: 10 tilings of a 10x10 grid over the observation box.
        # --------------------------------------------------------------
        n_tilings = features_cfg["n_tilings"]
        tilings = Tiles.generate(
            n_tilings,
            features_cfg["per_tiling_shape"],
            mdp.info.observation_space.low,
            mdp.info.observation_space.high,
        )
        features = Features(tilings=tilings)

        # --------------------------------------------------------------
        # Agent: TrueOnlineSARSALambda with linear approximator over the
        # tile-coded features.
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Baseline (pre-training) greedy evaluation — matches upstream
        # structure (eval-before-train to record epoch 0 J).
        # --------------------------------------------------------------
        core = Core(agent, mdp)
        t_eval0 = time.perf_counter()
        baseline_dataset = core.evaluate(
            n_episodes=eval_n_episodes, render=False, quiet=True
        )
        baseline_eval_wall_s = time.perf_counter() - t_eval0
        J_baseline = np.asarray(
            baseline_dataset.discounted_return, dtype=np.float64
        )
        print(
            f"[smoke] baseline discounted_return "
            f"(n_episodes={eval_n_episodes}): "
            f"mean={float(J_baseline.mean()):.4f}, "
            f"values={J_baseline.tolist()} "
            f"(wall={baseline_eval_wall_s:.2f}s)"
        )

        # --------------------------------------------------------------
        # Train. Use n_steps (not n_episodes) so we can hard-cap the
        # smoke wall-time; a single episode can be up to ``horizon``
        # transitions so ``n_steps=3_000`` at ``horizon=500`` is at
        # most ~6 episodes if every one times out.
        # --------------------------------------------------------------
        t0 = time.perf_counter()
        core.learn(
            n_steps=n_steps, n_steps_per_fit=1, quiet=True, render=False
        )
        train_wall_s = time.perf_counter() - t0
        print(
            f"[smoke] learn(n_steps={n_steps}) done in "
            f"{train_wall_s:.2f}s"
        )

        # --------------------------------------------------------------
        # Post-training greedy evaluation.
        # --------------------------------------------------------------
        t1 = time.perf_counter()
        eval_dataset = core.evaluate(
            n_episodes=eval_n_episodes, render=False, quiet=True
        )
        eval_wall_s = time.perf_counter() - t1

        # Upstream reports both discounted and undiscounted returns.
        J_disc = np.asarray(
            eval_dataset.discounted_return, dtype=np.float64
        )
        J_undisc = np.asarray(
            eval_dataset.undiscounted_return, dtype=np.float64
        )
        print(
            f"[smoke] evaluate(n_episodes={eval_n_episodes}) done in "
            f"{eval_wall_s:.2f}s; "
            f"discounted_returns={J_disc.tolist()}, "
            f"undiscounted_returns={J_undisc.tolist()}"
        )

        # --------------------------------------------------------------
        # Non-degeneracy checks.
        # --------------------------------------------------------------
        w_stats = _weights_stats(agent)
        weights_nonzero = w_stats["nnz"] > 0
        no_nan_inf = (not w_stats["has_nan"]) and (not w_stats["has_inf"])
        returns_finite = bool(
            np.all(np.isfinite(J_disc)) and np.all(np.isfinite(J_undisc))
        )
        non_degenerate = weights_nonzero and no_nan_inf and returns_finite

        print(
            f"[smoke] weight_stats: shape={w_stats['shape']}, "
            f"size={w_stats['size']}, nnz={w_stats['nnz']}, "
            f"min={w_stats['min']:.6f}, max={w_stats['max']:.6f}, "
            f"mean={w_stats['mean']:.6f}, l2_norm={w_stats['l2_norm']:.6f}, "
            f"has_nan={w_stats['has_nan']}, has_inf={w_stats['has_inf']}"
        )
        print(
            f"[smoke] J_disc: mean={float(J_disc.mean()):.4f}, "
            f"min={float(J_disc.min()):.4f}, "
            f"max={float(J_disc.max()):.4f}"
        )
        print(
            f"[smoke] J_undisc: mean={float(J_undisc.mean()):.4f}, "
            f"min={float(J_undisc.min()):.4f}, "
            f"max={float(J_undisc.max()):.4f}"
        )
        print(f"[smoke] non_degenerate={non_degenerate}")

        if not non_degenerate:
            print("[smoke] WARNING: non-degeneracy check FAILED")

        # Close the viewer / env (harmless without display).
        try:
            mdp.stop()
        except Exception as exc:
            print(f"[smoke] mdp.stop() raised (ignored): {exc!r}")

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

    total_wall_s = time.perf_counter() - t_total_0

    # --------------------------------------------------------------------
    # Persist run.json. Outside the stdout tee so the "[smoke] wrote ..."
    # line is written after the log file is closed.
    # --------------------------------------------------------------------
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
        task="puddle_world_sarsa",
        algorithm="TrueOnlineSARSALambda",
        seed=seed,
        extra={
            "suite": "smoke",
            "non_degenerate": non_degenerate,
            "total_wall_s": float(total_wall_s),
            "metrics": metrics,
            "stdout_log": "stdout.log",
        },
    )
    print(f"[smoke] wrote {run_json_path}")
    print(f"[smoke] total_wall_s={total_wall_s:.2f}")
    return 0 if non_degenerate else 1


if __name__ == "__main__":
    raise SystemExit(main())
