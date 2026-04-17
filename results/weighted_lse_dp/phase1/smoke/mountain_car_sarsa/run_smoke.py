"""Smoke reproduction of ``mushroom-rl-dev/examples/mountain_car_sarsa.py``.

Spec anchor: Phase I spec §4.2.5 / tasks/todo.md item 16.

One-off wrapper (NOT a reusable runner). Proves that the vendored
MushroomRL ``TrueOnlineSARSALambda`` + ``Gymnasium("MountainCar-v0")`` +
``Tiles`` + ``LinearApproximator`` trains end-to-end under the local
``.venv`` + ``experiments.weighted_lse_dp.common`` stack.

Differences vs. the upstream example
------------------------------------
- Seeded explicitly from config. The upstream example calls
  ``np.random.seed()`` with no argument (wall-clock seed) inside each
  repetition; here we call :func:`seed_everything(11)` AND
  ``mdp.seed(11)`` so the Gymnasium RNG also starts deterministically
  (lessons.md 2026-04-16 — Gymnasium reset must receive the seed
  explicitly).
- Drastically reduced budget: ``n_steps=3_000`` (vs ``n_episodes=40``
  with horizon ``1e4`` => up to 400k steps upstream). Target < 30s
  wall-time on the dev laptop.
- ``headless=True`` / ``render=False`` throughout so no viewer is
  opened in a smoke run.
- No ``joblib.Parallel``; sequential single experiment.
- No viewer evaluation pass. Non-degeneracy is probed directly from the
  linear approximator weights + a greedy evaluation on a tiny (2-episode,
  short-horizon) rollout that cannot dominate wall-time.

Non-degeneracy contract
-----------------------
The smoke asserts, for the trained agent:
    (a) the linear approximator has at least one non-zero weight,
    (b) no NaN / Inf in the weight vector,
    (c) the greedy evaluation return is finite.
Failures are flagged but ``run.json`` is still written so the failure is
auditable. Note: at 3,000 training steps with MountainCar the agent
almost certainly has NOT converged; the greedy return is expected to be
near -200 (episode-timeout penalty). Non-degeneracy just means updates
are happening.
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
        "/results/weighted_lse_dp/phase1/smoke/mountain_car_sarsa"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"

    # ------------------------------------------------------------------
    # Config (explicit, no hidden defaults).
    # ------------------------------------------------------------------
    seed = 11
    # Training budget: 3,000 online steps (vs upstream 40 ep * up to 1e4
    # horizon). Target < 30s wall-time.
    n_steps = 3_000
    # Eval budget: 2 greedy episodes with a short horizon cap so the
    # un-converged policy cannot chew up the whole wall-time on a single
    # timeout-length episode.
    eval_n_episodes = 2
    eval_horizon = 500

    mdp_cfg = {
        "kind": "Gymnasium",
        "name": "MountainCar-v0",
        "horizon": eval_horizon,  # used for training + smoke-eval
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
        from mushroom_rl.environments import Gymnasium
        from mushroom_rl.features import Features
        from mushroom_rl.features.tiles import Tiles
        from mushroom_rl.policy import EpsGreedy
        from mushroom_rl.rl_utils.parameters import Parameter

        logger = Logger(TrueOnlineSARSALambda.__name__, results_dir=None)
        logger.strong_line()
        logger.info(
            "Smoke reproduction of examples/mountain_car_sarsa.py "
            f"(n_steps={n_steps}, horizon={eval_horizon}, gamma=1.0)"
        )

        # --------------------------------------------------------------
        # MDP. Upstream uses horizon=1e4; smoke uses the short cap so
        # both training and evaluation are bounded.
        # --------------------------------------------------------------
        mdp = Gymnasium(
            name=mdp_cfg["name"],
            horizon=mdp_cfg["horizon"],
            gamma=mdp_cfg["gamma"],
            headless=mdp_cfg["headless"],
        )
        # Gymnasium wrapper stores _seed and consumes it on next reset.
        # Calling mdp.seed(seed) + seed_everything(seed) together covers
        # both the Gymnasium RNG and numpy/random/PYTHONHASHSEED.
        mdp.seed(seed)

        # --------------------------------------------------------------
        # Policy: greedy (epsilon=0) as in the upstream example.
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
        # Train. Use n_steps (not n_episodes) so we can hard-cap the
        # smoke wall-time; a single episode can be up to ``horizon``
        # transitions so ``n_steps=3_000`` at ``horizon=500`` is at most
        # ~6 episodes.
        # --------------------------------------------------------------
        core = Core(agent, mdp)
        t0 = time.perf_counter()
        core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True, render=False)
        train_wall_s = time.perf_counter() - t0
        print(
            f"[smoke] learn(n_steps={n_steps}) done in "
            f"{train_wall_s:.2f}s"
        )

        # --------------------------------------------------------------
        # Greedy evaluation. Tiny budget; cannot converge. This is a
        # sanity probe, not a performance measurement.
        # --------------------------------------------------------------
        # Re-seed Gymnasium for eval so it is reproducible per run.
        mdp.seed(seed + 1)
        t1 = time.perf_counter()
        eval_dataset = core.evaluate(
            n_episodes=eval_n_episodes,
            render=False,
            quiet=True,
        )
        eval_wall_s = time.perf_counter() - t1

        # Undiscounted return per evaluation episode. Use the Dataset
        # property `undiscounted_return` per the upstream example.
        J_eval = np.asarray(
            eval_dataset.undiscounted_return, dtype=np.float64
        )
        print(
            f"[smoke] evaluate(n_episodes={eval_n_episodes}) done in "
            f"{eval_wall_s:.2f}s; undiscounted_returns={J_eval.tolist()}"
        )

        # --------------------------------------------------------------
        # Non-degeneracy checks.
        # --------------------------------------------------------------
        w_stats = _weights_stats(agent)
        weights_nonzero = w_stats["nnz"] > 0
        no_nan_inf = (not w_stats["has_nan"]) and (not w_stats["has_inf"])
        returns_finite = bool(np.all(np.isfinite(J_eval)))
        non_degenerate = weights_nonzero and no_nan_inf and returns_finite

        print(
            f"[smoke] weight_stats: shape={w_stats['shape']}, "
            f"size={w_stats['size']}, nnz={w_stats['nnz']}, "
            f"min={w_stats['min']:.6f}, max={w_stats['max']:.6f}, "
            f"mean={w_stats['mean']:.6f}, l2_norm={w_stats['l2_norm']:.6f}, "
            f"has_nan={w_stats['has_nan']}, has_inf={w_stats['has_inf']}"
        )
        print(
            f"[smoke] J_mean={float(J_eval.mean()):.4f}, "
            f"J_min={float(J_eval.min()):.4f}, "
            f"J_max={float(J_eval.max()):.4f}"
        )
        print(f"[smoke] non_degenerate={non_degenerate}")

        if not non_degenerate:
            print("[smoke] WARNING: non-degeneracy check FAILED")

        # Close the viewer / env (harmless in headless mode).
        try:
            mdp.stop()
        except Exception as exc:
            print(f"[smoke] mdp.stop() raised (ignored): {exc!r}")

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

    total_wall_s = time.perf_counter() - t_total_0

    # --------------------------------------------------------------------
    # Persist run.json. Outside the stdout tee so the "[smoke] wrote ..."
    # line is written after the log file is closed.
    # --------------------------------------------------------------------
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
        task="mountain_car_sarsa",
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
