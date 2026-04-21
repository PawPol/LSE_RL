#!/usr/bin/env python
"""Phase IV-C: advanced estimator-stabilized RL.

Runs SafeDoubleQ, SafeTargetQ, and SafeTargetExpectedSARSA agents on the
frozen activation suite.  Records estimator-specific diagnostics (double-gap,
q_target_gap, target_sync_step) alongside standard learning curves and
outcome metrics.

Algorithms (``--algorithm`` values):

    safe_double_q                  (two Q-tables, evaluation-side bootstrap)
    safe_target_q                  (frozen target network, hard sync)
    safe_target_q_polyak           (frozen target network, Polyak averaging)
    safe_target_expected_sarsa     (target network + expected SARSA bootstrap)

Layout::

    <out_root>/phase4/advanced/<algorithm>/<task_tag>/seed_<N>/

CLI::

    python experiments/weighted_lse_dp/runners/run_phase4C_advanced_rl.py \\
        --config experiments/weighted_lse_dp/configs/phase4/advanced_estimators.json \\
        [--algorithm ALG | all] [--task TASK_TAG | all] [--seed N] \\
        [--out-root PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from experiments.weighted_lse_dp.common.calibration import get_task_sign  # noqa: E402
from experiments.weighted_lse_dp.common.seeds import seed_everything  # noqa: E402
from experiments.weighted_lse_dp.geometry.phase4_calibration_v3 import (  # noqa: E402
    build_schedule_v3_from_pilot,
)
from experiments.weighted_lse_dp.geometry.task_activation_search import (  # noqa: E402
    run_classical_pilot,
)
from experiments.weighted_lse_dp.runners.run_phase4_rl import (  # noqa: E402
    _load_activation_suite,
    _wrap_v3_schedule_for_betaschedule,
)
from experiments.weighted_lse_dp.tasks.phase4_operator_suite import (  # noqa: E402
    build_phase4_task,
)

__all__ = ["main", "run_single", "build_plan"]

ALL_ALGORITHMS: tuple[str, ...] = (
    "safe_double_q",
    "safe_target_q",
    "safe_target_q_polyak",
    "safe_target_expected_sarsa",
)

_TRAIN_STEPS = 20000
_CHECKPOINT_EVERY = 1000
_EVAL_EPISODES = 50
_EPSILON = 0.1
_LR = 0.1
_N_PILOT = 200
_SYNC_EVERY = 200
_POLYAK_TAU = 0.05
_SEEDS = [42, 123, 456]


def _build_schedule(
    *,
    cfg: dict[str, Any],
    seed: int,
    gamma: float,
    run_dir: Path,
    n_pilot: int,
) -> tuple[Any, str]:
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule

    sign_family = int(get_task_sign(cfg.get("family", "unknown")))
    pilot = run_classical_pilot(cfg=cfg, seed=seed, n_episodes=n_pilot,
                                sign_family=sign_family)
    r_max = float(cfg.get("reward_bound", pilot.get("reward_bound", 1.0)))
    schedule_v3_path = run_dir / "schedule_v3.json"
    v3 = build_schedule_v3_from_pilot(
        pilot_data=pilot, r_max=r_max, gamma_base=gamma, gamma_eval=gamma,
        task_family=str(cfg.get("family", "unknown")), sign_family=sign_family,
        source_phase="phase4C_advanced_rl",
        notes="Phase IV-C advanced estimator schedule",
        output_path=schedule_v3_path,
    )
    wrapped = _wrap_v3_schedule_for_betaschedule(v3, gamma=gamma)
    return BetaSchedule(wrapped), str(schedule_v3_path)


def _make_agent(
    algorithm: str,
    n_states: int,
    n_actions: int,
    schedule: Any,
    gamma: float,
    seed: int,
) -> Any:
    from lse_rl.algorithms import (
        SafeDoubleQLearning,
        SafeTargetExpectedSARSA,
        SafeTargetQLearning,
    )

    if algorithm == "safe_double_q":
        return SafeDoubleQLearning(
            n_states=n_states, n_actions=n_actions, schedule=schedule,
            learning_rate=_LR, gamma=gamma, seed=seed,
        )
    elif algorithm == "safe_target_q":
        return SafeTargetQLearning(
            n_states=n_states, n_actions=n_actions, schedule=schedule,
            learning_rate=_LR, gamma=gamma, sync_every=_SYNC_EVERY,
            polyak_tau=0.0, seed=seed,
        )
    elif algorithm == "safe_target_q_polyak":
        return SafeTargetQLearning(
            n_states=n_states, n_actions=n_actions, schedule=schedule,
            learning_rate=_LR, gamma=gamma, sync_every=_SYNC_EVERY,
            polyak_tau=_POLYAK_TAU, seed=seed,
        )
    elif algorithm == "safe_target_expected_sarsa":
        return SafeTargetExpectedSARSA(
            n_states=n_states, n_actions=n_actions, schedule=schedule,
            learning_rate=_LR, gamma=gamma, sync_every=_SYNC_EVERY,
            polyak_tau=0.0, epsilon=_EPSILON, seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")


def _greedy_action(agent: Any, state: int, stage: int) -> int:
    q = agent.get_Q(state, stage)
    max_q = float(np.max(q))
    ties = np.flatnonzero(q == max_q)
    return int(np.random.choice(ties))


def _epsilon_greedy_action(agent: Any, state: int, stage: int,
                            n_actions: int, epsilon: float, rng: Any) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return _greedy_action(agent, state, stage)


def _run_episode_train(
    mdp: Any,
    agent: Any,
    n_base: int,
    horizon: int,
    n_actions: int,
    epsilon: float,
    rng: Any,
    global_step_ref: list[int],
    algorithm: str,
) -> tuple[float, list[dict[str, Any]]]:
    """Run one training episode. Returns (episode_return, logs)."""
    state_aug, _ = mdp.reset()
    aug = int(np.asarray(state_aug).flat[0])
    logs = []
    ep_return = 0.0
    disc = 1.0

    for _ in range(horizon):
        stage = aug // n_base
        state = aug % n_base
        stage = min(stage, agent.T - 1)

        action = _epsilon_greedy_action(agent, state, stage, n_actions, epsilon, rng)
        next_aug_arr, reward, absorbing, info = mdp.step(np.array([action]))
        next_aug = int(np.asarray(next_aug_arr).flat[0])
        r = float(reward)
        done = bool(absorbing)
        ep_return += disc * r
        disc *= mdp.info.gamma

        next_stage = next_aug // n_base
        next_state = next_aug % n_base
        next_stage = min(next_stage, agent.T - 1)

        gs = global_step_ref[0]
        global_step_ref[0] += 1

        if algorithm in ("safe_target_q", "safe_target_q_polyak",
                         "safe_target_expected_sarsa"):
            log = agent.update(
                state=state, action=action, reward=r,
                next_state=next_state, absorbing=done,
                stage=stage, global_step=gs,
            )
        else:
            log = agent.update(
                state=state, action=action, reward=r,
                next_state=next_state, absorbing=done, stage=stage,
            )
        logs.append(log)
        aug = next_aug
        if done:
            break

    return ep_return, logs


def _eval_agent(mdp: Any, agent: Any, n_base: int, horizon: int,
                n_episodes: int, gamma: float) -> float:
    returns = []
    for _ in range(n_episodes):
        state_aug, _ = mdp.reset()
        aug = int(np.asarray(state_aug).flat[0])
        ep_return = 0.0
        disc = 1.0
        for _ in range(horizon):
            stage = aug // n_base
            state = aug % n_base
            stage = min(stage, agent.T - 1)
            action = _greedy_action(agent, state, stage)
            next_aug_arr, reward, absorbing, _ = mdp.step(np.array([action]))
            ep_return += disc * float(reward)
            disc *= gamma
            aug = int(np.asarray(next_aug_arr).flat[0])
            if absorbing:
                break
        returns.append(ep_return)
    return float(np.mean(returns)) if returns else 0.0


def build_plan(
    config: dict[str, Any],
    *,
    algorithm_filter: str | None,
    task_filter: str | None,
    seed_filter: int | None,
    out_root: Path,
) -> list[dict[str, Any]]:
    suite_path = Path(config.get(
        "activation_suite_path",
        "experiments/weighted_lse_dp/configs/phase4/activation_suite_4a2.json",
    ))
    tasks = _load_activation_suite(suite_path)
    algorithms = [algorithm_filter] if algorithm_filter else list(ALL_ALGORITHMS)
    seeds = config.get("seeds", _SEEDS)

    plan = []
    for algo in algorithms:
        for idx, task_info in enumerate(tasks):
            cfg = task_info.get("cfg", task_info)
            family = cfg.get("family", "unknown")
            tag = f"{family}_{idx}"
            if task_filter and task_filter != "all" and tag != task_filter:
                continue
            for seed in seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue
                plan.append({"algorithm": algo, "task_tag": tag,
                              "family": family, "cfg": cfg, "seed": seed})
    return plan


def run_single(
    algorithm: str,
    task_tag: str,
    cfg: dict[str, Any],
    seed: int,
    *,
    train_steps: int = _TRAIN_STEPS,
    checkpoint_every: int = _CHECKPOINT_EVERY,
    eval_episodes: int = _EVAL_EPISODES,
    n_pilot: int = _N_PILOT,
    out_root: Path,
) -> dict[str, Any]:
    import copy as _copy

    run_dir = out_root / "phase4" / "advanced" / algorithm / task_tag / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        _, mdp_rl, resolved_cfg = build_phase4_task(cfg, seed=seed)
        mdp_eval = _copy.deepcopy(mdp_rl)
        gamma = float(resolved_cfg.get("gamma", cfg.get("gamma", 0.95)))
        horizon = int(resolved_cfg.get("horizon", cfg.get("horizon", 20)))
        n_actions = mdp_rl.info.action_space.n
        n_base = mdp_rl.info.observation_space.n // horizon

        schedule, sched_path = _build_schedule(
            cfg=cfg, seed=seed, gamma=gamma, run_dir=run_dir, n_pilot=n_pilot,
        )
        agent = _make_agent(algorithm, n_base, n_actions, schedule, gamma, seed)

        eval_returns: list[float] = []
        steps_done = 0
        global_step_ref = [0]
        t0 = time.perf_counter()
        all_logs: list[dict[str, Any]] = []

        while steps_done < train_steps:
            target = steps_done + checkpoint_every
            while steps_done < target and steps_done < train_steps:
                ep_return, ep_logs = _run_episode_train(
                    mdp=mdp_rl, agent=agent, n_base=n_base, horizon=horizon,
                    n_actions=n_actions, epsilon=_EPSILON, rng=rng,
                    global_step_ref=global_step_ref, algorithm=algorithm,
                )
                steps_done += len(ep_logs)
                all_logs.extend(ep_logs)
            disc_ret = _eval_agent(mdp_eval, agent, n_base, horizon,
                                   eval_episodes, gamma)
            eval_returns.append(disc_ret)

        elapsed = time.perf_counter() - t0
        final_mean = (float(np.mean(eval_returns[-3:]))
                      if len(eval_returns) >= 3 else float(eval_returns[-1]))

        # Aggregate estimator-specific diagnostics
        diag_keys = {
            "safe_double_q": ["double_gap", "natural_shift_double", "beta_used"],
            "safe_target_q": ["q_target_gap", "beta_used"],
            "safe_target_q_polyak": ["q_target_gap", "beta_used"],
            "safe_target_expected_sarsa": ["q_target_gap", "beta_used"],
        }
        diag: dict[str, float] = {}
        for key in diag_keys.get(algorithm, ["beta_used"]):
            vals = [log[key] for log in all_logs if key in log]
            if vals:
                diag[f"mean_{key}"] = float(np.mean(vals))

        metrics = {
            "schema_version": "1.0.0", "phase": "phase4C",
            "task": task_tag, "algorithm": algorithm, "seed": seed,
            "train_steps": steps_done, "n_transitions": steps_done,
            "mean_return": float(np.mean(eval_returns)),
            "final_disc_return_mean": final_mean,
            **diag,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        run_json = {
            "schema_version": "1.0.0", "phase": "phase4C",
            "task": task_tag, "algorithm": algorithm, "seed": seed,
            "elapsed_s": elapsed, "schedule_v3_path": sched_path, "config": cfg,
            "final_mean_return": final_mean,
        }
        (run_dir / "run.json").write_text(json.dumps(run_json, indent=2))
        np.save(str(run_dir / "eval_returns.npy"), np.array(eval_returns))
        return {"algorithm": algorithm, "task_tag": task_tag, "seed": seed,
                "status": "pass", "final_mean_return": final_mean, "elapsed_s": elapsed}

    except Exception as exc:
        tb = traceback.format_exc()
        err = {"algorithm": algorithm, "task_tag": task_tag, "seed": seed,
               "status": "fail", "error": str(exc), "traceback": tb}
        (run_dir / "error.json").write_text(json.dumps(err, indent=2))
        return err


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    config.setdefault("activation_suite_path",
                      "experiments/weighted_lse_dp/configs/phase4/activation_suite_4a2.json")
    config.setdefault("seeds", _SEEDS)

    out_root = args.out_root
    plan = build_plan(
        config,
        algorithm_filter=args.algorithm if args.algorithm != "all" else None,
        task_filter=args.task if args.task != "all" else None,
        seed_filter=args.seed,
        out_root=out_root,
    )
    if not plan:
        print("[phase4C_advanced_rl] No runs matched filters.")
        return 1

    print(f"[phase4C_advanced_rl] Executing {len(plan)} run(s)")
    if args.dry_run:
        for r in plan:
            print(f"  DRY: {r['algorithm']}/{r['task_tag']}/seed_{r['seed']}")
        return 0

    train_steps = int(config.get("train_steps", _TRAIN_STEPS))
    n_pilot = int(config.get("n_pilot_episodes", _N_PILOT))

    n_pass = n_fail = 0
    for entry in plan:
        result = run_single(
            entry["algorithm"], entry["task_tag"], entry["cfg"], entry["seed"],
            train_steps=train_steps, n_pilot=n_pilot, out_root=out_root,
        )
        if result.get("status") == "pass":
            n_pass += 1
            print(f"  {entry['algorithm']:35s} {entry['task_tag']:25s} "
                  f"seed={entry['seed']:<6d} PASS  "
                  f"mean={result.get('final_mean_return', 'N/A'):.4f}")
        else:
            n_fail += 1
            print(f"  {entry['algorithm']:35s} {entry['task_tag']:25s} "
                  f"seed={entry['seed']:<6d} FAIL  {result.get('error', '')}")

    print(f"\n[phase4C_advanced_rl] Summary: {n_pass}/{n_pass+n_fail} runs passed")
    return 0 if n_fail == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("experiments/weighted_lse_dp/configs/phase4/advanced_estimators.json"))
    p.add_argument("--algorithm", default="all",
                   choices=list(ALL_ALGORITHMS) + ["all"])
    p.add_argument("--task", default="all")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=Path("results/weighted_lse_dp"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
