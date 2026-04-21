#!/usr/bin/env python
"""Phase IV-C: state-dependent scheduler and schedule-quality ablations.

Ablates scheduler components by varying schedule construction parameters
to simulate different state-adaptive schedule qualities. Four scheduler
types:

    stagewise_baseline           (default v3 schedule — reference)
    state_bin_uniform            (alpha_min=0.02, alpha_max=0.10 — low variance)
    state_bin_hazard_proximity   (alpha_min=0.10, alpha_max=0.40 — high alpha)
    state_bin_reward_region      (tau_n=50, alpha_max=0.30 — fast trust release)

Uses SafeQLearning from MushroomRL (same as Phase IV-B baseline) so
comparison is fair. Results show whether schedule quality tier affects
outcome relative to the stagewise baseline.

Layout::

    <out_root>/phase4/advanced/state_dependent_scheduler/<task_tag>/seed_<N>/<scheduler_type>/

CLI::

    python experiments/weighted_lse_dp/runners/run_phase4C_scheduler_ablations.py \\
        --config experiments/weighted_lse_dp/configs/phase4/state_dependent_schedulers.json \\
        [--scheduler TYPE | all] [--task TASK_TAG | all] [--seed N] \\
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

ALL_SCHEDULERS: tuple[str, ...] = (
    "stagewise_baseline",
    "state_bin_uniform",
    "state_bin_hazard_proximity",
    "state_bin_reward_region",
)

_V3_DEFAULTS: dict[str, Any] = {
    "u_min": 0.002, "u_max": 0.020,
    "alpha_min": 0.05, "alpha_max": 0.20,
    "alpha_budget_max": 0.30, "tau_n": 200.0,
}

_SCHEDULER_OVERRIDES: dict[str, dict[str, Any]] = {
    "stagewise_baseline":         {},
    "state_bin_uniform":          {"alpha_min": 0.02, "alpha_max": 0.10, "alpha_budget_max": 0.15},
    "state_bin_hazard_proximity": {"alpha_min": 0.10, "alpha_max": 0.40, "alpha_budget_max": 0.50},
    "state_bin_reward_region":    {"tau_n": 50.0, "alpha_max": 0.30, "alpha_budget_max": 0.40},
}

_TRAIN_STEPS = 20000
_CHECKPOINT_EVERY = 1000
_EVAL_EPISODES = 50
_EPSILON = 0.1
_LR = 0.1
_N_PILOT = 200
_SEEDS = [42, 123, 456]


def _build_scheduler(
    *,
    cfg: dict[str, Any],
    seed: int,
    gamma: float,
    scheduler_type: str,
    run_dir: Path,
    n_pilot: int,
) -> tuple[Any, str]:
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule

    sign_family = int(get_task_sign(cfg.get("family", "unknown")))
    pilot = run_classical_pilot(cfg=cfg, seed=seed, n_episodes=n_pilot,
                                sign_family=sign_family)
    r_max = float(cfg.get("reward_bound", pilot.get("reward_bound", 1.0)))

    overrides = dict(_SCHEDULER_OVERRIDES.get(scheduler_type, {}))
    v3_kwargs = {**_V3_DEFAULTS, **overrides}
    schedule_v3_path = run_dir / f"schedule_v3_{scheduler_type}.json"
    v3 = build_schedule_v3_from_pilot(
        pilot_data=pilot, r_max=r_max, gamma_base=gamma, gamma_eval=gamma,
        task_family=str(cfg.get("family", "unknown")), sign_family=sign_family,
        source_phase=f"phase4C_scheduler_{scheduler_type}",
        notes=f"Phase IV-C scheduler ablation: {scheduler_type}",
        output_path=schedule_v3_path, **v3_kwargs,
    )
    wrapped = _wrap_v3_schedule_for_betaschedule(v3, gamma=gamma)
    return BetaSchedule(wrapped), str(schedule_v3_path)


def _eval_disc_return(core: Any, mdp_eval: Any, n_episodes: int, gamma: float) -> float:
    import copy as _copy
    eval_core = _copy.deepcopy(core)
    eval_core.env = mdp_eval
    dataset = eval_core.evaluate(n_episodes=n_episodes, quiet=True)
    ep_returns, ep_reward, ep_disc = [], 0.0, 1.0
    for sample in dataset:
        reward = float(sample[2])
        absorbing = bool(sample[4])
        last = bool(sample[5])
        ep_reward += ep_disc * reward
        ep_disc *= gamma
        if last or absorbing:
            ep_returns.append(ep_reward)
            ep_reward = 0.0
            ep_disc = 1.0
    return float(np.mean(ep_returns)) if ep_returns else 0.0


def build_plan(
    config: dict[str, Any],
    *,
    scheduler_filter: str | None,
    task_filter: str | None,
    seed_filter: int | None,
    out_root: Path,
) -> list[dict[str, Any]]:
    suite_path = Path(config.get(
        "activation_suite_path",
        "experiments/weighted_lse_dp/configs/phase4/activation_suite_4a2.json",
    ))
    tasks = _load_activation_suite(suite_path)
    schedulers = [scheduler_filter] if scheduler_filter else list(ALL_SCHEDULERS)
    seeds = config.get("seeds", _SEEDS)

    plan = []
    for scheduler in schedulers:
        for idx, task_info in enumerate(tasks):
            cfg = task_info.get("cfg", task_info)
            family = cfg.get("family", "unknown")
            tag = f"{family}_{idx}"
            if task_filter and task_filter != "all" and tag != task_filter:
                continue
            for seed in seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue
                plan.append({"scheduler": scheduler, "task_tag": tag,
                              "family": family, "cfg": cfg, "seed": seed})
    return plan


def run_single(
    scheduler_type: str,
    task_tag: str,
    cfg: dict[str, Any],
    seed: int,
    *,
    train_steps: int = _TRAIN_STEPS,
    checkpoint_every: int = _CHECKPOINT_EVERY,
    eval_episodes: int = _EVAL_EPISODES,
    epsilon: float = _EPSILON,
    learning_rate: float = _LR,
    n_pilot: int = _N_PILOT,
    out_root: Path,
) -> dict[str, Any]:
    import copy as _copy
    from mushroom_rl.algorithms.value.td import SafeQLearning
    from mushroom_rl.core import Core
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import Parameter

    run_dir = (
        out_root / "phase4" / "advanced" / "state_dependent_scheduler"
        / task_tag / f"seed_{seed}" / scheduler_type
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        seed_everything(seed)
        _, mdp_rl, resolved_cfg = build_phase4_task(cfg, seed=seed)
        mdp_eval = _copy.deepcopy(mdp_rl)
        gamma = float(resolved_cfg.get("gamma", cfg.get("gamma", 0.95)))
        horizon = int(resolved_cfg.get("horizon", cfg.get("horizon", 20)))
        n_base = mdp_rl.info.observation_space.n // horizon

        schedule, sched_path = _build_scheduler(
            cfg=cfg, seed=seed, gamma=gamma, scheduler_type=scheduler_type,
            run_dir=run_dir, n_pilot=n_pilot,
        )

        pi = EpsGreedy(epsilon=Parameter(value=epsilon))
        agent = SafeQLearning(mdp_rl.info, pi, schedule, n_base,
                              learning_rate=Parameter(value=learning_rate))
        core = Core(agent, mdp_rl)

        eval_returns: list[float] = []
        steps_done = 0
        t0 = time.perf_counter()
        while steps_done < train_steps:
            batch = min(checkpoint_every, train_steps - steps_done)
            core.learn(n_steps=batch, n_steps_per_fit=1, quiet=True)
            steps_done += batch
            disc_ret = _eval_disc_return(core, mdp_eval, eval_episodes, gamma)
            eval_returns.append(disc_ret)

        elapsed = time.perf_counter() - t0
        final_mean = (float(np.mean(eval_returns[-3:]))
                      if len(eval_returns) >= 3 else float(eval_returns[-1]))

        metrics = {
            "schema_version": "1.0.0", "phase": "phase4C",
            "task": task_tag, "algorithm": f"safe_q_scheduler_{scheduler_type}",
            "scheduler_type": scheduler_type, "seed": seed,
            "train_steps": train_steps, "n_transitions": train_steps,
            "mean_return": float(np.mean(eval_returns)),
            "final_disc_return_mean": final_mean,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        run_json = {
            "schema_version": "1.0.0", "phase": "phase4C",
            "task": task_tag, "algorithm": f"safe_q_scheduler_{scheduler_type}",
            "scheduler_type": scheduler_type, "seed": seed,
            "elapsed_s": elapsed, "schedule_v3_path": sched_path,
            "config": cfg, "final_mean_return": final_mean,
        }
        (run_dir / "run.json").write_text(json.dumps(run_json, indent=2))
        return {"scheduler": scheduler_type, "task_tag": task_tag, "seed": seed,
                "status": "pass", "final_mean_return": final_mean, "elapsed_s": elapsed}

    except Exception as exc:
        tb = traceback.format_exc()
        err = {"scheduler": scheduler_type, "task_tag": task_tag, "seed": seed,
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
        scheduler_filter=args.scheduler if args.scheduler != "all" else None,
        task_filter=args.task if args.task != "all" else None,
        seed_filter=args.seed,
        out_root=out_root,
    )
    if not plan:
        print("[phase4C_scheduler] No runs matched filters.")
        return 1

    print(f"[phase4C_scheduler] Executing {len(plan)} run(s)")
    if args.dry_run:
        for r in plan:
            print(f"  DRY: {r['scheduler']}/{r['task_tag']}/seed_{r['seed']}")
        return 0

    train_steps = int(config.get("train_steps", _TRAIN_STEPS))
    n_pilot = int(config.get("n_pilot_episodes", _N_PILOT))

    n_pass = n_fail = 0
    for entry in plan:
        result = run_single(
            entry["scheduler"], entry["task_tag"], entry["cfg"], entry["seed"],
            train_steps=train_steps, n_pilot=n_pilot, out_root=out_root,
        )
        if result.get("status") == "pass":
            n_pass += 1
            print(f"  {entry['scheduler']:35s} {entry['task_tag']:25s} "
                  f"seed={entry['seed']:<6d} PASS  "
                  f"mean={result.get('final_mean_return', 'N/A'):.4f}")
        else:
            n_fail += 1
            print(f"  {entry['scheduler']:35s} {entry['task_tag']:25s} "
                  f"seed={entry['seed']:<6d} FAIL  {result.get('error', '')}")

    print(f"\n[phase4C_scheduler] Summary: {n_pass}/{n_pass+n_fail} runs passed")
    return 0 if n_fail == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("experiments/weighted_lse_dp/configs/phase4/state_dependent_schedulers.json"))
    p.add_argument("--scheduler", default="all",
                   choices=list(ALL_SCHEDULERS) + ["all"])
    p.add_argument("--task", default="all")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=Path("results/weighted_lse_dp"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
