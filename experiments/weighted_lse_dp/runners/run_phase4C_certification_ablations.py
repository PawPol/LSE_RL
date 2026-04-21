#!/usr/bin/env python
"""Phase IV-C: trust-region, adaptive-headroom, wrong-sign, constant-u, and raw-unclipped ablations.

Runs certification-mechanism ablations to isolate the contribution of each
safety component.  Each ablation disables or modifies one mechanism while
keeping all others at their default settings.

Ablation types (--ablation values):

    trust_region_off         (tau_n → 0, c_t → 1.0, trust cap inactive)
    trust_region_tighter     (tau_n = 2000, trust cap releases slowly)
    adaptive_headroom_off    (alpha_min = alpha_max = 0.05, constant headroom)
    adaptive_headroom_aggressive  (alpha_max = 0.50, large headroom budget)
    wrong_sign               (flip sign_family: +1 → -1 and vice versa)
    constant_u               (u_ref_t = mean(u_design_t), constant across stages)
    raw_unclipped            (no trust cap, large u_max=0.10)

Layout::

    <out_root>/phase4/advanced/ablations/<ablation_type>/<task_tag>/seed_<N>/

CLI::

    python experiments/weighted_lse_dp/runners/run_phase4C_certification_ablations.py \\
        --config experiments/weighted_lse_dp/configs/phase4/certification_ablations.json \\
        [--ablation TYPE | all]  [--task TASK_TAG | all] [--seed N] \\
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

from experiments.weighted_lse_dp.common.callbacks import (  # noqa: E402
    RLEvaluator,
    SafeTransitionLogger,
)
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

# ---------------------------------------------------------------------------
# Ablation registry
# ---------------------------------------------------------------------------

ALL_ABLATIONS: tuple[str, ...] = (
    "trust_region_off",
    "trust_region_tighter",
    "adaptive_headroom_off",
    "adaptive_headroom_aggressive",
    "wrong_sign",
    "constant_u",
    "raw_unclipped",
)

_V3_DEFAULTS: dict[str, Any] = {
    "u_min": 0.002,
    "u_max": 0.020,
    "alpha_min": 0.05,
    "alpha_max": 0.20,
    "alpha_budget_max": 0.30,
    "tau_n": 200.0,
}

_ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "trust_region_off": {"tau_n": 1e-9},
    "trust_region_tighter": {"tau_n": 2000.0},
    "adaptive_headroom_off": {"alpha_min": 0.05, "alpha_max": 0.05},
    "adaptive_headroom_aggressive": {"alpha_min": 0.05, "alpha_max": 0.50, "alpha_budget_max": 0.60},
    "wrong_sign": {"_flip_sign": True},
    "constant_u": {"_constant_u": True},
    "raw_unclipped": {"tau_n": 1e-9, "u_max": 0.10, "alpha_max": 0.40, "alpha_budget_max": 0.50},
}

_TRAIN_STEPS = 20000
_CHECKPOINT_EVERY = 1000
_EVAL_EPISODES = 50
_EPSILON = 0.1
_LR = 0.1
_N_PILOT_EPISODES = 200


def _build_ablated_schedule(
    *,
    cfg: dict[str, Any],
    seed: int,
    n_pilot_episodes: int,
    gamma: float,
    ablation: str,
    run_dir: Path,
) -> Any:
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule

    overrides = dict(_ABLATION_OVERRIDES.get(ablation, {}))
    flip_sign = overrides.pop("_flip_sign", False)
    constant_u = overrides.pop("_constant_u", False)

    base_sign = int(get_task_sign(cfg.get("family", "unknown")))
    # wrong_sign isolation: pilot always runs with correct base_sign so that
    # p_align_t, informativeness_t, and alpha_t are not co-mutated.
    # Only beta_used_t is negated post-hoc (see below).
    pilot = run_classical_pilot(cfg=cfg, seed=seed, n_episodes=n_pilot_episodes,
                                sign_family=base_sign)
    r_max = float(cfg.get("reward_bound", pilot.get("reward_bound", 1.0)))
    gamma_base = float(gamma)

    v3_kwargs = {**_V3_DEFAULTS, **overrides}
    schedule_v3_path = run_dir / f"schedule_v3_{ablation}.json"
    v3 = build_schedule_v3_from_pilot(
        pilot_data=pilot,
        r_max=r_max,
        gamma_base=gamma_base,
        gamma_eval=gamma_base,
        task_family=str(cfg.get("family", "unknown")),
        sign_family=base_sign,
        source_phase=f"phase4C_ablation_{ablation}",
        notes=f"Phase IV-C ablation: {ablation}",
        output_path=schedule_v3_path,
        **{k: v for k, v in v3_kwargs.items()},
    )

    if flip_sign:
        # Negate beta_used_t post-hoc: only the operator sign is flipped,
        # all diagnostics (p_align_t, informativeness_t, alpha_t) remain intact.
        v3["beta_used_t"] = [-b for b in v3["beta_used_t"]]
        v3["sign_family"] = -base_sign
        v3["notes"] = v3.get("notes", "") + " [wrong_sign: beta negated post-hoc, pilot unchanged]"
        with open(schedule_v3_path, "w") as f:
            json.dump(v3, f, indent=2)

    if constant_u:
        beta_arr = np.asarray(v3["beta_used_t"], dtype=np.float64)
        xi_arr = np.asarray(v3.get("xi_ref_t", np.ones(len(beta_arr))), dtype=np.float64)
        u_mean = float(np.mean(np.abs(beta_arr * xi_arr)))
        xi_safe = np.where(xi_arr > 1e-9, xi_arr, 1.0)
        beta_const = u_mean / xi_safe
        # Check for cap violations before wrapping
        beta_cap_arr = np.asarray(v3.get("beta_cap_t", np.full(len(beta_arr), np.inf)))
        clip_mask = np.abs(beta_const) > np.abs(beta_cap_arr)
        constant_u_clip_count = int(np.sum(clip_mask))
        if constant_u_clip_count > 0:
            import warnings
            warnings.warn(
                f"constant_u ablation: {constant_u_clip_count}/{len(beta_const)} stages "
                f"will be clipped by beta_cap_t after wrapping. "
                f"The resulting schedule is NOT constant-u at those stages.",
                stacklevel=2,
            )
        v3["beta_used_t"] = beta_const.tolist()
        v3["notes"] = (f"constant_u ablation: u_const={u_mean:.6f}, "
                       f"constant_u_clip_count={constant_u_clip_count}")
        with open(schedule_v3_path, "w") as f:
            json.dump(v3, f, indent=2)

    wrapped = _wrap_v3_schedule_for_betaschedule(v3, gamma=gamma_base)
    return BetaSchedule(wrapped), str(schedule_v3_path)


def build_plan(
    config: dict[str, Any],
    *,
    ablation_filter: str | None,
    task_filter: str | None,
    seed_filter: int | None,
    out_root: Path,
) -> list[dict[str, Any]]:
    suite_path = Path(config.get(
        "activation_suite_path",
        "experiments/weighted_lse_dp/configs/phase4/activation_suite_4a2.json",
    ))
    tasks = _load_activation_suite(suite_path)
    ablation_list = [ablation_filter] if ablation_filter else list(ALL_ABLATIONS)
    seeds = config.get("seeds", [42, 123, 456])

    plan = []
    for ablation in ablation_list:
        for idx, task_info in enumerate(tasks):
            cfg = task_info.get("cfg", task_info)
            family = cfg.get("family", "unknown")
            task_tag = f"{family}_{idx}"
            if task_filter and task_tag != task_filter:
                continue
            for seed in seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue
                plan.append({
                    "ablation": ablation,
                    "task_tag": task_tag,
                    "family": family,
                    "seed": seed,
                    "cfg": cfg,
                })
    return plan


def _eval_disc_return(
    core: Any, mdp_eval: Any, n_episodes: int, gamma: float
) -> float:
    """Evaluate discounted return by running core.evaluate and summing rewards."""
    import copy as _copy
    eval_core = _copy.deepcopy(core)
    eval_core.env = mdp_eval
    dataset = eval_core.evaluate(n_episodes=n_episodes, quiet=True)
    # Dataset: iterate episodes and compute discounted return
    ep_returns = []
    ep_reward = 0.0
    ep_disc = 1.0
    for sample in dataset:
        # sample is (state, action, reward, next_state, absorbing, last)
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


def run_single(
    ablation: str,
    task_tag: str,
    cfg: dict[str, Any],
    seed: int,
    *,
    train_steps: int = _TRAIN_STEPS,
    checkpoint_every: int = _CHECKPOINT_EVERY,
    eval_episodes: int = _EVAL_EPISODES,
    epsilon: float = _EPSILON,
    learning_rate: float = _LR,
    n_pilot_episodes: int = _N_PILOT_EPISODES,
    out_root: Path,
) -> dict[str, Any]:
    import copy as _copy
    from mushroom_rl.algorithms.value.td import SafeQLearning
    from mushroom_rl.core import Core
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import Parameter

    run_dir = (
        out_root / "phase4" / "advanced" / "ablations"
        / ablation / task_tag / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        seed_everything(seed)
        _, mdp_rl, _ = build_phase4_task(cfg, seed=seed)
        mdp_eval = _copy.deepcopy(mdp_rl)
        gamma = float(cfg.get("gamma", 0.95))
        horizon = int(cfg.get("horizon", 20))
        n_base = mdp_rl.info.observation_space.n // horizon

        schedule, sched_path = _build_ablated_schedule(
            cfg=cfg, seed=seed, n_pilot_episodes=n_pilot_episodes,
            gamma=gamma, ablation=ablation, run_dir=run_dir,
        )

        pi = EpsGreedy(epsilon=Parameter(value=epsilon))
        agent = SafeQLearning(mdp_rl.info, pi, schedule, n_base, Parameter(value=learning_rate))
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
        final_mean = float(np.mean(eval_returns[-3:])) if len(eval_returns) >= 3 else float(eval_returns[-1])

        metrics = {
            "schema_version": "1.0.0", "phase": "phase4C", "task": task_tag,
            "algorithm": f"safe_q_stagewise_ablation_{ablation}", "ablation": ablation,
            "seed": seed, "train_steps": train_steps, "n_transitions": train_steps,
            "mean_return": float(np.mean(eval_returns)),
            "final_disc_return_mean": final_mean,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        run_json = {
            "schema_version": "1.0.0", "phase": "phase4C", "task": task_tag,
            "algorithm": f"safe_q_stagewise_ablation_{ablation}", "ablation": ablation,
            "seed": seed, "elapsed_s": elapsed, "schedule_v3_path": sched_path,
            "config": cfg, "final_mean_return": final_mean,
            "leakage_limitation": "pilot and train share seed; no cross-fitting (spec §4.5)",
        }
        (run_dir / "run.json").write_text(json.dumps(run_json, indent=2))
        return {"ablation": ablation, "task_tag": task_tag, "seed": seed,
                "status": "pass", "final_mean_return": final_mean, "elapsed_s": elapsed}

    except Exception as exc:
        tb = traceback.format_exc()
        err = {"ablation": ablation, "task_tag": task_tag, "seed": seed,
               "status": "fail", "error": str(exc), "traceback": tb}
        (run_dir / "error.json").write_text(json.dumps(err, indent=2))
        return err


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    config.setdefault("activation_suite_path",
                      "experiments/weighted_lse_dp/configs/phase4/activation_suite_4a2.json")
    config.setdefault("seeds", [42, 123, 456])

    out_root = args.out_root
    plan = build_plan(
        config,
        ablation_filter=args.ablation if args.ablation != "all" else None,
        task_filter=args.task if args.task != "all" else None,
        seed_filter=args.seed,
        out_root=out_root,
    )

    if not plan:
        print("[phase4C_ablations] No runs matched filters.")
        return 1

    print(f"[phase4C_ablations] Executing {len(plan)} run(s)")
    if args.dry_run:
        for r in plan:
            print(f"  DRY: {r['ablation']}/{r['task_tag']}/seed_{r['seed']}")
        return 0

    train_steps = int(config.get("train_steps", _TRAIN_STEPS))
    n_pilot = int(config.get("n_pilot_episodes", _N_PILOT_EPISODES))

    n_pass = n_fail = 0
    for entry in plan:
        result = run_single(
            entry["ablation"], entry["task_tag"], entry["cfg"], entry["seed"],
            train_steps=train_steps, n_pilot_episodes=n_pilot, out_root=out_root,
        )
        if result.get("status") == "pass":
            n_pass += 1
            print(f"  {entry['ablation']:30s} {entry['task_tag']:25s} "
                  f"seed={entry['seed']:<6d} PASS  "
                  f"mean={result.get('final_mean_return', 'N/A'):.4f}")
        else:
            n_fail += 1
            print(f"  {entry['ablation']:30s} {entry['task_tag']:25s} "
                  f"seed={entry['seed']:<6d} FAIL  {result.get('error', '')}")

    print(f"\n[phase4C_ablations] Summary: {n_pass}/{n_pass+n_fail} runs passed")
    return 0 if n_fail == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("experiments/weighted_lse_dp/configs/phase4/certification_ablations.json"))
    p.add_argument("--ablation", default="all",
                   choices=list(ALL_ABLATIONS) + ["all"])
    p.add_argument("--task", default="all")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=Path("results/weighted_lse_dp"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
