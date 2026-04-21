#!/usr/bin/env python
"""Phase IV-C: geometry-prioritized asynchronous DP experiments.

Runs geometry-prioritized asynchronous value iteration on the frozen
activation suite across three priority modes:

    residual_only      (lambda_geom=0, lambda_u=0, lambda_kl=0)
    geometry_gain_only (lambda_geom=1, lambda_u=0, lambda_kl=0)
    combined           (lambda_geom=1, lambda_u=1, lambda_kl=1)

Compares backup efficiency (n_sweeps to convergence, wall_clock) and
convergence rate against standard async VI on the same tasks.

Layout::

    <out_root>/phase4/advanced/geometry_priority_dp/<task_tag>/seed_<N>/<mode>/

CLI::

    python experiments/weighted_lse_dp/runners/run_phase4C_geometry_dp.py \\
        --config experiments/weighted_lse_dp/configs/phase4/geometry_priority_dp.json \\
        [--task TASK_TAG | all] [--seed N] [--out-root PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
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
from experiments.weighted_lse_dp.planners.geometry_priority_dp import (  # noqa: E402
    GeometryPriorityDP,
)
from experiments.weighted_lse_dp.runners.run_phase4_rl import (  # noqa: E402
    _load_activation_suite,
    _wrap_v3_schedule_for_betaschedule,
)
from experiments.weighted_lse_dp.tasks.phase4_operator_suite import (  # noqa: E402
    build_phase4_task,
)

__all__ = ["main", "run_single", "build_plan"]

PRIORITY_MODES: dict[str, dict[str, float]] = {
    "residual_only":      {"lambda_geom": 0.0, "lambda_u": 0.0, "lambda_kl": 0.0},
    "geometry_gain_only": {"lambda_geom": 1.0, "lambda_u": 0.0, "lambda_kl": 0.0},
    "combined":           {"lambda_geom": 1.0, "lambda_u": 1.0, "lambda_kl": 1.0},
}

_N_PILOT = 200
_SEEDS = [42, 123, 456]


def _build_v3_schedule(
    cfg: dict[str, Any],
    seed: int,
    gamma: float,
    run_dir: Path,
) -> tuple[dict[str, Any], str]:
    sign_family = int(get_task_sign(cfg.get("family", "unknown")))
    pilot = run_classical_pilot(cfg=cfg, seed=seed, n_episodes=_N_PILOT,
                                sign_family=sign_family)
    r_max = float(cfg.get("reward_bound", pilot.get("reward_bound", 1.0)))
    schedule_v3_path = run_dir / "schedule_v3.json"
    v3 = build_schedule_v3_from_pilot(
        pilot_data=pilot, r_max=r_max, gamma_base=gamma, gamma_eval=gamma,
        task_family=str(cfg.get("family", "unknown")), sign_family=sign_family,
        source_phase="phase4C_geometry_dp",
        notes="Phase IV-C geometry-priority DP schedule",
        output_path=schedule_v3_path,
    )
    return v3, str(schedule_v3_path)


def build_plan(
    config: dict[str, Any],
    *,
    task_filter: str | None,
    seed_filter: int | None,
    out_root: Path,
) -> list[dict[str, Any]]:
    suite_path = Path(config.get(
        "activation_suite_path",
        "experiments/weighted_lse_dp/configs/phase4/activation_suite_4a2.json",
    ))
    tasks = _load_activation_suite(suite_path)
    seeds = config.get("seeds", _SEEDS)
    modes = config.get("priority_modes", list(PRIORITY_MODES.keys()))

    plan = []
    for idx, task_info in enumerate(tasks):
        cfg = task_info.get("cfg", task_info)
        family = cfg.get("family", "unknown")
        tag = f"{family}_{idx}"
        if task_filter and task_filter != "all" and tag != task_filter:
            continue
        for seed in seeds:
            if seed_filter is not None and seed != seed_filter:
                continue
            for mode in modes:
                plan.append({"task_tag": tag, "family": family, "cfg": cfg,
                              "seed": seed, "mode": mode})
    return plan


def run_single(
    task_tag: str,
    family: str,
    cfg: dict[str, Any],
    seed: int,
    mode: str,
    *,
    out_root: Path,
    n_pilot: int = _N_PILOT,
) -> dict[str, Any]:
    from mushroom_rl.algorithms.value.dp import extract_mdp_arrays

    run_dir = out_root / "phase4" / "advanced" / "geometry_priority_dp" / task_tag / f"seed_{seed}" / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        seed_everything(seed)
        mdp_base, mdp_rl, resolved_cfg = build_phase4_task(cfg, seed=seed)
        gamma = float(resolved_cfg.get("gamma", cfg.get("gamma", 0.95)))
        horizon = int(resolved_cfg.get("horizon", cfg.get("horizon", 20)))

        p, r, _, _ = extract_mdp_arrays(mdp_base)
        # r shape: (S, A, S') — reduce to expected reward (S, A)
        r_bar = np.einsum("ijk,ijk->ij", p, r)

        v3, sched_path = _build_v3_schedule(cfg, seed, gamma, run_dir)
        lambdas = PRIORITY_MODES.get(mode, PRIORITY_MODES["combined"])

        planner = GeometryPriorityDP(
            p=p, r=r_bar, gamma=gamma, horizon=horizon,
            schedule_v3=v3, **lambdas, seed=seed,
        )
        result = planner.plan(tol=1e-6, max_sweeps=500)

        metrics = {
            "schema_version": "1.0.0", "phase": "phase4C",
            "task": task_tag, "algorithm": f"geometry_priority_dp_{mode}",
            "mode": mode, "seed": seed,
            "n_sweeps": result["n_sweeps"],
            "n_backups": result["n_backups"],
            "final_residual": result["final_residual"],
            "convergence_sweep_1e-2": result.get("convergence_sweep_1e-2"),
            "wall_clock_s": result["wall_clock_s"],
            "frac_high_activation_backups": result["frac_high_activation_backups"],
            **{k: result[k] for k in ("lambda_geom", "lambda_u", "lambda_kl")},
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        np.save(str(run_dir / "residual_history.npy"),
                np.array(result["residual_history"]))
        np.save(str(run_dir / "V.npy"), result["V"])
        run_json = {
            "schema_version": "1.0.0", "phase": "phase4C",
            "task": task_tag, "algorithm": f"geometry_priority_dp_{mode}",
            "mode": mode, "seed": seed,
            "schedule_v3_path": sched_path, "config": cfg,
            "geom_gain_per_stage": result["geom_gain_per_stage"],
            "u_ref_per_stage": result["u_ref_per_stage"],
        }
        (run_dir / "run.json").write_text(json.dumps(run_json, indent=2))
        return {"task_tag": task_tag, "mode": mode, "seed": seed,
                "status": "pass", "n_sweeps": result["n_sweeps"],
                "final_residual": result["final_residual"]}
    except Exception as exc:
        tb = traceback.format_exc()
        err = {"task_tag": task_tag, "mode": mode, "seed": seed,
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
        task_filter=args.task if args.task != "all" else None,
        seed_filter=args.seed,
        out_root=out_root,
    )
    if not plan:
        print("[phase4C_geometry_dp] No runs matched filters.")
        return 1

    print(f"[phase4C_geometry_dp] Executing {len(plan)} run(s)")
    if args.dry_run:
        for r in plan:
            print(f"  DRY: {r['task_tag']}/seed_{r['seed']}/{r['mode']}")
        return 0

    n_pass = n_fail = 0
    for entry in plan:
        result = run_single(
            entry["task_tag"], entry["family"], entry["cfg"], entry["seed"],
            entry["mode"], out_root=out_root,
        )
        if result.get("status") == "pass":
            n_pass += 1
            print(f"  {entry['task_tag']:30s} seed={entry['seed']:<6d} {entry['mode']:22s} "
                  f"PASS  sweeps={result.get('n_sweeps', '?')}")
        else:
            n_fail += 1
            print(f"  {entry['task_tag']:30s} seed={entry['seed']:<6d} {entry['mode']:22s} "
                  f"FAIL  {result.get('error', '')}")

    print(f"\n[phase4C_geometry_dp] Summary: {n_pass}/{n_pass+n_fail} runs passed")
    return 0 if n_fail == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("experiments/weighted_lse_dp/configs/phase4/geometry_priority_dp.json"))
    p.add_argument("--task", default="all")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=Path("results/weighted_lse_dp"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
