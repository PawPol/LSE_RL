#!/usr/bin/env python
"""Phase IV-B: DP translation experiments on the frozen activation suite.

Runs classical and safe DP planners on the tasks identified by Phase IV-A (or
IV-A2) as having confirmed operator activation.  Records sweep-level
convergence curves, safe-specific diagnostics (beta/rho/eff-discount), and
calibration stats computed analytically from the exact DP Q/V tables.

Supported algorithms (``--algorithm`` values):

    classical_vi
    classical_async_vi
    safe_vi              (beta schedule from v3 calibration)
    safe_async_vi        (beta schedule from v3 calibration)
    safe_vi_zero         (BetaSchedule.zeros -- classical collapse)
    safe_async_vi_zero   (BetaSchedule.zeros -- classical collapse)

Layout::

    <out_root>/phase4/<suite>/<task_tag>/<algorithm>/seed_<N>/

CLI::

    python experiments/weighted_lse_dp/runners/run_phase4_dp.py \\
        --config experiments/weighted_lse_dp/configs/phase4/translation_study_4a2.json \\
        [--task TASK_TAG | all] [--algorithm ALG] [--seed N] \\
        [--out-root PATH] [--suite-suffix SUFFIX] [--dry-run]

Where ``TASK_TAG`` is ``<family>_<index>`` (e.g. ``dense_chain_cost_0``) --
the position of the task in the activation suite's ``selected_tasks`` list.
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

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from experiments.weighted_lse_dp.common.callbacks import (  # noqa: E402
    DPCurvesLogger,
)
from experiments.weighted_lse_dp.common.calibration import (  # noqa: E402
    build_calibration_stats_from_dp_tables,
    get_task_sign,
)
from experiments.weighted_lse_dp.common.schemas import (  # noqa: E402
    RunWriter,
)
from experiments.weighted_lse_dp.common.seeds import (  # noqa: E402
    seed_everything,
)
from experiments.weighted_lse_dp.common.manifests import (  # noqa: E402
    write_safe_provenance,
)
from experiments.weighted_lse_dp.geometry.phase4_calibration_v3 import (  # noqa: E402
    build_schedule_v3_from_pilot,
)
from experiments.weighted_lse_dp.geometry.task_activation_search import (  # noqa: E402
    run_classical_pilot,
)
from experiments.weighted_lse_dp.tasks.phase4_operator_suite import (  # noqa: E402
    build_phase4_task,
)

__all__ = ["main", "run_single", "build_plan"]


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

CLASSICAL_ALGORITHMS: tuple[str, ...] = (
    "classical_vi",
    "classical_async_vi",
)

SAFE_ALGORITHMS: tuple[str, ...] = (
    "safe_vi",
    "safe_async_vi",
    "safe_vi_zero",
    "safe_async_vi_zero",
)

ALL_ALGORITHMS: tuple[str, ...] = CLASSICAL_ALGORITHMS + SAFE_ALGORITHMS


def _is_safe(alg: str) -> bool:
    return alg in SAFE_ALGORITHMS


def _is_zero_schedule(alg: str) -> bool:
    return alg.endswith("_zero")


def _is_async(alg: str) -> bool:
    return "async" in alg


# ---------------------------------------------------------------------------
# Task tag helpers
# ---------------------------------------------------------------------------

def _task_tag(family: str, idx: int) -> str:
    """Canonical task tag used as a directory name."""
    return f"{family}_{idx}"


def _load_activation_suite(path: Path) -> list[dict[str, Any]]:
    """Load the frozen IV-A (or IV-A2) activation suite config."""
    with open(path) as f:
        payload = json.load(f)
    tasks = payload.get("selected_tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            f"activation suite at {path} has no 'selected_tasks' list."
        )
    return tasks


# ---------------------------------------------------------------------------
# Schedule construction (mirrors the Phase IV-B RL runner)
# ---------------------------------------------------------------------------

def _wrap_v3_schedule_for_betaschedule(
    v3: dict[str, Any],
    *,
    gamma: float,
) -> dict[str, Any]:
    """Adapt a v3 schedule dict to the BetaSchedule constructor contract.

    See ``run_phase4_rl.py`` for the full contract.  Summary: v3 emits
    ``beta_used_t``, ``alpha_t``, ``kappa_t``, ``Bhat_t`` but NOT
    ``beta_raw_t`` or ``beta_cap_t``.  The v3 ``beta_used_t`` already
    respects the trust region, so we set ``beta_raw_t = beta_used_t``
    (no clipping); ``beta_cap_t`` is recomputed from ``alpha_t`` via
    :func:`build_certification`.
    """
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
        build_certification,
    )

    alpha_t = np.asarray(v3["alpha_t"], dtype=np.float64)
    reward_bound = float(v3["reward_bound"])
    cert = build_certification(
        alpha_t=alpha_t, R_max=reward_bound, gamma=gamma
    )

    beta_used_t = np.asarray(v3["beta_used_t"], dtype=np.float64)
    beta_cap_t = cert["beta_cap_t"]
    # Clip to avoid floating-point slivers tripping BetaSchedule's consistency
    # check (beta_used_t == clip(beta_raw_t, -beta_cap_t, beta_cap_t)).
    beta_used_t = np.clip(beta_used_t, -beta_cap_t, beta_cap_t)

    T = len(beta_used_t)
    wrapped: dict[str, Any] = {
        "task_family": v3.get("task_family", ""),
        "gamma": float(gamma),
        "sign": int(v3["sign_family"]),
        "source_phase": v3.get("source_phase", "phase4"),
        "reward_bound": reward_bound,
        "alpha_t": alpha_t.tolist(),
        "kappa_t": cert["kappa_t"].tolist(),
        "Bhat_t": cert["Bhat_t"].tolist(),
        "beta_raw_t": beta_used_t.tolist(),
        "beta_cap_t": beta_cap_t.tolist(),
        "beta_used_t": beta_used_t.tolist(),
        "clip_active_t": [False] * T,
        "informativeness_t": [0.0] * T,
        "d_target_t": [float(gamma)] * T,
        "calibration_source_path": v3.get("calibration_source_path", ""),
        "calibration_hash": v3.get("calibration_hash", ""),
        "notes": v3.get("notes", "wrapped v3 -> BetaSchedule"),
    }
    return wrapped


def _build_stagewise_schedule(
    *,
    cfg: dict[str, Any],
    seed: int,
    n_pilot_episodes: int,
    gamma: float,
    run_dir: Path,
) -> Any:
    """Run a classical pilot, build a v3 schedule, wrap as BetaSchedule.

    Writes the raw v3 schedule to ``<run_dir>/schedule_v3.json`` for
    provenance, independently of the wrapped BetaSchedule instance.
    """
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule

    sign_family = int(get_task_sign(cfg.get("family", "unknown")))

    pilot = run_classical_pilot(
        cfg=cfg,
        seed=seed,
        n_episodes=n_pilot_episodes,
        sign_family=sign_family,
    )

    r_max = float(cfg.get("reward_bound", pilot.get("reward_bound", 1.0)))
    gamma_base = float(gamma)

    schedule_v3_path = run_dir / "schedule_v3.json"
    v3 = build_schedule_v3_from_pilot(
        pilot_data=pilot,
        r_max=r_max,
        gamma_base=gamma_base,
        gamma_eval=gamma_base,
        task_family=str(cfg.get("family", "unknown")),
        sign_family=sign_family,
        source_phase="phase4_dp",
        notes="Phase IV-B stagewise schedule from classical pilot (DP)",
        output_path=schedule_v3_path,
    )

    wrapped = _wrap_v3_schedule_for_betaschedule(v3, gamma=gamma_base)
    schedule = BetaSchedule(wrapped)
    return schedule, str(schedule_v3_path)


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------

def _make_planner(
    algorithm: str,
    mdp: Any,
    *,
    schedule: Any = None,
) -> Any:
    """Construct a DP planner for a Phase IV-B run."""
    from mushroom_rl.algorithms.value.dp import (
        ClassicalAsyncValueIteration,
        ClassicalValueIteration,
        SafeWeightedAsyncValueIteration,
        SafeWeightedValueIteration,
    )

    if algorithm == "classical_vi":
        return ClassicalValueIteration(mdp, n_sweeps=1, tol=0.0)
    elif algorithm == "classical_async_vi":
        return ClassicalAsyncValueIteration(
            mdp, n_sweeps=1, order="sequential", tol=0.0,
        )
    elif algorithm in ("safe_vi", "safe_vi_zero"):
        if schedule is None:
            raise ValueError(
                f"safe algorithm {algorithm!r} requires a schedule."
            )
        return SafeWeightedValueIteration(mdp, schedule=schedule)
    elif algorithm in ("safe_async_vi", "safe_async_vi_zero"):
        if schedule is None:
            raise ValueError(
                f"safe algorithm {algorithm!r} requires a schedule."
            )
        return SafeWeightedAsyncValueIteration(mdp, schedule=schedule)
    else:
        raise ValueError(f"Unknown Phase IV-B DP algorithm: {algorithm!r}")


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------

def build_plan(
    config: dict[str, Any],
    activation_tasks: list[dict[str, Any]],
    *,
    task_filter: str | None = None,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Build the list of (task, algorithm, seed) runs.

    Parameters
    ----------
    config:
        Loaded translation_study JSON.
    activation_tasks:
        ``selected_tasks`` list from the activation suite JSON.
    task_filter, algorithm_filter, seed_filter:
        CLI filters. ``task_filter='all'`` or ``None`` means no filter.
    """
    seeds = list(config.get("seeds_dp", config.get("seeds", [])))
    algorithms = list(config.get("dp_algorithms", ALL_ALGORITHMS))

    plan: list[dict[str, Any]] = []
    for idx, entry in enumerate(activation_tasks):
        cfg = dict(entry.get("cfg", {}))
        if "family" not in cfg and "family" in entry:
            cfg["family"] = entry["family"]
        family = str(cfg.get("family") or entry.get("family", "unknown"))
        tag = _task_tag(family, idx)

        if task_filter is not None and task_filter != "all" and task_filter != tag:
            continue

        for algo in algorithms:
            if algo not in ALL_ALGORITHMS:
                continue
            if algorithm_filter is not None and algorithm_filter != algo:
                continue
            for seed in seeds:
                if seed_filter is not None and int(seed) != int(seed_filter):
                    continue
                plan.append({
                    "task_tag": tag,
                    "task_idx": idx,
                    "family": family,
                    "cfg": cfg,
                    "algorithm": algo,
                    "seed": int(seed),
                })

    return plan


# ---------------------------------------------------------------------------
# Helpers mirrored from run_phase3_dp.py
# ---------------------------------------------------------------------------

def _get_sweep_times(planner: Any) -> list[float]:
    """Extract per-sweep wall-clock times from any planner type."""
    if hasattr(planner, "sweep_times_s"):
        return list(planner.sweep_times_s)
    if hasattr(planner, "iter_times_s"):
        return list(planner.iter_times_s)
    return []


def _v_for_sweep_index(planner: Any, sweep_idx: int) -> Any:
    """Return the value table after sweep ``sweep_idx``."""
    hist = getattr(planner, "V_sweep_history", None)
    if hist is not None and sweep_idx < len(hist):
        return hist[sweep_idx]
    return planner.V


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def run_single(
    *,
    task_tag: str,
    family: str,
    cfg: dict[str, Any],
    algorithm: str,
    seed: int,
    out_root: Path,
    suite: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute one (task, algorithm, seed) Phase IV-B DP run."""
    from mushroom_rl.algorithms.value.dp import (
        BetaSchedule,
        extract_mdp_arrays,
    )

    t_start = time.perf_counter()

    n_pilot_episodes = int(config.get("n_pilot_episodes", 200))

    gamma = float(cfg.get("gamma", 0.97))
    horizon = int(cfg.get("horizon", 20))

    # -- Seed everything ---------------------------------------------------
    seed_everything(seed)

    mdp_base, _mdp_rl, resolved_cfg = build_phase4_task(cfg, seed=seed)

    gamma = float(resolved_cfg.get("gamma", gamma))
    horizon = int(resolved_cfg.get("horizon", horizon))

    # -- Resolved config for run.json -------------------------------------
    resolved_config: dict[str, Any] = {
        "phase": "IV-B",
        "suite": suite,
        "task_tag": task_tag,
        "family": family,
        "algorithm": algorithm,
        "seed": seed,
        "gamma": gamma,
        "horizon": horizon,
        "n_pilot_episodes": n_pilot_episodes,
        "task_cfg": resolved_cfg,
        "source_activation_suite": config.get("activation_suite", ""),
    }

    # -- Create RunWriter --------------------------------------------------
    rw = RunWriter.create(
        base=out_root,
        phase="phase4",
        suite=suite,
        task=task_tag,
        algorithm=algorithm,
        seed=seed,
        config=resolved_config,
        storage_mode="dp_stagewise",
    )

    # -- Build schedule for safe algorithms --------------------------------
    schedule = None
    schedule_path: str = ""
    if _is_safe(algorithm):
        if _is_zero_schedule(algorithm):
            schedule = BetaSchedule.zeros(T=horizon, gamma=gamma)
            schedule_path = "BetaSchedule.zeros"
        else:
            schedule, schedule_path = _build_stagewise_schedule(
                cfg=resolved_cfg,
                seed=seed,
                n_pilot_episodes=n_pilot_episodes,
                gamma=gamma,
                run_dir=rw.run_dir,
            )
        resolved_config["schedule_sign"] = int(schedule.sign)
        resolved_config["schedule_T"] = int(schedule.T)
        resolved_config["schedule_path"] = schedule_path

    # -- Planner -----------------------------------------------------------
    planner = _make_planner(
        algorithm,
        mdp_base,
        schedule=schedule,
    )

    print(
        f"[phase4_dp] {task_tag}/{algorithm}/seed_{seed}: "
        f"horizon={horizon}, gamma={gamma:.3f}"
    )

    # -- Run the planner ---------------------------------------------------
    with rw.timer.phase("fit"):
        planner.run()

    # -- Extract MDP arrays for calibration stats -------------------------
    p, r, horizon_mdp, gamma_mdp = extract_mdp_arrays(mdp_base)
    n_states = int(p.shape[0])

    # For exact-pass DP (n_sweeps=1) planner.V is itself the exact fixed
    # point; use it as the reference for supnorm_to_exact.
    _v_exact_ref = planner.V

    dp_logger = DPCurvesLogger(
        run_writer=rw,
        v_exact=_v_exact_ref,
        task=task_tag,
    )

    sweep_times = _get_sweep_times(planner)
    residuals = list(planner.residuals)
    n_sweeps_actual = len(residuals)

    cumulative_time = 0.0
    for i in range(n_sweeps_actual):
        sweep_wall = sweep_times[i] if i < len(sweep_times) else 0.0
        cumulative_time += sweep_wall
        residual = residuals[i] if i < len(residuals) else 0.0

        dp_logger.record_sweep(
            sweep_idx=i,
            bellman_residual=residual,
            wall_clock_s=cumulative_time,
            v_current=_v_for_sweep_index(planner, i),
        )

    summary = dp_logger.summary()

    # -- Calibration stats from exact DP tables ---------------------------
    task_sign = int(get_task_sign(family))
    calib_stats = build_calibration_stats_from_dp_tables(
        Q=planner.Q,
        V=planner.V,
        P=p,
        R=r,
        gamma=gamma_mdp,
        horizon=horizon_mdp,
        sign=task_sign,
    )

    # -- Safe-specific per-stage diagnostics ------------------------------
    if _is_safe(algorithm):
        clip_summary = getattr(planner, "clipping_summary", {})
        T = int(horizon_mdp)

        safe_beta_used_arr = np.array(
            clip_summary.get(
                "stage_beta_used",
                schedule._beta_used_t if schedule is not None else [0.0] * T,
            ),
            dtype=np.float64,
        )
        safe_clip_active_arr = np.array(
            clip_summary.get("stage_clip_active", [False] * T),
        )
        safe_eff_discount_mean_arr = np.array(
            clip_summary.get("stage_eff_discount_mean", [gamma_mdp] * T),
            dtype=np.float64,
        )
        safe_eff_discount_std_arr = np.array(
            clip_summary.get("stage_eff_discount_std", [0.0] * T),
            dtype=np.float64,
        )
        safe_frac_lt_gamma_arr = np.array(
            clip_summary.get("stage_frac_eff_lt_gamma", [0.0] * T),
            dtype=np.float64,
        )

        _one_plus_gamma = 1.0 + gamma_mdp
        calib_stats["safe_rho_mean"] = (
            1.0 - safe_eff_discount_mean_arr[:T] / _one_plus_gamma
        )
        calib_stats["safe_rho_std"] = (
            safe_eff_discount_std_arr[:T] / _one_plus_gamma
        )
        calib_stats["safe_effective_discount_mean"] = (
            safe_eff_discount_mean_arr[:T]
        )
        calib_stats["safe_effective_discount_std"] = (
            safe_eff_discount_std_arr[:T]
        )
        calib_stats["safe_beta_used_min"] = safe_beta_used_arr[:T].copy()
        calib_stats["safe_beta_used_mean"] = safe_beta_used_arr[:T].copy()
        calib_stats["safe_beta_used_max"] = safe_beta_used_arr[:T].copy()
        calib_stats["safe_clip_fraction"] = (
            safe_clip_active_arr[:T].astype(np.float64)
        )
        calib_stats["safe_underdiscount_fraction"] = safe_frac_lt_gamma_arr[:T]

        # Per-stage Bellman residuals (VI fills with last-sweep global residual).
        safe_bellman_residual = np.full(T, np.nan, dtype=np.float64)
        if residuals:
            safe_bellman_residual[:] = residuals[-1]
        calib_stats["safe_bellman_residual"] = safe_bellman_residual

    rw.set_calibration_stats(calib_stats)

    # -- Safe provenance --------------------------------------------------
    if _is_safe(algorithm):
        raw = getattr(schedule, "_raw", {}) or {}
        write_safe_provenance(
            rw.run_dir,
            schedule_path=str(schedule_path),
            calibration_source_path=str(raw.get("calibration_source_path", "")),
            calibration_hash=str(raw.get("calibration_hash", "")),
            source_phase=str(raw.get("source_phase", "phase4_dp")),
        )

    # -- Flush ------------------------------------------------------------
    final_residual = residuals[-1] if residuals else 0.0
    metrics: dict[str, Any] = {
        "final_bellman_residual": float(final_residual),
        "n_sweeps": int(n_sweeps_actual),
        "wall_clock_s": float(planner.wall_clock_s),
        "converged": (
            bool(planner.converged) if hasattr(planner, "converged") else True
        ),
        "n_states": n_states,
    }

    if _is_safe(algorithm):
        schedule_report = getattr(planner, "schedule_report", {})
        if schedule_report:
            metrics["schedule_sign"] = int(schedule_report.get("sign", 0))
            beta_range = schedule_report.get("beta_used_range", [0.0, 0.0])
            metrics["beta_used_min"] = float(beta_range[0])
            metrics["beta_used_max"] = float(beta_range[1])
        clip_summary = getattr(planner, "clipping_summary", {})
        if clip_summary:
            metrics["n_stages_clipped"] = int(
                clip_summary.get("n_stages_clipped", 0)
            )
            metrics["clip_fraction"] = float(
                clip_summary.get("clip_fraction", 0.0)
            )

    for key, val in summary.items():
        if val is not None:
            metrics[key] = val

    rw.flush(metrics=metrics)

    wall_s = time.perf_counter() - t_start
    print(
        f"[phase4_dp] {task_tag}/{algorithm}/seed_{seed}: DONE in "
        f"{wall_s:.1f}s -> {rw.run_dir}"
    )
    print(
        f"    sweeps={n_sweeps_actual}, residual={final_residual:.2e}, "
        f"planner_wall={planner.wall_clock_s:.4f}s, n_states={n_states}"
    )

    return {
        "task_tag": task_tag,
        "algorithm": algorithm,
        "seed": seed,
        "passed": True,
        "wall_s": wall_s,
        "run_dir": str(rw.run_dir),
        "summary": (
            f"n_sweeps={n_sweeps_actual}, "
            f"residual={final_residual:.2e}"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_phase4_dp",
        description=__doc__,
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Translation-study config JSON.")
    p.add_argument("--task", type=str, default="all",
                   help="Task tag (e.g. dense_chain_cost_0) or 'all'.")
    p.add_argument("--algorithm", type=str, default=None,
                   choices=list(ALL_ALGORITHMS),
                   help="Filter to a single algorithm.")
    p.add_argument("--seed", type=int, default=None,
                   help="Filter to a single seed.")
    p.add_argument("--out-root", type=Path,
                   default=Path("results/weighted_lse_dp"),
                   help="Output root directory.")
    p.add_argument("--suite-suffix", type=str, default="",
                   help=(
                       "Optional suffix appended to the suite name from the "
                       "config file (used for ablation / sweep subdirectories)."
                   ))
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned runs without executing.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        return 1
    with open(config_path) as f:
        config = json.load(f)

    suite = config.get("suite", "translation")
    if args.suite_suffix:
        suite = f"{suite}{args.suite_suffix}"

    activation_path_raw = config.get("activation_suite")
    if not activation_path_raw:
        print(
            "ERROR: config missing required key 'activation_suite'",
            file=sys.stderr,
        )
        return 1
    activation_path = Path(activation_path_raw)
    if not activation_path.is_absolute():
        activation_path = _REPO_ROOT / activation_path
    if not activation_path.is_file():
        print(f"ERROR: activation suite not found at {activation_path}",
              file=sys.stderr)
        return 1
    activation_tasks = _load_activation_suite(activation_path)

    plan = build_plan(
        config,
        activation_tasks,
        task_filter=args.task,
        algorithm_filter=args.algorithm,
        seed_filter=args.seed,
    )

    if not plan:
        print("No runs matched the filters.", file=sys.stderr)
        return 1

    out_root = Path(args.out_root)

    if args.dry_run:
        print(f"[phase4_dp] DRY RUN -- {len(plan)} run(s) planned:")
        print(f"  config:           {config_path}")
        print(f"  activation_suite: {activation_path}")
        print(f"  suite:            {suite}")
        print(f"  out_root:         {out_root}")
        print()
        for i, entry in enumerate(plan, 1):
            print(
                f"  [{i:>3d}] task={entry['task_tag']:<24s} "
                f"algo={entry['algorithm']:<24s} "
                f"seed={entry['seed']:<6d} "
                f"family={entry['family']}"
            )
        return 0

    print(
        f"[phase4_dp] Executing {len(plan)} run(s) "
        f"(suite={suite}, out_root={out_root})"
    )

    results: list[dict[str, Any]] = []
    n_passed = 0
    for i, entry in enumerate(plan, 1):
        print(
            f"\n[phase4_dp] === Run {i}/{len(plan)}: "
            f"{entry['task_tag']}/{entry['algorithm']}/seed_{entry['seed']} ==="
        )
        try:
            result = run_single(
                task_tag=entry["task_tag"],
                family=entry["family"],
                cfg=entry["cfg"],
                algorithm=entry["algorithm"],
                seed=entry["seed"],
                out_root=out_root,
                suite=suite,
                config=config,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[phase4_dp] FAILED: {exc!r}")
            print(tb)
            result = {
                "task_tag": entry["task_tag"],
                "algorithm": entry["algorithm"],
                "seed": entry["seed"],
                "passed": False,
                "wall_s": 0.0,
                "run_dir": None,
                "summary": "EXCEPTION",
                "error": repr(exc),
                "traceback": tb,
            }
        results.append(result)
        if result["passed"]:
            n_passed += 1

    print(f"\n[phase4_dp] Summary: {n_passed}/{len(results)} runs passed")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['task_tag']:<24s} {r['algorithm']:<24s} "
            f"seed={r['seed']:<6d} {status}  {r.get('summary', '')}"
        )
    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
