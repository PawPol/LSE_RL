#!/usr/bin/env python
"""Phase IV-B: RL translation experiments on the frozen activation suite.

Runs classical and safe RL agents on the tasks identified by Phase IV-A (or
IV-A2) as having confirmed operator activation. Records learning curves,
safe-specific diagnostics, and outcome metrics for the translation analysis.

Supported algorithms (``--algorithm`` values):

    classical_q
    classical_expected_sarsa
    safe_q_stagewise                (beta schedule from v3 calibration)
    safe_expected_sarsa_stagewise   (beta schedule from v3 calibration)
    safe_q_zero                     (BetaSchedule.zeros -- classical collapse)
    safe_expected_sarsa_zero        (BetaSchedule.zeros -- classical collapse)

Layout::

    <out_root>/phase4/<suite>/<task_tag>/<algorithm>/seed_<N>/

CLI::

    python experiments/weighted_lse_dp/runners/run_phase4_rl.py \\
        --config experiments/weighted_lse_dp/configs/phase4/translation_study_4a2.json \\
        [--task TASK_TAG | all] [--algorithm ALG] [--seed N] \\
        [--out-root PATH] [--suite-suffix SUFFIX] [--dry-run]

Where ``TASK_TAG`` is ``<family>_<index>`` (e.g. ``dense_chain_cost_0``) --
the position of the task in the activation suite's ``selected_tasks`` list.
"""

from __future__ import annotations

import argparse
import copy
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
    RLEvaluator,
    SafeTransitionLogger,
    TransitionLogger,
)
from experiments.weighted_lse_dp.common.calibration import (  # noqa: E402
    aggregate_calibration_stats,
    get_task_sign,
)
from experiments.weighted_lse_dp.common.schemas import (  # noqa: E402
    RunWriter,
    aggregate_safe_stats,
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
    "classical_q",
    "classical_expected_sarsa",
)

SAFE_ALGORITHMS: tuple[str, ...] = (
    "safe_q_stagewise",
    "safe_expected_sarsa_stagewise",
    "safe_q_zero",
    "safe_expected_sarsa_zero",
)

ALL_ALGORITHMS: tuple[str, ...] = CLASSICAL_ALGORITHMS + SAFE_ALGORITHMS


def _is_safe(alg: str) -> bool:
    return alg in SAFE_ALGORITHMS


def _is_zero_schedule(alg: str) -> bool:
    return alg.endswith("_zero")


def _is_q_variant(alg: str) -> bool:
    return alg in (
        "classical_q",
        "safe_q_stagewise",
        "safe_q_zero",
    )


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
# Schedule construction
# ---------------------------------------------------------------------------

def _wrap_v3_schedule_for_betaschedule(
    v3: dict[str, Any],
    *,
    gamma: float,
) -> dict[str, Any]:
    """Adapt a v3 schedule dict to the BetaSchedule constructor contract.

    The v3 calibrator emits ``beta_used_t``, ``alpha_t``, ``kappa_t``,
    ``Bhat_t`` but NOT ``beta_raw_t`` or ``beta_cap_t``.  BetaSchedule
    requires both and also the scalar key ``gamma`` (v3 uses
    ``gamma_base``/``gamma_eval``).  The v3 ``beta_used_t`` already
    respects the trust-region / safe caps, so we can set
    ``beta_raw_t = beta_used_t`` (no clipping); ``beta_cap_t`` is then
    recomputed from ``alpha_t`` via :func:`build_certification` which
    matches BetaSchedule's round-trip consistency check.
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
    # beta_cap_t from v3's safe certification must bound |beta_used_t|;
    # if v3 respected its own caps (it does by construction), this
    # implies |beta_used_t| <= beta_cap_t already.
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
        source_phase="phase4_rl",
        notes="Phase IV-B stagewise schedule from classical pilot",
        output_path=schedule_v3_path,
    )

    wrapped = _wrap_v3_schedule_for_betaschedule(v3, gamma=gamma_base)
    schedule = BetaSchedule(wrapped)
    return schedule, str(schedule_v3_path)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def _make_agent(
    algorithm: str,
    mdp_info: Any,
    *,
    epsilon: float,
    learning_rate: float,
    schedule: Any = None,
    n_base: int | None = None,
) -> Any:
    """Construct a MushroomRL tabular RL agent for a Phase IV-B run."""
    from mushroom_rl.algorithms.value.td import (  # noqa: E402
        ExpectedSARSA,
        QLearning,
        SafeExpectedSARSA,
        SafeQLearning,
    )
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import Parameter

    pi = EpsGreedy(epsilon=Parameter(value=epsilon))
    lr = Parameter(value=learning_rate)

    if algorithm == "classical_q":
        agent = QLearning(mdp_info, pi, lr)
    elif algorithm == "classical_expected_sarsa":
        agent = ExpectedSARSA(mdp_info, pi, lr)
    elif algorithm in ("safe_q_stagewise", "safe_q_zero"):
        if schedule is None or n_base is None:
            raise ValueError(
                f"safe algorithm {algorithm!r} requires schedule and n_base."
            )
        agent = SafeQLearning(mdp_info, pi, schedule, n_base, learning_rate=lr)
    elif algorithm in (
        "safe_expected_sarsa_stagewise",
        "safe_expected_sarsa_zero",
    ):
        if schedule is None or n_base is None:
            raise ValueError(
                f"safe algorithm {algorithm!r} requires schedule and n_base."
            )
        agent = SafeExpectedSARSA(
            mdp_info, pi, schedule, n_base, learning_rate=lr
        )
    else:
        raise ValueError(f"Unknown Phase IV-B algorithm: {algorithm!r}")

    pi.set_q(agent.Q)
    return agent


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
    seeds = list(config.get("seeds_rl", config.get("seeds", [])))
    algorithms = list(config.get("rl_algorithms", ALL_ALGORITHMS))

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
    """Execute one (task, algorithm, seed) Phase IV-B RL run."""
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule
    from mushroom_rl.core import Core

    t_start = time.perf_counter()

    train_steps = int(config["train_steps"])
    checkpoint_every = int(config["checkpoint_every"])
    eval_episodes = int(config.get("eval_episodes", 50))
    epsilon = float(config.get("epsilon", 0.1))
    learning_rate = float(config.get("learning_rate", 0.1))
    success_threshold = float(config.get("success_threshold", 0.5))
    n_pilot_episodes = int(config.get("n_pilot_episodes", 200))

    gamma = float(cfg.get("gamma", 0.97))
    horizon = int(cfg.get("horizon", 20))

    # -- Seed everything before env construction --------------------------
    seed_everything(seed)

    mdp_base, mdp_rl, resolved_cfg = build_phase4_task(cfg, seed=seed)
    # Propagate the resolved gamma / horizon onto our locals if the
    # factory clamped anything.
    gamma = float(resolved_cfg.get("gamma", gamma))
    horizon = int(resolved_cfg.get("horizon", horizon))

    mdp_rl.info.gamma = gamma

    # n_base = |S_total| derived from the time-augmented env.
    n_base = mdp_rl.info.observation_space.n // horizon

    # -- Prepare resolved_config for run.json/config.json -----------------
    resolved_config: dict[str, Any] = {
        "phase": "IV-B",
        "suite": suite,
        "task_tag": task_tag,
        "family": family,
        "algorithm": algorithm,
        "seed": seed,
        "gamma": gamma,
        "horizon": horizon,
        "n_base": n_base,
        "train_steps": train_steps,
        "checkpoint_every": checkpoint_every,
        "eval_episodes": eval_episodes,
        "epsilon": epsilon,
        "learning_rate": learning_rate,
        "success_threshold": success_threshold,
        "n_pilot_episodes": n_pilot_episodes,
        "task_cfg": resolved_cfg,
        "source_activation_suite": config.get("activation_suite", ""),
    }

    # -- Create RunWriter early so we can write schedule_v3.json alongside
    rw = RunWriter.create(
        base=out_root,
        phase="phase4",
        suite=suite,
        task=task_tag,
        algorithm=algorithm,
        seed=seed,
        config=resolved_config,
        storage_mode="rl_online",
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

    # -- Agent ------------------------------------------------------------
    agent = _make_agent(
        algorithm,
        mdp_rl.info,
        epsilon=epsilon,
        learning_rate=learning_rate,
        schedule=schedule,
        n_base=n_base,
    )

    # -- Logger ------------------------------------------------------------
    if _is_safe(algorithm):
        logger: TransitionLogger = SafeTransitionLogger(
            agent, n_base=n_base, gamma=gamma
        )
    else:
        logger = TransitionLogger(agent, n_base=n_base, gamma=gamma)

    # -- Evaluator ---------------------------------------------------------
    # Eval on a deepcopy so eval rollouts do not perturb the training env's
    # RNG state or episode counter.
    mdp_eval = copy.deepcopy(mdp_rl)
    evaluator = RLEvaluator(
        agent=agent,
        env=mdp_eval,
        run_writer=rw,
        n_eval_episodes=eval_episodes,
        success_threshold=success_threshold,
        gamma=gamma,
    )

    # -- Core --------------------------------------------------------------
    callbacks_fit: list[Any] = []
    if _is_safe(algorithm) and hasattr(logger, "after_fit"):
        callbacks_fit.append(logger.after_fit)  # type: ignore[attr-defined]

    core = Core(
        agent, mdp_rl,
        callback_step=logger,
        callbacks_fit=callbacks_fit,
    )

    print(
        f"[phase4_rl] {task_tag}/{algorithm}/seed_{seed}: "
        f"train_steps={train_steps}, checkpoint_every={checkpoint_every}, "
        f"horizon={horizon}, n_base={n_base}"
    )

    # -- Training loop with evaluation checkpoints -------------------------
    with rw.timer.phase("train"):
        steps_done = 0
        for _ in range(0, train_steps, checkpoint_every):
            n_this = min(checkpoint_every, train_steps - steps_done)
            if n_this <= 0:
                break
            core.learn(n_steps=n_this, n_steps_per_fit=1, quiet=True)
            steps_done += n_this

            with rw.timer.phase("eval"):
                eval_result = evaluator.evaluate(steps=steps_done)

            print(
                f"  checkpoint {steps_done:>8d}/{train_steps}: "
                f"disc_return={eval_result['disc_return_mean']:.4f}, "
                f"success_rate={eval_result['success_rate']:.4f}"
            )

    # -- Build transitions payload & calibration stats --------------------
    with rw.timer.phase("post"):
        transitions_payload = logger.build_payload()
        rw.set_transitions(transitions_payload)

        task_sign = int(get_task_sign(family))
        calibration_stats = aggregate_calibration_stats(
            transitions_payload, horizon=horizon, sign=task_sign,
        )

        if _is_safe(algorithm):
            safe_payload = (
                logger.build_safe_payload()
                if hasattr(logger, "build_safe_payload")
                else None
            )
            if safe_payload is not None:
                safe_stats = aggregate_safe_stats(
                    safe_payload, T=horizon, gamma=gamma,
                )
                calibration_stats.update(safe_stats)

        # Missing event-level / adaptation scalars default to NaN/-1 so the
        # schema validates (CALIBRATION_ARRAYS requires these keys).
        _scalar_defaults = {
            "jackpot_event_rate":        np.array([np.nan]),
            "catastrophe_event_rate":    np.array([np.nan]),
            "regime_shift_episode":      np.array([-1.0]),
            "hazard_hit_rate":           np.array([np.nan]),
            "return_cvar_5pct":          np.array([np.nan]),
            "return_cvar_10pct":         np.array([np.nan]),
            "return_top5pct_mean":       np.array([np.nan]),
            "event_rate":                np.array([np.nan]),
            "event_conditioned_return":  np.array([np.nan]),
            "adaptation_pre_change_auc":  np.array([np.nan]),
            "adaptation_post_change_auc": np.array([np.nan]),
            "adaptation_lag_50pct":       np.array([-1.0]),
            "adaptation_lag_75pct":       np.array([-1.0]),
            "adaptation_lag_90pct":       np.array([-1.0]),
        }
        for k, default in _scalar_defaults.items():
            calibration_stats.setdefault(k, default)

        rw.set_calibration_stats(calibration_stats)

    # -- Summary metrics --------------------------------------------------
    eval_summary = evaluator.summary()

    # Per-episode returns over training -> mean_return metric.
    def _episode_returns(payload: dict[str, np.ndarray]) -> np.ndarray:
        ep = payload["episode_index"]
        rw_arr = payload["reward"]
        t_arr = payload["t"]
        if ep.size == 0:
            return np.zeros(0, dtype=np.float64)
        n_ep = int(ep.max()) + 1
        out = np.zeros(n_ep, dtype=np.float64)
        for i in range(ep.size):
            out[int(ep[i])] += (gamma ** int(t_arr[i])) * float(rw_arr[i])
        return out

    ep_returns = _episode_returns(transitions_payload)

    metrics: dict[str, Any] = {
        "train_steps": train_steps,
        "n_transitions": int(logger.n_transitions),
        "mean_return": float(np.mean(ep_returns)) if ep_returns.size else 0.0,
    }
    for k, v in eval_summary.items():
        metrics[k] = v

    rw.flush(
        metrics=metrics,
        step_count=train_steps,
        update_count=train_steps,
    )

    # -- Safe provenance --------------------------------------------------
    if _is_safe(algorithm):
        raw = getattr(schedule, "_raw", {}) or {}
        write_safe_provenance(
            rw.run_dir,
            schedule_path=str(schedule_path),
            calibration_source_path=str(raw.get("calibration_source_path", "")),
            calibration_hash=str(raw.get("calibration_hash", "")),
            source_phase=str(raw.get("source_phase", "phase4_rl")),
        )

    wall_s = time.perf_counter() - t_start
    print(
        f"[phase4_rl] {task_tag}/{algorithm}/seed_{seed}: DONE in "
        f"{wall_s:.1f}s -> {rw.run_dir}"
    )

    return {
        "task_tag": task_tag,
        "algorithm": algorithm,
        "seed": seed,
        "passed": True,
        "wall_s": wall_s,
        "run_dir": str(rw.run_dir),
        "summary": (
            f"mean_return={metrics['mean_return']:.4f}, "
            f"final_sr={eval_summary.get('final_10pct_success_rate', 0):.3f}"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_phase4_rl",
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
        print(f"[phase4_rl] DRY RUN -- {len(plan)} run(s) planned:")
        print(f"  config:           {config_path}")
        print(f"  activation_suite: {activation_path}")
        print(f"  suite:            {suite}")
        print(f"  out_root:         {out_root}")
        print()
        for i, entry in enumerate(plan, 1):
            print(
                f"  [{i:>3d}] task={entry['task_tag']:<24s} "
                f"algo={entry['algorithm']:<32s} "
                f"seed={entry['seed']:<6d} "
                f"family={entry['family']}"
            )
        return 0

    print(
        f"[phase4_rl] Executing {len(plan)} run(s) "
        f"(suite={suite}, out_root={out_root})"
    )

    results: list[dict[str, Any]] = []
    n_passed = 0
    for i, entry in enumerate(plan, 1):
        print(
            f"\n[phase4_rl] === Run {i}/{len(plan)}: "
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
            print(f"[phase4_rl] FAILED: {exc!r}")
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

    print(f"\n[phase4_rl] Summary: {n_passed}/{len(results)} runs passed")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['task_tag']:<24s} {r['algorithm']:<32s} "
            f"seed={r['seed']:<6d} {status}  {r.get('summary', '')}"
        )
    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
