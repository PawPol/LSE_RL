#!/usr/bin/env python
"""Phase IV-B: diagnostic-strength sweep.

Varies the reference-effect strength ``u_max`` (passed through to the
:func:`build_schedule_v3_from_pilot` calibration) to test monotone
activation-outcome correlation on the frozen IV-A2 activation suite.

The sweep iterates over ``config['sweep_values']`` (a list of floats) and,
for each value ``u_max``, runs the same algorithm specified in
``config['algorithm']`` (default: ``safe_q_stagewise``) on every task in
the activation suite for every seed in ``config['seeds_rl']``.  Results are
written with the suite name suffixed by ``_umax_<value>``, so each sweep
point lives in its own output subtree::

    <out_root>/phase4/<suite>_umax_0.010/<task_tag>/<algorithm>/seed_<N>/

CLI::

    python experiments/weighted_lse_dp/runners/run_phase4_diagnostic_sweep.py \\
        --config experiments/weighted_lse_dp/configs/phase4/diagnostic_strength_sweep.json \\
        [--task TASK_TAG | all] [--seed N] [--sweep-value V] \\
        [--out-root PATH] [--dry-run]
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

# Import the RL runner module; we monkey-patch its schedule builder so we can
# thread the ``u_max`` kwarg through without duplicating the full
# train/logger/flush pipeline.
from experiments.weighted_lse_dp.runners import run_phase4_rl  # noqa: E402
from experiments.weighted_lse_dp.geometry.phase4_calibration_v3 import (  # noqa: E402
    build_schedule_v3_from_pilot,
)
from experiments.weighted_lse_dp.geometry.task_activation_search import (  # noqa: E402
    run_classical_pilot,
)
from experiments.weighted_lse_dp.common.calibration import (  # noqa: E402
    get_task_sign,
)

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Schedule factory that threads u_max into the v3 calibrator.
# ---------------------------------------------------------------------------


def _make_stagewise_builder(u_max: float):
    """Return a ``_build_stagewise_schedule``-compatible callable.

    The returned callable has the same signature as
    :func:`run_phase4_rl._build_stagewise_schedule` but always invokes
    :func:`build_schedule_v3_from_pilot` with ``u_max=u_max``.  The RL
    runner's ``run_single`` is monkey-patched to use this factory for the
    duration of the sweep value.
    """

    def _build(
        *,
        cfg: dict[str, Any],
        seed: int,
        n_pilot_episodes: int,
        gamma: float,
        run_dir: Path,
    ) -> Any:
        from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
            BetaSchedule,
        )

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
            source_phase="phase4_diagnostic_sweep",
            notes=(
                f"Phase IV-B diagnostic sweep (u_max={u_max:.4f}) stagewise "
                f"schedule from classical pilot"
            ),
            output_path=schedule_v3_path,
            u_max=u_max,
        )

        wrapped = run_phase4_rl._wrap_v3_schedule_for_betaschedule(
            v3, gamma=gamma_base,
        )
        schedule = BetaSchedule(wrapped)
        return schedule, str(schedule_v3_path)

    return _build


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_phase4_diagnostic_sweep",
        description=__doc__,
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Diagnostic-strength sweep config JSON.")
    p.add_argument("--task", type=str, default="all",
                   help="Task tag (e.g. dense_chain_cost_0) or 'all'.")
    p.add_argument("--seed", type=int, default=None,
                   help="Filter to a single seed.")
    p.add_argument("--sweep-value", type=float, default=None,
                   help="Filter to a single sweep value (u_max).")
    p.add_argument("--out-root", type=Path,
                   default=Path("results/weighted_lse_dp"),
                   help="Output root directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned runs without executing.")
    return p.parse_args(argv)


def _load_activation_suite(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        payload = json.load(f)
    tasks = payload.get("selected_tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            f"activation suite at {path} has no 'selected_tasks' list."
        )
    return tasks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        return 1
    with open(config_path) as f:
        config = json.load(f)

    algorithm = str(config.get("algorithm", "safe_q_stagewise"))
    if algorithm not in run_phase4_rl.SAFE_ALGORITHMS:
        print(
            f"ERROR: diagnostic sweep requires a safe algorithm; "
            f"got {algorithm!r}.",
            file=sys.stderr,
        )
        return 1
    if algorithm.endswith("_zero"):
        print(
            f"ERROR: diagnostic sweep cannot run on a zero-schedule algorithm; "
            f"got {algorithm!r}.",
            file=sys.stderr,
        )
        return 1

    sweep_values = config.get("sweep_values", [])
    if not sweep_values:
        print("ERROR: config missing 'sweep_values'.", file=sys.stderr)
        return 1

    if args.sweep_value is not None:
        sweep_values = [v for v in sweep_values if float(v) == args.sweep_value]
        if not sweep_values:
            print(
                f"ERROR: --sweep-value {args.sweep_value} not in "
                f"config sweep_values.",
                file=sys.stderr,
            )
            return 1

    activation_path_raw = config.get("activation_suite")
    if not activation_path_raw:
        print(
            "ERROR: config missing 'activation_suite'.", file=sys.stderr,
        )
        return 1
    activation_path = Path(activation_path_raw)
    if not activation_path.is_absolute():
        activation_path = _REPO_ROOT / activation_path
    if not activation_path.is_file():
        print(
            f"ERROR: activation suite not found at {activation_path}",
            file=sys.stderr,
        )
        return 1
    activation_tasks = _load_activation_suite(activation_path)

    base_suite = config.get("suite", "diagnostic_sweep")
    out_root = Path(args.out_root)

    # Per-value translation config -- the RL runner expects seeds_rl +
    # rl_algorithms so we synthesise a minimal one.
    base_translation_config: dict[str, Any] = dict(config)
    base_translation_config.setdefault("rl_algorithms", [algorithm])
    base_translation_config["rl_algorithms"] = [algorithm]

    # Build global plan (sweep_value, task, seed) list --------------------
    global_plan: list[dict[str, Any]] = []
    for u_max in sweep_values:
        suite = f"{base_suite}_umax_{float(u_max):.4f}"
        sub_config = copy.deepcopy(base_translation_config)
        sub_config["suite"] = suite
        sub_config["u_max"] = float(u_max)

        plan = run_phase4_rl.build_plan(
            sub_config,
            activation_tasks,
            task_filter=args.task,
            algorithm_filter=algorithm,
            seed_filter=args.seed,
        )
        for entry in plan:
            entry = dict(entry)
            entry["u_max"] = float(u_max)
            entry["suite"] = suite
            entry["sub_config"] = sub_config
            global_plan.append(entry)

    if not global_plan:
        print("No runs matched the filters.", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            f"[phase4_sweep] DRY RUN -- {len(global_plan)} run(s) planned "
            f"over {len(sweep_values)} sweep value(s):"
        )
        print(f"  config:           {config_path}")
        print(f"  activation_suite: {activation_path}")
        print(f"  algorithm:        {algorithm}")
        print(f"  sweep_values:     {sweep_values}")
        print(f"  out_root:         {out_root}")
        print()
        for i, entry in enumerate(global_plan, 1):
            print(
                f"  [{i:>3d}] u_max={entry['u_max']:.4f}  "
                f"suite={entry['suite']:<36s} "
                f"task={entry['task_tag']:<24s} "
                f"seed={entry['seed']:<6d}"
            )
        return 0

    print(
        f"[phase4_sweep] Executing {len(global_plan)} run(s) across "
        f"{len(sweep_values)} sweep value(s) "
        f"(algorithm={algorithm}, out_root={out_root})"
    )

    results: list[dict[str, Any]] = []
    n_passed = 0

    # Save originals so we can restore after each sweep value.
    _original_builder = run_phase4_rl._build_stagewise_schedule

    try:
        current_u_max: float | None = None
        for i, entry in enumerate(global_plan, 1):
            u_max = entry["u_max"]
            if u_max != current_u_max:
                # Swap in the factory that bakes this u_max into every
                # pilot-driven schedule build.
                run_phase4_rl._build_stagewise_schedule = (  # type: ignore[assignment]
                    _make_stagewise_builder(u_max)
                )
                current_u_max = u_max

            print(
                f"\n[phase4_sweep] === Run {i}/{len(global_plan)}: "
                f"u_max={u_max:.4f} / {entry['task_tag']}/{algorithm}/"
                f"seed_{entry['seed']} ==="
            )
            t_start = time.perf_counter()
            try:
                result = run_phase4_rl.run_single(
                    task_tag=entry["task_tag"],
                    family=entry["family"],
                    cfg=entry["cfg"],
                    algorithm=algorithm,
                    seed=entry["seed"],
                    out_root=out_root,
                    suite=entry["suite"],
                    config=entry["sub_config"],
                )
                result["u_max"] = u_max
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[phase4_sweep] FAILED: {exc!r}")
                print(tb)
                result = {
                    "task_tag": entry["task_tag"],
                    "algorithm": algorithm,
                    "seed": entry["seed"],
                    "u_max": u_max,
                    "passed": False,
                    "wall_s": time.perf_counter() - t_start,
                    "run_dir": None,
                    "summary": "EXCEPTION",
                    "error": repr(exc),
                    "traceback": tb,
                }
            results.append(result)
            if result.get("passed", False):
                n_passed += 1
    finally:
        run_phase4_rl._build_stagewise_schedule = _original_builder  # type: ignore[assignment]

    print(
        f"\n[phase4_sweep] Summary: {n_passed}/{len(results)} runs passed"
    )
    for r in results:
        status = "PASS" if r.get("passed", False) else "FAIL"
        print(
            f"  u_max={r.get('u_max', float('nan')):.4f}  "
            f"{r['task_tag']:<24s} {algorithm:<32s} "
            f"seed={r['seed']:<6d} {status}  "
            f"{r.get('summary', '')}"
        )
    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
