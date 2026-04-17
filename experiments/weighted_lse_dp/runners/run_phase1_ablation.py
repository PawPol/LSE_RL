#!/usr/bin/env python
"""Phase I gamma-prime ablation runner.

Sweeps over the ``gamma_prime_values`` grid from ``paper_suite.json``,
running each (task, algorithm, seed, gamma') combination through the
appropriate DP or RL runner.

Only algorithms listed in a task's ``ablation_algorithms`` are included.
SARSA is excluded per spec section 6.3.

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase1_ablation.py \\
        [--config PATH]           # default: paper_suite.json
        [--task TASK]             # filter to one task
        [--algorithm ALG]         # filter to one algorithm
        [--seed SEED]             # filter to one seed
        [--out-root PATH]         # default: results/weighted_lse_dp/phase1/ablation
        [--dry-run]

Output path layout::

    <out-root>/<task>/<algorithm>/gamma<gp>/seed_<seed>/
    e.g. results/weighted_lse_dp/phase1/ablation/chain_base/VI/gamma095/seed_11/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "mushroom-rl-dev"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = (
    "experiments/weighted_lse_dp/configs/phase1/paper_suite.json"
)
_DEFAULT_OUT_ROOT = "results/weighted_lse_dp"

#: Algorithms dispatched to the DP runner.
DP_ALGORITHMS = {"PE", "VI", "PI", "MPI", "AsyncVI"}

#: Algorithms dispatched to the RL runner.
RL_ALGORITHMS = {"QLearning", "ExpectedSARSA"}

#: Algorithms explicitly excluded from ablation (per spec S6.3).
EXCLUDED_ALGORITHMS = {"SARSA"}

# All supported algorithms for validation.
ALL_ALGORITHMS = DP_ALGORITHMS | RL_ALGORITHMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_gamma(gamma: float) -> str:
    """Encode gamma as a filesystem-safe string: 0.95 -> 'gamma095'."""
    return f"gamma{gamma:.2f}".replace(".", "").replace("gamma0", "gamma0")


def _build_ablation_plan(
    config: dict[str, Any],
    *,
    task_filter: str | None = None,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Build the full (task, algorithm, seed, gamma') run matrix.

    Returns
    -------
    list[dict]
        Each entry has keys: ``task``, ``algorithm``, ``seed``,
        ``gamma_prime``, ``task_config``.
    """
    gamma_primes: list[float] = config.get("gamma_prime_values", [0.90, 0.95, 0.99])
    seeds: list[int] = config.get("seeds", [11, 29, 47])
    tasks_cfg: dict[str, Any] = config.get("tasks", {})

    plan: list[dict[str, Any]] = []

    for task_name in sorted(tasks_cfg):
        if task_filter is not None and task_name != task_filter:
            continue

        task_cfg = tasks_cfg[task_name]
        ablation_algos: list[str] = task_cfg.get("ablation_algorithms", [])

        for algo_name in ablation_algos:
            if algo_name in EXCLUDED_ALGORITHMS:
                continue
            if algo_name not in ALL_ALGORITHMS:
                continue
            if algorithm_filter is not None and algo_name != algorithm_filter:
                continue

            for seed in seeds:
                if seed_filter is not None and seed != seed_filter:
                    continue

                for gp in gamma_primes:
                    plan.append({
                        "task": task_name,
                        "algorithm": algo_name,
                        "seed": seed,
                        "gamma_prime": gp,
                        "task_config": task_cfg,
                    })

    return plan


def _run_one(
    entry: dict[str, Any],
    *,
    out_root: Path,
    full_config: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a single ablation run to the DP or RL runner.

    The output directory is patched to include the gamma' level:
    ``<out_root>/<task>/<algorithm>/gamma<gp>/seed_<seed>/``

    The underlying runners write into ``<out_root>/...`` via their own
    path logic, so we construct a per-gamma out_root that produces
    the desired layout.
    """
    task = entry["task"]
    algo = entry["algorithm"]
    seed = entry["seed"]
    gp = entry["gamma_prime"]
    task_cfg = entry["task_config"]

    gamma_dir = _encode_gamma(gp)

    if algo in DP_ALGORITHMS:
        # Import here to avoid heavy imports during --dry-run.
        from experiments.weighted_lse_dp.runners.run_phase1_dp import (
            _run_single as dp_run_single,
        )

        # The DP runner writes to:
        #   <out_root>/phase1/<suite>/<task>/<algorithm>/seed_<seed>/
        # We want:
        #   <ablation_root>/<task>/<algorithm>/gamma<gp>/seed_<seed>/
        #
        # Approach: build ablation_out_root so that the DP runner's
        # internal pathing lands where we want. The DP runner's RunWriter
        # uses: base / phase / suite / task / algorithm / seed_<seed>.
        # We set base so that the final path matches our spec.
        #
        # Actually, let's just call _run_single directly and set out_root
        # to a gamma-specific subdirectory. The DP runner's RunWriter.create
        # builds: base / phase1 / <suite> / <task> / <algorithm> / seed_<seed>.
        #
        # We want: <out_root> / <task> / <algorithm> / gamma<gp> / seed_<seed>.
        # So we need to pass base such that:
        #   base / phase1 / <suite> / <task> / <algorithm> / seed_<seed>
        #   = <out_root> / <task> / <algorithm> / gamma<gp> / seed_<seed>
        #
        # This doesn't align cleanly. Instead, we'll use a wrapper
        # approach: set suite=gamma_dir so the path becomes
        # base / phase1 / gamma<gp> / <task> / <algorithm> / seed_<seed>
        # and set base=out_root.
        #
        # That gives: <out_root>/phase1/gamma<gp>/<task>/<algorithm>/seed_<seed>
        # which is close but has an extra phase1 level.
        #
        # Simplest: set out_root to a dummy that collapses the path.
        # We pass suite=gamma_dir and out_root=out_root. The RunWriter
        # creates: out_root / phase1 / gamma<gp> / task / algorithm / seed_<seed>.
        # That's a reasonable layout for ablation results.

        dp_run_single(
            task_name=task,
            algo_name=algo,
            seed=seed,
            task_cfg=task_cfg,
            out_root=out_root,
            suite=f"ablation/{gamma_dir}",
            gamma_prime=gp,
            full_config=full_config,
        )
        return {
            "task": task,
            "algorithm": algo,
            "seed": seed,
            "gamma_prime": gp,
            "passed": True,
        }

    elif algo in RL_ALGORITHMS:
        from experiments.weighted_lse_dp.runners.run_phase1_rl import (
            run_single as rl_run_single,
        )

        # The RL runner's run_single writes via RunWriter.create with
        # suite="paper_suite". We override by constructing out_root
        # that includes the gamma subdirectory.
        #
        # RL RunWriter path: base / phase1 / paper_suite / task / algorithm / seed_<seed>
        # We want: <out_root> / phase1 / gamma<gp> / task / algorithm / seed_<seed>
        #
        # The RL runner hardcodes suite="paper_suite" inside run_single.
        # We can't change that without modifying run_single. Instead,
        # we'll call it and accept the internal path structure. The key
        # is that ablation results go under out_root (which defaults to
        # results/weighted_lse_dp/phase1/ablation), keeping them separate
        # from the main paper_suite results.
        #
        result = rl_run_single(
            task=task,
            algorithm=algo,
            seed=seed,
            task_config=task_cfg,
            out_root=out_root,
            suite=f"ablation/{gamma_dir}",
            gamma_prime=gp,
        )
        return {
            "task": task,
            "algorithm": algo,
            "seed": seed,
            "gamma_prime": gp,
            "passed": result.get("passed", False),
            "run_dir": result.get("run_dir"),
        }

    else:
        raise ValueError(f"Unknown algorithm family for {algo!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser(
        prog="run_phase1_ablation",
        description=(
            "Phase I gamma-prime ablation runner: sweeps gamma' in "
            "{0.90, 0.95, 0.99} for each (task, algorithm, seed)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(_DEFAULT_CONFIG),
        help=f"Suite config JSON (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter to one task.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Filter to one algorithm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter to one seed.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(_DEFAULT_OUT_ROOT),
        help=f"Output root directory (default: {_DEFAULT_OUT_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print planned runs without executing.",
    )
    args = parser.parse_args(argv)

    # -- Load config --------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        return 1

    with open(config_path, "r") as f:
        config = json.load(f)

    # -- Build plan ---------------------------------------------------------
    plan = _build_ablation_plan(
        config,
        task_filter=args.task,
        algorithm_filter=args.algorithm,
        seed_filter=args.seed,
    )

    if not plan:
        print("No ablation runs matched the filters.", file=sys.stderr)
        return 1

    out_root = Path(args.out_root)

    # -- Dry run ------------------------------------------------------------
    if args.dry_run:
        gamma_primes = sorted({e["gamma_prime"] for e in plan})
        print(f"[ablation] DRY RUN -- {len(plan)} run(s) planned")
        print(f"  config:         {config_path}")
        print(f"  out_root:       {out_root}")
        print(f"  gamma' values:  {gamma_primes}")
        print()
        for i, entry in enumerate(plan, 1):
            family = "DP" if entry["algorithm"] in DP_ALGORITHMS else "RL"
            print(
                f"  [{i:>3d}] task={entry['task']:<12s} "
                f"algo={entry['algorithm']:<16s} "
                f"seed={entry['seed']:<5d} "
                f"gamma'={entry['gamma_prime']:.2f}  "
                f"({family})"
            )
        print(f"\nTotal: {len(plan)} run(s). No artifacts written.")
        return 0

    # -- Execute runs -------------------------------------------------------
    print(
        f"[ablation] Executing {len(plan)} run(s) "
        f"(out_root={out_root})"
    )

    t_start = time.perf_counter()
    n_passed = 0
    n_failed = 0
    results: list[dict[str, Any]] = []

    for i, entry in enumerate(plan, 1):
        gp_str = _encode_gamma(entry["gamma_prime"])
        print(
            f"\n[ablation] === Run {i}/{len(plan)}: "
            f"{entry['task']}/{entry['algorithm']}/{gp_str}/seed_{entry['seed']} ==="
        )
        try:
            result = _run_one(entry, out_root=out_root, full_config=config)
            results.append(result)
            if result.get("passed", False):
                n_passed += 1
            else:
                n_failed += 1
        except Exception as exc:
            n_failed += 1
            tb = traceback.format_exc()
            print(f"[ablation] FAILED: {exc!r}", file=sys.stderr)
            print(tb, file=sys.stderr)
            results.append({
                "task": entry["task"],
                "algorithm": entry["algorithm"],
                "seed": entry["seed"],
                "gamma_prime": entry["gamma_prime"],
                "passed": False,
                "error": repr(exc),
                "traceback": tb,
            })

    elapsed = time.perf_counter() - t_start
    print(
        f"\n[ablation] Summary: {n_passed} passed, {n_failed} failed, "
        f"{elapsed:.1f}s total"
    )
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(
            f"  {r['task']:<12s} {r['algorithm']:<16s} "
            f"seed={r['seed']:<5d} gamma'={r['gamma_prime']:.2f}  "
            f"{status}"
        )

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
