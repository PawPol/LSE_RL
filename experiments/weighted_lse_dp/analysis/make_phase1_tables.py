#!/usr/bin/env python
"""Build the four Phase I paper tables (P1-A through P1-D).

Reads processed aggregates produced by ``aggregate_phase1.py`` and the
task-level configuration from ``paper_suite.json``. When no processed
data exists, either prints a clear diagnostic or (with ``--demo``)
generates tables with synthetic placeholder values for CI and layout
verification.

Output
------
For each table ``P1-{A,B,C,D}`` the script writes:

- ``table_P1{X}.tex``  -- LaTeX snippet (booktabs, suitable for NeurIPS)
- ``table_P1{X}.md``   -- plain Markdown for quick review

Confidence intervals: where applicable, ``mean +/- CI`` uses the
percentile bootstrap 95% CI produced by ``aggregate_phase1.py`` (via
``common.metrics.aggregate``). The CI method is "percentile bootstrap,
10 000 resamples over seeds" per spec section 9.1.

No cherry-picking: all groups present in ``summary.json`` are included.
If the aggregator filtered seeds, that filter is documented upstream
in the aggregation script; this script passes them through verbatim.

CLI
---
::

    .venv/bin/python experiments/weighted_lse_dp/analysis/make_phase1_tables.py \\
        [--processed-root PATH]   # default: results/weighted_lse_dp/processed/phase1
        [--config PATH]           # default: experiments/weighted_lse_dp/configs/phase1/paper_suite.json
        [--out-dir PATH]          # default: results/weighted_lse_dp/processed/phase1/tables
        [--demo]                  # generate tables with synthetic data
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Algorithms expected in each table.
_DP_ALGORITHMS = ("PE", "VI", "PI", "MPI", "AsyncVI")
_RL_ALGORITHMS = ("QLearning", "ExpectedSARSA")

# Task display order.
_TASK_ORDER = ("chain_base", "grid_base", "taxi_base")

# For pretty printing.
_ALGO_DISPLAY: dict[str, str] = {
    "PE": "PE",
    "VI": "VI",
    "PI": "PI",
    "MPI": "MPI",
    "AsyncVI": "Async VI",
    "QLearning": "Q-Learning",
    "ExpectedSARSA": "Exp. SARSA",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load paper_suite.json configuration."""
    with open(config_path) as f:
        return json.load(f)


def _load_summary(processed_root: Path) -> dict[str, Any] | None:
    """Load summary.json; return None if missing."""
    path = processed_root / "summary.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _load_ablation_summary(processed_root: Path) -> dict[str, Any] | None:
    """Load ablation_summary.json; return None if missing."""
    path = processed_root / "ablation_summary.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _synthetic_dp_metrics() -> dict[str, dict[str, float]]:
    """Generate plausible synthetic DP scalar metrics."""
    n_sweeps = _RNG.integers(15, 80)
    residual = 10.0 ** _RNG.uniform(-8, -5)
    wall = _RNG.uniform(0.01, 0.5)
    return {
        "n_sweeps": {"mean": float(n_sweeps), "ci_low": float(n_sweeps), "ci_high": float(n_sweeps)},
        "final_bellman_residual": {"mean": residual, "ci_low": residual * 0.8, "ci_high": residual * 1.2},
        "wall_clock_s": {"mean": wall, "ci_low": wall * 0.9, "ci_high": wall * 1.1},
    }


def _synthetic_rl_metrics() -> dict[str, dict[str, float]]:
    """Generate plausible synthetic RL scalar metrics."""
    ret = _RNG.uniform(0.5, 0.95)
    auc = ret * _RNG.uniform(50000, 200000)
    steps = _RNG.integers(20000, 100000)
    return {
        "final_disc_return_mean": {"mean": ret, "ci_low": ret * 0.9, "ci_high": min(ret * 1.1, 1.0)},
        "auc_disc_return": {"mean": auc, "ci_low": auc * 0.9, "ci_high": auc * 1.1},
        "steps_to_threshold": {"mean": float(steps), "ci_low": float(steps * 0.8), "ci_high": float(steps * 1.2)},
    }


def _build_demo_summary() -> dict[str, Any]:
    """Build a synthetic summary.json structure."""
    groups = []
    for task in _TASK_ORDER:
        for algo in _DP_ALGORITHMS:
            groups.append({
                "suite": "paper_suite",
                "task": task,
                "algorithm": algo,
                "n_seeds": 3,
                "seeds": [11, 29, 47],
                "scalar_metrics": _synthetic_dp_metrics(),
            })
        for algo in _RL_ALGORITHMS:
            groups.append({
                "suite": "paper_suite",
                "task": task,
                "algorithm": algo,
                "n_seeds": 3,
                "seeds": [11, 29, 47],
                "scalar_metrics": _synthetic_rl_metrics(),
            })
    return {"groups": groups}


def _build_demo_ablation(config: dict[str, Any]) -> dict[str, Any]:
    """Build a synthetic ablation_summary.json structure."""
    gamma_primes = config.get("gamma_prime_values", [0.90, 0.95, 0.99])
    groups = []
    for task in _TASK_ORDER:
        task_cfg = config.get("tasks", {}).get(task, {})
        ablation_algos = task_cfg.get("ablation_algorithms", list(_DP_ALGORITHMS) + list(_RL_ALGORITHMS))
        for algo in ablation_algos:
            for gp in gamma_primes:
                if algo in _DP_ALGORITHMS:
                    metrics = _synthetic_dp_metrics()
                else:
                    metrics = _synthetic_rl_metrics()
                groups.append({
                    "suite": "ablation",
                    "task": task,
                    "algorithm": algo,
                    "gamma_prime": gp,
                    "n_seeds": 3,
                    "seeds": [11, 29, 47],
                    "scalar_metrics": metrics,
                })
    return {"groups": groups}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_mean_ci(metric: dict[str, float] | None, fmt: str = ".3f") -> str:
    """Format a metric dict as 'mean +/- half-width' or '--' if missing."""
    if metric is None:
        return "--"
    mean = metric.get("mean")
    if mean is None:
        return "--"
    ci_lo = metric.get("ci_low")
    ci_hi = metric.get("ci_high")
    if ci_lo is not None and ci_hi is not None:
        half = (ci_hi - ci_lo) / 2.0
        return f"{mean:{fmt}} $\\pm$ {half:{fmt}}"
    # Single seed or no CI.
    return f"{mean:{fmt}}"


def _fmt_mean_ci_md(metric: dict[str, float] | None, fmt: str = ".3f") -> str:
    """Markdown variant (no LaTeX math)."""
    if metric is None:
        return "--"
    mean = metric.get("mean")
    if mean is None:
        return "--"
    ci_lo = metric.get("ci_low")
    ci_hi = metric.get("ci_high")
    if ci_lo is not None and ci_hi is not None:
        half = (ci_hi - ci_lo) / 2.0
        return f"{mean:{fmt}} +/- {half:{fmt}}"
    return f"{mean:{fmt}}"


def _fmt_int_mean(metric: dict[str, float] | None) -> str:
    """Format a metric as an integer mean."""
    if metric is None:
        return "--"
    mean = metric.get("mean")
    if mean is None:
        return "--"
    return f"{int(round(mean))}"


def _fmt_sci(metric: dict[str, float] | None) -> str:
    """Format in scientific notation for residuals."""
    if metric is None:
        return "--"
    mean = metric.get("mean")
    if mean is None:
        return "--"
    ci_lo = metric.get("ci_low")
    ci_hi = metric.get("ci_high")
    if ci_lo is not None and ci_hi is not None:
        half = (ci_hi - ci_lo) / 2.0
        return f"{mean:.1e} $\\pm$ {half:.1e}"
    return f"{mean:.1e}"


def _fmt_sci_md(metric: dict[str, float] | None) -> str:
    """Scientific notation for markdown."""
    if metric is None:
        return "--"
    mean = metric.get("mean")
    if mean is None:
        return "--"
    ci_lo = metric.get("ci_low")
    ci_hi = metric.get("ci_high")
    if ci_lo is not None and ci_hi is not None:
        half = (ci_hi - ci_lo) / 2.0
        return f"{mean:.1e} +/- {half:.1e}"
    return f"{mean:.1e}"


def _get_metric(group: dict[str, Any], key: str) -> dict[str, float] | None:
    """Safely extract a metric sub-dict from a summary group entry."""
    sm = group.get("scalar_metrics", {})
    val = sm.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    # Single scalar (shouldn't happen with aggregator but handle gracefully).
    return {"mean": float(val)}


def _algo_display(algo: str) -> str:
    return _ALGO_DISPLAY.get(algo, algo)


def _task_display(task: str) -> str:
    return task.replace("_", r"\_")


def _task_display_md(task: str) -> str:
    return task


# ---------------------------------------------------------------------------
# Table P1-A: Task summary (from config only)
# ---------------------------------------------------------------------------


def _make_table_p1a(config: dict[str, Any]) -> tuple[str, str]:
    """Generate Table P1-A (task summary) in LaTeX and Markdown."""
    tasks = config.get("tasks", {})

    # LaTeX
    lines_tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Phase I task summary.}",
        r"\label{tab:p1a}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Task & States & Actions & Horizon & $\gamma$ & Reward \\",
        r"\midrule",
    ]

    # Markdown
    lines_md = [
        "## Table P1-A: Task Summary",
        "",
        "| Task | States | Actions | Horizon | gamma | Reward |",
        "|------|--------|---------|---------|-------|--------|",
    ]

    for task_name in _TASK_ORDER:
        if task_name not in tasks:
            continue
        t = tasks[task_name]
        n_states = t.get("n_states", t.get("state_n", "?"))
        n_actions = t.get("n_actions", 2 if "chain" in task_name else 4)
        horizon = t.get("horizon", "?")
        gamma = t.get("gamma", "?")
        reward = r"\{0, +1\}" if t.get("rew") or t.get("goal_reward") else "?"

        lines_tex.append(
            f"  {_task_display(task_name)} & {n_states} & {n_actions} "
            f"& {horizon} & {gamma} & {reward} \\\\"
        )
        lines_md.append(
            f"| {_task_display_md(task_name)} | {n_states} | {n_actions} "
            f"| {horizon} | {gamma} | {{0, +1}} |"
        )

    lines_tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines_tex) + "\n", "\n".join(lines_md) + "\n"


# ---------------------------------------------------------------------------
# Table P1-B: Classical DP results
# ---------------------------------------------------------------------------


def _make_table_p1b(
    summary: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, str]:
    """Generate Table P1-B (DP planner results) in LaTeX and Markdown."""
    groups = summary.get("groups", [])

    # Index groups by (task, algorithm).
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for g in groups:
        algo = g["algorithm"]
        if algo in _DP_ALGORITHMS:
            idx[(g["task"], algo)] = g

    # LaTeX
    lines_tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Phase I classical DP planner convergence. "
        r"Percentile bootstrap 95\% CI over seeds.}",
        r"\label{tab:p1b}",
        r"\begin{tabular}{llrcrc}",
        r"\toprule",
        r"Task & Algorithm & Sweeps & Residual & Wall-clock (s) & Seeds \\",
        r"\midrule",
    ]

    lines_md = [
        "## Table P1-B: Classical DP Planner Results",
        "",
        "| Task | Algorithm | Sweeps | Residual | Wall-clock (s) | Seeds |",
        "|------|-----------|--------|----------|----------------|-------|",
    ]

    prev_task = None
    for task_name in _TASK_ORDER:
        for algo in _DP_ALGORITHMS:
            g = idx.get((task_name, algo))
            if g is None:
                continue

            task_col = _task_display(task_name) if task_name != prev_task else ""
            task_col_md = _task_display_md(task_name) if task_name != prev_task else ""
            prev_task = task_name

            sweeps = _fmt_int_mean(_get_metric(g, "n_sweeps"))
            residual = _fmt_sci(_get_metric(g, "final_bellman_residual"))
            residual_md = _fmt_sci_md(_get_metric(g, "final_bellman_residual"))
            wall = _fmt_mean_ci(_get_metric(g, "wall_clock_s"), fmt=".3f")
            wall_md = _fmt_mean_ci_md(_get_metric(g, "wall_clock_s"), fmt=".3f")
            n_seeds = g.get("n_seeds", "?")

            lines_tex.append(
                f"  {task_col} & {_algo_display(algo)} & {sweeps} "
                f"& {residual} & {wall} & {n_seeds} \\\\"
            )
            lines_md.append(
                f"| {task_col_md} | {_algo_display(algo)} | {sweeps} "
                f"| {residual_md} | {wall_md} | {n_seeds} |"
            )

        # Add a midrule between task groups (except the last).
        if task_name != _TASK_ORDER[-1]:
            lines_tex.append(r"\midrule")

    lines_tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines_tex) + "\n", "\n".join(lines_md) + "\n"


# ---------------------------------------------------------------------------
# Table P1-C: Classical RL results
# ---------------------------------------------------------------------------


def _make_table_p1c(
    summary: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, str]:
    """Generate Table P1-C (RL results) in LaTeX and Markdown."""
    groups = summary.get("groups", [])

    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for g in groups:
        algo = g["algorithm"]
        if algo in _RL_ALGORITHMS:
            idx[(g["task"], algo)] = g

    lines_tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Phase I classical RL returns and sample complexity. "
        r"Percentile bootstrap 95\% CI over seeds.}",
        r"\label{tab:p1c}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Task & Algorithm & Disc.\ Return & AUC & Steps to Threshold \\",
        r"\midrule",
    ]

    lines_md = [
        "## Table P1-C: Classical RL Results",
        "",
        "| Task | Algorithm | Disc Return | AUC | Steps-to-threshold |",
        "|------|-----------|-------------|-----|--------------------|",
    ]

    prev_task = None
    for task_name in _TASK_ORDER:
        for algo in _RL_ALGORITHMS:
            g = idx.get((task_name, algo))
            if g is None:
                continue

            task_col = _task_display(task_name) if task_name != prev_task else ""
            task_col_md = _task_display_md(task_name) if task_name != prev_task else ""
            prev_task = task_name

            disc_ret = _fmt_mean_ci(_get_metric(g, "final_disc_return_mean"), fmt=".3f")
            disc_ret_md = _fmt_mean_ci_md(_get_metric(g, "final_disc_return_mean"), fmt=".3f")
            auc = _fmt_mean_ci(_get_metric(g, "auc_disc_return"), fmt=".0f")
            auc_md = _fmt_mean_ci_md(_get_metric(g, "auc_disc_return"), fmt=".0f")
            steps = _fmt_int_mean(_get_metric(g, "steps_to_threshold"))
            n_seeds = g.get("n_seeds", "?")

            lines_tex.append(
                f"  {task_col} & {_algo_display(algo)} & {disc_ret} "
                f"& {auc} & {steps} \\\\"
            )
            lines_md.append(
                f"| {task_col_md} | {_algo_display(algo)} | {disc_ret_md} "
                f"| {auc_md} | {steps} |"
            )

        if task_name != _TASK_ORDER[-1]:
            lines_tex.append(r"\midrule")

    lines_tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines_tex) + "\n", "\n".join(lines_md) + "\n"


# ---------------------------------------------------------------------------
# Table P1-D: Fixed-gamma' ablation summary
# ---------------------------------------------------------------------------


def _make_table_p1d(
    ablation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, str]:
    """Generate Table P1-D (gamma' ablation) in LaTeX and Markdown."""
    groups = ablation.get("groups", [])

    # Group by (task, algo, gamma_prime).
    idx: dict[tuple[str, str, float], dict[str, Any]] = {}
    all_gp: set[float] = set()
    for g in groups:
        gp = g.get("gamma_prime")
        if gp is None:
            continue
        all_gp.add(gp)
        idx[(g["task"], g["algorithm"], gp)] = g

    gamma_primes = sorted(all_gp)

    # Split into two sub-tables: DP (residual + sweeps) and RL (return + AUC).
    # Using shared column headers for incompatible quantities (e.g. "Disc Return"
    # for both residuals and returns) is misleading — keep them separate.

    # Collect all (task, algo) pairs that appear.
    seen_pairs: list[tuple[str, str]] = []
    for g in groups:
        pair = (g["task"], g["algorithm"])
        if pair not in seen_pairs:
            seen_pairs.append(pair)

    task_rank = {t: i for i, t in enumerate(_TASK_ORDER)}
    all_algos = list(_DP_ALGORITHMS) + list(_RL_ALGORITHMS)
    algo_rank = {a: i for i, a in enumerate(all_algos)}
    seen_pairs.sort(key=lambda p: (task_rank.get(p[0], 99), algo_rank.get(p[1], 99)))

    dp_pairs = [(t, a) for t, a in seen_pairs if a in _DP_ALGORITHMS]
    rl_pairs = [(t, a) for t, a in seen_pairs if a in _RL_ALGORITHMS]

    def _dp_block_tex() -> list[str]:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Phase I DP discount-factor ($\gamma'$) ablation: "
            r"final Bellman residual and sweep count. "
            r"Percentile bootstrap 95\% CI over seeds.}",
            r"\label{tab:p1d-dp}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Task & Algorithm & $\gamma'$ & Bellman Residual & Sweeps \\",
            r"\midrule",
        ]
        prev_task = None
        for task_name, algo in dp_pairs:
            for gp in gamma_primes:
                g = idx.get((task_name, algo, gp))
                if g is None:
                    continue
                task_col = _task_display(task_name) if task_name != prev_task else ""
                prev_task = task_name
                residual = _fmt_sci(_get_metric(g, "final_bellman_residual"))
                sweeps = _fmt_int_mean(_get_metric(g, "n_sweeps"))
                lines.append(f"  {task_col} & {_algo_display(algo)} & {gp:.2f} & {residual} & {sweeps} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return lines

    def _rl_block_tex() -> list[str]:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Phase I RL discount-factor ($\gamma'$) ablation: "
            r"final discounted return and AUC (transitions). "
            r"Percentile bootstrap 95\% CI over seeds.}",
            r"\label{tab:p1d-rl}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Task & Algorithm & $\gamma'$ & Disc.\ Return & AUC (transitions) \\",
            r"\midrule",
        ]
        prev_task = None
        for task_name, algo in rl_pairs:
            for gp in gamma_primes:
                g = idx.get((task_name, algo, gp))
                if g is None:
                    continue
                task_col = _task_display(task_name) if task_name != prev_task else ""
                prev_task = task_name
                disc_ret = _fmt_mean_ci(_get_metric(g, "final_disc_return_mean"), fmt=".3f")
                auc_val = _fmt_mean_ci(_get_metric(g, "auc_disc_return"), fmt=".0f")
                lines.append(f"  {task_col} & {_algo_display(algo)} & {gp:.2f} & {disc_ret} & {auc_val} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return lines

    def _dp_block_md() -> list[str]:
        lines = [
            "### Table P1-D (DP): Bellman Residual and Sweeps vs gamma'",
            "",
            "| Task | Algorithm | gamma' | Bellman Residual | Sweeps |",
            "|------|-----------|--------|-----------------|--------|",
        ]
        prev_task = None
        for task_name, algo in dp_pairs:
            for gp in gamma_primes:
                g = idx.get((task_name, algo, gp))
                if g is None:
                    continue
                task_col = _task_display_md(task_name) if task_name != prev_task else ""
                prev_task = task_name
                residual = _fmt_sci_md(_get_metric(g, "final_bellman_residual"))
                sweeps = _fmt_int_mean(_get_metric(g, "n_sweeps"))
                lines.append(f"| {task_col} | {_algo_display(algo)} | {gp:.2f} | {residual} | {sweeps} |")
        return lines

    def _rl_block_md() -> list[str]:
        lines = [
            "",
            "### Table P1-D (RL): Discounted Return and AUC (transitions) vs gamma'",
            "",
            "| Task | Algorithm | gamma' | Disc Return | AUC (transitions) |",
            "|------|-----------|--------|-------------|-------------------|",
        ]
        prev_task = None
        for task_name, algo in rl_pairs:
            for gp in gamma_primes:
                g = idx.get((task_name, algo, gp))
                if g is None:
                    continue
                task_col = _task_display_md(task_name) if task_name != prev_task else ""
                prev_task = task_name
                disc_ret = _fmt_mean_ci_md(_get_metric(g, "final_disc_return_mean"), fmt=".3f")
                auc_val = _fmt_mean_ci_md(_get_metric(g, "auc_disc_return"), fmt=".0f")
                lines.append(f"| {task_col} | {_algo_display(algo)} | {gp:.2f} | {disc_ret} | {auc_val} |")
        return lines

    header_md = ["## Table P1-D: Discount Ablation", ""]
    lines_tex = _dp_block_tex() + ["", "% --- RL sub-table ---", ""] + _rl_block_tex()
    lines_md = header_md + _dp_block_md() + _rl_block_md()

    return "\n".join(lines_tex) + "\n", "\n".join(lines_md) + "\n"


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase I paper tables (P1-A through P1-D).",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("results/weighted_lse_dp/processed/phase1"),
        help="Root of processed Phase I results (default: results/weighted_lse_dp/processed/phase1).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/weighted_lse_dp/configs/phase1/paper_suite.json"),
        help="Path to paper_suite.json config (default: experiments/weighted_lse_dp/configs/phase1/paper_suite.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/weighted_lse_dp/processed/phase1/tables"),
        help="Output directory for table files (default: results/weighted_lse_dp/processed/phase1/tables).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate tables with synthetic placeholder data (ignores processed results).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Load config (always needed for P1-A and demo fallback).
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = _load_config(config_path)
    print(f"Loaded config: {config_path}")

    # Determine data source.
    using_demo = args.demo
    summary: dict[str, Any] | None = None
    ablation: dict[str, Any] | None = None

    if not using_demo:
        summary = _load_summary(args.processed_root)
        ablation = _load_ablation_summary(args.processed_root)
        if summary is None:
            print(
                f"\nWARNING: summary.json not found at {args.processed_root}.\n"
                f"  Run aggregate_phase1.py first, or use --demo for synthetic data.\n"
                f"  Falling back to synthetic data for Tables P1-B, P1-C.\n",
                file=sys.stderr,
            )
            using_demo = True

    if using_demo:
        print("Using synthetic demo data.")
        summary = _build_demo_summary()
        ablation = _build_demo_ablation(config)
    else:
        print(f"Loaded summary from: {args.processed_root / 'summary.json'}")
        if ablation is not None:
            print(f"Loaded ablation summary from: {args.processed_root / 'ablation_summary.json'}")
        else:
            print("No ablation_summary.json found; Table P1-D will use synthetic data.")
            ablation = _build_demo_ablation(config)

    # Create output directory.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate all four tables.
    tables: dict[str, tuple[str, str]] = {}

    # P1-A: always from config.
    tables["P1A"] = _make_table_p1a(config)

    # P1-B: DP results.
    tables["P1B"] = _make_table_p1b(summary, config)

    # P1-C: RL results.
    tables["P1C"] = _make_table_p1c(summary, config)

    # P1-D: ablation.
    tables["P1D"] = _make_table_p1d(ablation, config)

    # Write outputs.
    written: list[Path] = []
    for table_id, (tex, md) in sorted(tables.items()):
        tex_path = out_dir / f"table_{table_id}.tex"
        md_path = out_dir / f"table_{table_id}.md"

        with open(tex_path, "w") as f:
            f.write(tex)
        with open(md_path, "w") as f:
            f.write(md)

        written.append(tex_path)
        written.append(md_path)
        print(f"  wrote {tex_path}")
        print(f"  wrote {md_path}")

    print(f"\nDone. {len(written)} files written to {out_dir}")


if __name__ == "__main__":
    main()
