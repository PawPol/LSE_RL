#!/usr/bin/env python
"""Build the four Phase II paper tables (P2-A through P2-D).

Reads task configuration from ``paper_suite.json`` and processed
aggregates from the Phase II results tree.  When no processed data
exists the script emits placeholder LaTeX/CSV tables with correct
column schemas so that the paper build pipeline never breaks.

Output
------
For each table ``P2-{A,B,C,D}`` the script writes:

- ``table_P2{X}.tex``  -- LaTeX snippet (booktabs, NeurIPS style)
- ``table_P2{X}.csv``  -- machine-readable CSV

Confidence intervals: where applicable, ``mean +/- CI`` uses the
percentile bootstrap 95% CI over seeds, produced upstream by the
aggregation script.

No cherry-picking: all groups present in processed summaries are
included.  Any seed filtering is documented in the aggregation script.

CLI
---
::

    python make_phase2_tables.py --out-root results/weighted_lse_dp

"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DP_ALGORITHMS = ("PE", "VI", "PI", "MPI", "AsyncVI")
_RL_ALGORITHMS = ("QLearning", "ExpectedSARSA")
_ALL_ALGORITHMS = _DP_ALGORITHMS + _RL_ALGORITHMS

_ALGO_DISPLAY: dict[str, str] = {
    "PE": "PE",
    "VI": "VI",
    "PI": "PI",
    "MPI": "MPI",
    "AsyncVI": "Async VI",
    "QLearning": "Q-Learning",
    "ExpectedSARSA": "Exp. SARSA",
}

# Canonical task order from the config.
_TASK_ORDER = (
    "chain_sparse_long",
    "chain_jackpot",
    "chain_catastrophe",
    "chain_regime_shift",
    "grid_sparse_goal",
    "grid_hazard",
    "grid_regime_shift",
    "taxi_bonus_shock",
)

_REGIME_SHIFT_TASKS = ("chain_regime_shift", "grid_regime_shift")
_TAIL_TASKS = (
    "chain_jackpot",
    "chain_catastrophe",
    "grid_hazard",
    "taxi_bonus_shock",
)

# Human-readable stress-type descriptions.
_STRESS_TYPE_LABEL: dict[str, str] = {
    "sparse_reward": "Sparse reward",
    "jackpot": "Rare jackpot",
    "catastrophe": "Rare catastrophe",
    "regime_shift": "Regime shift",
    "hazard": "Stochastic hazard",
}

# What each task reduces to at severity=0.
_REDUCES_TO: dict[str, str] = {
    "chain_sparse_long": "chain_base (longer horizon)",
    "chain_jackpot": "chain_base",
    "chain_catastrophe": "chain_base",
    "chain_regime_shift": "chain_base (no shift)",
    "grid_sparse_goal": "grid_base (goal-only reward)",
    "grid_hazard": "grid_base",
    "grid_regime_shift": "grid_base (no shift)",
    "taxi_bonus_shock": "taxi_base",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file; return None if it does not exist."""
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _load_config(out_root: Path) -> dict[str, Any]:
    """Load paper_suite.json configuration."""
    candidates = [
        _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs" / "phase2" / "paper_suite.json",
        out_root / "configs" / "phase2" / "paper_suite.json",
    ]
    for p in candidates:
        if p.is_file():
            with open(p) as f:
                return json.load(f)
    print("WARNING: paper_suite.json not found; using minimal fallback.", file=sys.stderr)
    return {"tasks": {}}


def _load_task_summary(
    aggregated_root: Path,
    task: str,
    algorithm: str,
) -> dict[str, Any] | None:
    """Load ``summary.json`` for a single (task, algorithm) pair."""
    path = aggregated_root / task / algorithm / "summary.json"
    return _load_json(path)


def _load_phase1_summary(
    out_root: Path,
    task: str,
    algorithm: str,
) -> dict[str, Any] | None:
    """Load Phase I summary for baseline comparison."""
    path = out_root / "phase1" / "aggregated" / task / algorithm / "summary.json"
    return _load_json(path)


# ---------------------------------------------------------------------------
# Formatting helpers (LaTeX)
# ---------------------------------------------------------------------------


def _fmt_mean_ci(val: Any, fmt: str = ".3f") -> str:
    """Format a metric dict ``{mean, ci_low, ci_high}`` as LaTeX."""
    if val is None:
        return "--"
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    mean = val.get("mean")
    if mean is None:
        return "--"
    ci_lo = val.get("ci_low")
    ci_hi = val.get("ci_high")
    if ci_lo is not None and ci_hi is not None:
        half = (ci_hi - ci_lo) / 2.0
        return f"{mean:{fmt}} $\\pm$ {half:{fmt}}"
    return f"{mean:{fmt}}"


def _fmt_plain(val: Any, fmt: str = ".3f") -> str:
    """Plain-text variant for CSV."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    mean = val.get("mean")
    if mean is None:
        return ""
    ci_lo = val.get("ci_low")
    ci_hi = val.get("ci_high")
    if ci_lo is not None and ci_hi is not None:
        half = (ci_hi - ci_lo) / 2.0
        return f"{mean:{fmt}} +/- {half:{fmt}}"
    return f"{mean:{fmt}}"


def _get(summary: dict[str, Any] | None, *keys: str) -> Any:
    """Nested dict get returning None on any missing key."""
    cur: Any = summary
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _algo_display(algo: str) -> str:
    return _ALGO_DISPLAY.get(algo, algo)


def _task_tex(task: str) -> str:
    return task.replace("_", r"\_")


# ---------------------------------------------------------------------------
# Table P2-A: Task modifications summary (config-only, no run data)
# ---------------------------------------------------------------------------


def make_table_p2a(config: dict[str, Any], out_dir: Path) -> list[Path]:
    """Generate Table P2-A: task modification summary."""
    tasks = config.get("tasks", {})

    # -- LaTeX -----------------------------------------------------------------
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Phase II stress-task modifications.  "
        r"Severity $= 0$ reduces each task to its Phase I base.}",
        r"\label{tab:p2a}",
        r"\small",
        r"\begin{tabular}{llcccp{3.5cm}}",
        r"\toprule",
        r"Task family & Stress type & Severity param & States & Horizon "
        r"& Reduces to \\",
        r"\midrule",
    ]

    csv_rows: list[list[str]] = []
    csv_rows.append([
        "task_family", "stress_type", "severity_param", "states", "horizon",
        "reduces_to",
    ])

    for task_name in _TASK_ORDER:
        t = tasks.get(task_name, {})
        stress = t.get("stress_type", t.get("severity", ""))
        stress_label = _STRESS_TYPE_LABEL.get(stress, stress)
        n_states = t.get("n_states", t.get("state_n", "?"))
        horizon = t.get("horizon", "?")

        # Severity parameter description.
        sev_parts = []
        for key in ("jackpot_prob", "jackpot_reward", "risky_prob",
                     "catastrophe_reward", "hazard_prob", "hazard_reward",
                     "bonus_prob", "bonus_reward", "change_at_episode"):
            v = t.get(key)
            if v is not None:
                sev_parts.append(f"{key}={v}")
        severity_param = "; ".join(sev_parts) if sev_parts else stress

        reduces_to = _REDUCES_TO.get(task_name, "base")

        # Escape underscores for LaTeX.
        sev_tex = severity_param.replace("_", r"\_")
        reduces_tex = reduces_to.replace("_", r"\_")

        tex_lines.append(
            f"  {_task_tex(task_name)} & {stress_label} & "
            f"{sev_tex} & {n_states} & {horizon} & "
            f"{reduces_tex} \\\\"
        )
        csv_rows.append([
            task_name, stress_label, severity_param,
            str(n_states), str(horizon), reduces_to,
        ])

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _write_table(out_dir, "P2A", "\n".join(tex_lines) + "\n", csv_rows)


# ---------------------------------------------------------------------------
# Table P2-B: DP re-planning stats for regime-shift tasks
# ---------------------------------------------------------------------------


def make_table_p2b(
    config: dict[str, Any],
    aggregated_root: Path,
    out_dir: Path,
) -> list[Path]:
    """Generate Table P2-B: DP warm-start re-planning after regime shift."""
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{DP re-planning after regime shift.  "
        r"Warm-start speedup is ratio of cold-start to warm-start "
        r"iteration counts.}",
        r"\label{tab:p2b}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & Algorithm & Pre-shift $V^*$ range & Post-shift $V^*$ range "
        r"& Iter ratio & Wall-clock ratio \\",
        r"\midrule",
    ]

    csv_rows: list[list[str]] = []
    csv_rows.append([
        "task", "algorithm", "pre_shift_v_range", "post_shift_v_range",
        "iter_ratio", "wall_clock_ratio",
    ])

    has_data = False
    for task in _REGIME_SHIFT_TASKS:
        for algo in _DP_ALGORITHMS:
            summary = _load_task_summary(aggregated_root, task, algo)
            if summary is None:
                continue
            has_data = True

            pre_range = _get(summary, "scalar_metrics", "pre_shift_v_range")
            post_range = _get(summary, "scalar_metrics", "post_shift_v_range")
            iter_ratio = _get(summary, "scalar_metrics", "warmstart_iter_ratio")
            wall_ratio = _get(summary, "scalar_metrics", "warmstart_wall_ratio")

            tex_lines.append(
                f"  {_task_tex(task)} & {_algo_display(algo)} "
                f"& {_fmt_mean_ci(pre_range)} & {_fmt_mean_ci(post_range)} "
                f"& {_fmt_mean_ci(iter_ratio, fmt='.2f')} "
                f"& {_fmt_mean_ci(wall_ratio, fmt='.2f')} \\\\"
            )
            csv_rows.append([
                task, algo,
                _fmt_plain(pre_range), _fmt_plain(post_range),
                _fmt_plain(iter_ratio, fmt=".2f"),
                _fmt_plain(wall_ratio, fmt=".2f"),
            ])

    if not has_data:
        tex_lines.append(
            r"  \multicolumn{6}{c}{\emph{No regime-shift run data available yet.}} \\"
        )
        csv_rows.append(["(no data)", "", "", "", "", ""])

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _write_table(out_dir, "P2B", "\n".join(tex_lines) + "\n", csv_rows)


# ---------------------------------------------------------------------------
# Table P2-C: RL degradation on stress tasks
# ---------------------------------------------------------------------------


def make_table_p2c(
    config: dict[str, Any],
    aggregated_root: Path,
    phase1_root: Path,
    out_dir: Path,
) -> list[Path]:
    """Generate Table P2-C: RL degradation vs Phase I base."""
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Phase II classical RL performance on stress tasks.  "
        r"Delta is difference vs.\ Phase I base.  "
        r"Percentile bootstrap 95\% CI over seeds.}",
        r"\label{tab:p2c}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Task & Algorithm & Mean return & Std & IQR "
        r"& $\Delta$ vs base & Event rate \\",
        r"\midrule",
    ]

    csv_rows: list[list[str]] = []
    csv_rows.append([
        "task", "algorithm", "mean_return", "std", "iqr",
        "delta_vs_base", "event_rate",
    ])

    # Map stress task -> Phase I base task name.
    _base_task_map: dict[str, str] = {
        "chain_sparse_long": "chain_base",
        "chain_jackpot": "chain_base",
        "chain_catastrophe": "chain_base",
        "chain_regime_shift": "chain_base",
        "grid_sparse_goal": "grid_base",
        "grid_hazard": "grid_base",
        "grid_regime_shift": "grid_base",
        "taxi_bonus_shock": "taxi_base",
    }

    has_data = False
    prev_task = None
    for task in _TASK_ORDER:
        for algo in _RL_ALGORITHMS:
            summary = _load_task_summary(aggregated_root, task, algo)
            if summary is None:
                continue
            has_data = True

            sm = summary.get("scalar_metrics", {})
            mean_ret = _get(sm, "final_disc_return_mean")
            std_ret = _get(sm, "final_disc_return_std")
            iqr_ret = _get(sm, "iqr_disc_return")
            event_rate = _get(sm, "event_rate")

            # Compute delta vs Phase I base.
            delta_str_tex = "--"
            delta_str_csv = ""
            base_task = _base_task_map.get(task)
            if base_task is not None:
                base_summary = _load_phase1_summary(phase1_root, base_task, algo)
                base_mean = _get(base_summary, "scalar_metrics",
                                 "final_disc_return_mean", "mean")
                stress_mean = mean_ret.get("mean") if isinstance(mean_ret, dict) else mean_ret
                if base_mean is not None and stress_mean is not None:
                    delta = stress_mean - base_mean
                    delta_str_tex = f"{delta:+.3f}"
                    delta_str_csv = f"{delta:+.3f}"

            task_col = _task_tex(task) if task != prev_task else ""
            prev_task = task

            tex_lines.append(
                f"  {task_col} & {_algo_display(algo)} "
                f"& {_fmt_mean_ci(mean_ret)} & {_fmt_mean_ci(std_ret)} "
                f"& {_fmt_mean_ci(iqr_ret)} & {delta_str_tex} "
                f"& {_fmt_mean_ci(event_rate, fmt='.3f')} \\\\"
            )
            csv_rows.append([
                task, algo,
                _fmt_plain(mean_ret), _fmt_plain(std_ret),
                _fmt_plain(iqr_ret), delta_str_csv,
                _fmt_plain(event_rate, fmt=".3f"),
            ])

        if task != _TASK_ORDER[-1] and has_data:
            tex_lines.append(r"\midrule")

    if not has_data:
        tex_lines.append(
            r"  \multicolumn{7}{c}{\emph{No Phase II RL run data available yet.}} \\"
        )
        csv_rows.append(["(no data)", "", "", "", "", "", ""])

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _write_table(out_dir, "P2C", "\n".join(tex_lines) + "\n", csv_rows)


# ---------------------------------------------------------------------------
# Table P2-D: Tail metrics
# ---------------------------------------------------------------------------


def make_table_p2d(
    config: dict[str, Any],
    aggregated_root: Path,
    out_dir: Path,
) -> list[Path]:
    """Generate Table P2-D: tail-risk metrics for jackpot/catastrophe/hazard tasks."""
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Tail-risk metrics on stress tasks with rare events.  "
        r"Percentile bootstrap 95\% CI over seeds.}",
        r"\label{tab:p2d}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Task & Algorithm & CVaR-5\% & CVaR-10\% "
        r"& Top-10\% mean & Event rate & Event-cond.\ return \\",
        r"\midrule",
    ]

    csv_rows: list[list[str]] = []
    csv_rows.append([
        "task", "algorithm", "cvar_5pct", "cvar_10pct",
        "top_10pct_mean", "event_rate", "event_cond_return",
    ])

    has_data = False
    prev_task = None
    for task in _TAIL_TASKS:
        for algo in _RL_ALGORITHMS:
            summary = _load_task_summary(aggregated_root, task, algo)
            if summary is None:
                continue
            has_data = True

            sm = summary.get("scalar_metrics", {})
            cvar5 = _get(sm, "cvar_5pct")
            cvar10 = _get(sm, "cvar_10pct")
            top10 = _get(sm, "top_10pct_mean")
            event_rate = _get(sm, "event_rate")
            event_cond = _get(sm, "event_cond_return")

            task_col = _task_tex(task) if task != prev_task else ""
            prev_task = task

            tex_lines.append(
                f"  {task_col} & {_algo_display(algo)} "
                f"& {_fmt_mean_ci(cvar5)} & {_fmt_mean_ci(cvar10)} "
                f"& {_fmt_mean_ci(top10)} & {_fmt_mean_ci(event_rate, fmt='.3f')} "
                f"& {_fmt_mean_ci(event_cond)} \\\\"
            )
            csv_rows.append([
                task, algo,
                _fmt_plain(cvar5), _fmt_plain(cvar10),
                _fmt_plain(top10), _fmt_plain(event_rate, fmt=".3f"),
                _fmt_plain(event_cond),
            ])

        if task != _TAIL_TASKS[-1] and has_data:
            tex_lines.append(r"\midrule")

    if not has_data:
        tex_lines.append(
            r"  \multicolumn{7}{c}{\emph{No tail-risk run data available yet.}} \\"
        )
        csv_rows.append(["(no data)", "", "", "", "", "", ""])

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _write_table(out_dir, "P2D", "\n".join(tex_lines) + "\n", csv_rows)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_table(
    out_dir: Path,
    table_id: str,
    tex_content: str,
    csv_rows: list[list[str]],
) -> list[Path]:
    """Write a table as .tex and .csv; return list of written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    tex_path = out_dir / f"table_{table_id}.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    paths.append(tex_path)

    csv_path = out_dir / f"table_{table_id}.csv"
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in csv_rows:
        writer.writerow(row)
    csv_path.write_text(buf.getvalue(), encoding="utf-8")
    paths.append(csv_path)

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase II paper tables (P2-A through P2-D).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/weighted_lse_dp"),
        help=(
            "Root of the weighted_lse_dp results tree.  Tables are written to "
            "<out-root>/processed/phase2/tables/. "
            "Default: results/weighted_lse_dp"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to paper_suite.json. Default: auto-detected from "
            "experiments/weighted_lse_dp/configs/phase2/paper_suite.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: build all four Phase II tables."""
    args = _parse_args(argv)

    out_root = Path(args.out_root)
    out_dir = out_root / "processed" / "phase2" / "tables"
    aggregated_root = out_root / "phase2" / "aggregated"

    # Load config.
    if args.config is not None:
        config = _load_json(args.config) or {}
    else:
        config = _load_config(out_root)
    print(f"Phase II table generation")
    print(f"  out_dir:          {out_dir}")
    print(f"  aggregated_root:  {aggregated_root}")
    print()

    all_written: list[Path] = []

    # P2-A: task modification summary (config only).
    written = make_table_p2a(config, out_dir)
    all_written.extend(written)
    for p in written:
        print(f"  wrote {p}")

    # P2-B: DP re-planning stats.
    written = make_table_p2b(config, aggregated_root, out_dir)
    all_written.extend(written)
    for p in written:
        print(f"  wrote {p}")

    # P2-C: RL degradation.
    written = make_table_p2c(config, aggregated_root, out_root, out_dir)
    all_written.extend(written)
    for p in written:
        print(f"  wrote {p}")

    # P2-D: tail metrics.
    written = make_table_p2d(config, aggregated_root, out_dir)
    all_written.extend(written)
    for p in written:
        print(f"  wrote {p}")

    print(f"\nDone. {len(all_written)} files written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
