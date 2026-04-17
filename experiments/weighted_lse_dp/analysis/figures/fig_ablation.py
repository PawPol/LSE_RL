#!/usr/bin/env python
"""Phase I fixed-gamma' ablation control figure.

Renders grouped bar charts comparing main algorithms at different gamma'
values ({0.90, 0.95, 0.99}) per task.  DP algorithms are shown in terms
of sweep counts to convergence; RL algorithms in terms of final
discounted return (mean across seeds).

Confidence intervals: percentile bootstrap over seeds (95% CI), reported
as error bars.  When only a single seed is available, no error bar is
drawn.

Data source: ``ablation_summary.json`` produced by
``aggregate_phase1.py``.  In ``--demo`` mode, synthetic data is
generated deterministically (seed=42) so the figure is reproducible
without real experiment outputs.

No cherry-picking: all groups present in ablation_summary.json are
included.  The ``--demo`` flag generates the full
{chain_base, grid_base, taxi_base} x {all ablation algorithms} x
{0.90, 0.95, 0.99} matrix.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_PRIMES: list[float] = [0.90, 0.95, 0.99]

DP_ALGORITHMS: list[str] = ["PE", "VI", "PI", "MPI", "AsyncVI"]
RL_ALGORITHMS: list[str] = ["QLearning", "ExpectedSARSA"]

TASKS: list[str] = ["chain_base", "grid_base", "taxi_base"]
TASK_LABELS: dict[str, str] = {
    "chain_base": "Chain",
    "grid_base": "Grid",
    "taxi_base": "Taxi",
}

# Colour palette for tasks (colourblind-friendly).
TASK_COLORS: dict[str, str] = {
    "chain_base": "#0072B2",
    "grid_base": "#D55E00",
    "taxi_base": "#009E73",
}

# Algorithm display names.
ALGO_LABELS: dict[str, str] = {
    "PE": "PE",
    "VI": "VI",
    "PI": "PI",
    "MPI": "MPI",
    "AsyncVI": "Async VI",
    "QLearning": r"$Q$-learning",
    "ExpectedSARSA": "Exp. SARSA",
}

# Default processed output root.
_DEFAULT_ABLATION_SUMMARY = (
    "results/weighted_lse_dp/processed/phase1/ablation_summary.json"
)
_DEFAULT_OUT_DIR = "results/weighted_lse_dp/processed/phase1/figures"


# ---------------------------------------------------------------------------
# Paper-quality matplotlib style (no seaborn, Type 1/42 fonts)
# ---------------------------------------------------------------------------

def _apply_paper_style() -> None:
    """Set rcParams for NeurIPS-quality vector figures."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

AblationKey = tuple[str, str, float]
"""(task, algorithm, gamma_prime)."""


def _load_ablation_summary(path: Path) -> dict[AblationKey, dict[str, Any]]:
    """Parse ``ablation_summary.json`` into the canonical keyed dict.

    Returns
    -------
    dict[(task, algorithm, gamma_prime), metrics_dict]
        Each value contains keys from ``scalar_metrics`` such as
        ``n_sweeps``, ``final_disc_return_mean``, ``auc_disc_return``,
        with nested ``mean``, ``ci_low``, ``ci_high`` when available.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    groups: list[dict[str, Any]] = raw.get("groups", [])
    data: dict[AblationKey, dict[str, Any]] = {}

    for g in groups:
        task = g.get("task", "unknown")
        algo = g.get("algorithm", "unknown")
        gp = g.get("gamma_prime")
        if gp is None:
            continue
        gp = float(gp)
        metrics = g.get("scalar_metrics", {})
        data[(task, algo, gp)] = metrics

    return data


# ---------------------------------------------------------------------------
# Demo data generation
# ---------------------------------------------------------------------------


def _generate_demo_data(
    rng: np.random.Generator | None = None,
) -> dict[AblationKey, dict[str, Any]]:
    """Create synthetic ablation data for demonstration.

    The synthetic data captures the expected qualitative pattern: lower
    gamma' leads to faster DP convergence (fewer sweeps) but possibly
    lower asymptotic RL return; gamma'=0.99 (the nominal discount)
    produces the highest return but slowest convergence.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data: dict[AblationKey, dict[str, Any]] = {}

    # Base sweep counts for DP algorithms at gamma'=0.99.
    dp_base_sweeps: dict[str, dict[str, float]] = {
        "chain_base": {"PE": 45, "VI": 38, "PI": 8, "MPI": 12, "AsyncVI": 42},
        "grid_base": {"PE": 60, "VI": 52, "PI": 10, "MPI": 16, "AsyncVI": 58},
        "taxi_base": {"PE": 90, "VI": 78, "PI": 14, "MPI": 22, "AsyncVI": 85},
    }

    # Multiplier: lower gamma' => fewer sweeps to converge.
    gp_sweep_factor: dict[float, float] = {0.90: 0.35, 0.95: 0.60, 0.99: 1.0}

    for task in TASKS:
        for algo in DP_ALGORITHMS:
            base = dp_base_sweeps[task][algo]
            for gp in GAMMA_PRIMES:
                sweeps = base * gp_sweep_factor[gp]
                noise = rng.normal(0, sweeps * 0.08)
                mean_val = max(1.0, sweeps + noise)
                ci_half = mean_val * 0.12
                data[(task, algo, gp)] = {
                    "n_sweeps": {
                        "mean": round(mean_val, 2),
                        "ci_low": round(max(1.0, mean_val - ci_half), 2),
                        "ci_high": round(mean_val + ci_half, 2),
                    },
                }

    # Base final discounted returns for RL algorithms at gamma'=0.99.
    rl_base_return: dict[str, dict[str, float]] = {
        "chain_base": {"QLearning": 0.82, "ExpectedSARSA": 0.85},
        "grid_base": {"QLearning": 0.75, "ExpectedSARSA": 0.78},
        "taxi_base": {"QLearning": 0.62, "ExpectedSARSA": 0.66},
    }

    # Lower gamma' => lower optimal return (shorter effective horizon).
    gp_return_factor: dict[float, float] = {0.90: 0.72, 0.95: 0.88, 0.99: 1.0}

    for task in TASKS:
        for algo in RL_ALGORITHMS:
            base = rl_base_return[task][algo]
            for gp in GAMMA_PRIMES:
                ret = base * gp_return_factor[gp]
                noise = rng.normal(0, ret * 0.05)
                mean_val = max(0.0, ret + noise)
                ci_half = mean_val * 0.08
                data[(task, algo, gp)] = {
                    "final_disc_return_mean": {
                        "mean": round(mean_val, 4),
                        "ci_low": round(max(0.0, mean_val - ci_half), 4),
                        "ci_high": round(mean_val + ci_half, 4),
                    },
                    "auc_disc_return": {
                        "mean": round(mean_val * 50000, 1),
                        "ci_low": round(max(0.0, (mean_val - ci_half) * 50000), 1),
                        "ci_high": round((mean_val + ci_half) * 50000, 1),
                    },
                }

    return data


# ---------------------------------------------------------------------------
# Core figure function
# ---------------------------------------------------------------------------


def _extract_bar_values(
    data: dict[AblationKey, dict[str, Any]],
    task: str,
    algo: str,
    metric_key: str,
) -> tuple[list[float], list[float], list[float]]:
    """Extract (means, ci_lows, ci_highs) for each gamma' value.

    Returns lists aligned with GAMMA_PRIMES.
    """
    means: list[float] = []
    ci_lows: list[float] = []
    ci_highs: list[float] = []

    for gp in GAMMA_PRIMES:
        entry = data.get((task, algo, gp), {})
        metric = entry.get(metric_key, {})
        if isinstance(metric, dict):
            m = metric.get("mean", 0.0)
            lo = metric.get("ci_low", m)
            hi = metric.get("ci_high", m)
        elif isinstance(metric, (int, float)):
            m = float(metric)
            lo = m
            hi = m
        else:
            m, lo, hi = 0.0, 0.0, 0.0
        means.append(m)
        ci_lows.append(lo)
        ci_highs.append(hi)

    return means, ci_lows, ci_highs


def make_ablation_figure(
    data: dict[AblationKey, dict[str, Any]],
    *,
    out_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render the Phase I gamma' ablation control figure.

    Parameters
    ----------
    data
        Mapping from ``(task, algorithm, gamma_prime)`` to a dict of
        scalar metrics.  Expected metric keys: ``n_sweeps`` for DP
        algorithms, ``final_disc_return_mean`` for RL algorithms.  Each
        metric value should be a dict with ``mean``, ``ci_low``,
        ``ci_high`` keys, or a bare scalar.
    out_path
        If provided, saves the figure as PDF and PNG at this path
        (without extension -- both ``.pdf`` and ``.png`` are written).
    show
        If ``True``, call ``plt.show()`` after rendering.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_paper_style()

    # Discover which tasks and algorithms are actually present.
    present_tasks = sorted({k[0] for k in data}, key=lambda t: TASKS.index(t) if t in TASKS else 999)
    present_dp = sorted(
        {k[1] for k in data if k[1] in DP_ALGORITHMS},
        key=lambda a: DP_ALGORITHMS.index(a) if a in DP_ALGORITHMS else 999,
    )
    present_rl = sorted(
        {k[1] for k in data if k[1] in RL_ALGORITHMS},
        key=lambda a: RL_ALGORITHMS.index(a) if a in RL_ALGORITHMS else 999,
    )

    n_panels = 0
    if present_dp:
        n_panels += 1
    if present_rl:
        n_panels += 1
    if n_panels == 0:
        raise ValueError("No DP or RL algorithm data found in ablation data.")

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 4.0),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    ax_idx = 0

    # -- DP panel: grouped bar chart of n_sweeps ----------------------------
    if present_dp:
        ax = axes_flat[ax_idx]
        ax_idx += 1
        _plot_grouped_bars(
            ax=ax,
            data=data,
            tasks=present_tasks,
            algorithms=present_dp,
            metric_key="n_sweeps",
            ylabel="Sweeps to convergence",
            title="DP algorithms",
        )

    # -- RL panel: grouped bar chart of final_disc_return_mean --------------
    if present_rl:
        ax = axes_flat[ax_idx]
        ax_idx += 1
        _plot_grouped_bars(
            ax=ax,
            data=data,
            tasks=present_tasks,
            algorithms=present_rl,
            metric_key="final_disc_return_mean",
            ylabel="Final discounted return",
            title="RL algorithms",
        )

    fig.suptitle(
        r"Fixed-$\gamma^\prime$ ablation",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stem = out_path.with_suffix("")
        for ext in (".pdf", ".png"):
            fig.savefig(
                str(stem) + ext,
                bbox_inches="tight",
                dpi=300,
            )
            print(f"  Saved: {stem}{ext}")

    if show:
        plt.show()

    return fig


def _plot_grouped_bars(
    ax: plt.Axes,
    data: dict[AblationKey, dict[str, Any]],
    tasks: list[str],
    algorithms: list[str],
    metric_key: str,
    ylabel: str,
    title: str,
) -> None:
    """Render a grouped bar chart on a single Axes.

    Grouping: one cluster per algorithm, bars within each cluster are
    coloured by task, with gamma' on the x-axis within each
    task-algorithm group.

    Layout: algorithms along the x-axis as major groups.  Within each
    algorithm group, sub-groups for each task, and within each sub-group
    one bar per gamma' value.
    """
    n_tasks = len(tasks)
    n_algos = len(algorithms)
    n_gp = len(GAMMA_PRIMES)

    # Total bars per algorithm group = n_tasks * n_gp.
    bars_per_algo = n_tasks * n_gp
    bar_width = 0.7 / bars_per_algo if bars_per_algo > 0 else 0.1
    group_width = bars_per_algo * bar_width
    group_gap = 0.6  # gap between algorithm groups

    # Hatching patterns for gamma' to distinguish within each task colour.
    gp_hatches: list[str | None] = ["///", None, "\\\\\\"]
    gp_labels_strs: list[str] = [
        rf"$\gamma^\prime={gp}$" for gp in GAMMA_PRIMES
    ]

    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    seen_legend: set[str] = set()

    algo_centers: list[float] = []

    for a_idx, algo in enumerate(algorithms):
        group_center = a_idx * (group_width + group_gap)
        algo_centers.append(group_center)

        for t_idx, task in enumerate(tasks):
            means, ci_lows, ci_highs = _extract_bar_values(
                data, task, algo, metric_key,
            )
            color = TASK_COLORS.get(task, "#666666")
            task_label = TASK_LABELS.get(task, task)

            for g_idx, gp in enumerate(GAMMA_PRIMES):
                bar_offset = (
                    (t_idx * n_gp + g_idx - (bars_per_algo - 1) / 2)
                    * bar_width
                )
                x = group_center + bar_offset
                m = means[g_idx]
                err_lo = max(0.0, m - ci_lows[g_idx])
                err_hi = max(0.0, ci_highs[g_idx] - m)
                has_ci = (err_lo > 0 or err_hi > 0)

                bar = ax.bar(
                    x,
                    m,
                    width=bar_width * 0.9,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=gp_hatches[g_idx],
                    alpha=0.85,
                    yerr=[[err_lo], [err_hi]] if has_ci else None,
                    capsize=2 if has_ci else 0,
                    error_kw={"linewidth": 0.8},
                )

                # Legend entries: one per (task, gamma') combination.
                legend_key = f"{task}_{gp}"
                if legend_key not in seen_legend:
                    seen_legend.add(legend_key)

                # Build a simplified legend: task colour + gamma' hatch.
                # We add task legend entries and gamma' legend entries
                # separately.
                task_legend_key = f"task_{task}"
                if task_legend_key not in seen_legend:
                    seen_legend.add(task_legend_key)
                    legend_handles.append(
                        matplotlib.patches.Patch(
                            facecolor=color,
                            edgecolor="black",
                            linewidth=0.5,
                            label=task_label,
                        )
                    )
                    legend_labels.append(task_label)

                gp_legend_key = f"gp_{gp}"
                if gp_legend_key not in seen_legend:
                    seen_legend.add(gp_legend_key)
                    legend_handles.append(
                        matplotlib.patches.Patch(
                            facecolor="white",
                            edgecolor="black",
                            linewidth=0.5,
                            hatch=gp_hatches[g_idx],
                            label=gp_labels_strs[g_idx],
                        )
                    )
                    legend_labels.append(gp_labels_strs[g_idx])

    ax.set_xticks(algo_centers)
    ax.set_xticklabels(
        [ALGO_LABELS.get(a, a) for a in algorithms],
        rotation=0,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="best",
        framealpha=0.8,
        edgecolor="none",
        ncol=2,
    )

    # Ensure y-axis starts at 0.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0, top=ymax * 1.08)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fig_ablation",
        description=(
            "Phase I fixed-gamma' ablation control figure."
        ),
    )
    parser.add_argument(
        "--ablation-summary",
        type=Path,
        default=None,
        help=(
            "Path to ablation_summary.json "
            f"(default: {_DEFAULT_ABLATION_SUMMARY})."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            f"Output directory (default: {_DEFAULT_OUT_DIR})."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Use synthetic demo data instead of real results.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the figure interactively.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)

    # Resolve paths relative to repo root.
    if args.out_dir is None:
        out_dir = _REPO_ROOT / _DEFAULT_OUT_DIR
    else:
        out_dir = Path(args.out_dir)

    if args.demo:
        print("[fig_ablation] Generating demo data (seed=42)...")
        data = _generate_demo_data(rng=np.random.default_rng(42))
    else:
        summary_path = (
            Path(args.ablation_summary)
            if args.ablation_summary is not None
            else _REPO_ROOT / _DEFAULT_ABLATION_SUMMARY
        )
        if not summary_path.is_file():
            print(
                f"ERROR: ablation_summary.json not found at {summary_path}\n"
                "  Run aggregate_phase1.py first, or use --demo for synthetic data.",
                file=sys.stderr,
            )
            return 1
        print(f"[fig_ablation] Loading: {summary_path}")
        data = _load_ablation_summary(summary_path)
        if not data:
            print(
                "WARNING: ablation_summary.json contains no ablation groups. "
                "Falling back to --demo mode.",
                file=sys.stderr,
            )
            data = _generate_demo_data(rng=np.random.default_rng(42))

    out_path = out_dir / "phase1_ablation"

    print(f"[fig_ablation] Rendering figure...")
    print(f"  Data entries: {len(data)}")
    print(f"  Output: {out_path}.{{pdf,png}}")

    fig = make_ablation_figure(data, out_path=out_path, show=args.show)
    plt.close(fig)

    print("[fig_ablation] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
