#!/usr/bin/env python
"""Phase II figure 11.1.1: Base vs modified learning curves.

Generates a 1x3 figure (chain, grid, taxi) comparing Phase I base tasks
with Phase II stress variants.  Each panel overlays base and stress
learning curves for each algorithm.

x-axis: training step checkpoint.
y-axis: mean discounted return across seeds.
Shaded band: +/- 1 std across seeds (not bootstrap CI -- std is used
here because the comparison is visual, not inferential).

Data dependencies
-----------------
- ``results/weighted_lse_dp/phase2/aggregated/<task>/<algorithm>/summary.json``
  for Phase II stress tasks.
- ``results/weighted_lse_dp/phase1/aggregated/<base_task>/<algorithm>/summary.json``
  for Phase I base comparisons.

Each ``summary.json`` must contain at minimum::

    {
      "checkpoints": [int, ...],
      "disc_return_mean_per_seed": {
        "<seed>": [float, ...]
      }
    }

Seed policy: all seeds present in each summary.json are used.
No cherry-picking or filtering is applied.

Confidence intervals: +/- 1 sample std across seeds (band), not bootstrap.

Spec anchor: Phase II spec S11.1 item 1.

Regeneration: given identical summary.json files this script produces the
same figure (PDF timestamp pinned via SOURCE_DATE_EPOCH).

Usage::

    python plot_phase2_learning_curves.py --out-root results/weighted_lse_dp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Task families: (display_name, base_task, stress_tasks).
TASK_FAMILIES: list[tuple[str, str, list[str]]] = [
    ("Chain", "chain_base", [
        "chain_sparse_long", "chain_jackpot", "chain_catastrophe",
        "chain_regime_shift",
    ]),
    ("Grid", "grid_base", [
        "grid_sparse_goal", "grid_hazard", "grid_regime_shift",
    ]),
    ("Taxi", "taxi_base", [
        "taxi_bonus_shock",
    ]),
]

ALGORITHMS: list[str] = ["QLearning", "ExpectedSARSA"]

#: Consistent color palette.
BASE_COLOR = "#1f77b4"
STRESS_COLORS: list[str] = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

#: Line style for base vs stress.
BASE_LS = "-"
STRESS_LS = "--"


# ---------------------------------------------------------------------------
# NeurIPS-compliant rc overrides
# ---------------------------------------------------------------------------

def _apply_paper_rc() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_summary(path: Path) -> dict[str, Any] | None:
    """Load a summary.json, returning None if absent."""
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_curves(
    summary: dict[str, Any],
) -> tuple[Any, Any, Any] | None:
    """Extract (checkpoints, mean, std) from a summary dict.

    Returns None if the required keys are missing.
    """
    import numpy as np

    checkpoints = summary.get("checkpoints")
    per_seed = summary.get("disc_return_mean_per_seed")
    if checkpoints is None or per_seed is None:
        return None

    ckpts = np.asarray(checkpoints, dtype=float)
    seeds_arr = np.array(list(per_seed.values()), dtype=float)  # (n_seeds, n_ckpts)
    if seeds_arr.ndim != 2 or seeds_arr.shape[1] != len(ckpts):
        return None

    mean = seeds_arr.mean(axis=0)
    std = seeds_arr.std(axis=0, ddof=1) if seeds_arr.shape[0] > 1 else np.zeros_like(mean)
    return ckpts, mean, std


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    *,
    phase1_agg_root: Path,
    phase2_agg_root: Path,
    out_dir: Path,
    show: bool = False,
) -> Path | None:
    """Create the Phase II learning-curves comparison figure.

    Returns the PDF output path, or None if all data is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_paper_rc()

    # Try paper style, fall back gracefully.
    for style in ("seaborn-v0_8-paper", "seaborn-paper"):
        try:
            plt.style.use(style)
            _apply_paper_rc()  # re-apply after style
            break
        except OSError:
            continue

    n_families = len(TASK_FAMILIES)
    fig, axes = plt.subplots(1, n_families, figsize=(12, 4), constrained_layout=True)
    if n_families == 1:
        axes = [axes]

    any_data = False

    for ax, (family_name, base_task, stress_tasks) in zip(axes, TASK_FAMILIES):
        ax.set_title(family_name)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean discounted return")
        ax.grid(True, alpha=0.3)

        legend_handles = []

        for algo in ALGORITHMS:
            # Load base.
            base_path = phase1_agg_root / base_task / algo / "summary.json"
            base_summary = _load_summary(base_path)
            if base_summary is not None:
                curves = _extract_curves(base_summary)
                if curves is not None:
                    ckpts, mean, std = curves
                    line, = ax.plot(
                        ckpts, mean, color=BASE_COLOR, ls=BASE_LS,
                        label=f"{base_task} ({algo})",
                    )
                    ax.fill_between(ckpts, mean - std, mean + std,
                                    color=BASE_COLOR, alpha=0.15)
                    legend_handles.append(line)
                    any_data = True

            # Load stress variants.
            for si, stress_task in enumerate(stress_tasks):
                color = STRESS_COLORS[si % len(STRESS_COLORS)]
                stress_path = phase2_agg_root / stress_task / algo / "summary.json"
                stress_summary = _load_summary(stress_path)
                if stress_summary is not None:
                    curves = _extract_curves(stress_summary)
                    if curves is not None:
                        ckpts, mean, std = curves
                        line, = ax.plot(
                            ckpts, mean, color=color, ls=STRESS_LS,
                            label=f"{stress_task} ({algo})",
                        )
                        ax.fill_between(ckpts, mean - std, mean + std,
                                        color=color, alpha=0.15)
                        legend_handles.append(line)
                        any_data = True

        if legend_handles:
            ax.legend(
                handles=legend_handles, loc="upper left",
                fontsize=6, framealpha=0.8,
            )

    if not any_data:
        warnings.warn(
            "No Phase I or Phase II learning curve data found. "
            "Skipping figure generation.",
            stacklevel=2,
        )
        plt.close(fig)
        return None

    fig.suptitle(
        "Phase II: Base vs Stress-test Learning Curves (classical $\\beta=0$)",
        fontsize=11,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_11_1_1_learning_curves"
    fig.savefig(str(stem) + ".pdf", format="pdf")
    fig.savefig(str(stem) + ".png", format="png", dpi=150)
    print(f"Saved: {stem}.pdf")
    print(f"Saved: {stem}.png")

    if show:
        plt.show()
    plt.close(fig)
    return Path(str(stem) + ".pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase II figure 11.1.1: learning curves comparison.",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("results/weighted_lse_dp"),
        help="Root of the weighted_lse_dp results tree.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.out_root

    phase1_agg = root / "phase1" / "aggregated"
    phase2_agg = root / "phase2" / "aggregated"
    out_dir = root / "processed" / "phase2" / "figures"

    result = make_figure(
        phase1_agg_root=phase1_agg,
        phase2_agg_root=phase2_agg,
        out_dir=out_dir,
        show=args.show,
    )
    if result is None:
        print("WARNING: Figure skipped -- no data available.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
