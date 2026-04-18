#!/usr/bin/env python
"""Phase II figure 11.1.3: Change-point adaptation plots for regime-shift tasks.

Generates a 1x2 figure (chain_regime_shift, grid_regime_shift) showing
rolling mean return vs episode index with a vertical line at the
``change_at_episode`` boundary and shaded pre/post regions.

Data dependencies
-----------------
- ``results/weighted_lse_dp/phase2/aggregated/<task>/<algorithm>/summary.json``

Each ``summary.json`` must contain at minimum::

    {
      "episode_returns": {
        "<seed>": [float, ...]
      },
      "change_at_episode": int
    }

Alternatively, ``change_at_episode`` is read from the paper_suite config
if absent from summary.json.

Seed policy: all seeds in the summary are used.

Spec anchor: Phase II spec S11.1 item 3.

Regeneration: deterministic given identical input files.

Usage::

    python plot_phase2_adaptation.py --out-root results/weighted_lse_dp
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

REGIME_TASKS: list[tuple[str, str, int]] = [
    # (task_key, display_name, default_change_at_episode from paper_suite.json)
    ("chain_regime_shift", "Chain Regime Shift", 300),
    ("grid_regime_shift", "Grid Regime Shift", 200),
]

ALGORITHMS: list[str] = ["QLearning", "ExpectedSARSA"]

ALGO_COLORS: dict[str, str] = {
    "QLearning": "#1f77b4",
    "ExpectedSARSA": "#ff7f0e",
}

ROLLING_WINDOW: int = 20


# ---------------------------------------------------------------------------
# NeurIPS rc
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
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _rolling_mean(arr: Any, window: int) -> Any:
    """Compute rolling mean with numpy."""
    import numpy as np
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    *,
    phase2_agg_root: Path,
    out_dir: Path,
    show: bool = False,
) -> Path | None:
    """Create the adaptation plot figure.

    Returns PDF path or None if data is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_paper_rc()

    for style in ("seaborn-v0_8-paper", "seaborn-paper"):
        try:
            plt.style.use(style)
            _apply_paper_rc()
            break
        except OSError:
            continue

    n_tasks = len(REGIME_TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(12, 4), constrained_layout=True)
    if n_tasks == 1:
        axes = [axes]

    any_data = False

    for ax, (task_key, display_name, default_change_ep) in zip(axes, REGIME_TASKS):
        ax.set_title(display_name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rolling mean return")
        ax.grid(True, alpha=0.3)

        change_ep = default_change_ep
        panel_has_data = False

        for algo in ALGORITHMS:
            summary_path = phase2_agg_root / task_key / algo / "summary.json"
            summary = _load_summary(summary_path)
            if summary is None:
                continue

            # Read change point from summary or use default.
            change_ep = summary.get("change_at_episode", default_change_ep)

            episode_returns = summary.get("episode_returns")
            if episode_returns is None:
                continue

            # Average rolling mean across seeds.
            seed_curves = []
            for seed_key, returns_list in episode_returns.items():
                arr = np.asarray(returns_list, dtype=float)
                rolled = _rolling_mean(arr, ROLLING_WINDOW)
                seed_curves.append(rolled)

            if not seed_curves:
                continue

            # Align to shortest.
            min_len = min(len(c) for c in seed_curves)
            mat = np.stack([c[:min_len] for c in seed_curves], axis=0)
            mean = mat.mean(axis=0)
            std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)

            # x-axis is episode index (offset by half-window for rolling).
            x = np.arange(min_len) + ROLLING_WINDOW // 2

            color = ALGO_COLORS.get(algo, "#333333")
            ax.plot(x, mean, color=color, label=algo)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            panel_has_data = True
            any_data = True

        # Draw change-point line and shade regions.
        if panel_has_data:
            ylim = ax.get_ylim()
            ax.axvline(change_ep, color="red", ls=":", lw=1.0, label="Regime change")
            ax.axvspan(0, change_ep, alpha=0.04, color="blue", label="Pre-change")
            ax.axvspan(change_ep, ax.get_xlim()[1], alpha=0.04, color="red",
                       label="Post-change")
            ax.set_ylim(ylim)
            ax.legend(loc="best", fontsize=7, framealpha=0.8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

    if not any_data:
        warnings.warn(
            "No regime-shift adaptation data found. Skipping figure.",
            stacklevel=2,
        )
        plt.close(fig)
        return None

    fig.suptitle(
        "Phase II: Adaptation after Regime Shift (classical $\\beta=0$)",
        fontsize=11,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_11_1_3_adaptation"
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
        description="Phase II figure 11.1.3: regime-shift adaptation.",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("results/weighted_lse_dp"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.out_root
    phase2_agg = root / "phase2" / "aggregated"
    out_dir = root / "processed" / "phase2" / "figures"

    result = make_figure(
        phase2_agg_root=phase2_agg,
        out_dir=out_dir,
        show=args.show,
    )
    if result is None:
        print("WARNING: Figure skipped -- no data available.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
