#!/usr/bin/env python
"""Phase II figure 11.1.4: State-visitation heatmaps for grid tasks.

Generates a 1x3 figure showing 5x5 grid heatmaps of margin_q50 per
state (averaged over stages) for grid_base, grid_hazard, and
grid_regime_shift.

If state-level margin data is not available, falls back to a marginal
stage distribution bar chart as a placeholder.

Data dependencies
-----------------
- ``results/weighted_lse_dp/phase2/calibration/grid_hazard.json``
- ``results/weighted_lse_dp/phase2/calibration/grid_regime_shift.json``
- ``results/weighted_lse_dp/phase1/calibration/grid_base.json``

Each calibration JSON should contain at minimum::

    {
      "per_state_margin_q50": [float, ...],  # length 25 for 5x5 grid
      "n_states": 25
    }

If ``per_state_margin_q50`` is absent but ``per_stage`` data exists,
the script plots per-stage margin_q50 as a fallback.

Seed policy: calibration files aggregate all seeds.

Spec anchor: Phase II spec S11.1 item 4.

Regeneration: deterministic given identical calibration JSON.

Usage::

    python plot_phase2_heatmaps.py --out-root results/weighted_lse_dp
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

GRID_TASKS: list[tuple[str, str, str]] = [
    # (task_key, phase, display_name)
    ("grid_base", "phase1", "Grid Base"),
    ("grid_hazard", "phase2", "Grid Hazard"),
    ("grid_regime_shift", "phase2", "Grid Regime Shift"),
]

GRID_ROWS: int = 5
GRID_COLS: int = 5


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
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_calibration(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    *,
    root: Path,
    out_dir: Path,
    show: bool = False,
) -> Path | None:
    """Create the state-visitation heatmap figure.

    Returns PDF path or None if data is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    _apply_paper_rc()

    for style in ("seaborn-v0_8-paper", "seaborn-paper"):
        try:
            plt.style.use(style)
            _apply_paper_rc()
            break
        except OSError:
            continue

    n_tasks = len(GRID_TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(12, 4), constrained_layout=True)
    if n_tasks == 1:
        axes = [axes]

    any_data = False
    used_fallback = False

    for ax, (task_key, phase, display_name) in zip(axes, GRID_TASKS):
        ax.set_title(display_name)

        cal_path = root / phase / "calibration" / f"{task_key}.json"
        cal = _load_calibration(cal_path)

        if cal is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Try state-level margin data.
        per_state = cal.get("per_state_margin_q50")
        if per_state is not None and len(per_state) == GRID_ROWS * GRID_COLS:
            grid = np.asarray(per_state, dtype=float).reshape(GRID_ROWS, GRID_COLS)
            vmax = max(abs(grid.min()), abs(grid.max())) or 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            im = ax.imshow(grid, cmap="RdBu_r", norm=norm, aspect="equal",
                           origin="upper")
            fig.colorbar(im, ax=ax, shrink=0.8, label="margin $q_{50}$")

            # Annotate cells.
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    val = grid[i, j]
                    color = "white" if abs(val) > 0.6 * vmax else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

            ax.set_xticks(range(GRID_COLS))
            ax.set_yticks(range(GRID_ROWS))
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            any_data = True
        else:
            # Fallback: per-stage distribution.
            per_stage = cal.get("per_stage")
            if per_stage is not None:
                stages = sorted(per_stage.keys(), key=lambda s: int(s))
                margin_q50 = [
                    per_stage[s].get("margin_q50", 0.0) for s in stages
                ]
                counts = [
                    per_stage[s].get("count", 0) for s in stages
                ]
                x = np.arange(len(stages))
                ax.bar(x, margin_q50, color="#1f77b4", alpha=0.7,
                       label="margin $q_{50}$")
                ax.set_xticks(x)
                ax.set_xticklabels(stages, fontsize=6)
                ax.set_xlabel("Stage")
                ax.set_ylabel("Margin $q_{50}$")
                ax.legend(fontsize=7)
                ax.grid(True, axis="y", alpha=0.3)
                any_data = True
                used_fallback = True
            else:
                ax.text(0.5, 0.5, "No state/stage data",
                        transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])

    if not any_data:
        warnings.warn(
            "No grid heatmap data found. Skipping figure.",
            stacklevel=2,
        )
        plt.close(fig)
        return None

    title = "Phase II: State-level Margin Heatmaps (classical $\\beta=0$)"
    if used_fallback:
        title += " [stage-level fallback]"
    fig.suptitle(title, fontsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_11_1_4_heatmaps"
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
        description="Phase II figure 11.1.4: grid state-visitation heatmaps.",
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
    out_dir = root / "processed" / "phase2" / "figures"

    result = make_figure(root=root, out_dir=out_dir, show=args.show)
    if result is None:
        print("WARNING: Figure skipped -- no data available.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
