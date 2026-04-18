#!/usr/bin/env python
"""Phase II figure 11.1.5: Per-stage margin quantile ribbon plots.

Generates a figure with subplots per task family showing margin_q05,
margin_q50, margin_q95 vs stage for base (Phase I) vs stress (Phase II)
tasks as ribbon plots.

Data dependencies
-----------------
- ``results/weighted_lse_dp/phase2/calibration/<task>.json``
- ``results/weighted_lse_dp/phase1/calibration/<base_task>.json``

Each calibration JSON must contain at minimum::

    {
      "per_stage": {
        "<stage_index>": {
          "margin_q05": float,
          "margin_q50": float,
          "margin_q95": float
        }
      }
    }

Seed policy: calibration files aggregate all seeds. No filtering.

Spec anchor: Phase II spec S11.1 item 5.

Regeneration: deterministic given identical calibration JSON files.

Usage::

    python plot_phase2_margin_quantiles.py --out-root results/weighted_lse_dp
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

#: Task families: (display_name, [(task_key, phase, line_label), ...])
TASK_FAMILIES: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Chain", [
        ("chain_base", "phase1", "base"),
        ("chain_jackpot", "phase2", "jackpot"),
        ("chain_catastrophe", "phase2", "catastrophe"),
        ("chain_regime_shift", "phase2", "regime shift"),
    ]),
    ("Grid", [
        ("grid_base", "phase1", "base"),
        ("grid_hazard", "phase2", "hazard"),
        ("grid_regime_shift", "phase2", "regime shift"),
    ]),
    ("Taxi", [
        ("taxi_base", "phase1", "base"),
        ("taxi_bonus_shock", "phase2", "bonus shock"),
    ]),
]

TASK_COLORS: dict[str, str] = {
    "base": "#1f77b4",
    "jackpot": "#2ca02c",
    "catastrophe": "#d62728",
    "regime shift": "#ff7f0e",
    "hazard": "#9467bd",
    "bonus shock": "#8c564b",
}

QUANTILE_KEYS: list[str] = ["margin_q05", "margin_q50", "margin_q95"]


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
        "legend.fontsize": 7,
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

def _load_calibration(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_stage_quantiles(
    cal: dict[str, Any],
) -> tuple[Any, Any, Any, Any] | None:
    """Extract (stages, q05, q50, q95) arrays from calibration dict.

    Returns None if margin quantile data is missing.

    Primary source: top-level ``margin_quantiles`` dict written by
    ``build_calibration_json`` (R7-4 fix).  Falls back to ``per_stage``
    dict (legacy format).
    """
    import numpy as np

    # Primary path: margin_quantiles dict from aggregate_phase2.py
    mq = cal.get("margin_quantiles")
    if mq is not None:
        stages_raw = mq.get("stages")
        q05_raw = mq.get("q05")
        q50_raw = mq.get("q50")
        q95_raw = mq.get("q95")
        if stages_raw and q50_raw:
            stages = np.array(stages_raw, dtype=int)
            q05 = np.array(q05_raw or [0.0] * len(stages), dtype=float)
            q50 = np.array(q50_raw, dtype=float)
            q95 = np.array(q95_raw or [0.0] * len(stages), dtype=float)
            return stages, q05, q50, q95

    # Fallback: stagewise dict (key names differ from per_stage legacy format)
    stagewise = cal.get("stagewise")
    if stagewise is not None:
        stage_list = stagewise.get("stage")
        q05_list = stagewise.get("margin_q05_mean")
        q50_list = stagewise.get("margin_q50_mean")
        q95_list = stagewise.get("margin_q95_mean")
        if stage_list and q50_list:
            stages = np.array(stage_list, dtype=int)
            q05 = np.array(q05_list or [0.0] * len(stages), dtype=float)
            q50 = np.array(q50_list, dtype=float)
            q95 = np.array(q95_list or [0.0] * len(stages), dtype=float)
            return stages, q05, q50, q95

    # Legacy path: per_stage dict
    per_stage = cal.get("per_stage")
    if per_stage is None:
        return None

    stages_sorted = sorted(per_stage.keys(), key=lambda s: int(s))
    stages = np.array([int(s) for s in stages_sorted], dtype=int)
    q05 = np.array([per_stage[s].get("margin_q05", 0.0) for s in stages_sorted], dtype=float)
    q50 = np.array([per_stage[s].get("margin_q50", 0.0) for s in stages_sorted], dtype=float)
    q95 = np.array([per_stage[s].get("margin_q95", 0.0) for s in stages_sorted], dtype=float)

    return stages, q05, q50, q95


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    *,
    root: Path,
    out_dir: Path,
    show: bool = False,
) -> Path | None:
    """Create the margin quantile ribbon figure.

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

    n_families = len(TASK_FAMILIES)
    fig, axes = plt.subplots(1, n_families, figsize=(12, 4), constrained_layout=True)
    if n_families == 1:
        axes = [axes]

    any_data = False

    for ax, (family_name, tasks) in zip(axes, TASK_FAMILIES):
        ax.set_title(family_name)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Margin")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="black", lw=0.5, ls=":")

        for task_key, phase, line_label in tasks:
            cal_path = root / phase / "calibration" / f"{task_key}.json"
            cal = _load_calibration(cal_path)
            if cal is None:
                continue

            result = _extract_stage_quantiles(cal)
            if result is None:
                continue

            stages, q05, q50, q95 = result
            color = TASK_COLORS.get(line_label, "#333333")

            # Ribbon: q05 to q95.
            ax.fill_between(stages, q05, q95, color=color, alpha=0.15)
            # Median line.
            ax.plot(stages, q50, color=color, lw=1.5, label=line_label)
            # Light dashed for q05/q95 edges.
            ax.plot(stages, q05, color=color, lw=0.5, ls="--", alpha=0.5)
            ax.plot(stages, q95, color=color, lw=0.5, ls="--", alpha=0.5)
            any_data = True

        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", fontsize=7, framealpha=0.8)

    if not any_data:
        warnings.warn(
            "No margin quantile data found. Skipping figure.",
            stacklevel=2,
        )
        plt.close(fig)
        return None

    fig.suptitle(
        "Phase II: Per-stage Margin Quantiles -- Base vs Stress (classical $\\beta=0$)",
        fontsize=11,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_11_1_5_margin_quantiles"
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
        description="Phase II figure 11.1.5: per-stage margin quantiles.",
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
