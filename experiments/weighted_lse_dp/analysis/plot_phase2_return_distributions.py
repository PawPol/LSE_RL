#!/usr/bin/env python
"""Phase II figure 11.1.2: Return distribution plots for jackpot/catastrophe.

Generates box/violin plots of return quantiles (q05, q25, q50, q75, q95)
comparing chain_base with chain_jackpot and chain_catastrophe tasks.

Data dependencies
-----------------
- ``results/weighted_lse_dp/phase2/calibration/<task>.json`` for stress tasks.
- ``results/weighted_lse_dp/phase1/calibration/chain_base.json`` for baseline.

Each calibration JSON must contain at minimum::

    {
      "return_quantiles": {
        "q05": float,
        "q25": float,
        "q50": float,
        "q75": float,
        "q95": float
      },
      "per_seed_returns": {
        "<seed>": [float, ...]
      }
    }

Seed policy: all seeds present in the calibration files are used.

Spec anchor: Phase II spec S11.1 item 2.

Regeneration: deterministic given identical calibration JSON files.

Usage::

    python plot_phase2_return_distributions.py --out-root results/weighted_lse_dp
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

TASKS_TO_COMPARE: list[tuple[str, str, str]] = [
    # (display_label, phase, task_key)
    ("chain_base", "phase1", "chain_base"),
    ("chain_jackpot", "phase2", "chain_jackpot"),
    ("chain_catastrophe", "phase2", "chain_catastrophe"),
]

QUANTILE_KEYS: list[str] = ["q05", "q25", "q50", "q75", "q95"]

TASK_COLORS: dict[str, str] = {
    "chain_base": "#1f77b4",
    "chain_jackpot": "#2ca02c",
    "chain_catastrophe": "#d62728",
}


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
    """Create the return-distribution comparison figure.

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

    # Collect data.
    all_returns: dict[str, Any] = {}  # task -> array of returns or quantile dict
    any_data = False

    for label, phase, task_key in TASKS_TO_COMPARE:
        cal_path = root / phase / "calibration" / f"{task_key}.json"
        cal = _load_calibration(cal_path)
        if cal is None:
            warnings.warn(f"Calibration file not found: {cal_path}", stacklevel=2)
            continue

        # Prefer per-seed raw returns for box plot; fall back to quantiles.
        # R7-4 fix: also accept base_returns/stress_returns pooled arrays
        # written by build_calibration_json (the fields that actually exist).
        per_seed = cal.get("per_seed_returns")
        if per_seed is not None:
            flat = []
            for seed_returns in per_seed.values():
                flat.extend(seed_returns)
            all_returns[label] = np.asarray(flat, dtype=float)
            any_data = True
        else:
            # Try pooled return arrays from aggregate_phase2.py
            base_ret = cal.get("base_returns")
            stress_ret = cal.get("stress_returns")
            pooled = (base_ret or []) + (stress_ret or [])
            if pooled:
                all_returns[label] = np.asarray(pooled, dtype=float)
                any_data = True
            else:
                rq = cal.get("return_quantiles")
                if rq is not None:
                    all_returns[label] = {k: rq[k] for k in QUANTILE_KEYS if k in rq}
                    any_data = True

    if not any_data:
        warnings.warn(
            "No return distribution data found. Skipping figure.",
            stacklevel=2,
        )
        return None

    # Determine plot type: box if raw arrays, bar-quantile if only quantiles.
    has_raw = any(isinstance(v, np.ndarray) for v in all_returns.values())

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    if has_raw:
        # Box plot for tasks with raw returns.
        raw_tasks = [(k, v) for k, v in all_returns.items() if isinstance(v, np.ndarray)]
        positions = list(range(len(raw_tasks)))
        bp = ax.boxplot(
            [v for _, v in raw_tasks],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.4),
        )
        for i, (task_label, _) in enumerate(raw_tasks):
            color = TASK_COLORS.get(task_label, "#999999")
            bp["boxes"][i].set_facecolor(color)
            bp["boxes"][i].set_alpha(0.5)
            bp["medians"][i].set_color("black")

        ax.set_xticks(positions)
        ax.set_xticklabels([t for t, _ in raw_tasks], rotation=15)
        ax.set_ylabel("Discounted return")
        ax.set_title("Return Distributions: Base vs Jackpot vs Catastrophe")
    else:
        # Quantile bar chart fallback.
        task_labels = list(all_returns.keys())
        x = np.arange(len(task_labels))
        width = 0.15
        for qi, qk in enumerate(QUANTILE_KEYS):
            vals = [
                all_returns[t].get(qk, 0.0) if isinstance(all_returns[t], dict) else 0.0
                for t in task_labels
            ]
            ax.bar(x + qi * width, vals, width, label=qk, alpha=0.8)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(task_labels, rotation=15)
        ax.set_ylabel("Return quantile value")
        ax.set_title("Return Quantiles: Base vs Jackpot vs Catastrophe")
        ax.legend(loc="best")

    ax.grid(True, axis="y", alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_11_1_2_return_distributions"
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
        description="Phase II figure 11.1.2: return distribution comparison.",
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
