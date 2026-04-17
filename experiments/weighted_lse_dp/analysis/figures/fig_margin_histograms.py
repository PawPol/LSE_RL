"""Phase I stagewise margin histograms (spec S10.1.4).

Renders one figure with three subplots (chain, grid, taxi) showing the
distribution of ``margin_beta0 = reward - v_next_beta0`` (NO gamma)
across stages.  Each subplot shows:

- Outer ribbon: q05-q95 (light fill)
- Inner ribbon: q25-q75 (medium fill)
- Center line:  q50 (dark solid)

Data source: ``calibration_stats.npz`` from paper_suite RL runs.

Confidence intervals: not applicable (quantile bands are computed from
the full per-stage margin distribution, not bootstrapped).  When
multiple seeds are available, the bands shown are from the selected
seed (default: first found).

Cherry-picking filter: ``--seed`` selects exactly one seed; when
omitted, the first seed found on disk is used (alphabetical order).

Regeneratability: deterministic -- no randomness involved.  PDF
timestamps are pinned to 1970-01-01 via SOURCE_DATE_EPOCH.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

# Pin PDF timestamps for reproducibility.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

# NeurIPS requirement: Type 42 (TrueType) fonts in PDF output, no seaborn.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8

import matplotlib.pyplot as plt  # noqa: E402 (after rcParams set)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

# Task families in canonical order.
TASK_ORDER: list[str] = ["chain", "grid", "taxi"]

# Colour palette (colourblind-safe, no seaborn dependency).
_COLOURS: dict[str, dict[str, str]] = {
    "chain": {"line": "#1b9e77", "inner": "#1b9e7755", "outer": "#1b9e7722"},
    "grid":  {"line": "#d95f02", "inner": "#d95f0255", "outer": "#d95f0222"},
    "taxi":  {"line": "#7570b3", "inner": "#7570b355", "outer": "#7570b322"},
}


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------


def _make_demo_data() -> dict[str, dict[str, np.ndarray]]:
    """Synthesise plausible calibration data for ``--demo`` mode.

    Returns data for the three canonical tasks with varying horizons.
    The margins follow a pattern where the median drifts towards zero
    at later stages (typical of converged value functions).
    """
    rng = np.random.default_rng(42)
    data: dict[str, dict[str, np.ndarray]] = {}

    horizons = {"chain": 20, "grid": 15, "taxi": 25}

    for task, H in horizons.items():
        n_stages = H + 1
        stage = np.arange(n_stages, dtype=np.int64)

        # Simulate a margin distribution that narrows towards zero at
        # later stages -- a common pattern under value convergence.
        base_scale = rng.uniform(0.3, 0.8) if task != "chain" else 0.5
        median_drift = np.linspace(0.3, -0.1, n_stages) * base_scale
        spread = np.linspace(1.0, 0.4, n_stages) * base_scale

        margin_q50 = median_drift + rng.normal(0, 0.02, n_stages)
        margin_q25 = margin_q50 - 0.5 * spread + rng.normal(0, 0.01, n_stages)
        margin_q75 = margin_q50 + 0.5 * spread + rng.normal(0, 0.01, n_stages)
        margin_q05 = margin_q50 - 1.5 * spread + rng.normal(0, 0.01, n_stages)
        margin_q95 = margin_q50 + 1.5 * spread + rng.normal(0, 0.01, n_stages)

        pos_margin_mean = np.maximum(margin_q50, 0.0) * 1.2
        neg_margin_mean = np.maximum(-margin_q50, 0.0) * 1.2
        count = np.full(n_stages, 1000, dtype=np.int64)
        count[-1] = 0  # terminal stage

        data[task] = {
            "stage": stage,
            "margin_q05": margin_q05,
            "margin_q25": margin_q25,
            "margin_q50": margin_q50,
            "margin_q75": margin_q75,
            "margin_q95": margin_q95,
            "pos_margin_mean": pos_margin_mean,
            "neg_margin_mean": neg_margin_mean,
            "count": count,
        }

    return data


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_calibration_data(
    paper_suite_root: Path,
    algorithm: str,
    seed: int | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load calibration_stats.npz for each task family.

    Parameters
    ----------
    paper_suite_root:
        Root directory containing ``<task>/<algorithm>/seed_<seed>/``
        sub-trees.
    algorithm:
        Algorithm subdirectory name (e.g. ``"QLearning"``).
    seed:
        Specific seed to load.  If ``None``, uses the first seed found
        in alphabetical order.

    Returns
    -------
    dict mapping task name to its calibration arrays.
    """
    data: dict[str, dict[str, np.ndarray]] = {}

    for task in TASK_ORDER:
        algo_dir = paper_suite_root / task / algorithm
        if not algo_dir.is_dir():
            continue

        # Resolve seed directory.
        if seed is not None:
            seed_dir = algo_dir / f"seed_{seed}"
        else:
            seed_dirs = sorted(
                d for d in algo_dir.iterdir()
                if d.is_dir() and d.name.startswith("seed_")
            )
            if not seed_dirs:
                continue
            seed_dir = seed_dirs[0]

        cal_path = seed_dir / "calibration_stats.npz"
        if not cal_path.is_file():
            continue

        with np.load(cal_path, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files if k != "_schema"}

        # Validate required keys.
        required = [
            "stage", "margin_q05", "margin_q25", "margin_q50",
            "margin_q75", "margin_q95",
        ]
        missing = [k for k in required if k not in arrays]
        if missing:
            print(
                f"WARNING: {cal_path} missing keys {missing!r}, skipping",
                file=sys.stderr,
            )
            continue

        data[task] = arrays

    return data


# ---------------------------------------------------------------------------
# Public figure API
# ---------------------------------------------------------------------------


def make_margin_histograms_figure(
    data: dict[str, dict[str, np.ndarray]],
    *,
    out_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Create the stagewise margin quantile-band figure.

    Parameters
    ----------
    data:
        Mapping ``task_name -> {stage, margin_q05, ..., margin_q95, ...}``.
        Only tasks present in the dict are plotted; the subplot grid is
        always 1x3 (chain, grid, taxi) with empty axes for missing tasks.
    out_path:
        If provided, save the figure as PDF and PNG to this path (the
        extension is replaced).
    show:
        If ``True``, call ``plt.show()`` at the end.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.0, 2.4),
        constrained_layout=True,
        sharey=False,
    )

    for idx, task in enumerate(TASK_ORDER):
        ax: plt.Axes = axes[idx]
        colours = _COLOURS[task]

        if task not in data:
            ax.set_title(f"{task} (no data)")
            ax.set_xlabel("Stage $t$")
            if idx == 0:
                ax.set_ylabel(r"$r - V^{\beta_0}_{t+1}(s')$")
            continue

        d = data[task]
        stage = d["stage"]

        # Filter out terminal stage with count=0 if present.
        mask = np.ones(len(stage), dtype=bool)
        if "count" in d:
            mask = d["count"] > 0
        s = stage[mask]
        q05 = d["margin_q05"][mask]
        q25 = d["margin_q25"][mask]
        q50 = d["margin_q50"][mask]
        q75 = d["margin_q75"][mask]
        q95 = d["margin_q95"][mask]

        # Outer band: q05-q95.
        ax.fill_between(s, q05, q95, color=colours["outer"], label="q05-q95")
        # Inner band: q25-q75.
        ax.fill_between(s, q25, q75, color=colours["inner"], label="q25-q75")
        # Median line.
        ax.plot(s, q50, color=colours["line"], linewidth=1.5, label="q50")

        # Zero reference line.
        ax.axhline(0, color="0.5", linewidth=0.5, linestyle="--", zorder=0)

        ax.set_title(task.capitalize())
        ax.set_xlabel("Stage $t$")
        if idx == 0:
            ax.set_ylabel(r"margin $= r - V^{\beta_0}_{t+1}(s')$")

        ax.legend(fontsize=7, loc="best", framealpha=0.7)

    fig.suptitle(
        "Stagewise margin distribution (no $\\gamma$)",
        fontsize=11,
        fontweight="bold",
    )

    # Save outputs.
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path.with_suffix(".pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            out_path.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved: {out_path.with_suffix('.pdf')}")
        print(f"Saved: {out_path.with_suffix('.png')}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase I stagewise margin histograms (spec S10.1.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper-suite-root",
        type=Path,
        default=_REPO_ROOT / "results" / "weighted_lse_dp" / "paper_suite",
        help="Root of paper_suite run tree.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="QLearning",
        help="Algorithm subdirectory to read calibration data from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed index.  If omitted, uses the first seed found.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            _REPO_ROOT / "results" / "weighted_lse_dp" / "processed"
            / "phase1" / "figures"
        ),
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate figure from synthetic demo data (no disk reads).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.demo:
        data = _make_demo_data()
        print("Using synthetic demo data.")
    else:
        data = _load_calibration_data(
            paper_suite_root=args.paper_suite_root,
            algorithm=args.algorithm,
            seed=args.seed,
        )
        if not data:
            print(
                "ERROR: No calibration data found under "
                f"{args.paper_suite_root}.  Use --demo for synthetic data.",
                file=sys.stderr,
            )
            sys.exit(1)

    out_path = Path(args.out_dir) / "phase1_margin_histograms"

    make_margin_histograms_figure(data, out_path=out_path, show=args.show)


if __name__ == "__main__":
    main()
