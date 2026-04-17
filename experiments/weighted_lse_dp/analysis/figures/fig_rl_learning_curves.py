"""Phase I figure: RL learning curves (QLearning and ExpectedSARSA).

Generates a 1x3 figure (one subplot per task) with both algorithms
overlaid per panel.  x-axis = environment steps (from ``checkpoints``),
y-axis = mean discounted return across seeds with a 95% bootstrap
percentile CI shown as a shaded band.

Bootstrap details
-----------------
Method: percentile bootstrap on the seed-axis mean, 10 000 resamples,
95% two-sided CI.  Implemented by
``experiments.weighted_lse_dp.common.metrics.aggregate``.

Seed policy: all seeds present in ``results/processed/`` are included.
No filtering is applied.

Spec anchors: Phase I spec S9.1, S9.3, S10.1.3.

Regeneration
------------
Given the same ``results/weighted_lse_dp/paper_suite/`` directory, this
script produces the same figure (modulo PDF timestamp bytes; pinned to
epoch via ``SOURCE_DATE_EPOCH``).

Usage::

    .venv/bin/python experiments/weighted_lse_dp/analysis/figures/fig_rl_learning_curves.py \\
        [--paper-suite-root PATH] \\
        [--out-dir DIR] \\
        [--demo] \\
        [--show]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup -- mirror the convention used by the runners.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "mushroom-rl-dev"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend by default
import matplotlib.pyplot as plt  # noqa: E402

from experiments.weighted_lse_dp.common.metrics import aggregate  # noqa: E402

__all__ = ["make_rl_learning_curves_figure"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Pin PDF creation timestamp for byte-reproducibility.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

#: Tasks in display order.
TASKS: list[str] = ["chain_base", "grid_base", "taxi_base"]

#: Pretty names for subplot titles.
TASK_LABELS: dict[str, str] = {
    "chain_base": "Chain",
    "grid_base": "Grid",
    "taxi_base": "Taxi",
}

#: Algorithms.
ALGORITHMS: list[str] = ["QLearning", "ExpectedSARSA"]

#: Algorithm display labels.
ALGO_LABELS: dict[str, str] = {
    "QLearning": "Q-Learning",
    "ExpectedSARSA": "Expected SARSA",
}

#: Consistent color palette (no seaborn).
ALGO_COLORS: dict[str, str] = {
    "QLearning": "#1f77b4",       # blue
    "ExpectedSARSA": "#ff7f0e",   # orange
}

#: CI band alpha.
CI_ALPHA: float = 0.20

#: Default paper-suite root (relative to repo root).
_DEFAULT_PAPER_SUITE_ROOT = "results/weighted_lse_dp/phase1/paper_suite"

#: Default output directory.
_DEFAULT_OUT_DIR = "results/weighted_lse_dp/processed/phase1/figures"

#: Output filenames (stem).
_OUT_STEM = "phase1_rl_learning_curves"


# ---------------------------------------------------------------------------
# NeurIPS-compliant matplotlib rc overrides
# ---------------------------------------------------------------------------


def _apply_paper_rc() -> None:
    """Set matplotlib rcParams for NeurIPS submission figures.

    Type 42 (TrueType) fonts are used to satisfy the conference
    requirement of Type 1 or Type 42 fonts in PDF output.
    """
    plt.rcParams.update({
        # Fonts
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # PDF font type (Type 42 = TrueType outlines)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Layout
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        # Misc
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_paper_suite_data(
    root: Path,
) -> dict[str, dict[str, dict[int, dict[str, np.ndarray]]]]:
    """Load curves.npz files from the paper-suite directory tree.

    Expected layout::

        root/<task>/<algorithm>/seed_<seed>/curves.npz

    Returns
    -------
    dict
        ``data[task][algorithm][seed]`` is a dict with keys
        ``"checkpoints"`` and ``"disc_return_mean"``.
    """
    data: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {}

    for task in TASKS:
        data[task] = {}
        for algo in ALGORITHMS:
            data[task][algo] = {}
            algo_dir = root / task / algo
            if not algo_dir.is_dir():
                continue
            for seed_dir in sorted(algo_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                name = seed_dir.name
                if not name.startswith("seed_"):
                    continue
                try:
                    seed = int(name.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                curves_path = seed_dir / "curves.npz"
                if not curves_path.exists():
                    continue
                npz = np.load(curves_path)
                data[task][algo][seed] = {
                    "checkpoints": np.asarray(npz["checkpoints"]),
                    "disc_return_mean": np.asarray(npz["disc_return_mean"]),
                }
    return data


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def _generate_demo_data(
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, dict[int, dict[str, np.ndarray]]]]:
    """Generate synthetic learning-curve data for ``--demo`` mode.

    Three tasks, two algorithms, three seeds each.  Curves are smooth
    saturating functions with per-seed noise to give a plausible CI band.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_checkpoints = 50
    checkpoints = np.arange(1, n_checkpoints + 1) * 1000  # env steps

    # Asymptotic performance and rate vary by (task, algo).
    profiles: dict[str, dict[str, tuple[float, float, float]]] = {
        # (asymptote, rate, noise_scale)
        "chain_base": {
            "QLearning": (0.85, 0.08, 0.04),
            "ExpectedSARSA": (0.82, 0.06, 0.04),
        },
        "grid_base": {
            "QLearning": (0.70, 0.05, 0.05),
            "ExpectedSARSA": (0.72, 0.06, 0.05),
        },
        "taxi_base": {
            "QLearning": (6.0, 0.04, 0.8),
            "ExpectedSARSA": (5.5, 0.035, 0.8),
        },
    }

    seeds = [11, 29, 47]
    data: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {}

    for task in TASKS:
        data[task] = {}
        for algo in ALGORITHMS:
            data[task][algo] = {}
            asym, rate, noise = profiles[task][algo]
            for seed in seeds:
                seed_rng = np.random.default_rng(seed + hash(task + algo) % 2**31)
                t_norm = np.arange(n_checkpoints, dtype=float)
                curve = asym * (1.0 - np.exp(-rate * t_norm))
                curve += noise * seed_rng.standard_normal(n_checkpoints).cumsum() * 0.02
                # Ensure monotone-ish by light smoothing.
                kernel = np.ones(3) / 3.0
                curve = np.convolve(curve, kernel, mode="same")
                data[task][algo][seed] = {
                    "checkpoints": checkpoints.copy(),
                    "disc_return_mean": curve,
                }
    return data


# ---------------------------------------------------------------------------
# Core figure function
# ---------------------------------------------------------------------------


def make_rl_learning_curves_figure(
    data: dict[str, dict[str, dict[int, Any]]],
    *,
    out_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Create the Phase I RL learning-curves figure.

    Parameters
    ----------
    data:
        ``data[task][algorithm][seed]`` is a dict with keys
        ``"checkpoints"`` (1-D int64) and ``"disc_return_mean"``
        (1-D float64).
    out_path:
        If provided, save the figure to this path (without extension).
        Both ``.pdf`` and ``.png`` versions are written.
    show:
        If ``True``, call ``plt.show()`` after rendering.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_paper_rc()

    n_tasks = len(TASKS)
    fig, axes = plt.subplots(
        1, n_tasks,
        figsize=(3.2 * n_tasks, 2.6),
        constrained_layout=True,
    )
    if n_tasks == 1:
        axes = [axes]

    rng = np.random.default_rng(0)  # fixed seed for bootstrap reproducibility

    for ax, task in zip(axes, TASKS):
        task_data = data.get(task, {})
        for algo in ALGORITHMS:
            algo_data = task_data.get(algo, {})
            if not algo_data:
                continue

            # Stack seed curves: shape (n_seeds, n_checkpoints)
            seed_ids = sorted(algo_data.keys())
            checkpoints = algo_data[seed_ids[0]]["checkpoints"]
            curves = np.stack(
                [algo_data[s]["disc_return_mean"] for s in seed_ids],
                axis=0,
            )

            n_seeds = curves.shape[0]
            if n_seeds >= 2:
                agg = aggregate(
                    curves, axis=0, ci_level=0.95,
                    n_bootstrap=10_000, rng=np.random.default_rng(0),
                )
                mean = agg["mean"]
                ci_lo = agg["ci_low"]
                ci_hi = agg["ci_high"]
            else:
                mean = curves[0]
                ci_lo = ci_hi = mean

            color = ALGO_COLORS[algo]
            label = ALGO_LABELS[algo]
            ax.plot(checkpoints, mean, color=color, label=label)
            ax.fill_between(
                checkpoints, ci_lo, ci_hi,
                color=color, alpha=CI_ALPHA,
            )

        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Env steps")
        if ax is axes[0]:
            ax.set_ylabel("Discounted return")
        ax.grid(True, linestyle="--")
        ax.legend(loc="lower right", framealpha=0.7)

    fig.suptitle(
        "RL Learning Curves (mean $\\pm$ 95% bootstrap CI)",
        fontsize=10,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path.with_suffix(".pdf")))
        fig.savefig(str(out_path.with_suffix(".png")))
        print(f"Saved: {out_path.with_suffix('.pdf')}")
        print(f"Saved: {out_path.with_suffix('.png')}")

    if show:
        plt.rcParams["backend"] = "TkAgg"
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase I figure: RL learning curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper-suite-root",
        type=Path,
        default=_REPO_ROOT / _DEFAULT_PAPER_SUITE_ROOT,
        help="Root of paper_suite RL results.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / _DEFAULT_OUT_DIR,
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic data instead of real results.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    args = parser.parse_args()

    if args.demo:
        print("Running in --demo mode with synthetic data.")
        data = _generate_demo_data()
    else:
        root = args.paper_suite_root
        if not root.is_dir():
            print(
                f"ERROR: paper_suite root not found: {root}\n"
                "  Run with --demo for synthetic data, or specify "
                "--paper-suite-root.",
                file=sys.stderr,
            )
            sys.exit(1)
        data = load_paper_suite_data(root)
        # Validate that we got something.
        n_loaded = sum(
            len(seeds)
            for task_d in data.values()
            for seeds in task_d.values()
        )
        if n_loaded == 0:
            print(
                "ERROR: no curves.npz files found under "
                f"{root}. Check directory structure.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loaded {n_loaded} (task, algorithm, seed) curves.")

    out_path = args.out_dir / _OUT_STEM
    fig = make_rl_learning_curves_figure(
        data, out_path=out_path, show=args.show,
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
