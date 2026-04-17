#!/usr/bin/env python
"""Phase I figure: Bellman residual vs sweep index for DP algorithms.

Renders one subplot per mandatory task (chain_base, grid_base, taxi_base)
with VI, PI, MPI, and AsyncVI overlaid.  PE is excluded because it
performs only a single sweep.

Confidence intervals: when multiple seeds are present, the plot shows
the mean curve with a shaded band spanning the per-seed min/max envelope.
No bootstrap is applied because the number of seeds (3) is too small for
meaningful percentile CIs; the envelope is conservative.

No cherry-picking: all seeds found under the paper_suite directory for
each (task, algorithm) pair are loaded.  If a ``--seed`` filter is
supplied, only that seed is used (documented in the CLI help).

Regeneratability: given the same ``results/processed/`` (or ``--demo``
flag with the same code revision), the script produces identical output.
The random seed for ``--demo`` synthetic data is fixed at 42.

Output: ``figures/phase1_dp_residuals.{pdf,png}`` under the chosen
output directory.
"""

from __future__ import annotations

import argparse
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

TASKS: tuple[str, ...] = ("chain_base", "grid_base", "taxi_base")
"""Mandatory tasks in subplot order."""

TASK_LABELS: dict[str, str] = {
    "chain_base": "Chain (25 states)",
    "grid_base": "Grid (25 states)",
    "taxi_base": "Taxi (44 states)",
}

ALGORITHMS: tuple[str, ...] = ("VI", "PI", "MPI", "AsyncVI")
"""DP algorithms to include (PE excluded -- single sweep)."""

ALGO_COLORS: dict[str, str] = {
    "VI": "#1f77b4",       # blue
    "PI": "#ff7f0e",       # orange
    "MPI": "#2ca02c",      # green
    "AsyncVI": "#d62728",  # red
}

ALGO_LABELS: dict[str, str] = {
    "VI": "VI",
    "PI": "PI",
    "MPI": "MPI",
    "AsyncVI": "Async-VI",
}

DEFAULT_PAPER_SUITE_ROOT: Path = (
    _REPO_ROOT / "results" / "weighted_lse_dp" / "phase1" / "paper_suite"
)

DEFAULT_OUT_DIR: Path = (
    _REPO_ROOT
    / "results"
    / "weighted_lse_dp"
    / "processed"
    / "phase1"
    / "figures"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_paper_suite_data(
    root: Path,
    *,
    seed_filter: int | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Load curves.npz for every (task, algorithm, seed) under *root*.

    Returns
    -------
    dict
        ``data[task][algorithm]`` is a dict with keys
        ``"sweep_index"`` (n_sweeps,) int64 and
        ``"bellman_residual"`` (n_seeds, n_sweeps) float64.
        When seeds have different sweep counts the shorter ones are
        padded with their final residual value.
    """
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for task in TASKS:
        data[task] = {}
        for algo in ALGORITHMS:
            algo_dir = root / task / algo
            if not algo_dir.is_dir():
                continue

            # Discover seed directories.
            seed_dirs: list[Path] = sorted(
                p for p in algo_dir.iterdir()
                if p.is_dir() and p.name.startswith("seed_")
            )
            if seed_filter is not None:
                seed_dirs = [
                    p for p in seed_dirs if p.name == f"seed_{seed_filter}"
                ]

            if not seed_dirs:
                continue

            all_sweeps: list[np.ndarray] = []
            all_residuals: list[np.ndarray] = []

            for sd in seed_dirs:
                curves_path = sd / "curves.npz"
                if not curves_path.is_file():
                    continue
                with np.load(curves_path, allow_pickle=False) as npz:
                    if "sweep_index" in npz and "bellman_residual" in npz:
                        all_sweeps.append(np.asarray(npz["sweep_index"]))
                        all_residuals.append(
                            np.asarray(npz["bellman_residual"])
                        )

            if not all_residuals:
                continue

            # Align to the longest sweep sequence by right-padding shorter
            # ones with their last value.
            max_len = max(r.shape[0] for r in all_residuals)
            sweep_index = np.arange(max_len, dtype=np.int64)

            padded: list[np.ndarray] = []
            for r in all_residuals:
                if r.shape[0] < max_len:
                    pad_val = r[-1]
                    r = np.concatenate(
                        [r, np.full(max_len - r.shape[0], pad_val)]
                    )
                padded.append(r)

            residual_matrix = np.stack(padded, axis=0)  # (n_seeds, n_sweeps)

            data[task][algo] = {
                "sweep_index": sweep_index,
                "bellman_residual": residual_matrix,
            }

    return data


def _generate_demo_data() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Generate synthetic exponentially-decaying residuals for demo mode.

    Uses a fixed seed (42) for reproducibility.
    """
    rng = np.random.RandomState(42)
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    # Different sweep counts per algorithm to make the demo realistic.
    algo_params: dict[str, tuple[int, float, float]] = {
        # (n_sweeps, decay_rate, initial_residual)
        "VI": (80, 0.08, 5.0),
        "PI": (12, 0.45, 3.0),
        "MPI": (30, 0.18, 4.0),
        "AsyncVI": (120, 0.05, 6.0),
    }

    n_seeds = 3
    for task in TASKS:
        data[task] = {}
        for algo, (n_sweeps, decay, init) in algo_params.items():
            sweep_index = np.arange(n_sweeps, dtype=np.int64)
            residuals: list[np.ndarray] = []
            for _ in range(n_seeds):
                noise = rng.normal(0, 0.15, size=n_sweeps)
                curve = init * np.exp(-decay * sweep_index) + noise * 0.01
                curve = np.maximum(curve, 1e-16)  # keep positive
                residuals.append(curve)
            data[task][algo] = {
                "sweep_index": sweep_index,
                "bellman_residual": np.stack(residuals, axis=0),
            }

    return data


# ---------------------------------------------------------------------------
# Public figure-generation API
# ---------------------------------------------------------------------------


def make_dp_residuals_figure(
    data: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    out_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Create the Phase I DP-residual convergence figure.

    Parameters
    ----------
    data:
        ``data[task][algorithm]`` maps to a dict with keys
        ``"sweep_index"`` shape ``(n_sweeps,)`` int64 and
        ``"bellman_residual"`` shape ``(n_seeds, n_sweeps)`` float64.
    out_path:
        When provided, the figure is saved as both PDF and PNG at this
        stem (e.g. ``/a/b/fig`` saves ``/a/b/fig.pdf`` and
        ``/a/b/fig.png``).  Parent directories are created as needed.
    show:
        If ``True``, call ``plt.show()`` after rendering.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # -- matplotlib config for NeurIPS submission --------------------------
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,   # Type 42 (TrueType) for NeurIPS
        "ps.fonttype": 42,
        "text.usetex": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })

    fig, axes = plt.subplots(
        1, len(TASKS),
        figsize=(5.5 * len(TASKS) / 3 * 2, 3.0),
        sharey=False,
        constrained_layout=True,
    )
    if len(TASKS) == 1:
        axes = [axes]

    for ax, task in zip(axes, TASKS):
        task_data = data.get(task, {})
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Sweep index")

        for algo in ALGORITHMS:
            if algo not in task_data:
                continue
            entry = task_data[algo]
            sweep_idx = entry["sweep_index"]
            residuals = entry["bellman_residual"]  # (n_seeds, n_sweeps)

            # Ensure 2-D even if a single seed.
            if residuals.ndim == 1:
                residuals = residuals[np.newaxis, :]

            # Compute log10 of absolute residuals (clamp to avoid log(0)).
            log_res = np.log10(np.maximum(np.abs(residuals), 1e-16))

            mean_curve = log_res.mean(axis=0)
            color = ALGO_COLORS[algo]
            label = ALGO_LABELS[algo]

            ax.plot(
                sweep_idx, mean_curve,
                color=color, label=label, linewidth=1.5,
            )

            # Shaded min/max envelope when multiple seeds exist.
            if log_res.shape[0] > 1:
                lo = log_res.min(axis=0)
                hi = log_res.max(axis=0)
                ax.fill_between(
                    sweep_idx, lo, hi,
                    color=color, alpha=0.15, linewidth=0,
                )

    # Y-axis label on leftmost subplot only.
    axes[0].set_ylabel(r"$\log_{10}$(Bellman residual)")

    # Legend in first subplot only.
    axes[0].legend(loc="upper right", framealpha=0.9, edgecolor="0.8")

    # Save if requested.
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stem = out_path.with_suffix("")
        for ext in (".pdf", ".png"):
            save_path = stem.with_suffix(ext)
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase I figure: Bellman residual vs sweep index for DP algorithms."
        ),
    )
    parser.add_argument(
        "--paper-suite-root",
        type=Path,
        default=DEFAULT_PAPER_SUITE_ROOT,
        help=(
            "Root directory of the paper_suite results tree. "
            f"Default: {DEFAULT_PAPER_SUITE_ROOT}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter to a single seed (default: use all available seeds).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for figures. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use synthetic demo data instead of loading from disk. "
            "Useful for verifying the figure layout without real runs."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() after rendering (interactive display).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.demo:
        print("Demo mode: generating synthetic data (seed=42).")
        data = _generate_demo_data()
    else:
        root = Path(args.paper_suite_root)
        if not root.is_dir():
            print(
                f"ERROR: paper_suite root not found: {root}\n"
                "  Hint: run with --demo to generate a figure from synthetic data.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loading data from: {root}")
        data = _load_paper_suite_data(root, seed_filter=args.seed)

    # Check that we have at least one algorithm for at least one task.
    n_loaded = sum(
        len(algo_dict) for algo_dict in data.values()
    )
    if n_loaded == 0:
        print(
            "ERROR: no data loaded. Check the paper_suite_root path or use --demo.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_path = out_dir / "phase1_dp_residuals"

    fig = make_dp_residuals_figure(data, out_path=out_path, show=args.show)

    # Close to free memory in batch mode.
    if not args.show:
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
