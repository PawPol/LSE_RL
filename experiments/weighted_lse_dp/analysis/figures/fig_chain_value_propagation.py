#!/usr/bin/env python
"""Chain value propagation: stagewise value profile across VI sweeps.

Produces a 2x3 grid of heatmaps (one per selected sweep) showing the
value function V[t, s] for the chain_base task under classical value
iteration.  Each subplot has stage t on the y-axis and state s on the
x-axis, with a shared colorbar.

Confidence intervals: not applicable (single-seed exact DP; no
stochastic element).

Data source
-----------
``results/weighted_lse_dp/paper_suite/chain_base/VI/seed_11/curves.npz``
key ``v_table_snapshots`` -- shape ``(n_sweeps, H+1, S)`` float64.

When ``--demo`` is passed, synthetic data is generated that mimics the
progressive rightward propagation of value in a 25-state chain with
horizon 60.

Output
------
``figures/phase1_chain_value_propagation.{pdf,png}`` in the chosen
output directory (default:
``results/weighted_lse_dp/processed/phase1/figures/``).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend; overridden if --show
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_DATA_PATH = (
    _REPO_ROOT
    / "results"
    / "weighted_lse_dp"
    / "paper_suite"
    / "chain_base"
    / "VI"
    / "seed_11"
    / "curves.npz"
)
_DEFAULT_OUT_DIR = (
    _REPO_ROOT
    / "results"
    / "weighted_lse_dp"
    / "processed"
    / "phase1"
    / "figures"
)


# ---------------------------------------------------------------------------
# Style configuration (NeurIPS-compatible: Type 1/42 fonts, no seaborn)
# ---------------------------------------------------------------------------
def _apply_paper_style() -> None:
    """Set matplotlib rcParams for NeurIPS-style figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # Type 42 (TrueType) for NeurIPS
        "ps.fonttype": 42,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Sweep index selection
# ---------------------------------------------------------------------------
def _default_sweep_indices(n_sweeps: int) -> list[int]:
    """Pick up to 6 representative sweep indices.

    Strategy: first sweep, a few intermediate ones spaced roughly
    logarithmically, and the final sweep.  Returns 0-based indices.
    """
    if n_sweeps <= 6:
        return list(range(n_sweeps))

    # Include sweep 0, 1, and then ~log-spaced intermediates, plus last.
    candidates = sorted(set([
        0,
        1,
        max(1, n_sweeps // 10),
        max(1, n_sweeps // 4),
        max(1, n_sweeps // 2),
        n_sweeps - 1,
    ]))
    # Trim to at most 6.
    if len(candidates) > 6:
        candidates = candidates[:5] + [candidates[-1]]
    return candidates


# ---------------------------------------------------------------------------
# Demo data synthesis
# ---------------------------------------------------------------------------
def _synthesize_demo_data(
    n_sweeps: int = 30,
    H: int = 60,
    S: int = 25,
    gamma: float = 0.99,
    rng_seed: int = 42,
) -> np.ndarray:
    """Create synthetic V-table snapshots mimicking chain VI convergence.

    The synthetic value function for each sweep k is:
        V[t, s] = gamma^(S-1-s) * (1 - exp(-alpha_k * s)) * time_factor(t)
    where alpha_k increases with k, producing a rightward propagation
    of value that becomes sharper with more sweeps.

    Returns shape ``(n_sweeps, H+1, S)`` float64.
    """
    rng = np.random.default_rng(rng_seed)

    snapshots = np.zeros((n_sweeps, H + 1, S), dtype=np.float64)

    for k in range(n_sweeps):
        # Propagation strength increases with sweep count.
        alpha = 0.05 + 0.35 * (k / max(1, n_sweeps - 1))

        for s in range(S):
            # Value increases as the agent gets closer to goal (state S-1).
            spatial = (1.0 - np.exp(-alpha * s)) * gamma ** max(0, S - 1 - s)

            for t in range(H + 1):
                # Earlier stages have lower value (more steps remaining
                # does not help if too far from goal; but near goal it
                # does). Simplified: V decreases as t increases
                # (fewer steps left).
                remaining = H - t
                time_factor = 1.0 - np.exp(-0.05 * remaining)
                snapshots[k, t, s] = spatial * time_factor

    # Add tiny noise for realism.
    snapshots += rng.normal(0, 1e-4, size=snapshots.shape)
    snapshots = np.clip(snapshots, 0.0, None)

    return snapshots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_chain_value_propagation_figure(
    v_snapshots: np.ndarray,  # (n_sweeps, H+1, S)
    *,
    sweep_indices: list[int] | None = None,
    out_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Create the chain value propagation figure.

    Parameters
    ----------
    v_snapshots : np.ndarray
        Value table snapshots, shape ``(n_sweeps, H+1, S)``.
    sweep_indices : list[int] | None
        Which sweep indices to plot.  If ``None``, a default selection
        of up to 6 sweeps is chosen (first, a few intermediate, last).
    out_path : Path | None
        If provided, the figure is saved to this path.  The format is
        inferred from the suffix.  When the suffix is ``.pdf``, a
        companion ``.png`` is also saved.
    show : bool
        If ``True``, call ``plt.show()`` after rendering.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if v_snapshots.ndim != 3:
        raise ValueError(
            f"v_snapshots must be 3-D (n_sweeps, H+1, S), "
            f"got shape {v_snapshots.shape!r}"
        )

    n_sweeps, Hp1, S = v_snapshots.shape
    H = Hp1 - 1

    if sweep_indices is None:
        sweep_indices = _default_sweep_indices(n_sweeps)

    # Validate indices.
    for idx in sweep_indices:
        if idx < 0 or idx >= n_sweeps:
            raise IndexError(
                f"sweep index {idx} out of range for {n_sweeps} sweeps"
            )

    n_panels = len(sweep_indices)
    if n_panels == 0:
        raise ValueError("sweep_indices is empty; nothing to plot")

    # Grid layout: aim for 2 rows x 3 cols, adjust for fewer panels.
    if n_panels <= 3:
        nrows, ncols = 1, n_panels
    elif n_panels <= 6:
        nrows = 2
        ncols = (n_panels + 1) // 2
    else:
        nrows = 2
        ncols = 3
        sweep_indices = sweep_indices[:6]
        n_panels = 6

    _apply_paper_style()

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.0 * ncols, 2.5 * nrows + 0.6),
        constrained_layout=True,
        squeeze=False,
    )

    # Global color normalization across all selected panels.
    vmin = float(np.min(v_snapshots[sweep_indices]))
    vmax = float(np.max(v_snapshots[sweep_indices]))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = None
    for panel_idx, sweep_idx in enumerate(sweep_indices):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row, col]

        # V[t, s] with stage t on y-axis, state s on x-axis.
        data = v_snapshots[sweep_idx]  # (H+1, S)

        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(f"Sweep {sweep_idx + 1}", fontsize=9)
        ax.set_xlabel("State $s$")
        ax.set_ylabel("Stage $t$")

        # Tick reduction for readability.
        state_ticks = np.linspace(0, S - 1, min(6, S), dtype=int)
        ax.set_xticks(state_ticks)
        ax.set_xticklabels(state_ticks)

        stage_ticks = np.linspace(0, H, min(5, H + 1), dtype=int)
        ax.set_yticks(stage_ticks)
        ax.set_yticklabels(stage_ticks)

    # Hide unused subplots.
    for panel_idx in range(n_panels, nrows * ncols):
        row = panel_idx // ncols
        col = panel_idx % ncols
        axes[row, col].set_visible(False)

    # Shared colorbar.
    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02
        )
        cbar.set_label("$V(t, s)$", fontsize=9)

    fig.suptitle(
        "Chain value propagation: $V(t, s)$ across VI sweeps",
        fontsize=11,
        y=1.02,
    )

    # Save.
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Pin PDF timestamp for reproducibility.
        save_kwargs: dict = {}
        if out_path.suffix == ".pdf":
            save_kwargs["metadata"] = {"CreationDate": None}

        fig.savefig(out_path, bbox_inches="tight", **save_kwargs)
        print(f"Saved: {out_path}")

        # Also save companion PNG if primary is PDF.
        if out_path.suffix == ".pdf":
            png_path = out_path.with_suffix(".png")
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            print(f"Saved: {png_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chain value propagation figure (Phase I, spec 10.1.1).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=_DEFAULT_DATA_PATH,
        help="Path to curves.npz containing v_table_snapshots.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Output directory for the figure files.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic data instead of loading from disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after rendering.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.demo:
        print("Demo mode: synthesizing data for chain_base (S=25, H=60).")
        v_snapshots = _synthesize_demo_data()
    else:
        data_path = Path(args.data_path)
        if not data_path.is_file():
            print(
                f"ERROR: data file not found: {data_path}\n"
                f"Use --demo to generate synthetic data.",
                file=sys.stderr,
            )
            sys.exit(1)
        with np.load(data_path, allow_pickle=True) as npz:
            v_snapshots = npz["v_table_snapshots"]
        if v_snapshots.ndim != 3:
            print(
                f"ERROR: v_table_snapshots has unexpected shape "
                f"{v_snapshots.shape!r}; expected 3-D (n_sweeps, H+1, S).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Loaded v_table_snapshots: shape {v_snapshots.shape} "
            f"from {data_path}"
        )

    out_dir = Path(args.out_dir)
    out_path = out_dir / "phase1_chain_value_propagation.pdf"

    fig = make_chain_value_propagation_figure(
        v_snapshots,
        out_path=out_path,
        show=args.show,
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
