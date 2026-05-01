"""Phase VIII Contraction-UCB figures (spec §12.1 panel 8, M10 deliverable).

Produces:
    contraction_ucb_arm_probs.pdf       — per (game, subcase), per-arm
                                          pull-fraction (rolling mean over a
                                          50-episode window) over episodes
                                          for the ContractionUCB method.
    contraction_ucb_learning_curves.pdf — ContractionUCB vs ReturnUCB vs
                                          vanilla learning curves
                                          (return vs episode, mean ± SE).

Input contract: long-CSV per
``experiments.adaptive_beta.tab_six_games.analysis.aggregate``. Columns
read here: ``game, subcase, method, seed, episode, ucb_arm_index,
return``.

Spec references: §12.1, §12.3 contraction-adaptive table, §10.6.
Lessons: #1 (use repo `.venv`), #16 (no --demo synth path; production
read of pandas DataFrames only).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_CONTRACTION_METHOD = "contraction_ucb"
_LEARNING_METHODS: tuple[str, ...] = ("contraction_ucb", "return_ucb", "vanilla")
_ROLLING_WINDOW = 50


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """1-D rolling mean with reflect padding so output length == input length."""
    if values.size == 0:
        return values
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values.astype(float), pad_width=pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad : pad + values.size]


def _plot_arm_probs(df: pd.DataFrame, out_path: Path) -> Path:
    contraction = df[df["method"] == _CONTRACTION_METHOD]
    cells = sorted({(g, s) for g, s in zip(contraction["game"], contraction["subcase"])})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cells:
        return out_path

    n = len(cells)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows), squeeze=False)

    arm_indices = sorted(int(a) for a in contraction["ucb_arm_index"].dropna().unique() if a >= 0)
    cmap = plt.get_cmap("tab10")

    for ax, (game, subcase) in zip(axes.ravel(), cells):
        cell = contraction[
            (contraction["game"] == game) & (contraction["subcase"] == subcase)
        ]
        if cell.empty or not arm_indices:
            ax.set_title(f"{game} / {subcase} (no arm data)")
            ax.set_xlabel("episode")
            ax.set_ylabel("pull fraction")
            continue
        episodes = np.asarray(sorted(cell["episode"].unique()), dtype=int)
        # Aggregate seed votes per (episode, arm): pull-fraction = #seeds that
        # selected this arm at this episode / #seeds present at this episode.
        for k_idx, arm in enumerate(arm_indices):
            fractions = np.zeros_like(episodes, dtype=float)
            for i, ep in enumerate(episodes):
                ep_slice = cell[cell["episode"] == ep]
                if ep_slice.empty:
                    continue
                fractions[i] = float((ep_slice["ucb_arm_index"] == arm).mean())
            smoothed = _rolling_mean(fractions, _ROLLING_WINDOW)
            ax.plot(
                episodes,
                smoothed,
                label=f"arm {arm}",
                color=cmap(k_idx % cmap.N),
                linewidth=1.2,
            )
        ax.set_title(f"{game} / {subcase}")
        ax.set_xlabel("episode")
        ax.set_ylabel("pull fraction (rolling)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)

    for ax in axes.ravel()[len(cells):]:
        ax.set_visible(False)

    fig.suptitle("ContractionUCB — per-arm pull fractions (rolling mean)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_learning_curves(df: pd.DataFrame, out_path: Path) -> Path:
    sub = df[df["method"].isin(_LEARNING_METHODS)]
    cells = sorted({(g, s) for g, s in zip(sub["game"], sub["subcase"])})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cells:
        return out_path

    n = len(cells)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows), squeeze=False)

    cmap = plt.get_cmap("tab10")
    method_to_color = {m: cmap(i) for i, m in enumerate(_LEARNING_METHODS)}

    for ax, (game, subcase) in zip(axes.ravel(), cells):
        cell = sub[(sub["game"] == game) & (sub["subcase"] == subcase)]
        for method in _LEARNING_METHODS:
            method_slice = cell[cell["method"] == method]
            if method_slice.empty:
                continue
            grouped = method_slice.groupby("episode")["return"]
            eps = np.asarray(sorted(grouped.groups.keys()), dtype=float)
            means = np.array(
                [grouped.get_group(int(e)).mean() for e in eps], dtype=float
            )

            def _se(values: np.ndarray) -> float:
                v = values[~np.isnan(values)]
                if v.size < 2:
                    return 0.0
                return float(np.std(v, ddof=1) / np.sqrt(v.size))

            ses = np.array(
                [_se(grouped.get_group(int(e)).to_numpy()) for e in eps], dtype=float
            )
            color = method_to_color[method]
            ax.plot(eps, means, label=method, color=color, linewidth=1.4)
            ax.fill_between(eps, means - ses, means + ses, color=color, alpha=0.2)
        ax.set_title(f"{game} / {subcase}")
        ax.set_xlabel("episode")
        ax.set_ylabel("return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for ax in axes.ravel()[len(cells):]:
        ax.set_visible(False)

    fig.suptitle("Contraction-UCB vs Return-UCB vs vanilla (mean ± SE)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_figure(
    processed_long_csv: Path,
    out_dir: Path,
    *,
    env_filter: Optional[str] = None,
) -> Dict[str, Path]:
    """Render the two contraction-related panels.

    Returns
    -------
    dict[str, Path]
        ``{"contraction_ucb_arm_probs": ..., "contraction_ucb_learning_curves": ...}``.
    """
    df = pd.read_csv(processed_long_csv)
    if env_filter is not None:
        df = df[df["game"].astype(str).str.contains(env_filter, regex=False)]

    arm_path = _plot_arm_probs(df, out_dir / "figures" / "contraction_ucb_arm_probs.pdf")
    lc_path = _plot_learning_curves(
        df, out_dir / "figures" / "contraction_ucb_learning_curves.pdf"
    )
    return {
        "contraction_ucb_arm_probs": arm_path,
        "contraction_ucb_learning_curves": lc_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-long-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--env-filter", type=str, default=None)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    paths = make_figure(args.processed_long_csv, args.out_dir, env_filter=args.env_filter)
    for name, path in paths.items():
        print(f"{name}: {path}")
