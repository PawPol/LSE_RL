"""Phase VIII main learning curves (spec §12.1 panel 3, M7 deliverable).

Produces:
    main_learning_curves.pdf — per (game, subcase) subplot, one curve per
                               method (return vs episode), shaded with the
                               seed mean ± SE band.

Input contract: same long-CSV schema as
``experiments.adaptive_beta.tab_six_games.analysis.beta_sweep_plots``;
columns consumed here are ``game, subcase, method, seed, episode, return``.

Spec references: §12.1, §12.3 fixed-TAB-vs-baselines table, §10.3.
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


def _curve_with_se(
    sub: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a (method, game, subcase) slice into mean ± SE per episode.

    Returns
    -------
    episodes, mean, se :
        1-D arrays aligned over the episode axis. Episodes with fewer than
        2 contributing seeds report SE = 0.
    """
    grouped = sub.groupby("episode")["return"]
    episodes = np.asarray(sorted(grouped.groups.keys()), dtype=float)
    means = np.array([grouped.get_group(int(e)).mean() for e in episodes], dtype=float)

    def _se(values: np.ndarray) -> float:
        v = values[~np.isnan(values)]
        if v.size < 2:
            return 0.0
        return float(np.std(v, ddof=1) / np.sqrt(v.size))

    ses = np.array(
        [_se(grouped.get_group(int(e)).to_numpy()) for e in episodes], dtype=float
    )
    return episodes, means, ses


def make_figure(
    processed_long_csv: Path,
    out_dir: Path,
    *,
    env_filter: Optional[str] = None,
) -> Dict[str, Path]:
    """Render the main learning-curves panel.

    Parameters
    ----------
    processed_long_csv:
        Long-form per-episode CSV emitted by the M5 W1.A aggregator.
    out_dir:
        Output root; the PDF is written to ``out_dir / 'figures' /
        'main_learning_curves.pdf'``.
    env_filter:
        Optional substring matched against ``game``.

    Returns
    -------
    dict[str, Path]
        ``{"main_learning_curves": <pdf path>}``.
    """
    df = pd.read_csv(processed_long_csv)
    if env_filter is not None:
        df = df[df["game"].astype(str).str.contains(env_filter, regex=False)]

    cells = sorted({(g, s) for g, s in zip(df["game"], df["subcase"])})
    out_path = out_dir / "figures" / "main_learning_curves.pdf"

    if not cells:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return {"main_learning_curves": out_path}

    n = len(cells)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows), squeeze=False)

    cmap = plt.get_cmap("tab10")
    method_to_color: Dict[str, tuple[float, float, float, float]] = {}

    for ax, (game, subcase) in zip(axes.ravel(), cells):
        cell = df[(df["game"] == game) & (df["subcase"] == subcase)]
        for method in sorted(cell["method"].unique()):
            color = method_to_color.setdefault(
                method, cmap(len(method_to_color) % cmap.N)
            )
            episodes, means, ses = _curve_with_se(cell[cell["method"] == method])
            if episodes.size == 0:
                continue
            ax.plot(episodes, means, label=method, color=color, linewidth=1.4)
            ax.fill_between(episodes, means - ses, means + ses, alpha=0.2, color=color)
        ax.set_title(f"{game} / {subcase}")
        ax.set_xlabel("episode")
        ax.set_ylabel("return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for ax in axes.ravel()[len(cells):]:
        ax.set_visible(False)

    fig.suptitle("Phase VIII main learning curves (mean ± SE over seeds)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return {"main_learning_curves": out_path}


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
