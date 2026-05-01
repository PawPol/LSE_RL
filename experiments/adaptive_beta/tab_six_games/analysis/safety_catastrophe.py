"""Phase VIII safety / catastrophe panels (spec §12.1 panel 10).

Produces (cross-cutting; any stage):
    catastrophic_episodes_per_method.pdf  — bar chart, count of
                                            ``catastrophic`` episodes per
                                            (method, game).
    beta_clip_frequency.pdf               — bar chart, mean
                                            ``beta_clip_frequency`` per
                                            (method, game).
    worst_window_return_percentile.pdf    — bar chart, p5 of
                                            ``worst_window_return_percentile``
                                            per (method, game).

Input contract: long-CSV per
``experiments.adaptive_beta.tab_six_games.analysis.aggregate``.
Required columns: ``game, method, catastrophic, beta_clip_frequency,
worst_window_return_percentile``.

Spec references: §12.1 panel 10, §12.2 main_safety_catastrophe.pdf.
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


def _grouped_bar(
    df: pd.DataFrame,
    *,
    value_col: str,
    agg: str,  # 'sum', 'mean', or 'p5'
    title: str,
    ylabel: str,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or value_col not in df.columns:
        return out_path

    games = sorted(df["game"].unique())
    methods = sorted(df["method"].unique())
    if not games or not methods:
        return out_path

    cmap = plt.get_cmap("tab10")
    method_to_color = {m: cmap(i % cmap.N) for i, m in enumerate(methods)}

    n_games = len(games)
    n_methods = len(methods)
    bar_w = 0.8 / max(n_methods, 1)
    x_base = np.arange(n_games, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * n_games + 2.0), 4.0))
    for j, method in enumerate(methods):
        heights = []
        for game in games:
            cell = df[(df["game"] == game) & (df["method"] == method)][value_col]
            cell = cell.dropna()
            if cell.empty:
                heights.append(0.0)
                continue
            if agg == "sum":
                heights.append(float(cell.sum()))
            elif agg == "mean":
                heights.append(float(cell.mean()))
            elif agg == "p5":
                heights.append(float(np.percentile(cell.to_numpy(), 5)))
            else:
                raise ValueError(f"unknown agg={agg!r}")
        offset = (j - (n_methods - 1) / 2.0) * bar_w
        ax.bar(
            x_base + offset,
            heights,
            width=bar_w,
            color=method_to_color[method],
            label=method,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(games, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
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
    """Render the three safety / catastrophe bar panels.

    Returns
    -------
    dict[str, Path]
        ``{"catastrophic_episodes_per_method": ..., "beta_clip_frequency": ...,
        "worst_window_return_percentile": ...}``.
    """
    df = pd.read_csv(processed_long_csv)
    if env_filter is not None:
        df = df[df["game"].astype(str).str.contains(env_filter, regex=False)]

    catastrophic_path = _grouped_bar(
        df,
        value_col="catastrophic",
        agg="sum",
        title="Catastrophic episodes per method × game",
        ylabel="count",
        out_path=out_dir / "figures" / "catastrophic_episodes_per_method.pdf",
    )
    clip_path = _grouped_bar(
        df,
        value_col="beta_clip_frequency",
        agg="mean",
        title="β-clip frequency per method × game",
        ylabel="mean β-clip freq",
        out_path=out_dir / "figures" / "beta_clip_frequency.pdf",
    )
    worst_path = _grouped_bar(
        df,
        value_col="worst_window_return_percentile",
        agg="p5",
        title="p5 worst-window return per method × game",
        ylabel="p5 worst-window return",
        out_path=out_dir / "figures" / "worst_window_return_percentile.pdf",
    )
    return {
        "catastrophic_episodes_per_method": catastrophic_path,
        "beta_clip_frequency": clip_path,
        "worst_window_return_percentile": worst_path,
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
