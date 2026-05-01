"""Phase VIII sign-switching figures (spec §12.1 panels 5-7, M9 deliverable).

Produces:
    switch_aligned_return.pdf — episodes aligned to ``switch_event`` (window:
                                ±50 around the switch), return curves per
                                method (mean ± SE across switch instances
                                and seeds).
    switch_aligned_beta.pdf   — same alignment, ``oracle_beta`` (proxy for
                                deployed-β trajectory; the long CSV ships
                                oracle_beta and beta_sign_correct, see
                                §7.4 schema).
    beta_sign_accuracy.pdf    — ``beta_sign_correct`` vs episode (mean ± SE
                                across seeds), per method. The oracle is
                                always 1.0 by construction.

Input contract: long-CSV per
``experiments.adaptive_beta.tab_six_games.analysis.aggregate``.
Required columns: ``game, subcase, method, seed, episode, return,
switch_event, beta_sign_correct, oracle_beta, episodes_since_switch``.

Spec references: §12.1, §12.3 sign-switching adaptive table, §10.5.
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


_WINDOW = 50  # episodes on either side of a switch event


def _aligned_window(
    df: pd.DataFrame, value_col: str, window: int = _WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack value-column traces around each ``switch_event``.

    Returns
    -------
    offsets, mean, se :
        ``offsets`` ranges over ``[-window, +window]``; ``mean`` and ``se``
        are aggregated across (seed, switch instance) within ``df``.
    """
    offsets = np.arange(-window, window + 1, dtype=int)
    if df.empty or "switch_event" not in df.columns:
        return offsets, np.zeros_like(offsets, dtype=float), np.zeros_like(offsets, dtype=float)

    traces: list[np.ndarray] = []
    for (_run_id, _seed), seed_group in df.groupby(["run_id", "seed"], sort=False):
        seed_group = seed_group.sort_values("episode")
        switch_episodes = seed_group.loc[seed_group["switch_event"] == 1, "episode"].to_numpy()
        for s_ep in switch_episodes:
            trace = np.full(offsets.shape, np.nan, dtype=float)
            for i, off in enumerate(offsets):
                target = int(s_ep) + int(off)
                row = seed_group[seed_group["episode"] == target]
                if not row.empty:
                    trace[i] = float(row[value_col].iloc[0])
            traces.append(trace)

    if not traces:
        return offsets, np.zeros_like(offsets, dtype=float), np.zeros_like(offsets, dtype=float)

    stacked = np.vstack(traces)
    counts = np.sum(~np.isnan(stacked), axis=0)
    mean = np.zeros(offsets.shape, dtype=float)
    se = np.zeros(offsets.shape, dtype=float)
    valid = counts > 0
    if valid.any():
        with np.errstate(invalid="ignore"):
            mean[valid] = np.nanmean(stacked[:, valid], axis=0)
        valid_se = counts > 1
        if valid_se.any():
            with np.errstate(invalid="ignore"):
                std = np.nanstd(stacked[:, valid_se], axis=0, ddof=1)
            se[valid_se] = std / np.sqrt(counts[valid_se])
    return offsets, mean, se


def _plot_aligned(df: pd.DataFrame, value_col: str, ylabel: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted(df["method"].unique()) if not df.empty else []
    if not methods:
        return out_path

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    cmap = plt.get_cmap("tab10")
    plotted = 0
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        offsets, mean, se = _aligned_window(sub, value_col)
        if np.all(mean == 0.0) and np.all(se == 0.0):
            continue
        color = cmap(i % cmap.N)
        ax.plot(offsets, mean, label=method, color=color, linewidth=1.4)
        ax.fill_between(offsets, mean - se, mean + se, color=color, alpha=0.2)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return out_path

    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8, label="switch")
    ax.set_xlabel("episodes from switch event")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Switch-aligned {ylabel}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_beta_sign_accuracy(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "beta_sign_correct" not in df.columns:
        return out_path

    methods = sorted(df["method"].unique())
    if not methods:
        return out_path

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    cmap = plt.get_cmap("tab10")
    plotted = 0
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        grouped = sub.groupby("episode")["beta_sign_correct"]
        episodes = np.asarray(sorted(grouped.groups.keys()), dtype=float)
        if episodes.size == 0:
            continue
        means = np.array(
            [grouped.get_group(int(e)).mean() for e in episodes], dtype=float
        )

        def _se(values: np.ndarray) -> float:
            v = values[~np.isnan(values)]
            if v.size < 2:
                return 0.0
            return float(np.std(v, ddof=1) / np.sqrt(v.size))

        ses = np.array(
            [_se(grouped.get_group(int(e)).to_numpy()) for e in episodes], dtype=float
        )
        color = cmap(i % cmap.N)
        ax.plot(episodes, means, label=method, color=color, linewidth=1.4)
        ax.fill_between(episodes, means - ses, means + ses, color=color, alpha=0.2)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return out_path

    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8, label="oracle (1.0)")
    ax.set_xlabel("episode")
    ax.set_ylabel("mean β-sign correct")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("β-sign accuracy across training", fontsize=11)
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
    """Render the three sign-switching panels.

    Returns
    -------
    dict[str, Path]
        ``{"switch_aligned_return": ..., "switch_aligned_beta": ...,
        "beta_sign_accuracy": ...}``.
    """
    df = pd.read_csv(processed_long_csv)
    if env_filter is not None:
        df = df[df["game"].astype(str).str.contains(env_filter, regex=False)]

    return_path = _plot_aligned(
        df,
        value_col="return",
        ylabel="return",
        out_path=out_dir / "figures" / "switch_aligned_return.pdf",
    )
    # Use ``oracle_beta`` as the deployed-β trajectory proxy; the long CSV
    # ships it (§7.4) and it equals the schedule's β output on each episode.
    beta_path = _plot_aligned(
        df,
        value_col="oracle_beta",
        ylabel="β (deployed)",
        out_path=out_dir / "figures" / "switch_aligned_beta.pdf",
    )
    accuracy_path = _plot_beta_sign_accuracy(
        df, out_path=out_dir / "figures" / "beta_sign_accuracy.pdf"
    )
    return {
        "switch_aligned_return": return_path,
        "switch_aligned_beta": beta_path,
        "beta_sign_accuracy": accuracy_path,
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
