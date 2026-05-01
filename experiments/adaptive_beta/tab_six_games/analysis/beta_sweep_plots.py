"""Phase VIII β-sweep figures (spec §12.1 panel 1, M6 deliverable).

Produces:
    beta_vs_auc.pdf          — per game, β on x-axis, AUC mean ± SE on y.
    beta_vs_contraction.pdf  — per game, β on x-axis, mean
                               `bellman_residual` (contraction proxy) on y.

AUC computation
---------------
For each (game, subcase, method, seed) trajectory we integrate the per-
episode `return` curve over `episode` using `numpy.trapezoid`. The seed-
level AUCs are then averaged within each (game, method) to yield the
mean and the seed-bootstrap standard error.

Input contract (long-CSV produced by
`experiments.adaptive_beta.tab_six_games.analysis.aggregate`):

    run_id, config_hash, phase, stage, game, subcase, method, seed, episode,
    return, length, epsilon,
    alignment_rate, mean_signed_alignment, mean_advantage, mean_abs_advantage,
    mean_d_eff, median_d_eff, frac_d_eff_below_gamma, frac_d_eff_above_one,
    bellman_residual, td_target_abs_max, q_abs_max,
    catastrophic, success, regret, shift_event, divergence_event,
    contraction_reward, empirical_contraction_ratio, log_residual_reduction,
    ucb_arm_index, beta_clip_count, beta_clip_frequency,
    recovery_time_after_shift, beta_sign_correct, beta_lag_to_oracle,
    regret_vs_oracle, catastrophic_episodes, worst_window_return_percentile,
    trap_entries, constraint_violations, overflow_count,
    regime, switch_event, episodes_since_switch, oracle_beta,
    nan_count, diverged

Spec references: §12.1, §12.3 main β-sweep table, §10.2 (Stage 1).
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


# ---------------------------------------------------------------------------
# β-method ordering for x-axis. The 7 fixed-β arms are spec §6.4; adaptive
# variants are placed after the fixed grid so β=0 appears in the correct
# position on a numeric axis when present.
# ---------------------------------------------------------------------------
_FIXED_BETA_ARMS = [
    "fixed_beta_-2",
    "fixed_beta_-1",
    "fixed_beta_-0.5",
    "fixed_beta_0",
    "fixed_beta_+0.5",
    "fixed_beta_+1",
    "fixed_beta_+2",
]
_ADAPTIVE_METHODS = [
    "vanilla",
    "contraction_ucb",
    "return_ucb",
    "oracle",
    "hand_adaptive",
]


def _method_to_x(method: str) -> Optional[float]:
    """Map a method label to a numeric β value when one is encoded in the
    label, otherwise return ``None`` (categorical placement)."""
    if method.startswith("fixed_beta_"):
        try:
            return float(method.removeprefix("fixed_beta_"))
        except ValueError:
            return None
    return None


def _seed_auc(group: pd.DataFrame) -> float:
    """Trapezoidal integral of `return` vs `episode` for a single seed."""
    ordered = group.sort_values("episode")
    return float(np.trapezoid(ordered["return"].to_numpy(), ordered["episode"].to_numpy()))


def _agg_mean_se(values: np.ndarray) -> tuple[float, float]:
    """Mean and seed-level standard error (sample SE, ddof=1)."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), 0.0)
    return (float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size)))


def _plot_per_game(
    df: pd.DataFrame,
    *,
    y_col_name: str,
    y_compute: str,  # 'auc' or 'mean'
    title_metric: str,
    out_path: Path,
) -> Path:
    """Render a per-game subplot grid with β on x-axis."""
    games = sorted(df["game"].unique())
    if not games:
        # Nothing to plot; skip and signal absence to the caller.
        return out_path

    n = len(games)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows), squeeze=False)

    for ax, game in zip(axes.ravel(), games):
        sub = df[df["game"] == game]
        method_x: list[tuple[str, Optional[float], float, float]] = []
        for method in sorted(sub["method"].unique()):
            seed_groups = sub[sub["method"] == method].groupby(["subcase", "seed"], sort=False)
            if y_compute == "auc":
                seed_values = np.array([_seed_auc(g) for _, g in seed_groups], dtype=float)
            elif y_compute == "mean":
                seed_values = np.array(
                    [float(g[y_col_name].mean()) for _, g in seed_groups], dtype=float
                )
            else:
                raise ValueError(f"unknown y_compute={y_compute!r}")
            seed_values = seed_values[~np.isnan(seed_values)]
            mean, se = _agg_mean_se(seed_values)
            method_x.append((method, _method_to_x(method), mean, se))

        # Numeric β arms first (sorted by β), then categorical adaptive entries.
        numeric = sorted([m for m in method_x if m[1] is not None], key=lambda r: r[1])
        categorical = [m for m in method_x if m[1] is None]
        if numeric:
            xs = [r[1] for r in numeric]
            ys = [r[2] for r in numeric]
            ses = [r[3] for r in numeric]
            ax.errorbar(xs, ys, yerr=ses, fmt="o-", capsize=3, label="fixed-β")
        if categorical:
            offset = 0
            base_x = max((r[1] for r in numeric), default=0.0) + 1.0
            for method, _, mean, se in categorical:
                ax.errorbar(
                    [base_x + offset], [mean], yerr=[se], fmt="s", capsize=3, label=method
                )
                offset += 0.4
        ax.set_title(f"{game}")
        ax.set_xlabel("β")
        ax.set_ylabel(title_metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    # Hide any unused axes.
    for ax in axes.ravel()[len(games):]:
        ax.set_visible(False)

    fig.suptitle(f"Phase VIII β-sweep — {title_metric}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_figure(
    processed_long_csv: Path,
    out_dir: Path,
    *,
    env_filter: Optional[str] = None,
) -> Dict[str, Path]:
    """Render the two β-sweep panels.

    Parameters
    ----------
    processed_long_csv:
        Long-form per-episode CSV emitted by
        ``experiments.adaptive_beta.tab_six_games.analysis.aggregate``.
    out_dir:
        Output root; PDFs are written under ``out_dir / 'figures'``.
    env_filter:
        Optional substring matched against ``game``; when provided, only
        rows with ``env_filter in game`` are plotted.

    Returns
    -------
    dict[str, Path]
        Mapping ``panel_name -> path to PDF``.
    """
    df = pd.read_csv(processed_long_csv)
    if env_filter is not None:
        df = df[df["game"].astype(str).str.contains(env_filter, regex=False)]

    fig_dir = out_dir / "figures"
    auc_path = _plot_per_game(
        df,
        y_col_name="return",
        y_compute="auc",
        title_metric="AUC of return vs episode",
        out_path=fig_dir / "beta_vs_auc.pdf",
    )
    contraction_path = _plot_per_game(
        df,
        y_col_name="bellman_residual",
        y_compute="mean",
        title_metric="Mean Bellman residual",
        out_path=fig_dir / "beta_vs_contraction.pdf",
    )
    return {"beta_vs_auc": auc_path, "beta_vs_contraction": contraction_path}


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
