#!/usr/bin/env python
"""Regenerate all Phase III figures and tables from aggregated NPZ files.

Figures produced (saved to ``--out-dir``):
    fig43_effective_discount.{pdf,png}    -- S11.1.1
    fig44_planning_residuals.{pdf,png}    -- S11.1.2
    fig45_learning_curves.{pdf,png}       -- S11.1.3
    fig46_regime_shift.{pdf,png}          -- S11.1.4
    fig47_return_distribution.{pdf,png}   -- S11.1.5
    fig48_clip_activity.{pdf,png}         -- S11.1.6
    fig49_ablation_appendix.{pdf,png}     -- S11.2

Tables produced (saved to ``--out-dir/tables/``):
    P3-A_main_performance.{csv,tex}
    P3-B_dp_planning.{csv,tex}
    P3-C_rl_returns.{csv,tex}
    P3-D_clip_activity.{csv,tex}
    P3-E_ablation_summary.{csv,tex}

Confidence intervals
--------------------
Error bands show mean +/- 1 std across seeds (cross-seed std from
aggregated NPZ). Bootstrap is not applied here because the aggregated
data already stores the cross-seed standard deviation; the raw per-seed
arrays are not available in the aggregated files.

No cherry-picking
-----------------
All seeds present in the aggregated data are included.

Regeneratability
----------------
Given the same ``results/`` tree, this script produces identical output.
PDF timestamps are pinned via SOURCE_DATE_EPOCH=0.

Usage
-----
::

    python make_phase3_figures.py \\
        --out-dir figures/phase3 \\
        --results-root results/weighted_lse_dp

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------
_STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Classical = grey/blue family; Safe = orange/red family.
_C_CLASSICAL = {
    "VI": "#4878CF",
    "MPI": "#6ACC65",
    "AsyncVI": "#7F7F7F",
    "PE": "#B0B0B0",
    "PI": "#17BECF",
    "QLearning": "#1F77B4",
    "ExpectedSARSA": "#AEC7E8",
}
_C_SAFE = {
    "SafeVI": "#D62728",
    "SafeMPI": "#FF7F0E",
    "SafeAsyncVI": "#E377C2",
    "SafePE": "#FFBB78",
    "SafePI": "#F7B6D2",
    "SafeQLearning": "#D62728",
    "SafeExpectedSARSA": "#FF9896",
}

_GAMMA_REF = 0.99

# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------
_DP_TASKS = [
    "chain_sparse_long", "chain_jackpot", "chain_catastrophe",
    "grid_sparse_goal",
]
_DP_TASKS_REGIME = [
    "chain_regime_shift_pre_shift", "chain_regime_shift_post_shift",
    "grid_regime_shift_pre_shift", "grid_regime_shift_post_shift",
]
_RL_TASKS = [
    "chain_sparse_long", "chain_jackpot", "chain_catastrophe",
    "chain_regime_shift",
    "grid_sparse_goal", "grid_hazard", "grid_regime_shift",
    "taxi_bonus_shock",
]

_SAFE_DP_ALGOS = ["SafeVI", "SafeMPI", "SafeAsyncVI"]
_CLASSICAL_DP_ALGOS = ["VI", "MPI", "AsyncVI"]
_SAFE_RL_ALGOS = ["SafeQLearning", "SafeExpectedSARSA"]
_CLASSICAL_RL_ALGOS = ["QLearning", "ExpectedSARSA"]

_TASK_DISPLAY = {
    "chain_sparse_long": "Chain (sparse)",
    "chain_jackpot": "Chain (jackpot)",
    "chain_catastrophe": "Chain (catastrophe)",
    "chain_regime_shift": "Chain (regime)",
    "chain_regime_shift_pre_shift": "Chain regime (pre)",
    "chain_regime_shift_post_shift": "Chain regime (post)",
    "grid_sparse_goal": "Grid (sparse)",
    "grid_hazard": "Grid (hazard)",
    "grid_regime_shift": "Grid (regime)",
    "grid_regime_shift_pre_shift": "Grid regime (pre)",
    "grid_regime_shift_post_shift": "Grid regime (post)",
    "taxi_bonus_shock": "Taxi (bonus shock)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        warnings.warn(f"Missing file: {path}")
        return None
    return dict(np.load(path, allow_pickle=False))


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        warnings.warn(f"Missing file: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _savefig(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Save as PDF and PNG, return list of output paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in (".pdf", ".png"):
        p = out_dir / f"{stem}{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def _write_table(rows: list[dict], out_dir: Path, stem: str,
                 columns: list[str]) -> list[Path]:
    """Write table as CSV and LaTeX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # CSV
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    paths.append(csv_path)

    # LaTeX
    tex_path = out_dir / f"{stem}.tex"
    with open(tex_path, "w") as f:
        col_fmt = "l" * len(columns)
        f.write("\\begin{tabular}{" + col_fmt + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(columns) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            vals = [str(row.get(c, "")) for c in columns]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    paths.append(tex_path)
    return paths


# ===========================================================================
# Figure functions
# ===========================================================================

def fig_effective_discount(out_dir: Path, p3_agg: Path) -> list[Path]:
    """Fig 43 (S11.1.1): Effective discount vs classical gamma."""
    _apply_style()

    tasks = _DP_TASKS
    algos = _SAFE_DP_ALGOS
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.5 * n_tasks, 2.8),
                             squeeze=False)

    for ti, task in enumerate(tasks):
        ax = axes[0, ti]
        any_plotted = False
        for algo in algos:
            sw = _safe_load_npz(p3_agg / task / algo / "safe_stagewise.npz")
            if sw is None:
                continue
            stages = sw["stage_indices"]
            eff_disc = sw["safe_effective_discount_mean"]
            eff_std = sw.get("safe_effective_discount_mean_cross_seed_std",
                             np.zeros_like(eff_disc))
            ax.plot(stages, eff_disc, label=algo,
                    color=_C_SAFE[algo], linewidth=1.2)
            ax.fill_between(stages, eff_disc - eff_std, eff_disc + eff_std,
                            alpha=0.2, color=_C_SAFE[algo])
            any_plotted = True

        ax.axhline(_GAMMA_REF, color="black", ls="--", lw=0.8,
                    label=f"$\\gamma = {_GAMMA_REF}$")
        ax.set_xlabel("Stage $t$")
        if ti == 0:
            ax.set_ylabel("Effective discount")
        ax.set_title(_TASK_DISPLAY.get(task, task))
        if any_plotted:
            ax.legend(loc="best", frameon=False)

    fig.suptitle("Effective discount per stage (Safe DP planners)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _savefig(fig, out_dir, "fig43_effective_discount")


def fig_planning_residuals(out_dir: Path, p3_agg: Path,
                           p2_agg: Path) -> list[Path]:
    """Fig 44 (S11.1.2): Planning residual curves."""
    _apply_style()

    tasks = _DP_TASKS
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.5 * n_tasks, 2.8),
                             squeeze=False)

    for ti, task in enumerate(tasks):
        ax = axes[0, ti]

        # Phase II classical DP: plot final bellman residual as horizontal line
        for algo in _CLASSICAL_DP_ALGOS:
            summary = _safe_load_json(
                p2_agg / task / algo / "summary.json")
            if summary is None:
                continue
            fbr = summary.get("scalar_metrics", {}).get(
                "final_bellman_residual", {})
            if isinstance(fbr, dict):
                val = fbr.get("mean", None)
            else:
                val = fbr
            if val is not None:
                ax.axhline(val, color=_C_CLASSICAL[algo], ls=":",
                           lw=0.8, label=f"{algo} (final)")

        # Phase III safe DP: per-stage bellman residual
        for algo in _SAFE_DP_ALGOS:
            sw = _safe_load_npz(p3_agg / task / algo / "safe_stagewise.npz")
            if sw is None:
                continue
            stages = sw["stage_indices"]
            br = sw["safe_bellman_residual"]
            br_std = sw.get("safe_bellman_residual_cross_seed_std",
                            np.zeros_like(br))
            ax.plot(stages, br, label=algo,
                    color=_C_SAFE[algo], linewidth=1.2)
            ax.fill_between(stages, br - br_std, br + br_std,
                            alpha=0.2, color=_C_SAFE[algo])

        ax.set_xlabel("Stage $t$")
        if ti == 0:
            ax.set_ylabel("Bellman residual")
        ax.set_title(_TASK_DISPLAY.get(task, task))
        ax.legend(loc="best", frameon=False, fontsize=6)
        ax.set_yscale("symlog", linthresh=1e-6)

    fig.suptitle("Planning residuals: classical vs safe DP",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _savefig(fig, out_dir, "fig44_planning_residuals")


def fig_learning_curves(out_dir: Path, p3_agg: Path,
                        p2_agg: Path) -> list[Path]:
    """Fig 45 (S11.1.3): Learning curves, classical vs safe RL."""
    _apply_style()

    tasks = _RL_TASKS
    n_cols = 4
    n_rows = (len(tasks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 2.8 * n_rows),
                             squeeze=False)

    for ti, task in enumerate(tasks):
        ax = axes[ti // n_cols, ti % n_cols]

        # Phase II classical RL
        for algo in _CLASSICAL_RL_ALGOS:
            summary = _safe_load_json(
                p2_agg / task / algo / "summary.json")
            if summary is None:
                continue
            curves = summary.get("curves", {})
            steps = curves.get("steps", [])
            mean_ret = curves.get("mean_return", [])
            std_ret = curves.get("std_return", [])
            if steps and mean_ret:
                steps = np.array(steps, dtype=float)
                mean_ret = np.array(mean_ret, dtype=float)
                std_ret = np.array(std_ret, dtype=float) if std_ret else np.zeros_like(mean_ret)
                ax.plot(steps, mean_ret, label=algo,
                        color=_C_CLASSICAL[algo], linewidth=1.2)
                ax.fill_between(steps, mean_ret - std_ret,
                                mean_ret + std_ret,
                                alpha=0.2, color=_C_CLASSICAL[algo])

        # Phase III safe RL
        for algo in _SAFE_RL_ALGOS:
            c = _safe_load_npz(p3_agg / task / algo / "curves.npz")
            if c is None:
                continue
            ckpts = c.get("checkpoints_mean")
            dr_mean = c.get("disc_return_mean_mean")
            dr_std = c.get("disc_return_mean_std")
            if ckpts is not None and dr_mean is not None:
                ax.plot(ckpts, dr_mean, label=algo,
                        color=_C_SAFE[algo], linewidth=1.2)
                if dr_std is not None:
                    ax.fill_between(ckpts, dr_mean - dr_std,
                                    dr_mean + dr_std,
                                    alpha=0.2, color=_C_SAFE[algo])

        ax.set_xlabel("Training steps")
        if ti % n_cols == 0:
            ax.set_ylabel("Discounted return")
        ax.set_title(_TASK_DISPLAY.get(task, task))
        ax.legend(loc="best", frameon=False, fontsize=6)

    # Hide unused axes
    for ti in range(len(tasks), n_rows * n_cols):
        axes[ti // n_cols, ti % n_cols].set_visible(False)

    fig.suptitle("Learning curves: classical vs safe RL",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _savefig(fig, out_dir, "fig45_learning_curves")


def fig_regime_shift(out_dir: Path, p3_agg: Path,
                     p2_agg: Path) -> list[Path]:
    """Fig 46 (S11.1.4): Regime-shift adaptation recovery curves."""
    _apply_style()

    regime_tasks = ["chain_regime_shift", "grid_regime_shift"]
    fig, axes = plt.subplots(1, len(regime_tasks),
                             figsize=(3.5 * len(regime_tasks), 2.8),
                             squeeze=False)

    for ti, task in enumerate(regime_tasks):
        ax = axes[0, ti]

        # Classical RL
        for algo in _CLASSICAL_RL_ALGOS:
            summary = _safe_load_json(
                p2_agg / task / algo / "summary.json")
            if summary is None:
                continue
            curves = summary.get("curves", {})
            steps = curves.get("steps", [])
            mean_ret = curves.get("mean_return", [])
            std_ret = curves.get("std_return", [])
            if steps and mean_ret:
                steps = np.array(steps, dtype=float)
                mean_ret = np.array(mean_ret, dtype=float)
                std_ret = np.array(std_ret, dtype=float) if std_ret else np.zeros_like(mean_ret)
                ax.plot(steps, mean_ret, label=algo,
                        color=_C_CLASSICAL[algo], linewidth=1.2)
                ax.fill_between(steps, mean_ret - std_ret,
                                mean_ret + std_ret,
                                alpha=0.2, color=_C_CLASSICAL[algo])

            # Mark regime change point if available
            change_at = summary.get("change_at_episode")
            if change_at is not None and isinstance(change_at, (int, float)):
                ax.axvline(change_at, color="gray", ls="--", lw=0.7,
                           label="Regime shift")

        # Safe RL
        for algo in _SAFE_RL_ALGOS:
            c = _safe_load_npz(p3_agg / task / algo / "curves.npz")
            if c is None:
                continue
            ckpts = c.get("checkpoints_mean")
            dr_mean = c.get("disc_return_mean_mean")
            dr_std = c.get("disc_return_mean_std")
            if ckpts is not None and dr_mean is not None:
                ax.plot(ckpts, dr_mean, label=algo,
                        color=_C_SAFE[algo], linewidth=1.2)
                if dr_std is not None:
                    ax.fill_between(ckpts, dr_mean - dr_std,
                                    dr_mean + dr_std,
                                    alpha=0.2, color=_C_SAFE[algo])

        ax.set_xlabel("Training steps")
        if ti == 0:
            ax.set_ylabel("Discounted return")
        ax.set_title(_TASK_DISPLAY.get(task, task))
        ax.legend(loc="best", frameon=False, fontsize=6)

    fig.suptitle("Regime-shift adaptation: classical vs safe RL",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _savefig(fig, out_dir, "fig46_regime_shift")


def fig_return_distribution(out_dir: Path, p3_agg: Path,
                            p2_agg: Path) -> list[Path]:
    """Fig 47 (S11.1.5): Return distributions for catastrophe and jackpot."""
    _apply_style()

    tasks = ["chain_catastrophe", "chain_jackpot"]
    all_algos = _CLASSICAL_RL_ALGOS + _SAFE_RL_ALGOS

    fig, axes = plt.subplots(1, len(tasks),
                             figsize=(3.5 * len(tasks), 2.8),
                             squeeze=False)

    for ti, task in enumerate(tasks):
        ax = axes[0, ti]
        positions = []
        data_list = []
        labels = []
        colors = []
        pos = 0

        for algo in all_algos:
            # Collect per-seed final returns
            if algo in _CLASSICAL_RL_ALGOS:
                summary = _safe_load_json(
                    p2_agg / task / algo / "summary.json")
                if summary is None:
                    continue
                # Use episode_returns from each checkpoint's last entries
                ep_rets = summary.get("episode_returns", {})
                if isinstance(ep_rets, dict) and ep_rets:
                    # ep_rets is {seed_str: n_episodes}; use final
                    # disc_return_mean from curves
                    curves = summary.get("curves", {})
                    mean_ret = curves.get("mean_return", [])
                    if mean_ret:
                        # Use all checkpoint returns as distribution
                        data_list.append(np.array(mean_ret))
                    else:
                        continue
                else:
                    continue
                col = _C_CLASSICAL[algo]
            else:
                c = _safe_load_npz(p3_agg / task / algo / "curves.npz")
                if c is None:
                    continue
                dr_mean = c.get("disc_return_mean_mean")
                if dr_mean is None:
                    continue
                data_list.append(dr_mean)
                col = _C_SAFE[algo]

            positions.append(pos)
            labels.append(algo)
            colors.append(col)
            pos += 1

        if data_list:
            parts = ax.violinplot(data_list, positions=positions,
                                  showmeans=True, showmedians=True)
            for i, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)

        ax.set_ylabel("Discounted return")
        ax.set_title(_TASK_DISPLAY.get(task, task))

    fig.suptitle("Return distributions: classical vs safe RL",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _savefig(fig, out_dir, "fig47_return_distribution")


def fig_clip_activity(out_dir: Path, p3_agg: Path) -> list[Path]:
    """Fig 48 (S11.1.6): Clip activity and deployed beta per stage."""
    _apply_style()

    tasks = ["chain_sparse_long", "chain_jackpot"]
    algos = _SAFE_DP_ALGOS

    fig, axes = plt.subplots(2, len(tasks),
                             figsize=(3.5 * len(tasks), 5.0),
                             squeeze=False)

    for ti, task in enumerate(tasks):
        ax_beta = axes[0, ti]
        ax_clip = axes[1, ti]

        for algo in algos:
            sw = _safe_load_npz(p3_agg / task / algo / "safe_stagewise.npz")
            if sw is None:
                continue

            stages = sw["stage_indices"]
            beta_mean = sw["safe_beta_used_mean"]
            beta_std = sw.get("safe_beta_used_mean_cross_seed_std",
                              np.zeros_like(beta_mean))
            clip_frac = sw["safe_clip_fraction"]
            clip_std = sw.get("safe_clip_fraction_cross_seed_std",
                              np.zeros_like(clip_frac))

            ax_beta.plot(stages, beta_mean, label=algo,
                         color=_C_SAFE[algo], linewidth=1.2)
            ax_beta.fill_between(stages, beta_mean - beta_std,
                                 beta_mean + beta_std,
                                 alpha=0.2, color=_C_SAFE[algo])

            ax_clip.plot(stages, clip_frac, label=algo,
                         color=_C_SAFE[algo], linewidth=1.2)
            ax_clip.fill_between(stages, clip_frac - clip_std,
                                 clip_frac + clip_std,
                                 alpha=0.2, color=_C_SAFE[algo])

        ax_beta.set_title(_TASK_DISPLAY.get(task, task))
        ax_beta.set_ylabel("Deployed $\\beta$")
        ax_beta.legend(loc="best", frameon=False, fontsize=6)

        ax_clip.set_xlabel("Stage $t$")
        ax_clip.set_ylabel("Clip fraction")

    fig.suptitle("Deployed beta and clip activity per stage",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _savefig(fig, out_dir, "fig48_clip_activity")


def fig_ablation_appendix(out_dir: Path, p3_root: Path,
                          p3_agg: Path) -> list[Path]:
    """Fig 49 (S11.2): Ablation appendix -- alpha grid and constant-beta."""
    _apply_style()

    # --- Panel A: Alpha ablation (effective discount curves) ---
    alpha_values = ["0.00", "0.02", "0.05", "0.10", "0.20"]
    alpha_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_values)))

    # --- Panel B: Beta ablation (constant vs stagewise) ---
    beta_ablations = ["beta_constant_small", "beta_constant_large"]
    beta_colors = {"beta_constant_small": "#2CA02C",
                   "beta_constant_large": "#9467BD"}

    # Use chain_sparse_long as representative task
    rep_task = "chain_sparse_long"
    rep_algo_dp = "SafeVI"
    rep_algo_rl = "SafeQLearning"

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))

    # Panel (0,0): Alpha ablation -- effective discount
    ax = axes[0, 0]
    for ai, alpha in enumerate(alpha_values):
        abl_dir = p3_root / f"ablation_alpha_{alpha}" / rep_task / rep_algo_dp
        # Per-seed aggregation inline
        seed_dirs = sorted(abl_dir.glob("seed_*")) if abl_dir.exists() else []
        all_eff_disc = []
        for sd in seed_dirs:
            cs = _safe_load_npz(sd / "calibration_stats.npz")
            if cs is not None:
                # Field name is safe_effective_discount_mean in calibration_stats
                for key in ("safe_effective_discount_mean",
                            "safe_effective_discount"):
                    if key in cs:
                        all_eff_disc.append(cs[key])
                        break
        if not all_eff_disc:
            ax.text(0.5, 0.5 - ai * 0.1,
                    f"$\\alpha={alpha}$: no stage data",
                    transform=ax.transAxes, fontsize=7)
            continue
        # Stack and average
        min_len = min(len(x) for x in all_eff_disc)
        arr = np.array([x[:min_len] for x in all_eff_disc])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        stages = np.arange(min_len)
        ax.plot(stages, mean, label=f"$\\alpha={alpha}$",
                color=alpha_colors[ai], linewidth=1.0)
        ax.fill_between(stages, mean - std, mean + std,
                        alpha=0.15, color=alpha_colors[ai])

    # Also plot main schedule for reference
    sw_main = _safe_load_npz(
        p3_agg / rep_task / rep_algo_dp / "safe_stagewise.npz")
    if sw_main is not None:
        stages_main = sw_main["stage_indices"]
        eff_main = sw_main["safe_effective_discount_mean"]
        ax.plot(stages_main, eff_main, label="Main schedule",
                color="black", linewidth=1.5, ls="--")

    ax.axhline(_GAMMA_REF, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("Stage $t$")
    ax.set_ylabel("Effective discount")
    ax.set_title(f"Alpha ablation ({_TASK_DISPLAY[rep_task]})")
    ax.legend(loc="best", frameon=False, fontsize=6)

    # Panel (0,1): Alpha ablation -- RL AUC comparison (bar chart)
    ax = axes[0, 1]
    auc_data = {}
    for alpha in alpha_values:
        abl_dir = p3_root / f"ablation_alpha_{alpha}" / rep_task / rep_algo_rl
        seed_dirs = sorted(abl_dir.glob("seed_*")) if abl_dir.exists() else []
        aucs = []
        for sd in seed_dirs:
            mj = _safe_load_json(sd / "metrics.json")
            if mj is not None and "auc_disc_return" in mj:
                aucs.append(mj["auc_disc_return"])
        if aucs:
            auc_data[f"a={alpha}"] = (np.mean(aucs), np.std(aucs))

    # Main schedule AUC
    main_summary = _safe_load_json(
        p3_agg / rep_task / rep_algo_rl / "summary.json")
    if main_summary:
        m = main_summary.get("metrics", {}).get("auc_disc_return", {})
        if isinstance(m, dict):
            auc_data["main"] = (m.get("mean", 0), m.get("std", 0))

    if auc_data:
        x = np.arange(len(auc_data))
        means = [v[0] for v in auc_data.values()]
        stds = [v[1] for v in auc_data.values()]
        ax.bar(x, means, yerr=stds, capsize=3, color="steelblue",
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(list(auc_data.keys()), rotation=30, ha="right",
                           fontsize=7)
    ax.set_ylabel("AUC disc. return")
    ax.set_title(f"Alpha ablation AUC ({rep_algo_rl})")

    # Panel (1,0): Beta ablation -- effective discount
    ax = axes[1, 0]
    for abl_name in beta_ablations:
        abl_dir = p3_root / f"ablation_{abl_name}" / rep_task / rep_algo_dp
        seed_dirs = sorted(abl_dir.glob("seed_*")) if abl_dir.exists() else []
        all_eff_disc = []
        for sd in seed_dirs:
            cs = _safe_load_npz(sd / "calibration_stats.npz")
            if cs is not None:
                for key in ("safe_effective_discount_mean",
                            "safe_effective_discount"):
                    if key in cs:
                        all_eff_disc.append(cs[key])
                        break
        if not all_eff_disc:
            ax.text(0.5, 0.3, f"{abl_name}: no stage data",
                    transform=ax.transAxes, fontsize=7)
            continue
        min_len = min(len(x) for x in all_eff_disc)
        arr = np.array([x[:min_len] for x in all_eff_disc])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        stages = np.arange(min_len)
        label = abl_name.replace("beta_constant_", "const $\\beta$ ")
        ax.plot(stages, mean, label=label,
                color=beta_colors[abl_name], linewidth=1.0)
        ax.fill_between(stages, mean - std, mean + std,
                        alpha=0.15, color=beta_colors[abl_name])

    if sw_main is not None:
        ax.plot(sw_main["stage_indices"],
                sw_main["safe_effective_discount_mean"],
                label="Stagewise (main)", color="black", ls="--", lw=1.5)
    ax.axhline(_GAMMA_REF, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("Stage $t$")
    ax.set_ylabel("Effective discount")
    ax.set_title(f"Beta ablation ({_TASK_DISPLAY[rep_task]})")
    ax.legend(loc="best", frameon=False, fontsize=6)

    # Panel (1,1): Beta ablation -- RL AUC bar chart
    ax = axes[1, 1]
    auc_data_b = {}
    for abl_name in ["beta_zero"] + beta_ablations:
        abl_dir = p3_root / f"ablation_{abl_name}" / rep_task / rep_algo_rl
        seed_dirs = sorted(abl_dir.glob("seed_*")) if abl_dir.exists() else []
        aucs = []
        for sd in seed_dirs:
            mj = _safe_load_json(sd / "metrics.json")
            if mj is not None and "auc_disc_return" in mj:
                aucs.append(mj["auc_disc_return"])
        if aucs:
            label = abl_name.replace("beta_", "").replace("_", " ")
            auc_data_b[label] = (np.mean(aucs), np.std(aucs))

    if main_summary:
        m = main_summary.get("metrics", {}).get("auc_disc_return", {})
        if isinstance(m, dict):
            auc_data_b["main"] = (m.get("mean", 0), m.get("std", 0))

    if auc_data_b:
        x = np.arange(len(auc_data_b))
        means = [v[0] for v in auc_data_b.values()]
        stds = [v[1] for v in auc_data_b.values()]
        ax.bar(x, means, yerr=stds, capsize=3, color="coral",
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(list(auc_data_b.keys()), rotation=30, ha="right",
                           fontsize=7)
    ax.set_ylabel("AUC disc. return")
    ax.set_title(f"Beta ablation AUC ({rep_algo_rl})")

    fig.suptitle("Appendix: ablation analysis", fontsize=11, y=1.02)
    fig.tight_layout()
    return _savefig(fig, out_dir, "fig49_ablation_appendix")


# ===========================================================================
# Table functions
# ===========================================================================

def table_p3a_main_performance(out_dir: Path, p3_agg: Path,
                               p2_agg: Path) -> list[Path]:
    """Table P3-A: Main performance comparison, classical vs safe."""
    columns = ["task", "algorithm", "type", "auc_disc_return", "final_disc_return"]
    rows: list[dict] = []

    for task in _RL_TASKS:
        for algo in _CLASSICAL_RL_ALGOS:
            summary = _safe_load_json(p2_agg / task / algo / "summary.json")
            if summary is None:
                continue
            sm = summary.get("scalar_metrics", {})
            auc = sm.get("auc_disc_return", {})
            fdr = sm.get("final_disc_return_mean", {})
            rows.append({
                "task": task,
                "algorithm": algo,
                "type": "classical",
                "auc_disc_return": _fmt_pm(
                    auc.get("mean", float("nan")),
                    auc.get("std", float("nan"))) if isinstance(auc, dict) else str(auc),
                "final_disc_return": _fmt_pm(
                    fdr.get("mean", float("nan")),
                    fdr.get("std", float("nan"))) if isinstance(fdr, dict) else str(fdr),
            })

        for algo in _SAFE_RL_ALGOS:
            summary = _safe_load_json(p3_agg / task / algo / "summary.json")
            if summary is None:
                continue
            m = summary.get("metrics", {})
            auc = m.get("auc_disc_return", {})
            fdr = m.get("final_disc_return_mean", {})
            rows.append({
                "task": task,
                "algorithm": algo,
                "type": "safe",
                "auc_disc_return": _fmt_pm(
                    auc.get("mean", float("nan")),
                    auc.get("std", float("nan"))) if isinstance(auc, dict) else str(auc),
                "final_disc_return": _fmt_pm(
                    fdr.get("mean", float("nan")),
                    fdr.get("std", float("nan"))) if isinstance(fdr, dict) else str(fdr),
            })

    return _write_table(rows, out_dir / "tables", "P3-A_main_performance",
                        columns)


def table_p3b_dp_planning(out_dir: Path, p3_agg: Path,
                          p2_agg: Path) -> list[Path]:
    """Table P3-B: DP planning iterations and wall clock."""
    columns = ["task", "algorithm", "type", "n_sweeps", "final_bellman_residual",
               "wall_clock_s"]
    rows: list[dict] = []

    for task in _DP_TASKS:
        for algo in _CLASSICAL_DP_ALGOS:
            summary = _safe_load_json(p2_agg / task / algo / "summary.json")
            if summary is None:
                continue
            sm = summary.get("scalar_metrics", {})
            rows.append({
                "task": task,
                "algorithm": algo,
                "type": "classical",
                "n_sweeps": _fmt_pm(
                    sm.get("n_sweeps", {}).get("mean", float("nan")),
                    sm.get("n_sweeps", {}).get("std", float("nan"))),
                "final_bellman_residual": _fmt_pm(
                    sm.get("final_bellman_residual", {}).get("mean", float("nan")),
                    sm.get("final_bellman_residual", {}).get("std", float("nan"))),
                "wall_clock_s": _fmt_pm(
                    sm.get("wall_clock_s", {}).get("mean", float("nan")),
                    sm.get("wall_clock_s", {}).get("std", float("nan"))),
            })

        for algo in _SAFE_DP_ALGOS:
            summary = _safe_load_json(p3_agg / task / algo / "summary.json")
            if summary is None:
                continue
            m = summary.get("metrics", {})
            rows.append({
                "task": task,
                "algorithm": algo,
                "type": "safe",
                "n_sweeps": _fmt_pm(
                    m.get("n_sweeps", {}).get("mean", float("nan")),
                    m.get("n_sweeps", {}).get("std", float("nan"))) if isinstance(m.get("n_sweeps"), dict) else str(m.get("n_sweeps", "")),
                "final_bellman_residual": _fmt_pm(
                    m.get("final_bellman_residual", {}).get("mean", float("nan")),
                    m.get("final_bellman_residual", {}).get("std", float("nan"))) if isinstance(m.get("final_bellman_residual"), dict) else str(m.get("final_bellman_residual", "")),
                "wall_clock_s": _fmt_pm(
                    m.get("wall_clock_s", {}).get("mean", float("nan")),
                    m.get("wall_clock_s", {}).get("std", float("nan"))) if isinstance(m.get("wall_clock_s"), dict) else str(m.get("wall_clock_s", "")),
            })

    return _write_table(rows, out_dir / "tables", "P3-B_dp_planning", columns)


def table_p3c_rl_returns(out_dir: Path, p3_agg: Path,
                         p2_agg: Path) -> list[Path]:
    """Table P3-C: RL returns per task family, safe vs classical."""
    columns = ["task", "algorithm", "type", "auc_disc_return",
               "final_disc_return"]
    rows: list[dict] = []

    for task in _RL_TASKS:
        for algo in _CLASSICAL_RL_ALGOS:
            summary = _safe_load_json(p2_agg / task / algo / "summary.json")
            if summary is None:
                continue
            sm = summary.get("scalar_metrics", {})
            auc = sm.get("auc_disc_return", {})
            fdr = sm.get("final_disc_return_mean", {})
            rows.append({
                "task": task, "algorithm": algo, "type": "classical",
                "auc_disc_return": _fmt_pm(auc.get("mean", float("nan")), auc.get("std", float("nan"))) if isinstance(auc, dict) else str(auc),
                "final_disc_return": _fmt_pm(fdr.get("mean", float("nan")), fdr.get("std", float("nan"))) if isinstance(fdr, dict) else str(fdr),
            })

        for algo in _SAFE_RL_ALGOS:
            summary = _safe_load_json(p3_agg / task / algo / "summary.json")
            if summary is None:
                continue
            m = summary.get("metrics", {})
            auc = m.get("auc_disc_return", {})
            fdr = m.get("final_disc_return_mean", {})
            rows.append({
                "task": task, "algorithm": algo, "type": "safe",
                "auc_disc_return": _fmt_pm(auc.get("mean", float("nan")), auc.get("std", float("nan"))) if isinstance(auc, dict) else str(auc),
                "final_disc_return": _fmt_pm(fdr.get("mean", float("nan")), fdr.get("std", float("nan"))) if isinstance(fdr, dict) else str(fdr),
            })

    return _write_table(rows, out_dir / "tables", "P3-C_rl_returns", columns)


def table_p3d_clip_activity(out_dir: Path, p3_agg: Path) -> list[Path]:
    """Table P3-D: Clip activity summary."""
    columns = ["task", "algorithm", "mean_clip_fraction",
               "mean_effective_discount", "mean_beta_used"]
    rows: list[dict] = []

    all_tasks = _DP_TASKS + [t for t in _RL_TASKS if t not in _DP_TASKS]
    for task in all_tasks:
        for algo in _SAFE_DP_ALGOS + _SAFE_RL_ALGOS:
            sw = _safe_load_npz(p3_agg / task / algo / "safe_stagewise.npz")
            if sw is None:
                continue
            rows.append({
                "task": task,
                "algorithm": algo,
                "mean_clip_fraction": f"{sw['safe_clip_fraction'].mean():.4f}",
                "mean_effective_discount": f"{sw['safe_effective_discount_mean'].mean():.6f}",
                "mean_beta_used": f"{sw['safe_beta_used_mean'].mean():.4f}",
            })

    return _write_table(rows, out_dir / "tables", "P3-D_clip_activity",
                        columns)


def table_p3e_ablation_summary(out_dir: Path, p3_root: Path,
                               p3_agg: Path) -> list[Path]:
    """Table P3-E: Ablation summary -- AUC for key ablation variants."""
    columns = ["task", "algorithm", "ablation", "auc_disc_return"]
    rows: list[dict] = []

    ablation_names = ["beta_zero", "beta_constant_small",
                      "alpha_0.05"]
    rep_tasks = ["chain_sparse_long", "chain_jackpot", "chain_catastrophe"]

    for task in rep_tasks:
        # Main schedule
        for algo in _SAFE_RL_ALGOS:
            summary = _safe_load_json(
                p3_agg / task / algo / "summary.json")
            if summary is not None:
                m = summary.get("metrics", {}).get("auc_disc_return", {})
                if isinstance(m, dict):
                    rows.append({
                        "task": task, "algorithm": algo, "ablation": "main",
                        "auc_disc_return": _fmt_pm(
                            m.get("mean", float("nan")),
                            m.get("std", float("nan"))),
                    })

        # Ablations
        for abl_name in ablation_names:
            for algo in _SAFE_RL_ALGOS:
                abl_dir = (p3_root / f"ablation_{abl_name}"
                           / task / algo)
                seed_dirs = sorted(abl_dir.glob("seed_*")) if abl_dir.exists() else []
                aucs = []
                for sd in seed_dirs:
                    mj = _safe_load_json(sd / "metrics.json")
                    if mj is not None and "auc_disc_return" in mj:
                        aucs.append(mj["auc_disc_return"])
                if aucs:
                    rows.append({
                        "task": task, "algorithm": algo,
                        "ablation": abl_name,
                        "auc_disc_return": _fmt_pm(
                            np.mean(aucs), np.std(aucs)),
                    })

    return _write_table(rows, out_dir / "tables", "P3-E_ablation_summary",
                        columns)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all Phase III figures and tables.")
    parser.add_argument(
        "--out-dir", type=str,
        default="figures/phase3",
        help="Output directory for figures and tables.")
    parser.add_argument(
        "--results-root", type=str,
        default="results/weighted_lse_dp",
        help="Root of results tree.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_root = Path(args.results_root)
    p3_agg = results_root / "phase3" / "aggregated"
    p2_agg = results_root / "phase2" / "aggregated"
    p3_root = results_root / "phase3"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    all_outputs: list[Path] = []

    print("=" * 60)
    print("Phase III Figure & Table Generation")
    print("=" * 60)

    # --- Figures ---
    fig_funcs = [
        ("Fig 43: Effective discount",
         lambda: fig_effective_discount(out_dir, p3_agg)),
        ("Fig 44: Planning residuals",
         lambda: fig_planning_residuals(out_dir, p3_agg, p2_agg)),
        ("Fig 45: Learning curves",
         lambda: fig_learning_curves(out_dir, p3_agg, p2_agg)),
        ("Fig 46: Regime-shift adaptation",
         lambda: fig_regime_shift(out_dir, p3_agg, p2_agg)),
        ("Fig 47: Return distributions",
         lambda: fig_return_distribution(out_dir, p3_agg, p2_agg)),
        ("Fig 48: Clip activity",
         lambda: fig_clip_activity(out_dir, p3_agg)),
        ("Fig 49: Ablation appendix",
         lambda: fig_ablation_appendix(out_dir, p3_root, p3_agg)),
    ]

    for name, func in fig_funcs:
        print(f"\n  {name} ... ", end="", flush=True)
        try:
            paths = func()
            all_outputs.extend(paths)
            print(f"OK ({len(paths)} files)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # --- Tables ---
    table_funcs = [
        ("Table P3-A: Main performance",
         lambda: table_p3a_main_performance(out_dir, p3_agg, p2_agg)),
        ("Table P3-B: DP planning",
         lambda: table_p3b_dp_planning(out_dir, p3_agg, p2_agg)),
        ("Table P3-C: RL returns",
         lambda: table_p3c_rl_returns(out_dir, p3_agg, p2_agg)),
        ("Table P3-D: Clip activity",
         lambda: table_p3d_clip_activity(out_dir, p3_agg)),
        ("Table P3-E: Ablation summary",
         lambda: table_p3e_ablation_summary(out_dir, p3_root, p3_agg)),
    ]

    for name, func in table_funcs:
        print(f"\n  {name} ... ", end="", flush=True)
        try:
            paths = func()
            all_outputs.extend(paths)
            print(f"OK ({len(paths)} files)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # --- Manifest ---
    print("\n" + "=" * 60)
    print("Manifest")
    print("=" * 60)
    manifest = {}
    for p in sorted(all_outputs):
        sha = _sha256(p)
        manifest[str(p)] = sha
        print(f"  {p}  sha256:{sha[:16]}...")

    manifest_path = out_dir / "figures_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nManifest written to {manifest_path}")
    print(f"Total outputs: {len(all_outputs)}")


if __name__ == "__main__":
    main()
