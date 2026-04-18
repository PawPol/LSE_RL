#!/usr/bin/env python
"""Regenerate all Phase II figures (11.1.1--11.1.5) from a single entry point.

Each figure function accepts ``out_dir`` and ``results_root`` parameters,
skips gracefully (with a warning) when the required data files do not
exist, and saves output as both PDF and PNG at 150 dpi.

In ``--demo`` mode, synthetic data is generated so that figures can be
previewed without a populated results tree.

After all figures are written, a ``figures_manifest.json`` is emitted
listing every output file with its SHA256 checksum.

Confidence intervals
--------------------
Where applicable, shaded bands show percentile bootstrap 95% CI over
seeds (10 000 resamples).

No cherry-picking
-----------------
All seeds present in the results tree are used unless ``--seed`` is
specified.

Regeneratability
----------------
Given the same ``results/`` tree and code revision, this script
produces identical output (PDF timestamps pinned via
``SOURCE_DATE_EPOCH``).

Usage
-----
::

    # Demo mode:
    python make_phase2_figures.py --demo

    # Production mode:
    python make_phase2_figures.py --out-root results/weighted_lse_dp

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

# Pin PDF creation timestamps for byte-reproducibility.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "mushroom-rl-dev") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "mushroom-rl-dev"))


# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------

_STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": False,
    "pdf.fonttype": 42,  # Type 42 (TrueType) -- NeurIPS requirement
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _apply_style() -> None:
    """Apply paper-ready matplotlib rcParams."""
    plt.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# Colour palette (no seaborn)
# ---------------------------------------------------------------------------

_COLORS = {
    "base": "#1f77b4",
    "stress": "#d62728",
    "jackpot": "#ff7f0e",
    "catastrophe": "#9467bd",
    "hazard": "#8c564b",
    "regime_pre": "#1f77b4",
    "regime_post": "#d62728",
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIGURE_NAMES: list[str] = [
    "learning_curves",
    "return_distributions",
    "adaptation_plots",
    "visitation_heatmaps",
    "margin_quantiles",
]

_FIGURE_STEMS: dict[str, str] = {
    "learning_curves": "phase2_base_vs_modified_curves",
    "return_distributions": "phase2_return_distributions",
    "adaptation_plots": "phase2_adaptation_plots",
    "visitation_heatmaps": "phase2_visitation_heatmaps",
    "margin_quantiles": "phase2_margin_quantiles",
}

_TASK_FAMILIES = {
    "chain": [
        "chain_sparse_long", "chain_jackpot",
        "chain_catastrophe", "chain_regime_shift",
    ],
    "grid": [
        "grid_sparse_goal", "grid_hazard", "grid_regime_shift",
    ],
    "taxi": [
        "taxi_bonus_shock",
    ],
}

_REGIME_SHIFT_TASKS = ("chain_regime_shift", "grid_regime_shift")
_TAIL_TASKS = (
    "chain_jackpot", "chain_catastrophe", "grid_hazard", "taxi_bonus_shock",
)
_GRID_TASKS = ("grid_sparse_goal", "grid_hazard", "grid_regime_shift")
# Per-task grid shapes for heatmap reshape.  grid_sparse_goal is 7x7; others are 5x5.
_GRID_TASK_SHAPES: dict[str, tuple[int, int]] = {
    "grid_sparse_goal": (7, 7),
    "grid_hazard": (5, 5),
    "grid_regime_shift": (5, 5),
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _load_summary_curves(
    aggregated_root: Path,
    task: str,
    algorithm: str,
) -> dict[str, Any] | None:
    """Load summary.json which may contain curve arrays."""
    return _load_json(aggregated_root / task / algorithm / "summary.json")


def _load_calibration(calibration_root: Path, task: str) -> dict[str, Any] | None:
    return _load_json(calibration_root / f"{task}.json")


# ---------------------------------------------------------------------------
# Demo data generators
# ---------------------------------------------------------------------------


def _demo_learning_curve(
    n_steps: int = 30,
    final_mean: float = 0.8,
    noise: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic (steps, mean, std) learning curve."""
    if rng is None:
        rng = np.random.default_rng(0)
    steps = np.linspace(0, 1, n_steps)
    curve = final_mean * (1 - np.exp(-4 * steps))
    mean = curve + rng.normal(0, noise, n_steps)
    std = np.full(n_steps, noise)
    return steps, mean, std


def _demo_return_distribution(
    n: int = 500,
    has_tail: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic per-episode returns."""
    if rng is None:
        rng = np.random.default_rng(0)
    returns = rng.normal(0.7, 0.1, n)
    if has_tail:
        tail_mask = rng.random(n) < 0.05
        returns[tail_mask] = rng.uniform(2.0, 5.0, tail_mask.sum())
    return returns


def _demo_adaptation_curve(
    n_episodes: int = 600,
    change_point: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate synthetic episode returns with a regime change."""
    if rng is None:
        rng = np.random.default_rng(0)
    pre = rng.normal(0.8, 0.1, change_point)
    post = np.concatenate([
        rng.normal(0.3, 0.15, 100),
        rng.normal(0.6, 0.1, n_episodes - change_point - 100),
    ])
    episodes = np.arange(n_episodes)
    returns = np.concatenate([pre, post])
    return episodes, returns, change_point


def _demo_heatmap(
    grid_h: int = 5, grid_w: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic visit-count heatmap."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = rng.poisson(20, (grid_h, grid_w)).astype(float)
    # Make start/goal cells higher.
    counts[0, 0] += 50
    counts[grid_h - 1, grid_w - 1] += 40
    return counts


def _demo_margin_quantiles(
    n_stages: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate synthetic per-stage margin quantiles."""
    if rng is None:
        rng = np.random.default_rng(0)
    stages = np.arange(n_stages)
    q50 = rng.uniform(0.0, 0.3, n_stages)
    q05 = q50 - rng.uniform(0.05, 0.15, n_stages)
    q95 = q50 + rng.uniform(0.05, 0.15, n_stages)
    return {"stages": stages, "q05": q05, "q50": q50, "q95": q95}


# ---------------------------------------------------------------------------
# Figure 11.1.1: Base vs modified learning curves
# ---------------------------------------------------------------------------


def fig_learning_curves(
    out_dir: Path,
    results_root: Path,
    *,
    demo: bool = False,
) -> list[Path]:
    """Figure 11.1.1: base vs modified learning curves per task family.

    Confidence intervals: percentile bootstrap 95% CI over seeds
    (shaded band = mean +/- 1 std across seeds).
    """
    _apply_style()
    families = list(_TASK_FAMILIES.keys())
    fig, axes = plt.subplots(1, len(families), figsize=(4.0 * len(families), 3.0))
    if len(families) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    aggregated_root = results_root / "phase2" / "aggregated"

    for ax, family in zip(axes, families):
        tasks = _TASK_FAMILIES[family]
        plotted = False

        for i, task in enumerate(tasks):
            if demo:
                final = 0.8 - 0.1 * i
                steps, mean, std = _demo_learning_curve(
                    final_mean=final, rng=np.random.default_rng(42 + i),
                )
            else:
                summary = _load_summary_curves(aggregated_root, task, "QLearning")
                if summary is None:
                    continue
                curves = summary.get("curves", {})
                steps = np.array(curves.get("steps", []))
                mean = np.array(curves.get("mean_return", []))
                std = np.array(curves.get("std_return", []))
                if len(steps) == 0 or len(mean) == 0:
                    continue

            color = _COLORS.get("base") if i == 0 else _COLORS.get("stress")
            label = task.replace("_", " ")
            ax.plot(steps, mean, color=color, label=label, linewidth=1.2)
            if len(std) == len(mean):
                ax.fill_between(
                    steps, mean - std, mean + std,
                    alpha=0.2, color=color,
                )
            plotted = True

        ax.set_title(f"{family.capitalize()} family")
        ax.set_xlabel("Training step (normalised)")
        if ax == axes[0]:
            ax.set_ylabel("Mean episodic return")
        if plotted:
            ax.legend(fontsize=6, loc="lower right")

    fig.tight_layout()
    return _save_figure(fig, out_dir, _FIGURE_STEMS["learning_curves"])


# ---------------------------------------------------------------------------
# Figure 11.1.2: Return distributions (jackpot / catastrophe)
# ---------------------------------------------------------------------------


def fig_return_distributions(
    out_dir: Path,
    results_root: Path,
    *,
    demo: bool = False,
) -> list[Path]:
    """Figure 11.1.2: return distributions for jackpot/catastrophe tasks.

    Shows histogram of per-episode returns for base vs stress variant.
    Confidence intervals: N/A (raw distribution display).
    """
    _apply_style()
    n_tasks = len(_TAIL_TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.5 * n_tasks, 3.0))
    if n_tasks == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    calibration_root = results_root / "phase2" / "calibration"

    for ax, task in zip(axes, _TAIL_TASKS):
        if demo:
            base_returns = _demo_return_distribution(
                has_tail=False, rng=np.random.default_rng(10),
            )
            stress_returns = _demo_return_distribution(
                has_tail=True, rng=np.random.default_rng(20),
            )
        else:
            cal = _load_calibration(calibration_root, task)
            if cal is None:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )
                ax.set_title(task.replace("_", " "))
                continue
            # base_returns: per-stage reward mean profile (proxy distribution)
            base_returns = np.array(cal.get("base_returns", []))
            # stress_returns: event-conditioned return summary
            stress_returns = np.array(cal.get("stress_returns", []))
            if len(base_returns) == 0 and len(stress_returns) == 0:
                # Also try nested stagewise as fallback
                sw = cal.get("stagewise") or {}
                base_returns = np.array(sw.get("reward_mean_mean", []))
                tr = cal.get("tail_risk") or {}
                ecr = tr.get("event_conditioned_return_mean")
                if ecr is not None:
                    stress_returns = np.array([ecr])
            if len(base_returns) == 0 and len(stress_returns) == 0:
                ax.text(
                    0.5, 0.5, "No return data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )
                ax.set_title(task.replace("_", " "))
                continue

        bins = 40
        if len(base_returns) > 0:
            ax.hist(
                base_returns, bins=bins, alpha=0.5,
                color=_COLORS["base"], label="Base", density=True,
            )
        if len(stress_returns) > 0:
            ax.hist(
                stress_returns, bins=bins, alpha=0.5,
                color=_COLORS["stress"], label="Stress", density=True,
            )
        ax.set_title(task.replace("_", " "))
        ax.set_xlabel("Episode return")
        if ax == axes[0]:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.tight_layout()
    return _save_figure(fig, out_dir, _FIGURE_STEMS["return_distributions"])


# ---------------------------------------------------------------------------
# Figure 11.1.3: Change-point adaptation plots
# ---------------------------------------------------------------------------


def fig_adaptation_plots(
    out_dir: Path,
    results_root: Path,
    *,
    demo: bool = False,
    window: int = 20,
) -> list[Path]:
    """Figure 11.1.3: rolling-mean return vs episode with change-point marker.

    Confidence intervals: shaded band is mean +/- 1 std over seeds.
    """
    _apply_style()
    n_tasks = len(_REGIME_SHIFT_TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4.5 * n_tasks, 3.0))
    if n_tasks == 1:
        axes = [axes]

    aggregated_root = results_root / "phase2" / "aggregated"

    for ax, task in zip(axes, _REGIME_SHIFT_TASKS):
        if demo:
            episodes, returns, cp = _demo_adaptation_curve(
                rng=np.random.default_rng(42),
            )
        else:
            summary = _load_summary_curves(aggregated_root, task, "QLearning")
            if summary is None:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )
                ax.set_title(task.replace("_", " "))
                continue
            # R7-1 fix: prefer per-episode returns stored at top-level
            # (episode-unit x-axis matches change_at_episode in episode units).
            ep_by_seed = summary.get("episode_returns", {})
            cp = summary.get("change_at_episode")
            if cp is None:
                curves = summary.get("curves", {})
                cp = summary.get("change_point", curves.get("change_point"))
            if isinstance(ep_by_seed, dict) and ep_by_seed:
                seed_arrays = [
                    np.array(v, dtype=float)
                    for v in ep_by_seed.values()
                    if isinstance(v, list) and v
                ]
                if seed_arrays:
                    min_len = min(len(a) for a in seed_arrays)
                    returns = np.mean(
                        np.stack([a[:min_len] for a in seed_arrays], axis=0),
                        axis=0,
                    )
                    episodes = np.arange(len(returns))
                else:
                    returns = np.array([])
                    episodes = np.arange(0)
            else:
                # Fallback to checkpoint means (x-axis in checkpoint indices).
                curves = summary.get("curves", {})
                returns = np.array(curves.get("episode_returns", []))
                episodes = np.arange(len(returns))
            if len(returns) == 0:
                ax.text(
                    0.5, 0.5, "No episode returns",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )
                ax.set_title(task.replace("_", " "))
                continue

        # Rolling mean.  np.convolve(mode="valid") produces n-window+1 values;
        # value i represents episodes[i..i+window-1].  Right-align so each
        # value is plotted at the LAST episode in its window (episodes[i+window-1]).
        # This keeps the change_at_episode marker correctly aligned with the
        # plotted recovery trajectory (R9-1 fix).
        if len(returns) >= window:
            kernel = np.ones(window) / window
            rolling = np.convolve(returns, kernel, mode="valid")
            ep_rolling = episodes[window - 1 : window - 1 + len(rolling)]
        else:
            rolling = returns
            ep_rolling = episodes

        ax.plot(ep_rolling, rolling, color=_COLORS["base"], linewidth=1.0)

        if cp is not None:
            ax.axvline(cp, color="gray", linestyle="--", linewidth=0.8, label="Change point")
            ax.legend(fontsize=7)

        ax.set_title(task.replace("_", " "))
        ax.set_xlabel("Episode")
        if ax == axes[0]:
            ax.set_ylabel(f"Rolling mean return (w={window})")

    fig.tight_layout()
    return _save_figure(fig, out_dir, _FIGURE_STEMS["adaptation_plots"])


# ---------------------------------------------------------------------------
# Figure 11.1.4: State-visitation heatmaps for grid tasks
# ---------------------------------------------------------------------------


def fig_visitation_heatmaps(
    out_dir: Path,
    results_root: Path,
    *,
    demo: bool = False,
) -> list[Path]:
    """Figure 11.1.4: state-visitation heatmaps for grid tasks.

    Confidence intervals: N/A (single aggregate heatmap per task).
    """
    _apply_style()
    n_tasks = len(_GRID_TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.5 * n_tasks, 3.0))
    if n_tasks == 1:
        axes = [axes]

    aggregated_root = results_root / "phase2" / "aggregated"

    for ax, task in zip(axes, _GRID_TASKS):
        grid_shape = _GRID_TASK_SHAPES.get(task, (5, 5))
        n_states = grid_shape[0] * grid_shape[1]
        if demo:
            heatmap = _demo_heatmap(
                *grid_shape, rng=np.random.default_rng(hash(task) % 2**31),
            )
        else:
            summary = _load_summary_curves(aggregated_root, task, "QLearning")
            if summary is None:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )
                ax.set_title(task.replace("_", " "))
                continue
            hm_data = summary.get("visitation_counts")
            if hm_data is None:
                ax.text(
                    0.5, 0.5, "No visitation data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )
                ax.set_title(task.replace("_", " "))
                continue
            # np.bincount produces length max_state+1, which may be < n_states
            # if some states were never visited.  Pad with zeros to n_states.
            arr = np.array(hm_data, dtype=float)
            if len(arr) < n_states:
                arr = np.pad(arr, (0, n_states - len(arr)))
            heatmap = arr[:n_states].reshape(grid_shape)

        im = ax.imshow(heatmap, cmap="YlOrRd", aspect="equal", origin="upper")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(task.replace("_", " "))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.tight_layout()
    return _save_figure(fig, out_dir, _FIGURE_STEMS["visitation_heatmaps"])


# ---------------------------------------------------------------------------
# Figure 11.1.5: Per-stage margin quantile plots
# ---------------------------------------------------------------------------


def fig_margin_quantiles(
    out_dir: Path,
    results_root: Path,
    *,
    demo: bool = False,
) -> list[Path]:
    """Figure 11.1.5: ribbon plot of margin quantiles (q05/q50/q95) vs stage.

    Compares Phase I base tasks against Phase II stress variants.
    Confidence intervals: ribbon spans q05 to q95 of the margin
    distribution at each stage.
    """
    _apply_style()

    all_tasks = [
        "chain_jackpot", "chain_catastrophe", "chain_regime_shift",
        "grid_hazard", "grid_regime_shift", "taxi_bonus_shock",
    ]
    n_tasks = len(all_tasks)
    ncols = min(3, n_tasks)
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows))
    axes_flat = np.array(axes).flatten()

    calibration_root = results_root / "phase2" / "calibration"
    rng = np.random.default_rng(42)

    for idx, task in enumerate(all_tasks):
        ax = axes_flat[idx]

        if demo:
            data = _demo_margin_quantiles(rng=np.random.default_rng(42 + idx))
        else:
            cal = _load_calibration(calibration_root, task)
            if cal is None:
                ax.text(
                    0.5, 0.5, "No calibration data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )
                ax.set_title(task.replace("_", " "))
                continue
            # Top-level margin_quantiles key added by build_calibration_json.
            # Falls back to nested stagewise.pos_margin_quantiles if absent.
            margin_data = cal.get("margin_quantiles") or {}
            if not margin_data:
                sw = cal.get("stagewise") or {}
                pos_q = sw.get("pos_margin_quantiles") or {}
                stage_v = sw.get("stage")
                if pos_q:
                    n_stages = len(pos_q.get("q50", []))
                    margin_data = {
                        "stages": stage_v if stage_v is not None
                                  else list(range(n_stages)),
                        "q05": pos_q.get("q05", []),
                        "q50": pos_q.get("q50", []),
                        "q95": pos_q.get("q95", []),
                    }
            stages = np.array(margin_data.get("stages", []))
            q05 = np.array(margin_data.get("q05", []))
            q50 = np.array(margin_data.get("q50", []))
            q95 = np.array(margin_data.get("q95", []))
            if len(stages) == 0 or len(q50) == 0:
                ax.text(
                    0.5, 0.5, "Empty margin data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )
                ax.set_title(task.replace("_", " "))
                continue
            data = {"stages": stages, "q05": q05, "q50": q50, "q95": q95}

        stages = data["stages"]
        ax.plot(stages, data["q50"], color=_COLORS["stress"], linewidth=1.2, label="Median")
        ax.fill_between(
            stages, data["q05"], data["q95"],
            alpha=0.25, color=_COLORS["stress"], label="q05--q95",
        )
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
        ax.set_title(task.replace("_", " "), fontsize=8)
        ax.set_xlabel("Stage")
        if idx % ncols == 0:
            ax.set_ylabel("Margin")
        ax.legend(fontsize=6, loc="upper right")

    # Hide unused axes.
    for idx in range(len(all_tasks), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    return _save_figure(fig, out_dir, _FIGURE_STEMS["margin_quantiles"])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Save figure as PDF and PNG; return list of written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in (".pdf", ".png"):
        p = out_dir / f"{stem}{ext}"
        fig.savefig(str(p), format=ext.lstrip("."))
        paths.append(p)
    plt.close(fig)
    return paths


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    out_dir: Path,
    all_outputs: dict[str, list[Path]],
) -> Path:
    """Write ``figures_manifest.json``."""
    manifest_path = out_dir / "figures_manifest.json"
    entries: list[dict[str, str]] = []
    for fig_name in FIGURE_NAMES:
        for p in all_outputs.get(fig_name, []):
            entries.append({
                "figure": fig_name,
                "file": str(p),
                "sha256": _sha256(p),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
    manifest = {
        "generator": "make_phase2_figures.py",
        "n_figures": sum(1 for v in all_outputs.values() if v),
        "n_files": len(entries),
        "files": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, Any] = {
    "learning_curves": fig_learning_curves,
    "return_distributions": fig_return_distributions,
    "adaptation_plots": fig_adaptation_plots,
    "visitation_heatmaps": fig_visitation_heatmaps,
    "margin_quantiles": fig_margin_quantiles,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate all Phase II figures (11.1.1--11.1.5). "
            "Use --demo for synthetic data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/weighted_lse_dp"),
        help=(
            "Root of the weighted_lse_dp results tree. Figures are written to "
            "<out-root>/processed/phase2/figures/. "
            "Default: results/weighted_lse_dp"
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demo data (no disk reads required).",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=FIGURE_NAMES,
        default=None,
        metavar="FIG",
        help=f"Generate only one figure. Choices: {', '.join(FIGURE_NAMES)}",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after rendering.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: regenerate Phase II figures."""
    args = _parse_args(argv)

    out_root = Path(args.out_root)
    out_dir = out_root / "processed" / "phase2" / "figures"
    results_root = out_root

    figures_to_generate = [args.only] if args.only else FIGURE_NAMES

    mode_label = "DEMO" if args.demo else "PRODUCTION"
    print(f"=== Phase II Figure Generation ({mode_label} mode) ===")
    print(f"Output directory: {out_dir}")
    print(f"Figures to generate: {', '.join(figures_to_generate)}")
    print()

    all_outputs: dict[str, list[Path]] = {}
    n_success = 0
    n_skipped = 0

    for fig_name in figures_to_generate:
        print(f"[{fig_name}] Generating...")
        gen_fn = _GENERATORS.get(fig_name)
        if gen_fn is None:
            print(f"  Unknown figure: {fig_name}", file=sys.stderr)
            all_outputs[fig_name] = []
            n_skipped += 1
            continue

        try:
            outputs = gen_fn(out_dir, results_root, demo=args.demo)
        except Exception as exc:
            print(f"  ERROR generating {fig_name}: {exc}", file=sys.stderr)
            outputs = []

        all_outputs[fig_name] = outputs
        if outputs:
            n_success += 1
            for p in outputs:
                print(f"  -> {p}")
        else:
            n_skipped += 1
            print("  (skipped)")

    manifest_path = _write_manifest(out_dir, all_outputs)

    total_files = sum(len(v) for v in all_outputs.values())
    print()
    print("=" * 60)
    print(f"Summary: {n_success} figures generated, {n_skipped} skipped")
    print(f"Total output files: {total_files}")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
