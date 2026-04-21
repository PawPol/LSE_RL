#!/usr/bin/env python
"""Phase IV-A: generate activation-analysis figures.

Reads aggregated Phase IV-A results and per-transition NPZ diagnostics
to produce publication-quality figures for the activation search analysis.

Figures produced:
  1. activation_frontier.png       -- per-task operator component time series
  2. natural_shift_distribution.png -- histograms of natural_shift per task
  3. effective_discount_separation.png -- histograms of delta_effective_discount
  4. safe_vs_classical_target_separation.png -- histograms of target_gap_same_gamma_base
  5. task_search_frontier.png      -- scatter of all 168 candidates
  6. negative_control_replay.png   -- bar chart of mean|u| vs gate threshold

Confidence intervals: not applicable (single-seed counterfactual replay).
All seeds included; no filtering applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load NPZ file; return dict of arrays."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _get_replay_dirs(project_root: Path) -> dict[str, Path]:
    """Return {tag: path} for each counterfactual replay task directory."""
    base = project_root / "results/weighted_lse_dp/phase4/counterfactual_replay"
    if not base.is_dir():
        return {}
    return {d.name: d for d in sorted(base.iterdir()) if d.is_dir()}


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 1: activation_frontier
# ---------------------------------------------------------------------------

def _fig_activation_frontier(
    replay_dirs: dict[str, Path], output_dir: Path
) -> None:
    """Per-task subplots of U_safe_ref, u_target, u_ref_used, u_tr_cap by stage."""
    tags = sorted(replay_dirs.keys())
    if not tags:
        warnings.warn("No replay dirs found; skipping activation_frontier")
        return

    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), squeeze=False)

    for i, tag in enumerate(tags):
        ax = axes[0, i]
        npz_path = replay_dirs[tag] / "replay_diagnostics.npz"
        if not npz_path.exists():
            ax.set_title(f"{tag}\n(no NPZ)")
            continue

        d = _load_npz(npz_path)
        stages = d.get("stage", np.array([]))
        if stages.size == 0:
            ax.set_title(f"{tag}\n(empty)")
            continue

        unique_stages = np.sort(np.unique(stages))
        fields = {
            "U_safe_ref": "U_safe_ref",
            "u_target": "u_target",
            "u_ref_used": "u_ref_used",
            "u_tr_cap": "u_tr_cap",
        }

        for label, key in fields.items():
            if key not in d:
                continue
            arr = d[key]
            means = [np.mean(np.abs(arr[stages == s])) for s in unique_stages]
            ax.plot(unique_stages, means, marker=".", markersize=3, label=label, linewidth=1)

        ax.set_xlabel("stage")
        ax.set_ylabel("mean |value|")
        ax.set_title(tag, fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    fig.suptitle("Activation Frontier: Operator Components by Stage", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir / "activation_frontier.png")


# ---------------------------------------------------------------------------
# Figure 2: natural_shift_distribution
# ---------------------------------------------------------------------------

def _fig_natural_shift(replay_dirs: dict[str, Path], output_dir: Path) -> None:
    tags = sorted(replay_dirs.keys())
    if not tags:
        warnings.warn("No replay dirs; skipping natural_shift_distribution")
        return

    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)

    for i, tag in enumerate(tags):
        ax = axes[0, i]
        npz_path = replay_dirs[tag] / "replay_diagnostics.npz"
        if not npz_path.exists():
            ax.set_title(f"{tag}\n(no NPZ)")
            continue

        d = _load_npz(npz_path)
        ns = d.get("natural_shift", np.array([]))
        if ns.size == 0:
            ax.set_title(f"{tag}\n(empty)")
            continue

        ax.hist(ns, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(5e-3, color="red", linestyle="--", linewidth=1, label="+5e-3")
        ax.axvline(-5e-3, color="red", linestyle="--", linewidth=1, label="-5e-3")
        ax.set_xlabel("natural_shift")
        ax.set_ylabel("count")
        ax.set_title(tag, fontsize=9)
        ax.legend(fontsize=6)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

    fig.suptitle("Natural Shift Distribution", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir / "natural_shift_distribution.png")


# ---------------------------------------------------------------------------
# Figure 3: effective_discount_separation
# ---------------------------------------------------------------------------

def _fig_effective_discount(replay_dirs: dict[str, Path], output_dir: Path) -> None:
    tags = sorted(replay_dirs.keys())
    if not tags:
        warnings.warn("No replay dirs; skipping effective_discount_separation")
        return

    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)

    for i, tag in enumerate(tags):
        ax = axes[0, i]
        npz_path = replay_dirs[tag] / "replay_diagnostics.npz"
        if not npz_path.exists():
            ax.set_title(f"{tag}\n(no NPZ)")
            continue

        d = _load_npz(npz_path)
        dd = d.get("delta_effective_discount", np.array([]))
        if dd.size == 0:
            ax.set_title(f"{tag}\n(empty)")
            continue

        ax.hist(dd, bins=80, color="darkorange", edgecolor="none", alpha=0.8)
        ax.axvline(1e-3, color="red", linestyle="--", linewidth=1, label="+1e-3")
        ax.axvline(-1e-3, color="red", linestyle="--", linewidth=1, label="-1e-3")
        ax.set_xlabel("delta_effective_discount")
        ax.set_ylabel("count")
        ax.set_title(tag, fontsize=9)
        ax.legend(fontsize=6)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

    fig.suptitle("Effective Discount Separation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir / "effective_discount_separation.png")


# ---------------------------------------------------------------------------
# Figure 4: safe_vs_classical_target_separation
# ---------------------------------------------------------------------------

def _fig_target_separation(replay_dirs: dict[str, Path], output_dir: Path) -> None:
    tags = sorted(replay_dirs.keys())
    if not tags:
        warnings.warn("No replay dirs; skipping safe_vs_classical_target_separation")
        return

    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)

    for i, tag in enumerate(tags):
        ax = axes[0, i]
        npz_path = replay_dirs[tag] / "replay_diagnostics.npz"
        if not npz_path.exists():
            ax.set_title(f"{tag}\n(no NPZ)")
            continue

        d = _load_npz(npz_path)
        tg = d.get("target_gap_same_gamma_base", np.array([]))
        if tg.size == 0:
            ax.set_title(f"{tag}\n(empty)")
            continue

        ax.hist(tg, bins=80, color="seagreen", edgecolor="none", alpha=0.8)
        ax.set_xlabel("target_gap_same_gamma_base")
        ax.set_ylabel("count")
        ax.set_title(tag, fontsize=9)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

    fig.suptitle("Safe vs Classical Target Separation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir / "safe_vs_classical_target_separation.png")


# ---------------------------------------------------------------------------
# Figure 5: task_search_frontier
# ---------------------------------------------------------------------------

def _fig_task_search_frontier(project_root: Path, output_dir: Path) -> None:
    scores_path = project_root / "results/weighted_lse_dp/phase4/task_search/candidate_scores.csv"
    selected_path = project_root / "results/weighted_lse_dp/phase4/task_search/selected_tasks.json"

    if not scores_path.exists():
        warnings.warn(f"candidate_scores.csv not found; skipping task_search_frontier")
        return

    # Read candidates
    families: list[str] = []
    x_vals: list[float] = []
    y_vals: list[float] = []
    idxs: list[int] = []

    with open(scores_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("error"):
                continue
            try:
                families.append(r["family"])
                x_vals.append(float(r["mean_abs_u_pred"]))
                y_vals.append(float(r["frac_u_ge_5e3"]))
                idxs.append(int(r["idx"]))
            except (ValueError, KeyError):
                continue

    selected_idxs: set[int] = set()
    if selected_path.exists():
        raw = _load_json(selected_path)
        task_list = raw.get("tasks", raw) if isinstance(raw, dict) else raw
        for t in task_list:
            selected_idxs.add(t["idx"])

    # Assign colors per family
    unique_families = sorted(set(families))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(unique_families))
    fam_colors = {f: cmap(i) for i, f in enumerate(unique_families)}

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot non-selected first, then selected on top
    for j in range(len(x_vals)):
        if idxs[j] in selected_idxs:
            continue
        ax.scatter(
            x_vals[j], y_vals[j],
            c=[fam_colors[families[j]]], s=18, alpha=0.6, edgecolors="none"
        )

    for j in range(len(x_vals)):
        if idxs[j] not in selected_idxs:
            continue
        ax.scatter(
            x_vals[j], y_vals[j],
            c=[fam_colors[families[j]]], s=120, alpha=1.0,
            marker="*", edgecolors="black", linewidths=0.5
        )

    # Threshold lines
    ax.axvline(2e-3, color="grey", linestyle=":", linewidth=1, label="x=2e-3 threshold")
    ax.axhline(0.05, color="grey", linestyle="--", linewidth=1, label="y=0.05 threshold")

    # Legend for families
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=fam_colors[f],
               markersize=6, label=f)
        for f in unique_families
    ]
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="grey",
                          markeredgecolor="black", markersize=10, label="selected"))
    ax.legend(handles=handles, fontsize=7, loc="upper left")

    ax.set_xlabel("mean_abs_u_pred")
    ax.set_ylabel("frac_u_ge_5e3")
    ax.set_title("Task Search Frontier (168 candidates)", fontsize=11)
    fig.tight_layout()
    _savefig(fig, output_dir / "task_search_frontier.png")


# ---------------------------------------------------------------------------
# Figure 6: negative_control_replay
# ---------------------------------------------------------------------------

def _fig_negative_control(input_dir: Path, output_dir: Path) -> None:
    diag_path = input_dir / "activation_diagnostics.json"
    if not diag_path.exists():
        warnings.warn("activation_diagnostics.json not found; skipping negative_control_replay")
        return

    diags = _load_json(diag_path)
    tags = [d["tag"] for d in diags]
    mean_abs_u = [d["mean_abs_u"] for d in diags]

    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(len(tags))
    ax.bar(x_pos, mean_abs_u, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axhline(5e-3, color="red", linestyle="--", linewidth=1.2, label="gate threshold (5e-3)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean |u|")
    ax.set_title("Negative Control: mean|u| vs Gate Threshold", fontsize=11)
    ax.legend(fontsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    _savefig(fig, output_dir / "negative_control_replay.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Generate Phase IV-A figures."""
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[3]

    print(f"Phase IV-A figure generation")
    print(f"  input_dir:  {input_dir}")
    print(f"  output_dir: {output_dir}")
    print()

    replay_dirs = _get_replay_dirs(project_root)

    _fig_activation_frontier(replay_dirs, output_dir)
    _fig_natural_shift(replay_dirs, output_dir)
    _fig_effective_discount(replay_dirs, output_dir)
    _fig_target_separation(replay_dirs, output_dir)
    _fig_task_search_frontier(project_root, output_dir)
    _fig_negative_control(input_dir, output_dir)

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True,
                    help="Path to results/processed/phase4A/")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Path to output directory for figures")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
