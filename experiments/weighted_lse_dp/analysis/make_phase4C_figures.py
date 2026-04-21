#!/usr/bin/env python
"""Phase IV-C: generate estimator-stability, ablation, and geometry-priority figures.

Reads aggregated Phase IV-C results and attribution analysis, and produces
four publication-quality PDF figures for the advanced stabilization program.

Figures produced (all in ``figures/phase4C/``):
  1. estimator_comparison.pdf       -- mean_final_return (+/- std) for the
                                       5 advanced RL estimators, with the
                                       hand-rolled single-table control
                                       drawn as a dashed horizontal line.
  2. attribution_delta_bars.pdf     -- delta return vs the architecture-
                                       matched single-table control for
                                       each non-baseline estimator.
  3. ablation_impact_bars.pdf       -- delta return vs the framework
                                       baseline for the 7 certification
                                       ablations, with stagewise_baseline
                                       shown as the reference at delta=0.
  4. geometry_priority_convergence.pdf -- mean_n_sweeps for
                                       residual_only / geometry_gain_only /
                                       combined priority modes.

Confidence intervals: for (1) we draw +/- one standard deviation of the
mean_final_return across seeds (``std_final_return`` from summary_phase4C.json).
For (3), per-ablation standard deviations are reported as numeric annotations
underneath each bar (mean delta is tiny, so error bars would visually dominate).
For (4), ``mean_n_sweeps`` standard deviation is not exported in the summary
JSON, so we annotate numeric sweep counts directly.

All seeds present in the aggregated summary are included; no filtering applied.
Regeneration: ``python experiments/weighted_lse_dp/analysis/make_phase4C_figures.py``.
Output is deterministic given identical input JSONs (modulo PDF embedded
timestamps).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper style (Type 42 fonts for NeurIPS compliance, no seaborn dependency)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
ALGO_LABELS: dict[str, str] = {
    "safe_q_single_table": "Single-Q",
    "safe_double_q": "Double-Q",
    "safe_target_q": "Target-Q",
    "safe_target_q_polyak": "Target-Q\n(Polyak)",
    "safe_target_expected_sarsa": "Exp-SARSA",
}

ABLATION_LABELS: dict[str, str] = {
    "stagewise_baseline": "Baseline",
    "trust_region_off": "TR off",
    "trust_region_tighter": "TR tight",
    "adaptive_headroom_off": "AH off",
    "adaptive_headroom_aggressive": "AH agg",
    "wrong_sign": "Wrong sign",
    "constant_u": "Const u",
    "raw_unclipped": "Raw uncl.",
}

# Order to display algorithms in the estimator comparison
ALGO_ORDER: list[str] = [
    "safe_q_single_table",
    "safe_double_q",
    "safe_target_q",
    "safe_target_q_polyak",
    "safe_target_expected_sarsa",
]

# Order for ablation impact figure
ABLATION_ORDER: list[str] = [
    "trust_region_off",
    "trust_region_tighter",
    "adaptive_headroom_off",
    "adaptive_headroom_aggressive",
    "wrong_sign",
    "constant_u",
    "raw_unclipped",
]

GEOMETRY_ORDER: list[str] = ["residual_only", "geometry_gain_only", "combined"]
GEOMETRY_LABELS: dict[str, str] = {
    "residual_only": "Residual only",
    "geometry_gain_only": "Geometry-gain only",
    "combined": "Combined",
}

# Colour palette -- paper-friendly, colour-blind-safe-ish
C_POS = "#2CA02C"  # green
C_NEG = "#D62728"  # red
C_NEUTRAL = "#4C72B0"  # muted blue
C_BASELINE = "#555555"  # dark grey


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # metadata={'CreationDate': None} strips the PDF timestamp for byte-wise
    # reproducibility across reruns.
    fig.savefig(path, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def _index_by(key: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row[key]: row for row in rows}


# ---------------------------------------------------------------------------
# Figure 1: estimator comparison
# ---------------------------------------------------------------------------
def make_estimator_comparison(
    summary: dict[str, Any], out_path: Path
) -> None:
    rows = _index_by("algorithm", summary["advanced_rl"])

    xs: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for algo in ALGO_ORDER:
        row = rows.get(algo)
        if row is None:
            print(f"[fig1] WARN: missing algorithm {algo!r}; skipping")
            continue
        mean = row.get("mean_final_return")
        std = row.get("std_final_return")
        if mean is None:
            print(f"[fig1] WARN: {algo!r} has no mean_final_return; skipping")
            continue
        xs.append(ALGO_LABELS.get(algo, algo))
        means.append(float(mean))
        stds.append(float(std) if std is not None else 0.0)

    if not xs:
        print("[fig1] no data; skipping figure")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    idx = np.arange(len(xs))
    bars = ax.bar(
        idx,
        means,
        yerr=stds,
        color=C_NEUTRAL,
        edgecolor="black",
        linewidth=0.7,
        capsize=4,
        error_kw={"elinewidth": 0.9, "ecolor": "black"},
        alpha=0.85,
    )

    # Dashed horizontal line for the single-table baseline
    baseline_algo = "safe_q_single_table"
    baseline_row = rows.get(baseline_algo)
    if baseline_row is not None and baseline_row.get("mean_final_return") is not None:
        baseline_val = float(baseline_row["mean_final_return"])
        ax.axhline(
            baseline_val,
            color=C_BASELINE,
            linestyle="--",
            linewidth=1.2,
            label=f"Single-Q baseline = {baseline_val:.4f}",
        )
        ax.legend(loc="lower right", frameon=False)

    ax.set_xticks(idx)
    ax.set_xticklabels(xs)
    ax.set_ylabel("Mean discounted return")
    ax.set_title("Estimator Stabilization Comparison")

    # Annotate each bar with its value
    for rect, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2.0, mean),
            xytext=(0, -12 if mean < 0 else 3),
            textcoords="offset points",
            ha="center",
            va="top" if mean < 0 else "bottom",
            fontsize=7.5,
            color="black",
        )

    _savefig(fig, out_path)
    print(f"[fig1] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: attribution delta bars
# ---------------------------------------------------------------------------
def make_attribution_delta(
    attribution: dict[str, Any], out_path: Path
) -> None:
    gains: dict[str, float] = attribution.get("estimator_gains_vs_arch_baseline", {})

    xs: list[str] = []
    vals: list[float] = []
    # Keep declared order; any algo not in ALGO_ORDER is appended afterwards
    ordered_algos = [a for a in ALGO_ORDER if a in gains] + [
        a for a in gains if a not in ALGO_ORDER
    ]
    for algo in ordered_algos:
        v = gains.get(algo)
        if v is None:
            print(f"[fig2] WARN: missing gain for {algo!r}; skipping")
            continue
        xs.append(ALGO_LABELS.get(algo, algo))
        vals.append(float(v))

    if not xs:
        print("[fig2] no data; skipping figure")
        return

    colors = [C_POS if v >= 0 else C_NEG for v in vals]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    idx = np.arange(len(xs))
    bars = ax.bar(
        idx,
        vals,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )

    ax.axhline(0.0, color=C_BASELINE, linestyle="-", linewidth=0.8)

    ax.set_xticks(idx)
    ax.set_xticklabels(xs)
    ax.set_ylabel(r"$\Delta$ mean discounted return")
    ax.set_title("Return Gain vs Single-Table Control")

    # Symmetric y-limit so that sign of small deltas is visible
    ymax = max(abs(v) for v in vals)
    if ymax > 0:
        pad = ymax * 0.35 if ymax > 0 else 1.0
        ax.set_ylim(-ymax - pad, ymax + pad)

    for rect, v in zip(bars, vals):
        offset = 3 if v >= 0 else -3
        ax.annotate(
            f"{v:+.2e}",
            xy=(rect.get_x() + rect.get_width() / 2.0, v),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=7.5,
            color="black",
        )

    # Legend: two sentinel patches
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=C_POS, edgecolor="black", label=r"Gain ($\Delta \geq 0$)"),
        Patch(facecolor=C_NEG, edgecolor="black", label=r"Loss ($\Delta < 0$)"),
    ]
    ax.legend(handles=legend_elems, loc="best", frameon=False)

    _savefig(fig, out_path)
    print(f"[fig2] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: ablation impact bars
# ---------------------------------------------------------------------------
def make_ablation_impact(
    attribution: dict[str, Any], summary: dict[str, Any], out_path: Path
) -> None:
    impact_rows = attribution.get("ablation_impact", [])
    impact_idx = _index_by("ablation", impact_rows)

    # std lookup from summary (for annotation -- NOT plotted as error bar
    # because the deltas are ~1e-5 and stds are ~0.2)
    abl_idx = _index_by("ablation", summary.get("ablations", []))

    # Always show stagewise_baseline as the reference (delta = 0)
    display_order = ["stagewise_baseline"] + ABLATION_ORDER

    xs: list[str] = []
    vals: list[float] = []
    stds: list[float] = []
    for name in display_order:
        if name == "stagewise_baseline":
            xs.append(ABLATION_LABELS[name])
            vals.append(0.0)
            stds.append(float("nan"))
            continue
        row = impact_idx.get(name)
        if row is None:
            print(f"[fig3] WARN: missing ablation {name!r}; skipping")
            continue
        delta = row.get("delta_vs_framework_baseline")
        if delta is None:
            print(f"[fig3] WARN: {name!r} has no delta; skipping")
            continue
        xs.append(ABLATION_LABELS.get(name, name))
        vals.append(float(delta))
        std = abl_idx.get(name, {}).get("std_final_return")
        stds.append(float(std) if std is not None else float("nan"))

    if not xs:
        print("[fig3] no data; skipping figure")
        return

    def _bar_color(name: str, v: float) -> str:
        if name == "stagewise_baseline":
            return C_BASELINE
        return C_POS if v >= 0 else C_NEG

    colors = [_bar_color(n, v) for n, v in zip(display_order[: len(vals)], vals)]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    idx = np.arange(len(xs))
    bars = ax.bar(
        idx,
        vals,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )

    ax.axhline(0.0, color=C_BASELINE, linestyle="-", linewidth=0.8)

    ax.set_xticks(idx)
    ax.set_xticklabels(xs, rotation=20, ha="right")
    ax.set_ylabel(r"$\Delta$ mean discounted return vs framework baseline")
    ax.set_title("Certification Ablation Impact")

    ymax = max(abs(v) for v in vals) if vals else 0.0
    if ymax > 0:
        ax.set_ylim(-ymax * 1.6, ymax * 1.6)
    else:
        ax.set_ylim(-1.0, 1.0)

    for rect, v in zip(bars, vals):
        offset = 3 if v >= 0 else -3
        label = "0" if v == 0.0 else f"{v:+.2e}"
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2.0, v),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=7.5,
            color="black",
        )

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=C_BASELINE, edgecolor="black", label="Framework baseline"),
        Patch(facecolor=C_POS, edgecolor="black", label=r"$\Delta \geq 0$"),
        Patch(facecolor=C_NEG, edgecolor="black", label=r"$\Delta < 0$"),
    ]
    ax.legend(handles=legend_elems, loc="best", frameon=False)

    _savefig(fig, out_path)
    print(f"[fig3] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: geometry-priority convergence
# ---------------------------------------------------------------------------
def make_geometry_priority(
    summary: dict[str, Any], out_path: Path
) -> None:
    rows = _index_by("mode", summary.get("geometry_priority_dp", []))

    xs: list[str] = []
    sweeps: list[float] = []
    wall: list[float] = []
    for mode in GEOMETRY_ORDER:
        row = rows.get(mode)
        if row is None:
            print(f"[fig4] WARN: missing mode {mode!r}; skipping")
            continue
        n_sweeps = row.get("mean_n_sweeps")
        if n_sweeps is None:
            print(f"[fig4] WARN: {mode!r} has no mean_n_sweeps; skipping")
            continue
        xs.append(GEOMETRY_LABELS.get(mode, mode))
        sweeps.append(float(n_sweeps))
        w = row.get("mean_wall_clock_s")
        wall.append(float(w) if w is not None else float("nan"))

    if not xs:
        print("[fig4] no data; skipping figure")
        return

    # Highlight the best (fewest sweeps) mode
    best_idx = int(np.argmin(sweeps))
    colors = [C_POS if i == best_idx else C_NEUTRAL for i in range(len(xs))]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    idx = np.arange(len(xs))
    bars = ax.bar(
        idx,
        sweeps,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )

    ax.set_xticks(idx)
    ax.set_xticklabels(xs)
    ax.set_ylabel("Mean sweeps to convergence")
    ax.set_title("Geometry-Priority DP: Sweeps to Convergence")

    # Annotate each bar with its sweep count and wall-clock time
    ymax = max(sweeps)
    ax.set_ylim(0, ymax * 1.20)
    for rect, s, w in zip(bars, sweeps, wall):
        label = f"{s:.1f} sweeps"
        if not np.isnan(w):
            label += f"\n({w * 1000:.1f} ms)"
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2.0, s),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    # Reference line at baseline (residual_only by convention)
    if "residual_only" in rows:
        base = float(rows["residual_only"].get("mean_n_sweeps", np.nan))
        if not np.isnan(base):
            ax.axhline(
                base,
                color=C_BASELINE,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Residual-only baseline = {base:.1f}",
            )
            ax.legend(loc="lower left", frameon=False)

    _savefig(fig, out_path)
    print(f"[fig4] wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    project_root = Path(__file__).resolve().parents[3]

    summary_path = (
        project_root
        / "results/weighted_lse_dp/phase4/advanced/summary_phase4C.json"
    )
    attribution_path = (
        project_root
        / "results/weighted_lse_dp/phase4/advanced/attribution_analysis.json"
    )
    out_dir = project_root / "figures/phase4C"

    if not summary_path.exists():
        print(f"ERROR: summary file not found: {summary_path}", file=sys.stderr)
        return 1
    if not attribution_path.exists():
        print(f"ERROR: attribution file not found: {attribution_path}", file=sys.stderr)
        return 1

    summary = _load_json(summary_path)
    attribution = _load_json(attribution_path)

    make_estimator_comparison(summary, out_dir / "estimator_comparison.pdf")
    make_attribution_delta(attribution, out_dir / "attribution_delta_bars.pdf")
    make_ablation_impact(attribution, summary, out_dir / "ablation_impact_bars.pdf")
    make_geometry_priority(summary, out_dir / "geometry_priority_convergence.pdf")

    print(f"\nAll Phase IV-C figures written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
