#!/usr/bin/env python
"""Phase IV-B: generate translation-experiment figures.

Figures produced (spec §12.1 mandatory main figures):
  1. rl_learning_curves.png           -- RL learning curves on activation suite
  2. tail_risk_outcomes.png           -- tail-risk / adaptation / sample-efficiency
  3. outcome_vs_diagnostic_scatter.png -- outcome delta vs diagnostic delta scatter
  4. matched_control_comparison.png   -- classical vs safe-zero vs safe-nonlinear
  5. diagnostic_sweep.png             -- activation metrics and outcomes vs u_max
  6. negative_control_outcomes.png    -- Phase III replay outcome comparison

Appendix figures (spec §12.2):
  A1. per_stage_diagnostics.png
  A2. event_conditioned_diagnostics.png
  A3. null_translation_cases.png

CLI
---
  python make_phase4B_figures.py --results-dir <dir> [--output-dir <dir>]

  --results-dir   Top-level Phase IV-B results directory
                  (parent of translation/, diagnostic_sweep/, etc.)
  --output-dir    Destination for figures.
                  Defaults to <results-dir>/analysis/.  (spec §Q12)

Algorithm name mapping follows spec §Q2:
  *_stagewise -> safe-nonlinear; *_zero -> safe-zero; classical_* -> classical
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ---------------------------------------------------------------------------
# Palette / style
# ---------------------------------------------------------------------------

# Canonical color per algorithm class
_CLR: dict[str, str] = {
    "classical": "#2166ac",
    "safe-zero": "#74c476",
    "safe-nonlinear": "#d6604d",
}
_MARKERS: dict[str, str] = {
    "classical": "o",
    "safe-zero": "s",
    "safe-nonlinear": "^",
}


def _algo_class(name: str) -> str:
    """Map algorithm dir name to class label (spec §Q2)."""
    n = name.lower()
    if n.endswith("_stagewise"):
        return "safe-nonlinear"
    if n.endswith("_zero"):
        return "safe-zero"
    if n.startswith("classical"):
        return "classical"
    if n.startswith("safe"):
        return "safe-nonlinear"
    return "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def _ci_bar(ax: plt.Axes, x: float, lo: float, hi: float, **kw: Any) -> None:
    """Draw a vertical CI bar at position x."""
    ax.plot([x, x], [lo, hi], linewidth=2, **kw)


def _resolve_translation_dir(results_dir: Path) -> Path | None:
    """Return the directory holding per-task translation runs.

    The runner writes raw seed data to ``<results_dir>/<task>/<algo>/seed_N/``
    (task dirs directly under the suite root), while the spec Q12 layout wraps
    these in ``<results_dir>/translation/``. Accept both.
    """
    nested = results_dir / "translation"
    if nested.is_dir():
        return nested
    task_dirs = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name not in {
            "translation", "counterfactual_replay", "diagnostic_sweep",
            "analysis", "aggregated",
        }
    ]
    if not task_dirs:
        return None
    return results_dir


def _resolve_counterfactual_dir(results_dir: Path) -> Path | None:
    """Return the directory holding negative-control replay summaries.

    Prefer ``<results_dir>/counterfactual_replay/``; fall back to a sibling
    ``counterfactual_replay/`` at the parent level (the runner's default
    output site — spec §Q8 keeps the negative control suite-agnostic).
    """
    nested = results_dir / "counterfactual_replay"
    if nested.is_dir():
        return nested
    sibling = results_dir.parent / "counterfactual_replay"
    if sibling.is_dir():
        return sibling
    return None


# ---------------------------------------------------------------------------
# Figure 1: RL learning curves
# ---------------------------------------------------------------------------

def _fig_rl_learning_curves(results_dir: Path, output_dir: Path) -> None:
    """Plot discounted-return learning curves per task, coloured by class."""
    trans_dir = _resolve_translation_dir(results_dir)
    if trans_dir is None:
        warnings.warn("No translation dir; skipping rl_learning_curves")
        return

    excluded = {"translation", "counterfactual_replay", "diagnostic_sweep",
                "analysis", "aggregated"}
    task_dirs = sorted(
        d for d in trans_dir.iterdir()
        if d.is_dir() and d.name not in excluded
    )
    if not task_dirs:
        warnings.warn("No task dirs in translation/; skipping rl_learning_curves")
        return

    n = len(task_dirs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, task_dir in enumerate(task_dirs):
        ax = axes[0, i]
        ax.set_title(task_dir.name, fontsize=9)
        ax.set_xlabel("episode")
        ax.set_ylabel("discounted return")

        plotted_any = False
        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            cls = _algo_class(algo_dir.name)

            # Collect per-seed episode returns
            curves: list[np.ndarray] = []
            for seed_dir in sorted(algo_dir.glob("seed_*")):
                npz_path = seed_dir / "metrics.npz"
                if npz_path.exists():
                    try:
                        data = np.load(npz_path, allow_pickle=False)
                        if "episode_return" in data:
                            curves.append(data["episode_return"])
                    except Exception:
                        pass

            if not curves:
                # Try summary.json for a single mean curve
                summary_path = algo_dir / "summary.json"
                if summary_path.exists():
                    try:
                        d = _load_json(summary_path)
                        if "mean_return_curve" in d:
                            curves.append(np.array(d["mean_return_curve"]))
                    except Exception:
                        pass

            if not curves:
                continue

            min_len = min(len(c) for c in curves)
            mat = np.stack([c[:min_len] for c in curves])
            mean_curve = mat.mean(axis=0)
            std_curve = mat.std(axis=0)
            eps = np.arange(min_len)

            color = _CLR.get(cls, "grey")
            ax.plot(eps, mean_curve, color=color, linewidth=1.2, label=cls)
            ax.fill_between(eps, mean_curve - std_curve, mean_curve + std_curve,
                            color=color, alpha=0.2)
            plotted_any = True

        if plotted_any:
            ax.legend(fontsize=7)

    fig.suptitle("RL Learning Curves — Activation Suite", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir / "rl_learning_curves.png")


# ---------------------------------------------------------------------------
# Figure 2: tail-risk / adaptation / sample-efficiency outcomes
# ---------------------------------------------------------------------------

def _fig_tail_risk_outcomes(results_dir: Path, output_dir: Path) -> None:
    """Bar + CI plot for primary outcomes on activated tasks."""
    analysis_dir = results_dir / "analysis"
    step3_path = analysis_dir / "step3_matched_control.json"
    step4_path = analysis_dir / "step4_outcome_interpretation.json"

    if not step3_path.exists() or not step4_path.exists():
        warnings.warn("step3/step4 analysis JSONs missing; skipping tail_risk_outcomes")
        return

    s3 = _load_json(step3_path)
    s4 = _load_json(step4_path)

    tags = sorted(set(s3.keys()) & set(s4.keys()))
    if not tags:
        warnings.warn("No overlapping tasks in step3/step4; skipping tail_risk_outcomes")
        return

    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for i, tag in enumerate(tags):
        ax = axes[0, i]
        ax.set_title(f"{tag}\n{s4[tag].get('primary_metric', 'mean_return')}", fontsize=8)
        ax.set_ylabel("outcome delta (safe-NL minus reference)")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

        effects = {
            "nonlinear\nvs zero": s3[tag].get("nonlinearity_effect", {}),
            "nonlinear\nvs classical": s3[tag].get("total_effect", {}),
        }
        x_pos = np.arange(len(effects))
        for j, (label, eff) in enumerate(effects.items()):
            md = eff.get("mean_diff")
            lo = eff.get("lower")
            hi = eff.get("upper")
            if md is None:
                continue
            cls_key = "safe-nonlinear" if "nonlinear" in label else "classical"
            color = _CLR.get(cls_key, "grey")
            ax.bar(j, md, color=color, alpha=0.7, width=0.5)
            if lo is not None and hi is not None:
                _ci_bar(ax, j, lo, hi, color="black", solid_capstyle="round")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(effects.keys()), fontsize=8)

    fig.suptitle("Outcome Deltas — Matched Control Comparison", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir / "tail_risk_outcomes.png")


# ---------------------------------------------------------------------------
# Figure 3: outcome delta vs diagnostic delta scatter
# ---------------------------------------------------------------------------

def _fig_scatter_outcome_vs_diagnostic(results_dir: Path, output_dir: Path) -> None:
    """Scatter: outcome delta vs diagnostic delta per task, across sweep values."""
    analysis_dir = results_dir / "analysis"
    sweep_path = analysis_dir / "step2_translation_sweep.json"
    if not sweep_path.exists():
        warnings.warn("step2 sweep JSON missing; skipping scatter figure")
        return

    s2 = _load_json(sweep_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("diagnostic delta (mean_abs_natural_shift − baseline)")
    ax.set_ylabel("outcome delta (primary_outcome − baseline)")
    ax.set_title("Outcome Delta vs Diagnostic Delta\n(each point = one sweep value × task)",
                 fontsize=10)

    cmap = matplotlib.colormaps.get_cmap("tab10")
    all_tags = [t for t, v in s2.items() if isinstance(v, dict) and "diag_deltas" in v]
    for idx, tag in enumerate(sorted(all_tags)):
        info = s2[tag]
        diag = info["diag_deltas"]
        outcome = info["outcome_deltas"]
        color = cmap(idx % 10)
        ax.scatter(diag, outcome, color=color, s=40, alpha=0.8, label=tag, zorder=3)

        rho = info.get("spearman_rho")
        if rho is not None:
            # Annotate with rho value near last point
            if diag:
                ax.annotate(f"ρ={rho:.2f}", (diag[-1], outcome[-1]),
                            textcoords="offset points", xytext=(4, 2), fontsize=7,
                            color=color)

    if all_tags:
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    _savefig(fig, output_dir / "outcome_vs_diagnostic_scatter.png")


# ---------------------------------------------------------------------------
# Figure 4: matched-control comparison
# ---------------------------------------------------------------------------

def _fig_matched_control(results_dir: Path, output_dir: Path) -> None:
    """Three-way bar plot: classical vs safe-zero vs safe-nonlinear per task."""
    trans_dir = _resolve_translation_dir(results_dir)
    if trans_dir is None:
        warnings.warn("No translation dir; skipping matched_control_comparison")
        return

    excluded = {"translation", "counterfactual_replay", "diagnostic_sweep",
                "analysis", "aggregated"}
    task_dirs = sorted(
        d for d in trans_dir.iterdir()
        if d.is_dir() and d.name not in excluded
    )
    if not task_dirs:
        return

    n = len(task_dirs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)

    for i, task_dir in enumerate(task_dirs):
        ax = axes[0, i]
        ax.set_title(task_dir.name, fontsize=9)
        ax.set_ylabel("mean return (or primary outcome)")

        class_means: dict[str, list[float]] = {
            "classical": [],
            "safe-zero": [],
            "safe-nonlinear": [],
        }

        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            cls = _algo_class(algo_dir.name)
            if cls not in class_means:
                continue
            summary_path = algo_dir / "summary.json"
            if summary_path.exists():
                try:
                    d = _load_json(summary_path)
                    mean_r = d.get("primary_outcome", d.get("mean_return"))
                    if mean_r is not None:
                        class_means[cls].append(float(mean_r))
                except Exception:
                    pass

        classes = ["classical", "safe-zero", "safe-nonlinear"]
        x_pos = np.arange(len(classes))
        for j, cls in enumerate(classes):
            vals = class_means[cls]
            if not vals:
                continue
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            color = _CLR.get(cls, "grey")
            ax.bar(j, mean_v, yerr=std_v, color=color, alpha=0.8, width=0.55,
                   capsize=4, label=cls)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(classes, fontsize=8, rotation=20, ha="right")

    fig.suptitle("Matched Control Comparison\n(classical / safe-zero / safe-nonlinear)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _savefig(fig, output_dir / "matched_control_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5: diagnostic-strength sweep
# ---------------------------------------------------------------------------

def _fig_diagnostic_sweep(results_dir: Path, output_dir: Path) -> None:
    """Plot activation metrics and outcomes vs u_max for each task."""
    sweep_dir = results_dir / "diagnostic_sweep"
    if not sweep_dir.is_dir():
        warnings.warn("No diagnostic_sweep dir; skipping diagnostic_sweep figure")
        return

    task_dirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not task_dirs:
        return

    n = len(task_dirs)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7), squeeze=False)

    for i, task_dir in enumerate(task_dirs):
        tag = task_dir.name
        sweep_file = task_dir / "sweep_results.json"
        if not sweep_file.exists():
            # Gracefully skip (spec §Q4)
            warnings.warn(f"sweep_results.json missing for {tag}; skipping in figure")
            axes[0, i].set_title(f"{tag}\n(no data)", fontsize=8)
            axes[1, i].set_visible(False)
            continue

        try:
            sweep = _load_json(sweep_file)
        except Exception as exc:
            warnings.warn(f"Could not parse {sweep_file}: {exc}")
            continue

        points = sweep if isinstance(sweep, list) else sweep.get("sweep_points", [])
        if not points:
            continue
        points = sorted(points, key=lambda p: float(p.get("u_max", 0.0)))

        u_maxes = np.array([float(p["u_max"]) for p in points])
        mean_abs_u = np.array([
            float(p.get("mean_abs_natural_shift", p.get("mean_abs_u", 0.0)))
            for p in points
        ])
        outcomes = np.array([
            float(p.get("primary_outcome", p.get("mean_return", 0.0)))
            for p in points
        ])

        ax_top = axes[0, i]
        ax_bot = axes[1, i]

        ax_top.plot(u_maxes, mean_abs_u, color="#d6604d", marker="o", markersize=4,
                    linewidth=1.2)
        ax_top.set_xlabel("u_max")
        ax_top.set_ylabel("mean |natural_shift|")
        ax_top.set_title(tag, fontsize=9)
        ax_top.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        ax_bot.plot(u_maxes, outcomes, color="#2166ac", marker="s", markersize=4,
                    linewidth=1.2)
        ax_bot.set_xlabel("u_max")
        ax_bot.set_ylabel("primary outcome")
        ax_bot.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    axes[0, 0].annotate("activation metric", (-0.25, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", fontsize=9)
    axes[1, 0].annotate("outcome", (-0.25, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", fontsize=9)

    fig.suptitle("Diagnostic-Strength Sweep\nActivation Metrics and Outcomes vs u_max",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir / "diagnostic_sweep.png")


# ---------------------------------------------------------------------------
# Figure 6: negative-control replay outcomes
# ---------------------------------------------------------------------------

def _fig_negative_control_outcomes(results_dir: Path, output_dir: Path) -> None:
    """Bar chart of mean activation diagnostic vs threshold for Phase III replay.

    Uses counterfactual_replay/ (NOT counterfactual_replay_4a2/) (spec §Q8).
    """
    neg_dir = _resolve_counterfactual_dir(results_dir)  # negative control (spec §Q8)
    if neg_dir is None:
        warnings.warn("No counterfactual_replay dir; skipping negative_control_outcomes")
        return

    tags: list[str] = []
    mean_abs_u: list[float] = []

    for task_dir in sorted(neg_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        summary_path = task_dir / "replay_summary.json"
        if not summary_path.exists():
            continue
        try:
            d = _load_json(summary_path)
            tags.append(task_dir.name)
            mean_abs_u.append(float(d.get("mean_abs_u", 0.0)))
        except Exception:
            pass

    if not tags:
        warnings.warn("No replay summaries found; skipping negative_control_outcomes")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(tags) * 1.2), 4))
    x_pos = np.arange(len(tags))
    ax.bar(x_pos, mean_abs_u, color="#2166ac", alpha=0.8, edgecolor="none")
    ax.axhline(5e-3, color="red", linestyle="--", linewidth=1.2,
               label="gate threshold (5e-3)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean |u| (activation diagnostic)")
    ax.set_title("Negative-Control Replay: Phase III Families\n(should be below gate threshold)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    _savefig(fig, output_dir / "negative_control_outcomes.png")


# ---------------------------------------------------------------------------
# Appendix A1: per-stage activation diagnostics
# ---------------------------------------------------------------------------

def _fig_per_stage_diagnostics(results_dir: Path, output_dir: Path) -> None:
    """Per-stage mean |natural_shift| and |delta_effective_discount| by task."""
    trans_dir = _resolve_translation_dir(results_dir)
    if trans_dir is None:
        return

    excluded = {"translation", "counterfactual_replay", "diagnostic_sweep",
                "analysis", "aggregated"}
    task_dirs = sorted(
        d for d in trans_dir.iterdir()
        if d.is_dir() and d.name not in excluded
    )
    if not task_dirs:
        return

    n = len(task_dirs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, task_dir in enumerate(task_dirs):
        ax = axes[0, i]
        ax.set_title(task_dir.name, fontsize=9)
        ax.set_xlabel("stage")
        ax.set_ylabel("mean |value|")

        # Try to read per-stage data from safe-nonlinear summary
        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            if _algo_class(algo_dir.name) != "safe-nonlinear":
                continue
            summary_path = algo_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                d = _load_json(summary_path)
                per_stage = d.get("per_stage", [])
                if not per_stage:
                    continue
                stages = [s["stage"] for s in per_stage]
                ns_means = [s.get("mean_abs_natural_shift", 0.0) for s in per_stage]
                dd_means = [s.get("mean_abs_delta_effective_discount", 0.0) for s in per_stage]
                ax.plot(stages, ns_means, marker=".", label="|natural_shift|",
                        color="#d6604d", linewidth=1)
                ax.plot(stages, dd_means, marker=".", label="|Δeff_discount|",
                        color="#74c476", linewidth=1)
                ax.legend(fontsize=6)
                break
            except Exception:
                pass

    fig.suptitle("Per-Stage Activation Diagnostics (safe-nonlinear)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, output_dir / "per_stage_diagnostics.png")


# ---------------------------------------------------------------------------
# Appendix A2: event-conditioned diagnostics
# ---------------------------------------------------------------------------

def _fig_event_conditioned(results_dir: Path, output_dir: Path) -> None:
    """Bar chart of event-conditioned vs global mean |u|."""
    report_dir = results_dir / "activation_report"
    event_path = report_dir / "event_conditioned_diagnostics.json"
    if not event_path.exists():
        event_path = results_dir.parent / "activation_report" / "event_conditioned_diagnostics.json"
    if not event_path.exists():
        warnings.warn("event_conditioned_diagnostics.json not found; skipping appendix A2")
        return

    try:
        data = _load_json(event_path)
    except Exception as exc:
        warnings.warn(f"Could not load event diagnostics: {exc}")
        return

    entries = data if isinstance(data, list) else data.get("tasks", [])
    if not entries:
        return

    tags = [e.get("tag", "") for e in entries]
    global_u = [float(e.get("mean_abs_u_global", e.get("mean_abs_u", 0.0))) for e in entries]
    event_u = [float(e.get("mean_abs_u_event", 0.0)) for e in entries]

    x_pos = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(max(6, len(tags) * 1.4), 4))
    ax.bar(x_pos - 0.2, global_u, width=0.38, color="#2166ac", alpha=0.8, label="global")
    ax.bar(x_pos + 0.2, event_u, width=0.38, color="#d6604d", alpha=0.8, label="event-conditioned")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean |u|")
    ax.set_title("Event-Conditioned vs Global Activation Diagnostics", fontsize=10)
    ax.legend(fontsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    _savefig(fig, output_dir / "event_conditioned_diagnostics.png")


# ---------------------------------------------------------------------------
# Appendix A3: null translation cases
# ---------------------------------------------------------------------------

def _fig_null_cases(results_dir: Path, output_dir: Path) -> None:
    """Plot CI bars for null translation cases (CI containing zero)."""
    null_path = results_dir / "analysis" / "null_translation_cases.json"
    if not null_path.exists():
        return

    try:
        nulls = _load_json(null_path)
    except Exception:
        return

    if not nulls:
        return

    n = len(nulls)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.4), 4))
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("outcome delta (safe-nonlinear minus classical)")
    ax.set_title("Null Translation Cases\n(CI contains zero)", fontsize=10)

    for j, case in enumerate(nulls):
        md = case.get("mean_diff", 0.0)
        lo = case.get("ci_lower", md)
        hi = case.get("ci_upper", md)
        ax.bar(j, md, color="#d6604d", alpha=0.5, width=0.55)
        _ci_bar(ax, j, lo, hi, color="black", solid_capstyle="round")

    ax.set_xticks(range(n))
    ax.set_xticklabels([c.get("tag", f"task_{j}") for j, c in enumerate(nulls)],
                       rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_dir / "null_translation_cases.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Generate Phase IV-B figures."""
    results_dir = args.results_dir.resolve()
    # Default output-dir = <results-dir>/analysis/ (spec §Q12)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase IV-B figure generation")
    print(f"  results_dir: {results_dir}")
    print(f"  output_dir:  {output_dir}")
    print()

    # Main figures (spec §12.1)
    print("Main figures:")
    _fig_rl_learning_curves(results_dir, output_dir)
    _fig_tail_risk_outcomes(results_dir, output_dir)
    _fig_scatter_outcome_vs_diagnostic(results_dir, output_dir)
    _fig_matched_control(results_dir, output_dir)
    _fig_diagnostic_sweep(results_dir, output_dir)
    _fig_negative_control_outcomes(results_dir, output_dir)

    # Appendix figures (spec §12.2)
    print("\nAppendix figures:")
    _fig_per_stage_diagnostics(results_dir, output_dir)
    _fig_event_conditioned(results_dir, output_dir)
    _fig_null_cases(results_dir, output_dir)

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", type=Path, required=True,
        help="Top-level Phase IV-B results directory"
    )
    # No --config (spec §Q1: figures read data directly)
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for figures; defaults to <results-dir>/analysis/"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
