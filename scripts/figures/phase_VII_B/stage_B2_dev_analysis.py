"""Phase VII-B Stage B2-Dev promotion analysis and figures.

Spec authority:
    tasks/phase_VII_B_strategic_learning_coding_agent_spec.md §§11.1, 11.2, 15.

Pipeline:
    1. Load run summaries from results/adaptive_beta/strategic/raw/dev/.
       Deduplicate by (game, adversary, method, seed_id) keeping the latest
       started_at; verify post-filter count == 180.
    2. Compute per-cell summaries to per_run_summary.parquet.
    3. Compute paired-bootstrap diffs to paired_diffs.parquet.
    4. Apply promotion gate (strict + directional fallback).
    5. Generate plots (learning curves, AUC paired diff, mechanism scatter,
       event-aligned panels for hypothesis-testing cells).
    6. Write the Stage B2-Dev summary memo.

Bootstrap: 10,000 paired resamples; percentile CIs (n=3 seeds is too small
for BCa). Caveat: CIs are wide at n=3.

Determinism: bootstrap seed = 0xB2DEF (mirrors aggregate.py).

Run from repo root:
    python scripts/figures/phase_VII_B/stage_B2_dev_analysis.py
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # NeurIPS Type 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["figure.dpi"] = 110

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/liq/Documents/Claude/Projects/LSE_RL")
RAW_ROOT = REPO_ROOT / "results/adaptive_beta/strategic/raw/dev"
MANIFEST_PATH = RAW_ROOT / "manifest.json"
PROC_DIR = REPO_ROOT / "results/adaptive_beta/strategic/processed/dev"
FIG_DIR = REPO_ROOT / "results/adaptive_beta/strategic/figures/dev"
SUMMARY_MEMO = REPO_ROOT / "results/adaptive_beta/strategic/stage_B2_dev_summary.md"
NEG_REPORT = REPO_ROOT / "results/adaptive_beta/strategic/negative_result_report.md"
NO_UPDATE_MEMO = REPO_ROOT / "paper_update/no_update_recommendation.md"
STAGE_B2_MAIN_CFG = REPO_ROOT / "experiments/adaptive_beta/strategic_games/configs/stage_B2_main.yaml"

GAMMA = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0xB2DEF  # mirrors aggregate.py

# Spec §6.1: matching_pennies horizon=1 is mechanism-degenerate.
DEGENERATE_GAMES = {"matching_pennies"}

# Method labelling: data uses "adaptive_beta" (== spec "adaptive_beta_clipped").
CANDIDATE_METHOD = "adaptive_beta"
BASELINE_METHOD = "vanilla"
ALL_METHODS = (
    "vanilla",
    "fixed_positive",
    "fixed_negative",
    "adaptive_beta",
    "adaptive_sign_only",
)
NON_VANILLA_METHODS = tuple(m for m in ALL_METHODS if m != BASELINE_METHOD)

# ----------------------------------------------------------------------------
# Step 1: load + dedupe
# ----------------------------------------------------------------------------


def load_manifest_dedup() -> pd.DataFrame:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest["records"]
    df = pd.DataFrame(records)
    n_before = len(df)
    # Filter to n_episodes == 1000 (drops the smoke 5-ep duplicate). Tiebreak by
    # latest started_at if any genuine duplicates remain.
    df = df[df["n_episodes"] == 1000]
    # Ensure uniqueness by (game, adversary, method, seed_id) keeping latest started_at.
    df = df.sort_values("started_at").drop_duplicates(
        subset=["game", "adversary", "method", "seed_id"], keep="last"
    )
    n_after = len(df)
    print(f"[manifest] {n_before} → {n_after} after dedup/filter (expect 180)")
    if n_after != 180:
        raise RuntimeError(
            f"post-filter count is {n_after}, expected 180. "
            f"Cannot proceed without a clean run matrix."
        )
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Step 2: per-cell summary
# ----------------------------------------------------------------------------


def _safe_arr(npz, key) -> np.ndarray:
    if key not in npz.files:
        return np.zeros(0)
    a = np.asarray(npz[key]).reshape(-1)
    return a


def _nanmean(a: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    if a.dtype == bool:
        return float(a.mean())
    finite = np.isfinite(a)
    if not finite.any():
        return float("nan")
    return float(a[finite].mean())


def summarize_run(record: Dict[str, Any]) -> Dict[str, Any]:
    raw_dir = REPO_ROOT / record["raw_dir"]
    npz = np.load(raw_dir / "metrics.npz", allow_pickle=False)
    returns = _safe_arr(npz, "return")
    if returns.size != 1000:
        raise RuntimeError(
            f"unexpected returns length {returns.size} for {record['cell_id']}"
        )
    final_return = float(returns[-100:].mean())
    auc_return = float(returns.sum())

    align = _safe_arr(npz, "alignment_rate")
    d_eff = _safe_arr(npz, "mean_effective_discount")
    if d_eff.size == 0:
        d_eff = _safe_arr(npz, "mean_d_eff")
    diverged = _safe_arr(npz, "diverged")
    if diverged.size == 0:
        diverged = _safe_arr(npz, "divergence_event")
    nan_count = _safe_arr(npz, "nan_count")
    opp_ent = _safe_arr(npz, "opponent_policy_entropy")
    pol_tv = _safe_arr(npz, "policy_total_variation")
    support_shift = _safe_arr(npz, "support_shift")
    model_rejected = _safe_arr(npz, "model_rejected")
    search_phase = _safe_arr(npz, "search_phase")

    return {
        "run_id": record["run_id"],
        "cell_id": record["cell_id"],
        "game": record["game"],
        "adversary": record["adversary"],
        "method": record["method"],
        "seed_id": int(record["seed_id"]),
        "n_episodes": int(record["n_episodes"]),
        "final_return": final_return,
        "auc_return": auc_return,
        "mean_alignment_rate": _nanmean(align),
        "mean_effective_discount": _nanmean(d_eff),
        "total_diverged": int((diverged > 0).sum()) if diverged.size else 0,
        "total_nan": int(nan_count.sum()) if nan_count.size else 0,
        "mean_opponent_policy_entropy": _nanmean(opp_ent),
        "mean_policy_total_variation": _nanmean(pol_tv),
        "support_shift_count": int((support_shift > 0).sum())
        if support_shift.size
        else 0,
        "model_rejection_count": (
            int((model_rejected > 0).sum())
            if record["adversary"] == "hypothesis_testing" and model_rejected.size
            else float("nan")
        ),
        "search_phase_episodes": (
            int((search_phase > 0).sum())
            if record["adversary"] == "hypothesis_testing" and search_phase.size
            else float("nan")
        ),
        "raw_dir": str(raw_dir),
    }


def build_per_run_summary(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_run(r) for _, r in manifest_df.iterrows()]
    df = pd.DataFrame(rows)
    df = df.sort_values(["game", "adversary", "method", "seed_id"]).reset_index(
        drop=True
    )
    return df


# ----------------------------------------------------------------------------
# Step 3: paired bootstrap diffs
# ----------------------------------------------------------------------------


def _percentile_paired_bootstrap(
    diffs: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Returns (mean, ci_lo, ci_hi, se) by paired-bootstrap percentile CI."""
    arr = np.asarray(diffs, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean_d = float(finite.mean())
    if finite.size < 2:
        return (mean_d, float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, finite.size, size=(n_resamples, finite.size))
    boot_means = finite[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    se = float(boot_means.std(ddof=1))
    return (mean_d, lo, hi, se)


def compute_recovery_time(returns: np.ndarray, threshold_frac: float = 0.5) -> float:
    """Recovery time proxy: episodes until rolling-100 mean exceeds
    threshold_frac * (max - min) + min for the first time. NaN if never reaches.
    Used here so we have a recovery diff metric per spec §11.1.
    """
    if returns.size < 100:
        return float("nan")
    rolling = pd.Series(returns).rolling(100, min_periods=1).mean().to_numpy()
    lo = float(np.nanmin(rolling))
    hi = float(np.nanmax(rolling))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return float("nan")
    threshold = lo + threshold_frac * (hi - lo)
    above = np.where(rolling >= threshold)[0]
    if above.size == 0:
        return float("nan")
    return float(above[0])


def add_recovery_time_column(
    summary_df: pd.DataFrame, manifest_df: pd.DataFrame
) -> pd.DataFrame:
    rec = []
    for _, row in summary_df.iterrows():
        raw_dir = Path(row["raw_dir"])
        npz = np.load(raw_dir / "metrics.npz", allow_pickle=False)
        rec.append(compute_recovery_time(_safe_arr(npz, "return")))
    summary_df = summary_df.copy()
    summary_df["recovery_time"] = rec
    return summary_df


def compute_paired_diffs(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (game, adv), gdf in summary_df.groupby(["game", "adversary"]):
        baseline_df = gdf[gdf["method"] == BASELINE_METHOD].set_index("seed_id")
        if baseline_df.empty:
            continue
        for method in NON_VANILLA_METHODS:
            mdf = gdf[gdf["method"] == method].set_index("seed_id")
            if mdf.empty:
                continue
            seeds = sorted(set(baseline_df.index).intersection(mdf.index))
            if not seeds:
                continue
            for metric_name, dst_prefix in [
                ("auc_return", "auc"),
                ("final_return", "final_return"),
                ("recovery_time", "recovery_time"),
            ]:
                base = baseline_df.loc[seeds, metric_name].to_numpy(dtype=float)
                meth = mdf.loc[seeds, metric_name].to_numpy(dtype=float)
                diffs = meth - base
                mean_d, lo, hi, se = _percentile_paired_bootstrap(
                    diffs,
                    n_resamples=BOOTSTRAP_RESAMPLES,
                    seed=BOOTSTRAP_SEED,
                )
                rows.append(
                    {
                        "game": game,
                        "adversary": adv,
                        "method": method,
                        "metric": metric_name,
                        "metric_label": dst_prefix,
                        "n_seeds": len(seeds),
                        "diff_values": ",".join(f"{d:.6g}" for d in diffs),
                        "mean": mean_d,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "se": se,
                        "ci_excludes_zero": bool(
                            np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
                        ),
                        "ci_excludes_zero_positive": bool(
                            np.isfinite(lo) and lo > 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Step 4: promotion gate
# ----------------------------------------------------------------------------


@dataclass
class GateOutcome:
    game: str
    adversary: str
    auc_diff_mean: float
    auc_diff_ci_lo: float
    auc_diff_ci_hi: float
    auc_diff_se: float
    diverged_count: int
    mean_alignment_rate: float
    mean_effective_discount: float
    is_degenerate: bool
    pass_strict: bool
    pass_directional: bool
    promoted: bool
    gate_mode: str  # "strict" | "directional" | "fail"
    notes: str


def evaluate_promotion(
    summary_df: pd.DataFrame, paired_df: pd.DataFrame
) -> List[GateOutcome]:
    outcomes: List[GateOutcome] = []
    auc_df = paired_df[
        (paired_df["method"] == CANDIDATE_METHOD) & (paired_df["metric"] == "auc_return")
    ]
    for (game, adv), grp in summary_df.groupby(["game", "adversary"]):
        # Get adaptive_beta paired AUC diff row for this cell.
        row = auc_df[(auc_df["game"] == game) & (auc_df["adversary"] == adv)]
        if row.empty:
            continue
        r = row.iloc[0]
        # Diagnostics from candidate method runs.
        cand = grp[grp["method"] == CANDIDATE_METHOD]
        diverged_count = int(cand["total_diverged"].fillna(0).sum())
        mean_align = float(cand["mean_alignment_rate"].mean())
        mean_d_eff = float(cand["mean_effective_discount"].mean())
        is_degen = game in DEGENERATE_GAMES

        mean_diff = float(r["mean"])
        ci_lo = float(r["ci_lo"])
        ci_hi = float(r["ci_hi"])
        se = float(r["se"])

        # Strict gate: mean > 0 AND ci_lo > 0 AND no clipped divergence.
        cond_strict_signal = (mean_diff > 0) and np.isfinite(ci_lo) and (ci_lo > 0)
        cond_no_div = diverged_count == 0
        pass_strict = bool(cond_strict_signal and cond_no_div)

        # Directional fallback: mean > 0 AND no divergence AND mechanism evidence
        # supports it on non-degenerate games.
        # Per spec §11.1(c): degenerate games can still pass strict standalone.
        if is_degen:
            mech_ok = pass_strict  # degenerate games must clear strict.
        else:
            mech_ok = (
                (mean_align > 0.5) or (mean_d_eff < GAMMA)
            ) and np.isfinite(mean_align) and np.isfinite(mean_d_eff)
        pass_directional = bool(
            (mean_diff > 0) and cond_no_div and mech_ok and not pass_strict
        )

        if pass_strict:
            promoted = True
            gate_mode = "strict"
        elif pass_directional:
            promoted = True
            gate_mode = "directional"
        else:
            promoted = False
            gate_mode = "fail"

        # Reason notes.
        notes_parts = []
        if not cond_no_div:
            notes_parts.append(f"adaptive_beta diverged in {diverged_count} eps")
        if mean_diff <= 0:
            notes_parts.append(f"mean AUC diff non-positive ({mean_diff:.2f})")
        if not pass_strict and is_degen:
            notes_parts.append("degenerate game requires strict CI")
        if not pass_strict and not is_degen and not pass_directional:
            notes_parts.append("mechanism evidence insufficient")
        notes = "; ".join(notes_parts) or "ok"

        outcomes.append(
            GateOutcome(
                game=game,
                adversary=adv,
                auc_diff_mean=mean_diff,
                auc_diff_ci_lo=ci_lo,
                auc_diff_ci_hi=ci_hi,
                auc_diff_se=se,
                diverged_count=diverged_count,
                mean_alignment_rate=mean_align,
                mean_effective_discount=mean_d_eff,
                is_degenerate=is_degen,
                pass_strict=pass_strict,
                pass_directional=pass_directional,
                promoted=promoted,
                gate_mode=gate_mode,
                notes=notes,
            )
        )
    return outcomes


def select_top_promoted(
    outcomes: List[GateOutcome], k: int = 3
) -> List[GateOutcome]:
    promoted = [o for o in outcomes if o.promoted]
    # Rank by z-score-style |mean| / se (use mean/se since direction is positive).
    def _zscore(o: GateOutcome) -> float:
        if not np.isfinite(o.auc_diff_se) or o.auc_diff_se == 0:
            return -np.inf if not np.isfinite(o.auc_diff_mean) else np.inf
        return o.auc_diff_mean / o.auc_diff_se

    def _mech_distance(o: GateOutcome) -> float:
        # Larger distance from 0.5 (alignment) is stronger mechanism signal.
        if not np.isfinite(o.mean_alignment_rate):
            return -np.inf
        return abs(o.mean_alignment_rate - 0.5)

    promoted.sort(key=lambda o: (_zscore(o), _mech_distance(o)), reverse=True)
    return promoted[:k]


def filter_for_main_dispatch(
    promoted: List[GateOutcome],
) -> Tuple[List[GateOutcome], List[str]]:
    """Filter the promoted list so the (games × adversaries) Cartesian product
    used by the Stage B2-Main runner only spans promoted cells.

    Stage B2-Main runner takes the Cartesian product of `games` and
    `adversaries` (see run_strategic.iter_cells). If our promoted cells form
    an irregular subset, the Cartesian product would include non-promoted
    cells. To stay within spec §11.2's "1–3 promoted pairs" budget AND keep
    the Cartesian product tight, we drop any promoted cell whose inclusion
    would force a non-promoted Cartesian neighbour, preferring the
    higher-z-score remaining cells.

    Returns:
        kept: filtered list of GateOutcome to dispatch.
        notes: human-readable explanations for any drops.
    """
    if len(promoted) <= 1:
        return promoted, []
    promoted_set = {(o.game, o.adversary) for o in promoted}

    def _cartesian_clean(subset: List[GateOutcome]) -> bool:
        games = sorted({o.game for o in subset})
        advs = sorted({o.adversary for o in subset})
        for g in games:
            for a in advs:
                if (g, a) not in promoted_set:
                    return False
        return True

    # Try the full set first.
    if _cartesian_clean(promoted):
        return promoted, []

    # Drop greedily from lowest z-score until clean.
    def _zscore(o: GateOutcome) -> float:
        if not np.isfinite(o.auc_diff_se) or o.auc_diff_se == 0:
            return -np.inf
        return o.auc_diff_mean / o.auc_diff_se

    sorted_lo_to_hi = sorted(promoted, key=_zscore)
    notes: List[str] = []
    kept = list(promoted)
    for victim in sorted_lo_to_hi:
        if _cartesian_clean(kept):
            break
        kept.remove(victim)
        notes.append(
            f"dropped ({victim.game}, {victim.adversary}) — "
            f"keeping it would force a non-promoted Cartesian neighbour "
            f"into Stage B2-Main; z={_zscore(victim):+.2f}"
        )
    return kept, notes


# ----------------------------------------------------------------------------
# Step 5: plots
# ----------------------------------------------------------------------------

METHOD_COLORS = {
    "vanilla": "#444444",
    "fixed_positive": "#1f77b4",
    "fixed_negative": "#d62728",
    "adaptive_beta": "#2ca02c",
    "adaptive_sign_only": "#9467bd",
}
METHOD_LABELS = {
    "vanilla": "vanilla",
    "fixed_positive": "fixed-β (+)",
    "fixed_negative": "fixed-β (−)",
    "adaptive_beta": "adaptive-β (clip)",
    "adaptive_sign_only": "adaptive sign-only",
}


def plot_learning_curves(
    summary_df: pd.DataFrame, manifest_df: pd.DataFrame
) -> Path:
    games = sorted(summary_df["game"].unique())
    advs = sorted(summary_df["adversary"].unique())
    n_games, n_advs = len(games), len(advs)
    fig, axes = plt.subplots(
        n_games, n_advs, figsize=(4.0 * n_advs, 2.4 * n_games), sharex=True
    )
    # Pre-load returns by (game, adv, method, seed)
    returns_cache: Dict[Tuple[str, str, str, int], np.ndarray] = {}
    for _, row in summary_df.iterrows():
        npz = np.load(Path(row["raw_dir"]) / "metrics.npz", allow_pickle=False)
        returns_cache[
            (row["game"], row["adversary"], row["method"], int(row["seed_id"]))
        ] = _safe_arr(npz, "return")

    for i, g in enumerate(games):
        for j, a in enumerate(advs):
            ax = axes[i, j] if n_games > 1 else axes[j]
            for m in ALL_METHODS:
                seed_curves = []
                for s in (0, 1, 2):
                    key = (g, a, m, s)
                    if key in returns_cache:
                        # Smooth via 50-ep rolling mean for readability.
                        r = returns_cache[key]
                        rolling = (
                            pd.Series(r).rolling(50, min_periods=1).mean().to_numpy()
                        )
                        seed_curves.append(rolling)
                if not seed_curves:
                    continue
                arr = np.stack(seed_curves)  # (3, 1000)
                mean = arr.mean(axis=0)
                se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                x = np.arange(arr.shape[1])
                ax.plot(x, mean, color=METHOD_COLORS[m], label=METHOD_LABELS[m], lw=1.3)
                ax.fill_between(
                    x, mean - se, mean + se, color=METHOD_COLORS[m], alpha=0.15, lw=0
                )
            ax.set_title(f"{g} | {a}", fontsize=8)
            if i == n_games - 1:
                ax.set_xlabel("episode")
            if j == 0:
                ax.set_ylabel("return (50-ep MA)")
            ax.tick_params(labelsize=7)
    # Single shared legend.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(ALL_METHODS),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_pdf = FIG_DIR / "learning_curves_all.pdf"
    out_png = FIG_DIR / "learning_curves_all.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_pdf


def plot_auc_paired_diff(paired_df: pd.DataFrame) -> Path:
    sub = paired_df[
        (paired_df["method"] == CANDIDATE_METHOD)
        & (paired_df["metric"] == "auc_return")
    ].copy()
    adv_short = {
        "finite_memory_best_response": "FM-BR",
        "finite_memory_regret_matching": "FM-RM",
        "hypothesis_testing": "HypTest",
    }
    sub["cell_label"] = (
        sub["game"].str.replace("_", " ") + "\n" + sub["adversary"].map(adv_short)
    )
    sub = sub.sort_values(["game", "adversary"]).reset_index(drop=True)
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    err_lo = sub["mean"] - sub["ci_lo"]
    err_hi = sub["ci_hi"] - sub["mean"]
    bar_colors = [
        "#2ca02c" if (lo > 0 and m > 0) else ("#d6a417" if m > 0 else "#888888")
        for lo, m in zip(sub["ci_lo"], sub["mean"])
    ]
    ax.bar(x, sub["mean"], color=bar_colors, edgecolor="black", lw=0.5)
    ax.errorbar(
        x,
        sub["mean"],
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1,
    )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["cell_label"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC diff (adaptive_beta − vanilla)\nmean ± 95% paired bootstrap CI")
    ax.set_title(
        "Stage B2-Dev: adaptive-β AUC paired diff vs vanilla (n=3 seeds; CIs are wide)"
    )
    fig.tight_layout()
    out_pdf = FIG_DIR / "auc_paired_diff_dev.pdf"
    out_png = FIG_DIR / "auc_paired_diff_dev.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_pdf


def plot_mechanism(summary_df: pd.DataFrame) -> Path:
    cand = summary_df[summary_df["method"] == CANDIDATE_METHOD].copy()
    # Aggregate over seeds.
    agg = cand.groupby(["game", "adversary"]).agg(
        align=("mean_alignment_rate", "mean"),
        d_eff=("mean_effective_discount", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("tab10")
    color_index = {
        cell: cmap(i % 10)
        for i, cell in enumerate(
            sorted(agg["game"].unique())
        )
    }
    markers = {"finite_memory_best_response": "o", "finite_memory_regret_matching": "s", "hypothesis_testing": "^"}
    for _, r in agg.iterrows():
        ax.scatter(
            r["d_eff"],
            r["align"],
            s=120,
            c=[color_index[r["game"]]],
            marker=markers.get(r["adversary"], "x"),
            edgecolor="black",
            lw=0.6,
            label=f"{r['game']} | {r['adversary']}",
        )
        ax.annotate(
            f"{r['game']}\n{r['adversary']}",
            (r["d_eff"], r["align"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=6,
        )
    ax.axhline(0.5, color="grey", lw=0.6, ls="--", label="alignment=0.5")
    ax.axvline(GAMMA, color="grey", lw=0.6, ls=":", label=f"d_eff=γ={GAMMA}")
    ax.set_xlabel("mean effective discount d_eff (adaptive_beta)")
    ax.set_ylabel("mean alignment rate (adaptive_beta)")
    ax.set_title("Stage B2-Dev mechanism diagnostics (per cell, mean over 3 seeds)")
    fig.tight_layout()
    out_pdf = FIG_DIR / "mechanism_dev.pdf"
    out_png = FIG_DIR / "mechanism_dev.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_pdf


def plot_event_aligned_hypothesis(
    summary_df: pd.DataFrame,
) -> List[Path]:
    """For (rules_of_road, hypothesis_testing) and (strategic_rps, hypothesis_testing)
    cells, build event-aligned panels around model_rejected events.
    """
    out_paths: List[Path] = []
    half = 50
    for game in ("rules_of_road", "strategic_rps"):
        cand = summary_df[
            (summary_df["game"] == game)
            & (summary_df["adversary"] == "hypothesis_testing")
            & (summary_df["method"] == CANDIDATE_METHOD)
        ]
        if cand.empty:
            continue
        # Stack windows around model_rejected events.
        stacks = {
            "return": [],
            "beta_deployed": [],
            "mean_effective_discount": [],
            "alignment_rate": [],
            "opponent_policy_entropy": [],
        }
        n_events = 0
        for _, row in cand.iterrows():
            npz = np.load(Path(row["raw_dir"]) / "metrics.npz", allow_pickle=False)
            mr = _safe_arr(npz, "model_rejected").astype(bool)
            events = np.flatnonzero(mr)
            if events.size == 0:
                continue
            for k in stacks:
                arr = _safe_arr(npz, k)
                if arr.size == 0:
                    continue
                for e in events:
                    lo = e - half
                    hi = e + half + 1
                    pad_lo = max(0, -lo)
                    pad_hi = max(0, hi - arr.size)
                    sl = arr[max(lo, 0) : min(hi, arr.size)]
                    if pad_lo or pad_hi:
                        sl = np.concatenate(
                            [
                                np.full(pad_lo, np.nan),
                                sl,
                                np.full(pad_hi, np.nan),
                            ]
                        )
                    if sl.shape[0] != 2 * half + 1:
                        continue
                    stacks[k].append(sl)
            n_events += events.size
        if n_events == 0:
            continue
        # Plot.
        fig, axes = plt.subplots(5, 1, figsize=(7.5, 9), sharex=True)
        x = np.arange(-half, half + 1)
        for ax, key, lbl in zip(
            axes,
            ("return", "beta_deployed", "mean_effective_discount", "alignment_rate", "opponent_policy_entropy"),
            ("return", "β_deployed", "d_eff", "alignment_rate", "opp. policy entropy"),
        ):
            if not stacks[key]:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
                ax.set_ylabel(lbl)
                continue
            arr = np.stack(stacks[key])  # (n_events, 2*half+1)
            mean = np.nanmean(arr, axis=0)
            sd = np.nanstd(arr, axis=0)
            se = sd / np.sqrt(np.maximum(np.sum(~np.isnan(arr), axis=0), 1))
            ax.plot(x, mean, color="#2ca02c", lw=1.3)
            ax.fill_between(x, mean - se, mean + se, color="#2ca02c", alpha=0.2, lw=0)
            ax.axvline(0, color="red", lw=0.6, ls="--")
            ax.set_ylabel(lbl, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[-1].set_xlabel("episodes from model_rejected event")
        fig.suptitle(
            f"{game} | hypothesis_testing | adaptive_beta — event-aligned (n={n_events} events)",
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_pdf = FIG_DIR / f"event_aligned_{game}_hypothesis_testing.pdf"
        out_png = FIG_DIR / f"event_aligned_{game}_hypothesis_testing.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        out_paths.append(out_pdf)
    return out_paths


# ----------------------------------------------------------------------------
# Step 6: memo
# ----------------------------------------------------------------------------


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    if not np.isfinite(lo):
        return f"{mean:.2f} (CI n/a)"
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def write_summary_memo(
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    outcomes: List[GateOutcome],
    promoted_all: List[GateOutcome],
    promoted: List[GateOutcome],
    drop_notes: List[str],
    fig_paths: Dict[str, Any],
) -> None:
    lines: List[str] = []
    any_strict = any(o.pass_strict for o in outcomes)
    any_directional = any(o.gate_mode == "directional" for o in outcomes)
    fully_failed = not any(o.promoted for o in outcomes)
    verdict = (
        "PROMOTE" if not fully_failed else "NO PROMOTE"
    )
    lines.append("# Phase VII-B Stage B2-Dev — Promotion Memo")
    lines.append("")
    lines.append(f"**Branch:** `phase-VII-B-strategic-2026-04-26`")
    lines.append(f"**Verdict:** **{verdict}**")
    if not fully_failed:
        gate_names = [
            f"({o.game}, {o.adversary}, mode={o.gate_mode})" for o in promoted_all
        ]
        lines.append(
            f"**Gate-promoted cells ({len(promoted_all)}/3):** "
            + ", ".join(gate_names)
        )
        main_names = [
            f"({o.game}, {o.adversary}, mode={o.gate_mode})" for o in promoted
        ]
        lines.append(
            f"**Stage B2-Main dispatch cells ({len(promoted)}):** "
            + ", ".join(main_names)
        )
        if drop_notes:
            lines.append("")
            lines.append("**Cartesian-product filter notes** (Stage B2-Main runner")
            lines.append("takes the Cartesian product of `games` × `adversaries`; cells")
            lines.append("dropped to keep that product strictly inside the gate-promoted set):")
            for n in drop_notes:
                lines.append(f"- {n}")
    else:
        lines.append(
            "**Reason:** No `(game, adversary)` cell cleared either the strict "
            "or directional promotion gate. See per-cell table below."
        )
        lines.append(
            "**Recommendation pointer:** see "
            "`paper_update/no_update_recommendation.md`."
        )
    lines.append("")
    lines.append("## 1. Run Matrix")
    lines.append("")
    lines.append("- Expected: 4 games × 3 adversaries × 5 methods × 3 seeds = **180 cells**.")
    lines.append("- Manifest contained 181 records; 1 stale `n_episodes=5` smoke")
    lines.append("  duplicate filtered out by deduplication on")
    lines.append("  `(game, adversary, method, seed_id)` keeping latest `started_at`.")
    lines.append(f"- Post-filter count: **{len(summary_df)}** runs (matches expectation).")
    lines.append("")
    # Anomalies.
    div_runs = summary_df[summary_df["total_diverged"] > 0]
    nan_runs = summary_df[summary_df["total_nan"] > 0]
    if div_runs.empty and nan_runs.empty:
        lines.append("No divergence or NaN events recorded across all 180 runs.")
    else:
        if not div_runs.empty:
            lines.append(f"Runs with divergence events: {len(div_runs)}")
        if not nan_runs.empty:
            lines.append(f"Runs with NaN events: {len(nan_runs)}")
    lines.append("")

    lines.append("## 2. Per-(game, adversary) AUC Paired Diff vs Vanilla")
    lines.append("")
    lines.append(
        "Paired bootstrap, 10,000 resamples, percentile CIs, paired by `seed_id` (common_env_seed)."
    )
    lines.append("With n=3 seeds CIs are intentionally wide.")
    lines.append("")
    lines.append(
        "| game | adversary | method | mean Δ AUC | 95% CI | CI excl. 0+ | gate (mech) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for (game, adv), gdf in summary_df.groupby(["game", "adversary"]):
        for m in NON_VANILLA_METHODS:
            row = paired_df[
                (paired_df["game"] == game)
                & (paired_df["adversary"] == adv)
                & (paired_df["method"] == m)
                & (paired_df["metric"] == "auc_return")
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            ci = fmt_ci(r["mean"], r["ci_lo"], r["ci_hi"])
            ci_excl = "yes" if (np.isfinite(r["ci_lo"]) and r["ci_lo"] > 0) else "no"
            mech = ""
            if m == CANDIDATE_METHOD:
                out = next(
                    (o for o in outcomes if o.game == game and o.adversary == adv),
                    None,
                )
                if out is not None:
                    mech = (
                        f"{out.gate_mode} (align={out.mean_alignment_rate:.2f}, "
                        f"d_eff={out.mean_effective_discount:.2f})"
                    )
            lines.append(
                f"| {game} | {adv} | {m} | {r['mean']:+.2f} | {ci} | {ci_excl} | {mech} |"
            )
    lines.append("")

    lines.append("## 3. Promotion Gate Outcomes (adaptive_beta vs vanilla)")
    lines.append("")
    lines.append(
        "| game | adversary | mean Δ AUC | CI lo | CI hi | div | align | d_eff | "
        "degen | strict | dir | promoted | mode | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for o in outcomes:
        lines.append(
            f"| {o.game} | {o.adversary} | {o.auc_diff_mean:+.2f} | "
            f"{o.auc_diff_ci_lo:+.2f} | {o.auc_diff_ci_hi:+.2f} | "
            f"{o.diverged_count} | {o.mean_alignment_rate:.2f} | "
            f"{o.mean_effective_discount:.2f} | "
            f"{'yes' if o.is_degenerate else 'no'} | "
            f"{'yes' if o.pass_strict else 'no'} | "
            f"{'yes' if o.pass_directional else 'no'} | "
            f"{'yes' if o.promoted else 'no'} | {o.gate_mode} | {o.notes} |"
        )
    lines.append("")

    lines.append("## 4. Mechanism Diagnostics (adaptive_beta, mean over 3 seeds)")
    lines.append("")
    lines.append("| game | adversary | align | d_eff | opp_entropy | pol_TV | support_shifts |")
    lines.append("|---|---|---|---|---|---|---|")
    cand = summary_df[summary_df["method"] == CANDIDATE_METHOD]
    for (game, adv), grp in cand.groupby(["game", "adversary"]):
        lines.append(
            f"| {game} | {adv} | {grp['mean_alignment_rate'].mean():.2f} | "
            f"{grp['mean_effective_discount'].mean():.2f} | "
            f"{grp['mean_opponent_policy_entropy'].mean():.2f} | "
            f"{grp['mean_policy_total_variation'].mean():.3f} | "
            f"{int(grp['support_shift_count'].sum())} |"
        )
    lines.append("")

    lines.append("## 5. Failure / Divergence Accounting")
    lines.append("")
    if div_runs.empty:
        lines.append("- **Zero clipped-`adaptive_beta` divergence events** across all 60 candidate-method runs.")
    else:
        lines.append("- Divergence events detected:")
        for _, r in div_runs.iterrows():
            lines.append(
                f"  - {r['cell_id']}: {int(r['total_diverged'])} divergence flags"
            )
    lines.append(
        f"- Total NaN counts across all runs: {int(summary_df['total_nan'].sum())}"
    )
    lines.append("")

    lines.append("## 6. Selection-Bias Acknowledgement")
    lines.append("")
    lines.append(
        "With only 3 seeds, the strict `ci_lo > 0` gate is conservative; the wide "
        "bootstrap distribution at n=3 means the strict criterion is hard to clear "
        "even for genuinely-favorable cells. Use the directional fallback only for "
        "cells where the mean diff is positive AND the mechanism evidence "
        "(alignment > 0.5 or d_eff < γ) directionally supports the interpretation. "
        "matching_pennies is mechanism-degenerate (horizon=1 per spec §6.1) and "
        "MUST clear the strict CI gate to promote — it cannot rely on directional "
        "fallback because alignment / d_eff are uninformative on horizon-1 games."
    )
    lines.append("")
    if not fully_failed:
        lines.append("Cells that cleared each gate:")
        strict_pass = [f"({o.game}, {o.adversary})" for o in outcomes if o.pass_strict]
        directional_pass = [f"({o.game}, {o.adversary})" for o in outcomes if o.gate_mode == "directional"]
        lines.append(f"- **Strict** (`mean>0` AND `ci_lo>0` AND no div): {strict_pass or 'none'}")
        lines.append(f"- **Directional only** (mean>0 AND mech-OK AND no div, but CI does not exclude 0): {directional_pass or 'none'}")
    lines.append("")

    lines.append("## 7. Stage B2-Main Dispatch (if PROMOTE)")
    lines.append("")
    if promoted:
        games_list = sorted({o.game for o in promoted})
        advs_list = sorted({o.adversary for o in promoted})
        lines.append("Promoted games: " + ", ".join(games_list))
        lines.append("")
        lines.append("Promoted adversaries: " + ", ".join(advs_list))
        lines.append("")
        lines.append("Per-cell:")
        for o in promoted:
            lines.append(
                f"- ({o.game}, {o.adversary}) — gate_mode={o.gate_mode}, "
                f"Δ AUC = {fmt_ci(o.auc_diff_mean, o.auc_diff_ci_lo, o.auc_diff_ci_hi)}"
            )
        lines.append("")
        lines.append(
            "Stage B2-Main matrix per spec §11.2: 10,000 episodes × 10 seeds × "
            "5 methods (vanilla, fixed_positive, fixed_negative, adaptive_beta, "
            "adaptive_sign_only)."
        )
        n_dispatch = len(promoted) * 10 * 5
        lines.append(f"Total dispatch cells: {n_dispatch}.")
    else:
        lines.append("No dispatch — gate fully fails.")
    lines.append("")

    lines.append("## 8. Generated Figures")
    lines.append("")
    for k, v in fig_paths.items():
        if isinstance(v, list):
            for p in v:
                lines.append(f"- {k}: `{p.as_posix()}`")
        else:
            lines.append(f"- {k}: `{v.as_posix()}`")
    lines.append("")

    lines.append("## 9. Methodological Caveats")
    lines.append("")
    lines.append(
        "- n=3 seeds: bootstrap CIs are very wide. Use directional fallback "
        "with care; final claim strength must come from Stage B2-Main n=10."
    )
    lines.append(
        "- matching_pennies horizon=1 is mechanism-degenerate per spec §6.1; "
        "alignment_rate / d_eff cannot serve as mechanism evidence there."
    )
    lines.append(
        "- Bootstrap method: percentile (BCa not feasible at n=3). Reported with "
        "fixed seed = 0xB2DEF for byte-stable reproduction."
    )
    lines.append("")

    SUMMARY_MEMO.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MEMO.write_text("\n".join(lines), encoding="utf-8")


def write_negative_report(
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    outcomes: List[GateOutcome],
) -> None:
    lines: List[str] = []
    lines.append("# Phase VII-B Stage B2-Dev — Negative-Result Report")
    lines.append("")
    lines.append(
        "Per spec §19, this memo documents the negative result honestly — without "
        "burying the failure mode."
    )
    lines.append("")
    lines.append("## What was tested")
    lines.append("")
    lines.append("- 4 games × 3 strategic-learning adversaries × 5 methods × 3 seeds = 180 runs.")
    lines.append("- Adaptive-β (`adaptive_beta`, alias of spec `adaptive_beta_clipped`) was the candidate; ")
    lines.append("  vanilla was the seed-paired baseline.")
    lines.append("")
    lines.append("## Why the gate failed")
    lines.append("")
    lines.append(
        "No `(game, adversary)` cell satisfied **all three** criteria simultaneously: "
        "(a) positive mean AUC diff vs vanilla, (b) strict 95% CI excluding zero on the "
        "positive side OR directional mechanism support, (c) zero `adaptive_beta` "
        "divergence events. See the per-cell promotion table below."
    )
    lines.append("")
    lines.append("| game | adversary | mean Δ AUC | CI | div | align | d_eff | mode | notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for o in outcomes:
        ci = f"[{o.auc_diff_ci_lo:+.2f}, {o.auc_diff_ci_hi:+.2f}]"
        lines.append(
            f"| {o.game} | {o.adversary} | {o.auc_diff_mean:+.2f} | {ci} | "
            f"{o.diverged_count} | {o.mean_alignment_rate:.2f} | "
            f"{o.mean_effective_discount:.2f} | {o.gate_mode} | {o.notes} |"
        )
    lines.append("")

    lines.append("## Plausible follow-up directions")
    lines.append("")
    lines.append(
        "1. Increase seed count to n=10 even at Stage B2-Dev: the 95% CI at n=3 is "
        "uninformative; positive mean diffs may exist but be undetectable."
    )
    lines.append(
        "2. Lengthen episodes from 1k to 5k to allow ε-greedy decay + opponent-driven "
        "regime transitions to manifest. Many cells have monotone curves at 1k that may "
        "diverge later."
    )
    lines.append(
        "3. Re-encode matching_pennies state to include a finite-history window so that "
        "horizon > 1 (state = last m action pairs) — this would un-degenerate the "
        "mechanism evidence."
    )
    lines.append(
        "4. Tune adaptive-β hyperparameters per game family. Coordination games "
        "(rules_of_road, asymmetric coordination trap) may need different `k`, "
        "`beta_max`, and `beta_cap` than zero-sum cycling games."
    )
    lines.append(
        "5. Re-run with `adaptive_magnitude_only` and the `adaptive_beta_no_clip` "
        "ablations. If clip is the dominant force, an unclipped variant may show signal."
    )
    lines.append(
        "6. Consider a two-tier gate: promote on `final_return` paired diff (last 100 "
        "episodes) instead of AUC, since AUC averages over the early random phase."
    )
    lines.append("")
    lines.append("## Diagnostic tables")
    lines.append("")
    lines.append("### Per-method AUC paired diff vs vanilla")
    lines.append("")
    lines.append("| game | adversary | method | mean | ci_lo | ci_hi |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in paired_df[paired_df["metric"] == "auc_return"].iterrows():
        lines.append(
            f"| {r['game']} | {r['adversary']} | {r['method']} | "
            f"{r['mean']:+.2f} | {r['ci_lo']:+.2f} | {r['ci_hi']:+.2f} |"
        )
    lines.append("")

    NEG_REPORT.parent.mkdir(parents=True, exist_ok=True)
    NEG_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_no_update_recommendation(
    outcomes: List[GateOutcome],
) -> None:
    lines: List[str] = []
    lines.append("# Paper-Update Recommendation: NO UPDATE")
    lines.append("")
    lines.append("**Source:** Phase VII-B Stage B2-Dev gate verdict.")
    lines.append("**Decision:** Do **not** update the paper based on Stage B2-Dev results.")
    lines.append("")
    lines.append("## Rationale")
    lines.append("")
    lines.append(
        "Per spec §17, paper updates require evidence that adaptive-β provides "
        "strong AUC or recovery gains in at least two strategic settings, with "
        "supporting mechanism diagnostics. Stage B2-Dev produced **zero** "
        "promoted `(game, adversary)` cells under either the strict or directional "
        "criterion (see `results/adaptive_beta/strategic/stage_B2_dev_summary.md`)."
    )
    lines.append("")
    lines.append(
        "Per spec §17 *No paper update if*:"
    )
    lines.append("- results are weak — **applies**;")
    lines.append("- gains are only in trivial settings — **applies** (matching_pennies is degenerate);")
    lines.append("- adaptive-β is unstable — re-evaluate at Stage B2-Main if dispatched;")
    lines.append("- fixed-positive or fixed-negative dominates consistently — see paired diffs.")
    lines.append("")
    lines.append("## What to do instead")
    lines.append("")
    lines.append(
        "Follow the negative-result follow-ups in "
        "`results/adaptive_beta/strategic/negative_result_report.md` "
        "before re-attempting a paper claim on the strategic-learning suite."
    )
    lines.append("")
    NO_UPDATE_MEMO.parent.mkdir(parents=True, exist_ok=True)
    NO_UPDATE_MEMO.write_text("\n".join(lines), encoding="utf-8")


def update_main_config(promoted: List[GateOutcome]) -> None:
    games = sorted({o.game for o in promoted})
    advs = sorted({o.adversary for o in promoted})
    text = STAGE_B2_MAIN_CFG.read_text(encoding="utf-8")
    games_yaml = "[" + ", ".join(games) + "]"
    advs_yaml = "[" + ", ".join(advs) + "]"
    text = text.replace('games: "${promoted_games}"', f"games: {games_yaml}")
    text = text.replace(
        'adversaries: "${promoted_adversaries}"', f"adversaries: {advs_yaml}"
    )
    STAGE_B2_MAIN_CFG.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading manifest and deduplicating ...")
    manifest_df = load_manifest_dedup()

    print("[2/6] computing per-cell summaries ...")
    summary_df = build_per_run_summary(manifest_df)
    summary_df = add_recovery_time_column(summary_df, manifest_df)
    out_summary = PROC_DIR / "per_run_summary.parquet"
    summary_df.to_parquet(out_summary, index=False)
    print(f"  wrote {out_summary}")

    print("[3/6] computing paired bootstrap diffs ...")
    paired_df = compute_paired_diffs(summary_df)
    out_paired = PROC_DIR / "paired_diffs.parquet"
    paired_df.to_parquet(out_paired, index=False)
    print(f"  wrote {out_paired}")

    print("[4/6] applying promotion gate ...")
    outcomes = evaluate_promotion(summary_df, paired_df)
    promoted_all = select_top_promoted(outcomes, k=3)
    promoted, drop_notes = filter_for_main_dispatch(promoted_all)
    fully_failed = len(promoted) == 0
    print(f"  promoted (gate): {len(promoted_all)} cells")
    for o in promoted_all:
        print(
            f"    [gate] {o.game} | {o.adversary} | mode={o.gate_mode} | "
            f"Δ AUC = {o.auc_diff_mean:+.2f} [{o.auc_diff_ci_lo:+.2f}, {o.auc_diff_ci_hi:+.2f}]"
        )
    if drop_notes:
        print("  Cartesian-product filter:")
        for n in drop_notes:
            print(f"    - {n}")
    print(f"  promoted (Main dispatch): {len(promoted)} cells")
    for o in promoted:
        print(
            f"    [main] {o.game} | {o.adversary} | mode={o.gate_mode} | "
            f"Δ AUC = {o.auc_diff_mean:+.2f} [{o.auc_diff_ci_lo:+.2f}, {o.auc_diff_ci_hi:+.2f}]"
        )

    print("[5/6] generating figures ...")
    fig_paths: Dict[str, Any] = {}
    fig_paths["learning_curves"] = plot_learning_curves(summary_df, manifest_df)
    fig_paths["auc_paired_diff"] = plot_auc_paired_diff(paired_df)
    fig_paths["mechanism"] = plot_mechanism(summary_df)
    fig_paths["event_aligned"] = plot_event_aligned_hypothesis(summary_df)
    for k, v in fig_paths.items():
        print(f"  {k}: {v}")

    print("[6/6] writing memos ...")
    write_summary_memo(
        summary_df, paired_df, outcomes, promoted_all, promoted, drop_notes, fig_paths
    )
    print(f"  wrote {SUMMARY_MEMO}")

    if fully_failed:
        write_negative_report(summary_df, paired_df, outcomes)
        print(f"  wrote {NEG_REPORT}")
        write_no_update_recommendation(outcomes)
        print(f"  wrote {NO_UPDATE_MEMO}")
    else:
        update_main_config(promoted)
        print(f"  updated {STAGE_B2_MAIN_CFG}")

    # Print summary to stdout for caller.
    print()
    print("=" * 78)
    print("VERDICT:", "PROMOTE" if not fully_failed else "NO PROMOTE")
    print("=" * 78)


if __name__ == "__main__":
    main()
