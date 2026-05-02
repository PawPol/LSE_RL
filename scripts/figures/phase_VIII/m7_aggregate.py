"""M7.1 aggregator: join TAB-redispatch (3360 runs) + Stage 2 baselines
(480 runs) into a paired long CSV and compute paired-bootstrap CIs per
(cell, γ).

Spec authority: phase_VIII_tab_six_games.md §10.3 (Stage 2 acceptance).

Outputs:
- results/.../processed/m7_1_long.csv
- results/.../processed/m7_1_paired_comparison.csv
- prints summary table to stdout for memo construction
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TAB_ROOT = ROOT / "results/adaptive_beta/tab_six_games/raw/VIII/m7_1_tier2_tab_redispatch_10seeds"
BL_ROOT = ROOT / "results/adaptive_beta/tab_six_games/raw/VIII/m7_1_stage2_baselines_v2_headline"
SA_ROOT = ROOT / "results/adaptive_beta/tab_six_games/raw/VIII/m7_2_stage2_strategic_agents_headline"
OUT_DIR = ROOT / "results/adaptive_beta/tab_six_games/processed"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CELLS = ["AC-Trap", "RR-StationaryConvention", "SH-FiniteMemoryRegret", "DC-Long50"]
GAMMAS = [0.6, 0.8, 0.9, 0.95]


def beta_from_method(m: str) -> float:
    if m == "vanilla":
        return 0.0
    if m.startswith("fixed_beta_"):
        return float(m[len("fixed_beta_"):])
    return float("nan")  # baseline


def headline_auc(cell: str, ret: np.ndarray, br: np.ndarray,
                 br_beta: np.ndarray | None = None) -> float:
    if cell == "DC-Long50":
        # advance-only: -log Bellman residual AUC (v5b). TAB rows have
        # the β-specific `bellman_residual_beta` (preferred); baseline
        # rows do not (β is undefined for non-TAB), so fall back to the
        # generic `bellman_residual` per spec §6.3 — both sides are
        # comparable because the v5b metric only requires a per-episode
        # |TD-error| signal.
        signal = br_beta if br_beta is not None and br_beta.size else br
        return float((-np.log(np.maximum(signal, 1e-12))).sum())
    return float(ret.sum())


def aggregate(root: Path, is_baseline: bool) -> pd.DataFrame:
    rows = []
    for run_path in root.rglob("run.json"):
        rj = json.load(open(run_path))
        if rj.get("status") and rj.get("status") != "completed":
            continue
        npz_path = run_path.parent / "metrics.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as m:
            ret = m["return"] if "return" in m.files else m["return_"]
            br = m["bellman_residual"]
            br_beta = m["bellman_residual_beta"] if "bellman_residual_beta" in m.files else None
            align = m["alignment_rate"] if "alignment_rate" in m.files else np.zeros_like(ret)
            qmax = m["q_abs_max"] if "q_abs_max" in m.files else np.zeros_like(ret)
            div = m["divergence_event"] if "divergence_event" in m.files else np.zeros_like(ret)
        cell = rj["subcase"]
        method = rj["method"]
        rows.append(dict(
            stage=rj["stage"],
            game=rj["game"],
            cell=cell,
            method=method,
            beta=beta_from_method(method),
            gamma=float(rj["gamma"]),
            seed=int(rj["seed"]),
            is_baseline=is_baseline,
            return_AUC=float(ret.sum()),
            headline_AUC=headline_auc(cell, np.asarray(ret), np.asarray(br), np.asarray(br_beta) if br_beta is not None else None),
            return_last200=float(np.asarray(ret)[-200:].mean()),
            align_last200=float(np.asarray(align)[-200:].mean()) if align.size else float("nan"),
            q_abs_max_max=float(np.asarray(qmax).max()) if qmax.size else float("nan"),
            divergence_event_sum=int(np.asarray(div).sum()),
        ))
    return pd.DataFrame(rows)


def paired_ci(a: np.ndarray, b: np.ndarray, B: int = 20_000, rng_seed: int = 0):
    """Paired-bootstrap 95% CI of mean(a − b)."""
    rng = np.random.default_rng(rng_seed)
    diff = a - b
    n = len(diff)
    means = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main():
    print("[M7.1] aggregating TAB re-dispatch ...")
    df_tab = aggregate(TAB_ROOT, is_baseline=False)
    print(f"   TAB rows: {len(df_tab)}")
    print("[M7.1] aggregating Q-learning baselines ...")
    df_bl = aggregate(BL_ROOT, is_baseline=True)
    print(f"   BL  rows: {len(df_bl)}")
    df_sa = pd.DataFrame()
    if SA_ROOT.exists():
        print("[M7.2] aggregating strategic-learning agents ...")
        df_sa = aggregate(SA_ROOT, is_baseline=True)
        print(f"   SA  rows: {len(df_sa)}")

    long = pd.concat([df_tab, df_bl, df_sa], ignore_index=True)
    long_path = OUT_DIR / "m7_1_long.csv"
    long.to_csv(long_path, index=False)
    print(f"   wrote {long_path}  ({len(long)} rows)")

    # Build paired-comparison rows.
    # For each (cell, γ) compute:
    #   - vanilla baseline (TAB at β=0)
    #   - best_fixed_positive_TAB: argmax over β>0 of mean AUC across seeds
    #   - best_fixed_negative_TAB: argmax over β<0
    #   - best_fixed_beta_grid:    argmax over all β (incl 0)
    #   - 3 baselines (restart, sliding_window, tuned_eps)
    # Then paired-CI vs vanilla on a per-seed basis.
    pair_rows = []
    BASELINES = [
        "restart_Q_learning",
        "sliding_window_Q_learning",
        "tuned_epsilon_greedy_Q_learning",
        "regret_matching_agent",
        "smoothed_fictitious_play_agent",
    ]
    for cell in CELLS:
        for gamma in GAMMAS:
            sub_tab = long[(long["cell"] == cell) & (long["gamma"] == gamma) &
                           (~long["is_baseline"])]
            sub_bl = long[(long["cell"] == cell) & (long["gamma"] == gamma) &
                          (long["is_baseline"])]
            if len(sub_tab) == 0:
                continue

            van = sub_tab[sub_tab["method"] == "vanilla"].sort_values("seed")
            van_auc = van["headline_AUC"].to_numpy()
            seeds = van["seed"].to_numpy()

            # Mean per method to find best β arms.
            means = sub_tab.groupby("method")["headline_AUC"].mean()

            pos_methods = [m for m in means.index if beta_from_method(m) > 0]
            neg_methods = [m for m in means.index if beta_from_method(m) < 0]
            if pos_methods:
                best_pos = means.loc[pos_methods].idxmax()
            else:
                best_pos = None
            if neg_methods:
                best_neg = means.loc[neg_methods].idxmax()
            else:
                best_neg = None
            best_grid = means.idxmax()

            def auc_for(method):
                m_rows = sub_tab[sub_tab["method"] == method].sort_values("seed")
                return m_rows["headline_AUC"].to_numpy(), m_rows["seed"].to_numpy()

            comparisons = []
            if best_pos:
                comparisons.append(("best_fixed_positive_TAB", best_pos, *auc_for(best_pos)))
            if best_neg:
                comparisons.append(("best_fixed_negative_TAB", best_neg, *auc_for(best_neg)))
            comparisons.append(("best_fixed_beta_grid", best_grid, *auc_for(best_grid)))

            # Baselines.
            for bm in BASELINES:
                bl_rows = sub_bl[sub_bl["method"] == bm].sort_values("seed")
                if len(bl_rows) == 0:
                    continue
                comparisons.append((bm, bm, bl_rows["headline_AUC"].to_numpy(),
                                    bl_rows["seed"].to_numpy()))

            for label, source, vals, sds in comparisons:
                # Pair on seed: align seed lists.
                pair_seeds = np.intersect1d(seeds, sds)
                v_aligned = np.array([float(van[van["seed"] == s]["headline_AUC"].iloc[0])
                                      for s in pair_seeds])
                # Re-pull arm AUC for those seeds.
                arm_seeds = sds.tolist()
                a_aligned = np.array([vals[arm_seeds.index(s)] for s in pair_seeds])
                if len(pair_seeds) < 2:
                    continue
                d, lo, hi = paired_ci(a_aligned, v_aligned, B=20_000, rng_seed=int(gamma * 100))
                pair_rows.append(dict(
                    cell=cell,
                    gamma=gamma,
                    method=label,
                    method_source=source,
                    n_seeds=len(pair_seeds),
                    mean_arm_AUC=float(a_aligned.mean()),
                    mean_vanilla_AUC=float(v_aligned.mean()),
                    delta=d,
                    ci_lo=lo,
                    ci_hi=hi,
                    sig=("✓" if lo > 0 else ("✗" if hi < 0 else "0")),
                ))

    paired = pd.DataFrame(pair_rows)
    paired_path = OUT_DIR / "m7_1_paired_comparison.csv"
    paired.to_csv(paired_path, index=False)
    print(f"   wrote {paired_path}  ({len(paired)} rows)")

    # Print summary by cell, γ.
    print()
    print("=" * 110)
    print(f"{'cell':<26} {'γ':>5} {'method':<28} {'src':<22} {'Δ':>10} {'CI_lo':>10} {'CI_hi':>10} sig")
    print("-" * 110)
    for _, r in paired.iterrows():
        print(f"{r['cell']:<26} {r['gamma']:>5} {r['method']:<28} {r['method_source']:<22} "
              f"{r['delta']:>10.2f} {r['ci_lo']:>10.2f} {r['ci_hi']:>10.2f}  {r['sig']}")
    print()

    # G+ / G- classification per cell (aggregating across γ; H1/H2 from V10
    # already established γ-dependence, so we summarize by cell).
    print("=" * 70)
    print("G+ / G- CLASSIFICATION (paired-CI strictly above 0 vs vanilla)")
    print("-" * 70)
    for cell in CELLS:
        sub = paired[paired["cell"] == cell]
        pos_wins = sub[(sub["method"] == "best_fixed_positive_TAB") & (sub["sig"] == "✓")]
        neg_wins = sub[(sub["method"] == "best_fixed_negative_TAB") & (sub["sig"] == "✓")]
        print(f"{cell:<26}  G+ at γ ∈ {pos_wins['gamma'].tolist()}  |  "
              f"G- at γ ∈ {neg_wins['gamma'].tolist()}")
    print()


if __name__ == "__main__":
    main()
