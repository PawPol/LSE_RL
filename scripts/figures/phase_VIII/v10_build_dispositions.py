"""Build v10 H1/H2/H3 dispositions table.

Per spec: ~120 rows, one per (cell, gamma), with selection rule per G6c memo:
- Headline cells (AC-Trap, SH-FiniteMemoryRegret, RR-StationaryConvention, DC-Long50)
  use Tier II for all gamma values.
- Non-headline cells use Tier III for gamma in {0.60, 0.80, 0.90} and Tier I for gamma=0.95.

Columns:
  cell, gamma, source, best_beta, best_method, best_auc_mean,
  vanilla_auc_mean, paired_d_cohen, paired_bootstrap_auc_advantage_ci_lower,
  paired_bootstrap_auc_advantage_ci_upper, align_at_best_beta_final,
  align_at_best_beta_last200, h1_confirms, h2_evaluable, h2_ratio, h3_confirms

Bootstrap CIs for paired AUC advantage are taken from the G6c review at
results/adaptive_beta/tab_six_games/codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md
where computed (Tier II AC-Trap/SH-FMR/RR-StationaryConvention/DC-Long50 at gamma=0.60),
and computed locally with B=20,000 paired seed resamples for the remaining (cell, gamma).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/liq/Documents/Claude/Projects/LSE_RL")
LONG_CSV = ROOT / "results/adaptive_beta/tab_six_games/figures/v10/tables/v10_per_cell_per_method_per_gamma.csv"
OUT_CSV = ROOT / "results/adaptive_beta/tab_six_games/figures/v10/tables/v10_h1_h2_h3_dispositions.csv"

HEADLINE_CELLS = {"AC-Trap", "SH-FiniteMemoryRegret", "RR-StationaryConvention", "DC-Long50"}
GAMMAS = (0.60, 0.80, 0.90, 0.95)

# G6c paired bootstrap CI table (best minus vanilla AUC, Tier II, gamma=0.60, B=20,000)
# Source: codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md sec (b)
G6C_TIER2_CIS = {
    ("AC-Trap", 0.60): dict(best_beta=+0.10, best_auc=529077.80, vanilla_auc=528949.20,
                            advantage=128.60, ci_low=45.20, ci_high=212.00, p_best_pos=0.998),
    ("SH-FiniteMemoryRegret", 0.60): dict(best_beta=+0.35, best_auc=106698.30, vanilla_auc=106570.80,
                                          advantage=127.50, ci_low=-91.50, ci_high=360.00, p_best_pos=0.514),
    ("RR-StationaryConvention", 0.60): dict(best_beta=-0.50, best_auc=56293.20, vanilla_auc=56273.60,
                                            advantage=19.60, ci_low=6.40, ci_high=30.80, p_best_pos=0.000),
    ("DC-Long50", 0.60): dict(best_beta=-2.00, best_auc=182029.46, vanilla_auc=181569.81,
                              advantage=459.65, ci_low=459.65, ci_high=459.65, p_best_pos=0.000),
}


def select_source(cell: str, gamma: float) -> str:
    if cell in HEADLINE_CELLS:
        return "tier2"
    if abs(gamma - 0.95) < 1e-6:
        return "tier1"
    return "tier3"


def paired_bootstrap_advantage(best: np.ndarray, vanilla: np.ndarray, B: int = 20000,
                                seed: int = 20260502) -> tuple[float, float, float]:
    """Paired bootstrap of (best - vanilla) AUC. Returns (mean_adv, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    n = len(best)
    idx = rng.integers(0, n, size=(B, n))
    # paired: identical resample index
    diffs = best[idx] - vanilla[idx]
    means = diffs.mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sx = x.std(ddof=1) if nx > 1 else 0.0
    sy = y.std(ddof=1) if ny > 1 else 0.0
    if (nx + ny - 2) <= 0:
        return float("nan")
    pooled_var = ((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / np.sqrt(pooled_var))


def build_disposition(df_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    cells = sorted(df_long["subcase"].unique())
    for cell in cells:
        # H2 ratio (computed once per cell, using Tier II if eligible)
        h2_eligible = False
        h2_ratio = float("nan")
        if cell in HEADLINE_CELLS:
            t2 = df_long[(df_long["subcase"] == cell) & (df_long["stage"] == "tier2")]
            if len(t2) > 0:
                # gamma=0.95 best beta sign
                g95 = t2[t2["gamma"] == 0.95]
                g60 = t2[t2["gamma"] == 0.60]
                if len(g95) and len(g60):
                    g95_means = g95.groupby(["beta"])["AUC"].mean()
                    g60_means = g60.groupby(["beta"])["AUC"].mean()
                    if len(g95_means) and len(g60_means):
                        bb95 = g95_means.idxmax()
                        bb60 = g60_means.idxmax()
                        if bb95 < 0:
                            h2_eligible = True
                            van95 = g95[g95["beta"] == 0]["AUC"].values
                            best95 = g95[g95["beta"] == bb95]["AUC"].values
                            van60 = g60[g60["beta"] == 0]["AUC"].values
                            best60 = g60[g60["beta"] == bb60]["AUC"].values
                            d95 = cohen_d(best95, van95)
                            d60 = cohen_d(best60, van60)
                            if d95 == d95 and d60 == d60 and d95 > 0:
                                h2_ratio = float(d60 / d95)

        for gamma in GAMMAS:
            source = select_source(cell, gamma)
            sub = df_long[(df_long["subcase"] == cell) & (df_long["stage"] == source) &
                          (np.isclose(df_long["gamma"], gamma))]
            if len(sub) == 0:
                continue
            # Best beta = beta with max seed-mean AUC; ties broken in favor of vanilla
            # (matches G6c memo: when multiple arms have *exactly equal* return AUC mean,
            # vanilla is preferred; tiny float differences keep their non-zero argmax).
            means = sub.groupby("beta")["AUC"].mean()
            max_auc = means.max()
            tied = means[means == max_auc]
            if 0.0 in tied.index:
                best_beta = 0.0
            else:
                best_beta = float(tied.index[np.argmin(np.abs(tied.index.values))])
            best_auc = float(max_auc)
            best_method = "vanilla" if best_beta == 0 else (
                f"fixed_beta_{best_beta:+g}".replace("+0", "+0").replace("-0", "-0")
            )
            # Reformat method exactly like raw dirs (e.g. "+0.05" not "+0.05")
            best_method = sub[np.isclose(sub["beta"], best_beta)]["method"].iloc[0]

            best_auc_seeds = sub[np.isclose(sub["beta"], best_beta)].sort_values("seed")["AUC"].values
            van_seeds = sub[np.isclose(sub["beta"], 0.0)].sort_values("seed")["AUC"].values
            vanilla_auc = float(van_seeds.mean()) if len(van_seeds) else float("nan")

            d = cohen_d(best_auc_seeds, van_seeds) if len(van_seeds) else float("nan")

            # Bootstrap CI: prefer G6c's stored value; otherwise compute locally
            key = (cell, gamma)
            if key in G6C_TIER2_CIS:
                g = G6C_TIER2_CIS[key]
                advantage = g["advantage"]
                ci_low = g["ci_low"]
                ci_high = g["ci_high"]
            else:
                if len(best_auc_seeds) >= 2 and len(van_seeds) >= 2 and len(best_auc_seeds) == len(van_seeds):
                    advantage, ci_low, ci_high = paired_bootstrap_advantage(best_auc_seeds, van_seeds)
                else:
                    advantage = float(best_auc_seeds.mean() - van_seeds.mean()) if len(van_seeds) else float("nan")
                    ci_low = float("nan")
                    ci_high = float("nan")

            align_at_best_final = float(sub[np.isclose(sub["beta"], best_beta)]["align_final"].mean())
            align_at_best_l200 = float(sub[np.isclose(sub["beta"], best_beta)]["align_last200"].mean())

            # H1 confirms: only Tier II cells at gamma=0.60 are eligible. Confirm if best_beta>0
            # AND paired CI strictly above 0. AC-Trap is the only confirming arm per G6c.
            h1_confirms = False
            if cell in HEADLINE_CELLS and abs(gamma - 0.60) < 1e-6:
                # G6c rule: best_beta>0 AND best-beta CI excludes 0 AND paired AUC advantage CI > 0
                h1_confirms = (best_beta > 0) and (ci_low > 0) and (G6C_TIER2_CIS[key]["p_best_pos"] > 0.95)

            # H3 confirms: align_final >= 0.5
            h3_confirms = align_at_best_final >= 0.5

            rows.append(dict(
                cell=cell,
                gamma=gamma,
                source=source,
                best_beta=best_beta,
                best_method=best_method,
                best_auc_mean=best_auc,
                vanilla_auc_mean=vanilla_auc,
                paired_d_cohen=d,
                paired_bootstrap_auc_advantage_mean=advantage,
                paired_bootstrap_auc_advantage_ci_lower=ci_low,
                paired_bootstrap_auc_advantage_ci_upper=ci_high,
                align_at_best_beta_final=align_at_best_final,
                align_at_best_beta_last200=align_at_best_l200,
                h1_confirms=h1_confirms,
                h2_evaluable=h2_eligible,
                h2_ratio=h2_ratio,
                h3_confirms=h3_confirms,
            ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-csv", type=Path, default=LONG_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()
    df = pd.read_csv(args.long_csv)
    out = build_disposition(df)
    out = out.sort_values(["cell", "gamma"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} rows)")
    print()
    print("H3 confirmation by gamma:")
    print(out.groupby("gamma")["h3_confirms"].agg(["sum", "count"]))
    print()
    print("H1 confirmations:")
    print(out[out["h1_confirms"]][["cell", "gamma", "best_beta", "paired_bootstrap_auc_advantage_ci_lower",
                                    "paired_bootstrap_auc_advantage_ci_upper"]])


if __name__ == "__main__":
    main()
