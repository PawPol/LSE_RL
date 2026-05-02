"""Figure 5: DC-Long50 Bellman-residual contraction AUC vs beta (Tier I).

The headline metric for DC-Long50 cum-return AUC is invariant across beta because
the chain is deterministic. Use the v5b beta-specific Bellman-residual contraction
AUC instead: auc_neg_log_residual = trapezoid(-log(metrics.npz::bellman_residual + 1e-8), episode).
Plot per-beta seed mean +/- 1 sigma and individual seed lines as scatter dots.

Source: results/adaptive_beta/tab_six_games/figures/v10/tables/v10_per_cell_per_method_per_gamma.csv
Output:  results/adaptive_beta/tab_six_games/figures/v10/dc_long50_residual_decay_v10.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v10_fig_common import BETA_GRID, FIG_DIR, LONG_CSV, apply_paper_style


def main() -> None:
    apply_paper_style()
    df = pd.read_csv(LONG_CSV)
    sub = df[(df["stage"] == "tier1") & (df["subcase"] == "DC-Long50")]
    # bellman-residual AUC == AUC field for DC cells per spec routing
    grp_mean = sub.groupby("beta")["AUC"].mean().reindex(BETA_GRID)
    grp_std = sub.groupby("beta")["AUC"].std(ddof=1).reindex(BETA_GRID)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    x = np.array(BETA_GRID)
    m = grp_mean.values
    s = grp_std.values
    ax.fill_between(x, m - s, m + s, color="#3366aa", alpha=0.18, linewidth=0)
    # individual seed points
    for b in BETA_GRID:
        seed_vals = sub[np.isclose(sub["beta"], b)]["AUC"].values
        if len(seed_vals):
            ax.scatter([b] * len(seed_vals), seed_vals, color="#1f3a93",
                       s=8, alpha=0.45, edgecolor="none", zorder=2)
    ax.plot(x, m, color="#1f3a93", linewidth=1.4, marker="o", markersize=3, zorder=3)
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=0.7)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\int_0^T -\log(\mathrm{Bellman\ residual} + 10^{-8})\,d\,\mathrm{epi}$  (linear)")
    ax.set_title("DC-Long50 contraction AUC, linear axis", fontsize=10)
    ax.set_xlim(-2.15, 2.15)

    # log-y panel to show the 17-OOM separation
    axb = axes[1]
    # shift to positive: subtract min
    m_min = np.nanmin(m)
    m_pos = m - m_min + 1.0
    axb.plot(x, m_pos, color="#1f3a93", linewidth=1.4, marker="o", markersize=3)
    axb.set_yscale("log")
    axb.axvline(0.0, color="#888888", linestyle=":", linewidth=0.7)
    axb.set_xlabel(r"$\beta$")
    axb.set_ylabel(r"AUC$-$AUC$_{\min}+1$  (log scale)")
    axb.set_title(
        "DC-Long50 contraction AUC, log axis",
        fontsize=10,
    )
    axb.set_xlim(-2.15, 2.15)

    fig.suptitle(
        r"DC-Long50: $\beta$-specific Bellman-residual contraction AUC, Tier I, $\gamma=0.95$, 10 seeds. "
        r"$-\beta>0>+\beta$ separation is visible; positive $\beta$ arms diverge.",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "dc_long50_residual_decay_v10.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
