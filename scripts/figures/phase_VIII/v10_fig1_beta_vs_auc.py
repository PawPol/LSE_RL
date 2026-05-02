"""Figure 1: Tier I beta-vs-AUC across 30 cells (5x6 grid).

x-axis: beta in [-2, +2] (21 arms); y-axis: headline AUC (cumulative-return AUC
for matrix games, bellman-residual contraction AUC for DC-Long50/Medium20/Short10).
Per cell, mean across 10 seeds with +/-1sigma ribbon. Vertical dashed line at
beta=0; star marker at the seed-mean argmax. CI derivation: simple seed-level
mean and standard deviation; not bootstrap (Tier I has 10 seeds; ribbon is
visualization, not inferential).

Source: results/adaptive_beta/tab_six_games/figures/v10/tables/v10_per_cell_per_method_per_gamma.csv
Output:  results/adaptive_beta/tab_six_games/figures/v10/beta_vs_auc_v10.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v10_fig_common import BETA_GRID, DC_CELLS, FIG_DIR, LONG_CSV, apply_paper_style


CELLS_GAMES = [
    ("AC-FictitiousPlay", "asymmetric_coordination"),
    ("AC-Inertia", "asymmetric_coordination"),
    ("AC-SmoothedBR", "asymmetric_coordination"),
    ("AC-Trap", "asymmetric_coordination"),
    ("DC-Branching20", "delayed_chain"),
    ("DC-Long50", "delayed_chain"),
    ("DC-Medium20", "delayed_chain"),
    ("DC-Short10", "delayed_chain"),
    ("MP-FiniteMemoryBR", "matching_pennies"),
    ("MP-HypothesisTesting", "matching_pennies"),
    ("MP-RegretMatching", "matching_pennies"),
    ("MP-Stationary", "matching_pennies"),
    ("PG-BetterReplyInertia", "potential"),
    ("PG-Congestion", "potential"),
    ("PG-CoordinationPotential", "potential"),
    ("PG-SwitchingPayoff", "potential"),
    ("RR-ConventionSwitch", "rules_of_road"),
    ("RR-HypothesisTesting", "rules_of_road"),
    ("RR-Sparse", "rules_of_road_sparse"),
    ("RR-StationaryConvention", "rules_of_road"),
    ("RR-Tremble", "rules_of_road"),
    ("SH-FictitiousPlay", "shapley"),
    ("SH-FiniteMemoryRegret", "shapley"),
    ("SH-HypothesisTesting", "shapley"),
    ("SH-SmoothedFP", "shapley"),
    ("SO-AntiCoordination", "soda_uncertain"),
    ("SO-BiasedPreference", "soda_uncertain"),
    ("SO-Coordination", "soda_uncertain"),
    ("SO-TypeSwitch", "soda_uncertain"),
    ("SO-ZeroSum", "soda_uncertain"),
]


def main() -> None:
    apply_paper_style()
    df = pd.read_csv(LONG_CSV)
    t1 = df[df["stage"] == "tier1"].copy()

    nrows, ncols = 5, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 9.5), sharex=True)

    for ax, (cell, _game) in zip(axes.flat, CELLS_GAMES):
        sub = t1[t1["subcase"] == cell]
        # group by beta -> mean and std across seeds
        grp = sub.groupby("beta")["AUC"]
        mean = grp.mean().reindex(BETA_GRID)
        std = grp.std(ddof=1).reindex(BETA_GRID)
        n = grp.count().reindex(BETA_GRID).fillna(0).astype(int)

        x = np.array(BETA_GRID, dtype=float)
        m = mean.values
        s = std.values
        # Draw ribbon
        ax.fill_between(x, m - s, m + s, color="#4477AA", alpha=0.20, linewidth=0)
        ax.plot(x, m, color="#1f3a93", linewidth=1.2, marker="o", markersize=2.0)

        # vanilla marker
        ax.axvline(0.0, color="#aaaaaa", linestyle=":", linewidth=0.7)

        # best-beta marker
        if not np.isnan(m).all():
            best_i = int(np.nanargmax(m))
            ax.plot(x[best_i], m[best_i], marker="*", color="#cc3311", markersize=7.5,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)

        # axis label / title
        is_dc = cell in DC_CELLS
        unit = "Bellman-res AUC" if is_dc else "Return AUC"
        ax.set_title(f"{cell}", fontsize=9, pad=2)
        ax.tick_params(axis="both", which="major", pad=1)
        ax.set_xlim(-2.15, 2.15)
        # Tag DC cells visually with a different x-axis label color
        if is_dc:
            ax.set_facecolor("#fbfaf0")

    # Shared x label / y note
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\beta$", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Headline AUC", fontsize=9)

    fig.suptitle(
        r"Tier I $\beta$-vs-AUC, 21-arm grid, 10 seeds per arm, $\gamma=0.95$.  "
        r"DC-Long50/Medium20/Short10 use Bellman-residual AUC; all others use return AUC.  "
        r"Star = seed-mean argmax; dotted line = $\beta=0$; ribbon = $\pm1\sigma$.",
        fontsize=9, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = FIG_DIR / "beta_vs_auc_v10.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
