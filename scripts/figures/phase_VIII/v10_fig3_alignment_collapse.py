"""Figure 3: alignment-rate sign-bifurcation across the 21-arm beta grid.

For 4 representative cells (AC-Trap [Tier II gamma=0.95], SH-FiniteMemoryRegret [Tier II g=0.95],
MP-FiniteMemoryBR [Tier I gamma=0.95], RR-StationaryConvention [Tier II gamma=0.95]):
plot end-of-training alignment_rate (last-200 mean) as a function of beta. Shows
the predicted sign bifurcation: alignment is high for beta<0 and collapses at beta>=0.

Data sources are mixed because three of the four cells are headline cells served
from Tier II while MP-FiniteMemoryBR is non-headline. We use the same `source`
selection rule as the dispositions table: Tier II for headline cells at gamma=0.95,
Tier I for non-headline cells at gamma=0.95.

Source: results/adaptive_beta/tab_six_games/figures/v10/tables/v10_per_cell_per_method_per_gamma.csv
Output:  results/adaptive_beta/tab_six_games/figures/v10/alignment_collapse_v10.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v10_fig_common import BETA_GRID, FIG_DIR, LONG_CSV, apply_paper_style

CELLS = [
    ("AC-Trap", "tier2", 0.95, "#d62728"),
    ("SH-FiniteMemoryRegret", "tier2", 0.95, "#9467bd"),
    ("MP-FiniteMemoryBR", "tier1", 0.95, "#2ca02c"),
    ("RR-StationaryConvention", "tier2", 0.95, "#1f77b4"),
]


def main() -> None:
    apply_paper_style()
    df = pd.read_csv(LONG_CSV)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2))
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=0.7, zorder=1)
    ax.axhline(0.5, color="#bbbbbb", linestyle="-", linewidth=0.5, zorder=1)

    for cell, stage, gamma, color in CELLS:
        sub = df[(df["subcase"] == cell) & (df["stage"] == stage) &
                 (np.isclose(df["gamma"], gamma))]
        grp_mean = sub.groupby("beta")["align_last200"].mean().reindex(BETA_GRID)
        grp_std = sub.groupby("beta")["align_last200"].std(ddof=1).reindex(BETA_GRID)
        x = np.array(BETA_GRID, dtype=float)
        m = grp_mean.values
        s = grp_std.values
        s = np.nan_to_num(s, nan=0.0)
        ax.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=color, linewidth=1.4, marker="o", markersize=3,
                label=f"{cell}  ({stage}, $\\gamma$={gamma})")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"alignment\_rate, last-200 episodes (mean over seeds, $\pm1\sigma$)")
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=8)
    ax.set_title(r"Sign-bifurcation: alignment collapses for $\beta\geq0$", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "alignment_collapse_v10.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
