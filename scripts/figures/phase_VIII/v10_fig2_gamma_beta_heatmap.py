"""Figure 2: Tier II gamma x beta heatmap, 4 headline cells.

4-panel facet (AC-Trap, SH-FiniteMemoryRegret, RR-StationaryConvention, DC-Long50).
y-axis: gamma in (0.60, 0.80, 0.90, 0.95); x-axis: beta in 21-arm grid.
Color: AUC normalized per-row by subtracting the row's vanilla AUC, then dividing
by the row's |max - vanilla| so each gamma row has color in [-1, 1] with white at
vanilla. Diverging colormap. Star marker = best-beta per row (seed-mean argmax,
exact-equality tie broken in favor of vanilla).

Source: results/adaptive_beta/tab_six_games/figures/v10/tables/v10_per_cell_per_method_per_gamma.csv
Output:  results/adaptive_beta/tab_six_games/figures/v10/gamma_beta_heatmap_v10.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from v10_fig_common import BETA_GRID, FIG_DIR, LONG_CSV, apply_paper_style

GAMMAS = [0.60, 0.80, 0.90, 0.95]
HEADLINE = ["AC-Trap", "SH-FiniteMemoryRegret", "RR-StationaryConvention", "DC-Long50"]


def main() -> None:
    apply_paper_style()
    df = pd.read_csv(LONG_CSV)
    t2 = df[df["stage"] == "tier2"].copy()

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=False)

    for ax, cell in zip(axes, HEADLINE):
        sub = t2[t2["subcase"] == cell]
        # Build (gamma, beta) -> seed-mean AUC matrix
        mat = np.full((len(GAMMAS), len(BETA_GRID)), np.nan)
        norm_mat = np.full_like(mat, np.nan)
        best_idx_per_row: list[int] = []
        for i, g in enumerate(GAMMAS):
            row = sub[np.isclose(sub["gamma"], g)]
            means = row.groupby("beta")["AUC"].mean()
            for j, b in enumerate(BETA_GRID):
                if b in means.index:
                    mat[i, j] = means.loc[b]
            row_vals = mat[i]
            # vanilla AUC (beta=0)
            j_van = BETA_GRID.index(0.0)
            van = row_vals[j_van]
            # normalize: (auc - vanilla) / max(|auc - vanilla|)
            shift = row_vals - van
            denom = np.nanmax(np.abs(shift)) or 1.0
            norm_mat[i] = shift / denom
            # best-beta: exact-equality tie -> vanilla preferred
            row_max = np.nanmax(row_vals)
            tied = np.where(row_vals == row_max)[0]
            if 0.0 == BETA_GRID[j_van] and j_van in tied:
                best_idx_per_row.append(j_van)
            else:
                # smallest |beta| among ties
                best_idx_per_row.append(int(tied[np.argmin(np.abs(np.array(BETA_GRID)[tied]))]))

        # Diverging colormap centered on 0 (== row vanilla AUC)
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        im = ax.imshow(norm_mat, cmap="RdBu_r", norm=norm, aspect="auto",
                       interpolation="nearest", origin="lower")

        # ticks
        ax.set_xticks(np.arange(len(BETA_GRID)))
        ax.set_xticklabels([f"{b:+.2f}" if (i % 2 == 0) else "" for i, b in enumerate(BETA_GRID)],
                          rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(GAMMAS)))
        ax.set_yticklabels([f"{g:.2f}" for g in GAMMAS])
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\gamma$" if cell == HEADLINE[0] else "")
        ax.set_title(cell, fontsize=10)

        # vanilla column
        ax.axvline(BETA_GRID.index(0.0), color="black", linestyle=":", linewidth=0.6)

        # best beta markers
        for i, j in enumerate(best_idx_per_row):
            ax.plot(j, i, marker="*", color="#fef200", markersize=10,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    cbar_ax = fig.add_axes([0.92, 0.20, 0.012, 0.62])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label(r"AUC$-$AUC$_{\beta=0}$ (per-row, normalized)", fontsize=9)

    fig.suptitle(
        r"Tier II $\gamma\times\beta$ heatmaps, 4 headline cells, 5 seeds per arm.  "
        r"Color centered on each row's vanilla AUC; star = seed-mean argmax (vanilla preferred on exact ties).",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 0.91, 0.92))
    out = FIG_DIR / "gamma_beta_heatmap_v10.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
