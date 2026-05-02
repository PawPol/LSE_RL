"""Figure 4: Tier I per-cell divergence-event count, stacked by beta-sign.

Bar chart over the 30 Tier I cells (gamma=0.95). For each cell, count runs with
`metrics.npz::divergence_event.sum() > 0` summed across the 21-beta grid x 10 seeds
(max 210 per cell). Cells sorted by total divergence count descending. Bars are
stacked: positive-beta arms vs negative-beta arms vs vanilla. Highlights the
524 divergent runs the G6c review identified.

Source: results/adaptive_beta/tab_six_games/figures/v10/tables/v10_per_cell_per_method_per_gamma.csv
Output:  results/adaptive_beta/tab_six_games/figures/v10/divergence_signature_v10.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v10_fig_common import FIG_DIR, LONG_CSV, apply_paper_style


def main() -> None:
    apply_paper_style()
    df = pd.read_csv(LONG_CSV)
    t1 = df[df["stage"] == "tier1"].copy()
    # A run "diverged" if either flag is set; both should agree.
    t1["is_div"] = (t1["diverged"]) | (t1["divergence_event_sum"] > 0)
    t1["beta_sign"] = np.where(t1["beta"] > 0, "pos",
                               np.where(t1["beta"] < 0, "neg", "vanilla"))

    grp = t1.groupby(["subcase", "beta_sign"])["is_div"].sum().unstack(fill_value=0)
    for col in ("pos", "neg", "vanilla"):
        if col not in grp.columns:
            grp[col] = 0
    grp["total"] = grp[["pos", "neg", "vanilla"]].sum(axis=1)
    grp = grp.sort_values("total", ascending=False)

    cells = grp.index.tolist()
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.2))
    x = np.arange(len(cells))
    bars_neg = ax.bar(x, grp["neg"], color="#1f77b4", label=r"$\beta<0$ arms")
    bars_pos = ax.bar(x, grp["pos"], bottom=grp["neg"], color="#d62728", label=r"$\beta>0$ arms")
    bars_van = ax.bar(x, grp["vanilla"], bottom=grp["neg"] + grp["pos"],
                       color="#aaaaaa", label=r"$\beta=0$ (vanilla)")

    ax.set_xticks(x)
    ax.set_xticklabels(cells, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Divergent runs (count)")
    ax.set_title(
        r"Tier I divergence flags per cell, stacked by $\beta$ sign "
        r"(max 210 per cell = 21 arms $\times$ 10 seeds, $\gamma=0.95$).  "
        rf"Total: {int(grp['total'].sum())} divergent runs.",
        fontsize=10,
    )
    # Annotate total counts above each bar
    for xi, total in zip(x, grp["total"]):
        if total > 0:
            ax.text(xi, total + 1, str(int(total)), ha="center", va="bottom", fontsize=7)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylim(0, max(grp["total"].max() * 1.15, 1.0))

    fig.tight_layout()
    out = FIG_DIR / "divergence_signature_v10.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {out}  (Tier I total divergent: {int(grp['total'].sum())})")


if __name__ == "__main__":
    main()
