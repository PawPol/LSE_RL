"""Phase VI-F figure generator: dual-objective Pareto plot.

Produces figures/main/fig_dual_objective_eval.pdf from
results/rl_VI_F/dual_eval.parquet.  Panel A: gap_cl_eval vs
gap_safe_eval scatter with the six shortlisted stochastic Family A
tasks annotated.  Panel B: V^pi_cl and V^pi_safe per policy per task
grouped-bar chart.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(
    parquet_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("V_star_classical").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Panel A: Pareto scatter.
    ax = axes[0]
    ax.scatter(df["gap_cl_eval"], df["gap_safe_eval"],
               c=np.log10(df["V_star_classical"].clip(lower=1e-8)),
               cmap="viridis", s=120, edgecolor="black", linewidth=0.6)
    for _, row in df.iterrows():
        ax.annotate(
            row["task_id"],
            xy=(row["gap_cl_eval"], row["gap_safe_eval"]),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
        )
    ax.axvline(0, color="grey", alpha=0.4, linewidth=0.8)
    ax.axhline(0, color="grey", alpha=0.4, linewidth=0.8)
    ax.set_xlabel(r"$V^{\pi^*_{\mathrm{cl}}}_{\mathrm{cl}}(s_0) - V^{\pi^*_{\mathrm{safe}}}_{\mathrm{cl}}(s_0)$"
                  + "\n(classical-eval regret of safe policy)")
    ax.set_ylabel(r"$V^{\pi^*_{\mathrm{safe}}}_{\mathrm{safe}}(s_0) - V^{\pi^*_{\mathrm{cl}}}_{\mathrm{safe}}(s_0)$"
                  + "\n(safe-eval advantage of safe policy)")
    ax.set_title("Dual-objective Pareto frontier (Family A stochastic shortlist)")
    ax.grid(True, alpha=0.3)

    # Panel B: grouped bar chart.
    ax = axes[1]
    x = np.arange(len(df))
    width = 0.2
    ax.bar(x - 1.5 * width, df["V_cl_pi_cl_s0"], width,
           label=r"$V^{\pi^*_{\mathrm{cl}}}_{\mathrm{cl}}$", color="tab:blue")
    ax.bar(x - 0.5 * width, df["V_cl_pi_safe_s0"], width,
           label=r"$V^{\pi^*_{\mathrm{safe}}}_{\mathrm{cl}}$", color="tab:cyan")
    ax.bar(x + 0.5 * width, df["V_safe_pi_cl_s0"], width,
           label=r"$V^{\pi^*_{\mathrm{cl}}}_{\mathrm{safe}}$", color="tab:red")
    ax.bar(x + 1.5 * width, df["V_safe_pi_safe_s0"], width,
           label=r"$V^{\pi^*_{\mathrm{safe}}}_{\mathrm{safe}}$", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(df["task_id"].values, rotation=30, fontsize=8)
    ax.set_ylabel(r"$V(s_0)$  (log scale)")
    ax.set_yscale("log")
    ax.set_title("Policy × metric value at start state")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    render(
        parquet_path=pathlib.Path("results/rl_VI_F/dual_eval.parquet"),
        out_path=pathlib.Path("figures/main/fig_dual_objective_eval.pdf"),
    )
    print("Wrote figures/main/fig_dual_objective_eval.pdf")
