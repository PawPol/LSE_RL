"""Shared rcParams and helpers for v10 paper figures (NeurIPS-style)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pin a deterministic source-time so PDFs are byte-stable on regeneration.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # 1970-01-01

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,  # TrueType (Type 42), NeurIPS submission requirement
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 100,
    "savefig.dpi": 300,
}


def apply_paper_style() -> None:
    plt.rcParams.update(PAPER_RC)


ROOT = Path("/Users/liq/Documents/Claude/Projects/LSE_RL")
LONG_CSV = ROOT / "results/adaptive_beta/tab_six_games/figures/v10/tables/v10_per_cell_per_method_per_gamma.csv"
DISPO_CSV = ROOT / "results/adaptive_beta/tab_six_games/figures/v10/tables/v10_h1_h2_h3_dispositions.csv"
FIG_DIR = ROOT / "results/adaptive_beta/tab_six_games/figures/v10"

DC_CELLS = {"DC-Long50", "DC-Medium20", "DC-Short10"}

# Beta grid (sorted)
BETA_GRID = [-2.0, -1.7, -1.35, -1.0, -0.75, -0.5, -0.35, -0.2, -0.1, -0.05,
             0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.35, 1.7, 2.0]


def load_long():
    import pandas as pd
    return pd.read_csv(LONG_CSV)
