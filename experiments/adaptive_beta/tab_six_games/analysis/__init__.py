"""Phase VIII tab-six-games analysis subpackage.

Submodules:
    aggregate           — long-CSV aggregator (M5 W1.A; created in parallel).
    beta_sweep_plots    — Stage 1 β-grid sweep figures (M6).
    learning_curves     — Stage 2 baselines learning curves (M7).
    contraction_plots   — Stage 5 ContractionUCB figures (M10).
    sign_switching_plots — Stage 4 sign-switching composite figures (M9).
    safety_catastrophe  — Cross-cutting safety / clip / worst-window figures.

This file is intentionally empty of imports so that a partial worktree
(missing `aggregate.py` while W1.A is in flight) can still import each
plotting submodule independently.
"""
