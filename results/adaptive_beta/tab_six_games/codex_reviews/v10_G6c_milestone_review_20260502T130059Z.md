# Phase VIII v10 G6c Milestone-Close Review

- Branch: `phase-VIII-tab-six-games-2026-04-30`
- HEAD SHA: `52dd9a341fc19c2dbcdfbeda2158389bb570dbb1`
- Review timestamp: `2026-05-02T13:00:59Z`
- Reviewer: Codex G6c

## Scope and Evidence Rules

I re-derived the review from the raw Tier I/Tier II/Tier III run artifacts, not from `v10_summary.md`. The manifest scan found 10,980/10,980 completed expected runs and no duplicate `(stage, gamma, cell, method, seed)` rows: Tier I 6300, Tier II 1680, Tier III 3000. Source manifests: `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier1_canonical/manifest.jsonl:1-4`, `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/manifest.jsonl:1-4`, `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier3_gamma_cell_coverage/manifest.jsonl:1-4`.

Numerical fields: return AUC is `np.trapezoid(metrics.npz::return, episode)` per `experiments/adaptive_beta/tab_six_games/analysis/beta_sweep_plots.py:84-87`; delayed-chain contraction AUC is `np.trapezoid(-log(metrics.npz::bellman_residual + 1e-8))` per `experiments/adaptive_beta/tab_six_games/metrics.py:839-866`. Strict alignment is `metrics.npz::alignment_rate`, produced from `aligned.mean()` with strict `>` semantics at `experiments/adaptive_beta/agents.py:246-252` and persisted at `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py:700-724`.

Binary `metrics.npz` files have no line numbers; file:line citations below point to sibling `run.json` metadata lines identifying the cell/method/seed and listing the metric arrays (`alignment_rate`, `return`, `bellman_residual`, `divergence_event`, `nan_count`, `q_abs_max`) read from the adjacent `metrics.npz`.

## Per-Hypothesis Verdict Table

| Hypothesis | Verdict | Category | One-line rationale |
| --- | --- | --- | --- |
| H1 | CONFIRMED, narrowly | MAJOR attribution caveat | AC-Trap at gamma=0.60 has bootstrapped best-beta CI `[+0.10, +0.10]` and paired AUC advantage CI `[+45.20, +212.00]`; SH-FMR `+0.35` is not robust. |
| H2 | REFUTED | GENUINE negative finding | Among evaluable -beta winners at gamma=0.95, 0/2 have ratio > 1; DC-Long50 has zero seed variance so Cohen d is undefined. |
| H3 | REFUTED | MAJOR narrative correction | Only 64/120 (53.3%) tuples confirm by final-episode alignment, far below the 80% threshold; last-200 robustness is 60/120 (50.0%). |

## (a) Alignment Diagnostic (H3)

Selection rule: headline cells use Tier II for all gamma values; non-headline cells use Tier III for gamma in `{0.60, 0.80, 0.90}` and Tier I for canonical gamma=0.95. The table reports the seed-mean final `alignment_rate[-1]` for the observed best method; `align_last200` is a robustness check. H3 is scored on `align_final >= 0.5` exactly as requested.

| gamma | cell | source | best beta | best method | metric field | mean metric | align_final | align_last200 | confirms? |
| ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| 0.60 | AC-FictitiousPlay | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 522257.50 | 0.000 | 0.000 | NO |
| 0.80 | AC-FictitiousPlay | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 522100.90 | 0.000 | 0.000 | NO |
| 0.90 | AC-FictitiousPlay | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 522178.50 | 0.000 | 0.000 | NO |
| 0.95 | AC-FictitiousPlay | Tier I | -0.20 | `fixed_beta_-0.2` | `metrics.npz::return` | 522319.85 | 0.900 | 0.901 | YES |
| 0.60 | AC-Inertia | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 703376.80 | 0.000 | 0.000 | NO |
| 0.80 | AC-Inertia | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 699790.20 | 0.000 | 0.000 | NO |
| 0.90 | AC-Inertia | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 695827.00 | 0.000 | 0.000 | NO |
| 0.95 | AC-Inertia | Tier I | -0.20 | `fixed_beta_-0.2` | `metrics.npz::return` | 703320.70 | 0.915 | 0.904 | YES |
| 0.60 | AC-SmoothedBR | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 531039.10 | 0.000 | 0.000 | NO |
| 0.80 | AC-SmoothedBR | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 531041.30 | 0.000 | 0.000 | NO |
| 0.90 | AC-SmoothedBR | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 531068.70 | 0.000 | 0.000 | NO |
| 0.95 | AC-SmoothedBR | Tier I | +0.00 | `vanilla` | `metrics.npz::return` | 530848.50 | 0.000 | 0.000 | NO |
| 0.60 | AC-Trap | Tier II | +0.10 | `fixed_beta_+0.1` | `metrics.npz::return` | 529077.80 | 0.050 | 0.049 | NO |
| 0.80 | AC-Trap | Tier II | +0.00 | `vanilla` | `metrics.npz::return` | 528951.20 | 0.000 | 0.000 | NO |
| 0.90 | AC-Trap | Tier II | +0.00 | `vanilla` | `metrics.npz::return` | 529039.80 | 0.000 | 0.000 | NO |
| 0.95 | AC-Trap | Tier II | +0.00 | `vanilla` | `metrics.npz::return` | 529043.40 | 0.000 | 0.000 | NO |
| 0.60 | DC-Branching20 | Tier III | +1.00 | `fixed_beta_+1.0` | `metrics.npz::return` | -2989.70 | 0.184 | 0.160 | NO |
| 0.80 | DC-Branching20 | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | -2989.80 | 0.000 | 0.000 | NO |
| 0.90 | DC-Branching20 | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | -2989.70 | 0.806 | 0.837 | YES |
| 0.95 | DC-Branching20 | Tier I | +0.00 | `vanilla` | `metrics.npz::return` | -2950.35 | 0.000 | 0.000 | NO |
| 0.60 | DC-Long50 | Tier II | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 182029.46 | 0.980 | 0.980 | YES |
| 0.80 | DC-Long50 | Tier II | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 180099.33 | 0.980 | 0.980 | YES |
| 0.90 | DC-Long50 | Tier II | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 178230.58 | 0.980 | 0.980 | YES |
| 0.95 | DC-Long50 | Tier II | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 177304.19 | 0.980 | 0.980 | YES |
| 0.60 | DC-Medium20 | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 181785.97 | 0.950 | 0.950 | YES |
| 0.80 | DC-Medium20 | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 180753.72 | 0.950 | 0.950 | YES |
| 0.90 | DC-Medium20 | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 180282.60 | 0.950 | 0.950 | YES |
| 0.95 | DC-Medium20 | Tier I | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 180094.60 | 0.950 | 0.950 | YES |
| 0.60 | DC-Short10 | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 181946.13 | 0.900 | 0.900 | YES |
| 0.80 | DC-Short10 | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 181535.07 | 0.900 | 0.900 | YES |
| 0.90 | DC-Short10 | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 181374.55 | 0.900 | 0.900 | YES |
| 0.95 | DC-Short10 | Tier I | -2.00 | `fixed_beta_-2.0` | `metrics.npz::bellman_residual` | 181308.61 | 0.900 | 0.900 | YES |
| 0.60 | MP-FiniteMemoryBR | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 84251.40 | 0.680 | 0.713 | YES |
| 0.80 | MP-FiniteMemoryBR | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 84175.00 | 0.860 | 0.855 | YES |
| 0.90 | MP-FiniteMemoryBR | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 84231.40 | 0.880 | 0.874 | YES |
| 0.95 | MP-FiniteMemoryBR | Tier I | -0.35 | `fixed_beta_-0.35` | `metrics.npz::return` | 84327.30 | 0.905 | 0.905 | YES |
| 0.60 | MP-HypothesisTesting | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 51724.60 | 0.530 | 0.402 | YES |
| 0.80 | MP-HypothesisTesting | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 51722.80 | 0.550 | 0.490 | YES |
| 0.90 | MP-HypothesisTesting | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 51667.80 | 0.640 | 0.539 | YES |
| 0.95 | MP-HypothesisTesting | Tier I | -0.75 | `fixed_beta_-0.75` | `metrics.npz::return` | 52073.80 | 0.755 | 0.666 | YES |
| 0.60 | MP-RegretMatching | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 34983.80 | 0.490 | 0.390 | NO |
| 0.80 | MP-RegretMatching | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 34633.40 | 0.390 | 0.391 | NO |
| 0.90 | MP-RegretMatching | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 34316.20 | 0.000 | 0.000 | NO |
| 0.95 | MP-RegretMatching | Tier I | -0.75 | `fixed_beta_-0.75` | `metrics.npz::return` | 34634.10 | 0.385 | 0.422 | NO |
| 0.60 | MP-Stationary | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 424.20 | 0.090 | 0.110 | NO |
| 0.80 | MP-Stationary | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 197.20 | 0.060 | 0.079 | NO |
| 0.90 | MP-Stationary | Tier III | +2.00 | `fixed_beta_+2.0` | `metrics.npz::return` | 363.60 | 0.060 | 0.071 | NO |
| 0.95 | MP-Stationary | Tier I | +1.35 | `fixed_beta_+1.35` | `metrics.npz::return` | 51.60 | 0.075 | 0.085 | NO |
| 0.60 | PG-BetterReplyInertia | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 141876.16 | 0.930 | 0.907 | YES |
| 0.80 | PG-BetterReplyInertia | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 141872.49 | 0.930 | 0.907 | YES |
| 0.90 | PG-BetterReplyInertia | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 141893.03 | 0.930 | 0.907 | YES |
| 0.95 | PG-BetterReplyInertia | Tier I | -0.05 | `fixed_beta_-0.05` | `metrics.npz::return` | 141900.24 | 0.920 | 0.910 | YES |
| 0.60 | PG-Congestion | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 89410.78 | 0.910 | 0.914 | YES |
| 0.80 | PG-Congestion | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 89419.15 | 0.910 | 0.915 | YES |
| 0.90 | PG-Congestion | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 89408.77 | 0.910 | 0.915 | YES |
| 0.95 | PG-Congestion | Tier I | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 89432.86 | 0.920 | 0.914 | YES |
| 0.60 | PG-CoordinationPotential | Tier III | +1.00 | `fixed_beta_+1.0` | `metrics.npz::return` | 66815.50 | 0.130 | 0.096 | NO |
| 0.80 | PG-CoordinationPotential | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 66816.80 | 0.760 | 0.758 | YES |
| 0.90 | PG-CoordinationPotential | Tier III | +1.00 | `fixed_beta_+1.0` | `metrics.npz::return` | 66843.90 | 0.080 | 0.052 | NO |
| 0.95 | PG-CoordinationPotential | Tier I | -0.35 | `fixed_beta_-0.35` | `metrics.npz::return` | 66776.25 | 0.890 | 0.900 | YES |
| 0.60 | PG-SwitchingPayoff | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 100311.30 | 0.000 | 0.000 | NO |
| 0.80 | PG-SwitchingPayoff | Tier III | +1.00 | `fixed_beta_+1.0` | `metrics.npz::return` | 100260.70 | 0.070 | 0.052 | NO |
| 0.90 | PG-SwitchingPayoff | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 100160.30 | 0.640 | 0.777 | YES |
| 0.95 | PG-SwitchingPayoff | Tier I | -0.10 | `fixed_beta_-0.1` | `metrics.npz::return` | 100218.00 | 0.915 | 0.903 | YES |
| 0.60 | RR-ConventionSwitch | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 140012.20 | 0.000 | 0.000 | NO |
| 0.80 | RR-ConventionSwitch | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 140012.60 | 0.000 | 0.000 | NO |
| 0.90 | RR-ConventionSwitch | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 140013.00 | 0.000 | 0.000 | NO |
| 0.95 | RR-ConventionSwitch | Tier I | -0.05 | `fixed_beta_-0.05` | `metrics.npz::return` | 140019.80 | 0.900 | 0.902 | YES |
| 0.60 | RR-HypothesisTesting | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 53673.80 | 0.270 | 0.385 | NO |
| 0.80 | RR-HypothesisTesting | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 53535.60 | 0.000 | 0.000 | NO |
| 0.90 | RR-HypothesisTesting | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 53368.60 | 0.630 | 0.607 | YES |
| 0.95 | RR-HypothesisTesting | Tier I | -0.75 | `fixed_beta_-0.75` | `metrics.npz::return` | 53636.10 | 0.720 | 0.734 | YES |
| 0.60 | RR-Sparse | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 4580.25 | 0.000 | 0.000 | NO |
| 0.80 | RR-Sparse | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 4580.25 | 0.000 | 0.000 | NO |
| 0.90 | RR-Sparse | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 4580.25 | 0.000 | 0.000 | NO |
| 0.95 | RR-Sparse | Tier I | +0.00 | `vanilla` | `metrics.npz::return` | 4567.73 | 0.000 | 0.000 | NO |
| 0.60 | RR-StationaryConvention | Tier II | -0.50 | `fixed_beta_-0.5` | `metrics.npz::return` | 56293.20 | 0.490 | 0.410 | NO |
| 0.80 | RR-StationaryConvention | Tier II | -0.50 | `fixed_beta_-0.5` | `metrics.npz::return` | 56276.40 | 0.740 | 0.697 | YES |
| 0.90 | RR-StationaryConvention | Tier II | -0.50 | `fixed_beta_-0.5` | `metrics.npz::return` | 56263.60 | 0.820 | 0.818 | YES |
| 0.95 | RR-StationaryConvention | Tier II | -0.50 | `fixed_beta_-0.5` | `metrics.npz::return` | 56266.80 | 0.850 | 0.846 | YES |
| 0.60 | RR-Tremble | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 50548.80 | 0.380 | 0.341 | NO |
| 0.80 | RR-Tremble | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 50483.20 | 0.520 | 0.405 | YES |
| 0.90 | RR-Tremble | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 50464.40 | 0.600 | 0.474 | YES |
| 0.95 | RR-Tremble | Tier I | -0.75 | `fixed_beta_-0.75` | `metrics.npz::return` | 50624.70 | 0.695 | 0.665 | YES |
| 0.60 | SH-FictitiousPlay | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 142571.60 | 0.910 | 0.906 | YES |
| 0.80 | SH-FictitiousPlay | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 142558.80 | 0.000 | 0.000 | NO |
| 0.90 | SH-FictitiousPlay | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 142604.20 | 0.900 | 0.906 | YES |
| 0.95 | SH-FictitiousPlay | Tier I | -0.35 | `fixed_beta_-0.35` | `metrics.npz::return` | 142627.65 | 0.900 | 0.906 | YES |
| 0.60 | SH-FiniteMemoryRegret | Tier II | +0.35 | `fixed_beta_+0.35` | `metrics.npz::return` | 106698.30 | 0.050 | 0.068 | NO |
| 0.80 | SH-FiniteMemoryRegret | Tier II | -0.35 | `fixed_beta_-0.35` | `metrics.npz::return` | 106697.40 | 0.930 | 0.914 | YES |
| 0.90 | SH-FiniteMemoryRegret | Tier II | -0.20 | `fixed_beta_-0.2` | `metrics.npz::return` | 106695.90 | 0.910 | 0.916 | YES |
| 0.95 | SH-FiniteMemoryRegret | Tier II | -1.70 | `fixed_beta_-1.7` | `metrics.npz::return` | 106653.40 | 0.910 | 0.895 | YES |
| 0.60 | SH-HypothesisTesting | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 83936.80 | 0.600 | 0.591 | YES |
| 0.80 | SH-HypothesisTesting | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 83496.10 | 0.720 | 0.765 | YES |
| 0.90 | SH-HypothesisTesting | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 83488.50 | 0.800 | 0.841 | YES |
| 0.95 | SH-HypothesisTesting | Tier I | -1.70 | `fixed_beta_-1.7` | `metrics.npz::return` | 83407.30 | 0.895 | 0.876 | YES |
| 0.60 | SH-SmoothedFP | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 66751.70 | 0.600 | 0.636 | YES |
| 0.80 | SH-SmoothedFP | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 66852.90 | 0.680 | 0.658 | YES |
| 0.90 | SH-SmoothedFP | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 66745.10 | 0.650 | 0.716 | YES |
| 0.95 | SH-SmoothedFP | Tier I | -0.50 | `fixed_beta_-0.5` | `metrics.npz::return` | 66729.70 | 0.895 | 0.898 | YES |
| 0.60 | SO-AntiCoordination | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 133302.20 | 0.850 | 0.863 | YES |
| 0.80 | SO-AntiCoordination | Tier III | +2.00 | `fixed_beta_+2.0` | `metrics.npz::return` | 133324.40 | 0.060 | 0.059 | NO |
| 0.90 | SO-AntiCoordination | Tier III | -2.00 | `fixed_beta_-2.0` | `metrics.npz::return` | 133294.90 | 0.860 | 0.886 | YES |
| 0.95 | SO-AntiCoordination | Tier I | -1.35 | `fixed_beta_-1.35` | `metrics.npz::return` | 133398.30 | 0.910 | 0.903 | YES |
| 0.60 | SO-BiasedPreference | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 109582.20 | 0.000 | 0.000 | NO |
| 0.80 | SO-BiasedPreference | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 108523.60 | 0.000 | 0.000 | NO |
| 0.90 | SO-BiasedPreference | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 107672.50 | 0.000 | 0.000 | NO |
| 0.95 | SO-BiasedPreference | Tier I | -0.10 | `fixed_beta_-0.1` | `metrics.npz::return` | 108406.70 | 0.905 | 0.901 | YES |
| 0.60 | SO-Coordination | Tier III | +1.00 | `fixed_beta_+1.0` | `metrics.npz::return` | 66815.50 | 0.130 | 0.096 | NO |
| 0.80 | SO-Coordination | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 66816.80 | 0.760 | 0.758 | YES |
| 0.90 | SO-Coordination | Tier III | +1.00 | `fixed_beta_+1.0` | `metrics.npz::return` | 66843.90 | 0.080 | 0.052 | NO |
| 0.95 | SO-Coordination | Tier I | -0.35 | `fixed_beta_-0.35` | `metrics.npz::return` | 66776.25 | 0.890 | 0.900 | YES |
| 0.60 | SO-TypeSwitch | Tier III | +1.00 | `fixed_beta_+1.0` | `metrics.npz::return` | 74131.30 | 0.150 | 0.103 | NO |
| 0.80 | SO-TypeSwitch | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 73405.10 | 0.000 | 0.000 | NO |
| 0.90 | SO-TypeSwitch | Tier III | +0.00 | `vanilla` | `metrics.npz::return` | 73570.10 | 0.000 | 0.000 | NO |
| 0.95 | SO-TypeSwitch | Tier I | +0.10 | `fixed_beta_+0.1` | `metrics.npz::return` | 73914.95 | 0.085 | 0.074 | NO |
| 0.60 | SO-ZeroSum | Tier III | +2.00 | `fixed_beta_+2.0` | `metrics.npz::return` | 287.30 | 0.120 | 0.113 | NO |
| 0.80 | SO-ZeroSum | Tier III | -1.00 | `fixed_beta_-1.0` | `metrics.npz::return` | 296.20 | 0.190 | 0.165 | NO |
| 0.90 | SO-ZeroSum | Tier III | +1.00 | `fixed_beta_+1.0` | `metrics.npz::return` | 362.50 | 0.110 | 0.105 | NO |
| 0.95 | SO-ZeroSum | Tier I | -0.75 | `fixed_beta_-0.75` | `metrics.npz::return` | 291.75 | 0.135 | 0.140 | NO |

| gamma | confirms / tuples (final) | confirms / tuples (last200 robustness) |
| ---: | ---: | ---: |
| 0.60 | 11/30 | 10/30 |
| 0.80 | 14/30 | 12/30 |
| 0.90 | 17/30 | 16/30 |
| 0.95 | 22/30 | 22/30 |

**H3 final verdict: REFUTED.** Final-episode confirmation is 64/120 = 53.3%; last-200 smoothing makes the result weaker, 60/120 = 50.0%. This is not close to the preregistered 80% threshold. The most damaging counterexample is the H1-positive AC-Trap arm itself: Tier II AC-Trap gamma=0.60 `fixed_beta_+0.1` has best mean AUC 529077.80 but final alignment 0.050 and last-200 alignment 0.049; source identity and metric arrays are in `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/asymmetric_coordination/AC-Trap/gamma_0.60/fixed_beta_+0.1/seed_0/run.json:32-68`.

## (b) H1 Confirmation: Tier II Paired Bootstrap at gamma=0.60

Bootstrap details: B=20,000 paired seed resamples over the five Tier II seeds. I report two checks because the preregistration phrase "CI above 0" can be read two ways: (1) bootstrap CI of the selected best beta sign, and (2) paired AUC advantage of the observed best arm over vanilla. H1 fires only if a headline cell has strictly positive best-beta CI; the AUC-advantage CI is an artifact check.

| cell | observed best beta | mean best AUC | mean vanilla AUC | best-beta 95% CI | P(best beta > 0) | best - vanilla AUC mean | paired 95% CI | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AC-Trap | +0.10 | 529077.80 | 528949.20 | [+0.10, +0.10] | 0.998 | 128.60 | [45.20, 212.00] | CONFIRMS H1 |
| SH-FiniteMemoryRegret | +0.35 | 106698.30 | 106570.80 | [-2.00, +0.35] | 0.514 | 127.50 | [-91.50, 360.00] | positive mean only; CI not strictly >0 |
| RR-StationaryConvention | -0.50 | 56293.20 | 56273.60 | [-0.75, -0.50] | 0.000 | 19.60 | [6.40, 30.80] | not positive |
| DC-Long50 | -2.00 | 182029.46 | 181569.81 | [-2.00, -2.00] | 0.000 | 459.65 | [459.65, 459.65] | not positive |

**H1 final verdict: CONFIRMED, narrowly.** AC-Trap confirms the sign-flip criterion. The AC-Trap best and vanilla arms are identified in `run.json` at `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/asymmetric_coordination/AC-Trap/gamma_0.60/fixed_beta_+0.1/seed_0/run.json:32-68` and `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/asymmetric_coordination/AC-Trap/gamma_0.60/vanilla/seed_0/run.json:32-68`; both arms read `metrics.npz::return` and neither has NaN/divergence in the five seed files. SH-FMR does **not** confirm: `+0.35` has a positive mean but its best-beta CI includes negative arms and its paired AUC advantage CI includes 0. Sources: `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/shapley/SH-FiniteMemoryRegret/gamma_0.60/fixed_beta_+0.35/seed_0/run.json:32-68` and `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/shapley/SH-FiniteMemoryRegret/gamma_0.60/vanilla/seed_0/run.json:32-68`.

This is **not** ratified as an alignment-mechanism result: the AC-Trap confirming arm has final alignment below 0.5, so the H1 empirical sign flip must be described as a small statistical surface feature unless a separate mechanism test is added.

## (c) Bifurcation Width Ratios (H2)

Effect size is the local project convention: Cohen d against vanilla using pooled seed-level AUC standard deviation, matching the M6 summary convention (`results/adaptive_beta/tab_six_games/M6_summary.md:63-69`). H2 eligibility is restricted to cells where the Tier II best beta at gamma=0.95 is negative.

| cell | eligible? | best beta gamma=0.95 | d95 | best beta gamma=0.60 | d60 | ratio | H2 result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AC-Trap | NO | +0.00 | 0.000 | +0.10 | 0.274 | NA | not eligible |
| SH-FiniteMemoryRegret | YES | -1.70 | 0.806 | +0.35 | 0.725 | 0.899 | not widened |
| RR-StationaryConvention | YES | -0.50 | 0.812 | -0.50 | 0.047 | 0.057 | not widened |
| DC-Long50 | YES | -2.00 | NA | -2.00 | NA | NA | INCONCLUSIVE (d undefined) |

**H2 final verdict: REFUTED.** Evaluable -beta winners: 0/2 have ratio > 1; DC-Long50 is not evaluable under Cohen d because all five seed AUC values are identical within each arm. Source identities for the gamma=0.95 negative winners: SH-FMR `fixed_beta_-1.7` at `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/shapley/SH-FiniteMemoryRegret/gamma_0.95/fixed_beta_-1.7/seed_0/run.json:32-68`, RR `fixed_beta_-0.5` at `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/rules_of_road/RR-StationaryConvention/gamma_0.95/fixed_beta_-0.5/seed_0/run.json:35-68`, and DC-Long50 `fixed_beta_-2.0` at `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/delayed_chain/DC-Long50/gamma_0.95/fixed_beta_-2.0/seed_0/run.json:30-68`.

I treat this as a genuine negative finding rather than a heatmap artifact because the two evaluable stochastic winners, SH-FMR and RR-StationaryConvention, have no NaN/nonfinite runs and no divergence in the winning/vanilla arms; the only unevaluable eligible cell is explicitly marked inconclusive rather than counted against H2.

## (d) Gamma-Beta Heatmap Anomaly Report

### Integrity Scan

| stage | expected runs | completed artifacts | missing artifacts | NaN/nonfinite runs | diverged runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tier I | 6300 | 6300 | 0 | 0 | 299 |
| Tier II | 1680 | 1680 | 0 | 0 | 125 |
| Tier III | 3000 | 3000 | 0 | 0 | 100 |

**Anomalies found: MAJOR.** No runs are missing and no NaN/nonfinite arrays were found, but divergence is not zero. Raw `run.json::diverged` / `metrics.npz::divergence_event` flags fire in 524/10,980 main-pass runs. This directly contradicts the preliminary memo line `divergence flag fires | 0` in `results/adaptive_beta/tab_six_games/v10_summary.md:17-22` and the `No bug signatures` statement at `results/adaptive_beta/tab_six_games/v10_summary.md:27-29`.

Divergence concentration by stage/cell:

| stage | cell | diverged runs |
| --- | --- | ---: |
| Tier I | AC-FictitiousPlay | 64 |
| Tier I | AC-Inertia | 70 |
| Tier I | AC-SmoothedBR | 46 |
| Tier I | AC-Trap | 49 |
| Tier I | DC-Long50 | 70 |
| Tier II | AC-Trap | 35 |
| Tier II | DC-Long50 | 90 |
| Tier III | AC-FictitiousPlay | 19 |
| Tier III | AC-Inertia | 20 |
| Tier III | AC-SmoothedBR | 12 |
| Tier III | AC-Trap | 14 |
| Tier III | DC-Long50 | 35 |

Representative divergent run citations: AC-Trap Tier II gamma=0.95 `fixed_beta_+1.0` seed 0 has `diverged: true` and lists `divergence_event` in the persisted metric arrays at `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/asymmetric_coordination/AC-Trap/gamma_0.95/fixed_beta_+1.0/seed_0/run.json:15-58`; DC-Long50 Tier II gamma=0.95 `fixed_beta_+0.35` seed 0 is also `diverged: true` at `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/delayed_chain/DC-Long50/gamma_0.95/fixed_beta_+0.35/seed_0/run.json:13-56`; AC-Inertia Tier I `fixed_beta_+0.5` seed 0 is `diverged: true` at `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier1_canonical/asymmetric_coordination/AC-Inertia/fixed_beta_+0.5/seed_0/run.json:15-58`.

### Tier II Surface Shape Diagnostics

| cell | gamma | range | max adjacent jump | jump/range | sign changes | diverged runs | best beta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AC-Trap | 0.60 | 105098.60 | 58585.10 | 0.557 | 8 | 0 | +0.10 |
| AC-Trap | 0.80 | 106545.10 | 48729.50 | 0.457 | 6 | 0 | +0.00 |
| AC-Trap | 0.90 | 106359.90 | 40592.40 | 0.382 | 5 | 11 | +0.00 |
| AC-Trap | 0.95 | 109270.20 | 50953.60 | 0.466 | 8 | 24 | +0.00 |
| SH-FiniteMemoryRegret | 0.60 | 28074.10 | 11641.50 | 0.415 | 10 | 0 | +0.35 |
| SH-FiniteMemoryRegret | 0.80 | 35044.80 | 13115.90 | 0.374 | 8 | 0 | -0.35 |
| SH-FiniteMemoryRegret | 0.90 | 36222.60 | 11492.60 | 0.317 | 7 | 0 | -0.20 |
| SH-FiniteMemoryRegret | 0.95 | 36315.90 | 8883.90 | 0.245 | 7 | 0 | -1.70 |
| RR-StationaryConvention | 0.60 | 38654.00 | 17420.00 | 0.451 | 3 | 0 | -0.50 |
| RR-StationaryConvention | 0.80 | 47305.60 | 21794.20 | 0.461 | 1 | 0 | -0.50 |
| RR-StationaryConvention | 0.90 | 47250.80 | 16304.20 | 0.345 | 1 | 0 | -0.50 |
| RR-StationaryConvention | 0.95 | 46806.80 | 23333.60 | 0.499 | 1 | 0 | -0.50 |
| DC-Long50 | 0.60 | 38683.73 | 32297.84 | 0.835 | 0 | 5 | -2.00 |
| DC-Long50 | 0.80 | 105782.15 | 73222.71 | 0.692 | 0 | 20 | -2.00 |
| DC-Long50 | 0.90 | 131419.27 | 53222.11 | 0.405 | 1 | 30 | -2.00 |
| DC-Long50 | 0.95 | 143181.52 | 49202.45 | 0.344 | 0 | 35 | -2.00 |

Interpretation: the Tier II heatmaps are not cleanly smooth in the requested sense. AC-Trap and DC-Long50 have large adjacent jumps that coincide with explicit divergent positive-beta arms; SH-FMR and RR are non-monotone but do not have divergence in the winning arms. I therefore classify the heatmap anomaly as a **MAJOR reporting/detector issue**, not a blocker for the AC-Trap H1 statistic itself.

## (e) New-Cell Sanity Checks

Tier I per-cell checks use all 21 beta arms x 10 seeds at gamma=0.95. `spread` is max-min over method mean AUC. The `summary line` column cross-checks the Tier I best-beta table in `v10_summary.md`.

| cell | sanity verdict | Tier I best beta | best mean AUC | spread | NaN/nonfinite runs | diverged runs | alignment all-zero? | summary line |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AC-Inertia | FAIL | -0.20 | 703320.70 | 144046.70 | 0 | 70 | NO | `v10_summary.md:87` |
| MP-RegretMatching | PASS | -0.75 | 34634.10 | 33288.40 | 0 | 0 | NO | `v10_summary.md:96` |
| MP-HypothesisTesting | PASS | -0.75 | 52073.80 | 45174.60 | 0 | 0 | NO | `v10_summary.md:95` |
| RR-ConventionSwitch | PASS | -0.05 | 140019.80 | 97347.70 | 0 | 0 | NO | `v10_summary.md:102` |
| RR-HypothesisTesting | PASS | -0.75 | 53636.10 | 41839.20 | 0 | 0 | NO | `v10_summary.md:103` |
| SH-SmoothedFP | PASS | -0.50 | 66729.70 | 205.25 | 0 | 0 | NO | `v10_summary.md:110` |
| SH-HypothesisTesting | PASS | -1.70 | 83407.30 | 13968.65 | 0 | 0 | NO | `v10_summary.md:109` |
| SO-ZeroSum | PASS | -0.75 | 291.75 | 295.90 | 0 | 0 | NO | `v10_summary.md:115` |
| SO-BiasedPreference | PASS | -0.10 | 108406.70 | 16074.40 | 0 | 0 | NO | `v10_summary.md:112` |
| PG-Congestion | PASS | -2.00 | 89432.86 | 489.50 | 0 | 0 | NO | `v10_summary.md:99` |
| PG-BetterReplyInertia | PASS | -0.05 | 141900.24 | 73118.92 | 0 | 0 | NO | `v10_summary.md:98` |

New-cell conclusion: 10/11 pass the requested pathology screen. AC-Inertia fails because 70/210 Tier I runs diverged, all in positive-beta arms `+0.35` through `+2.0` (10/10 seeds for each arm). Its best-beta/AUC cross-check matches the preliminary table at `results/adaptive_beta/tab_six_games/v10_summary.md:87`, but the divergence-free claim in the same memo is false. The other ten new cells show no all-tied AUC, no NaN/nonfinite arrays, no divergence, and nonzero alignment in at least one beta arm.

## Summary Verdict

**Overall milestone status: CONDITIONAL PASS.** The raw dispatch is complete and the AC-Trap gamma=0.60 sign flip is statistically confirmed from Tier II raw data. However, the milestone cannot be closed with the current narrative: H2 is refuted, H3 is refuted, SH-FMR is not a robust H1 confirmation, and the preliminary `v10_summary.md` misses 524 raw divergence flags. The AC-Trap H1 result should be reported as a small empirical sign flip, not as an alignment-condition vindication.

## Action Items for BLOCKER/MAJOR Findings

1. **MAJOR: Fix the divergence detector and summary.** Regenerate the V10.4 detector table by reading `run.json::diverged` and/or `metrics.npz::divergence_event.sum()` for every manifest row. Replace `v10_summary.md:21` and the `No bug signatures` claim at `v10_summary.md:27-29`; annotate positive-beta collapse arms rather than suppressing them.
2. **MAJOR: Correct the hypothesis disposition.** Record H1 as AC-Trap-only, SH-FMR as not CI-supported, H2 as refuted/evaluable 0/2, and H3 as refuted at 64/120 final confirmations. Remove or qualify the alignment-vindication language around `v10_summary.md:47-57` and `v10_summary.md:126-143`.
3. **MAJOR: Do not attribute AC-Trap +0.10 to the registered alignment diagnostic without a new mechanism test.** The winning AC-Trap arm has `alignment_rate` below 0.5 at end-of-training. A concrete follow-up would plot paired seed trajectories for `metrics.npz::return`, `metrics.npz::alignment_rate`, and `metrics.npz::effective_discount_mean` for AC-Trap gamma=0.60 beta in `{vanilla,+0.05,+0.10,+0.20}` and test whether the +0.10 gain survives a separate seed expansion.
