# Phase II Codex Review R7

**Date:** 2026-04-18
**Base:** main
**Branch:** phase-II/closing (after R6 BLOCKER/MAJOR fixes + smoke-suite validation)
**Verdict:** needs-attention

## Summary

The core experiment/analysis pipeline has several data-schema mismatches: multiple plotting scripts cannot consume the JSON artifacts that the new aggregator writes, and the adaptation figure silently misinterprets checkpoint means as per-episode traces. Those issues make the advertised Phase II figure regeneration flow unreliable on real outputs.

## Findings

### [P1] Do not store checkpoint means as regime-shift episode traces

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:539-546`

`curves_agg["episode_returns"]` is populated from checkpoint-level evaluation means here, but `make_phase2_figures.py` later treats that field as a per-episode return series and compares it against `change_at_episode` in episode units. On real Phase II runs this makes the adaptation plot use x-values `0..n_checkpoints-1` while the change point is `200/300`, so the regime-shift marker is off-plot and the recovery curve is meaningless for every non-demo run.

### [P2] Emit visitation data or the heatmap figure will always be empty

**File:** `experiments/weighted_lse_dp/analysis/make_phase2_figures.py:530-531`

`fig_visitation_heatmaps` looks for `summary["visitation_counts"]`, but this patch never writes that key into any aggregated summary. In production mode every grid-task panel therefore falls into the `"No visitation data"` branch, so figure 11.1.4 cannot be regenerated from actual results.

### [P2] Preserve the top-level summary fields used by standalone plotters

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:1406-1416`

The new `summary.json` only exposes curve data under the nested `curves` object, but `plot_phase2_learning_curves.py` and `plot_phase2_adaptation.py` read top-level `checkpoints`, `disc_return_mean_per_seed`, `episode_returns`, and `change_at_episode`. With the schema written here those scripts immediately take their `no data` path on aggregated Phase II outputs, so the standalone plotting commands shipped in this patch are broken.

### [P2] Keep calibration JSON compatible with the new analysis scripts

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:1196-1212`

The calibration document written here lacks the fields that the new standalone analysis scripts consume: `plot_phase2_heatmaps.py` expects `per_state_margin_q50`/`per_stage`, `plot_phase2_return_distributions.py` expects `per_seed_returns` or `return_quantiles`, and `plot_phase2_margin_quantiles.py` expects `per_stage`. As a result those scripts cannot render from the calibration JSONs produced by this same patch and will skip or show empty panels on real data.
