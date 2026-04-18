# Phase II Codex Review R6

**Session:** 019d9e15-0efb-7493-a93b-fce383ba3f23
**Base:** main
**Branch:** phase-II/closing (after R5 BLOCKER+MAJOR fixes)
**Date:** 2026-04-18
**Verdict:** needs-attention

## Summary

The new Phase II reporting pipeline has multiple data-contract mismatches with its own figure generators. As written, at least the adaptation, heatmap, and margin-quantile figures are incorrect or non-regenerable from real outputs.

## Findings

### [high] Preserve episode-level returns for regime-shift adaptation plots

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:536-542`
**Confidence:** high

`fig_adaptation_plots()` consumes `summary["curves"]["episode_returns"]` as a per-episode trace and overlays the configured change-point episode on that axis, but this aggregation path fills `episode_returns` with checkpoint-level `disc_return_mean` values instead. For `chain_regime_shift`/`grid_regime_shift`, that collapses hundreds of training episodes into a few dozen checkpoints, so the rolling adaptation curve and change-point marker are plotted on incompatible scales and the Phase II adaptation figure is misleading even when the underlying runs are correct.

**Recommendation:** Store episode-level return traces (not checkpoint means) in a separate key (e.g. `episode_returns_raw`) for regime-shift tasks, or ensure `fig_adaptation_plots()` reads from the checkpoint key with a matching x-axis.

### [high] Emit visitation counts before generating production heatmaps

**File:** `experiments/weighted_lse_dp/analysis/make_phase2_figures.py:530-531`
**Confidence:** high

The new heatmap figure loader looks for `summary.json["visitation_counts"]`, but the aggregation pipeline never writes that field into the per-task summaries. In non-demo mode this means every grid heatmap goes down the `"No visitation data"` branch, so Figure 11.1.4 cannot be regenerated from actual Phase II outputs.

**Recommendation:** Add visitation count aggregation to `aggregate_group()` and write it to the per-task `summary.json`, or update `fig_visitation_heatmaps()` to look for the data in the correct field name.

### [medium] Export full margin quantiles instead of positive-only quantiles

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:1157-1171`
**Confidence:** medium

This top-level `margin_quantiles` alias is built from `pos_margin_quantiles`, but `fig_margin_quantiles()` treats it as the quantiles of the raw margin distribution. Any task with substantial negative margins will therefore have its lower tail clipped away in the Phase II margin figure, changing both the median and the q05/q95 ribbon that the paper is supposed to show.

**Recommendation:** Build the top-level `margin_quantiles` alias from the raw margin quantiles (q05..q95 of the full `margin_beta0` distribution), not from the positive-only `pos_margin_quantiles`.

## Next steps

1. Fix regime-shift adaptation curve storage so `episode_returns` contains per-episode data, not checkpoint means.
2. Add visitation count aggregation and write to `summary.json` so heatmap figures are regenerable from real outputs.
3. Fix `margin_quantiles` top-level alias to use raw margin quantiles (not pos-only).
