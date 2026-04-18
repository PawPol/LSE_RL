# Phase II Codex Adversarial Review R4

**Session:** review-mo3l48hq-1ncnzo (019d9df0-27ad-7740-9c07-7c4e7b17a9d5)
**Base:** main
**Branch:** phase-II/closing (after R3 BLOCKER+MAJOR fixes)
**Date:** 2026-04-18
**Verdict:** needs-attention (no-ship)
**Focus:** challenge whether stress families actually isolate the classical weakness and whether event logging is sufficient to drive Phase III schedule calibration.

## Summary

No-ship: the Phase II calibration export is still collapsing or biasing the exact statistics Phase III is supposed to consume, so the stress-suite evidence is not trustworthy enough to calibrate schedules from.

## Findings

### [high] Per-stage aligned-margin quantiles are computed from stage means, not from the underlying margin distribution

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:721-776`
**Confidence:** 0.98

`build_calibration_json()` advertises spec-12 quantiles, but it builds `pos_margin_quantiles` / `neg_margin_quantiles` from `aligned_positive_mean` and `aligned_negative_mean`. Those arrays are already per-stage means, so the percentiles here are percentiles of seed/algorithm averages, not quantiles of aligned margins themselves. For rare-event families this will erase the tails that Phase III is supposed to calibrate against, and can make a task look much less extreme than the logged transitions actually were.

**Recommendation:** Compute these quantiles from pooled per-transition aligned margins at each stage (from `transitions.npz`, or an equivalent exact-DP margin sample), and reserve `aligned_*_mean` for means only.

### [high] `empirical_r_max` is a heuristic from means/stds, so rare jackpot/catastrophe rewards can be understated

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:778-792`
**Confidence:** 0.96

The calibration JSON's `empirical_r_max` is derived from `reward_mean` and `reward_std` using `mean +/- 2*std`, not from any actual observed or exact reward maximum. In the Phase II families the important rewards are intentionally rare spikes; a 2-sigma proxy can sit well below the true absolute reward and understate the envelope that Phase III uses for safe schedule construction. This is especially risky for jackpot/catastrophe tasks, where the whole point is tail events.

**Recommendation:** Track the true absolute reward maximum per seed from raw transitions or exact model tables and aggregate it with `max`, not with a moment-based approximation.

### [medium] Event-conditioned stagewise margins are merged with an unweighted mean-of-means

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:826-856`
**Confidence:** 0.90

The aggregation keeps per-stage event counts, but when it builds the task-level `event_conditioned_margins.stagewise` block it averages each group's mean margin with equal weight. A run with 2 event hits and a run with 200 event hits influence the exported curve equally. On these stress families the event process is deliberately sparse, so this can materially skew the calibration profile toward noisy small-sample seeds or algorithms.

**Recommendation:** Pool the underlying event-conditioned samples across groups, or at minimum compute weighted means using `event_conditioned_margin_count` for each stage.

## Next steps

1. Rebuild the Phase II calibration JSON from raw per-transition/per-stage margin samples rather than from aggregated means.
2. Replace the `empirical_r_max` estimator with a true observed/model-derived maximum reward envelope.
3. Re-export the affected calibration files and re-check any Phase III schedule logic that consumed the current JSONs.
