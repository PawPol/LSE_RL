# Phase II Codex Adversarial Review R2

**Session:** bewezccsc  
**Base:** main  
**Branch:** phase-II/closing (after BLOCKER A+B + MAJOR C+D fixes)  
**Date:** 2026-04-17  
**Verdict:** needs-attention (no-ship)  
**Focus:** challenge whether stress families actually isolate the classical weakness and whether event logging is sufficient to drive Phase III schedule calibration.

## Summary

No-ship. The Phase II calibration output is not a faithful input to Phase III: it collapses the wrong statistics, omits required event-conditioned margin data, and fails to log shortcut usage for the catastrophe family.

## Findings

### [high] Calibration JSON computes margin "quantiles" from averaged algorithm summaries instead of empirical classical margins

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:537-611`  
**Confidence:** 0.98

`build_calibration_json` first averages per-stage calibration arrays across all algorithms for a task, then derives `pos_margin_quantiles` / `neg_margin_quantiles` from those already-aggregated arrays. That is not the statistic the docs require. Phase II/III calibration expects empirical per-stage aligned-margin quantiles from classical outputs (`Q_{0.75}(a_t | a_t > 0)` and related quantities), but this code takes percentiles across algorithm-level means, which destroys the underlying transition distribution and makes the result depend on which algorithms happened to run. The likely impact is a materially wrong `m_t*` / informativeness profile, so Phase III beta schedules can be calibrated to artifacts of aggregation rather than the stress task itself.

**Recommendation:** Load the relevant transition-level margin arrays for the chosen classical source and compute per-stage conditional quantiles directly from those samples; do not compute quantiles over algorithm-mean arrays.

### [high] Phase III calibration artifact omits required event-conditioned margin statistics

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:647-675`  
**Confidence:** 0.99

The spec requires `event-conditioned margin statistics`, but the calibration JSON emitted here only includes `event_conditioned_return` plus a note saying the per-stage margin arrays are deferred to "future processing." That means the generated Phase II artifact is not actually sufficient to drive the Phase III schedule-calibration workflow. If this ships, downstream Phase III code either has to silently invent fallback behavior or proceed without event-conditioned margin structure, which undermines the claimed reproducibility of the calibration pipeline.

**Recommendation:** Extend aggregation to read event flags from transition logs and emit per-stage event-conditioned margin statistics in the calibration JSON, rather than a placeholder note.

### [medium] Catastrophe runs never log `shortcut_action_taken`, so the risky-path mechanism is not observable

**File:** `experiments/weighted_lse_dp/runners/run_phase2_rl.py:199-225`  
**Confidence:** 0.95

`_AutoEventLogger` detects jackpot, catastrophe, regime-shift, and hazard events, but it never marks `shortcut_action_taken`. For `chain_catastrophe`, that means the logs only capture catastrophic failures and miss the successful uses of the risky shortcut itself. Under the Phase II spec, the point of this stress family is the safe-vs-risky tradeoff; with this logger, that behavior is invisible in actual RL runs.

**Recommendation:** Teach `_AutoEventLogger` (or the catastrophe environment/wrapper) to flag `shortcut_action_taken` whenever the risky shortcut action is selected/executed, and validate that the flag is populated in real Phase II RL runs.

## Next steps

1. Rework `aggregate_phase2.py` to compute calibration statistics from transition-level classical data instead of algorithm-level means.
2. Add event-conditioned margin statistics to the Phase II calibration JSON and fail aggregation if they cannot be produced.
3. Instrument and test `shortcut_action_taken` on `chain_catastrophe` so the risky-path mechanism is actually observable in run artifacts.
