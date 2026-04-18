# Phase II Codex Adversarial Review R5-B

**Session:** (019d9dfe-2651-7441-9975-114bcd2b1e8a)
**Base:** main
**Branch:** phase-II/closing (after R4 BLOCKER+MAJOR fixes)
**Date:** 2026-04-18
**Verdict:** needs-attention (no-ship)

## Summary

No-ship. One mandatory stress family is still excluded from classical DP, and the Phase II calibration artifact still fabricates aligned-margin quantiles instead of exporting the statistics Phase III is supposed to consume. The logging contract also does not enforce the new event arrays, so missing Phase II diagnostics can slip through validation unnoticed.

## Findings

### [high] Mandatory `grid_hazard` family is still skipped for DP, so the stress suite does not cover all classical baselines

**File:** `experiments/weighted_lse_dp/runners/run_phase2_dp.py:570-579`
**Confidence:** 0.97

`run_phase2_dp.py` returns success-with-skip for tasks marked RL-only, and `grid_hazard` is in that set. The Phase II spec treats the grid family as mandatory and requires classical `beta=0` baselines to be rerun on all mandatory stress-task families. As written, this branch produces no DP artifact for `grid_hazard`, so the claimed stress evidence is incomplete: the family does not isolate a classical DP weakness because DP is never run on the stressed task at all.

**Recommendation:** Either encode `grid_hazard` in a DP-solvable kernel and run the DP baselines, or explicitly remove it from the mandatory suite/config and downstream claims so Phase II does not present incomplete coverage as finished evidence.

### [high] Phase III calibration JSON invents aligned-margin quantiles from the wrong distribution

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:893-935`
**Confidence:** 0.99

The aggregator admits it does not have the per-transition positive/negative aligned samples needed for the required calibration output, then approximates `pos_margin_quantiles` and `neg_margin_quantiles` by averaging raw-margin `q05/q50/q95` summaries and even duplicates `q05` into `q25` and `q95` into `q75`. That is not a faithful export of the per-stage quantiles of positive and negative aligned margins required by the Phase II spec.

**Recommendation:** Preserve raw `aligned_positive` and `aligned_negative` samples, or at minimum exact per-stage quantile sketches for those arrays, through aggregation. If those statistics cannot be produced exactly, fail calibration export instead of emitting approximations under the spec-required field names.

### [medium] Event logging is not part of the enforced transitions schema, so required Phase II diagnostics can disappear without tripping validation

**File:** `experiments/weighted_lse_dp/common/schemas.py:482-500`
**Confidence:** 0.90

Phase II adds required event arrays (`jackpot_event`, `catastrophe_event`, `regime_post_change`, `hazard_cell_hit`, `shortcut_action_taken`), but `set_transitions()` only requires the Phase I transition keys and accepts everything else as optional extras. A run can pass repository-level schema validation while silently omitting the very event logs the Phase II spec says are needed for tail-risk and calibration analysis.

**Recommendation:** Promote the Phase II event arrays into an explicit required transitions schema for Phase II runs, and make `validate_transitions_npz()` enforce those keys when the phase/task family requires stress diagnostics.

## Next steps

1. Unblock mandatory-family coverage by either adding a DP formulation for `grid_hazard` or narrowing the Phase II mandatory suite and claims.
2. Rework calibration export so `pos_margin_quantiles` and `neg_margin_quantiles` are computed from true aligned-margin data, not reconstructed from raw-margin summaries.
3. Tighten the schema/validator contract so missing Phase II event arrays fail fast during run writing and verification.
