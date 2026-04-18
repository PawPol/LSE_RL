# Phase II Codex Adversarial Review R6

**Session:** 019d9e19-c138-70f1-833e-44a3ef10cfd1
**Base:** main (spec + implementation review, no diff restriction)
**Branch:** phase-II/closing (after R5 BLOCKER+MAJOR fixes)
**Date:** 2026-04-18
**Focus:** "challenge whether stress families actually isolate the classical weakness and whether event logging is sufficient to drive Phase III schedule calibration"
**Verdict:** needs-attention

## Summary

The patch introduces one calibration-corrupting aggregation change for regime-shift DP runs and one stress-task change that no longer matches the Phase II benchmark specification. Together they make the Phase II outputs unreliable as calibration inputs for Phase III.

## Findings

### [high] Keep pre- and post-shift DP runs out of the same aggregate group

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:208-221`
**Confidence:** high

Replacing `task` with `canonical_task_family` here makes `*_pre_shift` and `*_post_shift` DP runs for the same regime-shift seed collapse into one `(suite, task, algorithm)` group. `aggregate_group()` then counts both records as separate seeds and averages their curves/calibration arrays together, so the calibration JSON for `chain_regime_shift`/`grid_regime_shift` mixes pre-change and post-change statistics instead of preserving the post-change signal Phase II and Phase III depend on.

**Recommendation:** Keep `task` as the discriminating key (i.e. keep `chain_regime_shift_pre_shift` and `chain_regime_shift_post_shift` as separate groups), and add `canonical_task_family` only as a metadata field within the group record — not as the grouping key itself. Downstream calibration export should then select the `_post_shift` group as the authoritative Phase III input.

### [medium] Preserve the goal-only reward semantics of `grid_sparse_goal`

**File:** `experiments/weighted_lse_dp/tasks/stress_families.py:268-278`
**Confidence:** medium

The Phase II spec defines `grid_sparse_goal` as the same grid task with reward only at the goal and no per-step shaping. This change turns the default paper-suite task into a step-cost MDP by subtracting `0.05` on every non-goal transition, so the benchmark is no longer isolating sparse-reward propagation; it is measuring a different failure mode entirely. That changes both the reported Phase II stress result and the calibration statistics that Phase III will inherit for this family.

**Recommendation:** Revert the step-penalty default to 0 (or remove it entirely). If a step penalty is desired as a separate stress variant, it should be a distinct named task in the suite — not a silent default change to the existing `grid_sparse_goal` benchmark.

**Note:** This finding is in direct tension with adversarial R5-A finding [high] which identified that `grid_sparse_goal` with default params was behaviourally identical to `grid_base`. The question of whether the stress mechanism should be sparse-reward-only (spec) or step-penalty (R5-A fix) requires user/spec resolution before proceeding.

## Next steps

1. **Regime-shift grouping (BLOCKER)**: revert the `_discover_runs` grouping key to use `task` (not `canonical_task_family`) so pre/post-shift DP statistics remain separate. Add `canonical_task_family` as a metadata field only.
2. **grid_sparse_goal semantics (DISPUTE)**: surface the spec tension to the user — spec says goal-only, R5-A review said identical-to-base is wrong. User must decide the stress mechanism before this can close.
