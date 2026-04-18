# Phase II Codex Adversarial Review R5-A

**Session:** review-mo3lol08-fchzjo (019d9dfe-a3e9-7700-abf1-7106fb4ce0e5)
**Base:** main
**Branch:** phase-II/closing (after R4 BLOCKER+MAJOR fixes)
**Date:** 2026-04-18
**Verdict:** needs-attention (no-ship)

## Summary

No-ship: one mandatory stress family is still the Phase I baseline, and the Phase III calibration output is not reliable because regime-shift DP runs are keyed under synthetic task names while the calibration JSON builder knowingly fabricates aligned-margin quantiles.

## Findings

### [high] `grid_sparse_goal` is not a stress task at all under the shipped defaults

**File:** `experiments/weighted_lse_dp/tasks/stress_families.py:267-272`
**Confidence:** 0.99

This factory explicitly states that with `goal_reward=1.0` and `step_penalty=0` it is identical to `grid_base`, and the Phase II paper suite uses exactly those defaults. That means one of the mandatory Phase II stress families does not introduce any new Bellman-target pathology, so it cannot support the claim that the suite isolates classical weaknesses. The likely impact is false confidence from a base-vs-stress comparison where one side is unchanged, plus wasted calibration/logging artifacts for a task that contributes no Phase III signal.

**Recommendation:** Redesign `grid_sparse_goal` so the default paper-suite instance differs from `grid_base` in reward support or spatial structure, then add a degradation test that fails if the stress instance is behaviorally indistinguishable from the Phase I base task.

### [high] Phase III calibration quantiles are knowingly fabricated from summary statistics

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:893-935`
**Confidence:** 0.98

The calibration builder does not pool raw aligned-margin samples. When transition-derived summaries exist, it averages per-group `q05/q50/q95` and then writes `q25=q05` and `q75=q95` as a 'best approximation'. Those fields are required inputs for Phase III schedule calibration, so this code can silently emit materially wrong positive/negative aligned-margin quantiles even when the underlying runs are correct. The impact is a mis-calibrated schedule driven by invented tail structure, which is hard to detect downstream because the JSON still looks schema-complete.

**Recommendation:** Preserve raw per-stage aligned-positive and aligned-negative samples, or a mergeable summary structure, and compute `q05/q25/q50/q75/q95` from that data instead of fabricating quartiles from `q05/q50/q95` envelopes.

### [medium] Regime-shift DP runs are emitted under synthetic task names that break family-level calibration

**File:** `experiments/weighted_lse_dp/runners/run_phase2_dp.py:592-649`
**Confidence:** 0.91

Warm-started regime-shift DP runs are written as `chain_regime_shift_pre_shift`, `chain_regime_shift_post_shift`, etc. The Phase II calibration contract is one file per task family, but downstream aggregation groups by the raw task name from `run.json`; these suffixed names no longer match the canonical family names used elsewhere. The likely impact is either fragmented calibration output (`*_pre_shift.json` / `*_post_shift.json`) or a hard failure when later code expects canonical families and sign mappings.

**Recommendation:** Keep `task` stable as the canonical family name and add a separate dimension such as `regime_phase=pre_shift|post_shift`; update aggregation to merge pre/post into a single family-level calibration document.

## Next steps

1. Redesign or remove `grid_sparse_goal` from the paper suite before treating Phase II as evidence of a classical weakness.
2. Fix the calibration pipeline so aligned-margin quantiles are computed from mergeable raw data, not approximated envelopes.
3. Normalize regime-shift DP outputs under canonical task-family names and rerun aggregation to confirm one calibration JSON per family.
