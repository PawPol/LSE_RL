# Phase II Codex Adversarial Review R7

**Date:** 2026-04-18
**Base:** main (full codebase + spec)
**Branch:** phase-II/closing (after R6 BLOCKER/MAJOR fixes + smoke-suite validation)
**Focus:** "challenge whether stress families actually isolate the classical weakness and whether event logging is sufficient to drive Phase III schedule calibration"
**Verdict:** needs-attention

## Summary

The patch introduces calibration-output gaps that block the nonstationary families from producing the single, change-point-aware summaries required for Phase III, and one mandatory stress family no longer matches the Phase II spec. Those issues materially affect both the claimed stress isolation and the usefulness of the logged calibration artifacts.

## Findings

### [P1] Preserve a single calibration file per regime-shift family

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:1437-1447`

This writes calibration JSONs from the raw task labels, so warm-start DP outputs become `chain_regime_shift_pre_shift.json` / `chain_regime_shift_post_shift.json` instead of one `chain_regime_shift.json` family summary. That breaks spec §12's "one calibration summary file per stress-task family" requirement and means Phase III never sees the pre/post regime-shift statistics in one place, even though `canonical_task_family` is already persisted for exactly this regrouping step.

### [P1] Carry the regime-shift index into aggregated calibration output

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:142-148`

`run_phase2_rl.py` stores `regime_shift_episode` in each seed's `calibration_stats.npz`, but the aggregator never includes that scalar in any merged block here. As a result the emitted calibration JSON for `*_regime_shift` tasks has no change-point metadata, and downstream consumers cannot align pre/post statistics to the actual shift episode even though spec §§8.2 and 12 require that index.

### [P2] Keep `grid_sparse_goal` goal-only instead of adding shaping

**File:** `experiments/weighted_lse_dp/tasks/stress_families.py:261-278`

The Phase II spec defines `grid_sparse_goal` as the Phase I grid with reward only at the goal and no per-step shaping. Changing the default to `step_penalty=-0.05` turns this family into a dense negative-cost task, so it no longer isolates sparse-reward propagation as intended and its logged margins/variance will calibrate Phase III against a different failure mode than the paper claims.

**Note:** This is a repeat of R6-D1 DISPUTE. R5 adversarial review [high] found that goal-only `grid_sparse_goal` was behaviourally identical to `grid_base`. Spec says goal-only. Requires user resolution.

### [P2] Persist risky-shortcut frequency for catastrophe tasks

**File:** `experiments/weighted_lse_dp/runners/run_phase2_rl.py:785-799`

Although `shortcut_action_taken` is logged per transition, the catastrophe path here only reduces `catastrophe_event` to episode-level metrics. That means none of the run artifacts ever expose the required safe-path/risky-path selection frequency from spec §10.2, so you cannot tell whether a policy avoided the shortcut or merely took it and got lucky; this weakens the evidence that `chain_catastrophe` isolates tail-risk behavior.
