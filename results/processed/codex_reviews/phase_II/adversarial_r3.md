# Phase II Codex Adversarial Review R3

**Session:** review-mo2tjcbx-sp1s50 (019d9b2d-4cc0-7e51-b071-cc3a5ca05cfa)
**Base:** main
**Branch:** phase-II/closing (after R2 BLOCKER+MAJOR+MINOR fixes)
**Date:** 2026-04-17
**Verdict:** needs-attention (no-ship)
**Focus:** challenge whether stress families actually isolate the classical weakness and whether event logging is sufficient to drive Phase III schedule calibration.

## Summary

No-ship. The Phase II stress/calibration path still violates the spec in ways that can invalidate the claimed weakness isolation and miscalibrate Phase III schedules.

## Findings

### [high] `chain_catastrophe` does not implement the spec's safe-vs-risky tradeoff

**File:** `experiments/weighted_lse_dp/tasks/stress_families.py:356-423`
**Confidence:** 0.95

At `risky_state`, the code replaces action `0` with catastrophe-or-shortcut and leaves action `1` unchanged. In this chain family, action `1` is the left/backward move, not a slower forward safe path. That means any policy that ever reaches the goal must still eventually take the risky action at the same state; there is no modeled alternative that trades mean return for lower tail risk. This breaks the Phase II requirement to compare a risky shortcut against a slower safe path and makes metrics like safe-path selection frequency uninterpretable. The degradation you measure here is therefore partly an artifact of forced exposure, not evidence that classical methods underweight rare catastrophic outcomes.

**Recommendation:** Redesign the family so the agent has a genuine non-risky route to the goal at the risky decision point, then add a test proving a safe policy can reach the goal without ever taking the risky shortcut.

### [high] Phase III sign calibration silently defaults to `+1`, can flip catastrophe-family schedules to wrong sign

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:934-994`
**Confidence:** 0.97

`recommended_task_sign` is derived only from stage-0 positive/negative margin means and falls back to `+1` on missing or ambiguous data. The spec requires one sign per experiment family based on family semantics (`+1` jackpot/positive shift, `-1` catastrophe). Using stage 0 is brittle because many stress events occur later, and the default-to-optimistic branch can silently mislabel negative-shock families. That would drive Phase III schedule calibration with the wrong sign, directly undermining the operator behavior the schedule is supposed to induce.

**Recommendation:** Derive the sign from `task_family`/`stress_type` or explicit config metadata, and fail hard if the family is unknown instead of defaulting to `+1`. Use margin statistics only as a validation check, not the source of truth.

### [medium] Calibration JSON averages across all classical algorithms, washing out the sharpest classical failure mode

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:609-675`
**Confidence:** 0.90

The builder explicitly pools every algorithm group for a task family and averages their calibration arrays into one task-level profile. The spec calls for stagewise envelopes and margin summaries from the classical solution or best classical approximation. Averaging DP planners and online RL traces together produces a synthetic profile that may match no actual classical baseline and can mute exactly the tail/alignment structure Phase III needs to calibrate against.

**Recommendation:** Choose a single reference source per family for calibration generation: exact DP on model-based tasks, otherwise the best validated classical approximation under a documented rule. Keep cross-algorithm summaries separate from the Phase III calibration JSON.

## Next steps

1. Redesign `chain_catastrophe` so a safe policy exists and re-run the degradation evidence for that family.
2. Make calibration sign explicit per task family and regenerate all Phase II calibration JSON outputs.
3. Rebuild Phase III calibration inputs from a designated reference baseline instead of algorithm averages.
