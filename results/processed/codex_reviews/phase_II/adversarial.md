# Phase II Codex Adversarial Review

**Session:** bqm6fkd15  
**Base:** main  
**Branch:** phase-II/closing  
**Date:** 2026-04-17  
**Verdict:** needs-attention (no-ship)  
**Focus:** challenge whether stress families actually isolate the classical weakness and whether event logging is sufficient to drive Phase III schedule calibration per docs/specs/phase_II_stress_test_beta0_experiments.md.

## Summary

No-ship. The branch does not actually run several Phase II stress environments in the RL/DP harnesses, and the calibration export is missing spec-required fields/encoding for Phase III.

## Findings

### [critical] RL experiments bypass the stress wrappers

**File:** `experiments/weighted_lse_dp/runners/run_phase2_rl.py:553-607`  
**Confidence:** 0.99

`run_phase2_rl.py` builds both `wrapper_or_mdp` and `mdp_rl`, but then trains/evaluates `Core` on `mdp_rl` only. For wrapper-backed families (`chain_regime_shift`, `grid_regime_shift`, `grid_hazard`, `taxi_bonus_shock`), the factories return `mdp_rl` built from the pre-shift/base MDP, while the actual stress behavior lives in `wrapper_or_mdp`. The logger can inspect the wrapper, but the agent never interacts with it. Inference from the factory code: regime changes never occur during RL training/eval, hazard penalties are never injected, and taxi bonus shocks never fire. That means the reported degradation/event logging for those families can be materially wrong while still looking healthy on the happy path.

**Recommendation:** Run `Core` and evaluation against the wrapper-backed environment for wrapper tasks, or time-augment/wrap the stressed environment itself so training and logged transitions come from the same stressed dynamics.

### [high] DP results for `grid_hazard` and `taxi_bonus_shock` are computed on the unstressed base MDP

**File:** `experiments/weighted_lse_dp/runners/run_phase2_dp.py:99-101`  
**Confidence:** 0.96

The DP runner explicitly classifies `grid_hazard` and `taxi_bonus_shock` as wrapper tasks and then strips the wrapper via `wrapper._base` before planning. That removes the very hazard/bonus modifications that are supposed to expose the classical weakness, so the generated DP artifacts for those task families are baseline-task results under stress-task names. This undermines the core Phase II claim that the modified families isolate a failure mode, because two mandatory stress families are not actually what the planner solves.

**Recommendation:** Either encode these stress families directly into a finite MDP model that the DP planners can solve exactly, or exclude them from DP claims and outputs until the planner is operating on the stressed transition/reward kernel rather than `wrapper._base`.

### [high] Phase II calibration JSON does not satisfy the Phase III contract

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:548-703`  
**Confidence:** 0.95

The spec requires calibration files to include stagewise envelope estimates, per-stage quantiles of positive and negative aligned margins, event-conditioned margin statistics, and a family sign encoded as `+1`/`-1`. `aggregate_phase2.py` instead averages existing arrays into `*_mean` fields, emits no event-conditioned margin block, and derives `recommended_task_sign` as string values (`positive`/`negative`/`mixed`). Separately, `event_rates` are sourced only from `calibration_stats.npz`, but the RL runner computes tail/adaptation/event summaries outside that payload, so this export path cannot reliably provide the calibration inputs Phase III expects. The likely impact is that Phase III schedule calibration will either fail to consume these files or silently calibrate from the wrong summary statistics.

**Recommendation:** Change the calibration export to emit the spec-required fields explicitly: per-stage positive/negative aligned-margin quantiles, event-conditioned margin statistics, validated `+1`/`-1` family sign, and any event/adaptation scalars from a source the aggregator actually reads. Add a contract test that compares the JSON schema/content against `docs/specs/phase_II_stress_test_beta0_experiments.md`.

## Next steps

1. Wire RL training/evaluation through the stressed environments and add an integration test that proves a regime shift/hazard/bonus event changes observed rewards or transitions.
2. Stop DP from solving `wrapper._base` for wrapper-only stress families; either model the stressed kernel exactly or remove those DP outputs from the paper suite.
3. Add schema-level tests for `phase2/calibration/*.json` so missing Phase III fields or wrong sign encoding fail before artifacts are published.
