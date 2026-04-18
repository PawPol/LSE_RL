# Phase II Codex Standard Review R4

**Session:** review-mo3l1ks8-jradcp (019d9dee-4ac1-71c3-82c1-a96d2b251624)
**Base:** main
**Branch:** phase-II/closing (after R3 BLOCKER+MAJOR fixes)
**Date:** 2026-04-18

## Summary

Several Phase II reporting paths are producing incorrect metrics: DP convergence is normalized against each algorithm's own final table, RL stress metrics are computed from exploratory training trajectories, and aggregated figure inputs are populated with stagewise calibration means instead of true learning/adaptation curves.

## Findings

### [P1] Use a true VI baseline for `supnorm_to_exact`

**File:** `experiments/weighted_lse_dp/runners/run_phase2_dp.py:470-473`

When `v_exact` is `None`, this falls back to `planner.V`, i.e. the same algorithm's final table, not an exact reference solution. Because `_run_single()` passes `v_exact=None` for essentially every run, all non-VI `supnorm_to_exact` curves are measured against themselves and will misleadingly collapse to zero at convergence even if the planner stopped far from the optimal value function.

**Recommendation:** Compute a VI reference solution once per task at the start of each group, then pass it as `v_exact` to every planner in that group.

### [P1] Derive Phase II tail/adaptation metrics from final eval rollouts

**File:** `experiments/weighted_lse_dp/runners/run_phase2_rl.py:700-752`

`episode_returns` is computed from `transitions_payload`, which comes from the training callback attached to `Core.learn()`. For jackpot/catastrophe/hazard/regime-shift tasks, that means CVaR, event-conditioned return, and adaptation lag are all measured on the full epsilon-greedy training history instead of the learned policy, so the reported stress metrics change with exploration and training length rather than final performance. This also leaves the configured `eval_episodes_final` budget unused.

**Recommendation:** After training completes, run a separate eval rollout with epsilon=0 for `eval_episodes_final` episodes and compute all stress metrics from those transitions.

### [P2] Export real learning/adaptation curves instead of stagewise means

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:1091-1109`

The summary writer is synthesizing `curves.mean_return` and `curves.episode_returns` from calibration `stage`/`reward_mean` arrays, which are per-stage reward statistics, not checkpoint learning curves or per-episode returns. As a result, the Phase II plotting scripts read plausible-looking data but generate learning and adaptation figures that do not reflect training dynamics at all.

**Recommendation:** Store actual per-episode or per-checkpoint return arrays during training and emit them as `curves.*` in summary.json.
