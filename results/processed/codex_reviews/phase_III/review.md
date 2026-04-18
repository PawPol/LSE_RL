# Codex Standard Review — Phase III

Session ID: 019da11e-dd19-7733-94ac-cf226657a8c9
Base: 4fdbf0d (Phase II close commit)
Branch: phase-III/closing
Status: completed (2m 34s)

## Summary

The patch introduces at least one correctness issue in Phase III logging and two API/behavior regressions in the new safe DP planners. As written, it can produce misaligned safe transition artifacts and ignores documented warm-start inputs.

## Findings

- [P1] Stop logging `agent.swc.last_*` from `callback_step` — experiments/weighted_lse_dp/common/callbacks.py:262-285
  `SafeTransitionLogger` is reading `agent.swc.last_*` inside MushroomRL's `callback_step`, but `Core._run()` invokes that callback **before** `agent.fit(dataset)` updates the agent. In practice this makes every `safe_*` row stale by one transition during training (the first row is just the zero-initialized defaults), and during evaluation the fields never correspond to the sampled transition at all because no update happens there. That corrupts the Phase III transition logs and any calibration/aggregation built from them.

- [P2] Preserve the documented warm-start value table in `run()` — mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_value_iteration.py:346-349
  `v_init` is copied into `self.V` in `__init__`, but `run()` immediately zeroes `self.V` before the first sweep, so callers never get the advertised warm start. Any experiment that tries to seed VI from a previous solution will silently behave like a cold start, which also affects residual/timing diagnostics. The same reset pattern appears in the other safe DP planners that expose `v_init`.

- [P2] Validate schedule horizon before constructing the planner — mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_value_iteration.py:184-187
  This constructor never checks that `schedule.T` matches the MDP horizon. If the schedule is too short, the first backward pass will fail with an index error when it reaches a missing stage; if it is too long, the extra betas are silently ignored. `SafeWeightedPolicyIteration` already guards against this, so the other safe planners should reject mismatched schedules up front as well.
