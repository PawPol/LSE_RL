# Codex Standard Review — Phase III R6 (post-merge)

Session ID: 019da2f1-2830-72e0-85cd-06bd5a4f5040
Base: 859688c (Phase II merge commit)
Branch: main (phase-III/closing already merged)
Date: 2026-04-18

## Summary

Two P2 findings. No BLOCKERs. Operator math and certification chain are
correct. Issues are in observability logging and a hard-coded n_base table
in the RL runner.

## Findings

### [P2] safe_margin logged as greedy max-Q v_next, not policy-expected v_next
**File**: experiments/weighted_lse_dp/runners/run_phase3_rl.py:372-374

For stress-task safe RL runs with `SafeExpectedSARSA`, this logs `safe_margin`
as `reward - v_next`, where `v_next = self._v_next_beta0[-1]` is the greedy
max-Q bootstrap. The safe operator used the policy-expected bootstrap and
stored the correct margin in `swc.last_margin`. As a result, `safe_margin`
quantiles in Phase III aggregation are wrong for all RL ExpectedSARSA runs.
(The fix was applied in callbacks.py:SafeTransitionLogger but not in the
inline logger class in run_phase3_rl.py.)

### [P2] n_base hard-coded in _N_BASE lookup table
**File**: experiments/weighted_lse_dp/runners/run_phase3_rl.py:767-768

`n_base = _N_BASE[task]` uses a hard-coded lookup. If any task's base state
count differs from the table (config change, absorbing state added/removed),
safe TD decodes the wrong stage from the augmented state, corrupting safe
updates and logging. This is the same anti-pattern that was a BLOCKER in
Phase II R8. Current run values appear correct (64/64 passed), but the
structural risk remains.

## Previous round findings (R5, all resolved)
All R5 BLOCKERs resolved before merge to main:
- expm1/log1p → _EPS_BETA=1e-8 + logaddexp
- beta_raw_unclipped ablation cert bypass
- safe_margin source (callbacks.py fixed)
- schedule_file override honoured
- DP rho derived from eff_discount
Round: R5 (post R4-fix commit)
Status: completed

## Verdict

Needs-attention: three P2 (MAJOR) issues found in the Phase III pipeline —
safe margin logging is inconsistent for ExpectedSARSA, per-task schedule_file
overrides are silently ignored, and DP rho aggregates are all-NaN.

## Findings

- [P2] Log safe margins against the actual bootstrap target — experiments/weighted_lse_dp/common/callbacks.py:292-306
  `self._v_next_beta0[-1]` comes from `TransitionLogger`, which always stores
  `max_a Q(s',a)`. That matches `SafeQLearning`, but for `SafeExpectedSARSA`
  (and `SafeTD0`) the safe target was computed with a policy expectation instead,
  so `safe_margin` and `safe_td_error` are silently inconsistent with
  `swc.last_target`. Corrupts Phase III diagnostics and `target_stats.npz` for
  those runs even though the learning update itself is correct.
  Recommendation: Read `v_next` from `swc.last_target`-consistent quantity (e.g.
  the policy-expected value) for ExpectedSARSA/TD0 agents rather than the
  greedy max-Q.

- [P2] Respect per-task schedule_file overrides in Phase III runs — experiments/weighted_lse_dp/runners/run_phase3_rl.py:778-785
  The suite config carries a `schedule_file` per task, but the runner hard-codes
  `<schedule_dir>/<task>/schedule.json`. Any ablation or custom suite that points
  a task at a specific ablation schedule will silently use the default schedule
  and record misleading provenance/metrics. The DP runner has the same issue.
  Recommendation: Read `schedule_file` from the task config entry and use it when
  present; fall back to the default path only when absent.

- [P2] Emit DP rho aggregates instead of all-NaN placeholders — experiments/weighted_lse_dp/runners/run_phase3_dp.py:600-604
  `aggregate_phase3._aggregate_dp_safe_stagewise()` reads `safe_rho_mean`/
  `safe_rho_std` from each DP seed's `calibration_stats.npz`, but the DP runner
  unconditionally writes `NaN` for every stage. Every aggregated DP
  `safe_stagewise.npz` loses the responsibility curves the Phase III comparison
  is supposed to report, even though the planner computes rho on the (S,A) grid
  during each safe backup.
  Recommendation: In `run_phase3_dp.py`, compute and log `safe_rho_mean` /
  `safe_rho_std` per stage from the DP planner's `swc.last_rho` grid during
  each backup sweep.
