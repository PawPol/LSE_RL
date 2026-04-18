# Codex Standard Review — Phase III R5

Session ID: (r5_standard)
Base: 4fdbf0d (Phase II close commit)
Branch: phase-III/closing
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
