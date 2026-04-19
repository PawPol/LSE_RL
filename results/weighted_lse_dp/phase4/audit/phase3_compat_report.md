# Phase III Compatibility Audit Report

Generated: 2026-04-19T04:29:14.903714+00:00
Git SHA: 04ae83ce0ad58e01ae067f48c8d6b1da9f9e8ac7

## Code Audit

**Status**: FAIL

Schedule files found: 8
  - /Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a9240aec/results/weighted_lse_dp/phase3/calibration/chain_sparse_long/schedule.json
  - /Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a9240aec/results/weighted_lse_dp/phase3/calibration/chain_jackpot/schedule.json
  - /Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a9240aec/results/weighted_lse_dp/phase3/calibration/chain_catastrophe/schedule.json
  - /Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a9240aec/results/weighted_lse_dp/phase3/calibration/chain_regime_shift/schedule.json
  - /Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a9240aec/results/weighted_lse_dp/phase3/calibration/grid_sparse_goal/schedule.json
  - /Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a9240aec/results/weighted_lse_dp/phase3/calibration/grid_hazard/schedule.json
  - /Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a9240aec/results/weighted_lse_dp/phase3/calibration/grid_regime_shift/schedule.json
  - /Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a9240aec/results/weighted_lse_dp/phase3/calibration/taxi_bonus_shock/schedule.json

### Observability Gaps

- safe_weighted_common.py missing compute_safe_target_ev_batch (Issue 1 fix not applied)

**Notes**: chain_sparse_long: T=120 (from beta_used_t length); chain_jackpot: T=60 (from beta_used_t length); chain_catastrophe: T=60 (from beta_used_t length); chain_regime_shift: T=60 (from beta_used_t length); grid_sparse_goal: T=120 (from beta_used_t length); grid_hazard: T=80 (from beta_used_t length); grid_regime_shift: T=80 (from beta_used_t length); taxi_bonus_shock: T=120 (from beta_used_t length); run_phase3_rl.py: _N_BASE dict with env-derived n_base validation (R6-2 compliant); All tasks in paper_suite.json have reward_bound

## Result Audit

Result directories found: 8
Total run artifacts: 154

**Notes**: Total run artifact sets found: 154

## DP Replay Smoke Check

**Status**: PASS
- beta_used nonzero: True
- rho valid: True
- RL replay skipped: True

**Notes**: chain_sparse_long replay: V shape=(121, 60), V range=[0.000000, 31.778998], beta_used range=[0.000000, 0.000000], rho range=[0.502513, 0.502513], n_sweeps=1; chain_sparse_long: sparse-data fallback triggered (all beta_used=0). This is expected for tasks with insufficient aligned-margin data.; chain_jackpot replay (nonzero-beta verification): V shape=(61, 26), V range=[0.000000, 20.629085], beta_used range=[0.000000, 0.000201]

## Overall Verdict

**Phase III -> Phase IV compatibility**: ISSUES FOUND -- review gaps above
