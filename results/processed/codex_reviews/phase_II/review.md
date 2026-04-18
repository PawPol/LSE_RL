# Phase II Codex Standard Review

**Session:** byl14ca5j  
**Base:** main  
**Branch:** phase-II/closing  
**Date:** 2026-04-17

## Findings

### [P1] Train and evaluate RL on the wrapped stress environment

**File:** `experiments/weighted_lse_dp/runners/run_phase2_rl.py:597-607`

For `grid_hazard`, `grid_regime_shift`, and `taxi_bonus_shock`, `_call_factory()` returns the stressed wrapper in `wrapper_or_mdp` and an unmodified/time-augmented base MDP in `mdp_rl`. Passing `mdp_rl` into both `RLEvaluator` and `Core` means those runs never experience hazard penalties, regime switches, or bonus shocks at all; only the logger can see the wrapper, but that wrapper is never stepped. The RL results for these Phase II tasks therefore collapse to base-task behavior instead of the intended stress test.

### [P1] Solve hazard and bonus-shock DP runs on the stressed model

**File:** `experiments/weighted_lse_dp/runners/run_phase2_dp.py:223-224`

`_WRAPPER_TASKS` includes `grid_hazard` and `taxi_bonus_shock`, but `_get_base_mdp()` strips both wrappers and hands the planner `wrapper._base`. Those stresses are injected only inside `step()`, so exact DP on `_base` computes the Phase I task, not the stressed Phase II task. Every DP artifact for these two tasks will therefore miss the hazard penalty / bonus-shock dynamics and cannot support the intended degradation comparison.

### [P2] Populate Phase II scalar fields before writing calibration stats

**File:** `experiments/weighted_lse_dp/runners/run_phase2_rl.py:646-649`

`calibration_stats.npz` is staged immediately after `aggregate_calibration_stats()`, but the tail-risk/adaptation/event scalars are only computed later and never merged back into that payload. Downstream aggregation expects keys like `jackpot_event_rate`, `return_cvar_5pct`, and `adaptation_pre_change_auc` in `calibration_stats.npz`, so all of those values remain `NaN` for RL stress runs and the Phase II summaries/tables lose the scalar metrics they were added for.
