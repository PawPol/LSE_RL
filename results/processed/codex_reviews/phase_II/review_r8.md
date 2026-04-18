# Phase II R8 — Standard Codex Review

**Session**: 019d9e60-94d5-7ff3-a128-e683d2b31dce  
**Base**: main  
**Branch**: phase-II/closing  
**Model**: gpt-5.4  
**Date**: 2026-04-17

---

## Summary

The patch introduces at least one suite-affecting correctness issue in Phase II RL logging (`grid_sparse_goal` uses the wrong base-state count), which corrupts generated artifacts for that task. It also breaks the documented factory contract for wrapper-backed RL tasks by returning unstressed `mdp_rl` environments from two new factories.

---

## Findings

### [P1] Use the 49-state base size for `grid_sparse_goal` logging
**File**: `experiments/weighted_lse_dp/runners/run_phase2_rl.py:122`

`TransitionLogger` decodes each augmented state as `t = aug_id // n_base` and `base_state = aug_id % n_base`. Setting `grid_sparse_goal` to 25 here is inconsistent with the 7x7 task (`n_states=49`), so every logged transition for this task is mis-binned once `aug_id >= 25`. That corrupts the `transitions.npz` stage/state fields and, downstream, the aggregated calibration stats and visitation heatmaps for the default Phase II suite.

### [P2] Time-augment the hazard wrapper rather than the base grid
**File**: `experiments/weighted_lse_dp/tasks/hazard_wrappers.py:232-234`

`make_grid_hazard()` advertises that it returns the stressed RL environment in `mdp_rl`, but this line wraps `mdp_base` instead of `wrapper`. Any caller that uses the factory's second return value directly will train/evaluate on the plain Phase I grid and never observe hazard penalties or hazard-triggered terminations. The current RL runner works around this manually, but the factory contract is still wrong for direct callers.

### [P2] Time-augment the bonus-shock wrapper rather than the base taxi MDP
**File**: `experiments/weighted_lse_dp/tasks/stress_families.py:644-646`

This constructs `mdp_rl` from `mdp_base`, so the returned RL environment never applies the bonus-shock logic implemented in `TaxiBonusShockWrapper.step()`. Any code that consumes `make_taxi_bonus_shock()` as documented will silently run on the unstressed taxi task, which defeats the Phase II task design unless every caller adds the same manual re-wrapping workaround as `run_phase2_rl` does.
