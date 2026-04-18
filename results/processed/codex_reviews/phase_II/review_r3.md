# Phase II Codex Standard Review R3

**Session:** review-mo2thz1w-kohkfg (019d9b2c-5447-7612-a153-aefb4da798c3)
**Base:** main
**Branch:** phase-II/closing (after R2 BLOCKER+MAJOR+MINOR fixes)
**Date:** 2026-04-17

## Summary

The patch introduces at least one invalid experiment pathway: the hyperparameter ablation does not actually vary the hyperparameters it claims to sweep. It also exports incorrect data for the new return-distribution figure, so the analysis outputs are not reliable as written.

## Findings

### [P1] Honor hyperparameter overrides in RL ablation runs

**File:** `experiments/weighted_lse_dp/runners/run_phase2_rl.py:566-567`

When `run_phase2_ablation.py` launches the hyperparameter sweep, it passes `epsilon_override` and `lr_multiplier` through `task_config`, but `run_phase2_rl.run_single()` still hard-codes `_EPSILON` and `_LEARNING_RATE` here and then calls `_make_agent()` without consulting those overrides. As a result, every supposedly retuned run trains with the same exploration rate and step size, so the entire hyperparameter ablation produces mislabeled duplicate results.

### [P2] Export real episode-return samples for Figure 11.1.2

**File:** `experiments/weighted_lse_dp/runners/aggregate_phase2.py:835-845`

`make_phase2_figures.py` plots `base_returns` and `stress_returns` as histograms of episode returns, but this export populates `base_returns` from `stagewise.reward_mean_mean` (one value per stage) and `stress_returns` from a single `event_conditioned_return` scalar. On real Phase II outputs that makes the "return distribution" figure a histogram over timesteps plus a one-point stress sample, so the published distribution is not measuring returns across episodes or seeds at all.
