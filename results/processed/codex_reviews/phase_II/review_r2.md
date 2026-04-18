# Phase II Codex Standard Review R2

**Session:** b0nweghxr  
**Base:** main  
**Branch:** phase-II/closing (after BLOCKER A+B + MAJOR C+D fixes)  
**Date:** 2026-04-17

## Findings

### [P1] Read the Phase II summary schema before plotting curves

**File:** `experiments/weighted_lse_dp/analysis/make_phase2_figures.py:299-305`

`aggregate_phase2.write_outputs()` writes each `summary.json` with `scalar_metrics`, `tail_risk`, `adaptation`, and `event_rates` only; it never emits a `curves` block. In production mode this code therefore reads empty arrays for every task and silently skips the plotted data, so `learning_curves` comes out blank and the same bad assumption also breaks the adaptation/heatmap figure loaders later in this file.

### [P1] Consume the actual calibration JSON keys for Phase II plots

**File:** `experiments/weighted_lse_dp/analysis/make_phase2_figures.py:373-374`

The calibration documents produced by `aggregate_phase2.build_calibration_json()` contain `stagewise`, `tail_risk`, `adaptation`, and `event_rates`, not raw `base_returns` / `stress_returns` arrays or a top-level `margin_quantiles` object. As written, the production plotting path will label these panels as "No return arrays" / "Empty margin data" even when calibration files exist, so two of the five Phase II figures cannot be regenerated from real outputs.

### [P2] Handle regime-shift wrappers when warmstart_dp is disabled

**File:** `experiments/weighted_lse_dp/runners/run_phase2_dp.py:224-225`

The new config surface explicitly allows `warmstart_dp=false`, but in that branch `_get_base_mdp()` returns the regime-shift wrapper unchanged. `_build_ref_pi()` and planner construction then call `extract_mdp_arrays(mdp)` on the wrapper, which does not expose `p`/`r`, so any non-warmstarted `chain_regime_shift` or `grid_regime_shift` DP run fails before planning starts.
