# Codex Review — Phase I Round 3 (branch diff against main)

**Job**: review-mo2b27d3-t04uum  
**Completed**: 2026-04-17  
**Base**: main  
**Branch**: phase-I/closing  

## Summary

The patch introduces at least one runtime compatibility break on supported NumPy versions, and both direct ablation entry points can write results into the wrong suite layout. Those issues can either crash reporting code or silently corrupt experiment outputs.

## Findings

### [P1] Avoid NumPy-2-only `np.trapezoid` in supported environments
**File**: `experiments/weighted_lse_dp/common/metrics.py:193`  

The repository declares `numpy>=1.24`, but `np.trapezoid` is only available in NumPy 2.x. On a supported 1.24/1.26 install, this line raises `AttributeError` as soon as AUC is computed, breaking aggregation; the same incompatibility is also introduced in `RLEvaluator.summary()`.

### [P2] Pass the configured suite into RL run dispatch
**File**: `experiments/weighted_lse_dp/runners/run_phase1_rl.py:527-533`  

`main()` loads the suite config but drops it when it calls `run_single`, so every RL invocation writes to `phase1/paper_suite/...` via the default `suite="paper_suite"`. That silently misclassifies any non-default suite run (for example a direct `--gamma-prime` ablation or a future smoke/alt suite config) and can overwrite the baseline artifacts that `aggregate_phase1.py` expects to keep separate by suite.

### [P2] Route direct DP gamma-prime runs to an ablation suite
**File**: `experiments/weighted_lse_dp/runners/run_phase1_dp.py:455-462`  

The DP CLI advertises `--gamma-prime` as an ablation mode, but `main()` still forwards the config's normal suite (usually `paper_suite`) into `_run_single`. Running `run_phase1_dp.py --gamma-prime ...` therefore writes the ablated run back into `phase1/paper_suite/<task>/<algo>/seed_*`, replacing the baseline seed directory instead of creating the `phase1/ablation/gamma*` layout that the aggregator scans.
