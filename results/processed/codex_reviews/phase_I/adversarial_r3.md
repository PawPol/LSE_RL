# Codex Adversarial Review — Phase I Round 3 (branch diff against main)

**Job**: review-mo2b2a9r-4moryv  
**Completed**: 2026-04-17  
**Base**: main  
**Branch**: phase-I/closing  
**Focus**: "challenge finite-horizon DP correctness and the calibration-logging schema; flag any silent math or schema drift from docs/specs/phase_I_classical_beta0_experiments.md."

## Verdict

**needs-attention** — No-ship. The DP review path is still capable of certifying incorrect results and the Phase I calibration artifacts are not schema-safe under retry or across DP/RL modes.

## Findings

### [high] `supnorm_to_exact` is measured against the same planner's final output, not an independent exact optimum
**File**: `experiments/weighted_lse_dp/runners/run_phase1_dp.py:294-295`  
**Confidence**: 0.97

`run_phase1_dp.py` wires `DPCurvesLogger(v_exact=planner.V)` after the planner has already run. That means the reported `supnorm_to_exact` is only distance-to-this-planner's-final-table, not distance to the true exact solution required by spec §7.3. If PI/MPI/AsyncVI ever stop early, regress, or run with non-default budgets, the metric can still collapse to 0 on the final sweep and hide the error. This undermines the main finite-horizon DP correctness signal the review is supposed to trust.

**Recommendation**: Compute one independent exact reference table per task/gamma, e.g. via single-pass `ClassicalValueIteration`, and pass that reference into `DPCurvesLogger` for every planner. Fail the run if the reference cannot be produced.

### [high] DP calibration stats reuse the RL schema but silently change the meaning of `count` and all stage aggregates
**File**: `experiments/weighted_lse_dp/common/calibration.py:423-465`  
**Confidence**: 0.93

`build_calibration_stats_from_dp_tables` flattens every `(state, action)` pair at a stage and records `count[t] = S * A`, with quantiles/means taken over that synthetic table. The Phase I spec describes stagewise sample counts and empirical aggregates; RL runs populate the same schema from actual transition logs. Using the same `calibration_stats.npz` keys for occupancy-weighted RL data and uniform-over-`(s,a)` DP data creates silent schema drift: downstream consumers cannot tell whether `count`, margin quantiles, and alignment frequencies are empirical or analytic, so cross-run comparisons become misleading.

**Recommendation**: Either split analytic DP calibration into a distinct schema/storage mode with explicit provenance, or compute DP statistics under a documented occupancy measure compatible with the RL artifacts and stamp that weighting contract into the file header.

### [medium] Run directories are silently reused, so retries can publish stale artifacts from a previous attempt
**File**: `experiments/weighted_lse_dp/common/schemas.py:283-310`  
**Confidence**: 0.91

`RunWriter.create` calls `make_run_dir(..., exist_ok=True)` for the canonical `phase/suite/task/algorithm/seed` path. `flush()` then writes only the artifacts staged in memory. On a rerun after a crash or partial failure, old files in that directory are kept unless they happen to be overwritten in the new attempt. This makes schema validation and aggregation trust stale `curves.npz` / `transitions.npz` / `calibration_stats.npz` from an earlier run.

**Recommendation**: Do not reuse seed directories by default. Either require a clean/nonexistent run dir, or write to an attempt-specific temp dir and atomically promote it only after a full successful flush plus an explicit completion marker.

## Next Steps (from Codex)

1. Recompute DP curves with an independently exact reference value table.
2. Decide and document whether DP calibration stats are empirical, analytic-uniform, or occupancy-weighted, then encode that in the schema.
3. Make run publication atomic so stale artifacts cannot survive retries.
