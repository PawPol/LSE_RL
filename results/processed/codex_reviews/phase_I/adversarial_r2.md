# Codex Adversarial Review — Phase I Round 2 (branch diff against main)

**Job**: review-mo2a806x-9nl0s6  
**Completed**: 2026-04-17  
**Base**: main  
**Branch**: phase-I/closing  
**Focus**: "challenge finite-horizon DP correctness and the calibration-logging schema; flag any silent math or schema drift from docs/specs/phase_I_classical_beta0_experiments.md."

## Verdict

**needs-attention** — No-ship: the branch still drifts from the Phase I spec in ways that hide DP regressions and produce incompatible artifacts.

## Findings

### [high] DP runs never populate the mandatory `supnorm_to_exact` curve
**File**: `experiments/weighted_lse_dp/runners/run_phase1_dp.py:291-315`  
**Confidence**: 0.98

`run_phase1_dp.py` instantiates `DPCurvesLogger` with `v_exact=None` for every DP run, then replays sweeps into it. In `DPCurvesLogger`, that means `supnorm_to_exact` is recorded as `NaN` for all sweeps instead of the spec-required error-to-exact trace. Because every planner here is exact and `planner.V` is available after `run()`, this silently drops the main correctness signal that would catch finite-horizon DP regressions while still writing a superficially valid `curves.npz`.

**Recommendation**: Pass an actual exact table into `DPCurvesLogger` (for example `planner.V` after the run, or a separately computed exact baseline) and fail validation if `supnorm_to_exact` would otherwise be all-NaN for DP artifacts.

### [high] Calibration schema omits the spec-required aligned-margin frequency
**File**: `experiments/weighted_lse_dp/common/schemas.py:120-138`  
**Confidence**: 0.99

The declared `CALIBRATION_ARRAYS` contract has no field for aligned-margin frequency, and the calibration aggregator only emits counts, reward/value moments, margin quantiles, positive/negative margin means, and max-abs statistics. The Phase I spec explicitly requires aligned-margin frequency in the stored calibration metrics. As written, every `calibration_stats.npz` produced by this branch is structurally incapable of satisfying the documented schema, so downstream Phase III consumers cannot recover that metric and the tests will not catch the omission because they validate against the already-drifted tuple.

**Recommendation**: Add an aligned-margin-frequency field to `CALIBRATION_ARRAYS`, compute it in both RL and DP calibration aggregation paths, and update validators/tests to enforce the spec rather than the current reduced tuple.

### [medium] The RL runner's default output root double-nests `phase1/paper_suite`
**File**: `experiments/weighted_lse_dp/runners/run_phase1_rl.py:74-75`  
**Confidence**: 0.96

`_DEFAULT_OUT_ROOT` is already `results/weighted_lse_dp/phase1/paper_suite`, but `RunWriter.create()` always appends `/<phase>/<suite>/...` underneath its `base`. With the CLI defaults, paper-suite RL runs therefore land under `results/weighted_lse_dp/phase1/paper_suite/phase1/paper_suite/...` instead of the spec path. That is a schema/layout regression, not cosmetic: any aggregator or verifier expecting the documented tree will miss these runs or treat them as absent.

**Recommendation**: Change the default `--out-root` to `results/weighted_lse_dp` (or normalize `out_root` before calling `RunWriter.create`) so the writer produces the canonical `results/weighted_lse_dp/phase1/paper_suite/<task>/<algorithm>/seed_<seed>/` tree.

## Next Steps (from Codex)

1. Restore the DP correctness curve by writing real `supnorm_to_exact` values and adding a validator that rejects all-NaN DP traces.
2. Bring `calibration_stats.npz` back into spec by adding aligned-margin frequency end-to-end and updating tests to require it.
3. Fix the RL runner default output root and verify the produced directory tree matches the spec.
