# Codex Review — Phase I Round 2 (branch diff against main)

**Job**: review-mo2a5d6q-r1cyz4  
**Completed**: 2026-04-17  
**Base**: main  
**Branch**: phase-I/closing  

## Summary

The new Phase I experiment pipeline has multiple integration bugs: default RL runs are written outside the expected results layout, ablation runs are not discoverable by the aggregator, and RL summaries omit/rename fields that downstream aggregation and reporting require. Those issues would break the end-to-end workflow even if individual runs execute.

## Findings

### [P1] Write ablation runs under the suite layout the aggregator scans
**File**: `experiments/weighted_lse_dp/runners/run_phase1_ablation.py:189-197`

`aggregate_phase1.find_run_dirs()` only discovers ablation artifacts under `results/weighted_lse_dp/phase1/ablation/<task>/<algorithm>/seed_*`, but this wrapper forwards `suite=gamma_dir` (and, for RL, a gamma-specific `out_root`) into child runners that append their own `phase1/<suite>/...` segments again. In practice the outputs land under paths like `.../phase1/ablation/phase1/gamma095/...` or `.../phase1/ablation/gamma095/phase1/paper_suite/...`, so the aggregation step will miss every ablation run even when execution succeeds.

### [P1] Fix the RL runner's default output root before appending phase/suite
**File**: `experiments/weighted_lse_dp/runners/run_phase1_rl.py:75-75`

The default `--out-root` already includes `phase1/paper_suite`, but `RunWriter.create()` adds `phase="phase1"` and `suite="paper_suite"` again. Running the CLI with defaults therefore writes to `results/weighted_lse_dp/phase1/paper_suite/phase1/paper_suite/...`, which is outside the directory layout that `find_run_dirs()` and the rest of the pipeline expect, so a default run becomes effectively undiscoverable downstream.

### [P1] Emit the RL return metric name that aggregation and tables consume
**File**: `experiments/weighted_lse_dp/runners/run_phase1_rl.py:362-366`

The RL runner flushes `evaluator.summary()` verbatim, which only contains `final_10pct_disc_return`, while `aggregate_phase1`, `make_phase1_tables.py`, and `fig_ablation.py` all look for `final_disc_return_mean`. As a result every aggregated RL group will be missing its headline return metric, so the processed tables/figures cannot report RL performance even when the raw runs completed successfully.

### [P2] Preserve gamma' in RL run metadata using the key the aggregator reads
**File**: `experiments/weighted_lse_dp/runners/run_phase1_rl.py:259-265`

`aggregate_phase1.discover_runs()` extracts ablation levels from `config["gamma_prime_override"]`, but RL runs store the override under `gamma_prime`. If the ablation pathing is fixed, all RL ablation seeds at different gamma' values will still be grouped together because their `gamma_prime` field is effectively invisible to the aggregator, collapsing distinct ablation settings into one summary group.
