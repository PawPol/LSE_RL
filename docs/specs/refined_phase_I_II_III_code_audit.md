# Refined audit of Phase I / II / III code (with vendored `mushroom-rl-dev` attached)

Date: 2026-04-19

## What I verified

I unpacked both uploaded archives and ran the repository test suite against the vendored MushroomRL tree by setting:

- `PYTHONPATH=/mnt/data/review_stub:/mnt/data/review_repo:/mnt/data/review_mushroom/mushroom-rl-dev`

I also had to provide a tiny local `pygame` stub because the container lacks the `pygame` package and `mushroom_rl.utils.viewer` imports it eagerly at module import time. No render/viewer paths were exercised in this audit.

Commands run:

```bash
python3 -m pytest
python3 -m compileall -q experiments tests src
```

Result:

- `547 passed, 22 warnings in 10.26s`
- compile pass succeeded

The 22 warnings are all from `BetaSchedule` permissive override validation in `test_safe_dp_integration.py`; they do **not** indicate test failures.

## Bottom line

With the vendored framework present, the **core operator / DP / TD implementation is much stronger than it looked from the incomplete archive**: the current test suite passes end-to-end.

However, I still found several **real bugs and correctness problems** in the Phase I/II/III pipeline that the existing tests do not catch. The most important remaining issues are in:

1. DP reference-policy construction for PE / SafePE,
2. Phase II calibration aggregation for regime-shift tasks,
3. Phase III schedule construction (which can artificially drive `beta_cap` / `beta_used` toward zero),
4. RL evaluation semantics and env reuse.

## Confirmed remaining bugs

### 1. Phase II / III DP build the wrong reference policy for PE / SafePE on grid and taxi tasks

**Files**
- `experiments/weighted_lse_dp/runners/run_phase2_dp.py:331-346`
- `experiments/weighted_lse_dp/runners/run_phase3_dp.py:367-383`

Both `_build_ref_pi()` helpers ignore the suite config's `ref_policy` and return an all-zero action array for every state.

That is incompatible with the configs, which explicitly request policies like:
- `shortest_path`
- `pickup_then_deliver`

I verified this dynamically against the task-factory reference policies:

- `grid_base`: runner policy uses only action `0`, while the authoritative reference policy uses actions `{0,1,3}`
- `taxi_base`: runner policy differs on **42** states from the task-factory policy

Representative output from the audit script:

```text
grid_base same as runner phase2? False
grid_base same as runner phase3? False
grid phase2 unique actions [0] task_factory unique [0 1 3]
grid diff count 24

taxi_base same as runner phase2? False
taxi diff count 42
```

**Impact**
- `PE` / `SafePE` rows on grid/taxi tasks are evaluating the wrong policy.
- Any Phase II calibration averaged across DP algorithms is contaminated by incorrect PE data.
- Any claims comparing PE / SafePE on these tasks are not trustworthy until rerun.

### 2. `aggregate_phase2.build_calibration_json()` drops the regime-shift pre/post DP groups when building the stagewise calibration block

**File**
- `experiments/weighted_lse_dp/runners/aggregate_phase2.py:941-949`

The function first correctly collects `task_groups` using `task in raw_tasks`.
But when it later builds `calib_dicts_dp` / `calib_dicts_all`, it mistakenly re-filters with:

```python
if task != task_family:
    continue
```

That excludes suffixed groups such as:
- `chain_regime_shift_pre_shift`
- `chain_regime_shift_post_shift`
- `grid_regime_shift_pre_shift`
- `grid_regime_shift_post_shift`

I confirmed this with a minimal synthetic reproduction: calling `build_calibration_json()` with only `*_pre_shift` and `*_post_shift` groups returns `stagewise = None`.

Representative audit output:

```text
stagewise keys None
stagewise reward_mean None
```

**Impact**
- Regime-shift calibration JSONs are missing the very DP groups they are meant to merge.
- Phase III schedules derived from those calibration JSONs for regime-shift families are not based on the intended data.

### 3. The Phase III schedule builder uses the wrong statistic for representative margin and informativeness

**Files**
- `experiments/weighted_lse_dp/calibration/calibration_utils.py:61-67, 75-133`
- `experiments/weighted_lse_dp/calibration/build_schedule_from_phase12.py:137-147`
- upstream source of the statistic: `experiments/weighted_lse_dp/common/calibration.py:318, 540`

Current behavior:
- `extract_stagewise_arrays()` reads `aligned_positive_mean_mean`
- `compute_representative_margin()` uses that value directly as `m_star`
- `compute_informativeness()` multiplies it by `sqrt(aligned_margin_freq)`

But `aligned_positive_mean` is **not** a conditional positive-margin mean. It is:

```python
E[max(m, 0)] = P(m > 0) * E[m | m > 0]
```

So the current informativeness score is proportional to:

```text
E[m | m>0] * P(m>0)^(3/2)
```

not the intended “positive-margin size times sqrt-frequency”.

This suppresses exactly the rare-but-informative stages you care about.

A toy audit example:
- `P(m>0)=0.01`
- `E[m|m>0]=10`
- stored `aligned_positive_mean = 0.1`

Then the current code uses:

```text
raw used by current code = 0.01
if using conditional mean, raw would be = 1.0
```

So the stage is downweighted by **100x**.

**Impact**
- `alpha_t` can become much smaller than intended.
- resulting `beta_cap_t` can become much smaller than intended.
- this is a plausible direct cause of the “TAB collapses to near-classical everywhere” behavior you observed in Phase III.

This is not just a theoretical nicety; it directly affects the certification schedule that drives the experiments.

### 4. `RLEvaluator` counts any early absorbing termination as success, including catastrophic failure

**File**
- `experiments/weighted_lse_dp/common/callbacks.py:653-657`

The evaluator declares success whenever `absorbing=True` and `t < horizon`:

```python
episode_success = (t < self._horizon)
```

That means **any** early terminal transition is counted as success, even if it is a catastrophe.

I verified this dynamically with a dummy environment that always catastrophically terminates on the first step with reward `-10`:

```text
{'disc_return_mean': -10.0, 'success_rate': 1.0}
[True, True, True]
```

**Impact**
- `success_rate`, `final_10pct_success_rate`, and `steps_to_threshold` are wrong on tasks with non-goal absorbing failures.
- This especially affects catastrophe-style tasks and any future absorbing-failure stress tests.

### 5. All RL runners reuse the same environment object for training and evaluation

**Files**
- `experiments/weighted_lse_dp/runners/run_phase1_rl.py:313-320, 323`
- `experiments/weighted_lse_dp/runners/run_phase2_rl.py:656-667`
- `experiments/weighted_lse_dp/runners/run_phase3_rl.py:892-906`

Each runner does the equivalent of:

```python
evaluator = RLEvaluator(env=mdp_rl, ...)
core = Core(agent, mdp_rl, ...)
```

So checkpoint evaluation and training operate on the **same** env instance.

I verified the regime-shift consequence dynamically with a real `ChainRegimeShiftWrapper`:

```text
after one training reset: 1 False
after evaluator with 2 episodes: 3 True
after next training reset: 4 True
```

With `change_at_episode=2`, two evaluation episodes on the shared env advanced the wrapper's internal episode counter and triggered the regime shift **before the next training episode**.

**Impact**
- For regime-shift tasks, the change point happens after a mixture of training and evaluation episodes, not after the configured number of training episodes.
- Even on stationary tasks, evaluation perturbs the environment RNG trajectory seen by training.
- This invalidates clean causal interpretation of adaptation timing.

### 6. Phase III DP honors schedule overrides when loading, but writes the wrong schedule path into provenance

**Files**
- load override: `experiments/weighted_lse_dp/runners/run_phase3_dp.py:717-719`
- wrong provenance write: `run_phase3_dp.py:496-500, 643-649`

The runner correctly loads a per-task `schedule_file` override, but later records:

- `run_config["schedule_file"] = <default path>`
- `write_safe_provenance(schedule_path=<default path>)`

instead of the actual override path.

**Impact**
- Phase III DP ablation/custom-schedule runs can be mislabeled even when they executed with a different schedule.
- This is a provenance bug, not an algorithmic one, but it matters for reproducibility.

### 7. DP run provenance records the default suite config path rather than the actual CLI/config path

**Files**
- `experiments/weighted_lse_dp/runners/run_phase1_dp.py:258-260`
- `experiments/weighted_lse_dp/runners/run_phase2_dp.py:450-452`
- `experiments/weighted_lse_dp/runners/run_phase3_dp.py:496-499`

Each of these stamps:

```python
"suite_config_path": str(_DEFAULT_CONFIG)
```

regardless of which config file was actually passed.

**Impact**
- `run.json` provenance is wrong for custom runs, ablations, and alternate suite files.
- Not a numerical bug, but a real reproducibility defect.

### 8. Phase II visitation heatmaps omit terminal arrivals

**File**
- `experiments/weighted_lse_dp/runners/aggregate_phase2.py:610-619`

The aggregation bins only `transitions["state"]`, not terminal `next_state` arrivals.

**Impact**
- Goal states / absorbing terminal states are undercounted, sometimes to zero, in the published visitation diagnostics.
- This can visually hide exactly the absorbing/failure states you want to audit.

## Things that look good now

Compared with the earlier incomplete-archive audit, I can now say much more confidently that these parts are in good shape:

- vendored `mushroom_rl` safe DP / TD code imports and passes the current test suite
- `beta=0` equivalence tests pass
- clipping / certification tests pass
- time augmentation tests pass
- safe DP integration tests pass
- Phase III smoke tests pass
- the older missing-dependency blocker is resolved once the vendored tree is provided
- the previously reported `warmstart_dp=False` regime-shift DP crash appears fixed in current `run_phase2_dp.py`
- the previously reported Phase III RL `safe_margin` logging bug for ExpectedSARSA appears fixed in the current code (`swc.last_margin` is now used)

## Practical interpretation

If your question is “can I trust the *core safe TAB operator implementation*?” then the answer is **much more yes** than before: the test evidence is decent.

If your question is “can I trust the *current Phase I/II/III experimental pipeline outputs* as published evidence?” then the answer is **not fully yet**.

The most damaging remaining issues are:

1. wrong PE / SafePE reference policies,
2. regime-shift calibration JSON bug,
3. rare-stage suppression in schedule building,
4. RL evaluation success/env-sharing bugs.

Those four are enough to justify rerunning at least the affected Phase II/III summaries after fixes.

## Recommended fix order

1. **Fix the schedule-builder statistic bug** (Issue 3).
   - This is the one most likely tied to your observed `beta_cap ≈ 0` problem.
2. **Fix RL evaluation semantics and separate eval envs** (Issues 4 and 5).
   - Otherwise adaptation/tail-risk claims remain muddied.
3. **Fix PE / SafePE reference policy construction** (Issue 1).
   - Rerun affected DP rows and any calibration that used them.
4. **Fix regime-shift calibration aggregation** (Issue 2).
   - Regenerate regime-shift calibration JSONs and downstream schedules.
5. Clean up provenance/diagnostic issues (Issues 6–8).

