# Phase IV-C — Advanced stabilization, state-dependent scheduling, and geometry-priority ablations

This document is for the coding agent. Treat it as the **third, self-standing Phase IV-C implementation spec** for the TAB experiment stack.

Phase IV-C begins only after Phase IV-A has established certified operator activation and Phase IV-B has run the simple, interpretable translation study.

Phase IV-C adds advanced algorithmic and scheduling refinements:

1. SafeDoubleQLearning;
2. SafeTargetQLearning;
3. SafeTargetExpectedSARSA;
4. state-dependent frozen schedulers;
5. geometry-prioritized asynchronous DP;
6. estimator-stability ablations;
7. trust-region, adaptive-headroom, wrong-sign, and schedule-quality ablations.

These are not the first mechanism test. They are stabilization and enhancement experiments designed to answer:

> If Safe TAB is activated, can better estimators and more localized scheduling improve the translation from operator diagnostics to outcomes?

---

## 0. Non-negotiable workflow rules

1. Start with a written plan in `tasks/todo.md`.
2. Keep `tasks/todo.md` current and close each completed item with a short review note.
3. Add every correction, debugging lesson, and ablation surprise to `tasks/lessons.md`.
4. Do not overwrite Phase I/II/III/IV-A/IV-B outputs.
5. Treat Phase IV-C algorithms as enhancements or ablations, not as the only evidence for the TAB mechanism.
6. Any learned or state-dependent scheduler must be frozen during each Bellman-learning phase.
7. Any state-dependent scheduler must use hierarchical shrinkage / backoff.
8. Every advanced method must be compared against the simple Phase IV-B stagewise baseline.
9. When lower-base-`gamma` is used, matched classical and safe-zero controls remain mandatory.
10. Report whether gains come from estimator stabilization, schedule localization, geometry priority, or TAB nonlinearity.

---

## 1. Phase IV-C objectives

Phase IV-C is complete only if it can answer:

1. Does estimator stabilization reduce variance or overestimation enough to improve Safe TAB?
2. Do state-dependent frozen schedulers improve activation localization or outcomes relative to stagewise schedules?
3. Does geometry-prioritized DP improve planning speed or prioritization efficiency?
4. Are trust-region and safe-certification clips load-bearing in the activated regime?
5. Does wrong-sign scheduling harm or neutralize the effect as expected?
6. Are improvements still interpretable after adding advanced machinery?

---

## 2. Inputs from earlier phases

Before Phase IV-C starts, verify that these exist:

```text
results/weighted_lse_dp/phase4/task_search/selected_tasks.json
results/weighted_lse_dp/phase4/counterfactual_replay/
results/weighted_lse_dp/phase4/translation/
experiments/weighted_lse_dp/configs/phase4/activation_suite.json
experiments/weighted_lse_dp/configs/phase4/translation_study.json
```

Required reports:

1. Phase IV-A activation-gate report;
2. Phase IV-B translation-analysis report;
3. selected task-family labels: activated, event-conditioned activated, or low-activation control.

Do not run Phase IV-C on task families that failed activation unless they are explicitly labeled as negative controls.

---

## 3. Safe estimator-stabilized RL algorithms

### 3.1 SafeDoubleQLearning

Mandatory in Phase IV-C.

Reason:

- overestimation error corrupts the bootstrap value `v_next`;
- corrupted `v_next` corrupts the margin `r - v_next`;
- corrupted margin corrupts the natural shift `u = beta * margin`.

Required behavior:

1. maintain two Q tables, `Q_A` and `Q_B`;
2. use one table for action selection and the other for evaluation;
3. compute the safe target using the evaluation-side bootstrap;
4. log which table supplied selection and which supplied evaluation;
5. log `double_gap`.

Required logging fields:

- `q_a_next`
- `q_b_next`
- `selected_action_source`
- `evaluation_value_source`
- `double_gap`
- `margin_double`
- `natural_shift_double`

### 3.2 SafeTargetQLearning

Mandatory in Phase IV-C.

Support:

- hard sync every `K` updates;
- optional Polyak averaging.

Required behavior:

1. online Q table receives updates;
2. target Q table supplies `v_next` for the safe target;
3. target table is frozen between sync / Polyak updates;
4. schedule remains frozen during the Bellman-learning phase.

Required logging fields:

- `q_online_next`
- `q_target_next`
- `q_target_gap`
- `target_sync_step`
- `target_update_mode = hard | polyak`

### 3.3 SafeTargetExpectedSARSA

Mandatory in Phase IV-C.

The next-state expectation must use the frozen target table when configured.

Required behavior:

1. compute the next-action expectation under the behavior / target policy;
2. evaluate the expected value using target Q when enabled;
3. compute `margin = reward - v_next_expected_target`;
4. use that same margin in safe target and diagnostic logging.

### 3.4 Algorithm comparison matrix

For RL tasks run at least:

1. classical Q-learning;
2. classical ExpectedSARSA;
3. Phase IV-B SafeQLearning stagewise baseline;
4. Phase IV-B SafeExpectedSARSA stagewise baseline;
5. SafeDoubleQLearning;
6. SafeTargetQLearning;
7. SafeTargetExpectedSARSA.

Every advanced method must be paired with the same schedule family and matched controls used in Phase IV-B.

---

## 4. State-dependent frozen schedulers

State-dependent schedulers are Phase IV-C enhancements. They should not replace the simple stagewise scheduler as the main mechanism test.

### 4.1 Mainline recipe

1. construct state or next-state bins;
2. compute binwise `xi_ref_{t,b}` and `u_target_{t,b}`;
3. shrink toward the stagewise schedule with hierarchical backoff;
4. apply trust cap and safe cap;
5. freeze the scheduler during learning.

### 4.2 Hierarchical backoff

Default:

```text
u_hb_{t,b} = u_stage_t + w_{t,b} * (u_design_{t,b} - u_stage_t)
w_{t,b} = n_{t,b} / (n_{t,b} + tau_bin)
tau_bin = 100
```

### 4.3 Bin construction

Supported bin modes:

1. exact discrete state bins;
2. next-state bins;
3. event-region bins;
4. margin-quantile bins;
5. hand-coded task semantic bins for chain/grid only.

Every binning mode must be logged in the schedule file.

### 4.4 State-dependent schedule fields

Extend schedule v3 with:

```json
{
  "scheduler_mode": "statebin_u",
  "binning_mode": "exact_state | next_state | event_region | margin_quantile | semantic",
  "state_bin_ids": [],
  "bin_counts": [],
  "xi_ref_tb": [],
  "u_target_tb": [],
  "u_hb_tb": [],
  "theta_used_tb": [],
  "beta_used_tb": [],
  "trust_clip_active_tb": [],
  "safe_clip_active_tb": [],
  "backoff_weight_tb": []
}
```

### 4.5 Cross-fitting rule

State-dependent or learned frozen schedulers should be fit on pilot seeds / pilot episodes and evaluated on disjoint seeds / later episodes whenever feasible.

If cross-fitting is not feasible, log the limitation clearly.

### 4.6 Required comparison

Compare:

1. stagewise scheduler;
2. state-dependent scheduler;
3. state-dependent scheduler with stronger shrinkage;
4. state-dependent scheduler with weaker shrinkage.

Primary question:

> Does localization increase event-conditioned activation and improve the task-specific outcome without destabilizing learning?

---

## 5. Geometry-prioritized asynchronous DP

### 5.1 Priority score

Implement a geometry-priority planner with priority:

```text
geom_gain = abs(effective_discount_used - gamma_base)
priority = abs(residual)
         * (1 + lambda_geom * geom_gain
              + lambda_u * abs(u_used_ref_stage)
              + lambda_kl * KL_Bern(rho_used || p0))
```

Defaults:

```text
lambda_geom = 1.0
lambda_u = 1.0
lambda_kl = 1.0
```

### 5.2 Required DP variants

For DP tasks run at least:

1. safe synchronous VI;
2. safe asynchronous VI;
3. residual-priority async VI;
4. geometry-priority async VI;
5. geometry-priority MPI if feasible.

### 5.3 Required DP metrics

Report:

- sweeps to tolerance;
- number of state-stage-action backups;
- wall-clock time;
- residual curves;
- geometry-priority score distributions;
- fraction of backups spent on high-activation states;
- final sup-norm error to exact solution;
- overhead ratio vs classical and vs residual-priority async VI.

Primary question:

> Does geometry-priority reduce backups needed to resolve high-activation / high-impact states?

---

## 6. Advanced ablations

### 6.1 Trust-region ablation

Compare:

1. no trust cap, safe cap only;
2. trust cap + safe cap;
3. overly tight trust cap;
4. overly loose trust cap.

Report:

- activation diagnostics;
- TD-target variance;
- outcome metrics;
- instability flags;
- cap utilization.

### 6.2 Adaptive-headroom ablation

Compare:

1. constant alpha;
2. adaptive alpha;
3. low alpha budget;
4. high alpha budget.

Report:

- `U_safe_ref_t`;
- `u_ref_used_t`;
- safe clip activity;
- contraction headroom;
- outcome metrics.

### 6.3 Wrong-sign ablation

Run wrong-sign schedules on at least:

1. one positive-tail family;
2. one catastrophe / hazard family.

Purpose:

- show sign alignment matters;
- show the effect is not generic “more nonlinearity = better.”

### 6.4 Constant-`u` ablation

Compare the stagewise natural-shift scheduler against constant reference effect:

```text
u_ref_t = constant
```

Purpose:

- show schedule quality matters;
- separate stagewise calibration from generic nonlinear perturbation.

### 6.5 Raw-unclipped ablation

Run only on small subsets.

Purpose:

- demonstrate why certification is needed;
- show larger raw effects can destabilize or violate the certified bound.

Do not use raw-unclipped results as main performance evidence.

### 6.6 Optional deterministic schedule warmup

A deterministic global update multiplier for `u_target` may stabilize early RL when value estimates are poor.

But this changes the operator over training time, so it is appendix-only unless separately justified.

Keep it behind a config flag.

---

## 7. Estimator-stability diagnostics

Mandatory in Phase IV-C:

- target variance;
- TD-error variance;
- `q_target_gap`;
- `double_gap`;
- online vs target policy disagreement if applicable;
- across-seed schedule stability;
- margin-estimation variance;
- natural-shift variance;
- correlation between bootstrap-value error and natural-shift error.

Event-conditioned versions must be reported for jackpot / catastrophe / hazard tasks.

---

## 8. Logging additions

Reuse all Phase IV-A/B logging and add advanced fields.

### 8.1 Double-Q fields

- `q_a_current`
- `q_b_current`
- `q_a_next`
- `q_b_next`
- `selected_action_source`
- `evaluation_value_source`
- `double_gap`
- `margin_double`
- `natural_shift_double`

### 8.2 Target-table fields

- `q_online_next`
- `q_target_next`
- `q_target_gap`
- `target_sync_step`
- `target_update_mode`
- `target_polyak_tau`

### 8.3 State-dependent scheduler fields

- `state_bin_id`
- `next_state_bin_id`
- `bin_count`
- `u_stage_ref`
- `u_bin_design`
- `u_bin_used`
- `backoff_weight`
- `statebin_trust_clip_active`
- `statebin_safe_clip_active`

### 8.4 Geometry-priority DP fields

- `residual`
- `geom_gain`
- `priority_score`
- `KL_Bern_to_prior`
- `backup_rank`
- `backup_stage`
- `backup_state`
- `backup_action`

---

## 9. Tests

### 9.1 SafeDoubleQ tests

Create:

```text
tests/algorithms/test_phase4C_safe_double_q.py
```

Required:

1. two Q tables update correctly;
2. selection and evaluation tables are separated;
3. safe target uses the evaluation-side bootstrap;
4. `double_gap` is logged correctly;
5. zero-nonlinearity version matches classical Double Q under the same `gamma_base`.

### 9.2 Target-table tests

Create:

```text
tests/algorithms/test_phase4C_safe_target_q.py
```

Required:

1. hard sync works;
2. Polyak averaging works;
3. safe target uses frozen target table when configured;
4. `q_target_gap` is logged correctly;
5. zero-nonlinearity version matches classical target-table Q under the same `gamma_base`.

### 9.3 State-dependent scheduler tests

Create:

```text
tests/algorithms/test_phase4C_statebin_scheduler.py
```

Required:

1. bins are assigned correctly;
2. bin counts are correct;
3. hierarchical backoff formula is correct;
4. trust and safe caps are applied after backoff;
5. scheduler is frozen during learning;
6. missing / low-count bins back off to stagewise schedule.

### 9.4 Geometry-priority DP tests

Create:

```text
tests/algorithms/test_phase4C_geometry_priority_dp.py
```

Required:

1. priority score is computed correctly;
2. setting all lambdas to zero recovers residual priority;
3. high-activation states receive higher priority when residuals match;
4. planner converges to the same safe value as synchronous VI on toy tasks.

### 9.5 Ablation tests

Create:

```text
tests/algorithms/test_phase4C_ablations.py
```

Required:

1. wrong-sign schedules flip sign as intended;
2. constant-`u` schedule is generated correctly;
3. raw-unclipped schedule bypasses caps only when explicitly configured;
4. trust-region ablation changes trust clip activity;
5. adaptive-headroom ablation changes safe cap utilization.

### 9.6 End-to-end smoke tests

Create:

```text
tests/algorithms/test_phase4C_smoke_runs.py
```

Required:

1. one SafeDoubleQLearning smoke run completes;
2. one SafeTargetQLearning smoke run completes;
3. one SafeTargetExpectedSARSA smoke run completes;
4. one state-dependent scheduler smoke run completes;
5. one geometry-priority DP smoke run completes;
6. aggregation and figure generation run on smoke outputs.

---

## 10. Experiment matrix

### 10.1 RL advanced variants

For each selected activation-suite task where Phase IV-B showed either improvement or ambiguous estimator-noise limitation, run:

1. Phase IV-B SafeQLearning baseline;
2. Phase IV-B SafeExpectedSARSA baseline;
3. SafeDoubleQLearning;
4. SafeTargetQLearning;
5. SafeTargetExpectedSARSA;
6. SafeDoubleQLearning + state-dependent scheduler if feasible;
7. SafeTargetExpectedSARSA + state-dependent scheduler if feasible.

### 10.2 DP advanced variants

For each selected DP task run:

1. classical synchronous VI;
2. safe synchronous VI;
3. safe asynchronous VI;
4. residual-priority async VI;
5. geometry-priority async VI;
6. geometry-priority MPI if feasible.

### 10.3 Scheduler ablations

For each selected activation-suite family run:

1. stagewise natural-shift scheduler;
2. constant-`u` scheduler;
3. wrong-sign scheduler;
4. state-dependent scheduler;
5. state-dependent scheduler with stronger shrinkage;
6. state-dependent scheduler with weaker shrinkage.

### 10.4 Certification ablations

Run on a small subset:

1. trust cap on/off;
2. safe cap on/off;
3. adaptive alpha vs constant alpha;
4. raw-unclipped appendix-only.

---

## 11. Seeds and evaluation protocol

### 11.1 Main advanced runs

Use the same paired seeds as Phase IV-B.

Minimum:

- `5` seeds for all advanced variants;
- `10` seeds for the most important families if Phase IV-B already expanded them.

### 11.2 Pairing rule

Use paired seeds across:

- simple stagewise baseline;
- advanced estimator variants;
- state-dependent variants;
- ablations.

### 11.3 Cross-fitting rule

For state-dependent schedules:

- fit bins / scheduler statistics on pilot seeds or pilot episodes;
- evaluate on disjoint seeds or later episodes when feasible;
- log any leakage limitation.

---

## 12. Figures and tables

### 12.1 Mandatory figures

1. **Estimator-stability comparison**: TD-target variance and TD-error variance.
2. **Double-Q gap diagnostics**: `double_gap` vs natural-shift error.
3. **Target-table diagnostics**: `q_target_gap` and outcome stability.
4. **State-dependent activation localization**: event-conditioned `u` and target-gap distributions.
5. **Geometry-priority DP curves**: residual and backup count vs wall-clock.
6. **Trust-region ablation**.
7. **Adaptive-headroom ablation**.
8. **Wrong-sign ablation**.
9. **Constant-`u` vs stagewise schedule**.

### 12.2 Mandatory tables

1. `P4C-A`: estimator-stability comparison.
2. `P4C-B`: advanced RL performance comparison.
3. `P4C-C`: state-dependent scheduler comparison.
4. `P4C-D`: geometry-priority DP comparison.
5. `P4C-E`: trust / headroom / certification ablations.
6. `P4C-F`: wrong-sign and constant-`u` ablations.

---

## 13. Implementation artifacts to add

### 13.1 Geometry package additions

```text
experiments/weighted_lse_dp/geometry/
  schedule_smoothing.py
  state_bins.py
  geometry_priority.py
```

### 13.2 Algorithm additions

```text
mushroom_rl/algorithms/value/td/
  safe_double_q_learning.py
  safe_target_q_learning.py
  safe_target_expected_sarsa.py
```

### 13.3 Configs

```text
experiments/weighted_lse_dp/configs/phase4/
  advanced_estimators.json
  state_dependent_schedulers.json
  geometry_priority_dp.json
  certification_ablations.json
```

### 13.4 Runners

```text
experiments/weighted_lse_dp/runners/
  run_phase4C_advanced_rl.py
  run_phase4C_geometry_dp.py
  run_phase4C_scheduler_ablations.py
  run_phase4C_certification_ablations.py
  aggregate_phase4C.py
```

### 13.5 Analysis

```text
experiments/weighted_lse_dp/analysis/
  make_phase4C_tables.py
  make_phase4C_figures.py
  estimator_stability_analysis.py
  scheduler_localization_analysis.py
```

---

## 14. Phase IV-C exit criteria

Phase IV-C is complete only if all of the following are true.

1. SafeDoubleQLearning is implemented and tested.
2. SafeTargetQLearning is implemented and tested.
3. SafeTargetExpectedSARSA is implemented and tested.
4. State-dependent frozen schedulers are implemented and tested.
5. Geometry-priority asynchronous DP is implemented and tested.
6. Estimator-stability diagnostics are logged and aggregated.
7. State-dependent scheduler ablations are complete.
8. Trust-region and adaptive-headroom ablations are complete.
9. Wrong-sign and constant-`u` ablations are complete.
10. Raw-unclipped appendix-only ablation is run on a small subset if safe to do so.
11. Figures and tables are generated.
12. `tasks/todo.md` and `tasks/lessons.md` are updated.

At the end of Phase IV-C, answer these questions for each task family:

1. Did estimator stabilization reduce target variance or overestimation?
2. Did estimator stabilization improve Safe TAB outcomes beyond the simple Phase IV-B baseline?
3. Did state-dependent scheduling improve event-conditioned activation localization?
4. Did state-dependent scheduling improve outcomes or only diagnostics?
5. Did geometry-priority DP reduce backups or wall-clock to tolerance?
6. Were trust and safe clips load-bearing in the activated regime?
7. Did wrong-sign scheduling harm or neutralize the effect as expected?
8. Are the final gains still attributable to TAB nonlinearity, or mostly to estimator/scheduler engineering?
