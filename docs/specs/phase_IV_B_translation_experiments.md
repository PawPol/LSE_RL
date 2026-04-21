# Phase IV-B — Translation experiments for activated Safe TAB

This document is for the coding agent. Treat it as the **second, self-standing Phase IV-B implementation spec** for the TAB experiment stack.

Phase IV-B begins only after Phase IV-A has passed the activation gate. Phase IV-A establishes that Safe TAB is genuinely nonclassical under certification on a frozen activation suite. Phase IV-B asks whether that certified activation **translates** into better outcomes.

The goal is not merely “does average return improve?” The goal is to test whether stronger certified TAB diagnostics translate into at least one of:

- tail-risk improvement;
- adaptation after change;
- sample-efficiency improvement;
- planning speed / prioritization efficiency;
- estimator stability.

Phase IV-B must preserve interpretability. The main empirical claim should be supported by the simplest scheduler/algorithm variant that demonstrates certified activation and matched-control improvement. More complex variants belong to Phase IV-C.

---

## 0. Non-negotiable workflow rules

1. Start with a written plan in `tasks/todo.md`.
2. Keep `tasks/todo.md` current and close each completed item with a short review note.
3. Add every correction, debugging lesson, and translation-analysis surprise to `tasks/lessons.md`.
4. Do not overwrite Phase I/II/III/IV-A outputs.
5. Use the frozen activation suite from Phase IV-A. Do not redesign tasks based on safe returns.
6. Use paired seeds across matched comparisons.
7. When `gamma_base != gamma_eval`, always compare:
   - classical matched-`gamma_base`,
   - safe zero-nonlinearity matched-`gamma_base`,
   - safe nonlinear matched-`gamma_base`.
8. Outcome claims must be paired with activation diagnostics.
9. Report nulls honestly. Activated TAB without outcome improvement is a valid result.
10. Do not introduce SafeDoubleQ, target-table variants, or state-dependent schedulers as mainline unless the simple stagewise scheduler has already been evaluated. Those belong to Phase IV-C.

---

## 1. Phase IV-B objectives

Phase IV-B is complete only if it can answer the translation and mechanism questions clearly.

### 1.1 Translation question

When certified diagnostics become nontrivial, do they translate into improvements in at least one of:

- tail-risk;
- adaptation after change;
- sample-efficiency;
- planning speed / prioritization efficiency;
- estimator stability?

### 1.2 Mechanism question

If gains appear, are they due to:

- TAB nonlinearity itself;
- lower-base-`gamma` alone;
- schedule quality;
- estimator stabilization;
- or some interaction among them?

Phase IV-B focuses on the first three causes. Estimator stabilization is reserved primarily for Phase IV-C.

---

## 2. Inputs from Phase IV-A

Before Phase IV-B starts, verify that these Phase IV-A artifacts exist:

```text
results/weighted_lse_dp/phase4/audit/
results/weighted_lse_dp/phase4/task_search/selected_tasks.json
results/weighted_lse_dp/phase4/counterfactual_replay/
experiments/weighted_lse_dp/configs/phase4/activation_suite.json
experiments/weighted_lse_dp/configs/phase4/gamma_matched_controls.json
```

Required Phase IV-A reports:

1. `phase3_compat_report.md`;
2. `activation_search_report.md`;
3. counterfactual replay summaries;
4. activation-gate report.

Stop if the selected task family is not labeled as an activated Safe TAB case or an event-conditioned activation case.

---

## 3. Mainline algorithms for Phase IV-B

Keep the first translation study simple.

### 3.1 RL mainline

Run at least:

1. classical Q-learning;
2. classical ExpectedSARSA;
3. SafeQLearning with Phase IV-A stagewise natural-shift schedule;
4. SafeExpectedSARSA with Phase IV-A stagewise natural-shift schedule.

Do not make SafeDoubleQLearning or target-table algorithms the first mechanism test.

### 3.2 DP mainline

Run at least:

1. classical synchronous value iteration;
2. safe synchronous value iteration;
3. classical asynchronous value iteration;
4. safe asynchronous value iteration.

Geometry-priority DP belongs to Phase IV-C unless explicitly used as a secondary ablation.

### 3.3 Safe-zero control

For every safe nonlinear run, run a safe-zero control through the same Safe TAB code path with:

```text
u_target = 0
theta_used = 0
beta_used = 0
```

under the same `gamma_base`.

This isolates implementation-path effects from TAB nonlinearity.

---

## 4. Mandatory experiment matrix

### 4.1 Suites to run

Run two suites.

#### A. Negative-control replay suite

Use the original Phase III paper suite through the Phase IV code path.

Purpose:

- backward compatibility;
- low-activation control;
- calibration sanity;
- demonstrate that when activation diagnostics are near zero, outcome differences are also small.

#### B. Activation suite

Use the frozen activation suite from Phase IV-A.

Purpose:

- main Phase IV-B analysis;
- genuine TAB mechanism test;
- performance translation test.

### 4.2 Scheduler variants to compare

For each activation-suite family run at least:

1. classical target, `gamma = gamma_eval`;
2. classical matched-`gamma_base` target;
3. safe zero-nonlinearity matched-`gamma_base`;
4. Phase IV natural-shift stagewise scheduler;
5. Phase IV natural-shift stagewise scheduler + adaptive headroom;
6. Phase IV natural-shift stagewise scheduler + lower-base-`gamma`, if Phase IV-A selected it.

### 4.3 Diagnostic-strength sweep

For each activation-suite family, run a small sweep over reference effect strength:

```text
u_max in {0.0, 0.005, 0.010, 0.020}
```

or the equivalent `delta_discount_target` sweep approved in Phase IV-A.

Purpose:

- verify that operator diagnostics increase monotonically;
- test whether outcomes improve with that increase;
- support translation analysis.

### 4.4 Lower-base-`gamma` matched controls

For every selected task family and every candidate `gamma_base`, run:

1. classical matched-`gamma_base`;
2. safe zero-nonlinearity matched-`gamma_base`;
3. safe nonlinear matched-`gamma_base`.

Interpretation:

- classical matched vs safe zero: implementation / estimator-path effects;
- safe zero vs safe nonlinear: TAB nonlinearity effect;
- classical matched vs safe nonlinear: total effect under the same `gamma_base`.

---

## 5. Primary outcomes by task family

Predefine the primary outcome for each task family before running full experiments.

| Task family | Primary outcome | Secondary outcomes |
|---|---|---|
| chain sparse-credit | steps to threshold / early AUC | final return, DP sweeps |
| chain jackpot | top-decile return / jackpot capture | mean return, target variance |
| chain catastrophe | CVaR / risky-action frequency | mean return, catastrophe event rate |
| grid hazard | CVaR / hazard-hit rate / safe-detour success | mean return, success rate |
| regime shift | post-change AUC / recovery lag | final post-change return, policy-switch lag |
| taxi bonus | jackpot/bonus capture if stable | mean return, variance |

Do not choose the primary metric after seeing which metric improves.

---

## 6. Required logging

Reuse all Phase IV-A activation logging and add learning-specific fields.

### 6.1 Per-transition / per-backup logging

Store at least:

- `stage`
- `gamma_eval`
- `gamma_base`
- `reward_bound`
- `A_t`
- `xi_ref_t`
- `u_target_t`
- `u_ref_used_t`
- `theta_used_t`
- `beta_used_t`
- `margin = reward - v_next`
- `natural_shift = beta_used_t * margin`
- `rho_used`
- `effective_discount_used`
- `delta_effective_discount = effective_discount_used - gamma_base`
- `safe_target`
- `classical_target_same_gamma_base`
- `safe_target_gap_same_gamma_base`
- `td_error_safe`
- `td_error_classical_matched` when available
- task-specific event flags

### 6.2 Learning-curve logging

For every evaluation checkpoint store:

- discounted return under `gamma_eval`;
- discounted return under `gamma_base`;
- undiscounted return;
- success rate;
- primary task-specific outcome;
- CVaR / top-decile / adaptation lag where relevant;
- event rates;
- wall-clock and updates/sec.

### 6.3 Aggregate geometry diagnostics

Aggregate by stage, globally, and event-conditioned:

- mean/std/quantiles of `natural_shift`;
- fraction with `|natural_shift| >= 5e-3`;
- mean/std/quantiles of `delta_effective_discount`;
- fraction with `|delta_effective_discount| >= 1e-3`;
- mean/std/quantiles of `safe_target_gap_same_gamma_base`;
- fraction with `|safe_target_gap_same_gamma_base| >= 5e-3 * reward_bound`;
- trust clip fraction;
- safe clip fraction;
- cap utilization ratio;
- aligned occupancy;
- mean KL to prior.

---

## 7. Standard outcome metrics

As before:

- mean return;
- median return;
- AUC;
- steps to threshold;
- success rate;
- CVaR-5% / CVaR-10% where relevant;
- adaptation lag;
- event rates;
- wall-clock / sweep count.

Use paired bootstrap confidence intervals whenever possible.

---

## 8. Estimator-stability metrics in Phase IV-B

Even before Phase IV-C advanced algorithms, log:

- TD-target variance;
- TD-error variance;
- across-seed schedule stability;
- online value estimate variance;
- bootstrap-value variance by stage;
- margin-estimation variance by stage.

These are diagnostic in Phase IV-B and become central in Phase IV-C.

---

## 9. Mandatory translation analysis

### 9.1 Step 1 — activation verification

For every activation-suite family verify that moving from matched classical / safe-zero to safe nonlinear TAB increases at least some of:

- `|natural_shift|`;
- `|delta_effective_discount|`;
- `|safe_target_gap_same_gamma_base|`.

If this fails, that family is not an activated Safe TAB case and should be labeled as such.

### 9.2 Step 2 — within-family translation sweep

Use the diagnostic-strength sweep and test whether stronger deployed activation predicts stronger downstream changes.

Required analyses:

1. plot outcome delta vs diagnostic delta;
2. compute Spearman correlation between:
   - `mean_abs_natural_shift_delta` and outcome delta;
   - `mean_abs_target_gap_delta` and outcome delta;
   - `mean_abs_delta_effective_discount_delta` and outcome delta;
3. bootstrap paired differences across seeds.

### 9.3 Step 3 — matched-control mechanism isolation

For lower-base-`gamma` runs compare:

1. classical matched-`gamma_base`;
2. safe zero-nonlinearity matched-`gamma_base`;
3. safe nonlinear matched-`gamma_base`.

Interpretation:

- (1) vs (2): implementation / estimator-path effects;
- (2) vs (3): TAB nonlinearity effect;
- (1) vs (3): total effect.

### 9.4 Step 4 — outcome-specific interpretation

Do not collapse everything to average return.

Allowed conclusions include:

- “TAB was activated but the task did not reward that activation.”
- “Activation helped tail-risk but not mean return.”
- “Activation improved adaptation but not sample-efficiency.”
- “Activation reduced target variance but did not move policy outcomes.”

### 9.5 Step 5 — negative-control consistency

For the Phase III replay suite, verify:

- activation diagnostics remain near zero;
- outcome differences are small;
- any large outcome difference is investigated as a possible implementation or stochastic artifact.

---

## 10. Tests

### 10.1 Translation-analysis tests

Create:

```text
tests/algorithms/test_phase4B_translation_analysis.py
```

Required:

1. paired-difference computations are correct;
2. Spearman correlations are computed on matched task/scheduler/seed groups;
3. bootstrap paired confidence intervals preserve seed pairing;
4. null translation cases are reported rather than silently dropped.

### 10.2 Outcome-metric tests

Create:

```text
tests/algorithms/test_phase4B_outcome_metrics.py
```

Required:

1. CVaR calculations are correct;
2. top-decile return calculations are correct;
3. adaptation-lag calculations are correct;
4. event-conditioned metrics match event flags;
5. primary metric assignment is loaded from config, not chosen after aggregation.

### 10.3 Matched-control tests

Reuse and extend:

```text
tests/algorithms/test_phase4_gamma_matched_controls.py
```

Required:

1. every lower-base-`gamma` safe nonlinear run has a classical matched control;
2. every lower-base-`gamma` safe nonlinear run has a safe-zero control;
3. all target gaps are reported against `classical_target_same_gamma_base`;
4. aggregation refuses to compare lower-base safe nonlinear against nominal classical alone.

### 10.4 End-to-end smoke tests

Create:

```text
tests/algorithms/test_phase4B_smoke_runs.py
```

Required:

1. one short activation-suite RL run completes;
2. one short activation-suite DP run completes;
3. diagnostic-strength sweep runs on one toy task;
4. matched controls are emitted and aggregated;
5. translation-analysis script produces tables and figures.

---

## 11. Seeds, power, and evaluation protocol

### 11.1 Main activation-suite runs

Use at least `5` seeds for all activation-suite families.

For the most important three or four activation-suite families, expand to `10` seeds once the suite is frozen and the smoke runs pass.

### 11.2 Pairing rule

Use paired seeds across:

- classical nominal;
- classical matched-`gamma_base`;
- safe zero-nonlinearity;
- safe nonlinear;
- diagnostic-strength sweep variants.

### 11.3 Search-stage outputs are frozen

Do not update activation-suite selection after seeing Phase IV-B outcomes.

### 11.4 Reporting rule

Report mean, standard deviation, median, IQR, and paired bootstrap confidence intervals for all main metrics.

---

## 12. Figures and tables

### 12.1 Mandatory main figures

1. **RL learning curves** on the activation suite.
2. **Tail-risk / adaptation / sample-efficiency outcomes** for activated tasks.
3. **Outcome delta vs diagnostic delta** scatter plots.
4. **Lower-base-`gamma` matched-control comparison**.
5. **Diagnostic-strength sweep**: activation metrics and outcomes vs `u_max`.
6. **Negative-control replay outcome comparison**.

### 12.2 Mandatory appendix figures

1. per-stage activation diagnostics for all selected tasks;
2. event-conditioned activation diagnostics;
3. rejected candidate summaries inherited from Phase IV-A;
4. null translation cases;
5. paired-seed difference plots.

### 12.3 Mandatory tables

1. `P4B-A`: main performance comparison on the activation suite.
2. `P4B-B`: operator-activation diagnostics for the same runs.
3. `P4B-C`: matched classical vs safe-zero vs safe-nonlinear controls.
4. `P4B-D`: diagnostic-strength sweep summary.
5. `P4B-E`: translation-analysis summary.
6. `P4B-F`: negative-control replay summary.

---

## 13. Implementation artifacts to add

### 13.1 Configs

```text
experiments/weighted_lse_dp/configs/phase4/
  translation_study.json
  diagnostic_strength_sweep.json
  primary_outcomes.json
```

### 13.2 Runners

```text
experiments/weighted_lse_dp/runners/
  run_phase4_dp.py
  run_phase4_rl.py
  run_phase4_diagnostic_sweep.py
  aggregate_phase4B.py
```

### 13.3 Analysis

```text
experiments/weighted_lse_dp/analysis/
  make_phase4B_tables.py
  make_phase4B_figures.py
  translation_analysis.py
  paired_bootstrap.py
```

---

## 14. Phase IV-B exit criteria

Phase IV-B is complete only if all of the following are true.

1. Phase IV-A activation gate passed for at least two main activation-suite families.
2. Negative-control replay suite runs successfully through Phase IV-B aggregation.
3. Activation-suite RL and DP runs complete for the required matched comparisons.
4. Lower-base-`gamma` comparisons include classical matched controls and safe-zero controls.
5. Diagnostic-strength sweep completes.
6. Translation analysis is complete.
7. Outcome-specific primary metrics are reported.
8. Nulls are reported honestly.
9. Figures and tables are generated.
10. `tasks/todo.md` and `tasks/lessons.md` are updated.

At the end of Phase IV-B, answer these questions for each task family:

1. Was this task actually activated under certification during learning?
2. Did activation translate into better tail-risk, adaptation, sample-efficiency, planning speed, or estimator stability?
3. Did lower-base-`gamma` help because of smaller discount alone, TAB nonlinearity, or both?
4. Did stronger `u_max` produce stronger diagnostics?
5. Did stronger diagnostics predict stronger outcome changes?
6. If activation did not translate, was the failure due to task design, schedule design, estimator noise, or theory-to-outcome mismatch?
