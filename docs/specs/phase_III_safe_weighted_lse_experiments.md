# Phase III — Safe weighted-LSE (`beta \neq 0`) experiments with reverse-engineered stagewise schedules

This document is for the coding agent. Treat it as the implementation spec for Phase III.

Phase III runs the **same task families and experiment harness** from Phases I and II, but replaces the classical fixed-discount Bellman target with the calibrated safe weighted-LSE family. The schedule must be deterministic, stagewise, have a common sign within each experiment family, and be clipped against the certification thresholds implied by the safe theory.

This phase must be built on top of the Phase I/II infrastructure. Do not rebuild the pipeline from scratch.

---

## 0. Non-negotiable workflow rules for this phase

Use the same rules as before.

1. Start with a written plan in `tasks/todo.md`.
2. Keep `tasks/todo.md` current and close each item with a review note.
3. Add every correction to `tasks/lessons.md`.
4. Verify mathematical identities numerically before long runs.
5. Prefer additive extensions (new modules/classes) over invasive edits to stable MushroomRL code.
6. Do not allow the main experiments to depend on an unverified heuristic; every heuristic must be clearly labeled as an ablation.

---

## 1. Phase III objectives

By the end of Phase III we need:

1. safe weighted-LSE model-based DP planners,
2. safe weighted-LSE online RL algorithms matched to the Phase I/II classical baselines,
3. a schedule-calibration pipeline that uses only Phase I/II classical outputs,
4. exact logging of the deployed `beta_t`, clipping activity, responsibilities, and adaptive continuation coefficients,
5. main-result comparisons against the `beta = 0` baselines from Phases I/II,
6. ablations that show the gains are not due to trivial fixed-discount tuning.

---

## 2. Mathematical target to implement

Everything in Phase III should use the **safe** clipped stagewise target, not the raw unclipped operator, except in ablations that explicitly test raw unclipped behavior.

### 2.1 One-step target

For nominal task discount `gamma` and deployed clipped stagewise temperature `\tilde\beta_t`:

- prior: `p0 = 1 / (1 + gamma)`
- responsibility:
  \[
  \rho_t(r, v) = \sigma\bigl(\tilde\beta_t (r - v) + \log(1/\gamma)\bigr)
  \]
- safe weighted-LSE Bellman target:
  \[
  g_t^{\text{safe}}(r, v) =
  \begin{cases}
  \dfrac{1 + \gamma}{\tilde\beta_t}
  \left[
  \log\bigl(e^{\tilde\beta_t r} + \gamma e^{\tilde\beta_t v}\bigr) - \log(1+\gamma)
  \right], & \tilde\beta_t \neq 0, \\
  r + \gamma v, & \tilde\beta_t = 0.
  \end{cases}
  \]
- adaptive continuation derivative:
  \[
  d_t(r,v) = \partial_v g_t^{\text{safe}}(r,v) = (1+\gamma)(1-\rho_t(r,v)).
  \]

Implement the closed form with numerically stable `logaddexp`-style code.

### 2.2 Safe clipping and certification

Given headroom fractions `alpha_t in [0, 1)`:

- certification levels:
  \[
  \kappa_t = \gamma + \alpha_t (1 - \gamma)
  \]
- certified radii:
  \[
  \hat B_T = 0,
  \qquad
  \hat B_t = (1+\gamma) R_{\max} + \kappa_t \hat B_{t+1}
  \]
- stagewise clip cap:
  \[
  \beta^{\text{cap}}_t =
  \frac{\log\!\bigl(\kappa_t / [\gamma (1 + \gamma - \kappa_t)]\bigr)}{R_{\max} + \hat B_{t+1}}
  \]
- deployed stagewise temperature:
  \[
  \tilde\beta_t = \text{clip}(\beta_t^{\text{raw}}, -\beta_t^{\text{cap}}, \beta_t^{\text{cap}}).
  \]

This clipping is mandatory in the main experiments.

---

## 3. Code additions for Phase III

### 3.1 Safe DP planners

Add under:

```text
mushroom_rl/algorithms/value/dp/
  safe_weighted_value_iteration.py
  safe_weighted_policy_evaluation.py
  safe_weighted_policy_iteration.py
  safe_weighted_modified_policy_iteration.py
  safe_weighted_async_value_iteration.py
  safe_weighted_common.py
```

These planners should operate on finite-horizon models just like the classical Phase I planners, but use the stagewise safe target `g_t^{safe}`.

Required outputs:

- `Q[t, s, a]`
- `V[t, s]`
- `pi[t, s]`
- Bellman residual per sweep
- deployed schedule report
- clipping activity summary

### 3.2 Safe online RL algorithms

Add under:

```text
mushroom_rl/algorithms/value/td/
  safe_weighted_lse_base.py
  safe_td0.py
  safe_q_learning.py
  safe_expected_sarsa.py
```

Export them from the relevant `__init__.py` files.

The online algorithms must be matched to the paper and to Phase I/II classical baselines:

1. `SafeTD0` for fixed-policy prediction,
2. `SafeQLearning`,
3. `SafeExpectedSARSA`.

Do **not** make sampled-SARSA the main on-policy algorithm. Use expected-SARSA.

### 3.3 Safe common mixin/utilities

Implement one common helper/mixin that all safe algorithms use.

Required methods:

- `compute_safe_target(r, v_next, t)`
- `compute_rho(r, v_next, t)`
- `compute_effective_discount(r, v_next, t)`
- `compute_kl_term(rho)`
- `clip_beta(raw_beta_t, beta_cap_t)`
- `stage_from_augmented_state(state)`

Required instrumentation fields set after each update/backup:

- `last_stage`
- `last_beta_raw`
- `last_beta_cap`
- `last_beta_used`
- `last_clip_active`
- `last_rho`
- `last_effective_discount`
- `last_target`
- `last_margin`

These fields will drive the logging callbacks.

### 3.4 Calibration package

Add under:

```text
experiments/weighted_lse_dp/calibration/
  build_schedule_from_phase12.py
  calibration_utils.py
  schedule_schema.md
```

Main output:

```text
results/weighted_lse_dp/phase3/calibration/<task_family>/schedule.json
```

### 3.5 Optional appendix-only relaxation module

Only after the core exact safe operators work, optionally add:

```text
experiments/weighted_lse_dp/ablations/
  mc_allocation_relaxation.py
```

This is for the appendix-only Bernoulli / Binary Concrete / Gumbel-Softmax allocation relaxation experiment. It is not part of the main method.

---

## 4. State representation rule in Phase III

For the paper-suite RL tasks, use the **same time-augmented states** introduced in Phase I.

Reason:

The theory is finite-horizon and stage indexed. A stagewise schedule without stage-aware state representation would mix different time-to-go values into one Q-entry and break the intended comparison.

Therefore:

- discrete tasks: use the augmented discrete state id,
- continuous tasks (stretch only): append a time feature.

Do not remove time augmentation for Phase III.

---

## 5. How to reverse-engineer the raw stagewise schedule from Phase I/II

This section is mandatory. The schedule must come from Phase I/II classical outputs, not from hand tuning alone.

## 5.1 Sign selection (mandatory, one sign per experiment family)

Use a **common sign** within each experiment family.

Recommended default:

- `sign = +1` for positive-tail / jackpot / positive regime-change families,
- `sign = -1` for catastrophe / downside-risk families.

For mixed-sign tasks, do not force one main experiment. Split into separate families or treat the wrong-sign case as an ablation.

### 5.2 Calibration source

Use these defaults:

- base task families: calibrate from **Phase I** classical outputs,
- stress task families: calibrate from **Phase II** classical outputs.

Pooled Phase I+II calibration is an ablation, not the default.

### 5.3 Stagewise classical statistics to consume

From the relevant `calibration_stats.npz` or JSON summary, read:

- `gamma`
- empirical `R_max`
- stagewise empirical envelope estimate `B_emp_t`
- positive aligned margin quantiles by stage
- negative aligned margin quantiles by stage
- aligned-margin frequency by stage
- optional task-specific severity scores (adaptation lag, event rate, etc.)

### 5.4 Representative aligned margin per stage

Let `m_t = reward - v_next_beta0` from Phase I/II classical logs.

For a given sign `s in {+1, -1}`, define aligned margins:

\[
 a_t = s \cdot m_t.
\]

Use the empirical 75th percentile of positive aligned margins as the representative magnitude:

\[
 m_t^* = Q_{0.75}(a_t \mid a_t > 0).
\]

If there are too few aligned samples at stage `t`, set `m_t^* = 0` and fall back to `beta_t^{raw} = 0`.

### 5.5 Stagewise informativeness score

Compute a normalized informativeness score per stage from classical data.

Recommended default:

\[
 I_t = \text{normalize}\Bigl(
 Q_{0.75}(a_t \mid a_t > 0)
 \times
 \sqrt{\Pr(a_t > 0)}
 \Bigr)
\in [0,1].
\]

If the task has a change point or rare event statistics, you may multiply by a normalized event importance term, but keep the formula simple and fully reproducible.

### 5.6 Desired local derivative target

Use the classical discount as the reference and request a smaller local derivative on informative aligned stages.

Recommended default mapping:

\[
 d_t^{\text{target}} = \gamma \cdot (1 - \lambda_t),
\qquad
\lambda_t = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) I_t.
\]

Default constants for the main runs:

- `lambda_min = 0.10`
- `lambda_max = 0.50`

That means highly informative stages target a local derivative around `0.5 * gamma`, while weakly informative stages stay close to classical.

### 5.7 Raw `beta` magnitude from the target local derivative

Solve

\[
\frac{(1+\gamma)\gamma}{e^{|\beta_t^{raw}| m_t^*} + \gamma} = d_t^{\text{target}}
\]

for `|beta_t^{raw}|`, giving

\[
|\beta_t^{raw}| =
\frac{1}{m_t^*}
\log\left(
\frac{(1+\gamma)\gamma}{d_t^{\text{target}}} - \gamma
\right).
\]

Then restore the sign:

\[
\beta_t^{raw} = s \cdot |\beta_t^{raw}|.
\]

If `m_t^* = 0`, set `beta_t^{raw} = 0`.

### 5.8 Headroom fractions `alpha_t`

Do **not** present `kappa_t` directly to users as the main knob. When `gamma` is close to 1, that is visually unhelpful.

Use headroom fractions:

\[
\alpha_t \in [0,1),
\qquad
\kappa_t = \gamma + \alpha_t (1-\gamma).
\]

Default main-run rule:

\[
\alpha_t = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) I_t.
\]

Recommended default range:

- `alpha_min = 0.02`
- `alpha_max = 0.10`

Interpretation:

- `alpha_t = 0` means no extra headroom beyond classical contraction,
- larger `alpha_t` means more safe room for the nonlinear operator, but still under a certified global contraction budget.

Use constant-`alpha` ablations later.

### 5.9 Certified radius and clip cap

Use the empirical reward bound and either the theoretical or conservative empirical envelope recursion.

Default main-run rule:

1. set `R_max` to the configured absolute maximum reward for the task,
2. compute `kappa_t` from `alpha_t`,
3. compute `Bhat_t` recursively,
4. compute `beta_cap_t`,
5. clip:
   \[
   \tilde\beta_t = \text{clip}(\beta_t^{raw}, -\beta_t^{cap}, \beta_t^{cap}).
   \]

### 5.10 Schedule JSON schema

For every task family create:

```json
{
  "task_family": "chain_jackpot",
  "gamma": 0.99,
  "sign": 1,
  "source_phase": "phase2",
  "reward_bound": 20.0,
  "alpha_t": [...],
  "kappa_t": [...],
  "Bhat_t": [...],
  "margin_quantile": 0.75,
  "informativeness_t": [...],
  "d_target_t": [...],
  "beta_raw_t": [...],
  "beta_cap_t": [...],
  "beta_used_t": [...],
  "clip_active_t": [...],
  "notes": "derived from phase2 classical calibration summary"
}
```

### 5.11 Required fallback schedules

In addition to the reverse-engineered default schedule, generate these ablation schedules automatically:

1. `beta_zero`: all zeros,
2. `beta_constant_small`: constant signed `beta` before clipping,
3. `beta_constant_large`: constant signed `beta` before clipping,
4. `beta_raw_unclipped`: raw schedule with clipping disabled (appendix only),
5. `alpha_constant_grid`: constant `alpha in {0.00, 0.02, 0.05, 0.10, 0.20}`.

---

## 6. Algorithms to run in Phase III

Run the **same task families** from Phases I and II.

### 6.1 Exact safe DP planners (mandatory)

On every finite-model task family run:

1. safe policy evaluation,
2. safe value iteration,
3. safe policy iteration,
4. safe modified policy iteration,
5. safe asynchronous value iteration.

Compare directly against the matching Phase I/II classical planners.

### 6.2 Safe online RL (mandatory)

Run on the same time-augmented RL tasks:

1. `SafeTD0` for fixed-policy prediction tasks,
2. `SafeQLearning`,
3. `SafeExpectedSARSA`.

Use the same seed sets and evaluation checkpoints as in Phases I/II.

### 6.3 Optional stretch experiments

Only after the mandatory suite is stable:

- time-feature continuous tasks,
- MC allocation relaxation appendix experiment.

These are optional. They must not block the main paper suite.

---

## 7. Required logging in Phase III

Use the Phase I/II schema and add the safe-specific fields.

### 7.1 Per-transition safe logs

For every update, store:

- `stage`
- `beta_raw_t`
- `beta_cap_t`
- `beta_used_t`
- `clip_active`
- `rho_t`
- `effective_discount_t`
- `safe_target`
- `margin_safe = reward - v_next`
- `kl_term`
- `td_error_safe`

### 7.2 Per-stage aggregate safe stats

Aggregate by stage:

- mean and std of `rho_t`,
- mean and std of `effective_discount_t`,
- min/mean/max of `beta_used_t`,
- fraction of updates with clipping active,
- fraction of stages where `effective_discount_t < gamma`,
- empirical Bellman residuals for DP planners.

### 7.3 Calibration provenance logs

For every run, store:

- schedule file path,
- calibration source file path,
- calibration hash or checksum,
- whether the schedule came from Phase I, Phase II, or pooled calibration.

This is necessary for reproducibility.

---

## 8. Correctness tests required for Phase III

Add tests under `tests/`.

### 8.1 Operator identity tests

Create:

```text
tests/algorithms/test_safe_weighted_lse_operator.py
```

Required tests:

1. `g_t_safe(r, v)` equals the classical target `r + gamma v` when `beta_used_t = 0`.
2. Closed-form and variational implementations agree numerically on a grid of `(r, v, beta, gamma)` values.
3. The derivative computed analytically matches finite differences.
4. The computed responsibility is in `(0, 1)`.

### 8.2 Certification tests

Create:

```text
tests/algorithms/test_safe_clipping_certification.py
```

Required tests:

1. for every stage and every grid point in the certified box, `|partial_v g_t_safe(r,v)| <= kappa_t + tol`,
2. `alpha_t = 0` implies `beta_cap_t = 0` and the deployed target collapses to classical,
3. the safe operator maps the certified box into itself.

### 8.3 Classical equivalence tests

Create:

```text
tests/algorithms/test_safe_beta0_equivalence.py
```

Required tests:

1. safe value iteration with zero schedule matches classical value iteration exactly,
2. safe policy evaluation with zero schedule matches classical policy evaluation,
3. safe Q-learning update equals the classical Q-learning update when `beta_used_t = 0`,
4. safe expected-SARSA update equals classical expected-SARSA when `beta_used_t = 0`.

### 8.4 End-to-end smoke tests

Create:

```text
tests/algorithms/test_phase3_smoke_runs.py
```

Required tests:

1. one short safe DP run finishes and logs schedule fields,
2. one short safe Q-learning run finishes and logs `rho_t`, `effective_discount_t`, and clipping activity,
3. aggregation and figure generation scripts run on smoke outputs.

### 8.5 Optional MC-relaxation tests

Only if the appendix relaxation ablation is implemented.

Verify:

1. post-transition Bernoulli Monte Carlo target is unbiased for the exact safe Bellman target,
2. predictive Bernoulli target shows the expected bias sign,
3. Binary Concrete / Gumbel-Softmax relaxation converges toward the hard-sample target as temperature goes to zero.

---

## 9. Main comparisons and mandatory ablations

These ablations are mandatory because they protect the paper against obvious reviewer objections.

### 9.1 Main comparison

For every task family compare:

- classical `beta = 0` baseline from Phase I or Phase II,
- safe weighted-LSE with reverse-engineered clipped stagewise schedule.

### 9.2 Fixed-discount control ablation

Compare against the best classical tuned fixed-`gamma'` baseline from Phases I/II.

Purpose:

Show that the gain is not simply due to using a smaller global discount.

### 9.3 Constant-`beta` ablation

Compare against a schedule with the same sign but constant `beta` before clipping.

Purpose:

Show that stagewise reverse engineering matters.

### 9.4 Wrong-sign ablation

Run the wrong-sign schedule on at least one positive-tail family and one catastrophe family.

Purpose:

Show that sign alignment matters and the effect is not generic “more nonlinearity = better.”

### 9.5 Raw-unclipped ablation (appendix or internal)

Run the raw schedule without clipping on a small subset only.

Purpose:

Demonstrate why the safe calibration is needed.

### 9.6 `alpha` / headroom ablation

Run constant `alpha` in `{0.00, 0.02, 0.05, 0.10, 0.20}`.

Purpose:

Show the stability-flexibility tradeoff explicitly.

### 9.7 Calibration-source ablation

For at least one family compare schedules derived from:

- Phase I only,
- Phase II only,
- pooled Phase I+II.

Purpose:

Show that the reverse-engineering source matters but the main conclusion is stable.

### 9.8 Optional relaxation ablation

If implemented, compare:

- exact safe target,
- post-transition Bernoulli Monte Carlo target,
- predictive Bernoulli surrogate,
- Binary Concrete / Gumbel-Softmax surrogate.

This belongs in the appendix, not the main comparison table.

---

## 10. Phase III metrics to report

### 10.1 Standard aggregate metrics

As in Phases I/II:

- mean,
- std,
- median,
- IQR,
- bootstrap CI.

### 10.2 DP-specific metrics

- Bellman residual per sweep,
- sup-norm error to exact optimum,
- sweeps to threshold,
- local derivative statistics,
- fraction of state-stage points with `effective_discount < gamma`,
- wall-clock and overhead factor vs classical.

### 10.3 RL-specific metrics

- discounted return,
- undiscounted return,
- success rate,
- AUC,
- steps to threshold,
- adaptation lag,
- event rates,
- CVaR / top-decile return where relevant,
- update-target variance.

### 10.4 Safe-operator diagnostics (mandatory)

- mean and quantiles of `rho_t`,
- mean and quantiles of `effective_discount_t`,
- clip activation rate,
- fraction of updates where `effective_discount_t < gamma`,
- deployed `beta_t` histograms,
- empirical target bias/variance if using optional relaxations.

---

## 11. Phase III figures and tables to generate

### 11.1 Mandatory main-text figures

1. **Effective discount vs classical gamma**: empirical `effective_discount_t` distributions with a horizontal `gamma` reference line.
2. **Planning residual curves**: classical vs safe DP planners on exact model tasks.
3. **Learning curves**: classical vs safe online RL on base and stress tasks.
4. **Regime-shift adaptation**: post-change recovery curves.
5. **Return distribution plots**: catastrophe and jackpot tasks.
6. **Clip activity / deployed beta plot** by stage.

### 11.2 Mandatory appendix figures

1. `alpha` ablation,
2. constant-`beta` vs stagewise schedule,
3. wrong-sign ablation,
4. fixed-`gamma'` control,
5. optional MC-relaxation ablation.

### 11.3 Mandatory Phase III tables

1. **Table P3-A**: main performance comparison (classical vs safe).
2. **Table P3-B**: planning iteration/sweep counts and wall-clock.
3. **Table P3-C**: nonstationary adaptation metrics.
4. **Table P3-D**: tail-event metrics (CVaR, top-decile, event rate).
5. **Table P3-E**: compute overhead and clip activity.

---

## 12. Seed and compute policy for Phase III

Use the same main seeds as in Phases I/II:

- `11`
- `29`
- `47`

If a task family is cheap, run 5 seeds.

Always report the overhead ratio:

\[
\text{overhead} = \frac{\text{safe runtime}}{\text{classical runtime}}
\]

for matched task/algorithm/seed configurations.

---

## 13. Practical implementation notes specific to MushroomRL

These should be followed closely.

### 13.1 Do not patch the classical algorithms in place

Add new classes rather than editing:

- `QLearning`
- `ExpectedSARSA`
- `TD`

This preserves the classical baselines and reduces regression risk.

### 13.2 Keep the online update API compatible with `Core`

The online safe TD algorithms must still work with:

- `Core.learn(n_steps=..., n_steps_per_fit=1)`
- `Core.evaluate(...)`

Do not redesign the training loop.

### 13.3 Decode stage from the augmented state

The safe algorithms should not rely on hidden mutable global episode counters for stage indexing in the main paper suite. They should read the stage from the augmented state representation whenever possible.

### 13.4 Use stable numerics

For non-zero `beta`:

- use `np.logaddexp` or equivalent stable implementations,
- avoid direct exponentiation when arguments are large,
- special-case `beta = 0` explicitly.

### 13.5 Preserve serialization

Follow MushroomRL conventions:

- use `_add_save_attr(...)`,
- implement `_post_load(...)` when necessary,
- add export lines to the package `__init__.py` files.

### 13.6 Add test coverage before long runs

The repo already has strong TD tests under `tests/algorithms/test_td.py`. Mirror that style.

---

## 14. Phase III exit criteria

Phase III is complete only if all of the following are true.

1. Safe weighted-LSE DP planners exist and pass the operator/certification tests.
2. Safe online RL algorithms exist and pass the `beta = 0` equivalence tests.
3. Reverse-engineered schedules were generated automatically from Phase I/II outputs.
4. Main classical-vs-safe comparisons were run on all mandatory base and stress task families.
5. Mandatory ablations were run.
6. Main-text and appendix figures/tables were generated.
7. All result directories contain `schedule.json` or a clear reference to the schedule used.
8. `tasks/todo.md` contains a completed review section for Phase III.
9. `tasks/lessons.md` contains every bug found during safe-operator implementation and calibration.

At the end of this phase, you should be able to answer these questions clearly for each task family:

1. What schedule was derived from the classical logs?
2. How often was it clipped?
3. Did the deployed effective discount become smaller than classical on the informative states?
4. Did that correspond to better planning speed, adaptation, tail handling, or sample efficiency?
5. Could the same gain be reproduced by merely lowering the fixed classical discount?

If any of those answers is missing, the phase is not complete.

