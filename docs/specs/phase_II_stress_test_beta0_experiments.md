# Phase II — Stress-test environments for classical (`beta = 0`) DP/RL

This document is for the coding agent. Treat it as the implementation spec for Phase II.

Phase II keeps the algorithms classical (`beta = 0`) and keeps the experiment harness from Phase I. The only changes are to tasks and environments. The goal is to create controlled stress tests where fixed-discount classical dynamic programming and reinforcement learning are more likely to exhibit exactly the weaknesses the paper claims to address later: inertia, slow adaptation, tail averaging, and failure to emphasize informative transitions.

Do **not** start Phase III code in this phase. Phase II is still purely classical.

---

## 0. Non-negotiable workflow rules for this phase

Use the same rules as Phase I.

1. Start with a written plan in `tasks/todo.md`.
2. Update `tasks/todo.md` continuously and close every task with a short review note.
3. Add every correction and debugging lesson to `tasks/lessons.md`.
4. Reuse the Phase I harness; do not fork the pipeline.
5. Verify every environment modification with tests before running long jobs.
6. If a stress task does not actually degrade the classical baselines, stop and redesign it.

---

## 1. Phase II design principle: make the stress tests faithful to the theory

The paper’s strongest formal claim is local improvement on **aligned** regions, not universal domination everywhere.

For the weighted-LSE operator, the adaptive continuation coefficient is smaller than the classical discount only when the signed margin is aligned with the chosen sign:

- optimistic sign (`beta > 0`): improvement requires `r - v > 0`,
- pessimistic sign (`beta < 0`): improvement requires `r - v < 0`.

Therefore, the Phase II task modifications must create environments where the informative events are represented by **large signed immediate deviations** from the stale continuation estimate, not just by arbitrary difficulty.

Bad Phase II tasks:

- tasks that are simply harder because they are larger,
- tasks that mostly change exploration difficulty but not the Bellman targets,
- tasks where informative events only appear as continuation effects with no sharp local reward signal.

Good Phase II tasks:

- rare positive windfalls / jackpots with immediate rewards,
- rare negative catastrophes with immediate losses,
- reward-map or regime changes that cause the immediate reward to sharply disagree with the old continuation estimate,
- structured distractor/noise settings where averaging irrelevant transitions hurts.

If a modified task does not fit one of those patterns, do not use it in the main paper suite.

---

## 2. Phase II objectives

By the end of Phase II we need:

1. a set of modified stress-test task families paired with the Phase I base tasks,
2. verified evidence that classical `beta = 0` algorithms perform worse on these modified tasks,
3. per-stage and per-event logs that quantify the failure mode,
4. calibration-ready statistics for Phase III,
5. Phase II tables and figures that make the classical limitations obvious before introducing the new method.

---

## 3. What must stay unchanged from Phase I

The following must remain unchanged unless a modification is strictly necessary.

- experiment directory layout,
- result schema (`config.json`, `metrics.json`, `curves.npz`, `timings.json`, `transitions.npz`, `calibration_stats.npz`),
- seed handling,
- aggregation code shape,
- classical algorithms and hyperparameter search protocol,
- time augmentation for the paper-suite RL tasks,
- plotting style.

Phase II should look like “same harness, same algorithms, different task variants.”

---

## 4. Code additions for Phase II

Add new task factories under the experiments package, not the core library, unless a wrapper is generic enough to belong in `mushroom_rl`.

Create:

```text
experiments/weighted_lse_dp/tasks/
  __init__.py
  base_families.py
  stress_families.py
  nonstationary_wrappers.py
  hazard_wrappers.py
```

Recommended rule:

- if the code is a generic environment wrapper, put it under `mushroom_rl/environments/`;
- if the code exists only for this paper’s benchmark suite, keep it under `experiments/weighted_lse_dp/tasks/`.

---

## 5. Stress-task families to implement

Each modified task should reduce to its Phase I base task when the severity parameter is zero. That makes debugging much easier.

### 5.1 Chain family stress tasks (mandatory)

Start from the Phase I chain family.

#### A. `chain_sparse_long`

Purpose:

- stress delayed credit assignment and value propagation.

Specification:

- same left/right chain logic as Phase I,
- increase chain length significantly, e.g. `state_n = 60`,
- reward only on entry to the final goal state,
- no shaping reward,
- `gamma = 0.99`,
- horizon around `120`.

What to measure:

- DP sweep count to propagate the goal reward backward,
- online RL sample complexity to reach a success threshold,
- stagewise Bellman residual profile.

#### B. `chain_jackpot`

Purpose:

- create rare positive tail events that classical averaging underweights.

Specification:

- preserve the same chain state layout,
- add one “jackpot transition” near the later part of the chain,
- on a designated action, with small probability (start with `0.05` or `0.10`) give an immediate positive reward spike, e.g. `+10` or `+20`, and terminate,
- otherwise follow the normal transition dynamics,
- keep a smaller reliable goal reward in the background.

What to measure:

- average return,
- top-decile return,
- probability of learning the jackpot-seeking policy,
- stagewise positive margin quantiles,
- target variance.

#### C. `chain_catastrophe`

Purpose:

- create rare negative tail events that classical expectation underweights.

Specification:

- preserve the same chain layout,
- designate one or more actions as “risky shortcuts,”
- with small probability (start with `0.05` or `0.10`) the risky action gives an immediate large negative reward, e.g. `-10` or `-20`, and terminates,
- otherwise it progresses quickly toward the goal,
- keep a slower safe path with smaller mean return but much better tail risk.

What to measure:

- average return,
- catastrophic event rate,
- CVaR-5% or CVaR-10% of return,
- fraction of episodes using the risky path,
- negative margin quantiles.

#### D. `chain_regime_shift`

Purpose:

- test adaptation after structural breaks.

Specification:

- begin with the Phase I chain task,
- after a fixed global episode index or training-step index, change one of:
  - goal side (right end to left end),
  - action success probability,
  - reward sign/magnitude on one corridor.
- keep the state and action spaces unchanged.

For exact DP/planning:

- treat the regime shift as a model change,
- warm-start the planner from the pre-shift solution,
- measure sweeps to the post-shift optimum.

For RL:

- implement the shift as an environment wrapper driven by global episode/step count.

What to measure:

- adaptation lag after change point,
- area under post-change learning curve,
- number of episodes to recover 90% of new optimum,
- pre/post shift margin distributions.

### 5.2 Grid family stress tasks (mandatory)

Start from the Phase I grid family.

#### A. `grid_sparse_goal`

Purpose:

- cleaner sparse-reward spatial benchmark.

Specification:

- same grid as Phase I,
- only the goal gives reward,
- no per-step shaping,
- possibly increase grid size modestly.

#### B. `grid_hazard`

Purpose:

- rare catastrophic local events in a spatial setting.

Specification:

- add one or more hazard cells or hazard transitions,
- hazards give immediate negative reward spikes and optionally terminate,
- the shortest path should pass near the hazard while a longer safe path exists.

What to measure:

- success rate,
- catastrophic event rate,
- CVaR of return,
- learned state-visitation heatmaps.

#### C. `grid_regime_shift`

Purpose:

- nonstationarity in a spatial benchmark.

Specification:

- keep topology fixed,
- at a change point, move the goal, change slip probability, or flip one reward map region,
- keep state/action spaces unchanged.

What to measure:

- re-planning sweeps for DP,
- post-change adaptation lag for RL,
- state-visitation heatmap before and after shift.

### 5.3 Taxi family stress tasks (recommended, at least one variant mandatory)

Start from the Phase I taxi family.

#### A. `taxi_bonus_shock`

Purpose:

- rare positive windfalls in a combinatorial environment.

Specification:

- one passenger pickup or one route occasionally gives a large immediate bonus,
- keep base taxi dynamics otherwise intact.

#### B. `taxi_hazard`

Purpose:

- rare negative tail events.

Specification:

- designate one route or region as hazardous,
- entering it has a small chance of a large immediate negative reward and reset/termination.

#### C. `taxi_regime_shift`

Purpose:

- nonstationary task reward landscape.

Specification:

- after a change point, swap passenger or destination reward priorities,
- or invert which route is best.

Taxi tasks are valuable if they are stable. If they become too brittle or too slow, keep one modified taxi task in the appendix and do not let it block chain/grid.

### 5.4 Optional stretch tasks (only after mandatory stress suite works)

#### A. `mountaincar_shifted_goal` or `mountaincar_sparse_bonus`

Only implement after the discrete suite is healthy.

If you add a Gymnasium-based stretch task:

- use the Phase I time-feature augmentation,
- keep the same classical algorithm family as Phase I,
- treat it as a stretch result, not as a dependency for the main paper figures.

---

## 6. Algorithms to run in Phase II

Use the **same classical algorithms** as in Phase I.

### 6.1 Exact DP/planning

Run on the stress tasks where a model is available:

1. classical policy evaluation,
2. classical value iteration,
3. classical policy iteration,
4. classical modified policy iteration,
5. classical asynchronous value iteration.

For regime-shift stress tasks, explicitly test warm-started re-planning from the pre-shift solution.

### 6.2 Online RL

Run the same main online algorithms as Phase I:

1. `QLearning`
2. `ExpectedSARSA`

Keep the discount-tuned fixed-`gamma'` classical ablation from Phase I. It remains important here.

---

## 7. Hyperparameter policy for Phase II

Do **not** retune every algorithm from scratch on every stress task with an unconstrained search. That would make later comparisons noisy and unfair.

Use this policy:

1. Start from the Phase I hyperparameters.
2. Allow only a small controlled retuning window, e.g. learning rate multipliers in `{0.5, 1.0, 2.0}` and epsilon variants in a small grid.
3. Use the same tuning budget for all classical algorithms.
4. Keep `gamma` fixed at the task nominal value for the main Phase II classical baselines.
5. Keep the fixed-`gamma'` control ablation as a separate classical line.

Store all tuning grids in config files.

---

## 8. What must be logged in addition to Phase I

Phase II uses the Phase I logging schema and adds stress-specific diagnostics.

### 8.1 Event-level logging

For every run, store explicit binary event arrays:

- `jackpot_event`
- `catastrophe_event`
- `regime_post_change`
- `hazard_cell_hit`
- `shortcut_action_taken`

Use task-specific names if needed, but keep the semantics clear.

### 8.2 Adaptation metrics

For regime-shift tasks, store:

- change-point index,
- pre-change rolling return,
- post-change rolling return,
- lag to 50%, 75%, 90% recovery relative to the new optimum or best observed post-change plateau,
- Bellman residual immediately before and after the shift for DP warm starts.

### 8.3 Tail-risk metrics

For jackpot/catastrophe tasks, store:

- return quantiles,
- CVaR-5% and CVaR-10%,
- top-5% and top-10% return means,
- event rate,
- event-conditioned return.

### 8.4 Target-statistics logging

In addition to the Phase I `margin_beta0` logs, store:

- `aligned_positive = max(margin_beta0, 0)`
- `aligned_negative = max(-margin_beta0, 0)`
- running standard deviation of TD targets,
- running standard deviation of TD errors.

These are crucial for Phase III schedule calibration and for variance-related analysis.

---

## 9. Correctness tests required for Phase II

Add tests under `tests/`.

### 9.1 Stress-task reduction tests

Create:

```text
tests/environments/test_phase2_stress_tasks.py
```

For every modified task family, verify:

1. severity `= 0` recovers the Phase I base task exactly,
2. transition probabilities are valid and sum to one,
3. reward ranges match config,
4. horizons are correct,
5. regime-shift wrappers trigger exactly at the configured change point.

### 9.2 Event logging tests

Create:

```text
tests/algorithms/test_phase2_event_logging.py
```

Verify:

1. event arrays exist,
2. event counts match known deterministic toy cases,
3. post-change flags activate only after the change point,
4. CVaR and quantile computations are numerically stable.

### 9.3 Classical degradation sanity tests

Create:

```text
tests/algorithms/test_phase2_classical_degradation.py
```

These should be short-run tests that verify the modified tasks are not accidentally identical in behavior to the base tasks.

Examples:

- `chain_jackpot` produces heavier right-tail returns than `chain_base`,
- `chain_catastrophe` produces non-zero catastrophic event rate,
- `grid_regime_shift` causes a measurable post-change drop before recovery.

Do not require huge effect sizes in unit tests; just verify the stress mechanism exists.

---

## 10. Phase II metrics to report

### 10.1 Standard aggregate metrics

As in Phase I:

- mean,
- std,
- median,
- IQR,
- bootstrap CI.

### 10.2 Stress-task metrics (mandatory)

#### For nonstationary tasks

- pre-change AUC,
- post-change AUC,
- adaptation lag,
- re-planning sweep count after change,
- recovery time to threshold.

#### For jackpot tasks

- average return,
- top-decile return,
- jackpot capture rate,
- probability of learning jackpot-seeking behavior.

#### For catastrophe tasks

- average return,
- catastrophic event rate,
- CVaR-5%,
- CVaR-10%,
- safe-path selection frequency.

#### For all tasks

- target variance,
- TD error variance,
- aligned margin quantiles.

---

## 11. Phase II figures and tables to generate

### 11.1 Mandatory internal figures

1. **Base vs modified learning curves** for each task family.
2. **Return distribution plots** for jackpot/catastrophe tasks.
3. **Change-point adaptation plots** for regime-shift tasks.
4. **State-visitation heatmaps** for grid tasks.
5. **Per-stage margin quantile plots** comparing Phase I base and Phase II modified tasks.

### 11.2 Mandatory Phase II paper tables

1. **Table P2-A**: task modifications and why they matter.
2. **Table P2-B**: classical DP re-planning/adaptation statistics after regime shift.
3. **Table P2-C**: classical online RL degradation on stress tasks.
4. **Table P2-D**: tail metrics (CVaR, top-decile return, event rate).

The whole point of these tables is to show that classical `beta = 0` methods exhibit exactly the vulnerabilities we want Phase III to address.

---

## 12. Output needed specifically for Phase III calibration

Phase II is the main calibration source.

For every stress-task family, the aggregation script must produce one calibration summary file:

```text
results/weighted_lse_dp/phase2/calibration/<task_family>.json
```

Required contents:

- nominal task gamma,
- reward range and empirical `R_max`,
- stagewise empirical envelope estimates from the classical solution or best classical approximation,
- per-stage quantiles of positive aligned margins,
- per-stage quantiles of negative aligned margins,
- aligned-margin frequency by stage,
- change-point statistics if the task is nonstationary,
- event-conditioned margin statistics,
- recommended task sign for Phase III (`+1` for jackpot/positive-shift families, `-1` for catastrophe families, one sign only per experiment family).

Do not hand-pick these later. Generate them now from code.

---

## 13. Seed and compute policy for Phase II

Minimum main-result seeds:

- `11`
- `29`
- `47`

Because stress tasks can have higher variance, if a task is cheap enough run 5 seeds for that task family.

Always record:

- total wall-clock,
- time before change point,
- time after change point,
- updates/sec,
- DP sweeps/sec for planning tasks.

---

## 14. Phase II exit criteria

Phase II is complete only if all of the following are true.

1. Every modified stress task passes the reduction and validity tests.
2. Classical `beta = 0` baselines were rerun on all mandatory stress-task families.
3. At least one measurable classical weakness is visible for each mandatory stress-task family.
4. Calibration summary JSON files were generated for every mandatory stress-task family.
5. Phase II tables and figures were generated.
6. `tasks/todo.md` contains a completed review section for Phase II.
7. `tasks/lessons.md` includes every environment-design or logging bug found in this phase.

Phase II should end with a clear answer to this question for each task family:

> “What exactly goes wrong for classical fixed-discount DP/RL here, and which logged statistic exposes that failure?”

If that answer is not yet sharp, the phase is not done.

