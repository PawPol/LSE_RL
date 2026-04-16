# Phase I — Classical (`beta = 0`) baselines, harness verification, and calibration-ready logging

This document is for the coding agent. Treat it as the implementation spec for Phase I of the experiment stack.

The purpose of Phase I is not only to run classical baselines. It is to build the exact experimental infrastructure that later phases will reuse unchanged: task factories, wrappers, logging, plotting, tests, and result schemas. Phase I must leave behind a clean, trusted `beta = 0` reference implementation and a calibration-ready data lake for Phase III.

---

## 0. Non-negotiable workflow rules for this phase

These rules are mandatory for every task in this phase.

1. Enter plan mode before any non-trivial change. Write the active plan to `tasks/todo.md` as checkable items.
2. Keep `tasks/todo.md` updated as you progress. Add a short review section at the end of each completed task.
3. After any user correction or self-found mistake, update `tasks/lessons.md` with the failure pattern and the prevention rule.
4. Do not mark anything complete without verification: unit tests, smoke runs, file checks, and metric sanity checks are required.
5. Prefer elegant extensions over hacky patches. Add new modules/classes instead of editing stable MushroomRL code unless there is a clear benefit.
6. Use subagents if your execution environment supports them, especially for parallel code reading, environment design, and result auditing.
7. If any assumption fails, stop and re-plan. Do not push through with a broken architecture.

---

## 1. What was verified in the attached MushroomRL codebase

The attached zip is the `mushroom-rl` dev tree. The following points were checked directly in the repository and should guide the implementation.

### 1.1 Core execution model

- `mushroom_rl/core/core.py` provides `Core.learn(...)` and `Core.evaluate(...)`.
- `Core.learn` builds a `Dataset`, steps the environment, calls `agent.fit(dataset)`, then clears the dataset after each fit.
- Online TD algorithms in `mushroom_rl/algorithms/value/td/td.py` assert `len(dataset) == 1`, so for online TD control/prediction we must use `n_steps_per_fit=1`.
- `Core` supports callbacks through `callbacks_fit` and `callback_step`.

### 1.2 Agent and policy integration

- TD algorithms subclass `TD`, which subclasses `Agent`.
- `TD.__init__` calls `policy.set_q(approximator)` and stores the Q-function in `self.Q`.
- Discrete TD policies live in `mushroom_rl/policy/td_policy.py`:
  - `EpsGreedy`
  - `Boltzmann`
  - `Mellowmax`
- These policies rely on a Q-approximator exposing `.predict(...)` and `.n_actions`.

### 1.3 Discrete tabular convention

- Discrete environments use `mushroom_rl/rl_utils/spaces.py`.
- `Discrete.size` returns a tuple `(n,)`, not an integer.
- Therefore `mdp.info.size` becomes `(n_states, n_actions)` and `Table(mdp.info.size)` is the correct way to build a Q-table.
- Do **not** rewrite this convention.

### 1.4 Existing algorithms we should reproduce before adding new code

Relevant existing TD/value files:

- `mushroom_rl/algorithms/value/td/q_learning.py`
- `mushroom_rl/algorithms/value/td/expected_sarsa.py`
- `mushroom_rl/algorithms/value/td/sarsa.py`
- `mushroom_rl/algorithms/value/td/true_online_sarsa_lambda.py`
- `mushroom_rl/algorithms/value/batch_td/fqi.py`
- `mushroom_rl/algorithms/value/batch_td/lspi.py`
- `mushroom_rl/algorithms/value/dqn/abstract_dqn.py`

Relevant example scripts already in the repo:

- `examples/simple_chain_qlearning.py`
- `examples/grid_world_td.py`
- `examples/double_chain_q_learning/double_chain.py`
- `examples/taxi_mellow_sarsa/taxi_mellow.py`
- `examples/mountain_car_sarsa.py`
- `examples/puddle_world_sarsa.py`

Relevant environments/generators already in the repo:

- `mushroom_rl/environments/finite_mdp.py`
- `mushroom_rl/environments/generators/simple_chain.py`
- `mushroom_rl/environments/generators/grid_world.py`
- `mushroom_rl/environments/generators/taxi.py`
- `mushroom_rl/environments/grid_world.py`
- `mushroom_rl/environments/puddle_world.py`
- `mushroom_rl/environments/gymnasium_env.py`

Relevant tests already in the repo:

- `tests/algorithms/test_td.py`

Use these files as style and correctness references.

### 1.5 Logging utilities already available

- `mushroom_rl/core/logger/logger.py` and `data_logger.py`
- `mushroom_rl/utils/callbacks/collect_dataset.py`
- `mushroom_rl/utils/callbacks/collect_q.py`
- `mushroom_rl/utils/callbacks/collect_max_q.py`

We should reuse the built-in `Logger` where it helps, but for the paper we also need structured `.json` and `.npz` outputs.

---

## 2. Phase I objectives

By the end of Phase I we need all of the following:

1. **Reproduce existing classical MushroomRL baselines** as smoke tests.
2. **Add a finite-horizon exact DP module** to MushroomRL for classical model-based planning on finite MDPs.
3. **Add time augmentation wrappers/utilities** so the same state representation can be used in later safe `beta != 0` phases.
4. **Build the core paper benchmark suite** in the classical `beta = 0` setting.
5. **Log calibration-ready transition statistics** from the classical runs, especially per-stage margins and continuation values.
6. **Generate Phase I tables and figures** for the paper and for internal debugging.

Do not start Phase II until every required correctness test in this document passes.

---

## 3. Repository changes to make in Phase I

Create new code under the repo instead of modifying stable files unless there is a strong reason.

### 3.1 New experiment package

Create:

```text
experiments/weighted_lse_dp/
  README.md
  common/
    io.py
    seeds.py
    timing.py
    metrics.py
    plotting.py
    manifests.py
    schedules.py          # stub in Phase I; real use in Phase III
    time_aug.py
  configs/
    phase1/
      smoke_examples.json
      paper_suite.json
  assets/
    grids/
      phase1_base_grid.txt
      phase1_taxi_grid.txt
  runners/
    run_phase1_smoke.py
    run_phase1_dp.py
    run_phase1_rl.py
    aggregate_phase1.py
  analysis/
    make_phase1_tables.py
    make_phase1_figures.py
```

Use plain Python + JSON configs. Do not add heavy orchestration dependencies.

### 3.2 New exact DP module inside MushroomRL

Add a new subpackage:

```text
mushroom_rl/algorithms/value/dp/
  __init__.py
  finite_horizon_dp_utils.py
  classical_value_iteration.py
  classical_policy_evaluation.py
  classical_policy_iteration.py
  classical_modified_policy_iteration.py
  classical_async_value_iteration.py
```

These classes do **not** need to subclass `Agent`. They are model-based planners operating directly on finite-horizon models. Keep them lightweight and testable.

### 3.3 New generic time augmentation utilities

Add:

```text
mushroom_rl/environments/time_augmented_env.py
```

This file should implement wrappers/utilities for:

1. **Discrete environments**: map base state `s` and stage `t` to one integer state id, e.g. `t * n_base_states + s`.
2. **Continuous/Box environments**: append one extra observation feature for normalized time-to-go or normalized stage index.

The wrapper must:

- preserve rewards and transitions,
- increment stage at every step,
- reset stage to zero on reset,
- expose the new augmented observation space in `MDPInfo`,
- leave the action space unchanged,
- keep the original horizon.

For the paper suite, use time augmentation even for `beta = 0` so that Phase III comparisons are exactly matched.

### 3.4 New result schema

All runners must write to:

```text
results/weighted_lse_dp/
  phase1/
    smoke/
    paper_suite/
      <task>/<algorithm>/seed_<seed>/
        config.json
        metrics.json
        curves.npz
        timings.json
        transitions.npz
        calibration_stats.npz
        stdout.log
```

Required output files:

- `config.json`: full resolved config
- `metrics.json`: scalar summary metrics
- `curves.npz`: learning curves, Bellman residual curves, sweep curves
- `timings.json`: wall-clock, steps/sec, updates/sec, sweep time
- `transitions.npz`: calibration-ready per-transition logs
- `calibration_stats.npz`: aggregated per-stage statistics

---

## 4. Phase I implementation order

Implement in this order.

### Step 4.1 — Editable install and reproducible test environment

From repo root:

```bash
pip install -e .
pip install gymnasium pytest pytest-cov
```

Only add extras if needed for optional stretch tasks.

Record package versions and Python version in:

```text
results/weighted_lse_dp/phase1/environment_manifest.json
```

### Step 4.2 — Reproduce existing example scripts as smoke tests

Run small, controlled reproductions of the following scripts **without changing their semantics**:

1. `examples/simple_chain_qlearning.py`
2. `examples/grid_world_td.py` (small run count only; not the original 10,000)
3. `examples/double_chain_q_learning/double_chain.py` (single-seed smoke)
4. `examples/taxi_mellow_sarsa/taxi_mellow.py` (small run count only)
5. `examples/mountain_car_sarsa.py` (single-seed smoke)
6. `examples/puddle_world_sarsa.py` (single-seed smoke)

What to verify:

- scripts launch,
- environment wrappers work,
- learning produces non-degenerate values/returns,
- result files are produced,
- no silent shape mismatches,
- seeding works.

This smoke phase is not a paper figure. It is a codebase trust check.

### Step 4.3 — Add exact classical finite-horizon DP planners

Implement planners for finite MDPs using the existing `FiniteMDP.p`, `FiniteMDP.r`, `FiniteMDP.mu`, `gamma`, and `horizon`.

At minimum implement:

1. classical policy evaluation,
2. classical value iteration,
3. classical policy iteration,
4. classical modified policy iteration,
5. classical asynchronous value iteration.

Required behavior:

- operate on a finite-horizon time-unrolled model,
- return arrays `Q[t, s, a]`, `V[t, s]`, and greedy policy `pi[t, s]`,
- log Bellman residual per sweep,
- expose total sweeps and wall-clock time,
- support exact backward induction mode for evaluation and value iteration.

Do not fake infinite-horizon stationary planners. For the paper suite, everything must be finite-horizon and stage indexed.

### Step 4.4 — Add time-augmentation wrapper/utilities

Implement and test the wrapper from Section 3.3.

Use it in the paper-suite RL tasks so classical online algorithms already see the same augmented state space that safe `beta != 0` methods will later use.

### Step 4.5 — Build the Phase I paper-suite runners

Implement three runners:

1. `run_phase1_dp.py` for exact DP planners,
2. `run_phase1_rl.py` for tabular online RL,
3. `aggregate_phase1.py` for seed aggregation and calibration-stat extraction.

---

## 5. Phase I paper benchmark suite

This is the suite that will matter for the paper. Keep it small, clean, and tightly matched to the theory.

### 5.1 Mandatory task families

#### A. Chain family

Use `generate_simple_chain(...)` as the base constructor.

Base task `chain_base`:

- `state_n = 25`
- goal state: rightmost state
- `prob = 0.9`
- reward on entry to goal: `+1`
- `gamma = 0.99`
- `horizon = 60`
- time-augment the state for RL runs

Why this task exists:

- simplest sanity check for exact DP,
- easiest place to visualize value propagation,
- easiest calibration source for Phase III.

#### B. Grid family

Use either `GridWorld` or a file-based finite grid via `generate_grid_world(...)`.

Base task `grid_base`:

- 5x5 grid,
- one start cell,
- one goal cell,
- no hazards,
- stochastic action success probability `0.9`,
- goal reward `+1`,
- step reward `0`,
- `gamma = 0.99`,
- `horizon = 80`.

Why this task exists:

- spatial generalization over chain,
- interpretable state heatmaps,
- later Phase II regime-shift and hazard modifications are easy.

#### C. Taxi family

Use `generate_taxi(...)` with a small custom taxi grid stored under `experiments/weighted_lse_dp/assets/grids/phase1_taxi_grid.txt`.

Base task `taxi_base`:

- 1 goal,
- 1 or 2 passengers,
- success probability `0.9`,
- use modest reward magnitudes from the generator,
- `gamma = 0.99`,
- finite horizon (set explicitly; do not leave at `np.inf`).

Why this task exists:

- more complex discrete combinatorial structure,
- validates that the approach is not chain-only.

### 5.2 Optional stretch task family (only after all mandatory tasks pass)

#### D. MountainCar family

Use `Gymnasium(name="MountainCar-v0", ...)` and a time-feature augmentation wrapper.

Only run this if the mandatory finite MDP suite is stable.

Use a classical linear/value-function control baseline that the codebase already supports well. If you cannot cleanly implement a paper-matched classical algorithm here yet, do **not** block the mandatory suite on this task.

---

## 6. Algorithms to run in Phase I

### 6.1 Exact model-based DP (mandatory)

Run on all mandatory task families:

1. classical policy evaluation (fixed policy),
2. classical value iteration,
3. classical policy iteration,
4. classical modified policy iteration,
5. classical asynchronous value iteration.

For fixed-policy evaluation use one deterministic reference policy per task family and store the exact value table.

### 6.2 Online tabular RL (mandatory)

Run on all mandatory task families, using time-augmented states for the paper suite:

1. `QLearning`
2. `ExpectedSARSA`

For reference-only reproduction smoke tests, also run:

3. `SARSA`
4. `TrueOnlineSARSALambda` only on the original example tasks if desired

But the main paper comparisons should center on the algorithms we can later match in Phase III.

### 6.3 Discount-tuning ablation (mandatory)

For every main classical algorithm in the paper suite, run at least one classical fixed-discount ablation with tuned `gamma'`.

Why:

A reviewer may argue that any Phase III gain is just due to a smaller effective discount. We need a direct fixed-`gamma'` control.

Recommended grid:

- `gamma' in {0.90, 0.95, 0.99}` when the nominal task gamma is `0.99`
- or more generally `gamma' in {gamma - 0.09, gamma - 0.04, gamma}` clipped to `(0,1)`.

This ablation is still classical `beta = 0` and belongs in Phase I.

---

## 7. Required instrumentation and what must be stored

This is the most important Phase I requirement for Phase III calibration.

### 7.1 Per-transition logs (mandatory)

For every online RL training run, store in `transitions.npz` arrays with at least:

- `episode_index`
- `t`
- `state`
- `action`
- `reward`
- `next_state`
- `absorbing`
- `last`
- `q_current_beta0`
- `v_next_beta0`
- `margin_beta0 = reward - v_next_beta0`
- `td_target_beta0 = reward + gamma * v_next_beta0`
- `td_error_beta0`

For exact DP runs, store stagewise tables instead of raw transitions when more convenient.

### 7.2 Per-stage aggregate calibration stats (mandatory)

For each task and algorithm, aggregate by stage `t` and store:

- count of samples at stage `t`,
- mean and standard deviation of `reward`,
- mean and standard deviation of `v_next_beta0`,
- quantiles of `margin_beta0` at `{0.05, 0.25, 0.50, 0.75, 0.95}`,
- positive-aligned margin stats `max(margin, 0)`,
- negative-aligned margin stats `max(-margin, 0)`,
- empirical `max_abs_v_next`,
- empirical `max_abs_q_current`,
- Bellman residual statistics if exact DP values are available.

Store these in `calibration_stats.npz`.

### 7.3 Planning curves (mandatory for DP tasks)

For exact DP planners store:

- Bellman residual per sweep,
- sup-norm error to exact solution per sweep,
- sweep count to thresholds `1e-2`, `1e-4`, `1e-6`,
- wall-clock per sweep,
- value table snapshots every sweep for the chain task.

### 7.4 RL learning curves (mandatory)

Store for every seed:

- discounted return per evaluation checkpoint,
- undiscounted return per evaluation checkpoint,
- success rate per checkpoint,
- steps to threshold,
- AUC of the learning curve,
- final 10% average return,
- wall-clock and updates/sec.

Use at least 3 seeds for every main paper result.

---

## 8. Correctness tests required before Phase I is considered done

Add tests under `tests/`.

### 8.1 Exact DP tests

Create:

```text
tests/algorithms/test_classical_finite_horizon_dp.py
```

Required tests:

1. tiny hand-built `FiniteMDP` with known backward-induction solution,
2. value iteration matches backward induction exactly on the horizon DAG,
3. policy evaluation matches backward induction for a fixed policy,
4. policy iteration returns the same optimal policy/value as value iteration,
5. modified PI converges to the same fixed point,
6. asynchronous VI converges to the same optimum.

### 8.2 Time augmentation tests

Create:

```text
tests/environments/test_time_augmented_env.py
```

Required tests:

1. discrete wrapper maps `(t, s)` bijectively to augmented discrete state ids,
2. reset sets `t = 0`,
3. step increments `t`,
4. terminal horizon handling is correct,
5. reward and transition probabilities are unchanged after un-augmenting.

### 8.3 Classical RL regression tests

Create:

```text
tests/algorithms/test_phase1_classical_rl_regression.py
```

Required tests:

1. `QLearning` on a small wrapped finite MDP learns non-trivial Q-values,
2. `ExpectedSARSA` on a small wrapped finite MDP learns non-trivial Q-values,
3. logging files are produced,
4. Phase I runners complete one short smoke run.

### 8.4 Calibration log integrity tests

Create:

```text
tests/algorithms/test_phase1_calibration_logging.py
```

Required tests:

1. `transitions.npz` contains all mandatory arrays,
2. stage indices are valid,
3. margin values equal `reward - v_next_beta0`,
4. aggregated stats match raw transition reductions.

---

## 9. Phase I metrics to report

### 9.1 Standard aggregate metrics

For every main task/algorithm pair report across seeds:

- mean,
- standard deviation,
- median,
- interquartile range,
- 95% bootstrap confidence interval.

### 9.2 DP metrics

- Bellman residual after each sweep,
- sup-norm error to exact optimum,
- sweep count to tolerance,
- wall-clock time,
- policy loss if using approximate stopping.

### 9.3 RL metrics

- discounted return,
- undiscounted return,
- success rate,
- area under learning curve,
- steps to threshold,
- final performance,
- wall-clock,
- updates/sec.

### 9.4 Calibration metrics (must be stored even if not reported in the main text)

- per-stage margin quantiles,
- per-stage max absolute continuation value,
- aligned-margin frequency,
- stagewise visitation counts.

---

## 10. Phase I figures and tables to generate

### 10.1 Mandatory internal figures

1. **Chain value propagation plot**: stagewise value profile by sweep for classical value iteration.
2. **DP residual plot**: residual vs sweep for VI, PI, MPI, async VI on each mandatory task family.
3. **RL learning curves**: Q-learning and Expected-SARSA on each mandatory task family.
4. **Stagewise margin histograms**: one figure per task family, later used for Phase III calibration.
5. **Discount-tuning control**: fixed `gamma'` classical ablation curves.

### 10.2 Phase I paper tables

1. **Table P1-A**: task summary (states, actions, horizon, gamma, reward range).
2. **Table P1-B**: classical DP planner iteration counts and wall-clock.
3. **Table P1-C**: classical online RL returns, AUC, and threshold sample complexity.
4. **Table P1-D**: fixed-`gamma'` ablation.

These tables may later move to appendix, but generate them now.

---

## 11. Seed policy and compute policy

### 11.1 Seeds

Minimum seed list for main results:

- `11`
- `29`
- `47`

For cheap finite-MDP DP tasks, also run a larger cheap audit with 10 seeds if possible.

### 11.2 Compute timing

Always record with `time.perf_counter()`:

- total runtime,
- time in environment stepping,
- time in fitting/updating,
- time in evaluation,
- time in plotting/aggregation.

### 11.3 Stop conditions for smoke and full runs

- Smoke: 1 seed, small horizon, very small step budget.
- Full Phase I paper suite: at least 3 seeds and the full task budget.

Do not jump to full runs before smoke tests pass.

---

## 12. Phase I exit criteria

Phase I is complete only if **all** of the following are true.

1. Existing example scripts listed in Section 4.2 were smoke-tested successfully.
2. Exact classical finite-horizon DP planners were added and unit-tested.
3. Time augmentation wrapper exists and is tested.
4. Classical paper-suite runs completed for chain, grid, and taxi base tasks.
5. Classical fixed-`gamma'` tuning ablation completed.
6. `transitions.npz` and `calibration_stats.npz` were created for every main run.
7. All mandatory figures and tables were generated.
8. `tasks/todo.md` contains a completed review section for Phase I.
9. `tasks/lessons.md` was updated with every bug found during the phase.

Phase I should leave the repo in a state where Phase II only needs to add task modifications, not rebuild infrastructure.

