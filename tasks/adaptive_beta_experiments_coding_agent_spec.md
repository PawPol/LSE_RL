# Coding-Agent Specification: Adaptive-β Experiments for Selective Temporal Credit Assignment

**Target paper:** `neurips_selective_temporal_credit_assignment_positioned.tex`  
**Purpose:** implement a clean, reproducible experimental suite testing whether adaptive signed temperature scheduling improves learning dynamics in adversarial, sparse-reward, and non-stationary tabular environments.

---

## 0. High-Level Goal

Implement experiments that compare classical Bellman learning, fixed-temperature weighted-LSE Bellman learning, and adaptive-β weighted-LSE Bellman learning.

The empirical claim to test is:

> Adaptive β acts as a temporal credit-assignment controller. When the sign of β aligns with the local Bellman advantage, the operator reduces the effective continuation coefficient below the classical discount, accelerating propagation of informative transitions and improving recovery under non-stationarity while remaining stable under clipping.

The experiments should support or falsify the following measurable predictions:

1. Adaptive β increases alignment rate.
2. Adaptive β reduces effective continuation on informative transitions.
3. Adaptive β improves recovery after environment shifts.
4. Adaptive β reduces catastrophic episodes or drawdowns.
5. Adaptive β improves AUC and sample efficiency, even when final asymptotic return is similar.

---

## 1. Paper-Consistent Operator Definition

The paper uses the **centered and scaled weighted log-sum-exp Bellman target**

\[
 g_{\beta,\gamma}(r,v)
 =
 \begin{cases}
 \dfrac{1+\gamma}{\beta}
 \log\left(\dfrac{e^{\beta r}+\gamma e^{\beta v}}{1+\gamma}\right), & \beta \neq 0,\\[1.2ex]
 r+\gamma v, & \beta=0.
 \end{cases}
\]

This is the operator to implement. Do **not** implement the unscaled Bernoulli-prior log-sum-exp as the Bellman target, because that version limits to a weighted average rather than the classical Bellman target.

### 1.1 Numerically Stable Implementation

Implement with log-sum-exp:

```python
def tab_target(r, v, beta, gamma, beta_tol=1e-8):
    if abs(beta) < beta_tol:
        return r + gamma * v
    a = beta * r
    b = math.log(gamma) + beta * v
    m = max(a, b)
    log_num = m + math.log(math.exp(a - m) + math.exp(b - m))
    return (1.0 + gamma) / beta * (log_num - math.log1p(gamma))
```

Vectorized implementation is required for batch logging and value iteration.

### 1.2 Effective Continuation Coefficient

The effective continuation coefficient is

\[
 \rho_{\beta,\gamma}(r,v)
 =
 \frac{e^{\beta r}}{e^{\beta r}+\gamma e^{\beta v}},
\qquad
 d_{\beta,\gamma}(r,v)
 =
 \partial_v g_{\beta,\gamma}(r,v)
 =
 (1+\gamma)(1-\rho_{\beta,\gamma}(r,v)).
\]

Equivalently,

\[
 d_{\beta,\gamma}(r,v)
 =
 \frac{(1+\gamma)\gamma e^{\beta v}}{e^{\beta r}+\gamma e^{\beta v}}.
\]

Implement with stable sigmoid/logit form:

\[
 \rho = \sigma(\beta(r-v)-\log \gamma).
\]

Then

```python
rho = sigmoid(beta * (r - v) - math.log(gamma))
d = (1.0 + gamma) * (1.0 - rho)
```

For β = 0, return `d = gamma`.

### 1.3 Alignment Condition

The paper’s key local condition is

\[
 d_{\beta,\gamma}(r,v) \le \gamma
 \iff
 \beta(r-v) \ge 0.
\]

Log the binary event

```python
aligned = beta * (r - v) > 0
```

Use strict `>` for empirical alignment rate and optionally log non-strict `>=` separately.

---

## 2. Safe β Deployment

All deployed β values must be clipped before entering the operator.

### 2.1 Basic Experimental Clipping

For these experiments, use a simple symmetric cap:

\[
 \widetilde\beta_e = \operatorname{clip}(\beta^{raw}_e,-\bar\beta,\bar\beta).
\]

Default:

```yaml
beta_cap: 2.0
```

Run sensitivity with:

```yaml
beta_cap_grid: [0.5, 1.0, 2.0]
```

### 2.2 Required No-Clipping Ablation

Include an intentionally unsafe ablation:

```yaml
method: adaptive_beta_no_clip
```

This variant should use the same raw adaptive rule but bypass clipping. It is expected to show occasional instability. Log NaNs, overflows, extreme TD targets, and divergence events.

---

## 3. Adaptive β Scheduling

β is updated **between episodes only**. It is fixed within an episode.

### 3.1 Episode Signal

For episode `e`, compute the empirical temporal signal from data collected during the previous episode:

\[
 A_e = \frac{1}{H_e}\sum_{t=0}^{H_e-1}\left(r_t - v_t^{next}\right),
\]

where for Q-learning

\[
 v_t^{next}=\max_{a'}Q(s_{t+1},a')
\]

and for terminal states `v_next = 0`.

Important: use only past data. The β used in episode `e+1` is computed from episode `e`.

### 3.2 Raw Adaptive Rule

\[
 \beta_{e+1}^{raw}
 =
 \beta_{max}\tanh(k A_e).
\]

Then deploy

\[
 \widetilde\beta_{e+1}
 =
 \operatorname{clip}(\beta_{e+1}^{raw},-\bar\beta,\bar\beta).
\]

Default values:

```yaml
beta_max: 2.0
beta_cap: 2.0
k: 5.0
initial_beta: 0.0
```

### 3.3 Smoothing Option

Implement optional exponential smoothing:

\[
 \bar A_e = (1-\lambda)\bar A_{e-1}+\lambda A_e.
\]

Default should be no smoothing:

```yaml
advantage_smoothing_lambda: 1.0
```

Sensitivity values:

```yaml
advantage_smoothing_lambda_grid: [1.0, 0.5, 0.2]
```

---

## 4. Algorithms to Implement

Implement all methods under a common interface.

### 4.1 Required Method Set

| Method ID | Description |
|---|---|
| `vanilla` | Classical Bellman update, β = 0 |
| `fixed_positive` | Fixed β = +β0 |
| `fixed_negative` | Fixed β = −β0 |
| `wrong_sign` | Fixed β with intentionally wrong sign for the environment family |
| `adaptive_beta` | Adaptive sign and magnitude, clipped |
| `adaptive_beta_no_clip` | Adaptive sign and magnitude, no clipping |
| `adaptive_sign_only` | Adaptive sign, fixed magnitude |
| `adaptive_magnitude_only` | Adaptive magnitude, fixed sign |

### 4.2 Q-Learning Update

Primary algorithm for all stochastic environments:

\[
Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha\left[y_t-Q(s_t,a_t)\right],
\]

where

\[
y_t = g_{\widetilde\beta_e,\gamma}(r_t,\max_{a'}Q(s_{t+1},a')).
\]

For terminal states:

\[
y_t = g_{\widetilde\beta_e,\gamma}(r_t,0).
\]

### 4.3 Exploration

Use ε-greedy exploration with shared schedules across methods.

Default:

```yaml
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_episodes: 5000
```

Use identical random seeds and environment sequences across methods.

### 4.4 Learning Rate

Default:

```yaml
alpha: 0.1
gamma: 0.95
```

Keep fixed across methods unless running a separate sensitivity check.

---

## 5. Environments

Implement each environment as a finite-horizon tabular environment with deterministic reset, seed control, and explicit phase logging.

Each environment must expose:

```python
reset(seed=None) -> state
step(action) -> (next_state, reward, done, info)
current_phase -> str/int
oracle_value_or_best_action(optional)
```

The `info` dictionary must include enough metadata for analysis:

```python
info = {
    "phase": ..., 
    "is_shift_step": bool,
    "oracle_action": optional,
    "catastrophe": bool,
    "terminal_success": bool,
}
```

### 5.1 Adversarial Rock-Paper-Scissors

Type: repeated matrix game with adaptive opponent.

Config:

```yaml
horizon: 20
switch_period_episodes: [50, 100]
actions: [rock, paper, scissors]
rewards:
  win: 1.0
  draw: 0.0
  loss: -1.0
```

Opponent phases:

1. `biased_exploitable`: opponent plays one action with high probability, e.g. `[0.7, 0.15, 0.15]`.
2. `counter_exploit`: opponent estimates agent’s recent action distribution and plays the counter to the dominant action.
3. `uniform_random`: `[1/3, 1/3, 1/3]`.

Cycle phases repeatedly.

State representation options:

- minimal: current phase id if visible;
- harder/default: last opponent action and last agent action;
- hidden phase should be used for the main experiment, while visible phase can be an easy diagnostic.

Required variants:

```yaml
rps_visible_phase: false
rps_memory_length: 1
```

### 5.2 Switching Multi-Armed Bandit

Type: non-stationary K-armed bandit.

Config:

```yaml
num_arms: 5
horizon: 1
switch_period_episodes: [100, 250]
reward_type: bernoulli
best_arm_prob: 0.8
other_arm_prob: 0.2
```

Best arm rotates cyclically.

State can be a dummy singleton state. Log regret against the current best arm.

### 5.3 Gridworld with Adversarial Hazards

Type: finite-horizon tabular MDP.

Config:

```yaml
grid_size: [7, 7]
horizon: 50
goal_reward: 10.0
hazard_reward: -10.0
step_reward: -0.01
num_hazards: 5
hazard_switch_period_episodes: [100, 250]
start: [0, 0]
goal: [6, 6]
```

Hazards shift periodically. At each phase, hazards should lie on plausible shortest-path corridors often enough to create catastrophic windows.

Actions:

```yaml
actions: [up, down, left, right]
```

Terminal conditions:

- reaching goal;
- entering hazard;
- horizon expiration.

Log:

- hazard hit rate;
- goal success rate;
- recovery after hazard shift.

### 5.4 Delayed Reward Chain

Type: deterministic sparse-reward MDP.

Config:

```yaml
chain_length: 20
horizon: 25
terminal_reward: 50.0
step_reward: 0.0
fail_reward: 0.0
actions: [forward, reset_or_stay]
```

The `forward` action advances one state. The distractor action either resets to start or keeps the agent in place. Use reset as the default because it creates a meaningful credit-assignment challenge.

Main purpose:

- test optimistic propagation;
- fixed-positive and adaptive-positive phases should improve early learning if alignment is correct.

### 5.5 Self-Play RPS

Type: two-agent non-stationary learning system.

Both agents use the same algorithm variant in the symmetric condition.

Additional asymmetric condition:

- agent A uses adaptive β;
- agent B uses vanilla.

Required outputs:

- exploitability proxy;
- action entropy;
- cycling behavior;
- return variance.

This environment is secondary. Do not let it dominate the paper results unless it produces clear, stable evidence.

---

## 6. Metrics

### 6.1 Core Performance Metrics

Compute for every method, environment, and seed.

| Metric | Definition |
|---|---|
| `final_return` | mean return over last `N_final` episodes |
| `auc_return` | area under the episode-return curve |
| `sample_efficiency` | first episode reaching threshold for `W` consecutive episodes |
| `regret` | cumulative oracle regret where oracle is available |
| `recovery_time` | episodes needed after a shift to recover to pre-shift moving average level |
| `max_drawdown` | largest peak-to-trough drop in smoothed return |
| `catastrophic_episodes` | count of episodes below task-specific threshold |
| `success_rate` | goal/completion rate where applicable |

Defaults:

```yaml
N_final: 500
smoothing_window: 100
threshold_window: 100
```

### 6.2 Mechanism Metrics

These are mandatory because they connect the experiments to the paper theory.

#### Alignment Rate

Per transition:

```python
adv = r - v_next
aligned = beta * adv > 0
```

Per episode:

```python
alignment_rate = mean(aligned_t over transitions with abs(beta) > beta_tol)
```

Also log:

- `mean_signed_alignment = mean(beta * adv)`;
- `frac_positive_signed_alignment = mean(beta * adv > 0)`;
- `mean_abs_advantage = mean(abs(adv))`.

#### Effective Continuation

Per transition:

```python
d_eff = effective_discount(r, v_next, beta, gamma)
```

Per episode log:

- `mean_d_eff`;
- `median_d_eff`;
- `frac_d_eff_below_gamma`;
- `frac_d_eff_above_one` for no-clipping/diagnostic variants;
- `mean_gamma_minus_d_eff`.

#### Advantage Distribution

Save transition-level samples for post-analysis:

```csv
episode, t, seed, method, env, beta, r, v_next, advantage, d_eff, aligned, phase
```

At minimum, save either full transition logs or stratified samples to avoid huge files.

### 6.3 Stability Metrics

| Metric | Definition |
|---|---|
| `bellman_residual` | mean absolute TD error |
| `td_target_abs_max` | max absolute target per episode |
| `q_abs_max` | max absolute Q value |
| `nan_count` | count of NaN/Inf events |
| `divergence_event` | true if NaN/Inf or `q_abs_max > divergence_threshold` |

Default:

```yaml
divergence_threshold: 1e6
```

---

## 7. Experimental Protocol

### 7.1 Main Run Configuration

```yaml
episodes: 10000
seeds: 20
gamma: 0.95
alpha: 0.1
beta0: 1.0
beta_max: 2.0
beta_cap: 2.0
k: 5.0
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_episodes: 5000
```

Use at least 10 seeds for quick development and 20 seeds for final tables.

### 7.2 Development Mode

Provide a quick mode:

```yaml
episodes: 1000
seeds: 3
```

This mode must run in under a few minutes on CPU and validate the full pipeline.

### 7.3 Seed Discipline

For each environment and seed:

- use the same environment phase schedule across methods;
- use matched RNG streams where possible;
- store run config and resolved seed values.

Recommended seed structure:

```python
base_seed = 10000 + seed_id
agent_seed = base_seed + method_offset
common_env_seed = base_seed
```

The environment randomness should be common across methods to support paired statistical comparison.

---

## 8. Statistical Reporting

For every metric report:

- mean;
- standard deviation;
- standard error;
- 95% confidence interval;
- paired difference versus vanilla;
- paired bootstrap CI if easy to implement.

Main tables should compare:

1. adaptive β vs vanilla;
2. adaptive β vs best fixed β;
3. clipped adaptive β vs unclipped adaptive β.

Use paired seeds for all significance summaries.

---

## 9. Figures Required

Generate publication-quality figures in `results/figures/`.

### 9.1 Main Figures

1. `learning_curves_{env}.pdf`  
   Mean return ± standard error across seeds.

2. `regime_shift_recovery_{env}.pdf`  
   Zoom around regime shifts. Align episodes relative to shift time and average across shifts.

3. `alignment_rate_{env}.pdf`  
   Alignment rate over training.

4. `effective_discount_{env}.pdf`  
   Mean effective continuation coefficient over time, with horizontal line at γ.

5. `beta_trajectory_{env}.pdf`  
   β over episodes for adaptive methods.

6. `advantage_histogram_{env}.pdf`  
   Histogram of `r - v_next`, separated by phase or pre/post shift.

### 9.2 Suggested Single “Killer Figure”

Create a 4-panel summary figure:

- Panel A: learning curve after shifts;
- Panel B: recovery time bar plot;
- Panel C: alignment rate over time;
- Panel D: effective discount gap `γ - d_eff` on informative transitions.

Filename:

```text
adaptive_beta_mechanism_summary.pdf
```

This should be the main paper figure if results are strong.

---

## 10. Tables Required

### 10.1 Main Results Table

Columns:

| Environment | Method | Final Return | AUC | Recovery Time | Max Drawdown | Catastrophic Episodes | Align Rate | Mean d_eff |

### 10.2 Ablation Table

Columns:

| Environment | Variant | AUC | Final Return | Align Rate | Divergence Events | Mean d_eff | Notes |

### 10.3 Sensitivity Table

Columns:

| Environment | βmax | βcap | k | AUC | Recovery | Align Rate | Stability |

Save as both `.csv` and `.tex`.

---

## 11. Ablation Studies

### 11.1 β Variant Ablations

Required:

```yaml
methods:
  - adaptive_beta
  - adaptive_beta_no_clip
  - adaptive_sign_only
  - adaptive_magnitude_only
  - fixed_positive
  - fixed_negative
  - vanilla
```

Definitions:

- `adaptive_sign_only`: `beta = beta0 * sign(A_e)` with clipping.
- `adaptive_magnitude_only`: fixed sign chosen by environment, magnitude `beta_max * abs(tanh(k A_e))`.

### 11.2 Sensitivity Grid

Run for at least the strongest two environments:

```yaml
beta_max_grid: [0.5, 1.0, 2.0]
k_grid: [1.0, 5.0, 10.0]
beta_cap_grid: [0.5, 1.0, 2.0]
```

Do not run the full Cartesian product for every environment unless compute is cheap. Prioritize:

1. switching bandit;
2. adversarial RPS;
3. gridworld hazards.

### 11.3 Environment Difficulty

Vary:

- shift frequency;
- reward noise;
- hazard density;
- chain length.

The purpose is not to maximize performance, but to show where the mechanism activates.

---

## 12. Artifact Structure

Create this structure:

```text
experiments/
  adaptive_beta/
    README.md
    configs/
      main.yaml
      dev.yaml
      ablations.yaml
    tab_operator.py
    schedules.py
    agents.py
    envs/
      rps.py
      switching_bandit.py
      hazard_gridworld.py
      delayed_chain.py
      selfplay_rps.py
    run_experiment.py
    analyze.py
    plotting.py
    tests/
      test_operator.py
      test_schedules.py
      test_envs.py
      test_reproducibility.py
results/
  adaptive_beta/
    raw/
    processed/
    figures/
    tables/
    logs/
```

Adjust paths to match the existing repository layout, but preserve the logical separation.

---

## 13. Required Tests

### 13.1 Operator Tests

1. β → 0 recovery:

\[
 g_{0,\gamma}(r,v)=r+\gamma v.
\]

2. finite-difference derivative matches `effective_discount`.

3. alignment condition:

\[
 d_{\beta,\gamma}(r,v) \le \gamma
\iff
\beta(r-v)\ge 0.
\]

4. log-sum-exp stability for large positive/negative inputs.

### 13.2 Schedule Tests

1. β is constant within an episode.
2. β for episode `e+1` uses only data from episode `e` and earlier.
3. clipping never exceeds `beta_cap`.
4. no-clipping variant can exceed cap but should not silently overflow.

### 13.3 Environment Tests

1. phase switches occur at correct episodes.
2. rewards match specification.
3. terminal conditions work.
4. same seed reproduces same environment sequence.

### 13.4 Reproducibility Tests

Run a small dev config twice and assert identical raw logs for deterministic seeds.

---

## 14. Implementation Milestones

### Milestone 1: Operator and Schedule

Deliver:

- `tab_operator.py`;
- `schedules.py`;
- tests passing.

Acceptance criteria:

- β → 0 recovery error below `1e-6`;
- derivative finite-difference error below `1e-5`;
- clipping behavior verified.

### Milestone 2: Environments

Deliver:

- all five environments;
- environment unit tests;
- basic random-agent rollouts.

Acceptance criteria:

- all environments run with controlled seeds;
- phase metadata appears in logs.

### Milestone 3: Agent and Training Loop

Deliver:

- Q-learning agent;
- all methods;
- logging pipeline;
- dev config run.

Acceptance criteria:

- all methods complete `dev.yaml` without errors;
- raw logs contain required fields.

### Milestone 4: Main Experiments

Deliver:

- main runs for at least four environments;
- processed metrics;
- figures and tables.

Acceptance criteria:

- each environment has learning curves, β trajectories, alignment plots, effective-discount plots;
- tables saved as CSV and LaTeX.

### Milestone 5: Ablations and Final Report

Deliver:

- ablation runs;
- sensitivity summaries;
- final markdown report.

Acceptance criteria:

- identify environments where adaptive β helps, hurts, or is neutral;
- report mechanism metrics honestly;
- do not cherry-pick without documenting failed settings.

---

## 15. Logging Schema

### 15.1 Episode-Level CSV

Required columns:

```text
run_id, env, method, seed, episode, phase, beta_raw, beta_deployed,
return, length, epsilon, alignment_rate, mean_signed_alignment,
mean_advantage, mean_abs_advantage, mean_d_eff, median_d_eff,
frac_d_eff_below_gamma, frac_d_eff_above_one, bellman_residual,
td_target_abs_max, q_abs_max, catastrophic, success, regret,
shift_event, divergence_event
```

### 15.2 Transition-Level CSV or Parquet

Required columns:

```text
run_id, env, method, seed, episode, t, state, action, reward,
next_state, done, phase, beta_deployed, v_next, advantage,
td_target, td_error, d_eff, aligned, oracle_action, catastrophe
```

Transition logs may be large. If necessary, store full logs for dev and stratified samples for main runs.

---

## 16. Main Acceptance Criteria

The implementation is complete only when the following are true:

1. Classical β = 0 exactly reproduces standard Bellman Q-learning targets.
2. Fixed β methods use the same code path as adaptive β except for the schedule.
3. Adaptive β is updated only between episodes.
4. All methods use identical environment seeds.
5. Every result table includes alignment and effective-discount diagnostics.
6. At least one regime-shift figure shows recovery around a known shift point.
7. No failed runs are silently dropped.
8. The final report states whether the results support, partially support, or fail to support the paper’s empirical claim.

---

## 17. Recommended Final Report Structure

Create:

```text
results/adaptive_beta/final_report.md
```

with sections:

1. Summary of main findings.
2. Experimental setup.
3. Main performance results.
4. Mechanism diagnostics.
5. Ablations.
6. Failure cases and negative results.
7. Recommended figures/tables for the NeurIPS paper.
8. Open implementation questions.

---

## 18. Important Warnings

1. Do not optimize hyperparameters separately for each method unless explicitly marked as a tuned comparison.
2. Do not choose benchmark variants based only on adaptive β performance.
3. Do not hide no-clipping failures; they are useful stability evidence.
4. Do not use future episode information in β scheduling.
5. Do not conflate the centered/scaled paper operator with the unscaled KL-prior aggregator.
6. Final conclusions should emphasize mechanism evidence, not only raw return.

---

## 19. Minimal Command-Line Interface

Implement commands similar to:

```bash
python experiments/adaptive_beta/run_experiment.py --config experiments/adaptive_beta/configs/dev.yaml
python experiments/adaptive_beta/run_experiment.py --config experiments/adaptive_beta/configs/main.yaml
python experiments/adaptive_beta/analyze.py --results results/adaptive_beta/raw --out results/adaptive_beta/processed
python experiments/adaptive_beta/plotting.py --processed results/adaptive_beta/processed --out results/adaptive_beta/figures
```

Each command should write a machine-readable metadata file recording:

- git commit hash;
- command-line arguments;
- timestamp;
- Python version;
- package versions;
- resolved config.

---

## 20. Deliverables Checklist

- [ ] Operator implementation with tests.
- [ ] Adaptive β schedule implementation with tests.
- [ ] Five environments implemented.
- [ ] Shared Q-learning training loop.
- [ ] Main baseline methods.
- [ ] Ablation methods.
- [ ] Dev run completed.
- [ ] Main runs completed.
- [ ] Ablation runs completed.
- [ ] Figures generated.
- [ ] CSV and LaTeX tables generated.
- [ ] Final report generated.
- [ ] Clear recommendation for what to include in the NeurIPS paper.

