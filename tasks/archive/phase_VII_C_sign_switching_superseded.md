> **SUPERSEDED** by Phase VIII M9 — see
> docs/specs/phase_VIII_tab_six_games.md §10/M9 and
> experiments/adaptive_beta/tab_six_games/composites/sign_switching.py.
> Original Phase VII-C content preserved below for provenance.
> Archived 2026-04-30 in commit landing this header.
>
> ---
>
> [original content follows]

# Phase VII-C — Sign-Switching Adaptive-β Games and β-Speed Self-Play Duels

## 0. Purpose

This specification defines the next Phase VII follow-up after the Phase VII-B strategic-learning run.

Phase VII-B showed that the broader β-family can outperform vanilla RL, but fixed negative β dominated adaptive β in the strongest Shapley strategic settings. This is useful, but it does **not** yet support the adaptive-β claim. The likely reason is that the tested strategic environments had a nearly stationary optimal β sign: pessimistic temporal allocation was useful almost everywhere.

Phase VII-C should test the setting where adaptive β is actually necessary:

> No single fixed β sign is globally optimal because the useful temporal-credit bias changes across regimes.

The goal is to design environments where:
- fixed-positive β is best in some regimes,
- fixed-negative β is best in other regimes,
- vanilla β = 0 cannot exploit either bias,
- adaptive β can switch signs quickly enough to outperform all fixed baselines.

---

## 1. Main Experimental Claim

The target claim is:

> Adaptive β is useful when the optimal temporal-credit bias changes over time or across strategic regimes. In sign-switching environments, adaptive β can outperform vanilla RL and both fixed-positive and fixed-negative β by tracking the locally useful temporal allocation sign.

Do **not** claim that adaptive β universally dominates fixed β. The intended claim is conditional:

```text
adaptive β > fixed β when the optimal β sign changes across regimes
fixed β may dominate when the optimal β sign is stationary
```

---

## 2. Non-Negotiable Constraints

1. **Do not duplicate the TAB / safe weighted log-sum-exp operator.**
   - Reuse the shared kernel:
     `src/lse_rl/operator/tab_operator.py`

2. **Do not change stable Phase III–VI behavior.**
   - If stable infrastructure is touched, record the justification in:
     `tasks/lessons.md`

3. **No silent run drops.**
   - Every `(env, method, seed)` and every duel pairing must be accounted for.

4. **No main-paper edits yet.**
   - Produce paper-update proposals only.
   - Do not directly edit the `.tex` file unless explicitly authorized later.

5. **Use paired seeds.**
   - All methods must use the same seed grid.
   - Duel experiments must swap player order.

6. **Include oracle β as a diagnostic upper bound.**
   - Do not test adaptive β on a composite game unless oracle β first beats both fixed signs.

7. **Treat fixed-positive and fixed-negative as members of our model family.**
   - They are valid TAB / β-family baselines.
   - But they do not support the adaptive-β claim unless adaptive β beats them in sign-switching settings.

---

## 3. Interpretation of Phase VII-B

The Phase VII-B result should be treated as follows:

```text
fixed_negative > adaptive_beta >? vanilla
```

This means:
- β-family temporal credit assignment matters;
- vanilla β = 0 is not always optimal;
- fixed-negative is a strong baseline;
- adaptive β must now be tested in environments where fixed-negative is too rigid.

The mechanism failure mode was:

```text
adaptive β saturated to β = -2.0 within ~200 episodes
```

This indicates that the environment had a stationary pessimistic optimum. Phase VII-C should intentionally avoid this by constructing sign-switching regimes.

---

## 4. Required New Experiments

Phase VII-C has four required components:

1. **Sign-specialization pre-screen**
2. **Composite sign-switching games**
3. **Adaptive β speed ablation**
4. **Two-player β-speed self-play duels**

Optional components:
- state-conditioned β,
- transition-conditioned β,
- additional strategic adversaries.

---

# PART I — Sign-Specialization Pre-Screen

## 5. Goal

Find at least one setting where fixed-positive β is clearly better and at least one setting where fixed-negative β is clearly better.

Define:

```text
G_plus  = game/adversary setting where fixed_positive > fixed_negative and fixed_positive > vanilla
G_minus = game/adversary setting where fixed_negative > fixed_positive and fixed_negative > vanilla
```

Known likely candidate:

```text
G_minus:
  Shapley × finite-memory regret matching
  Shapley × hypothesis testing
```

Candidate `G_plus` settings:

```text
scripted adversarial RPS from Phase VII-A
delayed-reward strategic chain
coordination / rules-of-the-road recovery
trust-building coordination game
optimistic bait phase of bait-and-switch RPS
```

---

## 6. Pre-Screen Methods

Run:

```text
vanilla
fixed_positive
fixed_negative
adaptive_beta_clipped
```

Optional:

```text
adaptive_sign_only
adaptive_beta_fast
adaptive_beta_slow
```

---

## 7. Pre-Screen Budget

Use a fast development screen:

```text
episodes: 1000
seeds: 3
candidate settings: all available Phase VII-A/B plus new candidate G_plus settings
```

Promote candidates to sign-specialization validation if the direction is clear.

---

## 8. Sign-Specialization Validation

For promising candidates:

```text
episodes: 10000
seeds: 10
```

Promotion thresholds:

For `G_plus`:

```text
AUC(fixed_positive) - AUC(fixed_negative) > 0
AUC(fixed_positive) - AUC(vanilla) > 0
paired bootstrap CI should be positive or at least directionally stable
```

For `G_minus`:

```text
AUC(fixed_negative) - AUC(fixed_positive) > 0
AUC(fixed_negative) - AUC(vanilla) > 0
paired bootstrap CI should be positive or at least directionally stable
```

If no credible `G_plus` is found, stop and write:

```text
results/adaptive_beta/phase_VII_C/no_G_plus_found.md
```

Do not proceed to composite sign-switching games without a credible `G_plus`.

---

# PART II — Composite Sign-Switching Games

## 9. Goal

Create environments where the optimal β sign switches over time.

The composite game alternates between:

```text
G_plus regime  -> β > 0 useful
G_minus regime -> β < 0 useful
```

The regime should be hidden from the learner unless an explicit observability ablation is being run.

---

## 10. Composite Environment Variants

### 10.1 Exogenous Switching

Simplest version. Regime switches every `D` episodes.

```text
regime sequence:
  G_plus, G_minus, G_plus, G_minus, ...

dwell lengths:
  D ∈ {100, 250, 500, 1000}
```

Use this first for debugging and oracle validation.

---

### 10.2 Endogenous Switching

The adversary switches based on the agent's behavior.

Example trigger:

```text
if rolling_win_rate > 0.65 for 50 episodes:
    switch from G_plus to G_minus

if rolling_loss_rate > 0.55 for 50 episodes:
    switch from G_minus to G_plus or neutral reset
```

Alternative trigger:

```text
if rolling_action_concentration > 0.70:
    switch to punisher regime

if rolling_return < threshold:
    switch to recovery/exploitable regime
```

Use endogenous switching after exogenous switching is working.

---

## 11. Required Oracle β Diagnostic

Implement an oracle schedule:

```text
oracle_beta:
  β = +β_bar in G_plus
  β = -β_bar in G_minus
```

The oracle has access to the true regime. It is **not** a deployable method. It is a diagnostic upper bound.

Before claiming the composite environment tests adaptivity, require:

```text
AUC(oracle_beta) > AUC(fixed_positive)
AUC(oracle_beta) > AUC(fixed_negative)
AUC(oracle_beta) > AUC(vanilla)
```

If oracle β does not beat both fixed signs, the environment does not require adaptation. Do not use it as headline evidence.

---

## 12. Composite Methods

Required:

```text
vanilla
fixed_positive
fixed_negative
adaptive_beta_clipped
adaptive_sign_only
adaptive_beta_fast
adaptive_beta_slow
oracle_beta
```

Optional:

```text
adaptive_magnitude_only
adaptive_beta_no_clip
state_conditioned_beta
transition_conditioned_beta
```

---

## 13. Adaptive β Speed Update

Add a smoothed adaptive β update with learning speed `eta_beta`.

For agent/player `i`:

```text
beta_{i,e+1}
=
clip(
  (1 - eta_beta_i) * beta_{i,e}
  + eta_beta_i * beta_max * tanh(k * A_{i,e}),
  -beta_bar,
  beta_bar
)
```

where:

```text
A_{i,e} = episode-level advantage signal
eta_beta_i ∈ {0.03, 0.1, 0.3, 1.0}
```

Define named variants:

```text
adaptive_beta_slow:    eta_beta = 0.03
adaptive_beta_medium:  eta_beta = 0.10
adaptive_beta_fast:    eta_beta = 0.30
adaptive_beta_instant: eta_beta = 1.00
```

The original adaptive β schedule corresponds approximately to the instant version.

---

## 14. Composite Metrics

Primary:

```text
AUC
paired AUC difference vs vanilla
paired AUC difference vs fixed_positive
paired AUC difference vs fixed_negative
post-switch recovery time
catastrophic windows
final-window return
```

β-specific:

```text
beta_sign_accuracy
beta_lag_to_oracle
beta_switch_delay
mean_abs_beta_error
```

Mechanism:

```text
alignment_rate
mean_effective_discount
fraction_d_eff_below_gamma
advantage_distribution
Bellman residual
```

Switch/event metrics:

```text
return_before_switch
return_after_switch
recovery_after_switch
drawdown_after_switch
catastrophic_windows_after_switch
```

---

## 15. β Sign Accuracy

Define oracle sign:

```text
sign_star_e = sign(beta_oracle_e)
```

Define adaptive sign accuracy:

```text
beta_sign_accuracy = mean(sign(beta_adaptive_e) == sign_star_e)
```

Ignore episodes where oracle β = 0.

---

## 16. β Lag to Oracle

Define:

```text
beta_lag = mean(abs(beta_adaptive_e - beta_oracle_e))
```

Also compute event-local lag:

```text
beta_lag_pre_switch
beta_lag_0_50_after_switch
beta_lag_50_200_after_switch
beta_lag_200_plus_after_switch
```

---

## 17. β Switch Delay

For each regime switch, define the first episode after the switch where:

```text
sign(beta_adaptive_e) == sign(beta_oracle_e)
```

Then:

```text
beta_switch_delay = episode_index_first_correct_sign - switch_episode
```

If the adaptive method never switches correctly within the dwell window, record as censored and count it.

---

# PART III — Candidate Composite Games

## 18. Bait-and-Switch RPS

This is the highest-priority new environment.

### 18.1 Intuition

The opponent alternates between:

1. **Bait / exploitable regime**
   - opponent has a stable biased strategy;
   - agent can exploit it;
   - commitment and optimistic propagation help;
   - expected useful sign: β > 0.

2. **Punisher / trap regime**
   - once the agent over-commits, opponent switches to counter-strategy;
   - overestimated continuation values become dangerous;
   - pessimistic propagation helps;
   - expected useful sign: β < 0.

### 18.2 Actions

Use standard RPS:

```text
0 = Rock
1 = Paper
2 = Scissors
```

Rewards:

```text
+1 win
0 draw
-1 loss
```

Optional amplified trap reward:

```text
-2 or -3 when punished after over-commitment
```

### 18.3 Exogenous Version

Regimes alternate every `D` episodes:

```text
bait -> punisher -> bait -> punisher
```

Dwell lengths:

```text
D ∈ {100, 250, 500, 1000}
```

### 18.4 Endogenous Version

Switch to punisher when:

```text
rolling_win_rate(agent, window=50) > 0.65
```

or:

```text
agent_action_concentration(window=50) > 0.70
```

Switch back to bait/reset when:

```text
rolling_loss_rate(agent, window=50) > 0.55
```

or after a cooldown:

```text
punisher_duration >= D_punish
```

### 18.5 Required Outputs

For Bait-and-Switch RPS, generate:

```text
learning_curve_bait_switch_rps.pdf
switch_aligned_return_bait_switch_rps.pdf
switch_aligned_beta_bait_switch_rps.pdf
beta_sign_accuracy_bait_switch_rps.pdf
method_auc_table_bait_switch_rps.csv
```

---

## 19. Coordination-Then-Trap Game

This is the second-highest-priority environment.

### 19.1 Intuition

The agent alternates between trust-building and trap-avoidance regimes.

1. **Trust / coordination regime**
   - risky coordination gives delayed high payoff;
   - safe action gives small immediate payoff;
   - useful sign: β > 0.

2. **Betrayal / trap regime**
   - risky action gives immediate small reward but exposes future large loss;
   - safe action avoids catastrophe;
   - useful sign: β < 0.

### 19.2 Actions

```text
0 = Safe
1 = Risky
```

### 19.3 Rewards

Trust regime:

```text
Safe: +0.2 immediate
Risky: 0 immediate, but if coordinated/maintained for L rounds, +R_big
```

Trap regime:

```text
Safe: +0.1
Risky: +0.3 immediate but with delayed penalty -R_trap
```

Suggested values:

```text
L ∈ {3, 5, 10}
R_big ∈ {5, 10}
R_trap ∈ {5, 10}
```

### 19.4 Endogenous Switch

Switch from trust to trap when:

```text
rolling_risky_rate(window=50) > 0.65
```

Switch back to trust after:

```text
rolling_risky_rate(window=50) < 0.35
```

or after fixed cooldown.

### 19.5 Required Outputs

```text
learning_curve_coordination_trap.pdf
switch_aligned_return_coordination_trap.pdf
switch_aligned_beta_coordination_trap.pdf
catastrophic_windows_coordination_trap.pdf
method_auc_table_coordination_trap.csv
```

---

## 20. Sign-Switching Shapley/RPS Composite

Use existing Phase VII-B Shapley setting as `G_minus`.

Use either:
- Phase VII-A scripted RPS, or
- Bait RPS regime,
as `G_plus`.

Composite:

```text
G_plus block -> G_minus block -> G_plus block -> G_minus block
```

This can be implemented as a meta-environment wrapper around two existing environments.

Required diagnostic:
- oracle β must beat both fixed signs.

---

# PART IV — Two-Player β-Speed Self-Play Duels

## 21. Goal

Test whether faster β adaptation gives an advantage when both players use adaptive temporal-credit controllers.

Important theoretical expectation:

```text
equal beta-learning rates in symmetric games should produce no expected aggregate win advantage
```

But this does **not** necessarily imply:

```text
β -> 0
```

They may converge to:
- β ≈ 0,
- a shared nonzero β,
- oscillatory β dynamics,
- regime-dependent β cycles.

Therefore, track both aggregate score and β trajectories.

---

## 22. Self-Play Duel Environment

Use a sign-switching strategic game, preferably:

```text
Bait-and-Switch RPS
```

or:

```text
Coordination-Then-Trap
```

Avoid stationary Shapley as the main duel environment because Phase VII-B suggests it has a stationary pessimistic optimum.

---

## 23. Duel Pairings

Run:

```text
adaptive_fast vs adaptive_slow
adaptive_fast vs adaptive_medium
adaptive_medium vs adaptive_slow
adaptive_fast vs vanilla
adaptive_fast vs fixed_negative
adaptive_fast vs fixed_positive
adaptive_slow vs fixed_negative
adaptive_slow vs fixed_positive
adaptive_equal_speed vs adaptive_equal_speed
```

For equal-speed controls:

```text
adaptive_slow vs adaptive_slow
adaptive_medium vs adaptive_medium
adaptive_fast vs adaptive_fast
```

Always swap player order:

```text
A as player 1, B as player 2
B as player 1, A as player 2
```

---

## 24. Duel Metrics

Primary:

```text
aggregate_score_difference = sum(R_A) - sum(R_B)
win_rate_A
paired_score_difference_with_player_order_swap
AUC_A_minus_AUC_B
```

β metrics:

```text
beta_A_trajectory
beta_B_trajectory
beta_gap = beta_A - beta_B
beta_correlation
beta_sign_accuracy_A
beta_sign_accuracy_B
beta_switch_delay_A
beta_switch_delay_B
```

Symmetry metrics:

```text
equal_speed_score_difference_mean
equal_speed_score_difference_CI
player_order_bias
```

Mechanism:

```text
alignment_rate_A
alignment_rate_B
effective_discount_A
effective_discount_B
post_switch_recovery_A
post_switch_recovery_B
```

---

## 25. Duel Hypotheses

### H1: Equal-Speed Symmetry

For equal-speed adaptive-vs-adaptive pairings:

```text
aggregate_score_difference ≈ 0
```

This is a sanity check.

Do not require β to converge to 0. Report observed β behavior.

### H2: Fast Beats Slow in Sign-Switching Games

For fast-vs-slow adaptive pairings:

```text
aggregate_score_difference(fast - slow) > 0
```

Expected reason:
- fast β adaptation tracks regime changes more quickly;
- slow β adaptation lags and suffers around switches.

### H3: Fast Adaptive Beats Fixed Signs

In sign-switching games:

```text
fast_adaptive > fixed_positive
fast_adaptive > fixed_negative
```

Expected reason:
- fixed-positive loses in trap/punisher regimes;
- fixed-negative under-exploits bait/trust regimes.

### H4: Fixed Sign Still Wins in Stationary-Sign Games

In stationary pessimistic games such as the Phase VII-B Shapley setting:

```text
fixed_negative may beat adaptive
```

This should be acknowledged as a limitation and used as a control.

---

# PART V — Experiment Stages

## 26. Stage C0 — Sign-Specialization Screen

Budget:

```text
episodes: 1000
seeds: 3
methods:
  vanilla
  fixed_positive
  fixed_negative
  adaptive_beta_clipped
candidate settings: available Phase VII-A/B plus new G_plus candidates
```

Deliverable:

```text
results/adaptive_beta/phase_VII_C/sign_specialization_screen.md
```

Decision:
- If credible `G_plus` and `G_minus` exist, proceed.
- If no `G_plus`, stop.

---

## 27. Stage C1 — Composite Oracle Validation

Budget:

```text
episodes: 3000
seeds: 5
methods:
  vanilla
  fixed_positive
  fixed_negative
  adaptive_beta_clipped
  adaptive_beta_fast
  adaptive_beta_slow
  oracle_beta
```

Required condition:

```text
oracle_beta beats vanilla and both fixed signs
```

Deliverable:

```text
results/adaptive_beta/phase_VII_C/composite_oracle_validation.md
```

Decision:
- If oracle does not beat fixed signs, redesign the composite game.
- If oracle wins, proceed to C2.

---

## 28. Stage C2 — Main Adaptive-β Test

Budget:

```text
episodes: 10000
seeds: 10
best 1–2 composite games
methods:
  vanilla
  fixed_positive
  fixed_negative
  adaptive_beta_slow
  adaptive_beta_medium
  adaptive_beta_fast
  adaptive_beta_instant
  adaptive_sign_only
  oracle_beta
```

Deliverable:

```text
results/adaptive_beta/phase_VII_C/main_adaptive_test.md
```

Success condition:

```text
adaptive_beta_fast or adaptive_beta_medium beats:
  vanilla
  fixed_positive
  fixed_negative
```

on AUC and/or recovery, with mechanism evidence:
- β sign accuracy above chance,
- β switch delay lower than slow adaptive,
- event-aligned β tracks oracle sign.

---

## 29. Stage C3 — β-Speed Self-Play Duel

Budget:

```text
episodes: 10000
seeds: 10
duel pairings: core list from Section 23
player-order swaps: required
```

Deliverable:

```text
results/adaptive_beta/phase_VII_C/beta_speed_duel.md
```

Success condition:
- equal-speed adaptive pairings have approximately zero order-corrected aggregate score difference;
- fast adaptive beats slow adaptive in sign-switching game;
- fast adaptive beats fixed signs after player-order correction.

---

## 30. Stage C4 — Stress Tests

Run only if C2 or C3 is strong.

Stress knobs:

```text
dwell_length D
switch trigger threshold
eta_beta
beta_bar
beta_max
k
reward scale
trap penalty
payoff noise
opponent inertia
```

Deliverable:

```text
results/adaptive_beta/phase_VII_C/stress_tests.md
```

---

# PART VI — Implementation Details

## 31. Suggested Code Location

Use:

```text
experiments/adaptive_beta/sign_switching/
  __init__.py
  composite_env.py
  oracle_beta.py
  beta_speed_schedules.py
  bait_switch_rps.py
  coordination_trap.py
  duel_env.py
  duel_runner.py
  metrics.py
  configs/
    C0_sign_screen.yaml
    C1_oracle_validation.yaml
    C2_main_adaptive.yaml
    C3_beta_duel.yaml
  analysis/
    aggregate.py
    plot_switch_aligned.py
    plot_beta_oracle.py
    plot_duel_results.py
```

Reuse:

```text
experiments/adaptive_beta/strategic_games/
src/lse_rl/operator/tab_operator.py
```

Do not duplicate existing strategic-game utilities.

---

## 32. Required Config Fields

Every config must include:

```yaml
experiment_name:
stage:
seed_grid:
episodes:
horizon:
gamma:
alpha:
methods:
beta_bar:
beta_max:
k:
eta_beta_grid:
switching:
  type: exogenous_or_endogenous
  dwell_length:
  trigger_window:
  win_threshold:
  loss_threshold:
oracle:
  enabled:
  beta_plus:
  beta_minus:
logging:
  transition_logs:
  episode_logs:
  output_dir:
```

---

## 33. Required Logging Fields

Episode-level:

```text
run_id
stage
seed
env
method
episode
regime
oracle_beta
beta
eta_beta
return
auc_so_far
catastrophic
switch_event
episodes_since_switch
beta_sign_correct
beta_lag_to_oracle
beta_switch_delay
alignment_rate
mean_effective_discount
bellman_residual
diverged
nan_count
```

Duel-level:

```text
run_id
seed
env
episode
player_A_method
player_B_method
player_A_beta
player_B_beta
player_A_return
player_B_return
score_difference
player_order
regime
switch_event
player_A_beta_sign_correct
player_B_beta_sign_correct
```

---

## 34. Required Figures

Generate:

```text
learning_curves_composite.pdf
switch_aligned_returns.pdf
switch_aligned_beta_vs_oracle.pdf
beta_sign_accuracy.pdf
beta_switch_delay.pdf
auc_paired_differences.pdf
duel_score_differences.pdf
duel_beta_trajectories.pdf
duel_symmetry_controls.pdf
```

Save both PDF and PNG where practical.

---

## 35. Required Tables

Generate:

```text
sign_specialization_table.csv
oracle_validation_table.csv
main_adaptive_results_table.csv
beta_speed_ablation_table.csv
duel_score_table.csv
duel_symmetry_table.csv
failure_modes_table.csv
```

Each table should include:
- mean,
- std,
- 95% CI,
- paired difference vs vanilla,
- paired difference vs fixed-positive,
- paired difference vs fixed-negative,
- run count,
- failure count.

---

## 36. Tests

Add tests under:

```text
tests/adaptive_beta/sign_switching/
```

Required:

1. Exogenous switch occurs at correct dwell length.
2. Endogenous switch triggers on rolling win/loss thresholds.
3. Oracle β matches true regime sign.
4. Adaptive speed update matches formula.
5. β clipping works.
6. Composite env resets deterministically under seed.
7. Duel player-order swap is implemented.
8. Equal-speed duel has no hard-coded asymmetry in environment.
9. All run manifests account for every config triple.
10. No duplicate TAB operator implementation.

---

## 37. Review Gates

Run verifier after:
- C0 implementation,
- C1 oracle validation,
- C2 main test,
- C3 duel test.

Use review/adversarial review if:
- claiming adaptive β beats fixed signs,
- proposing paper update.

Review questions:
1. Does oracle β actually beat both fixed signs?
2. Does adaptive β beat both fixed signs, or only vanilla?
3. Are β trajectories switching for the right reason?
4. Is the regime hidden from non-oracle agents?
5. Are player-order effects controlled in duels?
6. Are weak or negative results reported honestly?

---

## 38. Paper-Update Policy

Only propose paper update if C2 or C3 succeeds.

### Main-text candidate if:
- adaptive β beats vanilla and both fixed signs in at least one sign-switching setting;
- oracle β validates that adaptation is useful;
- β sign accuracy and switch-delay metrics support the mechanism;
- fixed signs each fail in the opposite regime;
- duel results show fast adaptive beats slow adaptive or fixed signs after order correction.

### Appendix-only if:
- oracle β works but learned adaptive β only partially tracks it;
- adaptive beats vanilla but not the best fixed sign;
- duel results are suggestive but underpowered.

### No update if:
- oracle β does not beat fixed signs;
- fixed-negative still dominates all composites;
- adaptive β does not switch signs;
- duel results are symmetric with no speed advantage.

Produce one of:

```text
results/adaptive_beta/phase_VII_C/paper_update/main_patch.md
results/adaptive_beta/phase_VII_C/paper_update/appendix_patch.md
results/adaptive_beta/phase_VII_C/paper_update/no_update.md
```

Do not edit the paper directly.

---

## 39. Failure Handling

If no `G_plus` is found:
- stop before C1,
- write a negative-result memo.

If oracle β does not beat both fixed signs:
- redesign composite game once,
- rerun C1,
- if still failing, stop.

If adaptive β fails but oracle succeeds:
- write this as a schedule-learning failure;
- recommend improved β estimator/state-conditioned β.

If fast adaptive does not beat slow adaptive:
- analyze β lag, noise, and overshooting;
- do not claim speed advantage.

If equal-speed self-play has a large order-corrected score difference:
- inspect environment asymmetry;
- do not interpret duel results until fixed.

---

## 40. Final Deliverables

Minimum:

```text
results/adaptive_beta/phase_VII_C/sign_specialization_screen.md
results/adaptive_beta/phase_VII_C/composite_oracle_validation.md
results/adaptive_beta/phase_VII_C/main_adaptive_test.md
results/adaptive_beta/phase_VII_C/beta_speed_duel.md
results/adaptive_beta/phase_VII_C/final_recommendation.md
results/adaptive_beta/phase_VII_C/figures/
results/adaptive_beta/phase_VII_C/tables/
results/adaptive_beta/phase_VII_C/processed/
```

If stopped early, produce:

```text
results/adaptive_beta/phase_VII_C/early_stop_memo.md
```

Final recommendation must answer:

1. Did fixed-positive and fixed-negative specialize to different regimes?
2. Did oracle β beat both fixed signs?
3. Did learned adaptive β beat both fixed signs?
4. Did β sign accuracy support the mechanism?
5. Did faster β adaptation beat slower β adaptation?
6. Did equal-speed self-play behave symmetrically?
7. Should the paper be updated, appendix-only, or unchanged?
8. What failure mode remains?

---

## 41. Short Instruction to Coding Agent

Implement Phase VII-C sign-switching adaptive-β games. First identify G_plus and G_minus regimes where fixed-positive and fixed-negative specialize. Then build composite hidden-regime environments where oracle β beats both fixed signs. Compare adaptive β speeds against vanilla and both fixed β baselines. Finally run β-speed self-play duels with player-order swaps. The goal is to determine whether adaptive β can outperform fixed signs when the optimal temporal-credit sign changes over time.
