# Harness Instructions — Six-Game Safe TAB Experiment Suite

**Document version:** v2 (2026-04-30) — reconciled with repository ground truth.
**Status:** Implementation and review specification for Phase VIII.

---

## 0. Purpose

This document is a harness-facing implementation and review specification for
the six-game Safe TAB experiment suite.

The experiment suite supports the NeurIPS paper on the **safe Temporal
Allocation Bellman (TAB) operator** as a general framework for Dynamic
Programming and Reinforcement Learning. The goal is not to prove that adaptive
β always dominates fixed β. The goal is to show, in a disciplined and
reviewer-legible way, that:

1. TAB defines a meaningful family of Bellman / DP / RL operators indexed by β.
2. β changes learning geometry, temporal credit assignment, effective
   contraction, safety behaviour, and sample efficiency.
3. Fixed nonzero β can outperform vanilla β = 0 in controlled strategic RL
   tasks.
4. Safe clipping and certified operator use prevent unstable β behaviour.
5. Adaptive β is useful only when the best β changes across regimes; in
   stationary-sign tasks, the best fixed β may be the correct method.

This specification must be executed under the repository harness protocol in
`AGENTS.md` and the per-project rules in `CLAUDE.md`.

> **v2 reconciliation summary.** v1 of this document was written before
> repository recon. v2 replaces invented paths and assumed-missing components
> with the actual repository state. Phase VIII is now framed as an *extension*
> of the existing Phase VII adaptive-β strategic-games stack, not a green-field
> build. See §1.4 (existing inventory) and §10 (revised stage plan).

---

## 1. Repository Ground Truth (verified 2026-04-30)

### 1.1 AGENTS.md is authoritative, not a stub

`CLAUDE.md` §7 currently claims `AGENTS.md` is a stub. **This is out of date.**
`AGENTS.md` is a 227-line authoritative orchestration protocol with the
following named subagent roster:

```text
planner
env-builder
algo-implementer
operator-theorist
calibration-engineer
experiment-runner
test-author
plotter-analyst
verifier
review-triage
```

It defines main-orchestrator invariants, a dispatch decision table mapping task
tags to subagents (§4 of `AGENTS.md`), Codex review/adversarial-review gates,
worktree discipline, handoff contracts, and overnight-mode rules.

**Action item:** open a separate task to update `CLAUDE.md` §7 to match
`AGENTS.md`. Until then, `AGENTS.md` is authoritative for harness behaviour.

### 1.2 Phase VII is canonical and partly executed

```text
docs/specs/phase_VII_adaptive_beta.md          # 954 lines, canonical
results/adaptive_beta/strategic/
  final_recommendation.md                      # cross-reference, do not overwrite
  stage_B2_main_summary.md
  stage_B2_dev_summary.md
```

Phase VII Stage A and Stage B2 main passes have produced strategic-games
results across matching pennies, Shapley, rules of road, asymmetric
coordination, and strategic RPS. Phase VIII must read and incorporate these
artifacts as prior work. Do not re-derive Stage B2.

Phase VII is classified by the paper repo as **exploratory, not main-paper
content** unless the user explicitly promotes it. Phase VIII inherits this
default; promotion follows §15 of this document.

### 1.3 CLAUDE.md repository rules in effect

- Research code lives in `src/lse_rl/`. Framework code lives in
  `mushroom-rl-dev/`. Prefer adding new modules in `src/lse_rl/` (or
  experiment-scoped roots) over editing stable MushroomRL code.
- Phase specs in `docs/specs/phase_*.md` are load-bearing.
- Reproducibility defaults: every run takes `--seed` and `--config` (yaml or
  json); raw artifacts under `results/<scope>/raw/<phase>/<suite>/<task>/<algorithm>/seed_<seed>/`
  with `run.json`, `metrics.npz` (schema header), and where applicable
  `transitions.parquet`.
- Logging contract: reuse `mushroom_rl.core.logger.Logger` where it helps; in
  addition, every run emits `run.json` + `metrics.npz` per Phase I spec.

### 1.4 Existing implementation inventory

The following is verified by file reads on 2026-04-30. Tags:
**[DONE]** = exists and is tested; **[PARTIAL]** = exists with gaps;
**[TODO]** = absent and must be built.

#### Operator kernel — [DONE]

```text
src/lse_rl/operator/tab_operator.py
  g(beta, gamma, r, v) -> float
  rho(beta, gamma, r, v) -> float
  effective_discount(beta, gamma, r, v) -> float
  g_batch / rho_batch / effective_discount_batch
  _EPS_BETA = 1e-8                                  # classical-collapse threshold
  _is_classical(beta) -> bool                       # explicit β=0 branch

mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py
  class SafeWeightedCommon(...)                     # certified operator wrapper
  compute_kappa(alpha_t, gamma)
  compute_certified_radii(T, kappa_t, R_max, gamma)
  build_certification(alpha_t, R_max, gamma) -> {kappa_t, Bhat_t, beta_cap_t}
  compute_beta_cap(kappa_t, Bhat_t, R_max, gamma)
  per-stage clip: np.clip(beta_raw, -beta_cap_t, beta_cap_t)

tests/algorithms/test_safe_weighted_lse_operator.py
tests/algorithms/test_safe_beta0_equivalence.py
tests/algorithms/test_safe_clipping_certification.py
```

#### Repeated matrix-game environment — [DONE]

```text
experiments/adaptive_beta/strategic_games/
  matrix_game.py        class MatrixGameEnv          # MushroomRL-compatible
  history.py            dataclass GameHistory        # last(m), empirical_*_policy, rolling_return
  registry.py           GAME_REGISTRY, ADVERSARY_REGISTRY
```

#### Game implementations — [DONE for 5 of 6]

```text
strategic_games/games/
  matching_pennies.py            [DONE]
  shapley.py                     [DONE]
  rules_of_road.py               [DONE]
  asymmetric_coordination.py     [DONE]
  strategic_rps.py               [DONE] (RPS variants for Phase VII)
  soda_uncertain.py              [TODO] — only missing game
```

#### Opponent / adversary policies — [DONE for 9 classes]

```text
strategic_games/adversaries/
  base.py                        StrategicAdversary (ABC), info-key contract
  stationary.py                  StationaryMixedOpponent
  scripted_phase.py              ScriptedPhaseOpponent
  finite_memory_best_response.py FiniteMemoryBestResponse
  finite_memory_fictitious_play.py FiniteMemoryFictitiousPlay
  smoothed_fictitious_play.py    SmoothedFictitiousPlay
  regret_matching.py             RegretMatching
  finite_memory_regret_matching.py FiniteMemoryRegretMatching
  hypothesis_testing.py          HypothesisTestingAdversary
  realized_payoff_regret.py      RealizedPayoffRegret (optional stub)
  inertia.py                     [TODO]
  convention_switching.py        [TODO]
  sign_switching_regime.py       [TODO] (controls hidden ξ for composites)
```

#### TAB Q-learning agent — [DONE]

```text
experiments/adaptive_beta/agents.py
  class AdaptiveBetaQAgent
    - tabular Q of shape (n_states, n_actions)
    - single TD-update path _step_update (Phase VII spec §16.2)
    - β fetched once per episode via beta_schedule.beta_for_episode(e)
    - β=0 bit-identity guard (raises if td_target != classical when |β|≤_EPS_BETA)
    - imports from src.lse_rl.operator.tab_operator
```

#### β schedules — [PARTIAL]

```text
experiments/adaptive_beta/schedules.py
  Protocol BetaSchedule: beta_for_episode, update_after_episode, diagnostics
  build_schedule(method_id, env_canonical_sign, hyperparams) -> BetaSchedule
  ZeroBetaSchedule                        [DONE]  # METHOD_VANILLA
  FixedBetaSchedule                       [DONE]  # ±β₀
  WrongSignSchedule                       [DONE]
  AdaptiveBetaSchedule                    [DONE]  # β = clip(β_max·tanh(k·Ā_e), ±β_cap)
  AdaptiveSignOnlySchedule                [DONE]
  AdaptiveMagnitudeOnlySchedule           [DONE]
  OracleBetaSchedule                      [TODO]  # uses hidden regime ξ_t
  HandAdaptiveBetaSchedule                [TODO]  # pre-registered episode rule
  ContractionUCBBetaSchedule              [TODO]
  ReturnUCBBetaSchedule                   [TODO]
  HedgeBetaSchedule                       [TODO]  (appendix)
  DiscountedHedgeBetaSchedule             [TODO]  (appendix)
  GradientBetaSchedule                    [TODO]  (appendix)
  BilevelBetaSchedule (SOBBO-inspired)    [TODO]  (appendix)
```

#### External baseline agents — [TODO]

```text
experiments/adaptive_beta/baselines.py    [TODO]
  RestartQLearningAgent
  SlidingWindowQLearningAgent
  TunedEpsilonGreedyQLearningAgent
```

#### Manifest, schema, IO infrastructure — [DONE]

```text
experiments/weighted_lse_dp/common/io.py
  RESULT_ROOT = Path("results/weighted_lse_dp")
  make_run_dir(base, phase, suite, task, algorithm, seed) -> Path
  make_npz_schema(...)
  save_npz_with_schema(...)
  SCHEMA_VERSION = "1.0.0"
  stdout-tee context manager

experiments/weighted_lse_dp/common/manifests.py
  git_sha(), resolve_config(), write_run_json(), write_metrics_json(),
  write_safe_provenance(), load_run_json(), load_metrics_json(), find_run_dirs()

experiments/weighted_lse_dp/common/schemas.py
  class RunWriter
  CURVES_ARRAYS_RL, CURVES_ARRAYS_DP, TRANSITIONS_ARRAYS, CALIBRATION_ARRAYS,
  PHASE2_EVENT_ARRAYS, SAFE_TRANSITIONS_ARRAYS, SAFE_CALIBRATION_ARRAYS
  validate_*_npz(...)

experiments/weighted_lse_dp/common/schedules.py     # DP-side β schedule (distinct from RL)
  dataclass BetaSchedule with betas: shape (T+1,)  float64
  zero_schedule(), load_schedule_or_zero(), to_dict / from_dict
  SCHEDULE_VERSION = "1.0.0"
  source tags: "zero" | "calibrated" | "manual"
```

#### Test suite — [DONE]

```text
tests/adaptive_beta/strategic_games/
  test_matrix_game.py
  test_adversaries.py
  test_games.py
  test_history.py
  test_registry.py
  test_metrics.py
  test_logging.py
  test_reproducibility.py
  test_runner_smoke.py
```

### 1.5 Lessons backlog

`tasks/lessons.md` carries 32+ recorded prevention rules. Read before any
new code that touches operator math, calibration, manifest writing, or
aggregation. Particularly relevant to Phase VIII:

- #1 torch import — must run via `.venv/bin/python` (MushroomRL hard-imports torch).
- #6 `gamma` lives on `mdp.info`, not the agent or schedule.
- #16–#17 figure scripts must read the actual schema; config-dict overrides only fire if consumed.
- #19–#20 aggregation drift from summaries-of-summaries; hard-coded state counts.
- #22 MushroomRL `callback_step` fires before `agent.fit`.
- #23 certification `R_max` must be a configured absolute bound.
- #27 expm1/log1p underflow in negative tails (relevant to contraction reward log scale).
- #28 numpy ≥ 2.0 `int(state)` `TypeError` on shape-(1,) arrays.

---

## 2. Phase identity

### 2.1 Recommended

```text
Phase number:        VIII
Spec file:           docs/specs/phase_VIII_tab_six_games.md
Code roots:
  EXTEND             experiments/adaptive_beta/strategic_games/
  ADD                experiments/adaptive_beta/tab_six_games/
Results root:        results/adaptive_beta/tab_six_games/
Branch:              phase-VIII-tab-six-games-2026-04-30
```

### 2.2 Rationale for extending Phase VII rather than greenfield

`experiments/adaptive_beta/strategic_games/` already implements the env wrapper,
history, registry, 5 of 6 games, all 9 base adversaries, the TAB Q-learning
agent, six β schedules, and a complete test suite (§1.4). A greenfield
`experiments/tab_six_games/` would force duplication and re-validation of
~3000 lines of stable, tested code, and would break Phase VII test discoverability.

Phase VIII therefore extends Phase VII with the strict deltas:

```text
+ Soda / uncertain game (the missing 6th)
+ Inertia, convention-switching, sign-switching-regime adversaries
+ Oracle, hand-adaptive, contraction-UCB, return-UCB β schedules
+ Restart, sliding-window, tuned-ε external baseline agents
+ Sign-switching composite environments
+ Stage 1–5 Phase VIII runners + analysis
+ Phase VIII delta metrics (contraction reward, recovery time, etc.)
```

### 2.3 Hard prohibitions

- Do not create `experiments/tab_six_games/` at the repository root.
- Do not duplicate operator math; all β-aware updates flow through
  `src/lse_rl/operator/tab_operator.py`.
- Do not branch `AdaptiveBetaQAgent`. Add new methods by registering a new
  `BetaSchedule` subclass; the Phase VII single-update-path invariant is
  load-bearing (Phase VII spec §16.2).
- Do not edit `mushroom-rl-dev/...safe_weighted_common.py` unless strictly
  necessary; if you do, log justification in `tasks/lessons.md` and trigger
  ad-hoc Codex review.

---

## 3. High-level Experimental Narrative

The paper narrative is operator-first. Phase VIII preserves the following
interpretation:

```text
Classical RL:        β = 0
Fixed TAB:           β fixed positive or fixed negative
Safe TAB:            β clipped/certified for stability
Adaptive TAB:        β selected online (advantage, contraction, meta-learning)
Oracle TAB:          β uses hidden regime; diagnostic only
```

Main claim:

> The safe TAB operator defines a controlled family of Bellman updates.
> Different β values induce different temporal-credit and contraction
> behaviour. Fixed TAB can improve sample efficiency when one temporal-credit
> geometry is consistently useful; adaptive TAB is useful when the best β
> changes across regimes.

Do **not** claim:

```text
adaptive β always beats fixed β
TAB solves all strategic games
fixed-negative dominance is a failure
```

Correct interpretation of fixed β:

```text
fixed-positive and fixed-negative are members of our TAB model family.
If fixed β beats β = 0, that supports the TAB operator thesis.
If adaptive β beats the best fixed β in sign-switching settings, that
  supports the adaptive β thesis.
```

---

## 4. Common Mathematical and Implementation Framework

### 4.1 Setup

Each task is a repeated finite matrix game embedded in tabular RL.

At time \(t\):

\[
a_t \in \mathcal A_1,\qquad b_t \in \mathcal A_2,
\]

with \(a_t\) the TAB agent action and \(b_t\) the opponent/adversary action.

Reward: \(r_t = u_1(a_t,b_t;\xi_t)\), where \(\xi_t\) is an optional hidden
regime / hidden game type.

History: \(h_t=(a_0,b_0,r_0,\dots,a_{t-1},b_{t-1},r_{t-1})\).

State: \(s_t=\phi(h_t)\). The default state encoder
(`make_default_state_encoder` in `matrix_game.py`) indexes timestep + last
opponent action; a single-dummy-state encoder is also available.

The agent maintains a tabular \(Q(s,a)\) of shape `(n_states, n_actions)`
(`AdaptiveBetaQAgent._Q`).

### 4.2 TAB update

\[
Q(s_t,a_t) \leftarrow (1-\alpha) Q(s_t,a_t) + \alpha\, \mathcal G_{\beta,\gamma}(r_t, V(s_{t+1})),
\qquad V(s)=\max_a Q(s,a).
\]

In code:

```python
from src.lse_rl.operator.tab_operator import g, effective_discount, _EPS_BETA

td_target = g(beta=beta_e, gamma=gamma, r=r_t, v=v_next)        # β=0 collapse to r + γ·v
d_eff     = effective_discount(beta=beta_e, gamma=gamma, r=r_t, v=v_next)
```

The β=0 bit-identity guard in `AdaptiveBetaQAgent._step_update` must be
preserved by every new method: when `|β| ≤ _EPS_BETA`, `td_target` must equal
`r + γ·v` exactly. New schedules are responsible for honouring this; the agent
asserts it.

### 4.3 DP-side certification (cross-reference)

Phase VIII does not run new DP planners by default. If Stage 1 or Stage 5 needs
a certified DP reference, use:

```python
from mushroom_rl.algorithms.value.dp.safe_weighted_common import SafeWeightedCommon
# build_certification(alpha_t, R_max, gamma) -> {kappa_t, Bhat_t, beta_cap_t}
# Per-stage clip: np.clip(beta_raw, -beta_cap_t, beta_cap_t)
```

`R_max` must be a configured absolute bound (see `tasks/lessons.md` #23).

---

## 5. Required Games — current status

The six games host the strategic suite. Each game registers via
`register_game()` in
`experiments/adaptive_beta/strategic_games/registry.py`.

### 5.1 Matching Pennies — [DONE]

```text
games/matching_pennies.py
Actions: H, T
Reward: +1 match, -1 mismatch (zero-sum)
```

Subcases for Phase VIII:

```text
MP-Stationary           [DONE via existing adversary registry]
MP-FiniteMemoryBR       [DONE]
MP-RegretMatching       [DONE]
MP-HypothesisTesting    [DONE]
```

Role: adversarial zero-sum benchmark; sanity check; nonstationary opponent
benchmark.

### 5.2 Shapley Cyclic Game — [DONE]

```text
games/shapley.py
Actions: 0, 1, 2
Payoff: documented Shapley-style cyclic non-zero-sum (Brown–Robinson cycles
under greedy BR)
```

Subcases:

```text
SH-FictitiousPlay       [DONE]
SH-SmoothedFP           [DONE]
SH-FiniteMemoryRegret   [DONE]
SH-HypothesisTesting    [DONE]
```

Role: cyclic strategic dynamics; expected fixed-negative TAB specialist.

### 5.3 Rules of the Road — [DONE]

```text
games/rules_of_road.py
Actions: L, R
Reward: +c coordinated, -m miscoordinated; supports tremble_prob, payoff_bias
```

Subcases:

```text
RR-StationaryConvention [DONE]
RR-ConventionSwitch     [TODO opponent: convention_switching.py]
RR-Tremble              [DONE via tremble_prob]
RR-HypothesisTesting    [DONE]
```

Role: coordination recovery; convention-switch recovery.

### 5.4 Asymmetric Coordination — [DONE]

```text
games/asymmetric_coordination.py
Actions: A, B (stag-hunt-style)
Canonical sign: "+"
```

Subcases:

```text
AC-FictitiousPlay       [DONE]
AC-SmoothedBR           [DONE]
AC-Inertia              [TODO opponent: inertia.py]
AC-Trap                 [DONE]
```

Role: miscoordination traps; pathwise dynamics.

### 5.5 Soda / Uncertain Game — [TODO]

```text
games/soda_uncertain.py    [TO IMPLEMENT in M2]
Actions: 1..m
Hidden type ξ ∈ {coordination, anti_coordination, zero_sum, biased_preference,
  switching}
```

Subcases:

```text
SO-Coordination
SO-AntiCoordination
SO-ZeroSum
SO-BiasedPreference
SO-TypeSwitch
```

Role: hidden payoff-type uncertainty; appendix-style limits experiment. Hidden
type must be exposed via `info["regime"]` (oracle β reads this; non-oracle
methods must not).

### 5.6 Potential / Weakly Acyclic Game — [TODO unless Phase VII has it]

```text
games/potential.py         [VERIFY existence; if absent, implement in M2]
```

Subcases:

```text
PG-CoordinationPotential
PG-Congestion
PG-BetterReplyInertia
PG-SwitchingPayoff
```

Role: positive control; show TAB accelerates convergence where convergence is
already expected. **Verifier action:** confirm whether a potential-game
implementation exists; if not, add to M2 alongside Soda.

---

## 6. Required Baselines and Methods — current status

### 6.1 Core operator methods

```text
vanilla_beta_0          [DONE] ZeroBetaSchedule
fixed_beta_-2           [DONE] FixedBetaSchedule(beta0=-2.0)
fixed_beta_-1           [DONE] FixedBetaSchedule(beta0=-1.0)
fixed_beta_-0.5         [DONE] FixedBetaSchedule(beta0=-0.5)
fixed_beta_+0.5         [DONE] FixedBetaSchedule(beta0=+0.5)
fixed_beta_+1           [DONE] FixedBetaSchedule(beta0=+1.0)
fixed_beta_+2           [DONE] FixedBetaSchedule(beta0=+2.0)
best_fixed_beta_grid    reporting aggregate, not a separately trained method
fixed_positive_TAB      [DONE]
fixed_negative_TAB      [DONE]
safe_clipped_TAB        [DONE] (clip enforced inside SafeWeightedCommon for DP;
                                 RL agent honours per-step clip via schedule)
oracle_beta             [TODO] OracleBetaSchedule (uses info["regime"])
```

`safe_clipped_TAB` exercises the same safe clipping path as the paper (Phase VII spec §3.4).

### 6.2 Adaptive β methods

```text
hand_adaptive_beta            [TODO] HandAdaptiveBetaSchedule
                                     (pre-registered rule, no per-task tuning)
smoothed_adaptive_beta        [DONE] AdaptiveBetaSchedule (tanh-clip)
contraction_UCB_beta          [TODO] ContractionUCBBetaSchedule
return_UCB_beta               [TODO] ReturnUCBBetaSchedule
```

Optional / appendix:

```text
hedge_beta                    [TODO] HedgeBetaSchedule
discounted_hedge_beta         [TODO] DiscountedHedgeBetaSchedule
continuous_gradient_beta      [TODO] GradientBetaSchedule
bilevel_SOBBO_beta            [TODO] BilevelBetaSchedule
```

Optional methods are implemented only after the core suite is stable and
reviewed (M11).

### 6.3 External / simple baselines

```text
restart_Q_learning              [TODO] RestartQLearningAgent
sliding_window_Q_learning       [TODO] SlidingWindowQLearningAgent
tuned_epsilon_greedy_Q_learning [TODO] TunedEpsilonGreedyQLearningAgent
```

Optional appendix baselines (rely on existing adversaries-as-policies):

```text
UCB_style_Q_learning            [TODO]
regret_matching_agent           reuse RegretMatching wrapped as agent
fictitious_play_agent           reuse FiniteMemoryFictitiousPlay wrapped as agent
smoothed_fictitious_play_agent  reuse SmoothedFictitiousPlay wrapped as agent
```

These baselines exist so the paper is not benchmarked only against β = 0.

---

## 7. Metrics

### 7.1 Reuse Phase VII metric vocabulary verbatim

Per-episode (already produced by `RunWriter` and Phase VII metric pipeline):

```text
return, length, epsilon
alignment_rate, mean_signed_alignment, mean_advantage, mean_abs_advantage
mean_d_eff, median_d_eff, frac_d_eff_below_gamma, frac_d_eff_above_one
bellman_residual, td_target_abs_max, q_abs_max
nan_count, divergence_event
catastrophic, success, regret, shift_event
```

### 7.2 Phase VIII delta metrics — to implement in M5

```text
contraction_reward                M_e(β) = log(R_e + ε) - log(R_{e+1} + ε)
                                  with R = bellman_residual (NOT episode return)
empirical_contraction_ratio       (R_{e+1} + ε) / (R_e + ε)
log_residual_reduction            log(R_e + ε) - log(R_{e+1} + ε)
ucb_arm_count                     per-β arm pull count (vector over arms)
ucb_arm_value                     per-β empirical UCB reward (vector)
beta_clip_count                   episodes with β clipped
beta_clip_frequency               beta_clip_count / episode
recovery_time_after_shift         episodes from switch to rolling-return ≥ θ
beta_sign_correct                 1[ sign(β_e) == sign(oracle_β_e) ]
beta_lag_to_oracle                e - argmin_{e' ≤ e} |β_{e'} - oracle_β_e|
regret_vs_oracle                  cumulative oracle return minus method return
catastrophic_episodes             count of episodes with return ≤ θ_low
worst_window_return_percentile    rolling-window worst-percentile return
trap_entries, constraint_violations
overflow_count
```

### 7.3 Strategic metrics (where applicable)

```text
external_regret
policy_regret_proxy
coordination_rate
miscoordination_streak_length
cycling_amplitude
policy_total_variation
opponent_policy_entropy
support_shift_count
time_to_equilibrium
distance_to_reference_mixed_strategy
```

Many of these are already emitted by Phase VII strategic-games metric code;
verifier (M5) confirms field-name parity before extending.

---

## 8. Required Logging Schema

### 8.1 Reuse Phase VII run.json + metrics.npz schema

Per-run files written via `experiments/weighted_lse_dp/common/manifests.py`
helpers and `experiments/weighted_lse_dp/common/schemas.py::RunWriter`:

```text
run.json        git_sha, argv, seed_list, task_config,
                resolved_hyperparameters, timestamp, output_paths,
                phase, suite (=stage_id), task (=game/subcase),
                algorithm (=method_id)
                Phase VIII additions:
                  phase = "VIII"
                  stage = "stage1" | "stage2" | "stage3" | "stage4" | "stage5"
                  subcase
                  regime_info_present : bool

metrics.npz     all §7.1 fields (Phase VII contract)
                + §7.2 delta fields (Phase VIII new)
                schema header attached via save_npz_with_schema with
                SCHEMA_VERSION = "1.0.0" and a Phase VIII subschema id.

transitions.parquet (optional, Stage 5 / sign-switching)
                run_id, episode, t, state, agent_action, opponent_action,
                reward, next_state, done, beta, advantage,
                effective_discount, alignment_indicator, regime,
                opponent_info_json
```

### 8.2 Run roster manifest — new

```text
experiments/adaptive_beta/tab_six_games/manifests.py    [TODO]
  class Phase8RunRoster
    rows: run_id, config_hash, seed, game, subcase, method, status,
          start_time, end_time, result_path, failure_reason, git_commit
    statuses: pending | running | completed | failed | diverged
              | skipped | stopped-by-gate
    write_atomic(...), reconcile_with_disk(...), summarize(...)
```

No requested run may be absent from the manifest. The verifier (§11) confirms
roster completeness at every gate.

### 8.3 Episode-row CSV (analysis convenience)

Aggregator under `experiments/adaptive_beta/tab_six_games/analysis/aggregate.py`
produces a long CSV with the union of §7.1 + §7.2 fields plus `phase`, `stage`,
`game`, `subcase`, `method`, `seed`, `episode`, `regime`, `switch_event`,
`episodes_since_switch`, `oracle_beta`, `beta_sign_correct`,
`beta_lag_to_oracle`, `diverged`, `nan_count`, `overflow_count`. The CSV is the
source of truth for tables and figures.

---

## 9. Directory Layout

### 9.1 Existing roots — extend in place

```text
src/lse_rl/operator/tab_operator.py                        # NO CHANGE
mushroom-rl-dev/.../safe_weighted_common.py                # NO CHANGE preferred

experiments/adaptive_beta/strategic_games/
  matrix_game.py                          # NO CHANGE
  history.py                              # NO CHANGE
  registry.py                             # +register Soda, +new adversaries
  games/
    matching_pennies.py                   # NO CHANGE
    shapley.py                            # NO CHANGE
    rules_of_road.py                      # NO CHANGE
    asymmetric_coordination.py            # NO CHANGE
    strategic_rps.py                      # NO CHANGE
    soda_uncertain.py                     # NEW (M2)
    potential.py                          # NEW iff verifier confirms absent (M2)
  adversaries/
    base.py + 9 existing                  # NO CHANGE
    inertia.py                            # NEW (M3)
    convention_switching.py               # NEW (M3)
    sign_switching_regime.py              # NEW (M3) — controls hidden ξ for composites

experiments/adaptive_beta/agents.py       # NO CHANGE (single-update-path invariant)
experiments/adaptive_beta/schedules.py    # EXTEND with:
  OracleBetaSchedule                      # NEW (M4)
  HandAdaptiveBetaSchedule                # NEW (M4)
  ContractionUCBBetaSchedule              # NEW (M4)
  ReturnUCBBetaSchedule                   # NEW (M4)
  HedgeBetaSchedule                       # NEW (M11, optional)
  DiscountedHedgeBetaSchedule             # NEW (M11, optional)
  GradientBetaSchedule                    # NEW (M11, optional)
  BilevelBetaSchedule                     # NEW (M11, optional)
  build_schedule(...)                     # extend factory dispatch
experiments/adaptive_beta/baselines.py    # NEW (M4)
  RestartQLearningAgent
  SlidingWindowQLearningAgent
  TunedEpsilonGreedyQLearningAgent
```

### 9.2 New Phase VIII root

```text
experiments/adaptive_beta/tab_six_games/                    # NEW
  __init__.py
  composites/
    __init__.py
    sign_switching.py            # G_+ ↔ G_- composite environment wrapper
    composite_registry.py
  runners/
    run_phase_VIII_stage1_beta_sweep.py
    run_phase_VIII_stage2_baselines.py
    run_phase_VIII_stage3_sign_specialization.py
    run_phase_VIII_stage4_sign_switching.py
    run_phase_VIII_stage5_contraction_adaptive.py
    run_phase_VIII_appendix_advanced.py
  configs/
    dev.yaml
    stage1_beta_sweep.yaml
    stage2_baselines.yaml
    stage3_sign_specialization.yaml
    stage4_sign_switching.yaml
    stage5_contraction_adaptive.yaml
  manifests.py                   # Phase8RunRoster
  metrics.py                     # delta metrics from §7.2
  analysis/
    aggregate.py
    sign_specialization.py
    beta_sweep_plots.py
    learning_curves.py
    contraction_plots.py
    sign_switching_plots.py
    table_builder.py
```

### 9.3 Tests

```text
tests/adaptive_beta/strategic_games/                        # extend existing
  test_soda_uncertain.py                  # NEW (M2)
  test_inertia_adversary.py               # NEW (M3)
  test_convention_switching_adversary.py  # NEW (M3)
  test_sign_switching_regime_adversary.py # NEW (M3)
tests/adaptive_beta/tab_six_games/                          # NEW
  test_oracle_beta_schedule.py
  test_hand_adaptive_schedule.py
  test_contraction_ucb_schedule.py
  test_return_ucb_schedule.py
  test_baselines.py
  test_sign_switching_composite.py
  test_phase_VIII_metrics.py
  test_phase8_run_roster.py
  test_runner_smoke.py
```

### 9.4 Results

```text
results/adaptive_beta/tab_six_games/
  raw/                           # per-run dirs from make_run_dir
  processed/
  figures/
  tables/
  manifests/                     # Phase8RunRoster snapshots
  logs/
  paper_update/
```

### 9.5 Spec

```text
docs/specs/phase_VIII_tab_six_games.md     # planner deliverable, mirrors
                                             Phase VII spec structure
```

---

## 10. Stage Plan and Harness Dispatch

### M0 — Planning and Spec Materialization

Owner: `planner`
Tags: `[spec-read] [infra]`

Tasks:

1. Read in full:
   - `AGENTS.md`
   - `CLAUDE.md`
   - this instruction file (v2)
   - `docs/specs/phase_VII_adaptive_beta.md`
   - `results/adaptive_beta/strategic/final_recommendation.md`
   - `results/adaptive_beta/strategic/stage_B2_main_summary.md`
   - `tasks/lessons.md` (all 32+ entries)
   - `src/lse_rl/operator/tab_operator.py` (verify current API)
   - `experiments/adaptive_beta/agents.py` (verify single-update invariant)
   - `experiments/adaptive_beta/schedules.py` (verify factory + DONE schedules)
   - `experiments/adaptive_beta/strategic_games/registry.py` (verify
     GAME_REGISTRY, ADVERSARY_REGISTRY)
2. Confirm potential-game implementation status (§5.6) by searching
   `experiments/adaptive_beta/strategic_games/games/` and adjusting M2 scope.
3. Materialize:
   - `docs/specs/phase_VIII_tab_six_games.md` mirroring Phase VII §0–§23
     structure but scoped to the six-game suite.
   - Phase VIII block in `tasks/todo.md` per §16 of this document.
4. Open a separate task (out-of-scope of M0 implementation) to update
   `CLAUDE.md` §7 — `AGENTS.md` is no longer a stub.
5. Identify any architectural conflict before implementation. If a conflict
   arises (e.g., an existing schedule named the same as a TODO schedule),
   STOP and re-plan.

Gate: user approval unless explicitly running in authorized
overnight/autonomous mode (`AGENTS.md` overnight protocol).

Deliverable:

```text
docs/specs/phase_VIII_tab_six_games.md
tasks/todo.md (Phase VIII block)
```

### M1 — Infrastructure verification + delta

Owners: `verifier`, `experiment-runner`, `test-author`
Tags: `[infra] [logging] [test]`

This milestone is **mostly verification, not implementation.** Phase VII
infrastructure is presumed sufficient until proven otherwise.

Tasks:

1. [V] Verify `MatrixGameEnv`, `GameHistory`, `registry`, the `RunWriter`
   contract, `make_run_dir`, `write_run_json`, `write_metrics_json`,
   `save_npz_with_schema` all behave as documented in §1.4. Run the existing
   `tests/adaptive_beta/strategic_games/` suite under `.venv/bin/python`.
2. [V] Verify `tab_operator.g`, `effective_discount`, and `_EPS_BETA` against
   `tests/algorithms/test_safe_beta0_equivalence.py`.
3. [N] Implement `experiments/adaptive_beta/tab_six_games/manifests.py`
   (`Phase8RunRoster`).
4. [N] Implement `experiments/adaptive_beta/tab_six_games/metrics.py`
   (delta metrics from §7.2).
5. [N] Tests:
   - `test_phase_VIII_metrics.py` — delta-metric definitions, finite under smoke.
   - `test_phase8_run_roster.py` — atomic write, status enum, reconcile.

Gate: `verifier` runs §1.4 existing tests + new Phase VIII metric/manifest
tests. PASS required before M2.

Deliverable: `results/adaptive_beta/tab_six_games/infrastructure_verification.md`.

### M2 — Soda / Uncertain Game (and verify potential game)

Owner: `env-builder`
Tags: `[env] [test]`

Tasks:

1. [N] Implement `strategic_games/games/soda_uncertain.py` with subcases
   `SO-Coordination | SO-AntiCoordination | SO-ZeroSum | SO-BiasedPreference |
   SO-TypeSwitch`. Hidden type via `info["regime"]`.
2. [N] Register Soda in `GAME_REGISTRY`.
3. [V] Verify potential-game implementation (§5.6). If absent, implement
   `strategic_games/games/potential.py` with subcases
   `PG-CoordinationPotential | PG-Congestion | PG-BetterReplyInertia |
   PG-SwitchingPayoff`.
4. [N] `tests/adaptive_beta/strategic_games/test_soda_uncertain.py`,
   optionally `test_potential.py`.
5. Required tests:
   - payoff correctness per type;
   - hidden-type sampling determinism under seed;
   - state-encoder shape;
   - horizon termination;
   - `info["regime"]` schema.

Gate: `verifier` runs M2 tests.

Deliverable: `results/adaptive_beta/tab_six_games/game_implementation_report.md`.

### M3 — Delta adversaries

Owners: `env-builder`, `test-author`
Tags: `[env] [test]`

Tasks:

1. [N] `adversaries/inertia.py` — sticky-action opponent with
   `inertia_lambda` parameter; honours `StrategicAdversary` info-key contract.
2. [N] `adversaries/convention_switching.py` — periodic or stochastic
   convention switch for Rules of the Road `RR-ConventionSwitch`.
3. [N] `adversaries/sign_switching_regime.py` — controller that flips the
   payoff regime ξ_t between G_+ and G_- under exogenous dwell or endogenous
   trigger; consumed by composite envs in M9.
4. Register in `ADVERSARY_REGISTRY`.
5. Tests: probability normalization, finite-memory windows where applicable,
   determinism under seed, info-key compliance.

Gate: `verifier` runs M3 tests.

Deliverable: `results/adaptive_beta/tab_six_games/opponent_baseline_report.md`.

### M4 — Delta β schedules + external baselines

Owners: `algo-implementer`, `operator-theorist`, `calibration-engineer`,
`test-author`
Tags: `[algo] [scheduler] [safety] [test]`

Tasks:

1. [N] `OracleBetaSchedule` in `schedules.py`. Reads `info["regime"]` only
   (assertion: oracle is the only schedule allowed to do so).
2. [N] `HandAdaptiveBetaSchedule` — pre-registered episode rule (e.g.,
   sign-from-rolling-advantage with conservative magnitude), no per-task tuning.
3. [N] `ContractionUCBBetaSchedule` — UCB over β arms `{-2,-1,-0.5,0,0.5,1,2}`
   with reward `M_e(β) = log(R_e+ε) − log(R_{e+1}+ε)` (Bellman residual).
4. [N] `ReturnUCBBetaSchedule` — same UCB scaffold, reward = episode return.
5. [N] Extend `build_schedule(method_id, ...)` to dispatch new schedules.
6. [N] `experiments/adaptive_beta/baselines.py`:
   - `RestartQLearningAgent` — full Q-table reset on a configured trigger
     (rolling-return drop, NaN, divergence).
   - `SlidingWindowQLearningAgent` — sliding-window Q estimates under a
     window length parameter.
   - `TunedEpsilonGreedyQLearningAgent` — vanilla Q with a tuned ε schedule
     selected by Stage 1 dev sweep.
7. [V] Confirm no edits to `tab_operator.py` or `safe_weighted_common.py`.
   If any are made, log in `tasks/lessons.md` with justification and trigger
   ad-hoc Codex review (§11.2 operator focus).
8. [N] Tests in `tests/adaptive_beta/tab_six_games/`:
   - `test_oracle_beta_schedule.py` — uses regime, errors when regime absent.
   - `test_hand_adaptive_schedule.py` — deterministic under seed; rule documented.
   - `test_contraction_ucb_schedule.py` — UCB arm accounting, reward sign,
     finite contraction reward (lessons.md #27).
   - `test_return_ucb_schedule.py` — UCB accounting, return magnitude.
   - `test_baselines.py` — restart trigger, sliding-window discipline,
     ε-schedule shape.
   - `test_beta0_collapse_preserved.py` — every new schedule, when emitting
     β = 0, produces classical Bellman target bit-identically (regression of
     `AdaptiveBetaQAgent`'s assertion).
   - `test_clipping_bounds.py` — β never exits configured `[-β_cap, +β_cap]`.

Gate: `verifier` runs all M4 tests. Codex/adversarial review **required** if
operator or stable infrastructure was modified.

Deliverable: `results/adaptive_beta/tab_six_games/agent_operator_verification.md`.

### M5 — Phase VIII metrics + analysis scaffold

Owners: `experiment-runner`, `plotter-analyst`, `test-author`
Tags: `[logging] [analysis] [plot] [test]`

Tasks:

1. [V] Verify Phase VII metric pipeline emits §7.1 fields correctly under
   a smoke run. List any field the existing pipeline does not produce as a
   gap to fill.
2. [N] Implement `tab_six_games/metrics.py` (delta metrics from §7.2). Single
   source of truth for `contraction_reward`, `recovery_time_after_shift`,
   `beta_sign_correct`, `beta_lag_to_oracle`, `regret_vs_oracle`, etc.
3. [N] Implement `tab_six_games/analysis/aggregate.py` — long-CSV aggregator
   keyed on `(phase, stage, game, subcase, method, seed, episode)`.
4. [N] Plotting scripts (placeholders OK; data wiring tested via smoke):
   - `beta_sweep_plots.py`: β vs AUC, β vs contraction reward.
   - `learning_curves.py`.
   - `contraction_plots.py`: arm-probability traces, log-residual reduction.
   - `sign_switching_plots.py`: switch-aligned return, switch-aligned β.
   - `safety_catastrophe.py`: catastrophic_episodes, beta_clipping_frequency,
     worst_window_return_percentile.
5. Schema parity tests; figure smoke tests.

Gate: `verifier` validates schema parity + delta-metric finiteness.

Deliverable: `results/adaptive_beta/tab_six_games/metric_logging_verification.md`.

### M6 — Stage 1: Fixed-β operator sweep

Owners: `experiment-runner`, `plotter-analyst`
Tags: `[ablation] [analysis] [plot]`

Purpose: show β changes operator behaviour and β = 0 is not universally optimal.

Run (under `tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py` driven
by `configs/stage1_beta_sweep.yaml`):

```text
games:    all six (matching_pennies, shapley, rules_of_road,
                  asymmetric_coordination, soda_uncertain, potential)
subcases: canonical + ≥1 adaptive/nonstationary subcase per game
β grid:   {-2, -1, -0.5, 0, +0.5, +1, +2}
episodes: dev 1k, main 10k
seeds:    dev 3, main 10
```

Outputs:

```text
results/adaptive_beta/tab_six_games/stage1_beta_sweep.md
results/adaptive_beta/tab_six_games/tables/beta_sweep_table.csv
results/adaptive_beta/tab_six_games/figures/beta_vs_auc.pdf
results/adaptive_beta/tab_six_games/figures/beta_vs_contraction.pdf
```

Gate (verifier):

- All `(game, subcase, β, seed)` cells present in roster.
- No silent drops.
- β = 0 runs present and equal classical baseline within numerical tolerance.
- `bellman_residual` finite; `nan_count == 0`.
- At least one (game, subcase) shows β-dependent behaviour OR a negative
  result is explicitly reported.

Review: Codex review recommended after Stage 1 if results enter the main
paper (§11.2 fixed-β focus).

### M7 — Stage 2: Fixed TAB vs vanilla and external baselines

Owners: `experiment-runner`, `plotter-analyst`
Tags: `[ablation] [analysis] [plot]`

Purpose: compare TAB to classical RL and to nonstationary baselines.

Methods:

```text
vanilla_beta_0
best_fixed_positive_TAB         (selected via Stage 1 dev grid; held out for main)
best_fixed_negative_TAB         (selected via Stage 1 dev grid; held out for main)
best_fixed_beta_grid            (reporting aggregate)
restart_Q_learning
sliding_window_Q_learning
tuned_epsilon_greedy_Q_learning
```

`best_fixed_*` selection must use the dev-seed slice; the held-out main-seed
slice is reserved for the headline comparison (lessons.md #19, #20: no
test-set leakage in selection).

Run:

```text
episodes: 10,000
seeds:    10–20 main
games:    all six
subcases: promoted Stage 1 subcases
```

Outputs:

```text
results/adaptive_beta/tab_six_games/stage2_fixed_tab_vs_baselines.md
results/adaptive_beta/tab_six_games/tables/main_fixed_tab_results.csv
results/adaptive_beta/tab_six_games/figures/main_learning_curves.pdf
```

Gate: paired-seed comparison, baseline completeness, no silent drops.

Honesty rule: if sliding-window or restart wins in some cells, report it.

### M8 — Stage 3: Sign-specialization analysis

Owners: `plotter-analyst`, `experiment-runner`
Tags: `[analysis] [ablation]`

Purpose: identify positive-β and negative-β specialist regimes.

Definitions:

```text
G_plus:  fixed-positive beats fixed-negative AND vanilla on AUC (paired seeds)
G_minus: fixed-negative beats fixed-positive AND vanilla on AUC (paired seeds)
```

Tasks:

1. Analyse Stage 1 + Stage 2 results.
2. Cross-reference Phase VII `final_recommendation.md` for prior
   sign-specialization claims; mark agreement/disagreement explicitly.
3. Rank candidate G_+ and G_- subcases.
4. Produce sign-specialization table.
5. Decide whether sign-switching experiments (M9) are justified.

Output:

```text
results/adaptive_beta/tab_six_games/stage3_sign_specialization.md
results/adaptive_beta/tab_six_games/tables/sign_specialization_table.csv
```

Gate: if no credible G_+ exists, stop adaptive sign-switching work and write
a negative-result memo. Do not force adaptive β experiments without
opposite-sign regimes (Phase VII §22 lesson).

### M9 — Stage 4: Sign-switching composite games

Owners: `env-builder`, `experiment-runner`, `plotter-analyst`, `test-author`
Tags: `[env] [ablation] [analysis] [plot] [test]`

Purpose: test adaptive β where adaptation is actually needed.

Tasks:

1. [N] Implement `tab_six_games/composites/sign_switching.py` and
   `composite_registry.py`. Composite wraps a (G_+, G_-) pair and uses
   `sign_switching_regime.py` to control ξ_t.
2. [N] Composite tests: regime determinism, hidden ξ exposed only to oracle,
   payoff correctness within each regime.
3. [X] Oracle validation run: oracle β must beat both fixed signs on AUC and
   recovery for the composite to be a valid adaptivity benchmark. If oracle β
   does not beat fixed signs, redesign the composite once; if still failing,
   stop and write `results/adaptive_beta/tab_six_games/oracle_composite_failed.md`.
4. [X] Adaptive β comparison: methods

```text
vanilla_beta_0
fixed_positive_TAB
fixed_negative_TAB
best_fixed_beta_grid
hand_adaptive_beta
contraction_UCB_beta
oracle_beta
```

Switching variants:

```text
exogenous dwell D ∈ {100, 250, 500, 1000}
endogenous trigger on rolling win/loss/predictability
```

Run:

```text
episodes: 10,000
seeds:    10–20
```

Outputs:

```text
results/adaptive_beta/tab_six_games/stage4_sign_switching.md
results/adaptive_beta/tab_six_games/figures/switch_aligned_return.pdf
results/adaptive_beta/tab_six_games/figures/switch_aligned_beta.pdf
results/adaptive_beta/tab_six_games/figures/beta_sign_accuracy.pdf
```

Gate (verifier): oracle dominance, paired seeds, switch-event accounting.

Review: Codex / adversarial review required if adaptive β claims are
proposed (§11.2 sign-switching focus).

Success criterion:

```text
adaptive β beats vanilla AND beats both fixed signs on AUC and/or recovery
with β sign accuracy and switch-delay evidence.
```

### M10 — Stage 5: Contraction-adaptive β

Owners: `algo-implementer`, `experiment-runner`, `plotter-analyst`,
`test-author`
Tags: `[algo] [scheduler] [ablation] [analysis] [test]`

Purpose: use contraction speed as the reward for β selection.

Methods:

```text
contraction_UCB_beta
return_UCB_beta
contraction_UCB_beta_with_return_safeguard
```

β arms: `{-2, -1, -0.5, 0, +0.5, +1, +2}` (parameterized).
UCB reward (contraction): `M_e(β) = log(R_e_before + ε) − log(R_e_after + ε)`,
where R is Bellman residual, **not** episode return.

Run on:

1. stationary negative control (best Stage 3 G_-);
2. stationary positive control (best Stage 3 G_+);
3. sign-switching composite (validated in M9);
4. one coordination-recovery task (Rules of the Road `RR-ConventionSwitch`).

Outputs:

```text
results/adaptive_beta/tab_six_games/stage5_contraction_adaptive_beta.md
results/adaptive_beta/tab_six_games/figures/contraction_ucb_arm_probs.pdf
results/adaptive_beta/tab_six_games/figures/contraction_ucb_learning_curves.pdf
results/adaptive_beta/tab_six_games/tables/contraction_adaptive_results.csv
```

Gate: UCB accounting, arm counts, contraction-reward correctness, seed
pairing.

Review: Codex / adversarial review required if claiming adaptive β beats
fixed signs.

### M11 — Optional advanced adaptive β (appendix)

Owners: `algo-implementer`, `operator-theorist`, `experiment-runner`
Tags: `[algo] [scheduler] [ablation]`

Run only if M1–M10 are stable and the user authorizes.

Optional methods:

```text
hedge β / exponential weights over β arms
discounted hedge β
continuous gradient β
bilevel / SOBBO-inspired β controller
```

Do not let this delay the core paper result.

Outputs:

```text
results/adaptive_beta/tab_six_games/appendix_advanced_adaptive_beta.md
```

Gate: separate appendix-only review unless results are unexpectedly very
strong.

### M12 — Paper figures, tables, recommendation

Owners: `plotter-analyst`, `planner`, `review-triage`
Tags: `[plot] [analysis] [infra]`

Required main-paper candidate outputs:

```text
results/adaptive_beta/tab_six_games/figures/main_beta_grid_operator_diagnostic.pdf
results/adaptive_beta/tab_six_games/figures/main_learning_curves.pdf
results/adaptive_beta/tab_six_games/figures/main_sign_switching_beta.pdf
results/adaptive_beta/tab_six_games/figures/main_safety_catastrophe.pdf

results/adaptive_beta/tab_six_games/tables/main_table_fixed_tab.csv
results/adaptive_beta/tab_six_games/tables/main_table_baselines.csv
results/adaptive_beta/tab_six_games/tables/main_table_adaptive.csv
```

Required memos:

```text
results/adaptive_beta/tab_six_games/final_recommendation.md
results/adaptive_beta/tab_six_games/paper_update/main_patch.md
results/adaptive_beta/tab_six_games/paper_update/appendix_patch.md
results/adaptive_beta/tab_six_games/paper_update/no_update.md
```

Cross-reference: `results/adaptive_beta/strategic/final_recommendation.md`
(Phase VII). Phase VIII memo must explicitly state agreement, refinement,
or disagreement with Phase VII findings.

Only one paper-update file should recommend action as primary; others remain
as inactive alternatives.

No direct `.tex` paper edit unless the user explicitly authorizes it.

---

## 11. Review and Gate Protocol

### 11.1 Verifier gates

`verifier` runs after each milestone. Required evidence:

- tests run (with `.venv/bin/python` per lessons.md #1);
- `Phase8RunRoster` completeness;
- metric sanity checks (finite contraction reward, no `nan_count > 0`,
  no silent `divergence_event`);
- paired-seed checks for ablations;
- explicit PASS/FAIL per task.

### 11.2 Codex / adversarial review gates

Required:

1. After M4 if any operator or stable infrastructure was touched.
2. After M6 if fixed-β operator sweep enters main paper.
3. After M9 if adaptive β claims are proposed.
4. Final phase close after M12.

Adversarial review focus strings (preserve verbatim from v1):

#### Operator / M4

> Challenge whether TAB updates use the shared safe operator, whether β = 0
> collapses to the classical Bellman update, whether clipping/certification
> is preserved, and whether any duplicate operator math or numerical
> instability was introduced. Verify that no new schedule branches the
> agent's single TD-update path.

#### Fixed-β / M6

> Challenge whether the β-grid sweep fairly isolates operator effects,
> whether β = 0 is a valid baseline, whether `best_fixed_β` selection leaks
> test information, and whether contraction metrics are computed
> consistently across β.

#### Sign-switching / M9

> Challenge whether the sign-switching composite genuinely requires
> adaptive β, whether oracle β beats both fixed signs, whether the hidden
> regime is unavailable to non-oracle methods, and whether adaptive β is
> compared against the best fixed β rather than only vanilla.

#### Final close / M12

> Challenge all paper claims: fixed TAB vs vanilla, adaptive TAB vs fixed β,
> safety/clipping claims, contraction-speed interpretation, external
> baseline fairness, statistical reporting, and honest treatment of
> negative results. Compare Phase VIII final_recommendation.md against
> Phase VII final_recommendation.md and flag any unjustified reversal.

### 11.3 Review triage

All Codex / review results are routed through `review-triage`. Categories:

```text
BLOCKER  MAJOR  MINOR  NIT
```

BLOCKERs must be fixed before phase close. MAJORs must be fixed or
explicitly waived with rationale in `tasks/todo.md`, `tasks/lessons.md`,
and `results/adaptive_beta/tab_six_games/final_recommendation.md`.

---

## 12. Statistical Protocol

All comparisons use paired seeds where possible.

For every main comparison, report:

```text
mean
std
95% confidence interval
paired difference vs vanilla β = 0
paired difference vs best fixed β
paired difference vs sliding-window Q (where relevant)
run count
failure count
```

Primary endpoint: `AUC` (per Phase VII convention).

Secondary endpoints:

```text
episodes_to_threshold
bellman_residual_contraction
recovery_time_after_shift
catastrophic_episodes
final-window return
```

Adaptive β claims require comparison against:

```text
max(fixed_positive_TAB, fixed_negative_TAB, best_fixed_beta_grid)
```

Do not claim adaptive success from beating β = 0 alone.

---

## 13. Success and Failure Criteria

### 13.1 Strong success (NeurIPS-ready)

1. β-grid sweeps showing β = 0 is not universally optimal.
2. Fixed TAB beats vanilla in multiple games / subcases.
3. Safe clipping avoids instability.
4. Opposite-sign specialists G_+ and G_-.
5. A sign-switching composite where oracle β beats both fixed signs.
6. Contraction-adaptive β beats both fixed signs in at least one composite.
7. β trajectories and contraction metrics support the mechanism.
8. Findings are consistent with, or constitute a documented refinement of,
   Phase VII Stage B2 final recommendations.

### 13.2 Medium success

1. Fixed TAB beats vanilla in several games.
2. Fixed-negative dominates cyclic Shapley.
3. Adaptive β is mixed.
4. Sign-switching evidence is suggestive but not decisive.

Paper framing: TAB as safe operator family; adaptive β as promising but
problem-dependent; adaptive results in appendix.

### 13.3 Negative adaptive result

If adaptive β fails but fixed TAB works: report honestly.

> β selection matters, but online β selection is nontrivial. The TAB
> operator family remains useful; adaptive selection requires better
> contraction or bilevel controllers.

No paper claim may say adaptive β dominates fixed β.

---

## 14. Failure Handling

If a task fails:

1. Retry/fix up to two times for local/transient issues.
2. If still failing, stop the current milestone.
3. Preserve logs.
4. Write the appropriate memo:

```text
results/adaptive_beta/tab_six_games/failure_memo.md
results/adaptive_beta/tab_six_games/no_G_plus_found.md
results/adaptive_beta/tab_six_games/oracle_composite_failed.md
results/adaptive_beta/tab_six_games/adaptive_schedule_failure.md
results/adaptive_beta/tab_six_games/external_baseline_analysis.md
```

Do not suppress negative results.

---

## 15. Paper-Update Policy

No direct paper edit until results are reviewed.

Phase VII strategic-games results were classified as exploratory and held
outside the paper's main `§Experiments`. Phase VIII inherits this default
unless §13.1 strong success holds AND the M12 adversarial review passes.

After M12, produce one recommendation:

### Main-paper update if

```text
fixed TAB beats vanilla in multiple games
operator/contraction diagnostics are clean
adaptive β wins in at least one valid sign-switching composite
claims are backed by paired statistics and review gates
findings are consistent with Phase VII or a documented justified refinement
```

### Appendix-only if

```text
fixed TAB results are strong
adaptive β is mixed
sign-switching results are suggestive but not definitive
```

### No update if

```text
β = 0 dominates most tasks
operator diagnostics are unstable
external baselines dominate without clear TAB value
manifest or tests are incomplete
```

Patch outputs:

```text
results/adaptive_beta/tab_six_games/paper_update/main_patch.md
results/adaptive_beta/tab_six_games/paper_update/appendix_patch.md
results/adaptive_beta/tab_six_games/paper_update/no_update.md
```

Only one is the primary recommendation.

---

## 16. Mandatory Todo Skeleton

Add the following Phase VIII block to `tasks/todo.md`. Status legend:
**[V]** verify (existing implementation), **[N]** new implementation,
**[X]** experiment run, **[A]** analysis, **[audit]** verifier/Codex gate.

```text
## Phase VIII — Six-Game Safe TAB Experiment Suite

### M0 — Planning
- [ ] [spec-read] Read AGENTS.md, CLAUDE.md, this instruction file (v2),
      docs/specs/phase_VII_adaptive_beta.md,
      results/adaptive_beta/strategic/final_recommendation.md,
      results/adaptive_beta/strategic/stage_B2_main_summary.md,
      tasks/lessons.md (32+ entries).
- [ ] [spec-read] Verify §1.4 inventory by reading the listed modules
      (tab_operator.py, agents.py, schedules.py, registry.py).
- [ ] [infra] Open separate task: update CLAUDE.md §7 — AGENTS.md is no
      longer a stub.
- [ ] [infra] Verify potential-game implementation status (§5.6); adjust M2.
- [ ] [infra] Write docs/specs/phase_VIII_tab_six_games.md mirroring
      Phase VII §0–§23 structure.
- [ ] [infra] Expand tasks/todo.md Phase VIII block (M1–M12).

### M1 — Infrastructure verification + delta
- [ ] [V][test] Run tests/adaptive_beta/strategic_games/ under .venv/bin/python.
- [ ] [V][operator] Run tests/algorithms/test_safe_beta0_equivalence.py.
- [ ] [N][logging] Implement experiments/adaptive_beta/tab_six_games/manifests.py (Phase8RunRoster).
- [ ] [N][logging] Implement experiments/adaptive_beta/tab_six_games/metrics.py (delta metrics §7.2).
- [ ] [N][test] tests/adaptive_beta/tab_six_games/test_phase8_run_roster.py.
- [ ] [N][test] tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py.
- [ ] [audit] Verifier gate for M1.

### M2 — Soda / Uncertain Game (and verify potential game)
- [ ] [N][env] strategic_games/games/soda_uncertain.py (5 subcases).
- [ ] [N][env] Register Soda in GAME_REGISTRY.
- [ ] [V/N][env] Verify or implement strategic_games/games/potential.py (4 subcases).
- [ ] [N][test] test_soda_uncertain.py (and test_potential.py if implemented).
- [ ] [audit] Verifier gate for M2.

### M3 — Delta adversaries
- [ ] [N][env] adversaries/inertia.py.
- [ ] [N][env] adversaries/convention_switching.py.
- [ ] [N][env] adversaries/sign_switching_regime.py.
- [ ] [N][env] Register in ADVERSARY_REGISTRY.
- [ ] [N][test] Adversary tests (determinism, info-key compliance).
- [ ] [audit] Verifier gate for M3.

### M4 — Delta β schedules + external baselines
- [ ] [N][scheduler] OracleBetaSchedule (uses info["regime"] only).
- [ ] [N][scheduler] HandAdaptiveBetaSchedule.
- [ ] [N][scheduler] ContractionUCBBetaSchedule.
- [ ] [N][scheduler] ReturnUCBBetaSchedule.
- [ ] [N][scheduler] Extend build_schedule(...) factory.
- [ ] [N][algo] baselines.py with Restart/SlidingWindow/TunedEpsilonGreedy.
- [ ] [V][operator] Confirm no edits to tab_operator.py / safe_weighted_common.py.
- [ ] [N][test] test_oracle_beta_schedule.py, test_hand_adaptive_schedule.py,
      test_contraction_ucb_schedule.py, test_return_ucb_schedule.py,
      test_baselines.py, test_beta0_collapse_preserved.py,
      test_clipping_bounds.py.
- [ ] [audit] Verifier gate for M4.
- [ ] [audit] Codex review iff operator/stable code touched.

### M5 — Phase VIII metrics + analysis scaffold
- [ ] [V][analysis] Verify Phase VII metric pipeline emits §7.1 fields.
- [ ] [N][analysis] tab_six_games/analysis/aggregate.py.
- [ ] [N][plot] beta_sweep_plots.py, learning_curves.py, contraction_plots.py,
      sign_switching_plots.py, safety_catastrophe.py.
- [ ] [N][test] Schema parity + figure smoke tests.
- [ ] [audit] Verifier gate for M5.

### M6 — Stage 1 β-grid sweep
- [ ] [X][ablation] Dev β sweep (3 seeds × 1k episodes).
- [ ] [X][ablation] Main β sweep (10 seeds × 10k episodes).
- [ ] [A][analysis] beta_sweep_table.csv.
- [ ] [N][plot] beta_vs_auc.pdf, beta_vs_contraction.pdf.
- [ ] [audit] Verifier gate for M6.
- [ ] [audit] Codex review if results enter main paper.

### M7 — Stage 2 baselines
- [ ] [X][ablation] fixed TAB vs vanilla / external baselines.
- [ ] [A][analysis] main_fixed_tab_results.csv.
- [ ] [N][plot] main_learning_curves.pdf.
- [ ] [audit] Verifier gate for M7.

### M8 — Stage 3 sign specialization
- [ ] [A][analysis] Identify G_+ and G_- (paired-seed AUC).
- [ ] [A][analysis] sign_specialization_table.csv.
- [ ] [A][analysis] Cross-reference Phase VII final_recommendation.md.
- [ ] [audit] Gate: stop adaptive work if no credible G_+/G_-.

### M9 — Stage 4 sign-switching composite
- [ ] [N][env] composites/sign_switching.py + composite_registry.py.
- [ ] [N][test] Composite + oracle tests.
- [ ] [X][ablation] Oracle validation (must beat both fixed signs).
- [ ] [X][ablation] Adaptive β comparison (vanilla, fixed±, hand_adaptive,
      contraction_UCB, oracle).
- [ ] [N][plot] switch_aligned_return.pdf, switch_aligned_beta.pdf,
      beta_sign_accuracy.pdf.
- [ ] [audit] Verifier gate for M9.
- [ ] [audit] Codex/adversarial review if adaptive claims proposed.

### M10 — Stage 5 contraction-adaptive β
- [ ] [N][scheduler] Finalize contraction_UCB_beta + return_UCB_beta + safeguard.
- [ ] [X][ablation] Run on stationary G_+, stationary G_-, composite, recovery.
- [ ] [A][analysis] contraction_adaptive_results.csv.
- [ ] [N][plot] contraction_ucb_arm_probs.pdf, contraction_ucb_learning_curves.pdf.
- [ ] [audit] Verifier gate for M10.

### M11 — Optional advanced (appendix, only if authorized)
- [ ] [N][scheduler] Hedge β / discounted Hedge β.
- [ ] [N][scheduler] Gradient β / Bilevel β.
- [ ] [A][analysis] appendix_advanced_adaptive_beta.md.

### M12 — Final recommendation
- [ ] [A][analysis] Build main-paper candidate tables.
- [ ] [N][plot] Build main-paper candidate figures.
- [ ] [infra] final_recommendation.md (cross-reference Phase VII).
- [ ] [infra] paper_update/{main_patch,appendix_patch,no_update}.md.
- [ ] [audit] Final verifier.
- [ ] [audit] Final Codex / adversarial review.
- [ ] [audit] review-triage close.
```

---

## 17. Short Harness Instruction

> Implement Phase VIII as an extension of the existing Phase VII adaptive-β
> strategic-games stack under `AGENTS.md`. `AGENTS.md` is authoritative
> (`CLAUDE.md` §7's stub claim is out of date; open a separate task to
> reconcile). Read the Phase VII canonical spec, Stage B2 results, and the
> 32-entry `tasks/lessons.md` before planning. Materialize
> `docs/specs/phase_VIII_tab_six_games.md` mirroring Phase VII structure.
> Then add only the deltas: the Soda game (and verify potential game);
> inertia / convention-switching / sign-switching-regime adversaries;
> oracle / hand-adaptive / contraction-UCB / return-UCB β schedules;
> restart / sliding-window / tuned-ε external baselines; sign-switching
> composites; Phase VIII delta metrics; Stage 1–5 runners and analysis. Do
> NOT reimplement existing infrastructure: `MatrixGameEnv`, `GameHistory`,
> `registry`, `AdaptiveBetaQAgent`, `tab_operator.py`,
> `SafeWeightedCommon`, the 5 implemented games, the 9 implemented
> opponents, or the 6 implemented β schedules. Every milestone must pass
> `verifier`; Codex / adversarial review at gates per §11.2; cross-reference
> Phase VII findings in §M12.

---

## 18. v2 Changelog

- **§1 (new):** repository ground-truth section. Inventories existing
  modules with [DONE]/[PARTIAL]/[TODO] tags.
- **§1.1 (new):** flags `CLAUDE.md` §7 as out of date. `AGENTS.md` is the
  10-agent authoritative protocol.
- **§1.2 (new):** acknowledges Phase VII canonical spec and Stage B2
  results as prior art.
- **§2:** retitled "Phase identity"; recommends extending
  `experiments/adaptive_beta/strategic_games/` rather than building
  greenfield `experiments/tab_six_games/`. Adds explicit prohibitions.
- **§4:** replaces generic "shared safe TAB operator" with concrete
  module paths and code snippets.
- **§5:** every game annotated with [DONE]/[TODO]; only Soda and
  potentially the potential-game variant are new.
- **§6:** every method annotated with [DONE]/[TODO]; concrete schedule
  and agent class names.
- **§7.1:** reuses Phase VII metric vocabulary verbatim.
- **§7.2 (new):** Phase VIII delta metrics list (contraction reward,
  recovery time, β sign accuracy, β lag to oracle, etc.).
- **§8:** logging schema reuses Phase VII run.json + metrics.npz; adds
  `Phase8RunRoster` for run-roster status tracking.
- **§9:** directory layout split into "extend in place" + "new Phase VIII
  root" + tests + results. Names actual modules where they will live.
- **§10:** all milestones rewritten to distinguish [V] verification from
  [N] new implementation. M1 is largely verification; M2 implements only
  Soda (and verifies potential game); M3 implements only the three new
  adversaries; M4 implements only the four new β schedules + three external
  baselines; etc.
- **§11.2:** adversarial-review focus strings updated to require
  preservation of `AdaptiveBetaQAgent`'s single-update-path invariant
  and to compare Phase VIII against Phase VII recommendations.
- **§13.1:** strong-success criteria add "consistent with or documented
  refinement of Phase VII findings".
- **§15:** paper-update policy explicitly inherits Phase VII's
  exploratory-by-default classification.
- **§16:** todo skeleton rewritten with [V]/[N]/[X]/[A]/[audit] tags.
- **§17:** short orchestrator instruction tightened, names existing
  components that must NOT be reimplemented.
