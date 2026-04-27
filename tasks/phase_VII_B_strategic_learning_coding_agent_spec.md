# Phase VII-B Strategic Learning Adversary Suite — Coding Agent Instructions

## 0. Purpose

Extend the Phase VII adaptive-β experiment suite with **strategic-learning adversaries** inspired by H. Peyton Young's *Strategic Learning and Its Limits*.

The goal is to test whether adaptive β continues to improve learning dynamics when non-stationarity is **endogenous**: the environment is not merely switching phases by a clock, but is composed of other learners using finite memory, regret, fictitious-play, hypothesis-testing, or payoff-only adaptation.

The current Phase VII result is strong on adversarial RPS, especially on sample efficiency and mechanism metrics. This follow-up should determine whether the same mechanism survives more principled strategic adversaries.

---

## 1. High-Level Experimental Claim to Test

Adaptive β should act as a **low-information temporal credit-assignment controller** in interactive learning settings.

It should help most when:
- the opponent changes behavior in response to the agent,
- local regimes shift,
- fixed optimistic or fixed pessimistic β is globally too blunt,
- informative transitions occur near opponent adaptation events.

It is **not** expected to universally solve equilibrium learning.

The desired paper-safe claim is:

> Adaptive β improves sample efficiency and recovery in selected strategic-learning environments by modulating temporal continuation weight around informative transitions. Its benefit is strongest under endogenous opponent adaptation and weakest in settings where the adaptive signal is trivial, noisy, or too coarse.

---

## 2. Non-Negotiable Constraints

1. **Do not duplicate the TAB / safe weighted log-sum-exp operator.**
   - Reuse the shared operator from:
     `src/lse_rl/operator/tab_operator.py`

2. **Do not modify the already-stable Phase III–VI machinery unless strictly necessary.**
   - If stable infrastructure is touched, record the justification in:
     `tasks/lessons.md`

3. **No silent run drops.**
   - Every `(environment, adversary, method, seed)` triple must be accounted for.
   - Divergence, NaN, overflow, timeout, or early termination must be logged as outcomes.

4. **No forced paper update.**
   - Update paper results only if evidence is strong.
   - If results are weak or mixed, produce appendix/supplement text plus a recommendation memo.

5. **Reproducibility is mandatory.**
   - Same seed grid across methods.
   - Configs must be serializable.
   - Results must be regenerable from manifests.

6. **Bandit-style horizon-1 diagnostics are not mechanism evidence.**
   - Do not use trivial \(v_{next}=0\) settings for alignment-rate or effective-discount claims.

---

## 3. Files to Read Before Coding

Read these first:

```text
AGENTS.md
CLAUDE.md
docs/specs/phase_VII_adaptive_beta.md
tasks/todo.md
tasks/lessons.md
results/adaptive_beta/stage_A_summary.md
results/adaptive_beta/final_recommendation.md
src/lse_rl/operator/tab_operator.py
mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py
```

If any file is absent, record it in the run memo and proceed with the closest available equivalent.

---

## 4. New Code Location

Create a new strategic-games subpackage under the existing adaptive-beta experiment tree:

```text
experiments/adaptive_beta/strategic_games/
  __init__.py
  matrix_game.py
  registry.py
  history.py
  metrics.py

  games/
    __init__.py
    rps.py
    matching_pennies.py
    shapley.py
    rules_of_road.py
    asymmetric_coordination.py
    soda_game.py

  adversaries/
    __init__.py
    base.py
    stationary.py
    scripted_phase.py
    finite_memory_best_response.py
    finite_memory_fictitious_play.py
    smoothed_fictitious_play.py
    regret_matching.py
    finite_memory_regret_matching.py
    hypothesis_testing.py
    realized_payoff_regret.py
    self_play.py

  configs/
    stage_B2_dev.yaml
    stage_B2_main.yaml
    stage_B2_stress.yaml

  analysis/
    aggregate.py
    plot_recovery.py
    plot_beta_events.py
    plot_mechanism.py
    plot_strategic_metrics.py
```

The package may import existing Phase VII runner/logging infrastructure. Avoid forking the runner unless necessary.

---

## 5. Required API

### 5.1 Matrix Game

Implement a generic matrix-game environment that is compatible with the existing MushroomRL-oriented Phase VII stack.

Suggested API:

```python
class MatrixGameEnv(Environment):
    def __init__(
        self,
        payoff_agent: np.ndarray,
        payoff_opponent: np.ndarray | None,
        adversary: StrategicAdversary,
        horizon: int,
        state_encoder: StateEncoder,
        seed: int,
        game_name: str,
        metadata: dict | None = None,
    ):
        ...

    def reset(self, seed: int | None = None):
        ...

    def step(self, action: int):
        ...

    def current_phase(self) -> str:
        ...

    def info(self) -> dict:
        ...
```

The state may encode:
- current timestep,
- previous agent action,
- previous opponent action,
- rolling win/loss statistics,
- adversary phase ID if observable under the selected information regime.

Keep the default information regime **payoff/action-observing but model-hidden**:
- agent observes its own reward,
- agent may observe opponent action if state encoder includes it,
- agent does not observe adversary internal model unless explicitly configured.

---

### 5.2 Strategic Adversary

Implement a base adversary interface:

```python
class StrategicAdversary:
    def reset(self, seed: int | None = None) -> None:
        ...

    def act(self, history: GameHistory, agent_action: int | None = None) -> int:
        ...

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: float | None,
        info: dict | None = None,
    ) -> None:
        ...

    def info(self) -> dict:
        ...
```

Every adversary must expose stable metadata:

```python
{
  "adversary_type": str,
  "phase": str | None,
  "memory_m": int | None,
  "inertia_lambda": float | None,
  "temperature": float | None,
  "model_rejected": bool,
  "search_phase": bool,
  "hypothesis_id": int | None,
  "policy_entropy": float | None,
}
```

---

### 5.3 History Object

Implement a compact history object:

```python
@dataclass
class GameHistory:
    agent_actions: list[int]
    opponent_actions: list[int]
    agent_rewards: list[float]
    opponent_rewards: list[float]
    info: list[dict]

    def last(self, m: int) -> "GameHistory":
        ...

    def empirical_agent_policy(self, m: int | None = None) -> np.ndarray:
        ...

    def empirical_opponent_policy(self, m: int | None = None) -> np.ndarray:
        ...

    def rolling_return(self, m: int) -> float:
        ...
```

Use this for all finite-memory adversaries.

---

## 6. Games to Implement

### 6.1 Matching Pennies

Purpose:
- Low-dimensional zero-sum strategic-learning sanity check.
- Good target for hypothesis-testing adversaries.

Actions:
- 0 = Heads
- 1 = Tails

Agent payoff:

```text
+1 if actions match
-1 otherwise
```

Opponent payoff:
- negative of agent payoff.

Required adversaries:
- stationary mixed,
- finite-memory best response,
- regret matching,
- hypothesis testing.

---

### 6.2 Shapley Cycle Game

Purpose:
- Canonical cycling environment where simple learning rules may fail to converge pathwise.
- Direct test of adaptive β under endogenous cycles.

Implement a standard 3x3 Shapley-style game. Use the payoff convention documented in the code comments and config.

Required adversaries:
- fictitious play,
- smoothed fictitious play,
- finite-memory fictitious play,
- regret matching.

Metrics must include:
- cycling amplitude,
- rolling policy total variation,
- empirical distance to reference mixed strategy,
- return AUC,
- β behavior around support shifts.

---

### 6.3 Rules-of-the-Road Coordination Game

Purpose:
- Positive-control coordination environment.
- Tests recovery after miscoordination and trembles.

Actions:
- 0 = Left
- 1 = Right

Payoff example:

```text
          Opp L    Opp R
Agent L   +1,+1    -1,-1
Agent R   -1,-1    +1,+1
```

Variants:
- symmetric coordination,
- payoff-biased coordination,
- tremble probability ε.

Required adversaries:
- hypothesis testing,
- finite-memory best response,
- inertia adversary.

Metrics:
- time to coordination,
- miscoordination windows,
- recovery after tremble/model rejection,
- catastrophic episodes.

---

### 6.4 Asymmetric Coordination Trap

Purpose:
- Separate long-run empirical frequencies from period-by-period success.
- Detect whether adaptive β improves pathwise coordination.

Implement a 2x2 asymmetric coordination game with one risk-dominant and one payoff-dominant equilibrium.

Required adversaries:
- finite-memory fictitious play,
- smoothed best response,
- inertia.

Metrics:
- period-by-period coordination rate,
- time out of trap,
- drawdown,
- AUC,
- β sign after miscoordination.

---

### 6.5 Strategic RPS with Endogenous Opponents

Purpose:
- Extend the current strong RPS result beyond scripted phase switches.

Use the existing RPS game but replace or supplement scripted phase opponents with:
- finite-memory best response,
- finite-memory regret matching,
- smoothed fictitious play,
- hypothesis testing,
- self-play if Stage B2 dev results are strong.

Metrics:
- AUC,
- recovery after opponent support shift,
- policy cycling amplitude,
- exploitability proxy,
- β alignment,
- effective continuation.

---

### 6.6 Soda / Uncertain Game

Purpose:
- Higher-risk stress test for limited-information strategic learning.
- Agent may not know whether opponent is coordination-like, anti-coordination-like, or matching-pennies-like.

Implement only after the first five games pass dev tests.

Hidden opponent type:
- coordination,
- anti-coordination,
- zero-sum,
- biased preference.

Metrics:
- adaptation time after hidden-type resample,
- return AUC,
- regret,
- β behavior after type change.

This is optional for the first implementation pass.

---

## 7. Adversaries to Implement

### 7.1 Stationary Mixed Opponent

Inputs:
- fixed probability vector over actions.

Used for:
- sanity checks,
- baseline comparisons.

---

### 7.2 Scripted Phase Opponent

Inputs:
- phase schedule,
- per-phase policy vectors.

Used for:
- compatibility with current RPS setup,
- regression against Phase VII results.

---

### 7.3 Finite-Memory Best Response

At each step:
1. Estimate agent empirical action distribution over last `m` rounds.
2. Compute best response to that distribution.
3. With probability `inertia_lambda`, repeat previous action.
4. Otherwise play best response or smoothed best response.

Parameters:

```text
m ∈ {5, 20, 100}
inertia_lambda ∈ {0.0, 0.5, 0.9}
temperature ∈ {0.05, 0.2, 1.0}
```

---

### 7.4 Finite-Memory Fictitious Play

At each step:
1. Estimate opponent belief from history.
2. Best respond to empirical frequencies.
3. Optional smoothing via softmax.

This adversary observes agent actions.

---

### 7.5 Smoothed Fictitious Play

Use quantal/logit best response:

```text
π(a) ∝ exp(Q_br(a) / temperature)
```

Use log-sum-exp stabilization.

---

### 7.6 Regret Matching

Maintain cumulative regret for each action.

At step `t`:

```text
positive_regret = max(regret, 0)
if sum(positive_regret) > 0:
    policy = positive_regret / sum(positive_regret)
else:
    policy = uniform
```

Support both:
- full-information regret matching, where counterfactual payoffs are available from the payoff matrix,
- realized-payoff-only regret matching, where estimates are updated through experimentation.

---

### 7.7 Finite-Memory Regret Matching

Same as regret matching, but regrets are computed over the last `m` observations.

This is preferred for nonstationary adversarial tests because it creates endogenous support shifts.

---

### 7.8 Hypothesis-Testing Opponent

This is a priority adversary.

Behavior:
1. Hold a hypothesis about the agent's action distribution.
2. Play a best response or smoothed best response to the hypothesis.
3. Maintain a test window of length `s`.
4. If observed empirical distribution differs from hypothesis by more than tolerance `tau`, reject the hypothesis.
5. Enter search phase for `search_len` rounds.
6. Sample a new hypothesis and continue.

Parameters:

```text
s ∈ {100, 500, 1000}
tau ∈ {0.025, 0.05, 0.10}
search_len ∈ {10, 50, 100}
temperature ∈ {0.05, 0.2, 1.0}
```

Log:
- `model_rejected`,
- `search_phase`,
- `hypothesis_id`,
- distance between empirical policy and hypothesis.

---

### 7.9 Realized-Payoff-Only Regret Opponent

This is optional for first pass.

Behavior:
- Does not observe full payoff matrix counterfactuals.
- Occasionally experiments.
- Updates action-value estimates from realized payoff only.
- Converts estimated regrets into action probabilities.

Parameters:

```text
epsilon_experiment ∈ {0.01, 0.05, 0.10}
value_lr ∈ {0.01, 0.05, 0.10}
memory_m ∈ {20, 100}
```

---

### 7.10 Self-Play

Implement only if Stage B2 dev results justify it.

Variants:
- vanilla vs vanilla,
- adaptive-β vs vanilla,
- adaptive-β vs adaptive-β,
- adaptive-β vs fixed-positive,
- adaptive-β vs fixed-negative.

Self-play is secondary. Do not use it as the first headline result.

---

## 8. Methods to Compare

Use existing Phase VII methods where available.

Required methods:

```text
vanilla
fixed_positive
fixed_negative
adaptive_beta_clipped
adaptive_sign_only
```

Optional:
```text
adaptive_beta_no_clip
adaptive_magnitude_only
wrong_sign
```

Rules:
- `wrong_sign` is allowed only when a canonical sign is well-defined.
- For strategic matrix games, prefer fixed-positive and fixed-negative rather than forced wrong-sign.
- No-clip divergence is an outcome, not a test failure.

---

## 9. Metrics

### 9.1 Core Performance Metrics

Log per episode:
- return,
- AUC,
- final-window return,
- regret vs fixed oracle,
- regret vs empirical best response,
- catastrophic episode indicator,
- recovery time after adversary event,
- max drawdown.

---

### 9.2 Mechanism Metrics

For nontrivial horizon/state settings:
- alignment rate,
- mean effective continuation,
- fraction `d_eff < gamma`,
- β trajectory,
- advantage distribution,
- Bellman residual.

Important:
- Do not use horizon-1 trivial games as primary mechanism evidence unless state/history encoding makes `v_next` nontrivial.

---

### 9.3 Strategic-Learning Metrics

Add these metrics:

```text
policy_total_variation
rolling_policy_entropy
cycling_amplitude
distance_to_reference_mixed_policy
empirical_best_response_value
external_regret
conditional_regret_proxy
search_phase_return
stable_phase_return
post_rejection_recovery_time
miscoordination_rate
coordination_rate
support_shift_count
```

At minimum implement:
- rolling policy entropy,
- policy total variation,
- support shift count,
- external regret,
- search-phase vs stable-phase return for hypothesis-testing opponents.

---

## 10. Required Event-Aligned Analysis

For hypothesis-testing and finite-memory adversaries, produce event-aligned panels around:
- model rejection,
- search-phase start,
- support shift,
- tremble,
- hidden-type resample.

For each event type, plot:
1. return before/during/after event,
2. β trajectory,
3. effective continuation,
4. alignment rate,
5. opponent policy entropy.

These figures are high priority.

Output paths:

```text
results/adaptive_beta/strategic/figures/event_aligned_return.pdf
results/adaptive_beta/strategic/figures/event_aligned_beta.pdf
results/adaptive_beta/strategic/figures/event_aligned_effective_discount.pdf
results/adaptive_beta/strategic/figures/opponent_entropy.pdf
```

---

## 11. Experimental Stages

### 11.1 Stage B2-Dev

Purpose:
- Fast validation of implementation and signal.

Matrix:

```text
games:
  - matching_pennies
  - strategic_rps
  - shapley
  - rules_of_road

adversaries:
  - finite_memory_best_response
  - finite_memory_regret_matching
  - hypothesis_testing

methods:
  - vanilla
  - fixed_positive
  - fixed_negative
  - adaptive_beta_clipped
  - adaptive_sign_only

seeds:
  - 0
  - 1
  - 2

episodes:
  - 1000
```

Approximate runs:
```text
4 games × 3 adversaries × 5 methods × 3 seeds = 180 runs
```

Output:
```text
results/adaptive_beta/strategic/stage_B2_dev_summary.md
```

Promotion criteria:
- adaptive-β improves AUC over vanilla on paired seeds in at least one `(game, adversary)` setting,
- no clipped adaptive-β divergence,
- mechanism/event metrics directionally support the interpretation,
- effect is not solely from a trivial stationary opponent.

---

### 11.2 Stage B2-Main

Run only promoted settings.

Budget:
```text
1–3 promoted game/adversary pairs
10,000 episodes
10 seeds
core methods
limited ablations
```

Required comparisons:
- vanilla,
- fixed-positive,
- fixed-negative,
- adaptive-beta-clipped,
- adaptive-sign-only.

Optional if time:
- adaptive-magnitude-only,
- no-clip.

Output:
```text
results/adaptive_beta/strategic/stage_B2_main_summary.md
```

---

### 11.3 Stage B2-Stress

Run only if Stage B2-Main is strong.

Stress knobs:
- memory `m`,
- inertia `lambda`,
- hypothesis test tolerance `tau`,
- search length,
- adversary temperature,
- payoff noise,
- action tremble probability.

Output:
```text
results/adaptive_beta/strategic/stage_B2_stress_summary.md
```

---

## 12. Tests

### 12.1 Unit Tests

Create tests under:

```text
tests/adaptive_beta/strategic_games/
```

Required tests:
- payoff matrix indexing,
- environment reset determinism,
- environment step determinism,
- adversary deterministic behavior under fixed seed,
- finite-memory window correctness,
- regret-matching probability normalization,
- hypothesis rejection trigger,
- event logging schema,
- method matrix manifest completeness.

---

### 12.2 Regression Tests

Required:
- scripted RPS strategic implementation matches existing RPS behavior within tolerance when configured equivalently.
- TAB operator outputs are imported from shared kernel, not reimplemented.

---

### 12.3 Reproducibility Tests

Required:
- rerunning the same config with the same seed produces byte-identical summary metrics, or numerically identical arrays within documented tolerance.
- changing seed changes trajectory.

---

## 13. Logging Schema

Every episode row must include:

```text
run_id
seed
game
adversary
method
episode
return
auc_so_far
beta
alignment_rate
mean_effective_discount
bellman_residual
catastrophic
diverged
nan_count
opponent_policy_entropy
policy_total_variation
support_shift
model_rejected
search_phase
phase
memory_m
inertia_lambda
temperature
tau
```

Every transition row, if logged, must include:

```text
run_id
episode
t
state
agent_action
opponent_action
reward
next_state
done
beta
advantage
effective_discount
alignment_indicator
adversary_info_json
```

Use parquet for large logs.

---

## 14. Output Directory

Use:

```text
results/adaptive_beta/strategic/
  raw/
  processed/
  figures/
  tables/
  manifests/
  logs/
  stage_B2_dev_summary.md
  stage_B2_main_summary.md
  stage_B2_stress_summary.md
  final_recommendation.md
```

Every stage summary must include:
- run matrix,
- missing/failed runs,
- metrics table,
- statistical comparison vs vanilla,
- mechanism diagnostics,
- explicit recommendation.

---

## 15. Statistical Reporting

Use paired seeds wherever possible.

For each promoted setting, report:
- mean ± std,
- 95% confidence interval,
- paired difference vs vanilla,
- paired difference vs fixed-positive,
- paired difference vs fixed-negative.

Primary endpoint:
- AUC.

Secondary endpoints:
- recovery time,
- catastrophic episodes,
- drawdown,
- mechanism metrics.

Do not claim superiority from final return alone unless AUC and recovery are also consistent.

---

## 16. Recommended Figures

Generate these for Stage B2-Main:

1. Learning curves: mean ± standard error.
2. AUC paired difference vs vanilla.
3. Event-aligned recovery around adversary model rejection/support shift.
4. β trajectory around adversary events.
5. Effective continuation trajectory around adversary events.
6. Opponent policy entropy and support shifts.
7. Strategic metric table: regret, cycling amplitude, policy TV.

---

## 17. Paper-Update Policy

At the end of Stage B2-Main, decide:

### Main-paper update if:
- at least two strategic settings show strong adaptive-β AUC or recovery gains,
- clipped adaptive-β remains stable,
- event-aligned mechanism metrics support the story,
- effects are not purely from scripted or stationary opponents.

### Appendix-only update if:
- one setting is strong but others are mixed,
- mechanism evidence is good but performance is inconsistent,
- self-play is inconclusive.

### No paper update if:
- results are weak,
- gains are only in trivial settings,
- adaptive-β is unstable,
- fixed-positive or fixed-negative dominates consistently.

Produce one of:

```text
paper_update/main_experiment_patch.md
paper_update/appendix_patch.md
paper_update/no_update_recommendation.md
```

Do not directly edit the paper unless the repository protocol says to do so.

---

## 18. Review Process

Use the existing review protocol.

Required gates:
- verifier after implementation,
- verifier after Stage B2-Dev,
- `/codex:review` after Stage B2-Main,
- `/codex:adversarial-review` if a paper update is proposed.

Review focus:
- Is the strategic adversary genuinely endogenous?
- Are mechanisms nontrivial?
- Are runs complete with no silent drops?
- Are statistical claims paired and seed-consistent?
- Are weak or mixed results represented honestly?
- Is any operator math duplicated?

---

## 19. Failure Handling

If tests fail:
1. Retry/fix up to 2 times.
2. If still failing, stop and write:
   `results/adaptive_beta/strategic/failure_memo.md`

If experiments produce weak or negative results:
- Do not bury them.
- Write:
  `results/adaptive_beta/strategic/negative_result_report.md`

If no-clip diverges:
- Log it.
- Count it.
- Do not treat it as a test failure.

---

## 20. Suggested Initial Implementation Order

1. Matrix-game base environment.
2. History object.
3. Matching pennies.
4. Strategic RPS.
5. Stationary and scripted adversaries.
6. Finite-memory best response.
7. Regret matching.
8. Hypothesis-testing adversary.
9. Metrics and logging.
10. Stage B2-Dev config.
11. Tests.
12. Run Stage B2-Dev.
13. Promote best settings to Stage B2-Main.
14. Generate figures/tables.
15. Write recommendation memo.
16. Propose paper update only if justified.

---

## 21. Highest-Priority Experiments

Implement and run in this order:

1. **Hypothesis-testing matching pennies**
2. **Strategic RPS with finite-memory regret matching**
3. **Shapley cycle game with smoothed fictitious play**
4. **Rules-of-the-road coordination with trembles**
5. **Finite-memory best-response RPS**
6. **Asymmetric coordination trap**
7. **Soda / uncertain game** only if time remains

---

## 22. Final Deliverables

Minimum:

```text
experiments/adaptive_beta/strategic_games/
tests/adaptive_beta/strategic_games/
results/adaptive_beta/strategic/stage_B2_dev_summary.md
results/adaptive_beta/strategic/processed/
results/adaptive_beta/strategic/figures/
results/adaptive_beta/strategic/tables/
results/adaptive_beta/strategic/final_recommendation.md
```

If Stage B2-Main is reached:

```text
results/adaptive_beta/strategic/stage_B2_main_summary.md
results/adaptive_beta/strategic/paper_update/
```

Final memo must answer:

1. Which strategic settings produced adaptive-β gains?
2. Were gains sample-efficiency, final-return, or recovery gains?
3. Did mechanism metrics support the explanation?
4. Did any fixed β dominate?
5. Did any adversary expose a failure mode?
6. Should the paper be updated, appendix-only, or unchanged?

---

## 23. Short Instruction to Coding Agent

Build a strategic-learning adversary suite for Phase VII adaptive-β. Reuse the existing TAB operator and Phase VII infrastructure. Implement matrix games, finite-memory adversaries, regret-matching adversaries, and hypothesis-testing opponents. Start with matching pennies, strategic RPS, Shapley, and coordination games. Run a dev matrix, promote strong settings, generate event-aligned mechanism plots, and produce a recommendation about whether the results are strong enough for the paper.
