# Phase VIII — Six-Game Safe TAB Experiment Suite (canonical spec)

**Status:** active. Single source of truth for the Phase VIII empirical program
on the safe Temporal Allocation Bellman (TAB) operator across six matrix-game
environments.

**Role in the paper:** **exploratory by default** (inherits Phase VII §15
classification). Phase VIII does not modify the current main-paper
§Experiments. After M12 review, the user — not the planner, not the runner —
decides whether results land in the main paper, the appendix, or are held as
supplementary.

**Inputs that define this spec:**
- `tasks/six_game_safe_TAB_harness_instructions.md` (v2, 2026-04-30) — the
  harness-facing implementation and review specification authored against
  repository ground truth. Authoritative for game/adversary/schedule
  inventory, milestone graph, review-gate protocol, and acceptance criteria.
- User decisions locked 2026-04-30 (phase number, Phase VII Stage B2 reuse
  policy, M2 scope, UCB hyperparameters, six architectural-conflict
  resolutions). These supersede any conflicting default in v2.

If the two inputs conflict, the **2026-04-30 user decisions** win; otherwise
the v2 instructions govern.

**Authority over harness behaviour:** `AGENTS.md` (227 lines, 10-agent
roster, dispatch table, Codex protocol, worktree discipline). `CLAUDE.md`
§7's "stub" claim is out of date and a separate task (see §M0 todo block) is
filed to update it.

---

## 0. Scientific objective

Test whether the **safe Temporal Allocation Bellman (TAB) operator**, indexed
by a real β, defines a controlled family of Bellman / DP / RL operators with
empirically distinguishable behaviour, and whether different β regimes
correspond to different temporal-credit / contraction / safety profiles
across a six-game strategic-learning suite.

The empirical claims under test (v2 §0):

1. TAB defines a meaningful family of Bellman operators indexed by β.
2. β changes learning geometry, temporal credit assignment, effective
   contraction, safety behaviour, and sample efficiency.
3. Fixed nonzero β can outperform vanilla β = 0 in controlled strategic RL
   tasks.
4. Safe clipping and certified operator use prevent unstable β behaviour.
5. Adaptive β is useful only when the best β changes across regimes; in
   stationary-sign tasks, the best fixed β may be the correct method.

**Phase VIII is mechanism-and-operator-first.** β-grid sweeps, contraction
diagnostics, safety/clipping behaviour, and sign-specialization carry as
much weight as raw return. Adaptive-β superiority over fixed-β is not the
phase goal — it is one of several testable hypotheses, and a documented
negative adaptive result is an acceptable phase outcome.

---

## 1. What Phase I–VII established

1. Phases I–IV established the safe weighted-LSE operator, calibration
   pipeline, and certification machinery on finite-horizon tabular MDPs with
   per-stage `β̃_t` schedules (`SafeWeightedCommon`, `compute_kappa`,
   `compute_certified_radii`, `build_certification`, `compute_beta_cap`).
2. Phase V (Family A / C) produced the main-paper positive result
   (concentration contrast on aligned propagation) and the safety/stability
   story.
3. Phase VI added a stochastic Family A variant (VI-A through VI-D) and a
   risk-sensitive policy-evaluation finding (VI-G); the paper §Experiments
   was subsequently pruned at commit `40d02f2d` (was `ccec6965` pre-filter-repo).
4. Phase VII-A introduced per-episode adaptive β with the
   centered/scaled paper operator on five tabular envs (`rps`,
   `switching_bandit`, `hazard_gridworld`, `delayed_chain`, plus a Stage-B
   self-play track that did not run). The spec §22 open questions were
   resolved 2026-04-26.
5. Phase VII-B extended Phase VII to strategic-learning adversaries on five
   matrix games (`matching_pennies`, `shapley`, `rules_of_road`,
   `asymmetric_coordination`, `strategic_rps`) with nine endogenous
   adversaries. Stage B2-Main returned a NO UPDATE verdict
   (`results/adaptive_beta/strategic/final_recommendation.md`,
   `stage_B2_main_summary.md`): on the only promoted cells (Shapley ×
   {FM-RM, HypTest}), `fixed_negative` paired-bootstrap-significantly
   dominated `adaptive_beta`, with the adaptive controller saturating to
   the lower clip floor within ~200 episodes. Phase VII-A's Stage A
   adaptive-β gain on `strategic_rps` did not generalise to endogenous
   adversaries.
6. Phase VII delivered the canonical kernel
   `src/lse_rl/operator/tab_operator.py` (option (b) per Phase VII §22.1)
   and rewired `SafeWeightedCommon` to import from it. The kernel is the
   single source of truth for `g`, `rho`, `effective_discount` (scalar +
   batch).
7. Phase VIII is **exploratory** and runs in parallel to the main paper.
   It does not affect the pruned §Experiments unless and until the user
   explicitly promotes results from M12.

---

## 2. Non-negotiable rules

Adapted from Phase V §2 / Phase VII §2; Phase VIII–specific additions in
items 9–14.

1. **Operator.** All β-aware methods compute targets through the canonical
   kernel `src.lse_rl.operator.tab_operator.g(β, γ, r, v)`. Never duplicate
   the math. The classical branch fires at `|β| ≤ _EPS_BETA = 1e-8`,
   returning `r + γ·v` exactly.
2. **β scope (per-episode for adaptive variants).** β for adaptive
   schedules is updated **between episodes only**. Within an episode β is
   a constant. β for episode `e+1` is computed strictly from data observed
   in episode `e` and earlier. Fixed schedules emit a constant β for every
   episode. Oracle is an exception: it reads `info["regime"]` and is the
   only schedule allowed to do so (see §6).
3. **Paired seeds.** All methods on a given (game, subcase) share the same
   environment-RNG stream per seed slot, enabling paired statistical
   comparison. Method-specific RNG offsets exist only for ε-greedy.
4. **Single TD-update path.** `vanilla`, `fixed_*`, `adaptive_*`,
   `oracle_beta`, `hand_adaptive_beta`, `contraction_UCB_beta`,
   `return_UCB_beta`, and any optional appendix schedule all flow through
   `AdaptiveBetaQAgent._step_update`. New methods are new `BetaSchedule`
   subclasses registered in `build_schedule()`. **Do not branch the agent.**
5. **β = 0 bit-identity.** The agent's β=0 assertion (`agents.py` line 338)
   must continue to fire whenever `|β| ≤ _EPS_BETA`. Every new schedule
   that can emit β = 0 (notably the UCB schedules' β=0 arm) must respect
   this invariant.
6. **No future leakage.** β scheduling never reads future-episode data.
   Oracle is the sole exception and reads only the current episode's
   `info["regime"]`.
7. **No silent dropping.** Every requested run is recorded in
   `Phase8RunRoster` (§8.2) with `status ∈ {pending, running, completed,
   failed, diverged, skipped, stopped-by-gate}`. Verifier checks roster
   completeness at every gate.
8. **Cross-method hyperparameter parity.** ε-greedy schedule, learning
   rate, γ, and seed protocol are identical across methods within a stage.
   The only thing that varies is the schedule object (or, for external
   baselines, the agent class).
9. **Operator math is canonical.** Do NOT duplicate `g`, `rho`,
   `effective_discount` math. Do NOT edit
   `src/lse_rl/operator/tab_operator.py` or
   `mushroom-rl-dev/.../safe_weighted_common.py` unless strictly necessary.
   Any edit must be logged in `tasks/lessons.md` with justification and
   trigger ad-hoc Codex review per §11.2 operator focus.
10. **Extend in place.** Phase VIII code lives at
    `experiments/adaptive_beta/tab_six_games/`. Existing code under
    `experiments/adaptive_beta/strategic_games/` is **extended**, not
    forked. Do NOT create `experiments/tab_six_games/` at the repo root.
11. **Game metadata via `game_info()`, never `info()`.** MushroomRL's
    `Environment.info` is a `@property` returning `MDPInfo`. Phase VII
    resolved this collision by exposing spec-§5.1 metadata via
    `game_info()`. Phase VIII inherits the convention. Adding an `info()`
    method that shadows the MushroomRL property is forbidden.
12. **Result root must be explicit.** Every Phase VIII runner must call
    `make_run_dir(base=Path("results/adaptive_beta/tab_six_games"), ...)`
    explicitly. Relying on the module-level `RESULT_ROOT` default
    (`results/weighted_lse_dp`) is a bug and is regression-tested in
    §13.6.
13. **Stage gates are user-controlled.** M3.5 (after Stage A dev pass),
    M6/M7/M8 (Stage 1 → Stage 2 → sign-specialization), M9 (sign-switching
    composite), M10 (contraction-adaptive), M11 (optional appendix), and
    M12 (paper recommendation) all require explicit user sign-off before
    advancing. The runner/orchestrator stops at each gate and reports.
14. **No paper edits in Phase VIII.** Do not touch any `.tex` file. Paper
    deliverables live in
    `results/adaptive_beta/tab_six_games/paper_update/{main,appendix,no_update}_patch.md`.
    Only one is the primary recommendation; others remain as inactive
    alternatives.
15. **Final report verdict.** `results/adaptive_beta/tab_six_games/final_recommendation.md`
    must state, in §1, whether the suite supports a main-paper update,
    an appendix-only update, or no update. Cross-reference Phase VII
    `results/adaptive_beta/strategic/final_recommendation.md` and
    explicitly state agreement, refinement, or disagreement.

---

## 3. Phase identity

```text
Phase number:        VIII
Spec file:           docs/specs/phase_VIII_tab_six_games.md           (this file)
Code roots:
  EXTEND             experiments/adaptive_beta/strategic_games/
  ADD                experiments/adaptive_beta/tab_six_games/
Results root:        results/adaptive_beta/tab_six_games/
Branch:              phase-VIII-tab-six-games-2026-04-30
                     (cut from phase-VII-B-strategic-2026-04-26)
Superseded stub:     tasks/phase_VII_C_sign_switching_coding_agent_spec.md
                     → moved to tasks/archive/phase_VII_C_sign_switching_superseded.md
                     in the M0 commit; referenced in §23.
```

Rationale (v2 §2.2 + 2026-04-30 user decision (a)): Phase VIII's scope (4
new core schedules + 4 optional + 3 baselines + 2 envs + 3 adversaries +
sign-switching composites + 5 staged ablations + delta metrics) is broader
than any sub-phase letter (VII-A/B/C are individually narrow). Phase VIII
as a clean top-level positions the work as a candidate for the main-paper
§Experiments while still inheriting exploratory-by-default classification
(v2 §15).

---

## 4. Operator and effective-discount

### 4.1 Mathematical form (Phase VII §3.1, unchanged)

For β ≠ 0:

```
g_{β,γ}(r, v) = (1 + γ) / β · log((e^{β·r} + γ · e^{β·v}) / (1 + γ))
              = (1 + γ) / β · [logaddexp(β·r, β·v + log γ) − log(1+γ)]
```

For β = 0 (classical collapse):

```
g_{0,γ}(r, v) = r + γ · v
```

### 4.2 Effective continuation (Phase VII §3.2)

```
ρ_{β,γ}(r, v) = e^{β·r} / (e^{β·r} + γ · e^{β·v})
              = sigmoid(β · (r − v) − log γ)
d_{β,γ}(r, v) = ∂_v g_{β,γ}(r, v) = (1 + γ) · (1 − ρ_{β,γ}(r, v))
```

For β = 0, return `d = γ` and `ρ = 1/(1+γ)`.

### 4.3 Alignment condition (Phase VII §3.3)

```
d_{β,γ}(r, v) ≤ γ  ⇔  β · (r − v) ≥ 0
```

Per-transition log: `aligned = (β · (r − v) > 0)` (strict for the headline
metric; non-strict variant logged separately as
`frac_positive_signed_alignment`).

### 4.4 Canonical kernel module — single source of truth

```text
src/lse_rl/operator/tab_operator.py
  g(beta, gamma, r, v)                    -> float
  rho(beta, gamma, r, v)                  -> float
  effective_discount(beta, gamma, r, v)   -> float
  g_batch / rho_batch / effective_discount_batch
  _EPS_BETA = 1e-8                         (classical-collapse threshold)
  _is_classical(beta) -> bool
```

Both Phase III–VI (via
`mushroom_rl.algorithms.value.dp.safe_weighted_common.SafeWeightedCommon.compute_safe_target`
etc.) and Phase VIII (via
`experiments/adaptive_beta/agents.py::AdaptiveBetaQAgent._step_update`)
import from this module. Phase VIII does not edit the kernel.

### 4.5 DP-side certification (cross-reference)

Phase VIII does not run new DP planners by default. If Stage 1 or Stage 5
needs a certified DP reference, use:

```python
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    SafeWeightedCommon, build_certification, compute_kappa,
    compute_certified_radii, compute_beta_cap,
)
# build_certification(alpha_t, R_max, gamma) -> {kappa_t, Bhat_t, beta_cap_t}
# Per-stage clip: np.clip(beta_raw, -beta_cap_t, beta_cap_t)
```

`R_max` must be a configured absolute bound (lessons.md #23).

---

## 5. Environments

Each environment lives at
`experiments/adaptive_beta/strategic_games/games/<name>.py` and registers
itself in `GAME_REGISTRY` via `register_game(...)` at import time. All
envs subclass `mushroom_rl.core.Environment` (Phase VII §22.2 lock).
Game-level metadata is exposed via `game_info()` (a method); the
MushroomRL `info` property returns `MDPInfo` and is not shadowed.

`info` dict keys (mandatory; emitted by `MatrixGameEnv.step`):

```python
info = {
    "phase":              str | int,                # adversary phase string
    "is_shift_step":      bool,
    "catastrophe":        bool,
    "terminal_success":   bool,
    "agent_action":       int | None,
    "opponent_action":    int | None,
    "opponent_reward":    float | None,
    "adversary_info":     dict,
    "game_name":          str,
    "episode_index":      int,
    # Phase VIII additions (where applicable):
    "regime":             str | None,    # ξ_t for Soda, RR-ConventionSwitch,
                                         # sign-switching composites; None
                                         # for envs without a hidden regime
    "oracle_action":      int | None,    # populated where defined
}
```

### 5.1 Matching Pennies — `matching_pennies.py` [DONE]

Actions = {H, T}; reward +1 match, −1 mismatch (zero-sum); horizon
configurable. Subcases for Phase VIII: `MP-Stationary`, `MP-FiniteMemoryBR`,
`MP-RegretMatching`, `MP-HypothesisTesting` — all already provided by the
adversary registry.

<!-- patch-2026-05-01 §6 -->
Role: **null-cell sanity check**. Matching Pennies has expected
advantage $\bar A \to 0$ at the $(\tfrac12,\tfrac12)$ mixed Nash, so
β-induced effects on AUC are second-order on the return axis. A
near-zero result on MP is the *predicted* outcome and confirms the
operator's classical-limit fidelity rather than refuting the TAB story.
Mechanism-level metrics (`alignment_rate`, `frac_d_eff_below_gamma`)
ARE expected to differentiate across β even when AUC does not, and
that differentiation IS evidence of the TAB mechanism. `MP-Stationary`
serves as the bit-identity check against β=0; `MP-FiniteMemoryBR` /
`MP-RegretMatching` / `MP-HypothesisTesting` test second-order
β-effects under nonstationary opponents but should NOT be expected to
produce headline AUC differences. The honesty norm of §13 (negative
results reported faithfully) covers this case. Single-step horizon
variants are mechanism-degenerate (Phase VII §22.5 precedent) —
alignment-rate / `d_eff` panels suppressed for `H = 1`
matching-pennies cells per §10.1.

### 5.2 Shapley Cyclic Game — `shapley.py` [DONE]

Actions = {0, 1, 2}; cyclic non-zero-sum payoff (Brown–Robinson cycles
under greedy BR). Subcases: `SH-FictitiousPlay`, `SH-SmoothedFP`,
`SH-FiniteMemoryRegret`, `SH-HypothesisTesting`. Expected fixed-negative
TAB specialist (Phase VII-B Stage B2-Main empirical pattern).

### 5.3 Rules of the Road — `rules_of_road.py` [DONE]

Actions = {L, R}; reward +c coordinated, −m miscoordinated; supports
`tremble_prob`, `payoff_bias`. Subcases: `RR-StationaryConvention`
[DONE], `RR-ConventionSwitch` [requires M3 `convention_switching.py`],
`RR-Tremble` [DONE via `tremble_prob`], `RR-HypothesisTesting` [DONE].

Role: coordination recovery; convention-switch recovery. Canonical sign:
**+** (coordination favours optimistic propagation).

<!-- patch-2026-05-01 §1 -->
**`RR-Sparse` [TODO M2 reopen]** — sparse-terminal variant exposing TAB's
optimistic-propagation advantage on sparse-reward problems. Per-step
reward is **0**; only the terminal step pays out (`+c` if last action
was coordinated, `−m` if miscoordinated; `c = 1.0`, `m = 0.5` paper
default; tuneable in yaml). Horizon `H = 20` (longer than dense
subcases to stress credit assignment over multiple steps).
Regime-stationary (no hidden regime). Opponent parameterized — accepts
any of the existing RR opponent set (`StationaryConvention` default,
`ConventionSwitch` for stress). State encoding
`(timestep, last_opponent_action)` — same as other RR subcases.
Implementation flag: `sparse_terminal: bool = False` constructor
argument on `RulesOfTheRoadGame`; when `True`, per-step payoffs are
zeroed and only the final-step payoff fires. Registered as
`"rules_of_road_sparse"` in `GAME_REGISTRY`.

*Falsifiable claim* (per patch §1.3): under optimistic Q-init,
`AUC(fixed_+β) > AUC(β=0)` on RR-Sparse with paired-seed bootstrap 95% CI
strictly positive. Mechanism: terminal `+c` propagates backward via the
`max`-like aggregation $g_{+\beta,\gamma}(0, c) \to \gamma c$ as
$\beta \to +\infty$; at finite β the rate of convergence to $\gamma c$
is faster for `+β` than `β=0` on optimistically initialized Q.

### 5.4 Asymmetric Coordination — `asymmetric_coordination.py` [DONE]

Actions = {A, B} (stag-hunt-style). Canonical pre-registration sign:
**+**. Subcases: `AC-FictitiousPlay` [DONE], `AC-SmoothedBR` [DONE],
`AC-Inertia` [requires M3 `inertia.py`], `AC-Trap` [DONE].

Role: miscoordination traps; pathwise dynamics; **falsifiability cell**
for the claim that optimistic TAB selects payoff-dominant equilibria.

<!-- patch-2026-05-01-v7 -->
**AC-Trap is NOT a positive-control G+ cell after the 2026-05-01
pre-sweep.** The original claim (v2 patch §5.2: "+β selects
payoff-dominant equilibria where vanilla Q-learning is risk-dominated"
on stag-hunt) was empirically refuted by a 5-condition ablation
(45 runs across q_init ∈ {0, 5}, episodes ∈ {200, 1000}, opponents
∈ {regret-matching, inertia(0.9), uniform stationary}). In every
condition, fixed +β failed to beat vanilla and often strongly
underperformed; the ordering `AUC(−1) ≥ AUC(0) > AUC(+1)` held with
effect sizes |d(+1, vanilla)| up to 23.4. See
`results/adaptive_beta/tab_six_games/counter_intuitive_findings.md`
for the full table and Codex bug-hunt review.

The mechanism diagnosis is that **TAB sign must be judged by
bootstrap alignment β·(r − v_next), not by the equilibrium payoff
label**. Once learned Q makes `v_next` exceed the realized reward,
+β drives `d_eff` above γ and can exceed 1, destabilizing Q even in
a stag-hunt payoff-dominant environment. The asymptotic forms of
the implemented kernel `g_{β,γ}(r, v) = (1+γ)/β · [logaddexp(βr,
βv + log γ) − log(1+γ)]` are:

- `β → +∞`: `g → (1+γ) · max(r, v)` — when `v > r`, `d_eff → 1+γ > 1`.
- `β → −∞`: `g → (1+γ) · min(r, v)` — when `v > r`, `d_eff → 0`.

So +β stabilizes ONLY in the alignment regime where `v ≤ r`; on
stag-hunt, optimistic Q-init or value-function bootstrapping
quickly violates that condition.

**AC-Trap is therefore repositioned in the suite as a "TAB does not
help here" cell** — explicitly within the §13 honesty norms — and
serves as a useful negative result for the alignment-condition
diagnostic: the diagnostic correctly predicts AC-Trap is OUTSIDE
the +β regime when bootstrap-relative-to-V* is properly accounted
for (`alignment_rate` drops from 0.55–0.76 in the first 20 episodes
to ~0.05 over training in every fixed_beta_+1 run). See §13.5 for
the negative-result reporting requirements.

### 5.5 Soda / Uncertain Game — `soda_uncertain.py` [TODO M2]

Actions = {1..m}; hidden type ξ ∈ {`coordination`, `anti_coordination`,
`zero_sum`, `biased_preference`, `switching`}. Hidden type exposed via
`info["regime"]` (oracle β reads this; non-oracle methods must not).
Subcases: `SO-Coordination`, `SO-AntiCoordination`, `SO-ZeroSum`,
`SO-BiasedPreference`, `SO-TypeSwitch`.

Role: hidden payoff-type uncertainty; appendix-style limits experiment.
Canonical sign: **none** (regime-dependent); `wrong_sign` not defined.

### 5.6 Potential / Weakly Acyclic Game — `potential.py` [TODO M2]

**Verifier confirmed absent on 2026-04-30.** M2 implements alongside Soda.
Document the potential function `Φ` explicitly so M6 verifier can
sanity-check that fixed-positive TAB tracks better-reply dynamics.
Subcases: `PG-CoordinationPotential`, `PG-Congestion`,
`PG-BetterReplyInertia`, `PG-SwitchingPayoff`. Canonical sign: **+**
(potential games admit better-reply dynamics; positive β accelerates
convergence to a Nash equilibrium).

Role: positive control; show TAB accelerates convergence where convergence
is already expected.

<!-- patch-2026-05-01 §11 -->
### 5.7 Long-Horizon Delayed-Reward Chain — `delayed_chain.py` [TODO M2 reopen]

**New game added 2026-05-01 (patch §11).** Anchors the paper title
"Selective Temporal Credit Assignment" in directly testable terms
within the Phase VIII suite (no out-of-suite cross-citation required).
Reintroduces the Phase VII-A `delayed_chain` artifact in the unified
Phase VIII framework; tests TAB's value-propagation behavior on
horizons where temporal credit actually matters (`H ∈ {10, 20, 50}`).

Game contract:

```text
Game id:           delayed_chain
Action set:        subcase-dependent (Discrete(1) or Discrete(2))
Horizon:           subcase-dependent (10, 20, 50)
Reward:            0 at all non-terminal transitions
                   +1 on advance-action arrival at goal terminal
                   -1 on branch_wrong-action arrival at trap terminal
                       (DC-Branching subcases only)
Hidden regime:     none (delayed_chain is regime-stationary)
Opponent:          PassiveOpponent (no-op; opponent payoff irrelevant
                   so the env fits the 2-player MatrixGameEnv wrapper
                   without the opponent affecting payoffs)
State encoding:    integer state-index ∈ [0, L]; state L is goal
                   terminal; for branching subcases, additionally
                   index trap-terminal positions
Canonical sign:    "+"  (positive-β specialist regime; +β should
                          tighten temporal credit propagation
                          backward from terminal +1)
```

Subcases:

```text
DC-Short10        L=10, advance-only chain (1 action), +1 at L=10
                  Tests: short-horizon temporal credit baseline;
                          differentiation from β=0 should be small
                          but consistent in sign

DC-Medium20       L=20, advance-only, +1 at L=20
                  Tests: medium-horizon; differentiation expected

DC-Long50         L=50, advance-only, +1 at L=50
                  Tests: long-horizon temporal credit (the headline
                          cell for the paper title); differentiation
                          should be largest in this cell

DC-Branching20    L=20 with branching: at every state, agent chooses
                  advance OR branch_wrong; branch_wrong leads to a
                  5-state trap chain ending in -1 terminal; advance
                  proceeds toward +1 at goal
                  Tests: temporal credit + exploration in the
                          presence of deceptive negative terminals
```

*Falsifiable predictions* (per patch §11.3) under optimistic Q-init
$Q_0(s,a) \ge V^*(s)$:

```text
<!-- patch-2026-05-01-v3 — replaces P-Sign block per T11 halt resolution. -->
<!-- patch-2026-05-01-v4 — flips P-Contract sign per HALT 2 resolution. -->
P-Contract (v4):
               On advance-only subcases (DC-Short10, DC-Medium20,
               DC-Long50) under optimistic Q-init (Q_0 ≥ Q*),
               Q-convergence rate is monotonically ordered:
                   q_convergence_rate(-β) > q_convergence_rate(0) > q_convergence_rate(+β)
               where q_convergence_rate is the per-episode rate at
               which ||Q_e - Q*||_∞ decays toward 0, and Q* is the
               analytical optimum (Q*(s, advance) = γ^(L-1-s) for
               s ∈ [0, L-1]; Q*(L, ·) = 0; reward delivered on
               L-1 → L transition).

               Theoretical anchor: at every non-terminal step
               r=0 < v=V(s+1) (since optimistic init makes V > 0
               throughout). The alignment condition β·(r-v) ≥ 0
               (spec §3.3) then requires β ≤ 0 for d_{β,γ} ≤ γ.
               Asymptotic g_{β,γ}(0,v) → (1+γ)·v as β → +∞ shows
               why +β destabilizes: each TAB target overshoots the
               classical γ·v target by factor (1+γ)/γ ≈ 2.05
               (γ=0.95), causing Q to drift upward instead of
               converging to Q*. -β with g_{β,γ}(0,v) → 0 as
               β → -∞ is the operator's natural "skeptical
               bootstrap" mode, which on positive-only-terminal
               chains accelerates convergence by suppressing the
               misleadingly-large bootstrap.

               Note: under PESSIMISTIC Q-init (Q_0 < Q*), the sign
               would flip — alignment then requires β ≥ 0 at the
               terminal-propagation step. v4 scope is limited to
               optimistic init (the standard sparse-reward default);
               pessimistic-init contrast is recommended for a v2
               follow-up round.
P-Scaling:     |q_convergence_rate(+β) - q_convergence_rate(0)|
               grows monotonically with chain length L:
                   DC-Short10 < DC-Medium20 < DC-Long50
               Direct test of the long-horizon temporal-credit
               assignment claim from the paper title.
P-AUC-Branch:  On DC-Branching20 (Discrete(2) action space), AUC
               is the natural metric:
                   AUC(+β) > AUC(0) > AUC(-β)
               Effect operates through value-driven exploration: +β
               concentrates value mass on the advance-arm faster,
               accelerating the resolution of the explore/exploit
               tradeoff between advance and branch_wrong.
P-VII-Parity:  q_convergence_rate(0) on DC-Long50 within paired-
               bootstrap 95% CI of Phase VII-A `delayed_chain`
               reference at the matched L=50 setting (cross-validation
               with the prior phase). Note: Phase VII-A reported AUC
               on a chain MDP that may have used Discrete(2) — verify
               metric comparability before claiming parity; if Phase
               VII-A used a different metric, P-VII-Parity is ABSENT
               (mark as N/A in the M6 summary, not a halt).
```

Failure of `P-Contract` (advance-only subcases) or `P-AUC-Branch`
(DC-Branching20) is a **bug-hunt T-class trigger** (T11; see §11.2 /
addendum §3.1 extension below) and dispatches a focused Codex review
with HALT semantics per addendum §6 BLOCKER (this is the headline
temporal-credit-assignment cell — failure is paper-critical).
<!-- patch-2026-05-01-v3 — P-Sign → P-Contract / P-AUC-Branch -->

**Action-space note** (per v3 amendment):
<!-- patch-2026-05-01-v3 -->

```text
DC-Short10, DC-Medium20, DC-Long50:  Discrete(1) — single "advance"
  action. Policy is forced; β affects Q-value convergence speed but
  NOT episode return. Tested via q_convergence_rate metric, NOT AUC.

DC-Branching20:  Discrete(2) — "advance" or "branch_wrong". Policy
  is β-dependent through the Q-value comparison. Tested via AUC
  metric (standard).
```

Implementation hooks:
- `experiments/adaptive_beta/strategic_games/games/delayed_chain.py`
  with `class DelayedChainGame` inheriting the `BaseGame` interface
  used by `MatrixGameEnv`; `game_info()` exposes
  `game="delayed_chain"`, `subcase ∈ {DC-Short10, DC-Medium20,
  DC-Long50, DC-Branching20}`, `horizon=L`, `canonical_sign="+"`,
  `regime=None`, `action_labels=["advance"]` or
  `["advance", "branch_wrong"]`.
- `experiments/adaptive_beta/strategic_games/adversaries/passive.py`
  with `class PassiveOpponent(StrategicAdversary)` registered as
  `"passive"` in `ADVERSARY_REGISTRY` (no-op `act` returning `0`;
  `reset`/`observe` no-ops; `info()` returns the standard
  adversary-info contract with `adversary_type="passive"`,
  `phase="stationary"`, `memory_m=0`, `inertia_lambda=0.0`,
  `temperature=0.0`, `model_rejected=False`, `search_phase="none"`,
  `hypothesis_id=None`, `policy_entropy=0.0`).
- All 4 subcases registered in `GAME_REGISTRY` via `register_game()`.

<!-- patch-2026-05-01-v6 -->
**M6 sweep inclusion** (per patch §11.5): adds 4 subcases × 7 β × 10
seeds = **280 additional runs** to the M6 main pass (extends the
6-game suite to 7 games for sweep purposes). Wall-clock on
parallel-seed infrastructure: ~5–15 min additional; total M6 main
pass after fold-in: see §10.2.

> **Run-count footnote (HALT 5 OQ1, 2026-05-01)**: this section
> previously quoted ~2,800 delayed_chain runs and a ~4,340 total. v2
> patch arithmetic in §1.5 and §11.5 spuriously multiplied subcase
> counts by 10. Actual main-pass total = `1,260 (original 6 games)
> + 70 (RR-Sparse, 1 subcase × 7 β × 10 seeds) + 280 (delayed_chain,
> 4 subcases × 7 β × 10 seeds) + ~210 (additional promoted
> subcases) ≈ 1,820 runs`, validated by the M6 wave 1 runner's
> `# total_runs:` count.

**M7 baseline inclusion** (per patch §11.6): `delayed_chain ×
DC-Long50` MUST be included in the M7 promoted-subcases list.
Strategic-learning agent baselines (`regret_matching_agent`,
`smoothed_fictitious_play_agent`) have NO value-function — they
cannot solve `delayed_chain` and SHOULD produce AUC near random-policy
baseline. This is a **diagnostic feature, not a flaw**: it explicitly
demonstrates that strategic-learning agents without value bootstrapping
fail on temporal credit, justifying the TAB approach. Aggregator must
tolerate "agent cannot solve task" results — report mean ± std and let
the table builder decide whether to format as "—" or numerical.

**Bug-hunt T11 trigger** (extends addendum §3.1 detector list):

```text
T11 — paper-critical prediction failure on delayed_chain (REVISED v3):
       <!-- patch-2026-05-01-v3 -->

       For advance-only subcases (DC-Short10, DC-Medium20, DC-Long50)
       under optimistic Q-init (v4 sign per HALT 2 resolution):
       <!-- patch-2026-05-01-v4 -->
           q_convergence_rate(-β) ≤ q_convergence_rate(0) OR
           q_convergence_rate(0) ≤ q_convergence_rate(+β)
           on any advance-only subcase fires T11.
       (Sign reflects v4 P-Contract: -β should converge fastest, +β
       slowest under optimistic init.)

       For DC-Branching20:
           AUC(+β) ≤ AUC(0) on DC-Branching20 fires T11.

       T11 retains its paper-critical halt semantics: on fire, HALT
       for human review (NOT auto-fix), per addendum §6 BLOCKER
       semantics. The Codex bug-hunt review prompt is extended to
       include: "this contradicts the contraction-speed prediction
       for positive expected advantage on optimistically-initialized
       delayed-reward chains. Investigate (a) Q-init not sufficiently
       optimistic relative to V*, (b) episode horizon binding before
       terminal reward propagates back, (c) ε-greedy schedule
       preventing convergence within the horizon, (d) implementation
       off-by-one in chain transition, (e) implementation off-by-one
       in q_convergence_rate metric, (f) Q* analytical formula bug,
       OR (g) the prediction itself is theoretically misguided."
```

T11 is a sign-flip trigger and fires with high priority — it warrants
HALT for human review per addendum §6 BLOCKER semantics.

**Phase VII-A cross-reference** (per patch §11.8): if
`results/adaptive_beta/strategic/raw/.../phase_VII_A_delayed_chain*`
artifacts survive on disk (filter-repo expunged from git history but
may persist in the working tree), M6 wave 7 aggregator reads them as
historical-baseline reference and includes the comparison in
`M6_summary.md` as **read-only narrative cross-reference only** (same
caveat as M8 Phase VII Stage B2 cross-reference; unpaired across
phases). If artifacts do not exist, `M6_summary.md` notes "no Phase
VII-A reference artifacts available on disk; Phase VIII delayed_chain
results constitute the in-tree validation of the temporal-credit-
assignment claim". This is a documentation choice, NOT a halt
condition.

### 5.8 Adversaries — current status

```text
[DONE] (existing)              | [TODO] (M3)
stationary                      | inertia              (sticky-action; AC-Inertia)
scripted_phase                  | convention_switching (RR-ConventionSwitch; regime in info)
finite_memory_best_response     | sign_switching_regime (composite ξ controller)
finite_memory_fictitious_play   |
smoothed_fictitious_play        |
regret_matching                 |
finite_memory_regret_matching   |
hypothesis_testing              |
realized_payoff_regret (stub)   |
```

**Disambiguation (2026-04-30 conflict resolution 2):**
- `ScriptedPhaseOpponent`: deterministic phase clock, pre-registered
  phase sequence, regime NOT exposed in `info["regime"]`.
- `convention_switching` (M3): stochastic OR periodic switch between two
  conventions for Rules of the Road; regime exposed via
  `info["regime"] ∈ {"left", "right"}` so oracle β can read it.
- `sign_switching_regime` (M3): controller for G_+ ↔ G_- composite envs
  in M9; regime exposed via `info["regime"] ∈ {"plus", "minus"}`. Dwell
  time `D ∈ {100, 250, 500, 1000}` (exogenous variant) plus
  endogenous-trigger variant (rolling win/loss/predictability).

---

## 6. Methods

### 6.1 Core operator methods

```text
[DONE] vanilla                  ZeroBetaSchedule              METHOD_VANILLA
[DONE] fixed_positive           FixedBetaSchedule(+1, β0)     METHOD_FIXED_POSITIVE
[DONE] fixed_negative           FixedBetaSchedule(-1, β0)     METHOD_FIXED_NEGATIVE
[DONE] wrong_sign               WrongSignSchedule              METHOD_WRONG_SIGN
[DONE] adaptive_beta            AdaptiveBetaSchedule           METHOD_ADAPTIVE_BETA
[DONE] adaptive_beta_no_clip    AdaptiveBetaSchedule(no_clip)  METHOD_ADAPTIVE_BETA_NO_CLIP
[DONE] adaptive_sign_only       AdaptiveSignOnlySchedule       METHOD_ADAPTIVE_SIGN_ONLY
[DONE] adaptive_magnitude_only  AdaptiveMagnitudeOnlySchedule  METHOD_ADAPTIVE_MAGNITUDE_ONLY
[NEW]  oracle_beta              OracleBetaSchedule             METHOD_ORACLE_BETA
[NEW]  hand_adaptive_beta       HandAdaptiveBetaSchedule       METHOD_HAND_ADAPTIVE_BETA
[NEW]  contraction_UCB_beta    ContractionUCBBetaSchedule      METHOD_CONTRACTION_UCB_BETA
[NEW]  return_UCB_beta         ReturnUCBBetaSchedule           METHOD_RETURN_UCB_BETA
```

The β-grid sweep in M6 is parameterized over `fixed_beta_-2`,
`fixed_beta_-1`, `fixed_beta_-0.5`, `fixed_beta_+0.5`, `fixed_beta_+1`,
`fixed_beta_+2`, plus `vanilla` (β = 0) — i.e., the seven arms of the
canonical β grid (§6.4) instantiated as `FixedBetaSchedule(±1, β0=...)`
with `β0 ∈ {0.5, 1.0, 2.0}`.

### 6.2 Optional / appendix β schedules

```text
[NEW M11] hedge_beta              HedgeBetaSchedule           METHOD_HEDGE_BETA
[NEW M11] discounted_hedge_beta   DiscountedHedgeBetaSchedule METHOD_DISCOUNTED_HEDGE_BETA
[NEW M11] gradient_beta           GradientBetaSchedule        METHOD_GRADIENT_BETA
[NEW M11] bilevel_SOBBO_beta      BilevelBetaSchedule         METHOD_BILEVEL_SOBBO_BETA
```

Optional methods are implemented only after M1–M10 are stable and the user
authorizes M11 (v2 §6.2). Method-id strings use snake_case, mirroring the
existing `ALL_METHOD_IDS` convention in `schedules.py`.

### 6.3 External / simple baselines (`experiments/adaptive_beta/baselines.py`, [TODO M4])

```text
[NEW M4] restart_Q_learning              RestartQLearningAgent
[NEW M4] sliding_window_Q_learning       SlidingWindowQLearningAgent
[NEW M4] tuned_epsilon_greedy_Q_learning TunedEpsilonGreedyQLearningAgent
```

These baselines exist so the paper is not benchmarked only against β = 0.

<!-- patch-2026-05-01 §3 -->
**Strategic-learning agent baselines (PROMOTED from M11 to M7,
mandatory)**: pre-empts the highest-probability reviewer attack ("weak
baselines"). For a strategic-learning paper, the natural baselines
include strategic-learning agents themselves — fictitious play and
regret matching as *agents*, not opponents. Both are already
implemented as opponent classes in
`experiments/adaptive_beta/strategic_games/adversaries/`; promoting
them to agent baselines is a wrapper over existing code.

```text
[NEW M4 reopen] regret_matching_agent           RegretMatchingAgent
[NEW M4 reopen] smoothed_fictitious_play_agent  SmoothedFictitiousPlayAgent
```

Both classes wrap the existing `regret_matching.py` /
`smoothed_fictitious_play.py` opponents in the agent interface
(`begin_episode` / `step` / `end_episode` / `select_action`) used by
`AdaptiveBetaQAgent` so they slot into the Phase VIII runner without
runner changes. Both:

- Implement the same interface as `AdaptiveBetaQAgent`.
- Carry a `beta_schedule = ZeroBetaSchedule` that is unused (β not
  applicable; included so logger does not error on missing field).
- Emit per-episode metrics: `return`, `length`, `epsilon` (always 0),
  `bellman_residual` (placeholder = 0; document in metric notes that
  the field is undefined for non-Q-learning agents and the aggregator
  must drop it from comparisons), `nan_count`,
  `divergence_event` (always 0).
- Do NOT emit operator-mechanism metrics (`alignment_rate`,
  `mean_d_eff`, etc.) — these are TAB-specific. Aggregator must
  handle the missing fields gracefully.

The remaining optional appendix baseline `UCB_style_Q_learning`
remains deferred to M11.

### 6.4 β arm grid (UCB schedules and M6 sweep)

```yaml
beta_grid: [-2.0, -1.0, -0.5, 0.0, +0.5, +1.0, +2.0]
```

Seven arms. Each arm β is inside `[-β_cap, +β_cap]` with `β_cap = 2.0`
(Phase VII default), so no additional clipping is needed.

### 6.5 UCB schedule contracts (2026-04-30 user-approved)

#### `ContractionUCBBetaSchedule`

- **Reward:** `M_e(β) = log(R_{e-1} + ε) − log(R_e + ε)`, where
  `R_e = bellman_residual` (the per-episode mean of `|td_error|` produced
  by `AdaptiveBetaQAgent.end_episode`). Episode `e` is the episode for
  which the arm β was deployed.
- **ε floor:** `ε = 1e-8`. Implementation uses `np.log(R + ε)`, NOT
  `log1p` (lessons.md #27).
- **Standardisation:** per-arm running mean/std via Welford's algorithm.
  Reward is standardised before UCB scoring. `c = 1.0` for the
  standardised UCB constant.
- **Selection rule:** at the start of episode `e`, choose arm `i =
  argmax_j (μ̂_j_std + c · √(2 · log(N_total) / N_j))` where N_j is the
  pull count for arm j and N_total = Σ N_j. Ties break on lowest arm
  index.
- **Warm-start:** round-robin over arms in episodes 0..6 (one forced pull
  per arm). UCB selection from episode 7 onward.
- **Residual smoothing window:** `residual_smoothing_window = 1` default
  (single-episode contraction reward). Configurable in
  `stage5_contraction_adaptive.yaml`. Revisit in M10 if arm counts
  collapse to a single arm prematurely or oscillate (do NOT tune in M4).

#### `ReturnUCBBetaSchedule`

- **Reward:** episode return (sum of agent rewards over the episode).
- **Standardisation:** per-arm Welford running mean/std (same as
  ContractionUCB; matrix-game returns are not in [0, 1] across games, so
  raw UCB1 with `c = √2` would mis-weight exploration). Reward is
  standardised before UCB scoring.
- **UCB constant:** `c = √2` (canonical UCB1) against the standardised
  reward. The spec must document that `c = √2` is the canonical UCB1
  constant against the standardised reward, not the raw return.
- **Selection rule, warm-start:** identical to ContractionUCB.

#### Sanity test target (per Refinement 1)

The standardised reward stream must converge to mean ≈ 0, std ≈ 1 within
~200 episodes on stationary tasks. Asserted in
`tests/adaptive_beta/tab_six_games/test_return_ucb_standardisation.py`
and in the cross-cutting `test_contraction_ucb_arm_accounting.py`.

#### β = 0 collapse preserved

Whenever an arm β = 0 is selected, the per-step β=0 bit-identity guard in
`AdaptiveBetaQAgent._step_update` (line 338) fires. This is load-bearing;
do not relax. Regression-tested in
`test_beta0_collapse_preserved.py` for every new schedule.

### 6.6 Oracle and hand-adaptive contracts

#### `OracleBetaSchedule`

- Reads `info["regime"]` from the env's per-step info dict on each
  episode-end update. Reading any other field is a spec violation.
- Maps regime → β via a lookup table configured per env:
  - For sign-switching composites (`G_+`, `G_-`): `regime="plus" → β = +β₀`;
    `regime="minus" → β = −β₀` with `β₀ ∈ {0.5, 1.0, 2.0}` (default 1.0).
  - For Soda subcases: lookup table from §M2 deliverable.
- `info["regime"]` MUST be present; missing key raises `KeyError`. The
  oracle is the only schedule that may raise on missing regime.

#### `HandAdaptiveBetaSchedule`

- Pre-registered episode rule (no per-task tuning). Default rule:
  `β_{e+1} = sign(Ā_e) · β₀ · min(1, |Ā_e| / Ā_scale)` with
  `β₀ = 1.0`, `Ā_scale = 0.1`, `λ_smooth = 1.0`. Conservative magnitude;
  no per-task hyperparameter sweep.
- The rule is documented in the schedule's docstring and pinned by
  `test_hand_adaptive_schedule.py` (deterministic under seed).

---

## 7. Metrics

### 7.1 Reuse Phase VII metric vocabulary verbatim

Per-episode (already produced by `RunWriter` and Phase VII metric pipeline;
v2 §7.1):

```text
return, length, epsilon
alignment_rate, mean_signed_alignment, mean_advantage, mean_abs_advantage
mean_d_eff, median_d_eff, frac_d_eff_below_gamma, frac_d_eff_above_one
bellman_residual, td_target_abs_max, q_abs_max
nan_count, divergence_event
catastrophic, success, regret, shift_event
```

### 7.2 Phase VIII delta metrics — implemented in M5

```text
contraction_reward                M_e(β) = log(R_e + ε) − log(R_{e+1} + ε)
                                  with R = bellman_residual (NOT episode return)
empirical_contraction_ratio       (R_{e+1} + ε) / (R_e + ε)
log_residual_reduction            log(R_e + ε) − log(R_{e+1} + ε)
ucb_arm_count                     per-β arm pull count (vector over 7 arms)
ucb_arm_value                     per-β empirical UCB reward (vector)
beta_clip_count                   episodes with β clipped
beta_clip_frequency               beta_clip_count / episode
recovery_time_after_shift         episodes from switch to rolling-return ≥ θ
beta_sign_correct                 1[ sign(β_e) == sign(oracle_β_e) ]
beta_lag_to_oracle                e − argmin_{e' ≤ e} |β_{e'} − oracle_β_e|
regret_vs_oracle                  cumulative oracle return minus method return
catastrophic_episodes             count of episodes with return ≤ θ_low
worst_window_return_percentile    rolling-window worst-percentile return
trap_entries, constraint_violations
overflow_count
```

ε = 1e-8 throughout; use `np.log` directly (lessons.md #27).

### 7.3 Strategic metrics (where applicable)

```text
external_regret                  policy_regret_proxy
coordination_rate                miscoordination_streak_length
cycling_amplitude                policy_total_variation
opponent_policy_entropy          support_shift_count
time_to_equilibrium              distance_to_reference_mixed_strategy
```

Many of these are already emitted by Phase VII strategic-games metric code;
M5 verifier confirms field-name parity before extending.

### 7.4 Schema headers

Episode CSV columns (Phase VII §15.1 + Phase VIII delta):

```
run_id, env, method, seed, episode, phase, beta_raw, beta_deployed,
return, length, epsilon,
alignment_rate, mean_signed_alignment, mean_advantage, mean_abs_advantage,
mean_d_eff, median_d_eff, frac_d_eff_below_gamma, frac_d_eff_above_one,
bellman_residual, td_target_abs_max, q_abs_max,
catastrophic, success, regret, shift_event, divergence_event,
# Phase VIII delta:
contraction_reward, empirical_contraction_ratio, log_residual_reduction,
ucb_arm_index, beta_clip_count, beta_clip_frequency,
recovery_time_after_shift, beta_sign_correct, beta_lag_to_oracle,
regret_vs_oracle, catastrophic_episodes, worst_window_return_percentile,
trap_entries, constraint_violations, overflow_count,
regime, switch_event, episodes_since_switch, oracle_beta
```

Transition CSV/parquet columns (optional, Stage 5 / sign-switching):

```
run_id, env, method, seed, episode, t, state, agent_action,
opponent_action, reward, next_state, done, beta, advantage,
effective_discount, alignment_indicator, regime, opponent_info_json,
phase, oracle_action
```

Stage 1 (M6): full episode CSVs; transition logs only on a sampled
subset (every 10th transition + all `is_shift_step=True` and
`catastrophe=True`) to bound disk usage. Stage 4–5 (M9–M10) write
transition parquet on a similar stratified-sample policy.

---

## 8. Logging schema

### 8.1 Reuse Phase VII run.json + metrics.npz

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
                see §7.4
```

`make_run_dir(base=Path("results/adaptive_beta/tab_six_games"), phase="VIII",
suite=stage_id, task=task_name, algorithm=method_id, seed=seed)` is
called explicitly with the Phase VIII base (§13.6 regression test
enforces this).

### 8.2 Phase8RunRoster — new

```text
experiments/adaptive_beta/tab_six_games/manifests.py    [TODO M1]
  class Phase8RunRoster
    rows: run_id, config_hash, seed, game, subcase, method, status,
          start_time, end_time, result_path, failure_reason, git_commit
    statuses: pending | running | completed | failed | diverged
              | skipped | stopped-by-gate
    write_atomic(...), reconcile_with_disk(...), summarize(...)
```

The roster operates exclusively on `results/adaptive_beta/tab_six_games/`.
M8 sign-specialization analysis reads
`results/adaptive_beta/strategic/` as **read-only narrative reference
only**; no roster cross-population. No requested run may be absent from
the roster. The verifier (§11) confirms roster completeness at every gate.

### 8.3 Episode-row CSV (analysis convenience)

Aggregator under
`experiments/adaptive_beta/tab_six_games/analysis/aggregate.py` produces a
long CSV with the union of §7.1 + §7.2 fields plus `phase`, `stage`,
`game`, `subcase`, `method`, `seed`, `episode`, `regime`, `switch_event`,
`episodes_since_switch`, `oracle_beta`, `beta_sign_correct`,
`beta_lag_to_oracle`, `diverged`, `nan_count`, `overflow_count`. The CSV
is the source of truth for tables and figures.

---

## 9. Artifact layout

### 9.1 Existing roots — extend in place

```text
src/lse_rl/operator/tab_operator.py                        # NO CHANGE
mushroom-rl-dev/.../safe_weighted_common.py                # NO CHANGE preferred

experiments/adaptive_beta/strategic_games/
  matrix_game.py                          # NO CHANGE
  history.py                              # NO CHANGE
  registry.py                             # +register Soda, +potential, +new advs
  games/
    matching_pennies.py                   # NO CHANGE
    shapley.py                            # NO CHANGE
    rules_of_road.py                      # NO CHANGE
    asymmetric_coordination.py            # NO CHANGE
    strategic_rps.py                      # NO CHANGE
    soda_uncertain.py                     # NEW (M2)
    potential.py                          # NEW (M2; verifier confirmed absent)
  adversaries/
    base.py + 9 existing                  # NO CHANGE
    inertia.py                            # NEW (M3)
    convention_switching.py               # NEW (M3)
    sign_switching_regime.py              # NEW (M3)

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
    safety_catastrophe.py
    table_builder.py
```

### 9.3 Tests

```text
tests/adaptive_beta/strategic_games/                        # extend existing
  test_soda_uncertain.py                  # NEW (M2)
  test_potential.py                       # NEW (M2)
  test_inertia_adversary.py               # NEW (M3)
  test_convention_switching_adversary.py  # NEW (M3)
  test_sign_switching_regime_adversary.py # NEW (M3)

tests/adaptive_beta/tab_six_games/                          # NEW
  test_oracle_beta_schedule.py
  test_hand_adaptive_schedule.py
  test_contraction_ucb_schedule.py
  test_contraction_ucb_arm_accounting.py        # 2026-04-30 §13 add
  test_return_ucb_schedule.py
  test_return_ucb_standardisation.py            # 2026-04-30 §13 add
  test_contraction_reward_finite.py             # 2026-04-30 §13 add (regression for lessons.md #27)
  test_baselines.py
  test_sign_switching_composite.py
  test_phase_VIII_metrics.py
  test_phase8_run_roster.py
  test_phase_VIII_result_root.py                # 2026-04-30 §13 add (lessons.md #11)
  test_beta0_collapse_preserved.py
  test_clipping_bounds.py
  test_runner_smoke.py
```

Existing test inventory (from observed repository state, additive vs v2
§1.4): `tests/adaptive_beta/strategic_games/` also carries
`test_operator_shared_kernel.py`, `test_regression_rps.py`, and
`conftest.py`. These remain authoritative for the Phase VII operator
contract and are not modified by Phase VIII.

### 9.4 Results

```text
results/adaptive_beta/tab_six_games/
  raw/                           # per-run dirs from make_run_dir(base=...)
  processed/
  figures/
  tables/
  manifests/                     # Phase8RunRoster snapshots
  logs/
  paper_update/
    main_patch.md                # primary or alternative
    appendix_patch.md            # primary or alternative
    no_update.md                 # primary or alternative
  final_recommendation.md        # M12 deliverable
```

### 9.5 Spec

```text
docs/specs/phase_VIII_tab_six_games.md     # this file
```

---

## 10. Stage protocol — staged

Maps to milestones M6 (Stage 1), M7 (Stage 2), M8 (Stage 3), M9 (Stage 4),
M10 (Stage 5). Stage gates require explicit user sign-off (§2 rule 13).

### 10.1 Stage A — dev pass (3 seeds × 1k episodes)

Performed inside M6 before the main β sweep. Scope:

```yaml
episodes: 1000
seeds: 3
methods: [vanilla, fixed_beta_-2, fixed_beta_-1, fixed_beta_-0.5,
          fixed_beta_+0.5, fixed_beta_+1, fixed_beta_+2]
games: [matching_pennies, shapley, rules_of_road,
        asymmetric_coordination, soda_uncertain, potential]
subcases: canonical + ≥1 nonstationary subcase per game
sensitivity_grid: false
```

Purpose: throughput projection and early sanity check before launching the
main β sweep. **Acceptance:** all cells run; mechanism diagnostics
produced; throughput projection feeds M6 main-pass wall-clock estimate.

### 10.2 Stage 1 — fixed-β operator sweep (M6 main pass)

<!-- patch-2026-05-01 §1 §11 -->
```yaml
episodes: 10000
seeds: 10
methods: 7-arm β grid (vanilla + fixed_beta_{±0.5, ±1, ±2})
games: six core + delayed_chain (per patch §11.5)
       (matching_pennies, shapley, rules_of_road,
        asymmetric_coordination, soda_uncertain, potential,
        delayed_chain)
subcases: canonical + promoted nonstationary subcases (per Stage A)
          + RR-Sparse (per patch §1)
          + DC-Short10, DC-Medium20, DC-Long50, DC-Branching20
            (per patch §11.5)
```

<!-- patch-2026-05-01-v6 -->
Total M6 main-pass run count after fold-in:
`1,260 (original 6 games × promoted subcases × 7 β × 10 seeds)
+ 70 (RR-Sparse, 1 subcase × 7 β × 10 seeds)
+ 280 (delayed_chain, 4 subcases × 7 β × 10 seeds)
+ ~210 (additional promoted nonstationary subcases per Stage A)
≈ **1,820 runs**`.

> **Run-count footnote (HALT 5 OQ1, 2026-05-01)**: this section
> previously quoted ~4,340 runs. v2 patch arithmetic in §1.5 and
> §11.5 spuriously multiplied subcase counts by 10 (e.g.
> `4 × 7 × 10 = 2,800` instead of `4 × 7 × 10 = 280`). The error
> was caught by the M6 wave 1 runner's independent count
> (`stage1_beta_sweep.yaml: # total_runs: 1820`). The corrected
> total is 1,820 dispatched runs.

`alignment_rate_*.pdf` and `effective_discount_*.pdf` are **not produced
for `H = 1` matching-pennies cells** (Phase VII §22.5 precedent —
mechanism degenerate at horizon = 1). Those cells contribute to AUC,
final return, regret, and recovery only.

<!-- patch-2026-05-01 §5 -->
**M6 wave 1.5 — AC-Trap β-pre-sweep** (per patch §5.2). Inserted
immediately after runner + configs are built, before the Stage A
dev pass dispatches. The asymmetric coordination `AC-Trap` subcase
is the single cleanest cell for the "fixed positive TAB selects
payoff-dominant equilibria where vanilla Q-learning is risk-dominated"
claim; if it does not produce the expected effect, the strongest
single-cell result for the paper evaporates and we must know that
before committing to 2-4h of M6 main-pass compute.

```text
game:           asymmetric_coordination
subcase:        AC-Trap
β:              {-1, 0, +1}  (3-arm probe)
seeds:          3
episodes:       200
total runs:     9
estimated wall: ~3 min
expected:       AUC(+1) > AUC(0) > AUC(-1)
                with effect size at least Cohen's d > 0.5 vs vanilla
output:         results/adaptive_beta/tab_six_games/pre_sweep_AC_Trap.md
```

Disposition (per patch §5.3):
- **Match expected**: log success; proceed normally with M6 wave 2
  dev pass.
- **Contradicted**: dispatch a focused Codex bug-hunt review on the
  pre-sweep results (per addendum §3.1, treat as if T1/T3 fired).
  Codex verdict drives:
  - **GENUINE FINDING**: log to `counter_intuitive_findings.md`,
    flag for paper attention, proceed with M6 normally (the finding
    becomes a paper result — payoff-dominance claim gets an asterisk).
  - **BUG**: auto-fix loop per addendum §4.2; re-run pre-sweep after
    fix.
  - **Inconclusive**: HALT with `pre_sweep_inconclusive.md` memo.

<!-- patch-2026-05-01 §4 -->
**M6 wave 5 — figures-only β-grid sub-pass** (per patch §4.2). The
main β grid `{-2, -1, -0.5, 0, +0.5, +1, +2}` has a 0 → ±0.5 gap;
β-vs-AUC and β-vs-contraction curve interpolation across that gap
is visually unfaithful for the headline operator-diagnostic figures.
Add a SUPPLEMENTARY β grid for figures only:

```text
β_supplementary:   {-0.25, -0.1, +0.1, +0.25}
seeds:             5  (lighter than main pass; figures-only)
episodes:          10,000  (same as main pass for shape comparability)
games:             ONE G_+ subcase (preferred RR-StationaryConvention if
                     Stage 1 confirms it as G_+; else AC-FictitiousPlay)
                   + ONE G_- subcase (preferred SH-FiniteMemoryRegret
                     per Phase VII Stage B2)
total runs:        4 β × 5 seeds × 2 subcases = 40 runs
estimated wall:    < 30 minutes
```

The supplementary β values are **NOT** included in the main statistical
sweep. They appear ONLY in the headline figures (`beta_vs_auc.pdf`,
`beta_vs_contraction.pdf`) with a visual marker indicating
"figure-only points" so reviewers know they were not part of the
primary statistical comparison. Aggregator output tags these points
explicitly; tests verify they are excluded from main statistical
tables.

**Acceptance for M6 → M7 promotion:**
- All `(game, subcase, β, seed)` cells present in roster (no silent drops).
- β = 0 runs equal classical baseline within numerical tolerance.
- `bellman_residual` finite; `nan_count == 0` across all cells.
- At least one (game, subcase) shows β-dependent behaviour OR a negative
  result is explicitly reported.
- User signs off on the promoted subcases for M7.

### 10.3 Stage 2 — fixed TAB vs vanilla and external baselines (M7)

<!-- patch-2026-05-01 §3 -->
```yaml
episodes: 10000
seeds: 10–20 main
methods: [vanilla, best_fixed_positive_TAB, best_fixed_negative_TAB,
          best_fixed_beta_grid (reporting aggregate),
          restart_Q_learning, sliding_window_Q_learning,
          tuned_epsilon_greedy_Q_learning,
          regret_matching_agent,                # NEW — promoted from M11
          smoothed_fictitious_play_agent]       # NEW — promoted from M11
games: all six (+ delayed_chain × DC-Long50 per patch §11.6)
subcases: M6-promoted (delayed_chain × DC-Long50 mandatory in promoted list)
```

`best_fixed_*` selection uses the dev-seed slice; the held-out main-seed
slice is reserved for the headline comparison (lessons.md #19, #20: no
test-set leakage).

**Acceptance for M7 → M8 promotion:** paired-seed comparison, baseline
completeness, no silent drops. Honesty rule: if sliding-window or
restart wins in some cells, report it. Strategic-learning agents
(`regret_matching_agent`, `smoothed_fictitious_play_agent`) on
`delayed_chain × DC-Long50` are expected to fail (no value
bootstrapping); this is a diagnostic feature documenting the
necessity of the TAB approach.

### 10.4 Stage 3 — sign-specialization analysis (M8)

Definitions:

```text
G_plus:  fixed-positive beats fixed-negative AND vanilla on AUC (paired seeds)
G_minus: fixed-negative beats fixed-positive AND vanilla on AUC (paired seeds)
```

M8 is analysis-only (no new runs). Cross-references Phase VII
`results/adaptive_beta/strategic/final_recommendation.md` as **read-only
narrative reference**; no paired-seed comparison across phases.

**Acceptance for M8 → M9 promotion:** at least one credible G_+ subcase
AND one credible G_- subcase. If absent, stop adaptive sign-switching
work and write a negative-result memo. Do not force adaptive β
experiments without opposite-sign regimes (Phase VII §22 lesson; v2
§10 M8 gate).

### 10.5 Stage 4 — sign-switching composite games (M9)

Composite envs (v2 §M9 + 2026-04-30 conflict resolution 2):

- `tab_six_games/composites/sign_switching.py`: wraps a (G_+, G_-) pair
  and uses `sign_switching_regime.py` to control ξ_t.
- Switching variants: exogenous dwell `D ∈ {100, 250, 500, 1000}` and
  endogenous trigger (rolling win/loss/predictability).

```yaml
episodes: 10000
seeds: 10–20
methods: [vanilla, fixed_positive_TAB, fixed_negative_TAB,
          best_fixed_beta_grid, hand_adaptive_beta,
          contraction_UCB_beta, oracle_beta]
```

**Oracle-validation gate:** oracle β must beat both fixed signs on AUC
and recovery for the composite to be a valid adaptivity benchmark. If
oracle β does not beat fixed signs, redesign the composite once; if
still failing, stop and write
`results/adaptive_beta/tab_six_games/oracle_composite_failed.md`.

**Acceptance for M9 → M10 promotion:** oracle dominance, paired seeds,
switch-event accounting, regime exposed only to oracle (not to
non-oracle methods).

### 10.6 Stage 5 — contraction-adaptive β (M10)

```text
methods: contraction_UCB_beta, return_UCB_beta,
         contraction_UCB_beta_with_return_safeguard
```

Run on:

1. stationary negative control (best M8 G_-);
2. stationary positive control (best M8 G_+);
3. sign-switching composite (validated in M9);
4. one coordination-recovery task (Rules of the Road
   `RR-ConventionSwitch`).

**Acceptance for M10 → M11/M12:** UCB accounting, arm counts,
contraction-reward correctness, seed pairing.

---

## 11. Statistical reporting

For every metric report mean, std, SE, 95% CI, paired difference vs
`vanilla`, and a paired bootstrap 95% CI on that difference (10,000
resamples).

Required paired comparisons (v2 §12):

1. method vs `vanilla` (β = 0).
2. method vs the best fixed β (per env, declared before the run via
   M6 dev-seed slice; held out for M7+).
3. method vs `sliding_window_Q_learning` (where relevant; M7+).
4. <!-- patch-2026-05-01 §3 --> method vs `regret_matching_agent`
   (M7+; strategic-learning baseline per patch §3.5).
5. <!-- patch-2026-05-01 §3 --> method vs
   `smoothed_fictitious_play_agent` (M7+; strategic-learning
   baseline per patch §3.5).

Use **paired seeds** (same `common_env_seed`) for all significance
summaries. Do not present unpaired mean ± std for the headline
comparison.

Adaptive-β claims require comparison against:

```text
max(fixed_positive_TAB, fixed_negative_TAB, best_fixed_beta_grid)
```

Do not claim adaptive success from beating β = 0 alone.

Primary endpoint: `AUC` (per Phase VII convention).

Secondary endpoints: `episodes_to_threshold`, `bellman_residual_contraction`,
`recovery_time_after_shift`, `catastrophic_episodes`, final-window return.

---

## 12. Figures and tables

### 12.1 Required figures

- `beta_vs_auc.pdf` (M6) — per game, β on x-axis, AUC mean ± SE on y.
  <!-- patch-2026-05-01 §4 --> Includes the M6 wave 5 supplementary β
  points `{-0.25, -0.1, +0.1, +0.25}` on ONE G_+ + ONE G_- subcase,
  plotted with a distinct visual marker labeled "figure-only points"
  to indicate exclusion from the main statistical sweep.
- `beta_vs_contraction.pdf` (M6) — per game, β on x-axis,
  `bellman_residual_contraction` on y.
  <!-- patch-2026-05-01 §4 --> Same supplementary β-marker
  convention as `beta_vs_auc.pdf`.
- `main_learning_curves.pdf` (M7) — per (game, subcase), method curves.
- `sign_specialization_heatmap.pdf` (M8) — game × subcase × method.
- `switch_aligned_return.pdf` (M9) — episodes aligned relative to ξ
  switch event.
- `switch_aligned_beta.pdf` (M9) — β trajectory aligned relative to ξ
  switch event.
- `beta_sign_accuracy.pdf` (M9) — `beta_sign_correct` over training.
- `contraction_ucb_arm_probs.pdf` (M10) — per-arm pull frequency.
- `contraction_ucb_learning_curves.pdf` (M10).
- `safety_catastrophe.pdf` (any stage) — `catastrophic_episodes`,
  `beta_clip_frequency`, `worst_window_return_percentile`.

`alignment_rate_*.pdf` and `effective_discount_*.pdf` are produced per
(game, subcase) where mechanism is well-defined; **not produced for
`H = 1` cells** (Phase VII §22.5 precedent).

### 12.2 M12 paper-candidate figures

```text
results/adaptive_beta/tab_six_games/figures/main_beta_grid_operator_diagnostic.pdf
results/adaptive_beta/tab_six_games/figures/main_learning_curves.pdf
results/adaptive_beta/tab_six_games/figures/main_sign_switching_beta.pdf
results/adaptive_beta/tab_six_games/figures/main_safety_catastrophe.pdf
```

### 12.3 Required tables (saved as both `.csv` and `.tex`)

- **Main β-sweep table (M6):** Game × Subcase × β × {AUC, Final Return,
  Recovery, Mean d_eff, Align Rate, Bellman Residual}.
- **Main fixed-TAB-vs-baselines table (M7):**
  Game × Subcase × Method × {AUC, Final Return, Recovery,
  Catastrophic Episodes, Beta-Clip Freq}.
- **Sign-specialization table (M8):** Game × Subcase × {G_+ candidate
  flag, G_- candidate flag, AUC margin, paired CI}.
- **Sign-switching adaptive table (M9):** Composite × Method × {AUC,
  Recovery, β Sign Accuracy, β Lag-to-Oracle, Regret-vs-Oracle}.
- **Contraction-adaptive table (M10):** Setting × Method × {AUC,
  Bellman Residual Contraction, Arm Pull Distribution, Final Return}.

Mechanism columns (`Align Rate`, `Mean d_eff`) for `H = 1` cells are
reported as `n/a — degenerate at H=1` per Phase VII §22.5 precedent.

---

## 13. Tests

### 13.1 Operator tests (inherited; verified, not re-implemented in M1)

Cross-references `tests/algorithms/test_safe_weighted_lse_operator.py`,
`test_safe_beta0_equivalence.py`, `test_safe_clipping_certification.py`,
and `tests/adaptive_beta/strategic_games/test_operator_shared_kernel.py`.
M1 verifier runs these as part of the inheritance audit.

### 13.2 Schedule tests (M4)

1. `test_oracle_beta_schedule.py` — uses regime, errors when regime
   absent. Asserts oracle is the only schedule that reads
   `info["regime"]`.
2. `test_hand_adaptive_schedule.py` — deterministic under seed; rule
   documented; pre-registered hyperparameters fixed.
3. `test_contraction_ucb_schedule.py` — UCB arm accounting, reward sign,
   finite contraction reward (regression for lessons.md #27).
4. `test_contraction_ucb_arm_accounting.py` (2026-04-30) — total pulls =
   total episodes; each arm pulled at least once during warm-start
   (episodes 0..6); arm-value updates match Welford recursion
   bit-identically.
5. `test_return_ucb_schedule.py` — UCB accounting, return magnitude.
6. `test_return_ucb_standardisation.py` (2026-04-30) — per-arm
   standardised reward stream has empirical mean ≈ 0 and std ≈ 1 after
   1000 stationary episodes (tolerance ≤ 0.1 on each).
7. `test_contraction_reward_finite.py` (2026-04-30) — `M_e(β)` finite
   for all 7 arms across 1000 episodes on each of the 6 games (regression
   for lessons.md #27 expm1/log1p underflow).
8. `test_beta0_collapse_preserved.py` — every new schedule, when
   emitting β = 0, produces classical Bellman target bit-identically
   (regression of `AdaptiveBetaQAgent`'s assertion).
9. `test_clipping_bounds.py` — β never exits configured `[-β_cap, +β_cap]`
   for clipped variants.

### 13.3 Environment tests (M2 + M3)

1. `test_soda_uncertain.py` — payoff correctness per type;
   hidden-type sampling determinism under seed; state-encoder shape;
   horizon termination; `info["regime"]` schema.
2. `test_potential.py` — payoff correctness; potential function `Φ`
   pinned (deterministic given state/action); better-reply dynamics
   sanity (one-step BR strictly increases `Φ` from any non-equilibrium
   state).
3. `test_inertia_adversary.py` — sticky-action probability
   normalisation; finite-memory window; determinism.
4. `test_convention_switching_adversary.py` — periodic / stochastic
   switch determinism; `info["regime"]` exposed.
5. `test_sign_switching_regime_adversary.py` — exogenous dwell
   correctness; endogenous-trigger correctness; regime determinism.

### 13.4 Composite + baselines tests (M9 + M4)

1. `test_sign_switching_composite.py` — regime determinism, hidden ξ
   exposed only to oracle, payoff correctness within each regime,
   switch-event accounting.
2. `test_baselines.py` — restart trigger, sliding-window discipline,
   ε-schedule shape.

### 13.5 Phase VIII metrics + manifest tests (M1 + M5)

1. `test_phase_VIII_metrics.py` — delta-metric definitions, finite under
   smoke (1000 stationary episodes, all 6 games, all 4 new schedules).
2. `test_phase8_run_roster.py` — atomic write, status enum, reconcile
   with disk, append-only invariant, no duplicate `cell_id`.

### 13.6 Result-root regression test (M1; lessons.md #11)

`test_phase_VIII_result_root.py` (2026-04-30, conflict resolution 4):
instantiating `Phase8RunRoster` or calling any `tab_six_games` runner
with default args produces paths under
`results/adaptive_beta/tab_six_games/`, **never** under
`results/weighted_lse_dp/`. Mock `RESULT_ROOT` to assert the explicit
`base=...` argument is passed to `make_run_dir`.

### 13.7 Reproducibility test (M1)

Run `dev.yaml` twice with the same seed and assert byte-identical
`metrics.npz` after sorting nondeterministic dict keys. Inherits
Phase VII §13.4 contract.

### 13.8 No-clip-failure-honesty test (inherited)

Phase VII `tests/adaptive_beta/strategic_games/test_runner_smoke.py`
already exercises the no-clip honest-failure rule. M1 verifier
re-runs this; no Phase VIII delta unless new no-clip failure modes
emerge.

<!-- patch-2026-05-01 §6 -->
### 13.9 Null-cell expectation (Success/Failure criterion per patch §6.2)

(Patch §6.2 originally targeted "spec §13 Success/Failure criteria";
the current spec §13 is "Tests" and §15 is "Acceptance criteria".
Folding the patch text here to preserve the §13.4 numbering intent;
also referenced from §15 acceptance criteria.)

Matching Pennies subcases are expected to produce null AUC results.
This is **not a failure mode**. A non-null result on MP — particularly
a positive AUC differential for fixed +β or fixed −β — would itself
warrant scrutiny (suspect: seed contamination or evaluation-window
bias). Mechanism-level metrics (`alignment_rate`,
`frac_d_eff_below_gamma`) ARE expected to differentiate across β even
when AUC does not, and that differentiation IS the evidence of TAB
mechanism on MP cells.

<!-- patch-2026-05-01-v7 -->
### 13.10 Negative-result honesty: AC-Trap falsifiability cell

The AC-Trap pre-sweep is reported even when it contradicts the
original payoff-dominance prediction (v2 patch §5.2). A valid
M6/M7 report on the AC-Trap cell includes:

1. AUC and paired-seed effect sizes for `β ∈ {-1, 0, +1}` (Stage A
   dev) and the full β grid (Stage 1 main).
2. Alignment-rate traces (`alignment_rate` from per-episode
   `metrics.npz`) showing whether fixed `+β` remains in-regime
   (≥ 0.5) over training. The 2026-05-01 pre-sweep finding is that
   it does NOT — `alignment_rate` collapses from ~0.55–0.76 in the
   first 20 episodes to ~0.05 over training, in every condition
   (5/5 ablations).
3. `q_abs_max` / `divergence_event` diagnostics when `d_eff` rises
   above 1. The 2026-05-01 pre-sweep showed `q_abs_max[-1]` reaching
   1,085,189 on A2 long-horizon fixed_beta_+1 seed 0, with 311
   divergence events out of 1000 episodes (>30% rate).
4. A statement that **TAB sign is governed by `β · (r − v_next)`,
   not by the payoff-dominant equilibrium label**. The
   alignment-condition diagnostic correctly identifies AC-Trap as
   outside the +β regime; the original v2 §5.2 over-claim was at
   the game-theoretic-equilibrium level, not the local-bootstrap
   level the diagnostic actually measures.

**Success for the broader Phase VIII program does not require AC-Trap
to be positive.** It requires that AC-Trap's negative result be
correctly predicted by the alignment diagnostic and not hidden by
post-hoc cell selection. This is consistent with the §13.9 null-cell
honesty norm and the §13.8 no-clip-failure-honesty norm.

If wave 2 (Stage A dev) or wave 4 (Stage 1 main) unexpectedly produces
`+β > 0 > −β` on AC-Trap, that would be a fresh T1+T3 trigger (the
ablation should have caught any genuine +β-favoring regime); halt
for review. Until then, AC-Trap's `−β > 0 > +β` ordering is the
*expected* outcome.

---

## 14. Milestones

| ID    | Scope                                                  | Codex gate? |
|-------|--------------------------------------------------------|-------------|
| M0    | Spec + todo materialization (this commit)              | No          |
| M1    | Infrastructure verification + delta (`Phase8RunRoster`, delta metrics) | No (light)  |
| M2    | Soda + Potential games + tests                         | No (light)  |
| M3    | Inertia, convention-switching, sign-switching-regime adversaries + tests | No (light)  |
| M4    | Oracle, hand-adaptive, contraction-UCB, return-UCB schedules; external baselines; tests | **Yes if operator/stable infra touched** |
| M5    | Phase VIII metrics + analysis scaffold                 | No (light)  |
| M3.5  | (Inside M6) Stage A dev pass user sign-off             | n/a         |
| M6    | Stage 1 β-grid sweep                                   | **Yes if main paper** |
| M7    | Stage 2 fixed TAB vs baselines                         | No (light)  |
| M8    | Stage 3 sign-specialization analysis                   | No (light)  |
| M9    | Stage 4 sign-switching composite games                 | **Yes if adaptive claims** |
| M10   | Stage 5 contraction-adaptive β                         | **Yes if adaptive claims** |
| M11   | Optional advanced (appendix; user-authorized only)     | Appendix-only review |
| M12   | Final recommendation memo + paper patches              | **Yes**     |

### 14.1 Codex / adversarial review gates

Per AGENTS.md §5.2 + v2 §11.2:

1. After M4 if any operator or stable infrastructure was touched.
2. After M6 if fixed-β operator sweep enters main paper.
3. After M9 if adaptive-β claims are proposed.
4. After M10 if claiming adaptive β beats fixed signs.
5. M12 final close.

Adversarial focus strings (verbatim from v2 §11.2):

#### Operator / M4

> Challenge whether TAB updates use the shared safe operator, whether
> β = 0 collapses to the classical Bellman update, whether
> clipping/certification is preserved, and whether any duplicate
> operator math or numerical instability was introduced. Verify that
> no new schedule branches the agent's single TD-update path.

#### Fixed-β / M6

> Challenge whether the β-grid sweep fairly isolates operator effects,
> whether β = 0 is a valid baseline, whether `best_fixed_β` selection
> leaks test information, and whether contraction metrics are computed
> consistently across β.

#### Sign-switching / M9

> Challenge whether the sign-switching composite genuinely requires
> adaptive β, whether oracle β beats both fixed signs, whether the
> hidden regime is unavailable to non-oracle methods, and whether
> adaptive β is compared against the best fixed β rather than only
> vanilla.

#### Final close / M12

> Challenge all paper claims: fixed TAB vs vanilla, adaptive TAB vs
> fixed β, safety/clipping claims, contraction-speed interpretation,
> external baseline fairness, statistical reporting, and honest
> treatment of negative results. Compare Phase VIII
> final_recommendation.md against Phase VII final_recommendation.md
> and flag any unjustified reversal.

### 14.2 M0–M3, M5, M7, M8, M11 light gates

- `verifier` runs the relevant test directories under
  `.venv/bin/python` (lessons.md #1).
- No Codex review unless operator or stable infrastructure is touched.
- The orchestrator must still demand a structured handoff per
  AGENTS.md §7.

---

## 15. Acceptance criteria

The phase is complete when **all** of:

1. β = 0 path exactly reproduces classical Bellman Q-learning targets
   (verified by inherited Phase VII β=0 bit-identity test +
   `test_beta0_collapse_preserved.py` for every new schedule).
2. All methods use the same code path; only the schedule object
   differs (verified by `_step_update_call_counter` introspection in
   `test_baselines.py` and `test_oracle_beta_schedule.py` etc.).
3. β is updated only between episodes for adaptive variants
   (inherited from Phase VII).
4. All methods on a given (game, subcase, seed) share the same
   `common_env_seed` (verified by manifest cross-check).
5. Every result table includes mechanism columns where applicable
   (`align_rate`, `mean_d_eff`); H=1 cells use `n/a — degenerate at H=1`.
6. At least one regime-shift figure shows recovery around a known
   shift point on each Stage-4 environment.
7. No silently dropped runs — `Phase8RunRoster` accounts for every
   `(game, subcase, β, seed)` triple in every stage.
8. `final_recommendation.md` §1 explicitly states main-paper /
   appendix-only / no-update with paired-CI numbers.
9. Stage gates: M6/M7/M8/M9/M10 transitions have user sign-off
   recorded in `tasks/todo.md` (or equivalent durable record).
10. No `.tex` paper edits made in Phase VIII.
11. Phase VIII memo cross-references Phase VII
    `final_recommendation.md` and explicitly states agreement,
    refinement, or disagreement.
12. <!-- patch-2026-05-01 §6 --> Null-cell expectation per §13.9 is
    documented in the final memo: MP cells producing null AUC are
    reported as the predicted outcome (NOT as evidence against TAB);
    a non-null MP AUC differential is flagged for scrutiny.

---

## 16. Final report structure

`results/adaptive_beta/tab_six_games/final_recommendation.md`:

1. **Verdict.** Main-paper / appendix-only / no-update. Paired-CI
   numbers up front. Explicit cross-reference to Phase VII verdict
   with agreement / refinement / disagreement statement.
2. **Experimental setup.** Stages 1–5 scope, env list, method list,
   hyperparameters, seed protocol, β grid, UCB hyperparameters.
3. **Main performance results.** Tables and figures from §12.
4. **Mechanism diagnostics.** Alignment rate, effective-discount
   evidence per (game, subcase). H=1 cells marked degenerate.
5. **Sign-specialization analysis.** G_+ and G_- candidate list with
   paired-CI evidence; cross-reference Phase VII Stage B2 narrative.
6. **Sign-switching adaptive evidence.** Oracle dominance, adaptive
   vs best fixed β, β sign accuracy, β lag-to-oracle.
7. **Contraction-adaptive evidence.** UCB arm distributions, residual
   contraction trajectories, comparison against fixed signs.
8. **Ablations.** β-variant + sensitivity (where run) + difficulty
   knobs.
9. **Failure cases and negative results.** No-clip divergences,
   environments where adaptive β tied or hurt the best fixed β,
   selection bias notes from staged gating.
10. **Recommendation memo.** What (if anything) belongs in main paper
    / appendix / future revision. The user makes the final call.
11. **Open implementation questions.** Anything unresolved at end of
    phase.

---

## 17. Important warnings

1. Do not optimise hyperparameters separately for each method.
2. Do not choose env variants based on adaptive-β performance alone.
3. Do not hide no-clipping failures — they are evidence (Phase VII §18
   inherited).
4. Do not use future-episode information in β scheduling (oracle
   reads only the current `info["regime"]`).
5. Do not duplicate operator math (`g`, `rho`, `effective_discount`).
6. Final conclusions emphasise mechanism evidence + safety, not raw
   return alone.
7. Do not edit the main paper in this phase — that decision belongs
   to the user after reading `final_recommendation.md`.
8. Adaptive-β claims compare against the best fixed β, not only
   `vanilla` (lessons inherited from Phase VII-B Stage B2-Main NO
   UPDATE outcome).

---

## 18. Command-line interface

```bash
.venv/bin/python experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py \
    --config experiments/adaptive_beta/tab_six_games/configs/dev.yaml --seed 0
.venv/bin/python experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py \
    --config experiments/adaptive_beta/tab_six_games/configs/stage1_beta_sweep.yaml --seed 0
.venv/bin/python experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage2_baselines.py \
    --config experiments/adaptive_beta/tab_six_games/configs/stage2_baselines.yaml --seed 0
.venv/bin/python experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage4_sign_switching.py \
    --config experiments/adaptive_beta/tab_six_games/configs/stage4_sign_switching.yaml --seed 0
.venv/bin/python experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage5_contraction_adaptive.py \
    --config experiments/adaptive_beta/tab_six_games/configs/stage5_contraction_adaptive.yaml --seed 0
.venv/bin/python experiments/adaptive_beta/tab_six_games/analysis/aggregate.py \
    --results results/adaptive_beta/tab_six_games/raw \
    --out results/adaptive_beta/tab_six_games/processed
```

Every command writes a `run.json` recording git SHA, argv, timestamp,
Python version, package versions, resolved config (CLAUDE.md §4).

All commands must use `.venv/bin/python` (lessons.md #1).

---

## 19. Deliverables checklist (mapped to milestones)

- [ ] M0 — `docs/specs/phase_VIII_tab_six_games.md` + `tasks/todo.md`
      Phase VIII block.
- [ ] M0 — Move `tasks/phase_VII_C_sign_switching_coding_agent_spec.md`
      to `tasks/archive/phase_VII_C_sign_switching_superseded.md`
      (referenced in §23).
- [ ] M0 — Open separate task: update `CLAUDE.md` §7 stub claim.
- [ ] M1 — `Phase8RunRoster` + delta metrics + result-root regression
      test + verifier gate.
- [ ] M2 — Soda + Potential games + tests + register in `GAME_REGISTRY`.
- [ ] M3 — 3 new adversaries + tests + register in `ADVERSARY_REGISTRY`.
- [ ] M4 — 4 new schedules + 3 baselines + tests + Codex review iff
      operator/stable code touched.
- [ ] M5 — Metric pipeline + plotting scaffold + schema parity.
- [ ] M6 — Stage 1 β-grid sweep (Stage A dev → main pass) + figures
      + tables.
- [ ] M6 — Codex review if results enter main paper.
- [ ] M7 — Stage 2 baselines + figures + tables.
- [ ] M8 — Stage 3 sign-specialization analysis + table.
- [ ] M9 — Stage 4 composites + oracle validation + adaptive
      comparison + figures.
- [ ] M9 — Codex / adversarial review if adaptive claims proposed.
- [ ] M10 — Stage 5 contraction-adaptive β + figures + tables.
- [ ] M11 — Optional advanced (only if authorized).
- [ ] M12 — `final_recommendation.md` + paper patches + final Codex
      review + review-triage close.

---

## 20. Dependencies (blocking edges)

Stages: **A → 1 → 2 → 3 → 4 → 5** (each blocked by user sign-off).

Milestones: **M0 → M1 → {M2, M3, M4, M5} → M6 → M7 → M8 → M9 → M10 →
M11 → M12.** M2–M5 may run in parallel under worktree isolation
(AGENTS.md §6); M6+ are sequential.

Within milestones:

- M1 `[V][operator]` blocks M4 (no operator edits).
- M1 `[N][logging]` (`Phase8RunRoster`) blocks every later milestone's
  run dispatch.
- M1 `[N][logging]` (delta metrics) blocks M5 plotting.
- M2 envs block M6 dispatch (Soda + Potential subcases needed for
  six-game Stage 1).
- M3 adversaries block M9 (sign-switching composite).
- M4 schedules block M9 (oracle, hand-adaptive) and M10 (UCB).
- M4 baselines block M7.
- M5 plotting scaffold blocks M6/M7/M9/M10 figure tasks.
- M6 main pass blocks M7 (best-fixed-β selection from dev-seed slice).
- M7 outputs block M8.
- M8 sign-specialization gate blocks M9 (no G_+/G_- → halt).
- M9 oracle-validation gate blocks M9 adaptive run (oracle must
  dominate).
- M9 outputs + M6 outputs block M10 (composite + stationary
  controls).
- M10 outputs block M12 (or skip with documented justification).
- M12 Codex gate blocks phase closure.

Logging schema (`run.json`, `metrics.npz`, optional transition
parquet) is fixed in M1 and is a prerequisite for every plotting /
aggregation task in M5–M10 and M12.

---

## 21. Failure handling

If a task fails:

1. Retry/fix up to two times for local/transient issues.
2. If still failing, stop the current milestone.
3. Preserve logs.
4. Write the appropriate memo in
   `results/adaptive_beta/tab_six_games/`:
   - `failure_memo.md` — generic milestone failure.
   - `no_G_plus_found.md` — M8 sign-specialization gate failed.
   - `oracle_composite_failed.md` — M9 oracle-validation gate failed.
   - `adaptive_schedule_failure.md` — M10 contraction-UCB failed.
   - `external_baseline_analysis.md` — M7 baselines anomaly.

Do not suppress negative results.

---

## 22. Open questions

**Empty.** All four open decisions from v2 (phase number, Phase VII
Stage B2 reuse, potential-game status, UCB hyperparameters) and all six
architectural-conflict resolutions were locked by the user on
2026-04-30. Subsequent ambiguities encountered during implementation get
appended here and surface for user resolution before the relevant
milestone.

---

## 23. Document changelog

- **2026-04-30 — initial spec written by `planner`** from
  `tasks/six_game_safe_TAB_harness_instructions.md` (v2, 2026-04-30)
  plus user decisions of 2026-04-30:
  - Decision (a): Phase number = **VIII** (clean top-level), branch
    `phase-VIII-tab-six-games-2026-04-30` cut from
    `phase-VII-B-strategic-2026-04-26`. Code root
    `experiments/adaptive_beta/tab_six_games/`. Results root
    `results/adaptive_beta/tab_six_games/`.
  - Decision (b): Phase VII Stage B2 = **rerun** under the new β grid
    for end-to-end paired seeds across M6/M7/M8. M8 cites Phase VII
    Stage B2 numbers as **read-only narrative cross-reference only**.
    Stage A dev pass mandated before M6 main pass for throughput
    projection.
  - Decision (c): Potential game = **confirmed absent**; M2
    implements both `soda_uncertain.py` (5 subcases, hidden ξ in
    `info["regime"]`) and `potential.py` (4 subcases, canonical
    sign **+**, potential function `Φ` documented). M2 env-builder
    budget doubles.
  - Decision (d): UCB hyperparameters approved with two refinements:
    - Refinement 1: `ReturnUCBBetaSchedule` standardises rewards via
      Welford running mean/std (matrix-game returns are not in
      [0, 1] across games). `c = √2` is the canonical UCB1 constant
      against the **standardised** reward. Sanity test:
      standardised stream → mean ≈ 0, std ≈ 1 within ~200 episodes
      on stationary tasks.
    - Refinement 2: `residual_smoothing_window` is a config knob
      (default 1, configurable in
      `stage5_contraction_adaptive.yaml`); revisit in M10 if arm
      counts collapse or oscillate. No tuning in M4.
  - Architectural-conflict resolutions:
    1. `MatrixGameEnv.game_info()` (method) is the spec name;
       `Environment.info` is the MushroomRL `MDPInfo` property and is
       not shadowed.
    2. `scripted_phase` (existing, deterministic clock, regime not
       exposed) vs `convention_switching` (M3, stochastic/periodic,
       regime in `info`) vs `sign_switching_regime` (M3, ξ controller
       for composites with exogenous dwell + endogenous trigger).
    3. `build_schedule` factory dispatch is purely additive; new
       `METHOD_*` constants registered in `ALL_METHOD_IDS`; no name
       collision.
    4. `RESULT_ROOT` default-path drift mandates explicit
       `make_run_dir(base=Path("results/adaptive_beta/tab_six_games"))`
       in every Phase VIII runner, regression-tested by
       `test_phase_VIII_result_root.py` (lessons.md #11).
    5. `CLAUDE.md` §7 stub claim — single-line item in M0 todo block,
       out of scope for the M0 commit; addressed in a separate
       session before M1 verifier closes.
    6. `Phase8RunRoster` operates exclusively on
       `results/adaptive_beta/tab_six_games/`; M8 reads
       `results/adaptive_beta/strategic/` as read-only narrative
       reference only.
  - **Superseded:** `tasks/phase_VII_C_sign_switching_coding_agent_spec.md`
    (untracked stub in pre-Phase-VIII git status) is superseded by
    Phase VIII M9 sign-switching composite work. Moved to
    `tasks/archive/phase_VII_C_sign_switching_superseded.md` in the
    M0 commit.
  - **v2 §1.4 path typo correction:** v2 listed
    `mushroom-rl-dev/mushroom_rl/algorms/value/dp/safe_weighted_common.py`;
    the actual path is `algorithms/value/dp/...`. Spec uses the
    correct path throughout.
  - **Test-suite drift acknowledgement:** observed
    `tests/adaptive_beta/strategic_games/` contains 12 files (v2 §1.4
    catalogued 9). The additional `test_operator_shared_kernel.py`,
    `test_regression_rps.py`, and `conftest.py` are folded into §13
    inventory verbatim.

- **2026-05-01 — Pre-M6 amendment per
  `tasks/phase_VIII_spec_patches_2026-05-01.md` (v2).** Folds in
  patch §1 (RR-Sparse), §3 (FP/RM agents to M7), §4 (figures-only
  β-grid), §5 (AC-Trap pre-sweep), §6 (MP null-cell repositioning),
  §7 (potential-game lemma), §11 (delayed_chain game with 4 subcases
  + PassiveOpponent + bug-hunt T11). §2 RESERVED (high-magnitude
  payoff variant deferred to v2 post-M9). M2 + M4 reopened with new
  delta tasks; M6 main pass grows from ~1,260 to ~1,820 runs (v2
  patch initially quoted ~4,340; corrected at v6 per HALT 5 OQ1 —
  the v2 §1.5/§11.5 arithmetic spuriously multiplied subcase counts
  by 10). Each
  fold-in marked inline with `<!-- patch-2026-05-01 §N -->` for
  provenance.
- **2026-05-01 v3 — T11 halt resolution: advance-only delayed_chain
  subcases switch from AUC to `q_convergence_rate` metric per
  `tasks/phase_VIII_spec_patches_2026-05-01_v3_T11_resolution.md`.**
  Resolves the T11 halt at commit `09e7a262` (smoke prediction failure
  by construction on Discrete(1) advance-only chains). DC-Branching20
  retains AUC. P-Sign rewritten as P-Contract + P-Scaling +
  P-AUC-Branch + P-VII-Parity. T11 trigger semantics preserved (still
  paper-critical, still HALTS on fire). Adds `q_convergence_rate` +
  `q_star_delayed_chain` helpers to
  `experiments/adaptive_beta/tab_six_games/metrics.py` (4 new tests).
  Rewrites `tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py`.
  Aggregator routes `headline_metric` per subcase. Each fold-in
  marked inline with `<!-- patch-2026-05-01-v3 -->`.
- **2026-05-01 v4 — HALT 2 resolution: P-Contract sign flip + Q\*
  off-by-one + terminal-state slice.** v3 P-Contract claim had signs
  reversed; under optimistic Q-init `Q_0 = 1/(1-γ)` on advance-only
  chain, alignment condition `β·(r-v) ≥ 0` with `r=0 < v` requires
  `β ≤ 0` (NEGATIVE β tightens). v4 flips P-Contract ordering to
  `AUC(rate(-1)) > AUC(rate(0)) > AUC(rate(+1))`, fixes the
  `q_star_delayed_chain` exponent off-by-one (`γ^(L-1-s)` not
  `γ^(L-s)`), and adds the terminal-state-slice convention so the
  untouched `Q[L,:]` cell does not dominate the L∞ residual. Each
  fold-in marked inline with `<!-- patch-2026-05-01-v4 -->`.
- **2026-05-01 v5 — HALT 3 resolution: switch headline metric from
  `q_convergence_rate(Q, Q*_classical)` to β-specific Bellman residual
  `||T_β Q − Q||_∞`.** The classical-Q* residual was biased against
  β ≠ 0 because each TAB schedule has its OWN fixed point Q\*_β
  (the operator's asymptotic `g_{β,γ}(0,v) → (1+γ)·v` as `β → +∞` is
  NOT `γ·v`); residuals against Q\*_classical saturate at
  `|Q*_β − Q*_0|` even when the schedule IS converging — to a
  different fixed point. v5 introduces `bellman_residual_beta` and
  `auc_neg_log_residual` helpers in `metrics.py` (4 additional tests),
  and rewrites the DC-Long50 smoke against the new metric. Empirical:
  AUC(-log R_-1)=+8660, AUC(-log R_0)=+4607, AUC(-log R_+1)=-23599;
  final R separated by 17 orders of magnitude between β=-1 (3.45e-11,
  contracted) and β=+1 (2.69e+06, divergent). Each fold-in marked
  inline with `<!-- patch-2026-05-01-v5 -->`.
- **2026-05-01 v7 — HALT 6 resolution: AC-Trap repositioned as
  falsifiability cell after 5/5 ablation conditions refuted v2 §5.2
  payoff-dominance prediction.** M6 wave 1.5 baseline (q_init=0,
  200 ep, regret-matching opponent) showed `AUC(−1) > AUC(0) > AUC(+1)`
  with `d(+1, vanilla) = −3.04` — the OPPOSITE of the spec-predicted
  `AUC(+1) > AUC(0) > AUC(−1)`. A 36-run ablation across q_init ∈ {0, 5},
  episodes ∈ {200, 1000}, opponents ∈ {regret-matching, inertia(0.9),
  uniform stationary} confirmed the reversal in 5/5 conditions, with
  the largest effect under optimistic init (`d(+1, 0) = −8.60`) and
  longest training (`d(+1, 0) = −23.38`). Codex GENUINE-FINDING
  verdict (memo at
  `results/adaptive_beta/tab_six_games/codex_reviews/AC_Trap_pre_sweep_review_2026-05-01T16-52-53Z.md`)
  found no implementation bug across the operator, agent, schedule,
  game, and three adversaries; mechanism is theoretically sound
  (operator's `g_{β,γ}(r,v) → (1+γ)·max(r,v)` for `β → +∞` makes
  `d_eff → 1+γ > 1` whenever `v > r`). Spec §5.4 reframed: AC-Trap
  is a **falsifiability cell** for the diagnostic's scope; new §13.10
  documents the negative-result honesty requirements for AC-Trap
  reporting in M6/M7. Counter-intuitive finding logged at
  `results/adaptive_beta/tab_six_games/counter_intuitive_findings.md`.
  Each fold-in marked inline with `<!-- patch-2026-05-01-v7 -->`.
- **2026-05-01 v6 — HALT 5 resolution: M6 wave 1 OQ1+OQ2+OQ3+OQ4
  closures.** OQ3 (AC-Trap adversary): wired to
  `finite_memory_regret_matching(memory_m=20)` in both
  `configs/dev.yaml` and `configs/stage1_beta_sweep.yaml` — regret
  matching tests the payoff-dominance claim directly via
  counterfactual regret over the (Stag, Hare) payoff structure;
  single-parameter opponent avoids a hyperparameter pilot. OQ2
  (aggregator schema parity): extended
  `PHASE_VIII_EXPECTED_COLUMNS` and `LONG_CSV_COLUMNS` with
  `beta_raw`, `beta_used`, `effective_discount_mean`,
  `goal_reaches`; updated `test_aggregate_schema_parity.py`
  (column count 49 → 53). OQ1 (Stage 1 run count): corrected from
  ~4,340 → ~1,820 across §6.4 / §10.2 / §22 with arithmetic-error
  footnote; v2 patch §1.5 / §11.5 had spuriously multiplied
  subcase counts by 10 (`4 × 7 × 10 = 2,800` instead of
  `4 × 7 × 10 = 280` etc). OQ4 (RR/SO/PG stationary opponent
  probabilities): confirmed runner's defaults — RR `[0.7, 0.3]`
  (load-bearing for the convention-learning mechanism), SO/PG
  uniform. Each fold-in marked inline with
  `<!-- patch-2026-05-01-v6 -->`.
- **2026-05-01 v5b — HALT 4 resolution: replace Cohen's d guard with
  relative-gap floor for noiseless DC-Long50 testbed.** v5 metric +
  prediction unchanged; only test instrumentation. The DC-Long50
  advance-only chain is fully deterministic (Discrete(1) action space
  + ε=0 + PassiveOpponent + deterministic transitions), so per-seed
  AUCs are bit-identical and Cohen's d degenerates into an
  inter-method-gap-ratio test that was never the intended invariant.
  Smoke now asserts each gap ≥ 100 absolute floor AND
  `gap_small ≥ 0.10 · gap_large` relative floor. Records the
  orchestrator-self-correction lesson on patch directive scope:
  "Do NOT auto-patch beyond vN" should be scoped to design, not test
  instrumentation (test-instrumentation tweaks are MINOR under
  addendum §13). For Phase VIII going forward, T11 in M6 wave 6 uses
  the gap-based guard for deterministic cells (DC-Short10 /
  DC-Medium20 / DC-Long50) and Cohen's d ≥ 0.3 for stochastic cells
  (DC-Branching20 + all M7/M9/M10 ablations with ε > 0). Each fold-in
  marked inline with `<!-- patch-2026-05-01-v5b -->`.
