# Phase VIII Spec Patches — 2026-05-01 (pre-M6 amendment)

**Status:** AUTHORITATIVE amendment to
`docs/specs/phase_VIII_tab_six_games.md`. Apply BEFORE M6 dispatch in
session 3.
**Source:** ML-researcher re-evaluation of Phase VIII against
TAB-mechanism falsifiability criteria (see end of file for the source
critique excerpt).
**Authority:** Inserted between the spec and the overnight addendum in
the authority order. Where this patch contradicts the spec, the patch
wins.
**Lifecycle:** This file's amendments are folded into
`docs/specs/phase_VIII_tab_six_games.md` as part of M6 wave 0 (see §0
below). After fold-in, this file is moved to
`tasks/archive/phase_VIII_spec_patches_2026-05-01_applied.md`.

---

## 0. Apply protocol (M6 wave 0)

Before any M6 wave 1 dispatch, the orchestrator MUST:

1. Read this file in full.
2. Edit `docs/specs/phase_VIII_tab_six_games.md` to fold in the
   amendments from §1, §3, §4, §5, §6, §7, §11 below (§2 is
   reserved/deferred). Mark each fold-in with an inline comment of
   the form
   `<!-- patch-2026-05-01 §N -->` so the provenance is traceable.
3. Bump the spec's §23 changelog with a new entry:
   `2026-05-01 — Pre-M6 amendment per tasks/phase_VIII_spec_patches_2026-05-01.md`.
4. Append the new tasks below to the relevant existing milestone
   blocks in `tasks/todo.md`:

   - M2 reopen: implement RR-Sparse subcase (§1) — `[N][env]` env-builder
   - M2 reopen: implement delayed_chain game + PassiveOpponent (§11) — `[N][env]` env-builder
   - M4 reopen: implement FP/RM agent wrappers (§3) — `[N][algo]` algo-implementer
   - M6 wave 0: spec fold-in (this protocol) — `[infra]` planner
   - M6 wave 1: AC-Trap pre-sweep (§5) — `[X][ablation]` experiment-runner

5. Run the existing test suite under `.venv/bin/python` to confirm no
   regression from the fold-in (spec text changes shouldn't break
   tests, but verify).
6. Commit:

       git add docs/specs/phase_VIII_tab_six_games.md tasks/todo.md \
               tasks/phase_VIII_spec_patches_2026-05-01.md
       git commit -m "phase-VIII(M6 wave 0): fold pre-M6 spec amendment per researcher critique"

7. Recompute SHA-256 of harness + addendum + this patch file.
   Update `phase_VIII_autonomous_checkpoint.json` schema to include
   `spec_patch_2026_05_01_sha256` field; record the SHA. This locks
   the patch against silent drift.
8. THEN proceed with M2 reopen → M4 reopen → M6 waves per session 3
   resumption plan.

If any patch in §1, §3, §4, §5, §6, §7, or §11 cannot be applied
cleanly (e.g., a referenced file no longer exists or a symbol has
been renamed), HALT with a patch_apply_failed memo. Do NOT
partial-apply.

---

## 1. New subcase: Rules of the Road — Sparse Terminal (RR-Sparse)

**Patch target:** `experiments/adaptive_beta/strategic_games/games/rules_of_road.py`
**Spec target:** `docs/specs/phase_VIII_tab_six_games.md` §5.3

### 1.1 Motivation

TAB's most defensible advantage over vanilla Q is on sparse-reward
problems: positive β concentrates value on rare positive signals via
the optimistic propagation
$g_{\beta,\gamma}(r,v) \to \max(r, \gamma v)$ as $\beta \to +\infty$.
The current RR subcases all use dense per-step payoffs $(+c, -m)$,
which masks this advantage. RR-Sparse exposes it.

### 1.2 Specification

```text
Subcase id:        RR-Sparse
Action set:        L, R  (unchanged)
Horizon:           H = 20  (longer than dense subcases to stress
                            credit assignment over multiple steps)
Per-step reward:   0  (NO per-step shaping)
Terminal reward:   +c  if last action was coordinated
                   -m  if last action was miscoordinated
                   c = 1.0, m = 0.5  (paper config; tuneable in yaml)
Hidden regime:     none (RR-Sparse is regime-stationary)
Opponent:          parameterized — accept any of the existing RR
                   opponent set (StationaryConvention default;
                   ConventionSwitch for stress)
State encoding:    (timestep, last_opponent_action) — same as other RR subcases
```

### 1.3 Theoretical prediction

Under sparse terminal reward and a coordinating opponent:

- $r=0, v>0$ at every non-terminal step ⇒ $r-v < 0$ ⇒ alignment
  condition holds for $\beta < 0$ but TAB's optimistic propagation
  with $\beta > 0$ still helps because the terminal $r=+c$ gets
  amplified backward via $\max$-like aggregation.
- Specifically, $g_{+\beta,\gamma}$ approaches $\gamma \cdot v$ for
  $r=0, v>0$ but the BACKWARD propagation from terminal $+c$
  dominates: at the penultimate step,
  $g_{+\beta,\gamma}(0, c) \to \gamma c$ as $\beta \to +\infty$,
  matching the classical bootstrap. The differentiation comes from
  finite β: the rate of convergence to $\gamma c$ is faster for
  $+\beta$ than for $\beta=0$ on initialization with optimistic Q.
- **Falsifiable claim:** AUC(fixed_+β) > AUC(β=0) on RR-Sparse with
  optimistic Q-init, paired-seed bootstrap 95% CI strictly positive.

### 1.4 Implementation notes

- Reuse `RulesOfTheRoadGame` parameterization; add a flag
  `sparse_terminal: bool = False` to the constructor. When `True`,
  per-step payoffs are zeroed and only the final-step payoff fires.
- Register as `"rules_of_road_sparse"` in
  `GAME_REGISTRY`.
- Add to spec §5.3 subcase list and §M2 todo as a delta task (M2 is
  reopened by this patch — see §0 above).
- Add unit test:
  `tests/adaptive_beta/strategic_games/test_rules_of_road_sparse.py`
  asserting (i) per-step reward = 0 for all t < H, (ii) terminal
  reward in {+c, -m}, (iii) horizon termination at H = 20.

### 1.5 Inclusion in M6 sweep

Add `RR-Sparse × StationaryConvention` to the M6 game-subcase list
as a promoted subcase. Use the same β grid and seed count.

---

## 2. (RESERVED — P0.2 high-magnitude payoff for safety claim,
       DEFERRED to v2 round per researcher decision 2026-05-01)

The high-magnitude payoff variant for the safety/clipping
demonstration is deferred to a post-M9 v2 round of Phase VIII. The
demonstration requires deliberate payoff design (chosen so unclipped
β reliably diverges) and a pilot run to verify the divergence pattern;
folding it into the live overnight run risks adding a cell that does
not actually exhibit the divergence we need. Re-evaluate after M9
results show whether the existing β_clipping_frequency metric
provides any signal on bounded payoffs.

This section is preserved as a placeholder so §3+ numbering is stable
across patch revisions.

---

## 3. Promote FP/RM agents from M11 (optional) to M7 (mandatory baseline)

**Patch target:** new `experiments/adaptive_beta/baselines.py` additions
**Spec target:** `docs/specs/phase_VIII_tab_six_games.md` §6.3

### 3.1 Motivation

Phase VIII's M7 currently benchmarks fixed TAB against vanilla RL
baselines (`restart_Q_learning`, `sliding_window_Q_learning`,
`tuned_epsilon_greedy_Q_learning`). For a strategic-learning paper,
the natural baselines are also strategic-learning agents:
fictitious play and regret matching as *agents*, not opponents.
Both are already implemented as opponent classes; promoting them to
agent baselines is a wrapper over existing code.

The "weak baselines" rejection vector is the single highest-
probability reviewer attack on this suite.

### 3.2 Specification

Add to `experiments/adaptive_beta/baselines.py`:

```text
class RegretMatchingAgent:
    """Strategic-learning baseline: regret matching as the learning
    agent (no Q-learning).

    Wraps the existing strategic_games/adversaries/regret_matching.py
    in the agent interface (begin_episode / step / end_episode /
    select_action) used by AdaptiveBetaQAgent so it can be slotted
    into the Phase VIII runner without runner changes.
    """

class SmoothedFictitiousPlayAgent:
    """Strategic-learning baseline: smoothed fictitious play as the
    learning agent.

    Wraps strategic_games/adversaries/smoothed_fictitious_play.py.
    """
```

Both agents:

- Implement the same interface as `AdaptiveBetaQAgent`
- Carry a beta_schedule of `ZeroBetaSchedule` that is unused
  (β not applicable; included so logger does not error on missing
  field)
- Emit per-episode metrics: `return`, `length`, `epsilon` (always 0),
  `bellman_residual` (placeholder = 0; document in metric notes that
  the field is undefined for non-Q-learning agents and aggregator
  must drop it from comparisons), `nan_count`,
  `divergence_event` (always 0)
- Do NOT emit operator-mechanism metrics (`alignment_rate`,
  `mean_d_eff`, etc.) — these are TAB-specific. Aggregator
  must handle the missing fields gracefully (lessons.md
  candidate: aggregator field-presence check).

### 3.3 M7 method list update

Replace the M7 method list in spec §M7 with:

```text
vanilla_beta_0
best_fixed_positive_TAB         (Stage 1 dev grid → main hold-out)
best_fixed_negative_TAB         (Stage 1 dev grid → main hold-out)
best_fixed_beta_grid            (reporting aggregate)
restart_Q_learning              (existing)
sliding_window_Q_learning       (existing)
tuned_epsilon_greedy_Q_learning (existing)
regret_matching_agent           (NEW — promoted from M11)
smoothed_fictitious_play_agent  (NEW — promoted from M11)
```

### 3.4 Tests

Add to `tests/adaptive_beta/tab_six_games/test_baselines.py` (already
exists per M4 work):

```text
test_regret_matching_agent_interface_parity
test_regret_matching_agent_strategy_dynamics  (regret update fires)
test_smoothed_fp_agent_interface_parity
test_smoothed_fp_agent_logit_response_stable  (numerical stability)
test_strategic_agent_aggregator_handles_missing_fields
```

### 3.5 Stat protocol

Spec §12 statistical protocol must add to its primary-comparison set:

```text
paired difference vs regret_matching_agent
paired difference vs smoothed_fictitious_play_agent
```

These join `paired difference vs vanilla β=0`,
`paired difference vs best fixed β`, and
`paired difference vs sliding-window Q`.

---

## 4. Extend β-grid for headline figures only (figures-only sub-pass)

**Patch target:** M6 wave 5 — figures sub-pass within Stage 1
**Spec target:** `docs/specs/phase_VIII_tab_six_games.md` §M6, §12.1

### 4.1 Motivation

The β-vs-AUC and β-vs-contraction curves are M6's headline
operator-diagnostic figures. Their *shape near β=0* is the operator-
mechanism story (small-β regime is where TAB's classical limit is
visible and where the contraction reward
$M_e(\beta) = \log R_e - \log R_{e+1}$ is most sensitive). The
current β grid {-2, -1, -0.5, 0, +0.5, +1, +2} has a 0 → ±0.5 gap;
the curve interpolation across that gap is visually unfaithful.

### 4.2 Patch

In M6 wave 5 (figures sub-pass), add a SUPPLEMENTARY β grid
{-0.25, -0.1, +0.1, +0.25} on:

- ONE G_+ subcase (preferred: `RR-StationaryConvention` if
  Stage 1 confirms it as G_+; else `AC-FictitiousPlay`)
- ONE G_- subcase (preferred: `SH-FiniteMemoryRegret` per Phase VII
  Stage B2)

Configuration:

```text
β_supplementary:   {-0.25, -0.1, +0.1, +0.25}
seeds:             5  (lighter than main pass; figures-only)
episodes:          10,000  (same as main pass for shape comparability)
games:             1 G_+ subcase + 1 G_- subcase
total runs:        4 β × 5 seeds × 2 subcases = 40 runs
estimated wall:    < 30 minutes
```

The supplementary β values are NOT included in the main statistical
sweep. They appear ONLY in:

- `figures/beta_vs_auc.pdf` (with a visual marker indicating
  "figure-only points" so reviewers know they were not part of the
  primary statistical comparison)
- `figures/beta_vs_contraction.pdf`

### 4.3 Tests

Add to `tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py`:

```text
test_supplementary_beta_grid_separated_in_aggregate  (verify
  the supplementary points are tagged in the aggregator output and
  excluded from main statistical tables)
```

---

## 5. AC-Trap pre-sweep before M6 main pass

**Patch target:** M6 wave 1 (configuration phase)
**Spec target:** none (procedural; record in spec §10/M6 milestone notes)

### 5.1 Motivation

The asymmetric coordination AC-Trap subcase is the single cleanest
cell for the "fixed positive TAB selects payoff-dominant equilibria
where vanilla Q-learning is risk-dominated" claim. If this cell
does not produce the expected effect, the strongest single-cell
result for the paper evaporates — and we should know that before
sinking 2-4h of M6 main-pass compute.

### 5.2 Patch

Insert a pre-sweep into M6 wave 1 (immediately after runner +
configs are built, before Stage A dev pass dispatches):

```text
M6 wave 1.5 — AC-Trap β-pre-sweep:
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

### 5.3 Disposition

- If pre-sweep matches expectation: log success; proceed normally
  with M6 wave 2 dev pass.
- If pre-sweep contradicts expectation (e.g., +β does NOT beat
  vanilla on AC-Trap):
  - Dispatch a focused Codex bug-hunt review on the pre-sweep
    results (per addendum §3.1, treat as if T1/T3 fired).
  - Disposition per Codex verdict:
    - GENUINE FINDING: log to `counter_intuitive_findings.md`,
      flag for paper attention, proceed with M6 normally (the
      finding becomes a paper result — payoff-dominance claim
      gets an asterisk).
    - BUG: auto-fix loop per addendum §4.2; re-run pre-sweep
      after fix.
    - Inconclusive: HALT with pre_sweep_inconclusive memo.

---

## 6. Reposition Matching Pennies as null cell in spec §5.1

**Patch target:** `docs/specs/phase_VIII_tab_six_games.md` §5.1

### 6.1 Motivation

Matching Pennies has expected $\bar A \to 0$ at the
$(\tfrac12,\tfrac12)$ mixed Nash, so β-effects on the return axis
are second-order. The current spec text presents MP as a
"benchmark" without acknowledging this. A reviewer encountering a
null result on MP can use the spec's framing to argue the suite
fails to support TAB; pre-empting this requires explicit framing.

### 6.2 Patch

Replace spec §5.1 opening paragraph:

> Role: minimal adversarial zero-sum benchmark.

with:

> Role: **null-cell sanity check**. Matching Pennies has expected
> advantage $\bar A \to 0$ at the $(\tfrac12,\tfrac12)$ mixed Nash,
> so β-induced effects on AUC are second-order on the return axis.
> A near-zero result on MP is the *predicted* outcome and confirms
> the operator's classical-limit fidelity rather than refuting the
> TAB story. Mechanism-level metrics (`alignment_rate`,
> `frac_d_eff_below_gamma`) ARE expected to differentiate across
> β even when AUC does not, and that differentiation IS evidence of
> the TAB mechanism. MP-Stationary serves as the bit-identity check
> against β=0; MP-FiniteMemoryBR / MP-RegretMatching /
> MP-HypothesisTesting test second-order β-effects under
> nonstationary opponents but should NOT be expected to produce
> headline AUC differences. The honesty norm of spec §13 (negative
> results reported faithfully) covers this case.

Add to spec §13 (Success/Failure criteria) a new subsection:

```text
13.4 Null-cell expectation

Matching Pennies subcases are expected to produce null AUC results.
This is not a failure mode. A non-null result on MP — particularly
a positive AUC differential for fixed +β or fixed -β — would itself
warrant scrutiny (suspect: seed contamination or evaluation-window
bias).
```

---

## 7. Potential-game lemma (one-line proof) in potential.py docstring

**Patch target:** `experiments/adaptive_beta/strategic_games/games/potential.py`

### 7.1 Motivation

The positive-control prediction "fixed +β monotonically accelerates
convergence on potential games" is currently a conjecture. A
reviewer can ask whether the prediction has theoretical grounding.
A one-line lemma in the module docstring fixes this.

### 7.2 Patch

Add to `potential.py` module docstring (top of file, in the
`"""..."""` block):

```python
"""Potential / weakly acyclic game environments.

...

Theoretical anchor (positive-β monotonicity on potential games):

    Let Φ : A → R be the exact potential of the stage game, so
    u_i(a_i', a_{-i}) - u_i(a_i, a_{-i}) = Φ(a_i', a_{-i}) - Φ(a_i, a_{-i})
    for all i, a_i, a_i', a_{-i}.

    Under the better-reply dynamics induced by a tabular Q-learning
    agent with optimistic initialization Q_0(s,a) ≥ V*(s) for all
    (s,a), the TAB target

        T_β Q(s,a) = (1+γ)/β · [logaddexp(β r, β γ V(s')) − log(1+γ)]

    is monotonically increasing in β at fixed (r, V(s'), γ) when
    r ≥ V(s'), and decreasing when r < V(s'). On potential games
    with optimistic init, every better-reply move increases Φ along
    the dynamics, so the realized advantage A(s,a) := r − V(s)
    has positive sign in expectation under any better-reply policy.
    Therefore positive β tightens the contraction toward V*
    monotonically (alignment condition d_β,γ ≤ γ holds), proving
    +β cannot slow convergence relative to β=0 on potential games
    with this initialization.

    Negative β VIOLATES the alignment condition under positive
    expected advantage and is therefore predicted to slow or destabilize
    convergence on potential games. This gives the falsifiable
    sign-prediction (PG-CoordinationPotential, PG-Congestion):

        AUC(+β) ≥ AUC(0) ≥ AUC(-β)

    with strict inequality expected on the upper bound under
    optimistic init.

This is the positive-control prediction tested by Phase VIII
M6 Stage 1 sweep. Failure of this prediction would constitute
strong evidence of either (a) implementation bug, (b) violation of
the alignment-condition assumption (e.g., misspecified Q init,
non-better-reply opponent), or (c) a flaw in the theoretical
derivation above.

...
"""
```

Add corresponding test:
`tests/adaptive_beta/strategic_games/test_potential_lemma_prediction.py`
that runs a smoke version of PG-CoordinationPotential under
{-1, 0, +1} β with 3 seeds × 500 episodes and asserts the AUC
ordering. Failure of this smoke test does NOT block M6 — it goes
to bug-hunt review per addendum §3.1.

---

## 8. Items NOT applied in this patch (deferred / declined)

For provenance:

| Item | Source ref | Disposition |
| ---- | ---------- | ----------- |
| P0.2 high-magnitude payoff variant for safety claim | §4D | DEFERRED to v2 post-M9 round (requires deliberate divergence-tuned payoff design + pilot) |
| P1.5 add long-horizon delayed-reward env | §4A | **APPLIED — see §11** (revised 2026-05-01 v2). Paper title "Selective Temporal Credit Assignment" requires in-suite long-horizon evidence; cross-validates Phase VII-A `delayed_chain` artifacts in the unified Phase VIII framework. `hazard_gridworld` remains declined (single delayed-reward env covers the temporal axis). |
| §F mechanism-diagnostics stratification by (s,a) | §4F | POSTPONED — agent attempts at M6 wave 7 if mean_d_eff distribution looks coarse; pinning upfront is premature optimization |

These are not silent drops — they are explicit decisions recorded
here so a reviewer reading the file knows what was considered and
why it was excluded.

---

## 9. Source critique excerpt (provenance)

The following is the user-supplied analytical critique that motivated
this patch. Preserved verbatim for traceability.

```text
[Reframing — faithfulness to Young is irrelevant. The criterion is
whether each game gives a clean, falsifiable, mechanism-anchored
test of when fixed/safe/adaptive TAB beats β=0 ...]

Per-game suitability:
- Matching Pennies: weak demonstrator, useful only as null cell
- Shapley 3×3: strong G_- demonstrator (Phase VII-B confirmed)
- Rules of the Road: strong G_+ demonstrator + recovery test
- Asymmetric Coordination: strong G_+ demonstrator with
  payoff-dominance leverage; AC-Trap is the cleanest single cell
- Soda / Uncertain Game: strongest adaptive-β cell
- Potential / Weakly Acyclic: necessary positive control

Structural gaps:
A. Long-horizon temporal credit assignment is not tested
B. Sparse-reward / hard-exploration is not tested
C. Strategic-learning agent baselines are deferred (highest-
   probability rejection vector)
D. Safety/catastrophe demonstrator is weak
E. β-grid resolution near zero is too coarse
F. Mechanism diagnostics may have low resolution

Recommendations:
P0: sparse-reward subcase; high-magnitude payoff variant; promote
    FP/RM agents to M7
P1: extend β-grid for figures only; delayed-reward env or scope
    retitle; potential-game lemma
P2: reposition MP as null cell; AC-Trap pre-sweep
```

Full critique is preserved in the conversation log for session 3
of the overnight run.

---

## 11. New game: Long-Horizon Delayed-Reward Chain (delayed_chain)

**Patch target:** new `experiments/adaptive_beta/strategic_games/games/delayed_chain.py`
            and new `experiments/adaptive_beta/strategic_games/adversaries/passive.py`
**Spec target:** `docs/specs/phase_VIII_tab_six_games.md` §5 (insert
new §5.7 after Potential / Weakly Acyclic Game)

### 11.1 Motivation

The paper title is **"Selective Temporal Credit Assignment"**. The
matrix-game suite as designed has horizons typically H ≤ 10 — too
short to credibly demonstrate long-horizon temporal credit
assignment. Phase VII-A artifacts (`delayed_chain`,
`hazard_gridworld`) addressed this but were dropped from Phase VIII.
Reintroducing `delayed_chain` as a Phase VIII cell:

- Anchors the paper title in directly testable terms within the
  Phase VIII suite (no out-of-suite cross-citation required).
- Tests TAB's value-propagation behavior on horizons where temporal
  credit actually matters (H ∈ {20, 50}).
- Validates Phase VII-A results in the unified Phase VIII framework.
- Pre-empts the most direct reviewer attack: "where is the
  credit-assignment story in the experiments?"

### 11.2 Specification

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
                   so the env fits the 2-player MatrixGameEnv
                   wrapper without the opponent affecting payoffs)
State encoding:    integer state-index ∈ [0, L]; state L is goal
                   terminal; (0..L) for branching subcases additionally
                   index trap-terminal positions
Canonical sign:    "+"   (positive-β specialist regime; +β should
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

### 11.3 Theoretical prediction

Under optimistic Q-init $Q_0(s,a) \ge V^*(s)$:

- At every non-terminal step: $r_t = 0, V(s_{t+1}) > 0$
  ⇒ $r-V<0$ ⇒ alignment condition holds for $\beta<0$, but TAB +β
  still helps via amplified backward-from-terminal propagation.
- The optimistic propagation $g_{+\beta,\gamma}(0, V(s'))$ approaches
  $\gamma V(s')$ as $\beta \to +\infty$ FROM BELOW for finite β > 0,
  meaning terminal reward propagates at the discounted rate but
  with TIGHTER concentration of value mass per backward step. The
  classical β=0 propagation is also at rate γ but with less
  concentration in the optimistic regime, so empirically AUC at
  any fixed-budget episode count satisfies AUC(+β) > AUC(0).
- Negative β VIOLATES the alignment condition under the negative
  realized advantage, which on a positive-only-terminal chain
  predicts strict slowdown: AUC(0) > AUC(-β).

**Falsifiable predictions:**

```text
P-Sign:        AUC(+β) > AUC(0) > AUC(-β) on DC-Short10, DC-Medium20,
               DC-Long50 (paired-bootstrap 95% CI strictly ordered)
P-Scaling:     |AUC(+β) - AUC(0)| grows monotonically with chain
               length: DC-Short10 < DC-Medium20 < DC-Long50
P-Branch:      On DC-Branching20, AUC(+β) > AUC(0) survives the
               trap-arm exploration penalty; effect size may shrink
               vs DC-Medium20 but sign holds
P-VII-Parity:  AUC(0) on DC-Long50 within paired-bootstrap 95% CI of
               Phase VII-A `delayed_chain` reference at the matched
               L=50 setting (cross-validation with the prior phase)
```

Failure of P-Sign on any of the advance-only subcases is a
**bug-hunt T-class trigger** (see §11.7 below) and dispatches a
focused Codex review.

### 11.4 Implementation

```text
1. Create experiments/adaptive_beta/strategic_games/games/delayed_chain.py:
   - class DelayedChainGame inheriting from the BaseGame interface
     used by MatrixGameEnv
   - State space: integer ∈ [0, L] (encoded as Discrete(L+1) for
     advance-only subcases; Discrete(L+1+5) for DC-Branching20 where
     trap states are appended)
   - Action space: Discrete(1) for advance-only; Discrete(2) for
     branching (action 0 = advance, action 1 = branch_wrong)
   - reward(state, action) returns 0 for non-terminal transitions,
     ±1 at terminal arrivals
   - game_info() returns metadata per spec §5 contract:
       game = "delayed_chain"
       subcase ∈ {"DC-Short10","DC-Medium20","DC-Long50","DC-Branching20"}
       horizon = L
       canonical_sign = "+"
       regime = None  (regime-stationary)
       action_labels = ["advance"] or ["advance", "branch_wrong"]
   - Register all 4 subcases in GAME_REGISTRY via register_game().

2. Create experiments/adaptive_beta/strategic_games/adversaries/passive.py:
   - class PassiveOpponent(StrategicAdversary)
   - reset(seed): no-op
   - act(history, agent_action): always returns action 0 (the only
     valid opponent action; opponent payoff is unused since
     DelayedChainGame's payoff function only reads agent_action)
   - observe(...): no-op
   - info() returns:
       adversary_type = "passive"
       phase = "stationary"
       memory_m = 0
       inertia_lambda = 0.0
       temperature = 0.0
       model_rejected = False
       search_phase = "none"
       hypothesis_id = None
       policy_entropy = 0.0
   - Register as "passive" in ADVERSARY_REGISTRY.

3. Tests:
   tests/adaptive_beta/strategic_games/test_delayed_chain.py
     - test_chain_advance_reaches_terminal_at_L
     - test_chain_no_intermediate_reward       (r_t = 0 for t < H)
     - test_branching_wrong_terminal_negative  (DC-Branching20: -1 reward)
     - test_horizon_matches_subcase            (H == L for advance-only)
     - test_canonical_sign_metadata            (game_info["canonical_sign"] == "+")
     - test_state_encoder_shape
     - test_seed_determinism

   tests/adaptive_beta/strategic_games/test_passive_opponent.py
     - test_passive_opponent_no_op             (always returns action 0)
     - test_passive_opponent_info_contract     (all required info keys)
     - test_passive_opponent_seed_invariance   (no RNG state to seed)

   tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py
     - test_smoke_DC_Long50_AUC_ordering
       Run β ∈ {-1, 0, +1} on DC-Long50 with 3 seeds × 1k episodes
       under AdaptiveBetaQAgent + PassiveOpponent. Assert
       AUC(+1) > AUC(0) > AUC(-1) with at least Cohen's d > 0.3 on
       both gaps. Failure = T1/T3 bug-hunt trigger; do NOT auto-fix
       in M2 reopen (lessons.md candidate: smoke-prediction failures
       need researcher attention rather than implementer auto-fix).
       MARK as @pytest.mark.smoke so it can be selectively skipped
       under tight CI budget while remaining part of the
       milestone-close test sweep.
```

### 11.5 Inclusion in M6 sweep

Add to M6 game-subcase list (extends the 6 games to 7):

```text
delayed_chain × DC-Short10        (control: short-horizon credit)
delayed_chain × DC-Medium20       (medium-horizon)
delayed_chain × DC-Long50         (long-horizon, paper headline)
delayed_chain × DC-Branching20    (credit + exploration)
```

This grows the M6 main pass by 4 subcases × 7 β × 10 seeds × 10k
episodes = 2,800 additional runs. At ~1k episodes/sec/seed for
tabular Q on chain MDPs (which are cheaper per-step than matrix
games due to single-action choice), additional wall-clock ≈ 30-60
minutes serial; with the existing parallel-seed infrastructure,
expect ~5-15 minutes additional wall-clock.

Total M6 main pass after this patch: ~1,260 (original) + ~280 from
RR-Sparse (§1) + ~2,800 from delayed_chain = ~4,340 runs.

### 11.6 M7 baseline inclusion

`delayed_chain × DC-Long50` MUST be included in the M7 baseline
comparison (spec §M7 promoted-subcases list). The baseline lineup
from §3 (vanilla, restart-Q, sliding-window-Q, tuned-ε-Q,
regret_matching_agent, smoothed_fictitious_play_agent) is
particularly relevant here because:

- Restart-Q and sliding-window-Q have known difficulty with
  long-horizon credit (their forgetting mechanism interacts poorly
  with deferred rewards).
- regret_matching_agent and smoothed_fictitious_play_agent have NO
  value-function — they cannot solve delayed_chain at all and
  should produce AUC near random-policy baseline. This is a
  **diagnostic feature, not a flaw**: it explicitly demonstrates
  that strategic-learning agents without value bootstrapping fail
  on temporal credit, justifying the TAB approach.

The aggregator must therefore tolerate "agent cannot solve task"
results — agg should report mean ± std and let the table builder
decide whether to format as "—" (failure) or numerical value.

### 11.7 Bug-hunt trigger extension

Add a delayed-chain-specific suspicious-result trigger to the
addendum §3.1 detector list (this is an extension of the addendum,
not a modification — it's listed here for the orchestrator to
include in M6 wave 5 detector code):

```text
T11 — sign-prediction failure on delayed_chain (NEW):
       AUC(+β) ≤ AUC(0) on any advance-only delayed_chain subcase
       (DC-Short10, DC-Medium20, DC-Long50). This is a hard
       theoretical prediction failure; if it fires, dispatch a
       focused Codex bug-hunt review with the prompt extended to
       include: "this contradicts the alignment-condition prediction
       for positive expected advantage on delayed-reward chains.
       Investigate whether (a) Q-init is not sufficiently optimistic,
       (b) episode horizon is binding before terminal reward
       propagates back, (c) ε-greedy schedule prevents convergence,
       (d) implementation of the chain transition is off-by-one, or
       (e) the prediction itself is theoretically misguided."
```

T11 is a sign-flip trigger and should fire with high priority — if
the headline temporal-credit-assignment cell does not produce the
predicted ordering, that is paper-critical and warrants halting M6
for human review per addendum §6 BLOCKER semantics.

### 11.8 Phase VII-A cross-reference

If `results/adaptive_beta/strategic/raw/.../phase_VII_A_delayed_chain*`
artifacts survive on disk (they were `.npz` files which were
filter-repo expunged from git history but may persist in the working
tree), the M6 wave 7 aggregator reads them as historical-baseline
reference and includes the comparison in M6_summary.md (read-only
narrative cross-reference; same caveat as M8 Phase VII Stage B2
cross-reference: unpaired across phases).

If artifacts do not exist:

```text
- M6_summary.md notes "no Phase VII-A reference artifacts available
  on disk; Phase VIII delayed_chain results constitute the in-tree
  validation of the temporal-credit-assignment claim"
- M12 final-recommendation memo recommends paper text reference
  Phase VIII delayed_chain results directly (NOT Phase VII-A
  artifacts that no longer exist in tree)
- This is NOT a halt condition; it's a documentation choice.
```

---

## 12. Changelog

- 2026-05-01 v1: initial pre-M6 spec amendment. Applies
  RR-Sparse subcase, FP/RM agent promotion to M7,
  figures-only β-grid extension, AC-Trap pre-sweep, MP null-cell
  repositioning, and potential-game lemma docstring. Defers
  high-magnitude payoff variant. Declined long-horizon delayed-
  reward env (paper-text fix instead).
- 2026-05-01 v2: REVISES v1 to APPLY long-horizon delayed-reward
  env per researcher decision. Adds §11 (delayed_chain game with
  4 subcases DC-Short10/Medium20/Long50/Branching20 + PassiveOpponent
  + bug-hunt trigger T11). Updates §0 apply protocol to include
  delayed_chain in M2 reopen. Updates §8 disposition table for
  P1.5 from DECLINED to APPLIED. M6 main pass grows by ~2,800
  runs (~5-15 min additional wall-clock at parallel-seed scale).
  hazard_gridworld remains DECLINED (single delayed-reward env
  is sufficient for the temporal axis).
