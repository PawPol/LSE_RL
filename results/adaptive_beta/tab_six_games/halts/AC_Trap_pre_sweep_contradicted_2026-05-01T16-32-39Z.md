# HALT 6 — AC-Trap pre-sweep prediction CONTRADICTED with reverse direction

- **Halt UTC**: 2026-05-01T16:32:39Z
- **Milestone**: M6 wave 1.5 (AC-Trap β-pre-sweep)
- **Trigger class**: T1 + T3 (spec §10.2 patch §5.2/§5.3 — pre-sweep
  prediction reversed; headline single-cell paper claim at risk)
- **Severity**: paper-critical / BLOCKER per patch §5.2 ("if it does
  not produce the expected effect, the strongest single-cell result
  for the paper evaporates")
- **HEAD at halt**: `1d88e769` (M6 wave 1 + v6 commit)
- **Configuration**: `pre_sweep_AC_Trap.yaml`
  - 1 cell (asymmetric_coordination + AC-Trap)
  - β ∈ {-1, 0, +1} (3-arm probe)
  - 3 seeds × 200 episodes = 9 total runs
  - opponent: `finite_memory_regret_matching(memory_m=20)` (HALT 5
    OQ3 wiring)
  - q_init = 0.0 (the runner's default; matches dev.yaml /
    stage1_beta_sweep.yaml)
  - ε: linear decay 1.0 → 0.05 over 100 episodes

## Empirical result

| schedule | β | AUC (return × 200 ep) per seed | mean | std (ddof=1) |
| --- | ---: | --- | ---: | ---: |
| ZeroBeta (vanilla) | 0  | [10317.50, 10323.50, 10370.50] | 10337.17 | 29.02 |
| FixedNeg           | -1 | [10380.50, 10381.50, 10450.50] | 10404.17 | 40.13 |
| FixedPos           | +1 | [8890.50, 9611.00, 8312.00]    |  8937.83 | 650.79 |

Mean per-episode return:

| schedule | per-seed | mean |
| --- | --- | ---: |
| vanilla    | [51.83, 51.87, 52.13] | 51.94 |
| FixedNeg   | [52.15, 52.16, 52.53] | 52.28 |
| FixedPos   | [44.67, 48.30, 41.80] | 44.92 |

Last-episode return (post-ε-decay):

| schedule | per-seed |
| --- | --- |
| vanilla    | [57.0, 57.0, 60.0] |
| FixedNeg   | [57.0, 57.0, 60.0] |
| FixedPos   | [45.0, 54.0, 45.0] |

## Cohen's d vs vanilla (target > 0.5)

- `d(+1 vs vanilla) = -3.038`  (β=+1 dramatically WORSE; opposite
  sign of the prediction; |d| ≫ 0.5)
- `d(-1 vs vanilla) = +1.913`  (β=-1 materially BETTER; |d| ≫ 0.5)

## Comparison to spec prediction

Spec §10.2 wave 1.5 / patch §5.2:

> expected: AUC(+1) > AUC(0) > AUC(-1) with effect size at least
>           Cohen's d > 0.5 vs vanilla

Empirical:
- AUC(+1) = 8937.83 < AUC(0) = 10337.17 < AUC(-1) = 10404.17
- Direction is **REVERSED**, magnitude is large (|d| > 1.9 on both
  gaps), and intra-method variance is small for vanilla / FixedNeg
  but very large for FixedPos (std=651 ≈ 7% of the mean).

This is a paper-critical T1/T3 trigger per patch §5.3.

## Possible explanations (ranked by orchestrator's prior)

### (1) METHODOLOGICAL — q_init = 0 vs the spec's implicit assumption

The spec's canonical-sign argument for AC-Trap (§5.4 / §22.3) reads:
"Under optimistic-β direction propagates 'value of cooperation'
forward — adaptive β should push positive WHEN the agent nudges
into the payoff-dominant equilibrium". The "WHEN the agent nudges
into the payoff-dominant equilibrium" clause requires that the
initial bootstrap already reflect cooperation; with `q_init = 0.0`,
the agent starts at neutral Q values and the ε-greedy + regret-
matching opponent dynamics tend to settle into the risk-dominant
(Hare, Hare) basin first. Under that history, +β AMPLIFIES the
already-converged Hare bootstrap and suppresses Stag exploration —
which would explain the large negative effect for β=+1.

Test: re-run with `q_init = 5.0` (= `coop_payoff` = the
optimistic-init choice consistent with the spec's "value of
cooperation" wording). If +β then dominates, this is methodological;
if -β still dominates, the issue is deeper.

### (2) METHODOLOGICAL — episode budget too short (200) to escape risk-dom basin

200 episodes with regret-matching adversary and ε decaying over the
first 100 may not be enough for any β to convincingly select Stag.
Vanilla and -β may simply tie at the risk-dominant (Hare, Hare)
attractor; +β may be perturbing the dynamics in a way that briefly
visits low-payoff cells (mismatched Stag while opponent plays Hare
yields 0 reward).

Test: re-run with 1000 episodes. If at 1000 ep we see +β recover
above vanilla, this is a transient-dynamics artifact and the main
pass at 10k ep is not at risk; only the pre-sweep at 200 ep is.

### (3) METHODOLOGICAL — adversary choice (regret-matching) is the wrong stress

User selected `finite_memory_regret_matching` per HALT 5 OQ3 because
"regret matching tests the payoff-dominance claim directly via
counterfactual regret over the (Stag, Hare) payoff structure". But
regret matching is itself a learning algorithm that converges to a
correlated equilibrium of the payoff matrix; for stag-hunt, the
correlated equilibrium typically allocates more weight to (Hare,
Hare) than to (Stag, Stag). The opponent's behaviour may be biasing
the cell toward Hare-Hare regardless of the agent's β.

Test: re-run with `inertia` adversary (`inertia_lambda=0.9` —
runner's original guess); or `finite_memory_best_response`; or even
a `stationary_mixed` opponent with `probs=[0.5, 0.5]` to remove the
opponent-learning effect entirely. If +β recovers under any of
these, the regret-matching wiring is the root cause.

### (4) GENUINE FINDING — payoff-dominance claim doesn't hold under
this opponent at this q_init

If (1)–(3) all fail to recover the predicted ordering, the empirical
finding is that **fixed-positive TAB does NOT select payoff-dominant
equilibria in stag-hunt against a counterfactual-regret opponent
under uniform Q-init**. This is a publishable counter-intuitive
finding (per patch §5.3 disposition: log to
`counter_intuitive_findings.md`, paper-attention asterisk, proceed
with M6 normally) — but it weakens the headline claim and reshapes
the narrative.

### (5) BUG — the runner / metric / adversary has a defect

T1 trigger says "auto-fix loop per addendum §4.2 if Codex verdict is
BUG". Possible defects: agent ε exploration broken, AdaptiveBetaQAgent
β-targeting wrong-signed for stag-hunt, regret-matching adversary
producing wrong action distribution, AUC computed against wrong
reward signal. Codex bug-hunt focused on these surfaces is the next
step if (1)–(3) ablations all replicate the contradiction.

## Remediation options (orchestrator recommendation: do (a) before (b)/(c)/(d))

- **(a)** Cheap ablation pass — re-run AC-Trap pre-sweep at 4 (q_init,
  episodes) settings:
  | run | q_init | episodes | adversary | comment |
  |-----|-------:|---------:|-----------|---------|
  | A1 | 5.0  |  200 | regret_matching | spec-implicit optimistic init |
  | A2 | 0.0  | 1000 | regret_matching | longer horizon |
  | A3 | 0.0  |  200 | inertia(0.9)    | original opponent guess |
  | A4 | 0.0  |  200 | stationary[0.5,0.5] | opponent-learning removed |
  Wall-clock: ~12 min total. If any of A1–A4 recovers the prediction
  with d > 0.5 in the predicted direction, that result becomes the
  new wave-1.5 baseline; the contradiction is methodological.

- **(b)** Codex bug-hunt review on the pre-sweep results + the
  asymmetric_coordination game source + the agent + the schedule
  factory; if Codex returns "BUG" we hit the auto-fix loop.

- **(c)** Accept as GENUINE FINDING; log to
  `counter_intuitive_findings.md`; flag for paper attention with an
  asterisk on the payoff-dominance claim; proceed with M6 normally.

- **(d)** Dispatch (a), then if (a) fails to recover the prediction
  dispatch (b), then if Codex returns INCONCLUSIVE proceed to (c).

**Orchestrator's recommendation: (d)** — start with (a)A1 and (a)A4
because they are the cheapest discriminators between hypothesis (1)
(q_init) and hypothesis (3) (adversary). 4 runs × 3 methods × 3
seeds × 200 ep each = 36 runs total ≈ 4 min wall-clock. Decision
afterward depends on the result.

**Decision required from user** before:
- any further AC-Trap pre-sweep dispatch (a/b/c/d),
- wave 2 (Stage A dev pass) — Stage A includes AC-Trap with the
  same wiring as wave 1.5; if the issue is methodological we should
  fix the dev config first.
