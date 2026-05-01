# AC-Trap pre-sweep — 4-condition ablation memo

- **Memo created**: 2026-05-01 (HALT 6 step A.1; commit `1d88e769`).
- **Status**: pre-registered predictions recorded; runs pending.

## Baseline (HALT 6 trigger)

`pre_sweep_AC_Trap.yaml`: q_init=0, 200 ep, regret-matching opponent.

| schedule | β | mean AUC | std (ddof=1) |
| --- | ---: | ---: | ---: |
| vanilla    |  0 | 10337.17 |  29.02 |
| FixedNeg   | -1 | 10404.17 |  40.13 |
| FixedPos   | +1 |  8937.83 | 650.79 |

`d(+1, vanilla) = -3.04`; `d(-1, vanilla) = +1.91`. Direction REVERSED
from spec §10.2 patch §5.2 prediction `AUC(+1) > AUC(0) > AUC(-1)`.

## Pre-registered predictions (recorded BEFORE running, per HALT 6 §A.2)

The user's research-design hypothesis: "+β destabilizes once Q crosses
V*; alignment β·(r-v) ≥ 0 governs convergence to the BOOTSTRAP, not
to equilibrium-payoff structure". 22× variance of FixedPos vs vanilla
in baseline is the destabilization signature.

### A1 — q_init = 5.0 (= coop_payoff)

**Prediction**: +β should be EVEN WORSE than baseline because
alignment is violated from episode 0 (Q starts above V*). Ordering:
`-β > 0 > +β` with larger gaps than baseline.

### A2 — 1000 episodes (vs 200)

**Prediction**: same qualitative ordering as baseline; +β variance
grows further; -β slow convergence catches up to vanilla but does
not surpass it. Mean ordering preserved; absolute gaps may shrink
or grow.

### A3 — inertia(0.9) opponent

**Prediction**: same `-β > 0 > +β` ordering. Sticky opponent traps
agent into Hare basin faster; +β still destabilizes within Hare
basin.

### A4 — stationary uniform [0.5, 0.5]

**Prediction**: same ordering. Empirical Hare-mean (3.5) > Stag-mean
(2.5) under uniform opponent (Hare yields 3 always; Stag yields
0.5·5 + 0.5·0 = 2.5), so +β amplifies Hare convergence.

### Bug-hypothesis falsifier

If ALL four ablations confirm `-β > 0 > +β`, the finding is that
+β does NOT robustly select payoff-dominant equilibria across the
realistic parameter envelope; bug hypothesis (5) becomes unlikely.

If ANY ablation shows `+β > 0` (especially A1 with optimistic
init), the original v2 §5.2 claim is recoverable in a narrow
regime.

## Empirical results (2026-05-01, 36 ablation runs + 9 baseline = 45 total)

| condition                   | mean(−1) | mean(0)  | mean(+1) | std(+1) | d(+1, 0) | d(−1, 0) | ordering |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BASELINE q0/200/regret      | 10404.17 | 10337.17 |  8937.83 |  650.79 |  −3.038 | +1.913 | **−β > 0 > +β** (REVERSED) |
| A1 q5/200/regret            | 10446.83 | 10363.17 |  8139.67 |  359.48 |  **−8.595** | +1.382 | **−β > 0 > +β** (REVERSED, larger) |
| A2 q0/1000/regret           | 52909.33 | 52450.67 | 40025.33 |  750.97 | **−23.380** | +6.217 | **−β > 0 > +β** (REVERSED, persists & grows) |
| A3 q0/200/inertia(0.9)      | 12946.83 | 13776.50 | 13291.50 |  369.09 |  −1.515 | −2.454 | mixed (vanilla > +β > −β) |
| A4 q0/200/uniform[0.5,0.5]  | 11574.17 | 11264.00 | 10991.00 |  214.01 |  −1.472 | +2.782 | **−β > 0 > +β** (REVERSED) |

### Predictions vs observations

- **A1 (optimistic init)**: predicted "+β even worse"; observed **|d(+1,0)| =
  8.60** (vs baseline's 3.04). Effect is 2.8× larger than baseline.
  ✅ Confirms hypothesis (1) — alignment violation from episode 0.
- **A2 (longer horizon)**: predicted "+β variance grows"; observed
  std(+1) = 750.97 (vs baseline's 650.79; 1.15× larger), and
  d(+1,0) = −23.38 (8× larger than baseline). Destabilization is
  NOT transient and grows with training time. ✅ Confirms hypothesis (2)
  is *not* the cause (longer training does not rescue +β; it amplifies
  the destabilization).
- **A3 (inertia opponent)**: predicted "same ordering, traps faster";
  observed **mixed**: vanilla (13776) > +β (13292) > −β (12947).
  -β LOSES under sticky opponent, +β is closer to vanilla, neither
  beats vanilla. The sticky opponent locks the joint dynamics into
  the Hare basin so deterministically that the agent's β has weak
  influence. ❌ Hypothesis (3) (regret-matching is the cause) refuted —
  changing the opponent does not recover +β > 0; it just shrinks all
  gaps.
- **A4 (uniform opponent)**: predicted "same ordering"; observed
  −β > 0 > +β with d(+1,0) = −1.47. Confirmed.
  ✅ Hypothesis "agent-side dynamics alone produce the destabilization"
  validated; opponent-learning is not necessary for the effect.

### Bug-hypothesis falsification

**Zero of 5 conditions** show the predicted `+β > 0 > −β` ordering.
The destabilization signature (high std on FixedPos, mean below
vanilla) survives:
- both Q-init regimes (q=0 and q=5),
- both episode budgets (200 and 1000),
- three different opponent classes (regret-matching,
  inertia(0.9), uniform stationary).

Conclusion: bug hypothesis (5) is highly unlikely. The empirical
finding is robust across the realistic parameter envelope.

The user's research-design hypothesis is empirically confirmed:
> "+β destabilizes once Q crosses V*; alignment β·(r-v) ≥ 0 governs
> convergence to the BOOTSTRAP, not to equilibrium-payoff structure."

A1's larger-than-baseline destabilization with optimistic init is a
direct visual confirmation: starting Q above V* makes the alignment
violation manifest from the first episode, and +β amplifies it from
the start; baseline (q=0) only reaches the alignment-violating regime
after the agent's bootstrap has overshot, hence the smaller (but
still strongly negative) d.

A3's mixed (vanilla > +β > −β) ordering is a SECOND interesting
finding: under a strongly sticky opponent, β-effects are second-
order; opponent dynamics dominate.

## Disposition (HALT 6 step C)

This is the **GENUINE FINDING** branch (per patch §5.3). Bug hypothesis
(5) is implausible given 5/5 conditions failing the prediction across
diverse parameter regimes. The Codex bug-hunt review (HALT 6 step B)
is dispatched next regardless, per the user's "(d) ablation, then
Codex review, then disposition" procedure; Codex has the final word
on whether to additionally apply BUG-class fixes.

If Codex agrees with GENUINE-FINDING, apply HALT 6 step D (v7 spec
amendment repositioning AC-Trap as a falsifiability cell).

