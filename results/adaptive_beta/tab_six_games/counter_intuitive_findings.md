# Counter-intuitive findings — Phase VIII

This file logs paper-relevant negative or counter-intuitive results
discovered during Phase VIII. Per spec §13 (success/failure honesty
norms) and patch §5.3 (T1+T3 disposition path), these are reported
in the paper rather than hidden by post-hoc cell selection.

---

## 2026-05-01 — AC-Trap pre-sweep refutes "+β selects payoff-dominant equilibria" claim (5/5 conditions)

- **Affected spec claim**: §5.4 / §10.2 patch §5.2 — "fixed positive
  TAB selects payoff-dominant equilibria where vanilla Q-learning is
  risk-dominated"; predicted `AUC(+1) > AUC(0) > AUC(-1)` with
  Cohen's d > 0.5 vs vanilla on the AC-Trap stag-hunt subcase.
- **Trigger**: HALT 6 (paper-critical T1+T3 fired during M6 wave 1.5).
- **Disposition**: GENUINE FINDING per Codex bug-hunt review at
  `results/adaptive_beta/tab_six_games/codex_reviews/AC_Trap_pre_sweep_review_2026-05-01T16-52-53Z.md`.
  No implementation bug; theoretical mechanism confirmed by raw
  per-episode `alignment_rate` and `q_abs_max` traces.
- **HEAD when finding landed**: applied at v7 amendment commit (TBD,
  follows ablation commit `1d88e769`).

### Empirical signature (5-condition ablation)

| condition | mean(−1) | mean(0) | mean(+1) | std(+1) | d(+1, 0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BASELINE q0/200/regret      | 10404 | 10337 |  8938 |  651 |  −3.04 |
| A1 q5/200/regret            | 10447 | 10363 |  8140 |  359 |  −8.60 |
| A2 q0/1000/regret           | 52909 | 52451 | 40025 |  751 | −23.38 |
| A3 q0/200/inertia(0.9)      | 12947 | 13776 | 13292 |  369 |  −1.52 |
| A4 q0/200/uniform[0.5,0.5]  | 11574 | 11264 | 10991 |  214 |  −1.47 |

**Five of five conditions** (4 strict reversal + 1 mixed) — the
prediction `AUC(+1) > AUC(0) > AUC(-1)` holds in **zero**
conditions across q-init ∈ {0, 5}, episodes ∈ {200, 1000}, and
opponents ∈ {regret-matching, inertia(0.9), uniform stationary}.

### Mechanism (theoretical, confirmed by Codex review §2(a))

The TAB target `g_{β,γ}(r, v) = (1+γ)/β · [logaddexp(βr, βv + log γ)
− log(1+γ)]` has asymptotics:
- `β → +∞`: `g → (1+γ) · max(r, v)` — when `v > r`, `d_eff → 1+γ > 1`
  for γ < 1 close to 1.
- `β → −∞`: `g → (1+γ) · min(r, v)` — when `v > r`, `d_eff → 0`.

Therefore positive β stabilizes ONLY in the **alignment regime**
`β · (r − v) > 0`, i.e. when realized reward exceeds the bootstrap
estimate. Once Q overshoots V*, alignment flips and +β amplifies the
overshoot rather than damping it. **Equilibrium payoff structure does
not enter this calculation** — only the *current* relationship
between r and v_next does.

### Empirical signature in raw per-episode arrays

For all `fixed_beta_+1` runs across all 5 conditions:

- `alignment_rate[:20].mean()` is in the range 0.55–0.76 (initial
  bootstrap below realized reward; alignment holds);
- `alignment_rate[-20:].mean()` collapses to 0.05 (alignment FAILS
  late in training because Q has overshot V*);
- `q_abs_max[-1]` ranges from 81,678 (baseline) to 1,085,189 (A2 long)
  — well above the finite-horizon discounted-payoff bound of ~64.15;
- A2 long fixed_beta_+1 seed 0: `divergence_event.sum() = 311` over
  1000 episodes (>30% divergence rate).

These traces match the destabilization mechanism precisely.

### Why this strengthens the paper rather than weakening it

1. **The headline claim is refined, not refuted.** The alignment-
   condition diagnostic (spec §3.3 / §7.2) correctly identifies
   AC-Trap as outside the +β regime — `alignment_rate` drops
   below 0.5 over training in every condition. The diagnostic
   is *predictively accurate*; the over-claim was at the
   game-theoretic-equilibrium level.

2. **The narrative becomes scope-correct.** "+β helps where
   bootstrap V tracks expected reward direction without overshoot"
   is a tighter, defensible claim than "+β selects payoff-dominant
   equilibria". The latter requires a global game-theoretic
   property; the former is a local stochastic-approximation
   property and is exactly what alignment indicators measure.

3. **AC-Trap becomes a useful negative.** As a falsifiability cell,
   AC-Trap demonstrates that the diagnostic doesn't generalize
   from local-bootstrap properties to global equilibrium-selection
   properties — which is itself a paper-relevant finding (the
   diagnostic's scope is sharpened, not invalidated).

### Action items

1. **Spec amendment v7** (this commit): reframe AC-Trap in §5.4 as
   a falsifiability cell; add §13.5 negative-result subsection per
   Codex's recommended language (memo §6).
2. **Wave 2 dev pass**: AC-Trap retained in the suite with the
   regret-matching opponent; ordering `−β > 0 > +β` is the new
   *expected* outcome and not a halt trigger.
3. **Paper draft**: when M6 results are written up, AC-Trap appears
   in the negative-results section, with the alignment-rate
   collapse trace as the central evidence of mechanism.

### Predictions for wave 2

- AC-Trap will repeat the `−β > 0 > +β` pattern at full Stage A
  budget (1k episodes × 3 seeds × 7 β arms).
- Other AC subcases (`AC-FictitiousPlay`, `AC-SmoothedBR`) may show
  different orderings; the alignment indicator should explain each.
- If wave 2 unexpectedly shows `+β > 0` on AC-Trap at full budget,
  that would be a fresh T1+T3 trigger (the ablation should have
  caught any genuine +β-favoring regime).
