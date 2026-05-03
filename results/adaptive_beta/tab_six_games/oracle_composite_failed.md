# M9 — oracle-validated composite FAILED

- **Created**: 2026-05-02 (post M7.2 + M8 commit chain)
- **Spec authority**: `docs/specs/phase_VIII_tab_six_games.md` §10.5
  Stage 4 acceptance gate
- **Verdict**: **FAIL.** The (AC-Trap γ=0.60, β_+=+0.10) ⊕
  (RR-StationaryConvention γ=0.60, β_−=−0.5) sign-switching composite
  does NOT satisfy the spec §10.5 oracle-dominance criterion. Per
  spec, this halts adaptive-β experimentation: "If oracle β does
  not beat fixed signs, redesign the composite once; if still
  failing, stop and write `oracle_composite_failed.md`. Do not
  force adaptive β experiments without opposite-sign regimes."
- **Implication**: M9 → M10 promotion **BLOCKED**. The spec-
  registered claim that TAB admits an adaptive-β controller that
  outperforms fixed-β baselines on regime-switching tasks is
  empirically refuted at this composite design.

## 1. What was tested

**Primary composite (M9.1):**
- G_+ component: `asymmetric_coordination` × `AC-Trap` ×
  `finite_memory_regret_matching(memory_m=20)` at γ=0.60. M8 G_+ at
  β=+0.10 (paired-CI [+88.10, +174.20] vs vanilla, V10.9 §8.4
  10-seed confirmation).
- G_− component: `rules_of_road` × `RR-StationaryConvention` ×
  `stationary_mixed(probs=[0.7, 0.3])` at γ=0.60. M8 G_− at β=−0.5
  (paired-CI [+8.20, +42.80] vs vanilla).
- Switching: exogenous dwell D ∈ {100, 250, 500, 1000} with regime
  flips on episode boundaries (`sign_switching_regime.py`).
- Methods (3 spec-mandated for the gate):
  - `oracle_beta` — outputs +0.10 in G_+ regime, −0.50 in G_−
    regime, 0 in episode 0 before any regime observation.
  - `fixed_positive_TAB` — β=+0.10 always.
  - `fixed_negative_TAB` — β=−0.50 always.
- Seeds: 5 paired (per spec smoke + gate budget).
- Episodes per run: 10 000.

## 2. Gate-1 (initial dispatch, all 4 dwells × 5 seeds)

Per-method AUC, mean ± SD across 5 seeds (dwell=250):
- `oracle_beta`        : 406 992.6 ± 5 796.5
- `fixed_positive_TAB` : 383 951.4 ± 3 439.6
- `fixed_negative_TAB` : 412 184.8 ± 3 147.7

Paired-bootstrap 95% CIs (B=10 000):

| comparison | Δ_AUC | CI | gate |
|---|---:|---|---|
| oracle − fixed_positive | +23 041.2 | [+19 046.2, +25 899.8] | **PASS** |
| oracle − fixed_negative | **−5 192.2** | [**−8 099.4, −2 421.0**] | **FAIL** |

Spec §10.5 requires BOTH comparisons to PASS (oracle dominates both
fixed signs). The `oracle − fixed_negative` CI is strictly below 0
→ gate-1 FAIL.

## 3. Per-regime diagnostic (gate-1)

Mean per-step return averaged across 5 seeds × 5 000 episodes per
regime at dwell=250:

| | regime=+1 (AC-Trap, supposed G_+) | regime=−1 (RR-Stat, G_−) |
|---|---:|---:|
| `fixed_negative_TAB` | **78.0** | **4.59** |
| `oracle_beta` | 77.1 | 4.26 |
| `fixed_positive_TAB` | 73.3 | 3.50 |

**`fixed_negative_TAB` (β=−0.5 always) beats every alternative on
BOTH regimes** — including the supposedly G_+ regime where M8's
standalone classification said β=+0.10 wins. This is qualitatively
inconsistent with the M8 standalone result.

## 4. Hypothesis: cross-regime Q-table contamination

The composite uses a SHARED Q-table across both regimes (Q[s, a]
holds one value per (state, action), regardless of which regime
generated the transition). For the (AC-Trap, RR-Stat) pair this is
particularly hostile because:

- AC-Trap rewards (Stag-Hunt structure: coop=5, risk=3) live on a
  scale comparable to RR-StationaryConvention rewards (per-episode
  return ~ 1).
- Both envs use the same default state encoder (horizon=20,
  n_actions=2 → 60 states). State indexes overlap, so Q[s, a] in
  AC-Trap and Q[s, a] in RR-Stat are written into the SAME table
  cell.
- After every regime flip, the agent's Q-values reflect the
  PREVIOUS regime's payoff scale. Re-learning to the new regime's
  Q* takes a chunk of episodes; with dwell=250, the agent never
  fully converges in either regime.

Empirically: at dwell=250, mean per-step return in the G_+ regime
is 73.3 for fixed-+0.10 (the M8 G_+ winner) versus 78.0 for
fixed-−0.5. **A 6.4 % return gap in favour of the wrong sign.**
The contamination dominates the alignment-condition signal that
made β=+0.10 win on standalone AC-Trap.

This is consistent with the **AC-Trap γ=0.60 G_+ result being a
narrow Goldilocks point** at standalone equilibrium that does
NOT robustly survive transfer to a non-stationary environment with
shared parameters.

## 5. Redesign attempt (per spec §10.5)

Spec §10.5: "redesign the composite once". Most charitable
single-axis redesign: increase dwell from grid `{100, 250, 500,
1000}` to dwell=1000 alone (each regime gets 5 000 contiguous
episodes — maximum within the 10 000-episode budget). If the agent
needs time to re-learn its Q-table after a regime flip,
dwell=1000 is the most generous case.

**Redesign at dwell=1000 (5 paired seeds, 10k episodes each):**

| | mean AUC | std |
|---|---:|---:|
| `oracle_beta` | 396 170.8 | 1 776.6 |
| `fixed_positive_TAB` | 393 562.4 | 871.8 |
| `fixed_negative_TAB` | 396 816.2 | 1 215.0 |

| comparison | Δ_AUC | CI₉₅ | gate |
|---|---:|---|---|
| oracle − fixed_positive | +2 608.4 | [+1 827.6, +3 609.2] | **PASS** |
| oracle − fixed_negative | **−645.4** | [**−1 177.0, −60.0**] | **FAIL** |

The fixed_negative gap shrinks (Δ from −5 192 down to −645), but
the CI is still strictly below 0 — **gate-2 FAILS as well**.

**Redesign attempt: FAILED.** Per spec §10.5 the next mandated step
is this memo + halt adaptive-β experiments.

## 6. Why the design fails (mechanism summary)

Three independent lines of evidence converge:

1. **Q-table contamination** (gate-1 §4): fixed_negative_TAB beats
   the M8-classified G_+ regime by 6 % return.
2. **Oracle gap shrinks but persists** (gate-2): even with 5 000-
   episode contiguous regime segments, fixed_negative beats oracle
   by 0.16 % AUC. Oracle would need to do BETTER than fixed_negative
   in the G_− regime AND better than fixed_positive in the G_+
   regime; instead the oracle's regime-wise advantage in G_+
   (matching fixed_positive's payoff there) does not compensate
   for fixed_negative's transfer advantage from G_+ to G_−.
3. **Standalone AC-Trap G_+ at β=+0.10 is a narrow Goldilocks**
   (V10.9 §8.4 mechanism follow-up): the +0.10 advantage is rooted
   in early-episode (ep 0–100) alignment that decays to ~0.05 after
   ep 1000. Composite restarts repeatedly perturb this transient,
   so the early-episode advantage never accumulates.

## 7. Implications for the paper claim

Phase VII-A's adaptive-β controllers (per-episode signed step on
$\beta\cdot(r-v_{\text{next}})$) and Phase VIII's
`contraction_UCB_beta` / `return_UCB_beta` schedules (UCB1 over a
21-arm β grid) presuppose that:

- **(P1)** there exist (cell, γ) pairs where opposite β signs are
  individually better than vanilla (G_+ AND G_−); and
- **(P2)** within a sign-switching composite of a (G_+, G_−) pair, an
  oracle-aware β controller dominates the best fixed sign on AUC.

M8 establishes (P1) — barely — with **1 G_+ candidate cell**
(AC-Trap γ=0.60) and **9 G_− candidate cells**. M9 directly tests
(P2). **M9 refutes (P2)** for the only viable composite candidate
in the suite (AC-Trap γ=0.60 ⊕ RR γ=0.60), at the smallest and
largest dwells of the spec grid.

The paper therefore CANNOT claim that adaptive β empirically
outperforms fixed β on regime-switching tasks within this empirical
envelope. Two paper-headline implications:

1. **The reframed claim** ("TAB is a credit-assignment mechanism for
   value-bootstrapping-required tasks (DC-Long50) and learning-vs-
   learning cycling regimes (SH-FMR γ=0.80)") survives — these are
   single-cell standalone claims that do not depend on M9.
2. **The earlier claim** that an adaptive-β controller supplies a
   meta-strategy across G_+ and G_− regimes does NOT survive. Any
   paper text suggesting "TAB enables a sign-switching adaptive
   controller" must be cut or qualified as: "The sole G_+ candidate
   in this suite (AC-Trap γ=0.60 with β=+0.10) does not survive
   composite Q-table sharing; its standalone advantage is a narrow
   Goldilocks band that is destabilised by cross-regime parameter
   transfer."

## 8. Why no second redesign

Spec §10.5 says "redesign the composite ONCE". I considered four
plausible alternative redesigns and report each:

1. **Larger β_+ magnitude (β_+ = +1.0 or +2.0)**: V10.9 §8.4
   demonstrated that β=+0.20 collapses AC-Trap AUC by ~10 000
   relative to vanilla; β=+1.0 and β=+2.0 are even worse (V10
   Tier I). Increasing β_+ magnitude would lose the G_+ regime
   contribution entirely. **Not viable.**

2. **Per-regime Q-tables**: An architectural change to the agent
   that maintains separate Q[+1, s, a] and Q[−1, s, a] tables. This
   is a different operator framework — the paper claim becomes
   "TAB with per-regime context" which is materially different from
   the spec's TAB definition. **Out of scope; would need new spec.**

3. **Different G_+ component**: M8 has only ONE G_+ candidate
   across the entire 16-tuple lattice (AC-Trap γ=0.60). There is no
   alternative G_+ to swap in. **No alternative exists.**

4. **Endogenous trigger** (rolling win/loss/predictability per
   spec §10.5): does NOT change the underlying Q-table contamination
   problem. The trigger only changes WHEN regimes flip, not
   whether the agent can hold two Q-functions on one table.
   **Mechanistically equivalent failure expected.**

Given (1)–(4), no plausible single-axis redesign would resolve the
shared-Q-table contamination identified in §4. Per spec, the second
redesign attempt is not authorised, and this memo documents the halt.

## 9. Reproduction

```bash
# Initial gate (5 seeds × 4 dwell × 6 methods × 1k–10k episodes)
.venv/bin/python -m experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage4_composite \
    --config experiments/adaptive_beta/tab_six_games/configs/stage4_composite_AC_RR_gamma060.yaml \
    --seed-list 0,1,2,3,4 \
    --output-root results/adaptive_beta/tab_six_games

# Redesign at dwell=1000
.venv/bin/python -m experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage4_composite \
    --config experiments/adaptive_beta/tab_six_games/configs/stage4_composite_AC_RR_gamma060.yaml \
    --dwell 1000 --methods oracle_beta,fixed_positive_TAB,fixed_negative_TAB \
    --seed-list 0,1,2,3,4 \
    --output-root results/adaptive_beta/tab_six_games

# Aggregator (oracle-gate computation)
# Inline in this memo's §2 + §5 — single-script paired-bootstrap
# applied to metrics.npz arrays loaded per (method, seed).
```

Raw artifacts at:
- `raw/VIII/m9_stage4_composite_AC_RR_gamma060/` (initial gate +
  redesign attempt; ~120 run directories with run.json + metrics.npz).
- `tasks/M9_plan.md` — original build plan (preserved).

## 10. Status of M9 → M10 / paper

- ✓ M9 oracle-validation gate FAILED at gate-1 and gate-2 (one
  redesign attempt completed per spec §10.5).
- ✗ M9 → M10 promotion BLOCKED.
- ✗ Adaptive-β experiments (Phase VII-A controllers, Phase VIII
  UCB schedules) are NOT empirically validated within this
  envelope.
- ✓ Paper headline (per §1.3.1 of `PHASE_VIII_FULL_REPORT.md`)
  remains valid: "TAB is a credit-assignment mechanism for
  delayed-credit + learning-vs-learning cycling regimes" — these
  claims do not depend on M9.
- ⚠ Section 11 of the report and any paper text mentioning
  "adaptive β" / "sign-switching controller" must be cut or
  qualified per §7 above.

The substantive scientific finding from M9 is itself paper-worthy:
**the sole G_+ regime in the V10/M7/M8 envelope (AC-Trap γ=0.60,
β=+0.10) is destabilised by cross-regime parameter sharing, so a
regime-switching composite cannot validate adaptive β in this
empirical scope.** This is consistent with V10.9 §8.4's
characterisation of the G_+ effect as an early-episode alignment
transient rather than a stable mechanism, and refines (not
contradicts) the M8 → M9 acceptance-gate logic: passing the M8
gate is a *necessary* but *not sufficient* condition for adaptive
β. Composite-level oracle dominance is the additional
necessary-and-sufficient test, and it fails here.
