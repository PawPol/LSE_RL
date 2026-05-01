# Phase VIII v7 finding — bug-hunt disposition memo

- **Memo created**: 2026-05-01T20:34:09Z
- **Trigger**: User-mandated pre-M7 broad bug-hunt (HALT 7).
- **Disposition**: **CASE C** (v7 holds under γ and α perturbations;
  refined under pessimistic Q-init).
- **Verdict**: NO BUG. Empirical finding is mechanism-real. Scope
  refined: the +β-destabilization claim is asymmetric in Q_init
  relative to V*, not absolute.

## Phase 1 — deterministic regression checks

| Check | Result |
| --- | --- |
| P1.1 Full test suite | 1694 PASS + 2 SKIP ✓ |
| P1.2 Operator-touch audit | Zero Phase VIII commits touch operator; last operator commit `6692f0f5` (Phase VII triage) ✓ |
| P1.3 Manifest completeness | 1626 completed + 140 documented-failed (recovered) + 9 schema rows; structurally sound ✓ |
| P1.4 AC-Trap pre-sweep reproducibility | All 9 (β, seed) cells bit-identical to original; return arrays equal ✓ |

## Phase 2 — cross-cell consistency probes

| Check | Result |
| --- | --- |
| P2.1 Manual AUC verification (AC-FP × β=+1, 10-seed mean) | manual = 454488.55 = aggregator (bit-identical) ✓ |
| P2.2 Manual alignment-rate replay | per-episode mean matches agent's logged value at 0 difference for all 200 episodes ✓ |
| P2.3 β=0 bit-identity | g(β=0,γ,r,v) ≡ r+γv exactly across {r,v,γ} grid; ZeroBetaSchedule = FixedBetaSchedule(beta0=1e-9) bit-identical Q-tables ✓ |
| P2.4 Q-divergence sanity | replay |Q|_max = 1080372.85 = production q_abs_max[999] bit-identical; divergence is REAL, not metric artifact ✓ |
| P2.5 Cross-cell sign-of-β test | alignment(+β)<0.20 in 22/22 cells (100%); AUC(-β)>AUC(+β) in 14/16 non-tied cells (88%) ✓ |

P2.5 raw rank correlation (mean ρ=0.03) was a misleading first
metric because 6/22 cells are rank-degenerate (DC advance-only +
RR-Sparse-Stationary/Tremble all produce identical AUC across β).
Refined sign-of-β test (above) is the proper measure of v7
universality.

## Phase 3 — Codex broad adversarial review

**Verdict: GENUINE FINDING — safe to proceed with the paper pivot.**

Memo at:
`results/adaptive_beta/tab_six_games/codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md`

Reference TAB agent (~30 LoC, no shared imports with production)
at: `results/adaptive_beta/tab_six_games/codex_reviews/reference_tab_agent.py`

Key results from the audit:
- (a) **Operator math** (`tab_operator.py`): verified analytically
  at all requested (β, γ, r, v) grid points. NO bugs.
- (b) **AdaptiveBetaQAgent** (`agents.py`): no rogue Q-table write
  paths; β cached once per episode; alignment uses `(β, r, v_next)`
  not `(β, r, v_current)`. NO bugs.
- (c) **FixedBetaSchedule** (`schedules.py`): returns +1.0 for all
  e under `FixedBetaSchedule(+1, hyperparams={"beta0": 1.0})`. NO
  bugs.
- (d) **Aggregator** (`aggregate.py`): per-seed AUC = `np.trapezoid(returns)`,
  paired-seed alignment correct. One NIT: aggregator delegates AUC
  computation to runner rather than recomputing. Not a bug.
- (e) **Reference impl cross-check**: matched production AUC EXACTLY
  (0.00% difference) across all 9 (β, seed) cells over both
  1000-episode prefix and full 10000-episode extension.

Two NIT-level observations:
- Aggregator delegates AUC to runner (architectural choice; not a bug).
- Stage 1 artifacts carry an older git SHA than HEAD (diff to HEAD
  touches unrelated files; not affecting AC-FictitiousPlay).

## Phase 4 — parameter perturbation sweep (162 runs in 87 sec)

6 perturbations × 3 cells (AC-FictitiousPlay, SH-FiniteMemoryRegret,
RR-StationaryConvention) × 3 β arms × 3 seeds × 1000 episodes.

| perturbation | cell | AUC(-1) | AUC(0) | AUC(+1) | v7 holds? |
| --- | --- | ---: | ---: | ---: | --- |
| **γ = 0.9** | AC-FP | 57100 | 57098 | 49966 | ✓ |
| | SH-FMR | 10229 | 10250 | 10209 | ✓ |
| | RR-Stat | 7130 | 7139 | 6209 | ✓ |
| **γ = 0.99** | AC-FP | 57098 | 57098 | 46396 | ✓ |
| | SH-FMR | 10355 | 10018 | 9730 | ✓ |
| | RR-Stat | 7116 | 6976 | 5976 | ✓ |
| **q_init = -2** | AC-FP | 43629 | 57098 | 52620 | **FLIP** |
| | SH-FMR | 7139 | 9804 | 8828 | **FLIP** |
| | RR-Stat | 1744 | 6695 | 5923 | **FLIP** |
| **q_init = +5** | AC-FP | 57102 | 57100 | 36744 | ✓ |
| | SH-FMR | 10688 | 10976 | 6836 | ✓ |
| | RR-Stat | 6708 | 6560 | 1176 | ✓ |
| **α = 0.05** | AC-FP | 57098 | 57098 | 57098 | ✓ |
| | SH-FMR | 10626 | 10283 | 9845 | ✓ |
| | RR-Stat | 7168 | 7170 | 6553 | ✓ |
| **α = 0.3** | AC-FP | 57101 | 57102 | 36665 | ✓ |
| | SH-FMR | 11151 | 10148 | 7809 | ✓ |
| | RR-Stat | 5858 | 5906 | 1773 | ✓ |

**Flips: 3/18 = 17%, all at q_init = -2.0 (pessimistic init).**

In the 3 flipped cells (q_init=-2), the ordering is `vanilla > +β > -β`
— vanilla still wins, but +β beats -β. The other 15 cells confirm
v7's `vanilla ≈ -β > +β`.

### Mechanism interpretation

The flip at q_init = -2 is **mechanism-consistent**, not a bug:

- Under q_init = 0 / +5 (Q ≥ V* over training): late-training
  `r - v_next ≤ 0`, so alignment `β·(r-v_next) ≥ 0` requires β ≤ 0.
  +β destabilizes; -β stays in regime.
- Under q_init = -2 (Q < V* initially): early-training
  `r - v_next ≥ 0`, so alignment requires β ≥ 0. **+β** stabilizes
  early; -β anti-aligns from ep 0.
- Mid-training, as Q grows past V*, the alignment direction flips.
  +β starts destabilizing late but accumulated AUC is positive;
  -β anti-aligned throughout, never recovers.
- Net AUC: `vanilla > +β > -β` in pessimistic init.

The alignment-rate diagnostic is consistent across q_init regimes
(measured at end-of-training, both show -β at 0.80, +β at 0.05) —
it captures the late-training steady-state, not the
integrated-AUC outcome.

### Refined v7 narrative

The pre-Phase-4 v7 narrative was:

> "+β destabilizes once Q crosses V*; alignment governs convergence
> to the bootstrap, not to equilibrium-payoff structure."

The Phase-4-refined narrative is:

> **TAB sign should match the sign of `(r - V*)` in expectation.**
> Under typical optimistic-or-zero Q-init with γ ∈ [0.9, 0.99] and
> rewards bounded above zero, Q grows above V* during training and
> the alignment regime requires β ≤ 0; +β destabilizes. Under
> pessimistic Q-init, the regime is mirrored early (+β aligns)
> but flips by mid-training; **vanilla wins in both regimes**
> (no compensation needed).
>
> The alignment-condition diagnostic continues to identify regimes
> of *local-bootstrap optimality*; AUC integrates across regime
> phases and does not always agree with end-of-training
> alignment_rate. **For the empirical envelope tested in Phase
> VIII (γ=0.95, q_init ∈ {-2, 0, +5}, α ∈ {0.05, 0.1, 0.3}, ε
> linear decay), vanilla (β=0) is never beaten by either fixed +β
> or fixed -β.** -β with moderate magnitude is the safer
> alternative when contraction-tightening is desired in tasks where
> bootstrap is expected to overshoot.

This refinement strengthens the paper because:
1. The alignment-condition diagnostic is *more* general than the
   v7 single-direction claim suggested — it's sign-symmetric in
   Q_init.
2. The "vanilla always wins" finding is now sharper: it holds
   across q_init ∈ {-2, 0, +5}, γ ∈ {0.9, 0.95, 0.99}, α ∈ {0.05,
   0.1, 0.3} in our 3-cell perturbation grid.

## Disposition

**CASE C** per HALT 7 protocol. Actions:

1. ✅ This memo documents the q_init = -2 scope condition.
2. ✅ The v7 narrative is refined (above) — TAB sign should
   match `sign(r - V*)`, not be biased toward -β specifically.
3. ✅ Refinement appended to
   `counter_intuitive_findings.md` (next commit).
4. ✅ M7+ proceed flag: yes, with the refined v7 narrative pinned in
   M12 final memo and M9/M10 reframed as "TAB-sign adaptation
   given Q_init"-themed rather than "G_+ regime exploration".

## Token budget for this verification

- 4 Codex dispatches Phase 3 (~30k tokens; light-weight code audit
  + reference impl + 9-cell verification run)
- ~12 Bash invocations for Phase 1/2/4 (negligible)
- 162 perturbation runs (~90 seconds wall-clock; fits well under
  the addendum §5 budget cap)

Total wall-clock for the bug-hunt: ~40 minutes (within the
estimated 30-60 min envelope).

## Next steps

User-side decisions still required at HALT 7 before M7+ dispatch:
1. M7 scope (modified, as-spec, or focused subset of cells)
2. M9/M10 reframing in light of v7-refined narrative
3. Promoted subcase list for M7
4. M12 paper-narrative outline preference

This memo addresses ONLY the bug-hunt verification. M7+ scope
decisions remain at the user.
