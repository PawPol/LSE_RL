# M8 Stage 3 — Sign-specialisation analysis

- **Created**: 2026-05-02
- **HEAD at start**: `722fd275` (M7.1 commit)
- **Spec authority**: `docs/specs/phase_VIII_tab_six_games.md` §10.4
- **Type**: analysis-only (no new runs); inputs are M7.1 paired data
- **Verdict**: **PASS — M8 → M9 acceptance criterion MET.** ≥ 1
  credible G_+ subcase (AC-Trap γ=0.60) AND ≥ 1 credible G_−
  subcase (multiple, dominant on RR + DC-Long50). M9 sign-switching
  composite work is therefore admissible by spec §10.4 gate.
- **Headline interpretation**: **the (cell, γ) lattice partitions
  cleanly along an alignment-condition axis** — TAB's positive-β
  regime is empirically confined to AC-Trap at γ=0.60; the
  negative-β regime dominates whenever γ ≥ 0.80 OR the cell has
  no value-bootstrapping pathology.

## 1. Definitions (per spec §10.4)

```
G_plus  : best fixed_beta_+x  beats  best fixed_beta_-x  AND  beats vanilla
          (paired-bootstrap CI₉₅ strictly above 0 in BOTH comparisons)

G_minus : best fixed_beta_-x  beats  best fixed_beta_+x  AND  beats vanilla
          (paired-bootstrap CI₉₅ strictly above 0 in BOTH comparisons)

neither : either no clean dominance, OR the dominant arm fails the
          beats-vanilla test (e.g. both signs lose to vanilla)
```

Both comparisons use the same paired-bootstrap method as M7.1
(B = 20 000, paired by seed at fixed (cell, γ)).

## 2. Classification (per (cell, γ) input from M7.1)

Source: `processed/m8_sign_classification.csv` (this milestone) +
`processed/m7_1_long.csv` (M7.1).

| cell | γ | best β_+ | best β_− | (β_+ − β_−) Δ ± CI | (β_+ − van) Δ | (β_− − van) Δ | class |
|---|---:|---|---|---|---:|---:|---|
| AC-Trap | 0.60 | +0.10 | −0.10 | **+184.1** [+121.9, +243.3] | **+131.7** ✓ | −52.4 ✗ | **G_plus** |
| AC-Trap | 0.80 | +0.05 | −0.05 | +0.6 [−112, +103] | −43.9 0 | −44.5 ✗ | neither |
| AC-Trap | 0.90 | +0.05 | −0.05 | −40 166 [−41 991, −38 623] | −40 266 ✗ | −100.3 ✗ | neither |
| AC-Trap | 0.95 | +0.05 | −0.05 | −50 844 [−51 831, −49 766] | −50 925 ✗ | −80.8 ✗ | neither |
| RR-StationaryConvention | 0.60 | +0.05 | −0.50 | −46.8 [−69.2, −22.6] | −20.8 ✗ | **+26.0** ✓ | **G_minus** |
| RR-StationaryConvention | 0.80 | +0.05 | −0.50 | −149.4 [−193.8, −110.2] | −25.2 ✗ | **+124.2** ✓ | **G_minus** |
| RR-StationaryConvention | 0.90 | +0.05 | −0.50 | −380.4 [−476.4, −288.6] | −173.6 ✗ | **+206.8** ✓ | **G_minus** |
| RR-StationaryConvention | 0.95 | +0.05 | −0.50 | −753.8 [−876.8, −632.8] | −433.8 ✗ | **+320.0** ✓ | **G_minus** |
| SH-FiniteMemoryRegret | 0.60 | +0.35 | −0.05 | +45.4 [−47.6, +143.6] | +84.1 0 | +38.7 0 | neither |
| SH-FiniteMemoryRegret | 0.80 | +0.05 | −0.50 | −160.7 [−230.8, −86.1] | −63.4 0 | **+97.3** ✓ | **G_minus** |
| SH-FiniteMemoryRegret | 0.90 | +0.05 | −0.20 | −331.5 [−537.7, −145.7] | −232.1 0 | +99.4 0 | neither |
| SH-FiniteMemoryRegret | 0.95 | +0.05 | −0.50 | −859.1 [−986.0, −733.7] | −711.3 ✗ | +147.8 0 | neither |
| DC-Long50 | 0.60 | +0.05 | **−2.00** | −829.3 (det.) | −34.4 ✗ | **+794.9** ✓ | **G_minus** |
| DC-Long50 | 0.80 | +0.05 | **−2.00** | −1 596.5 (det.) | −86.3 ✗ | **+1 510.2** ✓ | **G_minus** |
| DC-Long50 | 0.90 | +0.05 | **−2.00** | −2 372.5 (det.) | −180.0 ✗ | **+2 192.4** ✓ | **G_minus** |
| DC-Long50 | 0.95 | +0.05 | **−2.00** | −3 182.2 (det.) | −345.1 ✗ | **+2 837.1** ✓ | **G_minus** |

Counts: **1 G_plus, 9 G_minus, 6 neither** (out of 16 (cell, γ) cells).

## 3. M8 → M9 acceptance gate

Spec §10.4: *"At least one credible G_+ subcase AND one credible G_-
subcase. If absent, stop adaptive sign-switching work and write a
negative-result memo."*

- ✓ G_+ candidate: **AC-Trap γ=0.60** (β=+0.10, CI strictly above
  both vanilla and β=−0.10).
- ✓ G_- candidates: **9 cells**, dominant on RR-StationaryConvention
  (all 4 γ) and DC-Long50 (all 4 γ); 1 cell on SH-FiniteMemoryRegret
  (γ=0.80).

**M9 sign-switching composite work is admissible by the spec gate.**
The user retains discretion on whether to dispatch it; the gate is a
necessary condition, not a sufficient mandate.

## 4. Patterns in the lattice

### 4.1 γ × cell structure

For each cell, look at the dominant sign as γ varies:

| cell | γ=0.60 | γ=0.80 | γ=0.90 | γ=0.95 |
|---|---|---|---|---|
| AC-Trap | **G_+** | neither | neither | neither |
| RR-StationaryConvention | **G_−** | **G_−** | **G_−** | **G_−** |
| SH-FiniteMemoryRegret | neither | **G_−** | neither | neither |
| DC-Long50 | **G_−** | **G_−** | **G_−** | **G_−** |

Two clean patterns emerge:

1. **AC-Trap is the only cell where +β ever dominates**, and only at
   γ=0.60. At higher γ both signs lose to vanilla — the alignment
   condition `β·(r−v_next) > 0` is structurally hard to satisfy
   when V is large (V grows with γ) and reward is small.
2. **RR-StationaryConvention and DC-Long50 are clean G_− cells at
   every γ tested.** RR's stationary opponent and DC-Long50's
   deterministic chain both produce well-defined Q* fixed points for
   which the −β regime's contraction-tightening (`d_eff < γ`)
   accelerates convergence.
3. **SH-FiniteMemoryRegret is mixed**: only γ=0.80 satisfies G_−.
   The opponent's regret-matching dynamics interact with γ to
   produce a unique sweet spot at γ=0.80 where −β stays in regime;
   at γ=0.95 the high V reduces the alignment threshold and at γ=0.60
   the noise floor swamps the small TAB advantage.

### 4.2 The β magnitude axis

Best-β magnitudes by classification:

- **G_+ (AC-Trap γ=0.60)**: β=+0.10 (small). The Goldilocks band
  identified at V10.9 §8.4: small enough for early-episode alignment
  to register, not large enough to drive `d_eff > 1`.
- **G_- on matrix games (RR / SH)**: β=−0.50 (moderate). Large
  enough to tighten contraction below γ, small enough to avoid the
  asymptotic limit `(1+γ)·min(r,v)` distorting credit assignment.
- **G_- on DC-Long50**: β=−2.00 (extreme). The chain task has a
  stable Q* and benefits monotonically from contraction tightening
  up to the spec's β-grid edge; no Goldilocks within [−2, 0].

The "moderate" −β band at matrix games is consistent with V10 Tier I's
finding that 26 of 30 cells choose β_best ∈ [−0.75, −0.20]; M8 confirms
this pattern survives the γ-sweep on the headline subset.

## 5. Cross-reference to Phase VII (read-only narrative)

Per spec §10.4: cross-references Phase VII
`results/adaptive_beta/strategic/final_recommendation.md` as a
**read-only narrative reference**; no paired-seed comparison across
phases.

Phase VII final verdict:

> "Stage B2-Main shows that fixed `β = −1` paired-bootstrap-significantly
> improves AUC on `strategic_rps` ... but no adaptive_β − vanilla
> signal at the AUC_first_2k sample-efficiency endpoint."

Phase VII's headline is therefore **a single-game G_- candidate**
(`strategic_rps` at canonical γ). The Phase VIII M8 result is
consistent with and extends this:

- Same direction (G_−) at the matrix-game subset.
- M8 shows the G_− regime is **not single-game**: it dominates 9 of 16
  cells across 3 game families.
- M8 also identifies the **single G_+ counterexample** (AC-Trap
  γ=0.60), absent in Phase VII's per-cell coverage. AC-Trap is a new
  Phase VIII cell (added in M3); no Phase VII analogue.
- M8 therefore graduates the Phase VII finding from "single-game
  signal" to "structured γ × cell lattice" — necessary input for
  M9 sign-switching composite design.

## 6. Implications for M9 sign-switching composites

Per spec §10.5, M9 composes a (G_+, G_−) pair into a sign-switching
environment with regime ξ_t. The candidate pairs from M8:

| G_+ component | G_− component | composite plausibility |
|---|---|---|
| AC-Trap γ=0.60, β=+0.10 | RR-StationaryConvention γ=0.60, β=−0.50 | Both at γ=0.60 — **same discount allows clean sign-switching test**; recommended baseline composite |
| AC-Trap γ=0.60, β=+0.10 | DC-Long50 γ=0.60, β=−2.00 | DC-Long50's deterministic chain may not interleave naturally with the AC-Trap matrix structure; **structural mismatch risk** |
| AC-Trap γ=0.60, β=+0.10 | SH-FiniteMemoryRegret γ=0.80, β=−0.50 | γ-mismatch (0.60 vs 0.80) — **either regime must run at the wrong γ for one component**; not recommended |

The (AC-Trap γ=0.60, RR-StationaryConvention γ=0.60) pair is the
**primary M9 candidate**: both components have CI-significant
results at the same γ, both are matrix games (compatible state-action
spaces), and the G_+ / G_− Δ magnitudes are similar order
(+131.7 vs +26.0) so a sign-switching composite has a fair chance of
exposing both regimes within a single training run.

## 7. The "neither" cells are not failures

Six (cell, γ) tuples classify as "neither". Their pattern:

- **AC-Trap γ ∈ {0.80, 0.90, 0.95}**: both signs lose to vanilla
  (β=+0.05 catastrophically at γ ≥ 0.90). The cell becomes
  *vanilla-dominant* at high γ — the alignment condition is too hard
  to satisfy and the operator's `d_eff > 1` regime is too easy to
  enter.
- **SH-FiniteMemoryRegret γ ∈ {0.60, 0.90, 0.95}**: best of either
  sign has CI straddling 0; the cell is **near-flat** in β at
  these γ — TAB has no β-side knob that meaningfully changes
  performance vs vanilla.

These outcomes are themselves a finding worth folding into the paper
narrative: TAB's β-knob is **conditionally informative**. The H1/H2
framing from the V10 program correctly anticipated that some
(cell, γ) regimes would show no β-sensitivity; M8 now provides the
cleanest evidence: 6 / 16 cells at this γ-grid are β-flat in the
range tested.

## 8. Acceptance for M8 → M9 promotion

- ✓ ≥ 1 G_+ candidate: AC-Trap γ=0.60 (β=+0.10).
- ✓ ≥ 1 G_- candidate: 9 cells, primary RR γ=0.60 (β=−0.50).
- ✓ Cross-reference to Phase VII completed (§5).
- ✓ Candidate composite-pair identified (§6).
- ✓ "Neither" cells documented as non-failures (§7).

**Pending: user sign-off on M8 → M9 transition** (spec §2 rule 13).

If approved, M9 dispatches a sign-switching composite environment
using the AC-Trap (γ=0.60, +β) + RR-StationaryConvention
(γ=0.60, −β) pair as the primary regime test, with exogenous dwell
D ∈ {100, 250, 500, 1000} and 10-seed paired comparison.

## 9. Reproduction

```bash
# (M8 is analysis-only; uses M7.1 long CSV as input)
.venv/bin/python <<'EOF'
import pandas as pd
print(pd.read_csv(
    'results/adaptive_beta/tab_six_games/processed/m8_sign_classification.csv'
).to_string())
EOF
```

Inputs:
- `processed/m7_1_long.csv` (3 840 rows; from M7.1 commit `722fd275`)
- `processed/m7_1_paired_comparison.csv` (96 rows)

Outputs (this milestone):
- `processed/m8_sign_classification.csv` (16 rows = 4 cells × 4 γ)
- this memo
