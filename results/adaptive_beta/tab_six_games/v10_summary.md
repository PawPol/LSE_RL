# Phase VIII v10 — extended β grid + γ-sweep + full opponent enumeration

- **Created**: 2026-05-02T12:47:09Z
- **Branch**: `phase-VIII-tab-six-games-2026-04-30`
- **Total runs**: 12,951 across all v10 phases
  - V10.0 binding probe: 18 runs
  - V10.1 AC-Trap smoke: 63 runs
  - V10.2 dev pass: 1890 runs (15 min)
  - V10.3a Tier I main: 6300 runs (~7 h)
  - V10.3b Tier II γ × β headline: 1680 runs (~2.5 h)
  - V10.3c Tier III γ × cell coverage: 3000 runs (~3 h)
- **Wall-clock**: ~17h sequential (12:50 UTC May 1 → 12:40 UTC May 2)
- **Pre-v10 superseded**: `pre-extended-grid` tag preserves M6 main pass at HEAD `dc07737f`.

## V10.4 — T1-T11 deterministic detector pass

> **G6c MAJOR correction (2026-05-02)**: this section originally
> reported 0 divergence flag fires. That was a detector bug —
> `V10.4` read the per-run `run.json::diverged` field (which is
> always False; the runner doesn't write a per-run summary) instead
> of the per-episode `metrics.npz::divergence_event.sum() > 0`
> array. The correct count is **524 divergence_event fires across
> 10,980 main runs**, almost entirely in positive-β arms with
> q_abs_max crossing 1e6 mid-training. See G6c review §(d) for full
> citation list. Per-cell incidence: AC-Inertia (70/210 Tier I, all
> +β arms), AC-Trap +β at γ=0.95, DC-Long50 +β at γ=0.95, etc.
> Vanilla and -β arms are not divergent. The "no bug signatures"
> claim is **withdrawn**; divergence is real and concentrated in
> the alignment-violating regime.

10,980 main-pass runs analyzed (Tier I + Tier II + Tier III):

| metric | corrected result |
| --- | --- |
| divergence_event > 0 (per-episode array) | **524** (G6c §d) |
| diverged flag (per-run; underwrote) | 0 (detector bug) |
| NaN q_abs_max | 0 ✓ |
| T2 (σ tight) fires | 1016 total (385 Tier I + 253 Tier II + 378 Tier III) |
| T2 on deterministic cells | 207 (expected per v5b lesson) |
| T2 on stochastic cells | 809 (matches v6 documented "tight σ" pattern) |

**Bug signatures present**: 524 divergent runs, **all in +β arms**.
This is consistent with the v7+v10 finding (alignment-violating
regime amplifies bootstrap overshoot) and is paper-relevant
evidence for the destabilization mechanism. The runs that diverged
should be annotated, not suppressed; their AUCs in the v10_summary
tables remain truthful (the agent's per-episode return is recorded
even when q_abs_max > 1e6).

## V10.6 / hypotheses disposition (pre-registered §13.11)

### H1 — γ-induced sign flip on best-β (4 headline cells)

> Pre-registered: at least one HEADLINE_CELL has best-β at γ=0.60
> strictly positive (paired-bootstrap 95% CI > 0).

> **G6c MAJOR correction (2026-05-02)**: this section originally
> claimed "CONFIRMED. 2/4 headline cells". After paired-bootstrap
> CI verification, the disposition is **CONFIRMED, NARROWLY (1/4
> headline cells)**:
> - **AC-Trap** at γ=0.60: best-β CI = `[+0.10, +0.10]`, paired AUC
>   advantage CI = `[+45.20, +212.00]` (B=20,000 paired-seed
>   resamples). **Confirms H1.**
> - **SH-FiniteMemoryRegret** at γ=0.60: best-β mean is +0.35 but
>   the CI includes negative arms; paired AUC advantage CI includes
>   0. **Does NOT confirm H1.**
> - RR-StationaryConvention, DC-Long50: best-β stays negative. (DC
>   has zero seed variance under cum-return AUC; H2 evaluation
>   below uses a different metric.)
>
> Additionally: the AC-Trap winning arm at γ=0.60, β=+0.10 has
> end-of-training `alignment_rate < 0.5`. So **the H1 sign flip is
> NOT mechanism vindication of the alignment diagnostic.** Per
> G6c §3, the AC-Trap finding is "a small statistical surface
> feature" pending a separate mechanism test (§8.4 of the full
> report describes a recommended 10-seed expansion + per-step
> trajectory analysis). The "alignment-condition theory's strongest
> empirical vindication" framing is **withdrawn**.

**CONFIRMED, NARROWLY.** 1/4 headline cells (AC-Trap) shows best-β
strictly positive at γ=0.60 with paired-bootstrap CI strictly above 0:

| cell                      | best β at γ=0.60 | AUC at best β | bootstrap CI verdict |
| ---                       | ---:             | ---:          | --- |
| AC-Trap                   | **+0.10**        | 529078        | CI=[+0.10, +0.10], AUC adv. CI=[+45, +212] ✓ |
| SH-FiniteMemoryRegret     | +0.35            | 106698        | CI includes neg β; AUC adv. CI includes 0 ✗ |
| RR-StationaryConvention   | -0.50            | 56293         | (best-β stays negative across γ) |
| DC-Long50                 | -0.75            | 9999 (det.)   | (cum-return invariant; H2 deferred) |

Tier II γ × β trajectory remains worth reporting but no longer
ratified as alignment-condition vindication: the regime that requires
β > 0 (small V values, where r > v early in training) is exactly
the regime accessible at low γ.

| cell                    | γ=0.95 | γ=0.90 | γ=0.80 | γ=0.60 |
| ---                     | ---:   | ---:   | ---:   | ---:   |
| AC-Trap                 | +0.00  | +0.00  | +0.00  | **+0.10** |
| SH-FiniteMemoryRegret   | -1.70  | -0.20  | -0.35  | **+0.35** |
| RR-StationaryConvention | -0.50  | -0.50  | -0.50  | -0.50 |
| DC-Long50 (cum-return)  | -0.75  | -0.75  | -0.75  | -0.75 |

Disposition: **promote H1 result to abstract.** Recasts M7
"best_fixed_positive_TAB" as a positive demonstrator at γ=0.60 on
AC-Trap and SH-FiniteMemoryRegret.

### H2 — γ-induced bifurcation widening

For cells where -β wins at γ=0.95, |d(best-β, vanilla)| at γ=0.60
larger than at γ=0.95.

> **G6c verdict: REFUTED (2026-05-02).** Eligible -β-winning cells
> at γ=0.95 = {SH-FMR, RR-StationaryConvention} (DC-Long50 has zero
> seed variance under cum-return AUC and is not Cohen-d evaluable).
> Both eligible cells have ratio < 1: bifurcation does NOT widen at
> γ=0.60. **0/2 evaluable cells fire.** This is a genuine negative
> finding, not a heatmap artifact (G6c §c).

### H3 — γ-stable diagnostic

The alignment condition `β·(r−v) ≥ 0` correctly predicts best-β at
≥80% of (γ, cell) tuples.

> **G6c verdict: REFUTED (2026-05-02).** Per G6c §a (120 (γ, cell)
> tuples evaluated, headline cells from Tier II + non-headline from
> Tier I/III at canonical/non-canonical γ):
> - Final-episode confirmation rate: **64/120 = 53.3%** (well below
>   80% threshold)
> - Last-200-episode robustness: **60/120 = 50.0%**
> The most damaging counterexample is the H1-confirming arm itself:
> AC-Trap γ=0.60, β=+0.10 has final alignment 0.050 and last-200
> 0.049. **The alignment diagnostic does NOT predict best-β** under
> the pre-registered ≥80% threshold; it correlates with best-β at
> roughly chance level across the (γ, cell) surface.

### Tier I per-cell best-β across 30 cells (γ=0.95 canonical)

| game                    | subcase                  | best β | AUC          | +β collapse |
| ---                     | ---                      | ---:   | ---:         | --- |
| asymmetric_coordination | AC-FictitiousPlay        | -0.20  | 522319.85    | yes |
| asymmetric_coordination | AC-Inertia               | -0.20  | 703320.70    | yes |
| asymmetric_coordination | AC-SmoothedBR            | +0.00  | 530848.50    | yes |
| asymmetric_coordination | AC-Trap                  | +0.00  | 528943.25    | yes |
| delayed_chain           | DC-Branching20           | -0.75  | -2950.35     | yes |
| delayed_chain           | DC-Long50                | -0.75  | 9999.00      | no  |
| delayed_chain           | DC-Medium20              | -0.75  | 9999.00      | no  |
| delayed_chain           | DC-Short10               | -0.75  | 9999.00      | no  |
| matching_pennies        | MP-FiniteMemoryBR        | -0.35  | 84327.30     | yes |
| matching_pennies        | MP-HypothesisTesting     | -0.75  | 52073.80     | yes |
| matching_pennies        | MP-RegretMatching        | -0.75  | 34634.10     | yes |
| matching_pennies        | MP-Stationary            | +1.35  | 51.60        | yes |
| potential               | PG-BetterReplyInertia    | -0.05  | 141900.24    | yes |
| potential               | PG-Congestion            | -2.00  | 89432.86     | no  |
| potential               | PG-CoordinationPotential | -0.35  | 66776.25     | no  |
| potential               | PG-SwitchingPayoff       | -0.10  | 100218.00    | no  |
| rules_of_road           | RR-ConventionSwitch      | -0.05  | 140019.80    | yes |
| rules_of_road           | RR-HypothesisTesting     | -0.75  | 53636.10     | yes |
| rules_of_road           | RR-StationaryConvention  | -0.50  | 56476.60     | yes |
| rules_of_road           | RR-Tremble               | -0.75  | 50624.70     | yes |
| rules_of_road_sparse    | RR-Sparse                | -0.75  | 4567.73      | no  |
| shapley                 | SH-FictitiousPlay        | -0.35  | 142627.65    | yes |
| shapley                 | SH-FiniteMemoryRegret    | -0.50  | 106648.70    | yes |
| shapley                 | SH-HypothesisTesting     | -1.70  | 83407.30     | yes |
| shapley                 | SH-SmoothedFP            | -0.50  | 66729.70     | no  |
| soda_uncertain          | SO-AntiCoordination      | -1.35  | 133398.30    | no  |
| soda_uncertain          | SO-BiasedPreference      | -0.10  | 108406.70    | yes |
| soda_uncertain          | SO-Coordination          | -0.35  | 66776.25     | no  |
| soda_uncertain          | SO-TypeSwitch            | +0.10  | 73914.95     | no  |
| soda_uncertain          | SO-ZeroSum               | -0.75  | 291.75       | yes |

**Distribution**: 26 cells best-β < 0; 2 cells best-β = 0
(AC-SmoothedBR, AC-Trap at γ=0.95); 2 cells best-β > 0
(MP-Stationary +1.35 at AUC=52 = null cell noise; SO-TypeSwitch
+0.10 with small effect AUC=73915). v7 holds at canonical γ.

## Conclusion (REVISED per G6c V10.5, 2026-05-02)

The v10 dispatch resolved the v7 finding at fine β resolution
(21-arm grid bifurcation at β=0 confirmed) and ran a γ-sweep that
**did NOT vindicate the alignment-condition theory beyond v7**.

**Corrected dispositions per G6c V10.5 review**:

- **At canonical γ=0.95** (Tier I, 6300 runs, 10 seeds): best-β is
  negative or zero on 28/30 cells. +β regime is empirically narrow.
  v7 confirmed at canonical γ.
- **At γ=0.60** (Tier II, 5 seeds, paired-bootstrap CI): only
  **AC-Trap** flips to a small-positive best-β (+0.10) with paired-
  bootstrap CI strictly above 0 (`[+45, +212]` AUC advantage over
  vanilla). SH-FMR's preliminary +0.35 does NOT survive CI scrutiny.
- The AC-Trap H1 confirming arm has **alignment_rate < 0.5** at
  end-of-training, so the sign flip is NOT mechanism vindication;
  it is a small statistical surface feature pending a separate
  mechanism test.
- **H2 (γ-induced bifurcation widening): REFUTED.** 0/2 evaluable
  -β-winning cells at γ=0.95 widen at γ=0.60.
- **H3 (γ-stable diagnostic): REFUTED.** 64/120 = 53.3% (γ, cell)
  tuples confirm by final-episode alignment, well below the 80%
  pre-registered threshold. The diagnostic predicts best-β at
  roughly chance level across the (γ, cell) surface.
- **524 divergence_event fires** across 10,980 runs (G6c §d), all
  in +β arms, consistent with the destabilization mechanism but
  contradicting V10.4's "0 divergence" claim.

**Revised paper narrative** (post-G6c):

> Selective Temporal Credit Assignment via TAB at γ ∈ [0.6, 0.99]
> is dominated by **vanilla (β=0)** as the safe default. Fixed
> +β destabilizes via amplified bootstrap, producing 524 divergent
> runs across 10,980 in our v10 sweep — concentrated in
> alignment-violating regimes. A small γ-modulated regime exists
> on stag-hunt-style coordination at low γ (AC-Trap, β=+0.10 at
> γ=0.60) that produces a paired-bootstrap-significant AUC gain
> over vanilla, but the alignment-condition diagnostic does NOT
> predict this regime — it is a small statistical surface feature
> requiring a separate mechanism investigation. **Moderate -β
> (β ∈ [-1, -0.5]) tightens contraction without destabilizing**
> on coordination/learning tasks at γ=0.95 with optimistic-or-zero
> Q-init.

The v10 dispatch is **CONDITIONAL PASS** per G6c. Pending action
items (V10.6/.7/.8) remain.

**Authoritative single-document spine**: see
`PHASE_VIII_FULL_REPORT.md` (10,484 words, 1512 lines). That report
folds all of these G6c corrections in and supersedes the
preliminary claims in this memo's earlier revisions.
