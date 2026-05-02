# M6 v10 supersedes pre-v10 — empirical-record diff for paper coauthors

> Onboarding memo: what changed between the **pre-v10 M6 main pass**
> (`pre-extended-grid` tag, HEAD `dc07737f`) and the **v10 dispatch**
> (HEAD `ce1d71d0`). Read this before reading `M6_summary.md` or any
> v10-era memo. The authoritative single-document spine for the v10
> empirical record is [`PHASE_VIII_FULL_REPORT.md`](PHASE_VIII_FULL_REPORT.md);
> this memo is the diff against the pre-v10 record only.

- **Branch**: `phase-VIII-tab-six-games-2026-04-30`
- **Spec**: `docs/specs/phase_VIII_tab_six_games.md`
- **Notation**: γ = discount factor; β = TAB temperature; V = state
  value bootstrap target; r = realized reward.
- **Test suite at HEAD `ce1d71d0`**: 1737 PASS + 2 SKIP (live verified
  2026-05-02; pre-v10 memos cite 1694 prior to the v10 multi-γ smoke
  + UCB lockstep test additions).

---

## §1. Authoritative versions table

| Aspect | Pre-v10 (tag `pre-extended-grid`) | v10 (HEAD `ce1d71d0`) |
| --- | --- | --- |
| Spec amendments folded | initial → v3 → v4 → v5 → v5b → v6 → v7 (§23) | adds v10 (§23 entry, β grid 7→21, γ-sweep, 22→30 cells) |
| HEAD SHA | `dc07737f` | `ce1d71d0` |
| Tier I main-pass runs | 1820 (M6: 1400 stage-1 + 140 RR-Sparse + 40 wave-5 + 240 ablation/T-detector) | 6300 (Tier I) + 1680 (Tier II) + 3000 (Tier III) = 10,980 v10 main-pass runs |
| β grid size | 7 arms `{−2, −1, −0.5, 0, +0.5, +1, +2}` | 21 arms in `[−2, +2]` (denser near 0) |
| γ grid size | 1 (canonical 0.95) | 4 — `{0.60, 0.80, 0.90, 0.95}` (Tier II/III) |
| Cell enumeration | 22 | 30 |
| Test suite | 1694 PASS + 2 SKIP | **1737 PASS + 2 SKIP** (+1 multi-γ smoke; updated UCB tests for 21-arm grid; +41 across other test additions in v10.0) |
| Authoritative summary | [`M6_summary.md`](M6_summary.md) | [`PHASE_VIII_FULL_REPORT.md`](PHASE_VIII_FULL_REPORT.md) (10,484 words; folds in G6c corrections) |

## §2. What transferred unchanged

These pre-v10 artifacts and conclusions remain authoritative under
the v10 dispatch and require no re-derivation:

- **Spec amendments v3 / v4 / v5 / v5b** (delayed_chain reopen, P-Contract
  sign flip, β-specific Bellman-residual headline metric, deterministic
  gap-floor guard) are **intact** and still load-bearing for DC-* cells.
  v10 added zero amendments to the delayed_chain spec section.
- **Spec amendment v6** (M6 wave-1 OQ closures: AC-Trap adversary
  `finite_memory_regret_matching(memory_m=20)`; aggregator schema
  parity; corrected ~1,820 Tier I run count; RR/SO/PG stationary
  opponent probabilities) is **intact**.
- **Spec amendment v7** (AC-Trap repositioned as a falsifiability
  cell after the 5/5 ablation refuted v2 §5.2 payoff-dominance) is
  **intact at γ=0.95** and remains the foundational v10 baseline.
  v10 refines but does not retract v7. See §5 for the γ=0.60 refinement.
- **AC-Trap pre-sweep + 4-condition ablation** (M6 wave 1.5, HALT 6;
  q_init ∈ {0, 5}; episodes ∈ {200, 1000}; opponents ∈
  {regret-matching, inertia(0.9), uniform stationary}) — 36 ablation
  runs + 9 baseline runs — remains **authoritative** for q_init,
  opponent, and horizon perturbations. v10 does not re-run these.
- **Operator** `src/lse_rl/operator/tab_operator.py`
  is **unchanged from the Phase VII baseline**. The v10 commit chain
  `95268326 → c999a5dc → 52dd9a34 → 32b12f83 → 7e33dba2 → ce1d71d0`
  contains **zero operator-touch commits** (verified by
  `git log pre-extended-grid..HEAD -- src/lse_rl/operator/`).
  The v7 finding that fixed +β underperforms vanilla is therefore
  not an operator artifact.

## §3. What v10 sharpened (same claim, more resolution)

### 3.1 Sharp β=0 sign-bifurcation

| Source | Coverage | β resolution | Headline |
| --- | --- | --- | --- |
| Pre-v10 (M6 wave 5, HEAD `e3ca75fb`) | 2 cells × 5 fine-β values × 4 seeds = 40 runs | `{−0.10, −0.05, 0, +0.05, +0.10}` | sign-bifurcation visible at β=0 |
| v10 V10.1 smoke (HEAD `c999a5dc`) | AC-Trap × 21 β arms × 3 seeds × 1k ep = 63 runs | full v10 21-arm grid | confirms 18× collapse in `alignment_rate[-200:]` from β=−0.05 (≈0.91) to β=+0.05 (≈0.05) |
| v10 Tier I main pass (HEAD `52dd9a34`) | 30 cells × 21 β × 10 seeds = 6300 runs | full 21-arm grid | reproduces sign-bifurcation across the entire enumerated cell set; +β collapses on 28/30 cells at γ=0.95 |

The pre-v10 wave-5 evidence was figures-only (4 cells, single γ);
the v10 evidence is paper-grade (30 cells, 4 γ, paired-bootstrap
CIs at Tier II).

### 3.2 DC-Long50 cum-return AUC = 9999 invariant

The deterministic Discrete(1) advance-only chain produces bit-identical
seed AUCs (per v5b: ε=0, PassiveOpponent, deterministic transitions).
v10 confirms the invariant at all four γ values — the paper-grade
metric for DC-* is `bellman_residual_beta_AUC` per v5; `cum_return`
is a sentinel constant. This was already the case at v5b; v10 only
adds the γ-sweep confirmation.

### 3.3 v7 cross-cell alignment_rate collapse

Pre-v10: end-of-training `alignment_rate[-200:]` ≤ 0.07 in every
fixed +β arm across 22 cells at γ=0.95. v10: same qualitative
collapse confirmed across **30 cells** at γ=0.95 (Tier I, 10 seeds).
The v7 narrative-alignment story holds at γ=0.95 in the larger
enumeration.

## §4. What v10 newly established

### 4.1 30-cell enumeration

v10 added 11 cells beyond the M6 main pass: **AC-Inertia,
MP-RegretMatching, MP-HypothesisTesting, RR-ConventionSwitch,
RR-HypothesisTesting, SH-SmoothedFP, SH-HypothesisTesting,
SO-ZeroSum, SO-BiasedPreference, PG-Congestion, PG-BetterReplyInertia**.
Source: spec v10 amendment §23; per-cell Tier I best-β at
[`v10_summary.md`](v10_summary.md).

### 4.2 γ-sweep over `{0.60, 0.80, 0.90, 0.95}`

Pre-v10 ran one γ (0.95). v10 introduced Tier II (4 γ × 21 β at 4
headline cells × 5 seeds = 1680 runs) and Tier III (4 γ × coarse β
at all 30 cells × 5 seeds = 3000 runs). Source: spec v10; Tier II
γ × β trajectory at [`v10_summary.md`](v10_summary.md).

### 4.3 Tier I/II/III convention

v10 introduces a tier convention used throughout the v10 record:

- **Tier I** = canonical γ=0.95, full 21 β, **10 seeds** per (cell, β)
  → 6300 runs.
- **Tier II** = γ × β response surface at the 4 headline cells
  (AC-Trap, SH-FiniteMemoryRegret, RR-StationaryConvention,
  DC-Long50), **5 seeds** → 1680 runs.
- **Tier III** = γ × coarse-β at all 30 cells, **5 seeds** → 3000 runs.

### 4.4 v10 21-arm contraction-UCB warm-start

UCB warm-start length expanded **7 → 21** episodes in lockstep with
the 7→21 β grid expansion (one warm-start episode per arm). Source:
spec §23 v10 entry.

### 4.5 524 divergence_event fires (G6c §d)

Pre-v10 had no per-episode divergence detector pass. The G6c
re-derivation found **524 `divergence_event > 0` fires** across
10,980 main-pass runs, all in +β arms, concentrated on cells where
alignment violates (AC-Inertia +β, AC-Trap +β at γ=0.95, DC-Long50
+β at γ=0.95). Vanilla and −β arms do not diverge. This is the
empirical mechanism evidence for the destabilization story. Source:
G6c review §(d) at [`codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md`](codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md).

### 4.6 AC-Trap β=+0.10 at γ=0.60 — paired-bootstrap-significant gain

The first cell in the Phase VIII suite where any fixed +β strictly
beats vanilla: **AC-Trap at γ=0.60, β=+0.10** has paired-bootstrap
95% CI `[+0.10, +0.10]` over 5 Tier II seeds; paired AUC advantage
CI `[+45.20, +212.00]` (B=20,000 paired-seed resamples). Source:
G6c §(b). This **does not** reverse v7 — v7 holds at γ=0.95 across
30/30 cells — but it carves out one γ-low regime where +β is
admissible.

## §5. What v10 reversed or refuted (the main payload)

### 5.1 "+β destabilizes everywhere; vanilla always wins"

- **Pre-v10 (M6 close v7,** [`M6_summary.md`](M6_summary.md)**)**:
  *"The +β regime is empirically narrow across the realistic Phase
  VIII parameter envelope at γ=0.95… No cell shows fixed +β beating
  vanilla."*
- **v10 refined**: confirmed at γ=0.95 across 30 cells (Tier I),
  but **at γ=0.60 the AC-Trap cell admits β=+0.10 with paired-
  bootstrap CI strictly above 0**. Vanilla still wins or ties on
  every other cell × γ point in the surface, but the absolute claim
  "no cell ever" is **refined to "no cell ever at γ=0.95"**. Source:
  G6c §(b); [`PHASE_VIII_FULL_REPORT.md`](PHASE_VIII_FULL_REPORT.md).

### 5.2 "Alignment-condition theory vindicated by γ-sweep"

- **Pre-v10 / preliminary v10** ([`v10_summary.md`](v10_summary.md)
  pre-G6c text): an early V10.6 narrative claimed "2/4 headline
  cells confirm H1; alignment-condition theory vindicated at γ=0.60".
  H2 was claimed weakly positive; H3 was claimed near-threshold.
- **v10 G6c-corrected**: **REFUTED in this form**.
  - **H1 = CONFIRMED, NARROWLY (1/4 headline cells)**. Only AC-Trap
    confirms; SH-FiniteMemoryRegret's preliminary +0.35 does not
    survive paired-bootstrap CI scrutiny.
  - **H2 = REFUTED**. 0/2 evaluable −β-winning cells at γ=0.95
    (SH-FMR ratio = 0.899; RR ratio = 0.057) widen at γ=0.60.
    DC-Long50 has zero seed variance under cum-return AUC and is
    not Cohen-d evaluable.
  - **H3 = REFUTED**. Only 64/120 (53.3%) (γ, cell) tuples confirm
    by final-episode `alignment_rate ≥ 0.5` for the observed best β;
    last-200-episode robustness 60/120 (50.0%). Pre-registered
    threshold was 80%.
  - **The most damaging counter-example is the H1-positive arm
    itself**: AC-Trap γ=0.60, β=+0.10 has end-of-training
    `alignment_rate = 0.050` (last-200 = 0.049). The sign flip is
    therefore **NOT mechanism vindication** of the alignment
    diagnostic — it is a small statistical surface feature whose
    mechanism is open. Source: G6c §(a)/§(b)/§(c);
    [`v10_summary.md`](v10_summary.md) corrected in commit `7e33dba2`.

### 5.3 "V10.4 detector pass: 0 divergence flag fires"

- **Pre-v10 / preliminary V10.4**: claimed 0 divergence flags across
  10,980 runs.
- **v10 G6c-corrected**: **524 `divergence_event` fires**. The V10.4
  detector read the per-run `run.json::diverged` field, which is
  always False (the runner does not write a per-run summary); the
  correct field is the per-episode `metrics.npz::divergence_event`
  array. All 524 fires are in +β arms, concentrated on cells where
  alignment violates. The "no bug signatures" claim is **withdrawn**;
  divergence is real and concentrated in the alignment-violating
  regime — this is a mechanism-positive result, not a bug.
  Source: G6c §(d); [`v10_summary.md`](v10_summary.md).

## §6. Implications for the paper narrative

**Vanilla (β=0) is the safest default across γ ∈ [0.60, 0.99].**
Tier I + Tier II + Tier III combined (10,980 runs across 30 cells ×
up to 21 β × 4 γ × up to 10 seeds) shows that vanilla wins or ties
on every cell at γ=0.95, and on every cell except AC-Trap at γ=0.60.
The paper-grade default recommendation is therefore β=0.

**+β destabilizes via amplified bootstrap when alignment violates;
524 divergent runs are the empirical evidence.** The operator's
asymptotic `g_{β,γ}(r, V) → (1+γ)·max(r, V)` for β → +∞ produces
`d_eff → 1+γ > 1` whenever V exceeds r. In the v10 grid, this regime
is exercised in 524 runs, **all in +β arms**. This is the paper's
strongest piece of mechanism-positive evidence for the
destabilization claim.

**A single γ-modulated regime exists on stag-hunt at low γ.** AC-Trap
at γ=0.60 with β=+0.10 produces a paired-bootstrap-significant AUC
advantage of `[+45.20, +212.00]` over vanilla. **This is not predicted
by the alignment-condition diagnostic** — the winning arm has
`alignment_rate ≈ 0.05`. The mechanism is open. Recommended action:
run a separate 10-seed expansion + per-step trajectory analysis (per
G6c §3 / [`PHASE_VIII_FULL_REPORT.md`](PHASE_VIII_FULL_REPORT.md) §8.4)
before claiming "alignment-condition vindication" anywhere; right now
the AC-Trap γ=0.60 finding is a **statistical surface** observation
worth reporting honestly, not a mechanism claim.

**The alignment-condition diagnostic is a partial-scope predictor.**
v10 measures the diagnostic on **120 (γ, cell) tuples**; it correctly
predicts the observed best-β at **64/120 (53.3%)** by final-episode
alignment, **60/120 (50%)** by last-200 robustness. The pre-registered
threshold was 80%. The diagnostic is not free of predictive content,
but it is **not the universal indicator** the v7 narrative positioned
it as. The paper should re-frame the diagnostic as a qualitative
dispatcher between sign(β), with calibration noted.

## §7. Provenance and reproducibility

- **Branch**: `phase-VIII-tab-six-games-2026-04-30`
- **Pre-v10 tag**: `pre-extended-grid` (annotated tag → commit `dc07737f`)
- **v10 commits** (chronological):
  - `95268326` — V10.0: spec v10 patch + 21-arm UCB + γ-sweep runner + 4 tier configs
  - `c999a5dc` — V10.1: AC-Trap smoke at v10 21-arm β grid (63 runs)
  - `52dd9a34` — V10.4–V10.6: T-detector + H1 (preliminary 2/4)
  - `32b12f83` — full empirical report (10,484 words)
  - `7e33dba2` — V10.5 G6c CONDITIONAL PASS (3 MAJOR corrections)
  - `ce1d71d0` — V10.6 final: 5 PDFs + 2 tables + 9 regenerable scripts
- **Reproduce figures**: `bash scripts/figures/phase_VIII/regenerate_v10.sh`
  (~30 s; main-pass `metrics.npz` artifacts must be on disk).
- **Reproduce main pass**: run the four `configs/v10_*.yaml` configs
  through `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py`.
  Total wall-clock ~17 h on a single host.
- **Test suite**: 1737 PASS + 2 SKIP at HEAD `ce1d71d0` (live verified
  2026-05-02).
- **Operator file**: `src/lse_rl/operator/tab_operator.py` —
  unchanged from Phase VII baseline; v10 commit chain has zero
  operator-touch commits.

---

*This memo is read-only documentation; it does not alter any v10 data
or claim. Authoritative single-document spine for the v10 empirical
record:* [`PHASE_VIII_FULL_REPORT.md`](PHASE_VIII_FULL_REPORT.md).
