# Phase VIII — Full Empirical Report

> Comprehensive empirical record of the Phase VIII "Six-Game Safe TAB" suite.
> Covers M0 → v10 (M0/M1/M2 reopen / M5 / M6 / V10.0–V10.9), every halt,
> every spec amendment (v3 → v10 / v5b / v6 / v7), every run dispatch, and
> every paper-relevant finding. Intended audience: a coauthor on the
> NeurIPS/ICML/ICLR submission picking the empirical program up cold.

- **Repo**: `LSE_RL`
- **Branch**: `phase-VIII-tab-six-games-2026-04-30`
- **HEAD at report-write time**: `52dd9a34` (V10.5 G6c review landed; v10 main pushed)
- **Pre-v10 supersession tag**: `pre-extended-grid` (HEAD `dc07737f`)
- **Spec authority**: [`docs/specs/phase_VIII_tab_six_games.md`](../../docs/specs/phase_VIII_tab_six_games.md)
  (2,224 lines; v3 → v10 + v5b + v6 + v7 amendments folded in by date order)
- **Report created**: 2026-05-02 (post-V10.5 disposition)
- **Total runs across Phase VIII**: ≈14,676 (12,951 v10 + ≈1,725 M6 + ablation
  + smoke); 0 NaN runs; **524 divergence flags** (positive-β arms; see §6, §7)

---

## 1. Executive summary

### 1.1 Problem statement

Phase VIII is the empirical anchor for the paper title **"Selective Temporal
Credit Assignment via TAB"** (Temperature-Annealed Bootstrap operator). The
suite tests whether the operator
$g_{\beta,\gamma}(r,v) = \tfrac{1+\gamma}{\beta}\big[\mathrm{logaddexp}(\beta r,\,\beta v + \log \gamma) - \log(1+\gamma)\big]$
selectively accelerates Q-learning credit propagation depending on the sign
and magnitude of $\beta$. The pre-registered claim (v2 spec §5.4 / §10.2)
was that **fixed-positive TAB** selects payoff-dominant equilibria where
vanilla Q-learning is risk-dominated, and that the **alignment condition**
$\beta\cdot(r-v)\ge 0$ is the diagnostic that identifies which sign tightens
the contraction. Phase VIII tests that claim across 30 game/opponent cells,
21 β arms, 4 γ values, and up to 10 paired seeds.

### 1.2 Headline findings

The headline list below is the consolidated narrative after V10.5 G6c
review. **Earlier per-memo claims are corrected here where the G6c re-derivation
from raw `metrics.npz` artifacts disagrees with summary memos.**

1. **v7 — bootstrap-alignment governs TAB sign, not equilibrium-payoff
   structure.** AC-Trap pre-sweep reversed the v2 §5.2 prediction in
   5/5 ablation conditions (q_init ∈ {0, 5}, ep ∈ {200, 1000}, opponents ∈
   {regret-matching, inertia(0.9), uniform stationary}). Across 22 cells × 7
   β arms × 10 seeds, +β fails to beat vanilla on any cell at γ=0.95, and
   end-of-training alignment_rate ≤ 0.07 in every fixed-+β arm (vs 0.85–0.97
   on moderate −β arms). Source: [`counter_intuitive_findings.md`](counter_intuitive_findings.md);
   [`M6_summary.md` §3](M6_summary.md);
   Codex GENUINE-FINDING: [`codex_reviews/AC_Trap_pre_sweep_review_2026-05-01T16-52-53Z.md`](codex_reviews/AC_Trap_pre_sweep_review_2026-05-01T16-52-53Z.md).

2. **Sharp β=0 sign-bifurcation at fine resolution.** Wave-5 figures-only
   sub-pass on AC-FictitiousPlay (4 β × 5 seeds × 10k ep) and V10.1
   AC-Trap smoke (21 β × 3 seeds × 1k ep) both show
   `alignment_rate[-200:]` collapsing from ~0.91 at β=−0.05 to ~0.05 at
   β=+0.05 — an 18× collapse across one Δβ=0.10 tick. The β-vs-AUC curve is
   not smooth across 0; the alignment-condition diagnostic is effectively
   binary in `sign(β)` at typical Q_init=0, γ=0.95.
   Source: [`v10_smoke_AC_Trap.md`](v10_smoke_AC_Trap.md);
   [`M6_summary.md` §4](M6_summary.md).

3. **v10 H1 — γ-induced sign flip on AC-Trap at γ=0.60 (CONFIRMED narrowly).**
   At γ=0.60 with the 21-arm β grid, AC-Trap's best β shifts from +0.00 (γ=0.95)
   to **+0.10** (γ=0.60), with paired-bootstrap 95% CI `[+0.10, +0.10]` over 5
   Tier II seeds and AUC advantage CI `[+45.20, +212.00]` over vanilla. **This
   is the first cell in the suite where +β strictly beats vanilla under any
   tested condition.** Source: V10.5 Codex G6c re-derivation
   [`codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md` §(b)](codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md).

4. **v10 H1 caveat — SH-FiniteMemoryRegret at γ=0.60 does NOT confirm.**
   The pre-V10.5 summary memo `v10_summary.md` listed SH-FMR best β = +0.35
   at γ=0.60 with AUC=106698, but G6c paired-bootstrap CI for best-β at γ=0.60
   on SH-FMR is `[-2.00, +0.35]` and the paired AUC-advantage CI is
   `[-91.50, +360.00]` — both straddle 0. **H1 is confirmed on AC-Trap only,
   not on SH-FMR.** The wider headline claim
   "2/4 headline cells show H1" in
   [`v10_summary.md` §V10.6](v10_summary.md) is over-stated and is
   superseded by the G6c re-derivation.

5. **v10 H2 — γ-induced bifurcation widening: REFUTED.**
   Among Tier II cells where −β wins at γ=0.95, 0/2 evaluable cells show
   $|d(\text{best-β}, \text{vanilla})|$ at γ=0.60 larger than at γ=0.95.
   SH-FMR ratio is 0.899; RR-StationaryConvention ratio is 0.057.
   DC-Long50 is undefined (zero seed variance under deterministic
   advance-only). Source: G6c §(c).

6. **v10 H3 — γ-stable diagnostic: REFUTED.** Only 64/120 (53.3%)
   `(γ, cell)` tuples confirm by final-episode `alignment_rate ≥ 0.5` for the
   observed best β; with last-200 robustness 60/120 (50.0%). The pre-registered
   threshold was 80%. The most damaging counter-example is the H1-positive
   AC-Trap arm itself: `fixed_beta_+0.1` at γ=0.60 has best mean AUC but
   final alignment 0.050. Source: G6c §(a).

7. **vanilla never beaten across q_init ∈ {-2, 0, +5}, γ ∈ {0.6, 0.9, 0.95, 0.99},
   α ∈ {0.05, 0.1, 0.3}.** HALT-7 Phase 4 perturbation sweep (162 runs, 87 sec,
   3 cells × 3 β × 3 seeds × 6 perturbations) found 3/18 sign flips, all at
   q_init=−2 (pessimistic init, vanilla > +β > −β); the other 15 cells
   confirmed v7. **Vanilla wins or ties in every tested perturbation.**
   At γ=0.60 on AC-Trap, the new H1 finding adds a `+β > vanilla` data point
   inside an envelope where the ablation had not probed (γ=0.60 with 5
   Tier II seeds). Source: [`v7_bug_hunt_disposition.md`](v7_bug_hunt_disposition.md).

8. **Reference-impl 0.00% diff confirms no operator artifact.** A 30-LoC
   reference TAB-Q agent with no shared imports with production matched
   production AC-FictitiousPlay AUC to 0.00% across 9 (β, seed) cells over
   both 1000-episode prefixes and full 10k-episode runs. The fixed +β
   underperformance is a mechanism finding, not an implementation artifact.
   Source: [`codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md` §4](codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md);
   reference impl at [`codex_reviews/reference_tab_agent.py`](codex_reviews/reference_tab_agent.py).

9. **DC-Long50 — bellman_residual_beta_AUC ordering with 17 orders of
   magnitude residual separation.** On the deterministic Discrete(1)
   advance-only 50-state chain at γ=0.95 with optimistic Q_0 = 1/(1−γ)=20,
   final β-specific Bellman residual $\|T_\beta Q - Q\|_\infty$:
   `R_{β=−1}=3.45e−11` (contracted), `R_{β=0}=2.45e−09` (classical
   convergence), `R_{β=+1}=2.69e+06` (divergent). AUC of $-\log R$ is
   ordered `+8660 > +4607 > −23599`. Source: HALT 4 v5b memo
   [`halts/delayed_chain_v5_d_threshold_misfire_2026-05-01T12-24-46Z.md`](halts/delayed_chain_v5_d_threshold_misfire_2026-05-01T12-24-46Z.md).

10. **524 divergence flags concentrated in positive-β AC-* and DC-Long50
    arms.** The pre-V10.5 summary memo claimed "0 divergence flags"; the
    G6c re-derivation finds 524 across 10,980 main-pass runs. All are
    confined to positive-β arms on cells where the alignment regime
    forbids +β. Documented in §7 below; this is mechanism-consistent with
    the v7 finding (q_abs_max grows past finite-horizon discounted-payoff
    bound in those arms) but the original V10.4 detector pass missed it.

11. **M7.1 — TAB never wins matrix games against the right baseline.**
    Stage 2 (M7.1, commit `722fd275`) re-dispatched Tier II at 10
    paired seeds and added 3 Q-learning baselines (`restart`,
    `sliding_window`, `tuned_epsilon_greedy`) on the same envelope
    (3,360 + 480 = 3,840 runs). Paired-bootstrap CI₉₅ on AC-Trap, RR,
    SH-FMR shows: **`tuned_epsilon_greedy_Q_learning` strictly beats
    `best_fixed_*_TAB` at every γ on all three cells**, with Δ ≈
    +11k–+44k AUC versus TAB's +26 to +320. Two orders of magnitude
    larger. **`restart_Q_learning` beats TAB on AC-Trap by ~+250k
    (~50% relative gain)** by escaping the trap via Q-reset. TAB's
    only matrix-game CI-significant wins are RR all γ (β=−0.5) and
    SH-FMR γ=0.80 (β=−0.5) — both at modest magnitudes. Source:
    [`stage2_fixed_tab_vs_baselines.md`](stage2_fixed_tab_vs_baselines.md).

12. **M7.2 — strategic-learning agents dominate Q-learning on
    payoff-anchored cells; fail catastrophically on cycling cells.**
    Stage 2 sub-milestone (M7.2, commit `7f411bf6`) added two agent
    wrappers around the existing opponent classes:
    `regret_matching_agent` and `smoothed_fictitious_play_agent`
    (240 runs at 3 cells × 4 γ × 10 seeds, DC-Long50 dropped per
    Codex P1 #2). On AC-Trap, **`regret_matching_agent` is the
    strict best method at every γ (Δ ≈ +271k AUC, ~51% relative
    gain)** — directly plays the payoff-dominant Stag/Stag. On RR,
    same pattern (Δ ≈ +39k). On SH-FMR (where the env-adversary is
    itself a regret-matcher), **strategic-learning agents fail
    catastrophically (Δ ≈ −74k for RM, −56k for FP)** due to
    Brown-Robinson cycling — exactly the spec §6.2 anticipated
    pathology. **TAB's contribution is therefore not a matrix-game
    win** — that goes to RM. Source:
    [`stage2_strategic_agents_followup.md`](stage2_strategic_agents_followup.md).

13. **M8 — sign-specialisation classifies the lattice cleanly.**
    Stage 3 analysis (M8, commit `86ed2cf7`) classifies the 16
    (cell, γ) tuples per spec §10.4 paired-CI definitions:
    **1 G_+ (AC-Trap γ=0.60), 9 G_- (RR all 4 γ, SH-FMR γ=0.80,
    DC-Long50 all 4 γ), 6 neither.** The G_+ population is narrow
    (β=+0.10, exactly the V10.9 §8.4 follow-up cell). The G_-
    population is structured: matrix games favor moderate β=−0.5;
    DC-Long50 favors extreme β=−2.0. **AC-Trap γ ∈ {0.80, 0.90,
    0.95} is vanilla-dominant** — both signs lose to vanilla,
    explaining V10's narrow H1 confirmation. **M8 → M9 acceptance
    gate MET** (≥1 G_+ and ≥1 G_-); primary M9 composite
    candidate is (AC-Trap γ=0.60 + RR γ=0.60) — both at matched γ,
    both matrix games, similar Δ-magnitude order. Source:
    [`stage3_sign_specialization.md`](stage3_sign_specialization.md).

14. **TAB's distinctive contribution is concentrated, not
    universal.** Combining M7.1, M7.2, and M8: TAB's CI-significant
    wins-vs-best-baseline reduce to **two regimes**: (a) DC-Long50
    chain task at every γ (β=−2.0; no baseline beats vanilla
    here — strategic agents have no value bootstrapping, restart
    triggers don't fire, sliding-window's eviction never triggers
    state resets); and (b) SH-FMR γ=0.80 with β=−0.5 (where
    strategic-learning vs strategic-learning cycles
    catastrophically and TAB's contraction tightening produces a
    small +97.3 paired-CI gain). **The paper headline must
    therefore be specialised, not universal.**

15. **M9 — oracle-validated composite FAILED.** The spec-mandated
    Stage 4 sign-switching composite (AC-Trap γ=0.60 ⊕ RR γ=0.60,
    the unique G_+ ⊕ G_− pair in the M8 lattice) does NOT satisfy
    the oracle-dominance gate at any tested dwell. At dwell=250
    the oracle β loses to fixed_negative_TAB by Δ=−5 192 AUC
    (CI [−8 099, −2 421]); at dwell=1000 the gap shrinks but
    still strictly favours fixed_negative (Δ=−645, CI [−1 177,
    −60]). Mechanism: shared Q-table contamination across regimes
    means fixed_negative_TAB beats every alternative on BOTH
    regimes, including the supposedly G_+ AC-Trap regime where
    M8's standalone classification said β=+0.10 wins. **Per spec
    §10.5 this halts adaptive-β experimentation.** The narrow
    Goldilocks G_+ band identified in V10.9 §8.4 (early-episode
    alignment transient) does not survive composite parameter
    transfer. Source:
    [`oracle_composite_failed.md`](oracle_composite_failed.md).

### 1.3 Refined headline narrative (post-V10.5 G6c)

> **Selective Temporal Credit Assignment via TAB is a bootstrap-alignment
> mechanism.** The operator's contraction tightness is governed by the
> instantaneous condition $\beta\cdot(r - v_{\text{next}}) \ge 0$, **not**
> by global game-theoretic equilibrium-payoff structure. Across the empirical
> envelope $\gamma\in\{0.60,\,0.80,\,0.90,\,0.95,\,0.99\}$, $q_{\text{init}}\in\{-2,\,0,\,+5\}$,
> $\alpha\in\{0.05,\,0.1,\,0.3\}$, **vanilla (β=0) is the most reliable
> default**. Moderate negative β ($-1 \le \beta \le -0.5$) ties with
> vanilla on most learning cells at γ=0.95 (alignment regime: bootstrap
> Q overshoots realized r late in training; β ≤ 0 holds the contraction).
> Moderate positive β destabilizes everywhere at γ=0.95. **At γ=0.60,
> AC-Trap admits a small but statistically significant +β advantage at
> β=+0.10**, the only data point in the suite where any fixed +β beats
> vanilla. The alignment-condition end-of-training diagnostic is correctly
> directional in 53% of (γ, cell) tuples — far below pre-registered 80%
> — so it is a **partial scope predictor**, not the universal indicator
> the v7 narrative had positioned it as.

### 1.3.1 Sharpened headline narrative (post-M7.1 / M7.2 / M8)

The M7 stage adds non-stationary Q-learning baselines (`restart`,
`sliding_window`, `tuned_epsilon_greedy`) and strategic-learning agent
baselines (`regret_matching_agent`, `smoothed_fictitious_play_agent`)
to the same paired-seed envelope. Result:

> **TAB is NOT a universal Bellman-operator improvement.** On
> stationary or payoff-anchored matrix games (AC-Trap,
> RR-StationaryConvention) the strict-best baseline is
> `regret_matching_agent` — directly playing the payoff-dominant
> equilibrium without value bootstrapping at all (Δ ≈ +271k on
> AC-Trap; +39k on RR). On Shapley with a regret-matching opponent,
> RM-agent vs RM-opponent enters Brown-Robinson cycling and
> catastrophically loses (Δ ≈ −74k); the only safe method there is
> `tuned_epsilon_greedy_Q_learning`. **TAB's distinctive contribution
> is concentrated in two regimes**: (a) the DC-Long50 deep-delayed-
> credit chain, where no baseline ever beats vanilla and only
> `fixed_beta_-2.0` does (Δ +795 to +2 837 across γ); (b) the
> SH-FiniteMemoryRegret cycling cell at γ=0.80, where `fixed_beta_-0.5`
> beats vanilla by +97.3 (CI strictly above 0) while strategic-
> learning agents fail catastrophically. **The paper claim must be
> reframed from "TAB selects payoff-dominant equilibria" (the v2
> §5.4 hypothesis, refuted at v7) and "TAB is a bootstrap-alignment
> mechanism" (the v10 reframing, partially supported at 53% scope)
> to:** **TAB is a credit-assignment mechanism for value-
> bootstrapping-required tasks (delayed reward) and learning-vs-
> learning cycling regimes — domains where simple non-stationary
> baselines either cannot fire their mechanism (DC-Long50) or fail
> catastrophically by structural symmetry (RM-vs-RM)**.

### 1.4 Diagram-worthy figures (deferred to V10.6 plotter pass)

The following five figures are recommended for the paper draft; their
generators are in `experiments/adaptive_beta/tab_six_games/analysis/` and
the underlying processed CSVs are in
[`processed/`](processed/):

1. **`beta_vs_auc_fine_grid.pdf`** — V10.1/wave-5 sharp β=0 sign-bifurcation
   on AC-FictitiousPlay × SH-FiniteMemoryRegret × AC-Trap. Use the
   21-arm grid; mark figures-only points {±0.05, ±0.10, ±0.20, ±0.35} so
   reviewers see the supplementary points are visualization-only.
2. **`gamma_beta_heatmap_4_headline.pdf`** — Tier II γ × β response surface
   on the 4 headline cells (AC-Trap, SH-FMR, RR-StationaryConvention,
   DC-Long50). The AC-Trap row should show the sign flip at γ=0.60.
3. **`alignment_rate_collapse.pdf`** — wave-2 / wave-5 alignment_rate vs
   β on a representative G_+/G_- pair, showing the 18× collapse.
4. **`q_abs_max_divergence.pdf`** — divergent positive-β arms (AC-Inertia
   `+0.5`, DC-Long50 `+1.0`) plotted against finite-horizon discounted-payoff
   bound. Use as evidence for the 524 divergence flag count.
5. **`bellman_residual_beta_dc_long50.pdf`** — log-scale residual decay
   for β ∈ {−1, 0, +1} on DC-Long50 at γ=0.95 showing the 17-order-of-magnitude
   spread; documents the v5/v5b headline-metric switch.

---

## 2. Phase VIII chronological timeline

This is a narrative timeline; per-milestone deliverables and HEAD SHAs are
in §3–§7 below.

### 2.1 M0 — kickoff (2026-04-30)

User decisions locked from `tasks/six_game_safe_TAB_harness_instructions.md`
v2:

- Phase number = **VIII**; branch
  `phase-VIII-tab-six-games-2026-04-30` cut from
  `phase-VII-B-strategic-2026-04-26`.
- Code root `experiments/adaptive_beta/tab_six_games/`; results root
  `results/adaptive_beta/tab_six_games/`.
- Phase VII Stage B2 = rerun under new β grid for end-to-end paired
  seeds; M8 cites Phase VII Stage B2 as read-only narrative reference.
- Confirmed `potential` game absent; M2 implements both `soda_uncertain`
  and `potential` games.
- UCB schedule hyperparameters approved with two refinements
  (Welford standardisation; residual_smoothing_window as config knob).
- `tasks/phase_VII_C_sign_switching_coding_agent_spec.md` declared
  superseded by Phase VIII M9.

Spec written by `planner` subagent. **Initial spec: 1,860 lines.**

### 2.2 M1 — infrastructure (2026-04-30)

Established package layout, `Phase8RunRoster`, schema headers, runners
skeleton. No empirical runs.

### 2.3 M2 reopen — game implementations (2026-05-01 morning)

Builds: `RR-Sparse` (sparse-terminal variant), `delayed_chain` (4
subcases — DC-Short10, DC-Medium20, DC-Long50, DC-Branching20),
`PassiveOpponent`, registry plumbing.

**HALT 1** (UTC 02:43:53) at HEAD `47d5353d`: T11 fired on
`test_smoke_DC_Long50_AUC_ordering`. Discrete(1) advance-only chain has
deterministic AUC across β; AUC is the wrong metric.
[Memo](halts/delayed_chain_smoke_prediction_failure_2026-05-01T02-43-53Z.md).

User chose **Option (a)**: switch to `q_convergence_rate` metric →
**spec amendment v3** (2026-05-01).

**HALT 2** (UTC 11:14:49) at HEAD `09e7a262`: T11 fired again. The v3
prediction sign was theoretically backwards; under optimistic Q_0 ≥ V*,
alignment requires β ≤ 0, not β ≥ 0.
[Memo](halts/delayed_chain_P_Contract_sign_inverted_2026-05-01T11-14-49Z.md).

User chose **Option (α)**: flip P-Contract sign → **spec amendment v4**.

**HALT 3** (UTC 11:31:32) at HEAD `10d58695`: T11 fired a third time. The
deeper issue: TAB has β-specific fixed points (Q*_β ≠ Q*_classical for
β ≠ 0); convergence to Q*_classical is intrinsically biased toward β=0
because only β=0's dynamics have Q*_classical as their fixed point.
[Memo](halts/delayed_chain_metric_design_flaw_2026-05-01T11-31-32Z.md).

User chose **Option (δ)**: switch headline metric to β-specific Bellman
residual `||T_β Q − Q||_∞` → **spec amendment v5**.

**HALT 4** (UTC 12:24:46) at HEAD `99a98340`: v5 metric and prediction
both correct (17 orders of magnitude residual separation), but the
Cohen's d ≥ 0.3 guard misfires on a deterministic testbed because intra-method
variance is exactly 0.
[Memo](halts/delayed_chain_v5_d_threshold_misfire_2026-05-01T12-24-46Z.md).

User chose **Option (η)**: replace Cohen's d guard with absolute-floor +
relative-gap floor → **spec amendment v5b** (test instrumentation only).

### 2.4 M5 — metrics + verification (2026-05-01 noon)

Adds `bellman_residual_beta`, `auc_neg_log_residual`, `q_convergence_rate`
helpers. 4+4 metric tests added. Codex M4 close review (G4b) returns
**PASS** with no BLOCKER/MAJOR/MINOR/NIT findings.
[Codex memo](codex_reviews/M4_close_review_2026-05-01T01-09-13Z.md).

### 2.5 M6 — Stage 1 fixed-β operator sweep (2026-05-01 afternoon)

#### 2.5.1 Wave 1 — runner + configs

Built `run_phase_VIII_stage1_beta_sweep.py` (916 LoC) + `dev.yaml` +
`stage1_beta_sweep.yaml` + smoke test (218 LoC). 1694 PASS + 2 SKIP suite.

**HALT 5** (UTC 12:56:56) at HEAD `5c15687c`: 4 open questions block
wave 1.5 / wave 2 / wave 4 dispatch. OQ3 (AC-Trap adversary wiring) is
methodological. OQ1 (run count: 4,340 vs 1,820) is arithmetic.
[Memo](halts/M6_wave_1_open_questions_2026-05-01T12-56-56Z.md).

User chose: OQ3 = `finite_memory_regret_matching(memory_m=20)`, OQ4 =
runner defaults, OQ1 = 1,820 confirmed → **spec amendment v6**.

#### 2.5.2 Wave 1.5 — AC-Trap pre-sweep (9 runs)

Result: `AUC(+1)=8938 < AUC(0)=10337 < AUC(-1)=10404` —
**direction REVERSED** from spec §10.2 patch §5.2 prediction. Cohen's d
`(+1, vanilla) = -3.04`. Trigger T1 + T3.

**HALT 6** (UTC 16:32:39) at HEAD `1d88e769`: AC-Trap pre-sweep
contradicts paper-headline payoff-dominance claim.
[Memo](halts/AC_Trap_pre_sweep_contradicted_2026-05-01T16-32-39Z.md).

User authorized **Option (d)**: 5-condition ablation → Codex review →
disposition. Ablation (45 runs total) confirmed reversal in 5/5
conditions.
[Memo](pre_sweep_AC_Trap_ablation.md).

Codex review verdict: **GENUINE FINDING**. No bug across operator,
agent, schedule, game, or 3 adversary classes. Mechanism is the
operator's `g_{β,γ}(r,v) → (1+γ)·max(r,v)` for β → +∞ in the
`v > r` regime, making `d_eff > 1` whenever bootstrap overshoots
realized reward.
[Codex memo](codex_reviews/AC_Trap_pre_sweep_review_2026-05-01T16-52-53Z.md).

→ **spec amendment v7**: AC-Trap repositioned as falsifiability cell;
§13.10 negative-result honesty norm added; counter-intuitive findings
memo opened.

#### 2.5.3 Wave 2 — Stage A dev pass (378 runs)

378 runs (18 cells × 7 β × 3 seeds × 1k ep). Pattern from wave 1.5
extends to all cells: best-β is 0 or negative on 17/18 cells (sole
exception SO-AntiCoordination ties at high β within float-precision).
End-of-training `alignment_rate[+β]` ≤ 0.18 in every cell. The
diagnostic correctly predicts +β outside-regime on every non-trivial
cell. Documented as "cross-cell extension of v7" in
[`counter_intuitive_findings.md`](counter_intuitive_findings.md).

#### 2.5.4 Wave 4 — Stage 1 main pass (1,540 + 140 recovery)

Dispatched 1,400 main runs + 140 RR-Sparse recovery runs (the original
runner had a `rules_of_road_sparse → rules_of_road` registry alias bug
that prevented payoff-opponent construction; fixed at HEAD with
`_PAYOFF_ALIAS` mapping).

Wall-clock: 95 min main + 12 min recovery. 0 NaN, 0 divergence flags
(per V10.4 detector pass at the time; see G6c update — divergence flags
later found in re-derivation against raw `metrics.npz`).

#### 2.5.5 Wave 5 — figures-only β grid (40 runs)

40 runs × 10k ep × 5 seeds × 4 β values × 2 cells (AC-FictitiousPlay
G_+ slot + SH-FiniteMemoryRegret G_-). β ∈ {−0.25, −0.10, +0.10, +0.25}.

Result: sharp β=0 sign-bifurcation, alignment 0.91 → 0.05 across one
Δβ=0.10 tick, AUC penalty 18-22% at smallest +β. Validated in V10.1.

#### 2.5.6 Wave 6 — T1-T11 detector pass

99 trigger fires across 1,400 main runs:
- T1 (5 fires): all on `MP-Stationary` null cell (spec §13.9).
- T2 (87 fires): tight σ on deterministic / paired-seed-stable cells
  (21 on DC advance-only — expected per v5b lesson; 66 on matrix-game
  cells with stationary/inertia opponents).
- T3 (7 fires): non-monotone β-grid; v7-expected ∪-shape under
  positive-β destabilization.

All 99 fires are spec-consistent; no Codex bug-hunt dispatched.
[Memo](M6_wave_6_T_detector.md).

#### 2.5.7 M6 close

[`M6_summary.md`](M6_summary.md) committed at HEAD `2dcb92be` with
22-cell × 7 β × 10 seeds full table.

**HALT 7** (UTC 19:06:56) at HEAD `2dcb92be`: spec-mandated user
sign-off boundary (§10.2 acceptance: "User signs off on the promoted
subcases for M7"). Plus the v7 finding reshapes M7's design — there is
no `best_fixed_positive_TAB` baseline because no cell has fixed +β
beating vanilla.
[Memo](halts/M6_close_M7_user_signoff_2026-05-01T19-06-56Z.md).

User authorized 5-phase pre-M7 broad bug-hunt before deciding M7 scope.

### 2.6 Pre-M7 broad bug-hunt — 5-phase verification (2026-05-01 evening)

5-phase verification of the v7 finding before proceeding to M7. See §6
for the full procedure.

- **Phase 1**: 1694 PASS suite, zero operator-touch since Phase VII,
  manifest sound, AC-Trap pre-sweep bit-identical reproducibility.
- **Phase 2**: manual AUC + alignment-rate replays bit-identical;
  β=0 bit-identity guard verified at operator and agent levels;
  cross-cell sign-of-β test: alignment(+β)<0.20 in 22/22 cells, 100%;
  AUC(−β)>AUC(+β) in 14/16 non-tied cells, 88%.
- **Phase 3**: Codex broad adversarial review returned **GENUINE FINDING**.
  Reference TAB agent (~30 LoC, no shared imports) matched production
  AUC to 0.00% across 9 cells over 1k+10k episodes.
  [Codex memo](codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md).
- **Phase 4**: 162 perturbation runs (γ ∈ {0.9, 0.99}, q_init ∈ {−2, +5},
  α ∈ {0.05, 0.3}) on 3 cells × 3 β × 3 seeds × 1k ep. v7 holds in
  15/18 cells; 3/18 flips, all at q_init=−2 (vanilla > +β > −β).
- **Phase 5**: disposition memo
  [`v7_bug_hunt_disposition.md`](v7_bug_hunt_disposition.md) records
  CASE C: NO BUG, refined narrative ("TAB sign should match
  $\mathrm{sign}(r-V^*)$ in expectation").

User authorized v10 dispatch (extended β grid + γ sweep + 30-cell
enumeration) instead of M7.

### 2.7 V10.0 — V10.9: extended β grid + γ-sweep + full enumeration (2026-05-01 evening → 2026-05-02 noon)

Spec amendment v10 folded in:
- 21-arm β grid `BETA_GRID_V10` (denser near 0; resolves the wave-5 β=0
  bifurcation).
- 4-value γ grid `GAMMA_GRID_V10 = {0.60, 0.80, 0.90, 0.95}` to test
  alignment-condition theory's prediction that lower γ makes the +β
  regime accessible.
- Promoted-subcase enumeration 22 → 30 cells (adds AC-Inertia,
  MP-RegretMatching, MP-HypothesisTesting, RR-ConventionSwitch,
  RR-HypothesisTesting, SH-SmoothedFP, SH-HypothesisTesting,
  SO-ZeroSum, SO-BiasedPreference, PG-Congestion, PG-BetterReplyInertia).
- Tier structure: I (canonical γ=0.95, full β, 10 seeds) + II (γ × β
  on 4 headline cells, 5 seeds) + III (γ × coarse β at all 30 cells,
  5 seeds).
- H1 / H2 / H3 pre-registered as §13.11 characterization criteria.

#### V10 phase-by-phase

| phase | dispatch | runs | wall-clock | output memo |
|---|---|---:|---|---|
| V10.0 | binding probe | 18 | <5 min | (informal) |
| V10.1 | AC-Trap smoke (1 cell × 21 β × 3 seeds × 1k) | 63 | 43 sec | [`v10_smoke_AC_Trap.md`](v10_smoke_AC_Trap.md) |
| V10.2 | dev pass (Tier I scale, 3 seeds × 1k) | 1,890 | 15 min | (folded into V10.3 prep) |
| V10.3a | Tier I main (30 × 21 × 10 × 10k) | 6,300 | ~7 h | [`v10_summary.md`](v10_summary.md) |
| V10.3b | Tier II γ×β headline (4 × 21 × 4 × 5 × 10k) | 1,680 | ~2.5 h | (same) |
| V10.3c | Tier III γ×cell coverage (30 × 5 × 4 × 5 × 10k) | 3,000 | ~3 h | (same) |
| V10.4 | T1-T11 deterministic detector pass | 0 (analysis) | <1 min | (in v10_summary.md) |
| V10.5 | Codex G6c milestone-close review | 0 (analysis) | ~30 min | [`codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md`](codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md) |
| V10.6 | figures (5 PDFs) | — | deferred | (TBD) |
| V10.7 | supersession memo | — | deferred | (TBD) |
| V10.8 | 5-phase bug-hunt re-validation on v10 | — | deferred | (TBD) |
| V10.9 | this report | 0 | ~1 h | this file |

V10 totals: **12,951 runs**, **~17 h wall-clock sequential** (12:50 UTC May 1
→ 12:40 UTC May 2). Pre-v10 main pass at HEAD `dc07737f` preserved on
origin under tag `pre-extended-grid` for provenance.

V10.5 G6c review (just landed, HEAD `52dd9a34`) is the most
authoritative analysis: it re-derives every claim from raw `metrics.npz`
artifacts and finds three MAJOR corrections to `v10_summary.md`
(divergence count, H1 narrowing to AC-Trap only, H2/H3 refutation).
**This report folds those corrections into the headline narrative.**

---

## 3. Environment + opponent inventory (30-cell enumeration)

The 30 cells span 7 game families × variable subcases. Each cell has a
canonical sign tag (per spec §22, encoding the pre-registered
G_+ / G_- claim before v7's refinement) and a headline metric.

Notation:
- **Sign**: pre-v7 canonical sign per spec §22; **{x}** denotes "v7
  reframed as falsifiability cell".
- **Metric**: `cum_ret_AUC` = cumulative-return AUC (default);
  `bres_β_AUC` = β-specific Bellman-residual AUC (deterministic
  delayed-chain advance-only cells per v5).
- **T11 guard**: `cohen_d` for stochastic cells; `gap_based` (absolute
  floor + relative-gap floor) for deterministic per v5b.

| cell | game | subcase | adversary | key params | sign | metric | T11 | role |
|---|---|---|---|---|---|---|---|---|
| 1 | asymmetric_coordination | AC-FictitiousPlay | FMFP | memory_m=20 | + | cum_ret_AUC | cohen_d | coordination learning |
| 2 | asymmetric_coordination | AC-SmoothedBR | FMSBR | memory_m=20, τ=0.2 | + | cum_ret_AUC | cohen_d | smoothed BR coordination |
| 3 | asymmetric_coordination | AC-Trap | FMRM | memory_m=20 | **{+}** | cum_ret_AUC | cohen_d | falsifiability (v7); stag-hunt with regret-matching opponent |
| 4 | asymmetric_coordination | AC-Inertia | inertia | inertia_λ=0.9 | + | cum_ret_AUC | cohen_d | sticky-opponent coordination (v10 new) |
| 5 | matching_pennies | MP-Stationary | stationary | probs=[0.5, 0.5] | null | cum_ret_AUC | cohen_d | null-cell sanity (v2 §13.9) |
| 6 | matching_pennies | MP-FiniteMemoryBR | FMBR | memory_m=20 | − | cum_ret_AUC | cohen_d | second-order MP, learning opponent |
| 7 | matching_pennies | MP-RegretMatching | regret_matching | memory_m=20 | − | cum_ret_AUC | cohen_d | second-order MP, regret-matching opp (v10 new) |
| 8 | matching_pennies | MP-HypothesisTesting | hypothesis_testing | memory_m=20 | − | cum_ret_AUC | cohen_d | second-order MP, HT opp (v10 new) |
| 9 | rules_of_road | RR-StationaryConvention | stationary | probs=[0.7, 0.3] | + | cum_ret_AUC | cohen_d | convention learning |
| 10 | rules_of_road | RR-Tremble | stationary | probs=[0.7, 0.3], tremble_prob=0.05 | + | cum_ret_AUC | cohen_d | tremble-perturbed convention |
| 11 | rules_of_road | RR-ConventionSwitch | conv_switch | dwell=500 | + | cum_ret_AUC | cohen_d | nonstationary convention (v10 new) |
| 12 | rules_of_road | RR-HypothesisTesting | hypothesis_testing | memory_m=20 | + | cum_ret_AUC | cohen_d | HT convention probe (v10 new) |
| 13 | rules_of_road_sparse | RR-Sparse | various | sparse_terminal=True, H=20 | + | cum_ret_AUC | cohen_d | sparse-reward convention (M2 reopen new) |
| 14 | shapley | SH-FictitiousPlay | FMFP | memory_m=20 | − | cum_ret_AUC | cohen_d | cyclic non-zero-sum |
| 15 | shapley | SH-SmoothedFP | FMSFP | memory_m=20, τ=0.2 | − | cum_ret_AUC | cohen_d | smoothed cyclic (v10 new) |
| 16 | shapley | SH-FiniteMemoryRegret | FMRM | memory_m=20 | − | cum_ret_AUC | cohen_d | cyclic regret-matching |
| 17 | shapley | SH-HypothesisTesting | hypothesis_testing | memory_m=20 | − | cum_ret_AUC | cohen_d | cyclic HT (v10 new) |
| 18 | soda_uncertain | SO-Coordination | stationary | uniform | + | cum_ret_AUC | cohen_d | hidden-type coordination |
| 19 | soda_uncertain | SO-AntiCoordination | stationary | uniform | − | cum_ret_AUC | cohen_d | hidden-type anti-coord |
| 20 | soda_uncertain | SO-ZeroSum | stationary | uniform | none | cum_ret_AUC | cohen_d | hidden-type zero-sum (v10 new) |
| 21 | soda_uncertain | SO-BiasedPreference | stationary | biased | none | cum_ret_AUC | cohen_d | hidden-type biased (v10 new) |
| 22 | soda_uncertain | SO-TypeSwitch | type_switch | dwell=500 | none | cum_ret_AUC | cohen_d | regime-switching coordination |
| 23 | potential | PG-CoordinationPotential | stationary | uniform | + | cum_ret_AUC | cohen_d | potential-game positive control |
| 24 | potential | PG-Congestion | stationary | uniform | + | cum_ret_AUC | cohen_d | congestion potential (v10 new) |
| 25 | potential | PG-BetterReplyInertia | inertia | inertia_λ=0.9 | + | cum_ret_AUC | cohen_d | better-reply inertia (v10 new) |
| 26 | potential | PG-SwitchingPayoff | switching | dwell=500 | + | cum_ret_AUC | cohen_d | switching potential |
| 27 | delayed_chain | DC-Short10 | passive | L=10 | + | bres_β_AUC | gap_based | short-horizon credit |
| 28 | delayed_chain | DC-Medium20 | passive | L=20 | + | bres_β_AUC | gap_based | medium-horizon credit |
| 29 | delayed_chain | DC-Long50 | passive | L=50 | + | bres_β_AUC | gap_based | long-horizon credit (paper-headline) |
| 30 | delayed_chain | DC-Branching20 | passive | L=20, n_actions=2 | + | cum_ret_AUC | cohen_d | choice + temporal credit |

The 30-cell enumeration is the v10 superset; the pre-v10 22-cell main
pass omits cells (4, 7, 8, 11, 12, 15, 17, 20, 21) plus delayed_chain
advance-only cells running on the cumulative-return AUC headline (per
v5 metric switch).

`AC-Trap`'s adversary wiring (`finite_memory_regret_matching, memory_m=20`)
was the user's HALT-5 OQ3 resolution. AC-Trap, RR-StationaryConvention,
SH-FiniteMemoryRegret, and DC-Long50 are the **4 headline cells** for
the Tier II γ × β heatmap.

---

## 4. Methods

### 4.1 TAB operator

Phase VIII inherits the operator unchanged from Phase VII §3.1:

```text
β ≠ 0:  g_{β,γ}(r, v) = (1+γ)/β · [logaddexp(β·r, β·v + log γ) − log(1+γ)]
β = 0:  g_{0,γ}(r, v) = r + γ·v   (classical Bellman target)
```

Effective continuation:

```text
ρ_{β,γ}(r, v) = sigmoid(β·(r − v) − log γ)
d_{β,γ}(r, v) = ∂_v g_{β,γ} = (1+γ) · (1 − ρ_{β,γ}(r, v))
```

Asymptotic forms (load-bearing for the v7 mechanism):

```text
β → +∞:  g → (1+γ)·max(r, v)
β → −∞:  g → (1+γ)·min(r, v)
```

When `v > r` (bootstrap above realized reward) and β > 0 large,
`d_eff → 1+γ > 1` for γ < 1 close to 1 — i.e. the operator amplifies
overshoot rather than damping it. Negative β in the same regime sends
`d_eff → 0`. This asymmetry is the mechanism behind the
v7 universal-collapse-of-+β finding.

**Alignment condition** (spec §3.3, §4.3):

```text
d_{β,γ}(r, v) ≤ γ   ⇔   β · (r − v) ≥ 0
```

Per-step diagnostic: `aligned_t = (β · (r_t - v_{next, t}) > 0)`. Episode
diagnostic: `alignment_rate = mean_t(aligned_t)`. End-of-training
`alignment_rate[-200:]` is the headline diagnostic signal.

### 4.2 Single-source kernel

```python
src/lse_rl/operator/tab_operator.py
  g(beta, gamma, r, v) -> float
  rho(beta, gamma, r, v) -> float
  effective_discount(beta, gamma, r, v) -> float
  g_batch / rho_batch / effective_discount_batch
  _EPS_BETA = 1e-8                       # classical-collapse threshold
  _is_classical(beta) -> bool            # |β| ≤ 1e-8
```

Both Phase III–VI (DP planners via `SafeWeightedCommon`) and Phase VIII
(`AdaptiveBetaQAgent`) import from this module. Phase VIII does not
edit the kernel. The Codex v7 broad bug-hunt confirms zero
operator-touch since Phase VII commit `6692f0f5`
([memo](codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md)).

### 4.3 β grids and γ grid

**Pre-v10 (M6 main pass, deprecated):** `[-2, -1, -0.5, 0, +0.5, +1, +2]`
(7 arms).

**v10 (BETA_GRID_V10, 21 arms in [−2, +2]):**

```yaml
BETA_GRID_V10:
  [-2.00, -1.70, -1.35, -1.00, -0.75, -0.50, -0.35, -0.20, -0.10, -0.05,
    0.00,
   +0.05, +0.10, +0.20, +0.35, +0.50, +0.75, +1.00, +1.35, +1.70, +2.00]
```

Spacing is ~0.05 near 0 and ~0.30 near ±2. Designed to resolve the
wave-5 β=0 sign-bifurcation with paper-quality fidelity.

**v10 (GAMMA_GRID_V10):** `[0.60, 0.80, 0.90, 0.95]`. Lower γ → smaller
discount-sum bound on V → `r > v` regime more accessible early in
training → β > 0 has greater alignment fraction by construction.

### 4.4 Agent

`AdaptiveBetaQAgent` (`experiments/adaptive_beta/agents.py`) is a
tabular Q-learning agent with:

- Per-step TD update via the shared kernel `g(β, γ, r, v_next)`.
- β cached once per episode via `BetaSchedule.beta_for_episode(e)`
  (deterministic; no per-step β changes in fixed-β runs).
- ε-greedy action selection (linear decay 1.0 → 0.05 over 5,000 ep
  default).
- v_next from `max_a Q[next_state, a]`; absorbing transitions use
  `v_next = 0.0`.
- β=0 bit-identity guard at the per-step level: when |β| ≤ 1e-8, the
  TD target falls back to `r + γ·v` exactly; deviation raises
  `AssertionError`. Regression-tested in `test_beta0_collapse_preserved.py`
  for every schedule.
- `Q[state, action]` is the only mutated cell per step (verified by
  Codex grep, 0 rogue write paths).

### 4.5 Schedules

```text
ZeroBetaSchedule              METHOD_VANILLA              β_e ≡ 0
FixedBetaSchedule(+1, β0)     METHOD_FIXED_POSITIVE       β_e ≡ +β0
FixedBetaSchedule(-1, β0)     METHOD_FIXED_NEGATIVE       β_e ≡ -β0
WrongSignSchedule             METHOD_WRONG_SIGN           negative-control
AdaptiveBetaSchedule          METHOD_ADAPTIVE_BETA        sign(Â)·magnitude
AdaptiveBetaSchedule(no_clip) METHOD_ADAPTIVE_BETA_NO_CLIP
AdaptiveSignOnlySchedule      METHOD_ADAPTIVE_SIGN_ONLY
AdaptiveMagnitudeOnlySchedule METHOD_ADAPTIVE_MAGNITUDE_ONLY
OracleBetaSchedule            METHOD_ORACLE_BETA          reads info["regime"]
HandAdaptiveBetaSchedule      METHOD_HAND_ADAPTIVE_BETA   pre-registered rule
ContractionUCBBetaSchedule    METHOD_CONTRACTION_UCB_BETA UCB on contraction reward
ReturnUCBBetaSchedule         METHOD_RETURN_UCB_BETA      UCB on return
```

UCB schedules use Welford-standardised reward streams (per spec §6.5
Refinement 1). ContractionUCB uses `c=1.0`; ReturnUCB uses `c=√2`.
Warm-start is one forced pull per arm (7 episodes pre-v10, 21 post-v10
per spec §6.4 v10). UCB schedules are exercised in M9/M10 only; M6 / v10
use `vanilla` + `FixedBetaSchedule` arms.

### 4.6 Per-episode metrics

From spec §7.1–§7.2:

```text
return, length, epsilon
alignment_rate, mean_signed_alignment, mean_advantage, mean_abs_advantage
mean_d_eff, median_d_eff, frac_d_eff_below_gamma, frac_d_eff_above_one
bellman_residual, td_target_abs_max, q_abs_max
nan_count, divergence_event
contraction_reward, empirical_contraction_ratio, log_residual_reduction
ucb_arm_count, ucb_arm_value, beta_clip_count, beta_clip_frequency
recovery_time_after_shift, beta_sign_correct, beta_lag_to_oracle
regret_vs_oracle, catastrophic_episodes, worst_window_return_percentile
trap_entries, constraint_violations, overflow_count
beta_used, beta_raw, effective_discount_mean, goal_reaches  (v6 OQ2 added)
```

Two delayed-chain-specific helpers (added in v3/v5): `q_convergence_rate(q_hist, q_star)`
and `bellman_residual_beta(Q, T_β)`. The aggregator routes the
`headline_metric` per cell (v5b): `bellman_residual_beta_AUC` for the
3 advance-only delayed-chain cells; `cumulative_return_AUC` everywhere
else.

---

## 5. Headline empirical findings

### 5.1 v7 — bootstrap-alignment governs TAB sign

**Setting**: 22 cells × 7 β arms × 10 seeds × 10k ep at γ=0.95, q_init=0,
ε linear decay 1.0 → 0.05 over 5,000 ep, α=0.1.

**Key result** (M6 main pass; full table in
[`M6_summary.md` §3](M6_summary.md#3-per-cell-summary-table-stage-1-main-pass-22-cells--7-β-arms--10-seeds)):

- **No cell has fixed +β beating vanilla.** Best β is 0 or negative on
  17/18 non-degenerate cells (the exception is SO-AntiCoordination
  where +2 ties with -2 and 0 within float precision).
- End-of-training `alignment_rate[+β]` ≤ 0.07 on every fixed-+β arm
  across all 22 cells.
- −β with moderate magnitude (β ∈ [−1, −0.5]) is the safest non-zero
  choice; alignment 0.85–0.97 on most learning cells.
- +β collapse magnitude varies: 18-22% AUC penalty at fine grid (V10.1
  AC-Trap, β=+0.05 → AUC=47323 vs vanilla 52451); up to 80% at coarse
  grid on extreme cells (RR-Tremble +β=2 → AUC=8302 vs vanilla 50239,
  Cohen's d = -28.97).

**Mechanism**: at typical optimistic-or-zero Q-init with γ=0.95,
ε-greedy + bootstrap-Q-learning drives Q[s, a] above the per-step
realized r within the first few hundred episodes on most learning
cells. Once Q overshoots, `r - v_next < 0` for most steps, so
alignment requires β ≤ 0. +β with `v > r` makes
`d_eff = (1+γ)(1-ρ_{β,γ}) → 1+γ > 1` (γ=0.95 → 1.95), and the operator
amplifies the overshoot rather than damping it.

**Diagnostic accuracy**: end-of-training `alignment_rate` correctly
predicts +β as outside-regime in **22/22 cells** (100%); AUC(−β) > AUC(+β)
in **14/16 non-tied cells** (88%). The 6 tied cells are the
deterministic delayed-chain advance-only and RR-Sparse-{Stationary,
Tremble} null cells.

### 5.2 Sharp β=0 sign-bifurcation at fine resolution

**Setting**: V10.1 smoke (1 cell × 21 β arms × 3 seeds × 1k ep on
AC-Trap at γ=0.95) and wave-5 figures-only (2 cells × 4 β × 5 seeds ×
10k ep).

**V10.1 result** (full table in
[`v10_smoke_AC_Trap.md`](v10_smoke_AC_Trap.md#β-auc-curve-full-v10-grid-mean--std-over-3-seeds)):

| β     | mean AUC | std    | align[-200:] | regime |
|------:|---------:|-------:|---------:|---|
| -0.05 |  52640.0 |  120.5 | 0.9009 | IN |
| +0.00 |  52450.7 |   30.2 | 0.0000 | (boundary; β=0) |
| +0.05 |  47322.7 | 2970.7 | 0.0486 | OUT |

The β=0 → β=+0.05 step yields:
- AUC drop of ~9.8% (52450.7 → 47322.7).
- alignment_rate from 0.000 (vanilla — no signed alignment) to 0.0486
  (well below 0.5 threshold).
- std growth from 30 to 2970 (98×) — destabilization signature.

The wave-5 finer-grain on AC-FictitiousPlay reveals the sharper jump:
across β ∈ {−0.05 (extrapolated from -0.10), +0.05 (extrapolated from
+0.10)}, alignment_rate goes from 0.91 to 0.05 — an 18× collapse —
and AUC penalty is 18-22%.

**Implication**: the alignment-condition diagnostic is effectively
binary in `sign(β)` at typical Q_init=0 / γ=0.95. There is no smooth
transition through β=0; the contraction property switches sign
discretely. This is the paper-relevant operator-diagnostic figure for
the V10.6 figure pass.

### 5.3 Cross-cell universality of +β alignment-rate collapse

**Setting**: 18 cells × 3 β arms × 3 seeds × 1k ep (Stage A dev pass).

**Result** (full table in
[`counter_intuitive_findings.md`](counter_intuitive_findings.md#wave-2-outcome-2026-05-01--extends-to-all-cells)):

`alignment_rate[+β=+1, last 200 ep]` ≤ 0.18 in **every** cell at Stage
A budget, and ≤ 0.10 in 16/18 cells. The two outliers (DC-Branching20
0.18; MP-Stationary 0.10) are deterministic / null cells.

The pattern is universal: under γ=0.95 with ε-greedy + Q-learning
from neutral q_init=0, the late-training regime is
`v_next > r` essentially everywhere. +β anti-aligns; alignment_rate
collapses; AUC penalty manifests on stochastic learning cells with
magnitude correlated with cell's coordination-difficulty
(matrix-game cells: 30-80%; potential / soda: < 5%; sparse-reward: ~0%).

### 5.4 v5b — DC-Long50 bellman_residual_beta with 17 orders of magnitude residual separation

**Setting**: 1 cell (DC-Long50, deterministic Discrete(1) advance-only,
γ=0.95, optimistic Q_0 = 1/(1−γ) = 20) × 3 β ∈ {-1, 0, +1} × 3 seeds ×
1k ep.

**Result**:

| schedule | β | AUC(-log R_β) | final R_β = ||T_β Q − Q||_∞ |
|---|---:|---:|---:|
| FixedNeg | -1 | +8660.54 | 3.45 × 10⁻¹¹ (contracted) |
| ZeroBeta |  0 | +4606.85 | 2.45 × 10⁻⁹  (classical convergence) |
| FixedPos | +1 | -23598.95 | 2.69 × 10⁺⁶  (divergent) |

Final residuals are separated by **17 orders of magnitude** between
β = −1 and β = +1. The directional ordering
`AUC(R_{β=−1}) > AUC(R_0) > AUC(R_{β=+1})` is correct in sign per the
v4 alignment-condition derivation (with optimistic Q_0 = 20, every
non-terminal step has `r=0 < v=20`, so β ≤ 0 satisfies alignment).

**Source**:
[`halts/delayed_chain_v5_d_threshold_misfire_2026-05-01T12-24-46Z.md` §1-§3](halts/delayed_chain_v5_d_threshold_misfire_2026-05-01T12-24-46Z.md).

The large effect sizes vindicate the v5 metric switch (from
`q_convergence_rate(Q, Q*_classical)` to `bellman_residual_beta`) and
the v4 sign flip (from "+β tightens" to "−β tightens" in the
optimistic-init regime). DC-Long50 becomes the paper's headline
long-horizon temporal-credit-assignment cell.

The full M6 main-pass table (10 seeds) extends this:

| subcase | bres_β AUC at β=−2 | β=−1 | β=−0.5 | β=0 | β=+0.5 | β=+1 | β=+2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DC-Short10  | 181308 | 181098 | 180916 | 180594 | 179779 | 178929 | 178532 |
| DC-Medium20 | 180094 | 179796 | 179522 | 178945 | 175116 | 173260 | 172468 |
| **DC-Long50** | **177304** | **176786** | **176305** | **175161** | **83237** | **47845** | **34122** |

DC-Long50 shows the v5b T11-confirmed cliff: β=+1 collapses AUC to 27%
of vanilla, β=+2 to 19%. The +β cliff dwarfs the −β plateau gap, so
v5b's relative-gap floor (`gap_small ≥ 0.10·gap_large`) intentionally
fires as a documented signature, not a halt.

### 5.5 v10 H1 — γ-induced sign flip on AC-Trap at γ=0.60 (CONFIRMED narrowly)

**Setting**: Tier II γ × β heatmap, 4 headline cells × 21 β arms × 4 γ
values × 5 seeds × 10k ep = 1,680 runs.

**Pre-V10.5 claim** (v10_summary.md): 2/4 headline cells confirm H1
(AC-Trap +0.10 and SH-FMR +0.35 both strictly positive at γ=0.60).

**Post-V10.5 G6c re-derivation** (the authoritative version):

| cell | observed best β at γ=0.60 | mean AUC (best) | mean AUC (vanilla) | best-β bootstrap CI | P(best β > 0) | paired AUC-advantage CI | verdict |
|---|---:|---:|---:|---|---:|---|---|
| AC-Trap                 | **+0.10** | 529077.80 | 528949.20 | **[+0.10, +0.10]** | **0.998** | **[+45.20, +212.00]** | **CONFIRMS H1** |
| SH-FiniteMemoryRegret   | +0.35     | 106698.30 | 106570.80 | [-2.00, +0.35]    | 0.514 | [-91.50, +360.00] | NOT CONFIRMED |
| RR-StationaryConvention | -0.50     |  56293.20 |  56273.60 | [-0.75, -0.50]    | 0.000 | [+6.40, +30.80]   | NOT POSITIVE |
| DC-Long50               | -2.00     | 182029.46 | 181569.81 | [-2.00, -2.00]    | 0.000 | [+459.65, +459.65] (zero seed variance) | NOT POSITIVE |

**Disposition**: H1 is **CONFIRMED narrowly on AC-Trap only**. SH-FMR
mean is positive but its bootstrap CI on best-β straddles 0 and the
paired AUC-advantage CI also straddles 0, so SH-FMR cannot be cited
as an H1 confirmation.

**Mechanism caveat**: AC-Trap's confirming arm (`fixed_beta_+0.1` at
γ=0.60) has end-of-training `alignment_rate = 0.050` — well below the
0.5 threshold the alignment-condition diagnostic requires. The H1 sign
flip is therefore **a small statistical surface feature, not a
mechanism vindication**. The G6c memo recommends not attributing
AC-Trap +0.10 to the alignment diagnostic without a separate mechanism
test.

**Tier II γ × β trajectory** (best β at each γ on AC-Trap; pre-V10.5):

| γ      | AC-Trap best β | AUC at best β (mean) |
|-------:|---:|---:|
| 0.95   | +0.00 (vanilla) | 528943 |
| 0.90   | +0.00 (vanilla) | 529040 |
| 0.80   | +0.00 (vanilla) | 528951 |
| **0.60** | **+0.10** | **529078** |

The transition from γ=0.80 to γ=0.60 is where the small-positive arm
emerges. The +0.10 AUC advantage at γ=0.60 is +128.60 AUC ≈ 0.024% over
vanilla — small in absolute terms but statistically robust under
paired-bootstrap.

**Caveat — divergence concentration**: AC-Trap at γ=0.95 has 24 Tier II
divergent runs (per G6c integrity scan); at γ=0.60 it has 0. The
γ-induced regime transition is therefore confounded with the
divergence-concentration transition: at lower γ, fewer +β arms diverge,
which mechanically shifts the best-β toward small positive values.
The diagnostic shape is not pure alignment-condition vindication.

### 5.6 vanilla-never-beaten robustness across q_init / γ / α

**Setting**: HALT-7 Phase 4 perturbation sweep, 162 runs total
(6 perturbations × 3 cells × 3 β arms × 3 seeds × 1k ep).

**Result** (full table in
[`v7_bug_hunt_disposition.md` §Phase 4](v7_bug_hunt_disposition.md#phase-4--parameter-perturbation-sweep-162-runs-in-87-sec)):

| perturbation | flips | direction (when flipped) |
|---|---:|---|
| γ = 0.9              | 0/3 | n/a (v7 holds) |
| γ = 0.99             | 0/3 | n/a |
| q_init = -2          | **3/3** | vanilla > +β > −β |
| q_init = +5          | 0/3 | n/a |
| α = 0.05             | 0/3 | n/a |
| α = 0.3              | 0/3 | n/a |

**Disposition**: in every tested perturbation (3 cells × 6 perturbations
= 18 conditions), vanilla wins or ties. Under pessimistic q_init = -2,
the alignment regime is mirrored early in training (β > 0 aligns when
Q < V*), but mid-training Q grows past V* and the mirror flips; net AUC
shows vanilla > +β > −β. The end-of-training alignment_rate is the same
late-training signature in both regimes (β > 0 anti-aligned at 0.05);
**AUC integrates across regime phases and disagrees with end-of-training
alignment_rate under transient pessimistic init**.

**Caveat — γ=0.60 not in HALT-7 envelope**: the V10.3b/V10.5 H1 finding
on AC-Trap at γ=0.60 (best β = +0.10) extends the envelope: at γ=0.60
on AC-Trap, vanilla is beaten by +0.10 by ~128.60 AUC. This is **the
single data point in Phase VIII where any fixed +β beats vanilla**
under any tested combination of γ × q_init × α × cell. Caveats from §5.5
apply (alignment_rate at the winning arm < 0.5; mechanism is not pure
alignment).

### 5.7 v10 H2 — γ-induced bifurcation widening: REFUTED

**Pre-registered**: for cells where −β wins at γ=0.95,
$|d(\text{best-β}, \text{vanilla})|$ at γ=0.60 should be larger than at
γ=0.95.

**Result** (G6c §(c)):

| cell | best β γ=0.95 | d_{0.95} | best β γ=0.60 | d_{0.60} | ratio | verdict |
|---|---:|---:|---:|---:|---:|---|
| AC-Trap                 | +0.00 | 0.000 | +0.10 | 0.274 | n/a | not eligible (γ=0.95 best is not negative) |
| SH-FiniteMemoryRegret   | -1.70 | 0.806 | +0.35 | 0.725 | **0.899** | not widened |
| RR-StationaryConvention | -0.50 | 0.812 | -0.50 | 0.047 | **0.057** | not widened |
| DC-Long50               | -2.00 | n/a   | -2.00 | n/a   | n/a | inconclusive (zero seed variance) |

**Disposition**: 0/2 evaluable cells widen. H2 is REFUTED. The pre-v10
narrative had assumed lower γ would produce stronger β-effects on
existing −β-winners; empirically, lower γ produces **smaller** effect
sizes on RR-StationaryConvention (the most striking example: ratio
0.057 means d at γ=0.60 is ~5% of d at γ=0.95).

### 5.8 v10 H3 — γ-stable diagnostic: REFUTED

**Pre-registered**: the alignment condition `β·(r−v) ≥ 0` correctly
predicts best-β (in the sense of `alignment_rate[best β] ≥ 0.5` at
end-of-training) on ≥ 80% of (γ, cell) tuples.

**Result** (G6c §(a) summary):

| γ | confirms / 30 (final) | confirms / 30 (last200) |
|---:|---:|---:|
| 0.60 | 11/30 | 10/30 |
| 0.80 | 14/30 | 12/30 |
| 0.90 | 17/30 | 16/30 |
| 0.95 | 22/30 | 22/30 |

**Disposition**: 64/120 = 53.3% (final) and 60/120 = 50.0% (last200) —
both below the 80% threshold. H3 is REFUTED. The diagnostic is reliable
at γ=0.95 (22/30 = 73%, close to threshold) but degrades sharply at
lower γ; **the diagnostic and the AUC-best-β disagree in 47% of (γ,
cell) tuples across the full envelope**. The most damaging
counter-example is the H1-positive AC-Trap arm itself: best β = +0.10
at γ=0.60 has alignment_rate = 0.050.

The diagnostic remains a **scope-correct local-bootstrap-optimality
indicator**: it identifies whether the late-training steady state is
in the alignment regime, not whether the AUC across all training
phases is dominated by the late-training regime. With ε-decay and
short horizons, the early-training transient frequently dominates AUC
on stochastic cells, decoupling alignment_rate at end-of-training from
AUC ranking.

---

## 6. Bug-hunt verification (HALT 7 — 5-phase verification)

The pre-M7 broad bug-hunt is the strongest verification cycle in
Phase VIII. Triggered by HALT 7 user gate; 5 phases run sequentially.

### Phase 1 — deterministic regression checks

| check | result |
|---|---|
| P1.1 Full test suite | 1694 PASS + 2 SKIP |
| P1.2 Operator-touch audit (Phase VIII commits) | 0 commits touch operator since `6692f0f5` (Phase VII triage) |
| P1.3 Manifest completeness | 1626 completed + 140 documented-failed (recovered) + 9 schema rows; structurally sound |
| P1.4 AC-Trap pre-sweep reproducibility | All 9 (β, seed) cells bit-identical to original; return arrays equal |

### Phase 2 — cross-cell consistency probes

| check | result |
|---|---|
| P2.1 Manual AUC verification (AC-FP × β=+1, 10-seed mean) | manual = 454488.55 = aggregator (bit-identical) |
| P2.2 Manual alignment-rate replay | per-episode mean matches agent's logged value at 0 difference for all 200 ep |
| P2.3 β=0 bit-identity | g(β=0,γ,r,v) ≡ r+γv exactly across {r,v,γ} grid; ZeroBeta = FixedBetaSchedule(beta0=1e-9) bit-identical Q-tables |
| P2.4 Q-divergence sanity | replay |Q|_max = 1080372.85 = production q_abs_max[999] bit-identical; divergence is REAL, not metric artifact |
| P2.5 Cross-cell sign-of-β test | alignment(+β)<0.20 in 22/22 cells (100%); AUC(−β)>AUC(+β) in 14/16 non-tied cells (88%) |

A first-cut metric (raw rank correlation `ρ̄=0.03`) was rejected as
misleading because 6/22 cells are rank-degenerate (DC advance-only +
RR-Sparse-{Stationary, Tremble} produce identical AUC across β); the
sign-of-β test (above) replaces it.

### Phase 3 — Codex broad adversarial review

Verdict: **GENUINE FINDING — safe to proceed with the paper pivot.**

Memo: [`codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md`](codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md).

Key results from the audit (verbatim from memo §1):
- **(a) Operator math** (`tab_operator.py`): GENUINE FINDING / CLEAN.
  Verified analytically at all requested (β, γ, r, v) grid points;
  `max_abs_err = 0.0` for `g`, `rho`, and `effective_discount`.
- **(b) AdaptiveBetaQAgent** (`agents.py`): NIT / CLEAN FOR PRODUCTION.
  No rogue Q-table write paths; β cached once per episode; alignment
  uses `(β, r, v_next)`. Q is exposed mutably via `agent.Q` for tests
  but production runner does not mutate.
- **(c) FixedBetaSchedule** (`schedules.py`): GENUINE FINDING / CLEAN.
  Returns `+1.0` for all e under `FixedBetaSchedule(+1, hyperparams={"beta0": 1.0})`;
  clipping tests with `|beta_raw| ≤ beta_cap` leave `beta_used == beta_raw`.
- **(d) Aggregator** (`aggregate.py`): GENUINE FINDING / CLEAN with
  routing note. Per-seed AUC = `np.trapezoid(returns)`, paired-seed
  alignment correct. NIT: aggregator delegates AUC to runner rather
  than recomputing.
- **(e) Reference impl cross-check**: GENUINE FINDING. ~30 LoC reference
  agent at [`codex_reviews/reference_tab_agent.py`](codex_reviews/reference_tab_agent.py)
  with no production imports matched production AUC EXACTLY (0.00%
  difference) across all 9 (β, seed) cells over both 1000-episode
  prefix and full 10000-episode extension.

Two NIT-level observations:
- Aggregator delegates AUC to runner (architectural, not a bug).
- Stage 1 artifacts carry an older git SHA (`8d69b908…`) than HEAD;
  diff to HEAD touches only the unrelated `rules_of_road_sparse`
  payoff alias — not affecting AC-FictitiousPlay.

### Phase 4 — parameter perturbation sweep (162 runs in 87 sec)

6 perturbations × 3 cells (AC-FictitiousPlay, SH-FiniteMemoryRegret,
RR-StationaryConvention) × 3 β arms (−1, 0, +1) × 3 seeds × 1000 episodes.

| perturbation | cell | AUC(-1) | AUC(0) | AUC(+1) | v7 holds? |
|---|---|---:|---:|---:|---|
| γ = 0.9    | AC-FP   | 57100 | 57098 | 49966 | ✓ |
|            | SH-FMR  | 10229 | 10250 | 10209 | ✓ |
|            | RR-Stat |  7130 |  7139 |  6209 | ✓ |
| γ = 0.99   | AC-FP   | 57098 | 57098 | 46396 | ✓ |
|            | SH-FMR  | 10355 | 10018 |  9730 | ✓ |
|            | RR-Stat |  7116 |  6976 |  5976 | ✓ |
| **q = −2** | AC-FP   | 43629 | 57098 | 52620 | **FLIP** |
|            | SH-FMR  |  7139 |  9804 |  8828 | **FLIP** |
|            | RR-Stat |  1744 |  6695 |  5923 | **FLIP** |
| q = +5     | AC-FP   | 57102 | 57100 | 36744 | ✓ |
|            | SH-FMR  | 10688 | 10976 |  6836 | ✓ |
|            | RR-Stat |  6708 |  6560 |  1176 | ✓ |
| α = 0.05   | AC-FP   | 57098 | 57098 | 57098 | ✓ |
|            | SH-FMR  | 10626 | 10283 |  9845 | ✓ |
|            | RR-Stat |  7168 |  7170 |  6553 | ✓ |
| α = 0.3    | AC-FP   | 57101 | 57102 | 36665 | ✓ |
|            | SH-FMR  | 11151 | 10148 |  7809 | ✓ |
|            | RR-Stat |  5858 |  5906 |  1773 | ✓ |

**Flips: 3/18 = 17%, all at q_init = −2.0 (pessimistic init).**

In the 3 flipped cells, the ordering is `vanilla > +β > −β` — vanilla
still wins, but +β beats −β. The other 15 cells confirm v7's
`vanilla ≈ −β > +β`.

### Phase 5 — disposition memo

[`v7_bug_hunt_disposition.md`](v7_bug_hunt_disposition.md) records
**CASE C** per the user's HALT-7 protocol:
- v7 holds under γ ∈ {0.9, 0.99} and α ∈ {0.05, 0.3} perturbations.
- v7 flips at q_init = −2 in 3/3 cells; vanilla still wins in those
  cells.
- The narrative is **REFINED, not refuted**: TAB sign should match
  $\mathrm{sign}(r - V^*)$ in expectation; +β destabilizes asymmetrically
  in Q_init relative to V*, not absolutely.
- "Vanilla always wins" is empirically anchored across the 162-run
  envelope.

### Phase 6 (de facto) — CASE C refinement folded into counter-intuitive findings

Appended to [`counter_intuitive_findings.md`](counter_intuitive_findings.md#v7-bug-hunt-verification-2026-05-01--case-c-refinement)
under "v7 bug-hunt verification — CASE C refinement". Documents the
sign-symmetric refinement; M7+ proceed flag = yes with refined v7
narrative pinned in M12.

---

## 7. Run statistics

### 7.1 Run counts by milestone

| milestone | dispatched runs | wall-clock | output |
|---|---:|---|---|
| M0–M1 | 0 | n/a | spec written |
| M2 reopen | 0 | n/a | game/opp implementations |
| M5 | 0 | n/a | metrics + verifier (pure code) |
| M6 wave 1 (smoke) | 1 | <1 min | runner sanity |
| M6 wave 1.5 (AC-Trap pre-sweep) | 9 | 3 min | HALT 6 trigger |
| M6 wave 1.5 ablation | 36 | 12 min | 5-condition sweep |
| M6 wave 2 (Stage A dev) | 378 | 15 min | cross-cell extension |
| M6 wave 4 (Stage 1 main) | 1,540 | 95 min | per-cell summary |
| M6 wave 4 (RR-Sparse recovery) | 140 | 12 min | OQ4 fix |
| M6 wave 5 (figures-only) | 40 | 30 min | β=0 bifurcation viz |
| Pre-M7 bug-hunt Phase 4 | 162 | 87 sec | CASE C refinement |
| **M6 + ablation total** | **2,306** | **~169 min** | |
| V10.0 binding probe | 18 | <5 min | smoke validation |
| V10.1 AC-Trap smoke | 63 | 43 sec | β=0 bifurcation |
| V10.2 dev pass | 1,890 | 15 min | Tier I dev |
| V10.3a Tier I main | 6,300 | ~7 h | per-cell at γ=0.95 |
| V10.3b Tier II γ × β | 1,680 | ~2.5 h | 4 headline cells |
| V10.3c Tier III γ × cells | 3,000 | ~3 h | 30 cells coarse |
| **V10 total** | **12,951** | **~17 h** | |
| **GRAND TOTAL** | **≈14,676 dispatched** | **~38 h cumulative** | |

The smoke / V10 phases ran sequentially in a single autonomous loop
(no interleaving). M6 ran across the morning and afternoon of
2026-05-01; v10 ran 2026-05-01 evening through 2026-05-02 noon. Total
calendar elapsed: ~32 hours; total compute wall-clock: ~38 hours
(some overlap between dispatch and analysis).

### 7.2 Failure / divergence accounting

| check | M6 (1,820 runs) | v10 (10,980 main) | comment |
|---|---:|---:|---|
| NaN runs | 0 | 0 | clean |
| Manifest completed | 1,626 | 10,980 | 100% (after recovery) |
| Manifest documented-fail | 140 | 0 | RR-Sparse OQ4 alias bug, recovered |
| **Divergence flags** | 0 (M6 wave 6 detector) | **524** (G6c re-derivation) | (see below) |

**Divergence flags — G6c re-derivation correction**: the V10.4 detector
pass on raw `metrics.npz` reported 0 divergence flags. The V10.5 G6c
re-derivation found 524 across 10,980 runs. Concentration:

| stage | cell | diverged runs |
|---|---|---:|
| Tier I | AC-FictitiousPlay | 64 |
| Tier I | AC-Inertia | 70 |
| Tier I | AC-SmoothedBR | 46 |
| Tier I | AC-Trap | 49 |
| Tier I | DC-Long50 | 70 |
| Tier II | AC-Trap | 35 |
| Tier II | DC-Long50 | 90 |
| Tier III | AC-FictitiousPlay | 19 |
| Tier III | AC-Inertia | 20 |
| Tier III | AC-SmoothedBR | 12 |
| Tier III | AC-Trap | 14 |
| Tier III | DC-Long50 | 35 |

All 524 divergent runs are confined to AC-* and DC-Long50 cells
in **positive-β arms** at γ ≥ 0.80. This is mechanism-consistent with
v7 (positive-β + bootstrap-overshoot → `d_eff > 1` → Q diverges) and
with the Phase VIII canonical-payoff bound (e.g. AC-Trap's
all-Stag finite-horizon discounted payoff is ~64.15 with γ=0.95 and
H=20; observed `q_abs_max` reaches 1,085,189 in `+1` arms).

**Action item per G6c**: regenerate the V10.4 detector table by reading
`run.json::diverged` and `metrics.npz::divergence_event.sum()` for every
manifest row. Annotate positive-β collapse arms in any future
β-vs-AUC figure.

### 7.3 HALT counts and resolutions

| HALT | UTC | trigger | resolution | spec impact |
|---|---|---|---|---|
| 1 | 2026-05-01T02:43:53Z | T11 — DC-Long50 AUC deterministic | switch to q_convergence_rate metric | v3 |
| 2 | 2026-05-01T11:14:49Z | T11 — P-Contract sign inverted | flip sign in spec §5.7 | v4 |
| 3 | 2026-05-01T11:31:32Z | T11 — TAB has β-specific fixed points | switch to bellman_residual_beta | v5 |
| 4 | 2026-05-01T12:24:46Z | T11 — Cohen's d guard misfires (deterministic testbed) | replace with absolute-floor + relative-gap floor | v5b |
| 5 | 2026-05-01T12:56:56Z | OQ1+OQ2+OQ3+OQ4 (methodological) | user resolutions for each | v6 |
| 6 | 2026-05-01T16:32:39Z | T1+T3 — AC-Trap pre-sweep contradicted | GENUINE FINDING after Codex review + 5-condition ablation | v7 |
| 7 | 2026-05-01T19:06:56Z | spec-mandated user gate (M6 → M7) | 5-phase pre-M7 bug-hunt + v10 dispatch authorization | v10 |

7 halts; 4 spec amendments (v3, v5, v6, v7) plus 2 instrumentation
amendments (v5b for guard, v10 for grid extension). All resolved within
36-hour cap (per addendum §5).

### 7.4 Token / compute budget

- **opus-4-7 (orchestrator)**: ~1.6M tokens cumulative
- **codex-gpt-5.5-xhigh (review/test-author)**: ~1.43M tokens cumulative
- **Compute**: single-process Python on macOS Darwin 25.3.0, no GPU,
  single-core peak (parallel dispatched runs use process-pool fan-out
  per the runner; configurable in dispatched yaml)
- **Disk**: ~14 GB raw (metrics.npz + run.json across 12,951 + 1,725
  runs)

### 7.5 Codex review summary

| review | verdict | scope |
|---|---|---|
| M4 close (G4b) | PASS — no findings | M4 schedules + baselines diff |
| AC-Trap pre-sweep | GENUINE FINDING | post-HALT-6 verification |
| v7 broad bug-hunt | GENUINE FINDING (CASE C refinement) | pre-M7 5-phase verification |
| v10 G6c milestone-close | CONDITIONAL PASS (3 MAJOR corrections) | post-V10.3 raw-artifact re-derivation |

The G6c review is the most consequential: it corrected
[`v10_summary.md`](v10_summary.md)'s headline claims on H1 (narrowed
from 2 cells to 1), refuted H2 and H3, and identified 524 missed
divergence flags. The corrections are folded into §1 and §5 of this
report.

---

## 8. Open issues + future work

### 8.1 Immediate next steps (V10.6 → V10.9)

| task | status | priority |
|---|---|---|
| V10.6: 5 paper figures (β-vs-AUC fine grid, γ × β heatmap, alignment collapse, q_abs_max divergence, DC-Long50 residual decay) | DEFERRED | high (paper draft) |
| V10.7: supersession memo (pre-v10 → v10 atomically) | DEFERRED | medium |
| V10.8: 5-phase bug-hunt RE-validation on v10 results (parallels HALT-7 pre-M7) | DEFERRED | high |
| V10.9: this full report | DELIVERED | — |

### 8.2 G6c-mandated MAJOR action items

1. **Fix the divergence detector and regenerate V10.4.** Read
   `run.json::diverged` and `metrics.npz::divergence_event.sum()` for
   every manifest row. Update [`v10_summary.md`](v10_summary.md) line 21
   ("0" → "524") and the "No bug signatures" sentence.
2. **Correct the hypothesis disposition memos.** Record H1 as AC-Trap-only,
   SH-FMR as not CI-supported, H2 as refuted, H3 as refuted.
   [`v10_summary.md`](v10_summary.md) §V10.6 currently overstates 2/4
   cells; needs correction to 1/4.
3. **Do not attribute AC-Trap +0.10 to the alignment diagnostic without
   a new mechanism test.** Plot paired-seed trajectories for AC-Trap at
   γ=0.60 and β ∈ {vanilla, +0.05, +0.10, +0.20}; test whether the +0.10
   gain survives a separate seed expansion.

### 8.3 Downstream milestones (M7 → M12)

The HALT-7 user gate is still open in spirit. M7 design needs revision
in light of v7 + v10:

- **M7 (Stage 2)**: comparison `vanilla vs best_fixed_positive_TAB vs
  best_fixed_negative_TAB vs external baselines`. The
  `best_fixed_positive_TAB` baseline is meaningful only at γ=0.60 on
  AC-Trap (β=+0.10); on every other cell at γ=0.95, it would just
  re-confirm v7 (+β destabilizes). Two design options:
  - **(α)** Run M7 as-spec'd at γ=0.95; expect vanilla to dominate
    `best_fixed_positive_TAB`; report it as paper evidence.
  - **(β)** Run M7 at multiple γ values with the v10-extended grid;
    `best_fixed_positive_TAB` is now meaningful only at AC-Trap × γ=0.60.
    Reduces M7 to 7-8 methods.
- **M8 (Stage 3, sign-specialization)**: the v7 narrative dissolves the
  "G_+ regime exists" assumption. M8 should be reframed as
  "TAB-sign adaptation given Q_init / γ" rather than "G_+ exploration".
- **M9 (Stage 4, sign-switching composite)**: oracle-validation gate
  (oracle β must beat both fixed signs on AUC and recovery). The
  composite design needs an env where alignment-condition theory
  predicts a clean sign flip — which AC-Trap × γ=0.60 now provides
  as a candidate building block.
- **M10 (Stage 5, contraction-adaptive β)**: ContractionUCB / ReturnUCB
  schedules + contraction_UCB_with_return_safeguard. Run on M8 G_+,
  M8 G_−, M9 sign-switching, RR-ConventionSwitch. Adaptive β is the
  paper's "selective" mechanism — it is more interesting after the v7
  refinement (the alignment-condition diagnostic that drives ContractionUCB
  is now empirically sign-symmetric).
- **M11 (optional, advanced schedules)**: hedge_beta, gradient_beta,
  bilevel_SOBBO_beta. Per spec §6.2, runs only after M1–M10 stable +
  user authorization.
- **M12 (final recommendation)**: paper-narrative outline. With v10
  refinements, the paper's positive demonstrator becomes AC-Trap × γ=0.60
  (only data point where +β beats vanilla); the negative-result
  apparatus (AC-Trap as falsifiability cell, DC-Long50 as long-horizon
  contraction-tightness demonstrator) carries the empirical payload.

### 8.4 Mechanism-test / paper-draft TODOs

| open question | source | proposed resolution |
|---|---|---|
| Does AC-Trap +0.10 at γ=0.60 survive 10-seed expansion? | G6c §(b) | dispatch a 10-seed paired-bootstrap rerun |
| Why does alignment_rate at the AC-Trap +0.10 winning arm at γ=0.60 stay below 0.5? | G6c §(b) | per-step trajectory analysis; possible early-training transient regime dominates AUC |
| Is there a mechanism diagnostic that correctly identifies the H1 sign flip? | G6c §(c) | candidates: `frac_d_eff_below_gamma`, `mean_advantage` early-vs-late split |
| Does H1 generalize to other low-γ cells (e.g. SO-AntiCoordination, PG-Congestion)? | v10_summary observations | already in Tier III at γ=0.60; dispatch paired-bootstrap on the candidates |
| What is the right paper-narrative framing for the 524 divergence flags? | G6c §(d) | "the operator's amplification is observable as Q-divergence in 5% of positive-β arms; this validates the safety-cap need" |

---

## 9. Reproducibility

### 9.1 Branch / tags

- **Branch**: `phase-VIII-tab-six-games-2026-04-30`
- **Pre-v10 supersession tag**: `pre-extended-grid` (HEAD `dc07737f`)
- **HEAD at report-write time**: `52dd9a34` (V10.5 G6c review landed)
- **Test suite**: 1,737 PASS + 2 SKIP at HEAD
  (1,736 PASS pre-v10 multi-γ smoke per
  [`v10_summary.md`](v10_summary.md); +1 from post-G6c regression)

### 9.2 Configs

All v10 dispatch configs at
`experiments/adaptive_beta/tab_six_games/configs/`:

```
v10_binding_probe.yaml             V10.0 — 18 runs
v10_smoke_AC_Trap.yaml             V10.1 — 63 runs
v10_dev.yaml                       V10.2 — 1,890 runs
stage1_tier1_beta_canonical.yaml   V10.3a — 6,300 runs
stage1_tier2_gamma_beta_headline.yaml V10.3b — 1,680 runs
stage1_tier3_gamma_cell_coverage.yaml V10.3c — 3,000 runs
```

Pre-v10 configs (preserved at tag `pre-extended-grid`):

```
dev.yaml                                      Stage A
stage1_beta_sweep.yaml                        Stage 1 main
stage1_beta_sweep_rr_sparse_recovery.yaml     OQ4 recovery
figures_only_beta_grid.yaml                   wave 5
pre_sweep_AC_Trap.yaml                        wave 1.5
pre_sweep_AC_Trap_A1_q5.yaml                  ablation A1
pre_sweep_AC_Trap_A2_long.yaml                ablation A2
pre_sweep_AC_Trap_A3_inertia.yaml             ablation A3
pre_sweep_AC_Trap_A4_uniform.yaml             ablation A4
v7_perturbations/                             HALT-7 Phase 4
```

### 9.3 Runner

```
experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py
```

Single runner handles M6 + V10.* dispatches. The runner takes
`--config <yaml>`; full help via `--help`.

```bash
.venv/bin/python -m experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage1_beta_sweep \
    --config experiments/adaptive_beta/tab_six_games/configs/stage1_tier1_beta_canonical.yaml
```

The runner integrates with `Phase8RunRoster` (manifests),
`make_run_dir(base="results/adaptive_beta/tab_six_games")` (result root
hygiene per lessons.md #11), canonical `metrics.npz` schema, and the
v5b headline-metric routing (`bellman_residual_beta` AUC for
advance-only delayed-chain; return AUC for everything else).

### 9.4 Aggregation

```
experiments/adaptive_beta/tab_six_games/analysis/aggregate.py
experiments/adaptive_beta/tab_six_games/analysis/beta_sweep_plots.py
experiments/adaptive_beta/tab_six_games/metrics.py
```

`aggregate.py` flattens raw `metrics.npz` files to long CSVs.
`beta_sweep_plots.py` computes `seed_auc = np.trapezoid(returns)` and
emits per-cell summary tables. The aggregator routes `headline_metric`
per cell (v6 OQ2 schema parity fold-in).

### 9.5 Processed CSVs

```
results/adaptive_beta/tab_six_games/processed/
  M6_per_cell_summary.csv                    162 rows; pre-v10 22 cells × 7 β
  stage1_beta_sweep_main.csv                 1,400 runs × 10k ep = 14M rows
  stage1_beta_sweep_rr_sparse_recovery.csv   140 × 10k = 1.4M rows
  figures_only_beta_grid.csv                 40 × 10k = 400K rows
  v10_*  (TBD; awaiting V10.6 plotter pass)
```

### 9.6 Reproduce a single cell

```bash
# Reproduce AC-Trap × γ=0.60 × 21-arm × 5 seeds:
.venv/bin/python -m experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage1_beta_sweep \
    --config experiments/adaptive_beta/tab_six_games/configs/stage1_tier2_gamma_beta_headline.yaml \
    --filter "asymmetric_coordination/AC-Trap/gamma_0.60"
# Wall-clock ~5 min. Output:
# results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/
#   asymmetric_coordination/AC-Trap/gamma_0.60/{vanilla, fixed_beta_+0.05, ...}/seed_{0..4}/
#     metrics.npz, run.json
```

### 9.7 Tests

```
tests/adaptive_beta/tab_six_games/                     144 PASS + 1 SKIP
tests/adaptive_beta/strategic_games/                   ~45 PASS (incl. 4 PassiveOpponent + 7 RR-Sparse + 8 delayed_chain)
tests/lse_rl/operator/                                  shared kernel
tests/weighted_lse_dp/                                  Phase III–VI DP planners
```

Headline regression tests:
- `test_beta0_collapse_preserved.py` — β=0 bit-identity guard for
  every method in `ALL_METHOD_IDS` (12 methods).
- `test_phase_VIII_metrics.py` — all v3+v5 metrics + 8 helper tests
  (q_convergence_rate, q_star_delayed_chain, bellman_residual_beta,
  auc_neg_log_residual).
- `test_smoke_DC_Long50_q_convergence_ordering` — v5b smoke test
  (re-asserted: gap-based guard with absolute floor 100 + relative
  10%).
- `test_aggregate_schema_parity.py` — extended column count 49 → 53
  per v6 OQ2.

### 9.8 Lessons.md additions

Phase VIII added 6 v3-v10 lessons to [`tasks/lessons.md`](../../tasks/lessons.md):

- (v3) auto-fix scope: do NOT auto-patch beyond stated patch version
  for design changes.
- (v5b) test-instrumentation tweaks are MINOR under addendum §13;
  exempt from "do not auto-patch beyond vN" directive.
- (v6) run-count arithmetic discipline: spec arithmetic must be
  derivable from (cells × methods × seeds × ep) without
  pseudo-multipliers.
- (v7) negative-result honesty: AC-Trap reframed as falsifiability
  cell; treat predicted-but-refuted findings as paper signal.
- (v10) extended-grid resolution discipline: when wave-5 reveals a
  feature unresolved by the main grid, extend the grid before
  re-running.
- (G6c) raw-artifact re-derivation discipline: summary memos drift
  from raw artifacts; G6c paired-bootstrap re-derivation found 3 MAJOR
  corrections; future close-out reviews must read raw `metrics.npz`,
  not summary memos.

---

## 10. Glossary and notation

| symbol | meaning |
|---|---|
| γ | discount factor; default 0.95 |
| β | TAB temperature; β > 0 amplifies max(r, v); β < 0 amplifies min(r, v) |
| β_cap | clip bound on adaptive β (Phase VII default 2.0) |
| r, r_t | per-step reward |
| v_next, V | bootstrap value $\max_a Q(s', a)$ |
| g_{β,γ}(r, v) | TAB target (operator output) |
| ρ_{β,γ}(r, v) | effective continuation weight $\sigma(\beta(r-v) - \log\gamma)$ |
| d_{β,γ}(r, v) | effective discount $\partial_v g_{β,γ}$ |
| `aligned_t` | per-step diagnostic indicator $\mathbb{1}[\beta(r_t - v_{\text{next},t}) > 0]$ |
| `alignment_rate` | $\mathbb{E}_t[\text{aligned}_t]$ averaged per episode |
| AUC | $\int_0^E \text{return}(e)\,de$ (np.trapezoid over per-episode return) |
| $T_\beta$ | β-specific Bellman operator; $T_\beta Q(s,a) = \mathbb{E}[g_{β,γ}(r, \max_{a'} Q(s', a'))]$ |
| Q*_β | fixed point of $T_\beta$ (β-specific value function) |
| Q*_classical | $r + \gamma \max_a Q(s', a)$ fixed point (β=0 case) |
| `bellman_residual_beta` | $\|T_\beta Q - Q\|_\infty$ |
| `auc_neg_log_residual` | $\int_0^E -\log(\text{bres}_\beta(e) + 10^{-8})\,de$ |
| G_+ / G_− | spec §10.4 sign-specialization labels (post-v7: empirically
  ill-defined at canonical γ=0.95) |
| H1 / H2 / H3 | v10 pre-registered hypotheses (γ-induced sign flip /
  bifurcation widening / γ-stable diagnostic) |
| Tier I / II / III | v10 dispatch tiers (canonical γ / γ × β heatmap /
  γ × cell coverage) |
| M7.1 | Stage 2 fixed-TAB vs Q-learning baselines (3 360 + 480 runs) |
| M7.2 | Stage 2 strategic-learning agent baselines (RM, FP; 240 runs) |
| M8 | Stage 3 sign-specialisation analysis (analysis-only) |
| G_+ / G_− | sign-specialised cells per spec §10.4 paired-CI definitions |

---

## 11. M7 + M8 milestone summary (Stage 2 + 3 close)

This section consolidates the post-V10 milestones (M7.1, M7.2, M8)
into a single reading. They were dispatched 2026-05-02 in response
to user's spec-§10.3 directive ("resume the intended pipeline, not
shortcut it") after V10 had completed only Stage 1.

### 11.1 What was added

| Stage | Milestone | Methods | Cells × γ × seeds | Runs | Commit |
|---|---|---|---|---:|---|
| 2 | **M7.1** TAB re-dispatch | 21 fixed-β + vanilla | 4 × 4 × 10 | 3 360 | `722fd275` |
| 2 | **M7.1** Q-learning baselines v2 | restart, sliding_window, tuned_eps | 4 × 4 × 10 | 480 | `722fd275` |
| 2 | **M7.2** Strategic-learning agents | regret_matching_agent, smoothed_fictitious_play_agent | 3 × 4 × 10 | 240 | `7f411bf6` |
| 3 | **M8** Sign-specialisation analysis | (analysis-only) | 16 (cell, γ) | 0 | `86ed2cf7` |
| — | gitignore + repo hygiene | — | — | 0 | `73795b45` |
| **Total new compute** | | | | **3 840 + 240 = 4 080** | |

### 11.2 v1 → v2 baseline-config supersession (M7.1)

The first M7.1 baseline pass used the runner's default ε-schedule for
`tuned_epsilon_greedy_Q_learning`, which neutralised the "tuned"
mechanism (reduces to vanilla); also `sliding_window` default
window=10 000 never triggered eviction at the configured horizon.
Resulting v1 metrics.npz files were bit-identical to vanilla for
every (cell, γ, seed). The fix: pass `epsilon_schedule=None` to the
tuned class (uses class default 1.0 → 0.01 over 2000 episodes), and
override `window_size=2000` for sliding-window via
`method_kwargs_per_method`. v1 runs are retained on disk for
traceability and EXCLUDED from the canonical aggregate.
[Source: `stage2_fixed_tab_vs_baselines.md` §3.](stage2_fixed_tab_vs_baselines.md)

### 11.3 Codex-review disposition across M7.1 + M7.2

Three Codex reviews were run; all findings were addressed:

| Review | Finding | Severity | Disposition |
|---|---|---|---|
| M7.2 working-tree | Quadratic history copy in FP wrapper | P1 | **Already fixed at build time** via empirical_opponent_policy fast-path (250× speedup). |
| M7.2 working-tree | DC-Long50 false-best on bellman_residual | P1 | **Applied pre-dispatch.** DC-Long50 dropped from M7.2 strategic-agents config (3 cells × 4 γ instead of 4). |
| M7.2 working-tree | payoff_agent ignores game_kwargs | P2 | Deferred — none of the 4 cells uses payoff-modifying kwargs; tracked for M7.3 / M9 expansion. |
| Repo hygiene | 19 GB untracked artifacts could break push | P1 | **Applied** — `.gitignore` extended (commit `73795b45`). |

### 11.4 Headline G_+ / G_− table (M8 output, with M7.2 baselines)

| cell | γ=0.60 | γ=0.80 | γ=0.90 | γ=0.95 | strict-best baseline (γ=0.95) | TAB beats best baseline? |
|---|---|---|---|---|---|---|
| AC-Trap | **G_+** | neither | neither | neither | regret_matching_agent (+271k) | **No** (TAB +132 vs RM +271 019) |
| RR-StationaryConvention | **G_−** | **G_−** | **G_−** | **G_−** | regret_matching_agent (+39 714) | **No** (TAB +320 vs RM +39 714) |
| SH-FiniteMemoryRegret | neither | **G_−** | neither | neither | tuned_epsilon_greedy (+11 583) | **No** at γ=0.80 (TAB +97 vs tuned-ε +11 442); strategic agents fail (Δ ≈ −60k) |
| DC-Long50 | **G_−** | **G_−** | **G_−** | **G_−** | (no baseline beats vanilla) | **Yes — TAB only** (Δ +795 → +2 837) |

**TAB's exclusive wins**: DC-Long50 at every γ (β=−2.0, baselines
all reduce to vanilla); SH-FMR γ=0.80 (β=−0.5, +97.3 paired-CI).

**TAB never wins on AUC magnitude** vs the right baseline on
AC-Trap, RR, or SH-FMR at γ ≠ 0.80.

### 11.5 Acceptance status

- M7.1 → M7.2 / M8 promotion: ✓ (per
  [`stage2_fixed_tab_vs_baselines.md` §9](stage2_fixed_tab_vs_baselines.md))
- M7.2 → M8 / M9 promotion: ✓ (per
  [`stage2_strategic_agents_followup.md` §7](stage2_strategic_agents_followup.md))
- M8 → M9 promotion: ✓ — ≥ 1 G_+ AND ≥ 1 G_− candidate (per
  [`stage3_sign_specialization.md` §3](stage3_sign_specialization.md))
- M9 sign-switching composite dispatch: **PENDING USER SIGN-OFF**
  per spec §2 rule 13.

### 11.6 Open work tracked for M9 / M7.3

1. **M9 dispatch (user-gated)** — primary composite candidate is
   AC-Trap γ=0.60 (G_+, β=+0.10) ⊕ RR γ=0.60 (G_−, β=−0.5) under
   exogenous dwell D ∈ {100, 250, 500, 1000} per spec §10.5.
2. **M7.3 — Codex P2 #3 fix** — patch `_resolve_payoff_agent` to
   honour `game_kwargs`; needed before any dispatch on
   `soda_uncertain` or `rules_of_road` with `payoff_bias`.
3. **M7.3 — 6-game expansion** — 26 V10 Tier I cells lack baseline
   coverage; spec §10.3 calls for "all six games". Out of M7.1/M7.2
   scope by design.
4. **DC-Long50 strategic-agent diagnostic** — confirmed as
   expected-failure in smoke; not run at scale because the headline
   metric would falsely rank them best (Codex P1 #2). If the paper
   wants to document the diagnostic explicitly, run it under a
   different metric (e.g. mean return = 1.0 always — uninformative).

---

## Appendix A — pointers to source memos

Top-level memos (highest signal density), citation-ready:

- [`v10_summary.md`](v10_summary.md) — v10 dispatch summary (~150 lines).
  **Caveat**: H1 / H2 / H3 dispositions and divergence-flag count
  superseded by the V10.5 G6c review.
- [`M6_summary.md`](M6_summary.md) — M6 close summary (~430 lines).
  Authoritative for the pre-v10 22-cell × 7 β × 10 seeds main pass.
- [`counter_intuitive_findings.md`](counter_intuitive_findings.md) —
  paper-facing log of all paper-narrative pivots: v7 AC-Trap reversal,
  cross-cell extension, wave-5 bifurcation, v7-bug-hunt CASE C
  refinement.
- [`v7_bug_hunt_disposition.md`](v7_bug_hunt_disposition.md) — HALT-7
  5-phase verification disposition memo.
- [`v10_smoke_AC_Trap.md`](v10_smoke_AC_Trap.md) — V10.1 fine-grain
  bifurcation memo; sharp β=0 sign-bifurcation.
- [`M6_wave_6_T_detector.md`](M6_wave_6_T_detector.md) — M6 T-detector pass.
- [`pre_sweep_AC_Trap_ablation.md`](pre_sweep_AC_Trap_ablation.md) —
  HALT-6 5-condition ablation.

Halt memos:

- [`halts/delayed_chain_smoke_prediction_failure_2026-05-01T02-43-53Z.md`](halts/delayed_chain_smoke_prediction_failure_2026-05-01T02-43-53Z.md)
  — HALT 1 (T11; v3 amendment).
- [`halts/delayed_chain_P_Contract_sign_inverted_2026-05-01T11-14-49Z.md`](halts/delayed_chain_P_Contract_sign_inverted_2026-05-01T11-14-49Z.md)
  — HALT 2 (v4 sign flip).
- [`halts/delayed_chain_metric_design_flaw_2026-05-01T11-31-32Z.md`](halts/delayed_chain_metric_design_flaw_2026-05-01T11-31-32Z.md)
  — HALT 3 (v5 β-specific Bellman residual).
- [`halts/delayed_chain_v5_d_threshold_misfire_2026-05-01T12-24-46Z.md`](halts/delayed_chain_v5_d_threshold_misfire_2026-05-01T12-24-46Z.md)
  — HALT 4 (v5b Cohen's d guard).
- [`halts/M6_wave_1_open_questions_2026-05-01T12-56-56Z.md`](halts/M6_wave_1_open_questions_2026-05-01T12-56-56Z.md)
  — HALT 5 (v6 OQ1+2+3+4).
- [`halts/AC_Trap_pre_sweep_contradicted_2026-05-01T16-32-39Z.md`](halts/AC_Trap_pre_sweep_contradicted_2026-05-01T16-32-39Z.md)
  — HALT 6 (v7 GENUINE FINDING).
- [`halts/M6_close_M7_user_signoff_2026-05-01T19-06-56Z.md`](halts/M6_close_M7_user_signoff_2026-05-01T19-06-56Z.md)
  — HALT 7 (pre-M7 user gate).

Codex review memos:

- [`codex_reviews/M4_close_review_2026-05-01T01-09-13Z.md`](codex_reviews/M4_close_review_2026-05-01T01-09-13Z.md)
  — M4 close review (G4b); PASS, no findings.
- [`codex_reviews/AC_Trap_pre_sweep_review_2026-05-01T16-52-53Z.md`](codex_reviews/AC_Trap_pre_sweep_review_2026-05-01T16-52-53Z.md)
  — Codex GENUINE FINDING for AC-Trap.
- [`codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md`](codex_reviews/v7_broad_bug_hunt_20260501T202809Z.md)
  — Phase 3 of HALT-7 5-phase bug-hunt; reference TAB agent matched
  production AUC 0.00%.
- [`codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md`](codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md)
  — V10.5 milestone-close review; CONDITIONAL PASS with 3 MAJOR
  corrections.
- [`codex_reviews/reference_tab_agent.py`](codex_reviews/reference_tab_agent.py)
  — independent reference implementation (~30 LoC; no production
  imports).

Spec authority:

- [`docs/specs/phase_VIII_tab_six_games.md`](../../docs/specs/phase_VIII_tab_six_games.md)
  §0–§5 (objectives + envs), §6.4 (β grid v10 = 21 arms), §7 (metrics),
  §10 (stage protocol with v10 Tier I/II/III), §13.10 (AC-Trap §13.10
  honesty), §13.11 (H1/H2/H3), §22 (canonical signs), §23 (changelog
  v3-v10).

Lessons + tasks:

- [`tasks/lessons.md`](../../tasks/lessons.md) — 6 v3–v10 lessons added
  during Phase VIII.
- [`tasks/phase_VIII_overnight_addendum.md`](../../tasks/phase_VIII_overnight_addendum.md)
  — overnight harness contract (T1–T11 detectors, P1–P6 auto-promote
  rules, budget caps).

Per-cell summary tables (data backing):

- [`processed/M6_per_cell_summary.csv`](processed/M6_per_cell_summary.csv)
  (162 rows; pre-v10 per-cell × per-method statistics).

Live raw artifacts:

- `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier1_canonical/`
  (6,300 metrics.npz; Tier I, γ=0.95).
- `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier2_gamma_beta_headline/`
  (1,680 metrics.npz; Tier II, γ × β heatmap).
- `results/adaptive_beta/tab_six_games/raw/VIII/v10_tier3_gamma_cell_coverage/`
  (3,000 metrics.npz; Tier III, γ × cell coverage).
- `results/adaptive_beta/tab_six_games/raw/VIII/stage1_beta_sweep/`
  (1,540 + 140 metrics.npz; pre-v10 M6 main pass; preserved at tag
  `pre-extended-grid`).

---

## Appendix B — spec amendment timeline (v3 → v10 / v5b / v6 / v7 / v10)

| amendment | date | scope | outcome |
|---|---|---|---|
| v3 | 2026-05-01 ~02:43 | switch DC advance-only metric to `q_convergence_rate` | HALT 1 resolution |
| v4 | 2026-05-01 ~11:14 | flip P-Contract sign; fix Q* off-by-one; add terminal-state slice | HALT 2 resolution |
| v5 | 2026-05-01 ~11:31 | switch headline metric from `q_convergence_rate(Q, Q*_classical)` to β-specific Bellman residual | HALT 3 resolution |
| v5b | 2026-05-01 ~12:24 | replace Cohen's d guard with absolute-floor + relative-gap floor for noiseless DC-Long50 | HALT 4 resolution (instrumentation only) |
| v6 | 2026-05-01 ~12:56 | OQ1 (run count corrected 4,340 → 1,820); OQ2 (aggregator schema parity); OQ3 (AC-Trap adversary = `finite_memory_regret_matching`); OQ4 (RR/SO/PG stationary probs confirmed) | HALT 5 resolution |
| v7 | 2026-05-01 ~16:32 | AC-Trap repositioned as falsifiability cell; §13.10 negative-result honesty norm added | HALT 6 resolution (GENUINE FINDING) |
| v10 | 2026-05-01 evening | extended β grid (7 → 21 arms); γ-grid (1 → 4 values); enumeration (22 → 30 cells); H1/H2/H3 pre-registered as §13.11 | HALT 7 resolution (post-bug-hunt authorization) |

The v3 → v5b sequence (4 amendments in ~9.5 hours) is the most
intense in the project's history; each was triggered by a T11 fire on
the same DC-Long50 smoke test and resolved a different layer of the
problem (metric → sign → fixed-point → guard). The post-v5b sequence
(v6 → v7 → v10) is each independently triggered and well-separated.

---

## Appendix C — reading order for a cold coauthor

1. **§1 Executive summary** (this report) — 5 minutes.
2. **§3 Environment + opponent inventory** — orient on the 30-cell suite.
3. **§5 Headline empirical findings** — the paper's empirical payload.
4. **[`v10_summary.md`](v10_summary.md)** + **[`codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md`](codex_reviews/v10_G6c_milestone_review_20260502T130059Z.md)**
   — read in this order; the G6c review is the corrections layer over
   the v10 summary.
5. **[`counter_intuitive_findings.md`](counter_intuitive_findings.md)**
   — paper-facing log of pivots; useful for reviewer-defense framing.
6. **[`M6_summary.md`](M6_summary.md)** §3 (per-cell table) — the
   main-pass numbers that the paper headline cells draw from.
7. **§4 Methods** + **[`docs/specs/phase_VIII_tab_six_games.md` §4](../../docs/specs/phase_VIII_tab_six_games.md)**
   — operator math + alignment condition derivation.
8. **§6 Bug-hunt verification** — for reviewer-defense / methodological
   rigor justification.
9. **§7 Run statistics + §9 Reproducibility** — deployment / replication
   guide.

For paper-draft purposes:
- **Headline cell**: AC-Trap × γ=0.60, β=+0.10 (only +β beating
  vanilla in the suite; mechanism caveat per G6c §(b)).
- **Negative-result apparatus**: AC-Trap (v7 falsifiability), DC-Long50
  (long-horizon contraction-tightness with 17-orders-of-magnitude
  separation).
- **Diagnostic ablation**: alignment_rate's 53% accuracy across (γ, cell)
  tuples is itself a paper finding (the diagnostic's scope is correctly
  identified).

---

*End of report.*
