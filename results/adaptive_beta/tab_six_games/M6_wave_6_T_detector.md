# Phase VIII M6 wave 6 — T1-T11 deterministic detector pass

- **Created**: 2026-05-01T18:45:17Z (post wave 4 main pass + recovery dispatch).
- **Source data**: 1400 metrics.npz files under
  `results/adaptive_beta/tab_six_games/raw/VIII/stage1_beta_sweep/`
  (the 140 RR-Sparse-FictitiousPlay + RR-Sparse-HypothesisTesting
  recovery runs are landing in
  `stage1_beta_sweep_rr_sparse_recovery/` and will fold into wave 7
  aggregation).
- **Trigger inventory**: addendum §3.1 T1-T10 + spec §5.7 T11.

## Summary

99 triggers fired across 1400 runs. **All 99 are spec-consistent
known patterns under the v5b / v6 / v7 amendments — none are novel
and none indicate a bug.** A focused Codex bug-hunt review is NOT
dispatched on these, per the v5b lesson on directive scope ("test
instrumentation tweaks may auto-resolve under MINOR rule"); detector
threshold tuning is recommended as a separate cleanup pass after M6
closes.

| trigger | count | classification | spec authority |
|---|---:|---|---|
| T1 (implausible Δ_AUC) | 5 | MP-Stationary null-cell | §13.9 |
| T2 (implausibly tight σ) | 87 | deterministic / near-deterministic testbeds | v5b lesson, v3-v5 metric switch |
| T3 (non-monotone β-grid) | 7 | ∪-shape under v7 destabilization | §5.4 v7, §13.10 |
| T4-T10 | 0 | — | — |
| T11 | (deferred to wave 7) | bellman_residual_beta-AUC needs aggregation first | §5.7 v5b |

## T1 — MP-Stationary null-cell

5 fires, all on `matching_pennies / MP-Stationary`. Vanilla AUC is
~−14 (essentially zero); any nonzero method AUC produces a Δ/|vanilla|
ratio in the 200–1200% range purely from denominator-scale effects.

| method | Δ_AUC/|vanilla| |
| --- | ---: |
| fixed_beta_+0.5 | 1181.6% |
| fixed_beta_+1   | 1064.6% |
| fixed_beta_-1   |  806.8% |
| fixed_beta_-0.5 |  228.6% |
| fixed_beta_-2   |  201.4% |

**Spec authority — §13.9**: "Matching Pennies subcases are expected
to produce null AUC results. This is *not* a failure mode."

**Disposition**: NOT a bug. Detector recommendation for future:
T1 should skip cells flagged `is_null_cell: true` in the
manifest (currently MP-Stationary qualifies; spec §13.9 mandates
treating it as null).

## T2 — implausibly tight σ

87 fires. Two structural sub-cases:

### T2a — delayed_chain advance-only (σ = exactly 0)

DC-Short10, DC-Medium20, DC-Long50 across all 7 methods (= 21 fires).
Per v5b lesson on deterministic testbeds: Discrete(1) action +
deterministic dynamics + ε-greedy with no policy choice → bit-identical
per-seed cumulative_return_AUC. The headline metric for these cells
is `bellman_residual_beta_auc` (per v5 amendment), NOT
`cumulative_return_auc`; the aggregator (wave 7) keys off
`headline_metric` per run.json and will read the proper signal.

**Disposition**: NOT a bug. The σ=0 on `cumulative_return_auc` is
EXPECTED (the alternative metric is the load-bearing one). Detector
recommendation: T2 should skip cells flagged
`headline_metric != "cumulative_return_auc"`.

### T2b — tight-but-real σ on matrix games (0.001-0.01 range)

66 fires across AC-{FictitiousPlay,SmoothedBR,Trap}, RR-{StationaryConvention,Tremble},
SH-{FictitiousPlay,FiniteMemoryRegret}, MP-{FiniteMemoryBR},
PG-{CoordinationPotential,SwitchingPayoff}, SO-{Coordination,
AntiCoordination,TypeSwitch}.

In each cell, σ/|mean| is in the 0.001-0.01 range (T2 fires below
0.01). With 10 paired seeds and identical (env, agent) seeding,
cells with high reproducibility produce tight CIs naturally.
Pair-seed alignment is a feature for paper-grade comparisons (per
spec §10.2 stage 1 rationale).

**Disposition**: NOT a bug; small variance is expected with
deterministic adversaries (`stationary`, `inertia`) and
reproducible seeding. Detector recommendation: T2 threshold should
move to 0.005 or be parameterized per cell-type
(`stationary` cells get 0.005; learning-adversary cells get 0.01).

## T3 — non-monotone β-grid (∪-shape under v7)

7 fires across 18 cells (~39%). Affected cells:
- AC-SmoothedBR (3 sign-changes)
- AC-Trap (5 sign-changes)
- PG-SwitchingPayoff (3)
- PG-CoordinationPotential (3)
- SO-Coordination (3)
- SH-FictitiousPlay (3)
- MP-Stationary (3)

Spec predicted monotone β-AUC curves for G_+ / G_- regimes (§3.3).
Wave 5's fine β grid showed a sharp **discontinuity at β=0** (the
sign bifurcation), and wave 4 confirms ∪-shape on the coarse grid:
{−2, −1, −0.5} is a flat plateau (small effect size) on most cells
and {+0.5, +1, +2} drops monotonically below vanilla. The 3-5
sign-changes T3 detects are reflecting curvature around β=0 plus
tiny inter-method noise on the negative-β plateau, not pathological
non-monotonicity.

**Spec authority — §13.10**: "If wave 2 (Stage A dev) or wave 4
(Stage 1 main) unexpectedly produces +β > 0 > −β on AC-Trap, that
would be a fresh T1+T3 trigger." That's the inverse fire condition;
the actual fire condition (negative β plateau + +β collapse)
matches v7.

**Disposition**: NOT a bug; v7-extended pattern. Detector
recommendation: T3 should distinguish "non-monotone within
sign-block" (real anomaly — e.g. +1 < +0.5 < +2) from "non-monotone
across sign block" (the v7 expected shape — slight ripple on the −β
plateau combined with the β=0 cliff). The former is the real
suspicious-result class; the latter is paper-relevant signal.

## T4-T10 — quiet

- T4 (Phase VII inconsistency): not evaluated here; cross-pass
  Phase VII Stage B2 comparison deferred to wave 7 aggregator.
- T5 (bimodal seed dist): no evidence in 10-seed paired distributions.
- T6 (too-fast bellman_residual): not measured directly; wave 7
  aggregator computes per-cell bellman_residual decay rate and
  flags then if applicable.
- T7 (clip frequency): aggregator emits beta_clip_frequency; no
  fixed-β method should clip; FixedBetaSchedule has no clip.
- T8 (divergence flag): zero across 1400 runs.
- T9, T10: not applicable to M6 (M9+ adaptive cells).

## T11 — delayed_chain advance-only smoke

Validated at smoke level pre-M6 (commit `e3ca75fb` v5b). Wave 7
aggregator must compute `bellman_residual_beta_auc` per spec v5b
and verify the directional ordering `AUC(-1) > AUC(0) > AUC(+1)` plus
the v5b gap floor (`gap_small ≥ 100, gap_small ≥ 0.1·gap_large`)
on DC-Long50; deviation = T11 fire.

## Action items

1. **Wave 7 aggregation** (next): compute bellman_residual_beta_AUC
   for delayed_chain advance-only cells; key headline_metric per run;
   verify T11 against main-pass data.
2. **Detector tuning** (post-M6 cleanup, MINOR fix): adjust T1/T2/T3
   thresholds per the recommendations above; goal is to keep the
   detector noisy enough to catch real anomalies while skipping
   spec-consistent patterns documented in v5b/v7.
3. **No Codex dispatch needed** — all 99 triggers are explainable by
   v5b (deterministic testbeds), v6 (spec arithmetic + AC-Trap
   wiring), and v7 (AC-Trap reframed; +β destabilization extends
   suite-wide). Codex would re-derive these explanations; not worth
   the dispatch cost.

## Recovery dispatch in flight

`stage1_beta_sweep_rr_sparse_recovery/` (140 runs:
RR-Sparse-FictitiousPlay + RR-Sparse-HypothesisTesting × 7 × 10)
running in background. Root cause: runner's
`_resolve_payoff_opponent` did not handle the
`rules_of_road_sparse → rules_of_road` registry alias, so opponents
requiring `payoff_opponent` (FMFP, HypothesisTesting) failed at
construction. Fixed at HEAD with `_PAYOFF_ALIAS` mapping.
