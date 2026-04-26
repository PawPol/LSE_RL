# Phase VII Stage C summary — 2026-04-26 (HEADLINE)

**Branch:** `phase-VII-overnight-2026-04-26`

**Source:** `results/summaries/phase_VII_manifest.json` (100 Stage C + 60 Stage B + 110 Stage A = 270 total runs)

**Wall-clock:** Stage C ≈ 300s on CPU (20 seeds × 10000 eps × 5 methods × rps).


## 1. Headline result

> On adversarial Rock-Paper-Scissors, adaptive-β Q-learning produces a
> paired-bootstrap-significant AUC improvement over classical Q-learning
> (paired diff +1732, 95% CI [+1473, +1970], n=20 seeds × 10000 eps),
> while remaining bit-stable (zero divergence events over 200 000
> episodes). The Bellman-advantage-driven mechanism is active throughout:
> alignment rate on informative transitions is 0.792 ± 0.003 (well above
> the 0.5 chance level), and the effective continuation `d_eff` is
> 0.568 ± 0.004 — well below the classical discount γ = 0.95 — confirming
> the spec §3.3 prediction that aligned β reduces effective continuation
> on informative transitions.

This satisfies **3 of 5 spec §0 quantitative predictions** at headline-paper
sample size on this environment (claims 1, 2, 5; claim 3 not significant
on rps; claim 4 not applicable since rps has no catastrophic episodes).


## 2. Headline numbers (rps, 20 seeds × 10000 eps)

| Method | AUC return (mean ± std) | final return (last 500 eps) | divergent eps | catastrophic |
|--------|-------------------------|------------------------------|----------------|--------------|
| `vanilla`               | 62919.85 ± 534.58 | +7.53 ± 0.40 | 0 | 0 |
| `fixed_positive`        | 32672.70 ± 307.72 | +2.71 ± 0.15 | 199 550 | 0 |
| `fixed_negative`        | 31629.05 ± 216.37 | +2.72 ± 0.16 | 193 788 | 0 |
| `adaptive_beta`         | **64651.85 ± 460.83** | +7.54 ± 0.27 | **0** | 0 |
| `adaptive_beta_no_clip` | **64926.80 ± 439.30** | +7.59 ± 0.28 | **0** | 0 |


## 3. Paired-seed differences vs `vanilla` (n=20 seeds; bootstrap 95% CI)

| Method | AUC diff (mean) | AUC diff 95% CI (paired bootstrap, 10k resamples) | final return diff | recovery time diff |
|--------|------------------|----------------------------------------------------|--------------------|---------------------|
| `fixed_positive`        | −30 247.2 | (large negative; uses unstable schedule) | −4.821 ± 0.091 | +1.95 ± 1.22 |
| `fixed_negative`        | −31 290.8 | (large negative; uses unstable schedule) | −4.816 ± 0.087 | +6.25 ± 2.82 |
| `adaptive_beta`         | **+1732.0** | **[+1473.4, +1970.2]** (CI excludes 0) | +0.008 ± 0.094 (n.s.) | +18.25 ± 17.57 |
| `adaptive_beta_no_clip` | **+2007.0** | (≈ similar to adaptive_β; full CI in paired_diffs.parquet) | +0.057 ± 0.110 (n.s.) | +6.40 ± 4.60 |

Detailed numbers in `results/adaptive_beta/processed/headline/paired_diffs.parquet`.

Plain-language reading: adaptive-β captures **+1732 AUC** points relative to
vanilla — a ~2.7 % cumulative-return advantage — concentrated in the early-
to-mid phase of training when faster credit assignment matters most.
Asymptotic (final-500-eps) return is statistically indistinguishable
(+0.008 ± 0.094); both methods converge to the same near-optimal
exploitation level once the opponent phase structure has been learned.

This matches the spec §0 prediction that adaptive-β acts as a **temporal
credit-assignment controller** rather than a final-performance booster —
its win is on sample efficiency, not on asymptotic ceiling.


## 4. Mechanism diagnostics — Stage C (adaptive_β, last 500 eps, 20 seeds)

| metric | value |
|---|---|
| `alignment_rate` (informative) | **0.7922 ± 0.0031** (~94σ above 0.5 chance) |
| `mean_d_eff` (informative)     | **0.5681 ± 0.0044** (~87σ below γ = 0.95) |
| `frac_d_eff < γ` (informative) | **0.7922** |
| β trajectory mean              | −1.428 |
| β trajectory range             | [−2.000, +2.000] (full envelope traversed) |
| informative transitions / seed (last 500 eps) | 1000 |

Stability of the mechanism across all three sample tiers:

| sample tier | alignment | d_eff | seeds × eps |
|---|---|---|---|
| Stage A initial (3 × 1000) | 0.807 ± 0.006 | 0.470 ± 0.008 | 3 × 1k |
| Stage A extended (10 × 5000) | 0.888 ± 0.004 | 0.431 ± 0.005 | 10 × 5k |
| Stage B (10 × 10000) | 0.790 ± 0.011 | 0.570 ± 0.018 | 10 × 10k |
| **Stage C (20 × 10000)** | **0.792 ± 0.003** | **0.568 ± 0.004** | **20 × 10k** |

Mechanism stays active across sample sizes; the modest decline in alignment
(0.89 → 0.79 between Stage A extended and Stage B) is the agent moving from
the early high-Bellman-advantage regime into the asymptotic exploitation
regime where r ≈ v_next on most transitions and the alignment metric loses
sensitivity.


## 5. Stability — Stage C (200 000 episodes per method)

| Method | divergence_event flagged | catastrophic episodes | divergence rate |
|---|---|---|---|
| `vanilla`               | 0 | 0 | 0 % |
| `fixed_positive`        | 199 550 | 0 | **99.78 %** |
| `fixed_negative`        | 193 788 | 0 | **96.89 %** |
| `adaptive_beta`         | **0** | 0 | **0 %** |
| `adaptive_beta_no_clip` | **0** | 0 | **0 %** |

The fixed-β methods drive the operator into the regime where
`np.logaddexp(β·r, β·v + log γ)` returns NaN due to extreme inputs (the agent
visits states with `|Q|` magnitudes that, multiplied by fixed β = ±1,
overflow the safe range for `expit`/`logaddexp`). All such episodes are
recorded honestly per spec §13.5; the runner completed all 40 fixed-β runs
without silently dropping any episode.

**Both adaptive variants — clipped and unclipped — show ZERO divergence
across 200 000 episodes per method.** The per-episode A_e-driven sign
selection actively keeps the operator inside the safe regime on rps. This
supports the spec §22.1 / §16.1 design contract: the centered/scaled
operator is stable when β is bounded by a clip cap **or** when β is
adaptively chosen via the spec §4.2 rule (with hyperparameter `beta_max =
beta_cap = 2.0`). The agreement of clipped and unclipped variants indicates
the clip cap is not actually binding under the adaptive rule on rps — the
schedule's `beta_max·tanh(k·A_e)` saturation handles bounding intrinsically.


## 6. Spec §0 prediction scoring on rps (Stage C)

| Prediction | Stage C verdict on rps |
|---|---|
| 1. Adaptive β increases alignment rate | **CONFIRMED** — 0.792 vs chance 0.5 |
| 2. Adaptive β reduces effective continuation on informative transitions | **CONFIRMED** — 0.568 vs γ = 0.95 |
| 3. Adaptive β improves recovery after shifts | **NOT CONFIRMED** — recovery diff +18.25 ± 17.57; n.s. |
| 4. Adaptive β reduces catastrophic episodes / drawdowns | **N/A** — 0 catastrophic episodes on rps |
| 5. Adaptive β improves AUC and sample efficiency, even when final return is similar | **CONFIRMED** — paired AUC +1732, 95% CI [+1473, +1970]; final-return diff n.s. (+0.008) |

**3 of 5 confirmed**, 1 not applicable, 1 not confirmed at this sample size.
Verdict: **partial support of the spec §0 claim, on a single environment.**
This is exactly the kind of moderately-strong, single-env, mechanism-validated
finding that belongs in an **appendix or supplement**, not in the main paper
§Experiments (which is currently pruned to two stronger claims at commit
`ccec6965`).


## 7. Open questions / notes

- The spec §0 prediction 3 ("improves recovery after shifts") is not
  confirmed because the per-shift recovery metric `recovery_time_first_shift`
  is high-variance at the per-shift level. A more powerful analysis would
  aggregate across all shift events (rps has 100 phase shifts per 10k-eps run);
  flagged for a future revision.
- Final-return diff being not significant is consistent with the
  credit-assignment-controller framing (spec §0): the win is on the AUC
  prefix, not the asymptotic ceiling.
- All Stage A → B → C promotion decisions cleared the strict pre-registered
  gate; no override was used.
- Self-play rps deferred (not implemented in this overnight run); not
  blocking — single-agent rps cleared all gates.
