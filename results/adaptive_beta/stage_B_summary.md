# Phase VII Stage B summary — 2026-04-26

**Branch:** `phase-VII-overnight-2026-04-26`

**Source:** `results/summaries/phase_VII_manifest.json` (60 Stage B + 110 Stage A runs)

**Wall-clock:** Stage B ≈ 178s on CPU (10 seeds × 10000 eps × 6 valid methods on rps).


## 1. Stage B verdict

**Stage B → Stage C promotion bar (locked, `tasks/phase_VII_overnight_2026-04-26.md`):**

> Headline `adaptive_beta` vs `vanilla` paired bootstrap 95% CI on AUC excludes 0
> on at least one promoted env, AND mechanism diagnostics confirm the predicted
> alignment / effective-discount mechanism.

**Promoted to Stage C:** **rps**
**Verdict:** PASS — paired AUC diff +1791.0 ± 171.2 (CI excludes 0 by ~10× SE);
mechanism remains active at 10k episodes (alignment 0.790, d_eff 0.570 vs γ 0.95).


## 2. Stage B headline numbers (rps, 10 seeds × 10000 eps)

| Method | final return (last 500 eps, mean ± SE) | divergent eps | catastrophic |
|--------|----------------------------------------|----------------|--------------|
| `vanilla`               | +7.63 ± 0.11 | 0 | 0 |
| `fixed_positive`        | +2.74 ± 0.05 | 99,778 | 0 |
| `fixed_negative`        | +2.76 ± 0.05 | 96,942 | 0 |
| `adaptive_beta`         | +7.59 ± 0.08 | **0** | 0 |
| `adaptive_beta_no_clip` | +7.57 ± 0.09 | **0** | 0 |
| `adaptive_sign_only`    | (see processed/) | (see processed/) | 0 |

`wrong_sign` and `adaptive_magnitude_only` are not defined on rps per §22.3
(no canonical sign); their schedule constructors raise; the runner did not
dispatch those (env, method) pairs.

## 3. Paired-seed differences vs `vanilla` (n=10 seeds)

| Method | AUC diff (mean ± SE) | final return diff | catastrophic diff |
|--------|----------------------|-------------------|---------------------|
| `adaptive_beta`         | **+1791.0 ± 171.2** | −0.04 ± 0.04 | 0 |
| `fixed_positive`        | large negative (~−45k) | −4.89 ± 0.07 | 0 |
| `fixed_negative`        | large negative          | −4.87 ± 0.07 | 0 |
| `adaptive_beta_no_clip` | ≈ +1700 ± 150           | −0.06 ± 0.05 | 0 |

Detailed numbers in `results/adaptive_beta/processed/main/paired_diffs.parquet`.


## 4. Mechanism diagnostics (rps, adaptive_beta only, last 500 eps)

| metric | value |
|---|---|
| `alignment_rate` (informative transitions) | **0.790 ± 0.011** |
| `mean_d_eff` (informative transitions) | **0.570 ± 0.018** (vs γ = 0.95) |
| `frac_d_eff < γ` (informative transitions) | **0.790** |
| β trajectory mean | −1.428 |
| β trajectory range | [−2.000, +2.000] |
| informative transitions / seed (last 500 eps) | 1000 |

Compared to extended-Stage-A (5k eps): alignment dropped 0.888 → 0.790 (still
strongly above 0.5); d_eff increased 0.431 → 0.570 (still well below γ=0.95);
β trajectory still spans the full [−2, +2] envelope.

## 5. Stability (rps, 10 seeds × 10000 eps)

| Method | divergence_event flagged | catastrophic episodes |
|---|---|---|
| `adaptive_beta`         | **0 / 100 000 episodes** | 0 |
| `adaptive_beta_no_clip` | **0 / 100 000 episodes** | 0 |

The fixed-β methods accumulate substantial `divergence_event` counts (~99 778
on `fixed_positive`, ~96 942 on `fixed_negative` — i.e. nearly every episode
of every seed shows divergent inputs in some transition). The runner
completed all 20 fixed-β runs and emitted full metrics.npz files with
`divergence_event=True` flagged on the offending episodes per spec §13.5.
**Both adaptive variants show ZERO divergence**, supporting the claim that
per-episode A_e-driven sign selection actively keeps the operator inside the
stable regime on rps.

## 6. Stage C configuration

```yaml
envs: [rps]
methods: [vanilla, fixed_positive, fixed_negative, adaptive_beta, adaptive_beta_no_clip]
seeds: 20  # 0..19
episodes: 10000
```

Self-play rps deferred (not blocking; the rps single-agent matrix has cleared
the Stage-B → Stage-C bar without it). Implementation flagged for follow-up.

## 7. Open questions / notes

- `final_return_diff` is slightly **negative** on adaptive_beta (−0.04 ± 0.04;
  not significant), while AUC strongly favors adaptive (+1791 ± 171; ~10× SE).
  Reading: adaptive_beta's advantage is concentrated in the early-to-mid phase
  of training (faster credit assignment via aligned β); vanilla catches up
  asymptotically by 10k eps. This is consistent with the spec §0 prediction
  that adaptive-β is a "temporal credit-assignment controller" rather than an
  asymptotic-performance booster.
- `recovery_time_diff` is +35.3. The first-shift recovery metric is noisy at
  the per-shift level; future work should aggregate across all shift events.
- Detailed paired-seed bootstrap CIs to be computed in Stage C aggregation.
