# Phase VII Stage A summary — 2026-04-26

**Branch:** phase-VII-overnight-2026-04-26 @ 0f19d19d

**Source:** `results/summaries/phase_VII_manifest.json` (96 runs)

**Wall-clock:** ~16s on CPU (Stage A run-only).


## 1. Promotion verdict

**Promoted envs (Stage B):** **none**

**Self-play (§22.4) Stage B inclusion:** no (§22.4: no Stage-A env cleared the bar)


## 2. Per-env results

| Env | adaptive_β AUC vs vanilla (mean ± SE, paired) | catastrophic Δ (mean ± SE) | mech. ev. | recovery Δ | gate verdict |
|-----|---|---|---|---|---|
| rps | -18.7 ± 107.5 | +0.0 ± 0.0 | align=0.81, d_eff=0.47 | +117.7 | FAIL |
| switching_bandit | -10.7 ± 9.1 | +0.0 ± 0.0 | n/a (§22.5) | +0.3 | FAIL |
| hazard_gridworld | -103.0 ± 47.6 | +2.3 ± 1.9 | align=0.26, d_eff=0.91 | +22.7 | FAIL |
| delayed_chain | +0.0 ± 0.0 | +0.0 ± 0.0 | align=nan, d_eff=nan | n/a | FAIL |

## 3. Mechanism diagnostics (non-bandit, adaptive_β only)

### rps

- mean alignment rate (informative transitions, last 500 eps): 0.807 ± 0.006 (across seeds)
- mean d_eff (informative transitions, last 500 eps): 0.470 ± 0.008  (γ = 0.95)
- frac d_eff < γ on informative transitions: 0.807
- β trajectory (mean across seeds): mean=-1.660, range=[-2.000, +1.983]
- informative transitions (last 500 eps, summed across seeds): 29509

### hazard_gridworld

- mean alignment rate (informative transitions, last 500 eps): 0.263 ± 0.094 (across seeds)
- mean d_eff (informative transitions, last 500 eps): 0.911 ± 0.011  (γ = 0.95)
- frac d_eff < γ on informative transitions: 0.263
- β trajectory (mean across seeds): mean=-1.796, range=[-2.000, +1.985]
- informative transitions (last 500 eps, summed across seeds): 24251

### delayed_chain

- mean alignment rate (informative transitions, last 500 eps): nan ± nan (across seeds)
- mean d_eff (informative transitions, last 500 eps): nan ± nan  (γ = 0.95)
- frac d_eff < γ on informative transitions: nan
- β trajectory (mean across seeds): mean=+0.000, range=[+0.000, +0.000]
- informative transitions (last 500 eps, summed across seeds): 0

## 4. Method comparisons (final return, last 500 eps; mean ± SE across 3 seeds)

| Env | vanilla | fixed_+ | fixed_− | adaptive_β | adaptive_β_no_clip |
|-----|---------|---------|---------|------------|--------------------|
| rps | +1.11 ± 0.18 | +0.02 ± 0.12 (2931 div eps) | -0.15 ± 0.07 (2184 div eps) | +1.19 ± 0.08 | +1.08 ± 0.12 |
| switching_bandit | +0.39 ± 0.01 | +0.37 ± 0.02 | +0.37 ± 0.01 | +0.38 ± 0.01 | +0.39 ± 0.00 |
| hazard_gridworld | -8.57 ± 0.66 | -8.66 ± 0.47 (388 div eps) | -8.83 ± 0.37 | -8.87 ± 0.49 | -8.92 ± 0.25 |
| delayed_chain | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.07 ± 0.03 |

## 5. Stability / divergence

| Env | adaptive_β: divergent eps (across 3 seeds) | no_clip: divergent eps |
|-----|---|---|
| rps | 0 | 0 |
| switching_bandit | 0 | 0 |
| hazard_gridworld | 0 | 0 |
| delayed_chain | 0 | 0 |

## 6. Stage B configuration

**No env cleared the Stage A bar.** Per `tasks/phase_VII_overnight_2026-04-26.md`, write a negative-result memo and stop. Stage B is not dispatched.

## 7. Open questions / notes

- `auc_return` is implemented as `np.sum(returns)` (equivalent up to a unit-spacing constant to `np.trapz`); chosen for simplicity.
- `recovery_time_first_shift` uses the SMOOTH_WINDOW=100 episodes preceding the first shift as the pre-shift baseline; if the first shift occurs before episode 100 the recovery time is NaN.
- `switching_bandit` mechanism columns are NaN per §22.5 (degenerate at H=1).
- Criterion 2 is checked with `cat_diff <= max(cat_diff_se, 0)`; ties at 0 (no catastrophes either side) PASS.
- Per-criterion verdicts in `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/processed/dev/promotion_gate.json`.