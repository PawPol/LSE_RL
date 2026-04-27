# Phase VII-B Stage B2-Dev — Promotion Memo

**Branch:** `phase-VII-B-strategic-2026-04-26`
**Verdict:** **PROMOTE**
**Gate-promoted cells (3/3):** (shapley, hypothesis_testing, mode=strict), (shapley, finite_memory_regret_matching, mode=strict), (matching_pennies, finite_memory_regret_matching, mode=strict)
**Stage B2-Main dispatch cells (2):** (shapley, hypothesis_testing, mode=strict), (shapley, finite_memory_regret_matching, mode=strict)

**Cartesian-product filter notes** (Stage B2-Main runner
takes the Cartesian product of `games` × `adversaries`; cells
dropped to keep that product strictly inside the gate-promoted set):
- dropped (matching_pennies, finite_memory_regret_matching) — keeping it would force a non-promoted Cartesian neighbour into Stage B2-Main; z=+2.57

## 1. Run Matrix

- Expected: 4 games × 3 adversaries × 5 methods × 3 seeds = **180 cells**.
- Manifest contained 181 records; 1 stale `n_episodes=5` smoke
  duplicate filtered out by deduplication on
  `(game, adversary, method, seed_id)` keeping latest `started_at`.
- Post-filter count: **180** runs (matches expectation).

No divergence or NaN events recorded across all 180 runs.

## 2. Per-(game, adversary) AUC Paired Diff vs Vanilla

Paired bootstrap, 10,000 resamples, percentile CIs, paired by `seed_id` (common_env_seed).
With n=3 seeds CIs are intentionally wide.

| game | adversary | method | mean Δ AUC | 95% CI | CI excl. 0+ | gate (mech) |
|---|---|---|---|---|---|---|
| matching_pennies | finite_memory_best_response | fixed_positive | -32.67 | -32.67 [-146.00, +88.00] | no |  |
| matching_pennies | finite_memory_best_response | fixed_negative | -61.33 | -61.33 [-148.00, +50.00] | no |  |
| matching_pennies | finite_memory_best_response | adaptive_beta | -27.33 | -27.33 [-158.00, +122.00] | no | fail (align=0.73, d_eff=0.62) |
| matching_pennies | finite_memory_best_response | adaptive_sign_only | -12.67 | -12.67 [-100.00, +156.00] | no |  |
| matching_pennies | finite_memory_regret_matching | fixed_positive | +19.33 | +19.33 [-6.00, +46.00] | no |  |
| matching_pennies | finite_memory_regret_matching | fixed_negative | +44.00 | +44.00 [-16.00, +114.00] | no |  |
| matching_pennies | finite_memory_regret_matching | adaptive_beta | +42.67 | +42.67 [+14.00, +82.00] | yes | strict (align=0.72, d_eff=0.63) |
| matching_pennies | finite_memory_regret_matching | adaptive_sign_only | +4.00 | +4.00 [-14.00, +26.00] | no |  |
| matching_pennies | hypothesis_testing | fixed_positive | +25.33 | +25.33 [-32.00, +56.00] | no |  |
| matching_pennies | hypothesis_testing | fixed_negative | +6.67 | +6.67 [-24.00, +40.00] | no |  |
| matching_pennies | hypothesis_testing | adaptive_beta | +2.67 | +2.67 [-18.00, +16.00] | no | fail (align=0.60, d_eff=0.81) |
| matching_pennies | hypothesis_testing | adaptive_sign_only | +29.33 | +29.33 [+0.00, +82.00] | no |  |
| rules_of_road | finite_memory_best_response | fixed_positive | -7290.00 | -7290.00 [-9014.00, -4600.00] | no |  |
| rules_of_road | finite_memory_best_response | fixed_negative | +303.33 | +303.33 [+150.00, +528.00] | yes |  |
| rules_of_road | finite_memory_best_response | adaptive_beta | +256.67 | +256.67 [-114.00, +822.00] | no | directional (align=0.84, d_eff=0.39) |
| rules_of_road | finite_memory_best_response | adaptive_sign_only | +93.33 | +93.33 [-92.00, +264.00] | no |  |
| rules_of_road | finite_memory_regret_matching | fixed_positive | -5578.67 | -5578.67 [-6020.00, -4764.00] | no |  |
| rules_of_road | finite_memory_regret_matching | fixed_negative | +1238.67 | +1238.67 [+562.00, +2198.00] | yes |  |
| rules_of_road | finite_memory_regret_matching | adaptive_beta | +1054.67 | +1054.67 [-254.00, +1962.00] | no | directional (align=0.78, d_eff=0.49) |
| rules_of_road | finite_memory_regret_matching | adaptive_sign_only | +840.67 | +840.67 [-376.00, +2390.00] | no |  |
| rules_of_road | hypothesis_testing | fixed_positive | -2471.33 | -2471.33 [-3168.00, -1492.00] | no |  |
| rules_of_road | hypothesis_testing | fixed_negative | -88.00 | -88.00 [-150.00, +34.00] | no |  |
| rules_of_road | hypothesis_testing | adaptive_beta | -196.00 | -196.00 [-326.00, +42.00] | no | fail (align=0.52, d_eff=0.86) |
| rules_of_road | hypothesis_testing | adaptive_sign_only | -338.00 | -338.00 [-526.00, -62.00] | no |  |
| shapley | finite_memory_best_response | fixed_positive | -1211.33 | -1211.33 [-1558.00, -980.00] | no |  |
| shapley | finite_memory_best_response | fixed_negative | +81.67 | +81.67 [+14.00, +164.00] | yes |  |
| shapley | finite_memory_best_response | adaptive_beta | +83.00 | +83.00 [-28.00, +149.00] | no | directional (align=0.85, d_eff=0.40) |
| shapley | finite_memory_best_response | adaptive_sign_only | +145.33 | +145.33 [+63.00, +246.00] | yes |  |
| shapley | finite_memory_regret_matching | fixed_positive | -525.67 | -525.67 [-869.00, -304.00] | no |  |
| shapley | finite_memory_regret_matching | fixed_negative | +473.33 | +473.33 [+196.00, +1005.00] | yes |  |
| shapley | finite_memory_regret_matching | adaptive_beta | +436.00 | +436.00 [+237.00, +830.00] | yes | strict (align=0.84, d_eff=0.45) |
| shapley | finite_memory_regret_matching | adaptive_sign_only | +480.33 | +480.33 [+117.00, +1073.00] | yes |  |
| shapley | hypothesis_testing | fixed_positive | -273.33 | -273.33 [-563.00, -16.00] | no |  |
| shapley | hypothesis_testing | fixed_negative | +209.33 | +209.33 [+48.00, +451.00] | yes |  |
| shapley | hypothesis_testing | adaptive_beta | +295.67 | +295.67 [+180.00, +369.00] | yes | strict (align=0.80, d_eff=0.55) |
| shapley | hypothesis_testing | adaptive_sign_only | +219.67 | +219.67 [+71.00, +348.00] | yes |  |
| strategic_rps | finite_memory_best_response | fixed_positive | -809.00 | -809.00 [-860.00, -759.00] | no |  |
| strategic_rps | finite_memory_best_response | fixed_negative | -78.33 | -78.33 [-213.00, +1.00] | no |  |
| strategic_rps | finite_memory_best_response | adaptive_beta | -744.67 | -744.67 [-868.00, -554.00] | no | fail (align=0.74, d_eff=0.62) |
| strategic_rps | finite_memory_best_response | adaptive_sign_only | -91.33 | -91.33 [-211.00, +16.00] | no |  |
| strategic_rps | finite_memory_regret_matching | fixed_positive | +70.33 | +70.33 [-117.00, +225.00] | no |  |
| strategic_rps | finite_memory_regret_matching | fixed_negative | -365.33 | -365.33 [-463.00, -233.00] | no |  |
| strategic_rps | finite_memory_regret_matching | adaptive_beta | -414.67 | -414.67 [-608.00, -121.00] | no | fail (align=0.54, d_eff=0.88) |
| strategic_rps | finite_memory_regret_matching | adaptive_sign_only | -468.67 | -468.67 [-572.00, -386.00] | no |  |
| strategic_rps | hypothesis_testing | fixed_positive | -512.00 | -512.00 [-890.00, -15.00] | no |  |
| strategic_rps | hypothesis_testing | fixed_negative | -594.67 | -594.67 [-804.00, -455.00] | no |  |
| strategic_rps | hypothesis_testing | adaptive_beta | -703.33 | -703.33 [-799.00, -632.00] | no | fail (align=0.56, d_eff=0.86) |
| strategic_rps | hypothesis_testing | adaptive_sign_only | -401.00 | -401.00 [-854.00, -166.00] | no |  |

## 3. Promotion Gate Outcomes (adaptive_beta vs vanilla)

| game | adversary | mean Δ AUC | CI lo | CI hi | div | align | d_eff | degen | strict | dir | promoted | mode | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matching_pennies | finite_memory_best_response | -27.33 | -158.00 | +122.00 | 0 | 0.73 | 0.62 | yes | no | no | no | fail | mean AUC diff non-positive (-27.33); degenerate game requires strict CI |
| matching_pennies | finite_memory_regret_matching | +42.67 | +14.00 | +82.00 | 0 | 0.72 | 0.63 | yes | yes | no | yes | strict | ok |
| matching_pennies | hypothesis_testing | +2.67 | -18.00 | +16.00 | 0 | 0.60 | 0.81 | yes | no | no | no | fail | degenerate game requires strict CI |
| rules_of_road | finite_memory_best_response | +256.67 | -114.00 | +822.00 | 0 | 0.84 | 0.39 | no | no | yes | yes | directional | ok |
| rules_of_road | finite_memory_regret_matching | +1054.67 | -254.00 | +1962.00 | 0 | 0.78 | 0.49 | no | no | yes | yes | directional | ok |
| rules_of_road | hypothesis_testing | -196.00 | -326.00 | +42.00 | 0 | 0.52 | 0.86 | no | no | no | no | fail | mean AUC diff non-positive (-196.00); mechanism evidence insufficient |
| shapley | finite_memory_best_response | +83.00 | -28.00 | +149.00 | 0 | 0.85 | 0.40 | no | no | yes | yes | directional | ok |
| shapley | finite_memory_regret_matching | +436.00 | +237.00 | +830.00 | 0 | 0.84 | 0.45 | no | yes | no | yes | strict | ok |
| shapley | hypothesis_testing | +295.67 | +180.00 | +369.00 | 0 | 0.80 | 0.55 | no | yes | no | yes | strict | ok |
| strategic_rps | finite_memory_best_response | -744.67 | -868.00 | -554.00 | 0 | 0.74 | 0.62 | no | no | no | no | fail | mean AUC diff non-positive (-744.67); mechanism evidence insufficient |
| strategic_rps | finite_memory_regret_matching | -414.67 | -608.00 | -121.00 | 0 | 0.54 | 0.88 | no | no | no | no | fail | mean AUC diff non-positive (-414.67); mechanism evidence insufficient |
| strategic_rps | hypothesis_testing | -703.33 | -799.00 | -632.00 | 0 | 0.56 | 0.86 | no | no | no | no | fail | mean AUC diff non-positive (-703.33); mechanism evidence insufficient |

## 4. Mechanism Diagnostics (adaptive_beta, mean over 3 seeds)

| game | adversary | align | d_eff | opp_entropy | pol_TV | support_shifts |
|---|---|---|---|---|---|---|
| matching_pennies | finite_memory_best_response | 0.73 | 0.62 | 0.00 | 0.086 | 259 |
| matching_pennies | finite_memory_regret_matching | 0.72 | 0.63 | 0.00 | 0.115 | 344 |
| matching_pennies | hypothesis_testing | 0.60 | 0.81 | 0.00 | 0.298 | 893 |
| rules_of_road | finite_memory_best_response | 0.84 | 0.39 | 0.24 | 0.103 | 655 |
| rules_of_road | finite_memory_regret_matching | 0.78 | 0.49 | 0.33 | 0.107 | 744 |
| rules_of_road | hypothesis_testing | 0.52 | 0.86 | 0.50 | 0.190 | 1815 |
| shapley | finite_memory_best_response | 0.85 | 0.40 | 0.92 | 0.380 | 2816 |
| shapley | finite_memory_regret_matching | 0.84 | 0.45 | 0.92 | 0.398 | 2909 |
| shapley | hypothesis_testing | 0.80 | 0.55 | 0.85 | 0.242 | 2357 |
| strategic_rps | finite_memory_best_response | 0.74 | 0.62 | 0.96 | 0.304 | 2690 |
| strategic_rps | finite_memory_regret_matching | 0.54 | 0.88 | 0.97 | 0.262 | 2555 |
| strategic_rps | hypothesis_testing | 0.56 | 0.86 | 0.77 | 0.263 | 2319 |

## 5. Failure / Divergence Accounting

- **Zero clipped-`adaptive_beta` divergence events** across all 60 candidate-method runs.
- Total NaN counts across all runs: 0

## 6. Selection-Bias Acknowledgement

With only 3 seeds, the strict `ci_lo > 0` gate is conservative; the wide bootstrap distribution at n=3 means the strict criterion is hard to clear even for genuinely-favorable cells. Use the directional fallback only for cells where the mean diff is positive AND the mechanism evidence (alignment > 0.5 or d_eff < γ) directionally supports the interpretation. matching_pennies is mechanism-degenerate (horizon=1 per spec §6.1) and MUST clear the strict CI gate to promote — it cannot rely on directional fallback because alignment / d_eff are uninformative on horizon-1 games.

Cells that cleared each gate:
- **Strict** (`mean>0` AND `ci_lo>0` AND no div): ['(matching_pennies, finite_memory_regret_matching)', '(shapley, finite_memory_regret_matching)', '(shapley, hypothesis_testing)']
- **Directional only** (mean>0 AND mech-OK AND no div, but CI does not exclude 0): ['(rules_of_road, finite_memory_best_response)', '(rules_of_road, finite_memory_regret_matching)', '(shapley, finite_memory_best_response)']

## 7. Stage B2-Main Dispatch (if PROMOTE)

Promoted games: shapley

Promoted adversaries: finite_memory_regret_matching, hypothesis_testing

Per-cell:
- (shapley, hypothesis_testing) — gate_mode=strict, Δ AUC = +295.67 [+180.00, +369.00]
- (shapley, finite_memory_regret_matching) — gate_mode=strict, Δ AUC = +436.00 [+237.00, +830.00]

Stage B2-Main matrix per spec §11.2: 10,000 episodes × 10 seeds × 5 methods (vanilla, fixed_positive, fixed_negative, adaptive_beta, adaptive_sign_only).
Total dispatch cells: 100.

## 8. Generated Figures

- learning_curves: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/dev/learning_curves_all.pdf`
- auc_paired_diff: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/dev/auc_paired_diff_dev.pdf`
- mechanism: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/dev/mechanism_dev.pdf`
- event_aligned: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/dev/event_aligned_rules_of_road_hypothesis_testing.pdf`
- event_aligned: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/adaptive_beta/strategic/figures/dev/event_aligned_strategic_rps_hypothesis_testing.pdf`

## 9. Methodological Caveats

- n=3 seeds: bootstrap CIs are very wide. Use directional fallback with care; final claim strength must come from Stage B2-Main n=10.
- matching_pennies horizon=1 is mechanism-degenerate per spec §6.1; alignment_rate / d_eff cannot serve as mechanism evidence there.
- Bootstrap method: percentile (BCa not feasible at n=3). Reported with fixed seed = 0xB2DEF for byte-stable reproduction.
