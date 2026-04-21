# Phase IV-A2 activation-search scoring report (FAST pilot)

- Run date: 2026-04-20
- Seed: 42
- Pilot episodes per candidate: 200
- Grid size: 80 (20 per family x 4 families)
- Schedule: calibration-v3 (`build_schedule_v3_from_pilot`)
- Scoring: `score_all_candidates` from
  `experiments/weighted_lse_dp/geometry/task_activation_search.py`
- Artifacts:
  - `phase4A2_candidate_scores_fast.csv` (80 rows)
  - `phase4A2_ratio_analysis.csv` (80 rows)
  - `phase4A2_per_family_stats.json`
  - `phase4A2_summary.json`

## 1. GATE 1 (design-point: `mean_abs_u_pred >= 5e-3`)

**Pass: 0 / 80.**

Not a single config in the Phase IV-A2 grid reaches the 5e-3 design-point
threshold. The whole distribution is compressed into the range
`[0, 2.22e-3]`, with the maximum (`2.216e-3`) coming from
`dense_grid_hazard` at T=20, gamma=0.95. This is roughly a factor of ~2.3x
short of the gate.

Top 5 by `mean_abs_u_pred`:

| idx | family | T | gamma | reward_bound | mean_abs_u_pred | frac_u_ge_5e3 |
|----:|--------|--:|------:|-------------:|----------------:|--------------:|
| 64  | dense_grid_hazard | 20 | 0.95 | 3.00 | 2.216e-3 | 0.20 |
| 72  | dense_grid_hazard | 20 | 0.95 | 3.00 | 2.216e-3 | 0.20 |
| 63  | dense_grid_hazard | 20 | 0.95 | 2.50 | 2.215e-3 | 0.20 |
| 69  | dense_grid_hazard | 20 | 0.95 | 2.50 | 2.215e-3 | 0.20 |
| 62  | dense_grid_hazard | 20 | 0.95 | 2.00 | 2.215e-3 | 0.20 |

Note: `frac_u_ge_5e3 = 0.20` in the grid-hazard rows is the fraction of
*stages* that carry `|u_ref_used_t| >= 5e-3`, not a pass-through on the
mean. The *mean* over stages is still below 5e-3 because the schedule
zeroes out most stages (xi_ref hits `xi_min`, so `u_target` is near
`u_min`).

## 2. Per-family summary (80-candidate aggregate)

All quantities are means across the 20 configs per family.

| family | n | u_min | u_mean | u_max | mean_A_t | ratio_mean | ratio_max | frac_ge_5e3_mean | info_frac_mean |
|--------|--:|------:|-------:|------:|---------:|-----------:|----------:|-----------------:|---------------:|
| dense_chain_cost  | 20 | 6.53e-4 | 1.01e-3 | 1.47e-3 | 28.12 | 0.0287 | 0.0407 | 0.073 | 0.194 |
| dense_grid_hazard | 20 | 1.86e-3 | 2.14e-3 | 2.22e-3 | 39.17 | 0.0180 | 0.0216 | 0.165 | 0.450 |
| shaped_chain      | 20 | 0.00e+0 | 4.20e-4 | 1.30e-3 | 23.61 | 0.0526 | 0.0936 | 0.035 | 0.069 |
| two_path_chain    | 20 | 9.25e-4 | 9.87e-4 | 1.53e-3 | 20.69 | 0.0761 | 0.1217 | 0.103 | 0.113 |

Observations:

- `dense_grid_hazard` has the highest `u_mean` (~2.1e-3) but the largest
  `mean_A_t` (~39.2) and consequently the *smallest* `ratio_max` (0.022).
  The grid is just too small to move the ratio even with ~45%
  informative stages.
- `two_path_chain` has the highest `ratio_max` (0.1217) and `ratio_mean`
  (0.076), driven by large raw `mean|margin|` (~2.3 to 3.0) from the
  risky-event branch. But `mean_A_t` is still ~20.7, so the ratio stays
  an order of magnitude below the 5e-3 target when re-expressed as
  `|beta * margin|`.
- `shaped_chain` is intermediate. `dense_chain_cost` is uniformly weak.
- `u_min == 0` for `shaped_chain` means at least one config produces a
  strictly zero `u_ref_used_t` vector — likely because the classical
  V*-pilot lands in a degenerate regime (V* >= r always) giving
  near-zero `xi_ref` at every stage.

## 3. Top 5 by `ratio_margin_over_A = mean|margin| / mean A_t`

All are `two_path_chain` configs at T=20.

| rank | idx | family | T | gamma | reward_bound | mean|margin| | mean_A_t | ratio |
|-----:|----:|--------|--:|------:|-------------:|-------------:|---------:|------:|
| 1 | 53 | two_path_chain | 20 | 0.97 | 1.10 | 2.281 | 18.74 | 0.1217 |
| 2 | 52 | two_path_chain | 20 | 0.95 | 1.10 | 1.977 | 16.94 | 0.1167 |
| 3 | 58 | two_path_chain | 20 | 0.97 | 1.60 | 3.038 | 27.26 | 0.1115 |
| 4 | 57 | two_path_chain | 20 | 0.95 | 1.60 | 2.637 | 24.65 | 0.1070 |
| 5 | 51 | two_path_chain | 20 | 0.97 | 1.10 | 1.777 | 18.74 | 0.0948 |

## 4. Key finding: does the `mean|margin|/A_t` analysis refute GATE 2a at T=20?

**Yes. Zero configs in the Phase IV-A2 grid have `ratio_margin_over_A >= 0.25`.**

The maximum observed ratio is `0.1217` (idx=53, `two_path_chain`), which
is roughly half the 0.25 heuristic target and would require a ~2x lift
to clear it. Moreover, inspecting the actual schedule for that
top-ratio candidate reveals a stronger structural blocker than the
ratio heuristic suggests:

- `beta_used_t` is **zero on 18 of the 20 stages** (t=0..17) and only
  non-zero at t=18 (`beta = -3.55e-3`) and t=19 (`beta = -6.32e-3`).
- On those two non-zero stages, the predicted stagewise
  `|beta_t| * mean|margin_t|` peaks at **4.62e-3**, still below the
  5e-3 GATE 2a threshold even before the informative+aligned filter
  shrinks the replay sample.
- Averaged over all 20 stages, the predicted mean `|beta * margin|` is
  **4.57e-4** — two orders of magnitude below the gate.

This confirms the structural concern raised before the run: because
`mean|u_replay| = u_ref_used_t / A_t * mean|margin|` and the
calibration-v3 schedule enforces `A_t` proportional to `reward_bound`
(all current configs have `reward_bound >= 1.05`), scaling up
`mean|margin|` by dense rewards only buys a linear factor while `A_t`
moves in lockstep. At T=20 the late-stage `A_t` collapses
(`A_t[T-1] = reward_bound`), but the schedule also drives `u_ref_used`
down via the trust-region / safe-cap minimums, so the net
`beta = u_ref/xi / A_t` stays small.

**No candidate reaches ratio >= 0.25.** A T=20 run from the current
phase4A2 families cannot pass GATE 2a under calibration-v3 as
specified. Passing GATE 2a would require either:

1. a longer horizon (T=30 is already in the grid and is strictly worse
   by construction — `A_t[0] = reward_bound * c_R` grows with T),
2. a schedule variant that does not couple `A_t` to `reward_bound`
   (i.e., a different normalisation in the fixed-point step), or
3. a task family whose classical V*-pilot produces margins that scale
   *faster* than `reward_bound` (e.g., margins derived from
   multi-modal return distributions rather than per-step shaped rewards).

## 5. Errors / data integrity

- 0 / 80 candidates raised exceptions during scoring.
- All 80 rows have populated `pilot_data`, `schedule`, and `scoring`.
- CSV line counts: 81 (header + 80) for both the scores CSV and the
  ratio analysis CSV.
- Per-family record counts: 20 / 20 / 20 / 20 as expected from the
  subsampler.

## 6. Recommendation

Do not advance Phase IV-B on the strength of the phase4A2 grid. GATE 1
is currently failing by 2.3x at the design-point. Even before running
counterfactual replay, the predicted GATE 2a ceiling
(`|beta * margin|` across informative stages) caps at roughly 4.6e-3,
below the 5e-3 threshold. The structural ratio argument identifies the
`A_t` coupling to `reward_bound` as the binding constraint — addressing
this is a calibration-design question, not a task-search question.
