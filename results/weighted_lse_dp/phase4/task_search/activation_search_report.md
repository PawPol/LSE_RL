# Phase IV-A Activation Search Report (mainline rerun)

- Seed: 42
- Mainline pilot episodes: 1000
- Appendix pilot episodes: 200
- Mainline horizon: T = 20
- Appendix horizons: T in [5, 10]
- Mainline candidates scored: 34
- Mainline selected: 6
- Appendix candidates scored: 68
- Appendix selected: 4
- Generated: 2026-04-20T01:31:49Z

## What changed vs. the 200-episode run

The original activation search used `n_pilot_episodes = 200`. At that budget, the trust-region confidence factor `c_t = (n_t / (n_t + tau_n)) * sqrt(p_align_t)` never releases enough for `u_tr_cap` to clear the `|u| >= 5e-3` gate: the best observed value was `mean_abs_u = 0.00356`, giving a FAILED activation gate (10/11 conditions).

This rerun uses `n_pilot_episodes = 1000` on the T=20 mainline subset of the search grid, keeping `tau_n = 200`. The extra pilot budget is the only change — the operator, schedule geometry, and gate threshold are unchanged.

## Gate status

- Best mainline `mean_abs_u_pred` = **0.00517** (threshold 0.005)
- Best mainline `frac(|u| >= 5e-3)` = **0.450** (threshold 0.1)
- Gate status: **PASS** (before rerun: FAIL)

## Suites

- Mainline families (3): `chain_jackpot, chain_sparse_credit, grid_hazard`
- Appendix families (2): `chain_sparse_credit, grid_hazard`
- See `activation_report/binding_cap_summary.json` for the per-family breakdown of trust-region vs safe-reference clipping.

## Mainline per-family ranking

### Family: `chain_catastrophe`
- Candidates: 9
- Selected: 0
- Errors: 0

| Rank | Idx | Score | mean_abs_u | frac(|u|>=5e-3) | Status |
|------|-----|-------|-----------|-----------------|--------|
| 1 | 15 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 2 | 16 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 3 | 17 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 4 | 18 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 5 | 19 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 6 | 20 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 7 | 21 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 8 | 22 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 9 | 23 | -3.3197 | 0.000000 | 0.0000 | rejected |

### Family: `chain_jackpot`
- Candidates: 9
- Selected: 2
- Errors: 0

| Rank | Idx | Score | mean_abs_u | frac(|u|>=5e-3) | Status |
|------|-----|-------|-----------|-----------------|--------|
| 1 | 8 | 3.3110 | 0.002409 | 0.3000 | SELECTED |
| 2 | 11 | 2.0663 | 0.002409 | 0.3000 | SELECTED |
| 3 | 7 | 1.4982 | 0.001704 | 0.1500 | rejected |
| 4 | 6 | 1.3633 | 0.001623 | 0.1000 | rejected |
| 5 | 14 | 1.3493 | 0.002454 | 0.3000 | rejected |
| 6 | 10 | 0.6553 | 0.001704 | 0.1500 | rejected |
| 7 | 9 | 0.4882 | 0.001623 | 0.1000 | rejected |
| 8 | 13 | 0.3790 | 0.001881 | 0.2500 | rejected |
| 9 | 12 | -0.0733 | 0.001629 | 0.1000 | rejected |

### Family: `chain_sparse_credit`
- Candidates: 6
- Selected: 2
- Errors: 0

| Rank | Idx | Score | mean_abs_u | frac(|u|>=5e-3) | Status |
|------|-----|-------|-----------|-----------------|--------|
| 1 | 5 | 9.1125 | 0.005175 | 0.4500 | SELECTED |
| 2 | 3 | 1.7458 | 0.002072 | 0.1500 | SELECTED |
| 3 | 0 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 4 | 1 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 5 | 2 | -3.3197 | 0.000000 | 0.0000 | rejected |
| 6 | 4 | -3.3197 | 0.000000 | 0.0000 | rejected |

### Family: `grid_hazard`
- Candidates: 9
- Selected: 2
- Errors: 0

| Rank | Idx | Score | mean_abs_u | frac(|u|>=5e-3) | Status |
|------|-----|-------|-----------|-----------------|--------|
| 1 | 26 | 2.8779 | 0.003764 | 0.3500 | SELECTED |
| 2 | 25 | 2.8763 | 0.003764 | 0.3500 | SELECTED |
| 3 | 24 | 2.8748 | 0.003764 | 0.3500 | rejected |
| 4 | 29 | 2.7038 | 0.003764 | 0.3500 | rejected |
| 5 | 28 | 2.7021 | 0.003764 | 0.3500 | rejected |
| 6 | 27 | 2.7005 | 0.003764 | 0.3500 | rejected |
| 7 | 32 | 2.6170 | 0.003764 | 0.3500 | rejected |
| 8 | 31 | 2.6150 | 0.003764 | 0.3500 | rejected |
| 9 | 30 | 2.6133 | 0.003764 | 0.3500 | rejected |

### Family: `regime_shift`
- Candidates: 1
- Selected: 0
- Errors: 0

| Rank | Idx | Score | mean_abs_u | frac(|u|>=5e-3) | Status |
|------|-----|-------|-----------|-----------------|--------|
| 1 | 33 | -3.3197 | 0.000000 | 0.0000 | rejected |
