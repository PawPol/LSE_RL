# Phase V WP0 consistency audit

Generated: 2026-04-24T00:57:56.320047+00:00
Git SHA: `28ea65220603adedb3204f81ac1bd1ddc12200de`
Schema version: 1.0.0

## Summary

- BLOCKER: 83
- MINOR:   18
- INFO:    2

Phases completed: phase1, phase2, phase3, phase4A, phase4B, phase4C

## BLOCKERs

| id | phase | check | artifact | expected | actual |
|---|---|---|---|---|---|
| C-phaseIII-001 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-002 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-003 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-004 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-005 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-006 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-007 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_67` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-008 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_67` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-009 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_83` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-010 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_83` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-011 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-012 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-013 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-014 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-015 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-016 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-017 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_67` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-018 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_67` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-019 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_83` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028461692308e+00 |
| C-phaseIII-020 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafeVI/seed_83` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=59 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826169e-02 |
| C-phaseIII-021 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-022 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826186e-02 |
| C-phaseIII-023 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-024 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826186e-02 |
| C-phaseIII-025 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-026 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826186e-02 |
| C-phaseIII-027 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_67` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-028 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_67` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826186e-02 |
| C-phaseIII-029 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_83` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-030 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafePE/seed_83` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826186e-02 |
| C-phaseIII-031 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-032 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826205e-02 |
| C-phaseIII-033 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-034 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826205e-02 |
| C-phaseIII-035 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-036 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826205e-02 |
| C-phaseIII-037 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_67` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-038 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_67` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826205e-02 |
| C-phaseIII-039 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_83` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.028462341633e+00 |
| C-phaseIII-040 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/chain_jackpot/SafeVI/seed_83` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=57 d_eff=1.028462e+00 kappa=9.902000e-01 diff=3.826205e-02 |
| C-phaseIII-041 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafePE/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030002599094e+00 |
| C-phaseIII-042 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafePE/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=75 d_eff=1.030003e+00 kappa=9.902001e-01 diff=3.980252e-02 |
| C-phaseIII-043 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafePE/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030002599094e+00 |
| C-phaseIII-044 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafePE/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=75 d_eff=1.030003e+00 kappa=9.902001e-01 diff=3.980252e-02 |
| C-phaseIII-045 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafePE/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030002599094e+00 |
| C-phaseIII-046 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafePE/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=75 d_eff=1.030003e+00 kappa=9.902001e-01 diff=3.980252e-02 |
| C-phaseIII-047 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafeVI/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-048 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafeVI/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-049 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafeVI/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-050 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafeVI/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-051 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafeVI/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-052 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_post_shift/SafeVI/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-053 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafePE/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-054 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafePE/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-055 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafePE/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-056 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafePE/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-057 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafePE/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-058 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafePE/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-059 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafeVI/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-060 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafeVI/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-061 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafeVI/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-062 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafeVI/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-063 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafeVI/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.030011148231e+00 |
| C-phaseIII-064 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_regime_shift_pre_shift/SafeVI/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=73 d_eff=1.030011e+00 kappa=9.902000e-01 diff=3.981115e-02 |
| C-phaseIII-065 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafePE/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.010416098429e+00 |
| C-phaseIII-066 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafePE/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=110 d_eff=1.010416e+00 kappa=9.902000e-01 diff=2.021610e-02 |
| C-phaseIII-067 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafePE/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.010416098429e+00 |
| C-phaseIII-068 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafePE/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=110 d_eff=1.010416e+00 kappa=9.902000e-01 diff=2.021610e-02 |
| C-phaseIII-069 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafePE/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.010416098429e+00 |
| C-phaseIII-070 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafePE/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=110 d_eff=1.010416e+00 kappa=9.902000e-01 diff=2.021610e-02 |
| C-phaseIII-071 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafeVI/seed_11` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.010416098429e+00 |
| C-phaseIII-072 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafeVI/seed_11` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=110 d_eff=1.010416e+00 kappa=9.902000e-01 diff=2.021610e-02 |
| C-phaseIII-073 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafeVI/seed_29` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.010416098429e+00 |
| C-phaseIII-074 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafeVI/seed_29` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=110 d_eff=1.010416e+00 kappa=9.902000e-01 diff=2.021610e-02 |
| C-phaseIII-075 | phaseIII | d_eff_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafeVI/seed_47` | max safe_effective_discount_mean <= 1 + 1e-08 | max=1.010416098429e+00 |
| C-phaseIII-076 | phaseIII | cert_bound | `results/weighted_lse_dp/phase3/paper_suite/grid_sparse_goal/SafeVI/seed_47` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=110 d_eff=1.010416e+00 kappa=9.902000e-01 diff=2.021610e-02 |
| C-phaseIV-B-001 | phaseIV-B | cert_bound | `results/weighted_lse_dp/phase4/translation_4a2/dense_chain_cost_0/safe_vi/seed_123` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=17 d_eff=9.958331e-01 kappa=9.614450e-01 diff=3.438805e-02 |
| C-phaseIV-B-002 | phaseIV-B | cert_bound | `results/weighted_lse_dp/phase4/translation_4a2/dense_chain_cost_0/safe_vi/seed_42` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=17 d_eff=9.958795e-01 kappa=9.611501e-01 diff=3.472946e-02 |
| C-phaseIV-B-003 | phaseIV-B | cert_bound | `results/weighted_lse_dp/phase4/translation_4a2/dense_chain_cost_0/safe_vi/seed_456` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=17 d_eff=9.957880e-01 kappa=9.617165e-01 diff=3.407148e-02 |
| C-phaseIV-B-004 | phaseIV-B | cert_bound | `results/weighted_lse_dp/phase4/translation_4a2/dense_chain_cost_1/safe_vi/seed_123` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=17 d_eff=9.958331e-01 kappa=9.614450e-01 diff=3.438805e-02 |
| C-phaseIV-B-005 | phaseIV-B | cert_bound | `results/weighted_lse_dp/phase4/translation_4a2/dense_chain_cost_1/safe_vi/seed_42` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=17 d_eff=9.958795e-01 kappa=9.611501e-01 diff=3.472946e-02 |
| C-phaseIV-B-006 | phaseIV-B | cert_bound | `results/weighted_lse_dp/phase4/translation_4a2/dense_chain_cost_1/safe_vi/seed_456` | safe_effective_discount_mean[t] <= kappa_t[t] + 1e-06 for all t | stage=17 d_eff=9.957880e-01 kappa=9.617165e-01 diff=3.407148e-02 |
| C-paper-001 | paper | text_vs_table | `paper clip-fraction claim vs grid_hazard` | clip_fraction(grid_hazard) ~= 0.79 (paper L1856) | recomputed mean clip_fraction = 0.9625 |

## MINORs (post-hoc manifest emission and similar)

| id | phase | check | artifact | expected | actual |
|---|---|---|---|---|---|
| C-phaseI-001 | phaseI | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase1_ablation.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseI-002 | phaseI | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase1_dp.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseI-003 | phaseI | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase1_rl.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseI-004 | phaseI | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase1_smoke.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseII-001 | phaseII | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase2_ablation.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseII-002 | phaseII | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase2_dp.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseII-003 | phaseII | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase2_rl.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIII-078 | phaseIII | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase3_dp.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIII-079 | phaseIII | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase3_rl.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-C-001 | phaseIV-C | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4C_advanced_rl.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-C-002 | phaseIV-C | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4C_certification_ablations.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-C-003 | phaseIV-C | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4C_geometry_dp.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-C-004 | phaseIV-C | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4C_scheduler_ablations.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-001 | phaseIV | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4_activation_search.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-002 | phaseIV | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-003 | phaseIV | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4_diagnostic_sweep.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-004 | phaseIV | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4_dp.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |
| C-phaseIV-005 | phaseIV | manifest_post_hoc | `experiments/weighted_lse_dp/runners/run_phase4_rl.py` | runner emits results/summaries/experiment_manifest.json (spec §9) | no 'experiment_manifest' string found in runner source |

## INFOs

| id | phase | check | artifact | expected | actual |
|---|---|---|---|---|---|
| C-phaseIII-077 | phaseIII | recompute_write | `results/audit/recomputed_tables/phase3/clip_activity.csv` | clip-activity table recomputed | 12 tasks |
| C-phaseIV-B-007 | phaseIV-B | table_diff | `results/processed/phase4b/P4B_A.csv` | recomputed table matches reference | 18 rows match within tolerance |
