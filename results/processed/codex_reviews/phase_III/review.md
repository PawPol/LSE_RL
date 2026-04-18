# Codex Standard Review — Phase III R3

Session ID: (019da1xx — standard review, codex review --base 4fdbf0d)
Base: 4fdbf0d (Phase II close commit)
Branch: phase-III/closing
Round: R3 (post R2-fix commit 424a73d)
Status: completed

## Summary

The new Phase III DP runner omits two configured task families entirely, and it records incorrect convergence targets for all `SafePE` runs. Those issues affect the completeness and correctness of the generated Phase III artifacts.

## Findings

- [P1] Stop skipping `grid_hazard` and `taxi_bonus_shock` in DP runs — experiments/weighted_lse_dp/runners/run_phase3_dp.py:318-320
  The Phase III suite config still declares `dp_algorithms` for `grid_hazard` and `taxi_bonus_shock`, and the file comment above `_RL_ONLY_TASKS` says DP should run on the base MDP for these families. However `_build_run_list()` drops both tasks entirely, and `_run_single()` returns early if they are requested explicitly, so `run_phase3_dp.py --task all` will never produce DP artifacts for two configured paper-suite tasks.

- [P2] Use the evaluated policy's value as `v_exact` for `SafePE` — experiments/weighted_lse_dp/runners/run_phase3_dp.py:519-528
  When `algo_name == "SafePE"`, this branch computes `v_exact` with a `SafeVI` solve and passes that to `DPCurvesLogger`. For policy-evaluation runs, that is the optimal control value of a different problem, not the exact fixed point of the supplied reference policy, so every `supnorm_to_exact` curve and summary for `SafePE` is misreported as error-to-optimal rather than convergence-to-exact-evaluation.
