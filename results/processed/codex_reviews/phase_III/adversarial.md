# Codex Adversarial Review — Phase III

Session ID: 019da120-c6a2-73b2-af18-aeeaa85903fc
Base: 4fdbf0d (Phase II close commit)
Branch: phase-III/closing
Focus: "challenge operator correctness (g_t^safe closed form, responsibility, local derivative), certification-box invariance (kappa_t, B_hat_t, beta_cap), β=0 collapse, and logaddexp numerical stability, per docs/specs/phase_III_safe_weighted_lse_experiments.md."
Status: completed (~5m)

## Verdict

No-ship: the Phase III RL telemetry is off by one update, the certification cap is built from sampled reward maxima instead of the task's configured bound, and most safe planners still accept mismatched schedules without enforcing the invariants the certification depends on.

## Findings

- [high] Phase III safe-transition logs record the previous update's operator diagnostics, not the current transition's — experiments/weighted_lse_dp/runners/run_phase3_rl.py:338-360
  `_AutoSafeEventLogger.__call__` appends `agent.swc.last_*` immediately inside `callback_step`, but `Core._run` invokes `callback_step(sample)` before `agent.fit(dataset)`. With `n_steps_per_fit=1`, the safe fields written for each row (`safe_stage`, `safe_beta_*`, `safe_rho`, `safe_effective_discount`, `safe_target`, `safe_td_error`) therefore belong to the previous transition/update, while `reward`, `q_current_beta0`, and `v_next_beta0` belong to the current sample. This breaks per-transition observability exactly where Phase III is supposed to verify responsibility/derivative behavior, and it can hide certification violations because the logged safe quantities are misaligned rather than missing. This conclusion is directly grounded by the logger code here and the call order in `mushroom_rl/core/core.py:108-117`.
  Recommendation: Capture safe diagnostics after the agent update, not in `callback_step` before `fit()`. A concrete fix is to log safe fields from a post-fit callback or to have the agent emit the just-used safe quantities together with the sample so the base transition fields and `swc.last_*` come from the same update.

- [high] Certification uses `empirical_r_max` instead of the absolute task reward bound required by the spec — experiments/weighted_lse_dp/calibration/build_schedule_from_phase12.py:116-119
  The schedule builder sets `R_max = float(cal["empirical_r_max"])` from the Phase II calibration JSON. The Phase III spec explicitly requires `R_max` to be the configured absolute maximum reward for the task, not whatever the Phase I/II runs happened to observe. If rare jackpot/catastrophe rewards were missed during calibration, `Bhat_t` is underestimated and the derived `beta_cap_t` is too loose, so the claimed certification-box invariance can fail even though the schedule JSON advertises `reward_bound`, `kappa_t`, and `beta_cap_t` as if they were certified.
  Recommendation: Derive `R_max` from task configuration or environment metadata that encodes the true absolute reward bound, and fail schedule construction if only an empirical bound is available.

- [medium] Most safe planners still do not enforce that the loaded schedule matches the MDP horizon/gamma — mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_value_iteration.py:169-188
  `SafeWeightedValueIteration` accepts any `BetaSchedule` and immediately starts indexing it against the MDP horizon, but unlike `SafeWeightedPolicyIteration` it does not verify `schedule.T == horizon`, nor does it check `schedule.gamma` against the MDP discount. A stale or wrong-task schedule can silently run with foreign calibration coefficients, invalidating the beta-cap / kappa guarantees without an upfront failure.
  Recommendation: Centralize schedule validation and enforce it in every safe planner/TD entry point: require `schedule.T == mdp.horizon`, require `abs(schedule.gamma - mdp.gamma) <= tol`, and reject schedules whose certification arrays do not match the runtime horizon.
