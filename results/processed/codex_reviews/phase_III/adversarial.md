# Codex Adversarial Review — Phase III R3

Session ID: 019da1c7-1367-7b93-ab4b-5d9398e39af1
Base: 4fdbf0d (Phase II close commit)
Branch: phase-III/closing
Round: R3 (post R2-fix commit 424a73d)
Focus: "challenge operator correctness (g_t^safe closed form, responsibility, local derivative), certification-box invariance (kappa_t, B_hat_t, beta_cap), β=0 collapse, and logaddexp numerical stability, per docs/specs/phase_III_safe_weighted_lse_experiments.md."
Status: completed

## Verdict

No-ship: the Phase III path still accepts schedules that exceed the certified beta cap, and its RL logging records safe-table values under `*_beta0` names, so both the safety invariant and the baseline telemetry can silently drift without detection.

## Findings

- [high] Phase III safe runs write safe Q/V estimates into fields labeled as beta=0 baselines — experiments/weighted_lse_dp/common/callbacks.py:125-142
  `TransitionLogger.__call__` reads `q_current` and `v_next` directly from `self._agent.Q` and stores them as `q_current_beta0` / `v_next_beta0`. `SafeTransitionLogger` then inherits that payload shape for safe agents. In Phase III, those arrays are later turned into `margin_beta0`, `td_target_beta0`, and `td_error_beta0`, so the artifacts and downstream calibration summaries are labeled as classical-beta0 references even though they come from the evolving safe agent. That makes regressions against the beta=0 baseline hard to detect and can contaminate any analysis that trusts the field names.
  Recommendation: For safe runs, either compute these arrays from an actual beta=0 reference estimator/policy, or stop emitting them under `*_beta0` names. At minimum, split the safe and classical telemetry schemas so downstream code cannot mistake safe-table values for beta=0 baselines.

- [high] Schedule validation explicitly permits beta caps larger than the certified cap — mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py:155-183
  The loader recomputes the certified `kappa_t`, `Bhat_t`, and `beta_cap_t`, but then intentionally accepts any stored `beta_cap_t` that is element-wise larger than the certified value. Because `beta_used_t` is only checked against the stored cap, a schedule can deploy `|beta_used_t| > beta_cap_t^{cert}` and still load successfully. That breaks the exact clipping rule in the spec and voids the certification-box / local-derivative guarantee precisely on the path the main runners use (`BetaSchedule.from_file`).
  Recommendation: Reject any production schedule whose stored `beta_cap_t` differs from the recomputed certified cap, or gate the permissive override behind an explicit test-only flag. Also recompute `beta_used_t` from the certified cap on load so an unsafe schedule cannot be deployed by construction.
