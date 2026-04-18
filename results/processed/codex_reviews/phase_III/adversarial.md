# Codex Adversarial Review — Phase III R2

Session ID: 019da1a0-8d9f-7420-84c2-f65b1a6a2c19
Base: 4fdbf0d (Phase II close commit)
Branch: phase-III/closing
Round: R2 (post R1-fix commit 11f2189)
Focus: "challenge operator correctness (g_t^safe closed form, responsibility, local derivative), certification-box invariance (kappa_t, B_hat_t, beta_cap), β=0 collapse, and logaddexp numerical stability, per docs/specs/phase_III_safe_weighted_lse_experiments.md."
Status: completed

## Verdict

No-ship. The safe operator path still trusts schedule JSONs without enforcing the certification invariants it claims, the RL runner does not fail fast on horizon/schema skew, and at least one new safe TD agent is not load-safe after serialization.

## Findings

- [high] Certified-safety claims are unenforced when schedules are loaded from disk — mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py:62-117
  `BetaSchedule` only checks array lengths and then exposes `beta_used_t`, `alpha_t`, `kappa_t`, and `Bhat_t` directly to the operator. It never verifies that `beta_used_t == clip(beta_raw_t, -beta_cap_t, beta_cap_t)`, that `alpha_t` stays in `[0,1)`, or that `kappa_t`/`Bhat_t`/`beta_cap_t` actually satisfy the certification recurrences from the spec. Because the runners load schedule JSONs verbatim and `compute_safe_target` uses `beta_used_at(t)` as-is, a stale or hand-edited schedule can silently bypass clipping while downstream logs still report the old certificate fields. The likely impact is shipping runs that are presented as certified-safe even though the contraction bound no longer holds.
  Recommendation: On load, recompute `kappa_t`, `Bhat_t`, and `beta_cap_t` from `alpha_t`, `reward_bound`, and `gamma`, and assert they match the serialized values within tolerance; also assert `beta_used_t` equals the clipped raw schedule and that caps are non-negative.

- [medium] Safe RL path accepts mismatched schedule horizons and will fail late or mis-run under schedule/task skew — experiments/weighted_lse_dp/runners/run_phase3_rl.py:778-807
  The RL runner loads `schedule.json` and records `schedule_T` in metadata, but unlike the DP path it never checks `schedule.T == horizon` before constructing the agent. The safe TD base then decodes `t` from the augmented state and indexes the schedule on every update. If a task config and schedule file drift out of sync, the run will either crash only after reaching a later stage (`IndexError` once `t >= schedule.T`) or silently ignore extra schedule entries if the schedule is too long.
  Recommendation: Immediately after `BetaSchedule.from_file`, assert `schedule.T == horizon` and reject mismatched files with a clear error, mirroring the DP constructors.

- [medium] Serialized SafeQLearning agents lose the safe operator state on reload — mushroom-rl-dev/mushroom_rl/algorithms/value/td/safe_q_learning.py:44-50
  `SafeQLearning` explicitly marks `_schedule` and `_swc` as non-serialized and does not provide a `_post_load` hook to rebuild them. The inherited `TD._post_load` only reconnects `policy.Q`; it does not restore the safe schedule/helper. A reloaded agent can therefore no longer compute safe targets or expose correct `last_*` instrumentation, which breaks checkpoint resume and any evaluation path that loads saved agents.
  Recommendation: Persist enough state to reconstruct the safe helper, and implement `_post_load` to rebuild `_swc` from the saved schedule and `n_base` before the agent is used again.
