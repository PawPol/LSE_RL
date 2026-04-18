# Codex Adversarial Review — Phase III R5

Session ID: 019da1df-cf35-7bc3-aaea-cac5bdf47302
Base: 4fdbf0d (Phase II close commit)
Branch: phase-III/closing
Round: R5 (post R4-fix commit)
Status: completed

## Verdict

Needs-attention: two issues — the expm1/log1p path underflows to -inf for
moderately negative logits, and beta_raw_unclipped ablation schedules cannot
be loaded by the default Phase III path.

## Findings

- [high] expm1/log1p fallback underflows to -inf for large-negative inputs — mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py:467-604
  The new "stable" branch uses `log1p((expm1(br) + gamma*expm1(bv)) / (1+gamma))`
  whenever `max(|br|,|bv|) <= 500`. That branch is not safe on the negative side:
  `expm1(x)` rounds to `-1.0` by about `x <= -38`, so r=v=-40, beta=1 drives
  `inner` to exactly `-1`, making `log1p(inner)` return `-inf` even though the
  original logaddexp form gives `-79.6`. Because Bhat_t >> 40 for Phase III tasks,
  this is reachable on nominal operator inputs.
  **Fixed**: Removed expm1/log1p branch entirely. Now uses `_EPS_BETA=1e-8`
  threshold — classical formula for `|beta| < _EPS_BETA`, logaddexp for all
  larger beta (logaddexp is stable for all finite inputs via its max-subtract
  trick internally).

- [medium] Default schedule loading rejects beta_raw_unclipped ablation artifacts — mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py:216-265
  `BetaSchedule.from_file()` with default `allow_uncertified_cap=False` raises
  ValueError for `beta_raw_unclipped` schedules (which intentionally set
  `beta_cap_t = |beta_raw_t| + 1`). Phase III runners use the default mode so
  these ablation schedules cannot be loaded.
  **Fixed**: `from_file` now auto-detects `"ablation_type"` in the schedule JSON
  and skips the strict certification check for ablation schedules. Production
  schedules (no `ablation_type`) still get the hard reject.
