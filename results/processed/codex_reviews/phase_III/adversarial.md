# Codex Adversarial Review — Phase III R6 (post-merge)

Session ID: manual-subagent-adversarial-r6
Base: 859688c (Phase II merge commit)
Branch: main (phase-III/closing already merged)
Date: 2026-04-18
Focus: operator correctness, certification-box invariance, β=0 collapse,
       logaddexp numerical stability, safe margin logging, BetaSchedule cert

## Summary

All 15 adversarial checks PASS. Operator math, certification chain,
responsibility/rho, β=0 collapse, logaddexp stability, and SafePE convergence
reference are all correct. No BLOCKERs, MAJORs, or disputed issues.

## Findings

### [CORRECT] Operator closed form g_t^safe
safe_weighted_common.py: logaddexp(β*r, β*γ*v + log_γ) scaled by (1+γ)/β
exactly matches the spec formula. Two-path dispatch at _EPS_BETA=1e-8 correct.

### [CORRECT] Responsibility/rho formula
ρ = sigmoid(β(r−v) + log(1/γ)) implemented via scipy.special.expit — numerically
stable for all finite β, r, v.

### [CORRECT] Effective discount
eff_d = (1+γ)(1−ρ). Scalar and batch paths both compute γ exactly for β=0.

### [CORRECT] β=0 collapse
|β| < 1e-8 returns exactly r + γv. No logaddexp called, no cancellation.

### [CORRECT] logaddexp numerical stability
For r=v=-40, β=1: logaddexp(-40, -40 + log_γ) is finite. No expm1/log1p
path exists (removed). Confirmed finite output for all finite inputs.

### [CORRECT] Certification box (κ_t, B̂_t, β_cap_t)
compute_kappa, compute_certified_radii, compute_beta_cap all match the spec
derivation. Backward recursion for B̂_t correct. β_cap formula with κ=γ
special case (→0) handled correctly.

### [CORRECT] Stage extraction from augmented state
stage_from_augmented_state: t = aug_id // n_base. Correct per finite-horizon
convention: state = t * n_base + base_state.

### [CORRECT] Safe margin logging (callbacks.py)
SafeTransitionLogger.after_fit reads swc.last_margin — the exact v_next
passed to compute_safe_target. Algorithm-specific (max-Q vs E_π[Q] vs V^π)
correctly captured via the operator's own field.
NOTE: run_phase3_rl.py inline logger still uses v_next_beta0 (caught by
standard review as P2).

### [CORRECT] BetaSchedule.from_file cert bypass
ablation_type key in JSON → strict cert check skipped. Normal schedules
with reward_bound → strict cert applied. Logic at from_file:275-282 correct.

### [CORRECT] SafePE convergence reference
SafeWeightedPolicyEvaluation evaluates V^π via Q_safe[t, states, π[t]] —
the policy-weighted fixed point, not V* from VI. Correct.

### [CORRECT] Beta schedule length validation
BetaSchedule constructor enforces len(betas)==T and len(Bhat)==T+1. ✓

### [CORRECT] Clip detection tolerance
clip_active uses 1e-15 tolerance. Reasonable; no false positives at machine
epsilon.

### [CORRECT] Negative beta support
Clipping to [-beta_cap, beta_cap] supports pessimistic sign conventions.
Operator formula valid for negative β.

### [CORRECT] n_steps_per_fit=1 constraint
RL runner sets n_steps_per_fit=1, so one transition per fit() call and one
after_fit() logging row per transition. Consistent.

### [CORRECT] Gamma validation at SafeWeightedCommon construction
schedule.gamma vs provided gamma checked at 1e-9 tolerance. Raises ValueError
before any computation if mismatched.
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
