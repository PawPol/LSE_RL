# Phase IV-A Repair Pass — Natural-Coordinate Scheduler, Stagewise Pilot, and Honest Replay Gate

This document is for the coding agent. Treat it as the next required implementation spec before Phase IV-B or any Phase IV-A2 task redesign.

Current status:

1. Phase I--IV-A tests pass under the current filtered test suite.
2. The Option C gate is implemented and currently reports that zero families are Phase IV-B eligible.
3. The current failure is not yet a scientific conclusion, because code review found likely calibration and replay issues that can directly explain the design-point versus replay activation mismatch.
4. Do **not** proceed to Phase IV-B.
5. Do **not** start Phase IV-A2 task redesign yet.
6. First complete this Phase IV-A repair pass and rerun the existing Phase IV-A gate.

The central issue to fix is that the Phase IV-A scheduler must use the natural-coordinate definition

```text
A_t = R_max + Bhat[t+1]
xi_t = (r - v) / A_t
theta_t = beta_t * A_t
u_t = beta_t * (r - v) = theta_t * xi_t
```

The current code appears to use `R_max` instead of `A_t` in parts of the schedule calibration. This can artificially inflate design-point activation and make deployed `beta` too small, which is consistent with the current observed gap:

```text
mean_abs_u_pred ≈ 5.17e-3
mean_abs_u_replay_informative ≈ 4.6e-5
```

The goal of this repair pass is to determine whether the current Phase IV-A tasks truly fail replay activation, or whether the failure is caused by calibration/replay implementation defects.

---

## 0. Non-negotiable workflow rules

1. Start by updating `tasks/todo.md` with this repair plan.
2. Add every discovered bug and prevention rule to `tasks/lessons.md`.
3. Do not overwrite Phase I/II/III artifacts.
4. Do not lower activation thresholds.
5. Do not change the main Safe TAB operator.
6. Do not proceed to Phase IV-B until the repaired Phase IV-A gate is rerun and the per-family eligibility report is regenerated.
7. Do not start Phase IV-A2 task redesign until these repairs are complete and the current task suite has been retested.
8. Selection and eligibility must still use only classical/DP pilot data, certification diagnostics, and counterfactual replay diagnostics. Do not use safe learning returns.

---

## 1. Repair target B1: compute `xi_ref_t` using `A_t`, not `R_max`

### Problem

The current scheduler appears to compute aligned normalized margins as something like:

```python
a_t = sign_family * margins / r_max
xi_ref_t = Q75(a_t | a_t > 0)
```

This violates the Phase IV natural-coordinate definition.

The correct definition is:

```text
A_t = R_max + Bhat[t+1]
a_t = sign_family * margin_t / A_t
xi_ref_t = Q75(a_t | a_t > 0)
```

where `margin_t = r_t - v_ref_next_t` and `v_ref_next_t` must be the correct finite-horizon continuation value.

### Required implementation

Modify the Phase IV calibration code, especially:

```text
experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py
```

so that final `xi_ref_t` is computed using `A_t`.

Because `A_t` depends on `Bhat_t`, and `Bhat_t` depends on `alpha_t`, use the fixed-point structure:

```text
1. Initialize alpha_t from a rough preliminary informativeness pass.
2. Compute kappa_t.
3. Compute Bhat_t by backward recursion.
4. Compute A_t = reward_bound + Bhat[t+1].
5. Recompute xi_ref_t = Q75(sign_family * margin_t / A_t | positive aligned).
6. Recompute p_align_t and informativeness I_t.
7. Recompute u_target_t, trust caps, safe caps, theta_used_t, and beta_used_t.
8. Update alpha_t only if feasibility requires it.
9. Iterate for 2--4 fixed-point iterations or until stable.
10. Emit final schedule fields using the final A_t and xi_ref_t.
```

Do not use `R_max` as the final denominator for `xi_ref_t`. It may only be used for a first initialization if needed.

### Required invariants

Add tests or assertions verifying:

```text
xi_ref_t == Q75(sign_family * margin_t / A_t | sign_family * margin_t > 0)
```

within numerical tolerance on a synthetic calibration example.

Also verify:

```text
u_ref_used_t == theta_used_t * xi_ref_t
beta_used_t == theta_used_t / A_t
u_ref_used_t == beta_used_t * (A_t * xi_ref_t)
```

---

## 2. Repair target B2: replace stationary `V_star` pilot with finite-horizon stagewise `V[t, s]`

### Problem

The pilot currently appears to use a single continuation vector:

```python
V_star[s]
```

and margins are computed as:

```python
margin = r - V_star[next_state]
```

This is wrong for the finite-horizon stage-indexed setting. The correct continuation at stage `t` is:

```text
V[t+1, s_next]
```

### Required implementation

Modify the pilot computation in:

```text
experiments/weighted_lse_dp/geometry/task_activation_search.py
```

or the relevant helper modules so that the pilot computes finite-horizon tables:

```text
V[t, s]
Q[t, s, a]
pi[t, s]
```

with backward induction:

```text
V[T, :] = 0
Q[t, s, a] = E[r(s,a,s') + gamma_base * V[t+1, s']]
V[t, s] = max_a Q[t, s, a]
pi[t, s] = argmax_a Q[t, s, a]
```

During pilot rollout and margin collection, use:

```python
v_next = V_table[t + 1, next_state]
margin = reward - v_next
```

The pilot action should be chosen using the stagewise policy:

```python
action = pi_table[t, state]
```

or an explicitly configured exploratory policy, but the logged reference value must remain stagewise.

### Required tests

Add a synthetic finite-horizon MDP test where `V[t, s]` differs across stages, and verify that the pilot margins use `V[t+1, s_next]`, not a stationary vector.

Required test properties:

1. `V[t, s]` changes with `t`.
2. Logged margin at transition `(t, s, a, r, s')` equals `r - V[t+1, s']`.
3. The old stationary-vector formula would give a different value.

---

## 3. Repair target B3: fix cap-binding flags

### Problem

The current `safe_clip_active_t` / `trust_clip_active_t` logic appears inverted or incomplete.

If the deployed reference shift is

```text
u_used = min(u_target, u_tr_cap, U_safe_ref)
```

then cap binding should be identified by which argument attains the minimum.

### Required implementation

Use the following logic, with tolerance `tol = 1e-10` or similar:

```python
limit_without_trust = np.minimum(u_target, U_safe_ref)
trust_clip_active = u_tr_cap < limit_without_trust - tol

limit_without_safe = np.minimum(u_target, u_tr_cap)
safe_clip_active = U_safe_ref < limit_without_safe - tol

target_unclipped = u_target <= np.minimum(u_tr_cap, U_safe_ref) + tol
```

If both caps are effectively tied, log both as active only if both are within tolerance of `u_used`. Also log a human-readable binding reason:

```text
target | trust | safe | trust_safe_tie | other_tie
```

### Required tests

Create synthetic vectors covering:

1. target binds;
2. trust cap binds;
3. safe cap binds;
4. trust and safe tie;
5. all three tie.

Verify all flags and binding labels.

---

## 4. Repair target B4: make counterfactual replay numerics match the core operator

### Problem

Counterfactual replay currently computes the target using a direct `logaddexp(beta*r, beta*v + log_gamma)` form and a very small beta tolerance. Phase IV-A uses very small beta values, so replay should match the core Safe TAB operator numerics exactly.

### Required implementation

Modify:

```text
experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py
```

so target evaluation uses the same stable formula and tolerance as the core Safe TAB implementation.

Preferred stable formula:

```text
u = beta * (r - v)

g_safe(r, v) = (1 + gamma) * v
             + ((1 + gamma) / beta)
               * [logaddexp(u, log(gamma)) - log1p(gamma)]
```

with Taylor/classical fallback for small beta:

```text
if |beta| <= beta_tol:
    g_safe ≈ r + gamma * v
             + gamma / (2 * (1 + gamma)) * beta * (r - v)^2
```

Use the same `beta_tol` as the core operator, e.g. `1e-8` if that is the existing convention.

### Required logging

For every replay transition, log:

```text
beta_used
margin = r - v_next
natural_shift_direct = beta_used * margin
log_abs_beta
log_abs_margin
log_abs_natural_shift
natural_shift_signed_log_product
relative_error_direct_vs_log_product
rho_used
effective_discount_used
safe_target
classical_target_same_gamma_base
safe_target_gap_same_gamma_base
```

The log-product diagnostic should be used for auditing, not as the main safe-target formula.

### Required tests

1. Replay target matches the core `SafeWeightedCommon` target on a grid of `(r, v, beta, gamma)` values.
2. Direct natural shift and signed log-product natural shift agree except at exact zeros or underflow-safe boundaries.
3. Taylor fallback agrees with finite-difference / exact formula near beta zero.

---

## 5. Repair target B5: verify time augmentation and task consistency

### Problem

Phase IV task factories appear inconsistent about whether they return time-augmented environments. Some chain tasks are time-augmented, while grid/taxi tasks may not be.

For Phase IV-B this is unacceptable unless the runner explicitly applies time augmentation to all finite-horizon RL tasks.

### Required action

Add an invariant check:

```text
All Phase IV-B RL environments must expose stage-aware state or an explicit stage feature.
```

For Phase IV-A pilots, if the environment is not time-augmented and stage is derived from step index, log this clearly.

Before Phase IV-B starts, either:

1. make all Phase IV-B RL task factories return stage-aware environments; or
2. make the Phase IV-B runner wrap all finite tasks in a time-augmentation wrapper consistently.

### Required tests

For each selected Phase IV task family, verify:

1. stage can be decoded from the state or observation;
2. the decoded stage equals the rollout step index;
3. stage resets to zero after environment reset;
4. horizon termination is consistent.

---

## 6. Repair target B6: verify reward modifications propagate to RL environments

### Problem

Some task factories mutate `mdp_base.r` after constructing `mdp_rl`. If `mdp_rl` copies the base MDP internally, reward changes may not propagate to the RL environment.

### Required action

For every Phase IV task factory with post-construction reward edits, verify both base and RL environments see the same reward table or transition reward behavior.

In particular, check:

```text
make_p4_chain_sparse_credit
```

when `step_cost != 0`.

### Required tests

Create a test where `step_cost` is nonzero and verify:

1. the base finite MDP reward table includes the step cost;
2. an RL rollout observes the same modified reward;
3. the calibration/pilot code and replay code see the same reward values.

---

## 7. Rerun requirements after repairs

After completing B1--B6, rerun the current Phase IV-A pipeline on the existing task suite before adding any new task families.

Regenerate:

```text
results/weighted_lse_dp/phase4/task_search/candidate_scores.csv
results/weighted_lse_dp/phase4/task_search/selected_tasks.json
results/weighted_lse_dp/phase4/counterfactual_replay/all_replay_summaries.json
results/weighted_lse_dp/phase4/activation_report/family_eligibility.json
results/weighted_lse_dp/phase4/activation_report/phase4A_gate_report.md
```

Then rerun:

```bash
PYTHONPATH=. python3 scripts/overnight/check_gate.py --phase IV-A
PYTHONPATH=. python3 -m pytest tests/ --tb=short -k "not phase4B and not phase4C"
```

Do not hide failures by adding broad ignores except for explicit Phase IV-B/C placeholder tests.

---

## 8. Required before/after report

Produce a concise before/after report comparing the pre-repair and post-repair Phase IV-A values.

For each family, report:

```text
family
T
gamma_base
reward_bound
A_t median
xi_ref_t median
beta_used_t median
u_ref_used_t median
mean_abs_u_pred
mean_abs_u_replay_global
mean_abs_u_replay_informative
median_abs_u_replay_informative
frac_informative(|u| >= 5e-3)
replay top-quartile mean_abs_u
ratio replay_informative_mean / mean_abs_u_pred
binding cap summary
IV-B eligibility
```

Also report the same diagnostics by informative stage for the best family.

The report must explicitly answer:

1. Did `xi_ref_t` shrink after switching from `R_max` normalization to `A_t` normalization?
2. Did `theta_used_t` and `beta_used_t` increase as expected?
3. Did realized replay activation move closer to design-point activation?
4. Does any family now pass both design-point and informative-replay gates?
5. If no family passes, is the remaining failure due to small margins, trust clipping, safe clipping, or task structure?

---

## 9. Phase IV-B decision rule after repairs

After the repair rerun:

### If at least one family passes both gates

Freeze that family as Phase IV-B eligible and proceed only with restricted Phase IV-B on eligible families.

Phase IV-B may include non-eligible families only as:

```text
weak_activation_control
inactive_negative_control
appendix_sanity
```

### If no family passes both gates

Do not proceed to Phase IV-B.

Then start Phase IV-A2 task redesign with dense-margin / shaped-reward / two-path tradeoff tasks. Use the existing Option C gate unchanged.

---

## 10. Expected commit and status report

Commit the repair pass with a message like:

```text
fix(phase4A): repair natural-coordinate calibration and stagewise replay activation
```

Final status report must include:

1. commit hash;
2. test command and pass/fail count;
3. gate command and pass/fail result;
4. list of files changed;
5. before/after activation table;
6. remaining blockers, if any;
7. explicit recommendation: proceed to Phase IV-B or proceed to Phase IV-A2.

Do not mark Phase IV-A complete unless at least one family is IV-B eligible under the repaired Option C gate.
