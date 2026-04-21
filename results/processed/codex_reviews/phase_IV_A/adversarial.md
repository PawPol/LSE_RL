## Phase IV-A Adversarial Review

Reviewer: Codex adversarial subagent
Date: 2026-04-19
Branch: phase-iv-a/closing
Base commit: 9ca5be4 (fix(operator): add compute_safe_target_ev_batch)

Files examined:
- `experiments/weighted_lse_dp/geometry/adaptive_headroom.py`
- `experiments/weighted_lse_dp/geometry/task_activation_search.py`
- `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py`
- `experiments/weighted_lse_dp/geometry/trust_region.py`
- `experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py`
- `experiments/weighted_lse_dp/runners/aggregate_phase4A.py`
- `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py`
- `results/weighted_lse_dp/phase4/activation_report/global_diagnostics.json`
- `results/weighted_lse_dp/phase4/task_search/selected_tasks.json`
- `docs/specs/phase_IV_A_activation_audit_and_counterfactual.md`

---

### BLOCKER issues (breaks a scientific claim or causes silent wrong results)

- [BLOCKER] `run_phase4_counterfactual_replay.py:155` — **Walrus-operator terminal-step mis-classification**. The condition `if absorbing or (i := step_idx) == horizon - 1:` uses a walrus assignment `i := step_idx` purely as a side-effect inside a boolean short-circuit expression. The walrus clause is evaluated only when `absorbing` is `False` (short-circuit). When `absorbing=True`, `i` is either unset (first iteration) or retains its value from the previous episode's last step. This is dead code at best; at worst, if Python evaluates the walrus on a path where `absorbing=True` was set simultaneously as `step_idx == horizon - 1`, the condition logic is correct by accident. More critically, `i` is never used elsewhere in the scope — it is a spurious binding that does not gate anything. The real defect is a conceptual one: even when the non-absorbing last step is reached (`step_idx == horizon - 1` but `absorbing=False`), `v_next` is forced to 0. But the classical pilot (`run_classical_pilot`) uses a different convention at this boundary: it only forces `v_next = 0` when `i == T_ep - 1 and len(ep_rewards) < max_steps`, i.e., when the episode terminated early. For trajectories that exhaust the horizon normally (non-absorbing), `run_classical_pilot` does NOT zero out `v_next` at the last step, whereas `_run_pilot_with_transitions` does. This creates a systematic margin mismatch between the pilot margins used to build the schedule and the transitions used in replay, violating the counterfactual isolation guarantee for any non-absorbing terminal transitions.

- [BLOCKER] `task_activation_search.py:340` — **Gate metric diluted over all T stages, not informative stages only**. `mean_abs_u_pred = float(np.mean(np.abs(u_ref_used)))` averages over all T stages including uninformative ones. The spec (§5.3 and §13) says the gate thresholds apply globally, but §9.3 test requirement 4 explicitly states "activation thresholds are evaluated only on informative stages." The `global_diagnostics.json` shows `mainline_best_mean_abs_u = 0.005174`, which barely clears the 5e-3 gate for the `chain_sparse_credit` family. If the mean were restricted to informative stages (those with `xi_ref * sqrt(p_align) >= 0.05`), the value could increase or decrease depending on the stage distribution. The issue is that the search-phase acceptance criterion in `select_activation_suite` uses `mean_abs_u_pred` averaged over all T=20 stages, but the gate criterion in `_evaluate_gate` (aggregate) is evaluated over all transitions from replay (not restricted to informative-stage transitions). These are not the same denominator, and neither matches the spec's informative-stage restriction in §9.3/requirement 4. The gate result `mean_abs_u: true` in `global_diagnostics.json` cannot be validated without knowing which denominator was used for `mainline_best_mean_abs_u`.

- [BLOCKER] `phase4_calibration_v3.py:270-272` — **U_safe_ref sign discarded; incorrect binding constraint**. Lines 271-272:
  ```python
  U_safe_abs = np.abs(U_safe_ref_t)
  u_ref_used_arr = np.minimum(np.minimum(u_target_arr, u_tr_cap_arr), U_safe_abs)
  ```
  `U_safe_ref_t = theta_safe_t * xi_ref_t`. `theta_safe_t = log(kappa / (gamma * (1+gamma-kappa)))`. For `kappa > gamma` (which is guaranteed since `alpha > 0`), `theta_safe > 0`. For `xi_ref > 0`, `U_safe_ref > 0`. So `np.abs` has no effect in the normal case. However: `theta_safe_t` is the cert-derived UPPER bound on theta. The spec (§6.10) says the binding constraint is `U_safe_ref_t`, which the code correctly computes as `Theta_safe_t * xi_ref_t`. But then the code uses `np.abs(U_safe_ref_t)` — if for any edge case `U_safe_ref_t` becomes negative (e.g., numerical issue in `log` when the denominator is very close to zero), `np.abs` would flip a binding negative constraint to a large positive value, effectively removing the safety cap. This is a silent correctness hazard. The code should assert `U_safe_ref_t >= 0` elementwise rather than silently taking the absolute value.

---

### MAJOR issues (materially weakens the result but doesn't invalidate it)

- [MAJOR] `phase4_calibration_v3.py:287-292` — **`safe_clip_active` flag is logically inverted / misleading**. The comment says "safe_clip_active: u_ref_used < u_tr_cap (trust wasn't binding) but safe cap was the binding constraint." The condition is:
  ```python
  safe_clip_active = (
      (u_ref_used_arr < U_safe_abs - 1e-10)
      & ~np.array(trust_clip_active)
  )
  ```
  `~trust_clip_active` means "trust was NOT the binding constraint." So the flag fires when (a) the result was below the safe ref AND (b) trust wasn't binding. But when `u_ref_used = min(u_target, u_tr_cap, U_safe_abs)` and trust did NOT clip, then `u_ref_used = min(u_target, U_safe_abs)`. The safe cap is binding if and only if `U_safe_abs < u_target`. But the condition checks `u_ref_used < U_safe_abs - eps`, which is the OPPOSITE: it fires when `u_ref_used` is BELOW `U_safe_abs`, i.e., when something else clipped it further down. The flag `safe_clip_active` is therefore TRUE exactly when the safe cap was NOT the binding constraint (assuming u_target was binding instead), and FALSE when the safe cap IS the binding constraint (u_ref_used == U_safe_abs). This is an inverted diagnostic field that will produce incorrect figures and tables in §12 and incorrect cap-utilization fractions in §13.

- [MAJOR] `trust_region.py:156-163` — **Bisection does not handle very small but nonzero eps_tr correctly; lower bound never narrows below 0**. When `eps_tr` is very small (e.g., ~1e-14, which occurs at early stages with very low alignment probability), `kl_bernoulli(rho(0), p0) = 0` exactly (since `rho(0) = sigmoid(eta0) = 1/(1+gamma) = p0` by construction). So `kl_mid` at `lo=0` is exactly 0, which is < eps_tr, so the bisection keeps pushing lo upward. This is correct. But the issue is: `kl_bernoulli(rho(0), p0)` should be exactly 0 only in exact arithmetic. With floating-point `_sigmoid` and `kl_bernoulli`, there can be tiny numerical residuals. If `kl(rho(0), p0) > eps_tr` due to floating-point residuals when eps_tr is at the sub-1e-10 level, the bisection returns `lo = 0` (since `kl_hi > eps_tr` at `hi=20`), giving `u_tr_cap = 0`. This would force the trust cap to zero at stages with nearly-zero alignment, collapsing those stages to classical behavior. This is not catastrophically wrong (conservative), but it is not the right behavior: when alignment is low and the trust budget is tiny, the cap should approach zero smoothly, and the behavior is correct, but the mechanism is fragile.

- [MAJOR] `task_activation_search.py:413-489` — **Acceptance thresholds in `select_activation_suite` are weaker than the spec gate thresholds**. `select_activation_suite` uses `min_mean_abs_u_pred=2e-3` and `min_frac_active_stages=0.05` as acceptance criteria (search-phase filter). The gate thresholds in the spec §13 are `mean_abs_u >= 5e-3` and `frac >= 0.10`. The search-phase filter allows in tasks that will fail the final gate. This is not unsound per se (tasks pass a preliminary screen, then face the harder gate), but it means the selected suite in `selected_tasks.json` contains tasks (e.g., `chain_jackpot` with `mean_abs_u_pred = 0.0024`, `frac_u_ge_5e3 = 0.30`) that already fail the spec's minimum acceptance criterion for `mean_abs_u >= 2e-3` but only barely, while their `frac(|u|>=5e-3) = 0.30` passes the search gate but should actually be compared against the 10% gate. More concretely: the `chain_jackpot` tasks selected have `mean_abs_u_pred = 0.0024`, which is BELOW the spec §5.3 `mean_abs_u_pred >= 2e-3` threshold but only by rounding. This is a borderline admission that, when replayed, is likely to fail the §13 gate of 5e-3. The `global_diagnostics.json` shows `mainline_best_mean_abs_u = 0.005174`, which is the best task — the jackpot tasks may be pulling the mean down in the final gate evaluation.

- [MAJOR] `adaptive_headroom.py:324-340` — **Fixed-point convergence criterion is based on theta-ratio heuristic, not on the actual u_target feasibility constraint**. The spec (§6.7) says: "increase alpha_t only where needed." The code checks `headroom_ratio = theta_safe_t / max(theta_safe_max, 1e-8) < 0.8` and bumps alpha by 1.3x. This does NOT check whether `u_target_t <= U_safe_ref_t` (which is the actual feasibility constraint from §6.6). A stage where `theta_safe_t` is already 80% of `theta_safe_max` but `U_safe_ref_t` is still below `u_target_t` would not get its alpha bumped. Conversely, a stage where `theta_safe_t / theta_safe_max < 0.8` but `u_target_t << U_safe_ref_t` (no constraint violation) would get an unnecessary alpha increase. The fixed-point therefore does not implement the spec's stated convergence condition, potentially increasing alpha unnecessarily (weakening certification) or failing to increase it where needed (leaving u_target unachievable).

- [MAJOR] `run_phase4_counterfactual_replay.py:84-167` — **Counterfactual replay re-runs the environment with a new RNG but uses the same seed as the pilot; non-idempotency if the MDP has internal randomness**. `_run_pilot_with_transitions` calls `seed_everything(seed)` and `np.random.default_rng(seed)` identically to `run_classical_pilot`. But `run_classical_pilot` was already called just before (line 82), and it also calls `seed_everything(seed)`. If the MDP `mdp_rl` retains any global random state (e.g., via `np.random.seed` or global `random` module), the second `seed_everything` resets that state. However the MushroomRL environments sometimes consume global numpy random state during initialization (`build_phase4_task` is called again at line 89), not just during `reset()`. If the environment init advances the global RNG, the transition sequences in the re-run may not match the pilot's sequences. The frozen transitions are then NOT the same transitions that generated `pilot_data`, which breaks the counterfactual isolation claim: the schedule was built from pilot_data (first run), but the replayed transitions come from a second run with potentially different trajectories.

---

### MINOR issues

- [MINOR] `task_activation_search.py:340` — `mean_abs_u_pred` uses `u_ref_used_t` (the clipped, deployed u), not `u_target_t`. This is correct for reporting the deployed activation, but the variable name `_pred` (predicted) may mislead: it is the predicted DEPLOYED u, not the unconstrained target. The distinction matters when trust or safe caps are binding. The spec §5.4 calls for `mean_abs_u_pred`, which should be the same as what is deployed under the schedule, so this is likely fine, but renaming to `mean_abs_u_deployed_pred` would prevent confusion.

- [MINOR] `phase4_calibration_v3.py:205-211` — When `q75 == 0.0` (no positive aligned margins), `xi_ref_arr[t]` is set to `xi_min`, but this branch is only triggered when `q75 == 0.0` AND the `clip` to `[xi_min, xi_max]` already takes care of it (since `clip(0.0, xi_min, xi_max) = xi_min`). The redundant explicit `xi_ref_arr[t] = xi_min` re-assignment on line 210 overwrites the correct `clip` result but produces the same value. This is harmless but dead code that could confuse a reader into thinking there is special handling for the zero-quantile case beyond the clip.

- [MINOR] `trust_region.py:64-67` — `_rho(u, gamma_base)` computes `eta0 = log(1/gamma_base) = -log(gamma_base)`. For `gamma_base = 0.97`, `eta0 = 0.03046`. For `u = 0`, `rho(0) = sigmoid(eta0) = 1/(1+exp(-eta0))`. Since `p0 = 1/(1+gamma_base)` and `sigmoid(log(1/gamma_base)) = 1/(1+gamma_base)` holds algebraically, this is correct. However, `eta0` is recomputed inside `_rho` on every call via `np.log(1.0/gamma_base)`, which is `log(1) - log(gamma_base) = -log(gamma_base)`. At `gamma_base` near 1, this is numerically fine. No issue, but caching `eta0` as a module-level constant or parameter would improve performance in the bisection loop.

- [MINOR] `run_phase4_counterfactual_replay.py:155` — Beyond the BLOCKER walrus issue, the expression `(i := step_idx) == horizon - 1` is syntactically valid Python 3.8+ but `i` is never referenced again in the scope. This produces a `SyntaxWarning` in some linters and is dead code. The intent is evidently `step_idx == horizon - 1`, and the walrus should be removed.

- [MINOR] `selected_tasks.json` — The `schedule_summary.sign` field is `null` for all selected tasks. The spec §6.11 requires `sign_family` to be stored. The `selected_tasks.json` is the output of the activation search, not the final schedule, so this may be expected (sign is selected inside `build_schedule_v3_from_pilot` called later). However, the JSON schema under `schedule_summary` does include a `sign` key whose null value could mislead downstream consumers expecting a populated sign to determine the operator direction without re-running the schedule builder.

- [MINOR] `global_diagnostics.json` — `generated_at: "2026-04-20T01:31:49Z"` is in the future relative to the code history. The most recent commit is dated based on the git log. This suggests either the system clock was incorrect when results were generated, or the results were pre-generated and the timestamp is synthetic. While this does not affect correctness, it should be noted in the paper supplemental as a reproducibility concern.

---

### DISPUTE (claim appears correct after investigation)

- [DISPUTE] Challenge 1 (natural-shift identity u = beta*(r-v) = theta*xi) — The concern is whether `A_t * xi_ref` correctly represents `(r_t - v_next)`. In the code, `theta_used = sign_family * u_ref_used / max(xi_ref, xi_floor)` and `beta_used = theta_used / max(A_t, 1e-8)`. So `u = beta * (r - v) = (theta / A_t) * (r - v)`. If we define `xi = (r-v)/A_t` (normalized margin), then `u = theta * xi`. The identity `u = beta*(r-v) = theta*xi` holds at the DESIGN level (where xi_ref is the reference normalized margin used to set theta). At the TRANSITION level, the actual xi = (r_actual - v_actual)/A_t may differ from xi_ref. So the identity holds exactly at the reference point and approximately at actual transitions. This is correctly documented in the spec §2.2 ("The safe operator's local behavior is governed by u, not by raw beta"). The approximation is not a bug — it IS the design philosophy of working in u-space. The concern does not hold.

- [DISPUTE] Challenge 2 (ex-ante purity of activation search) — `score_all_candidates` in `task_activation_search.py` imports only from: `experiments.weighted_lse_dp.geometry.phase4_calibration_v3` (build_schedule_v3_from_pilot, select_sign), `experiments.weighted_lse_dp.tasks.phase4_operator_suite` (build_phase4_task, get_search_grid), and `experiments.weighted_lse_dp.common.seeds` (seed_everything). None of these import from Phase III or Phase IV safe result files. The scoring pipeline is: run_classical_pilot -> build_schedule_v3_from_pilot -> compute_candidate_score. All inputs are derived from classical DP (V* via backward VI) and the calibration schedule built from pilot margins. There is no safe-operator data used for scoring. Purity is maintained.

- [DISPUTE] Challenge 3 (trust region reference distribution correctness) — `p0 = 1/(1+gamma_base)`. Since `rho(u) = sigmoid(eta0 + u)` with `eta0 = log(1/gamma_base)`, at `u=0`, `rho(0) = sigmoid(log(1/gamma_base)) = 1/(1+gamma_base) = p0`. So KL(rho(0) || p0) = 0, confirming p0 is the correct neutral reference. `eta0 = log(1/gamma_base)` is the log-odds shift from 1/2 (the sigmoid midpoint) to the prior p0. Monotonicity: for u >= 0, rho(u) = sigmoid(eta0+u) is strictly increasing (sigmoid is strictly increasing), moving away from p0 monotonically. KL(Bern(p) || Bern(p0)) for p > p0 is strictly increasing in p (standard result for KL of Bernoulli distributions on the same side of the mean). So KL(rho(u) || p0) is strictly increasing for u >= 0, as claimed. The bisection is sound.

- [DISPUTE] Challenge 4 (Phase III backward compatibility / Bhat recursion) — The old formula `kappa * (r_max + bhat) / (1 - kappa)` is the closed-form steady-state value of the geometric series, NOT the backward recursion. The backward recursion `Bhat[t] = (1+gamma)*r_max + kappa_t * Bhat[t+1]` is the correct finite-horizon formula from `safe_weighted_common.compute_certified_radii`. `adaptive_headroom.compute_bhat_backward` now delegates directly to `compute_certified_radii`, ensuring bit-for-bit identity with the operator layer. No other callers of the old formula pattern were found in the geometry package (confirmed by Grep). Phase III backward compatibility via delegation is sound.

- [DISPUTE] Challenge 5 (counterfactual replay isolation) — `_replay_task` calls `_run_pilot_with_transitions`, which computes V* via backward VI on the BASE MDP (not the safe operator's value function). The safe operator is never called anywhere in the transition collection or schedule building path. The schedule is built from classical pilot margins, and the replay computes `natural_shift = beta * (r - v_next)` where v_next comes from V* (classical DP). The safe operator's value function is not referenced. The counterfactual isolation is structurally sound, subject to the BLOCKER issue about the seed/trajectory-non-idempotency concern.

- [DISPUTE] Challenge 7 (systematic downward bias from negative margins) — When V*(s') > r for most transitions (which is common early in the trajectory when future value exceeds immediate reward), the raw margins `r - V*(s')` are negative. With `sign_family = +1`, `a_t = margin/r_max < 0`, and `q75(a_t | a_t > 0)` returns 0.0 (no positive samples), triggering the fallback to `xi_ref = xi_min = 0.02`. This is by design: the sign selection picks the sign that maximizes the alignment score, so if most margins are negative under `+1`, the algorithm would select `sign_family = -1`, flipping `a_t = -margin/r_max > 0` for most transitions. The `select_sign` function in `phase4_calibration_v3.py` does exactly this. The systematic bias concern does not hold because sign selection corrects for it.

---

### Summary table

| Issue | Severity | File | Line | Category |
|-------|----------|------|------|----------|
| Terminal v_next mismatch between pilot and replay | BLOCKER | `run_phase4_counterfactual_replay.py` | 155 | Counterfactual isolation |
| Gate metric denominator is all-T, not informative stages | BLOCKER | `task_activation_search.py` | 340 | Gate validity |
| U_safe_ref sign silently discarded via abs() | BLOCKER | `phase4_calibration_v3.py` | 271 | Safety certification |
| safe_clip_active logic inverted | MAJOR | `phase4_calibration_v3.py` | 290-292 | Diagnostic validity |
| Bisection fragile for sub-1e-10 eps_tr | MAJOR | `trust_region.py` | 156-163 | Trust region |
| Search acceptance thresholds weaker than gate | MAJOR | `task_activation_search.py` | 413-489 | Gate validity |
| Fixed-point uses theta-ratio heuristic, not feasibility | MAJOR | `adaptive_headroom.py` | 324-340 | Spec compliance |
| Re-run with same seed may produce different transitions | MAJOR | `run_phase4_counterfactual_replay.py` | 84-167 | Counterfactual isolation |
| mean_abs_u_pred naming confusing | MINOR | `task_activation_search.py` | 340 | Documentation |
| Redundant xi_ref xi_min fallback | MINOR | `phase4_calibration_v3.py` | 209 | Dead code |
| eta0 recomputed on each bisection call | MINOR | `trust_region.py` | 64 | Performance |
| Walrus operator dead binding | MINOR | `run_phase4_counterfactual_replay.py` | 155 | Code quality |
| schedule_summary.sign null in selected_tasks.json | MINOR | `results/.../selected_tasks.json` | — | Schema |
| Timestamp in future | MINOR | `global_diagnostics.json` | — | Reproducibility |
