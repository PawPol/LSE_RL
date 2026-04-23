# Phase IV-A — Audit, operator activation, natural-shift scheduling, and counterfactual replay

This document is for the coding agent. Treat it as the **first, self-standing Phase IV-A implementation spec** for the TAB experiment stack.

Phase IV-A is the activation-first stage. Its goal is not to prove final performance gains. Its goal is to verify that Safe TAB can be made **nontrivially nonlinear under certification** on a principled activation suite, while preserving full compatibility with Phase III.

Phase III established that:

1. the safe weighted-LSE / TAB operator and certification pipeline are implemented correctly;
2. the clip is load-bearing;
3. under the current task scales, horizons, and reward magnitudes, TAB is mostly **numerically classical**;
4. therefore the correct next step is to make the operator active under certification and measure that activation directly before running full RL comparisons.

The central local effect is controlled by the **natural-coordinate shift**

```text
u = beta * (r - v) = theta * xi
```

where

```text
A_t = R_max + Bhat_{t+1}
xi = (r - v) / A_t
theta = beta * A_t
u = theta * xi = beta * (r - v)
```

Phase IV-A therefore focuses on:

1. Phase III compatibility and audit;
2. Phase III observability fixes in the Phase IV code path;
3. operator-sensitive activation-suite construction using only classical pilot diagnostics and certification diagnostics;
4. natural-shift-first schedule calibration;
5. counterfactual target replay;
6. matched classical controls required for lower-base-`gamma` comparisons.

Do not proceed to Phase IV-B full RL translation experiments until the activation gate in Section 13 passes.

---

## 0. Non-negotiable workflow rules

1. Start with a written plan in `tasks/todo.md`.
2. Keep `tasks/todo.md` current and close each completed item with a short review note.
3. Add every correction, debugging lesson, and task-design surprise to `tasks/lessons.md`.
4. Do not overwrite Phase I/II/III outputs. Phase IV-A must be additive and backward-compatible.
5. Do not select or redesign tasks by looking at Phase IV safe-return improvements. Task selection must use **classical pilot diagnostics + certification diagnostics only**.
6. Any new heuristic must have: a config switch, explicit logging fields, and at least one ablation stub.
7. Any learned or state-dependent scheduler must be frozen during each Bellman-learning phase. In Phase IV-A, state-dependent schedulers are not mainline; reserve them for Phase IV-C unless explicitly needed for diagnostics.
8. When lower-base-`gamma` is used, always emit a **matched classical fixed-`gamma_base` control** and a **safe-zero-nonlinearity control**. No exceptions.
9. Preserve the original Phase III paper suite as a **negative-control replay suite**.
10. The main Phase IV-A claims must be based on activation diagnostics, not safe-return improvements.

---

## 1. Phase IV-A objectives

Phase IV-A is complete only if it can answer the activation question clearly:

> Can Safe TAB be made nontrivially nonlinear under certification on a principled benchmark suite?

This is not the same as asking whether `beta_used` is nonzero. The required quantities are:

- nontrivial `beta_used`,
- nontrivial `beta_used * margin`,
- nontrivial `effective_discount_used - gamma_base`,
- nontrivial `safe_target - classical_target_same_gamma_base`,
- all under certified clipped deployment.

Phase IV-A must produce:

1. a Phase III compatibility report;
2. an activation-search pipeline;
3. a frozen activation suite selected using ex ante diagnostics only;
4. schedule calibration v3 with natural-shift-first design;
5. counterfactual target replay results;
6. matched classical-control configs;
7. activation-frontier figures and tables.

---

## 2. Core geometry

### 2.1 Mainline safe operator remains unchanged

Mainline one-step safe TAB target at stage `t`:

```text
g_t_safe(r, v) =
  ((1 + gamma_base) / beta_used_t)
  * [ log(exp(beta_used_t * r) + gamma_base * exp(beta_used_t * v))
      - log(1 + gamma_base) ]
```

with the classical fallback

```text
r + gamma_base * v
```

when `beta_used_t = 0`.

Responsibility and effective discount:

```text
rho_t(r, v) = sigmoid(log(1 / gamma_base) + beta_used_t * (r - v))
d_t(r, v) = (1 + gamma_base) * (1 - rho_t(r, v))
```

### 2.2 Mainline effect variable is the natural shift `u`

Define the aligned normalized margin and natural shift:

```text
z = (r - v_ref) / A_t
a = sign_family * z
u = beta_used_t * (r - v) = theta_used_t * (r - v) / A_t
```

The safe operator’s local behavior is governed by `u`, not by raw `beta`.

### 2.3 Small-signal diagnostics

Near `u = 0`:

```text
d_t - gamma_base ≈ -(gamma_base / (1 + gamma_base)) * u
```

and

```text
g_t_safe(r, v) - (r + gamma_base v)
  ≈ (gamma_base / (2 * (1 + gamma_base))) * beta_used_t * (r - v)^2
```

These are diagnostics only. Do not use the approximations as implementation shortcuts.

### 2.4 Certification geometry lives in `theta`

Certification acts on

```text
theta = beta * A_t
```

not directly on `u`.

Phase IV-A design logic:

1. choose desired effect in `u` space;
2. convert to `theta` using `xi_ref`;
3. regularize / clip in geometry space;
4. deploy in `beta` space;
5. measure actual transition-level activation.

---

## 3. Mandatory Phase III audit

Do this before any new experiments.

### 3.1 Compatibility audit goals

Create a Phase IV audit layer that checks both code and artifacts.

Required checks:

1. Existing Phase III schedules load without manual editing.
2. Existing Phase III runs can be replayed through the Phase IV code path.
3. Phase IV safe target evaluation exactly matches Phase III when all new features are disabled.
4. Phase IV parsers can read old metrics/logs and write a compatibility report.
5. Stage extraction from time-augmented states is derived from environment metadata, not a hard-coded table.
6. `reward_bound` used for certification is verified to be a **one-step reward bound**, not a return-like quantity.

### 3.2 Required Phase III observability fixes to absorb into Phase IV-A

Even if Phase III files are preserved, the Phase IV-A code path must eliminate known weak points:

1. `safe_margin` for ExpectedSARSA / TD-style updates must come from the actual bootstrap used in the safe target, not greedy max-Q logging.
2. Per-task `schedule_file` overrides must be honored.
3. DP rho / responsibility aggregates must be real values, not all-NaN placeholders.
4. Stage decoding must not rely on hard-coded `n_base` tables.
5. Any Phase III config or adapter that lacks explicit `reward_bound` must be flagged in the compatibility report.

### 3.3 Audit artifacts

Write to:

```text
results/weighted_lse_dp/phase4/audit/
  phase3_code_audit.json
  phase3_result_audit.json
  phase3_compat_report.md
  phase3_replay_smoke/
```

### 3.4 Mandatory replay smoke checks

Replay at least one Phase III DP config and one Phase III RL config.

Required equalities up to numerical tolerance:

- same `beta_used_t`,
- same `rho_t`,
- same `effective_discount_t`,
- same one-step target values,
- same aggregate metrics on a fixed seed.

Stop if this fails.

---

## 4. Benchmark policy: negative controls plus activation suite

### 4.1 Negative-control replay suite

Keep the original Phase III tasks and rerun them through the Phase IV-A code path as controls.

Purpose:

- verify backward compatibility;
- show that when activation diagnostics stay near zero, performance differences should also be small;
- avoid overclaiming.

### 4.2 Activation suite

Create a new suite chosen **before** full safe runs using only:

- task semantics,
- classical pilot data,
- certification / schedule diagnostics,
- no Phase IV safe performance.

### 4.3 Activation suite design principles

Required design rules:

1. **Do not use huge one-step shocks** in the mainline activation suite.
   - Jackpot / catastrophe / hazard shocks should usually lie in `|reward| <= 3.0`.
   - Larger shocks are allowed only as negative-control or appendix variants.
2. **Do not use very long high-gamma horizons** in the mainline activation suite.
   - The combination `gamma_eval >= 0.99` and `H >= 80` should normally be treated as a negative control, not a mainline activation task.
3. **Use event probabilities that are rare but observable**.
   - Realized event rates around `1%` to `15%` are usually the right target.
4. **Keep the qualitative tradeoff while shrinking pathological scales**.
5. **Select tasks using activation diagnostics, not outcome wins**.

### 4.4 Horizon and nominal-discount redesign rule

For the activation suite, start from

```text
gamma_eval in {0.95, 0.97}
H in round(c / (1 - gamma_eval)) for c in {1.0, 1.5, 2.0}
```

Typical examples:

- `gamma_eval = 0.95` -> `H in {20, 30, 40}`
- `gamma_eval = 0.97` -> `H in {33, 50, 67}`

### 4.5 Family-specific redesign grids

Create additive task factories under a new Phase IV operator-sensitive suite. Do not mutate old Phase II/III tasks in place.

#### 4.5.1 Chain sparse-credit family

Search over at least:

- `state_n in {18, 24, 30}`
- `gamma_eval in {0.95, 0.97}`
- `horizon` from the rule above
- optional small step cost in `{0.0, -0.01, -0.02}`

Keep goal reward near `+1.0`.

#### 4.5.2 Chain jackpot family

Search over at least:

- `jackpot_reward in {1.5, 2.0, 3.0}`
- `jackpot_prob in {0.05, 0.10, 0.20}`
- `jackpot_state` chosen so the event is reachable with nontrivial frequency
- `gamma_eval in {0.95, 0.97}`
- `horizon` from the rule above
- background reliable goal reward near `+1.0`

#### 4.5.3 Chain catastrophe family

Search over at least:

- `catastrophe_reward in {-1.5, -2.0, -3.0}`
- `risky_prob in {0.05, 0.10, 0.20}`
- `shortcut_jump in {3, 4, 5}`
- `gamma_eval in {0.95, 0.97}`
- `horizon` from the rule above
- optional step cost in `{0.0, -0.01, -0.02}`

#### 4.5.4 Grid hazard family

Search over at least:

- `hazard_reward in {-1.0, -1.5, -2.0}`
- `hazard_prob in {0.10, 0.20, 0.30}`
- small detour lengths (`2` to `4` extra steps)
- `gamma_eval in {0.95, 0.97}`
- `horizon` from the rule above
- optional step cost in `{0.0, -0.01, -0.02}`

#### 4.5.5 Regime-shift family

Search over at least:

- `gamma_eval in {0.95, 0.97}`
- `horizon` from the rule above
- `change_at_episode` chosen so pre-change learning is meaningful but not complete
- goal flips / corridor reward flips / hazard flips with moderate one-step rewards
- optional small step cost

#### 4.5.6 Taxi bonus family

Search over at least:

- `bonus_reward in {1.5, 2.0, 3.0}`
- `bonus_prob in {0.05, 0.10, 0.15}`
- `gamma_eval in {0.95, 0.97}`
- `horizon in {40, 60}`

If taxi remains too noisy, keep it appendix-only and do not let it block chain/grid.

---

## 5. Operator-sensitive activation search

### 5.1 Required modules

Create:

```text
experiments/weighted_lse_dp/tasks/
  phase4_operator_suite.py

experiments/weighted_lse_dp/geometry/
  activation_metrics.py
  task_activation_search.py

experiments/weighted_lse_dp/configs/phase4/
  activation_search.json
  activation_suite.json
  paper_suite_replay.json

experiments/weighted_lse_dp/runners/
  run_phase4_activation_search.py
```

### 5.2 Selection protocol

For each candidate task variant:

1. run a short classical pilot;
2. build candidate safe schedules from pilot diagnostics;
3. compute predicted activation metrics under certification;
4. score the candidate;
5. freeze the selected tasks;
6. only then proceed to counterfactual replay and later full experiments.

The search runner must not read any Phase IV safe return files.

### 5.3 Minimum acceptance criteria

Let “informative stages” mean stages with aligned occupancy or aligned positive margin frequency at least `5%`.

Required thresholds:

```text
median_informative(abs(beta_cap_t) * max(1.0, reward_bound)) >= 5e-3
median_informative(U_safe_ref_t) >= 3e-3
mean_abs_u_pred >= 2e-3
frac(|u_pred| >= 5e-3) >= 0.10
mean_abs(d_pred - gamma_base) >= 1e-3
mean_abs(g_safe_pred - g_classical_same_gamma_base_pred) / max(1e-8, reward_bound) >= 5e-3
```

Preferred stronger activation target:

```text
mean_abs_u_pred >= 1e-2
frac(|u_pred| >= 1e-2) >= 0.10
mean_abs(d_pred - gamma_base) >= 5e-3
```

Task must also not be trivial:

- sparse tasks: baseline success/AUC not at ceiling or floor;
- tail-risk tasks: event rate in `[1%, 15%]`;
- regime-shift tasks: measurable adaptation lag;
- planning tasks: not solved instantly with no gradient.

### 5.4 Selection score

Use the following default score:

```text
score =
  w1 * standardized(mean_abs_u_pred)
+ w2 * standardized(mean_abs_delta_discount_pred)
+ w3 * standardized(mean_abs_target_gap_pred / reward_bound)
+ w4 * standardized(informative_stage_fraction)
- penalties_for_triviality
- penalties_for_runtime
- penalties_for_event_rate_out_of_band
```

Default weights:

```text
w1 = 1.0
w2 = 1.0
w3 = 1.0
w4 = 0.5
```

### 5.5 Required search artifacts

Write to:

```text
results/weighted_lse_dp/phase4/task_search/
  candidate_grid.json
  candidate_scores.csv
  selected_tasks.json
  activation_search_report.md
```

The report must explain why each mainline activation task was selected.

---

## 6. Schedule calibration v3: natural-shift-first, geometry-certified

### 6.1 Calibration source priority

For each task family, use the following order:

1. compatible Phase III safe pilot logs if they exist and pass audit;
2. otherwise Phase I/II classical logs plus a short Phase IV classical pilot;
3. otherwise Phase I/II classical logs only for stagewise fallback.

Log the source exactly.

### 6.2 Primary sign rule

Use a common sign per task family unless a state-dependent sign ablation is explicitly enabled.

For each sign candidate `s in {-1, +1}`:

```text
a_t^s = s * (r - v_ref) / A_t
p_align_t^s = P(a_t^s > 0)
xi_ref_t^s = Q_0.75(a_t^s | a_t^s > 0)
score_t^s = xi_ref_t^s * sqrt(max(p_align_t^s, 0))
```

Default: choose one family sign by maximizing the stage-averaged score.

Sign selection MUST use the provisional `A_t^{(0)}` defined in §6.3.1 below
(the same bootstrap denominator used for the first `xi_ref` pass), NOT the
raw `r_max`. Using `r_max` here is a known pitfall (MAJOR-7 in the Phase IV-A
review triage) and MUST be avoided.

### 6.3 Target natural shift

For the chosen sign, define:

```text
a_t = sign_family * (r - v_ref) / A_t
xi_ref_t = clip(Q_0.75(a_t | a_t > 0), xi_min, xi_max)
```

Defaults:

```text
xi_min = 0.02
xi_max = 1.00
```

Define informativeness:

```text
I_t = normalize(xi_ref_t * sqrt(P(a_t > 0)))
```

Target stagewise natural shift magnitude:

```text
u_target_t = u_min + (u_max - u_min) * I_t
```

Defaults:

```text
u_min = 0.002
u_max = 0.020
```

#### 6.3.1 Normative iteration order (SPEC-GAP resolution, 2026-04-22)

The quantities `A_t`, `alpha_t`, `I_t`, `xi_ref_t`, and `u_target_t` are
mutually dependent (`A_t = R_max + Bhat_{t+1}`, `Bhat_{t+1} = f(alpha_{>=t})`,
`alpha_t = g(I_t)`, `I_t = h(xi_ref_t, p_align_t)`, and
`xi_ref_t = Q_{0.75}(sign * margin / A_t | margin > 0)`). The spec §6.2 /
§6.3 equations are silent on iteration order; this subsection is normative.

Implementations MUST use the following **outer two-pass scheme with an
inner Gauss-Seidel fixed-point on `alpha`**:

1. **Outer pass 1 — bootstrap `xi_ref` using a provisional denominator.**
   A bootstrap scalar `D^{(0)}` is required to break the circular
   dependency. Two implementations are permitted, both of which produce
   a `xi_ref^{(0)}` that is in the same order-of-magnitude neighborhood
   of the true `A_t`:
   - (a) `D_t^{(0)} = R_max` (cheap; used by the current
     `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py`
     because `R_max <= A_t <= R_max * (T+1)` is a tight enough bound
     that pass 2's refinement suffices on all 146 configs regenerated
     in the 2026-04-22 sweep); or
   - (b) `D_t^{(0)} = A_t^{(0)} = R_max + Bhat_{t+1}^{(0)}` computed with
     `alpha_t^{(0)} = alpha_min` (mathematically tighter; required if
     `R_max`-bootstrap fails the `1e-6` convergence check in step 4).

   Compute the bootstrap
   `xi_ref_t^{(0)} = clip(Q_{0.75}(sign * margin / D_t^{(0)} | margin > 0),
   xi_min, xi_max)`. The same denominator MUST be used for sign
   selection (§6.2) and the bootstrap `xi_ref` computation to avoid an
   internal inconsistency between the selected sign and its justifying
   score.

2. **Inner Gauss-Seidel pass 1 — adaptive-headroom fixed point.**
   Run the §6.7 fixed-point loop with `xi_ref_t^{(0)}` held constant,
   iterating `alpha_t -> kappa_t -> Bhat -> A_t -> theta_safe_t ->
   U_safe_ref_t` until all stages satisfy
   `u_target_t <= U_safe_ref_t` or `max_fixed_point_iters` is hit.
   Let the converged outputs be `alpha_t^{(1)}, A_t^{(1)}, ...`.

3. **Outer pass 2 — refine `xi_ref` using the converged `A_t^{(1)}`.**
   Recompute
   `xi_ref_t^{(1)} = clip(Q_{0.75}(sign * margin / A_t^{(1)} | margin > 0),
   xi_min, xi_max)`.

4. **Convergence check.** If `xi_ref_t^{(1)} ≈ xi_ref_t^{(0)}` componentwise
   (absolute tolerance `1e-6` is the default), STOP and emit
   `alpha_t^{(1)}, A_t^{(1)}, xi_ref_t^{(1)}`.

5. **Inner Gauss-Seidel pass 2.** Otherwise rerun the §6.7 loop with
   `xi_ref_t^{(1)}` and produce the final schedule. At most ONE refinement
   cycle is performed; further cycles are NOT permitted (see convergence
   argument below).

**Convergence argument.** The inner §6.7 loop on `alpha` is monotone
non-decreasing in `alpha_t` (the feasibility trigger can only inflate
`alpha`, bounded above by `alpha_budget_max`) and therefore converges in
at most `max_fixed_point_iters` steps on the finite lattice induced by
the multiplicative bump. The outer map
`A_t -> xi_ref_t -> I_t -> alpha_t -> Bhat -> A_t` is not known to be a
strict Banach contraction in closed form, but is **operationally tight**
because:

- `xi_ref_t` depends on `A_t` only through division of a bounded margin
  by a denominator that changes by at most a factor of
  `(1 - gamma_base + alpha_budget_max * (1 - gamma_base))^{-1}` relative
  to the `alpha = 0` denominator, i.e. typically within a few percent;
- `Q_{0.75}` is Lipschitz in the denominator on the support of margins;
- the clip `[xi_min, xi_max]` further damps any remaining variation.

Empirically (`tests/algorithms/test_phase4_natural_shift_geometry.py` and
the calibration regeneration sweep of 2026-04-22) the two-pass scheme
converges with `xi_ref_t` changes below `1e-6` after the first refinement
on all 146 activation-suite configs. A second refinement would change the
schedule by less than the `xi_floor = 1e-3` safeguard downstream; more
iterations would therefore be operationally indistinguishable from two.
If a future task breaks this tight-bound regime (detected as
`xi_ref_t^{(1)}` changing by more than `1e-6` after an additional pass),
the implementation MUST log a warning and emit the last schedule rather
than iterating further; this preserves determinism.

**Rationale for Gauss-Seidel inside, two-pass outside.** Running the
§6.7 loop as a Jacobi update (recompute `xi_ref` simultaneously with
`alpha`) compounds two slow-moving dependencies and is measurably worse:
in early pilots the Jacobi variant required ~8 outer iterations versus
2 for Gauss-Seidel-inside + two-pass-outside, with identical final
schedules. Gauss-Seidel on `alpha` (the inner loop that actually governs
the certification geometry) is mathematically justified by the monotone
feasibility trigger; the outer `xi_ref` refinement is a one-shot
correction of the bootstrap denominator and does not need a second
fixed-point.

### 6.4 Optional target-discount-gap parameterization

Also support a config branch specifying

```text
delta_discount_target_t = delta_min + (delta_max - delta_min) * I_t
```

and converting to

```text
u_target_t ≈ ((1 + gamma_base) / gamma_base) * delta_discount_target_t
```

This is an alternative interface. The default mainline uses direct `u_target_t`.

### 6.5 Bernoulli KL trust-region cap

Let:

```text
eta0 = log(1 / gamma_base)
p0 = 1 / (1 + gamma_base)
rho(u) = sigmoid(eta0 + u)
```

Stagewise confidence:

```text
c_t = clip((n_t / (n_t + tau_n)) * sqrt(P(a_t > 0)), 0, 1)
```

Default:

```text
tau_n = 200
```

Design radius:

```text
eps_design_t = KL_Bern(rho(u_target_t) || p0)
eps_tr_t = c_t * eps_design_t
```

Solve for the largest nonnegative reference shift allowed by the trust ball:

```text
u_tr_cap_t >= 0 such that KL_Bern(rho(u_tr_cap_t) || p0) = eps_tr_t
```

### 6.6 Adaptive headroom

Baseline headroom:

```text
alpha_base_t = alpha_min + (alpha_max - alpha_min) * I_t
```

Defaults:

```text
alpha_min = 0.05
alpha_max = 0.20
```

Certification quantities:

```text
kappa_t = gamma_base + alpha_t * (1 - gamma_base)
Bhat_t via backward recursion
A_t = reward_bound + Bhat_{t+1}
Theta_safe_t = log(kappa_t / (gamma_base * (1 + gamma_base - kappa_t)))
U_safe_ref_t = Theta_safe_t * xi_ref_t
```

The requested effect is feasible without distortion if:

```text
u_target_t <= min(u_tr_cap_t, U_safe_ref_t)
```

### 6.7 Adaptive-headroom fixed-point loop

For each candidate `gamma_base`:

1. initialize `alpha_t = alpha_base_t`;
2. compute `Bhat_t`, `A_t`, and `xi_ref_t`;
3. compute `u_target_t`, trust caps, and `U_safe_ref_t`;
4. increase `alpha_t` only where needed, clipped by `alpha_budget_max`;
5. recompute for 2–4 iterations or until stable.

Defaults:

```text
alpha_budget_max = 0.30
max_fixed_point_iters = 4
```

### 6.8 Lower-base-`gamma` branch

Required candidate grid for activation-suite tasks:

```text
Gamma_base_grid = {gamma_eval, max(0.95, gamma_eval - 0.02), max(0.90, gamma_eval - 0.05)}
```

Initial restricted recommendation:

```text
gamma_eval = 0.97
gamma_base in {0.97, 0.95}
```

Expand only after the basic activation gate passes.

### 6.9 Matched controls for lower-base-`gamma`

For every safe run with `gamma_base != gamma_eval`, emit configs for:

1. classical matched-`gamma_base` control;
2. safe zero-nonlinearity control with `u_target = 0` / `theta_used = 0` under the same `gamma_base`.

### 6.10 Final deployed schedule

After trust regularization and adaptive headroom refinement:

```text
u_ref_used_t = min(u_target_t, u_tr_cap_t, U_safe_ref_t)
theta_used_t = sign_family * u_ref_used_t / max(xi_ref_t, xi_floor)
beta_used_t = theta_used_t / A_t
```

Default:

```text
xi_floor = 1e-3
```

Store separate clip flags:

- `trust_clip_active_t`
- `safe_clip_active_t`

Do not collapse them into one field.

### 6.11 Schedule file format v3

Required fields include at least:

```json
{
  "phase": "phase4",
  "schedule_version": 3,
  "task_family": "...",
  "scheduler_mode": "stagewise_u | lower_base_gamma",
  "gamma_eval": 0.97,
  "gamma_base": 0.95,
  "sign_family": 1,
  "reward_bound": 1.0,
  "alpha_t": [],
  "kappa_t": [],
  "Bhat_t": [],
  "A_t": [],
  "xi_ref_t": [],
  "u_target_t": [],
  "u_tr_cap_t": [],
  "U_safe_ref_t": [],
  "u_ref_used_t": [],
  "theta_used_t": [],
  "beta_used_t": [],
  "trust_clip_active_t": [],
  "safe_clip_active_t": [],
  "source_phase": "phase3 | phase12 | pilot",
  "notes": "..."
}
```

---

## 7. Counterfactual target replay

Counterfactual target replay is mandatory before full RL.

Add a runner that, on a frozen pilot transition set and frozen `v_next`, computes:

- classical target;
- safe target;
- target gap;
- effective-discount gap;
- natural shift;
- trust and safe cap utilization;
- event-conditioned diagnostics.

Create:

```text
experiments/weighted_lse_dp/runners/
  run_phase4_counterfactual_replay.py
```

Write to:

```text
results/weighted_lse_dp/phase4/counterfactual_replay/
```

Purpose:

- prove that TAB is active before the learning loop;
- isolate operator effects from estimator effects;
- prevent wasting compute on another near-classical suite.

---

## 8. Required diagnostics and logging

### 8.1 Per-transition / per-backup logging

Store at least:

- `stage`
- `gamma_eval`
- `gamma_base`
- `reward_bound`
- `A_t`
- `xi_ref_t`
- `u_target_t`
- `u_tr_cap_t`
- `U_safe_ref_t`
- `u_ref_used_t`
- `theta_used_t`
- `beta_used_t`
- `margin = reward - v_next`
- `margin_norm = margin / A_t`
- `natural_shift = beta_used_t * margin`
- `trust_clip_active`
- `safe_clip_active`
- `rho_used`
- `effective_discount_used`
- `delta_effective_discount = effective_discount_used - gamma_base`
- `safe_target`
- `classical_target_same_gamma_base = reward + gamma_base * v_next`
- `safe_target_gap_same_gamma_base = safe_target - classical_target_same_gamma_base`
- `classical_target_nominal_eval = reward + gamma_eval * v_next`
- `safe_target_gap_nominal_eval`
- `KL_to_prior = KL_Bern(rho_used || p0)`
- task-specific event flags (`jackpot_fired`, `catastrophe_fired`, `hazard_hit`, `post_change`, etc.)

### 8.2 Aggregate geometry diagnostics

Aggregate by stage and globally:

- mean/std/quantiles of `beta_used_t`;
- mean/std/quantiles of `natural_shift`;
- fraction with `|natural_shift| >= 5e-3`;
- mean/std/quantiles of `delta_effective_discount`;
- fraction with `|delta_effective_discount| >= 1e-3`;
- mean/std/quantiles of `safe_target_gap_same_gamma_base`;
- fraction with `|safe_target_gap_same_gamma_base| >= 5e-3 * reward_bound`;
- trust clip fraction;
- safe clip fraction;
- cap utilization ratio `u_ref_used / U_safe_ref`;
- aligned occupancy;
- mean KL to prior;
- effective-horizon proxy `1 / max(1e-8, 1 - mean_effective_discount_used)`.

### 8.3 Event-conditioned diagnostics

For jackpot / catastrophe / hazard / regime-shift tasks, aggregate the same diagnostics conditioned on event-relevant transitions:

- risky decision states;
- hazard-adjacent transitions;
- jackpot-state transitions;
- within `N` steps after a regime shift;
- top-decile aligned-margin transitions.

This is mandatory because average-over-all-transitions diagnostics can be diluted.

---

## 9. Required tests

### 9.1 Compatibility tests

Create:

```text
tests/algorithms/test_phase4_phase3_compat.py
```

Required:

1. Phase III schedule adapter reproduces the same `beta_used_t`.
2. Phase IV target eval matches Phase III when all new features are disabled.
3. Legacy result directories parse cleanly.
4. Stage decoding uses environment metadata and not a hard-coded `n_base` table.

### 9.2 Natural-shift geometry tests

Create:

```text
tests/algorithms/test_phase4_natural_shift_geometry.py
```

Required:

1. `u = beta * margin = theta * xi` identity holds numerically.
2. `delta_effective_discount` matches the exact derivative formula.
3. small-signal expansion of `delta_effective_discount` is accurate near zero.
4. small-signal expansion of `safe_target_gap` is accurate near zero.
5. trust and safe caps never increase `|u|`.

### 9.3 Activation-metric tests

Create:

```text
tests/algorithms/test_phase4_activation_metrics.py
```

Required:

1. `U_safe_ref_t = Theta_safe_t * xi_ref_t` is computed correctly.
2. event-conditioned aggregation is correct.
3. counterfactual replay metrics match direct recomputation.
4. **Primary gate MUST be evaluated on informative transitions only.**
   Activation thresholds that feed the §13.1 pass/fail decision
   (`mean_abs_u_informative`, `frac_informative(|u| >= 5e-3)`,
   `mean_abs(delta_effective_discount)_informative`,
   `mean_abs(target_gap)_informative / reward_bound`) MUST use the
   informative-transition mask defined in §13.1 (stage is informative iff
   `xi_ref_t * sqrt(p_align_t) >= 0.05`; transition is informative iff
   its stage is informative AND `margin_t > 0`). Global and
   event-conditioned denominators are secondary diagnostics (§13.2);
   tests MUST assert that the informative-masked numbers, not the
   globally-averaged ones, drive the gate.

### 9.4 Operator-sensitive task tests

Create:

```text
tests/environments/test_phase4_operator_sensitive_tasks.py
```

Required:

1. selected activation-suite task configs instantiate correctly.
2. realized event rates on short pilots are within intended bands.
3. severe variants preserve the intended semantics.
4. the original Phase III tasks are preserved and accessible as negative controls.

### 9.5 Selection-leakage test

Create:

```text
tests/algorithms/test_phase4_task_search_no_safe_leakage.py
```

Required:

- the activation-suite selection runner must not require Phase IV safe performance files; it should operate from classical pilot data plus closed-form diagnostics only.

### 9.6 Matched classical control tests

Create:

```text
tests/algorithms/test_phase4_gamma_matched_controls.py
```

Required:

1. whenever `gamma_base` differs from `gamma_eval`, the classical matched-gamma control is emitted.
2. safe-zero-nonlinearity control reproduces the classical target under the same `gamma_base`.
3. reported target gaps are always against the matched-`gamma_base` classical target.

### 9.7 End-to-end smoke tests

Create:

```text
tests/algorithms/test_phase4A_smoke_runs.py
```

Required:

1. audit runner completes.
2. activation-search runner completes.
3. counterfactual replay runner completes.
4. one short activation-suite DP target-evaluation replay completes and logs geometry fields.
5. aggregation and figure generation run on smoke outputs.

---

## 10. Phase IV-A experiment matrix

### 10.1 Suites

Run two suites:

1. negative-control replay suite: original Phase III suite through the Phase IV code path;
2. activation suite: newly selected operator-sensitive suite.

### 10.2 Scheduler variants for counterfactual replay

For each activation-suite family evaluate at least:

1. classical target, `gamma = gamma_eval`;
2. Phase III schedule replay through the Phase IV code path;
3. Phase IV natural-shift stagewise scheduler;
4. Phase IV natural-shift stagewise scheduler + adaptive headroom;
5. Phase IV natural-shift stagewise scheduler + lower-base-`gamma`;
6. safe-zero-nonlinearity control under the same `gamma_base`.

### 10.3 Diagnostic-strength sweep

For each activation-suite family, run a small sweep over reference effect strength:

```text
u_max in {0.0, 0.005, 0.010, 0.020}
```

or an equivalent `delta_discount_target` sweep.

Purpose:

- verify that operator diagnostics increase monotonically;
- choose configurations for Phase IV-B translation experiments.

---

## 11. Primary Phase IV-A metrics

### 11.1 Primary operator-activation metrics

These are first-class success criteria:

1. `beta_used` summary;
2. `beta_used * margin` summary;
3. `effective_discount_used - gamma_base` summary;
4. `safe_target - classical_target_same_gamma_base` summary;
5. reference cap utilization `u_ref_used / U_safe_ref`;
6. fraction of transitions with nontrivial `|natural_shift|`;
7. fraction of transitions with nontrivial `|safe_target_gap|`.

### 11.2 Activation-search metrics

Report:

- task candidate score;
- event rate;
- baseline classical AUC / success / adaptation lag;
- predicted mean absolute natural shift;
- predicted mean absolute discount gap;
- predicted mean absolute target gap;
- informative-stage fraction;
- rejection reason for rejected candidates.

---

## 12. Figures and tables

### 12.1 Mandatory Phase IV-A figures

1. **Activation frontier**: for each task family, `U_safe_ref`, `u_target`, `u_ref_used`, trust cap, and safe cap by stage.
2. **Natural-shift distribution**: histogram / density of `beta_used * margin`.
3. **Effective-discount separation**: distribution of `d_t - gamma_base`.
4. **Safe-vs-classical target separation**: distribution of `g_safe - g_classical_same_gamma_base`.
5. **Task-search frontier and rejected candidates**.
6. **Negative-control replay diagnostics**.

### 12.2 Mandatory Phase IV-A tables

1. `P4A-A`: activation-suite task definitions and pilot activation diagnostics.
2. `P4A-B`: operator-activation diagnostics by candidate and selected task.
3. `P4A-C`: matched classical-control configuration summary.
4. `P4A-D`: negative-control replay summary.
5. `P4A-E`: counterfactual replay summary.

---

## 13. Activation gate before Phase IV-B

Do not run full Phase IV-B RL translation experiments until counterfactual replay verifies certified nonclassical activation on the frozen activation suite.

### 13.1 Primary formal gate (informative-stage denominator)

The **primary, pass/fail gate** for Phase IV-A MUST be evaluated using the
**informative-stage denominator** defined in §5.3 and §9.3:

> a stage `t` is "informative" iff
> `xi_ref_t * sqrt(p_align_t) >= 0.05`
> (aligned occupancy or aligned positive-margin frequency at least `5%`);
> a transition is informative iff its stage is informative AND its aligned
> margin is positive (`margin_t > 0`).

All gate thresholds below are evaluated on the **informative transition
subset** of the counterfactual-replay output (§7):

Required gate (MUST pass):

```text
mean_abs_u_informative  >= 5e-3
frac_informative(|u| >= 5e-3) >= 0.10
mean_abs(delta_effective_discount)_informative >= 1e-3
mean_abs(target_gap)_informative / reward_bound >= 5e-3
```

Preferred stronger gate (SHOULD pass on at least one mainline family):

```text
mean_abs_u_informative  >= 1e-2
frac_informative(|u| >= 1e-2) >= 0.10
mean_abs(delta_effective_discount)_informative >= 5e-3
```

A family satisfying the Required gate is **IV-B eligible**. A family
failing the Required gate is kept as a **low-activation negative control**
and MUST NOT be used for the main Phase IV-B translation claim.

### 13.2 Secondary diagnostics (global and event-conditioned denominators)

Tables and reports MUST also surface — as **secondary diagnostics**, NOT
as gate conditions — the same four metrics evaluated with:

1. **Global denominator**: average over all replay transitions. Logged as
   `mean_abs_u_replay_global`, etc. This is the raw replay mean; it is
   dilution-dominated on activation-suite tasks (most stages are
   non-informative by construction) and SHOULD be labeled as a dilution
   diagnostic in every table and figure.
2. **Event-conditioned denominators**: averages over
   jackpot / catastrophe / hazard / regime-shift / top-decile-aligned-margin
   subsets, as defined in §8.3. These are also diagnostic; a family whose
   informative-stage gate fails but whose event-conditioned diagnostic
   passes MAY be reported as an **event-conditioned activation case** in
   Phase IV-B only if the report clearly labels it as such and the
   translation claim is restricted to those event subsets.

### 13.3 Rationale (certification-driven denominator choice)

The informative-stage denominator is the primary gate because the
**contraction certificate** `|d_t(r, v)| <= kappa_t` (§2.4, §6.6) is a
**stage-local pointwise** guarantee inside the certification box `Bhat_t`.
The certified activation quantity `U_safe_ref_t = Theta_safe_t * xi_ref_t`
(§6.6) is also stage-local and only meaningful where `xi_ref_t` reflects
actual aligned margin mass (i.e., on informative stages). Averaging
`|u|` over non-informative stages — where `xi_ref_t` collapses to
`xi_min` by fallback and the underlying transition density has no
aligned mass — dilutes the operator-activation signal with transitions
that the certification geometry never intended to activate. The
informative-stage gate therefore asks the mathematically correct
question: "does the operator activate where the contraction certificate
permits it to?"

The global and event-conditioned denominators remain useful as
**operational diagnostics** (are the informative stages a negligible
fraction of the task? are the events where activation matters actually
being hit?) and MUST be surfaced in tables P4A-B/E and in the
per-family eligibility report, but they do not gate Phase IV-B.

### 13.4 Cross-reference

§9.3 requirement 4 ("activation thresholds are evaluated only on
informative stages") is the **authoritative test contract** for the
primary gate in §13.1. Earlier versions of this spec (pre-2026-04-22)
implied a global denominator via the bare phrasing "globally and on
event-conditioned subsets"; that phrasing is superseded by §13.2 above,
which redesignates both global and event-conditioned views as
**secondary diagnostics**. Any implementation, test, or gate-check
script that still treats the global denominator as the pass/fail metric
(including `scripts/overnight/check_gate.py` callers, legacy
`aggregate_phase4A._evaluate_gate`, and any table that reports
`mean_abs_u` without an `_informative` / `_global` suffix) MUST be
updated to consume the informative denominator as primary.

---

## 14. Implementation artifacts to add

### 14.1 Geometry package

```text
experiments/weighted_lse_dp/geometry/
  natural_shift.py
  activation_metrics.py
  task_activation_search.py
  trust_region.py
  adaptive_headroom.py
  phase3_audit.py
  phase4_calibration_v3.py
  schedule_v3_schema.md
```

### 14.2 Task package

```text
experiments/weighted_lse_dp/tasks/
  phase4_operator_suite.py
```

### 14.3 Configs

```text
experiments/weighted_lse_dp/configs/phase4/
  paper_suite_replay.json
  activation_search.json
  activation_suite.json
  gamma_matched_controls.json
```

### 14.4 Runners

```text
experiments/weighted_lse_dp/runners/
  run_phase4_activation_search.py
  run_phase4_counterfactual_replay.py
  aggregate_phase4A.py
```

### 14.5 Analysis

```text
experiments/weighted_lse_dp/analysis/
  make_phase4A_tables.py
  make_phase4A_figures.py
```

---

## 15. Phase IV-A exit criteria

Phase IV-A is complete only if all of the following are true.

1. Phase III compatibility audit passes.
2. Known Phase III observability issues are fixed in the Phase IV code path.
3. The negative-control replay suite runs successfully.
4. The activation-search pipeline runs and freezes a main activation suite using ex ante criteria only.
5. Schedule calibration v3 with `u_target`, trust caps, adaptive headroom, and lower-base-`gamma` is implemented and tested.
6. Matched classical-control configs are emitted for every lower-base-`gamma` comparison.
7. Counterfactual target replay is implemented and logged.
8. Main activation metrics (`beta`, `beta*margin`, discount gap, target gap) are first-class outputs.
9. Activation-gate report is generated.
10. Figures and tables are generated.
11. `tasks/todo.md` and `tasks/lessons.md` are updated.

At the end of Phase IV-A, answer these questions for each task family:

1. Is this family a low-activation negative control or a true activated Safe TAB case?
2. How large are the deployed `beta_used`, `beta_used * margin`, discount-gap, and target-gap diagnostics under certification?
3. Did lower-base-`gamma` increase certified activation, and are matched controls ready?
4. Is the task eligible for Phase IV-B translation experiments?
5. If not, which mechanism failed: task design, schedule, certification geometry, or insufficient event-conditioned activation?
