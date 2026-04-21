# Phase IV-C — Standard code review

Scope: advanced estimator-stabilized agents, state-dependent scheduler
ablations, geometry-priority asynchronous DP, certification ablations,
aggregation, and the `tests/algorithms/test_phase4C_*.py` suite, against
`docs/specs/phase_IV_C_advanced_stabilization_and_geometry_ablations.md`.

This file is the "regular" review (correctness / schema / coverage).
The companion `adversarial.md` separately challenges mechanism-isolation
claims.

---

## Fix status (post-review)

| Finding | Status | Fix |
| ------- | ------ | --- |
| R1 — n_base wrong in certification ablations | **FIXED** | `mdp_rl.info.observation_space.n // horizon`; re-ran 42/42 |
| R2 — state-dependent schedulers not implemented | **PARTIALLY FIXED** | `state_bins.py`, `schedule_smoothing.py` implemented (§4.2–§4.3); runner still uses stagewise builder — documented with `scheduler_mode_note` in run.json |
| R3 — geometry priority formula wrong | **FIXED** | `abs(rho*gamma - gamma)`, KL uses `q=1/(1+gamma)`; re-ran 18/18 |
| R4 — no per-backup logging | **FIXED** | `plan(log_backups=True)` emits `backup_log.npz` with sweep/rank/stage/state/residual/priority/geom_gain/kl |
| R5 — ablation tests registry-only | **FIXED** | Added `test_wrong_sign_ablation_negates_beta`, `test_constant_u_transformation_produces_flat_u`, headroom ordering tests |
| R6 — constant_u silent clipping | **FIXED** | `constant_u_clip_count` computed and warned; logged in v3 notes |

---

## Summary

Implementation substantially covers the spec surface but has several
correctness issues that would reduce confidence in any headline claim.
The three `src/lse_rl/algorithms/safe_*.py` modules are sound and
well-tested. The runners and planner diverge from the spec in
non-trivial ways: the `state_dependent_scheduler` family is
mis-labeled (it is a stagewise ablation, not a state-bin scheduler);
`run_phase4C_certification_ablations.py` derives `n_base` from a
formula that is wrong for every currently-selected activation task; the
geometry priority planner implements a different priority formula from
the spec; three of the required geometry modules are `NotImplementedError`
stubs; and the ablation / statebin tests are registry-only and do not
assert any mechanism behavior.

**All 6 standard-review BLOCKERs have been resolved** (see fix table above).

| Tag      | Count |
| -------- | ----- |
| BLOCKER  | 6 → 0 |
| MAJOR    | 6     |
| MINOR    | 4     |
| NIT      | 3     |
| DISPUTE  | 0     |

---

## Findings

### [BLOCKER] R1 — `n_base` in `run_phase4C_certification_ablations.py` is wrong

File: `experiments/weighted_lse_dp/runners/run_phase4C_certification_ablations.py:259`

```python
n_base = horizon + 1
```

Every other Phase IV runner uses
`n_base = mdp_rl.info.observation_space.n // horizon`
(see `run_phase4_rl.py:405`, `run_phase4C_advanced_rl.py:305`,
`run_phase4C_scheduler_ablations.py:208`). `DiscreteTimeAugmentedEnv`
sets `observation_space = Discrete(horizon * n_base_states)`
(`mushroom-rl-dev/mushroom_rl/environments/time_augmented_env.py:117`),
so the correct formula is `obs_space.n // horizon`. `SafeQLearning`
decodes the stage via `t = aug // n_base` in
`SafeWeightedCommon.stage_from_augmented_state`. A wrong `n_base`
silently corrupts stage decoding.

For the current frozen suite (`activation_suite_4a2.json`), every task
is `dense_chain_cost` with `n_states = horizon = 20`, so the true
`n_base` is 20, not 21. Every published certification-ablation number
is therefore the output of an agent that was told stages are 21-wide
when they are actually 20-wide: target-stage pairs are shifted by a
periodic amount that grows with wall step. Must fix before results are
cited.

Acceptance criterion: replace line 259 with the observation-space
derivation, rerun, and diff the updated `metrics.json` against the old
ones.

### [BLOCKER] R2 — State-dependent schedulers are not implemented

Files:
`experiments/weighted_lse_dp/geometry/state_bins.py`,
`experiments/weighted_lse_dp/geometry/schedule_smoothing.py`,
`experiments/weighted_lse_dp/geometry/geometry_priority.py`.

All three modules are `NotImplementedError` stubs (`state_bins.py:30`,
`schedule_smoothing.py:28`, `geometry_priority.py:33`). Spec §4 and
§13.1 explicitly require:

> `experiments/weighted_lse_dp/geometry/schedule_smoothing.py`
> `experiments/weighted_lse_dp/geometry/state_bins.py`
> `experiments/weighted_lse_dp/geometry/geometry_priority.py`

and §4.1–§4.4 describe binwise `xi_ref_{t,b}`, `u_target_{t,b}`,
hierarchical backoff, bin construction modes, trust/safe caps after
backoff, and the v3 schedule extension `scheduler_mode: "statebin_u"`.
None of this exists.

Downstream, `run_phase4C_scheduler_ablations.py` names its four
variants `stagewise_baseline`, `state_bin_uniform`,
`state_bin_hazard_proximity`, `state_bin_reward_region`, but every
variant calls `build_schedule_v3_from_pilot` with different kwargs
for the **same** stagewise calibrator. There is no binning, no backoff,
no `xi_ref_tb`, no `bin_counts`. The runner's "state-dependent
scheduler comparison" is a stagewise (alpha_min/alpha_max/tau_n)
sweep relabelled to look state-dependent.

Acceptance criteria:

1. Implement `state_bins.construct_bins` for at least the
   `exact_state` and `margin_quantile` modes (spec §4.3).
2. Implement `schedule_smoothing.smooth_schedule` with the
   hierarchical-shrinkage formula in spec §4.2.
3. Emit the v3 extension fields listed in §4.4 when
   `scheduler_mode == "statebin_u"`.
4. Either make `run_phase4C_scheduler_ablations.py` consume the
   state-dependent scheduler, or rename it to
   `run_phase4C_stagewise_quality_ablations.py` and remove the
   misleading variant names.

### [BLOCKER] R3 — Geometry-priority score formula diverges from spec §5.1

File: `experiments/weighted_lse_dp/planners/geometry_priority_dp.py:78-97`

Spec §5.1:

```
geom_gain  = |effective_discount_used - gamma_base|
priority  = |residual| * (1 + lambda_geom * geom_gain
                             + lambda_u    * |u_used_ref_stage|
                             + lambda_kl   * KL_Bern(rho_used || p0))
```

where `p0 = 1/(1+gamma)` is the natural weighted-LSE prior.

Implementation:

```python
self._geom_gain_t[t] = self._gamma * abs(rho - 1.0/(1.0+gamma))  # wrong scale
self._u_ref_t[t]     = abs(beta * xi)                            # synthesized
self._kl_t[t]        = KL_Bern(rho, q=0.5)                       # wrong prior
```

Three separate divergences:

1. `geom_gain` should be `|(1+γ)(1-ρ) - γ| = |1 - (1+γ)ρ|`, but the
   code computes `γ · |ρ - 1/(1+γ)| = γ/(1+γ) · |1 - (1+γ)ρ|` — off by
   constant factor `γ/(1+γ)`. Affects reported scale and cross-task
   comparability, not ranking within a stage.
2. `KL_Bern(ρ || 0.5)` has no spec basis. Spec says
   `KL_Bern(ρ || 1/(1+γ))`, i.e., KL to the natural prior. At γ=0.95,
   `1/(1+γ) ≈ 0.513`, so for most tasks the numeric difference is
   small, but the *shape* (KL vanishes at ρ=1/(1+γ), not at ρ=0.5) is
   a systematic bias near the classical collapse.
3. `u_ref` is synthesized from a representative `xi` (`xi_ref_t`),
   not read from the schedule. Spec wants `|u_used_ref_stage|` which
   *is* a schedule field. Independent of the formula, this means the
   priority is never truly per-state — it is stagewise, then
   multiplied by `|residual[s,t]|`.

Acceptance criteria:

1. Add `geom_gain_t = abs((1+γ)(1 - rho_t) - γ)`.
2. Use `KL_Bern(rho_t, q=1/(1+γ))`.
3. Read `u_ref_t` from the v3 schedule if present, else compute from
   `beta_used_t * xi_ref_t` (current fallback) and flag in
   `run.json`.
4. Update `test_phase4C_geometry_priority_dp.py::test_priority_scoring_formula`
   to exercise all three corrections.

### [BLOCKER] R4 — Geometry-priority DP does not log per-backup fields required by spec §8.4

File: `experiments/weighted_lse_dp/planners/geometry_priority_dp.py:140-214`

Spec §8.4 requires:

- `residual`
- `geom_gain`
- `priority_score`
- `KL_Bern_to_prior`
- `backup_rank`
- `backup_stage`
- `backup_state`
- `backup_action`

Implementation writes `residual_history` (per-sweep scalars),
`geom_gain_per_stage`, `u_ref_per_stage`, and an aggregate
`frac_high_activation_backups`. None of the per-backup fields
(`backup_rank`, `backup_stage`, `backup_state`, `backup_action`,
per-backup `priority_score`) are logged. This blocks the downstream
"priority-distribution / fraction-of-high-activation-states" figure
in spec §12.1 #5.

Acceptance criterion: accumulate a list of
`{sweep, rank, stage, state, action, residual, priority, geom_gain,
kl}` records during `plan()` and dump to `backup_log.npz` in the run
directory.

### [BLOCKER] R5 — Ablation tests are registry-only

File: `tests/algorithms/test_phase4C_ablations.py`, and
`tests/algorithms/test_phase4C_statebin_scheduler.py`.

Spec §9.5 requires:

> 1. wrong-sign schedules flip sign as intended;
> 2. constant-u schedule is generated correctly;
> 3. raw-unclipped schedule bypasses caps only when explicitly configured;
> 4. trust-region ablation changes trust clip activity;
> 5. adaptive-headroom ablation changes safe cap utilization.

Every test in `test_phase4C_ablations.py` asserts only that the
override dict has a flag set (`_flip_sign is True`, `_constant_u is
True`, `tau_n < 1e-5`, …). None invoke the schedule builder, compare
resulting `beta_used_t`, compare trust-clip-activity, or check
cap-utilization vs baseline. Similarly,
`test_phase4C_statebin_scheduler.py` only asserts registry membership
and that `stagewise_baseline` has empty overrides.

Consequence: a regression that broke `_flip_sign` in the runner would
pass the tests. These files do not discharge the §9.5 spec item.

Acceptance criteria (per test function):

1. `test_wrong_sign_ablation`: build a schedule with
   `_flip_sign=True` via `_build_ablated_schedule` and assert that the
   resulting `beta_used_t` is the negative of the default schedule's
   `beta_used_t` element-wise (up to cap re-application).
2. `test_constant_u_ablation`: build the schedule and assert that
   `|beta_used_t * xi_ref_t|` is constant across stages up to
   floating-point tolerance, AND that this constant equals
   `mean(|default_beta * default_xi|)`.
3. `test_trust_region_*`: compare the number of clipped stages
   (`clip_active_t` true) between the default and tighter-tau_n
   schedules; assert strict ordering.
4. `test_adaptive_headroom_*`: compare `alpha_t` arrays; assert
   shape-level ordering (alpha_max_aggressive > alpha_max_default >
   alpha_max_off).

### [BLOCKER] R6 — `constant_u` ablation silently clips

File: `experiments/weighted_lse_dp/runners/run_phase4C_certification_ablations.py:150-160`

The constant-u rewrite
`v3["beta_used_t"] = (u_mean / xi_safe).tolist()`
is computed without checking that `|u_mean / xi_safe| <= beta_cap_t`.
`_wrap_v3_schedule_for_betaschedule` then silently clips any
exceedance to `beta_cap_t` (line 180 of `run_phase4_rl.py`).

If any stage's `xi_ref_t` is small enough that
`u_mean / xi_safe > beta_cap_t`, the resulting schedule is neither the
original nor actually constant-u — the "constant-u" label is false. No
warning or log field captures this.

Acceptance criterion: add a pre-wrap assertion / warning that records
`constant_u_clip_count` into the v3 notes field, and add a regression
test that triggers the case (small xi + large beta).

### [MAJOR] R7 — `run_phase4C_certification_ablations.py` does not use `resolved_cfg` from `build_phase4_task`

File: `experiments/weighted_lse_dp/runners/run_phase4C_certification_ablations.py:255-258`

```python
_, mdp_rl, _ = build_phase4_task(cfg, seed=seed)
...
gamma = float(cfg.get("gamma", 0.95))
horizon = int(cfg.get("horizon", 20))
```

The other IV-C runners use `resolved_cfg` (the third return value of
`build_phase4_task`), which contains the canonical gamma/horizon after
family-specific resolution. Using the raw `cfg` risks drift if the
family factory clamps or augments either field. At minimum,
consistency alone argues for fixing this.

### [MAJOR] R8 — Scheduler-ablation runner rewrites `u_min`/`u_max`, not `alpha` or `tau_bin`

File: `experiments/weighted_lse_dp/runners/run_phase4C_scheduler_ablations.py:79-84`

The registered "scheduler types" mutate
`alpha_min`, `alpha_max`, `alpha_budget_max`, `tau_n`. These are
headroom parameters, not bin parameters. Even if R2 is resolved and a
binning mode exists, none of the current scheduler variants would
select a different binning; they all still produce stagewise v3
schedules. As a result, the claim "state-dependent scheduler
comparison" in §12.1 #4 and Table `P4C-C` cannot be discharged by
these runs.

### [MAJOR] R9 — `aggregate_phase4C.py` attribution assumes baseline == `stagewise_baseline` scheduler

File: `experiments/weighted_lse_dp/runners/aggregate_phase4C.py:214-218`

```python
baseline_return = None
for r in scheduler:
    if r.get("scheduler") == "stagewise_baseline":
        baseline_return = r.get("mean_final_return")
        break
```

The "baseline" for advanced-RL gain and ablation deltas is the
scheduler-ablation baseline, which uses **MushroomRL's** `SafeQLearning`,
not the `src/lse_rl/algorithms/*.py` variants. So when the attribution
file reports
`safe_double_q.final_return − stagewise_baseline.final_return`, the
delta mixes two changes at once: (1) online/target table machinery
vs. single table; (2) any per-step code-path differences between
`SafeQLearning` (framework agent with callbacks, full Core loop) and
the handwritten `_run_episode_train` in `run_phase4C_advanced_rl.py`.
The "advanced_rl_gains_vs_baseline" number in
`attribution_analysis.json` is not a clean mechanism delta.

Acceptance criterion: add a `safe_q_learning_handwritten` control
(identical episode loop as `_run_episode_train` but using a single Q
table) and use *that* as the baseline_return for advanced-RL deltas.
Document the shift in the summary JSON.

### [MAJOR] R10 — `aggregate_phase4C.py` averages diagnostics across tasks

File: `experiments/weighted_lse_dp/runners/aggregate_phase4C.py:109-118`, 141-148.

`_aggregate_algo_group` pools runs from all `task_tag` directories
into one record and reports `mean_final_return`, `mean_double_gap`,
`mean_q_target_gap`. If the suite has more than one task family
(currently two dense_chain variants; will expand), pooling returns
weights each task equally regardless of absolute reward magnitude or
number of seeds. The spec's tables (§12.2) are per-task-family. At
minimum the aggregator should emit a per-task breakdown in addition
to the pooled number; at present only the pool is written.

### [MAJOR] R11 — `residual_history` semantics are pre-update

File: `experiments/weighted_lse_dp/planners/geometry_priority_dp.py:149-183`

Each sweep computes `all_residuals` for every `(t,s)`, then applies
only the top-k updates. `residual_history.append(max_residual)` stores
the pre-update max. With `top_k_fraction = 0.25` the top-k may miss
high-residual states in a given sweep, so `max_residual` is not
guaranteed to decrease monotonically across sweeps. The smoke test
(`test_combined_mode_ordering`) asserts only
`hist[-1] <= hist[0] + 1e-6`, which permits arbitrary oscillation in
between. Either document the non-monotone behavior or adopt a proper
priority sweep (Dijkstra-like queue) with full residual updates
after every backup.

### [MAJOR] R12 — Geometry DP backup count does not include action dimension

File: `experiments/weighted_lse_dp/planners/geometry_priority_dp.py:170-181`

Spec §5.3 requires "number of state-stage-action backups". The
implementation increments `n_backups` per `(s, t)` pair but each such
update computes `Q[s, a]` for all A actions inside `_safe_q_batch`.
The metric is therefore 1/A times the true backup count, which
distorts "overhead vs classical and residual-priority" comparisons
(§5.3 last bullet).

### [MINOR] R13 — `_greedy_action` / `_epsilon_greedy_action` use global numpy RNG

File: `experiments/weighted_lse_dp/runners/run_phase4C_advanced_rl.py:157`

```python
return int(np.random.choice(ties))
```

The function mixes seeded `rng.integers` (epsilon branch) with
unseeded `np.random.choice` (tie-break). Reproducibility is therefore
partial — two identical seeds can yield different greedy actions when
Q-value ties occur, which happens frequently in the chain families at
initialization.

### [MINOR] R14 — `SafeTargetQLearning._last_sync_step = 0` before the first sync conflates "no sync" with "sync at step 0"

File: `src/lse_rl/algorithms/safe_target_q.py:133, 207-208`

`target_sync_step` is logged as 0 until the first hard sync. With
`sync_every >= 1`, the first sync fires at `global_step == sync_every`,
so `0` means "no sync yet" — but a downstream reader cannot tell
that from a "synced at step 0" (which never happens by construction,
but the code's intent is not obvious). Suggest initialising to `-1`
or adding an explicit `has_synced: bool` log field.

### [MINOR] R15 — `SafeDoubleQLearning._rng` is shared between coin and tie-break

File: `src/lse_rl/algorithms/safe_double_q.py:123, 211, 239`

Both the selector-coin flip and the argmax tie-break draw from the
same `np.random.default_rng` stream. The `beta=0` classical-collapse
test threads a shared `ref_rng` in the same order — that's fine for
the test — but if the tie-break set size changes at runtime (e.g., a
value update removes a tie), the coin-flip stream drifts vs. a
reference implementation. Minor research-reproducibility concern.

### [MINOR] R16 — `SafeTargetQLearning._maybe_sync` fires AFTER the online update

File: `src/lse_rl/algorithms/safe_target_q.py:245-252`

```python
self._Q_online[t, s, a] = q_current + self._lr * (safe_target - q_current)
self._maybe_sync(gs)
```

With `sync_every = K`, at `gs = K` the target is synced AFTER the
k-th online update, meaning the k-th target bootstrap used the
pre-sync target (correct) but the k+1-th online update sees a target
that includes the k-th online update's change (usually intentional,
but worth stating in the docstring). Not a bug. Adding a docstring
line would prevent future confusion.

### [NIT] R17 — Hardcoded seeds/train_steps/lr constants at module scope

Files:
`run_phase4C_advanced_rl.py:75-83`,
`run_phase4C_scheduler_ablations.py:86-92`,
`run_phase4C_certification_ablations.py:104-109`.

`_SEEDS = [42, 123, 456]`, `_TRAIN_STEPS = 20000`, `_LR = 0.1`,
`_SYNC_EVERY = 200`, `_POLYAK_TAU = 0.05` are all at module scope.
The module's own config JSON (`advanced_estimators.json`,
`state_dependent_schedulers.json`, …) is loaded but never consulted
for these fields. Every CLI invocation silently uses the module
defaults. Moving them into `config.get(..., default)` would make the
configs authoritative.

### [NIT] R18 — Redundant `sys.path` bootstrap repeated in every file

Each of the 4 runners, 3 algorithms, 1 planner, and 6 test files
inserts `_REPO_ROOT`, `_MUSHROOM_DEV`, and `_SRC` into `sys.path`
with nearly identical code. Factor into a single
`experiments/weighted_lse_dp/_bootstrap.py` import or use the
installed package layout.

### [NIT] R19 — `convergence_sweep_1e-2` key contains a hyphen

Makes JSON-to-python attribute access awkward (cannot use
`obj.convergence_sweep_1e-2`). Prefer
`convergence_sweep_1em2` or `convergence_sweep_tol1e_2`.

---

## Schema compliance

| Spec field                                 | Logged? | Location                                              |
| ------------------------------------------ | ------- | ----------------------------------------------------- |
| `q_a_next, q_b_next`                       | yes     | `SafeDoubleQLearning.update` log                      |
| `selected_action_source, evaluation_value_source` | yes | same                                                  |
| `double_gap, margin_double, natural_shift_double` | yes | same                                                  |
| `q_online_next, q_target_next`             | yes     | `SafeTargetQLearning._build_log`                      |
| `q_target_gap, target_sync_step`           | yes     | same                                                  |
| `target_update_mode`                       | yes     | same                                                  |
| `target_polyak_tau` (spec §8.2)            | **no**  | not logged — only `target_update_mode` encodes it     |
| `state_bin_id, next_state_bin_id, bin_count, u_stage_ref, u_bin_design, u_bin_used, backoff_weight, statebin_trust_clip_active, statebin_safe_clip_active` | **no** | scheduler ablation is stagewise; no binning logs |
| `residual, geom_gain, priority_score, KL_Bern_to_prior, backup_rank, backup_stage, backup_state, backup_action` | **partial** | only per-stage aggregates; see R4 |

## Test coverage vs spec §9

| Spec test class                     | Required items | Implemented items | Pass? |
| ----------------------------------- | -------------- | ------------------ | ----- |
| §9.1 SafeDoubleQ                    | 5              | 5 (all five)       | Yes   |
| §9.2 Target-table                   | 5              | 5                   | Yes   |
| §9.3 State-dependent scheduler      | 6              | 0 (registry stubs)  | **No (R5)** |
| §9.4 Geometry-priority DP           | 4              | 4 (weak — R3 masks formula errors) | Partial |
| §9.5 Ablation                       | 5              | 0 (registry stubs)  | **No (R5)** |
| §9.6 End-to-end smoke               | 6              | 4 (no statebin, no figures) | Partial |

---

## What is solid

1. `SafeDoubleQLearning` cleanly separates selection / evaluation /
   update, uses the correct evaluation-side bootstrap, logs the
   required double-Q fields, and has a proof-by-simulation test
   (`test_beta0_reduces_to_classical_double_q`) that checks exact
   agreement with hand-rolled classical Double Q on a 200-step
   random trajectory.
2. `SafeTargetQLearning` correctly uses the frozen target table for
   the bootstrap, and `SafeTargetExpectedSARSA` reuses the same
   machinery with a clean eps-greedy expected bootstrap. The beta=0
   classical-collapse tests are real (compare to independent Python
   reference).
3. The safe math layer (`SafeWeightedCommon.compute_safe_target`) is
   reused from Phase III; the Phase IV-C agents do not reimplement
   the logaddexp formula.
4. The "sign flip" path in certification ablations is structurally
   real: it flips `sign_family` before pilot, so `beta_used_t` comes
   out negated.

## What is missing or weak

1. State-dependent scheduler machinery (R2, R8, R9 downstream).
2. Priority formula correctness (R3).
3. Per-backup logging (R4).
4. `n_base` bug in certification ablations (R1).
5. Ablation / statebin tests test only registry presence (R5).
6. Constant-u schedule has silent clipping (R6).
7. Attribution analysis mixes agent architectures (R9).

## Recommended next actions

1. Fix R1 (one-line fix) and rerun all certification ablations.
2. Fix R3 (priority formula) and rerun geometry DP.
3. Promote R2 (state-dependent schedulers) to its own work item; it
   is the only headline capability of spec §4 and is entirely
   missing.
4. Replace R5 (registry tests) with behavioral tests before closing
   Phase IV-C.
