# Phase IV-C — Adversarial review

Scope (per review brief): challenge estimator-stabilization correctness
(dual Q-tables, target networks, evaluation-side bootstrap),
state-dependent scheduler freeze discipline and hierarchical backoff,
geometry-priority scoring, trust-region and wrong-sign ablation
isolation, and whether gains are attributable to the claimed mechanism
vs. confounds, per
`docs/specs/phase_IV_C_advanced_stabilization_and_geometry_ablations.md`.

The standard review (`review.md`) lists the correctness bugs.
This file challenges each attribution/mechanism claim the code attempts
to support. If the run dirs were wiped tomorrow and a reviewer asked
"what exactly did you show?", these are the reasons the current
answer is less than the spec promises.

| Tag      | Count |
| -------- | ----- |
| BLOCKER  | 7     |
| MAJOR    | 8     |
| MINOR    | 3     |
| NIT      | 1     |
| DISPUTE  | 0     |

---

## Fix status (post-review)

| Finding | Status | Fix |
| ------- | ------ | --- |
| A1 — scheduler not state-bin | **DOCUMENTED** | `scheduler_mode_note` in run.json explains variants are stagewise quality sweeps; full §4 state-bin scheduling not yet in runner |
| A2 — attribution confounds architecture | **DOCUMENTED** | `architecture_note` in advanced_rl run.json records hand-rolled loop caveat; isolated single-table baseline not yet added |
| A3 — n_base broken in ablations | **FIXED** | Same as R1 |
| A4 — geometry priority formula wrong | **FIXED** | Same as R3; per-backup log added (R4) |
| A5 — wrong_sign co-mutates pilot | **FIXED** | Pilot always uses base_sign; only `beta_used_t` negated post-hoc |
| A6 — target freeze semantics | **DISPUTE** | Documented in `safe_target_q.py` docstring: "frozen between syncs" is the DQN-standard interpretation; the schedule (beta_t) is frozen for the full run |
| A7 — no cross-fitting | **DOCUMENTED** | `leakage_limitation` field added to all IV-C run.json files |

---

## Attribution challenges

### [BLOCKER] A1 — The `state_dependent_scheduler` comparison cannot test state-dependent scheduling

Spec §4 primary question:

> Does localization increase event-conditioned activation and improve
> the task-specific outcome without destabilizing learning?

The runner `run_phase4C_scheduler_ablations.py` declares four
scheduler types: `stagewise_baseline`, `state_bin_uniform`,
`state_bin_hazard_proximity`, `state_bin_reward_region`. For all
four, it calls `build_schedule_v3_from_pilot` — i.e., the **stagewise**
pilot-calibrated v3 builder — with different
`alpha_min`/`alpha_max`/`tau_n`/`alpha_budget_max` kwargs. None of the
four variants perform state-bin construction, none compute
`xi_ref_{t,b}` or `u_target_{t,b}`, none apply hierarchical backoff.

Consequently any positive result reported as "state-dependent
scheduling improves outcome" is actually a claim about
headroom/trust-region parameters of the stagewise schedule, not about
state localization. The state_bins.py module is a stub
(`NotImplementedError`). This is a headline claim that the
implementation cannot support.

See also review.md R2, R8.

### [BLOCKER] A2 — Attribution deltas conflate agent architecture with mechanism

File: `experiments/weighted_lse_dp/runners/aggregate_phase4C.py:208-247`

`_build_attribution` computes
`advanced_rl_gains_vs_baseline[algo] = advanced_rl[algo].mean_final_return - stagewise_baseline.mean_final_return`,
where `stagewise_baseline` comes from the scheduler-ablation runner,
which uses **MushroomRL's `SafeQLearning`** (framework Core.learn,
full callback pipeline), and `advanced_rl[algo]` uses hand-rolled
episode loops in `run_phase4C_advanced_rl.py:167-222` with different
action tie-breaking (global numpy), no callbacks, and handcrafted
per-step global_step tracking.

Any reported "SafeDoubleQ gain" therefore absorbs:

- dual-table vs single-table estimator change (the claim),
- framework vs handwritten training loop (confound),
- unseeded tie-break vs seeded tie-break (confound),
- callback-based logging vs inline logging (confound).

Cleanly isolating "dual-table helps" requires a single-table
hand-rolled baseline run with the same loop. That baseline does not
exist; the attribution file cannot isolate the estimator effect.

### [BLOCKER] A3 — `run_phase4C_certification_ablations.py` broken stage decoding

For the current activation suite (`dense_chain_cost` with
`n_states = horizon = 20`), `n_base = horizon + 1 = 21` shifts stage
decoding by one every horizon steps. A `SafeQLearning` agent with the
wrong `n_base` backs up the wrong `(t, s)` cell: after an episode
roll-out, stage-0 states are scored at `aug < 21` → `t = 0` (correct
for aug 0..19, wrong for aug=20 which is actually stage 1 state 0);
all subsequent stages are shifted. The resulting Q-values are
approximately correct "on average" (the shift is small and values are
averaged across stages during training), but they are not the
finite-horizon stage-indexed values the safe target expects. The
ablation returns therefore reflect the wrong Bellman backup, not the
certification mechanism under test.

Every numeric claim from certification ablations (trust_region_off,
wrong_sign, constant_u, raw_unclipped) is affected. See review.md R1.

### [BLOCKER] A4 — Geometry-priority DP: wrong priority, stagewise-only, unlogged per-backup trace

Three separate issues stack (review.md R3, R4, R11, R12):

1. The `geom_gain` formula uses `γ · |ρ − 1/(1+γ)|` rather than
   `|effective_discount_used − γ_base|`. Within a stage this is
   proportional to the spec formula (ranks agree), so single-stage
   micro-benchmarks still work — but across stages with different
   γ_base regimes (the lower-base-γ experimental setup in §10), the
   proportionality constant γ differs per-task and ranks may re-order.
2. `KL_Bern_to_prior` in the code is `KL(ρ || 0.5)`, not
   `KL(ρ || 1/(1+γ))`. At γ=0.95 the difference in minimum
   location is ~0.013, which is small but systematically biased
   toward 0.5 (not toward the weighted-LSE prior).
3. `priority_score` is built from stagewise `(geom_gain_t, u_ref_t,
   kl_t)` multiplied by per-state residual. This means "geometry
   priority" only adds stage-level weighting on top of pure
   residual priority. The advertised mechanism — biasing backups
   toward high-activation *states* — is not implemented; only
   high-activation *stages* receive extra priority.

Additionally, `residual_history` records pre-update residuals, and
only 25% of `(s,t)` pairs are updated per sweep, so the residual
curves in spec §12.1 #5 may legitimately oscillate. The smoke test
only asserts `hist[-1] <= hist[0] + 1e-6`.

Consequence: "does geometry-priority reduce backups needed to resolve
high-activation / high-impact states" cannot be answered from these
runs because the priority is not per-state geometric.

### [BLOCKER] A5 — The `wrong_sign` ablation is a mixed sign-flip, not an isolated one

File: `experiments/weighted_lse_dp/runners/run_phase4C_certification_ablations.py:127-148`

```python
base_sign = int(get_task_sign(cfg.get("family", "unknown")))
sign_family = -base_sign if flip_sign else base_sign
pilot = run_classical_pilot(..., sign_family=sign_family)
...
v3 = build_schedule_v3_from_pilot(..., sign_family=sign_family, ...)
```

`sign_family` is passed BOTH to the pilot (where it controls only
`p_align_t = P(sign_family * margin > 0)`) AND to the schedule builder
(where it controls the sign of `beta_used_t`). Flipping both at once
means the "wrong-sign" ablation simultaneously:

1. Flips the operator sign (the intended test).
2. Relabels which half of the margin distribution counts as "aligned"
   in the diagnostic `p_align_t`.

The second change affects `informativeness_t` and can ripple through
`alpha_t` (via the informativeness-weighted headroom schedule) if
`p_align_t` enters the `alpha_base_t` computation. A clean wrong-sign
ablation should rebuild the pilot with the correct sign_family and
then only negate `beta_used_t`. Otherwise a null wrong-sign result
could either mean "sign doesn't matter" (the interesting claim) or
"the informativeness-driven alpha schedule compensated for the sign
flip" (a confound).

Acceptance criterion: factor `_build_ablated_schedule` so the pilot
uses `base_sign` and only the final `beta_used_t` array is negated
post-hoc. Verify with a pair of runs where
`p_align_t`, `alpha_t`, `kappa_t`, `Bhat_t` are identical and only
`beta_used_t` is sign-flipped.

### [BLOCKER] A6 — Target network is NOT frozen across the "learning phase" — it is synced mid-learning

Spec §0.6:

> Any learned or state-dependent scheduler must be frozen during each
> Bellman-learning phase.

Spec §3.2:

> target table is frozen between sync / Polyak updates; schedule
> remains frozen during the Bellman-learning phase.

Implementation (`safe_target_q.py:193-208`):

- Hard-sync mode: every `sync_every = 200` global steps, target is
  fully copied from online.
- Polyak mode: every step, target += tau · (online − target) with
  `polyak_tau = 0.05`.

So "frozen between syncs" is respected in hard mode, but the
Bellman-learning phase spans all 20,000 training steps — during which
there are 100 syncs in hard mode and 20,000 soft updates in Polyak
mode. The target is never frozen "during the Bellman-learning phase"
in any run. Whether this is a spec violation depends on the reading
of §3.2: if "learning phase" means "between consecutive syncs" the
implementation is fine; if it means "during the entire training run"
the implementation never freezes.

Resolution path: the adversarial reading is the relevant one here,
because spec §3.4 demands comparison against "Phase IV-B
SafeQLearning stagewise baseline" which uses a *truly frozen* schedule
(calibrated once from pilot and never updated). A target-network that
updates 20,000 times is not a fair comparator for "scheduler remains
frozen during learning". The DISPUTE is documented in case the user
reads §3.2 as within-phase freezing.

Acceptance criterion: either (a) add an explicit "frozen target"
config where the target is synced only once from an initial pilot
and never again during the 20k steps — run that and compare to the
current mid-learning-sync runs, or (b) clarify in the phase report
that "frozen" in §3.2 means "between syncs" and accept the confound.

### [BLOCKER] A7 — No cross-fitting for schedule construction

Spec §4.5 / §11.3:

> State-dependent or learned frozen schedulers should be fit on pilot
> seeds / pilot episodes and evaluated on disjoint seeds / later
> episodes whenever feasible. If cross-fitting is not feasible, log
> the limitation clearly.

In every IV-C runner, the pilot that builds the schedule is run with
`seed` equal to the run seed (`run_classical_pilot(cfg=cfg, seed=seed)`),
and the same seed is used for `seed_everything(seed)` before training.
There is no pilot/eval seed separation, no held-out episode split.
This means every reported outcome number uses a schedule calibrated
on the same random stream that then evaluates it. Over 200 pilot
episodes + 20k training steps the leakage is small, but it is
structural and is not logged or surfaced.

Acceptance criterion: either implement the pilot-on-disjoint-seed
rule (pilot with `seed + 10_000` and train with `seed`) or add a
`leakage_limitation: "pilot and train share seed"` field to
`run.json` per §4.5.

### [MAJOR] A8 — "Evaluation-side bootstrap" is isolated, but selection-side coin entropy is shared

`SafeDoubleQLearning` uses `self._rng` for both the selector coin
(line 211) and the argmax tie-break (line 239). In the classical
Double Q reduction test this is fine (the reference uses the same
stream). But at runtime, when evaluating whether the evaluation-side
bootstrap is genuinely independent of the selection-side policy, the
following holds: if all Q-values are tied (initial conditions), the
tie-break consumes the RNG, which shifts the subsequent coin-flip
stream relative to a reference where ties are absent. Across seeds
this produces an unstated coupling between "how many Q-ties occurred"
and "which table got updated".

Not a correctness bug but weakens the attribution claim that the
estimator-stability gains in aggregated `mean_double_gap` can be
entirely explained by the estimator choice rather than by an
interaction between estimator structure and tie-break order.

### [MAJOR] A9 — `raw_unclipped` ablation does not actually bypass caps

File: `experiments/weighted_lse_dp/runners/run_phase4C_certification_ablations.py:94-102`

```python
"raw_unclipped": {"tau_n": 1e-9, "u_max": 0.10,
                  "alpha_max": 0.40, "alpha_budget_max": 0.50},
```

This relaxes the trust region (tau_n → 0) and inflates `u_max` and
`alpha_max`. It does NOT skip the safe-certification cap: the
resulting v3 still goes through `build_schedule_v3_from_pilot` which
computes `beta_cap_t` from `alpha_t` and enforces
`|beta_used_t| <= beta_cap_t` after trust-region relaxation. Spec
§6.5 literally says "raw-unclipped ablation bypasses caps only when
explicitly configured". The implementation does not bypass the safe
cap; it only widens the trust cap. Results labelled
"raw_unclipped" are therefore *certification-capped, trust-relaxed*
runs, not uncertified ones.

Acceptance criterion: add an explicit `"_skip_safe_cap": True` flag in
the override dict that, when set, replaces the clipping step in
`_wrap_v3_schedule_for_betaschedule` with a warning-only pass-through.
Run with that flag to get the true raw-unclipped behavior.

### [MAJOR] A10 — Ablation "impact" for wrong_sign is computed vs. a different-architecture baseline

File: `aggregate_phase4C.py:236-247`

```python
ablation_impact.append({
    "ablation": abl,
    "mean_final_return": ret,
    "delta_vs_baseline": ret - baseline_return,
})
```

`baseline_return` is the `stagewise_baseline` scheduler-ablation
result. But the ablations runner and the scheduler-ablations runner
BOTH use MushroomRL's `SafeQLearning` (good — same architecture), so
this particular delta is architecturally clean. HOWEVER, for
`wrong_sign_ablation_hurts` in `summary.summary`:

```python
"wrong_sign_ablation_hurts": any(
    (r.get("delta_vs_baseline") or 0) < -0.01
    for r in ablation_impact if r["ablation"] == "wrong_sign"
),
```

a single seed-task pair with delta `< -0.01` declares the sign-flip
harmful. This is not a significance test. With three seeds, two tasks,
and one chance event, P("any one seed-task shows >0.01 degradation"
under the null hypothesis of zero effect) is easily ~50% for a random
walk at realistic chain-reward scales. Phase II and III spec items
explicitly require paired-seed statistics. Dropping that here
weakens the wrong-sign claim.

Acceptance criterion: replace `any(...)` with a paired-seed t-test
or Wilcoxon sign-rank across seeds, per spec §11.2 pairing rule.

### [MAJOR] A11 — Adaptive headroom ablation does not disentangle alpha from beta_cap

`adaptive_headroom_off` sets `alpha_min = alpha_max = 0.05`. This
forces a flat informativeness schedule. But `build_certification`
derives `beta_cap_t` directly from `alpha_t`, so lowering/flattening
`alpha_t` also changes `beta_cap_t`, which can re-clip `beta_raw_t`
and change `clip_active_t`. The ablation therefore mixes "turn off
adaptive headroom" with "change beta caps". Spec §6.2 lists both
"U_safe_ref_t" and "safe clip activity" as things to report, which
is correct — but the aggregator does not currently log either. A
reader cannot tell from the metrics whether a change in outcome came
from the alpha schedule or from the resulting cap.

Acceptance criterion: log `mean_safe_clip_active`, `mean_trust_clip_active`,
`mean_beta_cap_t`, and `mean_alpha_t` per run; include in the
per-ablation summary.

### [MAJOR] A12 — No stability diagnostics are aggregated across seeds

Spec §7 (Estimator-stability diagnostics) requires:

- target variance;
- TD-error variance;
- `q_target_gap`;
- `double_gap`;
- online vs target policy disagreement if applicable;
- across-seed schedule stability;
- margin-estimation variance;
- natural-shift variance;
- correlation between bootstrap-value error and natural-shift error.

What is actually aggregated (`aggregate_phase4C.py:109-118`):

- `mean_final_return`,
- `mean_mean_return`,
- `mean_beta_used`,
- `mean_double_gap`,
- `mean_q_target_gap`.

Missing: target variance, TD-error variance, across-seed schedule
stability, margin variance, natural-shift variance, and the
bootstrap-error / shift-error correlation. The event-conditioned
versions (§7 last paragraph) for jackpot/catastrophe/hazard families
are entirely absent. Any "estimator stabilization reduced target
variance" conclusion requires those diagnostics; they are not
recorded.

### [MAJOR] A13 — Per-step logs (`all_logs`) are kept in memory, summarized to means, and discarded

File: `run_phase4C_advanced_rl.py:342-348`

```python
diag: dict[str, float] = {}
for key in diag_keys.get(algorithm, ["beta_used"]):
    vals = [log[key] for log in all_logs if key in log]
    if vals:
        diag[f"mean_{key}"] = float(np.mean(vals))
```

Only the per-key mean is kept. Variance, percentiles, stage-wise
breakdown, event-conditioned slicing — none of them is saved. This
means the downstream figures required by spec §12.1 #1–#4 cannot be
regenerated from the saved artifacts. The per-step logs are discarded
at the end of `run_single`.

Acceptance criterion: dump `all_logs` (or at minimum the estimator
diagnostic fields) to `transitions.npz` per run.

### [MAJOR] A14 — No comparison against Phase IV-B stagewise baseline

Spec §0.8:

> Every advanced method must be compared against the simple Phase
> IV-B stagewise baseline.

The scheduler-ablation runner produces a `stagewise_baseline` result,
but that is the *Phase IV-C* stagewise baseline (its own v3 defaults).
Phase IV-B's stagewise baseline is a *separate* calibrated schedule
with Phase IV-B's chosen `u_min`/`u_max` and its own pilot. Neither
the aggregator nor any runner reads from Phase IV-B result directories.
The comparison that §0.8 demands is not wired up.

### [MINOR] A15 — 3 seeds is below the §11.1 minimum of 5

File: `run_phase4C_{advanced_rl,scheduler_ablations,certification_ablations}.py` all define
`_SEEDS = [42, 123, 456]`.

Spec §11.1: "5 seeds for all advanced variants; 10 seeds for the
most important families". The default 3 seeds will produce wider
CIs than the spec anticipates. Run the full sweep at the spec
minimum before any paper numbers are cited.

### [MINOR] A16 — `convergence_sweep_1e-2` can be `None`

If the planner never reaches residual < 1e-2 (or only reaches it at
the same sweep as residual < tol=1e-6), `convergence_sweep_1e2` stays
`None`. The aggregator's `_mean_field("convergence_sweep_1e-2")` then
ignores those records, producing a mean over only the "fast-convergent"
subset — a selection bias in the mean sweep count. The aggregator
should either impute the max_sweeps upper bound or report the
non-convergent fraction explicitly.

### [MINOR] A17 — `test_plan_beta0_classical_collapse` compares two safe-operator runs, not safe vs. an independent classical VI

File: `tests/algorithms/test_phase4C_geometry_priority_dp.py:135-151`

Both `planner_safe` and `planner_classic` are `GeometryPriorityDP`
instances with different λ values but both driven by the same
β=0 schedule. The test asserts their `V` arrays agree, which is a
consistency check on the priority wrapper — not a classical-operator
correctness check. A proper test would construct V via
`np.einsum(p, r + γ·V)` recursion (or call `value_iteration` from
the MushroomRL DP utilities) and compare.

### [NIT] A18 — `summary.summary` in `attribution_analysis.json` uses `or -1e9`

File: `aggregate_phase4C.py:260`

```python
"best_advanced_algo": max(advanced_rl_gains, key=lambda k: advanced_rl_gains[k] or -1e9)
```

When `advanced_rl_gains[k]` is literally `0.0`, it is falsy and
replaced by `-1e9`. A zero-gain algorithm would be treated as worst
rather than tied. Use `-float("inf") if v is None else v` instead.

---

## Attribution scorecard

The spec closes with seven questions per task family (§14):

1. **Did estimator stabilization reduce target variance or
   overestimation?** — *Unanswerable from current artifacts* (A12,
   A13): variance is not aggregated.
2. **Did estimator stabilization improve Safe TAB outcomes beyond
   Phase IV-B?** — *Unanswerable* (A14): no Phase IV-B comparator
   is wired up; the "advanced_rl_gains_vs_baseline" delta uses a
   different architecture (A2).
3. **Did state-dependent scheduling improve event-conditioned
   activation localization?** — *Unanswerable* (A1): state-dependent
   scheduling is not implemented; the reported variants are
   stagewise.
4. **Did state-dependent scheduling improve outcomes or only
   diagnostics?** — *Unanswerable* (A1).
5. **Did geometry-priority DP reduce backups or wall-clock to
   tolerance?** — *Partially answerable* (A4): only stagewise
   priority is implemented; per-state geometric prioritization is
   not. Raw sweep counts are logged.
6. **Were trust and safe clips load-bearing in the activated
   regime?** — *Partially answerable* (A9, A11): `raw_unclipped`
   doesn't actually bypass the safe cap; adaptive-headroom ablation
   mixes alpha and cap changes and doesn't log cap utilization.
7. **Did wrong-sign scheduling harm or neutralize the effect as
   expected?** — *Weakly answerable* (A5, A10): the pilot and
   schedule both see the flipped sign; a null result could be
   the pilot compensating, not the operator doing nothing.
8. **Are final gains still attributable to TAB nonlinearity, or
   mostly to estimator/scheduler engineering?** — *Unanswerable*
   given A1, A2, A4, A12: the mechanism breakdown the spec asks for
   requires diagnostics and runs that are not yet collected.

## Bottom line

The advanced algorithm implementations (Safe{Double,Target,TargetES}Q)
are individually solid. The spec §3 tests demonstrate that they
reduce to their classical counterparts at β=0 and that the
evaluation-side bootstrap is correctly separated from the selection
table. Those are real results.

Everything above §3 — state-dependent schedulers, geometry-priority
backups, the certification ablation suite, and the attribution
analysis — either has a first-order correctness bug, does not
implement what the spec section name claims, or lacks the diagnostics
required to answer the spec's own questions. A Phase IV-C report
written today would have to either limit its scope to "we have
working SafeDoubleQ / SafeTargetQ / SafeTargetExpectedSARSA agents
that pass β=0 reductions" or claim results the artifacts do not
support.
