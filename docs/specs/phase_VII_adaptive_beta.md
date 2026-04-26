# Phase VII — Adaptive-β Experiments (canonical spec)

**Status:** active. Single source of truth for the Phase VII exploratory empirical
program on adaptive signed temperature scheduling.

**Role in the paper:** **exploratory only.** Phase VII does not modify the
current main-paper §Experiments (pruned at commit `ccec6965`). Final Phase VII
deliverable is a report + recommendation memo; the user — not the planner, not
the runner — decides whether the results land in the appendix, supplementary,
or a future revision.

**Inputs that define this spec:**
- `tasks/adaptive_beta_experiments_coding_agent_spec.md` — coarse coding-agent
  spec authored 2026-04-26. Authoritative for the operator math, schedule rule,
  environment list, metrics, figures, tables, and acceptance criteria.
- User decisions locked 2026-04-26 (orchestration, staging, code home, gate
  cadence, no-clip handling). These supersede any conflicting default in the
  coarse spec.

If the two inputs conflict, the **2026-04-26 user decisions** win; otherwise
the coarse spec governs.

---

## 0. Scientific objective

Test whether **adaptive signed temperature scheduling** for the paper-consistent
centered/scaled weighted-LSE Bellman operator improves learning dynamics in
adversarial, sparse-reward, and non-stationary tabular environments — relative
to classical Bellman learning and to fixed-β weighted-LSE Bellman learning.

The empirical claim under test (coarse-spec §0):

> Adaptive β acts as a temporal credit-assignment controller. When the sign of
> β aligns with the local Bellman advantage, the operator reduces the effective
> continuation coefficient below the classical discount, accelerating
> propagation of informative transitions and improving recovery under
> non-stationarity while remaining stable under clipping.

Five measurable predictions (verbatim from coarse-spec §0):
1. Adaptive β increases alignment rate.
2. Adaptive β reduces effective continuation on informative transitions.
3. Adaptive β improves recovery after environment shifts.
4. Adaptive β reduces catastrophic episodes or drawdowns.
5. Adaptive β improves AUC and sample efficiency, even when final asymptotic
   return is similar.

The spec is **mechanism-first**: alignment-rate and effective-discount
diagnostics carry as much weight as raw return.

---

## 1. What Phase I–VI already established

1. Phases I–IV established the safe weighted-LSE operator, calibration
   pipeline, and certification machinery on finite-horizon tabular MDPs with
   per-stage `β̃_t` schedules.
2. Phase V (Family A / C) produced the main-paper positive result (concentration
   contrast on aligned propagation) and the safety/stability story.
3. Phase VI added a stochastic Family A variant (VI-A through VI-D) and a
   risk-sensitive policy-evaluation finding (VI-G); the paper §Experiments was
   subsequently pruned to two positive claims (commit `ccec6965`).
4. None of the prior phases tested **per-episode** adaptive β with a sign
   driven by the empirical advantage signal `A_e = mean(r - v_next)`.
5. Phase VII is **exploratory** and runs in parallel to the main paper. It
   does not affect the pruned §Experiments unless and until the user
   explicitly promotes results from the Stage C headline run.

---

## 2. Non-negotiable rules

Adapted from Phase V §2; Phase VII–specific additions in items 6–11.

1. **Operator.** All methods compute targets through the centered/scaled paper
   operator `g_{β,γ}(r,v) = (1+γ)/β · log((e^{βr} + γ e^{βv}) / (1+γ))` (or
   classical `r + γv` at `|β| ≤ 1e-8`). Never substitute the unscaled
   Bernoulli-prior log-sum-exp aggregator.
2. **β scope.** β is updated **between episodes only**. Within an episode β is
   a constant. β for episode `e+1` is computed strictly from data observed in
   episode `e` and earlier.
3. **Paired seeds.** All methods on a given environment share the same
   environment-RNG stream per seed slot, enabling paired statistical
   comparison. Method-specific RNG offsets exist only for ε-greedy.
4. **Same code path.** `vanilla`, `fixed_*`, `adaptive_*`, and `wrong_sign`
   all flow through one Q-learning update routine; they differ only in the
   schedule object passed in.
5. **No future leakage.** β scheduling never reads future-episode data;
   episode `e+1`'s β depends only on `A_e` (and earlier, if smoothed).
6. **No-clip failures are data, not bugs.** `adaptive_beta_no_clip` runs that
   diverge, NaN, or overflow are recorded as first-class outcomes in
   `metrics.npz` (`divergence_event`, `nan_count`, `q_abs_max`). Tests
   explicitly assert these failures are *recorded*, never that they don't
   happen. Suppressing or silently restarting a divergent no-clip run is a
   spec violation.
7. **No silent dropping.** If a run errors out, the failure is logged in the
   manifest with stack trace and the manifest is still emitted.
8. **Cross-method hyperparameter parity.** ε-greedy schedule, learning rate,
   γ, and seed protocol are identical across methods. The only thing that
   varies is β (raw / clipped / scheduled).
9. **Final report verdict.** `results/adaptive_beta/final_report.md` must
   state, in §1, whether the claim is **supported / partially supported /
   not supported**, with explicit pointers to the supporting numbers.
10. **No paper edits in Phase VII.** Do not touch
    `paper/neurips_selective_temporal_credit_assignment_positioned.tex` or
    any other tex file in this phase. The final WP is "produce the report",
    not "rewrite §Experiments".
11. **Stage gates are user-controlled.** Stage A → B → C transitions require
    explicit user sign-off. The runner/orchestrator stops at each gate and
    reports.

---

## 3. Operator and effective-discount

### 3.1 Mathematical form (coarse-spec §1)

For β ≠ 0:

```
g_{β,γ}(r, v) = (1 + γ) / β · log((e^{β·r} + γ · e^{β·v}) / (1 + γ))
```

For β = 0 (classical collapse):

```
g_{0,γ}(r, v) = r + γ · v
```

### 3.2 Effective continuation (coarse-spec §1.2)

```
ρ_{β,γ}(r, v) = e^{β·r} / (e^{β·r} + γ · e^{β·v})
              = sigmoid(β · (r - v) - log γ)
d_{β,γ}(r, v) = ∂_v g_{β,γ}(r, v) = (1 + γ) · (1 - ρ_{β,γ}(r, v))
```

For β = 0, return `d = γ`.

### 3.3 Alignment condition (coarse-spec §1.3)

```
d_{β,γ}(r, v) ≤ γ  ⇔  β · (r - v) ≥ 0
```

Per-transition log: `aligned = (β · (r - v) > 0)` (strict for the headline
metric; non-strict variant logged separately).

### 3.4 Existing repo operator surface

The certified, finite-horizon variant of this operator already lives at:

- `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py`
  - `SafeWeightedCommon.compute_safe_target(r, v_next, t)`
  - `SafeWeightedCommon.compute_effective_discount(r, v_next, t)`
  - `SafeWeightedCommon.compute_rho(r, v_next, t)`

That class **requires a `BetaSchedule`** (per-stage, finite-horizon, certified).
Phase VII β is per-episode and uses a simple symmetric clip cap, not the
certified per-stage cap.

**Resolution (2026-04-26, user-locked).** The integration uses **option (b)**:
extract the stateless math kernel (the logaddexp branch at
`safe_weighted_common.py` lines 459–480) into a thin module under
`src/lse_rl/operator/tab_operator.py` and have **both** the existing
certified planner (`SafeWeightedCommon.compute_safe_target`,
`compute_effective_discount`, `compute_rho`) and the new Phase VII agent
import from it. The refactor must be isolated, behavior-preserving, and
backward-compatible; numerical equivalence (before vs after) must be pinned
by a regression test on a fixed (β, γ, r, v) grid (see test 13.1.6). The
MushroomRL-dev edit is recorded in `tasks/lessons.md` per CLAUDE.md §4.

---

## 4. Adaptive-β schedule

### 4.1 Episode signal (coarse-spec §3.1)

```
A_e = (1 / H_e) · Σ_{t=0..H_e-1} (r_t - v_t^next)
```

where for Q-learning `v_t^next = max_{a'} Q(s_{t+1}, a')` and `v_next = 0`
at terminals.

### 4.2 Raw rule and clipping (coarse-spec §3.2, §2.1)

```
β_{e+1}^raw = β_max · tanh(k · A_e)
β̃_{e+1}    = clip(β_{e+1}^raw, -β_cap, β_cap)
```

Default hyperparameters (Stage A and Stage B unless stated otherwise):

```yaml
beta_max: 2.0
beta_cap: 2.0
k: 5.0
initial_beta: 0.0
beta_tol: 1e-8
```

### 4.3 Smoothing (coarse-spec §3.3)

Optional: `Ā_e = (1 - λ) · Ā_{e-1} + λ · A_e`. Default `λ = 1.0` (no
smoothing). Sensitivity grid (Stage B only): `λ ∈ {1.0, 0.5, 0.2}`.

### 4.4 Stage-B sensitivity grid (coarse-spec §11.2)

Run only on the strongest 1–2 environments after Stage A:

```yaml
beta_max_grid: [0.5, 1.0, 2.0]
k_grid:        [1.0, 5.0, 10.0]
beta_cap_grid: [0.5, 1.0, 2.0]
```

Do **not** run the full Cartesian product on every environment. Coarse-spec
§11.2 priority ordering: switching bandit → adversarial RPS → gridworld
hazards.

---

## 5. Methods (coarse-spec §4.1)

| Method ID                  | Rule                                                                |
|----------------------------|---------------------------------------------------------------------|
| `vanilla`                  | β = 0 (classical Bellman)                                           |
| `fixed_positive`           | β = +β₀                                                             |
| `fixed_negative`           | β = −β₀                                                             |
| `wrong_sign`               | β = ±β₀ with sign deliberately wrong for env family — **only on `delayed_chain` (canonical +β → wrong −β) and `hazard_gridworld` (canonical −β → wrong +β)**; not defined elsewhere (see §22.3)     |
| `adaptive_beta`            | `β = clip(β_max·tanh(k·A_e), -β_cap, β_cap)` between episodes       |
| `adaptive_beta_no_clip`    | same as `adaptive_beta` but no clip; failures logged, not suppressed|
| `adaptive_sign_only`       | `β = β₀ · sign(A_e)`, clipped                                       |
| `adaptive_magnitude_only`  | `β = sign_env · β_max · |tanh(k·A_e)|`, clipped                     |

**Integration constraint:** all methods share the same Q-learning update code
path (coarse-spec §16.2). The schedule object is the only thing that varies.

Stage A core methods: `vanilla`, `fixed_positive`, `fixed_negative`,
`adaptive_beta`, `adaptive_beta_no_clip` (5 methods).
Stage B adds: `wrong_sign`, `adaptive_sign_only`, `adaptive_magnitude_only` (3
ablation methods).

**`wrong_sign` applicability (2026-04-26 resolution of §22.3):** the method
is only defined where the environment has a canonical sign:

| Environment        | canonical β | `wrong_sign` β | Notes                          |
|--------------------|-------------|----------------|--------------------------------|
| `delayed_chain`    | +β          | −β             | rewards optimistic propagation |
| `hazard_gridworld` | −β          | +β             | rewards pessimistic propagation|
| `rps`              | n/a         | not defined    | use `fixed_positive` / `fixed_negative` |
| `switching_bandit` | n/a         | not defined    | use `fixed_positive` / `fixed_negative` |
| `selfplay_rps`     | n/a         | not defined    | use `fixed_positive` / `fixed_negative` |

The schedule object for `wrong_sign` must raise on construction when
instantiated against an env without a canonical sign. Stage B ablation
runs for `wrong_sign` only land on the two envs above.

---

## 6. Environments

Each environment lives at `experiments/adaptive_beta/envs/<name>.py` and
exposes the coarse-spec §5 interface (`reset(seed)`, `step(action)`,
`current_phase`, optional `oracle_value_or_best_action`).

**Resolution (2026-04-26, user-locked).** The Phase VII envs subclass the
MushroomRL `Environment` base class (matching Phase I–VI precedent and the
existing runner / logging stack). Implementation must:
- Handle MushroomRL's `(state, info)` tuple convention from `reset()`.
- Use `int(np.asarray(x).flat[0])` to normalize numpy state scalars
  (`tasks/lessons.md`).
- Preserve compatibility with `Core`, `Dataset`, and the standard callback
  surface.

Plain Python classes are allowed only as internal helpers (e.g. an opponent
controller for adversarial RPS), not as the primary env API.

`info` dict keys (mandatory):

```python
info = {
    "phase":             str | int,
    "is_shift_step":     bool,
    "oracle_action":     Optional[int],   # where defined
    "catastrophe":       bool,
    "terminal_success":  bool,
}
```

### 6.1 Adversarial Rock-Paper-Scissors (`rps.py`) — coarse-spec §5.1

- horizon = 20, action set = {rock, paper, scissors}
- opponent phases cycle: `biased_exploitable` → `counter_exploit` →
  `uniform_random`
- switch_period_episodes ∈ {50, 100}
- main variant: hidden phase, memory length 1
- visible-phase variant retained as easy-diagnostic ablation
- oracle action: best-response to the *true* opponent distribution (used for
  regret only, not visible to agent)

### 6.2 Switching Multi-Armed Bandit (`switching_bandit.py`) — coarse-spec §5.2

- 5 arms, horizon = 1, Bernoulli rewards (`p_best=0.8`, `p_other=0.2`)
- best arm rotates cyclically every `switch_period_episodes ∈ {100, 250}`
- single dummy state
- regret signal: per-episode regret against the current best arm
- **Resolution (2026-04-26, user-locked, §22.5):** with horizon = 1, every
  transition is terminal so `v_next = 0`, `A_e = mean(r_t)`, and the
  alignment / effective-discount story is degenerate. The bandit is
  retained as a **performance benchmark only** — it contributes to
  `regret`, `auc_return`, `recovery_time`, and `final_return` claims, and
  is **excluded from alignment-rate and effective-discount mechanism
  panels** (figures `alignment_rate_*.pdf`, `effective_discount_*.pdf`)
  and from the mechanism columns of the main results table.

### 6.3 Gridworld with Adversarial Hazards (`hazard_gridworld.py`) — coarse-spec §5.3

- 7×7 grid, horizon = 50
- start = (0,0), goal = (6,6), num_hazards = 5
- rewards: goal = +10, hazard = −10, step = −0.01
- hazards switch every `hazard_switch_period_episodes ∈ {100, 250}`
- terminals: goal reached, hazard entered, horizon exhausted
- catastrophe flag fires on hazard entry
- diagnostics: hazard hit rate, goal success rate, recovery after shift
- canonical sign for `wrong_sign` ablation: **−β** (this env rewards
  pessimistic propagation around hazards); `wrong_sign = +β` here (see
  §22.3 resolution and §5)

### 6.4 Delayed Reward Chain (`delayed_chain.py`) — coarse-spec §5.4

- chain_length = 20, horizon = 25, terminal_reward = 50, step_reward = 0
- actions = {forward, reset_or_stay} (default = `reset` for the distractor
  action)
- deterministic; oracle_action = `forward` at every state
- canonical sign for `wrong_sign` ablation: **+β** (this env rewards
  optimistic propagation); `wrong_sign = −β` here (see §22.3 resolution
  and §5)

### 6.5 Self-Play RPS (`selfplay_rps.py`) — coarse-spec §5.5

- two-agent symmetric and asymmetric (A=adaptive, B=vanilla) conditions
- secondary environment per coarse-spec §5.5
- outputs: exploitability proxy, action entropy, cycling indicator, return
  variance
- **Resolution (2026-04-26, user-locked, §22.4):** self-play RPS is
  **excluded from Stage A**. It enters Stage B *fresh* (no Stage-A signal of
  its own) and **only if** at least one Stage-A environment (`rps`,
  `switching_bandit`, `hazard_gridworld`, `delayed_chain`) clears the
  Stage-A → Stage-B promotion bar with stable adaptive-β signal and
  controlled variance. Self-play is a secondary stress test, not a
  primary signal generator, and does not contribute to the Stage-A → B
  promotion decision.
- `wrong_sign` is **not defined** for self-play RPS (no canonical sign);
  use `fixed_positive` and `fixed_negative` instead (see §5 / §22.3).

---

## 7. Metrics

### 7.1 Core performance (coarse-spec §6.1)

Per (method, env, seed):

| Metric                  | Definition                                                           |
|-------------------------|----------------------------------------------------------------------|
| `final_return`          | mean return over last `N_final = 500` episodes                       |
| `auc_return`            | area under episode-return curve                                      |
| `sample_efficiency`     | first episode reaching threshold for `W = 100` consecutive episodes  |
| `regret`                | cumulative oracle regret where oracle is available                   |
| `recovery_time`         | episodes after a shift to recover to pre-shift moving avg            |
| `max_drawdown`          | largest peak-to-trough drop in smoothed return                       |
| `catastrophic_episodes` | count of episodes below env-specific threshold                       |
| `success_rate`          | goal/completion rate where applicable                                |

Smoothing window = 100. Threshold window = 100.

### 7.2 Mechanism metrics (coarse-spec §6.2)

Per transition: `adv = r - v_next`, `aligned = β · adv > 0`,
`d_eff = effective_discount(r, v_next, β, γ)`.

Per episode (mandatory): `alignment_rate`, `mean_signed_alignment`,
`frac_positive_signed_alignment`, `mean_abs_advantage`, `mean_d_eff`,
`median_d_eff`, `frac_d_eff_below_gamma`, `frac_d_eff_above_one`,
`mean_gamma_minus_d_eff`.

### 7.3 Stability metrics (coarse-spec §6.3)

`bellman_residual`, `td_target_abs_max`, `q_abs_max`, `nan_count`,
`divergence_event` (fires if NaN/Inf or `q_abs_max > 1e6`).

### 7.4 Schema headers (coarse-spec §15)

Episode CSV columns (verbatim from coarse-spec §15.1):

```
run_id, env, method, seed, episode, phase, beta_raw, beta_deployed,
return, length, epsilon, alignment_rate, mean_signed_alignment,
mean_advantage, mean_abs_advantage, mean_d_eff, median_d_eff,
frac_d_eff_below_gamma, frac_d_eff_above_one, bellman_residual,
td_target_abs_max, q_abs_max, catastrophic, success, regret,
shift_event, divergence_event
```

Transition CSV/parquet columns:

```
run_id, env, method, seed, episode, t, state, action, reward,
next_state, done, phase, beta_deployed, v_next, advantage,
td_target, td_error, d_eff, aligned, oracle_action, catastrophe
```

Stage A: full transition logs.
Stage B/C: stratified samples (e.g. every 10th transition + all
`is_shift_step=True` and `catastrophe=True`) to bound disk usage.

---

## 8. Experimental protocol — staged

### 8.1 Stage A — development pass (3 seeds × 1k episodes)

Scope:

```yaml
episodes: 1000
seeds: 3
methods: [vanilla, fixed_positive, fixed_negative, adaptive_beta, adaptive_beta_no_clip]
envs: [rps, switching_bandit, hazard_gridworld, delayed_chain]
sensitivity_grid: false
```

- 4 envs × 5 methods × 3 seeds = **60 runs** total (self-play excluded
  per §22.4 resolution)
- must complete in well under one hour on CPU
- `selfplay_rps` is **not** part of Stage A; it enters Stage B fresh, gated
  on Stage A producing a stable adaptive-β signal on at least one of the
  four Stage-A envs (§6.5, §22.4)

**Acceptance for Stage A → Stage B promotion:**
- All 75 runs complete (or document failures honestly).
- Mechanism diagnostics (`alignment_rate`, `mean_d_eff`) are produced and
  non-degenerate.
- A Stage-A summary identifies the 1–2 strongest environments by
  paired-mean improvement of `adaptive_beta` over `vanilla` on AUC and
  recovery time, **and** by mechanism evidence (alignment rate clearly
  > 0.5 on informative transitions; `mean_d_eff` clearly < γ on
  informative transitions).
- User signs off on the promotion list.

**Stage A blocks all Stage B work.** No Stage B runs are dispatched until the
user has reviewed `results/adaptive_beta/stage_A_summary.md` and explicitly
named the promoted environments.

### 8.2 Stage B — main pass (10 seeds × 10k episodes)

Scope (gated by Stage A):

```yaml
episodes: 10000
seeds: 10
methods: [vanilla, fixed_positive, fixed_negative, wrong_sign,
          adaptive_beta, adaptive_beta_no_clip,
          adaptive_sign_only, adaptive_magnitude_only]
envs: <1-2 strongest from Stage A>
sensitivity_grid: true   # but only on the strongest two envs
```

- Generates main learning curves, regime-shift recovery plots, mechanism
  panels, ablation table.
- Triggers full Codex gate at end of M4 (see §14).

**Acceptance for Stage B → Stage C promotion:**
- Headline `adaptive_beta` vs `vanilla` paired difference is statistically
  significant (paired bootstrap 95% CI excludes 0) on at least one promoted
  environment, **and**
- Mechanism diagnostics confirm the predicted alignment / effective-discount
  mechanism is active.
- User signs off on a single headline environment for Stage C.

**Stage B blocks Stage C.**

### 8.3 Stage C — headline pass (20 seeds × 10k episodes)

Scope (gated by Stage B):

```yaml
episodes: 10000
seeds: 20
methods: [vanilla, fixed_positive, fixed_negative,
          adaptive_beta, adaptive_beta_no_clip]
envs: <1 headline env from Stage B>
```

If Stage B effect is small, marginal, or unstable, Stage C is **skipped**
with documented justification. This is an acceptable outcome.

### 8.4 Seed discipline (coarse-spec §7.3)

```python
base_seed = 10000 + seed_id
agent_seed = base_seed + method_offset
common_env_seed = base_seed
```

Environment randomness is common across methods at fixed seed_id, supporting
paired statistical comparison (§9).

---

## 9. Statistical reporting

For every metric report mean, std, SE, 95% CI, paired difference vs `vanilla`,
and a paired bootstrap 95% CI on that difference (10,000 resamples).

Paired comparisons (coarse-spec §8):
1. `adaptive_beta` vs `vanilla`.
2. `adaptive_beta` vs the best fixed-β method (per env, declared before the
   run, frozen for Stage C).
3. `adaptive_beta` vs `adaptive_beta_no_clip`.

Use **paired seeds** (same `common_env_seed`) for all significance summaries.
Do not present unpaired mean ± std for the headline comparison.

---

## 10. Figures and tables

### 10.1 Required figures (coarse-spec §9.1)

Per environment that reaches Stage B:

1. `learning_curves_{env}.pdf` — mean return ± SE.
2. `regime_shift_recovery_{env}.pdf` — episodes aligned relative to shift.
3. `alignment_rate_{env}.pdf` — **not produced for `switching_bandit`**
   (mechanism degenerate at horizon = 1, §22.5).
4. `effective_discount_{env}.pdf` — with horizontal line at γ.
   **Not produced for `switching_bandit`** (§22.5).
5. `beta_trajectory_{env}.pdf` — adaptive methods only.
6. `advantage_histogram_{env}.pdf` — separated by phase / pre-vs-post shift.

### 10.2 Killer figure (coarse-spec §9.2)

`adaptive_beta_mechanism_summary.pdf`:
- Panel A: learning curve after shifts.
- Panel B: recovery-time bar plot.
- Panel C: alignment rate over training.
- Panel D: γ − d_eff on informative transitions.

This is the candidate headline figure if Stage C produces a clear effect.

### 10.3 Required tables (coarse-spec §10)

Saved as both `.csv` and `.tex`.

- **Main results table:** Env × Method × {Final Return, AUC, Recovery Time,
  Max Drawdown, Catastrophic Episodes, Align Rate, Mean d_eff}. The
  `Align Rate` and `Mean d_eff` cells for `switching_bandit` are reported
  as `n/a — degenerate at H=1` per §22.5.
- **Ablation table:** Env × Variant × {AUC, Final Return, Align Rate,
  Divergence Events, Mean d_eff, Notes}.
- **Sensitivity table:** Env × {β_max, β_cap, k} × {AUC, Recovery, Align
  Rate, Stability}.

---

## 11. Ablations (Stage B only)

### 11.1 β-variant ablations (coarse-spec §11.1)

Methods to include: `adaptive_beta`, `adaptive_beta_no_clip`,
`adaptive_sign_only`, `adaptive_magnitude_only`, `fixed_positive`,
`fixed_negative`, `vanilla`.

Definitions:
- `adaptive_sign_only`: `β = β₀ · sign(A_e)` with clipping.
- `adaptive_magnitude_only`: fixed sign chosen by env, magnitude
  `β_max · |tanh(k · A_e)|`.

### 11.2 Sensitivity grid (coarse-spec §11.2)

Run on the 1–2 strongest environments only. Priority ordering: switching
bandit → adversarial RPS → gridworld hazards. Do not run the full Cartesian
product on every environment.

### 11.3 Environment-difficulty knobs (coarse-spec §11.3)

Vary shift frequency, reward noise, hazard density, chain length on the
strongest environment to characterize where the mechanism activates.
Purpose is *characterization*, not maximizing performance.

---

## 12. Artifact layout

Adapted from coarse-spec §12 to repo conventions (CLAUDE.md §4):

```
experiments/adaptive_beta/
  README.md
  configs/
    dev.yaml          # Stage A
    main.yaml         # Stage B
    headline.yaml     # Stage C
    ablations.yaml
  schedules.py        # adaptive-β schedule object (per-episode)
  agents.py           # Q-learning with all 7+1 methods (one code path)
  envs/
    rps.py
    switching_bandit.py
    hazard_gridworld.py
    delayed_chain.py
    selfplay_rps.py
  run_experiment.py
  analyze.py
  plotting.py
  # Operator math imported from src/lse_rl/operator/tab_operator.py
  # (single shared kernel; see §3.4 + §22.1 resolution).

tests/adaptive_beta/
  test_operator.py
  test_schedules.py
  test_envs.py
  test_agent.py
  test_reproducibility.py

results/adaptive_beta/
  raw/                # per-run run.json + metrics.npz + transitions.parquet
  processed/          # per-(env, method, seed) summaries
  figures/
  tables/
  logs/
  stage_A_summary.md
  stage_B_summary.md  # written after Stage B
  stage_C_summary.md  # written after Stage C if it runs
  final_report.md     # written at end of M5
```

**Operator code lives in `src/lse_rl/operator/tab_operator.py`** (per
§22.1 resolution) and is *imported* by both `SafeWeightedCommon` (the
existing certified planner in mushroom-rl-dev) and
`experiments/adaptive_beta/agents.py`. Never duplicated.

---

## 13. Tests

### 13.1 Operator tests (coarse-spec §13.1)

1. `g_{0,γ}(r, v) == r + γ · v` exactly.
2. β → 0 limit: `|g_{ε,γ} - g_{0,γ}| < 1e-6` for `|ε| ∈ [1e-12, 1e-9]`.
3. Finite-difference derivative `(g(r, v+h) - g(r, v-h)) / (2h)` agrees
   with `effective_discount(r, v, β, γ)` to `1e-5` for `h = 1e-4`.
4. Alignment condition: for a 100-point grid, verify `d ≤ γ ⇔ β·(r-v) ≥ 0`.
5. Log-sum-exp stability: at `r = v = ±40, β = ±2, γ = 0.95`, output is
   finite and matches a reference quad-precision computation to `1e-10`.
6. **Numerical agreement with `SafeWeightedCommon`**: pick a fixed
   `(β, γ, r, v)` and assert the Phase VII operator returns the same value
   as `SafeWeightedCommon.compute_safe_target` configured with a single-stage
   schedule of that β. This test pins the spec's "use the existing operator"
   contract whichever import resolution path §22.1 takes.

### 13.2 Schedule tests (coarse-spec §13.2)

1. β is constant within an episode (assert by recording every β read inside
   one episode and checking equality).
2. β for episode `e+1` reads only data from `e` and earlier (assert via a
   spy that fails if `episode_index >= e+1` is ever queried during the
   computation of `β_{e+1}`).
3. Clipping never exceeds `β_cap` for `adaptive_beta`.
4. `adaptive_beta_no_clip` *can* exceed `β_cap` (assert at least one such
   excess in a divergence-prone synthetic trace) and must not silently
   overflow without setting the `divergence_event` flag.

### 13.3 Environment tests (coarse-spec §13.3)

1. Phase switches occur at the configured episodes for each env.
2. Rewards match the per-env spec.
3. Terminal conditions fire as specified.
4. Same seed reproduces the same env sequence (action-by-action equality).

### 13.4 Reproducibility test (coarse-spec §13.4)

Run `dev.yaml` twice with the same seed and assert byte-identical
`metrics.npz` (after sorting nondeterministic dict keys).

### 13.5 No-clip-failure-honesty test

Construct a synthetic divergent trace and assert: agent finishes,
`metrics.npz` contains `divergence_event=True`, `nan_count > 0` or
`q_abs_max > 1e6`, and the run-level summary correctly reports the failure
in the manifest. The test passes when the failure is honestly recorded; it
fails if the failure was suppressed.

---

## 14. Milestones

| ID    | Scope                                                  | Codex gate? |
|-------|--------------------------------------------------------|-------------|
| M1    | Operator import + adaptive-β schedule + tests          | No (light)  |
| M2    | Five environments + tests                              | No (light)  |
| M3    | Q-learning agent (one code path, 8 methods) + dev run  | No (light)  |
| M3.5  | **Stage A gate.** Orchestrator pauses for user review. | n/a         |
| M4    | Stage B main runs + figures + tables + ablations       | **Yes**     |
| M4.5  | Stage C headline run (conditional on Stage B sign-off) | No (light)  |
| M5    | Final report + recommendation memo                     | **Yes**     |

### 14.1 Codex gate scope (M4 + M5)

Per AGENTS.md §5.2:
1. `verifier` passes locally (tests + dev-mode smoke + metric sanity).
2. `/codex:review --base main --background`.
3. `/codex:adversarial-review --base main --background "<focus string>"`.
4. `review-triage` writes BLOCKER/MAJOR/MINOR/NIT entries to `tasks/todo.md`.
5. BLOCKER == ∅ before merge.

**Phase VII–specific adversarial focus string** (use for both M4 and M5):

> "challenge whether adaptive-β actually drives the claimed mechanism shift,
> or whether observed gains are confounded by ε-greedy schedule, paired-seed
> leakage, or selection bias from staged gating; verify no-clip failures are
> reported honestly; verify the same code path is used by all methods and
> that fixed_positive / fixed_negative differ from vanilla and adaptive_*
> only in the schedule object."

### 14.2 M1–M3 light gates

- `verifier` runs the full pytest suite under `tests/adaptive_beta/`.
- No Codex review.
- The orchestrator must still demand a structured handoff per AGENTS.md §7.

---

## 15. Logging schema (coarse-spec §15)

Episode CSV and transition CSV/parquet schemas as listed in §7.4.

In addition (CLAUDE.md §4 reproducibility contract):

- Every runner emits `run.json` with `git_sha`, exact argv, seed list, task
  config, resolved hyperparameters, timestamp, output paths.
- Every runner emits `metrics.npz` with a schema header (`schema_version`,
  `keys`, `shapes`).
- The aggregator emits `results/summaries/experiment_manifest.json` listing
  every Stage A / B / C run with status (`completed | failed | skipped`),
  failure reason, and pointer to raw artifacts. The manifest is appended to,
  never overwritten.

---

## 16. Acceptance criteria (coarse-spec §16 + stage gates)

The phase is complete when **all** of:

1. β = 0 path exactly reproduces classical Bellman Q-learning targets
   (verified by test 13.1.1 + a runtime assertion in `agents.py`).
2. Fixed-β methods use the same code path as adaptive-β; only the schedule
   object differs (verified by `[test] same-code-path` assertion + Codex
   adversarial focus string §14.1).
3. β is updated only between episodes (verified by test 13.2.1–13.2.2).
4. All methods on a given env-seed share the same `common_env_seed`
   (verified by test 13.3.4 + manifest cross-check).
5. Every result table includes `align_rate` and `mean_d_eff` columns.
6. At least one regime-shift figure shows recovery around a known shift
   point on each Stage B environment.
7. No silently dropped runs — manifest accounts for every (env, method,
   seed) triple in every stage.
8. `final_report.md` §1 explicitly states **support / partial / fail-to-
   support** with paired-CI numbers.
9. Stage gates: A→B and B→C transitions have user sign-off recorded in
   `tasks/todo.md` (or equivalent durable record).
10. No paper edits made in Phase VII (verified by `git log -- paper/` showing
    no commits attributable to Phase VII).

---

## 17. Final report structure (coarse-spec §17)

`results/adaptive_beta/final_report.md`:

1. **Verdict.** Support / partial / fail. Paired-CI numbers up front.
2. **Experimental setup.** Stages A/B/C scope, env list, method list,
   hyperparameters, seed protocol.
3. **Main performance results.** Tables and figures from §10.
4. **Mechanism diagnostics.** Alignment rate, effective-discount evidence
   per environment.
5. **Ablations.** β-variant + sensitivity grid + difficulty knobs.
6. **Failure cases and negative results.** Including no-clip divergences,
   environments where adaptive β tied or hurt vanilla, and selection bias
   notes from the staged gating.
7. **Recommendation memo.** What (if anything) belongs in
   appendix / supplementary / future paper revision. The user makes the
   final call.
8. **Open implementation questions.** Anything unresolved at end of phase.

---

## 18. Important warnings (coarse-spec §18)

1. Do not optimize hyperparameters separately for each method.
2. Do not choose env variants based on adaptive-β performance alone.
3. Do not hide no-clipping failures — they are evidence.
4. Do not use future-episode information in β scheduling.
5. Do not conflate the centered/scaled paper operator with the unscaled
   KL-prior aggregator.
6. Final conclusions emphasize mechanism evidence, not raw return alone.
7. Do not edit the main paper in this phase — that decision belongs to the
   user after reading `final_report.md`.

---

## 19. Command-line interface (coarse-spec §19)

```bash
python experiments/adaptive_beta/run_experiment.py --config experiments/adaptive_beta/configs/dev.yaml
python experiments/adaptive_beta/run_experiment.py --config experiments/adaptive_beta/configs/main.yaml
python experiments/adaptive_beta/run_experiment.py --config experiments/adaptive_beta/configs/headline.yaml
python experiments/adaptive_beta/analyze.py --results results/adaptive_beta/raw --out results/adaptive_beta/processed
python experiments/adaptive_beta/plotting.py --processed results/adaptive_beta/processed --out results/adaptive_beta/figures
```

Every command writes a metadata file recording git SHA, argv, timestamp,
Python version, package versions, resolved config (CLAUDE.md §4).

---

## 20. Deliverables checklist (coarse-spec §20, mapped to milestones)

- [ ] M1 — operator import resolved, adaptive-β schedule + tests
- [ ] M2 — five env factories + tests
- [ ] M3 — Q-learning agent + dev run + reproducibility test
- [ ] M3.5 — `stage_A_summary.md` + user sign-off
- [ ] M4 — Stage B runs, figures, tables, ablations
- [ ] M4 — Codex gate green
- [ ] M4.5 — Stage C headline run (or documented skip)
- [ ] M5 — `final_report.md` + recommendation memo
- [ ] M5 — Codex gate green
- [ ] M5 — manifest accounts for every run across all stages

---

## 21. Dependencies (blocking edges)

Stages: **A → B → C** (each blocked by user sign-off).

Milestones: **M1 → M2 → M3 → M3.5 → M4 → M4.5 → M5.**

Within milestones:
- M1 `[test] op-tests` blocks M3 `[algo] agent`.
- M1 `[scheduler] schedules` blocks M3 `[algo] agent`.
- M2 envs block M3 `[infra] dev-run`.
- M3 `[infra] dev-run` (Stage A) blocks M3.5 `[infra] stage-A-summary`.
- M3.5 user sign-off blocks all M4 work.
- M4 main runs block M4 plot/table tasks.
- M4 Codex gate blocks M4 → M4.5 transition.
- M4.5 (or its skip decision) + M4 outputs block M5.
- M5 Codex gate blocks phase closure.

Logging schema (`run.json`, `metrics.npz`, transition parquet) is fixed in
M3 and is a prerequisite for every plotting/aggregation task in M4 and M5.

---

## 22. Open questions

All five §22 items were locked by the user on 2026-04-26 (sources:
`tasks/phase_VII_clarifications.md`, `tasks/phase_VII_confirmation.md`).
Resolutions are recorded inline below and propagated into the relevant
spec sections (§3.4, §5, §6.x, §8.1, §10).

### 22.1 Operator import surface — RESOLVED 2026-04-26

**Decision:** option **(b)** — extract the stateless LSE/TAB kernel (the
logaddexp branch at `safe_weighted_common.py` lines 459–480) into
`src/lse_rl/operator/tab_operator.py`. Both Phase III–VI (via
`SafeWeightedCommon.compute_safe_target` etc.) and Phase VII (via
`experiments/adaptive_beta/agents.py`) import from this single source.

**Constraints:** isolated refactor, no behavioral changes, full backward
compatibility. A regression test must pin numerical equivalence (before vs
after refactor) across a fixed `(β, γ, r, v)` grid plus the
`compute_safe_target` numerical-agreement test (test 13.1.6). The
mushroom-rl-dev edit is recorded in `tasks/lessons.md` per CLAUDE.md §4.

### 22.2 Environment base class — RESOLVED 2026-04-26

**Decision:** MushroomRL `Environment` subclass for all five Phase VII
envs (matching Phase I–VI precedent and the existing runner / logging
stack). Implementation must handle the `(state, info)` reset convention
and use `int(np.asarray(x).flat[0])` for numpy state-scalar
normalization. Plain Python classes are allowed only as internal helpers
(e.g. opponent controllers), not as the primary env API.

### 22.3 Definition of `wrong_sign` per environment — RESOLVED 2026-04-26

**Decision:** `wrong_sign` is only defined where the env has a canonical
sign:

- `delayed_chain`: canonical +β; `wrong_sign` = −β.
- `hazard_gridworld`: canonical −β; `wrong_sign` = +β.
- `rps`, `switching_bandit`, `selfplay_rps`: **no canonical sign** — do
  not define `wrong_sign`; substitute `fixed_positive` and
  `fixed_negative` for those envs.

The schedule object for `wrong_sign` must raise on construction when
instantiated against an env without a canonical sign. See §5 method table.

### 22.4 Self-play RPS in Stage A — RESOLVED 2026-04-26

**Decision:** **excluded from Stage A.** Self-play RPS enters Stage B
fresh, only if at least one Stage-A environment (`rps`,
`switching_bandit`, `hazard_gridworld`, `delayed_chain`) clears the
Stage-A → Stage-B promotion bar with stable adaptive-β signal and
controlled variance. Self-play does not contribute to the Stage-A → B
promotion decision and is treated as a secondary stress test. Stage A
matrix is therefore 4 envs × 5 methods × 3 seeds = 60 runs (§8.1).

### 22.5 Bandit advantage signal at horizon = 1 — RESOLVED 2026-04-26

**Decision:** the mechanism story is **degenerate at H = 1** for the
switching bandit. The bandit is **kept as a performance benchmark only**
— it contributes to `regret`, `auc_return`, `recovery_time`, and
`final_return` claims, and is **excluded from**:

- `alignment_rate_{env}.pdf` and `effective_discount_{env}.pdf` figures.
- `Align Rate` and `Mean d_eff` columns of the main results table (those
  cells are reported as `n/a — degenerate at H=1`).

See §6.2 and §10 for propagation.

---

## 23. Document changelog

- 2026-04-26 — initial spec written by `planner` from
  `tasks/adaptive_beta_experiments_coding_agent_spec.md` plus user
  decisions of 2026-04-26 (phase number = VII, paper role = exploratory,
  staged compute, hybrid code home, light/full gate cadence, no-clip
  honesty rule).
- 2026-04-26 — §22 Open questions resolved by user
  (`tasks/phase_VII_clarifications.md` + `tasks/phase_VII_confirmation.md`):
  §22.1 → option (b) shared kernel at `src/lse_rl/operator/tab_operator.py`
  with `SafeWeightedCommon` rewired to import from it; §22.2 → MushroomRL
  `Environment` subclass; §22.3 → `wrong_sign` defined only on
  `delayed_chain` (+→−) and `hazard_gridworld` (−→+); §22.4 → self-play
  excluded from Stage A, fresh entry to Stage B gated on a Stage-A signal
  on the four primary envs; §22.5 → switching bandit excluded from
  alignment-rate / `d_eff` mechanism panels and from mechanism columns of
  the main results table (kept for performance metrics only). §3.4, §5,
  §6.2, §6.4, §6.5, §8.1, §10.1, §10.3 updated to match.
