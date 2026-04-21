# tasks/lessons.md — Corrections and prevention rules

Append-only log of lessons learned from user corrections or self-found
mistakes. Each entry follows the template below.

---

## Template

```
### YYYY-MM-DD — <short title>

**Pattern**: <the failure mode, described generally>

**Prevention rule**: <the concrete rule to follow next time>

**Source incident**: <one-line link to the conversation / commit / spec>
```

---

## Entries

### 2026-04-16 — torch dependency: MushroomRL hard-imports torch at package init

**Pattern**: Assumed `mushroom_rl` was importable after `pip install -e .` on the
`lse_rl` package. In fact `mushroom_rl/core/array_backend.py` imports torch at init
time, so any `import mushroom_rl` fails if torch is absent — even for tabular/CPU-only
MushroomRL usage. Caused task 12 smoke run to block.

**Prevention rule**: Before any task that imports `mushroom_rl`, verify the active
Python env has torch installed. The canonical env for this repo is `.venv/` at the
repo root (created 2026-04-16). All runner scripts, subagents, and test commands
must use `.venv/bin/python` (or activate the venv). Never assume system Python has
the full stack.

**Source incident**: Task 12 smoke run — subagent reported `ModuleNotFoundError: torch`
from `/usr/local/bin/python3`; user directed creation of `.venv` with torch==2.6.0
(torch==2.5.1 has no cp313 wheels; 2.6.0 is the correct pin for Python 3.13).

---

### 2026-04-16 — MushroomRL Dataset API: use .reward not .rewards

**Pattern**: `examples/grid_world_td.py` line 52 calls `collect_dataset.get().rewards`
(plural). Current vendored MushroomRL API is `.reward` (singular). Any code that reads
reward arrays from a collected `Dataset` must use `.reward`.

**Prevention rule**: When writing callbacks or post-processing that reads from a
`Dataset` object, always use `.reward` (singular). If unsure, use
`getattr(ds, 'reward', getattr(ds, 'rewards', None))` as a defensive fallback.
Do NOT edit `mushroom-rl-dev/` source to fix upstream examples — per CLAUDE.md §4.

**Source incident**: Task 13 smoke run — wrapper needed `getattr` fallback because
the upstream example used the old `.rewards` API.

---

### 2026-04-16 — margin_beta0 formula: do NOT include gamma

**Pattern**: When defining the "margin" quantity `reward - v_next`, the agent added
`gamma` to match standard TD-target form, writing `reward - gamma * v_next_beta0`.
This is wrong. The TD target and the margin are two distinct objects with different
mathematical roles.

**Prevention rule**:
- `margin_beta0 = reward - v_next_beta0`  (no gamma — drives responsibility/allocation geometry)
- `td_target_beta0 = reward + gamma * v_next_beta0`  (standard Bellman — drives value update)
The responsibility operator depends on `beta * (r - v)`, not `beta * (r - gamma*v)`.
Any time "margin" or "allocation" appears, check it uses the **no-gamma** form.

**Source incident**: Task 11 (schemas.py) — formula written with gamma, corrected by user
citing rho*(r,v) = sigma(beta*(r-v) + log(1/gamma)) from the paper.

---

### 2026-04-16 — chain_base action convention slip

**Pattern**: The resolved defaults said "always-right (action index 1)" for the chain_base reference policy, but MushroomRL's `generate_simple_chain` uses action 0 = right, action 1 = left. Trusting English descriptions over code led to an inverted reference policy.

**Prevention rule**: When specifying action conventions for any MushroomRL environment, always cite the generator source code (e.g., `generate_simple_chain` transition matrix layout), not English descriptions. Verify by inspecting P[s, a, s'] for a known action.

**Source incident**: Task 28 (chain_base factory) — reference policy corrected from all-ones to all-zeros after checking MushroomRL source.

---

### 2026-04-16 — MushroomRL 2.x reset API returns (state, info)

**Pattern**: Code assumed `env.reset()` returns a bare state array. MushroomRL 2.x (following Gymnasium convention) returns a `(state, info)` tuple. Any callback or evaluator that calls `env.reset()` and passes the result directly as a state will feed a tuple where an array is expected.

**Prevention rule**: Always unpack as `state, _ = env.reset()` in any RL callback, evaluator, or test code that calls `reset()`. Never assign the raw return value as a state.

**Source incident**: Task 37 (RLEvaluator) — evaluator crashed on first eval episode because it passed a tuple to `draw_action`.

---

### 2026-04-16 — ClassicalValueIteration reads gamma from mdp.info, not constructor

**Pattern**: Briefing notes told the algo-implementer that gamma would be passed as a constructor kwarg. In fact, the MushroomRL planner pattern reads gamma from `mdp.info.gamma` (set on the MDPInfo object at environment creation time). Passing gamma to the constructor caused a TypeError.

**Prevention rule**: Before briefing any agent on a MushroomRL class interface, check the actual constructor signature. For planners, gamma always comes from `mdp.info.gamma`.

**Source incident**: Task 21 (ClassicalValueIteration) — constructor signature mismatch caught during unit tests.

---

### 2026-04-16 — Q shape is (T, S, A), not (T+1, S, A)

**Pattern**: Initial implementation allocated Q with shape (T+1, S, A) to "match" V's shape (T+1, S). This is wrong: Q is defined for decision stages 0..T-1 only (T stages), while V includes the terminal boundary V[T] = 0 (T+1 stages). The extra Q[T,:,:] row is meaningless and causes off-by-one bugs in downstream consumers.

**Prevention rule**: Q.shape = (T, S, A) for decision stages 0..T-1. V.shape = (T+1, S) including terminal boundary V[T] = 0. Never assume Q and V have the same first dimension.

**Source incident**: Task 19 (finite_horizon_dp_utils) — shape mismatch caught when PE tried to index Q[T,:,:] and got inconsistent results.

---

### 2026-04-16 — `build_calibration_stats_from_dp_tables`: validate Q as (H, S, A), not (H+1, S, A)

**Pattern**: A stagewise calibration helper duplicated the mistake of treating Q as having the same leading dimension as V. The function validated `Q.shape == (horizon + 1, S, A)` while all finite-horizon planners expose `planner.Q` with shape `(T, S, A)` for stages `0..T-1` only. That made `run_phase1_dp` raise `ValueError` on every run after `planner.run()`.

**Prevention rule**: Any code that consumes **exact DP outputs** from `mushroom_rl.algorithms.value.dp` must treat **`Q` as `(H, S, A)`** and **`V` as `(H+1, S)`**. Re-read the existing lesson on Q vs V shapes before writing shape checks. Add or extend a unit test that calls the helper with the **same arrays a planner returns** (not hand-shaped tensors).

**Source incident**: Phase I review — `experiments/weighted_lse_dp/common/calibration.py` `build_calibration_stats_from_dp_tables`; fixed 2026-04-16; regression test `test_build_calibration_stats_from_dp_tables_accepts_planner_q_shape` in `tests/algorithms/test_phase1_calibration_logging.py`.

---

### 2026-04-16 — Phase I DP runner: iteration count is `len(residuals)`, not `n_sweeps`

**Pattern**: `run_phase1_dp` used `planner.n_sweeps` to drive the replay loop into `DPCurvesLogger`. **Policy iteration** and **modified policy iteration** expose **`n_iters`** (and may not define `n_sweeps` at all), so the runner could **`AttributeError`** or use the wrong count. Tabular VI / Async VI / PE use `n_sweeps`.

**Prevention rule**: For DP runners that iterate over “one entry per `planner.residuals`”, set **`n = len(planner.residuals)`** (or assert it matches `n_sweeps` or `n_iters` depending on algorithm). Do not assume a single attribute name across all planner classes.

**Source incident**: Phase I review — `experiments/weighted_lse_dp/runners/run_phase1_dp.py`; fixed 2026-04-16 (`n_sweeps_actual = len(residuals)`).

---

### 2026-04-16 — Multi-sweep DP curves: store and pass `V` after each sweep, not the final `V` every time

**Pattern**: After `planner.run()`, a naive replay loop called `record_sweep(..., v_current=planner.V)` for **every** sweep index. For multi-iteration planners (PI, MPI, multi-pass VI/AsyncVI), **`supnorm_to_exact`** and **per-sweep value snapshots** (e.g. chain propagation plots) were wrong: they used only the **final** value table.

**Prevention rule**: Finite-horizon DP planners that log **`residuals[i]`** per sweep/iteration should also populate **`V_sweep_history[i]`** = copy of `V` **after** that sweep (same length as `residuals`). Runners must pass **`V_sweep_history[i]`** into curve loggers when `i < len(V_sweep_history)`, else fall back to `planner.V`. Document this in any callback that takes “current V” per sweep.

**Source incident**: Phase I review — `mushroom_rl/.../dp/*.py` (`V_sweep_history` on PE, VI, PI, MPI, AsyncVI); `run_phase1_dp._v_for_sweep_index`; `experiments/weighted_lse_dp/common/callbacks.py` (`DPCurvesLogger` docstring); tests `TestVSweepHistory` in `tests/algorithms/test_classical_finite_horizon_dp.py`.

---

### 2026-04-16 — Import path typo: slash instead of dot in MushroomRL DP subpackage

**Pattern**: A subagent wrote `from mushroom_rl.algorithms/value.dp.finite_horizon_dp_utils import ...` (forward slash instead of dot) in `classical_value_iteration.py`. Python accepted this silently at write time but raises `SyntaxError` at import, blocking the entire DP subpackage and all downstream tests.

**Prevention rule**: After any agent writes or edits a `from X import Y` statement in `mushroom-rl-dev/`, verify the import path uses only dots as separators. Run `.venv/bin/python -c "import mushroom_rl.algorithms.value.dp"` as a quick sanity check before committing. Edit justified under CLAUDE.md §4: single-character typo fix in vendored code, isolated to one line.

**Source incident**: Phase I closing branch — `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/classical_value_iteration.py` line 44; caught by verifier on `phase-I/closing`; fixed 2026-04-16.

---

### 2026-04-17 — Default out-root constants that duplicate RunWriter's path structure

**Pattern**: Runner scripts set `_DEFAULT_OUT_ROOT` to a path that already includes the `phase/suite` directory levels (e.g. `results/weighted_lse_dp/phase1/paper_suite`). RunWriter.create then appends `phase/suite/task/algo/seed_*` again, producing a double-nested path that no aggregator or validator discovers. The same pattern appeared in both the RL runner (double-nesting `phase1/paper_suite`) and the ablation runner (double-nesting `phase1/<gamma_dir>`).

**Prevention rule**: Default out-root in any runner must be the bare results root (e.g. `results/weighted_lse_dp`). Let RunWriter (or whatever path-constructing helper the runner uses) be the single source of truth for the `phase/suite/task/algo/seed_*` hierarchy. Never embed phase or suite segments in the default out-root constant. When adding a new runner, verify the produced path by printing `rw.run_dir` in dry-run mode and confirming it matches the glob pattern in `find_run_dirs`.

**Source incident**: Phase I R2 Codex review — `run_phase1_rl.py:75` (`_DEFAULT_OUT_ROOT`), `run_phase1_ablation.py:189-227` (ablation path construction); both caused BLOCKERs in triage 2026-04-17.

---

### 2026-04-17 — FiniteMDP absorbing state with mu=None causes uniform init crash

**Pattern**: When a task factory adds an extra absorbing terminal state (e.g., jackpot or catastrophe terminus at `state_n + 1`), the resulting `P` array has shape `(state_n+1, A, state_n+1)` with an all-zero row for the absorbing state. Passing `mu=None` to `FiniteMDP` triggers uniform initialization over all states including the zero-P absorbing row. On the next `env.reset()`, MushroomRL samples a state from the discrete distribution defined by `mu`, which may draw the absorbing state, and then attempts a step from a state with zero transition probability — crashing with `ValueError: probabilities do not sum to 1`. This manifests only when `severity > 0` (absorbing state added); `severity=0` avoids the extra state and passes silently.

**Prevention rule**: Whenever a task factory adds absorbing states that are unreachable from the initial distribution, always compute `mu` explicitly: `mu = np.zeros(P.shape[0]); mu[initial_state] = 1.0`. Never pass `mu=None` to `FiniteMDP` when the state space has been extended beyond the base MDP. The guard is: `if P.shape[0] > base_state_n: mu = ...; else: mu = None`.

**Source incident**: Phase II task 20 degradation tests — `make_chain_jackpot` and `make_chain_catastrophe` both passed `mu=None` with the absorbing state present; crash caught by `test_phase2_classical_degradation.py`; fixed 2026-04-17 in `stress_families.py`.

---

### 2026-04-17 — Ablation runner must delegate to same-phase runners, not Phase I runners

**Pattern**: A phase-N ablation runner was written to delegate to Phase I task factory runners (`run_phase1_dp._run_single`, `run_phase1_rl.run_single`). Those runners only know Phase I task names (`chain_base`, `grid_base`, `taxi_base`). Phase II task names (`chain_jackpot`, `grid_hazard`, etc.) are unknown to Phase I dispatch tables, so any Phase II ablation run fails at task-factory lookup with a `KeyError` or dispatches to the wrong factory. The bug is silent at import time — it only surfaces when a Phase II task name is passed.

**Prevention rule**: Every phase-N ablation runner must import and delegate to the phase-N DP and RL runners, not earlier-phase runners. When writing `run_phase{N}_ablation.py`, immediately verify the import statement reads `from run_phase{N}_dp import _run_single as dp_run_single` (not `run_phase{N-1}_dp`). Gamma injection in ablation must use `task_cfg_gp = dict(task_cfg); task_cfg_gp["gamma"] = gp` rather than a separate kwarg, since the phase-N runner signature may differ from phase N-1.

**Source incident**: Phase II task 24 — initial `run_phase2_ablation.py` imported `run_phase1_dp._run_single`; caught during implementation review; corrected 2026-04-17 to import from `run_phase2_dp`.

---

### 2026-04-17 — Wrapper-based stress environments not wired through the training loop

**Pattern**: Task factories that return `(wrapper, mdp_rl, cfg)` build `mdp_rl` from `mdp_base` (the unwrapped MDP), not from the wrapper. When the runner passes `mdp_rl` to `Core` for RL training or strips the wrapper via `._base` for DP planning, the stress dynamics (hazard penalties, regime shifts, bonus shocks) never fire. The agent trains on the base task under the stress task's name. This affected 4 of 8 RL task families and 2 of 8 DP task families in Phase II, producing invalid results that passed all structural/schema validators because the output format was correct -- only the content was wrong.

**Prevention rule**: (1) For RL: when a stress task uses a wrapper, the factory must time-augment the wrapper itself (not `mdp_base`) and the runner must pass the augmented wrapper to `Core`. (2) For DP: if the wrapper's stress cannot be encoded in `(P, R)`, the task must not appear in the DP suite. If it can be encoded, build a stressed `(P, R)` model directly rather than wrapping and then unwrapping. (3) Add a per-wrapper integration test that runs a short episode and asserts the stress event fires at least once (non-zero event count in transitions). This catches the "wrapper never stepped" failure mode that structural validators miss.

**Source incident**: Phase II Codex reviews (byl14ca5j + bqm6fkd15) — `run_phase2_rl.py:607` passes `mdp_rl` (base) to Core; `run_phase2_dp.py:223-224` strips wrapper via `._base`; triaged 2026-04-17.

---

### 2026-04-17 — Figure scripts written against assumed JSON schema without contract testing

**Pattern**: The Phase II figure script (`make_phase2_figures.py`) was developed alongside a `--demo` mode that synthesizes data and bypasses all JSON/NPZ reads. The production code path reads keys (`summary["curves"]`, `cal["base_returns"]`, `cal["stress_returns"]`) that do not exist in the actual aggregator output. Because `--demo` mode was the only path exercised during development and review, the schema mismatch was invisible until the R2 Codex review. Result: 3 of 5 mandatory Phase II figures produce blank or placeholder-text panels in production mode.

**Prevention rule**: (1) Every figure script that reads structured data must have at least one integration test that runs it against a minimal but structurally correct data fixture -- NOT the `--demo` synthetic path. The test must assert non-empty plotted output and zero "No data" / "Empty" / placeholder text annotations. (2) Define output schemas (summary.json keys, calibration JSON keys) as shared constants or TypedDicts between the aggregator and the figure script, so a key rename in the aggregator causes an immediate import-time or type-check failure in the figure script. (3) Never ship a figure script whose production path has only been exercised via `--demo`.

**Source incident**: Phase II Codex R2 reviews (b0nweghxr + bewezccsc) — `make_phase2_figures.py:299-305` reads `summary["curves"]` (non-existent); lines 373-374 read `cal["base_returns"]` and `cal["stress_returns"]` (non-existent). Triaged 2026-04-17 as 2 BLOCKERs.

---

### 2026-04-17 — Config-dict override keys passed but never consumed by callee

**Pattern**: The ablation runner (`run_phase2_ablation.py`) carefully constructs `task_config` entries with `epsilon_override` and `lr_multiplier` keys and passes them to `run_phase2_rl.run_single()`. But `run_single()` never reads those keys -- it hard-codes `_EPSILON` and `_LEARNING_RATE` at lines 566-567 and calls `_make_agent()` without forwarding overrides. The resolved config even records the defaults as the "effective" values, making the bug invisible in logs. The entire hyperparameter ablation sweep produces mislabeled duplicate results.

**Prevention rule**: (1) When a caller passes override values through a dict to a callee, the callee must explicitly extract and use those overrides -- add a comment or assertion at the callee showing which keys it consumes. (2) Add an integration test that calls the callee with non-default overrides and asserts the downstream object (agent, optimizer) reflects the overridden values, not the module-level defaults. (3) When recording resolved config, derive values from the same code path that constructs the agent, not from separate constant references.

**Source incident**: Phase II Codex R3 standard review (review-mo2thz1w-kohkfg) — `run_phase2_rl.py:566-567,592` ignores `task_config["epsilon_override"]` and `task_config["lr_multiplier"]`. Triaged 2026-04-17 as BLOCKER R3-1.

---

### 2026-04-18 -- Grouping-key fix that replaces the discriminating key instead of adding metadata

**Pattern**: A review finding (R5-3) asked for family-level calibration grouping so that regime-shift pre/post runs would appear under a single family-level calibration document. The fix replaced the `task` grouping key with `canonical_task_family` in `_discover_runs`, which collapsed pre_shift and post_shift DP runs into the same aggregate group. Their calibration statistics were then averaged together, corrupting the post-change signal that Phase III depends on. The correct approach was to keep `task` as the discriminating key (preserving separate groups) and add `canonical_task_family` as a metadata field for downstream use.

**Prevention rule**: When a review asks for "family-level grouping" or "merge X into Y", distinguish between (a) adding a metadata/tag field that downstream consumers can use to select or combine groups, and (b) replacing the primary grouping key. If the items being "grouped" have semantically different data that must not be averaged (e.g. pre-change vs post-change statistics), the fix must be (a), not (b). Always add a test that asserts the number of groups produced by `_discover_runs` matches the expected count for a regime-shift task family (i.e. 2 groups, not 1).

**Source incident**: Phase II Codex R6 adversarial review (019d9e19-c138) -- `aggregate_phase2.py:208-221` regression from R5-3 fix. Triaged 2026-04-18 as BLOCKER R6-1.

---

### 2026-04-17 — Aggregation statistics computed from summaries-of-summaries instead of raw data

**Pattern**: Calibration statistics meant to capture distributional properties (quantiles, extremes) were computed from already-aggregated arrays (per-seed means, per-stage means) rather than from the underlying raw data. This appeared in two forms in Phase II R4: (1) margin quantiles computed from `aligned_positive_mean` arrays (percentiles of means, not quantiles of the margin distribution), and (2) `empirical_r_max` derived from `reward_mean + 2*reward_std` instead of `max(|r|)` over actual observations. Both silently produce plausible-looking numbers that are systematically biased, especially for rare-event / heavy-tail families where the whole point is tail behavior.

**Prevention rule**: (1) Any statistic that claims to represent a distributional property (quantiles, extremes, CVaR) must be computed from the most granular data available (per-transition, per-episode), never from pre-averaged summaries. (2) Name arrays precisely: `aligned_positive_mean` is a mean, not a sample -- code that treats it as a sample for percentile computation is wrong by construction. (3) Add a unit test that verifies calibration quantiles against a known synthetic distribution with fat tails; the test should fail if quantiles are computed from stage means instead of raw samples.

**Source incident**: Phase II Codex R4 adversarial review (review-mo3l48hq-1ncnzo) — `aggregate_phase2.py:721-776` (margin quantiles from means) and `aggregate_phase2.py:778-792` (r_max from moments). Triaged 2026-04-17 as BLOCKERs R4-3 and R4-4.

---

### 2026-04-17 — Hard-coded state-count constants not updated when task parameters change

**Pattern**: `_N_BASE` in `run_phase2_rl.py` maps task names to base state counts used by `TransitionLogger` for index decomposition (`aug_id // n_base`, `aug_id % n_base`). When `grid_sparse_goal` was changed from 5x5 (25 states) to 7x7 (49 states), the `_N_BASE` entry was not updated. Every transition with `aug_id >= 25` produces wrong timestep and state values, corrupting all downstream transition logs, calibration stats, and visitation heatmaps for this task. The bug is silent: no assertion fires, shapes are correct, values are plausible but wrong.

**Prevention rule**: (1) Never hard-code derived constants (like state counts) separately from the factory that produces the environment. Either compute `n_base` from the environment object at runtime (`mdp.info.observation_space.size`), or have the factory return it alongside the MDP. (2) If a hard-coded lookup table must exist, add an assertion at task creation time that `_N_BASE[task] == mdp.info.observation_space.size`. (3) When changing any task parameter that affects state-space size, grep for all hard-coded references to the old size.

**Source incident**: Phase II Codex R8 reviews (019d9e60) -- `run_phase2_rl.py:122` has `"grid_sparse_goal": 25` for a 49-state MDP. Triaged 2026-04-17 as BLOCKER R8-1.

---

### 2026-04-17 — Event detection thresholds derived from absent config keys (silent default)

**Pattern**: `run_phase2_rl.py:623` reads `task_config.get("jackpot_reward", 10.0)` for the taxi_bonus_shock task, but this config has no `jackpot_reward` field (it has `bonus_reward`). The threshold silently defaults to `10.0 * 0.5 = 5.0`, which happens to work when `bonus_reward=5.0` (total delivery = 6.0 > 5.0). If `bonus_reward` is lowered to 4.0 or below, jackpot events are silently never logged. The `.get()` with a default masks the missing key.

**Prevention rule**: (1) When deriving event detection thresholds from config, use explicit keys that are guaranteed to exist in the task config. If the key might not exist, fail loudly (KeyError) rather than defaulting silently. (2) Add a startup assertion that all required config keys for event detection are present: `assert "bonus_reward" in task_config` before deriving thresholds. (3) Log derived thresholds to `run.json` so post-hoc auditing can catch threshold mismatches.

**Source incident**: Phase II Codex R8 reviews (019d9e60) -- `run_phase2_rl.py:623` uses absent `jackpot_reward` key with silent default. Triaged 2026-04-17 as MAJOR R8-4.


---

### 2026-04-18 — MushroomRL callback_step fires before agent.fit: reading agent state in callback_step yields stale values

**Pattern**: `SafeTransitionLogger.__call__` (the callback_step hook) read `agent.swc.last_*` diagnostics to log safe fields alongside the current transition sample. But MushroomRL `Core._run()` calls `callback_step(sample)` at line 110 *before* `agent.fit(dataset)` at line 116. With `n_steps_per_fit=1`, every row in the safe transition log contains the *previous* update's safe diagnostics paired with the *current* transition's reward/state. The first row contains zero-initialized defaults. The bug is silent: shapes are correct, values are plausible, but they belong to different transitions. This corrupts all Phase III per-transition observability and can hide certification violations.

**Prevention rule**: (1) Never read agent internals (mixin state, last-computed quantities) from `callback_step` — they reflect the previous fit call, not the current sample. (2) Use `callbacks_fit` (the post-fit hook, fired via `Core._run()` after `agent.fit()`) for any logging that depends on the agent's computed values for the current transition. (3) Concretely: split the logger into two methods — `__call__` (callback_step) logs only fields derivable from the raw sample (state, action, reward, next_state), and `after_fit(dataset)` (callbacks_fit) logs agent-computed diagnostics. (4) Add a smoke test that asserts `safe_stage[k]` matches the stage encoded in `aug_state[k]` for every transition k.

**Source incident**: Phase III Codex R1 standard review (019da11e, P1) + adversarial review (019da120, high) — `callbacks.py:262-285` and `run_phase3_rl.py:338-360`. Fixed in commit 11f2189.

---

### 2026-04-18 — Certification R_max must be the configured absolute reward bound, not the empirical sample maximum

**Pattern**: `build_schedule_from_phase12.py` set `R_max = float(cal["empirical_r_max"])` — the largest reward observed during Phase I/II calibration runs. The Phase III spec (S2.2, S5.9) requires `R_max` to be the configured absolute maximum reward for the task. If rare jackpot or catastrophe events were not observed during the finite-sample calibration runs, `empirical_r_max` underestimates the true bound, which underestimates `Bhat_t` (the certification-box radius) and produces a `beta_cap_t` that is too loose. The schedule JSON then advertises certified invariants that can be violated in deployment.

**Prevention rule**: (1) `R_max` for certification must come from task configuration (the field that governs the reward generator), not from aggregate statistics of observed rewards. (2) Add `"reward_bound"` to every task-family config entry and fail loudly (raised exception or at minimum a `warnings.warn`) if only empirical data is available. (3) The schedule builder should log the source of `R_max` (`"configured"` vs `"empirical_fallback"`) in the emitted `schedule.json` so downstream audits can detect unsafe certification provenance.

**Source incident**: Phase III Codex R1 adversarial review (019da120, high) — `build_schedule_from_phase12.py:116-119`. Fixed in commit 11f2189 by adding `reward_bound` parameter and populating it in `paper_suite.json`.

---

### 2026-04-18 — Schedule/MDP parameter mismatch (horizon, gamma) must be enforced at planner construction time

**Pattern**: `SafeWeightedValueIteration` accepted any `BetaSchedule` and silently started indexing it during backward sweeps. If `schedule.T != mdp_horizon`, the planner either threw an off-by-one index error mid-run or silently ignored extra stages. If `schedule.gamma != mdp_gamma`, the certification coefficients (`kappa_t`, `Bhat_t`, `beta_cap_t`) were calibrated for a different discount and the invariant guarantees failed silently. `SafeWeightedPolicyIteration` already had the horizon guard; the other four planners did not.

**Prevention rule**: (1) Every safe planner that consumes a `BetaSchedule` must validate at construction time (in `__init__`): `schedule.T == self._T` and `abs(schedule.gamma - self._gamma) <= 1e-9`. Raise `ValueError` immediately — do not defer to runtime. (2) Centralize this validation in `SafeWeightedCommon.__init__` so all subclasses inherit it automatically, rather than duplicating guards in each planner. (3) When adding a new planner, add a test that passes a schedule with wrong T and wrong gamma and asserts `ValueError` is raised before `run()` is called.

**Source incident**: Phase III Codex R1 standard review (019da11e, P2) + adversarial review (019da120, medium) — `safe_weighted_value_iteration.py:169-188`. Fixed in commit 11f2189 with horizon check added to VI/PE/MPI/AsyncVI and gamma check added to `SafeWeightedCommon.__init__`.

---

### 2026-04-18 — Aggregation call sites written against assumed helper signatures without verification

**Pattern**: `aggregate_phase3.py` called `save_json(data, path)` (reversed args), `save_npz_with_schema(arrays, path)` (missing schema arg), and `aggregate_safe_stats(stages, values, n_stages, quantiles=...)` (completely wrong signature -- actual is `(payload, T, gamma)`). All three calls were written to match an imagined API rather than the actual function signatures in `common/io.py` and `common/schemas.py`. The errors are invisible at import time and only surface at runtime when the aggregation path is actually exercised.

**Prevention rule**: (1) Before writing a call to any helper function, read the callee's signature (at minimum the `def` line and parameter docstring). Never write calls from memory or assumed convention. (2) After writing aggregation code, add a minimal integration test that exercises the write path with a small synthetic dataset and asserts the output files are loadable. (3) For IO helpers, adopt a consistent convention (path-first or data-first) and document it in the module docstring. When wrapping third-party conventions, name parameters explicitly at call sites (`save_json(path=..., data=...)`).

**Source incident**: Phase III Codex R2 standard review (019da19f, P1+P2) -- `aggregate_phase3.py:281-287` (reversed args) and `:461-463` (wrong API). Triaged 2026-04-18 as BLOCKERs R2-1 and R2-2.

---

### 2026-04-18 — SafePE convergence reference must be V^π (PE fixed point), not V* (VI optimal)

**Pattern**: `SafeWeightedPolicyEvaluation` validated convergence by comparing iterates against `V_star` (the value of the optimal policy, computed by VI). A PE run that correctly converges to `V^π` of a suboptimal policy would therefore never satisfy the stopping criterion, causing PE to either run for max iterations or raise a false convergence failure. Worse, for tasks where the reference policy is far from optimal, the reported Bellman residuals were meaningless (they measured deviation from V*, not from the fixed point).

**Prevention rule**: Safe PE algorithms must converge to V^π (the unique fixed point of the policy-specific Bellman operator). The convergence reference is computed by running the same PE operator to numerical precision, NOT by calling VI. Any test that validates safe PE convergence must use a reference computed by the PE operator itself (e.g., run PE with a very small tolerance), never from VI or the optimal value function.

**Source incident**: Phase III Codex R3 standard + adversarial reviews — `safe_weighted_pe.py` convergence check compared against `V_star`. Fixed 2026-04-18: reference is now own-operator fixed point.

---

### 2026-04-18 — expm1/log1p negative-tail underflow: use _EPS_BETA threshold + logaddexp instead

**Pattern**: A "numerically stable" reformulation of the safe operator used `expm1` and `log1p` to avoid cancellation in `log(1 + x)` for small x. For large negative arguments (e.g. r=v=-40, β=1), `expm1(β*(r-v) + log_γ)` evaluates to `expm1(-∞) = -1.0` exactly in float64, and `log1p(-1.0) = -inf`. The resulting safe target is `-inf` instead of the correct ~-79.6. This is a silent correctness failure: no exception is raised, the shape is correct, only the value is wrong.

**Prevention rule**: (1) Do NOT use `expm1/log1p` in the safe operator implementation. Use a clean two-path dispatch: if `|β| < _EPS_BETA` (set to `1e-8`), use classical `r + γv`; else use `logaddexp(β*r, β*v + log_γ)` scaled by `(1+γ)/β`. `logaddexp` is numerically stable for all finite arguments at any `|β| ≥ 1e-8`. (2) Add a unit test that evaluates the operator at r=v=-40, β=1 and asserts the result matches the closed-form value (≈ -79.6). (3) Never use the threshold `_EPS_BETA` as the stability boundary for expm1 — the correct guard is `|β| < ε`, not `|x| < 500`.

**Source incident**: Phase III Codex R3 adversarial + R5 adversarial reviews — `safe_weighted_common.py` expm1/log1p branch. Fixed 2026-04-18: removed expm1/log1p entirely, replaced with `_EPS_BETA=1e-8` two-path dispatch.

---

### 2026-04-18 — numpy ≥2.0: int(state) TypeError on shape-(1,) arrays in MushroomRL TD agents

**Pattern**: MushroomRL `TD.fit()` passes `state` as `np.ndarray` of shape `(1,)` from `dataset.item()`. In numpy ≥2.0, `int(arr)` raises `TypeError` for non-0-dimensional arrays (numpy 1.x allowed it). Any safe TD `_update()` method that calls `int(state)` to extract the augmented state integer fails at the very first update step with a `TypeError`. The bug is absent in numpy 1.x and invisible in unit tests that pass plain Python ints.

**Prevention rule**: Always extract MushroomRL state/action integers as `int(np.asarray(x).flat[0])`, never `int(x)`. This is safe for scalars, 0-d arrays, and shape-(1,) arrays. Apply to every cast in `_update()`, `_stage_from_state()`, and any MushroomRL TD helper. When adding a new safe TD algorithm, add a smoke test that passes a numpy array (not a Python int) as state and asserts no TypeError.

**Source incident**: Phase III RL main runs (runtime failure) — `safe_q_learning.py`, `safe_expected_sarsa.py`, `safe_td0.py`, `safe_weighted_lse_base.py`. Fixed 2026-04-18. Memory record: `feedback_numpy_state.md`.

---

### 2026-04-18 — Ablation schedule T must be regenerated from the current schedule.json, not copied from an earlier one

**Pattern**: When `generate_ablation_schedules.py` was first run, it read T from a set of schedule.json files that predated the final Phase III horizon decisions. After those main schedules were updated (e.g., chain_sparse_long T=80→120), the ablation schedules retained the old T. Five of eight task families had T mismatches. Ablation runs using the wrong T either index out of bounds at stage T_wrong or silently use a truncated/extended schedule, producing runs that are structurally valid but scientifically invalid.

**Prevention rule**: (1) Always regenerate ablation schedules *after* finalising main schedule.json files — never copy ablation schedules from an earlier draft. (2) Ablation schedule generation scripts should read T directly from the main schedule.json rather than accepting it as a parameter. (3) Add a validation step at ablation runner startup that asserts `ablation_schedule.T == main_schedule.T` for the corresponding task family, failing loudly if mismatched.

**Source incident**: Phase III ablation runs (runtime) — 5 of 8 task families had wrong T in ablation schedules. Fixed 2026-04-18 by rerunning `generate_ablation_schedules.py` from current schedule.json files.

---

### 2026-04-18 — DP rho is derivable from effective_discount; do not leave it as all-NaN

**Pattern**: `run_phase3_dp.py` logged `safe_rho_mean` and `safe_rho_std` filled with `np.full(T, np.nan)` because DP runs have no per-transition rho readout (unlike online RL, which reads `swc.last_rho` per step). The arrays were included in the schema for completeness but never populated. Downstream table generators and verifiers reported NaN in rho columns for all DP runs, making cross-modality comparisons (DP vs RL rho distributions) impossible.

**Prevention rule**: For DP runs, rho is exactly derivable from effective_discount via `ρ = 1 − eff_d/(1+γ)` (a linear transformation with no approximation). Always compute `rho_mean = 1 - eff_discount_mean / (1+γ)` and `rho_std = eff_discount_std / (1+γ)` and write these to the output — do not fill with NaN. Add an assertion in the DP aggregator that `safe_rho_mean` is not all-NaN for any DP run.

**Source incident**: Phase III DP runs (runtime) — `run_phase3_dp.py` rho columns all-NaN. Fixed 2026-04-18 using `ρ = 1 − eff_d/(1+γ)` derivation.

---

### 2026-04-18 — safe_margin must read swc.last_margin, not v_next_beta0 (greedy max)

**Pattern**: `SafeTransitionLogger.after_fit` computed `safe_margin` as `reward - v_next_beta0`, where `v_next_beta0` was the greedy `max_a Q(s', a)` value. This is the correct formula for Q-learning (`v_next = max Q`), but wrong for SafeExpectedSARSA and SafeTD0, where `v_next` is the policy-weighted expectation. For those algorithms, `safe_margin` was systematically higher than the actual margin used by the operator, making the logged certification bounds look tighter than the deployed ones.

**Prevention rule**: Always read `safe_margin` from `swc.last_margin` — the exact `v_next` value that was passed to `compute_safe_target()`. Never recompute `v_next` outside the operator call and use it for margin logging. The `SafeWeightedCommon` mixin stores `last_margin` precisely for this purpose. Any callback that logs margin, rho, or certification quantities must read them from `swc.last_*` fields, not recompute them independently.

**Source incident**: Phase III Codex R4 adversarial review — `callbacks.py:SafeTransitionLogger.after_fit` using `v_next_beta0` for safe_margin. Fixed 2026-04-18: reads `float(np.asarray(swc.last_margin).item())`.

---

### 2026-04-18 — aggregate() requires per-key 1-D numpy arrays, not a List[Dict]

**Pattern**: `aggregate_phase3.py` accumulated per-seed scalar stats as `List[Dict[str, float]]` and then called `aggregate(per_seed_scalars)` directly, passing the list. The `aggregate()` helper from `common/statistics.py` expects a 1-D numpy array of floats and returns a dict of aggregation statistics. Passing a list of dicts causes a `TypeError` or produces silently nonsensical results (e.g., treating dict keys as array elements). The error only surfaces at runtime when seeds are aggregated.

**Prevention rule**: (1) When aggregating per-seed scalar statistics, collect values per key: `vals = [d[key] for d in per_seed_dicts]`, convert to `np.array(vals, dtype=np.float64)`, then call `aggregate(arr)`. Never pass a list of dicts to `aggregate()`. (2) Add an integration test that passes at least two seed dicts through the aggregator and asserts the output contains `mean`, `std`, and `n` keys for every scalar field. (3) When writing new aggregation code, check the `aggregate()` signature before use — do not assume it accepts dict inputs.

**Source incident**: Phase III RL main runs (runtime) — `aggregate_phase3.py` scalar aggregation path calling `aggregate(List[Dict])`. Fixed 2026-04-18: per-key iteration with `np.array(vals)`.

---

### 2026-04-19 — MushroomRL edit: compute_safe_target_ev_batch (correct stochastic Bellman backup)

**Pattern**: The Phase III DP runners were calling `compute_safe_target_batch(r_bar, E[V(s')])`, i.e., evaluating the nonlinear safe TAB operator at the *expected* next-state value. This is incorrect when beta != 0 and the MDP is stochastic: the correct Bellman backup is `E_{s'}[g_safe(r, V(s'))]` (expectation *after* the nonlinearity). The Phase III code audit (phase3_code_audit.json) flagged `compute_safe_target_ev_batch` as missing. The Phase II/III rerun (job b0ao4zigu) added the method and switched all DP runners to use it.

**Justification for mushroom-rl-dev edit**: The existing `compute_safe_target_batch` signature does not support per-next-state evaluation (it takes scalar `v_next`). Adding `compute_safe_target_ev_batch(r_bar, V_next, p, t)` — which takes the full transition tensor `p` and next-value vector `V_next` — is the minimal change that makes the backup correct. No other MushroomRL algorithm is affected; the new method is additive. The `< _EPS_BETA` → `<= _EPS_BETA` boundary fix is a one-character correctness improvement.

**Prevention rule**: When the safe operator is nonlinear (beta != 0) and transitions are stochastic, always use `compute_safe_target_ev_batch` for DP backups. Never pass `E[V(s')]` to `compute_safe_target_batch` as a substitute — this conflates E[g(r, V)] with g(r, E[V]), which are equal only when g is linear (beta = 0). Document any MushroomRL edit in this file immediately, not retroactively.

**Source incident**: Phase III code audit (results/weighted_lse_dp/phase4/audit/phase3_code_audit.json) — observability_gaps field. Edit applied by Phase II/III rerun 2026-04-19. Committed to phase-iv-a/closing.

---

### 2026-04-20 — Activation pilot must use DP V* (spec §S5.1), not random policy

**Pattern**: `task_activation_search.py` and `run_phase4_counterfactual_replay.py` both used random-policy episodes to collect pilot margins and compute the value proxy (discounted Monte Carlo return). On sparse-reward MDPs, random policy rarely reaches the goal, so nearly all margins are zero (r_t - 0 = 0 or r_t - MC_return ≈ 0). This made xi_ref ≈ 0, u_target ≈ u_min, and the gate metrics 2–3 orders of magnitude below threshold.

**Prevention rule**: Spec §S5.1 explicitly requires "a fresh Phase I/II calibration pilot (classical QL/ESARSA)" or "Phase III safe pilot logs." For tabular MDPs with exact (P, R), use backward VI to compute V* and run epsilon-greedy Q* episodes. Compute margins as r_t - V*(s') (no gamma — per lessons.md 2026-04-16). Always use DP V* when available; random-policy pilots produce zero margins on sparse-reward tasks and violate the spec.

**Source incident**: Phase IV-A overnight run — activation gate FAIL (0/11 → 10/11) after fixing pilot. Gate still fails due to certification A_t blowup; see next entry.

---

### 2026-04-20 — Certification Bhat_t exponential blowup prevents activation for T≥20, γ≥0.95

**Pattern**: The certification recursion `Bhat[t] = kappa_t * (r_max + Bhat[t+1]) / (1 - kappa_t)` grows as ~((kappa/(1-kappa))^T). With kappa=0.96 (from alpha=0.20, gamma=0.95), the ratio is 24. For T=20, Bhat[0] ~ 24^20 ~ 10^26. A_t = r_max + Bhat[t+1] ~ 10^26. Since beta = u_ref_used / (A_t * xi_ref) ≈ 6e-3 / 10^26 ≈ 1e-29, the operator is indistinguishable from classical. The gate threshold mean_abs_u ≥ 5e-3 cannot be met.

**Prevention rule**: Before designing Phase IV-A task configurations, check whether the certification recursion produces finite A_t values. A rule of thumb: kappa^T < 10^6 for tractable activation. With kappa = gamma + alpha*(1-gamma), this means: T * log(kappa) < 6*ln(10). For gamma=0.95 and alpha=0.20 (kappa=0.96): T < 6*ln(10)/ln(1/0.96) = 13.8*25.6 = 354 — so T=20 is fine? NO! The recursion uses the full expansion `kappa/(1-kappa)` which grows much faster. The correct bound is T * log(kappa/(1-kappa)) < 13.8, giving T < 13.8/ln(24) = 4.3 for kappa=0.96. **Maximum safe horizon at gamma=0.95 is T=4.**

**Source incident**: Phase IV-A gate analysis — `mean_abs_u_pred = 0.00356` vs threshold `0.005` after fixing pilot. Use T≤4 for gamma=0.95, or T≤10 for gamma=0.5.
