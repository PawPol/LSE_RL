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
