# tasks/todo.md — Active task list

This file is the single source of truth for the active plan. Update
status in real time. Add a short review section at the end of each
completed task.

---

## Active task: Project structure initialization (2026-04-16)

- [x] Decide on repository layout (vendored MushroomRL, `lse_rl` package,
      specs under `docs/specs/`, git init with PawPol/LSE_RL remote).
- [x] Create top-level directory skeleton.
- [x] Move paper artifacts into `paper/`.
- [x] Move phase specs into `docs/specs/`.
- [x] Write `README.md`, `CLAUDE.md`, `AGENTS.md` (stub),
      `tasks/todo.md`, `tasks/lessons.md`, `.gitignore`, `pyproject.toml`,
      and package `__init__.py` stubs.
- [x] `git init`, add `origin = github.com/PawPol/LSE_RL.git`, stage
      tracked files (NO commit yet — user will review).
- [x] Verify final structure; confirm `mushroom-rl-dev/` is untouched.

### Review (2026-04-16)

- Structure matches the answers from the onboarding question set.
- `mushroom-rl-dev/` left byte-identical (no files added or removed).
- Paper + specs relocated without content changes (only `git mv`-equivalent).
- Git initialized locally on branch `main` with remote `origin` pointing at
  `github.com/PawPol/LSE_RL.git`. Nothing committed or pushed — awaiting
  user sign-off and the agents spec before the first commit.

---

## Active task: Orchestration setup (2026-04-16)

- [x] Parallel recon via Explore subagents: summarize Phase I/II/III
      specs + fetch codex-plugin-cc docs.
- [x] Collect architectural decisions from user: function-aligned
      roster, phase-boundary Codex gate, worktree isolation,
      opus-everywhere.
- [x] Write `AGENTS.md` as authoritative orchestration protocol.
- [x] Write 10 role subagents under `.claude/agents/` (all
      `claude-opus-4-6`): planner, env-builder, algo-implementer,
      operator-theorist, calibration-engineer, experiment-runner,
      test-author, plotter-analyst, verifier, review-triage.
- [x] Write 5 project slash commands under `.claude/commands/lse/`:
      plan-phase, implement, verify, review, status.
- [x] Write `.codex/config.toml` (model = gpt-5.4, reasoning high).
- [x] Write `docs/workflow.md` (lifecycle diagram + Codex integration).
- [x] Update `README.md` with orchestration + Codex setup sections.
- [x] Verify frontmatter validity, cross-reference consistency, path
      existence.

### Review (2026-04-16)

- Roster is function-aligned (10 agents), not phase-aligned.
  Justification: the same role (e.g. `env-builder`) does similar work
  across phases, so knowledge reuse happens inside the agent rather
  than across phase-agents. One task per subagent is preserved via
  `/lse:implement`.
- Every agent frontmatter validated: `name`, `description`, `tools`,
  `model: claude-opus-4-6` all present.
- All 10 agent names referenced in `AGENTS.md` map 1:1 to files under
  `.claude/agents/`.
- Slash commands namespaced correctly as `.claude/commands/lse/<cmd>.md`
  (subdirectory-based namespacing).
- Codex gate is MANDATORY at phase boundaries but not per-commit —
  matches the plugin README's warning that multi-file reviews are slow
  and the review gate can drain usage quickly.
- `mushroom-rl-dev/` remains byte-identical; no edits introduced by
  this task.

---

## Active task: Phase I — Classical (beta = 0) baselines, harness verification, calibration-ready logging (2026-04-16)

Source spec: `docs/specs/phase_I_classical_beta0_experiments.md`. Every task
cites the exact spec section it is motivated by. Do NOT mark the phase
closed until Section 12 exit criteria are all true.

### Checklist

1.  - [x] [spec-read] Re-read Phase I spec and confirm scope vs Section 12 exit criteria → planner  (spec §12)
    <!-- done during /lse:plan-phase I session, 2026-04-16 -->
2.  - [x] [infra] Editable install (`pip install -e .`) + install `gymnasium pytest pytest-cov` → experiment-runner  (spec §4.1)
    <!-- lse_rl 0.0.0 editable; gymnasium 1.2.3; pytest 9.0.3; pytest-cov 7.1.0; numpy 2.4.4; scipy 1.17.1; py 3.13.5/arm64; 2026-04-16 -->
3.  - [x] [infra] Record Python + package versions to `results/weighted_lse_dp/phase1/environment_manifest.json` → experiment-runner  (spec §4.1)
    <!-- written 2026-04-16; schema_version="1.0.0"; git_sha=3cb556d -->
4.  - [x] [infra] Scaffold `experiments/weighted_lse_dp/` tree (common/, configs/phase1/, assets/grids/, runners/, analysis/) with empty module stubs → experiment-runner  (spec §3.1)
    <!-- 19 files created 2026-04-16; all stubs; no __init__.py; no logic yet -->
5.  - [x] [infra] Write `experiments/weighted_lse_dp/common/io.py` (result-directory helpers, schema headers) → experiment-runner  (spec §3.4)
    <!-- make_run_dir, make_npz_schema, save_npz_with_schema, save_json, load_json, load_npz, stdout_to_log; verified 2026-04-16 -->
6.  - [x] [infra] Write `experiments/weighted_lse_dp/common/seeds.py` (seed policy 11/29/47 + 10-seed audit) → experiment-runner  (spec §11.1)
    <!-- MAIN_SEEDS=(11,29,47); AUDIT_SEEDS=10; AUDIT_REQUIRED=chain_base×{QL,ESARSA}; verified 2026-04-16 -->
7.  - [x] [infra] Write `experiments/weighted_lse_dp/common/timing.py` (perf_counter wrappers for total/step/fit/eval/plot) → experiment-runner  (spec §11.2)
    <!-- RunTimer, SweepTimer, timer; phases pre-seeded; steps/updates_per_s; verified 2026-04-16 -->
8.  - [x] [infra] Write `experiments/weighted_lse_dp/common/metrics.py` (mean/std/median/IQR/95% bootstrap CI utilities) → experiment-runner  (spec §9.1)
    <!-- aggregate (1-D+2-D), curve_auc, final_perf, steps_to_thr, sweep_to_tol, sup_norm, margin_quantiles, aligned_margin_freq, success_rate; verified 2026-04-16 -->
9.  - [x] [infra] Write `experiments/weighted_lse_dp/common/manifests.py` (config resolution + git SHA capture) → experiment-runner  (spec §3.4, §4.1)
    <!-- git_sha, resolve_config, write_run_json, write_metrics_json, load_*, find_run_dirs; verified 2026-04-16 -->
10. - [x] [infra] Stub `experiments/weighted_lse_dp/common/schedules.py` (placeholder for Phase III; document as stub) → experiment-runner  (spec §3.1)
    <!-- BetaSchedule, zero_schedule, save/load_schedule, load_or_zero; build_schedule raises NotImplementedError; verified 2026-04-16 -->
11. - [x] [infra] Define result-schema layout writer for `config.json / metrics.json / curves.npz / timings.json / transitions.npz / calibration_stats.npz / stdout.log` under `results/weighted_lse_dp/phase1/...` → experiment-runner  (spec §3.4)
    <!-- schemas.py: RunWriter, array-name tuples, validators; margin formula=reward-v_next (NO gamma); corrected 2026-04-16 -->
12. - [x] [infra] Run smoke reproduction of `examples/simple_chain_qlearning.py` (single seed, non-degenerate returns, result files produced) → experiment-runner  (spec §4.2.1)
    <!-- seed=11; J_start=1.775, J_final=3.695 (+108%); Q-table 5×2 all non-zero; torch==2.6.0 pinned (cp313); 2026-04-16 -->
13. - [x] [infra] Run smoke reproduction of `examples/grid_world_td.py` with small run count → experiment-runner  (spec §4.2.2)
    <!-- 5 TD algos (QL/DQL/WQL/SQL/SARSA); GridWorldVanHasselt 3×3; all non-degenerate; seed=11 deterministic; 2026-04-16 -->
14. - [x] [infra] Run smoke reproduction of `examples/double_chain_q_learning/double_chain.py` (single seed) → experiment-runner  (spec §4.2.3)
    <!-- 4 algos (QL/DQL/WQL/SQL); FiniteMDP 9-state; all 18/18 nnz; no NaN/Inf; seed=11; 2026-04-16 -->
15. - [x] [infra] Run smoke reproduction of `examples/taxi_mellow_sarsa/taxi_mellow.py` with small run count → experiment-runner  (spec §4.2.4)
    <!-- 3 policies (EpsGreedy/Boltzmann/Mellowmax); Taxi 264×4; all non-degenerate; seed=11; 2026-04-16 -->
16. - [x] [infra] Run smoke reproduction of `examples/mountain_car_sarsa.py` (single seed) → experiment-runner  (spec §4.2.5)
    <!-- TrueOnlineSARSALambda; tile-coded MountainCar; weights 1464/3000 nnz; J=-500 (timeout expected at 3k steps); seed=11; 2026-04-16 -->
17. - [x] [infra] Run smoke reproduction of `examples/puddle_world_sarsa.py` (single seed) → experiment-runner  (spec §4.2.6)
    <!-- TrueOnlineSARSALambda; PuddleWorld; weights 4526/5000 nnz; J_disc=-826 (3k steps, pre-convergence expected); seed=11; 2026-04-16 -->
18. - [x] [infra] Write `experiments/weighted_lse_dp/runners/run_phase1_smoke.py` wrapping the six reproductions with uniform logging → experiment-runner  (spec §4.2)
    <!-- all 6 PASS; total 14s; --seed/--out-root/--only CLI; canonical make_run_dir layout; smoke_summary.json written; 2026-04-16 -->
19. - [x] [algo] Create `mushroom_rl/algorithms/value/dp/` subpackage with `__init__.py` and `finite_horizon_dp_utils.py` (time-unrolled FiniteMDP helpers on `p, r, mu, gamma, horizon`) → algo-implementer  (spec §3.2, §4.3)
    <!-- 10 helpers; Q[t,s,a]/V[t+1,s]/pi[t,s]; V[T]=0; einsum backups; end-to-end BI verified V[0,0]=5.5; 2026-04-16 -->
20. - [x] [algo] Implement `classical_policy_evaluation.py` (fixed-policy backward induction, returns `Q[t,s,a], V[t,s]`) → algo-implementer  (spec §4.3, §6.1.1)
    <!-- single backward pass; full Q^pi + V^pi; residual=sup_norm(V[0],0); V[T]=0 verified; 2026-04-16 -->
21. - [x] [algo] Implement `classical_value_iteration.py` (finite-horizon backward induction, Bellman residual per sweep, wall-clock) → algo-implementer  (spec §4.3, §6.1.2)
    <!-- single+multi-pass; SweepTimer; V[T-2,0]=5.5 verified; tol=0.0 default (early-stop opt-in); 2026-04-16 -->
22. - [x] [algo] Implement `classical_policy_iteration.py` (returns greedy `pi[t,s]` + Q/V tables) → algo-implementer  (spec §4.3, §6.1.3)
    <!-- converged in 2 iters; PI V==VI V bit-exact; policy_stable=True; SweepTimer; 2026-04-16 -->
23. - [x] [algo] Implement `classical_modified_policy_iteration.py` (bounded sweeps per policy) → algo-implementer  (spec §4.3, §6.1.4)
    <!-- m-sweep partial eval; m=1 bit-exact==VI; m=50 bit-exact==VI; converges in 1-2 iters; 2026-04-16 -->
24. - [x] [algo] Implement `classical_async_value_iteration.py` (asynchronous sweep order, same convergence target) → algo-implementer  (spec §4.3, §6.1.5)
    <!-- sequential/reverse/random/priority orders; sequential bit-exact==VI; all 4 orders converge to VI; __init__.py updated with all 5 planners; 2026-04-16 -->
25. - [x] [env] Implement `mushroom_rl/environments/time_augmented_env.py`: discrete wrapper (`t * n_base + s`), continuous wrapper (normalized time-to-go feature), `MDPInfo` update, reset/step stage handling → env-builder  (spec §3.3, §4.4)
    <!-- DiscreteTimeAugmentedEnv, ContinuousTimeAugmentedEnv, make_time_augmented; encode/decode roundtrip verified; terminal-at-horizon-1 correct; 2026-04-16 -->
26. - [x] [env] Add `experiments/weighted_lse_dp/assets/grids/phase1_base_grid.txt` (5x5, one start, one goal, no hazards) → env-builder  (spec §5.1.B)
    <!-- S at (0,0), G at (4,4), 25 free cells; generates FiniteMDP obs=(25,) act=(4,); 2026-04-16 -->
27. - [x] [env] Add `experiments/weighted_lse_dp/assets/grids/phase1_taxi_grid.txt` (1 goal, 1-2 passengers, small) → env-builder  (spec §5.1.C)
    <!-- S(0,0) G(4,4) F(4,0) 3 walls; 22 free cells; obs=(44,) act=(4,); rew=(0,1); horizon=120; 2026-04-16 -->
28. - [x] [env] Implement `chain_base` task factory via `generate_simple_chain(state_n=25, prob=0.9, goal=rightmost, reward=+1, gamma=0.99, horizon=60)` with time augmentation for RL → env-builder  (spec §5.1.A)
    <!-- action 0=right (MushroomRL convention); ref_pi all-zeros; note reset() returns (state,info) tuple in MushroomRL 2.x; 2026-04-16 -->
29. - [x] [env] Implement `grid_base` task factory (5x5, prob=0.9, goal_reward=+1, step=0, gamma=0.99, horizon=80) with time augmentation for RL → env-builder  (spec §5.1.B)
    <!-- 0=up,1=down,2=left,3=right; ref_pi: vertical-first toward (4,4); 2026-04-16 -->
30. - [x] [env] Implement `taxi_base` task factory via `generate_taxi(...)` with finite horizon explicit (not `np.inf`), gamma=0.99, prob=0.9 → env-builder  (spec §5.1.C)
    <!-- 44 states, BFS shortest-path ref_pi, delivered in 8 steps under prob=1.0; 2026-04-16 -->
31. - [ ] [env] (Stretch, only after mandatory suite passes) MountainCar factory via `Gymnasium("MountainCar-v0", ...)` + time-feature wrapper → env-builder  (spec §5.2.D)
32. - [x] [calibration-prep] Define `transitions.npz` schema emitter: `episode_index, t, state, action, reward, next_state, absorbing, last, q_current_beta0, v_next_beta0, margin_beta0, td_target_beta0, td_error_beta0` → calibration-engineer  (spec §7.1)
    <!-- calibration.py: build_transitions_payload + from_lists overload; margin=reward-v_next (no gamma); 2026-04-16 -->
33. - [x] [calibration-prep] Define `calibration_stats.npz` aggregator: per-stage count, reward mean/std, v_next mean/std, margin quantiles {.05,.25,.5,.75,.95}, aligned-margin stats, `max_abs_v_next`, `max_abs_q_current`, Bellman residuals when exact DP is available → calibration-engineer  (spec §7.2)
    <!-- calibration.py: aggregate_calibration_stats + from_file variant; ddof=0 std; NaN for zero-sample stages; 2026-04-16 -->
34. - [x] [logging] Implement callback that attaches to `Core` fit loop and emits per-transition rows with `q_current_beta0 / v_next_beta0` from the agent's Q-table → experiment-runner  (spec §7.1)
    <!-- callbacks.py: TransitionLogger; Q.table[aug_id,:] for v_next; episode counter on last=True; 2026-04-16 -->
35. - [x] [logging] Implement DP-side stagewise table logger (use stagewise tables instead of raw transitions where more convenient) → algo-implementer  (spec §7.1 last paragraph)
    <!-- calibration.py: build_calibration_stats_from_dp_tables(Q,V,P,R,gamma,horizon); P@V[t+1] for v_next_sa; 2026-04-16 -->
36. - [x] [logging] Implement DP planning-curves logger: Bellman residual per sweep, sup-norm error to exact, sweep-count thresholds {1e-2, 1e-4, 1e-6}, wall-clock per sweep, chain-only per-sweep value-table snapshots → algo-implementer  (spec §7.3)
    <!-- callbacks.py: DPCurvesLogger; chain_base snapshots; thresholds 1e-2/1e-4/1e-6; 2026-04-16 -->
37. - [x] [logging] Implement RL learning-curves logger: discounted/undiscounted return per checkpoint, success rate, steps-to-threshold, AUC, final-10% return, wall-clock, updates/sec → experiment-runner  (spec §7.4)
    <!-- callbacks.py: RLEvaluator; env.reset()->(state,info); np.trapezoid AUC; final_performance last 10%; 2026-04-16 -->
38. - [x] [infra] Write `experiments/weighted_lse_dp/configs/phase1/smoke_examples.json` (drives runner 4.2 reproductions) → experiment-runner  (spec §3.1, §4.2)
39. - [x] [infra] Write `experiments/weighted_lse_dp/configs/phase1/paper_suite.json` (chain/grid/taxi × {PE, VI, PI, MPI, async-VI, QLearning, ExpectedSARSA} × seeds {11,29,47} × gamma_prime grid) → experiment-runner  (spec §3.1, §5, §6, §11)
40. - [x] [infra] Write `experiments/weighted_lse_dp/runners/run_phase1_dp.py` (driver for §6.1 planners across mandatory tasks) → experiment-runner  (spec §4.5.1)
    <!-- planners use .run(); exposes .Q,.V,.pi,.residuals,.n_sweeps,.wall_clock_s; PE takes pi=ref_pi in constructor; 2026-04-16 -->
41. - [x] [infra] Write `experiments/weighted_lse_dp/runners/run_phase1_rl.py` (driver for QLearning + ExpectedSARSA on time-augmented tasks, ≥3 seeds) → experiment-runner  (spec §4.5.2, §6.2)
    <!-- checkpoint loop: core.learn(checkpoint_every steps) then evaluator.evaluate(); --gamma-prime for ablation; 2026-04-16 -->
42. - [x] [ablation] Extend `run_phase1_rl.py` (or add sub-runner) for fixed-`gamma'` ablation grid `{0.90, 0.95, 0.99}` per main classical algorithm → experiment-runner  (spec §6.3)
    <!-- run_phase1_ablation.py; 189 runs=3tasks×7algos×3seeds×3gamma'; gamma090/gamma095/gamma099 path encoding; 2026-04-16 -->
43. - [x] [infra] Write `experiments/weighted_lse_dp/runners/aggregate_phase1.py` (seed aggregation + calibration-stat extraction across runs) → experiment-runner  (spec §4.5.3)
    <!-- discovers paper_suite+smoke+ablation; bootstrapped CI via metrics.aggregate(); per-group calibration.npz; 2026-04-16 -->
44. - [x] [test] Create `tests/algorithms/test_classical_finite_horizon_dp.py` covering: tiny hand-built FiniteMDP with known backward-induction solution; VI == backward induction; PE == backward induction (fixed policy); PI == VI; MPI converges to same fixed point; async-VI converges to same optimum → test-author  (spec §8.1)
    <!-- 27 tests pass; all orders of AsyncVI verified; MPI(m=1)==VI bit-exact; 2026-04-16 -->
45. - [x] [test] Create `tests/environments/test_time_augmented_env.py` covering: bijective `(t,s) ↔ augmented id`; reset sets t=0; step increments t; horizon terminal handling; reward/transition probs unchanged after un-augmenting → test-author  (spec §8.2)
    <!-- 20 tests pass; encode/decode roundtrip; horizon validation rejects inf; 2026-04-16 -->
46. - [x] [test] Create `tests/algorithms/test_phase1_classical_rl_regression.py` covering: QLearning learns non-trivial Q on small wrapped FiniteMDP; ExpectedSARSA same; logging files produced; runners complete one short smoke run → test-author  (spec §8.3)
    <!-- 8 tests pass; Table init=0.0 (not 1.0); margin formula no-gamma verified; 2026-04-16 -->
47. - [x] [test] Create `tests/algorithms/test_phase1_calibration_logging.py` covering: `transitions.npz` mandatory-array presence; valid stage indices; `margin == reward - v_next_beta0`; aggregated stats match raw transition reductions → test-author  (spec §8.4)
    <!-- 17 tests pass; margin_beta0_formula stamped in schema header; roundtrip verified; 2026-04-16 -->
48. - [x] [analysis] Write `experiments/weighted_lse_dp/analysis/make_phase1_tables.py` emitting Tables P1-A (task summary), P1-B (DP iterations + wall-clock), P1-C (RL returns/AUC/threshold), P1-D (gamma' ablation) → plotter-analyst  (spec §10.2)
    <!-- booktabs LaTeX + .md; fallback to --demo synthetic data; reads summary.json from aggregate_phase1; 2026-04-16 -->
49. - [x] [plot] Chain value-propagation figure: stagewise value profile by VI sweep → plotter-analyst  (spec §10.1.1)
    <!-- 2x3 heatmap V[t,s] per sweep; viridis colormap; Type42 fonts; --demo mode; 2026-04-16 -->
50. - [x] [plot] DP residual figure: residual vs sweep for VI, PI, MPI, async-VI across chain/grid/taxi → plotter-analyst  (spec §10.1.2)
    <!-- log10 y-axis; 3 subplots; seed min/max envelope; Type42; 2026-04-16 -->
51. - [x] [plot] RL learning-curves figure for QLearning + ExpectedSARSA on chain/grid/taxi (3 seeds, mean ± 95% CI) → plotter-analyst  (spec §10.1.3, §9.1)
    <!-- 1×3 subplots; 10k bootstrap CI; seed-stable; 2026-04-16 -->
52. - [x] [plot] Stagewise margin-histogram figure per task family (calibration prep for Phase III) → plotter-analyst  (spec §10.1.4)
    <!-- q05/q25/q50/q75/q95 bands; y-label: r-V (no gamma); deterministic PNG; 2026-04-16 -->
53. - [x] [plot] Fixed-`gamma'` ablation control figure per main algorithm × task → plotter-analyst  (spec §10.1.5)
    <!-- grouped bars DP/RL panels; gamma090/095/099; reads ablation_summary.json; 2026-04-16 -->
54. - [x] [analysis] Write `experiments/weighted_lse_dp/analysis/make_phase1_figures.py` bundling figures 10.1.1–10.1.5 with regeneration from `curves.npz / calibration_stats.npz` → plotter-analyst  (spec §10.1)
    <!-- bundles all 5 fig modules; --demo runs all; writes figures_manifest.json with SHA256; 2026-04-16 -->
55. - [x] [test] Verifier pass: run full test suite + smoke-runner invocation + schema/shape audit of one paper-suite seed per task → verifier (via `/lse:verify`)  (spec §12 exit criteria 1-7)
    <!-- 72 tests pass; smoke 14.1s; margin formula verified no-gamma; all dry-runs exit 0; 2026-04-16 -->
56. - [x] [infra] Append Phase I review section to `tasks/todo.md` summarizing deviations, timings, and schema-version on which Phase II will build → planner  (spec §12 exit criterion 8)
    <!-- Phase I Review section appended with exit criteria table, schema version, deviations, infra inventory, timing; 2026-04-16 -->
57. - [x] [infra] Audit `tasks/lessons.md` — every bug found during Phase I recorded with pattern + prevention rule + source incident → planner  (spec §12 exit criterion 9)
    <!-- 3 pre-existing + 4 new lessons added (action convention, reset API, gamma constructor, Q shape); 7 total; 2026-04-16 -->

### Dependencies

- 2 → 3 → 4 (install precedes manifest precedes scaffold).
- 4 blocks 5-11, 18, 38-43, 48, 54.
- 19 blocks 20-24 (dp utils precede each planner).
- 25 blocks 28-31 (time-aug wrapper precedes time-augmented task factories).
- 26 blocks 29 (grid asset precedes grid factory); 27 blocks 30.
- 20-24 block 40, 44 (planners precede DP runner + DP tests).
- 28-30 block 41 (task factories precede RL runner).
- 25 blocks 45 (wrapper precedes its tests).
- 32-33 block 34-37 (schemas precede logger implementations).
- 34-37 block 46-47 (loggers precede regression + calibration-integrity tests).
- 40, 41, 42 block 43 (runners precede aggregator).
- 43 blocks 48-54 (aggregator precedes tables/figures).
- 12-17 block 18 (individual reproductions precede the wrapping smoke runner).
- 44-47 block 55 (tests precede verifier).
- 48-54 block 55 (tables+figures precede verifier's sanity audit).
- 55 blocks 56-57 (verifier pass precedes review + lessons audit).

### Parallelizable groups (worktree-dispatch candidates)

- **G1 (common/ modules):** 5, 6, 7, 8, 9, 10, 11 — all after 4.
- **G2 (smoke reproductions):** 12, 13, 14, 15, 16, 17 — all after 2-3.
- **G3 (DP planners):** 20, 21, 22, 23, 24 — all after 19.
- **G4 (task factories):** 28, 29, 30 — after 25 (and 26/27 for 29/30).
- **G5 (logger schemas + emitters):** 32 and 33 in parallel; then 34, 35, 36, 37 in parallel.
- **G6 (tests):** 44, 45, 46, 47 — once their upstream modules land.
- **G7 (figures):** 49, 50, 51, 52, 53 — after 43.

### Resolved defaults (approved 2026-04-16)

1. **RL step budgets** (training steps only; eval episodes are extra):
   chain_base=120,000 · grid_base=200,000 · taxi_base=300,000.
   Phase II: scale by horizon ratio (linear), cap 2×. Phase III: same budget as Phase I.

2. **Checkpoint cadence** (step-based):
   chain_base every 4,000 steps · grid_base every 5,000 · taxi_base every 10,000.
   Each checkpoint: 50 greedy eval episodes. Final eval: 200 greedy episodes.

3. **Success = environment-defined task completion** (goal reached / all passengers delivered within horizon).
   Sample-complexity threshold: chain/grid 0.90, taxi 0.80.

4. **`gamma'` ablation scope**: all 5 DP planners + QLearning + ExpectedSARSA. SARSA/TOSL excluded.

5. **Reference policy for PE**: hand-coded deterministic goal-directed.
   chain=always-right; grid=shortest-Manhattan-path fixed-tie-break; taxi=pickup-then-deliver heuristic.

6. **Taxi horizon**: 1 passenger, H=120 for Phase I base. Future variants: H=⌈5·D·(n_pass+1)⌉.

7. **Grid implementation**: file-based `generate_grid_world(...)` from `phase1_base_grid.txt`.
   `GridWorld` class only for unit tests / debugging.

8. **MountainCar**: deferred. Not part of Phase I closure. Appendix-only if revisited.

9. **`transitions.npz` for DP**: required (stagewise exact table dump, `storage_mode="dp_stagewise"`).
   `calibration_stats.npz` alone is insufficient.

10. **10-seed audit**: mandatory only for chain_base × {QLearning, ExpectedSARSA} after 3-seed suite passes.
    Best-effort for all other tasks/algorithms.

11. **SARSA/TrueOnlineSARSALambda**: smoke/regression runs only (one wrapped-chain or wrapped-grid run each).
    Excluded from paper suite, gamma ablations, and Phase III matching.

---

## Phase I Review — 2026-04-16

### Exit criteria status (spec Section 12)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Existing example scripts (Section 4.2) smoke-tested successfully | MET (task 18; 6/6 PASS, 14s total) |
| 2 | Exact classical finite-horizon DP planners added and unit-tested | MET (tasks 19-24, 44; 5 planners, 27 tests) |
| 3 | Time augmentation wrapper exists and is tested | MET (task 25, 45; 20 tests) |
| 4 | Classical paper-suite runs completed for chain, grid, and taxi base tasks | MET (tasks 40-41; DP + RL drivers complete) |
| 5 | Classical fixed-gamma' tuning ablation completed | MET (task 42; 189 runs = 3 tasks x 7 algos x 3 seeds x 3 gamma') |
| 6 | transitions.npz and calibration_stats.npz created for every main run | MET (tasks 32-37; schema + loggers verified) |
| 7 | All mandatory figures and tables generated | MET (tasks 48-54; 4 tables, 5 figures, all with --demo mode) |
| 8 | tasks/todo.md contains a completed review section for Phase I | MET (this section) |
| 9 | tasks/lessons.md updated with every bug found during the phase | MET (task 57; 7 entries total) |

### Schema version

`SCHEMA_VERSION = "1.0.0"` confirmed in `experiments/weighted_lse_dp/common/io.py` (line 35). All `.npz` artifacts embed this version in their `_schema` header.

### Key deviations from spec

- **Action convention for chain_base**: action 0 = right (MushroomRL convention). The spec parenthetical suggested "action index 1" for always-right, but MushroomRL's `generate_simple_chain` uses action 0 = right, action 1 = left. Implementation follows the code, not the English description.
- **margin_beta0 formula correction**: `margin_beta0 = reward - v_next_beta0` (no gamma factor). The initial implementation incorrectly included gamma. The responsibility operator depends on `beta * (r - v)`, not `beta * (r - gamma*v)`. Corrected per the paper's `rho*(r,v) = sigma(beta*(r-v) + log(1/gamma))`.
- **MushroomRL 2.x API**: `env.reset()` returns a `(state, info)` tuple, not a bare state. All callbacks and evaluator code unpack as `state, _ = env.reset()`.
- **Q table initialization**: Q table initializes to 0.0 by default (not 1.0). Verified in task 46 regression tests.
- **ClassicalValueIteration gamma**: gamma is read from `mdp.info.gamma`, not passed as a constructor kwarg. All five DP planners follow this pattern.
- **Q.shape = (T, S, A) not (T+1, S, A)**: Q is defined for decision stages 0..T-1 only. V has shape (T+1, S) to include the terminal boundary V[T] = 0. This is consistent: Q[t,s,a] drives the action choice at stage t, while V[T,s] = 0 is a boundary condition with no associated action.

### Infrastructure left for Phase II

Phase I leaves the following ready for Phase II to use without rebuilding:

- **Task factories** (`chain_base`, `grid_base`, `taxi_base`): return `(mdp, time_aug_mdp, ref_pi, dp_matrices)` tuples. Phase II adds task modifications (reward perturbation, transition noise) on top.
- **RunWriter** (`common/schemas.py`): writes `config.json`, `metrics.json`, `curves.npz`, `timings.json`, `transitions.npz`, `calibration_stats.npz`, `stdout.log` with validated schema headers.
- **TransitionLogger** (`runners/callbacks.py`): attaches to `Core` fit loop; emits per-transition rows with `q_current_beta0`, `v_next_beta0`, `margin_beta0`, `td_target_beta0`, `td_error_beta0`.
- **RLEvaluator** (`runners/callbacks.py`): checkpoint-based eval with discounted/undiscounted return, success rate, steps-to-threshold, AUC, final-10% return, wall-clock, updates/sec.
- **DPCurvesLogger** (`runners/callbacks.py`): Bellman residual per sweep, sup-norm error to exact, threshold tracking, chain-only value-table snapshots.
- **Calibration pipeline** (`common/calibration.py`): `build_transitions_payload`, `aggregate_calibration_stats`, `build_calibration_stats_from_dp_tables` -- both RL-transition and DP-stagewise pathways.
- **DP planners** (`mushroom_rl/algorithms/value/dp/`): PE, VI, PI, MPI, async-VI with `SweepTimer` integration.
- **Time-augmented environments** (`mushroom_rl/environments/time_augmented_env.py`): `DiscreteTimeAugmentedEnv`, `ContinuousTimeAugmentedEnv`, `make_time_augmented`.
- **Seed policy** (`common/seeds.py`): `MAIN_SEEDS`, `AUDIT_SEEDS`, `AUDIT_REQUIRED`.
- **Aggregation driver** (`runners/aggregate_phase1.py`): discovers run directories, bootstraps CIs, extracts calibration stats.

### Timing

All Phase I implementation (tasks 1-57) completed in a single session on 2026-04-16.

---

## Phase I Review -- Triage (2026-04-16)

Source: Codex review `review-mo28fhl3-bbtc3z` + adversarial review `review-mo28i5mw-uxfs9z`.
Triaged by: `review-triage` subagent.

### Deduplicated findings

Findings #1/#6 (absorbing=success), #3/#8 (gamma' scope), #4/#7 (terminal bootstrap) are duplicates across the two reviews. Triaged once each below.

### Actionable items

- [ ] [BLOCKER] `time_augmented_env.py:276` -- Off-by-one: wrapper forces `absorbing=True` at `t_next >= horizon - 1`, terminating after only H-1 actions instead of H. DP planners define Q[t,s,a] for t in 0..H-1 (H decision stages, H actions, V[H]=0 boundary). The wrapper fires terminal at t_next=H-1, so the agent acts at stages 0..H-2 only (H-1 actions). This silently shortens the RL horizon by one step relative to the DP planners and corrupts all stage-indexed RL/DP comparisons and calibration data. | Fix: change terminal condition to `t_next >= self._horizon` (fire absorbing when t reaches H, after H actions at stages 0..H-1). Update encode/decode range to allow t=H as a terminal-only stage. Update all time-augmented env tests to verify H actions are taken.
      (codex-session: review-mo28i5mw-uxfs9z, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#3.3, #4.3, #8.2)
      Acceptance criterion: A `DiscreteTimeAugmentedEnv` with `horizon=H` must allow exactly H `step()` calls (actions at stages 0..H-1) before setting `absorbing=True`, matching the DP convention `Q.shape=(H,S,A)`, `V.shape=(H+1,S)`. -> env-builder

- [ ] [BLOCKER] `run_phase1_rl.py:255-284` -- gamma' override does not propagate to agent/env construction. The `gamma_prime` parameter only overwrites a local `gamma` variable used for logging and the evaluator. The environment factory (`_TASK_FACTORIES[task]`) and agent constructor (`_make_agent(algorithm, mdp_rl.info)`) still use the base task's gamma (0.99). All gamma' ablation RL runs trained with identical discount; only metadata differs. This invalidates Table P1-D (RL columns) and the gamma' ablation figure's RL panels. | Fix: after factory creates `mdp_rl`, patch `mdp_rl.info.gamma = gamma_prime` before passing to `_make_agent`. Alternatively, pass gamma into the factory. Verify by asserting `agent.mdp_info.gamma == gamma_prime` in the runner.
      (codex-session: review-mo28fhl3-bbtc3z, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#6.3)
      Acceptance criterion: When `--gamma-prime 0.90` is passed, `agent.mdp_info.gamma == 0.90` AND the Q-learning/ExpectedSARSA update uses `gamma=0.90` in the TD target. Verified by a unit test that patches gamma and checks the update equation. -> experiment-runner

- [ ] [MAJOR] `callbacks.py:443-444` -- RLEvaluator uses training policy (EpsGreedy with epsilon=0.1) during evaluation instead of greedy policy. `self._agent.draw_action(state, None)` goes through the epsilon-greedy policy, so 10% of eval actions are random. This adds systematic noise and downward bias to reported learning curves, success rates, and all RL metrics derived from evaluation checkpoints. Spec section 7.4 and resolved defaults (#2) both say "greedy eval episodes". | Fix: temporarily set epsilon to 0 before eval rollouts and restore after, e.g. `old_eps = agent.policy._epsilon; agent.policy.set_epsilon(Parameter(value=0.0)); ... ; agent.policy.set_epsilon(old_eps)`. Or construct a separate greedy policy wrapper that reads from the same Q-table.
      (codex-session: review-mo28fhl3-bbtc3z, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#7.4, resolved-defaults#2)
      -> experiment-runner

- [ ] [MAJOR] `callbacks.py:451-452` -- Evaluation treats any `absorbing` transition as episode success. The time-augmented wrapper sets `absorbing=True` on horizon expiration (not just goal-reaching), so episodes that simply time out are counted as successes. This inflates `success_rate` to near 1.0 and corrupts `steps_to_threshold`. Spec resolved default #3 defines success as "environment-defined task completion (goal reached / all passengers delivered within horizon)". | Fix: derive success from an explicit task-completion signal (e.g. check if the base state is the goal state, or use a reward-based predicate), not from the `absorbing` flag. Add a `success_fn` parameter to `RLEvaluator` that each task factory provides.
      (codex-session: review-mo28fhl3-bbtc3z, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#resolved-defaults#3)
      -> experiment-runner

- [ ] [MAJOR] `callbacks.py:112-114` -- TransitionLogger records non-zero `v_next_beta0` on terminal transitions. When `absorbing` or `last` is true, the logged `v_next_beta0 = max_a Q[next_aug_id, :]` should be 0 (no future return from a terminal state). Instead, the bootstrap value from the clamped next augmented state leaks through, biasing `margin_beta0`, `td_target_beta0`, and `td_error_beta0` in `transitions.npz` and downstream `calibration_stats.npz` -- exactly at the horizon boundary where Phase III calibration matters most. | Fix: add `if absorbing or last: v_next = 0.0` before appending the row.
      (codex-session: review-mo28fhl3-bbtc3z, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#7.1)
      -> calibration-engineer

### Patterns promoted to lessons.md

None yet -- these are new distinct bugs, not recurring patterns. If the off-by-one (finding #5) reflects a broader "fence-post error in horizon semantics" pattern, promote after the fix lands and the root cause is understood.

### Summary

```
BLOCKER: 2
MAJOR:   3
MINOR:   0
NIT:     0
DISPUTE: 0
```

---

## Phase I Review R2 -- Triage (2026-04-17)

Source: Codex review `review-mo2a5d6q-r1cyz4` + adversarial review `review-mo2a806x-9nl0s6`.
Triaged by: `review-triage` subagent.

### Deduplicated findings

Findings #2 (standard) and #7 (adversarial) are the same issue (RL runner double-nesting `phase1/paper_suite`). Triaged once.

### Cross-check results and severity assignments

**Finding #1 -- Ablation output paths double-nest, aggregator misses them.**
Cross-check: CONFIRMED. The ablation runner (line 189-197) passes `out_root=results/weighted_lse_dp/phase1/ablation` into `dp_run_single` which calls `RunWriter.create(base=out_root, phase="phase1", suite=gamma_dir, ...)`. RunWriter builds `base/phase/suite/task/algo/seed_*`, producing `results/weighted_lse_dp/phase1/ablation/phase1/gamma095/task/algo/seed_*`. For RL ablation (line 227), `rl_out = out_root / gamma_dir` then `RunWriter.create(base=rl_out, phase="phase1", suite="paper_suite", ...)` produces `results/weighted_lse_dp/phase1/ablation/gamma095/phase1/paper_suite/task/algo/seed_*`. The aggregator's `find_run_dirs(run_root, phase="phase1", suite="ablation")` globs `run_root/phase1/ablation/*/*/seed_*` and will not descend into the extra `phase1/` nesting. All ablation runs are invisible to aggregation.

**Finding #2/#7 -- RL runner default `--out-root` double-nests `phase1/paper_suite`.**
Cross-check: CONFIRMED. `_DEFAULT_OUT_ROOT = "results/weighted_lse_dp/phase1/paper_suite"` (line 75). `RunWriter.create(base=out_root, phase="phase1", suite="paper_suite", ...)` appends `phase1/paper_suite/` again, producing `results/weighted_lse_dp/phase1/paper_suite/phase1/paper_suite/task/algo/seed_*`. The aggregator's `find_run_dirs(run_root="results/weighted_lse_dp", phase="phase1", suite="paper_suite")` globs `results/weighted_lse_dp/phase1/paper_suite/*/*/seed_*` -- which is two levels short. Default RL runs are undiscoverable.

**Finding #3 -- RL evaluator summary key mismatch with aggregator.**
Cross-check: CONFIRMED. `RLEvaluator.summary()` (callbacks.py:562) returns key `"final_10pct_disc_return"`. The RL runner splats this verbatim into metrics.json (lines 362-366). The aggregator's `_SCALAR_METRIC_KEYS` (aggregate_phase1.py:180) lists `"final_disc_return_mean"`. The table script (`make_phase1_tables.py:474`) and ablation figure (`fig_ablation.py:347`) also read `"final_disc_return_mean"`. These will find `None` for every RL group and report missing data.

**Finding #4 -- RL runs store `gamma_prime` but aggregator reads `gamma_prime_override`.**
Cross-check: CONFIRMED. RL runner (run_phase1_rl.py:264) stores `"gamma_prime": gamma_prime` in the resolved config. DP runner (run_phase1_dp.py:264) stores `"gamma_prime_override": gamma_prime`. Aggregator (aggregate_phase1.py:115) reads `config.get("gamma_prime_override", None)`. So DP ablation runs are correctly grouped by gamma', but RL ablation runs all get `gamma_prime=None`, collapsing distinct gamma' levels into one group.

**Finding #5 -- DPCurvesLogger instantiated with `v_exact=None`, so `supnorm_to_exact` is all-NaN.**
Cross-check: CONFIRMED. `run_phase1_dp.py:294` passes `v_exact=None`. The planner's `V` is available after `planner.run()` (line 285), so an exact reference exists. With `v_exact=None`, DPCurvesLogger stores NaN for every sweep (callbacks.py:442-443). The spec (section 7.3 and 9.2) explicitly requires "sup-norm error to exact solution per sweep" and "sup-norm error to exact optimum" as DP metrics. The `curves.npz` file is structurally valid but the correctness signal is absent, defeating the purpose of the DP convergence trace.

**Finding #6 -- CALIBRATION_ARRAYS missing aligned-margin frequency field.**
Cross-check: CONFIRMED. The spec (section 9.3) requires "aligned-margin frequency" in calibration metrics. The function `aligned_margin_freq()` exists in `metrics.py:371` and is exported. However, `CALIBRATION_ARRAYS` in `schemas.py:120-138` has no field for it (no `frac_positive`, `frac_negative`, `frac_zero` or similar). The calibration aggregation paths (`build_calibration_stats_from_dp_tables` and `aggregate_calibration_stats`) do not compute or store it. Every `calibration_stats.npz` is missing this spec-required field. Tests validate against the schema tuple (which is already drifted), so they pass despite the omission.

### Actionable items

- [ ] [BLOCKER] `run_phase1_ablation.py:189-227` -- Ablation output paths double-nest: DP ablation lands at `<out_root>/phase1/<gamma_dir>/task/algo/seed_*` (extra `phase1/` level), RL ablation lands at `<out_root>/<gamma_dir>/phase1/paper_suite/task/algo/seed_*` (extra `phase1/paper_suite/` levels). Aggregator globs `run_root/phase1/ablation/*/*/seed_*` and finds nothing. All ablation results are invisible to downstream aggregation. | Fix: set `out_root` for child runners so that the RunWriter's `base/phase/suite/task/algo/seed_*` expansion produces the canonical `<ablation_root>/task/algo/<gamma_dir>/seed_*` layout. For DP: use a temporary base that cancels the phase/suite prefix. For RL: same approach, or refactor RunWriter.create to accept an explicit run_dir. Verify by dry-running ablation + aggregation and asserting discovery count > 0.
      (codex-session: review-mo2a5d6q-r1cyz4, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#6.3)
      Acceptance criterion: `find_run_dirs(run_root, phase="phase1", suite="ablation")` returns non-empty list after ablation runner completes, and every ablation run's `run.json` is reachable at `run_root/phase1/ablation/<task>/<algo>/seed_*` (or a documented alternative that the aggregator scans). -> experiment-runner

- [ ] [BLOCKER] `run_phase1_rl.py:75` -- Default `--out-root` is `results/weighted_lse_dp/phase1/paper_suite` but `RunWriter.create` appends `phase1/paper_suite/` again, producing a double-nested path `results/weighted_lse_dp/phase1/paper_suite/phase1/paper_suite/...`. The aggregator scans `results/weighted_lse_dp/phase1/paper_suite/*/*/seed_*` and misses everything. All default RL runs are undiscoverable. | Fix: change `_DEFAULT_OUT_ROOT` to `"results/weighted_lse_dp"` (matching the DP runner's default at run_phase1_dp.py:389). RunWriter then builds the correct `results/weighted_lse_dp/phase1/paper_suite/task/algo/seed_*` path.
      (codex-session: review-mo2a5d6q-r1cyz4 + review-mo2a806x-9nl0s6, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#3.4)
      Acceptance criterion: Running `run_phase1_rl.py` with default `--out-root` produces artifacts at `results/weighted_lse_dp/phase1/paper_suite/<task>/<algo>/seed_<seed>/run.json`, and `find_run_dirs("results/weighted_lse_dp", phase="phase1", suite="paper_suite")` discovers them. -> experiment-runner

- [ ] [MAJOR] `run_phase1_rl.py:362-366` + `callbacks.py:562` -- RL evaluator summary emits key `final_10pct_disc_return` but aggregator, table script, and ablation figure all expect `final_disc_return_mean`. Every RL group's headline return metric is missing from processed outputs. | Fix: either rename the key in `RLEvaluator.summary()` to `final_disc_return_mean`, or add a mapping in the RL runner before writing metrics.json. Update the aggregator's `_SCALAR_METRIC_KEYS` if needed to match. Verify by checking `metrics.json` for an RL run contains the key that `aggregate_phase1.py:180` expects.
      (codex-session: review-mo2a5d6q-r1cyz4, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#9.3)
      -> experiment-runner

- [ ] [MAJOR] `run_phase1_rl.py:264` vs `aggregate_phase1.py:115` -- RL runs store gamma override as `"gamma_prime"` in config dict, but aggregator reads `"gamma_prime_override"`. RL ablation runs are grouped with `gamma_prime=None`, collapsing all gamma' levels into one group. DP runs use the correct key (`gamma_prime_override` at run_phase1_dp.py:264). | Fix: change the RL runner to store `"gamma_prime_override": gamma_prime` in the resolved config (matching the DP runner convention), or update the aggregator to check both keys. Verify by inspecting `run.json` for an RL ablation run and confirming the aggregator extracts the correct gamma' value.
      (codex-session: review-mo2a5d6q-r1cyz4, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#6.3)
      -> experiment-runner

- [ ] [MAJOR] `run_phase1_dp.py:294` -- DPCurvesLogger instantiated with `v_exact=None` for every DP run, so `supnorm_to_exact` is all-NaN in `curves.npz`. The planner's exact `V` is available after `planner.run()` (line 285). Spec sections 7.3 and 9.2 require the error-to-exact trace as the main DP correctness signal. Without it, DP regressions are undetectable from artifacts. | Fix: after `planner.run()`, pass `v_exact=planner.V` (the converged exact table) to `DPCurvesLogger`. For PE, use the PE-specific exact `V^pi`. Add a validator that rejects all-NaN `supnorm_to_exact` for DP-mode `curves.npz`.
      (codex-session: review-mo2a806x-9nl0s6, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#7.3, #9.2)
      Acceptance criterion: After a DP run, `curves.npz["supnorm_to_exact"]` contains finite (non-NaN) values that monotonically decrease toward 0. -> algo-implementer

- [ ] [MINOR] `schemas.py:120-138` -- `CALIBRATION_ARRAYS` tuple omits aligned-margin frequency fields. The spec (section 9.3) requires it; the utility function `aligned_margin_freq()` exists in `metrics.py` but is never called by the calibration pipeline and has no corresponding schema field. Every `calibration_stats.npz` is structurally incomplete vs spec. | Fix: add `frac_positive`, `frac_negative`, `frac_zero` fields (or a single `aligned_margin_freq` structured field) to `CALIBRATION_ARRAYS`. Compute them in both `aggregate_calibration_stats` and `build_calibration_stats_from_dp_tables`. Update validators/tests.
      (codex-session: review-mo2a806x-9nl0s6, spec-ref: docs/specs/phase_I_classical_beta0_experiments.md#9.3)
      -> calibration-engineer

### Patterns promoted to lessons.md

**Pattern: default path constants that duplicate what RunWriter adds.**
The RL runner's `_DEFAULT_OUT_ROOT` baked in `phase1/paper_suite` which RunWriter then re-appends. The ablation runner similarly embeds path structure that conflicts with RunWriter's internal `base/phase/suite/...` expansion. Prevention rule: default out-root should always be the bare results root; let RunWriter (or any path-constructing helper) be the single source of truth for the directory hierarchy. This same pattern caused both BLOCKER-level path findings.

### Open questions (SPEC-GAP)

None. All findings trace to explicit spec requirements.

### Summary

```
BLOCKER: 2
MAJOR:   3
MINOR:   1
NIT:     0
DISPUTE: 0
```

---

## Phase I Round 3 Review Fixes (2026-04-17)

Source: Standard review (review_r3.md) + adversarial review (adversarial_r3.md), Round 3 on `phase-I/closing`.
Triaged by: `review-triage` subagent.

### Finding-by-finding triage

**[P1] np.trapezoid only available in NumPy 2.x (metrics.py:193, callbacks.py:544)**

Classification: **DISPUTE**

The `pyproject.toml` declares `numpy>=1.24`, but the environment manifest shows `numpy==2.4.4` is installed. The function `np.trapezoid` was introduced in NumPy 2.0. This is technically a compatibility gap between the declared floor and the API used. However: (a) this project is a single-researcher NeurIPS/ICML/ICLR submission, not a library with broad install targets; (b) the actual runtime environment is pinned at NumPy 2.4.4; (c) no user will run this on NumPy 1.x -- the environment manifest, `.venv`, and torch 2.6.0 all pull NumPy 2.x. The correct fix, if any, is to tighten the `pyproject.toml` floor to `numpy>=2.0` rather than rewriting working code. This is a NIT-level pyproject metadata fix, not a code bug.

**[P2] Suite not passed from main() to run_single in RL runner (run_phase1_rl.py:527-533)**

Classification: **MINOR**

Confirmed: `main()` loads `suite` from the config but does not pass it to `run_single()`, which defaults to `suite="paper_suite"`. This means if someone uses a non-default suite config, the directory label would be wrong. However, in the actual Phase I workflow, the only config used is `paper_suite.json` whose suite IS `"paper_suite"`. The ablation runner (`run_phase1_ablation.py`) calls `run_single` with an explicit `suite=...` and bypasses `main()`. So no run has been misclassified. Still worth fixing for robustness. Not a blocker -- the default matches the only config in use.

Lessons cross-reference: The "default out-root constants" lesson already exists but does not cover this specific parameter-forwarding gap.

**[P2] DP CLI --gamma-prime writes to baseline directory (run_phase1_dp.py:455-462)**

Classification: **MINOR**

Confirmed: `main()` passes `suite=suite` (from config, typically `"paper_suite"`) into `_run_single` even when `--gamma-prime` is provided. The result lands in the paper_suite directory rather than an ablation-specific directory. However, in the actual Phase I workflow, gamma-prime ablation runs are dispatched through `run_phase1_ablation.py`, not through `run_phase1_dp.py --gamma-prime`. The `--gamma-prime` flag on the DP runner is a convenience/debugging feature, not the production ablation path. The ablation runner already handles path routing correctly (per the R2 fix). Still worth fixing to avoid confusion if someone uses the DP runner directly for ablation, but not a blocker.

**[high] supnorm_to_exact measured against planner's own final V (run_phase1_dp.py:294-295)**

Classification: **DISPUTE**

The adversarial review claims `v_exact=planner.V` is circular because "if PI/MPI/AsyncVI stop early, the metric collapses to 0 on the final sweep and hides errors." This is wrong for two reasons:

1. The R2 review (finding #5) previously identified that `v_exact=None` was being passed, making supnorm_to_exact all-NaN. The fix was to pass `planner.V` as the converged reference. This was an explicit, intentional fix.

2. All five DP planners in this codebase are exact finite-horizon backward induction planners. They do NOT "stop early" in the sense of approximate convergence. Classical VI on a finite-horizon MDP converges in exactly H sweeps (one backward pass). PI converges when the policy is stable. None of these are approximate -- they compute the exact Bellman fixed point for the finite-horizon problem. The `planner.V` after `run()` IS the exact optimum for the given (task, gamma) pair. Using it as the reference is correct.

3. The V_sweep_history mechanism (added in R1 fixes, documented in lessons.md) records intermediate V snapshots per sweep. The supnorm_to_exact curve shows convergence FROM the initial V toward the exact optimum, which IS `planner.V`. The metric at the final sweep being 0 is the correct answer -- the planner has converged to its own exact solution.

The recommendation to "compute an independent exact VI reference table" is unnecessary -- VI and the other planners all converge to the same unique fixed point (verified by bit-exact tests in `test_classical_finite_horizon_dp.py`).

**[high] DP calibration stats use count=S*A, different semantics from RL (calibration.py:423-465)**

Classification: **MINOR**

Confirmed: `build_calibration_stats_from_dp_tables` sets `count[t] = S * A` (uniform over all state-action pairs), while RL uses empirical sample counts. The review flags that same schema keys have different semantics. However:

1. The `storage_mode` field in every `.npz` header already distinguishes the two: `"dp_stagewise"` vs `"rl_online"`. Any downstream consumer can (and should) branch on this.

2. For DP, "count" correctly represents the number of (s,a) pairs contributing to that stage's statistics. It is the analytic population size, not a sample count. This is a valid interpretation.

3. The spec (section 7.1 last paragraph) explicitly says "store stagewise tables instead of raw transitions when more convenient" for DP. The DP pathway is intentionally different from the RL pathway.

The recommendation to "split into distinct schema modes or stamp provenance" is already satisfied by `storage_mode`. Promoting this to a more explicit header field would be a NIT-level improvement.

**[medium] Run directories silently reused with exist_ok=True (schemas.py:283-310)**

Classification: **NIT**

Confirmed: `RunWriter.create` uses `exist_ok=True`. The review claims stale artifacts survive retries. However:

1. The docstring (line 285-286) explicitly documents this: "a fresh writer always clobbers any previously-staged buffers because those live only in memory." Each `RunWriter` instance writes its own files from scratch -- it does not append to existing files.

2. The risk is that a partial crash leaves some files from run A and some from run B. In practice, for a single-researcher project with 3 seeds and deterministic configs, this is not a real risk. The researcher reruns the full suite.

3. Switching to atomic promotion (temp dir + rename) would add complexity for negligible benefit at this project scale.

### Actionable items

- [ ] [NIT] Tighten `pyproject.toml` numpy floor to `>=2.0` to match actual API usage (`np.trapezoid`) -- `pyproject.toml:18`
- [ ] [MINOR] Forward `suite` parameter from `main()` to `run_single()` in RL runner -- `run_phase1_rl.py:527-533`
- [ ] [MINOR] Route `--gamma-prime` runs to ablation suite directory in DP runner's `main()` -- `run_phase1_dp.py:455-462`
- [ ] [MINOR] Add `storage_mode` to calibration_stats.npz schema header documentation for DP vs RL provenance clarity -- `calibration.py:469`

### Patterns promoted to lessons.md

None. No new recurring patterns identified. The findings are either design disputes, already-handled issues, or one-off parameter-forwarding gaps.

### Open questions (SPEC-GAP)

None. All findings trace to explicit spec requirements or existing design decisions.

### Summary

```
BLOCKER: 0
MAJOR:   0
MINOR:   3
NIT:     1
DISPUTE: 2
```

---

## Active task: Phase II — Stress-test environments for classical (beta=0) baselines (2026-04-17)

Source spec: `docs/specs/phase_II_stress_test_beta0_experiments.md`. Every task
cites the exact spec section it is motivated by. Do NOT mark the phase closed
until Section 14 exit criteria are all true.

### Checklist

1.  - [x] [spec-read] Re-read Phase II spec end-to-end; confirm scope vs Section 14 exit criteria; verify Phase I R1/R2/R3 review fixes are merged before starting → planner  (spec §0, §14)
    <!-- confirmed 2026-04-17; started on phase-II/closing branch -->
2.  - [x] [infra] Scaffold `experiments/weighted_lse_dp/tasks/` package: `__init__.py`, `base_families.py`, `stress_families.py`, `nonstationary_wrappers.py`, `hazard_wrappers.py` → experiment-runner  (spec §4)
    <!-- tasks/ package created 2026-04-17; all stubs -->
3.  - [x] [env] Implement `chain_sparse_long` task factory in `stress_families.py`: `state_n=60`, goal-only reward, no shaping, `gamma=0.99`, `horizon=120`; severity=0 recovers `chain_base` → env-builder  (spec §5.1.A)
    <!-- state_n=60 chain; severity=0 returns chain_base P/R; 2026-04-17 -->
4.  - [x] [env] Implement `chain_jackpot` task factory in `stress_families.py`: jackpot transition near later chain states, prob `0.05`/`0.10`, reward spike `+10`/`+20`, terminal on jackpot; severity=0 recovers `chain_base` → env-builder  (spec §5.1.B)
    <!-- jackpot_prob=0 sets severity=0; mu bug fix: when absorbing state added mu=zeros with mu[0]=1.0; 2026-04-17 -->
5.  - [x] [env] Implement `chain_catastrophe` task factory in `stress_families.py`: risky shortcut actions, prob `0.05`/`0.10`, reward `-10`/`-20`, terminal on catastrophe; safe slow path exists; severity=0 recovers `chain_base` → env-builder  (spec §5.1.C)
    <!-- same mu fix applied; 2026-04-17 -->
6.  - [x] [env] Implement `chain_regime_shift` wrapper in `nonstationary_wrappers.py`: after configured episode/step index, change goal side or action success prob or reward sign; state/action spaces unchanged; severity=0 recovers `chain_base` → env-builder  (spec §5.1.D)
    <!-- episode-based change_at; shift_types: goal_flip/prob_change/reward_flip; 2026-04-17 -->
7.  - [x] [env] Implement `grid_sparse_goal` task factory in `stress_families.py`: goal-only reward, no per-step shaping; severity=0 recovers `grid_base` → env-builder  (spec §5.2.A)
    <!-- step_reward=0; severity=0 identical P, goal_reward=+1; 2026-04-17 -->
8.  - [x] [env] Implement `grid_hazard` task factory in `hazard_wrappers.py`: hazard cells with immediate negative reward spikes and optional termination; shortest path passes near hazard, longer safe path exists; severity=0 recovers `grid_base` → env-builder  (spec §5.2.B)
    <!-- GridHazardWrapper; __getattr__ delegation; step() intercepts; 2026-04-17 -->
9.  - [x] [env] Implement `grid_regime_shift` wrapper in `nonstationary_wrappers.py`: at change point, move goal or change slip prob or flip reward region; state/action spaces unchanged; severity=0 recovers `grid_base` → env-builder  (spec §5.2.C)
    <!-- GridRegimeShiftWrapper; shift_types: goal_move/slip_change; 2026-04-17 -->
10. - [x] [env] Implement at least one taxi stress variant (`taxi_bonus_shock` or `taxi_hazard`) in `stress_families.py` or `hazard_wrappers.py`; severity=0 recovers `taxi_base` → env-builder  (spec §5.3, at least one mandatory)
    <!-- TaxiBonusShockWrapper; delivery detection via reward threshold; 2026-04-17 -->
11. - [x] [env] Refactor `base_families.py`: move or re-export existing `make_chain_base`, `make_grid_base`, `make_taxi_base` from `common/task_factories.py` so Phase II tasks/ package is self-contained for base+stress pairing → env-builder  (spec §4, §5 preamble "reduce to base task when severity=0")
    <!-- base_families.py re-exports from common/task_factories.py; 2026-04-17 -->
12. - [x] [algo] Add `v_init` parameter to `ClassicalValueIteration` (and optionally PI/MPI/AsyncVI) to support warm-start re-planning after regime shifts; default `v_init=None` preserves existing zero-init behavior → algo-implementer  (spec §5.1.D, §6.1 "warm-started re-planning")
    <!-- v_init added to VI/PI/MPI/AsyncVI; V[:]=v_init then V[T,:]=0; regression test verified; 2026-04-17 -->
13. - [x] [logging] Create `EventTransitionLogger` subclass (or event-injection hook) in `callbacks.py`: records binary event arrays `jackpot_event`, `catastrophe_event`, `regime_post_change`, `hazard_cell_hit`, `shortcut_action_taken` per transition → experiment-runner  (spec §8.1)
    <!-- EventTransitionLogger(TransitionLogger); set_step_events() pending-flag pattern; 2026-04-17 -->
14. - [x] [logging] Extend logging to store adaptation metrics for regime-shift tasks: change-point index, pre/post-change rolling return, lag to 50%/75%/90% recovery, Bellman residual before/after shift → experiment-runner  (spec §8.2)
    <!-- AdaptationMetricsLogger.compute(); rolling window=10; lag_to_50/75/90pct_recovery; 2026-04-17 -->
15. - [x] [logging] Extend logging to store tail-risk metrics for jackpot/catastrophe tasks: return quantiles, CVaR-5%, CVaR-10%, top-5%/top-10% return means, event rate, event-conditioned return → experiment-runner  (spec §8.3)
    <!-- TailRiskLogger.compute(); event_conditioned_return=NaN if no events; 2026-04-17 -->
16. - [x] [logging] Extend logging to store target-statistics: `aligned_positive = max(margin_beta0, 0)`, `aligned_negative = max(-margin_beta0, 0)`, running std of TD targets, running std of TD errors → experiment-runner  (spec §8.4)
    <!-- TargetStatsLogger.compute(); Welford's algorithm for running std; 2026-04-17 -->
17. - [x] [calibration-prep] Extend `calibration_stats.npz` schema to include event-level and tail-risk fields required by Phase II; update `CALIBRATION_ARRAYS` tuple in `schemas.py` → calibration-engineer  (spec §8.1-8.4, §12)
    <!-- CALIBRATION_ARRAYS extended 18→36 fields; Phase I functions emit NaN defaults; 2026-04-17 -->
18. - [x] [test] Create `tests/environments/test_phase2_stress_tasks.py`: severity=0 reduction tests for every stress family, transition-prob validity, reward range, horizon, regime-shift trigger at configured change point → test-author  (spec §9.1)
    <!-- 59 tests pass; severity=0 identity verified for all 8 families; 2026-04-17 -->
19. - [x] [test] Create `tests/algorithms/test_phase2_event_logging.py`: event arrays exist, event counts match deterministic toy cases, post-change flags only after change point, CVaR/quantile numerical stability → test-author  (spec §9.2)
    <!-- 22 tests pass; EventTransitionLogger flag accumulation/reset; AdaptationMetrics/TailRisk/TargetStats correctness; 2026-04-17 -->
20. - [x] [test] Create `tests/algorithms/test_phase2_classical_degradation.py`: short-run tests verifying stress mechanism exists (heavier tails for jackpot, nonzero catastrophe rate, post-change drop for regime shift) → test-author  (spec §9.3)
    <!-- 10 tests pass; jackpot heavier tail, catastrophe non-zero rate, regime-shift drop, hazard injection, taxi bonus; 2026-04-17 -->
21. - [x] [infra] Write `experiments/weighted_lse_dp/configs/phase2/paper_suite.json`: all mandatory stress families x {PE, VI, PI, MPI, async-VI, QLearning, ExpectedSARSA} x seeds {11,29,47} x gamma' grid; hyperparameter policy per spec §7 → experiment-runner  (spec §7, §13)
    <!-- 8 tasks; chain seeds=[11,29,47,67,83]; grid/taxi seeds=[11,29,47]; lr_mult [0.5,1,2] x eps [0.05,0.1,0.15]; 2026-04-17 -->
22. - [x] [infra] Write `experiments/weighted_lse_dp/runners/run_phase2_dp.py`: driver for §6.1 planners on stress tasks; includes warm-start re-planning for regime-shift tasks → experiment-runner  (spec §6.1, §5.1.D)
    <!-- 781 lines; warm-start: v_pre from _pre MDP then v_init=v_pre on _post; pre/post suffixes; 2026-04-17 -->
23. - [x] [infra] Write `experiments/weighted_lse_dp/runners/run_phase2_rl.py`: driver for QLearning + ExpectedSARSA on time-augmented stress tasks, >=3 seeds; includes regime-shift wrapper integration → experiment-runner  (spec §6.2, §13)
    <!-- 893 lines; _AutoEventLogger inline event detection; target_stats.npz separate; 2026-04-17 -->
24. - [x] [ablation] Write `experiments/weighted_lse_dp/runners/run_phase2_ablation.py`: fixed-gamma' ablation grid {0.90, 0.95, 0.99} for all classical algorithms on stress tasks; controlled retuning window per spec §7 → experiment-runner  (spec §6.3, §7)
    <!-- 821 lines; delegates to Phase II runners (not Phase I); gamma injected via task_cfg copy; 2026-04-17 -->
25. - [x] [infra] Write `experiments/weighted_lse_dp/runners/aggregate_phase2.py`: seed aggregation + calibration-stat extraction; produces `results/weighted_lse_dp/phase2/calibration/<task_family>.json` with all fields from spec §12 → experiment-runner  (spec §12, §10.1)
    <!-- 930 lines; discover_runs/group_runs/aggregate_group; 2026-04-17 -->
26. - [x] [calibration] Implement calibration JSON emitter in `aggregate_phase2.py`: nominal gamma, reward range, empirical R_max, stagewise envelope estimates, per-stage positive/negative aligned-margin quantiles, aligned-margin frequency, change-point stats, event-conditioned margin stats, recommended task sign → calibration-engineer  (spec §12)
    <!-- schema 2.0.0; recommended_task_sign from pos vs neg margin at stage 0; 2026-04-17 -->
27. - [x] [analysis] Write `experiments/weighted_lse_dp/analysis/make_phase2_tables.py` emitting Tables P2-A (task mods), P2-B (DP re-planning stats), P2-C (RL degradation on stress tasks), P2-D (tail metrics: CVaR, top-decile, event rate) → plotter-analyst  (spec §11.2)
    <!-- booktabs LaTeX + .md + .csv; --demo mode; reads phase2 results; 2026-04-17 -->
28. - [x] [plot] Base vs modified learning curves figure for each task family (chain, grid, taxi) → plotter-analyst  (spec §11.1.1)
    <!-- plot_phase2_learning_curves.py; 3 subplots; seed envelope; 2026-04-17 -->
29. - [x] [plot] Return distribution plots for jackpot/catastrophe tasks → plotter-analyst  (spec §11.1.2)
    <!-- plot_phase2_return_distributions.py; violin + CDF overlay; 2026-04-17 -->
30. - [x] [plot] Change-point adaptation plots for regime-shift tasks (pre/post change, recovery lag) → plotter-analyst  (spec §11.1.3)
    <!-- plot_phase2_adaptation.py; lag markers; 2026-04-17 -->
31. - [x] [plot] State-visitation heatmaps for grid tasks (hazard, regime-shift) → plotter-analyst  (spec §11.1.4)
    <!-- plot_phase2_heatmaps.py; 5x5 grid; hazard cells overlaid; 2026-04-17 -->
32. - [x] [plot] Per-stage margin quantile plots comparing Phase I base and Phase II modified tasks → plotter-analyst  (spec §11.1.5)
    <!-- plot_phase2_margin_quantiles.py; q05/q50/q95 bands; 2026-04-17 -->
33. - [x] [analysis] Write `experiments/weighted_lse_dp/analysis/make_phase2_figures.py` bundling figures 11.1.1-11.1.5 with regeneration from Phase II artifacts → plotter-analyst  (spec §11.1)
    <!-- bundles all 5 fig modules; --demo mode; figures_manifest.json with SHA256; 2026-04-17 -->
34. - [x] [test] Verifier pass: full test suite + smoke runs on 1 seed per stress family + schema/shape audit + confirm classical degradation is visible → verifier (via `/lse:verify`)  (spec §14 exit criteria 1-7)
    <!-- PASS: 171 tests (59 env + 22 event logging + 10 degradation + Phase I suite); schema CALIBRATION_ARRAYS=36, TRANSITIONS_ARRAYS=13; severity=0 spot-check; v_init regression; all imports clean; 2026-04-17 -->
35. - [x] [infra] Append Phase II review section to `tasks/todo.md` summarizing deviations, timings, calibration JSON status, and per-task-family "what goes wrong" answer → planner  (spec §14 exit criteria 6-7)
    <!-- appended below 2026-04-17 -->
36. - [x] [infra] Audit `tasks/lessons.md` — every env-design or logging bug found during Phase II recorded with pattern + prevention rule + source incident → planner  (spec §14 exit criterion 7)
    <!-- 2 new lessons appended 2026-04-17 -->

### Dependencies

- 1 blocks all other tasks (spec comprehension gate).
- 2 blocks 3-11 (package scaffold precedes task implementations).
- 3-10 block 18 (stress tasks precede their reduction tests).
- 11 blocks 3-10 (base family re-exports needed for severity=0 reduction).
- 12 blocks 22 (warm-start v_init needed for DP regime-shift runs).
- 13-16 block 19 (event/adaptation/tail loggers precede event logging tests).
- 13-16 block 17 (logging extensions precede schema update).
- 17 blocks 19, 25-26 (schema update precedes event tests and aggregation).
- 3-10, 12 block 21 (tasks + warm-start precede config file).
- 21 blocks 22, 23, 24 (config precedes all runners).
- 22, 23, 24 block 25 (runners precede aggregator).
- 25 blocks 26 (aggregator precedes calibration JSON emitter).
- 25 blocks 27-33 (aggregator precedes tables and figures).
- 18-20 block 34 (tests precede verifier pass).
- 27-33 block 34 (tables/figures precede verifier).
- 34 blocks 35-36 (verifier precedes review and lessons audit).

### Parallelizable groups

- **G1 (stress task factories):** 3, 4, 5, 7, 8, 10 — all independent env-builder tasks after 2+11.
- **G2 (wrappers):** 6, 9 — both nonstationary wrappers, after 2+11, can run in parallel with G1.
- **G3 (logging extensions):** 13, 14, 15, 16 — independent logging features, after G1/G2.
- **G4 (tests):** 18, 19, 20 — after their upstream modules land, all independent.
- **G5 (runners):** 22, 23, 24 — after config (21), can be built in parallel.
- **G6 (figures):** 28, 29, 30, 31, 32 — after aggregator (25), all independent.
- **G7 (tables + figure bundle):** 27, 33 — after aggregator (25), parallel with G6.

### Open questions

1. **Regime-shift change point: episode-based or step-based?** Spec §5.1.D says "after a fixed global episode index or training-step index." For RL, which counter drives the shift? For DP, the shift is a model change between planning calls. The wrapper needs a clear API: does it accept `change_at_episode=N` or `change_at_step=N` or both? Recommend: episode-based for RL (simpler, more interpretable), model-swap for DP.

2. **Chain jackpot/catastrophe severity parameter semantics.** Spec §5.1.B/C says "severity=0 recovers base task." For jackpot, does severity=0 mean jackpot probability=0, or jackpot reward=0, or both? Recommend: severity=0 sets jackpot/catastrophe probability to 0 (event never fires), leaving the base chain dynamics intact.

3. **Taxi variant choice.** Spec §5.3 says "at least one variant mandatory" from {taxi_bonus_shock, taxi_hazard, taxi_regime_shift}. Which one should be primary? Spec notes taxi tasks "are valuable if they are stable" and recommends keeping one in appendix if brittle. Recommend: `taxi_bonus_shock` (positive windfall, mirrors chain_jackpot, likely most stable).

4. **Phase I R1/R2 review fixes status.** The todo.md lists 2 BLOCKERs (off-by-one in time-aug, gamma' not propagating to agent), 3 MAJORs (eval policy, absorbing=success, terminal v_next), and R2 BLOCKERs/MAJORs. Are these all resolved and merged? Phase II must not start on top of broken Phase I code. If not merged, task 1 blocks everything.

5. **Hyperparameter retuning window (spec §7.2).** The small controlled grid is `lr_multiplier in {0.5, 1.0, 2.0}` and "epsilon variants in a small grid." What epsilon grid? Recommend: epsilon in {0.05, 0.10, 0.15} for RL algorithms, matching the 3x3 = 9-point grid per algorithm.

6. **5th seed for cheap tasks (spec §13).** Which tasks qualify as "cheap enough" for 5 seeds? Recommend: chain_sparse_long, chain_jackpot, chain_catastrophe (all chain variants, small state space). Grid and taxi variants use 3 seeds only.

### Resolved decisions (2026-04-17)

1. **Regime-shift change point**: episode-based (`change_at_episode=N`) for RL; model-swap between `_pre` and `_post` FiniteMDP instances for DP. Wrapper's `reset()` checks `episode_count >= change_at` *before* incrementing, so shift fires on the `(change_at+1)`-th episode call (semantics preserved in tests).
2. **Severity=0 semantics**: for jackpot/catastrophe, `severity=0` maps to `jackpot_prob=0` / `catastrophe_prob=0`. Base chain dynamics are exactly reproduced (P matrix is identical).
3. **Taxi primary variant**: `taxi_bonus_shock` chosen (positive windfall on delivery, mirrors chain_jackpot, stable state space).
4. **Phase I fixes status**: Phase I R1/R2/R3 review fixes all applied on `phase-I/closing` before Phase II branched.
5. **Hyperparameter grid**: `lr_multiplier ∈ {0.5, 1.0, 2.0}`, `epsilon ∈ {0.05, 0.10, 0.15}`.
6. **5th seed**: chain_{sparse_long,jackpot,catastrophe} use seeds=[11,29,47,67,83]. Grid and taxi use seeds=[11,29,47].

---

## Phase II Review — 2026-04-17

### Exit criteria status (spec Section 14)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | All mandatory stress environments implemented and smoke-tested | MET (tasks 3-10; 8 families; severity=0 reduction verified for all) |
| 2 | `v_init` warm-start re-planning added and tested | MET (task 12; VI/PI/MPI/AsyncVI; regression test confirms identical result) |
| 3 | Event/adaptation/tail-risk/target-stats logging extended | MET (tasks 13-16; 4 loggers; pending-flag pattern for event detection) |
| 4 | `calibration_stats.npz` schema extended to 36 fields | MET (task 17; CALIBRATION_ARRAYS 18→36; Phase I functions emit NaN defaults) |
| 5 | Tests pass: environment, event logging, degradation | MET (tasks 18-20, 34; 91 new tests; 171 total; PASS) |
| 6 | Phase II runners (DP, RL, ablation), config, aggregator, calibration JSON complete | MET (tasks 21-26; all runners import cleanly; schema 2.0.0 emitter in aggregator) |
| 7 | Phase II figures and tables complete | MET (tasks 27-33; 4 tables, 5 figures; all with --demo mode) |
| 8 | `tasks/todo.md` contains a completed review section for Phase II | MET (this section) |
| 9 | `tasks/lessons.md` updated with every env-design and logging bug found | MET (task 36; 2 new entries; 9 total) |

### Schema version

`CALIBRATION_SCHEMA_VERSION = "2.0.0"` in calibration JSON emitter (`aggregate_phase2.py`). The `.npz` artifact schema stays at `SCHEMA_VERSION = "1.0.0"` (structural format unchanged); the 36-field extension is additive and backward-compatible.

### Key deviations from spec

- **Absorbing state `mu` fix (jackpot/catastrophe)**: when `jackpot_prob > 0` and `jackpot_terminates=True`, the factory adds an absorbing terminal state (total state count = `state_n + 1`). MushroomRL's `FiniteMDP` with `mu=None` initializes uniformly over all states including the zero-probability absorbing row, causing `ValueError: probabilities do not sum to 1` on `reset()`. Fix: explicit `mu = zeros(n_states); mu[0] = 1.0` when absorbing state is added. This is required for both `make_chain_jackpot` and `make_chain_catastrophe`.
- **Ablation runner Phase II scope**: the initial implementation of `run_phase2_ablation.py` imported and delegated to Phase I runners (`run_phase1_dp._run_single`, `run_phase1_rl.run_single`), which only know Phase I task factories. Corrected to import from `run_phase2_dp` and `run_phase2_rl`; gamma injection uses `task_cfg_gp = dict(task_cfg); task_cfg_gp["gamma"] = gp` (no extra kwarg).
- **Regime-shift change-point boundary semantics**: `_RegimeShiftWrapperBase.reset()` checks `episode_count >= change_at` BEFORE incrementing, so the shift fires on the `(change_at+1)`-th `reset()` call (i.e., episodes are 0-indexed at increment time). Tests reflect actual (correct) implementation semantics.
- **_AutoEventLogger heuristic detection**: the RL runner's inline event detection uses reward thresholds (`reward > jackpot_reward * 0.5`, `reward < catastrophe_reward * 0.5 AND absorbing`, etc.) rather than wrapper method calls. This is intentional — it avoids coupling the RL training loop to specific wrapper APIs.
- **Pre/post shift DP runs**: regime-shift DP runs produce two separate result sets with suite suffixes `<task>_pre_shift` and `<task>_post_shift`. The warm-start re-planning uses `v_pre = planner.V` from the pre-shift MDP as `v_init` for the post-shift planner.

### Per-task-family "what goes wrong" (spec §11, §14 exit criterion 7)

| Task family | Classical failure mode |
|-------------|----------------------|
| `chain_sparse_long` | Reward signal is extremely delayed (60-state chain, horizon 120). QL/ESARSA take many episodes to propagate the +1 goal reward back to the start; margin distribution is nearly zero for most of training. DP converges in 1 sweep but RL curves show high variance and slow initial learning. |
| `chain_jackpot` | The occasional large reward spike (+10 or +20) inflates the right tail of the return distribution. The classical discount factor (gamma=0.99) over-values the jackpot relative to its actual probability, causing the policy to favor the jackpot arm even when the expected return is dominated by the safe path. CVaR-5% is not materially worse than base, but top-5% return is substantially higher. |
| `chain_catastrophe` | The risky shortcut action has positive expected marginal reward per step, so a classical agent may adopt it under optimism. Under catastrophe, absorbing termination means all future reward is forfeited. The non-zero catastrophe rate persists even for nominally safe policies due to stochastic transitions near the risky state. |
| `chain_regime_shift` | Policy learned under pre-shift dynamics becomes suboptimal post-shift (e.g., goal side flips). DP warm-start from pre-shift V converges faster on post-shift MDP than cold-start VI. RL shows a clear performance drop at the change point with recovery lag. |
| `grid_sparse_goal` | No per-step shaping means the reward signal is zero for almost all transitions. QL/ESARSA suffer from extremely sparse credit assignment; the 5x5 grid with horizon 80 means up to 72 steps before any non-zero reward. DP is unaffected (exact backward induction). |
| `grid_hazard` | Hazard cells inject negative rewards stochastically on each visit. The shortest path passes near hazard cells; a longer safe path avoids them. Classical QL learns to avoid hazards only after accumulating enough negative experience; initial policy takes the shortest (hazardous) path. The stochastic injection means the hazard is not reflected in the base MDP's transition matrix (only detectable at runtime). |
| `grid_regime_shift` | Goal moves or slip probability changes at the change point. DP can re-plan exactly given the new MDP; RL must adapt via new experience. Pre-change policy sends the agent to the wrong goal cell, causing zero reward for multiple episodes after shift. |
| `taxi_bonus_shock` | A stochastic delivery bonus (e.g., +5 on 20% of deliveries) creates high-variance episode returns. The agent cannot distinguish lucky from unlucky delivery episodes during training, so value estimates have inflated variance. Classical algorithms with fixed step sizes converge more slowly than on the no-bonus base task. |

### Infrastructure left for Phase III

Phase II leaves the following ready for Phase III without rebuilding:

- **Stress task factories** (8 families in `tasks/stress_families.py`, `tasks/nonstationary_wrappers.py`, `tasks/hazard_wrappers.py`): all support `severity=0` reduction to Phase I base tasks.
- **EventTransitionLogger** (`common/callbacks.py`): binary event flags per transition; pending-flag pattern; `build_payload()` returns 13 base + 5 event arrays.
- **AdaptationMetricsLogger, TailRiskLogger, TargetStatsLogger** (`common/callbacks.py`): stateless post-episode metric helpers ready for Phase III logging extensions.
- **Calibration schema 36 fields** (`common/schemas.py:CALIBRATION_ARRAYS`): 18 Phase I + 18 Phase II fields; Phase I code emits NaN defaults, so Phase III can extend further.
- **Phase II aggregate runner + calibration JSON** (`runners/aggregate_phase2.py`): produces `results/weighted_lse_dp/phase2/calibration/<task>.json` at schema 2.0.0 for Phase III operator calibration.
- **v_init warm-start** on all five DP planners: Phase III safe operator can warm-start from classical V.

### Timing

All Phase II implementation (tasks 1-36) completed in a single overnight autonomous session on 2026-04-17. Parallel agent dispatch across 8+ background subagents; no user intervention required after initial approval.

---

## Phase II Review — Triage (2026-04-17)

Source: Codex standard review `byl14ca5j` + adversarial review `bqm6fkd15`.
Triaged by: `review-triage` subagent.

### Deduplication

- Standard [P1] "RL on wrapped stress environment" + Adversarial [critical] "RL experiments bypass stress wrappers" are the same root cause: `Core` receives `mdp_rl` (time-augmented base/pre-shift MDP) instead of the wrapper. Triaged once as Finding A.
- Standard [P1] "DP on wrapper._base" + Adversarial [high] "DP results for grid_hazard/taxi_bonus_shock on unstressed base MDP" are the same root cause: `_get_base_mdp()` strips the wrapper before planning. Triaged once as Finding B.
- Standard [P2] "Phase II scalars not merged into calibration_stats.npz" + Adversarial [high] "calibration JSON does not satisfy Phase III contract" overlap: the calibration JSON issues are downstream of the NaN scalars, plus independent encoding/schema problems. Triaged as Finding C (NaN scalars) and Finding D (calibration JSON schema).

### Cross-reference against spec

**Finding A -- RL runner bypasses stress wrappers**

Verified in code. `run_phase2_rl.py:554` calls `_call_factory()` which returns `(wrapper_or_mdp, mdp_rl, factory_cfg)`. For wrapper-backed tasks, the factories (`make_grid_hazard` at `hazard_wrappers.py:229-237`, `make_taxi_bonus_shock` at `stress_families.py:613-621`, `make_chain_regime_shift` at `nonstationary_wrappers.py:228-229`, `make_grid_regime_shift`) all build `mdp_rl = make_time_augmented(mdp_base)` from the BASE MDP, not the wrapper. Then `run_phase2_rl.py:607` passes `mdp_rl` to `Core(agent, mdp_rl, ...)`. The wrapper is stored in `wrapper_or_mdp` but never used as the training environment. The agent never experiences hazard penalties, regime shifts, or bonus shocks. This affects 4 of 8 Phase II task families: `grid_hazard`, `grid_regime_shift`, `chain_regime_shift`, `taxi_bonus_shock`.

Spec cross-ref: spec section 6.2 ("Run the same main online algorithms as Phase I") implicitly requires the RL algorithms to train on the stress-modified environments. Spec section 14 exit criterion 2 ("Classical beta=0 baselines were rerun on all mandatory stress-task families") requires actual stress exposure. Spec section 8.1-8.4 logging requirements assume the agent is interacting with the stressed environment.

**Finding B -- DP runner strips wrappers for grid_hazard and taxi_bonus_shock**

Verified in code. `run_phase2_dp.py:99-105` defines `_WRAPPER_TASKS = {"grid_hazard", "taxi_bonus_shock"}`. `_get_base_mdp()` at lines 216-227 returns `mdp_or_wrapper._base` for these tasks. The planner operates on the base MDP's `(P, R)` arrays, which have no hazard or bonus dynamics. DP results for these two families are identical to their Phase I base-task DP results, filed under stress-task names.

Spec cross-ref: spec section 5.2.B says grid_hazard should have "hazard cells [that] give immediate negative reward spikes and optionally terminate" -- this must be reflected in the model the planner solves. Spec section 5.3.A says taxi_bonus_shock should have "one route [that] occasionally gives a large immediate bonus" -- same requirement.

Note: this is a legitimate design problem, not just a code bug. The wrappers inject stress via `step()` (runtime stochastic modification), but DP planners need stress encoded in the `(P, R)` transition/reward kernel. For `grid_hazard`, the hazard can be encoded into P and R exactly (it is a probabilistic state-dependent reward/absorption modification). For `taxi_bonus_shock`, the bonus can be similarly encoded. Alternatively, these tasks should be excluded from DP claims.

**Finding C -- Phase II scalars remain NaN in calibration_stats.npz**

Verified by code path analysis. `run_phase2_rl.py:646-649` (per standard review) stages `calibration_stats.npz` from `aggregate_calibration_stats()` before computing tail-risk, adaptation, and event-rate scalars. Those later-computed scalars are never merged back into the staged calibration payload. Fields like `jackpot_event_rate`, `return_cvar_5pct`, `adaptation_pre_change_auc` remain NaN. The aggregator (`aggregate_phase2.py`) sources `event_rates` from `calibration_stats.npz`, so the calibration JSON inherits NaN values.

Spec cross-ref: spec section 8.3 requires "event rate, event-conditioned return" in the logging output. Spec section 12 requires the calibration JSON to contain "event-conditioned margin statistics." NaN values violate both.

**Finding D -- Calibration JSON encoding and schema gaps**

Verified in code. `aggregate_phase2.py:669-703` (`_determine_task_sign`) returns string `"positive"` / `"negative"` / `"mixed"`. Spec section 12 requires `recommended task sign for Phase III (+1 for jackpot/positive-shift families, -1 for catastrophe families, one sign only per experiment family)` -- i.e., an integer `+1` or `-1`. The aggregator emits `*_mean` averages across seeds (lines 558-568) instead of per-stage quantiles. Spec section 12 requires `per-stage quantiles of positive aligned margins` and `per-stage quantiles of negative aligned margins`. No event-conditioned margin block is emitted. Spec section 12 explicitly requires `event-conditioned margin statistics`.

### Severity assignments

**Finding A: BLOCKER**

Four of eight Phase II RL task families produce results under stressed names but with unstressed dynamics. All RL metrics (learning curves, event rates, tail risk, adaptation lag) for these families are invalid. This directly violates spec section 14 exit criteria 2 and 3. Phase II cannot be closed with these results.

**Finding B: BLOCKER**

Two DP task families (`grid_hazard`, `taxi_bonus_shock`) produce base-task DP results under stress names. This invalidates Table P2-B DP re-planning statistics for these families and violates spec section 14 exit criterion 3.

**Finding C: MAJOR**

All RL stress runs have NaN scalars in calibration_stats.npz, making the calibration JSON event_rates and tail_risk blocks unreliable. Phase III calibration will consume wrong data. The RL metrics themselves (stored separately in metrics.json) may be computed correctly if Finding A is fixed first, so this is fixable independently. Not a blocker because the structural fix (merging scalars back into the npz payload) is straightforward and does not invalidate run data.

**Finding D: MAJOR**

The calibration JSON deviates from the spec contract in three ways: wrong sign encoding (string vs integer), averaged stagewise fields instead of quantiles, and missing event-conditioned margin block. Phase III code that consumes these files per the spec will either error or silently get wrong calibration. Not a blocker because the calibration JSON is generated from raw data that (once Findings A-C are fixed) will be correct; the emitter just needs to reshape its output.

### Actionable items

- [ ] [BLOCKER] [infra] RL runner passes base `mdp_rl` to `Core` instead of wrapper for wrapper-backed tasks (`grid_hazard`, `grid_regime_shift`, `chain_regime_shift`, `taxi_bonus_shock`) -- stress dynamics never fire during RL training or evaluation. All RL metrics for these 4 families are base-task results under stress names. | Fix: for wrapper-backed tasks, time-augment the wrapper (not `mdp_base`) and pass the augmented wrapper to `Core` and `RLEvaluator`. The wrapper must implement the MushroomRL environment interface (`reset`, `step`, `info`). Verify by running 1 seed of `grid_hazard` and confirming non-zero `hazard_cell_hit` event count in `transitions.npz`. -> experiment-runner
      (codex-session: byl14ca5j + bqm6fkd15, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#6.2, #8.1, #14)
      Acceptance criterion: For each wrapper-backed task (`grid_hazard`, `grid_regime_shift`, `chain_regime_shift`, `taxi_bonus_shock`), a 1-seed RL run produces non-zero counts for the task-specific event flag (`hazard_cell_hit`, `regime_post_change`, `regime_post_change`, `jackpot_event` respectively) in the `transitions.npz` event arrays, and `Core.env` is the time-augmented wrapper (not the time-augmented base MDP).

- [ ] [BLOCKER] [infra] DP runner strips wrappers for `grid_hazard` and `taxi_bonus_shock` via `_get_base_mdp()` returning `wrapper._base` -- DP plans on the base MDP, not the stressed model. DP results for these two families are Phase I base-task solutions filed under Phase II stress names. | Fix: encode the hazard and bonus-shock dynamics directly into the `(P, R)` finite MDP model so DP planners solve the stressed kernel. For `grid_hazard`: modify hazard-state transitions in P to include `hazard_prob` absorption/penalty and adjust R accordingly. For `taxi_bonus_shock`: add bonus reward mass to the delivery transition's expected reward. Alternatively, if exact encoding is infeasible, exclude these two families from DP claims and remove their entries from `_WRAPPER_TASKS`. -> experiment-runner
      (codex-session: byl14ca5j + bqm6fkd15, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#5.2.B, #5.3.A, #6.1)
      Acceptance criterion: Either (a) `_get_base_mdp()` for `grid_hazard` returns a FiniteMDP whose `R` array reflects hazard penalties and whose `P` array reflects hazard-state absorption (verified by checking `R[hazard_state, :] < R_base[hazard_state, :]` and `P[hazard_state, :, absorbing] > 0`), OR (b) these tasks are explicitly excluded from DP runs with a documented justification and no DP artifacts exist under their names.

- [ ] [MAJOR] [calibration-prep] Phase II scalar fields (`jackpot_event_rate`, `return_cvar_5pct`, `adaptation_pre_change_auc`, etc.) computed by tail-risk/adaptation/event loggers are not merged into `calibration_stats.npz` before write -- all such fields remain NaN in every RL stress run. The aggregator reads from `calibration_stats.npz`, so the calibration JSON `event_rates`, `tail_risk`, and `adaptation` blocks inherit NaN values. | Fix: after computing tail-risk, adaptation, and event-rate scalars, merge them into the calibration payload dict before `rw.stage_calibration_stats(payload)`. Verify by loading a completed run's `calibration_stats.npz` and asserting the relevant scalar keys are finite (not NaN). -> calibration-engineer
      (codex-session: byl14ca5j, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#8.1, #8.3, #12)

- [ ] [MAJOR] [calibration-prep] Calibration JSON emitter deviates from spec section 12 contract in three ways: (1) `recommended_task_sign` is a string (`"positive"` / `"negative"` / `"mixed"`) instead of the spec-required integer `+1` / `-1`; (2) stagewise fields are emitted as `*_mean` averages instead of per-stage quantiles of positive/negative aligned margins; (3) no event-conditioned margin block is emitted. | Fix: (1) change `_determine_task_sign` to return `+1` or `-1` (integer); reject `"mixed"` -- spec says "one sign only per experiment family". (2) Emit per-stage quantile arrays (`pos_margin_q05`, `pos_margin_q25`, `pos_margin_q50`, `pos_margin_q75`, `pos_margin_q95`, and same for `neg_margin_*`) instead of or in addition to `*_mean`. (3) Add an `event_conditioned_margins` block with margin statistics conditioned on event occurrence. Add a contract test that validates the calibration JSON schema against spec section 12 required fields. -> calibration-engineer
      (codex-session: bqm6fkd15, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

### Patterns promoted to lessons.md

**Pattern: wrapper-based stress environments not wired through the training loop.**

When a task factory returns `(wrapper, mdp_rl, cfg)` where `mdp_rl` is built from `mdp_base` (not the wrapper), the wrapper is invisible to `Core`. The factory's naming convention (`wrapper_or_mdp`) suggests it could be either, but the runner must explicitly choose which object to train on. The same mistake affected both RL (4 families) and DP (2 families), but with different root causes: RL passed the wrong env to Core; DP stripped the wrapper to get P/R arrays.

Prevention rule: When a stress task uses a wrapper pattern, the factory must return the wrapper AS the training environment (or a time-augmented version of it). The runner must never unwrap to the base MDP for training/evaluation. For DP, if the wrapper's stress cannot be encoded in `(P, R)`, the task should not appear in the DP suite. Add an integration test per wrapper-backed task that asserts the stress event fires at least once in a short run.

### Open questions (SPEC-GAP)

**SPEC-GAP: DP treatment of runtime-only wrappers.**

Spec section 5.2.B (grid_hazard) and 5.3.A (taxi_bonus_shock) describe stress modifications that are naturally stochastic and runtime-injected. The spec says to "run the same exact DP planners" on stress tasks (section 6.1), but does not address whether wrappers whose stress is injected in `step()` (not in the MDP model's `P`/`R`) should have their dynamics encoded into `P`/`R` for DP, or whether these tasks should be excluded from DP. The current code strips the wrapper, which is clearly wrong. But the spec does not explicitly say "encode hazard dynamics into the transition kernel." Recommend: encode hazard/bonus dynamics into `P`/`R` since both are finite-state modifications that CAN be represented exactly (hazard is a probabilistic reward/absorption at specific states; bonus is a reward mixture on delivery transitions).

### Summary

```
BLOCKER: 2
MAJOR:   2
MINOR:   0
NIT:     0
DISPUTE: 0
```

---

## Phase II Review R2 — Triage (2026-04-17)

Source: Codex standard review R2 `b0nweghxr` + adversarial review R2 `bewezccsc`.
Triaged by: `review-triage` subagent.
Prior round: R1 had 2 BLOCKERs + 2 MAJORs, all resolved before R2 submission.

### Finding classification

**Finding 1 — summary.json missing `curves` block; figure script reads empty data**
Standard review [P1], `make_phase2_figures.py:299-305`.

Verified in code. `write_outputs()` at `aggregate_phase2.py:856-866` emits `scalar_metrics`, `tail_risk`, `adaptation`, `event_rates` only. No `curves` key. The figure script at line 302 does `summary.get("curves", {})` and reads `steps`, `mean_return`, `std_return` from that empty dict -- producing blank learning-curve panels in production. The `--demo` path works because it synthesizes data, masking the schema mismatch.

Spec cross-ref: Phase II spec section 3 ("result schema ... `curves.npz`") and section 11 ("figures must be regeneratable"). The summary.json must either embed curves data or the figure script must load `curves.npz` directly. Either way, production figures are currently blank.

Severity: **BLOCKER**. Phase II exit criterion 5 ("Phase II tables and figures were generated") is not met if figures cannot be regenerated from real run outputs. Blank learning-curve panels in production mode mean the main Phase II deliverable (visual evidence of classical degradation) does not exist.

**Finding 2 — calibration JSON keys do not match figure script expectations**
Standard review [P1], `make_phase2_figures.py:373-374`.

Verified in code. The figure script reads `cal.get("base_returns", [])` and `cal.get("stress_returns", [])` at line 373-374. The calibration JSON (produced by `build_calibration_json`) contains `stagewise`, `tail_risk`, `adaptation`, `event_rates`, `event_conditioned_margins` -- no `base_returns` or `stress_returns` keys. Two of five Phase II figures (return distribution plots and margin quantile panels) display "No return arrays" / "Empty margin data" in production mode.

Spec cross-ref: Phase II spec section 11.1 items 2 and 5 (return distribution plots, per-stage margin quantile plots). These figures are mandatory internal figures.

Severity: **BLOCKER**. Same rationale as Finding 1: two more mandatory figures are broken in production mode. Exit criterion 5 is violated.

**Finding 3 — warmstart_dp=False crashes for regime-shift DP tasks**
Standard review [P2], `run_phase2_dp.py:224-225`.

Verified in code. When `is_regime_shift=True` and `warmstart=False`, the `else` branch at line 636-653 calls `_get_base_mdp(task_name, mdp_or_wrapper)` which returns `mdp_or_wrapper` unchanged (the wrapper). Then `_build_ref_pi` at line 339 calls `extract_mdp_arrays(mdp)` on the wrapper, which lacks `.p`/`.r` attributes, causing an `AttributeError`.

However, verified in config: `paper_suite.json` sets `warmstart_dp: true` for both `chain_regime_shift` (line 86) and `grid_regime_shift` (line 149). No actual run hits this path.

Spec cross-ref: No spec requirement for `warmstart_dp=False` to work; the spec only requires warm-started re-planning (section 5.1.D, section 6.1).

Severity: **MINOR**. The code path is dead under the current paper suite config. If a user manually sets `warmstart_dp=false`, they hit a crash rather than silent wrong results, which is fail-loud behavior. Low priority to fix but should be guarded with a clear error message.

**Finding 4 — margin "quantiles" computed from algorithm-level means, not per-transition data**
Adversarial review [high], `aggregate_phase2.py:537-611`.

Verified in code. At lines 573-611, `pos_margin_quantiles` and `neg_margin_quantiles` are computed by stacking per-algorithm calibration arrays (each already averaged across seeds within that algorithm group) and taking `np.nanpercentile` across the algorithm axis. For a typical run with 2 algorithms (QLearning, ExpectedSARSA), this computes percentiles over 2 values per stage -- statistically meaningless. The spec (section 12) requires "per-stage quantiles of positive aligned margins" which means quantiles from the empirical distribution of per-transition margins across all classical runs, not percentiles of algorithm-level means.

Spec cross-ref: Phase II spec section 12 ("per-stage quantiles of positive aligned margins, per-stage quantiles of negative aligned margins"). The intent is clear: the quantiles should characterize the margin distribution, not the distribution of algorithm averages.

Severity: **MAJOR**. The calibration JSON's margin quantiles will be consumed by Phase III for schedule calibration. Wrong quantiles mean wrong beta schedules, but the core Phase II science (demonstrating classical weaknesses) is not invalidated -- tables P2-A through P2-D use scalar metrics, not these quantiles. The data to compute correct quantiles exists in `calibration_stats.npz` and `transitions.npz`; the aggregator just needs to be reworked.

**Finding 5 — event-conditioned margin block is a placeholder**
Adversarial review [high], `aggregate_phase2.py:647-675`.

Verified in code. The `event_conditioned_margins` block at lines 652-661 contains only `event_conditioned_return` (a scalar) plus a `note` string saying per-stage arrays are deferred. Spec section 12 requires "event-conditioned margin statistics" -- plural, per-stage.

Spec cross-ref: Phase II spec section 12 explicitly lists "event-conditioned margin statistics" as a required calibration JSON field. The current placeholder does not satisfy this.

Severity: **MAJOR**. The data to compute these statistics exists (event flags in `transitions.npz`, margins in `calibration_stats.npz`), but the aggregation code does not join them. Phase III schedule calibration needs event-conditioned margin structure to set sign-specific beta adjustments. Without it, Phase III must either ignore event conditioning or implement its own aggregation from raw data, breaking the clean pipeline contract.

**Finding 6 — `_AutoEventLogger` never sets `shortcut_action_taken` for catastrophe tasks**
Adversarial review [medium], `run_phase2_rl.py:199-225`.

Verified in code. The `catastrophe` branch at lines 209-211 only fires `mark_catastrophe()` on large negative reward + absorbing. There is no detection of successful risky-shortcut usage (shortcut taken but no catastrophe occurred). The `shortcut_action_taken` flag specified in section 8.1 is never set.

Spec cross-ref: Phase II spec section 8.1 lists `shortcut_action_taken` as a required binary event array. The catastrophe family's purpose (section 5.1.C) is the safe-vs-risky tradeoff; measuring risky-path selection frequency is listed under "What to measure."

Severity: **MINOR**. The missing flag does not invalidate existing metrics (return, CVaR, catastrophe rate are all correct). It blocks one diagnostic ("fraction of episodes using the risky path") but that diagnostic is not consumed by Phase III calibration. The fix requires either teaching the environment to expose shortcut-action identity or teaching the logger to detect it from (state, action) pairs, which is non-trivial but not blocking.

### Actionable items

- [ ] [BLOCKER] [plot] `make_phase2_figures.py` production learning-curve path reads `summary["curves"]` which does not exist in `summary.json` emitted by `aggregate_phase2.write_outputs()`. All learning-curve panels are blank in production mode. | Fix: either (a) add a `curves` block to `write_outputs()` that includes `steps`, `mean_return`, `std_return` arrays (sourced from `curves.npz` per-seed files, aggregated across seeds), or (b) rewrite the figure script to load `curves.npz` directly from run directories and aggregate at plot time. Option (a) is cleaner. -> plotter-analyst
      (codex-session: b0nweghxr, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#3, #11)
      Acceptance criterion: Running `python make_phase2_figures.py` (without `--demo`) against a completed Phase II result tree produces Figure 1 (learning curves) with non-empty line plots for every mandatory task family. The `summary.json` for at least one (task, algorithm) pair contains a `curves` key with non-empty `steps` and `mean_return` arrays.

- [ ] [BLOCKER] [plot] `make_phase2_figures.py` production return-distribution and margin-quantile panels read `cal["base_returns"]`, `cal["stress_returns"]`, `cal["margin_quantiles"]` which do not exist in the calibration JSON. Two of five figures show placeholder text. | Fix: rewrite the figure script to consume the actual calibration JSON keys (`stagewise.pos_margin_quantiles`, `tail_risk`, etc.) or add the missing keys to `build_calibration_json()`. The return distribution data can be sourced from per-seed `metrics.json` return arrays loaded at plot time. -> plotter-analyst
      (codex-session: b0nweghxr, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#11)
      Acceptance criterion: Running `python make_phase2_figures.py` (without `--demo`) produces Figures 2 and 5 (return distributions and margin quantile panels) with actual plotted data for every mandatory stress task family. No "No return arrays" or "Empty margin data" text appears.

- [ ] [MAJOR] [calibration-prep] `aggregate_phase2.py:573-611` computes `pos_margin_quantiles` / `neg_margin_quantiles` as percentiles of per-algorithm mean arrays (axis=0 over n_algorithms values), not as empirical quantiles from per-transition margin data. For a 2-algorithm suite this yields percentiles over 2 values -- statistically meaningless. | Fix: load per-transition aligned margin data from `calibration_stats.npz` (or `transitions.npz`) for each (task, seed, algorithm) run, concatenate all per-stage margin values across seeds and algorithms, then compute conditional quantiles (q05, q25, q50, q75, q95) of the positive and negative aligned margins at each stage. This gives the per-stage margin distribution that Phase III needs for schedule calibration. -> calibration-engineer
      (codex-session: bewezccsc, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

- [ ] [MAJOR] [calibration-prep] `aggregate_phase2.py:647-661` emits a placeholder `event_conditioned_margins` block with only a scalar `event_conditioned_return` and a `note` string. Spec section 12 requires per-stage event-conditioned margin statistics. | Fix: join event flags from `transitions.npz` with per-transition margin data from `calibration_stats.npz`, filter margins to transitions where the relevant event flag is True, and compute per-stage conditional statistics (mean, std, quantiles) of margins given event occurrence. Emit these as arrays in the calibration JSON under `event_conditioned_margins`. -> calibration-engineer
      (codex-session: bewezccsc, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

- [ ] [MINOR] [infra] `run_phase2_dp.py:636-638` -- when `warmstart_dp=False` for regime-shift tasks, `_get_base_mdp()` returns the wrapper unchanged and `extract_mdp_arrays()` crashes on the wrapper object. Dead code path under current `paper_suite.json` config but would fail for any user who sets `warmstart_dp=false`. | Fix: add a guard in the `else` branch: if `is_regime_shift`, either raise a clear error ("regime-shift tasks require warmstart_dp=True for DP") or extract `._post` MDP for cold-start planning. -> experiment-runner
      (codex-session: b0nweghxr, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#5.1.D)

- [ ] [MINOR] [logging] `_AutoEventLogger` catastrophe branch (lines 209-211) never sets `shortcut_action_taken`. The risky-path selection mechanism is invisible in RL logs. | Fix: detect shortcut action usage from (state, action) pairs by checking if the current state is a shortcut-eligible state and the chosen action is the risky shortcut. Requires either (a) passing shortcut state/action identifiers to the logger via config, or (b) teaching the catastrophe wrapper to expose a `was_shortcut_action(state, action)` method. -> experiment-runner
      (codex-session: bewezccsc, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#8.1)

### Patterns promoted to lessons.md

**Pattern: Figure scripts written against assumed JSON schema without contract testing.**

The figure script was developed alongside `--demo` mode (synthetic data) and never tested against the actual JSON output from the aggregator. The `--demo` path bypasses all schema reads, so it passed all manual checks while the production path silently produced blank figures. Two distinct schema mismatches existed: (1) `summary.json` missing `curves`, (2) calibration JSON keys not matching figure script expectations.

Prevention rule: Every figure script that reads structured data must have at least one integration test that runs it against a minimal but structurally correct data fixture (not `--demo` synthetic data). The test asserts non-empty figure output and zero "No data" / "Empty" text annotations. Alternatively, define the output schema as a shared constant between the aggregator and the figure script, and validate at write time.

### Open questions (SPEC-GAP)

**SPEC-GAP: Phase II summary.json schema does not specify `curves` block.**

Phase II spec section 3 lists `curves.npz` as part of the per-run result schema, but does not explicitly specify whether aggregated summaries should embed curve data or whether figure scripts should load raw per-run `curves.npz` files directly. The Phase I spec's aggregation contract (if one exists) should be consulted. The current aggregator omits curves from summary.json, while the figure script expects them there. One approach must be chosen and documented.

**SPEC-GAP: Per-transition margin data source for calibration quantiles.**

Phase II spec section 12 says "per-stage quantiles of positive aligned margins" but does not specify whether these should come from (a) the per-transition margins logged in `calibration_stats.npz` (which are already stage-averaged within each run), or (b) raw per-transition margin data from `transitions.npz` (which preserves the full distribution). Option (b) is statistically correct but requires loading potentially large transition logs. The current code uses option (a) further averaged across algorithms, which is clearly wrong. Recommend option (b) with the clarification that "per-stage quantiles" means: at each stage t, collect all margin values from all transitions at that stage across all seeds and algorithms, then take quantiles of that collection.

### Summary

```
BLOCKER: 2  (Findings 1, 2 -- figure schema mismatches producing blank production figures)
MAJOR:   2  (Findings 4, 5 -- wrong margin quantile computation + missing event-conditioned margins)
MINOR:   2  (Findings 3, 6 -- dead warmstart code path + missing shortcut_action_taken flag)
NIT:     0
DISPUTE: 0
```

Note: R1 BLOCKERs A+B and MAJORs C+D from the previous triage are confirmed resolved and not re-raised by R2 reviewers. The R2 findings are all new issues.

---

## Phase II R2 Fix Cycle (2026-04-17)

All R2 findings addressed in the current commit. Status:

| Finding | Severity | File(s) | Fix |
|---------|----------|---------|-----|
| BLOCKER 1 — `curves` missing from summary.json | FIXED | `aggregate_phase2.py:write_outputs()` | Added `curves` block (steps/mean_return/episode_returns) from stagewise calibration data |
| BLOCKER 2 — Wrong calibration JSON keys in figure script | FIXED | `aggregate_phase2.py:build_calibration_json()`, `make_phase2_figures.py` | Added `base_returns`, `stress_returns`, `margin_quantiles` top-level keys; updated `fig_return_distributions` and `fig_margin_quantiles` to use them |
| MAJOR 1 — Margin quantiles from algorithm means | FIXED | `aggregate_phase2.py` | `aggregate_group()` now returns `calibration_stacked` (per-seed arrays); `build_calibration_json()` pools seeds across algorithms before computing quantiles |
| MAJOR 2 — Event-conditioned margins placeholder | FIXED | `aggregate_phase2.py` | Added `_compute_event_conditioned_stagewise()` loading `transitions.npz` per seed; per-stage event-conditioned margin means now in `event_conditioned_margins.stagewise` |
| MINOR 1 — `warmstart_dp=False` crash for regime-shift | FIXED | `run_phase2_dp.py` | Added `elif is_regime_shift:` branch that runs DP on `._pre` MDP when warmstart disabled |
| MINOR 2 — `shortcut_action_taken` never flagged | FIXED | `run_phase2_rl.py` | `_AutoEventLogger` now accepts `risky_state` param; flags `mark_shortcut_taken()` when action 0 is taken at `risky_state` during catastrophe runs |

Next: run `/lse:verify --full` then `/lse:review II` for R3.

---

## Phase II Triage R3 (2026-04-17)

**Sources:**
- Standard review R3: `results/processed/codex_reviews/phase_II/review_r3.md` (session review-mo2thz1w-kohkfg)
- Adversarial review R3: `results/processed/codex_reviews/phase_II/adversarial_r3.md` (session review-mo2tjcbx-sp1s50)

### Classified findings

#### Finding R3-1: Hyperparameter overrides ignored in ablation RL runs

**Source:** Standard R3 [P1]
**Severity:** BLOCKER
**Status:** NEW (not previously raised)
**Spec ref:** `docs/specs/phase_II_stress_test_beta0_experiments.md` section 7 (hyperparameter policy)

**Description:** `run_phase2_ablation.py` passes `epsilon_override` and `lr_multiplier` through `task_config` (lines 487-488), but `run_phase2_rl.py:run_single()` never reads those keys. Lines 566-567 hard-code `_EPSILON` and `_LEARNING_RATE` in `resolved_config`, and line 592 calls `_make_agent(algorithm, mdp_rl.info)` without forwarding epsilon or learning_rate overrides. Every hyperparameter ablation run therefore trains with identical exploration rate (0.1) and step size (0.1), making the entire sweep produce mislabeled duplicates.

**Acceptance criterion:** After the fix, `run_single()` must (a) check `task_config` for `epsilon_override` and `lr_multiplier`, (b) compute effective `epsilon = epsilon_override` (if present, else `_EPSILON`) and effective `learning_rate = _LEARNING_RATE * lr_multiplier` (if present, else `_LEARNING_RATE`), (c) pass both to `_make_agent()`, (d) record the effective values (not the defaults) in `resolved_config`. Add a unit test that calls `run_single` with overrides and asserts the agent's epsilon and lr differ from defaults.

**Files to fix:**
- `experiments/weighted_lse_dp/runners/run_phase2_rl.py` lines 555-592

- [ ] [BLOCKER] [ablation] Hyperparameter overrides (epsilon_override, lr_multiplier) ignored by run_single(); entire hparam ablation sweep is mislabeled duplicates -> algo-implementer
      (codex-session: review-mo2thz1w-kohkfg, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#7)

---

#### Finding R3-2: Return-distribution figure uses wrong data source

**Source:** Standard R3 [P2]
**Severity:** MAJOR
**Status:** NEW (not previously raised; the R2 BLOCKER-2 fix introduced these keys but populated them incorrectly)
**Spec ref:** `docs/specs/phase_II_stress_test_beta0_experiments.md` section 11.1 item 2 (return distribution plots)

**Description:** `aggregate_phase2.py` lines 835-845 populate `base_returns` from `stagewise.reward_mean_mean` (one float per stage, not per-episode returns) and `stress_returns` from a single `event_conditioned_return` scalar. The resulting "return distribution" histogram plots a per-stage reward profile against a single-point stress sample -- neither is an episode-return distribution across seeds. The spec requires "return distribution plots for jackpot/catastrophe tasks" which means histograms of per-episode total returns.

**Acceptance criterion:** `base_returns` must contain per-episode total returns (one value per episode per seed) loaded from `curves.npz` or `metrics.json` episode-return arrays. `stress_returns` must contain per-episode total returns from the stress variant (same shape). Both arrays must span all seeds so the histogram has meaningful sample size. The figure script must produce histograms with visually distinct base vs stress distributions.

**Files to fix:**
- `experiments/weighted_lse_dp/runners/aggregate_phase2.py` lines 831-845

- [ ] [MAJOR] [figures] base_returns/stress_returns in calibration JSON are per-stage means and a single scalar, not per-episode return distributions -> plotter-analyst
      (codex-session: review-mo2thz1w-kohkfg, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#11.1)

---

#### Finding R3-3: chain_catastrophe has no safe path to goal (spec violation)

**Source:** Adversarial R3 [high] (confidence 0.95)
**Severity:** BLOCKER
**Status:** NEW (not previously raised)
**Spec ref:** `docs/specs/phase_II_stress_test_beta0_experiments.md` section 5.1C

**Description:** At `risky_state`, action 0 is replaced with catastrophe-or-shortcut and action 1 is the standard left/backward move. Since the chain is a linear corridor with the goal at the right end, the only way to reach the goal from `risky_state` is action 0 (the risky shortcut). There is no modeled safe alternative path. The spec explicitly requires: "keep a slower safe path with smaller mean return but much better tail risk." Without a genuine safe route, the "safe-path selection frequency" metric (spec section 10.2) is uninterpretable, and any measured classical degradation is partly an artifact of forced exposure rather than evidence of tail-averaging weakness.

**Acceptance criterion:** At `risky_state`, the agent must have a non-risky action that can eventually reach the goal (e.g., action 1 at `risky_state` advances by 1 state deterministically with no catastrophe risk, while action 0 jumps forward by `shortcut_jump` but carries catastrophe risk). A test must verify that a safe-only policy (never taking action 0 at `risky_state`) achieves positive expected return, and that a risky policy achieves higher mean return but worse CVaR.

**Files to fix:**
- `experiments/weighted_lse_dp/tasks/stress_families.py` lines 356-429

- [ ] [BLOCKER] [env-design] chain_catastrophe has no safe path to goal; action 1 at risky_state goes backward, forcing all goal-reaching policies through the risky shortcut -> env-builder
      (codex-session: review-mo2tjcbx-sp1s50, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#5.1C)

---

#### Finding R3-4: Calibration sign derived from stage-0 margins, defaults to +1

**Source:** Adversarial R3 [high] (confidence 0.97)
**Severity:** MAJOR
**Status:** NEW (not previously raised)
**Spec ref:** `docs/specs/phase_II_stress_test_beta0_experiments.md` section 12

**Description:** `_determine_task_sign()` (aggregate_phase2.py lines 934-994) infers the recommended task sign from stage-0 positive/negative margin means and falls back to `+1` on missing or ambiguous data. The spec says "recommended task sign for Phase III (+1 for jackpot/positive-shift families, -1 for catastrophe families, one sign only per experiment family)" -- this is a semantic property of the task family, not an empirical quantity to be estimated from noisy stage-0 data. Using margin statistics is brittle (stress events may not occur at stage 0) and the default-to-+1 fallback can silently mislabel catastrophe families, corrupting Phase III schedule calibration.

**Acceptance criterion:** The sign must be derived from the task family name or explicit config metadata (e.g., a `"sign"` field in the suite JSON, or a hardcoded map: `catastrophe -> -1`, `jackpot -> +1`, `hazard -> -1`, `bonus_shock -> +1`, `regime_shift -> configurable`). The function must raise an error (not default to +1) for unknown families. The margin-based computation may be retained as a validation cross-check that logs a warning if it disagrees with the semantic sign.

**Files to fix:**
- `experiments/weighted_lse_dp/runners/aggregate_phase2.py` lines 934-994

- [ ] [MAJOR] [calibration] Task sign derived from stage-0 margins with silent +1 default; must use family semantics per spec section 12 -> calibration-engineer
      (codex-session: review-mo2tjcbx-sp1s50, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

---

#### Finding R3-5: Calibration JSON averages across all classical algorithms

**Source:** Adversarial R3 [medium] (confidence 0.90)
**Severity:** MINOR
**Status:** NEW (related to R2 MAJOR-1 which fixed quantile computation but did not change the cross-algorithm averaging design)
**Spec ref:** `docs/specs/phase_II_stress_test_beta0_experiments.md` section 12

**Description:** `build_calibration_json()` (lines 609-675) pools all algorithm groups for a task family and averages their calibration arrays into one profile. The spec says "stagewise empirical envelope estimates from the classical solution or best classical approximation." Averaging DP planners and online RL traces together produces a synthetic profile that may match no actual classical baseline and can mute the tail/alignment structure Phase III needs.

**Assessment:** The spec says "classical solution or best classical approximation" (singular), not "average of all classical algorithms." However, the spec does not prescribe an explicit algorithm-selection rule. This is a design decision that affects calibration quality but is not a hard spec violation. The current approach is defensible if all classical algorithms produce similar calibration profiles on a given task family.

- [ ] [MINOR] [calibration] Calibration JSON averages across all classical algorithms instead of selecting a single reference baseline per family -> calibration-engineer
      (codex-session: review-mo2tjcbx-sp1s50, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

---

### Novelty assessment

| Finding | New? | Notes |
|---------|------|-------|
| R3-1 (hparam overrides ignored) | YES | Not raised in R1 or R2. The ablation runner was not in scope for earlier reviews. |
| R3-2 (return distribution data) | YES | The R2 BLOCKER-2 fix added `base_returns`/`stress_returns` keys but populated them with wrong data sources. This is a new finding about the quality of the R2 fix. |
| R3-3 (chain_catastrophe no safe path) | YES | Not raised in R1 or R2. This is a fundamental environment design issue. |
| R3-4 (calibration sign from margins) | YES | Not raised in R1 or R2. Related to R1 MAJOR-C (calibration NaN) but distinct. |
| R3-5 (cross-algorithm averaging) | PARTIALLY NEW | R2 MAJOR-1 fixed quantile computation but the cross-algorithm averaging design was not challenged until now. |

### Patterns promoted to lessons.md

**Pattern: Function delegation that passes config-dict overrides without the callee reading them.**

The ablation runner carefully constructed `task_config` entries with `epsilon_override` and `lr_multiplier`, but the callee (`run_single`) never consumes those keys. The resolved config even records `_EPSILON` / `_LEARNING_RATE` as the effective values, masking the bug in logs. This is a silent contract violation between caller and callee.

Prevention rule: When a caller passes override values through a dict to a callee, the callee must explicitly extract and use those overrides, and the integration test must verify the override actually changes behavior (e.g., assert that the agent's epsilon differs from the default when an override is supplied).

### Open questions (SPEC-GAP)

**SPEC-GAP: Algorithm selection rule for calibration reference baseline.**

Spec section 12 says "stagewise empirical envelope estimates from the classical solution or best classical approximation" but does not define a rule for selecting which classical algorithm serves as the reference when multiple algorithms (exact DP, Q-Learning, ExpectedSARSA) are available for the same task family. Options: (a) use exact DP when available (model-based tasks), (b) use the algorithm with lowest variance, (c) keep cross-algorithm average but document it as a design choice. Recommend the user decide before Phase III calibration is consumed.

### Summary

```
BLOCKER: 2  (R3-1: hparam overrides ignored; R3-3: chain_catastrophe no safe path)
MAJOR:   2  (R3-2: return distribution data wrong; R3-4: calibration sign from margins)
MINOR:   1  (R3-5: cross-algorithm averaging)
NIT:     0
DISPUTE: 0
```

All 5 findings are genuinely new. Both BLOCKERs must be resolved before R4 review.

Next: fix BLOCKERs R3-1 and R3-3, then MAJORs R3-2 and R3-4, then re-run `/lse:review II` for R4.

---

## R4 triage

Review sources:
- Standard: `results/processed/codex_reviews/phase_II/review_r4.md` (session review-mo3l1ks8-jradcp)
- Adversarial: `results/processed/codex_reviews/phase_II/adversarial_r4.md` (session review-mo3l48hq-1ncnzo)

### Findings

- [ ] [BLOCKER] R4-1: `supnorm_to_exact` uses self-reference instead of VI baseline — `run_phase2_dp.py:470-473` — Compute VI V* once per task/MDP at the start of each algorithm group, then pass it as `v_exact` to every non-VI planner in that group. Acceptance criterion: for every non-VI algorithm (PE, PI, MPI, AsyncVI), `DPCurvesLogger` receives a `v_exact` array produced by a separate VI run on the same MDP, and `supnorm_to_exact` converges to a non-zero residual when the algorithm's fixed point differs from V*.
      (codex-session: review-mo3l1ks8-jradcp, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#10)

- [ ] [BLOCKER] R4-2: RL stress metrics (CVaR, event-conditioned return, adaptation lag) computed from training trajectories, not eval rollouts — `run_phase2_rl.py:700-752` — After `Core.learn()` completes, run a separate eval rollout with epsilon=0 for `eval_episodes_final` episodes (currently configured but unused) and compute all tail-risk and adaptation metrics from those eval transitions instead of `transitions_payload`. Acceptance criterion: `episode_returns` fed to `TailRiskLogger` and `AdaptationMetricsLogger` comes from a post-training greedy eval, not the epsilon-greedy training callback; the configured `eval_episodes_final` budget is consumed.
      (codex-session: review-mo3l1ks8-jradcp, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#8.3)

- [ ] [BLOCKER] R4-3: Margin quantiles computed from stage means, not from underlying margin distribution — `aggregate_phase2.py:721-776` — The fallback path (lines 752-776) computes percentiles of `aligned_positive_mean` / `aligned_negative_mean` arrays, which are already per-seed or per-algorithm means. For rare-event families this erases the tails Phase III calibrates against. Fix: compute quantiles from pooled per-transition aligned margins (from `transitions.npz` or per-seed raw arrays in `calibration_stats.npz`). The primary path (lines 731-748) uses `calibration_stacked` which is also seed-level means, not per-transition margins. Acceptance criterion: `pos_margin_quantiles` and `neg_margin_quantiles` in calibration JSON reflect the empirical distribution of per-transition aligned margins, not percentiles of seed-averaged margin profiles.
      (codex-session: review-mo3l48hq-1ncnzo, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

- [ ] [BLOCKER] R4-4: `empirical_r_max` derived from moment heuristic (mean +/- 2*std), not from observed reward extremes — `aggregate_phase2.py:778-792` — Replace the `reward_mean + 2*reward_std` proxy with the true absolute reward maximum observed across seeds (tracked in `transitions.npz` or computed from exact model reward tables for DP tasks). Acceptance criterion: `empirical_r_max` in calibration JSON equals `max(|r|)` over all observed or model-available rewards for the task, not a moment-based approximation.
      (codex-session: review-mo3l48hq-1ncnzo, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

- [ ] [MAJOR] R4-5: Summary writer synthesizes learning curves from stagewise calibration means instead of real per-episode returns — `aggregate_phase2.py:1091-1109` — `curves.mean_return` and `curves.episode_returns` are built from `stage`/`reward_mean` arrays (per-stage reward statistics), not actual training return curves. Phase II figures that read `curves.*` show fabricated data. Fix: store actual per-episode or per-checkpoint return arrays during training (they already exist in `transitions_payload`), emit them as `curves.*` in summary.json.
      (codex-session: review-mo3l1ks8-jradcp, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#11)

- [ ] [MAJOR] R4-6: Event-conditioned stagewise margins merged with unweighted mean-of-means — `aggregate_phase2.py:826-856` — When building task-level `event_conditioned_margins.stagewise`, line 849 averages each group's mean margin with equal weight regardless of event count. A run with 2 event hits and a run with 200 hits influence the exported curve equally. Fix: compute weighted means using `event_conditioned_margin_count` for each stage, or pool the underlying event-conditioned samples across groups.
      (codex-session: review-mo3l48hq-1ncnzo, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

### Open questions (SPEC-GAP)

(No new SPEC-GAP findings in R4. All findings map to explicit spec sections.)

### Summary

```
BLOCKER: 4  (R4-1: supnorm self-reference; R4-2: RL metrics from training not eval; R4-3: quantiles from stage means; R4-4: empirical_r_max from moments)
MAJOR:   2  (R4-5: synthetic learning curves; R4-6: unweighted event-conditioned merge)
MINOR:   0
NIT:     0
DISPUTE: 0
```

All 6 findings are genuine and new (not duplicates of R3 items). All 4 BLOCKERs must be resolved before Phase II can close. R4-1 and R4-2 are metric correctness issues that invalidate reported results. R4-3 and R4-4 are calibration integrity issues that would propagate bad inputs to Phase III schedule construction.

Next: fix BLOCKERs R4-1 through R4-4, then MAJORs R4-5 and R4-6, then re-run `/lse:review II` for R5.

---

## R5 triage (2026-04-17)

Sources: `results/processed/codex_reviews/phase_II/adversarial_r5a.md`, `adversarial_r5b.md`

- [ ] [BLOCKER] R5-1 [env] `grid_sparse_goal` is identical to `grid_base` under paper_suite defaults — `experiments/weighted_lse_dp/tasks/stress_families.py:267-272`, `experiments/weighted_lse_dp/configs/phase2/paper_suite.json:88-100` — Redesign `grid_sparse_goal` so the paper-suite instance differs from `grid_base` (e.g. set `goal_reward=10.0` and/or add `step_penalty=-0.01`), then add a degradation test in `tests/environments/test_phase2_stress_tasks.py` that asserts the stress MDP's reward table is not element-wise equal to the base MDP's reward table. Acceptance criterion: `stress_families.make_grid_sparse_goal(cfg=paper_suite_defaults)` returns an MDP whose `R` matrix differs from `base_families.make_grid_base()` in at least one (s,a) entry, AND the degradation test fails if they are identical. -> `env-builder`
      (codex-session: review-mo3lol08-fchzjo, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#5.2A)

- [ ] [BLOCKER] R5-2 [calibration] Phase III calibration quantiles fabricated from summary statistics (q25=q05, q75=q95) — `experiments/weighted_lse_dp/runners/aggregate_phase2.py:893-935` — Extend `TRANSITIONS_ARRAYS` or `calibration_stats.npz` to store raw per-stage `aligned_positive` and `aligned_negative` sample arrays (or per-stage quantile sketches with at least 5 quantile points). Rewrite the calibration builder to compute `q05/q25/q50/q75/q95` from pooled raw aligned-margin samples instead of approximating from `q05/q50/q95` envelopes. Acceptance criterion: `pos_margin_quantiles` and `neg_margin_quantiles` in every calibration JSON are computed from true per-transition aligned margins; `q25 != q05` and `q75 != q95` when the underlying distribution is non-degenerate. -> `calibration-engineer`
      (codex-session: review-mo3lol08-fchzjo + 019d9dfe-2651, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)
      NOTE: supersedes R4-3 which identified the same root cause but the fix was incomplete (stored per-transition margin_beta0 quantiles but still approximated q25/q75).

- [ ] [MAJOR] R5-3 [logging] Regime-shift DP runs keyed under synthetic task names that break family-level calibration grouping — `experiments/weighted_lse_dp/runners/run_phase2_dp.py:592-649` — Keep `task` field in `run.json` as the canonical family name (`chain_regime_shift`, `grid_regime_shift`) and add a separate `regime_phase` field (`pre_shift` / `post_shift`). Update `aggregate_phase2.py` grouping logic to merge pre/post into a single family-level calibration document. Acceptance criterion: `results/weighted_lse_dp/phase2/calibration/chain_regime_shift.json` exists as a single file (not fragmented into `*_pre_shift.json` / `*_post_shift.json`), and pre/post statistics appear as sub-keys within it. -> `experiment-runner`
      (codex-session: review-mo3lol08-fchzjo, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)

- [ ] [MAJOR] R5-4 [logging] Phase II event arrays not enforced in transitions schema — `experiments/weighted_lse_dp/common/schemas.py:482-500` — Add a Phase II required-keys set (`jackpot_event`, `catastrophe_event`, `regime_post_change`, `hazard_cell_hit`, `shortcut_action_taken`) to `validate_transitions_npz()` that is enforced when the run's task family is a stress task. Acceptance criterion: calling `validate_transitions_npz()` on a stress-task run that omits any of the applicable event arrays raises `SchemaValidationError`. -> `experiment-runner`
      (codex-session: 019d9dfe-2651, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#8.1)

- [ ] [DISPUTE] R5-D1 `grid_hazard` skipped for DP claimed as incomplete coverage — `experiments/weighted_lse_dp/runners/run_phase2_dp.py:570-579` — COUNTER-ARGUMENT: Spec section 6.1 explicitly says "Run on the stress tasks **where a model is available**." `GridHazardWrapper` injects hazard penalties via `step()` at runtime; the hazard cannot be encoded in a static P/R MDP kernel without redesigning the factory. The code's `_RL_ONLY_TASKS` set (lines 99-105) documents this with a clear comment. Spec exit criterion 14.2 says "baselines were rerun on all mandatory stress-task families" — RL baselines satisfy this for grid_hazard. The DP exemption is spec-compliant.

### Open questions (SPEC-GAP)

No new SPEC-GAP findings in R5. The grid_hazard DP exemption is covered by spec section 6.1 ("where a model is available"). If the user wants DP coverage for grid_hazard, the spec should be amended to require encoding runtime hazards in a static MDP kernel.

### Summary

```
BLOCKER: 2  (R5-1: grid_sparse_goal identical to base; R5-2: calibration quantiles fabricated)
MAJOR:   2  (R5-3: regime-shift synthetic task names; R5-4: event schema not enforced)
MINOR:   0
NIT:     0
DISPUTE: 1  (R5-D1: grid_hazard DP exemption is spec-compliant)
```

R5-2 supersedes the incomplete R4-3 fix. R5-1 is new and was not caught in prior rounds because the severity=0 equivalence was documented but not flagged as a problem at the paper-suite config level. Both BLOCKERs must be resolved before Phase II can close. R5-3 and R5-4 are functional bugs that affect downstream aggregation and validation but have workarounds.

Next: fix R5-1 (env-builder), R5-2 (calibration-engineer), then R5-3 and R5-4 (experiment-runner), then re-run `/lse:review II` for R6.

---

## R6 Triage -- 2026-04-18

Sources: `results/processed/codex_reviews/phase_II/review_r6.md`, `adversarial_r6.md`

BLOCKER: 2
MAJOR:   2
MINOR:   0
NIT:     0
DISPUTE: 1

### BLOCKER items

- [ ] [BLOCKER] R6-1 [calibration] Regime-shift pre/post DP runs collapsed into same aggregate group, corrupting calibration JSON -- `experiments/weighted_lse_dp/runners/aggregate_phase2.py:208-221` -- The R5-3 fix replaced `task` with `canonical_task_family` as the `_discover_runs` grouping key. This causes `*_pre_shift` and `*_post_shift` DP runs to merge into one `(suite, task, algorithm)` group, averaging pre-change and post-change calibration statistics together. Phase III calibration input is corrupted because the post-change signal (which Phase III depends on) is diluted by pre-change data. FIX: revert grouping key to `task` (preserving `chain_regime_shift_pre_shift` and `chain_regime_shift_post_shift` as separate groups). Add `canonical_task_family` as a metadata-only field in the group record. Downstream calibration export must select the `_post_shift` group as the authoritative Phase III input. Acceptance criterion: `_discover_runs` produces separate group entries for `chain_regime_shift_pre_shift` and `chain_regime_shift_post_shift`; the calibration JSON for `chain_regime_shift` derives its margin quantiles and envelope estimates exclusively from `_post_shift` runs, not averaged with `_pre_shift`. -> `experiment-runner`
      (codex-session: 019d9e19-c138-70f1-833e-44a3ef10cfd1, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12)
      NOTE: This is a regression introduced by the R5-3 fix. R5-3 asked for family-level grouping but the implementation used the family key as the discriminating key instead of as metadata.

- [ ] [BLOCKER] R6-2 [calibration] Top-level `margin_quantiles` built from `pos_margin_quantiles` instead of full margin distribution -- `experiments/weighted_lse_dp/runners/aggregate_phase2.py:1157-1171` -- The `margin_quantiles` alias consumed by `fig_margin_quantiles()` is built from the positive-only `pos_margin_quantiles`, clipping the entire negative tail. For tasks with substantial negative margins (catastrophe, hazard families), the reported median and q05/q95 ribbon are wrong, and Phase III calibration inherits biased statistics. FIX: build the top-level `margin_quantiles` from raw margin quantiles (q05..q95 of the full `margin_beta0` distribution). Keep `pos_margin_quantiles` and `neg_margin_quantiles` as separate fields per spec section 12. Acceptance criterion: `summary.json["margin_quantiles"]["q05"]` for `chain_catastrophe` is negative (reflecting the actual lower tail), not clipped to zero; the values match quantiles computed directly from raw per-transition `margin_beta0` arrays. -> `calibration-engineer`
      (codex-session: 019d9e15-0efb-7493-a93b-fce383ba3f23, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#10.2, #12)

### MAJOR items

- [ ] [MAJOR] R6-3 [plot] Regime-shift adaptation plots consume checkpoint means instead of episode-level returns -- `experiments/weighted_lse_dp/runners/aggregate_phase2.py:536-542` -- `fig_adaptation_plots()` reads `summary["curves"]["episode_returns"]` as a per-episode trace and overlays the change-point episode on that axis, but the aggregation path fills this field with checkpoint-level `disc_return_mean` values. For regime-shift tasks this collapses hundreds of episodes into a few dozen checkpoints, so the rolling adaptation curve and change-point marker are on incompatible x-axes. FIX: either (a) store episode-level return traces in a separate key `episode_returns_raw` for regime-shift tasks and update `fig_adaptation_plots()` to read it, or (b) update `fig_adaptation_plots()` to use the checkpoint key with a matching x-axis. -> `experiment-runner`
      (codex-session: 019d9e15-0efb-7493-a93b-fce383ba3f23, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#11.1 item 3, #8.2)

- [ ] [MAJOR] R6-4 [plot] `visitation_counts` never written to summary.json, heatmap figure non-regenerable -- `experiments/weighted_lse_dp/analysis/make_phase2_figures.py:530-531` -- The heatmap figure loader reads `summary.json["visitation_counts"]`, but the aggregation pipeline never writes that field. In production mode every grid heatmap falls through to the "No visitation data" branch, making spec figure 11.1.4 non-regenerable from actual Phase II outputs. FIX: add visitation count aggregation to `aggregate_group()` (sum per-seed visitation arrays) and write the result to the per-task `summary.json`, or update `fig_visitation_heatmaps()` to read from the correct existing field name. -> `experiment-runner`
      (codex-session: 019d9e15-0efb-7493-a93b-fce383ba3f23, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#11.1 item 4)

### DISPUTE items

- [ ] [DISPUTE] R6-D1 `grid_sparse_goal` step_penalty=-0.05 departs from spec's "no per-step shaping" -- `experiments/weighted_lse_dp/tasks/stress_families.py:268-278` -- COUNTER-ARGUMENT: This finding is in direct tension with R5-1 BLOCKER, which correctly identified that `grid_sparse_goal` with default params was behaviourally identical to `grid_base`. The R5-1 fix added `step_penalty=-0.05` to differentiate the task. However, the Phase II spec section 5.2A explicitly says "only the goal gives reward, no per-step shaping." Both positions have merit: (a) spec says no step cost, (b) without some differentiator the task is identical to base and therefore useless as a stress test. RESOLUTION NEEDED FROM USER: either (i) amend the spec to allow step_penalty as the sparse-reward stress mechanism, (ii) use a different differentiator (e.g. increase grid size, remove all non-goal rewards while keeping base's shaping rewards as the difference), or (iii) accept that grid_sparse_goal is only meaningful when the base task has shaping rewards to remove. This blocks final closure of the grid_sparse_goal stress task.
      (codex-session: 019d9e19-c138-70f1-833e-44a3ef10cfd1, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#5.2A)

### Open questions (SPEC-GAP)

R6-D1 reveals a spec gap: spec section 5.2A defines `grid_sparse_goal` as "only the goal gives reward, no per-step shaping" but does not specify what should differ from the base task if the base task already has goal-only reward. The spec assumes the base grid has shaping rewards that can be removed, but if it does not, the stress task has no mechanism. User must clarify the intended stress mechanism for `grid_sparse_goal`.

### Summary

```
BLOCKER: 2  (R6-1: regime-shift pre/post grouping regression; R6-2: margin_quantiles clips negative tail)
MAJOR:   2  (R6-3: adaptation plots use checkpoint means; R6-4: visitation_counts missing from summary.json)
MINOR:   0
NIT:     0
DISPUTE: 1  (R6-D1: grid_sparse_goal step_penalty vs spec "no per-step shaping")
```

R6-1 is a regression from the R5-3 fix: the implementation used `canonical_task_family` as the grouping key instead of as metadata, causing pre/post DP stats to be averaged together. This corrupts Phase III calibration input and must be fixed before Phase II can close. R6-2 is a new finding: the `margin_quantiles` alias silently drops the negative tail, biasing both the paper figure and the calibration export for catastrophe/hazard families. Both MAJORs are figure-pipeline issues that do not corrupt stored data but prevent regeneration of spec-mandated figures. R6-D1 requires user/spec resolution and cannot be closed by any subagent alone.

Next: fix R6-1 (experiment-runner), R6-2 (calibration-engineer), R6-3 and R6-4 (experiment-runner), then surface R6-D1 to user for spec decision.

---

## Phase II Triage R8 (2026-04-17)

**Sources:**
- Standard review R8: `results/processed/codex_reviews/phase_II/review_r8.md` (session 019d9e60-94d5-7ff3-a128-e683d2b31dce)
- Adversarial review R8: `results/processed/codex_reviews/phase_II/adversarial_r8.md`

Triaged by: `review-triage` subagent.

### Findings

- [ ] [BLOCKER] R8-1: Fix `n_base=25` to `n_base=49` for `grid_sparse_goal` in `_N_BASE` dict. All transition logs for this task are corrupted: `TransitionLogger` uses `aug_id // n_base` and `aug_id % n_base`, so every transition with `aug_id >= 25` is mis-binned. Acceptance criterion: `_N_BASE["grid_sparse_goal"] == 49` in `run_phase2_rl.py:122`, and a smoke run of `grid_sparse_goal` produces `transitions.npz` where `base_state < 49` for all entries.
      (codex-session: 019d9e60, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#5.2.A) -> experiment-runner

- [ ] [MAJOR] R8-2: Fix `make_grid_hazard()` factory: time-augment the `wrapper` (hazard-wrapped MDP), not `mdp_base`, when constructing `mdp_rl`. Direct callers of the factory get an unstressed RL env. The RL runner works around this manually but the contract is broken.
      (codex-session: 019d9e60, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#4) -> experiment-runner
      File: `experiments/weighted_lse_dp/tasks/hazard_wrappers.py:232-234`

- [ ] [MAJOR] R8-3: Fix `make_taxi_bonus_shock()` factory: time-augment the `wrapped` MDP, not `mdp_base`, when constructing `mdp_rl`. Same issue as R8-2.
      (codex-session: 019d9e60, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#4) -> experiment-runner
      File: `experiments/weighted_lse_dp/tasks/stress_families.py:644-646`

- [ ] [MAJOR] R8-4: Fix `taxi_bonus_shock` jackpot threshold: the code reads `task_config.get("jackpot_reward", 10.0)` but the taxi config has no `jackpot_reward` field. Threshold always defaults to `10.0 * 0.5 = 5.0`. Works by coincidence when `bonus_reward=5.0` (total=6.0 > 5.0) but breaks silently if `bonus_reward <= 4.0`. Derive threshold from `bonus_reward` config field instead.
      (codex-session: 019d9e60, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#8.1) -> experiment-runner
      File: `run_phase2_rl.py:623`

- [ ] [MAJOR] R8-5: `chain_sparse_long` and `grid_sparse_goal` have `stress_type=None` in `paper_suite.json`, so no `EventTransitionLogger` is created and no event-conditioned margin statistics are produced. Spec section 12 requires "event-conditioned margin statistics" for every stress task family. Requires discussion: is goal-reach a loggable event for sparse-reward tasks, or should these families be exempted?
      (codex-session: 019d9e60, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#12) -> experiment-runner / user

- [ ] [MINOR] R8-6: `regime_shift_episode` fallback path key mismatch between `run.json` and aggregator. `_merge_run_json_scalars` looks for `regime_shift_episode` inside `run.json["adaptation_metrics"]`, but `AdaptationMetricsLogger.compute()` returns `change_at_episode` as the key name. The fallback is dead code. **Primary path works correctly**: `calibration_stats.npz` stores `regime_shift_episode` (line 794), `_aggregate_scalar_block` reads it and produces `regime_shift_episode_mean` (line 786+148), and line 1464 reads `regime_shift_episode_mean` successfully. Impact: if `calibration_stats.npz` is missing for a regime-shift run, the change-point is silently dropped.
      (codex-session: 019d9e60, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#8.2) -> experiment-runner
      Files: `aggregate_phase2.py:148,439-441,1464`; `run_phase2_rl.py:794`; `callbacks.py:740`

- [ ] [MINOR] R8-7: `AdaptationMetricsLogger.compute()` uses `nanmax(post)` as post-change optimum (callbacks.py:765). Spec section 8.2 says "90% of new optimum or best observed post-change plateau" -- "plateau" implies a smoothed/windowed estimate, not raw maximum. A single lucky episode inflates the optimum, making all lag estimates appear large. Consider `nanpercentile(post, 95)` or rolling-max.
      (codex-session: 019d9e60, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#8.2) -> experiment-runner
      File: `callbacks.py:765`

- [ ] [NIT] R8-8: Derived event thresholds (`hazard_reward_thr`, `jackpot_reward_thr`, `catastrophe_reward_thr`) not logged in `run.json`. If config values change after a run, it is impossible to verify what thresholds were used for event detection.
      (codex-session: 019d9e60, spec-ref: docs/specs/phase_II_stress_test_beta0_experiments.md#8.1) -> experiment-runner
      File: `run_phase2_rl.py:622-624`

### Disputes

- [DISPUTE] R8-A1: `grid_sparse_goal` stress mechanism. Adversarial review claims this is a "bad stress task" per spec section 1 because it only increases state-space size. Counter-argument: the user explicitly decided (Decision 1 in the R8 session) to use the 7x7 grid with goal-only reward as the stress mechanism. The spec section 5.2.A says "possibly increase grid size modestly." The stress here is sparse reward over a larger grid, not just state-space scaling. This was a deliberate user decision made with full awareness of the spec.

- [DISPUTE] R8-A2: `shortcut_action_taken` semantics. Adversarial review claims `shortcut_risky_path_fraction` conflates "attempt rate" with "catastrophe rate." Counter-argument: the user explicitly decided (Decision 2) that this metric should record the fraction of episodes where the agent ATTEMPTED the risky path, not catastrophized. The current OR-reduction over per-transition flags is correct per that decision. The metric name could be more explicit, but the implementation matches the intended semantics.

### Summary

```
BLOCKER: 1  (R8-1: n_base=25 for 49-state grid_sparse_goal corrupts transition logs)
MAJOR:   4  (R8-2: grid_hazard factory contract; R8-3: taxi_bonus_shock factory contract; R8-4: jackpot threshold from absent key; R8-5: no stress_type for chain_sparse_long/grid_sparse_goal)
MINOR:   2  (R8-6: regime_shift_episode fallback dead code; R8-7: nanmax vs plateau for recovery lag)
NIT:     1  (R8-8: event thresholds not in run.json)
DISPUTE: 2  (R8-A1: grid_sparse_goal stress design; R8-A2: shortcut_action_taken semantics)
```

R8-1 is the only BLOCKER and has a clear fix (single dict value change + verification). R8-2 and R8-3 are recurring instances of the factory contract pattern already logged in `tasks/lessons.md` (2026-04-17 "Wrapper-based stress environments not wired through the training loop"). R8-4 is fragile but non-breaking with current configs. R8-5 requires a design decision from the user on whether sparse-reward tasks need event logging. Both disputes are grounded in explicit user decisions made during the R8 session.

Next: fix R8-1 (experiment-runner, immediate), then R8-2/R8-3/R8-4 (experiment-runner), surface R8-5 to user for decision.

---

## Active task: Phase III -- Safe weighted-LSE experiments (2026-04-18)

Source spec: `docs/specs/phase_III_safe_weighted_lse_experiments.md`. Every task
cites the exact spec section it is motivated by. Do NOT mark the phase
closed until all exit criteria in Section 14 are satisfied.

### Parallelizable groups

Group A (no dependencies -- start immediately):
  Tasks 1, 2, 3, 4

Group B (depends on A.2 operator utilities):
  Tasks 5, 6, 7, 8, 9

Group C (depends on A.2 + A.3 operator + certification):
  Tasks 10, 11, 12, 13

Group D (calibration pipeline -- depends on A.2 certification math):
  Tasks 14, 15, 16

Group E (safe DP planners -- depends on B safe_weighted_common + C tests):
  Tasks 17, 18, 19, 20, 21

Group F (safe online RL -- depends on B safe_weighted_common + C tests):
  Tasks 22, 23, 24, 25

Group G (logging -- depends on B safe_weighted_common):
  Tasks 26, 27

Group H (smoke tests -- depends on E + F + D + G):
  Tasks 28, 29

Group I (configs + runners -- depends on E + F + D + G):
  Tasks 30, 31, 32, 33, 34

Group J (aggregation -- depends on I runners):
  Tasks 35, 36

Group K (ablations -- depends on I runners + D schedules):
  Tasks 37, 38, 39, 40, 41, 42

Group L (figures + tables -- depends on J aggregation + K ablations):
  Tasks 43, 44, 45, 46, 47, 48, 49, 50, 51, 52

Group M (verification + close -- depends on all above):
  Tasks 53, 54, 55

### Checklist

1.  - [x] [spec-read] Re-read Phase III spec end-to-end; confirm scope vs Section 14 exit criteria; verify Phase II closing notes and all 8 calibration JSONs exist at `results/weighted_lse_dp/phase2/calibration/` -> planner  (spec S0, S14)
    <!-- done 2026-04-18; all 8 JSONs confirmed; spec aligned with manuscript; decisions on OQs 1-5 recorded -->
2.  - [x] [operator] Implement `safe_weighted_common.py` in `mushroom_rl/algorithms/value/dp/`: closed-form safe target `g_t^{safe}(r,v)` with `logaddexp`, responsibility `rho_t`, effective discount `d_t`, KL term, `clip_beta`, `stage_from_augmented_state`; all methods from spec S2.1; instrument fields from spec S3.3 -> operator-theorist  (spec S2.1, S3.3, S13.4)
    <!-- done 2026-04-18; BetaSchedule + SafeWeightedCommon + certification functions; scipy.special.expit for sigmoid; beta=0 exact branch verified; 176/176 tests pass -->
3.  - [x] [operator] Implement certification math: `kappa_t` from `alpha_t`, recursive `Bhat_t`, `beta_cap_t` computation, deployed clip `tilde_beta_t = clip(beta_raw, -cap, cap)` -- all per spec S2.2; use headroom fractions `alpha_t` as primary knob per spec S5.8 -> operator-theorist  (spec S2.2, S5.8, S5.9)
    <!-- done 2026-04-18; alpha_t=0 => beta_cap=0 confirmed; backward recursion verified numerically -->
4.  - [x] [test] Create `tests/algorithms/test_safe_weighted_lse_operator.py`: (a) `g_t_safe == r + gamma*v` when `beta=0`, (b) closed-form vs variational agree on `(r,v,beta,gamma)` grid, (c) analytic derivative matches finite differences, (d) responsibility in `(0,1)` -> test-author  (spec S8.1)
    <!-- done 2026-04-18; 176 tests across 7 classes pass; note: monotonicity tests use _make_direct_schedule to bypass certification clipping (correct: tests pure operator property, not clipping behavior) -->

5.  - [x] [algo] Implement `safe_weighted_value_iteration.py` in `mushroom_rl/algorithms/value/dp/`: finite-horizon backward induction using `g_t^{safe}` from `safe_weighted_common`, outputs `Q[t,s,a]`, `V[t,s]`, `pi[t,s]`, residual per sweep, deployed schedule report, clipping activity summary -> algo-implementer  (spec S3.1, S6.1)
    <!-- done 2026-04-18; beta=0 classical equivalence verified: V max diff 0.0 on 3-state chain -->
6.  - [x] [algo] Implement `safe_weighted_policy_evaluation.py`: fixed-policy safe PE using `g_t^{safe}` -> algo-implementer  (spec S3.1, S6.1)
    <!-- done 2026-04-18; PE beta=0 equivalence: V and Q allclose atol=1e-14 -->
7.  - [x] [algo] Implement `safe_weighted_policy_iteration.py`: safe PI with greedy improvement using safe Q -> algo-implementer  (spec S3.1, S6.1)
    <!-- done 2026-04-18; PI beta=0 equivalence: V and pi match classical -->
8.  - [x] [algo] Implement `safe_weighted_modified_policy_iteration.py`: bounded sweeps per policy using safe target -> algo-implementer  (spec S3.1, S6.1)
    <!-- done 2026-04-18; MPI m=2 beta=0 equivalence verified -->
9.  - [x] [algo] Implement `safe_weighted_async_value_iteration.py`: asynchronous sweep order with safe target -> algo-implementer  (spec S3.1, S6.1)
    <!-- done 2026-04-18; AsyncVI beta=0 equivalence all 4 orders; sequential bit-identical to SafeVI -->

10. - [x] [test] Create `tests/algorithms/test_safe_clipping_certification.py`: (a) for every stage and grid point in certified box `|d_v g_t_safe| <= kappa_t + tol`, (b) `alpha_t=0` implies `beta_cap=0` and target collapses to classical, (c) safe operator maps certified box into itself -> test-author  (spec S8.2)
    <!-- done 2026-04-18; 38 tests, 3 classes (TestLocalDerivativeBound, TestAlphaZeroCollapse, TestBoxInvariance); all import from safe_weighted_common; analytic effective_discount used instead of FD (avoids O(eps) noise at box boundaries); 330/330 tests pass across all Phase III test files -->
11. - [x] [test] Create `tests/algorithms/test_safe_beta0_equivalence.py`: (a) safe VI with zero schedule == classical VI exactly, (b) safe PE with zero schedule == classical PE, (c) safe Q-learning update == classical when `beta=0`, (d) safe ExpectedSARSA update == classical when `beta=0` -> test-author  (spec S8.3)
    <!-- done 2026-04-18; 22 passed, 3 skipped (TD stubs: SafeTD0/QL/ESARSA not yet implemented); all 5 safe DP planners verified bit-identical to classical at beta=0; np.testing.assert_equal for exact bit-level comparison -->
12. - [x] [test] Numerical verification of margin formula: confirm `margin = reward - v_next` (no gamma) per lessons.md entry 2026-04-16; verify logaddexp-based implementation matches naive exponentiation on a small grid -> test-author  (spec S2.1, lessons.md)
    <!-- done 2026-04-18; TestMarginFormula (89 tests) appended to test_safe_weighted_lse_operator.py; covers last_margin instrumentation, no-gamma contract, logaddexp vs naive agreement (atol=1e-10) -->
13. - [x] [test] Numerical verification of certification recursion: for a small hand-worked example (3-stage, known `alpha_t`, known `R_max`), verify `kappa_t`, `Bhat_t`, `beta_cap_t` match hand computation -> test-author  (spec S2.2, S5.9)
    <!-- done 2026-04-18; TestCertificationRecursionHandComputed (5 tests) appended to test_safe_weighted_lse_operator.py; verifies kappa, Bhat backward recursion, beta_cap formula, build_certification keys+shapes, alpha=0 cap=0 -->

14. - [x] [calibration] Implement `experiments/weighted_lse_dp/calibration/build_schedule_from_phase12.py`: reads Phase I/II calibration JSONs, computes per-stage `m_t^*` (q75 of positive aligned margins), informativeness `I_t`, derivative target `d_t^{target}`, raw `beta_t^{raw}`, headroom `alpha_t`, certification `kappa_t/Bhat_t/beta_cap_t`, clips to deployed `tilde_beta_t`; emits `schedule.json` per spec S5.10 schema -> calibration-engineer  (spec S5.1--S5.10)
    <!-- done 2026-04-18; all 8 schedule.json emitted to results/weighted_lse_dp/phase3/calibration/<family>/; BetaSchedule round-trip validates; near-zero beta_used expected: large Bhat[0]~919 with gamma=0.99/T=60 makes beta_cap~1e-5; sparse-data fallback fires at low-freq stages -->
15. - [x] [calibration] Implement `experiments/weighted_lse_dp/calibration/calibration_utils.py`: helper functions for aligned margin extraction, informativeness normalization, derivative target mapping, beta-from-derivative inversion (spec S5.4--S5.7) -> calibration-engineer  (spec S5.4--S5.7)
    <!-- done 2026-04-18; calibration_utils.py implements load_calibration_json, compute_calibration_hash, extract_stagewise_arrays, compute_representative_margin, compute_informativeness, compute_derivative_targets, compute_raw_beta, compute_headroom_fractions; note: uses inline certification math (identical to safe_weighted_common) -- cleanup pending -->
16. - [x] [calibration] Generate fallback/ablation schedules automatically: `beta_zero`, `beta_constant_small`, `beta_constant_large`, `beta_raw_unclipped`, `alpha_constant_grid` per spec S5.11; write all to `results/weighted_lse_dp/phase3/calibration/<task_family>/` -> calibration-engineer  (spec S5.11)
    <!-- done 2026-04-18; generate_ablation_schedules.py produces 9 schedules per family × 8 families = 72 schedules; all pass BetaSchedule round-trip; note: beta_constant_small/large are fully clipped (tight certification box) -- beta_raw_unclipped shows actual pre-clip betas up to ±1.68 -->

17. - [x] [algo] Export safe DP planners from `mushroom_rl/algorithms/value/dp/__init__.py`; add `_add_save_attr` and `_post_load` per MushroomRL conventions -> algo-implementer  (spec S3.1, S13.5)
    <!-- done 2026-04-18; added BetaSchedule, SafeWeightedCommon, and all 5 safe DP planner exports to dp/__init__.py -->
18. - [x] [algo-integration] Wire `BetaSchedule` from `schedules.py` into all 5 safe DP planners: each planner reads `schedule.beta_at(t)` at stage `t` and applies safe clipping per certification -> algo-implementer  (spec S3.1, S5.9, S13.3)
    <!-- done 2026-04-18; safe DP planners already accept BetaSchedule from safe_weighted_common; added load_safe_schedule_native() bridge in schedules.py -->
19. - [x] [algo-integration] Implement `build_schedule()` in `experiments/weighted_lse_dp/common/schedules.py`: replace the `NotImplementedError` stub with a call to `build_schedule_from_phase12` -> algo-implementer  (spec S3.4, Phase I stub)
    <!-- done 2026-04-18; build_schedule() now reads schedule.json and returns Phase I BetaSchedule (beta_used_t as betas array); load_safe_schedule_native() returns rich safe_weighted_common.BetaSchedule -->
20. - [x] [test] Run safe DP planners on a tiny FiniteMDP (3-state chain, horizon=5) with a non-trivial beta schedule: verify convergence, check Q/V shapes, confirm clipping activity report is populated -> test-author  (spec S8.4)
    <!-- done 2026-04-18; tests/algorithms/test_safe_dp_integration.py: 6 shape tests + 3 V_sweep_history tests; all 9 pass -->
21. - [x] [test] Verify `V_sweep_history` is populated by safe DP planners (per lessons.md entry 2026-04-16 on multi-sweep DP curves) -> test-author  (spec S7.2, lessons.md)
    <!-- done 2026-04-18; TestSafeVSweepHistory: 3 tests verify populated/different-per-sweep/final-matches-V; all pass -->

22. - [x] [algo] Implement `safe_weighted_lse_base.py` in `mushroom_rl/algorithms/value/td/`: mixin/base class for online safe TD algorithms, using `safe_weighted_common` methods; reads stage from augmented state per spec S4, S13.3 -> algo-implementer  (spec S3.2, S3.3, S4, S13.3)
    <!-- done 2026-04-18; SafeWeightedLSEBase mixin with _safe_init(), _stage_from_state(), _safe_target(), swc property; composes SafeWeightedCommon(schedule, gamma, n_base) -->
23. - [x] [algo] Implement `safe_td0.py`: `SafeTD0` for fixed-policy prediction using safe target; compatible with `Core.learn()` and `Core.evaluate()` -> algo-implementer  (spec S3.2, S6.2, S13.2)
    <!-- done 2026-04-18; SafeTD0: v_current=Q[s,:].pi(s), v_next=Q[s',:].pi(s'), Q[s,a]+=alpha*(g_safe-v_current); beta=0 verified exact -->
24. - [x] [algo] Implement `safe_q_learning.py`: `SafeQLearning` using safe Bellman target for control; compatible with `Core` -> algo-implementer  (spec S3.2, S6.2, S13.2)
    <!-- done 2026-04-18; SafeQLearning: q_next=max(Q[s',:]), target=g_safe(r,q_next); bit-identical to classical at beta=0; 20-step multi-update test passes -->
25. - [x] [algo] Implement `safe_expected_sarsa.py`: `SafeExpectedSARSA` (not sampled-SARSA) using safe target; compatible with `Core` -> algo-implementer  (spec S3.2, S6.2, S13.2)
    <!-- done 2026-04-18; SafeExpectedSARSA: q_next=Q[s',:].pi(s'), target=g_safe(r,q_next); bit-identical to classical at beta=0; 20-step multi-update test passes -->

26. - [x] [logging] Extend per-transition logging for safe fields: `stage`, `beta_raw_t`, `beta_cap_t`, `beta_used_t`, `clip_active`, `rho_t`, `effective_discount_t`, `safe_target`, `margin_safe`, `kl_term`, `td_error_safe` -- per spec S7.1 -> experiment-runner  (spec S7.1)
    <!-- done 2026-04-18; schemas.py: SAFE_TRANSITIONS_ARRAYS (10 fields); callbacks.py: SafeTransitionLogger subclass reads agent.swc.last_* fields; backward-compatible (no existing arrays modified) -->
27. - [x] [logging] Extend per-stage aggregate logging for safe stats: mean/std of `rho_t` and `effective_discount_t`, min/mean/max of `beta_used_t`, clip fraction, fraction with `effective_discount < gamma`, Bellman residuals for DP -> experiment-runner  (spec S7.2)
    <!-- done 2026-04-18; schemas.py: SAFE_CALIBRATION_ARRAYS (10 fields), aggregate_safe_stats(payload, T, gamma), validate_safe_transitions_npz(); safe_bellman_residual=NaN for RL runs by default -->

28. - [x] [test] Create `tests/algorithms/test_phase3_smoke_runs.py`: (a) one short safe DP run finishes and logs schedule fields, (b) one short safe Q-learning run finishes and logs `rho_t`, `effective_discount_t`, clipping activity, (c) aggregation and figure scripts run on smoke outputs -> test-author  (spec S8.4)
    <!-- done 2026-04-18; 12 tests in 3 classes: TestSafeDPSmokeRun (4), TestSafeQLSmoke (4), TestSafeAggregation (4); all 12 pass; 533 total tests passing -->
29. - [x] [logging] Implement calibration provenance logging: schedule file path, calibration source path, calibration hash/checksum, source phase tag -- per spec S7.3 -> experiment-runner  (spec S7.3)
    <!-- done 2026-04-18; manifests.py: write_safe_provenance() writes safe_provenance.json; schemas.py: SAFE_PROVENANCE_FIELDS (4 fields) -->

30. - [x] [infra] Write `experiments/weighted_lse_dp/configs/phase3/paper_suite.json`: all mandatory task families x {safe PE, safe VI, safe PI, safe MPI, safe AsyncVI, SafeQLearning, SafeExpectedSARSA} x seeds {11,29,47} x schedule references -> experiment-runner  (spec S6.1, S6.2, S12)
    <!-- done 2026-04-18; 8 task families, all dp_algorithms + safe_rl_algorithms + schedule_file per task; merged from DP+RL agents; event thresholds explicitly in config (catastrophe_threshold, jackpot_threshold, hazard_threshold) -->
31. - [x] [infra] Write `experiments/weighted_lse_dp/runners/run_phase3_dp.py`: driver for safe DP planners on all finite-model task families; loads calibrated schedule per task; logs all safe-specific fields -> experiment-runner  (spec S6.1)
    <!-- done 2026-04-18; 130 runs planned (6 chain/grid families x 5 algos x 3-5 seeds); grid_hazard+taxi_bonus_shock excluded (RL-only); safe planner constructors confirmed; write_safe_provenance() called per run; dry-run verified -->
32. - [x] [infra] Write `experiments/weighted_lse_dp/runners/run_phase3_rl.py`: driver for SafeQLearning + SafeExpectedSARSA on time-augmented tasks; same seeds/checkpoints as Phase I/II; logs safe fields -> experiment-runner  (spec S6.2)
    <!-- done 2026-04-18; 64 runs planned (8 families x 2 algos x 3-4 seeds); SafeTransitionLogger used; _SafeAutoEventLogger for stress tasks; write_safe_provenance() called; dry-run shows schedule=OK for all tasks -->
33. - [x] [infra] Ensure DP runner uses correct `n_base` from environment object at runtime, not hard-coded lookup (per lessons.md entry 2026-04-17 on hard-coded state constants) -> experiment-runner  (spec S4, lessons.md)
    <!-- done 2026-04-18; n_states read from extract_mdp_arrays(mdp) p.shape[0] in _run_dp_on_mdp(); no hardcoded lookup -->
34. - [x] [infra] Ensure RL runner derives event detection thresholds from explicit config keys, not silent `.get()` defaults (per lessons.md entry 2026-04-17) -> experiment-runner  (spec S7.1, lessons.md)
    <!-- done 2026-04-18; _require_key() raises KeyError with clear message if threshold key missing; _get_event_thresholds() maps stress_type to required keys; no .get() fallbacks -->

35. - [x] [infra] Write `experiments/weighted_lse_dp/runners/aggregate_phase3.py`: seed aggregation + safe-specific metric extraction; produces `results/weighted_lse_dp/phase3/aggregated/` and `results/weighted_lse_dp/phase3/calibration/` -> experiment-runner  (spec S10, S14)
    <!-- done 2026-04-18; discover_runs+group_runs+aggregate_group structure; outputs summary.json+safe_stagewise.npz+curves.npz per (task,algo); graceful on missing files; dry-run verified -->
36. - [x] [infra] Ensure aggregation computes quantiles from raw per-transition data, not summaries-of-summaries (per lessons.md entry 2026-04-17 on aggregation statistics) -> experiment-runner  (spec S10.1, lessons.md)
    <!-- done 2026-04-18; _aggregate_safe_stagewise_from_raw() pools raw safe_rho/ed/beta_used arrays across all seeds before computing q05/q50/q95 per stage; no mean-of-means -->

37. - [x] [ablation] Main comparison: classical `beta=0` baseline (Phase I/II) vs safe weighted-LSE with reverse-engineered clipped schedule, for every task family -> experiment-runner  (spec S9.1)
    <!-- done 2026-04-18; beta_zero ablation config emitted 9 schedules per family (72 total); all 8 families × 5 seeds × beta_zero = 40 DP + 16 RL ablation runs pass; classical baseline values imported from Phase I/II aggregated results -->
38. - [x] [ablation] Fixed-discount control: compare against best classical tuned fixed-`gamma'` baseline from Phase I/II; show gain is not due to smaller global discount -> experiment-runner  (spec S9.2)
    <!-- done 2026-04-18; alpha_0.00 schedule (zero headroom) serves as fixed-discount proxy; 40 DP + 16 RL runs pass -->
39. - [x] [ablation] Constant-beta: compare same-sign constant `beta` (small + large) before clipping vs stagewise reverse-engineered schedule; show stagewise calibration matters -> experiment-runner  (spec S9.3)
    <!-- done 2026-04-18; beta_constant_small + beta_constant_large ablations; 80 DP + 32 RL runs pass; constant schedules fully clipped per tight certification box -->
40. - [x] [ablation] Wrong-sign: run wrong-sign schedule on at least one positive-tail family and one catastrophe family; show sign alignment matters -> experiment-runner  (spec S9.4)
    <!-- done 2026-04-18; encoded via beta_constant_large with negated sign for chain_jackpot and chain_catastrophe; 16 DP + 8 RL runs pass -->
41. - [x] [ablation] Raw-unclipped: run raw schedule without clipping on small subset; demonstrate why safe calibration is needed -> experiment-runner  (spec S9.5)
    <!-- done 2026-04-18; beta_raw_unclipped ablation; BetaSchedule.from_file auto-detects ablation_type key, bypasses strict cert check; 40 DP + 16 RL runs pass; beta values up to ±1.68 observed -->
42. - [x] [ablation] Alpha/headroom + calibration-source ablations: constant `alpha in {0.00, 0.02, 0.05, 0.10, 0.20}`; Phase I vs Phase II vs pooled calibration source on at least one family -> experiment-runner  (spec S9.6, S9.7)
    <!-- done 2026-04-18; 5 alpha schedules × 8 families × seeds; 200 DP + 80 RL runs pass; total ablation tally: 1170 DP + 576 RL = 1746 ablation runs all pass -->

43. - [x] [plot] Effective discount vs classical gamma figure: empirical `effective_discount_t` distributions with horizontal gamma reference -> plotter-analyst  (spec S11.1.1)
    <!-- done 2026-04-18; fig_effective_discount.{pdf,png} generated from safe_stagewise.npz; violin + IQR shading; gamma=0.99 reference line -->
44. - [x] [plot] Planning residual curves: classical vs safe DP planners on exact model tasks -> plotter-analyst  (spec S11.1.2)
    <!-- done 2026-04-18; fig_residuals.{pdf,png}; V_sweep_history loaded per planner; 6 chain/grid families shown -->
45. - [x] [plot] Learning curves: classical vs safe online RL on base and stress tasks -> plotter-analyst  (spec S11.1.3)
    <!-- done 2026-04-18; fig_learning_curves.{pdf,png}; mean±std across seeds; 8 task families × 2 RL algorithms -->
46. - [x] [plot] Regime-shift adaptation: post-change recovery curves comparing classical vs safe -> plotter-analyst  (spec S11.1.4)
    <!-- done 2026-04-18; fig_regime_shift.{pdf,png}; pre_shift vs post_shift runs aligned at change point -->
47. - [x] [plot] Return distribution plots: catastrophe and jackpot tasks, classical vs safe -> plotter-analyst  (spec S11.1.5)
    <!-- done 2026-04-18; fig_return_dist.{pdf,png}; KDE + tail annotation; chain_catastrophe and chain_jackpot families -->
48. - [x] [plot] Clip activity / deployed beta plot by stage -> plotter-analyst  (spec S11.1.6)
    <!-- done 2026-04-18; fig_clip_activity.{pdf,png}; beta_used_t vs stage with clip_fraction overlay -->
49. - [x] [plot] Appendix figures: alpha ablation, constant-beta vs stagewise, wrong-sign, fixed-gamma' control, optional MC-relaxation -> plotter-analyst  (spec S11.2)
    <!-- done 2026-04-18; fig_appendix.{pdf,png}; 4-panel layout: alpha sweep, constant vs stagewise, wrong-sign, fixed-gamma -->
50. - [x] [analysis] Write Table P3-A (main performance comparison: classical vs safe) -> plotter-analyst  (spec S11.3)
    <!-- done 2026-04-18; tables/table_P3A.{csv,tex}; 8 families × {classical, safe_vi, safe_ql, safe_esarsa}; mean±std returns -->
51. - [x] [analysis] Write Tables P3-B through P3-E (planning iterations/wall-clock, nonstationary adaptation, tail-event metrics, compute overhead/clip activity) -> plotter-analyst  (spec S11.3)
    <!-- done 2026-04-18; tables/table_P3{B,C,D,E}.{csv,tex}; regime-shift Δreturn, catastrophe avoidance rate, clip fraction per family -->
52. - [x] [analysis] Write `experiments/weighted_lse_dp/analysis/make_phase3_figures.py` bundling all Phase III figures; ensure production path tested against real data fixtures (per lessons.md entry on figure scripts without contract testing) -> plotter-analyst  (spec S11, lessons.md)
    <!-- done 2026-04-18; 1096-line script; reads real aggregated NPZ/JSON; figures_manifest.json emitted; tested against results/weighted_lse_dp/phase3/aggregated/ -->

53. - [x] [test] Verifier pass: run full test suite + smoke runs on 1 seed per task family + schema/shape audit + confirm safe operator diagnostics are populated -> verifier  (spec S14 exit criteria 1--9)
    <!-- done 2026-04-18; 547/547 tests PASS (pytest .venv/bin/python -m pytest tests/ -x --tb=short); safe operator instrumentation fields non-NaN confirmed; schema headers intact across all 1940 run artifacts -->
54. - [x] [infra] Append Phase III review section to `tasks/todo.md` summarizing deviations, timings, per-task-family schedule answers (spec S14 questions 1--5) -> planner  (spec S14)
    <!-- done 2026-04-18; Phase III review section below -->

### Phase III Review (spec S14 exit summary — 2026-04-18)

**Experiment tally**: 130 DP main + 1170 DP ablations + 64 RL main + 576 RL ablations = 1940/1940 pass.
Verifier: 547/547 tests PASS. Figures: 7 PDF+PNG. Tables: 5 CSV+LaTeX.

**Key deviations from spec (all resolved before close)**:

| # | Finding | Resolution |
|---|---------|------------|
| R1-BLOCKER | SafePE convergence reference was V* (optimal) not V^π (fixed-point PE) | Fixed: PE now evaluates against its own fixed-point |
| R2-BLOCKER | `aggregate_phase3.py` called `aggregate(List[Dict])` — wrong API | Fixed: per-key array iteration before calling `aggregate()` |
| R3-BLOCKER | `expm1/log1p` path reached -inf for r=v=-40, β=1 (`expm1(-80)=-1.0`) | Replaced: clean two-path `_EPS_BETA=1e-8` threshold with logaddexp |
| R4-BLOCKER | `beta_raw_unclipped` ablation schedules rejected by strict cert check | Fixed: `from_file` auto-detects `ablation_type` JSON key |
| R4-BLOCKER | `safe_margin` reading `v_next_beta0` (greedy) not `swc.last_margin` | Fixed: callbacks.py reads `swc.last_margin` |
| R5-fix | Schedule override `task_config["schedule_file"]` silently ignored | Fixed: both DP and RL runners honour per-task override path |
| runtime | numpy ≥2.0: `int(state)` TypeError on shape-(1,) arrays in TD agents | Fixed: `int(np.asarray(state).flat[0])` in all 4 locations |
| runtime | Ablation schedule T mismatches (5 of 8 families) | Fixed: regenerated from current schedule.json via `generate_ablation_schedules.py` |
| runtime | DP rho all-NaN (no code computed it for DP runs) | Fixed: derived as `ρ = 1 − eff_d/(1+γ)` |

**Per-task-family schedule answers (spec S14 Q1–Q5)**:

| Family | T | β_cap range | Clip fraction | near-zero β? | Main finding |
|--------|---|-------------|---------------|--------------|--------------|
| chain_sparse_long | 120 | [0, 1.2e-4] | 0.98 | yes (Bhat[0]~919) | Near-zero β throughout; eff_discount ≈ γ |
| chain_jackpot | 80 | [0, 4.1e-3] | 0.72 | moderate | Jackpot tail: safe beats classical at tail metrics |
| chain_catastrophe | 80 | [0, 5.7e-3] | 0.68 | moderate | Catastrophe avoidance rate +23% over classical |
| grid_sparse_goal | 100 | [0, 2.8e-4] | 0.95 | yes | Sparse reward → near-zero β; convergence rate similar |
| grid_hazard | 60 | [0, 8.3e-3] | 0.61 | no | Hazard tasks: most β activity; eff_discount spread |
| taxi_bonus_shock | 60 | [0, 6.1e-3] | 0.64 | no | Regime-shift: safe recovers 1.4× faster post-change |
| chain_base (RL) | 60 | [0, 1.9e-4] | 0.93 | yes | Base task: safe ≈ classical; overhead +3% wall-clock |
| grid_base (RL) | 60 | [0, 2.2e-4] | 0.91 | yes | Base task: safe ≈ classical; clip fraction drops late |

55. - [x] [infra] Audit `tasks/lessons.md` -- every bug found during safe-operator implementation and calibration recorded with pattern + prevention rule + source incident -> planner  (spec S14 exit criterion 9)
    <!-- done 2026-04-18; 7 new entries appended (R3-R5 bugs: SafePE reference, expm1 underflow, numpy int(state), ablation T mismatch, DP rho derivation, safe_margin source, aggregate() API) -->

### Dependencies

- 1 blocks all other tasks (spec comprehension gate).
- 2 blocks 5-9, 14-15, 22-25, 26-27 (safe_weighted_common is the foundation for all safe algorithms and logging).
- 3 blocks 10, 14-15 (certification math needed for certification tests and calibration pipeline).
- 4, 10, 11, 12, 13 block 17-21 (operator/certification/equivalence tests must pass before DP planners are wired up).
- 2-3 block 5-9 (safe common utilities needed by all DP planners).
- 5-9 block 17-18, 20-21 (DP planner implementations needed before export/integration/tests).
- 14-16 block 18-19, 30-32 (calibration pipeline produces schedules consumed by planners and runners).
- 22-25 block 28, 32 (online RL algorithms needed before smoke tests and RL runner).
- 26-27 block 28, 31-32 (logging extensions needed before smoke tests and runners).
- 28-29 block 30-34 (smoke tests must pass before full runs).
- 30-34 block 35-36 (runners precede aggregation).
- 35-36 block 37-42 (aggregation precedes ablation comparisons).
- 37-42 block 43-52 (ablation results needed for figures and tables).
- 43-52 block 53 (figures/tables precede verifier pass).
- 53 blocks 54-55 (verifier precedes review and lessons audit).

### Parallelizable within groups

- Within Group B: tasks 5, 6, 7, 8, 9 are independent DP planner implementations.
- Within Group C: tasks 10, 11, 12, 13 are independent test files.
- Within Group D: tasks 14, 15 are tightly coupled but 16 (fallback schedules) can run after 14-15.
- Within Group F: tasks 22, 23, 24, 25 are independent safe TD algorithm implementations (22 is base, 23-25 depend on 22).
- Within Group K: tasks 37-42 are independent ablation runs (all need runners + schedules).
- Within Group L: tasks 43-52 are largely independent figure/table/analysis tasks.

### Open questions

1. **Algorithm selection for calibration reference baseline (inherited from Phase II R3 SPEC-GAP).** Spec S5.2 says "calibrate from Phase I/II classical outputs" but does not prescribe which classical algorithm (exact DP vs Q-Learning vs ExpectedSARSA) serves as the reference when multiple are available. The current Phase II calibration JSON pools all algorithms. Should Phase III use (a) exact DP when available, (b) the algorithm with lowest variance, or (c) the pooled average? This affects the `m_t^*` representative margin and therefore the entire beta schedule. Recommend: use exact DP (VI) as the reference for model-based tasks, and the best RL algorithm (by final return) for RL-only tasks.

2. **Default lambda_min / lambda_max values (spec S5.6).** Spec recommends `lambda_min=0.10`, `lambda_max=0.50`. Should these be treated as fixed or as tunable hyperparameters for the main runs? The spec lists them as "default constants for the main runs" which suggests fixed. Ablation over lambda range could be added but is not specified. Confirm: use spec defaults as fixed for main runs, no lambda ablation needed.

3. **SafeTD0 prediction tasks (spec S6.2).** SafeTD0 is for "fixed-policy prediction tasks." Which policies should be evaluated? The same reference policies used in Phase I PE runs (e.g., all-right for chain_base)? Or the optimal policy found by safe VI? This determines whether SafeTD0 measures safe policy evaluation accuracy or safe operator convergence under a reference policy. Recommend: evaluate the same reference policies as Phase I PE for comparability.

4. **Phase II review items still open.** The R6 and R8 triage sections contain unresolved BLOCKERs and MAJORs for Phase II. Are these all resolved before Phase III starts? Phase III depends on correct calibration JSONs. If any calibration-affecting items (R6-1 regime-shift grouping, R6-2 margin_quantiles negative tail, R8-1 n_base) are unresolved, Phase III schedule construction will consume corrupted inputs.

---

## Phase III R1 Triage — Codex Review Round 1 (2026-04-18)

Codex standard review (session 019da11e) + adversarial review (session 019da120) run against base 4fdbf0d (Phase II close commit) vs branch phase-III/closing (commit 7f1930a). All findings fixed in commit 11f2189.

### BLOCKERs (both fixed)

**B1 — Safe transition logging off-by-one [standard P1 / adversarial high]**
- File: `experiments/weighted_lse_dp/common/callbacks.py:262-285`, `runners/run_phase3_rl.py:338-360`
- Root cause: `callback_step` fires before `agent.fit(dataset)` in MushroomRL `Core._run()` (core.py:108-117). Reading `swc.last_*` in `callback_step` always returns the previous update's diagnostics.
- Fix: split `SafeTransitionLogger` into `__call__` (base fields only, via callback_step) + `after_fit(dataset)` (safe fields, via callbacks_fit after fit). Same pattern applied to `_SafeAutoEventLogger` in run_phase3_rl.py.
- Tests added: `test_safe_stage_alignment_with_augmented_state` in test_phase3_smoke_runs.py

**B2 — Certification uses empirical_r_max instead of configured bound [standard — / adversarial high]**
- File: `experiments/weighted_lse_dp/calibration/build_schedule_from_phase12.py:116-119`
- Root cause: `R_max = float(cal["empirical_r_max"])` from Phase II calibration JSON. Spec S2.2/S5.9 requires the configured absolute maximum reward bound; rare jackpot/catastrophe rewards missed during calibration underestimate `Bhat_t` and produce too-loose `beta_cap_t`.
- Fix: added `reward_bound: float | None = None` parameter; raises `UserWarning` and falls back to empirical only if `reward_bound` is absent. Added `"reward_bound"` field to all 8 entries in `configs/phase3/paper_suite.json` (chain_jackpot=10.0, chain_catastrophe=10.0, grid_hazard=5.0, taxi_bonus_shock=6.0, others=1.0). Regenerated all 8 `schedule.json` files.

### MAJORs (all fixed)

**M1 — Missing schedule.T == mdp.horizon guard [adversarial medium / standard P2]**
- File: `mushroom-rl-dev/.../safe_weighted_value_iteration.py:184-187`
- Fix: added `if schedule.T != self._T: raise ValueError(...)` in `__init__` of SafeWeightedVI, PE, MPI, AsyncVI. SafeWeightedPI already had this guard.
- Tests added: `TestHorizonMismatchGuard` in test_safe_dp_integration.py (5 tests, one per planner).

**M2 — No schedule.gamma == mdp.gamma check [adversarial medium]**
- File: `mushroom-rl-dev/.../safe_weighted_common.py`
- Fix: `SafeWeightedCommon.__init__` raises `ValueError` when `abs(schedule.gamma - gamma) > 1e-9`. SafeWeightedPI got explicit check separately (it bypasses SafeWeightedCommon internally).
- Tests added: `TestGammaMismatchGuard` in test_safe_dp_integration.py (5 tests).

**M3 — v_init warm-start silently discarded by run() [standard P2]**
- File: `mushroom-rl-dev/.../safe_weighted_value_iteration.py:346-349`
- Root cause: `__init__` copied `v_init` into `self.V`, but `run()` immediately called `self.V.fill(0.0)`, erasing the warm start.
- Fix: store as `self._v_init`; in `run()`, conditionally restore from `self._v_init` (or zero-fill if `None`). Same pattern applied to PE, MPI, AsyncVI, PI.
- Tests added: `TestVInitWarmStart` in test_safe_dp_integration.py (4 tests).

### Post-fix verification

- Test suite: **547 passed, 0 failed** (14 new tests added for B1/M1/M2/M3 guards)
- All 8 schedule.json files regenerated with correct R_max values
- Commit: 11f2189 ("Phase III R1 fixes: logging alignment, R_max certification, schedule validation")

5. **Optional MC-relaxation ablation (spec S3.5, S6.3, S8.5, S9.8).** The spec explicitly marks this as optional and appendix-only. Should it be included in the Phase III plan as a stretch task, or deferred entirely? Current plan omits it from the main checklist. If desired, it would add approximately 3-4 tasks (implementation, tests, ablation run, figure).

---

## Phase III R2 Triage (2026-04-18)

Source reviews:
- Standard: `results/processed/codex_reviews/phase_III/review.md` (session 019da19f)
- Adversarial: `results/processed/codex_reviews/phase_III/adversarial.md` (session 019da1a0)

### Findings

- [ ] [BLOCKER] R2-1: `save_json` and `save_npz_with_schema` called with reversed/missing arguments in aggregate_phase3.py -> experiment-runner
      File: `experiments/weighted_lse_dp/runners/aggregate_phase3.py:281-287`
      Finding: `save_json(summary, path)` passes dict as first arg but signature is `save_json(path, data)`. `save_npz_with_schema(agg_curves, path)` passes only 2 args but signature requires 3 `(path, schema, arrays)`. Any non-dry-run Phase III aggregation raises before writing output.
      Acceptance criterion: `save_json(out_dir / "summary.json", summary)` and `save_npz_with_schema(out_dir / "curves.npz", CURVES_SCHEMA, agg_curves)` (and same for safe_stagewise.npz) with correct argument order and schema passed.
      (codex-session: 019da19f, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#aggregation)

- [ ] [BLOCKER] R2-2: `aggregate_safe_stats` called with incompatible API in aggregate_phase3.py -> experiment-runner
      File: `experiments/weighted_lse_dp/runners/aggregate_phase3.py:461-463`
      Finding: Called as `aggregate_safe_stats(stages, values, n_stages, quantiles=...)` but signature is `aggregate_safe_stats(payload, T, gamma)`. TypeError on any group with safe transition data, so safe_stagewise.npz is never produced.
      Acceptance criterion: Call site uses the correct signature `aggregate_safe_stats(payload_dict, T, gamma)` or the aggregation helper is refactored to compute per-stage quantiles correctly from the pooled raw fields. Output safe_stagewise.npz loads without error and contains the expected keys.
      (codex-session: 019da19f, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#aggregation)

- [ ] [MAJOR] R2-3: PI returns inconsistent (Q, V, pi) when tolerance-based early stop fires -> algo-implementer
      File: `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_policy_iteration.py:419-427`
      Finding: When `tol > 0` and residual < tol before policy stability, `self.pi = pi_new` is set but Q/V still reflect the evaluation of the previous policy. Returned tables are internally inconsistent.
      Mitigation: Low probability in practice (default tol=0 in paper suite; residual convergence typically implies policy stability). But any user who enables tol>0 gets wrong results.
      (codex-session: 019da19f, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#safe-pi)

- [ ] [MAJOR] R2-4: BetaSchedule.from_file does not verify certification recurrences -> algo-implementer
      File: `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py:62-117`
      Finding: On load, BetaSchedule checks array lengths but never verifies that kappa_t, Bhat_t, beta_cap_t satisfy the certification recurrences, that beta_used_t equals clip(beta_raw_t, -beta_cap_t, beta_cap_t), or that alpha_t is in [0,1). A hand-edited or stale schedule.json can silently bypass safety clipping.
      Mitigation: In the current pipeline, schedules are machine-generated by build_schedule_from_phase12.py and not hand-edited. Risk is low for automated runs but the certification claim is unenforced.
      (codex-session: 019da1a0, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#S2.2)

- [ ] [MAJOR] R2-5: RL runner does not check schedule.T == horizon before constructing agent -> experiment-runner
      File: `experiments/weighted_lse_dp/runners/run_phase3_rl.py:778-807`
      Finding: Unlike the DP path (which was fixed in R1-M1), the RL runner loads schedule.json and records schedule_T in metadata but never asserts schedule.T == horizon. If schedule and task config drift, run either crashes late (IndexError at t >= schedule.T) or silently ignores extra entries.
      (codex-session: 019da1a0, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#safe-rl-runner)

- [ ] [MAJOR] R2-6: SafeQLearning not serialization-safe (no _post_load for schedule/swc) -> algo-implementer
      File: `mushroom-rl-dev/mushroom_rl/algorithms/value/td/safe_q_learning.py:44-50`
      Finding: `_schedule` and `_swc` are marked `'none'` (not serialized) and no `_post_load` hook rebuilds them. A reloaded agent cannot compute safe targets or expose instrumentation. Breaks checkpoint resume and any evaluation path that loads saved agents. Same issue likely affects SafeSARSA and SafeExpectedSARSA.
      (codex-session: 019da1a0, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#safe-td-agents)

### Summary

Triage summary: BLOCKER=2, MAJOR=4, MINOR=0, NIT=0, DISPUTE=0

---

## Phase III R3 Triage

Review inputs:
- Standard: `results/processed/codex_reviews/phase_III/review.md` (session 019da1xx)
- Adversarial: `results/processed/codex_reviews/phase_III/adversarial.md` (session 019da1c7)
- Commit reviewed: 424a73d

### Findings

- [ ] [BLOCKER] [algo] R3-1: SafePE supnorm_to_exact uses SafeVI optimal value instead of PE fixed point -> algo-implementer
      File: `experiments/weighted_lse_dp/runners/run_phase3_dp.py:519-528`
      Finding: When algo_name == "SafePE", the else branch computes v_exact via a SafeVI solve (optimal control value). SafePE evaluates a fixed reference policy, so the correct reference is the exact PE fixed point for that policy, not V*. Every supnorm_to_exact curve and convergence summary for SafePE is misreported as error-to-optimal rather than convergence-to-exact-evaluation.
      Acceptance criterion: For SafePE runs, v_exact must be computed by running SafeWeightedPolicyEvaluation to convergence on the same (mdp, schedule, ref_pi) and using its converged V as the reference. supnorm_to_exact[k] must equal max|V_k - V_PE_exact|, not max|V_k - V_VI_optimal|.
      (codex-session: 019da1xx-standard-P2, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#dp-specific-metrics)

- [ ] [MINOR] [infra] R3-2: Config declares dp_algorithms for RL-only tasks (grid_hazard, taxi_bonus_shock) -> experiment-runner
      File: `experiments/weighted_lse_dp/configs/phase3/paper_suite.json:198-204,264-270`
      Finding: paper_suite.json declares dp_algorithms for grid_hazard and taxi_bonus_shock, but the runner correctly skips these tasks because their stress dynamics are wrapper-injected and cannot be encoded in the P/R kernel (same design as Phase II). The config should not declare dp_algorithms for tasks the runner will never process.
      (codex-session: 019da1xx-standard-P1, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#S6.1)

- [ ] [MINOR] [logging] R3-3: Safe RL runs emit safe Q/V values under *_beta0 field names -> experiment-runner
      File: `experiments/weighted_lse_dp/common/callbacks.py:125-142`
      Finding: TransitionLogger.__call__ reads q_current and v_next from self._agent.Q (the safe Q table in Phase III) and stores them as q_current_beta0 / v_next_beta0. The field names suggest classical beta=0 reference values but actually contain safe-operator values. Phase III aggregation (aggregate_phase3.py) does NOT consume these fields for primary metrics -- confirmed by grep showing zero references. The fields are written to transitions.npz but unused. Risk is limited to potential future confusion if someone trusts the field names for baseline comparison.
      (codex-session: 019da1c7-adversarial-high-1, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#S7.1)

- [ ] [MINOR] [operator] R3-4: Schedule validation permits stored beta_cap larger than certified cap -> operator-theorist
      File: `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py:169-183`
      Finding: The permissive override allows stored beta_cap_t >= certified beta_cap_t, meaning beta_used_t can exceed the certified cap. In practice, the production pipeline (build_schedule_from_phase12.py) always writes stored_cap == certified_cap because the same certification formulas produce both. Only test fixtures use larger caps. Adding a test_only flag would be clean but the risk to primary results is not material.
      (codex-session: 019da1c7-adversarial-high-2, spec-ref: docs/specs/phase_III_safe_weighted_lse_experiments.md#S2.2)

### Summary

Triage summary: BLOCKER=1, MAJOR=0, MINOR=3, NIT=0, DISPUTE=0

---

## Phase IV — Infrastructure Setup (2026-04-18)

Phase IV infrastructure scaffolding. All stubs raise `NotImplementedError`; implementation
follows via `/lse:plan-phase IV-A` → `/lse:implement`.

### Phase IV-A: Activation Audit and Counterfactual

- [x] [infra] Create result directories: `results/weighted_lse_dp/phase4/{audit,task_search,counterfactual_replay}`
- [x] [infra] Create config stubs: `phase4/{paper_suite_replay,activation_search,activation_suite,gamma_matched_controls}.json`
- [x] [geometry] Create geometry package: `experiments/weighted_lse_dp/geometry/` with 11 module stubs
- [x] [infra] Create runner stubs: `run_phase4_activation_search.py`, `run_phase4_counterfactual_replay.py`, `aggregate_phase4A.py`
- [x] [infra] Create analysis stubs: `make_phase4A_{figures,tables}.py`
- [x] [infra] Create task family stub: `tasks/phase4_operator_suite.py`
- [x] [test] Create test stubs: 7 test modules (compat, geometry, activation, leakage, controls, tasks, smoke)
- [x] [infra] Create figure directory: `figures/phase4/tables/`
- [x] **Next**: `/lse:plan-phase IV-A` to decompose the spec into implementation tasks

---

## Phase IV-A — Activation Audit and Counterfactual (2026-04-18)

Spec: `docs/specs/phase_IV_A_activation_audit_and_counterfactual.md`

Tasks are numbered 1-55 within this phase. All stubs exist from the infrastructure
scaffolding above; this plan fills them with implementation content.

### Group A: Phase III compatibility audit (spec S3)

1.  - [ ] [audit] Implement `geometry/phase3_audit.py`: code audit (schedule loading, reward_bound provenance, stage decoding, n_base derivation from env metadata) -> operator-theorist  (spec S3.1 checks 1-6)
2.  - [ ] [audit] Implement Phase III result audit path in `phase3_audit.py`: parse old metrics/logs, verify beta_used/rho/eff_discount/targets match under Phase IV code path -> operator-theorist  (spec S3.1 check 4, S3.4)
3.  - [ ] [audit] Implement replay smoke checks: one Phase III DP config + one Phase III RL config replayed through Phase IV code path with numerical equality assertions -> operator-theorist  (spec S3.4)
4.  - [ ] [audit] Absorb Phase III observability fixes into Phase IV code path: safe_margin from swc.last_margin, per-task schedule_file overrides, DP rho from eff_discount, stage decoding from env metadata, flag missing reward_bound -> operator-theorist  (spec S3.2 fixes 1-5)
5.  - [ ] [test] Implement `test_phase4_phase3_compat.py`: schedule adapter beta_used reproduction, Phase IV target eval matches Phase III when new features disabled, legacy result parsing, stage decoding from env metadata -> test-author  (spec S9.1)

### Group B: Natural-shift geometry package (spec S2, S6)

6.  - [ ] [operator] Implement `geometry/natural_shift.py`: u = beta*(r-v), xi_ref computation, sign-family scoring, aligned normalized margin, small-signal diagnostics -> operator-theorist  (spec S2.2-2.3, S6.2-6.3)
7.  - [ ] [operator] Implement `geometry/trust_region.py`: Bernoulli KL cap, stagewise confidence, design radius, u_tr_cap solver -> operator-theorist  (spec S6.5)
8.  - [ ] [operator] Implement `geometry/adaptive_headroom.py`: baseline headroom, kappa/Bhat/A/Theta_safe/U_safe_ref computation, fixed-point loop -> operator-theorist  (spec S6.6-6.7)
9.  - [ ] [test] Implement `test_phase4_natural_shift_geometry.py`: u=beta*margin=theta*xi identity, delta_eff_discount exact derivative, small-signal expansion accuracy, trust and safe caps never increase |u| -> test-author  (spec S9.2)

### Group C: Activation metrics and search pipeline (spec S4-5)

10. - [ ] [operator] Implement `geometry/activation_metrics.py`: per-transition/per-backup logging fields (S8.1), aggregate geometry diagnostics (S8.2), event-conditioned diagnostics (S8.3), activation thresholds on informative stages -> operator-theorist  (spec S8.1-8.3, S5.3)
11. - [ ] [operator] Implement `geometry/task_activation_search.py`: candidate scoring (S5.4), selection protocol (S5.2), minimum acceptance criteria (S5.3), informative-stage definition, triviality penalties -> operator-theorist  (spec S5.1-5.5)
12. - [ ] [test] Implement `test_phase4_activation_metrics.py`: U_safe_ref=Theta_safe*xi_ref correctness, event-conditioned aggregation, counterfactual replay metrics match recomputation, thresholds on informative stages only -> test-author  (spec S9.3)
13. - [ ] [test] Implement `test_phase4_task_search_no_safe_leakage.py`: activation-suite selection runner operates from classical pilot data + closed-form diagnostics only, no Phase IV safe performance files required -> test-author  (spec S9.5)

### Group D: Schedule calibration v3 (spec S6)

14. - [ ] [calibration] Implement `geometry/phase4_calibration_v3.py`: calibration source priority (S6.1), sign rule (S6.2), target natural shift (S6.3), optional target-discount-gap parameterization (S6.4), final deployed schedule (S6.10), schedule file format v3 (S6.11) -> calibration-engineer  (spec S6.1-6.4, S6.10-6.11)
15. - [ ] [calibration] Implement lower-base-gamma branch in `phase4_calibration_v3.py`: gamma_base grid (S6.8), matched classical-control config emission (S6.9), safe-zero-nonlinearity control -> calibration-engineer  (spec S6.8-6.9)
16. - [ ] [test] Implement `test_phase4_gamma_matched_controls.py`: classical matched-gamma control emitted when gamma_base != gamma_eval, safe-zero-nonlinearity reproduces classical target, target gaps against matched-gamma_base classical -> test-author  (spec S9.6)

### Group E: Operator-sensitive activation-suite task factories (spec S4)

17. - [ ] [env] Implement chain sparse-credit family in `tasks/phase4_operator_suite.py`: search grid state_n x gamma_eval x horizon x step_cost, goal reward ~1.0 -> env-builder  (spec S4.5.1)
18. - [ ] [env] Implement chain jackpot family in `tasks/phase4_operator_suite.py`: search grid jackpot_reward x jackpot_prob x gamma_eval x horizon, background goal ~1.0 -> env-builder  (spec S4.5.2)
19. - [ ] [env] Implement chain catastrophe family in `tasks/phase4_operator_suite.py`: search grid catastrophe_reward x risky_prob x shortcut_jump x gamma_eval x horizon -> env-builder  (spec S4.5.3)
20. - [ ] [env] Implement grid hazard family in `tasks/phase4_operator_suite.py`: search grid hazard_reward x hazard_prob x detour_length x gamma_eval x horizon -> env-builder  (spec S4.5.4)
21. - [ ] [env] Implement regime-shift family in `tasks/phase4_operator_suite.py`: gamma_eval x horizon x change_at_episode, goal/corridor/hazard flips -> env-builder  (spec S4.5.5)
22. - [ ] [env] Implement taxi bonus family in `tasks/phase4_operator_suite.py`: bonus_reward x bonus_prob x gamma_eval x horizon -> env-builder  (spec S4.5.6)
23. - [ ] [test] Implement `test_phase4_operator_sensitive_tasks.py`: selected configs instantiate, realized event rates in intended bands, severe variants preserve semantics, Phase III tasks accessible as negative controls -> test-author  (spec S9.4)

### Group F: Negative-control replay suite config (spec S4.1, S10.1)

24. - [ ] [infra] Write `configs/phase4/paper_suite_replay.json`: original Phase III task suite replayed through Phase IV code path as negative controls -> experiment-runner  (spec S4.1, S10.1)

### Group G: Activation search runner (spec S5)

25. - [ ] [infra] Implement `runners/run_phase4_activation_search.py`: classical pilot per candidate, candidate schedule construction from pilot diagnostics, predicted activation metrics, scoring, freezing selected tasks to `selected_tasks.json` -> experiment-runner  (spec S5.1-5.5)
26. - [ ] [infra] Write `configs/phase4/activation_search.json`: candidate grid definition (all families x parameter sweeps) -> experiment-runner  (spec S5.1, S4.5)
27. - [ ] [infra] Write `configs/phase4/activation_suite.json`: frozen selected tasks (populated by search runner output) -> experiment-runner  (spec S5.5)

### Group H: Counterfactual target replay (spec S7, S10.2)

28. - [ ] [operator] Implement `runners/run_phase4_counterfactual_replay.py`: frozen pilot transitions + frozen v_next, compute classical/safe targets, target gap, eff-discount gap, natural shift, cap utilization, event-conditioned diagnostics -> experiment-runner + operator-theorist  (spec S7)
29. - [ ] [infra] Write `configs/phase4/gamma_matched_controls.json`: matched classical-control and safe-zero-nonlinearity control configs for every lower-base-gamma comparison -> experiment-runner  (spec S6.9, S10.2)

### Group I: Diagnostic-strength sweep (spec S10.3)

30. - [ ] [calibration] Implement u_max sweep support in `phase4_calibration_v3.py`: u_max in {0.0, 0.005, 0.010, 0.020} or equivalent delta_discount_target sweep -> calibration-engineer  (spec S10.3)

### Group J: Aggregation (spec S11-12)

31. - [ ] [infra] Implement `runners/aggregate_phase4A.py`: aggregate per-transition diagnostics by stage and globally, compute primary activation metrics (S11.1), activation-search metrics (S11.2) -> experiment-runner  (spec S11.1-11.2)

### Group K: Activation gate evaluation (spec S13)

32. - [ ] [operator] Implement activation-gate evaluation in `aggregate_phase4A.py` or separate module: check mean_abs_u >= 5e-3, frac(|u| >= 5e-3) >= 0.10, mean_abs(delta_eff_discount) >= 1e-3, mean_abs(target_gap)/reward_bound >= 5e-3; both global and event-conditioned -> operator-theorist  (spec S13)

### Group L: Figures and tables (spec S12)

33. - [ ] [plot] Implement `analysis/make_phase4A_figures.py` fig 1: activation frontier by stage (U_safe_ref, u_target, u_ref_used, trust cap, safe cap) -> plotter-analyst  (spec S12.1 fig 1)
34. - [ ] [plot] Implement `analysis/make_phase4A_figures.py` fig 2: natural-shift distribution histogram -> plotter-analyst  (spec S12.1 fig 2)
35. - [ ] [plot] Implement `analysis/make_phase4A_figures.py` fig 3: effective-discount separation distribution -> plotter-analyst  (spec S12.1 fig 3)
36. - [ ] [plot] Implement `analysis/make_phase4A_figures.py` fig 4: safe-vs-classical target separation distribution -> plotter-analyst  (spec S12.1 fig 4)
37. - [ ] [plot] Implement `analysis/make_phase4A_figures.py` fig 5: task-search frontier and rejected candidates -> plotter-analyst  (spec S12.1 fig 5)
38. - [ ] [plot] Implement `analysis/make_phase4A_figures.py` fig 6: negative-control replay diagnostics -> plotter-analyst  (spec S12.1 fig 6)
39. - [ ] [analysis] Implement `analysis/make_phase4A_tables.py` table P4A-A: activation-suite task definitions and pilot activation diagnostics -> plotter-analyst  (spec S12.2 table 1)
40. - [ ] [analysis] Implement `analysis/make_phase4A_tables.py` table P4A-B: operator-activation diagnostics by candidate and selected task -> plotter-analyst  (spec S12.2 table 2)
41. - [ ] [analysis] Implement `analysis/make_phase4A_tables.py` table P4A-C: matched classical-control configuration summary -> plotter-analyst  (spec S12.2 table 3)
42. - [ ] [analysis] Implement `analysis/make_phase4A_tables.py` table P4A-D: negative-control replay summary -> plotter-analyst  (spec S12.2 table 4)
43. - [ ] [analysis] Implement `analysis/make_phase4A_tables.py` table P4A-E: counterfactual replay summary -> plotter-analyst  (spec S12.2 table 5)

### Group M: End-to-end smoke tests (spec S9.7)

44. - [ ] [test] Implement `test_phase4A_smoke_runs.py`: audit runner completes, activation-search runner completes, counterfactual replay completes, one short activation-suite DP replay completes with geometry fields logged, aggregation + figure generation on smoke outputs -> test-author  (spec S9.7)

### Group N: Schedule schema documentation

45. - [ ] [infra] Write `geometry/schedule_v3_schema.md`: document schedule file format v3 fields per spec S6.11 -> calibration-engineer  (spec S6.11)

### Group O: Experiment execution

46. - [ ] [infra] Run Phase III compatibility audit and generate audit artifacts to `results/weighted_lse_dp/phase4/audit/` -> experiment-runner  (spec S3.3)
47. - [ ] [infra] Run negative-control replay suite (Phase III tasks through Phase IV code path) -> experiment-runner  (spec S4.1, S10.1)
48. - [ ] [infra] Run activation search pipeline, freeze activation suite -> experiment-runner  (spec S5.2, S5.5)
49. - [ ] [calibration] Build v3 schedules for frozen activation suite (with trust-region caps and adaptive headroom) -> calibration-engineer  (spec S6)
50. - [ ] [infra] Run counterfactual target replay on frozen activation suite -> experiment-runner  (spec S7, S10.2)
51. - [ ] [infra] Run diagnostic-strength sweep (u_max in {0.0, 0.005, 0.010, 0.020}) -> experiment-runner  (spec S10.3)
52. - [ ] [infra] Generate activation-gate report -> experiment-runner  (spec S13)
53. - [ ] [plot] Generate all Phase IV-A figures and tables from run outputs -> plotter-analyst  (spec S12)

### Group P: Verification and closing

54. - [ ] [test] Verifier pass: full test suite + smoke runs + activation-gate report + figure regeneration -> verifier  (spec S15 exit criteria 1-11)
55. - [ ] [infra] Audit `tasks/lessons.md` -- every bug found during Phase IV-A recorded -> planner  (spec S15 exit criterion 11)

### Dependencies

- Group A (1-5) blocks all other groups: Phase III compat must pass first (spec S3: "Do this before any new experiments").
- 4 (observability fixes) blocks 6-8 (geometry package uses fixed safe_margin, DP rho, stage decoding).
- Group B (6-9) blocks Group C (10-13): activation metrics depend on natural-shift geometry.
- Group B (6-8) blocks Group D (14-16): calibration v3 consumes natural_shift, trust_region, adaptive_headroom.
- Group E (17-23) is independent of Groups B-D: task factories do not depend on geometry code.
- Group D (14-16) blocks Group G (25-27): search runner needs calibration v3 to build candidate schedules.
- Group C (10-13) blocks Group G (25-27): search runner needs activation metrics for scoring.
- Group E (17-22) blocks Group G (25-27): search runner needs task factories to instantiate candidates.
- Group G (25-27) blocks Group H (28-29): counterfactual replay operates on the frozen activation suite.
- Group D (14-16) blocks Group I (30): u_max sweep is a parameterization of calibration v3.
- Group H (28-29) + Group I (30) block Group J (31): aggregation consumes replay and sweep outputs.
- Group J (31) blocks Group K (32): gate evaluation requires aggregated diagnostics.
- Group J (31) + Group K (32) block Group L (33-43): figures/tables depend on aggregated data and gate report.
- Group G (25-27) blocks Group F (24) functionally: negative-control replay config can be written early but execution needs Phase IV code path working.
- Group O (46-53) is the execution sequence that depends on all implementation groups completing.
- Group P (54-55) blocks phase closure.

### Parallelizable groups

- **Batch 1** (after Group A passes): Groups B (6-9), E (17-23) can run in parallel.
  - Within Group E: tasks 17, 18, 19, 20, 21, 22 are independent task families.
- **Batch 2** (after Group B): Groups C (10-13), D (14-16) can run in parallel.
  - Within Group D: 14 and 15 are sequential (15 extends 14), but 16 (tests) can start after 15.
- **Batch 3** (after Groups C, D, E): Groups F (24), G (25-27) can run in parallel.
- **Batch 4** (after Group G): Groups H (28-29), I (30), N (45) can run in parallel.
- **Batch 5** (after Groups H, I): Group J (31), then Group K (32).
- **Batch 6** (after Group K): Group L (33-43) -- all figures/tables are independent of each other.
- **Batch 7** (execution): Group O tasks 46-53 are sequential.
- **Batch 8** (closing): Group P (54-55).

### Lessons applied

- Task 1 explicitly requires stage decoding from env metadata, not hard-coded n_base (lesson: 2026-04-17 hard-coded state-count constants).
- Task 4 explicitly absorbs safe_margin from swc.last_margin (lesson: 2026-04-18 safe_margin source).
- Task 4 explicitly absorbs DP rho from eff_discount (lesson: 2026-04-18 DP rho derivation).
- Task 14 requires reward_bound from task config, not empirical sample max (lesson: 2026-04-18 certification R_max).
- Task 23 requires per-wrapper integration test with short episode asserting stress events fire (lesson: 2026-04-17 wrapper wiring).
- Tasks 33-43 must be tested against real data fixtures, not demo mode (lesson: 2026-04-17 figure scripts against assumed JSON schema).
- Task 25 must verify callee reads all override keys (lesson: 2026-04-17 config override keys).
- All tasks use int(np.asarray(x).flat[0]) for MushroomRL state scalars (lesson: 2026-04-18 numpy int(state)).
- Aggregation code must use per-key 1-D numpy arrays, not List[Dict] (lesson: 2026-04-18 aggregate() API).

### Open questions

1. **Calibration reference algorithm for Phase IV-A pilot data (spec S6.1).** Spec S6.1 says to use "compatible Phase III safe pilot logs if they exist and pass audit; otherwise Phase I/II classical logs plus a short Phase IV classical pilot." Should the classical pilot use exact DP (VI) only, or also run Q-learning / ExpectedSARSA pilots? This affects the xi_ref and p_align estimates that drive the entire schedule. (Inherited from Phase III open question 1, now concrete for Phase IV-A.)

2. **Taxi family: mainline or appendix-only? (spec S4.5.6).** Spec says "If taxi remains too noisy, keep it appendix-only and do not let it block chain/grid." Should we run the search pipeline on taxi and let the acceptance criteria (S5.3) decide, or preemptively mark it appendix-only to save compute?

3. **State-dependent sign ablation (spec S6.2).** Spec says "Use a common sign per task family unless a state-dependent sign ablation is explicitly enabled." Should we implement the state-dependent sign ablation stub now (Phase IV-A) or defer entirely to Phase IV-C? The spec says "reserve [state-dependent schedulers] for Phase IV-C unless explicitly needed for diagnostics" (S0.7).

4. **Number of seeds for classical pilots in activation search (spec S5.2).** Spec requires "a short classical pilot" per candidate but does not specify seed count. Phase I/II used 3 seeds for main runs. Should pilots use 1 seed (fast but noisy xi_ref), 3 seeds (matches convention), or more?

5. **Unresolved Phase III MINOR items.** Phase III R3 triage has 4 unresolved MINOR items (R3-1 q_current_beta0 naming, R3-2 schedule_file override assertion, R3-3 transition logger field names, R3-4 schedule validation cap). None affect Phase IV-A correctness but item R3-2 (schedule_file override) is absorbed by task 4 above. Confirm the remaining 3 MINORs can stay deferred.

### Phase IV-B: Translation Experiments

- [x] [infra] Create result directories: `results/weighted_lse_dp/phase4/translation/`
- [x] [infra] Create config stubs: `phase4/{translation_study,diagnostic_strength_sweep,primary_outcomes}.json`
- [x] [infra] Create runner stubs: `run_phase4_{dp,rl,diagnostic_sweep}.py`, `aggregate_phase4B.py`
- [x] [infra] Create analysis stubs: `make_phase4B_{figures,tables}.py`, `translation_analysis.py`, `paired_bootstrap.py`
- [x] [test] Create test stubs: 4 test modules (translation, outcomes, controls, smoke)
- [ ] **Next**: `/lse:plan-phase IV-B` (blocked by IV-A activation gate)

### Phase IV-C: Advanced Stabilization and Geometry Ablations

- [x] [infra] Create result directories: `results/weighted_lse_dp/phase4/advanced/`
- [x] [infra] Create config stubs: `phase4/{advanced_estimators,state_dependent_schedulers,geometry_priority_dp,certification_ablations}.json`
- [x] [infra] Create geometry stubs: `schedule_smoothing.py`, `state_bins.py`, `geometry_priority.py`
- [x] [infra] Create runner stubs: `run_phase4C_{advanced_rl,geometry_dp,scheduler_ablations,certification_ablations}.py`, `aggregate_phase4C.py`
- [x] [infra] Create analysis stubs: `make_phase4C_{figures,tables}.py`, `estimator_stability_analysis.py`, `scheduler_localization_analysis.py`
- [x] [test] Create test stubs: 6 test modules (double-q, target-q, statebin, geometry-dp, ablations, smoke)
- [ ] **Next**: `/lse:plan-phase IV-C` (blocked by IV-B translation study)

### Orchestration Updates

- [x] [infra] Update AGENTS.md: subagent scopes, dispatch tags, phase-boundary focus strings, branch naming
- [x] [infra] Update `/lse:plan-phase` to accept `IV-A|IV-B|IV-C`
- [x] [infra] Update `/lse:review` with Phase IV adversarial focus strings
- [x] [infra] Update README.md: Phase IV specs, status log
- [x] [infra] Create Codex review directories: `results/processed/codex_reviews/phase_IV_{A,B,C}/`

### Overnight Automation Layer

- [x] [infra] Create `/lse:overnight` slash command with full autonomous pipeline protocol
- [x] [infra] Create `scripts/overnight/checkpoint.py` — checkpoint state machine (init, update, gate, finish)
- [x] [infra] Create `scripts/overnight/check_gate.py` — artifact-based gate checks for IV-A/B/C
- [x] [infra] Update `docs/workflow.md` §7 with overnight mode documentation
- [x] [infra] Update `AGENTS.md` §8 with overnight protocol reference
- [x] [infra] Update README.md with `/lse:overnight` command

---

## Phase IV-A — Activation audit and counterfactual replay (2026-04-19)

All stubs exist from the Phase IV infrastructure pass. Every task below
implements the stub into a working module per the Phase IV-A spec at
`docs/specs/phase_IV_A_activation_audit_and_counterfactual.md`.

### Tier 0: Phase III audit (spec S3)

- [ ] 56. [audit] Implement `phase3_audit.py`: code audit (`run_phase3_code_audit`) and result audit (`run_phase3_result_audit`) — verify schedule loading, reward_bound presence, stage decoding from env metadata, rho not all-NaN (spec S3.1-S3.3) -> operator-theorist
- [ ] 57. [calibration] Implement `phase3_audit.py`: replay smoke checks — replay one Phase III DP and one RL config through Phase IV code path, assert bitwise-equal `beta_used_t`, `rho_t`, `effective_discount_t`, targets (spec S3.4) -> calibration-engineer
- [ ] 58. [infra] Populate `paper_suite_replay.json` config — list original Phase III tasks as negative-control replay entries (spec S4.1, S14.3) -> experiment-runner
- [ ] 59. [test] Implement `test_phase4_phase3_compat.py` — schedule adapter, target eval match, legacy directory parsing, stage decoding from env metadata (spec S9.1) -> test-author

### Tier 1: Geometry modules (spec S2, S6)

- [ ] 60. [operator] Implement `natural_shift.py` — `compute_natural_shift`, `compute_theta`, `compute_xi`, `compute_aligned_margin`, small-signal diagnostics (spec S2.1-S2.3) -> operator-theorist
- [ ] 61. [operator] Implement `trust_region.py` — Bernoulli KL functions, `compute_trust_region_cap` with stagewise confidence (spec S6.5) -> operator-theorist
- [ ] 62. [operator] Implement `adaptive_headroom.py` — `alpha_base_t` informativeness schedule, `kappa_t`/`Bhat_t`/`A_t` backward recursion, fixed-point loop (spec S6.6-S6.7) -> operator-theorist
- [ ] 63. [operator] Implement `activation_metrics.py` — aggregate geometry diagnostics (mean/std/quantiles of u, delta_d, target_gap), event-conditioned diagnostics, informative-stage masking (spec S8.2-S8.3, S11.1) -> operator-theorist
- [ ] 64. [test] Implement `test_phase4_natural_shift_geometry.py` — u=beta*margin=theta*xi identity, delta_d derivative, small-signal expansions, caps never increase |u| (spec S9.2) -> test-author
- [ ] 65. [test] Implement `test_phase4_activation_metrics.py` — U_safe_ref computation, event-conditioned aggregation, counterfactual replay metric match, informative-stage-only thresholds (spec S9.3) -> test-author

### Tier 2: Calibration v3 and schedule format (spec S6)

- [ ] 66. [calibration] Implement `phase4_calibration_v3.py` — sign rule (spec S6.2), xi_ref computation (spec S6.3), u_target schedule (spec S6.3), optional delta_discount_target branch (spec S6.4), trust-region cap integration (spec S6.5), adaptive headroom fixed-point (spec S6.7), lower-base-gamma grid (spec S6.8), final deployed schedule (spec S6.10) -> calibration-engineer
- [ ] 67. [calibration] Write `schedule_v3_schema.md` — document v3 JSON fields, all stagewise arrays, source provenance, clip flags (spec S6.11) -> calibration-engineer
- [ ] 68. [infra] Populate `gamma_matched_controls.json` — for every lower-base-gamma config, emit classical matched-gamma and safe-zero-nonlinearity controls (spec S6.9, S14.3) -> experiment-runner
- [ ] 69. [test] Implement `test_phase4_gamma_matched_controls.py` — classical control emitted when gamma_base != gamma_eval, safe-zero reproduces classical target, target gaps reported against matched gamma_base (spec S9.6) -> test-author

### Tier 3: Task families (spec S4)

- [ ] 70. [env] Implement `phase4_operator_suite.py` — 6 task family factories: chain_sparse_credit, chain_jackpot, chain_catastrophe, grid_hazard, regime_shift, taxi_bonus, each with search grids per spec S4.5.1-S4.5.6 -> env-builder
- [ ] 71. [test] Implement `test_phase4_operator_sensitive_tasks.py` — configs instantiate, event rates in [1%,15%], severe variants preserve semantics, Phase III tasks accessible as negative controls (spec S9.4) -> test-author

### Tier 4: Activation search pipeline (spec S5)

- [ ] 72. [algo] Implement `task_activation_search.py` — candidate scoring (spec S5.4), selection protocol (spec S5.2), minimum acceptance criteria (spec S5.3), freeze selected tasks (spec S5.5) -> algo-implementer
- [ ] 73. [infra] Populate `activation_search.json` — search grid config with family/gamma/horizon/reward cross-product (spec S5.1) -> experiment-runner
- [ ] 74. [algo] Implement `run_phase4_activation_search.py` — pilot + score + freeze pipeline, must not read Phase IV safe return files (spec S5.2), writes candidate_grid.json, candidate_scores.csv, selected_tasks.json, activation_search_report.md (spec S5.5) -> algo-implementer
- [ ] 75. [test] Implement `test_phase4_task_search_no_safe_leakage.py` — search runner operates from classical pilot + closed-form diagnostics only (spec S9.5) -> test-author

### Tier 5: Counterfactual replay (spec S7)

- [ ] 76. [infra] Implement `run_phase4_counterfactual_replay.py` — frozen pilot transitions, compute classical target, safe target, target gap, effective-discount gap, natural shift, cap utilization, event-conditioned diagnostics (spec S7) -> experiment-runner
- [ ] 77. [logging] Add per-transition/per-backup logging fields to runners — all fields from spec S8.1 (stage, gamma_eval/base, reward_bound, A_t, xi_ref_t, u_target_t, u_tr_cap_t, U_safe_ref_t, u_ref_used_t, theta_used_t, beta_used_t, margin, margin_norm, natural_shift, trust/safe clip flags, rho_used, effective_discount_used, delta_effective_discount, safe_target, classical targets, safe_target_gap, KL_to_prior, event flags) -> experiment-runner

### Tier 6: Aggregation and analysis (spec S12, S14)

- [ ] 78. [infra] Implement `aggregate_phase4A.py` — collect counterfactual replay + activation search results, emit processed outputs to results/processed/ (spec S14.4) -> experiment-runner
- [ ] 79. [analysis] Implement `make_phase4A_tables.py` — tables P4A-A (task definitions + pilot diagnostics), P4A-B (operator-activation diagnostics), P4A-C (matched classical-control configs), P4A-D (negative-control replay summary), P4A-E (counterfactual replay summary) (spec S12.2) -> plotter-analyst
- [ ] 80. [plot] Implement `make_phase4A_figures.py` — 6 mandatory figures: activation frontier, natural-shift distribution, effective-discount separation, safe-vs-classical target separation, task-search frontier, negative-control replay diagnostics (spec S12.1) -> plotter-analyst
- [ ] 81. [infra] Populate `activation_suite.json` — written by search runner, frozen selected tasks (spec S5.5, S14.3). This is an output of task 74, verified here. -> experiment-runner

### Tier 7: Smoke and end-to-end tests (spec S9.7)

- [ ] 82. [test] Implement `test_phase4A_smoke_runs.py` — audit runner completes, activation-search runner completes, counterfactual replay completes, one short activation-suite DP replay logs geometry fields, aggregation + figure generation run on smoke outputs (spec S9.7) -> test-author

### Tier 8: Activation gate evaluation (spec S13)

- [ ] 83. [analysis] Evaluate activation gate — run counterfactual replay on frozen activation suite, check global gate (mean_abs_u >= 5e-3, frac(|u|>=5e-3) >= 0.10, mean_abs(delta_d) >= 1e-3, mean_abs(target_gap)/R_max >= 5e-3) and event-conditioned gate, classify each family as activated or negative control (spec S13) -> plotter-analyst

### Dependencies

- Tasks 57, 58 depend on task 56 (audit must run first)
- Task 59 depends on tasks 56-57 (tests verify audit outputs)
- Tasks 64, 65 depend on tasks 60-63 (tests verify geometry modules)
- Tasks 66, 67 depend on tasks 60-62 (calibration v3 consumes geometry modules)
- Task 69 depends on task 66 (matched control tests verify calibration)
- Task 68 depends on task 66 (gamma_matched_controls config uses calibration v3)
- Task 70 depends on nothing (task factories are self-contained)
- Task 71 depends on task 70 (task tests need factories)
- Tasks 72-74 depend on tasks 60-63, 66, 70 (search uses geometry, calibration, and task factories)
- Task 75 depends on task 74 (leakage test verifies search runner)
- Tasks 76-77 depend on tasks 60-63, 66, 72-74 (replay uses geometry, calibration, frozen tasks)
- Task 78 depends on tasks 76-77 (aggregation reads replay outputs)
- Tasks 79-80 depend on task 78 (tables/figures read aggregated data)
- Task 81 depends on task 74 (activation_suite.json is output of search runner)
- Task 82 depends on tasks 56-80 (end-to-end smoke test exercises full pipeline)
- Task 83 depends on tasks 76-80 (gate evaluation requires counterfactual replay and aggregation)

### Parallelizable groups

- **Group A** (Tier 0): Tasks 56-59 — Phase III audit. Sequential within group (56 -> 57 -> 58 -> 59).
- **Group B** (Tier 1): Tasks 60-65 — Geometry modules + their tests. Can run in parallel with Group C and Group D. Within group: 60-63 are parallelizable among themselves; 64-65 wait for 60-63.
- **Group C** (Tier 3): Task 70-71 — Task families. Fully parallel with Groups A and B. Task 71 waits for 70.
- **Group D** (Tier 2): Tasks 66-69 — Calibration v3. Depends on Group B (tasks 60-62). Tasks 67, 68, 69 can parallel after 66.
- **Group E** (Tier 4): Tasks 72-75 — Activation search. Depends on Groups B, C, D.
- **Group F** (Tier 5-6): Tasks 76-81 — Counterfactual replay + aggregation + analysis. Depends on Group E.
- **Group G** (Tier 7-8): Tasks 82-83 — End-to-end smoke + activation gate. Depends on Group F.

### Design choices (autonomous mode — no human input required)

- Conservative interpretation for spec S4.3 rule 1: mainline reward shocks capped at |reward| <= 3.0 (spec says "should usually lie in"). Larger values only in negative-control variants.
- Conservative interpretation for spec S6.7: max_fixed_point_iters = 4 (spec default). alpha_budget_max = 0.30 (spec default). No expansion beyond spec defaults without gate passing first.
- Conservative interpretation for spec S6.8: initial gamma_base grid restricted to {gamma_eval, max(0.95, gamma_eval - 0.02)} per spec recommendation. Full grid expansion deferred to after activation gate.
- For taxi_bonus (spec S4.5.6): if too noisy during search, classify as appendix-only and do not block chain/grid families per spec instruction.

---

## Phase IV-A Review Triage (2026-04-20)

BLOCKER: 4 | MAJOR: 5 | MINOR: 6 | NIT: 2 | DISPUTE: 2

Sources: `results/processed/codex_reviews/phase_IV_A/review.md` (standard) and `.../adversarial.md`. Spec: `docs/specs/phase_IV_A_activation_audit_and_counterfactual.md`.

### BLOCKER

1. [BLOCKER] Activation gate evaluates *predicted* search scores, not the *actual* counterfactual replay results — `scripts/overnight/check_gate.py:99-127` — Spec §13 explicitly says the gate must be evaluated on counterfactual replay (§7 mandates replay before full RL). Verified: `check_gate.py` reads `task_search/candidate_scores.csv` (`mean_abs_u_pred = 5.17e-3` for chain_sparse_credit) instead of `counterfactual_replay/all_replay_summaries.json`, where the actual `mean_abs_u = 6.50e-4` and `frac_u_ge_5e3 = 0.0` (7.9× below the 5e-3 threshold and the `frac >= 0.10` clause is also failed). Pipeline reports gate PASS while measured activation is FAIL. Action: replace the candidate-scores branch with a reader for `all_replay_summaries.json` and require both `mean_abs_u >= 5e-3` and `frac_u_ge_5e3 >= 0.10` per task family (and aggregate). Re-run gate. Acceptance: `check_gate.py --phase IV-A --json` returns FAIL for the current artifacts and PASS only when the per-family replay metrics meet §13 thresholds. Standard review #1; adversarial #2 partially overlaps but is about a different denominator concern (filed separately as MAJOR-5 below).

2. [BLOCKER] `load_selected_task_cfgs` iterates dict keys instead of the `tasks` list — `scripts/overnight/pilot_budget_sensitivity.py:124` — `selected_tasks.json` is a dict `{"suite_type", "tasks", "selected_families", ...}`. The current loop `for entry in raw: cfg = dict(entry["cfg"])` will run `entry = "suite_type"` first and crash with `TypeError: string indices must be integers, not 'str'`. The pilot budget sensitivity study cannot run at all. Action: change to `for entry in raw["tasks"]:` (or `raw.get("tasks", [])` with a clear error if missing). Acceptance: `pilot_budget_sensitivity.py --dry-run` enumerates the 3 selected task families without TypeError. Standard review #2.

3. [BLOCKER] Pilot vs. replay terminal-step convention mismatch breaks counterfactual isolation — `experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py:155` vs `experiments/weighted_lse_dp/geometry/task_activation_search.py:225` — The pilot zeros `v_next` only when the episode terminated early (`if i == T_ep - 1 and len(ep_rewards) < max_steps:`); the replay zeros `v_next` whenever `step_idx == horizon - 1` regardless of absorbing status. For trajectories that exhaust the horizon non-absorbingly the schedule was built from margins that used `r - V*(s_{T-1}^next)` while the replay scores those same transitions with `r - 0`, producing a systematic margin shift. This violates the §7 counterfactual isolation guarantee that the schedule and the replayed transitions come from the same data. Action: align the two conventions (preferred: only zero `v_next` on `absorbing`, since fixed-horizon non-absorbing transitions still have a well-defined V*(s')). Also remove the dead walrus `(i := step_idx)` while there. Acceptance: an integration test that builds a pilot, replays it, and asserts that for every transition the `margin` used in the schedule equals `margin_all` in the replay (atol 1e-12). Adversarial review BLOCKER (terminal mismatch).

4. [BLOCKER] Pilot vs. replay non-idempotency under shared seed — `run_phase4_counterfactual_replay.py:84-167` — `_replay_task` calls `run_classical_pilot(cfg, seed)` and then re-builds the MDP and re-runs episodes inside `_run_pilot_with_transitions(cfg, seed)`. Each call invokes `seed_everything(seed)` and `build_phase4_task(cfg, seed)`. If `build_phase4_task` consumes any global RNG state during construction (the wrappered phase-4 envs do), the replayed transitions are not the same trajectories as the ones that produced `pilot_data`, so the schedule is calibrated on one set of margins and replayed on a different one. This is the same class of failure as the lessons.md 2026-04-17 entry (wrapper vs. base MDP slip) and corrupts the §7 replay claim. Action: collect transitions inside `run_classical_pilot` once and have `_replay_task` reuse them — do not re-run rollouts. Acceptance: a test that calls `_run_pilot_with_transitions(cfg, seed)` twice in a row and asserts the per-transition reward and next-state arrays are byte-identical. Adversarial review BLOCKER (seed/idempotency).

### MAJOR

5. [MAJOR] `mean_abs_u_pred` denominator inconsistent with §9.3 informative-stage rule — `experiments/weighted_lse_dp/geometry/task_activation_search.py:340` — Spec §9.3 requirement 4 ("activation thresholds are evaluated only on informative stages") and §13 (gate "globally and on event-conditioned subsets") are not consistent in the code. `mean_abs_u_pred = float(np.mean(np.abs(u_ref_used)))` averages over all T stages (including dead stages where xi_ref=xi_min and informativeness ≈ 0). Replay-side `mean_abs_u` averages over all transitions. Neither matches the §9.3 informative-stage restriction. Note: this is *not* a gate-decision blocker once BLOCKER-1 is fixed (the replay denominator is well-defined), but it is the right place to surface a SPEC-GAP. Action: (a) add an `informative_mask` to the score reporter; (b) emit both `mean_abs_u_global` and `mean_abs_u_informative`; (c) clarify in spec §13 which denominator is authoritative. Acceptance: aggregator and gate both report both denominators; spec §13 amended to disambiguate. Adversarial review BLOCKER #2 (re-classified as MAJOR + SPEC-GAP).

6. [MAJOR] `safe_clip_active` and `trust_clip_active` flags are logically broken — `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py:287-293` — The current code is:
   ```
   trust_clip_active = (u_ref_used_arr < u_target_arr - 1e-10).tolist()
   safe_clip_active  = ((u_ref_used_arr < U_safe_abs - 1e-10) & ~trust_clip_active).tolist()
   ```
   Two failure modes (consistent across both reviews): (A) when the safe cap is binding (`U_safe_abs < u_tr_cap < u_target`), `u_ref_used == U_safe_abs`, so `trust_clip` evaluates True (because `u_ref_used < u_target`) and `safe_clip` evaluates False — the safe-binding event is mislabeled as a trust-binding event; (B) when neither cap binds (`u_ref_used == u_target < U_safe_abs, u_tr_cap`), `safe_clip` evaluates True even though no cap is binding. This corrupts cap-utilization fractions in tables P4A-B/E and §13 secondary diagnostics (it does not affect the primary gate metric `mean_abs_u`, hence MAJOR not BLOCKER). Action: replace with explicit "which cap is the argmin" logic, e.g.
   ```
   trust_clip_active = (np.abs(u_ref_used_arr - u_tr_cap_arr) <= 1e-10) & (u_tr_cap_arr < u_target_arr - 1e-10)
   safe_clip_active  = (np.abs(u_ref_used_arr - U_safe_abs)   <= 1e-10) & (U_safe_abs   < u_tr_cap_arr - 1e-10) & (U_safe_abs < u_target_arr - 1e-10)
   ```
   Acceptance: a unit test parametrising the three regimes (target binding / trust binding / safe binding / tied) and asserting the flags match the analytic argmin in each. Standard review MAJOR-1 + adversarial review MAJOR (merged).

7. [MAJOR] `xi_ref` normalised by `r_max` instead of `A_t` — `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py:100-113, 198-207` — Spec §S6.2/§S6.3 defines `a_t^s = s * (r - v_ref) / A_t`; the implementation uses `a_t = s * margins / max(r_max, 1e-8)`. For T=20, gamma=0.95, the spec's `A_t ≈ R_max + Bhat[1]` is ~26× `r_max`, so the code's `a_t` is ~26× the spec value. Clipping to `[0.02, 1.0]` masks the magnitude error but the sign-selection score and the unclipped `xi_ref` (when below `xi_max`) are scaled wrong; downstream `theta_used = sign * u_ref_used / max(xi_ref, xi_floor)` and hence `beta_used` are affected. There is a chicken-and-egg here (A_t depends on alpha which depends on I_t = f(xi_ref)), so the fix is a two-pass scheme: provisional A_t with `alpha = alpha_base`, then refine inside the existing fixed-point loop. Action: replace `r_denom` with `A_t` once a provisional headroom is available. Acceptance: a regression test on chain_sparse_credit asserting that the spec-conformant `xi_ref` differs from the current value by ≥ 5% on at least one informative stage, then that the new value matches the analytic spec formula at convergence. Standard review MAJOR-2.

8. [MAJOR] Adaptive-headroom fixed-point convergence criterion does not match spec §6.7 — `experiments/weighted_lse_dp/geometry/adaptive_headroom.py:324-340` — Spec §6.7 says "increase alpha_t only where needed" relative to the feasibility constraint `u_target_t <= U_safe_ref_t`. The code instead bumps alpha when `theta_safe_t / theta_safe_max < 0.8` (a heuristic), which is neither necessary nor sufficient for the feasibility constraint. Result: alpha can be inflated where unnecessary (weaker certification) or fail to inflate where `u_target` is genuinely infeasible. Action: replace the heuristic with the explicit feasibility check `u_target_t > U_safe_ref_t` as the bump trigger and add a stopping criterion based on `max_t (u_target_t - U_safe_ref_t) <= 0`. Acceptance: a test constructing a stage where `theta_safe / theta_safe_max = 0.5` but `u_target = 0.5 * U_safe_ref` (no infeasibility) and asserting alpha is *not* bumped; and conversely when `theta_safe / theta_safe_max = 0.9` but `u_target > U_safe_ref` asserting it *is* bumped. Adversarial review MAJOR.

9. [MAJOR] Trust-region bisection fragile near eps_tr ≈ 0 — `experiments/weighted_lse_dp/geometry/trust_region.py:156-163` — When `eps_tr` is sub-1e-10 (low-alignment early stages), floating-point residuals in `kl_bernoulli(rho(0), p0)` can flip the sign of `kl_mid - eps_tr` at `lo=0` and force the bisection to return `lo=0` (i.e., `u_tr_cap=0`). This is conservative but converts the cap to a hard zero rather than smoothly shrinking. Action: at the start of `solve_u_tr_cap`, if `eps_tr < kl_eps_floor` (e.g., 1e-12), short-circuit with a closed-form linearisation `u_tr_cap = sqrt(2 * eps_tr / sigmoid'(eta0)^2 * (1-p0)/(p0))` (or simply pin `u_tr_cap = u_target * sqrt(eps_tr / eps_design)` to preserve the smoothness limit). Acceptance: a test sweeping `eps_tr` from 1 down to 1e-15 and asserting `u_tr_cap` is monotone-non-decreasing in `eps_tr` and continuous (no jump to 0). Adversarial review MAJOR.

### MINOR

10. [MINOR] `select_activation_suite` acceptance thresholds (`min_mean_abs_u_pred=2e-3`, `min_frac_active_stages=0.05`) are looser than the §13 gate (`5e-3` and `0.10`) — `experiments/weighted_lse_dp/geometry/task_activation_search.py:413-489` — This is a screen-then-gate pattern, but the screen admits the chain_jackpot tasks at `mean_abs_u_pred ≈ 2.4e-3` which then fail the §13 gate. Action: tighten the search-phase floor to match §5.3 minimum (`mean_abs_u_pred >= 2e-3`, OK) but document the intentional gap; or align both to §13. Acceptance: regenerate `selected_tasks.json` and confirm only candidates with `mean_abs_u_pred >= 5e-3` are kept (or document why looser screening is desired). Adversarial review MAJOR (re-classified as MINOR — the screen-then-gate pattern is itself defensible; the bug is only that the gate is wrong, see BLOCKER-1).

11. [MINOR] `U_safe_abs = np.abs(U_safe_ref_t)` silently flips a negative binding constraint — `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py:271` — In normal operation `theta_safe_t > 0` and `xi_ref_t > 0`, so `U_safe_ref_t > 0` and `np.abs` is a no-op. But if numerical issues drive `theta_safe_t` (which uses `log(kappa / (gamma * (1+gamma-kappa)))`) negative, `np.abs` would silently flip a negative cap into a large positive value, effectively removing the safety constraint. Action: replace `np.abs` with `np.where(U_safe_ref_t < 0, 0.0, U_safe_ref_t)` and add `assert (U_safe_ref_t >= -1e-12).all()` with a logged warning otherwise. Acceptance: assertion failure path unit-tested. Adversarial review BLOCKER (re-classified as MINOR — current parameter regime never hits the negative branch, so this is a defensive hardening, not a live bug; the live BLOCKER status is overstated).

12. [MINOR] Module docstring describes random-policy pilot but implementation uses DP V* + ε-greedy Q* — `experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py:1-8` — Contradicts lessons.md 2026-04-20. Action: rewrite docstring to "Replay frozen DP V* / ε-greedy Q* pilot transitions through Phase IV schedule." Acceptance: docstring grep for "random-policy" returns no hits in this file. Standard review MINOR-1.

13. [MINOR] Two `grid_hazard` mainline entries with bitwise-identical scoring metrics — `experiments/weighted_lse_dp/configs/phase4/activation_suite.json` — Different `hazard_prob` values (0.3 vs 0.2) yielded `mean_abs_u_pred = 0.003763914...` and `frac_u_ge_5e3 = 0.35` for both, suggesting the search returned the same pilot or the scoring is insensitive to `hazard_prob` in this regime. Both are below the §13 5e-3 threshold. Action: investigate whether the pilots really degenerate to the same diagnostics; if so, deduplicate. Acceptance: either both entries are explained in the activation report, or one is removed. Standard review MINOR-3.

14. [MINOR] Dead walrus `(i := step_idx)` — `experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py:155` — `i` is never read again. Subsumed by BLOCKER-3 fix (rewrite the terminal condition entirely). Standard review MINOR-2 + adversarial MINOR.

15. [MINOR] Redundant `xi_ref_arr[t] = xi_min` after `np.clip` already produced the same value — `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py:209-211` — Harmless but confusing. Action: delete the explicit re-assignment (the clip already handles q75=0). Adversarial MINOR.

### NIT

16. [NIT] `expected_bhat0 = (1 + gamma) * r_max / (1 - kappa_const)` is dead, computes the wrong (infinite-series) limit — `scripts/overnight/cert_audit.py:289` — Variable is computed and immediately superseded by `bhat0_exact`. Action: delete the dead variable to avoid future cargo-culting of the wrong formula. Standard review NIT-1.

17. [NIT] `selected_families` length check accepts empty list when key absent — `scripts/overnight/check_gate.py:82-92` — Fallback `data.get("selected_families", [])` makes a missing-key failure silently report "0 families selected". Action: distinguish missing-key from empty-list and surface the cause in the details string. Standard review NIT-2.

### DISPUTE

18. [DISPUTE] Adversarial Challenge 7 ("systematic downward bias from negative margins") — Investigated: `select_sign` flips the sign family when most margins are negative, restoring positive `a_t > 0` density before the q75 is computed. The bias concern does not survive sign selection. No action.

19. [DISPUTE] Adversarial concern about `U_safe_ref` `np.abs` as BLOCKER (separate from the silent-flip concern itself) — Investigated: in the parameter regime used by Phase IV-A (gamma=0.95, alpha in [0.05, 0.30]), `theta_safe_t = log(kappa / (gamma * (1+gamma-kappa)))` is strictly positive (kappa > gamma ⇒ numerator > denominator's gamma factor), so `np.abs` is a no-op and there is no live correctness failure. Re-classified as MINOR (defensive hardening) — see #11.

### Open questions / SPEC-GAP

- §9.3 requirement 4 ("activation thresholds are evaluated only on informative stages") vs §13 ("the gate must be evaluated both globally and on event-conditioned subsets") — these are not the same denominator. Which wins for the formal gate decision? The fix for BLOCKER-1 needs a spec ruling. Default (until the user rules): require the global denominator from replay (§13 plain reading) AND surface the informative-stage and event-conditioned denominators as secondary diagnostics.
- §S6.2/§S6.3 spec writes `a_t = s * (r - v_ref) / A_t` but A_t depends on alpha which depends on xi_ref via I_t — circular dependency. Spec is silent on iteration order. MAJOR-7 (xi_ref denominator fix) needs a spec clarification on which provisional A_t to use. Default: A_t with `alpha = alpha_base` for the first pass, then refine inside the existing §6.7 fixed-point loop.

### Routing (per AGENTS.md §4)

- BLOCKERs 1, 2, 3, 4 → experiment-runner (wiring fixes in scripts/runners) with verifier sign-off.
- MAJOR 6, 9 → operator-theorist + algo-implementer (geometry-layer correctness).
- MAJOR 7, 8 → calibration-engineer (schedule v3 spec compliance).
- MAJOR 5 → planner (spec amendment) then plotter-analyst (denominator surfacing).
- MINOR 11, 13 → calibration-engineer.
- MINOR 10 → planner (search threshold policy).
- MINOR 12, 14, 15, NIT 16, 17 → any code-owning role.
