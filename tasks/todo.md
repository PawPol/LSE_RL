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
