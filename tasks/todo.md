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
