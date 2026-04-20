# Overnight Run — Phase IV
Started: 2026-04-19T04:14:10Z
Arguments: (none — default: IV-A → IV-B → IV-C, no skip-review, no dry-run)
Branch: phase-iv/overnight (from main @ 04ae83c)
---

## Auto-resolved questions

1. Phase II/III rerun (b0ao4zigu) running concurrently with Phase IV-A initialization.
   Resolution: Proceed. Phase IV-A writes only to results/weighted_lse_dp/phase4/ and is additive. Audit will read Phase III results at audit time.

---

## Phase IV-A

### [2026-04-19T04:14:10Z] INIT — Reading specs and initializing checkpoint

- Read docs/specs/phase_IV_A_activation_audit_and_counterfactual.md ✓
- Checkpoint initialized at tasks/overnight_checkpoint.json ✓
- Branch phase-iv/overnight created from main @ 04ae83c ✓

### [2026-04-19T04:14:10Z] PLAN — Spawning planner subagent for Phase IV-A


### [2026-04-19T04:14:10Z] AUTO-APPROVE — Phase IV-A plan: 28 tasks (56-83), 9 tiers

Plan logged: tasks 56-83 appended to tasks/todo.md.
Open questions from planner auto-resolved:
  1. Calibration pilots → Use classical DP (VI) only (conservative, exact)
  2. Taxi family → Run search, let acceptance criteria decide
  3. State-dependent sign ablation → Defer to Phase IV-C (spec §0.7)
  4. Pilot seeds → 3 seeds (matches Phase I/II convention)
  5. Phase III MINOR items R3-1,R3-3,R3-4 → Deferred; R3-2 absorbed by task 77

### [2026-04-19T04:14:10Z] DISPATCH — First parallel wave

- a757bba5 (operator-theorist, worktree): Tasks 60-63 — geometry modules
- a9240aec (calibration-engineer, worktree): Tasks 56-58 — Phase III audit + paper_suite_replay.json
- a331dc33 (env-builder, worktree): Task 70 — Phase IV-A task factories (6 families)

### [2026-04-19T04:45:00Z] COMPLETE — Tasks 60-63 (geometry modules)

- a757bba5 returned PASS
- natural_shift.py, trust_region.py, adaptive_headroom.py, activation_metrics.py implemented
- smoke test: PASS from main repo
- Deviations noted: trust_region.py operates on u (not theta per stub), adaptive_headroom.py fixed-point uses heuristic increase schedule (spec §6.7 doesn't prescribe exact schedule), activation_metrics.py extended signature
- Merged worktree to main repo (copy)

### [2026-04-19T05:30:00Z] COMPLETE — Tasks 56-58 (Phase III audit)

- a9240aec returned PASS
- phase3_audit.py implemented (code_audit, result_audit, replay_smoke, run_audit)
- paper_suite_replay.json populated with all 8 Phase III tasks
- Audit results: 8/8 schedules found, 154 run artifacts, DP replay PASS
- Key finding: beta_used=[0, 0.000201] for chain_jackpot — near-classical as expected
- Audit note: compute_safe_target_ev_batch not in git-tracked version (only in working copy — will resolve on commit)
- Merged artifacts to main repo

### [2026-04-19T05:30:00Z] COMPLETE — Task 70 (Phase IV-A task factories)

- a331dc33 returned PASS
- phase4_operator_suite.py: 6 families + search grid (168 candidates, 150 mainline)
- Fixed API mismatches: stress_families.py signatures differ from worktree assumptions
  - make_chain_sparse_long: no goal_reward/severity/step_cost params
  - make_chain_jackpot: no goal_reward/severity/step_cost params
  - make_chain_catastrophe: uses risky_state not risky_prob
  - make_regime_shift_chain → make_chain_regime_shift
  - wrapper.mdp → build_hazard_mdp (new helper added to hazard_wrappers.py)
- Added select_hazard_states + build_hazard_mdp to hazard_wrappers.py
- Smoke test PASS: all 6 families, 168-candidate search grid, reward_bound <= 3.0

### [2026-04-19T05:30:00Z] COMPLETE — Tasks 64-65 (geometry tests)

- a807d527 returned PASS (56/56 tests)
- test_phase4_natural_shift_geometry.py (42 tests) + test_phase4_activation_metrics.py (14 tests)
- Tests verified in main repo: 56 passed in 0.08s

### [2026-04-20T00:43:00Z] RETRY — Activation gate fix attempt (one retry per §2.3)

Diagnosis of original gate failure:
1. `task_activation_search.py` used random-policy pilot (non-spec-compliant — spec §S5.1 requires QL/DP)
2. `run_phase4_counterfactual_replay.py` also used random-policy pilot + Monte Carlo v_next
3. `check_gate.py` read wrong CSV column names (`mean_abs_u`, `frac_active` instead of `mean_abs_u_pred`, `frac_u_ge_5e3`)

Fixes applied:
1. `task_activation_search.py`: replaced random pilot with DP-backward-VI V* pilot (epsilon-greedy Q*)
2. `run_phase4_counterfactual_replay.py`: replaced Monte Carlo v_next with V*(s') from backward VI
3. `check_gate.py`: fixed column names to `mean_abs_u_pred` / `frac_u_ge_5e3` (with fallback to old names)
4. Re-ran activation search with n=200 pilot episodes

Results after fixes:
- 7 tasks selected (was 1)
- frac_u_ge_5e3 gate: PASS (max=0.35 across families)
- mean_abs_u gate: FAIL (best=0.00356, threshold=0.005)

### [2026-04-20T00:49:00Z] GATE — Phase IV-A gate re-check: FAIL (10/11)

```
Gate IV-A: FAIL (10/11 conditions met)
  [PASS] Phase III compat report
  [PASS] Phase III code audit
  [PASS] Phase III result audit
  [PASS] Selected tasks file
  [PASS] At least one task family selected
  [PASS] Activation search report
  [PASS] Counterfactual replay results non-empty
  [FAIL] mean_abs_u >= 5e-3 (best=0.00356, threshold=0.005)
  [PASS] frac_active >= 10% (best=0.35)
  [PASS] Frozen activation suite config
  [PASS] Matched controls config
```

Root cause of remaining failure:
A_t = r_max + Bhat[t+1] grows as ~24^T for T=20, γ=0.95, α_max=0.20.
beta = u_ref_used / (A_t * xi_ref) ≈ 6e-3 / 10^26 ≈ 1e-29.
The safe operator cannot produce nontrivial nonlinearity under the current certification constraints
for T≥20, γ=0.95. This is a mathematical constraint, not a code bug.

Per overnight invariant §0.4 (Gate-or-stop): Phase IV-A gate failed. Stopping pipeline.
Phase IV-B and IV-C NOT started.

---

## Final Report

Completed: 2026-04-20T00:50:00Z
Duration: ~20.6 hours (started 2026-04-19T04:14Z)

Phase IV-A: gate_failed (tasks 56-83 implemented; activation gate FAIL: 10/11)
Phase IV-B: not_started (blocked by IV-A gate)
Phase IV-C: not_started (blocked by IV-B)

### Gate results

| Phase | Gate | Detail |
|-------|------|--------|
| IV-A activation | FAIL (10/11) | mean_abs_u=0.00356 < 0.005; frac_active=0.35 ≥ 0.10 |
| IV-B translation | N/A | not started |
| IV-C completion | N/A | not started |

### Fixes applied during this run (code quality improvements regardless of gate)

1. **task_activation_search.py**: random-policy pilot → DP backward VI V* pilot (spec §S5.1 compliance)
2. **run_phase4_counterfactual_replay.py**: random-policy pilot + MC v_next → DP V* pilot
3. **check_gate.py**: fixed column names (`mean_abs_u_pred`, `frac_u_ge_5e3`)
4. Activation search with 200 episodes: 7 tasks selected (chain_sparse_credit, grid_hazard×2, regime_shift×2, taxi_bonus×2)

### Root cause of gate failure

A_t = r_max + Bhat[t+1] grows as ~24^T under the certification recursion for T=20, γ=0.95.
beta = u_ref_used / (A_t * xi_ref) ≈ 6e-3 / 10^26 ≈ 1e-29.
The operator is effectively classical at these task parameters.

### Recommended path forward (for user to decide)

1. **Use shorter horizons**: T=5 reduces A_t from 10^26 to ~12, making beta ≈ 0.001. Close to gate with T=5.
2. **Use lower gamma**: γ=0.5 with T=5 gives A_t ≈ 4.25, beta ≈ 0.005. Likely passes gate.
3. **Adjust gate threshold**: If 3.5e-3 is acceptable as "nontrivially nonlinear," lower the gate to 3e-3 (user decision — spec change).
4. **Accept negative-control finding**: Phase IV-A's scientific contribution is documenting that the certified operator is effectively classical for realistic task parameters. This may itself be publishable as a theoretical finding.

### Artifacts produced (all in results/weighted_lse_dp/phase4/)

- audit/: Phase III compatibility report, code audit, result audit, DP replay smoke
- task_search/: candidate_grid.json (168), candidate_scores.csv, selected_tasks.json (7), activation_search_report.md
- counterfactual_replay/: replay diagnostics for 1 task (pre-gate attempt)

### Lessons added

See tasks/lessons.md — random-policy pilot violation of §S5.1 spec, certification A_t blowup.


---

## Certification Geometry Audit — 2026-04-20 (resumed session)

### Audit script: `scripts/overnight/cert_audit.py`

Ran `python3 scripts/overnight/cert_audit.py` → exit code 0.

Output: `results/weighted_lse_dp/phase4/audit/certification_geometry_audit.json`

### Known-value check

| Quantity | Computed | Expected | Error |
|---|---|---|---|
| Bhat[0] (T=20, γ=0.95, κ=0.96, R=1) | 27.2024 | 27.2024 | 7.1e-15 |

Bhat bug confirmed fixed. Old formula gave ~1e26 (fixed-point form, diverges as κ→1). New recursion: Bhat_t = (1+γ)R_max + κ_t·Bhat_{t+1}, correct per Phase III spec §5.

### All 5 invariants PASS on all 7 audit tasks

| Task | κ_0 | Bhat[0] | A_t0 | θ_safe | u_tr_cap@200ep | u_tr_cap@2000ep |
|---|---|---|---|---|---|---|
| chain_sparse_credit (T=20, γ=0.95) | 0.9600 | 27.20 | 27.30 | 0.0205 | 0.00183 | 0.00485 |
| grid_hazard (T=20, γ=0.97) | 0.9760 | 31.59 | 31.35 | 0.0122 | 0.00183 | 0.00485 |
| short_horizon (T=5, γ=0.95) | 0.9600 | 9.00 | 8.34 | 0.0205 | 0.00343 | 0.00687 |
| long_horizon (T=50, γ=0.95) | 0.9600 | 42.42 | 43.15 | 0.0205 | 0.00114 | 0.00343 |

### Corrected gate failure root cause

**NOT** the Bhat blowup (which is now fixed). The binding constraint is the trust-region cap.

Chain: n_t = n_ep / T samples/stage → c_t = (n_t/(n_t+τ_n))·√p_align → eps_tr = c_t·eps_design → u_tr_cap

With 200 episodes, T=20:
- n_t ≈ 10 samples/stage
- c_t ≈ (10/210)·√0.5 ≈ 0.034  
- u_tr_cap ≈ 0.00183 << 0.005 gate threshold

With 2000 episodes, T=20:
- n_t ≈ 100 samples/stage
- c_t ≈ (100/300)·√0.5 ≈ 0.236
- u_tr_cap ≈ 0.00485 (still below for γ=0.97 tasks, marginally below for γ=0.95)

With 2000 episodes, T=5:
- u_tr_cap ≈ 0.00687 > 0.005 → gate passes

### Scientific conclusion

Phase IV-A gate failure is **not a fundamental operator failure**. The operator CAN produce certified activation above the 5e-3 threshold, but requires more pilot data than the 200 episodes used:
- T=5 tasks with 2000 pilot episodes: u_tr_cap ≈ 0.0069 → gate passes
- T=20 tasks require ~500-1000 pilot episodes depending on γ

The previous root-cause message ("A_t grows as ~24^T") was a pre-fix artefact. Post-fix A_t[0] ≈ 27-32 (not 10^26). The operator is NOT "effectively classical" — beta_used ≈ 1.55e-4 with correct Bhat, which is physically meaningful. The gate fails only because τ_n=200 demands many samples before releasing the trust region.

### Updated checkpoint

Gate status: FAIL (10/11) — same condition, new mechanistic explanation.

---

## Phase IV-A Pilot-Budget Sensitivity Study — 2026-04-20

### Study output: `results/weighted_lse_dp/phase4/pilot_budget_sensitivity/`

Grid: 204 conditions × {chain_sparse_credit, grid_hazard, regime_shift, taxi_bonus} × T∈{5,10,20,30,40,50,60,67} × n_ep∈{200,500,1000,2000} × tau_n∈{50,100,200}.

### Key result: binding cap is ALWAYS trust_clip (204/204 conditions)

Certification geometry (U_safe_ref) is never binding. The constraint is purely the trust-region cap driven by pilot sample count.

### Mainline tau_n=200, chain_sparse_credit, T=20

| n_ep | mean_abs_u | frac≥5e-3 | c_t_med | u_tr_cap_med | Gate |
|---|---|---|---|---|---|
| 200 | 0.00356 | 0.350 | 0.180 | 0.00231 | FAIL |
| 500 | 0.00444 | 0.400 | 0.223 | 0.00331 | FAIL |
| **1000** | **0.00517** | **0.450** | **0.276** | **0.00392** | **PASS** |
| 2000 | 0.00547 | 0.450 | 0.299 | 0.00395 | PASS |

### tau_n ablation, chain_sparse_credit, T=20, n_ep=200

| tau_n | mean_abs_u | Gate |
|---|---|---|
| 200 | 0.00356 | FAIL |
| 100 | 0.00411 | FAIL |
| 50 | 0.00450 | FAIL |

(tau_n=50 passes only at n_ep≥500: 0.00501)

### Short-horizon variants, tau_n=200

| T | n_ep=200 | n_ep=500 | Gate |
|---|---|---|---|
| 5 | 0.01259 | 0.01503 | **PASS at all budgets** |
| 10 | 0.01067 | 0.01248 | **PASS at all budgets** |

grid_hazard, regime_shift, taxi_bonus: FAIL at all settings (zero-signal pilot for short T; sparse alignment for long T).

### Regression test added

`tests/algorithms/test_phase4_natural_shift_geometry.py::test_bhat_backward_regression_known_value` — PASSES. Verifies Bhat[0]=27.2024 (not ~1e26) for T=20, γ=0.95, κ=0.96.
