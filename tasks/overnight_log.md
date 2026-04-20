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

