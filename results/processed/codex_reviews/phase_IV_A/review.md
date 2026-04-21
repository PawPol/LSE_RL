# Phase IV-A Code Review

**Branch**: `phase-iv-a/closing` vs `main`  
**Date**: 2026-04-19  
**Reviewer**: Codex subagent (standard review)

---

## BLOCKER issues (must fix before merge)

### [BLOCKER-1] `scripts/overnight/check_gate.py`:100–127 — Gate evaluates predicted metrics, not actual replay results

The Phase IV-A activation gate (spec §13) must be evaluated on counterfactual replay results. `check_gate.py` reads `candidate_scores.csv` (predicted `mean_abs_u_pred` from the activation search) rather than the actual replay summaries in `all_replay_summaries.json`. The actual counterfactual replay shows `mean_abs_u ≈ 6.5e-4` (vs threshold `5e-3`), a 7× shortfall. The predicted search score (`5.17e-3`) passes the gate threshold. This means the gate reports PASS in the automated pipeline while the actual measured activation would FAIL.

Confirmed against artifact:
- `results/weighted_lse_dp/phase4/counterfactual_replay/all_replay_summaries.json`: `mean_abs_u = 6.5e-4`, `frac_u_ge_5e3 = 0.0`
- `results/weighted_lse_dp/phase4/task_search/candidate_scores.csv`: `mean_abs_u_pred = 5.17e-3` for `chain_sparse_credit`

**Fix**: `check_gate.py` must read `all_replay_summaries.json` (key `tasks[*].mean_abs_u`, `frac_u_ge_5e3`) instead of, or in addition to, `candidate_scores.csv`. The spec §13 language is explicit: "the gate must be evaluated both globally and on event-conditioned subsets" based on replay.

---

### [BLOCKER-2] `scripts/overnight/pilot_budget_sensitivity.py`:124 — `load_selected_task_cfgs` iterates over dict keys, not task list

`selected_tasks.json` is a dict with schema `{"suite_type": ..., "tasks": [...], ...}`. The function does:

```python
for entry in raw:   # iterates dict KEYS: "suite_type", "tasks", "seed", ...
    cfg = dict(entry["cfg"])  # TypeError: string indices must be integers
```

This crashes immediately on the first iteration with `TypeError: string indices must be integers, not 'str'`. The pilot budget sensitivity study cannot run at all.

**Fix**: Change to `for entry in raw.get("tasks", []):` (or `raw["tasks"]` with an appropriate error message if missing).

---

## MAJOR issues (should fix, don't block)

### [MAJOR-1] `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py`:287–293 — `safe_clip_active` logic has two bugs

The current code:

```python
trust_clip_active = (u_ref_used_arr < u_target_arr - 1e-10).tolist()
safe_clip_active = (
    (u_ref_used_arr < U_safe_abs - 1e-10)
    & ~np.array(trust_clip_active)
).tolist()
```

**Bug A** — When the safe cap is the binding constraint (`U_safe_abs < u_tr_cap < u_target`):
- `u_ref_used = U_safe_abs`
- `trust_clip = (U_safe_abs < u_target - 1e-10)` = **True** (correct by coincidence)
- `safe_clip = (U_safe_abs < U_safe_abs - 1e-10) & ~True` = **False** (wrong — safe was binding)
- Result: `trust_clip=True, safe_clip=False`. The safe cap binding event is invisible.

**Bug B** — When neither cap binds (`u_ref_used = u_target < u_tr_cap, U_safe_abs`):
- `trust_clip = False`
- `safe_clip = (u_target < U_safe_abs - 1e-10) & True` = **True** (wrong — neither cap was binding)

**Correct logic**:
```python
trust_clip_active = (np.abs(u_ref_used_arr - u_tr_cap_arr) <= 1e-10) & (u_tr_cap_arr < u_target_arr - 1e-10)
safe_clip_active = (np.abs(u_ref_used_arr - U_safe_abs) <= 1e-10) & (U_safe_abs < u_tr_cap_arr - 1e-10) & (U_safe_abs < u_target_arr - 1e-10)
```

Impact: corrupts the cap utilization diagnostics in tables P4A-B/E and all logging fields that consume `trust_clip_active_t` / `safe_clip_active_t`.

---

### [MAJOR-2] `experiments/weighted_lse_dp/geometry/phase4_calibration_v3.py`:100–113, 198–207 — `xi_ref` normalized by `r_max` instead of `A_t` (spec §S6.2/6.3)

The spec (§S6.2 line 460, §S6.3 line 473) defines:

```
a_t^s = s * (r - v_ref) / A_t
xi_ref_t = clip(Q_0.75(a_t | a_t > 0), xi_min, xi_max)
```

Both `select_sign` and `build_schedule_v3` use:

```python
r_denom = max(r_max, 1e-8)
a_t = s * margins / r_denom   # uses r_max, not A_t
```

For T=20, gamma=0.95: `A_t[0] ≈ r_max + Bhat[1] ≈ 1 + 25 = 26`. The code's `a_t` is ~26× the spec's `a_t`. Since `xi_ref` is clipped to `[0.02, 1.0]`, the clipping masks most of the scale error — but the sign decision (which sign yields higher average score) can differ, and the scale of `xi_ref` relative to the theoretical `A_t`-normalized value is wrong. The downstream `theta_used = sign * u_ref_used / max(xi_ref, xi_floor)` depends on `xi_ref`, so `beta_used` is impacted.

This is a spec violation that affects the scientific validity of the schedule calibration.

**Note**: A chicken-and-egg problem makes the spec hard to implement exactly (A_t depends on alpha, which depends on I_t = f(xi_ref)), but the spec's intent is clear. The fix requires a two-pass scheme: compute a provisional A_t with default alpha, compute xi_ref/sign from it, then run the fixed-point iteration.

---

## MINOR issues

### [MINOR-1] `experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py`:1–8 — Stale module docstring describes random-policy pilot

The module docstring says: *"Freezes transitions from classical random-policy pilots..."*. The actual implementation uses DP V* and epsilon-greedy Q* (the fix from lessons.md 2026-04-20). This misleads readers about the pilot methodology and contradicts the lessons.md entry. Update the docstring to match the implementation.

---

### [MINOR-2] `experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py`:155 — Dead walrus operator

```python
if absorbing or (i := step_idx) == horizon - 1:
```

The walrus operator assigns `i = step_idx` but `i` is never read elsewhere. This is semantically equivalent to `if absorbing or step_idx == horizon - 1:` but the walrus is confusing and unused. Remove it.

---

### [MINOR-3] `experiments/weighted_lse_dp/configs/phase4/activation_suite.json` — Two `grid_hazard` mainline entries with identical metrics

The mainline suite has two `grid_hazard` entries with `hazard_prob=0.3` and `hazard_prob=0.2` respectively, but both have `mean_abs_u_pred=0.003763914...` and `frac_u_ge_5e3=0.35` (bitwise identical). This indicates the scoring pipeline used the same pilot or returned the same result for both configs, which is suspicious. Additionally, both entries are below the `5e-3` gate threshold — they pass the looser selection criterion (`min_mean_abs_u_pred=2e-3`) but neither passes the gate.

At minimum, this should be verified and the identical metrics investigated. If the pilots truly produce identical results, one entry should be removed.

---

## NITs

### [NIT-1] `scripts/overnight/cert_audit.py`:289 — `expected_bhat0` uses infinite-series formula, is never read

```python
expected_bhat0 = (1 + gamma) * r_max / (1 - kappa_const)  # geometric series sum (INFINITE)
```

This computes the infinite-series limit and is immediately superseded by `bhat0_exact` (the truncated sum, used for the actual pass check on line 297). The variable `expected_bhat0` is never used after assignment. Remove the dead variable to avoid confusion; the comment's formula is the wrong (infinite) one.

---

### [NIT-2] `scripts/overnight/check_gate.py`:82–92 — `selected_families` length check accepts empty list if key is absent

```python
families = data if isinstance(data, list) else data.get("selected_families", [])
checks.append(_check(len(families) > 0, ...))
```

If `selected_tasks.json` is missing the `selected_families` key (e.g. written by an older writer), `families = []` and the check silently fails without explaining which key was missing. Minor improvement: log the fallback path.

---

## PASS items (explicitly verified correct)

- **[PASS] Bhat backward recursion formula** — `compute_bhat_backward` in `adaptive_headroom.py` delegates entirely to `safe_weighted_common.compute_certified_radii`, which correctly implements `Bhat[t] = (1+gamma)*R_max + kappa_t*Bhat[t+1]`. The old geometric-series bug (`kappa*(r_max + Bhat[t+1])/(1-kappa)`) is gone. Verified by `test_bhat_backward_regression_known_value`: T=20, gamma=0.95, kappa=0.96 → Bhat[0] ≈ 27.2 (not ~1e26).

- **[PASS] Trust region confidence cap formula** — `c_t = (n_t/(n_t+tau_n)) * sqrt(p_align_t)` matches spec §S6.5. Bisection range [0,20] is sufficient: `kl_bernoulli(rho(20), p0) >> 1` for all reasonable gamma, so `eps_tr` is always within range.

- **[PASS] Margin formula (no gamma)** — Both `task_activation_search.py:227` (`margin = r_t - v_next`) and `run_phase4_counterfactual_replay.py:295` (`margin_all[i] = r - v_next`) correctly omit gamma. Consistent with lessons.md 2026-04-16.

- **[PASS] Ex-ante purity** — `score_all_candidates` and `run_classical_pilot` use only DP V* (backward VI on the base MDP) and closed-form schedule predictions. No Phase IV safe return files are imported or read. The comment "This function MUST NOT import or read any Phase IV safe result files" is present and enforced.

- **[PASS] Schema compatibility pipeline** — `run_phase4_counterfactual_replay.py`:439–445 handles three schema variants: `selected_tasks` (legacy), `tasks` (v2), and `mainline.tasks` (current `activation_suite.json`). The actual `activation_suite.json` uses `mainline.tasks`, which is correctly parsed.

- **[PASS] DP pilot uses V* not random policy** — `run_classical_pilot` performs finite-horizon backward VI (`_compute_vstar`) to get V* and Q*, then runs epsilon-greedy (eps=0.1) against Q*. Fixes the lessons.md 2026-04-20 defect.

- **[PASS] Time-augmented state decoding** — Stage extraction uses `state_idx // n_base` where `n_base = mdp_base.p.shape[0]` (from environment metadata), not a hard-coded table. Consistent with Phase III observability fix requirement.

- **[PASS] Gate check column names** — `check_gate.py` reads `mean_abs_u_pred` (with fallback `mean_abs_u`) and `frac_u_ge_5e3` (with fallback `frac_active`). These match the columns written by `write_candidate_scores_csv` in `run_phase4a_mainline_rerun.py`.

- **[PASS] Additive-only invariant** — None of the Phase IV-A scripts or runners write to `results/weighted_lse_dp/phase{1,2,3}/` paths. All Phase IV-A outputs go to `results/weighted_lse_dp/phase4/` or `results/processed/phase4A/`.

- **[PASS] Regression test strength** — `test_bhat_backward_regression_known_value` verifies both the closed-form value (Bhat[0] ≈ 27.2 ± 0.01) and explicitly asserts `Bhat[0] < 1e6` to catch any resurface of the old astronomical-value bug. The test is parameterized over four Phase IV-A task configs.

- **[PASS] Seed consistency** — `seed=42` is used consistently across `run_classical_pilot`, `run_phase4_counterfactual_replay.py`, and `run_phase4a_mainline_rerun.py`. `seed_everything(seed)` is called before any random operations in pilot runners.

- **[PASS] `gamma_matched_controls.json` spec compliance** — Since all mainline tasks have `gamma_base == gamma_eval == 0.95`, spec §6.9 requires only the classical matched-gamma control (not the safe-zero-nonlinearity control, which is only required when `gamma_base != gamma_eval`). The emitted config has the correct VI controls paired with each task.

- **[PASS] `solve_u_tr_cap` bisection correctness** — The bisection is on `KL_Bern(rho(u) || p0)` which is strictly increasing for `u ≥ 0` (since `rho(u)` is increasing and diverges from `p0` as `u` increases from 0). The boundary check `if eps_tr >= kl_hi: return hi` correctly handles saturation. The roundtrip test `test_solve_u_tr_cap_roundtrip` verifies the solver to `atol=1e-7`.

- **[PASS] Phase III schedule `compute_bhat_backward` delegation** — `adaptive_headroom.compute_bhat_backward` delegates to `mushroom_rl.algorithms.value.dp.safe_weighted_common.compute_certified_radii`. Both code paths are tested to be bit-for-bit identical in `test_bhat_backward_matches_operator_recursion`. This guarantees the geometry layer and the operator layer remain consistent.
