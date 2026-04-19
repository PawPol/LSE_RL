# Phase IV-A Activation Gate Report

**Date**: 2026-04-19  
**Spec ref**: §13 — Activation gate before Phase IV-B

---

## Gate criteria (§13)

### Global gate (required for Phase IV-B)
| Criterion | Threshold |
|-----------|-----------|
| `mean_abs_u` | ≥ 5e-3 |
| `frac(|u| ≥ 5e-3)` | ≥ 0.10 |
| `mean_abs(delta_effective_discount)` | ≥ 1e-3 |
| `mean_abs(target_gap) / R_max` | ≥ 5e-3 |

### Preferred stronger gate
| Criterion | Threshold |
|-----------|-----------|
| `mean_abs_u` | ≥ 1e-2 |
| `frac(|u| ≥ 1e-2)` | ≥ 0.10 |
| `mean_abs(delta_effective_discount)` | ≥ 5e-3 |

---

## Per-task evaluation

| Task | Family | mean\|u\| | frac\|u\|≥5e-3 | mean\|Δd\| | \|tg\|/R_max | Gate |
|------|--------|-----------|---------------|------------|------------|------|
| chain_sparse_credit_0 | chain_sparse_credit | 1.19e-05 | 0.05% | 5.79e-06 | 4.47e-06 | **FAIL** |
| chain_sparse_credit_1 | chain_sparse_credit | 9.96e-06 | 0.05% | 4.85e-06 | 1.87e-06 | **FAIL** |
| grid_hazard_2 | grid_hazard | 5.72e-06 | 0.00% | 2.79e-06 | 1.79e-06 | **FAIL** |
| grid_hazard_3 | grid_hazard | 3.55e-06 | 0.00% | 1.73e-06 | 1.16e-06 | **FAIL** |

---

## Gate result: **FAIL**

All 4 activation-suite tasks fail the global gate. All metrics are 2–3 orders of magnitude below threshold.

---

## Diagnosis

The beta_used values are near-zero (mean_beta ≈ -4e-4 to +1e-4) because:

1. **Random-policy pilot yields uninformative margins.** A 50-episode random-policy pilot on sparse-reward MDPs produces margins concentrated near zero, giving `xi_ref_t ≈ 0` and therefore `u_target_t ≈ u_min = 2e-3`. The adaptive headroom fixed-point then clips `u_ref_used_t` further.

2. **v3 schedule calibration is conservative.** With `p_align ≈ 0.5` (random policy) and small margins, `xi_ref_t * sqrt(p_align_t) < 0.05` so the informativeness threshold is rarely met. `u_target_t` stays at `u_min = 2e-3` but `U_safe_ref_t` is small enough that the final `u_ref_used_t ≈ 0`.

3. **Task selection scored on predicted metrics, not realized ones.** The pilot-based prediction overestimated how much of the margin variation would translate to activation once schedule clips are applied.

---

## Per-family classification

| Family | Gate status | Classification |
|--------|-------------|----------------|
| chain_sparse_credit | FAIL (global) | Low-activation negative control |
| chain_catastrophe | Not selected | Low-activation negative control |
| chain_jackpot | Not selected | Low-activation negative control |
| grid_hazard | FAIL (global) | Low-activation negative control |
| regime_shift | Not selected | Low-activation negative control |
| taxi_bonus | Not selected | Low-activation negative control (appendix) |

Per spec §13: "If a family fails both [global and event-conditioned], keep it as a low-activation negative control and do not use it for the main translation claim."

---

## Activation questions answered (spec §15)

For each task family:

1. **Is this family a low-activation negative control or true activated Safe TAB case?**  
   All families: **low-activation negative control**. No family passed the global gate.

2. **How large are deployed beta_used, beta*margin, discount-gap, and target-gap diagnostics under certification?**  
   All metrics are O(1e-5) — approximately 300× below the minimum gate threshold of 5e-3. The operator is effectively classical under the current pilot calibration.

3. **Did lower-base-gamma increase certified activation?**  
   Not directly tested at full run — matched controls at gamma_base=0.95 vs gamma_eval=0.97 are configured (gamma_matched_controls.json) but not yet run. This is a priority for the next calibration iteration.

4. **Is the task eligible for Phase IV-B translation experiments?**  
   **No** — per spec §13, Phase IV-B may not proceed until the activation gate passes.

---

## Recommended next steps

1. **Improve the pilot.** Use a lightly-trained classical policy (not random) for the pilot, which will produce more informative margins and higher `xi_ref`.

2. **Increase pilot episodes.** Try n=500 or n=1000 episodes.

3. **Adjust u_min/u_max.** Consider `u_min=0.005, u_max=0.05` to force stronger beta targets.

4. **Try lower gamma_base.** gamma_base=0.90 (instead of 0.95) amplifies the nonlinearity effect.

5. **Re-run task search.** After improving calibration, re-run `run_phase4_activation_search.py` — the current selected tasks may not be the most operator-sensitive under a better calibrated schedule.

**Phase IV-B is blocked pending activation gate pass.**
