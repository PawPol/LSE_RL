# Phase II R9 — Adversarial Codex Review

**Focus**: Challenge whether stress families isolate the classical β=0 weakness and whether event logging is sufficient for Phase III schedule calibration.  
**Branch**: phase-II/closing  
**Date**: 2026-04-17

---

## New Findings Only

Previously fixed (R8): n_base, factory contracts, jackpot threshold.  
Previously disputed: grid_sparse_goal design, shortcut semantics, stress_type=None (R8-5 open).

---

### [P2] `regime_shift` tail-risk silently absent from calibration JSON

**File**: `run_phase2_rl.py:813-827`

Tail-risk event-key dispatch is guarded by `stress_type in ("jackpot", "catastrophe", "hazard")`. The `regime_shift` type is excluded, so no `tail_risk` block is produced for `chain_regime_shift` and `grid_regime_shift`. Spec §12 lists no such exemption. Phase III calibration consumers expecting `tail_risk` for all families will find `null`. Either explicitly document the exemption or compute tail-risk using `regime_post_change` as the event flag (already logged).

### [P2] DP runner produces no `transitions.npz` — raw_margin_quantiles silently falls back to analytic distribution

**File**: `run_phase2_dp.py` (no `set_transitions` call); `aggregate_phase2.py:686-760`

`_compute_margin_quantiles_from_transitions` returns `None` for all DP algorithm groups (no `transitions.npz`). `build_calibration_json` then falls back to `pos_margin_quantiles` from the stagewise block — the analytic uniform-distribution marginal over all (s,a) pairs, not the on-policy distribution. This is the wrong quantity for Phase III β schedule calibration, which needs the empirical tail. No log warning is emitted.

### [P2] `aligned_margin_freq` not surfaced as top-level alias in calibration JSON

**File**: `aggregate_phase2.py:1244-1261`

Spec §12 requires "aligned-margin frequency by stage." The field is present deep in `stagewise["aligned_margin_freq"]` but no top-level alias is added (unlike `margin_quantiles`). Consumers must know to drill into `stagewise`, and the field is absent for DP-only groups where RL transitions are missing.

### [P3] `mark_regime_post_change(False)` has asymmetric cancel semantics

**File**: `experiments/weighted_lse_dp/common/callbacks.py`

All `mark_*` methods only set their flag `True`. `mark_regime_post_change(status=False)` can de-assert the flag — the only cancellable event. Undocumented asymmetry, latent maintenance hazard.

### [NIT] `schema_version` field emitted with no consumer validation

**File**: `aggregate_phase2.py` (`_CALIBRATION_SCHEMA_VERSION`)

No reader-side assertion checks version in `make_phase2_figures.py` or Phase III scaffolding. The field provides no protection without validation.
