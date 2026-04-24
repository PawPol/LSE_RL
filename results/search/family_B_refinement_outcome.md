# Family B refinement pass outcome (spec §14.3)

**Date:** 2026-04-23
**Runner invocation:**
```
.venv/bin/python3 -m experiments.weighted_lse_dp.runners.run_phase_V_search \
  --output-root /tmp/phaseV_family_B_refinement \
  --seed 42 \
  --family-b-variant refinement \
  --families B
```
**Config block:** `experiments/weighted_lse_dp/configs/phaseV/search.yaml` → `family_params.B_refinement`.
**Manifest:** `results/summaries/experiment_manifest.json` with `variant_set = "B_refinement"` and `family_b_variant = "refinement"`.

## Headline

- **Total B candidates admitted:** 213 (cap: 240 per §14.3; actual below cap).
- **Total B promoted:** 0.
- Grid size: 31 unique ψ × 7-point ε-band, prefiltered at `contest_gap_norm ≤ 0.02`.

## Per-variant breakdown

| variant | n admitted | n near-tie (strict) | max |vgn| (all) | max |vgn| (strict) | max mass_delta_d |
|---|---|---|---|---|---|
| single_event          | 70 | 64 | 3.77e-04 | 3.77e-04 | 7.50e-02 |
| warning_state         | 35 | 35 | 2.56e-05 | 2.56e-05 | 0.00e+00 |
| shallow_early         | 21 | 15 | 1.19e-05 | 1.19e-05 | 0.00e+00 |
| multi_event           | 66 | 48 | 6.70e-04 | 2.46e-04 | 2.14e-01 |
| matched_concentration | 21 | 15 | 1.38e-04 | 1.38e-04 | 2.50e-02 |

Per-variant promoted: **all zero**.

## Per-gate pass rates (all 213 admitted)

| gate | pass rate | pass count |
|------|-----------|------------|
| `contest_gap_norm ≤ 0.01`       | 83.1% | 177/213 |
| `mass_delta_d ≥ 0.10`           | **0.5%** | 1/213 |
| `|value_gap_norm| ≥ 0.005`      | **0.0%** | 0/213 |
| `disagree ≥ 0.05 OR start_flip` | 9.9%  | 21/213 |
| cert                             | 100%  | 213/213 |
| clip gate (binding OR safe-active) | 100%  | 213/213 |

**Zero candidates pass the joint gate** (tie AND mass AND vgn AND disagree-or-flip). Rows that pass tie + mass + cert + clip also fail vgn and fail disagree-or-flip (the flip rows all sit at `contest_gap_norm ≈ 0` with |vgn| ≤ 1e-6).

## Closest near-miss

- **ψ:** `{"variant":"single_event", "L":4, "gamma":0.95, "b":1.8, "p":0.30, "C":10.0}`
- **λ:** -0.8236 (vs `lam_tie` = -0.7721)
- **Metrics:**
  - `contest_gap_norm` = 5.15e-03 (inside strict band, gate ≤ 0.01)
  - `|value_gap_norm|` = **3.77e-04** — fails vgn gate (≥ 5e-3) by factor 13.3x
  - `mass_delta_d` = 7.50e-02 — fails mass gate (≥ 0.10) by factor 1.3x
  - `start_state_flip` = 0, `policy_disagreement` = 0.0 — fails flip-or-disagree gate
  - `clip_fraction` = 0.0 (safe schedule never clips; raw derivative within cert bound)

The best row across all 213 by raw |vgn| (outside strict band) is `multi_event` at the same (C=10, p=0.30, ratio=0.6, L=4, γ=0.95) corner, with |vgn| = 6.70e-04 — still 7.5x under the gate.

**vgn improved only ~1.5x versus v4 default pass** (v4 max = 4.35e-04 at multi_event, C=5, p=0.2, L=4; refinement max = 6.70e-04 at multi_event, C=10, p=0.3, L=4). Even with 4x larger catastrophe magnitudes, 3x larger probabilities, a wider asymmetry-ratio sweep, and an extended discount range, the two branches at x_c stay classically nearly equal AND geometrically nearly symmetric under the deployed clipped schedule.

## Interpretation

The shortfall remains exactly what the v4 diagnostic flagged: `value_gap_norm` is the dominant blocker. Even at the most aggressive refinement corner (C=20, γ=0.90, ratio=1.0, various p), the safe operator's pessimistic adjustment relative to classical produces a vgn one to two orders of magnitude below the 5e-3 gate. Family B's construction — a single contest between a rare-but-large catastrophe branch and a deterministic safe branch — does not generate enough value-differentiating geometry for the safe operator to move the planner's start-state value meaningfully away from the classical value, **even when mass_delta_d and policy_disagreement activate**.

## Recommendation

**Escalate to Family D (early-warning preventive intervention) per spec §14.3/§14.4.** Family B's bounded refinement pass is exhausted per the spec's explicit "one bounded pass, do not widen B indefinitely" directive (§14.3). Key reasons:

1. The vgn gate was never approached: max |vgn| after refinement = 6.70e-04 vs gate 5.00e-03 (13x gap on the winning combo inside strict band; 7.5x outside).
2. All 5 WP2 variants were exercised. `single_event` / `warning_state` / `shallow_early` / `matched_concentration` all stayed below |vgn| = 4e-04; only `multi_event` reached the 6-7e-04 ceiling.
3. Coverage axes were all stretched to the §14.3 limits (C ∈ {5, 10, 20}, γ ∈ {0.99, 0.95, 0.90}, p ∈ {0.02, 0.10, 0.30}, ratio ∈ {0.3, 0.6, 1.0}, L ∈ {4, 8}, warning_depth ∈ {1, 2, 3}); further widening violates the §14.3 bounded-refinement rule.
4. Spec §14.4 defines Family D precisely as the structural fix: an action pays a **preventive cost** that the safe operator can credit backward along the warning trajectory faster than classical — whereas Family B only offers a safe *alternative branch*, so the catastrophe-side information never propagates along the chosen path at planning time.

Next action (out of scope for this task): open Family D implementation work package under WP2 in `tasks/todo.md`.

## Open questions filed

None. No bug was discovered in `family_b_catastrophe.py` during this pass — the factory correctly honours `γ`, `C`, `p`, `b`, `b_safe`, and all 5 variants. The refinement was purely a psi-grid widening exercise in the runner + config, as §14.3 prescribed.
