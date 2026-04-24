# Family B failure diagnostic (working note)

Source: `/tmp/phaseV_rerun_v4/candidate_metrics.parquet` (120 admitted B candidates).
Purpose: inform the refinement grid per spec §14.3 (one bounded pass, no threshold drift).

## Per-variant max metrics

| variant        | n  | max |vgn| | max mass_delta_d | max policy_disagree | start_flip count |
|----------------|----|-----------|------------------|---------------------|------------------|
| single_event   | 36 | 1.53e-04  | 5.00e-02         | 2.5e-01             | 8                |
| warning_state  | 36 | 1.53e-04  | 5.00e-02         | 2.5e-01             | 8                |
| multi_event    | 48 | 4.35e-04  | 1.45e-01         | 2.5e-01             | 8                |

`shallow_early` and `matched_concentration` are WP2-implemented but excluded from the initial grid (see `_psi_grid_family_b` branches guarded by variant name). This is a grid-enumeration gap, not a factory gap.

## Closest to passing

- `multi_event` is nearest: max |vgn|=4.35e-04, hitting mass gate (0.145 ≥ 0.10). Single/warning are ~3x smaller in |vgn| and miss mass by 2x.
- Highest |vgn| row: `psi={C=5, L=4, p=0.2, event_depths=[2,3], event_mags=[5,2.5], event_probs=[0.2,0.1]}`, `lam=-0.187` vs `lam_tie=-0.176`. **Not** start-state-flipped; `policy_disagreement=0`. `contest_gap_norm=2.3e-03` (inside strict band).
- Highest |vgn| row with `start_state_flip=1`: |vgn|=6.0e-07. Flip rows all sit essentially AT the tie (contest_gap_norm ≈ 0), so classical Q values collapse to equal and the safe operator produces no appreciable shift — symptomatic of weak geometry asymmetry.

## Dominant shortfall

Per-gate pass rates on the 120 admitted B candidates:

| gate | pass | note |
|------|------|------|
| contest_gap_norm ≤ 0.01       | 84/120 | strict band |
| mass_delta_d ≥ 0.10           | **3/120** | mass signal only in multi_event |
| |value_gap_norm| ≥ 0.005      | **0/120** | dominant blocker |
| disagree ≥ 0.05 OR start_flip | 24/120 | all via start_flip at contest_gap≈0 |
| cert OK                       | 120/120 | clip never violates |
| clip gate (binding or safe)   | 120/120 | `safe_active_no_distortion` (Fam A pattern) |

**Conclusion — shortfall is dominated by `value_gap_norm` below gate (0 passes, gate=5e-3, max=4e-4).** Mass is secondary (3 passes). `contest_gap_norm` is fine. The subset with tie + cert + clip + disagree still tops out at |vgn|=6e-07.

This matches spec §14.3's framing: B activates (mass_delta_d hits 0.145 for multi_event) but the activation does not translate into a value difference — the two branches are classically balanced AND geometrically nearly symmetric under the deployed clipped schedule.

## Refinement directions implied

- **Larger C** (current max=5): a larger catastrophe magnitude widens the asymmetric tail that the safe operator penalizes non-linearly, directly targeting |vgn|.
- **Smaller γ** (current fixed 0.95 → add 0.90): steeper discount amplifies the relative weight of the immediate bonus vs delayed catastrophe, sharpening the geometric asymmetry between branches at x_c.
- **Enable `shallow_early` and `matched_concentration`**: both variants already exist in `family_b_catastrophe.py`; they concentrate reward differently on branch A, which is exactly the "concentration" lever §14.3 names.
- **Vary b/(p·C) ratio** (current fixed by b=1, C∈{2,5}, p∈{0.05,0.1,0.2}): sweeping this ratio pushes the classical tie to different magnitudes of `b_safe`, which changes the relative scale of reward_scale in the denominator of `value_gap_norm` and exposes higher-moment structure the classical tie cancels.
- **Multi-event already wins on mass**: keep K=2 but add more `C ∈ {10, 20}` points and vary `warning_depth` for `warning_state`.

## Grid-size budget check

~240-candidate ceiling per §14.3. Planned grid size after the expansion (5 variants × richer C/γ/ratio/p/L/warning_depth grids × 7-point eps band) is estimated ~200–240 after variant-specific compatibility pruning. Verified below in the runner output.
