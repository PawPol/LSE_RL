# HALT 7 — M6 close, M7 user sign-off required

- **Halt UTC**: 2026-05-01T19:06:56Z
- **Milestone**: M6 close → M7 entry
- **HEAD at halt**: `2dcb92be`
- **Trigger class**: spec-mandated halt boundary, NOT a bug or
  unexpected pattern. Spec §10.2 acceptance: "User signs off on the
  promoted subcases for M7."
- **Severity**: spec-mandated user gate; no auto-promote rule.

## Why halting

1. **Spec mandates user sign-off**. Spec §10.2 (M6 → M7
   acceptance, p. 1281) explicitly requires "User signs off on the
   promoted subcases for M7." This is not in the addendum §4 auto-
   decide rules; it's a hard human gate.

2. **v7 finding reshapes M7's design**. M7 (spec §10.3) is built
   around a comparison `vanilla vs best_fixed_positive_TAB vs
   best_fixed_negative_TAB vs external baselines`. The v7 main-pass
   evidence shows:
   - **No cell has fixed +β beating vanilla.** "best_fixed_positive_TAB"
     therefore has no meaningful selection — every +β method is
     dominated by vanilla. Including it as a baseline would just
     show further confirmation of the v7 finding (positive +β
     destabilizes), not an interesting comparison.
   - The cell-by-cell **best -β arm varies**: most cells have a
     -β plateau (-2, -1, -0.5 essentially tied), but RR-Stationary/Tremble
     show β=-2 *also* destabilizes (alignment 0.10-0.14), so
     "best_fixed_negative_TAB" should be β = -1 or β = -0.5 across
     the suite, not β = -2.
   - The "best_fixed_beta_grid (reporting aggregate)" method
     becomes essentially vanilla — best β is 0 on most cells.

3. **M7+ scope likely needs revision**. M9 (sign-switching) and
   M10 (contraction-adaptive β) both build on the assumption that
   sign-specialization is informative. The v7 finding changes what
   "G_+ regime" means (no clean G_+ exists at the tested
   parameters). M9/M10 may need scope refinement before dispatch.

## Current state

- M6 closed cleanly: 1540 main + 140 recovery + 40 figures-only +
  45 ablation + smoke runs all complete.
- All HALT 4-6 resolutions landed atomically.
- Spec §5.4, §13.10, §23 changelog all v7-amended.
- Counter-intuitive findings memo + Codex GENUINE-FINDING review
  documented.
- M6_summary.md committed (HEAD `2dcb92be`).

## Recommended user actions

1. **Review M6_summary.md** at
   `results/adaptive_beta/tab_six_games/M6_summary.md` and the
   counter-intuitive findings at
   `counter_intuitive_findings.md`.

2. **Decide on M7 scope**:
   - **(α) Run M7 as-spec'd**: include "best_fixed_positive_TAB"
     as a baseline. Expected outcome: vanilla beats positive-TAB on
     every cell; this becomes paper evidence for the v7 narrative
     refinement.
   - **(β) Modify M7 to reflect v7**: drop best_fixed_positive_TAB;
     compare vanilla vs best_fixed_negative_TAB (which is β=-1 or
     β=-0.5 on most cells) vs external baselines. Reduces M7 from
     9 methods to 7-8 methods.
   - **(γ) Promoted subcases**: which subcases from M6 should
     advance to M7? Per addendum §4.1 P1-P6 all green, the auto-
     promote criterion would advance ALL 22 cells. The user may
     prefer a focused subset (e.g. drop null cells like MP-Stationary,
     drop redundant SO-Coordination/PG-CoordinationPotential pair).

3. **Decide on M9/M10 scope** (downstream): sign-switching composite
   (M9) and contraction-adaptive β (M10) were designed under the
   v2 sign-specialization assumption. v7 may render these less
   informative; user should decide whether to:
   - run them anyway (M9/M10 produce data; the "switching" finding
     may itself be a paper result),
   - reduce them in scope,
   - or reframe them as falsifiability tests for adaptive-β methods.

4. **Decide on paper draft scope**: M11 (optional advanced) and
   M12 (final recommendation) are the deliverables. Given the v7
   finding has reshaped the headline narrative from "+β regime
   exists" to "alignment-condition diagnostic + scope-correct
   negative results", the user may want to draft M12's
   final_recommendation.md outline before running M11.

## Autonomous status

Background processes: none running.

Branch state: clean working tree, last push `2dcb92be`, all
amendments + run artifacts pushed to origin.

Wall-clock budget: ~21 hours elapsed of 36-hour cap (per addendum §5).
Plenty of headroom for the entire M7-M12 pipeline if user re-scopes.

## Token budgets (approximate, for billing reference)

- opus-4-7 (orchestrator): ~ 1.6M tokens
- codex-gpt-5.5-xhigh (review/test-author): ~ 1.43M tokens

(Both within the spec-soft budget; exact figures in addendum §5
budget log when next refreshed.)
