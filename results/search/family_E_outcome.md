# Family E Outcome and Phase V Search Freeze

**Date:** 2026-04-24
**Trigger:** Spec §15.4 FREEZE condition (Family E promotes 0 tasks).

## Per-family final results

| family | admitted | promoted | best `|value_gap_norm|` | promotion_mode |
|---|---|---|---|---|
| A (jackpot-vs-stream, concentration contrast) | 27  | **5**  | **7.82e-3** | `safe_active_no_distortion` |
| B (catastrophe, alternative branch)           | 120 | 0      | 4.34e-4     | — |
| B_refinement (widened B grid)                 | 213 | 0      | 6.70e-4     | — |
| C (raw stress, safety)                        | 18  | **6**  | (stress gate) | `binding_clip` |
| D (early-warning preventive intervention)     | 540 | 0      | 1.34e-3     | — |
| E1 (warning-react)                            | 858 | 0      | 8.22e-4     | — |
| E2 (opportunity-adapt)                        | 628 | 0      | 9.34e-4     | — |
| E3 (regime-switch)                            | 142 | 0      | 1.34e-3     | — |

**Final shortlist:** Family A (5 positive) + Family C (6 safety) = 11 tasks.

## The structural finding

Across 5 task families spanning 7 construction styles and ~2,000 exact-planning evaluations, a consistent pattern emerges:

| construction pattern | `|value_gap_norm|` ceiling | passes 5e-3 gate |
|---|---|---|
| **Two-branch side-by-side contest** (A: both branches propagate simultaneously from contest state) | 7.8e-3 | **yes** |
| **Sequential-decision chain** (B, D, E: action at t=0 gates a downstream branch structure) | ~1.3e-3 | no |

The safe weighted-LSE operator's value translation is **strong when both candidate branches are evaluated simultaneously from the same contest state under the same backward sweep**, and **weak when the decision gates a chain whose reward geometry is only realized in one branch**. Start-state flips DO happen in the chain-gated families (134 flips in E alone), but the operator's per-stage nonlinearity accumulates too little value differentiation backward through the sequential chain to clear the 5e-3 gate.

This is itself a theorem-level finding and belongs in the paper as a design insight.

## Spec-directed response

Per `docs/specs/phase_V_mechanism_experiments.md` §15.4:
> If E promotes 0 tasks: FREEZE the search. No Family F. Final empirical story:
> - A = constructive positive (concentration contrast).
> - C = safety/stability.
> - B/D/E = appendix-only design diagnostics showing propagation-depth variants dilute translation.

**Phase V search is now closed.** Next: WP3 (limited-backup planning diagnostics) + WP4 (safe-vs-raw stability) + WP5 (paired RL) on the frozen shortlist.

## Paper narrative (per spec §14.6 + §15.4)

Main text:
- **Figure 1:** Family A phase diagram — classical vs safe decision boundaries on the `(L, R, shape)` plane.
- **Figure 2:** Family A mechanism — signed margin, realized `d_t`, policy flip, value gap.
- **Figure 3:** Family C safety — raw instability/expansivity vs safe-clipped stability.
- **Figure 4:** RL translation for 1–2 promoted A tasks (+ optional C safety RL analogue).
- **Figure 5:** Mechanism diagnostics (`δ_d`, clip activity, value-gap traces).
- **Summary table:** shortlist metrics, baselines, main outcomes; `promotion_mode` per family.

Appendix:
- Families B, D, E as design diagnostics showing the side-by-side vs sequential-chain contrast and the value-translation ceiling.
- Phase I–IV as sanity material.
