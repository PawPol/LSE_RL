# Reply to the Coding Agent: Proceed with Family E (final bounded pass)

Option 1. Do not declare the shortlist complete with only Family A + Family C. We want a stronger NeurIPS empirical story: one constructive positive family is already found, but we need a second practically interpretable positive setting if possible.

Family E is the **final bounded refinement pass before WP3/WP4/WP5**. Not open-ended.

## Design principle (regime-shift / warning-revelation WITH concentration contrast)

Concentration-contrast designs translate cleanly (Family A: max `value_gap_norm = 7.8e-3`). Pure propagation-depth designs dilute the value gap (Family B: 6.7e-4; Family D: 1.34e-3). Family E must combine:

1. **Classical near-indifference** — declared `contest_state`, two actions classically near-tied with `gap_norm ≤ 0.01`.
2. **Regime or warning interpretation** — one branch = stale continuation / old regime; the other = reacting to a newly revealed warning or opportunity signal. Practical story: classical DP is too inertial because it smooths new information through a fixed discount; our operator selectively changes the continuation coefficient where the current signal disagrees with stale continuation.
3. **Concentration contrast** — one branch concentrates the informative signal in a short interval or warning state; the other distributes comparable classical value smoothly over time. Tune tie parameter for classical indifference; use geometry parameter to change temporal concentration while preserving classical value.
4. **Correct sign regime** — optimistic sign for opportunity/adaptation tasks where the immediate informative signal is aligned; pessimistic sign for warning/catastrophe tasks where the bad signal is aligned. Key diagnostic: signed-margin alignment `β̃_t · (r − V_{t+1})` on reachable contest and post-contest states.
5. **Promotion gate** — unchanged fixed thresholds. `gap_norm ≤ 0.01`, `mass_delta_d ≥ 0.10`, `|value_gap_norm| ≥ 0.005`, `policy_disagreement ≥ 0.05` OR `start_state_flip == 1`, zero cert violations. Non-safety families accept either `binding_clip` OR `safe_active_no_distortion` modes.

## Three templates

- **E1 — warning-revelation fork.** `ignore` vs `react`. Tune adjustment cost for classical tie. Geometry: warning depth, catastrophe concentration, adjustment-cost timing, p_warn, smoothness of old-regime branch. Practical: risk-control, regime adaptation.
- **E2 — opportunity-revelation fork.** `wait` (smooth baseline) vs `adapt` (immediate cost, unlocks concentrated upside after warning/revelation). Tune immediate cost for classical tie. Geometry: opportunity concentration, delay. Practical: sparse opportunity capture.
- **E3 — regime-switch with stale continuation.** Two branches with identical classical expected value; old-regime = smooth stale continuation; new-regime = sharp immediate signal that old continuation is unreliable. Goal: reachable states with large aligned signed margin so realized `d_t < γ`.

## Budget

One bounded pass. ≤ 5,000 candidates. Exact planning only. No RL. No threshold relaxation. Outputs: `family_E_candidate_metrics.parquet`, `family_E_shortlist.csv`, `family_E_diagnostic_report.md`.

## Branching

- **If Family E promotes ≥ 1:** proceed to WP3/WP4/WP5 with best 2–3 Family A tasks, 1–2 Family E tasks, best Family C safety tasks.
- **If Family E promotes 0:** freeze. Final empirical story:
  - Family A: constructive positive mechanism where concentration contrast creates policy/value translation.
  - Family C: safety/stability demonstration where clipping prevents raw instability.
  - B/D/E: appendix-only design diagnostics showing propagation-depth variants dilute value translation.

This is not a negative-result framing — it is a constructive theory-paper framing: the paper claims a local operator mechanism and demonstrates it in settings where the signed-margin and concentration conditions are present.

## Main-paper target

1. Phase diagram for Family A (classical vs safe decision boundaries).
2. Mechanism plot for Family A (signed margin, realized `d_t`, policy flip, value gap).
3. If Family E promotes: same compact plot for Family E.
4. Safety plot for Family C (raw instability/expansivity vs safe clipped stability).
5. Small RL pilot only on exact-planning-promoted tasks.

No compute on RL until Family E either promotes or is exhausted.
