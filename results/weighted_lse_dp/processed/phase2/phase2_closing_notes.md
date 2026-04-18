# Phase II Closing Notes

**Status:** Approved to close — 2026-04-18
**Commit range:** `f4daef9` (initial scaffolding) → `1da886a` (P1D/P2C table fixes)
**Full run:** 194 experiments (64 RL + 130 DP), all seeds, all tasks, `paper_suite` config.

---

## Documented Caveats

### 1. `chain_jackpot` — Appendix / Secondary

The mechanism is correct: jackpot events fire exclusively at state 20, conditioned on
`jackpot_prob=0.05`. However, the event exposure is too low for primary calibration:

- Training jackpot events: ≈ 1.2 per run (`jackpot_state=20` visited ~40 times;
  5% fire rate yields ~2 events per run).
- Eval event rate: **0.0008** (4 events across 5,000 pooled eval episodes; only
  2 of 5 seeds observed any jackpot in 500 eval episodes).
- Median return: 0.0 (policy fails to reach goal in ~99% of eval episodes).

Root cause: `state_n=25, jackpot_state=20`. With horizon=60 and 120k training
transitions, the policy reaches state 20 too infrequently to accumulate stable
upper-tail statistics.

**`taxi_bonus_shock` is the primary positive-tail calibration family** for Phase III.
`chain_jackpot` is retained as an appendix / secondary task. If strengthened in a
future run, acceptable options are: move `jackpot_state` to 10–12, increase
`jackpot_prob` to 0.10, or increase eval episodes to 2,000.

### 2. `delta_vs_base` Unavailable in P2C

Phase I aggregated summaries (`phase1/aggregated/chain_base`, `grid_base`,
`taxi_base`) do not exist in this archive. Phase I in this project used continuous-state
tasks (`puddle_world`, `mountain_car` via TrueOnlineSARSALambda), which are
structurally incompatible with the Phase II tabular task families.

- The `delta_vs_base` column has been **removed** from P2C (commit `1da886a`).
- The P2C caption states this explicitly.
- **Do not make base-vs-stress degradation claims from this run.**
- The Learning Curves figure (Fig. 11.1.1) is titled "Phase II stress tasks" with
  no Phase I base overlay.

To populate `delta_vs_base` in a future run: execute Phase I tabular RL experiments
(`chain_base`, `grid_base`, `taxi_base` with QLearning/ExpectedSARSA), aggregate
under `phase1/aggregated/`, and re-run `make_phase2_tables.py`.

### 3. `grid_hazard` — No Eval Event-Conditioned Return

Eval `event_rate = 0.000` for both QLearning and ExpectedSARSA. The greedy policy
learned to route around hazard cell 12, producing zero eval hazard hits across
1,500 episodes per algo. `event_conditioned_return = None` is a correct consequence
of zero eval events — no episodes to condition on.

**This is expected behavior, not a logging bug.**

Training-time hazard hit rates confirm the mechanism is active during exploration:
- QLearning: 6.7×10⁻⁴ (≈ 80 hits per 120k training transitions)
- ExpectedSARSA: 3.9×10⁻⁴ (≈ 47 hits per 120k training transitions)

The hazard cell stress mechanism is correctly implemented. The policy simply learned
to avoid it — which is the intended agent behavior under this task design.

For Phase III, `grid_hazard` is the **primary negative-tail calibration family**
based on training-time margin statistics (q05 range [−1.0, 0.0] across 72/80 stages,
strongly negative), not eval event-conditioned returns.

### 4. Phase I RL Final-Checkpoint QA — Deferred

No `phase1/aggregated/` directory for tabular RL exists in the current archive.
The Phase I RL learning curve final-checkpoint inspection (checking for suspicious
drops at the last checkpoint) cannot be completed from this data.

**Action required in a future run:** execute Phase I tabular RL experiments,
aggregate them, then inspect `curves.mean_return[-3:]` for anomalous drops at the
final checkpoint (padded values, missing final eval, unequal-length curve averaging).

This item is deferred. It does not block Phase II close, but should be verified
before Phase I or Phase II RL figures appear in a paper submission.

---

## Phase III Calibration Family Assignments

| Role | Family | Rationale |
|------|--------|-----------|
| **Primary positive-tail** | `taxi_bonus_shock` | Eval event rate 4.5–5.7%, event-cond. return 4.3, 67–85 eval events |
| **Primary negative-tail** | `grid_hazard` | Strong negative margins (q05 down to −1.0); training-time events confirmed |
| **Secondary negative-tail** | `chain_catastrophe` | CVaR-5% = −1.15 (QL); eval event rate 1%; shortcut fraction 19–25% |
| **Appendix / secondary positive-tail** | `chain_jackpot` | Mechanism correct; event exposure insufficient for primary calibration |
| **Regime-shift reference** | `chain_regime_shift`, `grid_regime_shift` | Adaptation metrics present; large post-change windows (≥ 90% of budget) |
| **Sparse-reward baseline** | `chain_sparse_long`, `grid_sparse_goal` | Non-tail tasks; used for convergence and margin-distribution baseline |

---

## Artifact Locations

| Artifact | Path |
|----------|------|
| Raw runs | `results/weighted_lse_dp/phase2/paper_suite/` |
| Aggregated summaries | `results/weighted_lse_dp/phase2/aggregated/` |
| Calibration JSONs | `results/weighted_lse_dp/phase2/calibration/` |
| Tables (tex + csv) | `results/weighted_lse_dp/processed/phase2/tables/` |
| Figures (pdf + png) | `results/weighted_lse_dp/processed/phase2/figures/` |

---

## Phase III Entry Conditions Met

- [x] All 8 task families have calibration JSONs with `nominal_gamma`, `reward_range`,
  `n_seeds`, `base_returns`, `stress_returns`, `stagewise`, `margin_quantiles`
- [x] Calibration signs explicit and semantic for all 8 families (hardcoded lookup,
  no data-inferred defaults)
- [x] Raw signed `margin_quantiles` preserved (not aliased to `pos_margin_quantiles`)
- [x] Negative-tail information present: `grid_hazard` q05 < 0 at 72/80 stages;
  `chain_catastrophe` q05 < 0 at 36/60 stages
- [x] Regime-shift adaptation metrics present (`pre/post_change_auc`, `lag_50/75pct`)
- [x] Suite priority correct: `paper_suite` data takes precedence over `smoke`
- [x] `chain_jackpot` demoted; primary families documented above
- [x] `delta_vs_base` handled explicitly (removed from table, caveat in caption)
