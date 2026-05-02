# M7 — Stage 2 (Fixed TAB vs Baselines) dispatch plan

## Spec authority

- `docs/specs/phase_VIII_tab_six_games.md` §10.3 (Stage 2)
- §6.3 (baselines: `RestartQLearningAgent`, `SlidingWindowQLearningAgent`,
  `TunedEpsilonGreedyQLearningAgent` — all defined in
  `experiments/adaptive_beta/baselines.py`)
- §6.3 patch-2026-05-01 §3 (strategic-learning agent baselines:
  `regret_matching_agent`, `smoothed_fictitious_play_agent` — promoted
  from M11 to M7; **agent wrappers do NOT yet exist as committed
  code**; the underlying opponent classes do under
  `experiments/adaptive_beta/strategic_games/adversaries/`)

## User-narrowed scope (this dispatch)

- **Methods (6)**: `fixed_positive_TAB`, `fixed_negative_TAB`,
  `best_fixed_beta_grid` (reporting aggregate),
  `restart_Q_learning`, `sliding_window_Q_learning`,
  `tuned_epsilon_greedy_Q_learning`
- **Cells (4 Tier II headline)**: AC-Trap, SH-FMR,
  RR-StationaryConvention, DC-Long50
- **γ grid (4)**: {0.60, 0.80, 0.90, 0.95}
- **Seeds (10)**: 0–9 (match V10 consistency)
- **Episodes**: 10 000

## Dispatch arithmetic

| Method group | Re-run? | Runs |
|---|---|---|
| TAB fixed_positive / fixed_negative / best_fixed_beta_grid | derivable from V10 Tier II (4 cells × 4 γ × 21 β × 10 seeds = 3 360 already on disk) | 0 new runs |
| 3 Q-learning baselines | new dispatch needed | 3 × 4 × 4 × 10 = 480 |

Total new runs: **480** (not 960). The TAB arms are extracted from V10
Tier II raw artifacts via aggregator, not re-run, to avoid duplicate
compute and preserve paired-seed integrity (same RNG seeds drove both
TAB and baseline arms in the same script).

## ⚠️ Open scoping question for user — BLOCKING

**Q: TAB arms — extract from V10 Tier II (re-use, paired by seed), or
re-dispatch in Stage 2 alongside baselines?**

- **(a)** Extract from V10 (480 new runs total, ~25 min wall, paired
  seeds preserved; faster, no duplicate compute, identical
  `q_init=0`, `α=0.1`, `ε-greedy 1.0→0.05 decay 5000`).
- **(b)** Re-dispatch (1 440 runs total, ~75 min wall, fresh seeds;
  slower, but Stage 2 runner is fully self-contained).

Spec §10.3 does not mandate a single approach; "paired-seed
comparison" is satisfied either way as long as `seed_list` matches.
Recommendation is (a). **Awaiting confirmation before building.**

## Build steps (after approval)

- [ ] **[runner]** Author `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage2_baselines.py`
  - Mirror Stage 1 runner CLI/IO contract.
  - Method dispatch: `restart_Q_learning` →
    `RestartQLearningAgent`; `sliding_window_Q_learning` →
    `SlidingWindowQLearningAgent`; `tuned_epsilon_greedy_Q_learning`
    → `TunedEpsilonGreedyQLearningAgent`.
  - Per spec §6.3: emits `return`, `length`, `epsilon`,
    `bellman_residual`, `nan_count`, `divergence_event=0`. Does NOT
    emit operator-mechanism metrics.

- [ ] **[config]** `configs/stage2_baselines_headline.yaml`
  - 3 methods × 4 cells × 4 γ × 10 seeds = 480 runs.
  - `t11_guard: cohens_d` for stochastic cells; relative-gap floor
    for DC-Long50 (per v5b).

- [ ] **[smoke]** Single-seed smoke at AC-Trap γ=0.95 (3 runs, ~2 min)
  to validate runner emits expected schema.

- [ ] **[test]** Add `tests/adaptive_beta/tab_six_games/test_stage2_runner_smoke.py`
  covering the 3 baseline-method codepaths.

- [ ] **[main]** Dispatch the 480-run main pass (option a) or 1 440-run
  pass (option b).

- [ ] **[aggregate]** `scripts/figures/phase_VIII/m7_aggregate.py` —
  joins V10 Tier II TAB rows + Stage 2 baseline rows; outputs
  `processed/main_fixed_tab_results.csv` and
  `figures/main_learning_curves.pdf`.

- [ ] **[memo]** `results/adaptive_beta/tab_six_games/stage2_fixed_tab_vs_baselines.md`
  - Paired-bootstrap CIs (B=20 000) for each baseline vs vanilla and
    vs `best_fixed_TAB` per (cell, γ).
  - Honesty rule per spec §10.3: report ALL outcomes including
    baseline wins.
  - **G_+ / G_− preliminary classification** — feeds M8 input.

- [ ] **[verify]** Test suite full pass; commit chain
  `phase-VIII(M7.{1..7})` per step.

## Deferred / out-of-scope-this-dispatch

- **Strategic-learning agent baselines** (`regret_matching_agent`,
  `smoothed_fictitious_play_agent`) — spec promotes these from M11 to
  M7, but **agent wrappers are not yet implemented** (only opponent
  classes exist). Implementation = wrap existing
  `regret_matching.py` / `smoothed_fictitious_play.py` opponents into
  the `AdaptiveBetaQAgent` agent interface (~150 LOC × 2 + tests).
  Splitting this from the Q-learning-baseline dispatch keeps the
  critical-path Stage 2 result unblocked. **Recommend addressing as a
  separate M7.2 sub-milestone after M7.1 (this plan) closes.**

- **6-game scope** — spec §10.3 calls for "all six games + DC-Long50";
  user narrowed to the 4 Tier II headline cells. The remaining 26
  cells have `fixed_beta` only at γ=0.95 in V10 Tier I, so a true
  6-game M7 would require expanding Tier II to all 30 cells × 4 γ
  for the baseline arms (~3 000 extra runs). Recommend keeping this
  out of scope until M7.1 acceptance is in hand.

## Acceptance for M7 → M8 promotion

- Paired-seed comparison TAB vs baselines vs vanilla on all 4 cells × 4 γ.
- No silent drops.
- Honesty rule: explicit baseline-win cells reported.
- ≥ 1 G_+ candidate AND ≥ 1 G_− candidate identified (else stop
  adaptive sign-switching work per spec §10.4).
- User sign-off on M7 → M8 transition.
