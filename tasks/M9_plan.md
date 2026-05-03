# M9 — Stage 4 sign-switching composite (build plan)

## Spec authority

- `docs/specs/phase_VIII_tab_six_games.md` §10.5
- §6.1 (`oracle_beta`, `hand_adaptive_beta`)
- §6.5 (`contraction_UCB_beta`, `return_UCB_beta` schedules)

## Scope (this dispatch)

**Primary composite (M9.1):**
- G_+ component: AC-Trap, γ=0.60, β=+0.10 (per M8 G_+ classification,
  V10.9 §8.4 confirmation)
- G_− component: RR-StationaryConvention, γ=0.60, β=−0.5
  (per M8 G_- classification)
- Switching: exogenous dwell D ∈ {100, 250, 500, 1000}; ξ_t flips
  per `sign_switching_regime.py` (already exists)

**Methods (spec §10.5):**
- `vanilla` (β=0; baseline)
- `fixed_positive_TAB` (β=+0.10; from G_+ component)
- `fixed_negative_TAB` (β=−0.5; from G_− component)
- `best_fixed_beta_grid` (best across 21-arm grid; reporting)
- `hand_adaptive_beta` (rule-based: switches sign when win-rate drops)
- **`oracle_beta`** (regime-aware: β=+0.10 if ξ=G_+, β=−0.5 if ξ=G_−)
- `contraction_UCB_beta` (UCB1 over 21-arm β grid)

**Seeds:** 10 (paired with M7.1 / M7.2 / M8 envelope)
**Episodes:** 10 000

**Total runs:** 7 methods × 4 dwell values × 10 seeds = **280 runs**
(estimate ~30–45 min wall serial)

## Existing infrastructure

✅ `experiments/adaptive_beta/strategic_games/adversaries/sign_switching_regime.py` —
   regime-switching ADVERSARY (already wraps two opponents; use as-is).
✅ `tests/.../test_sign_switching_regime_adversary.py` — covers it.
✅ `experiments/adaptive_beta/tab_six_games/analysis/sign_switching_plots.py` —
   analysis-only plotter; reusable.
✅ `experiments/adaptive_beta/schedules.py` — has `OracleBetaSchedule`,
   `HandAdaptiveBetaSchedule`, `ContractionUCBBetaSchedule`,
   `ReturnUCBBetaSchedule` definitions per spec §6.

❌ `tab_six_games/composites/sign_switching.py` — **DOES NOT EXIST**.
   Needs to be built. This is the Mushroom-RL `Environment` subclass
   that wraps a (G_+, G_−) game pair, holds the ξ_t regime variable,
   and routes per-step transitions through the regime-current
   underlying env.

❌ M9 runner — Stage 1/2 runners do not dispatch `oracle_beta`,
   `hand_adaptive_beta`, or `contraction_UCB_beta` methods. Stage 1
   handles `fixed_beta_*` only; Stage 2 handles Q-learning baselines
   only. **Need a Stage 4 runner OR Stage 1 extension.**

## Build steps

1. **[env]** `experiments/adaptive_beta/tab_six_games/composites/__init__.py` +
   `composites/sign_switching.py` (~250 LOC)
   - Subclass `Environment` (Mushroom-RL).
   - Constructor takes `(env_g_plus, env_g_minus, dwell_schedule, seed)`.
   - State space: union (or product, or augmented with ξ flag —
     decide based on whether regime is observable to non-oracle methods;
     spec says "regime exposed only to oracle"). **Use augmented
     observation `(s, ξ)` for oracle, `s` only for others** — but the
     env emits the *same* state regardless; the runner reads `ξ` only
     when constructing oracle agent.
   - Step function: route per-step `(s, a, r, s')` through
     `env_g_plus.step` if ξ=+1 else `env_g_minus.step`. Increment
     dwell counter; flip ξ when counter hits dwell.
   - Expose `regime_history` for oracle-β consumption + analysis.

2. **[runner]** Extend
   `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py`
   OR build a new `runners/run_phase_VIII_stage4_composite.py` (~600 LOC).
   - Method dispatch for `oracle_beta`, `hand_adaptive_beta`,
     `contraction_UCB_beta`. The schedules already exist in
     `experiments/adaptive_beta/schedules.py`; the runner just builds
     them with the right hyperparameters.
   - **Oracle β needs regime access** — pass the composite env's
     regime history (or current ξ) to the oracle schedule per step.
     Other methods don't see it.
   - Emit composite-specific metrics: `regime_change_count`,
     `recovery_time_per_switch`, `auc_in_g_plus`, `auc_in_g_minus`.

3. **[config]** `configs/stage4_composite_AC_RR_gamma060.yaml`
   - 4 dwell values × 7 methods × 10 seeds = 280 runs.
   - composite component spec (cell_+, β_+, cell_−, β_−).

4. **[smoke]** Single-seed × 1-dwell × all 7 methods × 1k episodes
   smoke (~5 min wall).

5. **[oracle-validation gate]**: spec-mandated check before main pass —
   oracle β AUC must beat both fixed_positive and fixed_negative on
   the composite. If not: write `oracle_composite_failed.md` and
   STOP (one redesign attempt allowed per spec).

6. **[main]** 280-run main dispatch.

7. **[aggregate]** Extend `m7_aggregate.py` or new `m9_aggregate.py`
   for paired comparison. Headline: oracle vs fixed signs vs UCB
   adaptive.

8. **[memo]** `results/adaptive_beta/tab_six_games/stage4_sign_switching_composite.md`
   - Oracle-validation gate result
   - Per-dwell paired comparison
   - Recovery-time analysis (episodes-since-switch alignment)

9. **[verify + commit]** Test suite full pass; commit chain
   `phase-VIII(M9.{1..7})`.

## Acceptance for M9 → M10 promotion

- Oracle dominance on AUC (oracle > fixed_positive AND oracle > fixed_negative
  on paired-bootstrap CI, all 4 dwell values).
- Oracle dominance on recovery-time (oracle adapts faster post-ξ-flip
  than either fixed sign).
- Paired-seed comparison correct (regime synchronised across methods
  at fixed seed).
- Regime exposed ONLY to oracle (regression-tested: non-oracle methods
  do not access `regime_history`).

If acceptance fails: spec mandates one redesign of the composite, then
write `oracle_composite_failed.md` if it still fails. Do not force
adaptive β experiments without an oracle-validated composite.

## Out of scope (defer to M10)

- `contraction_UCB_beta` is INCLUDED in M9 (per spec §10.5 method list)
  but evaluated as just another scheduled β arm; the full UCB-arm
  accounting (`ucb_arm_count`, `ucb_arm_value`, switch tracking) is
  M10's domain.
- Endogenous trigger variants (rolling win/loss/predictability per
  spec §10.5) — defer to M9.2 if M9.1 oracle-gate passes.
- `RR-ConventionSwitch` recovery cell (per spec M10) — out of M9 scope.
