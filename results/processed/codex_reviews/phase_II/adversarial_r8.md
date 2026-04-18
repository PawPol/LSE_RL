# Phase II R8 — Adversarial Codex Review

**Focus**: Challenge whether stress families actually isolate the classical weakness and whether event logging is sufficient to drive Phase III schedule calibration per docs/specs/phase_II_stress_test_beta0_experiments.md.  
**Branch**: phase-II/closing  
**Date**: 2026-04-17

---

## Summary Table

| Tag | Issue | File | Impact |
|-----|-------|------|--------|
| [P1] | `grid_sparse_goal` is a "bad" stress task per spec §1 — tests state-space size, not β=0 weakness | `stress_families.py`, spec §1 | Entire task family produces invalid Phase III evidence |
| [P1] | `n_base=25` for 49-state `grid_sparse_goal` corrupts all transition logs | `run_phase2_rl.py:122` | All stagewise/margin data wrong for this task |
| [P1] | `regime_shift_episode` key name mismatch between `calibration_stats.npz` and `run.json` makes fallback path dead | `aggregate_phase2.py:1464`, `run_phase2_rl.py:794` | Calibration JSON silently loses change-point if npz is missing |
| [P2] | `taxi_bonus_shock` jackpot threshold derived from absent `jackpot_reward` key; works by coincidence, breaks if `bonus_reward` ≤ 4.0 | `run_phase2_rl.py:623` | Jackpot events silently not logged for modified taxi configs |
| [P2] | `shortcut_action_taken` records attempts, not catastrophes; `shortcut_risky_path_fraction` overstates tail-risk exposure | `run_phase2_rl.py:217–221` | Phase III calibration overestimates fraction "facing tail risk" |
| [P2] | `chain_sparse_long` and `grid_sparse_goal` have no `stress_type`: zero event logging, no `event_conditioned_margins` in calibration JSON | `paper_suite.json:13–107` | Spec §12 "event-conditioned margin statistics" unmet for two families |
| [P3] | `_WRAPPER_TASKS_RL` requires manual registration; no contract ensures new wrapper tasks are registered | `run_phase2_rl.py:137–142` | Maintenance risk |
| [P3] | `_compute_episode_event_flags` OR-reduces per-transition flags; conflates "tried risky action" with outcome | `run_phase2_rl.py:483–504` | Semantic gap in shortcut fraction metric |
| [P3] | `AdaptationMetricsLogger` uses `nanmax(post)` as optimum, not smoothed plateau; inflates lag estimates | `callbacks.py:765` | High seed variance in recovery lag metrics |
| [NIT] | Derived event thresholds not written to `run.json`; event detection not reproducible by inspection | `run_phase2_rl.py:622–624` | Auditability gap |

---

## Detailed Findings

### [P1] `grid_sparse_goal` is a "bad" stress task per spec §1

**File**: `experiments/weighted_lse_dp/tasks/stress_families.py` lines 278–280; spec §1 "Bad Phase II tasks"

The spec is explicit: "Bad Phase II tasks: tasks that are simply harder because they are larger." `grid_sparse_goal` increases the grid from 5×5 to 7×7 (25→49 states) with goal-only reward that is identical in kind to `grid_base`. The stress mechanism is "increased state space + longer propagation chains." This is precisely the bad case: no rare high-magnitude immediate reward deviations, no structural alignment between `r - v` and any sign, no jackpot/hazard/regime-change events. Classical DP and β=0 RL exhibit slower convergence due to exploration depth and larger Q-table, not any structural β=0 weakness. A β>0 operator gains nothing here because `margin_beta0 = reward - v_next` is always zero except at the goal. The positive margin is structurally inevitable on every goal-reaching episode regardless of β.

No `stress_type` is assigned in `paper_suite.json` for this task, confirming the implementation itself does not claim it generates events. The task should be redesigned as a grid task with a rare large-magnitude event (e.g., `grid_jackpot` or `grid_sparse_hazard`) or removed from the mandatory paper suite.

### [P1] `n_base=25` corrupts ALL transition logs for `grid_sparse_goal`

**File**: `experiments/weighted_lse_dp/runners/run_phase2_rl.py:122`

Same as standard review finding [P1]. `TransitionLogger` uses `aug_id // n_base` and `aug_id % n_base` for timestep and state recovery. `n_base=25` on a 49-state MDP produces garbage for every transition where `aug_id >= 25`.

### [P1] `regime_shift_episode` key name mismatch — fallback to `run.json` is dead code

**Files**: `runners/aggregate_phase2.py:1464`; `run_phase2_rl.py:794`

`_CALIB_SCALAR_ADAPTATION_KEYS` contains `"regime_shift_episode"`. In `run.json`, the adaptation dict uses key `"change_at_episode"` (from `AdaptationMetricsLogger.compute()`). The fallback path in `aggregate_phase2.py` that reads `run.json` will never find `"regime_shift_episode"` — the key name differs. If `calibration_stats.npz` is missing, the change-point is silently dropped from the calibration JSON.

### [P2] `taxi_bonus_shock` jackpot threshold derived from absent `jackpot_reward` key

**File**: `run_phase2_rl.py:623`

```python
jackpot_reward_thr = float(task_config.get("jackpot_reward", 10.0)) * 0.5  # = 5.0
```

`taxi_bonus_shock` config has no `jackpot_reward` field. Threshold defaults to 5.0. The taxi base delivery reward is 1.0; bonus_reward is 5.0; total on bonus = 6.0 > 5.0 → detection works by coincidence. If `bonus_reward` ≤ 4.0, total ≤ 5.0 and jackpot events are silently not logged. The threshold should be derived from `bonus_reward` from config.

### [P2] `shortcut_action_taken` records risky attempts, not catastrophe outcomes

**File**: `run_phase2_rl.py:217–221`

`shortcut_action_taken` fires whenever `base_state == risky_state and action == 0`, regardless of whether the catastrophe outcome occurred. This fires ~95% of the time when the agent takes the risky shortcut (since `risky_prob=0.05`). `shortcut_risky_path_fraction` is the fraction of episodes where the agent **attempted** the risky action ≥ once — not the fraction where a catastrophe occurred. This conflation may mislead Phase III calibration: "fraction risking" ≠ "fraction catastrophizing." The semantics should be documented prominently.

### [P2] `chain_sparse_long` and `grid_sparse_goal` have no `stress_type`

**File**: `experiments/weighted_lse_dp/configs/phase2/paper_suite.json`

When `stress_type is None`, the runner creates a plain `TransitionLogger` (not `EventTransitionLogger`) and computes no `tail_risk`, `event_rates`, or `event_conditioned_margins`. The calibration JSON for these tasks has no event-conditioned data. Spec §12 requires "event-conditioned margin statistics" for every stress task family. For sparse-reward tasks, the relevant event is goal-reach — not logging it means the calibration JSON cannot distinguish goal-reach vs non-goal-reach episode distributions.

### [P3] `_WRAPPER_TASKS_RL` requires manual registration

**File**: `run_phase2_rl.py:137–142`

New wrapper tasks must be manually added to `_WRAPPER_TASKS_RL`. No assertion or contract catches omission. A future developer adding a regime-shift task without registering it will silently train on the un-stressed environment.

### [P3] `_compute_episode_event_flags` OR-reduces; conflates attempt rate with outcome rate

**File**: `run_phase2_rl.py:483–504`

An episode is flagged as "risky" if **any** transition has `shortcut_action_taken=True`. For agents that always attempt the risky path, `shortcut_risky_path_fraction` = 100%, even though catastrophe rate is 5%. The metric is the attempt rate, not the catastrophe rate. Semantically correct per Decision 2, but should be explicitly documented.

### [P3] `AdaptationMetricsLogger` uses `nanmax(post)` as optimum — inflates lag estimates

**File**: `experiments/weighted_lse_dp/common/callbacks.py:765`

`post_optimum = float(np.nanmax(post))` is the single best episode return after the change point. Recovery thresholds are `α * post_optimum`. A single lucky exploration episode early after the shift sets an inflated optimum, making all lag estimates appear large. The spec §8.2 says "90% of new optimum or best observed post-change plateau" — "plateau" implies a smoothed/windowed estimate, not raw maximum. `nanpercentile(post, 90)` or rolling-max would be more robust.

### [NIT] Derived event thresholds not logged in `run.json`

**File**: `run_phase2_rl.py:622–624`

`hazard_reward_thr`, `jackpot_reward_thr`, `catastrophe_reward_thr` are derived at runtime from config but not stored in `run.json`. If config values change, it is impossible to verify after the fact what thresholds were used for event detection.
