# Phase IV-A2 Eligibility Report

Option C gate evaluation on the best micro-reward candidate from the updated
Phase IV-A2 search grid.

## Context

- Grid: `get_phase4a2_search_grid(max_per_family=20)` — 80 configs total
  (20 per family: `dense_chain_cost`, `shaped_chain`, `two_path_chain`,
  `dense_grid_hazard`), including 10 micro-reward configs in Family A
  (`terminal_reward=0.0`, `reward_bound=|step_reward|`).
- Pilot budget: `n_pilot_episodes = 1000` (scoring) and `n_episodes = 1000`
  (replay).
- Seed: 42.

## Best micro-reward candidate

| field | value |
|---|---|
| family | `dense_chain_cost` |
| step_reward | -0.02 |
| terminal_reward | 0.0 |
| horizon | 20 |
| gamma | 0.95 |
| reward_bound | 0.02 |
| n_states | 20 |
| micro_reward | true |

CSV row idx=0 has the highest `mean_abs_u_pred` (0.01652); the three other
micro configs at `T=20, gamma=0.95` tie on scoring metrics (identical
schedules because `reward_bound` scales out of `u_pred`) but differ in
`reward_bound`, which matters for replay. We picked idx=0 (smallest
`step_reward` magnitude) per the `argmax(mean_abs_u_pred)` rule.

## Option C gate evaluation

### GATE 1 — `mean_abs_u_pred >= 5e-3` (scoring-time)

Schedule-predicted mean `|u|` at `n_ep=1000`:

| metric | value | threshold | status |
|---|---|---|---|
| `mean_abs_u_pred` | **0.01652** | >= 5e-3 | **PASS** |
| `frac_u_ge_5e3` | 1.000 | — | all 20 stages active |
| `informative_stage_frac` | 1.000 | — | all stages informative |

### GATE 2a — `mean_abs_u_replay_informative` OR `median_abs_u_replay_informative` >= 5e-3 (replay-time)

Measured over 20,000 replayed transitions (1000 episodes x 20 stages):

| metric | value | threshold | status |
|---|---|---|---|
| `mean_abs_u_replay_informative` | **0.006348** | >= 5e-3 | **PASS** |
| `median_abs_u_replay_informative` | **0.006708** | >= 5e-3 | **PASS** |
| `frac_informative_u_ge_5e3` | 0.8355 | — | 83.6% of informative transitions are active |

### GATE 2b — `frac_informative >= 10%` (replay-time)

| metric | value | threshold | status |
|---|---|---|---|
| `frac_informative_transitions` | **0.8807** (88.07%) | >= 10% | **PASS** |
| `n_informative_transitions` | 17,613 / 20,000 | — | — |

## Verdict

**All three Option C gates PASS** for the best micro-reward candidate. The
`dense_chain_cost` family with `step_reward=-0.02, terminal_reward=0.0,
horizon=20, gamma=0.95` is eligible for the Phase IV-A2 mainline activation
suite.

## Replay diagnostics (full)

Global replay metrics (all 20,000 transitions):

| metric | value |
|---|---|
| `mean_abs_u_replay_global` | 0.005760 |
| `frac_u_ge_5e3` | 0.7532 |
| `mean_abs_delta_d_global` | 0.002806 |
| `frac_delta_d_ge_1e3` | 0.9179 |
| `target_gap_norm_global` | 0.009428 |
| `mean_beta_used` | 0.0770 |
| `mean_KL_to_prior` | 4.89e-06 |

Informative-conditioned subset (n=17,613, margin>0 and informative stage):

| metric | value |
|---|---|
| `mean_abs_u_replay_informative` | 0.006348 |
| `median_abs_u_replay_informative` | 0.006708 |
| `frac_informative_u_ge_5e3` | 0.8355 |
| `mean_abs_delta_discount_informative` | 0.003092 |
| `target_gap_norm_informative` | 0.010659 |

Top-quartile fallback (|margin| >= 0.1747, n=5,666):

| metric | value |
|---|---|
| `mean_abs_u_replay_topquartile` | 0.007028 |
| `frac_topquartile_u_ge_5e3` | 1.000 |

## Schedule audit: beta_used_t

`beta_used_t` over T=20 stages ranges from 0.0326 (t=0) to 0.3333 (t=19),
monotonically increasing with one mild non-monotonicity at t=16 (0.0947
following t=15 at 0.1089). **No `beta_used_t=0` entries** — the
informativeness signal is non-degenerate at every stage. `u_ref_used_t` is
pinned at 0.01826 for stages 0–15 (trust-region active) and decays to
~0.0067 at stages 18–19 (near-terminal stages where safe clipping binds).

## Provenance

- Scoring CSV: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/weighted_lse_dp/phase4/task_search/phase4A2_candidate_scores.csv`
- Best cfg: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/weighted_lse_dp/phase4/task_search/phase4A2_best_micro.json`
- Best schedule: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/weighted_lse_dp/phase4/task_search/phase4A2_best_micro_schedule.json`
- Suite: `/Users/liq/Documents/Claude/Projects/LSE_RL/experiments/weighted_lse_dp/configs/phase4/phase4A2_activation_suite.json`
- Replay output: `/Users/liq/Documents/Claude/Projects/LSE_RL/results/weighted_lse_dp/phase4/counterfactual_replay_4a2/`
- Seed: 42

## Harness note: micro_reward kwarg handling

`build_phase4a2_task` dispatch strips only `{"family", "reward_bound",
"appendix_only", "severe_variant"}` as meta-keys; `micro_reward` therefore
propagates as a factory kwarg and raises `TypeError`. The scoring harness
stripped the flag from each cfg before calling `score_all_candidates`
and re-attached it to the output rows post-hoc. The suite JSON passed to
the replay runner also omits `micro_reward` from the cfg payload. No
source files were modified; the fix is in the harness only.
