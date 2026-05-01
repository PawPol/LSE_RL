# Phase VIII M4 — Milestone Summary

**Status:** COMPLETE
**Date opened:** 2026-04-30T23:38Z (approx)
**Date closed:** 2026-05-01T01:17Z (approx)
**Wall-clock:** ~99 minutes (across 2 sessions)
**Branch:** phase-VIII-tab-six-games-2026-04-30

## Headline

```
schedules registered:    12 (was 8 — added oracle_beta, hand_adaptive_beta,
                          contraction_ucb_beta, return_ucb_beta)
baseline agents added:   3 (Restart, SlidingWindow, TunedEpsilonGreedy)
new tests:               87 (10 test files; spec §13.2 contracts)
test sweep:              1612 passed + 2 skipped (codex verification @ b4dd828a)
G4b Codex review:        PASS (0 BLOCKER / 0 MAJOR / 0 MINOR / 0 NIT)
auto-fixes applied:      1 (M4 W2.B arm-accounting; succeeded attempt 1)
M4 verdict:              PASS
```

## Wave structure

| Wave | Task | Role | Model | Output | Verdict |
|---|---|---|---|---|---|
| 1 | M4 W1.A | algo-implementer | Opus 4.7 | schedules.py +535 lines (4 new BetaSchedule + factory + Welford + warm-start + UCB; Protocol extended with episode_info, bellman_residual, episode_return optional kwargs) | PASS |
| 1 | M4 W1.B | algo-implementer | Opus 4.7 | baselines.py 722 lines (Restart, SlidingWindow, TunedEpsilonGreedy) | PASS |
| 2 | M4 W2.A | test-author | codex gpt-5.5 xhigh | test_oracle (8) + test_hand_adaptive (7) | PASS |
| 2 | M4 W2.B | test-author | codex gpt-5.5 xhigh | test_contraction_ucb (8) + arm_accounting (4) + reward_finite (22) | FAIL→AUTO-FIX→PASS |
| 2 | M4 W2.C | test-author | codex gpt-5.5 xhigh | test_return_ucb (7) + standardisation (1) | PASS |
| 2 | M4 W2.D | test-author | codex gpt-5.5 xhigh | test_baselines (9) + beta0_collapse_preserved (12) + clipping_bounds (9) | PASS |
| 3 | M4 G4b | review-triage | codex gpt-5.5 xhigh | M4_close_review_2026-05-01T01-09-13Z.md | PASS |

## G4b Codex review verdict (mandatory per harness §9)

```
verdict:                       PASS
n_blockers:                    0
n_majors:                      0
n_minors:                      0
n_nits:                        0
operator_touch_clean:          YES   (zero edits to tab_operator.py / safe_weighted_common.py)
single_update_path_clean:      YES   (agents.py unchanged)
beta0_guard_clean:             YES   (agents.py:338 untouched; 12/12 collapse tests PASS)
log1p_clean:                   YES   (np.log throughout; ε=1e-8; lessons.md #27)
ucb_warm_start_correct:        YES   (round-robin episodes 0..6; UCB from 7; ties break on lowest arm)
welford_correct:               YES   (per-arm; standardisation correct in both schedules)
oracle_correct:                YES   (reads episode_info["regime"] only; raises on missing)
state_extraction_correct:      YES   (int(np.asarray(x).flat[0]) throughout baselines.py)
result_root_clean:             YES   (no results/weighted_lse_dp leak; lessons.md #11)
auto_fix_correct:              YES   (pull-count separation correct; spec §6.5 invariant restored)
codex_tokens:                  161,946
review_memo:                   results/adaptive_beta/tab_six_games/codex_reviews/M4_close_review_2026-05-01T01-09-13Z.md
```

## Auto-fix loop (addendum §4.2)

W2.B initial verdict: 32/34 PASS (2 FAIL in test_contraction_ucb_arm_accounting).

Root cause: `_BaseUCBBetaSchedule._compute_next_beta` only incremented `_pull_counts` inside `_record_arm_reward`, which is skipped when `_episode_reward()` returns None. ContractionUCB's `M_e = log(R_{e-1}+ε) − log(R_e+ε)` is None at episode 0 (no `R_{-1}`), so arm 0's deployment was never counted as a pull.

Fix (orchestrator-applied; 1 attempt): separate the pull-count increment from the reward-attribution path. Pull counts now track DEPLOYMENTS (every episode the arm was deployed for), not REWARD ATTRIBUTIONS. Spec §6.5 invariant ("each warm-start episode pulls each arm exactly once") is restored. 4/4 arm-accounting tests PASS post-fix. G4b confirmed the fix is correct.

## Key invariants honored

- **Operator-touch alarm: clean** across all 7 dispatches (G4b confirmed via diff).
- **Single TD-update path: clean.** `AdaptiveBetaQAgent._step_update` not modified.
- **β=0 bit-identity guard at agents.py:338 untouched.** 12/12 collapse tests PASS.
- **No `expm1` / `log1p` anywhere** (lessons.md #27).
- **State extraction via `int(np.asarray(x).flat[0])`** in baselines.py (lessons.md #28).
- **No path leaks under `results/weighted_lse_dp/`** (lessons.md #11).
- **All pytest under `.venv/bin/python`** (lessons.md #1).
- **Worktree merge integrity:** registry.py 4-way conflict (M3) resolved cleanly; M4 ran without worktree isolation issues after the W1.A re-dispatch with explicit authorization.

## Token + dispatch budget consumed (M4)

| Role | Model | Dispatches | Tokens |
|---|---|---:|---:|
| algo-implementer | Opus 4.7 | 3 (M4.W1.A, M4.W1.A re-dispatch, M4.W1.B) | ~280,000 |
| test-author | codex gpt-5.5 xhigh | 4 (W2.A, W2.B, W2.C, W2.D) | ~280,000 |
| review-triage | codex gpt-5.5 xhigh | 1 (G4b) | ~162,000 |
| **M4 total** | | **8** | **~722,000** |

**Cumulative (M0–M4):** 30/300 dispatches; ~2.5 M tokens; ~3 hours of 36h cap.

## Bug-hunt protocol disposition

No suspicious-result triggers (T1–T10) fire on M4 — engineering milestone with no experimental results yet. M6 is the first stage where T1–T10 can fire.

## Next: M5 (metrics + analysis scaffold)

Per addendum overnight mode, M4 close is silent (G4b PASS). Auto-proceed to M5:
- Verify Phase VII metric pipeline emits §7.1 fields under smoke run.
- Build `experiments/adaptive_beta/tab_six_games/analysis/aggregate.py` (long-CSV aggregator).
- Build 5 plotting scripts (beta_sweep, learning_curves, contraction, sign_switching, safety_catastrophe).
- Schema parity tests + figure smoke tests.

M5 is Tier 1 — autonomous proceed. No Codex review at M5 close (light gate).
