# Phase VIII M3 — Milestone Summary

**Status:** COMPLETE
**Date opened:** 2026-04-30T23:18:00Z (approx)
**Date closed:** 2026-04-30T23:38:00Z (approx)
**Wall-clock:** ~20 minutes
**Branch:** phase-VIII-tab-six-games-2026-04-30

## Headline

```
Phase VIII adversaries registered: 13 (was 10; added inertia, convention_switching, sign_switching_regime)
strategic_games + tab_six_games tests: 309 passed + 1 skipped
M3 verdict: PASS
```

## Wave structure

| Wave | Task | Role | Model | Output | Verdict |
|---|---|---|---|---|---|
| 1 | M3 W1.A | env-builder | Opus 4.7 | inertia.py + registry +4 + adversaries/__init__.py +4 | PASS |
| 1 | M3 W1.B | env-builder | Opus 4.7 | convention_switching.py + registry +4 | PASS |
| 1 | M3 W1.C | env-builder | Opus 4.7 | sign_switching_regime.py + registry +4 | PASS |
| 2 | M3 W2.A | test-author | codex | test_inertia_adversary.py (11 PASS) + test_registry.py count fix | PASS |
| 2 | M3 W2.B | test-author | codex | test_convention_switching_adversary.py (10 PASS) | PASS |
| 2 | M3 W2.C | test-author | codex | test_sign_switching_regime_adversary.py (12 PASS) | PASS |

**Verifier gate:** orchestrator pytest sweep on `tests/adaptive_beta/{strategic_games,tab_six_games}/` returned **309 passed + 1 skipped, 1.79s**. M3 close PASS.

## Adversaries delivered (spec §M3 + §5.7)

```
inertia.py                  180 lines
  Sticky-action with inertia_lambda ∈ [0,1].
  Used by: AC-Inertia subcase.
  info["regime"]: not used (inertia is regime-agnostic).

convention_switching.py     268 lines
  Periodic OR stochastic switch between two conventions.
  regime ∈ {"left", "right"} exposed via info["regime"].
  Used by: RR-ConventionSwitch subcase.

sign_switching_regime.py    409 lines
  G_+ ↔ G_- controller for M9 composites.
  Two modes: exogenous dwell {100, 250, 500, 1000} OR endogenous trigger
  on rolling reward window.
  regime ∈ {"plus", "minus"} exposed via info["regime"] for OracleBetaSchedule.
  Used by: M9 sign-switching composite envs (composites/sign_switching.py
  to land in M9).
```

## Test coverage delivered (33 new tests)

```
test_inertia_adversary.py            11 tests (registration, distribution
                                              under λ extremes, validation,
                                              determinism, info schema, reset
                                              round-trip)
test_convention_switching_adversary.py 10 tests (periodic period correctness,
                                                stochastic determinism, both
                                                regimes seen, constant
                                                within episode, info schema,
                                                on_episode_end hook)
test_sign_switching_regime_adversary.py 12 tests (exogenous dwell grid
                                                 {100,250,500,1000}, bidirectional
                                                 endogenous trigger, info schema,
                                                 determinism, per-regime callable
                                                 dispatch, constructor validation)
```

Plus `test_registry.py` updated: ADVERSARY_REGISTRY count assertion 10 → 13 (M3 wave 2 W2.A fix).

## Key invariants honored

- **Operator-touch alarm: clean** across all 6 dispatches.
- **Single TD-update path: clean.** `agents.py` not modified.
- **Registry edit boundary respected.** Only `registry.py` and `adversaries/__init__.py` edited (the documented mechanisms).
- **No source edits in test-author dispatches** beyond test_registry.py count fix.
- **No `expm1`/`log1p`** (lessons.md #27).
- **All pytest under `.venv/bin/python`** (lessons.md #1).
- **Three sequential merges with manual conflict resolution** at registry.py (each adversary added to import block + ADVERSARY_REGISTRY dict; combined cleanly).

## Bug-hunt protocol disposition

No suspicious-result triggers (T1–T10) fire on M3 — adversary+test milestone with no experimental results yet.

## Next: M4 (delta β schedules + external baselines)

Per addendum overnight mode, M3 close is silent. Auto-proceed to M4. M4 is heavier:
- 4 new schedules (Oracle, HandAdaptive, ContractionUCB, ReturnUCB) + factory dispatch update.
- 3 baseline agents (Restart, SlidingWindow, TunedEpsilonGreedy).
- 9 test files.
- M4 Codex review gate fires UNCONDITIONALLY at close per harness §9 (G4b).

M4 hard prohibitions reiterated:
- NO edits to `tab_operator.py` or `safe_weighted_common.py` (operator alarm).
- NO branching of `AdaptiveBetaQAgent._step_update` (single-update-path invariant).
- β=0 bit-identity guard MUST continue to fire.
- New schedules registered in `ALL_METHOD_IDS`; `build_schedule()` factory extended (additive).
