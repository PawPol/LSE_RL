# Phase VIII M2 — Milestone Summary

**Status:** COMPLETE
**Date opened:** 2026-04-30T22:57:00Z
**Date closed:** 2026-04-30T23:18:00Z (approx)
**Wall-clock:** ~21 minutes
**Branch:** phase-VIII-tab-six-games-2026-04-30

## Headline

```
Phase VIII games registered:    7 (was 5; added soda_uncertain + potential)
strategic_games tests:        219 (was 171; +48 new = 23 soda + 25 potential)
strategic_games pytest:       PASS, 1.76 s
M2 verdict:                   PASS
```

## Wave structure

| Wave | Task | Role | Model | Output | Verdict |
|---|---|---|---|---|---|
| 1 | M2 W1.A | env-builder | Opus 4.7 | soda_uncertain.py + registry +13 lines | PASS |
| 1 | M2 W1.B | env-builder | Opus 4.7 | potential.py + registry +6 lines | PASS |
| 2 | M2 W2.A | test-author | codex gpt-5.5 xhigh | test_soda_uncertain.py (23 PASS) + test_registry.py (5→7 count fix) | PASS |
| 2 | M2 W2.B | test-author | codex gpt-5.5 xhigh | test_potential.py (25 PASS) | PASS |

**Verifier gate:** independent orchestrator pytest sweep on
`tests/adaptive_beta/strategic_games/` returned **219 passed in 1.76 s**.
No separate codex verifier dispatch needed for M2 close (env + test
milestone, autonomous per harness §3.1 task-tag table).

## Soda Game (spec §5.5)

```
games/soda_uncertain.py            422 lines
subcases:                          5 (SO-Coordination, SO-AntiCoordination,
                                      SO-ZeroSum, SO-BiasedPreference,
                                      SO-TypeSwitch)
hidden ξ exposure:                 info["regime"] (oracle-only)
canonical_sign:                    None (wrong_sign undefined per spec §5.7)
SO-TypeSwitch:                     deterministic 4-type rotation modulo
                                   episode index (not stochastic)
test coverage:                     23 tests covering payoff correctness,
                                   determinism, type-switch period,
                                   info["regime"] schema, canonical_sign,
                                   horizon termination, encoder shape
```

## Potential Game (spec §5.6)

```
games/potential.py                 585 lines
subcases:                          4 (PG-CoordinationPotential,
                                      PG-Congestion, PG-BetterReplyInertia,
                                      PG-SwitchingPayoff)
canonical_sign:                    "+" (potential games admit better-reply
                                   dynamics; positive β accelerates Nash
                                   convergence)
potential function Φ:              documented in module docstring; exposed
                                   via compute_potential() helper for
                                   test verification
test coverage:                     25 tests covering payoff correctness,
                                   Φ pinned values, one-step BR strictly
                                   increases Φ, inertia penalty applied,
                                   switching-payoff period, canonical_sign,
                                   info["regime"], determinism
```

## Key invariants honored

- **Operator-touch alarm: clean.** Zero edits to `tab_operator.py` or `safe_weighted_common.py` across all 4 dispatches.
- **Single TD-update path: clean.** `agents.py` not modified.
- **Registry edit boundary: respected.** Only `registry.py` edited (the documented mechanism); no edits to `matrix_game.py`, `history.py`, existing `games/*.py`, or any adversary.
- **No source edits in test-author dispatches.** Test-author edited only test files (test_soda_uncertain.py, test_potential.py, test_registry.py count fix).
- **No `expm1`/`log1p`** (lessons.md #27).
- **All pytest under `.venv/bin/python`** (lessons.md #1).

## Open follow-up

Spec §6.6 line 513 defers oracle β lookup table for Soda subcases to a
later milestone. M9 (sign-switching composite) is where oracle beta
schedules consume `info["regime"]`; the lookup table for soda will be
declared then. Logged as M2.W1.A handoff Open Question 1.

## Token + dispatch budget consumed (M2)

| Role | Model | Dispatches | Tokens |
|---|---|---:|---:|
| env-builder | Opus 4.7 | 2 (M2 W1.A, M2 W1.B) | ~193,000 |
| test-author | codex gpt-5.5 xhigh | 2 (M2 W2.A, M2 W2.B) | ~160,000 |
| **M2 total** | | **4** | **~353,000** |

**Cumulative (M0+M1+M2):** 11/300 dispatches, ~960,000 tokens.

## Bug-hunt protocol disposition

No suspicious-result triggers (T1–T10) fire on M2 — env+test milestone with no experimental results yet.

## Next: M3 (delta adversaries)

Per addendum overnight mode, M2 close is silent. Auto-proceed to M3:
- env-builder × 3 (Opus 4.7) parallel: inertia.py, convention_switching.py, sign_switching_regime.py.
- test-author × 3 (codex gpt-5.5 xhigh) parallel.
- Verifier gate.
