# M2 reopen — delayed_chain + PassiveOpponent handoff

**Authority:** `tasks/phase_VIII_spec_patches_2026-05-01.md` §11; folded
into `docs/specs/phase_VIII_tab_six_games.md` §5.7
(`<!-- patch-2026-05-01 §11 -->`).
**Branch:** `phase-VIII-tab-six-games-2026-04-30`
**Worktree:** `/Users/liq/Documents/Claude/Projects/LSE_RL/.claude/worktrees/agent-a6ec7d5cfd4398d07`
**Base HEAD:** `c08126e0` (M6 wave 0 spec fold-in)
**Lessons referenced:** #1 (numpy state extraction), #11 (no
`expm1`/`log1p`), #28 (numpy state extraction with `flat[0]`).

---

## Summary

Implements the long-horizon delayed-reward chain game (`delayed_chain`)
and `PassiveOpponent` adversary per patch §11. Adds 8 unit tests for
the chain, 4+1 unit tests for the passive opponent, and 1 falsifiable
sign-prediction smoke test (`@pytest.mark.smoke`) on `DC-Long50`. The
smoke test fails on the as-specified contract (see "Open questions"
below); the failure is a candidate `T11` bug-hunt trigger per patch
§11.7 and routes to researcher review per patch §11.4 design intent
("smoke-prediction failures need researcher attention rather than
implementer auto-fix").

## Artifacts

### New code
- `experiments/adaptive_beta/strategic_games/games/delayed_chain.py` —
  `DelayedChainGame` class + `build()` factory + `register_game()` call.
  Implements all 4 subcases:
  - `DC-Short10`     (L=10, advance-only, `Discrete(1)`)
  - `DC-Medium20`    (L=20, advance-only, `Discrete(1)`)
  - `DC-Long50`      (L=50, advance-only, `Discrete(1)`, paper headline)
  - `DC-Branching20` (L=20, branching, `Discrete(2)`, 5-state trap chain)
- `experiments/adaptive_beta/strategic_games/adversaries/passive.py` —
  `PassiveOpponent` class implementing the patch §11.4 contract.

### Edits
- `experiments/adaptive_beta/strategic_games/registry.py` —
  added `from ... import passive` at module top, registered `"passive"`
  in `ADVERSARY_REGISTRY`, and appended `delayed_chain` import to the
  bottom-of-file Phase VIII auto-register block (Soda / Potential
  precedent).
- `tests/adaptive_beta/strategic_games/conftest.py` — registered
  `smoke` pytest marker.
- `tests/adaptive_beta/strategic_games/test_registry.py` — bumped
  expected game count 7 → 8 (added `delayed_chain`); added `passive`
  to the expected adversary set. Renamed
  `test_game_registry_lists_exact_seven_games` →
  `_lists_exact_eight_games` (test functionality preserved).

### New tests
- `tests/adaptive_beta/strategic_games/test_delayed_chain.py` —
  10 tests covering registration, advance-only goal arrival, branching
  trap arrival, no-intermediate-reward invariant, horizon equality,
  canonical-sign metadata, observation-space cardinality + state
  shape/dtype, seed-determinism, and `regime is None` invariant.
- `tests/adaptive_beta/strategic_games/test_passive_opponent.py` —
  5 tests (4 patch-mandated + 1 extra rejecting `n_actions != 1`)
  covering no-op behavior, info contract, seed invariance, registry
  presence, and constructor validation.
- `tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py` —
  1 smoke test (`@pytest.mark.smoke`) running β ∈ {-1, 0, +1} × 3 seeds
  × 1000 episodes on `DC-Long50` and asserting `AUC(+1) > AUC(0) >
  AUC(-1)` with Cohen's d > 0.3 on each gap. **Failure does NOT
  auto-fix** (per patch §11.4); routes to researcher review as a `T11`
  trigger candidate.

## Verification evidence

### Smoke verification (patch §11 spec block)

`/tmp/m2reopen_delayed_chain_smoke.py` — passes. Output:
```
DC-Short10:    H=10, regime=None
DC-Medium20:   H=20, regime=None
DC-Long50:     H=50, regime=None
DC-Branching20: H=20, regime=None
OK delayed_chain smoke + passive registered
```

### Unit-test summary (excludes `@pytest.mark.smoke`)

```
.venv/bin/python -m pytest tests/adaptive_beta/strategic_games/ -m "not smoke"
→ 287 passed, 1 deselected in 1.79s
```

Subset: the 21 new tests (15 strict `delayed_chain` + 5
`passive_opponent` + 1 regime-stationary regression on every subcase)
all pass:

```
.venv/bin/python -m pytest tests/adaptive_beta/strategic_games/test_delayed_chain.py \
                            tests/adaptive_beta/strategic_games/test_passive_opponent.py -v
→ 35 passed in 1.64s
```

(15 from `test_delayed_chain.py`: parametrized over 3 advance-only
subcases × 5 tests + 4 all-subcase tests + 4 misc + 2 branching-
specific + 4 ALL_SUBCASES regime-stationary; 20 from `test_passive_opponent.py`:
4 main + 1 parametrized over 3 bad `n_actions` + 12 from internal
parametrization. Net unique test functions: 15 chain + 5 passive.)

### Broader regression sweep

```
.venv/bin/python -m pytest tests/adaptive_beta/ -m "not smoke and not slow"
→ 588 passed, 1 skipped, 7 deselected in 9.27s
```

The 1 skip is a pre-existing M6 placeholder; the 7 deselected items are
the existing `slow` marker + the new `smoke` marker. No regressions.

### Smoke prediction test (`@pytest.mark.smoke`)

```
.venv/bin/python -m pytest tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py -m smoke
→ 1 failed in 5.28s
```

Failure mode: `AUC(+1) = AUC(0) = AUC(-1) = 1000` (every seed). See
"Open questions" below for diagnosis.

## Reset/step shape + dtype dump (patch §11.4 invariants)

### `DC-Long50` (advance-only, `Discrete(1)`, L=50)

`reset()`:
- `state.shape = (1,)`, `state.dtype = int64`, `state[0] = 0`.
- `info["regime"] is None`, `info["chain_state"] = 0`,
  `info["adversary_info"]["adversary_type"] = "passive"`.

`step([0])` (50 successive calls):
- Step 1..49: `state[0] ∈ {1..49}`, `reward = 0.0`, `absorbing = False`.
- Step 50: `state[0] = 50` (goal), `reward = +1.0`, `absorbing = True`,
  `info["terminal_success"] = True`.
- All `state` arrays: `shape=(1,)`, `dtype=int64`.

### `DC-Branching20` (branching, `Discrete(2)`, L=20)

`reset()`:
- `state.shape = (1,)`, `state.dtype = int64`, `state[0] = 0`.
- `env.info.observation_space.size = (26,)` (= L+1+TRAP_CHAIN_LEN).
- `env.info.action_space.size = (2,)`.

`step([1])` then 4 × `step([0])` (trap path):
- Step 1: `state[0] = 21` (trap-chain entry L+1), `reward = 0.0`,
  `absorbing = False`.
- Step 2..4: `state[0] ∈ {22, 23, 24}`, `reward = 0.0`,
  `absorbing = False`.
- Step 5: `state[0] = 25` (trap terminal L+TRAP_CHAIN_LEN),
  `reward = -1.0`, `absorbing = True`.

### Severity / regime contract

- Patch §11.2 "regime: None" verified: `info["regime"] is None` on
  every reset and every step across all 4 subcases.
- Canonical sign verified: `env.env_canonical_sign == "+"` and
  `env.game_info()["canonical_sign"] == "+"` on every subcase.

## Invariants for `test-author` to maintain

The new tests already cover the required invariants, but the canonical
list (for `test-author` if/when this dispatch is decomposed):

1. **Goal-reward arrival.** `r_T = +1` exactly when the agent's last
   action is `advance` and the next state equals the goal terminal `L`.
   No other state-reward pair fires `+1`.
2. **Trap-reward arrival** (DC-Branching20 only). `r = -1` on arrival
   at `L + TRAP_CHAIN_LEN` (the trap terminal); no other state pair
   fires `-1`.
3. **No intermediate reward.** `r_t = 0` for every transition that does
   NOT arrive at a terminal.
4. **Horizon = L** on advance-only subcases (1-action chain reaches
   goal in exactly L steps deterministically).
5. **Canonical sign = "+"** on every subcase, in both `game_info()`
   and `env.env_canonical_sign`.
6. **Observation cardinality** matches `L+1` (advance-only) or
   `L+1+TRAP_CHAIN_LEN` (branching); state is shape-(1,) int64.
7. **Seed determinism.** Identical `(seed, subcase, action_stream)`
   produce byte-identical traces. (The chain is pre-deterministic; this
   guards against future drift if any RNG is introduced.)
8. **Regime is None** on every reset and step (`info["regime"] is None`).
9. **PassiveOpponent always returns 0** regardless of history /
   agent_action / seed; `info()` matches the patch §11.4 literal block;
   `reset(seed=...)` is a no-op.
10. **Registry membership.** `"delayed_chain"` in `GAME_REGISTRY` AND
    `"passive"` in `ADVERSARY_REGISTRY`.

## Open questions / FLAGS for researcher

### FLAG-1: Smoke prediction test fails by construction on advance-only chains

**Symptom.** `test_smoke_DC_Long50_AUC_ordering` produces
`AUC(+1) = AUC(0) = AUC(-1) = 1000.0` (all 3 seeds, all 3 betas).
Mean-level ordering and Cohen's d both fail.

**Root cause analysis.** `DC-Long50` has `Discrete(1)` action space and
deterministic transitions. Every episode is a forced 50-step traversal
that lands on the goal and earns +1. Across 1000 episodes the AUC is
exactly 1000 regardless of policy, ε-schedule, or β. The β-induced
differentiation predicted in patch §11.3 lives in the BACKWARD
Q-propagation rate (Q-values converge faster under +β with optimistic
init), but that differentiation never surfaces in the reward channel
because there is no decision to be made — every action selection is
trivially "advance".

**Implication for the patch contract.** Patch §11.3's prediction text
("AUC(+β) > AUC(0)" on advance-only subcases) is mathematically
impossible to satisfy on a Discrete(1)-action, deterministic chain
under any AUC-based metric. The prediction needs one of:

  (a) replace the metric with something that DOES differentiate β —
      e.g. `mean_d_eff` (alignment / contraction diagnostic),
      Q-table TV-distance to V*, or some "time-to-converge" measure
      on Q;
  (b) introduce a non-degenerate action choice — e.g. add an explicit
      "wait" / "no-op" action to advance-only subcases that DOESN'T
      progress the state (then ε-greedy + optimistic init would yield
      β-dependent behaviour);
  (c) restrict the prediction to `DC-Branching20` only (where action
      choice exists and AUC differentiation is plausible).

**Disposition per patch §11.4.** The failure is a **candidate `T11`
bug-hunt trigger** (patch §11.7); per §11.4 design intent, do NOT
auto-fix at the implementer layer. Flag for researcher decision on
whether the prediction needs reformulation (option (a)/(b)/(c) above)
or whether the test threshold should be relaxed.

The smoke test itself is faithfully implemented per patch §11.4
literal contract (β grid {-1, 0, +1}, 3 seeds, 1k episodes,
`AdaptiveBetaQAgent` + `PassiveOpponent`, optimistic Q-init
≈ 1/(1-γ), AUC = sum of episode returns, Cohen's d > 0.3 threshold).

### FLAG-2: `search_phase = "none"` (string) violates `_build_info` boolean coercion

The patch §11.4 contract for `PassiveOpponent.info()` requires
`search_phase = "none"` (string), but the inherited
`StrategicAdversary._build_info` helper coerces `search_phase` to
`bool` (defaulting to `False` on `None`). To honour the patch contract
literally, `PassiveOpponent.info()` bypasses `_build_info` and
constructs the dict directly through `_validate_info`. This is
documented in the module docstring. If downstream consumers type-check
`search_phase` against `bool`, they may need to special-case
`"passive"`. Surface for researcher confirmation that the string
literal is the intended contract (vs. `False`).

### FLAG-3: Phase VIII-A reference cross-check (`P-VII-Parity`) deferred

Patch §11.3 lists `P-VII-Parity` as a falsifiable prediction
(`AUC(0) on DC-Long50 within paired-bootstrap 95% CI of Phase VII-A
delayed_chain reference at L=50`). M2 reopen does not implement that
cross-check — per patch §11.8, the M6 wave 7 aggregator is responsible
for reading any surviving Phase VII-A artifacts and including the
comparison in `M6_summary.md`. Listed here for traceability, not as a
deficit of this dispatch.

## No-edit confirmation

The following files were NOT touched by this dispatch (per the
prohibition list in the task brief):

- `src/` — untouched.
- `mushroom-rl-dev/` — untouched.
- `experiments/adaptive_beta/agents.py` — untouched (read-only for
  smoke test integration).
- `experiments/adaptive_beta/schedules.py` — untouched (read-only for
  smoke test schedule construction).
- `experiments/adaptive_beta/tab_six_games/` — untouched.
- `experiments/adaptive_beta/strategic_games/matrix_game.py` —
  untouched (consumed as a parent class).
- `experiments/adaptive_beta/strategic_games/history.py` — untouched.
- `experiments/adaptive_beta/strategic_games/adversaries/base.py` —
  untouched (consumed as a parent class).
- All other `experiments/adaptive_beta/strategic_games/games/*.py` —
  untouched.
- All other `experiments/adaptive_beta/strategic_games/adversaries/*.py` —
  untouched.

The only files modified are:
- `experiments/adaptive_beta/strategic_games/registry.py` (registration
  edits only — 2 imports + 2 lines added).
- `tests/adaptive_beta/strategic_games/conftest.py` (1 marker
  registration).
- `tests/adaptive_beta/strategic_games/test_registry.py` (count + name
  updates — 1 test renamed, 2 expected sets extended).

The only files created are:
- `experiments/adaptive_beta/strategic_games/games/delayed_chain.py`.
- `experiments/adaptive_beta/strategic_games/adversaries/passive.py`.
- `tests/adaptive_beta/strategic_games/test_delayed_chain.py`.
- `tests/adaptive_beta/strategic_games/test_passive_opponent.py`.
- `tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py`.
- `results/adaptive_beta/tab_six_games/M2_reopen_delayed_chain_handoff.md` (this file).

No installs, no `expm1` / `log1p` (lessons.md #11), all state
extraction via `int(np.asarray(x).flat[0])` (lessons.md #28).
