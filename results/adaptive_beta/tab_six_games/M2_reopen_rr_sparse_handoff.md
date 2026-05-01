# M2 reopen handoff — RR-Sparse subcase (patch §1)

**Branch:** `worktree-agent-a48cc90c761964cad` (worktree off
`phase-VIII-tab-six-games-2026-04-30`)
**Base HEAD:** `c08126e0` ("phase-VIII(M6 wave 0): fold pre-M6 spec
amendment v2 per researcher critique")
**Spec authority:**
`tasks/phase_VIII_spec_patches_2026-05-01.md` §1; folded-in spec at
`docs/specs/phase_VIII_tab_six_games.md` §5.3 (RR-Sparse subcase,
marked `<!-- patch-2026-05-01 §1 -->`).
**Lessons consulted:** #1 (.venv enforced), #11 (no `expm1` / `log1p`
— not invoked here, no operator math touched).

## Summary

- Added a `sparse_terminal: bool = False` flag to
  `experiments/adaptive_beta/strategic_games/games/rules_of_road.py`
  (and supporting `c: float = 1.0`, `m: float = 0.5` paper-config
  knobs). Default `False` preserves the dense per-step shaping
  exactly — the existing `MatrixGameEnv` is returned unchanged.
- When `sparse_terminal=True`, the factory returns a thin
  `_SparseTerminalRoREnv(MatrixGameEnv)` subclass whose `step` masks
  the per-step reward to `0.0` for every non-terminal step and lets
  the terminal step pay out the matrix value. The matrix is built
  with `+c` on the coordinated diagonal and `-m` off-diagonal so the
  terminal payoff is exactly `+c` (coordinated) or `-m`
  (miscoordinated). Adversary `observe`/`info` calls and history
  bookkeeping are untouched — only the agent-visible reward is
  masked.
- Registered a new factory key `rules_of_road_sparse` in
  `experiments/adaptive_beta/strategic_games/registry.py` that wraps
  `rules_of_road.build(..., sparse_terminal=True)` with default
  `H = 20` (per patch §1.2). The dense `rules_of_road` registration
  is unchanged. Passing `sparse_terminal=` to the sparse factory is
  rejected so the registry key is the single source of truth.
- Created `tests/adaptive_beta/strategic_games/test_rules_of_road_sparse.py`
  with the seven test cases requested: registry membership, per-step
  reward zeroing, terminal-coordinated payoff, terminal-miscoordinated
  payoff, default horizon, canonical-sign parity with dense RR, and
  seed determinism (with anti-degeneracy cross-seed check).

## Artifacts

- Edited:
  `experiments/adaptive_beta/strategic_games/games/rules_of_road.py`
  (added `_SparseTerminalRoREnv` subclass, extended `build` signature
  with `sparse_terminal`, `c`, `m`; preserved dense behavior under
  default args; updated module docstring; added `Tuple` import).
- Edited:
  `experiments/adaptive_beta/strategic_games/registry.py` (added a
  `_build_rules_of_road_sparse` factory wrapper at module bottom and
  registered it under `"rules_of_road_sparse"`; the dense `rules_of_road`
  registration via the games subpackage import remains unchanged).
- Created:
  `tests/adaptive_beta/strategic_games/test_rules_of_road_sparse.py`
  (7 tests, all PASS).

## Verification evidence

### Smoke (task-supplied script)

    sparse total reward: 1.0 (should be in {+1.0, -0.5})
    horizon: 20
    OK rr_sparse smoke

### Unit tests

    $ .venv/bin/python -m pytest tests/adaptive_beta/strategic_games/test_rules_of_road_sparse.py -q
    .......                                                                  [100%]
    7 passed in 1.30s

### Reset / step shape + dtype dump

    reset: state.shape=(1,), state.dtype=int64,
           info_keys=['adversary_info', 'agent_action', 'catastrophe',
                      'episode_index', 'game_name', 'is_shift_step',
                      'opponent_action', 'phase', 'terminal_success']
    step (t=0):
           state.shape=(1,), state.dtype=int64, r=0.0, done=False,
           info_keys=['adversary_info', 'agent_action', 'catastrophe',
                      'episode_index', 'game_name', 'is_shift_step',
                      'opponent_action', 'opponent_reward', 'phase',
                      'terminal_success']

    metadata.canonical_sign     = None  (unchanged from dense)
    metadata.env_canonical_sign = None  (unchanged from dense)
    metadata.sparse_terminal    = True
    metadata.c                  = 1.0
    metadata.m                  = 0.5

### Severity-0 reduction analog (dense default unchanged)

    rules_of_road.build(..., sparse_terminal=False)  -> MatrixGameEnv      (dense)
    rules_of_road.build(..., sparse_terminal=True)   -> _SparseTerminalRoREnv

The dense default returns exactly the existing `MatrixGameEnv`
class — no subclass swap when `sparse_terminal=False`. All 8 existing
`test_games.py::test_rules_of_road_*` tests still pass (dense
trembling-hand and payoff-bias paths intact).

### Wider regression sweep

    .venv/bin/python -m pytest tests/adaptive_beta/strategic_games/
    -> 1 FAILED, 258 PASS

The single failure is **`test_registry.py::test_game_registry_lists_exact_seven_games`**:
that test hard-codes `len(GAME_REGISTRY) == 7` and a frozen seven-key
expected set. Adding `rules_of_road_sparse` legitimately grows the
registry to 8 keys. Updating that test's `EXPECTED_GAMES` set is
**outside this dispatch's allowed-edit list** (only
`rules_of_road.py`, `registry.py`, the new sparse test file, and this
handoff memo are allowed). The test fix is a one-line edit (add
`"rules_of_road_sparse"` to `EXPECTED_GAMES` and bump the literal `7`
to `8`) and is flagged below for the next dispatch.

All other adaptive_beta tests pass (broader sweep:
`tests/adaptive_beta/ --ignore=test_registry.py` → 559 pass, 1 skipped).

## Invariants for `test-author` to lock in

1. **Default-preserves-dense invariant.** Calling
   `rules_of_road.build(...)` with no `sparse_terminal` kwarg must
   return a `MatrixGameEnv` (NOT the sparse subclass) and reproduce
   the original payoff matrix bit-identically. Already covered
   indirectly by existing dense tests; consider adding an explicit
   `assert type(env) is MatrixGameEnv` guard.
2. **Sparse-flag idempotence.** Asking for the sparse factory with
   `sparse_terminal=True` (passed positionally) must be rejected
   from the `rules_of_road_sparse` registry path — currently raised
   as `TypeError` in the wrapper.
3. **Terminal-only reward identity.** Episode return must equal
   exactly `±terminal_payoff` for any deterministic-action
   trajectory — covered.
4. **Horizon override.** `make_game("rules_of_road_sparse",
   adversary=..., horizon=H)` for `H != 20` must respect the
   override (the wrapper's `horizon=20` default is overridable).
   Suggested follow-up unit test.
5. **Adversary still observes payoffs.** The sparse mask only
   affects the agent-visible reward; `adversary.observe(...)` still
   receives the underlying matrix payoff. This is intentional (so
   inertia / regret / FP adversaries that track payoffs don't see
   a corrupted signal). Worth a focused test if the M6 sweep
   surfaces unexpected adversary dynamics on RR-Sparse.

## Follow-ups for next dispatch (NOT in this scope)

- **Update `test_registry.py`** to include `"rules_of_road_sparse"`
  in `EXPECTED_GAMES` and bump the count from 7 to 8. One-line fix;
  blocked here only by the allow-list.
- Wire `RR-Sparse × StationaryConvention` (and `× ConventionSwitch`
  for stress) into the M6 sweep configs per patch §1.5. Out of M2
  reopen scope.
- Consider exposing `c` / `m` as overridable parameters on the
  `rules_of_road_sparse` registry wrapper if YAML configs need to
  tune them per the patch §1.2 "tuneable in yaml" note. The
  underlying `build` already accepts both kwargs.

## No-edit confirmation

No edits to:
- `src/`, `mushroom-rl-dev/`
- `experiments/adaptive_beta/strategic_games/agents.py`,
  `schedules.py`, `matrix_game.py`, `history.py`, `base.py`
- `experiments/adaptive_beta/tab_six_games/`
- Any `experiments/adaptive_beta/strategic_games/games/*.py` other
  than `rules_of_road.py`
- Any `experiments/adaptive_beta/strategic_games/adversaries/*.py`
- Any test file other than the new
  `tests/adaptive_beta/strategic_games/test_rules_of_road_sparse.py`

No `expm1` / `log1p` introduced; no operator math touched (lessons
#11 N/A here).

No `pip install` invoked.
