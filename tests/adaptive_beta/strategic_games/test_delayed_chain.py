"""Tests for the Phase VIII §5.7 delayed-reward chain game.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §5.7
(``<!-- patch-2026-05-01 §11 -->``); upstream patch
``tasks/phase_VIII_spec_patches_2026-05-01.md`` §11.4.

Invariants guarded
------------------
- All four DC subcases register & build through the registry.
- Advance-only chains reach the goal terminal in exactly L steps.
- Reward is 0 at every non-terminal step on advance-only chains.
- DC-Branching20 delivers -1 on arrival at the trap terminal.
- Horizon equals chain length L on advance-only subcases.
- ``game_info()['canonical_sign'] == "+"`` on every subcase.
- Observation space cardinality matches ``L+1`` (advance-only) or
  ``L+1+5`` (DC-Branching20).
- Identical ``(seed, subcase)`` produce byte-identical trajectories.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from experiments.adaptive_beta.strategic_games import make_game
from experiments.adaptive_beta.strategic_games.adversaries.passive import (
    PassiveOpponent,
)
from experiments.adaptive_beta.strategic_games.games.delayed_chain import (
    ALL_SUBCASES,
    SUBCASE_BRANCHING20,
    SUBCASE_LONG50,
    SUBCASE_MEDIUM20,
    SUBCASE_SHORT10,
    TRAP_CHAIN_LEN,
    build,
)
from experiments.adaptive_beta.strategic_games.registry import GAME_REGISTRY


ADVANCE_ONLY_SUBCASES: Tuple[str, ...] = (
    SUBCASE_SHORT10,
    SUBCASE_MEDIUM20,
    SUBCASE_LONG50,
)
SUBCASE_TO_L = {
    SUBCASE_SHORT10:     10,
    SUBCASE_MEDIUM20:    20,
    SUBCASE_LONG50:      50,
    SUBCASE_BRANCHING20: 20,
}


def _make_env(subcase: str, *, seed: int = 0):
    """Build a delayed_chain env with a 1-action passive opponent."""
    adv = PassiveOpponent(n_actions=1)
    return build(subcase=subcase, adversary=adv, seed=seed)


def _rollout(env, actions: List[int]) -> List[Tuple[int, float, bool]]:
    """Drive ``env`` with the given action stream; return per-step trace.

    Returns ``[(next_chain_state, reward, absorbing), ...]``.
    """
    state, _ = env.reset()
    del state
    trace: List[Tuple[int, float, bool]] = []
    for a in actions:
        s, r, done, info = env.step(np.asarray([a], dtype=np.int64))
        trace.append(
            (int(np.asarray(s).flat[0]), float(r), bool(done))
        )
        if done:
            break
    return trace


# ---------------------------------------------------------------------------
# 1. test_chain_advance_reaches_terminal_at_L
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subcase", ADVANCE_ONLY_SUBCASES)
def test_chain_advance_reaches_terminal_at_L(subcase: str) -> None:
    """Pure-advance rollout reaches state ``L`` after exactly ``L`` steps."""
    L = SUBCASE_TO_L[subcase]
    env = _make_env(subcase, seed=0)
    trace = _rollout(env, [0] * L)

    assert len(trace) == L, (
        f"{subcase}: expected exactly L={L} steps, got {len(trace)}"
    )
    states = [s for s, _, _ in trace]
    rewards = [r for _, r, _ in trace]
    dones = [d for _, _, d in trace]
    assert states == list(range(1, L + 1)), (
        f"{subcase}: deterministic chain trace mismatch; got {states}"
    )
    # Only the last transition delivers +1; only the last is absorbing.
    assert rewards[-1] == pytest.approx(1.0)
    assert all(r == pytest.approx(0.0) for r in rewards[:-1])
    assert dones[-1] is True
    assert not any(dones[:-1])


# ---------------------------------------------------------------------------
# 1b. Branching subcase reaches goal under all-advance.
# ---------------------------------------------------------------------------
def test_branching_advance_only_reaches_goal() -> None:
    """DC-Branching20: action 0 ('advance') across the chain hits the goal."""
    L = SUBCASE_TO_L[SUBCASE_BRANCHING20]
    env = _make_env(SUBCASE_BRANCHING20, seed=0)
    trace = _rollout(env, [0] * L)

    assert len(trace) == L
    final_state, final_reward, final_done = trace[-1]
    assert final_state == L
    assert final_reward == pytest.approx(1.0)
    assert final_done is True


# ---------------------------------------------------------------------------
# 2. test_chain_no_intermediate_reward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subcase", ADVANCE_ONLY_SUBCASES)
def test_chain_no_intermediate_reward(subcase: str) -> None:
    """Advance-only subcases: r_t = 0 for every t < H-1.

    Patch §11.2: "0 at all non-terminal transitions; +1 on advance-action
    arrival at goal terminal".
    """
    L = SUBCASE_TO_L[subcase]
    env = _make_env(subcase, seed=42)
    trace = _rollout(env, [0] * L)

    rewards = [r for _, r, _ in trace]
    # First (L-1) rewards must be exactly 0.
    for t, r in enumerate(rewards[:-1]):
        assert r == 0.0, (
            f"{subcase}: r_{t} = {r}; expected 0 at every non-terminal step"
        )


# ---------------------------------------------------------------------------
# 3. test_branching_wrong_terminal_negative
# ---------------------------------------------------------------------------
def test_branching_wrong_terminal_negative() -> None:
    """DC-Branching20: branch_wrong + ``TRAP_CHAIN_LEN`` advances → -1 reward.

    Action 1 from the start jumps to the trap-chain entry (state L+1);
    subsequent advances roll forward until arrival at the trap terminal
    (state L+TRAP_CHAIN_LEN), which delivers -1.
    """
    L = SUBCASE_TO_L[SUBCASE_BRANCHING20]
    env = _make_env(SUBCASE_BRANCHING20, seed=0)

    # Step 1: branch_wrong → state L+1, reward 0.
    # Steps 2..TRAP_CHAIN_LEN: advance through trap chain → trap terminal.
    actions = [1] + [0] * (TRAP_CHAIN_LEN - 1)
    trace = _rollout(env, actions)

    assert len(trace) == TRAP_CHAIN_LEN, (
        f"expected exactly TRAP_CHAIN_LEN={TRAP_CHAIN_LEN} steps, "
        f"got {len(trace)}"
    )

    # Step 1 lands at trap entry, reward 0, not absorbing.
    s0, r0, done0 = trace[0]
    assert s0 == L + 1
    assert r0 == pytest.approx(0.0)
    assert done0 is False

    # Trap-chain interior steps: reward 0, not absorbing (until terminal).
    for t in range(1, TRAP_CHAIN_LEN - 1):
        s, r, done = trace[t]
        assert s == L + 1 + t, f"trap-chain state mismatch at step {t}"
        assert r == pytest.approx(0.0)
        assert done is False

    # Final step: arrival at trap terminal, reward -1, absorbing.
    s_last, r_last, done_last = trace[-1]
    assert s_last == L + TRAP_CHAIN_LEN
    assert r_last == pytest.approx(-1.0)
    assert done_last is True


# ---------------------------------------------------------------------------
# 4. test_horizon_matches_subcase
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subcase", ADVANCE_ONLY_SUBCASES)
def test_horizon_matches_subcase(subcase: str) -> None:
    """Horizon ``H`` equals chain length ``L`` on advance-only subcases."""
    L = SUBCASE_TO_L[subcase]
    env = _make_env(subcase, seed=0)
    info = env.game_info()
    assert info["horizon"] == L
    # MDPInfo horizon should match too.
    assert env.info.horizon == L


# ---------------------------------------------------------------------------
# 5. test_canonical_sign_metadata
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subcase", ALL_SUBCASES)
def test_canonical_sign_metadata(subcase: str) -> None:
    """Every subcase declares ``canonical_sign == "+"`` (patch §11.2)."""
    env = _make_env(subcase, seed=0)
    info = env.game_info()
    assert info["canonical_sign"] == "+", (
        f"{subcase}: canonical_sign should be '+', got {info['canonical_sign']!r}"
    )
    # The env-level attribute used by schedule selectors must agree.
    assert env.env_canonical_sign == "+"


# ---------------------------------------------------------------------------
# 6. test_state_encoder_shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subcase", ALL_SUBCASES)
def test_state_encoder_shape(subcase: str) -> None:
    """Discrete observation space matches ``L+1`` (or ``L+1+5`` for branching).

    State returned by ``reset`` and ``step`` must be a shape-(1,) int64
    array whose value lies in [0, n_states).
    """
    L = SUBCASE_TO_L[subcase]
    expected_n_states = L + 1 + (
        TRAP_CHAIN_LEN if subcase == SUBCASE_BRANCHING20 else 0
    )
    env = _make_env(subcase, seed=0)
    assert env.info.observation_space.size == (expected_n_states,), (
        f"{subcase}: observation_space.size = "
        f"{env.info.observation_space.size}, expected ({expected_n_states},)"
    )

    # Reset state is shape-(1,) int64, value 0.
    state, _ = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (1,)
    assert state.dtype == np.int64
    assert int(np.asarray(state).flat[0]) == 0

    # First step state is shape-(1,) int64, value 1.
    s_next, _, _, _ = env.step(np.asarray([0], dtype=np.int64))
    assert isinstance(s_next, np.ndarray)
    assert s_next.shape == (1,)
    assert s_next.dtype == np.int64
    assert int(np.asarray(s_next).flat[0]) == 1

    # Action-space cardinality.
    expected_n_a = 2 if subcase == SUBCASE_BRANCHING20 else 1
    assert env.info.action_space.size == (expected_n_a,)


# ---------------------------------------------------------------------------
# 7. test_seed_determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subcase", ALL_SUBCASES)
def test_seed_determinism(subcase: str) -> None:
    """Two envs with identical seed produce byte-identical trajectories.

    The chain dynamics are deterministic by construction; this test
    nevertheless guards against future drift (e.g., a refactor that
    introduces RNG into the transition).
    """
    L = SUBCASE_TO_L[subcase]
    # Use a "branch_wrong then advance" stream to exercise both branches
    # for DC-Branching20; advance-only chains ignore the action 1 and
    # would error, so use [0]*L for them.
    if subcase == SUBCASE_BRANCHING20:
        actions = [1] + [0] * (TRAP_CHAIN_LEN - 1)
    else:
        actions = [0] * L

    env_a = _make_env(subcase, seed=12345)
    env_b = _make_env(subcase, seed=12345)
    trace_a = _rollout(env_a, actions)
    trace_b = _rollout(env_b, actions)
    assert trace_a == trace_b, (
        f"{subcase}: same seed produced different traces"
    )


# ---------------------------------------------------------------------------
# 8. test_all_4_subcases_register
# ---------------------------------------------------------------------------
def test_all_4_subcases_register() -> None:
    """``delayed_chain`` registers a single factory; all 4 subcases build cleanly."""
    assert "delayed_chain" in GAME_REGISTRY, (
        f"GAME_REGISTRY must contain 'delayed_chain'; current keys: "
        f"{sorted(GAME_REGISTRY.keys())}"
    )
    for sub in ALL_SUBCASES:
        env = make_game(
            "delayed_chain",
            subcase=sub,
            adversary=PassiveOpponent(n_actions=1),
            seed=7,
        )
        info = env.game_info()
        assert info["subcase"] == sub
        assert info["canonical_sign"] == "+"
        # Confirm initial reset / step run without errors.
        state, _ = env.reset()
        assert state.shape == (1,)
        s_next, _, _, _ = env.step(np.asarray([0], dtype=np.int64))
        assert s_next.shape == (1,)


# ---------------------------------------------------------------------------
# 9. (extra) regime is None on every subcase (spec §5.7).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subcase", ALL_SUBCASES)
def test_regime_is_none(subcase: str) -> None:
    """Patch §11.2 / spec §5.7: ``info['regime'] is None`` (regime-stationary)."""
    env = _make_env(subcase, seed=0)
    _, reset_info = env.reset()
    assert reset_info["regime"] is None
    _, _, _, step_info = env.step(np.asarray([0], dtype=np.int64))
    assert step_info["regime"] is None
