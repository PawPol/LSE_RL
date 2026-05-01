"""Tests for the Phase VIII §5.7 passive opponent.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §5.7
(``<!-- patch-2026-05-01 §11 -->``); upstream patch
``tasks/phase_VIII_spec_patches_2026-05-01.md`` §11.4.

Invariants guarded
------------------
- ``act`` always returns 0, regardless of history / agent_action / seed.
- ``info`` returns the full :data:`ADVERSARY_INFO_KEYS` set with the
  patch §11.4 contract values.
- ``reset(seed=...)`` is a no-op (no RNG state to seed); identical
  call sequences produce identical action streams independent of seed.
- ``"passive"`` is registered in :data:`ADVERSARY_REGISTRY`.
"""

from __future__ import annotations

import pytest

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    ADVERSARY_INFO_KEYS,
)
from experiments.adaptive_beta.strategic_games.adversaries.passive import (
    PassiveOpponent,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory
from experiments.adaptive_beta.strategic_games.registry import (
    ADVERSARY_REGISTRY,
    make_adversary,
)


# Required (key, value) pairs for the passive opponent's info() per
# patch §11.4. Keys outside this set must still be present (see
# ADVERSARY_INFO_KEYS) but the values listed here are the exact
# patch-mandated literals.
EXPECTED_INFO: dict = {
    "adversary_type":  "passive",
    "phase":           "stationary",
    "memory_m":        0,
    "inertia_lambda":  0.0,
    "temperature":     0.0,
    "model_rejected":  False,
    "search_phase":    "none",
    "hypothesis_id":   None,
    "policy_entropy":  0.0,
}


# ---------------------------------------------------------------------------
# 1. test_passive_opponent_no_op
# ---------------------------------------------------------------------------
def test_passive_opponent_no_op() -> None:
    """``act`` always returns ``0`` independent of history / agent_action."""
    opp = PassiveOpponent(n_actions=1)
    empty = GameHistory()

    # Empty history, no agent_action.
    assert opp.act(empty) == 0
    # Repeated calls with same history.
    for _ in range(5):
        assert opp.act(empty) == 0
    # With an agent_action passed explicitly.
    assert opp.act(empty, agent_action=0) == 0

    # With a non-empty history, action should still be 0.
    h = GameHistory()
    for t in range(3):
        h.append(
            agent_action=t % 2,
            opponent_action=0,
            agent_reward=0.1 * t,
            opponent_reward=-0.1 * t,
            info={"phase": "stationary"},
        )
    assert opp.act(h) == 0

    # ``observe`` is a no-op — should not raise nor change subsequent acts.
    opp.observe(
        agent_action=0,
        opponent_action=0,
        agent_reward=1.0,
        opponent_reward=-1.0,
        info={"phase": "stationary"},
    )
    assert opp.act(h) == 0


# ---------------------------------------------------------------------------
# 2. test_passive_opponent_info_contract
# ---------------------------------------------------------------------------
def test_passive_opponent_info_contract() -> None:
    """``info()`` returns the full required key set with patch values."""
    opp = PassiveOpponent(n_actions=1)
    info = opp.info()

    # Mandatory keys present (validated also inside info() via _validate_info).
    missing = ADVERSARY_INFO_KEYS - set(info.keys())
    assert not missing, f"missing required info keys: {sorted(missing)}"

    # Patch §11.4 literals.
    for k, v in EXPECTED_INFO.items():
        assert k in info, f"info missing key {k!r}"
        assert info[k] == v, (
            f"info[{k!r}] = {info[k]!r}, expected {v!r}"
        )

    # ``adversary_type`` class attribute matches the info field.
    assert opp.adversary_type == "passive"
    assert PassiveOpponent.adversary_type == "passive"


# ---------------------------------------------------------------------------
# 3. test_passive_opponent_seed_invariance
# ---------------------------------------------------------------------------
def test_passive_opponent_seed_invariance() -> None:
    """No RNG state to seed: identical action streams independent of seed.

    Construct two opponents with different seeds, drive each through the
    same history sequence, and assert their action streams agree.
    Also exercise ``reset(seed=...)`` with a fresh seed and confirm that
    too is a no-op (does not raise, action stream unchanged).
    """
    opp_a = PassiveOpponent(n_actions=1, seed=0)
    opp_b = PassiveOpponent(n_actions=1, seed=999)

    h_a = GameHistory()
    h_b = GameHistory()

    actions_a, actions_b = [], []
    for t in range(20):
        a_a = opp_a.act(h_a)
        a_b = opp_b.act(h_b)
        actions_a.append(a_a)
        actions_b.append(a_b)
        h_a.append(
            agent_action=t % 1,
            opponent_action=a_a,
            agent_reward=0.0,
            opponent_reward=0.0,
            info={},
        )
        h_b.append(
            agent_action=t % 1,
            opponent_action=a_b,
            agent_reward=0.0,
            opponent_reward=0.0,
            info={},
        )
        opp_a.observe(t % 1, a_a, 0.0, 0.0, info={})
        opp_b.observe(t % 1, a_b, 0.0, 0.0, info={})

    assert actions_a == [0] * 20
    assert actions_a == actions_b

    # ``reset`` with a different seed must still leave behaviour unchanged.
    opp_a.reset(seed=42)
    opp_b.reset(seed=None)
    assert opp_a.act(GameHistory()) == 0
    assert opp_b.act(GameHistory()) == 0


# ---------------------------------------------------------------------------
# 4. test_passive_opponent_in_registry
# ---------------------------------------------------------------------------
def test_passive_opponent_in_registry() -> None:
    """``"passive"`` resolves to :class:`PassiveOpponent` via the registry."""
    assert "passive" in ADVERSARY_REGISTRY, (
        f"ADVERSARY_REGISTRY must contain 'passive'; current keys: "
        f"{sorted(ADVERSARY_REGISTRY.keys())}"
    )
    factory = ADVERSARY_REGISTRY["passive"]
    assert factory is PassiveOpponent

    # ``make_adversary("passive", n_actions=1)`` builds a working instance.
    opp = make_adversary("passive", n_actions=1)
    assert isinstance(opp, PassiveOpponent)
    assert opp.act(GameHistory()) == 0


# ---------------------------------------------------------------------------
# 5. (extra) Constructor rejects n_actions != 1.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_n", [0, 2, 5])
def test_passive_opponent_rejects_multi_action(bad_n: int) -> None:
    """``PassiveOpponent`` is only well-defined for ``n_actions == 1``."""
    with pytest.raises(ValueError):
        PassiveOpponent(n_actions=bad_n)
