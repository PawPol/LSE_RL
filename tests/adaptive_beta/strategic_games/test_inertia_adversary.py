"""Tests for the Phase VIII M3 ``inertia`` strategic adversary.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §M3 and
§5.7 (sticky-action adversary for the AC-Inertia subcase).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.adaptive_beta.strategic_games import ADVERSARY_REGISTRY
from experiments.adaptive_beta.strategic_games.adversaries.base import (
    ADVERSARY_INFO_KEYS,
)
from experiments.adaptive_beta.strategic_games.adversaries.inertia import (
    InertiaOpponent,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


def _action_stream(
    *,
    n_actions: int = 3,
    inertia_lambda: float = 0.5,
    seed: int = 0,
    n_steps: int = 100,
) -> list[int]:
    """Drive an inertia adversary and record its realised action stream."""
    adv = InertiaOpponent(
        n_actions=n_actions,
        inertia_lambda=inertia_lambda,
        seed=seed,
    )
    history = GameHistory()
    actions: list[int] = []
    for _ in range(n_steps):
        action = adv.act(history)
        actions.append(action)
        history.append(
            agent_action=0,
            opponent_action=action,
            agent_reward=0.0,
            opponent_reward=0.0,
            info=adv.info(),
        )
    return actions


def test_register_in_registry() -> None:
    """`spec §5.7` — the sticky-action adversary is registry-addressable."""
    assert "inertia" in ADVERSARY_REGISTRY
    assert ADVERSARY_REGISTRY["inertia"] is InertiaOpponent


def test_action_distribution_under_high_inertia() -> None:
    """`spec §5.7` — high inertia makes the sequence stick to one action."""
    actions = _action_stream(inertia_lambda=0.95, seed=8, n_steps=100)
    counts = np.bincount(actions, minlength=3)
    assert int(counts.max()) > 80


def test_action_distribution_under_zero_inertia() -> None:
    """`spec §5.7` — zero inertia reduces to uniform random actions."""
    n_actions = 4
    n_steps = 2_000
    actions = _action_stream(
        n_actions=n_actions,
        inertia_lambda=0.0,
        seed=2026,
        n_steps=n_steps,
    )
    counts = np.bincount(actions, minlength=n_actions)
    expected = n_steps / n_actions
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    assert counts.min() / counts.max() > 0.90
    assert chi_square < 16.27  # df=3, p=0.001 critical value.


@pytest.mark.parametrize("bad_lambda", [-0.01, 1.01, math.nan, math.inf])
def test_lambda_validation(bad_lambda: float) -> None:
    """`spec §5.7` — inertia_lambda must lie in the closed interval [0, 1]."""
    with pytest.raises(ValueError, match="inertia_lambda"):
        InertiaOpponent(n_actions=3, inertia_lambda=bad_lambda, seed=0)


def test_determinism() -> None:
    """`spec §5.2 / §5.7` — identical seeds give identical action streams."""
    actions_a = _action_stream(inertia_lambda=0.7, seed=1234, n_steps=200)
    actions_b = _action_stream(inertia_lambda=0.7, seed=1234, n_steps=200)
    assert actions_a == actions_b


def test_act_returns_int() -> None:
    """`spec §5.2` — act returns a plain Python integer action."""
    adv = InertiaOpponent(n_actions=3, inertia_lambda=0.5, seed=0)
    action = adv.act(GameHistory())
    assert type(action) is int
    assert 0 <= action < adv.n_actions


def test_info_keys_present() -> None:
    """`spec §5.2 / §5.7` — info exposes mandatory and inertia fields."""
    adv = InertiaOpponent(n_actions=3, inertia_lambda=0.25, seed=7)
    before = adv.info()
    assert ADVERSARY_INFO_KEYS <= set(before)
    assert before["phase"] == "inertia"
    assert before["inertia_lambda"] == pytest.approx(0.25)
    assert before["last_action"] is None

    action = adv.act(GameHistory())
    after = adv.info()
    assert ADVERSARY_INFO_KEYS <= set(after)
    assert after["phase"] == "inertia"
    assert after["inertia_lambda"] == pytest.approx(0.25)
    assert after["last_action"] == action


def test_reset_round_trip() -> None:
    """`spec §5.2` — reset(seed=42) reproduces construct(seed=42)."""
    fresh = _action_stream(inertia_lambda=0.6, seed=42, n_steps=100)

    adv = InertiaOpponent(n_actions=3, inertia_lambda=0.6, seed=999)
    contaminated_history = GameHistory()
    for _ in range(25):
        action = adv.act(contaminated_history)
        contaminated_history.append(0, action, 0.0, 0.0, adv.info())

    adv.reset(seed=42)
    reset_history = GameHistory()
    reset_actions: list[int] = []
    for _ in range(100):
        action = adv.act(reset_history)
        reset_actions.append(action)
        reset_history.append(0, action, 0.0, 0.0, adv.info())

    assert reset_actions == fresh
