"""Tests for the Phase VIII M3 sign-switching-regime adversary."""

from __future__ import annotations

import math
from typing import Iterable, List

import pytest

from experiments.adaptive_beta.strategic_games.adversaries.sign_switching_regime import (
    SignSwitchingRegimeOpponent,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory
from experiments.adaptive_beta.strategic_games.registry import ADVERSARY_REGISTRY


def _episode_regimes(
    adversary: SignSwitchingRegimeOpponent,
    *,
    episodes: int,
) -> List[str]:
    """Regime visible during each episode before the boundary clock advances."""
    regimes: List[str] = []
    for _ in range(episodes):
        regimes.append(adversary.info()["regime"])
        adversary.on_episode_end()
    return regimes


def _drive_reward_episodes(
    adversary: SignSwitchingRegimeOpponent,
    rewards: Iterable[float],
) -> List[str]:
    regimes: List[str] = []
    for reward in rewards:
        regimes.append(adversary.info()["regime"])
        adversary.observe(
            agent_action=0,
            opponent_action=0,
            agent_reward=float(reward),
            opponent_reward=-float(reward),
        )
        adversary.on_episode_end()
    regimes.append(adversary.info()["regime"])
    return regimes


def test_register_in_registry() -> None:
    assert "sign_switching_regime" in ADVERSARY_REGISTRY


def test_exogenous_dwell_correctness() -> None:
    adversary = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="exogenous",
        dwell_episodes=3,
        initial_regime="plus",
        seed=0,
    )

    assert _episode_regimes(adversary, episodes=10) == [
        "plus",
        "plus",
        "plus",
        "minus",
        "minus",
        "minus",
        "plus",
        "plus",
        "plus",
        "minus",
    ]


def test_exogenous_dwell_grid() -> None:
    for dwell in (100, 250, 500, 1000):
        adversary = SignSwitchingRegimeOpponent(
            n_actions=2,
            mode="exogenous",
            dwell_episodes=dwell,
            initial_regime="plus",
            seed=0,
        )
        trajectory = _episode_regimes(adversary, episodes=dwell + 1)

        assert trajectory[:dwell] == ["plus"] * dwell
        assert trajectory[dwell] == "minus"


def test_endogenous_trigger_high() -> None:
    adversary = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="endogenous",
        trigger_window=3,
        trigger_threshold_high=0.5,
        trigger_threshold_low=-0.5,
        initial_regime="plus",
        seed=0,
    )

    regimes = _drive_reward_episodes(adversary, [1.0, 1.0, 1.0])

    assert regimes[-1] == "minus"
    assert adversary.info()["switches_so_far"] == 1


def test_endogenous_trigger_low() -> None:
    adversary = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="endogenous",
        trigger_window=3,
        trigger_threshold_high=0.5,
        trigger_threshold_low=-0.5,
        initial_regime="minus",
        seed=0,
    )

    regimes = _drive_reward_episodes(adversary, [-1.0, -1.0, -1.0])

    assert regimes[-1] == "plus"
    assert adversary.info()["switches_so_far"] == 1


def test_endogenous_bidirectional() -> None:
    adversary = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="endogenous",
        trigger_window=3,
        trigger_threshold_high=0.5,
        trigger_threshold_low=-0.5,
        initial_regime="plus",
        seed=0,
    )

    high_regimes = _drive_reward_episodes(adversary, [1.0, 1.0, 1.0])
    assert high_regimes[-1] == "minus"

    low_regimes = _drive_reward_episodes(adversary, [-1.0, -1.0, -1.0])
    assert low_regimes[-1] == "plus"
    assert adversary.info()["switches_so_far"] == 2


def test_info_regime_in_set() -> None:
    adversary = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="exogenous",
        dwell_episodes=2,
        initial_regime="plus",
        seed=0,
    )

    for regime in _episode_regimes(adversary, episodes=8):
        assert regime in {"plus", "minus"}


def test_info_keys_schema() -> None:
    adversary = SignSwitchingRegimeOpponent(n_actions=2, seed=0)

    assert {
        "regime",
        "mode",
        "switches_so_far",
        "episodes_in_current_regime",
    }.issubset(adversary.info().keys())


def test_determinism() -> None:
    exogenous_a = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="exogenous",
        dwell_episodes=3,
        seed=2026,
    )
    exogenous_b = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="exogenous",
        dwell_episodes=3,
        seed=2026,
    )
    assert _episode_regimes(exogenous_a, episodes=12) == _episode_regimes(
        exogenous_b,
        episodes=12,
    )

    reward_script = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0] * 2
    endogenous_a = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="endogenous",
        trigger_window=3,
        trigger_threshold_high=0.5,
        trigger_threshold_low=-0.5,
        seed=2026,
    )
    endogenous_b = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="endogenous",
        trigger_window=3,
        trigger_threshold_high=0.5,
        trigger_threshold_low=-0.5,
        seed=2026,
    )
    assert _drive_reward_episodes(endogenous_a, reward_script) == (
        _drive_reward_episodes(endogenous_b, reward_script)
    )


def test_per_regime_callable_dispatch() -> None:
    calls = {"plus": 0, "minus": 0}

    def policy_plus_actions(history: GameHistory) -> int:
        calls["plus"] += 1
        assert isinstance(history, GameHistory)
        return 0

    def policy_minus_actions(history: GameHistory) -> int:
        calls["minus"] += 1
        assert isinstance(history, GameHistory)
        return 1

    adversary = SignSwitchingRegimeOpponent(
        n_actions=2,
        mode="exogenous",
        dwell_episodes=2,
        initial_regime="plus",
        policy_plus_actions=policy_plus_actions,
        policy_minus_actions=policy_minus_actions,
        seed=0,
    )

    history = GameHistory()
    actions: List[int] = []
    regimes: List[str] = []
    for _ in range(6):
        regimes.append(adversary.info()["regime"])
        action = adversary.act(history)
        actions.append(action)
        history.append(
            agent_action=0,
            opponent_action=action,
            agent_reward=0.0,
            opponent_reward=0.0,
            info=adversary.info(),
        )
        adversary.on_episode_end()

    assert regimes == ["plus", "plus", "minus", "minus", "plus", "plus"]
    assert actions == [0, 0, 1, 1, 0, 0]
    assert calls == {"plus": 4, "minus": 2}


def test_initial_regime_plus_or_minus() -> None:
    for initial_regime in ("plus", "minus"):
        adversary = SignSwitchingRegimeOpponent(
            n_actions=2,
            initial_regime=initial_regime,
            seed=0,
        )

        assert adversary.regime == initial_regime
        assert adversary.info()["regime"] == initial_regime


def test_constructor_validation() -> None:
    invalid_kwargs = (
        {"mode": "invalid"},
        {"dwell_episodes": -1},
        {"trigger_threshold_high": math.nan},
        {"trigger_threshold_low": math.nan},
    )

    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            SignSwitchingRegimeOpponent(n_actions=2, **kwargs)
