"""Tests for the Phase VIII M2 Soda / Uncertain game.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` sec. 5.5.

Invariants guarded
------------------
- All five Soda subcases build through the game registry.
- The four stationary subcases install the documented payoff matrices.
- ``SO-TypeSwitch`` exposes deterministic regime trajectories and period
  cadence through ``info["regime"]`` without exposing the regime in state.
- Soda has no canonical sign, so wrong-sign style schedules remain undefined.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.strategic_games import make_game
from experiments.adaptive_beta.strategic_games.adversaries.stationary import (
    StationaryMixedOpponent,
)
from experiments.adaptive_beta.strategic_games.games import soda_uncertain


SODA_SUBCASES = (
    "SO-Coordination",
    "SO-AntiCoordination",
    "SO-ZeroSum",
    "SO-BiasedPreference",
    "SO-TypeSwitch",
)
STATIONARY_SUBCASES = SODA_SUBCASES[:4]
DOCUMENTED_REGIMES = {
    "coordination",
    "anti_coordination",
    "zero_sum",
    "biased_preference",
}


def _stationary_adversary(m: int, action: int = 0) -> StationaryMixedOpponent:
    probs = np.zeros(m, dtype=np.float64)
    probs[action] = 1.0
    return StationaryMixedOpponent(probs=probs, seed=0)


def _build_soda(subcase: str, *, horizon: int = 3, m: int = 3, **kwargs):
    return soda_uncertain.build(
        subcase=subcase,
        adversary=_stationary_adversary(m),
        horizon=horizon,
        m=m,
        seed=123,
        **kwargs,
    )


def _expected_payoffs(subcase: str, m: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if subcase == "SO-Coordination":
        p = np.eye(m, dtype=np.float64)
        return p, p.copy()

    if subcase == "SO-AntiCoordination":
        p = np.ones((m, m), dtype=np.float64) - np.eye(m, dtype=np.float64)
        return p, p.copy()

    if subcase == "SO-ZeroSum":
        pa = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                diff = (i - j) % m
                if diff == 1:
                    pa[i, j] = 1.0
                elif diff == m - 1:
                    pa[i, j] = -1.0
        return pa, -pa

    if subcase == "SO-BiasedPreference":
        p = np.zeros((m, m), dtype=np.float64)
        p[0, 0] = 2.0
        for i in range(1, m):
            p[i, i] = 1.0
        return p, p.copy()

    raise ValueError(f"unexpected stationary Soda subcase: {subcase}")


def _regime_trajectory(env, episodes: int) -> list[str]:
    regimes: list[str] = []
    for _ in range(episodes):
        _, reset_info = env.reset()
        regimes.append(str(reset_info["regime"]))
        done = False
        while not done:
            _, _, done, step_info = env.step(0)
            assert step_info["regime"] == reset_info["regime"]
    return regimes


@pytest.mark.parametrize("subcase", SODA_SUBCASES)
def test_soda_5_subcases_register(subcase: str) -> None:
    """Phase VIII M2 spec sec. 5.5: each Soda subcase builds via registry."""
    env = make_game(
        "soda_uncertain",
        subcase=subcase,
        adversary=_stationary_adversary(3),
        horizon=3,
        m=3,
        seed=7,
    )
    state, info = env.reset()
    assert state.shape == (1,)
    assert info["regime"] in DOCUMENTED_REGIMES
    assert env.game_info()["subcase"] == subcase


@pytest.mark.parametrize("subcase", STATIONARY_SUBCASES)
def test_payoff_correctness_per_type(subcase: str) -> None:
    """Stationary Soda subcases install the documented payoff matrices."""
    env = _build_soda(subcase, horizon=4, m=3)
    expected_agent, expected_opponent = _expected_payoffs(subcase, m=3)
    np.testing.assert_array_equal(env._payoff_agent, expected_agent)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(env._payoff_opponent, expected_opponent)  # type: ignore[attr-defined]


def test_hidden_type_sampling_determinism() -> None:
    """Same seed yields the same SO-TypeSwitch regime trajectory."""
    env1 = _build_soda(
        "SO-TypeSwitch", horizon=1, m=3, switch_period_episodes=3
    )
    env2 = _build_soda(
        "SO-TypeSwitch", horizon=1, m=3, switch_period_episodes=3
    )
    assert _regime_trajectory(env1, episodes=50) == _regime_trajectory(
        env2, episodes=50
    )


def test_state_encoder_shape() -> None:
    """Default Soda state is a discrete scalar with H * (m + 1) states."""
    horizon = 5
    m = 3
    env = _build_soda("SO-Coordination", horizon=horizon, m=m)
    state, _ = env.reset()
    assert state.shape == (1,)
    assert state.dtype == np.int64
    assert int(env.info.observation_space.size[0]) == horizon * (m + 1)
    assert int(env.info.action_space.size[0]) == m
    assert 0 <= int(state.flat[0]) < horizon * (m + 1)


def test_horizon_termination() -> None:
    """Soda episodes terminate exactly at the configured horizon."""
    env = _build_soda("SO-Coordination", horizon=3, m=3)
    env.reset()
    absorbing_flags = [env.step(0)[2] for _ in range(3)]
    assert absorbing_flags == [False, False, True]


@pytest.mark.parametrize("subcase", SODA_SUBCASES)
def test_info_regime_schema(subcase: str) -> None:
    """``info["regime"]`` is one of the documented Soda regime strings."""
    env = _build_soda(subcase, horizon=2, m=3, switch_period_episodes=2)
    _, reset_info = env.reset()
    assert reset_info["regime"] in DOCUMENTED_REGIMES
    _, _, _, step_info = env.step(0)
    assert step_info["regime"] in DOCUMENTED_REGIMES
    if subcase in STATIONARY_SUBCASES:
        assert step_info["regime"] == reset_info["regime"]


@pytest.mark.parametrize("subcase", SODA_SUBCASES)
def test_canonical_sign_none(subcase: str) -> None:
    """Soda has no single canonical sign because the payoff type can vary."""
    env = _build_soda(subcase, horizon=2, m=3)
    assert env.env_canonical_sign is None
    assert env.game_info()["env_canonical_sign"] is None
    assert env.game_info()["canonical_sign"] is None


def test_type_switch_period() -> None:
    """``SO-TypeSwitch`` changes regime every ``switch_period_episodes`` episodes."""
    env = _build_soda(
        "SO-TypeSwitch", horizon=1, m=3, switch_period_episodes=2
    )
    expected = [
        "coordination",
        "coordination",
        "anti_coordination",
        "anti_coordination",
        "zero_sum",
        "zero_sum",
        "biased_preference",
        "biased_preference",
        "coordination",
        "coordination",
    ]
    assert _regime_trajectory(env, episodes=len(expected)) == expected
