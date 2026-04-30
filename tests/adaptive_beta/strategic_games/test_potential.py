"""Tests for Phase VIII M2 potential / weakly-acyclic strategic games.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` section 5.6.
"""

from __future__ import annotations

from itertools import product
from typing import Any, List, Tuple

import numpy as np
import pytest

from experiments.adaptive_beta.strategic_games import GAME_REGISTRY, make_game
from experiments.adaptive_beta.strategic_games.adversaries.stationary import (
    StationaryMixedOpponent,
)
from experiments.adaptive_beta.strategic_games.games import potential


M = 3
SUBCASES = potential.ALL_SUBCASES


def _stationary_action(action: int, *, m: int = M, seed: int = 0) -> StationaryMixedOpponent:
    probs = np.zeros(m, dtype=np.float64)
    probs[int(action)] = 1.0
    return StationaryMixedOpponent(probs=probs, seed=seed)


def _build_env(
    subcase: str,
    *,
    opponent_action: int = 0,
    horizon: int = 5,
    seed: int = 0,
    **kwargs: Any,
) -> potential.PotentialGameEnv:
    return potential.build(
        subcase=subcase,
        adversary=_stationary_action(opponent_action, seed=seed),
        horizon=horizon,
        m=M,
        seed=seed,
        **kwargs,
    )


def _phi(subcase: str, agent_action: int, opponent_action: int) -> float:
    scale = 2.0 if subcase == potential.SUBCASE_SWITCHING else 1.0
    return potential.compute_potential(
        (agent_action, opponent_action),
        subcase=subcase,
        m=M,
        scale=scale,
    )


def _is_phi_local_maximum(subcase: str, agent_action: int, opponent_action: int) -> bool:
    if subcase == potential.SUBCASE_CONGESTION:
        return agent_action != opponent_action
    return agent_action == opponent_action


def test_potential_4_subcases_register() -> None:
    """The registry and direct factory construct every section 5.6 subcase."""
    assert GAME_REGISTRY["potential"] is potential.build

    for subcase in SUBCASES:
        direct = _build_env(subcase)
        direct_state, direct_info = direct.reset()
        assert direct_state.shape == (1,)
        assert direct_info["game_name"] == "potential"
        assert direct.game_info()["subcase"] == subcase

        via_registry = make_game(
            "potential",
            subcase=subcase,
            adversary=_stationary_action(0),
            horizon=3,
            m=M,
            seed=0,
        )
        assert via_registry.game_info()["subcase"] == subcase


def test_payoff_correctness() -> None:
    """Realized u_1 and u_2 match the documented payoff form once per subcase."""
    env = _build_env(potential.SUBCASE_COORDINATION, opponent_action=1)
    env.reset()
    _, reward, _, info = env.step(1)
    assert reward == pytest.approx(1.0)
    assert info["opponent_reward"] == pytest.approx(1.0)

    env = _build_env(potential.SUBCASE_CONGESTION, opponent_action=2)
    env.reset()
    _, reward, _, info = env.step(2)
    assert reward == pytest.approx(1.0 / 3.0)
    assert info["opponent_reward"] == pytest.approx(1.0 / 3.0)

    lambda_inertia = 0.25
    env = _build_env(
        potential.SUBCASE_INERTIA,
        opponent_action=1,
        horizon=3,
        lambda_inertia=lambda_inertia,
    )
    env.reset()
    env.step(0)
    _, reward, _, info = env.step(1)
    assert reward == pytest.approx(1.0 - lambda_inertia)
    assert info["opponent_reward"] == pytest.approx(1.0)

    env = _build_env(
        potential.SUBCASE_SWITCHING,
        opponent_action=2,
        switch_period_episodes=2,
    )
    env.reset()
    _, reward, _, info = env.step(2)
    assert reward == pytest.approx(1.0)
    assert info["opponent_reward"] == pytest.approx(1.0)


def test_potential_function_phi_pinned() -> None:
    """``compute_potential`` pins Phi(a, b) = 1[a == b] on a 3x3 grid."""
    expected = np.eye(M, dtype=np.float64)
    actual = np.array(
        [
            [
                potential.compute_potential(
                    (agent_action, opponent_action),
                    subcase=potential.SUBCASE_COORDINATION,
                    m=M,
                )
                for opponent_action in range(M)
            ]
            for agent_action in range(M)
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("subcase", SUBCASES)
def test_one_step_better_reply_strictly_increases_phi(subcase: str) -> None:
    """Non-equilibrium agent deviations increase Phi; equilibria are local maxima."""
    for agent_action, opponent_action in product(range(M), repeat=2):
        current = _phi(subcase, agent_action, opponent_action)
        deviations = [
            (_phi(subcase, candidate, opponent_action), candidate)
            for candidate in range(M)
            if candidate != agent_action
        ]
        improving = [
            candidate
            for candidate_phi, candidate in deviations
            if candidate_phi > current
        ]

        if _is_phi_local_maximum(subcase, agent_action, opponent_action):
            assert not improving, (subcase, agent_action, opponent_action, improving)
        else:
            assert improving, (subcase, agent_action, opponent_action)


@pytest.mark.parametrize("subcase", SUBCASES)
def test_canonical_sign_plus(subcase: str) -> None:
    env = _build_env(subcase)
    assert env.env_canonical_sign == "+"
    assert env.game_info()["env_canonical_sign"] == "+"


def test_inertia_penalty_applied() -> None:
    """For the same current profile, switching is penalized relative to staying."""
    lambda_inertia = 0.3

    staying_env = _build_env(
        potential.SUBCASE_INERTIA,
        opponent_action=1,
        horizon=3,
        lambda_inertia=lambda_inertia,
    )
    staying_env.reset()
    staying_env.step(1)
    _, staying_reward, _, staying_info = staying_env.step(1)

    switching_env = _build_env(
        potential.SUBCASE_INERTIA,
        opponent_action=1,
        horizon=3,
        lambda_inertia=lambda_inertia,
    )
    switching_env.reset()
    switching_env.step(0)
    _, switching_reward, _, switching_info = switching_env.step(1)

    assert staying_info["last_agent_action"] == 1
    assert switching_info["last_agent_action"] == 0
    assert staying_reward == pytest.approx(1.0)
    assert switching_reward == pytest.approx(1.0 - lambda_inertia)
    assert staying_reward - switching_reward == pytest.approx(lambda_inertia)


def test_switching_payoff_period() -> None:
    """Switching payoff scale changes every configured episode block."""
    env = _build_env(
        potential.SUBCASE_SWITCHING,
        opponent_action=0,
        horizon=2,
        switch_period_episodes=2,
    )

    expected = [
        (potential.REGIME_SCALE_LO, 1.0),
        (potential.REGIME_SCALE_LO, 1.0),
        (potential.REGIME_SCALE_HI, 2.0),
        (potential.REGIME_SCALE_HI, 2.0),
        (potential.REGIME_SCALE_LO, 1.0),
    ]
    for expected_regime, expected_scale in expected:
        _, reset_info = env.reset()
        assert reset_info["regime"] == expected_regime

        _, reward, absorbing, step_info = env.step(0)
        assert absorbing is False
        assert reward == pytest.approx(expected_scale)
        assert step_info["regime"] == expected_regime

        _, terminal_reward, absorbing, _ = env.step(0)
        assert terminal_reward == pytest.approx(expected_scale)
        assert absorbing is True


@pytest.mark.parametrize("subcase", SUBCASES)
def test_horizon_termination(subcase: str) -> None:
    env = _build_env(subcase, horizon=3)
    env.reset()
    for step_index in range(3):
        _, _, absorbing, _ = env.step(0)
        assert absorbing is (step_index == 2)


@pytest.mark.parametrize("subcase", SUBCASES)
def test_info_regime_present(subcase: str) -> None:
    env = _build_env(subcase)
    _, reset_info = env.reset()
    assert "regime" in reset_info

    _, _, _, step_info = env.step(0)
    assert "regime" in step_info
    if subcase == potential.SUBCASE_SWITCHING:
        assert step_info["regime"] in {
            potential.REGIME_SCALE_LO,
            potential.REGIME_SCALE_HI,
        }
    else:
        assert step_info["regime"] == subcase


def _rollout(subcase: str) -> List[Tuple[Any, ...]]:
    env = potential.build(
        subcase=subcase,
        adversary=StationaryMixedOpponent(probs=[0.2, 0.3, 0.5], seed=999),
        horizon=6,
        m=M,
        seed=123,
        switch_period_episodes=2,
        lambda_inertia=0.3,
    )
    state, info = env.reset()
    trajectory: List[Tuple[Any, ...]] = [(tuple(state.tolist()), info["regime"])]
    for action in [0, 1, 2, 0, 2, 1]:
        next_state, reward, absorbing, step_info = env.step(action)
        trajectory.append(
            (
                tuple(next_state.tolist()),
                reward,
                absorbing,
                step_info["agent_action"],
                step_info["opponent_action"],
                step_info["opponent_reward"],
                step_info["regime"],
                step_info.get("last_agent_action"),
            )
        )
    return trajectory


@pytest.mark.parametrize("subcase", SUBCASES)
def test_determinism(subcase: str) -> None:
    assert _rollout(subcase) == _rollout(subcase)
