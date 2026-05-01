"""Hand-adaptive beta schedule contract tests for Phase VIII M4."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import (
    ALL_METHOD_IDS,
    METHOD_HAND_ADAPTIVE_BETA,
    HandAdaptiveBetaSchedule,
    build_schedule,
)


def _arr(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _update_with_advantage(schedule, episode_index: int, advantage: float) -> None:
    schedule.update_after_episode(
        episode_index,
        rewards=_arr([advantage]),
        v_next=_arr([0.0]),
    )


def _beta_sequence(advantages: list[float]) -> list[float]:
    schedule = build_schedule(
        METHOD_HAND_ADAPTIVE_BETA,
        env_canonical_sign=None,
        hyperparams={},
    )
    betas: list[float] = []
    for episode_index, advantage in enumerate(advantages):
        _update_with_advantage(schedule, episode_index, advantage)
        betas.append(schedule.beta_for_episode(episode_index + 1))
    return betas


def test_hand_adaptive_in_all_method_ids() -> None:
    assert METHOD_HAND_ADAPTIVE_BETA in ALL_METHOD_IDS


def test_hand_adaptive_factory_construct() -> None:
    schedule = build_schedule(
        METHOD_HAND_ADAPTIVE_BETA,
        env_canonical_sign=None,
        hyperparams={},
    )

    assert isinstance(schedule, HandAdaptiveBetaSchedule)
    assert schedule.name == METHOD_HAND_ADAPTIVE_BETA


def test_hand_adaptive_default_constants_pinned() -> None:
    assert HandAdaptiveBetaSchedule.DEFAULT_BETA0 == pytest.approx(1.0)
    assert HandAdaptiveBetaSchedule.DEFAULT_A_SCALE == pytest.approx(0.1)
    assert HandAdaptiveBetaSchedule.DEFAULT_LAMBDA_SMOOTH == pytest.approx(1.0)


def test_hand_adaptive_rule_sign() -> None:
    positive = build_schedule(
        METHOD_HAND_ADAPTIVE_BETA,
        env_canonical_sign=None,
        hyperparams={},
    )
    negative = build_schedule(
        METHOD_HAND_ADAPTIVE_BETA,
        env_canonical_sign=None,
        hyperparams={},
    )

    _update_with_advantage(positive, 0, +0.05)
    _update_with_advantage(negative, 0, -0.05)

    assert positive.beta_for_episode(1) > 0.0
    assert negative.beta_for_episode(1) < 0.0


def test_hand_adaptive_rule_magnitude() -> None:
    small = build_schedule(
        METHOD_HAND_ADAPTIVE_BETA,
        env_canonical_sign=None,
        hyperparams={},
    )
    large = build_schedule(
        METHOD_HAND_ADAPTIVE_BETA,
        env_canonical_sign=None,
        hyperparams={},
    )

    _update_with_advantage(
        small,
        0,
        advantage=0.5 * HandAdaptiveBetaSchedule.DEFAULT_A_SCALE,
    )
    _update_with_advantage(
        large,
        0,
        advantage=2.0 * HandAdaptiveBetaSchedule.DEFAULT_A_SCALE,
    )

    beta0 = HandAdaptiveBetaSchedule.DEFAULT_BETA0
    assert 0.0 < abs(small.beta_for_episode(1)) < beta0
    assert abs(large.beta_for_episode(1)) == pytest.approx(beta0)


def test_hand_adaptive_determinism() -> None:
    advantages = [0.02, 0.11, -0.04, -0.20, 0.00, 0.07]

    first = _beta_sequence(advantages)
    second = _beta_sequence(advantages)

    assert second == pytest.approx(first)


def test_hand_adaptive_initial_beta_zero() -> None:
    schedule = build_schedule(
        METHOD_HAND_ADAPTIVE_BETA,
        env_canonical_sign=None,
        hyperparams={},
    )

    assert schedule.beta_for_episode(0) == pytest.approx(0.0)
