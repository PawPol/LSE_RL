"""Phase VIII M4 W2.B contracts for ContractionUCBBetaSchedule."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import (
    ALL_METHOD_IDS,
    DEFAULT_BETA_ARM_GRID,
    METHOD_CONTRACTION_UCB_BETA,
    ContractionUCBBetaSchedule,
    build_schedule,
)

EPS = 1e-8
EMPTY_TRACE = np.array([0.0], dtype=np.float64)


def _update(
    schedule: ContractionUCBBetaSchedule,
    episode_index: int,
    bellman_residual: float,
) -> None:
    schedule.update_after_episode(
        episode_index,
        EMPTY_TRACE,
        EMPTY_TRACE,
        bellman_residual=float(bellman_residual),
    )


def _log_reward(prev_residual: float, residual: float) -> float:
    return float(np.log(prev_residual + EPS) - np.log(residual + EPS))


def test_in_all_method_ids() -> None:
    assert METHOD_CONTRACTION_UCB_BETA in ALL_METHOD_IDS


def test_factory_construct() -> None:
    schedule = build_schedule(
        METHOD_CONTRACTION_UCB_BETA,
        env_canonical_sign=None,
        hyperparams={},
    )

    assert isinstance(schedule, ContractionUCBBetaSchedule)
    assert schedule.name == METHOD_CONTRACTION_UCB_BETA
    assert schedule.arm_grid == DEFAULT_BETA_ARM_GRID


def test_warm_start_round_robin_episodes_0_to_6() -> None:
    schedule = ContractionUCBBetaSchedule()

    for episode_index, expected_beta in enumerate(DEFAULT_BETA_ARM_GRID):
        assert schedule.beta_for_episode(episode_index) == expected_beta
        _update(schedule, episode_index, bellman_residual=10.0 + episode_index)


def test_warm_start_episode_k_returns_arm_k() -> None:
    """v10: each warm-start episode k returns arm k of the 21-arm grid."""
    schedule = ContractionUCBBetaSchedule()

    for episode_index in range(len(DEFAULT_BETA_ARM_GRID) - 1):
        _update(schedule, episode_index, bellman_residual=10.0 + episode_index)

    next_idx = len(DEFAULT_BETA_ARM_GRID) - 1
    assert schedule.beta_for_episode(next_idx) == pytest.approx(
        DEFAULT_BETA_ARM_GRID[next_idx]
    )


def test_ucb_selection_after_warm_start() -> None:
    """v10: UCB selection takes over after the 21-arm warm-start completes."""
    schedule = ContractionUCBBetaSchedule()
    n_arms = len(DEFAULT_BETA_ARM_GRID)

    for episode_index in range(n_arms):
        _update(schedule, episode_index, bellman_residual=10.0 - episode_index)

    assert schedule.beta_for_episode(n_arms) in DEFAULT_BETA_ARM_GRID


def test_reward_uses_np_log_not_log1p() -> None:
    schedule = ContractionUCBBetaSchedule()
    residuals = [1e-12, 1e10, 1e-12]

    for episode_index, residual in enumerate(residuals):
        _update(schedule, episode_index, bellman_residual=residual)

    arm_means = schedule.arm_means()
    expected_arm_1 = _log_reward(1e-12, 1e10)
    expected_arm_2 = _log_reward(1e10, 1e-12)

    assert np.isfinite(expected_arm_1)
    assert np.isfinite(expected_arm_2)
    np.testing.assert_allclose(arm_means[1], expected_arm_1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(arm_means[2], expected_arm_2, rtol=0.0, atol=1e-12)
    assert np.all(np.isfinite(schedule.arm_means()))
    assert np.all(np.isfinite(schedule.arm_stds()))


def test_residual_smoothing_window_default_1() -> None:
    schedule = ContractionUCBBetaSchedule()
    residuals = [10.0, 1.0, 100.0]

    for episode_index, residual in enumerate(residuals):
        _update(schedule, episode_index, bellman_residual=residual)

    expected = [_log_reward(10.0, 1.0), _log_reward(1.0, 100.0)]
    np.testing.assert_allclose(
        [schedule.arm_means()[1], schedule.arm_means()[2]],
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_residual_smoothing_window_configurable() -> None:
    schedule = ContractionUCBBetaSchedule(residual_smoothing_window=3)
    residuals = [9.0, 3.0, 0.0, 6.0]

    for episode_index, residual in enumerate(residuals):
        _update(schedule, episode_index, bellman_residual=residual)

    smoothed = [9.0, 6.0, 4.0, 3.0]
    expected = [
        _log_reward(smoothed[0], smoothed[1]),
        _log_reward(smoothed[1], smoothed[2]),
        _log_reward(smoothed[2], smoothed[3]),
    ]
    np.testing.assert_allclose(
        [schedule.arm_means()[1], schedule.arm_means()[2], schedule.arm_means()[3]],
        expected,
        rtol=0.0,
        atol=1e-12,
    )
