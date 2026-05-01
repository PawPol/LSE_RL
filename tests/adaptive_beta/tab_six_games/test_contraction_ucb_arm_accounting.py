"""Phase VIII M4 W2.B arm-accounting tests for ContractionUCB."""

from __future__ import annotations

import numpy as np

from experiments.adaptive_beta.schedules import ContractionUCBBetaSchedule
from experiments.adaptive_beta.tab_six_games import metrics

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


def test_total_pulls_equals_total_episodes() -> None:
    schedule = ContractionUCBBetaSchedule()
    n_episodes = 14

    for episode_index in range(n_episodes):
        _update(schedule, episode_index, bellman_residual=1.0 + episode_index)

    assert sum(schedule.pull_counts()) == n_episodes


def test_each_arm_pulled_at_least_once_in_warm_start() -> None:
    schedule = ContractionUCBBetaSchedule()

    for episode_index in range(7):
        _update(schedule, episode_index, bellman_residual=1.0 + episode_index)

    assert schedule.pull_counts() == (1, 1, 1, 1, 1, 1, 1)


def test_welford_recursion_bit_identical() -> None:
    """Welford uses population std, ddof=0, for per-arm reward dispersion."""

    schedule = ContractionUCBBetaSchedule()
    reward_sequence = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    for reward in reward_sequence:
        schedule._record_arm_reward(2, float(reward))

    assert schedule.arm_means()[2] == float(np.mean(reward_sequence))
    assert schedule.arm_stds()[2] == float(np.std(reward_sequence, ddof=0))


def test_arm_value_zero_for_unpulled_arm() -> None:
    reward_history = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    arm_idx_history = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    arm_value = metrics.ucb_arm_value(reward_history, arm_idx_history, n_arms=7)

    assert arm_value[5] == 0.0
