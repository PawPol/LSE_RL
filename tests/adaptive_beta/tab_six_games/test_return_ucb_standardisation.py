"""ReturnUCB standardised reward stream tests for spec §13.2 #6."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import ReturnUCBBetaSchedule


def _attach_standardised_reward_capture(
    schedule: ReturnUCBBetaSchedule,
) -> list[list[float]]:
    """Capture the exact per-arm standardised rewards used by UCB scoring."""
    standardised_rewards: list[list[float]] = [[] for _ in schedule.arm_grid]
    original_record_arm_reward = schedule._record_arm_reward

    def wrapped_record_arm_reward(arm_idx: int, raw_reward: float) -> None:
        previous_n = schedule._n_std_samples[arm_idx]
        previous_mean = schedule._standardised_running_mean[arm_idx]

        original_record_arm_reward(arm_idx, raw_reward)

        current_n = schedule._n_std_samples[arm_idx]
        current_mean = schedule._standardised_running_mean[arm_idx]
        assert current_n == previous_n + 1
        standardised_reward = current_n * current_mean - previous_n * previous_mean
        standardised_rewards[arm_idx].append(float(standardised_reward))

    schedule._record_arm_reward = wrapped_record_arm_reward
    return standardised_rewards


def test_stationary_episode_returns_standardise_per_arm_reward_stream() -> None:
    """v10: 21-arm grid; bump episode budget so all arms cross 50-pull threshold."""
    from experiments.adaptive_beta.schedules import DEFAULT_BETA_ARM_GRID

    n_arms = len(DEFAULT_BETA_ARM_GRID)
    # v10: 21 arms ⇒ need enough pulls per arm for sample-mean to clear the
    # 0.1 hard cap. Empirically, 10×n_arms×50 still dropped one arm's mean
    # to 0.106 (just over 0.1). 25 × n_arms × 50 = 26250 episodes guarantees
    # min ≈ 660 pulls ⇒ sample-mean std ≈ 0.039 ≪ 0.1 with safety margin.
    n_episodes = 25 * n_arms * 50

    schedule = ReturnUCBBetaSchedule()
    standardised_rewards = _attach_standardised_reward_capture(schedule)
    rng = np.random.default_rng(38)
    episode_returns = rng.normal(loc=10.0, scale=2.0, size=n_episodes)
    zeros = np.zeros(1, dtype=np.float64)

    assert schedule.diagnostics()["ucb_c"] == pytest.approx(math.sqrt(2.0))

    for episode_index, episode_return in enumerate(episode_returns):
        schedule.update_after_episode(
            episode_index,
            zeros,
            zeros,
            episode_return=float(episode_return),
        )

    pull_counts = schedule.pull_counts()
    eligible_arms = [arm for arm, count in enumerate(pull_counts) if count >= 50]

    assert eligible_arms == list(range(n_arms)), (
        f"stationary seeded ReturnUCB run should leave all {n_arms} arms eligible "
        f"for per-arm standardisation checks; counts={pull_counts}"
    )

    for arm_idx in eligible_arms:
        stream = np.asarray(standardised_rewards[arm_idx], dtype=np.float64)
        assert stream.size == pull_counts[arm_idx]

        empirical_mean = float(stream.mean())
        empirical_std = float(stream.std(ddof=0))

        assert abs(empirical_mean) <= 0.1, (
            f"arm {arm_idx} standardised reward mean drifted: "
            f"mean={empirical_mean}, pulls={stream.size}"
        )
        assert abs(empirical_std - 1.0) <= 0.1, (
            f"arm {arm_idx} standardised reward std drifted: "
            f"std={empirical_std}, pulls={stream.size}"
        )
