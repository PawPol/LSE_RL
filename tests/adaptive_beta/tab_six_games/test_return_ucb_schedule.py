"""ReturnUCB schedule contract tests for Phase VIII M4 W2.C."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import (
    ALL_METHOD_IDS,
    DEFAULT_BETA_ARM_GRID,
    METHOD_RETURN_UCB_BETA,
    ReturnUCBBetaSchedule,
    build_schedule,
)


def _drive_return_episode(
    schedule: ReturnUCBBetaSchedule,
    episode_index: int,
    episode_return: float,
    *,
    rewards: np.ndarray | None = None,
    v_next: np.ndarray | None = None,
) -> None:
    schedule.update_after_episode(
        episode_index,
        np.zeros(1, dtype=np.float64) if rewards is None else rewards,
        np.zeros(1, dtype=np.float64) if v_next is None else v_next,
        episode_return=episode_return,
    )


def test_in_all_method_ids() -> None:
    assert METHOD_RETURN_UCB_BETA in ALL_METHOD_IDS


def test_factory_construct() -> None:
    schedule = build_schedule(METHOD_RETURN_UCB_BETA, env_canonical_sign=None, hyperparams={})

    assert isinstance(schedule, ReturnUCBBetaSchedule)
    assert schedule.name == METHOD_RETURN_UCB_BETA

    custom = build_schedule(
        METHOD_RETURN_UCB_BETA,
        env_canonical_sign=None,
        hyperparams={"arm_grid": [-1.0, 0.0, 1.0], "ucb_c": 3.0, "epsilon_floor": 1e-6},
    )

    assert isinstance(custom, ReturnUCBBetaSchedule)
    assert custom.arm_grid == (-1.0, 0.0, 1.0)
    assert custom.diagnostics()["ucb_c"] == pytest.approx(3.0)


def test_warm_start_round_robin() -> None:
    """v10: warm-start round-robins through all 21 arms."""
    schedule = ReturnUCBBetaSchedule()
    observed_betas: list[float] = []

    for episode_index, expected_beta in enumerate(DEFAULT_BETA_ARM_GRID):
        observed_betas.append(schedule.beta_for_episode(episode_index))
        _drive_return_episode(schedule, episode_index, episode_return=10.0 + expected_beta)

    np.testing.assert_allclose(
        np.asarray(observed_betas, dtype=np.float64),
        np.asarray(DEFAULT_BETA_ARM_GRID, dtype=np.float64),
        atol=0.0,
    )
    assert schedule.pull_counts() == tuple([1] * len(DEFAULT_BETA_ARM_GRID))


def test_ucb_constant_is_sqrt2() -> None:
    assert ReturnUCBBetaSchedule._DEFAULT_UCB_C == pytest.approx(math.sqrt(2.0))
    assert ReturnUCBBetaSchedule().diagnostics()["ucb_c"] == pytest.approx(math.sqrt(2.0))


def test_reward_is_episode_return() -> None:
    schedule = ReturnUCBBetaSchedule()

    _drive_return_episode(
        schedule,
        0,
        episode_return=42.5,
        rewards=np.array([-100.0, -200.0], dtype=np.float64),
        v_next=np.array([5.0, 5.0], dtype=np.float64),
    )

    assert schedule.pull_counts()[0] == 1
    assert schedule.arm_means()[0] == pytest.approx(42.5)

    n_arms = len(DEFAULT_BETA_ARM_GRID)
    for episode_index in range(1, n_arms):
        _drive_return_episode(schedule, episode_index, episode_return=100.0 + episode_index)

    # After warm-start (n_arms episodes), the next episode cycles back to arm 0.
    assert schedule.beta_for_episode(n_arms) == pytest.approx(DEFAULT_BETA_ARM_GRID[0])

    _drive_return_episode(
        schedule,
        n_arms,
        episode_return=-7.5,
        rewards=np.array([1000.0], dtype=np.float64),
        v_next=np.array([-1000.0], dtype=np.float64),
    )

    assert schedule.pull_counts()[0] == 2
    assert schedule.arm_means()[0] == pytest.approx((42.5 - 7.5) / 2.0)


def test_arm_grid_default() -> None:
    """v10: 21-arm grid in [-2, 2] with denser spacing near 0."""
    schedule = ReturnUCBBetaSchedule()

    expected = [
        -2.00, -1.70, -1.35, -1.00, -0.75, -0.50, -0.35, -0.20, -0.10, -0.05,
         0.00,
        +0.05, +0.10, +0.20, +0.35, +0.50, +0.75, +1.00, +1.35, +1.70, +2.00,
    ]
    assert list(schedule.arm_grid) == expected


def test_ucb_selection_after_warm_start() -> None:
    """v10: warm-start spans 21 episodes; UCB selection from episode 21."""
    schedule = ReturnUCBBetaSchedule()
    n_arms = 21

    for episode_index in range(n_arms):
        _drive_return_episode(schedule, episode_index, episode_return=1.0)

    assert schedule.pull_counts() == tuple([1] * n_arms)
    assert schedule.diagnostics()["current_arm_idx"] == 0
    assert schedule.beta_for_episode(n_arms) == pytest.approx(DEFAULT_BETA_ARM_GRID[0])
