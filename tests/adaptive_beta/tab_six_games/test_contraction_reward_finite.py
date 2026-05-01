"""Finite contraction-reward regression tests for lessons.md #27."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import (
    DEFAULT_BETA_ARM_GRID,
    ContractionUCBBetaSchedule,
)
from experiments.adaptive_beta.tab_six_games import metrics

EMPTY_TRACE = np.array([0.0], dtype=np.float64)


class RecordingContractionUCBBetaSchedule(ContractionUCBBetaSchedule):
    """Collect raw contraction rewards before Welford standardisation."""

    def __init__(self, arm_grid: list[float]) -> None:
        self.recorded_rewards: list[tuple[int, float]] = []
        super().__init__(arm_grid=arm_grid)

    def _record_arm_reward(self, arm_idx: int, raw_reward: float) -> None:
        self.recorded_rewards.append((int(arm_idx), float(raw_reward)))
        super()._record_arm_reward(arm_idx, raw_reward)


def _extreme_cycle() -> np.ndarray:
    values = np.array([0.0, 1e-15, 1e-8, 1.0, 1e6, 1e10], dtype=np.float64)
    return np.resize(values, 1000)


def _monotone_decreasing() -> np.ndarray:
    positive_tail = np.geomspace(1e10, 1e-15, 999, dtype=np.float64)
    return np.concatenate([positive_tail, np.array([0.0], dtype=np.float64)])


def _monotone_increasing() -> np.ndarray:
    positive_tail = np.geomspace(1e-15, 1e10, 999, dtype=np.float64)
    return np.concatenate([np.array([0.0], dtype=np.float64), positive_tail])


RESIDUAL_SEQUENCES = (
    ("extreme_cycle", _extreme_cycle()),
    ("monotone_decreasing", _monotone_decreasing()),
    ("monotone_increasing", _monotone_increasing()),
)


def _run_schedule(arm_beta: float, residuals: np.ndarray) -> np.ndarray:
    schedule = RecordingContractionUCBBetaSchedule(arm_grid=[float(arm_beta)])
    for episode_index, residual in enumerate(residuals):
        schedule.update_after_episode(
            episode_index,
            EMPTY_TRACE,
            EMPTY_TRACE,
            bellman_residual=float(residual),
        )

    rewards = np.array(
        [reward for _, reward in schedule.recorded_rewards],
        dtype=np.float64,
    )
    assert rewards.size >= residuals.size - 1
    assert np.all(np.isfinite(schedule.arm_means()))
    assert np.all(np.isfinite(schedule.arm_stds()))
    return rewards


@pytest.mark.parametrize(
    "arm_beta",
    DEFAULT_BETA_ARM_GRID,
    ids=lambda beta: f"beta_{beta:g}",
)
@pytest.mark.parametrize(
    "sequence_name,residuals",
    RESIDUAL_SEQUENCES,
    ids=[name for name, _ in RESIDUAL_SEQUENCES],
)
def test_contraction_reward_finite_for_all_arms_and_extreme_sequences(
    arm_beta: float,
    sequence_name: str,
    residuals: np.ndarray,
) -> None:
    metric_rewards = metrics.contraction_reward(residuals)
    schedule_rewards = _run_schedule(arm_beta, residuals)

    assert residuals.shape == (1000,), sequence_name
    assert metric_rewards.shape == (999,)
    assert np.all(np.isfinite(metric_rewards)), sequence_name
    assert np.all(np.isfinite(schedule_rewards)), (arm_beta, sequence_name)


def test_lessons_27_zero_to_zero_underflow_case_is_zero() -> None:
    residuals = np.array([0.0, 0.0], dtype=np.float64)

    metric_rewards = metrics.contraction_reward(residuals)
    schedule_rewards = _run_schedule(arm_beta=0.0, residuals=residuals)

    np.testing.assert_allclose(metric_rewards, np.array([0.0]), rtol=0.0, atol=0.0)
    assert np.isfinite(metric_rewards[0])
    assert np.isfinite(schedule_rewards[-1])
    assert schedule_rewards[-1] == pytest.approx(0.0, abs=0.0)
