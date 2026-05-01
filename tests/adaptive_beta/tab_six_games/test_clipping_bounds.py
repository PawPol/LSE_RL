"""Clipping-bound tests for Phase VIII adaptive-beta schedules.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §13.2 item 9.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import (
    AdaptiveBetaSchedule,
    AdaptiveMagnitudeOnlySchedule,
    AdaptiveSignOnlySchedule,
)


def _adaptive_beta(beta_cap: float):
    return AdaptiveBetaSchedule(
        no_clip=False,
        hyperparams={
            "beta_max": 4.0 * beta_cap,
            "beta_cap": beta_cap,
            "k": 10.0,
            "lambda_smooth": 1.0,
        },
    )


def _adaptive_sign_only(beta_cap: float):
    return AdaptiveSignOnlySchedule(
        hyperparams={
            "beta0": 4.0 * beta_cap,
            "beta_cap": beta_cap,
            "lambda_smooth": 1.0,
        }
    )


def _adaptive_magnitude_only(beta_cap: float):
    return AdaptiveMagnitudeOnlySchedule(
        env_canonical_sign="+",
        hyperparams={
            "beta_max": 4.0 * beta_cap,
            "beta_cap": beta_cap,
            "k": 10.0,
            "lambda_smooth": 1.0,
        },
    )


@pytest.mark.parametrize("beta_cap", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    ("schedule_name", "schedule_factory"),
    [
        ("adaptive_beta", _adaptive_beta),
        ("adaptive_sign_only", _adaptive_sign_only),
        ("adaptive_magnitude_only", _adaptive_magnitude_only),
    ],
)
def test_clipped_schedules_respect_configured_beta_cap(
    schedule_name: str,
    schedule_factory: Callable[[float], object],
    beta_cap: float,
) -> None:
    schedule = schedule_factory(beta_cap)
    rewards = np.full(4, 10.0, dtype=np.float64)
    v_next = np.zeros(4, dtype=np.float64)
    raw_exceed_count = 0
    beta_used_values = [schedule.beta_for_episode(0)]

    for episode_index in range(100):
        current_beta = schedule.beta_for_episode(episode_index)
        assert -beta_cap <= current_beta <= beta_cap

        schedule.update_after_episode(
            episode_index,
            rewards=rewards,
            v_next=v_next,
            divergence_event=False,
        )
        diag = schedule.diagnostics()
        raw_beta = float(diag["beta_raw"])
        beta_used = float(diag["beta_used"])
        beta_used_values.append(schedule.beta_for_episode(episode_index + 1))

        assert abs(raw_beta) > beta_cap, (
            f"{schedule_name} did not receive an over-cap raw beta at "
            f"episode {episode_index}: raw={raw_beta}, cap={beta_cap}"
        )
        raw_exceed_count += 1
        assert -beta_cap <= beta_used <= beta_cap

    assert raw_exceed_count == 100
    assert all(-beta_cap <= beta <= beta_cap for beta in beta_used_values)
