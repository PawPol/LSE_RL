"""Regression tests for the AdaptiveBetaQAgent beta-zero collapse guard.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §13.2 item 8
and §16.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.agents import AdaptiveBetaQAgent
from experiments.adaptive_beta.schedules import (
    ALL_METHOD_IDS,
    METHOD_ADAPTIVE_BETA,
    METHOD_ADAPTIVE_BETA_NO_CLIP,
    METHOD_ADAPTIVE_MAGNITUDE_ONLY,
    METHOD_ADAPTIVE_SIGN_ONLY,
    METHOD_CONTRACTION_UCB_BETA,
    METHOD_FIXED_NEGATIVE,
    METHOD_FIXED_POSITIVE,
    METHOD_HAND_ADAPTIVE_BETA,
    METHOD_ORACLE_BETA,
    METHOD_RETURN_UCB_BETA,
    METHOD_VANILLA,
    METHOD_WRONG_SIGN,
    build_schedule,
)
from src.lse_rl.operator.tab_operator import _EPS_BETA


_ADAPTIVE_ZERO_METHODS = {
    METHOD_ADAPTIVE_BETA,
    METHOD_ADAPTIVE_BETA_NO_CLIP,
    METHOD_ADAPTIVE_SIGN_ONLY,
    METHOD_ADAPTIVE_MAGNITUDE_ONLY,
    METHOD_HAND_ADAPTIVE_BETA,
}


def _drive_zero_advantage_episode(schedule, episode_index: int) -> None:
    schedule.update_after_episode(
        episode_index,
        rewards=np.asarray([1.0, -1.0], dtype=np.float64),
        v_next=np.asarray([1.0, -1.0], dtype=np.float64),
        divergence_event=False,
    )


def _schedule_at_zero_beta(method_id: str):
    if method_id == METHOD_VANILLA:
        return build_schedule(method_id, env_canonical_sign=None, hyperparams={}), 0

    if method_id in {METHOD_FIXED_POSITIVE, METHOD_FIXED_NEGATIVE}:
        return (
            build_schedule(
                method_id,
                env_canonical_sign=None,
                hyperparams={"beta0": 0.0},
            ),
            0,
        )

    if method_id == METHOD_WRONG_SIGN:
        return (
            build_schedule(
                method_id,
                env_canonical_sign="+",
                hyperparams={"beta0": 0.0},
            ),
            0,
        )

    if method_id in _ADAPTIVE_ZERO_METHODS:
        schedule = build_schedule(method_id, env_canonical_sign="+", hyperparams={})
        _drive_zero_advantage_episode(schedule, 0)
        return schedule, 1

    if method_id == METHOD_ORACLE_BETA:
        schedule = build_schedule(
            method_id,
            env_canonical_sign=None,
            hyperparams={"regime_to_beta": {"zero": 0.0}},
        )
        schedule.update_after_episode(
            0,
            rewards=np.asarray([2.0], dtype=np.float64),
            v_next=np.asarray([0.0], dtype=np.float64),
            divergence_event=False,
            episode_info={"regime": "zero"},
        )
        return schedule, 1

    if method_id == METHOD_CONTRACTION_UCB_BETA:
        schedule = build_schedule(method_id, env_canonical_sign=None, hyperparams={})
        for e, residual in enumerate([1.0, 0.5, 0.25]):
            schedule.update_after_episode(
                e,
                rewards=np.asarray([0.0], dtype=np.float64),
                v_next=np.asarray([0.0], dtype=np.float64),
                divergence_event=False,
                bellman_residual=residual,
            )
        return schedule, 3

    if method_id == METHOD_RETURN_UCB_BETA:
        schedule = build_schedule(method_id, env_canonical_sign=None, hyperparams={})
        for e in range(3):
            schedule.update_after_episode(
                e,
                rewards=np.asarray([0.0], dtype=np.float64),
                v_next=np.asarray([0.0], dtype=np.float64),
                divergence_event=False,
                episode_return=0.0,
            )
        return schedule, 3

    raise AssertionError(f"unhandled method_id in ALL_METHOD_IDS: {method_id}")


@pytest.mark.parametrize("method_id", ALL_METHOD_IDS)
def test_beta0_collapse_preserved_for_all_method_ids(method_id: str) -> None:
    schedule, episode_index = _schedule_at_zero_beta(method_id)
    beta = schedule.beta_for_episode(episode_index)
    assert abs(beta) <= _EPS_BETA

    gamma = 0.5
    reward = 1.0
    v_next = 2.0
    classical_target = reward + gamma * v_next
    agent = AdaptiveBetaQAgent(
        n_states=3,
        n_actions=2,
        gamma=gamma,
        learning_rate=1.0,
        epsilon_schedule=lambda _e: 0.0,
        beta_schedule=schedule,
        rng=np.random.default_rng(0),
    )
    agent.Q[1, :] = np.asarray([v_next, 1.0], dtype=np.float64)

    agent.begin_episode(episode_index)
    diag = agent.step(
        state=0,
        action=0,
        reward=reward,
        next_state=1,
        absorbing=False,
        episode_index=episode_index,
    )

    assert diag["beta_used"] == beta
    assert diag["td_target"] == classical_target
    assert agent.Q[0, 0] == classical_target
