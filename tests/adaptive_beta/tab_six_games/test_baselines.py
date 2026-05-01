"""Phase VIII M4 baseline-agent tests.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §6.3 and
§13.4 item 2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from experiments.adaptive_beta.baselines import (
    RestartQLearningAgent,
    SlidingWindowQLearningAgent,
    TunedEpsilonGreedyQLearningAgent,
)


def _zero_epsilon(_episode_index: int) -> float:
    return 0.0


def _run_one_step_episode(agent, episode_index: int, reward: float = 1.0) -> dict:
    agent.begin_episode(episode_index)
    action = agent.select_action(0, episode_index)
    agent.step(
        state=0,
        action=action,
        reward=reward,
        next_state=1,
        absorbing=True,
        episode_index=episode_index,
    )
    return agent.end_episode(episode_index)


def _restart_agent(
    *,
    restart_window: int = 200,
    restart_drop: float = 0.5,
    divergence_threshold: float = 1.0e6,
    q_init: float = 0.0,
) -> RestartQLearningAgent:
    return RestartQLearningAgent(
        n_states=4,
        n_actions=2,
        gamma=0.9,
        learning_rate=0.25,
        epsilon_schedule=_zero_epsilon,
        rng=np.random.default_rng(0),
        restart_window=restart_window,
        restart_drop=restart_drop,
        divergence_threshold=divergence_threshold,
        q_init=q_init,
    )


def _sliding_agent(
    *,
    n_states: int = 4,
    window_size: int = 10_000,
    q_init: float = 0.0,
) -> SlidingWindowQLearningAgent:
    return SlidingWindowQLearningAgent(
        n_states=n_states,
        n_actions=2,
        gamma=0.9,
        learning_rate=0.25,
        epsilon_schedule=_zero_epsilon,
        rng=np.random.default_rng(1),
        window_size=window_size,
        q_init=q_init,
    )


def _tuned_agent(
    epsilon_schedule: Callable[[int], float] | None = None,
) -> TunedEpsilonGreedyQLearningAgent:
    return TunedEpsilonGreedyQLearningAgent(
        n_states=4,
        n_actions=2,
        gamma=0.9,
        learning_rate=0.25,
        epsilon_schedule=epsilon_schedule,
        rng=np.random.default_rng(2),
    )


def test_restart_construct_and_episode_loop() -> None:
    agent = _restart_agent()

    diags = [_run_one_step_episode(agent, e, reward=float(e)) for e in range(5)]

    assert [d["episode_index"] for d in diags] == list(range(5))
    assert [d["length"] for d in diags] == [1] * 5
    assert np.isfinite(agent.Q).all()


def test_restart_trigger_on_nan() -> None:
    agent = _restart_agent(q_init=0.25)
    agent.Q[0, 0] = np.nan

    agent.begin_episode(0)
    assert np.isfinite(agent.Q).all()
    np.testing.assert_allclose(agent.Q, np.full_like(agent.Q, 0.25))
    diag = agent.end_episode(0)

    assert diag["restart_event"] is True
    assert diag["nan_count"] == 0


def test_restart_trigger_on_divergence() -> None:
    agent = _restart_agent(divergence_threshold=1.0, q_init=-0.5)
    agent.Q[1, 1] = 1.5

    agent.begin_episode(0)
    np.testing.assert_allclose(agent.Q, np.full_like(agent.Q, -0.5))
    diag = agent.end_episode(0)

    assert diag["restart_event"] is True
    assert diag["q_abs_max"] == pytest.approx(0.5)


def test_restart_trigger_on_rolling_drop() -> None:
    agent = _restart_agent(restart_window=3, restart_drop=0.5)

    warmup_diags = [
        _run_one_step_episode(agent, e, reward=10.0)
        for e in range(3)
    ]
    assert [d["restart_event"] for d in warmup_diags] == [False, False, False]
    assert warmup_diags[-1]["best_rolling_mean_return"] == pytest.approx(10.0)

    drop_diag = _run_one_step_episode(agent, 3, reward=-10.0)
    assert drop_diag["restart_event"] is False
    assert drop_diag["rolling_mean_return"] < (
        drop_diag["best_rolling_mean_return"] - 0.5
    )

    agent.begin_episode(4)
    restart_diag = agent.end_episode(4)
    assert restart_diag["restart_event"] is True


def test_sliding_window_construct_and_episode_loop() -> None:
    agent = _sliding_agent()

    diags = [_run_one_step_episode(agent, e, reward=1.0) for e in range(5)]

    assert [d["episode_index"] for d in diags] == list(range(5))
    assert [d["length"] for d in diags] == [1] * 5
    assert all(d["buffer_size"] == e + 1 for e, d in enumerate(diags))


def test_sliding_window_state_eviction() -> None:
    q_init = -0.25
    n_states = 6
    window_size = 10_000
    agent = _sliding_agent(
        n_states=n_states,
        window_size=window_size,
        q_init=q_init,
    )

    agent.begin_episode(0)
    for t in range(window_size + 5):
        s = t % 2
        ns = (t + 1) % 2
        agent.step(
            state=s,
            action=0,
            reward=1.0,
            next_state=ns,
            absorbing=False,
            episode_index=0,
        )

    agent.Q[2:, :] = 123.0
    diag = agent.end_episode(0)

    assert diag["buffer_size"] == window_size
    assert diag["n_states_reset"] == n_states - 2
    np.testing.assert_allclose(agent.Q[2:, :], q_init)
    assert not np.allclose(agent.Q[:2, :], q_init)


def test_tuned_epsilon_greedy_default_schedule() -> None:
    agent = _tuned_agent()

    assert agent.TUNED_EPSILON_START == 1.0
    assert agent.TUNED_EPSILON_END == 0.01
    assert agent.TUNED_EPSILON_DECAY_EPISODES == 2000

    expected = {0: 1.0, 2000: 0.01, 5000: 0.01}
    for episode_index, epsilon in expected.items():
        agent.begin_episode(episode_index)
        diag = agent.end_episode(episode_index)
        assert diag["epsilon"] == pytest.approx(epsilon)


def test_tuned_epsilon_greedy_custom_schedule_override() -> None:
    custom = lambda e: 0.123 + 0.001 * e
    agent = _tuned_agent(epsilon_schedule=custom)

    agent.begin_episode(7)
    step_diag = agent.step(0, 0, 1.0, 1, True, 7)
    end_diag = agent.end_episode(7)

    assert step_diag["epsilon"] == pytest.approx(custom(7))
    assert end_diag["epsilon"] == pytest.approx(custom(7))


def test_all_three_runner_uniform_interface() -> None:
    agents = [
        _restart_agent(),
        _sliding_agent(),
        _tuned_agent(epsilon_schedule=_zero_epsilon),
    ]

    for agent in agents:
        assert callable(agent.select_action)
        assert callable(agent.begin_episode)
        assert callable(agent.step)
        assert callable(agent.end_episode)
        assert isinstance(agent.Q, np.ndarray)
        assert agent.Q.shape == (4, 2)
        assert agent.Q.dtype == np.float64
