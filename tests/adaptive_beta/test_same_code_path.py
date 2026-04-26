"""Single shared TD-update code path across every method ID.

Spec authority: ``docs/specs/phase_VII_adaptive_beta.md`` §16.2 (and
§16.1 for the schedule-class / β-equivalence relationship).

This test enforces the architectural invariant that every method ID
flows through ``AdaptiveBetaQAgent._step_update`` — i.e. fixed-β,
adaptive-β, and adaptive-β-no-clip all share the same TD-update kernel
and differ only in the schedule object they were constructed with. The
``_step_update_call_counter`` instrumentation hook (M3.1) makes this
externally observable.

Failure mode this test guards against
-------------------------------------
If a future refactor were to introduce a fast-path that bypassed
``_step_update`` for, say, ``vanilla`` (e.g., "since β=0 collapses to
classical Q-learning, just inline the update there"), the counter would
report a value other than 10 for that method. That is *exactly* the
silent-divergence-of-code-paths failure §16.2 forbids.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from experiments.adaptive_beta.agents import (
    AdaptiveBetaQAgent,
    linear_epsilon_schedule,
)
from experiments.adaptive_beta.envs.delayed_chain import DelayedChain
from experiments.adaptive_beta.schedules import (
    METHOD_ADAPTIVE_BETA,
    METHOD_ADAPTIVE_BETA_NO_CLIP,
    METHOD_ADAPTIVE_MAGNITUDE_ONLY,
    METHOD_ADAPTIVE_SIGN_ONLY,
    METHOD_FIXED_NEGATIVE,
    METHOD_FIXED_POSITIVE,
    METHOD_VANILLA,
    METHOD_WRONG_SIGN,
    build_schedule,
)


# Mapping from method id to the concrete schedule class name the factory
# returns. If any of these mappings drift the agent's behaviour also
# drifts, so we pin them here as a contract check (spec §16.1 / §16.2).
_EXPECTED_SCHEDULE_CLASS: Dict[str, str] = {
    METHOD_VANILLA: "ZeroBetaSchedule",
    METHOD_FIXED_POSITIVE: "FixedBetaSchedule",
    METHOD_FIXED_NEGATIVE: "FixedBetaSchedule",
    METHOD_WRONG_SIGN: "WrongSignSchedule",
    METHOD_ADAPTIVE_BETA: "AdaptiveBetaSchedule",
    METHOD_ADAPTIVE_BETA_NO_CLIP: "AdaptiveBetaSchedule",
    METHOD_ADAPTIVE_SIGN_ONLY: "AdaptiveSignOnlySchedule",
    METHOD_ADAPTIVE_MAGNITUDE_ONLY: "AdaptiveMagnitudeOnlySchedule",
}

# Method IDs we drive through the agent. All eight Phase VII methods are
# covered; ``wrong_sign`` and ``adaptive_magnitude_only`` require a
# canonical-sign env (DelayedChain has ``env_canonical_sign == "+"``).
_ALL_METHODS: Tuple[str, ...] = (
    METHOD_VANILLA,
    METHOD_FIXED_POSITIVE,
    METHOD_FIXED_NEGATIVE,
    METHOD_WRONG_SIGN,
    METHOD_ADAPTIVE_BETA,
    METHOD_ADAPTIVE_BETA_NO_CLIP,
    METHOD_ADAPTIVE_SIGN_ONLY,
    METHOD_ADAPTIVE_MAGNITUDE_ONLY,
)

# Number of synthetic transitions to drive per agent. Spec §16.2 makes no
# claim about a magic number; 10 is enough that an off-by-one in the
# counter would be obvious and small enough to keep the test trivially
# fast.
_N_TRANSITIONS: int = 10


def _build_agent(method_id: str) -> AdaptiveBetaQAgent:
    """Construct an agent for ``method_id`` against DelayedChain shape.

    DelayedChain has ``chain_length=20`` (n_states=20), 2 actions, and
    canonical sign ``+``. We use the same dims here so transitions we
    feed in are within the Q-table shape.
    """
    env = DelayedChain(seed=0)
    n_states = int(env.info.observation_space.size[0])
    n_actions = int(env.info.action_space.size[0])
    schedule = build_schedule(
        method_id,
        env_canonical_sign=env.env_canonical_sign,
        hyperparams=None,
    )
    return AdaptiveBetaQAgent(
        n_states=n_states,
        n_actions=n_actions,
        gamma=0.95,
        learning_rate=0.1,
        epsilon_schedule=linear_epsilon_schedule(1.0, 0.05, 100),
        beta_schedule=schedule,
        rng=np.random.default_rng(0),
        env_canonical_sign=env.env_canonical_sign,
    )


@pytest.mark.parametrize("method_id", _ALL_METHODS)
def test_step_update_called_for_every_method(method_id: str) -> None:
    """Spec §16.2: 10 transitions => counter == 10 for every method ID.

    A counter value of 0 (or any value != 10) for any method would
    indicate a separate code path was taken, violating the single-shared-
    update invariant.
    """
    agent = _build_agent(method_id)

    # Synthetic transitions — episode loop, but no real env required.
    # State / action / reward arbitrary; the only requirement is that
    # state and next_state are within ``n_states``.
    n_states = agent.Q.shape[0]
    n_actions = agent.Q.shape[1]
    rng = np.random.default_rng(1234)

    agent.begin_episode(0)
    for t in range(_N_TRANSITIONS):
        s = int(rng.integers(0, n_states))
        a = int(rng.integers(0, n_actions))
        r = float(rng.normal(0.0, 1.0))
        ns = int(rng.integers(0, n_states))
        # Last transition absorbing so end_episode sees a closed
        # episode; everything before is non-terminal.
        ab = bool(t == _N_TRANSITIONS - 1)
        agent.step(s, a, r, ns, ab, episode_index=0)
    agent.end_episode(0)

    assert agent.step_update_call_counter == _N_TRANSITIONS, (
        f"method_id={method_id!r}: expected "
        f"_step_update_call_counter == {_N_TRANSITIONS}, "
        f"got {agent.step_update_call_counter}. A value other than "
        f"{_N_TRANSITIONS} indicates this method ID took a separate "
        f"TD-update code path, violating spec §16.2."
    )


@pytest.mark.parametrize("method_id", _ALL_METHODS)
def test_schedule_class_matches_method_id(method_id: str) -> None:
    """Contract: ``build_schedule`` dispatches each method to the right class.

    If this drifts, the agent silently swaps schedule semantics under a
    method id and §16.1 / §16.2 invariants no longer hold.
    """
    agent = _build_agent(method_id)
    actual = agent._beta_schedule.__class__.__name__
    expected = _EXPECTED_SCHEDULE_CLASS[method_id]
    assert actual == expected, (
        f"method_id={method_id!r}: schedule class is {actual!r}, "
        f"expected {expected!r}"
    )


@pytest.mark.parametrize("method_id", _ALL_METHODS)
def test_q_table_shape_matches_env(method_id: str) -> None:
    """Contract: ``Q.shape == (n_states, n_actions)`` for every method.

    Cheap shape pin so any method that initialised its own divergent Q
    table (or accidentally swapped axes) fails loudly here, before the
    runner buries the symptom in a 1000-episode run.
    """
    agent = _build_agent(method_id)
    env = DelayedChain(seed=0)
    n_states = int(env.info.observation_space.size[0])
    n_actions = int(env.info.action_space.size[0])
    assert agent.Q.shape == (n_states, n_actions), (
        f"method_id={method_id!r}: Q.shape={agent.Q.shape}, expected "
        f"({n_states}, {n_actions})"
    )
