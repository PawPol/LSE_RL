"""
Safe weighted-LSE Expected SARSA with a frozen target network (Phase IV-C).

Same target-network machinery as :class:`SafeTargetQLearning`, but the
bootstrap is the *policy-expected* value under an epsilon-greedy policy
read from the target table::

    pi_target(a | s') = (1 - eps) * 1[a == argmax_a Q_target[t, s', a]]
                       + eps / n_actions
    v_next = sum_a pi_target(a | s') * Q_target[t, s', a]
           = (1 - eps) * max_a Q_target[t, s', a]
             + eps / n_actions * sum_a Q_target[t, s', a]

For ``eps == 0`` this collapses to the greedy case (SafeTargetQLearning).
For ``eps == 1`` it is the uniform-policy expected value.

Diagnostic ``q_online_next`` mirrors the same formula applied to the
*online* table.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .safe_target_q import SafeTargetQLearning
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
)

__all__ = ["SafeTargetExpectedSARSA"]


class SafeTargetExpectedSARSA(SafeTargetQLearning):
    """Safe weighted-LSE Expected SARSA with a frozen target network.

    Parameters
    ----------
    n_states, n_actions, schedule, learning_rate, gamma, sync_every,
    polyak_tau, seed:
        See :class:`SafeTargetQLearning`.
    epsilon:
        Epsilon-greedy parameter for the target policy used to compute
        the expected next-state value.  ``epsilon in [0, 1]``.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        schedule: BetaSchedule,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        sync_every: int = 200,
        polyak_tau: float = 0.0,
        epsilon: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            schedule=schedule,
            learning_rate=learning_rate,
            gamma=gamma,
            sync_every=sync_every,
            polyak_tau=polyak_tau,
            seed=seed,
        )
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(
                f"epsilon must be in [0, 1]; got {epsilon}."
            )
        self._epsilon = float(epsilon)

    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def set_epsilon(self, value: float) -> None:
        """Update the epsilon used in the expected-value bootstrap."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1]; got {value}.")
        self._epsilon = float(value)

    # ------------------------------------------------------------------
    # Overridden bootstrap computations
    # ------------------------------------------------------------------

    def _expected_value(self, Q_slice: np.ndarray) -> float:
        """E_{pi_eps}[Q_slice] with pi_eps epsilon-greedy over Q_slice itself."""
        eps = self._epsilon
        n = self._n_actions
        if n <= 0:
            return 0.0
        max_v = float(np.max(Q_slice))
        mean_v = float(np.mean(Q_slice))
        # (1 - eps) * max + eps * mean.  Equivalent to
        # (1 - eps) * max + eps/n * sum.
        return (1.0 - eps) * max_v + eps * mean_v

    def _compute_v_next_target(
        self, next_state: int, stage: int, absorbing: bool
    ) -> float:
        if absorbing:
            return 0.0
        Q_slice = self._Q_target[stage, next_state, :]
        return self._expected_value(Q_slice)

    def _compute_v_next_online(
        self, next_state: int, stage: int, absorbing: bool
    ) -> float:
        if absorbing:
            return 0.0
        Q_slice = self._Q_online[stage, next_state, :]
        return self._expected_value(Q_slice)

    # ------------------------------------------------------------------
    # update() is inherited; log dict is identical.
    # ------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        absorbing: bool,
        stage: int,
        global_step: int,
    ) -> dict[str, Any]:
        log = super().update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            absorbing=absorbing,
            stage=stage,
            global_step=global_step,
        )
        log["epsilon"] = self._epsilon
        return log
