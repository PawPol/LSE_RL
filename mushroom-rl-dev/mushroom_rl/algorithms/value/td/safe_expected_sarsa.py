"""
Safe weighted-LSE Expected SARSA (NOT sampled SARSA).

Uses safe target ``g_t^safe(r, E_pi[Q[s', :]])`` where the bootstrap
value is the policy-expected Q-value at the next state.

When ``beta_used == 0`` at every stage, this algorithm produces updates
bit-identical to classical Expected SARSA (within numerical tolerance).

Augmented state convention: ``state = t * n_base + base_state``.
"""
from __future__ import annotations

import numpy as np

from mushroom_rl.algorithms.value.td.td import TD
from mushroom_rl.algorithms.value.td.safe_weighted_lse_base import (
    SafeWeightedLSEBase,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule
from mushroom_rl.approximators.table import Table

__all__ = ["SafeExpectedSARSA"]


class SafeExpectedSARSA(SafeWeightedLSEBase, TD):
    """
    Safe weighted-LSE Expected SARSA.

    Parameters
    ----------
    mdp_info : MDPInfo
        MDP descriptor (observation space, action space, gamma, horizon).
    policy : Policy
        Evaluation/behaviour policy.
    schedule : BetaSchedule
        Calibrated per-stage beta schedule.
    n_base : int
        Number of base (un-augmented) states.
    learning_rate : Parameter
        Step-size schedule alpha(s, a).
    """

    def __init__(self, mdp_info, policy, schedule, n_base, learning_rate):
        Q = Table(mdp_info.size)
        TD.__init__(self, mdp_info, policy, Q, learning_rate)
        self._safe_init(schedule, n_base)
        self._schedule_dict = schedule._raw  # JSON-serializable dict
        self._add_save_attr(
            _schedule_dict='primitive', _n_base='primitive', _swc='none',
        )

    def _post_load(self):
        super()._post_load()
        from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule
        schedule = BetaSchedule(self._schedule_dict)
        self._safe_init(schedule, self._n_base)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        if not absorbing:
            q_next = self.Q[next_state, :].dot(self.policy(next_state))
        else:
            q_next = 0.0

        # Safe target replaces classical target r + gamma * q_next
        safe_target = self._safe_target(
            float(reward), float(q_next), int(np.asarray(state).flat[0])
        )

        self.Q[state, action] = q_current + self._alpha(state, action) * (
            safe_target - q_current
        )
