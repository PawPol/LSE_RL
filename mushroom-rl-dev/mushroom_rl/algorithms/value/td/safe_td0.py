"""
Safe weighted-LSE TD(0) for fixed-policy prediction.

Computes safe targets ``g_t^safe(r, v_next)`` where ``v_next`` is the
expected value under the current (fixed) policy at the next state.

When ``beta_used == 0`` at every stage, this algorithm produces updates
bit-identical to classical TD(0) prediction (within numerical tolerance).

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

__all__ = ["SafeTD0"]


class SafeTD0(SafeWeightedLSEBase, TD):
    """
    Safe weighted-LSE TD(0) for fixed-policy value prediction.

    Uses a Q-table internally (consistent with MushroomRL's TD base class).
    The value estimate at state s is ``V(s) = Q[s, :] . pi(s)``, i.e. the
    expectation of Q under the evaluation policy.

    Parameters
    ----------
    mdp_info : MDPInfo
        MDP descriptor (observation space, action space, gamma, horizon).
    policy : Policy
        Fixed evaluation policy.
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
        # Current value estimate: V(s) = E_pi[Q(s, .)]
        v_current = self.Q[state, :].dot(self.policy(state))

        if not absorbing:
            v_next = self.Q[next_state, :].dot(self.policy(next_state))
        else:
            v_next = 0.0

        # Safe target replaces classical target r + gamma * v_next
        safe_target = self._safe_target(
            float(reward), float(v_next), int(np.asarray(state).flat[0])
        )

        td_error = safe_target - v_current

        # Update Q[state, action] by stepping toward the safe target
        self.Q[state, action] = (
            self.Q[state, action] + self._alpha(state, action) * td_error
        )
