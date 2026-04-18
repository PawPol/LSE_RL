"""
Base mixin for safe weighted-LSE online TD algorithms.

Provides stage extraction from augmented state and delegates math to
SafeWeightedCommon.  All safe TD algorithms inherit from both
SafeWeightedLSEBase and TD (or a subclass).

Augmented state convention: state = t * n_base + base_state
    => t = state // n_base
    => base_state = state % n_base

n_base MUST be provided at construction time (NOT inferred from any table),
per lessons.md entry 2026-04-17 on hard-coded state constants.
"""
from __future__ import annotations

import numpy as np

from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
)

__all__ = ["SafeWeightedLSEBase"]


class SafeWeightedLSEBase:
    """Mixin providing safe weighted-LSE math for online TD algorithms.

    Subclasses must call ``_safe_init(schedule, n_base)`` from their
    ``__init__`` before using any safe methods.

    The mixin composes a :class:`SafeWeightedCommon` instance (``_swc``)
    which holds all instrumentation fields (``last_stage``,
    ``last_beta_used``, ``last_rho``, etc.) and core math.
    """

    def _safe_init(self, schedule: BetaSchedule, n_base: int) -> None:
        """Initialise safe operator state.

        Parameters
        ----------
        schedule : BetaSchedule
            Calibrated per-stage beta schedule.
        n_base : int
            Number of base states (used to decode stage from augmented state).
        """
        self._schedule = schedule
        self._n_base = int(n_base)
        gamma = float(schedule.gamma)
        self._swc = SafeWeightedCommon(
            schedule=schedule, gamma=gamma, n_base=n_base,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def _stage_from_state(self, state: int) -> int:
        """Decode stage t from augmented state integer.

        Delegates to ``SafeWeightedCommon.stage_from_augmented_state``.
        """
        return self._swc.stage_from_augmented_state(int(np.asarray(state).flat[0]))

    def _safe_target(self, reward: float, v_next: float, state: int) -> float:
        """Compute safe target g_t^safe(reward, v_next) at stage from state.

        Also populates all ``_swc.last_*`` instrumentation fields.

        Parameters
        ----------
        reward : float
            Immediate reward r.
        v_next : float
            Bootstrap value (max_a Q[s',a] for Q-learning, E_pi[Q[s',:]]
            for ExpectedSARSA, E_pi[V(s')] for TD(0), etc.).
        state : int
            Augmented state encoding ``t * n_base + base_state``.

        Returns
        -------
        float
            Safe one-step target g_t^safe(r, v_next).
        """
        t = self._stage_from_state(state)
        return self._swc.compute_safe_target(float(reward), float(v_next), t)

    @property
    def swc(self) -> SafeWeightedCommon:
        """Public accessor for the SafeWeightedCommon instance.

        Useful for reading instrumentation fields after an update:
        ``agent.swc.last_beta_used``, ``agent.swc.last_rho``, etc.
        """
        return self._swc
