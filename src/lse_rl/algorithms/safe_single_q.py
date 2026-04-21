"""
Safe weighted-LSE Single Q-Learning (Phase IV-C attribution baseline).

Identical architecture to the hand-rolled Phase IV-C training loop but
with one Q-table.  Used as the architectural control for isolating
estimator-stabilization gains:

    safe_q_single_table     ← this class (one table, same loop)
    safe_double_q           ← two tables, evaluation-side bootstrap
    safe_target_q           ← one online + one frozen target table

Any performance difference between safe_q_single_table and the advanced
estimators is attributable to the estimator mechanism, not to the training
loop or schedule architecture.

Mathematical form
-----------------
Standard Q-Learning update with the safe weighted-LSE one-step target::

    v_next  = max_a Q[t, s', a]   (or 0 if absorbing)
    g_safe  = ((1+γ)/β_t) * (logaddexp(β_t*r, β_t*v_next + log γ) - log(1+γ))

At β_t = 0 this collapses to r + γ * v_next (classical Q-Learning).
"""
from __future__ import annotations

import sys
import pathlib
from typing import Any

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
if str(_MUSHROOM_DEV) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM_DEV))

from mushroom_rl.algorithms.value.dp.safe_weighted_common import (  # noqa: E402
    BetaSchedule,
    SafeWeightedCommon,
)

__all__ = ["SafeSingleQLearning"]


class SafeSingleQLearning:
    """Safe weighted-LSE Q-Learning with a single Q-table.

    This is the architectural control for the Phase IV-C attribution
    analysis.  It uses the same hand-rolled training loop and safe Bellman
    operator as SafeDoubleQLearning / SafeTargetQLearning, but maintains
    only one Q-table and performs standard one-table Q-Learning updates.

    Parameters
    ----------
    n_states:
        Number of base (un-augmented) states.
    n_actions:
        Number of actions.
    schedule:
        Calibrated per-stage beta schedule.
    learning_rate:
        Constant step-size in (0, 1].
    gamma:
        Nominal discount factor (must match schedule.gamma).
    seed:
        Unused (present for interface compatibility with other agents).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        schedule: BetaSchedule,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        seed: int = 0,
    ) -> None:
        if abs(schedule.gamma - gamma) > 1e-9:
            raise ValueError(
                f"SafeSingleQLearning: gamma={gamma} does not match "
                f"schedule.gamma={schedule.gamma}."
            )
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1]; got {learning_rate}.")
        self._n_states = int(n_states)
        self._n_actions = int(n_actions)
        self._T = int(schedule.T)
        self._lr = float(learning_rate)
        self._gamma = float(gamma)
        self._schedule = schedule

        self._Q = np.zeros((self._T, self._n_states, self._n_actions), dtype=np.float64)

        self._swc = SafeWeightedCommon(
            schedule=schedule, gamma=self._gamma, n_base=self._n_states,
        )
        # seed arg kept for interface parity; not used (no stochastic selection)
        self._seed = int(seed)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def Q(self) -> np.ndarray:
        """Single Q-table, shape ``(T, n_states, n_actions)``."""
        return self._Q

    @property
    def swc(self) -> SafeWeightedCommon:
        return self._swc

    @property
    def T(self) -> int:
        return self._T

    def get_Q(self, state: int, stage: int) -> np.ndarray:
        """Return Q-vector at (state, stage)."""
        return self._Q[int(stage), int(np.asarray(state).flat[0]), :]

    def get_V(self, state: int, stage: int) -> float:
        return float(np.max(self.get_Q(state, stage)))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        absorbing: bool,
        stage: int,
    ) -> dict[str, Any]:
        """One safe single-table Q-Learning update.

        Returns
        -------
        dict
            Per-step instrumentation with the same keys as
            SafeDoubleQLearning where applicable (beta_used, rho,
            effective_discount, margin, td_error, target, clip_active).
        """
        s = int(np.asarray(state).flat[0])
        a = int(np.asarray(action).flat[0])
        sn = int(np.asarray(next_state).flat[0])
        t = int(stage)
        r = float(reward)
        done = bool(absorbing)

        if not (0 <= t < self._T):
            raise IndexError(
                f"stage {t} out of range [0, {self._T}); schedule T={self._T}."
            )

        # Bootstrap: greedy value from the single table
        if done:
            v_next = 0.0
        else:
            v_next = float(np.max(self._Q[t, sn, :]))

        # Safe target via math layer
        safe_target = self._swc.compute_safe_target(r, v_next, t)

        # Standard Q-Learning update
        q_current = float(self._Q[t, s, a])
        td_error = float(safe_target) - q_current
        self._Q[t, s, a] = q_current + self._lr * td_error

        beta_used = float(self._swc.last_beta_used)
        margin = float(np.asarray(self._swc.last_margin).item())

        return {
            "beta_used": beta_used,
            "beta_raw": float(self._swc.last_beta_raw),
            "beta_cap": float(self._swc.last_beta_cap),
            "clip_active": bool(self._swc.last_clip_active),
            "rho": float(np.asarray(self._swc.last_rho).item()),
            "effective_discount": float(
                np.asarray(self._swc.last_effective_discount).item()
            ),
            "stage": t,
            "target": float(np.asarray(self._swc.last_target).item()),
            "margin": margin,
            "natural_shift": beta_used * margin,
            "td_error": td_error,
            "q_current_pre_update": q_current,
            "v_next": v_next,
        }
