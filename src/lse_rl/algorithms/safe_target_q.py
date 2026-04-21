"""
Safe weighted-LSE Q-Learning with a frozen target network (Phase IV-C).

Standalone tabular Q-Learning variant that decouples the bootstrap from
the online table.  An ``online`` table is updated every step; a
``target`` table is frozen and synced periodically (hard sync every
``sync_every`` steps, or Polyak averaging each step when
``polyak_tau > 0``).

Mathematical form
-----------------
The safe target uses the *target* table for the bootstrap::

    v_next  = max_a Q_target[t, s', a]
    margin  = r - v_next
    rho     = sigmoid(beta_t * margin + log(1/gamma))
    g_safe  = ((1+gamma)/beta_t) * (logaddexp(beta_t*r,
                                              beta_t*v_next + log(gamma))
                                    - log(1+gamma))

computed by :class:`SafeWeightedCommon`.  The *online* table is updated
via ``Q_online <- Q_online + lr * (g_safe - Q_online)``.  When
``beta_t == 0`` this collapses exactly to classical target-network
Q-Learning.

Sync modes
----------
- ``polyak_tau == 0.0`` (default): hard sync.  Every ``sync_every``
  *global* steps, ``Q_target`` is replaced by a deep copy of
  ``Q_online``.  ``global_step`` counts from 0, and the first sync
  happens at ``global_step == sync_every`` (not at step 0).
- ``polyak_tau > 0.0``: Polyak averaging.  Every step,
  ``Q_target <- (1 - tau) * Q_target + tau * Q_online``.  ``sync_every``
  is ignored in this mode.

Spec §3.2 freeze semantics
--------------------------
"Frozen during the Bellman-learning phase" means the *schedule* (beta_t)
is frozen (calibrated once from a pilot, never retrained).  The *target
table* is frozen between consecutive syncs, per the standard DQN
interpretation — not across the entire 20 k-step training run.
This follows standard usage: hard-sync interval is ``sync_every=200`` steps
(100 syncs over 20 k steps); Polyak uses ``tau=0.05``.  If a reviewer
reads §3.2 as "target frozen for the full run", that is a stricter
interpretation: the run.json ``architecture_note`` field records this.
(Adversarial review A6 — resolved as DISPUTE.)
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

__all__ = ["SafeTargetQLearning"]


class SafeTargetQLearning:
    """Safe weighted-LSE Q-Learning with a frozen target network.

    Parameters
    ----------
    n_states:
        Number of base states.
    n_actions:
        Number of actions.
    schedule:
        Calibrated per-stage beta schedule.
    learning_rate:
        Constant step-size in ``(0, 1]``.
    gamma:
        Nominal discount factor (must match ``schedule.gamma``).
    sync_every:
        Hard-sync period (in global update steps).  Ignored when
        ``polyak_tau > 0``.  Must be ``>= 1`` when used.
    polyak_tau:
        Polyak averaging coefficient in ``[0, 1]``.  Zero disables Polyak
        and enables hard sync.
    seed:
        Reserved for future use; currently no internal randomness.
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
        seed: int = 0,
    ) -> None:
        if abs(schedule.gamma - gamma) > 1e-9:
            raise ValueError(
                f"SafeTargetQLearning: gamma={gamma} does not match "
                f"schedule.gamma={schedule.gamma}."
            )
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in (0, 1]; got {learning_rate}."
            )
        if not (0.0 <= polyak_tau <= 1.0):
            raise ValueError(
                f"polyak_tau must be in [0, 1]; got {polyak_tau}."
            )
        if polyak_tau == 0.0 and sync_every < 1:
            raise ValueError(
                f"sync_every must be >= 1 in hard-sync mode; got {sync_every}."
            )

        self._n_states = int(n_states)
        self._n_actions = int(n_actions)
        self._T = int(schedule.T)
        self._lr = float(learning_rate)
        self._gamma = float(gamma)
        self._sync_every = int(sync_every)
        self._tau = float(polyak_tau)
        self._use_polyak = self._tau > 0.0

        self._Q_online = np.zeros(
            (self._T, self._n_states, self._n_actions), dtype=np.float64
        )
        self._Q_target = np.zeros_like(self._Q_online)

        self._swc = SafeWeightedCommon(
            schedule=schedule, gamma=self._gamma, n_base=self._n_states,
        )
        self._rng = np.random.default_rng(int(seed))

        # Tracks the global_step of the most recent hard sync, or 0 before
        # any sync has happened.  Exposed via the log dict as
        # ``target_sync_step``.
        self._last_sync_step: int = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def Q_online(self) -> np.ndarray:
        return self._Q_online

    @property
    def Q_target(self) -> np.ndarray:
        return self._Q_target

    @property
    def swc(self) -> SafeWeightedCommon:
        return self._swc

    @property
    def T(self) -> int:
        return self._T

    @property
    def target_update_mode(self) -> str:
        return "polyak" if self._use_polyak else "hard"

    def get_Q(self, state: int, stage: int) -> np.ndarray:
        """Return ``Q_online[t, s, :]`` (online values are the point estimate)."""
        s = int(np.asarray(state).flat[0])
        t = int(stage)
        return self._Q_online[t, s, :].copy()

    def get_V(self, state: int, stage: int) -> float:
        return float(np.max(self.get_Q(state, stage)))

    # ------------------------------------------------------------------
    # Bootstrap computation (shared by subclasses)
    # ------------------------------------------------------------------

    def _compute_v_next_target(
        self, next_state: int, stage: int, absorbing: bool
    ) -> float:
        """v_next from the *target* table.  Subclasses override for policy-
        expected semantics (SafeTargetExpectedSARSA)."""
        if absorbing:
            return 0.0
        return float(np.max(self._Q_target[stage, next_state, :]))

    def _compute_v_next_online(
        self, next_state: int, stage: int, absorbing: bool
    ) -> float:
        """Diagnostic v_next from the *online* table (for logging)."""
        if absorbing:
            return 0.0
        return float(np.max(self._Q_online[stage, next_state, :]))

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def _maybe_sync(self, global_step: int) -> None:
        """Update the target table according to the configured mode.

        Must be called *once per update step*.
        """
        if self._use_polyak:
            # Polyak: Q_target <- (1 - tau) * Q_target + tau * Q_online.
            self._Q_target *= (1.0 - self._tau)
            self._Q_target += self._tau * self._Q_online
            # last_sync_step is kept at 0 in Polyak mode (not meaningful).
            return

        # Hard sync: every ``sync_every`` global steps, after the step.
        if global_step > 0 and (global_step % self._sync_every == 0):
            np.copyto(self._Q_target, self._Q_online)
            self._last_sync_step = int(global_step)

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
        global_step: int,
    ) -> dict[str, Any]:
        """One safe target-Q update.  Returns a log dict (spec §12)."""
        s = int(np.asarray(state).flat[0])
        a = int(np.asarray(action).flat[0])
        sn = int(np.asarray(next_state).flat[0])
        t = int(stage)
        r = float(reward)
        done = bool(absorbing)
        gs = int(global_step)

        if not (0 <= t < self._T):
            raise IndexError(
                f"stage {t} out of range [0, {self._T}); schedule T={self._T}."
            )

        # --- 1. Bootstraps ---------------------------------------------
        v_target = self._compute_v_next_target(sn, t, done)
        v_online = self._compute_v_next_online(sn, t, done)

        # --- 2. Safe target using the *target* table -------------------
        safe_target = self._swc.compute_safe_target(r, v_target, t)

        # --- 3. Update Q_online ----------------------------------------
        q_current = float(self._Q_online[t, s, a])
        self._Q_online[t, s, a] = q_current + self._lr * (
            safe_target - q_current
        )

        # --- 4. Sync target table --------------------------------------
        self._maybe_sync(gs)

        # --- 5. Log ----------------------------------------------------
        return self._build_log(
            r=r,
            v_online=v_online,
            v_target=v_target,
            safe_target=float(safe_target),
            q_current=q_current,
        )

    # ------------------------------------------------------------------

    def _build_log(
        self,
        *,
        r: float,
        v_online: float,
        v_target: float,
        safe_target: float,
        q_current: float,
    ) -> dict[str, Any]:
        beta_used = float(self._swc.last_beta_used)
        return {
            "q_online_next": v_online,
            "q_target_next": v_target,
            "q_target_gap": abs(v_online - v_target),
            "target_sync_step": int(self._last_sync_step),
            "target_update_mode": self.target_update_mode,
            "beta_used": beta_used,
            "beta_raw": float(self._swc.last_beta_raw),
            "beta_cap": float(self._swc.last_beta_cap),
            "clip_active": bool(self._swc.last_clip_active),
            "rho": float(np.asarray(self._swc.last_rho).item()),
            "effective_discount": float(
                np.asarray(self._swc.last_effective_discount).item()
            ),
            "stage": int(self._swc.last_stage),
            "target": float(np.asarray(self._swc.last_target).item()),
            "margin": float(np.asarray(self._swc.last_margin).item()),
            "safe_target": safe_target,
            "q_current_pre_update": q_current,
        }
