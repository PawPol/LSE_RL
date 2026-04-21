"""
Safe weighted-LSE Double Q-Learning (Phase IV-C).

Standalone tabular Double Q-Learning with the safe weighted-LSE one-step
target.  Two Q-tables ``Q_A`` and ``Q_B`` are maintained.  On each update
a fair coin chooses the *selection* table (used to pick the greedy next
action) and the *evaluation* table (used to bootstrap the value of that
action).  The selection table is the one whose entry is updated; the
evaluation table's bootstrap enters the safe target.

Mathematical form
-----------------
Let ``v_next = Q_eval[s', argmax_a Q_sel[s', a]]``.  The safe target is::

    margin = r - v_next
    rho    = sigmoid(beta_t * margin + log(1/gamma))
    g_safe = ((1+gamma)/beta_t) * (logaddexp(beta_t*r,
                                             beta_t*v_next + log(gamma))
                                   - log(1+gamma))

which is computed via ``SafeWeightedCommon.compute_safe_target`` (using
the numerically stable ``np.logaddexp`` formulation).  When ``beta_t == 0``
the target collapses exactly to ``r + gamma * v_next`` and the algorithm
reduces to classical Double Q-Learning.

Table shapes
------------
``Q_A`` and ``Q_B`` have shape ``(T, n_states, n_actions)`` where ``T``
is the horizon length read from the :class:`BetaSchedule`.  The caller
passes ``(state, stage)`` separately; no augmented-state decoding is done
inside this class.  The Phase IV-C runner is responsible for mapping
MushroomRL's augmented state ``t * n_base + s`` to ``(state=s, stage=t)``
before invoking :meth:`update`.
"""
from __future__ import annotations

import sys
import pathlib
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap so the safe-operator math layer is importable whether
# lse_rl is installed or imported directly from a source checkout.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
if str(_MUSHROOM_DEV) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM_DEV))

from mushroom_rl.algorithms.value.dp.safe_weighted_common import (  # noqa: E402
    BetaSchedule,
    SafeWeightedCommon,
)

__all__ = ["SafeDoubleQLearning"]


class SafeDoubleQLearning:
    """Safe weighted-LSE Double Q-Learning.

    Parameters
    ----------
    n_states:
        Number of base (un-augmented) states.  Each Q-table has shape
        ``(T, n_states, n_actions)``.
    n_actions:
        Number of actions.
    schedule:
        Calibrated per-stage beta schedule.  ``schedule.T`` determines the
        stage dimension of the Q-tables; ``schedule.gamma`` is the nominal
        discount factor.
    learning_rate:
        Constant step-size in ``(0, 1]``.  The safe update is
        ``Q <- Q + lr * (g_safe - Q)``.
    gamma:
        Nominal discount factor.  Must match ``schedule.gamma``.
    seed:
        Integer seed for the internal selection-coin RNG.  Two agents with
        the same seed produce identical coin sequences.
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
                f"SafeDoubleQLearning: gamma={gamma} does not match "
                f"schedule.gamma={schedule.gamma}."
            )
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in (0, 1]; got {learning_rate}."
            )
        self._n_states = int(n_states)
        self._n_actions = int(n_actions)
        self._T = int(schedule.T)
        self._lr = float(learning_rate)
        self._gamma = float(gamma)
        self._schedule = schedule

        # Two Q-tables, shape (T, S, A).
        self._Q_A = np.zeros((self._T, self._n_states, self._n_actions),
                             dtype=np.float64)
        self._Q_B = np.zeros_like(self._Q_A)

        # Math layer.  ``n_base`` is required by the constructor but only
        # used by ``stage_from_augmented_state`` which we do not call here
        # (the caller passes ``stage`` explicitly).  Pass ``n_states`` for
        # self-consistency; safe.
        self._swc = SafeWeightedCommon(
            schedule=schedule, gamma=self._gamma, n_base=self._n_states,
        )

        # Deterministic RNG for the selection coin.
        self._rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def Q_A(self) -> np.ndarray:
        """Q-table A, shape ``(T, n_states, n_actions)``."""
        return self._Q_A

    @property
    def Q_B(self) -> np.ndarray:
        """Q-table B, shape ``(T, n_states, n_actions)``."""
        return self._Q_B

    @property
    def swc(self) -> SafeWeightedCommon:
        """Underlying math helper (exposes ``last_*`` instrumentation)."""
        return self._swc

    @property
    def T(self) -> int:
        """Horizon length (number of stages)."""
        return self._T

    def get_Q(self, state: int, stage: int) -> np.ndarray:
        """Return the averaged Q-vector ``0.5 * (Q_A + Q_B)`` at ``(state, stage)``.

        The averaged table is the point estimate for greedy action
        selection at evaluation time.
        """
        s = int(np.asarray(state).flat[0])
        t = int(stage)
        return 0.5 * (self._Q_A[t, s, :] + self._Q_B[t, s, :])

    def get_V(self, state: int, stage: int) -> float:
        """Return ``max_a 0.5 * (Q_A[t,s,a] + Q_B[t,s,a])``."""
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
        """Perform one safe Double-Q update and return a log dict.

        Parameters
        ----------
        state, action, next_state:
            Integer indices into the base state / action spaces.  Scalars,
            0-d numpy arrays, and shape-(1,) arrays are all accepted per
            the ``flat[0]`` convention.
        reward:
            Immediate reward.
        absorbing:
            If ``True``, the bootstrap value is zero (terminal next state).
        stage:
            Current stage ``t in [0, T)`` at which the transition was
            observed.  The safe target uses ``schedule.beta_used_at(t)``.

        Returns
        -------
        dict
            Per-step instrumentation (see module docstring and spec §12).
        """
        s = int(np.asarray(state).flat[0])
        a = int(np.asarray(action).flat[0])
        sn = int(np.asarray(next_state).flat[0])
        t = int(stage)
        r = float(reward)
        done = bool(absorbing)

        if not (0 <= t < self._T):
            raise IndexError(
                f"stage {t} out of range [0, {self._T}); "
                f"schedule T={self._T}."
            )

        # --- 1. Flip the selection coin --------------------------------
        use_A_as_selector = bool(self._rng.integers(0, 2) == 0)
        if use_A_as_selector:
            Q_sel = self._Q_A
            Q_eval = self._Q_B
            sel_source = "A"
            eval_source = "B"
        else:
            Q_sel = self._Q_B
            Q_eval = self._Q_A
            sel_source = "B"
            eval_source = "A"

        # --- 2. Bootstrap values ----------------------------------------
        # "Diagnostic" greedy values from each table at s'.  These are
        # used for logging (double_gap, q_a_next, q_b_next) and for the
        # classical Double Q semantics.
        if done:
            q_a_next = 0.0
            q_b_next = 0.0
            v_next = 0.0
            a_star = -1
        else:
            # Greedy action selected by the *selection* table.
            q_sel_next = Q_sel[t, sn, :]
            # Tie-break uniformly at random over argmax set (mirrors
            # mushroom_rl.algorithms.value.td.double_q_learning).
            max_val = float(np.max(q_sel_next))
            argmax_set = np.flatnonzero(q_sel_next == max_val)
            a_star = int(self._rng.choice(argmax_set))
            # Evaluation-side bootstrap: Q_eval at (s', a_star).
            v_next = float(Q_eval[t, sn, a_star])
            # Per-table max values (for logging only).
            q_a_next = float(np.max(self._Q_A[t, sn, :]))
            q_b_next = float(np.max(self._Q_B[t, sn, :]))

        # --- 3. Safe target via the math layer --------------------------
        safe_target = self._swc.compute_safe_target(r, v_next, t)

        # --- 4. Update the SELECTION table ------------------------------
        q_current = float(Q_sel[t, s, a])
        Q_sel[t, s, a] = q_current + self._lr * (safe_target - q_current)

        # --- 5. Build log dict -----------------------------------------
        # double_gap: |Q_A[s', a*] - Q_B[s', a*]| at the selected action.
        if done or a_star < 0:
            double_gap = 0.0
        else:
            double_gap = abs(
                float(self._Q_A[t, sn, a_star])
                - float(self._Q_B[t, sn, a_star])
            )

        margin_double = r - v_next
        beta_used = self._swc.last_beta_used
        natural_shift_double = float(beta_used) * margin_double

        return {
            # double-Q specific
            "q_a_next": q_a_next,
            "q_b_next": q_b_next,
            "selected_action_source": sel_source,
            "evaluation_value_source": eval_source,
            "double_gap": double_gap,
            "margin_double": margin_double,
            "natural_shift_double": natural_shift_double,
            "safe_target_double": float(safe_target),
            # standard safe instrumentation (spec §12)
            "beta_used": float(beta_used),
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
            # update-side bookkeeping
            "a_star": int(a_star),
            "q_current_pre_update": q_current,
        }
