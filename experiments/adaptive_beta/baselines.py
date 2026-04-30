"""External Q-learning baselines for Phase VIII Stage 2 / M7.

Phase VIII spec authority:
``docs/specs/phase_VIII_tab_six_games.md`` §6.3 (baseline contracts),
§13.4 (test contract for `test_baselines.py`), §13.2 (β=0 and
schedule-test discipline — these baselines have NO β, by design),
§10.3 (Stage-2 fixed-TAB vs vanilla and external baselines).

These three classes are FULL agents (not :class:`BetaSchedule`
subclasses) that exist so the paper is benchmarked against more than
β=0 / vanilla Q-learning. They mirror :class:`AdaptiveBetaQAgent`'s
public surface (``select_action``, ``begin_episode``, ``step``,
``end_episode``, ``Q``) so the Stage-2 runner can drive them through a
uniform interface.

State-extraction contract: every public method that receives a state
or action coerces it via ``int(np.asarray(x).flat[0])`` per
``tasks/lessons.md`` #28 — this is the only safe pattern across
plain Python ints, 0-d arrays, and shape-(1,) MushroomRL state arrays
(numpy ≥ 2.0 raises ``TypeError`` on the latter when called as
``int(arr)``).

Safety / divergence convention (mirroring §6.3 + the agent contract in
``agents.py``): episode-level diagnostics expose ``q_abs_max``,
``nan_count``, and a boolean ``divergence_event``. Crossing
``divergence_threshold`` or hitting NaN is reported in the diag dict.
:class:`RestartQLearningAgent` *acts* on this signal by triggering a
full table reset; the other two classes only report it.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

# Read-only import of the project's tuned linear-ε helper. The brief
# explicitly permits this (M4 W1.B). We do not edit ``agents.py``.
from experiments.adaptive_beta.agents import linear_epsilon_schedule


__all__ = [
    "RestartQLearningAgent",
    "SlidingWindowQLearningAgent",
    "TunedEpsilonGreedyQLearningAgent",
]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _as_int(x: Any) -> int:
    """Coerce a state/action scalar to a Python int.

    Handles plain Python ints, numpy scalars, 0-d arrays, and
    shape-(1,) arrays uniformly. See ``tasks/lessons.md`` #28 for
    the regression context (numpy ≥2.0 disallows ``int(shape-(1,))``).
    """
    return int(np.asarray(x).flat[0])


def _validate_common_args(
    n_states: int,
    n_actions: int,
    gamma: float,
    learning_rate: float,
) -> None:
    """Shared sanity check used by every baseline constructor."""
    if n_states <= 0 or n_actions <= 0:
        raise ValueError(
            f"n_states ({n_states}) and n_actions ({n_actions}) must be > 0"
        )
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")


# ----------------------------------------------------------------------
# 1. RestartQLearningAgent
# ----------------------------------------------------------------------
class RestartQLearningAgent:
    """Vanilla Q-learning with periodic full-table reset.

    Algorithm
    ---------
    Standard tabular Q-learning with the canonical update

    .. math::
        Q[s, a] \\leftarrow Q[s, a] + \\alpha\\,
            \\bigl(r + \\gamma\\,\\max_{a'} Q[s', a'] - Q[s, a]\\bigr).

    Action selection is ε-greedy with deterministic tie-breaking on
    the lowest action index (``np.argmax`` returns the first
    occurrence of the maximum).

    Restart trigger (spec §6.3, Addendum §6)
    ----------------------------------------
    A *full* Q-table reset (back to ``q_init``) is fired at
    :meth:`begin_episode` (i.e. between episodes) when ANY of the
    following holds, evaluated using the rolling mean of episode
    returns over the last ``restart_window`` completed episodes:

    1. ``rolling_mean_return < best_rolling_mean_return - restart_drop``
       (rolling-return drop relative to the all-time best). Only
       evaluated once at least ``restart_window`` episodes have been
       completed (otherwise rolling stats are not yet meaningful).
    2. Any NaN appears in the Q-table.
    3. ``q_abs_max > divergence_threshold``.

    On reset:

    - The Q-table is rewritten to ``q_init``.
    - The all-time-best rolling-mean tracker is cleared (the agent
      starts fresh; a new "best" is established after the next window
      fills).
    - ``restart_event`` is reported True for the *episode in which the
      reset happened* (i.e. in the diag dict returned by
      :meth:`end_episode` of that episode), to make the event easy to
      mark on a learning curve.

    Hyperparameters
    ---------------
    ``restart_window=200`` and ``restart_drop=0.5``
        Defaults from the M4 brief. ``restart_window`` is the number
        of completed episodes the rolling-mean return is computed
        over; ``restart_drop`` is the absolute return drop (in the
        same units as the env reward) that triggers a reset.
    ``divergence_threshold=1.0e6``
        Matches :class:`AdaptiveBetaQAgent`'s default (agents.py).
    ``q_init=0.0``
        Initial Q-table value. Reset on trigger restores this.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_schedule: Callable[[int], float],
        rng: np.random.Generator,
        restart_window: int = 200,
        restart_drop: float = 0.5,
        divergence_threshold: float = 1.0e6,
        q_init: float = 0.0,
    ) -> None:
        _validate_common_args(n_states, n_actions, gamma, learning_rate)
        if restart_window <= 0:
            raise ValueError(
                f"restart_window must be > 0, got {restart_window}"
            )
        if restart_drop <= 0.0:
            raise ValueError(
                f"restart_drop must be > 0, got {restart_drop}"
            )
        if divergence_threshold <= 0.0:
            raise ValueError(
                f"divergence_threshold must be > 0, got {divergence_threshold}"
            )

        self._n_states = int(n_states)
        self._n_actions = int(n_actions)
        self._gamma = float(gamma)
        self._lr = float(learning_rate)
        self._epsilon_schedule = epsilon_schedule
        self._rng = rng

        self._restart_window = int(restart_window)
        self._restart_drop = float(restart_drop)
        self._divergence_threshold = float(divergence_threshold)
        self._q_init = float(q_init)

        self._Q: np.ndarray = np.full(
            (self._n_states, self._n_actions),
            self._q_init,
            dtype=np.float64,
        )

        # Rolling history of completed-episode returns and the
        # all-time-best rolling mean seen so far.
        self._returns_history: Deque[float] = deque(maxlen=self._restart_window)
        self._best_rolling_mean: Optional[float] = None

        # Per-episode state.
        self._current_episode: int = -1
        self._ep_return: float = 0.0
        self._ep_length: int = 0
        self._ep_divergence_event: bool = False
        self._ep_restart_event: bool = False

    # ------------------------------------------------------------------
    @property
    def Q(self) -> np.ndarray:
        return self._Q

    # ------------------------------------------------------------------
    def select_action(self, state: Any, episode_index: int) -> int:
        """ε-greedy with deterministic tie-break (lowest action index)."""
        eps = float(self._epsilon_schedule(int(episode_index)))
        if self._rng.random() < eps:
            return int(self._rng.integers(0, self._n_actions))
        s = _as_int(state)
        return int(np.argmax(self._Q[s]))

    # ------------------------------------------------------------------
    def begin_episode(self, episode_index: int) -> None:
        """Clear per-episode buffers; check + execute restart trigger.

        The restart predicate is evaluated *here* (at episode start)
        so that the diag from :meth:`end_episode` of the *prior*
        episode reports ``divergence_event``, and the diag of *this*
        episode reports ``restart_event=True`` for the episode that
        first runs on the freshly-reset table.
        """
        self._current_episode = int(episode_index)
        self._ep_return = 0.0
        self._ep_length = 0
        self._ep_divergence_event = False
        self._ep_restart_event = False

        if self._should_restart():
            self._restart()
            self._ep_restart_event = True

    def step(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        absorbing: bool,
        episode_index: int,
    ) -> Dict[str, Any]:
        """Vanilla Q-learning TD update; returns per-step diag."""
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"step received episode_index={episode_index} but current "
                f"episode is {self._current_episode}; call begin_episode() "
                f"first"
            )

        s = _as_int(state)
        a = _as_int(action)
        ns = _as_int(next_state)
        r = float(reward)
        absorbing = bool(absorbing)

        # Terminal v_next is 0 (no successor on absorbing transition;
        # avoids reading a stale Q[ns] cell).
        v_next = 0.0 if absorbing else float(np.max(self._Q[ns]))
        td_target = r + self._gamma * v_next
        td_error = td_target - self._Q[s, a]
        self._Q[s, a] += self._lr * td_error

        self._ep_return += r
        self._ep_length += 1

        new_q = self._Q[s, a]
        if not np.isfinite(new_q) or not np.isfinite(td_target):
            self._ep_divergence_event = True
        elif abs(new_q) > self._divergence_threshold:
            self._ep_divergence_event = True

        q_abs_max_running = float(np.max(np.abs(self._Q)))
        return {
            "td_target": float(td_target),
            "td_error": float(td_error),
            "v_next": float(v_next),
            "reward": r,
            "q_abs_max_running": q_abs_max_running,
        }

    def end_episode(self, episode_index: int) -> Dict[str, Any]:
        """Push episode return into rolling history; return diag."""
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"end_episode received {episode_index} but current episode "
                f"is {self._current_episode}"
            )

        # Push the just-completed episode return into the rolling window.
        self._returns_history.append(self._ep_return)
        rolling_mean = (
            float(np.mean(self._returns_history))
            if len(self._returns_history) > 0
            else 0.0
        )
        # Maintain the all-time-best rolling mean *only* once the
        # window is full — partial windows give noisy "bests" that
        # would never be re-attainable and prevent drop detection.
        if len(self._returns_history) >= self._restart_window:
            if (
                self._best_rolling_mean is None
                or rolling_mean > self._best_rolling_mean
            ):
                self._best_rolling_mean = rolling_mean

        q_abs_max = float(np.max(np.abs(self._Q))) if self._Q.size else 0.0
        nan_count = int(np.isnan(self._Q).sum())
        if nan_count > 0 or q_abs_max > self._divergence_threshold:
            self._ep_divergence_event = True

        return {
            "episode_index": int(self._current_episode),
            "return": float(self._ep_return),
            "length": int(self._ep_length),
            "rolling_mean_return": rolling_mean,
            "best_rolling_mean_return": (
                float(self._best_rolling_mean)
                if self._best_rolling_mean is not None
                else float("nan")
            ),
            "q_abs_max": q_abs_max,
            "nan_count": nan_count,
            "divergence_event": bool(self._ep_divergence_event),
            "restart_event": bool(self._ep_restart_event),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _should_restart(self) -> bool:
        """Return True iff any of the three restart conditions hold."""
        # NaN trigger.
        if np.isnan(self._Q).any():
            return True
        # Divergence-magnitude trigger.
        if self._Q.size and float(np.max(np.abs(self._Q))) > self._divergence_threshold:
            return True
        # Rolling-return drop trigger (only once the window is full
        # AND a "best" has been established — otherwise there is
        # nothing to drop *from*).
        if (
            len(self._returns_history) >= self._restart_window
            and self._best_rolling_mean is not None
        ):
            rolling_mean = float(np.mean(self._returns_history))
            if rolling_mean < self._best_rolling_mean - self._restart_drop:
                return True
        return False

    def _restart(self) -> None:
        """Full Q-table reset; clear rolling-best tracker."""
        self._Q = np.full(
            (self._n_states, self._n_actions),
            self._q_init,
            dtype=np.float64,
        )
        # Re-arming the rolling-best tracker: we want the post-reset
        # agent to be free to re-establish a "best" without being
        # immediately re-triggered by a stale pre-reset benchmark.
        self._best_rolling_mean = None
        # Note: we deliberately do NOT clear ``_returns_history``;
        # keeping it lets ``rolling_mean_return`` stay continuous in
        # the diag stream and avoids a sequence of identical resets
        # at the start of every episode after the trigger fires.


# ----------------------------------------------------------------------
# 2. SlidingWindowQLearningAgent
# ----------------------------------------------------------------------
class SlidingWindowQLearningAgent:
    """Q-learning with an approximate sliding-window forgetting rule.

    Algorithm
    ---------
    Online tabular Q-learning (same TD update as
    :class:`RestartQLearningAgent`). In addition, every transition
    ``(s, a, r, s', absorbing)`` is appended to a fixed-size FIFO
    buffer of length ``window_size`` (a :class:`collections.deque`
    with ``maxlen=window_size``).

    Approximate sliding-window contract
    -----------------------------------
    The naïve "exact" sliding-window reading is to *recompute* Q from
    the entire window each episode — O(window_size · S · A) work that
    quickly dominates training. We use the simpler **state-level
    eviction** approximation documented in the M4 task brief:

    - Maintain the (s, a, r, s', absorbing) FIFO buffer of length
      ``window_size``.
    - Run the standard online TD update at every step (no replay).
    - At episode boundaries, the set of states present in the FIFO is
      re-derived. **States not present in the buffer** — i.e. states
      that have not been visited in the last ``window_size``
      transitions — have their entire row of Q reset to ``q_init``.

    This is a bounded-memory, O(window_size + S) per-episode rule
    that retains the qualitative effect of a sliding window
    (forgetting old experience) without the O(window_size · S · A)
    cost of a full replay. This approximation is documented in this
    docstring and in the dispatch handoff (M4_W1.B).

    Hyperparameters
    ---------------
    ``window_size=10000``
        Default per the M4 brief.
    ``q_init=0.0``
        Initial Q-table value; rows of unvisited states are reset
        to this.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_schedule: Callable[[int], float],
        rng: np.random.Generator,
        window_size: int = 10000,
        q_init: float = 0.0,
    ) -> None:
        _validate_common_args(n_states, n_actions, gamma, learning_rate)
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        self._n_states = int(n_states)
        self._n_actions = int(n_actions)
        self._gamma = float(gamma)
        self._lr = float(learning_rate)
        self._epsilon_schedule = epsilon_schedule
        self._rng = rng
        self._window_size = int(window_size)
        self._q_init = float(q_init)

        self._Q: np.ndarray = np.full(
            (self._n_states, self._n_actions),
            self._q_init,
            dtype=np.float64,
        )

        # FIFO of (s, a, r, s', absorbing) tuples. ``maxlen`` enforces
        # the sliding window without a manual pop step.
        self._buffer: Deque[Tuple[int, int, float, int, bool]] = deque(
            maxlen=self._window_size
        )

        self._current_episode: int = -1
        self._ep_return: float = 0.0
        self._ep_length: int = 0
        self._ep_divergence_event: bool = False

    # ------------------------------------------------------------------
    @property
    def Q(self) -> np.ndarray:
        return self._Q

    # ------------------------------------------------------------------
    def select_action(self, state: Any, episode_index: int) -> int:
        eps = float(self._epsilon_schedule(int(episode_index)))
        if self._rng.random() < eps:
            return int(self._rng.integers(0, self._n_actions))
        s = _as_int(state)
        return int(np.argmax(self._Q[s]))

    # ------------------------------------------------------------------
    def begin_episode(self, episode_index: int) -> None:
        self._current_episode = int(episode_index)
        self._ep_return = 0.0
        self._ep_length = 0
        self._ep_divergence_event = False

    def step(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        absorbing: bool,
        episode_index: int,
    ) -> Dict[str, Any]:
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"step received episode_index={episode_index} but current "
                f"episode is {self._current_episode}; call begin_episode() "
                f"first"
            )

        s = _as_int(state)
        a = _as_int(action)
        ns = _as_int(next_state)
        r = float(reward)
        absorbing = bool(absorbing)

        v_next = 0.0 if absorbing else float(np.max(self._Q[ns]))
        td_target = r + self._gamma * v_next
        td_error = td_target - self._Q[s, a]
        self._Q[s, a] += self._lr * td_error

        # Push the transition into the FIFO. ``maxlen`` evicts the
        # oldest entry automatically once the buffer is full.
        self._buffer.append((s, a, r, ns, absorbing))

        self._ep_return += r
        self._ep_length += 1

        new_q = self._Q[s, a]
        if not np.isfinite(new_q) or not np.isfinite(td_target):
            self._ep_divergence_event = True

        q_abs_max_running = float(np.max(np.abs(self._Q)))
        return {
            "td_target": float(td_target),
            "td_error": float(td_error),
            "v_next": float(v_next),
            "reward": r,
            "q_abs_max_running": q_abs_max_running,
            "buffer_size": len(self._buffer),
        }

    def end_episode(self, episode_index: int) -> Dict[str, Any]:
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"end_episode received {episode_index} but current episode "
                f"is {self._current_episode}"
            )

        # Determine which states still appear in the FIFO. Both
        # ``s`` (origin) and ``s'`` (successor) count as "visited":
        # the agent has up-to-date Q information about both endpoints
        # of every retained transition.
        if len(self._buffer) > 0:
            visited: set = set()
            for (s, _a, _r, ns, _ab) in self._buffer:
                visited.add(s)
                visited.add(ns)
            # Reset rows for states absent from the buffer.
            n_reset = 0
            for s_idx in range(self._n_states):
                if s_idx not in visited:
                    self._Q[s_idx, :] = self._q_init
                    n_reset += 1
        else:
            n_reset = 0

        q_abs_max = float(np.max(np.abs(self._Q))) if self._Q.size else 0.0
        nan_count = int(np.isnan(self._Q).sum())
        if nan_count > 0:
            self._ep_divergence_event = True

        return {
            "episode_index": int(self._current_episode),
            "return": float(self._ep_return),
            "length": int(self._ep_length),
            "buffer_size": len(self._buffer),
            "n_states_reset": int(n_reset),
            "q_abs_max": q_abs_max,
            "nan_count": nan_count,
            "divergence_event": bool(self._ep_divergence_event),
        }


# ----------------------------------------------------------------------
# 3. TunedEpsilonGreedyQLearningAgent
# ----------------------------------------------------------------------
class TunedEpsilonGreedyQLearningAgent:
    """Vanilla Q-learning with a longer / lower-floor ε-greedy schedule.

    Algorithm
    ---------
    Identical to :class:`RestartQLearningAgent` minus the restart
    machinery: standard tabular Q-learning with ε-greedy exploration
    and deterministic tie-breaking on lowest action index.

    Tuned ε schedule (default)
    --------------------------
    The "tuned" choice — fixed in the M4 brief — is

    .. code:: python

        linear_epsilon_schedule(start=1.0, end=0.01, decay_episodes=2000)

    i.e. ε starts higher (same start as the project default) but
    decays slower (2000 episodes vs the default 1000) and to a
    smaller floor (0.01 vs 0.05). This baseline therefore does more
    exploration over the run horizon than the project default and is
    a fairer comparator for adaptive-β methods that effectively
    re-explore via β shifts.

    The constructor accepts a custom ``epsilon_schedule`` so the
    Stage-2 runner can override the default if needed; passing
    ``None`` (the default) instantiates the tuned schedule above.
    """

    # Tuned schedule constants — held as class attributes so tests
    # and downstream code can assert on the canonical defaults.
    TUNED_EPSILON_START: float = 1.0
    TUNED_EPSILON_END: float = 0.01
    TUNED_EPSILON_DECAY_EPISODES: int = 2000

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_schedule: Optional[Callable[[int], float]] = None,
        rng: Optional[np.random.Generator] = None,
        q_init: float = 0.0,
    ) -> None:
        _validate_common_args(n_states, n_actions, gamma, learning_rate)
        if rng is None:
            raise ValueError(
                "rng must be provided (no implicit seed; reproducibility "
                "default per CLAUDE.md §4)"
            )

        self._n_states = int(n_states)
        self._n_actions = int(n_actions)
        self._gamma = float(gamma)
        self._lr = float(learning_rate)
        self._rng = rng
        self._q_init = float(q_init)

        if epsilon_schedule is None:
            self._epsilon_schedule = linear_epsilon_schedule(
                start=self.TUNED_EPSILON_START,
                end=self.TUNED_EPSILON_END,
                decay_episodes=self.TUNED_EPSILON_DECAY_EPISODES,
            )
        else:
            self._epsilon_schedule = epsilon_schedule

        self._Q: np.ndarray = np.full(
            (self._n_states, self._n_actions),
            self._q_init,
            dtype=np.float64,
        )

        self._current_episode: int = -1
        self._ep_return: float = 0.0
        self._ep_length: int = 0
        self._ep_divergence_event: bool = False

    # ------------------------------------------------------------------
    @property
    def Q(self) -> np.ndarray:
        return self._Q

    # ------------------------------------------------------------------
    def select_action(self, state: Any, episode_index: int) -> int:
        eps = float(self._epsilon_schedule(int(episode_index)))
        if self._rng.random() < eps:
            return int(self._rng.integers(0, self._n_actions))
        s = _as_int(state)
        return int(np.argmax(self._Q[s]))

    # ------------------------------------------------------------------
    def begin_episode(self, episode_index: int) -> None:
        self._current_episode = int(episode_index)
        self._ep_return = 0.0
        self._ep_length = 0
        self._ep_divergence_event = False

    def step(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        absorbing: bool,
        episode_index: int,
    ) -> Dict[str, Any]:
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"step received episode_index={episode_index} but current "
                f"episode is {self._current_episode}; call begin_episode() "
                f"first"
            )

        s = _as_int(state)
        a = _as_int(action)
        ns = _as_int(next_state)
        r = float(reward)
        absorbing = bool(absorbing)

        v_next = 0.0 if absorbing else float(np.max(self._Q[ns]))
        td_target = r + self._gamma * v_next
        td_error = td_target - self._Q[s, a]
        self._Q[s, a] += self._lr * td_error

        self._ep_return += r
        self._ep_length += 1

        new_q = self._Q[s, a]
        if not np.isfinite(new_q) or not np.isfinite(td_target):
            self._ep_divergence_event = True

        q_abs_max_running = float(np.max(np.abs(self._Q)))
        return {
            "td_target": float(td_target),
            "td_error": float(td_error),
            "v_next": float(v_next),
            "reward": r,
            "q_abs_max_running": q_abs_max_running,
            "epsilon": float(self._epsilon_schedule(int(episode_index))),
        }

    def end_episode(self, episode_index: int) -> Dict[str, Any]:
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"end_episode received {episode_index} but current episode "
                f"is {self._current_episode}"
            )

        q_abs_max = float(np.max(np.abs(self._Q))) if self._Q.size else 0.0
        nan_count = int(np.isnan(self._Q).sum())
        if nan_count > 0:
            self._ep_divergence_event = True

        return {
            "episode_index": int(self._current_episode),
            "return": float(self._ep_return),
            "length": int(self._ep_length),
            "epsilon": float(self._epsilon_schedule(int(self._current_episode))),
            "q_abs_max": q_abs_max,
            "nan_count": nan_count,
            "divergence_event": bool(self._ep_divergence_event),
        }
