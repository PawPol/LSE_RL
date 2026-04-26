"""Gridworld with adversarial hazards (Phase VII spec §6.3).

A 7×7 gridworld in which a fixed number of hazard cells reshuffle every
``hazard_switch_period_episodes``. The agent receives ``goal_reward`` on
entering the goal, ``hazard_reward`` on entering a hazard cell (terminal,
catastrophe), and ``step_reward`` per step otherwise. Canonical sign for
the ``wrong_sign`` ablation is ``-β`` (rewards pessimistic propagation
around hazards).

Encoding
--------
- State: ``row * grid_size + col`` as an int64 in ``[0, grid_size**2)``.
- Action: ``0=up, 1=right, 2=down, 3=left``.
- Walls clamp moves: out-of-bounds moves keep the agent in place.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


# (drow, dcol) deltas for actions {up, right, down, left}.
_ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)


class HazardGridworld(Environment):
    """Gridworld with periodically-reshuffling hazards (Phase VII spec §6.3)."""

    env_canonical_sign: str = "-"

    def __init__(
        self,
        grid_size: int = 7,
        horizon: int = 50,
        num_hazards: int = 5,
        goal_reward: float = 10.0,
        hazard_reward: float = -10.0,
        step_reward: float = -0.01,
        hazard_switch_period_episodes: Literal[100, 250] = 250,
        gamma: float = 0.95,
        seed: Optional[int] = None,
    ):
        if grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {grid_size}")
        if hazard_switch_period_episodes not in (100, 250):
            raise ValueError(
                "hazard_switch_period_episodes must be 100 or 250, "
                f"got {hazard_switch_period_episodes}"
            )

        self._grid_size = int(grid_size)
        self._horizon = int(horizon)
        self._num_hazards = int(num_hazards)
        self._goal_reward = float(goal_reward)
        self._hazard_reward = float(hazard_reward)
        self._step_reward = float(step_reward)
        self._switch_period = int(hazard_switch_period_episodes)

        self._start_rc: tuple[int, int] = (0, 0)
        self._goal_rc: tuple[int, int] = (
            self._grid_size - 1,
            self._grid_size - 1,
        )

        # Sanity: enough non-{start, goal} cells to host the hazards while
        # honoring the corridor constraint.
        n_cells = self._grid_size * self._grid_size
        if self._num_hazards >= n_cells - 2:
            raise ValueError(
                f"num_hazards={self._num_hazards} too large for "
                f"{self._grid_size}x{self._grid_size} grid"
            )

        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._hazards: frozenset[tuple[int, int]] = frozenset()
        self._state: Optional[np.ndarray] = None
        self._step_in_episode: int = 0
        self._episode_index: int = -1  # incremented on each reset
        # Set during reset() to capture whether THIS episode begins on a
        # period boundary (needed because is_shift_step() is queried at
        # step() time, not reset time).
        self._is_shift_episode_start: bool = False

        observation_space = spaces.Discrete(n_cells)
        action_space = spaces.Discrete(4)
        mdp_info = MDPInfo(
            observation_space=observation_space,
            action_space=action_space,
            gamma=float(gamma),
            horizon=self._horizon,
        )
        super().__init__(mdp_info)

        # Initial layout (episode 0).
        self._reshuffle_hazards()

    # ---------- Phase VII API surface ----------

    @property
    def current_phase(self) -> str:
        # The phase is identified by the layout epoch (period index). This
        # is what diagnostics need to track recovery after a switch.
        return f"epoch_{max(self._episode_index, 0) // self._switch_period}"

    def oracle_action(self) -> Optional[int]:
        if self._state is None:
            return None
        s = int(np.asarray(self._state).flat[0])
        rc = self._decode(s)
        return self._oracle_action_from_rc(rc)

    def is_shift_step(self) -> bool:
        # True iff the *current* step is the first step of an episode that
        # itself begins a new layout epoch (and is not the very first
        # episode). Re-evaluated at step time using cached reset-time flag.
        return bool(self._is_shift_episode_start and self._step_in_episode == 0)

    # ---------- MushroomRL Environment API ----------

    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Re-roll the layout under the new seed; downstream callers that
        # call seed() typically intend a deterministic restart.
        self._episode_index = -1
        self._reshuffle_hazards()

    def reset(self, state=None):
        self._episode_index += 1
        # Detect period boundary (skip episode 0 — there's nothing to
        # "shift" from).
        if (
            self._episode_index > 0
            and self._episode_index % self._switch_period == 0
        ):
            self._is_shift_episode_start = True
            self._reshuffle_hazards()
        else:
            self._is_shift_episode_start = False

        if state is None:
            s = self._encode(self._start_rc)
        else:
            s = int(np.asarray(state).flat[0])
            if not 0 <= s < self._grid_size * self._grid_size:
                raise ValueError(f"reset state {s} outside grid")

        self._state = np.array([s], dtype=np.int64)
        self._step_in_episode = 0
        info = {
            "phase": self.current_phase,
            "is_shift_step": self._is_shift_episode_start,
            "oracle_action": self._oracle_action_from_rc(self._decode(s)),
            "catastrophe": False,
            "terminal_success": False,
        }
        return self._state, info

    def step(self, action):
        if self._state is None:
            raise RuntimeError("step() called before reset()")

        # Capture shift-step flag for *this* step before we mutate the
        # internal step counter.
        info_is_shift = self.is_shift_step()

        a = int(np.asarray(action).flat[0])
        if not 0 <= a < 4:
            raise ValueError(f"invalid action {a}; expected 0..3")

        s = int(np.asarray(self._state).flat[0])
        rc = self._decode(s)
        drow, dcol = _ACTION_DELTAS[a]
        new_r = max(0, min(self._grid_size - 1, rc[0] + drow))
        new_c = max(0, min(self._grid_size - 1, rc[1] + dcol))
        next_rc = (new_r, new_c)

        reached_goal = next_rc == self._goal_rc
        entered_hazard = next_rc in self._hazards

        if reached_goal:
            reward = self._goal_reward
        elif entered_hazard:
            reward = self._hazard_reward
        else:
            reward = self._step_reward

        self._step_in_episode += 1
        horizon_exhausted = self._step_in_episode >= self._horizon
        absorbing = bool(reached_goal or entered_hazard or horizon_exhausted)

        next_s = self._encode(next_rc)
        self._state = np.array([next_s], dtype=np.int64)

        info = {
            "phase": self.current_phase,
            "is_shift_step": info_is_shift,
            "oracle_action": self._oracle_action_from_rc(next_rc),
            "catastrophe": bool(entered_hazard),
            "terminal_success": bool(reached_goal),
        }
        return self._state, reward, absorbing, info

    # ---------- Internals ----------

    def _encode(self, rc: tuple[int, int]) -> int:
        return rc[0] * self._grid_size + rc[1]

    def _decode(self, s: int) -> tuple[int, int]:
        return (s // self._grid_size, s % self._grid_size)

    def _on_corridor(self, rc: tuple[int, int]) -> bool:
        """Manhattan-shortest-path corridor: cells with row<=goal_row AND
        col<=goal_col, measured from start=(0,0)."""
        return rc[0] <= self._goal_rc[0] and rc[1] <= self._goal_rc[1]

    def _reshuffle_hazards(self) -> None:
        """Sample a hazard layout that (i) excludes start and goal, and
        (ii) places at least one hazard on the manhattan-corridor."""
        forbidden = {self._start_rc, self._goal_rc}
        all_cells = [
            (r, c)
            for r in range(self._grid_size)
            for c in range(self._grid_size)
            if (r, c) not in forbidden
        ]
        corridor_cells = [c for c in all_cells if self._on_corridor(c)]
        # The corridor always has interior cells given grid_size>=3 and the
        # start/goal at opposite corners, so at least one hazard fits.
        if not corridor_cells:
            raise RuntimeError("no corridor cells available for hazards")

        # Sample until the constraint is satisfied. With 5 hazards on a
        # 7x7 grid (corridor covers ~49 - 2 cells), one shot almost always
        # suffices; bound retries for safety under degenerate sizes.
        for _ in range(64):
            idx = self._rng.choice(
                len(all_cells), size=self._num_hazards, replace=False
            )
            picks = [all_cells[i] for i in idx]
            if any(p in corridor_cells for p in picks):
                self._hazards = frozenset(picks)
                return

        # Deterministic fallback: pin one hazard onto the corridor and
        # fill the rest from the remainder.
        forced = corridor_cells[
            int(self._rng.integers(0, len(corridor_cells)))
        ]
        remaining = [c for c in all_cells if c != forced]
        idx = self._rng.choice(
            len(remaining), size=self._num_hazards - 1, replace=False
        )
        self._hazards = frozenset([forced, *(remaining[i] for i in idx)])

    def _oracle_action_from_rc(self, rc: tuple[int, int]) -> int:
        """Greedy manhattan-toward-goal with adjacent-hazard avoidance.

        Considers the four primitive actions in canonical order
        ``{up, right, down, left}``. For each, computes the resulting cell
        (clamped at walls) and rejects it if (a) the cell is a hazard, or
        (b) any 4-neighbor of the cell is a hazard. Among the survivors,
        picks the one minimizing manhattan distance to the goal; ties go
        to the lower action int. If no safe action survives, returns 0.
        """
        gr, gc = self._goal_rc
        best_action = None
        best_dist = None
        for a in range(4):
            drow, dcol = _ACTION_DELTAS[a]
            nr = max(0, min(self._grid_size - 1, rc[0] + drow))
            nc = max(0, min(self._grid_size - 1, rc[1] + dcol))
            cand = (nr, nc)
            if cand in self._hazards:
                continue
            adjacent_hazard = any(
                (
                    max(0, min(self._grid_size - 1, nr + ar)),
                    max(0, min(self._grid_size - 1, nc + ac)),
                )
                in self._hazards
                for (ar, ac) in _ACTION_DELTAS
            )
            if adjacent_hazard:
                continue
            dist = abs(nr - gr) + abs(nc - gc)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_action = a
        if best_action is None:
            return 0
        return best_action
