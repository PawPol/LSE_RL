"""Delayed-reward chain (Phase VII spec §6.4).

A deterministic chain of ``chain_length`` states. ``forward`` advances the
index by +1 (clamped at the terminal-reward state); ``reset_or_stay``
returns to state 0. Reward of ``terminal_reward`` is paid exactly once on
entering the terminal state. No phase shifts; canonical sign for the
``wrong_sign`` ablation is ``+β`` (rewards optimistic propagation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


class DelayedChain(Environment):
    """Delayed-reward chain (Phase VII spec §6.4)."""

    env_canonical_sign: str = "+"

    def __init__(
        self,
        chain_length: int = 20,
        horizon: int = 25,
        terminal_reward: float = 50.0,
        step_reward: float = 0.0,
        gamma: float = 0.95,
        seed: Optional[int] = None,
    ):
        if chain_length < 2:
            raise ValueError(f"chain_length must be >= 2, got {chain_length}")
        if horizon < chain_length:
            # Without enough horizon to cover the chain, the terminal reward
            # is unreachable even for the oracle policy.
            raise ValueError(
                f"horizon ({horizon}) must be >= chain_length ({chain_length})"
            )

        self._chain_length = int(chain_length)
        self._terminal_state = self._chain_length - 1
        self._horizon = int(horizon)
        self._terminal_reward = float(terminal_reward)
        self._step_reward = float(step_reward)

        # Seeded RNG retained for API symmetry with the hazard env, even
        # though dynamics are fully deterministic.
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._state: Optional[np.ndarray] = None
        self._step_in_episode: int = 0
        self._episode_index: int = -1  # incremented on each reset

        observation_space = spaces.Discrete(self._chain_length)
        action_space = spaces.Discrete(2)
        mdp_info = MDPInfo(
            observation_space=observation_space,
            action_space=action_space,
            gamma=float(gamma),
            horizon=self._horizon,
        )
        super().__init__(mdp_info)

    # ---------- Phase VII API surface ----------

    @property
    def current_phase(self) -> str:
        return "static"

    def oracle_action(self) -> Optional[int]:
        return 0  # forward, regardless of state

    def is_shift_step(self) -> bool:
        return False

    # ---------- MushroomRL Environment API ----------

    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, state=None):
        if state is None:
            self._state = np.array([0], dtype=np.int64)
        else:
            s = int(np.asarray(state).flat[0])
            if not 0 <= s < self._chain_length:
                raise ValueError(
                    f"reset state {s} outside chain [0, {self._chain_length})"
                )
            self._state = np.array([s], dtype=np.int64)

        self._step_in_episode = 0
        self._episode_index += 1
        info = {
            "phase": self.current_phase,
            "is_shift_step": self.is_shift_step(),
            "oracle_action": self.oracle_action(),
            "catastrophe": False,
            "terminal_success": False,
        }
        return self._state, info

    def step(self, action):
        if self._state is None:
            raise RuntimeError("step() called before reset()")

        a = int(np.asarray(action).flat[0])
        s = int(np.asarray(self._state).flat[0])

        if a == 0:  # forward
            next_s = min(s + 1, self._terminal_state)
        elif a == 1:  # reset_or_stay
            next_s = 0
        else:
            raise ValueError(f"invalid action {a}; expected 0 or 1")

        reached_terminal = next_s == self._terminal_state
        reward = (
            self._terminal_reward if reached_terminal else self._step_reward
        )

        self._step_in_episode += 1
        horizon_exhausted = self._step_in_episode >= self._horizon
        absorbing = bool(reached_terminal or horizon_exhausted)

        info = {
            "phase": self.current_phase,
            "is_shift_step": self.is_shift_step(),
            "oracle_action": self.oracle_action(),
            "catastrophe": False,
            "terminal_success": bool(reached_terminal),
        }

        self._state = np.array([next_s], dtype=np.int64)
        return self._state, reward, absorbing, info
