"""5-arm Bernoulli switching bandit (Phase VII M2.2).

Spec authority: ``docs/specs/phase_VII_adaptive_beta.md`` §6.2 and §22.5.

Performance-only env (spec §22.5 resolution)
--------------------------------------------
Mechanism degenerate at H=1; switching bandit is performance-only (regret,
AUC, recovery, final return). Excluded from ``alignment_rate_*.pdf`` /
``effective_discount_*.pdf`` figures and from the mechanism columns of the
main results table.

Design summary
--------------
- 5 Bernoulli arms; ``p_best = 0.8`` for the current best arm,
  ``p_other = 0.2`` for the rest.
- Horizon = 1 (every transition is terminal); single dummy state.
- Best-arm rotates cyclically through arms ``0..n_arms-1`` every
  ``switch_period_episodes`` episodes.
- ``env_canonical_sign = None`` (spec §22.3) — ``wrong_sign`` and
  ``adaptive_magnitude_only`` are not defined for this env and will
  fail-fast on construction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


class SwitchingBandit(Environment):
    """5-arm Bernoulli bandit with periodically rotating best arm.

    Per spec §6.2 + §22.5: this env is **performance-only**. With H=1,
    every transition is terminal so ``v_next = 0``, ``A_e = mean(r_t)``,
    and the alignment / effective-discount story is degenerate. Mechanism
    figures and table columns must report ``n/a — degenerate at H=1``
    for this env.

    Parameters
    ----------
    n_arms
        Number of Bernoulli arms. Default 5.
    p_best
        Bernoulli success probability of the current best arm. Default 0.8.
    p_other
        Bernoulli success probability of every non-best arm. Default 0.2.
    switch_period_episodes
        Number of episodes between best-arm rotations. Spec §6.2 allows
        ``100`` or ``250``; default 250. The best arm at episode ``e`` is
        ``(e // switch_period_episodes) % n_arms``.
    gamma
        Discount factor for ``MDPInfo``. Default 0.95. Note that with H=1
        gamma is effectively unused but is kept for API uniformity.
    seed
        Optional integer seed. Reward Bernoulli RNG is re-seeded
        deterministically per ``(seed, episode_index)`` from
        ``SeedSequence([seed, episode_index])`` (spec §13.3 repro). ``None``
        falls back to nondeterministic seeding (smoke tests only).
    """

    # Spec §22.3: switching bandit has no canonical sign. wrong_sign and
    # adaptive_magnitude_only must not be constructed against it.
    env_canonical_sign: Optional[str] = None

    def __init__(
        self,
        n_arms: int = 5,
        p_best: float = 0.8,
        p_other: float = 0.2,
        switch_period_episodes: int = 250,
        gamma: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError(f"n_arms must be >= 2, got {n_arms}")
        if not (0.0 <= p_best <= 1.0):
            raise ValueError(f"p_best must be in [0, 1], got {p_best}")
        if not (0.0 <= p_other <= 1.0):
            raise ValueError(f"p_other must be in [0, 1], got {p_other}")
        if switch_period_episodes not in (100, 250):
            # Spec §6.2 fixes the allowed values; off-grid runs are likely
            # bugs and should fail loudly.
            raise ValueError(
                f"switch_period_episodes must be 100 or 250 (spec §6.2), "
                f"got {switch_period_episodes}"
            )

        self._n_arms: int = int(n_arms)
        self._p_best: float = float(p_best)
        self._p_other: float = float(p_other)
        self._switch_period: int = int(switch_period_episodes)
        self._seed: Optional[int] = None if seed is None else int(seed)

        # Single dummy state; horizon = 1.
        observation_space = spaces.Discrete(1)
        action_space = spaces.Discrete(self._n_arms)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon=1)
        super().__init__(mdp_info)

        # Episode bookkeeping.
        self._episode_index: int = 0
        self._step_in_episode: int = 0
        self._state: np.ndarray = np.array([0])
        self._reward_rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Spec §6 contract surface
    # ------------------------------------------------------------------
    @property
    def current_phase(self) -> int:
        """Index of the current best arm (also the schema ``phase`` value)."""
        return (self._episode_index // self._switch_period) % self._n_arms

    def oracle_action(self) -> Optional[int]:
        """Best-response action — i.e. the current best arm."""
        return int(self.current_phase)

    def is_shift_step(self) -> bool:
        """True iff this is step 0 of the first episode after a best-arm
        rotation. The very first episode of a fresh run does *not* emit a
        shift event.
        """
        return (
            self._step_in_episode == 0
            and self._episode_index > 0
            and self._episode_index % self._switch_period == 0
        )

    # ------------------------------------------------------------------
    # MushroomRL Environment interface
    # ------------------------------------------------------------------
    def reset(self, state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Reset to the start of the current episode.

        Re-seeds the reward RNG from ``SeedSequence([seed, episode_index])``
        so re-running an episode produces the same Bernoulli draw. ``state``
        argument is accepted for MushroomRL API compatibility but ignored
        (single-state env).
        """
        if self._seed is not None:
            ss = np.random.SeedSequence([self._seed, self._episode_index])
            self._reward_rng = np.random.default_rng(ss)
        else:
            self._reward_rng = np.random.default_rng()

        self._step_in_episode = 0
        self._state = np.array([0])

        info: Dict = {
            "phase": self.current_phase,
            "is_shift_step": self.is_shift_step(),
            "oracle_action": self.oracle_action(),
            "catastrophe": False,
            "terminal_success": False,
        }
        return self._state, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        arm: int = int(np.asarray(action).flat[0])
        if not (0 <= arm < self._n_arms):
            raise ValueError(
                f"action must be in [0, {self._n_arms - 1}], got {arm}"
            )

        best_arm: int = int(self.current_phase)
        p: float = self._p_best if arm == best_arm else self._p_other
        # Bernoulli draw via the per-episode-seeded RNG.
        reward: float = float(self._reward_rng.random() < p)

        self._step_in_episode += 1
        absorbing: bool = True  # H=1 — every step is terminal.

        info: Dict = {
            "phase": best_arm,
            # Always False at step time: shift latches only on step 0 of the
            # post-switch episode.
            "is_shift_step": False,
            "oracle_action": best_arm,
            "catastrophe": False,
            # Bandit has no goal terminal — every episode just terminates.
            "terminal_success": False,
        }

        # Episode is ending; advance episode counter for the *next* reset()
        # to pick up the correct phase / RNG seed.
        self._episode_index += 1
        # State stays the dummy 0; recreate to avoid aliasing.
        self._state = np.array([0])

        return self._state, reward, absorbing, info
