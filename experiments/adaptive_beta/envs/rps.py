"""Adversarial Rock-Paper-Scissors environment (Phase VII M2.1).

Spec authority: ``docs/specs/phase_VII_adaptive_beta.md`` §6.1 (env interface)
and §22.2 (MushroomRL ``Environment`` subclass), §22.3 (no canonical sign
for RPS — ``wrong_sign`` / ``adaptive_magnitude_only`` are not defined here).

Design summary
--------------
- Horizon = 20 actions per episode.
- Action set: ``{0: rock, 1: paper, 2: scissors}``.
- Standard RPS payoff: +1 win, 0 tie, -1 loss (from the agent's POV).
- Opponent has a *hidden* (default) or *visible* phase that cycles every
  ``switch_period_episodes`` episodes through:

    1. ``biased_exploitable``: rock 80%, paper 10%, scissors 10%.
       Best-response = paper.
    2. ``counter_exploit``: scissors 80%, rock 10%, paper 10%.
       Best-response = rock. (One-liner: opponent assumes the agent is
       still exploiting phase 1 by playing paper, so it preempts with
       scissors; the agent's best counter is rock.)
    3. ``uniform_random``: 1/3 each. No best-response (oracle returns
       ``None`` — every action ties expected reward).

- Phase transitions occur **between** episodes (never mid-episode).
- ``env_canonical_sign = None`` (spec §22.3).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces

# Phase ID strings (single source of truth for the phase column / oracle
# best-response table).
PHASE_BIASED_EXPLOITABLE = "biased_exploitable"
PHASE_COUNTER_EXPLOIT = "counter_exploit"
PHASE_UNIFORM_RANDOM = "uniform_random"

_PHASE_CYCLE: Tuple[str, ...] = (
    PHASE_BIASED_EXPLOITABLE,
    PHASE_COUNTER_EXPLOIT,
    PHASE_UNIFORM_RANDOM,
)

# Per-phase opponent action distribution over {0:rock, 1:paper, 2:scissors}.
_OPPONENT_DIST: Dict[str, np.ndarray] = {
    PHASE_BIASED_EXPLOITABLE: np.array([0.8, 0.1, 0.1], dtype=np.float64),
    PHASE_COUNTER_EXPLOIT: np.array([0.1, 0.1, 0.8], dtype=np.float64),
    PHASE_UNIFORM_RANDOM: np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
}

# Best-response action under the *true* opponent distribution. ``None`` for
# uniform_random because every action has identical expected payoff.
_PHASE_ORACLE: Dict[str, Optional[int]] = {
    PHASE_BIASED_EXPLOITABLE: 1,   # paper beats rock (the dominant opponent action)
    PHASE_COUNTER_EXPLOIT: 0,      # rock beats scissors (the dominant opponent action)
    PHASE_UNIFORM_RANDOM: None,
}


def _rps_reward(agent_action: int, opponent_action: int) -> float:
    """Standard RPS payoff from the agent's perspective.

    Returns +1 if the agent beats the opponent, -1 on loss, 0 on tie.
    Encoding: 0=rock, 1=paper, 2=scissors. ``(agent - opponent) mod 3 == 1``
    means the agent's action beats the opponent's; ``mod 3 == 2`` means loss.
    """
    diff = (agent_action - opponent_action) % 3
    if diff == 0:
        return 0.0
    if diff == 1:
        return 1.0
    return -1.0


class RPS(Environment):
    """Adversarial Rock-Paper-Scissors with cycling opponent phases.

    Parameters
    ----------
    horizon
        Episode length (number of agent actions per episode). Default 20.
    switch_period_episodes
        Number of episodes per opponent phase. Spec §6.1 allows ``50`` or
        ``100``; default 100. Phase transitions are episode-aligned, so a
        switch at episode boundary ``e`` takes effect from episode ``e``
        onwards (and ``is_shift_step()`` returns True only at step 0 of
        that episode).
    visible_phase
        If True, the env's single state encodes the current phase index
        (0/1/2) — used for an easy-diagnostic ablation (spec §6.1, "visible-
        phase variant"). If False (default), the state is a fixed dummy
        ``0`` and the agent must learn from action sequences alone.
    gamma
        Discount factor passed into ``MDPInfo``. Default 0.95.
    seed
        Optional integer seed. The opponent RNG for episode ``e`` is derived
        deterministically from ``SeedSequence([seed, e])`` so re-running an
        episode reproduces the same opponent sequence (spec §13.3 repro).
        ``None`` falls back to nondeterministic seeding (use only in
        unit-test smoke checks).
    """

    # Spec §22.3: RPS has no canonical sign — wrong_sign / magnitude-only
    # methods fail-fast when constructed against this env.
    env_canonical_sign: Optional[str] = None

    # Action encoding (kept as class constants for downstream consumers).
    ROCK: int = 0
    PAPER: int = 1
    SCISSORS: int = 2

    def __init__(
        self,
        horizon: int = 20,
        switch_period_episodes: int = 100,
        visible_phase: bool = False,
        gamma: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        if switch_period_episodes not in (50, 100):
            # Spec §6.1 fixes the allowed values; loud failure beats a silent
            # off-grid run that nobody can reproduce later.
            raise ValueError(
                f"switch_period_episodes must be 50 or 100 (spec §6.1), "
                f"got {switch_period_episodes}"
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self._horizon: int = int(horizon)
        self._switch_period: int = int(switch_period_episodes)
        self._visible_phase: bool = bool(visible_phase)
        self._seed: Optional[int] = None if seed is None else int(seed)

        # State-space cardinality: 1 dummy state when phase is hidden, 3
        # phase-indexed states when visible.
        n_states = 3 if self._visible_phase else 1
        observation_space = spaces.Discrete(n_states)
        action_space = spaces.Discrete(3)
        mdp_info = MDPInfo(observation_space, action_space, gamma, self._horizon)
        super().__init__(mdp_info)

        # Episode bookkeeping (private mutable state; never expose to agent).
        self._episode_index: int = 0
        self._step_in_episode: int = 0
        self._state: np.ndarray = np.array([0])
        # Per-episode opponent RNG — re-seeded deterministically in reset().
        self._opponent_rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Spec §6 contract surface
    # ------------------------------------------------------------------
    @property
    def current_phase(self) -> str:
        """Return the opponent phase string for the current episode."""
        return _PHASE_CYCLE[self._episode_index // self._switch_period % len(_PHASE_CYCLE)]

    def oracle_action(self) -> Optional[int]:
        """Best-response action under the true opponent distribution.

        Returns ``None`` for ``uniform_random`` (no strict best-response —
        every action ties expected reward at 0).
        """
        return _PHASE_ORACLE[self.current_phase]

    def is_shift_step(self) -> bool:
        """True iff this is step 0 of an episode immediately following a
        phase switch (i.e. ``episode_index > 0`` and the episode_index is a
        multiple of ``switch_period``, and we are at step 0). The very
        first phase of a fresh run does *not* emit a shift event.
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

        Re-seeds the opponent RNG from ``SeedSequence([seed, episode_index])``
        so re-running an episode reproduces the same opponent sequence.
        ``state`` argument is accepted for MushroomRL API compatibility but
        is ignored except for jumping to a specific phase index in the
        ``visible_phase`` variant.
        """
        # Re-seed the opponent RNG deterministically from (seed, episode).
        # SeedSequence with a list-form entropy gives stable, composable seeds.
        if self._seed is not None:
            ss = np.random.SeedSequence([self._seed, self._episode_index])
            self._opponent_rng = np.random.default_rng(ss)
        else:
            # Nondeterministic fallback — unit-test smoke only.
            self._opponent_rng = np.random.default_rng()

        self._step_in_episode = 0

        if self._visible_phase:
            # Phase-indexed state: 0/1/2 corresponding to the current phase.
            phase_idx = _PHASE_CYCLE.index(self.current_phase)
            self._state = np.array([phase_idx])
        else:
            # Single-dummy-state hidden-phase variant.
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
        # Normalize numpy state scalar per tasks/lessons.md.
        agent_action: int = int(np.asarray(action).flat[0])
        if agent_action not in (0, 1, 2):
            raise ValueError(
                f"RPS action must be in {{0, 1, 2}}, got {agent_action}"
            )

        phase = self.current_phase
        opponent_dist = _OPPONENT_DIST[phase]
        opponent_action: int = int(self._opponent_rng.choice(3, p=opponent_dist))

        reward: float = _rps_reward(agent_action, opponent_action)

        # Advance step counter; check terminal.
        self._step_in_episode += 1
        absorbing: bool = self._step_in_episode >= self._horizon

        # Build info BEFORE we advance the episode counter, so reported
        # `phase` etc. correspond to the just-completed transition.
        info: Dict = {
            "phase": phase,
            # is_shift_step latches only on step 0 of a post-switch episode;
            # during a step() call we are necessarily past step 0, so this is
            # always False for step-time info dicts.
            "is_shift_step": False,
            "oracle_action": _PHASE_ORACLE[phase],
            "catastrophe": False,
            # RPS has no goal terminal — every episode-ending transition is
            # just horizon-exhaustion.
            "terminal_success": False,
            # Useful diagnostic for downstream loggers; not part of the
            # mandatory key set but harmless and zero-cost.
            "opponent_action": opponent_action,
        }

        if absorbing:
            # Episode is ending. Increment the episode counter so the next
            # reset() picks up the correct phase / RNG.
            self._episode_index += 1

        # Update the observation for the visible-phase variant — this only
        # matters when the *next* episode starts a new phase, but the
        # convention is to update the cached state here for consistency.
        if self._visible_phase:
            phase_idx = _PHASE_CYCLE.index(self.current_phase)
            self._state = np.array([phase_idx])
        else:
            self._state = np.array([0])

        return self._state, reward, absorbing, info
