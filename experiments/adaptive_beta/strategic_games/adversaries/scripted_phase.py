"""Scripted-phase opponent (spec §7.2).

A phase-cycle adversary that switches between configured mixed
strategies at fixed episode boundaries. Used for regression
compatibility against ``experiments/adaptive_beta/envs/rps.py`` (which
cycles through three RPS opponent distributions every
``switch_period_episodes`` episodes).

Two phase-clock modes are supported:

- ``mode='step'``  — one phase per ``switch_period`` consecutive
  ``act`` calls. Useful for environments that don't model episode
  boundaries explicitly.
- ``mode='episode'`` — phase increments only when the environment calls
  ``on_episode_end()``. This matches the existing Phase VII RPS
  scheduler exactly: phase transitions occur **between** episodes,
  never mid-episode.

Determinism: the per-phase mixed strategies are sampled via
``self._rng``, re-seeded by ``reset(seed=...)``. The phase index itself
is a deterministic function of step / episode count.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


_VALID_MODES = ("step", "episode")


class ScriptedPhaseOpponent(StrategicAdversary):
    """Cycle through a list of fixed mixed strategies with a phase clock.

    Parameters
    ----------
    phase_policies
        Sequence of probability vectors, one per phase. Each must be a
        1-D array of length ``n_actions`` summing to 1.
    switch_period
        Number of steps (or episodes, depending on ``mode``) between
        phase transitions.
    phase_names
        Optional list of human-readable phase identifiers, co-indexed
        with ``phase_policies``. Defaults to ``["phase_0", "phase_1", ...]``.
    mode
        Phase-clock mode: ``"episode"`` (matches Phase VII RPS) or
        ``"step"``.
    n_actions
        Action-space cardinality. Inferred from the first phase policy
        if not passed.
    seed
        Integer seed for the action-sampling RNG.
    """

    adversary_type: str = "scripted_phase"

    def __init__(
        self,
        phase_policies: Sequence[Sequence[float]],
        switch_period: int,
        phase_names: Optional[Sequence[str]] = None,
        mode: str = "episode",
        n_actions: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if len(phase_policies) == 0:
            raise ValueError("phase_policies must contain at least one phase")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {mode!r}"
            )
        if switch_period <= 0:
            raise ValueError(f"switch_period must be >= 1, got {switch_period}")

        # Normalise phase policies into a (n_phases, n_actions) array.
        phase_arrs: List[np.ndarray] = []
        for i, p in enumerate(phase_policies):
            arr = np.asarray(p, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    f"phase_policies[{i}] must be 1-D, got shape {arr.shape}"
                )
            if np.any(arr < 0.0):
                raise ValueError(f"phase_policies[{i}] must be non-negative")
            total = float(arr.sum())
            if not np.isfinite(total) or abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"phase_policies[{i}] must sum to 1 within 1e-9, got {total}"
                )
            phase_arrs.append(arr / arr.sum())

        n_actions_inferred = int(phase_arrs[0].shape[0])
        n = n_actions_inferred if n_actions is None else int(n_actions)
        if any(a.shape[0] != n for a in phase_arrs):
            raise ValueError(
                "All phase policies must share the same n_actions"
            )

        super().__init__(n_actions=n, seed=seed)

        # Store as a (n_phases, n_actions) ndarray for fast lookup.
        self._policies: np.ndarray = np.stack(phase_arrs, axis=0)  # (P, A)
        self._n_phases: int = self._policies.shape[0]
        self._switch_period: int = int(switch_period)
        self._mode: str = mode

        if phase_names is None:
            self._phase_names: Tuple[str, ...] = tuple(
                f"phase_{i}" for i in range(self._n_phases)
            )
        else:
            if len(phase_names) != self._n_phases:
                raise ValueError(
                    f"phase_names length {len(phase_names)} != n_phases {self._n_phases}"
                )
            self._phase_names = tuple(str(n) for n in phase_names)

        self._step_count: int = 0
        self._episode_count: int = 0

    # ------------------------------------------------------------------
    # Phase-clock helpers
    # ------------------------------------------------------------------
    @property
    def phase_index(self) -> int:
        """Current phase index in ``[0, n_phases)``."""
        if self._mode == "step":
            tick = self._step_count
        else:
            tick = self._episode_count
        return (tick // self._switch_period) % self._n_phases

    @property
    def phase_name(self) -> str:
        return self._phase_names[self.phase_index]

    def on_episode_end(self) -> None:
        """Advance the episode-mode phase clock.

        The matrix-game environment calls this at the end of each
        episode (after ``absorbing=True`` from ``step``). Step-mode
        adversaries ignore the call.
        """
        self._episode_count += 1

    # ------------------------------------------------------------------
    # ABC interface
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        # NOTE on regression compatibility with the existing RPS env:
        # the existing env re-seeds the opponent RNG per-episode from
        # ``SeedSequence([seed, episode_index])``, which gives a finer
        # level of determinism. The strategic-RPS regression test
        # (todo VII-B-25) is responsible for choosing whether to
        # mirror that scheme exactly or to compare action-frequency
        # statistics within tolerance. Here we re-seed once at adversary
        # reset and let the action stream evolve under that single
        # generator — a simpler, equally deterministic contract.
        self._step_count = 0
        self._episode_count = 0

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        probs = self._policies[self.phase_index]  # shape (n_actions,)
        return int(self._rng.choice(self.n_actions, p=probs))

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Step-mode phase clock advances once per observed step.
        self._step_count += 1

    def info(self) -> Dict[str, Any]:
        probs = self._policies[self.phase_index]
        return self._build_info(
            phase=self.phase_name,
            policy_entropy=self._entropy(probs),
        )
