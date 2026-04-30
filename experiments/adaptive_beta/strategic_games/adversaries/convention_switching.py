"""Convention-switching opponent (Phase VIII §5.7, §5.3 / RR-ConventionSwitch).

Spec authority:
- ``docs/specs/phase_VIII_tab_six_games.md`` §5.7 (adversary roster) and
  §5.3 (Rules of the Road `RR-ConventionSwitch` subcase).
- ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md`` §5.2
  (``StrategicAdversary`` ABC + ``ADVERSARY_INFO_KEYS``).

Role
----
A two-action coordination opponent that commits to one of two
"conventions" (left = action 0, right = action 1) and **periodically or
stochastically** switches between them at episode boundaries. The
agent must coordinate (Rules of the Road payoff +c match / −m
mismatch) without observing the convention directly; an oracle β
schedule may consume the regime via ``info["regime"] ∈ {"left",
"right"}`` per Phase VIII §5.7 disambiguation. Non-oracle methods
must NOT read this field (spec §6.6 / §2 rule 6).

Disambiguation vs ``ScriptedPhaseOpponent`` (Phase VIII §5.7,
2026-04-30 conflict resolution 2):

- ``ScriptedPhaseOpponent`` plays a *mixed* strategy that cycles
  through a deterministic phase clock; its phase id is NOT exposed via
  ``info["regime"]``.
- ``ConventionSwitchingOpponent`` (this module) plays a *pure*
  strategy (one action with probability 1 throughout an episode) and
  exposes the current convention via ``info["regime"]`` for the oracle
  schedule.

Determinism contract
--------------------
The ``stochastic`` mode samples a Bernoulli(``switch_prob``) flip flag
once per episode end through ``self._rng``; under a fixed seed the
flip stream is byte-identical across runs. The ``periodic`` mode is
fully deterministic (depends only on the integer episode count). The
within-episode action is a pure constant — ``act`` does NOT consume
``self._rng``.

Episode-clock semantics
-----------------------
``on_episode_end()`` is called by ``MatrixGameEnv`` after the terminal
``step`` of each episode (mirroring ``ScriptedPhaseOpponent``). The
hook increments ``_episode_count`` and applies the switch logic:

- periodic: flip the convention iff
  ``_episode_count % switch_period_episodes == 0`` (so the first flip
  happens after exactly ``switch_period_episodes`` completed
  episodes).
- stochastic: flip with probability ``switch_prob``.

The convention reported by ``act`` / ``info`` during episode ``e`` is
therefore the value that was in force *throughout* episode ``e``: the
hook's flip becomes visible only from episode ``e+1`` onward. This
matches the §2 rule 2 invariant (between-episode state changes only).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


_VALID_MODES = ("periodic", "stochastic")

# Convention index → human-readable regime label exposed via ``info["regime"]``.
# Two-action Rules of the Road convention: 0 = left, 1 = right.
_REGIME_LABELS = ("left", "right")


class ConventionSwitchingOpponent(StrategicAdversary):
    """Two-convention opponent for ``RR-ConventionSwitch`` (Phase VIII §5.7).

    Parameters
    ----------
    n_actions
        Action-space cardinality. Must equal 2 for the canonical
        Rules-of-the-Road encoding (left / right). The parameter is
        retained for ABC parity with the other adversaries; values
        other than 2 raise ``ValueError``.
    mode
        ``"periodic"`` (deterministic flip every
        ``switch_period_episodes``) or ``"stochastic"`` (Bernoulli flip
        with probability ``switch_prob`` at each episode boundary).
    switch_period_episodes
        Period of the periodic schedule, in episodes. Ignored in
        stochastic mode.
    switch_prob
        Per-episode-boundary flip probability for stochastic mode.
        Must lie in ``[0, 1]``. Ignored in periodic mode.
    initial_convention
        Starting convention, ``0`` (left) or ``1`` (right).
    seed
        Integer seed for the per-episode-end flip RNG (stochastic
        mode only). ``None`` allows non-deterministic behaviour;
        callers wanting reproducibility must pass an explicit seed.
    """

    adversary_type: str = "convention_switching"

    def __init__(
        self,
        n_actions: int = 2,
        mode: str = "periodic",
        switch_period_episodes: int = 100,
        switch_prob: float = 0.01,
        initial_convention: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        if int(n_actions) != 2:
            raise ValueError(
                f"ConventionSwitchingOpponent requires n_actions=2 "
                f"(left / right); got {n_actions}"
            )
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {mode!r}"
            )
        if int(switch_period_episodes) <= 0:
            raise ValueError(
                f"switch_period_episodes must be >= 1, got {switch_period_episodes}"
            )
        sp = float(switch_prob)
        if not np.isfinite(sp) or sp < 0.0 or sp > 1.0:
            raise ValueError(
                f"switch_prob must lie in [0, 1], got {switch_prob}"
            )
        ic = int(initial_convention)
        if ic not in (0, 1):
            raise ValueError(
                f"initial_convention must be 0 (left) or 1 (right), got {initial_convention}"
            )

        super().__init__(n_actions=int(n_actions), seed=seed)

        self._mode: str = mode
        self._switch_period_episodes: int = int(switch_period_episodes)
        self._switch_prob: float = sp
        self._initial_convention: int = ic

        # Rolling state: convention currently in force, episode counter,
        # cumulative number of flips applied.
        self._convention: int = ic
        self._episode_count: int = 0
        self._switches_so_far: int = 0

    # ------------------------------------------------------------------
    # Introspection helpers (used by tests and by the env's info pipe)
    # ------------------------------------------------------------------
    @property
    def convention(self) -> int:
        """Current convention index in ``{0, 1}``."""
        return self._convention

    @property
    def regime(self) -> str:
        """Human-readable regime label (``"left"`` or ``"right"``)."""
        return _REGIME_LABELS[self._convention]

    # ------------------------------------------------------------------
    # Episode-clock hook
    # ------------------------------------------------------------------
    def on_episode_end(self) -> None:
        """Advance the episode clock and apply the switch rule.

        The matrix-game environment calls this after the terminal
        ``step`` of each episode. The flip applied here becomes visible
        only from the *next* episode onward — within-episode actions
        and metadata remain constant (Phase VIII §2 rule 2).
        """
        self._episode_count += 1

        if self._mode == "periodic":
            if self._episode_count % self._switch_period_episodes == 0:
                self._flip()
        elif self._mode == "stochastic":
            # Bernoulli(switch_prob) draw via the seeded generator.
            if float(self._rng.random()) < self._switch_prob:
                self._flip()
        else:  # pragma: no cover — guarded at __init__
            raise RuntimeError(f"unreachable mode {self._mode!r}")

    def _flip(self) -> None:
        """Toggle the current convention and bump the switch counter."""
        self._convention = 1 - self._convention
        self._switches_so_far += 1

    # ------------------------------------------------------------------
    # ABC interface
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset rolling state to the configured initial convention.

        If ``seed`` is provided it overrides the constructor seed and
        re-seeds the flip RNG so that re-running with the same seed
        produces a byte-identical action / regime stream (Phase VIII
        §13.7 reproducibility invariant).
        """
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._convention = self._initial_convention
        self._episode_count = 0
        self._switches_so_far = 0

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        """Return the current convention as a pure action.

        Within an episode the convention is constant; ``history`` and
        ``agent_action`` are ignored (the opponent commits to its
        convention regardless of the agent's play).
        """
        return int(self._convention)

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op: no within-episode state to update.

        The convention only changes at episode boundaries via
        ``on_episode_end`` (Phase VIII §2 rule 2).
        """
        return None

    def info(self) -> Dict[str, Any]:
        """Return the §5.2 metadata block plus convention-switching extras.

        Extras (Phase VIII §5.7):

        - ``regime``           : ``"left"`` | ``"right"`` — current convention.
        - ``mode``             : ``"periodic"`` | ``"stochastic"``.
        - ``switches_so_far``  : cumulative number of flips applied.
        - ``convention_index`` : ``0`` | ``1`` — numeric form of regime.
        - ``episode_count``    : episodes completed so far (post-flip basis).
        - ``switch_period_episodes`` : configured periodic period.
        - ``switch_prob``      : configured stochastic flip probability.

        Phase metadata is set to ``"convention_switching"`` so that
        downstream metric pipelines can group rows by adversary type
        without parsing ``adversary_type``.
        """
        # The opponent plays a pure (degenerate) policy: entropy is 0.
        return self._build_info(
            phase="convention_switching",
            policy_entropy=0.0,
            regime=self.regime,
            mode=self._mode,
            switches_so_far=int(self._switches_so_far),
            convention_index=int(self._convention),
            episode_count=int(self._episode_count),
            switch_period_episodes=int(self._switch_period_episodes),
            switch_prob=float(self._switch_prob),
        )
