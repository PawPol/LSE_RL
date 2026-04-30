"""Soda / Uncertain Game (Phase VIII spec §5.5).

A hidden-payoff-type m-action repeated game. The active payoff matrix is
governed by a hidden type ``ξ ∈ {coordination, anti_coordination,
zero_sum, biased_preference}``; for the four "fixed-type" subcases the
type is constant within a run, and for ``SO-TypeSwitch`` it rotates on
a per-episode clock. The hidden type is exposed to the environment's
per-step ``info`` dict via ``info["regime"]``. Per spec §6.6 only the
``OracleBetaSchedule`` is permitted to read this field; non-oracle
schedules MUST NOT branch on it.

Subcases (spec §5.5)
--------------------

- ``SO-Coordination`` — symmetric pure-coordination payoff.
  ``payoff[i, j] = 1 if i == j else 0`` (both players).
- ``SO-AntiCoordination`` — symmetric anti-coordination payoff.
  ``payoff[i, j] = 0 if i == j else 1`` (both players).
- ``SO-ZeroSum`` — zero-sum on action-difference (mod m). The agent
  earns ``+1`` when ``(i − j) mod m == 1``, ``−1`` when
  ``(i − j) mod m == m − 1``, and ``0`` elsewhere; the opponent earns
  the negation. Reduces to matching-pennies-flavoured zero-sum at m = 2
  and to a rock-paper-scissors-flavoured zero-sum at m = 3.
- ``SO-BiasedPreference`` — biased symmetric coordination favouring
  action 0:
  ``payoff[0, 0] = 2``, ``payoff[i, i] = 1`` for ``i > 0``, else ``0``
  (both players). Action 0 is the payoff-dominant equilibrium.
- ``SO-TypeSwitch`` — hidden ``ξ_t`` cycles deterministically through
  ``["coordination", "anti_coordination", "zero_sum",
  "biased_preference"]`` with period ``switch_period_episodes`` measured
  in episodes. Determinism: the regime in episode ``e`` is
  ``types[(e // switch_period_episodes) % 4]``, computed from the env's
  ``_episode_index`` only. No env RNG draw is used to pick ``ξ_t``.

Action encoding
---------------

Actions are integer indices in ``[0, m)``. m is configurable via the
``m`` factory kwarg (default 3). Action labels are not assigned (the
game is symmetric across actions for SO-Coordination,
SO-AntiCoordination, SO-ZeroSum); for SO-BiasedPreference action 0 is
the payoff-dominant action.

Canonical sign
--------------

``None`` (regime-dependent). Coordination subcases would prefer ``+``,
zero-sum and anti-coordination prefer ``None``, and the switching
subcase has no single canonical direction. We tag the env with
``env_canonical_sign = None`` so that the schedule factory raises if
``wrong_sign`` / ``adaptive_magnitude_only`` are instantiated against
this env (spec §6.1, parent Phase VII §22.3 precedent).

References
----------

- Phase VIII spec §5.5 (Soda contract), §5 (info["regime"] schema),
  §6.6 (oracle reads info["regime"]).
- ``tasks/lessons.md`` #1 (numpy state extraction with ``flat[0]``);
  #11 (no ``expm1``/``log1p``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.matrix_game import (
    MatrixGameEnv,
    StateEncoder,
    make_default_state_encoder,
)
from experiments.adaptive_beta.strategic_games.registry import register_game


GAME_NAME: str = "soda_uncertain"

# Canonical regime ordering for SO-TypeSwitch (spec §5.5). The four
# fixed-type names are also the legal values of ``info["regime"]`` for
# any subcase.
_REGIMES_FIXED: Tuple[str, ...] = (
    "coordination",
    "anti_coordination",
    "zero_sum",
    "biased_preference",
)

# Subcase id (spec §5.5) → canonical regime string. ``SO-TypeSwitch`` has
# no static regime; its current value is computed per-episode below.
_SUBCASE_TO_REGIME: Dict[str, str] = {
    "SO-Coordination":      "coordination",
    "SO-AntiCoordination":  "anti_coordination",
    "SO-ZeroSum":           "zero_sum",
    "SO-BiasedPreference":  "biased_preference",
}

VALID_SUBCASES: Tuple[str, ...] = tuple(_SUBCASE_TO_REGIME.keys()) + (
    "SO-TypeSwitch",
)


def _coordination_payoffs(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric pure-coordination matrices, shape ``(m, m)``.

    ``P[i, j] = 1`` iff ``i == j`` else ``0``; both players use the
    same matrix.
    """
    p = np.eye(m, dtype=np.float64)
    return p, p.copy()


def _anti_coordination_payoffs(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric anti-coordination matrices, shape ``(m, m)``.

    ``P[i, j] = 0`` iff ``i == j`` else ``1``; both players use the
    same matrix.
    """
    p = np.ones((m, m), dtype=np.float64) - np.eye(m, dtype=np.float64)
    return p, p.copy()


def _zero_sum_payoffs(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-sum on action-difference (mod m), shape ``(m, m)``.

    Agent reward ``+1`` when ``(i − j) mod m == 1``, ``−1`` when
    ``(i − j) mod m == m − 1``, and ``0`` elsewhere. Opponent reward
    is the negation. For m = 2 the table reduces to ``[[0, -1], [1, 0]]``
    (diagonal zero, off-diagonal sign-pair); for m = 3 it reduces to a
    rock-paper-scissors-shaped zero-sum. The construction is well-defined
    for m >= 2; for m == 1 the matrix is trivially zero (no off-diagonal
    cells).
    """
    pa = np.zeros((m, m), dtype=np.float64)
    if m >= 2:
        for i in range(m):
            for j in range(m):
                d = (i - j) % m
                if d == 1:
                    pa[i, j] = 1.0
                elif d == m - 1:
                    pa[i, j] = -1.0
                # else: 0
    po = -pa
    return pa, po


def _biased_preference_payoffs(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Biased symmetric coordination favouring action 0, shape ``(m, m)``.

    ``P[0, 0] = 2``; ``P[i, i] = 1`` for ``i > 0``; ``0`` off-diagonal.
    Both players use the same matrix.
    """
    p = np.zeros((m, m), dtype=np.float64)
    if m >= 1:
        p[0, 0] = 2.0
    for i in range(1, m):
        p[i, i] = 1.0
    return p, p.copy()


# Regime name → payoff-builder. Used by both the static-subcase factory
# branch and the SO-TypeSwitch per-episode rotator.
_REGIME_BUILDERS = {
    "coordination":       _coordination_payoffs,
    "anti_coordination":  _anti_coordination_payoffs,
    "zero_sum":           _zero_sum_payoffs,
    "biased_preference":  _biased_preference_payoffs,
}


class SodaUncertainEnv(MatrixGameEnv):
    """Soda / Uncertain game env (spec §5.5).

    Thin subclass of ``MatrixGameEnv`` that

    1. injects ``info["regime"]`` into both ``reset`` and ``step`` info
       dicts so that the oracle β schedule can read ``ξ_t`` (spec §6.6),
    2. for the ``SO-TypeSwitch`` subcase, rotates the active payoff
       matrices at the start of each episode according to a deterministic
       per-episode clock keyed off ``self._episode_index``.

    The non-oracle methods do not see ``ξ_t``: the state encoder is the
    default ``(timestep, prev_opponent_action)`` encoder (no regime in
    the observation), and only the ``info`` channel surfaces the regime.
    """

    def __init__(
        self,
        *,
        subcase: str,
        m: int,
        switch_period_episodes: int,
        adversary: StrategicAdversary,
        horizon: int,
        state_encoder: StateEncoder,
        n_states: int,
        seed: Optional[int],
        metadata: Dict[str, Any],
        gamma: float,
    ) -> None:
        if subcase not in VALID_SUBCASES:
            raise ValueError(
                f"unknown subcase {subcase!r}; valid: {VALID_SUBCASES}"
            )
        if m < 2:
            raise ValueError(
                f"m must be >= 2 for soda_uncertain; got {m}"
            )
        if switch_period_episodes < 1:
            raise ValueError(
                "switch_period_episodes must be >= 1; got "
                f"{switch_period_episodes}"
            )

        # Build the initial payoff matrices. For static subcases the
        # payoffs are fixed for the run; for SO-TypeSwitch we install the
        # type-0 (``coordination``) matrices and rotate at reset time.
        initial_regime = (
            _REGIMES_FIXED[0]
            if subcase == "SO-TypeSwitch"
            else _SUBCASE_TO_REGIME[subcase]
        )
        pa0, po0 = _REGIME_BUILDERS[initial_regime](m)

        super().__init__(
            payoff_agent=pa0,
            payoff_opponent=po0,
            adversary=adversary,
            horizon=horizon,
            state_encoder=state_encoder,
            n_states=n_states,
            seed=seed,
            game_name=GAME_NAME,
            metadata=metadata,
            gamma=gamma,
        )

        self._subcase: str = subcase
        self._m: int = int(m)
        self._switch_period_episodes: int = int(switch_period_episodes)
        # The four-element rotation cycle for SO-TypeSwitch.
        self._regime_cycle: Tuple[str, ...] = _REGIMES_FIXED
        self._current_regime: str = initial_regime
        # Canonical sign is regime-dependent; the schedule factory must
        # raise on ``wrong_sign`` / ``adaptive_magnitude_only`` against
        # this env. ``MatrixGameEnv.env_canonical_sign`` is a class
        # attribute defaulting to ``None``; we set it explicitly on the
        # instance to make the contract explicit.
        self.env_canonical_sign = None

    # ------------------------------------------------------------------
    # Hidden-type accounting
    # ------------------------------------------------------------------
    def _regime_for_episode(self, episode_index: int) -> str:
        """Return the regime active in the given episode index.

        For the four static subcases this is constant; for
        ``SO-TypeSwitch`` it rotates on a deterministic clock keyed off
        ``episode_index`` (no RNG draw).
        """
        if self._subcase != "SO-TypeSwitch":
            return _SUBCASE_TO_REGIME[self._subcase]
        slot = (episode_index // self._switch_period_episodes) % 4
        return self._regime_cycle[slot]

    def _install_regime_payoffs(self, regime: str) -> None:
        """Replace the env's active payoff matrices with the regime's.

        Mutates ``self._payoff_agent`` and ``self._payoff_opponent``
        in place. Only used for ``SO-TypeSwitch``; static subcases keep
        the matrices installed at construction.
        """
        pa, po = _REGIME_BUILDERS[regime](self._m)
        self._payoff_agent = pa
        self._payoff_opponent = po

    # ------------------------------------------------------------------
    # Environment overrides
    # ------------------------------------------------------------------
    def reset(
        self, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset; for SO-TypeSwitch also rotate the active payoff matrices.

        The regime swap happens BEFORE ``super().reset`` so the
        adversary / state-encoder / history machinery all see the new
        episode boundary at the same time.
        """
        regime = self._regime_for_episode(self._episode_index)
        if self._subcase == "SO-TypeSwitch":
            self._install_regime_payoffs(regime)
        self._current_regime = regime

        obs, info = super().reset(state=state)
        info["regime"] = regime
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the env; inject ``info["regime"]`` for the just-completed step.

        ``self._current_regime`` is set in ``reset`` and is the regime
        active for THIS episode. ``MatrixGameEnv.step`` increments
        ``self._episode_index`` only on the absorbing step, so the
        in-episode regime is stable.
        """
        obs, reward, absorbing, info = super().step(action)
        info["regime"] = self._current_regime
        return obs, reward, absorbing, info


def build(
    *,
    subcase: str,
    adversary: StrategicAdversary,
    horizon: int = 20,
    m: int = 3,
    switch_period_episodes: int = 100,
    seed: Optional[int] = None,
    state_encoder: Optional[StateEncoder] = None,
    **kwargs: Any,
) -> MatrixGameEnv:
    """Construct a Soda / Uncertain ``MatrixGameEnv`` (spec §5.5).

    Parameters
    ----------
    subcase
        One of ``SO-Coordination``, ``SO-AntiCoordination``,
        ``SO-ZeroSum``, ``SO-BiasedPreference``, ``SO-TypeSwitch``.
    adversary
        Pre-built ``StrategicAdversary``; ``adversary.n_actions`` must
        equal ``m``.
    horizon
        Episode length. Default 20.
    m
        Action-space cardinality. Default 3 (per spec §5.5).
    switch_period_episodes
        Episodes per regime in the ``SO-TypeSwitch`` subcase. Default
        100. Ignored for the four static subcases. Must be >= 1.
    seed
        Optional integer seed.
    state_encoder
        Optional encoder override. ``None`` selects the default
        ``make_default_state_encoder(horizon, m)`` encoder. The default
        encoder does NOT expose ``ξ_t`` to the agent — the regime is
        observable only through ``info["regime"]`` (oracle-only per
        spec §6.6).
    **kwargs
        Forwarded to ``MatrixGameEnv``. Supported: ``gamma``,
        ``metadata`` (merged on top of the game-supplied metadata),
        and ``n_states`` (only when ``state_encoder`` is overridden).
    """
    if subcase not in VALID_SUBCASES:
        raise ValueError(
            f"unknown subcase {subcase!r}; valid: {VALID_SUBCASES}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if m < 2:
        raise ValueError(f"m must be >= 2, got {m}")
    if switch_period_episodes < 1:
        raise ValueError(
            "switch_period_episodes must be >= 1, got "
            f"{switch_period_episodes}"
        )
    if adversary.n_actions != m:
        raise ValueError(
            f"adversary.n_actions={adversary.n_actions} != m={m} "
            f"(soda_uncertain)"
        )

    if state_encoder is None:
        encoder, n_states = make_default_state_encoder(
            horizon=horizon, n_actions=m
        )
    else:
        encoder = state_encoder
        n_states = int(kwargs.pop("n_states", 1))

    metadata: Dict[str, Any] = {
        "canonical_sign": None,  # regime-dependent; oracle-only access
        "mechanism_degenerate": horizon == 1,
        "is_zero_sum": subcase == "SO-ZeroSum",
        "subcase": subcase,
        "m": int(m),
        "switch_period_episodes": int(switch_period_episodes),
        "regime_cycle": _REGIMES_FIXED if subcase == "SO-TypeSwitch" else None,
        "valid_regimes": _REGIMES_FIXED,
    }
    user_meta = kwargs.pop("metadata", None)
    if user_meta:
        metadata.update(user_meta)

    gamma = float(kwargs.pop("gamma", 0.95))
    if kwargs:
        raise TypeError(
            f"build(soda_uncertain) got unexpected kwargs: {sorted(kwargs)}"
        )

    return SodaUncertainEnv(
        subcase=subcase,
        m=m,
        switch_period_episodes=switch_period_episodes,
        adversary=adversary,
        horizon=horizon,
        state_encoder=encoder,
        n_states=n_states,
        seed=seed,
        metadata=metadata,
        gamma=gamma,
    )


# Spec §4 contract: register at module-import time. The registry edit in
# ``registry.py`` (a one-line import) ensures this module is loaded when
# the registry is first consulted.
register_game(GAME_NAME, build)
