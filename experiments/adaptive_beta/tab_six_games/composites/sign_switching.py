"""Sign-switching (G_+, G_-) composite environment for Phase VIII M9.

Spec authority
--------------
- ``docs/specs/phase_VIII_tab_six_games.md`` §10.5 (Stage 4
  sign-switching composite, oracle-validation gate).
- §6.6 (``OracleBetaSchedule`` reads ``info["regime"]`` exactly once
  per episode-end update; **non-oracle methods MUST NOT read it**).
- §5.7 (``sign_switching_regime`` adversary; the controller used in
  the standalone single-env variant — NOT used here, the composite
  routes through two complete envs instead).
- ``tasks/M9_plan.md`` §1 (canonical scope), §"Build steps" §1
  (env contract).

Role
----
Wraps a pair of pre-constructed MushroomRL matrix-game envs:

- ``env_g_plus`` — the G_+ component (TAB-positive-β beats vanilla),
  e.g. AC-Trap × finite-memory regret-matching at γ=0.60.
- ``env_g_minus`` — the G_- component (TAB-negative-β beats vanilla),
  e.g. RR-StationaryConvention × stationary_mixed at γ=0.60.

Both envs MUST have compatible state and action spaces; the
constructor asserts compatibility and refuses mismatched inputs (the
runner's responsibility is to pass a verified pair).

A hidden regime variable ξ_t ∈ {+1, -1} drives per-step routing:

- ``regime = +1`` → step routes through ``env_g_plus``;
- ``regime = -1`` → step routes through ``env_g_minus``.

ξ flips on **episode boundaries** (NOT per step) every ``dwell``
episodes, mirroring the exogenous-clock convention used by
``sign_switching_regime.py`` (§5.7). Mid-episode flips are forbidden
because matrix-game envs use intra-episode timestep encoding (the
default ``make_default_state_encoder`` packs ``step_in_episode``
into the state index); a mid-episode flip would invalidate the
encoder.

Regime-exposure contract (spec §10.5)
-------------------------------------
The env emits the *same* observation regardless of ξ. The ``regime``
attribute is exposed publicly so the runner can read it when
constructing the OracleBetaSchedule's ``episode_info`` dict at
episode boundaries, but the observation tensor itself does NOT carry
the regime. Non-oracle methods do not have access to ``env.regime``
in the runner code path — the runner only reads it inside the
oracle-method branch.

Regime label convention
-----------------------
``regime`` is exposed as a string ``"plus"`` | ``"minus"`` to match
:class:`OracleBetaSchedule`'s lookup-table convention (§6.6); the
internal numeric ξ ∈ {+1, -1} is also exposed via :attr:`regime_int`
for analysis convenience. ``regime_history`` is the list of regime
labels observed at episode-start, indexed by completed-episode count.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


# Regime label constants (mirror :data:`sign_switching_regime.REGIMES`).
_REGIME_PLUS: str = "plus"
_REGIME_MINUS: str = "minus"
_REGIMES: Tuple[str, str] = (_REGIME_PLUS, _REGIME_MINUS)


def _toggle_regime(label: str) -> str:
    if label == _REGIME_PLUS:
        return _REGIME_MINUS
    if label == _REGIME_MINUS:
        return _REGIME_PLUS
    raise ValueError(
        f"unknown regime label {label!r}; expected one of {_REGIMES}"
    )


def _regime_to_int(label: str) -> int:
    if label == _REGIME_PLUS:
        return +1
    if label == _REGIME_MINUS:
        return -1
    raise ValueError(
        f"unknown regime label {label!r}; expected one of {_REGIMES}"
    )


class SignSwitchingComposite(Environment):
    """Two-env composite that routes per-step transitions by hidden ξ_t.

    Parameters
    ----------
    env_g_plus
        Pre-constructed G_+ component env. Must already be configured
        at the desired γ_+ and adversary_+. The composite owns the
        env's lifecycle from this point on (it calls ``reset`` and
        ``step`` on the underlying env, but does NOT re-seed).
    env_g_minus
        Pre-constructed G_- component env. Same ownership contract.
    dwell
        Number of *episodes* between regime flips. Must be >= 1.
        Per spec §10.5 the M9 grid is ``{100, 250, 500, 1000}``,
        but smaller values are permitted for tests / smoke runs.
    seed
        Optional integer seed. Currently unused by the composite
        itself (regime evolution is deterministic given dwell);
        retained on the constructor surface so it matches the
        :class:`MatrixGameEnv` contract and so future stochastic
        variants (e.g. randomised dwell) can re-seed without an
        API break.
    initial_regime
        Initial ξ label, default ``"plus"``. Mirrors the
        ``sign_switching_regime`` adversary's default.

    Raises
    ------
    ValueError
        If ``env_g_plus`` and ``env_g_minus`` have incompatible
        observation- or action-space cardinalities, or if ``dwell``
        is not strictly positive.
    """

    #: Composite envs do not declare a fixed canonical sign — that's
    #: the whole point: the sign of the optimal β depends on which
    #: regime is active. None tells :class:`WrongSignSchedule` to
    #: refuse construction (which is the correct behavior — wrong-sign
    #: is undefined on a composite).
    env_canonical_sign: Optional[str] = None

    def __init__(
        self,
        env_g_plus: Environment,
        env_g_minus: Environment,
        dwell: int,
        seed: Optional[int] = None,
        initial_regime: str = _REGIME_PLUS,
    ) -> None:
        if dwell < 1:
            raise ValueError(f"dwell must be >= 1, got {dwell}")
        if initial_regime not in _REGIMES:
            raise ValueError(
                f"initial_regime must be one of {_REGIMES}, "
                f"got {initial_regime!r}"
            )

        # Validate state/action-space compatibility.
        info_plus: MDPInfo = env_g_plus.info
        info_minus: MDPInfo = env_g_minus.info
        os_plus = info_plus.observation_space
        os_minus = info_minus.observation_space
        as_plus = info_plus.action_space
        as_minus = info_minus.action_space

        # Both AC and RR use Discrete(n_states); compare the
        # ``size`` tuple (Discrete carries a 1-tuple in MushroomRL).
        os_plus_size = tuple(np.asarray(os_plus.size).tolist())
        os_minus_size = tuple(np.asarray(os_minus.size).tolist())
        as_plus_size = tuple(np.asarray(as_plus.size).tolist())
        as_minus_size = tuple(np.asarray(as_minus.size).tolist())

        if os_plus_size != os_minus_size:
            raise ValueError(
                "env_g_plus.observation_space.size "
                f"({os_plus_size}) must match env_g_minus.observation_space.size "
                f"({os_minus_size}). The M9 composite expects observation-space "
                "compatibility; pad the smaller env's encoder if needed."
            )
        if as_plus_size != as_minus_size:
            raise ValueError(
                "env_g_plus.action_space.size "
                f"({as_plus_size}) must match env_g_minus.action_space.size "
                f"({as_minus_size})."
            )

        # γ and horizon: the composite inherits γ from the active env
        # via routing; for MushroomRL's MDPInfo we report the G_+ env's
        # γ (and assert equality with G_- — the M9 spec mandates a
        # single γ across the composite, since the agent uses ONE Q
        # table with ONE γ).
        if not np.isclose(info_plus.gamma, info_minus.gamma):
            raise ValueError(
                f"env_g_plus.gamma ({info_plus.gamma}) must equal "
                f"env_g_minus.gamma ({info_minus.gamma}); the M9 "
                "composite routes through a single agent with one γ."
            )
        if int(info_plus.horizon) != int(info_minus.horizon):
            raise ValueError(
                f"env_g_plus.horizon ({info_plus.horizon}) must equal "
                f"env_g_minus.horizon ({info_minus.horizon})."
            )

        mdp_info = MDPInfo(
            observation_space=spaces.Discrete(int(os_plus_size[0])),
            action_space=spaces.Discrete(int(as_plus_size[0])),
            gamma=float(info_plus.gamma),
            horizon=int(info_plus.horizon),
        )
        super().__init__(mdp_info)

        self._env_g_plus: Environment = env_g_plus
        self._env_g_minus: Environment = env_g_minus
        self._dwell: int = int(dwell)
        self._seed: Optional[int] = None if seed is None else int(seed)
        self._initial_regime: str = initial_regime

        # Mutable state.
        self._regime: str = initial_regime
        # Episode counter: number of completed episodes since the last
        # regime flip. Flips occur when this hits ``dwell`` (i.e. after
        # ``dwell`` complete episodes inside the current regime).
        self._episodes_in_current_regime: int = 0
        self._regime_history: List[str] = []
        self._switch_count: int = 0
        # Episodes-since-switch counter, tracked per episode — useful
        # for the §8.2 ``episodes_since_switch`` and
        # ``recovery_time_per_switch`` analysis columns.
        self._episodes_since_switch: int = 0
        self._step_in_episode: int = 0
        # The most recent absorbing flag; used to gate flips so they
        # only fire on an episode boundary.
        self._just_completed_episode: bool = False

    # ------------------------------------------------------------------
    # Public introspection (read by runner + analysis)
    # ------------------------------------------------------------------
    @property
    def regime(self) -> str:
        """Current regime label ``"plus"`` | ``"minus"``.

        The OracleBetaSchedule is the ONLY caller authorised by spec
        §10.5 / §6.6 to read this. The runner's oracle method branch
        consumes it once per episode-end via ``env.regime``; non-oracle
        method branches do NOT read it.
        """
        return self._regime

    @property
    def regime_int(self) -> int:
        """Current ξ_t ∈ {+1, -1}. +1 ↔ "plus", -1 ↔ "minus"."""
        return _regime_to_int(self._regime)

    @property
    def regime_history(self) -> List[str]:
        """Per-episode regime labels (regime active at episode start).

        Length equals the number of completed episodes. Element ``e``
        is the regime label that was active at the start of episode
        ``e`` (i.e. the regime under which episode ``e`` was played).
        """
        return list(self._regime_history)

    @property
    def dwell(self) -> int:
        """Fixed dwell parameter (episodes between flips)."""
        return self._dwell

    @property
    def switch_count(self) -> int:
        """Number of regime flips that have occurred so far."""
        return int(self._switch_count)

    @property
    def episodes_since_switch(self) -> int:
        """Episodes elapsed since the most recent regime flip.

        Set to 0 after a flip, increments on each completed episode.
        Useful for the §8.2 ``episodes_since_switch`` analysis column.
        """
        return int(self._episodes_since_switch)

    @property
    def env_g_plus(self) -> Environment:
        """Read-only access to the bound G_+ component env."""
        return self._env_g_plus

    @property
    def env_g_minus(self) -> Environment:
        """Read-only access to the bound G_- component env."""
        return self._env_g_minus

    def _active_env(self) -> Environment:
        """Return the underlying env corresponding to the current regime."""
        if self._regime == _REGIME_PLUS:
            return self._env_g_plus
        return self._env_g_minus

    # ------------------------------------------------------------------
    # MushroomRL Environment interface
    # ------------------------------------------------------------------
    def reset(
        self, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the active underlying env to the start of the next episode.

        The composite does NOT cross-reset the inactive env; finite-memory
        adversaries inside both envs continue accumulating across episodes
        per §7 spec convention. The regime variable does NOT reset across
        episode boundaries; it flips deterministically every ``dwell``
        completed episodes.
        """
        # Apply pending dwell-clock flip on episode boundary, BEFORE
        # resetting the new episode's active env (so the upcoming
        # episode is played under the post-flip regime).
        if self._just_completed_episode:
            self._maybe_flip_regime()
            self._just_completed_episode = False

        active_env = self._active_env()
        state_out, info = active_env.reset(state=state)

        self._step_in_episode = 0
        # Record the regime under which THIS episode is being played.
        # Append AT EPISODE START so ``regime_history[e]`` is the
        # regime active during episode e.
        self._regime_history.append(self._regime)

        # Annotate the info dict so non-oracle code paths that
        # accidentally peek at it see the same surface as a
        # MatrixGameEnv. The ``regime`` field is present here for
        # completeness, but the runner is responsible for not
        # forwarding it to non-oracle schedules (spec §10.5 enforcement
        # is at the runner boundary, not the env boundary — the
        # composite must be capable of feeding the oracle, hence the
        # field MUST exist somewhere).
        composite_info: Dict[str, Any] = dict(info)
        composite_info["regime"] = self._regime
        composite_info["regime_int"] = self.regime_int
        composite_info["composite_episode_index"] = (
            len(self._regime_history) - 1
        )
        return state_out, composite_info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Route the step through the regime-active underlying env.

        Mid-episode regime flips are FORBIDDEN: state encoders use
        intra-episode timestep, and a mid-episode flip would mix the
        two envs' encoder state machines. The flip is deferred to the
        next ``reset`` call.
        """
        active_env = self._active_env()
        state, reward, absorbing, info = active_env.step(action)

        self._step_in_episode += 1

        # Annotate the step info with regime metadata. The ``regime``
        # field exists here in the info dict regardless of which
        # method branch is reading it — runner-side discipline (not
        # env-side concealment) enforces the spec §10.5 contract.
        composite_info: Dict[str, Any] = dict(info)
        composite_info["regime"] = self._regime
        composite_info["regime_int"] = self.regime_int
        composite_info["composite_episode_index"] = (
            len(self._regime_history) - 1
        )

        if absorbing:
            self._just_completed_episode = True
            self._episodes_in_current_regime += 1
            self._episodes_since_switch += 1

        return state, reward, absorbing, composite_info

    def seed(self, seed: int) -> None:
        """Override the composite seed and propagate to both subenvs.

        Resets per-episode bookkeeping. Both subenvs are seeded so a
        downstream call to their ``reset`` is deterministic.
        """
        self._seed = int(seed)
        for sub in (self._env_g_plus, self._env_g_minus):
            sub_seed_fn = getattr(sub, "seed", None)
            if callable(sub_seed_fn):
                sub_seed_fn(int(seed))
        self._regime = self._initial_regime
        self._episodes_in_current_regime = 0
        self._switch_count = 0
        self._episodes_since_switch = 0
        self._regime_history = []
        self._step_in_episode = 0
        self._just_completed_episode = False

    def render(self, record: bool = False) -> None:
        # No-op — matrix games are not visual.
        return None

    # ------------------------------------------------------------------
    # Internal regime-clock logic
    # ------------------------------------------------------------------
    def _maybe_flip_regime(self) -> bool:
        """If the dwell counter has reached ``dwell``, flip ξ.

        Returns True iff a flip occurred. Called from ``reset`` on
        episode boundaries (i.e. between the absorbing step and the
        next episode's first step).
        """
        if self._episodes_in_current_regime >= self._dwell:
            self._regime = _toggle_regime(self._regime)
            self._switch_count += 1
            self._episodes_in_current_regime = 0
            self._episodes_since_switch = 0
            return True
        return False
