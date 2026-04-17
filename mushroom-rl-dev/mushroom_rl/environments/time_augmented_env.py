"""
Time-augmented environment wrappers.

These wrappers lift a base MushroomRL :class:`Environment` into an
equivalent finite-horizon MDP whose state carries the current stage
index ``t``. The underlying reward distribution and transition kernel
are unchanged (Phase I spec sections 1.3, 3.3, 4.4): we merely expose
the stage so that stage-indexed planners and non-stationary policies
can read it from the observation alone without a hidden episode
counter.

Two wrappers are provided:

* :class:`DiscreteTimeAugmentedEnv` — for environments whose
  observation space is :class:`~mushroom_rl.rl_utils.spaces.Discrete`.
  The augmented state id is ``t * n_base_states + s``.

* :class:`ContinuousTimeAugmentedEnv` — for environments whose
  observation space is :class:`~mushroom_rl.rl_utils.spaces.Box`.
  The original observation is concatenated with a single normalized
  time-to-go feature ``(horizon - 1 - t) / (horizon - 1)``.

The factory :func:`make_time_augmented` picks the right wrapper based
on the base env's observation space.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mushroom_rl.core.environment import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces

__all__ = [
    "DiscreteTimeAugmentedEnv",
    "ContinuousTimeAugmentedEnv",
    "make_time_augmented",
]


def _validate_horizon(horizon: int) -> int:
    """Ensure ``horizon`` is a finite positive integer."""
    if horizon is None:
        raise ValueError("horizon must be a finite positive integer, got None.")
    if isinstance(horizon, float):
        if not np.isfinite(horizon):
            raise ValueError(
                f"horizon must be finite; time augmentation requires a "
                f"bounded episode length, got {horizon}."
            )
        if not float(horizon).is_integer():
            raise ValueError(f"horizon must be an integer, got {horizon}.")
        horizon = int(horizon)
    if not isinstance(horizon, (int, np.integer)):
        raise TypeError(
            f"horizon must be an int, got {type(horizon).__name__}."
        )
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    return horizon


class DiscreteTimeAugmentedEnv(Environment):
    """
    Wrap a discrete MushroomRL environment and augment its state with a
    stage index ``t``.

    The augmented observation space is
    :class:`Discrete(horizon * n_base_states)`. Augmented state ids use
    the row-major encoding

    .. math::

        \\mathrm{augmented\\_id} = t \\cdot n_{\\text{base}} + s,

    so that ``s = augmented_id % n_base_states`` and
    ``t = augmented_id // n_base_states``.

    The action space and the base env's reward / transition distribution
    are preserved exactly. The stage advances by one at every
    :meth:`step` call and is reset to zero by :meth:`reset`.

    Terminal handling: at the final stage ``t == horizon - 1`` the
    ``absorbing`` flag returned by :meth:`step` is forced to ``True``
    regardless of the base env's signal, matching the finite-horizon
    convention used by the DP planners.

    Args:
        env: a MushroomRL :class:`Environment` with a
            :class:`Discrete` observation space;
        horizon: finite positive integer horizon ``T``. The wrapped env
            advertises this same horizon in its :class:`MDPInfo`.
    """

    def __init__(self, env: Environment, horizon: int) -> None:
        if not isinstance(env.info.observation_space, spaces.Discrete):
            raise TypeError(
                "DiscreteTimeAugmentedEnv requires a Discrete observation "
                f"space, got {type(env.info.observation_space).__name__}."
            )
        if not isinstance(env.info.action_space, spaces.Discrete):
            raise TypeError(
                "DiscreteTimeAugmentedEnv only supports Discrete action "
                f"spaces, got {type(env.info.action_space).__name__}."
            )

        horizon = _validate_horizon(horizon)

        self._env = env
        self._horizon = horizon
        self._n_base_states = int(env.info.observation_space.n)
        self._t: int = 0

        augmented_space = spaces.Discrete(horizon * self._n_base_states)
        mdp_info = MDPInfo(
            observation_space=augmented_space,
            action_space=env.info.action_space,
            gamma=env.info.gamma,
            horizon=horizon,
            dt=env.info.dt,
            backend=env.info.backend,
        )
        super().__init__(mdp_info)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def n_base_states(self) -> int:
        """Number of states in the base (un-augmented) environment."""
        return self._n_base_states

    @property
    def current_stage(self) -> int:
        """Current stage index ``t`` (0-indexed)."""
        return self._t

    @property
    def base_env(self) -> Environment:
        """The wrapped base environment."""
        return self._env

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_state(self, t: int, s: int) -> int:
        """
        Encode ``(t, s)`` into an augmented state id.

        Args:
            t: stage index in ``[0, horizon - 1]``;
            s: base state id in ``[0, n_base_states - 1]``.

        Returns:
            The augmented state id ``t * n_base_states + s``.
        """
        t = int(t)
        s = int(s)
        if not (0 <= t < self._horizon):
            raise ValueError(
                f"stage t={t} out of range [0, {self._horizon - 1}]."
            )
        if not (0 <= s < self._n_base_states):
            raise ValueError(
                f"base state s={s} out of range [0, {self._n_base_states - 1}]."
            )
        return t * self._n_base_states + s

    def decode_state(self, augmented_id: int) -> tuple[int, int]:
        """
        Decode an augmented state id back into ``(t, s)``.

        Args:
            augmented_id: integer in
                ``[0, horizon * n_base_states - 1]``.

        Returns:
            ``(t, s)`` with ``s = augmented_id % n_base_states`` and
            ``t = augmented_id // n_base_states``.
        """
        aid = int(augmented_id)
        total = self._horizon * self._n_base_states
        if not (0 <= aid < total):
            raise ValueError(
                f"augmented_id={aid} out of range [0, {total - 1}]."
            )
        t, s = divmod(aid, self._n_base_states)
        return t, s

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    def seed(self, seed: int) -> None:  # pragma: no cover - delegates
        """Seed the base environment (MushroomRL default is a no-op)."""
        self._env.seed(seed)

    def reset(
        self, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the wrapped env and set the stage index to zero.

        Args:
            state: optional base-env state to force the reset to. If a
                single integer (scalar or 1-D length-1 numpy array) is
                passed whose value is larger than ``n_base_states`` it
                is treated as an already-augmented id and decoded. This
                lets stage-indexed planners round-trip augmented ids
                through the wrapper.

        Returns:
            ``(augmented_state, info)`` where ``augmented_state`` is a
            1-D numpy array of length 1 containing the augmented id.
        """
        self._t = 0
        base_state_arg: np.ndarray | None = None

        if state is not None:
            arr = np.asarray(state).reshape(-1)
            if arr.size != 1:
                raise ValueError(
                    "Discrete reset expects a scalar state id, got array "
                    f"of shape {np.asarray(state).shape}."
                )
            aid = int(arr[0])
            total = self._horizon * self._n_base_states
            if 0 <= aid < self._n_base_states:
                # Treat as a raw base state id.
                base_state_arg = np.array([aid], dtype=arr.dtype)
            elif self._n_base_states <= aid < total:
                # Treat as an already-augmented id.
                t, s = self.decode_state(aid)
                self._t = t
                base_state_arg = np.array([s], dtype=arr.dtype)
            else:
                raise ValueError(
                    f"reset state id {aid} out of range "
                    f"[0, {total - 1}]."
                )

        base_state, info = self._env.reset(base_state_arg)
        s = int(np.asarray(base_state).reshape(-1)[0])
        augmented_id = self.encode_state(self._t, s)
        return np.array([augmented_id], dtype=np.int64), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance the base env by one step and increment the stage.

        Args:
            action: the action to execute, shape ``(1,)`` as per the
                base env's discrete action API.

        Returns:
            ``(augmented_next_state, reward, absorbing, info)``.
            ``absorbing`` is the base-env flag OR-ed with the terminal
            stage condition ``t_next == horizon - 1``.
        """
        next_base_state, reward, base_absorbing, info = self._env.step(action)
        self._t += 1
        t_next = self._t
        # Clamp the stage stored in the augmented id to horizon - 1 so
        # decode_state stays well-defined even if the caller keeps
        # stepping past termination (we still report absorbing=True).
        t_encoded = min(t_next, self._horizon - 1)
        s_next = int(np.asarray(next_base_state).reshape(-1)[0])
        augmented_id = self.encode_state(t_encoded, s_next)

        terminal_stage = t_next >= self._horizon - 1
        absorbing = bool(base_absorbing) or bool(terminal_stage)

        return (
            np.array([augmented_id], dtype=np.int64),
            float(reward),
            absorbing,
            info,
        )

    def render(self, record: bool = False) -> Any:  # pragma: no cover
        """Delegate rendering to the base environment."""
        return self._env.render(record=record)

    def stop(self) -> None:  # pragma: no cover
        """Delegate stop to the base environment."""
        self._env.stop()


class ContinuousTimeAugmentedEnv(Environment):
    """
    Wrap a continuous (:class:`Box` observation) MushroomRL environment
    and append a normalized time-to-go feature to the observation.

    The augmented observation is

    .. code-block:: text

        [o_1, ..., o_d, (horizon - 1 - t) / (horizon - 1)]

    so the time feature equals ``1.0`` at ``t = 0`` and ``0.0`` at
    ``t = horizon - 1``. For ``horizon == 1`` the normalizer degenerates
    and the feature is fixed to ``0.0``.

    Args:
        env: a MushroomRL :class:`Environment` with a :class:`Box`
            observation space;
        horizon: finite positive integer horizon ``T``.
    """

    def __init__(self, env: Environment, horizon: int) -> None:
        if not isinstance(env.info.observation_space, spaces.Box):
            raise TypeError(
                "ContinuousTimeAugmentedEnv requires a Box observation "
                f"space, got {type(env.info.observation_space).__name__}."
            )
        horizon = _validate_horizon(horizon)

        base_space = env.info.observation_space
        base_low = np.asarray(base_space.low, dtype=float).reshape(-1)
        base_high = np.asarray(base_space.high, dtype=float).reshape(-1)

        if base_low.shape != base_high.shape:
            raise ValueError(
                "Base observation space has inconsistent low/high shapes: "
                f"{base_low.shape} vs {base_high.shape}."
            )

        # time_to_go_normalized lives in [0, 1].
        augmented_low = np.concatenate([base_low, np.array([0.0])])
        augmented_high = np.concatenate([base_high, np.array([1.0])])
        augmented_space = spaces.Box(
            low=augmented_low,
            high=augmented_high,
            shape=(augmented_low.size,),
            data_type=base_space.data_type,
        )

        self._env = env
        self._horizon = horizon
        self._t: int = 0
        self._base_dim = int(base_low.size)

        mdp_info = MDPInfo(
            observation_space=augmented_space,
            action_space=env.info.action_space,
            gamma=env.info.gamma,
            horizon=horizon,
            dt=env.info.dt,
            backend=env.info.backend,
        )
        super().__init__(mdp_info)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> int:
        """Current stage index ``t`` (0-indexed)."""
        return self._t

    @property
    def base_env(self) -> Environment:
        """The wrapped base environment."""
        return self._env

    @property
    def base_dim(self) -> int:
        """Dimensionality of the base observation (without the time feature)."""
        return self._base_dim

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _time_to_go(self, t: int) -> float:
        """Return the normalized time-to-go feature for stage ``t``."""
        if self._horizon <= 1:
            return 0.0
        t_clamped = min(max(int(t), 0), self._horizon - 1)
        return float(self._horizon - 1 - t_clamped) / float(self._horizon - 1)

    def _augment(self, base_state: np.ndarray, t: int) -> np.ndarray:
        """Concatenate the time-to-go feature onto a base observation."""
        base = np.asarray(base_state).reshape(-1).astype(float, copy=False)
        return np.concatenate([base, np.array([self._time_to_go(t)])])

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    def seed(self, seed: int) -> None:  # pragma: no cover - delegates
        """Seed the base environment (MushroomRL default is a no-op)."""
        self._env.seed(seed)

    def reset(
        self, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the wrapped env and set the stage index to zero.

        Args:
            state: optional forced initial state. May be either a base
                observation (length ``base_dim``) or an already
                augmented observation (length ``base_dim + 1``) — in
                the latter case the trailing time feature is dropped
                before being passed to the base env (the wrapper always
                resets to stage ``0``).

        Returns:
            ``(augmented_state, info)`` where ``augmented_state`` is a
            1-D numpy array of length ``base_dim + 1``.
        """
        self._t = 0
        base_state_arg: np.ndarray | None = None

        if state is not None:
            arr = np.asarray(state).reshape(-1)
            if arr.size == self._base_dim:
                base_state_arg = arr.copy()
            elif arr.size == self._base_dim + 1:
                base_state_arg = arr[: self._base_dim].copy()
            else:
                raise ValueError(
                    "reset state has length "
                    f"{arr.size}; expected {self._base_dim} (base) or "
                    f"{self._base_dim + 1} (augmented)."
                )

        base_state, info = self._env.reset(base_state_arg)
        return self._augment(base_state, self._t), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance the base env by one step and increment the stage.

        Returns:
            ``(augmented_next_state, reward, absorbing, info)``.
            ``absorbing`` is the base-env flag OR-ed with the terminal
            stage condition ``t_next == horizon - 1``.
        """
        next_base_state, reward, base_absorbing, info = self._env.step(action)
        self._t += 1
        t_next = self._t
        augmented_next = self._augment(next_base_state, t_next)

        terminal_stage = t_next >= self._horizon - 1
        absorbing = bool(base_absorbing) or bool(terminal_stage)

        return augmented_next, float(reward), absorbing, info

    def render(self, record: bool = False) -> Any:  # pragma: no cover
        """Delegate rendering to the base environment."""
        return self._env.render(record=record)

    def stop(self) -> None:  # pragma: no cover
        """Delegate stop to the base environment."""
        self._env.stop()


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def make_time_augmented(env: Environment, horizon: int) -> Environment:
    """
    Dispatch to the correct time-augmented wrapper for ``env``.

    Args:
        env: a MushroomRL :class:`Environment`.
        horizon: finite positive integer horizon ``T``.

    Returns:
        A :class:`DiscreteTimeAugmentedEnv` when ``env`` has a
        :class:`Discrete` observation space, else a
        :class:`ContinuousTimeAugmentedEnv` when ``env`` has a
        :class:`Box` observation space.

    Raises:
        TypeError: if the base env's observation space is neither
            :class:`Discrete` nor :class:`Box`.
    """
    obs_space = env.info.observation_space
    if isinstance(obs_space, spaces.Discrete):
        return DiscreteTimeAugmentedEnv(env, horizon=horizon)
    if isinstance(obs_space, spaces.Box):
        return ContinuousTimeAugmentedEnv(env, horizon=horizon)
    raise TypeError(
        "make_time_augmented: unsupported observation space "
        f"{type(obs_space).__name__}; expected Discrete or Box."
    )
