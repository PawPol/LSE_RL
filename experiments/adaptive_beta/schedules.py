"""Per-episode adaptive-beta schedule objects (Phase VII M1.3).

All schedules conform to the :class:`BetaSchedule` protocol and share the
following invariants (spec ``docs/specs/phase_VII_adaptive_beta.md`` §4, §5,
§13.2):

1. ``beta_for_episode(e)`` is constant within an episode and depends only on
   data observed strictly before episode ``e``. The value is cached after the
   first call so the agent's per-step lookups are O(1).
2. ``update_after_episode`` is the *only* mutator. It is called exactly once
   at the end of each episode with the full ``(rewards, v_next)`` traces and
   an optional ``divergence_event`` flag. Episode indices must increase by
   one each call (asserted).
3. No operator imports. Schedules are pure scheduling logic; the operator
   math lives in ``src/lse_rl/operator/tab_operator.py``.

The factory :func:`build_schedule` is the single entry point used by the
agent and runner; they never import the concrete classes directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Literal, Optional, Protocol

import numpy as np

CanonicalSign = Literal["+", "-"]

# Method ID strings (single source of truth, mirrored by the factory).
METHOD_VANILLA = "vanilla"
METHOD_FIXED_POSITIVE = "fixed_positive"
METHOD_FIXED_NEGATIVE = "fixed_negative"
METHOD_WRONG_SIGN = "wrong_sign"
METHOD_ADAPTIVE_BETA = "adaptive_beta"
METHOD_ADAPTIVE_BETA_NO_CLIP = "adaptive_beta_no_clip"
METHOD_ADAPTIVE_SIGN_ONLY = "adaptive_sign_only"
METHOD_ADAPTIVE_MAGNITUDE_ONLY = "adaptive_magnitude_only"

ALL_METHOD_IDS = (
    METHOD_VANILLA,
    METHOD_FIXED_POSITIVE,
    METHOD_FIXED_NEGATIVE,
    METHOD_WRONG_SIGN,
    METHOD_ADAPTIVE_BETA,
    METHOD_ADAPTIVE_BETA_NO_CLIP,
    METHOD_ADAPTIVE_SIGN_ONLY,
    METHOD_ADAPTIVE_MAGNITUDE_ONLY,
)

# Default hyperparameters per spec §4.2.
_DEFAULT_HPARAMS: Dict[str, float] = {
    "beta_max": 2.0,
    "beta_cap": 2.0,
    "k": 5.0,
    "initial_beta": 0.0,
    "beta_tol": 1.0e-8,
    "lambda_smooth": 1.0,
    "beta0": 1.0,
}


def _resolved_hparams(overrides: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Return defaults overlaid with ``overrides`` (None or {} ok).

    Unknown keys are silently ignored here; the strict per-class allow-list
    is enforced in :func:`build_schedule` so concrete classes constructed
    directly remain easy to use in tests, while config-driven construction
    catches typos loudly.
    """
    h = dict(_DEFAULT_HPARAMS)
    if overrides:
        for k, v in overrides.items():
            if k not in h:
                # Tolerate extra config keys; schedule consumers ignore them.
                continue
            h[k] = float(v)
    return h


# Per-class allow-lists for strict typo detection in :func:`build_schedule`.
# Spec §4.2 / §5: each concrete schedule consumes only the listed keys.
_ALLOWED_KEYS: Dict[str, FrozenSet[str]] = {
    "ZeroBetaSchedule": frozenset(),
    "FixedBetaSchedule": frozenset({"beta0"}),
    "WrongSignSchedule": frozenset({"beta0"}),
    "AdaptiveBetaSchedule": frozenset(
        {"beta_max", "beta_cap", "k", "initial_beta", "beta_tol", "lambda_smooth"}
    ),
    "AdaptiveSignOnlySchedule": frozenset({"beta0", "beta_cap", "lambda_smooth"}),
    "AdaptiveMagnitudeOnlySchedule": frozenset(
        {"beta_max", "beta_cap", "k", "initial_beta", "beta_tol", "lambda_smooth"}
    ),
}


def _check_allowed_keys(class_name: str, hyperparams: Optional[Dict[str, Any]]) -> None:
    """Raise ValueError on any key not whitelisted for ``class_name``.

    Empty / None overrides are always accepted. Used by :func:`build_schedule`
    so config typos like ``beta_caps`` (s) or ``lamda_smooth`` fail loudly
    instead of silently defaulting.
    """
    if not hyperparams:
        return
    allowed = _ALLOWED_KEYS[class_name]
    unknown = set(hyperparams) - allowed
    if unknown:
        raise ValueError(
            f"Unknown hyperparameter keys for {class_name}: {sorted(unknown)}; "
            f"expected one of {sorted(allowed)}"
        )


def _canonical_sign_to_value(s: Optional[str]) -> int:
    """Translate ``'+' / '-'`` to ``+1 / -1``; raise on anything else."""
    if s == "+":
        return +1
    if s == "-":
        return -1
    raise ValueError(
        "wrong_sign / adaptive_magnitude_only not defined for env without "
        "canonical sign"
    )


class BetaSchedule(Protocol):
    """Protocol every per-episode schedule satisfies."""

    name: str

    def beta_for_episode(self, episode_index: int) -> float: ...

    def update_after_episode(
        self,
        episode_index: int,
        rewards: np.ndarray,
        v_next: np.ndarray,
        divergence_event: bool = False,
    ) -> None: ...

    def diagnostics(self) -> Dict[str, Any]: ...


@dataclass
class _ScheduleState:
    """Mutable bookkeeping shared by every concrete schedule."""

    last_update_episode: int = -1   # episode index of the most recent update
    last_A_e: float = 0.0           # raw advantage from the most recent episode
    smoothed_A: float = 0.0         # exponential-moving-average of A_e
    last_beta_raw: float = 0.0      # pre-clip beta for the *current* episode
    last_beta_used: float = 0.0     # post-clip beta for the *current* episode
    divergence_event: bool = False  # sticky flag (spec §13.5)
    next_expected_episode: int = 0  # asserted on update_after_episode
    current_episode: int = 0        # the episode `last_beta_used` is for


def _compute_A_e(rewards: np.ndarray, v_next: np.ndarray) -> float:
    """Episode advantage A_e = (1 / H_e) * sum(r_t - v_next_t).

    Shapes: rewards (H_e,), v_next (H_e,). Empty traces yield ``A_e = 0``.
    """
    rewards_arr: np.ndarray = np.asarray(rewards, dtype=np.float64)
    v_next_arr: np.ndarray = np.asarray(v_next, dtype=np.float64)
    if rewards_arr.shape != v_next_arr.shape:
        raise ValueError(
            f"rewards shape {rewards_arr.shape} != v_next shape {v_next_arr.shape}"
        )
    if rewards_arr.size == 0:
        return 0.0
    advantage: np.ndarray = rewards_arr - v_next_arr  # (H_e,)
    return float(advantage.mean())


class _BaseSchedule:
    """Shared implementation surface; concrete classes override ``_compute_next_beta``."""

    name: str = "base"

    def __init__(self, hyperparams: Optional[Dict[str, Any]] = None) -> None:
        self._h = _resolved_hparams(hyperparams)
        self._state = _ScheduleState()
        # Beta to use for episode 0 (before any data exists) is the literal
        # ``initial_beta`` for adaptive variants; subclasses can override
        # ``_initial_beta`` for fixed variants.
        self._state.last_beta_raw = self._initial_beta()
        self._state.last_beta_used = self._initial_beta()

    # ------------------------------------------------------------------
    # Public protocol surface
    # ------------------------------------------------------------------
    def beta_for_episode(self, episode_index: int) -> float:
        """Return the cached beta_used for the *current* episode.

        Strict-current-only contract (design (a), spec §2 rule 5):
        - The agent calls ``beta_for_episode(e)`` exactly when entering
          episode ``e``. The cached beta is always for the episode whose
          number equals ``self._state.current_episode``.
        - Any call with a stale (``< current_episode``) or future
          (``> current_episode``) index raises ``ValueError``. This makes
          delayed-logging off-by-one bugs and accidental future-leakage in
          the agent loop loud rather than silent.
        - After ``update_after_episode(e, ...)`` the schedule advances
          ``current_episode`` to ``e + 1``; the next valid call is
          ``beta_for_episode(e + 1)``.
        """
        if episode_index < 0:
            raise ValueError(f"episode_index must be >= 0, got {episode_index}")
        current = self._state.current_episode
        if episode_index != current:
            kind = "stale" if episode_index < current else "future"
            raise ValueError(
                f"beta_for_episode received {kind} episode_index="
                f"{episode_index}; current_episode={current}. The schedule "
                f"only caches beta for the current episode (spec §2 rule 5)."
            )
        return float(self._state.last_beta_used)

    def update_after_episode(
        self,
        episode_index: int,
        rewards: np.ndarray,
        v_next: np.ndarray,
        divergence_event: bool = False,
    ) -> None:
        # Spec §4: episode (e+1)'s beta uses only data from episodes <= e.
        # We assert strict episode monotonicity to make accidental reuse or
        # future-leakage impossible.
        expected = self._state.next_expected_episode
        if episode_index != expected:
            raise AssertionError(
                f"update_after_episode: expected episode_index={expected}, "
                f"got {episode_index}"
            )

        A_e = _compute_A_e(rewards, v_next)
        lam = float(self._h["lambda_smooth"])
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"lambda_smooth must be in (0, 1], got {lam}")
        if self._state.last_update_episode < 0:
            # First update: smoothed_A_{-1} = 0 per spec §4.3 default.
            self._state.smoothed_A = lam * A_e
        else:
            self._state.smoothed_A = (
                (1.0 - lam) * self._state.smoothed_A + lam * A_e
            )
        self._state.last_A_e = A_e
        self._state.last_update_episode = episode_index
        self._state.next_expected_episode = episode_index + 1

        # Sticky divergence flag (spec §13.5): once True, stays True.
        if divergence_event:
            self._state.divergence_event = True

        beta_raw, beta_used = self._compute_next_beta(self._state.smoothed_A)
        self._state.last_beta_raw = float(beta_raw)
        self._state.last_beta_used = float(beta_used)
        # The cached beta is now intended for the next episode. Advance the
        # current_episode pointer so beta_for_episode(e+1) is the only valid
        # call until the next update.
        self._state.current_episode = episode_index + 1

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "beta_raw": float(self._state.last_beta_raw),
            "beta_used": float(self._state.last_beta_used),
            "A_e": float(self._state.last_A_e),
            "smoothed_A": float(self._state.smoothed_A),
            "last_update_episode": int(self._state.last_update_episode),
            "divergence_event": bool(self._state.divergence_event),
        }

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _initial_beta(self) -> float:
        """Beta returned by ``beta_for_episode(0)`` before any update."""
        return float(self._h["initial_beta"])

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        """Return ``(beta_raw, beta_used)`` for the upcoming episode."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete schedules
# ---------------------------------------------------------------------------
class ZeroBetaSchedule(_BaseSchedule):
    """Vanilla classical Bellman: beta == 0 always."""

    name = METHOD_VANILLA

    def _initial_beta(self) -> float:
        return 0.0

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        return 0.0, 0.0


class FixedBetaSchedule(_BaseSchedule):
    """``+beta0`` (sign=+1) or ``-beta0`` (sign=-1) for every episode."""

    def __init__(
        self,
        sign: int,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        if sign not in (-1, +1):
            raise ValueError(f"FixedBetaSchedule sign must be +/-1, got {sign}")
        self._sign = int(sign)
        self.name = (
            METHOD_FIXED_POSITIVE if sign == +1 else METHOD_FIXED_NEGATIVE
        )
        super().__init__(hyperparams)

    def _initial_beta(self) -> float:
        return self._sign * float(self._h["beta0"])

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        beta = self._sign * float(self._h["beta0"])
        return beta, beta


class WrongSignSchedule(_BaseSchedule):
    """Beta = -(env_canonical_sign) * beta0 (spec §22.3)."""

    name = METHOD_WRONG_SIGN

    def __init__(
        self,
        env_canonical_sign: Optional[str],
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        canonical = _canonical_sign_to_value(env_canonical_sign)  # raises if invalid
        # "wrong" sign is the canonical sign FLIPPED.
        self._effective_sign = -canonical
        super().__init__(hyperparams)

    def _initial_beta(self) -> float:
        return self._effective_sign * float(self._h["beta0"])

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        beta = self._effective_sign * float(self._h["beta0"])
        return beta, beta


class AdaptiveBetaSchedule(_BaseSchedule):
    """``beta = clip(beta_max * tanh(k * A_bar_e), -beta_cap, +beta_cap)``.

    With ``no_clip=True``, the raw rule is emitted unclipped (spec §5).
    """

    def __init__(
        self,
        no_clip: bool = False,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._no_clip = bool(no_clip)
        self.name = (
            METHOD_ADAPTIVE_BETA_NO_CLIP if no_clip else METHOD_ADAPTIVE_BETA
        )
        super().__init__(hyperparams)

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        beta_max = float(self._h["beta_max"])
        beta_cap = float(self._h["beta_cap"])
        k = float(self._h["k"])
        beta_raw = beta_max * float(np.tanh(k * smoothed_A))
        if self._no_clip:
            return beta_raw, beta_raw
        beta_used = float(np.clip(beta_raw, -beta_cap, +beta_cap))
        return beta_raw, beta_used


class AdaptiveSignOnlySchedule(_BaseSchedule):
    """``beta = clip(beta0 * sign(A_bar_e), -beta_cap, +beta_cap)``.

    ``np.sign(0) == 0`` so a zero advantage yields ``beta == 0`` exactly.
    """

    name = METHOD_ADAPTIVE_SIGN_ONLY

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        beta0 = float(self._h["beta0"])
        beta_cap = float(self._h["beta_cap"])
        beta_raw = beta0 * float(np.sign(smoothed_A))
        beta_used = float(np.clip(beta_raw, -beta_cap, +beta_cap))
        return beta_raw, beta_used


class AdaptiveMagnitudeOnlySchedule(_BaseSchedule):
    """``beta = sign_env * beta_max * |tanh(k * A_bar_e)|`` (clipped)."""

    name = METHOD_ADAPTIVE_MAGNITUDE_ONLY

    def __init__(
        self,
        env_canonical_sign: Optional[str],
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._env_sign = _canonical_sign_to_value(env_canonical_sign)
        super().__init__(hyperparams)

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        beta_max = float(self._h["beta_max"])
        beta_cap = float(self._h["beta_cap"])
        k = float(self._h["k"])
        magnitude = beta_max * float(np.abs(np.tanh(k * smoothed_A)))
        beta_raw = self._env_sign * magnitude
        beta_used = float(np.clip(beta_raw, -beta_cap, +beta_cap))
        return beta_raw, beta_used


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_schedule(
    method_id: str,
    env_canonical_sign: Optional[str],
    hyperparams: Optional[Dict[str, Any]],
) -> BetaSchedule:
    """Return the schedule object matching ``method_id``.

    The factory is the only API the agent and runner know about; concrete
    schedule classes are not imported elsewhere.
    """
    if method_id == METHOD_VANILLA:
        _check_allowed_keys("ZeroBetaSchedule", hyperparams)
        return ZeroBetaSchedule(hyperparams)
    if method_id == METHOD_FIXED_POSITIVE:
        _check_allowed_keys("FixedBetaSchedule", hyperparams)
        return FixedBetaSchedule(+1, hyperparams)
    if method_id == METHOD_FIXED_NEGATIVE:
        _check_allowed_keys("FixedBetaSchedule", hyperparams)
        return FixedBetaSchedule(-1, hyperparams)
    if method_id == METHOD_WRONG_SIGN:
        _check_allowed_keys("WrongSignSchedule", hyperparams)
        return WrongSignSchedule(env_canonical_sign, hyperparams)
    if method_id == METHOD_ADAPTIVE_BETA:
        _check_allowed_keys("AdaptiveBetaSchedule", hyperparams)
        return AdaptiveBetaSchedule(no_clip=False, hyperparams=hyperparams)
    if method_id == METHOD_ADAPTIVE_BETA_NO_CLIP:
        _check_allowed_keys("AdaptiveBetaSchedule", hyperparams)
        return AdaptiveBetaSchedule(no_clip=True, hyperparams=hyperparams)
    if method_id == METHOD_ADAPTIVE_SIGN_ONLY:
        _check_allowed_keys("AdaptiveSignOnlySchedule", hyperparams)
        return AdaptiveSignOnlySchedule(hyperparams)
    if method_id == METHOD_ADAPTIVE_MAGNITUDE_ONLY:
        _check_allowed_keys("AdaptiveMagnitudeOnlySchedule", hyperparams)
        return AdaptiveMagnitudeOnlySchedule(env_canonical_sign, hyperparams)
    raise ValueError(
        f"unknown schedule method_id={method_id!r}; valid IDs: {ALL_METHOD_IDS}"
    )
