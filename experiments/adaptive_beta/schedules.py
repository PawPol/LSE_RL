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

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, FrozenSet, List, Literal, Optional, Protocol, Tuple

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

# Phase VIII M4 W1.A: 4 new schedules (spec §6.4–§6.6).
METHOD_ORACLE_BETA = "oracle_beta"
METHOD_HAND_ADAPTIVE_BETA = "hand_adaptive_beta"
METHOD_CONTRACTION_UCB_BETA = "contraction_ucb_beta"
METHOD_RETURN_UCB_BETA = "return_ucb_beta"

ALL_METHOD_IDS = (
    METHOD_VANILLA,
    METHOD_FIXED_POSITIVE,
    METHOD_FIXED_NEGATIVE,
    METHOD_WRONG_SIGN,
    METHOD_ADAPTIVE_BETA,
    METHOD_ADAPTIVE_BETA_NO_CLIP,
    METHOD_ADAPTIVE_SIGN_ONLY,
    METHOD_ADAPTIVE_MAGNITUDE_ONLY,
    METHOD_ORACLE_BETA,
    METHOD_HAND_ADAPTIVE_BETA,
    METHOD_CONTRACTION_UCB_BETA,
    METHOD_RETURN_UCB_BETA,
)

# Default β arm grid for UCB schedules (spec §6.4). Seven arms; each value is
# inside [-β_cap, +β_cap] with β_cap = 2.0, so no extra clipping.
DEFAULT_BETA_ARM_GRID: Tuple[float, ...] = (
    -2.0, -1.0, -0.5, 0.0, +0.5, +1.0, +2.0,
)

# Round-robin warm-start length: episodes 0..(N_arms - 1) force one pull per
# arm, UCB selection takes over from episode N_arms onward (spec §6.5).
_WARM_START_LEN = len(DEFAULT_BETA_ARM_GRID)  # = 7

# Default hyperparameters per spec §4.2 + Phase VIII §6.6 (HandAdaptive).
_DEFAULT_HPARAMS: Dict[str, float] = {
    "beta_max": 2.0,
    "beta_cap": 2.0,
    "k": 5.0,
    "initial_beta": 0.0,
    "beta_tol": 1.0e-8,
    "lambda_smooth": 1.0,
    "beta0": 1.0,
    "A_scale": 0.1,  # Phase VIII M4: HandAdaptiveBetaSchedule scale factor.
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
# Phase VIII M4 (§6.5–§6.6): four new schedules. Note that ``OracleBetaSchedule``,
# ``ContractionUCBBetaSchedule`` and ``ReturnUCBBetaSchedule`` accept
# *non-float* keys (``regime_to_beta`` dict, ``arm_grid`` list) which the
# factory passes through directly as constructor kwargs rather than via
# ``_resolved_hparams`` (which would call ``float(v)``).
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
    "OracleBetaSchedule": frozenset({"regime_to_beta"}),
    "HandAdaptiveBetaSchedule": frozenset({"beta0", "A_scale", "lambda_smooth"}),
    "ContractionUCBBetaSchedule": frozenset(
        {"arm_grid", "ucb_c", "epsilon_floor", "residual_smoothing_window"}
    ),
    "ReturnUCBBetaSchedule": frozenset({"arm_grid", "ucb_c", "epsilon_floor"}),
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
    """Protocol every per-episode schedule satisfies.

    Phase VIII M4 (§6.5–§6.6) extended ``update_after_episode`` with three
    optional kwargs consumed by the new oracle / hand-adaptive / UCB
    schedules; existing concrete schedules ignore them. Default ``None``
    keeps every Phase VII callsite backward-compatible (the legacy agent
    keeps passing only ``divergence_event``).
    """

    name: str

    def beta_for_episode(self, episode_index: int) -> float: ...

    def update_after_episode(
        self,
        episode_index: int,
        rewards: np.ndarray,
        v_next: np.ndarray,
        divergence_event: bool = False,
        episode_info: Optional[Dict[str, Any]] = None,
        bellman_residual: Optional[float] = None,
        episode_return: Optional[float] = None,
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
        # Phase VIII M4: transient kwargs captured by ``update_after_episode``
        # for subclass consumption. Reset to None each episode by the caller
        # of ``update_after_episode`` (see signature). Initialised to None
        # so subclasses can rely on the attribute existing.
        self._last_episode_info: Optional[Dict[str, Any]] = None
        self._last_bellman_residual: Optional[float] = None
        self._last_episode_return: Optional[float] = None
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
        episode_info: Optional[Dict[str, Any]] = None,
        bellman_residual: Optional[float] = None,
        episode_return: Optional[float] = None,
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

        # Phase VIII M4 (§6.5–§6.6): expose the three optional kwargs as
        # transient instance attributes so concrete subclasses
        # (Oracle / UCB / HandAdaptive) can read them inside
        # ``_compute_next_beta``. Phase VII subclasses ignore them.
        self._last_episode_info = episode_info
        self._last_bellman_residual = bellman_residual
        self._last_episode_return = episode_return

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
# Phase VIII M4 (§6.5–§6.6): Oracle, HandAdaptive, ContractionUCB, ReturnUCB
# ---------------------------------------------------------------------------


class _WelfordTracker:
    """Single-stream running mean / variance via Welford's online algorithm.

    Used for per-arm reward standardisation in the UCB schedules
    (spec §6.5). Standardised reward is computed from each arm's *own*
    running mean/std so the UCB exploration bonus is unitless. With a
    single sample we return std = 0.0 and ``standardise`` falls back to
    the raw centred deviation (zero) — sensible default that does not
    blow up downstream.
    """

    __slots__ = ("count", "mean", "_M2")

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self._M2: float = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        # Population std (ddof=0): consistent across arms regardless of N.
        return math.sqrt(self._M2 / self.count)

    def standardise(self, x: float) -> float:
        sd = self.std
        if sd <= 0.0:
            return 0.0
        return (x - self.mean) / sd


class OracleBetaSchedule(_BaseSchedule):
    """Per-episode oracle β: looks up ``regime_to_beta[info['regime']]``.

    Spec §6.6. The oracle is the only schedule allowed to read
    ``info["regime"]`` from the env; sign-switching composites guarantee
    the regime is constant within an episode, so the runner passes a
    single ``episode_info`` dict (typically the last step's info) rather
    than a list-of-dicts.

    Missing ``episode_info`` or missing ``"regime"`` key raises ``KeyError``
    — the oracle is the only schedule that may raise on missing data
    (spec §6.6).
    """

    name = METHOD_ORACLE_BETA

    def __init__(
        self,
        regime_to_beta: Dict[str, float],
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(regime_to_beta, dict) or not regime_to_beta:
            raise ValueError(
                "OracleBetaSchedule requires a non-empty regime_to_beta dict"
            )
        # Eagerly cast to float so downstream lookups are numeric.
        self._regime_to_beta: Dict[str, float] = {
            str(k): float(v) for k, v in regime_to_beta.items()
        }
        super().__init__(hyperparams)

    def _initial_beta(self) -> float:
        # No regime observed yet for episode 0 -> return 0.0. The first
        # ``update_after_episode`` will set β_1 from the observed regime.
        return 0.0

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        info = self._last_episode_info
        if info is None:
            raise KeyError(
                "OracleBetaSchedule.update_after_episode requires "
                "episode_info=dict(regime=...); got None."
            )
        if "regime" not in info:
            raise KeyError(
                "OracleBetaSchedule.update_after_episode: episode_info "
                f"missing 'regime' key (got keys={sorted(info.keys())})."
            )
        regime = info["regime"]
        if regime not in self._regime_to_beta:
            raise KeyError(
                f"OracleBetaSchedule: regime={regime!r} not in lookup table "
                f"(known regimes: {sorted(self._regime_to_beta.keys())})."
            )
        beta = float(self._regime_to_beta[regime])
        # Oracle is exact: no clipping or smoothing on top of the lookup.
        return beta, beta


class HandAdaptiveBetaSchedule(_BaseSchedule):
    """β = sign(Ā) · β₀ · min(1, |Ā| / A_scale) (spec §6.6).

    Pre-registered episode rule, no per-task tuning. Defaults pinned as
    class-level constants for tests:
      β₀ = 1.0, A_scale = 0.1, λ_smooth = 1.0.

    The smoothed advantage ``Ā`` is the existing
    ``_BaseSchedule._state.smoothed_A`` (EMA with rate ``λ_smooth``).
    """

    name = METHOD_HAND_ADAPTIVE_BETA

    # Pinned defaults — tests assert these are the configured values.
    DEFAULT_BETA0: float = 1.0
    DEFAULT_A_SCALE: float = 0.1
    DEFAULT_LAMBDA_SMOOTH: float = 1.0

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        beta0 = float(self._h["beta0"])
        A_scale = float(self._h["A_scale"])
        if A_scale <= 0.0:
            raise ValueError(f"A_scale must be > 0, got {A_scale}")
        magnitude_ratio = min(1.0, abs(smoothed_A) / A_scale)
        sign = float(np.sign(smoothed_A))  # np.sign(0) == 0
        beta = sign * beta0 * magnitude_ratio
        # No clip layer (the rule already lives in [-β₀, +β₀]).
        return beta, beta


class _BaseUCBBetaSchedule(_BaseSchedule):
    """Shared scaffolding for the two 7-arm UCB schedules (spec §6.5).

    Subclasses override ``_episode_reward(...)`` to extract the per-episode
    raw reward signal (contraction-log-ratio vs. episode return). All
    other moving parts — round-robin warm-start, per-arm Welford
    standardisation, UCB1 selection with lowest-index tie-break — live
    here.
    """

    # To be overridden by concrete subclasses.
    name: str = "_base_ucb"
    _DEFAULT_UCB_C: float = 1.0
    _USE_LOG_REWARD: bool = False  # ContractionUCB sets True

    def __init__(
        self,
        arm_grid: Optional[List[float]] = None,
        ucb_c: Optional[float] = None,
        epsilon_floor: float = 1e-8,
        residual_smoothing_window: int = 1,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        if arm_grid is None:
            arm_grid_t: Tuple[float, ...] = DEFAULT_BETA_ARM_GRID
        else:
            if len(arm_grid) == 0:
                raise ValueError("arm_grid must be non-empty")
            arm_grid_t = tuple(float(b) for b in arm_grid)
        self._arm_grid: Tuple[float, ...] = arm_grid_t
        self._n_arms: int = len(arm_grid_t)
        self._ucb_c: float = float(self._DEFAULT_UCB_C if ucb_c is None else ucb_c)
        if epsilon_floor <= 0.0:
            raise ValueError(f"epsilon_floor must be > 0, got {epsilon_floor}")
        self._eps: float = float(epsilon_floor)
        if int(residual_smoothing_window) < 1:
            raise ValueError(
                "residual_smoothing_window must be >= 1, got "
                f"{residual_smoothing_window}"
            )
        self._smoothing_window: int = int(residual_smoothing_window)

        # Per-arm running stats and pull counts.
        self._welford: List[_WelfordTracker] = [
            _WelfordTracker() for _ in range(self._n_arms)
        ]
        self._pull_counts: List[int] = [0] * self._n_arms
        # Per-arm running mean of the *standardised* reward stream — the
        # ``μ̂_j_std`` term in spec §6.5's UCB rule.
        self._standardised_running_mean: List[float] = [0.0] * self._n_arms
        self._n_std_samples: List[int] = [0] * self._n_arms
        # Index of the arm deployed in the *current* episode (i.e. the arm
        # whose β is cached in ``_state.last_beta_used``). Initially arm 0
        # for the round-robin warm-start.
        self._current_arm_idx: int = 0
        # Smoothed-residual deque (ContractionUCB uses it; ReturnUCB ignores).
        self._residual_window: Deque[float] = deque(maxlen=self._smoothing_window)
        # Previous-episode smoothed residual (for log-ratio reward).
        self._prev_smoothed_residual: Optional[float] = None
        super().__init__(hyperparams)

    # ------------------------------------------------------------------
    # Public introspection helpers (used by tests + M5 metric pipeline).
    # ------------------------------------------------------------------
    @property
    def arm_grid(self) -> Tuple[float, ...]:
        return self._arm_grid

    def pull_counts(self) -> Tuple[int, ...]:
        return tuple(self._pull_counts)

    def arm_means(self) -> Tuple[float, ...]:
        return tuple(w.mean for w in self._welford)

    def arm_stds(self) -> Tuple[float, ...]:
        return tuple(w.std for w in self._welford)

    def diagnostics(self) -> Dict[str, Any]:  # type: ignore[override]
        d = super().diagnostics()
        d.update(
            {
                "current_arm_idx": int(self._current_arm_idx),
                "pull_counts": list(self._pull_counts),
                "arm_grid": list(self._arm_grid),
                "ucb_c": float(self._ucb_c),
            }
        )
        return d

    # ------------------------------------------------------------------
    # Schedule mechanics
    # ------------------------------------------------------------------
    def _initial_beta(self) -> float:
        # Episode 0: round-robin arm 0. ``_current_arm_idx`` defaults to 0
        # in __init__ (above ``super().__init__()`` chain).
        # Note: at the time _BaseSchedule.__init__ calls _initial_beta the
        # subclass attributes are already set because we set them BEFORE
        # super().__init__ above.
        return float(self._arm_grid[0])

    def _episode_reward(self) -> Optional[float]:
        """Override in concrete subclasses. Return None to skip the update.

        For ContractionUCB this is the log-ratio M_e; for ReturnUCB it is
        the standardised episode return. The base class handles
        standardisation and Welford bookkeeping for both.
        """
        raise NotImplementedError

    def _select_next_arm(self, next_episode: int) -> int:
        """Pick the arm to deploy in ``next_episode``.

        - Episodes 0..(N_arms - 1): forced round-robin (arm = next_episode).
        - From N_arms onward: argmax over ``μ̂_j_std + c · √(2 ln N / N_j)``.
          Ties break on lowest arm index (numpy argmax convention).
        """
        if next_episode < self._n_arms:
            return int(next_episode)
        # Defensive: every arm has been pulled at least once after warm-start,
        # so log(N_total) > 0 and N_j > 0 are both safe.
        n_total = sum(self._pull_counts)
        if n_total <= 0:
            return 0
        log_N = math.log(float(n_total))
        scores = np.empty(self._n_arms, dtype=np.float64)
        for j in range(self._n_arms):
            n_j = self._pull_counts[j]
            if n_j <= 0:
                # Should not happen post-warm-start, but be defensive: an
                # unpulled arm is selected first to satisfy UCB1 invariants.
                return j
            # μ̂_j_std (spec §6.5): running mean of the *standardised*
            # reward stream for arm j. Tracked separately from the raw
            # Welford because we standardise each new sample against the
            # arm's own running stats (after the sample has been folded
            # in via ``_record_arm_reward``).
            mean_std = self._standardised_running_mean[j]
            bonus = self._ucb_c * math.sqrt(2.0 * log_N / float(n_j))
            scores[j] = mean_std + bonus
        # numpy argmax breaks ties on lowest index (spec §6.5).
        return int(np.argmax(scores))

    def _record_arm_reward(self, arm_idx: int, raw_reward: float) -> None:
        """Update Welford for ``arm_idx`` and push standardised reward into
        the per-arm running mean used for UCB scoring.

        Standardisation uses the arm's *own* Welford stats (after the
        new sample is incorporated). This is the canonical pattern: if
        std == 0 (single sample), the standardised value is 0.
        """
        w = self._welford[arm_idx]
        w.update(raw_reward)
        self._pull_counts[arm_idx] += 1
        std_reward = w.standardise(raw_reward)
        # Maintain a running mean of *standardised* rewards per arm. This
        # is what UCB1 scores against (μ̂_j_std in spec §6.5). We compute
        # it via a Welford-style update specialised to mean-only:
        n = self._n_std_samples[arm_idx] + 1
        prev_mean = self._standardised_running_mean[arm_idx]
        self._standardised_running_mean[arm_idx] = prev_mean + (std_reward - prev_mean) / n
        self._n_std_samples[arm_idx] = n

    def _compute_next_beta(self, smoothed_A: float) -> tuple[float, float]:
        # Step 1: attribute the just-finished episode's reward to the arm
        # we deployed (``_current_arm_idx``). Subclasses produce the raw
        # reward; ``None`` signals "skip" (e.g., first contraction step
        # where we have no previous residual).
        raw_reward = self._episode_reward()
        if raw_reward is not None:
            self._record_arm_reward(self._current_arm_idx, float(raw_reward))

        # Step 2: select the arm for the upcoming episode.
        next_episode = self._state.next_expected_episode  # = e + 1
        next_arm = self._select_next_arm(next_episode)
        self._current_arm_idx = next_arm
        beta = float(self._arm_grid[next_arm])
        return beta, beta


class ContractionUCBBetaSchedule(_BaseUCBBetaSchedule):
    """7-arm UCB on the per-episode contraction log-ratio (spec §6.5).

    Reward (per episode e, attributed to the arm deployed in episode e):

        M_e = log(R_{e-1} + ε) − log(R_e + ε)

    where ``R_e`` is the bellman_residual produced by the agent. Uses
    ``np.log`` (NOT ``log1p``; lessons.md #27). Per-arm Welford running
    mean/std standardise the reward stream before UCB scoring; UCB
    constant ``c = 1.0`` against the standardised reward.

    Round-robin warm-start over the 7 arms (episodes 0..6), UCB selection
    from episode 7 onward. Ties on lowest arm index.

    Residual smoothing: rolling mean of the last
    ``residual_smoothing_window`` raw residuals (default 1, i.e. no
    smoothing).
    """

    name = METHOD_CONTRACTION_UCB_BETA
    _DEFAULT_UCB_C = 1.0
    _USE_LOG_REWARD = True

    def __init__(
        self,
        arm_grid: Optional[List[float]] = None,
        ucb_c: float = 1.0,
        epsilon_floor: float = 1e-8,
        residual_smoothing_window: int = 1,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            arm_grid=arm_grid,
            ucb_c=ucb_c,
            epsilon_floor=epsilon_floor,
            residual_smoothing_window=residual_smoothing_window,
            hyperparams=hyperparams,
        )

    def _episode_reward(self) -> Optional[float]:
        # Bellman residual from the agent (post-update_after_episode kwargs).
        R = self._last_bellman_residual
        if R is None:
            # No residual fed; cannot compute a reward this episode.
            # Keep the arm's pull count untouched. Warm-start mechanics
            # are unaffected (they pull deterministically by episode index).
            return None
        # Apply the rolling-window smoothing before computing M_e.
        self._residual_window.append(float(R))
        smoothed_R = float(sum(self._residual_window) / len(self._residual_window))
        prev = self._prev_smoothed_residual
        self._prev_smoothed_residual = smoothed_R
        if prev is None:
            # First residual ever — no log-ratio yet.
            return None
        # M_e = log(R_{e-1} + ε) − log(R_e + ε). np.log, not log1p (lessons #27).
        return float(np.log(prev + self._eps) - np.log(smoothed_R + self._eps))


class ReturnUCBBetaSchedule(_BaseUCBBetaSchedule):
    """7-arm UCB on the standardised episode return (spec §6.5).

    Reward (per episode e): the agent's total episode return passed via
    ``update_after_episode(... episode_return=...)``. Per-arm Welford
    standardisation; UCB constant ``c = √2`` (canonical UCB1) against
    the **standardised** reward — NOT the raw return. Matrix-game
    returns are not in [0, 1] across games, so raw UCB1 with c = √2
    would mis-weight exploration; standardisation makes c = √2 the
    canonical choice.

    Round-robin warm-start over the 7 arms (episodes 0..6); UCB selection
    from episode 7. Ties on lowest arm index.
    """

    name = METHOD_RETURN_UCB_BETA
    _DEFAULT_UCB_C = math.sqrt(2.0)
    _USE_LOG_REWARD = False

    def __init__(
        self,
        arm_grid: Optional[List[float]] = None,
        ucb_c: float = math.sqrt(2.0),
        epsilon_floor: float = 1e-8,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ReturnUCB ignores ``residual_smoothing_window``; it is hard-coded
        # to 1 to keep _BaseUCBBetaSchedule contract uniform.
        super().__init__(
            arm_grid=arm_grid,
            ucb_c=ucb_c,
            epsilon_floor=epsilon_floor,
            residual_smoothing_window=1,
            hyperparams=hyperparams,
        )

    def _episode_reward(self) -> Optional[float]:
        ret = self._last_episode_return
        if ret is None:
            return None
        return float(ret)


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

    # ------------------------------------------------------------------
    # Phase VIII M4 (§6.5–§6.6) schedules
    # ------------------------------------------------------------------
    if method_id == METHOD_ORACLE_BETA:
        _check_allowed_keys("OracleBetaSchedule", hyperparams)
        if not hyperparams or "regime_to_beta" not in hyperparams:
            raise ValueError(
                "OracleBetaSchedule requires hyperparams['regime_to_beta'] "
                "(non-empty dict mapping regime label -> β)."
            )
        regime_to_beta = hyperparams["regime_to_beta"]
        # ``regime_to_beta`` is a dict, not float-coercible; do not pass it
        # through ``_resolved_hparams``. No other keys are allowed for the
        # Oracle, so we pass ``None`` for the float-only hyperparam channel.
        return OracleBetaSchedule(regime_to_beta=regime_to_beta, hyperparams=None)

    if method_id == METHOD_HAND_ADAPTIVE_BETA:
        _check_allowed_keys("HandAdaptiveBetaSchedule", hyperparams)
        return HandAdaptiveBetaSchedule(hyperparams=hyperparams)

    if method_id == METHOD_CONTRACTION_UCB_BETA:
        _check_allowed_keys("ContractionUCBBetaSchedule", hyperparams)
        # Extract complex-typed UCB constructor kwargs; bypass _resolved_hparams
        # for non-float keys (``arm_grid`` is a list, the others are scalars
        # but the constructor takes them positionally).
        h = dict(hyperparams) if hyperparams else {}
        arm_grid = h.pop("arm_grid", None)
        ucb_c = float(h.pop("ucb_c", 1.0))
        epsilon_floor = float(h.pop("epsilon_floor", 1e-8))
        residual_smoothing_window = int(h.pop("residual_smoothing_window", 1))
        return ContractionUCBBetaSchedule(
            arm_grid=list(arm_grid) if arm_grid is not None else None,
            ucb_c=ucb_c,
            epsilon_floor=epsilon_floor,
            residual_smoothing_window=residual_smoothing_window,
            hyperparams=None,
        )

    if method_id == METHOD_RETURN_UCB_BETA:
        _check_allowed_keys("ReturnUCBBetaSchedule", hyperparams)
        h = dict(hyperparams) if hyperparams else {}
        arm_grid = h.pop("arm_grid", None)
        ucb_c = float(h.pop("ucb_c", math.sqrt(2.0)))
        epsilon_floor = float(h.pop("epsilon_floor", 1e-8))
        return ReturnUCBBetaSchedule(
            arm_grid=list(arm_grid) if arm_grid is not None else None,
            ucb_c=ucb_c,
            epsilon_floor=epsilon_floor,
            hyperparams=None,
        )

    raise ValueError(
        f"unknown schedule method_id={method_id!r}; valid IDs: {ALL_METHOD_IDS}"
    )
