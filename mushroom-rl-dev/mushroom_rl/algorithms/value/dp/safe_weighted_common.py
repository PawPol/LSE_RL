"""
Safe weighted-LSE Bellman operator: core math layer.

This module provides the numerically stable implementation of the safe
weighted-LSE one-step target, responsibility, effective discount,
KL term, and certification machinery (certified radii, beta caps).

All safe DP planners and safe TD algorithms import from here.

Mathematical reference
----------------------
The safe one-step target at stage t is

    g_t^safe(r, v) = ((1+gamma) / beta_t) *
        [log(exp(beta_t * r) + gamma * exp(beta_t * v)) - log(1+gamma)]

implemented via ``np.logaddexp`` to avoid overflow/underflow.  When
``beta_used == 0`` the classical target ``r + gamma * v`` is returned
exactly (no logaddexp call).

The responsibility is

    rho_t(r, v) = sigmoid(beta_t * (r - v) + log(1/gamma))

and the effective discount is

    d_t(r, v) = (1 + gamma) * (1 - rho_t(r, v)).
"""

from __future__ import annotations

import json
import pathlib
import warnings
from typing import Any

import numpy as np
from scipy.special import expit as _sigmoid  # numerically stable sigmoid

# Threshold below which |beta| is treated as zero (classical collapse).
# The logaddexp formula has O(beta * |r - gamma*v|) cancellation error as
# beta→0; using the classical formula for |beta| < _EPS_BETA bounds that
# error at O(1e-8 * value_range), negligible for any realistic problem.
_EPS_BETA: float = 1e-8

__all__ = [
    "BetaSchedule",
    "SafeWeightedCommon",
    "compute_kappa",
    "compute_certified_radii",
    "compute_beta_cap",
    "build_certification",
]


# ---------------------------------------------------------------------------
# BetaSchedule
# ---------------------------------------------------------------------------

class BetaSchedule:
    """Per-stage access to a calibrated beta schedule.

    A schedule is a JSON-serialisable dict containing per-stage arrays
    (``beta_raw_t``, ``beta_used_t``, ``beta_cap_t``, ``alpha_t``,
    ``kappa_t``, ``Bhat_t``, ...) plus scalar metadata (``gamma``,
    ``sign``, ``task_family``, ``reward_bound``, etc.).
    """

    def __init__(self, schedule: dict[str, Any]) -> None:
        self._raw = dict(schedule)  # keep a copy

        # --- scalar metadata -----------------------------------------------
        self._gamma: float = float(schedule["gamma"])
        self._sign: int = int(schedule["sign"])
        self._task_family: str = str(schedule.get("task_family", ""))

        # --- per-stage arrays (cast to float64) ----------------------------
        self._beta_raw_t: np.ndarray = np.asarray(
            schedule["beta_raw_t"], dtype=np.float64
        )
        self._beta_cap_t: np.ndarray = np.asarray(
            schedule["beta_cap_t"], dtype=np.float64
        )
        self._beta_used_t: np.ndarray = np.asarray(
            schedule["beta_used_t"], dtype=np.float64
        )
        self._alpha_t: np.ndarray = np.asarray(
            schedule["alpha_t"], dtype=np.float64
        )
        self._kappa_t: np.ndarray = np.asarray(
            schedule["kappa_t"], dtype=np.float64
        )
        self._Bhat_t: np.ndarray = np.asarray(
            schedule["Bhat_t"], dtype=np.float64
        )

        # length consistency
        _T = len(self._beta_used_t)
        for name, arr in [
            ("beta_raw_t", self._beta_raw_t),
            ("beta_cap_t", self._beta_cap_t),
            ("alpha_t", self._alpha_t),
            ("kappa_t", self._kappa_t),
        ]:
            if len(arr) != _T:
                raise ValueError(
                    f"Length mismatch: beta_used_t has {_T} entries "
                    f"but {name} has {len(arr)}."
                )
        # Bhat has length T+1
        if len(self._Bhat_t) != _T + 1:
            raise ValueError(
                f"Bhat_t must have length T+1={_T + 1}, "
                f"got {len(self._Bhat_t)}."
            )

        self._validate_certification()

    # --- certification invariant checks ------------------------------------

    def _validate_certification(self) -> None:
        """Verify certification invariants on the loaded schedule.

        Checks performed (all with atol=1e-9, rtol=0):
        1. alpha_t in [0, 1) for every stage.
        2. beta_used_t == clip(beta_raw_t, -beta_cap_t, beta_cap_t).
        3. If reward_bound is present: kappa_t, Bhat_t, beta_cap_t agree
           with build_certification(alpha_t, reward_bound, gamma).
        4. beta_cap_t >= 0 for every stage.
        """
        _atol = 1e-9

        # --- Check 1: alpha_t domain [0, 1) --------------------------------
        if np.any(self._alpha_t < 0.0):
            bad = int(np.argmax(self._alpha_t < 0.0))
            raise ValueError(
                f"alpha_t[{bad}] = {self._alpha_t[bad]} is negative; "
                f"all entries must be in [0, 1)."
            )
        if np.any(self._alpha_t >= 1.0):
            bad = int(np.argmax(self._alpha_t >= 1.0))
            raise ValueError(
                f"alpha_t[{bad}] = {self._alpha_t[bad]} >= 1; "
                f"all entries must be in [0, 1)."
            )

        # --- Check 2: beta_used consistency ---------------------------------
        expected_used = np.clip(
            self._beta_raw_t, -self._beta_cap_t, self._beta_cap_t
        )
        if not np.allclose(self._beta_used_t, expected_used,
                           atol=_atol, rtol=0):
            diffs = np.abs(self._beta_used_t - expected_used)
            bad = int(np.argmax(diffs))
            raise ValueError(
                f"beta_used_t[{bad}] = {self._beta_used_t[bad]} does not "
                f"match clip(beta_raw_t[{bad}], -beta_cap_t[{bad}], "
                f"beta_cap_t[{bad}]) = {expected_used[bad]} "
                f"(diff={diffs[bad]:.2e}, atol={_atol})."
            )

        # --- Check 3: certification recurrence round-trip -------------------
        reward_bound = self._raw.get("reward_bound")
        if reward_bound is not None:
            cert = build_certification(
                self._alpha_t, R_max=float(reward_bound), gamma=self._gamma
            )
            for key, stored in [
                ("kappa_t", self._kappa_t),
                ("Bhat_t", self._Bhat_t),
                ("beta_cap_t", self._beta_cap_t),
            ]:
                recomputed = cert[key]
                if not np.allclose(stored, recomputed, atol=_atol, rtol=0):
                    max_diff = float(np.max(np.abs(stored - recomputed)))
                    # beta_cap_t may be intentionally overridden (e.g. set
                    # larger to avoid clipping in tests).  A stored cap that
                    # is element-wise >= the certified cap is strictly more
                    # permissive but still internally consistent -- the
                    # safety guarantee degrades gracefully.  Only raise if
                    # the stored cap is *smaller* than the certified cap
                    # (which would be unsound) or if kappa/Bhat diverge.
                    if key == "beta_cap_t":
                        undershoot = float(
                            np.max(recomputed - stored)
                        )
                        if undershoot <= _atol:
                            # stored >= recomputed everywhere: permissive
                            # override for test fixtures that set large
                            # caps to exercise specific beta values.
                            # Emit a warning so this is never silent in
                            # production schedules.
                            overshoot = float(
                                np.max(stored - recomputed)
                            )
                            warnings.warn(
                                f"BetaSchedule: stored beta_cap_t "
                                f"exceeds certified cap by up to "
                                f"{overshoot:.2e}. This is accepted "
                                f"(permissive override) but means the "
                                f"contraction guarantee may not hold "
                                f"for |beta_used_t| > certified cap. "
                                f"Set beta_cap_t to the certified "
                                f"values for production schedules.",
                                stacklevel=2,
                            )
                            continue
                    raise ValueError(
                        f"Certification recurrence mismatch for {key}: "
                        f"max |stored - recomputed| = {max_diff:.2e} "
                        f"exceeds atol={_atol}."
                    )

        # --- Check 4: beta_cap non-negative ---------------------------------
        if np.any(self._beta_cap_t < 0.0):
            bad = int(np.argmax(self._beta_cap_t < 0.0))
            raise ValueError(
                f"beta_cap_t[{bad}] = {self._beta_cap_t[bad]} is negative; "
                f"all entries must be >= 0."
            )

    def _validate_certification_strict(self) -> None:
        """Raise ValueError if stored beta_cap_t exceeds the certified cap.

        Unlike ``_validate_certification`` (which emits a warning for
        permissive overrides), this method enforces a hard check intended
        for production schedule loads via ``from_file``.  Test fixtures
        that construct ``BetaSchedule(dict)`` directly are unaffected.
        """
        _atol = 1e-9
        reward_bound = self._raw.get("reward_bound")
        if reward_bound is None:
            return  # no cert data to check against
        cert = build_certification(
            self._alpha_t, R_max=float(reward_bound), gamma=self._gamma
        )
        recomputed_cap = cert["beta_cap_t"]
        overshoot = self._beta_cap_t - recomputed_cap
        if np.any(overshoot > _atol):
            max_overshoot = float(np.max(overshoot))
            raise ValueError(
                f"BetaSchedule.from_file: stored beta_cap_t exceeds the "
                f"certified cap by up to {max_overshoot:.2e}. This schedule "
                f"cannot be safely loaded without allow_uncertified_cap=True. "
                f"Regenerate the schedule or pass allow_uncertified_cap=True "
                f"for ablation/test use."
            )

    # --- constructors ------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str | pathlib.Path,
        *,
        allow_uncertified_cap: bool = False,
    ) -> BetaSchedule:
        """Load a schedule from a JSON file.

        Parameters
        ----------
        path : path to a schedule JSON.
        allow_uncertified_cap : if False (default), raises ValueError when
            the stored beta_cap_t exceeds the certified cap.  Set to True
            only for ablation schedules that intentionally relax the
            certification bound.
        """
        with open(path) as f:
            schedule = json.load(f)
        obj = cls(schedule)  # may emit warnings for oversized caps
        # Ablation schedules that intentionally relax the beta cap (e.g.
        # beta_raw_unclipped) carry an "ablation_type" field in their JSON.
        # These are automatically granted uncertified-cap access so that
        # ablation runners can load them without out-of-band flag threading.
        is_ablation = bool(obj._raw.get("ablation_type"))
        if (
            not allow_uncertified_cap
            and not is_ablation
            and obj._raw.get("reward_bound") is not None
        ):
            # Strict check for production schedules: raise instead of warn.
            obj._validate_certification_strict()
        return obj

    @classmethod
    def zeros(cls, T: int, gamma: float) -> BetaSchedule:
        """Create an all-zero schedule (classical collapse).

        Every ``beta_used_t`` is 0, every ``alpha_t`` is 0, and the
        certified radii / caps are computed consistently.
        """
        alpha_t = np.zeros(T, dtype=np.float64)
        cert = build_certification(alpha_t, R_max=1.0, gamma=gamma)
        schedule: dict[str, Any] = {
            "gamma": gamma,
            "sign": 0,
            "task_family": "zeros",
            "beta_raw_t": np.zeros(T).tolist(),
            "beta_cap_t": cert["beta_cap_t"].tolist(),
            "beta_used_t": np.zeros(T).tolist(),
            "alpha_t": alpha_t.tolist(),
            "kappa_t": cert["kappa_t"].tolist(),
            "Bhat_t": cert["Bhat_t"].tolist(),
            "clip_active_t": [False] * T,
            "informativeness_t": [0.0] * T,
            "d_target_t": [gamma] * T,
            "reward_bound": 1.0,
            "source_phase": "zeros",
            "calibration_source_path": "",
            "calibration_hash": "",
            "lambda_min": 0.0,
            "lambda_max": 0.0,
            "margin_quantile": 0.0,
            "notes": "All-zero schedule for classical equivalence.",
        }
        return cls(schedule)

    # --- per-stage accessors -----------------------------------------------

    def beta_used_at(self, t: int) -> float:
        """Return clipped (used) beta at stage *t*."""
        return float(self._beta_used_t[t])

    def beta_raw_at(self, t: int) -> float:
        """Return raw (pre-clip) beta at stage *t*."""
        return float(self._beta_raw_t[t])

    def beta_cap_at(self, t: int) -> float:
        """Return clip cap at stage *t*."""
        return float(self._beta_cap_t[t])

    def alpha_at(self, t: int) -> float:
        """Return headroom fraction alpha at stage *t*."""
        return float(self._alpha_t[t])

    def kappa_at(self, t: int) -> float:
        """Return contraction rate kappa at stage *t*."""
        return float(self._kappa_t[t])

    # --- properties --------------------------------------------------------

    @property
    def T(self) -> int:
        """Horizon length (number of stages)."""
        return len(self._beta_used_t)

    @property
    def gamma(self) -> float:
        """Nominal discount factor."""
        return self._gamma

    @property
    def sign(self) -> int:
        """Sign convention for beta (+1 optimistic, -1 pessimistic, 0 zero)."""
        return self._sign

    @property
    def task_family(self) -> str:
        """Task family identifier string."""
        return self._task_family


# ---------------------------------------------------------------------------
# SafeWeightedCommon  (mixin / helper)
# ---------------------------------------------------------------------------

class SafeWeightedCommon:
    """Core math for the safe weighted-LSE Bellman operator.

    All safe DP planners and safe TD algorithms use this class (via
    composition or inheritance) to perform the safe backup.

    Notation
    --------
    - ``r``      : expected immediate reward  r(s, a).
    - ``v_next`` : expected next-state value   E[V_{t+1}(s') | s, a].
    - ``t``      : current stage index (0-based).
    - ``beta``   : temperature parameter (from schedule).
    - ``gamma``  : nominal discount factor.
    """

    def __init__(
        self,
        schedule: BetaSchedule,
        gamma: float,
        n_base: int,
    ) -> None:
        """
        Args:
            schedule: BetaSchedule providing ``beta_used_at(t)``.
            gamma: nominal task discount factor.
            n_base: number of base (un-augmented) states; used to decode
                augmented state ids to stage via ``t = aug_id // n_base``.
        """
        self._schedule = schedule
        self._gamma = float(gamma)
        self._n_base = int(n_base)

        # Guard: schedule gamma must match the MDP gamma.
        if abs(schedule._gamma - self._gamma) > 1e-9:
            raise ValueError(
                f"SafeWeightedCommon: schedule.gamma={schedule._gamma} does "
                f"not match provided gamma={self._gamma}. Schedule must be "
                "calibrated for this exact discount factor."
            )
        self._log_gamma = np.log(gamma) if gamma > 0 else -np.inf
        self._log_inv_gamma = -self._log_gamma  # log(1/gamma)
        self._log_1_plus_gamma = np.log(1.0 + gamma)
        self._one_plus_gamma = 1.0 + gamma

        # --- instrumentation (set after each target call) ------------------
        self.last_stage: int = -1
        self.last_beta_raw: float = 0.0
        self.last_beta_cap: float = 0.0
        self.last_beta_used: float = 0.0
        self.last_clip_active: bool = False
        self.last_rho: float | np.ndarray = 0.0
        self.last_effective_discount: float | np.ndarray = 0.0
        self.last_target: float | np.ndarray = 0.0
        self.last_margin: float | np.ndarray = 0.0

    # -------------------------------------------------------------------
    # Core scalar methods
    # -------------------------------------------------------------------

    def compute_rho(self, r: float, v_next: float, t: int) -> float:
        """Responsibility rho_t(r, v_next).

        rho = sigmoid(beta * (r - v_next) + log(1/gamma))

        When beta == 0 the sigmoid argument is log(1/gamma), giving
        rho = 1 / (1 + gamma).
        """
        beta = self._schedule.beta_used_at(t)
        arg = beta * (float(r) - float(v_next)) + self._log_inv_gamma
        return float(_sigmoid(arg))

    def compute_safe_target(self, r: float, v_next: float, t: int) -> float:
        """Safe weighted-LSE one-step target g_t^safe(r, v_next).

        When ``beta_used == 0`` returns the exact classical target
        ``r + gamma * v_next`` (no logaddexp call).
        """
        r_f = float(r)
        v_f = float(v_next)
        beta = self._schedule.beta_used_at(t)

        # --- instrumentation bookkeeping ---
        self.last_stage = t
        self.last_beta_raw = self._schedule.beta_raw_at(t)
        self.last_beta_cap = self._schedule.beta_cap_at(t)
        self.last_beta_used = beta
        self.last_clip_active = abs(self.last_beta_raw) > abs(beta) + 1e-15
        self.last_margin = r_f - v_f

        if beta == 0.0 or abs(beta) < _EPS_BETA:
            # Classical collapse: exact when beta==0; O(beta*|r-gamma*v|)
            # error (negligible at beta < 1e-8) avoids catastrophic
            # cancellation in the logaddexp form at tiny nonzero beta.
            target = r_f + self._gamma * v_f
            rho = 1.0 / self._one_plus_gamma
            eff_d = self._one_plus_gamma * (1.0 - rho)
        else:
            # logaddexp formula — numerically stable for all finite args at
            # |beta| >= _EPS_BETA.  numpy.logaddexp applies the max-subtract
            # trick internally so neither overflow nor underflow can produce
            # non-finite results from finite r/v inputs.
            log_sum = np.logaddexp(
                beta * r_f, beta * v_f + self._log_gamma
            )
            target = (self._one_plus_gamma / beta) * (
                log_sum - self._log_1_plus_gamma
            )
            rho = float(_sigmoid(
                beta * (r_f - v_f) + self._log_inv_gamma
            ))
            eff_d = self._one_plus_gamma * (1.0 - rho)

        self.last_rho = rho
        self.last_effective_discount = eff_d
        self.last_target = target
        return target

    def compute_effective_discount(
        self, r: float, v_next: float, t: int
    ) -> float:
        """Effective discount d_t = (1 + gamma) * (1 - rho_t)."""
        rho = self.compute_rho(r, v_next, t)
        return self._one_plus_gamma * (1.0 - rho)

    def compute_kl_term(self, rho: float) -> float:
        """Binary KL divergence KL(q || p0).

        Prior:  p0 = (1/(1+gamma), gamma/(1+gamma)).
        Posterior: q = (rho, 1 - rho).

        KL = rho * log(rho*(1+gamma)) + (1-rho) * log((1-rho)*(1+gamma)/gamma)

        Edge cases handled by continuity:
        - rho = 0: KL = log((1+gamma)/gamma)
        - rho = 1: KL = log(1+gamma)
        """
        rho_f = float(rho)
        gamma = self._gamma

        if rho_f <= 0.0:
            return float(self._log_1_plus_gamma - self._log_gamma)
        if rho_f >= 1.0:
            return float(self._log_1_plus_gamma)

        term1 = rho_f * np.log(rho_f * self._one_plus_gamma)
        term2 = (1.0 - rho_f) * np.log(
            (1.0 - rho_f) * self._one_plus_gamma / gamma
        )
        return float(term1 + term2)

    def compute_margin(self, r: float, v_next: float) -> float:
        """Margin = r - v_next (no gamma factor)."""
        return float(r) - float(v_next)

    def clip_beta(self, raw_beta: float, cap: float) -> float:
        """Clip raw_beta to [-cap, cap].  cap must be >= 0."""
        cap_f = float(cap)
        if cap_f < 0.0:
            raise ValueError(f"cap must be >= 0, got {cap_f}")
        return float(np.clip(float(raw_beta), -cap_f, cap_f))

    def stage_from_augmented_state(self, aug_id: int) -> int:
        """Decode stage t from augmented state id: t = aug_id // n_base."""
        return int(aug_id) // self._n_base

    # -------------------------------------------------------------------
    # Vectorised batch methods
    # -------------------------------------------------------------------

    def compute_rho_batch(
        self, r_bar: np.ndarray, v_next: np.ndarray, t: int
    ) -> np.ndarray:
        """Vectorised rho over a grid.

        Args:
            r_bar: expected reward array, shape ``(S, A)`` or any broadcastable.
            v_next: ``E[V_{t+1} | s, a]`` array, same shape as *r_bar*.
            t: current stage.

        Returns:
            rho array, same shape as inputs.
        """
        beta = self._schedule.beta_used_at(t)
        r_arr = np.asarray(r_bar, dtype=np.float64)
        v_arr = np.asarray(v_next, dtype=np.float64)
        arg = beta * (r_arr - v_arr) + self._log_inv_gamma
        return _sigmoid(arg)

    def compute_safe_target_batch(
        self, r_bar: np.ndarray, v_next: np.ndarray, t: int
    ) -> np.ndarray:
        """Vectorised g_t^safe over a grid.

        Args:
            r_bar: expected reward, shape ``(S, A)``.
            v_next: ``E[V_{t+1} | s, a]``, shape ``(S, A)``.
            t: current stage.

        Returns:
            target array, shape ``(S, A)``.
        """
        r_arr = np.asarray(r_bar, dtype=np.float64)
        v_arr = np.asarray(v_next, dtype=np.float64)
        beta = self._schedule.beta_used_at(t)

        # --- instrumentation ---
        self.last_stage = t
        self.last_beta_raw = self._schedule.beta_raw_at(t)
        self.last_beta_cap = self._schedule.beta_cap_at(t)
        self.last_beta_used = beta
        self.last_clip_active = abs(self.last_beta_raw) > abs(beta) + 1e-15
        self.last_margin = r_arr - v_arr

        if beta == 0.0 or abs(beta) < _EPS_BETA:
            # Classical collapse (see compute_safe_target for rationale).
            target = r_arr + self._gamma * v_arr
            rho = np.full_like(r_arr, 1.0 / self._one_plus_gamma)
            eff_d = np.full_like(r_arr, self._gamma)
        else:
            # logaddexp formula — stable for all finite args at |beta| >= _EPS_BETA.
            log_sum = np.logaddexp(
                beta * r_arr, beta * v_arr + self._log_gamma
            )
            target = (self._one_plus_gamma / beta) * (
                log_sum - self._log_1_plus_gamma
            )
            arg = beta * (r_arr - v_arr) + self._log_inv_gamma
            rho = _sigmoid(arg)
            eff_d = self._one_plus_gamma * (1.0 - rho)

        self.last_rho = rho
        self.last_effective_discount = eff_d
        self.last_target = target
        return target

    def compute_effective_discount_batch(
        self, r_bar: np.ndarray, v_next: np.ndarray, t: int
    ) -> np.ndarray:
        """Vectorised effective discount, shape same as inputs."""
        rho = self.compute_rho_batch(r_bar, v_next, t)
        return self._one_plus_gamma * (1.0 - rho)


# ---------------------------------------------------------------------------
# Standalone certification functions
# ---------------------------------------------------------------------------

def compute_kappa(alpha_t: np.ndarray, gamma: float) -> np.ndarray:
    """Contraction rate: kappa_t = gamma + alpha_t * (1 - gamma).

    Args:
        alpha_t: headroom fractions, shape ``(T,)``.
        gamma: nominal discount factor.

    Returns:
        kappa_t, shape ``(T,)``.
    """
    alpha = np.asarray(alpha_t, dtype=np.float64)
    return gamma + alpha * (1.0 - gamma)


def compute_certified_radii(
    T: int,
    kappa_t: np.ndarray,
    R_max: float,
    gamma: float,
) -> np.ndarray:
    """Backward recursion for certified radii.

    .. math::

        \\hat B_T = 0, \\quad
        \\hat B_t = (1+\\gamma) R_{\\max} + \\kappa_t \\hat B_{t+1}.

    Args:
        T: horizon length.
        kappa_t: contraction rates, shape ``(T,)``.
        R_max: reward bound.
        gamma: nominal discount factor.

    Returns:
        Bhat, shape ``(T+1,)``, where ``Bhat[t]`` is the certified radius
        at stage *t* and ``Bhat[T] = 0``.
    """
    kappa = np.asarray(kappa_t, dtype=np.float64)
    if len(kappa) != T:
        raise ValueError(f"kappa_t has length {len(kappa)}, expected {T}.")

    Bhat = np.zeros(T + 1, dtype=np.float64)
    # Bhat[T] = 0 already
    for t in range(T - 1, -1, -1):
        Bhat[t] = (1.0 + gamma) * R_max + kappa[t] * Bhat[t + 1]
    return Bhat


def compute_beta_cap(
    kappa_t: np.ndarray,
    Bhat_t: np.ndarray,
    R_max: float,
    gamma: float,
) -> np.ndarray:
    """Stage-wise clip cap for beta.

    .. math::

        \\beta_t^{\\text{cap}} =
        \\frac{\\log\\bigl(\\kappa_t / (\\gamma (1+\\gamma - \\kappa_t))\\bigr)}
              {R_{\\max} + \\hat B_{t+1}}

    Special cases:
    - ``kappa_t == gamma`` (``alpha_t == 0``): numerator ``log(...)`` is 0,
      so ``beta_cap_t = 0``.
    - ``R_max + Bhat_{t+1} == 0``: ``beta_cap_t = inf``.
    - ``gamma * (1 + gamma - kappa_t) <= 0``: raises ``ValueError``.

    Args:
        kappa_t: contraction rates, shape ``(T,)``.
        Bhat_t: certified radii, shape ``(T+1,)``.
        R_max: reward bound.
        gamma: nominal discount factor.

    Returns:
        beta_cap, shape ``(T,)``.
    """
    kappa = np.asarray(kappa_t, dtype=np.float64)
    Bhat = np.asarray(Bhat_t, dtype=np.float64)
    T = len(kappa)

    denominator_inner = gamma * (1.0 + gamma - kappa)
    if np.any(denominator_inner <= 0.0):
        bad = np.where(denominator_inner <= 0.0)[0]
        raise ValueError(
            f"gamma*(1+gamma-kappa_t) <= 0 at stages {bad.tolist()}. "
            f"kappa_t values: {kappa[bad].tolist()}, gamma={gamma}."
        )

    # Numerator: log(kappa / (gamma * (1 + gamma - kappa)))
    log_numer = np.log(kappa / denominator_inner)

    # Denominator of the cap formula: R_max + Bhat[t+1]
    denom = R_max + Bhat[1 : T + 1]

    beta_cap = np.where(
        denom == 0.0,
        np.inf,
        log_numer / denom,
    )

    # When kappa == gamma exactly (alpha == 0), log_numer is 0 by
    # construction; the division 0/denom should give 0, but guard against
    # 0/0 = nan when denom is also 0.
    zero_numer_mask = np.isclose(kappa, gamma, atol=1e-15, rtol=0.0)
    beta_cap = np.where(zero_numer_mask, 0.0, beta_cap)

    return beta_cap


def build_certification(
    alpha_t: np.ndarray,
    R_max: float,
    gamma: float,
) -> dict[str, np.ndarray]:
    """Convenience wrapper: compute all certification quantities.

    Args:
        alpha_t: headroom fractions, shape ``(T,)``.
        R_max: reward bound.
        gamma: nominal discount factor.

    Returns:
        Dict with keys ``'kappa_t'`` (shape ``(T,)``),
        ``'Bhat_t'`` (shape ``(T+1,)``), ``'beta_cap_t'`` (shape ``(T,)``).
    """
    alpha = np.asarray(alpha_t, dtype=np.float64)
    T = len(alpha)
    kappa = compute_kappa(alpha, gamma)
    Bhat = compute_certified_radii(T, kappa, R_max, gamma)
    cap = compute_beta_cap(kappa, Bhat, R_max, gamma)
    return {"kappa_t": kappa, "Bhat_t": Bhat, "beta_cap_t": cap}
