"""Private helpers for Phase V WP2 task factories (Family A / B / C).

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5.

Pulled out of the public factory modules to keep each factory file well
under the 500-LOC budget while sharing:

* the zero-sum-under-gamma-weight shape-basis projection used by Family A,
* per-shape raw basis generators (flat / front-loaded-compensated / one-
  bump / two-bump / ramp) before projection,
* a thin ``FiniteMDP``-building wrapper that attaches the
  ``initial_state`` attribute consumed by WP1a ``compute_d_ref`` /
  ``evaluate_candidate``.

Everything here is private (underscore-prefixed) and imports only numpy
plus MushroomRL's ``FiniteMDP``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the repo-local mushroom-rl-dev tree importable without polluting
# the user's sys.path order.  (Mirrors the pattern in
# ``planners/geometry_priority_dp.py``.)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mushroom_rl.environments.finite_mdp import FiniteMDP  # noqa: E402

__all__ = [
    "FiniteMDP",
    "build_finite_mdp",
    "project_to_gamma_null_space",
    "shape_basis",
    "SHAPE_NAMES",
]


SHAPE_NAMES: tuple[str, ...] = (
    "flat",
    "front_loaded_compensated",
    "one_bump",
    "two_bump",
    "ramp",
)


# ---------------------------------------------------------------------------
# FiniteMDP constructor with the Phase V contract baked in
# ---------------------------------------------------------------------------

def build_finite_mdp(
    P: np.ndarray,       # (S, A, S')
    R: np.ndarray,       # (S, A, S')
    *,
    gamma: float,
    horizon: int,
    initial_state: int,
) -> FiniteMDP:
    """Build a MushroomRL ``FiniteMDP`` with the Phase V contract.

    Contract (see ``search/family_spec.py`` docstring on ``FamilySpec``):
        * stationary transitions ``p`` of shape ``(S, A, S')``,
        * stationary rewards ``r`` of shape ``(S, A, S')``,
        * finite horizon ``T = int(horizon)``,
        * point-mass initial distribution at ``initial_state`` — also
          attached as the ``initial_state`` attribute so WP1a's occupancy
          helper finds it without falling back to ``mdp.mu``.

    Row-stochasticity and shape checks are performed here once (rather
    than in every factory) so the downstream DP / tie solver sees a
    well-formed MDP.
    """
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if P.ndim != 3:
        raise ValueError(f"P must have ndim 3; got shape {P.shape}")
    if P.shape != R.shape:
        raise ValueError(
            f"P and R must share shape; got P={P.shape}, R={R.shape}"
        )
    S = P.shape[0]
    row_sums = P.sum(axis=2)                                # (S, A)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        bad = int(np.unravel_index(np.argmax(np.abs(row_sums - 1.0)), row_sums.shape)[0])
        raise ValueError(
            f"P rows must sum to 1 (atol=1e-8); worst row s={bad}, "
            f"max deviation={np.max(np.abs(row_sums - 1.0)):.3e}."
        )
    if not (0 <= int(initial_state) < S):
        raise ValueError(
            f"initial_state={initial_state} must lie in [0, {S})."
        )

    mu_0 = np.zeros(S, dtype=np.float64)
    mu_0[int(initial_state)] = 1.0
    mdp = FiniteMDP(
        p=P,
        rew=R,
        mu=mu_0,
        gamma=float(gamma),
        horizon=int(horizon),
    )
    # Attach Phase V point-mass attribute so compute_d_ref / evaluate_candidate
    # don't have to argmax over ``mu``.
    mdp.initial_state = int(initial_state)
    return mdp


# ---------------------------------------------------------------------------
# Shape-basis projection and per-shape raw generators (Family A)
# ---------------------------------------------------------------------------

def project_to_gamma_null_space(
    h_raw: np.ndarray,       # (L,)
    gamma: float,
) -> np.ndarray:
    """Project ``h_raw`` onto the orthogonal complement of ``(gamma^k)_k``.

    Returns ``h_proj`` of the same shape ``(L,)`` satisfying
    ``sum_k gamma^k * h_proj[k] == 0`` exactly (up to float round-off).

    Formula::

        num = sum_k gamma^k * h_raw[k]
        den = sum_k (gamma^k)^2
        h_proj[k] = h_raw[k] - (num / den) * gamma^k

    This is the unique minimum-L2 perturbation that preserves the
    classical discounted value ``sum_k gamma^k (c_tie + h_k) == sum_k gamma^k c_tie``.
    """
    h_raw = np.asarray(h_raw, dtype=np.float64)
    if h_raw.ndim != 1:
        raise ValueError(f"h_raw must be 1D; got shape {h_raw.shape}")
    L = h_raw.shape[0]
    k = np.arange(L, dtype=np.float64)                         # (L,)
    gk = np.power(float(gamma), k)                             # (L,)
    num = float(np.dot(gk, h_raw))
    den = float(np.dot(gk, gk))
    if den <= 0.0:
        return h_raw.copy()
    h_proj = h_raw - (num / den) * gk                          # (L,)
    return h_proj


def _raw_flat(L: int, psi: dict[str, Any]) -> np.ndarray:
    """Flat (identity) basis; returns zeros of shape ``(L,)``."""
    return np.zeros(int(L), dtype=np.float64)


def _raw_front_loaded(L: int, psi: dict[str, Any]) -> np.ndarray:
    """Positive rectangular bump on ``[0, bump_width)`` of strength ``bump_strength``.

    The tail (``k >= bump_width``) is left at zero in the raw basis; the
    subsequent gamma-null-space projection handles the compensating
    negative tail automatically, so the post-projection basis satisfies
    ``sum_k gamma^k h_k == 0`` exactly.
    """
    width = int(psi.get("bump_width", max(1, L // 4)))
    strength = float(psi.get("bump_strength", 1.0))
    width = min(max(1, width), int(L))
    h = np.zeros(int(L), dtype=np.float64)
    h[:width] = strength
    return h


def _raw_one_bump(L: int, psi: dict[str, Any]) -> np.ndarray:
    """Gaussian bump centered at ``bump_center`` with width ``bump_width``."""
    center = float(psi.get("bump_center", (int(L) - 1) / 2.0))
    width = float(psi.get("bump_width", max(1.0, int(L) / 6.0)))
    strength = float(psi.get("bump_strength", 1.0))
    k = np.arange(int(L), dtype=np.float64)
    h = strength * np.exp(-0.5 * ((k - center) / max(width, 1e-8)) ** 2)
    return h


def _raw_two_bump(L: int, psi: dict[str, Any]) -> np.ndarray:
    """Sum of two Gaussian bumps.

    Parameters (read from ``psi`` with defaults):
        * ``bump_centers`` : ``(c1, c2)``; default ``(L/4, 3L/4)``
        * ``bump_widths``  : ``(w1, w2)``; default ``(L/8, L/8)``
        * ``bump_strengths`` : ``(s1, s2)``; default ``(+1.0, -1.0)``
          (opposite signs emphasize two-bump-vs-one-bump contrast).
    """
    L_int = int(L)
    centers = psi.get("bump_centers", (L_int / 4.0, 3 * L_int / 4.0))
    widths = psi.get("bump_widths", (max(1.0, L_int / 8.0), max(1.0, L_int / 8.0)))
    strengths = psi.get("bump_strengths", (1.0, -1.0))
    c1, c2 = float(centers[0]), float(centers[1])
    w1, w2 = float(widths[0]), float(widths[1])
    s1, s2 = float(strengths[0]), float(strengths[1])
    k = np.arange(L_int, dtype=np.float64)
    h = (
        s1 * np.exp(-0.5 * ((k - c1) / max(w1, 1e-8)) ** 2)
        + s2 * np.exp(-0.5 * ((k - c2) / max(w2, 1e-8)) ** 2)
    )
    return h


def _raw_ramp(L: int, psi: dict[str, Any]) -> np.ndarray:
    """Linear ramp from ``-a`` to ``+a``; projection enforces exact zero-sum-under-gamma."""
    a = float(psi.get("ramp_amplitude", 1.0))
    L_int = int(L)
    if L_int <= 1:
        return np.zeros(L_int, dtype=np.float64)
    k = np.arange(L_int, dtype=np.float64)
    # Linearly interpolate from -a at k=0 to +a at k=L-1.
    h = -a + 2.0 * a * k / float(L_int - 1)
    return h


_SHAPE_REGISTRY = {
    "flat": _raw_flat,
    "front_loaded_compensated": _raw_front_loaded,
    "one_bump": _raw_one_bump,
    "two_bump": _raw_two_bump,
    "ramp": _raw_ramp,
}


def shape_basis(
    L: int,
    gamma: float,
    psi: dict[str, Any],
) -> np.ndarray:
    """Return ``h_k(psi)`` of shape ``(L,)`` with ``sum_k gamma^k h_k == 0``.

    Dispatches on ``psi["shape"]`` (one of :data:`SHAPE_NAMES`), then
    projects onto the gamma-null-space so that the classical discounted
    value ``sum_k gamma^k (c_tie + h_k)`` is invariant to the basis.
    """
    shape = str(psi.get("shape", "flat"))
    if shape not in _SHAPE_REGISTRY:
        raise ValueError(
            f"unknown shape {shape!r}; must be one of {SHAPE_NAMES!r}."
        )
    raw = _SHAPE_REGISTRY[shape](int(L), psi)              # (L,)
    return project_to_gamma_null_space(raw, float(gamma))  # (L,)
