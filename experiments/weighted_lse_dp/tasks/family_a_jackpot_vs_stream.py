"""Phase V WP2 — Family A: delayed jackpot vs smooth stream.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5
("Family A — delayed jackpot vs smooth stream (aligned propagation)").

Two-branch contest at the designated contest state ``x_c = (t_c, s_c) =
(0, 0)``:

* **Action A — delayed jackpot.** Zero reward for ``L - 1`` steps, then a
  terminal reward ``R`` at depth ``L``.
* **Action B — smooth stream.** Per-step reward ``c_k(psi)`` for ``L``
  steps, where ``c_k(psi) = c_tie + h_k(psi)`` and ``h_k(psi)`` satisfies
  the zero-sum-under-gamma-weight constraint ``sum_k gamma^k h_k == 0``.
  This preserves the classical discounted value of the stream branch
  (keeping the classical tie in place) while changing temporal
  concentration so the safe weighted-LSE operator sees the two branches
  differently.

Closed-form classical tie:
    c_tie(L, R, gamma) = gamma^{L-1} * R * (1 - gamma) / (1 - gamma^L).

Shape bases implemented (``psi["shape"]``):
    * ``"flat"``                      — identity; recovers the constant-c stream.
    * ``"front_loaded_compensated"``  — positive rectangular bump on early k.
    * ``"one_bump"``                  — Gaussian centered at bump_center.
    * ``"two_bump"``                  — sum of two Gaussians.
    * ``"ramp"``                      — linear ramp from -a to +a.

All five are post-processed by
``_family_helpers.project_to_gamma_null_space`` so the zero-sum-under-
gamma-weight identity holds exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.weighted_lse_dp.search.family_spec import (
    ContestState,
    FamilySpec,
)

from ._family_helpers import (
    FiniteMDP,
    build_finite_mdp,
    project_to_gamma_null_space,
    shape_basis,
    SHAPE_NAMES,
)

__all__ = [
    "SHAPE_NAMES",
    "c_tie_closed_form",
    "build_family_a_mdp",
    "family_a",
    "contest_state",
]


# ---------------------------------------------------------------------------
# Closed-form tie
# ---------------------------------------------------------------------------

def c_tie_closed_form(L: int, R: float, gamma: float) -> float:
    """Classical-tie per-step stream reward.

    Derivation
    ----------
    Let ``V_A = gamma^{L-1} R`` be the classical value of the jackpot
    branch at the contest state, and ``V_B = c * (1 - gamma^L) / (1 -
    gamma)`` the classical value of the constant-``c`` stream branch
    (geometric sum).  Setting ``V_A = V_B`` and solving for ``c``:

        c_tie = gamma^{L-1} R * (1 - gamma) / (1 - gamma^L).

    For ``gamma == 1`` the stream sum degenerates; we clamp ``L`` to
    ``L - 1`` effectively by returning ``R / L`` in that edge case to
    avoid a ``0 / 0``.  Callers should not set ``gamma == 1`` in the
    Phase V search — but the clamp keeps the helper robust.
    """
    L_int = int(L)
    gamma_f = float(gamma)
    R_f = float(R)
    if L_int < 2:
        raise ValueError(f"L must be >= 2; got L={L}")
    if not (0.0 < gamma_f <= 1.0):
        raise ValueError(f"gamma must lie in (0, 1]; got gamma={gamma}")
    if gamma_f == 1.0:
        return R_f / float(L_int)
    num = (gamma_f ** (L_int - 1)) * R_f * (1.0 - gamma_f)
    den = 1.0 - (gamma_f ** L_int)
    return float(num / den)


# ---------------------------------------------------------------------------
# MDP construction
# ---------------------------------------------------------------------------

# State layout (two-branch deterministic chain + absorbing terminals):
#
#   s = 0                       contest state (2 actions)
#   s = 1, ..., L               jackpot-branch states (s=L is absorbing)
#   s = L+1, ..., 2*L           stream-branch states (s=2L is absorbing)
#
# Horizon T = L.  Transitions on the two branches are deterministic when
# ``psi["stochastic"]`` is ``False`` (default); with
# ``psi["stochastic"] = True`` each forward edge succeeds with probability
# ``psi["p_transit"]`` and falls back to a self-loop otherwise.

def _check_contest_reachable(P: np.ndarray, initial_state: int, t_c: int, s_c: int) -> None:
    """Assert that ``(t_c, s_c)`` is reachable under the point-mass mu_0."""
    # With a point-mass at ``initial_state``, the stage-0 occupancy is
    # a Kronecker delta at initial_state.  The contest state in Family A
    # is always (t=0, s=0), so reachability reduces to initial_state == s_c
    # when t_c == 0.  Guard both branches for safety.
    if t_c == 0 and int(initial_state) != int(s_c):
        raise ValueError(
            f"contest_state (t={t_c}, s={s_c}) unreachable: "
            f"initial_state={initial_state}."
        )


def build_family_a_mdp(lam: float, psi: dict[str, Any]) -> FiniteMDP:
    """Build the Family A two-branch contest MDP.

    Parameters
    ----------
    lam : float
        The tie parameter — the per-step stream reward baseline ``c``.
        The stream branch pays ``c_k = lam + h_k(psi)`` at step ``k``,
        where ``h_k`` is the zero-sum-under-gamma-weight shape basis.
        Pass ``c_tie_closed_form(L, R, gamma)`` to hit the classical tie.
    psi : dict
        Geometry parameters.  Required keys: ``L`` (int >= 2),
        ``R`` (float > 0), ``gamma`` (float in (0, 1]).  Optional keys:
        ``shape`` (one of :data:`SHAPE_NAMES`, default ``"flat"``),
        ``stochastic`` (bool, default False), ``p_transit``
        (float in (0, 1], default 1.0), plus any shape-specific params
        consumed by :func:`shape_basis`.

    Returns
    -------
    FiniteMDP
        A MushroomRL ``FiniteMDP`` with ``2*L + 1`` states, 2 actions,
        finite horizon ``L``, and point-mass initial state at ``s = 0``.
    """
    L = int(psi["L"])
    if L < 2:
        raise ValueError(f"L must be >= 2; got L={L}")
    R = float(psi["R"])
    gamma = float(psi["gamma"])
    c = float(lam)

    stochastic = bool(psi.get("stochastic", False))
    p_transit = float(psi.get("p_transit", 1.0))
    if stochastic and not (0.0 < p_transit <= 1.0):
        raise ValueError(
            f"p_transit must lie in (0, 1]; got p_transit={p_transit}"
        )

    # Shape basis h_k with sum_k gamma^k h_k == 0 (projection enforced).
    h = shape_basis(L, gamma, psi)                         # (L,)
    # Per-step stream reward: c_k = c + h_k for k = 0, 1, ..., L - 1.
    c_per_step = c + h                                     # (L,)

    S = 2 * L + 1
    A = 2
    P = np.zeros((S, A, S), dtype=np.float64)              # (S, A, S')
    R_mat = np.zeros((S, A, S), dtype=np.float64)          # (S, A, S')

    p_fwd = p_transit if stochastic else 1.0
    p_stay = 1.0 - p_fwd

    # ---- Contest state s=0 ------------------------------------------------
    # Action 0 -> jackpot branch (enters s=1 with prob p_fwd; self-loop w/ p_stay).
    P[0, 0, 1] = p_fwd
    P[0, 0, 0] = p_stay
    R_mat[0, 0, 1] = 0.0                                   # jackpot pays only at tail
    R_mat[0, 0, 0] = 0.0
    # Action 1 -> stream branch (reward c_per_step[0] on entry).
    P[0, 1, L + 1] = p_fwd
    P[0, 1, 0] = p_stay
    R_mat[0, 1, L + 1] = float(c_per_step[0])
    R_mat[0, 1, 0] = float(c_per_step[0]) if stochastic else 0.0
    # ^ self-loop on stream branch still pays the stream reward on a "stay" —
    #   the reward is the reward of the attempted action; deterministic case
    #   uses p_stay = 0 so the self-loop entry is unreachable anyway.

    # ---- Jackpot branch --------------------------------------------------
    # Deterministic (or p_fwd stochastic) forward traversal s=i -> s=i+1 for
    # i = 1, ..., L - 1.  The terminal jackpot reward is paid on the edge
    # s=L-1 -> s=L (i.e. at depth k = L - 1 from the start state, which is
    # L hops away due to the initial s=0 -> s=1 edge).  Since the contest
    # state is at t=0 and the branch enters s=1 at t=1, the jackpot is paid
    # at t = L, which is the terminal boundary — matching the spec.
    for i in range(1, L):
        P[i, 0, i + 1] = p_fwd
        P[i, 1, i + 1] = p_fwd
        P[i, 0, i] = p_stay
        P[i, 1, i] = p_stay
        if i == L - 1:
            R_mat[i, 0, i + 1] = R
            R_mat[i, 1, i + 1] = R
    # s=L absorbing self-loop.
    P[L, 0, L] = 1.0
    P[L, 1, L] = 1.0

    # ---- Stream branch ---------------------------------------------------
    # s = L+1 corresponds to step k=1 in the stream (k=0 was the s=0 -> s=L+1
    # entry edge), so on the edge s=L+k -> s=L+k+1 the reward is
    # c_per_step[k].
    for k in range(1, L):
        s_from = L + k
        s_to = L + k + 1
        P[s_from, 0, s_to] = p_fwd
        P[s_from, 1, s_to] = p_fwd
        P[s_from, 0, s_from] = p_stay
        P[s_from, 1, s_from] = p_stay
        R_mat[s_from, 0, s_to] = float(c_per_step[k])
        R_mat[s_from, 1, s_to] = float(c_per_step[k])
        if stochastic:
            # self-loop pays the stream reward as well (see rationale above).
            R_mat[s_from, 0, s_from] = float(c_per_step[k])
            R_mat[s_from, 1, s_from] = float(c_per_step[k])
    # s = 2L absorbing self-loop.
    P[2 * L, 0, 2 * L] = 1.0
    P[2 * L, 1, 2 * L] = 1.0

    _check_contest_reachable(P, initial_state=0, t_c=0, s_c=0)

    mdp = build_finite_mdp(
        P=P,
        R=R_mat,
        gamma=gamma,
        horizon=L,
        initial_state=0,
    )
    return mdp


# ---------------------------------------------------------------------------
# FamilySpec
# ---------------------------------------------------------------------------

contest_state: ContestState = ContestState(t=0, s=0, action_pair=(0, 1))


def _warm_start(psi: dict[str, Any]) -> float:
    return c_tie_closed_form(int(psi["L"]), float(psi["R"]), float(psi["gamma"]))


def _scan_bracket(psi: dict[str, Any]) -> tuple[float, float]:
    hint = _warm_start(psi)
    # (0, 2 * c_tie) per the WP2 spec bracket instruction; keeps the tie
    # solver searching in the physically sensible positive-stream-reward
    # half-line.
    return (0.0, 2.0 * hint if hint > 0.0 else 1.0)


family_a: FamilySpec = FamilySpec(
    name="jackpot_vs_stream",
    build_mdp=build_family_a_mdp,
    contest_state=contest_state,
    warm_start_lambda=_warm_start,
    scan_bracket=_scan_bracket,
    metadata={
        "family": "A",
        "description": "delayed jackpot vs smooth stream (aligned propagation)",
        "tie_parameter": "c (per-step stream reward)",
        "geometry_parameter_keys": (
            "L", "R", "gamma", "shape",
            "bump_width", "bump_strength", "bump_center",
            "bump_centers", "bump_widths", "bump_strengths",
            "ramp_amplitude", "stochastic", "p_transit",
        ),
    },
)


# Re-export for tests that want the projection helper directly.
__all__.append("project_to_gamma_null_space")
