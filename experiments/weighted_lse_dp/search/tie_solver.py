"""Classical-tie bisection solver (Phase V WP1a).

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5
step 2 ("Solve for ``lambda_tie(psi)`` such that
``Delta_0(x_c; lambda_tie(psi), psi) approx 0`` -- bisection over exact
DP, warm-started with closed-form ties when available.").

For a given geometry ``psi`` and a family-defined contest state
``(t_c, s_c)`` with contested action pair ``(a1, a2)``, this module finds
the tie-parameter value ``lam`` such that the classical action gap

    Delta_0(lam) = Q*_cl(t_c, s_c, a1; lam, psi)
                 - Q*_cl(t_c, s_c, a2; lam, psi)

equals zero (to numerical tolerance).  The classical Q-function is
obtained from an exact finite-horizon backward sweep on the ``FiniteMDP``
returned by ``family.build_mdp(lam, psi)``.

Implementation notes
--------------------
* Root finder: :func:`scipy.optimize.brentq` with auto-bracket expansion
  around the warm start when no explicit bracket is supplied.  Bracket
  is expanded geometrically (x2 per step) up to 8x the initial
  half-width, then reported as :class:`TieNotBracketed` if the root is
  still not bracketed.
* Contest action pair: taken from ``family.contest_state.action_pair``
  when set.  When ``None`` the top-2 classical-Q actions at ``(t_c, s_c)``
  of the MDP built at the warm start are used; this is computed once and
  reused across bisection iterations to avoid non-deterministic drift of
  the contest pair as ``lam`` varies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq

from .family_spec import ContestState, FamilySpec

__all__ = ["TieNotBracketed", "lambda_tie"]


class TieNotBracketed(RuntimeError):
    """Raised when ``Delta_0`` does not change sign in the scan window.

    The exception payload carries the scan record (bracket endpoints,
    sampled ``Delta_0`` values, warm start, final expansion factor) so
    that WP1c's search driver can emit a reproducible failure manifest.
    """

    def __init__(self, message: str, *, scan_record: dict[str, Any]):
        super().__init__(message)
        self.scan_record = scan_record


# ---------------------------------------------------------------------------
# Exact classical finite-horizon DP (beta = 0)
# ---------------------------------------------------------------------------

def _extract_mdp_tensors(mdp: Any) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Pull ``(P, R, gamma, T)`` tensors out of a MushroomRL ``FiniteMDP``.

    Returns
    -------
    P : (S, A, S') transition probabilities
    R : (S, A, S') reward tensor
    gamma : discount factor
    T : finite horizon (int)
    """
    P = np.asarray(mdp.p, dtype=np.float64)   # [S, A, S']
    R = np.asarray(mdp.r, dtype=np.float64)   # [S, A, S']
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)
    return P, R, gamma, T


def _classical_q(
    mdp: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact finite-horizon classical Q*_cl and V*_cl via backward DP.

    Parameters
    ----------
    mdp : MushroomRL ``FiniteMDP`` with stationary ``p``/``r`` and finite
        horizon ``T``.  Rewards are treated as ``r(s, a, s')``.

    Returns
    -------
    Q : ndarray, shape ``(T, S, A)``
        Classical action-value under optimal policy (stage-0 = first step).
    V : ndarray, shape ``(T + 1, S)``
        ``V[T] = 0`` terminal boundary; ``V[t] = max_a Q[t, s, a]``.
    """
    P, R, gamma, T = _extract_mdp_tensors(mdp)
    S, A = P.shape[0], P.shape[1]
    # r_bar[s, a] = sum_s' P[s, a, s'] * R[s, a, s']
    r_bar = np.einsum("ijk,ijk->ij", P, R)  # (S, A)

    V = np.zeros((T + 1, S), dtype=np.float64)  # V[T] = 0
    Q = np.zeros((T, S, A), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        # Q[t, s, a] = r_bar[s, a] + gamma * sum_s' P[s, a, s'] * V[t+1, s']
        E_v = np.einsum("ijk,k->ij", P, V[t + 1])  # (S, A)
        Q[t] = r_bar + gamma * E_v
        V[t] = np.max(Q[t], axis=1)
    return Q, V


def _delta_zero(
    lam: float,
    psi: dict,
    family: FamilySpec,
    action_pair: tuple[int, int],
) -> float:
    """Classical action gap ``Q*_cl(a1) - Q*_cl(a2)`` at the contest state."""
    mdp = family.build_mdp(float(lam), psi)
    Q, _ = _classical_q(mdp)
    t_c = int(family.contest_state.t)
    s_c = int(family.contest_state.s)
    a1, a2 = action_pair
    return float(Q[t_c, s_c, int(a1)] - Q[t_c, s_c, int(a2)])


def _resolve_action_pair(
    family: FamilySpec,
    psi: dict,
    warm_start_lam: float,
) -> tuple[int, int]:
    """Return the contest action pair, defaulting to top-2 classical Q."""
    if family.contest_state.action_pair is not None:
        return tuple(family.contest_state.action_pair)  # type: ignore[return-value]
    mdp = family.build_mdp(float(warm_start_lam), psi)
    Q, _ = _classical_q(mdp)
    t_c = int(family.contest_state.t)
    s_c = int(family.contest_state.s)
    # Top-2 by Q-value (descending).
    q_row = Q[t_c, s_c]  # (A,)
    order = np.argsort(q_row)[::-1]
    return int(order[0]), int(order[1])


def _expand_bracket(
    f: "callable",
    lo: float,
    hi: float,
    *,
    max_expansion_factor: float = 8.0,
) -> tuple[float, float, float, float, float, list[dict[str, float]]]:
    """Geometrically expand ``[lo, hi]`` around the midpoint until f(lo)*f(hi) < 0.

    Returns
    -------
    lo_final, hi_final, f_lo, f_hi, expansion_factor, trace
    """
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if half <= 0.0:
        raise ValueError(
            f"bracket must have positive width; got (lo={lo}, hi={hi})."
        )
    factor = 1.0
    trace: list[dict[str, float]] = []
    f_lo = f(lo)
    f_hi = f(hi)
    trace.append({"factor": factor, "lo": lo, "hi": hi, "f_lo": f_lo, "f_hi": f_hi})
    while f_lo * f_hi > 0.0 and factor < max_expansion_factor:
        factor *= 2.0
        lo = center - factor * half
        hi = center + factor * half
        f_lo = f(lo)
        f_hi = f(hi)
        trace.append({"factor": factor, "lo": lo, "hi": hi, "f_lo": f_lo, "f_hi": f_hi})
    return lo, hi, f_lo, f_hi, factor, trace


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lambda_tie(
    psi: dict,
    family: FamilySpec,
    *,
    tol: float = 1e-10,
    bracket: tuple[float, float] | None = None,
    warm_start: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Solve ``Delta_0(lambda; psi) = 0`` via exact DP + brentq bisection.

    Parameters
    ----------
    psi : dict
        Geometry-parameter dict consumed by ``family.build_mdp``.
    family : FamilySpec
        Phase V family protocol (see :class:`FamilySpec`).
    tol : float
        Absolute tolerance on the root (passed as brentq ``xtol``).
    bracket : tuple[float, float] | None
        Optional explicit ``(lo, hi)`` bracket.  When ``None`` the
        family's ``scan_bracket(psi)`` hook supplies the default, which
        is then expanded around the warm start if no sign change is
        detected.
    warm_start : float | None
        Optional warm-start ``lambda``.  Defaults to
        ``family.warm_start_lambda(psi)``.

    Returns
    -------
    lam_tie : float
        Root of ``Delta_0(lambda; psi) = 0`` at tolerance ``tol``.
    diagnostics : dict
        ``{iters, final_gap_abs, bracket_final, warm_start_used,
        expansion_factor, action_pair, trace}``.

    Raises
    ------
    TieNotBracketed
        If ``Delta_0`` has the same sign at both endpoints of the
        expanded bracket (up to ``8x`` the initial half-width).
    """
    warm = (
        float(warm_start)
        if warm_start is not None
        else float(family.warm_start_lambda(psi))
    )
    action_pair = _resolve_action_pair(family, psi, warm)

    # Objective: Delta_0 at contest state as function of lam.
    def f(lam: float) -> float:
        return _delta_zero(lam, psi, family, action_pair)

    # Initial bracket.
    bracket_user_supplied = bracket is not None
    if bracket is None:
        lo, hi = family.scan_bracket(psi)
    else:
        lo, hi = bracket
    lo = float(lo)
    hi = float(hi)
    if not bracket_user_supplied and not (lo <= warm <= hi):
        # Default bracket should always contain the warm start; if the
        # family's scan_bracket hook returns something inconsistent,
        # recenter on the warm start.  An explicit caller-supplied
        # bracket is honored verbatim.
        half = 0.5 * (hi - lo) if hi > lo else max(abs(warm), 1.0)
        lo = warm - half
        hi = warm + half

    lo, hi, f_lo, f_hi, factor, trace = _expand_bracket(f, lo, hi)

    if f_lo * f_hi > 0.0:
        record = {
            "bracket_initial": (
                float(bracket[0]) if bracket is not None else float(family.scan_bracket(psi)[0]),
                float(bracket[1]) if bracket is not None else float(family.scan_bracket(psi)[1]),
            ),
            "bracket_final": (float(lo), float(hi)),
            "f_lo": float(f_lo),
            "f_hi": float(f_hi),
            "warm_start": float(warm),
            "expansion_factor": float(factor),
            "trace": trace,
            "action_pair": action_pair,
        }
        raise TieNotBracketed(
            f"Delta_0 did not change sign in bracket "
            f"[{lo:.6g}, {hi:.6g}] (expansion x{factor:g}); "
            f"f(lo)={f_lo:.6g}, f(hi)={f_hi:.6g}.",
            scan_record=record,
        )

    # Count brentq iterations via the `full_output=True` return.
    lam_star, result = brentq(
        f,
        lo,
        hi,
        xtol=tol,
        maxiter=40,
        full_output=True,
    )

    final_gap = float(f(float(lam_star)))
    diagnostics = {
        "iters": int(result.iterations),
        "function_calls": int(result.function_calls),
        "final_gap_abs": float(abs(final_gap)),
        "bracket_final": (float(lo), float(hi)),
        "warm_start_used": float(warm),
        "expansion_factor": float(factor),
        "action_pair": action_pair,
        "trace": trace,
    }
    return float(lam_star), diagnostics
