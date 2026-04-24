"""Family protocol for Phase V mechanism search (WP1a).

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5
("Construction recipe for candidate families").

A Phase V task family is described by a :class:`FamilySpec` dataclass with
five hooks that together let the search driver:

1. Solve for the classical tie ``lambda_tie(psi)`` by exact DP bisection
   over :class:`FiniteMDP` instances (``build_mdp`` and ``scan_bracket``).
2. Warm-start the bisection with any closed-form tie hint the family
   offers (``warm_start_lambda``).
3. Designate the reachable contest state `(t, s)` and the contested action
   pair `(a1, a2)` at which the classical decision boundary is probed
   (``contest_state``).

WP2 task factories (Family A / B / C) implement this protocol; WP1c's
search driver consumes it.  The protocol is deliberately independent of
the legacy ``experiments.weighted_lse_dp.tasks`` factories so that WP1a
does not need to touch any Phase I--IV code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from mushroom_rl.environments.finite_mdp import FiniteMDP


__all__ = ["ContestState", "FamilySpec"]


@dataclass(frozen=True)
class ContestState:
    """Designated near-tie contest point for a candidate family.

    Parameters
    ----------
    t : int
        Stage index in ``[0, T)`` on the time-augmented chain.
    s : int
        Base-MDP state index at stage ``t``.
    action_pair : tuple[int, int] | None
        Optional ``(a1, a2)`` contest pair.  When ``None`` the tie solver
        defaults to the top-2 actions by classical ``Q_cl(t, s, a)`` at the
        warm-start MDP (evaluated once and reused across iterations).
    """

    t: int
    s: int
    action_pair: tuple[int, int] | None = None


@dataclass
class FamilySpec:
    """Family protocol for the Phase V search (spec section 5 + planner-reply).

    Attributes
    ----------
    name : str
        Short family identifier (e.g. ``"family_a_jackpot"``).
    build_mdp : Callable[[float, dict], FiniteMDP]
        ``(lam, psi) -> FiniteMDP``.  Returns a finite-horizon MDP with the
        tie parameter ``lam`` and geometry parameter dict ``psi`` baked in.
        The returned MDP must expose ``info.gamma``, ``info.horizon`` and
        the MushroomRL ``p``/``r`` tensors (``[S, A, S']``).  A point-mass
        initial-state attribute ``initial_state: int`` is expected on the
        returned MDP (WP2 factories attach it explicitly).
    contest_state : ContestState
        Designated near-tie contest state for this family.
    warm_start_lambda : Callable[[dict], float]
        ``psi -> lam_hint``.  Closed-form tie hint if the family has one
        (A: ``c_tie = gamma^{L-1} R (1-gamma)/(1-gamma^L)``; B:
        ``b_safe = b - p gamma^{L-1} C``); otherwise the caller's
        heuristic initial guess.  Used both as brentq starting point and
        to cache the contest action pair when it is unspecified.
    scan_bracket : Callable[[dict], tuple[float, float]]
        ``psi -> (lo, hi)``.  Default bracket around ``warm_start_lambda``
        in which to search for the tie.  Bracket expansion in
        :func:`lambda_tie` is applied relative to this default when the
        caller supplies no explicit ``bracket`` argument.
    metadata : dict[str, Any]
        Free-form metadata (e.g. R_max, L, gamma) persisted to the
        candidate-metrics parquet by WP1c.  Not consumed by WP1a.
    """

    name: str
    build_mdp: Callable[[float, dict], "FiniteMDP"]
    contest_state: ContestState
    warm_start_lambda: Callable[[dict], float]
    scan_bracket: Callable[[dict], tuple[float, float]]
    metadata: dict[str, Any] = field(default_factory=dict)
