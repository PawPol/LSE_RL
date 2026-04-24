"""Phase V WP2 — Family C: misaligned raw-operator stress / safety.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5
("Family C — misaligned stress / safety").

Family C is the **safety/stability** family.  It is engineered so that a
raw weighted-LSE schedule with elevated temperature enters a locally
expansive region on visited states while the safe clipped counterpart
stays stable.  Family C is NOT a classical-tie family: the two contested
actions at ``s = 0`` are constructed to behave identically under the
classical Bellman target, so there is no tie-parameter sweep in the
Family A / Family B sense.

The engineered stress branch is a deterministic tail chain of length
``L_tail`` where every step pays a negative reward ``-R_penalty`` and the
terminal step pays a positive reward ``+R_terminal`` on absorption.
Under ``gamma`` near 1 and ``R_terminal`` large enough that the tail
states carry positive ``V_cl``, the safe / classical Bellman target sees
a large negative signed margin ``r - V_next`` at every visited
transition.  When the raw schedule sits at
``beta_raw = beta_raw_multiplier * beta_cap`` with multiplier > 1, the
local raw continuation derivative

    d_raw(r, v) = (1 + gamma) * (1 - sigmoid(beta_raw * (r - v) + log(1/gamma)))

inflates toward ``(1 + gamma)``, far above the certified safe bound
``gamma``.  The clip is triggered on every stage (because
``|beta_raw| > beta_cap`` by construction), so ``clip_fraction`` on
visited transitions exceeds the 0.05 threshold.

This realizes the Family C design constraint from ``tasks/todo.md`` WP2:
``raw_local_deriv_stats["p90"] > certified_bound`` and
``clip_fraction > 0.05``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.weighted_lse_dp.search.family_spec import (
    ContestState,
    FamilySpec,
)

from ._family_helpers import FiniteMDP, build_finite_mdp

__all__ = [
    "build_family_c_mdp",
    "family_c",
    "contest_state",
]


# ---------------------------------------------------------------------------
# MDP construction
# ---------------------------------------------------------------------------

# State layout:
#
#   s = 0                         : contest state (2 actions; both route
#                                     deterministically into the tail chain).
#   s = 1, 2, ..., L_tail - 1     : interior tail states.  Each edge pays
#                                     ``-R_penalty``.
#   s = L_tail                    : absorbing terminal; last entry edge
#                                     pays ``-R_penalty + R_terminal``.
#
# Horizon T = L_tail.

def build_family_c_mdp(lam: float, psi: dict[str, Any]) -> FiniteMDP:
    """Build the Family C raw-stress MDP.

    Parameters
    ----------
    lam : float
        **Unused** — Family C is not a classical-tie family.  The
        ``FamilySpec`` signature still requires a ``lam``, so it is
        accepted and ignored.
    psi : dict
        Geometry parameters.  Required keys:
            * ``L_tail`` (int >= 2) — depth of the tail chain.
            * ``R_penalty`` (float > 0) — magnitude of the per-step
              misaligned (negative) reward.
            * ``R_terminal`` (float) — magnitude of the terminal reward;
              positive values put the tail states on a positive classical
              V path, producing the ``r - V_next < 0`` regime that
              inflates the raw derivative.
            * ``gamma`` (float in (0, 1]).
        Optional keys (consumed by WP1c + WP4, not by this builder):
            * ``beta_raw_multiplier`` (float > 1) — how far above
              ``beta_cap`` the raw schedule sits.

    Returns
    -------
    FiniteMDP
        A MushroomRL ``FiniteMDP`` with ``L_tail + 1`` states, 2 actions,
        finite horizon ``L_tail``, point-mass initial state at ``s = 0``.
    """
    L_tail = int(psi["L_tail"])
    if L_tail < 2:
        raise ValueError(f"L_tail must be >= 2; got L_tail={L_tail}")
    R_penalty = float(psi["R_penalty"])
    if R_penalty <= 0.0:
        raise ValueError(f"R_penalty must be > 0; got R_penalty={R_penalty}")
    R_terminal = float(psi.get("R_terminal", 10.0 * R_penalty))
    gamma = float(psi["gamma"])
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must lie in (0, 1]; got gamma={gamma}")

    S = L_tail + 1
    A = 2
    P = np.zeros((S, A, S), dtype=np.float64)                # (S, A, S')
    R = np.zeros((S, A, S), dtype=np.float64)                # (S, A, S')

    # Contest state s=0: both actions route to s=1 with reward -R_penalty.
    # This makes the classical Delta_0(x_c) == 0 exactly (not a tie in the
    # policy-flip sense; the two actions are genuinely identical).  The
    # action_pair=(0, 1) on ContestState is still well-formed.
    for a in range(A):
        P[0, a, 1] = 1.0
        R[0, a, 1] = -R_penalty

    # Tail chain s=i -> s=i+1 for i = 1, ..., L_tail - 1.
    # Reward on each interior edge is -R_penalty.
    # Reward on the final edge (s = L_tail - 1 -> s = L_tail) is
    # -R_penalty + R_terminal, so the terminal absorbing state is entered
    # with a positive-valued "bailout" that makes V_cl at early states
    # positive and produces the ``r - V_next < 0`` pattern.
    for i in range(1, L_tail):
        for a in range(A):
            P[i, a, i + 1] = 1.0
            if i == L_tail - 1:
                R[i, a, i + 1] = -R_penalty + R_terminal
            else:
                R[i, a, i + 1] = -R_penalty

    # Absorbing terminal.
    for a in range(A):
        P[L_tail, a, L_tail] = 1.0

    return build_finite_mdp(
        P=P,
        R=R,
        gamma=gamma,
        horizon=L_tail,
        initial_state=0,
    )


# ---------------------------------------------------------------------------
# FamilySpec
# ---------------------------------------------------------------------------

contest_state: ContestState = ContestState(t=0, s=0, action_pair=(0, 1))


def _warm_start(psi: dict[str, Any]) -> float:
    """No classical tie — return 0.0 per the WP2 FamilySpec fragment."""
    del psi
    return 0.0


def _scan_bracket(psi: dict[str, Any]) -> tuple[float, float]:
    """No classical-tie sweep — return a token ``(-1, 1)`` bracket."""
    del psi
    return (-1.0, 1.0)


family_c: FamilySpec = FamilySpec(
    name="raw_stress",
    build_mdp=build_family_c_mdp,
    contest_state=contest_state,
    warm_start_lambda=_warm_start,
    scan_bracket=_scan_bracket,
    metadata={
        "family": "C",
        "description": "misaligned raw-operator stress / safety",
        "tie_parameter": None,
        "geometry_parameter_keys": (
            "L_tail", "R_penalty", "R_terminal", "gamma",
            "beta_raw_multiplier",
        ),
        "classical_tie_sweep_supported": False,
    },
)
