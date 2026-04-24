"""Phase V WP2 — Family A unit tests.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5
Family A and ``tasks/todo.md`` WP2 test checklist.

Three checks:
1. Closed-form tie: for 3 parameter triples ``(L, R, gamma)``, solve
   classical DP on ``build_family_a_mdp(c_tie_closed_form(...), psi={"shape":"flat"})``
   and assert ``|Delta_0(x_c)| <= 1e-8``.
2. Shape-basis zero-sum-under-gamma-weight: for each of the 5 shapes,
   assert ``|sum_k gamma^k h_k(psi)| <= 1e-12``.
3. ``lambda_tie`` bisection recovers the closed-form tie within ``1e-8``
   for the ``"flat"`` shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.search.family_spec import (  # noqa: E402
    ContestState,
    FamilySpec,
)
from experiments.weighted_lse_dp.search.tie_solver import (  # noqa: E402
    lambda_tie,
)
from experiments.weighted_lse_dp.tasks.family_a_jackpot_vs_stream import (  # noqa: E402
    SHAPE_NAMES,
    build_family_a_mdp,
    c_tie_closed_form,
    family_a,
    project_to_gamma_null_space,
)
from experiments.weighted_lse_dp.tasks._family_helpers import (  # noqa: E402
    shape_basis,
)


# ---------------------------------------------------------------------------
# Helpers (mini classical DP, matched to the WP1a tie solver)
# ---------------------------------------------------------------------------

def _classical_qv(mdp) -> tuple[np.ndarray, np.ndarray]:
    """Finite-horizon classical DP on ``mdp.p`` / ``mdp.r`` / ``mdp.info``."""
    P = np.asarray(mdp.p, dtype=np.float64)
    R = np.asarray(mdp.r, dtype=np.float64)
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)
    S = P.shape[0]
    r_bar = np.einsum("ijk,ijk->ij", P, R)
    V = np.zeros((T + 1, S), dtype=np.float64)
    Q = np.zeros((T, S, P.shape[1]), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        E_v = np.einsum("ijk,k->ij", P, V[t + 1])
        Q[t] = r_bar + gamma * E_v
        V[t] = np.max(Q[t], axis=1)
    return Q, V


# ---------------------------------------------------------------------------
# 1. Closed-form tie check on the FiniteMDP builder
# ---------------------------------------------------------------------------

CLOSED_FORM_TRIPLES = [
    (4, 1.0, 0.9),
    (6, 2.0, 0.95),
    (10, 0.5, 0.99),
]


@pytest.mark.parametrize("L,R,gamma", CLOSED_FORM_TRIPLES)
def test_family_a_closed_form_tie(L: int, R: float, gamma: float) -> None:
    """At ``lam = c_tie_closed_form(...)``, classical Delta_0(x_c) ~= 0."""
    c_tie = c_tie_closed_form(L, R, gamma)
    psi = {"L": L, "R": R, "gamma": gamma, "shape": "flat"}
    mdp = build_family_a_mdp(c_tie, psi)
    Q, _ = _classical_qv(mdp)
    t_c, s_c = 0, 0
    a1, a2 = 0, 1
    delta = float(Q[t_c, s_c, a1] - Q[t_c, s_c, a2])
    assert abs(delta) <= 1e-8, (
        f"closed-form tie residual too large: |Delta_0|={abs(delta):.3e} "
        f"for L={L}, R={R}, gamma={gamma}"
    )


# ---------------------------------------------------------------------------
# 2. Shape-basis zero-sum-under-gamma-weight
# ---------------------------------------------------------------------------

ZERO_SUM_CASES = [
    # (shape, L, gamma, extra_psi_keys)
    ("flat", 8, 0.9, {}),
    ("flat", 12, 0.99, {}),
    ("front_loaded_compensated", 8, 0.9, {"bump_width": 3, "bump_strength": 0.5}),
    ("front_loaded_compensated", 12, 0.99, {"bump_width": 5, "bump_strength": 1.0}),
    ("one_bump", 8, 0.9, {"bump_center": 3.0, "bump_width": 1.5, "bump_strength": 0.5}),
    ("one_bump", 12, 0.99, {"bump_center": 6.0, "bump_width": 2.0}),
    ("two_bump", 10, 0.95, {}),
    (
        "two_bump",
        12,
        0.99,
        {
            "bump_centers": (3.0, 8.0),
            "bump_widths": (1.0, 1.2),
            "bump_strengths": (0.7, -0.6),
        },
    ),
    ("ramp", 8, 0.9, {"ramp_amplitude": 0.5}),
    ("ramp", 12, 0.99, {"ramp_amplitude": 1.0}),
]


@pytest.mark.parametrize("shape,L,gamma,extra", ZERO_SUM_CASES)
def test_shape_basis_zero_sum(
    shape: str,
    L: int,
    gamma: float,
    extra: dict,
) -> None:
    """|sum_k gamma^k h_k(psi)| <= 1e-12 for every shape."""
    psi = {"shape": shape, **extra}
    h = shape_basis(L, gamma, psi)                                  # (L,)
    assert h.shape == (L,), f"unexpected shape: {h.shape}"
    k = np.arange(L, dtype=np.float64)
    gk = np.power(float(gamma), k)                                  # (L,)
    total = float(np.dot(gk, h))
    assert abs(total) <= 1e-12, (
        f"shape={shape!r} L={L} gamma={gamma}: |sum_k gamma^k h_k|={abs(total):.3e}"
    )


def test_shape_basis_names_exhaustive() -> None:
    """SHAPE_NAMES enumerates every valid shape string."""
    assert set(SHAPE_NAMES) == {
        "flat",
        "front_loaded_compensated",
        "one_bump",
        "two_bump",
        "ramp",
    }


def test_project_idempotent() -> None:
    """Projecting twice equals projecting once."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        L = int(rng.integers(4, 15))
        gamma = float(rng.uniform(0.85, 0.999))
        h0 = rng.normal(size=L)
        h1 = project_to_gamma_null_space(h0, gamma)
        h2 = project_to_gamma_null_space(h1, gamma)
        assert np.allclose(h1, h2, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. lambda_tie bisection recovers the closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L,R,gamma", CLOSED_FORM_TRIPLES)
def test_lambda_tie_bisection_recovers_closed_form(
    L: int,
    R: float,
    gamma: float,
) -> None:
    """Bisect |Delta_0| via tie_solver; assert |lam_solved - c_tie| <= 1e-8."""
    psi = {"L": L, "R": R, "gamma": gamma, "shape": "flat"}
    c_closed = c_tie_closed_form(L, R, gamma)
    lam_solved, diag = lambda_tie(psi, family_a, tol=1e-12)
    assert abs(lam_solved - c_closed) <= 1e-8, (
        f"bisection residual: solved={lam_solved}, closed={c_closed}, "
        f"diff={abs(lam_solved - c_closed):.3e}"
    )
    assert diag["final_gap_abs"] <= 1e-7
