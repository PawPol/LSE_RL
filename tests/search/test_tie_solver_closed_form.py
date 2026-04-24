"""Phase V WP1a -- tie solver closed-form agreement tests.

Family A (jackpot vs stream) and Family B (catastrophe vs safe branch)
both have closed-form tie expressions; we verify brentq recovers them
to 1e-10.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.search.family_spec import (  # noqa: E402
    ContestState,
    FamilySpec,
)
from experiments.weighted_lse_dp.search.tie_solver import (  # noqa: E402
    TieNotBracketed,
    lambda_tie,
)


# ---------------------------------------------------------------------------
# Minimal FiniteMDP shim (avoids importing MushroomRL for unit tests)
# ---------------------------------------------------------------------------

class _ShimMDP:
    """Numpy-only stand-in for MushroomRL's FiniteMDP.

    Exposes ``p`` (S, A, S'), ``r`` (S, A, S'), and ``info.gamma`` /
    ``info.horizon`` so that :func:`tie_solver._classical_q` can plan on
    it.  Also carries an ``initial_state`` attribute consumed by the
    WP1a occupancy helper.
    """

    def __init__(self, p: np.ndarray, r: np.ndarray, gamma: float, T: int, s0: int = 0):
        self.p = np.asarray(p, dtype=np.float64)
        self.r = np.asarray(r, dtype=np.float64)
        self.info = SimpleNamespace(gamma=float(gamma), horizon=int(T))
        self.initial_state = int(s0)


# ---------------------------------------------------------------------------
# Family A: jackpot vs stream
# ---------------------------------------------------------------------------

def _build_family_a(c: float, psi: dict) -> _ShimMDP:
    """Family A MDP: two-branch contest at s0, closed-form tie at ``c_tie``.

    State layout:
      s=0 : start state (2 actions)
      s=1..L  : jackpot-chain states 1..L (absorbing at s=L)
      s=L+1..2L : stream-chain states L+1..2L (absorbing at s=2L)

    Horizon T = L.  Gamma and L come from psi.
    """
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    R = float(psi["R"])
    n_states = 2 * L + 1
    S = n_states
    A = 2
    P = np.zeros((S, A, S), dtype=np.float64)
    r = np.zeros((S, A, S), dtype=np.float64)

    # s=0: action 0 -> jackpot chain; action 1 -> stream chain.
    P[0, 0, 1] = 1.0
    P[0, 1, L + 1] = 1.0
    # reward on the step *out of* s=0 is 0 for jackpot, c for stream.
    r[0, 0, 1] = 0.0
    r[0, 1, L + 1] = float(c)

    # Jackpot chain: s=1..L-1 -> next state (reward 0); s=L-1 -> s=L (reward R).
    for i in range(1, L):
        for a in range(A):
            P[i, a, i + 1] = 1.0
            # reward R on the final edge of the jackpot chain
            if i == L - 1:
                r[i, a, i + 1] = R
    # s=L absorbing (self-loop, zero reward).
    for a in range(A):
        P[L, a, L] = 1.0

    # Stream chain: s=L+1..2L-1 -> next; each edge pays c.
    for i in range(L + 1, 2 * L):
        for a in range(A):
            P[i, a, i + 1] = 1.0
            r[i, a, i + 1] = float(c)
    # s=2L absorbing.
    for a in range(A):
        P[2 * L, a, 2 * L] = 1.0

    return _ShimMDP(P, r, gamma=gamma, T=L, s0=0)


def _family_a_spec() -> FamilySpec:
    def build_mdp(lam: float, psi: dict) -> _ShimMDP:
        return _build_family_a(c=lam, psi=psi)

    def warm_start(psi: dict) -> float:
        gamma = float(psi["gamma"])
        L = int(psi["L"])
        R = float(psi["R"])
        return gamma ** (L - 1) * R * (1.0 - gamma) / (1.0 - gamma ** L)

    def bracket(psi: dict) -> tuple[float, float]:
        hint = warm_start(psi)
        half = max(abs(hint), 1.0) * 0.5
        return (hint - half, hint + half)

    return FamilySpec(
        name="family_a_shim",
        build_mdp=build_mdp,
        contest_state=ContestState(t=0, s=0, action_pair=(0, 1)),
        warm_start_lambda=warm_start,
        scan_bracket=bracket,
    )


# ---------------------------------------------------------------------------
# Family B: catastrophe vs safe
# ---------------------------------------------------------------------------

def _build_family_b(b_safe: float, psi: dict) -> _ShimMDP:
    """Family B MDP: catastrophe branch vs deterministic safe branch.

    Here ``b_safe`` is the **tie parameter** -- the deterministic safe
    reward on action 1 -- which the tie solver will adjust.  All other
    parameters (b, p, C, L, gamma) come from psi and are fixed.

    State layout:
      s=0 : start (2 actions)
      s=1..L : catastrophe chain (with split on prob p at s=1)
      s=L+1..2L : zero chain (safe fallback from the stochastic split)
      s=2L+1 : absorbing terminal for the safe branch.
    """
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    b = float(psi["b"])
    p = float(psi["p"])
    C = float(psi["C"])

    S = 2 * L + 2
    A = 2
    P = np.zeros((S, A, S), dtype=np.float64)
    r = np.zeros((S, A, S), dtype=np.float64)

    # s=0: action 0 -> catastrophe branch (both transitions pay immediate b);
    #      action 1 -> safe terminal with reward b_safe.
    # Transition from s=0 (action 0) to s=1 always, reward b.
    P[0, 0, 1] = 1.0
    r[0, 0, 1] = b
    # Transition from s=0 (action 1) to absorbing safe state with reward b_safe.
    P[0, 1, 2 * L + 1] = 1.0
    r[0, 1, 2 * L + 1] = float(b_safe)

    # Catastrophe branch traversal: at s=1 split with prob p on the first
    # step -> enter catastrophe chain (terminal reward -C at depth L-1 from
    # s=1, i.e. s=L).  With prob (1-p) -> enter zero-chain.
    # We embed the split at s=1 -> {s=2 (cat chain), s=L+1 (zero chain)}.
    for a in range(A):
        P[1, a, 2] = p
        P[1, a, L + 1] = 1.0 - p
        # no reward on the split step
    # Catastrophe chain: s=2..L-1 -> next, zero reward; s=L-1 -> s=L reward -C.
    for i in range(2, L):
        for a in range(A):
            P[i, a, i + 1] = 1.0
            if i == L - 1:
                r[i, a, i + 1] = -C
    # s=L absorbing.
    for a in range(A):
        P[L, a, L] = 1.0

    # Zero chain: s=L+1..2L-1 -> next (zero reward); s=2L absorbing.
    for i in range(L + 1, 2 * L):
        for a in range(A):
            P[i, a, i + 1] = 1.0
    for a in range(A):
        P[2 * L, a, 2 * L] = 1.0

    # Safe terminal absorbing.
    for a in range(A):
        P[2 * L + 1, a, 2 * L + 1] = 1.0

    return _ShimMDP(P, r, gamma=gamma, T=L, s0=0)


def _family_b_spec() -> FamilySpec:
    def build_mdp(lam: float, psi: dict) -> _ShimMDP:
        return _build_family_b(b_safe=lam, psi=psi)

    def warm_start(psi: dict) -> float:
        gamma = float(psi["gamma"])
        L = int(psi["L"])
        b = float(psi["b"])
        p = float(psi["p"])
        C = float(psi["C"])
        return b - p * gamma ** (L - 1) * C

    def bracket(psi: dict) -> tuple[float, float]:
        hint = warm_start(psi)
        half = max(abs(hint), 1.0) * 0.5
        return (hint - half, hint + half)

    return FamilySpec(
        name="family_b_shim",
        build_mdp=build_mdp,
        contest_state=ContestState(t=0, s=0, action_pair=(0, 1)),
        warm_start_lambda=warm_start,
        scan_bracket=bracket,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

FAMILY_A_TRIPLES = [
    # (L, R, gamma)
    (4, 1.0, 0.9),
    (6, 2.0, 0.95),
    (10, 0.5, 0.99),
    (3, 3.0, 0.8),
]


@pytest.mark.parametrize("L,R,gamma", FAMILY_A_TRIPLES)
def test_family_a_closed_form(L: int, R: float, gamma: float) -> None:
    family = _family_a_spec()
    psi = {"L": L, "R": R, "gamma": gamma}
    c_closed = gamma ** (L - 1) * R * (1.0 - gamma) / (1.0 - gamma ** L)
    lam_solved, diag = lambda_tie(psi, family, tol=1e-12)
    assert abs(lam_solved - c_closed) <= 1e-10, (
        f"L={L} R={R} gamma={gamma}: closed={c_closed} solved={lam_solved} "
        f"diff={abs(lam_solved - c_closed):.3e}"
    )
    assert diag["final_gap_abs"] <= 1e-8


FAMILY_B_TRIPLES = [
    # (L, b, p, C, gamma)
    (4, 1.0, 0.1, 10.0, 0.9),
    (6, 0.5, 0.05, 5.0, 0.95),
    (8, 2.0, 0.2, 3.0, 0.99),
    (3, 1.5, 0.3, 2.0, 0.85),
]


@pytest.mark.parametrize("L,b,p,C,gamma", FAMILY_B_TRIPLES)
def test_family_b_closed_form(L: int, b: float, p: float, C: float, gamma: float) -> None:
    family = _family_b_spec()
    psi = {"L": L, "gamma": gamma, "b": b, "p": p, "C": C}
    b_safe_closed = b - p * gamma ** (L - 1) * C
    lam_solved, diag = lambda_tie(psi, family, tol=1e-12)
    assert abs(lam_solved - b_safe_closed) <= 1e-10, (
        f"L={L} b={b} p={p} C={C} gamma={gamma}: "
        f"closed={b_safe_closed} solved={lam_solved} "
        f"diff={abs(lam_solved - b_safe_closed):.3e}"
    )
    assert diag["final_gap_abs"] <= 1e-8


def test_not_bracketed_raises() -> None:
    """If Delta_0 has the same sign across an absurd bracket, raise.

    We pick a narrow bracket ``[M, M + epsilon]`` with large ``M`` so that
    the 8x expansion stays inside ``[M - 4 epsilon, M + 4 epsilon]`` -- still
    on the same (negative-Delta_0) side of the tie.
    """
    family = _family_a_spec()
    psi = {"L": 5, "R": 1.0, "gamma": 0.9}
    M = 1.0e6
    eps = 1.0e-3
    with pytest.raises(TieNotBracketed) as excinfo:
        lambda_tie(psi, family, bracket=(M, M + eps))
    assert "did not change sign" in str(excinfo.value)
    assert "bracket_final" in excinfo.value.scan_record
