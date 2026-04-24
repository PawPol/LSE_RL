"""Phase V WP2 — Family E unit tests.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section
15 and ``tasks/todo.md`` task #15.

Checks: closed-form tie per variant (`|Delta_0| <= 1e-8`); ψ
validation; MDP well-formedness; start-state-flip reachability per
variant; concentration-contrast single-step invariant (E1, E2);
FamilySpec integrity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    evaluate_candidate,
)
from experiments.weighted_lse_dp.search.family_spec import (  # noqa: E402
    ContestState,
)
from experiments.weighted_lse_dp.tasks.family_e_regime_shift import (  # noqa: E402
    VARIANT_NAMES,
    build_family_e_mdp,
    contest_state,
    family_e,
    lam_tie_closed_form,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classical_qv(mdp) -> tuple[np.ndarray, np.ndarray]:
    """Finite-horizon classical DP; returns ``Q[T, S, A]``, ``V[T+1, S]``."""
    P = np.asarray(mdp.p, dtype=np.float64)                  # (S, A, S')
    R = np.asarray(mdp.r, dtype=np.float64)                  # (S, A, S')
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)
    S = P.shape[0]
    A = P.shape[1]
    r_bar = np.einsum("ijk,ijk->ij", P, R)                   # (S, A)
    V = np.zeros((T + 1, S), dtype=np.float64)
    Q = np.zeros((T, S, A), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        E_v = np.einsum("ijk,k->ij", P, V[t + 1])            # (S, A)
        Q[t] = r_bar + gamma * E_v
        V[t] = np.max(Q[t], axis=1)
    return Q, V


def _delta_0_contest(mdp) -> float:
    """``Q_cl[0, 0, 0] - Q_cl[0, 0, 1]`` at ``x_c = (t=0, s=0)``."""
    Q, _ = _classical_qv(mdp)
    return float(Q[0, 0, 0] - Q[0, 0, 1])


def _base_E1(**overrides: Any) -> dict[str, Any]:
    psi = {
        "variant": "E1_warning_react",
        "L": 6,
        "gamma": 0.95,
        "warning_lead": 2,
        "p_warn": 0.3,
        "C_conc": 10.0,
        "r_smooth": 0.3,
    }
    psi.update(overrides)
    return psi


def _base_E2(**overrides: Any) -> dict[str, Any]:
    psi = {
        "variant": "E2_opportunity_adapt",
        "L": 6,
        "gamma": 0.95,
        "warning_lead": 2,
        "p_opp": 0.5,
        "U_conc": 10.0,
        "r_baseline": 0.3,
    }
    psi.update(overrides)
    return psi


def _base_E3(**overrides: Any) -> dict[str, Any]:
    psi = {
        "variant": "E3_regime_switch",
        "L": 6,
        "gamma": 0.95,
        "t_switch": 2,
        "R_rev": 10.0,
        "r_smooth": 0.3,
    }
    psi.update(overrides)
    return psi


# ---------------------------------------------------------------------------
# 1. Closed-form tie per variant
# ---------------------------------------------------------------------------

E1_PSI_CASES = [
    # (L, gamma, warning_lead, p_warn, C_conc, r_smooth)
    (6, 0.95, 2, 0.3, 10.0, 0.3),
    (8, 0.99, 3, 0.5, 20.0, 0.1),
    (10, 0.90, 2, 0.1, 5.0, 0.5),
]


@pytest.mark.parametrize("L,gamma,wl,p_warn,C_conc,r_smooth", E1_PSI_CASES)
def test_E1_tie(L, gamma, wl, p_warn, C_conc, r_smooth) -> None:
    """E1 closed-form tie: |Delta_0| <= 1e-8 at lam_tie_closed_form(psi)."""
    psi = _base_E1(L=L, gamma=gamma, warning_lead=wl, p_warn=p_warn,
                   C_conc=C_conc, r_smooth=r_smooth)
    mdp = build_family_e_mdp(lam_tie_closed_form(psi), psi)
    delta = _delta_0_contest(mdp)
    assert abs(delta) <= 1e-8, f"E1 |Delta_0|={abs(delta):.3e} at psi={psi}"


E2_PSI_CASES = [
    # (L, gamma, warning_lead, p_opp, U_conc, r_baseline)
    (6, 0.95, 2, 0.5, 10.0, 0.3),
    (8, 0.99, 3, 0.3, 20.0, 0.1),
    (10, 0.90, 2, 0.7, 5.0, 0.2),
]


@pytest.mark.parametrize("L,gamma,wl,p_opp,U_conc,r_baseline", E2_PSI_CASES)
def test_E2_tie(L, gamma, wl, p_opp, U_conc, r_baseline) -> None:
    """E2 closed-form tie."""
    psi = _base_E2(L=L, gamma=gamma, warning_lead=wl, p_opp=p_opp,
                   U_conc=U_conc, r_baseline=r_baseline)
    mdp = build_family_e_mdp(lam_tie_closed_form(psi), psi)
    delta = _delta_0_contest(mdp)
    assert abs(delta) <= 1e-8, f"E2 |Delta_0|={abs(delta):.3e} at psi={psi}"


E3_PSI_CASES = [
    # (L, gamma, t_switch, R_rev, r_smooth)
    (6, 0.95, 2, 10.0, 0.3),
    (8, 0.99, 3, 20.0, 0.1),
    (10, 0.90, 1, 5.0, 0.2),
]


@pytest.mark.parametrize("L,gamma,t_switch,R_rev,r_smooth", E3_PSI_CASES)
def test_E3_tie(L, gamma, t_switch, R_rev, r_smooth) -> None:
    """E3 closed-form tie."""
    psi = _base_E3(L=L, gamma=gamma, t_switch=t_switch,
                   R_rev=R_rev, r_smooth=r_smooth)
    mdp = build_family_e_mdp(lam_tie_closed_form(psi), psi)
    delta = _delta_0_contest(mdp)
    assert abs(delta) <= 1e-8, f"E3 |Delta_0|={abs(delta):.3e} at psi={psi}"


# ---------------------------------------------------------------------------
# 2+3. Variant + bound validation
# ---------------------------------------------------------------------------

VALIDATION_CASES = [
    # (psi_builder, overrides, expected_match)
    (_base_E1, {"variant": "bogus"}, "variant"),
    (_base_E1, {"warning_lead": 0}, "warning_lead"),
    (_base_E1, {"warning_lead": 6}, "warning_lead"),           # == L
    (_base_E1, {"warning_lead": 10}, "warning_lead"),          # > L
    (_base_E1, {"L": 2, "warning_lead": 1}, "L"),              # L < 3
    (_base_E1, {"gamma": 1.0}, "gamma"),
    (_base_E1, {"p_warn": 0.0}, "p_warn"),
    (_base_E1, {"p_warn": 1.0}, "p_warn"),
    (_base_E1, {"C_conc": -1.0}, "C_conc"),
    (_base_E1, {"r_smooth": -0.1}, "r_smooth"),
    (_base_E2, {"warning_lead": 0}, "warning_lead"),
    (_base_E2, {"warning_lead": 6}, "warning_lead"),           # == L
    (_base_E2, {"p_opp": 0.0}, "p_opp"),
    (_base_E2, {"U_conc": -1.0}, "U_conc"),
    (_base_E2, {"r_baseline": -0.1}, "r_baseline"),
    (_base_E3, {"t_switch": 0}, "t_switch"),
    (_base_E3, {"t_switch": 6}, "t_switch"),                   # == L
    (_base_E3, {"t_switch": 10}, "t_switch"),                  # > L
    (_base_E3, {"R_rev": -1.0}, "R_rev"),
    (_base_E3, {"r_smooth": -0.1}, "r_smooth"),
]


@pytest.mark.parametrize("builder,overrides,match", VALIDATION_CASES)
def test_psi_validation_variant_and_bounds(
    builder, overrides: dict, match: str,
) -> None:
    """Every documented invariant violation surfaces as ValueError."""
    psi = builder(**overrides)
    with pytest.raises(ValueError, match=match):
        build_family_e_mdp(lam=0.0, psi=psi)


# ---------------------------------------------------------------------------
# 4. Start-state flip reachable on a tiny grid
# ---------------------------------------------------------------------------

def _build_safe_schedule(T: int, beta_cap_value: float) -> dict[str, Any]:
    """Stationary safe schedule at ``beta_used = beta_cap``.

    Mirrors ``tests/families/test_family_d._build_safe_schedule``.
    """
    beta_cap_t = np.full((T,), float(beta_cap_value), dtype=np.float64)
    beta_used_t = np.clip(beta_cap_t, -beta_cap_t, beta_cap_t)
    return {
        "beta_used_t": beta_used_t,
        "beta_cap_t": beta_cap_t,
        "beta_raw_t": beta_used_t.copy(),
    }


_FLIP_SWEEPS: dict[str, list[dict[str, Any]]] = {
    "E1_warning_react": [
        _base_E1(L=6, warning_lead=2, p_warn=0.3, C_conc=20.0, r_smooth=0.3, gamma=0.99),
        _base_E1(L=8, warning_lead=3, p_warn=0.5, C_conc=20.0, r_smooth=0.3, gamma=0.95),
        _base_E1(L=6, warning_lead=2, p_warn=0.5, C_conc=20.0, r_smooth=0.5, gamma=0.95),
        _base_E1(L=10, warning_lead=2, p_warn=0.3, C_conc=20.0, r_smooth=0.3, gamma=0.99),
    ],
    "E2_opportunity_adapt": [
        _base_E2(L=6, warning_lead=2, p_opp=0.5, U_conc=20.0, r_baseline=0.3, gamma=0.99),
        _base_E2(L=8, warning_lead=2, p_opp=0.3, U_conc=20.0, r_baseline=0.5, gamma=0.95),
        _base_E2(L=6, warning_lead=3, p_opp=0.7, U_conc=20.0, r_baseline=0.5, gamma=0.95),
        _base_E2(L=10, warning_lead=3, p_opp=0.5, U_conc=20.0, r_baseline=0.3, gamma=0.99),
    ],
    "E3_regime_switch": [
        _base_E3(L=6, t_switch=2, R_rev=20.0, r_smooth=0.3, gamma=0.99),
        _base_E3(L=8, t_switch=2, R_rev=20.0, r_smooth=0.3, gamma=0.95),
        _base_E3(L=6, t_switch=3, R_rev=20.0, r_smooth=0.5, gamma=0.95),
        _base_E3(L=10, t_switch=2, R_rev=20.0, r_smooth=0.3, gamma=0.99),
    ],
}


def _evaluate_start_state_flip(psi: dict[str, Any]) -> int:
    """Build at tie, calibrate a safe schedule, evaluate, return flip bit."""
    mdp = build_family_e_mdp(lam_tie_closed_form(psi), psi)
    schedule = _build_safe_schedule(T=int(mdp.info.horizon), beta_cap_value=0.25)
    metrics = evaluate_candidate(mdp=mdp, schedule=schedule,
                                 contest_state=contest_state)
    return int(metrics["start_state_flip"])


@pytest.mark.parametrize("variant", sorted(_FLIP_SWEEPS.keys()))
def test_start_state_flip_reachable(variant: str) -> None:
    """At least one psi per variant produces ``start_state_flip == 1``.

    Sanity check that the family CAN translate — NOT a promotion gate.
    """
    flips = [_evaluate_start_state_flip(psi) for psi in _FLIP_SWEEPS[variant]]
    assert any(f == 1 for f in flips), (
        f"Family E variant {variant!r} failed start_state_flip on "
        f"{len(flips)} psi; flips={flips}"
    )


# ---------------------------------------------------------------------------
# 5. MDP well-formed
# ---------------------------------------------------------------------------

_WELL_FORMED_PSIS = [
    _base_E1(L=8, warning_lead=3, p_warn=0.3, C_conc=10.0, r_smooth=0.3, gamma=0.95),
    _base_E2(L=8, warning_lead=3, p_opp=0.5, U_conc=10.0, r_baseline=0.3, gamma=0.95),
    _base_E3(L=8, t_switch=3, R_rev=10.0, r_smooth=0.3, gamma=0.95),
]


@pytest.mark.parametrize("psi", _WELL_FORMED_PSIS)
def test_mdp_well_formed(psi: dict[str, Any]) -> None:
    """P / R shapes, row sums, gamma, horizon, initial_state all valid."""
    lam = lam_tie_closed_form(psi)
    mdp = build_family_e_mdp(lam, psi)

    P = np.asarray(mdp.p, dtype=np.float64)                  # (S, A, S')
    R = np.asarray(mdp.r, dtype=np.float64)                  # (S, A, S')
    S, A = P.shape[0], P.shape[1]

    assert P.ndim == 3, f"P.ndim={P.ndim} (expected 3)"
    assert P.shape == (S, A, S), f"P.shape={P.shape}"
    assert R.shape == (S, A, S), f"R.shape={R.shape}"
    # Row-stochasticity (build_finite_mdp enforces this; we re-verify).
    row_sums = P.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-8), (
        f"row-sum deviation up to {float(np.max(np.abs(row_sums - 1.0))):.3e}"
    )
    # Absorbing terminal self-loops with reward 0.
    s_end = int(mdp.s_end_idx)
    assert abs(P[s_end, 0, s_end] - 1.0) <= 1e-12
    assert abs(P[s_end, 1, s_end] - 1.0) <= 1e-12
    assert R[s_end, 0, s_end] == 0.0
    assert R[s_end, 1, s_end] == 0.0
    # Phase V contract.
    L = int(psi["L"])
    assert float(mdp.info.gamma) == float(psi["gamma"])
    assert int(mdp.info.horizon) == L
    assert int(mdp.initial_state) == 0
    assert A == 2
    # State budget: S <= 3L + 4.
    assert S <= 3 * L + 4, (
        f"state budget exceeded: S={S} > 3L + 4 = {3 * L + 4}; psi={psi}"
    )


# ---------------------------------------------------------------------------
# 6. Concentration contrast — single-step concentrated reward/cost
# ---------------------------------------------------------------------------

def _count_reward_matches(R: np.ndarray, target: float, atol: float = 1e-12) -> int:
    """Count ``(s, a, s')`` transitions where ``R == target`` (up to ``atol``)."""
    return int(np.sum(np.abs(R - target) <= atol))


def test_concentration_contrast_E1() -> None:
    """E1: exactly ONE (s, s') transition pays ``r_smooth - C_conc``.

    Non-contest states share transitions across actions, so we collapse
    ``target_mask`` over the action axis before counting — we want the
    count of distinct ``(s, s')`` pairs, not ``(s, a, s')`` triples.
    """
    psi = _base_E1(L=8, warning_lead=3, p_warn=0.3, C_conc=10.0,
                   r_smooth=0.3, gamma=0.95)
    mdp = build_family_e_mdp(lam_tie_closed_form(psi), psi)
    R = np.asarray(mdp.r, dtype=np.float64)                  # (S, A, S')
    target = float(psi["r_smooth"]) - float(psi["C_conc"])
    target_mask = np.abs(R - target) <= 1e-12                # (S, A, S')
    n_edges = int(np.sum(target_mask.any(axis=1)))           # (S, S') count
    assert n_edges == 1, (
        f"E1: expected 1 (s, s') with reward {target:.3e}; got {n_edges}"
    )


def test_concentration_contrast_E2() -> None:
    """E2: exactly ONE (s, s') transition pays ``+U_conc``."""
    psi = _base_E2(L=8, warning_lead=3, p_opp=0.5, U_conc=10.0,
                   r_baseline=0.3, gamma=0.95)
    mdp = build_family_e_mdp(lam_tie_closed_form(psi), psi)
    R = np.asarray(mdp.r, dtype=np.float64)                  # (S, A, S')
    target = float(psi["U_conc"])
    target_mask = np.abs(R - target) <= 1e-12
    n_edges = int(np.sum(target_mask.any(axis=1)))
    assert n_edges == 1, (
        f"E2: expected 1 (s, s') with reward {target:.3e}; got {n_edges}"
    )


# ---------------------------------------------------------------------------
# Bonus: FamilySpec integrity
# ---------------------------------------------------------------------------

def test_family_e_spec_integrity() -> None:
    """family_e is a well-formed FamilySpec with Family E conventions."""
    assert family_e.name == "regime_shift_concentration"
    assert isinstance(family_e.contest_state, ContestState)
    assert family_e.contest_state.t == 0
    assert family_e.contest_state.s == 0
    assert family_e.contest_state.action_pair == (0, 1)

    # warm_start_lambda returns the closed-form tie on every variant.
    for builder in (_base_E1, _base_E2, _base_E3):
        psi = builder()
        want = lam_tie_closed_form(psi)
        got = family_e.warm_start_lambda(psi)
        assert abs(got - want) <= 1e-12

    # scan_bracket endpoints contain lam_tie strictly.
    for builder in (_base_E1, _base_E2, _base_E3):
        psi = builder()
        lam = lam_tie_closed_form(psi)
        lo, hi = family_e.scan_bracket(psi)
        assert lo <= lam <= hi, (
            f"scan_bracket for {psi['variant']}: lam_tie={lam:.3e} "
            f"not in [{lo:.3e}, {hi:.3e}]"
        )
        assert lo < hi

    # VARIANT_NAMES exhaustive (no hidden variants, no duplicates).
    assert set(VARIANT_NAMES) == {
        "E1_warning_react",
        "E2_opportunity_adapt",
        "E3_regime_switch",
    }
    assert len(VARIANT_NAMES) == 3
