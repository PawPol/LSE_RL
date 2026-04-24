"""Phase V WP2 — Family D unit tests.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` sections
14.1 / 14.4 and ``tasks/todo.md`` task #14.

Five checks:

1. ``test_single_warning_tie`` — for 3 parameter sextuples
   ``(L, C, gamma, p_warn, warning_lead, p_cat_given_warn=1.0)``, build the
   MDP at ``lam = lam_tie_closed_form(...)`` and assert
   ``|Delta_0(x_contest)| <= 1e-8`` via exact classical DP.
2. ``test_shallow_early_warning_tie`` — same check for 2 septuples with
   ``r_shallow > 0``.
3. ``test_psi_validation`` — ``build_family_d_mdp`` raises
   ``ValueError`` on every documented invariant violation.
4. ``test_start_state_flip_reachable`` — sweep a tiny psi grid, evaluate
   each candidate with a small safe schedule, assert at least one grid
   point produces ``start_state_flip == 1``.
5. ``test_mdp_well_formed`` — shape / row-sum / gamma / horizon /
   initial_state invariants on one representative MDP per variant.
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

from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    evaluate_candidate,
)
from experiments.weighted_lse_dp.search.family_spec import (  # noqa: E402
    ContestState,
)
from experiments.weighted_lse_dp.tasks.family_d_preventive import (  # noqa: E402
    VARIANT_NAMES,
    build_family_d_mdp,
    contest_state,
    family_d,
    lam_tie_closed_form,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classical_qv(mdp) -> tuple[np.ndarray, np.ndarray]:
    """Finite-horizon classical DP; returns ``Q[T, S, A]``, ``V[T+1, S]``."""
    P = np.asarray(mdp.p, dtype=np.float64)                # (S, A, S')
    R = np.asarray(mdp.r, dtype=np.float64)
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)
    S = P.shape[0]
    A = P.shape[1]
    r_bar = np.einsum("ijk,ijk->ij", P, R)
    V = np.zeros((T + 1, S), dtype=np.float64)
    Q = np.zeros((T, S, A), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        E_v = np.einsum("ijk,k->ij", P, V[t + 1])
        Q[t] = r_bar + gamma * E_v
        V[t] = np.max(Q[t], axis=1)
    return Q, V


def _delta_0_contest(mdp) -> float:
    """``Q_cl[0, 0, 0] - Q_cl[0, 0, 1]`` at ``x_c = (t=0, s=0)``."""
    Q, _ = _classical_qv(mdp)
    return float(Q[0, 0, 0] - Q[0, 0, 1])


def _base_psi(**overrides) -> dict:
    """Base single_warning psi; override any key."""
    psi = {
        "variant": "single_warning",
        "L": 6,
        "C": 5.0,
        "gamma": 0.95,
        "p_warn": 0.3,
        "p_cat_given_warn": 1.0,
        "warning_lead": 2,
    }
    psi.update(overrides)
    return psi


# ---------------------------------------------------------------------------
# 1. single_warning closed-form tie
# ---------------------------------------------------------------------------

SINGLE_WARNING_SEXTUPLES = [
    # (L, C, gamma, p_warn, warning_lead, p_cat_given_warn)
    (6, 5.0, 0.95, 0.3, 2, 1.0),
    (10, 20.0, 0.99, 0.1, 3, 1.0),
    (8, 10.0, 0.9, 0.5, 2, 1.0),
]


@pytest.mark.parametrize(
    "L,C,gamma,p_warn,warning_lead,p_cat", SINGLE_WARNING_SEXTUPLES,
)
def test_single_warning_tie(
    L: int, C: float, gamma: float, p_warn: float,
    warning_lead: int, p_cat: float,
) -> None:
    """|Delta_0(x_c)| <= 1e-8 at the closed-form tie for single_warning."""
    lam = lam_tie_closed_form(
        p_warn=p_warn, p_cat_given_warn=p_cat, gamma=gamma,
        L=L, C=C, r_shallow=0.0, warning_lead=warning_lead,
    )
    psi = _base_psi(
        L=L, C=C, gamma=gamma, p_warn=p_warn,
        p_cat_given_warn=p_cat, warning_lead=warning_lead,
    )
    mdp = build_family_d_mdp(lam, psi)
    delta = _delta_0_contest(mdp)
    assert abs(delta) <= 1e-8, (
        f"single_warning tie residual: |Delta_0|={abs(delta):.3e} "
        f"for L={L}, C={C}, gamma={gamma}, p_warn={p_warn}, "
        f"warning_lead={warning_lead}"
    )


# ---------------------------------------------------------------------------
# 2. shallow_early_warning closed-form tie
# ---------------------------------------------------------------------------

SHALLOW_SEPTUPLES = [
    # (L, C, gamma, p_warn, warning_lead, p_cat_given_warn, r_shallow_frac_of_C)
    (6, 10.0, 0.95, 0.3, 2, 1.0, 0.1),   # r_shallow = 1.0
    (10, 5.0, 0.99, 0.5, 3, 1.0, 0.1),   # r_shallow = 0.5
]


@pytest.mark.parametrize(
    "L,C,gamma,p_warn,warning_lead,p_cat,r_sh_frac", SHALLOW_SEPTUPLES,
)
def test_shallow_early_warning_tie(
    L: int, C: float, gamma: float, p_warn: float,
    warning_lead: int, p_cat: float, r_sh_frac: float,
) -> None:
    """|Delta_0(x_c)| <= 1e-8 at the closed-form tie for shallow_early_warning."""
    r_shallow = r_sh_frac * C
    lam = lam_tie_closed_form(
        p_warn=p_warn, p_cat_given_warn=p_cat, gamma=gamma,
        L=L, C=C, r_shallow=r_shallow, warning_lead=warning_lead,
    )
    # Default shallow ratio keeps lam non-negative (see docstring of
    # lam_tie_closed_form; gamma^warning_lead > 0.1 at every valid ψ).
    assert lam >= 0.0, f"lam_tie became negative: {lam}"
    psi = _base_psi(
        variant="shallow_early_warning",
        L=L, C=C, gamma=gamma, p_warn=p_warn,
        p_cat_given_warn=p_cat, warning_lead=warning_lead,
        r_shallow=r_shallow,
    )
    mdp = build_family_d_mdp(lam, psi)
    delta = _delta_0_contest(mdp)
    assert abs(delta) <= 1e-8, (
        f"shallow_early_warning tie residual: |Delta_0|={abs(delta):.3e} "
        f"for L={L}, C={C}, gamma={gamma}, p_warn={p_warn}, "
        f"warning_lead={warning_lead}, r_shallow={r_shallow}"
    )


# ---------------------------------------------------------------------------
# 3. psi validation — ValueError on every documented invariant violation
# ---------------------------------------------------------------------------

VALIDATION_CASES = [
    # (override_dict, expected_match_substring)
    ({"warning_lead": 0}, "warning_lead"),          # below 1
    ({"warning_lead": 6}, "warning_lead"),          # == L
    ({"warning_lead": 10}, "warning_lead"),         # > L
    ({"L": 2, "warning_lead": 1}, "L"),             # L < 3
    ({"p_warn": 0.0}, "p_warn"),                    # boundary 0
    ({"p_warn": 1.0}, "p_warn"),                    # boundary 1
    ({"C": -1.0}, "C"),                             # non-positive C
    ({"gamma": 1.0}, "gamma"),                      # boundary gamma
    ({"p_cat_given_warn": 0.0}, "p_cat_given_warn"),
    ({"variant": "bogus"}, "variant"),
    ({"r_shallow": 0.5}, "r_shallow"),              # single_warning requires r_shallow==0
]


@pytest.mark.parametrize("overrides,match", VALIDATION_CASES)
def test_psi_validation(overrides: dict, match: str) -> None:
    """Every documented invariant violation surfaces as ValueError."""
    psi = _base_psi(**overrides)
    with pytest.raises(ValueError, match=match):
        build_family_d_mdp(lam=0.0, psi=psi)


# ---------------------------------------------------------------------------
# 4. start_state_flip reachable on a small grid
# ---------------------------------------------------------------------------

def _build_safe_schedule(T: int, beta_cap_value: float) -> dict:
    """Stationary safe schedule at ``beta_used = beta_cap``.

    Mirrors ``tests/families/test_family_c._build_schedule`` at
    multiplier = 1.0.  No phase4_calibration_v3 dependency; Phase V
    search owns schedule construction downstream.  We only need a finite
    ``beta_used`` here to drive the safe DP away from the classical fixed
    point.
    """
    beta_cap_t = np.full((T,), float(beta_cap_value), dtype=np.float64)
    beta_used_t = np.clip(beta_cap_t, -beta_cap_t, beta_cap_t)   # == beta_cap_t
    return {
        "beta_used_t": beta_used_t,
        "beta_cap_t": beta_cap_t,
        "beta_raw_t": beta_used_t.copy(),
    }


def test_start_state_flip_reachable() -> None:
    """At least one psi in a tiny grid produces ``start_state_flip == 1``.

    Sanity check that the family CAN translate — NOT a promotion gate.
    """
    grid = [
        # (L, C, gamma, p_warn, warning_lead, variant)
        (6, 20.0, 0.99, 0.3, 2, "single_warning"),
        (6, 10.0, 0.95, 0.5, 3, "single_warning"),
        (10, 20.0, 0.99, 0.3, 2, "single_warning"),
        (10, 10.0, 0.95, 0.5, 2, "shallow_early_warning"),
    ]
    any_flip = False
    flip_details: list[str] = []
    for (L, C, gamma, p_warn, warning_lead, variant) in grid:
        r_shallow = 0.1 * C if variant == "shallow_early_warning" else 0.0
        psi = _base_psi(
            variant=variant, L=L, C=C, gamma=gamma, p_warn=p_warn,
            warning_lead=warning_lead, r_shallow=r_shallow,
        )
        lam = lam_tie_closed_form(
            p_warn=p_warn, p_cat_given_warn=1.0, gamma=gamma,
            L=L, C=C, r_shallow=r_shallow, warning_lead=warning_lead,
        )
        mdp = build_family_d_mdp(lam, psi)
        schedule = _build_safe_schedule(
            T=int(mdp.info.horizon), beta_cap_value=0.25,
        )
        metrics = evaluate_candidate(
            mdp=mdp, schedule=schedule, contest_state=contest_state,
        )
        flip = int(metrics["start_state_flip"])
        flip_details.append(
            f"  L={L}, C={C}, gamma={gamma}, p_warn={p_warn}, "
            f"warning_lead={warning_lead}, variant={variant}: flip={flip}"
        )
        if flip == 1:
            any_flip = True
    assert any_flip, (
        "Family D failed to produce start_state_flip on a 4-point grid:\n"
        + "\n".join(flip_details)
    )


# ---------------------------------------------------------------------------
# 5. MDP well-formedness invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_mdp_well_formed(variant: str) -> None:
    """P / R shapes, row sums, gamma, horizon, initial_state all valid."""
    L = 8
    warning_lead = 2
    gamma = 0.95
    C = 10.0
    p_warn = 0.3
    r_shallow = 0.1 * C if variant == "shallow_early_warning" else 0.0
    psi = _base_psi(
        variant=variant, L=L, C=C, gamma=gamma, p_warn=p_warn,
        warning_lead=warning_lead, r_shallow=r_shallow,
    )
    lam = lam_tie_closed_form(
        p_warn=p_warn, p_cat_given_warn=1.0, gamma=gamma,
        L=L, C=C, r_shallow=r_shallow, warning_lead=warning_lead,
    )
    mdp = build_family_d_mdp(lam, psi)

    P = np.asarray(mdp.p, dtype=np.float64)
    R = np.asarray(mdp.r, dtype=np.float64)
    S, A = P.shape[0], P.shape[1]

    assert P.ndim == 3, f"P.ndim={P.ndim} (expected 3)"
    assert P.shape == (S, A, S), f"P.shape={P.shape}"
    assert R.shape == (S, A, S), f"R.shape={R.shape}"
    # All rows sum to exactly 1.0 (absorbing states self-loop).
    row_sums = P.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-8), (
        f"row-sum deviation up to {float(np.max(np.abs(row_sums - 1.0))):.3e}"
    )
    # Absorbing terminals: zero-reward self-loops.
    s_cat = int(mdp.s_catastrophe_idx)
    s_safe = int(mdp.s_safe_end_idx)
    assert abs(P[s_cat, 0, s_cat] - 1.0) <= 1e-12
    assert abs(P[s_safe, 0, s_safe] - 1.0) <= 1e-12
    assert R[s_cat, 0, s_cat] == 0.0
    assert R[s_safe, 0, s_safe] == 0.0
    # Phase V contract.
    assert float(mdp.info.gamma) == gamma
    assert int(mdp.info.horizon) == L
    assert int(mdp.initial_state) == 0
    assert A == 2
    # State-count identity from the module docstring: S == 2L + k + 1.
    assert S == 2 * L + warning_lead + 1, (
        f"expected S == 2L + k + 1 = {2 * L + warning_lead + 1}; got S={S}"
    )


# ---------------------------------------------------------------------------
# Bonus: FamilySpec integrity
# ---------------------------------------------------------------------------

def test_family_d_spec_integrity() -> None:
    """family_d is a well-formed FamilySpec with Family D conventions."""
    assert family_d.name == "preventive_intervention"
    assert isinstance(family_d.contest_state, ContestState)
    assert family_d.contest_state.t == 0
    assert family_d.contest_state.s == 0
    assert family_d.contest_state.action_pair == (0, 1)
    psi = _base_psi(L=6, C=10.0)
    want = lam_tie_closed_form(
        p_warn=0.3, p_cat_given_warn=1.0, gamma=0.95, L=6, C=10.0,
        r_shallow=0.0, warning_lead=2,
    )
    got = family_d.warm_start_lambda(psi)
    assert abs(got - want) <= 1e-12
    lo, hi = family_d.scan_bracket(psi)
    assert lo == 0.0 and hi == 20.0
    # VARIANT_NAMES exhaustive.
    assert set(VARIANT_NAMES) == {"single_warning", "shallow_early_warning"}
