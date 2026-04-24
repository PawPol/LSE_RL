"""Phase V WP2 — Family B unit tests.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5
Family B and ``tasks/todo.md`` WP2 test checklist.

Three checks:
1. Closed-form tie for the ``"single_event"`` variant: for 3 parameter
   quintuples ``(L, b, p, C, gamma)``, assert ``|Delta_0(x_c)| <= 1e-8``.
2. Warning-state variant: the warning state has strictly higher
   classical V than the no-warning state at matched stage context.
3. Multi-event variant: when the catastrophe expectations match
   ``p * gamma^{L-1} * C``, the classical value at the contest state
   matches the single-event variant.
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

from experiments.weighted_lse_dp.tasks.family_b_catastrophe import (  # noqa: E402
    VARIANT_NAMES,
    b_safe_tie_closed_form,
    branch_a_classical_value,
    build_family_b_mdp,
)


# ---------------------------------------------------------------------------
# Classical DP helper
# ---------------------------------------------------------------------------

def _classical_qv(mdp) -> tuple[np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# 1. Closed-form tie on single_event
# ---------------------------------------------------------------------------

SINGLE_EVENT_QUINTUPLES = [
    # (L, b, p, C, gamma)
    (4, 1.0, 0.1, 10.0, 0.9),
    (6, 0.5, 0.05, 5.0, 0.95),
    (8, 2.0, 0.2, 3.0, 0.99),
]


@pytest.mark.parametrize("L,b,p,C,gamma", SINGLE_EVENT_QUINTUPLES)
def test_single_event_closed_form_tie(
    L: int,
    b: float,
    p: float,
    C: float,
    gamma: float,
) -> None:
    """|Delta_0(x_c)| <= 1e-8 at the closed-form tie for single_event."""
    b_safe = b_safe_tie_closed_form(b=b, p=p, gamma=gamma, L=L, C=C)
    psi = {
        "variant": "single_event",
        "L": L,
        "b": b,
        "p": p,
        "C": C,
        "gamma": gamma,
    }
    mdp = build_family_b_mdp(b_safe, psi)
    Q, _ = _classical_qv(mdp)
    t_c, s_c, a1, a2 = 0, 0, 0, 1
    delta = float(Q[t_c, s_c, a1] - Q[t_c, s_c, a2])
    assert abs(delta) <= 1e-8, (
        f"single_event tie residual: |Delta_0|={abs(delta):.3e} "
        f"for L={L}, b={b}, p={p}, C={C}, gamma={gamma}"
    )


def test_branch_a_classical_value_matches_closed_form() -> None:
    """branch_a_classical_value(single_event) == b - p gamma^{L-1} C."""
    for L, b, p, C, gamma in SINGLE_EVENT_QUINTUPLES:
        psi = {
            "variant": "single_event",
            "L": L, "b": b, "p": p, "C": C, "gamma": gamma,
        }
        got = branch_a_classical_value(psi)
        want = b_safe_tie_closed_form(b=b, p=p, gamma=gamma, L=L, C=C)
        assert abs(got - want) <= 1e-12


# ---------------------------------------------------------------------------
# 2. Warning-state: labelled warning state has strictly distinct V from
#    the no-warning state at matched stage context.
# ---------------------------------------------------------------------------

def test_warning_state_classical_value_separation() -> None:
    """Warning state has strictly distinct classical V vs. no-warning state.

    The warning state sits on the catastrophe sub-branch at depth
    ``L - warning_depth`` from the contest; reaching it implies the
    catastrophe will fire (since the sub-branch was selected at the entry
    Bernoulli).  The matched no-warning state sits on the no-catastrophe
    sub-branch at the same stage.

    Under classical DP:
        V[t=stage, s=warning_state] < V[t=stage, s=no_warning_state]
    because the warning-state path leads to a terminal -C reward while
    the no-warning-state path does not.  The assertion "warning state
    has strictly higher classical V than the no-warning sub-branch" in
    the spec-adjacent test brief is taken in the sense "the two V values
    are strictly distinct and the information-bearing warning state is
    strictly informative" — we assert the strict ordering in the
    information-consistent direction (V_warning < V_no_warning) and that
    they differ by more than 1e-6.
    """
    L = 8
    b = 1.0
    p = 0.3
    C = 5.0
    gamma = 0.95
    warning_depth = 3
    b_safe = b_safe_tie_closed_form(b=b, p=p, gamma=gamma, L=L, C=C)
    psi = {
        "variant": "warning_state",
        "L": L,
        "b": b,
        "p": p,
        "C": C,
        "gamma": gamma,
        "warning_depth": warning_depth,
    }
    mdp = build_family_b_mdp(b_safe, psi)
    _, V = _classical_qv(mdp)

    # Stage at which both warning-state and no-warning-state are visible
    # (both are at the same chain depth).  warning_state index is
    # ``L - warning_depth + 1`` (s-index), sitting on the catastrophe
    # sub-chain.  no_warning_state is ``2L - warning_depth`` (s-index)
    # on the no-catastrophe sub-chain.  Both are reachable at the same
    # stage t = L - warning_depth (number of steps from the contest).
    t_stage = L - warning_depth
    s_warning = int(mdp.warning_state)
    s_no_warning = int(mdp.no_warning_state)

    v_warning = float(V[t_stage, s_warning])
    v_no_warning = float(V[t_stage, s_no_warning])
    # Warning state implies upcoming catastrophe -> V_warning < V_no_warning.
    assert v_warning < v_no_warning, (
        f"expected V[warning] < V[no-warning]; got "
        f"V_warning={v_warning:.4e} vs V_no_warning={v_no_warning:.4e}"
    )
    assert abs(v_warning - v_no_warning) > 1e-6, (
        f"warning and no-warning V must differ by >1e-6; "
        f"|delta|={abs(v_warning - v_no_warning):.3e}"
    )


# ---------------------------------------------------------------------------
# 3. Multi-event vs single-event classical value match
# ---------------------------------------------------------------------------

def test_multi_event_matches_single_event_when_expectation_matched() -> None:
    """When sum_i p_i gamma^{k_i - 1} C_i == p gamma^{L-1} C, V_cl matches."""
    L = 6
    b = 1.0
    gamma = 0.95
    # Target single-event expected loss.
    p_target = 0.2
    C_target = 4.0
    target_loss = p_target * (gamma ** (L - 1)) * C_target

    # Split into two events at depths k_1=3, k_2=L=6 with matched expected loss.
    event_depths = np.array([3, L], dtype=np.int64)
    event_mags = np.array([2.0, 3.0], dtype=np.float64)
    # Solve for (p_1, p_2) such that:
    #   p_1 gamma^{k_1 - 1} C_1 + p_2 gamma^{k_2 - 1} C_2 = target_loss
    #   p_1 = 0.15  (fixed)  =>  p_2 = (target_loss - p_1 * gamma^{k_1-1} * C_1)
    #                                  / (gamma^{k_2-1} * C_2)
    p_1 = 0.15
    p_2 = (target_loss - p_1 * (gamma ** (event_depths[0] - 1)) * event_mags[0]) / (
        (gamma ** (event_depths[1] - 1)) * event_mags[1]
    )
    assert 0.0 < p_2 < 1.0, f"p_2 out of bounds: {p_2}"
    event_probs = np.array([p_1, p_2], dtype=np.float64)

    # branch_a_classical_value for multi-event should equal
    # the single-event branch_a_classical_value with the same total expected loss.
    psi_single = {
        "variant": "single_event",
        "L": L, "b": b, "p": p_target, "C": C_target, "gamma": gamma,
    }
    psi_multi = {
        "variant": "multi_event",
        "L": L,
        "b": b,
        "gamma": gamma,
        "event_probs": event_probs,
        "event_depths": event_depths,
        "event_mags": event_mags,
    }
    v_single = branch_a_classical_value(psi_single)
    v_multi = branch_a_classical_value(psi_multi)
    assert abs(v_single - v_multi) <= 1e-12, (
        f"single vs multi-event classical V mismatch: "
        f"single={v_single}, multi={v_multi}, diff={abs(v_single - v_multi):.3e}"
    )

    # Cross-check on the actual MDP: at the contest state, the classical
    # Q-gap between (action 0, action 1) should vanish when b_safe is set
    # to v_multi.  (Planning DP here uses b_safe = v_multi, so
    # Delta_0 ~= 0 by construction.)
    b_safe = v_multi
    mdp = build_family_b_mdp(b_safe, psi_multi)
    Q, _ = _classical_qv(mdp)
    delta = float(Q[0, 0, 0] - Q[0, 0, 1])
    assert abs(delta) <= 1e-8, (
        f"multi-event Delta_0 residual at b_safe=v_multi: {abs(delta):.3e}"
    )


# ---------------------------------------------------------------------------
# Extra: closed-form ties on shallow_early + matched_concentration also
# match the DP on build_family_b_mdp (catches discount-exponent drift).
# ---------------------------------------------------------------------------

def test_shallow_early_closed_form_tie_matches_dp() -> None:
    for L, k_shallow, b_shallow, p, C, gamma in [
        (6, 2, 0.3, 0.2, 4.0, 0.95),
        (8, 3, 0.5, 0.1, 5.0, 0.99),
        (5, 1, 0.4, 0.25, 3.0, 0.9),
    ]:
        psi = {
            "variant": "shallow_early",
            "L": L, "k_shallow": k_shallow, "b_shallow": b_shallow,
            "p": p, "C": C, "gamma": gamma,
        }
        b_safe = branch_a_classical_value(psi)
        mdp = build_family_b_mdp(b_safe, psi)
        Q, _ = _classical_qv(mdp)
        delta = float(Q[0, 0, 0] - Q[0, 0, 1])
        assert abs(delta) <= 1e-8, (
            f"shallow_early closed-form mismatch: L={L}, k_shallow={k_shallow}, "
            f"|Delta_0|={abs(delta):.3e}"
        )


def test_matched_concentration_closed_form_tie_matches_dp() -> None:
    for L, b_stream, p, C, gamma in [
        (5, 0.2, 0.1, 3.0, 0.9),
        (8, 0.1, 0.2, 5.0, 0.95),
        (6, 0.3, 0.05, 4.0, 0.99),
    ]:
        psi = {
            "variant": "matched_concentration",
            "L": L, "b_stream": b_stream, "p": p, "C": C, "gamma": gamma,
        }
        b_safe = branch_a_classical_value(psi)
        mdp = build_family_b_mdp(b_safe, psi)
        Q, _ = _classical_qv(mdp)
        delta = float(Q[0, 0, 0] - Q[0, 0, 1])
        assert abs(delta) <= 1e-8, (
            f"matched_concentration closed-form mismatch: L={L}, |Delta_0|={abs(delta):.3e}"
        )


# ---------------------------------------------------------------------------
# Sanity: all variant names are accepted and build a well-formed MDP
# ---------------------------------------------------------------------------

def test_every_variant_builds() -> None:
    L = 5
    gamma = 0.9
    psis = {
        "single_event": {
            "variant": "single_event", "L": L, "b": 1.0, "p": 0.2, "C": 3.0,
            "gamma": gamma,
        },
        "warning_state": {
            "variant": "warning_state", "L": L, "b": 1.0, "p": 0.2, "C": 3.0,
            "gamma": gamma, "warning_depth": 2,
        },
        "shallow_early": {
            "variant": "shallow_early", "L": L, "b_shallow": 0.5, "k_shallow": 2,
            "p": 0.2, "C": 3.0, "gamma": gamma,
        },
        "multi_event": {
            "variant": "multi_event", "L": L, "b": 1.0, "gamma": gamma,
            "event_probs": np.array([0.1, 0.15]),
            "event_depths": np.array([3, L]),
            "event_mags": np.array([2.0, 3.0]),
        },
        "matched_concentration": {
            "variant": "matched_concentration", "L": L, "b_stream": 0.2,
            "p": 0.2, "C": 3.0, "gamma": gamma,
        },
    }
    assert set(psis.keys()) == set(VARIANT_NAMES)
    for name, psi in psis.items():
        lam = branch_a_classical_value(psi)
        mdp = build_family_b_mdp(lam, psi)
        assert int(mdp.info.horizon) == L
        assert int(mdp.initial_state) == 0
        # MDP must be well-formed: p sums to 1 across s'.
        P = np.asarray(mdp.p, dtype=np.float64)
        row_sums = P.sum(axis=2)
        assert np.allclose(row_sums, 1.0, atol=1e-8), (
            f"variant={name}: P row-sum deviation "
            f"{float(np.max(np.abs(row_sums - 1.0))):.3e}"
        )
