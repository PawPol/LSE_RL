"""Phase V WP1a -- candidate metric tests.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 6
and planner-resolution addendum section 13.
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

from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    _evaluate_core,
    evaluate_candidate,
)
from experiments.weighted_lse_dp.search.family_spec import ContestState  # noqa: E402


class _ShimMDP:
    def __init__(self, p, r, gamma, T, s0=0):
        self.p = np.asarray(p, dtype=np.float64)
        self.r = np.asarray(r, dtype=np.float64)
        self.info = SimpleNamespace(gamma=float(gamma), horizon=int(T))
        self.initial_state = int(s0)


# ---------------------------------------------------------------------------
# Toy 1: classical == safe everywhere (zero schedule)
# ---------------------------------------------------------------------------

def _build_trivial_mdp() -> _ShimMDP:
    """3-state, 2-action deterministic chain with identical rewards per action.

    Identical-reward-per-action guarantees argmax agreement between safe
    and classical regardless of the operator -- useful as a null.
    """
    S, A = 3, 2
    P = np.zeros((S, A, S), dtype=np.float64)
    r = np.zeros((S, A, S), dtype=np.float64)
    P[0, :, 1] = 1.0
    P[1, :, 2] = 1.0
    P[2, :, 2] = 1.0
    r[0, :, 1] = 0.5      # symmetric across actions at s=0
    r[1, :, 2] = 0.25
    return _ShimMDP(P, r, gamma=0.9, T=3, s0=0)


def test_zero_schedule_agreement() -> None:
    mdp = _build_trivial_mdp()
    T = mdp.info.horizon
    schedule = {
        "beta_used_t": np.zeros(T, dtype=np.float64),
        "beta_cap_t": np.zeros(T, dtype=np.float64),
        "beta_raw_t": np.zeros(T, dtype=np.float64),
    }
    contest = ContestState(t=0, s=0, action_pair=(0, 1))
    out = evaluate_candidate(mdp, schedule, contest_state=contest)
    # With beta=0 everywhere, the safe operator collapses to classical.
    assert out["policy_disagreement"] == pytest.approx(0.0, abs=0.0)
    assert out["start_state_flip"] == 0
    assert abs(out["value_gap"]) <= 1e-12
    assert out["clip_fraction"] == 0.0
    assert out["clip_inactive_fraction"] == 1.0
    assert out["clip_saturation_fraction"] == 0.0
    assert out["margin_pos"] >= 0.0
    # No clipping when beta_raw == beta_used == 0 regardless of cap.
    assert out["raw_convergence_status"] == "not_evaluated"


# ---------------------------------------------------------------------------
# Toy 2: constructed Q arrays that flip argmax only at s0
# ---------------------------------------------------------------------------

def test_start_state_flip_injected_q() -> None:
    """Inject Q_cl / Q_safe that differ only in argmax at (t=0, s=s0).

    Expected:
      * start_state_flip == 1
      * policy_disagreement == d_ref[0, s0] / sum(d_ref) == 1/T
        (when the chain is fully deterministic with point-mass mu_0
        and Q-arrays agree on all other (t, s), the disagreement mass
        is exactly 1 at (t=0, s=s0) normalized by T stages).
    """
    # Deterministic 3-state chain, 2 actions, T=4.
    S, A, T = 3, 2, 4
    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A, S), dtype=np.float64)
    P[0, :, 1] = 1.0
    P[1, :, 2] = 1.0
    P[2, :, 2] = 1.0
    gamma = 0.9

    # Construct Q_cl: at (t=0, s=0) prefer action 0 strictly; elsewhere tied.
    Q_cl = np.zeros((T, S, A), dtype=np.float64)
    Q_cl[0, 0, 0] = 1.0
    Q_cl[0, 0, 1] = 0.0
    # All other entries = 0  (ties -> argmax picks action 0).

    # Construct Q_safe: flip argmax at (t=0, s=0) -> prefer action 1.
    Q_safe = Q_cl.copy()
    Q_safe[0, 0, 0] = 0.0
    Q_safe[0, 0, 1] = 1.0
    # V arrays: just take max along action axis.
    V_cl = np.zeros((T + 1, S), dtype=np.float64)
    V_safe = np.zeros((T + 1, S), dtype=np.float64)
    V_cl[:T] = np.max(Q_cl, axis=2)
    V_safe[:T] = np.max(Q_safe, axis=2)

    schedule_beta = np.zeros(T, dtype=np.float64)
    mu_0 = np.zeros(S, dtype=np.float64)
    mu_0[0] = 1.0

    out = _evaluate_core(
        P=P, R=R, gamma=gamma, T=T, S=S, A=A,
        Q_cl=Q_cl, V_cl=V_cl, Q_safe=Q_safe, V_safe=V_safe,
        beta_used_t=schedule_beta,
        beta_cap_t=schedule_beta.copy(),
        beta_raw_t=schedule_beta.copy(),
        mu_0=mu_0, s0=0,
        contest_state=ContestState(t=0, s=0, action_pair=(0, 1)),
        reward_scale=1.0,
    )

    assert out["start_state_flip"] == 1

    # On this deterministic chain with mu_0 at s0, the two policies lead
    # to the same successor states (same chain), so d_cl and d_safe
    # agree on stages t >= 1.  The disagreement mask is 1 only at
    # (t=0, s=0) where d_ref = 1.0, yielding disagreement mass = 1
    # over total d_ref mass = T.
    expected = 1.0 / T
    assert out["policy_disagreement"] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Dimension / schema test
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "margin_pos",
    "margin_pos_norm",
    "delta_d",
    "mass_delta_d",
    "policy_disagreement",
    "start_state_flip",
    "value_gap",
    "value_gap_norm",
    "contest_gap_abs",
    "contest_gap_norm",
    "contest_occupancy_ref",
    "clip_fraction",
    "clip_inactive_fraction",
    "clip_saturation_fraction",
    "raw_local_deriv_stats",
    "raw_convergence_status",
}


def test_schema_completeness_and_finite() -> None:
    mdp = _build_trivial_mdp()
    T = mdp.info.horizon
    schedule = {
        "beta_used_t": np.zeros(T, dtype=np.float64),
        "beta_cap_t": np.full(T, 0.5, dtype=np.float64),
        "beta_raw_t": np.zeros(T, dtype=np.float64),
    }
    contest = ContestState(t=0, s=0, action_pair=(0, 1))
    out = evaluate_candidate(mdp, schedule, contest_state=contest)
    # All required keys present.
    assert set(out.keys()) >= REQUIRED_KEYS
    # Every numeric value is finite (except the status sentinel).
    for k, v in out.items():
        if k == "raw_convergence_status":
            assert isinstance(v, str)
            continue
        if k == "raw_local_deriv_stats":
            assert set(v.keys()) == {"mean", "p50", "p90", "max"}
            for vv in v.values():
                assert isinstance(vv, float) and np.isfinite(vv)
            continue
        if k == "start_state_flip":
            assert v in (0, 1)
            continue
        assert isinstance(v, float), f"{k} is not float: {type(v)}"
        assert np.isfinite(v), f"{k} is not finite: {v}"


# ---------------------------------------------------------------------------
# Clip metric sanity: when beta_raw > beta_cap on every stage, clip_fraction=1
# ---------------------------------------------------------------------------

def test_clip_fraction_all_stages_clipped() -> None:
    mdp = _build_trivial_mdp()
    T = mdp.info.horizon
    beta_cap = np.full(T, 0.1, dtype=np.float64)
    beta_raw = np.full(T, 1.0, dtype=np.float64)       # hugely over-cap
    beta_used = np.clip(beta_raw, -beta_cap, beta_cap)  # = 0.1 everywhere
    schedule = {
        "beta_used_t": beta_used,
        "beta_cap_t": beta_cap,
        "beta_raw_t": beta_raw,
    }
    contest = ContestState(t=0, s=0, action_pair=(0, 1))
    out = evaluate_candidate(mdp, schedule, contest_state=contest)
    assert out["clip_fraction"] == pytest.approx(1.0, abs=1e-9)
    assert out["clip_inactive_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert out["clip_saturation_fraction"] == pytest.approx(1.0, abs=1e-9)
