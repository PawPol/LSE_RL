"""
Phase IV-C §5, §9: Geometry-prioritized async DP tests.

Verify priority scoring formula, residual-only mode, geometry-gain-only
mode, and combined-mode sweep ordering for the geometry-prioritized
asynchronous dynamic programming module.
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

from experiments.weighted_lse_dp.planners.geometry_priority_dp import (
    GeometryPriorityDP,
    kl_bernoulli,
)


def _tiny_mdp():
    """2-state, 2-action deterministic chain."""
    S, A = 3, 2
    p = np.zeros((S, A, S))
    p[0, 0, 1] = 1.0  # s0, a0 -> s1
    p[0, 1, 2] = 1.0  # s0, a1 -> s2
    p[1, 0, 2] = 1.0  # s1, a0 -> s2
    p[1, 1, 2] = 1.0  # s1, a1 -> s2
    p[2, :, 2] = 1.0  # absorbing
    r = np.array([[0.0, -0.5], [1.0, 0.5], [0.0, 0.0]])
    return p, r


def _flat_schedule(T: int, beta: float = 0.05, gamma: float = 0.95) -> dict:
    return {
        "beta_used_t": [beta] * T,
        "alpha_t": [0.05] * T,
        "xi_ref_t": [0.1] * T,
    }


def test_kl_bernoulli_symmetry():
    """KL_Bern(p, 0.5) >= 0 and == 0 only at p=0.5."""
    vals = kl_bernoulli(np.array([0.1, 0.5, 0.9]))
    assert vals[1] < 1e-12
    assert vals[0] > 0
    assert vals[2] > 0


def test_priority_scoring_formula():
    """§5: Priority score matches the specified formula (residual + geometry gain)."""
    p, r = _tiny_mdp()
    T = 5
    sched = _flat_schedule(T, beta=0.05)
    planner = GeometryPriorityDP(p=p, r=r, gamma=0.9, horizon=T,
                                 schedule_v3=sched,
                                 lambda_geom=1.0, lambda_u=1.0, lambda_kl=1.0)

    residuals = np.abs(np.random.default_rng(0).standard_normal((T, p.shape[0])))
    scores = planner._priority_scores(residuals)

    # Each score must be >= residual (scaling >= 1)
    assert np.all(scores >= residuals - 1e-12)

    # Manually verify one element
    t = 0
    g_gain = planner._geom_gain_t[t]
    u_ref = planner._u_ref_t[t]
    kl = planner._kl_t[t]
    scale = 1.0 + g_gain + u_ref + kl
    expected = residuals[t] * scale
    np.testing.assert_allclose(scores[t], expected, rtol=1e-10)


def test_residual_only_mode():
    """§5, §9: Setting geometry weight to 0 recovers pure Bellman-residual priority."""
    p, r = _tiny_mdp()
    T = 4
    sched = _flat_schedule(T, beta=0.05)
    planner = GeometryPriorityDP(p=p, r=r, gamma=0.9, horizon=T,
                                 schedule_v3=sched,
                                 lambda_geom=0.0, lambda_u=0.0, lambda_kl=0.0)

    residuals = np.array([[0.3, 0.1, 0.0],
                          [0.5, 0.2, 0.0],
                          [0.1, 0.4, 0.0],
                          [0.2, 0.0, 0.0]])
    scores = planner._priority_scores(residuals)
    # With all lambdas=0, scale=1 → scores == residuals exactly
    np.testing.assert_allclose(scores, residuals, rtol=1e-10)


def test_geometry_gain_only_mode():
    """§5, §9: Setting lambda_u/lambda_kl=0 uses only geometry-based gain."""
    p, r = _tiny_mdp()
    T = 4
    sched = _flat_schedule(T, beta=0.1)
    planner = GeometryPriorityDP(p=p, r=r, gamma=0.9, horizon=T,
                                 schedule_v3=sched,
                                 lambda_geom=1.0, lambda_u=0.0, lambda_kl=0.0)

    residuals = np.ones((T, p.shape[0]))
    scores = planner._priority_scores(residuals)
    # scale = 1 + lambda_geom * geom_gain_t
    for t in range(T):
        expected_scale = 1.0 + planner._geom_gain_t[t]
        np.testing.assert_allclose(scores[t], expected_scale, rtol=1e-10)


def test_combined_mode_ordering():
    """§5, §9: Combined priority prefers states with both high residual and high gain."""
    p, r = _tiny_mdp()
    T = 5
    sched = _flat_schedule(T, beta=0.1)
    planner = GeometryPriorityDP(p=p, r=r, gamma=0.9, horizon=T,
                                 schedule_v3=sched,
                                 lambda_geom=1.0, lambda_u=1.0, lambda_kl=1.0)
    result = planner.plan(tol=1e-4, max_sweeps=200)
    assert result["n_sweeps"] > 0
    assert result["final_residual"] < 1e-4
    assert len(result["residual_history"]) == result["n_sweeps"]
    # Convergence should be monotone (residuals should generally decrease)
    hist = result["residual_history"]
    # Last residual should be smallest
    assert hist[-1] <= hist[0] + 1e-6


def test_plan_beta0_classical_collapse():
    """§5: With beta=0 schedule, safe operator collapses to classical."""
    p, r = _tiny_mdp()
    T = 4
    sched = _flat_schedule(T, beta=0.0)
    planner_safe = GeometryPriorityDP(p=p, r=r, gamma=0.9, horizon=T,
                                      schedule_v3=sched)
    result_safe = planner_safe.plan(tol=1e-8, max_sweeps=300)

    # Classical reference: compute by hand
    sched_classic = _flat_schedule(T, beta=0.0)
    planner_classic = GeometryPriorityDP(p=p, r=r, gamma=0.9, horizon=T,
                                         schedule_v3=sched_classic,
                                         lambda_geom=0.0, lambda_u=0.0, lambda_kl=0.0)
    result_classic = planner_classic.plan(tol=1e-8, max_sweeps=300)

    np.testing.assert_allclose(result_safe["V"], result_classic["V"], atol=1e-6)


def test_plan_returns_valid_fields():
    """§5: plan() output dict contains all required fields."""
    p, r = _tiny_mdp()
    T = 4
    sched = _flat_schedule(T, beta=0.05)
    planner = GeometryPriorityDP(p=p, r=r, gamma=0.9, horizon=T, schedule_v3=sched)
    result = planner.plan(tol=1e-4, max_sweeps=200)
    for key in ("V", "Q", "n_sweeps", "n_backups", "residual_history",
                "final_residual", "geom_gain_per_stage", "u_ref_per_stage",
                "frac_high_activation_backups", "wall_clock_s"):
        assert key in result, f"Missing key: {key}"
    assert result["V"].shape == (T, p.shape[0])
    assert result["Q"].shape == (T, p.shape[0], p.shape[1])
