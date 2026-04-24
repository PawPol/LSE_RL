"""Phase V WP1a -- reference-occupancy helper tests.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 3.
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

from experiments.weighted_lse_dp.search.reference_occupancy import (  # noqa: E402
    compute_d_ref,
    save_occupancy,
)


class _ShimMDP:
    def __init__(self, p, r, gamma, T, s0=0):
        self.p = np.asarray(p, dtype=np.float64)
        self.r = np.asarray(r, dtype=np.float64)
        self.info = SimpleNamespace(gamma=float(gamma), horizon=int(T))
        self.initial_state = int(s0)


def _build_three_state_chain() -> _ShimMDP:
    """Deterministic 3-state chain with 2 actions that both advance.

    Transitions: s=0 -> s=1 -> s=2 -> s=2 (absorbing), both actions.
    """
    S = 3
    A = 2
    P = np.zeros((S, A, S), dtype=np.float64)
    r = np.zeros((S, A, S), dtype=np.float64)
    P[0, :, 1] = 1.0
    P[1, :, 2] = 1.0
    P[2, :, 2] = 1.0
    return _ShimMDP(P, r, gamma=0.9, T=5, s0=0)


def test_per_stage_mass_conservation() -> None:
    mdp = _build_three_state_chain()
    T = mdp.info.horizon
    S = mdp.p.shape[0]
    pi_a = np.zeros((T, S), dtype=np.int64)            # always action 0
    pi_b = np.ones((T, S), dtype=np.int64)             # always action 1
    out = compute_d_ref(mdp, pi_a, pi_b)
    for t in range(T):
        np.testing.assert_allclose(out["d_cl"][t].sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(out["d_safe"][t].sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(out["d_ref"][t].sum(), 1.0, atol=1e-10)


def test_d_ref_is_pointwise_average() -> None:
    mdp = _build_three_state_chain()
    T = mdp.info.horizon
    S = mdp.p.shape[0]
    pi_a = np.zeros((T, S), dtype=np.int64)
    pi_b = np.ones((T, S), dtype=np.int64)
    out = compute_d_ref(mdp, pi_a, pi_b)
    expected = 0.5 * out["d_cl"] + 0.5 * out["d_safe"]
    np.testing.assert_allclose(out["d_ref"], expected, atol=1e-12)


def test_roundtrip_save_and_load(tmp_path: Path) -> None:
    mdp = _build_three_state_chain()
    T = mdp.info.horizon
    S = mdp.p.shape[0]
    pi_a = np.zeros((T, S), dtype=np.int64)
    pi_b = np.ones((T, S), dtype=np.int64)
    out = compute_d_ref(mdp, pi_a, pi_b)
    path = tmp_path / "occupancy.npz"
    save_occupancy(path, out)
    loaded = np.load(path)
    np.testing.assert_allclose(loaded["d_cl"], out["d_cl"])
    np.testing.assert_allclose(loaded["d_safe"], out["d_safe"])
    np.testing.assert_allclose(loaded["d_ref"], out["d_ref"])
    assert int(loaded["horizon"]) == T
    np.testing.assert_allclose(float(loaded["gamma"]), out["gamma"])
    np.testing.assert_array_equal(loaded["time_augmented_shape"], [T, S])


def test_stochastic_policy_accepted() -> None:
    """A uniform stochastic policy sums to 1 per row and is accepted."""
    mdp = _build_three_state_chain()
    T = mdp.info.horizon
    S = mdp.p.shape[0]
    A = mdp.p.shape[1]
    pi_uniform = np.full((T, S, A), 1.0 / A, dtype=np.float64)
    out = compute_d_ref(mdp, pi_uniform, pi_uniform)
    for t in range(T):
        np.testing.assert_allclose(out["d_ref"][t].sum(), 1.0, atol=1e-10)


def test_mu_0_array_accepted() -> None:
    """Explicit initial distribution overrides the point-mass default."""
    mdp = _build_three_state_chain()
    T = mdp.info.horizon
    S = mdp.p.shape[0]
    pi = np.zeros((T, S), dtype=np.int64)
    mu0 = np.array([0.2, 0.8, 0.0], dtype=np.float64)
    out = compute_d_ref(mdp, pi, pi, mu_0=mu0)
    np.testing.assert_allclose(out["d_cl"][0], mu0, atol=1e-12)
    # After one step from the deterministic chain, 0.2 mass stays at s=1 (from s=0)
    # and 0.8 mass goes to s=2 (from s=1).
    np.testing.assert_allclose(
        out["d_cl"][1], np.array([0.0, 0.2, 0.8]), atol=1e-12
    )


def test_mu_0_must_sum_to_one() -> None:
    mdp = _build_three_state_chain()
    T = mdp.info.horizon
    S = mdp.p.shape[0]
    pi = np.zeros((T, S), dtype=np.int64)
    with pytest.raises(ValueError, match="must sum to 1"):
        compute_d_ref(mdp, pi, pi, mu_0=np.array([0.5, 0.3, 0.1]))
