"""
Integration tests for safe weighted-LSE DP planners on a tiny FiniteMDP.

Tasks 20+21 of the Phase III Group E integration work.

These tests verify that all five safe DP planners:
  - Accept a non-trivial beta schedule and run to completion.
  - Produce output arrays with the correct shapes.
  - Populate instrumentation fields (clipping diagnostics).
  - Track V_sweep_history across multiple sweeps (Task 21).

The tests use a 3-state chain MDP with horizon T=5 and small but non-zero
betas so that the safe operator diverges from classical behavior.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------

def _find_repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mushroom-rl-dev").is_dir():
            return parent
    return here.parents[2]


_REPO_ROOT = _find_repo_root()
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
_SRC = _REPO_ROOT / "src"

for _p in (_REPO_ROOT, _MUSHROOM_DEV, _SRC):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import numpy as np
import pytest

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    build_certification,
)
from mushroom_rl.algorithms.value.dp import ClassicalValueIteration
from mushroom_rl.algorithms.value.dp.safe_weighted_value_iteration import (
    SafeWeightedValueIteration,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_policy_evaluation import (
    SafeWeightedPolicyEvaluation,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_policy_iteration import (
    SafeWeightedPolicyIteration,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_modified_policy_iteration import (
    SafeWeightedModifiedPolicyIteration,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_async_value_iteration import (
    SafeWeightedAsyncValueIteration,
)


# =========================================================================
# Constants
# =========================================================================

_GAMMA = 0.99
_HORIZON = 5
_S = 3
_A = 2
_R_MAX = 1.0


# =========================================================================
# Helpers
# =========================================================================

def _make_chain_arrays():
    """Build raw P, R arrays for the 3-state deterministic chain.

    States: {0, 1, 2}, Actions: {0=right, 1=stay}
    Transitions:
        P[s, 0, min(s+1, 2)] = 1.0   (right: advance, clamp at 2)
        P[s, 1, s]            = 1.0   (stay)
    Rewards:
        R[1, 0, 2] = 1.0              (reward for entering state 2 from 1)
    """
    P = np.zeros((_S, _A, _S), dtype=np.float64)
    R = np.zeros((_S, _A, _S), dtype=np.float64)
    for s in range(_S):
        P[s, 0, min(s + 1, _S - 1)] = 1.0  # right
        P[s, 1, s] = 1.0                     # stay
    R[1, 0, 2] = 1.0
    return P, R


def _make_chain_mdp(gamma: float = _GAMMA, horizon: int = _HORIZON) -> FiniteMDP:
    """Create the FiniteMDP for the 3-state chain."""
    P, R = _make_chain_arrays()
    return FiniteMDP(P, R, mu=None, gamma=gamma, horizon=horizon)


def _make_direct_schedule(
    T: int = _HORIZON,
    gamma: float = _GAMMA,
    beta_used_t: "list[float] | None" = None,
) -> BetaSchedule:
    """Build a BetaSchedule (safe_weighted_common) with explicit betas.

    Uses small non-zero betas by default. beta_cap_t is set large
    enough (10.0) so no clipping occurs.
    """
    if beta_used_t is None:
        beta_used_t = [0.5, 0.4, 0.3, 0.2, 0.1]

    assert len(beta_used_t) == T, (
        f"beta_used_t length {len(beta_used_t)} != T={T}"
    )

    # Compute certification quantities from a small alpha schedule.
    alpha_t = np.full(T, 0.05, dtype=np.float64)
    cert = build_certification(alpha_t, R_max=_R_MAX, gamma=gamma)

    # Override beta_cap to be large so betas are not clipped.
    beta_cap_t = np.full(T, 10.0, dtype=np.float64)

    schedule_dict: Dict[str, Any] = {
        "gamma": gamma,
        "sign": -1,
        "task_family": "test_chain",
        "reward_bound": _R_MAX,
        "beta_raw_t": list(beta_used_t),
        "beta_cap_t": beta_cap_t.tolist(),
        "beta_used_t": list(beta_used_t),
        "alpha_t": alpha_t.tolist(),
        "kappa_t": cert["kappa_t"].tolist(),
        "Bhat_t": cert["Bhat_t"].tolist(),
        "clip_active_t": [False] * T,
        "informativeness_t": [0.0] * T,
        "d_target_t": [gamma] * T,
        "calibration_source_path": "",
        "calibration_hash": "",
    }
    return BetaSchedule(schedule_dict)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def chain_mdp():
    return _make_chain_mdp()


@pytest.fixture
def schedule():
    return _make_direct_schedule()


@pytest.fixture
def all_zero_policy():
    """Deterministic policy: always take action 0 at every (t, s)."""
    return np.zeros((_HORIZON, _S), dtype=np.int64)


# =========================================================================
# Task 20: TestSafeDPIntegration
# =========================================================================

class TestSafeDPIntegration:
    """Integration tests for all five safe DP planners on a tiny MDP."""

    def test_safe_vi_shapes(self, chain_mdp, schedule):
        """SafeWeightedVI produces Q, V, pi with correct shapes."""
        vi = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=1
        ).run()
        assert vi.Q.shape == (_HORIZON, _S, _A), (
            f"Q.shape = {vi.Q.shape}, expected ({_HORIZON}, {_S}, {_A})"
        )
        assert vi.V.shape == (_HORIZON + 1, _S), (
            f"V.shape = {vi.V.shape}, expected ({_HORIZON + 1}, {_S})"
        )
        assert vi.pi.shape == (_HORIZON, _S), (
            f"pi.shape = {vi.pi.shape}, expected ({_HORIZON}, {_S})"
        )
        # Terminal row must be zero.
        np.testing.assert_array_equal(
            vi.V[_HORIZON], np.zeros(_S),
            err_msg="V[T, :] must be zero (terminal)."
        )

    def test_safe_vi_clipping_activity(self, chain_mdp, schedule):
        """With non-trivial beta, last_clip_active is populated."""
        vi = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=1
        ).run()
        # The clipping_summary dict must be populated.
        cs = vi.clipping_summary
        assert "stage_clip_active" in cs, (
            "clipping_summary must contain 'stage_clip_active'"
        )
        assert len(cs["stage_clip_active"]) == _HORIZON, (
            f"stage_clip_active length = {len(cs['stage_clip_active'])}, "
            f"expected {_HORIZON}"
        )
        # With large caps, no clipping should have occurred.
        assert cs["n_stages_clipped"] == 0, (
            f"Expected 0 clipped stages with large caps, got "
            f"{cs['n_stages_clipped']}"
        )

    def test_safe_pe_shapes(self, chain_mdp, schedule, all_zero_policy):
        """SafeWeightedPE produces Q, V with correct shapes."""
        pe = SafeWeightedPolicyEvaluation(
            chain_mdp, pi=all_zero_policy, schedule=schedule
        ).run()
        assert pe.Q.shape == (_HORIZON, _S, _A), (
            f"Q.shape = {pe.Q.shape}, expected ({_HORIZON}, {_S}, {_A})"
        )
        assert pe.V.shape == (_HORIZON + 1, _S), (
            f"V.shape = {pe.V.shape}, expected ({_HORIZON + 1}, {_S})"
        )
        np.testing.assert_array_equal(
            pe.V[_HORIZON], np.zeros(_S),
            err_msg="V[T, :] must be zero (terminal)."
        )

    def test_safe_pi_shapes(self, chain_mdp, schedule):
        """SafeWeightedPI produces Q, V, pi with correct shapes."""
        pi_alg = SafeWeightedPolicyIteration(
            chain_mdp, schedule=schedule, max_iter=10
        ).run()
        assert pi_alg.Q.shape == (_HORIZON, _S, _A)
        assert pi_alg.V.shape == (_HORIZON + 1, _S)
        assert pi_alg.pi.shape == (_HORIZON, _S)
        np.testing.assert_array_equal(
            pi_alg.V[_HORIZON], np.zeros(_S),
            err_msg="V[T, :] must be zero (terminal)."
        )

    def test_safe_mpi_shapes(self, chain_mdp, schedule):
        """SafeWeightedMPI with m=3 produces correct shapes."""
        mpi = SafeWeightedModifiedPolicyIteration(
            chain_mdp, schedule=schedule, m=3, max_iter=10
        ).run()
        assert mpi.Q.shape == (_HORIZON, _S, _A)
        assert mpi.V.shape == (_HORIZON + 1, _S)
        assert mpi.pi.shape == (_HORIZON, _S)
        np.testing.assert_array_equal(
            mpi.V[_HORIZON], np.zeros(_S),
            err_msg="V[T, :] must be zero (terminal)."
        )

    def test_safe_async_vi_shapes(self, chain_mdp, schedule):
        """SafeWeightedAsyncVI produces correct shapes."""
        avi = SafeWeightedAsyncValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=1
        ).run()
        assert avi.Q.shape == (_HORIZON, _S, _A)
        assert avi.V.shape == (_HORIZON + 1, _S)
        assert avi.pi.shape == (_HORIZON, _S)
        np.testing.assert_array_equal(
            avi.V[_HORIZON], np.zeros(_S),
            err_msg="V[T, :] must be zero (terminal)."
        )


# =========================================================================
# Task 21: TestSafeVSweepHistory
# =========================================================================

class TestSafeVSweepHistory:
    """Tests for V_sweep_history tracking across multiple sweeps."""

    def test_v_sweep_history_populated_after_run(self, chain_mdp, schedule):
        """V_sweep_history has one entry per sweep, each with shape (T+1, S)."""
        n_sweeps = 3
        vi = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=n_sweeps
        ).run()
        assert len(vi.V_sweep_history) == n_sweeps, (
            f"V_sweep_history length = {len(vi.V_sweep_history)}, "
            f"expected {n_sweeps}"
        )
        for k, V_snap in enumerate(vi.V_sweep_history):
            assert V_snap.shape == (_HORIZON + 1, _S), (
                f"V_sweep_history[{k}].shape = {V_snap.shape}, "
                f"expected ({_HORIZON + 1}, {_S})"
            )

    def test_v_sweep_history_different_per_sweep(self, chain_mdp, schedule):
        """Consecutive V_sweep_history entries differ (at least for sweep 0 vs 1).

        The first sweep propagates value from the zero-initialised table;
        after that the values are already converged (finite-horizon DAG).
        So sweep 0 vs the zero-init should differ.
        """
        n_sweeps = 3
        vi = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=n_sweeps
        ).run()

        # The first sweep's V must differ from all-zeros (value was propagated).
        V_zeros = np.zeros((_HORIZON + 1, _S), dtype=np.float64)
        assert not np.array_equal(vi.V_sweep_history[0], V_zeros), (
            "First sweep V should differ from all-zeros."
        )

    def test_v_sweep_history_final_matches_v(self, chain_mdp, schedule):
        """The last entry in V_sweep_history matches the planner's V."""
        n_sweeps = 3
        vi = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=n_sweeps
        ).run()
        np.testing.assert_array_equal(
            vi.V_sweep_history[-1], vi.V,
            err_msg=(
                "V_sweep_history[-1] must match the final V after run()."
            ),
        )


# =========================================================================
# M1: schedule.T != mdp.horizon guard
# =========================================================================

class TestHorizonMismatchGuard:
    """All five safe DP planners must raise ValueError when schedule.T != horizon."""

    def _wrong_T_schedule(self) -> BetaSchedule:
        """Schedule with T=3, but the MDP has T=5."""
        return _make_direct_schedule(T=3, gamma=_GAMMA, beta_used_t=[0.5, 0.4, 0.3])

    def test_vi_rejects_wrong_T(self, chain_mdp):
        with pytest.raises(ValueError, match="schedule.T="):
            SafeWeightedValueIteration(chain_mdp, schedule=self._wrong_T_schedule())

    def test_pe_rejects_wrong_T(self, chain_mdp, all_zero_policy):
        # PE needs a policy with matching T -- use T=5 policy, but pass T=3 schedule.
        with pytest.raises(ValueError, match="schedule.T="):
            SafeWeightedPolicyEvaluation(
                chain_mdp, pi=all_zero_policy, schedule=self._wrong_T_schedule()
            )

    def test_pi_rejects_wrong_T(self, chain_mdp):
        with pytest.raises(ValueError, match="does not match"):
            SafeWeightedPolicyIteration(chain_mdp, schedule=self._wrong_T_schedule())

    def test_mpi_rejects_wrong_T(self, chain_mdp):
        with pytest.raises(ValueError, match="schedule.T="):
            SafeWeightedModifiedPolicyIteration(
                chain_mdp, schedule=self._wrong_T_schedule()
            )

    def test_async_vi_rejects_wrong_T(self, chain_mdp):
        with pytest.raises(ValueError, match="schedule.T="):
            SafeWeightedAsyncValueIteration(
                chain_mdp, schedule=self._wrong_T_schedule()
            )


# =========================================================================
# M2: schedule.gamma != mdp.gamma guard
# =========================================================================

class TestGammaMismatchGuard:
    """All safe planners must raise ValueError when schedule.gamma != mdp.gamma."""

    def _wrong_gamma_schedule(self) -> BetaSchedule:
        """Schedule with gamma=0.95, but the MDP has gamma=0.99."""
        return _make_direct_schedule(T=_HORIZON, gamma=0.95)

    def test_vi_rejects_wrong_gamma(self, chain_mdp):
        with pytest.raises(ValueError, match="schedule.gamma="):
            SafeWeightedValueIteration(
                chain_mdp, schedule=self._wrong_gamma_schedule()
            )

    def test_pe_rejects_wrong_gamma(self, chain_mdp, all_zero_policy):
        with pytest.raises(ValueError, match="schedule.gamma="):
            SafeWeightedPolicyEvaluation(
                chain_mdp, pi=all_zero_policy,
                schedule=self._wrong_gamma_schedule()
            )

    def test_pi_rejects_wrong_gamma(self, chain_mdp):
        with pytest.raises(ValueError, match="schedule.gamma="):
            SafeWeightedPolicyIteration(
                chain_mdp, schedule=self._wrong_gamma_schedule()
            )

    def test_mpi_rejects_wrong_gamma(self, chain_mdp):
        with pytest.raises(ValueError, match="schedule.gamma="):
            SafeWeightedModifiedPolicyIteration(
                chain_mdp, schedule=self._wrong_gamma_schedule()
            )

    def test_async_vi_rejects_wrong_gamma(self, chain_mdp):
        with pytest.raises(ValueError, match="schedule.gamma="):
            SafeWeightedAsyncValueIteration(
                chain_mdp, schedule=self._wrong_gamma_schedule()
            )


# =========================================================================
# M3: v_init warm-start preserved through run()
# =========================================================================

class TestVInitWarmStart:
    """Safe VI with v_init set to classical solution must work correctly."""

    def test_vi_warm_start_runs_without_error(self, chain_mdp, schedule):
        """Safe VI with v_init = classical V runs without error."""
        classical = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()
        vi = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=1, v_init=classical.V
        ).run()
        # Verify non-trivial output (not all zeros).
        assert np.max(np.abs(vi.V[0])) > 0.0, (
            "V[0] should be non-zero after warm-started safe VI."
        )

    def test_vi_warm_start_idempotent(self, chain_mdp, schedule):
        """Calling run() twice on a warm-started VI gives the same result."""
        classical = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()
        vi = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=1, v_init=classical.V
        )
        vi.run()
        V_first = vi.V.copy()
        vi.run()
        V_second = vi.V.copy()
        np.testing.assert_array_equal(
            V_first, V_second,
            err_msg="run() must be idempotent: warm-start restored on each call."
        )

    def test_vi_warm_start_v_init_not_zeroed(self, chain_mdp, schedule):
        """The v_init is not discarded by run() -- V before backward pass is non-zero.

        We verify this indirectly: with n_sweeps=2, the first sweep residual
        should differ from the cold-start residual because the initial V is
        non-zero.
        """
        # Cold start
        vi_cold = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=2
        ).run()

        # Warm start with classical solution
        classical = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()
        vi_warm = SafeWeightedValueIteration(
            chain_mdp, schedule=schedule, n_sweeps=2, v_init=classical.V
        ).run()

        # The first-sweep residuals should differ because the starting V
        # tables differ (zero vs classical).
        assert vi_cold.residuals[0] != vi_warm.residuals[0], (
            "First-sweep residuals should differ between cold and warm start, "
            "proving v_init is not discarded."
        )
