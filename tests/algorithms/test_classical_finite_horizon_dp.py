"""
Tests for the finite-horizon classical DP planners.

Spec reference: docs/specs/phase_I_*.md SS8.1

Each test class exercises one planner against a hand-built 3-state
deterministic chain MDP with a known backward-induction solution.
The tests enforce the following invariants:

1. VI == backward induction (bit-exact)
2. PE == backward induction for a fixed policy (bit-exact)
3. PI == VI (bit-exact V*)
4. MPI converges to the same fixed point as VI
5. Async-VI (sequential order) is bit-exact with VI; all orders converge
   to the same optimum.
"""
from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------------------
# sys.path inserts so the test can find mushroom-rl-dev and the experiments
# tree (needed by the planners' SweepTimer import).
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
_SRC = _REPO_ROOT / "src"

for _p in (_REPO_ROOT, _MUSHROOM_DEV, _SRC):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import numpy as np
import pytest

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.algorithms.value.dp import (
    ClassicalValueIteration,
    ClassicalPolicyEvaluation,
    ClassicalPolicyIteration,
    ClassicalModifiedPolicyIteration,
    ClassicalAsyncValueIteration,
)
from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    expected_reward,
)


# =========================================================================
# Fixtures / helpers
# =========================================================================

_GAMMA = 0.99
_HORIZON = 5
_S = 3
_A = 2


def _make_chain_arrays():
    """Build raw P, R arrays for the 3-state deterministic chain.

    States: {0, 1, 2}
    Actions: {0=right, 1=stay}
    Transitions:
        P[s, 0, min(s+1, 2)] = 1.0   (right: advance, clamp at 2)
        P[s, 1, s]            = 1.0   (stay)
    Rewards:
        R[1, 0, 2] = 1.0              (reward for entering state 2 from 1)
        All other R = 0.
    """
    P = np.zeros((_S, _A, _S), dtype=np.float64)
    R = np.zeros((_S, _A, _S), dtype=np.float64)
    for s in range(_S):
        P[s, 0, min(s + 1, _S - 1)] = 1.0  # right
        P[s, 1, s] = 1.0                     # stay
    R[1, 0, 2] = 1.0
    return P, R


def make_chain_mdp(gamma=_GAMMA, horizon=_HORIZON):
    """Create the FiniteMDP for the 3-state chain."""
    P, R = _make_chain_arrays()
    return FiniteMDP(P, R, mu=None, gamma=gamma, horizon=horizon)


def backward_induction_reference(P, R, gamma, horizon):
    """Hand-coded backward induction -- the ground-truth solver.

    Returns (Q, V, pi) with shapes:
        Q:  (T, S, A)  float64
        V:  (T+1, S)   float64   (V[T,:] = 0)
        pi: (T, S)     int64
    """
    S, A, _ = P.shape
    T = horizon
    r_bar = np.einsum("ijk,ijk->ij", P, R).astype(np.float64)  # (S, A)

    Q = np.zeros((T, S, A), dtype=np.float64)
    V = np.zeros((T + 1, S), dtype=np.float64)
    pi = np.zeros((T, S), dtype=np.int64)

    for t in range(T - 1, -1, -1):
        for s in range(S):
            for a in range(A):
                Q[t, s, a] = r_bar[s, a] + gamma * np.dot(P[s, a, :], V[t + 1, :])
            V[t, s] = Q[t, s, :].max()
            pi[t, s] = Q[t, s, :].argmax()
    return Q, V, pi


def backward_induction_pe_reference(P, R, gamma, horizon, ref_pi):
    """Hand-coded backward induction for policy evaluation under ref_pi.

    ref_pi: (T, S) int64 -- deterministic policy to evaluate.

    Returns (Q, V) with shapes:
        Q:  (T, S, A)  float64   -- full Q^pi for all actions
        V:  (T+1, S)   float64   (V[T,:] = 0)
    """
    S, A, _ = P.shape
    T = horizon
    r_bar = np.einsum("ijk,ijk->ij", P, R).astype(np.float64)

    Q = np.zeros((T, S, A), dtype=np.float64)
    V = np.zeros((T + 1, S), dtype=np.float64)

    for t in range(T - 1, -1, -1):
        # V^pi[t,s] via the policy action
        for s in range(S):
            a_pi = ref_pi[t, s]
            V[t, s] = r_bar[s, a_pi] + gamma * np.dot(P[s, a_pi, :], V[t + 1, :])
        # Full Q^pi[t,s,a] for all actions (using V^pi[t+1])
        for s in range(S):
            for a in range(A):
                Q[t, s, a] = r_bar[s, a] + gamma * np.dot(P[s, a, :], V[t + 1, :])
    return Q, V


@pytest.fixture
def chain_mdp():
    return make_chain_mdp()


@pytest.fixture
def chain_arrays():
    return _make_chain_arrays()


@pytest.fixture
def reference_solution(chain_arrays):
    """Ground-truth backward induction solution for the chain MDP."""
    P, R = chain_arrays
    Q, V, pi = backward_induction_reference(P, R, _GAMMA, _HORIZON)
    return Q, V, pi


# =========================================================================
# 1. Value Iteration
# =========================================================================


class TestValueIteration:
    """Test ClassicalValueIteration against hand-coded backward induction.

    Invariant guarded: VI produces the exact (bit-identical) backward-
    induction solution on a deterministic chain MDP.
    Spec line: SS8.1 item (1) -- VI == backward induction (bit-exact).
    """

    def test_vi_matches_backward_induction(self, chain_mdp, reference_solution):
        """VI Q, V, pi must be bit-exact with the reference backward induction.
        # docs/specs/phase_I_*.md SS8.1 item (1)
        """
        Q_ref, V_ref, pi_ref = reference_solution
        vi = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()

        np.testing.assert_equal(vi.Q, Q_ref)
        np.testing.assert_equal(vi.V, V_ref)
        np.testing.assert_equal(vi.pi, pi_ref)

    def test_vi_v_shape(self, chain_mdp):
        """V must have shape (T+1, S) with terminal row zero.
        # docs/specs/phase_I_*.md SS8.1
        """
        vi = ClassicalValueIteration(chain_mdp).run()
        assert vi.V.shape == (_HORIZON + 1, _S)
        np.testing.assert_equal(vi.V[_HORIZON, :], np.zeros(_S))

    def test_vi_q_shape(self, chain_mdp):
        """Q must have shape (T, S, A).
        # docs/specs/phase_I_*.md SS8.1
        """
        vi = ClassicalValueIteration(chain_mdp).run()
        assert vi.Q.shape == (_HORIZON, _S, _A)

    def test_vi_pi_shape_and_dtype(self, chain_mdp):
        """pi must have shape (T, S) and dtype int64.
        # docs/specs/phase_I_*.md SS8.1
        """
        vi = ClassicalValueIteration(chain_mdp).run()
        assert vi.pi.shape == (_HORIZON, _S)
        assert vi.pi.dtype == np.int64

    def test_vi_known_values(self, chain_mdp):
        """Spot-check known hand-computed values.
        # docs/specs/phase_I_*.md SS8.1 item (1)

        For the 3-state chain with gamma=0.99, horizon=5:
          V[4,:] = [0, 1, 0]   (last decision stage; only R_bar[1,0]=1)
          V[0,:] = [0.99, 1, 0]
          pi[t,:] = [0, 0, 0] for all t (always go right)
        """
        vi = ClassicalValueIteration(chain_mdp).run()

        np.testing.assert_equal(vi.V[_HORIZON, :], np.array([0.0, 0.0, 0.0]))
        np.testing.assert_equal(vi.V[4, :], np.array([0.0, 1.0, 0.0]))
        np.testing.assert_equal(vi.V[0, :], np.array([0.99, 1.0, 0.0]))

        # Optimal policy: always go right (action 0)
        for t in range(_HORIZON):
            np.testing.assert_equal(vi.pi[t, :], np.array([0, 0, 0]))

    def test_vi_converged_single_sweep(self, chain_mdp):
        """Single-sweep VI must report converged=True.
        # docs/specs/phase_I_*.md SS8.1
        """
        vi = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()
        assert vi.converged is True
        assert vi.n_sweeps == 1

    def test_vi_multi_sweep_second_residual_zero(self, chain_mdp, reference_solution):
        """Multi-sweep VI: second sweep residual must be 0 (exact on first pass).
        # docs/specs/phase_I_*.md SS8.1
        """
        Q_ref, V_ref, pi_ref = reference_solution
        vi = ClassicalValueIteration(chain_mdp, n_sweeps=3).run()

        # After the first sweep, subsequent sweeps should produce zero residual.
        assert len(vi.residuals) == 3
        np.testing.assert_allclose(vi.residuals[1], 0.0, atol=1e-15)
        np.testing.assert_allclose(vi.residuals[2], 0.0, atol=1e-15)

        # Result is still bit-exact with reference.
        np.testing.assert_equal(vi.Q, Q_ref)
        np.testing.assert_equal(vi.V, V_ref)


# =========================================================================
# 2. Policy Evaluation
# =========================================================================


class TestPolicyEvaluation:
    """Test ClassicalPolicyEvaluation against hand-coded PE.

    Invariant guarded: PE under a fixed policy produces the exact
    backward-induction policy-evaluation solution (bit-identical).
    Spec line: SS8.1 item (2) -- PE == backward induction for fixed policy.
    """

    def test_pe_matches_backward_induction_ref_policy(
        self, chain_mdp, chain_arrays, reference_solution
    ):
        """PE under the optimal policy must match hand-coded PE bit-exactly.
        # docs/specs/phase_I_*.md SS8.1 item (2)
        """
        P, R = chain_arrays
        _, _, pi_opt = reference_solution

        Q_pe_ref, V_pe_ref = backward_induction_pe_reference(
            P, R, _GAMMA, _HORIZON, pi_opt
        )

        pe = ClassicalPolicyEvaluation(chain_mdp, pi=pi_opt).run()
        np.testing.assert_equal(pe.V, V_pe_ref)
        np.testing.assert_equal(pe.Q, Q_pe_ref)

    def test_pe_optimal_policy_matches_vi_v(
        self, chain_mdp, reference_solution
    ):
        """PE under the optimal policy must recover V* (the optimal V).
        # docs/specs/phase_I_*.md SS8.1 item (2)

        When evaluating the optimal policy, V^pi = V* and Q^pi = Q*.
        """
        Q_ref, V_ref, pi_opt = reference_solution

        pe = ClassicalPolicyEvaluation(chain_mdp, pi=pi_opt).run()
        np.testing.assert_equal(pe.V, V_ref)
        np.testing.assert_equal(pe.Q, Q_ref)

    def test_pe_suboptimal_policy(self, chain_mdp, chain_arrays):
        """PE under the stay-everywhere policy yields zero value.
        # docs/specs/phase_I_*.md SS8.1 item (2)

        If the agent always stays (action 1), it never reaches state 2
        from state 1, so it collects zero reward from every initial state.
        """
        P, R = chain_arrays
        stay_policy = np.ones((_HORIZON, _S), dtype=np.int64)  # always stay

        Q_pe_ref, V_pe_ref = backward_induction_pe_reference(
            P, R, _GAMMA, _HORIZON, stay_policy
        )

        pe = ClassicalPolicyEvaluation(chain_mdp, pi=stay_policy).run()
        np.testing.assert_equal(pe.V, V_pe_ref)
        # Under the stay policy, V should be all zeros (no reward collected).
        np.testing.assert_equal(pe.V, np.zeros((_HORIZON + 1, _S)))

    def test_pe_shapes(self, chain_mdp, reference_solution):
        """PE output shapes must match the MDP dimensions.
        # docs/specs/phase_I_*.md SS8.1
        """
        _, _, pi_opt = reference_solution
        pe = ClassicalPolicyEvaluation(chain_mdp, pi=pi_opt).run()
        assert pe.Q.shape == (_HORIZON, _S, _A)
        assert pe.V.shape == (_HORIZON + 1, _S)
        assert pe.pi.shape == (_HORIZON, _S)
        assert pe.n_sweeps == 1


# =========================================================================
# 3. Policy Iteration
# =========================================================================


class TestPolicyIteration:
    """Test ClassicalPolicyIteration matches VI.

    Invariant guarded: PI converges to the same V* and pi* as VI
    (bit-exact).
    Spec line: SS8.1 item (3) -- PI == VI (bit-exact V*).
    """

    def test_pi_matches_vi(self, chain_mdp, reference_solution):
        """PI V* and pi* must be bit-exact with VI solution.
        # docs/specs/phase_I_*.md SS8.1 item (3)
        """
        Q_ref, V_ref, pi_ref = reference_solution

        pi_alg = ClassicalPolicyIteration(chain_mdp).run()
        np.testing.assert_equal(pi_alg.V, V_ref)
        np.testing.assert_equal(pi_alg.pi, pi_ref)
        np.testing.assert_equal(pi_alg.Q, Q_ref)

    def test_pi_policy_stable(self, chain_mdp):
        """PI must terminate with policy_stable=True on this small MDP.
        # docs/specs/phase_I_*.md SS8.1 item (3)
        """
        pi_alg = ClassicalPolicyIteration(chain_mdp).run()
        assert pi_alg.policy_stable is True

    def test_pi_few_iterations(self, chain_mdp):
        """PI on the 3-state chain should converge in very few iterations.
        # docs/specs/phase_I_*.md SS8.1 item (3)
        """
        pi_alg = ClassicalPolicyIteration(chain_mdp).run()
        # On this trivial MDP, PI should converge in at most 3 iterations.
        assert pi_alg.n_iters <= 3


# =========================================================================
# 4. Modified Policy Iteration
# =========================================================================


class TestModifiedPolicyIteration:
    """Test ClassicalModifiedPolicyIteration converges to VI fixed point.

    Invariant guarded: MPI converges to the same V* as VI regardless
    of the inner-sweep count m.
    Spec line: SS8.1 item (4) -- MPI converges to same fixed point as VI.
    """

    def test_mpi_matches_vi(self, chain_mdp, reference_solution):
        """MPI V* must match VI V* (bit-exact after convergence).
        # docs/specs/phase_I_*.md SS8.1 item (4)
        """
        Q_ref, V_ref, pi_ref = reference_solution

        mpi = ClassicalModifiedPolicyIteration(
            chain_mdp, m=5, max_iter=100
        ).run()
        np.testing.assert_equal(mpi.V, V_ref)
        np.testing.assert_equal(mpi.pi, pi_ref)

    @pytest.mark.parametrize("m", [1, 2, 3, 10])
    def test_mpi_various_m_converge_to_vi(self, chain_mdp, reference_solution, m):
        """MPI must converge to VI V* for any m >= 1.
        # docs/specs/phase_I_*.md SS8.1 item (4)
        """
        _, V_ref, pi_ref = reference_solution

        mpi = ClassicalModifiedPolicyIteration(
            chain_mdp, m=m, max_iter=100
        ).run()
        np.testing.assert_equal(mpi.V, V_ref)
        np.testing.assert_equal(mpi.pi, pi_ref)

    def test_mpi_m1_single_iter_equals_vi_single_sweep(
        self, chain_mdp, reference_solution
    ):
        """MPI with m=1, max_iter=1 must be bit-exact with VI n_sweeps=1.
        # docs/specs/phase_I_*.md SS8.1 item (4)

        This tests the documented extreme-case equivalence: m=1 collapses
        MPI to a single VI backward sweep.
        """
        Q_ref, V_ref, _ = reference_solution

        mpi = ClassicalModifiedPolicyIteration(
            chain_mdp, m=1, max_iter=1
        ).run()
        vi = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()

        np.testing.assert_equal(mpi.Q, vi.Q)
        np.testing.assert_equal(mpi.V, vi.V)
        np.testing.assert_equal(mpi.pi, vi.pi)


# =========================================================================
# 5. Async Value Iteration
# =========================================================================


class TestAsyncVI:
    """Test ClassicalAsyncValueIteration.

    Invariant guarded:
    (a) Sequential-order async VI is bit-exact with synchronous VI.
    (b) All update orders converge to the same V*.
    Spec line: SS8.1 item (5) -- Async-VI converges to same optimum.
    """

    def test_async_vi_sequential_matches_vi(self, chain_mdp, reference_solution):
        """Async VI (sequential order) must be bit-exact with sync VI.
        # docs/specs/phase_I_*.md SS8.1 item (5)
        """
        Q_ref, V_ref, pi_ref = reference_solution

        avi = ClassicalAsyncValueIteration(
            chain_mdp, n_sweeps=1, order="sequential"
        ).run()

        np.testing.assert_equal(avi.Q, Q_ref)
        np.testing.assert_equal(avi.V, V_ref)
        np.testing.assert_equal(avi.pi, pi_ref)

    @pytest.mark.parametrize("order", ["sequential", "reverse", "random", "priority"])
    def test_async_vi_converges_all_orders(
        self, chain_mdp, reference_solution, order
    ):
        """All async-VI orders must converge to V*.
        # docs/specs/phase_I_*.md SS8.1 item (5)
        """
        _, V_ref, pi_ref = reference_solution

        kwargs = {"n_sweeps": 1, "order": order}
        if order == "random":
            kwargs["seed"] = 42

        avi = ClassicalAsyncValueIteration(chain_mdp, **kwargs).run()
        np.testing.assert_equal(avi.V, V_ref)
        np.testing.assert_equal(avi.pi, pi_ref)

    def test_async_vi_sequential_multi_sweep_residual(
        self, chain_mdp
    ):
        """Multi-sweep async VI: second sweep residual must be 0.
        # docs/specs/phase_I_*.md SS8.1
        """
        avi = ClassicalAsyncValueIteration(
            chain_mdp, n_sweeps=3, order="sequential"
        ).run()
        assert len(avi.residuals) == 3
        np.testing.assert_allclose(avi.residuals[1], 0.0, atol=1e-15)
        np.testing.assert_allclose(avi.residuals[2], 0.0, atol=1e-15)

    def test_async_vi_converged_flag(self, chain_mdp):
        """Single-sweep async VI must report converged=True.
        # docs/specs/phase_I_*.md SS8.1
        """
        avi = ClassicalAsyncValueIteration(
            chain_mdp, n_sweeps=1, order="sequential"
        ).run()
        assert avi.converged is True


# =========================================================================
# 6. Per-sweep value snapshots (curve logging)
# =========================================================================


class TestVSweepHistory:
    """``V_sweep_history`` must align with ``residuals`` for DP runners."""

    def test_vi_multi_sweep_history_length(self, chain_mdp) -> None:
        vi = ClassicalValueIteration(chain_mdp, n_sweeps=4).run()
        assert len(vi.V_sweep_history) == len(vi.residuals) == vi.n_sweeps

    def test_async_vi_history_length(self, chain_mdp) -> None:
        avi = ClassicalAsyncValueIteration(
            chain_mdp, n_sweeps=3, order="sequential"
        ).run()
        assert len(avi.V_sweep_history) == len(avi.residuals) == avi.n_sweeps

    def test_pi_history_length(self, chain_mdp) -> None:
        pi_alg = ClassicalPolicyIteration(chain_mdp, max_iter=20).run()
        assert len(pi_alg.V_sweep_history) == len(pi_alg.residuals) == pi_alg.n_iters

    def test_mpi_history_length(self, chain_mdp) -> None:
        mpi = ClassicalModifiedPolicyIteration(
            chain_mdp, m=2, max_iter=15
        ).run()
        assert len(mpi.V_sweep_history) == len(mpi.residuals) == mpi.n_iters

    def test_pe_single_snapshot(self, chain_mdp, reference_solution) -> None:
        _, _, pi_opt = reference_solution
        pe = ClassicalPolicyEvaluation(chain_mdp, pi=pi_opt).run()
        assert len(pe.V_sweep_history) == len(pe.residuals) == 1
