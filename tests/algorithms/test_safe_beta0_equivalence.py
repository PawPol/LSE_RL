"""
Tests verifying that all safe DP planners with a zero beta schedule produce
bit-identical results to their classical counterparts.

Spec reference: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3

Required invariants:
1. Safe VI with zero schedule matches classical VI exactly.
2. Safe PE with zero schedule matches classical PE exactly.
3. Safe PI with zero schedule matches classical PI exactly.
4. Safe MPI with zero schedule matches classical MPI exactly.
5. Safe async VI with zero schedule matches classical async VI exactly
   (all orders).
6. Safe Q-learning == classical when beta_used_t = 0 (stub -- not yet
   implemented).
7. Safe expected-SARSA == classical when beta_used_t = 0 (stub -- not yet
   implemented).
"""
from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------------------
# sys.path setup so the test can find mushroom-rl-dev and the experiments
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
from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule
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


@pytest.fixture
def chain_mdp():
    return make_chain_mdp()


@pytest.fixture
def zero_schedule():
    """BetaSchedule.zeros(T, gamma) -- the classical-collapse schedule."""
    return BetaSchedule.zeros(_HORIZON, _GAMMA)


@pytest.fixture
def all_zero_policy():
    """Deterministic policy: always take action 0 at every (t, s)."""
    return np.zeros((_HORIZON, _S), dtype=np.int64)


@pytest.fixture
def all_stay_policy():
    """Deterministic policy: always take action 1 (stay) at every (t, s)."""
    return np.ones((_HORIZON, _S), dtype=np.int64)


# =========================================================================
# 1. Safe Value Iteration -- beta=0 equivalence
# =========================================================================


class TestSafeVIBeta0EquivalenceToClassical:
    """Safe VI with zero schedule must be bit-identical to classical VI.

    Invariant guarded: when beta_used == 0 at every stage, the safe
    weighted-LSE backup collapses to the exact classical Bellman backup
    r + gamma * v, so the safe planner must produce identical Q, V, pi.

    Spec line: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (1)
    """

    def test_safe_vi_zero_schedule_matches_classical_v(
        self, chain_mdp, zero_schedule
    ):
        """Safe VI V must be bit-equal to classical VI V.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (1)
        """
        classical = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()
        safe = SafeWeightedValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=1
        ).run()

        np.testing.assert_equal(
            safe.V, classical.V,
            err_msg="Safe VI V differs from classical VI V at beta=0"
        )

    def test_safe_vi_zero_schedule_matches_classical_q(
        self, chain_mdp, zero_schedule
    ):
        """Safe VI Q must be bit-equal to classical VI Q.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (1)
        """
        classical = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()
        safe = SafeWeightedValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=1
        ).run()

        np.testing.assert_equal(
            safe.Q, classical.Q,
            err_msg="Safe VI Q differs from classical VI Q at beta=0"
        )

    def test_safe_vi_zero_schedule_matches_classical_pi(
        self, chain_mdp, zero_schedule
    ):
        """Safe VI pi must be bit-equal to classical VI pi.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (1)
        """
        classical = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()
        safe = SafeWeightedValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=1
        ).run()

        np.testing.assert_equal(
            safe.pi, classical.pi,
            err_msg="Safe VI pi differs from classical VI pi at beta=0"
        )

    def test_safe_vi_zero_schedule_multi_sweep(
        self, chain_mdp, zero_schedule
    ):
        """Safe VI with multiple sweeps must match classical multi-sweep VI.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (1)
        """
        classical = ClassicalValueIteration(chain_mdp, n_sweeps=3).run()
        safe = SafeWeightedValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=3
        ).run()

        np.testing.assert_equal(safe.V, classical.V)
        np.testing.assert_equal(safe.Q, classical.Q)
        np.testing.assert_equal(safe.pi, classical.pi)


# =========================================================================
# 2. Safe Policy Evaluation -- beta=0 equivalence
# =========================================================================


class TestSafePEBeta0EquivalenceToClassical:
    """Safe PE with zero schedule must be bit-identical to classical PE.

    Invariant guarded: when beta_used == 0, safe PE produces the same
    Q^pi and V^pi as classical PE for any fixed deterministic policy.

    Spec line: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (2)
    """

    def test_safe_pe_zero_schedule_optimal_policy(
        self, chain_mdp, zero_schedule
    ):
        """Safe PE under the optimal policy matches classical PE.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (2)
        """
        # Get optimal policy from classical VI
        vi = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()
        pi_opt = vi.pi

        classical = ClassicalPolicyEvaluation(chain_mdp, pi=pi_opt).run()
        safe = SafeWeightedPolicyEvaluation(
            chain_mdp, pi=pi_opt, schedule=zero_schedule
        ).run()

        np.testing.assert_equal(
            safe.V, classical.V,
            err_msg="Safe PE V differs from classical PE V at beta=0"
        )
        np.testing.assert_equal(
            safe.Q, classical.Q,
            err_msg="Safe PE Q differs from classical PE Q at beta=0"
        )

    def test_safe_pe_zero_schedule_all_zero_policy(
        self, chain_mdp, zero_schedule, all_zero_policy
    ):
        """Safe PE under the all-action-0 policy matches classical PE.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (2)
        """
        classical = ClassicalPolicyEvaluation(
            chain_mdp, pi=all_zero_policy
        ).run()
        safe = SafeWeightedPolicyEvaluation(
            chain_mdp, pi=all_zero_policy, schedule=zero_schedule
        ).run()

        np.testing.assert_equal(safe.V, classical.V)
        np.testing.assert_equal(safe.Q, classical.Q)

    def test_safe_pe_zero_schedule_stay_policy(
        self, chain_mdp, zero_schedule, all_stay_policy
    ):
        """Safe PE under the stay-everywhere policy matches classical PE.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (2)

        The stay policy collects zero reward, so V should be all zeros.
        """
        classical = ClassicalPolicyEvaluation(
            chain_mdp, pi=all_stay_policy
        ).run()
        safe = SafeWeightedPolicyEvaluation(
            chain_mdp, pi=all_stay_policy, schedule=zero_schedule
        ).run()

        np.testing.assert_equal(safe.V, classical.V)
        np.testing.assert_equal(safe.Q, classical.Q)
        # Both should be all zeros under the stay policy.
        np.testing.assert_equal(safe.V, np.zeros((_HORIZON + 1, _S)))


# =========================================================================
# 3. Safe Policy Iteration -- beta=0 equivalence
# =========================================================================


class TestSafePIBeta0EquivalenceToClassical:
    """Safe PI with zero schedule must be bit-identical to classical PI.

    Invariant guarded: when beta_used == 0, safe PI converges to the
    same V* and pi* as classical PI (both should match VI solution).

    Spec line: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (1)
    (PI is an extension of item 1 -- safe DP planners with zero schedule
    must match their classical counterparts.)
    """

    def test_safe_pi_zero_schedule_matches_classical(
        self, chain_mdp, zero_schedule
    ):
        """Safe PI V, Q, pi must match classical PI.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
        """
        classical = ClassicalPolicyIteration(chain_mdp).run()
        safe = SafeWeightedPolicyIteration(
            chain_mdp, schedule=zero_schedule
        ).run()

        np.testing.assert_equal(
            safe.V, classical.V,
            err_msg="Safe PI V differs from classical PI V at beta=0"
        )
        np.testing.assert_equal(
            safe.Q, classical.Q,
            err_msg="Safe PI Q differs from classical PI Q at beta=0"
        )
        np.testing.assert_equal(
            safe.pi, classical.pi,
            err_msg="Safe PI pi differs from classical PI pi at beta=0"
        )

    def test_safe_pi_zero_schedule_policy_stable(
        self, chain_mdp, zero_schedule
    ):
        """Safe PI must terminate with policy_stable=True at beta=0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
        """
        safe = SafeWeightedPolicyIteration(
            chain_mdp, schedule=zero_schedule
        ).run()
        assert safe.policy_stable is True


# =========================================================================
# 4. Safe Modified Policy Iteration -- beta=0 equivalence
# =========================================================================


class TestSafeMPIBeta0EquivalenceToClassical:
    """Safe MPI with zero schedule must be bit-identical to classical MPI.

    Invariant guarded: when beta_used == 0, safe MPI produces the same
    V* and pi* as classical MPI for any value of m.

    Spec line: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
    """

    @pytest.mark.parametrize("m", [1, 2, 3, 10])
    def test_safe_mpi_zero_schedule_matches_classical(
        self, chain_mdp, zero_schedule, m
    ):
        """Safe MPI must match classical MPI for all m values at beta=0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
        """
        classical = ClassicalModifiedPolicyIteration(
            chain_mdp, m=m, max_iter=100
        ).run()
        safe = SafeWeightedModifiedPolicyIteration(
            chain_mdp, schedule=zero_schedule, m=m, max_iter=100
        ).run()

        np.testing.assert_equal(
            safe.V, classical.V,
            err_msg=f"Safe MPI V differs from classical MPI V at beta=0, m={m}"
        )
        np.testing.assert_equal(
            safe.pi, classical.pi,
            err_msg=f"Safe MPI pi differs from classical MPI pi at beta=0, m={m}"
        )

    def test_safe_mpi_m1_single_iter_equals_safe_vi(
        self, chain_mdp, zero_schedule
    ):
        """Safe MPI m=1, max_iter=1 must equal safe VI n_sweeps=1.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3

        This tests the extreme-case equivalence: m=1 collapses MPI to VI.
        """
        safe_vi = SafeWeightedValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=1
        ).run()
        safe_mpi = SafeWeightedModifiedPolicyIteration(
            chain_mdp, schedule=zero_schedule, m=1, max_iter=1
        ).run()

        np.testing.assert_equal(safe_mpi.Q, safe_vi.Q)
        np.testing.assert_equal(safe_mpi.V, safe_vi.V)
        np.testing.assert_equal(safe_mpi.pi, safe_vi.pi)


# =========================================================================
# 5. Safe Async Value Iteration -- beta=0 equivalence
# =========================================================================


class TestSafeAsyncVIBeta0EquivalenceToClassical:
    """Safe async VI with zero schedule must match classical async VI.

    Invariant guarded: when beta_used == 0, the safe async VI planner
    produces bit-identical results to the classical async VI for every
    update order.

    Spec line: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (1)
    """

    def test_safe_async_vi_sequential_matches_classical(
        self, chain_mdp, zero_schedule
    ):
        """Sequential order: safe async VI == classical async VI at beta=0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
        """
        classical = ClassicalAsyncValueIteration(
            chain_mdp, n_sweeps=1, order="sequential"
        ).run()
        safe = SafeWeightedAsyncValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=1, order="sequential"
        ).run()

        np.testing.assert_equal(safe.V, classical.V)
        np.testing.assert_equal(safe.Q, classical.Q)
        np.testing.assert_equal(safe.pi, classical.pi)

    def test_safe_async_vi_reverse_matches_classical(
        self, chain_mdp, zero_schedule
    ):
        """Reverse order: safe async VI == classical async VI at beta=0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
        """
        classical = ClassicalAsyncValueIteration(
            chain_mdp, n_sweeps=1, order="reverse"
        ).run()
        safe = SafeWeightedAsyncValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=1, order="reverse"
        ).run()

        np.testing.assert_equal(safe.V, classical.V)
        np.testing.assert_equal(safe.Q, classical.Q)
        np.testing.assert_equal(safe.pi, classical.pi)

    def test_safe_async_vi_random_matches_classical(
        self, chain_mdp, zero_schedule
    ):
        """Random order with fixed seed: safe == classical at beta=0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
        """
        seed = 42
        classical = ClassicalAsyncValueIteration(
            chain_mdp, n_sweeps=1, order="random", seed=seed
        ).run()
        safe = SafeWeightedAsyncValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=1,
            order="random", seed=seed
        ).run()

        np.testing.assert_equal(safe.V, classical.V)
        np.testing.assert_equal(safe.Q, classical.Q)
        np.testing.assert_equal(safe.pi, classical.pi)

    def test_safe_async_vi_priority_matches_classical(
        self, chain_mdp, zero_schedule
    ):
        """Priority order: safe async VI == classical async VI at beta=0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
        """
        classical = ClassicalAsyncValueIteration(
            chain_mdp, n_sweeps=1, order="priority"
        ).run()
        safe = SafeWeightedAsyncValueIteration(
            chain_mdp, schedule=zero_schedule, n_sweeps=1, order="priority"
        ).run()

        np.testing.assert_equal(safe.V, classical.V)
        np.testing.assert_equal(safe.Q, classical.Q)
        np.testing.assert_equal(safe.pi, classical.pi)

    @pytest.mark.parametrize(
        "order", ["sequential", "reverse", "random", "priority"]
    )
    def test_safe_async_vi_all_orders_converge_to_vi(
        self, chain_mdp, zero_schedule, order
    ):
        """All async-VI orders at beta=0 must converge to the same V* as VI.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
        """
        vi = ClassicalValueIteration(chain_mdp, n_sweeps=1).run()

        kwargs = {
            "n_sweeps": 1,
            "order": order,
        }
        if order == "random":
            kwargs["seed"] = 42

        safe = SafeWeightedAsyncValueIteration(
            chain_mdp, schedule=zero_schedule, **kwargs
        ).run()

        np.testing.assert_equal(safe.V, vi.V)
        np.testing.assert_equal(safe.pi, vi.pi)


# =========================================================================
# Shared helpers for online TD equivalence tests
# =========================================================================

from mushroom_rl.rl_utils.spaces import Discrete
from mushroom_rl.core.environment import MDPInfo
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.policy.td_policy import EpsGreedy
from mushroom_rl.algorithms.value.td.q_learning import QLearning
from mushroom_rl.algorithms.value.td.expected_sarsa import ExpectedSARSA
from mushroom_rl.algorithms.value.td.safe_q_learning import SafeQLearning
from mushroom_rl.algorithms.value.td.safe_expected_sarsa import SafeExpectedSARSA
from mushroom_rl.algorithms.value.td.safe_td0 import SafeTD0

_TD_T = 5          # horizon
_TD_N_BASE = 3     # number of base states
_TD_N_AUG = _TD_T * _TD_N_BASE   # 15 augmented states
_TD_N_ACT = 2
_TD_GAMMA = _GAMMA  # reuse 0.99 from module level
_TD_ALPHA = 0.5


def _make_td_mdp_info():
    """Minimal MDPInfo for augmented-state TD tests."""
    obs = Discrete(_TD_N_AUG)
    act = Discrete(_TD_N_ACT)
    return MDPInfo(obs, act, _TD_GAMMA, _TD_T)


def _td_zero_schedule():
    """BetaSchedule.zeros for the augmented-state horizon."""
    return BetaSchedule.zeros(_TD_T, _TD_GAMMA)


def _make_greedy_policy(Q):
    eps = Parameter(0.0)
    p = EpsGreedy(eps)
    p.set_q(Q)
    return p


def _init_q_same(alg1, alg2, seed=7):
    """Initialise both Q-tables to the same random values."""
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal((_TD_N_AUG, _TD_N_ACT))
    alg1.Q.table[:] = vals
    alg2.Q.table[:] = vals


# =========================================================================
# 6. Safe TD0 -- beta=0 equivalence
# =========================================================================


class TestSafeTD0Beta0Equivalence:
    """Safe TD(0) with beta=0 must match classical TD(0).

    Spec line: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3
    """

    def test_safe_td0_matches_classical_at_beta0(self):
        """Safe TD(0) update must equal classical TD(0) update when beta_used_t = 0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3

        SafeTD0._update formula (beta=0):
            v_current = Q[s,:].dot(pi(s))
            v_next    = Q[s',:].dot(pi(s'))
            target    = r + gamma * v_next   (exact classical at beta=0)
            Q[s,a]   += alpha * (target - v_current)

        We verify by re-implementing the same formula manually and checking
        that SafeTD0 produces the identical result.
        """
        mdp_info = _make_td_mdp_info()
        sch = _td_zero_schedule()

        safe = SafeTD0(mdp_info, EpsGreedy(Parameter(0.0)), sch, _TD_N_BASE, Parameter(_TD_ALPHA))

        # Set known Q values
        rng = np.random.default_rng(7)
        vals = rng.standard_normal((_TD_N_AUG, _TD_N_ACT))
        safe.Q.table[:] = vals
        safe.policy.set_q(safe.Q)

        state, action, reward, next_state = 0, 0, 1.0, 1

        # Manual classical TD(0) reference (same formula as SafeTD0._update at beta=0)
        v_current_ref = safe.Q[state, :].dot(safe.policy(state))
        v_next_ref = safe.Q[next_state, :].dot(safe.policy(next_state))
        classical_target = float(reward) + _TD_GAMMA * v_next_ref
        q_before = float(vals[state, action])
        expected_q = q_before + _TD_ALPHA * (classical_target - v_current_ref)

        safe._update(state, action, reward, next_state, False)

        # At beta=0, safe_target == r + gamma * v_next exactly
        np.testing.assert_allclose(
            safe.Q[state, action], expected_q, atol=1e-14,
            err_msg="SafeTD0 Q-value does not match classical TD(0) formula at beta=0"
        )


# =========================================================================
# 7. Safe Q-learning -- beta=0 equivalence
# =========================================================================


class TestSafeQLBeta0Equivalence:
    """Safe Q-learning with beta=0 must match classical Q-learning.

    Spec line: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (3)
    """

    def test_safe_qlearning_matches_classical_at_beta0(self):
        """Safe Q-learning update == classical when beta_used_t = 0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (3)
        """
        mdp_info = _make_td_mdp_info()
        sch = _td_zero_schedule()

        classical = QLearning(mdp_info, EpsGreedy(Parameter(0.0)), Parameter(_TD_ALPHA))
        safe = SafeQLearning(mdp_info, EpsGreedy(Parameter(0.0)), sch, _TD_N_BASE, Parameter(_TD_ALPHA))

        _init_q_same(classical, safe)
        classical.policy.set_q(classical.Q)
        safe.policy.set_q(safe.Q)

        state, action, reward, next_state = 0, 0, 1.0, 1

        classical._update(state, action, reward, next_state, False)
        safe._update(state, action, reward, next_state, False)

        np.testing.assert_equal(
            safe.Q[state, action], classical.Q[state, action],
            err_msg="SafeQLearning Q-value differs from classical QLearning at beta=0"
        )

    def test_safe_qlearning_matches_classical_multiple_updates(self):
        """Multiple safe QL updates == classical when beta=0 across transitions.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (3)
        """
        mdp_info = _make_td_mdp_info()
        sch = _td_zero_schedule()

        classical = QLearning(mdp_info, EpsGreedy(Parameter(0.0)), Parameter(_TD_ALPHA))
        safe = SafeQLearning(mdp_info, EpsGreedy(Parameter(0.0)), sch, _TD_N_BASE, Parameter(_TD_ALPHA))

        _init_q_same(classical, safe)
        classical.policy.set_q(classical.Q)
        safe.policy.set_q(safe.Q)

        rng = np.random.default_rng(42)
        for _ in range(20):
            s = int(rng.integers(0, _TD_N_AUG))
            a = int(rng.integers(0, _TD_N_ACT))
            r = float(rng.standard_normal())
            s_next = int(rng.integers(0, _TD_N_AUG))
            classical._update(s, a, r, s_next, False)
            safe._update(s, a, r, s_next, False)

        np.testing.assert_array_equal(
            safe.Q.table, classical.Q.table,
            err_msg="SafeQLearning Q-table differs from classical after 20 updates at beta=0"
        )


# =========================================================================
# 8. Safe Expected-SARSA -- beta=0 equivalence
# =========================================================================


class TestSafeESARSABeta0Equivalence:
    """Safe expected-SARSA with beta=0 must match classical expected-SARSA.

    Spec line: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (4)
    """

    def test_safe_expected_sarsa_matches_classical_at_beta0(self):
        """Safe expected-SARSA update == classical when beta_used_t = 0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (4)
        """
        mdp_info = _make_td_mdp_info()
        sch = _td_zero_schedule()

        classical = ExpectedSARSA(mdp_info, EpsGreedy(Parameter(0.0)), Parameter(_TD_ALPHA))
        safe = SafeExpectedSARSA(mdp_info, EpsGreedy(Parameter(0.0)), sch, _TD_N_BASE, Parameter(_TD_ALPHA))

        _init_q_same(classical, safe)
        classical.policy.set_q(classical.Q)
        safe.policy.set_q(safe.Q)

        state, action, reward, next_state = 0, 0, 1.0, 1

        classical._update(state, action, reward, next_state, False)
        safe._update(state, action, reward, next_state, False)

        np.testing.assert_equal(
            safe.Q[state, action], classical.Q[state, action],
            err_msg="SafeExpectedSARSA Q-value differs from classical at beta=0"
        )

    def test_safe_expected_sarsa_matches_classical_multiple_updates(self):
        """Multiple safe ESARSA updates == classical when beta=0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.3 item (4)
        """
        mdp_info = _make_td_mdp_info()
        sch = _td_zero_schedule()

        classical = ExpectedSARSA(mdp_info, EpsGreedy(Parameter(0.0)), Parameter(_TD_ALPHA))
        safe = SafeExpectedSARSA(mdp_info, EpsGreedy(Parameter(0.0)), sch, _TD_N_BASE, Parameter(_TD_ALPHA))

        _init_q_same(classical, safe)
        classical.policy.set_q(classical.Q)
        safe.policy.set_q(safe.Q)

        rng = np.random.default_rng(99)
        for _ in range(20):
            s = int(rng.integers(0, _TD_N_AUG))
            a = int(rng.integers(0, _TD_N_ACT))
            r = float(rng.standard_normal())
            s_next = int(rng.integers(0, _TD_N_AUG))
            classical._update(s, a, r, s_next, False)
            safe._update(s, a, r, s_next, False)

        np.testing.assert_array_equal(
            safe.Q.table, classical.Q.table,
            err_msg="SafeExpectedSARSA Q-table differs from classical after 20 updates at beta=0"
        )
