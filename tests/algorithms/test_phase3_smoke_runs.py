"""
End-to-end smoke tests for Phase III safe DP and safe QL runs.

Spec reference: docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4

Required tests:
1. One short safe DP run finishes and logs schedule fields.
2. One short safe Q-learning run finishes and logs rho_t, effective_discount_t,
   and clipping activity.
3. Aggregation (validate_safe_transitions_npz) runs on smoke outputs.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
from typing import Any

# ---------------------------------------------------------------------------
# sys.path setup (standard pattern for all Phase III tests)
# ---------------------------------------------------------------------------

def _find_repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mushroom-rl-dev").is_dir():
            return parent
    return here.parents[2]


_REPO_ROOT = _find_repo_root()
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
_EXPERIMENTS = _REPO_ROOT / "experiments"

for _p in (_REPO_ROOT, _MUSHROOM_DEV, _EXPERIMENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pytest

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.algorithms.value.dp.safe_weighted_value_iteration import (
    SafeWeightedValueIteration,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
)
from mushroom_rl.algorithms.value.td.safe_q_learning import SafeQLearning
from mushroom_rl.rl_utils.spaces import Discrete
from mushroom_rl.core.environment import MDPInfo
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.policy.td_policy import EpsGreedy

from experiments.weighted_lse_dp.common.callbacks import SafeTransitionLogger
from experiments.weighted_lse_dp.common.schemas import (
    SAFE_TRANSITIONS_ARRAYS,
    SAFE_CALIBRATION_ARRAYS,
    aggregate_safe_stats,
    validate_safe_transitions_npz,
)
from experiments.weighted_lse_dp.common.io import save_npz_with_schema, make_npz_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def chain_mdp():
    """3-state chain FiniteMDP with T=5, gamma=0.9.

    State 0 -> 1 -> 2 (action 0 = forward, action 1 = stay).
    Reward: R[1, 0, 2] = 1.0 (reaching state 2 from state 1).
    """
    P = np.zeros((3, 2, 3))
    R = np.zeros((3, 2, 3))
    for s in range(3):
        P[s, 0, min(s + 1, 2)] = 1.0
        P[s, 1, s] = 1.0
    R[1, 0, 2] = 1.0
    return FiniteMDP(P, R, mu=None, gamma=0.9, horizon=5)


@pytest.fixture()
def zero_schedule():
    """All-zero BetaSchedule (classical collapse), T=5, gamma=0.9."""
    return BetaSchedule.zeros(5, 0.9)


@pytest.fixture()
def ql_agent(zero_schedule):
    """SafeQLearning agent on a 3-state chain with T=5, augmented state space."""
    T = 5
    n_base = 3
    n_aug = T * n_base
    n_act = 2
    gamma = 0.9

    obs = Discrete(n_aug)
    act = Discrete(n_act)
    mdp_info = MDPInfo(obs, act, gamma, T)
    policy = EpsGreedy(Parameter(0.0))

    agent = SafeQLearning(mdp_info, policy, zero_schedule, n_base, Parameter(0.5))
    policy.set_q(agent.Q)
    return agent


# ===========================================================================
# Part (a): Safe DP smoke run
# ===========================================================================


class TestSafeDPSmokeRun:
    """Smoke tests for SafeWeightedValueIteration.

    Invariant guarded: SafeWeightedValueIteration runs to completion on a
    tiny MDP and produces correctly shaped V/Q tables, accessible schedule
    fields, and per-sweep residuals/history. If SafeWeightedValueIteration
    is broken (wrong backward pass, missing attributes), these tests fail.
    # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4 item 1
    """

    def test_safe_vi_runs_to_completion(self, chain_mdp, zero_schedule):
        """Safe VI runs without exception and produces V, Q of correct shape.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        planner = SafeWeightedValueIteration(chain_mdp, zero_schedule, n_sweeps=2)
        result = planner.run()

        assert result is planner, "run() should return self for fluent chaining"
        # V shape: (T+1, S) = (6, 3)
        assert planner.V.shape == (6, 3), f"V.shape = {planner.V.shape}, expected (6, 3)"
        # Q shape: (T, S, A) = (5, 3, 2)
        assert planner.Q.shape == (5, 3, 2), f"Q.shape = {planner.Q.shape}, expected (5, 3, 2)"

    def test_safe_vi_schedule_fields_accessible(self, zero_schedule):
        """BetaSchedule exposes beta_used_at, beta_cap_at, kappa_at as finite floats.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        beta_used = zero_schedule.beta_used_at(0)
        beta_cap = zero_schedule.beta_cap_at(0)
        kappa = zero_schedule.kappa_at(0)

        assert np.isfinite(beta_used), f"beta_used_at(0) = {beta_used} is not finite"
        assert np.isfinite(beta_cap), f"beta_cap_at(0) = {beta_cap} is not finite"
        assert np.isfinite(kappa), f"kappa_at(0) = {kappa} is not finite"
        assert isinstance(beta_used, float)
        assert isinstance(beta_cap, float)
        assert isinstance(kappa, float)

    def test_safe_vi_residuals_populated(self, chain_mdp, zero_schedule):
        """After 2-sweep run, residuals is a list of length 2 with finite entries.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        planner = SafeWeightedValueIteration(chain_mdp, zero_schedule, n_sweeps=2)
        planner.run()

        assert len(planner.residuals) == 2, (
            f"Expected 2 residuals, got {len(planner.residuals)}"
        )
        for i, r in enumerate(planner.residuals):
            assert np.isfinite(r), f"residuals[{i}] = {r} is not finite"

    def test_safe_vi_v_sweep_history_populated(self, chain_mdp, zero_schedule):
        """After 2-sweep run, V_sweep_history has length 2 and correct shapes.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        planner = SafeWeightedValueIteration(chain_mdp, zero_schedule, n_sweeps=2)
        planner.run()

        assert len(planner.V_sweep_history) == 2, (
            f"Expected 2 V snapshots, got {len(planner.V_sweep_history)}"
        )
        for i, v_snap in enumerate(planner.V_sweep_history):
            assert v_snap.shape == (6, 3), (
                f"V_sweep_history[{i}].shape = {v_snap.shape}, expected (6, 3)"
            )


# ===========================================================================
# Part (b): Safe QL smoke run with SafeTransitionLogger
# ===========================================================================


class TestSafeQLSmoke:
    """Smoke tests for SafeQLearning with SafeTransitionLogger.

    Invariant guarded: SafeQLearning._update() populates swc instrumentation
    fields and SafeTransitionLogger accumulates them into a payload with all
    SAFE_TRANSITIONS_ARRAYS keys. If the safe target or logger is broken,
    these tests fail.
    # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4 item 2
    """

    def test_safe_ql_update_populates_swc_fields(self, ql_agent):
        """After one _update, swc.last_* fields are populated (not defaults).
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        # Perform one update: state=0 (stage 0, base_state 0), action=0,
        # reward=1.0, next_state=1 (stage 0, base_state 1), not absorbing.
        ql_agent._update(0, 0, 1.0, 1, False)

        swc = ql_agent.swc
        assert swc.last_stage == 0, f"last_stage = {swc.last_stage}, expected 0"
        assert np.isfinite(swc.last_beta_used), (
            f"last_beta_used = {swc.last_beta_used} is not finite"
        )
        assert np.isfinite(float(swc.last_rho)), (
            f"last_rho = {swc.last_rho} is not finite"
        )
        assert np.isfinite(float(swc.last_effective_discount)), (
            f"last_effective_discount = {swc.last_effective_discount} is not finite"
        )
        assert isinstance(swc.last_clip_active, bool), (
            f"last_clip_active type = {type(swc.last_clip_active)}"
        )

    def test_safe_transition_logger_accumulates(self, ql_agent):
        """SafeTransitionLogger accumulates 5 transitions with all 10 safe keys.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        n_base = 3
        gamma = 0.9
        n_aug = 5 * n_base
        logger = SafeTransitionLogger(ql_agent, n_base=n_base, gamma=gamma)

        for i in range(5):
            state = i % n_aug
            next_state = (i + 1) % n_aug
            # _update populates swc fields
            ql_agent._update(state, 0, 1.0, next_state, False)
            sample = (
                np.array([state]),
                np.array([0]),
                1.0,
                np.array([next_state]),
                False,
                (i == 4),
            )
            logger(sample)

        payload = logger.build_safe_payload()

        # Check all 10 SAFE_TRANSITIONS_ARRAYS keys are present.
        assert len(SAFE_TRANSITIONS_ARRAYS) == 10, (
            f"Expected 10 safe array keys, got {len(SAFE_TRANSITIONS_ARRAYS)}"
        )
        for key in SAFE_TRANSITIONS_ARRAYS:
            assert key in payload, f"Missing key: {key}"
            assert len(payload[key]) == 5, (
                f"payload['{key}'] has length {len(payload[key])}, expected 5"
            )

    def test_safe_ql_rho_in_valid_range(self, ql_agent):
        """After 10 updates, last_rho is in [0, 1].
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        n_aug = 15
        for i in range(10):
            state = i % n_aug
            next_state = (i + 1) % n_aug
            ql_agent._update(state, 0, 1.0, next_state, False)

        rho = float(ql_agent.swc.last_rho)
        assert 0.0 <= rho <= 1.0, f"last_rho = {rho}, expected in [0, 1]"

    def test_safe_ql_effective_discount_positive(self, ql_agent):
        """After updates, last_effective_discount > 0.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        ql_agent._update(0, 0, 1.0, 1, False)
        ed = float(ql_agent.swc.last_effective_discount)
        assert ed > 0.0, f"last_effective_discount = {ed}, expected > 0"


# ===========================================================================
# Part (c): Aggregation tests
# ===========================================================================


class TestSafeAggregation:
    """Tests for aggregate_safe_stats and validate_safe_transitions_npz.

    Invariant guarded: aggregate_safe_stats produces per-stage arrays with
    correct shapes and value ranges; validate_safe_transitions_npz correctly
    detects missing/present keys. If the schema contract or aggregation logic
    is broken, these tests fail.
    # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4 item 3
    """

    @staticmethod
    def _synthetic_payload(n_transitions: int = 25, T: int = 5) -> dict[str, np.ndarray]:
        """Build a synthetic safe payload with n_transitions spread across T stages."""
        rng = np.random.default_rng(42)
        stages = np.tile(np.arange(T), n_transitions // T + 1)[:n_transitions]
        return {
            "safe_stage": stages.astype(np.int64),
            "safe_beta_raw": rng.uniform(-0.1, 0.1, n_transitions),
            "safe_beta_cap": rng.uniform(0.0, 1.0, n_transitions),
            "safe_beta_used": rng.uniform(-0.1, 0.1, n_transitions),
            "safe_clip_active": rng.choice([True, False], n_transitions),
            "safe_rho": rng.uniform(0.0, 1.0, n_transitions),
            "safe_effective_discount": rng.uniform(0.0, 1.9, n_transitions),
            "safe_target": rng.standard_normal(n_transitions),
            "safe_margin": rng.standard_normal(n_transitions),
            "safe_td_error": rng.standard_normal(n_transitions),
        }

    def test_aggregate_safe_stats_shapes(self):
        """aggregate_safe_stats returns 10 keys each of shape (T,).
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        T = 5
        gamma = 0.9
        payload = self._synthetic_payload(n_transitions=25, T=T)
        result = aggregate_safe_stats(payload, T=T, gamma=gamma)

        assert len(result) == len(SAFE_CALIBRATION_ARRAYS), (
            f"Expected {len(SAFE_CALIBRATION_ARRAYS)} keys, got {len(result)}"
        )
        for key in SAFE_CALIBRATION_ARRAYS:
            assert key in result, f"Missing key: {key}"
            assert result[key].shape == (T,), (
                f"result['{key}'].shape = {result[key].shape}, expected ({T},)"
            )

    def test_aggregate_safe_stats_clip_fraction_range(self):
        """safe_clip_fraction values are in [0, 1].
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        T = 5
        payload = self._synthetic_payload(n_transitions=25, T=T)
        result = aggregate_safe_stats(payload, T=T, gamma=0.9)

        clip_frac = result["safe_clip_fraction"]
        # Values should be in [0, 1] (they are means of boolean arrays).
        assert np.all((clip_frac >= 0.0) & (clip_frac <= 1.0)), (
            f"safe_clip_fraction values out of [0,1]: {clip_frac}"
        )

    def test_validate_safe_transitions_npz_missing_keys(self, tmp_path):
        """validate_safe_transitions_npz returns non-empty list for incomplete NPZ.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        # Save an NPZ with only one of the safe keys.
        path = tmp_path / "incomplete.npz"
        schema = make_npz_schema(
            phase="phase3", task="test", algorithm="safe_vi",
            seed=0, storage_mode="test", arrays=["safe_stage"],
        )
        save_npz_with_schema(
            path, schema, {"safe_stage": np.array([0, 1, 2], dtype=np.int64)}
        )
        missing = validate_safe_transitions_npz(path)
        assert len(missing) > 0, "Expected non-empty missing keys for incomplete NPZ"
        # Should be missing 9 of the 10 keys.
        assert len(missing) == 9, f"Expected 9 missing keys, got {len(missing)}: {missing}"

    def test_validate_safe_transitions_npz_all_present(self, tmp_path):
        """validate_safe_transitions_npz returns empty list when all keys present.
        # docs/specs/phase_III_safe_weighted_lse_experiments.md SS8.4
        """
        path = tmp_path / "complete.npz"
        payload = self._synthetic_payload(n_transitions=10, T=5)
        schema = make_npz_schema(
            phase="phase3", task="test", algorithm="safe_vi",
            seed=0, storage_mode="test", arrays=list(payload.keys()),
        )
        save_npz_with_schema(path, schema, payload)
        missing = validate_safe_transitions_npz(path)
        assert missing == [], f"Expected no missing keys, got {missing}"
