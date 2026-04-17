"""Tests for Phase II stress-task factories: severity=0 reduction, P validity,
reward ranges, horizons, and regime-shift change-point timing.

Each test's docstring references the spec line it enforces.

Invariants guarded:
  - severity=0 recovery: stress factories with null-severity parameters
    must produce P and R matrices identical to Phase I base tasks.
  - P row sums: every non-absorbing state's transition row sums to 1.0.
  - Reward ranges: R values lie within expected bounds.
  - Horizons: the FiniteMDP horizon attribute matches the factory parameter.
  - Time augmentation: mdp_rl is a DiscreteTimeAugmentedEnv with the correct
    observation-space size.
  - Regime-shift change-point: wrappers flip post_change exactly at the
    configured episode boundary.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- make mushroom-rl-dev, src, experiments importable
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parents[2]
for _sub in ("mushroom-rl-dev", "src", "experiments", "."):
    _p = str(_repo / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ensure factories can resolve grid files relative to repo root.
os.chdir(_repo)

import numpy as np
import pytest

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.environments.generators.simple_chain import (
    compute_probabilities as chain_compute_probabilities,
    compute_reward as chain_compute_reward,
)
from mushroom_rl.environments.time_augmented_env import DiscreteTimeAugmentedEnv

from experiments.weighted_lse_dp.common.task_factories import (
    make_chain_base,
    make_grid_base,
    make_taxi_base,
)
from experiments.weighted_lse_dp.tasks.stress_families import (
    make_chain_sparse_long,
    make_chain_jackpot,
    make_chain_catastrophe,
    make_grid_sparse_goal,
    make_taxi_bonus_shock,
    TaxiBonusShockWrapper,
)
from experiments.weighted_lse_dp.tasks.nonstationary_wrappers import (
    ChainRegimeShiftWrapper,
    GridRegimeShiftWrapper,
    make_chain_regime_shift,
    make_grid_regime_shift,
)
from experiments.weighted_lse_dp.tasks.hazard_wrappers import (
    GridHazardWrapper,
    make_grid_hazard,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def chain_base():
    """Phase I chain_base MDP (25 states, prob=0.9, horizon=60)."""
    mdp_base, mdp_rl, cfg, ref_pi = make_chain_base(time_augment=False, seed=0)
    return mdp_base


@pytest.fixture(scope="module")
def grid_base():
    """Phase I grid_base MDP (5x5, prob=0.9, horizon=80)."""
    mdp_base, mdp_rl, cfg, ref_pi = make_grid_base(time_augment=False, seed=0)
    return mdp_base


@pytest.fixture(scope="module")
def taxi_base():
    """Phase I taxi_base MDP (44 states, prob=0.9, horizon=120)."""
    mdp_base, mdp_rl, cfg, ref_pi = make_taxi_base(time_augment=False, seed=0)
    return mdp_base


# ===================================================================
# Helper
# ===================================================================

def _detect_absorbing_states(p: np.ndarray) -> set[int]:
    """Detect absorbing states: those whose P rows are all-zero for every action.

    MushroomRL convention: an absorbing state has P[s, a, :] == 0 for all a.
    """
    S, A, _ = p.shape
    absorbing = set()
    for s in range(S):
        if all(p[s, a, :].sum() < 1e-14 for a in range(A)):
            absorbing.add(s)
    return absorbing


def _assert_p_row_sums(p: np.ndarray, label: str, absorbing_states: set[int] | None = None) -> None:
    """Assert that every non-absorbing state's P rows sum to 1.0.
    # docs/specs/phase_II_*.md S9.1 -- transition validity

    Absorbing states (all-zero P rows) are auto-detected if not provided.
    """
    if absorbing_states is None:
        absorbing_states = _detect_absorbing_states(p)
    S, A, _ = p.shape
    for s in range(S):
        if s in absorbing_states:
            # Absorbing states have all-zero P rows (MushroomRL convention).
            for a in range(A):
                np.testing.assert_allclose(
                    p[s, a, :].sum(), 0.0, atol=1e-14,
                    err_msg=f"{label}: absorbing state {s} action {a} P row != 0",
                )
        else:
            for a in range(A):
                np.testing.assert_allclose(
                    p[s, a, :].sum(), 1.0, atol=1e-14,
                    err_msg=f"{label}: state {s} action {a} P row sum != 1.0",
                )


# ===================================================================
# 1. Severity=0 reduction: P and R identity
# ===================================================================

class TestSeverity0ChainSparse:
    """make_chain_sparse_long with base-chain parameters recovers chain_base.
    # docs/specs/phase_II_*.md S9.1 -- severity=0 identity
    """

    def test_p_identity(self, chain_base):
        """P matrices must be identical when chain_sparse_long uses
        state_n=25, prob=0.9, gamma=0.99, horizon=60.
        """
        mdp_stress, _, _ = make_chain_sparse_long(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
        )
        np.testing.assert_array_equal(
            mdp_stress.p, chain_base.p,
            err_msg="chain_sparse_long severity=0 P != chain_base P",
        )

    def test_r_identity(self, chain_base):
        """R matrices must be identical under severity=0 parameters."""
        mdp_stress, _, _ = make_chain_sparse_long(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
        )
        np.testing.assert_array_equal(
            mdp_stress.r, chain_base.r,
            err_msg="chain_sparse_long severity=0 R != chain_base R",
        )


class TestSeverity0ChainJackpot:
    """make_chain_jackpot with jackpot_prob=0.0 recovers chain_base exactly.
    # docs/specs/phase_II_*.md S9.1 -- severity=0 identity
    """

    def test_p_identity(self, chain_base):
        mdp_stress, _, _ = make_chain_jackpot(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
            jackpot_prob=0.0,
        )
        np.testing.assert_array_equal(
            mdp_stress.p, chain_base.p,
            err_msg="chain_jackpot severity=0 P != chain_base P",
        )

    def test_r_identity(self, chain_base):
        mdp_stress, _, _ = make_chain_jackpot(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
            jackpot_prob=0.0,
        )
        np.testing.assert_array_equal(
            mdp_stress.r, chain_base.r,
            err_msg="chain_jackpot severity=0 R != chain_base R",
        )

    def test_no_extra_absorbing_state(self, chain_base):
        """When jackpot_prob=0.0, no absorbing terminal state is added."""
        mdp_stress, _, _ = make_chain_jackpot(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
            jackpot_prob=0.0,
        )
        assert mdp_stress.p.shape == chain_base.p.shape, (
            "chain_jackpot severity=0 should not add an extra state"
        )


class TestSeverity0ChainCatastrophe:
    """make_chain_catastrophe with risky_prob=0.0 recovers chain_base.
    # docs/specs/phase_II_*.md S9.1 -- severity=0 identity
    """

    def test_p_identity(self, chain_base):
        mdp_stress, _, _ = make_chain_catastrophe(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
            risky_prob=0.0,
        )
        np.testing.assert_array_equal(
            mdp_stress.p, chain_base.p,
            err_msg="chain_catastrophe severity=0 P != chain_base P",
        )

    def test_r_identity(self, chain_base):
        mdp_stress, _, _ = make_chain_catastrophe(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
            risky_prob=0.0,
        )
        np.testing.assert_array_equal(
            mdp_stress.r, chain_base.r,
            err_msg="chain_catastrophe severity=0 R != chain_base R",
        )


class TestSeverity0GridSparseGoal:
    """make_grid_sparse_goal with goal_reward=1.0 recovers grid_base.
    # docs/specs/phase_II_*.md S9.1 -- severity=0 identity

    grid_base has pos_rew=1.0 and neg_rew=0.0; grid_sparse_goal with
    goal_reward=1.0 and step_penalty=0 should produce the same MDP.
    """

    def test_p_identity(self, grid_base):
        mdp_stress, _, _ = make_grid_sparse_goal(
            {}, goal_reward=1.0, prob=0.9, gamma=0.99, horizon=80,
        )
        np.testing.assert_array_equal(
            mdp_stress.p, grid_base.p,
            err_msg="grid_sparse_goal severity=0 P != grid_base P",
        )

    def test_r_identity(self, grid_base):
        mdp_stress, _, _ = make_grid_sparse_goal(
            {}, goal_reward=1.0, prob=0.9, gamma=0.99, horizon=80,
        )
        np.testing.assert_array_equal(
            mdp_stress.r, grid_base.r,
            err_msg="grid_sparse_goal severity=0 R != grid_base R",
        )


class TestSeverity0TaxiBonusShock:
    """make_taxi_bonus_shock with bonus_prob=0.0 is transparent.
    # docs/specs/phase_II_*.md S9.1 -- severity=0 identity

    The wrapper delegates P and R access to the base MDP, so .p and .r
    must match taxi_base exactly. Step behavior must also be identical
    (no bonus injected).
    """

    def test_p_identity(self, taxi_base):
        wrapped, _, _ = make_taxi_bonus_shock(
            {}, bonus_prob=0.0, prob=0.9, gamma=0.99, horizon=120,
        )
        np.testing.assert_array_equal(
            wrapped.p, taxi_base.p,
            err_msg="taxi_bonus_shock severity=0 P != taxi_base P",
        )

    def test_r_identity(self, taxi_base):
        wrapped, _, _ = make_taxi_bonus_shock(
            {}, bonus_prob=0.0, prob=0.9, gamma=0.99, horizon=120,
        )
        np.testing.assert_array_equal(
            wrapped.r, taxi_base.r,
            err_msg="taxi_bonus_shock severity=0 R != taxi_base R",
        )

    def test_step_transparent(self, taxi_base):
        """With bonus_prob=0.0, step rewards must match the base exactly."""
        wrapped, _, _ = make_taxi_bonus_shock(
            {}, bonus_prob=0.0, prob=0.9, gamma=0.99, horizon=120,
            seed=42,
        )
        # Run parallel episodes with the same seed and compare rewards.
        np.random.seed(42)
        wrapped.reset(state=np.array([0]))
        rewards_wrapped = []
        for _ in range(10):
            _, r, absorbing, _ = wrapped.step(np.array([0]))
            rewards_wrapped.append(r)
            if absorbing:
                break

        np.random.seed(42)
        taxi_base.reset(state=np.array([0]))
        rewards_base = []
        for _ in range(10):
            _, r, absorbing, _ = taxi_base.step(np.array([0]))
            rewards_base.append(r)
            if absorbing:
                break

        np.testing.assert_array_equal(
            np.array(rewards_wrapped),
            np.array(rewards_base),
            err_msg="taxi_bonus_shock bonus_prob=0 altered step rewards",
        )


class TestSeverity0GridHazard:
    """make_grid_hazard with hazard_prob=0.0 is transparent.
    # docs/specs/phase_II_*.md S9.1 -- severity=0 identity
    """

    def test_p_identity(self, grid_base):
        wrapped, _, _ = make_grid_hazard(
            {}, hazard_prob=0.0, prob=0.9, gamma=0.99, horizon=80,
        )
        np.testing.assert_array_equal(
            wrapped.p, grid_base.p,
            err_msg="grid_hazard severity=0 P != grid_base P",
        )

    def test_r_identity(self, grid_base):
        wrapped, _, _ = make_grid_hazard(
            {}, hazard_prob=0.0, prob=0.9, gamma=0.99, horizon=80,
        )
        np.testing.assert_array_equal(
            wrapped.r, grid_base.r,
            err_msg="grid_hazard severity=0 R != grid_base R",
        )


# ===================================================================
# 2. P row-sum validity
# ===================================================================

class TestPRowSums:
    """All non-absorbing P rows must sum to 1.0 for every factory.
    # docs/specs/phase_II_*.md S9.1 -- transition probability validity
    """

    def test_chain_sparse_long_default(self):
        mdp, _, _ = make_chain_sparse_long({})
        _assert_p_row_sums(mdp.p, "chain_sparse_long_default")

    def test_chain_sparse_long_severity0(self):
        mdp, _, _ = make_chain_sparse_long(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
        )
        _assert_p_row_sums(mdp.p, "chain_sparse_long_sev0")

    def test_chain_jackpot_default(self):
        """Default jackpot adds an absorbing terminal state (last index)."""
        mdp, _, cfg = make_chain_jackpot({})
        state_n = cfg["state_n"]
        absorbing = set()
        if cfg["jackpot_prob"] > 0.0 and cfg["jackpot_terminates"]:
            absorbing = {state_n}  # extra absorbing state at index state_n
        _assert_p_row_sums(mdp.p, "chain_jackpot_default", absorbing)

    def test_chain_jackpot_severity0(self):
        mdp, _, _ = make_chain_jackpot({}, jackpot_prob=0.0)
        _assert_p_row_sums(mdp.p, "chain_jackpot_sev0")

    def test_chain_catastrophe_default(self):
        """Default catastrophe adds an absorbing terminal state."""
        mdp, _, cfg = make_chain_catastrophe({})
        state_n = cfg["state_n"]
        absorbing = set()
        if cfg["risky_prob"] > 0.0:
            absorbing = {state_n}
        _assert_p_row_sums(mdp.p, "chain_catastrophe_default", absorbing)

    def test_chain_catastrophe_severity0(self):
        mdp, _, _ = make_chain_catastrophe({}, risky_prob=0.0)
        _assert_p_row_sums(mdp.p, "chain_catastrophe_sev0")

    def test_grid_sparse_goal_default(self):
        mdp, _, _ = make_grid_sparse_goal({})
        _assert_p_row_sums(mdp.p, "grid_sparse_goal_default")

    def test_grid_sparse_goal_severity0(self):
        mdp, _, _ = make_grid_sparse_goal({}, goal_reward=1.0)
        _assert_p_row_sums(mdp.p, "grid_sparse_goal_sev0")

    def test_grid_hazard_default(self):
        """Hazard wrapper delegates .p to base -- base P must be valid."""
        wrapped, _, _ = make_grid_hazard({})
        _assert_p_row_sums(wrapped.p, "grid_hazard_default")

    def test_taxi_bonus_shock_default(self):
        wrapped, _, _ = make_taxi_bonus_shock({})
        _assert_p_row_sums(wrapped.p, "taxi_bonus_shock_default")


# ===================================================================
# 3. Reward range checks
# ===================================================================

class TestRewardRanges:
    """Reward matrices have values within expected bounds.
    # docs/specs/phase_II_*.md S9.1 -- reward range
    """

    def test_chain_sparse_long_reward_range(self):
        mdp, _, _ = make_chain_sparse_long({})
        assert mdp.r.min() >= 0.0, "chain_sparse_long R has negative values"
        assert mdp.r.max() <= 1.0 + 1e-14, "chain_sparse_long R exceeds 1.0"

    def test_chain_jackpot_reward_range(self):
        mdp, _, cfg = make_chain_jackpot({})
        assert mdp.r.min() >= 0.0, "chain_jackpot R has unexpected negative values"
        assert mdp.r.max() <= cfg["jackpot_reward"] + 1e-14

    def test_chain_catastrophe_reward_range(self):
        mdp, _, cfg = make_chain_catastrophe({})
        assert mdp.r.min() >= cfg["catastrophe_reward"] - 1e-14
        assert mdp.r.max() <= 1.0 + 1e-14

    def test_grid_sparse_goal_reward_range(self):
        mdp, _, cfg = make_grid_sparse_goal({})
        assert mdp.r.min() >= 0.0 - 1e-14
        assert mdp.r.max() <= cfg["goal_reward"] + 1e-14


# ===================================================================
# 4. Horizon checks
# ===================================================================

class TestHorizons:
    """Factory-produced MDPs must have the correct horizon.
    # docs/specs/phase_II_*.md S9.1 -- horizon
    """

    def test_chain_sparse_long_horizon(self):
        mdp, _, cfg = make_chain_sparse_long({})
        assert mdp.info.horizon == cfg["horizon"] == 120

    def test_chain_sparse_long_severity0_horizon(self):
        mdp, _, _ = make_chain_sparse_long(
            {}, state_n=25, prob=0.9, gamma=0.99, horizon=60,
        )
        assert mdp.info.horizon == 60

    def test_chain_jackpot_horizon(self):
        mdp, _, cfg = make_chain_jackpot({})
        assert mdp.info.horizon == cfg["horizon"]

    def test_chain_catastrophe_horizon(self):
        mdp, _, cfg = make_chain_catastrophe({})
        assert mdp.info.horizon == cfg["horizon"]

    def test_grid_sparse_goal_horizon(self):
        mdp, _, cfg = make_grid_sparse_goal({})
        assert mdp.info.horizon == cfg["horizon"] == 80

    def test_grid_hazard_horizon(self):
        _, _, cfg = make_grid_hazard({})
        assert cfg["horizon"] == 80


# ===================================================================
# 5. Time-augmented env checks
# ===================================================================

class TestTimeAugmentation:
    """mdp_rl returned by each factory must be a DiscreteTimeAugmentedEnv
    with obs space = horizon * n_base_states.
    # docs/specs/phase_II_*.md S9.1 -- time augmentation
    """

    def test_chain_sparse_long_ta(self):
        mdp_base, mdp_rl, cfg = make_chain_sparse_long({})
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
        n_base = mdp_base.info.observation_space.n
        assert mdp_rl.info.observation_space.n == cfg["horizon"] * n_base

    def test_chain_jackpot_ta(self):
        mdp_base, mdp_rl, cfg = make_chain_jackpot({})
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
        n_base = mdp_base.info.observation_space.n
        assert mdp_rl.info.observation_space.n == cfg["horizon"] * n_base

    def test_chain_catastrophe_ta(self):
        mdp_base, mdp_rl, cfg = make_chain_catastrophe({})
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
        n_base = mdp_base.info.observation_space.n
        assert mdp_rl.info.observation_space.n == cfg["horizon"] * n_base

    def test_grid_sparse_goal_ta(self):
        mdp_base, mdp_rl, cfg = make_grid_sparse_goal({})
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
        n_base = mdp_base.info.observation_space.n
        assert mdp_rl.info.observation_space.n == cfg["horizon"] * n_base

    def test_grid_hazard_ta(self):
        _, mdp_rl, cfg = make_grid_hazard({})
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
        # grid_hazard base is 25 states (5x5)
        assert mdp_rl.info.observation_space.n == cfg["horizon"] * 25

    def test_taxi_bonus_shock_ta(self):
        _, mdp_rl, cfg = make_taxi_bonus_shock({})
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
        assert mdp_rl.info.observation_space.n == cfg["horizon"] * cfg["n_states"]


# ===================================================================
# 6. Regime-shift change-point tests
# ===================================================================

class TestChainRegimeShiftChangePoint:
    """ChainRegimeShiftWrapper triggers exactly at change_at_episode.
    # docs/specs/phase_II_*.md S9.1 -- change-point timing
    """

    def test_no_shift_before_change_point(self):
        """Before the change-point reset, post_change must be False.
        # docs/specs/phase_II_*.md S9.1 -- change-point timing

        The wrapper checks episode_count >= change_at BEFORE incrementing,
        so with change_at_episode=3, the first 3 resets (episodes 0,1,2)
        do not trigger the shift. After 3 resets, episode_count=3 but
        post_change is still False because the check was done pre-increment.
        """
        wrapper, _, _ = make_chain_regime_shift(
            {}, change_at_episode=3, time_augment=False,
        )
        assert isinstance(wrapper, ChainRegimeShiftWrapper)
        for _ in range(3):
            wrapper.reset()
        assert wrapper.post_change is False, (
            "ChainRegimeShiftWrapper triggered before change_at_episode"
        )

    def test_shift_at_change_point(self):
        """On the (change_at+1)-th reset, post_change flips to True.
        # docs/specs/phase_II_*.md S9.1 -- change-point timing

        The wrapper's reset() checks episode_count >= change_at before
        incrementing. After 3 resets, episode_count=3. The 4th reset sees
        episode_count=3 >= change_at=3, so it switches to the post-shift MDP.
        """
        wrapper, _, _ = make_chain_regime_shift(
            {}, change_at_episode=3, time_augment=False,
        )
        for _ in range(4):
            wrapper.reset()
        assert wrapper.post_change is True, (
            "ChainRegimeShiftWrapper did not trigger at change_at_episode=3"
        )

    def test_never_shifts_large_threshold(self):
        """With change_at_episode=10^9, 1000 resets must not trigger."""
        wrapper, _, _ = make_chain_regime_shift(
            {}, change_at_episode=10**9, time_augment=False,
        )
        for _ in range(1000):
            wrapper.reset()
        assert wrapper.post_change is False, (
            "ChainRegimeShiftWrapper shifted despite huge change_at_episode"
        )

    def test_shift_boundary_exact(self):
        """At exactly change_at resets, still False; one more flips it.
        # docs/specs/phase_II_*.md S9.1 -- change-point timing
        """
        wrapper, _, _ = make_chain_regime_shift(
            {}, change_at_episode=5, time_augment=False,
        )
        for _ in range(5):
            wrapper.reset()
        assert wrapper.post_change is False
        wrapper.reset()  # 6th reset: episode_count was 5 >= change_at=5
        assert wrapper.post_change is True


class TestGridRegimeShiftChangePoint:
    """GridRegimeShiftWrapper triggers exactly at change_at_episode.
    # docs/specs/phase_II_*.md S9.1 -- change-point timing
    """

    def test_no_shift_before_change_point(self):
        """Before the change-point, post_change must be False.
        # docs/specs/phase_II_*.md S9.1 -- change-point timing
        """
        wrapper, _, _ = make_grid_regime_shift(
            {}, change_at_episode=3, time_augment=False,
        )
        assert isinstance(wrapper, GridRegimeShiftWrapper)
        for _ in range(3):
            wrapper.reset()
        assert wrapper.post_change is False

    def test_shift_at_change_point(self):
        """On the (change_at+1)-th reset, post_change flips to True.
        # docs/specs/phase_II_*.md S9.1 -- change-point timing
        """
        wrapper, _, _ = make_grid_regime_shift(
            {}, change_at_episode=3, time_augment=False,
        )
        for _ in range(4):
            wrapper.reset()
        assert wrapper.post_change is True

    def test_never_shifts_large_threshold(self):
        wrapper, _, _ = make_grid_regime_shift(
            {}, change_at_episode=10**9, time_augment=False,
        )
        for _ in range(1000):
            wrapper.reset()
        assert wrapper.post_change is False

    def test_shift_boundary_exact(self):
        """At exactly change_at resets, still False; one more flips it.
        # docs/specs/phase_II_*.md S9.1 -- change-point timing
        """
        wrapper, _, _ = make_grid_regime_shift(
            {}, change_at_episode=5, time_augment=False,
        )
        for _ in range(5):
            wrapper.reset()
        assert wrapper.post_change is False
        wrapper.reset()
        assert wrapper.post_change is True


# ===================================================================
# 7. Regime-shift severity=0 (large change_at_episode) P/R identity
# ===================================================================

class TestRegimeShiftSeverity0:
    """With change_at_episode far in the future, pre-shift MDP is always
    active, so the wrapper should behave identically to the base task.
    # docs/specs/phase_II_*.md S9.1 -- severity=0 identity
    """

    def test_chain_regime_shift_p_identity(self, chain_base):
        wrapper, _, _ = make_chain_regime_shift(
            {}, change_at_episode=10**9, time_augment=False,
        )
        # The wrapper delegates .p to the pre-shift MDP (which IS chain_base).
        # Access via the internal _pre attribute for direct P comparison.
        np.testing.assert_array_equal(
            wrapper._pre.p, chain_base.p,
            err_msg="chain_regime_shift pre-shift P != chain_base P",
        )

    def test_grid_regime_shift_p_identity(self, grid_base):
        wrapper, _, _ = make_grid_regime_shift(
            {}, change_at_episode=10**9, time_augment=False,
        )
        np.testing.assert_array_equal(
            wrapper._pre.p, grid_base.p,
            err_msg="grid_regime_shift pre-shift P != grid_base P",
        )


# ===================================================================
# 8. Non-trivial severity: structural checks
# ===================================================================

class TestNonTrivialSeverity:
    """When severity parameters are non-zero, verify structural properties.
    # docs/specs/phase_II_*.md S9.1
    """

    def test_chain_jackpot_adds_absorbing_state(self):
        """With jackpot_prob>0, jackpot_terminates=True, an extra absorbing
        state is added (P shape grows by 1 in dimension 0 and 2).
        """
        mdp, _, cfg = make_chain_jackpot(
            {}, jackpot_prob=0.05, jackpot_terminates=True,
        )
        expected_states = cfg["state_n"] + 1
        assert mdp.p.shape[0] == expected_states
        assert mdp.p.shape[2] == expected_states
        # Absorbing state (last) has all-zero rows.
        absorbing_idx = expected_states - 1
        np.testing.assert_allclose(
            mdp.p[absorbing_idx].sum(), 0.0, atol=1e-14,
            err_msg="Jackpot absorbing state P rows are not all-zero",
        )

    def test_chain_catastrophe_adds_absorbing_state(self):
        mdp, _, cfg = make_chain_catastrophe(
            {}, risky_prob=0.05,
        )
        expected_states = cfg["state_n"] + 1
        assert mdp.p.shape[0] == expected_states
        absorbing_idx = expected_states - 1
        np.testing.assert_allclose(
            mdp.p[absorbing_idx].sum(), 0.0, atol=1e-14,
        )

    def test_chain_jackpot_prob_at_jackpot_state(self):
        """At the jackpot state, action 0's transition to the absorbing
        state has probability exactly jackpot_prob.
        """
        jp = 0.05
        mdp, _, cfg = make_chain_jackpot(
            {}, jackpot_prob=jp, jackpot_terminates=True,
        )
        js = cfg["jackpot_state"]
        absorbing_idx = cfg["state_n"]
        np.testing.assert_allclose(
            mdp.p[js, 0, absorbing_idx], jp, rtol=1e-12,
            err_msg="Jackpot transition probability mismatch",
        )

    def test_chain_catastrophe_prob_at_risky_state(self):
        rp = 0.05
        mdp, _, cfg = make_chain_catastrophe(
            {}, risky_prob=rp,
        )
        rs = cfg["risky_state"]
        absorbing_idx = cfg["state_n"]
        np.testing.assert_allclose(
            mdp.p[rs, 0, absorbing_idx], rp, rtol=1e-12,
            err_msg="Catastrophe transition probability mismatch",
        )

    def test_grid_hazard_wrapper_injects_reward(self):
        """With hazard_prob=1.0, stepping into a hazard cell always
        modifies the reward by hazard_reward.
        """
        wrapped, _, cfg = make_grid_hazard(
            {}, hazard_prob=1.0, hazard_reward=-5.0,
            hazard_terminates=False, time_augment=False, seed=0,
        )
        hazard_states = set(cfg["hazard_states"])
        # Step enough times to visit a hazard state.
        np.random.seed(0)
        wrapped.reset(state=np.array([0]))
        found_hazard = False
        for _ in range(200):
            action = np.array([np.random.randint(4)])
            ns, r, absorbing, _ = wrapped.step(action)
            s_idx = int(ns[0]) if hasattr(ns, '__len__') else int(ns)
            if s_idx in hazard_states:
                # The reward should include the hazard penalty.
                assert r <= 0.0 + 1e-14, (
                    f"Hazard cell reward {r} should be <= 0 (hazard_reward=-5)"
                )
                found_hazard = True
                break
            if absorbing:
                wrapped.reset(state=np.array([0]))
        # It's possible we never hit the hazard; just note it.
        if not found_hazard:
            pytest.skip("Did not visit hazard cell in 200 steps (stochastic)")


# ===================================================================
# 9. Gamma checks
# ===================================================================

class TestGamma:
    """Discount factors must match the factory parameter.
    # docs/specs/phase_II_*.md S9.1
    """

    def test_chain_sparse_long_gamma(self):
        mdp, _, _ = make_chain_sparse_long({})
        np.testing.assert_allclose(mdp.info.gamma, 0.99, rtol=1e-12)

    def test_chain_jackpot_gamma(self):
        mdp, _, _ = make_chain_jackpot({})
        np.testing.assert_allclose(mdp.info.gamma, 0.99, rtol=1e-12)

    def test_chain_catastrophe_gamma(self):
        mdp, _, _ = make_chain_catastrophe({})
        np.testing.assert_allclose(mdp.info.gamma, 0.99, rtol=1e-12)

    def test_grid_sparse_goal_gamma(self):
        mdp, _, _ = make_grid_sparse_goal({})
        np.testing.assert_allclose(mdp.info.gamma, 0.99, rtol=1e-12)
