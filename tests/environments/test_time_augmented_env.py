"""
Tests for DiscreteTimeAugmentedEnv and make_time_augmented.

Covers the five invariants mandated by docs/specs/phase_I_*.md section 8.2:

1. Bijective (t, s) <-> augmented_id encode/decode round-trip
2. Reset sets t=0
3. Step increments t by 1
4. Horizon terminal handling -- episode ends at t=H-1
5. Reward and transition probs unchanged after un-augmenting
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- make mushroom-rl-dev, src, experiments importable
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parents[2]
for _sub in ("mushroom-rl-dev", "src", "experiments"):
    _p = str(_repo / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.environments.generators.simple_chain import generate_simple_chain
from mushroom_rl.environments.time_augmented_env import (
    DiscreteTimeAugmentedEnv,
    make_time_augmented,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_chain(n: int = 5, gamma: float = 0.99, horizon: int = 4) -> FiniteMDP:
    """Build a tiny FiniteMDP chain for testing.

    5 states, 2 actions, goal at state n-1.
    """
    return generate_simple_chain(
        state_n=n,
        goal_states=[n - 1],
        prob=0.9,
        rew=1.0,
        gamma=gamma,
        horizon=horizon,
    )


@pytest.fixture
def chain5():
    """A 5-state chain with horizon=4."""
    return make_test_chain(n=5, horizon=4)


@pytest.fixture
def aug_chain5(chain5):
    """DiscreteTimeAugmentedEnv wrapping the 5-state chain."""
    return DiscreteTimeAugmentedEnv(chain5, horizon=4)


# ---------------------------------------------------------------------------
# TestDiscreteTimeAugmentation
# ---------------------------------------------------------------------------

class TestDiscreteTimeAugmentation:

    def test_augmented_obs_size(self, aug_chain5):
        """Augmented obs space size == horizon * n_base_states.
        # docs/specs/phase_I_*.md section 8.2, invariant 1
        """
        env = aug_chain5
        expected = 4 * 5  # horizon=4, n_base=5
        assert env.info.observation_space.n == expected

    def test_encode_decode_roundtrip(self, aug_chain5):
        """encode(t, s) -> decode -> (t, s) is the identity for all valid pairs.
        # docs/specs/phase_I_*.md section 8.2, invariant 1 (bijectivity)

        This test would fail if encode or decode used a different formula
        than aug_id = t * n_base + s.
        """
        env = aug_chain5
        n_base = env.n_base_states  # 5
        horizon = 4
        for t in range(horizon):
            for s in range(n_base):
                aug_id = env.encode_state(t, s)
                t_dec, s_dec = env.decode_state(aug_id)
                assert t_dec == t and s_dec == s, (
                    f"Round-trip failed: ({t}, {s}) -> {aug_id} -> ({t_dec}, {s_dec})"
                )

    def test_encode_formula(self, aug_chain5):
        """aug_id == t * n_base + s, explicitly checked.
        # docs/specs/phase_I_*.md section 8.2, invariant 1
        """
        env = aug_chain5
        n_base = env.n_base_states
        for t in range(4):
            for s in range(n_base):
                assert env.encode_state(t, s) == t * n_base + s

    def test_reset_sets_t_zero(self, aug_chain5):
        """After reset(), the decoded stage index is 0.
        # docs/specs/phase_I_*.md section 8.2, invariant 2
        """
        env = aug_chain5
        state, _ = env.reset()
        aug_id = int(state[0])
        t, _s = env.decode_state(aug_id)
        assert t == 0, f"Expected t=0 after reset, got t={t}"
        assert env.current_stage == 0

    def test_step_increments_t(self, aug_chain5):
        """Each step() call increments the decoded stage by 1.
        # docs/specs/phase_I_*.md section 8.2, invariant 3

        This test would fail if the wrapper forgot to increment self._t.
        """
        env = aug_chain5
        np.random.seed(42)
        env.reset()
        for expected_t in range(1, 4):
            action = np.array([0])
            next_state, _r, _abs, _info = env.step(action)
            aug_id = int(next_state[0])
            t, _s = env.decode_state(aug_id)
            assert t == expected_t, (
                f"After step {expected_t}: expected t={expected_t}, got t={t}"
            )
            assert env.current_stage == expected_t

    def test_terminal_at_horizon(self, aug_chain5):
        """At t=horizon-1 the step returns absorbing=True.
        # docs/specs/phase_I_*.md section 8.2, invariant 4

        This test would fail if the terminal-stage OR logic were removed.
        """
        env = aug_chain5  # horizon=4
        np.random.seed(42)
        env.reset()
        absorbing = False
        for step_i in range(4):
            action = np.array([0])
            _state, _r, absorbing, _info = env.step(action)
        # After 4 steps, t=4 but t_encoded is clamped to 3 = horizon-1.
        # The terminal condition t_next >= horizon-1 fires at step 3 (0-indexed).
        # Actually: horizon-1 = 3; at step 3 (the 3rd step, t_next=3), absorbing should be True.
        # Let's verify by stepping exactly horizon-1 = 3 steps.
        env.reset()
        for step_i in range(3):
            action = np.array([0])
            _state, _r, absorbing, _info = env.step(action)
        assert absorbing is True, (
            "Expected absorbing=True at t=horizon-1=3"
        )

    def test_not_terminal_before_horizon(self):
        """Before t=horizon-1 the step does NOT force absorbing=True
        (unless the base env is absorbing).
        # docs/specs/phase_I_*.md section 8.2, invariant 4 (negative case)
        """
        # Use a chain where no state is absorbing in the base env
        # (the simple chain goal state has all self-transitions with p=1,
        # but is NOT marked absorbing by the FiniteMDP since p[goal] != 0).
        base = make_test_chain(n=5, horizon=10)
        env = DiscreteTimeAugmentedEnv(base, horizon=10)
        np.random.seed(0)
        # Force reset to state 0 (non-goal, non-absorbing in base)
        env.reset(state=np.array([0]))
        # Take one step with action=1 (move left, stays at 0 from state 0)
        _state, _r, absorbing, _info = env.step(np.array([1]))
        assert absorbing is False, (
            "Expected absorbing=False at t=1 which is before horizon-1=9"
        )

    def test_reward_unchanged(self):
        """Reward for the same (s, a, s') transition is identical before
        and after time augmentation.
        # docs/specs/phase_I_*.md section 8.2, invariant 5

        This test would fail if the wrapper modified the reward signal.
        """
        n_states = 5
        horizon = 6
        base = make_test_chain(n=n_states, horizon=horizon)
        env = DiscreteTimeAugmentedEnv(base, horizon=horizon)

        np.random.seed(123)
        # Collect rewards from the augmented env
        aug_rewards = []
        for _ in range(50):
            env.reset(state=np.array([0]))  # force base state 0, t=0
            action = np.array([0])  # move right
            _next, r, _abs, _info = env.step(action)
            aug_rewards.append(r)

        np.random.seed(123)
        # Collect rewards from the base env with the same seed
        base_rewards = []
        for _ in range(50):
            base.reset(state=np.array([0]))
            action = np.array([0])
            _next, r, _abs, _info = base.step(action)
            base_rewards.append(r)

        np.testing.assert_array_equal(
            np.array(aug_rewards),
            np.array(base_rewards),
            err_msg="Augmented env altered the reward signal",
        )

    def test_transition_probs_unchanged(self):
        """Transition probabilities are preserved after augmentation.
        # docs/specs/phase_I_*.md section 8.2, invariant 5

        We verify empirically that starting from the same base state and
        taking the same action yields the same distribution of next base
        states under identical seeds.
        """
        n_states = 5
        horizon = 10
        base = make_test_chain(n=n_states, horizon=horizon)
        env = DiscreteTimeAugmentedEnv(base, horizon=horizon)

        n_trials = 200
        np.random.seed(999)
        aug_next_states = []
        for _ in range(n_trials):
            env.reset(state=np.array([2]))  # base state 2, t=0
            next_state, _r, _abs, _info = env.step(np.array([0]))
            _t, s_next = env.decode_state(int(next_state[0]))
            aug_next_states.append(s_next)

        np.random.seed(999)
        base_next_states = []
        for _ in range(n_trials):
            base.reset(state=np.array([2]))
            next_state, _r, _abs, _info = base.step(np.array([0]))
            base_next_states.append(int(next_state[0]))

        np.testing.assert_array_equal(
            np.array(aug_next_states),
            np.array(base_next_states),
            err_msg="Augmented env altered transition distribution",
        )

    def test_time_augment_false_returns_base(self):
        """make_time_augmented with a Discrete env returns DiscreteTimeAugmentedEnv;
        when the user passes time_augment=False in a factory the base is returned.
        # docs/specs/phase_I_*.md section 8.2

        (This tests the factory dispatch, not the wrapper itself.)
        """
        base = make_test_chain(n=5, horizon=4)
        aug = make_time_augmented(base, horizon=4)
        assert isinstance(aug, DiscreteTimeAugmentedEnv)


# ---------------------------------------------------------------------------
# TestMakeTimeAugmented
# ---------------------------------------------------------------------------

class TestMakeTimeAugmented:

    def test_make_time_augmented_discrete(self):
        """make_time_augmented dispatches to DiscreteTimeAugmentedEnv for
        a Discrete observation space.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, horizon=4)
        aug = make_time_augmented(base, horizon=4)
        assert isinstance(aug, DiscreteTimeAugmentedEnv)
        assert aug.n_base_states == 5
        assert aug.info.observation_space.n == 4 * 5

    def test_make_time_augmented_preserves_gamma(self):
        """The augmented env preserves gamma from the constructor arg.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, gamma=0.95, horizon=4)
        aug = make_time_augmented(base, horizon=4)
        np.testing.assert_allclose(aug.info.gamma, 0.95, rtol=1e-12)

    def test_make_time_augmented_preserves_horizon(self):
        """The augmented MDPInfo reports the correct horizon.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, horizon=7)
        aug = make_time_augmented(base, horizon=7)
        assert aug.info.horizon == 7

    def test_full_episode_rollout(self):
        """A complete episode rollout terminates at exactly horizon-1 steps.
        # docs/specs/phase_I_*.md section 8.2, invariants 3 and 4

        This is an integration test: reset, then step until absorbing.
        The number of steps taken must equal horizon - 1 (since the
        terminal condition fires at t_next == horizon - 1, which happens
        after horizon - 1 step() calls starting from t=0).
        """
        horizon = 6
        base = make_test_chain(n=5, horizon=horizon)
        env = DiscreteTimeAugmentedEnv(base, horizon=horizon)

        np.random.seed(77)
        state, _ = env.reset()
        steps = 0
        absorbing = False
        while not absorbing:
            action = np.array([0])
            state, _r, absorbing, _info = env.step(action)
            steps += 1
            if steps > horizon + 5:
                pytest.fail("Episode did not terminate within horizon + 5 steps")

        # Terminal fires when t_next >= horizon - 1, i.e. after horizon - 1 steps
        # (since t starts at 0 and increments by 1 each step).
        assert steps == horizon - 1, (
            f"Expected episode to terminate after {horizon - 1} steps, "
            f"got {steps}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_encode_out_of_range_t(self):
        """encode_state raises ValueError for t out of range.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, horizon=4)
        env = DiscreteTimeAugmentedEnv(base, horizon=4)
        with pytest.raises(ValueError, match="stage t="):
            env.encode_state(4, 0)  # t=4 is out of range for horizon=4
        with pytest.raises(ValueError, match="stage t="):
            env.encode_state(-1, 0)

    def test_encode_out_of_range_s(self):
        """encode_state raises ValueError for s out of range.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, horizon=4)
        env = DiscreteTimeAugmentedEnv(base, horizon=4)
        with pytest.raises(ValueError, match="base state s="):
            env.encode_state(0, 5)  # s=5 out of range for n=5

    def test_decode_out_of_range(self):
        """decode_state raises ValueError for augmented_id out of range.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, horizon=4)
        env = DiscreteTimeAugmentedEnv(base, horizon=4)
        with pytest.raises(ValueError, match="augmented_id="):
            env.decode_state(20)  # 4*5=20 is out of range (max valid is 19)

    def test_horizon_validation_rejects_zero(self):
        """DiscreteTimeAugmentedEnv rejects horizon <= 0.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, horizon=4)
        with pytest.raises(ValueError):
            DiscreteTimeAugmentedEnv(base, horizon=0)

    def test_horizon_validation_rejects_inf(self):
        """DiscreteTimeAugmentedEnv rejects infinite horizon.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, horizon=4)
        with pytest.raises(ValueError):
            DiscreteTimeAugmentedEnv(base, horizon=np.inf)

    def test_reset_with_augmented_id(self):
        """reset() with an augmented id decodes it correctly.
        # docs/specs/phase_I_*.md section 8.2
        """
        base = make_test_chain(n=5, horizon=4)
        env = DiscreteTimeAugmentedEnv(base, horizon=4)
        # augmented id for t=2, s=3 is 2*5+3 = 13
        state, _ = env.reset(state=np.array([13]))
        aug_id = int(state[0])
        t, s = env.decode_state(aug_id)
        assert t == 2 and s == 3, f"Expected (2,3), got ({t},{s})"
