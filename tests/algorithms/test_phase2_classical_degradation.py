"""Phase II classical degradation sanity tests.

Verify that each stress-task mechanism produces measurably different
behavior from its Phase I base task. These are SHORT tests -- no
thousands of episodes -- designed to catch the case where a stress
wrapper is accidentally transparent.

Spec reference: docs/specs/phase_II_stress_test_beta0_experiments.md S9.3

Invariants guarded
------------------
- chain_jackpot (jackpot_prob=1.0) produces heavier right-tail returns
  than chain_base.
- chain_catastrophe (risky_prob=1.0) produces catastrophic episodes
  (absorbing at terminal with large negative reward).
- grid_regime_shift causes a measurable post-change drop in return
  when using the pre-shift optimal policy after the shift.
- grid_hazard (hazard_prob=1.0) injects hazard_reward on entry to a
  hazard cell, differing from the base grid step reward.
- taxi_bonus_shock (bonus_prob=1.0) adds positive bonus rewards on
  delivery, exceeding base taxi returns.

Source bugs observed (NOT fixed -- tests work around them)
----------------------------------------------------------
- ``make_chain_jackpot`` and ``make_chain_catastrophe`` pass ``mu=None``
  to FiniteMDP when ``jackpot_prob > 0`` / ``risky_prob > 0``, causing
  uniform sampling over *all* states including the absorbing terminal.
  Stepping from the absorbing state crashes (``probabilities do not sum
  to 1``) because its P row is all zeros. Workaround: force initial
  state via ``reset(np.array([0]))``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path setup so imports resolve regardless of working directory
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
for _p in [str(_REPO), str(_REPO / "src"), str(_REPO / "experiments")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os as _os

_os.chdir(_REPO)

from experiments.weighted_lse_dp.tasks.stress_families import (
    make_chain_jackpot,
    make_chain_catastrophe,
    make_taxi_bonus_shock,
    TaxiBonusShockWrapper,
)
from experiments.weighted_lse_dp.tasks.nonstationary_wrappers import (
    make_chain_regime_shift,
    make_grid_regime_shift,
)
from experiments.weighted_lse_dp.tasks.hazard_wrappers import (
    GridHazardWrapper,
    make_grid_hazard,
)
from experiments.weighted_lse_dp.common.task_factories import (
    make_chain_base,
    make_grid_base,
    make_taxi_base,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_episodes_finite_mdp(
    mdp, action_fn, n_episodes, horizon, seed=42, force_start=None
):
    """Run ``n_episodes`` on a FiniteMDP (or wrapper), return per-episode returns.

    Args:
        mdp: environment with ``reset`` / ``step`` interface.
        action_fn: callable(state) -> int, returning the action index.
        n_episodes: how many episodes to run.
        horizon: max steps per episode.
        seed: numpy legacy RNG seed for reproducibility.
        force_start: if not None, pass this array to ``reset(state=...)``.
            Workaround for MDPs with broken mu (e.g. mu=None including
            absorbing states).

    Returns:
        List of floats -- undiscounted cumulative reward per episode.
    """
    np.random.seed(seed)
    returns = []
    for _ in range(n_episodes):
        if force_start is not None:
            state, _ = mdp.reset(force_start)
        else:
            state, _ = mdp.reset()
        ep_return = 0.0
        for _t in range(horizon):
            a = action_fn(state)
            state, reward, absorbing, _ = mdp.step(np.array([a]))
            ep_return += float(reward)
            if absorbing:
                break
        returns.append(ep_return)
    return returns


def _always_action(a):
    """Return a policy function that always picks action ``a``."""
    def _policy(state):
        return a
    return _policy


# ---------------------------------------------------------------------------
# Test 1: chain_jackpot produces heavier right-tail than chain_base
# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3 example 1
# ---------------------------------------------------------------------------


class TestChainJackpotHeavierTail:
    """chain_jackpot with jackpot_prob=1.0 must produce higher returns
    than chain_base under the same policy. When every visit to the
    jackpot state triggers a reward of +10, the 95th-percentile return
    must exceed the base chain's 95th-percentile return.

    Invariant: the jackpot mechanism is NOT transparent at severity > 0.
    """

    def test_jackpot_prob1_exceeds_base_95th_percentile(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        n_episodes = 50
        horizon = 60

        # Base chain (always go right = action 0).
        mdp_base, _, _, _ = make_chain_base(time_augment=False, seed=42)
        returns_base = _run_episodes_finite_mdp(
            mdp_base, _always_action(0), n_episodes, horizon, seed=42
        )

        # Jackpot chain with guaranteed jackpot (prob=1.0, reward=10).
        # NOTE: force_start=np.array([0]) works around the mu=None source bug.
        mdp_jack, _, _ = make_chain_jackpot(
            cfg={},
            jackpot_prob=1.0,
            jackpot_reward=10.0,
            jackpot_state=20,
            jackpot_terminates=True,
        )
        returns_jack = _run_episodes_finite_mdp(
            mdp_jack, _always_action(0), n_episodes, horizon, seed=42,
            force_start=np.array([0]),
        )

        p95_base = np.percentile(returns_base, 95)
        p95_jack = np.percentile(returns_jack, 95)

        assert p95_jack > p95_base, (
            f"Jackpot 95th percentile ({p95_jack:.4f}) should exceed "
            f"base 95th percentile ({p95_base:.4f})"
        )

    def test_jackpot_mean_exceeds_base_mean(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        n_episodes = 50
        horizon = 60

        mdp_base, _, _, _ = make_chain_base(time_augment=False, seed=42)
        returns_base = _run_episodes_finite_mdp(
            mdp_base, _always_action(0), n_episodes, horizon, seed=42
        )

        mdp_jack, _, _ = make_chain_jackpot(
            cfg={},
            jackpot_prob=1.0,
            jackpot_reward=10.0,
            jackpot_state=20,
            jackpot_terminates=True,
        )
        returns_jack = _run_episodes_finite_mdp(
            mdp_jack, _always_action(0), n_episodes, horizon, seed=42,
            force_start=np.array([0]),
        )

        assert np.mean(returns_jack) > np.mean(returns_base), (
            f"Jackpot mean return ({np.mean(returns_jack):.4f}) should "
            f"exceed base mean return ({np.mean(returns_base):.4f})"
        )


# ---------------------------------------------------------------------------
# Test 2: chain_catastrophe produces catastrophic episodes
# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3 example 2
# ---------------------------------------------------------------------------


class TestChainCatastropheNonZeroRate:
    """chain_catastrophe with risky_prob=1.0 must produce at least one
    episode that terminates early with large negative reward.

    Invariant: the catastrophe mechanism fires and is not invisible.
    """

    def test_catastrophe_prob1_produces_negative_returns(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        n_episodes = 20
        horizon = 60

        mdp_cat, _, _ = make_chain_catastrophe(
            cfg={},
            risky_prob=1.0,
            catastrophe_reward=-10.0,
            risky_state=15,
        )
        # NOTE: force_start works around mu=None source bug.
        returns = _run_episodes_finite_mdp(
            mdp_cat, _always_action(0), n_episodes, horizon, seed=42,
            force_start=np.array([0]),
        )

        # At least one episode should have a negative return due to the
        # catastrophe reward of -10.
        negative_count = sum(1 for r in returns if r < 0)
        assert negative_count >= 1, (
            f"Expected at least 1 episode with negative return when "
            f"risky_prob=1.0, but got {negative_count} out of "
            f"{n_episodes} episodes. Returns: {returns}"
        )

    def test_catastrophe_absorbs_episode(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        # Directly step into the risky state and confirm absorption.
        mdp_cat, _, _ = make_chain_catastrophe(
            cfg={},
            risky_prob=1.0,
            catastrophe_reward=-10.0,
            risky_state=15,
        )
        np.random.seed(42)

        # Place agent at risky_state and take action 0 (forward).
        mdp_cat.reset(np.array([15]))
        next_state, reward, absorbing, _ = mdp_cat.step(np.array([0]))

        # With risky_prob=1.0, the agent must transition to the absorbing
        # catastrophe state and receive the catastrophe reward.
        assert absorbing, (
            "Expected episode to be absorbing after catastrophe at "
            f"risky_state=15 with risky_prob=1.0, but got absorbing=False. "
            f"next_state={next_state}, reward={reward}"
        )
        np.testing.assert_allclose(
            reward, -10.0, atol=1e-12,
            err_msg="Catastrophe reward should be -10.0"
        )


# ---------------------------------------------------------------------------
# Test 3: grid_regime_shift causes post-change performance drop
# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3 example 3
# ---------------------------------------------------------------------------


class TestGridRegimeShiftDrop:
    """After a regime shift (goal_move), continuing to use the pre-shift
    policy must produce lower mean return than before the shift.

    Invariant: the regime-shift wrapper actually changes the MDP dynamics.
    """

    def test_post_shift_return_drops(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        # The wrapper checks ``episode_count >= change_at`` at reset, then
        # increments. So ``change_at=n_pre`` means the shift occurs on the
        # (n_pre + 1)-th reset (i.e., the first post-shift episode).
        n_pre = 5
        n_post = 5
        horizon = 80

        wrapper, _, resolved = make_grid_regime_shift(
            cfg={},
            change_at_episode=n_pre,
            shift_type="goal_move",
            time_augment=False,
        )

        # Policy that navigates toward (4,4) via RIGHT then DOWN.
        # This path does NOT pass through (4,0) -- the post-shift goal --
        # so after the shift the agent will fail to reach the new goal.
        # Path: (0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4)->...->( 4,4).
        def _right_then_down(state):
            s = int(state[0]) if hasattr(state, '__len__') else int(state)
            row, col = divmod(s, 5)
            if col < 4:
                return 3  # right
            elif row < 4:
                return 1  # down
            else:
                return 0  # at (4,4), any action

        np.random.seed(42)

        # Pre-shift episodes.
        pre_returns = []
        for _ in range(n_pre):
            state, _ = wrapper.reset()
            ep_ret = 0.0
            for _t in range(horizon):
                a = _right_then_down(state)
                state, reward, absorbing, _ = wrapper.step(np.array([a]))
                ep_ret += float(reward)
                if absorbing:
                    break
            pre_returns.append(ep_ret)

        assert not wrapper.post_change, (
            "Shift should NOT have occurred yet after exactly n_pre resets"
        )

        # Post-shift: goal is now at (4,0) but the policy still targets (4,4).
        # Since the right-then-down path never visits column 0 on the
        # bottom row, the agent will NOT find the new goal.
        post_returns = []
        for _ in range(n_post):
            state, _ = wrapper.reset()
            ep_ret = 0.0
            for _t in range(horizon):
                a = _right_then_down(state)
                state, reward, absorbing, _ = wrapper.step(np.array([a]))
                ep_ret += float(reward)
                if absorbing:
                    break
            post_returns.append(ep_ret)

        assert wrapper.post_change, (
            "Expected regime shift to have occurred during post episodes"
        )

        mean_pre = np.mean(pre_returns)
        mean_post = np.mean(post_returns)

        assert mean_post < mean_pre, (
            f"Post-shift mean return ({mean_post:.4f}) should be less than "
            f"pre-shift mean return ({mean_pre:.4f}) when using the "
            f"pre-shift policy after a goal_move regime shift."
        )


# ---------------------------------------------------------------------------
# Test 4: grid_hazard wrapper fires with hazard_prob=1.0
# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3 example 4
# ---------------------------------------------------------------------------


class TestGridHazardFires:
    """GridHazardWrapper with hazard_prob=1.0 must inject hazard_reward
    when the agent enters a hazard cell. The base MDP at the same state
    must NOT produce the same reward.

    Invariant: the hazard wrapper modifies rewards at hazard cells.
    """

    def test_hazard_reward_injected_at_hazard_state(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        hazard_state = 12  # center of 5x5 grid = (2,2)
        hazard_reward = -5.0

        wrapper, _, _ = make_grid_hazard(
            cfg={},
            hazard_states=[hazard_state],
            hazard_prob=1.0,
            hazard_reward=hazard_reward,
            hazard_terminates=False,  # do not terminate so we can observe reward
            time_augment=False,
            seed=42,
        )

        # Also build the base grid for comparison.
        mdp_base, _, _, _ = make_grid_base(time_augment=False, seed=42)

        # Place agent at state 7 (row=1, col=2) and go down (action 1).
        # With prob=0.9 this lands on state 12. Retry with different seeds.
        hazard_seen = False
        base_reward_at_12 = None
        wrapper_reward_at_12 = None

        for trial_seed in range(100):
            np.random.seed(trial_seed)
            mdp_base.reset(np.array([7]))
            ns_base, rew_base, _, _ = mdp_base.step(np.array([1]))
            if int(ns_base[0]) == hazard_state:
                base_reward_at_12 = float(rew_base)

                np.random.seed(trial_seed)
                wrapper.reset(np.array([7]))
                ns_wrap, rew_wrap, _, _ = wrapper.step(np.array([1]))
                if int(ns_wrap[0]) == hazard_state:
                    wrapper_reward_at_12 = float(rew_wrap)
                    hazard_seen = True
                    break

        assert hazard_seen, (
            "Could not transition into hazard state 12 after 100 trials."
        )
        assert wrapper_reward_at_12 is not None
        assert base_reward_at_12 is not None

        # The wrapper reward must include the hazard penalty.
        expected_wrapper_reward = base_reward_at_12 + hazard_reward
        np.testing.assert_allclose(
            wrapper_reward_at_12, expected_wrapper_reward, atol=1e-12,
            err_msg=(
                f"Hazard wrapper reward at state 12 should be "
                f"base_reward ({base_reward_at_12}) + hazard_reward "
                f"({hazard_reward}) = {expected_wrapper_reward}, "
                f"got {wrapper_reward_at_12}"
            ),
        )

    def test_hazard_differs_from_base(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        # Run short episodes using a policy that passes through state 12.
        # State 12 = (2,2). Policy: right, right, down*4, right*2 pattern.
        # Simpler: go diagonally toward (4,4) via the center.
        n_episodes = 20
        horizon = 80

        def _diagonal_policy(state):
            """Navigate toward (4,4) through the center (2,2)."""
            s = int(state[0]) if hasattr(state, '__len__') else int(state)
            row, col = divmod(s, 5)
            # Alternate right and down to traverse diagonally.
            if col < 4 and row <= col:
                return 1  # down
            elif col < 4:
                return 3  # right
            elif row < 4:
                return 1  # down
            else:
                return 0  # at goal

        mdp_base, _, _, _ = make_grid_base(time_augment=False, seed=42)
        returns_base = _run_episodes_finite_mdp(
            mdp_base, _diagonal_policy, n_episodes, horizon, seed=42
        )

        wrapper, _, _ = make_grid_hazard(
            cfg={},
            hazard_states=[12],
            hazard_prob=1.0,
            hazard_reward=-5.0,
            hazard_terminates=True,
            time_augment=False,
            seed=42,
        )
        returns_hazard = _run_episodes_finite_mdp(
            wrapper, _diagonal_policy, n_episodes, horizon, seed=42
        )

        # With hazard at state 12 on the diagonal path, hazard returns
        # should differ from base. The hazard terminates episodes early
        # with a -5 penalty, so mean return should be lower.
        assert np.mean(returns_hazard) < np.mean(returns_base) or \
               not np.allclose(returns_hazard, returns_base), (
            f"Hazard returns should differ from base returns. "
            f"Hazard mean={np.mean(returns_hazard):.4f}, "
            f"Base mean={np.mean(returns_base):.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: taxi_bonus_shock adds positive bonuses
# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3 example 5
# ---------------------------------------------------------------------------


class TestTaxiBonusShock:
    """TaxiBonusShockWrapper with bonus_prob=1.0 must produce higher
    returns than the base taxi MDP on episodes where deliveries occur.

    Invariant: the bonus mechanism fires and is not invisible.
    """

    def test_bonus_prob1_increases_delivery_reward(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        mdp_base, _, _, ref_pi = make_taxi_base(time_augment=False, seed=42)

        bonus_reward = 5.0

        def _ref_policy(state):
            s = int(state[0]) if hasattr(state, '__len__') else int(state)
            return int(ref_pi[0, s])

        n_episodes = 20
        horizon = 120

        returns_base = _run_episodes_finite_mdp(
            mdp_base, _ref_policy, n_episodes, horizon, seed=42
        )

        # Re-create wrapper (fresh RNG state).
        wrapper = TaxiBonusShockWrapper(
            base_mdp=mdp_base,
            bonus_prob=1.0,
            bonus_reward=bonus_reward,
            rng_seed=42,
        )
        returns_bonus = _run_episodes_finite_mdp(
            wrapper, _ref_policy, n_episodes, horizon, seed=42
        )

        total_base = sum(returns_base)
        total_bonus = sum(returns_bonus)

        assert total_bonus > total_base, (
            f"Bonus total return ({total_bonus:.4f}) should exceed "
            f"base total return ({total_base:.4f}) when bonus_prob=1.0"
        )

    def test_bonus_prob0_is_transparent(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        mdp_base, _, _, ref_pi = make_taxi_base(time_augment=False, seed=42)

        wrapper = TaxiBonusShockWrapper(
            base_mdp=mdp_base,
            bonus_prob=0.0,
            bonus_reward=5.0,
            rng_seed=42,
        )

        def _ref_policy(state):
            s = int(state[0]) if hasattr(state, '__len__') else int(state)
            return int(ref_pi[0, s])

        n_episodes = 10
        horizon = 120

        returns_base = _run_episodes_finite_mdp(
            mdp_base, _ref_policy, n_episodes, horizon, seed=42
        )
        returns_wrapped = _run_episodes_finite_mdp(
            wrapper, _ref_policy, n_episodes, horizon, seed=42
        )

        np.testing.assert_allclose(
            returns_wrapped, returns_base, atol=1e-12,
            err_msg="bonus_prob=0.0 wrapper should be transparent"
        )


# ---------------------------------------------------------------------------
# Test 6: chain regime shift causes drop (chain variant)
# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3
# ---------------------------------------------------------------------------


class TestChainRegimeShiftDrop:
    """After a chain regime shift (goal_flip), continuing to use the
    pre-shift always-right policy must produce lower mean return than
    before the shift (since the goal moves to state 0, and the agent
    is still going right).

    Invariant: chain regime-shift wrapper changes behavior.
    """

    def test_chain_goal_flip_drops_return(self):
        """# docs/specs/phase_II_stress_test_beta0_experiments.md S9.3"""
        # The wrapper checks ``episode_count >= change_at`` at reset, then
        # increments. So ``change_at=n_pre`` means the shift occurs on the
        # (n_pre + 1)-th reset (i.e., the first post-shift episode).
        n_pre = 5
        n_post = 10
        horizon = 60

        wrapper, _, resolved = make_chain_regime_shift(
            cfg={},
            change_at_episode=n_pre,
            shift_type="goal_flip",
            time_augment=False,
        )

        np.random.seed(42)

        # Pre-shift: always go right (action 0) toward goal at state 24.
        pre_returns = []
        for _ in range(n_pre):
            state, _ = wrapper.reset()
            ep_ret = 0.0
            for _t in range(horizon):
                state, reward, absorbing, _ = wrapper.step(np.array([0]))
                ep_ret += float(reward)
                if absorbing:
                    break
            pre_returns.append(ep_ret)

        assert not wrapper.post_change, (
            "Shift should NOT have occurred yet after exactly n_pre resets"
        )

        # Post-shift: same always-right policy, but goal is now at state 0.
        post_returns = []
        for _ in range(n_post):
            state, _ = wrapper.reset()
            ep_ret = 0.0
            for _t in range(horizon):
                state, reward, absorbing, _ = wrapper.step(np.array([0]))
                ep_ret += float(reward)
                if absorbing:
                    break
            post_returns.append(ep_ret)

        assert wrapper.post_change, (
            "Expected regime shift to have occurred during post episodes"
        )

        mean_pre = np.mean(pre_returns)
        mean_post = np.mean(post_returns)

        assert mean_post < mean_pre, (
            f"Post-shift mean return ({mean_post:.4f}) should be less than "
            f"pre-shift mean return ({mean_pre:.4f}) after goal_flip. "
            f"The always-right policy should fail when the goal is at state 0."
        )
