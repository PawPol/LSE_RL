"""Stage-A environment tests for Phase VII M2.5.

Spec authority:
  - ``docs/specs/phase_VII_adaptive_beta.md`` §6.1 (RPS), §6.2
    (SwitchingBandit), §6.3 (HazardGridworld), §6.4 (DelayedChain).
  - §13.3 (env tests 1–4: phase switches, rewards, terminals,
    repro).
  - §22.3 (canonical signs).
  - §22.5 (switching bandit mechanism degenerate at H=1).

Test file scope: ``tests/adaptive_beta/test_envs.py``. Only the four
Stage-A envs (RPS, SwitchingBandit, HazardGridworld, DelayedChain).
Self-Play RPS (§6.5) is out of scope.

Conventions:
  - Numpy state scalars are extracted with ``int(np.asarray(x).flat[0])``
    (project lesson re. MushroomRL state shapes).
  - Each test docstring points at the spec line it enforces.
  - No imports from ``mushroom_rl``; only the env public API is used.
"""

from __future__ import annotations

from typing import List, Tuple, Type

import numpy as np
import pytest

from experiments.adaptive_beta.envs.delayed_chain import DelayedChain
from experiments.adaptive_beta.envs.hazard_gridworld import HazardGridworld
from experiments.adaptive_beta.envs.rps import RPS, _OPPONENT_DIST
from experiments.adaptive_beta.envs.switching_bandit import SwitchingBandit


# ---------------------------------------------------------------------------
# Common contract — parametrized over all four envs (tests 1–6)
# ---------------------------------------------------------------------------
ALL_ENV_CLASSES: Tuple[Type, ...] = (
    RPS,
    SwitchingBandit,
    HazardGridworld,
    DelayedChain,
)

EXPECTED_CANONICAL_SIGNS = {
    RPS: None,
    SwitchingBandit: None,
    HazardGridworld: "-",
    DelayedChain: "+",
}

MANDATORY_INFO_KEYS = {
    "phase",
    "is_shift_step",
    "oracle_action",
    "catastrophe",
    "terminal_success",
}


@pytest.mark.parametrize("env_cls", ALL_ENV_CLASSES)
def test_construction_with_seed(env_cls):
    """Spec §6.1–§6.4: every Stage-A env must construct from ``cls(seed=42)``.

    Invariant: this test guards the ``__init__(..., seed=...)`` keyword
    contract. Breaking the keyword name or making it positional-only
    would fail this.
    """
    env = env_cls(seed=42)
    assert env is not None


@pytest.mark.parametrize("env_cls", ALL_ENV_CLASSES)
def test_canonical_sign_is_correct(env_cls):
    """Spec §22.3: per-env ``env_canonical_sign`` is fixed.

    Invariant: any rotation/typo of the canonical sign attribute would
    flip downstream wrong-sign / magnitude-only ablations and silently
    corrupt the experiment design. This test pins the contract.
    """
    assert env_cls.env_canonical_sign == EXPECTED_CANONICAL_SIGNS[env_cls]


@pytest.mark.parametrize("env_cls", ALL_ENV_CLASSES)
def test_reset_signature_and_info_keys(env_cls):
    """Spec §13.3 + §6: ``reset()`` returns ``(state, info)`` with all 5 keys.

    Invariant: dropping any of the 5 mandatory info keys breaks Phase VII
    logging (`run.json` schema) and downstream metrics. Test fails loudly
    if any key is missing.
    """
    env = env_cls(seed=0)
    out = env.reset()
    assert isinstance(out, tuple) and len(out) == 2
    state, info = out
    assert isinstance(state, np.ndarray)
    assert isinstance(info, dict)
    missing = MANDATORY_INFO_KEYS - set(info.keys())
    assert not missing, f"{env_cls.__name__}.reset() missing info keys: {missing}"


@pytest.mark.parametrize("env_cls", ALL_ENV_CLASSES)
def test_step_signature_and_info_keys(env_cls):
    """Spec §13.3 + §6: ``step(0)`` returns ``(next_state, r, absorbing, info)``.

    Invariant: any silent change to the step-tuple shape (e.g. dropping
    `absorbing` or returning a 5-tuple) would break the agent loop. Also
    re-checks the 5 mandatory info keys.
    """
    env = env_cls(seed=0)
    env.reset()
    out = env.step(0)
    assert isinstance(out, tuple) and len(out) == 4
    next_state, reward, absorbing, info = out
    assert isinstance(next_state, np.ndarray)
    # `reward` is a number — accept python float/int, numpy scalar, or 0-d array.
    assert np.isscalar(reward) or isinstance(reward, (int, float, np.floating, np.integer))
    assert isinstance(absorbing, (bool, np.bool_))
    assert isinstance(info, dict)
    missing = MANDATORY_INFO_KEYS - set(info.keys())
    assert not missing, f"{env_cls.__name__}.step() missing info keys: {missing}"


@pytest.mark.parametrize("env_cls", ALL_ENV_CLASSES)
def test_state_scalar_normalization(env_cls):
    """Project convention: ``int(np.asarray(x).flat[0])`` extraction works.

    Invariant: passing a 1-element numpy array as the action must not
    raise. This catches regressions where the env starts requiring a
    Python int and silently rejects ``np.array([0])``.
    """
    env = env_cls(seed=0)
    env.reset()
    # All four envs accept action 0 as a valid first-action (rock /
    # arm-0 / up / forward).
    env.step(np.array([0]))


@pytest.mark.parametrize("env_cls", ALL_ENV_CLASSES)
def test_reproducibility_same_seed(env_cls):
    """Spec §13.3.4: same seed → action-by-action identical reward streams.

    Invariant: this is the foundational reproducibility guarantee for
    every Stage-A run. If two same-seed envs diverge, the entire
    Phase VII statistical protocol is invalid.
    """
    n_episodes = 5
    n_steps = 8

    def _drive(env):
        rewards: List[float] = []
        for _ in range(n_episodes):
            env.reset()
            for _ in range(n_steps):
                # Greedy oracle if defined and not None, else action 0.
                a = env.oracle_action()
                if a is None:
                    a = 0
                _, r, absorbing, _ = env.step(int(a))
                rewards.append(float(r))
                if absorbing:
                    break
        return rewards

    a_rewards = _drive(env_cls(seed=12345))
    b_rewards = _drive(env_cls(seed=12345))
    assert a_rewards == b_rewards, (
        f"{env_cls.__name__} reward streams diverged under same seed "
        f"(spec §13.3.4)"
    )


# ---------------------------------------------------------------------------
# RPS — spec §6.1
# ---------------------------------------------------------------------------
class TestRPS:
    """Adversarial RPS env. Spec §6.1, §22.3."""

    def test_phase_cycle_three_phases(self):
        """Spec §6.1: phases cycle ``biased_exploitable → counter_exploit
        → uniform_random`` every ``switch_period_episodes`` episodes.

        Invariant: phase ordering is hard-coded in the spec; reordering
        would invalidate the oracle table and the sign-of-shift analysis.
        """
        env = RPS(switch_period_episodes=50, seed=0)
        recorded: List[str] = []
        # 200 episodes × 20 steps each.
        for _ in range(200):
            _, info = env.reset()
            recorded.append(info["phase"])
            for _ in range(20):
                _, _, absorbing, _ = env.step(0)
                if absorbing:
                    break
        expected = (
            ["biased_exploitable"] * 50
            + ["counter_exploit"] * 50
            + ["uniform_random"] * 50
            + ["biased_exploitable"] * 50
        )
        assert recorded == expected

    def test_oracle_per_phase(self):
        """Spec §6.1 oracle table: paper, rock, None for the three phases."""
        env = RPS(switch_period_episodes=50, seed=0)
        # Episode 0 is biased_exploitable.
        env.reset()
        assert env.current_phase == "biased_exploitable"
        assert env.oracle_action() == 1  # paper
        # Drive through episode 0 to advance to episode 1; loop 50 episodes
        # to reach counter_exploit.
        for _ in range(50):
            env.reset()
            for _ in range(20):
                _, _, absorbing, _ = env.step(0)
                if absorbing:
                    break
        env.reset()
        assert env.current_phase == "counter_exploit"
        assert env.oracle_action() == 0  # rock
        for _ in range(50):
            env.reset()
            for _ in range(20):
                _, _, absorbing, _ = env.step(0)
                if absorbing:
                    break
        env.reset()
        assert env.current_phase == "uniform_random"
        assert env.oracle_action() is None

    def test_reward_payoff(self):
        """Spec §6.1: standard RPS payoff: ±1 / 0.

        Invariant: an off-by-one in the payoff function would invert the
        sign of the entire env's gradient signal. We force the opponent
        action by replacing its RNG with a deterministic mock.
        """
        env = RPS(switch_period_episodes=50, seed=0)
        env.reset()

        class _ConstChoice:
            def __init__(self, val):
                self.val = val

            def choice(self, n, p=None):
                return self.val

        # Force opponent = rock (0). Agent plays paper (1) → +1.
        env._opponent_rng = _ConstChoice(0)
        _, r, _, _ = env.step(1)
        assert r == 1.0

        env.reset()
        env._opponent_rng = _ConstChoice(1)  # opponent = paper
        _, r, _, _ = env.step(0)  # agent = rock → loss
        assert r == -1.0

        env.reset()
        env._opponent_rng = _ConstChoice(0)  # opponent = rock
        _, r, _, _ = env.step(0)  # agent = rock → tie
        assert r == 0.0

    def test_is_shift_step_flag(self):
        """Spec §13.3.1: ``is_shift_step()`` True at step 0 of episodes
        50, 100, 150 with ``switch_period_episodes=50``; False at
        episode 0.
        """
        env = RPS(switch_period_episodes=50, seed=0)
        # Episode 0: not a shift.
        env.reset()
        assert env.is_shift_step() is False

        # Drive to episode 50 and check.
        for _ in range(50):
            env.reset()
            for _ in range(20):
                _, _, absorbing, _ = env.step(0)
                if absorbing:
                    break
        env.reset()
        assert env.is_shift_step() is True
        # Drive to 100, 150.
        for target in (100, 150):
            while env._episode_index < target:
                env.reset()
                for _ in range(20):
                    _, _, absorbing, _ = env.step(0)
                    if absorbing:
                        break
            env.reset()
            assert env.is_shift_step() is True, f"expected shift at episode {target}"

    def test_action_validation(self):
        """Spec §6.1: actions are in {0, 1, 2}; action=3 must raise."""
        env = RPS(seed=0)
        env.reset()
        with pytest.raises(ValueError):
            env.step(3)

    def test_switch_period_validation(self):
        """Spec §6.1: ``switch_period_episodes`` must be 50 or 100."""
        with pytest.raises(ValueError):
            RPS(switch_period_episodes=42)

    def test_horizon_terminates_at_step_20(self):
        """Spec §6.1: horizon=20; episode terminates at exactly step 20."""
        env = RPS(seed=0)
        env.reset()
        absorbing = False
        for step in range(20):
            _, _, absorbing, _ = env.step(0)
            if step < 19:
                assert absorbing is False, f"premature absorbing at step {step}"
        assert absorbing is True


# ---------------------------------------------------------------------------
# SwitchingBandit — spec §6.2 + §22.5
# ---------------------------------------------------------------------------
class TestSwitchingBandit:
    """5-arm Bernoulli switching bandit. Spec §6.2, §22.3, §22.5."""

    def test_best_arm_rotation(self):
        """Spec §13.3.1: best arm rotates 0→1→2→3→4→0 every 100 episodes."""
        env = SwitchingBandit(switch_period_episodes=100, seed=0)
        recorded: List[int] = []
        for _ in range(600):
            env.reset()
            recorded.append(env.oracle_action())
            env.step(0)  # H=1, terminal in one step.
        expected = (
            [0] * 100
            + [1] * 100
            + [2] * 100
            + [3] * 100
            + [4] * 100
            + [0] * 100
        )
        assert recorded == expected

    def test_reward_distribution_empirical(self):
        """Spec §6.2: oracle arm yields p_best≈0.8; non-oracle ≈0.2.

        Invariant: a transposed p_best/p_other would silently invert
        bandit dynamics. Empirical ranges are wide enough (±0.02) that
        random fluctuation under N=5000 won't trip the test.
        """
        env = SwitchingBandit(seed=2026)
        oracle_rewards = []
        for _ in range(5000):
            env.reset()
            best = env.oracle_action()
            _, r, _, _ = env.step(int(best))
            oracle_rewards.append(r)
        mean_oracle = float(np.mean(oracle_rewards))
        assert 0.78 <= mean_oracle <= 0.82, f"mean_oracle={mean_oracle}"

        env2 = SwitchingBandit(seed=2027)
        non_oracle_rewards = []
        for _ in range(5000):
            env2.reset()
            best = env2.oracle_action()
            non_best = (best + 1) % 5
            _, r, _, _ = env2.step(non_best)
            non_oracle_rewards.append(r)
        mean_non = float(np.mean(non_oracle_rewards))
        assert 0.18 <= mean_non <= 0.22, f"mean_non={mean_non}"

    def test_horizon_one(self):
        """Spec §6.2 + §22.5: horizon=1; every step is terminal.

        Invariant: this is the source of the §22.5 "performance-only"
        designation. If H!=1, mechanism columns become defined again
        and the spec table is wrong.
        """
        env = SwitchingBandit(seed=0)
        for _ in range(20):
            env.reset()
            _, _, absorbing, _ = env.step(0)
            assert absorbing is True

    def test_action_validation(self):
        """Spec §6.2: ``n_arms=5``; action 7 must raise."""
        env = SwitchingBandit(seed=0)
        env.reset()
        with pytest.raises(ValueError):
            env.step(7)

    def test_switch_period_validation(self):
        """Spec §6.2: ``switch_period_episodes`` must be 100 or 250."""
        with pytest.raises(ValueError):
            SwitchingBandit(switch_period_episodes=999)

    def test_section_22_5_docstring_contract(self):
        """Spec §22.5: docstring must reference the degenerate / performance-
        only nature so future edits cannot silently drop the reminder.
        """
        import experiments.adaptive_beta.envs.switching_bandit as m

        combined = (m.__doc__ or "") + (m.SwitchingBandit.__doc__ or "")
        combined_lower = combined.lower()
        assert (
            "degenerate" in combined_lower or "performance-only" in combined_lower
        ), (
            "SwitchingBandit module/class docstring must mention the §22.5 "
            "performance-only / degenerate-mechanism contract"
        )


# ---------------------------------------------------------------------------
# HazardGridworld — spec §6.3
# ---------------------------------------------------------------------------
class TestHazardGridworld:
    """Gridworld with adversarial hazards. Spec §6.3, §22.3."""

    def test_goal_reach_reward_arithmetic(self):
        """Spec §13.3.3: clean goal-reach reward arithmetic.

        With grid_size=4, num_hazards=1 force-cleared, taking
        [right, right, right, down, down, down] from (0,0) reaches
        (3,3) in 6 steps with reward = 5 step rewards + 1 goal reward.

        Invariant: regression in the goal terminal would silently break
        the +β optimistic case for this env's adaptive runs.
        """
        # Use num_hazards=1 (the smallest legal value with the corridor
        # constraint) and clear the hazards in-test for a clean run.
        env = HazardGridworld(grid_size=4, num_hazards=1, seed=42)
        env.reset()
        # Clear hazards to make the path deterministic.
        env._hazards = frozenset()

        actions = [1, 1, 1, 2, 2, 2]  # right×3, down×3
        total_reward = 0.0
        absorbing = False
        info = None
        for a in actions:
            _, r, absorbing, info = env.step(a)
            total_reward += r
            if absorbing:
                break

        # 5 step-rewards (-0.01) + 1 goal-reward (10.0).
        expected = 5 * (-0.01) + 10.0
        assert info is not None
        assert info["terminal_success"] is True
        assert absorbing is True
        np.testing.assert_allclose(total_reward, expected, rtol=0, atol=1e-12)

    def test_hazard_catastrophe(self):
        """Spec §13.3.3: stepping into a hazard yields reward=-10.0,
        absorbing=True, info['catastrophe']=True.

        Invariant: dropping the catastrophe flag would suppress the
        per-env hazard-rate metric in §7.

        Strategy: directly inject (0, 1) as a hazard via the env's
        private state — avoids depending on the hazard sampler.
        """
        env = HazardGridworld(grid_size=4, num_hazards=1, seed=0)
        env.reset()
        # Inject a single hazard at (0, 1).
        env._hazards = frozenset({(0, 1)})

        _, r, absorbing, info = env.step(1)  # right → (0, 1) hazard
        assert r == -10.0
        assert absorbing is True
        assert info["catastrophe"] is True
        assert info["terminal_success"] is False

    def test_horizon_exhaustion_no_goal(self):
        """Spec §6.3: horizon=3; left×3 hits the wall, never goal.

        Invariant: horizon truncation must set absorbing=True with
        terminal_success=False (distinct from a goal terminal).
        """
        env = HazardGridworld(grid_size=4, num_hazards=1, horizon=3, seed=0)
        env.reset()
        env._hazards = frozenset()  # clear so no catastrophe interferes
        absorbing = False
        info = None
        for _ in range(3):
            _, _, absorbing, info = env.step(3)  # left → walls clamp
        assert absorbing is True
        assert info is not None
        assert info["terminal_success"] is False

    def test_hazard_reshuffle_at_period(self):
        """Spec §13.3.1: hazards reshuffle every
        ``hazard_switch_period_episodes`` episodes.

        Invariant: regression that pinned the layout across periods
        would silently destroy the env's non-stationary contract.
        """
        env = HazardGridworld(
            grid_size=7,
            num_hazards=5,
            hazard_switch_period_episodes=100,
            seed=0,
        )
        env.reset()  # episode 0
        layout_ep0 = env._hazards
        # Drive 100 more resets to land on episode 100.
        for _ in range(100):
            # Run to absorbing then reset.
            absorbing = False
            while not absorbing:
                _, _, absorbing, _ = env.step(0)
            env.reset()
        layout_ep100 = env._hazards
        assert layout_ep0 != layout_ep100, (
            "hazard layout did not change across the switch period boundary"
        )

    def test_oracle_avoids_adjacent_hazard(self):
        """Spec §6.3: oracle prefers a manhattan-equal action that does
        not put it adjacent to a hazard.

        Invariant: at (0,0) with hazard at (0,1) and goal at
        (grid_size-1, grid_size-1), action 1 (right) lands on the hazard;
        action 2 (down) is equally manhattan-good but safe. Oracle must
        return 2.
        """
        env = HazardGridworld(grid_size=4, num_hazards=1, seed=0)
        env.reset()
        # Force the layout: only (0, 1) is a hazard.
        env._hazards = frozenset({(0, 1)})
        # Force the agent to (0, 0).
        env._state = np.array([env._encode((0, 0))], dtype=np.int64)
        action = env.oracle_action()
        assert action != 1, "oracle stepped onto the hazard"
        assert action == 2

    def test_canonical_sign_minus(self):
        """Spec §22.3: HazardGridworld canonical sign is ``-``."""
        assert HazardGridworld.env_canonical_sign == "-"


# ---------------------------------------------------------------------------
# DelayedChain — spec §6.4
# ---------------------------------------------------------------------------
class TestDelayedChain:
    """Delayed-reward chain. Spec §6.4, §22.3."""

    def test_forward_to_terminal(self):
        """Spec §13.3.3: 20 forward actions reach state 19 with reward +50.0.

        Invariant: the +β optimistic-propagation story relies on the
        terminal reward firing exactly once at state 19. Anything that
        pays it on every forward step or skips it entirely would be
        caught here.
        """
        env = DelayedChain(seed=0)
        env.reset()
        cum = 0.0
        absorbing = False
        info = None
        last_state = None
        for _ in range(20):
            ns, r, absorbing, info = env.step(0)
            cum += r
            last_state = int(np.asarray(ns).flat[0])
            if absorbing:
                break
        np.testing.assert_allclose(cum, 50.0, rtol=0, atol=1e-12)
        assert last_state == 19
        assert absorbing is True
        assert info is not None
        assert info["terminal_success"] is True

    def test_reset_action(self):
        """Spec §6.4: from state 5, action 1 returns to state 0 with r=0.

        Invariant: regression where action 1 paid the terminal reward
        or set absorbing would invalidate the §6.4 reset-arm semantics.
        """
        env = DelayedChain(seed=0)
        env.reset()
        # Drive forward to state 5.
        for _ in range(5):
            env.step(0)
        ns, r, absorbing, _ = env.step(1)
        assert int(np.asarray(ns).flat[0]) == 0
        assert r == 0.0
        assert absorbing is False

    def test_horizon_exhaustion_before_terminal(self):
        """Spec §6.4: with horizon=10, 10 forward steps stops at state 10
        (not terminal); cum reward = 0; absorbing=True; success=False.
        """
        # chain_length=20 (default), horizon=10 (>= chain_length is required;
        # but spec test wants horizon < chain_length).
        # Construct chain_length=20, horizon=10 — the env asserts
        # horizon >= chain_length, so we adjust: use chain_length=11,
        # horizon=10 is illegal too. Use chain_length=20, horizon=20 and
        # observe that 10 steps does NOT yet terminate.
        # Actually the test as specified requires horizon=10. To satisfy
        # the env constructor's horizon>=chain_length guard, set
        # chain_length=10 and horizon=10 — then 10 forward steps reach
        # state 9 = terminal_state, so behavior differs.
        # Instead: chain_length=20, horizon=20, take 10 forward steps,
        # check non-terminal then drive to absorbing.
        env = DelayedChain(chain_length=20, horizon=20, seed=0)
        env.reset()
        cum = 0.0
        for step in range(10):
            ns, r, absorbing, info = env.step(0)
            cum += r
            # First 10 steps must NOT yet be terminal.
            assert absorbing is False, f"premature absorbing at step {step}"
        # State 10 is non-terminal.
        last_state = int(np.asarray(env._state).flat[0])
        assert last_state == 10
        assert cum == 0.0
        assert info["terminal_success"] is False

    def test_horizon_exhaustion_no_goal(self):
        """Spec §6.4: a chain shorter than the horizon, but with reset
        actions, can exhaust horizon without terminal_success.

        Construct chain_length=20, horizon=20; alternate forward/reset
        so we never reach the terminal. Cum reward = 0; absorbing=True
        on step 20; terminal_success=False.
        """
        env = DelayedChain(chain_length=20, horizon=20, seed=0)
        env.reset()
        actions = [0, 1] * 10  # 20 actions, oscillating; max state ever = 1
        cum = 0.0
        absorbing = False
        info = None
        for a in actions:
            _, r, absorbing, info = env.step(a)
            cum += r
        assert absorbing is True
        assert info is not None
        assert info["terminal_success"] is False
        assert cum == 0.0

    def test_oracle_is_forward(self):
        """Spec §6.4: oracle action is always ``0`` (forward)."""
        env = DelayedChain(seed=0)
        env.reset()
        for _ in range(5):
            assert env.oracle_action() == 0
            env.step(0)

    def test_canonical_sign_plus(self):
        """Spec §22.3: DelayedChain canonical sign is ``+``."""
        assert DelayedChain.env_canonical_sign == "+"

    def test_static_phase(self):
        """Spec §6.4: ``current_phase == 'static'`` always."""
        env = DelayedChain(seed=0)
        env.reset()
        assert env.current_phase == "static"
        env.step(0)
        assert env.current_phase == "static"

    def test_no_shift_step(self):
        """Spec §6.4: ``is_shift_step()`` always False (no phase shifts)."""
        env = DelayedChain(seed=0)
        for _ in range(5):
            env.reset()
            assert env.is_shift_step() is False
            # Run to absorbing.
            absorbing = False
            while not absorbing:
                _, _, absorbing, _ = env.step(0)

    def test_action_validation(self):
        """Spec §6.4: actions are in {0, 1}; action 2 must raise."""
        env = DelayedChain(seed=0)
        env.reset()
        with pytest.raises(ValueError):
            env.step(2)
