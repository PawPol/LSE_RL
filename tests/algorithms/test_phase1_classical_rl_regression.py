"""Phase I classical RL regression tests (spec S8.3).

These tests verify that QLearning and ExpectedSARSA learn non-trivial Q
tables on a small wrapped FiniteMDP (chain_base), that TransitionLogger
produces structurally correct payloads, and that RLEvaluator.summary()
returns the expected keys.

Invariants guarded
------------------
- QLearning Q table changes from zero initialization after training
  (guards: the training loop actually updates the table).
- ExpectedSARSA Q table changes from zero initialization after training
  (guards: same).
- TransitionLogger.build_payload() returns all 13 TRANSITIONS_ARRAYS keys
  (guards: schema completeness of the callback payload).
- Each payload array is 1-D with matching length (guards: shape contract).
- margin_beta0 = reward - v_next_beta0 (no gamma) (guards: margin formula).
- RLEvaluator.summary() returns all four required keys (guards: summary
  contract for downstream aggregation).
- A short smoke run completes without error (guards: end-to-end wiring).
"""

from __future__ import annotations

import os
import sys
import pathlib
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path setup so we can import from mushroom-rl-dev, src, experiments
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_paths = [
    str(_REPO_ROOT / "mushroom-rl-dev"),
    str(_REPO_ROOT / "src"),
    str(_REPO_ROOT / "experiments"),
    str(_REPO_ROOT),
]
for p in _paths:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports after path setup
# ---------------------------------------------------------------------------
from mushroom_rl.algorithms.value.td import QLearning, ExpectedSARSA
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.core import Core

from experiments.weighted_lse_dp.common.task_factories import make_chain_base
from experiments.weighted_lse_dp.common.callbacks import (
    TransitionLogger,
    RLEvaluator,
)
from experiments.weighted_lse_dp.common.schemas import TRANSITIONS_ARRAYS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain_env():
    """Create the chain_base task (time-augmented) with fixed seed.

    Returns (mdp_base, mdp_rl, config).
    """
    # docs/specs/phase_I_*.md S5.1.A
    mdp_base, mdp_rl, cfg, _ref_pi = make_chain_base(seed=11)
    return mdp_base, mdp_rl, cfg


def _make_agent(mdp_rl, algo_cls, epsilon=0.3, learning_rate=0.1):
    """Helper: build a QLearning or ExpectedSARSA agent on the RL env."""
    policy = EpsGreedy(Parameter(value=epsilon))
    agent = algo_cls(mdp_rl.info, policy, learning_rate=Parameter(value=learning_rate))
    return agent


# ---------------------------------------------------------------------------
# QLearning regression tests
# ---------------------------------------------------------------------------

class TestQLearningRegression:
    """Regression tests for QLearning on chain_base."""

    def test_q_values_change_after_training(self, chain_env):
        """Q values must differ from initial (all-zeros) after learning.

        # docs/specs/phase_I_*.md S8.3 -- QLearning learns a non-trivial Q
        Invariant: after 5000 training steps on chain_base, the Q table
        is no longer all zeros (i.e., at least one update happened).
        5000 steps chosen because with ε=0.3 and a symmetric random walk
        on chain_base (n=25, goal at state 24), the expected hitting time
        from state 0 is ~576 steps; 5000 steps gives ~8 full expected
        hitting times across ~83 episodes.
        """
        _mdp_base, mdp_rl, cfg = chain_env
        agent = _make_agent(mdp_rl, QLearning)

        # Verify initial Q is all zeros (Table default initial_value=0.)
        assert np.all(agent.Q.table == 0.0), "Q table should start at zero"

        core = Core(agent, mdp_rl)
        core.learn(n_steps=5000, n_steps_per_fit=1)

        # After training, Q should have changed
        assert not np.all(agent.Q.table == 0.0), (
            "After 5000 steps of QLearning, Q table must not be all zeros"
        )

    def test_transition_logger_payload_structure(self, chain_env):
        """TransitionLogger.build_payload() must have all 13 TRANSITIONS_ARRAYS keys.

        # docs/specs/phase_I_*.md S8.3 -- TransitionLogger produces correct structure
        Invariant: the payload dict contains exactly the 13 keys defined in
        TRANSITIONS_ARRAYS, each mapping to a 1-D numpy array of the same length.
        """
        _mdp_base, mdp_rl, cfg = chain_env
        agent = _make_agent(mdp_rl, QLearning)

        n_base = cfg["state_n"]  # 25
        gamma = cfg["gamma"]     # 0.99
        logger = TransitionLogger(agent, n_base=n_base, gamma=gamma)

        core = Core(agent, mdp_rl, callback_step=logger)
        core.learn(n_steps=200, n_steps_per_fit=1)

        payload = logger.build_payload()

        # All 13 keys must be present
        for key in TRANSITIONS_ARRAYS:
            assert key in payload, f"Missing key '{key}' in payload"

        # Each value must be a 1-D numpy array
        lengths = set()
        for key in TRANSITIONS_ARRAYS:
            arr = payload[key]
            assert isinstance(arr, np.ndarray), (
                f"payload['{key}'] must be np.ndarray, got {type(arr)}"
            )
            assert arr.ndim == 1, (
                f"payload['{key}'] must be 1-D, got shape {arr.shape}"
            )
            lengths.add(arr.shape[0])

        # All arrays must have the same length
        assert len(lengths) == 1, (
            f"All payload arrays must have equal length, got {lengths}"
        )

    def test_transition_logger_captures_transitions(self, chain_env):
        """After 100 steps, logger should have >0 transitions.

        # docs/specs/phase_I_*.md S8.3 -- TransitionLogger captures transitions
        Invariant: n_transitions == number of env steps taken.
        """
        _mdp_base, mdp_rl, cfg = chain_env
        agent = _make_agent(mdp_rl, QLearning)

        n_base = cfg["state_n"]
        gamma = cfg["gamma"]
        logger = TransitionLogger(agent, n_base=n_base, gamma=gamma)

        core = Core(agent, mdp_rl, callback_step=logger)
        core.learn(n_steps=100, n_steps_per_fit=1)

        assert logger.n_transitions > 0, (
            "TransitionLogger must capture transitions during training"
        )
        assert logger.n_transitions == 100, (
            f"Expected 100 transitions after 100 steps, got {logger.n_transitions}"
        )

    def test_margin_formula_no_gamma(self, chain_env):
        """margin_beta0 = reward - v_next_beta0 (no gamma).

        # docs/specs/phase_I_*.md S7.1 -- margin formula
        Invariant: for every row, margin_beta0 == reward - v_next_beta0
        exactly (no discount factor).
        """
        _mdp_base, mdp_rl, cfg = chain_env
        agent = _make_agent(mdp_rl, QLearning)

        n_base = cfg["state_n"]
        gamma = cfg["gamma"]
        logger = TransitionLogger(agent, n_base=n_base, gamma=gamma)

        core = Core(agent, mdp_rl, callback_step=logger)
        core.learn(n_steps=100, n_steps_per_fit=1)

        payload = logger.build_payload()

        expected_margin = payload["reward"] - payload["v_next_beta0"]
        np.testing.assert_allclose(
            payload["margin_beta0"],
            expected_margin,
            rtol=0,
            atol=1e-15,
            err_msg="margin_beta0 must equal reward - v_next_beta0 (no gamma)",
        )


# ---------------------------------------------------------------------------
# ExpectedSARSA regression tests
# ---------------------------------------------------------------------------

class TestExpectedSARSARegression:
    """Regression tests for ExpectedSARSA on chain_base."""

    def test_q_values_change_after_training(self, chain_env):
        """Q values must differ from initial (all-zeros) after learning.

        # docs/specs/phase_I_*.md S8.3 -- ExpectedSARSA learns a non-trivial Q
        Invariant: after 5000 training steps on chain_base, the Q table
        is no longer all zeros.
        5000 steps chosen for the same reason as the QLearning test above
        (expected hitting time to goal ~576 steps; 5000 provides robust margin).
        """
        _mdp_base, mdp_rl, cfg = chain_env
        agent = _make_agent(mdp_rl, ExpectedSARSA)

        assert np.all(agent.Q.table == 0.0), "Q table should start at zero"

        core = Core(agent, mdp_rl)
        core.learn(n_steps=5000, n_steps_per_fit=1)

        assert not np.all(agent.Q.table == 0.0), (
            "After 5000 steps of ExpectedSARSA, Q table must not be all zeros"
        )


# ---------------------------------------------------------------------------
# RLEvaluator regression tests
# ---------------------------------------------------------------------------

class _DummyRunWriter:
    """Minimal stub for RunWriter to avoid file I/O in tests."""

    def __init__(self):
        self.checkpoints = []

    def record_rl_checkpoint(self, **kwargs):
        self.checkpoints.append(kwargs)


class TestRLEvaluatorRegression:
    """Regression tests for RLEvaluator on chain_base."""

    def test_evaluator_summary_keys(self, chain_env):
        """RLEvaluator.summary() must have all four required keys.

        # docs/specs/phase_I_*.md S9.3 -- summary stat keys
        Invariant: summary dict has keys steps_to_threshold,
        auc_disc_return, final_10pct_disc_return, final_10pct_success_rate.
        """
        _mdp_base, mdp_rl, cfg = chain_env
        agent = _make_agent(mdp_rl, QLearning, epsilon=0.1)

        # Train a bit first so evaluation is not on a blank agent
        core = Core(agent, mdp_rl)
        core.learn(n_steps=200, n_steps_per_fit=1)

        rw = _DummyRunWriter()
        evaluator = RLEvaluator(
            agent=agent,
            env=mdp_rl,
            run_writer=rw,
            n_eval_episodes=5,
            success_threshold=0.90,
            gamma=cfg["gamma"],
        )

        # Record a few checkpoints so summary() has data
        evaluator.evaluate(steps=100)
        evaluator.evaluate(steps=200)
        evaluator.evaluate(steps=300)

        summary = evaluator.summary()

        expected_keys = {
            "steps_to_threshold",
            "auc_disc_return",
            "final_10pct_disc_return",
            "final_10pct_success_rate",
        }
        assert set(summary.keys()) == expected_keys, (
            f"summary keys mismatch: got {set(summary.keys())}, "
            f"expected {expected_keys}"
        )

    def test_evaluator_evaluate_returns_dict(self, chain_env):
        """RLEvaluator.evaluate() must return a dict with disc_return_mean, success_rate.

        # docs/specs/phase_I_*.md S9.3
        """
        _mdp_base, mdp_rl, cfg = chain_env
        agent = _make_agent(mdp_rl, QLearning, epsilon=0.1)

        rw = _DummyRunWriter()
        evaluator = RLEvaluator(
            agent=agent,
            env=mdp_rl,
            run_writer=rw,
            n_eval_episodes=3,
            success_threshold=0.90,
            gamma=cfg["gamma"],
        )

        result = evaluator.evaluate(steps=0)
        assert "disc_return_mean" in result
        assert "success_rate" in result
        assert isinstance(result["disc_return_mean"], float)
        assert isinstance(result["success_rate"], float)


# ---------------------------------------------------------------------------
# Smoke test: end-to-end short run
# ---------------------------------------------------------------------------

class TestSmokeRun:
    """End-to-end smoke test: short training + evaluation completes."""

    def test_smoke_qlearning_with_callbacks(self, chain_env):
        """A short QLearning run with TransitionLogger + RLEvaluator completes.

        # docs/specs/phase_I_*.md S8.3 -- smoke run
        Invariant: no exceptions raised; payload and summary are valid.
        """
        _mdp_base, mdp_rl, cfg = chain_env
        agent = _make_agent(mdp_rl, QLearning, epsilon=0.3, learning_rate=0.1)

        n_base = cfg["state_n"]
        gamma = cfg["gamma"]
        logger = TransitionLogger(agent, n_base=n_base, gamma=gamma)

        core = Core(agent, mdp_rl, callback_step=logger)
        core.learn(n_steps=300, n_steps_per_fit=1)

        # Payload must be valid
        payload = logger.build_payload()
        assert len(payload) >= 13

        # Evaluator must work
        rw = _DummyRunWriter()
        evaluator = RLEvaluator(
            agent=agent,
            env=mdp_rl,
            run_writer=rw,
            n_eval_episodes=3,
            success_threshold=0.90,
            gamma=gamma,
        )
        evaluator.evaluate(steps=300)
        summary = evaluator.summary()
        assert isinstance(summary, dict)
        assert len(summary) == 4
