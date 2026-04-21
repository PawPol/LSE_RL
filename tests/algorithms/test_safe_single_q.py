"""
Phase IV-C §A2: SafeSingleQLearning unit tests.

Verifies the architectural-control baseline:
- β=0 collapse to classical Q-Learning
- Update is applied only to the accessed (state, stage) cell
- Stage-indexed beta is used correctly
- Log dict contains expected keys
- Interface parity with SafeDoubleQLearning (get_Q, get_V, T property)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
_SRC = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _MUSHROOM_DEV, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lse_rl.algorithms.safe_single_q import SafeSingleQLearning


def _make_schedule(T: int = 5, beta: float = 0.0, gamma: float = 0.95):
    """Minimal valid BetaSchedule for testing."""
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule
    beta_cap = 10.0  # large — no clipping in tests
    return BetaSchedule({
        "gamma": gamma,
        "sign": 1 if beta >= 0 else -1,
        "task_family": "test",
        "alpha_t": [0.05] * T,
        "kappa_t": [0.9] * T,
        "Bhat_t": [0.0] * (T + 1),
        "beta_raw_t": [beta] * T,
        "beta_cap_t": [beta_cap] * T,
        "beta_used_t": [beta] * T,
    })


class TestSafeSingleQLearning:
    def test_beta0_collapse_to_classical(self):
        """β=0: safe target == r + γ * v_next (classical Q-Learning)."""
        gamma = 0.95
        sched = _make_schedule(T=3, beta=0.0, gamma=gamma)
        agent = SafeSingleQLearning(n_states=4, n_actions=2, schedule=sched,
                                    learning_rate=1.0, gamma=gamma, seed=0)
        # Set Q[1, 2, :] = [0.5, 0.3] so v_next = 0.5
        agent.Q[1, 2, 0] = 0.5
        agent.Q[1, 2, 1] = 0.3

        r = 1.0
        log = agent.update(state=0, action=0, reward=r, next_state=2,
                           absorbing=False, stage=1)

        expected_target = r + gamma * 0.5
        np.testing.assert_allclose(log["target"], expected_target, rtol=1e-9)
        # lr=1 → Q[1,0,0] = expected_target
        np.testing.assert_allclose(agent.Q[1, 0, 0], expected_target, rtol=1e-9)

    def test_absorbing_zero_bootstrap(self):
        """Absorbing transitions bootstrap with v_next = 0."""
        sched = _make_schedule(T=3, beta=0.0, gamma=0.95)
        agent = SafeSingleQLearning(n_states=4, n_actions=2, schedule=sched,
                                    learning_rate=1.0, gamma=0.95, seed=0)
        # Give next_state nonzero Q to ensure it's not used
        agent.Q[2, 1, 0] = 99.0

        log = agent.update(state=0, action=0, reward=2.0, next_state=1,
                           absorbing=True, stage=2)
        np.testing.assert_allclose(log["v_next"], 0.0)
        np.testing.assert_allclose(log["target"], 2.0, rtol=1e-9)

    def test_update_touches_only_sa_cell(self):
        """Update modifies Q[stage, state, action] and no other cell."""
        sched = _make_schedule(T=4, beta=0.0, gamma=0.9)
        agent = SafeSingleQLearning(n_states=5, n_actions=3, schedule=sched,
                                    learning_rate=0.5, gamma=0.9, seed=0)
        Q_before = agent.Q.copy()

        agent.update(state=2, action=1, reward=0.5, next_state=3,
                     absorbing=False, stage=0)

        # Only cell (stage=0, state=2, action=1) should change
        mask = np.ones_like(Q_before, dtype=bool)
        mask[0, 2, 1] = False
        np.testing.assert_array_equal(agent.Q[mask], Q_before[mask])

    def test_log_keys_present(self):
        """Log dict contains all required keys."""
        sched = _make_schedule(T=3, beta=0.05, gamma=0.95)
        agent = SafeSingleQLearning(n_states=4, n_actions=2, schedule=sched,
                                    learning_rate=0.1, gamma=0.95, seed=0)
        log = agent.update(state=0, action=0, reward=1.0, next_state=1,
                           absorbing=False, stage=0)
        required = {"beta_used", "rho", "effective_discount", "margin",
                    "natural_shift", "td_error", "target", "clip_active",
                    "stage", "v_next"}
        assert required.issubset(log.keys()), \
            f"Missing keys: {required - log.keys()}"

    def test_get_Q_returns_correct_row(self):
        """get_Q(state, stage) returns Q[stage, state, :]."""
        sched = _make_schedule(T=3, beta=0.0, gamma=0.9)
        agent = SafeSingleQLearning(n_states=4, n_actions=2, schedule=sched,
                                    learning_rate=0.1, gamma=0.9, seed=0)
        agent.Q[2, 1, :] = [3.0, 7.0]
        np.testing.assert_array_equal(agent.get_Q(1, 2), [3.0, 7.0])

    def test_get_V_is_max_Q(self):
        """get_V(state, stage) == max(Q[stage, state, :])."""
        sched = _make_schedule(T=3, beta=0.0, gamma=0.9)
        agent = SafeSingleQLearning(n_states=4, n_actions=2, schedule=sched,
                                    learning_rate=0.1, gamma=0.9, seed=0)
        agent.Q[0, 0, :] = [1.0, 5.0]
        assert agent.get_V(0, 0) == pytest.approx(5.0)

    def test_T_property(self):
        sched = _make_schedule(T=7, beta=0.0, gamma=0.9)
        agent = SafeSingleQLearning(n_states=3, n_actions=2, schedule=sched,
                                    learning_rate=0.1, gamma=0.9, seed=0)
        assert agent.T == 7

    def test_gamma_mismatch_raises(self):
        sched = _make_schedule(T=3, beta=0.0, gamma=0.9)
        with pytest.raises(ValueError, match="gamma"):
            SafeSingleQLearning(n_states=3, n_actions=2, schedule=sched,
                                learning_rate=0.1, gamma=0.95, seed=0)

    def test_stage_out_of_range_raises(self):
        sched = _make_schedule(T=3, beta=0.0, gamma=0.9)
        agent = SafeSingleQLearning(n_states=4, n_actions=2, schedule=sched,
                                    learning_rate=0.1, gamma=0.9, seed=0)
        with pytest.raises(IndexError):
            agent.update(state=0, action=0, reward=1.0, next_state=1,
                         absorbing=False, stage=5)

    def test_safe_operator_nonzero_beta(self):
        """With β > 0, safe target differs from classical."""
        gamma = 0.9
        sched_classic = _make_schedule(T=3, beta=0.0, gamma=gamma)
        sched_safe = _make_schedule(T=3, beta=0.5, gamma=gamma)
        agent_c = SafeSingleQLearning(n_states=4, n_actions=2, schedule=sched_classic,
                                      learning_rate=1.0, gamma=gamma, seed=0)
        agent_s = SafeSingleQLearning(n_states=4, n_actions=2, schedule=sched_safe,
                                      learning_rate=1.0, gamma=gamma, seed=0)
        # Fix v_next for both
        agent_c.Q[0, 1, 0] = 0.4
        agent_s.Q[0, 1, 0] = 0.4

        log_c = agent_c.update(state=0, action=0, reward=1.0, next_state=1,
                                absorbing=False, stage=0)
        log_s = agent_s.update(state=0, action=0, reward=1.0, next_state=1,
                                absorbing=False, stage=0)
        # Targets must differ
        assert log_c["target"] != pytest.approx(log_s["target"], abs=1e-6)
