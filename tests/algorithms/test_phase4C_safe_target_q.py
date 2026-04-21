"""
Phase IV-C S3.2, S9: SafeTargetQLearning tests.

Verify target network update mechanics, safe target bootstrap from the
target network, Q-target gap logging, and beta=0 reduction to classical
Target Q-Learning.

Also covers SafeTargetExpectedSARSA (S3.3) since it shares the same
target-network machinery and only differs in the bootstrap expectation.
"""
from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------------------
# sys.path bootstrap so tests can find mushroom-rl-dev and src/lse_rl.
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

from mushroom_rl.algorithms.value.dp.safe_weighted_common import (  # noqa: E402
    BetaSchedule,
    SafeWeightedCommon,
    build_certification,
)
from lse_rl.algorithms import (  # noqa: E402
    SafeTargetExpectedSARSA,
    SafeTargetQLearning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_schedule(T: int = 5, gamma: float = 0.9) -> BetaSchedule:
    return BetaSchedule.zeros(T=T, gamma=gamma)


def _nonzero_schedule(T: int = 5, gamma: float = 0.9) -> BetaSchedule:
    alpha_t = np.full(T, 0.2, dtype=np.float64)
    cert = build_certification(alpha_t, R_max=1.0, gamma=gamma)
    beta_cap = cert["beta_cap_t"]
    beta_raw = 0.5 * beta_cap
    beta_used = np.clip(beta_raw, -beta_cap, beta_cap)
    schedule = {
        "gamma": gamma,
        "sign": 1,
        "task_family": "test_nonzero",
        "beta_raw_t": beta_raw.tolist(),
        "beta_cap_t": beta_cap.tolist(),
        "beta_used_t": beta_used.tolist(),
        "alpha_t": alpha_t.tolist(),
        "kappa_t": cert["kappa_t"].tolist(),
        "Bhat_t": cert["Bhat_t"].tolist(),
        "clip_active_t": [False] * T,
        "informativeness_t": [0.0] * T,
        "d_target_t": [gamma] * T,
        "reward_bound": 1.0,
        "source_phase": "test",
        "calibration_source_path": "",
        "calibration_hash": "",
        "lambda_min": 0.0,
        "lambda_max": 0.0,
        "margin_quantile": 0.0,
        "notes": "handcrafted non-zero schedule",
    }
    return BetaSchedule(schedule)


def _make_agent(
    schedule: BetaSchedule,
    *,
    n_states: int = 4,
    n_actions: int = 3,
    lr: float = 0.5,
    sync_every: int = 5,
    polyak_tau: float = 0.0,
    seed: int = 0,
) -> SafeTargetQLearning:
    return SafeTargetQLearning(
        n_states=n_states,
        n_actions=n_actions,
        schedule=schedule,
        learning_rate=lr,
        gamma=schedule.gamma,
        sync_every=sync_every,
        polyak_tau=polyak_tau,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# SafeTargetQLearning tests
# ---------------------------------------------------------------------------


def test_target_network_hard_sync() -> None:
    """Hard sync: Q_target copies Q_online at global_step == sync_every (but
    not before)."""
    agent = _make_agent(_zero_schedule(T=3), sync_every=3, seed=0)
    # Drive 2 updates (global_step = 1, 2): no sync yet.
    for gs in range(1, 3):
        agent.update(0, 0, 1.0, 1, False, stage=0, global_step=gs)
    # Q_online has moved; Q_target still zero.
    assert np.any(agent.Q_online != 0.0)
    assert np.allclose(agent.Q_target, 0.0)

    # 3rd update triggers sync.
    agent.update(0, 0, 1.0, 1, False, stage=0, global_step=3)
    np.testing.assert_allclose(agent.Q_target, agent.Q_online,
                               atol=1e-15, rtol=0)
    # Sync step is recorded.
    assert agent.update(0, 0, 0.0, 1, False, stage=0, global_step=4)[
        "target_sync_step"
    ] == 3


def test_polyak_averaging() -> None:
    """Polyak: target partially updates toward online each step."""
    tau = 0.3
    agent = _make_agent(
        _zero_schedule(T=2), polyak_tau=tau, sync_every=1, seed=0,
    )
    # Manually set Q_online and Q_target to known non-zero values so we
    # can verify the update formula.
    agent.Q_online[:] = 0.0
    agent.Q_target[:] = 0.0

    # First update writes to Q_online[0, 0, 0].  After the update,
    # _maybe_sync fires and averages toward online.
    log = agent.update(
        state=0, action=0, reward=1.0, next_state=1,
        absorbing=False, stage=0, global_step=1,
    )
    # Post-update Q_online[0,0,0] = lr * (safe_target - 0) = 0.5 * (1 + 0.9*0) = 0.5.
    assert agent.Q_online[0, 0, 0] == pytest.approx(0.5)
    # Q_target = (1-tau) * 0 + tau * Q_online (elementwise).
    expected_target = tau * agent.Q_online
    np.testing.assert_allclose(agent.Q_target, expected_target,
                               atol=1e-15, rtol=0)
    # Mode is "polyak".
    assert log["target_update_mode"] == "polyak"


def test_hard_sync_vs_polyak_mode_field() -> None:
    """target_update_mode reports the configured sync mode."""
    hard = _make_agent(_zero_schedule(T=2), polyak_tau=0.0, sync_every=5)
    polyak = _make_agent(_zero_schedule(T=2), polyak_tau=0.2)
    assert hard.target_update_mode == "hard"
    assert polyak.target_update_mode == "polyak"


def test_safe_target_uses_target_bootstrap() -> None:
    """safe_target == g_safe(r, max_a Q_target[t, s', a])."""
    schedule = _nonzero_schedule(T=3, gamma=0.9)
    agent = _make_agent(schedule, n_states=3, n_actions=3, sync_every=100,
                        seed=0)
    # Set Q_online and Q_target to *different* values so we can tell which
    # was used as bootstrap.
    agent.Q_online[:] = 0.0
    agent.Q_target[:] = 0.0
    agent.Q_online[1, 2, :] = [10.0, 20.0, 30.0]  # online max = 30.0
    agent.Q_target[1, 2, :] = [0.1, 0.2, 0.3]     # target max = 0.3

    swc_ref = SafeWeightedCommon(schedule=schedule, gamma=schedule.gamma,
                                 n_base=3)
    r = 0.0
    expected_safe = swc_ref.compute_safe_target(r, 0.3, 1)

    log = agent.update(
        state=0, action=0, reward=r, next_state=2,
        absorbing=False, stage=1, global_step=1,
    )

    assert log["q_target_next"] == pytest.approx(0.3)
    assert log["q_online_next"] == pytest.approx(30.0)
    assert log["safe_target"] == pytest.approx(expected_safe, rel=1e-12)


def test_q_target_gap_logged() -> None:
    agent = _make_agent(_zero_schedule(T=2), n_states=3, n_actions=2, seed=0)
    agent.Q_online[0, 1, :] = [5.0, 6.0]  # max = 6.0
    agent.Q_target[0, 1, :] = [1.0, 2.0]  # max = 2.0
    log = agent.update(0, 0, 0.0, 1, False, stage=0, global_step=1)
    assert log["q_target_gap"] == pytest.approx(abs(6.0 - 2.0))


def test_beta0_reduces_to_classical_target_q() -> None:
    """With beta=0 the safe target equals r + gamma * max Q_target[s',:]
    and the online update matches classical target-network Q.
    """
    rng = np.random.default_rng(0)
    T = 3
    n_states = 4
    n_actions = 3
    gamma = 0.9
    lr = 0.4
    sync_every = 7

    agent = SafeTargetQLearning(
        n_states=n_states, n_actions=n_actions,
        schedule=_zero_schedule(T=T, gamma=gamma),
        learning_rate=lr, gamma=gamma,
        sync_every=sync_every, polyak_tau=0.0, seed=0,
    )
    # Reference tables maintained manually.
    Q_online_ref = np.zeros_like(agent.Q_online)
    Q_target_ref = np.zeros_like(agent.Q_target)

    n_steps = 50
    for step in range(1, n_steps + 1):
        s = int(rng.integers(0, n_states))
        a = int(rng.integers(0, n_actions))
        sn = int(rng.integers(0, n_states))
        t = int(rng.integers(0, T))
        r = float(rng.normal())
        done = False

        v_next_ref = 0.0 if done else float(np.max(Q_target_ref[t, sn, :]))
        classical_target = r + gamma * v_next_ref
        q_cur = Q_online_ref[t, s, a]
        Q_online_ref[t, s, a] = q_cur + lr * (classical_target - q_cur)
        if step % sync_every == 0:
            np.copyto(Q_target_ref, Q_online_ref)

        log = agent.update(s, a, r, sn, done, stage=t, global_step=step)

        # beta=0 branch collapses to classical; targets match exactly.
        assert log["safe_target"] == pytest.approx(
            classical_target, rel=0, abs=1e-12
        )

    np.testing.assert_allclose(agent.Q_online, Q_online_ref,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(agent.Q_target, Q_target_ref,
                               atol=1e-12, rtol=0)


def test_absorbing_v_next_is_zero() -> None:
    agent = _make_agent(_zero_schedule(T=2), seed=0)
    agent.Q_online[:] = 0.7
    agent.Q_target[:] = 0.9
    log = agent.update(0, 0, 0.5, 1, absorbing=True, stage=0, global_step=1)
    assert log["q_target_next"] == 0.0
    assert log["q_online_next"] == 0.0
    assert log["q_target_gap"] == 0.0


def test_numpy_state_scalar_handling_target_q() -> None:
    """Guard against the numpy>=2.0 int(state) TypeError."""
    agent = _make_agent(_zero_schedule(T=2), seed=0)
    log = agent.update(
        state=np.array([0]),
        action=np.array([1]),
        reward=0.1,
        next_state=np.array([2]),
        absorbing=False,
        stage=0,
        global_step=1,
    )
    assert isinstance(log["safe_target"], float)


def test_stage_out_of_range_raises_target_q() -> None:
    agent = _make_agent(_zero_schedule(T=3), seed=0)
    with pytest.raises(IndexError):
        agent.update(0, 0, 0.0, 1, False, stage=3, global_step=1)


# ---------------------------------------------------------------------------
# SafeTargetExpectedSARSA tests
# ---------------------------------------------------------------------------


def test_expected_sarsa_uses_eps_greedy_expectation() -> None:
    """v_next = (1 - eps) * max Q_target + eps * mean Q_target."""
    schedule = _zero_schedule(T=2, gamma=0.9)
    agent = SafeTargetExpectedSARSA(
        n_states=3, n_actions=4, schedule=schedule,
        learning_rate=0.5, gamma=0.9, sync_every=100, polyak_tau=0.0,
        epsilon=0.25, seed=0,
    )
    agent.Q_target[0, 1, :] = [1.0, 2.0, 3.0, 4.0]
    # Expected v_next = 0.75 * 4.0 + 0.25 * 2.5 = 3.0 + 0.625 = 3.625.
    expected_v = 0.75 * 4.0 + 0.25 * 2.5
    log = agent.update(0, 0, 0.0, 1, False, stage=0, global_step=1)
    assert log["q_target_next"] == pytest.approx(expected_v, rel=1e-12)
    # With beta=0 the safe target equals r + gamma * v_next.
    assert log["safe_target"] == pytest.approx(0.9 * expected_v, rel=1e-12)


def test_expected_sarsa_beta0_matches_classical() -> None:
    """beta=0: online updates match classical target-ExpectedSARSA."""
    rng = np.random.default_rng(7)
    T = 2
    n_states = 3
    n_actions = 3
    gamma = 0.9
    lr = 0.3
    eps = 0.2
    sync_every = 4

    agent = SafeTargetExpectedSARSA(
        n_states=n_states, n_actions=n_actions,
        schedule=_zero_schedule(T=T, gamma=gamma),
        learning_rate=lr, gamma=gamma,
        sync_every=sync_every, polyak_tau=0.0,
        epsilon=eps, seed=0,
    )
    Q_online_ref = np.zeros_like(agent.Q_online)
    Q_target_ref = np.zeros_like(agent.Q_target)

    for step in range(1, 31):
        s = int(rng.integers(0, n_states))
        a = int(rng.integers(0, n_actions))
        sn = int(rng.integers(0, n_states))
        t = int(rng.integers(0, T))
        r = float(rng.normal())

        # Reference expected-value bootstrap.
        Q_slice = Q_target_ref[t, sn, :]
        v_ref = (1.0 - eps) * float(np.max(Q_slice)) + eps * float(np.mean(Q_slice))
        classical_target = r + gamma * v_ref
        q_cur = Q_online_ref[t, s, a]
        Q_online_ref[t, s, a] = q_cur + lr * (classical_target - q_cur)
        if step % sync_every == 0:
            np.copyto(Q_target_ref, Q_online_ref)

        log = agent.update(s, a, r, sn, False, stage=t, global_step=step)
        assert log["safe_target"] == pytest.approx(
            classical_target, rel=0, abs=1e-12
        )

    np.testing.assert_allclose(agent.Q_online, Q_online_ref,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(agent.Q_target, Q_target_ref,
                               atol=1e-12, rtol=0)


def test_expected_sarsa_epsilon_zero_matches_greedy() -> None:
    """With epsilon=0 the expected bootstrap equals greedy max."""
    schedule = _zero_schedule(T=2, gamma=0.9)
    exp_sarsa = SafeTargetExpectedSARSA(
        n_states=3, n_actions=3, schedule=schedule,
        learning_rate=0.5, gamma=0.9, sync_every=100,
        epsilon=0.0, seed=0,
    )
    target_q = SafeTargetQLearning(
        n_states=3, n_actions=3, schedule=schedule,
        learning_rate=0.5, gamma=0.9, sync_every=100, seed=0,
    )
    # Set identical target tables.
    exp_sarsa.Q_target[0, 1, :] = [1.0, 5.0, 2.0]
    target_q.Q_target[0, 1, :] = [1.0, 5.0, 2.0]

    log_es = exp_sarsa.update(0, 0, 0.5, 1, False, stage=0, global_step=1)
    log_q = target_q.update(0, 0, 0.5, 1, False, stage=0, global_step=1)

    assert log_es["q_target_next"] == pytest.approx(log_q["q_target_next"])
    assert log_es["safe_target"] == pytest.approx(log_q["safe_target"])
