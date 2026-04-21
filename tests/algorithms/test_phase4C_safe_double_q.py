"""
Phase IV-C S3.1, S9: SafeDoubleQLearning tests.

Verify dual Q-table maintenance, selection-evaluation separation, safe
target bootstrap usage, double-gap logging, and beta=0 reduction to
classical Double Q-Learning.
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
)
from lse_rl.algorithms import SafeDoubleQLearning  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_schedule(T: int = 5, gamma: float = 0.9) -> BetaSchedule:
    """Return a schedule with beta_used == 0 and alpha == 0 at every stage."""
    return BetaSchedule.zeros(T=T, gamma=gamma)


def _nonzero_schedule(T: int = 5, gamma: float = 0.9) -> BetaSchedule:
    """Return a schedule with non-zero beta and positive alpha headroom.

    We construct it by hand so we can set beta_used_t to whatever
    triggers the safe (non-classical) branch of ``compute_safe_target``.
    """
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
        build_certification,
    )

    alpha_t = np.full(T, 0.2, dtype=np.float64)
    cert = build_certification(alpha_t, R_max=1.0, gamma=gamma)
    # Choose beta_raw half of the certified cap -> always uncclipped.
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
    seed: int = 0,
) -> SafeDoubleQLearning:
    return SafeDoubleQLearning(
        n_states=n_states,
        n_actions=n_actions,
        schedule=schedule,
        learning_rate=lr,
        gamma=schedule.gamma,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dual_q_tables_maintained() -> None:
    """Two independent Q-tables are maintained; exactly one changes per update."""
    agent = _make_agent(_zero_schedule(T=3), seed=123)
    # Fix both tables with distinct values.
    agent.Q_A[:] = 0.1
    agent.Q_B[:] = 0.7

    before_A = agent.Q_A.copy()
    before_B = agent.Q_B.copy()

    log = agent.update(
        state=0, action=1, reward=1.0, next_state=2,
        absorbing=False, stage=1,
    )

    changed_A = not np.allclose(agent.Q_A, before_A)
    changed_B = not np.allclose(agent.Q_B, before_B)

    # Exactly one table updated.
    assert changed_A ^ changed_B, (
        f"Exactly one of Q_A/Q_B must change per update; "
        f"changed_A={changed_A}, changed_B={changed_B}."
    )

    # Log reports which side was the selector.
    if log["selected_action_source"] == "A":
        assert changed_A and not changed_B
        # And the updated cell is the one we acted on.
        assert agent.Q_A[1, 0, 1] != before_A[1, 0, 1]
    else:
        assert changed_B and not changed_A
        assert agent.Q_B[1, 0, 1] != before_B[1, 0, 1]


def test_selection_evaluation_separation() -> None:
    """Selection and evaluation sources are always different per update."""
    agent = _make_agent(_zero_schedule(T=4), seed=7)
    for step in range(50):
        log = agent.update(
            state=step % 4, action=step % 3, reward=0.5,
            next_state=(step + 1) % 4, absorbing=False,
            stage=step % 4,
        )
        assert log["selected_action_source"] != log["evaluation_value_source"]
        assert log["selected_action_source"] in {"A", "B"}
        assert log["evaluation_value_source"] in {"A", "B"}


def test_safe_target_uses_evaluation_bootstrap() -> None:
    """safe_target_double == g_safe(r, Q_eval[s', argmax Q_sel[s', ·]])."""
    schedule = _nonzero_schedule(T=3, gamma=0.9)
    agent = _make_agent(schedule, n_states=3, n_actions=3, seed=0)

    # Fabricate very different Q tables so the greedy action under A and B
    # differs deterministically.
    agent.Q_A[:] = 0.0
    agent.Q_B[:] = 0.0
    # Stage 1, next_state = 2 -- set Q_A argmax = action 2; Q_B argmax = action 0.
    agent.Q_A[1, 2, :] = [0.1, 0.2, 3.0]  # argmax = 2, max = 3.0
    agent.Q_B[1, 2, :] = [5.0, 0.3, 0.4]  # argmax = 0, max = 5.0

    r = 0.2
    # Manually compute the expected safe_target for both possible coin
    # outcomes using the math layer directly.
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
        SafeWeightedCommon,
    )
    swc_ref = SafeWeightedCommon(schedule=schedule, gamma=schedule.gamma,
                                 n_base=3)

    # Case A-selector: a* = argmax Q_A[.,2,.] = 2 ; v_next = Q_B[.,2,2] = 0.4.
    expected_A = swc_ref.compute_safe_target(r, 0.4, 1)
    # Case B-selector: a* = argmax Q_B[.,2,.] = 0 ; v_next = Q_A[.,2,0] = 0.1.
    expected_B = swc_ref.compute_safe_target(r, 0.1, 1)

    # Run the update and pick the right reference.
    log = agent.update(
        state=0, action=0, reward=r, next_state=2,
        absorbing=False, stage=1,
    )
    if log["selected_action_source"] == "A":
        assert log["safe_target_double"] == pytest.approx(expected_A, rel=1e-12)
        assert log["margin_double"] == pytest.approx(r - 0.4, rel=1e-12)
    else:
        assert log["safe_target_double"] == pytest.approx(expected_B, rel=1e-12)
        assert log["margin_double"] == pytest.approx(r - 0.1, rel=1e-12)


def test_double_gap_logged() -> None:
    """double_gap equals |Q_A[s',a*] - Q_B[s',a*]| at the selected action."""
    agent = _make_agent(_zero_schedule(T=2), n_states=3, n_actions=4, seed=5)
    agent.Q_A[:] = 0.0
    agent.Q_B[:] = 0.0
    # Stage 0, next_state 1.
    agent.Q_A[0, 1, :] = [0.0, 1.0, 0.2, 0.3]  # argmax = 1
    agent.Q_B[0, 1, :] = [0.4, 0.5, 0.6, 0.7]  # argmax = 3

    log = agent.update(
        state=0, action=0, reward=0.0, next_state=1,
        absorbing=False, stage=0,
    )
    a_star = log["a_star"]
    expected_gap = abs(
        float(agent.Q_A[0, 1, a_star]) - float(agent.Q_B[0, 1, a_star])
    )
    assert log["double_gap"] == pytest.approx(expected_gap, rel=0, abs=1e-15)

    # Also sanity-check q_a_next / q_b_next are table maxima.
    assert log["q_a_next"] == pytest.approx(1.0)
    assert log["q_b_next"] == pytest.approx(0.7)


def test_absorbing_double_gap_is_zero() -> None:
    """Absorbing transitions: v_next, double_gap, q_*_next are all zero."""
    agent = _make_agent(_zero_schedule(T=2), seed=0)
    agent.Q_A[:] = 1.5
    agent.Q_B[:] = -0.8
    log = agent.update(
        state=0, action=0, reward=0.3, next_state=1,
        absorbing=True, stage=0,
    )
    assert log["q_a_next"] == 0.0
    assert log["q_b_next"] == 0.0
    assert log["double_gap"] == 0.0
    assert log["margin_double"] == pytest.approx(0.3)


def test_beta0_reduces_to_classical_double_q() -> None:
    """With beta=0 at every stage the update equals classical Double Q.

    We run two agents with identical seeds: one safe (beta=0), one that
    implements the classical r + gamma * v_next rule directly.  After
    many updates the tables must agree to machine precision.
    """
    rng = np.random.default_rng(42)
    T = 4
    n_states = 5
    n_actions = 3
    gamma = 0.9
    lr = 0.3

    schedule = _zero_schedule(T=T, gamma=gamma)
    safe = _make_agent(
        schedule,
        n_states=n_states, n_actions=n_actions, lr=lr, seed=123,
    )

    # Reference: classical Double Q-Learning with the *same* coin stream
    # and the *same* argmax-tiebreak stream.
    ref_rng = np.random.default_rng(123)
    Q_A_ref = np.zeros_like(safe.Q_A)
    Q_B_ref = np.zeros_like(safe.Q_B)

    n_steps = 200
    for _ in range(n_steps):
        s = int(rng.integers(0, n_states))
        a = int(rng.integers(0, n_actions))
        sn = int(rng.integers(0, n_states))
        t = int(rng.integers(0, T))
        r = float(rng.normal())
        done = bool(rng.integers(0, 2) == 0 and t == T - 1)

        # Run reference first, consuming ref_rng in the same order as the
        # safe agent's internal RNG: (1) selector coin, (2) argmax tiebreak.
        use_A_sel = bool(ref_rng.integers(0, 2) == 0)
        if use_A_sel:
            Q_sel, Q_eval = Q_A_ref, Q_B_ref
        else:
            Q_sel, Q_eval = Q_B_ref, Q_A_ref
        if done:
            v_next = 0.0
            a_star = -1
        else:
            q_sel_next = Q_sel[t, sn, :]
            max_val = float(np.max(q_sel_next))
            argmax_set = np.flatnonzero(q_sel_next == max_val)
            a_star = int(ref_rng.choice(argmax_set))
            v_next = float(Q_eval[t, sn, a_star])
        classical_target = r + gamma * v_next
        Q_sel[t, s, a] = (1.0 - lr) * Q_sel[t, s, a] + lr * classical_target

        # Then run the safe agent with the same transition.
        log = safe.update(
            state=s, action=a, reward=r, next_state=sn,
            absorbing=done, stage=t,
        )
        # Safety: the safe target must equal the classical one (beta=0 branch).
        assert log["safe_target_double"] == pytest.approx(
            classical_target, rel=0, abs=1e-12
        )
        # Selector decisions must match bit-for-bit.
        expected_src = "A" if use_A_sel else "B"
        assert log["selected_action_source"] == expected_src

    # Tables must agree exactly.
    np.testing.assert_allclose(safe.Q_A, Q_A_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(safe.Q_B, Q_B_ref, atol=1e-12, rtol=0)


def test_get_v_averages_tables() -> None:
    """get_V returns max over the averaged Q_A/Q_B table."""
    agent = _make_agent(_zero_schedule(T=2), n_states=3, n_actions=3, seed=0)
    agent.Q_A[1, 2, :] = [1.0, 3.0, 2.0]
    agent.Q_B[1, 2, :] = [0.0, 1.0, 4.0]
    # avg = [0.5, 2.0, 3.0] -> max = 3.0.
    assert agent.get_V(state=2, stage=1) == pytest.approx(3.0)
    np.testing.assert_allclose(
        agent.get_Q(state=2, stage=1),
        np.array([0.5, 2.0, 3.0]),
    )


def test_numpy_state_scalar_handling() -> None:
    """Passing shape-(1,) numpy arrays for state/action must not raise.

    Guards against the numpy>=2.0 int(state) TypeError (lessons 2026-04-18).
    """
    agent = _make_agent(_zero_schedule(T=2), seed=0)
    log = agent.update(
        state=np.array([0]),
        action=np.array([1]),
        reward=0.1,
        next_state=np.array([2]),
        absorbing=False,
        stage=0,
    )
    assert isinstance(log["safe_target_double"], float)


def test_stage_out_of_range_raises() -> None:
    agent = _make_agent(_zero_schedule(T=3), seed=0)
    with pytest.raises(IndexError):
        agent.update(0, 0, 0.0, 1, False, stage=3)
    with pytest.raises(IndexError):
        agent.update(0, 0, 0.0, 1, False, stage=-1)
