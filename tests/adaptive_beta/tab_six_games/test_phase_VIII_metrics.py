"""Phase VIII delta-metric tests for tab-six-games metrics."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.tab_six_games import metrics


def f64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def test_epsilon_constant_is_pinned() -> None:
    assert metrics.EPSILON == pytest.approx(1e-8)


def test_contraction_reward_monotone_decreasing_is_positive() -> None:
    residual = f64([1.0, 0.5, 0.25, 0.125])

    reward = metrics.contraction_reward(residual)

    assert reward.shape == (3,)
    assert np.all(reward > 0.0)


def test_contraction_reward_empty_equal_and_zero_residual_cases() -> None:
    empty = metrics.contraction_reward(f64([]))
    equal = metrics.contraction_reward(f64([0.5, 0.5]))
    zeros = metrics.contraction_reward(f64([0.0, 0.0]))

    assert empty.shape == (0,)
    np.testing.assert_allclose(equal, f64([0.0]), atol=1e-12)
    np.testing.assert_allclose(zeros, f64([0.0]), atol=1e-12)
    assert np.all(np.isfinite(zeros))


def test_empirical_contraction_ratio_monotone_decreasing_is_below_one() -> None:
    residual = f64([1.0, 0.5, 0.25, 0.125])

    ratio = metrics.empirical_contraction_ratio(residual)

    assert ratio.shape == (3,)
    assert np.all(ratio < 1.0)


def test_log_residual_reduction_matches_contraction_reward() -> None:
    residual = f64([1.0, 0.5, 0.25, 0.125])

    np.testing.assert_allclose(
        metrics.log_residual_reduction(residual),
        metrics.contraction_reward(residual),
        atol=1e-12,
    )


def test_ucb_arm_count_counts_pulls_per_arm() -> None:
    arms = f64([0.0, 1.0, 0.0, 2.0, 1.0, 0.0])

    counts = metrics.ucb_arm_count(arms, n_arms=3)

    np.testing.assert_array_equal(counts, np.array([3, 2, 1], dtype=np.int64))


def test_ucb_arm_count_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError):
        metrics.ucb_arm_count(f64([0.0, 7.0]), n_arms=7)


def test_ucb_arm_value_computes_mean_reward_per_arm() -> None:
    rewards = f64([1.0, 2.0, 3.0, 4.0])
    arms = f64([0.0, 1.0, 0.0, 1.0])

    values = metrics.ucb_arm_value(rewards, arms, n_arms=7)

    expected = f64([2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(values, expected, atol=1e-12)


def test_ucb_arm_value_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics.ucb_arm_value(f64([1.0, 2.0]), f64([0.0]), n_arms=7)


def test_beta_clip_count_and_frequency() -> None:
    beta_raw = f64([2.5, 0.5, -3.0])
    beta_used = f64([2.0, 0.5, -2.0])

    assert metrics.beta_clip_count(beta_raw, beta_used) == 2
    assert metrics.beta_clip_frequency(beta_raw, beta_used) == pytest.approx(2.0 / 3.0)


def test_beta_clip_frequency_empty_is_zero() -> None:
    assert metrics.beta_clip_frequency(f64([]), f64([])) == pytest.approx(0.0)


def test_beta_clip_count_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics.beta_clip_count(f64([1.0]), f64([1.0, 2.0]))


def test_recovery_time_after_shift_known_lag_and_sentinels() -> None:
    returns = f64([-1.0, -0.5, 0.1, 0.3, 0.8, 1.2])

    assert metrics.recovery_time_after_shift(returns, shift_episode=2, threshold=0.75) == 2
    assert metrics.recovery_time_after_shift(returns, shift_episode=2, threshold=2.0) == -1
    assert metrics.recovery_time_after_shift(returns, shift_episode=20, threshold=0.75) == -1


def test_beta_sign_correct_truth_table_including_oracle_zero() -> None:
    beta_used = f64([2.0, -2.0, -0.5, 0.5, 0.0, 0.1])
    oracle = f64([1.0, 1.0, -1.0, -1.0, 0.0, 0.0])

    correct = metrics.beta_sign_correct(beta_used, oracle)

    np.testing.assert_array_equal(
        correct,
        np.array([True, False, True, False, True, False], dtype=bool),
    )


def test_beta_sign_correct_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics.beta_sign_correct(f64([0.0]), f64([0.0, 1.0]))


def test_beta_lag_to_oracle_identical_streams_and_flip_lag() -> None:
    identical = f64([-2.0, -1.0, 0.5, 2.0])
    np.testing.assert_array_equal(
        metrics.beta_lag_to_oracle(identical, identical),
        np.zeros(4, dtype=np.int64),
    )

    oracle = np.zeros(15, dtype=np.float64)
    oracle[10:] = 1.0
    used = np.full(15, -2.0, dtype=np.float64)
    used[10] = 1.0
    used[11] = 3.0
    used[12:] = -3.0

    lag = metrics.beta_lag_to_oracle(used, oracle)

    assert lag[12] == 2


def test_beta_lag_to_oracle_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics.beta_lag_to_oracle(f64([0.0]), f64([0.0, 1.0]))


def test_regret_vs_oracle_positive_when_oracle_dominates() -> None:
    oracle_returns = f64([2.0, 2.0, 2.0])
    method_returns = f64([1.0, 0.5, -1.0])

    regret = metrics.regret_vs_oracle(oracle_returns, method_returns)

    assert regret > 0.0
    assert regret == pytest.approx(5.5)


def test_regret_vs_oracle_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics.regret_vs_oracle(f64([1.0]), f64([1.0, 2.0]))


def test_catastrophic_episodes_counts_returns_below_threshold() -> None:
    returns = f64([-1.0, 0.5, -2.0])

    assert metrics.catastrophic_episodes(returns, theta_low=-0.5) == 2


def test_worst_window_return_percentile_flat_and_drifting_returns() -> None:
    flat = np.full(120, 3.0, dtype=np.float64)
    drifting = np.arange(60, dtype=np.float64)
    window = 50
    rolling_means = np.array(
        [np.mean(drifting[i : i + window]) for i in range(drifting.size - window + 1)],
        dtype=np.float64,
    )

    assert metrics.worst_window_return_percentile(flat, window=100, pct=5.0) == pytest.approx(3.0)
    assert metrics.worst_window_return_percentile(drifting, window=window, pct=5.0) == pytest.approx(
        float(np.percentile(rolling_means, 5.0))
    )


def test_worst_window_return_percentile_short_input_falls_back_to_single_window() -> None:
    returns = f64([1.0, 3.0, 5.0])

    assert metrics.worst_window_return_percentile(returns, window=100, pct=5.0) == pytest.approx(3.0)


def test_worst_window_return_percentile_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError):
        metrics.worst_window_return_percentile(f64([1.0, 2.0]), window=0)


def test_trap_entries_counts_distinct_entries() -> None:
    states = f64([0.0, 2.0, 2.0, 1.0, 3.0, 3.0, 0.0])
    trap_states = f64([2.0, 3.0])

    assert metrics.trap_entries(states, trap_states) == 2


def test_trap_entries_empty_trap_states_is_zero() -> None:
    assert metrics.trap_entries(f64([1.0, 2.0, 3.0]), f64([])) == 0


def test_constraint_violations_counts_nonzero_entries() -> None:
    assert metrics.constraint_violations(f64([0.0, 1.0, 0.0, 2.0])) == 2


def test_overflow_count_counts_values_at_or_above_threshold() -> None:
    q_abs_max = f64([1e3, 1e7, 5e5])

    assert metrics.overflow_count(q_abs_max, threshold=1e6) == 1


@pytest.mark.parametrize(
    ("metric_name", "metric_call"),
    [
        ("contraction_reward", lambda empty: metrics.contraction_reward(empty)),
        ("empirical_contraction_ratio", lambda empty: metrics.empirical_contraction_ratio(empty)),
        ("log_residual_reduction", lambda empty: metrics.log_residual_reduction(empty)),
        ("ucb_arm_count", lambda empty: metrics.ucb_arm_count(empty, n_arms=7)),
        ("ucb_arm_value", lambda empty: metrics.ucb_arm_value(empty, empty, n_arms=7)),
        ("beta_clip_count", lambda empty: metrics.beta_clip_count(empty, empty)),
        ("beta_clip_frequency", lambda empty: metrics.beta_clip_frequency(empty, empty)),
        ("recovery_time_after_shift", lambda empty: metrics.recovery_time_after_shift(empty, 0, 0.0)),
        ("beta_sign_correct", lambda empty: metrics.beta_sign_correct(empty, empty)),
        ("beta_lag_to_oracle", lambda empty: metrics.beta_lag_to_oracle(empty, empty)),
        ("regret_vs_oracle", lambda empty: metrics.regret_vs_oracle(empty, empty)),
        ("catastrophic_episodes", lambda empty: metrics.catastrophic_episodes(empty, theta_low=0.0)),
        ("worst_window_return_percentile", lambda empty: metrics.worst_window_return_percentile(empty)),
        ("trap_entries", lambda empty: metrics.trap_entries(empty, empty)),
        ("constraint_violations", lambda empty: metrics.constraint_violations(empty)),
        ("overflow_count", lambda empty: metrics.overflow_count(empty)),
    ],
)
def test_empty_array_safety_for_all_metrics(metric_name, metric_call) -> None:
    empty = f64([])

    result = metric_call(empty)

    if isinstance(result, np.ndarray):
        assert result.size >= 0, metric_name
    else:
        assert result is not None, metric_name


# ---------------------------------------------------------------------------
# Phase VIII §5.7 / patch v3 (2026-05-01) — Q-convergence rate metric tests
# ---------------------------------------------------------------------------


def test_q_convergence_rate_shape() -> None:
    rng = np.random.default_rng(0)
    q_hist = rng.normal(size=(100, 11, 1))
    q_star = metrics.q_star_delayed_chain(L=10, gamma=0.95)
    rate = metrics.q_convergence_rate(q_hist, q_star)
    assert rate.shape == (99,)
    assert rate.dtype == np.float64


def test_q_convergence_rate_monotone_under_perfect_decay() -> None:
    # Residual = exp(-0.05 * t), so log-residual decays linearly at
    # rate 0.05/step. Stays well above eps=1e-8 across the full
    # 100-episode horizon (residual at t=99 ≈ 0.0067 >> 1e-8); the
    # original v3 patch test used exp(-arange) which underflows below
    # eps after ~18 steps and saturates rate to 0.
    L, gamma = 10, 0.95
    q_star = metrics.q_star_delayed_chain(L, gamma)
    q_hist = q_star[None, :, :] + np.exp(
        -0.05 * np.arange(100)
    )[:, None, None]
    rate = metrics.q_convergence_rate(q_hist, q_star)
    assert np.all(rate > 0), f"some rates non-positive: {rate[:5]}"
    # Rate per step ≈ 0.05 (since residual ~ e^{-0.05 t}).
    assert np.abs(rate.mean() - 0.05) < 0.01


def test_q_star_delayed_chain_geometric() -> None:
    L, gamma = 5, 0.9
    q_star = metrics.q_star_delayed_chain(L, gamma)
    # Q*(s=0, advance) = gamma**(L-1) (v4 fix: reward delivered on
    # L-1 → L transition, so Q*(L-1, advance) = 1 = γ^0).
    assert np.isclose(q_star[0, 0], gamma ** (L - 1))
    # Q*(s=L-1, advance) = 1 (one step before the +1 reward).
    assert np.isclose(q_star[L - 1, 0], 1.0)
    # Q*(s=L, .) = 0 (terminal).
    assert np.isclose(q_star[L, 0], 0.0)


def test_q_convergence_rate_eps_floor_safety() -> None:
    # Q exactly equal to Q* should not produce -inf or NaN.
    L, gamma = 10, 0.95
    q_star = metrics.q_star_delayed_chain(L, gamma)
    q_hist = np.broadcast_to(
        q_star[None, :, :], (10, L + 1, 1)
    ).copy()
    rate = metrics.q_convergence_rate(q_hist, q_star)
    assert np.all(np.isfinite(rate))
    assert not np.any(np.isnan(rate))


# ---------------------------------------------------------------------------
# Phase VIII §5.7 / patch v5 (2026-05-01) — β-specific Bellman residual tests
# ---------------------------------------------------------------------------


def _make_chain_transition(L: int):
    """Return the deterministic delayed_chain transition function.

    advance-only chain: state s ∈ [0, L]; action 0 → s+1 with reward 0,
    except the last advance (s == L-1, a=0) yields reward +1 and absorbs
    at s' = L. State L is terminal (any action returns reward 0 and
    keeps state at L).
    """
    def env_transition(s, a):
        if s >= L:
            return [(1.0, 0.0, L)]
        r = 1.0 if (s + 1) == L else 0.0
        return [(1.0, r, s + 1)]
    return env_transition


def test_bellman_residual_beta_zero_at_fixed_point() -> None:
    """At Q*_classical the β=0 residual is 0 exactly."""
    L, gamma = 10, 0.95
    Q_star = metrics.q_star_delayed_chain(L=L, gamma=gamma)
    # Pad with terminal Q[L,0] = 0 to match (L+1, 1) shape used by env_transition.
    env_t = _make_chain_transition(L)
    residual = metrics.bellman_residual_beta(
        Q=Q_star, beta=0.0, gamma=gamma,
        env_transition=env_t, n_states=L + 1, n_actions=1,
    )
    # Q* satisfies the classical Bellman equation; residual should be 0
    # to floating-point precision.
    assert residual < 1e-10, f"expected residual ≈ 0, got {residual}"


def test_bellman_residual_beta_classical_collapse() -> None:
    """β=0 residual matches classical |T*Q − Q|."""
    L, gamma = 5, 0.9
    Q = np.full((L + 1, 1), 5.0, dtype=np.float64)
    env_t = _make_chain_transition(L)
    residual_beta_zero = metrics.bellman_residual_beta(
        Q=Q, beta=0.0, gamma=gamma,
        env_transition=env_t, n_states=L + 1, n_actions=1,
    )
    # Classical Bellman residual ||T* Q − Q||_∞ for constant Q=5:
    # at non-terminal s != L-1: target = 0 + 0.9*5 = 4.5; |4.5 − 5| = 0.5
    # at s = L-1: target = 1 + 0.9*5 = 5.5; |5.5 − 5| = 0.5
    # at s = L (terminal): target = 0 + 0.9*5 = 4.5; |4.5 − 5| = 0.5
    # max = 0.5
    assert np.isclose(residual_beta_zero, 0.5)


def test_auc_neg_log_residual_monotone() -> None:
    """A converging residual sequence has higher AUC than a non-converging one."""
    R_converging = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=np.float64)
    R_flat = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    auc_conv = metrics.auc_neg_log_residual(R_converging)
    auc_flat = metrics.auc_neg_log_residual(R_flat)
    assert auc_conv > auc_flat, (
        f"converging AUC ({auc_conv}) should exceed flat AUC ({auc_flat})"
    )


def test_bellman_residual_beta_handles_divergent_Q() -> None:
    """Passing extremely large Q values should not produce NaN."""
    L, gamma = 5, 0.95
    # Bound at 1e6 (matches the smoke-test cap) — the operator's output
    # should be finite at this magnitude.
    Q = np.full((L + 1, 1), 1.0e6, dtype=np.float64)
    env_t = _make_chain_transition(L)
    for beta in [-1.0, 0.0, 1.0]:
        residual = metrics.bellman_residual_beta(
            Q=Q, beta=beta, gamma=gamma,
            env_transition=env_t, n_states=L + 1, n_actions=1,
        )
        assert np.isfinite(residual), (
            f"residual non-finite for β={beta}: {residual}"
        )
