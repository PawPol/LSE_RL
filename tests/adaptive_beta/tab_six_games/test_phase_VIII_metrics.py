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
