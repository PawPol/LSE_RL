"""Tests for ``metrics.py`` (Phase VII-B spec §9.3).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§§9.3, 10 (event-aligned panels).

Each metric is exercised against a hand-computed reference. Empty-input
behaviour returns ``np.nan`` rather than crashing — this is the §9.3
contract for downstream NaN-aware aggregation.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.strategic_games.metrics import (
    coordination_rate,
    cycling_amplitude,
    empirical_best_response_value,
    event_aligned_window,
    external_regret,
    miscoordination_rate,
    policy_total_variation,
    rolling_policy_entropy,
    search_phase_vs_stable_phase_return,
    support_shift_count,
)


# ---------------------------------------------------------------------------
# rolling_policy_entropy
# ---------------------------------------------------------------------------

def test_rolling_policy_entropy_uniform_two_actions_equals_ln2() -> None:
    """`spec §9.3` — uniform binary distribution has Shannon entropy ``ln 2``."""
    actions = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    h = rolling_policy_entropy(actions, n_actions=2, window=6)
    np.testing.assert_allclose(h, np.log(2.0), atol=1e-12)


def test_rolling_policy_entropy_pure_actions_equals_zero() -> None:
    """`spec §9.3` — degenerate distribution has zero entropy."""
    actions = np.zeros(10, dtype=np.int64)
    h = rolling_policy_entropy(actions, n_actions=2, window=10)
    np.testing.assert_allclose(h, 0.0, atol=1e-12)


def test_rolling_policy_entropy_empty_returns_nan() -> None:
    """`spec §9.3` — empty input returns NaN, never crashes."""
    h = rolling_policy_entropy(np.array([], dtype=np.int64), n_actions=2)
    assert np.isnan(h)


# ---------------------------------------------------------------------------
# policy_total_variation
# ---------------------------------------------------------------------------

def test_policy_total_variation_disjoint_pmfs_equals_one() -> None:
    """`spec §9.3` — TV distance between two disjoint pmfs equals 1."""
    a = np.zeros(10, dtype=np.int64)         # pure action 0
    b = np.ones(10, dtype=np.int64)          # pure action 1
    tv = policy_total_variation(a, b, n_actions=2)
    np.testing.assert_allclose(tv, 1.0, atol=1e-12)


def test_policy_total_variation_identical_pmfs_equals_zero() -> None:
    """`spec §9.3` — TV between identical pmfs is 0."""
    a = np.array([0, 0, 1, 1], dtype=np.int64)
    tv = policy_total_variation(a, a.copy(), n_actions=2)
    np.testing.assert_allclose(tv, 0.0, atol=1e-12)


def test_policy_total_variation_empty_returns_nan() -> None:
    """`spec §9.3` — both windows empty ⇒ NaN."""
    tv = policy_total_variation(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), n_actions=2,
    )
    assert np.isnan(tv)


# ---------------------------------------------------------------------------
# support_shift_count
# ---------------------------------------------------------------------------

def test_support_shift_count_strictly_above_threshold() -> None:
    """`spec §9.3` — shifts are counted strictly above ``threshold_tv``;
    equality (TV exactly == threshold) is NOT a shift.
    """
    # Two-row trace: row1 = uniform, row2 differs by exact threshold.
    # threshold = 0.1 → TV must exceed 0.1 to register.
    p0 = np.array([0.5, 0.5])
    p1 = np.array([0.6, 0.4])  # TV = 0.5 * (|0.1| + |-0.1|) = 0.1 (boundary)
    P = np.stack([p0, p1])
    n_at_eq = support_shift_count(P, threshold_tv=0.1)
    assert n_at_eq == 0  # equality → no shift

    p2 = np.array([0.61, 0.39])  # TV = 0.11, strictly above
    P_above = np.stack([p0, p2])
    n_above = support_shift_count(P_above, threshold_tv=0.1)
    assert n_above == 1


def test_support_shift_count_short_traces_return_zero() -> None:
    """`spec §9.3` — single-row or empty traces yield zero shifts."""
    assert support_shift_count(np.zeros((0, 3))) == 0
    assert support_shift_count(np.array([[0.5, 0.5]])) == 0


# ---------------------------------------------------------------------------
# external_regret
# ---------------------------------------------------------------------------

def test_external_regret_handcrafted_value_two_actions() -> None:
    """`spec §9.3` — agent always 0, opponent always 1, payoffs [[1,0],[0,2]]:
    realised mean per step = M[0, 1] = 0; best fixed agent action = 1
    (collects M[1, 1] = 2 every step). Regret = 2 - 0 = 2.0.
    """
    M = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    T = 50
    a = np.zeros(T, dtype=np.int64)
    o = np.ones(T, dtype=np.int64)
    r = external_regret(a, M, o)
    np.testing.assert_allclose(r, 2.0, atol=1e-12)


def test_external_regret_zero_when_agent_plays_best_response() -> None:
    """`spec §9.3` — when the agent plays the best fixed action, regret is 0."""
    M = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    T = 50
    a = np.ones(T, dtype=np.int64)
    o = np.ones(T, dtype=np.int64)
    r = external_regret(a, M, o)
    np.testing.assert_allclose(r, 0.0, atol=1e-12)


def test_external_regret_empty_returns_nan() -> None:
    """`spec §9.3` — empty input ⇒ NaN."""
    M = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    r = external_regret(
        np.array([], dtype=np.int64), M, np.array([], dtype=np.int64),
    )
    assert np.isnan(r)


# ---------------------------------------------------------------------------
# cycling_amplitude
# ---------------------------------------------------------------------------

def test_cycling_amplitude_zero_for_constant_actions() -> None:
    """`spec §9.3` — constant agent + opponent action streams have zero amplitude."""
    a = np.zeros(50, dtype=np.int64)
    o = np.zeros(50, dtype=np.int64)
    amp = cycling_amplitude(a, o, window=5)
    np.testing.assert_allclose(amp, 0.0, atol=1e-12)


def test_cycling_amplitude_positive_for_alternation_window_1() -> None:
    """`spec §9.3` — alternating actions over a window=1 reading produce
    positive amplitude (peak-to-peak frequency spread = 1.0 per coordinate
    × 4 coordinates / 2 = nonzero).
    """
    a = np.tile([0, 1], 50)
    o = np.tile([1, 0], 50)
    amp = cycling_amplitude(a, o, window=1)
    assert amp > 0.0


def test_cycling_amplitude_empty_returns_nan() -> None:
    """`spec §9.3` — empty inputs ⇒ NaN."""
    amp = cycling_amplitude(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), window=5,
    )
    assert np.isnan(amp)


# ---------------------------------------------------------------------------
# event_aligned_window
# ---------------------------------------------------------------------------

def test_event_aligned_window_shape_and_centre() -> None:
    """`spec §10` — shape ``(K, 2*half_window+1)``; column ``half_window`` is
    the event step.
    """
    values = np.arange(100, dtype=np.float64)
    events = np.array([10, 50], dtype=np.int64)
    panel = event_aligned_window(values, events, half_window=5)
    assert panel.shape == (2, 11)
    # Column 5 of each row is the event index value itself.
    assert panel[0, 5] == 10
    assert panel[1, 5] == 50


def test_event_aligned_window_boundary_padding_left() -> None:
    """`spec §10` — left-boundary event at index 0 → first 5 columns are NaN."""
    values = np.arange(10, dtype=np.float64)
    events = np.array([0], dtype=np.int64)
    panel = event_aligned_window(values, events, half_window=5)
    # First 5 entries of row 0 are NaN, then [0, 1, 2, 3, 4, 5].
    assert np.all(np.isnan(panel[0, :5]))
    np.testing.assert_array_equal(panel[0, 5:], np.arange(6, dtype=np.float64))


def test_event_aligned_window_boundary_padding_right() -> None:
    """`spec §10` — right-boundary event at the last index → last 5 columns are NaN."""
    values = np.arange(10, dtype=np.float64)
    events = np.array([9], dtype=np.int64)
    panel = event_aligned_window(values, events, half_window=5)
    # First 6 entries of row 0 are [4, 5, 6, 7, 8, 9], then 5 NaNs.
    np.testing.assert_array_equal(panel[0, :6], np.arange(4, 10, dtype=np.float64))
    assert np.all(np.isnan(panel[0, 6:]))


def test_event_aligned_window_no_events_returns_empty_panel() -> None:
    """`spec §10` — zero events ⇒ shape ``(0, width)`` panel."""
    panel = event_aligned_window(np.arange(10), np.array([], dtype=np.int64), 5)
    assert panel.shape == (0, 11)


def test_event_aligned_window_negative_half_window_raises() -> None:
    """`spec §10` — negative half-window is rejected."""
    with pytest.raises(ValueError):
        event_aligned_window(np.arange(10), np.array([5]), half_window=-1)


# ---------------------------------------------------------------------------
# search_phase / coordination
# ---------------------------------------------------------------------------

def test_search_phase_vs_stable_phase_return_partition() -> None:
    """`spec §9.3` — masks correctly partition the return array."""
    returns = np.array([1.0, 2.0, 3.0, 4.0])
    flags = np.array([True, True, False, False])
    s, b = search_phase_vs_stable_phase_return(returns, flags)
    np.testing.assert_allclose(s, 1.5)
    np.testing.assert_allclose(b, 3.5)


def test_search_phase_vs_stable_phase_return_empty_returns_nan() -> None:
    """`spec §9.3` — empty inputs ⇒ (NaN, NaN)."""
    s, b = search_phase_vs_stable_phase_return(
        np.array([], dtype=np.float64), np.array([], dtype=bool),
    )
    assert np.isnan(s) and np.isnan(b)


@pytest.mark.parametrize(
    "agent,opp,expected",
    [
        (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 0.0),  # full coord
        (np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), 1.0),  # full miscoord
        (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), 0.5),  # half / half
    ],
)
def test_miscoordination_rate_basic(agent, opp, expected: float) -> None:
    """`spec §9.3` — miscoordination = fraction of off-diagonal joint actions."""
    np.testing.assert_allclose(miscoordination_rate(agent, opp), expected)


def test_coordination_plus_miscoordination_equals_one() -> None:
    """`spec §9.3` — ``coordination_rate + miscoordination_rate == 1`` always."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.integers(0, 3, size=50)
        o = rng.integers(0, 3, size=50)
        c = coordination_rate(a, o)
        m = miscoordination_rate(a, o)
        np.testing.assert_allclose(c + m, 1.0, atol=1e-12)


def test_miscoordination_rate_empty_returns_nan() -> None:
    """`spec §9.3` — empty inputs ⇒ NaN."""
    m = miscoordination_rate(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
    )
    assert np.isnan(m)


# ---------------------------------------------------------------------------
# empirical_best_response_value
# ---------------------------------------------------------------------------

def test_empirical_best_response_value_handcrafted() -> None:
    """`spec §9.3` — best response to ``[1, 0]`` against ``M = [[1, 0], [0, 2]]``
    is action 0 with value 1.
    """
    M = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    v = empirical_best_response_value(np.array([1.0, 0.0]), M)
    np.testing.assert_allclose(v, 1.0, atol=1e-12)


def test_empirical_best_response_value_renormalises() -> None:
    """`spec §9.3` — frequencies that don't sum to 1 are renormalised."""
    M = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    v = empirical_best_response_value(np.array([3.0, 0.0]), M)  # sum=3
    np.testing.assert_allclose(v, 1.0, atol=1e-12)


def test_empirical_best_response_value_degenerate_returns_nan() -> None:
    """`spec §9.3` — zero-sum or negative frequency vector ⇒ NaN."""
    M = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    assert np.isnan(empirical_best_response_value(np.array([0.0, 0.0]), M))


# ---------------------------------------------------------------------------
# Tripwire: NaN-safe contract
# ---------------------------------------------------------------------------

def test_invariant_metrics_never_crash_on_empty_input() -> None:
    """`spec §9.3` — every metric in this module returns ``np.nan`` (or 0
    for integer counters) on empty input rather than raising. A regression
    that swallowed the empty-input guard would crash the aggregator on
    skipped runs.
    """
    empty = np.array([], dtype=np.int64)
    # Pairs of (callable, expected_kind) where _kind ∈ {"nan", "zero"}.
    cases = [
        (lambda: rolling_policy_entropy(empty, n_actions=2), "nan"),
        (lambda: policy_total_variation(empty, empty, n_actions=2), "nan"),
        (lambda: support_shift_count(np.zeros((0, 2))), "zero"),
        (lambda: external_regret(empty, np.eye(2), empty), "nan"),
        (lambda: cycling_amplitude(empty, empty, window=5), "nan"),
        (lambda: miscoordination_rate(empty, empty), "nan"),
        (lambda: coordination_rate(empty, empty), "nan"),
    ]
    for fn, kind in cases:
        out = fn()
        if kind == "nan":
            assert np.isnan(out), f"{fn} did not return NaN on empty input"
        else:
            assert out == 0
