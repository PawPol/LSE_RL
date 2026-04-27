"""Tests for ``GameHistory`` (Phase VII-B spec §5.3).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md`` §5.3.

Invariants guarded
------------------
- ``last(m)`` slicing returns the most recent ``m`` steps without
  mutating the parent.
- ``empirical_*_policy`` returns shape ``(n_actions,)`` and sums to 1
  on any non-empty window; uniform fallback on empty windows.
- ``rolling_return(m)`` matches a numpy reference on a synthetic trace.
- ``append`` preserves chronological order and out-of-range ``m`` is
  clamped to history length.

Each test docstring points at the spec line it enforces.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.adaptive_beta.strategic_games.history import GameHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(n_steps: int = 50, n_actions: int = 3, seed: int = 0) -> GameHistory:
    """Build a deterministic 50-step trace for round-trip / window tests."""
    rng = np.random.default_rng(seed)
    h = GameHistory()
    for t in range(n_steps):
        a = int(rng.integers(0, n_actions))
        b = int(rng.integers(0, n_actions))
        r = float(rng.standard_normal())
        h.append(
            agent_action=a,
            opponent_action=b,
            agent_reward=r,
            opponent_reward=-r,
            info={"t": t},
        )
    return h


# ---------------------------------------------------------------------------
# last(m)
# ---------------------------------------------------------------------------

def test_last_m_returns_last_m_entries() -> None:
    """`spec §5.3` — ``last(m)`` returns the most recent ``m`` steps."""
    h = _make_history(n_steps=10, n_actions=2)
    last3 = h.last(3)
    assert len(last3) == 3
    assert last3.agent_actions == h.agent_actions[-3:]
    assert last3.opponent_actions == h.opponent_actions[-3:]
    assert last3.agent_rewards == h.agent_rewards[-3:]


def test_last_zero_returns_empty() -> None:
    """`spec §5.3` — ``last(0)`` returns an empty history."""
    h = _make_history(n_steps=5, n_actions=2)
    empty = h.last(0)
    assert len(empty) == 0
    assert empty.agent_actions == []


def test_last_m_clamps_to_history_length() -> None:
    """`spec §5.3` — ``m > len(self)`` returns the full history (no padding)."""
    h = _make_history(n_steps=5, n_actions=2)
    full = h.last(100)
    assert len(full) == 5
    assert full.agent_actions == h.agent_actions


def test_last_returns_independent_copy() -> None:
    """`spec §5.3` — ``last(m)`` does not let mutations leak back to parent."""
    h = _make_history(n_steps=5, n_actions=2)
    snap = h.last(3)
    snap.agent_actions.append(999)
    assert 999 not in h.agent_actions


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------

def test_append_preserves_order() -> None:
    """`spec §5.3` — chronological order is preserved across appends."""
    h = GameHistory()
    expected_a = list(range(7))
    for i in expected_a:
        h.append(
            agent_action=i,
            opponent_action=i,
            agent_reward=float(i),
            opponent_reward=-float(i),
            info=None,
        )
    assert h.agent_actions == expected_a
    assert h.opponent_actions == expected_a
    assert h.agent_rewards == [float(i) for i in expected_a]


def test_append_normalises_numpy_scalars() -> None:
    """`spec §5.3` / lessons.md — numpy scalar inputs are coerced to Python types."""
    h = GameHistory()
    h.append(
        agent_action=np.int64(2),
        opponent_action=np.array([1]),
        agent_reward=np.float32(0.5),
    )
    assert isinstance(h.agent_actions[0], int)
    assert isinstance(h.opponent_actions[0], int)
    assert isinstance(h.agent_rewards[0], float)
    assert h.agent_actions[0] == 2
    assert h.opponent_actions[0] == 1


# ---------------------------------------------------------------------------
# empirical_*_policy
# ---------------------------------------------------------------------------

def test_empirical_agent_policy_shape_and_sum() -> None:
    """`spec §5.3` — ``empirical_*_policy`` returns shape ``(n_actions,)``, sums to 1."""
    h = _make_history(n_steps=20, n_actions=3)
    p = h.empirical_agent_policy(m=10, n_actions=3)
    assert p.shape == (3,)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)


def test_empirical_opponent_policy_shape_and_sum() -> None:
    """`spec §5.3` — opponent variant matches the agent variant's contract."""
    h = _make_history(n_steps=20, n_actions=3)
    p = h.empirical_opponent_policy(m=10, n_actions=3)
    assert p.shape == (3,)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)


def test_empirical_policy_uniform_on_empty_window() -> None:
    """`spec §5.3` — empty window returns the uniform distribution."""
    h = GameHistory()
    p = h.empirical_agent_policy(m=10, n_actions=4)
    assert p.shape == (4,)
    np.testing.assert_allclose(p, np.full(4, 0.25), atol=1e-12)


def test_empirical_policy_matches_handcrafted_window() -> None:
    """`spec §5.3` — empirical frequencies match the hand-counted window."""
    h = GameHistory()
    actions = [0, 1, 1, 2, 0, 1, 1, 1, 2, 2]
    for a in actions:
        h.append(
            agent_action=a, opponent_action=a, agent_reward=0.0,
        )
    # Trailing window of length 5: [1, 1, 1, 2, 2] -> [0, 3/5, 2/5].
    p = h.empirical_agent_policy(m=5, n_actions=3)
    np.testing.assert_allclose(p, [0.0, 0.6, 0.4], atol=1e-12)


# ---------------------------------------------------------------------------
# rolling_return(m)
# ---------------------------------------------------------------------------

def test_rolling_return_matches_numpy_reference() -> None:
    """`spec §5.3` — rolling_return matches numpy sum over the trailing window."""
    h = _make_history(n_steps=50, n_actions=3, seed=42)
    rewards = np.asarray(h.agent_rewards, dtype=np.float64)
    for m in (1, 5, 17, 50):
        ref = float(rewards[-m:].sum())
        got = h.rolling_return(m)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_rolling_return_clamped_above_history() -> None:
    """`spec §5.3` — out-of-range ``m`` is clamped to history length."""
    h = _make_history(n_steps=10, n_actions=2)
    rewards = np.asarray(h.agent_rewards, dtype=np.float64)
    np.testing.assert_allclose(h.rolling_return(1000), float(rewards.sum()), atol=1e-12)


def test_rolling_return_zero_on_empty_or_zero_m() -> None:
    """`spec §5.3` — rolling_return returns 0.0 on empty history or m<=0."""
    h_empty = GameHistory()
    assert h_empty.rolling_return(5) == 0.0
    h = _make_history(n_steps=3, n_actions=2)
    assert h.rolling_return(0) == 0.0
    assert h.rolling_return(-3) == 0.0


def test_rolling_return_propagates_nan() -> None:
    """`spec §5.3` — non-finite rewards in the window produce NaN."""
    h = GameHistory()
    h.append(agent_action=0, opponent_action=0, agent_reward=1.0)
    h.append(agent_action=0, opponent_action=0, agent_reward=float("nan"))
    h.append(agent_action=0, opponent_action=0, agent_reward=2.0)
    out = h.rolling_return(3)
    assert np.isnan(out)


# ---------------------------------------------------------------------------
# Invariant tripwire: change behaviour and the test should fail.
# ---------------------------------------------------------------------------

def test_invariant_window_slicing_is_chronological() -> None:
    """`spec §5.3` — `last(m)` returns the trailing window in chronological
    order. If the implementation accidentally reversed the slice, every
    finite-memory adversary's empirical estimator would silently flip.
    """
    h = GameHistory()
    for i in range(10):
        h.append(agent_action=i, opponent_action=i, agent_reward=float(i))
    last5 = h.last(5)
    assert last5.agent_actions == [5, 6, 7, 8, 9]
    # Rewards strictly increasing: confirm direction.
    assert all(
        last5.agent_rewards[i] < last5.agent_rewards[i + 1]
        for i in range(len(last5) - 1)
    )
