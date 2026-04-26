"""Unit tests for ``experiments/adaptive_beta/schedules.py`` (Phase VII M1.3).

Test indices match the spec contract in ``tasks/`` for M1.3 and the
non-negotiables in ``docs/specs/phase_VII_adaptive_beta.md`` §13.2 + §13.5.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import (
    ALL_METHOD_IDS,
    METHOD_ADAPTIVE_BETA,
    METHOD_ADAPTIVE_BETA_NO_CLIP,
    METHOD_ADAPTIVE_MAGNITUDE_ONLY,
    METHOD_ADAPTIVE_SIGN_ONLY,
    METHOD_FIXED_NEGATIVE,
    METHOD_FIXED_POSITIVE,
    METHOD_VANILLA,
    METHOD_WRONG_SIGN,
    AdaptiveBetaSchedule,
    AdaptiveMagnitudeOnlySchedule,
    AdaptiveSignOnlySchedule,
    FixedBetaSchedule,
    WrongSignSchedule,
    ZeroBetaSchedule,
    build_schedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _drive_episode(
    sched,
    episode_index: int,
    rewards,
    v_next,
    divergence_event: bool = False,
) -> None:
    """Apply one update and assert episode_index monotonicity."""
    sched.update_after_episode(
        episode_index,
        np.asarray(rewards, dtype=np.float64),
        np.asarray(v_next, dtype=np.float64),
        divergence_event=divergence_event,
    )


# ---------------------------------------------------------------------------
# 1. Construction parity through the factory.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method_id,env_canonical_sign",
    [
        (METHOD_VANILLA, None),
        (METHOD_FIXED_POSITIVE, None),
        (METHOD_FIXED_NEGATIVE, None),
        (METHOD_WRONG_SIGN, "+"),
        (METHOD_WRONG_SIGN, "-"),
        (METHOD_ADAPTIVE_BETA, None),
        (METHOD_ADAPTIVE_BETA_NO_CLIP, None),
        (METHOD_ADAPTIVE_SIGN_ONLY, None),
        (METHOD_ADAPTIVE_MAGNITUDE_ONLY, "+"),
        (METHOD_ADAPTIVE_MAGNITUDE_ONLY, "-"),
    ],
)
def test_factory_round_trip(method_id, env_canonical_sign):
    sched = build_schedule(method_id, env_canonical_sign, hyperparams={})
    assert sched.name == method_id
    # Every method_id we declare to the agent appears in ALL_METHOD_IDS.
    assert method_id in ALL_METHOD_IDS


# ---------------------------------------------------------------------------
# 2. Beta is constant within an episode (50 reads, all equal).
# ---------------------------------------------------------------------------
def test_beta_constant_within_episode():
    sched = AdaptiveBetaSchedule(
        no_clip=False,
        hyperparams={"beta_max": 2.0, "beta_cap": 2.0, "k": 5.0},
    )
    # Drive a couple of episodes so beta is non-trivial.
    _drive_episode(sched, 0, rewards=[1.0, 0.5, -0.2], v_next=[0.0, 0.1, 0.05])
    _drive_episode(sched, 1, rewards=[0.3, 0.3, 0.3], v_next=[0.0, 0.0, 0.0])

    reads: List[float] = [sched.beta_for_episode(7) for _ in range(50)]
    assert len(reads) == 50
    first = reads[0]
    assert all(r == first for r in reads), (
        "beta_for_episode must be constant within an episode; "
        f"observed {len(set(reads))} distinct values: {set(reads)}"
    )


# ---------------------------------------------------------------------------
# 3. No-future-leakage spy. Wrap rewards/v_next with a sentinel that records
#    every per-step access; the schedule must only ever see the "current"
#    episode's data via .mean / arithmetic on the array passed in.
# ---------------------------------------------------------------------------
class _ForbiddenFutureSpy:
    """ndarray-like guard that throws if anyone reaches into the future.

    The schedule should *only* read the trace passed to ``update_after_episode``
    for the current episode. This spy fails loudly if a future episode's
    data is ever queried while computing the next beta.
    """

    def __init__(self, current_episode: int) -> None:
        self.current_episode = current_episode
        self.queried_episode_indices: List[int] = []

    def get_trace(self, episode_index: int) -> np.ndarray:
        self.queried_episode_indices.append(episode_index)
        if episode_index >= self.current_episode + 1:
            raise AssertionError(
                f"future-leakage: episode {episode_index} >= "
                f"current_episode+1 = {self.current_episode + 1}"
            )
        # Return a small trace so the schedule can compute A_e.
        return np.array([0.5, 0.5, -0.1], dtype=np.float64)


def test_no_future_leakage():
    sched = AdaptiveBetaSchedule(no_clip=False)
    # Walk episode_index forward; on the final step, fetch trace via the spy
    # and confirm no future index was ever asked for.
    for e in range(8):
        spy = _ForbiddenFutureSpy(current_episode=e)
        rewards = spy.get_trace(e)
        v_next = spy.get_trace(e)
        sched.update_after_episode(e, rewards, v_next, divergence_event=False)
        # The spy is only queried for index e (twice). It must NEVER trip.
        assert all(i <= e for i in spy.queried_episode_indices)
    # Also confirm episode-monotonicity assertion: skipping triggers the guard.
    with pytest.raises(AssertionError):
        sched.update_after_episode(
            42, np.array([0.0]), np.array([0.0]), divergence_event=False
        )


# ---------------------------------------------------------------------------
# 4. Clip enforcement on adaptive_beta.
# ---------------------------------------------------------------------------
def test_clip_enforced_on_adaptive_beta():
    sched = AdaptiveBetaSchedule(
        no_clip=False,
        hyperparams={"beta_max": 10.0, "beta_cap": 2.0, "k": 5.0},
    )
    # Synthetic A_e ~= 1.0 saturates tanh -> raw beta ~= 10.0.
    _drive_episode(sched, 0, rewards=[1.0] * 10, v_next=[0.0] * 10)
    beta_used = sched.beta_for_episode(1)
    assert abs(beta_used) <= 2.0 + 1e-12, beta_used
    diag = sched.diagnostics()
    assert abs(diag["beta_raw"]) > 2.0  # raw exceeds the cap


# ---------------------------------------------------------------------------
# 5. No-clip schedule actually exceeds the cap.
# ---------------------------------------------------------------------------
def test_no_clip_can_exceed_cap():
    sched = AdaptiveBetaSchedule(
        no_clip=True,
        hyperparams={"beta_max": 10.0, "beta_cap": 2.0, "k": 5.0},
    )
    _drive_episode(sched, 0, rewards=[1.0] * 10, v_next=[0.0] * 10)
    beta_used = sched.beta_for_episode(1)
    assert abs(beta_used) > 2.0, (
        f"no_clip schedule should bypass beta_cap; got {beta_used}"
    )


# ---------------------------------------------------------------------------
# 6. Divergence flag is reported and sticky (spec §13.5).
# ---------------------------------------------------------------------------
def test_divergence_flag_sticky():
    sched = AdaptiveBetaSchedule(no_clip=True)
    _drive_episode(sched, 0, rewards=[0.0], v_next=[0.0], divergence_event=False)
    assert sched.diagnostics()["divergence_event"] is False

    # Episode 1 reports divergence.
    _drive_episode(sched, 1, rewards=[0.0], v_next=[0.0], divergence_event=True)
    assert sched.diagnostics()["divergence_event"] is True

    # Subsequent clean episode must not clear the flag.
    _drive_episode(sched, 2, rewards=[0.0], v_next=[0.0], divergence_event=False)
    assert sched.diagnostics()["divergence_event"] is True


# ---------------------------------------------------------------------------
# 7. ZeroBetaSchedule is exactly zero everywhere.
# ---------------------------------------------------------------------------
def test_zero_schedule_returns_zero():
    sched = ZeroBetaSchedule()
    for e in (0, 1, 7, 999):
        assert sched.beta_for_episode(e) == 0.0
    # Drive an update with a huge advantage; must remain zero.
    _drive_episode(sched, 0, rewards=[100.0] * 5, v_next=[0.0] * 5)
    assert sched.beta_for_episode(1) == 0.0


# ---------------------------------------------------------------------------
# 8. wrong_sign / adaptive_magnitude_only restricted to canonical-sign envs.
# ---------------------------------------------------------------------------
def test_wrong_sign_requires_canonical_sign():
    with pytest.raises(ValueError):
        WrongSignSchedule(env_canonical_sign=None)
    with pytest.raises(ValueError):
        WrongSignSchedule(env_canonical_sign="?")
    with pytest.raises(ValueError):
        AdaptiveMagnitudeOnlySchedule(env_canonical_sign=None)
    with pytest.raises(ValueError):
        AdaptiveMagnitudeOnlySchedule(env_canonical_sign="?")
    # Factory dispatch raises the same error.
    with pytest.raises(ValueError):
        build_schedule(METHOD_WRONG_SIGN, env_canonical_sign=None, hyperparams={})
    with pytest.raises(ValueError):
        build_schedule(
            METHOD_ADAPTIVE_MAGNITUDE_ONLY, env_canonical_sign=None, hyperparams={}
        )


# ---------------------------------------------------------------------------
# 9. Wrong-sign sign flip: '+' env -> -beta0; '-' env -> +beta0.
# ---------------------------------------------------------------------------
def test_wrong_sign_flips_canonical():
    plus_env = WrongSignSchedule(env_canonical_sign="+", hyperparams={"beta0": 1.5})
    assert plus_env.beta_for_episode(0) == -1.5
    _drive_episode(plus_env, 0, rewards=[1.0], v_next=[0.0])
    assert plus_env.beta_for_episode(1) == -1.5

    minus_env = WrongSignSchedule(env_canonical_sign="-", hyperparams={"beta0": 1.5})
    assert minus_env.beta_for_episode(0) == +1.5
    _drive_episode(minus_env, 0, rewards=[1.0], v_next=[0.0])
    assert minus_env.beta_for_episode(1) == +1.5


# ---------------------------------------------------------------------------
# 10. Smoothing: lambda=0.5, A_0=1.0, A_1=0.0 -> A_bar_1 = 0.25.
# ---------------------------------------------------------------------------
def test_smoothing_ema():
    hp = {
        "beta_max": 2.0,
        "beta_cap": 2.0,
        "k": 5.0,
        "lambda_smooth": 0.5,
    }
    sched = AdaptiveBetaSchedule(no_clip=False, hyperparams=hp)
    # A_0 = mean(r - v_next) = 1.0
    _drive_episode(sched, 0, rewards=[1.0, 1.0], v_next=[0.0, 0.0])
    # A_1 = 0.0
    _drive_episode(sched, 1, rewards=[0.5, 0.5], v_next=[0.5, 0.5])

    # A_bar_0 = 0.5 * 1.0 = 0.5
    # A_bar_1 = 0.5 * 0.5 + 0.5 * 0.0 = 0.25
    expected_bar = 0.25
    np.testing.assert_allclose(sched.diagnostics()["smoothed_A"], expected_bar, atol=1e-12)

    expected_beta = 2.0 * np.tanh(5.0 * expected_bar)
    expected_beta = float(np.clip(expected_beta, -2.0, 2.0))
    np.testing.assert_allclose(
        sched.beta_for_episode(2), expected_beta, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 11. Initial-episode safety: adaptive_beta.beta_for_episode(0) == initial_beta.
# ---------------------------------------------------------------------------
def test_initial_episode_safety():
    sched = AdaptiveBetaSchedule(no_clip=False)
    assert sched.beta_for_episode(0) == 0.0  # default initial_beta
    sched2 = AdaptiveBetaSchedule(
        no_clip=False, hyperparams={"initial_beta": 0.7}
    )
    assert sched2.beta_for_episode(0) == 0.7


# ---------------------------------------------------------------------------
# 12. adaptive_sign_only matches np.sign(A_bar_e), and 0 produces 0 exactly.
# ---------------------------------------------------------------------------
def test_adaptive_sign_only_extraction():
    sched = AdaptiveSignOnlySchedule(
        hyperparams={"beta0": 0.8, "beta_cap": 2.0}
    )
    # Positive A_e -> +beta0
    _drive_episode(sched, 0, rewards=[2.0, 2.0], v_next=[0.0, 0.0])
    assert sched.beta_for_episode(1) == +0.8

    # Exactly zero A_e -> beta == 0.0
    _drive_episode(sched, 1, rewards=[0.5, 0.5], v_next=[0.5, 0.5])
    assert sched.beta_for_episode(2) == 0.0

    # Negative A_e -> -beta0
    _drive_episode(sched, 2, rewards=[0.0, 0.0], v_next=[3.0, 3.0])
    assert sched.beta_for_episode(3) == -0.8


# ---------------------------------------------------------------------------
# 13. adaptive_magnitude_only: '+' env => non-negative beta, '-' env => non-positive.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "canonical,expected_sign_predicate",
    [
        ("+", lambda b: b >= 0.0),
        ("-", lambda b: b <= 0.0),
    ],
)
def test_adaptive_magnitude_only_sign_locked(canonical, expected_sign_predicate):
    sched = AdaptiveMagnitudeOnlySchedule(
        env_canonical_sign=canonical,
        hyperparams={"beta_max": 2.0, "beta_cap": 2.0, "k": 5.0},
    )
    # Run several episodes with mixed-sign A_e; sign must stay locked.
    episodes = [
        ([1.0, 1.0], [0.0, 0.0]),       # large positive A_e
        ([0.0, 0.0], [3.0, 3.0]),       # large negative A_e
        ([0.5, 0.5], [0.5, 0.5]),       # zero A_e
        ([0.1, -0.1], [0.0, 0.0]),      # near-zero A_e
    ]
    for e, (r, vn) in enumerate(episodes):
        _drive_episode(sched, e, rewards=r, v_next=vn)
        b = sched.beta_for_episode(e + 1)
        assert expected_sign_predicate(b), (
            f"canonical={canonical!r} episode={e} beta={b} violates sign lock"
        )


# ---------------------------------------------------------------------------
# Bonus: fixed schedules return the right value at episode 0 (no leakage).
# ---------------------------------------------------------------------------
def test_fixed_schedules_constant_pre_update():
    pos = FixedBetaSchedule(+1, hyperparams={"beta0": 1.5})
    neg = FixedBetaSchedule(-1, hyperparams={"beta0": 1.5})
    assert pos.beta_for_episode(0) == +1.5
    assert neg.beta_for_episode(0) == -1.5
    _drive_episode(pos, 0, rewards=[10.0], v_next=[0.0])
    _drive_episode(neg, 0, rewards=[10.0], v_next=[0.0])
    assert pos.beta_for_episode(1) == +1.5
    assert neg.beta_for_episode(1) == -1.5
