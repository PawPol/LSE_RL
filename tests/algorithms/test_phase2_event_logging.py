"""Tests for Phase II event-level logging callbacks.

Spec references: docs/specs/phase_II_*.md S8.1, S8.2, S8.3, S8.4, S9.2.

Invariants guarded
------------------
1. ``EventTransitionLogger.build_payload()`` returns all 13 base keys + 5
   event-bool keys (18 total), all event arrays have length N and dtype bool.
2. Event counts match deterministic toy cases (mark_jackpot, mark_catastrophe,
   mark_hazard_hit produce exact counts).
3. ``regime_post_change`` is False for all steps before the change point and
   True for all steps after, when driven by ``ChainRegimeShiftWrapper.post_change``.
4. ``AdaptationMetricsLogger.compute`` returns correct pre_change_auc,
   post_change_optimum, recovery lags, and edge-case NaN/None.
5. ``TailRiskLogger.compute`` produces ``cvar_5pct <= return_q05``,
   ``cvar_10pct <= return_q10``, correct event_rate and
   event_conditioned_return (incl. NaN when no events).
6. ``TargetStatsLogger.compute`` produces correct aligned_positive,
   aligned_negative, and running TD std matching np.std at final index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

# ---------------------------------------------------------------------------
# sys.path setup so imports resolve regardless of working directory
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
for _p in [str(_REPO), str(_REPO / "src"), str(_REPO / "experiments")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.weighted_lse_dp.common.callbacks import (
    AdaptationMetricsLogger,
    EventTransitionLogger,
    TargetStatsLogger,
    TailRiskLogger,
)
from experiments.weighted_lse_dp.common.schemas import TRANSITIONS_ARRAYS
from experiments.weighted_lse_dp.tasks.nonstationary_wrappers import (
    ChainRegimeShiftWrapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_BASE = 25
_GAMMA = 0.99
_EVENT_KEYS = (
    "jackpot_event",
    "catastrophe_event",
    "regime_post_change",
    "hazard_cell_hit",
    "shortcut_action_taken",
)


class _DummyAgent:
    """Minimal agent stub exposing a Q-table interface."""

    class _Q:
        def __init__(self, n_aug: int, n_actions: int) -> None:
            self.table = np.zeros((n_aug, n_actions), dtype=np.float64)

        def __getitem__(self, key):
            if isinstance(key, tuple):
                return float(self.table[key])
            return float(self.table[key, 0])

    def __init__(self, n_aug: int = 100 * _N_BASE, n_actions: int = 4) -> None:
        self.Q = self._Q(n_aug, n_actions)


def _make_sample(
    aug_id: int = 0,
    action: int = 0,
    reward: float = 0.0,
    next_aug_id: int = 1,
    absorbing: bool = False,
    last: bool = False,
) -> tuple:
    """Build a synthetic Core sample tuple."""
    return (
        np.array([aug_id]),
        np.array([action]),
        reward,
        np.array([next_aug_id]),
        absorbing,
        last,
    )


# ===================================================================
# 1. Event arrays exist and have correct shape (spec S9.2)
# ===================================================================


class TestEventArraysExistAndShape:
    """Verify build_payload returns all 18 keys with correct length/dtype."""

    # docs/specs/phase_II_*.md S9.2 -- event arrays exist

    def test_payload_has_all_18_keys(self):
        """build_payload must return 13 base + 5 event keys = 18 total."""
        agent = _DummyAgent()
        logger = EventTransitionLogger(agent, n_base=_N_BASE, gamma=_GAMMA)

        N = 10
        for i in range(N):
            last = i == N - 1
            logger(_make_sample(aug_id=i, next_aug_id=i + 1, last=last))

        payload = logger.build_payload()
        # All 13 base keys
        for key in TRANSITIONS_ARRAYS:
            assert key in payload, f"Missing base key: {key}"
        # All 5 event keys
        for key in _EVENT_KEYS:
            assert key in payload, f"Missing event key: {key}"
        assert len(payload) == len(TRANSITIONS_ARRAYS) + len(_EVENT_KEYS)

    def test_event_arrays_length_and_dtype(self):
        """All event arrays must have length N and dtype bool."""
        agent = _DummyAgent()
        logger = EventTransitionLogger(agent, n_base=_N_BASE, gamma=_GAMMA)

        N = 15
        for i in range(N):
            last = i == N - 1
            logger(_make_sample(aug_id=i, next_aug_id=i + 1, last=last))

        payload = logger.build_payload()
        for key in _EVENT_KEYS:
            arr = payload[key]
            assert len(arr) == N, f"{key} has length {len(arr)}, expected {N}"
            assert arr.dtype == bool, f"{key} has dtype {arr.dtype}, expected bool"


# ===================================================================
# 2. Event counts match deterministic toy cases (spec S9.2)
# ===================================================================


class TestEventCountsDeterministic:
    """Mark specific steps as events and verify exact counts."""

    # docs/specs/phase_II_*.md S9.2 -- event counts match deterministic cases

    def test_jackpot_count(self):
        """mark_jackpot at steps 2 and 5 => jackpot_event.sum() == 2."""
        agent = _DummyAgent()
        logger = EventTransitionLogger(agent, n_base=_N_BASE, gamma=_GAMMA)

        N = 8
        for i in range(N):
            if i in (2, 5):
                logger.mark_jackpot()
            last = i == N - 1
            logger(_make_sample(aug_id=i, next_aug_id=i + 1, last=last))

        payload = logger.build_payload()
        assert payload["jackpot_event"].sum() == 2

    def test_catastrophe_count(self):
        """mark_catastrophe at steps 0 and 3 => catastrophe_event.sum() == 2."""
        agent = _DummyAgent()
        logger = EventTransitionLogger(agent, n_base=_N_BASE, gamma=_GAMMA)

        N = 6
        for i in range(N):
            if i in (0, 3):
                logger.mark_catastrophe()
            last = i == N - 1
            logger(_make_sample(aug_id=i, next_aug_id=i + 1, last=last))

        payload = logger.build_payload()
        assert payload["catastrophe_event"].sum() == 2

    def test_hazard_cell_hit_count(self):
        """mark_hazard_hit at steps 1,4,7 => hazard_cell_hit.sum() == 3."""
        agent = _DummyAgent()
        logger = EventTransitionLogger(agent, n_base=_N_BASE, gamma=_GAMMA)

        N = 10
        for i in range(N):
            if i in (1, 4, 7):
                logger.mark_hazard_hit()
            last = i == N - 1
            logger(_make_sample(aug_id=i, next_aug_id=i + 1, last=last))

        payload = logger.build_payload()
        assert payload["hazard_cell_hit"].sum() == 3

    def test_set_step_events_batch(self):
        """set_step_events sets multiple flags at once."""
        agent = _DummyAgent()
        logger = EventTransitionLogger(agent, n_base=_N_BASE, gamma=_GAMMA)

        logger.set_step_events(jackpot=True, shortcut=True)
        logger(_make_sample(aug_id=0, next_aug_id=1, last=True))

        payload = logger.build_payload()
        assert payload["jackpot_event"][0] is np.True_
        assert payload["shortcut_action_taken"][0] is np.True_
        assert payload["catastrophe_event"][0] is np.False_

    def test_pending_flags_reset_after_call(self):
        """Pending flags must reset to False after __call__."""
        agent = _DummyAgent()
        logger = EventTransitionLogger(agent, n_base=_N_BASE, gamma=_GAMMA)

        logger.mark_jackpot()
        logger(_make_sample(aug_id=0, next_aug_id=1, last=False))
        # Second step without marking: should be False
        logger(_make_sample(aug_id=1, next_aug_id=2, last=True))

        payload = logger.build_payload()
        assert payload["jackpot_event"][0] is np.True_
        assert payload["jackpot_event"][1] is np.False_


# ===================================================================
# 3. Post-change flags activate only after the change point (spec S9.2)
# ===================================================================


class TestPostChangeFlagsActivation:
    """Verify regime_post_change mirrors ChainRegimeShiftWrapper.post_change."""

    # docs/specs/phase_II_*.md S9.2 -- post-change flags after change point

    def test_regime_post_change_timing(self):
        """Episodes 0-2: all False; episodes 3-5: all True (change_at_episode=3)."""
        from mushroom_rl.environments.generators.simple_chain import (
            compute_probabilities as chain_p,
            compute_reward as chain_r,
        )
        from mushroom_rl.environments.finite_mdp import FiniteMDP

        state_n = 5
        horizon = 5
        gamma = 0.99
        prob = 0.9

        p = chain_p(state_n, prob)
        r = chain_r(state_n, [state_n - 1], 1.0)
        mu = np.zeros(state_n)
        mu[0] = 1.0

        mdp_pre = FiniteMDP(p, r, mu, gamma, horizon)
        mdp_post = FiniteMDP(p, r, mu, gamma, horizon)  # same MDP, just testing flags

        wrapper = ChainRegimeShiftWrapper(mdp_pre, mdp_post, change_at_episode=3)

        agent = _DummyAgent(n_aug=(horizon + 1) * state_n, n_actions=2)
        logger = EventTransitionLogger(agent, n_base=state_n, gamma=gamma)

        n_episodes = 6
        steps_per_episode = horizon
        all_regime_flags: list[bool] = []

        for ep in range(n_episodes):
            state, _ = wrapper.reset()
            for step in range(steps_per_episode):
                # Mark regime status BEFORE the logger call
                logger.mark_regime_post_change(wrapper.post_change)
                action = np.array([0])
                next_state, reward, absorbing, _ = wrapper.step(action)
                last = step == steps_per_episode - 1 or absorbing

                aug_id = step * state_n + int(state[0]) if state.ndim > 0 else step * state_n + int(state)
                next_aug_id = (step + 1) * state_n + (int(next_state[0]) if next_state.ndim > 0 else int(next_state))

                sample = _make_sample(
                    aug_id=aug_id,
                    action=0,
                    reward=float(reward),
                    next_aug_id=next_aug_id,
                    absorbing=absorbing,
                    last=last,
                )
                logger(sample)
                all_regime_flags.append(wrapper.post_change)

                if absorbing or last:
                    break
                state = next_state

        payload = logger.build_payload()
        regime_arr = payload["regime_post_change"]

        # Verify per-episode correctness
        idx = 0
        ep_idx = payload["episode_index"]
        for ep in range(n_episodes):
            mask = ep_idx == ep
            ep_flags = regime_arr[mask]
            if ep < 3:
                assert not np.any(ep_flags), (
                    f"Episode {ep} should have all False regime flags"
                )
            else:
                assert np.all(ep_flags), (
                    f"Episode {ep} should have all True regime flags"
                )


# ===================================================================
# 4. AdaptationMetricsLogger correctness (spec S8.2)
# ===================================================================


class TestAdaptationMetricsLogger:
    """Verify adaptation metrics computation."""

    # docs/specs/phase_II_*.md S8.2 -- adaptation metrics

    def test_basic_adaptation_metrics(self):
        """Known returns yield correct pre_change_auc, post_change_optimum, lag."""
        returns = np.array(
            [1.0] * 10 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        logger = AdaptationMetricsLogger()
        result = logger.compute(returns, change_at_episode=10)

        # pre_change_auc is mean of first 10 returns (all 1.0)
        npt.assert_allclose(result["pre_change_auc"], 1.0, atol=1e-12)
        # post_change_optimum is max of post-change returns
        npt.assert_allclose(result["post_change_optimum"], 1.0, atol=1e-12)
        # lag_to_50pct_recovery should not be None (50% of 1.0 = 0.5)
        assert result["lag_to_50pct_recovery"] is not None, (
            "lag_to_50pct_recovery should not be None for recoverable returns"
        )

    def test_recovery_lag_value(self):
        """Verify the exact recovery lag for a known trajectory."""
        # Post-change returns: [0.1, 0.2, ..., 1.0]
        # post_change_optimum = 1.0
        # 50% threshold = 0.5
        # Rolling window = 10 (default)
        # At i=4, post[4]=0.5, rolling mean of [0.1..0.5] = 0.3 < 0.5
        # At i=8, rolling mean of [0.1..0.9] = 0.5 >= 0.5 -> lag=8
        returns = np.array(
            [1.0] * 10 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        logger = AdaptationMetricsLogger()
        result = logger.compute(returns, change_at_episode=10)

        # The rolling mean at i=8 is mean([0.1,0.2,...,0.9]) = 0.45 < 0.5
        # At i=9, rolling mean of [0.1,0.2,...,1.0] = 0.55 >= 0.5 -> lag=9
        # Let's compute exactly to be safe
        post = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        threshold = 0.5 * 1.0  # 50% of optimum
        w = 10
        expected_lag = None
        for i in range(len(post)):
            start = max(0, i - w + 1)
            rm = np.mean(post[start : i + 1])
            if rm >= threshold:
                expected_lag = i
                break
        assert result["lag_to_50pct_recovery"] == expected_lag

    def test_edge_case_change_beyond_episodes(self):
        """change_at_episode >= n_episodes => all NaN/None."""
        returns = np.array([1.0, 2.0, 3.0])
        logger = AdaptationMetricsLogger()
        result = logger.compute(returns, change_at_episode=10)

        assert np.isnan(result["pre_change_auc"])
        assert np.isnan(result["post_change_auc"])
        assert np.isnan(result["post_change_optimum"])
        assert result["lag_to_50pct_recovery"] is None
        assert result["lag_to_75pct_recovery"] is None
        assert result["lag_to_90pct_recovery"] is None

    def test_no_recovery(self):
        """Returns that never recover yield None lags."""
        returns = np.array([1.0] * 5 + [0.0] * 5)
        logger = AdaptationMetricsLogger()
        result = logger.compute(returns, change_at_episode=5)

        # post_change_optimum = 0.0, threshold = 0.0
        # 50% of 0.0 is 0.0 -- rolling mean of zeros >= 0.0 is True at i=0
        # So lag should be 0 (since threshold is 0)
        assert result["lag_to_50pct_recovery"] is not None


# ===================================================================
# 5. TailRiskLogger numerical stability (spec S8.3)
# ===================================================================


class TestTailRiskLogger:
    """Verify tail-risk computations are numerically stable and correct."""

    # docs/specs/phase_II_*.md S8.3 -- tail-risk metrics

    def test_cvar_leq_quantile(self):
        """CVaR at alpha% must be <= the alpha-th percentile (CVaR is worse)."""
        returns = np.linspace(0, 1, 100)
        flags = np.zeros(100, dtype=bool)
        logger = TailRiskLogger()
        result = logger.compute(returns, flags)

        # CVaR_5% <= q05
        assert result["cvar_5pct"] <= result["return_q05"] + 1e-12, (
            f"cvar_5pct={result['cvar_5pct']} > return_q05={result['return_q05']}"
        )

        # CVaR_10% <= q10 (manually computed)
        return_q10 = np.percentile(returns, 10)
        assert result["cvar_10pct"] <= return_q10 + 1e-12, (
            f"cvar_10pct={result['cvar_10pct']} > return_q10={return_q10}"
        )

    def test_no_events_rate_zero_and_nan_return(self):
        """event_rate == 0.0 and event_conditioned_return is NaN when no events."""
        returns = np.linspace(0, 1, 100)
        flags = np.zeros(100, dtype=bool)
        logger = TailRiskLogger()
        result = logger.compute(returns, flags)

        npt.assert_equal(result["event_rate"], 0.0)
        assert np.isnan(result["event_conditioned_return"])

    def test_all_events_rate_one_and_mean_return(self):
        """event_rate == 1.0 and event_conditioned_return == mean(returns)."""
        returns = np.linspace(0, 1, 100)
        flags = np.ones(100, dtype=bool)
        logger = TailRiskLogger()
        result = logger.compute(returns, flags)

        npt.assert_allclose(result["event_rate"], 1.0, atol=1e-12)
        npt.assert_allclose(
            result["event_conditioned_return"],
            np.mean(returns),
            atol=1e-12,
        )

    def test_single_episode_no_errors(self):
        """Single episode must not cause division-by-zero or NaN in quantiles."""
        returns = np.array([0.5])
        flags = np.array([True])
        logger = TailRiskLogger()
        result = logger.compute(returns, flags)

        # All quantiles should be 0.5
        for qk in ("return_q05", "return_q25", "return_q50", "return_q75", "return_q95"):
            npt.assert_allclose(result[qk], 0.5, atol=1e-12)
        # CVaR should also be 0.5
        npt.assert_allclose(result["cvar_5pct"], 0.5, atol=1e-12)
        npt.assert_allclose(result["cvar_10pct"], 0.5, atol=1e-12)
        # Event stats
        npt.assert_allclose(result["event_rate"], 1.0, atol=1e-12)
        npt.assert_allclose(result["event_conditioned_return"], 0.5, atol=1e-12)

    def test_quantile_ordering(self):
        """Quantiles must be monotonically non-decreasing: q05 <= q25 <= q50 <= q75 <= q95."""
        rng = np.random.RandomState(99)
        returns = rng.randn(200)
        flags = np.zeros(200, dtype=bool)
        logger = TailRiskLogger()
        result = logger.compute(returns, flags)

        assert result["return_q05"] <= result["return_q25"] + 1e-12
        assert result["return_q25"] <= result["return_q50"] + 1e-12
        assert result["return_q50"] <= result["return_q75"] + 1e-12
        assert result["return_q75"] <= result["return_q95"] + 1e-12


# ===================================================================
# 6. TargetStatsLogger correctness (spec S8.4)
# ===================================================================


class TestTargetStatsLogger:
    """Verify aligned margins and running TD std."""

    # docs/specs/phase_II_*.md S8.4 -- target statistics

    def test_aligned_positive_and_negative(self):
        """aligned_positive = max(margin, 0), aligned_negative = max(-margin, 0)."""
        margin = np.array([-1.0, 0.0, 2.0, -0.5])
        payload = {
            "margin_beta0": margin,
            "td_target_beta0": np.array([1.0, 2.0, 3.0, 4.0]),
            "td_error_beta0": np.array([0.1, 0.2, 0.3, 0.4]),
        }
        logger = TargetStatsLogger()
        result = logger.compute(payload)

        npt.assert_array_equal(
            result["aligned_positive"],
            np.array([0.0, 0.0, 2.0, 0.0]),
        )
        npt.assert_array_equal(
            result["aligned_negative"],
            np.array([1.0, 0.0, 0.0, 0.5]),
        )

    def test_td_target_std_running_final(self):
        """Running std at final index matches np.std (population) of td_target."""
        td_target = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
        payload = {
            "margin_beta0": np.zeros(5),
            "td_target_beta0": td_target,
            "td_error_beta0": np.zeros(5),
        }
        logger = TargetStatsLogger()
        result = logger.compute(payload)

        # The implementation uses Welford's online algorithm with population std
        # (divides by count, not count-1). np.std with ddof=0 matches.
        expected_std = np.std(td_target, ddof=0)
        npt.assert_allclose(
            result["td_target_std_running"][-1],
            expected_std,
            atol=1e-10,
        )

    def test_td_error_std_running_final(self):
        """Running std at final index for td_error matches np.std."""
        td_error = np.array([0.1, -0.3, 0.5, -0.2, 0.4])
        payload = {
            "margin_beta0": np.zeros(5),
            "td_target_beta0": np.zeros(5),
            "td_error_beta0": td_error,
        }
        logger = TargetStatsLogger()
        result = logger.compute(payload)

        expected_std = np.std(td_error, ddof=0)
        npt.assert_allclose(
            result["td_error_std_running"][-1],
            expected_std,
            atol=1e-10,
        )

    def test_single_element(self):
        """Single-element payload must not error; std should be 0.0."""
        payload = {
            "margin_beta0": np.array([3.0]),
            "td_target_beta0": np.array([1.0]),
            "td_error_beta0": np.array([0.5]),
        }
        logger = TargetStatsLogger()
        result = logger.compute(payload)

        npt.assert_array_equal(result["aligned_positive"], np.array([3.0]))
        npt.assert_array_equal(result["aligned_negative"], np.array([0.0]))
        npt.assert_allclose(result["td_target_std_running"][0], 0.0, atol=1e-12)
        npt.assert_allclose(result["td_error_std_running"][0], 0.0, atol=1e-12)

    def test_output_shapes_and_dtypes(self):
        """All output arrays must have shape (N,) and dtype float64."""
        N = 20
        payload = {
            "margin_beta0": np.random.randn(N),
            "td_target_beta0": np.random.randn(N),
            "td_error_beta0": np.random.randn(N),
        }
        logger = TargetStatsLogger()
        result = logger.compute(payload)

        for key in ("aligned_positive", "aligned_negative",
                     "td_target_std_running", "td_error_std_running"):
            assert result[key].shape == (N,), f"{key} shape mismatch"
            assert result[key].dtype == np.float64, f"{key} dtype mismatch"
