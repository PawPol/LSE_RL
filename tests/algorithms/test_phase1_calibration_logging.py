"""Tests for Phase I calibration logging: transitions payload, calibration
aggregation, and RunWriter round-trip persistence.

Spec references: docs/specs/phase_I_*.md S7.1, S7.2, S7.4, S8.4.

Invariants guarded
------------------
- ``transitions.npz`` contains all 13 keys from ``TRANSITIONS_ARRAYS``.
- Stage indices ``t`` lie in ``[0, horizon-1]``.
- ``margin_beta0 == reward - v_next_beta0`` (NO gamma factor) -- exact.
- ``td_target_beta0 == reward + gamma * v_next_beta0`` -- exact.
- ``td_error_beta0 == td_target_beta0 - q_current_beta0`` -- exact.
- Aggregated calibration stats (17 keys) match raw-transition reductions.
- ``RunWriter`` round-trips transitions and calibration through disk.
- The ``margin_beta0_formula`` string is stamped in the schema header.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path setup so imports resolve regardless of working directory
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
for _p in [str(_REPO), str(_REPO / "src"), str(_REPO / "experiments")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.weighted_lse_dp.common.calibration import (
    aggregate_calibration_stats,
    build_calibration_stats_from_dp_tables,
    build_transitions_payload,
)
from experiments.weighted_lse_dp.common.io import load_npz
from experiments.weighted_lse_dp.common.schemas import (
    CALIBRATION_ARRAYS,
    MARGIN_BETA0_FORMULA,
    TRANSITIONS_ARRAYS,
    RunWriter,
    validate_calibration_npz,
    validate_transitions_npz,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)
_HORIZON = 5
_N = 60  # total transitions (divisible by _HORIZON for uniform stages)
_GAMMA = 0.99


@pytest.fixture()
def transitions_payload() -> dict[str, np.ndarray]:
    """Build a synthetic but structurally valid transitions payload."""
    rng = np.random.RandomState(7)

    # Stage indices cycle 0.._HORIZON-1 so every stage has _N//_HORIZON samples
    t = np.tile(np.arange(_HORIZON, dtype=np.int64), _N // _HORIZON)
    episode_index = np.repeat(np.arange(_N // _HORIZON, dtype=np.int64), _HORIZON)

    state = rng.randint(0, 10, size=_N).astype(np.int64)
    action = rng.randint(0, 3, size=_N).astype(np.int64)
    reward = rng.randn(_N).astype(np.float64)
    next_state = rng.randint(0, 10, size=_N).astype(np.int64)
    absorbing = np.zeros(_N, dtype=bool)
    last = np.zeros(_N, dtype=bool)
    # Mark last step of each episode
    last[_HORIZON - 1 :: _HORIZON] = True

    q_current_beta0 = rng.randn(_N).astype(np.float64) + 1.0
    v_next_beta0 = rng.randn(_N).astype(np.float64) + 0.5

    return build_transitions_payload(
        episode_index=episode_index,
        t=t,
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        absorbing=absorbing,
        last=last,
        q_current_beta0=q_current_beta0,
        v_next_beta0=v_next_beta0,
        gamma=_GAMMA,
    )


@pytest.fixture()
def calibration_stats(transitions_payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Aggregate calibration stats from the transitions fixture."""
    return aggregate_calibration_stats(
        transitions_payload, horizon=_HORIZON,
    )


# ===========================================================================
# TestTransitionsPayload
# ===========================================================================


class TestTransitionsPayload:
    """Verify the per-transition payload built by build_transitions_payload."""

    def test_all_mandatory_keys_present(
        self, transitions_payload: dict[str, np.ndarray]
    ) -> None:
        """transitions payload must contain all 13 TRANSITIONS_ARRAYS keys.
        # docs/specs/phase_I_*.md S7.1
        """
        missing = [k for k in TRANSITIONS_ARRAYS if k not in transitions_payload]
        assert missing == [], f"Missing keys: {missing}"

    def test_margin_formula_no_gamma(
        self, transitions_payload: dict[str, np.ndarray]
    ) -> None:
        """margin_beta0 must equal reward - v_next_beta0, NOT reward - gamma*v_next.
        # docs/specs/phase_I_*.md S7.1
        """
        p = transitions_payload
        np.testing.assert_allclose(
            p["margin_beta0"],
            p["reward"] - p["v_next_beta0"],
            rtol=1e-10,
            atol=0,
            err_msg="margin_beta0 must be reward - v_next (no gamma)",
        )

        # Verify gamma was NOT applied (v_next is nonzero in our fixture)
        gamma_version = p["reward"] - _GAMMA * p["v_next_beta0"]
        assert not np.allclose(p["margin_beta0"], gamma_version), (
            "margin_beta0 must NOT be reward - gamma*v_next"
        )

    def test_td_target_uses_gamma(
        self, transitions_payload: dict[str, np.ndarray]
    ) -> None:
        """td_target_beta0 must equal reward + gamma * v_next_beta0.
        # docs/specs/phase_I_*.md S7.1
        """
        p = transitions_payload
        np.testing.assert_allclose(
            p["td_target_beta0"],
            p["reward"] + _GAMMA * p["v_next_beta0"],
            rtol=1e-10,
            atol=0,
            err_msg="td_target_beta0 must be reward + gamma*v_next",
        )

    def test_td_error_formula(
        self, transitions_payload: dict[str, np.ndarray]
    ) -> None:
        """td_error_beta0 = td_target_beta0 - q_current_beta0.
        # docs/specs/phase_I_*.md S7.1
        """
        p = transitions_payload
        np.testing.assert_allclose(
            p["td_error_beta0"],
            p["td_target_beta0"] - p["q_current_beta0"],
            rtol=1e-10,
            atol=0,
            err_msg="td_error_beta0 must be td_target - q_current",
        )

    def test_shape_consistency(
        self, transitions_payload: dict[str, np.ndarray]
    ) -> None:
        """All arrays must have the same length N.
        # docs/specs/phase_I_*.md S7.1
        """
        lengths = {k: v.shape[0] for k, v in transitions_payload.items()}
        unique = set(lengths.values())
        assert len(unique) == 1, f"Inconsistent lengths: {lengths}"

    def test_stage_indices_valid(
        self, transitions_payload: dict[str, np.ndarray]
    ) -> None:
        """Stage indices t must lie in [0, horizon-1].
        # docs/specs/phase_I_*.md S8.4
        """
        t = transitions_payload["t"]
        assert t.min() >= 0, f"Negative stage index: {t.min()}"
        assert t.max() <= _HORIZON - 1, f"Stage index too large: {t.max()}"


# ===========================================================================
# TestCalibrationStatsAggregation
# ===========================================================================


class TestCalibrationStatsAggregation:
    """Verify aggregate_calibration_stats matches raw-transition reductions."""

    def test_all_mandatory_keys_present(
        self, calibration_stats: dict[str, np.ndarray]
    ) -> None:
        """Calibration stats must contain all 17 CALIBRATION_ARRAYS keys.
        # docs/specs/phase_I_*.md S7.2
        """
        missing = [k for k in CALIBRATION_ARRAYS if k not in calibration_stats]
        assert missing == [], f"Missing keys: {missing}"

    def test_stage_indices(
        self, calibration_stats: dict[str, np.ndarray]
    ) -> None:
        """stats['stage'] == np.arange(horizon+1).
        # docs/specs/phase_I_*.md S7.2
        """
        np.testing.assert_array_equal(
            calibration_stats["stage"],
            np.arange(_HORIZON + 1, dtype=np.int64),
        )

    def test_count_sums_to_n(
        self,
        calibration_stats: dict[str, np.ndarray],
        transitions_payload: dict[str, np.ndarray],
    ) -> None:
        """Sum of per-stage counts must equal N (total transitions).
        # docs/specs/phase_I_*.md S7.2
        """
        total = calibration_stats["count"].sum()
        expected = transitions_payload["t"].shape[0]
        assert total == expected, f"count sum {total} != N={expected}"

    def test_reward_mean_matches_raw(
        self,
        calibration_stats: dict[str, np.ndarray],
        transitions_payload: dict[str, np.ndarray],
    ) -> None:
        """Per-stage reward mean matches manual computation from transitions.
        # docs/specs/phase_I_*.md S7.2 / S8.4
        """
        t_arr = transitions_payload["t"]
        reward = transitions_payload["reward"]
        for stage in range(_HORIZON):
            mask = t_arr == stage
            if mask.sum() == 0:
                continue
            expected_mean = np.mean(reward[mask])
            np.testing.assert_allclose(
                calibration_stats["reward_mean"][stage],
                expected_mean,
                rtol=1e-12,
                atol=0,
                err_msg=f"reward_mean mismatch at stage {stage}",
            )

    def test_margin_quantiles_correct(
        self,
        calibration_stats: dict[str, np.ndarray],
        transitions_payload: dict[str, np.ndarray],
    ) -> None:
        """margin_q50 at each stage matches np.median of raw margins.
        # docs/specs/phase_I_*.md S7.2 / S8.4
        """
        t_arr = transitions_payload["t"]
        margin = transitions_payload["margin_beta0"]
        for stage in range(_HORIZON):
            mask = t_arr == stage
            if mask.sum() == 0:
                continue
            expected_median = np.median(margin[mask])
            np.testing.assert_allclose(
                calibration_stats["margin_q50"][stage],
                expected_median,
                rtol=1e-12,
                atol=0,
                err_msg=f"margin_q50 mismatch at stage {stage}",
            )

    def test_pos_neg_margin_nonnegative(
        self, calibration_stats: dict[str, np.ndarray]
    ) -> None:
        """pos_margin_mean and neg_margin_mean must be >= 0 at populated stages.
        # docs/specs/phase_I_*.md S7.2
        """
        for stage in range(_HORIZON):
            if calibration_stats["count"][stage] == 0:
                continue
            assert calibration_stats["pos_margin_mean"][stage] >= 0.0, (
                f"pos_margin_mean negative at stage {stage}"
            )
            assert calibration_stats["neg_margin_mean"][stage] >= 0.0, (
                f"neg_margin_mean negative at stage {stage}"
            )

    def test_bellman_nan_when_not_provided(
        self, calibration_stats: dict[str, np.ndarray]
    ) -> None:
        """bellman_residual_mean/std must be NaN when bellman_residuals=None.
        # docs/specs/phase_I_*.md S7.2
        """
        assert np.all(np.isnan(calibration_stats["bellman_residual_mean"])), (
            "bellman_residual_mean should be all-NaN when not provided"
        )
        assert np.all(np.isnan(calibration_stats["bellman_residual_std"])), (
            "bellman_residual_std should be all-NaN when not provided"
        )

    def test_terminal_stage_has_zero_count(
        self, calibration_stats: dict[str, np.ndarray]
    ) -> None:
        """Stage H (terminal) has count=0 since no transitions start there.
        # docs/specs/phase_I_*.md S7.2
        """
        # Our fixture only has stages 0..H-1, so stage H must have count 0
        assert calibration_stats["count"][_HORIZON] == 0, (
            f"Terminal stage {_HORIZON} should have count=0"
        )


# ===========================================================================
# TestRunWriterCalibrationRoundtrip
# ===========================================================================


class TestRunWriterCalibrationRoundtrip:
    """Verify RunWriter persistence of transitions and calibration stats."""

    def _make_writer(self, tmp_path: Path) -> RunWriter:
        """Create a RunWriter pointing at a temporary directory."""
        return RunWriter.create(
            base=tmp_path,
            phase="phase1",
            suite="test",
            task="chain_test",
            algorithm="TestAlgo",
            seed=0,
            config={"gamma": _GAMMA, "horizon": _HORIZON},
            storage_mode="rl_online",
        )

    def test_roundtrip_transitions(
        self,
        tmp_path: Path,
        transitions_payload: dict[str, np.ndarray],
    ) -> None:
        """Write transitions via RunWriter.set_transitions + flush, then validate.
        # docs/specs/phase_I_*.md S7.4
        """
        rw = self._make_writer(tmp_path)
        rw.set_transitions(transitions_payload)
        rw.flush(metrics={"test": 1.0})

        npz_path = rw.run_dir / "transitions.npz"
        assert npz_path.exists(), "transitions.npz not written"

        missing = validate_transitions_npz(npz_path)
        assert missing == [], f"Missing keys after round-trip: {missing}"

        loaded = load_npz(npz_path)
        for key in TRANSITIONS_ARRAYS:
            np.testing.assert_array_equal(
                loaded[key],
                transitions_payload[key],
                err_msg=f"Round-trip mismatch for key '{key}'",
            )

    def test_roundtrip_calibration(
        self,
        tmp_path: Path,
        calibration_stats: dict[str, np.ndarray],
    ) -> None:
        """Write calibration via RunWriter.set_calibration_stats + flush, then validate.
        # docs/specs/phase_I_*.md S7.4
        """
        rw = self._make_writer(tmp_path)
        rw.set_calibration_stats(calibration_stats)
        rw.flush(metrics={"test": 1.0})

        npz_path = rw.run_dir / "calibration_stats.npz"
        assert npz_path.exists(), "calibration_stats.npz not written"

        missing = validate_calibration_npz(npz_path)
        assert missing == [], f"Missing keys after round-trip: {missing}"

        loaded = load_npz(npz_path)
        for key in CALIBRATION_ARRAYS:
            np.testing.assert_array_equal(
                loaded[key],
                calibration_stats[key],
                err_msg=f"Round-trip mismatch for key '{key}'",
            )

    def test_margin_formula_stamped_in_schema(
        self,
        tmp_path: Path,
        transitions_payload: dict[str, np.ndarray],
    ) -> None:
        """transitions.npz schema header must contain margin_beta0_formula key.
        # docs/specs/phase_I_*.md S7.1
        """
        rw = self._make_writer(tmp_path)
        rw.set_transitions(transitions_payload)
        rw.flush(metrics={"test": 1.0})

        loaded = load_npz(rw.run_dir / "transitions.npz")
        assert "_schema" in loaded, "Schema header missing from transitions.npz"

        schema = json.loads(bytes(loaded["_schema"]).decode("utf-8"))
        assert "margin_beta0_formula" in schema, (
            "margin_beta0_formula not stamped in schema header"
        )
        assert schema["margin_beta0_formula"] == MARGIN_BETA0_FORMULA, (
            f"Formula mismatch: {schema['margin_beta0_formula']!r} "
            f"!= {MARGIN_BETA0_FORMULA!r}"
        )


def test_build_calibration_stats_from_dp_tables_accepts_planner_q_shape() -> None:
    """DP planners expose Q as (H, S, A) and V as (H+1, S); stats builder must accept that."""
    H, S, A = 3, 4, 2
    rng = np.random.RandomState(0)
    Q = rng.randn(H, S, A)
    V = rng.randn(H + 1, S)
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            w = rng.rand(S)
            P[s, a] = w / w.sum()
    R = rng.randn(S, A, S)

    out = build_calibration_stats_from_dp_tables(
        Q=Q, V=V, P=P, R=R, gamma=0.99, horizon=H,
    )

    assert set(out.keys()) == set(CALIBRATION_ARRAYS)
    assert out["stage"].shape == (H + 1,)
    assert out["count"][H] == 0
    assert np.isnan(out["reward_mean"][H])
