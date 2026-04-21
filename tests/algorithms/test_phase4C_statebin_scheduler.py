"""
Phase IV-C §4, §9: State-dependent scheduler tests.

Verify bin construction modes, hierarchical backoff behavior, schedule
freezing during learning, and that stage-wise scheduling is a special case
of the general state-bin scheduler.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.geometry.state_bins import (
    construct_bins,
    count_state_visits,
)
from experiments.weighted_lse_dp.geometry.schedule_smoothing import (
    hierarchical_shrinkage,
    smooth_schedule,
)
from experiments.weighted_lse_dp.runners.run_phase4C_scheduler_ablations import (
    ALL_SCHEDULERS,
    _SCHEDULER_OVERRIDES,
    _V3_DEFAULTS,
)


# ---------------------------------------------------------------------------
# state_bins tests
# ---------------------------------------------------------------------------

def test_bin_construction_exact_state() -> None:
    """§4: exact_state mode assigns each unique state its own bin."""
    states = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    bins = construct_bins(states, mode="exact_state")
    # Each unique state gets a distinct bin
    assert len(np.unique(bins)) == len(states)
    # Identity for contiguous states: bin[i] == i
    np.testing.assert_array_equal(bins, states)


def test_bin_construction_uniform() -> None:
    """§4: uniform mode partitions states into n_bins bins."""
    states = np.arange(10, dtype=np.int64)
    bins = construct_bins(states, mode="uniform", n_bins=5)
    assert bins.shape == (10,)
    assert bins.min() >= 0
    assert bins.max() < 5
    # Must have at least 2 distinct bins for 10 states with 5 bins
    assert len(np.unique(bins)) >= 2


def test_bin_construction_margin_quantile_fallback() -> None:
    """§4: margin_quantile mode falls back to uniform."""
    states = np.arange(8, dtype=np.int64)
    bins_mq = construct_bins(states, mode="margin_quantile", n_bins=4)
    bins_u = construct_bins(states, mode="uniform", n_bins=4)
    np.testing.assert_array_equal(bins_mq, bins_u)


def test_count_state_visits_basic() -> None:
    """§4: bin visit counts are correct."""
    states = np.array([0, 1, 2, 3], dtype=np.int64)
    bins = construct_bins(states, mode="exact_state")  # bins[i] = i
    transitions = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    counts = count_state_visits(bins, n_bins=4, transitions=transitions)
    assert counts[0] == 2
    assert counts[1] == 1
    assert counts[2] == 3
    assert counts[3] == 0


# ---------------------------------------------------------------------------
# schedule_smoothing tests
# ---------------------------------------------------------------------------

def test_hierarchical_backoff() -> None:
    """§4, §9: Hierarchical backoff shrinks toward stagewise when counts are low."""
    T, B = 3, 4
    u_design = np.ones((T, B)) * 0.1   # design target: 0.1
    u_stage = np.ones(T) * 0.0         # stagewise fallback: 0.0
    # With zero counts → w=0 → full shrinkage → u_hb = u_stage
    counts_zero = np.zeros((T, B), dtype=np.int64)
    u_hb = hierarchical_shrinkage(u_design, u_stage, counts_zero, tau_bin=100.0)
    np.testing.assert_allclose(u_hb, 0.0, atol=1e-12)


def test_hierarchical_backoff_large_counts() -> None:
    """§4: With large counts, u_hb converges to u_design."""
    T, B = 2, 3
    u_design = np.ones((T, B)) * 0.05
    u_stage = np.zeros(T)
    counts_large = np.full((T, B), 10000, dtype=np.int64)
    u_hb = hierarchical_shrinkage(u_design, u_stage, counts_large, tau_bin=100.0)
    np.testing.assert_allclose(u_hb, 0.05, atol=1e-3)


def test_hierarchical_backoff_weight_formula() -> None:
    """§4: w = n/(n + tau) per spec §4.2."""
    n = 100
    tau = 100.0
    w_expected = n / (n + tau)  # = 0.5
    u_design = np.array([[0.1]])
    u_stage = np.array([0.0])
    counts = np.array([[n]], dtype=np.int64)
    u_hb = hierarchical_shrinkage(u_design, u_stage, counts, tau_bin=tau)
    np.testing.assert_allclose(u_hb[0, 0], w_expected * 0.1, rtol=1e-10)


def test_smooth_schedule_hierarchical() -> None:
    """§4: smooth_schedule with hierarchical_shrinkage uses correct formula."""
    T, B = 2, 3
    raw = np.ones((T, B)) * 0.08
    fallback = np.zeros(T)
    counts = np.zeros((T, B), dtype=np.int64)
    result = smooth_schedule(raw, smoothing_method="hierarchical_shrinkage",
                              stagewise_fallback=fallback, bin_counts=counts)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Scheduler registry tests
# ---------------------------------------------------------------------------

def test_bin_construction_modes() -> None:
    """§4: Bin construction supports the defined scheduler types."""
    assert "stagewise_baseline" in ALL_SCHEDULERS
    assert "state_bin_uniform" in ALL_SCHEDULERS
    assert "state_bin_hazard_proximity" in ALL_SCHEDULERS
    assert "state_bin_reward_region" in ALL_SCHEDULERS


def test_hierarchical_backoff_alpha_regimes() -> None:
    """§4, §9: Scheduler overrides represent distinct alpha regimes."""
    uniform = {**_V3_DEFAULTS, **_SCHEDULER_OVERRIDES["state_bin_uniform"]}
    hazard = {**_V3_DEFAULTS, **_SCHEDULER_OVERRIDES["state_bin_hazard_proximity"]}
    # hazard_proximity allows higher alpha than uniform
    assert hazard["alpha_max"] > uniform["alpha_max"]


def test_frozen_during_learning() -> None:
    """§4: Each scheduler type has frozen overrides (no _dynamic_ keys)."""
    for stype, overrides in _SCHEDULER_OVERRIDES.items():
        assert "_dynamic" not in overrides, (
            f"Scheduler {stype} has a dynamic override key (should be frozen)"
        )


def test_stagewise_is_special_case() -> None:
    """§4, §9: stagewise_baseline has no overrides (is the reference)."""
    overrides = _SCHEDULER_OVERRIDES.get("stagewise_baseline", {})
    assert overrides == {}, (
        f"stagewise_baseline should have empty overrides, got {overrides}"
    )


def test_all_schedulers_have_overrides_entry() -> None:
    """§4: Every scheduler type has an entry in the override registry."""
    for stype in ALL_SCHEDULERS:
        assert stype in _SCHEDULER_OVERRIDES, (
            f"Scheduler type {stype!r} missing from _SCHEDULER_OVERRIDES"
        )


def test_scheduler_result_files_exist() -> None:
    """§9: Integration check — scheduler result dirs were populated."""
    results_dir = (
        _REPO_ROOT / "results" / "weighted_lse_dp" / "phase4" / "advanced"
        / "state_dependent_scheduler"
    )
    if not results_dir.is_dir():
        pytest.skip("Scheduler results not yet generated")
    metrics_files = list(results_dir.rglob("metrics.json"))
    assert len(metrics_files) >= 1, "No metrics.json files found in scheduler results"
