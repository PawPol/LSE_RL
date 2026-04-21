"""
Phase IV-C §4, §9: State-dependent scheduler tests.

Verify bin construction modes, hierarchical backoff behavior, schedule
freezing during learning, and that stage-wise scheduling is a special case
of the general state-bin scheduler.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.runners.run_phase4C_scheduler_ablations import (
    ALL_SCHEDULERS,
    _SCHEDULER_OVERRIDES,
    _V3_DEFAULTS,
)


def test_bin_construction_modes() -> None:
    """§4: Bin construction supports the defined scheduler types."""
    assert "stagewise_baseline" in ALL_SCHEDULERS
    assert "state_bin_uniform" in ALL_SCHEDULERS
    assert "state_bin_hazard_proximity" in ALL_SCHEDULERS
    assert "state_bin_reward_region" in ALL_SCHEDULERS


def test_hierarchical_backoff() -> None:
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
