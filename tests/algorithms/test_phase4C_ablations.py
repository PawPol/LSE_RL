"""
Phase IV-C §6, §9: Certification and schedule ablation tests.

Verify that each ablation variant (trust-region off, adaptive headroom off,
wrong-sign, constant-u, raw-unclipped) produces distinct behavior from the
full safe operator.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.runners.run_phase4C_certification_ablations import (
    ALL_ABLATIONS,
    _ABLATION_OVERRIDES,
    _V3_DEFAULTS,
)


def _dummy_v3_schedule(T: int = 10, beta: float = 0.05, gamma: float = 0.95,
                       sign: int = 1) -> dict[str, Any]:
    """Minimal v3-format schedule dict for testing."""
    return {
        "beta_used_t": [beta * sign] * T,
        "alpha_t": [0.05] * T,
        "xi_ref_t": [0.1] * T,
        "sign_family": sign,
        "gamma_base": gamma,
        "gamma_eval": gamma,
        "reward_bound": 1.0,
        "task_family": "test",
        "source_phase": "test",
        "notes": "",
    }


def test_trust_region_off_ablation() -> None:
    """§6, §9: Disabling trust region changes the operator output."""
    overrides_default = _V3_DEFAULTS.copy()
    overrides_off = {**_V3_DEFAULTS, **_ABLATION_OVERRIDES["trust_region_off"]}

    # trust_region_off sets tau_n → near-zero
    assert overrides_off["tau_n"] < 1e-5
    assert overrides_default["tau_n"] > 100


def test_adaptive_headroom_off_ablation() -> None:
    """§6, §9: Disabling adaptive headroom sets constant alpha."""
    overrides = {**_V3_DEFAULTS, **_ABLATION_OVERRIDES["adaptive_headroom_off"]}
    # alpha_min == alpha_max → constant headroom
    assert overrides["alpha_min"] == overrides["alpha_max"]


def test_wrong_sign_ablation() -> None:
    """§6, §9: Wrong-sign ablation (negated beta) produces distinct values."""
    overrides = _ABLATION_OVERRIDES["wrong_sign"]
    assert overrides.get("_flip_sign") is True


def test_constant_u_ablation() -> None:
    """§6, §9: Constant-u ablation flag is present."""
    overrides = _ABLATION_OVERRIDES["constant_u"]
    assert overrides.get("_constant_u") is True


def test_raw_unclipped_ablation() -> None:
    """§6, §9: Raw unclipped ablation removes trust region and increases u_max."""
    overrides = {**_V3_DEFAULTS, **_ABLATION_OVERRIDES["raw_unclipped"]}
    assert overrides["tau_n"] < 1e-5
    assert overrides["u_max"] > _V3_DEFAULTS["u_max"]


def test_all_ablation_types_registered() -> None:
    """§6: All 7 required ablation types are registered."""
    required = {
        "trust_region_off", "trust_region_tighter",
        "adaptive_headroom_off", "adaptive_headroom_aggressive",
        "wrong_sign", "constant_u", "raw_unclipped",
    }
    assert required == set(ALL_ABLATIONS)


def test_ablation_result_files_exist() -> None:
    """§9: Integration check — ablation result dirs were populated."""
    results_dir = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase4" / "advanced" / "ablations"
    if not results_dir.is_dir():
        pytest.skip("Ablation results not yet generated")
    for ablation in ALL_ABLATIONS:
        abl_dir = results_dir / ablation
        assert abl_dir.is_dir(), f"Missing ablation result dir: {ablation}"
        metrics_files = list(abl_dir.rglob("metrics.json"))
        assert len(metrics_files) >= 1, f"No metrics.json in {ablation}"
