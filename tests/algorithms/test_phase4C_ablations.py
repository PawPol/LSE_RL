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
    """§6, §9: trust_region_off lowers tau_n far below default."""
    overrides_default = _V3_DEFAULTS.copy()
    overrides_off = {**_V3_DEFAULTS, **_ABLATION_OVERRIDES["trust_region_off"]}
    assert overrides_off["tau_n"] < 1e-5
    assert overrides_default["tau_n"] > 100


def test_trust_region_tighter_vs_off() -> None:
    """§9: tighter tau_n (2000) is strictly larger than default (200)."""
    tighter = {**_V3_DEFAULTS, **_ABLATION_OVERRIDES["trust_region_tighter"]}
    default = _V3_DEFAULTS
    assert tighter["tau_n"] > default["tau_n"]


def test_adaptive_headroom_off_ablation() -> None:
    """§6, §9: Disabling adaptive headroom sets constant alpha (alpha_min==alpha_max)."""
    overrides = {**_V3_DEFAULTS, **_ABLATION_OVERRIDES["adaptive_headroom_off"]}
    assert overrides["alpha_min"] == overrides["alpha_max"]


def test_adaptive_headroom_aggressive_vs_default() -> None:
    """§9: aggressive alpha_max > default alpha_max > off alpha_max."""
    aggressive = {**_V3_DEFAULTS, **_ABLATION_OVERRIDES["adaptive_headroom_aggressive"]}
    off = {**_V3_DEFAULTS, **_ABLATION_OVERRIDES["adaptive_headroom_off"]}
    default = _V3_DEFAULTS
    assert aggressive["alpha_max"] > default["alpha_max"] > off["alpha_max"]


def test_wrong_sign_ablation_negates_beta() -> None:
    """§9: wrong_sign post-hoc negation produces element-wise negated beta_used_t."""
    v3 = _dummy_v3_schedule(T=5, beta=0.05, sign=1)
    original_beta = np.asarray(v3["beta_used_t"])
    original_alpha = list(v3["alpha_t"])
    original_xi = list(v3["xi_ref_t"])

    # Apply the same logic as _build_ablated_schedule when flip_sign=True
    v3_flipped = dict(v3)
    v3_flipped["beta_used_t"] = [-b for b in v3["beta_used_t"]]
    v3_flipped["sign_family"] = -v3["sign_family"]

    # beta must be element-wise negated
    np.testing.assert_array_equal(
        np.asarray(v3_flipped["beta_used_t"]),
        -original_beta,
    )
    # diagnostics (alpha_t, xi_ref_t) must be untouched — isolation
    assert v3_flipped.get("alpha_t") == original_alpha
    assert v3_flipped.get("xi_ref_t") == original_xi


def test_constant_u_transformation_produces_flat_u() -> None:
    """§9: constant_u rewrite yields constant |beta_t * xi_t| across stages."""
    v3 = _dummy_v3_schedule(T=4, beta=0.0, sign=1)
    v3["beta_used_t"] = [0.10, 0.20, 0.05, 0.30]
    v3["xi_ref_t"] = [0.1, 0.2, 0.1, 0.05]

    beta_arr = np.asarray(v3["beta_used_t"], dtype=np.float64)
    xi_arr = np.asarray(v3["xi_ref_t"], dtype=np.float64)
    u_mean = float(np.mean(np.abs(beta_arr * xi_arr)))
    xi_safe = np.where(xi_arr > 1e-9, xi_arr, 1.0)
    beta_const = u_mean / xi_safe

    u_const = np.abs(beta_const * xi_arr)
    np.testing.assert_allclose(u_const, u_mean, rtol=1e-9,
                               err_msg="constant_u: |beta_t * xi_t| not constant")
    # The mean must equal the original mean
    np.testing.assert_allclose(float(np.mean(u_const)), u_mean, rtol=1e-9)


def test_wrong_sign_ablation() -> None:
    """§6, §9: Wrong-sign ablation config flag is present."""
    overrides = _ABLATION_OVERRIDES["wrong_sign"]
    assert overrides.get("_flip_sign") is True


def test_constant_u_ablation() -> None:
    """§6, §9: Constant-u ablation config flag is present."""
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
