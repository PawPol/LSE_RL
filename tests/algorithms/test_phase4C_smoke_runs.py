"""
Phase IV-C §9: End-to-end smoke runs for advanced experiments.

Short runs verifying that advanced RL, geometry-prioritized DP, scheduler
ablation, and certification ablation pipelines execute without errors.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
_SRC = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _MUSHROOM_DEV, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _tiny_chain_cfg() -> dict:
    return {
        "family": "dense_chain_cost",
        "n_states": 5,
        "horizon": 5,
        "gamma": 0.95,
        "step_reward": -0.02,
        "terminal_reward": 1.0,
        "reward_bound": 1.0,
    }


def test_advanced_rl_smoke() -> None:
    """§9: Advanced RL experiment (SafeDoubleQ) completes on a tiny MDP."""
    from experiments.weighted_lse_dp.runners.run_phase4C_advanced_rl import run_single

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_single(
            algorithm="safe_double_q",
            task_tag="dense_chain_cost_smoke",
            cfg=_tiny_chain_cfg(),
            seed=42,
            train_steps=100,
            checkpoint_every=50,
            eval_episodes=5,
            n_pilot=10,
            out_root=Path(tmpdir),
        )
    assert result["status"] == "pass", f"FAIL: {result.get('error', '')}"
    assert "final_mean_return" in result


def test_geometry_dp_smoke() -> None:
    """§9: Geometry-prioritized async DP completes on a tiny MDP."""
    from experiments.weighted_lse_dp.runners.run_phase4C_geometry_dp import run_single

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_single(
            task_tag="dense_chain_cost_smoke",
            family="dense_chain_cost",
            cfg=_tiny_chain_cfg(),
            seed=42,
            mode="combined",
            out_root=Path(tmpdir),
            n_pilot=10,
        )
    assert result["status"] == "pass", f"FAIL: {result.get('error', '')}"
    assert result["final_residual"] < 1e-4


def test_scheduler_ablation_smoke() -> None:
    """§9: State-bin scheduler ablation sweep completes without error."""
    from experiments.weighted_lse_dp.runners.run_phase4C_scheduler_ablations import run_single

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_single(
            scheduler_type="stagewise_baseline",
            task_tag="dense_chain_cost_smoke",
            cfg=_tiny_chain_cfg(),
            seed=42,
            train_steps=100,
            checkpoint_every=50,
            eval_episodes=5,
            n_pilot=10,
            out_root=Path(tmpdir),
        )
    assert result["status"] == "pass", f"FAIL: {result.get('error', '')}"


def test_certification_ablation_smoke() -> None:
    """§9: Certification ablation suite completes without error."""
    from experiments.weighted_lse_dp.runners.run_phase4C_certification_ablations import run_single

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_single(
            ablation="trust_region_off",
            task_tag="dense_chain_cost_smoke",
            cfg=_tiny_chain_cfg(),
            seed=42,
            train_steps=100,
            checkpoint_every=50,
            eval_episodes=5,
            n_pilot_episodes=10,
            out_root=Path(tmpdir),
        )
    assert result["status"] == "pass", f"FAIL: {result.get('error', '')}"
