"""Reproducibility tests for the strategic-games runner (Phase VII-B spec §12.3).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§12.3 (rerunning the same config with the same seed produces byte-identical
summary metrics; changing seed changes trajectory).

Invariants guarded
------------------
- Same config + same seed across two independent dispatches produces
  numerically identical ``metrics.npz`` arrays (NaN-aware).
- Changing the seed produces a different trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

from experiments.adaptive_beta.strategic_games.run_strategic import (
    dispatch_from_config,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEV_CONFIG = _REPO_ROOT / "experiments/adaptive_beta/strategic_games/configs/stage_B2_dev.yaml"


def _load_dev_cfg(*, episodes: int) -> Dict[str, Any]:
    with open(_DEV_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["episodes"] = int(episodes)
    return cfg


def _run_metrics(
    cfg: Dict[str, Any], *, output_dir: Path, limit_cells: int,
) -> List[Dict[str, np.ndarray]]:
    """Dispatch the cells under ``output_dir`` and return their metrics arrays."""
    records = dispatch_from_config(
        cfg, output_dir=output_dir, limit_cells=limit_cells,
    )
    metrics: List[Dict[str, np.ndarray]] = []
    for rec in records:
        if rec["status"] != "completed":
            continue
        cell_dir = (
            output_dir
            / f"{rec['game']}_{rec['adversary']}_{rec['method']}"
            / f"seed_{rec['seed_id']}"
        )
        archive = np.load(cell_dir / "metrics.npz", allow_pickle=False)
        metrics.append({k: archive[k] for k in archive.files})
    return metrics


def _assert_arrays_identical_nanaware(
    a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
) -> None:
    """Compare two metrics dicts. NaN entries in the same positions are
    treated as equal; everything else must match exactly.
    """
    assert set(a.keys()) == set(b.keys())
    for key in sorted(a.keys()):
        if a[key].dtype.kind in {"U", "S", "O"}:
            # String arrays: dtype ndarray of objects; compare element-wise.
            if a[key].shape == ():
                assert str(a[key]) == str(b[key]), (
                    f"key {key!r}: scalar string differs: "
                    f"{a[key]!r} vs {b[key]!r}"
                )
            else:
                np.testing.assert_array_equal(a[key], b[key])
            continue
        x = np.asarray(a[key], dtype=np.float64)
        y = np.asarray(b[key], dtype=np.float64)
        if x.shape != y.shape:
            raise AssertionError(
                f"key {key!r}: shape differs ({x.shape} vs {y.shape})"
            )
        nan_x = np.isnan(x)
        nan_y = np.isnan(y)
        if not np.array_equal(nan_x, nan_y):
            raise AssertionError(f"key {key!r}: NaN positions differ")
        # Non-NaN entries must be byte-identical.
        np.testing.assert_array_equal(x[~nan_x], y[~nan_x])


# ---------------------------------------------------------------------------
# Same config + same seed ⇒ identical metrics
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_byte_identical_reruns_same_seed(tmp_path: Path) -> None:
    """`spec §12.3` — running the same config twice produces numerically
    identical ``metrics.npz`` arrays.
    """
    cfg = _load_dev_cfg(episodes=5)
    metrics_a = _run_metrics(cfg, output_dir=tmp_path / "run_a", limit_cells=3)
    metrics_b = _run_metrics(cfg, output_dir=tmp_path / "run_b", limit_cells=3)
    assert len(metrics_a) == len(metrics_b) >= 1
    for m_a, m_b in zip(metrics_a, metrics_b):
        _assert_arrays_identical_nanaware(m_a, m_b)


# ---------------------------------------------------------------------------
# Changing the seed changes the trajectory
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_different_seed_changes_trajectory(tmp_path: Path) -> None:
    """`spec §12.3` — bumping the seed list to a different value must
    change the produced trajectory (otherwise reproducibility is vacuous).
    """
    cfg_a = _load_dev_cfg(episodes=10)
    cfg_a["seeds"] = [0]
    cfg_b = _load_dev_cfg(episodes=10)
    cfg_b["seeds"] = [42]
    metrics_a = _run_metrics(cfg_a, output_dir=tmp_path / "seed_0", limit_cells=1)
    metrics_b = _run_metrics(cfg_b, output_dir=tmp_path / "seed_42", limit_cells=1)
    assert metrics_a and metrics_b
    # At least one float array must differ on a non-NaN entry.
    a, b = metrics_a[0], metrics_b[0]
    differs = False
    for key in a.keys() & b.keys():
        if a[key].dtype.kind in {"U", "S", "O"}:
            continue
        x = np.asarray(a[key], dtype=np.float64)
        y = np.asarray(b[key], dtype=np.float64)
        if x.shape != y.shape:
            continue
        finite_mask = ~(np.isnan(x) | np.isnan(y))
        if np.any(finite_mask) and not np.array_equal(x[finite_mask], y[finite_mask]):
            differs = True
            break
    assert differs, (
        "metrics arrays were identical across seeds 0 and 42; the runner "
        "is not propagating the seed into the agent or env stream"
    )


# ---------------------------------------------------------------------------
# Tripwire
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_invariant_run_json_records_seed_and_git_sha(tmp_path: Path) -> None:
    """`spec §12.3` — the per-cell ``run.json`` records ``seed_id`` and
    ``git_sha``; both are required for reproducibility audits.
    """
    import json
    cfg = _load_dev_cfg(episodes=3)
    records = dispatch_from_config(cfg, output_dir=tmp_path, limit_cells=1)
    rec = records[0]
    cell_dir = (
        tmp_path
        / f"{rec['game']}_{rec['adversary']}_{rec['method']}"
        / f"seed_{rec['seed_id']}"
    )
    with open(cell_dir / "run.json", "r", encoding="utf-8") as f:
        run_json = json.load(f)
    assert "seed_id" in run_json
    assert "git_sha" in run_json
    assert "common_env_seed" in run_json
    assert "agent_seed" in run_json
