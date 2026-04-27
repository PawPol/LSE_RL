"""End-to-end smoke tests for ``run_strategic`` (Phase VII-B spec §13–§14).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§§13 (logging schema), 14 (output directory contract), 22.3 (canonical-sign
skip rule).

Invariants guarded
------------------
- A 1-cell smoke run writes ``manifest.json``, ``run.json``, ``episodes.csv``,
  ``metrics.npz``, and either ``transitions.parquet`` or a CSV fallback.
- Manifest schema: ``schema_version, stage, records[]`` with required fields.
- Skip path: ``wrong_sign × matching_pennies`` (canonical sign None) records
  ``status="skipped"`` with reason text containing ``"§22.3"``.
- Placeholder guard: dispatching ``stage_B2_main.yaml`` (with ``${promoted_games}``
  literal) raises ``ValueError``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from experiments.adaptive_beta.strategic_games.run_strategic import (
    MANIFEST_SCHEMA_VERSION,
    dispatch_from_config,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEV_CONFIG = _REPO_ROOT / "experiments/adaptive_beta/strategic_games/configs/stage_B2_dev.yaml"
_MAIN_CONFIG = _REPO_ROOT / "experiments/adaptive_beta/strategic_games/configs/stage_B2_main.yaml"


def _load_dev_config(episodes: int = 5) -> Dict[str, Any]:
    """Load the Stage B2-Dev YAML and shrink ``episodes`` for smoke runs."""
    with open(_DEV_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["episodes"] = int(episodes)
    return cfg


# ---------------------------------------------------------------------------
# 1-cell smoke
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_runner_smoke_single_cell_writes_required_artifacts(tmp_path: Path) -> None:
    """`spec §13–§14` — 1-cell dispatch writes the four artifact files +
    a manifest.
    """
    cfg = _load_dev_config(episodes=5)
    out_dir = tmp_path / "raw"
    records = dispatch_from_config(cfg, output_dir=out_dir, limit_cells=1)
    assert len(records) == 1
    rec = records[0]
    assert rec["status"] == "completed", f"smoke run did not complete: {rec}"

    cell_dir = (
        out_dir
        / f"{rec['game']}_{rec['adversary']}_{rec['method']}"
        / f"seed_{rec['seed_id']}"
    )
    assert cell_dir.is_dir()
    # Required artifacts.
    assert (cell_dir / "run.json").exists()
    assert (cell_dir / "episodes.csv").exists()
    assert (cell_dir / "metrics.npz").exists()
    # Either parquet (preferred) or CSV fallback.
    has_parquet = (cell_dir / "transitions.parquet").exists()
    has_csv = (cell_dir / "transitions.csv").exists()
    assert has_parquet or has_csv, (
        "no transitions.parquet or transitions.csv in cell directory"
    )
    # Manifest at output_dir root.
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_runner_manifest_schema_fields(tmp_path: Path) -> None:
    """`spec §14` — manifest.json has top-level ``schema_version``, ``stage``,
    ``records`` (list); each record has the required identity fields.
    """
    cfg = _load_dev_config(episodes=3)
    out_dir = tmp_path / "raw"
    dispatch_from_config(cfg, output_dir=out_dir, limit_cells=1)
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["stage"] == "B2_dev"
    assert isinstance(manifest["records"], list) and len(manifest["records"]) == 1
    rec = manifest["records"][0]
    for key in (
        "cell_id", "stage", "game", "adversary", "method",
        "seed_id", "status", "started_at", "completed_at",
    ):
        assert key in rec, f"manifest record missing required key {key!r}"


# ---------------------------------------------------------------------------
# Skip path: wrong_sign × matching_pennies
# ---------------------------------------------------------------------------
def test_runner_skip_path_wrong_sign_records_skipped_status(tmp_path: Path) -> None:
    """`spec §22.3` — ``wrong_sign × matching_pennies`` is recorded with
    ``status='skipped'`` and a reason mentioning ``§22.3``.
    """
    cfg = {
        "stage": "B2_dev",
        "episodes": 3,
        "seeds": [0],
        "games": {"matching_pennies": {"horizon": 1}},
        "adversaries": {"stationary": {"probs": [0.5, 0.5]}},
        "methods": ["wrong_sign"],
        "beta0": 1.0,
        "gamma": 0.95,
        "learning_rate": 0.1,
        "epsilon_start": 0.5,
        "epsilon_end": 0.05,
        "epsilon_decay_episodes": 3,
        "stratify_every": 1,
    }
    out_dir = tmp_path / "raw"
    records = dispatch_from_config(cfg, output_dir=out_dir)
    assert len(records) == 1
    rec = records[0]
    assert rec["status"] == "skipped"
    assert "§22.3" in (rec.get("reason") or ""), (
        f"reason missing §22.3 reference: {rec.get('reason')!r}"
    )
    assert rec.get("raw_dir") is None


# ---------------------------------------------------------------------------
# Placeholder guard
# ---------------------------------------------------------------------------
def test_runner_rejects_unfilled_main_config_placeholder(tmp_path: Path) -> None:
    """`spec §11.2 / runner` — placeholder strings are rejected by
    ``_normalise_blocks`` regardless of whether they live in the live Main
    config file. We synthesize the placeholder rather than reading
    `stage_B2_main.yaml` because the live config is overwritten in-place by
    the promotion gate (per its own header comment).
    """
    cfg = _load_dev_config(episodes=3)
    cfg["stage"] = "B2_main"
    cfg["games"] = "${promoted_games}"
    cfg["adversaries"] = "${promoted_adversaries}"
    out_dir = tmp_path / "raw"
    with pytest.raises(ValueError, match="placeholder"):
        dispatch_from_config(cfg, output_dir=out_dir, limit_cells=1)


# ---------------------------------------------------------------------------
# Tripwire
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_invariant_smoke_run_metrics_npz_is_loadable(tmp_path: Path) -> None:
    """`spec §13` — ``metrics.npz`` is a structured archive loadable by numpy.

    A regression that wrote a non-loadable / non-positional file would
    silently break the aggregator and any downstream paper figures.
    """
    cfg = _load_dev_config(episodes=3)
    out_dir = tmp_path / "raw"
    records = dispatch_from_config(cfg, output_dir=out_dir, limit_cells=1)
    rec = records[0]
    cell_dir = (
        out_dir
        / f"{rec['game']}_{rec['adversary']}_{rec['method']}"
        / f"seed_{rec['seed_id']}"
    )
    archive = np.load(cell_dir / "metrics.npz", allow_pickle=False)
    # At minimum ``return`` and ``episode`` columns from the parent
    # EpisodeLogger should be present as named arrays.
    keys = set(archive.files)
    assert "schema_version" in keys
    assert any(k.endswith("return") or k == "episode_return" for k in keys), (
        f"metrics.npz missing a return-style array; keys={sorted(keys)}"
    )
