"""Phase VIII Stage 1 fixed-β-sweep runner smoke test (M6 wave 1).

Drives the dispatcher end-to-end on a minimal in-memory config (1 game ×
1 subcase × 1 method × 2 seeds × 50 episodes) and asserts:

(a) the raw run dirs land under ``<output_root>/raw/`` (lessons.md #11
    + spec §13.6 result-root regression);
(b) ``run.json`` carries the Phase VIII required keys
    (``config``, ``env``, ``seed``, ``git_sha``, ``phase = "VIII"``,
    ``stage = "stage1_beta_sweep"``, ``method``, ``game``, ``subcase``);
(c) ``metrics.npz`` opens cleanly and contains per-episode arrays for
    ``return``, ``bellman_residual``, ``beta_used``, ``beta_raw``,
    ``alignment_rate``, ``effective_discount_mean``;
(d) ``Phase8RunRoster`` registered every dispatched run with
    ``status == "completed"``.

Budget: < 30 seconds total (2 × 50-episode matching-pennies cells run
in well under a second on the dev box).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from experiments.adaptive_beta.tab_six_games.manifests import (
    Phase8RunRoster,
)
from experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage1_beta_sweep import (  # noqa: E501
    REQUIRED_METRICS,
    dispatch,
)


REQUIRED_RUN_JSON_KEYS = (
    "config",
    "env",
    "seed",
    "git_sha",
    "phase",
    "stage",
    "method",
    "game",
    "subcase",
)


def _make_smoke_config(n_episodes: int = 50) -> Dict[str, Any]:
    """Build a minimal in-memory Stage 1 config.

    Uses ``matching_pennies`` × ``finite_memory_best_response`` because:

    * matching_pennies has a 2-action payoff matrix exposed at module
      level, so :func:`_resolve_payoff_opponent` finds it without
      requiring user-supplied ``probs``;
    * ``finite_memory_best_response`` accepts the auto-injected
      ``payoff_opponent`` kwarg, exercising the runner's signature-driven
      adversary builder.

    The combination keeps the smoke under a second on the dev box while
    exercising every Phase VIII contract this test cares about.
    """
    return {
        "stage": "stage1_beta_sweep",
        "phase": "VIII",
        "episodes": int(n_episodes),
        "seeds": [0, 1],
        "methods": ["fixed_beta_+1"],
        "gamma": 0.95,
        "learning_rate": 0.1,
        "q_init": 0.0,
        "epsilon": {
            "start": 1.0,
            "end": 0.05,
            "decay_episodes": max(1, n_episodes // 2),
        },
        "subcases": [
            {
                "id": "MP-FiniteMemoryBR",
                "game": "matching_pennies",
                "game_kwargs": {"horizon": 5},
                "adversary": "finite_memory_best_response",
                "adversary_kwargs": {
                    "memory_m": 10,
                    "inertia_lambda": 0.5,
                    "temperature": 0.2,
                },
                "headline_metric": "cumulative_return_auc",
                "t11_guard": "cohens_d",
            }
        ],
    }


@pytest.mark.smoke
def test_runner_smoke_writes_phase8_artifacts(tmp_path: Path) -> None:
    """End-to-end smoke for the Phase VIII Stage 1 dispatcher.

    The test writes to a pytest-managed tmp_path so the canonical
    ``results/adaptive_beta/tab_six_games/`` tree is never polluted.
    The runner is configured to point at ``tmp_path / "phase8_root"``
    via the ``output_root`` argument.
    """
    n_episodes = 50
    config = _make_smoke_config(n_episodes=n_episodes)
    output_root = tmp_path / "phase8_root"

    roster = dispatch(
        config=config,
        seed_override=None,
        output_root=output_root,
        config_path=None,
        fail_fast=True,
    )

    # ---- (d) Phase8RunRoster registered every requested run.
    assert isinstance(roster, Phase8RunRoster), (
        f"dispatch() must return a Phase8RunRoster, got {type(roster)!r}"
    )
    n_methods = len(config["methods"])
    n_seeds = len(config["seeds"])
    n_subcases = len(config["subcases"])
    expected_runs = n_methods * n_seeds * n_subcases
    assert len(roster) == expected_runs, (
        f"roster must hold {expected_runs} rows, got {len(roster)}"
    )
    statuses = {row.status for row in roster.rows}
    assert statuses == {"completed"}, (
        f"every smoke row must complete; got status set {statuses!r}"
    )

    # Roster snapshot file is also written on disk (incremental
    # write_atomic).
    manifest_path = output_root / "raw" / "VIII" / config["stage"] / "manifest.jsonl"
    assert manifest_path.exists(), (
        f"manifest snapshot missing at {manifest_path}"
    )

    raw_root = output_root / "raw"

    for row in roster.rows:
        run_dir = Path(row.result_path)

        # ---- (a) raw run dir lives under <output_root>/raw/.
        assert raw_root in run_dir.parents, (
            f"run dir {run_dir} must live under {raw_root}"
        )
        # And not under the canonical weighted_lse_dp tree (lessons.md #11).
        assert "weighted_lse_dp" not in run_dir.as_posix(), (
            f"run dir {run_dir} accidentally inherits weighted_lse_dp root"
        )

        # ---- (b) run.json schema header.
        run_json_path = run_dir / "run.json"
        assert run_json_path.exists(), f"missing run.json under {run_dir}"
        with open(run_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for key in REQUIRED_RUN_JSON_KEYS:
            assert key in payload, (
                f"run.json missing required key {key!r} (path={run_json_path})"
            )

        assert payload["phase"] == "VIII", (
            f"run.json phase must equal 'VIII', got {payload['phase']!r}"
        )
        assert payload["stage"] == "stage1_beta_sweep", (
            f"run.json stage must equal 'stage1_beta_sweep', got "
            f"{payload['stage']!r}"
        )
        assert payload["method"] == "fixed_beta_+1", payload["method"]
        assert payload["game"] == "matching_pennies", payload["game"]
        assert payload["subcase"] == "MP-FiniteMemoryBR", payload["subcase"]
        # config must be a dict with both 'path' and 'hash' fields per
        # the runner's logging contract.
        assert isinstance(payload["config"], dict), payload["config"]
        assert "hash" in payload["config"], payload["config"]
        # env must be a dict with the runtime cardinalities.
        assert isinstance(payload["env"], dict), payload["env"]
        assert "n_states" in payload["env"], payload["env"]
        assert "n_actions" in payload["env"], payload["env"]
        # seed is an integer that must match the roster row.
        assert isinstance(payload["seed"], int), payload["seed"]
        assert payload["seed"] == int(row.seed)
        # git_sha is at least a non-empty string (or "unknown" outside
        # a git checkout).
        assert isinstance(payload["git_sha"], str), payload["git_sha"]
        assert payload["git_sha"], "git_sha must be a non-empty string"
        # The episode-budget round trips.
        assert payload["episodes"] == n_episodes

        # Single-γ (Tier I) backwards-compat: the run dir MUST NOT
        # contain a ``gamma_<value>/`` segment in its path. The γ-sweep
        # outer loop is the only path that injects this segment.
        assert "gamma_" not in run_dir.as_posix(), (
            f"single-γ Tier I path must not contain a gamma_<value> "
            f"segment, got {run_dir.as_posix()!r}"
        )

        # ---- (c) metrics.npz opens with all required arrays.
        metrics_path = run_dir / "metrics.npz"
        assert metrics_path.exists(), f"missing metrics.npz under {run_dir}"
        with np.load(metrics_path, allow_pickle=False) as data:
            on_disk = set(data.keys())
            for name in REQUIRED_METRICS:
                assert name in on_disk, (
                    f"metrics.npz missing required array {name!r}; on-disk "
                    f"keys: {sorted(on_disk)} (path={metrics_path})"
                )
                arr = data[name]
                # Per-episode shape (n_episodes,).
                assert arr.shape == (n_episodes,), (
                    f"array {name!r} has shape {arr.shape}, expected "
                    f"({n_episodes},) (path={metrics_path})"
                )
            # Schema header round-trips (the saver writes _schema as
            # uint8 JSON bytes; we should be able to decode it).
            assert "_schema" in on_disk, "metrics.npz missing _schema header"
            schema_bytes = bytes(data["_schema"])
            schema = json.loads(schema_bytes.decode("utf-8"))
            assert schema.get("phase") == "VIII", schema
            assert schema.get("schema_version") == "phaseVIII.metrics.v1", schema


def _make_smoke_config_multi_gamma(n_episodes: int = 50) -> Dict[str, Any]:
    """Minimal v10 multi-γ Stage 1 config: 2 γ × 1 cell × 1 method × 1 seed.

    Same game/subcase/adversary as the single-γ smoke (matching_pennies ×
    finite_memory_best_response), but adds a ``gamma_grid`` outer loop
    of two values (0.60, 0.95) chosen to exercise the v10 §6.4 grid
    edges. The scalar ``gamma`` field is intentionally omitted so the
    test verifies that ``gamma_grid`` alone is sufficient.
    """
    return {
        "stage": "stage1_beta_sweep",
        "phase": "VIII",
        "episodes": int(n_episodes),
        "seeds": [0],
        "methods": ["fixed_beta_+1"],
        "gamma_grid": [0.60, 0.95],
        "learning_rate": 0.1,
        "q_init": 0.0,
        "epsilon": {
            "start": 1.0,
            "end": 0.05,
            "decay_episodes": max(1, n_episodes // 2),
        },
        "subcases": [
            {
                "id": "MP-FiniteMemoryBR",
                "game": "matching_pennies",
                "game_kwargs": {"horizon": 5},
                "adversary": "finite_memory_best_response",
                "adversary_kwargs": {
                    "memory_m": 10,
                    "inertia_lambda": 0.5,
                    "temperature": 0.2,
                },
                "headline_metric": "cumulative_return_auc",
                "t11_guard": "cohens_d",
            }
        ],
    }


@pytest.mark.smoke
def test_runner_smoke_multi_gamma(tmp_path: Path) -> None:
    """v10 Tier II/III smoke: ``gamma_grid`` produces per-γ run dirs.

    Asserts:

    (a) two run dirs are created at ``gamma_0.60/`` and ``gamma_0.95/``
        (same (game, subcase, method, seed); only γ differs);
    (b) each ``run.json["gamma"]`` matches its path segment;
    (c) each ``metrics.npz["gamma"]`` is a 0-dim scalar equal to the
        ``run.json`` γ;
    (d) ``Phase8RunRoster`` holds two rows with distinct γ values and
        distinct cell_ids (γ disambiguates the cell tuple).
    """
    n_episodes = 50
    config = _make_smoke_config_multi_gamma(n_episodes=n_episodes)
    output_root = tmp_path / "phase8_root"

    roster = dispatch(
        config=config,
        seed_override=None,
        output_root=output_root,
        config_path=None,
        fail_fast=True,
    )

    # ---- (d) roster cardinality and γ-distinct rows.
    assert isinstance(roster, Phase8RunRoster), type(roster)
    assert len(roster) == 2, (
        f"multi-γ smoke must produce 2 roster rows (1 γ_a + 1 γ_b), "
        f"got {len(roster)}"
    )
    gamma_values = sorted(float(row.gamma) for row in roster.rows)
    assert gamma_values == [0.60, 0.95], gamma_values
    cell_ids = {row.cell_id for row in roster.rows}
    assert len(cell_ids) == 2, (
        f"γ must disambiguate cell_ids; got duplicate set {cell_ids!r}"
    )
    statuses = {row.status for row in roster.rows}
    assert statuses == {"completed"}, statuses

    raw_root = output_root / "raw"
    seen_gamma_segments: set[str] = set()

    for row in roster.rows:
        run_dir = Path(row.result_path)

        # ---- (a) run dirs land under ``raw/...gamma_<g>/...``.
        assert raw_root in run_dir.parents, run_dir
        # The ``gamma_<value>/`` segment must appear exactly once in the
        # path (between the subcase dir and the method dir).
        path_parts = run_dir.parts
        gamma_segments = [p for p in path_parts if p.startswith("gamma_")]
        assert len(gamma_segments) == 1, (
            f"expected exactly one gamma_<value> path segment, got "
            f"{gamma_segments!r} (path={run_dir.as_posix()})"
        )
        seg = gamma_segments[0]
        assert seg in {"gamma_0.60", "gamma_0.95"}, seg
        seen_gamma_segments.add(seg)

        # ---- (b) run.json.gamma matches the path segment.
        run_json_path = run_dir / "run.json"
        assert run_json_path.exists(), run_json_path
        with open(run_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        run_gamma = float(payload["gamma"])
        expected_seg = f"gamma_{run_gamma:.2f}"
        assert seg == expected_seg, (
            f"path segment {seg!r} does not match run.json gamma "
            f"{run_gamma} (expected segment {expected_seg!r})"
        )
        # Roster row γ also matches the run.json γ.
        assert float(row.gamma) == run_gamma, (run_gamma, row.gamma)

        # ---- (c) metrics.npz['gamma'] is a 0-dim scalar equal to run.json.
        metrics_path = run_dir / "metrics.npz"
        assert metrics_path.exists(), metrics_path
        with np.load(metrics_path, allow_pickle=False) as data:
            assert "gamma" in data.files, sorted(data.files)
            arr = data["gamma"]
            assert arr.ndim == 0, (
                f"metrics.npz['gamma'] must be a 0-dim scalar, got "
                f"shape {arr.shape}"
            )
            assert float(arr) == run_gamma, (float(arr), run_gamma)

    # Both γ segments must have been observed (one each).
    assert seen_gamma_segments == {"gamma_0.60", "gamma_0.95"}, (
        seen_gamma_segments
    )
