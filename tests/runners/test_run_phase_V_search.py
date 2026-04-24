"""Phase V WP1c -- integration tests for the search driver.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` sections 4,
7 (WP1c), 8, 13.

Tests (per the orchestrator brief):
1. End-to-end smoke on a tiny synthetic grid (~30 candidates) across
   A/B/C: runner exits cleanly, every spec-section-8 output file is
   created with a non-zero row count, ``experiment_manifest.json``
   contains ``git_sha``, ``exact_argv``, ``seed_list``.
2. Prefilter reduces DP calls by >= 50% on a grid where most candidates
   have ``contest_gap_norm > 0.02``.
3. Promotion gate deterministic on a hand-crafted metrics DataFrame.
4. Empty-shortlist contract: runner emits
   ``shortlist_refinement_manifest.md`` and returns 0 when no row
   promotes.
5. Hard-cap enforcement: raises a clear ``ConfigError`` with a message
   pointing to ``--max-candidates``.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root + vendored MushroomRL must be on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MUSHROOM = _REPO_ROOT / "mushroom-rl-dev"
if _MUSHROOM.exists() and str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from experiments.weighted_lse_dp.runners.run_phase_V_search import (  # noqa: E402
    ConfigError,
    DEFAULT_CONFIG,
    _classical_contest_gap,
    _enumerate_admitted,
    _FAMILY_REGISTRY,
    apply_promotion_gate,
    main,
    run_search,
)


# ---------------------------------------------------------------------------
# Tiny config fixture (reused by smoke + prefilter tests)
# ---------------------------------------------------------------------------

def _tiny_config() -> dict:
    """Narrow grid yielding ~30 (lam, psi) candidates across A/B/C."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["family_params"]["A"] = {
        "L_range": [4],
        "R_range": [2.0],
        "gamma": 0.95,
        "shapes": ["flat"],
        "eps_band_pts": 7,
        "eps_band_frac": 0.2,
    }
    cfg["family_params"]["B"] = {
        "L_range": [4],
        "p_range": [0.10],
        "C_range": [2.0],
        "b": 1.0,
        "variants": ["single_event", "warning_state"],
        "gamma": 0.95,
        "eps_band_pts": 7,
        "eps_band_frac": 0.2,
    }
    cfg["family_params"]["C"] = {
        "L_tail_range": [4],
        "R_penalty_range": [1.0, 2.0],
        "beta_raw_multiplier_range": [2.5, 4.0],
        "gamma": 0.95,
    }
    return cfg


# ---------------------------------------------------------------------------
# 1. End-to-end smoke test
# ---------------------------------------------------------------------------

def test_smoke_end_to_end(tmp_path: Path) -> None:
    """Runner produces every spec-section-8 output with non-zero rows."""
    cfg = _tiny_config()
    out_dir = tmp_path / "out"
    result = run_search(
        cfg,
        output_root=out_dir,
        seed=42,
        exact_argv=["test", "--seed", "42"],
    )

    # spec section 8 output files.
    expected_files = [
        "candidate_grid.parquet",
        "candidate_metrics.parquet",
        "near_indifference_catalog.csv",
        "shortlist.csv",
        "shortlist_report.md",
        "phase_diagram_data.parquet",
        "resolved_config.yaml",
    ]
    for name in expected_files:
        p = out_dir / name
        assert p.is_file(), f"missing output {name}"
        assert p.stat().st_size > 0, f"empty output {name}"

    # parquet files have a non-zero row count.
    grid = pd.read_parquet(out_dir / "candidate_grid.parquet")
    metrics = pd.read_parquet(out_dir / "candidate_metrics.parquet")
    phase = pd.read_parquet(out_dir / "phase_diagram_data.parquet")
    assert len(grid) > 0
    assert len(metrics) > 0
    assert len(phase) > 0
    assert len(grid) == len(metrics) == len(phase)
    assert result["n_admitted"] == len(metrics)

    # experiment_manifest.json contract (spec section 9).
    manifest_path = _REPO_ROOT / "results" / "summaries" / "experiment_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path) as f:
        m = json.load(f)
    for key in ("git_sha", "exact_argv", "seed_list", "schema_version",
                "n_candidates_admitted", "n_candidates_promoted",
                "psi_grid_summary", "family_list", "output_paths"):
        assert key in m, f"manifest missing {key}"
    assert m["seed_list"] == [42]
    assert m["exact_argv"] == ["test", "--seed", "42"]
    # git_sha is either the repo's SHA or the literal "unknown" -- never
    # empty.
    assert isinstance(m["git_sha"], str) and len(m["git_sha"]) > 0


# ---------------------------------------------------------------------------
# 2. Prefilter efficiency (>= 50% reduction)
# ---------------------------------------------------------------------------

def test_prefilter_reduces_dp_calls(tmp_path: Path) -> None:
    """Family A at lam_tie +/- large epsilon should have most candidates > soft band.

    We inflate ``eps_band_frac`` to 2.0 so that the epsilon band covers
    ``[-lam_tie, 3*lam_tie]``, pushing most (lam, psi) pairs far from the
    classical tie and thus far outside ``contest_gap_norm <= 0.02``.
    We then verify that the prefilter admits <= 50% of the enumerated grid
    (a stronger version of the ">= 50% DP-call reduction" assertion).
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["family_params"]["A"] = {
        "L_range": [4, 8],
        "R_range": [2.0],
        "gamma": 0.95,
        "shapes": ["flat"],
        "eps_band_pts": 11,
        "eps_band_frac": 2.0,          # wide -- most points far from tie
    }
    cfg["families"] = ["A"]

    family = _FAMILY_REGISTRY["A"]
    from experiments.weighted_lse_dp.runners.run_phase_V_search import (  # noqa: E402
        _psi_grid_family_a,
    )
    grid = _psi_grid_family_a(cfg["family_params"]["A"])
    n_enumerated = len(grid) * cfg["family_params"]["A"]["eps_band_pts"]

    tie_diag: list[dict] = []
    admitted = _enumerate_admitted(
        "A", family, grid,
        soft_band=float(cfg["prefilter_soft_band"]),
        eps_cfg=cfg["family_params"]["A"],
        tie_diagnostics=tie_diag,
    )
    n_admitted = len(admitted)
    # A >=50% reduction means admitted <= 50% of enumerated.
    assert n_admitted <= 0.5 * n_enumerated, (
        f"prefilter admitted {n_admitted} / {n_enumerated} "
        f"candidates (>50%); the soft-band test is mis-seeded."
    )


# ---------------------------------------------------------------------------
# 3. Promotion-gate determinism
# ---------------------------------------------------------------------------

def _synthesize_metric_row(
    family: str,
    *,
    policy_disagreement: float = 0.0,
    start_state_flip: int = 0,
    mass_delta_d: float = 0.0,
    value_gap_norm: float = 0.0,
    contest_gap_norm: float = 0.0,
    clip_fraction: float = 0.0,
    raw_max: float = 0.0,
    raw_p90: float = 0.0,
    kappa_t_max: float = 0.96,
) -> dict:
    return {
        "family": family,
        "psi_json": "{}",
        "lam": 0.0,
        "lam_tie": 0.0,
        "reward_scale": 1.0,
        "kappa_t_max": float(kappa_t_max),
        "kappa_t_json": "[0.96]",
        "sign_family": 1,
        "horizon": 4,
        "gamma": 0.95,
        "contest_gap_abs": 0.0,
        "contest_gap_norm": float(contest_gap_norm),
        "contest_occupancy_ref": 0.0,
        "margin_pos": 0.0,
        "margin_pos_norm": 0.0,
        "delta_d": 0.0,
        "mass_delta_d": float(mass_delta_d),
        "policy_disagreement": float(policy_disagreement),
        "start_state_flip": int(start_state_flip),
        "value_gap": 0.0,
        "value_gap_norm": float(value_gap_norm),
        "clip_fraction": float(clip_fraction),
        "clip_inactive_fraction": 1.0 - float(clip_fraction),
        "clip_saturation_fraction": 0.0,
        "raw_local_deriv_mean": 0.0,
        "raw_local_deriv_p50": 0.0,
        "raw_local_deriv_p90": float(raw_p90),
        "raw_local_deriv_max": float(raw_max),
        "raw_convergence_status": "not_evaluated",
    }


def test_promotion_gate_deterministic() -> None:
    """Hand-crafted metric rows produce the expected shortlist under the gate."""
    # Row 1: passes the positive-family gate.
    row_pass_a = _synthesize_metric_row(
        "A",
        policy_disagreement=0.10,           # >= 0.05
        mass_delta_d=0.15,                  # >= 0.10
        value_gap_norm=0.01,                # abs >= 0.005
        contest_gap_norm=0.005,             # <= 0.01
        clip_fraction=0.30,                 # in [0.05, 0.80]
        raw_max=0.80, kappa_t_max=0.96,      # cert OK
    )
    # Row 2: fails -- contest_gap_norm = 0.015 (in soft band, above strict).
    row_fail_soft_band = _synthesize_metric_row(
        "A",
        policy_disagreement=0.10, mass_delta_d=0.15, value_gap_norm=0.01,
        contest_gap_norm=0.015,              # FAIL: > strict_band=0.01
        clip_fraction=0.30,
    )
    # Row 3: fails -- clip_fraction = 0.02 (below min).
    row_fail_clip_lo = _synthesize_metric_row(
        "A", policy_disagreement=0.10, mass_delta_d=0.15, value_gap_norm=0.01,
        contest_gap_norm=0.005, clip_fraction=0.02,
    )
    # Row 4: fails -- policy_disagreement below min AND no start_state_flip.
    row_fail_disagreement = _synthesize_metric_row(
        "A", policy_disagreement=0.01, mass_delta_d=0.15, value_gap_norm=0.01,
        contest_gap_norm=0.005, clip_fraction=0.30,
    )
    # Row 5: Family C passes stress gate.
    row_pass_c = _synthesize_metric_row(
        "C", raw_p90=0.97, raw_max=0.97,      # > kappa_t_max=0.96
        clip_fraction=0.80, kappa_t_max=0.96,
    )
    # Row 6: Family C fails stress gate (raw_p90 below kappa).
    row_fail_c = _synthesize_metric_row(
        "C", raw_p90=0.85, raw_max=0.85, clip_fraction=0.80,
        kappa_t_max=0.96,
    )
    # Row 7: Family B start_state_flip promotes even with zero disagreement.
    row_pass_b_flip = _synthesize_metric_row(
        "B", policy_disagreement=0.0, start_state_flip=1,
        mass_delta_d=0.15, value_gap_norm=0.02,
        contest_gap_norm=0.005, clip_fraction=0.30,
    )
    # Row 8: Family A cert violation (raw_max > kappa_t_max + tol).
    row_fail_cert = _synthesize_metric_row(
        "A", policy_disagreement=0.10, mass_delta_d=0.15, value_gap_norm=0.02,
        contest_gap_norm=0.005, clip_fraction=0.30,
        raw_max=1.01, kappa_t_max=0.96,
    )

    df = pd.DataFrame([
        row_pass_a, row_fail_soft_band, row_fail_clip_lo, row_fail_disagreement,
        row_pass_c, row_fail_c, row_pass_b_flip, row_fail_cert,
    ])
    promoted = apply_promotion_gate(
        df,
        promotion=DEFAULT_CONFIG["promotion"],
        strict_band=DEFAULT_CONFIG["strict_band"],
        family_c_cfg=DEFAULT_CONFIG["family_c_stress"],
    )
    # Expected promotions: rows 0, 4, 6 (indices).
    assert promoted["_promoted"].tolist() == [
        True, False, False, False, True, False, True, False,
    ]


# ---------------------------------------------------------------------------
# 4. Empty-shortlist contract
# ---------------------------------------------------------------------------

def test_empty_shortlist_emits_refinement_manifest(tmp_path: Path) -> None:
    """When no row promotes, runner emits refinement manifest and returns 0."""
    # Drop Family C entirely so the safety gate fails, and make the positive
    # thresholds unreachable by raising them to 1.0 (no row can hit
    # policy_disagreement >= 1.0 on a 2-action MDP).
    cfg = _tiny_config()
    cfg["families"] = ["A", "B"]
    cfg["promotion"]["policy_disagreement_min"] = 1.0
    cfg["promotion"]["mass_delta_d_min"] = 1.0
    cfg["promotion"]["value_gap_norm_min"] = 1.0

    out_dir = tmp_path / "out"
    result = run_search(
        cfg, output_root=out_dir, seed=42,
        exact_argv=["test"],
    )
    assert result["n_promoted"] == 0
    assert result["refinement_required"] is True
    manifest = out_dir / "shortlist_refinement_manifest.md"
    assert manifest.is_file()
    text = manifest.read_text()
    assert "Family A" in text
    assert "Family B" in text
    assert "Family C" in text
    assert "Thresholds MUST NOT be relaxed" in text


# ---------------------------------------------------------------------------
# 5. Hard-cap enforcement
# ---------------------------------------------------------------------------

def test_hard_cap_raises_config_error(tmp_path: Path) -> None:
    """A grid that exceeds max_candidates after prefilter raises ConfigError."""
    cfg = _tiny_config()
    cfg["families"] = ["C"]              # Family C admits unconditionally
    # Family C grid product = |L_tail| * |R_penalty| * |beta_raw_mult|
    # Inflate to ~7 * 5 * 10 = 350 candidates.
    cfg["family_params"]["C"] = {
        "L_tail_range": [4, 5, 6, 7, 8, 9, 10],
        "R_penalty_range": [0.5, 1.0, 1.5, 2.0, 3.0],
        "beta_raw_multiplier_range": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0],
        "gamma": 0.95,
    }
    cfg["max_candidates"] = 100          # well below 350

    out_dir = tmp_path / "out"
    with pytest.raises(ConfigError, match="max_candidates"):
        run_search(cfg, output_root=out_dir, seed=42, exact_argv=["test"])


# ---------------------------------------------------------------------------
# CLI smoke (--dry-run sanity)
# ---------------------------------------------------------------------------

def test_cli_dry_run(tmp_path: Path) -> None:
    """CLI --dry-run returns 0 on the tiny config."""
    # Write a tiny yaml so the CLI path gets exercised (not the DEFAULT_CONFIG
    # import).
    cfg = _tiny_config()
    yaml_path = tmp_path / "cfg.yaml"
    import yaml as _yaml
    with open(yaml_path, "w") as f:
        _yaml.safe_dump(cfg, f, sort_keys=False)
    out_dir = tmp_path / "out"
    rc = main([
        "--config", str(yaml_path),
        "--seed", "42",
        "--output-root", str(out_dir),
        "--dry-run",
    ])
    assert rc == 0
    assert (out_dir / "shortlist.csv").is_file()
