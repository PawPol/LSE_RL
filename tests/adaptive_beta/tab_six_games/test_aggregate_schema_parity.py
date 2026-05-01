"""Schema-parity tests for the Phase VIII tab-six-games long CSV."""

from __future__ import annotations

import gzip
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.adaptive_beta.tab_six_games.analysis.aggregate import (
    LONG_CSV_COLUMNS,
    Phase8RunRoster,
    aggregate_to_long_csv,
)


EXPECTED_LONG_CSV_COLUMNS = [
    "run_id",
    "config_hash",
    "phase",
    "stage",
    "game",
    "subcase",
    "method",
    "seed",
    "episode",
    "return",
    "length",
    "epsilon",
    "alignment_rate",
    "mean_signed_alignment",
    "mean_advantage",
    "mean_abs_advantage",
    "mean_d_eff",
    "median_d_eff",
    "frac_d_eff_below_gamma",
    "frac_d_eff_above_one",
    "bellman_residual",
    "td_target_abs_max",
    "q_abs_max",
    "catastrophic",
    "success",
    "regret",
    "shift_event",
    "divergence_event",
    "contraction_reward",
    "empirical_contraction_ratio",
    "log_residual_reduction",
    "ucb_arm_index",
    "beta_clip_count",
    "beta_clip_frequency",
    "recovery_time_after_shift",
    "beta_sign_correct",
    "beta_lag_to_oracle",
    "regret_vs_oracle",
    "catastrophic_episodes",
    "worst_window_return_percentile",
    "trap_entries",
    "constraint_violations",
    "overflow_count",
    "regime",
    "switch_event",
    "episodes_since_switch",
    "oracle_beta",
    "nan_count",
    "diverged",
]

METADATA_COLUMNS = {
    "run_id",
    "config_hash",
    "phase",
    "stage",
    "game",
    "subcase",
    "method",
    "seed",
    "episode",
}


def _metric_payload(n_episodes: int) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for col in LONG_CSV_COLUMNS:
        if col in METADATA_COLUMNS:
            continue
        if col == "regime":
            payload[col] = np.asarray(["stable"] * n_episodes, dtype="<U12")
        elif col == "return":
            payload[col] = np.linspace(1.0, float(n_episodes), n_episodes)
        elif col in {"switch_event", "shift_event"}:
            values = np.zeros(n_episodes, dtype=np.float64)
            if n_episodes:
                values[min(1, n_episodes - 1)] = 1.0
            payload[col] = values
        else:
            payload[col] = np.linspace(0.1, 0.1 * n_episodes, n_episodes)
    return payload


def _write_run(
    raw_root: Path,
    *,
    run_id: str = "run_0",
    phase: str = "VIII",
    stage: str = "stage1",
    game: str = "matching_pennies",
    subcase: str = "canonical",
    method: str = "fixed_beta_0",
    seed: int = 7,
    n_episodes: int = 5,
    write_metrics: bool = True,
    extra_metrics: dict[str, np.ndarray] | None = None,
) -> Path:
    run_dir = raw_root / run_id
    run_dir.mkdir(parents=True)
    run_json = {
        "run_id": run_id,
        "config_hash": f"cfg_{run_id}",
        "phase": phase,
        "stage": stage,
        "game": game,
        "subcase": subcase,
        "method": method,
        "seed": seed,
        "n_episodes": n_episodes,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    if write_metrics:
        payload = _metric_payload(n_episodes)
        if extra_metrics:
            payload.update(extra_metrics)
        np.savez(run_dir / "metrics.npz", **payload)
    return run_dir


def _read_long_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def test_long_csv_columns_count() -> None:
    assert len(LONG_CSV_COLUMNS) == 49


def test_long_csv_columns_byte_identical() -> None:
    assert list(LONG_CSV_COLUMNS) == EXPECTED_LONG_CSV_COLUMNS


def test_default_raw_root() -> None:
    signature = inspect.signature(aggregate_to_long_csv)
    default = Path(signature.parameters["raw_root"].default)

    assert default == Path("results/adaptive_beta/tab_six_games/raw")
    assert "results/weighted_lse_dp" not in default.as_posix()


def test_per_episode_expansion(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    out_path = tmp_path / "long.csv"
    _write_run(raw_root, n_episodes=5)

    summary = aggregate_to_long_csv(raw_root=raw_root, out_path=out_path)
    df = _read_long_csv(out_path)

    assert summary["total_episodes"] == 5
    assert len(df) == 5


def test_metadata_replication(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    out_path = tmp_path / "long.csv"
    metadata = {
        "run_id": "run_metadata",
        "phase": "VIII",
        "stage": "stage5",
        "game": "shapley",
        "subcase": "hard",
        "method": "contraction_ucb",
        "seed": 13,
    }
    _write_run(raw_root, n_episodes=5, **metadata)

    aggregate_to_long_csv(raw_root=raw_root, out_path=out_path)
    df = _read_long_csv(out_path)

    for col, expected in metadata.items():
        assert set(df[col].dropna().unique()) == {expected}


def test_missing_metrics_npz_recorded(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    out_path = tmp_path / "long.csv"
    run_dir = _write_run(raw_root, run_id="run_missing", write_metrics=False)

    summary = aggregate_to_long_csv(raw_root=raw_root, out_path=out_path)

    assert {
        "run_dir": str(run_dir),
        "reason": "metrics_npz_missing",
    } in summary["missing_runs"]


def test_schema_drift_detected(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    out_path = tmp_path / "long.csv"
    run_dir = _write_run(
        raw_root,
        run_id="run_schema_drift",
        extra_metrics={"foreign_column": np.arange(5, dtype=np.float64)},
    )

    summary = aggregate_to_long_csv(raw_root=raw_root, out_path=out_path)

    assert {
        "run_dir": str(run_dir),
        "reason": "schema_drift",
        "foreign_columns": ["foreign_column"],
    } in summary["missing_runs"]


def test_include_phase_VII_read_only(tmp_path: Path) -> None:
    adaptive_root = tmp_path / "results" / "adaptive_beta"
    raw_root = adaptive_root / "tab_six_games" / "raw"
    phase_vii_root = adaptive_root / "strategic" / "raw"
    out_path = tmp_path / "long.csv"
    _write_run(
        phase_vii_root,
        run_id="phase_vii_run",
        phase="VII-old-tag",
        stage="stage_B2",
        game="strategic_rps",
        subcase="sign_switching",
        method="return_ucb",
        seed=2,
        n_episodes=3,
    )
    roster = Phase8RunRoster(base_path=tmp_path / "roster")

    summary = aggregate_to_long_csv(
        raw_root=raw_root,
        out_path=out_path,
        roster=roster,
        include_phase_VII=True,
    )
    df = _read_long_csv(out_path)

    assert summary["total_runs"] == 1
    assert set(df["phase"].unique()) == {"VII-B"}
    assert roster.summarize()["total"] == 0


def test_gzip_output(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    plain_path = tmp_path / "long.csv"
    gzip_path = tmp_path / "long.csv.gz"
    _write_run(raw_root, n_episodes=4)

    aggregate_to_long_csv(raw_root=raw_root, out_path=plain_path)
    aggregate_to_long_csv(raw_root=raw_root, out_path=gzip_path)

    with open(plain_path, "r", encoding="utf-8", newline="") as fh:
        plain_text = fh.read()
    with gzip.open(gzip_path, "rt", encoding="utf-8", newline="") as fh:
        gzip_text = fh.read()
    assert gzip_text == plain_text


def test_pandas_round_trip(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    out_path = tmp_path / "long.csv"
    _write_run(raw_root, n_episodes=5)
    aggregate_to_long_csv(raw_root=raw_root, out_path=out_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = pd.read_csv(out_path)

    dtype_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, pd.errors.DtypeWarning)
    ]
    assert dtype_warnings == []
    assert list(df.columns) == EXPECTED_LONG_CSV_COLUMNS
    assert len(df) == 5
