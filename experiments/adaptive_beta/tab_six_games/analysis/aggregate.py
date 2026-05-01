"""Phase VIII tab-six-games long-CSV aggregator (spec §8.3).

This module produces the analysis-side "long CSV" — the single source
of truth for tables and figures emitted under Phase VIII (spec §8.3).
The CSV is keyed on
``(phase, stage, game, subcase, method, seed, episode)`` and holds the
union of:

* §7.1 (Phase VII per-episode metric vocabulary, reused verbatim);
* §7.2 (Phase VIII delta metrics, written by
  :mod:`experiments.adaptive_beta.tab_six_games.metrics`);
* the dispatch metadata replicated from each run's ``run.json``.

The aggregator's contract (per spec §8.3):

* walk ``raw_root`` for ``run.json`` files;
* for each one, load the sibling ``metrics.npz``;
* expand the per-episode arrays into rows, replicating the metadata;
* never retroactively compute delta metrics — runners populate
  ``metrics.npz`` at write-time, and the aggregator only flattens
  what is on disk (lessons.md, "aggregation drift");
* default missing columns to ``np.nan`` (numeric) or ``""`` (string);
* validate each ``metrics.npz`` against the expected §7.1+§7.2 column
  set — foreign columns are recorded to ``missing_runs`` with a
  ``schema_drift`` note (lessons.md, figure-script schema rule);
* default ``raw_root`` is
  ``results/adaptive_beta/tab_six_games/raw`` — never
  ``results/weighted_lse_dp`` (lessons.md, default-root drift);
* support an optional ``include_phase_VII`` flag for
  M8 read-only narrative cross-reference; Phase VII rows are tagged
  ``phase="VII-B"`` and DO NOT enter the
  :class:`Phase8RunRoster` (spec §10.4 / addendum §10.5).

Stream-writes the CSV via stdlib :mod:`csv` (``QUOTE_MINIMAL``); the
``.csv.gz`` suffix triggers transparent gzip compression. Returns a
small summary dict (``total_runs``, ``total_episodes``,
``missing_runs``, ``schema_columns``).

Notes
-----
The W2.A test author writes the schema-parity / figure smoke tests; this
module ships only the aggregator and a CLI entry point.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Re-export Phase8RunRoster for caller convenience (spec §8.3 contract).
from experiments.adaptive_beta.tab_six_games.manifests import Phase8RunRoster

__all__ = [
    "AggregateRow",
    "Phase8RunRoster",
    "LONG_CSV_COLUMNS",
    "PHASE_VIII_EXPECTED_COLUMNS",
    "aggregate_to_long_csv",
]


# ---------------------------------------------------------------------------
# Column inventory (spec §7.1 + §7.2 + §8.3 metadata)
# ---------------------------------------------------------------------------

#: Long-CSV column order. Matches spec §8.3 verbatim. Any reordering
#: here is a backwards-incompatible schema change — bump the schema
#: header and the figure-script consumers in lock-step (lessons.md,
#: figure-script schema).
LONG_CSV_COLUMNS: Tuple[str, ...] = (
    # Run identity / dispatch metadata (from run.json).
    "run_id",
    "config_hash",
    "phase",
    "stage",
    "game",
    "subcase",
    "method",
    "seed",
    "episode",
    # §7.1 reused metrics.
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
    # §7.2 Phase VIII delta metrics.
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
    # Regime / oracle state (Stage 4 / §8.3 union).
    "regime",
    "switch_event",
    "episodes_since_switch",
    "oracle_beta",
    # Diagnostics.
    "nan_count",
    "diverged",
    # Per HALT 5 OQ2 (2026-05-01): runner emits these per-episode but
    # they were missing from the schema-parity guard, producing soft
    # `schema_drift` warnings on every run. `beta_raw` / `beta_used`
    # are the schedule's pre-clip / post-clip outputs; the existing
    # `mean_d_eff` is retained, and `effective_discount_mean` is the
    # runner's verbose alias kept for downstream-figure ergonomics
    # (rename to `mean_d_eff` is deferred to a separate cleanup pass).
    # `goal_reaches` is a delayed_chain-only diagnostic counting
    # episodes that reached the goal terminal.
    "beta_raw",
    "beta_used",
    "effective_discount_mean",
    "goal_reaches",
)

#: Columns sourced from ``run.json`` (replicated across every per-episode
#: row of a given run). Everything else is sourced from
#: ``metrics.npz`` per-episode arrays.
_METADATA_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "config_hash",
    "phase",
    "stage",
    "game",
    "subcase",
    "method",
    "seed",
)

#: Per-episode integer index column.
_EPISODE_COLUMN: str = "episode"

#: Columns expected in every Phase VIII ``metrics.npz`` per spec
#: §7.1 + §7.2. Used by the schema-parity guard. The "stage" /
#: "regime" / "diverged" columns are tolerated as either run-level
#: scalars or per-episode arrays; both forms are accepted.
PHASE_VIII_EXPECTED_COLUMNS: frozenset[str] = frozenset(
    set(LONG_CSV_COLUMNS) - set(_METADATA_COLUMNS) - {_EPISODE_COLUMN}
)

#: numpy savez forbids the bare keyword ``return`` (Python keyword) so
#: callers commonly persist it as ``return_``. Both forms are accepted
#: by the loader; both are normalised to ``"return"`` at row time.
_RETURN_ALIASES: Tuple[str, ...] = ("return", "return_")

#: String-typed columns. All other columns are emitted as float / int.
#: Missing string values render as ``""`` (empty); missing numerics
#: render as ``""`` from ``str(np.nan)`` — but to keep CSV downstream
#: pandas-friendly we preserve ``np.nan`` -> ``""`` empty cell.
_STRING_COLUMNS: frozenset[str] = frozenset(
    {
        "run_id",
        "config_hash",
        "phase",
        "stage",
        "game",
        "subcase",
        "method",
        "regime",
    }
)

#: Reserved npz key (the schema header, written by
#: :func:`experiments.weighted_lse_dp.common.io.save_npz_with_schema`).
_SCHEMA_HEADER_KEY: str = "_schema"


# ---------------------------------------------------------------------------
# AggregateRow dataclass
# ---------------------------------------------------------------------------

@dataclass
class AggregateRow:
    """One per-episode row of the Phase VIII long CSV (spec §8.3).

    Field order matches :data:`LONG_CSV_COLUMNS` exactly. Numeric
    fields default to ``np.nan`` (a "documented absence" in pandas);
    string fields default to ``""``. The ``to_csv_row`` helper renders
    ``np.nan`` as the empty cell to match pandas
    ``read_csv(..., keep_default_na=True)`` round-tripping.
    """

    # Metadata.
    run_id: str = ""
    config_hash: str = ""
    phase: str = ""
    stage: str = ""
    game: str = ""
    subcase: str = ""
    method: str = ""
    seed: float = float("nan")
    episode: float = float("nan")
    # §7.1 reused metrics.
    return_value: float = float("nan")  # Python keyword -> attribute alias
    length: float = float("nan")
    epsilon: float = float("nan")
    alignment_rate: float = float("nan")
    mean_signed_alignment: float = float("nan")
    mean_advantage: float = float("nan")
    mean_abs_advantage: float = float("nan")
    mean_d_eff: float = float("nan")
    median_d_eff: float = float("nan")
    frac_d_eff_below_gamma: float = float("nan")
    frac_d_eff_above_one: float = float("nan")
    bellman_residual: float = float("nan")
    td_target_abs_max: float = float("nan")
    q_abs_max: float = float("nan")
    catastrophic: float = float("nan")
    success: float = float("nan")
    regret: float = float("nan")
    shift_event: float = float("nan")
    divergence_event: float = float("nan")
    # §7.2 delta metrics.
    contraction_reward: float = float("nan")
    empirical_contraction_ratio: float = float("nan")
    log_residual_reduction: float = float("nan")
    ucb_arm_index: float = float("nan")
    beta_clip_count: float = float("nan")
    beta_clip_frequency: float = float("nan")
    recovery_time_after_shift: float = float("nan")
    beta_sign_correct: float = float("nan")
    beta_lag_to_oracle: float = float("nan")
    regret_vs_oracle: float = float("nan")
    catastrophic_episodes: float = float("nan")
    worst_window_return_percentile: float = float("nan")
    trap_entries: float = float("nan")
    constraint_violations: float = float("nan")
    overflow_count: float = float("nan")
    # Regime / oracle.
    regime: str = ""
    switch_event: float = float("nan")
    episodes_since_switch: float = float("nan")
    oracle_beta: float = float("nan")
    # Diagnostics.
    nan_count: float = float("nan")
    diverged: float = float("nan")

    def to_csv_row(self) -> List[str]:
        """Render this row as a list of CSV cells (string)."""
        out: List[str] = []
        for col in LONG_CSV_COLUMNS:
            attr = "return_value" if col == "return" else col
            value = getattr(self, attr)
            if col in _STRING_COLUMNS:
                # Preserve empty string default; coerce None to "".
                out.append("" if value is None else str(value))
            else:
                # Numeric: NaN renders as the empty cell (pandas-round-trippable).
                if value is None:
                    out.append("")
                else:
                    try:
                        if isinstance(value, float) and math.isnan(value):
                            out.append("")
                        else:
                            out.append(repr_numeric(value))
                    except (TypeError, ValueError):
                        out.append(str(value))
        return out


def repr_numeric(value: Any) -> str:
    """Render a numeric scalar without forcing scientific notation.

    Integers render as ``"3"``; floats round-trip via ``repr`` so that
    downstream ``pandas.read_csv`` recovers the same float. Booleans
    are coerced to int (0/1) per CSV convention.
    """
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return ""
        if f == int(f) and abs(f) < 1e16:
            # Render exact integers without trailing ".0" only for huge magnitudes;
            # otherwise keep the float form for downstream type stability.
            return repr(f)
        return repr(f)
    return str(value)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_run_json(run_json_path: Path) -> Dict[str, Any]:
    """Load and minimally validate a ``run.json``."""
    with open(run_json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_metadata(
    run_json: Dict[str, Any],
    *,
    fallback_phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Pull metadata fields from a ``run.json`` payload.

    Tolerates the Phase VII / Phase VIII naming conventions:

    * ``run_id`` may sit at top level or be inferred from the run dir;
    * ``phase`` defaults to ``fallback_phase`` if absent;
    * ``stage`` defaults to empty string;
    * ``method`` and ``algorithm`` are aliases (Phase VII used the
      latter);
    * ``task`` and ``game`` are aliases (Phase VII used the former).
    """
    run_id = str(
        run_json.get("run_id")
        or run_json.get("run_directory_name")
        or run_json.get("created_at")
        or ""
    )
    config_hash = str(
        run_json.get("config_hash")
        or run_json.get("config", {}).get("config_hash")
        or ""
    )
    phase_value = str(run_json.get("phase") or fallback_phase or "")
    stage_value = str(run_json.get("stage") or run_json.get("suite") or "")
    game_value = str(run_json.get("game") or run_json.get("task") or "")
    subcase_value = str(run_json.get("subcase") or "")
    method_value = str(
        run_json.get("method") or run_json.get("algorithm") or ""
    )
    seed_raw = run_json.get("seed")
    if seed_raw is None:
        seed_value: float = float("nan")
    else:
        try:
            seed_value = float(int(seed_raw))
        except (TypeError, ValueError):
            seed_value = float("nan")

    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "phase": phase_value,
        "stage": stage_value,
        "game": game_value,
        "subcase": subcase_value,
        "method": method_value,
        "seed": seed_value,
    }


def _load_metrics_npz(
    metrics_path: Path,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load metrics.npz; return (data_dict, foreign_keys).

    ``data_dict`` is keyed by the column names expected in §7.1 + §7.2
    plus :data:`_METADATA_COLUMNS` aliases. ``foreign_keys`` lists any
    keys that did not match the expected schema (logged to
    ``missing_runs`` with a ``schema_drift`` note).

    The reserved ``_schema`` header is stripped silently.
    """
    # ``allow_pickle=False`` is the safe default; the schema header is a
    # uint8 array, all other arrays are numeric.
    with np.load(metrics_path, allow_pickle=False) as loaded:
        keys = list(loaded.files)
        data: Dict[str, np.ndarray] = {}
        for key in keys:
            if key == _SCHEMA_HEADER_KEY:
                continue
            data[key] = np.asarray(loaded[key])

    # Normalise the ``return`` keyword alias.
    for alias in _RETURN_ALIASES:
        if alias in data and alias != "return":
            data["return"] = data.pop(alias)

    # Identify foreign columns. We compare against the expected union
    # plus the ``return`` alias acceptance.
    expected_with_alias: set[str] = set(PHASE_VIII_EXPECTED_COLUMNS) | {
        "return"
    }
    foreign_keys = sorted(
        k for k in data.keys() if k not in expected_with_alias
    )
    return data, foreign_keys


def _episode_count(data: Dict[str, np.ndarray]) -> int:
    """Determine the per-episode array length.

    Picks the first array with ``ndim >= 1``. Returns 0 if none found.
    """
    for arr in data.values():
        if arr.ndim >= 1 and arr.size >= 1:
            return int(arr.shape[0])
    return 0


def _scalar_from_metrics_array(
    arr: np.ndarray, episode_idx: int
) -> Any:
    """Extract a per-episode scalar from a metrics array.

    Handles three shapes:

    * ``ndim == 0`` (run-level scalar)  -> broadcast verbatim;
    * ``ndim == 1`` (per-episode 1-D)   -> pick ``arr[episode_idx]``;
    * ``ndim >= 2``                     -> pick ``arr[episode_idx, ...]``
      and return the row as a Python list (caller decides what to do;
      the long CSV in this version does not project these).

    Out-of-range indices return ``np.nan`` rather than raising — a 1-D
    array shorter than the inferred episode count is a known edge case
    in dev runs and should not bring the aggregator down.
    """
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim == 1:
        if episode_idx < arr.shape[0]:
            return arr[episode_idx].item()
        return float("nan")
    # ndim >= 2: skip per-episode projection; row-multi-dim arrays go
    # through ucb_arm_count / ucb_arm_value which the spec emits as
    # vector-valued and which the long CSV does not flatten.
    return float("nan")


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_rows_for_run(
    metadata: Dict[str, Any],
    metrics: Dict[str, np.ndarray],
    n_episodes_hint: Optional[int] = None,
) -> Iterator[AggregateRow]:
    """Generate per-episode rows for a single run."""
    n_episodes = (
        int(n_episodes_hint) if n_episodes_hint else _episode_count(metrics)
    )
    if n_episodes <= 0:
        return

    for ep in range(n_episodes):
        row = AggregateRow()
        # Metadata replication.
        row.run_id = str(metadata.get("run_id", ""))
        row.config_hash = str(metadata.get("config_hash", ""))
        row.phase = str(metadata.get("phase", ""))
        row.stage = str(metadata.get("stage", ""))
        row.game = str(metadata.get("game", ""))
        row.subcase = str(metadata.get("subcase", ""))
        row.method = str(metadata.get("method", ""))
        row.seed = float(metadata.get("seed", float("nan")))
        row.episode = float(ep)

        # Per-episode metric expansion.
        for col in LONG_CSV_COLUMNS:
            if col in _METADATA_COLUMNS or col == _EPISODE_COLUMN:
                continue
            if col in _STRING_COLUMNS:
                # String-typed metric columns (e.g. ``regime``).
                if col in metrics:
                    value = _scalar_from_metrics_array(metrics[col], ep)
                    if value is None or (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        # Leave default "".
                        continue
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    setattr(row, col, str(value))
                continue
            # Numeric column.
            if col in metrics:
                value = _scalar_from_metrics_array(metrics[col], ep)
                attr = "return_value" if col == "return" else col
                try:
                    setattr(row, attr, float(value))
                except (TypeError, ValueError):
                    setattr(row, attr, float("nan"))

        yield row


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _open_writer(out_path: Path) -> Tuple[Any, Any]:
    """Open an output stream for the long CSV.

    Returns ``(file_handle, csv_writer)``. ``.csv.gz`` triggers gzip;
    plain ``.csv`` writes uncompressed. Caller is responsible for
    closing the returned file handle.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".gz":
        # Open as binary gzip; wrap a TextIOWrapper for csv.writer.
        gz = gzip.open(out_path, "wb")
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        writer = csv.writer(text, quoting=csv.QUOTE_MINIMAL)
        return text, writer
    text = open(out_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(text, quoting=csv.QUOTE_MINIMAL)
    return text, writer


# ---------------------------------------------------------------------------
# Walk helpers
# ---------------------------------------------------------------------------

def _iter_run_dirs(raw_root: Path) -> Iterator[Path]:
    """Yield run directories (containing ``run.json``) under ``raw_root``.

    Sorted for deterministic CSV ordering.
    """
    if not raw_root.exists():
        return
    candidates = sorted(raw_root.rglob("run.json"))
    for run_json in candidates:
        if run_json.is_file():
            yield run_json.parent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_to_long_csv(
    raw_root: Path = Path("results/adaptive_beta/tab_six_games/raw"),
    out_path: Path = Path(
        "results/adaptive_beta/tab_six_games/processed/long.csv"
    ),
    roster: Optional[Phase8RunRoster] = None,
    *,
    include_phase_VII: bool = False,
) -> Dict[str, Any]:
    """Walk ``raw_root`` and emit the Phase VIII long CSV (spec §8.3).

    Parameters
    ----------
    raw_root:
        Root directory holding per-run subdirectories. Each subdir is
        expected to contain a ``run.json`` and a sibling
        ``metrics.npz``. Default is
        ``results/adaptive_beta/tab_six_games/raw`` (lessons.md,
        default-root drift). Phase VIII results never land under
        ``results/weighted_lse_dp/``.
    out_path:
        Destination CSV path. ``.csv.gz`` triggers gzip compression;
        any other suffix writes plain CSV. Parent directories are
        created.
    roster:
        Optional :class:`Phase8RunRoster` used for cross-validation.
        Currently only consulted to record which runs exist on disk
        but are absent from the roster (and vice versa) — the caller
        decides what to do with that information. Phase VII rows
        (when ``include_phase_VII=True``) NEVER enter the roster, per
        spec §10.4 / addendum §10.5.
    include_phase_VII:
        When ``True``, additionally walk
        ``results/adaptive_beta/strategic/raw/`` (sibling to
        ``raw_root``'s parent) and tag those rows ``phase="VII-B"``.
        Read-only narrative reference for M8 sign-specialization
        analysis. Phase VII rows do NOT enter ``Phase8RunRoster``.

    Returns
    -------
    dict
        Summary with keys:

        * ``"total_runs"`` — number of runs whose ``metrics.npz``
          loaded successfully.
        * ``"total_episodes"`` — total per-episode rows written.
        * ``"missing_runs"`` — list of dicts describing skipped /
          drift-flagged runs (each has ``run_dir``, ``reason``, and
          optionally ``foreign_columns``).
        * ``"schema_columns"`` — the long-CSV column list (verbatim).
    """
    raw_root = Path(raw_root)
    out_path = Path(out_path)

    total_runs = 0
    total_episodes = 0
    missing_runs: List[Dict[str, Any]] = []

    file_handle, writer = _open_writer(out_path)
    try:
        # Header.
        writer.writerow(list(LONG_CSV_COLUMNS))

        # Pass 1: Phase VIII raw root.
        for run_dir in _iter_run_dirs(raw_root):
            run_json_path = run_dir / "run.json"
            metrics_path = run_dir / "metrics.npz"

            try:
                run_json = _load_run_json(run_json_path)
            except (OSError, json.JSONDecodeError) as exc:
                missing_runs.append(
                    {
                        "run_dir": str(run_dir),
                        "reason": "run_json_unreadable",
                        "detail": str(exc),
                    }
                )
                continue

            if not metrics_path.is_file():
                missing_runs.append(
                    {
                        "run_dir": str(run_dir),
                        "reason": "metrics_npz_missing",
                    }
                )
                continue

            metadata = _extract_metadata(
                run_json, fallback_phase="VIII"
            )

            try:
                data, foreign_keys = _load_metrics_npz(metrics_path)
            except (OSError, ValueError) as exc:
                missing_runs.append(
                    {
                        "run_dir": str(run_dir),
                        "reason": "metrics_npz_unreadable",
                        "detail": str(exc),
                    }
                )
                continue

            if foreign_keys:
                # Schema-parity guard (lessons.md, figure-script schema).
                missing_runs.append(
                    {
                        "run_dir": str(run_dir),
                        "reason": "schema_drift",
                        "foreign_columns": foreign_keys,
                    }
                )

            # Even with drift, we still emit rows for the columns we
            # recognise; the schema-drift note records the problem.
            n_episodes_hint = run_json.get("n_episodes")
            try:
                n_episodes_hint = (
                    int(n_episodes_hint)
                    if n_episodes_hint is not None
                    else None
                )
            except (TypeError, ValueError):
                n_episodes_hint = None

            run_episode_count = 0
            for row in _build_rows_for_run(
                metadata, data, n_episodes_hint=n_episodes_hint
            ):
                writer.writerow(row.to_csv_row())
                run_episode_count += 1

            if run_episode_count == 0:
                missing_runs.append(
                    {
                        "run_dir": str(run_dir),
                        "reason": "no_episodes_in_metrics_npz",
                    }
                )
                continue

            total_runs += 1
            total_episodes += run_episode_count

        # Pass 2: optional Phase VII narrative cross-reference (M8 read-only).
        if include_phase_VII:
            phase_vii_root = _phase_vii_default_root(raw_root)
            for run_dir in _iter_run_dirs(phase_vii_root):
                run_json_path = run_dir / "run.json"
                metrics_path = run_dir / "metrics.npz"

                try:
                    run_json = _load_run_json(run_json_path)
                except (OSError, json.JSONDecodeError) as exc:
                    missing_runs.append(
                        {
                            "run_dir": str(run_dir),
                            "reason": "phase_VII_run_json_unreadable",
                            "detail": str(exc),
                        }
                    )
                    continue

                if not metrics_path.is_file():
                    missing_runs.append(
                        {
                            "run_dir": str(run_dir),
                            "reason": "phase_VII_metrics_npz_missing",
                        }
                    )
                    continue

                metadata = _extract_metadata(
                    run_json, fallback_phase="VII-B"
                )
                # Force the phase tag — Phase VII rows enter the long
                # CSV with ``phase == "VII-B"`` regardless of what the
                # original run.json says (spec §10.4 / addendum §10.5).
                metadata["phase"] = "VII-B"

                try:
                    data, foreign_keys = _load_metrics_npz(metrics_path)
                except (OSError, ValueError) as exc:
                    missing_runs.append(
                        {
                            "run_dir": str(run_dir),
                            "reason": "phase_VII_metrics_npz_unreadable",
                            "detail": str(exc),
                        }
                    )
                    continue

                if foreign_keys:
                    missing_runs.append(
                        {
                            "run_dir": str(run_dir),
                            "reason": "phase_VII_schema_drift",
                            "foreign_columns": foreign_keys,
                        }
                    )

                n_episodes_hint = run_json.get("n_episodes")
                try:
                    n_episodes_hint = (
                        int(n_episodes_hint)
                        if n_episodes_hint is not None
                        else None
                    )
                except (TypeError, ValueError):
                    n_episodes_hint = None

                run_episode_count = 0
                for row in _build_rows_for_run(
                    metadata, data, n_episodes_hint=n_episodes_hint
                ):
                    writer.writerow(row.to_csv_row())
                    run_episode_count += 1

                if run_episode_count == 0:
                    missing_runs.append(
                        {
                            "run_dir": str(run_dir),
                            "reason": "phase_VII_no_episodes_in_metrics_npz",
                        }
                    )
                    continue

                total_runs += 1
                total_episodes += run_episode_count
    finally:
        file_handle.close()

    # The roster is currently read-only here — full reconciliation
    # lives in Phase8RunRoster.reconcile_with_disk(). The parameter is
    # accepted for API symmetry with the spec §8.3 contract; W2.A test
    # author will add reconciliation coverage.
    _ = roster

    return {
        "total_runs": total_runs,
        "total_episodes": total_episodes,
        "missing_runs": missing_runs,
        "schema_columns": list(LONG_CSV_COLUMNS),
    }


def _phase_vii_default_root(phase_viii_raw_root: Path) -> Path:
    """Locate the Phase VII strategic raw root from a Phase VIII root.

    Phase VIII layout:
        ``results/adaptive_beta/tab_six_games/raw/...``
    Phase VII layout:
        ``results/adaptive_beta/strategic/raw/...``

    Walks two parents up (``raw -> tab_six_games -> adaptive_beta``)
    then descends into ``strategic/raw``. Returns the resolved path
    even if it doesn't exist on disk; the caller's ``_iter_run_dirs``
    handles the missing-directory case.
    """
    # phase_viii_raw_root usually ends in ``/raw``. The grandparent
    # is the ``adaptive_beta`` results root.
    p = Path(phase_viii_raw_root)
    grandparent = p.parent.parent
    return grandparent / "strategic" / "raw"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Phase VIII tab-six-games per-run artifacts into "
            "the long CSV (spec §8.3). Default raw_root and out_path "
            "match the Phase VIII layout (lessons.md, default-root drift)."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("results/adaptive_beta/tab_six_games/raw"),
        help=(
            "Root directory containing per-run subdirs with run.json + "
            "metrics.npz. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/adaptive_beta/tab_six_games/processed/long.csv"
        ),
        help=(
            "Output CSV path. Suffix .csv.gz triggers gzip. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--include-phase-VII",
        action="store_true",
        help=(
            "Also walk results/adaptive_beta/strategic/raw/ and tag "
            "those rows phase=VII-B. Read-only narrative reference; "
            "Phase VII rows do NOT enter Phase8RunRoster."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the long-CSV aggregator."""
    args = _parse_cli(argv)
    summary = aggregate_to_long_csv(
        raw_root=args.raw_root,
        out_path=args.out,
        include_phase_VII=args.include_phase_VII,
    )
    # Compact summary to stdout; full missing-runs list to stderr.
    print(
        json.dumps(
            {
                "total_runs": summary["total_runs"],
                "total_episodes": summary["total_episodes"],
                "n_missing_runs": len(summary["missing_runs"]),
                "out": str(args.out),
            }
        )
    )
    if summary["missing_runs"]:
        print(
            json.dumps(
                {"missing_runs": summary["missing_runs"]},
                indent=2,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
