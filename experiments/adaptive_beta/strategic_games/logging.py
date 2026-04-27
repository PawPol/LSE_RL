"""Strategic-learning logging schema for Phase VII-B (planner VII-B-28).

Spec authority:
- ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md`` §13 (logging
  schema, episode + transition columns).
- Parent: ``docs/specs/phase_VII_adaptive_beta.md`` §7.4.

This module is an **additive extension** over
:mod:`experiments.adaptive_beta.logging_callbacks`. It does NOT fork the
existing Phase VII loggers; it wraps them and appends the strategic-specific
columns. Spec §1 of the parent CLAUDE rules: prefer adding new modules over
editing stable infrastructure.

Components
----------
- :class:`StrategicEpisodeRow` — schema dataclass mirroring spec §13 episode
  columns, used as the canonical row type for ``episodes.csv`` produced by
  Phase VII-B runs.
- :class:`StrategicTransitionRow` — schema dataclass for transition rows
  (spec §13 transition column list).
- :func:`episode_to_row`, :func:`transition_to_row` — helpers that pack
  arbitrary kwargs into the dataclass schema with type coercion.
- :class:`StrategicLogger` — wraps a parent :class:`EpisodeLogger` +
  :class:`TransitionLogger` and accumulates the strategic columns alongside
  them; flush writes a CSV (episodes) and a Parquet file (transitions, with
  a CSV fallback if pyarrow is unavailable).

Schema versioning: the strategic schema bumps the Phase VII version to
``phaseVII.episodes.v_strategic_1`` / ``phaseVII.transitions.v_strategic_1``
in the file metadata. Downstream aggregators must check the version
explicitly.
"""

from __future__ import annotations

import json
import logging as _stdlib_logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.adaptive_beta.logging_callbacks import (
    EPISODE_COLUMNS as _PARENT_EPISODE_COLUMNS,
    TRANSITION_COLUMNS as _PARENT_TRANSITION_COLUMNS,
    EpisodeLogger,
    TransitionLogger,
    _RunIdentity,
)

_LOG = _stdlib_logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema versions (bumped per planner VII-B-28 acceptance).
# ---------------------------------------------------------------------------
SCHEMA_VERSION_EPISODES_STRATEGIC = "phaseVII.episodes.v_strategic_1"
SCHEMA_VERSION_TRANSITIONS_STRATEGIC = "phaseVII.transitions.v_strategic_1"


# ---------------------------------------------------------------------------
# Spec §13 episode-row schema.
# ---------------------------------------------------------------------------
EPISODE_COLUMNS_STRATEGIC: Tuple[str, ...] = (
    "run_id",
    "seed",
    "game",
    "adversary",
    "method",
    "episode",
    "return",
    "auc_so_far",
    "beta",
    "alignment_rate",
    "mean_effective_discount",
    "bellman_residual",
    "catastrophic",
    "diverged",
    "nan_count",
    "opponent_policy_entropy",
    "policy_total_variation",
    "support_shift",
    "model_rejected",
    "search_phase",
    "phase",
    "memory_m",
    "inertia_lambda",
    "temperature",
    "tau",
)

TRANSITION_COLUMNS_STRATEGIC: Tuple[str, ...] = (
    "run_id",
    "episode",
    "t",
    "state",
    "agent_action",
    "opponent_action",
    "reward",
    "next_state",
    "done",
    "beta",
    "advantage",
    "effective_discount",
    "alignment_indicator",
    "adversary_info_json",
)


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StrategicEpisodeRow:
    """Spec §13 episode-row schema as a typed dataclass.

    All fields use the column names listed in spec §13 verbatim. Optional
    adversary parameters (``memory_m``, ``inertia_lambda``, ``temperature``,
    ``tau``) accept ``None`` since not every adversary exposes them.
    """

    run_id: str
    seed: int
    game: str
    adversary: str
    method: str
    episode: int
    return_: float  # ``return`` is a Python keyword; written as ``return`` in CSV.
    auc_so_far: float
    beta: float
    alignment_rate: float
    mean_effective_discount: float
    bellman_residual: float
    catastrophic: bool
    diverged: bool
    nan_count: int
    opponent_policy_entropy: float
    policy_total_variation: float
    support_shift: bool
    model_rejected: bool
    search_phase: bool
    phase: str
    memory_m: Optional[int] = None
    inertia_lambda: Optional[float] = None
    temperature: Optional[float] = None
    tau: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return the row as a plain dict with the canonical ``return`` key."""
        d = asdict(self)
        d["return"] = d.pop("return_")
        return d


@dataclass
class StrategicTransitionRow:
    """Spec §13 transition-row schema as a typed dataclass.

    ``adversary_info_json`` is the JSON-encoded adversary ``info()`` dict
    captured at the time of the transition (spec §5.2 mandatory keys plus
    any extras, e.g. ``hypothesis_id`` / ``hypothesis_distance``).
    """

    run_id: str
    episode: int
    t: int
    state: int
    agent_action: int
    opponent_action: int
    reward: float
    next_state: int
    done: bool
    beta: float
    advantage: float
    effective_discount: float
    alignment_indicator: bool
    adversary_info_json: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Type-coercion helpers
# ---------------------------------------------------------------------------

def _coerce_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    return int(arr.flat[0])


def _coerce_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    return float(arr.flat[0])


def _coerce_int(x: Any) -> int:
    return int(np.asarray(x).flat[0])


def _coerce_float(x: Any) -> float:
    return float(np.asarray(x).flat[0])


def _coerce_bool(x: Any) -> bool:
    return bool(np.asarray(x).flat[0])


def episode_to_row(
    *,
    run_id: str,
    seed: int,
    game: str,
    adversary: str,
    method: str,
    episode: int,
    episode_return: float,
    auc_so_far: float,
    beta: float,
    alignment_rate: float,
    mean_effective_discount: float,
    bellman_residual: float,
    catastrophic: bool,
    diverged: bool,
    nan_count: int,
    opponent_policy_entropy: float,
    policy_total_variation: float,
    support_shift: bool,
    model_rejected: bool,
    search_phase: bool,
    phase: str,
    memory_m: Optional[int] = None,
    inertia_lambda: Optional[float] = None,
    temperature: Optional[float] = None,
    tau: Optional[float] = None,
) -> StrategicEpisodeRow:
    """Pack the spec §13 episode fields into a :class:`StrategicEpisodeRow`.

    Type coercions are explicit to keep the CSV/Parquet column types
    deterministic across runs (booleans never silently become ints).
    """
    return StrategicEpisodeRow(
        run_id=str(run_id),
        seed=_coerce_int(seed),
        game=str(game),
        adversary=str(adversary),
        method=str(method),
        episode=_coerce_int(episode),
        return_=_coerce_float(episode_return),
        auc_so_far=_coerce_float(auc_so_far),
        beta=_coerce_float(beta),
        alignment_rate=_coerce_float(alignment_rate),
        mean_effective_discount=_coerce_float(mean_effective_discount),
        bellman_residual=_coerce_float(bellman_residual),
        catastrophic=_coerce_bool(catastrophic),
        diverged=_coerce_bool(diverged),
        nan_count=_coerce_int(nan_count),
        opponent_policy_entropy=_coerce_float(opponent_policy_entropy),
        policy_total_variation=_coerce_float(policy_total_variation),
        support_shift=_coerce_bool(support_shift),
        model_rejected=_coerce_bool(model_rejected),
        search_phase=_coerce_bool(search_phase),
        phase=str(phase),
        memory_m=_coerce_int_or_none(memory_m),
        inertia_lambda=_coerce_float_or_none(inertia_lambda),
        temperature=_coerce_float_or_none(temperature),
        tau=_coerce_float_or_none(tau),
    )


def transition_to_row(
    *,
    run_id: str,
    episode: int,
    t: int,
    state: int,
    agent_action: int,
    opponent_action: int,
    reward: float,
    next_state: int,
    done: bool,
    beta: float,
    advantage: float,
    effective_discount: float,
    alignment_indicator: bool,
    adversary_info: Dict[str, Any],
) -> StrategicTransitionRow:
    """Pack a transition into :class:`StrategicTransitionRow`.

    ``adversary_info`` is JSON-serialised here so downstream Parquet/CSV
    storage stays a single string column — saves us a per-key schema
    explosion across heterogeneous adversaries.
    """
    return StrategicTransitionRow(
        run_id=str(run_id),
        episode=_coerce_int(episode),
        t=_coerce_int(t),
        state=_coerce_int(state),
        agent_action=_coerce_int(agent_action),
        opponent_action=_coerce_int(opponent_action),
        reward=_coerce_float(reward),
        next_state=_coerce_int(next_state),
        done=_coerce_bool(done),
        beta=_coerce_float(beta),
        advantage=_coerce_float(advantage),
        effective_discount=_coerce_float(effective_discount),
        alignment_indicator=_coerce_bool(alignment_indicator),
        adversary_info_json=json.dumps(
            _json_safe(adversary_info), sort_keys=True, default=str
        ),
    )


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars / arrays to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# StrategicLogger
# ---------------------------------------------------------------------------

# Try-import pyarrow once at module load; fall back to CSV if missing.
try:  # pragma: no cover - import guard
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    _HAS_PYARROW = True
except Exception:  # pragma: no cover
    _HAS_PYARROW = False
    pa = None  # type: ignore
    pq = None  # type: ignore


class StrategicLogger:
    """Wraps :class:`EpisodeLogger` + :class:`TransitionLogger` and adds the
    Phase VII-B strategic columns.

    Architectural note
    ------------------
    The parent loggers own the Phase VII schema (``EPISODE_COLUMNS`` +
    ``TRANSITION_COLUMNS``). This wrapper does NOT modify them; it stores
    the strategic-specific columns in an internal column-store and writes
    a separate ``episodes_strategic.csv`` / ``transitions_strategic.parquet``
    pair. Phase VII consumers that only know the parent schema continue to
    work; Phase VII-B consumers read the additional files via
    :func:`experiments.adaptive_beta.strategic_games.analysis.aggregate.load_run_summary`.

    Parameters
    ----------
    identity
        Run-level tags propagated into every row.
    parent_episode_logger
        Optional pre-existing :class:`EpisodeLogger`. If ``None`` a fresh
        instance is constructed.
    parent_transition_logger
        Optional pre-existing :class:`TransitionLogger`. If ``None`` a
        fresh instance is constructed with no stratification.
    game, adversary
        Constant tags for the strategic columns (``game`` and ``adversary``
        are NOT in the parent Phase VII schema; they're added here).
    """

    def __init__(
        self,
        identity: _RunIdentity,
        *,
        game: str,
        adversary: str,
        parent_episode_logger: Optional[EpisodeLogger] = None,
        parent_transition_logger: Optional[TransitionLogger] = None,
        stratify_every: int = 1,
    ) -> None:
        self._id = identity
        self._game = str(game)
        self._adversary = str(adversary)
        self._parent_ep = (
            parent_episode_logger
            if parent_episode_logger is not None
            else EpisodeLogger(identity)
        )
        self._parent_tr = (
            parent_transition_logger
            if parent_transition_logger is not None
            else TransitionLogger(identity, stratify_every=stratify_every)
        )
        # Strategic-only column buffers (one list per column).
        self._ep_cols: Dict[str, List[Any]] = {
            c: [] for c in EPISODE_COLUMNS_STRATEGIC
        }
        self._tr_cols: Dict[str, List[Any]] = {
            c: [] for c in TRANSITION_COLUMNS_STRATEGIC
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def parent_episode_logger(self) -> EpisodeLogger:
        return self._parent_ep

    @property
    def parent_transition_logger(self) -> TransitionLogger:
        return self._parent_tr

    @property
    def game(self) -> str:
        return self._game

    @property
    def adversary(self) -> str:
        return self._adversary

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_episode_strategic(self, row: StrategicEpisodeRow) -> None:
        """Append one strategic episode row.

        The caller is expected to also call ``parent_episode_logger.record(...)``
        with the matching Phase VII columns; this wrapper does NOT auto-mirror
        because the parent schema requires fields (``mean_advantage``,
        ``frac_d_eff_below_gamma``) that have no Phase VII-B analogue and
        must be supplied by the runner.
        """
        d = row.as_dict()
        for c in EPISODE_COLUMNS_STRATEGIC:
            self._ep_cols[c].append(d[c])

    def record_transition_strategic(self, row: StrategicTransitionRow) -> None:
        """Append one strategic transition row."""
        d = row.as_dict()
        for c in TRANSITION_COLUMNS_STRATEGIC:
            self._tr_cols[c].append(d[c])

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------
    def flush_episodes_csv(self, path: Path) -> int:
        """Write strategic-episode CSV. Returns row count."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self._ep_cols["episode"])
        # Header line is the canonical spec §13 column order.
        with open(path, "w", encoding="utf-8") as f:
            f.write("# schema_version=" + SCHEMA_VERSION_EPISODES_STRATEGIC + "\n")
            f.write(",".join(EPISODE_COLUMNS_STRATEGIC) + "\n")
            for i in range(n):
                row_vals: List[str] = []
                for c in EPISODE_COLUMNS_STRATEGIC:
                    v = self._ep_cols[c][i]
                    row_vals.append(_csv_format(v))
                f.write(",".join(row_vals) + "\n")
        return n

    def flush_transitions(self, path: Path) -> int:
        """Write strategic-transition Parquet (or CSV fallback).

        If pyarrow is unavailable, writes ``<path>.csv`` and logs a warning.
        Returns the row count.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self._tr_cols["episode"])
        if not _HAS_PYARROW:
            csv_path = path.with_suffix(".csv")
            warnings.warn(
                "pyarrow unavailable; strategic-transition log degrading to "
                f"CSV at {csv_path}. Install pyarrow for production runs.",
                RuntimeWarning,
                stacklevel=2,
            )
            _LOG.warning(
                "StrategicLogger.flush_transitions: pyarrow missing; wrote %s",
                csv_path,
            )
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "# schema_version="
                    + SCHEMA_VERSION_TRANSITIONS_STRATEGIC
                    + "\n"
                )
                f.write(",".join(TRANSITION_COLUMNS_STRATEGIC) + "\n")
                for i in range(n):
                    row_vals = [
                        _csv_format(self._tr_cols[c][i])
                        for c in TRANSITION_COLUMNS_STRATEGIC
                    ]
                    f.write(",".join(row_vals) + "\n")
            return n

        # pyarrow path.
        arrays: List["pa.Array"] = []  # type: ignore[name-defined]
        fields: List["pa.Field"] = []  # type: ignore[name-defined]
        int_cols = {
            "episode", "t", "state", "agent_action",
            "opponent_action", "next_state",
        }
        bool_cols = {"done", "alignment_indicator"}
        float_cols = {"reward", "beta", "advantage", "effective_discount"}
        for c in TRANSITION_COLUMNS_STRATEGIC:
            data = self._tr_cols[c]
            if c in int_cols:
                arr = pa.array(data, type=pa.int64())  # type: ignore[arg-type]
                fields.append(pa.field(c, pa.int64()))
            elif c in bool_cols:
                arr = pa.array(data, type=pa.bool_())  # type: ignore[arg-type]
                fields.append(pa.field(c, pa.bool_()))
            elif c in float_cols:
                arr = pa.array(data, type=pa.float64())  # type: ignore[arg-type]
                fields.append(pa.field(c, pa.float64()))
            else:  # run_id, adversary_info_json
                arr = pa.array(data, type=pa.string())  # type: ignore[arg-type]
                fields.append(pa.field(c, pa.string()))
            arrays.append(arr)
        schema = pa.schema(
            fields,
            metadata={
                b"schema_version": SCHEMA_VERSION_TRANSITIONS_STRATEGIC.encode(),
                b"game": self._game.encode(),
                b"adversary": self._adversary.encode(),
            },
        )
        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, path)
        return n


def _csv_format(v: Any) -> str:
    """Compact, loss-free CSV serialisation matching parent EpisodeLogger."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v)
    return str(v)


__all__ = [
    "EPISODE_COLUMNS_STRATEGIC",
    "TRANSITION_COLUMNS_STRATEGIC",
    "SCHEMA_VERSION_EPISODES_STRATEGIC",
    "SCHEMA_VERSION_TRANSITIONS_STRATEGIC",
    "StrategicEpisodeRow",
    "StrategicTransitionRow",
    "episode_to_row",
    "transition_to_row",
    "StrategicLogger",
]
