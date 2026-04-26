"""Episode + transition logging callbacks for Phase VII (M3.2).

Spec authority: ``docs/specs/phase_VII_adaptive_beta.md`` §7 (metrics) and
§7.4 (schema columns). The callbacks here are the *only* place that writes
``episodes.csv`` and ``transitions.parquet``; the runner constructs them once
per ``(env, method, seed)`` triple and feeds them per-step / per-episode
dicts emitted by :class:`AdaptiveBetaQAgent`.

Design summary
--------------
Two collector classes:

- :class:`TransitionLogger` buffers one row per environment step. Stage A
  writes every step (``stratify=False``). Stage B/C may pass an integer
  ``stratify_every`` to keep one in N rows plus all rows where
  ``is_shift_step=True`` or ``catastrophe=True`` (spec §7.4 stratified
  rule). Buffered rows are flushed to a single Parquet file at run end.

- :class:`EpisodeLogger` accumulates one row per episode and writes a CSV
  at run end. Columns are exactly the spec §7.4 set.

Both loggers are *passive* — they expose ``record_*`` methods the runner
calls explicitly. There is no MushroomRL ``Callback`` subclass interaction;
that keeps the dependency surface tiny and the test surface explicit.

Schema constants are exported so tests, the runner, and downstream
aggregators can all reference one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Schema (spec §7.4) — single source of truth.
# ---------------------------------------------------------------------------
EPISODE_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "env",
    "method",
    "seed",
    "episode",
    "phase",
    "beta_raw",
    "beta_deployed",
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
)

TRANSITION_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "env",
    "method",
    "seed",
    "episode",
    "t",
    "state",
    "action",
    "reward",
    "next_state",
    "done",
    "phase",
    "beta_deployed",
    "v_next",
    "advantage",
    "td_target",
    "td_error",
    "d_eff",
    "aligned",
    "oracle_action",
    "catastrophe",
)

# Schema version pinned in the parquet metadata for downstream consumers.
SCHEMA_VERSION_EPISODES = "phaseVII.episodes.v1"
SCHEMA_VERSION_TRANSITIONS = "phaseVII.transitions.v1"


@dataclass
class _RunIdentity:
    """Constant per-run identity tags written into every row."""

    run_id: str
    env: str
    method: str
    seed: int


class TransitionLogger:
    """Buffers per-step rows and writes Parquet on flush.

    Parameters
    ----------
    identity
        Constant tags for the run.
    stratify_every
        If > 1, keep only every Nth ordinary transition. Always keep
        rows where ``shift_event=True`` (passed in via ``record``) or
        ``catastrophe=True`` (in the row payload). ``stratify_every=1``
        keeps every row. Spec §7.4: Stage A uses 1; Stage B/C uses 10.
    """

    def __init__(
        self,
        identity: _RunIdentity,
        stratify_every: int = 1,
    ) -> None:
        if stratify_every < 1:
            raise ValueError(
                f"stratify_every must be >= 1, got {stratify_every}"
            )
        self._id = identity
        self._stratify_every = int(stratify_every)
        # One Python list per column (column-store layout makes the
        # Parquet build O(1) and avoids per-row dict allocation).
        self._cols: Dict[str, List[Any]] = {c: [] for c in TRANSITION_COLUMNS}
        self._row_counter: int = 0

    def record(
        self,
        episode: int,
        t: int,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        phase: Any,
        beta_deployed: float,
        v_next: float,
        advantage: float,
        td_target: float,
        td_error: float,
        d_eff: float,
        aligned: bool,
        oracle_action: Optional[int],
        catastrophe: bool,
        shift_event: bool = False,
    ) -> None:
        """Append one transition. Stratification rules applied here."""
        keep = (
            self._stratify_every == 1
            or shift_event
            or catastrophe
            or (self._row_counter % self._stratify_every == 0)
        )
        self._row_counter += 1
        if not keep:
            return
        cols = self._cols
        cols["run_id"].append(self._id.run_id)
        cols["env"].append(self._id.env)
        cols["method"].append(self._id.method)
        cols["seed"].append(int(self._id.seed))
        cols["episode"].append(int(episode))
        cols["t"].append(int(t))
        cols["state"].append(int(state))
        cols["action"].append(int(action))
        cols["reward"].append(float(reward))
        cols["next_state"].append(int(next_state))
        cols["done"].append(bool(done))
        cols["phase"].append(str(phase))
        cols["beta_deployed"].append(float(beta_deployed))
        cols["v_next"].append(float(v_next))
        cols["advantage"].append(float(advantage))
        cols["td_target"].append(float(td_target))
        cols["td_error"].append(float(td_error))
        cols["d_eff"].append(float(d_eff))
        cols["aligned"].append(bool(aligned))
        cols["oracle_action"].append(
            -1 if oracle_action is None else int(oracle_action)
        )
        cols["catastrophe"].append(bool(catastrophe))

    def flush_parquet(self, path: Path) -> int:
        """Write the buffered rows to ``path``. Returns row count.

        Empty buffers still write a zero-row Parquet file with the full
        schema, so downstream loaders never have to special-case missing
        files.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Build pyarrow arrays explicitly to pin schema even on empty cols.
        arrays: List[pa.Array] = []
        fields: List[pa.Field] = []
        for c in TRANSITION_COLUMNS:
            data = self._cols[c]
            if c in {
                "seed", "episode", "t", "state", "action",
                "next_state", "oracle_action",
            }:
                arr = pa.array(data, type=pa.int64())
                fields.append(pa.field(c, pa.int64()))
            elif c in {"done", "aligned", "catastrophe"}:
                arr = pa.array(data, type=pa.bool_())
                fields.append(pa.field(c, pa.bool_()))
            elif c in {
                "reward", "beta_deployed", "v_next", "advantage",
                "td_target", "td_error", "d_eff",
            }:
                arr = pa.array(data, type=pa.float64())
                fields.append(pa.field(c, pa.float64()))
            else:  # run_id, env, method, phase
                arr = pa.array(data, type=pa.string())
                fields.append(pa.field(c, pa.string()))
            arrays.append(arr)
        schema = pa.schema(
            fields,
            metadata={b"schema_version": SCHEMA_VERSION_TRANSITIONS.encode()},
        )
        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, path)
        return len(self._cols["episode"])


class EpisodeLogger:
    """Accumulates per-episode rows and writes CSV on flush."""

    def __init__(self, identity: _RunIdentity) -> None:
        self._id = identity
        self._cols: Dict[str, List[Any]] = {c: [] for c in EPISODE_COLUMNS}

    def record(
        self,
        episode: int,
        phase: Any,
        beta_raw: float,
        beta_deployed: float,
        episode_return: float,
        length: int,
        epsilon: float,
        alignment_rate: float,
        mean_signed_alignment: float,
        mean_advantage: float,
        mean_abs_advantage: float,
        mean_d_eff: float,
        median_d_eff: float,
        frac_d_eff_below_gamma: float,
        frac_d_eff_above_one: float,
        bellman_residual: float,
        td_target_abs_max: float,
        q_abs_max: float,
        catastrophic: bool,
        success: bool,
        regret: float,
        shift_event: bool,
        divergence_event: bool,
    ) -> None:
        cols = self._cols
        cols["run_id"].append(self._id.run_id)
        cols["env"].append(self._id.env)
        cols["method"].append(self._id.method)
        cols["seed"].append(int(self._id.seed))
        cols["episode"].append(int(episode))
        cols["phase"].append(str(phase))
        cols["beta_raw"].append(float(beta_raw))
        cols["beta_deployed"].append(float(beta_deployed))
        cols["return"].append(float(episode_return))
        cols["length"].append(int(length))
        cols["epsilon"].append(float(epsilon))
        cols["alignment_rate"].append(float(alignment_rate))
        cols["mean_signed_alignment"].append(float(mean_signed_alignment))
        cols["mean_advantage"].append(float(mean_advantage))
        cols["mean_abs_advantage"].append(float(mean_abs_advantage))
        cols["mean_d_eff"].append(float(mean_d_eff))
        cols["median_d_eff"].append(float(median_d_eff))
        cols["frac_d_eff_below_gamma"].append(float(frac_d_eff_below_gamma))
        cols["frac_d_eff_above_one"].append(float(frac_d_eff_above_one))
        cols["bellman_residual"].append(float(bellman_residual))
        cols["td_target_abs_max"].append(float(td_target_abs_max))
        cols["q_abs_max"].append(float(q_abs_max))
        cols["catastrophic"].append(bool(catastrophic))
        cols["success"].append(bool(success))
        cols["regret"].append(float(regret))
        cols["shift_event"].append(bool(shift_event))
        cols["divergence_event"].append(bool(divergence_event))

    def flush_csv(self, path: Path) -> int:
        """Write the buffered rows to ``path`` as CSV. Returns row count."""
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self._cols["episode"])
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(EPISODE_COLUMNS) + "\n")
            for i in range(n):
                row_vals = []
                for c in EPISODE_COLUMNS:
                    v = self._cols[c][i]
                    if isinstance(v, bool):
                        row_vals.append("1" if v else "0")
                    elif isinstance(v, float):
                        # Compact but loss-free round-trip for floats.
                        row_vals.append(repr(v))
                    else:
                        row_vals.append(str(v))
                f.write(",".join(row_vals) + "\n")
        return n

    def episode_returns(self) -> np.ndarray:
        """Return the per-episode return array (used by metrics.npz)."""
        return np.asarray(self._cols["return"], dtype=np.float64)

    def collected_arrays(self) -> Dict[str, np.ndarray]:
        """Return all numeric columns as a dict of numpy arrays."""
        out: Dict[str, np.ndarray] = {}
        for c in EPISODE_COLUMNS:
            data = self._cols[c]
            if c in {"run_id", "env", "method", "phase"}:
                out[c] = np.asarray(data, dtype=object)
            elif c in {
                "catastrophic", "success", "shift_event", "divergence_event",
            }:
                out[c] = np.asarray(data, dtype=bool)
            elif c in {"seed", "episode", "length"}:
                out[c] = np.asarray(data, dtype=np.int64)
            else:
                out[c] = np.asarray(data, dtype=np.float64)
        return out


__all__ = [
    "EPISODE_COLUMNS",
    "TRANSITION_COLUMNS",
    "SCHEMA_VERSION_EPISODES",
    "SCHEMA_VERSION_TRANSITIONS",
    "TransitionLogger",
    "EpisodeLogger",
    "_RunIdentity",
]
