"""Result-schema layout writer for Phase I/II/III weighted-LSE DP runs.

Responsibilities (spec §3.4, §7.1-7.4):

- Declare the exact array names required in every ``.npz`` artifact type
  (``curves.npz`` for both RL and DP modes, ``transitions.npz``,
  ``calibration_stats.npz``). These tuples double as contracts for
  downstream aggregation and as checklists for the validators below.
- Provide :class:`RunWriter`, a single orchestration object that a
  runner constructs once per ``(phase, suite, task, algorithm, seed)``
  combination and uses to accumulate RL / DP curve data, transitions,
  and calibration stats, flushing everything to disk in one atomic step.
- Provide ``validate_*`` helpers that aggregators and the verification
  suite use to confirm on-disk artifacts carry every mandatory key.

This module is the glue layer above ``io`` (raw writer primitives),
``manifests`` (``run.json`` / ``metrics.json``), and ``timing``
(``RunTimer``). It adds no new persistence primitives of its own; every
byte it puts on disk goes through one of those three modules.

The ``margin_beta0`` formula recorded in ``transitions.npz`` is
``reward - v_next_beta0`` (NO gamma factor).  This matches spec §7.1
and the paper's responsibility operator rho*(r,v) = sigma(beta*(r-v)
+ log(1/gamma)), which depends on (r - v), not (r - gamma*v).
The TD target ``td_target_beta0 = reward + gamma * v_next_beta0``
remains standard Bellman.  The two objects serve different roles:
margin drives allocation geometry; TD target drives value updates.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# The ``common/`` directory has no ``__init__.py`` (runs are invoked as
# scripts), so pull in sibling helpers by inserting the repo root on
# ``sys.path``. This mirrors the pattern used in ``manifests.py`` /
# ``timing.py``; see those files for the layout explanation.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.io import (  # noqa: E402
    SCHEMA_VERSION,
    make_npz_schema,
    make_run_dir,
    save_json,
    save_npz_with_schema,
)
from experiments.weighted_lse_dp.common.manifests import (  # noqa: E402
    write_metrics_json,
    write_run_json,
)
from experiments.weighted_lse_dp.common.timing import RunTimer  # noqa: E402


__all__ = [
    "CURVES_ARRAYS_RL",
    "CURVES_ARRAYS_DP",
    "TRANSITIONS_ARRAYS",
    "CALIBRATION_ARRAYS",
    "MARGIN_BETA0_FORMULA",
    "RunWriter",
    "validate_transitions_npz",
    "validate_calibration_npz",
    "validate_curves_npz",
]


# ----------------------------------------------------------------------------
# NPZ array-name contracts (spec §7.1-7.4)
# ----------------------------------------------------------------------------


CURVES_ARRAYS_RL: tuple[str, ...] = (
    "checkpoints",          # (n_checkpoints,) int64  - env steps at each checkpoint
    "disc_return_mean",     # (n_checkpoints,) float64 - mean discounted return over eval episodes
    "disc_return_std",      # (n_checkpoints,) float64
    "undisc_return_mean",   # (n_checkpoints,) float64
    "undisc_return_std",    # (n_checkpoints,) float64
    "success_rate",         # (n_checkpoints,) float64
)
"""Array names in ``curves.npz`` when the run is an online RL run."""


CURVES_ARRAYS_DP: tuple[str, ...] = (
    "sweep_index",          # (n_sweeps,) int64
    "bellman_residual",     # (n_sweeps,) float64
    "supnorm_to_exact",     # (n_sweeps,) float64  - ||V_sweep - V_exact||_inf
    "wall_clock_s",         # (n_sweeps,) float64  - cumulative
    "v_table_snapshots",    # (n_sweeps, H+1, S) float64 - chain task only; empty otherwise
)
"""Array names in ``curves.npz`` when the run is an exact-DP sweep run."""


TRANSITIONS_ARRAYS: tuple[str, ...] = (
    "episode_index",        # (N,) int64
    "t",                    # (N,) int64  - stage within episode
    "state",                # (N,) int64  - base (un-augmented) state id
    "action",               # (N,) int64
    "reward",               # (N,) float64
    "next_state",           # (N,) int64
    "absorbing",            # (N,) bool
    "last",                 # (N,) bool
    "q_current_beta0",      # (N,) float64
    "v_next_beta0",         # (N,) float64
    "margin_beta0",         # (N,) float64  = reward - v_next_beta0  (no gamma)
    "td_target_beta0",      # (N,) float64  = reward + gamma * v_next_beta0
    "td_error_beta0",       # (N,) float64
)
"""Array names in ``transitions.npz`` (per-transition calibration log)."""


CALIBRATION_ARRAYS: tuple[str, ...] = (
    "stage",                # (H+1,) int64  - stage indices 0..H
    "count",                # (H+1,) int64  - samples at each stage
    "reward_mean",          # (H+1,) float64
    "reward_std",           # (H+1,) float64
    "v_next_mean",          # (H+1,) float64
    "v_next_std",           # (H+1,) float64
    "margin_q05",           # (H+1,) float64
    "margin_q25",           # (H+1,) float64
    "margin_q50",           # (H+1,) float64
    "margin_q75",           # (H+1,) float64
    "margin_q95",           # (H+1,) float64
    "pos_margin_mean",      # (H+1,) float64  - mean of max(margin, 0)
    "neg_margin_mean",      # (H+1,) float64  - mean of max(-margin, 0)
    "max_abs_v_next",       # (H+1,) float64
    "max_abs_q_current",    # (H+1,) float64
    "bellman_residual_mean",# (H+1,) float64  - NaN if exact DP not available
    "bellman_residual_std", # (H+1,) float64
    "aligned_margin_freq",  # (H+1,) float64  - fraction of transitions with margin_beta0 > 0
)
"""Array names in ``calibration_stats.npz`` (aggregated per-stage stats)."""


MARGIN_BETA0_FORMULA: str = "reward - v_next_beta0"
"""Formula used to compute ``margin_beta0``; stamped into the schema header.

The ticket resolved this to the standard TD margin; spec §7.1 writes it
without the ``gamma`` factor. The string is recorded verbatim in every
``transitions.npz`` header so any downstream consumer can recover the
exact convention that produced the on-disk numbers.
"""


# ----------------------------------------------------------------------------
# RunWriter
# ----------------------------------------------------------------------------


@dataclass
class _RLCheckpointBuffer:
    """In-memory staging area for :meth:`RunWriter.record_rl_checkpoint`.

    Each call appends one checkpoint's eval-episode summaries; at flush
    time the lists are stacked into the six arrays in
    :data:`CURVES_ARRAYS_RL`.
    """

    checkpoints: list[int] = field(default_factory=list)
    disc_return_mean: list[float] = field(default_factory=list)
    disc_return_std: list[float] = field(default_factory=list)
    undisc_return_mean: list[float] = field(default_factory=list)
    undisc_return_std: list[float] = field(default_factory=list)
    success_rate: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.checkpoints)


@dataclass
class _DPSweepBuffer:
    """In-memory staging area for :meth:`RunWriter.record_dp_sweep`.

    ``v_table_snapshots`` is a list of 2-D arrays (one per sweep); at
    flush time they are stacked into a 3-D ``(n_sweeps, H+1, S)`` array.
    If every snapshot is ``None`` (i.e. the task is not the chain task),
    the flushed array is a 1-D empty ``float64`` array of length 0.
    """

    sweep_index: list[int] = field(default_factory=list)
    bellman_residual: list[float] = field(default_factory=list)
    supnorm_to_exact: list[float] = field(default_factory=list)
    wall_clock_s: list[float] = field(default_factory=list)
    v_table_snapshots: list[np.ndarray | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sweep_index)


class RunWriter:
    """Orchestrate every on-disk artifact for one seed's run.

    A runner constructs this once at the top of a seed's execution,
    hands ``rw.timer`` to whatever timing scopes it uses, records
    checkpoints / sweeps / transitions / calibration stats as they
    arrive, and calls :meth:`flush` once at the end (or from a
    ``finally`` block on failure) to persist everything.

    The writer never silently drops data: if a key is missing from a
    transitions / calibration payload it raises at flush time with the
    list of missing names. Callers that need partial writes should
    construct an explicit zero-filled payload rather than omitting keys.

    Attributes
    ----------
    run_dir:
        The canonical seed directory (``<base>/<phase>/<suite>/<task>/
        <algorithm>/seed_<seed>/``). Created at construction.
    timer:
        A fresh :class:`RunTimer` the caller should use for all timing
        scopes. At :meth:`flush` the timer is serialised to
        ``timings.json``.
    """

    # Attribute type hints for static analysers.
    run_dir: Path
    timer: RunTimer

    def __init__(
        self,
        run_dir: Path,
        *,
        phase: str,
        suite: str,
        task: str,
        algorithm: str,
        seed: int,
        config: dict[str, Any],
        storage_mode: str,
    ) -> None:
        """Low-level constructor; prefer :meth:`create`.

        The :meth:`create` classmethod builds the run directory and
        instantiates the writer together. Direct construction is useful
        when an existing run directory should be reused (e.g. in
        unit tests that pre-create a tempdir).
        """
        self.run_dir = Path(run_dir)
        self.timer = RunTimer()

        self._phase = str(phase)
        self._suite = str(suite)
        self._task = str(task)
        self._algorithm = str(algorithm)
        self._seed = int(seed)
        self._config = config
        self._storage_mode = str(storage_mode)

        # Per-artifact staging buffers. All are optional: a DP-only run
        # leaves the RL buffer empty and vice versa; a run without
        # transitions never calls :meth:`set_transitions`.
        self._rl_buf = _RLCheckpointBuffer()
        self._dp_buf = _DPSweepBuffer()
        self._transitions: dict[str, np.ndarray] | None = None
        self._calibration: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        base: Path | str,
        phase: str,
        suite: str,
        task: str,
        algorithm: str,
        seed: int,
        config: dict[str, Any],
        storage_mode: str = "rl_online",
    ) -> "RunWriter":
        """Create the run directory and return a fresh :class:`RunWriter`.

        Wraps :func:`common.io.make_run_dir` with ``exist_ok=True`` so
        resumed runs do not error out; a fresh writer always clobbers
        any previously-staged buffers because those live only in
        memory.

        Parameters
        ----------
        base:
            Root under which the run tree is built. Typically
            :data:`common.io.RESULT_ROOT`; tests pass a tempdir.
        phase, suite, task, algorithm, seed:
            Identify the run. Directory layout mirrors these verbatim.
        config:
            Fully-resolved config dict; stored inside ``run.json`` and
            written verbatim to ``config.json``.
        storage_mode:
            Free-form tag written into every ``.npz`` header. Common
            values: ``"rl_online"`` (per-transition logs),
            ``"dp_stagewise"`` (sweep-based planners).
        """
        run_dir = make_run_dir(
            base=base,
            phase=phase,
            suite=suite,
            task=task,
            algorithm=algorithm,
            seed=seed,
            exist_ok=True,
        )
        return cls(
            run_dir=run_dir,
            phase=phase,
            suite=suite,
            task=task,
            algorithm=algorithm,
            seed=seed,
            config=config,
            storage_mode=storage_mode,
        )

    # ------------------------------------------------------------------
    # RL curve accumulation
    # ------------------------------------------------------------------

    def record_rl_checkpoint(
        self,
        steps: int,
        disc_returns: Sequence[float],
        undisc_returns: Sequence[float],
        successes: Sequence[bool],
    ) -> None:
        """Append one checkpoint's eval summaries to the RL curve buffer.

        ``disc_returns`` / ``undisc_returns`` / ``successes`` are the raw
        per-episode eval results; this method reduces them to the
        ``mean`` / ``std`` / ``rate`` scalars that populate
        :data:`CURVES_ARRAYS_RL`. The buffer is flushed at :meth:`flush`
        into ``curves.npz``.

        Parameters
        ----------
        steps:
            Total environment steps at the time this checkpoint fires.
        disc_returns:
            Discounted eval-episode returns (one scalar per episode).
        undisc_returns:
            Undiscounted eval-episode returns (one scalar per episode).
        successes:
            Per-episode success flags; the ``success_rate`` row is the
            arithmetic mean of these (float in ``[0, 1]``).

        Raises
        ------
        ValueError
            If the three sequences disagree in length, or if
            ``disc_returns`` is empty (an eval with no episodes would
            produce a NaN summary and is almost always a bug).
        """
        if not isinstance(steps, (int, np.integer)):
            raise TypeError(f"steps must be int, got {type(steps).__name__}")

        d = np.asarray(list(disc_returns), dtype=np.float64)
        u = np.asarray(list(undisc_returns), dtype=np.float64)
        s = np.asarray(list(successes), dtype=bool)

        if d.size == 0:
            raise ValueError("disc_returns must contain at least one episode")
        if d.shape != u.shape or d.shape != s.shape:
            raise ValueError(
                f"disc/undisc/success shapes disagree: "
                f"{d.shape}, {u.shape}, {s.shape}"
            )

        self._rl_buf.checkpoints.append(int(steps))
        self._rl_buf.disc_return_mean.append(float(d.mean()))
        # ``ddof=0`` matches numpy default and the spec's "std across eval
        # episodes" phrasing.
        self._rl_buf.disc_return_std.append(float(d.std()))
        self._rl_buf.undisc_return_mean.append(float(u.mean()))
        self._rl_buf.undisc_return_std.append(float(u.std()))
        self._rl_buf.success_rate.append(float(s.mean()))

    # ------------------------------------------------------------------
    # DP sweep accumulation
    # ------------------------------------------------------------------

    def record_dp_sweep(
        self,
        sweep_idx: int,
        bellman_residual: float,
        supnorm_to_exact: float | None,
        wall_clock_s: float,
        v_table_snapshot: np.ndarray | None = None,
    ) -> None:
        """Append one DP sweep row to the DP curve buffer.

        ``supnorm_to_exact`` is allowed to be ``None`` (e.g. tasks where
        no closed-form exact value function is available); it is stored
        as ``NaN`` in the flushed array so downstream code can rely on
        a fixed dtype.

        ``v_table_snapshot`` is kept as-is in the buffer and stacked at
        flush time. It should be 2-D with shape ``(H+1, S)`` for the
        chain task; pass ``None`` for every other task.

        Parameters
        ----------
        sweep_idx:
            Monotonic index of this sweep (starting at 0 or 1 - caller's
            choice, recorded verbatim).
        bellman_residual:
            Sweep's Bellman residual (spec §7.3).
        supnorm_to_exact:
            ``||V_sweep - V_exact||_inf`` when exact values are known,
            else ``None``.
        wall_clock_s:
            Cumulative wall-clock at the end of this sweep (typically
            from :class:`common.timing.SweepTimer.cumulative_s`).
        v_table_snapshot:
            Optional 2-D value table snapshot; required for the chain
            task, ignored otherwise.
        """
        if not isinstance(sweep_idx, (int, np.integer)):
            raise TypeError(
                f"sweep_idx must be int, got {type(sweep_idx).__name__}"
            )

        snap: np.ndarray | None
        if v_table_snapshot is None:
            snap = None
        else:
            snap = np.asarray(v_table_snapshot, dtype=np.float64)
            if snap.ndim != 2:
                raise ValueError(
                    f"v_table_snapshot must be 2-D (H+1, S), "
                    f"got shape {snap.shape!r}"
                )

        self._dp_buf.sweep_index.append(int(sweep_idx))
        self._dp_buf.bellman_residual.append(float(bellman_residual))
        self._dp_buf.supnorm_to_exact.append(
            float("nan") if supnorm_to_exact is None else float(supnorm_to_exact)
        )
        self._dp_buf.wall_clock_s.append(float(wall_clock_s))
        self._dp_buf.v_table_snapshots.append(snap)

    # ------------------------------------------------------------------
    # Bulk payload setters
    # ------------------------------------------------------------------

    def set_transitions(self, transitions: dict[str, np.ndarray]) -> None:
        """Stage the full transitions payload for ``transitions.npz``.

        The payload must contain every key in :data:`TRANSITIONS_ARRAYS`;
        extra keys are accepted and written verbatim. The payload is
        kept in memory until :meth:`flush`. Calling this method twice
        overwrites the previous payload (intentional: a mid-run recount
        is a valid pattern).

        Raises
        ------
        KeyError
            If any required key from :data:`TRANSITIONS_ARRAYS` is
            missing. The error message lists every missing key.
        """
        self._transitions = _validate_and_copy_payload(
            payload=transitions,
            required=TRANSITIONS_ARRAYS,
            payload_name="transitions",
        )

    def set_calibration_stats(self, stats: dict[str, np.ndarray]) -> None:
        """Stage the per-stage calibration payload for ``calibration_stats.npz``.

        The payload must contain every key in :data:`CALIBRATION_ARRAYS`;
        extra keys are accepted and written verbatim.

        Raises
        ------
        KeyError
            If any required key is missing (see :data:`CALIBRATION_ARRAYS`).
        """
        self._calibration = _validate_and_copy_payload(
            payload=stats,
            required=CALIBRATION_ARRAYS,
            payload_name="calibration",
        )

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(
        self,
        metrics: dict[str, float | int | str | None],
        step_count: int | None = None,
        update_count: int | None = None,
    ) -> None:
        """Persist every staged artifact to :attr:`run_dir`.

        Writes (in order):

        1. ``run.json``   - canonical run header (via
           :func:`common.manifests.write_run_json`).
        2. ``config.json``- resolved config, stored standalone for
           aggregators that do not want to parse the full ``run.json``.
        3. ``metrics.json``- scalar summary (via
           :func:`common.manifests.write_metrics_json`).
        4. ``timings.json``- :attr:`timer` serialised via
           :meth:`common.timing.RunTimer.save`.
        5. ``curves.npz`` - whichever of the RL / DP buffers is
           non-empty; both buffers populating the same file is allowed
           and flags both modes in the schema header.
        6. ``transitions.npz`` - if :meth:`set_transitions` was called.
        7. ``calibration_stats.npz`` - if :meth:`set_calibration_stats`
           was called.

        Parameters
        ----------
        metrics:
            Flat dict of scalar metrics for ``metrics.json``.
        step_count:
            Optional total env-step count; forwarded to
            :meth:`RunTimer.set_step_count` so ``timings.json`` carries
            the derived ``steps_per_s`` field.
        update_count:
            Optional total agent-update count; forwarded to
            :meth:`RunTimer.set_update_count` for ``updates_per_s``.
        """
        if step_count is not None:
            self.timer.set_step_count(int(step_count))
        if update_count is not None:
            self.timer.set_update_count(int(update_count))

        # 1. run.json
        write_run_json(
            self.run_dir,
            config=self._config,
            phase=self._phase,
            task=self._task,
            algorithm=self._algorithm,
            seed=self._seed,
            extra={"suite": self._suite, "storage_mode": self._storage_mode},
        )

        # 2. config.json (standalone copy of the resolved config)
        save_json(self.run_dir / "config.json", self._config)

        # 3. metrics.json
        write_metrics_json(
            self.run_dir,
            metrics,
            phase=self._phase,
            task=self._task,
            algorithm=self._algorithm,
            seed=self._seed,
        )

        # 4. timings.json
        self.timer.save(self.run_dir / "timings.json")

        # 5. curves.npz (may contain RL keys, DP keys, or both)
        if len(self._rl_buf) > 0 or len(self._dp_buf) > 0:
            self._flush_curves()

        # 6. transitions.npz
        if self._transitions is not None:
            self._flush_transitions()

        # 7. calibration_stats.npz
        if self._calibration is not None:
            self._flush_calibration()

    # ------------------------------------------------------------------
    # Internal flush helpers
    # ------------------------------------------------------------------

    def _flush_curves(self) -> None:
        """Write ``curves.npz`` with whichever mode(s) were recorded."""
        arrays: dict[str, np.ndarray] = {}
        modes: list[str] = []

        if len(self._rl_buf) > 0:
            modes.append("rl")
            arrays["checkpoints"] = np.asarray(
                self._rl_buf.checkpoints, dtype=np.int64
            )
            arrays["disc_return_mean"] = np.asarray(
                self._rl_buf.disc_return_mean, dtype=np.float64
            )
            arrays["disc_return_std"] = np.asarray(
                self._rl_buf.disc_return_std, dtype=np.float64
            )
            arrays["undisc_return_mean"] = np.asarray(
                self._rl_buf.undisc_return_mean, dtype=np.float64
            )
            arrays["undisc_return_std"] = np.asarray(
                self._rl_buf.undisc_return_std, dtype=np.float64
            )
            arrays["success_rate"] = np.asarray(
                self._rl_buf.success_rate, dtype=np.float64
            )

        if len(self._dp_buf) > 0:
            modes.append("dp")
            arrays["sweep_index"] = np.asarray(
                self._dp_buf.sweep_index, dtype=np.int64
            )
            arrays["bellman_residual"] = np.asarray(
                self._dp_buf.bellman_residual, dtype=np.float64
            )
            arrays["supnorm_to_exact"] = np.asarray(
                self._dp_buf.supnorm_to_exact, dtype=np.float64
            )
            arrays["wall_clock_s"] = np.asarray(
                self._dp_buf.wall_clock_s, dtype=np.float64
            )
            arrays["v_table_snapshots"] = _stack_v_snapshots(
                self._dp_buf.v_table_snapshots
            )

        schema = make_npz_schema(
            phase=self._phase,
            task=self._task,
            algorithm=self._algorithm,
            seed=self._seed,
            storage_mode=self._storage_mode,
            arrays=list(arrays.keys()),
        )
        schema["curves_modes"] = modes
        save_npz_with_schema(self.run_dir / "curves.npz", schema, arrays)

    def _flush_transitions(self) -> None:
        """Write ``transitions.npz`` and stamp the margin formula."""
        assert self._transitions is not None  # guarded in flush()
        schema = make_npz_schema(
            phase=self._phase,
            task=self._task,
            algorithm=self._algorithm,
            seed=self._seed,
            storage_mode=self._storage_mode,
            arrays=list(self._transitions.keys()),
        )
        # Preserving the formula in the header lets downstream code
        # reconstruct margins from first principles and flags the
        # spec/ticket convention mismatch explicitly.
        schema["margin_beta0_formula"] = MARGIN_BETA0_FORMULA
        save_npz_with_schema(
            self.run_dir / "transitions.npz", schema, self._transitions
        )

    def _flush_calibration(self) -> None:
        """Write ``calibration_stats.npz``."""
        assert self._calibration is not None  # guarded in flush()
        schema = make_npz_schema(
            phase=self._phase,
            task=self._task,
            algorithm=self._algorithm,
            seed=self._seed,
            storage_mode=self._storage_mode,
            arrays=list(self._calibration.keys()),
        )
        save_npz_with_schema(
            self.run_dir / "calibration_stats.npz", schema, self._calibration
        )


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _validate_and_copy_payload(
    payload: dict[str, np.ndarray],
    required: Iterable[str],
    payload_name: str,
) -> dict[str, np.ndarray]:
    """Check that every ``required`` key is present and return a shallow copy.

    Extra keys are retained verbatim. Values are converted via
    :func:`numpy.asarray` without dtype coercion so each payload
    preserves its caller-chosen precision.
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"{payload_name} payload must be dict, got "
            f"{type(payload).__name__}"
        )

    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(
            f"{payload_name} payload missing required keys: {missing!r}"
        )

    # Shallow copy with asarray for each value: avoids the caller
    # mutating our staged payload after the call returns, while still
    # leaving dtype control in the caller's hands.
    return {name: np.asarray(arr) for name, arr in payload.items()}


def _stack_v_snapshots(
    snapshots: list[np.ndarray | None],
) -> np.ndarray:
    """Stack per-sweep value-table snapshots into a 3-D array.

    If every snapshot is ``None`` (tasks other than the chain task),
    returns a 1-D empty ``float64`` array so the key can be present in
    the .npz file with a fixed dtype. When some but not all snapshots
    are ``None`` we raise: that inconsistency almost always means the
    runner is buggy.
    """
    total = len(snapshots)
    if total == 0:
        return np.empty(0, dtype=np.float64)

    nones = sum(1 for s in snapshots if s is None)
    if nones == total:
        return np.empty(0, dtype=np.float64)
    if nones != 0:
        raise ValueError(
            f"v_table_snapshots is partially populated "
            f"({total - nones}/{total} non-None); either record every "
            f"sweep or none of them."
        )

    # All non-None: validate consistent shape before stacking.
    shapes = {s.shape for s in snapshots if s is not None}
    if len(shapes) != 1:
        raise ValueError(
            f"v_table_snapshots have inconsistent shapes: {shapes!r}"
        )
    return np.stack(
        [np.asarray(s, dtype=np.float64) for s in snapshots if s is not None],
        axis=0,
    )


# ----------------------------------------------------------------------------
# Validators
# ----------------------------------------------------------------------------


def _missing_keys(path: Path, required: Iterable[str]) -> list[str]:
    """Return the sorted list of ``required`` keys absent from ``path``.

    Uses ``allow_pickle=False`` via :func:`common.io.load_npz`. The
    ``_schema`` header is ignored for presence checks; only data arrays
    count.
    """
    # Local import to avoid a circular import during module init: io
    # does not import schemas, but co-locating the import with its sole
    # call site keeps the module graph minimal.
    from experiments.weighted_lse_dp.common.io import load_npz

    path = Path(path)
    loaded = load_npz(path)
    present = set(loaded.keys()) - {"_schema"}
    return sorted(k for k in required if k not in present)


def validate_transitions_npz(path: Path) -> list[str]:
    """Return the list of :data:`TRANSITIONS_ARRAYS` keys missing from ``path``.

    An empty list means the file is valid with respect to the schema
    contract. Extra keys are ignored.
    """
    return _missing_keys(path, TRANSITIONS_ARRAYS)


def validate_calibration_npz(path: Path) -> list[str]:
    """Return the list of :data:`CALIBRATION_ARRAYS` keys missing from ``path``.

    An empty list means the file is valid.
    """
    return _missing_keys(path, CALIBRATION_ARRAYS)


def validate_curves_npz(path: Path, mode: str) -> list[str]:
    """Return the list of curve-array keys missing from ``path`` for ``mode``.

    Parameters
    ----------
    path:
        The ``.npz`` path to validate.
    mode:
        ``"rl"`` checks against :data:`CURVES_ARRAYS_RL`; ``"dp"``
        checks against :data:`CURVES_ARRAYS_DP`. Any other value raises
        :class:`ValueError`.
    """
    if mode == "rl":
        required: tuple[str, ...] = CURVES_ARRAYS_RL
    elif mode == "dp":
        required = CURVES_ARRAYS_DP
    else:
        raise ValueError(f"mode must be 'rl' or 'dp', got {mode!r}")
    return _missing_keys(path, required)
