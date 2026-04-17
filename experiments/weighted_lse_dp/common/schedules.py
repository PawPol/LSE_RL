"""Per-stage beta schedule container and loader for weighted LSE runners.

# ============================================================
# PHASE I STUB -- real schedule building lives in Phase III.
# calibration-engineer fills in:  build_schedule(calibration_stats_path, ...) -> BetaSchedule
# DO NOT implement calibration logic here until Phase III planning.
# ============================================================

Responsibilities (Phase I):
    - Provide a typed :class:`BetaSchedule` record that runners can pass
      down into operators without branching on "safe vs classical".
    - Provide :func:`zero_schedule` (all-zeros beta) so Phase I classical
      runs always have a concrete schedule object with the correct shape.
    - Provide JSON serialisation so Phase III calibrated schedules can be
      saved alongside ``calibration_stats.npz`` and reloaded verbatim.
    - Provide :func:`load_schedule_or_zero` so a runner works identically
      in Phase I (no file on disk -> zeros) and in Phase III (file present
      -> calibrated betas), without changing the call site.

Responsibilities (Phase III, deliberately *not* implemented here):
    - ``build_schedule(calibration_stats_path, ...) -> BetaSchedule``
      is the calibration-engineer's contract. It will consume the stats
      produced by the calibration pass and emit per-stage betas. That
      logic belongs in a dedicated ``calibration/`` module and must not
      leak into this file.

Dependencies: stdlib (``json``, ``pathlib``, ``dataclasses``) + numpy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

SCHEDULE_VERSION: str = "1.0.0"
"""Schema version embedded in every serialised :class:`BetaSchedule`."""


_ALLOWED_SOURCES: frozenset[str] = frozenset({"zero", "calibrated", "manual"})
"""Permitted values for :attr:`BetaSchedule.source`."""


# ----------------------------------------------------------------------------
# Dataclass
# ----------------------------------------------------------------------------


@dataclass
class BetaSchedule:
    """Per-stage beta schedule for one ``(task, algorithm)`` pair.

    In Phase I every runner uses :func:`zero_schedule`, i.e. ``betas`` is
    uniformly ``0.0`` and the weighted LSE operator degenerates to the
    classical max/expectation baseline. In Phase III the calibration
    engineer populates ``betas`` from ``calibration_stats.npz`` via a
    yet-to-be-written ``build_schedule`` function; the on-disk format is
    defined by :meth:`to_dict` / :meth:`from_dict`.

    Attributes
    ----------
    task:
        Task identifier (e.g. ``"chain_base"``).
    algorithm:
        Algorithm identifier (e.g. ``"QLearning"``).
    phase:
        Phase tag the schedule was generated under (e.g. ``"phase1"``).
    betas:
        Shape ``(horizon + 1,)`` float64 array. Entry ``t`` is the beta
        used at stage ``t``; the trailing entry corresponds to the
        terminal stage and is kept for symmetry with stage-indexed
        operators.
    schema_version:
        Matches :data:`SCHEDULE_VERSION` at write time. Readers should
        reject files whose major version disagrees.
    source:
        Provenance tag. One of ``"zero"`` (Phase I classical default),
        ``"calibrated"`` (Phase III build_schedule output), or
        ``"manual"`` (hand-authored for ablations / unit tests).
    """

    task: str
    algorithm: str
    phase: str
    betas: np.ndarray
    schema_version: str = SCHEDULE_VERSION
    source: str = "zero"

    def __post_init__(self) -> None:
        # Normalise to a contiguous float64 array so downstream operators
        # can rely on a predictable dtype / memory layout.
        arr = np.ascontiguousarray(self.betas, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"betas must be 1-D, got shape {arr.shape!r}"
            )
        if arr.size == 0:
            raise ValueError("betas must contain at least one entry")
        if not np.all(np.isfinite(arr)):
            raise ValueError("betas must be finite (no NaN/inf)")
        self.betas = arr

        if self.source not in _ALLOWED_SOURCES:
            raise ValueError(
                f"source must be one of {sorted(_ALLOWED_SOURCES)!r}, "
                f"got {self.source!r}"
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def beta_at(self, t: int) -> float:
        """Return ``beta`` for stage ``t``, clamped to ``[0, horizon]``.

        Clamping matters because some RL runners briefly step past the
        nominal horizon (e.g. when logging a terminal transition). We
        prefer silently returning the last entry over raising, so a
        logging path never crashes a run.
        """
        if not isinstance(t, (int, np.integer)):
            raise TypeError(f"t must be int, got {type(t).__name__}")
        idx = int(t)
        if idx < 0:
            idx = 0
        last = self.betas.shape[0] - 1
        if idx > last:
            idx = last
        return float(self.betas[idx])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (``betas`` as a Python list)."""
        return {
            "schema_version": str(self.schema_version),
            "task": str(self.task),
            "algorithm": str(self.algorithm),
            "phase": str(self.phase),
            "source": str(self.source),
            "betas": [float(x) for x in self.betas.tolist()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BetaSchedule":
        """Reconstruct from :meth:`to_dict` output.

        Missing ``schema_version`` defaults to :data:`SCHEDULE_VERSION`
        for forward-compat; missing ``source`` defaults to ``"manual"``
        because anything hand-crafted enough to omit it is, by
        definition, not calibrated.
        """
        required = ("task", "algorithm", "phase", "betas")
        missing = [k for k in required if k not in d]
        if missing:
            raise KeyError(f"BetaSchedule dict missing keys: {missing!r}")

        return cls(
            task=str(d["task"]),
            algorithm=str(d["algorithm"]),
            phase=str(d["phase"]),
            betas=np.asarray(d["betas"], dtype=np.float64),
            schema_version=str(d.get("schema_version", SCHEDULE_VERSION)),
            source=str(d.get("source", "manual")),
        )


# ----------------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------------


def zero_schedule(task: str, algorithm: str, horizon: int) -> BetaSchedule:
    """Return an all-zeros :class:`BetaSchedule` of length ``horizon + 1``.

    This is the Phase I classical baseline: beta is zero everywhere, so
    the weighted LSE operator collapses to the classical max/expectation
    it shadows. Runners call this unconditionally when no calibrated
    schedule is on disk, via :func:`load_schedule_or_zero`.

    Parameters
    ----------
    task:
        Task identifier stored on the schedule.
    algorithm:
        Algorithm identifier stored on the schedule.
    horizon:
        Non-negative integer; the produced ``betas`` has shape
        ``(horizon + 1,)``.
    """
    if not isinstance(horizon, (int, np.integer)):
        raise TypeError(f"horizon must be int, got {type(horizon).__name__}")
    h = int(horizon)
    if h < 0:
        raise ValueError(f"horizon must be non-negative, got {h}")

    return BetaSchedule(
        task=str(task),
        algorithm=str(algorithm),
        phase="phase1",
        betas=np.zeros(h + 1, dtype=np.float64),
        schema_version=SCHEDULE_VERSION,
        source="zero",
    )


# ----------------------------------------------------------------------------
# JSON I/O
# ----------------------------------------------------------------------------


def save_schedule(path: str | Path, schedule: BetaSchedule) -> None:
    """Write ``schedule`` as indented JSON to ``path``.

    Parent directories are created as needed. The on-disk format is the
    dict returned by :meth:`BetaSchedule.to_dict`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(schedule.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_schedule(path: str | Path) -> BetaSchedule:
    """Load a :class:`BetaSchedule` from a JSON file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. Callers that want to gracefully fall
        back to a zero schedule should use :func:`load_schedule_or_zero`
        instead.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return BetaSchedule.from_dict(payload)


def load_schedule_or_zero(
    path: str | Path,
    task: str,
    algorithm: str,
    horizon: int,
) -> BetaSchedule:
    """Load a schedule from ``path`` or fall back to :func:`zero_schedule`.

    Phase I runners call this unconditionally: if a calibrated schedule
    has been produced (Phase III), it is loaded verbatim; otherwise the
    runner continues with all-zero betas, which recovers the classical
    baseline. The fallback is silent by design -- absence of a file is
    not an error in Phase I.
    """
    p = Path(path)
    if not p.exists():
        return zero_schedule(task=task, algorithm=algorithm, horizon=horizon)
    return load_schedule(p)


# ----------------------------------------------------------------------------
# Phase III placeholder (intentionally not implemented)
# ----------------------------------------------------------------------------


def build_schedule(*args: Any, **kwargs: Any) -> BetaSchedule:  # noqa: D401
    """Phase III entry point -- NOT implemented in Phase I.

    The calibration-engineer will implement this against
    ``calibration_stats.npz`` during Phase III. Keeping the symbol here
    with a clear error makes premature calls fail loudly instead of
    silently constructing a wrong schedule.
    """
    raise NotImplementedError(
        "build_schedule is a Phase III responsibility; in Phase I use "
        "zero_schedule() or load_schedule_or_zero()."
    )


# ----------------------------------------------------------------------------
# Explicit exports
# ----------------------------------------------------------------------------

__all__ = [
    "SCHEDULE_VERSION",
    "BetaSchedule",
    "zero_schedule",
    "save_schedule",
    "load_schedule",
    "load_schedule_or_zero",
    "build_schedule",
]
