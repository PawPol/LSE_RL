"""Timing utilities for Phase I/II/III runs.

Responsibilities (spec §11.2):
    Always record with :func:`time.perf_counter`:

    - total runtime,
    - time in environment stepping,
    - time in fitting / updating,
    - time in evaluation,
    - time in plotting / aggregation.

    Results are written to ``timings.json`` (spec §3.4) via
    :func:`experiments.weighted_lse_dp.common.io.save_json`.

This module is stdlib-only plus the sibling ``io.py`` helper. It exposes:

- :func:`timer` - a generic re-entrant ``@contextmanager`` that accumulates
  elapsed seconds into a per-instance registry (kept for ad-hoc scripts and
  tests).
- :class:`RunTimer` - the primary aggregator: one instance per run; supports
  the canonical phases ``step``, ``fit``, ``eval``, ``plot``, ``other``;
  exposes ``to_dict()`` / ``save(path)`` that emit the flat schema written
  to ``timings.json``.
- :class:`SweepTimer` - a lightweight per-sweep timer used by DP planners;
  exposes per-sweep and cumulative wall-clock arrays plus summary stats.
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Generator

# The ``common/`` directory has no ``__init__.py`` (runs are invoked as
# scripts), so pull in ``io.save_json`` by inserting the repository root
# on ``sys.path``. The path is this file's great-great-grandparent:
#   experiments/weighted_lse_dp/common/timing.py
#   parents[0] -> common/
#   parents[1] -> weighted_lse_dp/
#   parents[2] -> experiments/
#   parents[3] -> <repo-root>
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.io import save_json  # noqa: E402


__all__ = ["RunTimer", "SweepTimer", "timer"]


# ----------------------------------------------------------------------------
# Generic accumulating timer
# ----------------------------------------------------------------------------


# Module-level registry for the bare :func:`timer` context manager. Kept as
# a plain dict keyed by label so ad-hoc scripts can do::
#
#     with timer("fit"):
#         ...
#     timer_registry["fit"]  # accumulated seconds
#
# For production runs, use :class:`RunTimer` instead, which keeps its own
# registry on the instance.
_GLOBAL_REGISTRY: dict[str, float] = {}


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Accumulate elapsed wall-clock seconds into a shared module registry.

    On exit, the measured interval is added to ``_GLOBAL_REGISTRY[label]``.
    The registry is a process-wide dict; for run-scoped bookkeeping prefer
    :class:`RunTimer`.

    Parameters
    ----------
    label:
        Free-form string key used to accumulate this interval. Any string
        is accepted.

    Yields
    ------
    None
        This is a pure timing context manager; no resource is returned.
    """
    if not isinstance(label, str):
        raise TypeError(f"label must be str, got {type(label).__name__}")

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _GLOBAL_REGISTRY[label] = _GLOBAL_REGISTRY.get(label, 0.0) + elapsed


# ----------------------------------------------------------------------------
# RunTimer
# ----------------------------------------------------------------------------


class RunTimer:
    """Accumulate wall-clock time for the named phases of a single run.

    One instance per run. All intervals are measured via
    :func:`time.perf_counter`. Phase names are free-form; the canonical
    set for Phase I/II/III (spec §11.2) is exposed as :attr:`PHASES`, and
    entering an unknown phase issues a :class:`UserWarning` rather than
    raising, so callers can extend the schema without breaking runs.

    ``to_dict()`` returns a flat, JSON-serialisable dict with the keys
    documented in :meth:`to_dict`; ``save(path)`` writes that dict to
    ``timings.json`` via :func:`common.io.save_json`.

    Notes
    -----
    - :meth:`total` returns wall-clock seconds from the first ``phase``
      entry to the last ``phase`` exit (i.e. the span of observed
      activity), not the sum of per-phase elapsed times. This matters
      when phases overlap (nested ``with`` blocks) or when the run
      contains untimed gaps.
    - Nested ``phase()`` blocks for the same name are safe: each one is
      timed independently and added to the accumulator.

    Examples
    --------
    >>> rt = RunTimer()
    >>> with rt.phase("step"):
    ...     ...  # env.step(action)
    >>> with rt.phase("fit"):
    ...     ...  # agent.fit(dataset)
    >>> rt.set_step_count(1000)
    >>> rt.to_dict()["steps_per_s"]  # doctest: +SKIP
    """

    PHASES: ClassVar[tuple[str, ...]] = ("step", "fit", "eval", "plot", "other")
    """Canonical phase names recorded in ``timings.json``.

    Entering a phase not listed here issues a :class:`UserWarning` but is
    otherwise permitted; the accumulator is still populated and appears
    in :meth:`to_dict` under ``"<name>_s"``.
    """

    def __init__(self) -> None:
        """Create an empty run timer with all canonical phases zeroed."""
        # Seed with the canonical phases so to_dict() always has the full
        # schema even if a phase was never entered.
        self._elapsed: dict[str, float] = {name: 0.0 for name in self.PHASES}
        # Span bookkeeping for total(): first-entry and last-exit wall
        # times (perf_counter). ``None`` means "no phase ever entered".
        self._first_start: float | None = None
        self._last_end: float | None = None
        # Optional throughput counters.
        self._step_count: int | None = None
        self._update_count: int | None = None

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Accumulate wall-clock time into phase ``name``.

        The elapsed interval is added to the accumulator for ``name`` on
        exit (including when the body raises). Unknown phase names are
        accepted but emit a :class:`UserWarning` so typos are visible
        without breaking runs.

        Parameters
        ----------
        name:
            Phase label. Canonical values live in :attr:`PHASES`.

        Yields
        ------
        None
        """
        if not isinstance(name, str):
            raise TypeError(f"phase name must be str, got {type(name).__name__}")
        if name not in self.PHASES:
            warnings.warn(
                f"RunTimer.phase({name!r}) is not a canonical phase "
                f"(expected one of {self.PHASES}); recording anyway.",
                UserWarning,
                stacklevel=2,
            )
            # Ensure the key exists so elapsed()/to_dict() expose it.
            self._elapsed.setdefault(name, 0.0)

        start = time.perf_counter()
        if self._first_start is None:
            self._first_start = start
        try:
            yield
        finally:
            end = time.perf_counter()
            # ``end`` is monotonically non-decreasing across all phases
            # in a single process, so tracking the max is equivalent to
            # "last to exit".
            if self._last_end is None or end > self._last_end:
                self._last_end = end
            self._elapsed[name] = self._elapsed.get(name, 0.0) + (end - start)

    def elapsed(self, name: str) -> float:
        """Return total accumulated seconds for phase ``name``.

        Returns ``0.0`` if the phase was never entered (canonical phases
        are pre-seeded with zero at construction).
        """
        return float(self._elapsed.get(name, 0.0))

    def total(self) -> float:
        """Return wall-clock seconds spanning all observed phase activity.

        Measured from the first :meth:`phase` entry to the last
        :meth:`phase` exit. Returns ``0.0`` if no phase was ever entered.
        """
        if self._first_start is None or self._last_end is None:
            return 0.0
        return float(self._last_end - self._first_start)

    def set_step_count(self, n: int) -> None:
        """Record total environment steps taken during the run.

        Used by :meth:`to_dict` to compute ``steps_per_s = n / step_s``
        when ``step_s > 0``.

        Parameters
        ----------
        n:
            Non-negative integer step count.
        """
        if not isinstance(n, int):
            raise TypeError(f"step count must be int, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"step count must be non-negative, got {n}")
        self._step_count = n

    def set_update_count(self, n: int) -> None:
        """Record total agent updates (fits) performed during the run.

        Used by :meth:`to_dict` to compute ``updates_per_s = n / fit_s``
        when ``fit_s > 0``.

        Parameters
        ----------
        n:
            Non-negative integer update count.
        """
        if not isinstance(n, int):
            raise TypeError(f"update count must be int, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"update count must be non-negative, got {n}")
        self._update_count = n

    def to_dict(self) -> dict[str, float]:
        """Return a flat dict suitable for ``timings.json``.

        Keys
        ----
        ``total_s``
            Wall-clock span from first phase entry to last phase exit.
        ``step_s``, ``fit_s``, ``eval_s``, ``plot_s``, ``other_s``
            Accumulated seconds for each canonical phase. Any extra
            non-canonical phases introduced via :meth:`phase` are also
            included as ``"<name>_s"``.
        ``steps_per_s``
            ``step_count / step_s`` if both are available and
            ``step_s > 0``; otherwise ``None``.
        ``updates_per_s``
            ``update_count / fit_s`` if both are available and
            ``fit_s > 0``; otherwise ``None``.
        """
        out: dict[str, float] = {"total_s": self.total()}
        # Emit canonical phases first in a stable, spec-aligned order...
        for name in self.PHASES:
            out[f"{name}_s"] = float(self._elapsed.get(name, 0.0))
        # ...then any ad-hoc phases the caller may have introduced.
        for name, value in self._elapsed.items():
            if name in self.PHASES:
                continue
            out[f"{name}_s"] = float(value)

        step_s = out["step_s"]
        fit_s = out["fit_s"]
        out["steps_per_s"] = (
            float(self._step_count) / step_s
            if self._step_count is not None and step_s > 0.0
            else None
        )
        out["updates_per_s"] = (
            float(self._update_count) / fit_s
            if self._update_count is not None and fit_s > 0.0
            else None
        )
        return out

    def save(self, path: Path) -> None:
        """Write :meth:`to_dict` to ``path`` as JSON.

        Delegates to :func:`common.io.save_json`, which creates parent
        directories on demand.
        """
        save_json(path, self.to_dict())


# ----------------------------------------------------------------------------
# SweepTimer
# ----------------------------------------------------------------------------


class SweepTimer:
    """Per-sweep wall-clock timer for DP planners.

    Each call to :meth:`sweep` measures one Bellman sweep and appends the
    elapsed seconds to :attr:`sweep_times_s`; the running sum is kept in
    :attr:`cumulative_s`. Intended for value-iteration / policy-iteration
    runners where per-sweep timing drives convergence plots.

    Examples
    --------
    >>> st = SweepTimer()
    >>> for _ in range(3):
    ...     with st.sweep():
    ...         ...  # do_bellman_sweep()
    >>> len(st.sweep_times_s)
    3
    """

    def __init__(self) -> None:
        """Create an empty sweep timer."""
        self._sweep_times_s: list[float] = []
        self._cumulative_s: list[float] = []

    @contextmanager
    def sweep(self) -> Generator[None, None, None]:
        """Time one Bellman sweep.

        On exit, the elapsed interval is appended to
        :attr:`sweep_times_s`, and its running sum is appended to
        :attr:`cumulative_s`.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._sweep_times_s.append(float(elapsed))
            prev = self._cumulative_s[-1] if self._cumulative_s else 0.0
            self._cumulative_s.append(float(prev + elapsed))

    @property
    def sweep_times_s(self) -> list[float]:
        """List of per-sweep elapsed seconds, in sweep order."""
        # Return a shallow copy so external callers cannot mutate state.
        return list(self._sweep_times_s)

    @property
    def cumulative_s(self) -> list[float]:
        """List of cumulative elapsed seconds after each sweep, in order."""
        return list(self._cumulative_s)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary dict.

        Keys
        ----
        ``n_sweeps`` : int
            Number of completed sweeps.
        ``sweep_times_s`` : list[float]
            Per-sweep elapsed seconds.
        ``cumulative_s`` : list[float]
            Running sum of ``sweep_times_s``.
        ``mean_sweep_s`` : float
            Arithmetic mean of ``sweep_times_s``; ``0.0`` when empty.
        ``total_sweep_s`` : float
            Sum of all sweep times (``cumulative_s[-1]`` when non-empty).
        """
        n = len(self._sweep_times_s)
        total = self._cumulative_s[-1] if self._cumulative_s else 0.0
        mean = total / n if n > 0 else 0.0
        return {
            "n_sweeps": int(n),
            "sweep_times_s": list(self._sweep_times_s),
            "cumulative_s": list(self._cumulative_s),
            "mean_sweep_s": float(mean),
            "total_sweep_s": float(total),
        }
