"""I/O helpers for Phase I runs.

Responsibilities:
    - Resolve and create canonical run directories under
      ``results/weighted_lse_dp/<phase>/<suite>/<task>/<algorithm>/seed_<seed>/``.
    - Build schema headers and attach them to ``.npz`` artifacts so every
      on-disk array file is self-describing.
    - Provide small JSON read/write helpers and an ``allow_pickle=False``
      ``.npz`` loader.
    - Provide a stdout-tee context manager for capturing ``stdout.log``.

All filesystem writes create parent directories on demand. This module is
stdlib + numpy only (no Hydra, no torch, no MushroomRL).
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from io import TextIOBase
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

RESULT_ROOT: Path = Path("results/weighted_lse_dp")
"""Canonical root for every Phase I/II/III artifact tree."""

SCHEMA_VERSION: str = "1.0.0"
"""Schema version embedded in every ``.npz`` header written via this module."""

_SCHEMA_KEY: str = "_schema"
"""Reserved array name inside ``.npz`` files that stores the JSON header."""


# ----------------------------------------------------------------------------
# Run directory resolution
# ----------------------------------------------------------------------------


def make_run_dir(
    base: str | Path,
    phase: str,
    suite: str,
    task: str,
    algorithm: str,
    seed: int,
    exist_ok: bool = False,
) -> Path:
    """Create and return the canonical run directory for a single seed.

    The returned path has the layout
    ``<base>/<phase>/<suite>/<task>/<algorithm>/seed_<seed>/``. Parent
    directories are created as needed.

    Parameters
    ----------
    base:
        Root under which the tree is created. Pass ``RESULT_ROOT`` for
        production runs or a tempdir for tests.
    phase:
        E.g. ``"phase1"``.
    suite:
        E.g. ``"smoke"`` or ``"paper_suite"``.
    task:
        E.g. ``"chain_base"``.
    algorithm:
        E.g. ``"QLearning"``.
    seed:
        Integer seed (becomes the directory suffix ``seed_<seed>``).
    exist_ok:
        If ``False`` (default), raise ``FileExistsError`` when the leaf
        directory already exists. If ``True``, silently reuse it.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    run_dir = (
        Path(base)
        / phase
        / suite
        / task
        / algorithm
        / f"seed_{int(seed)}"
    )
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    return run_dir


# ----------------------------------------------------------------------------
# Schema header helpers
# ----------------------------------------------------------------------------


def make_npz_schema(
    phase: str,
    task: str,
    algorithm: str,
    seed: int,
    storage_mode: str,
    arrays: list[str],
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return a schema header dict for a ``.npz`` artifact.

    The dict is JSON-serialisable and intended to be consumed by
    :func:`save_npz_with_schema`. ``created_at`` is ISO-8601 UTC with a
    trailing ``Z``.

    Parameters
    ----------
    phase, task, algorithm, seed:
        Identify the originating run. Redundant with the directory path
        but kept in the header so files remain self-describing after
        moves/copies.
    storage_mode:
        Free-form tag: ``"rl_online"`` for per-transition logs,
        ``"dp_stagewise"`` for sweeps, etc.
    arrays:
        Names of the non-schema arrays that will be written alongside
        the header.
    schema_version:
        Defaults to :data:`SCHEMA_VERSION`.
    """
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    # Normalise trailing "+00:00" to "Z" for compactness.
    if created_at.endswith("+00:00"):
        created_at = created_at[: -len("+00:00")] + "Z"

    return {
        "schema_version": str(schema_version),
        "created_at": created_at,
        "phase": str(phase),
        "task": str(task),
        "algorithm": str(algorithm),
        "seed": int(seed),
        "storage_mode": str(storage_mode),
        "arrays": list(arrays),
    }


def _encode_schema(schema: dict[str, Any]) -> np.ndarray:
    """Serialise schema to a uint8 numpy array of UTF-8 JSON bytes.

    Returned array is 1-D (shape: ``(len(json_bytes),)``). ``bytes(arr)``
    round-trips to the original JSON payload.
    """
    payload = json.dumps(schema, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    # shape: (n_bytes,), dtype uint8
    return np.frombuffer(payload, dtype=np.uint8).copy()


def save_npz_with_schema(
    path: str | Path,
    schema: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    """Save ``arrays`` to ``path`` with the header stored under ``_schema``.

    The header is encoded as a 1-D ``uint8`` array of UTF-8 JSON bytes so
    that ``bytes(loaded['_schema'])`` recovers the original JSON string.

    Parameters
    ----------
    path:
        Destination ``.npz`` path (extension added by ``np.savez`` if
        missing). Parent directories are created as needed.
    schema:
        Header dict from :func:`make_npz_schema`. Must not already contain
        the reserved ``_schema`` key in its data-arrays neighbours.
    arrays:
        Mapping of array name -> numpy array. Keys must not collide with
        ``_schema``; ideally they match ``schema["arrays"]``.
    """
    if _SCHEMA_KEY in arrays:
        raise ValueError(
            f"arrays must not contain the reserved key {_SCHEMA_KEY!r}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {_SCHEMA_KEY: _encode_schema(schema)}
    # shape-preserving copy into the savez payload
    for name, arr in arrays.items():
        payload[name] = np.asarray(arr)

    np.savez(path, **payload)


# ----------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` as indented (indent=2) JSON.

    Parent directories are created as needed. Uses ``sort_keys=False`` so
    caller-controlled ordering is preserved, and ``ensure_ascii=False`` so
    non-ASCII metadata survives round-trips.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON from ``path`` and return it as a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# NPZ load helper
# ----------------------------------------------------------------------------


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a ``.npz`` file into a plain dict (``allow_pickle=False``).

    The ``_schema`` entry, if present, is returned as a 1-D ``uint8``
    array; ``bytes(result["_schema"]).decode("utf-8")`` yields the JSON
    header, which can then be parsed with :func:`json.loads`.
    """
    path = Path(path)
    # NpzFile lazy-loads arrays; realise them all so the caller gets a
    # plain dict and the file handle can be closed.
    with np.load(path, allow_pickle=False) as npz:
        # shape-preserving copy: values inherit whatever dtype/shape was
        # written.
        return {name: np.asarray(npz[name]) for name in npz.files}


# ----------------------------------------------------------------------------
# Stdout tee
# ----------------------------------------------------------------------------


class _Tee(TextIOBase):
    """Minimal text stream that fans writes out to multiple sinks."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for stream in self._streams:
            n = stream.write(s)
        return n

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                # A downstream sink failing to flush must not kill the run.
                pass

    def isatty(self) -> bool:
        # Behave like a tty if any underlying stream does; helps tqdm etc.
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self._streams
        )


@contextmanager
def stdout_to_log(path: str | Path) -> Iterator[Path]:
    """Tee ``sys.stdout`` to ``path`` while also keeping terminal output.

    On entry, replaces ``sys.stdout`` with a tee that writes to both the
    original stream and an append-mode file at ``path``. On exit (or on
    exception), restores the original stdout and closes the file.

    Yields the ``Path`` being written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original = sys.stdout
    log_file = path.open("a", encoding="utf-8", buffering=1)  # line-buffered
    sys.stdout = _Tee(original, log_file)
    try:
        yield path
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = original
        log_file.close()


__all__ = [
    "RESULT_ROOT",
    "SCHEMA_VERSION",
    "make_run_dir",
    "make_npz_schema",
    "save_npz_with_schema",
    "save_json",
    "load_json",
    "load_npz",
    "stdout_to_log",
]
