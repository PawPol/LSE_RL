"""Run manifest handling for the weighted-LSE DP experimental program.

Responsibilities (spec §3.4, §4.1 and CLAUDE.md §4):

- Capture the current git SHA without ever raising.
- Deep-merge a base config with user/CLI overrides into a single resolved
  dict that is safe to persist.
- Write ``run.json`` (structured run header: config, env, seed, timing,
  git SHA) and ``metrics.json`` (scalar summary) into a run directory.
- Load those manifests back and enumerate existing run directories under
  the canonical ``results/weighted_lse_dp/<phase>/<suite>/<task>/<algorithm>/seed_*/``
  layout (see :mod:`experiments.weighted_lse_dp.common.io`).

Dependencies are stdlib only (``json``, ``pathlib``, ``socket``,
``subprocess``, ``datetime``, ``copy``) plus the sibling ``io`` module for
``save_json`` / ``load_json``.
"""

from __future__ import annotations

import copy
import pathlib
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The ``common/`` directory has no ``__init__.py`` (runs are invoked as
# scripts), so pull in the ``io`` helpers by inserting the repo root on
# ``sys.path``. The path is this file's great-great-grandparent:
#   experiments/weighted_lse_dp/common/manifests.py
#   parents[0] -> common/
#   parents[1] -> weighted_lse_dp/
#   parents[2] -> experiments/
#   parents[3] -> <repo-root>
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.io import (  # noqa: E402
    SCHEMA_VERSION,
    load_json,
    save_json,
)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "git_sha",
    "resolve_config",
    "write_run_json",
    "write_metrics_json",
    "load_run_json",
    "load_metrics_json",
    "find_run_dirs",
]


#: Schema version stamped on manifests written by this module. Kept in
#: lock-step with :data:`experiments.weighted_lse_dp.common.io.SCHEMA_VERSION`
#: so that aggregators can key off a single version string.
MANIFEST_SCHEMA_VERSION: str = SCHEMA_VERSION


# ----------------------------------------------------------------------------
# Timestamp helper (kept local to avoid importing io internals)
# ----------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as ISO-8601 with a trailing ``Z``."""
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if ts.endswith("+00:00"):
        ts = ts[: -len("+00:00")] + "Z"
    return ts


# ----------------------------------------------------------------------------
# Git SHA capture
# ----------------------------------------------------------------------------


def _find_repo_root(start: Path) -> Path | None:
    """Walk upward from ``start`` looking for a ``.git`` directory or file.

    Returns the containing directory (i.e. the repo root) or ``None`` if
    no ancestor contains ``.git``.
    """
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def git_sha(
    repo_root: Path | str | None = None,
    short: bool = False,
) -> str:
    """Return the current HEAD git SHA, or ``"unknown"`` on any failure.

    Parameters
    ----------
    repo_root:
        Optional explicit repository root. When ``None`` (default), the
        function walks upward from this module's own path to locate a
        ``.git`` directory. If no ancestor contains ``.git`` the function
        returns ``"unknown"``.
    short:
        If ``True``, return the first 8 characters of the SHA (or the
        literal ``"unknown"`` if capture failed). The default ``False``
        returns the full 40-character SHA.

    Notes
    -----
    This function never raises. Any failure (``git`` missing from PATH,
    repo in a detached or corrupt state, subprocess error, decode error)
    is swallowed and the literal ``"unknown"`` is returned. The caller is
    expected to persist the returned string verbatim.
    """
    try:
        if repo_root is None:
            root = _find_repo_root(Path(__file__).resolve().parent)
            if root is None:
                return "unknown"
        else:
            root = Path(repo_root).resolve()
            if not (root / ".git").exists():
                return "unknown"

        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            return "unknown"

        sha = (result.stdout or "").strip()
        if not sha or any(c not in "0123456789abcdefABCDEF" for c in sha):
            return "unknown"
        return sha[:8] if short else sha
    except Exception:
        # Broad catch per contract: never raise from git_sha.
        return "unknown"


# ----------------------------------------------------------------------------
# Config resolution
# ----------------------------------------------------------------------------


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Return a new dict: recursive merge of ``overrides`` into ``base``.

    Rules:
        - If both sides hold a dict at the same key, recurse.
        - Otherwise the ``overrides`` value wins and is deep-copied so
          the returned dict shares no mutable state with the inputs.
        - Keys present only in ``base`` are deep-copied into the result.
        - Keys present only in ``overrides`` are deep-copied into the
          result.
    """
    result: dict[Any, Any] = {}
    for key, value in base.items():
        if key in overrides:
            other = overrides[key]
            if isinstance(value, dict) and isinstance(other, dict):
                result[key] = _deep_merge(value, other)
            else:
                # Scalars and lists are replaced wholesale (spec contract).
                result[key] = copy.deepcopy(other)
        else:
            result[key] = copy.deepcopy(value)

    for key, value in overrides.items():
        if key not in base:
            result[key] = copy.deepcopy(value)

    return result


def resolve_config(
    base: dict,
    overrides: dict | None = None,
) -> dict:
    """Deep-merge ``overrides`` into ``base`` and return the merged config.

    The merge is recursive on nested dicts: a dict at key ``k`` on both
    sides is merged key-by-key. Scalars and lists are replaced wholesale
    (no list concatenation, no numeric addition) to keep the resolution
    rule simple and auditable.

    Parameters
    ----------
    base:
        Baseline configuration (typically loaded from a yaml file under
        ``experiments/configs/``). Not mutated.
    overrides:
        Optional mapping of overrides. May be ``None`` (treated as an
        empty dict). Not mutated.

    Returns
    -------
    dict
        A brand-new dict sharing no mutable state with either input.

    Raises
    ------
    TypeError
        If ``base`` is not a dict, or if ``overrides`` is not ``None`` /
        dict.
    """
    if not isinstance(base, dict):
        raise TypeError(f"base must be dict, got {type(base).__name__}")
    if overrides is None:
        return copy.deepcopy(base)
    if not isinstance(overrides, dict):
        raise TypeError(
            f"overrides must be dict or None, got {type(overrides).__name__}"
        )
    return _deep_merge(base, overrides)


# ----------------------------------------------------------------------------
# run.json / metrics.json writers
# ----------------------------------------------------------------------------


def write_run_json(
    run_dir: Path,
    *,
    config: dict,
    phase: str,
    task: str,
    algorithm: str,
    seed: int,
    git_sha_val: str | None = None,
    host: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write ``run_dir/run.json`` with the canonical run header.

    The emitted document has at minimum the keys::

        schema_version, created_at, phase, task, algorithm, seed,
        git_sha, host, config

    Additional keys from ``extra`` are merged in at the top level. If a
    key in ``extra`` collides with a reserved header key it overrides the
    automatically-computed value, which is occasionally useful in tests.

    Parameters
    ----------
    run_dir:
        Destination run directory (usually produced by
        :func:`experiments.weighted_lse_dp.common.io.make_run_dir`). Must
        already exist.
    config:
        Fully-resolved config dict (see :func:`resolve_config`). Stored
        verbatim under the ``"config"`` key.
    phase, task, algorithm, seed:
        Identify the originating run. Mirror the directory layout.
    git_sha_val:
        Explicit git SHA. When ``None`` (default) it is captured via
        :func:`git_sha`.
    host:
        Explicit hostname. When ``None`` (default) it is captured via
        :func:`socket.gethostname`; failures there fall back to
        ``"unknown"``.
    extra:
        Optional extra top-level fields (e.g. ``start_ts``, ``end_ts``,
        ``schedule_hash``). Must be a dict of JSON-serialisable values.

    Returns
    -------
    pathlib.Path
        The path that was written (``run_dir / "run.json"``).
    """
    run_dir = Path(run_dir)

    if git_sha_val is None:
        git_sha_val = git_sha()
    if host is None:
        try:
            host = socket.gethostname() or "unknown"
        except Exception:
            host = "unknown"

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "phase": str(phase),
        "task": str(task),
        "algorithm": str(algorithm),
        "seed": int(seed),
        "git_sha": str(git_sha_val),
        "host": str(host),
        "config": config,
    }

    if extra is not None:
        if not isinstance(extra, dict):
            raise TypeError(
                f"extra must be dict or None, got {type(extra).__name__}"
            )
        for key, value in extra.items():
            payload[key] = value

    out = run_dir / "run.json"
    save_json(out, payload)
    return out


def write_metrics_json(
    run_dir: Path,
    metrics: dict[str, float | int | str | None],
    *,
    phase: str,
    task: str,
    algorithm: str,
    seed: int,
) -> Path:
    """Write ``run_dir/metrics.json`` with the canonical scalar summary.

    The header fields ``schema_version``, ``created_at``, ``phase``,
    ``task``, ``algorithm``, ``seed`` are prepended to the document, and
    ``metrics`` is merged on top. Metric keys that collide with a header
    key override it (flagged by the caller, not silently coerced).

    Parameters
    ----------
    run_dir:
        Destination run directory. Must already exist.
    metrics:
        Flat mapping of metric name to scalar value. Values must be JSON
        serialisable (``float``, ``int``, ``str``, ``bool``, ``None``).
        Non-scalar payloads belong in ``curves.npz`` / ``transitions.npz``
        per spec §3.4.
    phase, task, algorithm, seed:
        Identify the originating run.

    Returns
    -------
    pathlib.Path
        The path that was written (``run_dir / "metrics.json"``).
    """
    if not isinstance(metrics, dict):
        raise TypeError(
            f"metrics must be dict, got {type(metrics).__name__}"
        )

    run_dir = Path(run_dir)

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "phase": str(phase),
        "task": str(task),
        "algorithm": str(algorithm),
        "seed": int(seed),
    }
    for key, value in metrics.items():
        payload[key] = value

    out = run_dir / "metrics.json"
    save_json(out, payload)
    return out


# ----------------------------------------------------------------------------
# Manifest loaders
# ----------------------------------------------------------------------------


def load_run_json(run_dir: Path) -> dict:
    """Load and return ``run_dir/run.json`` as a dict.

    Raises
    ------
    FileNotFoundError
        If ``run.json`` is absent from ``run_dir``.
    """
    run_dir = Path(run_dir)
    return load_json(run_dir / "run.json")


def load_metrics_json(run_dir: Path) -> dict:
    """Load and return ``run_dir/metrics.json`` as a dict.

    Raises
    ------
    FileNotFoundError
        If ``metrics.json`` is absent from ``run_dir``.
    """
    run_dir = Path(run_dir)
    return load_json(run_dir / "metrics.json")


# ----------------------------------------------------------------------------
# Run directory enumeration (for aggregators)
# ----------------------------------------------------------------------------


def find_run_dirs(
    results_root: Path,
    phase: str,
    suite: str,
    task: str | None = None,
    algorithm: str | None = None,
) -> list[Path]:
    """Enumerate existing seed directories under the canonical layout.

    Walks ``results_root/<phase>/<suite>/<task>/<algorithm>/seed_*/`` and
    returns every leaf that contains a ``run.json`` file. ``task`` and
    ``algorithm`` may be ``None`` to widen the glob (all tasks / all
    algorithms).

    Parameters
    ----------
    results_root:
        Root of the raw results tree. For production this is typically
        :data:`experiments.weighted_lse_dp.common.io.RESULT_ROOT`.
    phase:
        Phase directory name (e.g. ``"phase1"``).
    suite:
        Suite directory name (e.g. ``"paper_suite"`` or ``"smoke"``).
    task:
        Optional task filter. ``None`` matches all tasks.
    algorithm:
        Optional algorithm filter. ``None`` matches all algorithms.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of absolute seed directories containing a
        ``run.json``. Empty list if nothing matches.
    """
    results_root = Path(results_root)

    task_glob = task if task is not None else "*"
    algo_glob = algorithm if algorithm is not None else "*"

    pattern = f"{phase}/{suite}/{task_glob}/{algo_glob}/seed_*"
    matches = [
        p
        for p in results_root.glob(pattern)
        if p.is_dir() and (p / "run.json").is_file()
    ]
    return sorted(matches)
