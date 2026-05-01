"""Phase VIII tab-six-games run roster (spec §8.2).

This module provides :class:`Phase8RunRoster`, the manifest used by
Phase VIII's tab-six-games stages to track every requested
``(game, subcase, method, seed)`` cell from dispatch through
completion. The roster is the single source of truth for
"no silently dropped runs" (spec §15 acceptance criterion 7) and is
read by the verifier at every milestone gate.

Design choices
--------------

* **Serialization format: JSONL.**  Append-friendly, crash-safe, and
  trivially re-readable by ``pandas.read_json(path, lines=True)`` for
  ad-hoc analysis. Mirrors the temp-file-then-rename atomic write
  pattern used by
  ``experiments/weighted_lse_dp/common/manifests.py::write_run_json``.
  We re-implement the helper locally rather than importing it so the
  ``tab_six_games`` package stays self-contained and free of
  cross-experiment imports (per the boundary rules in CLAUDE.md §4).
* **Default ``base_path`` is ``results/adaptive_beta/tab_six_games``.**
  Phase VIII results must never land under
  ``results/weighted_lse_dp/`` (lessons.md, default-root drift). The
  result-root regression test (spec §13.6) asserts this.
* **Append-only with duplicate-cell detection.**  The roster keys each
  row by ``cell_id = f"{game}/{subcase}/{method}/{seed}"`` and refuses
  to append a duplicate (lessons.md, manifest pollution / no duplicate
  ``cell_id``).
* **Strict state-machine.**  Status transitions are validated; an
  invalid transition raises ``ValueError`` so silent corruption of
  run state is impossible.

Dependencies are stdlib only (``dataclasses``, ``json``, ``os``,
``pathlib``, ``typing``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional


__all__ = [
    "SCHEMA_VERSION",
    "VALID_STATUSES",
    "RosterRow",
    "Phase8RunRoster",
]


#: Schema version stamped on every roster snapshot. Bump on any
#: backwards-incompatible change to :class:`RosterRow` fields, the
#: status set, or the JSONL header.
SCHEMA_VERSION: str = "1.0.0"

#: All statuses recognised by :class:`Phase8RunRoster`. Kept as a
#: frozenset for O(1) membership checks. Defined as module-level so
#: that callers (verifier, runners, tests) can import it directly.
VALID_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "running",
        "completed",
        "failed",
        "diverged",
        "skipped",
        "stopped-by-gate",
    }
)


#: Allowed status-machine transitions, encoded as a mapping from
#: source status to the set of admissible target statuses. Terminal
#: statuses (``completed``, ``failed``, ``diverged``, ``skipped``,
#: ``stopped-by-gate``) have no outgoing transitions and therefore
#: do not appear as keys.
_VALID_TRANSITIONS: Dict[str, frozenset[str]] = {
    "pending": frozenset(
        {"running", "skipped", "stopped-by-gate"}
    ),
    "running": frozenset(
        {"completed", "failed", "diverged", "stopped-by-gate"}
    ),
}


_DEFAULT_BASE_PATH: Path = Path("results/adaptive_beta/tab_six_games")


@dataclass
class RosterRow:
    """One row of the Phase VIII run roster (spec §8.2).

    Attributes
    ----------
    run_id:
        Unique identifier for the run (typically a UUID or
        timestamped slug). Must be unique across the roster.
    config_hash:
        Hash of the resolved config dict that produced this run.
    seed:
        Integer seed used for environment + agent RNG.
    game:
        Game identifier (e.g. ``"matching_pennies"``).
    subcase:
        Subcase identifier within the game (e.g. ``"canonical"``).
    method:
        Method identifier (e.g. ``"vanilla"``,
        ``"fixed_beta_-1"``).
    status:
        Current status; must be one of :data:`VALID_STATUSES`.
    start_time:
        UTC ISO-8601 timestamp of when the run transitioned out of
        ``pending``. ``None`` while still pending / never started.
    end_time:
        UTC ISO-8601 timestamp of when the run reached a terminal
        status. ``None`` while running or pending.
    result_path:
        Path (string) to the run's result directory, set when the
        run starts. ``None`` for pending / skipped rows.
    failure_reason:
        Free-text reason populated on ``failed``, ``diverged``,
        ``skipped``, or ``stopped-by-gate`` transitions; ``None``
        otherwise.
    git_commit:
        Git SHA of the codebase at the time of dispatch.
    gamma:
        Discount factor γ used for this run. Recorded so v10 Tier II /
        Tier III γ-sweeps (spec §6.4 / §10.2.γ) keep distinct rows for
        the same (game, subcase, method, seed) at different γ.
        ``None`` is tolerated for backwards compatibility with rosters
        written before the v10 patch; new rows from the runner always
        populate it.
    """

    run_id: str
    config_hash: str
    seed: int
    game: str
    subcase: str
    method: str
    status: str
    git_commit: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    result_path: Optional[str] = None
    failure_reason: Optional[str] = None
    gamma: Optional[float] = None

    #: Field order used when serialising the JSONL header's
    #: ``_columns`` array. Iterating ``fields(self)`` would also
    #: work, but pinning the column order on the class makes the
    #: schema header deterministic across Python versions.
    #: ``gamma`` was appended in lockstep with the v10 patch (spec §6.4
    #: / §10.2.γ); the trailing position keeps the column order stable
    #: for pre-v10 readers.
    COLUMNS: ClassVar[List[str]] = [
        "run_id",
        "config_hash",
        "seed",
        "game",
        "subcase",
        "method",
        "status",
        "start_time",
        "end_time",
        "result_path",
        "failure_reason",
        "git_commit",
        "gamma",
    ]

    @property
    def cell_id(self) -> str:
        """Uniqueness key for the experimental cell.

        Pre-v10 form: ``"<game>/<subcase>/<method>/<seed>"`` (γ implicit).
        Post-v10 form when ``gamma`` is populated:
        ``"<game>/<subcase>/<method>/<seed>/gamma_<value>"`` so that the
        same (game, subcase, method, seed) at multiple γ values does
        NOT collide (spec §6.4 / §10.2.γ).

        Two rows with the same ``cell_id`` represent the same
        requested experimental cell and must not coexist in a
        roster.
        """
        base = f"{self.game}/{self.subcase}/{self.method}/{self.seed}"
        if self.gamma is None:
            return base
        return f"{base}/gamma_{float(self.gamma):.2f}"

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain JSON-serialisable dict for this row."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RosterRow":
        """Construct a :class:`RosterRow` from a JSONL payload dict.

        Unknown keys are ignored so that a future schema bump can
        add fields without breaking forward-readers; missing
        optional keys default to ``None``. Required keys (the
        non-``Optional`` fields of the dataclass) raise
        ``KeyError`` if absent.
        """
        known: Dict[str, Any] = {}
        field_names = {f.name for f in fields(cls)}
        for key, value in payload.items():
            if key in field_names:
                known[key] = value
        return cls(**known)


class Phase8RunRoster:
    """Append-only roster of Phase VIII tab-six-games runs (spec §8.2).

    The roster is the verifier's source of truth for run accounting.
    Construction defaults the root to
    ``results/adaptive_beta/tab_six_games`` to honour lessons.md's
    default-root drift rule; the result-root regression test
    (spec §13.6) asserts this default.

    Typical lifecycle:

    >>> roster = Phase8RunRoster()
    >>> roster.append(
    ...     run_id="run_0001",
    ...     config_hash="abc123",
    ...     seed=0,
    ...     game="matching_pennies",
    ...     subcase="canonical",
    ...     method="vanilla",
    ...     git_commit="deadbeef",
    ... )
    >>> roster.update_status(
    ...     "run_0001",
    ...     status="running",
    ...     start_time="2026-04-30T18:00:00Z",
    ...     result_path="results/adaptive_beta/tab_six_games/raw/...",
    ... )
    >>> roster.update_status(
    ...     "run_0001",
    ...     status="completed",
    ...     end_time="2026-04-30T18:05:00Z",
    ... )
    >>> roster.write_atomic(Path("/tmp/snap.jsonl"))
    >>> reread = Phase8RunRoster.read(Path("/tmp/snap.jsonl"))
    """

    def __init__(
        self,
        base_path: Path = _DEFAULT_BASE_PATH,
    ) -> None:
        """Construct an empty roster rooted at ``base_path``.

        The default ``base_path`` is
        ``results/adaptive_beta/tab_six_games`` — never
        ``results/weighted_lse_dp`` (lessons.md, default-root
        drift). The path is stored verbatim; no filesystem I/O
        happens at construction time.
        """
        self.base_path: Path = Path(base_path)
        self._rows: List[RosterRow] = []
        # Indices for O(1) lookups; kept in sync with ``_rows``.
        self._by_run_id: Dict[str, int] = {}
        self._by_cell_id: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def rows(self) -> List[RosterRow]:
        """Return a shallow copy of the roster's rows.

        Returning a copy prevents callers from mutating the
        internal list and bypassing the duplicate-cell / state-
        machine invariants.
        """
        return list(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterable[RosterRow]:
        return iter(self._rows)

    def get(self, run_id: str) -> RosterRow:
        """Return the row with ``run_id``; raise ``KeyError`` if missing."""
        idx = self._by_run_id.get(run_id)
        if idx is None:
            raise KeyError(f"run_id not in roster: {run_id!r}")
        return self._rows[idx]

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        run_id: str,
        config_hash: str,
        seed: int,
        game: str,
        subcase: str,
        method: str,
        git_commit: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        result_path: Optional[str] = None,
        failure_reason: Optional[str] = None,
        status: str = "pending",
        gamma: Optional[float] = None,
    ) -> None:
        """Append a new row to the roster.

        Defaults to ``status="pending"`` and disallows reuse of an
        existing ``run_id`` or ``cell_id``. Duplicate ``cell_id``
        raises ``ValueError`` (lessons.md, manifest pollution / no
        duplicate ``cell_id``); duplicate ``run_id`` likewise.
        Unknown statuses raise ``ValueError``.

        ``gamma`` is optional; when provided it disambiguates rows in a
        v10 γ-sweep (spec §6.4 / §10.2.γ). Pre-v10 callers continue to
        pass no ``gamma``, in which case the row's ``cell_id`` matches
        the pre-v10 four-tuple form.
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"unknown status {status!r}; "
                f"must be one of {sorted(VALID_STATUSES)}"
            )

        if run_id in self._by_run_id:
            raise ValueError(
                f"duplicate run_id in roster: {run_id!r}"
            )

        row = RosterRow(
            run_id=run_id,
            config_hash=config_hash,
            seed=int(seed),
            game=game,
            subcase=subcase,
            method=method,
            status=status,
            git_commit=git_commit,
            start_time=start_time,
            end_time=end_time,
            result_path=result_path,
            failure_reason=failure_reason,
            gamma=None if gamma is None else float(gamma),
        )

        if row.cell_id in self._by_cell_id:
            existing = self._rows[self._by_cell_id[row.cell_id]]
            raise ValueError(
                f"duplicate cell_id in roster: {row.cell_id!r} "
                f"(already held by run_id={existing.run_id!r})"
            )

        idx = len(self._rows)
        self._rows.append(row)
        self._by_run_id[run_id] = idx
        self._by_cell_id[row.cell_id] = idx

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def update_status(
        self,
        run_id: str,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Transition ``run_id`` to ``status`` and patch optional fields.

        Validates the transition against
        :data:`_VALID_TRANSITIONS`. Allowed kwargs: ``start_time``,
        ``end_time``, ``result_path``, ``failure_reason``. Any
        other kwarg raises ``TypeError``. Unknown ``status`` or
        invalid transition raises ``ValueError``.
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"unknown status {status!r}; "
                f"must be one of {sorted(VALID_STATUSES)}"
            )

        idx = self._by_run_id.get(run_id)
        if idx is None:
            raise KeyError(f"run_id not in roster: {run_id!r}")

        row = self._rows[idx]
        current = row.status
        allowed = _VALID_TRANSITIONS.get(current, frozenset())
        if status not in allowed:
            raise ValueError(
                f"invalid status transition for run_id={run_id!r}: "
                f"{current!r} -> {status!r} "
                f"(allowed targets: {sorted(allowed)})"
            )

        # Validate kwargs early so partial mutations are impossible.
        allowed_keys = {
            "start_time",
            "end_time",
            "result_path",
            "failure_reason",
        }
        unknown = set(kwargs).difference(allowed_keys)
        if unknown:
            raise TypeError(
                f"unexpected keyword arguments to update_status: "
                f"{sorted(unknown)}"
            )

        row.status = status
        for key, value in kwargs.items():
            setattr(row, key, value)

    # ------------------------------------------------------------------
    # Atomic JSONL serialisation
    # ------------------------------------------------------------------

    def _schema_header(self) -> Dict[str, Any]:
        """Build the first-line header object for the JSONL snapshot."""
        return {
            "_schema": "Phase8RunRoster",
            "_version": SCHEMA_VERSION,
            "_columns": list(RosterRow.COLUMNS),
        }

    def write_atomic(self, path: Path) -> None:
        """Serialise the roster to ``path`` as JSONL atomically.

        Writes to ``<path>.tmp``, ``fsync``s it, then ``os.replace``
        to ``path``. The first line is the schema header
        (``{"_schema": "Phase8RunRoster", ...}``); subsequent
        lines are JSON objects, one per row, in append order.

        Parent directories are created if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_name(path.name + ".tmp")
        header = self._schema_header()

        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(header, sort_keys=True))
            fh.write("\n")
            for row in self._rows:
                fh.write(json.dumps(row.to_dict(), sort_keys=True))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_path, path)

    @classmethod
    def read(cls, path: Path) -> "Phase8RunRoster":
        """Read a JSONL snapshot back into a populated roster.

        Validates the schema header (``_schema`` ==
        ``"Phase8RunRoster"``); raises ``ValueError`` on mismatch
        or unsupported version. Empty / blank lines are skipped.
        Duplicate detection is re-run on read so a corrupt file
        cannot smuggle in conflicting cells.
        """
        path = Path(path)
        roster = cls()
        # Wipe the default to indicate "loaded from disk" — the base
        # path on a re-read does not know where the snapshot came
        # from. Callers wanting to keep the canonical default should
        # construct fresh; this is a deliberate, conservative choice.
        # We retain the default in case the caller relies on it.

        with open(path, "r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]

        if not lines:
            raise ValueError(f"empty roster file: {path}")

        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"failed to parse schema header in {path}: {exc}"
            ) from exc

        if (
            not isinstance(header, dict)
            or header.get("_schema") != "Phase8RunRoster"
        ):
            raise ValueError(
                f"missing or wrong schema header in {path}: "
                f"{header!r}"
            )

        version = header.get("_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Phase8RunRoster schema version "
                f"{version!r} in {path}; expected {SCHEMA_VERSION!r}"
            )

        for raw in lines[1:]:
            payload = json.loads(raw)
            row = RosterRow.from_dict(payload)
            # Use append() so duplicate / status validation rules
            # apply to anything we read off disk. We pass the row's
            # current status straight through; default-pending check
            # is bypassed because append() accepts an explicit status.
            roster.append(
                run_id=row.run_id,
                config_hash=row.config_hash,
                seed=row.seed,
                game=row.game,
                subcase=row.subcase,
                method=row.method,
                git_commit=row.git_commit,
                start_time=row.start_time,
                end_time=row.end_time,
                result_path=row.result_path,
                failure_reason=row.failure_reason,
                status=row.status,
                gamma=row.gamma,
            )

        return roster

    # ------------------------------------------------------------------
    # Disk reconciliation
    # ------------------------------------------------------------------

    def reconcile_with_disk(self, raw_root: Path) -> Dict[str, Any]:
        """Cross-check the roster against ``run.json`` files on disk.

        Walks ``raw_root`` for any ``run.json`` files. For each
        roster row whose ``result_path`` is set:

        * if the path is present on disk, no change;
        * if the path is absent, the row is transitioned to
          ``failed`` with ``failure_reason="result path absent"``
          (only when the current status permits the transition;
          otherwise the row is recorded under ``"unreconcilable"``
          in the report and left untouched).

        For each ``run.json`` discovered on disk that does not
        match any row's ``result_path`` (matched by run-directory),
        the run-directory is collected under ``"orphan_run_dirs"``.

        Returns
        -------
        dict
            Summary with keys:

            * ``"checked_rows"`` — total roster rows considered.
            * ``"missing_results"`` — roster ``run_id`` list that
              were marked failed because the result path is gone.
            * ``"unreconcilable"`` — roster ``run_id`` list whose
              result path is missing but whose status forbids the
              transition to ``failed``.
            * ``"orphan_run_dirs"`` — list of run-directory paths
              (as strings) that hold a ``run.json`` but are not
              referenced by any roster row.
            * ``"raw_root"`` — string of the inspected root.
        """
        raw_root = Path(raw_root)

        # Map known result-path-as-directory -> run_id.
        known_run_dirs: Dict[Path, str] = {}
        for row in self._rows:
            if row.result_path is None:
                continue
            try:
                rp = Path(row.result_path).resolve(strict=False)
            except OSError:
                rp = Path(row.result_path)
            known_run_dirs[rp] = row.run_id

        missing_results: List[str] = []
        unreconcilable: List[str] = []
        for row in self._rows:
            if row.result_path is None:
                continue
            rp = Path(row.result_path)
            # Directory presence is the contract; ``run.json``
            # presence is a stronger check we do not require here
            # (a partial run dir without ``run.json`` is still a
            # legitimate disk artifact for an in-flight run).
            if rp.exists():
                continue
            try:
                self.update_status(
                    row.run_id,
                    status="failed",
                    failure_reason="result path absent",
                )
                missing_results.append(row.run_id)
            except ValueError:
                unreconcilable.append(row.run_id)

        # Discover orphans.
        orphan_run_dirs: List[str] = []
        if raw_root.exists():
            for candidate in raw_root.rglob("run.json"):
                if not candidate.is_file():
                    continue
                run_dir = candidate.parent
                resolved = run_dir.resolve(strict=False)
                if resolved in known_run_dirs:
                    continue
                # Also accept an unresolved match (covers cases
                # where ``result_path`` is stored as a relative
                # path that does not resolve under raw_root).
                if run_dir in known_run_dirs:
                    continue
                orphan_run_dirs.append(str(run_dir))

        return {
            "raw_root": str(raw_root),
            "checked_rows": len(self._rows),
            "missing_results": missing_results,
            "unreconcilable": unreconcilable,
            "orphan_run_dirs": sorted(orphan_run_dirs),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self) -> Dict[str, Any]:
        """Return aggregate counts useful for verifier dashboards.

        Keys:

        * ``"total"`` — total row count.
        * ``"by_status"`` — ``{status: count}`` over all rows.
        * ``"by_method"`` — ``{method: count}``.
        * ``"by_game_subcase"`` — ``{f"{game}/{subcase}": count}``.
        """
        by_status: Dict[str, int] = {}
        by_method: Dict[str, int] = {}
        by_game_subcase: Dict[str, int] = {}

        for row in self._rows:
            by_status[row.status] = by_status.get(row.status, 0) + 1
            by_method[row.method] = by_method.get(row.method, 0) + 1
            key = f"{row.game}/{row.subcase}"
            by_game_subcase[key] = by_game_subcase.get(key, 0) + 1

        return {
            "total": len(self._rows),
            "by_status": by_status,
            "by_method": by_method,
            "by_game_subcase": by_game_subcase,
        }
