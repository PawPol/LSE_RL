"""Tests for Phase VIII tab-six-games run roster manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.adaptive_beta.tab_six_games import manifests
from experiments.adaptive_beta.tab_six_games.manifests import (
    SCHEMA_VERSION,
    Phase8RunRoster,
)


def _append_row(
    roster: Phase8RunRoster,
    *,
    run_id: str = "run_0",
    seed: int = 0,
    game: str = "matching_pennies",
    subcase: str = "canonical",
    method: str = "vanilla",
    status: str = "pending",
    result_path: str | None = None,
) -> None:
    roster.append(
        run_id=run_id,
        config_hash=f"cfg_{run_id}",
        seed=seed,
        game=game,
        subcase=subcase,
        method=method,
        git_commit="deadbeef",
        status=status,
        result_path=result_path,
    )


def _single_row_roster(run_id: str = "run_0") -> Phase8RunRoster:
    roster = Phase8RunRoster(base_path=Path("/tmp/unused"))
    _append_row(roster, run_id=run_id)
    return roster


def test_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "roster.jsonl"

    old_roster = _single_row_roster("old_run")
    old_roster.write_atomic(dest)
    before = dest.read_text(encoding="utf-8")

    new_roster = _single_row_roster("new_run")

    def raise_during_replace(src: str | Path, dst: str | Path) -> None:
        raise RuntimeError("simulated interruption before atomic replace")

    monkeypatch.setattr(manifests.os, "replace", raise_during_replace)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        new_roster.write_atomic(dest)

    assert dest.read_text(encoding="utf-8") == before
    assert Phase8RunRoster.read(dest).summarize() == old_roster.summarize()


def test_status_enum_valid_transitions() -> None:
    running_terminals = ("completed", "failed", "diverged", "stopped-by-gate")
    for terminal_status in running_terminals:
        roster = _single_row_roster(f"run_to_{terminal_status}")
        roster.update_status("run_to_" + terminal_status, "running")
        roster.update_status("run_to_" + terminal_status, terminal_status)
        assert roster.get("run_to_" + terminal_status).status == terminal_status

    for terminal_status in ("skipped", "stopped-by-gate"):
        roster = _single_row_roster(f"pending_to_{terminal_status}")
        roster.update_status("pending_to_" + terminal_status, terminal_status)
        assert roster.get("pending_to_" + terminal_status).status == terminal_status


def test_status_enum_invalid_transitions() -> None:
    running_roster = _single_row_roster("running_run")
    running_roster.update_status("running_run", "running")
    with pytest.raises(ValueError, match="running.*pending"):
        running_roster.update_status("running_run", "pending")

    completed_roster = _single_row_roster("completed_run")
    completed_roster.update_status("completed_run", "running")
    completed_roster.update_status("completed_run", "completed")
    with pytest.raises(ValueError, match="completed.*running"):
        completed_roster.update_status("completed_run", "running")

    pending_roster = _single_row_roster("pending_run")
    with pytest.raises(ValueError, match="pending.*completed"):
        pending_roster.update_status("pending_run", "completed")


def test_append_only_no_duplicate_cell_id() -> None:
    roster = Phase8RunRoster(base_path=Path("/tmp/unused"))
    _append_row(
        roster,
        run_id="run_a",
        game="shapley",
        subcase="baseline",
        method="adaptive_beta",
        seed=7,
    )

    with pytest.raises(ValueError, match="duplicate cell_id"):
        _append_row(
            roster,
            run_id="run_b",
            game="shapley",
            subcase="baseline",
            method="adaptive_beta",
            seed=7,
        )


def test_reconcile_with_disk(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    roster = Phase8RunRoster(base_path=tmp_path)
    run_json_paths: list[Path] = []

    for idx in range(3):
        run_dir = raw_root / f"run_{idx}"
        run_dir.mkdir(parents=True)
        run_json = run_dir / "run.json"
        run_json.write_text(json.dumps({"run_id": f"run_{idx}"}), encoding="utf-8")
        run_json_paths.append(run_json)
        _append_row(
            roster,
            run_id=f"run_{idx}",
            seed=idx,
            status="running",
            result_path=str(run_json),
        )

    run_json_paths[1].unlink()

    report = roster.reconcile_with_disk(raw_root)

    assert report["checked_rows"] == 3
    assert report["missing_results"] == ["run_1"]
    assert report["unreconcilable"] == []
    assert roster.get("run_1").status == "failed"
    assert roster.get("run_1").failure_reason == "result path absent"
    assert roster.get("run_0").status == "running"
    assert roster.get("run_2").status == "running"


def test_summarize_counts() -> None:
    roster = Phase8RunRoster(base_path=Path("/tmp/unused"))
    statuses = [
        "pending",
        "running",
        "running",
        "completed",
        "failed",
        "diverged",
        "skipped",
        "stopped-by-gate",
    ]
    for idx, status in enumerate(statuses):
        _append_row(
            roster,
            run_id=f"run_{idx}",
            seed=idx,
            method="vanilla" if idx % 2 == 0 else "adaptive_beta",
            status=status,
        )

    summary = roster.summarize()

    assert summary["total"] == 8
    assert summary["by_status"] == {
        "pending": 1,
        "running": 2,
        "completed": 1,
        "failed": 1,
        "diverged": 1,
        "skipped": 1,
        "stopped-by-gate": 1,
    }


def test_atomic_write_durability(tmp_path: Path) -> None:
    path = tmp_path / "roster.jsonl"
    roster = _single_row_roster("durable_run")
    roster.write_atomic(path)
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty roster file|schema header"):
        Phase8RunRoster.read(path)


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "roster.jsonl"
    round_tripped_path = tmp_path / "round_tripped.jsonl"
    roster = Phase8RunRoster(base_path=tmp_path)
    _append_row(roster, run_id="run_pending", seed=0)
    _append_row(roster, run_id="run_completed", seed=1)
    roster.update_status(
        "run_completed",
        "running",
        start_time="2026-04-30T20:00:00Z",
        result_path=str(tmp_path / "raw" / "run_completed"),
    )
    roster.update_status(
        "run_completed",
        "completed",
        end_time="2026-04-30T20:01:00Z",
    )

    roster.write_atomic(path)
    loaded = Phase8RunRoster.read(path)
    loaded.write_atomic(round_tripped_path)

    assert loaded.summarize() == roster.summarize()
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == json.loads(
        round_tripped_path.read_text(encoding="utf-8").splitlines()[0]
    )


def test_default_base_path() -> None:
    roster = Phase8RunRoster()

    default_path = roster.base_path.as_posix()
    assert default_path.endswith("results/adaptive_beta/tab_six_games")
    assert "results/weighted_lse_dp" not in default_path


def test_schema_header(tmp_path: Path) -> None:
    path = tmp_path / "roster.jsonl"
    roster = _single_row_roster("schema_run")
    roster.write_atomic(path)

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    header = json.loads(first_line)

    assert header["_schema"] == "Phase8RunRoster"
    assert header["_version"] == SCHEMA_VERSION
    assert header["_columns"]
