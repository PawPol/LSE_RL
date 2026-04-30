"""Phase VIII result-root regression tests (spec §13.6).

Lessons.md default-root drift rule: Phase VIII artifact producers must
route through ``results/adaptive_beta/tab_six_games`` explicitly and must
not inherit ``experiments.weighted_lse_dp.common.io.RESULT_ROOT``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.adaptive_beta.tab_six_games.manifests import Phase8RunRoster
from experiments.weighted_lse_dp.common import io as common_io


PHASE8_ROOT = Path("results/adaptive_beta/tab_six_games")
WEIGHTED_LSE_ROOT = Path("results/weighted_lse_dp")


def _is_under(path: Path, root: Path) -> bool:
    try:
        Path(path).relative_to(root)
    except ValueError:
        return False
    return True


def _disable_real_mkdir(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    mkdir_calls: list[Path] = []

    def fake_mkdir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        mkdir_calls.append(Path(self))

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    return mkdir_calls


def test_phase8_run_roster_default_base() -> None:
    """Default roster construction must point at the Phase VIII result root."""
    roster = Phase8RunRoster()

    base_path = Path(roster.base_path)
    assert base_path == PHASE8_ROOT
    assert not _is_under(base_path, WEIGHTED_LSE_ROOT)


def test_make_run_dir_explicit_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit Phase VIII base yields the expected canonical run path."""
    mkdir_calls = _disable_real_mkdir(monkeypatch)

    run_dir = common_io.make_run_dir(
        base=PHASE8_ROOT,
        phase="VIII",
        suite="stage1",
        task="shapley/SH-FictitiousPlay",
        algorithm="adaptive_beta",
        seed=0,
    )

    assert _is_under(run_dir, PHASE8_ROOT)
    assert not _is_under(run_dir, WEIGHTED_LSE_ROOT)
    assert run_dir == (
        PHASE8_ROOT
        / "VIII"
        / "stage1"
        / "shapley"
        / "SH-FictitiousPlay"
        / "adaptive_beta"
        / "seed_0"
    )
    assert mkdir_calls == [run_dir]


def test_make_run_dir_default_base_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared helper's default root remains weighted_lse_dp.

    Current ``make_run_dir`` requires ``base`` explicitly. If a future
    version adds a default, this test exercises that path; otherwise it
    falls back to the module-level ``RESULT_ROOT`` to document why Phase
    VIII callers must override the root.
    """
    _disable_real_mkdir(monkeypatch)
    kwargs = {
        "phase": "VIII",
        "suite": "stage1",
        "task": "shapley/SH-FictitiousPlay",
        "algorithm": "adaptive_beta",
        "seed": 0,
    }

    try:
        run_dir = common_io.make_run_dir(**kwargs)
    except TypeError as exc:
        if "base" not in str(exc):
            raise
        run_dir = common_io.make_run_dir(base=common_io.RESULT_ROOT, **kwargs)

    assert common_io.RESULT_ROOT == WEIGHTED_LSE_ROOT
    assert _is_under(run_dir, WEIGHTED_LSE_ROOT)
    assert not _is_under(run_dir, PHASE8_ROOT)


@pytest.mark.skip(
    reason="Phase VIII runners not yet implemented; will be enabled in M6",
)
def test_phase_VIII_runner_uses_explicit_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future M6 runner test: all run dirs pass the Phase VIII base."""
    from experiments.adaptive_beta.tab_six_games import runners

    make_run_dir_calls = []

    def fake_make_run_dir(*args, **kwargs):
        make_run_dir_calls.append({"args": args, "kwargs": kwargs})
        return (
            PHASE8_ROOT
            / "VIII"
            / "stage1"
            / "placeholder"
            / "adaptive_beta"
            / "seed_0"
        )

    monkeypatch.setattr(runners, "make_run_dir", fake_make_run_dir)
    runners.run_stage1_smoke(seed=0, dry_run=True)

    assert make_run_dir_calls
    assert all(
        Path(call["kwargs"]["base"]) == PHASE8_ROOT
        for call in make_run_dir_calls
    )


def test_phase8_roster_write_base_under_phase_viii(tmp_path: Path) -> None:
    """``write_atomic`` writes exactly where the caller asks it to write."""
    roster = Phase8RunRoster()
    out_path = tmp_path / "caller-controlled" / "manifest.jsonl"

    roster.write_atomic(out_path)

    assert out_path.exists()
    assert out_path.parent == tmp_path / "caller-controlled"
    assert PHASE8_ROOT.as_posix() not in out_path.as_posix()
    assert out_path.read_text(encoding="utf-8").splitlines()[0].startswith(
        '{"_columns":'
    )
