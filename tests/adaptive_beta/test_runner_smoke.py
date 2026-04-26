"""End-to-end smoke test for the Phase VII runner CLI.

Spec authority: ``docs/specs/phase_VII_adaptive_beta.md`` §13.4
(reproducibility / artifact contract) and §15 (every runner emits
``run.json`` + ``metrics.npz`` + ``episodes.csv`` + ``transitions.parquet``).

This is the minimal evidence-of-life that ``python -m
experiments.adaptive_beta.run_experiment`` produces all the expected
artifacts in the right place when invoked with the M3.4 narrowing flags.

Failure mode this test guards against
-------------------------------------
A regression where the runner silently drops one of the four required
output files (e.g. ``transitions.parquet`` because the writer raised and
was caught), or where the CLI flags ``--out`` / ``--only-env`` /
``--only-method`` / ``--episodes`` stop being honoured. Either would
break the M3.4 reproducibility test below it; this smoke test is the
fast, isolated tripwire.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq
import pytest

from experiments.adaptive_beta.logging_callbacks import (
    EPISODE_COLUMNS,
    SCHEMA_VERSION_EPISODES,
    TRANSITION_COLUMNS,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEV_CONFIG = _REPO_ROOT / "experiments/adaptive_beta/configs/dev.yaml"
_RUNNER_MODULE = "experiments.adaptive_beta.run_experiment"


def _invoke_runner(
    out_dir: Path,
    *,
    seed: int = 0,
    env: str = "delayed_chain",
    method: str = "vanilla",
    episodes: int = 50,
) -> subprocess.CompletedProcess:
    """Invoke the runner via ``python -m`` and capture output for diagnostics."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            _RUNNER_MODULE,
            "--config",
            str(_DEV_CONFIG),
            "--seed",
            str(seed),
            "--out",
            str(out_dir),
            "--only-env",
            env,
            "--only-method",
            method,
            "--episodes",
            str(episodes),
        ],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def test_runner_smoke_produces_all_artifacts() -> None:
    """Spec §15: a 50-episode run emits all four artifacts loadable.

    Asserts:
    - exit code 0
    - ``<out>/dev/delayed_chain/vanilla/0/`` exists
    - ``metrics.npz`` loadable, has the spec-§7.4 numeric columns + a
      ``schema_version`` array
    - ``episodes.csv`` has exactly 51 lines (header + 50 rows)
    - ``transitions.parquet`` loadable with pyarrow
    - ``run.json`` parses as JSON with status == "completed"
    """
    n_episodes = 50
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        proc = _invoke_runner(out_dir, episodes=n_episodes)
        assert proc.returncode == 0, (
            f"runner exited with {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

        # --- Directory layout ---
        # Runner writes under <out>/<stage>/<env>/<method>/<seed>/.
        # dev.yaml has stage=dev.
        run_dir = out_dir / "dev" / "delayed_chain" / "vanilla" / "0"
        assert run_dir.is_dir(), (
            f"expected run directory {run_dir} not created;\n"
            f"out tree: {sorted(p.relative_to(out_dir) for p in out_dir.rglob('*'))}"
        )

        # --- run.json ---
        run_json_path = run_dir / "run.json"
        assert run_json_path.is_file()
        with open(run_json_path, "r", encoding="utf-8") as f:
            run_json = json.load(f)
        # We don't pin every key here (those are exercised by other tests);
        # we only require a parseable doc whose status reports completed.
        # Note: run_json itself does not carry a top-level ``status``
        # field — that lives in the manifest. Instead we assert the
        # ``completed_at`` timestamp is non-null, which is the in-record
        # signal the runner finished its episode loop.
        assert run_json.get("completed_at"), (
            f"run.json missing completed_at — runner did not finish: "
            f"{run_json}"
        )
        assert run_json.get("env") == "delayed_chain"
        assert run_json.get("method") == "vanilla"
        assert int(run_json.get("seed_id")) == 0
        assert int(run_json.get("n_episodes")) == n_episodes
        # Cross-check the runner reports completing as many episodes as
        # the override demanded.
        assert int(run_json.get("n_episodes_written")) == n_episodes

        # --- metrics.npz ---
        metrics_path = run_dir / "metrics.npz"
        assert metrics_path.is_file()
        with np.load(metrics_path, allow_pickle=True) as data:
            files = set(data.files)
            # Schema version array is always written.
            assert "schema_version" in files
            assert str(data["schema_version"]) == SCHEMA_VERSION_EPISODES
            # The runner strips the four string-valued columns
            # (run_id, env, method, phase) from metrics.npz; everything
            # else from EPISODE_COLUMNS must be present.
            stripped = {"run_id", "env", "method", "phase"}
            for col in EPISODE_COLUMNS:
                if col in stripped:
                    continue
                assert col in files, (
                    f"metrics.npz missing column {col!r}; "
                    f"present: {sorted(files)}"
                )
            # Every numeric/bool column has length n_episodes.
            assert data["return"].shape == (n_episodes,)

        # --- episodes.csv ---
        csv_path = run_dir / "episodes.csv"
        assert csv_path.is_file()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Exactly header + n_episodes rows.
        assert len(rows) == n_episodes + 1, (
            f"episodes.csv has {len(rows)} lines, expected {n_episodes + 1}"
        )
        assert tuple(rows[0]) == EPISODE_COLUMNS

        # --- transitions.parquet ---
        parquet_path = run_dir / "transitions.parquet"
        assert parquet_path.is_file()
        table = pq.read_table(parquet_path)
        # Schema columns must match exactly (order-insensitive, name-sensitive).
        assert set(table.schema.names) == set(TRANSITION_COLUMNS), (
            f"transitions.parquet schema mismatch: "
            f"got {sorted(table.schema.names)}, expected {sorted(TRANSITION_COLUMNS)}"
        )
        # We don't pin row count (depends on how many env steps fired
        # over 50 random-walk episodes), only that it's > 0.
        assert table.num_rows > 0, (
            "transitions.parquet has 0 rows; logger never recorded anything"
        )
