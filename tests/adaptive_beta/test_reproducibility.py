"""Byte-identical reproducibility of ``metrics.npz`` across two runs.

Spec authority: ``docs/specs/phase_VII_adaptive_beta.md`` §13.4.

Run ``dev.yaml`` twice with ``--seed 0`` (different temp output
directories) and assert ``metrics.npz`` is byte-equivalent across the
two runs. We narrow to one ``(env, method)`` slice for runtime
(``delayed_chain × adaptive_beta`` — a canonical-+ env that exercises
the full schedule path including the smoothing/clipping update).

Failure mode this test guards against
-------------------------------------
A regression where the runner introduces hidden nondeterminism — e.g.
seeds the agent RNG from ``time.time()``, iterates a Python ``set``
unordered, or shells out to a non-pinned subprocess. Any of those would
shift one or more numeric columns and the byte-equality assertion would
fail.

Note: ``run.json`` is *not* checked for byte-equality because by design
it carries timestamps, hostname, and a wall-clock-derived ``run_id``;
the spec contract is on ``metrics.npz`` (and by extension the array
columns the analysis pipeline consumes).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEV_CONFIG = _REPO_ROOT / "experiments/adaptive_beta/configs/dev.yaml"
_RUNNER_MODULE = "experiments.adaptive_beta.run_experiment"

# 100 episodes is enough to: (1) drive the schedule's
# update_after_episode loop a meaningful number of times so smoothing
# and clipping both fire, and (2) accumulate enough Q-table mutations
# that any RNG drift would produce a byte-level diff in metrics.npz.
# Larger budgets here only cost time without strengthening the
# invariant.
_REPRO_EPISODES: int = 100


def _run_one(out_dir: Path, *, seed: int = 0) -> subprocess.CompletedProcess:
    """Drive the runner once, narrowed to the canonical reproducibility slice."""
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
            # Spec §13.4: pick a canonical-+ env that triggers the
            # adaptive schedule end-to-end.
            "--only-env",
            "delayed_chain",
            "--only-method",
            "adaptive_beta",
            "--episodes",
            str(_REPRO_EPISODES),
        ],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def test_dev_yaml_seed_zero_reproducible() -> None:
    """Spec §13.4: same seed twice => byte-identical ``metrics.npz``.

    Asserts every column is array-equal (bit-exact, not within tol).
    Any drift triggers a regression alert with the column name.
    """
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        proc1 = _run_one(Path(t1))
        assert proc1.returncode == 0, (
            f"first run failed: stdout={proc1.stdout}\nstderr={proc1.stderr}"
        )
        proc2 = _run_one(Path(t2))
        assert proc2.returncode == 0, (
            f"second run failed: stdout={proc2.stdout}\nstderr={proc2.stderr}"
        )

        # Path layout: <out>/<stage>/<env>/<method>/<seed>/metrics.npz
        # (dev.yaml's ``stage: dev``).
        rel = Path("dev") / "delayed_chain" / "adaptive_beta" / "0" / "metrics.npz"
        a_path = Path(t1) / rel
        b_path = Path(t2) / rel
        assert a_path.is_file(), f"first run did not write {a_path}"
        assert b_path.is_file(), f"second run did not write {b_path}"

        with np.load(a_path, allow_pickle=True) as a, np.load(
            b_path, allow_pickle=True
        ) as b:
            keys_a = set(a.files)
            keys_b = set(b.files)
            assert keys_a == keys_b, (
                f"metrics.npz key sets differ: only_in_a="
                f"{sorted(keys_a - keys_b)}, only_in_b="
                f"{sorted(keys_b - keys_a)}"
            )
            # Sort to make the iteration order deterministic for the
            # error path; np.testing.assert_array_equal is bit-exact for
            # numeric arrays.
            for key in sorted(keys_a):
                np.testing.assert_array_equal(
                    a[key],
                    b[key],
                    err_msg=(
                        f"metrics.npz column {key!r} differs across two "
                        f"identical-seed runs (spec §13.4 reproducibility "
                        f"violated)"
                    ),
                )
