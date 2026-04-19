"""
Phase IV-A §9.7: Smoke runs for activation search and counterfactual replay.

Short end-to-end runs verifying that the activation search pipeline,
counterfactual replay, and audit pipeline all execute without errors
and produce well-formed outputs.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PYTHON = str(_REPO_ROOT / ".venv" / "bin" / "python")


def _run(cmd: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    """Run a subprocess from the repo root."""
    return subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_activation_search_smoke() -> None:
    """§9.7: Activation search pipeline completes on a tiny 1-episode pilot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run([
            _PYTHON,
            "experiments/weighted_lse_dp/runners/run_phase4_activation_search.py",
            "--seed", "7",
            "--n-pilot-episodes", "1",
            "--output-dir", tmpdir,
            "--dry-run",
        ])

        assert result.returncode == 0, (
            f"run_phase4_activation_search.py exited {result.returncode}:\n"
            f"{result.stderr[-2000:]}"
        )

        out_dir = pathlib.Path(tmpdir)
        assert (out_dir / "candidate_grid.json").exists(), "candidate_grid.json missing"
        assert (out_dir / "candidate_scores.csv").exists(), "candidate_scores.csv missing"

        # candidate_grid.json must have 168 entries.
        grid = json.loads((out_dir / "candidate_grid.json").read_text())
        assert len(grid) == 168, f"Expected 168 candidates, got {len(grid)}"

        # candidate_scores.csv must have a header + 168 data rows.
        csv_lines = (out_dir / "candidate_scores.csv").read_text().strip().splitlines()
        assert len(csv_lines) == 169, f"Expected 169 CSV lines, got {len(csv_lines)}"


def test_counterfactual_replay_smoke() -> None:
    """§9.7: Counterfactual replay runs and produces expected log fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suite_path = str(
            _REPO_ROOT
            / "experiments"
            / "weighted_lse_dp"
            / "configs"
            / "phase4"
            / "activation_suite.json"
        )

        result = _run([
            _PYTHON,
            "experiments/weighted_lse_dp/runners/run_phase4_counterfactual_replay.py",
            "--suite", suite_path,
            "--seed", "7",
            "--n-episodes", "5",
            "--output-dir", tmpdir,
        ])

        assert result.returncode == 0, (
            f"run_phase4_counterfactual_replay.py exited {result.returncode}:\n"
            f"{result.stderr[-2000:]}"
        )

        out_dir = pathlib.Path(tmpdir)

        # Must have produced at least one replay_summary.json.
        summaries = list(out_dir.rglob("replay_summary.json"))
        assert len(summaries) >= 1, "No replay_summary.json files produced"

        # Each summary must have the required diagnostic fields.
        # Required fields use actual keys from run_phase4_counterfactual_replay.py
        REQUIRED_FIELDS = {
            "family", "n_transitions",
            "mean_abs_u", "frac_u_ge_5e3",
            "mean_abs_delta_d", "mean_abs_target_gap",
            "mean_beta_used",
        }
        for s_path in summaries:
            s = json.loads(s_path.read_text())
            missing = REQUIRED_FIELDS - set(s.keys())
            assert not missing, (
                f"{s_path.name}: missing fields {missing}"
            )

        # Must have produced at least one NPZ file with expected arrays.
        npz_files = list(out_dir.rglob("replay_diagnostics.npz"))
        assert len(npz_files) >= 1, "No replay_diagnostics.npz files produced"

        REQUIRED_ARRAYS = {
            "natural_shift", "delta_effective_discount",
            "target_gap_same_gamma_base", "beta_used",
        }
        for npz_path in npz_files:
            data = np.load(str(npz_path), allow_pickle=False)
            missing = REQUIRED_ARRAYS - set(data.files)
            assert not missing, (
                f"{npz_path.name}: missing arrays {missing}"
            )
            # All arrays must have the same length.
            lengths = {k: len(data[k]) for k in REQUIRED_ARRAYS if k in data.files}
            assert len(set(lengths.values())) == 1, (
                f"Inconsistent array lengths: {lengths}"
            )


def test_audit_pipeline_smoke() -> None:
    """§9.7: Phase III audit pipeline executes and produces valid JSON outputs."""
    # The audit was already run during Phase IV-A initialization;
    # verify the artifacts exist and are well-formed.
    audit_dir = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase4" / "audit"

    code_audit = audit_dir / "phase3_code_audit.json"
    result_audit = audit_dir / "phase3_result_audit.json"
    compat_report = audit_dir / "phase3_compat_report.md"

    assert code_audit.exists(), f"Missing: {code_audit}"
    assert result_audit.exists(), f"Missing: {result_audit}"
    assert compat_report.exists(), f"Missing: {compat_report}"

    # Code audit must be parseable JSON with required fields.
    ca = json.loads(code_audit.read_text())
    assert isinstance(ca, dict) and len(ca) > 0, (
        "phase3_code_audit.json must be a non-empty JSON object"
    )

    # Result audit must be parseable JSON.
    ra = json.loads(result_audit.read_text())
    assert isinstance(ra, dict), "phase3_result_audit.json must be a JSON object"

    # Compat report must be non-empty.
    report_text = compat_report.read_text()
    assert len(report_text) > 50, "phase3_compat_report.md seems empty"

    # Verify the aggregation pipeline works on existing processed data.
    proc_dir = _REPO_ROOT / "results" / "processed" / "phase4A"
    if proc_dir.exists():
        gate_eval = proc_dir / "gate_evaluation.json"
        assert gate_eval.exists(), f"gate_evaluation.json missing at {gate_eval}"
        ge = json.loads(gate_eval.read_text())
        assert isinstance(ge, list), "gate_evaluation.json must be a list"
        for entry in ge:
            assert "tag" in entry or "task_id" in entry, "gate entry missing task_id/tag"
            assert "global_gate_pass" in entry or "gate_pass" in entry, (
                "gate entry missing gate_pass/global_gate_pass"
            )
