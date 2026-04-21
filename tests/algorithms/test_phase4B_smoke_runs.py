"""
Phase IV-B §10.4: End-to-end smoke runs for translation experiments.

Verifies:
  1. translation_analysis.main() runs on a minimal synthetic directory.
  2. make_phase4B_figures.main() runs without error on synthetic data.
  3. make_phase4B_tables.main() runs without error on synthetic data.
  4. Diagnostic-strength sweep parsing works end-to-end.
  5. Matched controls emit analysis JSON files.

All tests use purely synthetic data in tmp_path — no real Phase IV-B
experiment artifacts are required.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Fixtures: synthetic Phase IV-B data tree
# ---------------------------------------------------------------------------

def _build_synthetic_phase4b(base: Path) -> None:
    """Build a minimal but structurally correct Phase IV-B results tree."""
    # ---- translation/<task>/<algo>/summary.json ----
    tasks = ["chain_sparse_credit_0", "chain_catastrophe_1"]
    algos = {
        "classical_vi": "classical",
        "safe_vi_zero": "safe-zero",
        "safe_vi_stagewise": "safe-nonlinear",
    }
    for task in tasks:
        for algo, cls in algos.items():
            d = base / "translation" / task / algo
            d.mkdir(parents=True, exist_ok=True)
            is_nl = cls == "safe-nonlinear"
            summary = {
                "mean_return": 0.7 if is_nl else 0.5,
                "std_return": 0.1,
                "primary_outcome": 0.7 if is_nl else 0.5,
                "seed_returns": [0.6, 0.7, 0.8] if is_nl else [0.4, 0.5, 0.6],
                "n_seeds": 3,
                "mean_abs_natural_shift": 0.015 if is_nl else 0.001,
                "mean_abs_delta_effective_discount": 0.008 if is_nl else 0.0002,
                "mean_abs_target_gap": 0.012 if is_nl else 0.001,
                "per_stage": [
                    {"stage": s,
                     "mean_abs_natural_shift": 0.01 * (s + 1) if is_nl else 0.001,
                     "mean_abs_delta_effective_discount": 0.005 * (s + 1) if is_nl else 0.0001}
                    for s in range(4)
                ],
            }
            (d / "summary.json").write_text(json.dumps(summary))

    # ---- diagnostic_sweep/<task>/sweep_results.json ----
    for task in tasks:
        sweep_dir = base / "diagnostic_sweep" / task
        sweep_dir.mkdir(parents=True, exist_ok=True)
        u_maxes = [0.0, 0.005, 0.01, 0.02]
        points = [
            {
                "u_max": u,
                "mean_abs_natural_shift": u * 3.0,
                "mean_abs_u": u * 3.0,
                "primary_outcome": 0.5 + u * 10.0,
                "mean_return": 0.5 + u * 10.0,
            }
            for u in u_maxes
        ]
        (sweep_dir / "sweep_results.json").write_text(json.dumps(points))

    # ---- counterfactual_replay/<task>/replay_summary.json (negative control) ----
    for fam in ["chain_sparse_credit", "grid_hazard"]:
        for idx in range(2):
            tag = f"{fam}_{idx}"
            d = base / "counterfactual_replay" / tag
            d.mkdir(parents=True, exist_ok=True)
            summary = {
                "family": fam,
                "tag": tag,
                "n_transitions": 5000,
                "gamma_base": 0.95,
                "r_max": 1.0,
                "seed": 42,
                # Near-zero activation (negative control)
                "mean_abs_u": 0.001,
                "frac_u_ge_5e3": 0.02,
                "mean_abs_delta_d": 0.0005,
                "mean_natural_shift": 0.0005,
                "mean_abs_target_gap": 0.0008,
            }
            (d / "replay_summary.json").write_text(json.dumps(summary))

    # ---- activation_report/ (for event-conditioned figure fallback) ----
    act_dir = base / "activation_report"
    act_dir.mkdir(exist_ok=True)
    event_diags = [
        {"tag": t, "family": t.rsplit("_", 1)[0],
         "mean_abs_u_global": 0.005, "mean_abs_u_event": 0.012}
        for t in tasks
    ]
    (act_dir / "event_conditioned_diagnostics.json").write_text(
        json.dumps(event_diags)
    )


# ---------------------------------------------------------------------------
# Test 1: translation_analysis.main() smoke run
# ---------------------------------------------------------------------------

class TestTranslationAnalysisSmoke:
    def test_main_runs_on_synthetic_tree(self, tmp_path: Path) -> None:
        """translation_analysis.main() completes and writes expected JSON files."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        _build_synthetic_phase4b(tmp_path)
        ns = argparse.Namespace(
            input_dir=tmp_path,
            output_dir=tmp_path / "analysis",
        )
        ta.main(ns)  # Must not raise

        analysis_dir = tmp_path / "analysis"
        expected_files = [
            "step1_activation_verification.json",
            "step2_translation_sweep.json",
            "step3_matched_control.json",
            "step4_outcome_interpretation.json",
            "step5_negative_control.json",
            "null_translation_cases.json",
            "translation_analysis_summary.json",
        ]
        for fname in expected_files:
            fpath = analysis_dir / fname
            assert fpath.exists(), f"Expected output file not found: {fname}"
            data = json.loads(fpath.read_text())
            assert data is not None

    def test_step1_detects_activation(self, tmp_path: Path) -> None:
        """Step 1 flags safe-nonlinear as having higher diagnostics than safe-zero."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        _build_synthetic_phase4b(tmp_path)
        results = ta.step1_activation_verification(tmp_path)
        for tag in results:
            if results[tag].get("activated") is not None:
                assert results[tag]["activated"] is True, (
                    f"Task {tag} should be activated in synthetic data"
                )

    def test_step5_negative_control_near_zero(self, tmp_path: Path) -> None:
        """Step 5 confirms negative-control tasks are below gate threshold."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        _build_synthetic_phase4b(tmp_path)
        result = ta.step5_negative_control(tmp_path)
        assert result.get("all_near_zero") is True, (
            f"All synthetic negative-control tasks should be near-zero; "
            f"large_diff_tasks = {result.get('large_diff_tasks')}"
        )

    def test_default_output_dir(self, tmp_path: Path) -> None:
        """When --output-dir is None, output goes to <input-dir>/analysis/."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        _build_synthetic_phase4b(tmp_path)
        ns = argparse.Namespace(input_dir=tmp_path, output_dir=None)
        ta.main(ns)
        assert (tmp_path / "analysis" / "translation_analysis_summary.json").exists()


# ---------------------------------------------------------------------------
# Test 2: make_phase4B_figures.main() smoke run
# ---------------------------------------------------------------------------

class TestFiguresSmoke:
    def test_figures_main_runs(self, tmp_path: Path) -> None:
        """make_phase4B_figures.main() completes without error on synthetic data."""
        from experiments.weighted_lse_dp.analysis import make_phase4B_figures as figs

        _build_synthetic_phase4b(tmp_path)
        # Pre-run translation_analysis so step-JSON files are present
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta
        ta.main(argparse.Namespace(input_dir=tmp_path, output_dir=tmp_path / "analysis"))

        ns = argparse.Namespace(
            results_dir=tmp_path,
            output_dir=tmp_path / "analysis",
        )
        figs.main(ns)  # Must not raise

    def test_figures_default_output_dir(self, tmp_path: Path) -> None:
        """When --output-dir is None, figures go to <results-dir>/analysis/."""
        from experiments.weighted_lse_dp.analysis import make_phase4B_figures as figs

        _build_synthetic_phase4b(tmp_path)
        ns = argparse.Namespace(results_dir=tmp_path, output_dir=None)
        figs.main(ns)
        # At least the directory should have been created
        assert (tmp_path / "analysis").is_dir()


# ---------------------------------------------------------------------------
# Test 3: make_phase4B_tables.main() smoke run
# ---------------------------------------------------------------------------

class TestTablesSmoke:
    def test_tables_main_runs(self, tmp_path: Path) -> None:
        """make_phase4B_tables.main() completes and writes CSV files."""
        from experiments.weighted_lse_dp.analysis import make_phase4B_tables as tbls

        _build_synthetic_phase4b(tmp_path)
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta
        ta.main(argparse.Namespace(input_dir=tmp_path, output_dir=tmp_path / "analysis"))

        ns = argparse.Namespace(
            results_dir=tmp_path,
            output_dir=tmp_path / "analysis",
        )
        tbls.main(ns)

        # P4B_A should exist and be non-empty (we have actual data)
        csv_path = tmp_path / "analysis" / "P4B_A.csv"
        assert csv_path.exists(), "P4B_A.csv not created"
        content = csv_path.read_text()
        assert len(content.splitlines()) > 1, "P4B_A.csv is empty"

    def test_tables_contain_required_names(self, tmp_path: Path) -> None:
        """All six required tables P4B_A through P4B_F are created."""
        from experiments.weighted_lse_dp.analysis import make_phase4B_tables as tbls

        _build_synthetic_phase4b(tmp_path)
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta
        ta.main(argparse.Namespace(input_dir=tmp_path, output_dir=tmp_path / "analysis"))

        ns = argparse.Namespace(results_dir=tmp_path, output_dir=tmp_path / "analysis")
        tbls.main(ns)

        for name in ["P4B_A", "P4B_B", "P4B_C", "P4B_D", "P4B_E", "P4B_F"]:
            assert (tmp_path / "analysis" / f"{name}.csv").exists(), (
                f"Table {name}.csv not created"
            )
            assert (tmp_path / "analysis" / f"{name}.md").exists(), (
                f"Table {name}.md not created"
            )

    def test_tables_default_output_dir(self, tmp_path: Path) -> None:
        """When --output-dir is None, tables go to <results-dir>/analysis/."""
        from experiments.weighted_lse_dp.analysis import make_phase4B_tables as tbls

        _build_synthetic_phase4b(tmp_path)
        ns = argparse.Namespace(results_dir=tmp_path, output_dir=None)
        tbls.main(ns)
        assert (tmp_path / "analysis").is_dir()


# ---------------------------------------------------------------------------
# Test 4: Diagnostic sweep smoke run
# ---------------------------------------------------------------------------

class TestDiagnosticSweepSmoke:
    def test_sweep_parsed_correctly(self, tmp_path: Path) -> None:
        """Sweep JSON is parsed; deltas computed relative to u_max=0 baseline."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        _build_synthetic_phase4b(tmp_path)
        results = ta.step2_translation_sweep(tmp_path)

        for tag in ["chain_sparse_credit_0", "chain_catastrophe_1"]:
            assert tag in results, f"Task {tag} missing from sweep results"
            info = results[tag]
            assert info.get("status") != "missing_sweep_file", (
                f"Sweep file should exist for {tag}"
            )
            assert "diag_deltas" in info
            assert "outcome_deltas" in info
            # Baseline (u_max=0) delta should be zero
            assert abs(info["diag_deltas"][0]) < 1e-10
            assert abs(info["outcome_deltas"][0]) < 1e-10
            # Deltas should be non-decreasing (monotone synthetic data)
            for i in range(1, len(info["diag_deltas"])):
                assert info["diag_deltas"][i] >= info["diag_deltas"][i - 1] - 1e-10

    def test_missing_sweep_file_skipped_with_warning(
        self, tmp_path: Path
    ) -> None:
        """Missing sweep_results.json is skipped gracefully, not raised."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        # Create sweep dir without sweep_results.json
        (tmp_path / "diagnostic_sweep" / "phantom_task").mkdir(parents=True)

        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = ta.step2_translation_sweep(tmp_path)

        assert "phantom_task" in results
        assert results["phantom_task"].get("status") == "missing_sweep_file"
        warning_msgs = [str(w.message) for w in caught]
        assert any("sweep_results.json" in m for m in warning_msgs), (
            "Expected warning about missing sweep file"
        )


# ---------------------------------------------------------------------------
# Test 5: Matched controls emitted and aggregated
# ---------------------------------------------------------------------------

class TestMatchedControlsAggregated:
    def test_step3_produces_three_way_comparison(self, tmp_path: Path) -> None:
        """step3 produces nonlinearity, total, and path effects for each task."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        _build_synthetic_phase4b(tmp_path)
        results = ta.step3_matched_control(tmp_path)

        for tag in ["chain_sparse_credit_0", "chain_catastrophe_1"]:
            assert tag in results, f"Task {tag} missing from step3 results"
            info = results[tag]
            assert "nonlinearity_effect" in info
            assert "total_effect" in info
            assert "path_effect" in info

            # With 3 seeds per class, all effects should have data
            for key in ["nonlinearity_effect", "total_effect", "path_effect"]:
                eff = info[key]
                if eff.get("n_pairs", 0) >= 2:
                    assert eff["mean_diff"] is not None
                    assert eff["lower"] is not None
                    assert eff["upper"] is not None

    def test_safe_nonlinear_has_higher_mean_return(self, tmp_path: Path) -> None:
        """In synthetic data, safe-nonlinear should dominate classical."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        _build_synthetic_phase4b(tmp_path)
        results = ta.step3_matched_control(tmp_path)

        for tag, info in results.items():
            total = info.get("total_effect", {})
            if total.get("mean_diff") is not None:
                # Synthetic: safe-NL mean=0.7, classical mean=0.5 -> diff=+0.2
                assert total["mean_diff"] > 0, (
                    f"Task {tag}: expected positive total effect in synthetic data; "
                    f"got {total['mean_diff']}"
                )
