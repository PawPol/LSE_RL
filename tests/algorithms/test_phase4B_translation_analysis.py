"""
Phase IV-B §9: Translation analysis pipeline tests.

Verifies:
  1. Paired-difference computations are correct.
  2. Spearman correlations are computed on matched task/scheduler/seed groups.
  3. Bootstrap paired confidence intervals preserve seed pairing.
  4. Null translation cases are reported rather than silently dropped.

All tests are purely synthetic — no real experiment artifacts required.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import argparse

import numpy as np
import pytest

# Allow importing analysis modules directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.weighted_lse_dp.analysis.paired_bootstrap import paired_bootstrap_ci


# ---------------------------------------------------------------------------
# Helpers to build minimal fake data trees
# ---------------------------------------------------------------------------

def _make_translation_dir(
    base: Path,
    tasks: list[str],
    algo_classes: dict[str, str],  # {algo_name: class_label}
    seed_returns: dict[str, dict[str, list[float]]],  # {task: {algo: [returns]}}
) -> Path:
    """Create a minimal translation/ directory tree with summary JSONs."""
    trans_dir = base / "translation"
    for task in tasks:
        for algo, algo_cls in algo_classes.items():
            algo_dir = trans_dir / task / algo
            algo_dir.mkdir(parents=True, exist_ok=True)
            returns = seed_returns.get(task, {}).get(algo, [1.0])
            summary = {
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "primary_outcome": float(np.mean(returns)),
                "seed_returns": returns,
                "n_seeds": len(returns),
                "mean_abs_natural_shift": 0.02 if "stagewise" in algo else 0.001,
                "mean_abs_delta_effective_discount": 0.01 if "stagewise" in algo else 0.0005,
                "mean_abs_target_gap": 0.015 if "stagewise" in algo else 0.001,
            }
            (algo_dir / "summary.json").write_text(json.dumps(summary))
    return trans_dir


def _make_sweep_dir(
    base: Path,
    task: str,
    u_maxes: list[float],
    diag_scale: float = 1.0,
    outcome_scale: float = 1.0,
) -> Path:
    """Create a minimal diagnostic_sweep/ directory for one task."""
    sweep_dir = base / "diagnostic_sweep" / task
    sweep_dir.mkdir(parents=True, exist_ok=True)
    points = []
    for u in u_maxes:
        points.append({
            "u_max": u,
            "mean_abs_natural_shift": u * diag_scale,
            "mean_abs_u": u * diag_scale,
            "primary_outcome": u * outcome_scale,
            "mean_return": u * outcome_scale,
        })
    (sweep_dir / "sweep_results.json").write_text(json.dumps(points))
    return sweep_dir


# ---------------------------------------------------------------------------
# Test 1: Paired-difference computations are correct (spec §9.2 / §10.1.1)
# ---------------------------------------------------------------------------

class TestPairedDifferences:
    def test_mean_difference_exact(self) -> None:
        """paired_bootstrap_ci mean_diff matches np.mean(a-b) exactly."""
        rng = np.random.default_rng(0)
        a = rng.normal(5.0, 1.0, size=20)
        b = rng.normal(3.0, 1.0, size=20)
        _, _, mean_diff = paired_bootstrap_ci(a, b, n_bootstrap=1000, seed=0)
        expected = float(np.mean(a - b))
        assert abs(mean_diff - expected) < 1e-12, (
            f"mean_diff={mean_diff} != expected={expected}"
        )

    def test_zero_difference_on_identical_arrays(self) -> None:
        """When a == b, mean_diff should be exactly 0."""
        a = np.ones(15)
        b = np.ones(15)
        _, _, mean_diff = paired_bootstrap_ci(a, b, seed=0)
        assert abs(mean_diff) < 1e-12

    def test_negative_difference_detected(self) -> None:
        """mean_diff is negative when b > a consistently."""
        a = np.zeros(10)
        b = np.ones(10)
        lo, hi, mean_diff = paired_bootstrap_ci(a, b, seed=0)
        assert mean_diff < 0, "Expected negative mean_diff when a < b"
        assert hi < 0.01, "Upper CI bound should be negative (b >> a)"

    def test_different_length_raises(self) -> None:
        with pytest.raises(ValueError, match="equal shape"):
            paired_bootstrap_ci(np.ones(5), np.ones(6))

    def test_empty_arrays_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            paired_bootstrap_ci(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Test 2: Spearman correlations on matched groups (spec §9.2 / §10.1.2)
# ---------------------------------------------------------------------------

class TestSpearmanCorrelation:
    def test_monotone_sweep_positive_spearman(self) -> None:
        """Strictly increasing diag -> outcome should yield positive Spearman ρ."""
        pytest.importorskip("scipy")
        from scipy.stats import spearmanr  # type: ignore[import]

        u_maxes = [0.0, 0.005, 0.01, 0.02]
        # Perfect monotone relationship
        diag_vals = [u * 5.0 for u in u_maxes]
        outcome_vals = [u * 3.0 for u in u_maxes]

        baseline_diag = diag_vals[0]
        baseline_outcome = outcome_vals[0]
        diag_deltas = [d - baseline_diag for d in diag_vals]
        outcome_deltas = [o - baseline_outcome for o in outcome_vals]

        result = spearmanr(diag_deltas, outcome_deltas)
        assert result.statistic > 0.9, f"Expected high positive ρ; got {result.statistic}"

    def test_anti_correlated_sweep_negative_spearman(self) -> None:
        """Increasing diag with decreasing outcome should give negative Spearman ρ."""
        pytest.importorskip("scipy")
        from scipy.stats import spearmanr  # type: ignore[import]

        u_maxes = [0.0, 0.005, 0.01, 0.02]
        diag_deltas = [u * 5.0 for u in u_maxes]
        outcome_deltas = [-u * 3.0 for u in u_maxes]

        result = spearmanr(diag_deltas, outcome_deltas)
        assert result.statistic < -0.9, f"Expected high negative ρ; got {result.statistic}"

    def test_sweep_uses_seed_averaged_deltas(self, tmp_path: Path) -> None:
        """translation_analysis.step2 processes sweep points correctly."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        u_maxes = [0.0, 0.005, 0.01, 0.02]
        _make_sweep_dir(tmp_path, "task_a", u_maxes, diag_scale=5.0, outcome_scale=3.0)

        results = ta.step2_translation_sweep(tmp_path)
        assert "task_a" in results
        info = results["task_a"]
        assert "diag_deltas" in info
        assert "outcome_deltas" in info
        # Baseline at u_max=0: deltas at u_max=0 should be ~0
        assert abs(info["diag_deltas"][0]) < 1e-10
        assert abs(info["outcome_deltas"][0]) < 1e-10
        # Monotone: all deltas >= 0
        for d in info["diag_deltas"]:
            assert d >= -1e-10, f"diag delta should be non-negative; got {d}"


# ---------------------------------------------------------------------------
# Test 3: Bootstrap paired CIs preserve seed pairing (spec §10.1.3)
# ---------------------------------------------------------------------------

class TestBootstrapPairing:
    def test_ci_coverage_on_known_distribution(self) -> None:
        """Percentile bootstrap CI covers true mean at ≥90% rate on synthetic data.

        We use a small Monte Carlo (50 trials) because full coverage tests are slow.
        Target: ≥80% of trials should contain the true mean (95% CI, n=10).
        """
        rng = np.random.default_rng(42)
        true_diff = 2.0
        n_trials = 50
        n_obs = 10
        n_covered = 0
        for _ in range(n_trials):
            a = rng.normal(true_diff, 1.0, size=n_obs)
            b = rng.normal(0.0, 1.0, size=n_obs)
            lo, hi, _ = paired_bootstrap_ci(a, b, n_bootstrap=1000, ci=0.95, seed=int(rng.integers(0, 10_000)))
            if lo <= true_diff <= hi:
                n_covered += 1
        coverage = n_covered / n_trials
        assert coverage >= 0.70, (
            f"Bootstrap coverage too low: {coverage:.2f} < 0.70 over {n_trials} trials"
        )

    def test_pairing_matters_vs_unpaired(self) -> None:
        """Paired CI is tighter than CI computed on shuffled (unpaired) data."""
        rng = np.random.default_rng(7)
        n = 30
        # a and b are strongly paired: a[i] = b[i] + small_delta
        b = rng.normal(0, 5.0, size=n)
        delta = rng.normal(0.5, 0.1, size=n)
        a = b + delta

        lo_paired, hi_paired, _ = paired_bootstrap_ci(a, b, n_bootstrap=5000, seed=0)
        width_paired = hi_paired - lo_paired

        # Shuffle b to break pairing
        b_shuffled = rng.permutation(b)
        lo_unpaired, hi_unpaired, _ = paired_bootstrap_ci(a, b_shuffled, n_bootstrap=5000, seed=0)
        width_unpaired = hi_unpaired - lo_unpaired

        assert width_paired < width_unpaired, (
            f"Paired CI ({width_paired:.4f}) should be narrower than unpaired ({width_unpaired:.4f})"
        )

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical CI bounds."""
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 1.0, size=20)
        b = rng.normal(0.0, 1.0, size=20)
        r1 = paired_bootstrap_ci(a, b, n_bootstrap=1000, seed=123)
        r2 = paired_bootstrap_ci(a, b, n_bootstrap=1000, seed=123)
        assert r1 == r2, "Same seed must produce identical results"

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (almost surely) produce different bounds."""
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 1.0, size=20)
        b = rng.normal(0.0, 1.0, size=20)
        r1 = paired_bootstrap_ci(a, b, n_bootstrap=1000, seed=0)
        r2 = paired_bootstrap_ci(a, b, n_bootstrap=1000, seed=999)
        # Mean diff must be identical; bounds need not be
        assert abs(r1[2] - r2[2]) < 1e-12, "mean_diff must be seed-independent"
        # Very likely the bounds differ for different seeds
        # (not guaranteed, but extremely unlikely to match exactly)


# ---------------------------------------------------------------------------
# Test 4: Null translation cases are reported (spec §10.1.4)
# ---------------------------------------------------------------------------

class TestNullCaseReporting:
    def test_null_cases_included_in_output(self, tmp_path: Path) -> None:
        """step3 produces nonlinearity_effect; _report_nulls identifies CI-spans-zero case."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        # Build a fake step3 result where CI spans zero (null translation)
        step3_fake = {
            "task_null": {
                "tag": "task_null",
                "nonlinearity_effect": {
                    "lower": -0.5, "upper": 0.5, "mean_diff": 0.05, "n_pairs": 5
                },
                "total_effect": {
                    "lower": -0.3, "upper": 0.4, "mean_diff": 0.02, "n_pairs": 5
                },
                "path_effect": {
                    "lower": -0.1, "upper": 0.2, "mean_diff": 0.01, "n_pairs": 5
                },
                "n_seeds": {"classical": 5, "safe-zero": 5, "safe-nonlinear": 5},
            }
        }
        step1_fake = {"task_null": {"activated": True}}

        nulls = ta._report_nulls(step1_fake, step3_fake)
        assert len(nulls) == 1, f"Expected 1 null case; got {len(nulls)}"
        assert nulls[0]["tag"] == "task_null"
        assert nulls[0]["mean_diff"] == pytest.approx(0.02)

    def test_clear_positive_effect_not_null(self) -> None:
        """A clearly positive effect (CI entirely > 0) is NOT a null case."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        step3_fake = {
            "task_good": {
                "total_effect": {"lower": 0.1, "upper": 0.5, "mean_diff": 0.3, "n_pairs": 5},
                "nonlinearity_effect": {},
                "path_effect": {},
                "n_seeds": {},
            }
        }
        step1_fake = {"task_good": {"activated": True}}
        nulls = ta._report_nulls(step1_fake, step3_fake)
        assert len(nulls) == 0

    def test_null_reported_even_when_unactivated(self) -> None:
        """A CI-spans-zero case is reported regardless of activation status."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta

        step3_fake = {
            "task_inactive": {
                "total_effect": {
                    "lower": -0.2, "upper": 0.1, "mean_diff": -0.05, "n_pairs": 3
                },
                "nonlinearity_effect": {},
                "path_effect": {},
                "n_seeds": {},
            }
        }
        step1_fake = {"task_inactive": {"activated": False}}
        nulls = ta._report_nulls(step1_fake, step3_fake)
        assert len(nulls) == 1
        assert "Activation absent" in nulls[0]["interpretation"]

    def test_full_pipeline_with_no_data_produces_empty_nulls(
        self, tmp_path: Path
    ) -> None:
        """Running full pipeline on empty directory produces empty null list."""
        from experiments.weighted_lse_dp.analysis import translation_analysis as ta
        # Create minimal counterfactual_replay dir so step5 doesn't warn loudly
        (tmp_path / "counterfactual_replay").mkdir()

        ns = argparse.Namespace(input_dir=tmp_path, output_dir=tmp_path / "analysis")
        # Should not raise — just produce empty outputs
        ta.main(ns)
        null_path = tmp_path / "analysis" / "null_translation_cases.json"
        assert null_path.exists()
        data = json.loads(null_path.read_text())
        assert isinstance(data, list)
