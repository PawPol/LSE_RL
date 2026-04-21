#!/usr/bin/env python
"""Phase IV-B: translation analysis pipeline.

Five-step mandatory analysis pipeline (spec §9):
  1. Activation verification: confirm operator is active per task family.
  2. Within-family translation sweep: dose-response (outcome delta vs
     diagnostic delta) and Spearman correlations.
  3. Matched-control mechanism isolation: classical vs safe-zero vs
     safe-nonlinear comparison.
  4. Outcome-specific interpretation: task-type primary metric.
  5. Negative-control consistency: Phase III replay suite near-zero check.

CLI
---
  python translation_analysis.py --input-dir <dir> [--output-dir <dir>]

  --input-dir   Top-level results directory containing:
                  translation/<task_tag>/          (matched-control runs)
                  diagnostic_sweep/<task_tag>/      (sweep runs)
                  counterfactual_replay/            (negative control)
                  activation_report/                (from Phase IV-A aggregation)
  --output-dir  Destination for analysis JSON/CSV.
                Defaults to <input-dir>/analysis/.  (spec §Q12)

Algorithm name mapping (spec §Q2):
  Directory suffix _stagewise -> safe-nonlinear (safe_q_stagewise, safe_vi)
  Directory suffix _zero      -> safe-zero (safe_q_zero, safe_vi_zero)
  No suffix                   -> classical (classical_q, classical_vi, etc.)

Runner subagent is responsible for matching directory names to these
canonical names.  Here we only read summary JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Allow running as a script from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.weighted_lse_dp.analysis.paired_bootstrap import paired_bootstrap_ci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Activation gate thresholds (reused from Phase IV-A)
_GATE_MEAN_ABS_U: float = 5e-3
_GATE_FRAC_ACTIVE: float = 0.10
_GATE_MEAN_ABS_DD: float = 1e-3
_GATE_MEAN_ABS_TG: float = 5e-3

# Diagnostic field names logged per transition (spec §6.1)
_DIAG_FIELDS: list[str] = [
    "mean_abs_natural_shift",
    "mean_abs_delta_effective_discount",
    "mean_abs_target_gap",
]

# Bootstrap params (spec §Q5)
_N_BOOTSTRAP: int = 10_000
_CI_LEVEL: float = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path.name}")


def _algo_class(algo_name: str) -> str:
    """Map algorithm directory name to canonical class label.

    Mapping (spec §Q3):
      *_stagewise  -> safe-nonlinear
      *_zero       -> safe-zero
      classical_*  -> classical
      safe_vi/safe_sync_vi/safe_async_vi -> safe-nonlinear (no suffix = mainline)
    """
    n = algo_name.lower()
    if n.endswith("_stagewise"):
        return "safe-nonlinear"
    if n.endswith("_zero"):
        return "safe-zero"
    if n.startswith("classical"):
        return "classical"
    if n.startswith("safe"):
        # safe without _zero suffix is the nonlinear mainline
        return "safe-nonlinear"
    return "unknown"


# ---------------------------------------------------------------------------
# Step 1 — Activation verification
# ---------------------------------------------------------------------------

def step1_activation_verification(
    input_dir: Path,
) -> dict[str, Any]:
    """Verify that safe-nonlinear runs show elevated operator diagnostics.

    Reads per-task summary JSONs from translation/<task_tag>/ and checks
    that the safe-nonlinear variant has higher diagnostics than safe-zero.

    Returns a dict keyed by task_tag with verification results.
    """
    trans_dir = input_dir / "translation"
    if not trans_dir.is_dir():
        warnings.warn(f"No translation dir at {trans_dir}; skipping step 1")
        return {}

    results: dict[str, Any] = {}
    for task_dir in sorted(trans_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tag = task_dir.name
        per_algo: dict[str, dict] = {}

        for summary_file in sorted(task_dir.glob("*/summary.json")):
            algo = summary_file.parent.name
            try:
                per_algo[algo] = _load_json(summary_file)
            except Exception as exc:
                warnings.warn(f"Could not read {summary_file}: {exc}")

        if not per_algo:
            results[tag] = {"status": "no_data"}
            continue

        # Collect diagnostic means by class
        by_class: dict[str, list[dict]] = {}
        for algo, data in per_algo.items():
            cls = _algo_class(algo)
            by_class.setdefault(cls, []).append(data)

        def _mean_diag(entries: list[dict], field: str) -> float | None:
            vals = []
            for e in entries:
                if field in e:
                    vals.append(e[field])
                elif "metrics" in e and field in e["metrics"]:
                    v = e["metrics"][field]
                    vals.append(v["mean"] if isinstance(v, dict) else v)
            return float(np.mean(vals)) if vals else None

        nonlinear = by_class.get("safe-nonlinear", [])
        safe_zero = by_class.get("safe-zero", [])

        checks: dict[str, bool] = {}
        for fld in _DIAG_FIELDS:
            nl_val = _mean_diag(nonlinear, fld)
            sz_val = _mean_diag(safe_zero, fld)
            if nl_val is not None and sz_val is not None:
                # Non-linear should have strictly larger mean absolute diagnostic
                checks[f"{fld}_nonlinear_gt_zero"] = nl_val > sz_val
            else:
                checks[f"{fld}_nonlinear_gt_zero"] = None  # type: ignore[assignment]

        # Activated = at least one diagnostic field shows nonlinear > zero
        definite_checks = [v for v in checks.values() if v is not None]
        activated = any(definite_checks) if definite_checks else None
        if not activated:
            warnings.warn(
                f"Task {tag}: no elevated diagnostics in safe-nonlinear vs "
                "safe-zero — not an activated Safe TAB case"
            )

        results[tag] = {
            "tag": tag,
            "activated": activated,
            "checks": checks,
            "algos_found": list(per_algo.keys()),
            "classes_found": list(by_class.keys()),
        }

    return results


# ---------------------------------------------------------------------------
# Step 2 — Within-family translation sweep (dose-response)
# ---------------------------------------------------------------------------

def step2_translation_sweep(
    input_dir: Path,
) -> dict[str, Any]:
    """Compute dose-response between diagnostic strength and outcome delta.

    Reads diagnostic_sweep/<task_tag>/sweep_results.json.
    Gracefully skips missing files (spec §Q4).

    For each task, computes Spearman correlation between seed-averaged
    diagnostic deltas and seed-averaged outcome deltas across u_max sweep
    values (spec §Q6: one point per sweep value per task).

    Delta baseline: value at u_max=X minus value at u_max=0 (spec §Q7).
    """
    try:
        from scipy.stats import spearmanr  # type: ignore[import]
    except ImportError:
        spearmanr = None  # type: ignore[assignment]
        warnings.warn("scipy not available; Spearman correlations will be skipped")

    sweep_dir = input_dir / "diagnostic_sweep"
    if not sweep_dir.is_dir():
        warnings.warn(f"No diagnostic_sweep dir at {sweep_dir}; skipping step 2")
        return {}

    results: dict[str, Any] = {}
    for task_dir in sorted(sweep_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tag = task_dir.name
        sweep_file = task_dir / "sweep_results.json"
        if not sweep_file.exists():
            # Gracefully skip with warning (spec §Q4)
            warnings.warn(
                f"sweep_results.json not found for {tag}; skipping sweep analysis"
            )
            results[tag] = {"status": "missing_sweep_file"}
            continue

        try:
            sweep = _load_json(sweep_file)
        except Exception as exc:
            warnings.warn(f"Could not parse sweep file for {tag}: {exc}")
            results[tag] = {"status": "parse_error", "error": str(exc)}
            continue

        # Expected structure: list of {u_max, mean_abs_u, outcome_metric, ...}
        # or dict with "sweep_points" key
        points = sweep if isinstance(sweep, list) else sweep.get("sweep_points", [])
        if not points:
            results[tag] = {"status": "empty_sweep"}
            continue

        # Sort by u_max
        points = sorted(points, key=lambda p: float(p.get("u_max", 0.0)))

        u_maxes = [float(p["u_max"]) for p in points]
        diag_vals = [float(p.get("mean_abs_natural_shift", p.get("mean_abs_u", 0.0)))
                     for p in points]
        # Try primary outcome metric; fall back to mean_return
        outcome_vals = [
            float(p.get("primary_outcome", p.get("mean_return", 0.0)))
            for p in points
        ]

        # Find baseline at u_max=0 (spec §Q7, option b)
        baseline_diag: float | None = None
        baseline_outcome: float | None = None
        for p in points:
            if abs(float(p.get("u_max", -1.0))) < 1e-9:
                baseline_diag = float(
                    p.get("mean_abs_natural_shift", p.get("mean_abs_u", 0.0))
                )
                baseline_outcome = float(
                    p.get("primary_outcome", p.get("mean_return", 0.0))
                )
                break

        if baseline_diag is None:
            warnings.warn(
                f"Task {tag}: no u_max=0 baseline in sweep; "
                "using first point as baseline"
            )
            baseline_diag = diag_vals[0]
            baseline_outcome = outcome_vals[0]

        diag_deltas = [d - baseline_diag for d in diag_vals]
        outcome_deltas = [o - baseline_outcome for o in outcome_vals]  # type: ignore[operator]

        # Spearman correlation across sweep values (spec §Q6)
        spearman_rho: float | None = None
        spearman_p: float | None = None
        if spearmanr is not None and len(diag_deltas) >= 3:
            try:
                res = spearmanr(diag_deltas, outcome_deltas)
                spearman_rho = float(res.statistic)
                spearman_p = float(res.pvalue)
            except Exception as exc:
                warnings.warn(f"Spearman failed for {tag}: {exc}")

        results[tag] = {
            "tag": tag,
            "u_maxes": u_maxes,
            "diag_deltas": diag_deltas,
            "outcome_deltas": outcome_deltas,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "n_sweep_points": len(points),
        }

    return results


# ---------------------------------------------------------------------------
# Step 3 — Matched-control mechanism isolation
# ---------------------------------------------------------------------------

def step3_matched_control(
    input_dir: Path,
) -> dict[str, Any]:
    """Isolate TAB nonlinearity effect via matched classical / safe-zero / safe-nonlinear.

    Reads per-task/per-algo summary JSONs from translation/<task_tag>/<algo>/.
    For each task computes:
      - safe-nonlinear minus safe-zero: nonlinearity effect (spec §Q7)
      - classical minus safe-nonlinear: total effect
    Paired bootstrap CIs across seeds (spec §Q5, Q7).
    """
    trans_dir = input_dir / "translation"
    if not trans_dir.is_dir():
        warnings.warn(f"No translation dir at {trans_dir}; skipping step 3")
        return {}

    results: dict[str, Any] = {}
    for task_dir in sorted(trans_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tag = task_dir.name

        # Collect per-seed metrics by algorithm class
        # Expect summary.json in translation/<tag>/<algo>/summary.json
        # or per-seed files; use what's available
        class_seeds: dict[str, list[float]] = {
            "classical": [],
            "safe-zero": [],
            "safe-nonlinear": [],
        }

        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            cls = _algo_class(algo_dir.name)
            if cls not in class_seeds:
                continue

            # Try per-seed JSON files first, then summary.json
            seed_returns: list[float] = []
            for seed_file in sorted(algo_dir.glob("seed_*/metrics.json")):
                try:
                    d = _load_json(seed_file)
                    seed_returns.append(float(d.get("primary_outcome", d.get("mean_return", 0.0))))
                except Exception:
                    pass

            if not seed_returns:
                summary_path = algo_dir / "summary.json"
                if summary_path.exists():
                    try:
                        d = _load_json(summary_path)
                        seed_returns = list(d.get("seed_returns", [d.get("mean_return", 0.0)]))
                    except Exception:
                        pass

            class_seeds[cls].extend(seed_returns)

        def _paired_ci(
            a_vals: list[float], b_vals: list[float], label: str
        ) -> dict[str, Any]:
            """Compute paired bootstrap CI; truncate to min length (spec Q5)."""
            n = min(len(a_vals), len(b_vals))
            if n < 2:
                return {"lower": None, "upper": None, "mean_diff": None,
                        "n_pairs": n, "note": "insufficient_pairs"}
            a = np.array(a_vals[:n])
            b = np.array(b_vals[:n])
            lo, hi, md = paired_bootstrap_ci(a, b, n_bootstrap=_N_BOOTSTRAP,
                                              ci=_CI_LEVEL, seed=0)
            return {"lower": lo, "upper": hi, "mean_diff": md, "n_pairs": n}

        nl = class_seeds["safe-nonlinear"]
        sz = class_seeds["safe-zero"]
        cl = class_seeds["classical"]

        # nonlinearity effect: safe-nonlinear minus safe-zero (spec §Q7)
        nonlinearity_effect = _paired_ci(nl, sz, "nonlinearity")
        # total effect: safe-nonlinear minus classical
        total_effect = _paired_ci(nl, cl, "total")
        # path effect: safe-zero minus classical
        path_effect = _paired_ci(sz, cl, "path")

        results[tag] = {
            "tag": tag,
            "nonlinearity_effect": nonlinearity_effect,
            "total_effect": total_effect,
            "path_effect": path_effect,
            "n_seeds": {k: len(v) for k, v in class_seeds.items()},
        }

    return results


# ---------------------------------------------------------------------------
# Step 4 — Outcome-specific interpretation
# ---------------------------------------------------------------------------

def step4_outcome_interpretation(
    input_dir: Path,
) -> dict[str, Any]:
    """Load primary outcomes config and map task families.

    Reads configs/phase4/primary_outcomes.json to assign primary metric
    per family type — metric is loaded from config, not chosen post-hoc
    (spec §10.2 rule 5).
    """
    # Resolve repo root relative to this file
    repo_root = Path(__file__).resolve().parents[3]
    outcomes_cfg_path = (
        repo_root
        / "experiments/weighted_lse_dp/configs/phase4/primary_outcomes.json"
    )
    if not outcomes_cfg_path.exists():
        warnings.warn(f"primary_outcomes.json not found at {outcomes_cfg_path}")
        return {}

    cfg = _load_json(outcomes_cfg_path)
    outcomes_by_type: dict[str, dict] = cfg.get("outcomes_by_family_type", {})

    trans_dir = input_dir / "translation"
    if not trans_dir.is_dir():
        return {}

    results: dict[str, Any] = {}
    for task_dir in sorted(trans_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tag = task_dir.name

        # Infer family type from tag (best-effort; runner names dir by task tag)
        family_type: str | None = None
        for ftype in outcomes_by_type:
            if ftype in tag:
                family_type = ftype
                break

        primary_metric: str | None = None
        secondary_metrics: list[str] = []
        if family_type and family_type in outcomes_by_type:
            primary_metric = outcomes_by_type[family_type]["primary"]
            secondary_metrics = outcomes_by_type[family_type].get("secondary", [])

        results[tag] = {
            "tag": tag,
            "family_type": family_type,
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
        }

    return results


# ---------------------------------------------------------------------------
# Step 5 — Negative-control consistency
# ---------------------------------------------------------------------------

def step5_negative_control(
    input_dir: Path,
) -> dict[str, Any]:
    """Verify Phase III replay families show near-zero activation diagnostics.

    Reads counterfactual_replay/ (negative control, original 5 families).
    Uses counterfactual_replay/ NOT counterfactual_replay_4a2/ (spec §Q8).
    """
    neg_ctrl_dir = input_dir / "counterfactual_replay"  # negative control (spec §Q8)
    if not neg_ctrl_dir.is_dir():
        warnings.warn(f"No counterfactual_replay dir at {neg_ctrl_dir}; skipping step 5")
        return {}

    results: dict[str, Any] = {}
    large_diff_flags: list[str] = []

    for task_dir in sorted(neg_ctrl_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tag = task_dir.name
        summary_path = task_dir / "replay_summary.json"
        if not summary_path.exists():
            results[tag] = {"status": "no_summary"}
            continue

        try:
            d = _load_json(summary_path)
        except Exception as exc:
            results[tag] = {"status": "parse_error", "error": str(exc)}
            continue

        mean_abs_u = float(d.get("mean_abs_u", 0.0))
        frac_active = float(d.get("frac_u_ge_5e3", 0.0))
        mean_abs_dd = float(d.get("mean_abs_delta_d", 0.0))

        # Near-zero thresholds (below Phase IV-A gate)
        near_zero = (
            mean_abs_u < _GATE_MEAN_ABS_U
            and frac_active < _GATE_FRAC_ACTIVE
            and mean_abs_dd < _GATE_MEAN_ABS_DD
        )

        if not near_zero:
            large_diff_flags.append(tag)
            warnings.warn(
                f"Negative-control task {tag} has non-trivial activation "
                f"(mean_abs_u={mean_abs_u:.4e}); investigate as possible "
                "implementation artifact"
            )

        results[tag] = {
            "tag": tag,
            "mean_abs_u": mean_abs_u,
            "frac_active": frac_active,
            "mean_abs_delta_d": mean_abs_dd,
            "near_zero": near_zero,
        }

    return {
        "per_task": results,
        "large_diff_tasks": large_diff_flags,
        "all_near_zero": len(large_diff_flags) == 0,
    }


# ---------------------------------------------------------------------------
# Null-case reporting
# ---------------------------------------------------------------------------

def _report_nulls(
    step1: dict[str, Any],
    step3: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect tasks where activation did not translate. Never silently dropped."""
    nulls: list[dict[str, Any]] = []
    for tag, info in step3.items():
        te = info.get("total_effect", {})
        md = te.get("mean_diff")
        if md is None:
            continue
        lo = te.get("lower")
        hi = te.get("upper")
        # Null: CI contains zero AND activation was present
        activated = step1.get(tag, {}).get("activated")
        if lo is not None and hi is not None and lo < 0 < hi:
            nulls.append({
                "tag": tag,
                "activated": activated,
                "mean_diff": md,
                "ci_lower": lo,
                "ci_upper": hi,
                "interpretation": (
                    "Activation present but outcome CI contains zero — "
                    "null translation" if activated else "Activation absent"
                ),
            })
    return nulls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Run Phase IV-B five-step translation analysis pipeline."""
    input_dir = args.input_dir.resolve()
    # Default output-dir = <input-dir>/analysis/ (spec §Q12)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase IV-B translation analysis")
    print(f"  input_dir:  {input_dir}")
    print(f"  output_dir: {output_dir}")
    print()

    print("Step 1: activation verification …")
    s1 = step1_activation_verification(input_dir)
    _write_json(s1, output_dir / "step1_activation_verification.json")

    print("Step 2: translation sweep (dose-response) …")
    s2 = step2_translation_sweep(input_dir)
    _write_json(s2, output_dir / "step2_translation_sweep.json")

    print("Step 3: matched-control mechanism isolation …")
    s3 = step3_matched_control(input_dir)
    _write_json(s3, output_dir / "step3_matched_control.json")

    print("Step 4: outcome-specific interpretation …")
    s4 = step4_outcome_interpretation(input_dir)
    _write_json(s4, output_dir / "step4_outcome_interpretation.json")

    print("Step 5: negative-control consistency …")
    s5 = step5_negative_control(input_dir)
    _write_json(s5, output_dir / "step5_negative_control.json")

    print("Null-case report …")
    nulls = _report_nulls(s1, s3)
    _write_json(nulls, output_dir / "null_translation_cases.json")
    if nulls:
        print(f"  {len(nulls)} null translation case(s) — see null_translation_cases.json")
    else:
        print("  No null cases (or insufficient data for determination)")

    # Top-level summary
    n_activated = sum(
        1 for v in s1.values() if v.get("activated") is True
    )
    n_null = len(nulls)
    n_neg_ctrl = len(s5.get("per_task", {}))
    neg_ok = s5.get("all_near_zero", True)

    summary = {
        "n_tasks_verified": len(s1),
        "n_activated": n_activated,
        "n_null_translation": n_null,
        "n_sweep_tasks": len(s2),
        "n_matched_control_tasks": len(s3),
        "n_negative_control_tasks": n_neg_ctrl,
        "negative_control_all_near_zero": neg_ok,
    }
    _write_json(summary, output_dir / "translation_analysis_summary.json")

    print(f"\n--- Translation Analysis Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nOutput dir: {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir", type=Path, required=True,
        help="Top-level Phase IV-B results directory"
    )
    # No --config (spec §Q1: analysis reads data directly)
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory; defaults to <input-dir>/analysis/"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
