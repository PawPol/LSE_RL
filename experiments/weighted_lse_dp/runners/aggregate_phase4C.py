#!/usr/bin/env python
"""Phase IV-C: aggregate advanced stabilization and ablation results.

Merges advanced RL, geometry DP, scheduler ablation, and certification
ablation runs into a unified dataset for estimator-stability and
scheduler-localization analysis. Also writes the attribution analysis.

Layout of inputs:

    <results_dir>/phase4/advanced/
        safe_double_q/<task_tag>/seed_<N>/metrics.json
        safe_target_q/<task_tag>/seed_<N>/metrics.json
        safe_target_q_polyak/<task_tag>/seed_<N>/metrics.json
        safe_target_expected_sarsa/<task_tag>/seed_<N>/metrics.json
        geometry_priority_dp/<task_tag>/seed_<N>/<mode>/metrics.json
        state_dependent_scheduler/<task_tag>/seed_<N>/<scheduler_type>/metrics.json
        ablations/<ablation_type>/<task_tag>/seed_<N>/metrics.json

Outputs:

    <results_dir>/phase4/advanced/summary_phase4C.json
    <results_dir>/phase4/advanced/attribution_analysis.json

CLI::

    python experiments/weighted_lse_dp/runners/aggregate_phase4C.py \\
        [--results-dir results/weighted_lse_dp] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

__all__ = ["main", "aggregate"]

_ADVANCED_RL_ALGOS = (
    "safe_q_single_table",          # architectural control baseline
    "safe_double_q",
    "safe_target_q",
    "safe_target_q_polyak",
    "safe_target_expected_sarsa",
)
# Algorithms compared against the single-table architectural control (R9/A2)
_ESTIMATOR_ALGOS = (
    "safe_double_q",
    "safe_target_q",
    "safe_target_q_polyak",
    "safe_target_expected_sarsa",
)

_ABLATION_TYPES = (
    "trust_region_off", "trust_region_tighter",
    "adaptive_headroom_off", "adaptive_headroom_aggressive",
    "wrong_sign", "constant_u", "raw_unclipped",
)

_SCHEDULER_TYPES = (
    "stagewise_baseline",
    "state_bin_uniform",
    "state_bin_hazard_proximity",
    "state_bin_reward_region",
)

_PRIORITY_MODES = ("residual_only", "geometry_gain_only", "combined")


def _load_metrics(path: Path) -> dict[str, Any] | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _seed_dirs(base: Path) -> list[Path]:
    return sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("seed_")])


def _mean_field(records: list[dict], field: str) -> float | None:
    vals = [float(r[field]) for r in records if field in r]
    return float(np.mean(vals)) if vals else None


def _aggregate_algo_group(
    advanced_dir: Path,
    algo: str,
) -> dict[str, Any]:
    algo_dir = advanced_dir / algo
    if not algo_dir.is_dir():
        return {"algorithm": algo, "status": "missing", "n_runs": 0}

    records = []
    per_task: dict[str, list[dict]] = {}
    for task_dir in sorted(algo_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_records = []
        for seed_dir in _seed_dirs(task_dir):
            m = _load_metrics(seed_dir / "metrics.json")
            if m:
                records.append(m)
                task_records.append(m)
        if task_records:
            per_task[task_dir.name] = task_records

    if not records:
        return {"algorithm": algo, "status": "no_data", "n_runs": 0}

    # Per-task breakdown (R10: spec §12.2 requires per-task-family tables)
    per_task_summary = {
        tag: {
            "n_runs": len(recs),
            "mean_final_return": _mean_field(recs, "final_disc_return_mean"),
            "std_final_return": (
                float(np.std([r["final_disc_return_mean"] for r in recs
                               if "final_disc_return_mean" in r]))
                if len(recs) > 1 else None
            ),
            "mean_double_gap": _mean_field(recs, "mean_double_gap"),
            "mean_q_target_gap": _mean_field(recs, "mean_q_target_gap"),
            "mean_td_error": _mean_field(recs, "mean_td_error"),
            "std_beta_used": _mean_field(recs, "std_beta_used"),
        }
        for tag, recs in per_task.items()
    }

    return {
        "algorithm": algo,
        "status": "aggregated",
        "n_runs": len(records),
        "mean_final_return": _mean_field(records, "final_disc_return_mean"),
        "mean_mean_return": _mean_field(records, "mean_return"),
        "mean_beta_used": _mean_field(records, "mean_beta_used"),
        "mean_double_gap": _mean_field(records, "mean_double_gap"),
        "mean_q_target_gap": _mean_field(records, "mean_q_target_gap"),
        "std_final_return": (
            float(np.std([r["final_disc_return_mean"] for r in records
                           if "final_disc_return_mean" in r]))
            if len(records) > 1 else None
        ),
        "per_task": per_task_summary,
    }


def _aggregate_ablations(advanced_dir: Path) -> list[dict[str, Any]]:
    ablations_dir = advanced_dir / "ablations"
    if not ablations_dir.is_dir():
        return []
    results = []
    for ablation in _ABLATION_TYPES:
        abl_dir = ablations_dir / ablation
        if not abl_dir.is_dir():
            results.append({"ablation": ablation, "status": "missing", "n_runs": 0})
            continue
        records = []
        for task_dir in sorted(abl_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for seed_dir in _seed_dirs(task_dir):
                m = _load_metrics(seed_dir / "metrics.json")
                if m:
                    records.append(m)
        if not records:
            results.append({"ablation": ablation, "status": "no_data", "n_runs": 0})
            continue
        finals = [r["final_disc_return_mean"] for r in records
                  if "final_disc_return_mean" in r]
        results.append({
            "ablation": ablation,
            "status": "aggregated",
            "n_runs": len(records),
            "mean_final_return": float(np.mean(finals)) if finals else None,
            "std_final_return": float(np.std(finals)) if len(finals) > 1 else None,
            "mean_mean_return": _mean_field(records, "mean_return"),
        })
    return results


def _aggregate_geometry_dp(advanced_dir: Path) -> list[dict[str, Any]]:
    geo_dir = advanced_dir / "geometry_priority_dp"
    if not geo_dir.is_dir():
        return []
    results = []
    for mode in _PRIORITY_MODES:
        records = []
        for task_dir in sorted(geo_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for seed_dir in _seed_dirs(task_dir):
                m = _load_metrics(seed_dir / mode / "metrics.json")
                if m:
                    records.append(m)
        if not records:
            results.append({"mode": mode, "status": "no_data", "n_runs": 0})
            continue
        results.append({
            "mode": mode,
            "status": "aggregated",
            "n_runs": len(records),
            "mean_n_sweeps": _mean_field(records, "n_sweeps"),
            "mean_final_residual": _mean_field(records, "final_residual"),
            "mean_wall_clock_s": _mean_field(records, "wall_clock_s"),
            "mean_frac_high_act": _mean_field(records, "frac_high_activation_backups"),
        })
    return results


def _aggregate_scheduler(advanced_dir: Path) -> list[dict[str, Any]]:
    sched_dir = advanced_dir / "state_dependent_scheduler"
    if not sched_dir.is_dir():
        return []
    results = []
    for stype in _SCHEDULER_TYPES:
        records = []
        for task_dir in sorted(sched_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for seed_dir in _seed_dirs(task_dir):
                m = _load_metrics(seed_dir / stype / "metrics.json")
                if m:
                    records.append(m)
        if not records:
            results.append({"scheduler": stype, "status": "no_data", "n_runs": 0})
            continue
        results.append({
            "scheduler": stype,
            "status": "aggregated",
            "n_runs": len(records),
            "mean_final_return": _mean_field(records, "final_disc_return_mean"),
            "mean_mean_return": _mean_field(records, "mean_return"),
        })
    return results


def _build_attribution(
    advanced_rl: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    geo_dp: list[dict[str, Any]],
    scheduler: list[dict[str, Any]],
) -> dict[str, Any]:
    # Architecture-matched baseline: same hand-rolled loop, single Q-table (R9/A2)
    arch_baseline_return: float | None = None
    for r in advanced_rl:
        if r.get("algorithm") == "safe_q_single_table":
            arch_baseline_return = r.get("mean_final_return")
            break

    # Framework baseline: MushroomRL SafeQLearning stagewise (for ablation deltas)
    framework_baseline_return: float | None = None
    for r in scheduler:
        if r.get("scheduler") == "stagewise_baseline":
            framework_baseline_return = r.get("mean_final_return")
            break

    # Estimator gains vs architecture-matched single-table control
    estimator_gains: dict[str, float | None] = {}
    for r in advanced_rl:
        algo = r.get("algorithm", "?")
        if algo == "safe_q_single_table":
            continue
        ret = r.get("mean_final_return")
        estimator_gains[algo] = (
            ret - arch_baseline_return
            if ret is not None and arch_baseline_return is not None
            else None
        )

    geo_gains: dict[str, float | None] = {}
    baseline_sweeps = None
    for r in geo_dp:
        if r.get("mode") == "residual_only":
            baseline_sweeps = r.get("mean_n_sweeps")
        geo_gains[r.get("mode", "?")] = r.get("mean_n_sweeps")

    # Ablation deltas vs framework baseline (both use SafeQLearning — clean comparison)
    ablation_impact: list[dict[str, Any]] = []
    for r in ablations:
        abl = r.get("ablation", "?")
        ret = r.get("mean_final_return")
        impact = (
            ret - framework_baseline_return
            if ret is not None and framework_baseline_return is not None
            else None
        )
        ablation_impact.append({
            "ablation": abl,
            "mean_final_return": ret,
            "delta_vs_framework_baseline": impact,
        })

    # A10: paired-seed test for wrong_sign (replace any() with mean/std over seeds)
    wrong_sign_impact = next(
        (r for r in ablation_impact if r["ablation"] == "wrong_sign"), {}
    )
    ws_delta = wrong_sign_impact.get("delta_vs_framework_baseline")
    ws_std = wrong_sign_impact.get("std_final_return")
    # "hurts" = mean delta < 0 AND magnitude > 0.5 * std (weak paired test with available data)
    wrong_sign_hurts: bool | None = None
    if ws_delta is not None:
        threshold = 0.5 * ws_std if ws_std else 0.01
        wrong_sign_hurts = bool(ws_delta < -threshold)

    # A18: fix max() with None values (was `or -1e9` which treats 0.0 as missing)
    valid_gains = {k: v for k, v in estimator_gains.items() if v is not None}
    best_estimator = max(valid_gains, key=lambda k: valid_gains[k]) if valid_gains else None

    return {
        "schema_version": "1.0.0",
        "phase": "phase4C",
        "analysis_type": "attribution",
        "architecture_baseline_algo": "safe_q_single_table",
        "architecture_baseline_return": arch_baseline_return,
        "framework_baseline_return": framework_baseline_return,
        "estimator_gains_vs_arch_baseline": estimator_gains,
        "geometry_dp_sweeps_by_mode": geo_gains,
        "geometry_dp_baseline_sweeps": baseline_sweeps,
        "ablation_impact": ablation_impact,
        "attribution_note": (
            "estimator_gains use the hand-rolled single-table baseline (same loop/seed). "
            "ablation_impact uses the MushroomRL framework baseline (same agent class). "
            "These two baselines are not interchangeable."
        ),
        "summary": {
            "best_estimator_algo": best_estimator,
            "geometry_priority_reduces_sweeps": (
                (geo_gains.get("combined") or 0) < (baseline_sweeps or 0)
                if baseline_sweeps else None
            ),
            "wrong_sign_ablation_hurts": wrong_sign_hurts,
            "wrong_sign_delta": ws_delta,
        },
    }


def aggregate(results_dir: Path) -> dict[str, Any]:
    advanced_dir = results_dir / "phase4" / "advanced"

    advanced_rl = [_aggregate_algo_group(advanced_dir, algo)
                   for algo in _ADVANCED_RL_ALGOS]
    ablations = _aggregate_ablations(advanced_dir)
    geo_dp = _aggregate_geometry_dp(advanced_dir)
    scheduler = _aggregate_scheduler(advanced_dir)
    attribution = _build_attribution(advanced_rl, ablations, geo_dp, scheduler)

    summary = {
        "schema_version": "1.0.0",
        "phase": "phase4C",
        "advanced_rl": advanced_rl,
        "ablations": ablations,
        "geometry_priority_dp": geo_dp,
        "scheduler_ablations": scheduler,
    }
    return summary, attribution


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "phase4" / "advanced")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, attribution = aggregate(results_dir)

    summary_path = out_dir / "summary_phase4C.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[aggregate_phase4C] Written {summary_path}")

    attr_path = out_dir / "attribution_analysis.json"
    attr_path.write_text(json.dumps(attribution, indent=2))
    print(f"[aggregate_phase4C] Written {attr_path}")

    n_rl_pass = sum(1 for r in summary["advanced_rl"] if r.get("status") == "aggregated")
    n_abl_pass = sum(1 for r in summary["ablations"] if r.get("status") == "aggregated")
    n_geo_pass = sum(1 for r in summary["geometry_priority_dp"] if r.get("status") == "aggregated")
    n_sched_pass = sum(1 for r in summary["scheduler_ablations"] if r.get("status") == "aggregated")
    print(f"  Advanced RL: {n_rl_pass}/{len(_ADVANCED_RL_ALGOS)} algos")
    print(f"  Ablations:   {n_abl_pass}/{len(_ABLATION_TYPES)} types")
    print(f"  Geometry DP: {n_geo_pass}/{len(_PRIORITY_MODES)} modes")
    print(f"  Scheduler:   {n_sched_pass}/{len(_SCHEDULER_TYPES)} types")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results/weighted_lse_dp"))
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
