#!/usr/bin/env python
"""Phase IV gate checks for the overnight autonomous pipeline.

Each sub-phase has mandatory exit criteria that must pass before the pipeline
can proceed to the next sub-phase. This script checks those criteria by
inspecting result artifacts, test outputs, and checkpoint state.

Usage:
    python scripts/overnight/check_gate.py --phase IV-A [--results-dir results/weighted_lse_dp/phase4]
    python scripts/overnight/check_gate.py --phase IV-B [--results-dir results/weighted_lse_dp/phase4]
    python scripts/overnight/check_gate.py --phase IV-C [--results-dir results/weighted_lse_dp/phase4]

Exit codes:
    0 = PASS (all gate conditions met)
    1 = FAIL (at least one condition not met; details printed to stdout)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(condition: bool, description: str, details: str = "") -> dict:
    """Return a gate-check result dict."""
    return {
        "condition": description,
        "passed": condition,
        "details": details,
    }


def _file_exists(path: Path, description: str) -> dict:
    return _check(path.exists(), f"File exists: {description}", str(path))


def _dir_nonempty(path: Path, description: str) -> dict:
    """Check directory exists and has real content (ignoring .gitkeep)."""
    if not path.is_dir():
        return _check(False, f"Non-empty directory: {description}", str(path))
    contents = [f for f in path.iterdir() if f.name != ".gitkeep"]
    return _check(len(contents) > 0, f"Non-empty directory: {description}", str(path))


# ---------------------------------------------------------------------------
# IV-A: Activation gate (spec §13) — Option C, three-condition design
# ---------------------------------------------------------------------------

GATE_VERSION = "option_c_v1"
U_THRESHOLD = 5e-3
FRAC_THRESHOLD = 0.10


def _read_candidate_predictions(scores_file: Path) -> dict[str, list[float]]:
    """Return {family: [mean_abs_u_pred, ...]} from candidate_scores.csv."""
    out: dict[str, list[float]] = defaultdict(list)
    if not scores_file.exists():
        return out
    import csv
    with open(scores_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fam = row.get("family", "").strip()
            if not fam:
                continue
            try:
                v = float(row.get("mean_abs_u_pred", 0.0) or 0.0)
            except ValueError:
                v = 0.0
            out[fam].append(v)
    return out


def _family_eligibility(
    replay_tasks: list[dict],
    predictions_by_family: dict[str, list[float]],
) -> list[dict]:
    """Build the per-family eligibility records (Option C, v1).

    For each family observed in the replay, evaluate conditions 1/2a/2b and
    tag IV-B eligibility. For families present in the prediction CSV but not
    in the replay, the record still appears (with gate_pass=false for the
    replay-side conditions) so the report is honest.
    """
    # Group replay entries by family; best-by-metric within each group.
    by_family: dict[str, list[dict]] = defaultdict(list)
    for t in replay_tasks:
        fam = str(t.get("family", "unknown"))
        by_family[fam].append(t)

    families = sorted(set(list(by_family.keys()) + list(predictions_by_family.keys())))
    records: list[dict] = []
    for fam in families:
        preds = predictions_by_family.get(fam, [])
        # Design-point prediction: best (max) over all candidates for the
        # family, matching the activation-search convention.
        mean_abs_u_pred = max(preds) if preds else 0.0
        c1_pass = mean_abs_u_pred >= U_THRESHOLD

        tasks = by_family.get(fam, [])
        if tasks:
            # Pick the task with the strongest informative mean to report.
            best = max(
                tasks,
                key=lambda t: float(t.get("mean_abs_u_replay_informative", 0.0)),
            )
            mean_info = float(best.get("mean_abs_u_replay_informative", 0.0))
            median_info = float(best.get("median_abs_u_replay_informative", 0.0))
            frac_info_u = float(best.get("frac_informative_u_ge_5e3", 0.0))
            n_info = int(best.get("n_informative_transitions", 0))
            frac_info = float(best.get("frac_informative_transitions", 0.0))
            mean_global = float(best.get("mean_abs_u_replay_global",
                                         best.get("mean_abs_u", 0.0)))
            mean_topq = float(best.get("mean_abs_u_replay_topquartile", 0.0))
            frac_topq_u = float(best.get("frac_topquartile_u_ge_5e3", 0.0))
            n_topq = int(best.get("n_topquartile_transitions", 0))
            c2a_pass = (mean_info >= U_THRESHOLD) or (median_info >= U_THRESHOLD)
            c2b_pass = frac_info_u >= FRAC_THRESHOLD
            iv_b_eligible = bool(c1_pass and c2a_pass and c2b_pass)
            reason_parts = []
            if not c1_pass:
                reason_parts.append(
                    f"design-point {mean_abs_u_pred:.6f} < {U_THRESHOLD:.0e}"
                )
            if not c2a_pass:
                reason_parts.append(
                    f"informative mean={mean_info:.6f} median={median_info:.6f} "
                    f"both < {U_THRESHOLD:.0e}"
                )
            if not c2b_pass:
                reason_parts.append(
                    f"informative frac(|u|>={U_THRESHOLD:.0e})={frac_info_u:.4f} "
                    f"< {FRAC_THRESHOLD:.2f}"
                )
            reason = (
                "all three conditions pass"
                if iv_b_eligible
                else "; ".join(reason_parts)
            )
            rec = {
                "family": fam,
                "design_point": {
                    "mean_abs_u_pred": mean_abs_u_pred,
                    "gate_pass": bool(c1_pass),
                },
                "informative_replay": {
                    "mean_abs_u": mean_info,
                    "median_abs_u": median_info,
                    "frac_u_ge_5e3": frac_info_u,
                    "n_informative": n_info,
                    "frac_informative": frac_info,
                    "gate_pass_mean_or_median": bool(c2a_pass),
                    "gate_pass_frac": bool(c2b_pass),
                },
                "topquartile_replay": {
                    "mean_abs_u": mean_topq,
                    "frac_u_ge_5e3": frac_topq_u,
                    "n_topquartile": n_topq,
                },
                "global_replay": {
                    "mean_abs_u": mean_global,
                    "is_dilution_only": True,
                },
                "iv_b_eligible": iv_b_eligible,
                "eligibility_reason": reason,
                "tag": str(best.get("tag", "")),
            }
        else:
            rec = {
                "family": fam,
                "design_point": {
                    "mean_abs_u_pred": mean_abs_u_pred,
                    "gate_pass": bool(c1_pass),
                },
                "informative_replay": {
                    "mean_abs_u": 0.0,
                    "median_abs_u": 0.0,
                    "frac_u_ge_5e3": 0.0,
                    "n_informative": 0,
                    "frac_informative": 0.0,
                    "gate_pass_mean_or_median": False,
                    "gate_pass_frac": False,
                },
                "topquartile_replay": {
                    "mean_abs_u": 0.0,
                    "frac_u_ge_5e3": 0.0,
                    "n_topquartile": 0,
                },
                "global_replay": {
                    "mean_abs_u": 0.0,
                    "is_dilution_only": True,
                },
                "iv_b_eligible": False,
                "eligibility_reason": "no replay entry for this family",
                "tag": "",
            }
        records.append(rec)
    return records


def _write_eligibility_report(
    results_dir: Path,
    records: list[dict],
    suffix: str = "",
) -> Path:
    """Persist the Option C eligibility report to JSON and return the path."""
    report_dir = results_dir / "activation_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate_version": GATE_VERSION,
        "thresholds": {
            "mean_abs_u": U_THRESHOLD,
            "frac_informative_u_ge_5e3": FRAC_THRESHOLD,
        },
        "families": records,
    }
    out_path = report_dir / f"family_eligibility{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def check_gate_iva(results_dir: Path, configs_dir: Path, suffix: str = "") -> list[dict]:
    """Phase IV-A activation gate (Option C, v1).

    Three-condition activation gate:

    * Condition 1 — design-point activation: at least one family has
      ``mean_abs_u_pred >= 5e-3`` in ``task_search/candidate_scores.csv``.
    * Condition 2a — informative-conditioned replay activation: at least one
      replay task has ``mean_abs_u_replay_informative >= 5e-3`` OR
      ``median_abs_u_replay_informative >= 5e-3``.
    * Condition 2b — informative active fraction: at least one replay task
      has ``frac_informative_u_ge_5e3 >= 0.10``.

    The global replay mean is reported as an ``[INFO]`` dilution diagnostic
    only; it is not a pass/fail condition.
    """
    checks: list[dict] = []
    audit_dir = results_dir / "audit"
    search_dir = results_dir / "task_search"
    replay_dir = results_dir / f"counterfactual_replay{suffix}"

    # 1. Audit artifacts (shared; no suffix — Phase III compat is not namespaced)
    checks.append(_file_exists(audit_dir / "phase3_compat_report.md", "Phase III compat report"))
    checks.append(_file_exists(audit_dir / "phase3_code_audit.json", "Phase III code audit"))
    checks.append(_file_exists(audit_dir / "phase3_result_audit.json", "Phase III result audit"))

    # 2. Activation search
    selected = search_dir / f"selected_tasks{suffix}.json"
    checks.append(_file_exists(selected, "Selected tasks file"))
    if selected.exists():
        try:
            data = json.loads(selected.read_text())
            families = data if isinstance(data, list) else data.get("selected_families", [])
            checks.append(_check(
                len(families) > 0,
                "At least one task family selected",
                f"{len(families)} families selected",
            ))
        except (json.JSONDecodeError, KeyError) as e:
            checks.append(_check(False, "selected_tasks.json is valid JSON", str(e)))
    else:
        checks.append(_check(False, "At least one task family selected", "File missing"))

    checks.append(_file_exists(
        search_dir / f"activation_search_report{suffix}.md", "Activation search report"
    ))

    # 3. Counterfactual replay
    checks.append(_dir_nonempty(replay_dir, "Counterfactual replay results"))

    # 4. Three-condition activation gate (Option C, v1)
    replay_summary_file = replay_dir / "all_replay_summaries.json"
    scores_file = search_dir / f"candidate_scores{suffix}.csv"

    predictions_by_family = _read_candidate_predictions(scores_file)

    # Condition 1: design-point activation (always evaluated against the
    # prediction CSV — this is authoritative for the upstream schedule).
    best_design = max(
        (v for vs in predictions_by_family.values() for v in vs),
        default=0.0,
    )
    c1_pass = best_design >= U_THRESHOLD
    if scores_file.exists():
        checks.append(_check(
            c1_pass,
            "[GATE 1] Design-point: any family mean_abs_u_pred >= 5e-3",
            f"best={best_design:.6f} across "
            f"{sum(len(v) for v in predictions_by_family.values())} candidates",
        ))
    else:
        checks.append(_check(
            False,
            "[GATE 1] Design-point: any family mean_abs_u_pred >= 5e-3",
            f"candidate_scores.csv not found at {scores_file}",
        ))

    replay_tasks: list[dict] = []
    replay_parse_error: str | None = None
    if replay_summary_file.exists():
        try:
            with open(replay_summary_file) as f:
                replay_data = json.load(f)
            replay_tasks = (
                replay_data.get("tasks", [])
                if isinstance(replay_data, dict) else []
            )
        except Exception as e:
            replay_parse_error = str(e)
            replay_tasks = []

    if replay_summary_file.exists() and replay_parse_error is None:
        if replay_tasks:
            # Condition 2a
            has_2a = any(
                (
                    float(t.get("mean_abs_u_replay_informative", 0.0)) >= U_THRESHOLD
                    or float(t.get("median_abs_u_replay_informative", 0.0)) >= U_THRESHOLD
                )
                for t in replay_tasks
            )
            best_mean_info = max(
                (float(t.get("mean_abs_u_replay_informative", 0.0))
                 for t in replay_tasks),
                default=0.0,
            )
            best_median_info = max(
                (float(t.get("median_abs_u_replay_informative", 0.0))
                 for t in replay_tasks),
                default=0.0,
            )
            checks.append(_check(
                has_2a,
                "[GATE 2a] Informative replay: mean OR median >= 5e-3",
                f"best_mean={best_mean_info:.6f} best_median={best_median_info:.6f} "
                f"across {len(replay_tasks)} replay tasks",
            ))

            # Condition 2b
            has_2b = any(
                float(t.get("frac_informative_u_ge_5e3", 0.0)) >= FRAC_THRESHOLD
                for t in replay_tasks
            )
            best_frac_info = max(
                (float(t.get("frac_informative_u_ge_5e3", 0.0))
                 for t in replay_tasks),
                default=0.0,
            )
            checks.append(_check(
                has_2b,
                "[GATE 2b] Informative replay: frac(|u|>=5e-3) >= 10%",
                f"best={best_frac_info:.4f} across {len(replay_tasks)} replay tasks",
            ))

            # [INFO] line (not a gate) — global replay diagnostic.
            best_global = max(
                (float(t.get("mean_abs_u_replay_global",
                             t.get("mean_abs_u", 0.0)))
                 for t in replay_tasks),
                default=0.0,
            )
            checks.append(_check(
                True,
                f"[INFO] Global replay mean_abs_u (dilution diagnostic, not gated)"
                f" best={best_global:.6f}",
                f"Global mean is diluted across non-informative stages; "
                f"use informative-conditioned metrics for gating.",
            ))
        else:
            checks.append(_check(
                False,
                "[GATE 2a] Informative replay: mean OR median >= 5e-3",
                "No tasks in all_replay_summaries.json",
            ))
            checks.append(_check(
                False,
                "[GATE 2b] Informative replay: frac(|u|>=5e-3) >= 10%",
                "No tasks in all_replay_summaries.json",
            ))
            checks.append(_check(
                True,
                "[INFO] Global replay mean_abs_u (dilution diagnostic, not gated)",
                "No replay tasks to report.",
            ))
    elif replay_parse_error is not None:
        checks.append(_check(False, "Replay summaries parseable", replay_parse_error))
        checks.append(_check(
            False,
            "[GATE 2a] Informative replay: mean OR median >= 5e-3",
            replay_parse_error,
        ))
        checks.append(_check(
            False,
            "[GATE 2b] Informative replay: frac(|u|>=5e-3) >= 10%",
            replay_parse_error,
        ))
    else:
        checks.append(_check(
            False,
            "[GATE 2a] Informative replay: mean OR median >= 5e-3",
            f"{replay_summary_file} missing",
        ))
        checks.append(_check(
            False,
            "[GATE 2b] Informative replay: frac(|u|>=5e-3) >= 10%",
            f"{replay_summary_file} missing",
        ))

    # 5. Per-family eligibility: build records, persist, and attach as an
    # [INFO] check block so the output surfaces them alongside the gates.
    eligibility_records = _family_eligibility(replay_tasks, predictions_by_family)
    try:
        eligibility_path = _write_eligibility_report(results_dir, eligibility_records, suffix=suffix)
        checks.append(_check(
            True,
            f"[INFO] Wrote per-family eligibility report",
            str(eligibility_path),
        ))
    except Exception as e:
        checks.append(_check(
            False,
            "Write per-family eligibility report",
            str(e),
        ))

    # Attach the records to the last [INFO] check via a magic key so main()
    # can render them without a second pass through disk.
    checks[-1]["_eligibility_records"] = eligibility_records

    # 6. Configs
    checks.append(_file_exists(
        configs_dir / f"activation_suite{suffix}.json", "Frozen activation suite config"
    ))
    checks.append(_file_exists(configs_dir / "gamma_matched_controls.json", "Matched controls config"))

    return checks


# ---------------------------------------------------------------------------
# IV-B: Translation gate (spec §14)
# ---------------------------------------------------------------------------

def check_gate_ivb(results_dir: Path, configs_dir: Path, suffix: str = "") -> list[dict]:
    """Phase IV-B translation gate.

    Conditions (from spec §14):
    1. IV-A gate artifacts still present.
    2. Translation results directory is non-empty.
    3. All matched comparisons completed (classical, safe-zero, safe-nonlinear).
    4. Diagnostic-strength sweep complete.
    5. Translation analysis pipeline complete.
    6. Translation study config exists.
    """
    checks: list[dict] = []
    translation_dir = results_dir / f"translation{suffix}"

    # 1. IV-A artifacts still intact — check both unsuffixed and suffixed selected_tasks
    selected_tasks_path = results_dir / "task_search" / f"selected_tasks{suffix}.json"
    fallback_path = results_dir / "task_search" / "selected_tasks.json"
    prereq_ok = selected_tasks_path.exists() or fallback_path.exists()
    checks.append(_check(prereq_ok, "IV-A selected tasks (prerequisite)",
                         str(selected_tasks_path)))

    # 2. Translation results — check both the raw dir and aggregated subdir
    trans_ok = translation_dir.is_dir() and (
        any(translation_dir.glob("*/*/seed_*")) or
        (translation_dir / "aggregated").is_dir()
    )
    checks.append(_check(trans_ok, "Translation experiment results",
                         str(translation_dir)))

    # 3. Check for matched comparison artifacts (classical, safe-zero, safe-nonlinear).
    # Accept either old layout (classical_matched/ etc.) or new aggregated layout
    # (aggregated/<task>/classical_*/summary.json).
    for comparison_type, algo_patterns in [
        ("classical_matched", ["classical_q", "classical_expected_sarsa", "classical_vi"]),
        ("safe_zero", ["safe_q_zero", "safe_expected_sarsa_zero", "safe_vi_zero"]),
        ("safe_nonlinear", ["safe_q_stagewise", "safe_expected_sarsa_stagewise", "safe_vi"]),
    ]:
        old_path = translation_dir / comparison_type
        alt_path = translation_dir / f"{comparison_type}.json"
        agg_path = translation_dir / "aggregated"
        # New layout: any aggregated summary matching one of the algo_patterns
        new_layout_ok = agg_path.is_dir() and any(
            agg_path.glob(f"*/{pat}/summary.json")
            for pat in algo_patterns
        )
        exists = old_path.exists() or alt_path.exists() or new_layout_ok
        checks.append(_check(exists, f"Matched comparison: {comparison_type}",
                             f"Checked {old_path}, {alt_path}, or aggregated/{comparison_type} patterns"))

    # 4. Diagnostic sweep — check both old layout and separate sweep dirs
    sweep_exists = (
        (translation_dir / "diagnostic_sweep").exists()
        or (translation_dir / "diagnostic_sweep.json").exists()
        or any(translation_dir.glob("*sweep*"))
        or any(results_dir.glob(f"diagnostic_sweep{suffix}*"))
    ) if translation_dir.exists() else False
    checks.append(_check(sweep_exists, "Diagnostic-strength sweep results"))

    # 5. Translation analysis
    analysis_exists = (
        (translation_dir / "analysis" / "translation_analysis_summary.json").exists()
        or (translation_dir / "translation_analysis.json").exists()
        or (translation_dir / "translation_analysis").exists()
        or any(translation_dir.glob("*analysis*"))
    ) if translation_dir.exists() else False
    checks.append(_check(analysis_exists, "Translation analysis pipeline output"))

    # 6. Config — accept translation_study.json or translation_study_4a2.json
    config_ok = (
        (configs_dir / "translation_study.json").exists() or
        (configs_dir / f"translation_study{suffix}.json").exists()
    )
    checks.append(_check(config_ok, "Translation study config"))

    return checks


# ---------------------------------------------------------------------------
# IV-C: Completion gate (spec §14)
# ---------------------------------------------------------------------------

def check_gate_ivc(results_dir: Path, configs_dir: Path) -> list[dict]:
    """Phase IV-C completion gate.

    Conditions (from spec §14):
    1. Advanced estimator results exist.
    2. State-dependent scheduler comparison complete.
    3. Geometry-priority DP comparison complete.
    4. All ablation types run (trust-region, adaptive-headroom, wrong-sign,
       constant-u, raw-unclipped, trust-region-tighter, adaptive-headroom-aggressive).
    5. Attribution analysis complete.
    6. All configs exist.
    """
    checks: list[dict] = []
    advanced_dir = results_dir / "advanced"

    # 1. Advanced estimator results
    for algo in ["safe_double_q", "safe_target_q", "safe_target_expected_sarsa"]:
        path = advanced_dir / algo
        alt = advanced_dir / f"{algo}.json"
        exists = (path.exists() or alt.exists()) if advanced_dir.exists() else False
        checks.append(_check(exists, f"Advanced estimator results: {algo}"))

    # 2. State-dependent scheduler
    scheduler_exists = (
        (advanced_dir / "state_dependent_scheduler").exists()
        or any(advanced_dir.glob("*scheduler*"))
    ) if advanced_dir.exists() else False
    checks.append(_check(scheduler_exists, "State-dependent scheduler comparison"))

    # 3. Geometry-priority DP
    geo_exists = (
        (advanced_dir / "geometry_priority_dp").exists()
        or any(advanced_dir.glob("*geometry*"))
    ) if advanced_dir.exists() else False
    checks.append(_check(geo_exists, "Geometry-priority DP comparison"))

    # 4. Ablations
    ablation_types = [
        "trust_region_off", "trust_region_tighter",
        "adaptive_headroom_off", "adaptive_headroom_aggressive",
        "wrong_sign", "constant_u", "raw_unclipped",
    ]
    for ablation in ablation_types:
        path = advanced_dir / "ablations" / ablation
        alt = advanced_dir / f"ablation_{ablation}.json"
        exists = (path.exists() or alt.exists()) if advanced_dir.exists() else False
        checks.append(_check(exists, f"Ablation: {ablation}"))

    # 5. Attribution analysis
    attr_exists = (
        (advanced_dir / "attribution_analysis.json").exists()
        or (advanced_dir / "attribution_analysis").exists()
        or any(advanced_dir.glob("*attribution*"))
    ) if advanced_dir.exists() else False
    checks.append(_check(attr_exists, "Attribution analysis"))

    # 6. Configs
    for cfg in ["advanced_estimators", "state_dependent_schedulers",
                "geometry_priority_dp", "certification_ablations"]:
        checks.append(_file_exists(configs_dir / f"{cfg}.json", f"Config: {cfg}"))

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GATE_MAP = {
    "IV-A": check_gate_iva,
    "IV-B": check_gate_ivb,
    "IV-C": check_gate_ivc,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase IV gate checker")
    parser.add_argument("--phase", required=True, choices=["IV-A", "IV-B", "IV-C"])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/weighted_lse_dp/phase4"),
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("experiments/weighted_lse_dp/configs/phase4"),
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--suffix",
        default="",
        help="Artifact namespace suffix (e.g. '_4a2' for Phase IV-A2 artifacts).",
    )
    args = parser.parse_args()

    gate_fn = GATE_MAP[args.phase]
    import inspect
    if "suffix" in inspect.signature(gate_fn).parameters:
        checks = gate_fn(args.results_dir, args.configs_dir, suffix=args.suffix)
    else:
        checks = gate_fn(args.results_dir, args.configs_dir)

    # [INFO] checks are diagnostic and do not contribute to pass/fail.
    def _is_info(c: dict) -> bool:
        return c.get("condition", "").lstrip().startswith("[INFO]")

    gated = [c for c in checks if not _is_info(c)]
    passed = all(c["passed"] for c in gated)
    n_pass = sum(1 for c in gated if c["passed"])
    n_total = len(gated)

    # Eligibility payload (attached to the info check that wrote the report).
    eligibility_records: list[dict] = []
    for c in checks:
        if "_eligibility_records" in c:
            eligibility_records = c.pop("_eligibility_records")

    result = {
        "phase": args.phase,
        "result": "PASS" if passed else "FAIL",
        "passed": n_pass,
        "total": n_total,
        "checks": checks,
        "eligibility": eligibility_records if args.phase == "IV-A" else [],
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if passed else "FAIL"
        print(f"Gate {args.phase}: {status} ({n_pass}/{n_total} gated conditions met)")
        print()
        for c in checks:
            if _is_info(c):
                icon = "INFO"
            else:
                icon = "PASS" if c["passed"] else "FAIL"
            print(f"  [{icon}] {c['condition']}")
            # Print details for failures AND for [INFO] lines (the info is
            # the whole point of those rows).
            if c.get("details") and (not c["passed"] or _is_info(c)):
                print(f"         {c['details']}")

        if args.phase == "IV-A" and eligibility_records:
            print()
            print("Per-family eligibility report:")
            print()
            for rec in eligibility_records:
                dp = rec["design_point"]
                info = rec["informative_replay"]
                glob = rec["global_replay"]
                topq = rec.get("topquartile_replay", {})
                print(f"Family: {rec['family']}"
                      + (f"  (replay tag: {rec['tag']})" if rec.get("tag") else ""))
                print(
                    f"  Design-point:  mean_abs_u_pred={dp['mean_abs_u_pred']:.6f}"
                    f"  [{'PASS' if dp['gate_pass'] else 'FAIL'}]"
                )
                print(
                    f"  Informative:   mean={info['mean_abs_u']:.6f}"
                    f"  median={info['median_abs_u']:.6f}"
                    f"  frac_ge5e3={info['frac_u_ge_5e3']:.4f}"
                    f"  n_informative={info['n_informative']}"
                    f" ({100.0*info['frac_informative']:.2f}%)"
                    f"  [2a={'PASS' if info['gate_pass_mean_or_median'] else 'FAIL'},"
                    f" 2b={'PASS' if info['gate_pass_frac'] else 'FAIL'}]"
                )
                if topq:
                    print(
                        f"  TopQuartile:   mean={topq.get('mean_abs_u', 0.0):.6f}"
                        f"  frac_ge5e3={topq.get('frac_u_ge_5e3', 0.0):.4f}"
                        f"  n_topQ={topq.get('n_topquartile', 0)}  [fallback]"
                    )
                print(
                    f"  Global:        mean_abs_u={glob['mean_abs_u']:.6f}"
                    f"  [INFO - dilution diagnostic]"
                )
                print(
                    f"  IV-B Eligible: "
                    f"{'YES' if rec['iv_b_eligible'] else 'NO'}"
                    f"  ({rec['eligibility_reason']})"
                )
                print()

        if not passed:
            failed = [c for c in gated if not c["passed"]]
            print(f"\n{len(failed)} condition(s) not met. Fix before proceeding.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
