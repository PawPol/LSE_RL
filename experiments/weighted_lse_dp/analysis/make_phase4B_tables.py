#!/usr/bin/env python
"""Phase IV-B: generate tables P4B-A through P4B-F.

Tables produced (spec §12.3):
  P4B_A  Main performance comparison on the activation suite
  P4B_B  Operator-activation diagnostics for the same runs
  P4B_C  Matched classical vs safe-zero vs safe-nonlinear controls
  P4B_D  Diagnostic-strength sweep summary
  P4B_E  Translation-analysis summary
  P4B_F  Negative-control replay summary

Each table is written as both CSV and Markdown.

CLI
---
  python make_phase4B_tables.py --results-dir <dir> [--output-dir <dir>]

  --results-dir   Top-level Phase IV-B results directory
  --output-dir    Destination for tables.
                  Defaults to <results-dir>/analysis/.  (spec §Q12)

Algorithm name mapping follows spec §Q2:
  *_stagewise -> safe-nonlinear; *_zero -> safe-zero; classical_* -> classical
Runner subagent is responsible for matching directory names.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _fmt(val: object, precision: int = 4) -> str:
    """Format a scalar value for table output."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        if abs(val) < 1e-2 and val != 0.0:
            return f"{val:.{precision}e}"
        return f"{val:.{precision}f}"
    return str(val)


def _rows_to_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        warnings.warn(f"No rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _rows_to_md(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_table(rows: list[dict], out_dir: Path, name: str) -> None:
    _rows_to_csv(rows, out_dir / f"{name}.csv")
    _rows_to_md(rows, out_dir / f"{name}.md")
    print(f"  wrote {name}.csv  ({len(rows)} rows)")


def _algo_class(name: str) -> str:
    """Map algorithm dir name to canonical class label (spec §Q2)."""
    n = name.lower()
    if n.endswith("_stagewise"):
        return "safe-nonlinear"
    if n.endswith("_zero"):
        return "safe-zero"
    if n.startswith("classical"):
        return "classical"
    if n.startswith("safe"):
        return "safe-nonlinear"
    return "unknown"


# ---------------------------------------------------------------------------
# P4B-A: Main performance comparison on the activation suite
# ---------------------------------------------------------------------------

def _build_P4B_A(results_dir: Path) -> list[dict]:
    """P4B-A: Per-task/per-class mean return and primary outcome with CI."""
    trans_dir = results_dir / "translation"
    step3_path = results_dir / "analysis" / "step3_matched_control.json"
    step4_path = results_dir / "analysis" / "step4_outcome_interpretation.json"

    # Load step3 (paired CIs) and step4 (primary metric labels) if available
    step3: dict[str, Any] = {}
    step4: dict[str, Any] = {}
    if step3_path.exists():
        step3 = _load_json(step3_path)
    if step4_path.exists():
        step4 = _load_json(step4_path)

    rows: list[dict] = []
    if not trans_dir.is_dir():
        warnings.warn("No translation dir; P4B-A will be empty")
        return rows

    for task_dir in sorted(trans_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tag = task_dir.name
        primary_metric = step4.get(tag, {}).get("primary_metric", "mean_return")

        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            algo = algo_dir.name
            cls = _algo_class(algo)
            summary_path = algo_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                d = _load_json(summary_path)
            except Exception as exc:
                warnings.warn(f"Could not read {summary_path}: {exc}")
                continue

            mean_r = d.get("primary_outcome", d.get("mean_return", ""))
            std_r = d.get("std_return", d.get("std_primary_outcome", ""))
            n_seeds = d.get("n_seeds", "")

            rows.append({
                "task": tag,
                "algorithm": algo,
                "class": cls,
                "primary_metric": primary_metric,
                "mean": _fmt(mean_r),
                "std": _fmt(std_r),
                "n_seeds": n_seeds,
            })

    return rows


# ---------------------------------------------------------------------------
# P4B-B: Operator-activation diagnostics
# ---------------------------------------------------------------------------

def _build_P4B_B(results_dir: Path) -> list[dict]:
    """P4B-B: Per-task operator activation diagnostics from analysis step1."""
    step1_path = results_dir / "analysis" / "step1_activation_verification.json"
    if not step1_path.exists():
        warnings.warn("step1_activation_verification.json not found; P4B-B will be empty")
        return []

    step1 = _load_json(step1_path)
    rows: list[dict] = []
    for tag, info in sorted(step1.items()):
        if not isinstance(info, dict):
            continue
        checks = info.get("checks", {})
        row = {
            "task": tag,
            "activated": info.get("activated", ""),
            "classes_found": "|".join(sorted(info.get("classes_found", []))),
        }
        # Flatten check flags
        for k, v in checks.items():
            row[k] = _fmt(v)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# P4B-C: Matched classical vs safe-zero vs safe-nonlinear
# ---------------------------------------------------------------------------

def _build_P4B_C(results_dir: Path) -> list[dict]:
    """P4B-C: Effect decomposition from step3 matched-control analysis."""
    step3_path = results_dir / "analysis" / "step3_matched_control.json"
    if not step3_path.exists():
        warnings.warn("step3_matched_control.json not found; P4B-C will be empty")
        return []

    step3 = _load_json(step3_path)
    rows: list[dict] = []
    for tag, info in sorted(step3.items()):
        if not isinstance(info, dict):
            continue

        def _effect_cols(eff: dict, prefix: str) -> dict:
            return {
                f"{prefix}_mean_diff": _fmt(eff.get("mean_diff")),
                f"{prefix}_ci_lo": _fmt(eff.get("lower")),
                f"{prefix}_ci_hi": _fmt(eff.get("upper")),
                f"{prefix}_n_pairs": eff.get("n_pairs", ""),
            }

        row: dict = {"task": tag}
        row.update(_effect_cols(info.get("nonlinearity_effect", {}), "nonlinearity"))
        row.update(_effect_cols(info.get("total_effect", {}), "total"))
        row.update(_effect_cols(info.get("path_effect", {}), "path"))
        n_seeds = info.get("n_seeds", {})
        row["n_classical"] = n_seeds.get("classical", "")
        row["n_safe_zero"] = n_seeds.get("safe-zero", "")
        row["n_safe_nl"] = n_seeds.get("safe-nonlinear", "")
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# P4B-D: Diagnostic-strength sweep summary
# ---------------------------------------------------------------------------

def _build_P4B_D(results_dir: Path) -> list[dict]:
    """P4B-D: Spearman correlation and sweep stats from step2."""
    step2_path = results_dir / "analysis" / "step2_translation_sweep.json"
    if not step2_path.exists():
        warnings.warn("step2_translation_sweep.json not found; P4B-D will be empty")
        return []

    step2 = _load_json(step2_path)
    rows: list[dict] = []
    for tag, info in sorted(step2.items()):
        if not isinstance(info, dict):
            continue
        if info.get("status"):
            rows.append({
                "task": tag,
                "status": info["status"],
                "n_sweep_points": "",
                "spearman_rho": "",
                "spearman_p": "",
                "max_diag_delta": "",
                "max_outcome_delta": "",
            })
            continue

        diag = info.get("diag_deltas", [])
        outcome = info.get("outcome_deltas", [])
        rows.append({
            "task": tag,
            "status": "ok",
            "n_sweep_points": info.get("n_sweep_points", len(diag)),
            "spearman_rho": _fmt(info.get("spearman_rho")),
            "spearman_p": _fmt(info.get("spearman_p")),
            "max_diag_delta": _fmt(max(abs(x) for x in diag) if diag else None),
            "max_outcome_delta": _fmt(max(abs(x) for x in outcome) if outcome else None),
        })

    return rows


# ---------------------------------------------------------------------------
# P4B-E: Translation-analysis summary
# ---------------------------------------------------------------------------

def _build_P4B_E(results_dir: Path) -> list[dict]:
    """P4B-E: Top-level translation analysis summary and null cases."""
    summary_path = results_dir / "analysis" / "translation_analysis_summary.json"
    null_path = results_dir / "analysis" / "null_translation_cases.json"

    rows: list[dict] = []

    if summary_path.exists():
        try:
            summary = _load_json(summary_path)
            rows.append({"metric": k, "value": str(v)} for k, v in summary.items())  # type: ignore[assignment]
            # Flatten to list of dicts
            rows = [{"metric": k, "value": str(v)} for k, v in summary.items()]
        except Exception as exc:
            warnings.warn(f"Could not read translation_analysis_summary.json: {exc}")

    if null_path.exists():
        try:
            nulls = _load_json(null_path)
            for nc in nulls:
                rows.append({
                    "metric": f"null_case:{nc.get('tag', '?')}",
                    "value": nc.get("interpretation", ""),
                })
        except Exception:
            pass

    if not rows:
        warnings.warn("No translation summary data; P4B-E will be empty")
    return rows


# ---------------------------------------------------------------------------
# P4B-F: Negative-control replay summary
# ---------------------------------------------------------------------------

def _build_P4B_F(results_dir: Path) -> list[dict]:
    """P4B-F: Per-task negative-control verification from step5.

    Uses counterfactual_replay/ (NOT counterfactual_replay_4a2/) (spec §Q8).
    """
    step5_path = results_dir / "analysis" / "step5_negative_control.json"
    if not step5_path.exists():
        # Fall back to reading replay summaries directly
        neg_dir = results_dir / "counterfactual_replay"
        if not neg_dir.is_dir():
            warnings.warn("No negative-control data found; P4B-F will be empty")
            return []
        rows: list[dict] = []
        for task_dir in sorted(neg_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            sp = task_dir / "replay_summary.json"
            if not sp.exists():
                continue
            try:
                d = _load_json(sp)
                rows.append({
                    "task": task_dir.name,
                    "mean_abs_u": _fmt(d.get("mean_abs_u")),
                    "frac_active": _fmt(d.get("frac_u_ge_5e3")),
                    "mean_abs_delta_d": _fmt(d.get("mean_abs_delta_d")),
                    "near_zero": "",
                })
            except Exception:
                pass
        return rows

    try:
        s5 = _load_json(step5_path)
    except Exception as exc:
        warnings.warn(f"Could not read step5_negative_control.json: {exc}")
        return []

    per_task = s5.get("per_task", {})
    rows = []
    for tag in sorted(per_task.keys()):
        info = per_task[tag]
        rows.append({
            "task": tag,
            "mean_abs_u": _fmt(info.get("mean_abs_u")),
            "frac_active": _fmt(info.get("frac_active")),
            "mean_abs_delta_d": _fmt(info.get("mean_abs_delta_d")),
            "near_zero": str(info.get("near_zero", "")),
        })

    # Append summary row
    rows.append({
        "task": "__summary__",
        "mean_abs_u": "",
        "frac_active": "",
        "mean_abs_delta_d": "",
        "near_zero": f"all_near_zero={s5.get('all_near_zero', '')}",
    })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Generate Phase IV-B tables."""
    results_dir = args.results_dir.resolve()
    # Default output-dir = <results-dir>/analysis/ (spec §Q12)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase IV-B table generation")
    print(f"  results_dir: {results_dir}")
    print(f"  output_dir:  {output_dir}")
    print()

    tables = {
        "P4B_A": _build_P4B_A(results_dir),
        "P4B_B": _build_P4B_B(results_dir),
        "P4B_C": _build_P4B_C(results_dir),
        "P4B_D": _build_P4B_D(results_dir),
        "P4B_E": _build_P4B_E(results_dir),
        "P4B_F": _build_P4B_F(results_dir),
    }

    for name, rows in tables.items():
        _write_table(rows, output_dir, name)

    total_rows = sum(len(r) for r in tables.values())
    print(f"\nDone. {total_rows} total rows across {len(tables)} tables.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", type=Path, required=True,
        help="Top-level Phase IV-B results directory"
    )
    # No --config (spec §Q1: tables read data directly)
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for tables; defaults to <results-dir>/analysis/"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
