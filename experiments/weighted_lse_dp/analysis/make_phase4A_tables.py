#!/usr/bin/env python
"""Phase IV-A: generate tables P4A-A through P4A-E.

Reads aggregated Phase IV-A results and produces CSV and Markdown tables
for the activation search analysis.

Tables produced:
  P4A-A  Activation-suite task definitions and pilot activation diagnostics
  P4A-B  Operator-activation diagnostics by task
  P4A-C  Matched classical-control configuration summary
  P4A-D  Negative-control replay summary (Phase III tasks)
  P4A-E  Counterfactual replay summary
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import warnings
from pathlib import Path

# Allow running from the analysis directory directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _fmt(val: object, precision: int = 6) -> str:
    """Format a value for display: floats get scientific notation when tiny."""
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
        warnings.warn(f"No rows to write for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _rows_to_md(rows: list[dict], path: Path) -> None:
    if not rows:
        warnings.warn(f"No rows to write for {path.name}")
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


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _build_P4A_A(input_dir: Path, project_root: Path) -> list[dict]:
    """P4A-A: Activation-suite task definitions and pilot activation diagnostics."""
    scores_path = project_root / "results/weighted_lse_dp/phase4/task_search/candidate_scores.csv"
    selected_path = project_root / "results/weighted_lse_dp/phase4/task_search/selected_tasks.json"

    selected_tags: set[str] = set()
    if selected_path.exists():
        sel = _load_json(selected_path)
        for t in sel:
            selected_tags.add(t.get("tag", t.get("cfg", {}).get("family", "") + "_" + str(t.get("idx", ""))))

    rows: list[dict] = []
    if not scores_path.exists():
        warnings.warn(f"candidate_scores.csv not found at {scores_path}")
        return rows

    with open(scores_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            family = r.get("family", "")
            idx = r.get("idx", "")
            rows.append({
                "idx": idx,
                "family": family,
                "mean_abs_u_pred": _fmt(float(r["mean_abs_u_pred"])) if r.get("mean_abs_u_pred") else "",
                "frac_u_ge_5e3": _fmt(float(r["frac_u_ge_5e3"])) if r.get("frac_u_ge_5e3") else "",
                "informative_stage_frac": _fmt(float(r["informative_stage_frac"])) if r.get("informative_stage_frac") else "",
                "total_score": _fmt(float(r["total_score"])) if r.get("total_score") else "",
                "selected": "yes" if any(s.get("idx") == int(idx) for s in (_load_json(selected_path) if selected_path.exists() else [])) else "no",
            })
    return rows


def _build_P4A_B(input_dir: Path) -> list[dict]:
    """P4A-B: Operator-activation diagnostics by task."""
    diag_path = input_dir / "activation_diagnostics.json"
    gate_path = input_dir / "gate_evaluation.json"

    if not diag_path.exists():
        warnings.warn(f"activation_diagnostics.json not found at {diag_path}")
        return []

    diags = _load_json(diag_path)
    gate_map: dict[str, bool] = {}
    if gate_path.exists():
        for g in _load_json(gate_path):
            gate_map[g["tag"]] = g["global_gate_pass"]

    rows: list[dict] = []
    for d in diags:
        tag = d["tag"]
        rows.append({
            "task_id": tag,
            "family": d["family"],
            "mean_abs_u": _fmt(d["mean_abs_u"]),
            "frac_u_ge_5e3": _fmt(d["frac_u_ge_5e3"]),
            "mean_abs_delta_d": _fmt(d["mean_abs_delta_d"]),
            "frac_delta_d_ge_1e3": _fmt(d.get("frac_delta_d_ge_1e3", "")),
            "mean_abs_target_gap_normed": _fmt(d.get("mean_abs_target_gap", d.get("mean_abs_target_gap_normed", ""))),
            "mean_beta_used": _fmt(d.get("mean_beta_used", "")),
            "mean_KL_to_prior": _fmt(d.get("mean_KL_to_prior", "")),
            "gate_pass": gate_map.get(tag, ""),
        })
    return rows


def _build_P4A_C(project_root: Path) -> list[dict]:
    """P4A-C: Matched classical-control configuration summary.

    Reads from configs/phase4/gamma_matched_controls.json if it exists.
    Otherwise, generates a placeholder from the selected tasks.
    """
    cfg_path = project_root / "configs/phase4/gamma_matched_controls.json"
    if cfg_path.exists():
        data = _load_json(cfg_path)
        rows: list[dict] = []
        for entry in data:
            rows.append({
                "task_family": entry.get("task_family", ""),
                "gamma_eval": _fmt(entry.get("gamma_eval", "")),
                "gamma_base": _fmt(entry.get("gamma_base", "")),
                "control_type": entry.get("control_type", ""),
                "notes": entry.get("notes", ""),
            })
        return rows

    # Fallback: derive from selected tasks
    sel_path = project_root / "results/weighted_lse_dp/phase4/task_search/selected_tasks.json"
    if not sel_path.exists():
        warnings.warn("No gamma_matched_controls.json or selected_tasks.json found for P4A-C")
        return []

    sel = _load_json(sel_path)
    rows = []
    for t in sel:
        cfg = t.get("cfg", {})
        rows.append({
            "task_family": cfg.get("family", ""),
            "gamma_eval": _fmt(cfg.get("gamma", "")),
            "gamma_base": _fmt(t.get("schedule_summary", {}).get("gamma_base", "")),
            "control_type": "classical_dp",
            "notes": "auto-derived from selected_tasks.json (no gamma_matched_controls.json found)",
        })
    return rows


def _build_P4A_D(project_root: Path) -> list[dict]:
    """P4A-D: Negative-control replay summary (Phase III tasks)."""
    audit_path = project_root / "results/weighted_lse_dp/phase4/audit/phase3_result_audit.json"
    if not audit_path.exists():
        warnings.warn(f"phase3_result_audit.json not found at {audit_path}")
        return []

    audit = _load_json(audit_path)

    # The audit file is a top-level dict with metadata.
    # Check for per-task replay data in the smoke directory.
    smoke_dir = project_root / "results/weighted_lse_dp/phase4/audit/phase3_replay_smoke"
    rows: list[dict] = []

    if smoke_dir.is_dir():
        for p in sorted(smoke_dir.iterdir()):
            if p.suffix == ".json":
                try:
                    d = _load_json(p)
                    rows.append({
                        "task": p.stem,
                        "beta_used_mean": _fmt(d.get("mean_beta_used", d.get("beta_used_mean", ""))),
                        "natural_shift_mean": _fmt(d.get("mean_natural_shift", d.get("natural_shift_mean", ""))),
                        "gate_pass": d.get("gate_pass", d.get("global_gate_pass", "")),
                        "classification": d.get("classification", "negative_control"),
                    })
                except Exception as e:
                    warnings.warn(f"Error reading {p}: {e}")

    if not rows:
        # Provide a summary row from the audit metadata
        rows.append({
            "task": "ALL_PHASE3",
            "beta_used_mean": "",
            "natural_shift_mean": "",
            "gate_pass": "",
            "classification": f"audit found {audit.get('result_count', '?')} run artifacts across {len(audit.get('result_dirs_found', []))} task families",
        })
    return rows


def _build_P4A_E(input_dir: Path) -> list[dict]:
    """P4A-E: Counterfactual replay summary."""
    diag_path = input_dir / "activation_diagnostics.json"
    if not diag_path.exists():
        warnings.warn(f"activation_diagnostics.json not found at {diag_path}")
        return []

    diags = _load_json(diag_path)
    rows: list[dict] = []
    for d in diags:
        mean_abs_u = d["mean_abs_u"]
        threshold = 5e-3
        classification = "active" if mean_abs_u >= threshold else "inactive"
        rows.append({
            "task_id": d["tag"],
            "family": d["family"],
            "n_transitions": d["n_transitions"],
            "mean_abs_u": _fmt(mean_abs_u),
            "frac_u_ge_5e3": _fmt(d["frac_u_ge_5e3"]),
            "mean_abs_delta_d": _fmt(d["mean_abs_delta_d"]),
            "mean_abs_tg_normed": _fmt(d.get("mean_abs_target_gap", "")),
            "classification": classification,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Generate Phase IV-A tables."""
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[3]

    print(f"Phase IV-A table generation")
    print(f"  input_dir:  {input_dir}")
    print(f"  output_dir: {output_dir}")
    print()

    tables = {
        "P4A_A": _build_P4A_A(input_dir, project_root),
        "P4A_B": _build_P4A_B(input_dir),
        "P4A_C": _build_P4A_C(project_root),
        "P4A_D": _build_P4A_D(project_root),
        "P4A_E": _build_P4A_E(input_dir),
    }

    for name, rows in tables.items():
        _write_table(rows, output_dir, name)

    print(f"\nDone. {sum(len(r) for r in tables.values())} total rows across {len(tables)} tables.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True,
                    help="Path to results/processed/phase4A/")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Path to output directory for tables")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
