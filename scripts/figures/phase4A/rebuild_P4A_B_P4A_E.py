#!/usr/bin/env python
"""Rebuild Phase IV-A paper tables P4A-B and P4A-E post-MAJOR 6/8/9 calibration fixes.

After the three calibration fixes landed on 2026-04-22
(cap-binding argmin, adaptive-headroom feasibility check, trust-region
Taylor floor), 146 `schedule_v3.json` artifacts under
`results/weighted_lse_dp/phase4/` were regenerated.  Each regenerated
file has a `.pre_major689.json` sidecar preserving the pre-fix state.

The canonical `P4A_B` (operator-activation diagnostics by task) and
`P4A_E` (counterfactual replay summary) tables are produced by
`experiments/weighted_lse_dp/analysis/make_phase4A_tables.py` from
`activation_diagnostics.json` / `gate_evaluation.json` /
`replay_summary.json` and were unaffected by the calibration fix.
This script emits two supplementary tables that carry the
cap-binding / feasibility information the paper needs, under
distinct filenames:

  * P4A_B_cap_binding: cap-binding breakdown table — counts and
    fractions of `target_binding` / `trust_clip` / `safe_clip` stages,
    broken down by task_id and algorithm, aggregated over seeds (and
    geometry_priority_dp variants where present).
  * P4A_E_headroom: adaptive-headroom fixed-point outputs — mean
    per-stage `U_safe_ref`, `u_target`, `u_ref_used`, cap-utilization
    ratio, feasibility-bump rate (fraction of stages with
    `alpha_t > alpha_max = 0.20`), and alpha_budget-saturation rate
    (fraction of stages with `alpha_t >= alpha_budget_max = 0.30`).

Both supplementary tables include a parallel `_pre_major689` summary
so the paper can cite the before→after change produced by the MAJOR
6/8/9 fixes.

Inputs
------
`results/weighted_lse_dp/phase4/**/schedule_v3.json`
  Post-fix schedules (canonical input, 146 files).

`results/weighted_lse_dp/phase4/**/schedule_v3.pre_major689.json`
  Pre-fix sidecars (146 files) used for the before→after diff.

Outputs
-------
`results/processed/phase4A/tables/P4A_B_cap_binding.{csv,md}`
`results/processed/phase4A/tables/P4A_E_headroom.{csv,md}`

The spec-§12.2 canonical ``P4A_B.*`` (operator-activation diagnostics
by task) and ``P4A_E.*`` (counterfactual replay summary) filenames are
reserved for the output of
``experiments/weighted_lse_dp/analysis/make_phase4A_tables.py`` and are
NOT written by this script.  The cap-binding breakdown and
adaptive-headroom outputs produced here are supplementary paper
tables living alongside them.

Notes
-----
The aggregation operates directly on the schedule JSON files.  No
file under `results/raw/` is read and no processed file elsewhere in
`results/processed/phase4A/` is overwritten — only the two
supplementary `*_cap_binding` / `*_headroom` tables.  The script is
idempotent: running it twice produces byte-identical tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/liq/Documents/Claude/Projects/LSE_RL")
SCHEDULE_ROOT = PROJECT_ROOT / "results" / "weighted_lse_dp" / "phase4"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "phase4A" / "tables"

# Calibration-v3 defaults (see docs/specs/phase_IV_A_activation_audit_and_counterfactual.md §6.6 / §6.7)
ALPHA_MIN = 0.05
ALPHA_MAX = 0.20
ALPHA_BUDGET_MAX = 0.30
TOL = 1e-10  # must match the argmin tolerance used by the regenerator


# ---------------------------------------------------------------------------
# Classification of a schedule file's provenance
# ---------------------------------------------------------------------------

def classify_path(rel: Path) -> dict[str, str]:
    """Map a path relative to SCHEDULE_ROOT to (suite, algorithm, task_id, variant, seed).

    The phase-4 subtree uses a few different layouts:
      advanced/<algo>/<task>/seed_<N>/schedule_v3.json
      advanced/geometry_priority_dp/<task>/seed_<N>/<variant>/schedule_v3.json
      diagnostic_sweep_4a2_umax_<x>/<task>/<algo>/seed_<N>/schedule_v3.json
      translation_4a2/<task>/<algo>/seed_<N>/schedule_v3.json

    Returns a dict with keys: suite, algorithm, task_id, variant ('' if none),
    seed, and ``algo_key`` which is ``algo`` (or ``algo/variant`` for
    geometry_priority_dp) used as the algorithm-breakdown key.
    """
    parts = rel.parts
    assert parts[-1] == "schedule_v3.json"
    suite = parts[0]
    if suite == "advanced":
        algo = parts[1]
        task_id = parts[2]
        seed = parts[3]
        if algo == "geometry_priority_dp":
            variant = parts[4]
            algo_key = f"{algo}/{variant}"
        else:
            variant = ""
            algo_key = algo
    elif suite.startswith("diagnostic_sweep") or suite == "translation_4a2":
        task_id = parts[1]
        algo = parts[2]
        seed = parts[3]
        variant = ""
        # For the diagnostic sweep we keep u_max in the suite column so the
        # reader can tell the four sweeps apart without digging into configs.
        algo_key = algo
    else:
        # Defensive fallback.  The script is run over 146 known files so this
        # branch should never fire in practice.
        task_id = parts[1] if len(parts) > 2 else ""
        algo = parts[2] if len(parts) > 3 else ""
        seed = ""
        variant = ""
        algo_key = algo
    return dict(
        suite=suite,
        algorithm=algo_key,
        task_id=task_id,
        variant=variant,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Per-file aggregates
# ---------------------------------------------------------------------------

def schedule_stats(path: Path) -> dict[str, Any]:
    """Compute the per-schedule statistics that feed P4A-B and P4A-E.

    Returns a dict with:
      n_stages, n_trust_clip, n_safe_clip, n_target_binding,
      u_target_sum, u_ref_used_sum, U_safe_ref_sum, u_tr_cap_sum,
      cap_util_sum (u_ref_used / U_safe_ref, sum over stages with U_safe_ref > 0),
      cap_util_n (count of stages in the cap_util denominator),
      n_feasibility_bumped (alpha_t > ALPHA_MAX + TOL),
      n_budget_saturated   (alpha_t >= ALPHA_BUDGET_MAX - TOL).
    """
    with open(path) as f:
        sched = json.load(f)
    trust = np.asarray(sched["trust_clip_active_t"], dtype=bool)
    safe = np.asarray(sched["safe_clip_active_t"], dtype=bool)
    n = len(trust)
    # target_binding = neither cap binds (post-fix flags partition the stages).
    target_binding = ~(trust | safe)

    alpha = np.asarray(sched["alpha_t"], dtype=float)
    u_target = np.asarray(sched["u_target_t"], dtype=float)
    u_ref = np.asarray(sched["u_ref_used_t"], dtype=float)
    U_safe = np.abs(np.asarray(sched["U_safe_ref_t"], dtype=float))
    u_tr = np.asarray(sched["u_tr_cap_t"], dtype=float)

    cap_util_mask = U_safe > 1e-12
    cap_util_vals = np.where(cap_util_mask, u_ref / np.maximum(U_safe, 1e-18), 0.0)

    return dict(
        n_stages=int(n),
        n_trust_clip=int(trust.sum()),
        n_safe_clip=int(safe.sum()),
        n_target_binding=int(target_binding.sum()),
        u_target_sum=float(u_target.sum()),
        u_ref_used_sum=float(u_ref.sum()),
        U_safe_ref_sum=float(U_safe.sum()),
        u_tr_cap_sum=float(u_tr.sum()),
        cap_util_sum=float(cap_util_vals[cap_util_mask].sum()),
        cap_util_n=int(cap_util_mask.sum()),
        n_feasibility_bumped=int((alpha > ALPHA_MAX + TOL).sum()),
        n_budget_saturated=int((alpha >= ALPHA_BUDGET_MAX - TOL).sum()),
    )


def aggregate(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum-aggregate per-file stats into per-group totals."""
    agg = dict(
        n_files=len(stats_list),
        n_stages=0,
        n_trust_clip=0,
        n_safe_clip=0,
        n_target_binding=0,
        u_target_sum=0.0,
        u_ref_used_sum=0.0,
        U_safe_ref_sum=0.0,
        u_tr_cap_sum=0.0,
        cap_util_sum=0.0,
        cap_util_n=0,
        n_feasibility_bumped=0,
        n_budget_saturated=0,
    )
    for s in stats_list:
        for k in agg:
            if k == "n_files":
                continue
            agg[k] += s[k]
    return agg


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_breakdown_rows(
    per_file_stats: list[tuple[dict[str, str], dict[str, Any]]],
    label: str,
) -> list[dict[str, Any]]:
    """Build P4A-B rows.

    ``per_file_stats`` is a list of (provenance, stats) pairs (provenance
    from :func:`classify_path`).  We group by (suite, task_id, algorithm)
    and report counts, fractions, and the number of seeds folded in.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for prov, st in per_file_stats:
        key = (prov["suite"], prov["task_id"], prov["algorithm"])
        groups[key].append(st)

    rows: list[dict[str, Any]] = []
    for (suite, task_id, algo), stats in sorted(groups.items()):
        agg = aggregate(stats)
        n_stages = max(agg["n_stages"], 1)
        rows.append(
            {
                "version": label,
                "suite": suite,
                "task_id": task_id,
                "algorithm": algo,
                "n_files": agg["n_files"],
                "n_stages_total": agg["n_stages"],
                "n_target_binding": agg["n_target_binding"],
                "n_trust_clip": agg["n_trust_clip"],
                "n_safe_clip": agg["n_safe_clip"],
                "frac_target_binding": agg["n_target_binding"] / n_stages,
                "frac_trust_clip": agg["n_trust_clip"] / n_stages,
                "frac_safe_clip": agg["n_safe_clip"] / n_stages,
            }
        )
    # Global summary row: sum across all groups.
    all_stats = [s for _, s in per_file_stats]
    agg_all = aggregate(all_stats)
    n_stages_all = max(agg_all["n_stages"], 1)
    rows.append(
        {
            "version": label,
            "suite": "ALL",
            "task_id": "ALL",
            "algorithm": "ALL",
            "n_files": agg_all["n_files"],
            "n_stages_total": agg_all["n_stages"],
            "n_target_binding": agg_all["n_target_binding"],
            "n_trust_clip": agg_all["n_trust_clip"],
            "n_safe_clip": agg_all["n_safe_clip"],
            "frac_target_binding": agg_all["n_target_binding"] / n_stages_all,
            "frac_trust_clip": agg_all["n_trust_clip"] / n_stages_all,
            "frac_safe_clip": agg_all["n_safe_clip"] / n_stages_all,
        }
    )
    return rows


def build_headroom_rows(
    per_file_stats: list[tuple[dict[str, str], dict[str, Any]]],
    label: str,
) -> list[dict[str, Any]]:
    """Build P4A-E rows: adaptive-headroom fixed-point outputs."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for prov, st in per_file_stats:
        key = (prov["suite"], prov["task_id"], prov["algorithm"])
        groups[key].append(st)

    rows: list[dict[str, Any]] = []
    for (suite, task_id, algo), stats in sorted(groups.items()):
        agg = aggregate(stats)
        n = max(agg["n_stages"], 1)
        cap_util_denom = max(agg["cap_util_n"], 1)
        rows.append(
            {
                "version": label,
                "suite": suite,
                "task_id": task_id,
                "algorithm": algo,
                "n_files": agg["n_files"],
                "n_stages_total": agg["n_stages"],
                "mean_u_target": agg["u_target_sum"] / n,
                "mean_u_ref_used": agg["u_ref_used_sum"] / n,
                "mean_U_safe_ref": agg["U_safe_ref_sum"] / n,
                "mean_u_tr_cap": agg["u_tr_cap_sum"] / n,
                "mean_cap_util_ratio": agg["cap_util_sum"] / cap_util_denom,
                "n_feasibility_bumped": agg["n_feasibility_bumped"],
                "frac_feasibility_bumped": agg["n_feasibility_bumped"] / n,
                "n_budget_saturated": agg["n_budget_saturated"],
                "frac_budget_saturated": agg["n_budget_saturated"] / n,
            }
        )
    all_stats = [s for _, s in per_file_stats]
    agg_all = aggregate(all_stats)
    n_all = max(agg_all["n_stages"], 1)
    cap_util_denom_all = max(agg_all["cap_util_n"], 1)
    rows.append(
        {
            "version": label,
            "suite": "ALL",
            "task_id": "ALL",
            "algorithm": "ALL",
            "n_files": agg_all["n_files"],
            "n_stages_total": agg_all["n_stages"],
            "mean_u_target": agg_all["u_target_sum"] / n_all,
            "mean_u_ref_used": agg_all["u_ref_used_sum"] / n_all,
            "mean_U_safe_ref": agg_all["U_safe_ref_sum"] / n_all,
            "mean_u_tr_cap": agg_all["u_tr_cap_sum"] / n_all,
            "mean_cap_util_ratio": agg_all["cap_util_sum"] / cap_util_denom_all,
            "n_feasibility_bumped": agg_all["n_feasibility_bumped"],
            "frac_feasibility_bumped": agg_all["n_feasibility_bumped"] / n_all,
            "n_budget_saturated": agg_all["n_budget_saturated"],
            "frac_budget_saturated": agg_all["n_budget_saturated"] / n_all,
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def fmt_val(v: Any, precision: int = 6) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v == 0.0:
            return "0.000000"
        if abs(v) < 1e-2:
            return f"{v:.{precision}e}"
        return f"{v:.{precision}f}"
    return str(v)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt_val(r[c]) for c in cols})


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        lines.append("| " + " | ".join(fmt_val(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def gather(
    schedule_root: Path, sidecar: bool = False
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    glob = "**/schedule_v3.pre_major689.json" if sidecar else "**/schedule_v3.json"
    out: list[tuple[dict[str, str], dict[str, Any]]] = []
    for path in sorted(schedule_root.rglob(glob)):
        rel_to_root = path.relative_to(schedule_root)
        # Normalise the relative path so both the post-fix and the pre-fix
        # variants classify to the same (suite, algorithm, task_id, variant, seed).
        rel_norm = Path(str(rel_to_root).replace(".pre_major689", ""))
        prov = classify_path(rel_norm)
        stats = schedule_stats(path)
        out.append((prov, stats))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=SCHEDULE_ROOT,
        help="Root under which schedule_v3.json files live",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for rebuilt P4A_B and P4A_E tables",
    )
    args = parser.parse_args()

    post_stats = gather(args.schedule_root, sidecar=False)
    pre_stats = gather(args.schedule_root, sidecar=True)

    if len(post_stats) != len(pre_stats):
        raise RuntimeError(
            f"post/pre count mismatch: {len(post_stats)} post vs {len(pre_stats)} pre"
        )

    p4a_b_rows = build_breakdown_rows(post_stats, "post_major689")
    p4a_b_rows += build_breakdown_rows(pre_stats, "pre_major689")

    p4a_e_rows = build_headroom_rows(post_stats, "post_major689")
    p4a_e_rows += build_headroom_rows(pre_stats, "pre_major689")

    write_csv(p4a_b_rows, args.output_dir / "P4A_B_cap_binding.csv")
    write_md(p4a_b_rows, args.output_dir / "P4A_B_cap_binding.md")
    write_csv(p4a_e_rows, args.output_dir / "P4A_E_headroom.csv")
    write_md(p4a_e_rows, args.output_dir / "P4A_E_headroom.md")

    # Human-readable headline
    all_post = [r for r in p4a_b_rows if r["version"] == "post_major689" and r["task_id"] == "ALL"][0]
    all_pre = [r for r in p4a_b_rows if r["version"] == "pre_major689" and r["task_id"] == "ALL"][0]
    print(
        "[P4A-B] pre → post trust_clip frac:  "
        f"{all_pre['frac_trust_clip']:.4f} → {all_post['frac_trust_clip']:.4f}"
    )
    print(
        "[P4A-B] pre → post safe_clip  frac:  "
        f"{all_pre['frac_safe_clip']:.4f} → {all_post['frac_safe_clip']:.4f}"
    )
    print(
        "[P4A-B] pre → post target_binding:   "
        f"{all_pre['frac_target_binding']:.4f} → {all_post['frac_target_binding']:.4f}"
    )
    all_post_e = [r for r in p4a_e_rows if r["version"] == "post_major689" and r["task_id"] == "ALL"][0]
    all_pre_e = [r for r in p4a_e_rows if r["version"] == "pre_major689" and r["task_id"] == "ALL"][0]
    print(
        "[P4A-E] pre → post frac_feasibility_bumped:  "
        f"{all_pre_e['frac_feasibility_bumped']:.4f} → {all_post_e['frac_feasibility_bumped']:.4f}"
    )
    print(
        "[P4A-E] pre → post mean_cap_util_ratio:      "
        f"{all_pre_e['mean_cap_util_ratio']:.4f} → {all_post_e['mean_cap_util_ratio']:.4f}"
    )
    print(f"Wrote {args.output_dir / 'P4A_B_cap_binding.csv'}")
    print(f"Wrote {args.output_dir / 'P4A_B_cap_binding.md'}")
    print(f"Wrote {args.output_dir / 'P4A_E_headroom.csv'}")
    print(f"Wrote {args.output_dir / 'P4A_E_headroom.md'}")


if __name__ == "__main__":
    main()
