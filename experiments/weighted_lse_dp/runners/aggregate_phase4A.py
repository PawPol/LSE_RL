#!/usr/bin/env python
"""Phase IV-A: aggregate activation search, counterfactual replay, and audit results.

Merges per-task activation metrics, counterfactual replay deltas, and
search scores into a single summary dataset for downstream analysis
and figure generation.

Outputs (all under --output-dir):
  activation_diagnostics.json   per-task aggregate stats from replay
  search_results_summary.json   per-family summary of search scores
  gate_evaluation.json          activation gate pass/fail per task
  phase4A_summary.json          top-level counts and status
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def _load_replay_summaries(replay_dir: Path) -> list[dict[str, Any]]:
    """Load all replay_summary.json files from subdirectories."""
    summaries: list[dict[str, Any]] = []
    if not replay_dir.is_dir():
        logger.warning("Replay directory does not exist: %s", replay_dir)
        return summaries

    for child in sorted(replay_dir.iterdir()):
        summary_path = child / "replay_summary.json"
        if child.is_dir() and summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            data["source_dir"] = str(child.name)
            summaries.append(data)

    logger.info("Loaded %d replay summaries from %s", len(summaries), replay_dir)
    return summaries


def _load_search_scores(search_dir: Path) -> list[dict[str, Any]]:
    """Load candidate_scores.csv from the search directory.

    Returns a list of dicts (one per row). If the file is missing,
    returns an empty list and logs a warning.
    """
    csv_path = search_dir / "candidate_scores.csv"
    if not csv_path.exists():
        logger.warning(
            "candidate_scores.csv not found at %s; "
            "search_results_summary will be empty.",
            csv_path,
        )
        return []

    import csv
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)

    logger.info("Loaded %d search score rows from %s", len(rows), csv_path)
    return rows


# --------------------------------------------------------------------------
# Gate evaluation
# --------------------------------------------------------------------------

def _evaluate_gate(
    summary: dict[str, Any],
    min_mean_abs_u: float = 5e-3,
    min_frac_active: float = 0.10,
    min_mean_abs_delta_d: float = 1e-3,
    min_mean_abs_target_gap_normed: float = 5e-3,
) -> dict[str, Any]:
    """Evaluate activation gate for a single task.

    Returns a dict with individual check results and the global pass flag.
    """
    r_max = max(float(summary.get("r_max", 1.0)), 1e-8)
    mean_abs_u = float(summary.get("mean_abs_u", 0.0))
    frac_active = float(summary.get("frac_u_ge_5e3", 0.0))
    mean_abs_dd = float(summary.get("mean_abs_delta_d", 0.0))
    mean_abs_tg = float(summary.get("mean_abs_target_gap", 0.0))
    normalized_tg = mean_abs_tg / r_max

    checks = {
        "mean_abs_u_ge_5e3": mean_abs_u >= min_mean_abs_u,
        "frac_abs_u_ge_5e3_ge_0.10": frac_active >= min_frac_active,
        "mean_abs_delta_d_ge_1e3": mean_abs_dd >= min_mean_abs_delta_d,
        "mean_abs_target_gap_normed_ge_5e3": normalized_tg >= min_mean_abs_target_gap_normed,
    }

    return {
        "family": summary.get("family", "unknown"),
        "tag": summary.get("tag", "unknown"),
        "global_gate_pass": all(checks.values()),
        "individual_checks": checks,
        "values": {
            "mean_abs_u": mean_abs_u,
            "frac_active": frac_active,
            "mean_abs_delta_d": mean_abs_dd,
            "normalized_mean_abs_target_gap": normalized_tg,
        },
    }


# --------------------------------------------------------------------------
# Search results summary
# --------------------------------------------------------------------------

def _build_search_summary(
    search_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build per-family summary from search score rows."""
    from collections import defaultdict

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in search_rows:
        family = str(row.get("family", "unknown"))
        by_family[family].append(row)

    family_summaries: dict[str, Any] = {}
    for family, rows in sorted(by_family.items()):
        scores = [float(r.get("total_score", 0.0)) for r in rows]
        family_summaries[family] = {
            "n_candidates": len(rows),
            "mean_score": sum(scores) / max(len(scores), 1),
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
        }

    return {
        "n_total_candidates": len(search_rows),
        "n_families": len(by_family),
        "per_family": family_summaries,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Run Phase IV-A aggregation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    replay_dir = Path(args.replay_dir)
    search_dir = Path(args.search_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load replay summaries
    replay_summaries = _load_replay_summaries(replay_dir)

    # 2. Load search scores
    search_rows = _load_search_scores(search_dir)

    # 3. Activation diagnostics (per-task from replay)
    activation_diagnostics: list[dict[str, Any]] = []
    for s in replay_summaries:
        entry: dict[str, Any] = {
            "family": s.get("family", "unknown"),
            "tag": s.get("tag", "unknown"),
            "n_transitions": s.get("n_transitions", 0),
            "gamma_base": s.get("gamma_base"),
            "r_max": s.get("r_max"),
        }
        # Copy all diagnostic keys
        for k in [
            "mean_natural_shift", "std_natural_shift",
            "p25_natural_shift", "p75_natural_shift",
            "mean_abs_u",
            "mean_delta_effective_discount", "std_delta_effective_discount",
            "p25_delta_effective_discount", "p75_delta_effective_discount",
            "mean_abs_delta_d",
            "mean_target_gap_same_gamma_base", "std_target_gap_same_gamma_base",
            "p25_target_gap_same_gamma_base", "p75_target_gap_same_gamma_base",
            "mean_abs_target_gap",
            "frac_u_ge_5e3", "frac_delta_d_ge_1e3",
            "frac_target_gap_ge_5e3_normed",
            "mean_beta_used", "mean_KL_to_prior",
        ]:
            if k in s:
                entry[k] = s[k]
        activation_diagnostics.append(entry)

    with open(output_dir / "activation_diagnostics.json", "w") as f:
        json.dump(activation_diagnostics, f, indent=2)
    logger.info(
        "Wrote activation_diagnostics.json with %d tasks.",
        len(activation_diagnostics),
    )

    # 4. Search results summary
    search_summary = _build_search_summary(search_rows)
    with open(output_dir / "search_results_summary.json", "w") as f:
        json.dump(search_summary, f, indent=2)
    logger.info("Wrote search_results_summary.json.")

    # 5. Gate evaluation
    gate_results: list[dict[str, Any]] = []
    for s in replay_summaries:
        gate = _evaluate_gate(s)
        gate_results.append(gate)

    with open(output_dir / "gate_evaluation.json", "w") as f:
        json.dump(gate_results, f, indent=2)
    logger.info("Wrote gate_evaluation.json with %d evaluations.", len(gate_results))

    # 6. Top-level summary
    n_pass = sum(1 for g in gate_results if g["global_gate_pass"])
    n_fail = len(gate_results) - n_pass

    phase4a_summary: dict[str, Any] = {
        "n_tasks_replayed": len(replay_summaries),
        "n_search_candidates": len(search_rows),
        "n_gate_pass": n_pass,
        "n_gate_fail": n_fail,
        "tasks_passing": [
            g["tag"] for g in gate_results if g["global_gate_pass"]
        ],
        "tasks_failing": [
            g["tag"] for g in gate_results if not g["global_gate_pass"]
        ],
    }

    with open(output_dir / "phase4A_summary.json", "w") as f:
        json.dump(phase4a_summary, f, indent=2)
    logger.info(
        "Phase IV-A aggregation complete: %d tasks, %d pass gate, %d fail.",
        len(replay_summaries), n_pass, n_fail,
    )

    # Print summary to stdout
    print(f"\n--- Phase IV-A Summary ---")
    print(f"Tasks replayed:    {len(replay_summaries)}")
    print(f"Search candidates: {len(search_rows)}")
    print(f"Gate PASS:         {n_pass}")
    print(f"Gate FAIL:         {n_fail}")
    if phase4a_summary["tasks_passing"]:
        print(f"Passing tasks:     {', '.join(phase4a_summary['tasks_passing'])}")
    if phase4a_summary["tasks_failing"]:
        print(f"Failing tasks:     {', '.join(phase4a_summary['tasks_failing'])}")
    print(f"Output dir:        {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--replay-dir", type=str,
        default="results/weighted_lse_dp/phase4/counterfactual_replay/",
        help="Directory containing per-task replay results.",
    )
    p.add_argument(
        "--search-dir", type=str,
        default="results/weighted_lse_dp/phase4/task_search/",
        help="Directory containing candidate_scores.csv.",
    )
    p.add_argument(
        "--output-dir", type=str,
        default="results/processed/phase4A/",
        help="Output directory for aggregated results.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
