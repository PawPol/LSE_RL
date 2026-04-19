#!/usr/bin/env python
"""Phase IV-A: activation search over candidate task families.

Uses only classical + certification diagnostics to identify tasks where the
safe operator produces measurable activation (effective-discount shift,
responsibility weight usage, clipping frequency) relative to classical
baselines.

Usage
-----
    python experiments/weighted_lse_dp/runners/run_phase4_activation_search.py \
        --seed 42 --n-pilot-episodes 50

    # Dry run (score only, do not freeze suite):
    python experiments/weighted_lse_dp/runners/run_phase4_activation_search.py \
        --seed 42 --n-pilot-episodes 3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root and vendored MushroomRL must be importable regardless of cwd.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, str(Path(_REPO_ROOT) / "mushroom-rl-dev"))

from experiments.weighted_lse_dp.geometry.task_activation_search import (  # noqa: E402
    score_all_candidates,
    select_activation_suite,
)
from experiments.weighted_lse_dp.tasks.phase4_operator_suite import get_search_grid  # noqa: E402
from experiments.weighted_lse_dp.common.io import save_json  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Best-effort git SHA; returns 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _now_iso() -> str:
    """Return ISO-8601 UTC timestamp with trailing Z."""
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if ts.endswith("+00:00"):
        ts = ts[: -len("+00:00")] + "Z"
    return ts


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types and ndarrays to JSON-safe types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_candidate_grid(
    output_dir: Path,
    search_grid: list[dict[str, Any]],
) -> Path:
    """Write candidate_grid.json -- all candidate configs."""
    path = output_dir / "candidate_grid.json"
    save_json(path, _make_serialisable(search_grid))
    logger.info("Wrote %s (%d candidates)", path, len(search_grid))
    return path


def _write_candidate_scores(
    output_dir: Path,
    scored_candidates: list[dict[str, Any]],
) -> Path:
    """Write candidate_scores.csv -- one row per candidate."""
    path = output_dir / "candidate_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "idx",
        "family",
        "total_score",
        "mean_abs_u_pred",
        "mean_abs_delta_d_pred",
        "mean_abs_target_gap_norm",
        "informative_stage_frac",
        "frac_u_ge_5e3",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in scored_candidates:
            metrics = c["scoring"]["raw_metrics"]
            row = {
                "idx": c["idx"],
                "family": c["family"],
                "total_score": f"{c['scoring']['total_score']:.8f}",
                "mean_abs_u_pred": f"{metrics['mean_abs_u_pred']:.8f}",
                "mean_abs_delta_d_pred": f"{metrics['mean_abs_delta_d_pred']:.8f}",
                "mean_abs_target_gap_norm": f"{metrics['mean_abs_target_gap_norm']:.8f}",
                "informative_stage_frac": f"{metrics['informative_stage_frac']:.8f}",
                "frac_u_ge_5e3": f"{metrics['frac_u_ge_5e3']:.8f}",
                "error": c.get("error", ""),
            }
            writer.writerow(row)

    logger.info("Wrote %s (%d rows)", path, len(scored_candidates))
    return path


def _write_selected_tasks(
    output_dir: Path,
    selected: list[dict[str, Any]],
) -> Path:
    """Write selected_tasks.json -- the frozen activation suite."""
    path = output_dir / "selected_tasks.json"

    # Build a clean list with cfg, scoring, schedule metadata, and reason
    clean: list[dict[str, Any]] = []
    for c in selected:
        entry: dict[str, Any] = {
            "idx": c["idx"],
            "family": c["family"],
            "cfg": c["cfg"],
            "scoring": c["scoring"],
            "acceptance_status": c.get("acceptance_status", "accepted"),
            "selected_reason": c.get("selected_reason", ""),
        }
        # Include schedule metadata (gamma_base, schedule_id) but not
        # the full per-stage arrays (too large for JSON).
        if c.get("schedule") is not None:
            sched = c["schedule"]
            entry["schedule_summary"] = {
                "gamma_base": sched.get("gamma_base"),
                "schedule_id": sched.get("schedule_id", ""),
                "sign": sched.get("sign"),
                "n_stages": len(sched.get("u_ref_used_t", [])),
            }
        clean.append(entry)

    save_json(path, _make_serialisable(clean))
    logger.info("Wrote %s (%d selected tasks)", path, len(clean))
    return path


def _write_activation_report(
    output_dir: Path,
    scored_candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    seed: int,
    n_pilot_episodes: int,
) -> Path:
    """Write activation_search_report.md."""
    path = output_dir / "activation_search_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    selected_idxs = {c["idx"] for c in selected}

    lines: list[str] = []
    lines.append("# Phase IV-A Activation Search Report\n")
    lines.append(f"- Seed: {seed}")
    lines.append(f"- Pilot episodes: {n_pilot_episodes}")
    lines.append(f"- Total candidates: {len(scored_candidates)}")
    lines.append(f"- Selected: {len(selected)}")
    lines.append(f"- Generated: {_now_iso()}")
    lines.append("")

    # Group by family
    from collections import defaultdict
    by_family: dict[str, list[dict]] = defaultdict(list)
    for c in scored_candidates:
        by_family[c["family"]].append(c)

    for family in sorted(by_family.keys()):
        candidates = by_family[family]
        n_selected = sum(1 for c in candidates if c["idx"] in selected_idxs)
        n_errors = sum(1 for c in candidates if c.get("error"))
        lines.append(f"## Family: `{family}`")
        lines.append(f"- Candidates: {len(candidates)}")
        lines.append(f"- Selected: {n_selected}")
        lines.append(f"- Errors: {n_errors}")
        lines.append("")

        # Sort by score descending
        sorted_cands = sorted(
            candidates,
            key=lambda c: c["scoring"]["total_score"],
            reverse=True,
        )
        lines.append("| Rank | Idx | Score | mean|u| | frac_active | Status |")
        lines.append("|------|-----|-------|---------|-------------|--------|")
        for rank, c in enumerate(sorted_cands, 1):
            m = c["scoring"]["raw_metrics"]
            status = "SELECTED" if c["idx"] in selected_idxs else "rejected"
            if c.get("error"):
                status = f"ERROR: {c['error'][:40]}"
            lines.append(
                f"| {rank} | {c['idx']} "
                f"| {c['scoring']['total_score']:.4f} "
                f"| {m['mean_abs_u_pred']:.6f} "
                f"| {m['frac_u_ge_5e3']:.4f} "
                f"| {status} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _write_frozen_suite_config(
    selected: list[dict[str, Any]],
    seed: int,
    n_pilot_episodes: int,
) -> Path:
    """Write the frozen activation_suite.json config."""
    config_path = Path(
        "experiments/weighted_lse_dp/configs/phase4/activation_suite.json"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)

    suite_entries: list[dict[str, Any]] = []
    for c in selected:
        entry: dict[str, Any] = {
            "family": c["family"],
            "cfg": c["cfg"],
        }
        if c.get("schedule") is not None:
            sched = c["schedule"]
            entry["gamma_base"] = sched.get("gamma_base")
            entry["schedule_id"] = sched.get("schedule_id", "")
        suite_entries.append(entry)

    payload = {
        "phase": "IV-A",
        "status": "frozen",
        "generated_by": "run_phase4_activation_search.py",
        "seed": seed,
        "n_pilot_episodes": n_pilot_episodes,
        "generated_at": _now_iso(),
        "git_sha": _git_sha(),
        "selected_tasks": suite_entries,
    }

    save_json(config_path, _make_serialisable(payload))
    logger.info("Wrote frozen suite config: %s", config_path)
    return config_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Run Phase IV-A activation search."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    seed = args.seed
    n_pilot_episodes = args.n_pilot_episodes
    output_dir = Path(args.output_dir)
    dry_run = args.dry_run

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase IV-A activation search starting")
    logger.info("  seed=%d  n_pilot_episodes=%d  dry_run=%s", seed, n_pilot_episodes, dry_run)
    logger.info("  output_dir=%s", output_dir)

    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Step 1: Get search grid
    # ------------------------------------------------------------------
    search_grid = get_search_grid()
    logger.info("Search grid: %d candidates", len(search_grid))

    # ------------------------------------------------------------------
    # Step 2: Score all candidates
    # ------------------------------------------------------------------
    logger.info("Scoring all candidates (this may take a while)...")
    scored_candidates = score_all_candidates(
        search_grid=search_grid,
        seed=seed,
        n_pilot_episodes=n_pilot_episodes,
    )
    n_errors = sum(1 for c in scored_candidates if c.get("error"))
    logger.info(
        "Scoring complete: %d scored, %d errors",
        len(scored_candidates) - n_errors,
        n_errors,
    )

    # ------------------------------------------------------------------
    # Step 3: Write grid and scores (always, even on dry-run)
    # ------------------------------------------------------------------
    _write_candidate_grid(output_dir, search_grid)
    _write_candidate_scores(output_dir, scored_candidates)

    # ------------------------------------------------------------------
    # Step 4: Select suite (unless dry-run)
    # ------------------------------------------------------------------
    if dry_run:
        logger.info("Dry run: skipping selection and freeze. Done.")
        _write_activation_report(
            output_dir, scored_candidates, [], seed, n_pilot_episodes,
        )
        elapsed = time.monotonic() - t0
        logger.info("Elapsed: %.1f s", elapsed)
        return

    selected = select_activation_suite(
        scored_candidates,
        min_per_family=1,
        max_per_family=2,
        min_mean_abs_u_pred=2e-3,
        min_frac_active_stages=0.05,
    )
    logger.info("Selected %d tasks for the activation suite", len(selected))

    # ------------------------------------------------------------------
    # Step 5: Write selected tasks and report
    # ------------------------------------------------------------------
    _write_selected_tasks(output_dir, selected)
    _write_activation_report(
        output_dir, scored_candidates, selected, seed, n_pilot_episodes,
    )
    _write_frozen_suite_config(selected, seed, n_pilot_episodes)

    elapsed = time.monotonic() - t0
    logger.info("Phase IV-A activation search complete. Elapsed: %.1f s", elapsed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-pilot-episodes", type=int, default=50)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/weighted_lse_dp/phase4/task_search/"),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Score candidates but do not select/freeze the suite.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
