"""Regenerate stale Phase IV schedule_v3.json files after MAJOR 6/8/9 fixes.

The Phase IV-A adversarial review identified three calibration defects
(MAJOR 6, MAJOR 8, MAJOR 9) that were already patched in the source code
under ``experiments/weighted_lse_dp/geometry/``:

* MAJOR 6 — ``phase4_calibration_v3.py`` now uses an argmin-style mutual-
  exclusion rule when assigning ``trust_clip_active_t`` vs
  ``safe_clip_active_t`` (lines ~315-330).
* MAJOR 8 — ``adaptive_headroom.run_fixed_point`` takes ``u_target_t`` and
  only bumps alpha where the feasibility criterion ``u_target > U_safe_ref``
  is violated.
* MAJOR 9 — ``trust_region.solve_u_tr_cap`` uses a Taylor short-circuit
  below ``_KL_EPS_FLOOR = 1e-7``.

All three fixes were landed into the source tree and the 777-case test
suite passes, but the 146 ``schedule_v3.json`` artifacts under
``results/weighted_lse_dp/phase4/`` were built with the pre-fix logic and
therefore carry the wrong clip flags (among other derived fields). This
script regenerates every stale schedule in place while preserving the
pre-fix artifact as a ``.pre_major689.json`` sidecar for auditability
(user-approved scope, Q1 = B, Q2 = A).

Determinism
-----------
``run_classical_pilot`` is deterministic in ``(cfg, seed, n_episodes)``;
the schedule builder has no non-deterministic branches. Running this
script twice on the same tree therefore is safe: on the second run each
schedule is skipped because its sidecar already exists (idempotent
guard).

Usage
-----
``python scripts/calibration/regenerate_schedule_v3_post_major689.py``

or constrain the walk with ``--root`` / ``--dry-run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path setup — mirror the runners' convention so we can import the
# calibration + pilot modules without installing the repo.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
_SRC = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _MUSHROOM_DEV, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Silence cv2/pygame SDL-dup warnings emitted during heavy imports.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np  # noqa: E402

from experiments.weighted_lse_dp.common.calibration import (  # noqa: E402
    get_task_sign,
)
from experiments.weighted_lse_dp.geometry.phase4_calibration_v3 import (  # noqa: E402
    build_schedule_v3_from_pilot,
)
from experiments.weighted_lse_dp.geometry.task_activation_search import (  # noqa: E402
    run_classical_pilot,
)


# ---------------------------------------------------------------------------
# Runner-family classification
# ---------------------------------------------------------------------------
#
# Each entry maps a *category* (derived from the path under ``phase4/``) to
# the exact (source_phase, notes, extra_kwargs) tuple that the originating
# runner passed to ``build_schedule_v3_from_pilot``. Keeping these strings
# identical to the runner definitions preserves provenance across
# regeneration so downstream diffing only surfaces the corrected flag
# arrays, not string noise.
#
# Sources (all in ``experiments/weighted_lse_dp/runners/``):
#
# * ``run_phase4C_advanced_rl.py`` — ``phase4/advanced/<algo>/...`` where
#   algo != geometry_priority_dp.
# * ``run_phase4C_geometry_dp.py`` — ``phase4/advanced/geometry_priority_dp/...``.
# * ``run_phase4_rl.py`` — ``phase4/translation_4a2/...``.
# * ``run_phase4_diagnostic_sweep.py`` — ``phase4/diagnostic_sweep_4a2_umax_<v>/...``;
#   the sweep value is encoded in the subtree name.


CATEGORY_ADVANCED_RL = "advanced_rl"
CATEGORY_GEOMETRY_DP = "geometry_priority_dp"
CATEGORY_TRANSLATION_RL = "translation_rl"
CATEGORY_TRANSLATION_DP = "translation_dp"
CATEGORY_DIAGNOSTIC_SWEEP = "diagnostic_sweep"

# In translation_4a2/, the algorithm subdir disambiguates which runner
# produced the schedule: run_phase4_dp.py emits safe_vi; every other
# algorithm goes through run_phase4_rl.py.
_TRANSLATION_DP_ALGORITHMS: frozenset[str] = frozenset({"safe_vi"})


def classify(schedule_path: Path) -> tuple[str, dict[str, Any]]:
    """Classify a ``schedule_v3.json`` path into a runner category.

    Returns the category label plus any category-specific extras (e.g. the
    ``u_max`` parsed out of a diagnostic sweep directory name).
    """
    parts = schedule_path.parts
    try:
        i = parts.index("phase4")
    except ValueError as exc:
        raise ValueError(
            f"cannot locate 'phase4' segment in path: {schedule_path}"
        ) from exc

    # parts[i+1] is the top-level suite directory under phase4/.
    suite = parts[i + 1]

    if suite == "advanced":
        # parts[i+2] is the algorithm. geometry_priority_dp is its own
        # runner; everything else goes through run_phase4C_advanced_rl.py.
        algo = parts[i + 2]
        if algo == "geometry_priority_dp":
            return CATEGORY_GEOMETRY_DP, {}
        return CATEGORY_ADVANCED_RL, {"algorithm": algo}

    if suite == "translation_4a2":
        # Layout: translation_4a2/<task>/<algorithm>/seed_<N>/schedule_v3.json
        # i.e. parts[i+3] is the algorithm subdir.
        algo = parts[i + 3]
        if algo in _TRANSLATION_DP_ALGORITHMS:
            return CATEGORY_TRANSLATION_DP, {"algorithm": algo}
        return CATEGORY_TRANSLATION_RL, {"algorithm": algo}

    if suite.startswith("diagnostic_sweep_4a2_umax_"):
        # e.g. "diagnostic_sweep_4a2_umax_0.0100" -> 0.01
        u_max_str = suite[len("diagnostic_sweep_4a2_umax_"):]
        try:
            u_max = float(u_max_str)
        except ValueError as exc:
            raise ValueError(
                f"cannot parse u_max from suite name {suite!r}"
            ) from exc
        return CATEGORY_DIAGNOSTIC_SWEEP, {"u_max": u_max}

    raise ValueError(
        f"unrecognised phase4 suite {suite!r} for path {schedule_path}"
    )


# ---------------------------------------------------------------------------
# Extraction helpers — pull (cfg, seed, gamma, n_pilot) from the sibling
# ``run.json`` in a category-aware way.
# ---------------------------------------------------------------------------


def _extract_from_run_json(
    run_json_path: Path,
    category: str,
) -> dict[str, Any]:
    """Return the (cfg, seed, gamma, n_pilot) tuple for the given run.

    The two runner families format ``run.json`` differently:

    * Advanced-RL + geometry-priority-DP: the top-level ``config`` key IS
      the task cfg. ``gamma`` is read from that config. Neither runner
      records ``n_pilot_episodes`` in run.json, so we use the hard-coded
      ``_N_PILOT = 200`` from both runners.
    * Translation + diagnostic_sweep: ``config`` is the full experiment
      config containing ``task_cfg`` (the pilot config) plus ``gamma``,
      ``n_pilot_episodes``, and ``seed``.
    """
    if not run_json_path.exists():
        raise FileNotFoundError(
            f"pilot cannot be reconstructed — sibling run.json missing: "
            f"{run_json_path}"
        )

    run_info = json.loads(run_json_path.read_text())
    seed = int(run_info["seed"])

    if category in (CATEGORY_ADVANCED_RL, CATEGORY_GEOMETRY_DP):
        cfg = run_info["config"]
        # Both runners default to 200; confirmed against source.
        n_pilot = 200
        gamma = float(cfg.get("gamma", 0.95))
    elif category in (
        CATEGORY_TRANSLATION_RL,
        CATEGORY_TRANSLATION_DP,
        CATEGORY_DIAGNOSTIC_SWEEP,
    ):
        exp_cfg = run_info["config"]
        cfg = exp_cfg["task_cfg"]
        n_pilot = int(exp_cfg.get("n_pilot_episodes", 200))
        gamma = float(exp_cfg.get("gamma", cfg.get("gamma", 0.95)))
    else:  # pragma: no cover — classify() already validates.
        raise ValueError(f"unknown category {category!r}")

    if "family" not in cfg:
        raise ValueError(
            f"task cfg extracted from {run_json_path} is missing 'family'"
        )

    return {"cfg": cfg, "seed": seed, "gamma": gamma, "n_pilot": n_pilot}


# ---------------------------------------------------------------------------
# Category-specific (source_phase, notes, extra build kwargs)
# ---------------------------------------------------------------------------


def _build_kwargs_for(
    category: str,
    extras: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Return (source_phase, notes, extra build_schedule kwargs)."""
    if category == CATEGORY_ADVANCED_RL:
        return (
            "phase4C_advanced_rl",
            "Phase IV-C advanced estimator schedule",
            {},
        )
    if category == CATEGORY_GEOMETRY_DP:
        return (
            "phase4C_geometry_dp",
            "Phase IV-C geometry-priority DP schedule",
            {},
        )
    if category == CATEGORY_TRANSLATION_RL:
        return (
            "phase4_rl",
            "Phase IV-B stagewise schedule from classical pilot",
            {},
        )
    if category == CATEGORY_TRANSLATION_DP:
        return (
            "phase4_dp",
            "Phase IV-B stagewise schedule from classical pilot (DP)",
            {},
        )
    if category == CATEGORY_DIAGNOSTIC_SWEEP:
        u_max = float(extras["u_max"])
        return (
            "phase4_diagnostic_sweep",
            (
                f"Phase IV-B diagnostic sweep (u_max={u_max:.4f}) "
                f"stagewise schedule from classical pilot"
            ),
            {"u_max": u_max},
        )
    raise ValueError(f"unknown category {category!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core regeneration loop
# ---------------------------------------------------------------------------


def regenerate_one(
    schedule_path: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Regenerate a single ``schedule_v3.json``; return per-file report."""
    report: dict[str, Any] = {
        "path": str(schedule_path),
        "status": "unknown",
        "old_hash": None,
        "new_hash": None,
        "category": None,
        "sidecar": None,
        "message": "",
    }

    # Idempotency guard.
    sidecar = schedule_path.with_suffix(".pre_major689.json")
    if sidecar.exists():
        report["status"] = "skipped"
        report["sidecar"] = str(sidecar)
        report["message"] = "sidecar already exists — already regenerated"
        return report

    # Classification + schema-header assertion on the existing file.
    category, extras = classify(schedule_path)
    report["category"] = category

    old_content = json.loads(schedule_path.read_text())
    # Schema assertions — spec-compliant schedule_v3 must carry these keys
    # and values. The script is not allowed to silently skip anything that
    # doesn't match the v3 contract.
    if old_content.get("schedule_version") != 3:
        raise AssertionError(
            f"schema mismatch: {schedule_path} has schedule_version="
            f"{old_content.get('schedule_version')!r} (expected 3)"
        )
    if old_content.get("phase") != "phase4":
        raise AssertionError(
            f"schema mismatch: {schedule_path} has phase="
            f"{old_content.get('phase')!r} (expected 'phase4')"
        )

    old_hash = _sha256(schedule_path)
    report["old_hash"] = old_hash

    # Pull pilot inputs from the sibling run.json.
    run_json_path = schedule_path.parent / "run.json"
    extracted = _extract_from_run_json(run_json_path, category)

    cfg = extracted["cfg"]
    seed = extracted["seed"]
    gamma = extracted["gamma"]
    n_pilot = extracted["n_pilot"]

    family = str(cfg.get("family", "unknown"))
    sign_family = int(get_task_sign(family))
    r_max = float(cfg.get("reward_bound", 1.0))

    source_phase, notes, extra_kwargs = _build_kwargs_for(category, extras)

    # Sanity check: if source_phase on disk disagrees with the canonical
    # value for its category, something about provenance tracking is off;
    # refuse to overwrite silently.
    disk_source_phase = old_content.get("source_phase")
    if disk_source_phase != source_phase:
        raise AssertionError(
            f"source_phase mismatch at {schedule_path}: disk="
            f"{disk_source_phase!r} vs expected={source_phase!r} "
            f"(category={category}); refusing to regenerate silently."
        )

    if dry_run:
        report["status"] = "dry_run"
        report["message"] = (
            f"would regenerate cfg={cfg!r} seed={seed} gamma={gamma} "
            f"n_pilot={n_pilot} source_phase={source_phase}"
        )
        return report

    # Run the deterministic pilot and recompute the schedule.
    pilot = run_classical_pilot(
        cfg=cfg,
        seed=seed,
        n_episodes=n_pilot,
        sign_family=sign_family,
    )

    # Move original to sidecar BEFORE overwriting.
    schedule_path.rename(sidecar)
    report["sidecar"] = str(sidecar)

    try:
        build_schedule_v3_from_pilot(
            pilot_data=pilot,
            r_max=r_max,
            gamma_base=gamma,
            gamma_eval=gamma,
            task_family=family,
            sign_family=sign_family,
            source_phase=source_phase,
            notes=notes,
            output_path=schedule_path,
            **extra_kwargs,
        )
    except Exception:
        # Best-effort: restore sidecar so we don't leave a hole.
        if schedule_path.exists():
            schedule_path.unlink()
        sidecar.rename(schedule_path)
        report["status"] = "failed"
        report["sidecar"] = None
        report["message"] = traceback.format_exc()
        return report

    report["new_hash"] = _sha256(schedule_path)
    report["status"] = "regenerated"
    report["message"] = (
        f"OK: cfg={family}/{cfg.get('n_states', '?')} seed={seed} "
        f"n_pilot={n_pilot}"
    )
    return report


def walk_and_regenerate(
    root: Path,
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    targets = sorted(root.rglob("schedule_v3.json"))
    print(f"[regen] found {len(targets)} schedule_v3.json files under {root}")
    reports: list[dict[str, Any]] = []
    for idx, path in enumerate(targets, 1):
        try:
            rep = regenerate_one(path, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 — surface all failures
            rep = {
                "path": str(path),
                "status": "failed",
                "old_hash": None,
                "new_hash": None,
                "category": None,
                "sidecar": None,
                "message": f"{exc!r}\n{traceback.format_exc()}",
            }
        reports.append(rep)
        tag = rep["status"].upper()
        old = (rep["old_hash"] or "none")[:12]
        new = (rep["new_hash"] or "n/a")[:12]
        print(
            f"[regen][{idx:>3d}/{len(targets)}] {tag:<11s} "
            f"{old} -> {new}  {rep['path']}"
        )
        if rep["status"] == "failed":
            print(f"[regen]   FAIL_DETAIL: {rep['message']}")
    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="regenerate_schedule_v3_post_major689",
        description=__doc__,
    )
    p.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "results" / "weighted_lse_dp" / "phase4",
        help="root directory under which to walk for schedule_v3.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="classify + read inputs but do not move sidecars or write",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="optional path to dump the per-file report JSON",
    )
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    reports = walk_and_regenerate(args.root, dry_run=args.dry_run)
    wall_s = time.perf_counter() - t0

    counts: dict[str, int] = {}
    for rep in reports:
        counts[rep["status"]] = counts.get(rep["status"], 0) + 1

    print()
    print(f"[regen] DONE in {wall_s:.1f}s")
    print(f"[regen] status counts: {counts}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {"wall_s": wall_s, "counts": counts, "reports": reports},
                indent=2,
            )
        )
        print(f"[regen] summary written to {args.summary_json}")

    # Non-zero exit if anything failed.
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
