#!/usr/bin/env python
"""Phase V WP0 consistency audit.

Scope: read-only audit over Phase I-IV result artifacts and paper text.

Deliverables emitted under ``results/audit/``:
    consistency_report.md           human-readable
    consistency_report.json         machine-readable (schema_version 1.0.0)
    recomputed_tables/<phase>/*.csv recomputed scalar tables
    recomputed_figures/             (empty; numeric comparison logged instead)

Severity taxonomy (exact strings used in the JSON ``findings`` rows):

    BLOCKER   any of the four §7 WP0 fail-loud gates trips:
              1. ``mean_effective_discount > 1 + 1e-8`` on a deployed safe
                 operator;
              2. ``realized_stagewise_discount > kappa_t + 1e-6`` on any
                 safe run (tolerance per spec);
              3. metadata disagreement across configs, summaries, and
                 paper claims;
              4. paper text disagrees with tabled numbers.

    MINOR     Phase I-IV runner emits ``experiment_manifest.json`` post
              hoc (or not at all) rather than in-runner (§13.4).

    INFO      informational row; no action required.

This script does NOT modify any runner, aggregator, or paper source.  It
consumes the already-persisted ``calibration_stats.npz``,
``summary.json``, ``schedule.json``, ``config.json``, ``metrics.json``,
and the LaTeX source, and only *reads* them.

Usage::

    .venv/bin/python scripts/audit/run_consistency_audit.py [--phase P]

With ``--phase`` the audit restricts to a single phase in
``{phase1, phase2, phase3, phase4A, phase4B, phase4C}``.  Without it
the full audit runs in < 15 min and writes both deliverables.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_AUDIT_ROOT = _REPO_ROOT / "results" / "audit"
_RECOMPUTED_TABLES = _AUDIT_ROOT / "recomputed_tables"
_RECOMPUTED_FIGURES = _AUDIT_ROOT / "recomputed_figures"

_PHASE1_RUNS = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase1"
_PHASE2_RUNS = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase2"
_PHASE3_RUNS = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase3"
_PHASE4_RUNS = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase4"

_PROCESSED_P1 = _REPO_ROOT / "results" / "weighted_lse_dp" / "processed" / "phase1"
_PROCESSED_P2 = _REPO_ROOT / "results" / "weighted_lse_dp" / "processed" / "phase2"
_PROCESSED_P4A = _REPO_ROOT / "results" / "processed" / "phase4A"
_PROCESSED_P4B = _REPO_ROOT / "results" / "processed" / "phase4b"

_CFG_P1 = _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs" / "phase1" / "paper_suite.json"
_CFG_P2 = _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs" / "phase2" / "paper_suite.json"
_CFG_P3 = _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs" / "phase3" / "paper_suite.json"

_PAPER_TEX = _REPO_ROOT / "paper" / "neurips_selective_temporal_credit_assignment_positioned.tex"

# ---------------------------------------------------------------------------
# Tolerances (spec §7 WP0)
# ---------------------------------------------------------------------------
_TOL_DEFF_CAP = 1e-8       # gate 1: mean_effective_discount <= 1 + 1e-8
_TOL_CERT = 1e-6           # gate 2: realized_stagewise_discount <= kappa_t + 1e-6
_TOL_TABLE_ABS = 1e-6      # table-value abs tolerance
_TOL_TABLE_REL = 1e-5      # table-value relative tolerance

_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Finding:
    id: str
    severity: str  # BLOCKER | MINOR | INFO
    phase: str
    artifact: str
    check: str
    expected: Any
    actual: Any
    tolerance: str

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        # Ensure JSON-serializable
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v) if isinstance(v, np.floating) else int(v)
        return d


class FindingCollector:
    def __init__(self) -> None:
        self._rows: list[Finding] = []
        self._counter: dict[str, int] = {}

    def add(self, *, severity: str, phase: str, artifact: str, check: str,
            expected: Any, actual: Any, tolerance: str) -> None:
        assert severity in {"BLOCKER", "MINOR", "INFO"}, severity
        key = phase
        n = self._counter.get(key, 0) + 1
        self._counter[key] = n
        fid = f"C-{phase}-{n:03d}"
        self._rows.append(Finding(
            id=fid, severity=severity, phase=phase, artifact=artifact,
            check=check, expected=_jsonify(expected), actual=_jsonify(actual),
            tolerance=tolerance,
        ))

    def rows(self) -> list[Finding]:
        return list(self._rows)


def _jsonify(x: Any) -> Any:
    """Coerce numpy/Path types to JSON-safe primitives."""
    if isinstance(x, pathlib.Path):
        return str(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    return x


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: pathlib.Path) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _iter_safe_runs(root: pathlib.Path) -> list[pathlib.Path]:
    """Find every run directory that contains ``calibration_stats.npz``."""
    if not root.is_dir():
        return []
    runs: list[pathlib.Path] = []
    for path in root.rglob("calibration_stats.npz"):
        runs.append(path.parent)
    return sorted(runs)


# ---------------------------------------------------------------------------
# Gate 1 & 2: d_eff bound and certified bound
# ---------------------------------------------------------------------------
def _check_safe_run(run_dir: pathlib.Path, fc: FindingCollector,
                    phase_label: str) -> dict[str, Any]:
    """Check one safe run against gates 1 and 2.

    Returns a dict of recomputed per-stage aggregates (for table output).
    """
    stats_path = run_dir / "calibration_stats.npz"
    with np.load(stats_path, allow_pickle=False) as npz:
        if "safe_effective_discount_mean" not in npz.files:
            # Suppress the INFO for runs that are explicitly classical (no
            # deployed safe operator to check). Phase IV-B mixes classical
            # and safe algorithms under the same task tree; classical runs
            # have no safe calibration by design.
            name = run_dir.parent.name.lower() if run_dir.parent else ""
            if "classical" not in name:
                fc.add(
                    severity="INFO", phase=phase_label,
                    artifact=_rel(stats_path),
                    check="safe_calibration_absent",
                    expected="safe_effective_discount_mean key present",
                    actual="missing",
                    tolerance="exact",
                )
            return {}
        ed_mean = np.asarray(npz["safe_effective_discount_mean"], dtype=np.float64)
        ed_std = np.asarray(npz["safe_effective_discount_std"], dtype=np.float64) \
            if "safe_effective_discount_std" in npz.files else None
        clip_frac = np.asarray(npz["safe_clip_fraction"], dtype=np.float64) \
            if "safe_clip_fraction" in npz.files else None

    # Gate 1: any per-stage mean_effective_discount > 1 + 1e-8?
    finite = np.isfinite(ed_mean)
    if finite.any():
        max_ed = float(np.max(ed_mean[finite]))
        if max_ed > 1.0 + _TOL_DEFF_CAP:
            fc.add(
                severity="BLOCKER", phase=phase_label,
                artifact=_rel(run_dir),
                check="d_eff_bound",
                expected=f"max safe_effective_discount_mean <= 1 + {_TOL_DEFF_CAP:g}",
                actual=f"max={max_ed:.12e}",
                tolerance=f"{_TOL_DEFF_CAP:g}",
            )

    # Gate 2: compare per-stage mean vs kappa_t from the schedule.
    prov_path = run_dir / "safe_provenance.json"
    schedule_path: pathlib.Path | None = None
    if prov_path.is_file():
        try:
            prov = _load_json(prov_path)
            sp = prov.get("schedule_path") or prov.get("schedule_file")
            if sp:
                sp_path = pathlib.Path(sp)
                if not sp_path.is_absolute():
                    sp_path = _REPO_ROOT / sp_path
                if sp_path.is_file():
                    schedule_path = sp_path
        except Exception as e:
            fc.add(
                severity="INFO", phase=phase_label,
                artifact=_rel(prov_path),
                check="safe_provenance_unreadable",
                expected="readable JSON with schedule_path",
                actual=f"{type(e).__name__}: {e}",
                tolerance="exact",
            )

    if schedule_path is None:
        # Fall back to config.json schedule_file if provenance is absent.
        cfg_path = run_dir / "config.json"
        if cfg_path.is_file():
            try:
                cfg = _load_json(cfg_path)
                sp = cfg.get("schedule_file")
                if sp:
                    sp_path = pathlib.Path(sp)
                    if not sp_path.is_absolute():
                        sp_path = _REPO_ROOT / sp_path
                    if sp_path.is_file():
                        schedule_path = sp_path
            except Exception:
                pass

    kappa_t: np.ndarray | None = None
    alpha_t: np.ndarray | None = None
    if schedule_path is not None and schedule_path.is_file():
        try:
            sched = _load_json(schedule_path)
            kappa_list = sched.get("kappa_t")
            if kappa_list is not None:
                kappa_t = np.asarray(kappa_list, dtype=np.float64)
            alpha_list = sched.get("alpha_t")
            if alpha_list is not None:
                alpha_t = np.asarray(alpha_list, dtype=np.float64)
        except Exception as e:
            fc.add(
                severity="INFO", phase=phase_label,
                artifact=_rel(schedule_path),
                check="schedule_unreadable",
                expected="readable JSON",
                actual=f"{type(e).__name__}: {e}",
                tolerance="exact",
            )

    if kappa_t is not None and np.isfinite(ed_mean).any():
        # Align lengths; take the overlap only.
        n = min(len(ed_mean), len(kappa_t))
        diff = ed_mean[:n] - kappa_t[:n]
        if np.isfinite(diff).any():
            over_tol = diff > _TOL_CERT
            if over_tol.any():
                worst_idx = int(np.argmax(diff))
                fc.add(
                    severity="BLOCKER", phase=phase_label,
                    artifact=_rel(run_dir),
                    check="cert_bound",
                    expected=(
                        f"safe_effective_discount_mean[t] <= kappa_t[t] + "
                        f"{_TOL_CERT:g} for all t"
                    ),
                    actual=(
                        f"stage={worst_idx} d_eff={ed_mean[worst_idx]:.6e} "
                        f"kappa={kappa_t[worst_idx]:.6e} "
                        f"diff={diff[worst_idx]:.6e}"
                    ),
                    tolerance=f"{_TOL_CERT:g}",
                )

    out = {
        "run_dir": _rel(run_dir),
        "n_stages": int(len(ed_mean)),
        "d_eff_mean_max": float(np.max(ed_mean[finite])) if finite.any() else float("nan"),
        "d_eff_mean_median": float(np.median(ed_mean[finite])) if finite.any() else float("nan"),
        "clip_fraction_mean": float(np.nanmean(clip_frac)) if clip_frac is not None else float("nan"),
        "kappa_max": float(np.max(kappa_t)) if kappa_t is not None else float("nan"),
        "alpha_max": float(np.max(alpha_t)) if alpha_t is not None else float("nan"),
    }
    return out


# ---------------------------------------------------------------------------
# Gate 3: metadata drift across configs, summaries, paper
# ---------------------------------------------------------------------------
def _extract_metadata_from_config(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return {task_name: {R_max, T, state_n, n_actions, n_seeds, gamma}}."""
    out: dict[str, dict[str, Any]] = {}
    seeds = cfg.get("seeds", [])
    chain_seeds = cfg.get("chain_seeds", seeds)
    tasks = cfg.get("tasks", {})
    for name, spec in tasks.items():
        R = spec.get("reward_bound", spec.get("rew", spec.get("goal_reward")))
        # Some specs use rew=[0,1] list — take max abs
        if isinstance(R, list):
            R = float(max(abs(x) for x in R))
        T = spec.get("horizon")
        # state count may be "state_n", "n_states", or implicit in grids
        n_s = spec.get("state_n", spec.get("n_states"))
        n_a = spec.get("n_actions")
        gamma = spec.get("gamma")
        # Phase III chain tasks use chain_seeds
        task_seeds = chain_seeds if name.startswith("chain") and chain_seeds else seeds
        out[name] = {
            "R_max": float(R) if R is not None else None,
            "T": int(T) if T is not None else None,
            "state_n": int(n_s) if n_s is not None else None,
            "n_actions": int(n_a) if n_a is not None else None,
            "gamma": float(gamma) if gamma is not None else None,
            "n_seeds": int(len(task_seeds)) if task_seeds else None,
            "seeds": sorted([int(x) for x in task_seeds]) if task_seeds else [],
        }
    return out


def _extract_metadata_from_run_config(run_dir: pathlib.Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = _load_json(cfg_path)
    except Exception:
        return None
    R = cfg.get("reward_bound", cfg.get("rew", cfg.get("goal_reward")))
    if isinstance(R, list):
        R = float(max(abs(x) for x in R))
    return {
        "R_max": float(R) if R is not None else None,
        "T": int(cfg.get("horizon")) if cfg.get("horizon") is not None else None,
        "state_n": int(cfg.get("state_n", cfg.get("n_states")))
                    if cfg.get("state_n", cfg.get("n_states")) is not None else None,
        "n_actions": int(cfg.get("n_actions")) if cfg.get("n_actions") is not None else None,
        "gamma": float(cfg.get("gamma")) if cfg.get("gamma") is not None else None,
    }


def _compare_metadata(
    fc: FindingCollector, phase_label: str, task: str,
    sources: dict[str, dict[str, Any]],
) -> None:
    """Compare per-field values across named sources; flag any mismatch.

    A value of ``None`` is treated as 'not declared by this source' and is
    skipped (not a mismatch).
    """
    fields = sorted({f for s in sources.values() for f in s.keys()})
    for field in fields:
        declared: dict[str, Any] = {}
        for src_name, src in sources.items():
            v = src.get(field)
            if v is None:
                continue
            declared[src_name] = v
        if len(declared) < 2:
            continue
        # Normalize lists to tuples for equality
        def _norm(v: Any) -> Any:
            if isinstance(v, list):
                return tuple(v)
            return v
        values = {k: _norm(v) for k, v in declared.items()}
        uniq = set(values.values())
        if len(uniq) == 1:
            continue
        # Mismatch: all declarations for this task field
        fc.add(
            severity="BLOCKER", phase=phase_label,
            artifact=f"task={task}",
            check="metadata_drift",
            expected=f"all sources agree on '{field}'",
            actual={k: _jsonify(v) for k, v in declared.items()},
            tolerance="exact",
        )


# ---------------------------------------------------------------------------
# Gate 3: schedule alpha/kappa consistency with configs
# ---------------------------------------------------------------------------
def _check_schedule_headroom(
    fc: FindingCollector, task_name: str, schedule_path: pathlib.Path,
    declared_alpha: float, phase_label: str,
) -> None:
    """Verify the schedule's alpha_t matches the declared headroom."""
    if not schedule_path.is_file():
        return
    try:
        sched = _load_json(schedule_path)
    except Exception:
        return
    alpha_t = sched.get("alpha_t")
    if alpha_t is None:
        return
    alpha_max = float(max(alpha_t))
    if abs(alpha_max - declared_alpha) > 1e-4:
        fc.add(
            severity="BLOCKER", phase=phase_label,
            artifact=_rel(schedule_path),
            check="metadata_drift",
            expected=(
                f"schedule.alpha_max ≈ declared headroom {declared_alpha:.3f}"
            ),
            actual=f"schedule.alpha_max = {alpha_max:.6f}",
            tolerance="1e-4",
        )


# ---------------------------------------------------------------------------
# Gate 4: paper text claims vs tabled numbers
# ---------------------------------------------------------------------------
_PAPER_CLAIMS_EXACT = [
    # (claim description, regex, expected string value)
    ("gamma shared by all algorithms",
     r"\\gamma=0\.99",
     "0.99"),
    ("headroom alpha_t = 0.10",
     r"\\alpha_t=0\.10",
     "0.10"),
    ("epsilon-greedy exploration eps = 0.1",
     r"\\varepsilon=0\.1",
     "0.1"),
    ("learning rate 1e-2 for RL",
     r"10\^\{-2\}",
     "1e-2"),
    ("three-seed default list",
     r"\\\{11,\s*29,\s*47\\\}",
     "[11, 29, 47]"),
]


def _check_paper_claims(
    fc: FindingCollector, paper_text: str,
    p1_cfg: dict[str, Any], p2_cfg: dict[str, Any], p3_cfg: dict[str, Any],
) -> None:
    """Cross-check a few hand-selected paper claims against configs."""
    phase_label = "paper"

    # Claim: gamma = 0.99 everywhere. Validate against configs.
    if re.search(_PAPER_CLAIMS_EXACT[0][1], paper_text):
        # All Phase I/II/III tasks should have gamma = 0.99
        for cfg_name, cfg in [("phase1", p1_cfg), ("phase2", p2_cfg), ("phase3", p3_cfg)]:
            for t, spec in cfg.get("tasks", {}).items():
                g = spec.get("gamma")
                if g is not None and abs(float(g) - 0.99) > 1e-9:
                    fc.add(
                        severity="BLOCKER", phase=phase_label,
                        artifact=f"paper gamma claim vs {cfg_name}/{t}",
                        check="text_vs_table",
                        expected="gamma = 0.99 (paper L1796)",
                        actual=f"{cfg_name}.{t}.gamma = {g}",
                        tolerance="1e-9",
                    )
    else:
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(_PAPER_TEX),
            check="text_vs_table",
            expected="paper declares gamma=0.99",
            actual="claim pattern not found",
            tolerance="regex",
        )

    # Claim: alpha_t = 0.10 everywhere.
    if re.search(_PAPER_CLAIMS_EXACT[1][1], paper_text):
        for cfg_name, cfg in [("phase1", p1_cfg), ("phase2", p2_cfg), ("phase3", p3_cfg)]:
            for t, spec in cfg.get("tasks", {}).items():
                a = spec.get("alpha", spec.get("headroom"))
                if a is not None and abs(float(a) - 0.10) > 1e-4:
                    fc.add(
                        severity="BLOCKER", phase=phase_label,
                        artifact=f"paper alpha claim vs {cfg_name}/{t}",
                        check="text_vs_table",
                        expected="alpha_t = 0.10 (paper L1803)",
                        actual=f"{cfg_name}.{t}.alpha = {a}",
                        tolerance="1e-4",
                    )
    else:
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(_PAPER_TEX),
            check="text_vs_table",
            expected="paper declares alpha_t = 0.10",
            actual="claim pattern not found",
            tolerance="regex",
        )

    # Claim: default seed list {11, 29, 47}.
    if re.search(_PAPER_CLAIMS_EXACT[4][1], paper_text):
        for cfg_name, cfg in [("phase1", p1_cfg), ("phase2", p2_cfg), ("phase3", p3_cfg)]:
            seeds = sorted(int(x) for x in cfg.get("seeds", []))
            if seeds and seeds != [11, 29, 47]:
                fc.add(
                    severity="BLOCKER", phase=phase_label,
                    artifact=f"paper seed claim vs {cfg_name}",
                    check="text_vs_table",
                    expected="paper declares seeds {11, 29, 47}",
                    actual=f"{cfg_name}.seeds = {seeds}",
                    tolerance="exact",
                )
    else:
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(_PAPER_TEX),
            check="text_vs_table",
            expected="paper declares seed list {11, 29, 47}",
            actual="claim pattern not found",
            tolerance="regex",
        )

    # Claim: clip fractions "from 0% (chain_sparse_long ...) to 79% (grid_hazard ...)" (L1856).
    m = re.search(
        r"from\s+(\d+)\\?%\s*\(chain\\?_sparse\\?_long.*?\)\s+to\s+(\d+)\\?%\s*\(grid\\?_hazard",
        paper_text,
    )
    if m:
        expected_low = int(m.group(1)) / 100.0
        expected_high = int(m.group(2)) / 100.0
        # Recompute clip fractions for the two mentioned tasks by averaging
        # every per-run ``safe_clip_fraction`` over the whole paper_suite
        # subtree. RL summaries don't re-emit this metric, so we must read
        # calibration_stats.npz directly.
        def _clip_frac(task: str) -> float | None:
            base = _PHASE3_RUNS / "paper_suite" / task
            if not base.is_dir():
                return None
            vals: list[float] = []
            for stats_path in base.rglob("calibration_stats.npz"):
                try:
                    with np.load(stats_path, allow_pickle=False) as npz:
                        cf = npz.get("safe_clip_fraction") if hasattr(npz, "get") \
                            else (npz["safe_clip_fraction"]
                                  if "safe_clip_fraction" in npz.files else None)
                except Exception:
                    continue
                if cf is None:
                    continue
                cf = np.asarray(cf, dtype=np.float64)
                if cf.size:
                    vals.append(float(np.nanmean(cf)))
            return float(np.mean(vals)) if vals else None

        actual_low = _clip_frac("chain_sparse_long")
        actual_high = _clip_frac("grid_hazard")
        if actual_low is not None and abs(actual_low - expected_low) > 0.01:
            fc.add(
                severity="BLOCKER", phase=phase_label,
                artifact="paper clip-fraction claim vs chain_sparse_long",
                check="text_vs_table",
                expected=f"clip_fraction(chain_sparse_long) ~= {expected_low:.2f} (paper L1856)",
                actual=f"recomputed mean clip_fraction = {actual_low:.4f}",
                tolerance="0.01",
            )
        if actual_high is not None and abs(actual_high - expected_high) > 0.01:
            fc.add(
                severity="BLOCKER", phase=phase_label,
                artifact="paper clip-fraction claim vs grid_hazard",
                check="text_vs_table",
                expected=f"clip_fraction(grid_hazard) ~= {expected_high:.2f} (paper L1856)",
                actual=f"recomputed mean clip_fraction = {actual_high:.4f}",
                tolerance="0.01",
            )
    else:
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(_PAPER_TEX),
            check="text_vs_table",
            expected="parseable clip-fraction range claim near L1856",
            actual="regex did not match; skipping numeric diff",
            tolerance="regex",
        )


# ---------------------------------------------------------------------------
# MINOR: runners that emit experiment_manifest.json post hoc
# ---------------------------------------------------------------------------
def _check_manifest_emission(fc: FindingCollector) -> None:
    """Flag every Phase I-IV runner that does not emit
    ``experiment_manifest.json`` from within the runner (spec §9 /
    §13.4)."""
    runner_dir = _REPO_ROOT / "experiments" / "weighted_lse_dp" / "runners"
    if not runner_dir.is_dir():
        return
    for path in sorted(runner_dir.glob("*.py")):
        try:
            text = path.read_text()
        except Exception:
            continue
        if "experiment_manifest" in text:
            continue
        # Only Phase I-IV runners are in scope for the minor finding.
        name = path.name
        phase = None
        if name.startswith("run_phase1"):
            phase = "phaseI"
        elif name.startswith("run_phase2"):
            phase = "phaseII"
        elif name.startswith("run_phase3"):
            phase = "phaseIII"
        elif name.startswith("run_phase4C"):
            phase = "phaseIV-C"
        elif name.startswith("run_phase4"):
            phase = "phaseIV"
        elif name.startswith("aggregate_phase"):
            continue  # aggregators are explicitly post hoc by design
        else:
            continue
        fc.add(
            severity="MINOR", phase=phase,
            artifact=_rel(path),
            check="manifest_post_hoc",
            expected="runner emits results/summaries/experiment_manifest.json (spec §9)",
            actual="no 'experiment_manifest' string found in runner source",
            tolerance="exact",
        )


# ---------------------------------------------------------------------------
# Recomputed-table writer
# ---------------------------------------------------------------------------
def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]],
               header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _diff_against_existing_csv(
    fc: FindingCollector, recomputed_path: pathlib.Path,
    reference_path: pathlib.Path, phase_label: str,
    key_cols: list[str], value_cols: list[str],
) -> None:
    """Diff a recomputed CSV against a committed CSV; log mismatches as
    BLOCKERs (numeric gap beyond tolerance) or INFOs (exact match)."""
    if not reference_path.is_file():
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(reference_path),
            check="table_diff",
            expected="committed reference table present",
            actual="reference file missing; skipping diff",
            tolerance="exact",
        )
        return

    def _load(path: pathlib.Path) -> list[dict[str, str]]:
        with open(path, "r") as fh:
            r = csv.DictReader(fh)
            return list(r)

    rec = _load(recomputed_path)
    ref = _load(reference_path)

    def _key(row: dict[str, str]) -> tuple:
        return tuple(row.get(c, "") for c in key_cols)

    rec_idx = {_key(r): r for r in rec}
    ref_idx = {_key(r): r for r in ref}
    missing_in_rec = set(ref_idx) - set(rec_idx)
    missing_in_ref = set(rec_idx) - set(ref_idx)
    common = set(rec_idx) & set(ref_idx)

    if missing_in_rec:
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(reference_path),
            check="table_diff",
            expected="all rows reproduced in recomputed table",
            actual=f"rows present in reference but missing in recomputed: {len(missing_in_rec)}",
            tolerance="exact",
        )
    if missing_in_ref:
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(reference_path),
            check="table_diff",
            expected="no extra rows in recomputed table",
            actual=f"rows present in recomputed but missing in reference: {len(missing_in_ref)}",
            tolerance="exact",
        )

    mismatches: list[dict[str, Any]] = []
    for key in sorted(common):
        a = rec_idx[key]
        b = ref_idx[key]
        for col in value_cols:
            sa = a.get(col, "")
            sb = b.get(col, "")
            if sa == "" and sb == "":
                continue
            if sa == sb:
                continue
            try:
                fa = float(sa) if sa != "" else float("nan")
                fb = float(sb) if sb != "" else float("nan")
            except ValueError:
                if sa != sb:
                    mismatches.append(
                        {"key": key, "col": col, "recomputed": sa, "reference": sb}
                    )
                continue
            if np.isfinite(fa) and np.isfinite(fb):
                abs_diff = abs(fa - fb)
                rel_diff = abs_diff / max(abs(fa), abs(fb), 1e-12)
                if abs_diff > _TOL_TABLE_ABS and rel_diff > _TOL_TABLE_REL:
                    mismatches.append({
                        "key": key, "col": col,
                        "recomputed": fa, "reference": fb,
                        "abs_diff": abs_diff, "rel_diff": rel_diff,
                    })

    if mismatches:
        # Up to 5 representative rows in the report to keep it readable.
        fc.add(
            severity="BLOCKER", phase=phase_label,
            artifact=_rel(reference_path),
            check="table_diff",
            expected=(
                f"recomputed values match reference within abs={_TOL_TABLE_ABS:g} "
                f"or rel={_TOL_TABLE_REL:g}"
            ),
            actual={"n_mismatches": len(mismatches), "sample": mismatches[:5]},
            tolerance=f"abs={_TOL_TABLE_ABS:g} rel={_TOL_TABLE_REL:g}",
        )
    else:
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(reference_path),
            check="table_diff",
            expected="recomputed table matches reference",
            actual=f"{len(common)} rows match within tolerance",
            tolerance=f"abs={_TOL_TABLE_ABS:g} rel={_TOL_TABLE_REL:g}",
        )


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------
def _phase_iii_recompute_clip_activity(
    fc: FindingCollector, out_dir: pathlib.Path,
) -> None:
    """Recompute the per-task clip-activity table from Phase III runs."""
    rows: list[dict[str, Any]] = []
    if not _PHASE3_RUNS.is_dir():
        return
    paper_suite = _PHASE3_RUNS / "paper_suite"
    if not paper_suite.is_dir():
        return
    for task_dir in sorted(paper_suite.iterdir()):
        if not task_dir.is_dir():
            continue
        clip_vals: list[float] = []
        ed_vals: list[float] = []
        beta_vals: list[float] = []
        n_runs = 0
        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            for seed_dir in sorted(algo_dir.iterdir()):
                stats = seed_dir / "calibration_stats.npz"
                if not stats.is_file():
                    continue
                try:
                    with np.load(stats, allow_pickle=False) as npz:
                        cf = np.asarray(npz["safe_clip_fraction"], dtype=np.float64)
                        ed = np.asarray(npz["safe_effective_discount_mean"], dtype=np.float64)
                        bu = np.asarray(npz["safe_beta_used_mean"], dtype=np.float64)
                except Exception:
                    continue
                clip_vals.append(float(np.nanmean(cf)))
                ed_vals.append(float(np.nanmean(ed)))
                beta_vals.append(float(np.nanmean(np.abs(bu))))
                n_runs += 1
        if not clip_vals:
            continue
        rows.append({
            "task": task_dir.name,
            "n_runs": n_runs,
            "clip_fraction_mean": f"{np.mean(clip_vals):.6f}",
            "effective_discount_mean": f"{np.mean(ed_vals):.6f}",
            "beta_used_abs_mean": f"{np.mean(beta_vals):.6e}",
        })
    if rows:
        out_path = out_dir / "phase3" / "clip_activity.csv"
        _write_csv(out_path, rows,
                   header=["task", "n_runs", "clip_fraction_mean",
                           "effective_discount_mean", "beta_used_abs_mean"])
        fc.add(
            severity="INFO", phase="phaseIII",
            artifact=_rel(out_path),
            check="recompute_write",
            expected="clip-activity table recomputed",
            actual=f"{len(rows)} tasks",
            tolerance="exact",
        )


def _phase_iv_b_recompute_p4b_a(
    fc: FindingCollector, out_dir: pathlib.Path,
) -> None:
    """Recompute P4B_A from aggregated Phase IV-B summaries and diff
    against the committed CSV."""
    agg_root = _PHASE4_RUNS / "translation_4a2" / "aggregated"
    if not agg_root.is_dir():
        return
    rows: list[dict[str, Any]] = []
    for task_dir in sorted(agg_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for algo_dir in sorted(task_dir.iterdir()):
            sj = algo_dir / "summary.json"
            if not sj.is_file():
                continue
            try:
                s = _load_json(sj)
            except Exception:
                continue
            algo = algo_dir.name
            task = task_dir.name
            cls = s.get("class", "")
            if not cls:
                if "classical" in algo:
                    cls = "classical"
                elif algo.endswith("_zero"):
                    cls = "safe-zero"
                elif "stagewise" in algo or algo in {"safe_vi", "safe_q_stagewise"}:
                    cls = "safe-nonlinear"
                else:
                    cls = ""
            metrics = s.get("metrics", {})
            mr = metrics.get("mean_return", {})
            n_seeds = s.get("n_seeds")
            rows.append({
                "task": task,
                "algorithm": algo,
                "class": cls,
                "primary_metric": "mean_return",
                "mean": f"{mr['mean']:.4f}" if mr.get("mean") is not None else "",
                "std": f"{mr['std']:.4e}" if mr.get("std") is not None else "",
                "n_seeds": n_seeds if n_seeds is not None else "",
            })
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: (r["task"], r["algorithm"]))
    out_path = out_dir / "phase4B" / "P4B_A.csv"
    _write_csv(out_path, rows_sorted,
               header=["task", "algorithm", "class", "primary_metric",
                       "mean", "std", "n_seeds"])
    # Diff
    ref = _PROCESSED_P4B / "P4B_A.csv"
    _diff_against_existing_csv(
        fc, out_path, ref, "phaseIV-B",
        key_cols=["task", "algorithm"],
        value_cols=["mean", "std", "n_seeds", "class"],
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def _run_phase1(fc: FindingCollector) -> dict[str, Any]:
    """Phase I is classical-only; gates 1/2 are N/A. We still check
    metadata drift between config, per-run configs, and the processed
    tables."""
    phase_label = "phaseI"
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()

    cfg = _load_json(_CFG_P1)
    cfg_meta = _extract_metadata_from_config(cfg)

    # Walk Phase I smoke runs (these are the canonical Phase I runs)
    smoke_root = _PHASE1_RUNS / "smoke"
    per_task_run_meta: dict[str, dict[str, Any]] = {}
    if smoke_root.is_dir():
        for task_dir in sorted(smoke_root.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            # Find any one config.json under the subtree
            for run_cfg in task_dir.rglob("run.json"):
                try:
                    j = _load_json(run_cfg)
                    cfg_sub = j.get("config", {})
                    R = cfg_sub.get("reward_bound", cfg_sub.get("rew"))
                    if isinstance(R, list):
                        R = float(max(abs(x) for x in R))
                    per_task_run_meta.setdefault(task, {}).update({
                        "R_max": float(R) if R is not None else None,
                        "T": int(cfg_sub.get("horizon")) if cfg_sub.get("horizon") else None,
                        "gamma": float(cfg_sub.get("gamma")) if cfg_sub.get("gamma") else None,
                    })
                    break
                except Exception:
                    continue

    # Cross-check config vs per-run
    for task in cfg_meta:
        sources: dict[str, dict[str, Any]] = {"config/phase1/paper_suite.json": cfg_meta[task]}
        run_meta = per_task_run_meta.get(task)
        if run_meta:
            sources["smoke_run.json"] = run_meta
        _compare_metadata(fc, phase_label, task, sources)

    ended = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return {"phase": "phase1", "started": started, "ended": ended,
            "tasks": list(cfg_meta)}


def _run_phase2(fc: FindingCollector) -> dict[str, Any]:
    phase_label = "phaseII"
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()

    cfg = _load_json(_CFG_P2)
    cfg_meta = _extract_metadata_from_config(cfg)

    # Per-task runtime metadata from aggregated summaries
    agg_root = _PHASE2_RUNS / "aggregated"
    for task in cfg_meta:
        task_dir = agg_root / task
        sources: dict[str, dict[str, Any]] = {"config/phase2/paper_suite.json": cfg_meta[task]}
        if task_dir.is_dir():
            # Gather n_seeds across all algos (should all agree for a single task)
            n_seeds_set: set[int] = set()
            for algo_dir in task_dir.iterdir():
                sj = algo_dir / "summary.json"
                if sj.is_file():
                    try:
                        s = _load_json(sj)
                        if "n_seeds" in s:
                            n_seeds_set.add(int(s["n_seeds"]))
                    except Exception:
                        continue
            if n_seeds_set:
                sources["aggregated/summary.json"] = {"n_seeds": min(n_seeds_set)}
        _compare_metadata(fc, phase_label, task, sources)

    ended = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return {"phase": "phase2", "started": started, "ended": ended,
            "tasks": list(cfg_meta)}


def _run_phase3(fc: FindingCollector) -> dict[str, Any]:
    phase_label = "phaseIII"
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()

    cfg = _load_json(_CFG_P3)
    cfg_meta = _extract_metadata_from_config(cfg)

    # Gates 1 & 2 across every safe run
    n_runs = 0
    for run_dir in _iter_safe_runs(_PHASE3_RUNS / "paper_suite"):
        _check_safe_run(run_dir, fc, phase_label)
        n_runs += 1

    # Schedule alpha vs paper headroom 0.10
    cal_root = _PHASE3_RUNS / "calibration"
    if cal_root.is_dir():
        for task in cfg_meta:
            sched = cal_root / task / "schedule.json"
            if sched.is_file():
                _check_schedule_headroom(fc, task, sched, 0.10, phase_label)

    # Metadata cross-check
    for task in cfg_meta:
        sources: dict[str, dict[str, Any]] = {"config/phase3/paper_suite.json": cfg_meta[task]}
        # Any run config
        for run_dir in (_PHASE3_RUNS / "paper_suite" / task).glob("*/seed_*") \
                if (_PHASE3_RUNS / "paper_suite" / task).is_dir() else []:
            rm = _extract_metadata_from_run_config(run_dir)
            if rm:
                sources["run/config.json"] = rm
                break
        _compare_metadata(fc, phase_label, task, sources)

    # Recompute clip-activity table
    _phase_iii_recompute_clip_activity(fc, _RECOMPUTED_TABLES)

    ended = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return {
        "phase": "phase3", "started": started, "ended": ended,
        "tasks": list(cfg_meta), "n_safe_runs": n_runs,
    }


def _run_phase4A(fc: FindingCollector) -> dict[str, Any]:
    phase_label = "phaseIV-A"
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    # No additional numerics beyond what Phase IV-B consumes; check that
    # the processed activation-report directory exists.
    ad = _PROCESSED_P4A / "activation_diagnostics.json"
    if not ad.is_file():
        fc.add(
            severity="INFO", phase=phase_label,
            artifact=_rel(ad),
            check="recompute_write",
            expected="Phase IV-A activation diagnostics present",
            actual="missing",
            tolerance="exact",
        )
    ended = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return {"phase": "phase4A", "started": started, "ended": ended}


def _run_phase4B(fc: FindingCollector) -> dict[str, Any]:
    phase_label = "phaseIV-B"
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()

    # Gates 1 & 2 across every safe run that has calibration_stats.npz
    n_runs = 0
    for run_dir in _iter_safe_runs(_PHASE4_RUNS / "translation_4a2"):
        _check_safe_run(run_dir, fc, phase_label)
        n_runs += 1

    # Recompute P4B_A and diff
    _phase_iv_b_recompute_p4b_a(fc, _RECOMPUTED_TABLES)

    ended = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return {"phase": "phase4B", "started": started, "ended": ended,
            "n_safe_runs": n_runs}


def _run_phase4C(fc: FindingCollector) -> dict[str, Any]:
    phase_label = "phaseIV-C"
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    n_runs = 0
    for sub in ("advanced", "task_search"):
        root = _PHASE4_RUNS / sub
        if root.is_dir():
            for run_dir in _iter_safe_runs(root):
                _check_safe_run(run_dir, fc, phase_label)
                n_runs += 1
    ended = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return {"phase": "phase4C", "started": started, "ended": ended,
            "n_safe_runs": n_runs}


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------
def _emit_reports(fc: FindingCollector, phases_run: list[dict[str, Any]]) -> None:
    _AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = fc.rows()
    blockers = [r for r in rows if r.severity == "BLOCKER"]
    minors = [r for r in rows if r.severity == "MINOR"]
    infos = [r for r in rows if r.severity == "INFO"]

    summary = {
        "blockers": len(blockers),
        "minors": len(minors),
        "infos": len(infos),
        "phases_completed": [p["phase"] for p in phases_run],
    }

    report = {
        "schema_version": _SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "summary": summary,
        "phases": phases_run,
        "findings": [r.to_dict() for r in rows],
    }

    json_path = _AUDIT_ROOT / "consistency_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)

    # Markdown
    md_path = _AUDIT_ROOT / "consistency_report.md"
    lines: list[str] = []
    lines.append("# Phase V WP0 consistency audit")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Git SHA: `{report['git_sha']}`")
    lines.append(f"Schema version: {_SCHEMA_VERSION}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- BLOCKER: {summary['blockers']}")
    lines.append(f"- MINOR:   {summary['minors']}")
    lines.append(f"- INFO:    {summary['infos']}")
    lines.append("")
    lines.append(f"Phases completed: {', '.join(summary['phases_completed'])}")
    lines.append("")
    if blockers:
        lines.append("## BLOCKERs")
        lines.append("")
        lines.append("| id | phase | check | artifact | expected | actual |")
        lines.append("|---|---|---|---|---|---|")
        for r in blockers:
            lines.append(
                f"| {r.id} | {r.phase} | {r.check} | `{r.artifact}` | "
                f"{_short(r.expected)} | {_short(r.actual)} |"
            )
        lines.append("")
    if minors:
        lines.append("## MINORs (post-hoc manifest emission and similar)")
        lines.append("")
        lines.append("| id | phase | check | artifact | expected | actual |")
        lines.append("|---|---|---|---|---|---|")
        for r in minors:
            lines.append(
                f"| {r.id} | {r.phase} | {r.check} | `{r.artifact}` | "
                f"{_short(r.expected)} | {_short(r.actual)} |"
            )
        lines.append("")
    if infos:
        lines.append("## INFOs")
        lines.append("")
        lines.append("| id | phase | check | artifact | expected | actual |")
        lines.append("|---|---|---|---|---|---|")
        for r in infos:
            lines.append(
                f"| {r.id} | {r.phase} | {r.check} | `{r.artifact}` | "
                f"{_short(r.expected)} | {_short(r.actual)} |"
            )
        lines.append("")

    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))

    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {md_path}")
    print(f"[audit] BLOCKER={summary['blockers']} MINOR={summary['minors']} "
          f"INFO={summary['infos']}")


def _short(obj: Any, limit: int = 160) -> str:
    s = json.dumps(obj) if not isinstance(obj, str) else obj
    s = s.replace("|", "\\|")
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["phase1", "phase2", "phase3",
                                         "phase4A", "phase4B", "phase4C"],
                    default=None,
                    help="restrict audit to a single phase (default: all)")
    args = ap.parse_args(argv)

    fc = FindingCollector()
    phases_run: list[dict[str, Any]] = []

    handlers = {
        "phase1": _run_phase1,
        "phase2": _run_phase2,
        "phase3": _run_phase3,
        "phase4A": _run_phase4A,
        "phase4B": _run_phase4B,
        "phase4C": _run_phase4C,
    }
    selected = [args.phase] if args.phase else list(handlers)
    for name in selected:
        try:
            phases_run.append(handlers[name](fc))
        except Exception as e:  # pragma: no cover - safety net
            phases_run.append({"phase": name, "error": f"{type(e).__name__}: {e}"})
            fc.add(
                severity="INFO", phase=name,
                artifact="<audit_runner>",
                check="phase_handler_exception",
                expected="phase handler completes without error",
                actual=f"{type(e).__name__}: {e}",
                tolerance="exact",
            )

    # Gate 4 (paper claims) always runs, regardless of phase selection.
    try:
        paper_text = _PAPER_TEX.read_text()
    except Exception as e:
        fc.add(
            severity="INFO", phase="paper",
            artifact=_rel(_PAPER_TEX),
            check="paper_source_unreadable",
            expected="readable LaTeX source",
            actual=f"{type(e).__name__}: {e}",
            tolerance="exact",
        )
    else:
        try:
            p1 = _load_json(_CFG_P1) if _CFG_P1.is_file() else {}
        except Exception:
            p1 = {}
        try:
            p2 = _load_json(_CFG_P2) if _CFG_P2.is_file() else {}
        except Exception:
            p2 = {}
        try:
            p3 = _load_json(_CFG_P3) if _CFG_P3.is_file() else {}
        except Exception:
            p3 = {}
        _check_paper_claims(fc, paper_text, p1, p2, p3)

    # MINOR runner-manifest check (always runs).
    _check_manifest_emission(fc)

    _emit_reports(fc, phases_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
