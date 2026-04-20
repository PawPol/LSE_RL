#!/usr/bin/env python
"""Phase IV-A mainline activation-suite rerun (1000 pilot episodes).

Re-scores the activation search grid at the higher pilot budget so the
trust-region confidence cap releases enough to pass the activation gate
(mean_abs_u >= 5e-3 on at least one family). Also builds a short-horizon
(T in {5, 10}) appendix suite at 200 pilot episodes for sanity checks.

Outputs
-------
results/weighted_lse_dp/phase4/task_search/
    candidate_scores.csv               (overwritten: 1000-ep mainline)
    selected_tasks.json                (mainline suite, suite_type=mainline)
    selected_tasks_appendix.json       (short-horizon appendix suite)
    activation_search_report.md        (updated summary)

results/weighted_lse_dp/phase4/activation_report/
    gate_table_by_family.md
    global_diagnostics.json
    event_conditioned_diagnostics.json
    binding_cap_summary.json

experiments/weighted_lse_dp/configs/phase4/
    activation_suite.json              (frozen mainline + appendix for IV-B)
    gamma_matched_controls.json        (classical controls)

Run
---
    PYTHONPATH=. python3 scripts/overnight/run_phase4a_mainline_rerun.py
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Repo root on PYTHONPATH
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
if _MUSHROOM_DEV.exists() and str(_MUSHROOM_DEV) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM_DEV))

from experiments.weighted_lse_dp.geometry.task_activation_search import (  # noqa: E402
    score_all_candidates,
    select_activation_suite,
)
from experiments.weighted_lse_dp.tasks.phase4_operator_suite import (  # noqa: E402
    get_search_grid,
)
from experiments.weighted_lse_dp.common.io import save_json  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
# Quiet the activation-search INFO lines ("Scoring candidate ...")
logging.getLogger(
    "experiments.weighted_lse_dp.geometry.task_activation_search"
).setLevel(logging.WARNING)
logger = logging.getLogger("phase4a_mainline_rerun")


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

RESULTS_ROOT = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase4"
SEARCH_DIR = RESULTS_ROOT / "task_search"
REPORT_DIR = RESULTS_ROOT / "activation_report"
CONFIG_DIR = _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs" / "phase4"

SEED = 42
MAINLINE_PILOT_EPISODES = 1000
APPENDIX_PILOT_EPISODES = 200
MAINLINE_T = 20
APPENDIX_T_VARIANTS = [5, 10]
GATE_U_THRESHOLD = 5e-3
GATE_FRAC_THRESHOLD = 0.10
EVENT_REWARD_FRACTION = 0.3  # event stage: mean reward > 0.3 * reward_bound


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_REPO_ROOT),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _now_iso() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return ts.replace("+00:00", "Z")


def _make_serialisable(obj: Any) -> Any:
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
# Grid filtering and appendix construction
# ---------------------------------------------------------------------------


def build_mainline_grid(
    full_grid: list[dict[str, Any]], T_target: int = MAINLINE_T
) -> list[dict[str, Any]]:
    """Filter the full search grid to the T=T_target mainline subset.

    Mainline rule: horizon == T_target. Every family's T=20 variants are
    retained. Taxi (gamma=0.97 only, horizons {40, 60}) has no T=20 and
    is therefore excluded from the mainline by construction.
    """
    out = [cfg for cfg in full_grid if int(cfg.get("horizon", -1)) == T_target]
    logger.info(
        "Mainline grid (T=%d): %d candidates (full grid has %d)",
        T_target, len(out), len(full_grid),
    )
    return out


def build_appendix_grid(
    full_grid: list[dict[str, Any]], T_variants: list[int] = APPENDIX_T_VARIANTS
) -> list[dict[str, Any]]:
    """Build the short-horizon appendix grid by horizon override.

    For every non-taxi mainline candidate (T=MAINLINE_T), emit one copy
    per T in T_variants with ``horizon`` overridden. Taxi is skipped
    because changing horizon on the step-intercepting wrapper is not
    equivalent to a fresh DP recomputation.
    """
    mainline = build_mainline_grid(full_grid, T_target=MAINLINE_T)
    out: list[dict[str, Any]] = []
    for cfg in mainline:
        if cfg.get("family") == "taxi_bonus":
            continue
        for T in T_variants:
            variant = dict(cfg)
            variant["horizon"] = T
            out.append(variant)
    logger.info(
        "Appendix grid (T in %s): %d candidates",
        T_variants, len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_candidate_scores_csv(
    output_path: Path, scored_candidates: list[dict[str, Any]]
) -> None:
    """Write candidate_scores.csv with the mainline schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "idx", "family", "total_score",
        "mean_abs_u_pred", "mean_abs_delta_d_pred",
        "mean_abs_target_gap_norm", "informative_stage_frac",
        "frac_u_ge_5e3", "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in scored_candidates:
            m = c["scoring"]["raw_metrics"]
            w.writerow({
                "idx": c["idx"],
                "family": c["family"],
                "total_score": f"{c['scoring']['total_score']:.8f}",
                "mean_abs_u_pred": f"{m['mean_abs_u_pred']:.8f}",
                "mean_abs_delta_d_pred": f"{m['mean_abs_delta_d_pred']:.8f}",
                "mean_abs_target_gap_norm": f"{m['mean_abs_target_gap_norm']:.8f}",
                "informative_stage_frac": f"{m['informative_stage_frac']:.8f}",
                "frac_u_ge_5e3": f"{m['frac_u_ge_5e3']:.8f}",
                "error": c.get("error", ""),
            })
    logger.info("Wrote %s (%d rows)", output_path, len(scored_candidates))


def _task_entry(c: dict[str, Any]) -> dict[str, Any]:
    """Build a selected-tasks entry from a scored candidate."""
    entry: dict[str, Any] = {
        "idx": c["idx"],
        "family": c["family"],
        "cfg": c["cfg"],
        "scoring": c["scoring"],
        "acceptance_status": c.get("acceptance_status", "accepted"),
        "selected_reason": c.get("selected_reason", ""),
    }
    if c.get("schedule") is not None:
        s = c["schedule"]
        entry["schedule_summary"] = {
            "gamma_base": s.get("gamma_base"),
            "schedule_id": s.get("schedule_id", ""),
            "sign": s.get("sign"),
            "n_stages": len(s.get("u_ref_used_t", [])),
        }
    return entry


def write_selected_tasks_json(
    output_path: Path,
    selected: list[dict[str, Any]],
    suite_type: str,
    n_pilot_episodes: int,
    tau_n: float,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "suite_type": suite_type,
        "n_pilot_episodes": n_pilot_episodes,
        "tau_n": tau_n,
        "seed": SEED,
        "generated_at": _now_iso(),
        "git_sha": _git_sha(),
        "selected_families": sorted({c["family"] for c in selected}),
        "tasks": [_task_entry(c) for c in selected],
    }
    if extra:
        payload.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, _make_serialisable(payload))
    logger.info(
        "Wrote %s (%d selected, suite_type=%s)",
        output_path, len(selected), suite_type,
    )


def write_activation_report_md(
    output_path: Path,
    scored_mainline: list[dict[str, Any]],
    selected_mainline: list[dict[str, Any]],
    scored_appendix: list[dict[str, Any]],
    selected_appendix: list[dict[str, Any]],
) -> None:
    """Write a high-level narrative report summarising the rerun."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_idxs = {c["idx"] for c in selected_mainline}

    lines: list[str] = []
    lines.append("# Phase IV-A Activation Search Report (mainline rerun)\n")
    lines.append(f"- Seed: {SEED}")
    lines.append(f"- Mainline pilot episodes: {MAINLINE_PILOT_EPISODES}")
    lines.append(f"- Appendix pilot episodes: {APPENDIX_PILOT_EPISODES}")
    lines.append(f"- Mainline horizon: T = {MAINLINE_T}")
    lines.append(f"- Appendix horizons: T in {APPENDIX_T_VARIANTS}")
    lines.append(f"- Mainline candidates scored: {len(scored_mainline)}")
    lines.append(f"- Mainline selected: {len(selected_mainline)}")
    lines.append(f"- Appendix candidates scored: {len(scored_appendix)}")
    lines.append(f"- Appendix selected: {len(selected_appendix)}")
    lines.append(f"- Generated: {_now_iso()}")
    lines.append("")
    lines.append("## What changed vs. the 200-episode run")
    lines.append("")
    lines.append(
        "The original activation search used `n_pilot_episodes = 200`. "
        "At that budget, the trust-region confidence factor `c_t = "
        "(n_t / (n_t + tau_n)) * sqrt(p_align_t)` never releases enough "
        "for `u_tr_cap` to clear the `|u| >= 5e-3` gate: the best observed "
        "value was `mean_abs_u = 0.00356`, giving a FAILED activation gate "
        "(10/11 conditions)."
    )
    lines.append("")
    lines.append(
        "This rerun uses `n_pilot_episodes = 1000` on the T=20 mainline "
        "subset of the search grid, keeping `tau_n = 200`. The extra pilot "
        "budget is the only change — the operator, schedule geometry, and "
        "gate threshold are unchanged."
    )
    lines.append("")
    lines.append("## Gate status")
    lines.append("")
    if scored_mainline:
        mains_u = max(
            c["scoring"]["raw_metrics"]["mean_abs_u_pred"]
            for c in scored_mainline
        )
        mains_frac = max(
            c["scoring"]["raw_metrics"]["frac_u_ge_5e3"]
            for c in scored_mainline
        )
    else:
        mains_u = mains_frac = 0.0
    lines.append(
        f"- Best mainline `mean_abs_u_pred` = **{mains_u:.5f}** "
        f"(threshold {GATE_U_THRESHOLD})"
    )
    lines.append(
        f"- Best mainline `frac(|u| >= 5e-3)` = **{mains_frac:.3f}** "
        f"(threshold {GATE_FRAC_THRESHOLD})"
    )
    gate_ok = mains_u >= GATE_U_THRESHOLD and mains_frac >= GATE_FRAC_THRESHOLD
    lines.append(
        f"- Gate status: **{'PASS' if gate_ok else 'FAIL'}** "
        f"(before rerun: FAIL)"
    )
    lines.append("")
    lines.append("## Suites")
    lines.append("")
    mainline_families = sorted({c["family"] for c in selected_mainline})
    appendix_families = sorted({c["family"] for c in selected_appendix})
    lines.append(f"- Mainline families ({len(mainline_families)}): "
                 f"`{', '.join(mainline_families) or '(none)'}`")
    lines.append(f"- Appendix families ({len(appendix_families)}): "
                 f"`{', '.join(appendix_families) or '(none)'}`")
    lines.append(
        "- See `activation_report/binding_cap_summary.json` for the "
        "per-family breakdown of trust-region vs safe-reference clipping."
    )
    lines.append("")
    lines.append("## Mainline per-family ranking")
    lines.append("")

    by_family: dict[str, list[dict]] = defaultdict(list)
    for c in scored_mainline:
        by_family[c["family"]].append(c)
    for family in sorted(by_family.keys()):
        cands = by_family[family]
        n_sel = sum(1 for c in cands if c["idx"] in selected_idxs)
        n_err = sum(1 for c in cands if c.get("error"))
        lines.append(f"### Family: `{family}`")
        lines.append(f"- Candidates: {len(cands)}")
        lines.append(f"- Selected: {n_sel}")
        lines.append(f"- Errors: {n_err}")
        lines.append("")
        cands_sorted = sorted(
            cands, key=lambda c: c["scoring"]["total_score"], reverse=True
        )
        lines.append("| Rank | Idx | Score | mean_abs_u | frac(|u|>=5e-3) | Status |")
        lines.append("|------|-----|-------|-----------|-----------------|--------|")
        for rank, c in enumerate(cands_sorted, 1):
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

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Activation-report diagnostics
# ---------------------------------------------------------------------------


def _stage_diagnostics(
    c: dict[str, Any]
) -> dict[str, Any]:
    """Return per-stage diagnostic arrays for a scored candidate.

    Extracts c_t, u_tr_cap_t, U_safe_ref_t, and u_ref_used_t from the
    schedule, plus the stage-level mean reward for event-conditioning.
    """
    if c.get("schedule") is None:
        return {}
    s = c["schedule"]
    pilot = c.get("pilot_data")
    u_ref_used = np.asarray(s["u_ref_used_t"], dtype=np.float64)
    u_tr_cap = np.asarray(s["u_tr_cap_t"], dtype=np.float64)
    U_safe_ref = np.asarray(s["U_safe_ref_t"], dtype=np.float64)
    xi_ref = np.asarray(s["xi_ref_t"], dtype=np.float64)
    gamma_base = float(s["gamma_base"])

    # Recompute c_t from pilot data + tau_n=200 for report consistency
    if pilot is not None:
        n_by_stage = np.asarray(pilot["n_by_stage"], dtype=np.float64)
        p_align = np.asarray(pilot["p_align_by_stage"], dtype=np.float64)
        tau_n = 200.0
        T = len(u_ref_used)
        T_p = min(T, len(p_align), len(n_by_stage))
        n_slice = n_by_stage[:T_p] if T_p > 0 else np.zeros(0)
        p_slice = np.maximum(p_align[:T_p], 0.0) if T_p > 0 else np.zeros(0)
        denom = np.where(n_slice + tau_n > 0, n_slice + tau_n, 1.0)
        c_t = np.clip((n_slice / denom) * np.sqrt(p_slice), 0.0, 1.0)
    else:
        c_t = np.zeros(len(u_ref_used))

    # Stage mean reward (for event conditioning). Compute per-stage mean
    # r_t across pilot episodes from the margins (r_t = margin + V*(s'));
    # without v_next stored at the pilot level, approximate using the
    # per-stage margin means as a reward proxy.
    # We use a reward proxy: at stages where p_align is high AND xi_ref
    # is high, the reward is activation-relevant. For event classification,
    # we rely on the raw margin magnitude since V*(s') subtracts the
    # value baseline.
    stage_mean_reward = np.zeros(len(u_ref_used))
    if pilot is not None:
        margins_by_stage = pilot["margins_by_stage"]
        for t, m_arr in enumerate(margins_by_stage):
            if t >= len(stage_mean_reward):
                break
            m_arr = np.asarray(m_arr, dtype=np.float64)
            stage_mean_reward[t] = (
                float(np.mean(np.abs(m_arr))) if m_arr.size > 0 else 0.0
            )

    return {
        "u_ref_used": u_ref_used,
        "u_tr_cap": u_tr_cap,
        "U_safe_ref": U_safe_ref,
        "xi_ref": xi_ref,
        "c_t": c_t,
        "gamma_base": gamma_base,
        "stage_mean_reward": stage_mean_reward,
        "reward_bound": float(c["cfg"].get("reward_bound", 1.0)),
    }


def write_gate_table_by_family(
    output_path: Path,
    scored_mainline: list[dict[str, Any]],
    scored_appendix: list[dict[str, Any]],
) -> None:
    """Markdown table with per-family activation diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    headers = [
        "family", "T", "n_ep", "tau_n", "mean_abs_u", "frac_ge5e3",
        "c_t_median", "u_tr_cap_median", "U_safe_median",
        "gate_pass", "suite_label",
    ]

    def _row(c: dict[str, Any], suite_label: str,
             n_ep: int, tau_n: int) -> list[str]:
        diag = _stage_diagnostics(c)
        m = c["scoring"]["raw_metrics"]
        mean_abs_u = m["mean_abs_u_pred"]
        frac = m["frac_u_ge_5e3"]
        c_t_med = (
            float(np.median(diag["c_t"])) if diag and diag["c_t"].size
            else float("nan")
        )
        u_tr_med = (
            float(np.median(diag["u_tr_cap"])) if diag and diag["u_tr_cap"].size
            else float("nan")
        )
        U_safe_med = (
            float(np.median(np.abs(diag["U_safe_ref"])))
            if diag and diag["U_safe_ref"].size else float("nan")
        )
        gate_pass = (mean_abs_u >= GATE_U_THRESHOLD) and (
            frac >= GATE_FRAC_THRESHOLD
        )
        return [
            c["family"], str(int(c["cfg"].get("horizon", 0))),
            str(n_ep), str(tau_n),
            f"{mean_abs_u:.6f}", f"{frac:.3f}",
            f"{c_t_med:.4f}", f"{u_tr_med:.6f}", f"{U_safe_med:.6f}",
            "YES" if gate_pass else "no", suite_label,
        ]

    # Best per-family for mainline and appendix
    def _best_by_family(
        cands: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_family: dict[str, dict] = {}
        for c in cands:
            fam = c["family"]
            cur = by_family.get(fam)
            if cur is None or (
                c["scoring"]["raw_metrics"]["mean_abs_u_pred"]
                > cur["scoring"]["raw_metrics"]["mean_abs_u_pred"]
            ):
                by_family[fam] = c
        return [by_family[k] for k in sorted(by_family.keys())]

    for c in _best_by_family(scored_mainline):
        rows.append(_row(c, "mainline", MAINLINE_PILOT_EPISODES, 200))
    # Appendix: per-family best at each T variant
    app_by_key: dict[tuple[str, int], dict] = {}
    for c in scored_appendix:
        key = (c["family"], int(c["cfg"].get("horizon", 0)))
        cur = app_by_key.get(key)
        if cur is None or (
            c["scoring"]["raw_metrics"]["mean_abs_u_pred"]
            > cur["scoring"]["raw_metrics"]["mean_abs_u_pred"]
        ):
            app_by_key[key] = c
    for key in sorted(app_by_key.keys()):
        c = app_by_key[key]
        rows.append(_row(c, "appendix_sanity", APPENDIX_PILOT_EPISODES, 200))

    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    header = (
        "# Phase IV-A Activation Gate — Per-Family Diagnostics\n\n"
        f"- Seed: {SEED}\n"
        f"- Gate condition: `mean_abs_u >= {GATE_U_THRESHOLD}` "
        f"AND `frac(|u| >= 5e-3) >= {GATE_FRAC_THRESHOLD:.0%}`\n"
        f"- Mainline: T = {MAINLINE_T}, n_ep = {MAINLINE_PILOT_EPISODES}, "
        f"tau_n = 200\n"
        f"- Appendix: T in {APPENDIX_T_VARIANTS}, "
        f"n_ep = {APPENDIX_PILOT_EPISODES}, tau_n = 200\n"
        "- One row per (family, T, suite). Best-by-mean_abs_u within "
        "each group.\n\n"
    )
    output_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output_path)


def write_global_diagnostics(
    output_path: Path,
    scored_mainline: list[dict[str, Any]],
    scored_appendix: list[dict[str, Any]],
    selected_mainline: list[dict[str, Any]],
    selected_appendix: list[dict[str, Any]],
) -> None:
    n_tasks = len(scored_mainline) + len(scored_appendix)
    main_best_u = max(
        (c["scoring"]["raw_metrics"]["mean_abs_u_pred"] for c in scored_mainline),
        default=0.0,
    )
    main_best_frac = max(
        (c["scoring"]["raw_metrics"]["frac_u_ge_5e3"] for c in scored_mainline),
        default=0.0,
    )
    main_mean_dd = (
        float(np.mean([
            c["scoring"]["raw_metrics"]["mean_abs_delta_d_pred"]
            for c in scored_mainline
        ])) if scored_mainline else 0.0
    )
    main_mean_tg = (
        float(np.mean([
            c["scoring"]["raw_metrics"]["mean_abs_target_gap_norm"]
            for c in scored_mainline
        ])) if scored_mainline else 0.0
    )
    app_best_u = max(
        (c["scoring"]["raw_metrics"]["mean_abs_u_pred"] for c in scored_appendix),
        default=0.0,
    )

    out = {
        "n_tasks_evaluated": n_tasks,
        "n_mainline_scored": len(scored_mainline),
        "n_appendix_scored": len(scored_appendix),
        "n_mainline_selected": len(selected_mainline),
        "n_appendix_selected": len(selected_appendix),
        "mainline_best_mean_abs_u": main_best_u,
        "mainline_best_frac_ge5e3": main_best_frac,
        "mainline_mean_abs_delta_eff_discount": main_mean_dd,
        "mainline_mean_abs_target_gap_norm": main_mean_tg,
        "appendix_best_mean_abs_u": app_best_u,
        "gate_conditions_met": {
            "mean_abs_u": bool(main_best_u >= GATE_U_THRESHOLD),
            "frac_ge5e3": bool(main_best_frac >= GATE_FRAC_THRESHOLD),
        },
        "gate_threshold_u": GATE_U_THRESHOLD,
        "gate_threshold_frac": GATE_FRAC_THRESHOLD,
        "n_pilot_episodes_mainline": MAINLINE_PILOT_EPISODES,
        "n_pilot_episodes_appendix": APPENDIX_PILOT_EPISODES,
        "seed": SEED,
        "generated_at": _now_iso(),
        "git_sha": _git_sha(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, _make_serialisable(out))
    logger.info("Wrote %s", output_path)


def write_event_conditioned_diagnostics(
    output_path: Path,
    scored_mainline: list[dict[str, Any]],
) -> None:
    """Per-family breakdown of activation metrics on event vs non-event stages.

    An event stage is one where the mean absolute margin at that stage
    (a proxy for stage-level reward magnitude, using |r_t - V*(s')|) is
    > EVENT_REWARD_FRACTION * reward_bound.
    """
    per_family: dict[str, dict] = {}
    for c in scored_mainline:
        fam = c["family"]
        diag = _stage_diagnostics(c)
        if not diag:
            continue
        u_ref = diag["u_ref_used"]
        stage_r = diag["stage_mean_reward"]
        rb = max(diag["reward_bound"], 1e-8)
        event_mask = stage_r > EVENT_REWARD_FRACTION * rb
        nonevent_mask = ~event_mask

        slot = per_family.setdefault(fam, {
            "event_u": [], "nonevent_u": [],
            "event_frac_hit": 0, "nonevent_frac_hit": 0,
            "event_n": 0, "nonevent_n": 0,
        })
        slot["event_u"].extend(u_ref[event_mask].tolist())
        slot["nonevent_u"].extend(u_ref[nonevent_mask].tolist())
        slot["event_frac_hit"] += int(
            np.sum(np.abs(u_ref[event_mask]) >= GATE_U_THRESHOLD)
        )
        slot["nonevent_frac_hit"] += int(
            np.sum(np.abs(u_ref[nonevent_mask]) >= GATE_U_THRESHOLD)
        )
        slot["event_n"] += int(np.sum(event_mask))
        slot["nonevent_n"] += int(np.sum(nonevent_mask))

    out_list = []
    for fam in sorted(per_family.keys()):
        s = per_family[fam]
        event_u = np.asarray(s["event_u"], dtype=np.float64)
        nonev_u = np.asarray(s["nonevent_u"], dtype=np.float64)
        out_list.append({
            "family": fam,
            "event_stages": {
                "mean_abs_u": (
                    float(np.mean(np.abs(event_u)))
                    if event_u.size else 0.0
                ),
                "frac_ge5e3": (
                    float(s["event_frac_hit"] / s["event_n"])
                    if s["event_n"] > 0 else 0.0
                ),
                "n_stages": int(s["event_n"]),
            },
            "non_event_stages": {
                "mean_abs_u": (
                    float(np.mean(np.abs(nonev_u)))
                    if nonev_u.size else 0.0
                ),
                "frac_ge5e3": (
                    float(s["nonevent_frac_hit"] / s["nonevent_n"])
                    if s["nonevent_n"] > 0 else 0.0
                ),
                "n_stages": int(s["nonevent_n"]),
            },
        })
    payload = {
        "event_reward_fraction": EVENT_REWARD_FRACTION,
        "notes": (
            "Event stage = |mean stage margin| > "
            f"{EVENT_REWARD_FRACTION} * reward_bound. "
            "Aggregated across all mainline candidates per family."
        ),
        "families": out_list,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, _make_serialisable(payload))
    logger.info("Wrote %s", output_path)


def write_binding_cap_summary(
    output_path: Path,
    scored_mainline: list[dict[str, Any]],
) -> None:
    """Summarise whether trust-region or safe-reference cap is binding."""
    tol = 1e-10
    totals = {"trust": 0, "safe": 0, "neither": 0, "total": 0}
    per_family: dict[str, dict] = {}
    u_tr_all: list[float] = []
    U_safe_all: list[float] = []

    for c in scored_mainline:
        diag = _stage_diagnostics(c)
        if not diag:
            continue
        u_ref = diag["u_ref_used"]
        u_tr = diag["u_tr_cap"]
        U_abs = np.abs(diag["U_safe_ref"])
        trust_mask = (np.abs(u_ref - u_tr) <= tol) & (u_tr <= U_abs + tol)
        safe_mask = (np.abs(u_ref - U_abs) <= tol) & (U_abs < u_tr - tol)
        neither_mask = ~(trust_mask | safe_mask)
        n_trust = int(np.sum(trust_mask))
        n_safe = int(np.sum(safe_mask))
        n_neither = int(np.sum(neither_mask))
        n_total = int(len(u_ref))

        totals["trust"] += n_trust
        totals["safe"] += n_safe
        totals["neither"] += n_neither
        totals["total"] += n_total

        u_tr_all.extend(u_tr.tolist())
        U_safe_all.extend(U_abs.tolist())

        fam = c["family"]
        s = per_family.setdefault(fam, {
            "trust": 0, "safe": 0, "neither": 0, "total": 0,
            "u_tr": [], "U_safe": [],
        })
        s["trust"] += n_trust
        s["safe"] += n_safe
        s["neither"] += n_neither
        s["total"] += n_total
        s["u_tr"].extend(u_tr.tolist())
        s["U_safe"].extend(U_abs.tolist())

    u_tr_arr = np.asarray(u_tr_all, dtype=np.float64)
    U_safe_arr = np.asarray(U_safe_all, dtype=np.float64)
    u_tr_median = float(np.median(u_tr_arr)) if u_tr_arr.size else 0.0
    U_safe_median = float(np.median(U_safe_arr)) if U_safe_arr.size else 0.0
    # Avoid inf / NaN which break strict JSON parsers (e.g. jq, js).
    # Use u_tr_cap *positive-tail mean* as the denominator if the median is 0
    # (which happens when many stages have zero pilot coverage).
    if u_tr_median > 0:
        ratio: float | None = U_safe_median / u_tr_median
    else:
        pos = u_tr_arr[u_tr_arr > 0.0]
        if pos.size > 0:
            u_tr_pos_median = float(np.median(pos))
            ratio = U_safe_median / u_tr_pos_median
        else:
            ratio = None  # JSON null
    ratio_str = f"{ratio:.1f}x" if ratio is not None else "inf (u_tr_cap_median=0)"

    trust_frac = (
        totals["trust"] / totals["total"] if totals["total"] > 0 else 0.0
    )
    by_family_out = []
    for fam in sorted(per_family.keys()):
        s = per_family[fam]
        utra = np.asarray(s["u_tr"], dtype=np.float64)
        usa = np.asarray(s["U_safe"], dtype=np.float64)
        by_family_out.append({
            "family": fam,
            "trust_clip_stages": s["trust"],
            "safe_clip_stages": s["safe"],
            "neither_stages": s["neither"],
            "total_stages": s["total"],
            "trust_clip_fraction": (
                s["trust"] / s["total"] if s["total"] > 0 else 0.0
            ),
            "u_tr_cap_median": (
                float(np.median(utra)) if utra.size else 0.0
            ),
            "U_safe_ref_median": (
                float(np.median(usa)) if usa.size else 0.0
            ),
        })

    payload = {
        "total_stages_evaluated": totals["total"],
        "trust_clip_stages": totals["trust"],
        "safe_clip_stages": totals["safe"],
        "neither_stages": totals["neither"],
        "trust_clip_fraction": trust_frac,
        "u_tr_cap_median": u_tr_median,
        "U_safe_ref_median": U_safe_median,
        "U_safe_over_u_tr_median_ratio": ratio,
        "conclusion": (
            f"Trust-region cap is binding on {trust_frac:.1%} of stages. "
            f"U_safe_ref median ({U_safe_median:.4f}) is {ratio_str} "
            f"larger than u_tr_cap median ({u_tr_median:.4f})."
        ),
        "by_family": by_family_out,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, _make_serialisable(payload))
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Frozen configs for Phase IV-B
# ---------------------------------------------------------------------------


def write_frozen_suite_config(
    output_path: Path,
    selected_mainline: list[dict[str, Any]],
    selected_appendix: list[dict[str, Any]],
    top_n_mainline: int = 3,
    top_n_appendix: int = 4,
) -> None:
    """Write activation_suite.json used by Phase IV-B."""
    # Order mainline by mean_abs_u_pred desc, keep top N
    main_sorted = sorted(
        selected_mainline,
        key=lambda c: c["scoring"]["raw_metrics"]["mean_abs_u_pred"],
        reverse=True,
    )[:top_n_mainline]
    app_sorted = sorted(
        selected_appendix,
        key=lambda c: c["scoring"]["raw_metrics"]["mean_abs_u_pred"],
        reverse=True,
    )[:top_n_appendix]

    def _task_cfg_entry(c: dict[str, Any]) -> dict[str, Any]:
        e: dict[str, Any] = {
            "family": c["family"],
            "cfg": c["cfg"],
            "mean_abs_u_pred": c["scoring"]["raw_metrics"]["mean_abs_u_pred"],
            "frac_u_ge_5e3": c["scoring"]["raw_metrics"]["frac_u_ge_5e3"],
        }
        if c.get("schedule") is not None:
            e["gamma_base"] = c["schedule"].get("gamma_base")
        return e

    payload = {
        "suite_version": "1.0",
        "phase": "IV-A",
        "status": "frozen",
        "generated_by": "run_phase4a_mainline_rerun.py",
        "generated_at": _now_iso(),
        "git_sha": _git_sha(),
        "seed": SEED,
        "mainline": {
            "n_pilot_episodes": MAINLINE_PILOT_EPISODES,
            "tau_n": 200,
            "gamma": 0.95,
            "T_target": MAINLINE_T,
            "tasks": [_task_cfg_entry(c) for c in main_sorted],
        },
        "appendix": {
            "n_pilot_episodes": APPENDIX_PILOT_EPISODES,
            "tau_n": 200,
            "T_variants": APPENDIX_T_VARIANTS,
            "tasks": [_task_cfg_entry(c) for c in app_sorted],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, _make_serialisable(payload))
    logger.info("Wrote %s", output_path)


def write_gamma_matched_controls(
    output_path: Path,
    selected_mainline: list[dict[str, Any]],
) -> None:
    """Write gamma_matched_controls.json — classical controls paired with
    safe-operator mainline tasks."""
    controls = []
    seen_keys: set[tuple] = set()
    for c in selected_mainline:
        fam = c["family"]
        cfg = c["cfg"]
        gamma = float(cfg.get("gamma", 0.95))
        T = int(cfg.get("horizon", MAINLINE_T))
        key = (fam, gamma, T)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        controls.append({
            "family": fam,
            "gamma": gamma,
            "T": T,
            "algo": "VI",
            "n_episodes": MAINLINE_PILOT_EPISODES,
            "cfg": cfg,
        })

    payload = {
        "note": (
            "Classical controls matched to each safe-operator task by "
            "gamma and T. VI on the same (gamma, T, cfg) as the mainline "
            "safe run; the pairing is used for per-family overhead and "
            "effect-size reporting in Phase IV-B."
        ),
        "phase": "IV-A",
        "generated_by": "run_phase4a_mainline_rerun.py",
        "generated_at": _now_iso(),
        "git_sha": _git_sha(),
        "seed": SEED,
        "controls": controls,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, _make_serialisable(payload))
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    t_start = time.time()
    logger.info("Phase IV-A mainline activation-suite rerun")
    logger.info(
        "  seed=%d  mainline n_ep=%d  appendix n_ep=%d  mainline T=%d  appendix T in %s",
        SEED, MAINLINE_PILOT_EPISODES, APPENDIX_PILOT_EPISODES,
        MAINLINE_T, APPENDIX_T_VARIANTS,
    )

    full_grid = get_search_grid()
    mainline_grid = build_mainline_grid(full_grid, T_target=MAINLINE_T)
    appendix_grid = build_appendix_grid(
        full_grid, T_variants=APPENDIX_T_VARIANTS
    )

    # ---- Score mainline at 1000 episodes ----
    logger.info(
        "Scoring %d mainline candidates at n_pilot_episodes=%d...",
        len(mainline_grid), MAINLINE_PILOT_EPISODES,
    )
    t0 = time.time()
    scored_mainline = score_all_candidates(
        search_grid=mainline_grid,
        seed=SEED,
        n_pilot_episodes=MAINLINE_PILOT_EPISODES,
    )
    logger.info(
        "Mainline scoring complete in %.1fs (%d errors)",
        time.time() - t0,
        sum(1 for c in scored_mainline if c.get("error")),
    )

    # ---- Score appendix at 200 episodes ----
    logger.info(
        "Scoring %d appendix candidates at n_pilot_episodes=%d...",
        len(appendix_grid), APPENDIX_PILOT_EPISODES,
    )
    t0 = time.time()
    scored_appendix = score_all_candidates(
        search_grid=appendix_grid,
        seed=SEED,
        n_pilot_episodes=APPENDIX_PILOT_EPISODES,
    )
    logger.info(
        "Appendix scoring complete in %.1fs (%d errors)",
        time.time() - t0,
        sum(1 for c in scored_appendix if c.get("error")),
    )

    # ---- Suite selection ----
    selected_mainline = select_activation_suite(
        scored_mainline,
        min_per_family=1,
        max_per_family=2,
        min_mean_abs_u_pred=2e-3,
        min_frac_active_stages=0.05,
    )
    logger.info(
        "Mainline selected: %d tasks across %d families",
        len(selected_mainline),
        len({c["family"] for c in selected_mainline}),
    )

    selected_appendix = select_activation_suite(
        scored_appendix,
        min_per_family=1,
        max_per_family=2,
        min_mean_abs_u_pred=2e-3,
        min_frac_active_stages=0.05,
    )
    logger.info(
        "Appendix selected: %d tasks across %d families",
        len(selected_appendix),
        len({c["family"] for c in selected_appendix}),
    )

    # ---- Write task_search artifacts ----
    write_candidate_scores_csv(
        SEARCH_DIR / "candidate_scores.csv", scored_mainline
    )
    write_selected_tasks_json(
        SEARCH_DIR / "selected_tasks.json",
        selected_mainline,
        suite_type="mainline",
        n_pilot_episodes=MAINLINE_PILOT_EPISODES,
        tau_n=200,
        extra={
            "T_target": MAINLINE_T,
            "gamma": 0.95,
            "label": "mainline T=20 activation suite",
        },
    )
    write_selected_tasks_json(
        SEARCH_DIR / "selected_tasks_appendix.json",
        selected_appendix,
        suite_type="appendix_sanity",
        n_pilot_episodes=APPENDIX_PILOT_EPISODES,
        tau_n=200,
        extra={
            "label": "short-horizon high-activation sanity checks",
            "note": "Do not use as primary evidence for main paper claim",
            "T_variants": APPENDIX_T_VARIANTS,
        },
    )
    write_activation_report_md(
        SEARCH_DIR / "activation_search_report.md",
        scored_mainline, selected_mainline,
        scored_appendix, selected_appendix,
    )

    # ---- Write activation_report/ diagnostics ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_gate_table_by_family(
        REPORT_DIR / "gate_table_by_family.md",
        scored_mainline, scored_appendix,
    )
    write_global_diagnostics(
        REPORT_DIR / "global_diagnostics.json",
        scored_mainline, scored_appendix,
        selected_mainline, selected_appendix,
    )
    write_event_conditioned_diagnostics(
        REPORT_DIR / "event_conditioned_diagnostics.json",
        scored_mainline,
    )
    write_binding_cap_summary(
        REPORT_DIR / "binding_cap_summary.json",
        scored_mainline,
    )

    # ---- Write frozen configs for Phase IV-B ----
    write_frozen_suite_config(
        CONFIG_DIR / "activation_suite.json",
        selected_mainline, selected_appendix,
    )
    write_gamma_matched_controls(
        CONFIG_DIR / "gamma_matched_controls.json",
        selected_mainline,
    )

    elapsed = time.time() - t_start
    logger.info("Phase IV-A rerun complete in %.1fs", elapsed)

    # ---- Brief summary to stdout ----
    main_best_u = max(
        (c["scoring"]["raw_metrics"]["mean_abs_u_pred"]
         for c in scored_mainline),
        default=0.0,
    )
    main_best_frac = max(
        (c["scoring"]["raw_metrics"]["frac_u_ge_5e3"]
         for c in scored_mainline),
        default=0.0,
    )
    gate_ok = (
        main_best_u >= GATE_U_THRESHOLD
        and main_best_frac >= GATE_FRAC_THRESHOLD
    )
    logger.info(
        "Summary: mainline best mean_abs_u=%.5f  frac>=5e-3=%.3f  "
        "gate=%s",
        main_best_u, main_best_frac, "PASS" if gate_ok else "FAIL",
    )
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
