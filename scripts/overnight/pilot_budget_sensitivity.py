"""Phase IV-A pilot-budget sensitivity study.

Runs a controlled grid over (task_family_config, n_pilot_episodes, tau_n)
to diagnose whether the activation gate fails due to restrictive
operator geometry or due to under-powered pilot statistics driving the
trust-region confidence cap below the gate threshold.

Gate condition: ``mean_abs_u >= 5e-3`` on at least one task family.
Current status (200 pilot episodes, tau_n=200): best observed
``mean_abs_u = 0.00356``.

Trust-region confidence cap formula:

    c_t = (n_t / (n_t + tau_n)) * sqrt(p_align_t)     # n_t per stage
    eps_tr = c_t * eps_design(u_target, gamma_base)
    u_tr_cap = solve_u_tr_cap(eps_tr, gamma_base)
    u_ref_used = min(u_target, u_tr_cap, |U_safe_ref|)

The script intentionally does NOT use any safe-return performance for
selection and does NOT touch the gate threshold or the main operator.

Output
------
results/weighted_lse_dp/phase4/pilot_budget_sensitivity/
    results.json       -- schema_version=1, one entry per condition
    summary_table.md   -- human-readable table over all conditions

Run
---
    PYTHONPATH=. python3 scripts/overnight/pilot_budget_sensitivity.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Project root on PYTHONPATH
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.seeds import seed_everything  # noqa: E402
from experiments.weighted_lse_dp.geometry.phase4_calibration_v3 import (  # noqa: E402
    build_schedule_v3_from_pilot,
)
from experiments.weighted_lse_dp.geometry.task_activation_search import (  # noqa: E402
    run_classical_pilot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("pilot_budget_sensitivity")
# Silence verbose INFO from the activation-search pipeline
logging.getLogger(
    "experiments.weighted_lse_dp.geometry.task_activation_search"
).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SELECTED_TASKS_JSON = (
    _REPO_ROOT
    / "results"
    / "weighted_lse_dp"
    / "phase4"
    / "task_search"
    / "selected_tasks.json"
)

OUT_DIR = (
    _REPO_ROOT
    / "results"
    / "weighted_lse_dp"
    / "phase4"
    / "pilot_budget_sensitivity"
)

RESULTS_JSON = OUT_DIR / "results.json"
SUMMARY_MD = OUT_DIR / "summary_table.md"

SCHEMA_VERSION = 1
GATE_THRESHOLD = 5e-3
INFORMATIVE_MIN = 0.05  # xi_ref * sqrt(p_align) >= 0.05

PILOT_EPISODES_GRID: list[int] = [200, 500, 1000, 2000]
TAU_N_GRID: list[float] = [200.0, 100.0, 50.0]
SHORTER_HORIZONS: list[int] = [5, 10]

PILOT_SEED = 42


# ---------------------------------------------------------------------------
# Loading selected task configs
# ---------------------------------------------------------------------------


def load_selected_task_cfgs() -> list[dict[str, Any]]:
    """Load unique task configs from the Phase IV task search output.

    Deduplicates on (family, gamma, horizon, identifying params) to
    avoid running multiple near-identical mainline configs.
    """
    if not SELECTED_TASKS_JSON.exists():
        raise FileNotFoundError(
            f"selected_tasks.json not found at {SELECTED_TASKS_JSON}"
        )

    with open(SELECTED_TASKS_JSON) as f:
        raw = json.load(f)

    # The selected_tasks.json schema is a dict with a top-level "tasks" key
    # (or, on legacy variants, "selected_tasks" or a bare list). Iterating
    # over the dict directly yields keys (strings), not entries.
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = raw.get("tasks") or raw.get("selected_tasks") or []
    else:
        entries = []

    # One representative cfg per (family, horizon) pair — take the first
    # (highest-ranked) entry per unique key.
    seen: set[tuple] = set()
    cfgs: list[dict[str, Any]] = []
    for entry in entries:
        cfg = dict(entry["cfg"])
        key = (cfg.get("family"), cfg.get("horizon"), cfg.get("gamma"))
        if key in seen:
            continue
        seen.add(key)
        cfgs.append(cfg)
    return cfgs


def derive_shorter_horizon_variants(
    cfg: dict[str, Any], horizons: list[int] = SHORTER_HORIZONS
) -> list[dict[str, Any]]:
    """Produce shorter-horizon variants by overriding ``horizon``.

    Only certain families support horizon overrides without breaking
    invariants (e.g. regime_shift pegs ``change_at_episode`` to a fraction
    of horizon, and chain_sparse_credit/grid_hazard just use horizon as
    episode length). We override conservatively and let downstream DP
    compute a (possibly trivially-zero) V*. If the task truly has no
    signal at T=5, the pilot will report near-zero margins, which is
    itself a useful data point for this sensitivity study.
    """
    family = cfg.get("family")
    if family not in {
        "chain_sparse_credit",
        "grid_hazard",
        "chain_catastrophe",
        "chain_jackpot",
        "regime_shift",
    }:
        return []

    variants: list[dict[str, Any]] = []
    for H in horizons:
        v = dict(cfg)
        v["horizon"] = H
        variants.append(v)
    return variants


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _triplet(
    arr: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Return (min, median, max) of arr[mask] (or arr if mask is None)."""
    if mask is not None:
        data = arr[mask]
    else:
        data = arr
    if data.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (float(np.min(data)), float(np.median(data)), float(np.max(data)))


def compute_sensitivity_metrics(
    pilot_data: dict[str, Any],
    schedule: dict[str, Any],
    reward_bound: float,
) -> dict[str, Any]:
    """Compute the 9 per-condition diagnostic metrics.

    Returns
    -------
    dict
        Keys listed in the task specification; ``binding_cap`` is
        ``"trust_clip"`` / ``"safe_clip"`` / ``"neither"``.
    """
    u_ref_used = np.asarray(schedule["u_ref_used_t"], dtype=np.float64)
    u_target = np.asarray(schedule["u_target_t"], dtype=np.float64)
    u_tr_cap = np.asarray(schedule["u_tr_cap_t"], dtype=np.float64)
    U_safe_ref = np.asarray(schedule["U_safe_ref_t"], dtype=np.float64)
    beta_used = np.asarray(schedule["beta_used_t"], dtype=np.float64)
    alpha_t = np.asarray(schedule["alpha_t"], dtype=np.float64)
    xi_ref = np.asarray(schedule["xi_ref_t"], dtype=np.float64)
    gamma_base = float(schedule["gamma_base"])

    p_align = np.asarray(pilot_data["p_align_by_stage"], dtype=np.float64)
    n_by_stage = np.asarray(pilot_data["n_by_stage"], dtype=np.float64)
    margins_by_stage = pilot_data["margins_by_stage"]

    T = int(u_ref_used.shape[0])
    T_p = min(T, int(p_align.shape[0]), int(n_by_stage.shape[0]))

    # Per-stage c_t (recompute from the actual scheduler inputs so the
    # report matches what the scheduler used, not a post-hoc formula).
    tau_n = float(schedule.get("tau_n_used", 200.0))  # may be overwritten below
    # The schedule does not persist tau_n itself; we pass it in via the
    # caller by reading it from the condition dict. We recompute c_t from
    # (n_by_stage, p_align, tau_n) directly via the formula.

    # ---- 1. mean |u_ref_used| over all stages ----
    mean_abs_u = float(np.mean(np.abs(u_ref_used)))

    # ---- 2. fraction of stages with |u_ref_used| >= 5e-3 ----
    frac_u_ge_5e3 = float(np.mean(np.abs(u_ref_used) >= GATE_THRESHOLD))

    # ---- 3. mean |delta effective discount| ~ mean |alpha_t * (1 - gamma_base)| ----
    mean_abs_delta_eff_discount = float(
        np.mean(np.abs(alpha_t) * (1.0 - gamma_base))
    )

    # ---- 4. mean |target_gap| / reward_bound (small-signal) ----
    # target_gap_t ~ (gamma / (2*(1+gamma))) * |beta_t| * E[margin^2 at stage t]
    coeff_tg = gamma_base / (2.0 * (1.0 + gamma_base))
    target_gap_terms: list[float] = []
    for t in range(min(T, len(margins_by_stage))):
        m_arr = np.asarray(margins_by_stage[t], dtype=np.float64)
        mean_m2 = float(np.mean(m_arr**2)) if m_arr.size > 0 else 0.0
        target_gap_terms.append(coeff_tg * abs(float(beta_used[t])) * mean_m2)
    mean_abs_target_gap = (
        float(np.mean(target_gap_terms)) if target_gap_terms else 0.0
    )
    rb = max(float(reward_bound), 1e-8)
    mean_abs_target_gap_norm = mean_abs_target_gap / rb

    # ---- informative-stage mask (xi_ref * sqrt(p_align) >= 0.05) ----
    if T_p > 0:
        xi_p = xi_ref[:T_p] * np.sqrt(np.maximum(p_align[:T_p], 0.0))
        informative = xi_p >= INFORMATIVE_MIN
    else:
        informative = np.zeros(0, dtype=bool)
    informative_frac = (
        float(np.mean(informative)) if informative.size > 0 else 0.0
    )

    # ---- 5. median n_t on informative stages ----
    if informative.any():
        median_n_t_informative = float(np.median(n_by_stage[:T_p][informative]))
    else:
        median_n_t_informative = float("nan")

    # ---- 6. c_t (min/median/max) on informative stages ----
    # Recomputed with the actual tau_n driver used to build this schedule.
    n_slice = n_by_stage[:T_p] if T_p > 0 else np.zeros(0)
    p_slice = np.maximum(p_align[:T_p], 0.0) if T_p > 0 else np.zeros(0)

    # c_t depends on tau_n, which the caller must supply as
    # schedule["_tau_n_used"]. Fall back to 200 if missing.
    tau_n_used = float(schedule.get("_tau_n_used", 200.0))
    if n_slice.size > 0:
        denom = np.where(n_slice + tau_n_used > 0, n_slice + tau_n_used, 1.0)
        c_t = np.clip((n_slice / denom) * np.sqrt(p_slice), 0.0, 1.0)
    else:
        c_t = np.zeros(0)
    if informative.any():
        c_t_min, c_t_med, c_t_max = _triplet(c_t, informative)
    else:
        c_t_min, c_t_med, c_t_max = _triplet(c_t) if c_t.size else (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    # ---- 7. u_tr_cap (min/median/max) across all stages ----
    u_tr_cap_min, u_tr_cap_med, u_tr_cap_max = _triplet(u_tr_cap)

    # ---- 8. |U_safe_ref| (min/median/max) across all stages ----
    U_safe_abs_min, U_safe_abs_med, U_safe_abs_max = _triplet(np.abs(U_safe_ref))

    # ---- 9. binding_cap ----
    # For each stage:
    #    trust-binding   iff u_ref_used == u_tr_cap  AND u_tr_cap <= |U_safe|
    #    safe-binding    iff u_ref_used == |U_safe|  AND |U_safe| <  u_tr_cap
    # "Majority" means at least half the stages.
    U_safe_abs = np.abs(U_safe_ref)
    tol = 1e-10
    trust_binding = (np.abs(u_ref_used - u_tr_cap) <= tol) & (
        u_tr_cap <= U_safe_abs + tol
    )
    safe_binding = (np.abs(u_ref_used - U_safe_abs) <= tol) & (
        U_safe_abs < u_tr_cap - tol
    )
    n_trust = int(np.sum(trust_binding))
    n_safe = int(np.sum(safe_binding))
    if T == 0:
        binding_cap = "neither"
    elif n_trust >= max(n_safe, 1) and n_trust >= T // 2:
        binding_cap = "trust_clip"
    elif n_safe > n_trust and n_safe >= T // 2:
        binding_cap = "safe_clip"
    elif n_trust > 0 or n_safe > 0:
        binding_cap = "trust_clip" if n_trust >= n_safe else "safe_clip"
    else:
        # Neither cap is binding: u_ref_used == u_target everywhere.
        binding_cap = "neither"

    metrics = {
        "mean_abs_u": mean_abs_u,
        "frac_u_ge_5e3": frac_u_ge_5e3,
        "mean_abs_delta_eff_discount": mean_abs_delta_eff_discount,
        "mean_abs_target_gap_norm": mean_abs_target_gap_norm,
        "median_n_t_informative": median_n_t_informative,
        "c_t_min_med_max": [c_t_min, c_t_med, c_t_max],
        "u_tr_cap_min_med_max": [u_tr_cap_min, u_tr_cap_med, u_tr_cap_max],
        "U_safe_ref_min_med_max": [
            U_safe_abs_min,
            U_safe_abs_med,
            U_safe_abs_max,
        ],
        "informative_stage_frac": informative_frac,
        "trust_binding_stages": n_trust,
        "safe_binding_stages": n_safe,
        "n_stages": T,
    }
    return metrics, binding_cap


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------


def run_one_condition(
    cfg: dict[str, Any],
    n_episodes: int,
    tau_n: float,
    seed: int = PILOT_SEED,
) -> dict[str, Any] | None:
    """Run a single (cfg, n_ep, tau_n) condition and compute metrics.

    Returns the condition dict (JSON-serializable). Returns ``None`` if
    the pilot or schedule build fails (the error is logged).
    """
    family = cfg.get("family", "unknown")
    horizon = int(cfg.get("horizon", 0))
    gamma = float(cfg.get("gamma", 0.95))

    t0 = time.time()
    try:
        seed_everything(seed)
        pilot = run_classical_pilot(cfg, seed=seed, n_episodes=n_episodes)
        resolved_cfg = pilot["resolved_cfg"]
        reward_bound = float(resolved_cfg.get("reward_bound", 1.0))
        r_max = reward_bound

        schedule = build_schedule_v3_from_pilot(
            pilot_data=pilot,
            r_max=r_max,
            gamma_base=gamma,
            gamma_eval=gamma,
            task_family=family,
            source_phase="pilot",
            notes=(
                f"pilot_budget_sensitivity cfg={family} T={horizon} "
                f"n_ep={n_episodes} tau_n={tau_n}"
            ),
            tau_n=tau_n,
        )
        # Tag the schedule with the tau_n the scheduler was called with, so
        # the metrics layer can recompute c_t against the right driver.
        schedule["_tau_n_used"] = float(tau_n)

        metrics, binding_cap = compute_sensitivity_metrics(
            pilot, schedule, reward_bound
        )
        dt = time.time() - t0

        gate_pass = bool(metrics["mean_abs_u"] >= GATE_THRESHOLD)

        entry: dict[str, Any] = {
            "family": family,
            "T": horizon,
            "gamma": gamma,
            "reward_bound": reward_bound,
            "n_episodes": n_episodes,
            "tau_n": float(tau_n),
            "cfg": {
                k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                for k, v in cfg.items()
            },
            "metrics": metrics,
            "binding_cap": binding_cap,
            "gate_pass": gate_pass,
            "wallclock_s": round(dt, 3),
        }
        logger.info(
            "family=%-22s T=%2d n_ep=%4d tau_n=%3d  "
            "mean_abs_u=%.6f  gate_pass=%s  bind=%s  "
            "n_inf_frac=%.2f  t=%.1fs",
            family,
            horizon,
            n_episodes,
            int(tau_n),
            metrics["mean_abs_u"],
            "YES" if gate_pass else "no ",
            binding_cap,
            metrics["informative_stage_frac"],
            dt,
        )
        return entry

    except Exception as exc:  # pragma: no cover -- diagnostic path
        logger.exception(
            "FAILED family=%s T=%d n_ep=%d tau_n=%s: %s",
            family,
            horizon,
            n_episodes,
            tau_n,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------


def build_condition_grid() -> list[tuple[dict[str, Any], int, float, str]]:
    """Return a list of (cfg, n_episodes, tau_n, tag) tuples for the run."""
    mainline_cfgs = load_selected_task_cfgs()

    # Add shorter-horizon variants per mainline cfg (T=5, T=10).
    short_cfgs: list[dict[str, Any]] = []
    for cfg in mainline_cfgs:
        short_cfgs.extend(derive_shorter_horizon_variants(cfg))

    all_cfgs: list[tuple[dict[str, Any], str]] = [
        (cfg, "mainline") for cfg in mainline_cfgs
    ] + [(cfg, "short_horizon") for cfg in short_cfgs]

    grid: list[tuple[dict[str, Any], int, float, str]] = []
    for cfg, tag in all_cfgs:
        for n_ep in PILOT_EPISODES_GRID:
            for tau_n in TAU_N_GRID:
                grid.append((cfg, n_ep, tau_n, tag))
    return grid


def format_triplet(t: list[float], fmt: str = "{:.4f}") -> str:
    """Format a [min, median, max] list as ``min/med/max``."""
    try:
        return "/".join(fmt.format(float(x)) for x in t)
    except Exception:
        return "nan/nan/nan"


def write_summary_table(conditions: list[dict[str, Any]], path: Path) -> None:
    """Write a human-readable markdown table with one row per condition."""
    headers = [
        "family",
        "T",
        "gamma",
        "n_ep",
        "tau_n",
        "tag",
        "mean_abs_u",
        "frac_u_ge_5e3",
        "mean_abs_d_eff_disc",
        "mean_abs_tg_norm",
        "med_n_t_inf",
        "c_t (min/med/max)",
        "u_tr_cap (min/med/max)",
        "|U_safe| (min/med/max)",
        "binding_cap",
        "gate_pass",
    ]
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]

    # Sort: mainline first (by family then T then n_ep then tau_n desc),
    # then short_horizon
    def sort_key(c: dict[str, Any]) -> tuple:
        return (
            0 if c.get("tag", "mainline") == "mainline" else 1,
            c.get("family", ""),
            int(c.get("T", 0)),
            int(c.get("n_episodes", 0)),
            -float(c.get("tau_n", 0.0)),
        )

    sorted_conds = sorted(conditions, key=sort_key)

    for c in sorted_conds:
        m = c["metrics"]
        row = [
            str(c.get("family", "")),
            str(int(c.get("T", 0))),
            f"{float(c.get('gamma', 0.0)):.2f}",
            str(int(c.get("n_episodes", 0))),
            str(int(c.get("tau_n", 0.0))),
            str(c.get("tag", "mainline")),
            f"{m['mean_abs_u']:.6f}",
            f"{m['frac_u_ge_5e3']:.3f}",
            f"{m['mean_abs_delta_eff_discount']:.6f}",
            f"{m['mean_abs_target_gap_norm']:.2e}",
            (
                f"{m['median_n_t_informative']:.1f}"
                if not np.isnan(m["median_n_t_informative"])
                else "nan"
            ),
            format_triplet(m["c_t_min_med_max"]),
            format_triplet(m["u_tr_cap_min_med_max"]),
            format_triplet(m["U_safe_ref_min_med_max"]),
            str(c.get("binding_cap", "neither")),
            "YES" if c.get("gate_pass", False) else "no",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # Header note
    header = (
        "# Phase IV-A Pilot-Budget Sensitivity\n"
        "\n"
        "- Gate condition: `mean_abs_u >= 5e-3` on at least one condition.\n"
        f"- Pilot seed: {PILOT_SEED}\n"
        f"- n_episodes grid: {PILOT_EPISODES_GRID}\n"
        f"- tau_n grid (mainline=200): {TAU_N_GRID}\n"
        "- Informative stages: `xi_ref * sqrt(p_align) >= 0.05`.\n"
        "- binding_cap: which cap governs `u_ref_used` on a majority of stages.\n"
        "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        f.write("\n".join(lines))
        f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logger.info("Loading selected task configs from %s", SELECTED_TASKS_JSON)
    grid = build_condition_grid()
    logger.info("Total conditions to run: %d", len(grid))
    logger.info(
        "pilot_episodes grid=%s  tau_n grid=%s  short_horizons=%s",
        PILOT_EPISODES_GRID,
        TAU_N_GRID,
        SHORTER_HORIZONS,
    )

    conditions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    t_start = time.time()

    for i, (cfg, n_ep, tau_n, tag) in enumerate(grid):
        logger.info(
            "[%3d/%3d] family=%s T=%s n_ep=%d tau_n=%s tag=%s",
            i + 1,
            len(grid),
            cfg.get("family"),
            cfg.get("horizon"),
            n_ep,
            int(tau_n),
            tag,
        )
        entry = run_one_condition(cfg, n_episodes=n_ep, tau_n=tau_n)
        if entry is None:
            failures.append(
                {
                    "cfg": cfg,
                    "n_episodes": n_ep,
                    "tau_n": tau_n,
                    "tag": tag,
                }
            )
            continue
        entry["tag"] = tag
        conditions.append(entry)

    total_s = time.time() - t_start
    logger.info(
        "Completed %d/%d conditions (%d failures) in %.1fs",
        len(conditions),
        len(grid),
        len(failures),
        total_s,
    )

    # Overall gate check
    any_gate_pass = any(c.get("gate_pass") for c in conditions)
    best = max(
        conditions, key=lambda c: c["metrics"]["mean_abs_u"], default=None
    )
    best_mean_abs_u = best["metrics"]["mean_abs_u"] if best else float("nan")

    out = {
        "schema_version": SCHEMA_VERSION,
        "gate_threshold": GATE_THRESHOLD,
        "pilot_seed": PILOT_SEED,
        "pilot_episodes_grid": PILOT_EPISODES_GRID,
        "tau_n_grid": TAU_N_GRID,
        "shorter_horizons": SHORTER_HORIZONS,
        "best_mean_abs_u": best_mean_abs_u,
        "any_gate_pass": bool(any_gate_pass),
        "n_conditions": len(conditions),
        "n_failures": len(failures),
        "wallclock_s": round(total_s, 2),
        "conditions": conditions,
        "failures": failures,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    logger.info("Wrote %s", RESULTS_JSON)

    write_summary_table(conditions, SUMMARY_MD)
    logger.info("Wrote %s", SUMMARY_MD)

    logger.info(
        "Best mean_abs_u = %.6f (threshold %.4f).  Any gate pass: %s",
        best_mean_abs_u,
        GATE_THRESHOLD,
        "YES" if any_gate_pass else "no",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
