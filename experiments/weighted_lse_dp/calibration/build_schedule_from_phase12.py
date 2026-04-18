"""
Build Phase III beta schedules from Phase I/II calibration statistics.

Usage::

    python build_schedule_from_phase12.py [--cal-dir PATH] [--out-dir PATH]
                                          [--lambda-min FLOAT] [--lambda-max FLOAT]
                                          [--alpha-min FLOAT] [--alpha-max FLOAT]
                                          [--source-phase PHASE]

Spec reference: docs/specs/phase_III_safe_weighted_lse_experiments.md S5.1--S5.10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap: repo root is three levels up from this file
# experiments / weighted_lse_dp / calibration / <this file>
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV, str(_THIS_DIR)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from calibration_utils import (
    load_calibration_json,
    compute_calibration_hash,
    get_authoritative_T,
    extract_stagewise_arrays,
    compute_representative_margin,
    compute_informativeness,
    compute_derivative_targets,
    compute_raw_beta,
    compute_headroom_fractions,
    build_certification,
    clip_beta,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_FAMILIES = [
    "chain_catastrophe",
    "chain_jackpot",
    "chain_regime_shift",
    "chain_sparse_long",
    "grid_hazard",
    "grid_regime_shift",
    "grid_sparse_goal",
    "taxi_bonus_shock",
]

_DEFAULT_CAL_DIR = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase2" / "calibration"
_DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "weighted_lse_dp" / "phase3" / "calibration"


# ---------------------------------------------------------------------------
# Git SHA helper (best-effort)
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_schedule_for_family(
    cal_path: pathlib.Path,
    out_dir: pathlib.Path,
    source_phase: str = "phase2",
    lambda_min: float = 0.10,
    lambda_max: float = 0.50,
    alpha_min: float = 0.02,
    alpha_max: float = 0.10,
    min_freq_threshold: float = 0.01,
    reward_bound: float | None = None,
) -> dict:
    """Build and save a schedule.json for one task family.

    Parameters
    ----------
    cal_path : path to the Phase I/II calibration JSON
    out_dir  : base output directory (family subdirectory created automatically)
    source_phase : "phase1", "phase2", or "phase12"
    lambda_min, lambda_max : derivative-target interpolation bounds
    alpha_min, alpha_max   : headroom fraction bounds
    min_freq_threshold     : sparse-data fallback threshold
    reward_bound : explicit reward bound from config; if None, falls back
        to ``empirical_r_max`` from the calibration JSON with a warning.

    Returns
    -------
    schedule : dict  (also written to ``out_dir/<family>/schedule.json``)
    """
    # --- Load ---
    cal = load_calibration_json(cal_path)
    cal_hash = compute_calibration_hash(cal_path)

    family = cal["task_family"]
    gamma = float(cal["nominal_gamma"])
    sign = int(cal["recommended_task_sign"])

    if reward_bound is not None:
        R_max = float(reward_bound)
    else:
        import warnings
        R_max = float(cal["empirical_r_max"])
        warnings.warn(
            f"{family}: reward_bound not provided in config, falling back to "
            f"empirical_r_max={R_max} from calibration JSON. Set reward_bound "
            f"in the task config for deterministic results.",
            stacklevel=2,
        )
    T = get_authoritative_T(cal)

    # --- Extract stagewise arrays (first T entries only) ---
    sw = extract_stagewise_arrays(cal, T)
    aligned_margin_freq = sw["aligned_margin_freq"]
    aligned_positive_mean = sw["aligned_positive_mean"]

    # --- S5.4: representative margin ---
    m_star = compute_representative_margin(
        aligned_positive_mean, aligned_margin_freq, min_freq_threshold
    )

    # --- S5.5: informativeness ---
    I_t = compute_informativeness(aligned_positive_mean, aligned_margin_freq)

    # --- S5.6: derivative targets ---
    d_target = compute_derivative_targets(I_t, gamma, lambda_min, lambda_max)

    # --- S5.7: raw beta ---
    beta_raw = compute_raw_beta(m_star, d_target, gamma, sign)

    # --- S5.8: headroom fractions ---
    alpha_t = compute_headroom_fractions(I_t, alpha_min, alpha_max)

    # --- S5.9: certification and clip ---
    cert = build_certification(alpha_t, R_max, gamma)
    kappa_t = cert["kappa_t"]
    Bhat_t = cert["Bhat_t"]
    beta_cap_t = cert["beta_cap_t"]

    beta_used, clip_active = clip_beta(beta_raw, beta_cap_t)

    # --- S5.10: assemble schedule dict ---
    schedule = {
        "task_family": family,
        "gamma": gamma,
        "sign": sign,
        "source_phase": source_phase,
        "reward_bound": R_max,
        "alpha_t": alpha_t.tolist(),
        "kappa_t": kappa_t.tolist(),
        "Bhat_t": Bhat_t.tolist(),
        "margin_quantile": 0.75,
        "informativeness_t": I_t.tolist(),
        "d_target_t": d_target.tolist(),
        "beta_raw_t": beta_raw.tolist(),
        "beta_cap_t": beta_cap_t.tolist(),
        "beta_used_t": beta_used.tolist(),
        "clip_active_t": clip_active,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "calibration_source_path": str(cal_path),
        "calibration_hash": cal_hash,
        "git_sha": _get_git_sha(),
        "calibration_code_version": "0.1.0",
        "notes": f"derived from {source_phase} classical calibration summary",
    }

    # --- Write ---
    family_dir = out_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    out_path = family_dir / "schedule.json"
    with open(out_path, "w") as f:
        json.dump(schedule, f, indent=2)

    return schedule


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_diagnostic(schedule: dict) -> str:
    """One-line diagnostic for a schedule."""
    family = schedule["task_family"]
    beta_used = np.array(schedule["beta_used_t"])
    clip_active = schedule["clip_active_t"]
    T = len(beta_used)
    n_clip = sum(clip_active)
    beta_min = beta_used.min()
    beta_max = beta_used.max()
    sign = schedule["sign"]
    line = (
        f"{family:25s}: T={T:3d}, sign={sign:+d}, "
        f"beta_used range=[{beta_min:+.4f}, {beta_max:+.4f}], "
        f"clip_active={n_clip}/{T}"
    )
    if np.allclose(beta_used, 0.0):
        line += "  [ALL ZERO]"
    return line


def _verify_schedule(schedule: dict) -> list[str]:
    """Run sanity checks on a schedule dict.  Returns list of issues."""
    issues: list[str] = []
    T = len(schedule["beta_used_t"])

    # Length checks
    if len(schedule["Bhat_t"]) != T + 1:
        issues.append(
            f"Bhat_t length {len(schedule['Bhat_t'])} != T+1={T+1}"
        )
    for key in [
        "alpha_t", "kappa_t", "informativeness_t", "d_target_t",
        "beta_raw_t", "beta_cap_t", "beta_used_t", "clip_active_t",
    ]:
        if len(schedule[key]) != T:
            issues.append(f"{key} length {len(schedule[key])} != T={T}")

    # Type checks
    if not all(isinstance(v, bool) for v in schedule["clip_active_t"]):
        issues.append("clip_active_t contains non-bool entries")

    if not schedule["calibration_hash"]:
        issues.append("calibration_hash is empty")

    # Round-trip check: beta_used should equal clip(beta_raw, -cap, cap)
    beta_raw = np.array(schedule["beta_raw_t"])
    beta_cap = np.array(schedule["beta_cap_t"])
    beta_used = np.array(schedule["beta_used_t"])
    expected = np.clip(beta_raw, -beta_cap, beta_cap)
    if not np.allclose(beta_used, expected, atol=1e-12):
        max_diff = np.max(np.abs(beta_used - expected))
        issues.append(f"Round-trip clip check failed, max diff={max_diff:.2e}")

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase III beta schedules from Phase I/II calibration."
    )
    parser.add_argument(
        "--cal-dir", type=str, default=str(_DEFAULT_CAL_DIR),
        help="Directory containing <family>.json calibration files.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
        help="Output directory for schedule JSONs.",
    )
    parser.add_argument("--lambda-min", type=float, default=0.10)
    parser.add_argument("--lambda-max", type=float, default=0.50)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.10)
    parser.add_argument(
        "--source-phase", type=str, default="phase2",
        choices=["phase1", "phase2", "phase12"],
    )
    parser.add_argument(
        "--min-freq-threshold", type=float, default=0.01,
        help="Sparse-data guard: stages with aligned_margin_freq below "
             "this threshold fall back to beta_raw=0.",
    )
    parser.add_argument(
        "--suite-config", type=str, default=None,
        help="Path to paper_suite.json config; if provided, reward_bound "
             "is read from each task entry.",
    )
    args = parser.parse_args()

    cal_dir = pathlib.Path(args.cal_dir)
    out_dir = pathlib.Path(args.out_dir)

    # Load suite config for reward_bound if provided
    suite_tasks: dict[str, dict] = {}
    if args.suite_config is not None:
        suite_path = pathlib.Path(args.suite_config)
        with open(suite_path, "r") as f:
            suite = json.load(f)
        suite_tasks = suite.get("tasks", {})
        print(f"Suite config:                  {suite_path}")

    print(f"Calibration source directory: {cal_dir}")
    print(f"Output directory:             {out_dir}")
    print(f"Source phase:                  {args.source_phase}")
    print(f"lambda range:                  [{args.lambda_min}, {args.lambda_max}]")
    print(f"alpha range:                   [{args.alpha_min}, {args.alpha_max}]")
    print(f"min_freq_threshold:            {args.min_freq_threshold}")
    print()

    schedules: dict[str, dict] = {}
    all_ok = True

    for family in TASK_FAMILIES:
        cal_path = cal_dir / f"{family}.json"
        if not cal_path.exists():
            print(f"WARNING: {cal_path} not found, skipping {family}")
            continue

        # Read reward_bound from suite config if available
        family_reward_bound = suite_tasks.get(family, {}).get("reward_bound", None)

        schedule = build_schedule_for_family(
            cal_path=cal_path,
            out_dir=out_dir,
            source_phase=args.source_phase,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            min_freq_threshold=args.min_freq_threshold,
            reward_bound=family_reward_bound,
        )
        schedules[family] = schedule

        # Diagnostics
        diag = _print_diagnostic(schedule)
        issues = _verify_schedule(schedule)
        status = "OK" if not issues else "ISSUES"
        print(f"  [{status}] {diag}")
        for issue in issues:
            print(f"         ! {issue}")
            all_ok = False

    print()

    # --- Sparse-data fallback report ---
    print("--- Sparse-data fallback report ---")
    any_fallback = False
    for family, sched in schedules.items():
        beta_raw = np.array(sched["beta_raw_t"])
        zero_mask = np.abs(beta_raw) < 1e-15
        n_zero = int(zero_mask.sum())
        if n_zero > 0:
            # Find which stages
            zero_stages = np.where(zero_mask)[0].tolist()
            preview = zero_stages[:5]
            suffix = f"... ({n_zero} total)" if n_zero > 5 else ""
            print(f"  {family}: beta_raw=0 at stages {preview}{suffix}")
            any_fallback = True
    if not any_fallback:
        print("  (no sparse-data fallbacks triggered)")

    print()

    # --- Round-trip verification ---
    print("--- Round-trip verification (schedule -> clip -> reconstruct |beta_raw|) ---")
    for family, sched in schedules.items():
        beta_raw = np.array(sched["beta_raw_t"])
        beta_cap = np.array(sched["beta_cap_t"])
        beta_used = np.array(sched["beta_used_t"])
        reconstructed = np.clip(beta_raw, -beta_cap, beta_cap)
        max_err = float(np.max(np.abs(beta_used - reconstructed)))
        print(f"  {family}: max |beta_used - clip(beta_raw)| = {max_err:.2e}  "
              f"{'PASS' if max_err < 1e-12 else 'FAIL'}")

    print()

    # --- Output summary ---
    print("--- Output files ---")
    for family in schedules:
        out_path = out_dir / family / "schedule.json"
        print(f"  {out_path}")

    print()
    if all_ok:
        print("All schedules built and verified successfully.")
    else:
        print("Some schedules had issues -- see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
