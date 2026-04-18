"""
Generate fallback and ablation beta schedules for Phase III.

Reads the default ``schedule.json`` for each task family and derives
the five ablation families required by spec S5.11:

1. ``beta_zero``          -- all-zero betas (classical collapse).
2. ``beta_constant_small`` -- constant |beta| = BETA_SMALL * sign, certified.
3. ``beta_constant_large`` -- constant |beta| = BETA_LARGE * sign, certified.
4. ``beta_raw_unclipped``  -- main beta_raw applied *without* clipping.
5. ``alpha_constant_grid`` -- constant alpha in {0.00, 0.02, 0.05, 0.10, 0.20}
                              with main beta_raw re-clipped to new certification.

All schedules are written alongside the default schedule:
    results/weighted_lse_dp/phase3/calibration/<family>/<schedule_name>.json

Spec reference: docs/specs/phase_III_safe_weighted_lse_experiments.md S5.11
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (experiments/weighted_lse_dp/calibration/ → repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.calibration.calibration_utils import (  # noqa: E402
    build_certification,
    clip_beta,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BETA_SMALL: float = 0.1   # |beta| for beta_constant_small (pre-clip)
BETA_LARGE: float = 2.0   # |beta| for beta_constant_large (pre-clip)
ALPHA_GRID: list[float] = [0.00, 0.02, 0.05, 0.10, 0.20]

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


# ---------------------------------------------------------------------------
# Helper: build a schedule dict from components
# ---------------------------------------------------------------------------

def _make_schedule(
    *,
    task_family: str,
    gamma: float,
    sign: int,
    source_phase: str,
    reward_bound: float,
    alpha_t: np.ndarray,
    kappa_t: np.ndarray,
    Bhat_t: np.ndarray,
    beta_raw_t: np.ndarray,
    beta_cap_t: np.ndarray,
    beta_used_t: np.ndarray,
    clip_active_t: list[bool],
    informativeness_t: list[float],
    d_target_t: list[float],
    calibration_source_path: str,
    calibration_hash: str,
    ablation_type: str,
    notes: str,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble a schedule dict in the S5.10 schema with ablation metadata."""
    T = len(beta_used_t)
    assert len(alpha_t) == T
    assert len(kappa_t) == T
    assert len(Bhat_t) == T + 1
    assert len(beta_raw_t) == T
    assert len(beta_cap_t) == T
    assert len(clip_active_t) == T

    sch: dict[str, Any] = {
        "task_family": task_family,
        "gamma": gamma,
        "sign": sign,
        "source_phase": source_phase,
        "reward_bound": reward_bound,
        "ablation_type": ablation_type,
        "alpha_t": alpha_t.tolist(),
        "kappa_t": kappa_t.tolist(),
        "Bhat_t": Bhat_t.tolist(),
        "margin_quantile": 0.75,
        "informativeness_t": informativeness_t,
        "d_target_t": d_target_t,
        "beta_raw_t": beta_raw_t.tolist(),
        "beta_cap_t": beta_cap_t.tolist(),
        "beta_used_t": beta_used_t.tolist(),
        "clip_active_t": clip_active_t,
        "calibration_source_path": calibration_source_path,
        "calibration_hash": calibration_hash,
        "notes": notes,
    }
    sch.update(extra)
    return sch


# ---------------------------------------------------------------------------
# Ablation generators
# ---------------------------------------------------------------------------

def gen_beta_zero(default: dict[str, Any]) -> dict[str, Any]:
    """All-zero beta schedule (classical collapse).

    alpha_t = 0 → kappa_t = gamma → beta_cap_t = 0 → beta_used_t = 0.
    Certification is the classical value-function envelope.
    """
    T = len(default["beta_used_t"])
    gamma = default["gamma"]
    R_max = default["reward_bound"]

    alpha_t = np.zeros(T)
    cert = build_certification(alpha_t, R_max, gamma)
    beta_raw_t = np.zeros(T)
    beta_used_t = np.zeros(T)
    clip_active_t = [False] * T

    return _make_schedule(
        task_family=default["task_family"],
        gamma=gamma,
        sign=0,
        source_phase=default["source_phase"],
        reward_bound=R_max,
        alpha_t=alpha_t,
        kappa_t=cert["kappa_t"],
        Bhat_t=cert["Bhat_t"],
        beta_raw_t=beta_raw_t,
        beta_cap_t=cert["beta_cap_t"],
        beta_used_t=beta_used_t,
        clip_active_t=clip_active_t,
        informativeness_t=[0.0] * T,
        d_target_t=[gamma] * T,
        calibration_source_path=default["calibration_source_path"],
        calibration_hash=default["calibration_hash"],
        ablation_type="beta_zero",
        notes="All-zero beta schedule: exact classical collapse. alpha_t=0 => beta_cap=0.",
    )


def gen_beta_constant(
    default: dict[str, Any],
    abs_beta: float,
    ablation_type: str,
) -> dict[str, Any]:
    """Constant |beta| schedule, re-clipped to main certification caps.

    Uses the same alpha_t (and hence the same kappa_t, Bhat, beta_cap) as
    the default schedule, but sets beta_raw to the constant value before
    applying the existing clip cap.
    """
    T = len(default["beta_used_t"])
    gamma = default["gamma"]
    sign = default["sign"]
    if sign == 0:
        sign = 1  # neutral fallback

    alpha_t = np.asarray(default["alpha_t"])
    kappa_t = np.asarray(default["kappa_t"])
    Bhat_t = np.asarray(default["Bhat_t"])
    beta_cap_t = np.asarray(default["beta_cap_t"])

    beta_raw_t = np.full(T, sign * abs_beta)
    beta_used_t, clip_active_t = clip_beta(beta_raw_t, beta_cap_t)

    return _make_schedule(
        task_family=default["task_family"],
        gamma=gamma,
        sign=sign,
        source_phase=default["source_phase"],
        reward_bound=default["reward_bound"],
        alpha_t=alpha_t,
        kappa_t=kappa_t,
        Bhat_t=Bhat_t,
        beta_raw_t=beta_raw_t,
        beta_cap_t=beta_cap_t,
        beta_used_t=beta_used_t,
        clip_active_t=clip_active_t,
        informativeness_t=default.get("informativeness_t", [0.5] * T),
        d_target_t=default.get("d_target_t", [gamma * 0.7] * T),
        calibration_source_path=default["calibration_source_path"],
        calibration_hash=default["calibration_hash"],
        ablation_type=ablation_type,
        notes=(
            f"Constant |beta|={abs_beta:.2f} (sign={sign:+d}) pre-clipping, "
            f"re-clipped to main schedule's beta_cap_t. "
            f"Same alpha_t/kappa_t as default schedule."
        ),
        constant_abs_beta=abs_beta,
    )


def gen_beta_raw_unclipped(default: dict[str, Any]) -> dict[str, Any]:
    """Raw beta schedule with clipping DISABLED.

    beta_used_t = beta_raw_t from the default schedule.
    beta_cap_t is set to |beta_raw_t| + 1.0 (guarantees no actual clipping).
    For appendix use only.
    """
    T = len(default["beta_used_t"])
    gamma = default["gamma"]
    sign = default["sign"]

    alpha_t = np.asarray(default["alpha_t"])
    kappa_t = np.asarray(default["kappa_t"])
    Bhat_t = np.asarray(default["Bhat_t"])
    beta_raw_t = np.asarray(default["beta_raw_t"])

    # Set cap strictly above |beta_raw| so no clipping occurs
    beta_cap_t = np.abs(beta_raw_t) + 1.0
    beta_used_t = beta_raw_t.copy()
    clip_active_t = [False] * T  # no clipping by construction

    return _make_schedule(
        task_family=default["task_family"],
        gamma=gamma,
        sign=sign,
        source_phase=default["source_phase"],
        reward_bound=default["reward_bound"],
        alpha_t=alpha_t,
        kappa_t=kappa_t,
        Bhat_t=Bhat_t,
        beta_raw_t=beta_raw_t,
        beta_cap_t=beta_cap_t,
        beta_used_t=beta_used_t,
        clip_active_t=clip_active_t,
        informativeness_t=default.get("informativeness_t", [0.5] * T),
        d_target_t=default.get("d_target_t", [gamma * 0.7] * T),
        calibration_source_path=default["calibration_source_path"],
        calibration_hash=default["calibration_hash"],
        ablation_type="beta_raw_unclipped",
        notes=(
            "Raw (unclipped) schedule: beta_used = beta_raw from the default "
            "schedule; certification clipping is disabled. "
            "Appendix use only — no contraction guarantee."
        ),
    )


def gen_alpha_constant(
    default: dict[str, Any],
    alpha_val: float,
) -> dict[str, Any]:
    """Constant-alpha schedule with main beta_raw re-clipped to new certification.

    Recomputes kappa_t, Bhat_t, beta_cap_t from constant alpha_val, then
    re-clips the main schedule's beta_raw_t.  This isolates the effect of
    the headroom parameter alpha on the deployed betas.
    """
    T = len(default["beta_used_t"])
    gamma = default["gamma"]
    sign = default["sign"]
    R_max = default["reward_bound"]

    alpha_t = np.full(T, alpha_val)
    cert = build_certification(alpha_t, R_max, gamma)

    beta_raw_t = np.asarray(default["beta_raw_t"])
    beta_used_t, clip_active_t = clip_beta(beta_raw_t, cert["beta_cap_t"])

    return _make_schedule(
        task_family=default["task_family"],
        gamma=gamma,
        sign=sign,
        source_phase=default["source_phase"],
        reward_bound=R_max,
        alpha_t=alpha_t,
        kappa_t=cert["kappa_t"],
        Bhat_t=cert["Bhat_t"],
        beta_raw_t=beta_raw_t,
        beta_cap_t=cert["beta_cap_t"],
        beta_used_t=beta_used_t,
        clip_active_t=clip_active_t,
        informativeness_t=default.get("informativeness_t", [0.5] * T),
        d_target_t=default.get("d_target_t", [gamma * 0.7] * T),
        calibration_source_path=default["calibration_source_path"],
        calibration_hash=default["calibration_hash"],
        ablation_type=f"alpha_constant_{alpha_val:.2f}",
        notes=(
            f"Constant alpha={alpha_val:.2f} schedule. Recomputes certification "
            f"from constant alpha, re-clips main beta_raw_t. "
            f"alpha=0.00 collapses to classical (beta_cap=0, beta_used=0)."
        ),
        constant_alpha=alpha_val,
    )


# ---------------------------------------------------------------------------
# Per-family generation
# ---------------------------------------------------------------------------

def generate_ablations_for_family(
    default_path: pathlib.Path,
    out_dir: pathlib.Path,
    verbose: bool = True,
) -> None:
    """Generate all S5.11 ablation schedules for one task family."""
    with open(default_path) as f:
        default = json.load(f)

    family = default["task_family"]
    T = len(default["beta_used_t"])
    sign = default["sign"]

    schedules: list[tuple[str, dict[str, Any]]] = []

    # 1. beta_zero
    schedules.append(("beta_zero", gen_beta_zero(default)))

    # 2. beta_constant_small
    schedules.append((
        "beta_constant_small",
        gen_beta_constant(default, BETA_SMALL, "beta_constant_small"),
    ))

    # 3. beta_constant_large
    schedules.append((
        "beta_constant_large",
        gen_beta_constant(default, BETA_LARGE, "beta_constant_large"),
    ))

    # 4. beta_raw_unclipped
    schedules.append(("beta_raw_unclipped", gen_beta_raw_unclipped(default)))

    # 5. alpha_constant_grid — one file per alpha value
    for alpha_val in ALPHA_GRID:
        name = f"alpha_{alpha_val:.2f}"
        schedules.append((name, gen_alpha_constant(default, alpha_val)))

    # Write all schedules
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sch in schedules:
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(sch, f, indent=2)

    if verbose:
        print(f"{family} (T={T}, sign={sign:+d}):")
        for name, sch in schedules:
            n_clip = sum(sch["clip_active_t"])
            b_min = min(sch["beta_used_t"])
            b_max = max(sch["beta_used_t"])
            print(
                f"  {name:<28} beta_used=[{b_min:+.4f}, {b_max:+.4f}]  "
                f"clip={n_clip}/{T}"
            )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate S5.11 ablation schedules for all task families."
    )
    parser.add_argument(
        "--cal-dir",
        default=str(
            _REPO_ROOT / "results" / "weighted_lse_dp" / "phase3" / "calibration"
        ),
        help="Directory containing per-family default schedule.json files.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-family diagnostic output."
    )
    args = parser.parse_args()

    cal_dir = pathlib.Path(args.cal_dir)
    n_ok = 0
    n_fail = 0

    for family in TASK_FAMILIES:
        default_path = cal_dir / family / "schedule.json"
        if not default_path.exists():
            print(f"SKIP  {family}: no default schedule.json at {default_path}")
            n_fail += 1
            continue
        try:
            generate_ablations_for_family(
                default_path=default_path,
                out_dir=cal_dir / family,
                verbose=not args.quiet,
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {family}: {exc}")
            n_fail += 1

    print(f"\nDone: {n_ok} families OK, {n_fail} failed.")


if __name__ == "__main__":
    main()
