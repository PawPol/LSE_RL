"""Phase IV-A S6: Schedule calibration v3.

Natural-shift-first design with trust-region caps and adaptive headroom.
Pipeline: margins -> xi_ref -> fixed-point headroom -> u_target -> trust cap
           -> u_ref_used -> theta -> beta -> schedule JSON.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from experiments.weighted_lse_dp.geometry.adaptive_headroom import (
    compute_informativeness,
    run_fixed_point,
)
from experiments.weighted_lse_dp.geometry.trust_region import (
    compute_trust_region_cap,
)

__all__ = [
    "select_sign",
    "build_schedule_v3",
    "build_schedule_v3_from_pilot",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantile_positive(arr: NDArray[np.float64], q: float) -> float:
    """Quantile of positive entries; returns 0.0 if none exist."""
    pos = arr[arr > 0.0]
    if len(pos) == 0:
        return 0.0
    return float(np.quantile(pos, q))


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


def _array_hash(arr: NDArray[np.float64]) -> str:
    """Deterministic SHA-256 of array bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sign selection
# ---------------------------------------------------------------------------


def select_sign(
    margins_by_stage: Sequence[NDArray[np.float64]],
    r_max: float,
    xi_min: float = 0.02,
    xi_max: float = 1.00,
) -> int:
    """Select the operator sign family (+1 or -1) that yields higher informativeness.

    For each candidate sign s in {-1, +1}:
        a_t^s = s * margins / max(r_max, 1e-8)
        xi_ref_t^s = clip(Q_0.75(a_t^s | a_t^s > 0), xi_min, xi_max)
        p_align_t^s = fraction of a_t^s > 0
        score_s = mean_t( xi_ref_t^s * sqrt(max(p_align_t^s, 0)) )

    Parameters
    ----------
    margins_by_stage : list of NDArray[np.float64]
        Raw (r - v_ref) margins per stage.
    r_max : float
        One-step reward bound.
    xi_min, xi_max : float
        Clipping bounds for xi_ref.

    Returns
    -------
    int
        +1 or -1, the sign family with higher score.
    """
    r_denom = max(r_max, 1e-8)
    best_sign = 1
    best_score = -np.inf

    for s in (-1, +1):
        xi_refs = []
        p_aligns = []
        for margins in margins_by_stage:
            a_t = s * np.asarray(margins, dtype=np.float64) / r_denom
            p_al = float(np.mean(a_t > 0))
            q75 = _quantile_positive(a_t, 0.75)
            xi_ref = float(np.clip(q75, xi_min, xi_max))
            xi_refs.append(xi_ref)
            p_aligns.append(p_al)

        score = float(np.mean(
            np.array(xi_refs) * np.sqrt(np.maximum(np.array(p_aligns), 0.0))
        ))
        if score > best_score:
            best_score = score
            best_sign = s

    return best_sign


# ---------------------------------------------------------------------------
# Core schedule builder
# ---------------------------------------------------------------------------


def build_schedule_v3(
    margins_by_stage: Sequence[NDArray[np.float64]],
    p_align_by_stage: Sequence[float],
    n_by_stage: Sequence[int],
    r_max: float,
    gamma_base: float,
    gamma_eval: float,
    sign_family: int,
    task_family: str = "unknown",
    source_phase: str = "pilot",
    notes: str = "",
    *,
    u_min: float = 0.002,
    u_max: float = 0.020,
    alpha_min: float = 0.05,
    alpha_max: float = 0.20,
    alpha_budget_max: float = 0.30,
    xi_min: float = 0.02,
    xi_max: float = 1.00,
    xi_floor: float = 1e-3,
    tau_n: float = 200.0,
    max_fixed_point_iters: int = 4,
) -> dict[str, Any]:
    """Build a v3 schedule with natural-shift-first calibration.

    Parameters
    ----------
    margins_by_stage : list of NDArray, length T
        Raw (r - v_ref) margins per stage.
    p_align_by_stage : list of float, length T
        Fraction of positive aligned margins per stage (pre-computed).
    n_by_stage : list of int, length T
        Sample counts per stage.
    r_max : float
        One-step reward bound.
    gamma_base : float
        Discount factor for the operator.
    gamma_eval : float
        Evaluation discount (for matched-control identification).
    sign_family : int
        +1 or -1, chosen family sign.
    task_family : str
        Name of the task family.
    source_phase : str
        Source phase label ("phase3", "phase12", or "pilot").
    notes : str
        Free-text notes.
    u_min, u_max : float
        Natural-shift target range.
    alpha_min, alpha_max : float
        Headroom interpolation range.
    alpha_budget_max : float
        Hard ceiling on alpha.
    xi_min, xi_max : float
        Clipping bounds for xi_ref.
    xi_floor : float
        Floor for xi_ref in beta computation.
    tau_n : float
        Trust-region confidence parameter.
    max_fixed_point_iters : int
        Maximum fixed-point iterations.

    Returns
    -------
    dict
        Schedule v3 JSON-serializable dict.
    """
    T = len(margins_by_stage)
    r_denom = max(r_max, 1e-8)

    # ------------------------------------------------------------------
    # Step 1: Compute xi_ref per stage (pass 1) using r_max as denominator.
    # This is a bootstrap estimate; pass 2 below refines using true A_t.
    # ------------------------------------------------------------------
    xi_ref_arr = np.zeros(T, dtype=np.float64)
    for t in range(T):
        a_t = sign_family * np.asarray(margins_by_stage[t], dtype=np.float64) / r_denom
        q75 = _quantile_positive(a_t, 0.75)
        # np.clip(0.0, xi_min, xi_max) == xi_min (xi_min > 0), so the
        # q75 == 0.0 case is already handled by the clip.
        xi_ref_arr[t] = float(np.clip(q75, xi_min, xi_max))

    p_align_arr = np.asarray(p_align_by_stage, dtype=np.float64)
    n_arr = np.asarray(n_by_stage, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 2: Fixed-point headroom iteration (pass 1 — bootstrap xi_ref)
    # u_target is precomputed so the fixed-point can use the spec S6.7
    # feasibility criterion (u_target > U_safe_ref triggers alpha bump).
    # ------------------------------------------------------------------
    I_t_pass1 = compute_informativeness(xi_ref_arr, p_align_arr)
    u_target_pass1 = u_min + (u_max - u_min) * I_t_pass1

    fp_result = run_fixed_point(
        xi_ref_t=xi_ref_arr,
        p_align_t=p_align_arr,
        r_max=r_max,
        gamma_base=gamma_base,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_budget_max=alpha_budget_max,
        u_target_t=u_target_pass1,
        max_iters=max_fixed_point_iters,
    )
    A_t_pass1 = fp_result["A_t"]          # shape (T,)

    # ------------------------------------------------------------------
    # Step 2b: Refine xi_ref using natural coordinates (pass 2).
    # Spec: xi_ref_t = Q75(sign * margin / A_t, margin > 0).
    # A_t was not available for pass 1; now recompute with true A_t.
    # ------------------------------------------------------------------
    xi_ref_arr_refined = np.zeros(T, dtype=np.float64)
    for t in range(T):
        a_denom = max(float(A_t_pass1[t]), 1e-8)
        a_t = sign_family * np.asarray(margins_by_stage[t], dtype=np.float64) / a_denom
        q75 = _quantile_positive(a_t, 0.75)
        # np.clip already maps q75 == 0.0 to xi_min (see pass-1 note above).
        xi_ref_arr_refined[t] = float(np.clip(q75, xi_min, xi_max))

    # Only use refined xi_ref if it differs meaningfully (avoids churn when
    # A_t ≈ r_max, e.g. at horizon boundary where Bhat=0).
    if not np.allclose(xi_ref_arr_refined, xi_ref_arr, atol=1e-6):
        xi_ref_arr = xi_ref_arr_refined
        I_t_pass2 = compute_informativeness(xi_ref_arr, p_align_arr)
        u_target_pass2 = u_min + (u_max - u_min) * I_t_pass2
        fp_result = run_fixed_point(
            xi_ref_t=xi_ref_arr,
            p_align_t=p_align_arr,
            r_max=r_max,
            gamma_base=gamma_base,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_budget_max=alpha_budget_max,
            u_target_t=u_target_pass2,
            max_iters=max_fixed_point_iters,
        )

    alpha_t = fp_result["alpha_t"]
    kappa_t = fp_result["kappa_t"]
    bhat = fp_result["bhat"]        # shape (T+1,)
    A_t = fp_result["A_t"]          # shape (T,)
    theta_safe_t = fp_result["theta_safe_t"]
    U_safe_ref_t = fp_result["U_safe_ref_t"]

    # ------------------------------------------------------------------
    # Step 3: Compute informativeness (from final xi_ref_arr)
    # ------------------------------------------------------------------
    I_t = compute_informativeness(xi_ref_arr, p_align_arr)

    # ------------------------------------------------------------------
    # Step 4: Compute u_target per stage
    # ------------------------------------------------------------------
    u_target_arr = u_min + (u_max - u_min) * I_t

    # ------------------------------------------------------------------
    # Step 5: Trust-region cap per stage
    # ------------------------------------------------------------------
    u_tr_cap_arr = np.zeros(T, dtype=np.float64)
    eps_design_arr = np.zeros(T, dtype=np.float64)
    c_arr = np.zeros(T, dtype=np.float64)
    eps_tr_arr = np.zeros(T, dtype=np.float64)

    for t in range(T):
        u_cap, eps_d, c_val, eps_tr = compute_trust_region_cap(
            u_target=u_target_arr[t],
            p_align_t=p_align_arr[t],
            n_t=n_arr[t],
            gamma_base=gamma_base,
            tau_n=tau_n,
        )
        u_tr_cap_arr[t] = float(u_cap)
        eps_design_arr[t] = float(eps_d)
        c_arr[t] = float(c_val)
        eps_tr_arr[t] = float(eps_tr)

    # ------------------------------------------------------------------
    # Step 7: u_ref_used = min(u_target, u_tr_cap, U_safe_ref)
    # ------------------------------------------------------------------
    # In the current parameter regime (kappa > gamma and kappa < 1 + gamma),
    # theta_safe_t = log(kappa / (gamma * (1 + gamma - kappa))) is strictly
    # positive, so U_safe_ref_t = theta_safe_t * xi_ref_t > 0. Defensive
    # hardening: if numerical drift ever drives theta_safe_t negative,
    # silently np.abs-ing the cap would flip a negative safety constraint
    # into a large positive value (effectively disabling it). Instead we
    # assert that any negativity is at round-off scale only and clamp to 0
    # so the safe cap binds maximally rather than being removed.
    assert (U_safe_ref_t >= -1e-12).all(), (
        f"U_safe_ref_t contains significantly-negative values; "
        f"min={U_safe_ref_t.min():.3e}. Check theta_safe_t positivity."
    )
    U_safe_abs = np.where(U_safe_ref_t < 0, 0.0, U_safe_ref_t)
    u_ref_used_arr = np.minimum(np.minimum(u_target_arr, u_tr_cap_arr), U_safe_abs)

    # ------------------------------------------------------------------
    # Step 8: theta_used = sign_family * u_ref_used / max(xi_ref, xi_floor)
    # ------------------------------------------------------------------
    theta_used_arr = sign_family * u_ref_used_arr / np.maximum(xi_ref_arr, xi_floor)

    # ------------------------------------------------------------------
    # Step 9: beta_used = theta_used / max(A_t, 1e-8)
    # ------------------------------------------------------------------
    beta_used_arr = theta_used_arr / np.maximum(A_t, 1e-8)

    # ------------------------------------------------------------------
    # Step 10: Clip flags
    # ------------------------------------------------------------------
    # Identify which cap is the binding argmin (mutually exclusive labels).
    # trust_clip: u_ref_used equals u_tr_cap and trust cap is strictly below u_target.
    # safe_clip: u_ref_used equals U_safe_abs and safe cap is stricter than both trust and target.
    _atol = 1e-10
    trust_clip_active = (
        (np.abs(u_ref_used_arr - u_tr_cap_arr) <= _atol)
        & (u_tr_cap_arr < u_target_arr - _atol)
    ).tolist()
    safe_clip_active = (
        (np.abs(u_ref_used_arr - U_safe_abs) <= _atol)
        & (U_safe_abs < u_tr_cap_arr - _atol)
        & (U_safe_abs < u_target_arr - _atol)
    ).tolist()

    # ------------------------------------------------------------------
    # Assemble schedule dict
    # ------------------------------------------------------------------
    schedule: dict[str, Any] = {
        "phase": "phase4",
        "schedule_version": 3,
        "task_family": task_family,
        "scheduler_mode": "stagewise_u",
        "gamma_eval": float(gamma_eval),
        "gamma_base": float(gamma_base),
        "sign_family": int(sign_family),
        "reward_bound": float(r_max),
        "alpha_t": alpha_t.tolist(),
        "kappa_t": kappa_t.tolist(),
        "Bhat_t": bhat.tolist(),             # shape T+1
        "A_t": A_t.tolist(),                 # shape T
        "xi_ref_t": xi_ref_arr.tolist(),     # shape T
        "u_target_t": u_target_arr.tolist(),
        "u_tr_cap_t": u_tr_cap_arr.tolist(),
        "U_safe_ref_t": U_safe_ref_t.tolist(),
        "u_ref_used_t": u_ref_used_arr.tolist(),
        "theta_used_t": theta_used_arr.tolist(),
        "beta_used_t": beta_used_arr.tolist(),
        "trust_clip_active_t": trust_clip_active,
        "safe_clip_active_t": safe_clip_active,
        "source_phase": source_phase,
        "notes": notes,
        "provenance": {
            "git_sha": _git_sha(),
            "calibration_code_version": "v3.0",
            "input_hashes": {
                "margins": _array_hash(
                    np.concatenate([np.asarray(m) for m in margins_by_stage])
                ),
                "p_align": _array_hash(p_align_arr),
                "n_by_stage": _array_hash(n_arr),
            },
        },
    }
    return schedule


# ---------------------------------------------------------------------------
# Top-level convenience wrapper
# ---------------------------------------------------------------------------


def build_schedule_v3_from_pilot(
    pilot_data: dict[str, Any],
    r_max: float,
    gamma_base: float,
    gamma_eval: float,
    task_family: str = "unknown",
    sign_family: int | None = None,
    source_phase: str = "pilot",
    notes: str = "",
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a v3 schedule from a pilot_data dict.

    Parameters
    ----------
    pilot_data : dict
        Must contain keys:
            margins_by_stage : list of NDArray, length T
            p_align_by_stage : list of float, length T
            n_by_stage : list of int, length T
    r_max : float
        One-step reward bound.
    gamma_base : float
        Discount factor for the operator.
    gamma_eval : float
        Evaluation discount.
    task_family : str
        Name of the task family.
    sign_family : int or None
        +1 or -1; if None, auto-selected via ``select_sign``.
    source_phase : str
        Source phase label.
    notes : str
        Free-text notes.
    output_path : str or Path or None
        If provided, write the schedule JSON to this path.
    **kwargs
        Forwarded to ``build_schedule_v3`` (u_min, u_max, alpha_min, etc.).

    Returns
    -------
    dict
        Schedule v3 dict.
    """
    margins_by_stage = pilot_data["margins_by_stage"]
    p_align_by_stage = pilot_data["p_align_by_stage"]
    n_by_stage = pilot_data["n_by_stage"]

    if sign_family is None:
        sign_family = select_sign(margins_by_stage, r_max)

    schedule = build_schedule_v3(
        margins_by_stage=margins_by_stage,
        p_align_by_stage=p_align_by_stage,
        n_by_stage=n_by_stage,
        r_max=r_max,
        gamma_base=gamma_base,
        gamma_eval=gamma_eval,
        sign_family=sign_family,
        task_family=task_family,
        source_phase=source_phase,
        notes=notes,
        **kwargs,
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(schedule, f, indent=2)

    return schedule
