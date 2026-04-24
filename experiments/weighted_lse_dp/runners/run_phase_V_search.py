#!/usr/bin/env python
"""Phase V WP1c -- search driver + shortlist generator.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` sections 4,
5, 7 (WP1c), 8, 9, 13.  Supersedes the legacy Phase IV activation search
(``run_phase4_activation_search.py``) which is retained for archival
reproducibility only (spec section 7 WP1c).

Pipeline (executed per family in order)
---------------------------------------
1. Enumerate a psi grid from the family's config parameters.
2. Solve ``lambda_tie(psi)`` by exact-DP bisection (families A, B).
3. Sweep ``lam = lam_tie + epsilon`` over a narrow symmetric band.  For
   Family C (``metadata["classical_tie_sweep_supported"] is False``) sweep
   only psi at ``lam = 0``.
4. Cheap prefilter: build the classical DP once per ``(lam, psi)`` and
   accept only candidates with ``contest_gap_norm <= 0.02`` (soft band,
   exploratory).  Family C is always admitted.
5. Enforce the 5,000-candidate aggregate cap (per invocation, not lifetime
   -- spec section 13.3).  Warn if any single family exceeds 60% of the
   cap.  Error out if the aggregate admitted grid exceeds the cap.
6. Calibrate the deployed safe schedule via
   ``geometry.phase4_calibration_v3.build_schedule_v3`` on a synthetic
   pilot (classical-optimal rollout on the FiniteMDP).  One schedule per
   ``(family, psi, lam)`` -- seed is shared.
7. Evaluate every admitted candidate via
   ``search.candidate_metrics.evaluate_candidate`` and persist every spec
   section 6 field into ``candidate_metrics.parquet``.
8. Apply the spec section 4 promotion gate (all conditions conjunctive)
   to build ``shortlist.csv``.  Family C bypasses tie-related criteria in
   favor of the stress-family criterion (spec section 5 Family C).
9. Empty-shortlist contract (spec section 7 WP1c + 13.2): if the
   shortlist contains < 2 promotable positive families or < 1 safety
   family, emit ``shortlist_refinement_manifest.md`` with concrete lever
   recommendations.  Do NOT relax thresholds.  Do NOT auto-fall-back to
   the soft band.
10. Emit ``phase_diagram_data.parquet`` and
    ``summaries/experiment_manifest.json`` from within the runner.

CLI
---
    python -m experiments.weighted_lse_dp.runners.run_phase_V_search \\
        --config experiments/weighted_lse_dp/configs/phaseV/search.yaml \\
        --seed 42 --output-root results/search/ \\
        [--families A B C] [--max-candidates 5000] [--dry-run]

Outputs
-------
``<output-root>/``
    candidate_grid.parquet
    candidate_metrics.parquet
    near_indifference_catalog.csv
    shortlist.csv
    shortlist_report.md
    phase_diagram_data.parquet
    shortlist_refinement_manifest.md    (emitted only when shortlist short)
    resolved_config.yaml                 (resolved config alongside outputs)

``results/summaries/experiment_manifest.json`` (always, spec section 9).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import yaml

# Repo root + vendored MushroomRL on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MUSHROOM = _REPO_ROOT / "mushroom-rl-dev"
if _MUSHROOM.exists() and str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from experiments.weighted_lse_dp.common.manifests import git_sha  # noqa: E402
from experiments.weighted_lse_dp.geometry.phase4_calibration_v3 import (  # noqa: E402
    build_schedule_v3,
)
from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    evaluate_candidate,
)
from experiments.weighted_lse_dp.search.family_spec import FamilySpec  # noqa: E402
from experiments.weighted_lse_dp.search.tie_solver import (  # noqa: E402
    TieNotBracketed,
    lambda_tie,
)
from experiments.weighted_lse_dp.tasks.family_a_jackpot_vs_stream import (  # noqa: E402
    family_a,
)
from experiments.weighted_lse_dp.tasks.family_b_catastrophe import (  # noqa: E402
    family_b,
)
from experiments.weighted_lse_dp.tasks.family_c_raw_stress import (  # noqa: E402
    family_c,
)

__all__ = [
    "ConfigError",
    "main",
    "run_search",
    "apply_promotion_gate",
    "DEFAULT_CONFIG",
]

logger = logging.getLogger("phaseV.search")


class ConfigError(ValueError):
    """Raised when the search grid would breach an invariant (e.g. cap)."""


# ---------------------------------------------------------------------------
# Default config (duplicated inline for dry-runs without a YAML file).
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "families": ["A", "B", "C"],
    "family_params": {
        "A": {
            "L_range": [4, 8, 12],
            "R_range": [1.0, 2.0, 5.0],
            "gamma": 0.95,
            "shapes": ["flat", "one_bump", "ramp"],
            "eps_band_pts": 7,
            "eps_band_frac": 0.2,
        },
        "B": {
            "L_range": [4, 8],
            "p_range": [0.05, 0.10, 0.20],
            "C_range": [2.0, 5.0],
            "b": 1.0,
            "variants": ["single_event", "warning_state", "multi_event"],
            "gamma": 0.95,
            "eps_band_pts": 7,
            "eps_band_frac": 0.2,
        },
        "C": {
            "L_tail_range": [4, 6, 8],
            "R_penalty_range": [1.0, 2.0],
            "beta_raw_multiplier_range": [1.5, 2.5, 4.0],
            "gamma": 0.95,
        },
    },
    "max_candidates": 5000,
    "per_family_warn_frac": 0.60,
    "prefilter_soft_band": 0.02,
    "strict_band": 0.01,
    "promotion": {
        "policy_disagreement_min": 0.05,
        "mass_delta_d_min": 0.10,
        "value_gap_norm_min": 0.005,
        "clip_fraction_min": 0.05,
        "clip_fraction_max": 0.80,
        "cert_tolerance": 1.0e-6,
    },
    "family_c_stress": {
        "raw_deriv_p90_over_kappa": True,
        "clip_fraction_min": 0.05,
    },
    "empty_shortlist_gate": {
        "min_positive_families": 2,
        "min_safety_families": 1,
    },
    "pilot": {
        "n_episodes": 30,
        "eps_greedy": 0.1,
    },
    "seed": 42,
}

_SCHEMA_VERSION = "phaseV.search.v1"


# Registry of families. Keyed by the single-letter label used in the CLI
# and the config.  Each entry is a (FamilySpec, enumerate_grid) pair.
_FAMILY_REGISTRY: dict[str, FamilySpec] = {
    "A": family_a,
    "B": family_b,
    "C": family_c,
}


# ---------------------------------------------------------------------------
# psi-grid enumeration (one generator per family)
# ---------------------------------------------------------------------------

def _psi_grid_family_a(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Family A: cartesian product of (L, R, shape), gamma fixed."""
    grid: list[dict[str, Any]] = []
    gamma = float(params["gamma"])
    for L, R, shape in itertools.product(
        params["L_range"], params["R_range"], params["shapes"]
    ):
        psi: dict[str, Any] = {
            "L": int(L),
            "R": float(R),
            "gamma": gamma,
            "shape": str(shape),
        }
        # Default shape-basis parameters (must match shape_basis() expected
        # knobs in _family_helpers).
        if shape == "one_bump":
            psi["bump_center"] = int(L) // 2
            psi["bump_width"] = 1.0
            psi["bump_strength"] = 0.5
        elif shape == "ramp":
            psi["ramp_amplitude"] = 0.5
        grid.append(psi)
    return grid


def _psi_grid_family_b(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Family B: cartesian product of (L, p, C, variant), b/gamma fixed."""
    grid: list[dict[str, Any]] = []
    gamma = float(params["gamma"])
    b = float(params["b"])
    for L, p, C, variant in itertools.product(
        params["L_range"],
        params["p_range"],
        params["C_range"],
        params["variants"],
    ):
        psi: dict[str, Any] = {
            "variant": str(variant),
            "L": int(L),
            "gamma": gamma,
            "b": b,
            "p": float(p),
            "C": float(C),
        }
        if variant == "warning_state":
            psi["warning_depth"] = max(1, int(L) // 2)
        if variant == "multi_event":
            # Minimal two-event configuration (K=2); keeps 2^K branching
            # tractable per the multi-event MDP builder's K<=6 bound.
            if L < 3:
                # Need depth >= 2 for both events; skip too-short L.
                continue
            psi["event_probs"] = [float(p), float(p) * 0.5]
            psi["event_depths"] = [2, max(3, int(L) - 1)]
            psi["event_mags"] = [float(C), float(C) * 0.5]
        if variant == "shallow_early":
            # Not in the default config; included for completeness.
            psi["b_shallow"] = 0.2 * b
            psi["k_shallow"] = max(1, int(L) // 3)
        if variant == "matched_concentration":
            psi["b_stream"] = 0.1 * b
        grid.append(psi)
    return grid


def _psi_grid_family_c(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Family C: cartesian product of (L_tail, R_penalty, beta_raw_mult)."""
    grid: list[dict[str, Any]] = []
    gamma = float(params["gamma"])
    for L_tail, R_pen, mult in itertools.product(
        params["L_tail_range"],
        params["R_penalty_range"],
        params["beta_raw_multiplier_range"],
    ):
        psi: dict[str, Any] = {
            "L_tail": int(L_tail),
            "R_penalty": float(R_pen),
            "gamma": gamma,
            "beta_raw_multiplier": float(mult),
        }
        grid.append(psi)
    return grid


_PSI_GRID_BUILDERS = {
    "A": _psi_grid_family_a,
    "B": _psi_grid_family_b,
    "C": _psi_grid_family_c,
}


# ---------------------------------------------------------------------------
# Cheap contest_gap_norm prefilter
# ---------------------------------------------------------------------------

def _classical_contest_gap(
    family: FamilySpec,
    lam: float,
    psi: dict[str, Any],
) -> tuple[float, float, float]:
    """Return (contest_gap_abs, contest_gap_norm, reward_scale) via classical DP.

    Shape annotations: internal tensors are ``P, R: (S, A, S')``,
    ``V: (T+1, S)``, ``Q: (T, S, A)``.
    """
    mdp = family.build_mdp(float(lam), psi)
    P = np.asarray(mdp.p, dtype=np.float64)                 # (S, A, S')
    Rr = np.asarray(mdp.r, dtype=np.float64)                 # (S, A, S')
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)
    S, A = P.shape[0], P.shape[1]
    r_bar = np.einsum("ijk,ijk->ij", P, Rr)                  # (S, A)
    V = np.zeros((T + 1, S), dtype=np.float64)
    Q = np.zeros((T, S, A), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        E_v = np.einsum("ijk,k->ij", P, V[t + 1])
        Q[t] = r_bar + gamma * E_v
        V[t] = np.max(Q[t], axis=1)

    t_c = int(family.contest_state.t)
    s_c = int(family.contest_state.s)
    ap = family.contest_state.action_pair
    if ap is None:
        order = np.argsort(Q[t_c, s_c])[::-1]
        a1, a2 = int(order[0]), int(order[1])
    else:
        a1, a2 = int(ap[0]), int(ap[1])
    gap_abs = float(abs(Q[t_c, s_c, a1] - Q[t_c, s_c, a2]))

    init_s = int(getattr(mdp, "initial_state", 0))
    R_max = float(np.max(np.abs(Rr))) if Rr.size else 0.0
    reward_scale = float(max(R_max, abs(float(V[0, init_s])), 1e-8))
    return gap_abs, gap_abs / reward_scale, reward_scale


# ---------------------------------------------------------------------------
# Synthetic pilot: roll out the classical-optimal policy on the FiniteMDP
# ---------------------------------------------------------------------------

def _synth_pilot_from_mdp(
    mdp: Any,
    *,
    seed: int,
    n_episodes: int,
    eps_greedy: float,
    sign_family: int,
) -> dict[str, Any]:
    """Synthesize per-stage (margin, p_align, n) arrays for build_schedule_v3.

    This is the WP1c analogue of
    ``geometry.task_activation_search.run_classical_pilot``: it uses exact
    classical DP to get ``V*_cl``, then runs ``n_episodes`` epsilon-greedy
    rollouts directly on the FiniteMDP's ``(P, R)`` tensors (no gym wrapper,
    no MushroomRL Core loop).  The returned dict satisfies the
    ``build_schedule_v3`` pilot contract.
    """
    P = np.asarray(mdp.p, dtype=np.float64)                 # (S, A, S')
    R = np.asarray(mdp.r, dtype=np.float64)                 # (S, A, S')
    gamma = float(mdp.info.gamma)
    T = int(mdp.info.horizon)
    S, A = P.shape[0], P.shape[1]
    r_bar = np.einsum("ijk,ijk->ij", P, R)                   # (S, A)
    V = np.zeros((T + 1, S), dtype=np.float64)
    Q = np.zeros((T, S, A), dtype=np.float64)
    for t in range(T - 1, -1, -1):
        E_v = np.einsum("ijk,k->ij", P, V[t + 1])
        Q[t] = r_bar + gamma * E_v
        V[t] = np.max(Q[t], axis=1)

    rng = np.random.default_rng(seed)
    margins_by_stage: list[list[float]] = [[] for _ in range(T)]
    init_s = int(getattr(mdp, "initial_state", 0))

    for _ep in range(int(n_episodes)):
        s = init_s
        for t in range(T):
            row = P[s]                                        # (A, S')
            # Absorbing: no outgoing mass. Break to avoid degenerate choice.
            if row.sum() <= 0.0:
                break
            # eps-greedy using stage-dependent Q.
            if rng.random() < float(eps_greedy):
                a = int(rng.integers(0, A))
            else:
                a = int(np.argmax(Q[t, s]))
            probs = P[s, a]
            mass = probs.sum()
            if mass <= 0.0:
                break
            probs = probs / mass                              # renorm
            s_next = int(rng.choice(S, p=probs))
            r_val = float(R[s, a, s_next])
            v_next = float(V[t + 1, s_next]) if (t + 1) <= T else 0.0
            margins_by_stage[t].append(r_val - v_next)
            s = s_next

    margins_arr_list: list[np.ndarray] = []
    p_align_list: list[float] = []
    n_list: list[int] = []
    for t in range(T):
        arr = np.asarray(margins_by_stage[t], dtype=np.float64)
        if arr.size == 0:
            # Stage not visited in any rollout: supply a single 0.0 sample.
            arr = np.zeros(1, dtype=np.float64)
        margins_arr_list.append(arr)
        p_align_list.append(float(np.mean(sign_family * arr > 0.0)))
        n_list.append(int(arr.size))

    return {
        "margins_by_stage": margins_arr_list,
        "p_align_by_stage": p_align_list,
        "n_by_stage": n_list,
        "reward_bound": float(np.max(np.abs(R))) if R.size else 1.0,
        "gamma": gamma,
        "horizon": T,
    }


def _beta_cap_from_alpha(
    alpha_t: np.ndarray,           # (T,)
    R_max: float,
    gamma: float,
) -> np.ndarray:
    """Compute ``beta_cap_t`` from the certification recursion (spec S2.2).

    Mirrors ``calibration_utils.build_certification``.  Kept inline to
    avoid pulling in the full module (we only need beta_cap).
    """
    T = int(alpha_t.shape[0])
    kappa_t = gamma + alpha_t * (1.0 - gamma)                 # (T,)
    Bhat = np.zeros(T + 1, dtype=np.float64)
    for t in range(T - 1, -1, -1):
        Bhat[t] = (1.0 + gamma) * R_max + kappa_t[t] * Bhat[t + 1]
    beta_cap = np.zeros(T, dtype=np.float64)
    for t in range(T):
        denom = R_max + Bhat[t + 1]
        numer_arg = kappa_t[t] / (gamma * (1.0 + gamma - kappa_t[t]))
        if numer_arg > 0.0 and denom > 0.0:
            beta_cap[t] = float(np.log(numer_arg) / denom)
        else:
            beta_cap[t] = 0.0
    return beta_cap


def _calibrate_schedule(
    mdp: Any,
    *,
    family_label: str,
    pilot_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Build a deployed safe clipped schedule for one candidate.

    Returns a dict with keys ``beta_used_t``, ``beta_cap_t``, ``beta_raw_t``,
    ``kappa_t``, ``Bhat_t``, ``alpha_t``, and ``sign_family``.  The first
    three satisfy the contract of
    ``search.candidate_metrics.evaluate_candidate``.
    """
    # Family C is the stress family: we deliberately push beta_raw above
    # beta_cap by ``beta_raw_multiplier`` so clip fires on every stage.
    # For Families A and B we use the standard v3 recipe.
    gamma = float(mdp.info.gamma)
    P = np.asarray(mdp.p, dtype=np.float64)
    R = np.asarray(mdp.r, dtype=np.float64)
    R_max = float(np.max(np.abs(R))) if R.size else 1.0

    # Sign family heuristic: Family B (catastrophe) is -1, A/C are +1.
    sign_family = int(-1 if family_label == "B" else 1)

    pilot = _synth_pilot_from_mdp(
        mdp,
        seed=int(seed),
        n_episodes=int(pilot_cfg.get("n_episodes", 30)),
        eps_greedy=float(pilot_cfg.get("eps_greedy", 0.1)),
        sign_family=sign_family,
    )
    v3 = build_schedule_v3(
        margins_by_stage=pilot["margins_by_stage"],
        p_align_by_stage=pilot["p_align_by_stage"],
        n_by_stage=pilot["n_by_stage"],
        r_max=R_max,
        gamma_base=gamma,
        gamma_eval=gamma,
        sign_family=sign_family,
        task_family=f"phaseV_family_{family_label}",
        source_phase="phaseV_search",
        notes="WP1c synthetic pilot from classical-optimal rollout",
    )
    alpha_t = np.asarray(v3["alpha_t"], dtype=np.float64)
    beta_cap_t = _beta_cap_from_alpha(alpha_t, R_max, gamma)     # (T,)
    beta_used_t = np.asarray(v3["beta_used_t"], dtype=np.float64)

    if family_label == "C":
        # Stress family: beta_raw = multiplier * beta_cap; beta_used is the
        # clipped version.  This reproduces the raw_deriv_p90 > kappa
        # condition on visited transitions (spec section 5 Family C).
        # The multiplier comes from psi; fall back to 2.5 when missing.
        mult = float(getattr(mdp, "_beta_raw_multiplier", 2.5))
        # Preserve sign_family on beta_raw.  Use a strictly positive cap
        # baseline so the clip actually fires; v3 may have emitted beta_used
        # near zero on stages where aligned margins were absent.
        safe_cap = np.where(beta_cap_t > 0.0, beta_cap_t, 1.0e-3)
        beta_raw_t = float(sign_family) * mult * safe_cap          # (T,)
        beta_used_t = np.clip(beta_raw_t, -beta_cap_t, beta_cap_t) # (T,)
    else:
        # Standard v3: beta_used already respects the trust region; we set
        # beta_raw = beta_used so no further clip.
        beta_raw_t = beta_used_t.copy()
        # Clip to avoid floating-point slivers (mirrors run_phase4_dp).
        beta_used_t = np.clip(beta_used_t, -beta_cap_t, beta_cap_t)

    return {
        "beta_used_t": beta_used_t,
        "beta_cap_t": beta_cap_t,
        "beta_raw_t": beta_raw_t,
        "kappa_t": np.asarray(v3["kappa_t"], dtype=np.float64),
        "Bhat_t": np.asarray(v3["Bhat_t"], dtype=np.float64),
        "alpha_t": alpha_t,
        "sign_family": int(sign_family),
    }


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

def _psi_reduction(psi: dict[str, Any]) -> tuple[float, float]:
    """Compact 1D/2D reduction of psi for phase_diagram_data (spec section 8).

    We pick two floats: the first and second numeric entries of psi in
    insertion order (falling back to 0 when fewer are available).  This is
    intentionally simple; WP3 plotters that need a richer embedding should
    re-derive from the original ``psi_json`` column.
    """
    nums: list[float] = []
    for v in psi.values():
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(
            v, bool
        ):
            nums.append(float(v))
        if len(nums) >= 2:
            break
    while len(nums) < 2:
        nums.append(0.0)
    return nums[0], nums[1]


def apply_promotion_gate(
    metrics_df: pd.DataFrame,
    *,
    promotion: dict[str, float],
    strict_band: float,
    family_c_cfg: dict[str, Any],
    kappa_t_max_column: str = "kappa_t_max",
) -> pd.DataFrame:
    """Return the subset of ``metrics_df`` passing the spec section 4 gate.

    ``metrics_df`` columns must include every key emitted by
    ``evaluate_candidate`` plus ``family, kappa_t_max, raw_local_deriv_max,
    raw_local_deriv_p90``.
    """
    df = metrics_df.copy()
    cert_tol = float(promotion["cert_tolerance"])

    def _cert_ok(row: pd.Series) -> bool:
        return bool(
            float(row["raw_local_deriv_max"])
            <= float(row[kappa_t_max_column]) + cert_tol
        )

    def _positive_gate(row: pd.Series) -> tuple[bool, str]:
        """Positive-family gate per spec §14 (decision doc v4/v5, 2026-04-24).

        A candidate promotes if every mechanism gate passes AND clipping is
        either bound nondegenerately (``binding_clip``) or provably inactive
        because the raw schedule is already safe on the certified domain
        (``safe_active_no_distortion``).

        Returns ``(ok, promotion_mode)`` where ``promotion_mode`` is one of
        ``"binding_clip"``, ``"safe_active_no_distortion"``, ``"none"``.
        """
        disagree = float(row["policy_disagreement"]) >= float(
            promotion["policy_disagreement_min"]
        )
        start_flip = int(row["start_state_flip"]) == 1
        mass_ok = float(row["mass_delta_d"]) >= float(
            promotion["mass_delta_d_min"]
        )
        vgap_ok = abs(float(row["value_gap_norm"])) >= float(
            promotion["value_gap_norm_min"]
        )
        tie_ok = float(row["contest_gap_norm"]) <= float(strict_band)
        cert = _cert_ok(row)

        mechanism_ok = (
            (disagree or start_flip) and mass_ok and vgap_ok and tie_ok and cert
        )
        if not mechanism_ok:
            return False, "none"

        clip_frac = float(row["clip_fraction"])
        clip_binding = (
            float(promotion["clip_fraction_min"]) <= clip_frac
            <= float(promotion["clip_fraction_max"])
        )
        # "Provably inactive because the raw schedule is already safe on the
        # certified domain": clip never bit AND the raw-operator local
        # derivative stays within the certified stagewise cap on visited
        # mass (_cert_ok already asserts this).
        clip_inactive_safe = (clip_frac == 0.0) and cert

        if clip_binding:
            return True, "binding_clip"
        if clip_inactive_safe:
            return True, "safe_active_no_distortion"
        return False, "none"

    def _stress_gate(row: pd.Series) -> tuple[bool, str]:
        # Family C bypasses the tie criteria; stress criterion per spec
        # §5 Family C and §14 (safety family keeps strict clip gate).
        p90_ok = float(row["raw_local_deriv_p90"]) > float(row[kappa_t_max_column])
        clip_ok = float(row["clip_fraction"]) > float(
            family_c_cfg.get("clip_fraction_min", 0.05)
        )
        if p90_ok and clip_ok:
            return True, "binding_clip"
        return False, "none"

    mask: list[bool] = []
    reasons: list[str] = []
    modes: list[str] = []
    for _, row in df.iterrows():
        fam = str(row["family"])
        if fam == "C":
            ok, mode = _stress_gate(row)
            reasons.append(
                "Family-C stress (raw_p90 > kappa_max AND clip_fraction > min)"
                if ok
                else "Family-C stress gate FAIL"
            )
        else:
            ok, mode = _positive_gate(row)
            if ok:
                reasons.append(f"positive-family PASS (mode={mode})")
            else:
                reasons.append("positive-family gate FAIL")
        mask.append(ok)
        modes.append(mode)
    df["_promoted"] = mask
    df["_promotion_reason"] = reasons
    df["promotion_mode"] = modes
    return df


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    """Single admitted candidate after prefilter."""

    family: str
    psi: dict[str, Any]
    lam: float
    contest_gap_abs: float
    contest_gap_norm: float
    reward_scale: float
    lam_tie: float
    lam_tie_diag: dict[str, Any]


def _epsilon_band(lam_tie: float, cfg: dict[str, Any]) -> list[float]:
    """Symmetric epsilon band around ``lam_tie``.

    The band has ``eps_band_pts`` points linearly spaced over
    ``lam_tie * (1 +/- eps_band_frac)`` (or ``+/- 1.0`` when lam_tie = 0).
    """
    pts = int(cfg.get("eps_band_pts", 7))
    frac = float(cfg.get("eps_band_frac", 0.2))
    if abs(lam_tie) < 1e-12:
        half = 1.0
    else:
        half = abs(lam_tie) * frac
    lo = lam_tie - half
    hi = lam_tie + half
    return list(np.linspace(lo, hi, pts))


def _enumerate_admitted(
    family_label: str,
    family: FamilySpec,
    grid: list[dict[str, Any]],
    *,
    soft_band: float,
    eps_cfg: dict[str, Any] | None,
    tie_diagnostics: list[dict[str, Any]],
) -> list[_Candidate]:
    """Per-family: solve tie, sweep epsilon band, apply prefilter."""
    admitted: list[_Candidate] = []
    supports_tie = family.metadata.get("classical_tie_sweep_supported", True)

    for psi in grid:
        if supports_tie:
            try:
                lam_tie, diag = lambda_tie(psi, family)
            except TieNotBracketed as exc:
                logger.warning(
                    "family=%s tie not bracketed for psi=%s: %s",
                    family_label,
                    json.dumps(psi, default=str, sort_keys=True),
                    exc,
                )
                tie_diagnostics.append({
                    "family": family_label,
                    "psi_json": json.dumps(psi, default=str, sort_keys=True),
                    "status": "not_bracketed",
                    "lam_tie": float("nan"),
                    "iters": 0,
                    "final_gap_abs": float("nan"),
                    "expansion_factor": float(exc.scan_record.get("expansion_factor", 0.0)),
                })
                continue
            tie_diagnostics.append({
                "family": family_label,
                "psi_json": json.dumps(psi, default=str, sort_keys=True),
                "status": "converged",
                "lam_tie": float(lam_tie),
                "iters": int(diag["iters"]),
                "final_gap_abs": float(diag["final_gap_abs"]),
                "expansion_factor": float(diag["expansion_factor"]),
            })
            lam_band = _epsilon_band(lam_tie, eps_cfg or {})
        else:
            # Family C -- no tie sweep, single lam point.
            lam_tie = 0.0
            lam_band = [0.0]
            tie_diagnostics.append({
                "family": family_label,
                "psi_json": json.dumps(psi, default=str, sort_keys=True),
                "status": "no_tie_sweep",
                "lam_tie": 0.0,
                "iters": 0,
                "final_gap_abs": 0.0,
                "expansion_factor": 1.0,
            })

        for lam in lam_band:
            gap_abs, gap_norm, reward_scale = _classical_contest_gap(
                family, float(lam), psi
            )
            # Prefilter: Family C always admits; others require gap_norm <= soft_band.
            if not supports_tie or gap_norm <= float(soft_band):
                admitted.append(
                    _Candidate(
                        family=family_label,
                        psi=psi,
                        lam=float(lam),
                        contest_gap_abs=float(gap_abs),
                        contest_gap_norm=float(gap_norm),
                        reward_scale=float(reward_scale),
                        lam_tie=float(lam_tie),
                        lam_tie_diag=dict(),
                    )
                )
    return admitted


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_search(
    cfg: dict[str, Any],
    *,
    output_root: Path,
    seed: int,
    families_override: list[str] | None = None,
    max_candidates_override: int | None = None,
    dry_run: bool = False,
    exact_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the full WP1c search pipeline.  Returns a summary dict."""
    t_start = time.time()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    families = list(families_override or cfg["families"])
    max_cap = int(max_candidates_override or cfg["max_candidates"])
    per_family_warn = float(cfg.get("per_family_warn_frac", 0.60))
    soft_band = float(cfg["prefilter_soft_band"])
    strict_band = float(cfg["strict_band"])

    # ------ 1. psi grids ------
    total_psi = 0
    grids_summary: dict[str, int] = {}
    grids: dict[str, list[dict[str, Any]]] = {}
    for fam_label in families:
        if fam_label not in _FAMILY_REGISTRY:
            raise ConfigError(f"unknown family label: {fam_label!r}")
        params = cfg["family_params"][fam_label]
        grid = _PSI_GRID_BUILDERS[fam_label](params)
        grids[fam_label] = grid
        grids_summary[fam_label] = len(grid)
        total_psi += len(grid)
    logger.info("psi grid sizes: %s (total=%d)", grids_summary, total_psi)

    # ------ 2-4. tie + epsilon band + prefilter ------
    tie_diagnostics: list[dict[str, Any]] = []
    admitted: list[_Candidate] = []
    per_family_counts: dict[str, int] = {}
    for fam_label in families:
        family = _FAMILY_REGISTRY[fam_label]
        eps_cfg = cfg["family_params"][fam_label]
        fam_admitted = _enumerate_admitted(
            fam_label,
            family,
            grids[fam_label],
            soft_band=soft_band,
            eps_cfg=eps_cfg,
            tie_diagnostics=tie_diagnostics,
        )
        per_family_counts[fam_label] = len(fam_admitted)
        admitted.extend(fam_admitted)
        logger.info(
            "family %s: %d psi, %d admitted after prefilter",
            fam_label,
            len(grids[fam_label]),
            len(fam_admitted),
        )

    # 5. Hard cap + per-family soft warning.
    for fam_label, n in per_family_counts.items():
        if n > per_family_warn * max_cap:
            logger.warning(
                "family %s admitted %d candidates > %.0f%% of max_candidates=%d; "
                "consider narrowing the psi grid",
                fam_label, n, per_family_warn * 100, max_cap,
            )
    if len(admitted) > max_cap:
        raise ConfigError(
            f"admitted candidate count {len(admitted)} exceeds max_candidates="
            f"{max_cap}. Narrow the psi grid (--max-candidates is the hard "
            f"cap; the per-family 60% warning threshold is "
            f"{int(per_family_warn * max_cap)})."
        )

    # ------ 6-7. per-candidate schedule + metrics ------
    # Persist candidate_grid.parquet even in --dry-run so tests can round-trip.
    candidate_grid_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    phase_diag_rows: list[dict[str, Any]] = []
    near_indiff_rows: list[dict[str, Any]] = []

    for cand in admitted:
        mdp = _FAMILY_REGISTRY[cand.family].build_mdp(cand.lam, cand.psi)
        # Stash beta_raw_multiplier on mdp for Family C calibration lookup.
        if cand.family == "C":
            mdp._beta_raw_multiplier = float(cand.psi.get("beta_raw_multiplier", 2.5))  # type: ignore[attr-defined]
        sched = _calibrate_schedule(
            mdp,
            family_label=cand.family,
            pilot_cfg=cfg["pilot"],
            seed=int(seed),
        )
        metrics = evaluate_candidate(
            mdp,
            {
                "beta_used_t": sched["beta_used_t"],
                "beta_cap_t": sched["beta_cap_t"],
                "beta_raw_t": sched["beta_raw_t"],
            },
            contest_state=_FAMILY_REGISTRY[cand.family].contest_state,
        )
        kappa_max = float(np.max(sched["kappa_t"]))
        psi_r1, psi_r2 = _psi_reduction(cand.psi)

        base = {
            "family": cand.family,
            "psi_json": json.dumps(cand.psi, default=str, sort_keys=True),
            "lam": float(cand.lam),
            "lam_tie": float(cand.lam_tie),
            "reward_scale": float(cand.reward_scale),
            "kappa_t_max": kappa_max,
            "kappa_t_json": json.dumps(sched["kappa_t"].tolist()),
            "sign_family": int(sched["sign_family"]),
            "horizon": int(mdp.info.horizon),
            "gamma": float(mdp.info.gamma),
        }
        candidate_grid_rows.append({**base})
        # spec section 6 fields -- every key from evaluate_candidate.
        mrow = {
            **base,
            "contest_gap_abs": float(metrics["contest_gap_abs"]),
            "contest_gap_norm": float(metrics["contest_gap_norm"]),
            "contest_occupancy_ref": float(metrics["contest_occupancy_ref"]),
            "margin_pos": float(metrics["margin_pos"]),
            "margin_pos_norm": float(metrics["margin_pos_norm"]),
            "delta_d": float(metrics["delta_d"]),
            "mass_delta_d": float(metrics["mass_delta_d"]),
            "policy_disagreement": float(metrics["policy_disagreement"]),
            "start_state_flip": int(metrics["start_state_flip"]),
            "value_gap": float(metrics["value_gap"]),
            "value_gap_norm": float(metrics["value_gap_norm"]),
            "clip_fraction": float(metrics["clip_fraction"]),
            "clip_inactive_fraction": float(metrics["clip_inactive_fraction"]),
            "clip_saturation_fraction": float(metrics["clip_saturation_fraction"]),
            "raw_local_deriv_mean": float(metrics["raw_local_deriv_stats"]["mean"]),
            "raw_local_deriv_p50": float(metrics["raw_local_deriv_stats"]["p50"]),
            "raw_local_deriv_p90": float(metrics["raw_local_deriv_stats"]["p90"]),
            "raw_local_deriv_max": float(metrics["raw_local_deriv_stats"]["max"]),
            "raw_convergence_status": str(metrics["raw_convergence_status"]),
        }
        metrics_rows.append(mrow)
        # phase_diagram_data.parquet -- compact 2D reduction of psi.
        phase_diag_rows.append({
            "family": cand.family,
            "psi_reduction_x": float(psi_r1),
            "psi_reduction_y": float(psi_r2),
            "lam": float(cand.lam),
            "contest_gap_sign": float(np.sign(
                metrics["value_gap"] if metrics["contest_gap_abs"] == 0.0
                else metrics["contest_gap_abs"]
            )),
            "safe_gap_sign": float(np.sign(metrics["value_gap"])),
            "clip_fraction": float(metrics["clip_fraction"]),
        })
        near_indiff_rows.append({
            "family": cand.family,
            "psi_json": json.dumps(cand.psi, default=str, sort_keys=True),
            "lam": float(cand.lam),
            "lam_tie": float(cand.lam_tie),
            "contest_gap_abs": float(cand.contest_gap_abs),
            "contest_gap_norm": float(cand.contest_gap_norm),
        })

    # ------ 8. promotion gate ------
    metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame(
        columns=_EMPTY_METRICS_COLUMNS
    )
    promoted_df = apply_promotion_gate(
        metrics_df,
        promotion=cfg["promotion"],
        strict_band=strict_band,
        family_c_cfg=cfg["family_c_stress"],
    )
    shortlist_df = promoted_df[promoted_df["_promoted"]].copy()

    # ------ Persist artifacts ------
    _write_outputs(
        output_root=output_root,
        candidate_grid_rows=candidate_grid_rows,
        metrics_df=metrics_df,
        shortlist_df=shortlist_df,
        phase_diag_rows=phase_diag_rows,
        near_indiff_rows=near_indiff_rows,
        tie_diagnostics=tie_diagnostics,
        cfg=cfg,
    )

    # ------ 9. empty-shortlist contract ------
    positive_families = sorted(set(
        str(x) for x in shortlist_df["family"].tolist() if str(x) in {"A", "B"}
    ))
    safety_families = sorted(set(
        str(x) for x in shortlist_df["family"].tolist() if str(x) == "C"
    ))
    empty_gate = cfg.get("empty_shortlist_gate", {})
    min_positive = int(empty_gate.get("min_positive_families", 2))
    min_safety = int(empty_gate.get("min_safety_families", 1))
    refinement_required = (
        len(positive_families) < min_positive or len(safety_families) < min_safety
    )
    if refinement_required:
        _write_refinement_manifest(
            output_root / "shortlist_refinement_manifest.md",
            positive_families=positive_families,
            safety_families=safety_families,
            metrics_df=metrics_df,
            cfg=cfg,
        )

    # ------ 10. experiment_manifest.json (spec section 9) ------
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "phase": "phaseV",
        "runner": "run_phase_V_search",
        "exact_argv": list(exact_argv) if exact_argv is not None else list(sys.argv),
        "seed_list": [int(seed)],
        "family_list": list(families),
        "psi_grid_summary": grids_summary,
        "n_candidates_admitted": int(len(admitted)),
        "n_candidates_promoted": int(len(shortlist_df)),
        "output_paths": {
            "candidate_grid": str(output_root / "candidate_grid.parquet"),
            "candidate_metrics": str(output_root / "candidate_metrics.parquet"),
            "near_indifference_catalog": str(output_root / "near_indifference_catalog.csv"),
            "shortlist": str(output_root / "shortlist.csv"),
            "shortlist_report": str(output_root / "shortlist_report.md"),
            "phase_diagram_data": str(output_root / "phase_diagram_data.parquet"),
            "shortlist_refinement_manifest": str(
                output_root / "shortlist_refinement_manifest.md"
            ),
        },
        "git_sha": git_sha(),
        "host": _safe_hostname(),
        "timestamp": _utc_now_iso(),
        "dry_run": bool(dry_run),
        "config": cfg,
        "elapsed_sec": float(time.time() - t_start),
    }

    summaries_dir = _REPO_ROOT / "results" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = summaries_dir / "experiment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return {
        "n_admitted": int(len(admitted)),
        "n_promoted": int(len(shortlist_df)),
        "psi_grid_summary": grids_summary,
        "per_family_counts": per_family_counts,
        "refinement_required": bool(refinement_required),
        "elapsed_sec": float(time.time() - t_start),
        "manifest_path": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_EMPTY_METRICS_COLUMNS = [
    "family", "psi_json", "lam", "lam_tie", "reward_scale", "kappa_t_max",
    "kappa_t_json", "sign_family", "horizon", "gamma",
    "contest_gap_abs", "contest_gap_norm", "contest_occupancy_ref",
    "margin_pos", "margin_pos_norm", "delta_d", "mass_delta_d",
    "policy_disagreement", "start_state_flip", "value_gap", "value_gap_norm",
    "clip_fraction", "clip_inactive_fraction", "clip_saturation_fraction",
    "raw_local_deriv_mean", "raw_local_deriv_p50", "raw_local_deriv_p90",
    "raw_local_deriv_max", "raw_convergence_status",
]


def _write_outputs(
    *,
    output_root: Path,
    candidate_grid_rows: list[dict[str, Any]],
    metrics_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    phase_diag_rows: list[dict[str, Any]],
    near_indiff_rows: list[dict[str, Any]],
    tie_diagnostics: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    """Persist every WP1c output artifact under ``output_root``."""
    # Candidate grid parquet (spec section 8).
    grid_df = pd.DataFrame(
        candidate_grid_rows if candidate_grid_rows else [],
        columns=["family", "psi_json", "lam", "lam_tie", "reward_scale",
                 "kappa_t_max", "kappa_t_json", "sign_family", "horizon",
                 "gamma"],
    )
    _write_parquet(grid_df, output_root / "candidate_grid.parquet")

    # Candidate metrics parquet -- schema_version stamped into metadata column
    # on the first row for consumer round-tripping.
    _write_parquet(metrics_df, output_root / "candidate_metrics.parquet")

    # Near-indifference catalog (CSV).
    near_df = pd.DataFrame(
        near_indiff_rows if near_indiff_rows else [],
        columns=["family", "psi_json", "lam", "lam_tie",
                 "contest_gap_abs", "contest_gap_norm"],
    )
    near_df.to_csv(output_root / "near_indifference_catalog.csv", index=False)

    # Shortlist CSV + markdown report.
    shortlist_df.to_csv(output_root / "shortlist.csv", index=False)
    _write_shortlist_report(
        output_root / "shortlist_report.md",
        shortlist_df=shortlist_df,
        full_df=metrics_df,
        cfg=cfg,
        tie_diagnostics=tie_diagnostics,
    )

    # Phase diagram parquet (spec section 8).
    phase_df = pd.DataFrame(
        phase_diag_rows if phase_diag_rows else [],
        columns=["family", "psi_reduction_x", "psi_reduction_y", "lam",
                 "contest_gap_sign", "safe_gap_sign", "clip_fraction"],
    )
    _write_parquet(phase_df, output_root / "phase_diagram_data.parquet")

    # Resolved config alongside outputs.
    resolved_path = output_root / "resolved_config.yaml"
    with open(resolved_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet with pyarrow (fallback: fastparquet)."""
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
    except ImportError:  # pragma: no cover - exercised only without pyarrow
        try:
            df.to_parquet(path, engine="fastparquet", index=False)
        except ImportError as exc:
            raise RuntimeError(
                f"parquet engines unavailable (pyarrow and fastparquet both missing); "
                f"cannot write {path}."
            ) from exc


def _write_shortlist_report(
    path: Path,
    *,
    shortlist_df: pd.DataFrame,
    full_df: pd.DataFrame,
    cfg: dict[str, Any],
    tie_diagnostics: list[dict[str, Any]],
) -> None:
    """Emit ``shortlist_report.md`` summarising promotion outcomes."""
    positives = shortlist_df[shortlist_df["family"].isin(["A", "B"])]
    safety = shortlist_df[shortlist_df["family"] == "C"]
    fam_counts = shortlist_df.groupby("family").size().to_dict() if not shortlist_df.empty else {}
    full_counts = full_df.groupby("family").size().to_dict() if not full_df.empty else {}
    lines: list[str] = []
    lines.append("# Phase V WP1c shortlist report\n")
    lines.append(
        f"Generated {_utc_now_iso()}.  Schema version: `{_SCHEMA_VERSION}`.\n"
    )
    lines.append("## Thresholds (spec section 4; HARD-CODED)\n")
    for k, v in cfg["promotion"].items():
        lines.append(f"- `{k}` = {v:g}" if isinstance(v, float) else f"- `{k}` = {v}")
    lines.append(f"- `strict_band` (contest_gap_norm ceiling) = {cfg['strict_band']:g}")
    lines.append(f"- `prefilter_soft_band` (exploration only) = {cfg['prefilter_soft_band']:g}")
    lines.append("")
    lines.append("## Candidate totals\n")
    for fam in sorted(full_counts.keys()):
        lines.append(
            f"- family {fam}: {full_counts.get(fam, 0)} evaluated, "
            f"{fam_counts.get(fam, 0)} promoted"
        )
    lines.append("")
    lines.append(f"## Shortlist ({len(shortlist_df)} rows)\n")
    if shortlist_df.empty:
        lines.append("_Shortlist is empty._  See `shortlist_refinement_manifest.md` for lever recommendations.\n")
    else:
        for _, row in shortlist_df.iterrows():
            fam = str(row["family"])
            lines.append(
                f"- **{fam}** `lam={float(row['lam']):.4g}` "
                f"`psi={row['psi_json']}`: "
                f"`policy_disagreement={float(row['policy_disagreement']):.3e}`, "
                f"`value_gap_norm={float(row['value_gap_norm']):.3e}`, "
                f"`clip_fraction={float(row['clip_fraction']):.3e}`, "
                f"`contest_gap_norm={float(row['contest_gap_norm']):.3e}`, "
                f"`raw_p90={float(row['raw_local_deriv_p90']):.3e}`, "
                f"`kappa_t_max={float(row['kappa_t_max']):.3e}`."
            )
        lines.append("")

    lines.append("## Tie-solver diagnostics\n")
    n_converged = sum(1 for d in tie_diagnostics if d.get("status") == "converged")
    n_bracket = sum(1 for d in tie_diagnostics if d.get("status") == "not_bracketed")
    n_no_tie = sum(1 for d in tie_diagnostics if d.get("status") == "no_tie_sweep")
    lines.append(
        f"- converged: {n_converged}; not bracketed: {n_bracket}; no-tie-sweep: {n_no_tie}"
    )
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_refinement_manifest(
    path: Path,
    *,
    positive_families: list[str],
    safety_families: list[str],
    metrics_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    """Emit the spec section 7 empty-shortlist refinement manifest.

    Recommendations are concrete (which parameter, which direction, which
    family) per the task brief.  Order follows spec section 5
    "family expansion order".
    """
    lines: list[str] = []
    lines.append("# Phase V WP1c shortlist refinement manifest\n")
    lines.append(
        f"Generated {_utc_now_iso()}.  "
        "Shortlist failed the conjunctive stopping rule "
        f"(need >= {cfg['empty_shortlist_gate']['min_positive_families']} positive "
        f"families AND >= {cfg['empty_shortlist_gate']['min_safety_families']} safety family).\n"
    )
    lines.append("## Observed shortlist\n")
    lines.append(f"- promotable positive families: {positive_families or '(none)'}")
    lines.append(f"- promotable safety families: {safety_families or '(none)'}")
    lines.append("")
    lines.append(
        "Thresholds MUST NOT be relaxed (spec section 4).  Below is the "
        "spec section 5 family-expansion order; pick the next unexhausted "
        "lever and refine the corresponding task family.\n"
    )
    lines.append("## Recommended levers (in priority order)\n")
    # Family A levers (spec section 5 step 1).
    if "A" not in positive_families:
        lines.append("### Family A (jackpot vs stream) -- refine first\n")
        lines.append(
            "- **tie lever**: widen the epsilon band (`eps_band_frac` 0.2 -> 0.4) "
            "and tighten `lambda_tie` bisection (ensure `final_gap_abs <= 1e-12`). "
            "Inspect `candidate_metrics.parquet` for rows where "
            "`contest_gap_norm` hovers in `(0.01, 0.02]` -- those are in the "
            "soft band but cannot promote under strict gate."
        )
        lines.append(
            "- **geometry lever**: add the second-tier shape basis "
            "(e.g. `front_loaded_compensated`, `two_bump`) and sharper "
            "concentration (narrower one-bump `bump_width`, steeper ramp). "
            "Add `L_range` entries up to 16 or 20 if `gamma^L` is still "
            "resolvable."
        )
        lines.append(
            "- **delay lever**: extend `L_range` toward the `kappa^T` "
            "tractability edge while keeping `gamma` fixed; do not exceed "
            "the lessons-2026-04-20 safe horizon."
        )
        lines.append("")
    # Family B levers (spec section 5 step 2).
    if "B" not in positive_families:
        lines.append("### Family B (rare catastrophe vs safe branch) -- refine second\n")
        lines.append(
            "- **warning-depth lever**: include `variant=\"warning_state\"` with "
            "`warning_depth` in `[1, L//2]`; vary `warning_depth` to find "
            "delayed vs early revelation effects."
        )
        lines.append(
            "- **multi-event lever**: switch to `variant=\"multi_event\"` with "
            "`K=3` events at staggered depths (e.g. `event_depths=[2, L//2, L-1]`)."
        )
        lines.append(
            "- **asymmetry lever**: vary `p * gamma^{L-1} * C` while holding "
            "`b_safe = b - p * gamma^{L-1} * C` fixed -- this produces "
            "matched-classical-mean variants with different higher-moment "
            "risk shape (spec section 5 Family B \"matched-classical-value\")."
        )
        lines.append("")
    # Family C levers (spec section 5 step 3).
    if "C" not in safety_families:
        lines.append("### Family C (raw-operator stress) -- refine safety family\n")
        lines.append(
            "- **concentration lever**: raise `beta_raw_multiplier_range` to "
            "`[3.0, 5.0, 8.0]` so `|beta_raw|` more decisively exceeds "
            "`beta_cap` on every visited stage."
        )
        lines.append(
            "- **tail-depth lever**: extend `L_tail_range` to `[10, 12, 16]`; "
            "longer tails widen the visited-distribution support and make "
            "`raw_local_deriv_p90` more separated from `kappa_t_max`."
        )
        lines.append(
            "- **R_terminal lever** (if beta_raw_multiplier is already at "
            "[5, 8]): push `R_terminal` higher (e.g. 20 * R_penalty) to "
            "deepen the negative signed margin regime."
        )
        lines.append("")
    # Families D / E (spec section 5 steps 4, 5).
    lines.append("### Family D / E expansion (if A+B refinement insufficient)\n")
    lines.append(
        "- **Family D**: add matched-classical-value tasks with different "
        "revelation structure (information arrives mid-episode vs at start)."
    )
    lines.append(
        "- **Family E**: add regime-shift / warning-revelation tasks where "
        "the classical planner cannot distinguish pre-shift from post-shift "
        "but the nonlinear target does."
    )
    lines.append("")
    lines.append("## Pipeline invariants (do NOT alter)\n")
    lines.append(
        "- Thresholds are hard-coded in `promotion.*` of the resolved "
        "config; any change requires orchestrator sign-off."
    )
    lines.append(
        "- The 5,000-candidate cap is per invocation; each refinement pass "
        "resets the budget (spec section 13.3)."
    )
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if ts.endswith("+00:00"):
        ts = ts[: -len("+00:00")] + "Z"
    return ts


def _safe_hostname() -> str:
    try:
        return socket.gethostname() or "unknown"
    except Exception:
        return "unknown"


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase V WP1c -- mechanism-first search driver.",
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Optional YAML config (default: inline DEFAULT_CONFIG).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=Path,
                   default=_REPO_ROOT / "results" / "search",
                   help="Output directory for search artifacts.")
    p.add_argument("--families", nargs="+", default=None,
                   help="Subset of families to search (e.g. --families A B).")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Override max_candidates cap (spec section 13.3).")
    p.add_argument("--dry-run", action="store_true",
                   help="Execute the pipeline but skip writing the "
                        "experiment_manifest.json to results/summaries/.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 on success (empty shortlist is valid)."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = _load_config(args.config)
    if args.families is not None:
        cfg["families"] = list(args.families)

    result = run_search(
        cfg,
        output_root=args.output_root,
        seed=int(args.seed),
        families_override=args.families,
        max_candidates_override=args.max_candidates,
        dry_run=bool(args.dry_run),
        exact_argv=(sys.argv if argv is None else list(argv)),
    )

    logger.info(
        "search complete: admitted=%d promoted=%d refinement_required=%s elapsed=%.2fs",
        result["n_admitted"],
        result["n_promoted"],
        result["refinement_required"],
        result["elapsed_sec"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
