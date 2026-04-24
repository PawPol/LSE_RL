"""Phase V WP2 — Family C unit tests.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5
Family C and ``tasks/todo.md`` WP2 test checklist.

Two checks:

1. **Stress diagnostic.**  Build the Family C MDP at
   ``psi["beta_raw_multiplier"] = 2.5``, synthesize a schedule where
   ``beta_raw = multiplier * beta_cap``, compute raw-operator local
   derivative via ``evaluate_candidate``, and assert
   ``raw_local_deriv_stats["p90"] > certified_bound`` and
   ``clip_fraction > 0.05``.
2. **Sanity.**  Under the safe clipped schedule,
   ``raw_convergence_status == "not_evaluated"`` (WP4 will overwrite).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.weighted_lse_dp.search.candidate_metrics import (  # noqa: E402
    evaluate_candidate,
)
from experiments.weighted_lse_dp.search.family_spec import (  # noqa: E402
    ContestState,
)
from experiments.weighted_lse_dp.search.reference_occupancy import (  # noqa: E402
    compute_d_ref,
)
from experiments.weighted_lse_dp.tasks.family_c_raw_stress import (  # noqa: E402
    build_family_c_mdp,
    contest_state,
    family_c,
)


# ---------------------------------------------------------------------------
# Helpers: build the raw schedule used by the stress test
# ---------------------------------------------------------------------------

def _build_schedule(T: int, beta_cap_value: float, multiplier: float) -> dict:
    """Return a schedule dict consumed by ``evaluate_candidate``.

    Parameters
    ----------
    T : int
        Horizon (``mdp.info.horizon``).
    beta_cap_value : float
        Per-stage clip cap.  ``beta_cap_t`` is set to a constant across
        stages for the stress test (Phase V WP2 uses a stationary cap).
    multiplier : float
        ``beta_raw = multiplier * beta_cap`` — controls how far above
        the certified cap the raw schedule sits.

    Returns
    -------
    dict with ``beta_used_t``, ``beta_cap_t``, ``beta_raw_t`` each shape ``(T,)``.
    """
    beta_cap_t = np.full((T,), float(beta_cap_value), dtype=np.float64)
    beta_raw_t = float(multiplier) * beta_cap_t
    # Safe "used" schedule = clip(beta_raw, -beta_cap, beta_cap) = sign(beta_raw) * beta_cap.
    beta_used_t = np.clip(beta_raw_t, -beta_cap_t, beta_cap_t)
    return {
        "beta_used_t": beta_used_t,
        "beta_cap_t": beta_cap_t,
        "beta_raw_t": beta_raw_t,
    }


# ---------------------------------------------------------------------------
# 1. Stress diagnostic
# ---------------------------------------------------------------------------

def test_family_c_stress_diagnostic() -> None:
    """p90 of raw local derivative > certified bound AND clip_fraction > 0.05."""
    L_tail = 6
    R_penalty = 1.0
    R_terminal = 10.0
    gamma = 0.99
    multiplier = 2.5
    psi = {
        "L_tail": L_tail,
        "R_penalty": R_penalty,
        "R_terminal": R_terminal,
        "gamma": gamma,
        "beta_raw_multiplier": multiplier,
    }
    mdp = build_family_c_mdp(lam=0.0, psi=psi)

    # Schedule: a modest beta_cap so beta_used * (|r - V|) stays small
    # (the safe DP collapses close to classical), and a raw schedule that
    # sits 2.5x above the cap to trigger clipping on every stage.
    beta_cap_value = 0.3
    schedule = _build_schedule(
        T=int(mdp.info.horizon),
        beta_cap_value=beta_cap_value,
        multiplier=multiplier,
    )
    assert schedule["beta_raw_t"].shape == (L_tail,)
    assert np.all(schedule["beta_raw_t"] > schedule["beta_cap_t"]), (
        "raw schedule must exceed cap on every stage by construction"
    )

    # Certified bound: gamma is the classical contraction ceiling; the
    # safe certificate sits at gamma as well for stationary schedules,
    # so we use gamma.max() as the floor.  The raw derivative under the
    # designed stress regime approaches (1 + gamma) ~= 1.99, well above
    # gamma ~= 0.99.
    certified_bound = float(gamma)

    metrics = evaluate_candidate(
        mdp=mdp,
        schedule=schedule,
        contest_state=contest_state,
    )

    raw_stats = metrics["raw_local_deriv_stats"]
    p90 = float(raw_stats["p90"])
    clip_fraction = float(metrics["clip_fraction"])

    assert p90 > certified_bound, (
        f"Family C design failure: raw p90 derivative {p90:.4e} does not "
        f"exceed certified bound {certified_bound:.4e} on visited transitions"
    )
    assert clip_fraction > 0.05, (
        f"Family C design failure: clip_fraction {clip_fraction:.4e} <= 0.05"
    )


# ---------------------------------------------------------------------------
# 2. Sanity: raw_convergence_status == "not_evaluated" under WP1a
# ---------------------------------------------------------------------------

def test_family_c_raw_convergence_sentinel() -> None:
    """evaluate_candidate returns ``raw_convergence_status == 'not_evaluated'``.

    WP1a does not classify raw convergence; that's WP4's job.  This test
    guards against accidental drift in the WP1a sentinel contract.
    """
    psi = {
        "L_tail": 4,
        "R_penalty": 1.0,
        "R_terminal": 5.0,
        "gamma": 0.95,
    }
    mdp = build_family_c_mdp(lam=0.0, psi=psi)
    schedule = _build_schedule(
        T=int(mdp.info.horizon),
        beta_cap_value=0.2,
        multiplier=1.0,
    )  # multiplier=1 => clip fraction near-zero, still covered by the WP1a path.
    metrics = evaluate_candidate(
        mdp=mdp,
        schedule=schedule,
        contest_state=contest_state,
    )
    assert metrics["raw_convergence_status"] == "not_evaluated"


# ---------------------------------------------------------------------------
# Bonus: FamilySpec integrity
# ---------------------------------------------------------------------------

def test_family_c_spec_integrity() -> None:
    """family_c is a well-formed FamilySpec with Family C contest conventions."""
    assert family_c.name == "raw_stress"
    assert isinstance(family_c.contest_state, ContestState)
    assert family_c.contest_state.t == 0
    assert family_c.contest_state.s == 0
    assert family_c.contest_state.action_pair == (0, 1)
    # lam is unused for Family C — warm_start_lambda returns 0.0.
    assert family_c.warm_start_lambda({"dummy": True}) == 0.0
    lo, hi = family_c.scan_bracket({"dummy": True})
    assert lo == -1.0 and hi == 1.0


def test_family_c_d_ref_reachability() -> None:
    """Contest state d_ref[0, 0] = 1.0 under point-mass initial state."""
    psi = {
        "L_tail": 5,
        "R_penalty": 1.0,
        "R_terminal": 8.0,
        "gamma": 0.99,
    }
    mdp = build_family_c_mdp(lam=0.0, psi=psi)
    T = int(mdp.info.horizon)
    S = int(mdp.p.shape[0])
    A = int(mdp.p.shape[1])
    # Deterministic all-zeros policy (both actions are identical).
    pi_uniform = np.full((T, S, A), 1.0 / A, dtype=np.float64)
    occ = compute_d_ref(mdp=mdp, pi_classical=pi_uniform, pi_safe=pi_uniform)
    assert abs(float(occ["d_ref"][0, 0]) - 1.0) <= 1e-12
