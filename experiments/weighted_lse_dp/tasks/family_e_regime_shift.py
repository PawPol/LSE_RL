"""Phase V WP2 — Family E: regime-shift / warning-revelation with
concentration contrast.

Spec: ``docs/specs/phase_V_mechanism_experiments.md`` section 15
(Family E addendum, 2026-04-24); ``tasks/todo.md`` task #15.

Family E keeps the regime-shift narrative of Families B / D but routes
the decisive event through a **single-step concentrated reward/cost**
rather than spreading it across the horizon.  Pure propagation-depth
designs (B, D) dilute value translation; concentration-contrast
designs (A) translate cleanly.  Family E merges the two.

Variants (``psi["variant"]``)
-----------------------------
* ``"E1_warning_react"``   — ``a_ignore=0`` (smooth ``r_smooth`` each
  step) vs ``a_react=1`` (pay ``-lam``, avoid ``-C_conc`` firing at
  ``t_cat = L - warning_lead`` w.p. ``p_warn`` on the ignore branch).
* ``"E2_opportunity_adapt"`` — ``a_wait=0`` (``r_baseline`` each step)
  vs ``a_adapt=1`` (pay ``-lam``, catch ``+U_conc`` at
  ``t_opp = warning_lead`` w.p. ``p_opp``).
* ``"E3_regime_switch"``   — ``a_stay=0`` (``r_smooth`` each step) vs
  ``a_switch=1`` (pay ``-lam``, collect ``+R_rev`` at ``t_switch``;
  tail zero).

Closed-form ties (``G_L(gamma) = sum_{k=0}^{L-1} gamma^k``, sign-
consistent with ``V(action=1, s_start) = -lam_tie``)::

    E1 : lam_tie = p_warn * gamma^{t_cat} * C_conc - G_L * r_smooth
    E2 : lam_tie = p_opp * gamma^{t_opp} * U_conc - G_L * r_baseline
    E3 : lam_tie = gamma^{t_switch} * R_rev - G_L * r_smooth

Sign convention: ``lam_tie`` is the action-1 stage-0 cost; tie
condition is ``V_action=1(s_start) = V_action=0(s_start)``, i.e.
``-lam_tie + E_active_no_lam = E_passive``.

E1 sign correction: the orchestrator prompt literally writes ``G_L *
r_smooth - p_warn * gamma^{t_cat} * C_conc`` which equals
``E[ignore]`` (the expected passive return), not ``-E[ignore]``.
Shipping the sign-consistent form here so exact DP hits the tie
(``|Delta_0(x_c)| <= 1e-8``).  Precedent: Family D WP2 open question
(2026-04-23).

Concentration-contrast invariant: the concentrated value appears as the
exact reward on a SINGLE ``(s, a, s')`` transition (E1: reward
``r_smooth - C_conc``; E2: reward ``+U_conc``; E3: reward ``+R_rev``).

State budgets: every variant satisfies ``S <= 3L + 4``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.weighted_lse_dp.search.family_spec import (
    ContestState,
    FamilySpec,
)

from ._family_helpers import FiniteMDP, build_finite_mdp

__all__ = [
    "VARIANT_NAMES",
    "lam_tie_closed_form",
    "build_family_e_mdp",
    "family_e",
    "contest_state",
]


VARIANT_NAMES: tuple[str, ...] = (
    "E1_warning_react",
    "E2_opportunity_adapt",
    "E3_regime_switch",
)


# ---------------------------------------------------------------------------
# Closed-form tie (dispatches on variant)
# ---------------------------------------------------------------------------

def _geometric_sum(gamma: float, L: int) -> float:
    """``G_L(gamma) = sum_{k=0}^{L-1} gamma^k = (1 - gamma^L) / (1 - gamma)``."""
    g = float(gamma)
    L_int = int(L)
    if g == 1.0:
        return float(L_int)
    return (1.0 - g ** L_int) / (1.0 - g)


def _lam_tie_E1(psi: dict[str, Any]) -> float:
    """E1 tie: ``lam_tie = p_warn * gamma^{t_cat} * C_conc - G_L * r_smooth``.

    Sign-consistent form (matches V(ignore) = V(react) = -lam_tie).
    The task prompt literally writes
    ``G_L * r_smooth - p_warn * gamma^{t_cat} * C_conc`` which equals
    ``E[ignore]``; the true tie is ``-E[ignore] + E[react] == -E[ignore]``
    under ``E[react_no_lam] = 0``.  We ship the sign-consistent form so
    that ``V_react(s_start) = -lam_tie == E[ignore]`` and the exact-DP
    residual ``|Delta_0(x_c)|`` evaluates to zero (spec §5 step 2, WP2
    open-question precedent set by Family D).
    """
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    warning_lead = int(psi["warning_lead"])
    p_warn = float(psi["p_warn"])
    C_conc = float(psi["C_conc"])
    r_smooth = float(psi["r_smooth"])
    t_cat = L - warning_lead
    return float(p_warn * (gamma ** t_cat) * C_conc
                 - _geometric_sum(gamma, L) * r_smooth)


def _lam_tie_E2(psi: dict[str, Any]) -> float:
    """E2 tie: ``p_opp * gamma^{t_opp} * U_conc - G_L * r_baseline``."""
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    t_opp = int(psi["warning_lead"])
    p_opp = float(psi["p_opp"])
    U_conc = float(psi["U_conc"])
    r_baseline = float(psi["r_baseline"])
    return float(p_opp * (gamma ** t_opp) * U_conc
                 - _geometric_sum(gamma, L) * r_baseline)


def _lam_tie_E3(psi: dict[str, Any]) -> float:
    """E3 tie: ``gamma^{t_switch} * R_rev - G_L * r_smooth``."""
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    t_switch = int(psi["t_switch"])
    R_rev = float(psi["R_rev"])
    r_smooth = float(psi["r_smooth"])
    return float((gamma ** t_switch) * R_rev
                 - _geometric_sum(gamma, L) * r_smooth)


def lam_tie_closed_form(psi: dict[str, Any]) -> float:
    """Classical tie for Family E; dispatches on ``psi["variant"]``.

    Returns ``lam_tie`` such that ``Delta_0(x_contest; lam_tie, psi) ~ 0``
    at ``(t=0, s=0)`` with action pair ``(0, 1)`` via exact DP.
    """
    _validate_psi(psi)
    variant = str(psi["variant"])
    if variant == "E1_warning_react":
        return _lam_tie_E1(psi)
    if variant == "E2_opportunity_adapt":
        return _lam_tie_E2(psi)
    if variant == "E3_regime_switch":
        return _lam_tie_E3(psi)
    raise ValueError(  # pragma: no cover - guarded by _validate_psi
        f"unknown variant {variant!r}; must be one of {VARIANT_NAMES!r}."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require_present(psi: dict[str, Any], keys: tuple[str, ...], tag: str) -> None:
    """Raise ``ValueError`` if any of ``keys`` is missing from ``psi``."""
    for k in keys:
        if k not in psi:
            raise ValueError(f"{tag} psi missing required key {k!r}.")


def _require_stage(name: str, val: int, L: int) -> None:
    """Enforce ``1 <= val <= L - 1``; raise ``ValueError`` otherwise."""
    if not (1 <= val <= L - 1):
        raise ValueError(
            f"{name} must lie in [1, L-1]=[1, {L - 1}]; got {name}={val}"
        )


def _validate_psi(psi: dict[str, Any]) -> None:
    """Raise ``ValueError`` on every documented invariant violation."""
    variant = str(psi.get("variant", ""))
    if variant not in VARIANT_NAMES:
        raise ValueError(
            f"unknown variant {variant!r}; must be one of {VARIANT_NAMES!r}."
        )
    _require_present(psi, ("L", "gamma"), tag=variant)
    L = int(psi["L"])
    if L < 3:
        raise ValueError(f"L must be >= 3; got L={L}")
    gamma = float(psi["gamma"])
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"gamma must lie in (0, 1); got gamma={gamma}")

    if variant == "E1_warning_react":
        _require_present(
            psi, ("warning_lead", "p_warn", "C_conc", "r_smooth"), tag="E1"
        )
        _require_stage("warning_lead", int(psi["warning_lead"]), L)
        p_warn = float(psi["p_warn"])
        if not (0.0 < p_warn < 1.0):
            raise ValueError(f"p_warn must lie in (0, 1); got p_warn={p_warn}")
        if float(psi["C_conc"]) <= 0.0:
            raise ValueError(f"C_conc must be > 0; got C_conc={psi['C_conc']}")
        if float(psi["r_smooth"]) < 0.0:
            raise ValueError(f"r_smooth must be >= 0; got r_smooth={psi['r_smooth']}")
        return

    if variant == "E2_opportunity_adapt":
        _require_present(
            psi, ("warning_lead", "p_opp", "U_conc", "r_baseline"), tag="E2"
        )
        _require_stage("warning_lead", int(psi["warning_lead"]), L)
        p_opp = float(psi["p_opp"])
        if not (0.0 < p_opp <= 1.0):
            raise ValueError(f"p_opp must lie in (0, 1]; got p_opp={p_opp}")
        if float(psi["U_conc"]) <= 0.0:
            raise ValueError(f"U_conc must be > 0; got U_conc={psi['U_conc']}")
        if float(psi["r_baseline"]) < 0.0:
            raise ValueError(
                f"r_baseline must be >= 0; got r_baseline={psi['r_baseline']}"
            )
        return

    # variant == "E3_regime_switch"
    _require_present(psi, ("t_switch", "R_rev", "r_smooth"), tag="E3")
    _require_stage("t_switch", int(psi["t_switch"]), L)
    if float(psi["R_rev"]) <= 0.0:
        raise ValueError(f"R_rev must be > 0; got R_rev={psi['R_rev']}")
    if float(psi["r_smooth"]) < 0.0:
        raise ValueError(f"r_smooth must be >= 0; got r_smooth={psi['r_smooth']}")


# ---------------------------------------------------------------------------
# MDP builders (per variant)
# ---------------------------------------------------------------------------

def _chain_edges(
    P: np.ndarray,                   # (S, A, S')
    R: np.ndarray,                   # (S, A, S')
    *,
    lo: int,
    hi: int,
    reward: float,
) -> None:
    """Wire deterministic chain edges ``s -> s+1`` for ``s in [lo, hi - 1)``.

    Both actions share the transition (non-contest states).  Reward on
    every edge is ``reward``.  No-op when ``hi - lo <= 1``.
    """
    for s in range(int(lo), int(hi) - 1):
        P[s, :, s + 1] = 1.0
        if reward != 0.0:
            R[s, :, s + 1] = float(reward)

def _build_E1(lam: float, psi: dict[str, Any]) -> FiniteMDP:
    """E1 warning_react.  Layout (``t_cat = L - warning_lead``,
    ``tail_len = L - 1 - t_cat``)::

        0 s_start; [1, 1+t_cat) left_pre; [lt_no_lo, lt_no_hi) no-cat
        tail (size max(1, tail_len)); [lt_cat_lo, lt_cat_hi) cat tail
        (same); [right_lo, right_hi) react chain (size L-1); s_end.

    Split at stage t_cat: safe fork pays r_smooth, cat fork pays
    ``r_smooth - C_conc`` (SINGLE concentrated transition).  When
    ``tail_len == 0`` the 1-state stubs self-absorb with reward 0.
    Total ``S <= 3L + 1 <= 3L + 4``.
    """
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    warning_lead = int(psi["warning_lead"])
    p_warn = float(psi["p_warn"])
    C_conc = float(psi["C_conc"])
    r_smooth = float(psi["r_smooth"])
    lam_f = float(lam)
    t_cat = L - warning_lead                                 # stage of split edge
    tail_len = L - 1 - t_cat                                 # tail states per fork

    # State indices.  When tail_len == 0 we still allocate a 1-state
    # "stub" per fork to hold distinct reward labels on incoming edges.
    eff_tail = max(1, tail_len)
    s_start = 0
    left_pre_lo = 1
    left_pre_hi = left_pre_lo + t_cat
    lt_no_lo = left_pre_hi
    lt_no_hi = lt_no_lo + eff_tail
    lt_cat_lo = lt_no_hi
    lt_cat_hi = lt_cat_lo + eff_tail
    right_lo = lt_cat_hi
    right_hi = right_lo + (L - 1)
    s_end = right_hi
    S = s_end + 1
    A = 2

    P = np.zeros((S, A, S), dtype=np.float64)                # (S, A, S')
    R = np.zeros((S, A, S), dtype=np.float64)                # (S, A, S')

    # ---- s_start edges --------------------------------------------------
    P[s_start, 0, left_pre_lo] = 1.0
    R[s_start, 0, left_pre_lo] = r_smooth
    P[s_start, 1, right_lo] = 1.0
    R[s_start, 1, right_lo] = -lam_f

    # ---- left_pre interior edges (pay r_smooth) -------------------------
    _chain_edges(P, R, lo=left_pre_lo, hi=left_pre_hi, reward=r_smooth)

    # ---- split edges at stage t_cat -------------------------------------
    # Safe fork pays r_smooth; cat fork pays r_smooth - C_conc (the SINGLE
    # concentrated catastrophe transition in R).
    split_src = left_pre_hi - 1
    P[split_src, :, lt_no_lo] = 1.0 - p_warn
    R[split_src, :, lt_no_lo] = r_smooth
    P[split_src, :, lt_cat_lo] = p_warn
    R[split_src, :, lt_cat_lo] = r_smooth - C_conc

    # ---- tail interiors + absorb ----------------------------------------
    if tail_len > 0:
        _chain_edges(P, R, lo=lt_no_lo, hi=lt_no_hi, reward=r_smooth)
        _chain_edges(P, R, lo=lt_cat_lo, hi=lt_cat_hi, reward=r_smooth)
        # Terminal tail edges pay r_smooth at stage L-1.
        P[lt_no_hi - 1, :, s_end] = 1.0
        R[lt_no_hi - 1, :, s_end] = r_smooth
        P[lt_cat_hi - 1, :, s_end] = 1.0
        R[lt_cat_hi - 1, :, s_end] = r_smooth
    else:
        # tail_len == 0 stubs: self-absorb into s_end with reward 0
        # (no stage left; stage L reached on the split edge itself).
        P[lt_no_hi - 1, :, s_end] = 1.0
        P[lt_cat_hi - 1, :, s_end] = 1.0

    # ---- react chain (zero rewards; absorbs into s_end) -----------------
    _chain_edges(P, R, lo=right_lo, hi=right_hi, reward=0.0)
    P[right_hi - 1, :, s_end] = 1.0

    # ---- absorbing terminal self-loop ----------------------------------
    P[s_end, :, s_end] = 1.0

    mdp = build_finite_mdp(P=P, R=R, gamma=gamma, horizon=L, initial_state=s_start)
    mdp.s_end_idx = int(s_end)
    mdp.lt_cat_lo_idx = int(lt_cat_lo)
    mdp.t_cat = int(t_cat)
    return mdp


def _build_E2(lam: float, psi: dict[str, Any]) -> FiniteMDP:
    """E2 opportunity_adapt.  Layout (``t_opp = warning_lead``,
    ``post_len = L - 1 - t_opp``)::

        0 s_start; [1, L) left wait chain (L-1 states);
        [right_pre_lo, right_pre_hi) adapt pre-split chain (size
        t_opp >= 1; -lam on entry edge); right_fire (+U_conc on
        incoming edge -- SINGLE concentrated transition); right_no
        (edge 0); [right_post_lo, right_post_hi) shared post-opp tail
        (both forks merge); s_end.  Total ``S = 2L + 2 <= 3L + 4``.
    """
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    t_opp = int(psi["warning_lead"])
    p_opp = float(psi["p_opp"])
    U_conc = float(psi["U_conc"])
    r_baseline = float(psi["r_baseline"])
    lam_f = float(lam)
    post_len = L - 1 - t_opp

    # State indices.
    s_start = 0
    left_lo = 1
    left_hi = left_lo + (L - 1)
    right_pre_lo = left_hi
    right_pre_hi = right_pre_lo + t_opp                      # size t_opp
    right_fire = right_pre_hi
    right_no = right_fire + 1
    right_post_lo = right_no + 1
    right_post_hi = right_post_lo + post_len
    s_end = right_post_hi
    S = s_end + 1
    A = 2

    P = np.zeros((S, A, S), dtype=np.float64)                # (S, A, S')
    R = np.zeros((S, A, S), dtype=np.float64)                # (S, A, S')

    # ---- s_start edges --------------------------------------------------
    P[s_start, 0, left_lo] = 1.0
    R[s_start, 0, left_lo] = r_baseline
    P[s_start, 1, right_pre_lo] = 1.0
    R[s_start, 1, right_pre_lo] = -lam_f

    # ---- wait chain -----------------------------------------------------
    _chain_edges(P, R, lo=left_lo, hi=left_hi, reward=r_baseline)
    P[left_hi - 1, :, s_end] = 1.0
    R[left_hi - 1, :, s_end] = r_baseline

    # ---- adapt pre-split chain (zero interior rewards) ------------------
    _chain_edges(P, R, lo=right_pre_lo, hi=right_pre_hi, reward=0.0)

    # ---- split: right_pre[t_opp-1] -> {right_fire, right_no} ------------
    # +U_conc lives on the right_fire edge (SINGLE concentrated-upside
    # transition in R); right_no edge pays 0.
    split_src = right_pre_hi - 1
    P[split_src, :, right_fire] = p_opp
    R[split_src, :, right_fire] = U_conc
    P[split_src, :, right_no] = 1.0 - p_opp

    # ---- shared post-opp tail (zero rewards) -----------------------------
    if post_len > 0:
        P[right_fire, :, right_post_lo] = 1.0
        P[right_no, :, right_post_lo] = 1.0
        _chain_edges(P, R, lo=right_post_lo, hi=right_post_hi, reward=0.0)
        P[right_post_hi - 1, :, s_end] = 1.0
    else:
        P[right_fire, :, s_end] = 1.0
        P[right_no, :, s_end] = 1.0

    # ---- absorbing terminal self-loop ----------------------------------
    P[s_end, :, s_end] = 1.0

    mdp = build_finite_mdp(P=P, R=R, gamma=gamma, horizon=L, initial_state=s_start)
    mdp.s_end_idx = int(s_end)
    mdp.right_fire_idx = int(right_fire)
    mdp.right_no_idx = int(right_no)
    mdp.t_opp = int(t_opp)
    return mdp


def _build_E3(lam: float, psi: dict[str, Any]) -> FiniteMDP:
    """E3 regime_switch (deterministic).  Layout: ``0 s_start;
    [1, L) stay chain; [right_lo, right_hi) switch chain; s_end``.
    Total ``S = 2L <= 3L + 4``.  Reveal at stage t_switch on the edge
    out of ``right[t_switch - 1]`` to ``right[t_switch]`` (or to
    ``s_end`` when ``t_switch == L - 1``); ``-lam`` on the distinct
    stage-0 edge ``s_start -> right[0]``.
    """
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    t_switch = int(psi["t_switch"])
    R_rev = float(psi["R_rev"])
    r_smooth = float(psi["r_smooth"])
    lam_f = float(lam)

    # State indices.
    s_start = 0
    left_lo = 1
    left_hi = left_lo + (L - 1)
    right_lo = left_hi
    right_hi = right_lo + (L - 1)
    s_end = right_hi
    S = s_end + 1
    A = 2

    P = np.zeros((S, A, S), dtype=np.float64)                # (S, A, S')
    R = np.zeros((S, A, S), dtype=np.float64)                # (S, A, S')

    # ---- s_start edges --------------------------------------------------
    P[s_start, 0, left_lo] = 1.0
    R[s_start, 0, left_lo] = r_smooth
    P[s_start, 1, right_lo] = 1.0
    R[s_start, 1, right_lo] = -lam_f

    # ---- stay chain (r_smooth each step; terminal edge pays r_smooth) ---
    _chain_edges(P, R, lo=left_lo, hi=left_hi, reward=r_smooth)
    P[left_hi - 1, :, s_end] = 1.0
    R[left_hi - 1, :, s_end] = r_smooth

    # ---- switch chain ---------------------------------------------------
    # Reveal at stage t_switch has source right[t_switch - 1] (right[0]
    # is at stage 1 on the right branch), destination right[t_switch]
    # when t_switch <= L - 2 else s_end.
    reveal_src = right_lo + (t_switch - 1)
    reveal_dst = right_lo + t_switch if t_switch <= L - 2 else s_end
    _chain_edges(P, R, lo=right_lo, hi=right_hi, reward=0.0)
    # Overlay the single concentrated reveal reward.
    if reveal_dst != s_end:
        R[reveal_src, :, reveal_dst] = R_rev
    # Terminal edge right[L-2] -> s_end (pays R_rev iff reveal lands here).
    P[right_hi - 1, :, s_end] = 1.0
    if reveal_dst == s_end:
        R[right_hi - 1, :, s_end] = R_rev

    # ---- absorbing terminal self-loop ----------------------------------
    P[s_end, :, s_end] = 1.0

    mdp = build_finite_mdp(P=P, R=R, gamma=gamma, horizon=L, initial_state=s_start)
    mdp.s_end_idx = int(s_end)
    mdp.reveal_edge_dst_idx = int(reveal_dst)
    mdp.reveal_edge_src_idx = int(reveal_src)
    mdp.t_switch = int(t_switch)
    return mdp


# ---------------------------------------------------------------------------
# Public MDP entry point
# ---------------------------------------------------------------------------

def build_family_e_mdp(lam: float, psi: dict[str, Any]) -> FiniteMDP:
    """Build the Family E MDP; dispatches on ``psi["variant"]``.

    ``lam`` is the action-1 stage-0 cost; ``lam_tie_closed_form(psi)``
    hits the classical tie.  Required ``psi`` keys: ``variant``, ``L``,
    ``gamma``; variant-specific keys are documented on the module
    docstring and enforced by :func:`_validate_psi`.
    """
    _validate_psi(psi)
    variant = str(psi["variant"])
    if variant == "E1_warning_react":
        return _build_E1(float(lam), psi)
    if variant == "E2_opportunity_adapt":
        return _build_E2(float(lam), psi)
    return _build_E3(float(lam), psi)                         # E3_regime_switch


# ---------------------------------------------------------------------------
# FamilySpec plumbing
# ---------------------------------------------------------------------------

contest_state: ContestState = ContestState(t=0, s=0, action_pair=(0, 1))


def _warm_start(psi: dict[str, Any]) -> float:
    """Closed-form warm-start for the Family E tie."""
    return lam_tie_closed_form(psi)


def _scan_bracket_family_e(psi: dict[str, Any]) -> tuple[float, float]:
    """Bisection bracket: ``(lam - 2|lam+1e-3|, lam + 2|lam+1e-3|)``,
    clipped to ``[-M, M]`` where ``M`` scales the largest reward
    magnitude in the family by ``L``.
    """
    lam = lam_tie_closed_form(psi)
    half_width = 2.0 * abs(lam + 1e-3)
    variant = str(psi["variant"])
    L = int(psi["L"])
    _reward_key = {
        "E1_warning_react": "C_conc",
        "E2_opportunity_adapt": "U_conc",
        "E3_regime_switch": "R_rev",
    }[variant]
    M = float(psi[_reward_key]) * max(1.0, float(L))
    lo = max(lam - half_width, -M)
    hi = min(lam + half_width, M)
    if hi <= lo:
        hi = lo + 1.0
    return (float(lo), float(hi))


family_e: FamilySpec = FamilySpec(
    name="regime_shift_concentration",
    build_mdp=build_family_e_mdp,
    contest_state=contest_state,
    warm_start_lambda=_warm_start,
    scan_bracket=_scan_bracket_family_e,
    metadata={
        "family": "E",
        "description": (
            "regime-shift / warning-revelation with concentration contrast; "
            "combines the regime-shift narrative of B/D with a single-step "
            "concentrated reward/cost that Family A uses to translate cleanly"
        ),
        "tie_parameter": "lam (adjustment / adaptation / switch cost)",
        "variants": VARIANT_NAMES,
        "geometry_parameter_keys": (
            "L", "gamma", "variant",
            # E1-specific
            "warning_lead", "p_warn", "C_conc", "r_smooth",
            # E2-specific
            "p_opp", "U_conc", "r_baseline",
            # E3-specific
            "t_switch", "R_rev",
        ),
    },
)
