"""Phase V WP2 — Family D: early-warning preventive intervention.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` sections
14.1, 14.4 (Family D fallback / early-warning preventive intervention)
and ``tasks/todo.md`` task #14.

Two-branch contest at ``x_c = (t_c, s_c) = (0, 0)``.  At ``s_start`` the
agent chooses at stage ``t = 0``:

* ``a_nominal = 0``.  Continues along a nominal chain of length
  ``L - 1 - warning_lead`` zero-reward steps.  At stage
  ``L - 1 - warning_lead`` a Bernoulli event with success probability
  ``p_warn`` reveals whether a warning has fired: on success the agent
  transits into ``s_warn_p``, on failure into ``s_warn_n``.  The
  warning-fired chain runs ``warning_lead`` additional steps and then
  pays ``-C`` at stage ``t = L - 1`` with conditional probability
  ``p_cat_given_warn``.  The no-warning chain runs the same number of
  zero-reward steps and absorbs at ``s_safe_end``.
* ``a_preventive = 1``.  Pays ``-lam`` immediately (at stage ``t = 0``)
  and transitions deterministically through the preventive chain to
  ``s_safe_end`` with no further reward.

Key mechanism (spec §14.4): the safe weighted-LSE operator propagates
the warning-conditioned future back through ``s_warn_p`` faster than the
classical Bellman operator, so the classical tie ``lam_tie`` becomes a
safe-vs-classical action flip at the decision point without requiring
the agent to observe the warning before acting.

Closed-form classical tie (single_warning)::

    lam_tie = p_warn * p_cat_given_warn * gamma^{L - 1} * C.

Closed-form classical tie (shallow_early_warning)::

    lam_tie = p_warn * (p_cat_given_warn * gamma^{L - 1} * C
                        - gamma^{L - 1 - warning_lead} * r_shallow).

Sign rationale: ``V_preventive(s_start) = -lam`` and
``V_nominal(s_start) = p_warn * (gamma^{L-1-warning_lead} * r_shallow
  - p_cat_given_warn * gamma^{L-1} * C)``.  Setting
``V_nominal == V_preventive`` ⇒
``lam = -V_nominal``, i.e. the expression above.

Variants (``psi["variant"]``):
    * ``"single_warning"``           — baseline; ``r_shallow = 0``.
    * ``"shallow_early_warning"``   — pays ``r_shallow`` at the warning
                                       state when warning fires.

State layout (total states ``S = 2 * L + warning_lead + 1 <= 3L``):
    0                                           contest s_start
    1, ..., L - 1 - warning_lead                nominal pre-split chain
                                                 (empty iff L - 1 - warning_lead == 0)
    (warn_p head), (warn_p tail states)         warning-fired chain of
                                                 length ``warning_lead``
    (warn_n head), (warn_n tail states)         no-warning chain of
                                                 length ``warning_lead``
    preventive_0, ..., preventive_{L - 2}       preventive chain
                                                 (``L - 1`` states)
    s_catastrophe (absorbing)                   absorbing terminal (-C edge)
    s_safe_end   (absorbing)                    shared safe terminal
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
    "build_family_d_mdp",
    "family_d",
    "contest_state",
]


VARIANT_NAMES: tuple[str, ...] = (
    "single_warning",
    "shallow_early_warning",
)


# ---------------------------------------------------------------------------
# Closed-form tie
# ---------------------------------------------------------------------------

def lam_tie_closed_form(
    p_warn: float,
    p_cat_given_warn: float,
    gamma: float,
    L: int,
    C: float,
    r_shallow: float = 0.0,
    warning_lead: int = 2,
) -> float:
    """Classical tie parameter (preventive cost at indifference).

    Derivation
    ----------
    Let stage indexing follow the MushroomRL convention
    ``V_0 = sum_{t=0}^{L-1} gamma^t r_t`` (reward ``r_t`` collected on the
    transition out of stage ``t``).  The warning-split transition happens
    at stage ``t = L - 1 - warning_lead`` (discount
    ``gamma^{L - 1 - warning_lead}``) and the catastrophe transition at
    stage ``t = L - 1`` (discount ``gamma^{L - 1}``).  Therefore:

        V_nominal(s_start) = p_warn * (gamma^{L - 1 - warning_lead}
                                       * r_shallow
                                       - p_cat_given_warn
                                       * gamma^{L - 1} * C).

    The preventive branch pays ``-lam`` on the very first transition
    (stage ``t = 0``, discount ``1``) and nothing afterwards, so
    ``V_preventive(s_start) = -lam``.  Setting the two equal:

        lam_tie = p_warn * (p_cat_given_warn * gamma^{L - 1} * C
                            - gamma^{L - 1 - warning_lead} * r_shallow).

    For ``r_shallow = 0`` this reduces to
    ``p_warn * p_cat_given_warn * gamma^{L - 1} * C``.
    """
    L_int = int(L)
    k = int(warning_lead)
    if L_int < 3:
        raise ValueError(f"L must be >= 3; got L={L}")
    if not (1 <= k <= L_int - 1):
        raise ValueError(
            f"warning_lead must lie in [1, L-1]=[1, {L_int - 1}]; "
            f"got warning_lead={k}"
        )
    gamma_f = float(gamma)
    if not (0.0 < gamma_f < 1.0):
        raise ValueError(f"gamma must lie in (0, 1); got gamma={gamma}")
    p_w = float(p_warn)
    if not (0.0 < p_w < 1.0):
        raise ValueError(f"p_warn must lie in (0, 1); got p_warn={p_warn}")
    p_cat = float(p_cat_given_warn)
    if not (0.0 < p_cat <= 1.0):
        raise ValueError(
            f"p_cat_given_warn must lie in (0, 1]; got p_cat={p_cat_given_warn}"
        )
    C_f = float(C)
    if C_f <= 0.0:
        raise ValueError(f"C must be > 0; got C={C}")
    r_sh = float(r_shallow)

    cat_term = p_cat * (gamma_f ** (L_int - 1)) * C_f
    shallow_term = (gamma_f ** (L_int - 1 - k)) * r_sh
    return float(p_w * (cat_term - shallow_term))


# ---------------------------------------------------------------------------
# MDP construction
# ---------------------------------------------------------------------------

def _validate_psi(psi: dict[str, Any]) -> dict[str, Any]:
    """Return a dict of validated ``(variant, L, gamma, p_warn, p_cat, C,
    warning_lead, r_shallow)`` values.  Raises ``ValueError`` on any
    invariant violation.
    """
    variant = str(psi.get("variant", "single_warning"))
    if variant not in VARIANT_NAMES:
        raise ValueError(
            f"unknown variant {variant!r}; must be one of {VARIANT_NAMES!r}."
        )
    L = int(psi["L"])
    if L < 3:
        raise ValueError(f"L must be >= 3; got L={L}")
    gamma = float(psi["gamma"])
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"gamma must lie in (0, 1); got gamma={gamma}")
    p_warn = float(psi["p_warn"])
    if not (0.0 < p_warn < 1.0):
        raise ValueError(f"p_warn must lie in (0, 1); got p_warn={p_warn}")
    p_cat = float(psi.get("p_cat_given_warn", 1.0))
    if not (0.0 < p_cat <= 1.0):
        raise ValueError(
            f"p_cat_given_warn must lie in (0, 1]; got p_cat_given_warn={p_cat}"
        )
    C = float(psi["C"])
    if C <= 0.0:
        raise ValueError(f"C must be > 0; got C={C}")
    warning_lead = int(psi.get("warning_lead", 2))
    if warning_lead < 1:
        raise ValueError(
            f"warning_lead must be >= 1; got warning_lead={warning_lead}"
        )
    if warning_lead > L - 1:
        raise ValueError(
            f"warning_lead must be <= L - 1 = {L - 1}; "
            f"got warning_lead={warning_lead}"
        )
    r_shallow = float(psi.get("r_shallow", 0.0))
    if variant == "single_warning" and r_shallow != 0.0:
        raise ValueError(
            f"variant 'single_warning' requires r_shallow == 0.0; "
            f"got r_shallow={r_shallow}"
        )
    return {
        "variant": variant,
        "L": L,
        "gamma": gamma,
        "p_warn": p_warn,
        "p_cat_given_warn": p_cat,
        "C": C,
        "warning_lead": warning_lead,
        "r_shallow": r_shallow,
    }


def build_family_d_mdp(lam: float, psi: dict[str, Any]) -> FiniteMDP:
    """Build the Family D preventive-intervention MDP.

    Parameters
    ----------
    lam : float
        Preventive cost paid on the ``s_start -> s_preventive_0`` edge.
        Passing ``lam = lam_tie_closed_form(...)`` hits the classical tie
        (``|Delta_0(x_c)| <= 1e-8`` in exact DP).
    psi : dict
        Required keys: ``L`` (int >= 3), ``gamma`` (float in (0, 1)),
        ``p_warn`` (float in (0, 1)), ``C`` (float > 0).  Optional keys:
        ``variant`` (one of :data:`VARIANT_NAMES`; default
        ``"single_warning"``), ``p_cat_given_warn`` (float in (0, 1];
        default ``1.0``), ``warning_lead`` (int in ``[1, L - 1]``;
        default ``2``), ``r_shallow`` (float; default ``0.0``; must be
        ``0.0`` for ``single_warning``).

    Returns
    -------
    FiniteMDP
        MushroomRL ``FiniteMDP`` with horizon ``L``, 2 actions, and a
        point-mass initial distribution at ``s_start = 0``.  Exposes
        attributes ``warning_state_idx``, ``no_warning_state_idx``,
        ``s_catastrophe_idx``, ``s_safe_end_idx`` for introspection.
    """
    v = _validate_psi(psi)
    L: int = v["L"]
    gamma: float = v["gamma"]
    p_warn: float = v["p_warn"]
    p_cat: float = v["p_cat_given_warn"]
    C: float = v["C"]
    k: int = v["warning_lead"]
    r_shallow: float = v["r_shallow"]
    lam_f = float(lam)

    # State index layout.
    #   0                              : s_start (contest)
    #   [pre_lo, pre_hi)               : nominal pre-split chain (size = L - 1 - k)
    #   warn_p_lo = pre_hi             : s_warn_p head; warning-fired chain
    #                                    occupies [warn_p_lo, warn_p_lo + k)
    #   warn_n_lo = warn_p_lo + k      : s_warn_n head; no-warning chain
    #                                    occupies [warn_n_lo, warn_n_lo + k)
    #   prev_lo = warn_n_lo + k        : preventive chain, size L - 1
    #   s_catastrophe = prev_lo + L - 1
    #   s_safe_end    = s_catastrophe + 1
    pre_lo = 1
    pre_hi = 1 + (L - 1 - k)                          # empty iff L - 1 - k == 0
    warn_p_lo = pre_hi
    warn_p_hi = warn_p_lo + k
    warn_n_lo = warn_p_hi
    warn_n_hi = warn_n_lo + k
    prev_lo = warn_n_hi
    prev_hi = prev_lo + (L - 1)
    s_cat = prev_hi
    s_safe = s_cat + 1
    S = s_safe + 1                                    # == 2*L + k + 1
    A = 2

    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A, S), dtype=np.float64)

    # ---- Contest state s=0 ------------------------------------------------
    # a_nominal=0: step into the nominal pre-split chain (or straight into
    # the warning split if pre_hi == pre_lo, i.e. L - 1 - k == 0).  Reward 0.
    if pre_hi > pre_lo:
        # Enter first pre-split state.
        P[0, 0, pre_lo] = 1.0
    else:
        # No pre-split chain: s=0 directly splits into s_warn_p / s_warn_n at stage 0.
        # The shallow reward (if any) is paid on this split edge (stage 0).
        _wire_split_edges(
            P, R, from_state=0, action=0,
            warn_p_lo=warn_p_lo, warn_n_lo=warn_n_lo,
            p_warn=p_warn, r_shallow=r_shallow,
        )

    # a_preventive=1: step into s_preventive_0, paying -lam on the edge.
    P[0, 1, prev_lo] = 1.0
    R[0, 1, prev_lo] = -lam_f

    # ---- Nominal pre-split chain ----------------------------------------
    # Deterministic zero-reward transitions between consecutive pre-split
    # states; the final pre-split state splits into warn_p / warn_n under
    # both actions (action is a no-op after stage 0).
    for i in range(pre_lo, pre_hi - 1):
        P[i, :, i + 1] = 1.0
    if pre_hi > pre_lo:
        # Last pre-split state -> warning split.
        last_pre = pre_hi - 1
        for a in range(A):
            _wire_split_edges(
                P, R, from_state=last_pre, action=a,
                warn_p_lo=warn_p_lo, warn_n_lo=warn_n_lo,
                p_warn=p_warn, r_shallow=r_shallow,
            )

    # ---- Warning-fired chain (length k) ----------------------------------
    # States warn_p_lo .. warn_p_hi - 1.  From warn_p_lo the agent walks
    # deterministically ``k - 1`` zero-reward steps; the final state
    # (warn_p_hi - 1) transitions with probability ``p_cat`` to
    # s_catastrophe (paying -C) and with probability ``1 - p_cat`` to
    # s_safe_end (paying 0).
    for i in range(warn_p_lo, warn_p_hi - 1):
        P[i, :, i + 1] = 1.0
    last_warn_p = warn_p_hi - 1
    # Transition from the last warning-fired state happens at stage
    # ``L - 1`` (catastrophe-fire stage).
    P[last_warn_p, :, s_cat] = p_cat
    R[last_warn_p, :, s_cat] = -C
    P[last_warn_p, :, s_safe] = 1.0 - p_cat
    # R[last_warn_p, :, s_safe] = 0.0  (already zero)

    # ---- No-warning chain (length k) ------------------------------------
    # All zero rewards; the final state absorbs into s_safe_end.
    for i in range(warn_n_lo, warn_n_hi - 1):
        P[i, :, i + 1] = 1.0
    last_warn_n = warn_n_hi - 1
    P[last_warn_n, :, s_safe] = 1.0

    # ---- Preventive chain (length L - 1) --------------------------------
    # Deterministic zero-reward transitions; last state absorbs into s_safe_end.
    for i in range(prev_lo, prev_hi - 1):
        P[i, :, i + 1] = 1.0
    last_prev = prev_hi - 1
    P[last_prev, :, s_safe] = 1.0

    # ---- Absorbing terminals --------------------------------------------
    P[s_cat, :, s_cat] = 1.0
    P[s_safe, :, s_safe] = 1.0

    mdp = build_finite_mdp(
        P=P,
        R=R,
        gamma=gamma,
        horizon=L,
        initial_state=0,
    )
    # Introspection hooks (mirrors Family B's warning_state attribute).
    mdp.warning_state_idx = int(warn_p_lo)
    mdp.no_warning_state_idx = int(warn_n_lo)
    mdp.s_catastrophe_idx = int(s_cat)
    mdp.s_safe_end_idx = int(s_safe)
    return mdp


def _wire_split_edges(
    P: np.ndarray,                       # (S, A, S)
    R: np.ndarray,                       # (S, A, S)
    *,
    from_state: int,
    action: int,
    warn_p_lo: int,
    warn_n_lo: int,
    p_warn: float,
    r_shallow: float,
) -> None:
    """Wire the warning-split edges from ``(from_state, action)``.

    On the ``s -> s_warn_p`` edge pays ``r_shallow`` (zero for
    ``single_warning``); on the ``s -> s_warn_n`` edge pays zero.  The
    split edge carries the warning-stage reward when the shallow variant
    is selected — the discount at this stage is
    ``gamma^{L - 1 - warning_lead}`` as required by the tie formula.
    """
    P[from_state, action, warn_p_lo] = p_warn
    R[from_state, action, warn_p_lo] = float(r_shallow)
    P[from_state, action, warn_n_lo] = 1.0 - p_warn
    # R[from_state, action, warn_n_lo] = 0.0  (already zero)


# ---------------------------------------------------------------------------
# FamilySpec
# ---------------------------------------------------------------------------

contest_state: ContestState = ContestState(t=0, s=0, action_pair=(0, 1))


def _warm_start(psi: dict[str, Any]) -> float:
    """Closed-form warm-start for the preventive cost tie."""
    return lam_tie_closed_form(
        p_warn=float(psi["p_warn"]),
        p_cat_given_warn=float(psi.get("p_cat_given_warn", 1.0)),
        gamma=float(psi["gamma"]),
        L=int(psi["L"]),
        C=float(psi["C"]),
        r_shallow=float(psi.get("r_shallow", 0.0)),
        warning_lead=int(psi.get("warning_lead", 2)),
    )


def _scan_bracket(psi: dict[str, Any]) -> tuple[float, float]:
    """Bisection bracket: ``(0, 2 * C)``.

    Preventive cost must be non-negative by construction; ``2 * C`` is a
    loose upper bound covering every admissible tie value since
    ``lam_tie <= p_warn * p_cat * gamma^{L-1} * C <= C``.
    """
    C = float(psi["C"])
    return (0.0, 2.0 * C)


family_d: FamilySpec = FamilySpec(
    name="preventive_intervention",
    build_mdp=build_family_d_mdp,
    contest_state=contest_state,
    warm_start_lambda=_warm_start,
    scan_bracket=_scan_bracket,
    metadata={
        "family": "D",
        "description": (
            "early-warning preventive intervention; safe operator propagates "
            "warning backward faster than classical at the decision point"
        ),
        "tie_parameter": "lam (preventive cost)",
        "variants": VARIANT_NAMES,
        "geometry_parameter_keys": (
            "L", "C", "gamma", "p_warn", "p_cat_given_warn",
            "warning_lead", "r_shallow", "variant",
        ),
    },
)
