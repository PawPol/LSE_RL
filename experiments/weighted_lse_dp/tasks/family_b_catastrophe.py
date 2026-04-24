"""Phase V WP2 — Family B: rare catastrophe vs safe branch (policy-flip).

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 5
Family B.  Two-branch contest at ``x_c = (t_c, s_c) = (0, 0)``: action A
(rare catastrophe) vs action B (deterministic safe payoff ``b_safe``,
the tie parameter).

Closed-form classical tie (single-event baseline)::

    b_safe_tie(b, p, gamma, L, C) = b - p * gamma^{L-1} * C.

Required variants (``psi["variant"]``):
    * ``"single_event"``         — baseline: impulse ``b`` then -C at L w.p. p.
    * ``"warning_state"``        — labeled warning state on cat sub-branch.
    * ``"shallow_early"``        — ``b_shallow`` for ``k_shallow`` steps,
                                    then delayed catastrophe.
    * ``"multi_event"``          — K events at depths ``k_i`` w.p. ``p_i``.
    * ``"matched_concentration"`` — stream ``b_stream`` + tail catastrophe.
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
    "b_safe_tie_closed_form",
    "branch_a_classical_value",
    "build_family_b_mdp",
    "family_b",
    "contest_state",
]


VARIANT_NAMES: tuple[str, ...] = (
    "single_event",
    "warning_state",
    "shallow_early",
    "multi_event",
    "matched_concentration",
)


# ---------------------------------------------------------------------------
# Closed-form ties (per variant)
# ---------------------------------------------------------------------------

def b_safe_tie_closed_form(
    b: float,
    p: float,
    gamma: float,
    L: int,
    C: float,
) -> float:
    """Single-event closed-form tie: ``b - p * gamma^{L-1} * C``.

    Spec reference: Family B baseline.  Callers that use a non-baseline
    variant should prefer :func:`branch_a_classical_value` which
    dispatches on ``psi["variant"]``.
    """
    return float(b) - float(p) * (float(gamma) ** (int(L) - 1)) * float(C)


def branch_a_classical_value(psi: dict[str, Any]) -> float:
    """Exact classical V of branch A at the contest state — the tie ``b_safe``.

    Per-variant formula (by ``psi["variant"]``):
        single_event          : b - p gamma^{L-1} C
        warning_state         : same as single_event (warning only
                                 changes information, not expected return)
        shallow_early         : b_shallow (1 - gamma^{k_shallow}) / (1 - gamma)
                                 - p gamma^{L-1} C
        multi_event           : b - sum_i p_i gamma^{k_i - 1} C_i
        matched_concentration : b_stream (1 - gamma^L) / (1 - gamma)
                                 - p gamma^{L-1} C
    """
    variant = str(psi.get("variant", "single_event"))
    gamma = float(psi["gamma"])
    L = int(psi["L"])

    if variant == "single_event" or variant == "warning_state":
        b = float(psi["b"])
        p = float(psi["p"])
        C = float(psi["C"])
        return b - p * (gamma ** (L - 1)) * C

    if variant == "shallow_early":
        b_shallow = float(psi["b_shallow"])
        k_shallow = int(psi["k_shallow"])
        p = float(psi["p"])
        C = float(psi["C"])
        # Geometric sum sum_{k=0..k_shallow-1} gamma^k * b_shallow.
        if gamma == 1.0:
            stream = b_shallow * float(k_shallow)
        else:
            stream = b_shallow * (1.0 - gamma ** k_shallow) / (1.0 - gamma)
        return stream - p * (gamma ** (L - 1)) * C

    if variant == "multi_event":
        b = float(psi["b"])
        event_probs = np.asarray(psi["event_probs"], dtype=np.float64)   # (K,)
        event_depths = np.asarray(psi["event_depths"], dtype=np.int64)   # (K,)
        event_mags = np.asarray(psi["event_mags"], dtype=np.float64)     # (K,)
        if event_probs.shape != event_depths.shape or event_probs.shape != event_mags.shape:
            raise ValueError(
                "event_probs / event_depths / event_mags must share shape; "
                f"got {event_probs.shape} / {event_depths.shape} / {event_mags.shape}."
            )
        # V contribution of each event: -p_i * gamma^{k_i - 1} * C_i
        # (depth ``k_i`` from the contest state; reward paid at step k_i,
        # discounted by gamma^{k_i - 1} relative to the first-step baseline).
        exponent = event_depths.astype(np.float64) - 1.0
        contrib = -event_probs * np.power(gamma, exponent) * event_mags
        return float(b + contrib.sum())

    if variant == "matched_concentration":
        b_stream = float(psi["b_stream"])
        p = float(psi["p"])
        C = float(psi["C"])
        if gamma == 1.0:
            stream = b_stream * float(L)
        else:
            stream = b_stream * (1.0 - gamma ** L) / (1.0 - gamma)
        return stream - p * (gamma ** (L - 1)) * C

    raise ValueError(
        f"unknown variant {variant!r}; must be one of {VARIANT_NAMES!r}."
    )


# ---------------------------------------------------------------------------
# MDP construction (per variant)
# ---------------------------------------------------------------------------
#
# Common state-layout conventions:
#   s = 0                     : contest state (2 actions)
#   safe terminal (last idx)  : absorbing; edge s=0 -> safe terminal pays b_safe
#                               on action 1.
# Branch-A states are variant-specific; see each _mdp_<variant> builder.

def build_family_b_mdp(lam: float, psi: dict[str, Any]) -> FiniteMDP:
    """Build the Family B catastrophe-vs-safe MDP for ``psi["variant"]``.

    ``lam`` = safe-branch payoff ``b_safe`` on action 1 at ``s=0``.
    Setting ``lam = branch_a_classical_value(psi)`` gives the classical tie
    ``Delta_0(x_c) == 0``.  ``psi`` shared keys: ``variant``
    (:data:`VARIANT_NAMES`), ``L >= 2``, ``gamma in (0, 1]``.
    Variant-specific keys: see :func:`branch_a_classical_value`.
    """
    variant = str(psi.get("variant", "single_event"))
    L = int(psi["L"])
    gamma = float(psi["gamma"])
    if L < 2:
        raise ValueError(f"L must be >= 2; got L={L}")
    if variant not in VARIANT_NAMES:
        raise ValueError(
            f"unknown variant {variant!r}; must be one of {VARIANT_NAMES!r}."
        )

    b_safe = float(lam)

    dispatch = {
        "single_event": _mdp_single_event,
        "warning_state": _mdp_warning_state,
        "shallow_early": _mdp_shallow_early,
        "multi_event": _mdp_multi_event,
        "matched_concentration": _mdp_matched_concentration,
    }
    return dispatch[variant](psi, b_safe=b_safe, L=L, gamma=gamma)


def _build_split_chain(
    *,
    L: int,
    b_entry: float,
    b_safe: float,
    p: float,
    C: float,
    stream_reward: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared 2-chain topology used by single_event / warning_state / matched_concentration.

    States: 0=contest, 1=branch-A entry, 2..L=catastrophe chain (L absorbing
    after -C), L+1..2L-1=no-cat chain (2L-1 absorbing), 2L=safe terminal.
    s=0->s=1 pays ``b_entry``; s=0->s=2L pays ``b_safe``; the s=1 split and
    every interior chain edge pays ``stream_reward`` (0 for single_event /
    warning_state, ``b_stream`` for matched_concentration); s=L-1->s=L
    additionally pays ``-C``.  Returns ``(P, R)`` each shape ``(S, 2, S)``.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must lie in [0, 1]; got p={p}")
    S = 2 * L + 1
    P = np.zeros((S, 2, S), dtype=np.float64)
    R = np.zeros((S, 2, S), dtype=np.float64)
    sr = float(stream_reward)

    # Contest edges.
    P[0, 0, 1] = 1.0
    R[0, 0, 1] = float(b_entry)
    P[0, 1, 2 * L] = 1.0
    R[0, 1, 2 * L] = float(b_safe)

    # Branch-A split at s=1.
    if L >= 2:
        P[1, :, 2] = p
        P[1, :, L + 1] = 1.0 - p
        R[1, :, 2] = sr
        R[1, :, L + 1] = sr

    # Catastrophe chain; terminal edge additionally pays -C.
    for i in range(2, L):
        P[i, :, i + 1] = 1.0
        R[i, :, i + 1] = sr + (-float(C) if i == L - 1 else 0.0)
    P[L, :, L] = 1.0   # absorbing

    # No-catastrophe chain.
    for i in range(L + 1, 2 * L - 1):
        P[i, :, i + 1] = 1.0
        R[i, :, i + 1] = sr
    P[2 * L - 1, :, 2 * L - 1] = 1.0   # absorbing
    P[2 * L, :, 2 * L] = 1.0            # safe terminal absorbing

    return P, R


# ---- single_event ----------------------------------------------------------

def _mdp_single_event(
    psi: dict[str, Any],
    *,
    b_safe: float,
    L: int,
    gamma: float,
) -> FiniteMDP:
    """Single-event variant: impulse ``b`` on entry, catastrophe ``-C`` at depth L."""
    P, R = _build_split_chain(
        L=L,
        b_entry=float(psi["b"]),
        b_safe=b_safe,
        p=float(psi["p"]),
        C=float(psi["C"]),
        stream_reward=0.0,
    )
    return build_finite_mdp(P, R, gamma=gamma, horizon=L, initial_state=0)


# ---- warning_state ---------------------------------------------------------

def _mdp_warning_state(
    psi: dict[str, Any],
    *,
    b_safe: float,
    L: int,
    gamma: float,
) -> FiniteMDP:
    """Warning-state variant: labeled warning state on the catastrophe sub-branch.

    Topology identical to single_event; the "warning state" is a label
    attached to the catastrophe-chain state at depth ``L - warning_depth``
    from the contest.  Under classical DP the warning state has strictly
    lower V than its matched-stage counterpart on the no-catastrophe
    sub-branch (bad news).
    """
    warning_depth = int(psi.get("warning_depth", max(1, L // 2)))
    if not (1 <= warning_depth <= L - 1):
        raise ValueError(
            f"warning_depth must lie in [1, L-1]=[1, {L-1}]; "
            f"got warning_depth={warning_depth}"
        )
    P, R = _build_split_chain(
        L=L,
        b_entry=float(psi["b"]),
        b_safe=b_safe,
        p=float(psi["p"]),
        C=float(psi["C"]),
        stream_reward=0.0,
    )
    mdp = build_finite_mdp(P, R, gamma=gamma, horizon=L, initial_state=0)
    mdp.warning_state = int(L - warning_depth + 1)
    mdp.no_warning_state = int(2 * L - warning_depth)
    return mdp


# ---- shallow_early ---------------------------------------------------------

def _mdp_shallow_early(
    psi: dict[str, Any],
    *,
    b_safe: float,
    L: int,
    gamma: float,
) -> FiniteMDP:
    """Shallow-early variant: ``b_shallow`` on first ``k_shallow`` stages, then -C at L."""
    b_shallow = float(psi["b_shallow"])
    k_shallow = int(psi["k_shallow"])
    p = float(psi["p"])
    C = float(psi["C"])
    if not (1 <= k_shallow <= L):
        raise ValueError(f"k_shallow must lie in [1, L]=[1, {L}]; got {k_shallow}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must lie in [0, 1]; got p={p}")

    # Same state layout as _build_split_chain, but reward is stage-dependent
    # (shallow only on first k_shallow stages), so we construct inline.
    # Stage for s=i on cat chain: t=i.  Stage for s=L+j on no-cat chain: t=j+1.
    S = 2 * L + 1
    A = 2
    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A, S), dtype=np.float64)

    # Contest edges.
    P[0, 0, 1] = 1.0
    R[0, 0, 1] = b_shallow if k_shallow >= 1 else 0.0
    P[0, 1, 2 * L] = 1.0
    R[0, 1, 2 * L] = b_safe

    # Split + shallow-conditional branch-A edges.
    if L >= 2:
        r_split = b_shallow if 1 < k_shallow else 0.0
        P[1, :, 2] = p
        P[1, :, L + 1] = 1.0 - p
        R[1, :, 2] = r_split
        R[1, :, L + 1] = r_split

    # Catastrophe chain: stage i; pay b_shallow iff i < k_shallow, additive -C at i=L-1.
    for i in range(2, L):
        P[i, :, i + 1] = 1.0
        r = b_shallow if i < k_shallow else 0.0
        if i == L - 1:
            r = r - C
        R[i, :, i + 1] = r
    P[L, :, L] = 1.0

    # No-catastrophe chain: stage (i - L + 1); pay shallow iff (i-L+1) < k_shallow.
    for i in range(L + 1, 2 * L - 1):
        P[i, :, i + 1] = 1.0
        if (i - L + 1) < k_shallow:
            R[i, :, i + 1] = b_shallow
    P[2 * L - 1, :, 2 * L - 1] = 1.0

    P[2 * L, :, 2 * L] = 1.0
    return build_finite_mdp(P, R, gamma=gamma, horizon=L, initial_state=0)


# ---- multi_event -----------------------------------------------------------

def _mdp_multi_event(
    psi: dict[str, Any],
    *,
    b_safe: float,
    L: int,
    gamma: float,
) -> FiniteMDP:
    """Multi-event variant: catastrophes at depths ``k_i`` with probs ``p_i``, mags ``C_i``.

    Independent Bernoulli events realized at the impulse edge; joint mask
    outcomes flattened into ``2^K`` sub-branches.  Tractability bound
    K <= 6 (tests use K = 2).  ``event_depths[i] in [2, L]`` — depth 1
    would collide with the impulse edge (fold a depth-1 event into ``b``
    via single_event instead).  ``psi`` keys: ``b``, ``event_probs``,
    ``event_depths``, ``event_mags``.
    """
    b = float(psi["b"])
    event_probs = np.asarray(psi["event_probs"], dtype=np.float64)
    event_depths = np.asarray(psi["event_depths"], dtype=np.int64)
    event_mags = np.asarray(psi["event_mags"], dtype=np.float64)
    if event_probs.shape != event_depths.shape or event_probs.shape != event_mags.shape:
        raise ValueError(
            "event_probs / event_depths / event_mags must share shape; "
            f"got {event_probs.shape} / {event_depths.shape} / {event_mags.shape}."
        )
    K = int(event_probs.shape[0])
    if not (1 <= K <= 6):
        raise ValueError(f"multi_event K must be in [1, 6]; got K={K}.")
    if np.any(event_probs < 0.0) or np.any(event_probs > 1.0):
        raise ValueError(f"event_probs must lie in [0, 1]; got {event_probs.tolist()}.")
    if np.any(event_depths < 2) or np.any(event_depths > L):
        raise ValueError(
            f"event_depths must lie in [2, L]=[2, {L}] (depth 1 collides "
            f"with the impulse edge); got {event_depths.tolist()}."
        )

    # 2^K sub-branches; s=0 routes directly to sub-branch depth-1 state so
    # chain-depth k_i sits at stage t=k_i-1 (discount gamma^{k_i-1},
    # matching branch_a_classical_value).  Depth L is absorbing.
    n_branches = 2 ** K
    base = 1
    S = base + n_branches * L + 1
    safe_terminal = S - 1
    P = np.zeros((S, 2, S), dtype=np.float64)
    R = np.zeros((S, 2, S), dtype=np.float64)

    def _branch_start(mask: int) -> int:
        return base + mask * L

    # Contest: action 0 -> each sub-branch with joint-mask probability; b on edge.
    for mask in range(n_branches):
        prob = 1.0
        for i in range(K):
            p_i = float(event_probs[i])
            prob *= p_i if (mask >> i) & 1 else 1.0 - p_i
        if prob > 0.0:
            s_target = _branch_start(mask)
            P[0, 0, s_target] += prob
            R[0, 0, s_target] = b
    P[0, 1, safe_terminal] = 1.0
    R[0, 1, safe_terminal] = b_safe

    # Sub-branch chains: s=start+d-1 is depth d at stage t=d; edge d->d+1
    # pays -C_i for firing event_i with event_depths[i] == d+1.
    for mask in range(n_branches):
        start = _branch_start(mask)
        for d in range(1, L):
            s_from = start + d - 1
            s_to = start + d
            edge_r = 0.0
            for i in range(K):
                if (mask >> i) & 1 and int(event_depths[i]) == d + 1:
                    edge_r += -float(event_mags[i])
            P[s_from, :, s_to] = 1.0
            R[s_from, :, s_to] = edge_r
        P[start + L - 1, :, start + L - 1] = 1.0
    P[safe_terminal, :, safe_terminal] = 1.0

    return build_finite_mdp(P, R, gamma=gamma, horizon=L, initial_state=0)


# ---- matched_concentration -------------------------------------------------

def _mdp_matched_concentration(
    psi: dict[str, Any],
    *,
    b_safe: float,
    L: int,
    gamma: float,
) -> FiniteMDP:
    """Matched-concentration variant: branch A pays a stream AND carries the catastrophe."""
    P, R = _build_split_chain(
        L=L,
        b_entry=float(psi["b_stream"]),
        b_safe=b_safe,
        p=float(psi["p"]),
        C=float(psi["C"]),
        stream_reward=float(psi["b_stream"]),
    )
    return build_finite_mdp(P, R, gamma=gamma, horizon=L, initial_state=0)


# ---------------------------------------------------------------------------
# FamilySpec
# ---------------------------------------------------------------------------

contest_state: ContestState = ContestState(t=0, s=0, action_pair=(0, 1))


def _warm_start(psi: dict[str, Any]) -> float:
    """Variant-aware warm-start: returns the classical V of branch A at x_c."""
    return branch_a_classical_value(psi)


def _scan_bracket(psi: dict[str, Any]) -> tuple[float, float]:
    """Brentq bracket.  ``single_event``/``warning_state`` use (-|C|, 2|b|)
    per the WP2 FamilySpec fragment; other variants use a hint-centered
    symmetric bracket with ``span = max(|hint|, |C|, 1.0)``."""
    variant = str(psi.get("variant", "single_event"))
    hint = _warm_start(psi)
    if variant in ("single_event", "warning_state"):
        C = abs(float(psi.get("C", 1.0)))
        b = abs(float(psi.get("b", 1.0)))
        return (-C, 2.0 * b if b > 0.0 else 1.0)
    if variant == "multi_event":
        C_max = float(np.max(np.abs(np.asarray(psi.get("event_mags", [1.0])))))
        b = abs(float(psi.get("b", 1.0)))
        return (-C_max, 2.0 * b if b > 0.0 else 1.0)
    # shallow_early / matched_concentration: symmetric around hint.
    span = max(abs(hint), abs(float(psi.get("C", 1.0))), 1.0)
    return (hint - span, hint + span)


family_b: FamilySpec = FamilySpec(
    name="catastrophe_vs_safe",
    build_mdp=build_family_b_mdp,
    contest_state=contest_state,
    warm_start_lambda=_warm_start,
    scan_bracket=_scan_bracket,
    metadata={"family": "B", "variants": VARIANT_NAMES,
              "tie_parameter": "b_safe"},
)
