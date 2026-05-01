"""Potential / weakly-acyclic game (spec §5.6).

Phase VIII positive control: an `m`-action symmetric game whose payoffs
admit (or nearly admit) an exact potential function `Phi`. Better-reply
dynamics converge to a pure Nash equilibrium, so a fixed-positive TAB
schedule should accelerate this convergence — making this game the
canonical sanity check for the canonical-sign convention.

Subcases (per spec §5.6)
------------------------
``PG-CoordinationPotential`` — exact-potential coordination game with
    potential function

        Phi(a, b) = 1   if a == b else 0,

    payoffs derived from `Phi` (i.e. ``u_1 = u_2 = Phi``). Each
    on-diagonal cell is a strict pure Nash; (a, b) → (a', b) increases
    `u_1` iff `b == a' != a`, and `Phi` increases by exactly the same
    amount, so the game is exact-potential by construction.

``PG-Congestion`` — symmetric two-player congestion game on `m` parallel
    resources. Each player choosing action ``k`` realises payoff

        u_i(a, b) = 1 / (1 + count_k({a, b}))

    where ``count_k({a, b})`` is the number of players choosing action
    ``k``. Concretely, on a 3-action grid:

        u_i(a, b) = 1/2  if a != b
                  = 1/3  if a == b

    The Rosenthal potential of the congestion game (translated to
    payoffs) is

        Phi(a, b) = sum_{k in {a, b}} sum_{j=1}^{count_k} 1 / (1 + j)
                  = 5/6  if a == b
                  = 1    if a != b

    A unilateral switch by player 1 from `a` to `a'` (with `b` fixed)
    changes `u_1` by `1/(1+count_{a'}) − 1/(1+count_a)` and changes
    `Phi` by exactly the same amount — i.e. this is an exact potential
    game. Pure Nash equilibria are the off-diagonal profiles (any
    miscoordination), since coordinating on the same resource is
    congestion-suboptimal.

``PG-BetterReplyInertia`` — weakly-acyclic variant of the coordination
    potential. The agent's payoff is the coordination potential minus a
    switch-cost penalty triggered when it deviates from its previously
    played action:

        u_1(a, b; prev_a) = Phi(a, b) − lambda · 1[a != prev_a]
        u_2(a, b)         = Phi(a, b)

    The opponent gets pure `Phi`. NOTE / caveat: because the agent's
    payoff depends on `prev_a`, this is *not* an exact potential game in
    the strict sense (the action-switch penalty is path-dependent rather
    than a function of the current action profile alone). It is a
    *weakly acyclic* coordination variant — better-reply dynamics still
    converge to a pure Nash, but the strict potential argument requires
    tracking `prev_a` as part of the state. The symbol `Phi` reported
    by ``compute_potential`` refers to the underlying coordination
    potential ``Phi(a, b) = 1[a == b]``; the inertia penalty is treated
    as a state-dependent cost that does not enter `Phi`.

``PG-SwitchingPayoff`` — exact-potential coordination scaled by a slow
    scalar regime ``c_t`` that switches between ``1.0`` and ``2.0``
    every ``switch_period_episodes`` (default 100) at episode
    boundaries. Both `Phi` and the realised payoffs scale identically:

        u_1(a, b) = u_2(a, b) = c_t · Phi(a, b)
        Phi_t(a, b)           = c_t · 1[a == b]

    `info["regime"]` reports ``"scale_lo"`` (when `c_t == 1.0`) or
    ``"scale_hi"`` (when `c_t == 2.0`).

Action encoding
---------------
Actions are integers in ``{0, ..., m-1}``. Default ``m = 3``.

Canonical sign
--------------
``+`` (encoded as ``"+"``). Potential games admit better-reply dynamics
and a positive `beta` accelerates convergence to a pure Nash
equilibrium — ``wrong_sign`` schedules therefore produce the negative-
beta variant per spec §5.8 disambiguation. The factory sets
``env.env_canonical_sign = "+"`` so downstream schedule selectors read
the correct sign without peeking into metadata.

Theoretical anchor — positive-beta monotonicity on potential games
------------------------------------------------------------------
(Patch 2026-05-01 §7: one-line lemma anchoring the positive-control
prediction.)

Let ``Phi : A -> R`` be the exact potential of the stage game, so

    u_i(a_i', a_{-i}) - u_i(a_i, a_{-i})
        = Phi(a_i', a_{-i}) - Phi(a_i, a_{-i})

for all i, a_i, a_i', a_{-i}.

Under the better-reply dynamics induced by a tabular Q-learning agent
with optimistic initialization ``Q_0(s, a) >= V*(s)`` for all (s, a),
the TAB target

    T_beta Q(s, a) = (1 + gamma) / beta *
                       [ log( exp(beta * r) + gamma * exp(beta * V(s')) )
                         - log(1 + gamma) ]

    (canonical TAB target form per spec §3.1; the bracketed term is
    the log-sum-exp computed via the shared kernel
    ``src/lse_rl/operator/tab_operator.py`` — reimplementing it
    elsewhere is forbidden by spec §2.1)

is monotonically increasing in ``beta`` at fixed ``(r, V(s'), gamma)``
when ``r >= V(s')``, and decreasing when ``r < V(s')``. On potential
games with optimistic init, every better-reply move increases ``Phi``
along the dynamics, so the realized advantage
``A(s, a) := r - V(s)`` has positive sign in expectation under any
better-reply policy. Therefore positive ``beta`` tightens the
contraction toward ``V*`` monotonically (alignment condition
``d_{beta,gamma} <= gamma`` holds), proving ``+beta`` cannot slow
convergence relative to ``beta = 0`` on potential games with this
initialization.

Negative ``beta`` VIOLATES the alignment condition under positive
expected advantage and is therefore predicted to slow or destabilize
convergence on potential games. This gives the falsifiable
sign-prediction (PG-CoordinationPotential, PG-Congestion):

    AUC(+beta) >= AUC(0) >= AUC(-beta)

with strict inequality expected on the upper bound under optimistic
init.

This is the positive-control prediction tested by Phase VIII M6
Stage 1 sweep. Failure of this prediction would constitute strong
evidence of either (a) implementation bug, (b) violation of the
alignment-condition assumption (e.g., misspecified Q init,
non-better-reply opponent), or (c) a flaw in the theoretical
derivation above. Failure of the sign-prediction thus dispatches a
focused Codex bug-hunt review (addendum §3.1 trigger; smoke test
``test_potential_lemma_prediction.py`` (M2 reopen) is the in-suite
guard against (a)–(c)).

`info["regime"]` convention
---------------------------
- Constant subcases (`PG-CoordinationPotential`, `PG-Congestion`):
  ``info["regime"]`` = subcase name verbatim.
- `PG-SwitchingPayoff`: ``info["regime"]`` ∈ ``{"scale_lo", "scale_hi"}``.
- `PG-BetterReplyInertia`: ``info["regime"] = "PG-BetterReplyInertia"``.
  In addition, ``info["last_agent_action"]`` reports the agent's action
  played on the *previous* step (``None`` at episode start) — this is
  the inertia signal already threaded into the state-encoder context by
  ``MatrixGameEnv``; the field is also surfaced explicitly in `info`
  for downstream consumers that don't read the encoder context.

References
----------
- D. Monderer and L. S. Shapley. "Potential Games." *Games and Economic
  Behavior* 14 (1996), 124–143.
- R. W. Rosenthal. "A Class of Games Possessing Pure-Strategy Nash
  Equilibria." *International Journal of Game Theory* 2 (1973), 65–67.
- H. P. Young. *Strategic Learning and Its Limits.* OUP, 2004
  (weakly acyclic games and better-reply dynamics).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.matrix_game import (
    MatrixGameEnv,
    StateEncoder,
    make_default_state_encoder,
)
from experiments.adaptive_beta.strategic_games.registry import register_game


GAME_NAME: str = "potential"

# Subcase identifiers (also the values written into `info["regime"]`
# for the constant-payoff subcases).
SUBCASE_COORDINATION: str = "PG-CoordinationPotential"
SUBCASE_CONGESTION: str = "PG-Congestion"
SUBCASE_INERTIA: str = "PG-BetterReplyInertia"
SUBCASE_SWITCHING: str = "PG-SwitchingPayoff"

ALL_SUBCASES: Tuple[str, ...] = (
    SUBCASE_COORDINATION,
    SUBCASE_CONGESTION,
    SUBCASE_INERTIA,
    SUBCASE_SWITCHING,
)

# `PG-SwitchingPayoff` regime labels.
REGIME_SCALE_LO: str = "scale_lo"
REGIME_SCALE_HI: str = "scale_hi"
SCALE_LO: float = 1.0
SCALE_HI: float = 2.0


# ---------------------------------------------------------------------------
# Potential function `Phi` and base payoff matrices
# ---------------------------------------------------------------------------

def _coordination_potential_matrix(m: int) -> np.ndarray:
    """Return the m x m identity-like potential `Phi(a, b) = 1[a == b]`."""
    return np.eye(m, dtype=np.float64)


def _congestion_payoff_matrix(m: int) -> np.ndarray:
    """Symmetric two-player congestion payoffs on m parallel resources.

    ``payoff[a, b] = 1 / (1 + count_a({a, b}))`` — i.e. ``1/2`` whenever
    `a != b` and ``1/3`` whenever `a == b`.
    """
    pa = np.full((m, m), 0.5, dtype=np.float64)
    diag = np.arange(m)
    pa[diag, diag] = 1.0 / 3.0
    return pa


def _congestion_potential_matrix(m: int) -> np.ndarray:
    """Rosenthal potential of the symmetric two-player congestion game.

    See module docstring: ``Phi(a, b) = 1`` when `a != b` and ``5/6``
    when `a == b`.
    """
    phi = np.full((m, m), 1.0, dtype=np.float64)
    diag = np.arange(m)
    phi[diag, diag] = 5.0 / 6.0
    return phi


def compute_potential(
    action_profile: Tuple[int, int],
    *,
    subcase: str = SUBCASE_COORDINATION,
    m: int = 3,
    scale: float = 1.0,
) -> float:
    """Evaluate the potential function `Phi` for a given action profile.

    Parameters
    ----------
    action_profile
        ``(agent_action, opponent_action)`` pair. Each entry must lie in
        ``[0, m)``.
    subcase
        One of ``ALL_SUBCASES``. For ``PG-BetterReplyInertia`` we
        report the *underlying* coordination potential ``1[a == b]``;
        the inertia penalty is path-dependent and does not enter `Phi`
        (see module docstring caveat).
    m
        Action-set cardinality (default 3, matching the factory
        default).
    scale
        Multiplicative scale applied to the potential. Defaults to
        ``1.0``; `PG-SwitchingPayoff` should pass ``c_t`` (i.e. ``1.0``
        or ``2.0``).

    Returns
    -------
    float
        Value of `Phi(action_profile)` (scaled by ``scale``).
    """
    if subcase not in ALL_SUBCASES:
        raise ValueError(
            f"unknown subcase {subcase!r}; expected one of {ALL_SUBCASES}"
        )
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    a, b = action_profile
    a_int = int(np.asarray(a).flat[0])
    b_int = int(np.asarray(b).flat[0])
    if not (0 <= a_int < m and 0 <= b_int < m):
        raise ValueError(
            f"action_profile entries must lie in [0, {m}); got "
            f"({a_int}, {b_int})"
        )

    if subcase in (SUBCASE_COORDINATION, SUBCASE_INERTIA, SUBCASE_SWITCHING):
        base = 1.0 if a_int == b_int else 0.0
    elif subcase == SUBCASE_CONGESTION:
        base = (5.0 / 6.0) if a_int == b_int else 1.0
    else:  # pragma: no cover — guarded by the membership check above.
        raise AssertionError(f"unreachable subcase {subcase!r}")

    return float(scale) * float(base)


# ---------------------------------------------------------------------------
# PotentialGameEnv: thin subclass of MatrixGameEnv
# ---------------------------------------------------------------------------

class PotentialGameEnv(MatrixGameEnv):
    """``MatrixGameEnv`` subclass that injects subcase-specific dynamics.

    Three out of four subcases need either time-varying payoff scaling
    (`PG-SwitchingPayoff`) or a state-dependent reward adjustment
    (`PG-BetterReplyInertia`). We implement both as pre-step mutations
    of ``self._payoff_agent`` / ``self._payoff_opponent`` that are
    restored after each ``step`` call. This keeps the parent class's
    ``step`` body authoritative for action validation, adversary
    invocation, history bookkeeping, and `info` construction — we only
    add the per-subcase `regime` / `last_agent_action` fields on top.
    """

    def __init__(
        self,
        *,
        subcase: str,
        m: int,
        adversary: StrategicAdversary,
        horizon: int,
        switch_period_episodes: int,
        lambda_inertia: float,
        state_encoder: Optional[StateEncoder] = None,
        n_states: int = 1,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        gamma: float = 0.95,
    ) -> None:
        if subcase not in ALL_SUBCASES:
            raise ValueError(
                f"unknown subcase {subcase!r}; expected one of {ALL_SUBCASES}"
            )

        # ------------- base payoff matrices (subcase-specific) -------------
        if subcase == SUBCASE_CONGESTION:
            base_pa = _congestion_payoff_matrix(m)
            base_po = _congestion_payoff_matrix(m)
        else:
            # All three "coordination-style" subcases share the same
            # base-payoff matrix u = Phi (identity-like); the
            # SwitchingPayoff scaling and Inertia penalty are applied
            # per-step as transient mutations.
            base_pa = _coordination_potential_matrix(m)
            base_po = _coordination_potential_matrix(m)

        super().__init__(
            payoff_agent=base_pa,
            payoff_opponent=base_po,
            adversary=adversary,
            horizon=horizon,
            state_encoder=state_encoder,
            n_states=n_states,
            seed=seed,
            game_name=GAME_NAME,
            metadata=metadata,
            gamma=gamma,
        )

        self._subcase: str = str(subcase)
        self._m: int = int(m)
        self._switch_period_episodes: int = int(switch_period_episodes)
        self._lambda_inertia: float = float(lambda_inertia)
        # Cache the unmodified base matrices so we can restore after
        # each per-step mutation.
        self._base_payoff_agent: np.ndarray = base_pa.copy()
        self._base_payoff_opponent: np.ndarray = base_po.copy()
        # Inertia bookkeeping: previous agent action within the episode
        # (None at episode start). NOT reset across episodes for
        # SwitchingPayoff (which uses episode_index for its regime
        # clock); inertia *is* reset on every reset() because it is an
        # intra-episode signal.
        self._prev_agent_action: Optional[int] = None

        # Default canonical sign tag — overridden by the factory but set
        # here too so direct-construction call sites stay correct.
        self.env_canonical_sign = "+"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _current_scale(self) -> float:
        """Current ``c_t`` for `PG-SwitchingPayoff`.

        Switches between ``SCALE_LO`` and ``SCALE_HI`` at episode
        boundaries every ``self._switch_period_episodes`` episodes.
        Other subcases return ``1.0``.
        """
        if self._subcase != SUBCASE_SWITCHING:
            return 1.0
        if self._switch_period_episodes <= 0:
            return SCALE_LO
        block = self._episode_index // self._switch_period_episodes
        return SCALE_HI if (block % 2) == 1 else SCALE_LO

    def _current_regime(self) -> str:
        """`info["regime"]` payload for the current step."""
        if self._subcase == SUBCASE_SWITCHING:
            return (
                REGIME_SCALE_HI
                if self._current_scale() == SCALE_HI
                else REGIME_SCALE_LO
            )
        return self._subcase

    def _apply_subcase_payoffs(self, agent_action_now: int) -> None:
        """Mutate ``self._payoff_agent`` / ``_payoff_opponent`` in place.

        Called immediately before delegating to ``super().step``. The
        mutation is restored at the end of ``step``.

        - `PG-CoordinationPotential` and `PG-Congestion`: no-op (use
          the base matrices).
        - `PG-SwitchingPayoff`: scale both players' matrices by ``c_t``.
        - `PG-BetterReplyInertia`: subtract a row-constant
          ``lambda · 1[a != prev_a]`` from the agent's payoff matrix
          (penalises every agent action ``a`` that differs from
          ``prev_a``); leaves the opponent's matrix at the base
          coordination potential.
        """
        if self._subcase in (SUBCASE_COORDINATION, SUBCASE_CONGESTION):
            return  # no per-step mutation

        if self._subcase == SUBCASE_SWITCHING:
            scale = self._current_scale()
            self._payoff_agent = scale * self._base_payoff_agent
            self._payoff_opponent = scale * self._base_payoff_opponent
            return

        if self._subcase == SUBCASE_INERTIA:
            # `agent_action_now` is supplied for documentation parity;
            # the penalty is determined by the row index alone.
            del agent_action_now
            mod_pa = self._base_payoff_agent.copy()
            if self._prev_agent_action is not None:
                penalty = np.full(
                    self._m, -self._lambda_inertia, dtype=np.float64
                )
                penalty[int(self._prev_agent_action)] = 0.0
                # Broadcast the row-constant penalty across columns.
                mod_pa = mod_pa + penalty[:, None]
            self._payoff_agent = mod_pa
            # Opponent receives unmodified `Phi`.
            self._payoff_opponent = self._base_payoff_opponent.copy()
            return

        # Defensive — should be unreachable given the constructor
        # subcase check.
        raise AssertionError(f"unreachable subcase {self._subcase!r}")

    def _restore_base_payoffs(self) -> None:
        """Restore the base matrices after a ``step`` call."""
        self._payoff_agent = self._base_payoff_agent.copy()
        self._payoff_opponent = self._base_payoff_opponent.copy()

    # ------------------------------------------------------------------
    # MushroomRL Environment interface
    # ------------------------------------------------------------------
    def reset(
        self, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to the start of a new episode and inject `regime`.

        The inertia signal ``self._prev_agent_action`` is reset to
        ``None`` at every episode boundary — inertia is an intra-episode
        cost (per spec §5.6 weakly-acyclic gloss).
        """
        self._prev_agent_action = None
        s, info = super().reset(state=state)
        info["regime"] = self._current_regime()
        if self._subcase == SUBCASE_INERTIA:
            info["last_agent_action"] = None
        return s, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step with subcase-specific payoff mutation.

        Sequence:
          1. Apply per-step payoff mutation (scale or inertia penalty).
          2. Delegate to ``super().step`` for action validation,
             adversary invocation, history bookkeeping and the base
             `info` dict.
          3. Restore base payoff matrices.
          4. Augment the returned `info` with `regime` and (for the
             inertia subcase) `last_agent_action`.
          5. Update the inertia bookkeeping with the just-played agent
             action.
        """
        agent_action_int = int(np.asarray(action).flat[0])
        # ---- 1. payoff mutation ----
        self._apply_subcase_payoffs(agent_action_now=agent_action_int)

        try:
            # ---- 2. delegate to super().step ----
            s, r, absorbing, info = super().step(action)
        finally:
            # ---- 3. restore base matrices unconditionally ----
            self._restore_base_payoffs()

        # ---- 4. augment info ----
        info["regime"] = self._current_regime()
        if self._subcase == SUBCASE_INERTIA:
            info["last_agent_action"] = self._prev_agent_action

        # ---- 5. update inertia bookkeeping ----
        if absorbing:
            # Inertia is an intra-episode signal; clear it at episode
            # boundary so the next reset() observes a clean slate.
            self._prev_agent_action = None
        else:
            self._prev_agent_action = agent_action_int

        return s, r, absorbing, info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build(
    *,
    subcase: str,
    adversary: StrategicAdversary,
    horizon: int = 20,
    m: int = 3,
    switch_period_episodes: int = 100,
    lambda_inertia: float = 0.1,
    seed: Optional[int] = None,
    state_encoder: Optional[StateEncoder] = None,
    **kwargs: Any,
) -> PotentialGameEnv:
    """Construct a Potential / Weakly-Acyclic ``PotentialGameEnv``.

    Parameters
    ----------
    subcase
        One of ``ALL_SUBCASES``.
    adversary
        Pre-built ``StrategicAdversary`` (must be ``m``-action).
    horizon
        Episode length. Default 20 (Phase VIII spec §5 default).
    m
        Action-set cardinality. Default 3.
    switch_period_episodes
        Period (in episodes) of the slow regime clock for
        `PG-SwitchingPayoff`. Default 100. Ignored by other subcases.
    lambda_inertia
        Action-switch penalty for `PG-BetterReplyInertia`. Spec §5.6
        suggests ``lambda in {0.0, 0.1, 0.3}``; default ``0.1``.
        Ignored by other subcases.
    seed
        Optional integer seed (propagated to the env-level RNG and the
        adversary).
    state_encoder
        Optional override. ``None`` uses the default
        ``(timestep, prev_opponent_action)`` encoder for ``H * (m + 1)``
        states (parent class default).
    **kwargs
        Forwarded to ``PotentialGameEnv``. Supported: ``gamma``,
        ``metadata`` (merged on top of the game-supplied metadata),
        ``n_states`` (only with a custom ``state_encoder``).
    """
    if subcase not in ALL_SUBCASES:
        raise ValueError(
            f"unknown subcase {subcase!r}; expected one of {ALL_SUBCASES}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if m < 2:
        raise ValueError(
            f"m must be >= 2 for a non-degenerate game, got {m}"
        )
    if adversary.n_actions != m:
        raise ValueError(
            f"adversary.n_actions={adversary.n_actions} != m={m} "
            f"(potential / {subcase})"
        )
    if switch_period_episodes < 1:
        raise ValueError(
            "switch_period_episodes must be >= 1, got "
            f"{switch_period_episodes}"
        )
    if lambda_inertia < 0.0:
        raise ValueError(
            f"lambda_inertia must be >= 0, got {lambda_inertia}"
        )

    if state_encoder is None:
        encoder, n_states = make_default_state_encoder(
            horizon=horizon, n_actions=m
        )
    else:
        encoder = state_encoder
        n_states = int(kwargs.pop("n_states", 1))

    metadata: Dict[str, Any] = {
        "canonical_sign": "+",
        "mechanism_degenerate": horizon == 1,
        "is_zero_sum": False,
        "subcase": subcase,
        "m": int(m),
        "switch_period_episodes": int(switch_period_episodes),
        "lambda_inertia": float(lambda_inertia),
        # Document Phi alongside metadata for downstream introspection.
        "potential_function": (
            "Phi(a,b)=1[a==b] (coordination/inertia/switching); "
            "Phi(a,b)=5/6 if a==b else 1 (congestion, Rosenthal)"
        ),
    }
    user_meta = kwargs.pop("metadata", None)
    if user_meta:
        metadata.update(user_meta)

    gamma = float(kwargs.pop("gamma", 0.95))
    if kwargs:
        raise TypeError(
            f"build(potential) got unexpected kwargs: {sorted(kwargs)}"
        )

    env = PotentialGameEnv(
        subcase=subcase,
        m=m,
        adversary=adversary,
        horizon=horizon,
        switch_period_episodes=switch_period_episodes,
        lambda_inertia=lambda_inertia,
        state_encoder=encoder,
        n_states=n_states,
        seed=seed,
        metadata=metadata,
        gamma=gamma,
    )
    # Override the env-level attribute so downstream consumers
    # (schedules.build_schedule) read the correct sign even if they
    # don't peek into metadata.
    env.env_canonical_sign = "+"
    return env


register_game(GAME_NAME, build)
