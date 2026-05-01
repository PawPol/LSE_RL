"""Long-horizon delayed-reward chain game (Phase VIII spec §5.7).

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §5.7
(``<!-- patch-2026-05-01 §11 -->``); upstream patch
``tasks/phase_VIII_spec_patches_2026-05-01.md`` §11.

Motivation
----------
The Phase VIII matrix-game suite has horizons typically ``H <= 10`` —
too short to credibly test the paper-title claim "Selective Temporal
Credit Assignment". ``delayed_chain`` is a sparse-reward chain MDP whose
only non-zero reward is at the goal (and, for ``DC-Branching20``, a trap
terminal on the wrong branch). Long horizons (up to ``L = 50``) make
this the cell where TAB's value-propagation behaviour matters most.

Subcases (patch §11.2)
----------------------
- ``DC-Short10``       : ``L=10``, advance-only, 1-action chain.
- ``DC-Medium20``      : ``L=20``, advance-only.
- ``DC-Long50``        : ``L=50``, advance-only (paper headline).
- ``DC-Branching20``   : ``L=20`` with a 5-state trap chain on
  ``branch_wrong``. ``Discrete(2)`` action set: ``0`` = advance,
  ``1`` = branch_wrong.

State encoding
--------------
- Advance-only subcases:     ``state ∈ {0, 1, ..., L}``;
                              state ``L`` is the goal terminal,
                              ``Discrete(L+1)``.
- ``DC-Branching20``:         ``state ∈ {0, ..., L} ∪ {L+1, ..., L+5}``;
                              states ``L+1..L+4`` are trap-chain
                              non-terminals, state ``L+5`` is the trap
                              terminal, ``Discrete(L+1+5) = Discrete(L+6)``.

Reward
------
- ``r_t = 0`` for every non-terminal transition.
- ``r_t = +1`` on arrival at the goal terminal (state ``L``) via the
  advance action.
- ``r_t = -1`` on arrival at the trap terminal (state ``L+5`` for
  ``DC-Branching20``) via ``branch_wrong``.

Canonical sign
--------------
``"+"``. Under optimistic Q-init ``Q_0(s,a) >= V*(s)`` the realized
advantage ``A(s,a) := r - V(s)`` has positive sign in expectation
(reward is non-negative on advance-only chains and value increases
toward the terminal), so positive ``beta`` tightens credit
propagation backward from the terminal +1 (alignment condition
``d_{β,γ} <= γ`` holds). Negative ``beta`` violates the alignment
condition and is predicted to slow convergence.

Theoretical predictions (patch §11.3)
-------------------------------------
- ``P-Sign``     :  ``AUC(+β) > AUC(0) > AUC(-β)`` on every advance-only
                    subcase under optimistic init.
- ``P-Scaling``  :  ``|AUC(+β) - AUC(0)|`` grows monotonically in chain
                    length: ``DC-Short10 < DC-Medium20 < DC-Long50``.
- ``P-Branch``   :  On ``DC-Branching20``, sign of the gap survives the
                    trap-arm exploration penalty.
- ``P-VII-Parity``: ``AUC(0)`` on ``DC-Long50`` lies within the paired-
                    bootstrap 95% CI of the Phase VII-A reference.

Failure of ``P-Sign`` on any advance-only subcase is a paper-critical
event and dispatches a focused Codex bug-hunt review (patch §11.7
trigger ``T11``).

Implementation notes
--------------------
- ``DelayedChainGame`` is a thin subclass of
  :class:`MatrixGameEnv`. The matrix-game wrapper provides the
  MDPInfo / observation space / adversary lifecycle / history
  bookkeeping that the rest of the Phase VIII machinery relies on; we
  only override ``reset`` / ``step`` to substitute chain dynamics for
  the matrix-game payoff lookup. The "opponent" (a one-action
  :class:`PassiveOpponent`) is consumed by the parent class to keep
  ``adversary_info`` flowing into logs but does not affect rewards.
- Per spec §5.7 the chain is regime-stationary: ``info["regime"]`` is
  set to ``None``.
- State extraction follows lessons.md #28: ``int(np.asarray(x).flat[0])``.
- No ``expm1`` / ``log1p`` (lessons.md #11).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mushroom_rl.core import MDPInfo
from mushroom_rl.rl_utils import spaces

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory
from experiments.adaptive_beta.strategic_games.matrix_game import MatrixGameEnv
from experiments.adaptive_beta.strategic_games.registry import register_game


GAME_NAME: str = "delayed_chain"

# Subcase identifiers (patch §11.2).
SUBCASE_SHORT10: str = "DC-Short10"
SUBCASE_MEDIUM20: str = "DC-Medium20"
SUBCASE_LONG50: str = "DC-Long50"
SUBCASE_BRANCHING20: str = "DC-Branching20"

ALL_SUBCASES: Tuple[str, ...] = (
    SUBCASE_SHORT10,
    SUBCASE_MEDIUM20,
    SUBCASE_LONG50,
    SUBCASE_BRANCHING20,
)

# Subcase → chain length L (patch §11.2).
_SUBCASE_TO_L: Dict[str, int] = {
    SUBCASE_SHORT10:     10,
    SUBCASE_MEDIUM20:    20,
    SUBCASE_LONG50:      50,
    SUBCASE_BRANCHING20: 20,
}

# Trap-chain length appended after the goal state for DC-Branching20.
# States ``L+1..L+TRAP_LEN`` index the trap chain; ``L+TRAP_LEN`` is the
# trap terminal that delivers the -1 reward on arrival.
TRAP_CHAIN_LEN: int = 5

# Reward magnitudes (patch §11.2).
GOAL_REWARD: float = 1.0
TRAP_REWARD: float = -1.0


class DelayedChainGame(MatrixGameEnv):
    """Sparse-reward chain MDP wrapped as a Phase VIII matrix-game env.

    Parameters
    ----------
    subcase
        One of :data:`ALL_SUBCASES`.
    adversary
        A 1-action :class:`StrategicAdversary` (see
        :class:`PassiveOpponent`). Its action does not affect transitions
        or rewards; it is kept in the loop so that ``adversary_info``
        flows into logs identically to the matrix-game suite.
    seed
        Optional integer seed; propagated to the env-level RNG and the
        adversary.
    metadata
        Optional extra metadata merged on top of the game-supplied
        block.
    gamma
        MDPInfo discount factor. Default ``0.95``.

    Notes
    -----
    The parent :class:`MatrixGameEnv` builds an :class:`MDPInfo` from
    ``(n_states, n_agent_actions, gamma, horizon)``. We pass:

    - ``n_states = L + 1`` for advance-only subcases,
    - ``n_states = L + 1 + TRAP_CHAIN_LEN`` for ``DC-Branching20``,
    - ``n_agent_actions = 1`` for advance-only, ``2`` for branching,
    - ``horizon = L``.

    A trivial ``(n_agent_actions, 1)`` payoff matrix is supplied to
    satisfy the parent constructor; the actual rewards come from
    :meth:`step`'s chain-dynamics override.
    """

    def __init__(
        self,
        *,
        subcase: str,
        adversary: StrategicAdversary,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        gamma: float = 0.95,
    ) -> None:
        if subcase not in ALL_SUBCASES:
            raise ValueError(
                f"unknown subcase {subcase!r}; valid: {ALL_SUBCASES}"
            )
        if adversary.n_actions != 1:
            raise ValueError(
                f"DelayedChainGame requires a 1-action adversary "
                f"(PassiveOpponent); got n_actions={adversary.n_actions}"
            )

        L = int(_SUBCASE_TO_L[subcase])
        is_branching = subcase == SUBCASE_BRANCHING20
        n_agent_actions = 2 if is_branching else 1
        n_states = (L + 1) + (TRAP_CHAIN_LEN if is_branching else 0)
        action_labels = (
            ["advance", "branch_wrong"] if is_branching else ["advance"]
        )

        # Trivial payoff matrix to satisfy MatrixGameEnv's constructor.
        # Shape: (n_agent_actions, n_opponent_actions=1). The values are
        # ignored — `step` overrides reward computation entirely.
        dummy_payoff = np.zeros((n_agent_actions, 1), dtype=np.float64)

        # Identity state encoder: declares cardinality to the parent
        # constructor's MDPInfo machinery. We do not use the encoder
        # output (we set self._state directly in reset/step), but we do
        # register one with the correct cardinality so the parent's
        # initial encoder call in reset() yields a valid shape.
        def _identity_encoder(_ctx: Dict[str, Any]) -> np.ndarray:
            return np.array(
                [int(np.asarray(self._chain_state).flat[0])],
                dtype=np.int64,
            )

        # Default metadata; merged with user-supplied metadata below.
        meta: Dict[str, Any] = {
            "canonical_sign": "+",
            "subcase": subcase,
            "horizon": L,
            "regime": None,
            "action_labels": action_labels,
            "is_branching": is_branching,
            "trap_chain_len": TRAP_CHAIN_LEN if is_branching else 0,
            "goal_reward": float(GOAL_REWARD),
            "trap_reward": float(TRAP_REWARD) if is_branching else None,
            "is_zero_sum": False,
            "mechanism_degenerate": False,
        }
        if metadata:
            meta.update(metadata)

        # Initialize chain-state cache before super().__init__ — the
        # parent constructor invokes the encoder once.
        self._chain_state: int = 0

        super().__init__(
            payoff_agent=dummy_payoff,
            payoff_opponent=None,  # zero-sum default; never read
            adversary=adversary,
            horizon=L,
            state_encoder=_identity_encoder,
            n_states=n_states,
            seed=seed,
            game_name=GAME_NAME,
            metadata=meta,
            gamma=gamma,
        )

        # Cache subcase config for fast lookup in step / reset.
        self._subcase: str = str(subcase)
        self._L: int = L
        self._is_branching: bool = is_branching
        self._goal_state: int = L
        # Trap-chain absolute indices (only meaningful when branching).
        self._trap_first: int = L + 1
        self._trap_terminal: int = L + TRAP_CHAIN_LEN

        # Canonical-sign tag for downstream schedule selectors.
        self.env_canonical_sign = "+"

        # Override the parent's MDPInfo to ensure observation_space
        # cardinality matches our chain (parent constructor used the
        # n_states we passed, but we re-state it explicitly here as
        # a sanity check; this also documents the contract in code).
        # The parent already built an MDPInfo with these params, so we
        # do not need to rebuild it — the assertion below is a guard
        # against future refactors.
        assert self.info.observation_space.size == (n_states,), (
            f"MDPInfo observation_space.size = "
            f"{self.info.observation_space.size}, "
            f"expected ({n_states},)"
        )
        assert self.info.action_space.size == (n_agent_actions,), (
            f"MDPInfo action_space.size = "
            f"{self.info.action_space.size}, "
            f"expected ({n_agent_actions},)"
        )

    # ------------------------------------------------------------------
    # Chain-dynamics overrides
    # ------------------------------------------------------------------
    def _state_array(self) -> np.ndarray:
        """Return the current chain state as a shape-``(1,)`` int64 array."""
        return np.array([int(self._chain_state)], dtype=np.int64)

    def _is_terminal(self, state: int) -> bool:
        """True iff ``state`` is one of the absorbing chain states."""
        if state == self._goal_state:
            return True
        if self._is_branching and state == self._trap_terminal:
            return True
        return False

    def reset(
        self, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to the start of a new episode.

        Always restarts at chain state ``0``. The ``state`` argument is
        ignored (chain games have a deterministic start) but accepted
        to match the MushroomRL :class:`Environment` contract.
        """
        # Re-seed env RNG and reset adversary (first episode only) by
        # delegating to the parent's reset machinery. We then override
        # the cached state with the chain start.
        del state  # explicit: ignored
        # Re-seed env RNG and adversary via the parent so adversary
        # info / history bookkeeping stay consistent across the suite.
        if self._seed is not None:
            ss = np.random.SeedSequence([self._seed, self._episode_index])
            self._env_rng = np.random.default_rng(ss)
        else:
            self._env_rng = np.random.default_rng()

        if self._episode_index == 0:
            self._adversary.reset(seed=self._seed)
            self._history = GameHistory()

        self._step_in_episode = 0
        self._chain_state = 0
        self._state = self._state_array()

        info: Dict[str, Any] = {
            "phase": self.current_phase(),
            "is_shift_step": False,
            "catastrophe": False,
            "terminal_success": False,
            "opponent_action": None,
            "agent_action": None,
            "adversary_info": self._adversary.info(),
            "game_name": self._game_name,
            "episode_index": self._episode_index,
            "regime": None,            # spec §5.7: regime-stationary
            "chain_state": int(self._chain_state),
        }
        return self._state, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance the chain one step, applying chain dynamics & reward.

        Transition logic
        ----------------
        Let ``s`` be the current chain state (always non-terminal here —
        the runner stops calling ``step`` after ``absorbing=True``).

        Advance-only subcases (action 0 only):
            - if ``s < L``: deterministic ``s -> s+1``;
              reward ``+1`` iff ``s+1 == L`` else ``0``.
            - terminal flag ``True`` iff ``s+1 == L`` OR
              ``self._step_in_episode + 1 >= horizon``.

        ``DC-Branching20`` (actions 0/1):
            - in the main chain (``0 <= s < L``):
                * action 0 (advance): ``s -> s+1``;
                  reward ``+1`` on arrival at ``L``, else ``0``.
                * action 1 (branch_wrong): ``s -> L+1`` (trap-chain
                  entry); reward ``0`` (trap reward fires only at the
                  trap terminal).
            - in the trap chain (``L+1 <= s < L+TRAP_CHAIN_LEN``):
                * action 0 (advance): trap states are absorbing in
                  practice — re-entering them via ``advance`` would
                  contradict the contract (we model the trap chain as a
                  forced one-way path). Concretely, advance moves
                  ``s -> s+1`` along the trap chain; arrival at the
                  trap terminal delivers reward ``-1``.
                * action 1 (branch_wrong): same as advance once inside
                  the trap chain (no further branching). The trap chain
                  is a forced one-way path to the trap terminal.
        """
        # --- 0. Validate / normalise the agent action --------------------
        agent_action: int = int(np.asarray(action).flat[0])
        n_a: int = int(self._n_agent_actions)
        if not (0 <= agent_action < n_a):
            raise ValueError(
                f"agent action must lie in [0, {n_a}), got {agent_action}"
            )

        # --- 1. Adversary act (kept in loop for log parity) --------------
        # Opponent has 1 action; output is unused for chain dynamics.
        opp_raw = self._adversary.act(self._history, agent_action=None)
        opponent_action: int = int(np.asarray(opp_raw).flat[0])
        if not (0 <= opponent_action < self._n_opponent_actions):
            raise ValueError(
                f"adversary returned out-of-range action {opponent_action}"
            )

        # --- 2. Chain transition -----------------------------------------
        s = int(self._chain_state)
        if self._is_branching and s >= self._trap_first:
            # Inside the trap chain — forced advance toward trap terminal.
            next_state = s + 1
            # Reward fires exactly on arrival at the trap terminal.
            reward = (
                float(TRAP_REWARD)
                if next_state == self._trap_terminal
                else 0.0
            )
            terminal_success = False
        elif self._is_branching and agent_action == 1:
            # branch_wrong from the main chain: jump to trap-chain entry.
            next_state = self._trap_first
            reward = 0.0
            terminal_success = False
        else:
            # Advance from the main chain (also covers Discrete(1) advance-only).
            next_state = s + 1
            if next_state == self._goal_state:
                reward = float(GOAL_REWARD)
                terminal_success = True
            else:
                reward = 0.0
                terminal_success = False

        # Update the cached chain state.
        self._chain_state = int(next_state)

        # --- 3. Adversary observe + history bookkeeping ------------------
        self._adversary.observe(
            agent_action=agent_action,
            opponent_action=opponent_action,
            agent_reward=float(reward),
            opponent_reward=0.0,
            info=None,
        )
        adv_info_post = self._adversary.info()
        self._history.append(
            agent_action=agent_action,
            opponent_action=opponent_action,
            agent_reward=float(reward),
            opponent_reward=0.0,
            info=adv_info_post,
        )

        # --- 4. Termination logic ----------------------------------------
        self._step_in_episode += 1
        # Chain absorbs at goal / trap terminals AND on the horizon-cap.
        absorbing: bool = (
            self._is_terminal(self._chain_state)
            or self._step_in_episode >= self._horizon
        )

        # --- 5. State observation ---------------------------------------
        self._state = self._state_array()

        info: Dict[str, Any] = {
            "phase": str(adv_info_post.get("phase") or "unknown"),
            "is_shift_step": False,
            "catastrophe": False,
            "terminal_success": bool(terminal_success),
            "agent_action": agent_action,
            "opponent_action": opponent_action,
            "opponent_reward": 0.0,
            "adversary_info": adv_info_post,
            "game_name": self._game_name,
            "episode_index": self._episode_index,
            "regime": None,                 # spec §5.7: regime-stationary
            "chain_state": int(self._chain_state),
        }

        if absorbing:
            on_ep_end = getattr(self._adversary, "on_episode_end", None)
            if callable(on_ep_end):
                on_ep_end()
            self._episode_index += 1

        return self._state, float(reward), bool(absorbing), info


def build(
    *,
    subcase: str,
    adversary: StrategicAdversary,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> DelayedChainGame:
    """Construct a :class:`DelayedChainGame` (patch §11.4).

    Parameters
    ----------
    subcase
        One of :data:`ALL_SUBCASES`.
    adversary
        Pre-built 1-action :class:`StrategicAdversary` (the canonical
        choice is :class:`PassiveOpponent`).
    seed
        Optional integer seed.
    **kwargs
        Forwarded to :class:`DelayedChainGame`. Supported: ``gamma``,
        ``metadata`` (merged on top of the game-supplied metadata).
    """
    if subcase not in ALL_SUBCASES:
        raise ValueError(
            f"unknown subcase {subcase!r}; valid: {ALL_SUBCASES}"
        )

    metadata = kwargs.pop("metadata", None)
    gamma = float(kwargs.pop("gamma", 0.95))
    if kwargs:
        raise TypeError(
            f"build(delayed_chain) got unexpected kwargs: {sorted(kwargs)}"
        )

    return DelayedChainGame(
        subcase=subcase,
        adversary=adversary,
        seed=seed,
        metadata=metadata,
        gamma=gamma,
    )


# Single registry entry — subcase is a build kwarg (consistent with
# soda_uncertain / potential precedent).
register_game(GAME_NAME, build)
