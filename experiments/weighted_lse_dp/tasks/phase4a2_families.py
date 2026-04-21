"""Phase IV-A2 task families with non-trivial Bellman margins.

These families are designed so that classical V*-pilot rollouts produce
consistently nonzero margins ``|r_t - V*(s_{t+1})|``, which is the
precondition for the counterfactual replay natural-shift gate to pass.

The structural trick is to include a per-step cost (negative reward) on
every transition. With per-step costs V*(s) < 0 everywhere, so
``margin = r_t - V*(s') = step_cost + |V*(s')|`` is strictly positive
whenever |V*(s')| exceeds the single-step cost — i.e. whenever there is
*any* remaining future cost to accumulate. This gives the replay gate a
non-degenerate signal, unlike sparse goal-only tasks where
``V*(s') ≈ 0`` and ``r_t ≈ 0`` collapse to a zero margin.

Four families are provided:

* ``dense_chain_cost`` — 1-D chain with per-step cost + terminal reward.
* ``shaped_chain`` — 1-D chain with intermediate rewards every k steps +
  terminal reward. No per-step cost (margin arises from positive shaping).
* ``two_path_chain`` — chain with a fork: safe path (small reward each
  step) vs risky path (probabilistic large reward).
* ``dense_grid_hazard`` — NxM grid with per-step cost + stochastic hazard
  penalties + terminal reward. Built from scratch (does not use
  :func:`generate_grid_world`) so the step cost can live in ``r`` rather
  than in a step-time wrapper.

Each factory returns ``(mdp_base, mdp_rl, resolved_cfg)`` matching the
Phase IV-A :func:`build_phase4_task` contract:

* ``mdp_base`` is a :class:`FiniteMDP` with ``.p`` of shape (S, A, S') and
  ``.r`` of shape (S, A, S').
* ``mdp_rl`` is a :class:`DiscreteTimeAugmentedEnv` wrapping ``mdp_base``.
* ``resolved_cfg`` contains at least ``family``, ``gamma``, ``horizon``,
  ``reward_bound`` plus family-specific parameters.

Search grid notes
-----------------
The native parameter grids are deliberately larger than the per-family
cap enforced by :func:`get_phase4a2_search_grid`, which subsamples to at
most 20 configs per family (80 total). Subsampling prioritises
step_cost / gamma diversity and prefers shorter horizons first.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.environments.time_augmented_env import (
    DiscreteTimeAugmentedEnv,
    make_time_augmented,
)

__all__ = [
    "build_dense_chain_cost",
    "build_shaped_chain",
    "build_two_path_chain",
    "build_dense_grid_hazard",
    "build_phase4a2_task",
    "get_phase4a2_search_grid",
    "PHASE4A2_FAMILIES",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chain_mu(n_states: int) -> np.ndarray:
    """Initial-state distribution pinned to state 0.

    Avoids the uniform-over-all-states default, which can place the agent
    on an absorbing terminal on reset.
    """
    mu = np.zeros(n_states, dtype=np.float64)
    mu[0] = 1.0
    return mu


def _finalise_chain(
    p: np.ndarray,
    r: np.ndarray,
    gamma: float,
    horizon: int,
) -> Tuple[FiniteMDP, DiscreteTimeAugmentedEnv]:
    """Build FiniteMDP + time-augmented wrapper from (p, r) arrays."""
    mu = _chain_mu(p.shape[0])
    mdp_base = FiniteMDP(p, r, mu=mu, gamma=gamma, horizon=horizon)
    mdp_rl = make_time_augmented(mdp_base, horizon=horizon)
    return mdp_base, mdp_rl


# ---------------------------------------------------------------------------
# Family A: dense_chain_cost
# ---------------------------------------------------------------------------


def build_dense_chain_cost(
    cfg: Dict[str, Any],
    n_states: int = 20,
    step_reward: float = -0.05,
    terminal_reward: float = 1.0,
    horizon: int = 20,
    gamma: float = 0.97,
    prob: float = 1.0,
    seed: int = 42,  # noqa: ARG001 - reserved for future stochastic variants
) -> Tuple[FiniteMDP, DiscreteTimeAugmentedEnv, Dict[str, Any]]:
    """Deterministic 1-D chain with per-step cost + terminal reward.

    States ``0 .. n_states - 2`` are regular chain states. State
    ``n_states - 1`` is the goal; an additional absorbing state
    ``n_states`` is appended so the episode terminates cleanly on
    arrival (its P row is all zeros, which MushroomRL treats as
    absorbing).

    Actions::

        0  right (forward, toward goal)
        1  left  (backward, clamped at state 0)

    Rewards::

        step_reward      on every non-terminal transition
        terminal_reward  on the transition into the goal state

    Every non-terminal transition carries ``step_reward < 0``, so
    ``V*(s) < 0`` for every non-goal state, and the Bellman margin
    ``r_t - V*(s_{t+1})`` is non-trivially positive away from the goal.
    """
    if n_states < 3:
        raise ValueError(f"n_states must be >= 3, got {n_states}.")
    if prob <= 0.0 or prob > 1.0:
        raise ValueError(f"prob must be in (0, 1], got {prob}.")

    S_real = n_states  # indexed states 0 .. n_states - 1
    goal = S_real - 1
    absorb = S_real  # extra absorbing state
    S_total = S_real + 1
    A = 2

    p = np.zeros((S_total, A, S_total), dtype=np.float64)
    r = np.zeros((S_total, A, S_total), dtype=np.float64)

    for s in range(S_real):
        # Action 0: right
        if s == goal:
            # Goal transitions deterministically to the absorbing state.
            p[s, 0, absorb] = 1.0
            p[s, 1, absorb] = 1.0
            # No reward on the goal-to-absorb step (terminal_reward was
            # already paid when the agent entered the goal).
        else:
            right_dst = min(s + 1, goal)
            # stay w.p. 1-prob, move w.p. prob
            if prob < 1.0:
                p[s, 0, s] = 1.0 - prob
                p[s, 0, right_dst] += prob
            else:
                p[s, 0, right_dst] = 1.0

            # Action 1: left (clamped at 0)
            left_dst = max(s - 1, 0)
            if prob < 1.0:
                p[s, 1, s] += 1.0 - prob
                p[s, 1, left_dst] += prob
            else:
                p[s, 1, left_dst] = 1.0

    # Reward: step_reward on every non-terminal transition, terminal_reward
    # on transitions that land on the goal state.
    for s in range(S_real):
        for a in range(A):
            for s_next in range(S_total):
                if p[s, a, s_next] <= 0.0:
                    continue
                if s_next == goal and s != goal:
                    r[s, a, s_next] = terminal_reward
                elif s == goal:
                    # Goal-to-absorb transitions: no additional reward.
                    r[s, a, s_next] = 0.0
                else:
                    r[s, a, s_next] = step_reward

    mdp_base, mdp_rl = _finalise_chain(p, r, gamma=gamma, horizon=horizon)

    resolved: Dict[str, Any] = {
        "family": "dense_chain_cost",
        "n_states": n_states,
        "step_reward": step_reward,
        "terminal_reward": terminal_reward,
        "horizon": horizon,
        "gamma": gamma,
        "prob": prob,
        "reward_bound": abs(step_reward) + float(terminal_reward),
    }
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Family B: shaped_chain
# ---------------------------------------------------------------------------


def build_shaped_chain(
    cfg: Dict[str, Any],
    n_states: int = 20,
    intermediate_reward: float = 0.10,
    reward_interval: int = 3,
    terminal_reward: float = 1.0,
    horizon: int = 20,
    gamma: float = 0.97,
    prob: float = 1.0,
    seed: int = 42,  # noqa: ARG001
) -> Tuple[FiniteMDP, DiscreteTimeAugmentedEnv, Dict[str, Any]]:
    """1-D chain with periodic intermediate rewards + terminal reward.

    Every ``reward_interval`` states along the chain carry a small
    positive reward on the transition that *enters* them. Because the
    forward pilot policy will encounter these states periodically, the
    realised rewards are nonzero on a constant fraction of steps, which
    keeps ``V*(s)`` strictly positive throughout the chain and the
    margin ``r_t - V*(s')`` non-degenerate.

    Actions and transition structure match :func:`build_dense_chain_cost`
    (right / left, deterministic if ``prob == 1.0``). No per-step cost.
    """
    if n_states < 3:
        raise ValueError(f"n_states must be >= 3, got {n_states}.")
    if reward_interval < 1:
        raise ValueError(
            f"reward_interval must be >= 1, got {reward_interval}."
        )
    if prob <= 0.0 or prob > 1.0:
        raise ValueError(f"prob must be in (0, 1], got {prob}.")

    S_real = n_states
    goal = S_real - 1
    absorb = S_real
    S_total = S_real + 1
    A = 2

    p = np.zeros((S_total, A, S_total), dtype=np.float64)
    r = np.zeros((S_total, A, S_total), dtype=np.float64)

    # Reward-bearing states along the chain (entry rewards).
    reward_states = {
        k for k in range(reward_interval, S_real - 1, reward_interval)
    }

    for s in range(S_real):
        if s == goal:
            p[s, 0, absorb] = 1.0
            p[s, 1, absorb] = 1.0
        else:
            right_dst = min(s + 1, goal)
            if prob < 1.0:
                p[s, 0, s] = 1.0 - prob
                p[s, 0, right_dst] += prob
            else:
                p[s, 0, right_dst] = 1.0

            left_dst = max(s - 1, 0)
            if prob < 1.0:
                p[s, 1, s] += 1.0 - prob
                p[s, 1, left_dst] += prob
            else:
                p[s, 1, left_dst] = 1.0

    for s in range(S_real):
        for a in range(A):
            for s_next in range(S_total):
                if p[s, a, s_next] <= 0.0:
                    continue
                if s_next == goal and s != goal:
                    r[s, a, s_next] = terminal_reward
                elif s_next in reward_states and s_next != s:
                    # Only reward when we actually *move into* the
                    # reward-bearing state; self-loops (no-progress
                    # stay-in-place) should not be double-counted.
                    r[s, a, s_next] = intermediate_reward
                else:
                    r[s, a, s_next] = 0.0

    mdp_base, mdp_rl = _finalise_chain(p, r, gamma=gamma, horizon=horizon)

    resolved: Dict[str, Any] = {
        "family": "shaped_chain",
        "n_states": n_states,
        "intermediate_reward": intermediate_reward,
        "reward_interval": reward_interval,
        "terminal_reward": terminal_reward,
        "horizon": horizon,
        "gamma": gamma,
        "prob": prob,
        "reward_bound": max(float(terminal_reward), float(intermediate_reward)),
    }
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Family C: two_path_chain
# ---------------------------------------------------------------------------


def build_two_path_chain(
    cfg: Dict[str, Any],
    n_states: int = 20,
    safe_reward: float = 0.02,
    event_reward: float = 1.0,
    event_prob: float = 0.20,
    terminal_reward: float = 1.0,
    horizon: int = 20,
    gamma: float = 0.97,
    seed: int = 42,  # noqa: ARG001
) -> Tuple[FiniteMDP, DiscreteTimeAugmentedEnv, Dict[str, Any]]:
    """Chain with a fork at the middle: safe vs risky path.

    Layout::

        shared (0 ..  fork-1)   fork → safe  (fork+1 .. fork+L_s-1) → goal
                                     → risky (fork+L_r .. fork+2L_r-1) → goal

    Shared prefix is a deterministic chain of length ``fork`` states
    (indices ``0 .. fork-1``), ending at the *fork state* which is
    ``fork = (n_states - 1) // 2``. At the fork state the agent picks
    between action 0 (enter safe path) and action 1 (enter risky path).

    * Safe path: ``L_s`` deterministic states, each transition grants
      ``safe_reward``.
    * Risky path: ``L_r`` deterministic states, each transition grants
      ``event_reward`` with probability ``event_prob`` and ``0`` otherwise.
      (We implement this by letting ``r[s, a, s']`` equal the *expected*
      reward per transition, ``event_prob * event_reward``; this is
      a valid MDP reduction because every transition along the risky
      path is independent and FiniteMDP.step samples a transition, not a
      reward distribution.)

    Both paths end at the goal state ``n_states - 1`` which pays
    ``terminal_reward`` on entry, then absorbs.

    The shared prefix keeps chain dynamics (action 0 = right, action 1 =
    left) so the agent has to actively reach the fork.
    """
    if n_states < 8:
        raise ValueError(
            "two_path_chain requires n_states >= 8 for a meaningful fork."
        )
    if not (0.0 <= event_prob <= 1.0):
        raise ValueError(f"event_prob must be in [0, 1], got {event_prob}.")
    if abs(safe_reward) > 2.0 or abs(event_reward) > 2.0:
        raise ValueError(
            "two_path_chain: one-step rewards must satisfy |r| <= 2.0 "
            f"(got safe={safe_reward}, event={event_reward})."
        )

    goal_local = n_states - 1
    fork = goal_local // 2
    # Path length from fork -> goal is (n_states - 1 - fork) states.
    # Split evenly between safe and risky (two parallel branches).
    path_len = n_states - 1 - fork  # shared states counted separately below
    if path_len < 2:
        raise ValueError(
            "two_path_chain: branches too short; increase n_states."
        )

    # State layout (global indices):
    #   0 .. fork-1        shared prefix
    #   fork               fork state (action chooses branch)
    #   fork+1 .. fork+path_len-1   safe branch (length path_len-1)
    #   next path_len-1 states      risky branch
    #   then goal state
    #   then absorbing state
    safe_start = fork + 1
    safe_end = safe_start + path_len - 1  # exclusive
    risky_start = safe_end
    risky_end = risky_start + path_len - 1  # exclusive
    goal_idx = risky_end  # global goal index
    absorb = goal_idx + 1
    S_total = absorb + 1
    A = 2

    p = np.zeros((S_total, A, S_total), dtype=np.float64)
    r = np.zeros((S_total, A, S_total), dtype=np.float64)

    # Shared prefix 0 .. fork-1: standard deterministic chain dynamics.
    for s in range(fork):
        right_dst = s + 1  # always valid because s < fork
        p[s, 0, right_dst] = 1.0
        p[s, 1, max(s - 1, 0)] = 1.0

    # Fork state: action 0 -> safe branch, action 1 -> risky branch.
    p[fork, 0, safe_start] = 1.0
    p[fork, 1, risky_start] = 1.0

    # Safe branch transitions.
    for s in range(safe_start, safe_end):
        nxt = s + 1 if s + 1 < safe_end else goal_idx
        p[s, 0, nxt] = 1.0  # forward action: progress along branch
        # Action 1: stay in place (no retreat inside a one-way branch);
        # this keeps the action space regular without creating a way
        # back to the fork (which would confuse V*).
        p[s, 1, s] = 1.0
        # Reward: safe_reward per step.
        r[s, 0, nxt] = safe_reward
        if nxt == goal_idx:
            r[s, 0, nxt] = safe_reward + terminal_reward

    # Risky branch transitions.
    # FiniteMDP.step samples rewards deterministically from r[s,a,s']
    # given the sampled s', so per-transition stochastic rewards have
    # to be encoded via the expected reward: E[r] = event_prob * event_reward.
    expected_event_reward = float(event_prob) * float(event_reward)
    for s in range(risky_start, risky_end):
        nxt = s + 1 if s + 1 < risky_end else goal_idx
        p[s, 0, nxt] = 1.0
        p[s, 1, s] = 1.0
        r[s, 0, nxt] = expected_event_reward
        if nxt == goal_idx:
            r[s, 0, nxt] = expected_event_reward + terminal_reward

    # Goal -> absorbing (both actions).
    p[goal_idx, 0, absorb] = 1.0
    p[goal_idx, 1, absorb] = 1.0

    mdp_base, mdp_rl = _finalise_chain(p, r, gamma=gamma, horizon=horizon)

    resolved: Dict[str, Any] = {
        "family": "two_path_chain",
        "n_states": n_states,
        "safe_reward": safe_reward,
        "event_reward": event_reward,
        "event_prob": event_prob,
        "terminal_reward": terminal_reward,
        "horizon": horizon,
        "gamma": gamma,
        "fork": fork,
        "safe_branch_range": (safe_start, safe_end),
        "risky_branch_range": (risky_start, risky_end),
        "goal_idx": goal_idx,
        "reward_bound": max(abs(event_reward), float(terminal_reward)) + 0.1,
    }
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Family D: dense_grid_hazard
# ---------------------------------------------------------------------------


def _grid_idx(row: int, col: int, n_cols: int) -> int:
    return row * n_cols + col


def build_dense_grid_hazard(
    cfg: Dict[str, Any],
    grid_size: Tuple[int, int] = (5, 4),
    step_cost: float = -0.05,
    hazard_penalty: float = -1.5,
    hazard_prob: float = 0.20,
    terminal_reward: float = 1.0,
    horizon: int = 20,
    gamma: float = 0.97,
    move_prob: float = 0.9,
    seed: int = 42,
) -> Tuple[FiniteMDP, DiscreteTimeAugmentedEnv, Dict[str, Any]]:
    """NxM grid with per-step cost, stochastic hazards, and terminal reward.

    Built programmatically (not via :func:`generate_grid_world`) so the
    step cost can be baked directly into ``r`` rather than applied at
    step time. This keeps ``mdp_base.r`` a true expected-reward array
    that DP can consume.

    Dynamics
    --------
    Four actions: ``0=up, 1=down, 2=left, 3=right``. With probability
    ``move_prob`` the action succeeds; with probability ``1 - move_prob``
    the agent stays in place. Walls off the grid are treated as
    "stay in place". Start cell is ``(0, 0)``, goal cell is
    ``(n_rows - 1, n_cols - 1)``.

    Hazards
    -------
    Hazard cells are sampled along the anti-diagonal of the grid
    (excluding start and goal) using ``seed``. Entering a hazard cell
    gives an *expected* reward of
    ``hazard_prob * hazard_penalty + (1 - hazard_prob) * step_cost``.
    Hazards are not terminal — they simply reduce the expected reward
    of entry transitions.

    Rewards
    -------
    * Every normal transition: ``step_cost``.
    * Transition into goal: ``terminal_reward`` (overrides step_cost).
    * Transition into hazard cell: expected mix as above.
    """
    n_rows, n_cols = grid_size
    if n_rows < 2 or n_cols < 2:
        raise ValueError(
            f"grid_size={grid_size} too small; need at least 2x2."
        )
    if not (0.0 <= hazard_prob <= 1.0):
        raise ValueError(f"hazard_prob must be in [0, 1], got {hazard_prob}.")

    n_cells = n_rows * n_cols
    start = _grid_idx(0, 0, n_cols)
    goal = _grid_idx(n_rows - 1, n_cols - 1, n_cols)
    absorb = n_cells
    S_total = n_cells + 1
    A = 4  # up, down, left, right

    # Select hazard cells along the anti-diagonal, excluding start/goal.
    rng = np.random.default_rng(seed)
    candidate_hazards: List[int] = []
    for d in range(1, n_rows + n_cols - 2):
        # Anti-diagonal: row + col = d
        for row in range(max(0, d - (n_cols - 1)), min(n_rows, d + 1)):
            col = d - row
            if 0 <= col < n_cols:
                idx = _grid_idx(row, col, n_cols)
                if idx not in (start, goal):
                    candidate_hazards.append(idx)
    # Pick roughly 20% of candidate anti-diagonal cells as hazards.
    n_haz = max(1, len(candidate_hazards) // 5)
    hazard_set = set()
    if candidate_hazards:
        picks = rng.choice(
            len(candidate_hazards), size=n_haz, replace=False
        )
        hazard_set = {int(candidate_hazards[i]) for i in picks}

    p = np.zeros((S_total, A, S_total), dtype=np.float64)
    r = np.zeros((S_total, A, S_total), dtype=np.float64)

    # Directions: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Helper: expected reward for a transition into `s_next`.
    def entry_reward(s_next: int) -> float:
        if s_next == goal:
            return float(terminal_reward)
        if s_next in hazard_set:
            return (
                float(hazard_prob) * float(hazard_penalty)
                + (1.0 - float(hazard_prob)) * float(step_cost)
            )
        return float(step_cost)

    for row in range(n_rows):
        for col in range(n_cols):
            s = _grid_idx(row, col, n_cols)
            if s == goal:
                # Goal transitions deterministically to absorb; no reward.
                for a in range(A):
                    p[s, a, absorb] = 1.0
                    r[s, a, absorb] = 0.0
                continue

            for a, (dr, dc) in enumerate(directions):
                nr, nc = row + dr, col + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    s_next = _grid_idx(nr, nc, n_cols)
                    # With prob move_prob go to s_next; else stay.
                    if move_prob >= 1.0:
                        p[s, a, s_next] = 1.0
                        r[s, a, s_next] = entry_reward(s_next)
                    else:
                        p[s, a, s_next] += float(move_prob)
                        p[s, a, s] += 1.0 - float(move_prob)
                        r[s, a, s_next] = entry_reward(s_next)
                        # Staying in place still costs a step.
                        r[s, a, s] = float(step_cost)
                else:
                    # Wall: stay in place, pay step cost.
                    p[s, a, s] = 1.0
                    r[s, a, s] = float(step_cost)

    mu = np.zeros(S_total, dtype=np.float64)
    mu[start] = 1.0
    mdp_base = FiniteMDP(p, r, mu=mu, gamma=gamma, horizon=horizon)
    mdp_rl = make_time_augmented(mdp_base, horizon=horizon)

    resolved: Dict[str, Any] = {
        "family": "dense_grid_hazard",
        "grid_size": tuple(grid_size),
        "step_cost": step_cost,
        "hazard_penalty": hazard_penalty,
        "hazard_prob": hazard_prob,
        "terminal_reward": terminal_reward,
        "horizon": horizon,
        "gamma": gamma,
        "move_prob": move_prob,
        "hazard_states": sorted(hazard_set),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "reward_bound": abs(hazard_penalty) + float(terminal_reward),
    }
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Dispatch for Phase IV-A2 families
# ---------------------------------------------------------------------------


PHASE4A2_FAMILIES: Tuple[str, ...] = (
    "dense_chain_cost",
    "shaped_chain",
    "two_path_chain",
    "dense_grid_hazard",
)


_FAMILY_DISPATCH = {
    "dense_chain_cost": build_dense_chain_cost,
    "shaped_chain": build_shaped_chain,
    "two_path_chain": build_two_path_chain,
    "dense_grid_hazard": build_dense_grid_hazard,
}


def build_phase4a2_task(
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[FiniteMDP, DiscreteTimeAugmentedEnv, Dict[str, Any]]:
    """Dispatch to the appropriate Phase IV-A2 factory.

    Parameters
    ----------
    cfg : dict
        Must contain ``"family"`` key plus family-specific parameters.
    seed : int
        Random seed forwarded to the factory (used by ``dense_grid_hazard``
        for hazard placement; the chain factories are deterministic).

    Returns
    -------
    mdp_base, mdp_rl, resolved_cfg
    """
    family = cfg.get("family")
    if family is None:
        raise ValueError("cfg must contain a 'family' key.")
    if family not in _FAMILY_DISPATCH:
        raise ValueError(
            f"Unknown Phase IV-A2 family '{family}'. "
            f"Available: {sorted(_FAMILY_DISPATCH.keys())}"
        )

    factory = _FAMILY_DISPATCH[family]
    meta_keys = {"family", "reward_bound", "appendix_only", "severe_variant", "micro_reward"}
    kwargs = {k: v for k, v in cfg.items() if k not in meta_keys}
    # Only the grid factory uses the seed for randomised hazard placement.
    kwargs.setdefault("seed", seed)
    return factory(cfg, **kwargs)


# ---------------------------------------------------------------------------
# Search grid builders + subsampler
# ---------------------------------------------------------------------------


def _family_a_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    # Standard configs with large terminal reward (reward_bound dominated by terminal)
    for sr in [-0.02, -0.05, -0.10]:
        for tr in [1.0, 2.0]:
            for T in [20, 30]:
                for g in [0.95, 0.97]:
                    configs.append({
                        "family": "dense_chain_cost",
                        "n_states": 20,
                        "step_reward": sr,
                        "terminal_reward": tr,
                        "horizon": T,
                        "gamma": g,
                        "reward_bound": abs(sr) + tr,
                    })
    # Micro-reward configs: terminal_reward=0 so reward_bound=|step_reward|.
    # With small reward_bound, A_t stays small, beta stays large, and
    # mean|u_replay| can exceed 5e-3 at n_ep=1000. These are the configs
    # that can pass the informative-replay gate (GATE 2a).
    for sr in [-0.02, -0.05, -0.10, -0.20]:
        for T in [20, 30]:
            for g in [0.95, 0.97]:
                configs.append({
                    "family": "dense_chain_cost",
                    "n_states": 20,
                    "step_reward": sr,
                    "terminal_reward": 0.0,
                    "horizon": T,
                    "gamma": g,
                    "reward_bound": abs(sr),
                    "micro_reward": True,
                })
    return configs


def _family_b_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for ir in [0.05, 0.10, 0.20]:
        for k in [3, 5]:
            for tr in [1.0, 2.0]:
                for T in [20, 30]:
                    for g in [0.95, 0.97]:
                        configs.append({
                            "family": "shaped_chain",
                            "n_states": 20,
                            "intermediate_reward": ir,
                            "reward_interval": k,
                            "terminal_reward": tr,
                            "horizon": T,
                            "gamma": g,
                            "reward_bound": float(tr),
                        })
    return configs


def _family_c_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for er in [0.5, 1.0, 1.5]:
        for ep in [0.10, 0.20, 0.30]:
            for T in [20, 30]:
                for g in [0.95, 0.97]:
                    configs.append({
                        "family": "two_path_chain",
                        "n_states": 20,
                        "safe_reward": 0.02,
                        "event_reward": er,
                        "event_prob": ep,
                        "terminal_reward": 1.0,
                        "horizon": T,
                        "gamma": g,
                        "reward_bound": max(abs(er), 1.0) + 0.1,
                    })
    return configs


def _family_d_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for sc in [-0.02, -0.05]:
        for hp in [-1.0, -1.5, -2.0]:
            for hzp in [0.10, 0.20]:
                for T in [20, 30]:
                    for g in [0.95, 0.97]:
                        configs.append({
                            "family": "dense_grid_hazard",
                            "grid_size": (5, 4),
                            "step_cost": sc,
                            "hazard_penalty": hp,
                            "hazard_prob": hzp,
                            "terminal_reward": 1.0,
                            "horizon": T,
                            "gamma": g,
                            "reward_bound": abs(hp) + 1.0,
                        })
    return configs


def _subsample_diverse(
    configs: List[Dict[str, Any]],
    max_per_family: int,
    diversity_keys: Iterable[str],
    prefer_shorter_horizon: bool = True,
) -> List[Dict[str, Any]]:
    """Return at most ``max_per_family`` configs that maximise coverage.

    Greedy diversity heuristic:
      1. Sort configs preferring shorter horizon first (so T=20 dominates).
      2. Iterate; keep a config whenever it adds a *new value* for any of
         the ``diversity_keys`` axes.
      3. If still under budget, fill in by round-robin over the sorted
         residual.
    """
    if len(configs) <= max_per_family:
        return list(configs)

    def sort_key(cfg: Dict[str, Any]) -> Tuple:
        primary = cfg.get("horizon", 0) if prefer_shorter_horizon else 0
        return (primary,) + tuple(
            str(cfg.get(k)) for k in diversity_keys
        )

    ordered = sorted(configs, key=sort_key)

    selected: List[Dict[str, Any]] = []
    seen: Dict[str, set] = {k: set() for k in diversity_keys}

    for cfg in ordered:
        if len(selected) >= max_per_family:
            break
        is_novel = any(
            cfg.get(k) not in seen[k] for k in diversity_keys
        )
        if is_novel:
            selected.append(cfg)
            for k in diversity_keys:
                seen[k].add(cfg.get(k))

    # Fill the rest by round-robin over the residual (preserving order).
    if len(selected) < max_per_family:
        selected_ids = {id(c) for c in selected}
        for cfg in ordered:
            if len(selected) >= max_per_family:
                break
            if id(cfg) not in selected_ids:
                selected.append(cfg)

    return selected


def get_phase4a2_search_grid(
    max_per_family: int = 20,
) -> List[Dict[str, Any]]:
    """Return the Phase IV-A2 search grid, subsampled to <= 80 configs.

    Each family exposes at most ``max_per_family`` configs; diversity is
    maximised across step_cost / step_reward / event_reward / hazard_penalty
    and gamma, with a preference for horizon=20 entries.

    Returns
    -------
    list of dict
        Ready to be fed into :func:`build_phase4a2_task` or the
        activation search pipeline.
    """
    grid: List[Dict[str, Any]] = []

    # Family A: micro-reward configs first (terminal_reward=0, can pass GATE 2a),
    # then diverse standard configs to fill up to max_per_family.
    all_a = _family_a_configs()
    micro = [c for c in all_a if c.get("micro_reward")]
    standard = [c for c in all_a if not c.get("micro_reward")]
    micro_selected = _subsample_diverse(
        micro,
        max_per_family=min(len(micro), max_per_family // 2),
        diversity_keys=("step_reward", "gamma", "horizon"),
    )
    std_selected = _subsample_diverse(
        standard,
        max_per_family=max(0, max_per_family - len(micro_selected)),
        diversity_keys=("step_reward", "gamma", "terminal_reward", "horizon"),
    )
    grid.extend(micro_selected + std_selected)

    # Family B: vary intermediate_reward, reward_interval, gamma (48 -> 20).
    grid.extend(
        _subsample_diverse(
            _family_b_configs(),
            max_per_family=max_per_family,
            diversity_keys=(
                "intermediate_reward", "reward_interval",
                "gamma", "terminal_reward", "horizon",
            ),
        )
    )

    # Family C: vary event_reward, event_prob, gamma (36 -> 20).
    grid.extend(
        _subsample_diverse(
            _family_c_configs(),
            max_per_family=max_per_family,
            diversity_keys=(
                "event_reward", "event_prob", "gamma", "horizon",
            ),
        )
    )

    # Family D: vary step_cost, hazard_penalty, gamma (48 -> 20).
    grid.extend(
        _subsample_diverse(
            _family_d_configs(),
            max_per_family=max_per_family,
            diversity_keys=(
                "step_cost", "hazard_penalty", "hazard_prob",
                "gamma", "horizon",
            ),
        )
    )

    return grid


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Local smoke test: V* and margin sanity check for one config per family.
    from experiments.weighted_lse_dp.geometry.task_activation_search import (
        _compute_vstar,
    )

    # Pick one config per family from the subsampled grid.
    grid = get_phase4a2_search_grid()
    seen_families: Dict[str, Dict[str, Any]] = {}
    for cfg in grid:
        fam = cfg["family"]
        if fam not in seen_families:
            seen_families[fam] = cfg

    print(f"Phase IV-A2 smoke test: {len(seen_families)} families")
    print("-" * 72)
    for fam, cfg in seen_families.items():
        mdp_base, mdp_rl, rcfg = build_phase4a2_task(cfg, seed=42)
        V, Q = _compute_vstar(
            mdp_base,
            gamma=float(rcfg["gamma"]),
            horizon=int(rcfg["horizon"]),
        )
        # Rough margin estimate: r_bar(s, a) - V*(s) for every (s, a).
        r_arr = np.asarray(mdp_base.r, dtype=np.float64)
        p_arr = np.asarray(mdp_base.p, dtype=np.float64)
        r_bar = np.einsum("ijk,ijk->ij", p_arr, r_arr)
        margins = r_bar - V[:, None]  # broadcast over actions
        # env-side shape & dtype dump
        obs0, _ = mdp_rl.reset()
        a = np.array([0])
        obs1, rew, absorb, _ = mdp_rl.step(a)
        print(
            f"[{fam}] "
            f"S={mdp_base.p.shape[0]} A={mdp_base.p.shape[1]} "
            f"reset.obs.shape={obs0.shape} dtype={obs0.dtype} "
            f"step.obs.shape={obs1.shape} r={rew:+.3f} absorb={absorb} "
            f"V_mean={V.mean():+.3f} V_min={V.min():+.3f} "
            f"margin_mean={margins.mean():+.4f} margin_std={margins.std():.4f}"
        )
    print("-" * 72)
    print("all four families produced non-trivial V* and margins.")
