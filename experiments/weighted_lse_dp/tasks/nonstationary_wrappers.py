"""Nonstationary environment wrappers for Phase II regime-shift stress tasks.

Wrappers trigger a configurable structural change (goal position, reward
sign, slip probability) after a fixed number of episodes or training steps.
The state and action spaces are never modified by the wrapper.

Severity=0 contract:
  - ``ChainRegimeShiftWrapper`` with ``change_at_episode >= total_episodes``
    never shifts and recovers the base chain exactly.
  - ``GridRegimeShiftWrapper`` with ``change_at_episode >= total_episodes``
    never shifts and recovers the base grid exactly.

Implemented here:
  - ``ChainRegimeShiftWrapper``  (spec S5.1.D)
  - ``make_chain_regime_shift``  factory
  - ``GridRegimeShiftWrapper``   (spec S5.2.C)
  - ``make_grid_regime_shift``   factory
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.environments.generators.simple_chain import (
    compute_probabilities as chain_compute_probabilities,
    compute_reward as chain_compute_reward,
)
from mushroom_rl.environments.generators.grid_world import (
    generate_grid_world,
    parse_grid,
    compute_probabilities as grid_compute_probabilities,
    compute_reward as grid_compute_reward,
    compute_mu as grid_compute_mu,
)
from mushroom_rl.environments.time_augmented_env import (
    DiscreteTimeAugmentedEnv,
    make_time_augmented,
)

__all__: list[str] = [
    "ChainRegimeShiftWrapper",
    "make_chain_regime_shift",
    "GridRegimeShiftWrapper",
    "make_grid_regime_shift",
]


# ---------------------------------------------------------------------------
# Base wrapper class (shared logic)
# ---------------------------------------------------------------------------


class _RegimeShiftWrapperBase:
    """Common regime-shift logic shared by chain and grid wrappers.

    Tracks episode count and switches from the pre-shift MDP to the
    post-shift MDP exactly at the configured change point. State and
    action spaces are never modified.

    The stage index is derivable from the augmented state alone (no hidden
    episode counters are used for state representation -- the episode
    counter only drives the regime switch).
    """

    def __init__(
        self,
        pre_shift_mdp: FiniteMDP,
        post_shift_mdp: FiniteMDP,
        change_at_episode: int,
    ) -> None:
        self._pre = pre_shift_mdp
        self._post = post_shift_mdp
        self._change_at = change_at_episode
        self._episode_count: int = 0
        self._post_change: bool = False
        # MDPInfo from pre-shift (spaces never change).
        self.info = pre_shift_mdp.info
        self._current = pre_shift_mdp

    def reset(self, state: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Reset episode; switch to post-shift MDP at the configured episode."""
        if self._episode_count >= self._change_at:
            self._current = self._post
            self._post_change = True
        else:
            self._current = self._pre
        self._episode_count += 1
        result = self._current.reset(state)
        # MushroomRL 2.x: reset returns (state, info).
        if isinstance(result, tuple):
            return result
        return result, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Step using whichever MDP is currently active."""
        return self._current.step(action)

    @property
    def post_change(self) -> bool:
        """True if the regime has shifted."""
        return self._post_change

    @property
    def change_episode(self) -> int:
        """The episode index at which the shift occurs."""
        return self._change_at


# ---------------------------------------------------------------------------
# Chain regime shift (spec S5.1.D)
# ---------------------------------------------------------------------------


class ChainRegimeShiftWrapper(_RegimeShiftWrapperBase):
    """Nonstationary chain: reward/dynamics change after change_at_episode episodes.

    Implements the regime-shift stress task for Phase II. The wrapper tracks
    episode count and switches from the pre-shift MDP to the post-shift MDP
    exactly at the configured change point.

    State/action spaces are NEVER modified by the wrapper.

    Severity=0 (change_at_episode very large): recovers base chain exactly.
    """

    pass


def _build_chain_mdp(
    state_n: int,
    goal_states: list[int],
    prob: float,
    rew: float,
    gamma: float,
    horizon: int,
) -> FiniteMDP:
    """Build a FiniteMDP for a simple chain with given goal states."""
    p = chain_compute_probabilities(state_n, prob)
    r = chain_compute_reward(state_n, goal_states, rew)
    mu = np.zeros(state_n)
    mu[0] = 1.0  # start at state 0 (leftmost)
    return FiniteMDP(p, r, mu, gamma, horizon)


def make_chain_regime_shift(
    cfg: dict[str, Any],
    state_n: int = 25,
    prob: float = 0.9,
    gamma: float = 0.99,
    horizon: int = 60,
    change_at_episode: int = 500,
    shift_type: str = "goal_flip",
    post_prob: float = 0.7,
    time_augment: bool = True,
) -> tuple[ChainRegimeShiftWrapper, FiniteMDP | DiscreteTimeAugmentedEnv, dict]:
    """Factory: returns (wrapper, mdp_rl_pre, resolved_cfg).

    The wrapper is the ``ChainRegimeShiftWrapper`` used as the environment
    for DP/RL planners. ``mdp_rl_pre`` is the time-augmented pre-shift MDP
    for RL algorithms.

    Args:
        cfg: user config dict (overrides are merged on top of defaults).
        state_n: number of states in the chain.
        prob: transition success probability (pre-shift).
        gamma: discount factor.
        horizon: episode horizon.
        change_at_episode: episode index at which the shift occurs (0-based).
        shift_type: one of ``"goal_flip"``, ``"prob_change"``,
            ``"reward_flip"``.
        post_prob: transition probability for post-shift (used when
            ``shift_type="prob_change"``).
        time_augment: whether to time-augment the RL env.

    Returns:
        ``(wrapper, mdp_rl, resolved_cfg)`` triple.
    """
    # Merge user overrides.
    resolved = {
        "task": f"chain_regime_shift_{shift_type}",
        "state_n": state_n,
        "prob": prob,
        "gamma": gamma,
        "horizon": horizon,
        "change_at_episode": change_at_episode,
        "shift_type": shift_type,
        "post_prob": post_prob,
    }
    resolved.update(cfg)

    sn = int(resolved["state_n"])
    p = float(resolved["prob"])
    g = float(resolved["gamma"])
    h = int(resolved["horizon"])
    st = str(resolved["shift_type"])
    pp = float(resolved["post_prob"])
    cae = int(resolved["change_at_episode"])

    # Pre-shift: standard chain with goal at rightmost state.
    goal_pre = sn - 1
    mdp_pre = _build_chain_mdp(sn, [goal_pre], p, 1.0, g, h)

    # Post-shift: depends on shift_type.
    if st == "goal_flip":
        # Goal moves from rightmost to leftmost.
        goal_post = 0
        mdp_post = _build_chain_mdp(sn, [goal_post], p, 1.0, g, h)
    elif st == "prob_change":
        # Same goal, different transition probability.
        mdp_post = _build_chain_mdp(sn, [goal_pre], pp, 1.0, g, h)
    elif st == "reward_flip":
        # Reward now at the opposite end (leftmost instead of rightmost).
        goal_post = 0
        mdp_post = _build_chain_mdp(sn, [goal_post], p, 1.0, g, h)
    else:
        raise ValueError(
            f"Unknown chain regime shift_type: {st!r}. "
            f"Expected one of 'goal_flip', 'prob_change', 'reward_flip'."
        )

    wrapper = ChainRegimeShiftWrapper(mdp_pre, mdp_post, cae)

    if time_augment:
        mdp_rl = make_time_augmented(mdp_pre, horizon=h)
    else:
        mdp_rl = mdp_pre

    return wrapper, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Grid regime shift (spec S5.2.C)
# ---------------------------------------------------------------------------


class GridRegimeShiftWrapper(_RegimeShiftWrapperBase):
    """Nonstationary grid: reward/dynamics change after change_at_episode episodes.

    Implements the regime-shift stress task for Phase II. The wrapper tracks
    episode count and switches from the pre-shift MDP to the post-shift MDP
    exactly at the configured change point.

    State/action spaces are NEVER modified by the wrapper.

    Severity=0 (change_at_episode very large): recovers base grid exactly.
    """

    pass


def _build_grid_mdp_from_file(
    grid_file: str | Path,
    prob: float,
    pos_rew: float,
    neg_rew: float,
    gamma: float,
    horizon: int,
) -> FiniteMDP:
    """Build a grid FiniteMDP from a grid file."""
    return generate_grid_world(
        grid=str(grid_file),
        prob=prob,
        pos_rew=pos_rew,
        neg_rew=neg_rew,
        gamma=gamma,
        horizon=horizon,
    )


def _build_grid_mdp_with_moved_goal(
    original_grid_file: str | Path,
    new_goal_cell: tuple[int, int],
    prob: float,
    pos_rew: float,
    neg_rew: float,
    gamma: float,
    horizon: int,
) -> FiniteMDP:
    """Build a grid FiniteMDP with a moved goal by rewriting the grid file.

    Reads the original grid, moves 'G' to ``new_goal_cell`` (replacing
    whatever was there, turning the old 'G' into '.'), then constructs
    the FiniteMDP from the modified grid.
    """
    with open(str(original_grid_file), "r") as f:
        content = f.read()

    # Parse into a mutable grid.
    rows = content.strip().split("\n")
    grid = [list(row) for row in rows]

    # Clear old goal(s).
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == "G":
                grid[r][c] = "."

    # Place new goal.
    gr, gc = new_goal_cell
    grid[gr][gc] = "G"

    # Write to a temporary file and build the MDP.
    modified_content = "\n".join("".join(row) for row in grid) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as tmp:
        tmp.write(modified_content)
        tmp_path = tmp.name

    try:
        mdp = generate_grid_world(
            grid=tmp_path,
            prob=prob,
            pos_rew=pos_rew,
            neg_rew=neg_rew,
            gamma=gamma,
            horizon=horizon,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return mdp


def make_grid_regime_shift(
    cfg: dict[str, Any],
    change_at_episode: int = 300,
    shift_type: str = "goal_move",
    post_prob: float = 0.7,
    time_augment: bool = True,
) -> tuple[GridRegimeShiftWrapper, FiniteMDP | DiscreteTimeAugmentedEnv, dict]:
    """Factory: returns (wrapper, mdp_rl_pre, resolved_cfg).

    The wrapper is the ``GridRegimeShiftWrapper`` used as the environment
    for DP/RL planners.

    Args:
        cfg: user config dict (overrides are merged on top of defaults).
        change_at_episode: episode index at which the shift occurs.
        shift_type: one of ``"goal_move"``, ``"slip_change"``.
        post_prob: slip probability for post-shift (used when
            ``shift_type="slip_change"``).
        time_augment: whether to time-augment the RL env.

    Returns:
        ``(wrapper, mdp_rl, resolved_cfg)`` triple.
    """
    from experiments.weighted_lse_dp.common.task_factories import GRID_BASE_CONFIG

    # Merge defaults with user overrides.
    resolved = {
        "task": f"grid_regime_shift_{shift_type}",
        "change_at_episode": change_at_episode,
        "shift_type": shift_type,
        "post_prob": post_prob,
    }
    # Pull grid config defaults.
    for k in ("grid_file", "n_rows", "n_cols", "prob", "pos_rew",
              "neg_rew", "gamma", "horizon", "goal_cell"):
        resolved.setdefault(k, GRID_BASE_CONFIG[k])
    resolved.update(cfg)

    grid_file = str(resolved["grid_file"])
    prob = float(resolved["prob"])
    pos_rew = float(resolved["pos_rew"])
    neg_rew = float(resolved["neg_rew"])
    g = float(resolved["gamma"])
    h = int(resolved["horizon"])
    st = str(resolved["shift_type"])
    pp = float(resolved["post_prob"])
    cae = int(resolved["change_at_episode"])

    # Pre-shift: standard grid_base.
    mdp_pre = _build_grid_mdp_from_file(grid_file, prob, pos_rew, neg_rew, g, h)

    # Post-shift: depends on shift_type.
    if st == "goal_move":
        # Move goal from (4,4) to (4,0) -- bottom-left corner.
        new_goal = (4, 0)
        resolved["post_goal_cell"] = new_goal
        mdp_post = _build_grid_mdp_with_moved_goal(
            grid_file, new_goal, prob, pos_rew, neg_rew, g, h
        )
    elif st == "slip_change":
        # Same goal, different slip probability.
        mdp_post = _build_grid_mdp_from_file(
            grid_file, pp, pos_rew, neg_rew, g, h
        )
    else:
        raise ValueError(
            f"Unknown grid regime shift_type: {st!r}. "
            f"Expected one of 'goal_move', 'slip_change'."
        )

    wrapper = GridRegimeShiftWrapper(mdp_pre, mdp_post, cae)

    if time_augment:
        mdp_rl = make_time_augmented(mdp_pre, horizon=h)
    else:
        mdp_rl = mdp_pre

    return wrapper, mdp_rl, resolved
