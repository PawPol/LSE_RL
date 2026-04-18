"""Hazard-cell environment wrappers for Phase II catastrophe stress tasks.

Wrappers add one or more hazard cells/transitions that give immediate large
negative rewards and optionally terminate the episode.  The base MDP
transition matrix and action space are unchanged; hazard events are
injected at step time.

Implemented here:
  - ``GridHazardWrapper``  (spec S5.2.B)
  - ``make_grid_hazard``   (factory for grid + hazard wrapper)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mushroom_rl.environments.generators.grid_world import generate_grid_world
from mushroom_rl.environments.time_augmented_env import (
    DiscreteTimeAugmentedEnv,
    make_time_augmented,
)

from experiments.weighted_lse_dp.common.seeds import seed_everything

__all__: list[str] = [
    "GridHazardWrapper",
    "make_grid_hazard",
]


class GridHazardWrapper:
    """Grid with hazard cells: immediate negative reward on entry, optional termination.

    When the agent transitions into a hazard cell, with probability
    ``hazard_prob`` it receives ``hazard_reward`` (added to the base reward)
    and optionally the episode terminates (``hazard_terminates=True``).

    The wrapper intercepts ``step`` and checks the *next* state against the
    hazard set.  It does NOT modify the underlying transition matrix, so the
    base MDP's ``p`` and ``r`` arrays are still available for DP planners that
    need access to the original model.

    severity=0 equivalence: ``hazard_states=[]`` or ``hazard_prob=0.0``
    makes this wrapper fully transparent.
    """

    def __init__(
        self,
        base_mdp,
        hazard_states: list[int],
        hazard_prob: float = 1.0,
        hazard_reward: float = -5.0,
        hazard_terminates: bool = True,
        rng_seed: int | None = None,
    ):
        self._base = base_mdp
        self._hazard_set: frozenset[int] = frozenset(hazard_states)
        self._hazard_prob = hazard_prob
        self._hazard_reward = hazard_reward
        self._hazard_terminates = hazard_terminates
        self.info = base_mdp.info
        self._rng = np.random.default_rng(rng_seed)
        self._current_state = None

    def __getattr__(self, name: str):
        """Delegate attribute access to the base MDP for anything not
        explicitly defined on the wrapper (e.g., ``p``, ``r``, etc.)."""
        return getattr(self._base, name)

    def reset(self, state=None):
        """Reset the base MDP and track the current state."""
        result = self._base.reset(state)
        if isinstance(result, tuple):
            self._current_state = result[0]
        else:
            self._current_state = result
        return result

    def step(self, action):
        """Step the base MDP and inject hazard if next_state is a hazard cell."""
        result = self._base.step(action)
        next_state, reward, absorbing, info = result

        # Decode state to integer index.
        s = int(next_state[0]) if hasattr(next_state, '__len__') else int(next_state)

        if s in self._hazard_set and not absorbing:
            if self._rng.random() < self._hazard_prob:
                reward = reward + self._hazard_reward
                absorbing = self._hazard_terminates

        self._current_state = next_state
        return next_state, reward, absorbing, info

    def stop(self) -> None:
        """No-op stop; required by MushroomRL Core/DiscreteTimeAugmentedEnv."""


# ---------------------------------------------------------------------------
# make_grid_hazard factory  (spec S5.2.B)
# ---------------------------------------------------------------------------

#: Frozen configuration for the ``grid_hazard`` stress task.
GRID_HAZARD_CONFIG: dict = {
    "task": "grid_hazard",
    "grid_file": (
        "experiments/weighted_lse_dp/assets/grids/phase1_base_grid.txt"
    ),
    "n_rows": 5,
    "n_cols": 5,
    "prob": 0.9,
    "pos_rew": 1.0,
    "neg_rew": 0.0,
    "gamma": 0.99,
    "horizon": 80,
    "goal_cell": (4, 4),
    # Hazard defaults: cell 12 = (2,2) center of 5x5 grid, on shortest path.
    "hazard_states": [12],
    "hazard_prob": 1.0,
    "hazard_reward": -5.0,
    "hazard_terminates": True,
    # RL training schedule
    "train_steps": 200_000,
    "checkpoint_every": 5_000,
    "eval_episodes_checkpoint": 50,
    "eval_episodes_final": 200,
    "success_threshold": 0.70,
}


def make_grid_hazard(
    cfg: dict,
    prob: float = 0.9,
    gamma: float = 0.99,
    horizon: int = 80,
    hazard_states: list[int] | None = None,
    hazard_prob: float = 1.0,
    hazard_reward: float = -5.0,
    hazard_terminates: bool = True,
    *,
    time_augment: bool = True,
    seed: int | None = None,
) -> tuple:
    """Create the Phase II ``grid_hazard`` stress task (spec S5.2.B).

    Builds the Phase I base 5x5 grid MDP, then wraps it in
    :class:`GridHazardWrapper` which injects hazard events (negative
    rewards and optional termination) when the agent enters designated
    cells.

    Default hazard cell: state 12 = cell (2,2) in row-major order, which
    lies on the Manhattan shortest path from (0,0) to (4,4).

    severity=0 equivalence: ``hazard_states=[]`` or ``hazard_prob=0.0``
    recovers the base grid task exactly.

    Args:
        cfg: caller-supplied overrides.
        prob: action success probability.
        gamma: discount factor.
        horizon: finite horizon length.
        hazard_states: state indices of hazard cells (default: [12]).
        hazard_prob: probability hazard fires on entry.
        hazard_reward: immediate reward when hazard fires.
        hazard_terminates: whether hazard absorbs (ends episode).
        time_augment: whether to wrap the RL env in
            :class:`DiscreteTimeAugmentedEnv`.
        seed: optional RNG seed for reproducibility.

    Returns:
        ``(wrapper, mdp_rl, resolved_cfg)`` where ``wrapper`` is a
        :class:`GridHazardWrapper` around the base grid MDP.
    """
    if seed is not None:
        seed_everything(seed)

    resolved = dict(GRID_HAZARD_CONFIG)
    resolved.update({
        "prob": prob,
        "gamma": gamma,
        "horizon": horizon,
        "hazard_prob": hazard_prob,
        "hazard_reward": hazard_reward,
        "hazard_terminates": hazard_terminates,
    })
    if hazard_states is not None:
        resolved["hazard_states"] = list(hazard_states)
    resolved.update(cfg)

    grid_path = Path(resolved["grid_file"])
    if not grid_path.is_file():
        raise FileNotFoundError(
            f"grid_hazard grid file not found at {grid_path!s} "
            f"(cwd={Path.cwd()!s})."
        )

    mdp_base = generate_grid_world(
        grid=str(grid_path),
        prob=float(resolved["prob"]),
        pos_rew=float(resolved["pos_rew"]),
        neg_rew=float(resolved["neg_rew"]),
        gamma=float(resolved["gamma"]),
        horizon=int(resolved["horizon"]),
    )

    n_states_expected = resolved["n_rows"] * resolved["n_cols"]
    if mdp_base.info.observation_space.n != n_states_expected:
        raise ValueError(
            f"grid_hazard: generator returned "
            f"{mdp_base.info.observation_space.n} states but "
            f"{n_states_expected} were expected."
        )

    # Validate hazard states are within bounds.
    for hs in resolved["hazard_states"]:
        if not (0 <= hs < n_states_expected):
            raise ValueError(
                f"hazard_state={hs} out of range [0, {n_states_expected})."
            )

    wrapper = GridHazardWrapper(
        base_mdp=mdp_base,
        hazard_states=resolved["hazard_states"],
        hazard_prob=float(resolved["hazard_prob"]),
        hazard_reward=float(resolved["hazard_reward"]),
        hazard_terminates=bool(resolved["hazard_terminates"]),
        rng_seed=seed,
    )

    if time_augment:
        # R8-2: time-augment the hazard wrapper, not the bare base MDP.
        # The RL agent must train on the stressed environment.
        mdp_rl = make_time_augmented(
            wrapper, horizon=int(resolved["horizon"])
        )
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
    else:
        mdp_rl = wrapper

    return wrapper, mdp_rl, resolved
