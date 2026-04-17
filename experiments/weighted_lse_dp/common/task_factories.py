"""Task factories for the weighted-LSE DP empirical program.

Each factory returns a triple/quadruple

    (mdp_base, mdp_rl, config, [ref_pi])

where

* ``mdp_base`` is the raw MushroomRL environment used by the DP planners
  (no time augmentation);
* ``mdp_rl`` is the time-augmented environment used by the RL algorithms
  (see :mod:`mushroom_rl.environments.time_augmented_env`);
* ``config`` is a copy of the frozen task config dict (training schedule
  and evaluation protocol included);
* ``ref_pi`` is a reference policy of shape ``(horizon, n_states)`` used
  by validation scripts — it is NOT required to be optimal.

The factories are deliberately side-effect free apart from seeding
(when ``seed`` is provided, we route through
:func:`experiments.weighted_lse_dp.common.seeds.seed_everything` so that
the :class:`FiniteMDP` resets are reproducible).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from mushroom_rl.environments.generators.grid_world import generate_grid_world
from mushroom_rl.environments.generators.simple_chain import generate_simple_chain
from mushroom_rl.environments.generators.taxi import generate_taxi, parse_grid
from mushroom_rl.environments.time_augmented_env import (
    DiscreteTimeAugmentedEnv,
    make_time_augmented,
)
from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    deterministic_policy_array,
)

from experiments.weighted_lse_dp.common.seeds import seed_everything

__all__ = [
    "CHAIN_BASE_CONFIG",
    "GRID_BASE_CONFIG",
    "TAXI_BASE_CONFIG",
    "make_chain_base",
    "make_grid_base",
    "make_taxi_base",
]


# ---------------------------------------------------------------------------
# grid_base
# ---------------------------------------------------------------------------


#: Frozen configuration for the Phase I ``grid_base`` task (spec §5.1.B).
GRID_BASE_CONFIG: dict = {
    "task": "grid_base",
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
    # RL training schedule
    "train_steps": 200_000,
    "checkpoint_every": 5_000,
    "eval_episodes_checkpoint": 50,
    "eval_episodes_final": 200,
    "success_threshold": 0.90,
}


def _grid_base_reference_action_per_state(
    n_rows: int,
    n_cols: int,
    goal_cell: tuple[int, int],
) -> np.ndarray:
    """
    Build the shortest-Manhattan-path action for every cell of a dense
    rectangular grid (all cells walkable, no walls/holes).

    The base env uses the generator's action convention
    (``generate_grid_world``):

    * 0: up    (row - 1)
    * 1: down  (row + 1)
    * 2: left  (col - 1)
    * 3: right (col + 1)

    Tie-break rule (spec §5.1.B): when both row and column differ,
    move vertically first. The action at the goal cell is arbitrary —
    we pick ``up`` for determinism (the state is absorbing anyway).

    Args:
        n_rows: number of rows in the grid.
        n_cols: number of columns in the grid.
        goal_cell: ``(row, col)`` of the unique goal.

    Returns:
        An ``int64`` array of shape ``(n_rows * n_cols,)`` giving the
        action to take in each cell (row-major order).
    """
    gr, gc = int(goal_cell[0]), int(goal_cell[1])
    n_states = n_rows * n_cols
    actions = np.zeros(n_states, dtype=np.int64)
    for r in range(n_rows):
        for c in range(n_cols):
            s = r * n_cols + c
            if r == gr and c == gc:
                # Arbitrary; state is absorbing in practice.
                actions[s] = 0
                continue
            if r != gr:
                # Prefer vertical movement first (row then col).
                actions[s] = 1 if r < gr else 0  # down / up
            else:
                actions[s] = 3 if c < gc else 2  # right / left
    return actions


def make_grid_base(
    *,
    time_augment: bool = True,
    grid_file: str | Path | None = None,
    seed: int | None = None,
) -> tuple:
    """
    Create the Phase I ``grid_base`` task.

    The task is a 5x5 open grid (all cells walkable) with a single start
    at ``(0, 0)`` and a single goal at ``(4, 4)``. Actions are the four
    cardinal moves; each succeeds with probability ``0.9`` and leaves
    the agent in place with probability ``0.1``. The goal transition
    yields reward ``+1``; every other transition yields ``0``.
    ``gamma = 0.99``, ``horizon = 80``. See spec §5.1.B.

    Args:
        time_augment: whether to wrap the RL env in
            :class:`DiscreteTimeAugmentedEnv` (default ``True``). The DP
            env is never time-augmented.
        grid_file: optional override for the grid file path. When
            ``None`` we resolve the path stored in
            :data:`GRID_BASE_CONFIG` relative to the repo root
            (the current working directory at call time).
        seed: when provided, seed Python's ``random``, numpy's legacy
            global RNG and ``PYTHONHASHSEED``. This is the RNG used by
            :class:`~mushroom_rl.environments.finite_mdp.FiniteMDP` for
            ``reset`` and ``step`` sampling.

    Returns:
        ``(mdp_base, mdp_rl, config, ref_pi)`` where
        * ``mdp_base`` is a :class:`FiniteMDP`,
        * ``mdp_rl`` is the same MDP wrapped in
          :class:`DiscreteTimeAugmentedEnv` (when
          ``time_augment=True``; otherwise identical to ``mdp_base``),
        * ``config`` is a deep-ish copy of :data:`GRID_BASE_CONFIG`,
        * ``ref_pi`` is an ``int64`` array of shape ``(horizon, 25)``
          giving the shortest-Manhattan-path reference policy.
    """
    if seed is not None:
        seed_everything(seed)

    cfg = dict(GRID_BASE_CONFIG)
    # Nested tuple must be copied too so downstream mutations cannot
    # leak back into the module-level config.
    cfg["goal_cell"] = tuple(cfg["goal_cell"])

    grid_path = Path(grid_file) if grid_file is not None else Path(cfg["grid_file"])
    if not grid_path.is_file():
        raise FileNotFoundError(
            f"grid_base grid file not found at {grid_path!s} "
            f"(cwd={Path.cwd()!s})."
        )

    mdp_base = generate_grid_world(
        grid=str(grid_path),
        prob=cfg["prob"],
        pos_rew=cfg["pos_rew"],
        neg_rew=cfg["neg_rew"],
        gamma=cfg["gamma"],
        horizon=cfg["horizon"],
    )

    # Invariants on the base MDP (catch grid-file drift early).
    n_states_expected = cfg["n_rows"] * cfg["n_cols"]
    if mdp_base.info.observation_space.n != n_states_expected:
        raise ValueError(
            f"grid_base: generator returned {mdp_base.info.observation_space.n}"
            f" states but {n_states_expected} were expected — the grid file "
            f"at {grid_path} likely contains walls or has the wrong shape."
        )
    if mdp_base.info.action_space.n != 4:
        raise ValueError(
            f"grid_base: expected 4 actions, got {mdp_base.info.action_space.n}."
        )

    if time_augment:
        mdp_rl = make_time_augmented(mdp_base, horizon=cfg["horizon"])
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
    else:
        mdp_rl = mdp_base

    action_per_state = _grid_base_reference_action_per_state(
        n_rows=cfg["n_rows"],
        n_cols=cfg["n_cols"],
        goal_cell=cfg["goal_cell"],
    )
    ref_pi = deterministic_policy_array(
        horizon=cfg["horizon"],
        n_states=n_states_expected,
        action_per_state=action_per_state,
    )

    return mdp_base, mdp_rl, cfg, ref_pi


# ---------------------------------------------------------------------------
# taxi_base (Phase I spec §5.1.C)
# ---------------------------------------------------------------------------


#: Frozen configuration for the Phase I ``taxi_base`` task (spec §5.1.C).
#:
#: One passenger on a 5x5 map with 3 walls (22 free cells). The
#: passenger-state factor doubles the state space to 44. Action success
#: probability is 0.9, with slips drawn from the two perpendicular
#: directions with equal probability. Delivery yields reward +1; every
#: other transition yields 0. ``gamma = 0.99``, ``horizon = 120``.
TAXI_BASE_CONFIG: dict = {
    "task": "taxi_base",
    "grid_file": (
        "experiments/weighted_lse_dp/assets/grids/phase1_taxi_grid.txt"
    ),
    "n_passengers": 1,
    "prob": 0.9,
    "rew": (0, 1),             # (no_delivery, delivery) for 1 passenger
    "gamma": 0.99,
    "horizon": 120,
    "n_states": 44,            # 22 free cells * 2 passenger states
    # RL training schedule
    "train_steps": 300_000,
    "checkpoint_every": 10_000,
    "eval_episodes_checkpoint": 50,
    "eval_episodes_final": 200,
    "success_threshold": 0.80,
}


# Action convention used by ``generate_taxi``
# (mushroom_rl.environments.generators.taxi): ``directions`` is
# ``[[-1, 0], [1, 0], [0, -1], [0, 1]]`` ==> UP, DOWN, LEFT, RIGHT.
_TAXI_ACTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # 0: up
    (1, 0),   # 1: down
    (0, -1),  # 2: left
    (0, 1),   # 3: right
)


def _taxi_bfs_distances(
    grid_map: list[list[str]],
    target: tuple[int, int],
) -> np.ndarray:
    """
    BFS shortest-path distances on the taxi grid, treating ``'#'`` as a
    wall and every other cell (``'.'``, ``'S'``, ``'F'``, ``'G'``) as
    traversable.

    Unreachable cells receive ``np.iinfo(np.int32).max`` so the heuristic
    in :func:`_taxi_base_reference_action_per_state` can compare them
    with ``<`` without a dedicated guard.
    """
    rows = len(grid_map)
    cols = len(grid_map[0]) if rows > 0 else 0
    sentinel = np.iinfo(np.int32).max
    dist = np.full((rows, cols), sentinel, dtype=np.int32)

    tr, tc = int(target[0]), int(target[1])
    if grid_map[tr][tc] == "#":
        raise ValueError(f"BFS target cell {target!r} is a wall.")
    dist[tr, tc] = 0

    queue: deque[tuple[int, int]] = deque([(tr, tc)])
    while queue:
        r, c = queue.popleft()
        d = int(dist[r, c])
        for dr, dc in _TAXI_ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid_map[nr][nc] != "#":
                if dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc))
    return dist


def _taxi_base_reference_action_per_state(
    cell_list: list[list[int]],
    grid_map: list[list[str]],
    pickup_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> np.ndarray:
    """
    Build a stationary heuristic ``pickup-then-deliver`` policy over the
    44 taxi states (spec §5.1.C reference baseline).

    State encoding (from ``generate_taxi.compute_probabilities``):

        ``state = cell_idx + n_cells * passenger_idx``

    where ``n_cells = len(cell_list) = 22`` and ``passenger_idx in {0,
    1}``. ``passenger_idx = 0`` means the passenger has not yet been
    picked up (so the agent should head to ``F``); ``passenger_idx = 1``
    means the passenger is in the taxi (head to ``G``).

    Heuristic:
        * Choose the neighbouring in-grid, non-wall cell that minimises
          BFS distance to the current target (``F`` before pickup,
          ``G`` after).
        * Tie-break by action index ``[up, down, left, right]`` which
          implements the spec's "row first" rule.
        * At the target cell itself: F-before-pickup is a transient
          state (pickup is triggered on arrival, so the MDP never
          actually lingers there — the generator's transition table
          even marks it non-outgoing); G-after-delivery is absorbing.
          We still assign action 0 for determinism.

    Returns:
        An ``int64`` array of shape ``(n_cells * 2,) = (44,)``.
    """
    n_cells = len(cell_list)
    actions = np.zeros(n_cells * 2, dtype=np.int64)

    dist_to_pickup = _taxi_bfs_distances(grid_map, pickup_cell)
    dist_to_goal = _taxi_bfs_distances(grid_map, goal_cell)

    rows = len(grid_map)
    cols = len(grid_map[0]) if rows > 0 else 0
    sentinel = int(np.iinfo(np.int32).max)

    def _best_action(r: int, c: int, dist_map: np.ndarray) -> int:
        best_a = 0
        best_d = sentinel
        for a, (dr, dc) in enumerate(_TAXI_ACTIONS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid_map[nr][nc] == "#":
                continue
            d = int(dist_map[nr, nc])
            if d < best_d:
                best_d = d
                best_a = a
        return best_a

    for cell_idx, rc in enumerate(cell_list):
        r, c = int(rc[0]), int(rc[1])

        # passenger_idx = 0 — head to pickup.
        if (r, c) == pickup_cell:
            actions[cell_idx] = 0  # arrival triggers pickup regardless
        else:
            actions[cell_idx] = _best_action(r, c, dist_to_pickup)

        # passenger_idx = 1 — head to goal.
        flat_idx = cell_idx + n_cells
        if (r, c) == goal_cell:
            actions[flat_idx] = 0  # absorbing; any action
        else:
            actions[flat_idx] = _best_action(r, c, dist_to_goal)

    return actions


def make_taxi_base(
    *,
    time_augment: bool = True,
    grid_file: str | Path | None = None,
    seed: int | None = None,
) -> tuple:
    """
    Create the Phase I ``taxi_base`` task (spec §5.1.C).

    Configuration (see :data:`TAXI_BASE_CONFIG`):
        * 1 passenger on the 5x5 Phase I taxi grid (3 walls, 22 free
          cells ==> 44 joint states).
        * ``prob = 0.9`` per-action success probability; slips go to one
          of the two perpendicular cells with equal probability.
        * ``rew = (0, 1)`` — reward is ``+1`` on delivering the
          passenger to ``G`` and ``0`` otherwise.
        * ``gamma = 0.99``, ``horizon = 120`` (finite, explicit — not
          ``np.inf``).

    Args:
        time_augment: when ``True`` (default) wrap the base FiniteMDP in
            a :class:`DiscreteTimeAugmentedEnv` of horizon 120. RL
            agents and stage-indexed planners consume this wrapped env.
            When ``False`` the same ``mdp_base`` is returned in the
            ``mdp_rl`` slot (handy for DP-only smoke tests).
        grid_file: optional override for the grid path. Defaults to the
            spec-mandated file stored under
            ``experiments/weighted_lse_dp/assets/grids/``.
        seed: optional non-negative integer. When provided we route
            through :func:`seeds.seed_everything` so that
            :class:`~mushroom_rl.environments.finite_mdp.FiniteMDP`
            resets and transitions are reproducible.

    Returns:
        ``(mdp_base, mdp_rl, config, ref_pi)`` where
        * ``mdp_base`` is a :class:`FiniteMDP` with 44 states, 4
          actions, ``gamma=0.99`` and ``horizon=120``;
        * ``mdp_rl`` is a :class:`DiscreteTimeAugmentedEnv` of size
          ``120 * 44 = 5280`` when ``time_augment=True``, else
          ``mdp_base`` itself;
        * ``config`` is a shallow copy of :data:`TAXI_BASE_CONFIG` with
          the ``rew`` tuple preserved;
        * ``ref_pi`` is an ``int64`` array of shape ``(120, 44)``
          giving the stationary heuristic pickup-then-deliver policy
          tiled across stages (see the reference-policy note below).

    Reference policy heuristic:
        We decode each joint state into ``(cell_idx, passenger_idx)``
        using the encoding ``state = cell_idx + 22 * passenger_idx``
        baked into ``generate_taxi``. When the passenger has not been
        picked up (``passenger_idx = 0``) the policy BFS-navigates to
        the pickup cell ``F = (4, 0)``; once the passenger is in the
        taxi (``passenger_idx = 1``) it BFS-navigates to the goal
        ``G = (4, 4)``. Action ties (in case of equal BFS distance) are
        broken by action order ``[up, down, left, right]``, which
        implements the spec's row-first rule. At the target cell itself
        (F before pickup is a transient state that the generator marks
        as non-outgoing; G after delivery is absorbing) we return
        action 0 deterministically. The policy is stationary across
        stages — the shortest-path length is bounded by the grid
        diameter (<< the horizon of 120).
    """
    if seed is not None:
        seed_everything(seed)

    cfg = dict(TAXI_BASE_CONFIG)
    # Preserve tuple-typed fields so downstream consumers can rely on
    # immutability.
    cfg["rew"] = tuple(cfg["rew"])

    grid_path = (
        Path(grid_file) if grid_file is not None else Path(cfg["grid_file"])
    )
    if not grid_path.is_file():
        raise FileNotFoundError(
            f"taxi_base grid file not found at {grid_path!s} "
            f"(cwd={Path.cwd()!s})."
        )

    mdp_base = generate_taxi(
        grid=str(grid_path),
        prob=float(cfg["prob"]),
        rew=cfg["rew"],
        gamma=float(cfg["gamma"]),
        horizon=int(cfg["horizon"]),
    )

    # Invariants on the base MDP (catch grid-file / passenger-count drift
    # early — a wrong grid would silently alter n_states).
    n_states_actual = int(mdp_base.info.observation_space.n)
    if n_states_actual != cfg["n_states"]:
        raise ValueError(
            f"taxi_base: generator returned {n_states_actual} states but "
            f"{cfg['n_states']} were expected — grid file at {grid_path} "
            f"may have the wrong shape / wall count / passenger count."
        )
    if mdp_base.info.action_space.n != 4:
        raise ValueError(
            f"taxi_base: expected 4 actions, got "
            f"{mdp_base.info.action_space.n}."
        )
    if mdp_base.info.horizon != cfg["horizon"]:
        raise ValueError(
            f"taxi_base: expected horizon={cfg['horizon']}, got "
            f"{mdp_base.info.horizon}."
        )

    # Parse the grid a second time so we can recover ``cell_list`` /
    # ``passenger_list`` / the ``G`` location for the heuristic policy;
    # ``generate_taxi`` does not expose them via the returned FiniteMDP.
    grid_map, cell_list, passenger_list = parse_grid(str(grid_path))
    if len(passenger_list) != cfg["n_passengers"]:
        raise ValueError(
            f"taxi_base: grid {grid_path} has {len(passenger_list)} "
            f"passenger cell(s); expected {cfg['n_passengers']}."
        )
    pickup_cell: tuple[int, int] = (
        int(passenger_list[0][0]),
        int(passenger_list[0][1]),
    )
    goal_cell: tuple[int, int] | None = None
    for r, row in enumerate(grid_map):
        for c, ch in enumerate(row):
            if ch == "G":
                goal_cell = (r, c)
                break
        if goal_cell is not None:
            break
    if goal_cell is None:
        raise ValueError(f"No 'G' cell found in taxi grid {grid_path}.")

    if time_augment:
        mdp_rl = make_time_augmented(mdp_base, horizon=cfg["horizon"])
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
    else:
        mdp_rl = mdp_base

    action_per_state = _taxi_base_reference_action_per_state(
        cell_list=cell_list,
        grid_map=grid_map,
        pickup_cell=pickup_cell,
        goal_cell=goal_cell,
    )
    ref_pi = deterministic_policy_array(
        horizon=cfg["horizon"],
        n_states=cfg["n_states"],
        action_per_state=action_per_state,
    )

    return mdp_base, mdp_rl, cfg, ref_pi


# ---------------------------------------------------------------------------
# chain_base (Phase I spec §5.1.A)
# ---------------------------------------------------------------------------


#: Frozen configuration for the Phase I ``chain_base`` task (spec §5.1.A).
#:
#: A 25-state simple chain with the goal at the rightmost state (index
#: 24). Each action succeeds with probability ``0.9``; with probability
#: ``0.1`` the agent stays in place. Reaching the goal yields ``+1``;
#: every other transition yields ``0``. ``gamma = 0.99``,
#: ``horizon = 60``.
CHAIN_BASE_CONFIG: dict = {
    "task": "chain_base",
    "state_n": 25,
    "goal": 24,
    "prob": 0.9,
    "rew": 1.0,
    "gamma": 0.99,
    "horizon": 60,
    # RL training schedule
    "train_steps": 200_000,
    "checkpoint_every": 5_000,
    "eval_episodes_checkpoint": 50,
    "eval_episodes_final": 200,
    "success_threshold": 0.90,
}


def make_chain_base(
    *,
    time_augment: bool = True,
    seed: int | None = None,
) -> tuple:
    """
    Create the Phase I ``chain_base`` task (spec §5.1.A).

    The task is a 25-state simple chain with the goal at the rightmost
    state (index 24). Action ``0`` moves the agent RIGHT (toward the
    goal) with probability ``0.9`` and leaves it in place with
    probability ``0.1``; action ``1`` moves it LEFT (away from the
    goal) with the same success probability. Reaching the goal yields
    reward ``+1``; every other transition yields ``0``.
    ``gamma = 0.99``, ``horizon = 60``.

    Action convention (confirmed against
    :func:`mushroom_rl.environments.generators.simple_chain.compute_probabilities`):

    * ``0``: move right (``i -> i + 1`` with ``prob``),
    * ``1``: move left  (``i -> i - 1`` with ``prob``).

    The reference policy is therefore the always-right policy
    ``ref_pi[:] = 0``.

    Args:
        time_augment: whether to wrap the RL env in
            :class:`DiscreteTimeAugmentedEnv` (default ``True``). The DP
            env (``mdp_base``) is never time-augmented.
        seed: when provided, seed Python's ``random``, numpy's legacy
            global RNG and ``PYTHONHASHSEED``. This is the RNG used by
            :class:`~mushroom_rl.environments.finite_mdp.FiniteMDP` for
            ``reset`` and ``step`` sampling.

    Returns:
        ``(mdp_base, mdp_rl, config, ref_pi)`` where
        * ``mdp_base`` is a :class:`FiniteMDP` with 25 states and 2
          actions;
        * ``mdp_rl`` is the same MDP wrapped in
          :class:`DiscreteTimeAugmentedEnv` of size ``60 * 25 = 1500``
          when ``time_augment=True``, else ``mdp_base`` itself;
        * ``config`` is a shallow copy of :data:`CHAIN_BASE_CONFIG`;
        * ``ref_pi`` is an ``int64`` array of shape ``(60, 25)`` with
          every entry equal to ``0`` (always-right reference policy).
    """
    if seed is not None:
        seed_everything(seed)

    cfg = dict(CHAIN_BASE_CONFIG)

    # ``generate_simple_chain`` expects ``goal_states`` as a list.
    mdp_base = generate_simple_chain(
        state_n=int(cfg["state_n"]),
        goal_states=[int(cfg["goal"])],
        prob=float(cfg["prob"]),
        rew=float(cfg["rew"]),
        gamma=float(cfg["gamma"]),
        horizon=int(cfg["horizon"]),
    )

    # Invariants on the base MDP.
    n_states = mdp_base.info.observation_space.size[0]
    n_actions = mdp_base.info.action_space.size[0]
    assert n_states == 25, f"chain_base: expected 25 states, got {n_states}"
    assert n_actions == 2, f"chain_base: expected 2 actions, got {n_actions}"

    if time_augment:
        mdp_rl = make_time_augmented(mdp_base, horizon=cfg["horizon"])
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
    else:
        mdp_rl = mdp_base

    action_per_state = np.zeros(n_states, dtype=np.int64)
    ref_pi = deterministic_policy_array(
        horizon=cfg["horizon"],
        n_states=n_states,
        action_per_state=action_per_state,
    )
    assert np.all(ref_pi == 0), (
        "chain_base: ref_pi must be always-right (action 0)"
    )

    return mdp_base, mdp_rl, cfg, ref_pi
