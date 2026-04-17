"""Stress-test task factories for Phase II of the weighted-LSE DP paper suite.

Each factory reduces to its Phase I base task when the severity parameter is
zero.  All factories follow the same signature convention as the Phase I
factories in ``experiments.weighted_lse_dp.common.task_factories``.

Implemented here:
  - ``make_chain_sparse_long`` (spec S5.1.A)
  - ``make_chain_jackpot``     (spec S5.1.B)
  - ``make_chain_catastrophe`` (spec S5.1.C)
  - ``make_grid_sparse_goal``  (spec S5.2.A)
  - ``make_taxi_bonus_shock``  (spec S5.3.A)

Wrappers for nonstationary/hazard tasks live in the sibling modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.environments.generators.grid_world import generate_grid_world
from mushroom_rl.environments.generators.simple_chain import (
    compute_probabilities,
    compute_reward,
    generate_simple_chain,
)
from mushroom_rl.environments.generators.taxi import generate_taxi
from mushroom_rl.environments.time_augmented_env import (
    DiscreteTimeAugmentedEnv,
    make_time_augmented,
)

from experiments.weighted_lse_dp.common.seeds import seed_everything

__all__: list[str] = [
    "make_chain_sparse_long",
    "make_chain_jackpot",
    "make_chain_catastrophe",
    "make_grid_sparse_goal",
    "make_taxi_bonus_shock",
    "TaxiBonusShockWrapper",
]


# ---------------------------------------------------------------------------
# chain_sparse_long  (Phase II spec S5.1.A)
# ---------------------------------------------------------------------------


def make_chain_sparse_long(
    cfg: dict,
    state_n: int = 60,
    prob: float = 0.9,
    gamma: float = 0.99,
    horizon: int = 120,
) -> tuple:
    """Longer chain with goal-only reward (no shaping).

    When ``state_n=25, prob=0.9, gamma=0.99, horizon=60`` this recovers
    the Phase I ``chain_base`` exactly (severity=0 semantics).

    Args:
        cfg: caller-supplied overrides (reserved for future use).
        state_n: number of chain states (default 60).
        prob: action success probability.
        gamma: discount factor.
        horizon: episode length.

    Returns:
        ``(mdp_base, mdp_rl, resolved_cfg)`` triple.
    """
    goal_states = [state_n - 1]

    mdp_base = generate_simple_chain(
        state_n=state_n,
        goal_states=goal_states,
        prob=prob,
        rew=1.0,
        mu=None,
        gamma=gamma,
        horizon=horizon,
    )

    mdp_rl = make_time_augmented(mdp_base, horizon=horizon)

    resolved_cfg = {
        "task": "chain_sparse_long",
        "state_n": state_n,
        "prob": prob,
        "gamma": gamma,
        "horizon": horizon,
        "goal_states": goal_states,
    }
    resolved_cfg.update(cfg)

    return mdp_base, mdp_rl, resolved_cfg


# ---------------------------------------------------------------------------
# chain_jackpot  (Phase II spec S5.1.B)
# ---------------------------------------------------------------------------


def make_chain_jackpot(
    cfg: dict,
    state_n: int = 25,
    prob: float = 0.9,
    gamma: float = 0.99,
    horizon: int = 60,
    jackpot_state: int = 20,
    jackpot_prob: float = 0.05,
    jackpot_reward: float = 10.0,
    jackpot_terminates: bool = True,
) -> tuple:
    """Chain with a rare jackpot transition at a designated state.

    At ``jackpot_state``, action 0 (right/forward) triggers the jackpot
    with probability ``jackpot_prob``: immediate reward ``jackpot_reward``
    and (when ``jackpot_terminates=True``) the episode ends by
    transitioning to a dedicated absorbing state. With probability
    ``1 - jackpot_prob`` the normal chain dynamics apply.

    **severity=0**: ``jackpot_prob=0.0`` recovers ``chain_base`` exactly.

    Args:
        cfg: caller-supplied overrides (reserved for future use).
        state_n: number of chain states (excluding any absorbing
            terminal added when ``jackpot_terminates=True``).
        prob: action success probability for non-jackpot transitions.
        gamma: discount factor.
        horizon: episode length.
        jackpot_state: which state has the jackpot action (must be in
            ``[0, state_n - 2]`` to avoid clashing with the goal).
        jackpot_prob: probability of the jackpot firing on action 0 at
            ``jackpot_state``.
        jackpot_reward: immediate reward on jackpot.
        jackpot_terminates: if ``True``, the jackpot transitions to a
            dedicated absorbing terminal state (episode ends). If
            ``False``, the agent self-loops at ``jackpot_state``.

    Returns:
        ``(mdp_base, mdp_rl, resolved_cfg)`` triple.
    """
    if not (0 <= jackpot_state < state_n - 1):
        raise ValueError(
            f"jackpot_state={jackpot_state} must be in [0, {state_n - 2}]."
        )

    # Start from the standard chain P and R (S, 2, S).
    p = compute_probabilities(state_n, prob)
    r = compute_reward(state_n, [state_n - 1], 1.0)

    if jackpot_prob > 0.0:
        if jackpot_terminates:
            # Add a dedicated absorbing state (index = state_n).
            # An all-zero P row makes FiniteMDP.step set absorbing=True.
            S_total = state_n + 1
            A = 2
            p_ext = np.zeros((S_total, A, S_total), dtype=np.float64)
            r_ext = np.zeros((S_total, A, S_total), dtype=np.float64)

            # Copy original transitions into the extended arrays.
            p_ext[:state_n, :, :state_n] = p
            r_ext[:state_n, :, :state_n] = r

            # Absorbing terminal: P row is all zeros (MushroomRL convention).
            # (already zero from initialization)

            # Modify jackpot_state, action 0.
            normal_row = p_ext[jackpot_state, 0, :].copy()
            normal_rew = r_ext[jackpot_state, 0, :].copy()

            p_ext[jackpot_state, 0, :] = (1.0 - jackpot_prob) * normal_row
            r_ext[jackpot_state, 0, :] = normal_rew  # base reward unchanged

            # Jackpot outcome: transition to absorbing state.
            p_ext[jackpot_state, 0, S_total - 1] = jackpot_prob
            r_ext[jackpot_state, 0, S_total - 1] = jackpot_reward

            p = p_ext
            r = r_ext
        else:
            # Jackpot self-loops at jackpot_state (no termination).
            normal_row = p[jackpot_state, 0, :].copy()

            p[jackpot_state, 0, :] = (1.0 - jackpot_prob) * normal_row
            p[jackpot_state, 0, jackpot_state] += jackpot_prob

            # Jackpot reward on the self-loop transition.
            r[jackpot_state, 0, jackpot_state] = jackpot_reward

    mdp_base = FiniteMDP(p, r, mu=None, gamma=gamma, horizon=horizon)
    mdp_rl = make_time_augmented(mdp_base, horizon=horizon)

    resolved_cfg = {
        "task": "chain_jackpot",
        "state_n": state_n,
        "prob": prob,
        "gamma": gamma,
        "horizon": horizon,
        "jackpot_state": jackpot_state,
        "jackpot_prob": jackpot_prob,
        "jackpot_reward": jackpot_reward,
        "jackpot_terminates": jackpot_terminates,
    }
    resolved_cfg.update(cfg)

    return mdp_base, mdp_rl, resolved_cfg


# ---------------------------------------------------------------------------
# grid_sparse_goal  (Phase II spec S5.2.A)
# ---------------------------------------------------------------------------


#: Frozen configuration for the ``grid_sparse_goal`` stress task.
GRID_SPARSE_GOAL_CONFIG: dict = {
    "task": "grid_sparse_goal",
    "grid_file": (
        "experiments/weighted_lse_dp/assets/grids/phase1_base_grid.txt"
    ),
    "n_rows": 5,
    "n_cols": 5,
    "prob": 0.9,
    "goal_reward": 1.0,
    "step_penalty": 0.0,
    "gamma": 0.99,
    "horizon": 80,
    "goal_cell": (4, 4),
    # RL training schedule
    "train_steps": 200_000,
    "checkpoint_every": 5_000,
    "eval_episodes_checkpoint": 50,
    "eval_episodes_final": 200,
    "success_threshold": 0.80,
}


def make_grid_sparse_goal(
    cfg: dict,
    grid_file: str = "experiments/weighted_lse_dp/assets/grids/phase1_base_grid.txt",
    prob: float = 0.9,
    gamma: float = 0.99,
    horizon: int = 80,
    goal_reward: float = 1.0,
    *,
    time_augment: bool = True,
    seed: int | None = None,
) -> tuple:
    """Create the Phase II ``grid_sparse_goal`` stress task (spec S5.2.A).

    Same 5x5 open grid as Phase I ``grid_base`` but with goal-only reward:
    the agent receives ``goal_reward`` upon reaching the goal cell and 0
    everywhere else (no step penalty / shaping).

    severity=0 equivalence: when ``goal_reward=1.0`` and step_penalty=0
    this is identical to ``grid_base`` (which already has ``neg_rew=0``).

    Args:
        cfg: caller-supplied overrides (merged into defaults).
        grid_file: path to the grid text file.
        prob: action success probability.
        gamma: discount factor.
        horizon: finite horizon length.
        goal_reward: reward for reaching the goal cell.
        time_augment: whether to wrap the RL env in
            :class:`DiscreteTimeAugmentedEnv`.
        seed: optional RNG seed for reproducibility.

    Returns:
        ``(mdp_base, mdp_rl, resolved_cfg)``
    """
    if seed is not None:
        seed_everything(seed)

    resolved = dict(GRID_SPARSE_GOAL_CONFIG)
    resolved.update({
        "grid_file": grid_file,
        "prob": prob,
        "gamma": gamma,
        "horizon": horizon,
        "goal_reward": goal_reward,
    })
    resolved.update(cfg)

    grid_path = Path(resolved["grid_file"])
    if not grid_path.is_file():
        raise FileNotFoundError(
            f"grid_sparse_goal grid file not found at {grid_path!s} "
            f"(cwd={Path.cwd()!s})."
        )

    # pos_rew = goal_reward, neg_rew = 0 (no step penalty / hole penalty).
    mdp_base = generate_grid_world(
        grid=str(grid_path),
        prob=float(resolved["prob"]),
        pos_rew=float(resolved["goal_reward"]),
        neg_rew=float(resolved["step_penalty"]),
        gamma=float(resolved["gamma"]),
        horizon=int(resolved["horizon"]),
    )

    n_states_expected = resolved["n_rows"] * resolved["n_cols"]
    if mdp_base.info.observation_space.n != n_states_expected:
        raise ValueError(
            f"grid_sparse_goal: generator returned "
            f"{mdp_base.info.observation_space.n} states but "
            f"{n_states_expected} were expected."
        )

    if time_augment:
        mdp_rl = make_time_augmented(
            mdp_base, horizon=int(resolved["horizon"])
        )
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
    else:
        mdp_rl = mdp_base

    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# chain_catastrophe  (Phase II spec S5.1.C)
# ---------------------------------------------------------------------------


def make_chain_catastrophe(
    cfg: dict,
    state_n: int = 25,
    prob: float = 0.9,
    gamma: float = 0.99,
    horizon: int = 60,
    risky_state: int = 15,
    risky_prob: float = 0.05,
    catastrophe_reward: float = -10.0,
    shortcut_prob: float = 0.90,
    shortcut_jump: int = 5,
) -> tuple:
    """Chain with a risky shortcut: rare catastrophe vs fast forward jump.

    At ``risky_state``, action 0 (right/forward) is modified:

    * With probability ``risky_prob``: catastrophe -- immediate reward
      ``catastrophe_reward`` and the agent is sent to a dedicated
      absorbing terminal state (episode ends).
    * With probability ``1 - risky_prob``: the agent jumps forward
      ``shortcut_jump`` states (clamped to the goal state).

    Action 1 (left) at ``risky_state`` is unchanged (normal chain
    dynamics). All other states follow standard chain dynamics.

    **severity=0**: ``risky_prob=0.0`` recovers ``chain_base`` exactly
    (the risky state retains its normal chain dynamics when the
    catastrophe probability is zero).

    Args:
        cfg: caller-supplied overrides (reserved for future use).
        state_n: number of chain states (excluding the absorbing
            catastrophe terminal state).
        prob: action success probability for non-risky transitions.
        gamma: discount factor.
        horizon: episode length.
        risky_state: which state has the risky shortcut (must be in
            ``[0, state_n - 2]``).
        risky_prob: probability of catastrophe on action 0 at
            ``risky_state``.
        catastrophe_reward: immediate reward on catastrophe.
        shortcut_prob: retained for API compat; the shortcut fires
            deterministically when the catastrophe does not.
        shortcut_jump: number of states to jump forward on successful
            shortcut.

    Returns:
        ``(mdp_base, mdp_rl, resolved_cfg)`` triple.
    """
    if not (0 <= risky_state < state_n - 1):
        raise ValueError(
            f"risky_state={risky_state} must be in [0, {state_n - 2}]."
        )

    # Start from the standard chain P and R.
    p = compute_probabilities(state_n, prob)
    r = compute_reward(state_n, [state_n - 1], 1.0)

    goal = state_n - 1

    if risky_prob > 0.0:
        # Add a dedicated absorbing catastrophe state (all-zero P row).
        S_total = state_n + 1
        A = 2
        p_ext = np.zeros((S_total, A, S_total), dtype=np.float64)
        r_ext = np.zeros((S_total, A, S_total), dtype=np.float64)

        p_ext[:state_n, :, :state_n] = p
        r_ext[:state_n, :, :state_n] = r

        # Risky state, action 0: catastrophe or shortcut.
        # Clear the normal transitions for action 0 at risky_state.
        p_ext[risky_state, 0, :] = 0.0
        r_ext[risky_state, 0, :] = 0.0

        # Catastrophe outcome.
        p_ext[risky_state, 0, S_total - 1] = risky_prob
        r_ext[risky_state, 0, S_total - 1] = catastrophe_reward

        # Shortcut outcome: jump forward, clamped to goal.
        shortcut_dest = min(risky_state + shortcut_jump, goal)
        p_ext[risky_state, 0, shortcut_dest] += 1.0 - risky_prob

        # If the shortcut lands on the goal, give the goal reward.
        if shortcut_dest == goal:
            r_ext[risky_state, 0, shortcut_dest] = 1.0

        p = p_ext
        r = r_ext
    # else: risky_prob == 0.0 => keep standard chain P/R unchanged
    # (severity=0 identity recovery).

    mdp_base = FiniteMDP(p, r, mu=None, gamma=gamma, horizon=horizon)
    mdp_rl = make_time_augmented(mdp_base, horizon=horizon)

    resolved_cfg = {
        "task": "chain_catastrophe",
        "state_n": state_n,
        "prob": prob,
        "gamma": gamma,
        "horizon": horizon,
        "risky_state": risky_state,
        "risky_prob": risky_prob,
        "catastrophe_reward": catastrophe_reward,
        "shortcut_prob": shortcut_prob,
        "shortcut_jump": shortcut_jump,
    }
    resolved_cfg.update(cfg)

    return mdp_base, mdp_rl, resolved_cfg


# ---------------------------------------------------------------------------
# taxi_bonus_shock  (Phase II spec S5.3.A)
# ---------------------------------------------------------------------------


class TaxiBonusShockWrapper:
    """Taxi wrapper that adds a rare positive windfall on delivery.

    With probability ``bonus_prob``, a successful delivery (base reward > 0.5)
    receives an additional ``bonus_reward``.  When ``bonus_prob=0`` this
    wrapper is transparent and the MDP is identical to the base taxi task.
    """

    def __init__(
        self,
        base_mdp,
        bonus_prob: float,
        bonus_reward: float,
        rng_seed: int | None = None,
    ):
        self._base = base_mdp
        self._bonus_prob = bonus_prob
        self._bonus_reward = bonus_reward
        self.info = base_mdp.info
        self._rng = np.random.default_rng(rng_seed)

    def __getattr__(self, name: str):
        """Delegate attribute access to the base MDP for anything not
        explicitly defined on the wrapper (e.g., ``p``, ``r``, etc.)."""
        return getattr(self._base, name)

    def reset(self, state=None):
        """Reset the base MDP.  Returns whatever the base returns."""
        return self._base.reset(state)

    def step(self, action):
        """Step the base MDP and inject bonus shock on delivery."""
        result = self._base.step(action)
        next_state, reward, absorbing, info = result
        # Delivery in the base taxi gives reward=1; detect with > 0.5.
        if reward > 0.5 and self._rng.random() < self._bonus_prob:
            reward = reward + self._bonus_reward
        return next_state, reward, absorbing, info


#: Frozen configuration for the ``taxi_bonus_shock`` stress task.
TAXI_BONUS_SHOCK_CONFIG: dict = {
    "task": "taxi_bonus_shock",
    "grid_file": (
        "experiments/weighted_lse_dp/assets/grids/phase1_taxi_grid.txt"
    ),
    "n_passengers": 1,
    "prob": 0.9,
    "rew": (0, 1),
    "gamma": 0.99,
    "horizon": 120,
    "n_states": 44,
    "bonus_prob": 0.05,
    "bonus_reward": 5.0,
    # RL training schedule
    "train_steps": 300_000,
    "checkpoint_every": 10_000,
    "eval_episodes_checkpoint": 50,
    "eval_episodes_final": 200,
    "success_threshold": 0.70,
}


def make_taxi_bonus_shock(
    cfg: dict,
    grid_file: str = "experiments/weighted_lse_dp/assets/grids/phase1_taxi_grid.txt",
    prob: float = 0.9,
    gamma: float = 0.99,
    horizon: int = 120,
    bonus_prob: float = 0.05,
    bonus_reward: float = 5.0,
    *,
    time_augment: bool = True,
    seed: int | None = None,
) -> tuple:
    """Create the Phase II ``taxi_bonus_shock`` stress task (spec S5.3.A).

    Builds the Phase I base taxi MDP, then wraps it in
    :class:`TaxiBonusShockWrapper` which injects a rare positive windfall
    on delivery.  With probability ``bonus_prob`` a successful delivery
    yields an extra ``bonus_reward`` on top of the base delivery reward.

    severity=0 equivalence: ``bonus_prob=0.0`` recovers the base taxi
    task exactly (the wrapper becomes transparent).

    Args:
        cfg: caller-supplied overrides.
        grid_file: path to the taxi grid text file.
        prob: action success probability.
        gamma: discount factor.
        horizon: finite horizon length.
        bonus_prob: probability of bonus on delivery.
        bonus_reward: extra reward on bonus delivery.
        time_augment: whether to wrap the RL env in
            :class:`DiscreteTimeAugmentedEnv`.
        seed: optional RNG seed for reproducibility.

    Returns:
        ``(wrapped_mdp, mdp_rl, resolved_cfg)`` where ``wrapped_mdp``
        is a :class:`TaxiBonusShockWrapper` around the base taxi MDP.
    """
    if seed is not None:
        seed_everything(seed)

    resolved = dict(TAXI_BONUS_SHOCK_CONFIG)
    resolved.update({
        "grid_file": grid_file,
        "prob": prob,
        "gamma": gamma,
        "horizon": horizon,
        "bonus_prob": bonus_prob,
        "bonus_reward": bonus_reward,
    })
    resolved.update(cfg)

    grid_path = Path(resolved["grid_file"])
    if not grid_path.is_file():
        raise FileNotFoundError(
            f"taxi_bonus_shock grid file not found at {grid_path!s} "
            f"(cwd={Path.cwd()!s})."
        )

    mdp_base = generate_taxi(
        grid=str(grid_path),
        prob=float(resolved["prob"]),
        rew=tuple(resolved["rew"]),
        gamma=float(resolved["gamma"]),
        horizon=int(resolved["horizon"]),
    )

    n_states_actual = int(mdp_base.info.observation_space.n)
    if n_states_actual != resolved["n_states"]:
        raise ValueError(
            f"taxi_bonus_shock: generator returned {n_states_actual} states "
            f"but {resolved['n_states']} were expected."
        )

    # Wrap with bonus shock.
    wrapped = TaxiBonusShockWrapper(
        base_mdp=mdp_base,
        bonus_prob=float(resolved["bonus_prob"]),
        bonus_reward=float(resolved["bonus_reward"]),
        rng_seed=seed,
    )

    if time_augment:
        mdp_rl = make_time_augmented(
            mdp_base, horizon=int(resolved["horizon"])
        )
        assert isinstance(mdp_rl, DiscreteTimeAugmentedEnv)
    else:
        mdp_rl = mdp_base

    return wrapped, mdp_rl, resolved
