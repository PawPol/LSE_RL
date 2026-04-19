"""Phase IV-A operator-sensitive task factory module.

Provides six task families tuned for operator activation diagnostics,
plus a search grid registry and a dispatch function.

Design rules (spec section 4):
  - Mainline reward shocks: |reward| <= 3.0.
  - gamma_eval in {0.95, 0.97}.
  - Horizons: H = round(c / (1 - gamma_eval)) for c in {1.0, 1.5, 2.0}.
  - Event rates: 1%--15% target range.
  - Task selection uses ONLY classical pilot diagnostics.

Each factory returns ``(mdp_base, mdp_rl, resolved_cfg)`` where:
  - ``mdp_base`` is the unmodified FiniteMDP (severity=0 / baseline).
  - ``mdp_rl`` is the activation-tuned variant for the algorithm.
  - ``resolved_cfg`` echoes all effective parameters plus ``reward_bound``.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mushroom_rl.environments.finite_mdp import FiniteMDP
from mushroom_rl.environments.generators.simple_chain import generate_simple_chain
from mushroom_rl.environments.generators.grid_world import generate_grid_world
from mushroom_rl.environments.generators.taxi import generate_taxi

from experiments.weighted_lse_dp.tasks.stress_families import (
    make_chain_jackpot,
    make_chain_catastrophe,
    make_chain_sparse_long,
    make_grid_sparse_goal,
    TaxiBonusShockWrapper,
)
from experiments.weighted_lse_dp.tasks.hazard_wrappers import (
    GridHazardWrapper,
    select_hazard_states,
    build_hazard_mdp,
)
from experiments.weighted_lse_dp.tasks.nonstationary_wrappers import (
    make_chain_regime_shift,
)
from experiments.weighted_lse_dp.common.seeds import seed_everything

_GRIDS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "grids")

# Mainline reward cap (spec section 4.3 rule 1)
_MAINLINE_REWARD_CAP = 3.0


# ---------------------------------------------------------------------------
# Family 1: Sparse-credit chain
# ---------------------------------------------------------------------------

def make_p4_chain_sparse_credit(
    cfg: Dict[str, Any],
    state_n: int = 24,
    prob: float = 0.9,
    gamma: float = 0.97,
    horizon: int = 50,
    step_cost: float = 0.0,
) -> Tuple[FiniteMDP, FiniteMDP, Dict[str, Any]]:
    """Sparse-credit chain with activation-tuned parameters.

    Goal reward is fixed at +1.0.  Optional tiny step cost penalises
    wandering without reaching the goal.

    Parameters
    ----------
    cfg : dict
        Caller-supplied overrides (passed through to resolved_cfg).
    state_n : int
        Number of chain states.  Mainline: {18, 24, 30}.
    prob : float
        Transition success probability.
    gamma : float
        Discount factor.  Must be in {0.95, 0.97}.
    horizon : int
        Episode horizon.  Derived from gamma via H = round(c/(1-gamma)).
    step_cost : float
        Per-step cost in {0.0, -0.01, -0.02}.  Applied to all non-goal
        transitions.
    """
    mdp_base, mdp_rl, base_cfg = make_chain_sparse_long(
        cfg,
        state_n=state_n,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
    )

    # Apply step cost by modifying only the base MDP reward matrix.
    # mdp_rl is a DiscreteTimeAugmentedEnv and has no .r attribute.
    if step_cost != 0.0:
        r = mdp_base.r
        goal = state_n - 1
        for s in range(r.shape[0]):
            for a in range(r.shape[1]):
                for s_next in range(r.shape[2]):
                    if s_next != goal and mdp_base.p[s, a, s_next] > 0 and r[s, a, s_next] == 0.0:
                        r[s, a, s_next] = step_cost

    resolved = dict(
        family="chain_sparse_credit",
        state_n=state_n,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
        goal_reward=1.0,
        step_cost=step_cost,
        reward_bound=1.0,
    )
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Family 2: Jackpot chain
# ---------------------------------------------------------------------------

def make_p4_chain_jackpot(
    cfg: Dict[str, Any],
    state_n: int = 24,
    prob: float = 0.9,
    gamma: float = 0.97,
    horizon: int = 50,
    jackpot_reward: float = 2.0,
    jackpot_prob: float = 0.10,
    jackpot_state: Optional[int] = None,
    step_cost: float = 0.0,
) -> Tuple[FiniteMDP, FiniteMDP, Dict[str, Any]]:
    """Chain with occasional jackpot at an interior state.

    Parameters
    ----------
    jackpot_reward : float
        Mainline values: {1.5, 2.0, 3.0}.  Values > 3.0 trigger a warning
        and set ``"severe_variant": True`` in resolved_cfg.
    jackpot_prob : float
        Probability of jackpot firing per visit.  Mainline: {0.05, 0.10, 0.20}.
    jackpot_state : int or None
        State index for the jackpot.  Defaults to ``state_n // 2``.
    """
    if jackpot_state is None:
        jackpot_state = state_n // 2

    goal_reward = 1.0
    severe = False
    if abs(jackpot_reward) > _MAINLINE_REWARD_CAP:
        warnings.warn(
            f"jackpot_reward={jackpot_reward} exceeds mainline cap "
            f"({_MAINLINE_REWARD_CAP}).  Marking as severe_variant.",
            stacklevel=2,
        )
        severe = True

    mdp_base, mdp_rl, base_cfg = make_chain_jackpot(
        cfg,
        state_n=state_n,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
        jackpot_reward=jackpot_reward,
        jackpot_prob=jackpot_prob,
        jackpot_state=jackpot_state,
    )

    resolved = dict(
        family="chain_jackpot",
        state_n=state_n,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
        jackpot_reward=jackpot_reward,
        jackpot_prob=jackpot_prob,
        jackpot_state=jackpot_state,
        goal_reward=goal_reward,
        step_cost=step_cost,
        reward_bound=max(goal_reward, jackpot_reward),
    )
    if severe:
        resolved["severe_variant"] = True
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Family 3: Catastrophe chain
# ---------------------------------------------------------------------------

def make_p4_chain_catastrophe(
    cfg: Dict[str, Any],
    state_n: int = 24,
    prob: float = 0.9,
    gamma: float = 0.97,
    horizon: int = 50,
    catastrophe_reward: float = -2.0,
    risky_prob: float = 0.10,
    shortcut_jump: int = 4,
    step_cost: float = 0.0,
) -> Tuple[FiniteMDP, FiniteMDP, Dict[str, Any]]:
    """Chain with catastrophic shortcut.

    Parameters
    ----------
    catastrophe_reward : float
        Mainline values: {-1.5, -2.0, -3.0}.
    risky_prob : float
        Probability of the catastrophic shortcut firing.  {0.05, 0.10, 0.20}.
    shortcut_jump : int
        Number of states the shortcut skips.  {3, 4, 5}.
    """
    goal_reward = 1.0

    if abs(catastrophe_reward) > _MAINLINE_REWARD_CAP:
        warnings.warn(
            f"catastrophe_reward={catastrophe_reward} exceeds mainline cap "
            f"({_MAINLINE_REWARD_CAP}).  Consider using mainline values.",
            stacklevel=2,
        )

    # make_chain_catastrophe places risky_state at state_n//2 by default
    risky_state = state_n // 2
    mdp_base, mdp_rl, base_cfg = make_chain_catastrophe(
        cfg,
        state_n=state_n,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
        catastrophe_reward=catastrophe_reward,
        risky_prob=risky_prob,
        risky_state=risky_state,
        shortcut_jump=shortcut_jump,
    )

    resolved = dict(
        family="chain_catastrophe",
        state_n=state_n,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
        catastrophe_reward=catastrophe_reward,
        risky_prob=risky_prob,
        shortcut_jump=shortcut_jump,
        goal_reward=goal_reward,
        step_cost=step_cost,
        reward_bound=max(abs(catastrophe_reward), goal_reward),
    )
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Family 4: Grid hazard
# ---------------------------------------------------------------------------

def make_p4_grid_hazard(
    cfg: Dict[str, Any],
    n_rows: int = 5,
    n_cols: int = 5,
    prob: float = 0.9,
    gamma: float = 0.97,
    horizon: int = 50,
    hazard_reward: float = -1.5,
    hazard_prob: float = 0.15,
    detour_len: int = 3,
    step_cost: float = 0.0,
) -> Tuple[FiniteMDP, FiniteMDP, Dict[str, Any]]:
    """Grid world with hazard zones.

    Uses a 5x5 grid (25 navigable states, matching Phase III).
    Hazards are placed along the shortest path, forcing a detour
    of ``detour_len`` extra steps.

    Parameters
    ----------
    hazard_reward : float
        Negative reward for entering hazard.  Mainline: {-1.0, -1.5, -2.0}.
    hazard_prob : float
        Probability of hazard triggering on entry.  {0.10, 0.20, 0.30}.
    detour_len : int
        Extra steps to avoid hazard.  {2, 3, 4}.
    """
    grid_file = os.path.join(_GRIDS_DIR, "phase1_base_grid.txt")

    # Build base grid MDP
    pos_rew = 1.0
    neg_rew = 0.0  # no holes in base grid
    mdp_base = generate_grid_world(grid_file, prob, pos_rew, neg_rew,
                                   gamma=gamma, horizon=horizon)

    # Select hazard states
    hazard_states = select_hazard_states(
        grid_file, n_rows=n_rows, n_cols=n_cols, detour_len=detour_len,
    )

    # Build hazard-modified MDP (rewards baked into p/r matrices for DP)
    mdp_rl = build_hazard_mdp(
        mdp_base,
        hazard_states=hazard_states,
        hazard_prob=hazard_prob,
        hazard_reward=hazard_reward,
    )

    # Apply step cost if requested
    if step_cost != 0.0:
        r_base = mdp_base.r.copy()
        r_rl = mdp_rl.r.copy()
        n_states = r_base.shape[0]
        n_actions = r_base.shape[1]
        for s in range(n_states):
            for a in range(n_actions):
                for s_next in range(n_states):
                    if r_base[s, a, s_next] == 0.0 and mdp_base.p[s, a, s_next] > 0:
                        r_base[s, a, s_next] = step_cost
                    if r_rl[s, a, s_next] == 0.0 and mdp_rl.p[s, a, s_next] > 0:
                        r_rl[s, a, s_next] = step_cost
        mdp_base = FiniteMDP(mdp_base.p.copy(), r_base, mdp_base.mu,
                             gamma, horizon)
        mdp_rl = FiniteMDP(mdp_rl.p.copy(), r_rl, mdp_rl.mu,
                           gamma, horizon)

    resolved = dict(
        family="grid_hazard",
        n_rows=n_rows,
        n_cols=n_cols,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
        hazard_reward=hazard_reward,
        hazard_prob=hazard_prob,
        detour_len=detour_len,
        hazard_states=hazard_states,
        step_cost=step_cost,
        reward_bound=max(abs(hazard_reward), 1.0),
    )
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Family 5: Regime shift
# ---------------------------------------------------------------------------

def make_p4_regime_shift(
    cfg: Dict[str, Any],
    state_n: int = 24,
    prob: float = 0.9,
    gamma: float = 0.97,
    horizon: int = 50,
    change_at_episode: Optional[int] = None,
    step_cost: float = 0.0,
) -> Tuple[FiniteMDP, FiniteMDP, Dict[str, Any]]:
    """Regime-shift chain: goal flips from right to left.

    Parameters
    ----------
    change_at_episode : int or None
        Episode at which the goal flips.  Defaults to ``horizon // 2``.
    """
    if change_at_episode is None:
        change_at_episode = horizon // 2

    mdp_base, mdp_rl, base_cfg = make_chain_regime_shift(
        cfg,
        state_n=state_n,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
        change_at_episode=change_at_episode,
    )

    resolved = dict(
        family="regime_shift",
        state_n=state_n,
        prob=prob,
        gamma=gamma,
        horizon=horizon,
        change_at_episode=change_at_episode,
        step_cost=step_cost,
        reward_bound=1.0,
    )
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Family 6: Taxi bonus
# ---------------------------------------------------------------------------

def make_p4_taxi_bonus(
    cfg: Dict[str, Any],
    gamma: float = 0.97,
    horizon: int = 60,
    bonus_reward: float = 2.0,
    bonus_prob: float = 0.10,
    seed: int = 42,
) -> Tuple[FiniteMDP, FiniteMDP, Dict[str, Any]]:
    """Taxi with occasional bonus state.

    Parameters
    ----------
    bonus_reward : float
        Mainline: {1.5, 2.0, 3.0}.
    bonus_prob : float
        Fraction of states receiving bonus.  {0.05, 0.10, 0.15}.
    seed : int
        RNG seed for bonus state selection.

    Notes
    -----
    Marked ``appendix_only`` per spec section 4.5.6 (taxi is noisy).
    """
    taxi_grid = os.path.join(_GRIDS_DIR, "phase1_taxi_grid.txt")
    mdp_base = generate_taxi(taxi_grid, prob=0.9,
                             rew=(0, 1),  # matches phase1_taxi_grid.txt (2 pickup locations)
                             gamma=gamma, horizon=horizon)

    wrapper = TaxiBonusShockWrapper(
        mdp_base,
        bonus_reward=bonus_reward,
        bonus_prob=bonus_prob,
        rng_seed=seed,
    )
    mdp_rl = wrapper  # step-intercepting wrapper IS the RL environment

    resolved = dict(
        family="taxi_bonus",
        gamma=gamma,
        horizon=horizon,
        bonus_reward=bonus_reward,
        bonus_prob=bonus_prob,
        seed=seed,
        reward_bound=bonus_reward,
        appendix_only=True,
    )
    resolved.update({k: v for k, v in cfg.items() if k not in resolved})
    return mdp_base, mdp_rl, resolved


# ---------------------------------------------------------------------------
# Search grid registry
# ---------------------------------------------------------------------------

def get_search_grid() -> list[dict]:
    """Return the full Phase IV-A activation search parameter grid.

    Each entry is a config dict with keys: family, and family-specific params.
    Used by run_phase4_activation_search.py.
    """
    grid: list[dict] = []

    # chain_sparse_credit: state_n x gamma x step_cost
    for state_n in [18, 24, 30]:
        for gamma, horizons in [(0.95, [20, 30, 40]), (0.97, [33, 50, 67])]:
            for horizon in horizons:
                for step_cost in [0.0, -0.01]:
                    grid.append({
                        "family": "chain_sparse_credit",
                        "state_n": state_n, "gamma": gamma,
                        "horizon": horizon, "step_cost": step_cost,
                        "reward_bound": 1.0,
                    })

    # chain_jackpot: jackpot_reward x jackpot_prob x gamma
    for jackpot_reward in [1.5, 2.0, 3.0]:
        for jackpot_prob in [0.05, 0.10, 0.20]:
            for gamma, horizons in [(0.95, [20, 30]), (0.97, [33, 50])]:
                for horizon in horizons:
                    grid.append({
                        "family": "chain_jackpot",
                        "jackpot_reward": jackpot_reward,
                        "jackpot_prob": jackpot_prob,
                        "gamma": gamma, "horizon": horizon,
                        "state_n": 24,
                        "reward_bound": jackpot_reward,
                    })

    # chain_catastrophe: catastrophe_reward x risky_prob x gamma
    for cat_reward in [-1.5, -2.0, -3.0]:
        for risky_prob in [0.05, 0.10, 0.20]:
            for gamma, horizons in [(0.95, [20, 30]), (0.97, [33, 50])]:
                for horizon in horizons:
                    grid.append({
                        "family": "chain_catastrophe",
                        "catastrophe_reward": cat_reward,
                        "risky_prob": risky_prob,
                        "gamma": gamma, "horizon": horizon,
                        "state_n": 24,
                        "reward_bound": max(abs(cat_reward), 1.0),
                    })

    # grid_hazard: hazard_reward x hazard_prob x gamma
    for haz_reward in [-1.0, -1.5, -2.0]:
        for haz_prob in [0.10, 0.20, 0.30]:
            for gamma, horizons in [(0.95, [20, 30]), (0.97, [33, 50])]:
                for horizon in horizons:
                    grid.append({
                        "family": "grid_hazard",
                        "hazard_reward": haz_reward,
                        "hazard_prob": haz_prob,
                        "gamma": gamma, "horizon": horizon,
                        "n_rows": 5, "n_cols": 5,
                        "reward_bound": max(abs(haz_reward), 1.0),
                    })

    # regime_shift: gamma only
    for gamma, horizons in [(0.95, [20, 30, 40]), (0.97, [33, 50, 67])]:
        for horizon in horizons:
            grid.append({
                "family": "regime_shift",
                "gamma": gamma, "horizon": horizon,
                "state_n": 24,
                "reward_bound": 1.0,
            })

    # taxi_bonus: appendix-only
    for bonus_reward in [1.5, 2.0, 3.0]:
        for bonus_prob in [0.05, 0.10, 0.15]:
            for gamma, horizon in [(0.97, 40), (0.97, 60)]:
                grid.append({
                    "family": "taxi_bonus",
                    "bonus_reward": bonus_reward,
                    "bonus_prob": bonus_prob,
                    "gamma": gamma, "horizon": horizon,
                    "reward_bound": bonus_reward,
                    "appendix_only": True,
                })

    return grid


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

_FAMILY_DISPATCH = {
    "chain_sparse_credit": make_p4_chain_sparse_credit,
    "chain_jackpot": make_p4_chain_jackpot,
    "chain_catastrophe": make_p4_chain_catastrophe,
    "grid_hazard": make_p4_grid_hazard,
    "regime_shift": make_p4_regime_shift,
    "taxi_bonus": make_p4_taxi_bonus,
}


def build_phase4_task(
    cfg: dict,
    seed: int = 42,
) -> Tuple[FiniteMDP, FiniteMDP, Dict[str, Any]]:
    """Dispatch to the correct Phase IV-A factory based on ``cfg["family"]``.

    Parameters
    ----------
    cfg : dict
        Must contain ``"family"`` key plus family-specific parameters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mdp_base : FiniteMDP
        The unmodified baseline MDP.
    mdp_rl : FiniteMDP
        The activation-tuned MDP for algorithm evaluation.
    resolved_cfg : dict
        All effective parameters.
    """
    family = cfg.get("family")
    if family is None:
        raise ValueError("cfg must contain a 'family' key.")
    if family not in _FAMILY_DISPATCH:
        raise ValueError(
            f"Unknown family '{family}'. "
            f"Available: {sorted(_FAMILY_DISPATCH.keys())}"
        )

    seed_everything(seed)

    # Extract family-specific kwargs (everything except 'family' and
    # meta-keys like 'reward_bound', 'appendix_only')
    factory = _FAMILY_DISPATCH[family]
    meta_keys = {"family", "reward_bound", "appendix_only", "severe_variant"}
    kwargs = {k: v for k, v in cfg.items() if k not in meta_keys}

    # Pass seed to taxi factory
    if family == "taxi_bonus":
        kwargs.setdefault("seed", seed)

    return factory(kwargs, **kwargs)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "make_p4_chain_sparse_credit",
    "make_p4_chain_jackpot",
    "make_p4_chain_catastrophe",
    "make_p4_grid_hazard",
    "make_p4_regime_shift",
    "make_p4_taxi_bonus",
    "get_search_grid",
    "build_phase4_task",
]
