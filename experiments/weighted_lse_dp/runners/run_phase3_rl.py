"""Phase III RL runner: safe weighted-LSE online tabular RL runs.

For each (task, safe_algorithm, seed) in the Phase III paper suite, this
runner:

1. Loads the stress task via the matching factory.
2. Loads the calibrated BetaSchedule from the per-task schedule.json.
3. Creates the safe RL agent (SafeQLearning or SafeExpectedSARSA) with
   fixed epsilon-greedy exploration (eps=0.1) and constant learning rate.
4. Attaches a :class:`SafeTransitionLogger` as ``callback_step`` on Core,
   and an event-detecting wrapper for stress tasks.
5. Runs ``Core.learn(...)`` in checkpoint segments, evaluating with
   :class:`RLEvaluator` at each checkpoint.
6. After training: builds transitions payload (base + safe), computes
   target stats, tail-risk metrics, adaptation metrics, safe aggregate
   stats, and flushes everything via :class:`RunWriter`.
7. Writes safe calibration provenance (schedule lineage).

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase3_rl.py \\
        [--config PATH]            # default: experiments/weighted_lse_dp/configs/phase3/paper_suite.json
        [--task TASK | all]        # filter to one task, or 'all' for all tasks
        [--algorithm ALG]          # SafeQLearning or SafeExpectedSARSA
        [--seed SEED]              # filter to one seed
        [--out-root PATH]          # default: results/weighted_lse_dp
        [--schedule-dir PATH]      # default: results/weighted_lse_dp/phase3/calibration
        [--dry-run]                # print plan, no execution

Spec anchors: Phase III spec S6.2, S7.1, S13.2.

Task 34 compliance: event detection thresholds are read from EXPLICIT
config keys (jackpot_threshold, catastrophe_threshold, hazard_threshold).
If a required key is missing, the runner raises KeyError with a clear
message. No silent .get() defaults for event thresholds.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping (spec S6.2)
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from experiments.weighted_lse_dp.common.callbacks import (  # noqa: E402
    AdaptationMetricsLogger,
    EventTransitionLogger,
    RLEvaluator,
    SafeTransitionLogger,
    TailRiskLogger,
    TargetStatsLogger,
    TransitionLogger,
)
from experiments.weighted_lse_dp.common.calibration import (  # noqa: E402
    aggregate_calibration_stats,
)
from experiments.weighted_lse_dp.common.manifests import (  # noqa: E402
    write_safe_provenance,
)
from experiments.weighted_lse_dp.common.schemas import (  # noqa: E402
    RunWriter,
    aggregate_safe_stats,
)
from experiments.weighted_lse_dp.common.seeds import (  # noqa: E402
    seed_everything,
)
from experiments.weighted_lse_dp.common.io import (  # noqa: E402
    make_npz_schema,
    save_npz_with_schema,
)

# -- Stress task factories ---------------------------------------------------
from experiments.weighted_lse_dp.tasks.stress_families import (  # noqa: E402
    make_chain_sparse_long,
    make_chain_jackpot,
    make_chain_catastrophe,
    make_grid_sparse_goal,
    make_taxi_bonus_shock,
)
from experiments.weighted_lse_dp.tasks.nonstationary_wrappers import (  # noqa: E402
    make_chain_regime_shift,
    make_grid_regime_shift,
)
from experiments.weighted_lse_dp.tasks.hazard_wrappers import (  # noqa: E402
    make_grid_hazard,
)
from mushroom_rl.environments.time_augmented_env import (  # noqa: E402
    DiscreteTimeAugmentedEnv,
)

__all__ = ["main", "run_single", "build_plan"]

# ---------------------------------------------------------------------------
# Algorithm registry (safe TD only)
# ---------------------------------------------------------------------------
# Deferred import: we only need these at run time, not at plan/dry-run time.
# The registry maps config names to module-level lazy references.

SAFE_RL_ALGORITHMS: dict[str, str] = {
    "SafeQLearning": "SafeQLearning",
    "SafeExpectedSARSA": "SafeExpectedSARSA",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = (
    "experiments/weighted_lse_dp/configs/phase3/paper_suite.json"
)

_DEFAULT_OUT_ROOT = "results/weighted_lse_dp"

_DEFAULT_SCHEDULE_DIR = "results/weighted_lse_dp/phase3/calibration"

#: Map task name -> factory function.
#: Factories return (mdp_or_wrapper, mdp_rl, resolved_cfg) -- 3-tuples.
_TASK_FACTORIES: dict[str, Any] = {
    "chain_sparse_long": make_chain_sparse_long,
    "chain_jackpot": make_chain_jackpot,
    "chain_catastrophe": make_chain_catastrophe,
    "chain_regime_shift": make_chain_regime_shift,
    "grid_sparse_goal": make_grid_sparse_goal,
    "grid_hazard": make_grid_hazard,
    "grid_regime_shift": make_grid_regime_shift,
    "taxi_bonus_shock": make_taxi_bonus_shock,
}

#: Map task name -> n_base (number of un-augmented base states).
#: Mirrors Phase II runner exactly.
_N_BASE: dict[str, int] = {
    "chain_sparse_long": 60,
    "chain_jackpot": 26,        # 25 + 1 absorbing terminal
    "chain_catastrophe": 26,    # 25 + 1 absorbing terminal
    "chain_regime_shift": 25,
    "grid_sparse_goal": 49,     # 7x7 grid
    "grid_hazard": 25,
    "grid_regime_shift": 25,
    "taxi_bonus_shock": 44,
}

#: Tasks whose factories return a stress wrapper (not a bare MDP).
_WRAPPER_TASKS_RL: frozenset[str] = frozenset({
    "chain_regime_shift",
    "grid_regime_shift",
    "grid_hazard",
    "taxi_bonus_shock",
})

#: Fixed training hyperparameters.
_EPSILON = 0.1
_LEARNING_RATE = 0.1

# ---------------------------------------------------------------------------
# Task 34: Explicit event detection threshold extraction
# ---------------------------------------------------------------------------
#
# LESSON (2026-04-17): Never use task_cfg.get("key", default) for event
# detection thresholds. If a config key is missing, FAIL LOUDLY.
# The config file is the contract.


def _require_key(task_cfg: dict, key: str, task_name: str) -> Any:
    """Extract a required config key, raising KeyError if missing.

    This enforces Task 34: no silent .get() defaults for event detection
    thresholds. Every threshold must be explicitly specified in the
    config file.
    """
    if key not in task_cfg:
        raise KeyError(
            f"task '{task_name}' config missing required key '{key}'. "
            f"Add it to paper_suite.json. "
            f"Available keys: {sorted(task_cfg.keys())}"
        )
    return task_cfg[key]


def _get_event_thresholds(
    task_name: str,
    task_cfg: dict[str, Any],
    stress_type: str | None,
) -> dict[str, float | int | None]:
    """Extract event detection thresholds from config.

    For each stress_type, the required threshold keys are:
    - "jackpot": requires "jackpot_threshold"
    - "catastrophe": requires "catastrophe_threshold" and "risky_state"
    - "hazard": requires "hazard_threshold"
    - "regime_shift": no reward threshold needed (uses wrapper attribute)
    - None: no thresholds needed

    Returns a dict with standardised keys for _SafeAutoEventLogger.
    """
    result: dict[str, float | int | None] = {
        "jackpot_reward_threshold": None,
        "catastrophe_reward_threshold": None,
        "hazard_reward_threshold": None,
        "risky_state": None,
    }

    if stress_type == "jackpot":
        result["jackpot_reward_threshold"] = float(
            _require_key(task_cfg, "jackpot_threshold", task_name)
        )

    elif stress_type == "catastrophe":
        result["catastrophe_reward_threshold"] = float(
            _require_key(task_cfg, "catastrophe_threshold", task_name)
        )
        result["risky_state"] = int(
            _require_key(task_cfg, "risky_state", task_name)
        )

    elif stress_type == "hazard":
        result["hazard_reward_threshold"] = float(
            _require_key(task_cfg, "hazard_threshold", task_name)
        )

    # regime_shift and None: no reward thresholds required.
    return result


# ---------------------------------------------------------------------------
# Event-detecting callback wrapper for safe agents
# ---------------------------------------------------------------------------


class _SafeAutoEventLogger(EventTransitionLogger):
    """EventTransitionLogger + SafeTransitionLogger fields for safe agents.

    Combines:
    - EventTransitionLogger: binary event flags (jackpot, catastrophe, etc.)
    - SafeTransitionLogger logic: safe_stage, safe_beta_used, safe_rho, etc.

    Since we cannot inherit from both EventTransitionLogger and
    SafeTransitionLogger (diamond with TransitionLogger), we compose:
    inherit from EventTransitionLogger and manually add the safe field
    accumulation from SafeTransitionLogger.

    Parameters
    ----------
    agent:
        Safe RL agent with ``Q`` and ``swc`` attributes.
    n_base:
        Number of base (un-augmented) states.
    gamma:
        Discount factor.
    stress_type:
        One of "jackpot", "catastrophe", "regime_shift", "hazard", or None.
    wrapper:
        Environment wrapper (for regime-shift detection).
    jackpot_reward_threshold:
        Reward above this triggers jackpot flag.
    catastrophe_reward_threshold:
        Reward below this (and absorbing) triggers catastrophe flag.
    hazard_reward_threshold:
        Reward below this triggers hazard flag.
    risky_state:
        Base state id for shortcut detection in catastrophe tasks.
    """

    def __init__(
        self,
        agent: object,
        *,
        n_base: int,
        gamma: float,
        stress_type: str | None,
        wrapper: object | None = None,
        jackpot_reward_threshold: float | None = None,
        catastrophe_reward_threshold: float | None = None,
        hazard_reward_threshold: float | None = None,
        risky_state: int | None = None,
    ) -> None:
        super().__init__(agent, n_base=n_base, gamma=gamma)
        self._stress_type = stress_type
        self._wrapper = wrapper
        self._jackpot_thr = jackpot_reward_threshold
        self._catastrophe_thr = catastrophe_reward_threshold
        self._hazard_thr = hazard_reward_threshold
        self._risky_state = risky_state

        # Safe-specific accumulation lists (mirrors SafeTransitionLogger).
        self._safe_stage: list[int] = []
        self._safe_beta_raw: list[float] = []
        self._safe_beta_cap: list[float] = []
        self._safe_beta_used: list[float] = []
        self._safe_clip_active: list[bool] = []
        self._safe_rho: list[float] = []
        self._safe_effective_discount: list[float] = []
        self._safe_target: list[float] = []
        self._safe_margin: list[float] = []
        self._safe_td_error: list[float] = []

    def __call__(self, sample: tuple) -> None:
        """callback_step hook: detect events, set flags, record base + event fields.

        Safe fields are logged in :meth:`after_fit`, which must be
        registered as a ``callbacks_fit`` entry on Core so it fires
        after ``agent.fit(dataset)`` -- that is when
        ``agent.swc.last_*`` fields reflect the current update.
        """
        reward = float(sample[2])
        absorbing = bool(sample[4])

        # -- Event detection (same logic as Phase II _AutoEventLogger) --
        if self._stress_type == "jackpot":
            if self._jackpot_thr is not None and reward > self._jackpot_thr:
                self.mark_jackpot()

        elif self._stress_type == "catastrophe":
            if self._catastrophe_thr is not None and reward < self._catastrophe_thr and absorbing:
                self.mark_catastrophe()
            if self._risky_state is not None:
                base_state = int(sample[0][0]) % self._n_base
                action = int(sample[1][0])
                if base_state == self._risky_state and action == 0:
                    self.mark_shortcut_taken()

        elif self._stress_type == "regime_shift":
            if self._wrapper is not None and hasattr(self._wrapper, "post_change"):
                self.mark_regime_post_change(self._wrapper.post_change)

        elif self._stress_type == "hazard":
            if self._hazard_thr is not None and reward < self._hazard_thr:
                self.mark_hazard_hit()

        # -- Delegate to EventTransitionLogger (appends base + event) --
        super().__call__(sample)

    def after_fit(self, dataset: list) -> None:
        """callbacks_fit hook: log safe fields after agent.fit() runs.

        Must be passed as a ``callbacks_fit`` entry to
        :class:`mushroom_rl.core.Core` so it fires after each
        ``agent.fit(dataset)`` call -- that is when
        ``agent.swc.last_*`` fields reflect the most recent update.

        For evaluation (no fit calls), this method is never invoked and
        safe fields are intentionally not logged.
        """
        swc = self._agent.swc  # type: ignore[attr-defined]
        reward = self._reward[-1]
        v_next = self._v_next_beta0[-1]

        self._safe_stage.append(int(swc.last_stage))
        self._safe_beta_raw.append(float(np.asarray(swc.last_beta_raw).item()))
        self._safe_beta_cap.append(float(np.asarray(swc.last_beta_cap).item()))
        self._safe_beta_used.append(float(np.asarray(swc.last_beta_used).item()))
        self._safe_clip_active.append(bool(swc.last_clip_active))
        self._safe_rho.append(float(np.asarray(swc.last_rho).item()))
        self._safe_effective_discount.append(
            float(np.asarray(swc.last_effective_discount).item())
        )

        safe_target = float(np.asarray(swc.last_target).item())
        self._safe_target.append(safe_target)
        # R6-1: read the margin from swc.last_margin (the exact v_next the
        # operator used), not from v_next_beta0 which is always the greedy
        # max-Q bootstrap — wrong for SafeExpectedSARSA.
        self._safe_margin.append(float(np.asarray(swc.last_margin).item()))

        q_current = self._q_current_beta0[-1]
        self._safe_td_error.append(safe_target - q_current)

    def build_safe_payload(self) -> dict[str, np.ndarray]:
        """Return safe-specific transition arrays.

        Returns empty arrays if no safe fields were logged (e.g. during
        evaluation where ``after_fit`` is never called).
        """
        if len(self._safe_stage) == 0:
            empty_f = np.array([], dtype=np.float64)
            empty_i = np.array([], dtype=np.int64)
            empty_b = np.array([], dtype=bool)
            return {
                "safe_stage": empty_i,
                "safe_beta_raw": empty_f,
                "safe_beta_cap": empty_f,
                "safe_beta_used": empty_f,
                "safe_clip_active": empty_b,
                "safe_rho": empty_f,
                "safe_effective_discount": empty_f,
                "safe_target": empty_f,
                "safe_margin": empty_f,
                "safe_td_error": empty_f,
            }
        return {
            "safe_stage": np.array(self._safe_stage, dtype=np.int64),
            "safe_beta_raw": np.array(self._safe_beta_raw, dtype=np.float64),
            "safe_beta_cap": np.array(self._safe_beta_cap, dtype=np.float64),
            "safe_beta_used": np.array(self._safe_beta_used, dtype=np.float64),
            "safe_clip_active": np.array(self._safe_clip_active, dtype=bool),
            "safe_rho": np.array(self._safe_rho, dtype=np.float64),
            "safe_effective_discount": np.array(
                self._safe_effective_discount, dtype=np.float64
            ),
            "safe_target": np.array(self._safe_target, dtype=np.float64),
            "safe_margin": np.array(self._safe_margin, dtype=np.float64),
            "safe_td_error": np.array(self._safe_td_error, dtype=np.float64),
        }

    def build_payload(self) -> dict[str, np.ndarray]:
        """Return the full transitions payload: base + event + safe merged.

        Safe fields are only included when ``after_fit`` was called at
        least once (i.e. during training). During evaluation the
        payload contains only base + event fields.
        """
        payload = super().build_payload()  # base + event flags
        if len(self._safe_stage) > 0:
            payload.update(self.build_safe_payload())
        return payload

    def reset(self) -> None:
        """Clear all accumulated data including safe-specific fields."""
        super().reset()
        self._safe_stage.clear()
        self._safe_beta_raw.clear()
        self._safe_beta_cap.clear()
        self._safe_beta_used.clear()
        self._safe_clip_active.clear()
        self._safe_rho.clear()
        self._safe_effective_discount.clear()
        self._safe_target.clear()
        self._safe_margin.clear()
        self._safe_td_error.clear()


class _PlainSafeLogger(SafeTransitionLogger):
    """SafeTransitionLogger for tasks without stress events.

    Used for chain_sparse_long and grid_sparse_goal which have no
    stress_type and therefore no event detection.
    """
    pass  # SafeTransitionLogger already has everything we need.


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------


def build_plan(
    config: dict[str, Any],
    *,
    task_filter: str | None = None,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Build the list of (task, algorithm, seed) runs from a config dict.

    Parameters
    ----------
    config:
        Loaded paper_suite.json config.
    task_filter:
        When set, include only this task. "all" means no filter.
    algorithm_filter:
        When set, include only this algorithm.
    seed_filter:
        When set, include only this seed.

    Returns
    -------
    list[dict]
        Each entry has keys: task, algorithm, seed, task_config.
    """
    tasks_cfg = config.get("tasks", {})
    seeds_default = tuple(config.get("seeds", [11, 29, 47]))
    chain_seeds = tuple(config.get("chain_seeds", [11, 29, 47, 67, 83]))

    plan: list[dict[str, Any]] = []

    for task_name, task_cfg in sorted(tasks_cfg.items()):
        if task_filter is not None and task_filter != "all" and task_name != task_filter:
            continue
        if task_name not in _TASK_FACTORIES:
            continue

        safe_rl_algorithms = task_cfg.get("safe_rl_algorithms", [])

        is_chain = task_name.startswith("chain_")
        base_seeds = chain_seeds if is_chain else seeds_default

        for algo_name in safe_rl_algorithms:
            if algorithm_filter is not None and algo_name != algorithm_filter:
                continue
            if algo_name not in SAFE_RL_ALGORITHMS:
                continue

            seeds = base_seeds
            if seed_filter is not None:
                if seed_filter in seeds:
                    seeds = (seed_filter,)
                else:
                    seeds = (seed_filter,)

            for seed in seeds:
                plan.append({
                    "task": task_name,
                    "algorithm": algo_name,
                    "seed": seed,
                    "task_config": task_cfg,
                })

    return plan


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------


def _make_safe_agent(
    algo_name: str,
    mdp_info: Any,
    schedule: Any,
    n_base: int,
    *,
    epsilon: float = _EPSILON,
    learning_rate: float = _LEARNING_RATE,
) -> Any:
    """Construct a safe MushroomRL tabular RL agent.

    Deferred import so the module can be imported without MushroomRL on
    sys.path (e.g. for --dry-run).
    """
    from mushroom_rl.algorithms.value.td import SafeQLearning, SafeExpectedSARSA
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import Parameter

    pi = EpsGreedy(epsilon=Parameter(value=epsilon))

    if algo_name == "SafeQLearning":
        agent = SafeQLearning(
            mdp_info, pi, schedule, n_base,
            learning_rate=Parameter(value=learning_rate),
        )
    elif algo_name == "SafeExpectedSARSA":
        agent = SafeExpectedSARSA(
            mdp_info, pi, schedule, n_base,
            learning_rate=Parameter(value=learning_rate),
        )
    else:
        raise ValueError(f"Unsupported safe RL algorithm: {algo_name!r}")

    # Bind the Q table to the policy.
    pi.set_q(agent.Q)
    return agent


def _call_factory(
    task: str,
    task_config: dict[str, Any],
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Call the appropriate factory for *task* and return (wrapper_or_mdp, mdp_rl, resolved_cfg).

    Mirrors Phase II runner exactly.
    """
    factory = _TASK_FACTORIES[task]
    cfg = dict(task_config)

    if task == "chain_sparse_long":
        return factory(
            cfg,
            state_n=int(cfg.get("state_n", 60)),
            prob=float(cfg.get("prob", 0.9)),
            gamma=float(cfg.get("gamma", 0.99)),
            horizon=int(cfg.get("horizon", 120)),
        )

    elif task == "chain_jackpot":
        return factory(
            cfg,
            state_n=int(cfg.get("state_n", 25)),
            prob=float(cfg.get("prob", 0.9)),
            gamma=float(cfg.get("gamma", 0.99)),
            horizon=int(cfg.get("horizon", 60)),
            jackpot_state=int(cfg.get("jackpot_state", 20)),
            jackpot_prob=float(cfg.get("jackpot_prob", 0.05)),
            jackpot_reward=float(cfg.get("jackpot_reward", 10.0)),
            jackpot_terminates=bool(cfg.get("jackpot_terminates", True)),
        )

    elif task == "chain_catastrophe":
        return factory(
            cfg,
            state_n=int(cfg.get("state_n", 25)),
            prob=float(cfg.get("prob", 0.9)),
            gamma=float(cfg.get("gamma", 0.99)),
            horizon=int(cfg.get("horizon", 60)),
            risky_state=int(cfg.get("risky_state", 15)),
            risky_prob=float(cfg.get("risky_prob", 0.05)),
            catastrophe_reward=float(cfg.get("catastrophe_reward", -10.0)),
            shortcut_jump=int(cfg.get("shortcut_jump", 5)),
        )

    elif task == "chain_regime_shift":
        return factory(
            cfg,
            state_n=int(cfg.get("state_n", 25)),
            prob=float(cfg.get("prob", 0.9)),
            gamma=float(cfg.get("gamma", 0.99)),
            horizon=int(cfg.get("horizon", 60)),
            change_at_episode=int(cfg.get("change_at_episode", 500)),
            shift_type=str(cfg.get("shift_type", "goal_flip")),
            post_prob=float(cfg.get("post_prob", 0.7)),
        )

    elif task == "grid_sparse_goal":
        return factory(
            cfg,
            grid_file=str(cfg.get(
                "grid_file",
                "experiments/weighted_lse_dp/assets/grids/phase1_base_grid.txt",
            )),
            prob=float(cfg.get("prob", 0.9)),
            gamma=float(cfg.get("gamma", 0.99)),
            horizon=int(cfg.get("horizon", 80)),
            goal_reward=float(cfg.get("goal_reward", 1.0)),
            seed=seed,
        )

    elif task == "grid_hazard":
        hazard_states = cfg.get("hazard_states", [12])
        return factory(
            cfg,
            prob=float(cfg.get("prob", 0.9)),
            gamma=float(cfg.get("gamma", 0.99)),
            horizon=int(cfg.get("horizon", 80)),
            hazard_states=[int(s) for s in hazard_states],
            hazard_prob=float(cfg.get("hazard_prob", 0.10)),
            hazard_reward=float(cfg.get("hazard_reward", -5.0)),
            hazard_terminates=bool(cfg.get("hazard_terminates", False)),
            seed=seed,
        )

    elif task == "grid_regime_shift":
        return factory(
            cfg,
            change_at_episode=int(cfg.get("change_at_episode", 200)),
            shift_type=str(cfg.get("shift_type", "goal_move")),
        )

    elif task == "taxi_bonus_shock":
        return factory(
            cfg,
            grid_file=str(cfg.get(
                "grid_file",
                "experiments/weighted_lse_dp/assets/grids/phase1_taxi_grid.txt",
            )),
            prob=float(cfg.get("prob", 0.9)),
            gamma=float(cfg.get("gamma", 0.99)),
            horizon=int(cfg.get("horizon", 120)),
            bonus_prob=float(cfg.get("bonus_prob", 0.05)),
            bonus_reward=float(cfg.get("bonus_reward", 5.0)),
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown task: {task!r}")


def _compute_episode_returns(
    transitions_payload: dict[str, np.ndarray],
    gamma: float,
) -> np.ndarray:
    """Compute per-episode discounted returns from the transitions payload."""
    ep_idx = transitions_payload["episode_index"]
    rewards = transitions_payload["reward"]
    t_stages = transitions_payload["t"]

    n_episodes = int(ep_idx.max()) + 1 if len(ep_idx) > 0 else 0
    returns = np.zeros(n_episodes, dtype=np.float64)

    for i in range(len(ep_idx)):
        e = int(ep_idx[i])
        t = int(t_stages[i])
        returns[e] += (gamma ** t) * float(rewards[i])

    return returns


def _compute_episode_event_flags(
    transitions_payload: dict[str, np.ndarray],
    event_key: str,
) -> np.ndarray:
    """Compute per-episode boolean event flags."""
    ep_idx = transitions_payload["episode_index"]
    events = transitions_payload.get(event_key, np.zeros(len(ep_idx), dtype=bool))

    n_episodes = int(ep_idx.max()) + 1 if len(ep_idx) > 0 else 0
    flags = np.zeros(n_episodes, dtype=bool)

    for i in range(len(ep_idx)):
        if events[i]:
            flags[int(ep_idx[i])] = True

    return flags


def run_single(
    task: str,
    algorithm: str,
    seed: int,
    task_config: dict[str, Any],
    *,
    out_root: Path,
    schedule_dir: Path,
    suite: str = "paper_suite",
) -> dict[str, Any]:
    """Execute one (task, safe_algorithm, seed) run.

    Parameters
    ----------
    task:
        Task identifier (e.g. "chain_catastrophe", "grid_hazard").
    algorithm:
        Safe algorithm identifier ("SafeQLearning" or "SafeExpectedSARSA").
    seed:
        Integer seed for reproducibility.
    task_config:
        Per-task config block from the suite JSON.
    out_root:
        Base output directory.
    schedule_dir:
        Directory containing per-task schedule.json files.
        Expected layout: schedule_dir/<task>/schedule.json.
    suite:
        Suite name for RunWriter path construction.

    Returns
    -------
    dict
        Summary dict with keys: task, algorithm, seed, passed, wall_s,
        run_dir, summary, and optionally error/traceback.
    """
    from mushroom_rl.core import Core  # noqa: E402
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import BetaSchedule  # noqa: E402

    t_start = time.perf_counter()

    # -- Resolve parameters from task config --------------------------------
    train_steps = int(task_config["train_steps"])
    checkpoint_every = int(task_config["checkpoint_every"])
    eval_episodes_checkpoint = int(task_config.get("eval_episodes_checkpoint", 50))
    success_threshold = float(task_config["success_threshold"])
    gamma = float(task_config["gamma"])
    horizon = int(task_config["horizon"])
    stress_type: str | None = task_config.get("stress_type", None)

    # n_base: preliminary value from registry; validated against the
    # environment after construction (R6-2).
    n_base = _N_BASE[task]

    # -- Read hyperparameter overrides (used by ablation runner) -----------
    _eps_override = task_config.get("epsilon_override")
    _lr_mult = task_config.get("lr_multiplier", 1.0)
    effective_epsilon: float = (
        float(_eps_override) if _eps_override is not None else _EPSILON
    )
    effective_lr: float = _LEARNING_RATE * float(_lr_mult)

    # -- Load schedule -------------------------------------------------------
    # Honour per-task schedule_file override from the suite config (used by
    # ablation suites); fall back to the canonical default path.
    _schedule_file_override = task_config.get("schedule_file")
    if _schedule_file_override is not None:
        schedule_path = Path(_schedule_file_override)
    else:
        schedule_path = schedule_dir / task / "schedule.json"
    if not schedule_path.is_file():
        raise FileNotFoundError(
            f"Schedule not found at {schedule_path}. "
            f"Run calibration first or pass --schedule-dir."
        )
    schedule = BetaSchedule.from_file(schedule_path)

    if schedule.T != horizon:
        raise ValueError(
            f"Schedule T={schedule.T} does not match task horizon={horizon} "
            f"for task={task!r}. Regenerate the schedule or check the config."
        )

    # -- Task 34: Extract event detection thresholds EXPLICITLY --------------
    event_thresholds = _get_event_thresholds(task, task_config, stress_type)

    # -- Build resolved config for run.json ---------------------------------
    resolved_config: dict[str, Any] = {
        "task": task,
        "algorithm": algorithm,
        "seed": seed,
        "gamma": gamma,
        "horizon": horizon,
        "n_base": n_base,
        "train_steps": train_steps,
        "checkpoint_every": checkpoint_every,
        "eval_episodes_checkpoint": eval_episodes_checkpoint,
        "success_threshold": success_threshold,
        "epsilon": effective_epsilon,
        "learning_rate": effective_lr,
        "stress_type": stress_type,
        "schedule_path": str(schedule_path),
        "schedule_sign": schedule.sign,
        "schedule_T": schedule.T,
        "event_thresholds": {
            k: v for k, v in event_thresholds.items() if v is not None
        },
        "task_config": task_config,
    }

    # -- Seed everything ----------------------------------------------------
    seed_everything(seed)

    # -- Create environment -------------------------------------------------
    wrapper_or_mdp, mdp_rl, factory_cfg = _call_factory(task, task_config, seed)
    resolved_config["factory_config"] = factory_cfg

    # For wrapper-backed tasks, train/eval on the stressed wrapper.
    if task in _WRAPPER_TASKS_RL:
        mdp_rl = DiscreteTimeAugmentedEnv(wrapper_or_mdp, horizon=horizon)

    # Propagate gamma into the RL env.
    mdp_rl.info.gamma = gamma

    # R6-2: derive n_base from the augmented environment (observation_space.n
    # = T * n_base for all DiscreteTimeAugmented envs) and validate against
    # the registry to catch any future task-parameter drift early.
    n_base_derived = mdp_rl.info.observation_space.n // horizon
    if n_base_derived != n_base:
        raise ValueError(
            f"n_base registry mismatch for {task!r}: _N_BASE={n_base}, "
            f"env-derived={n_base_derived}. Update _N_BASE or check factory."
        )
    n_base = n_base_derived
    resolved_config["n_base"] = n_base  # update after env-derived value

    # -- Create safe agent ---------------------------------------------------
    agent = _make_safe_agent(
        algorithm, mdp_rl.info, schedule, n_base,
        epsilon=effective_epsilon,
        learning_rate=effective_lr,
    )

    # -- Create RunWriter ---------------------------------------------------
    rw = RunWriter.create(
        base=out_root,
        phase="phase3",
        suite=suite,
        task=task,
        algorithm=algorithm,
        seed=seed,
        config=resolved_config,
        storage_mode="rl_online",
    )

    # -- Create callbacks ---------------------------------------------------
    if stress_type is not None:
        logger: TransitionLogger = _SafeAutoEventLogger(
            agent,
            n_base=n_base,
            gamma=gamma,
            stress_type=stress_type,
            wrapper=wrapper_or_mdp,
            jackpot_reward_threshold=event_thresholds["jackpot_reward_threshold"],
            catastrophe_reward_threshold=event_thresholds["catastrophe_reward_threshold"],
            hazard_reward_threshold=event_thresholds["hazard_reward_threshold"],
            risky_state=event_thresholds["risky_state"],
        )
    else:
        # Tasks without stress events: use plain SafeTransitionLogger.
        logger = _PlainSafeLogger(agent, n_base=n_base, gamma=gamma)

    evaluator = RLEvaluator(
        agent=agent,
        env=mdp_rl,
        run_writer=rw,
        n_eval_episodes=eval_episodes_checkpoint,
        success_threshold=success_threshold,
        gamma=gamma,
    )

    # -- Create Core --------------------------------------------------------
    # Safe fields are logged in after_fit (callbacks_fit), not callback_step,
    # because Core fires callback_step BEFORE agent.fit() — so swc.last_*
    # would be stale by one transition if read in callback_step.
    core = Core(agent, mdp_rl, callback_step=logger,
                callbacks_fit=[logger.after_fit])

    # -- Training loop with checkpoints ------------------------------------
    print(
        f"[phase3_rl] {task}/{algorithm}/seed_{seed}: "
        f"train_steps={train_steps}, checkpoint_every={checkpoint_every}"
    )

    with rw.timer.phase("train"):
        steps_done = 0
        for ckpt_start in range(0, train_steps, checkpoint_every):
            n_steps_this_segment = min(checkpoint_every, train_steps - steps_done)
            core.learn(
                n_steps=n_steps_this_segment,
                n_steps_per_fit=1,
                quiet=True,
            )
            steps_done += n_steps_this_segment

            with rw.timer.phase("eval"):
                eval_result = evaluator.evaluate(steps=steps_done)

            print(
                f"  checkpoint {steps_done:>8d}/{train_steps}: "
                f"disc_return={eval_result['disc_return_mean']:.4f}, "
                f"success_rate={eval_result['success_rate']:.4f}"
            )

    # -- Post-training: greedy eval for stress metrics ----------------------
    eval_episodes_final = int(
        task_config.get("eval_episodes_final", eval_episodes_checkpoint * 2)
    )

    _saved_eps = None
    if hasattr(agent, "policy") and hasattr(agent.policy, "set_epsilon"):
        _saved_eps = agent.policy._epsilon
        agent.policy.set_epsilon(0.0)

    # Fresh logger for eval (same type as training logger).
    if stress_type is not None:
        eval_logger: TransitionLogger = _SafeAutoEventLogger(
            agent,
            n_base=n_base,
            gamma=gamma,
            stress_type=stress_type,
            wrapper=wrapper_or_mdp,
            jackpot_reward_threshold=event_thresholds["jackpot_reward_threshold"],
            catastrophe_reward_threshold=event_thresholds["catastrophe_reward_threshold"],
            hazard_reward_threshold=event_thresholds["hazard_reward_threshold"],
            risky_state=event_thresholds["risky_state"],
        )
    else:
        eval_logger = _PlainSafeLogger(agent, n_base=n_base, gamma=gamma)

    eval_core = Core(agent, mdp_rl, callback_step=eval_logger)
    try:
        with rw.timer.phase("eval_final"):
            eval_core.evaluate(n_episodes=eval_episodes_final, quiet=True)
        eval_payload = eval_logger.build_payload()
        eval_episode_returns = _compute_episode_returns(eval_payload, gamma)
        print(
            f"[phase3_rl] {task}/{algorithm}/seed_{seed}: "
            f"greedy eval ({eval_episodes_final} eps): "
            f"mean_return={float(np.mean(eval_episode_returns)):.4f}"
            if len(eval_episode_returns) > 0 else
            f"[phase3_rl] {task}/{algorithm}/seed_{seed}: greedy eval: no episodes"
        )
    except Exception as _e:
        print(
            f"[phase3_rl] {task}/{algorithm}/seed_{seed}: "
            f"[WARN] greedy eval failed ({_e}); falling back to training data"
        )
        eval_payload = None
        eval_episode_returns = np.array([], dtype=np.float64)
    finally:
        if _saved_eps is not None:
            agent.policy._epsilon = _saved_eps

    # -- Post-training: build transitions payload ---------------------------
    print(
        f"[phase3_rl] {task}/{algorithm}/seed_{seed}: "
        f"transitions logged: {logger.n_transitions}"
    )

    with rw.timer.phase("post"):
        transitions_payload = logger.build_payload()
        rw.set_transitions(transitions_payload)

        # Build calibration stats from the training transitions.
        calibration_stats = aggregate_calibration_stats(
            transitions_payload, horizon=horizon
        )

        # Per-episode returns for stress metrics: prefer greedy eval.
        if len(eval_episode_returns) > 0:
            episode_returns = eval_episode_returns
        else:
            episode_returns = _compute_episode_returns(transitions_payload, gamma)

        # -- Phase III safe aggregate stats ---------------------------------
        safe_payload = (
            logger.build_safe_payload()
            if hasattr(logger, "build_safe_payload")
            else None
        )
        if safe_payload is not None:
            safe_stats = aggregate_safe_stats(
                safe_payload, T=horizon, gamma=gamma
            )
            # Merge safe stats into calibration stats.
            calibration_stats.update(safe_stats)

        # -- Event-level scalars (same as Phase II) -------------------------
        if "jackpot_event" in transitions_payload:
            calibration_stats["jackpot_event_rate"] = np.array(
                [float(np.mean(transitions_payload["jackpot_event"]))],
            )
        if "catastrophe_event" in transitions_payload:
            calibration_stats["catastrophe_event_rate"] = np.array(
                [float(np.mean(transitions_payload["catastrophe_event"]))],
            )

        _payload_for_shortcut = None
        if eval_payload is not None and "shortcut_action_taken" in eval_payload:
            _payload_for_shortcut = eval_payload
        elif "shortcut_action_taken" in transitions_payload:
            _payload_for_shortcut = transitions_payload
        if _payload_for_shortcut is not None:
            risky_ep_flags = _compute_episode_event_flags(
                _payload_for_shortcut, "shortcut_action_taken"
            )
            shortcut_fraction = (
                float(np.mean(risky_ep_flags)) if len(risky_ep_flags) > 0 else 0.0
            )
            calibration_stats["shortcut_risky_path_fraction"] = np.array(
                [shortcut_fraction]
            )

        if "hazard_cell_hit" in transitions_payload:
            calibration_stats["hazard_hit_rate"] = np.array(
                [float(np.mean(transitions_payload["hazard_cell_hit"]))],
            )
        if stress_type == "regime_shift":
            calibration_stats["regime_shift_episode"] = np.array(
                [float(task_config.get("change_at_episode", -1))],
            )
        else:
            calibration_stats["regime_shift_episode"] = np.array([-1.0])

        # -- Tail-risk scalars -----------------------------------------------
        tail_risk: dict[str, float] | None = None
        if stress_type in ("jackpot", "catastrophe", "hazard"):
            event_key_map = {
                "jackpot": "jackpot_event",
                "catastrophe": "catastrophe_event",
                "hazard": "hazard_cell_hit",
            }
            event_key = event_key_map[stress_type]
            _payload_for_events = (
                eval_payload
                if eval_payload is not None and event_key in eval_payload
                else transitions_payload
            )
            episode_event_flags = _compute_episode_event_flags(
                _payload_for_events, event_key
            )

            tail_risk_logger = TailRiskLogger()
            tail_risk = tail_risk_logger.compute(episode_returns, episode_event_flags)

            calibration_stats["return_cvar_5pct"] = np.array([tail_risk["cvar_5pct"]])
            calibration_stats["return_cvar_10pct"] = np.array([tail_risk["cvar_10pct"]])
            calibration_stats["return_top5pct_mean"] = np.array([tail_risk["top5pct_mean"]])
            calibration_stats["event_rate"] = np.array([tail_risk["event_rate"]])
            calibration_stats["event_conditioned_return"] = np.array(
                [tail_risk["event_conditioned_return"]],
            )

        # -- Adaptation scalars (regime_shift) -------------------------------
        adaptation: dict[str, Any] | None = None
        if stress_type == "regime_shift":
            change_at_episode = int(task_config.get("change_at_episode", 300))
            _train_episode_returns = _compute_episode_returns(transitions_payload, gamma)
            adaptation_logger = AdaptationMetricsLogger()
            adaptation = adaptation_logger.compute(_train_episode_returns, change_at_episode)

            calibration_stats["adaptation_pre_change_auc"] = np.array(
                [float(adaptation["pre_change_auc"])],
            )
            calibration_stats["adaptation_post_change_auc"] = np.array(
                [float(adaptation["post_change_auc"])],
            )
            _lag50 = adaptation.get("lag_to_50pct_recovery")
            _lag75 = adaptation.get("lag_to_75pct_recovery")
            _lag90 = adaptation.get("lag_to_90pct_recovery")
            calibration_stats["adaptation_lag_50pct"] = np.array(
                [float(_lag50) if _lag50 is not None else np.nan],
            )
            calibration_stats["adaptation_lag_75pct"] = np.array(
                [float(_lag75) if _lag75 is not None else np.nan],
            )
            calibration_stats["adaptation_lag_90pct"] = np.array(
                [float(_lag90) if _lag90 is not None else np.nan],
            )

        # Stage calibration stats.
        rw.set_calibration_stats(calibration_stats)

        # -- Target statistics (spec S8.4) ----------------------------------
        target_stats_logger = TargetStatsLogger()
        target_stats = target_stats_logger.compute(transitions_payload)

        target_stats_schema = make_npz_schema(
            phase="phase3",
            task=task,
            algorithm=algorithm,
            seed=seed,
            storage_mode="rl_online",
            arrays=list(target_stats.keys()),
        )
        save_npz_with_schema(
            rw.run_dir / "target_stats.npz",
            target_stats_schema,
            {k: np.asarray(v) for k, v in target_stats.items()},
        )

    # -- Compute summary metrics --------------------------------------------
    eval_summary = evaluator.summary()

    metrics: dict[str, Any] = {
        "train_steps": train_steps,
        "n_transitions": logger.n_transitions,
        **{k: v for k, v in eval_summary.items()},
    }

    if tail_risk is not None:
        metrics["tail_risk_metrics"] = tail_risk
        resolved_config["tail_risk_metrics"] = tail_risk

    if adaptation is not None:
        metrics["adaptation_metrics"] = adaptation
        resolved_config["adaptation_metrics"] = adaptation

    if "shortcut_risky_path_fraction" in calibration_stats:
        metrics["shortcut_risky_path_fraction"] = float(
            calibration_stats["shortcut_risky_path_fraction"][0]
        )

    # Episode returns for return-distribution figures.
    if len(episode_returns) > 0:
        metrics["episode_returns"] = episode_returns.tolist()
        _ef: np.ndarray | None = None
        if stress_type in ("jackpot", "catastrophe", "hazard"):
            _ef = episode_event_flags  # noqa: F821
        if _ef is not None and len(_ef) == len(episode_returns):
            metrics["episode_returns_noevent"] = (
                episode_returns[~_ef.astype(bool)].tolist()
            )
            metrics["episode_returns_event"] = (
                episode_returns[_ef.astype(bool)].tolist()
            )
        else:
            metrics["episode_returns_noevent"] = episode_returns.tolist()
            metrics["episode_returns_event"] = []
    else:
        metrics["episode_returns"] = []
        metrics["episode_returns_noevent"] = []
        metrics["episode_returns_event"] = []

    # -- Flush everything to disk -------------------------------------------
    rw.flush(
        metrics=metrics,
        step_count=train_steps,
        update_count=train_steps,
    )

    # -- Write safe provenance (spec S7.3) -----------------------------------
    schedule_raw = schedule._raw if hasattr(schedule, "_raw") else {}
    write_safe_provenance(
        rw.run_dir,
        schedule_path=str(schedule_path),
        calibration_source_path=schedule_raw.get("calibration_source_path", ""),
        calibration_hash=schedule_raw.get("calibration_hash", ""),
        source_phase=schedule_raw.get("source_phase", "phase2"),
    )

    wall_s = time.perf_counter() - t_start
    print(
        f"[phase3_rl] {task}/{algorithm}/seed_{seed}: "
        f"DONE in {wall_s:.1f}s -> {rw.run_dir}"
    )

    return {
        "task": task,
        "algorithm": algorithm,
        "seed": seed,
        "passed": True,
        "wall_s": wall_s,
        "run_dir": str(rw.run_dir),
        "summary": (
            f"steps_to_thr={eval_summary.get('steps_to_threshold')}, "
            f"final_sr={eval_summary.get('final_10pct_success_rate', 0):.3f}"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on full success, 1 if any run fails."""
    parser = argparse.ArgumentParser(
        prog="run_phase3_rl",
        description=(
            "Phase III RL runner: safe weighted-LSE online tabular RL runs "
            "for the paper suite."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(_DEFAULT_CONFIG),
        help=f"Suite config JSON (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=list(_TASK_FACTORIES.keys()) + ["all"],
        help="Filter to one task, or 'all' for all tasks.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=list(SAFE_RL_ALGORITHMS.keys()),
        help="Filter to one safe algorithm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter to one seed.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(_DEFAULT_OUT_ROOT),
        help=f"Output root directory (default: {_DEFAULT_OUT_ROOT}).",
    )
    parser.add_argument(
        "--schedule-dir",
        type=Path,
        default=Path(_DEFAULT_SCHEDULE_DIR),
        help=f"Schedule directory (default: {_DEFAULT_SCHEDULE_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print planned runs without executing.",
    )
    args = parser.parse_args(argv)

    # -- Load config --------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        return 1

    with open(config_path, "r") as f:
        config = json.load(f)

    # -- Build plan ---------------------------------------------------------
    plan = build_plan(
        config,
        task_filter=args.task,
        algorithm_filter=args.algorithm,
        seed_filter=args.seed,
    )

    if not plan:
        print("No runs matched the filters.", file=sys.stderr)
        return 1

    out_root = Path(args.out_root)
    schedule_dir = Path(args.schedule_dir)
    suite = config.get("suite", "paper_suite")

    # -- Dry run: print plan and exit --------------------------------------
    if args.dry_run:
        print(f"[phase3_rl] DRY RUN -- {len(plan)} run(s) planned:")
        print(f"  config:       {config_path}")
        print(f"  out_root:     {out_root}")
        print(f"  schedule_dir: {schedule_dir}")
        print()
        for i, entry in enumerate(plan, 1):
            tc = entry["task_config"]
            sched_file = schedule_dir / entry["task"] / "schedule.json"
            sched_exists = sched_file.is_file()
            print(
                f"  [{i:>3d}] task={entry['task']:<22s} "
                f"algorithm={entry['algorithm']:<20s} "
                f"seed={entry['seed']:<5d} "
                f"train_steps={tc['train_steps']:>8d} "
                f"schedule={'OK' if sched_exists else 'MISSING'}"
            )
        return 0

    # -- Execute runs -------------------------------------------------------
    print(
        f"[phase3_rl] Executing {len(plan)} run(s) "
        f"(out_root={out_root}, schedule_dir={schedule_dir})"
    )

    results: list[dict[str, Any]] = []
    n_passed = 0

    for i, entry in enumerate(plan, 1):
        print(
            f"\n[phase3_rl] === Run {i}/{len(plan)}: "
            f"{entry['task']}/{entry['algorithm']}/seed_{entry['seed']} ==="
        )
        try:
            result = run_single(
                task=entry["task"],
                algorithm=entry["algorithm"],
                seed=entry["seed"],
                task_config=entry["task_config"],
                out_root=out_root,
                schedule_dir=schedule_dir,
                suite=suite,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[phase3_rl] FAILED: {exc!r}")
            print(tb)
            result = {
                "task": entry["task"],
                "algorithm": entry["algorithm"],
                "seed": entry["seed"],
                "passed": False,
                "wall_s": 0.0,
                "run_dir": None,
                "summary": "EXCEPTION",
                "error": repr(exc),
                "traceback": tb,
            }
        results.append(result)
        if result["passed"]:
            n_passed += 1

    # -- Summary ------------------------------------------------------------
    print(f"\n[phase3_rl] Summary: {n_passed}/{len(results)} runs passed")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['task']:<22s} {r['algorithm']:<20s} seed={r['seed']:<5d} "
            f"{status}  {r.get('summary', '')}"
        )

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
