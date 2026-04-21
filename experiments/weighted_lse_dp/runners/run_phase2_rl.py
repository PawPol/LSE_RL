"""Phase II RL runner: online tabular RL runs on stress tasks.

For each (task, algorithm, seed) in the Phase II paper suite, this runner:

1. Loads the stress task via the matching ``make_*`` factory.
2. Creates the RL agent (QLearning or ExpectedSARSA) with fixed
   epsilon-greedy exploration (eps=0.1) and constant learning rate 0.1.
3. Attaches an :class:`EventTransitionLogger` (or plain
   :class:`TransitionLogger` for tasks without a stress type) as
   ``callback_step`` on Core.
4. Runs ``Core.learn(...)`` in checkpoint segments, evaluating with
   :class:`RLEvaluator` at each checkpoint.
5. After training: builds transitions payload, computes target stats,
   tail-risk metrics, adaptation metrics (as applicable), and flushes
   everything via :class:`RunWriter`.

CLI::

    .venv/bin/python experiments/weighted_lse_dp/runners/run_phase2_rl.py \\
        [--config PATH]            # default: experiments/weighted_lse_dp/configs/phase2/paper_suite.json
        [--task TASK | all]        # filter to one task, or 'all' for all tasks
        [--algorithm ALG]          # QLearning or ExpectedSARSA
        [--seed SEED]              # filter to one seed
        [--out-root PATH]          # default: results/weighted_lse_dp
        [--dry-run]                # print plan, no execution

Spec anchors: Phase II spec S5, S7, S8.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure repo root and vendored MushroomRL are importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "mushroom-rl-dev"))

import numpy as np  # noqa: E402

from experiments.weighted_lse_dp.common.callbacks import (  # noqa: E402
    AdaptationMetricsLogger,
    EventTransitionLogger,
    RLEvaluator,
    TailRiskLogger,
    TargetStatsLogger,
    TransitionLogger,
)
from experiments.weighted_lse_dp.common.calibration import (  # noqa: E402
    aggregate_calibration_stats,
    get_task_sign,
)
from experiments.weighted_lse_dp.common.schemas import RunWriter  # noqa: E402
from experiments.weighted_lse_dp.common.seeds import (  # noqa: E402
    seed_everything,
)
from experiments.weighted_lse_dp.common.io import (  # noqa: E402
    make_npz_schema,
    save_npz_with_schema,
)

# -- Stress task factories ----------------------------------------------------
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
# Constants
# ---------------------------------------------------------------------------

#: Default config path (relative to repo root).
_DEFAULT_CONFIG = (
    "experiments/weighted_lse_dp/configs/phase2/paper_suite.json"
)

#: Default output root. RunWriter.create appends phase/suite/task/algo/seed.
_DEFAULT_OUT_ROOT = "results/weighted_lse_dp"

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
#: chain_jackpot and chain_catastrophe have an extra absorbing state,
#: but n_base refers to the time-augmented MDP's base state count which
#: is the observation_space.n of the inner (non-augmented) MDP.
_N_BASE: dict[str, int] = {
    "chain_sparse_long": 60,
    "chain_jackpot": 26,        # 25 + 1 absorbing terminal
    "chain_catastrophe": 26,    # 25 + 1 absorbing terminal
    "chain_regime_shift": 25,
    "grid_sparse_goal": 49,        # 7x7 grid (Decision 1, R8-1)
    "grid_hazard": 25,
    "grid_regime_shift": 25,
    "taxi_bonus_shock": 44,
}

#: Supported RL algorithms.
_ALGORITHMS: dict[str, str] = {
    "QLearning": "QLearning",
    "ExpectedSARSA": "ExpectedSARSA",
}

#: Tasks whose factories return a stress wrapper (not a bare MDP).
#: For these tasks, RL training/eval must use the wrapper so that
#: hazard penalties, regime shifts, and bonus shocks fire during step().
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
# Event-detecting callback wrapper
# ---------------------------------------------------------------------------


class _AutoEventLogger(EventTransitionLogger):
    """EventTransitionLogger that auto-detects stress events from sample data.

    Since MushroomRL's ``Core.learn()`` owns the step loop, the runner
    cannot call ``mark_*()`` between steps.  This subclass overrides
    ``__call__`` to inspect the sample tuple and set the appropriate
    pending flags before delegating to the parent.

    Parameters
    ----------
    agent, n_base, gamma:
        Forwarded to :class:`EventTransitionLogger`.
    stress_type:
        One of ``"jackpot"``, ``"catastrophe"``, ``"regime_shift"``,
        ``"hazard"``, or ``None``.
    wrapper:
        The environment wrapper (needed for regime-shift ``post_change``
        attribute and hazard detection).
    hazard_reward_threshold:
        Step reward below this value triggers a hazard flag (default -4.0).
    jackpot_reward_threshold:
        Step reward above this value triggers a jackpot flag (default 2.0).
    catastrophe_reward_threshold:
        Step reward below this value triggers a catastrophe flag (default -2.0).
    """

    def __init__(
        self,
        agent: object,
        *,
        n_base: int,
        gamma: float,
        stress_type: str | None,
        wrapper: object | None = None,
        hazard_reward_threshold: float = -4.0,
        jackpot_reward_threshold: float = 2.0,
        catastrophe_reward_threshold: float = -2.0,
        risky_state: int | None = None,
    ) -> None:
        super().__init__(agent, n_base=n_base, gamma=gamma)
        self._stress_type = stress_type
        self._wrapper = wrapper
        self._hazard_thr = hazard_reward_threshold
        self._jackpot_thr = jackpot_reward_threshold
        self._catastrophe_thr = catastrophe_reward_threshold
        self._risky_state = risky_state

    def __call__(self, sample: tuple) -> None:
        """Detect events from sample, set flags, then delegate to parent."""
        reward = float(sample[2])
        absorbing = bool(sample[4])

        if self._stress_type == "jackpot":
            # Jackpot: unusually high reward (above threshold).
            if reward > self._jackpot_thr:
                self.mark_jackpot()

        elif self._stress_type == "catastrophe":
            # Catastrophe: large negative reward and absorbing.
            if reward < self._catastrophe_thr and absorbing:
                self.mark_catastrophe()
            # Shortcut action taken: action 0 at the designated risky state.
            # sample[0][0] is the augmented state id; base_state = id % n_base.
            if self._risky_state is not None:
                base_state = int(sample[0][0]) % self._n_base
                action = int(sample[1][0])
                if base_state == self._risky_state and action == 0:
                    self.mark_shortcut_taken()

        elif self._stress_type == "regime_shift":
            # Regime shift: check wrapper's post_change attribute.
            if self._wrapper is not None and hasattr(self._wrapper, "post_change"):
                self.mark_regime_post_change(self._wrapper.post_change)

        elif self._stress_type == "hazard":
            # Hazard: step reward below hazard threshold.
            if reward < self._hazard_thr:
                self.mark_hazard_hit()

        # Delegate to parent (appends transition + snapshots flags).
        super().__call__(sample)


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
        When set, include only this task. ``"all"`` means no filter.
    algorithm_filter:
        When set, include only this algorithm.
    seed_filter:
        When set, include only this seed.

    Returns
    -------
    list[dict]
        Each entry has keys: ``task``, ``algorithm``, ``seed``,
        ``task_config`` (the per-task config block from the JSON).
    """
    tasks_cfg = config.get("tasks", {})
    # Chain tasks use chain_seeds (5 seeds); others use seeds (3 seeds).
    seeds_default = tuple(config.get("seeds", [11, 29, 47]))
    chain_seeds = tuple(config.get("chain_seeds", [11, 29, 47, 67, 83]))

    plan: list[dict[str, Any]] = []

    for task_name, task_cfg in sorted(tasks_cfg.items()):
        if task_filter is not None and task_filter != "all" and task_name != task_filter:
            continue
        if task_name not in _TASK_FACTORIES:
            continue

        rl_algorithms = task_cfg.get("rl_algorithms", [])

        # Determine seed list: chain tasks use chain_seeds.
        is_chain = task_name.startswith("chain_")
        base_seeds = chain_seeds if is_chain else seeds_default

        for algo_name in rl_algorithms:
            if algorithm_filter is not None and algo_name != algorithm_filter:
                continue
            if algo_name not in _ALGORITHMS:
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


def _make_agent(
    algo_name: str,
    mdp_info: Any,
    *,
    epsilon: float = _EPSILON,
    learning_rate: float = _LEARNING_RATE,
) -> Any:
    """Construct a MushroomRL tabular RL agent.

    Deferred import so the module can be imported without MushroomRL on
    sys.path (e.g. for ``--dry-run``).
    """
    from mushroom_rl.algorithms.value.td import ExpectedSARSA, QLearning
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.rl_utils.parameters import Parameter

    pi = EpsGreedy(epsilon=Parameter(value=epsilon))

    if algo_name == "QLearning":
        return QLearning(
            mdp_info, pi, learning_rate=Parameter(value=learning_rate)
        )
    elif algo_name == "ExpectedSARSA":
        return ExpectedSARSA(
            mdp_info, pi, learning_rate=Parameter(value=learning_rate)
        )
    else:
        raise ValueError(f"Unsupported RL algorithm: {algo_name!r}")


def _call_factory(
    task: str,
    task_config: dict[str, Any],
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Call the appropriate factory for *task* and return (wrapper_or_mdp, mdp_rl, resolved_cfg).

    Each factory has a slightly different signature, so this helper
    normalises the call conventions.
    """
    factory = _TASK_FACTORIES[task]
    cfg = dict(task_config)  # shallow copy to avoid mutation

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
    """Compute per-episode discounted returns from the transitions payload.

    Returns a 1-D float64 array of length ``n_episodes``.
    """
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
    """Compute per-episode boolean event flags.

    An episode is flagged True if any transition in it has
    ``event_key == True``.

    Returns a 1-D bool array of length ``n_episodes``.
    """
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
    suite: str = "paper_suite",
) -> dict[str, Any]:
    """Execute one (task, algorithm, seed) run.

    Parameters
    ----------
    task:
        Task identifier (e.g. ``chain_jackpot``, ``grid_hazard``).
    algorithm:
        Algorithm identifier (``QLearning`` or ``ExpectedSARSA``).
    seed:
        Integer seed for reproducibility.
    task_config:
        Per-task config block from the suite JSON.
    out_root:
        Base output directory. RunWriter appends ``/<phase>/<suite>/...``.
    suite:
        Suite name for RunWriter path construction.

    Returns
    -------
    dict
        Summary dict with keys: ``task``, ``algorithm``, ``seed``,
        ``passed``, ``wall_s``, ``run_dir``, ``summary``, and
        optionally ``error`` / ``traceback``.
    """
    from mushroom_rl.core import Core  # noqa: E402

    t_start = time.perf_counter()

    # -- Resolve parameters from task config --------------------------------
    train_steps = int(task_config["train_steps"])
    checkpoint_every = int(task_config["checkpoint_every"])
    eval_episodes_checkpoint = int(task_config.get("eval_episodes_checkpoint", 50))
    success_threshold = float(task_config["success_threshold"])
    gamma = float(task_config["gamma"])
    horizon = int(task_config["horizon"])
    stress_type: str | None = task_config.get("stress_type", None)
    n_base = _N_BASE[task]

    # -- Read hyperparameter overrides (used by ablation runner) -----------
    # run_phase2_ablation passes epsilon_override and lr_multiplier via
    # task_config so that every sweep run uses its own hyperparameters.
    _eps_override = task_config.get("epsilon_override")
    _lr_mult = task_config.get("lr_multiplier", 1.0)
    effective_epsilon: float = (
        float(_eps_override) if _eps_override is not None else _EPSILON
    )
    effective_lr: float = _LEARNING_RATE * float(_lr_mult)

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
        "task_config": task_config,
    }

    # -- Seed everything ----------------------------------------------------
    seed_everything(seed)

    # -- Create environment -------------------------------------------------
    wrapper_or_mdp, mdp_rl, factory_cfg = _call_factory(task, task_config, seed)

    # Merge factory resolved config into our resolved config.
    resolved_config["factory_config"] = factory_cfg

    # BLOCKER A fix: for wrapper-backed tasks, train/eval on the stressed
    # wrapper.  The base mdp_rl only reflects pre-shift/base dynamics;
    # the wrapper's step() injects hazard penalties, regime shifts, and
    # bonus shocks.
    if task in _WRAPPER_TASKS_RL:
        mdp_rl = DiscreteTimeAugmentedEnv(wrapper_or_mdp, horizon=horizon)

    # Propagate gamma into the RL env.
    mdp_rl.info.gamma = gamma

    # -- Create agent -------------------------------------------------------
    agent = _make_agent(
        algorithm, mdp_rl.info,
        epsilon=effective_epsilon,
        learning_rate=effective_lr,
    )

    # -- Create RunWriter ---------------------------------------------------
    rw = RunWriter.create(
        base=out_root,
        phase="phase2",
        suite=suite,
        task=task,
        algorithm=algorithm,
        seed=seed,
        config=resolved_config,
        storage_mode="rl_online",
    )

    # -- Create callbacks ---------------------------------------------------
    # Determine event-detection thresholds from config.
    hazard_reward_thr = float(task_config.get("hazard_reward", -5.0)) + 1.0
    # R8-4: derive jackpot threshold from the correct config key.
    # chain_jackpot uses "jackpot_reward"; taxi_bonus_shock uses "bonus_reward"
    # (no jackpot_reward key).  Use goal_reward (base delivery reward, default 1.0)
    # + bonus_reward * 0.5 as the threshold when jackpot_reward is absent.
    if "jackpot_reward" in task_config:
        jackpot_reward_thr = float(task_config["jackpot_reward"]) * 0.5
    else:
        bonus_reward = float(task_config.get("bonus_reward", 5.0))
        goal_reward = float(task_config.get("goal_reward", 1.0))
        jackpot_reward_thr = goal_reward + bonus_reward * 0.5
    catastrophe_reward_thr = float(task_config.get("catastrophe_reward", -10.0)) * 0.5

    # For catastrophe tasks, pass the risky state so shortcut_action_taken
    # can be flagged whenever action 0 is selected at that state.
    risky_state_for_logger: int | None = None
    if stress_type == "catastrophe":
        risky_state_for_logger = int(task_config.get("risky_state", 15))

    if stress_type is not None:
        logger: TransitionLogger = _AutoEventLogger(
            agent,
            n_base=n_base,
            gamma=gamma,
            stress_type=stress_type,
            wrapper=wrapper_or_mdp,
            hazard_reward_threshold=hazard_reward_thr,
            jackpot_reward_threshold=jackpot_reward_thr,
            catastrophe_reward_threshold=catastrophe_reward_thr,
            risky_state=risky_state_for_logger,
        )
    else:
        logger = TransitionLogger(agent, n_base=n_base, gamma=gamma)

    # Catastrophe tasks have absorbing failure states (early termination with
    # negative reward).  The default success criterion (any absorbing at t <
    # horizon) would wrongly count those as successes.
    _eval_success_fn = None
    if stress_type == "catastrophe":
        _eval_success_fn = lambda r, a, t, h: bool(a and t < h and float(r) > 0.0)

    # Use a separate env for evaluation so that evaluation episodes do not
    # advance the training env's episode counter or perturb its RNG state.
    # This is critical for regime-shift tasks where the change point is
    # controlled by the training env's episode count.
    mdp_eval = copy.deepcopy(mdp_rl)
    evaluator = RLEvaluator(
        agent=agent,
        env=mdp_eval,
        run_writer=rw,
        n_eval_episodes=eval_episodes_checkpoint,
        success_threshold=success_threshold,
        gamma=gamma,
        success_fn=_eval_success_fn,
    )

    # -- Create Core --------------------------------------------------------
    core = Core(agent, mdp_rl, callback_step=logger)

    # -- Training loop with checkpoints ------------------------------------
    print(
        f"[phase2_rl] {task}/{algorithm}/seed_{seed}: "
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

    # -- Post-training: greedy eval for stress metrics (R4-2 fix) -----------
    # CVaR, event-conditioned return, and adaptation lag must be measured on
    # the learned policy (epsilon=0), not the epsilon-greedy training history.
    eval_episodes_final = int(
        task_config.get("eval_episodes_final", eval_episodes_checkpoint * 2)
    )

    # Temporarily set exploration to 0 for greedy evaluation.
    _saved_eps = None
    if hasattr(agent, "policy") and hasattr(agent.policy, "set_epsilon"):
        _saved_eps = agent.policy._epsilon
        agent.policy.set_epsilon(0.0)

    # Fresh logger for eval (same type as training logger, no history).
    if stress_type is not None:
        eval_logger: TransitionLogger = _AutoEventLogger(
            agent,
            n_base=n_base,
            gamma=gamma,
            stress_type=stress_type,
            wrapper=wrapper_or_mdp,
            hazard_reward_threshold=hazard_reward_thr,
            jackpot_reward_threshold=jackpot_reward_thr,
            catastrophe_reward_threshold=catastrophe_reward_thr,
            risky_state=risky_state_for_logger,
        )
    else:
        eval_logger = TransitionLogger(agent, n_base=n_base, gamma=gamma)

    eval_core = Core(agent, mdp_rl, callback_step=eval_logger)
    try:
        with rw.timer.phase("eval_final"):
            eval_core.evaluate(n_episodes=eval_episodes_final, quiet=True)
        eval_payload = eval_logger.build_payload()
        eval_episode_returns = _compute_episode_returns(eval_payload, gamma)
        print(
            f"[phase2_rl] {task}/{algorithm}/seed_{seed}: "
            f"greedy eval ({eval_episodes_final} eps): "
            f"mean_return={float(np.mean(eval_episode_returns)):.4f}"
            if len(eval_episode_returns) > 0 else
            f"[phase2_rl] {task}/{algorithm}/seed_{seed}: greedy eval: no episodes"
        )
    except Exception as _e:
        print(
            f"[phase2_rl] {task}/{algorithm}/seed_{seed}: "
            f"[WARN] greedy eval failed ({_e}); falling back to training data for stress metrics"
        )
        eval_payload = None
        eval_episode_returns = np.array([], dtype=np.float64)
    finally:
        # Restore exploration rate.
        if _saved_eps is not None:
            agent.policy._epsilon = _saved_eps

    # -- Post-training: build transitions payload ---------------------------
    print(
        f"[phase2_rl] {task}/{algorithm}/seed_{seed}: "
        f"transitions logged: {logger.n_transitions}"
    )

    with rw.timer.phase("post"):
        transitions_payload = logger.build_payload()
        rw.set_transitions(transitions_payload)

        # Build calibration stats from the training transitions.
        calibration_stats = aggregate_calibration_stats(
            transitions_payload, horizon=horizon, sign=get_task_sign(task)
        )

        # -- Per-episode returns for stress metrics: use greedy eval data ---
        # Fall back to training data only if eval failed (shouldn't happen).
        if len(eval_episode_returns) > 0:
            episode_returns = eval_episode_returns
        else:
            episode_returns = _compute_episode_returns(transitions_payload, gamma)

        # -- Phase II event-level scalars ----------------------------------
        if "jackpot_event" in transitions_payload:
            calibration_stats["jackpot_event_rate"] = np.array(
                [float(np.mean(transitions_payload["jackpot_event"]))],
            )
        if "catastrophe_event" in transitions_payload:
            calibration_stats["catastrophe_event_rate"] = np.array(
                [float(np.mean(transitions_payload["catastrophe_event"]))],
            )
        # R7-A4: episode-level risky-path fraction for catastrophe tasks.
        # "Used the risky path" = at least one shortcut_action_taken=True
        # in that episode.  Use eval payload (learned policy) when available,
        # fall back to training payload.
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

        # -- Phase II tail-risk scalars ------------------------------------
        # Use greedy eval payload for event flags so tail risk reflects the
        # learned policy, not the training exploration history (R4-2).
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

        # -- Phase II adaptation scalars -----------------------------------
        # Adaptation lag measures recovery in the training curve around the
        # change point — greedy eval data post-training doesn't carry this
        # temporal structure, so use the full training episode returns here.
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

        # Stage calibration stats (now with all scalars populated).
        rw.set_calibration_stats(calibration_stats)

        # -- Target statistics (spec S8.4) ---------------------------------
        target_stats_logger = TargetStatsLogger()
        target_stats = target_stats_logger.compute(transitions_payload)

        # Store target stats in a separate NPZ file since these 4 keys
        # are not part of TRANSITIONS_ARRAYS.
        target_stats_schema = make_npz_schema(
            phase="phase2",
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

    # Merge tail-risk and adaptation into metrics/resolved_config.
    if tail_risk is not None:
        metrics["tail_risk_metrics"] = tail_risk
        resolved_config["tail_risk_metrics"] = tail_risk

    if adaptation is not None:
        metrics["adaptation_metrics"] = adaptation
        resolved_config["adaptation_metrics"] = adaptation

    # R7-A4: shortcut_risky_path_fraction — written to metrics.json for
    # downstream aggregation and reporting (spec §10.2).
    if "shortcut_risky_path_fraction" in calibration_stats:
        metrics["shortcut_risky_path_fraction"] = float(
            calibration_stats["shortcut_risky_path_fraction"][0]
        )

    # -- Store episode returns for return-distribution figures (MAJOR R3-3) --
    # Split by event flag so aggregation can build base_returns (no event)
    # and stress_returns (event occurred) for figure 11.1.2.
    if len(episode_returns) > 0:
        metrics["episode_returns"] = episode_returns.tolist()
        # episode_event_flags is only defined when stress_type is set;
        # fall back to "all non-event" for base tasks.
        _ef: np.ndarray | None = None
        if stress_type in ("jackpot", "catastrophe", "hazard"):
            _ef = episode_event_flags  # noqa: F821  (set in the block above)
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
        update_count=train_steps,  # n_steps_per_fit=1 => one update per step
    )

    wall_s = time.perf_counter() - t_start
    print(
        f"[phase2_rl] {task}/{algorithm}/seed_{seed}: "
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
    """CLI entry point. Returns 0 on full success, 1 if any run fails.

    Parameters
    ----------
    argv:
        Optional argv list for testing. When ``None``, ``sys.argv[1:]``
        is used.
    """
    parser = argparse.ArgumentParser(
        prog="run_phase2_rl",
        description=(
            "Phase II RL runner: online tabular RL runs for the "
            "weighted-LSE DP stress-task paper suite."
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
        choices=list(_ALGORITHMS.keys()),
        help="Filter to one algorithm.",
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
    suite = config.get("suite", "paper_suite")

    # -- Dry run: print plan and exit --------------------------------------
    if args.dry_run:
        print(f"[phase2_rl] DRY RUN -- {len(plan)} run(s) planned:")
        print(f"  config:    {config_path}")
        print(f"  out_root:  {out_root}")
        print()
        for i, entry in enumerate(plan, 1):
            tc = entry["task_config"]
            print(
                f"  [{i:>3d}] task={entry['task']:<22s} "
                f"algorithm={entry['algorithm']:<16s} "
                f"seed={entry['seed']:<5d} "
                f"train_steps={tc['train_steps']:>8d} "
                f"checkpoint_every={tc['checkpoint_every']:>6d}"
            )
        return 0

    # -- Execute runs -------------------------------------------------------
    print(
        f"[phase2_rl] Executing {len(plan)} run(s) "
        f"(out_root={out_root})"
    )

    results: list[dict[str, Any]] = []
    n_passed = 0

    for i, entry in enumerate(plan, 1):
        print(
            f"\n[phase2_rl] === Run {i}/{len(plan)}: "
            f"{entry['task']}/{entry['algorithm']}/seed_{entry['seed']} ==="
        )
        try:
            result = run_single(
                task=entry["task"],
                algorithm=entry["algorithm"],
                seed=entry["seed"],
                task_config=entry["task_config"],
                out_root=out_root,
                suite=suite,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[phase2_rl] FAILED: {exc!r}")
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
    print(f"\n[phase2_rl] Summary: {n_passed}/{len(results)} runs passed")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['task']:<22s} {r['algorithm']:<16s} seed={r['seed']:<5d} "
            f"{status}  {r.get('summary', '')}"
        )

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
