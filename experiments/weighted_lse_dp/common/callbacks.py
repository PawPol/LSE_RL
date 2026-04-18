"""MushroomRL ``callback_step`` callback for per-transition calibration logging.

This module provides :class:`TransitionLogger`, a callable that attaches
to :class:`mushroom_rl.core.Core` via its ``callback_step`` hook.  After
every environment step, the callback decodes the time-augmented state,
extracts Q-values from the agent's table, and appends a row of
calibration data.

Once training is complete, :meth:`TransitionLogger.build_payload` returns
the full transitions dict (all 13 keys from
:data:`~schemas.TRANSITIONS_ARRAYS`) ready for
:meth:`~schemas.RunWriter.set_transitions`.

Usage::

    logger = TransitionLogger(agent=agent, n_base=25, gamma=0.99)
    core = Core(agent, env, callback_step=logger)
    core.learn(...)
    payload = logger.build_payload()
    rw.set_transitions(payload)

The margin formula is ``reward - v_next_beta0`` (NO gamma).  The
``gamma`` parameter is only used for ``td_target_beta0`` computation
inside :func:`~calibration.build_transitions_payload`.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.calibration import (  # noqa: E402
    build_transitions_payload_from_lists,
)

__all__ = [
    "TransitionLogger",
    "DPCurvesLogger",
    "RLEvaluator",
    "EventTransitionLogger",
    "AdaptationMetricsLogger",
    "TailRiskLogger",
    "TargetStatsLogger",
]


class TransitionLogger:
    """Stateful ``callback_step`` that accumulates per-transition calibration data.

    The callback is designed to be passed directly to
    :class:`mushroom_rl.core.Core` as ``callback_step``.  Core calls
    ``callback_step(sample)`` after every environment step, where
    ``sample`` is an 8-tuple (we use the first 6 elements).

    Parameters
    ----------
    agent:
        MushroomRL agent with a ``Q`` attribute (a
        :class:`mushroom_rl.approximators.table.Table` or compatible
        object exposing ``.table`` and ``__getitem__``).
    n_base:
        Number of base (un-augmented) states.  Used to decode
        ``aug_id -> (t, base_state)`` via integer division.
    gamma:
        Discount factor.  Forwarded to
        :func:`~calibration.build_transitions_payload_from_lists` for
        ``td_target_beta0`` computation.  NOT used in ``margin_beta0``.
    """

    def __init__(self, agent: object, *, n_base: int, gamma: float) -> None:
        self._agent = agent
        self._n_base = int(n_base)
        self._gamma = float(gamma)

        # Mutable accumulation state -- reset via :meth:`reset`.
        self._episode: int = 0
        self._episode_index: list[int] = []
        self._t: list[int] = []
        self._state: list[int] = []
        self._action: list[int] = []
        self._reward: list[float] = []
        self._next_state: list[int] = []
        self._absorbing: list[bool] = []
        self._last: list[bool] = []
        self._q_current_beta0: list[float] = []
        self._v_next_beta0: list[float] = []

    # ------------------------------------------------------------------
    # Core callback interface
    # ------------------------------------------------------------------

    def __call__(self, sample: tuple) -> None:
        """Called by Core after every environment step.

        Parameters
        ----------
        sample:
            ``(state, action, reward, next_state, absorbing, last, ...)``
            where ``state`` and ``next_state`` are augmented ids.
        """
        aug_id = int(sample[0][0])
        t = aug_id // self._n_base
        base_state = aug_id % self._n_base

        action = int(sample[1][0])
        reward = float(sample[2])

        next_aug_id = int(sample[3][0])
        next_base_state = next_aug_id % self._n_base

        absorbing = bool(sample[4])
        last = bool(sample[5])

        # Q-value extraction
        q_current = float(self._agent.Q[aug_id, action])
        v_next = float(np.max(self._agent.Q.table[next_aug_id, :]))
        # Finite-horizon contract: V[H]=0 — no continuation from terminal state.
        if absorbing or last:
            v_next = 0.0

        # Append row
        self._episode_index.append(self._episode)
        self._t.append(t)
        self._state.append(base_state)
        self._action.append(action)
        self._reward.append(reward)
        self._next_state.append(next_base_state)
        self._absorbing.append(absorbing)
        self._last.append(last)
        self._q_current_beta0.append(q_current)
        self._v_next_beta0.append(v_next)

        # Increment episode counter when the episode ends
        if last:
            self._episode += 1

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def build_payload(self) -> dict[str, np.ndarray]:
        """Build the transitions payload dict with all 13 TRANSITIONS_ARRAYS keys.

        Delegates to
        :func:`~calibration.build_transitions_payload_from_lists` which
        computes ``margin_beta0``, ``td_target_beta0``, and
        ``td_error_beta0`` from the raw accumulated data.

        Returns
        -------
        dict[str, np.ndarray]
            Ready for :meth:`RunWriter.set_transitions`.

        Raises
        ------
        ValueError
            If no transitions have been recorded (empty lists).
        """
        return build_transitions_payload_from_lists(
            episode_index=self._episode_index,
            t=self._t,
            state=self._state,
            action=self._action,
            reward=self._reward,
            next_state=self._next_state,
            absorbing=self._absorbing,
            last=self._last,
            q_current_beta0=self._q_current_beta0,
            v_next_beta0=self._v_next_beta0,
            gamma=self._gamma,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated data and reset the episode counter."""
        self._episode = 0
        self._episode_index.clear()
        self._t.clear()
        self._state.clear()
        self._action.clear()
        self._reward.clear()
        self._next_state.clear()
        self._absorbing.clear()
        self._last.clear()
        self._q_current_beta0.clear()
        self._v_next_beta0.clear()

    @property
    def n_transitions(self) -> int:
        """Number of transitions accumulated so far."""
        return len(self._episode_index)


# -----------------------------------------------------------------------
# DP planning-curves logger (spec S7.3)
# -----------------------------------------------------------------------

_THRESHOLDS: tuple[float, ...] = (1e-2, 1e-4, 1e-6)
"""Bellman-residual thresholds whose first-crossing sweep index is tracked."""


class DPCurvesLogger:
    """Accumulates DP planning curves and records them via RunWriter.

    After every Bellman sweep the runner calls :meth:`record_sweep` with the
    sweep index, Bellman residual, wall-clock time, and the value table
    **after that sweep** (e.g. ``planner.V_sweep_history[i]`` from the DP
    planners — not the final ``planner.V`` repeated for every ``i``).
    The logger forwards each sweep to
    :meth:`~schemas.RunWriter.record_dp_sweep`, computing
    ``supnorm_to_exact`` on the fly when an exact solution is provided.

    Once planning terminates, :meth:`summary` returns a dict of
    sweep-count thresholds (first sweep index where the Bellman residual
    drops below 1e-2, 1e-4, 1e-6).

    Usage::

        dp_logger = DPCurvesLogger(run_writer=rw, v_exact=V_star, task="chain_base")

        # Called by runner after each sweep:
        dp_logger.record_sweep(sweep_idx, bellman_residual, wall_clock_s, v_current)

        # After all sweeps:
        summary = dp_logger.summary()  # thresholds dict

    Parameters
    ----------
    run_writer:
        A :class:`~schemas.RunWriter` instance. Each call to
        :meth:`record_sweep` delegates to
        :meth:`RunWriter.record_dp_sweep`.
    v_exact:
        Optional exact value function with shape ``(H+1, S)``.  When
        provided, the sup-norm ``||V_current - V_exact||_inf`` is
        computed and recorded for every sweep.
    task:
        Task identifier string.  When equal to ``"chain_base"``, value-
        table snapshots are forwarded to the RunWriter; otherwise
        ``None`` is passed (saving disk space on larger tasks).
    """

    def __init__(
        self,
        run_writer: object,
        *,
        v_exact: np.ndarray | None = None,
        task: str = "",
    ) -> None:
        self._rw = run_writer
        self._v_exact = (
            np.asarray(v_exact, dtype=np.float64) if v_exact is not None else None
        )
        self._record_snapshots: bool = task == "chain_base"

        # Internal bookkeeping for threshold summary.
        self._residuals: list[tuple[int, float]] = []  # (sweep_idx, residual)

    # ------------------------------------------------------------------
    # Per-sweep recording
    # ------------------------------------------------------------------

    def record_sweep(
        self,
        sweep_idx: int,
        bellman_residual: float,
        wall_clock_s: float,
        v_current: np.ndarray,
    ) -> None:
        """Record one sweep and forward it to the RunWriter.

        Parameters
        ----------
        sweep_idx:
            Monotonic sweep index (0-based or 1-based, caller's choice).
        bellman_residual:
            The Bellman residual for this sweep.
        wall_clock_s:
            Cumulative wall-clock time at the end of this sweep.
        v_current:
            Current value table, shape ``(H+1, S)``.
        """
        # Sup-norm to exact solution.
        supnorm: float | None = None
        if self._v_exact is not None:
            supnorm = float(np.max(np.abs(np.asarray(v_current) - self._v_exact)))

        # Value-table snapshot (chain_base only).
        snapshot: np.ndarray | None = None
        if self._record_snapshots:
            snapshot = np.asarray(v_current, dtype=np.float64)

        # Delegate to RunWriter.
        self._rw.record_dp_sweep(  # type: ignore[attr-defined]
            sweep_idx=sweep_idx,
            bellman_residual=bellman_residual,
            supnorm_to_exact=supnorm,
            wall_clock_s=wall_clock_s,
            v_table_snapshot=snapshot,
        )

        # Track for threshold summary.
        self._residuals.append((int(sweep_idx), float(bellman_residual)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, int | None]:
        """Return sweep-count thresholds.

        Returns
        -------
        dict
            Keys are ``"sweeps_to_1e-2"``, ``"sweeps_to_1e-4"``,
            ``"sweeps_to_1e-6"``.  Each value is the first sweep index
            where ``bellman_residual < threshold``, or ``None`` if the
            threshold was never reached.
        """
        result: dict[str, int | None] = {}
        for thr in _THRESHOLDS:
            # Format: "sweeps_to_1e-2", "sweeps_to_1e-4", "sweeps_to_1e-6"
            exp = int(np.log10(thr))
            key = f"sweeps_to_1e{exp}"
            first: int | None = None
            for idx, res in self._residuals:
                if res < thr:
                    first = idx
                    break
            result[key] = first
        return result


# -----------------------------------------------------------------------
# RL learning-curves evaluator (spec S7.4)
# -----------------------------------------------------------------------


class RLEvaluator:
    """Run evaluation rollouts and record learning-curve data via RunWriter.

    At each training checkpoint the caller invokes :meth:`evaluate` which
    executes ``n_eval_episodes`` greedy rollouts on the environment,
    computes discounted/undiscounted returns and success flags, then
    forwards the raw per-episode arrays to
    :meth:`~schemas.RunWriter.record_rl_checkpoint`.

    After training, :meth:`summary` aggregates across all recorded
    checkpoints to produce the scalar summary metrics required by spec
    S9.3: steps-to-threshold, AUC of the discounted-return curve, and
    final-10% averages.

    Usage::

        evaluator = RLEvaluator(
            agent=agent,
            env=env,
            run_writer=rw,
            n_eval_episodes=50,
            success_threshold=0.90,
            gamma=0.99,
        )

        # Called at each checkpoint:
        evaluator.evaluate(steps=5000)

        # After training:
        summary = evaluator.summary()

    Parameters
    ----------
    agent:
        MushroomRL agent (or any object exposing
        ``draw_action(state, policy_state) -> (action, policy_state)``).
    env:
        MushroomRL environment with ``reset() -> (state, info)`` and
        ``step(action) -> (next_state, reward, absorbing, info)``.
        The horizon is read from ``env.info.horizon``.
    run_writer:
        A :class:`~schemas.RunWriter` instance.
    n_eval_episodes:
        Number of greedy rollouts per checkpoint.
    success_threshold:
        Success-rate target for :meth:`summary`'s
        ``steps_to_threshold`` computation.
    gamma:
        Discount factor used to compute discounted returns.
    """

    def __init__(
        self,
        agent: object,
        env: object,
        run_writer: object,
        *,
        n_eval_episodes: int,
        success_threshold: float,
        gamma: float,
    ) -> None:
        self._agent = agent
        self._env = env
        self._rw = run_writer
        self._n_eval_episodes = int(n_eval_episodes)
        self._success_threshold = float(success_threshold)
        self._gamma = float(gamma)

        # Read horizon from the environment's MDPInfo.
        self._horizon: int = int(self._env.info.horizon)  # type: ignore[union-attr]

        # Accumulate per-checkpoint summaries for the final summary.
        self._checkpoints: list[int] = []
        self._disc_return_means: list[float] = []
        self._success_rates: list[float] = []

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, steps: int) -> dict[str, float]:
        """Run ``n_eval_episodes`` greedy rollouts (argmax Q-table, bypasses EpsGreedy policy) and record via RunWriter.

        ``episode_success`` is True only when the base environment signals
        ``absorbing=True`` before the horizon is exhausted (i.e.
        ``t < horizon``).  A horizon-timeout (``absorbing`` at
        ``t == horizon``) does NOT count as success.

        Parameters
        ----------
        steps:
            Total environment steps at this checkpoint (used as the
            x-axis value in the learning curve).

        Returns
        -------
        dict[str, float]
            ``{"disc_return_mean": ..., "success_rate": ...}`` for
            immediate caller-side logging.
        """
        disc_returns: list[float] = []
        undisc_returns: list[float] = []
        successes: list[bool] = []

        for _ in range(self._n_eval_episodes):
            state, _ = self._env.reset()  # type: ignore[union-attr]
            episode_disc_return = 0.0
            episode_undisc_return = 0.0
            episode_success = False
            t = 0

            while True:
                # Greedy action: argmax over Q-table, bypassing EpsGreedy policy.
                _q_vals = self._agent.Q.table[int(state[0]), :]  # type: ignore[union-attr]
                action = np.array([int(np.argmax(_q_vals))], dtype=np.int64)
                next_state, reward, absorbing, _ = self._env.step(action)  # type: ignore[union-attr]

                episode_disc_return += (self._gamma ** t) * float(reward)
                episode_undisc_return += float(reward)
                t += 1

                if absorbing:
                    # absorbing at t < horizon means the base env signalled a
                    # real task goal; absorbing at t == horizon is the wrapper's
                    # horizon-timeout signal.  Only the former counts as success.
                    episode_success = (t < self._horizon)
                    break
                if t >= self._horizon:
                    break

                state = next_state

            disc_returns.append(episode_disc_return)
            undisc_returns.append(episode_undisc_return)
            successes.append(episode_success)

        # Forward to RunWriter.
        self._rw.record_rl_checkpoint(  # type: ignore[attr-defined]
            steps=steps,
            disc_returns=disc_returns,
            undisc_returns=undisc_returns,
            successes=successes,
        )

        # Cache for summary().
        dr_mean = float(np.mean(disc_returns))
        sr = float(np.mean(successes))
        self._checkpoints.append(int(steps))
        self._disc_return_means.append(dr_mean)
        self._success_rates.append(sr)

        return {"disc_return_mean": dr_mean, "success_rate": sr}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, float | int | None]:
        """Compute final summary stats over all recorded checkpoints.

        Returns
        -------
        dict
            ``steps_to_threshold``: first checkpoint step where
            ``success_rate >= success_threshold``, or ``None``.

            ``auc_disc_return``: trapezoidal AUC of the discounted-return
            curve, normalised by the step range (i.e. divided by
            ``max_steps - min_steps``). If fewer than 2 checkpoints were
            recorded, returns ``0.0``.

            ``final_disc_return_mean``: mean discounted return over the
            last 10% of checkpoints.

            ``final_10pct_success_rate``: mean success rate over the last
            10% of checkpoints.
        """
        from experiments.weighted_lse_dp.common.metrics import (
            curve_auc,
            final_performance,
            steps_to_threshold,
        )

        chk = np.asarray(self._checkpoints, dtype=np.int64)
        dr = np.asarray(self._disc_return_means, dtype=np.float64)
        sr = np.asarray(self._success_rates, dtype=np.float64)

        # steps_to_threshold: first checkpoint where success_rate >= threshold.
        stt: float | int | None = steps_to_threshold(
            returns=sr,
            threshold=self._success_threshold,
            checkpoints=chk,
        ) if chk.size >= 1 else None
        # Convert to int if not None.
        if stt is not None:
            stt = int(stt)

        # AUC of the discounted return curve (trapezoidal).
        if chk.size >= 2:
            # Use actual checkpoint spacing for trapezoidal integration,
            # then normalise by the total step range so AUC is
            # scale-invariant.
            step_range = float(chk[-1] - chk[0])
            if step_range > 0:
                raw_auc = float(np.trapezoid(dr, x=chk.astype(np.float64)))
                auc_val = raw_auc / step_range
            else:
                auc_val = 0.0
        else:
            auc_val = 0.0

        # Final 10% averages.
        if dr.size >= 1:
            f10_dr = final_performance(dr, frac=0.10)
            f10_sr = final_performance(sr, frac=0.10)
        else:
            f10_dr = 0.0
            f10_sr = 0.0

        return {
            "steps_to_threshold": stt,
            "auc_disc_return": auc_val,
            "final_disc_return_mean": f10_dr,
            "final_10pct_success_rate": f10_sr,
        }


# -----------------------------------------------------------------------
# Phase II: Event-level transition logger (spec S8.1)
# -----------------------------------------------------------------------


class EventTransitionLogger(TransitionLogger):
    """Extension of :class:`TransitionLogger` that also records binary event flags.

    The runner sets pending event flags (via :meth:`set_step_events` or
    individual ``mark_*`` methods) **before** the Core step callback fires.
    When :meth:`__call__` executes, the pending flags are appended to their
    respective lists and then reset to ``False``.

    Parameters
    ----------
    agent, n_base, gamma:
        Forwarded to :class:`TransitionLogger`.
    """

    def __init__(self, agent: object, *, n_base: int, gamma: float) -> None:
        super().__init__(agent, n_base=n_base, gamma=gamma)

        # Pending flags (set by runner before each step).
        self._pending_jackpot: bool = False
        self._pending_catastrophe: bool = False
        self._pending_regime_post: bool = False
        self._pending_hazard: bool = False
        self._pending_shortcut: bool = False

        # Accumulated event arrays.
        self._jackpot_event: list[bool] = []
        self._catastrophe_event: list[bool] = []
        self._regime_post_change: list[bool] = []
        self._hazard_cell_hit: list[bool] = []
        self._shortcut_action_taken: list[bool] = []

    # ------------------------------------------------------------------
    # Mark helpers (runner calls these before each step)
    # ------------------------------------------------------------------

    def set_step_events(
        self,
        *,
        jackpot: bool = False,
        catastrophe: bool = False,
        regime_post: bool = False,
        hazard: bool = False,
        shortcut: bool = False,
    ) -> None:
        """Set all pending event flags for the upcoming step at once."""
        self._pending_jackpot = jackpot
        self._pending_catastrophe = catastrophe
        self._pending_regime_post = regime_post
        self._pending_hazard = hazard
        self._pending_shortcut = shortcut

    def mark_jackpot(self) -> None:
        """Mark the upcoming step as a jackpot event."""
        self._pending_jackpot = True

    def mark_catastrophe(self) -> None:
        """Mark the upcoming step as a catastrophe event."""
        self._pending_catastrophe = True

    def mark_regime_post_change(self, status: bool = True) -> None:
        """Mark the upcoming step as occurring after a regime change."""
        self._pending_regime_post = status

    def mark_hazard_hit(self) -> None:
        """Mark the upcoming step as a hazard-cell hit."""
        self._pending_hazard = True

    def mark_shortcut_taken(self) -> None:
        """Mark the upcoming step as a shortcut action."""
        self._pending_shortcut = True

    # ------------------------------------------------------------------
    # Core callback interface (extends parent)
    # ------------------------------------------------------------------

    def __call__(self, sample: tuple) -> None:  # type: ignore[override]
        """Record the transition and snapshot pending event flags."""
        # Delegate base transition logging.
        super().__call__(sample)

        # Snapshot and reset pending flags.
        self._jackpot_event.append(self._pending_jackpot)
        self._catastrophe_event.append(self._pending_catastrophe)
        self._regime_post_change.append(self._pending_regime_post)
        self._hazard_cell_hit.append(self._pending_hazard)
        self._shortcut_action_taken.append(self._pending_shortcut)

        self._pending_jackpot = False
        self._pending_catastrophe = False
        self._pending_regime_post = False
        self._pending_hazard = False
        self._pending_shortcut = False

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def build_payload(self) -> dict[str, np.ndarray]:
        """Return the base transitions payload plus 5 event-flag arrays."""
        payload = super().build_payload()
        payload["jackpot_event"] = np.asarray(self._jackpot_event, dtype=bool)
        payload["catastrophe_event"] = np.asarray(self._catastrophe_event, dtype=bool)
        payload["regime_post_change"] = np.asarray(self._regime_post_change, dtype=bool)
        payload["hazard_cell_hit"] = np.asarray(self._hazard_cell_hit, dtype=bool)
        payload["shortcut_action_taken"] = np.asarray(
            self._shortcut_action_taken, dtype=bool
        )
        return payload

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated data including event flags."""
        super().reset()
        self._pending_jackpot = False
        self._pending_catastrophe = False
        self._pending_regime_post = False
        self._pending_hazard = False
        self._pending_shortcut = False
        self._jackpot_event.clear()
        self._catastrophe_event.clear()
        self._regime_post_change.clear()
        self._hazard_cell_hit.clear()
        self._shortcut_action_taken.clear()


# -----------------------------------------------------------------------
# Phase II: Adaptation metrics (spec S8.2)
# -----------------------------------------------------------------------


class AdaptationMetricsLogger:
    """Compute adaptation metrics from per-episode returns around a regime change.

    This is a stateless helper: call :meth:`compute` with episode returns
    and the change-point episode index to get the metrics dict.
    """

    _ROLLING_WINDOW: int = 10
    """Window size for rolling-mean recovery lag computation."""

    def compute(
        self,
        episode_returns: np.ndarray,
        change_at_episode: int,
    ) -> dict[str, float | int | None | np.ndarray]:
        """Compute adaptation metrics.

        Parameters
        ----------
        episode_returns:
            Shape ``(n_episodes,)`` float64 array of per-episode returns.
        change_at_episode:
            Episode index at which the regime change occurs.

        Returns
        -------
        dict
            Keys: ``change_at_episode``, ``pre_change_auc``,
            ``post_change_auc``, ``lag_to_50pct_recovery``,
            ``lag_to_75pct_recovery``, ``lag_to_90pct_recovery``,
            ``post_change_optimum``.
        """
        episode_returns = np.asarray(episode_returns, dtype=np.float64)
        n_episodes = len(episode_returns)

        # Edge case: change point is at or beyond all episodes.
        if change_at_episode >= n_episodes:
            return {
                "change_at_episode": int(change_at_episode),
                "pre_change_auc": float("nan"),
                "post_change_auc": float("nan"),
                "lag_to_50pct_recovery": None,
                "lag_to_75pct_recovery": None,
                "lag_to_90pct_recovery": None,
                "post_change_optimum": float("nan"),
            }

        pre = episode_returns[:change_at_episode]
        post = episode_returns[change_at_episode:]

        pre_auc = float(np.nanmean(pre)) if len(pre) > 0 else float("nan")
        post_auc = float(np.nanmean(post)) if len(post) > 0 else float("nan")
        post_optimum = float(np.nanmax(post)) if len(post) > 0 else float("nan")

        # Recovery lags via rolling mean.
        w = self._ROLLING_WINDOW
        lags: dict[str, int | None] = {}
        for pct_label, pct in [
            ("lag_to_50pct_recovery", 0.50),
            ("lag_to_75pct_recovery", 0.75),
            ("lag_to_90pct_recovery", 0.90),
        ]:
            threshold = pct * post_optimum
            lag: int | None = None
            for i in range(len(post)):
                start = max(0, i - w + 1)
                rolling_mean = float(np.mean(post[start : i + 1]))
                if rolling_mean >= threshold:
                    lag = i
                    break
            lags[pct_label] = lag

        return {
            "change_at_episode": int(change_at_episode),
            "pre_change_auc": pre_auc,
            "post_change_auc": post_auc,
            "lag_to_50pct_recovery": lags["lag_to_50pct_recovery"],
            "lag_to_75pct_recovery": lags["lag_to_75pct_recovery"],
            "lag_to_90pct_recovery": lags["lag_to_90pct_recovery"],
            "post_change_optimum": post_optimum,
        }


# -----------------------------------------------------------------------
# Phase II: Tail-risk metrics (spec S8.3)
# -----------------------------------------------------------------------


class TailRiskLogger:
    """Compute tail-risk metrics from per-episode returns and event flags.

    This is a stateless helper: call :meth:`compute` with episode returns
    and boolean event flags to get quantiles, CVaR, and event statistics.
    """

    def compute(
        self,
        episode_returns: np.ndarray,
        event_flags: np.ndarray,
    ) -> dict[str, float]:
        """Compute tail-risk metrics.

        Parameters
        ----------
        episode_returns:
            Shape ``(n_episodes,)`` float64 array of per-episode returns.
        event_flags:
            Shape ``(n_episodes,)`` bool array.  True when the event of
            interest occurred in that episode.

        Returns
        -------
        dict
            Keys: ``return_q05``, ``return_q25``, ``return_q50``,
            ``return_q75``, ``return_q95``, ``cvar_5pct``,
            ``cvar_10pct``, ``top5pct_mean``, ``top10pct_mean``,
            ``event_rate``, ``event_conditioned_return``.
        """
        r = np.asarray(episode_returns, dtype=np.float64)
        flags = np.asarray(event_flags, dtype=bool)
        n = len(r)

        # Quantiles.
        q05, q25, q50, q75, q95 = np.nanpercentile(r, [5, 25, 50, 75, 95])

        # CVaR: sort ascending, take bottom alpha% slice.
        sorted_r = np.sort(r)
        k5 = max(1, int(np.ceil(n * 0.05)))
        k10 = max(1, int(np.ceil(n * 0.10)))
        cvar_5 = float(np.nanmean(sorted_r[:k5]))
        cvar_10 = float(np.nanmean(sorted_r[:k10]))

        # Top percentile means (descending).
        sorted_desc = sorted_r[::-1]
        top5_k = max(1, int(np.ceil(n * 0.05)))
        top10_k = max(1, int(np.ceil(n * 0.10)))
        top5_mean = float(np.nanmean(sorted_desc[:top5_k]))
        top10_mean = float(np.nanmean(sorted_desc[:top10_k]))

        # Event statistics.
        event_rate = float(np.mean(flags)) if n > 0 else 0.0
        if np.any(flags):
            event_cond_ret = float(np.nanmean(r[flags]))
        else:
            event_cond_ret = float("nan")

        return {
            "return_q05": float(q05),
            "return_q25": float(q25),
            "return_q50": float(q50),
            "return_q75": float(q75),
            "return_q95": float(q95),
            "cvar_5pct": cvar_5,
            "cvar_10pct": cvar_10,
            "top5pct_mean": top5_mean,
            "top10pct_mean": top10_mean,
            "event_rate": event_rate,
            "event_conditioned_return": event_cond_ret,
        }


# -----------------------------------------------------------------------
# Phase II: Target-statistics logger (spec S8.4)
# -----------------------------------------------------------------------


class TargetStatsLogger:
    """Compute target-statistics from a transitions payload.

    This is a stateless helper: call :meth:`compute` with a transitions
    payload dict (as returned by :meth:`TransitionLogger.build_payload`)
    to get aligned margins and running standard deviations.
    """

    def compute(
        self,
        transitions_payload: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute target statistics.

        Parameters
        ----------
        transitions_payload:
            Dict containing at least ``margin_beta0``,
            ``td_target_beta0``, and ``td_error_beta0`` arrays of
            shape ``(N,)``.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``aligned_positive``, ``aligned_negative``,
            ``td_target_std_running``, ``td_error_std_running``.
            All arrays have shape ``(N,)`` and dtype float64.
        """
        margin = np.asarray(transitions_payload["margin_beta0"], dtype=np.float64)
        td_target = np.asarray(transitions_payload["td_target_beta0"], dtype=np.float64)
        td_error = np.asarray(transitions_payload["td_error_beta0"], dtype=np.float64)

        n = len(margin)

        aligned_pos = np.maximum(margin, 0.0)
        aligned_neg = np.maximum(-margin, 0.0)

        # Running (expanding-window) standard deviation.
        td_target_std = np.empty(n, dtype=np.float64)
        td_error_std = np.empty(n, dtype=np.float64)

        if n > 0:
            # Welford's online algorithm for numerical stability.
            t_mean = 0.0
            t_m2 = 0.0
            e_mean = 0.0
            e_m2 = 0.0
            for i in range(n):
                count = i + 1
                # TD target running std.
                delta_t = td_target[i] - t_mean
                t_mean += delta_t / count
                delta_t2 = td_target[i] - t_mean
                t_m2 += delta_t * delta_t2
                td_target_std[i] = np.sqrt(t_m2 / count) if count > 1 else 0.0

                # TD error running std.
                delta_e = td_error[i] - e_mean
                e_mean += delta_e / count
                delta_e2 = td_error[i] - e_mean
                e_m2 += delta_e * delta_e2
                td_error_std[i] = np.sqrt(e_m2 / count) if count > 1 else 0.0

        return {
            "aligned_positive": aligned_pos,
            "aligned_negative": aligned_neg,
            "td_target_std_running": td_target_std,
            "td_error_std_running": td_error_std,
        }
