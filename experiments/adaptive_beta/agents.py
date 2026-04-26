"""Tabular Q-learning agent with pluggable per-episode beta schedule.

Phase VII M3.1. Spec authority:
``docs/specs/phase_VII_adaptive_beta.md`` §3 (operator), §4 (schedule
shape), §5 (methods), §16.1 (β=0 bit-identity), §16.2 (single code
path), §22.1 (operator import), §22.3 (canonical sign).

The agent is method-agnostic by construction: every method ID flows
through the same private ``_step_update`` routine; only the
:class:`BetaSchedule` object passed in differs (spec §16.2). β is
constant within an episode and is fetched once via
``schedule.beta_for_episode(e)`` at episode start (spec §2 rule 2,
§13.2.1). At ``|β| <= 1e-8`` the kernel collapses to ``r + γ·v_next``
exactly, giving bit-identical classical Q-learning targets (spec §16.1).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from experiments.adaptive_beta.schedules import BetaSchedule
from src.lse_rl.operator.tab_operator import (
    _EPS_BETA,
    effective_discount,
    g,
)


__all__ = [
    "AdaptiveBetaQAgent",
    "linear_epsilon_schedule",
]


def linear_epsilon_schedule(
    start: float = 1.0,
    end: float = 0.05,
    decay_episodes: int = 1000,
) -> Callable[[int], float]:
    """Linear ε decay from ``start`` at e=0 to ``end`` at e=decay_episodes.

    Constant at ``end`` thereafter. Default schedule mandated by the M3.1
    task spec.
    """
    if decay_episodes <= 0:
        raise ValueError(f"decay_episodes must be > 0, got {decay_episodes}")
    if not 0.0 <= end <= start <= 1.0:
        raise ValueError(
            f"require 0 <= end <= start <= 1, got start={start}, end={end}"
        )

    def _eps(e: int) -> float:
        if e <= 0:
            return float(start)
        if e >= decay_episodes:
            return float(end)
        frac = e / decay_episodes
        return float(start + (end - start) * frac)

    return _eps


class AdaptiveBetaQAgent:
    """Tabular Q-learning agent with a per-episode β schedule.

    Public API matches the M3.1 task spec. Single private
    :meth:`_step_update` is the lone TD-update entry point so all
    method IDs share one code path (spec §16.2). Constructed via
    :func:`build_schedule` upstream; the agent itself is schedule-
    agnostic.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_schedule: Callable[[int], float],
        beta_schedule: BetaSchedule,
        rng: np.random.Generator,
        env_canonical_sign: Optional[str] = None,
        q_init: float = 0.0,
        divergence_threshold: float = 1.0e6,
    ) -> None:
        if n_states <= 0 or n_actions <= 0:
            raise ValueError(
                f"n_states ({n_states}) and n_actions ({n_actions}) must be > 0"
            )
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

        self._n_states = int(n_states)
        self._n_actions = int(n_actions)
        self._gamma = float(gamma)
        self._lr = float(learning_rate)
        self._epsilon_schedule = epsilon_schedule
        self._beta_schedule = beta_schedule
        self._rng = rng
        self._env_canonical_sign = env_canonical_sign
        self._divergence_threshold = float(divergence_threshold)

        self._Q: np.ndarray = np.full(
            (self._n_states, self._n_actions),
            float(q_init),
            dtype=np.float64,
        )

        # Per-instance code-path counter. Exposed publicly for test
        # introspection (spec §16.2: assert all methods enter
        # _step_update). Resets only via the dedicated helper.
        self._step_update_call_counter: int = 0

        # Per-episode state. Cleared in begin_episode().
        self._current_episode: int = -1
        self._current_beta: float = 0.0   # cached at begin_episode
        self._ep_rewards: List[float] = []
        self._ep_v_nexts: List[float] = []
        self._ep_aligned: List[bool] = []
        self._ep_d_eff: List[float] = []
        self._ep_signed_align: List[float] = []  # β · (r - v_next), per step
        self._ep_advantages: List[float] = []     # r - v_next, per step
        self._ep_td_errors: List[float] = []
        self._ep_td_targets: List[float] = []
        self._ep_divergence_event: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @property
    def step_update_call_counter(self) -> int:
        """Read-only accessor for the test suite (spec §16.2)."""
        return self._step_update_call_counter

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: int, episode_index: int) -> int:
        """ε-greedy with deterministic tie-breaking on lowest action int.

        ``np.argmax`` returns the first occurrence of the maximum, which
        is the desired tie-break per the M3.1 task contract.
        """
        eps = float(self._epsilon_schedule(int(episode_index)))
        if self._rng.random() < eps:
            return int(self._rng.integers(0, self._n_actions))
        s = int(state)
        return int(np.argmax(self._Q[s]))

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def begin_episode(self, episode_index: int) -> None:
        """Cache β for this episode; clear per-episode buffers.

        Caching β once at episode start enforces the §13.2.1 invariant
        (β constant within episode) at the agent boundary as well as at
        the schedule boundary; even if a buggy caller called the
        schedule mid-episode, the agent still uses the cached value.
        """
        self._current_episode = int(episode_index)
        self._current_beta = float(
            self._beta_schedule.beta_for_episode(self._current_episode)
        )
        self._ep_rewards = []
        self._ep_v_nexts = []
        self._ep_aligned = []
        self._ep_d_eff = []
        self._ep_signed_align = []
        self._ep_advantages = []
        self._ep_td_errors = []
        self._ep_td_targets = []
        self._ep_divergence_event = False

    def step(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        absorbing: bool,
        episode_index: int,
    ) -> Dict[str, Any]:
        """One TD update; returns per-step diagnostic dict."""
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"step received episode_index={episode_index} but current "
                f"episode is {self._current_episode}; call begin_episode() "
                f"first"
            )
        return self._step_update(
            int(state),
            int(action),
            float(reward),
            int(next_state),
            bool(absorbing),
        )

    def end_episode(self, episode_index: int) -> Dict[str, Any]:
        """Push episode trace into the schedule; return diagnostics."""
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"end_episode received {episode_index} but current episode "
                f"is {self._current_episode}"
            )

        rewards = np.asarray(self._ep_rewards, dtype=np.float64)
        v_nexts = np.asarray(self._ep_v_nexts, dtype=np.float64)
        aligned = np.asarray(self._ep_aligned, dtype=bool)
        d_eff = np.asarray(self._ep_d_eff, dtype=np.float64)
        signed = np.asarray(self._ep_signed_align, dtype=np.float64)
        advantages = np.asarray(self._ep_advantages, dtype=np.float64)
        td_errors = np.asarray(self._ep_td_errors, dtype=np.float64)
        td_targets = np.asarray(self._ep_td_targets, dtype=np.float64)

        # Divergence (spec §7.3): NaN/Inf or |Q| > threshold.
        q_abs_max = float(np.max(np.abs(self._Q))) if self._Q.size else 0.0
        nan_count = int(np.isnan(self._Q).sum())
        if nan_count > 0 or q_abs_max > self._divergence_threshold:
            self._ep_divergence_event = True

        # Push to the schedule so it can compute β_{e+1}.
        self._beta_schedule.update_after_episode(
            self._current_episode,
            rewards,
            v_nexts,
            divergence_event=self._ep_divergence_event,
        )

        # Episode-level aggregates. ``aligned`` uses the strict ``>``
        # convention (spec §3.3 / §7.2). The non-strict variant is
        # logged separately as `frac_positive_signed_alignment` per
        # spec §7.2 (= aligned with non-strict ≥ semantics).
        if rewards.size > 0:
            alignment_rate = float(aligned.mean())
            mean_signed_alignment = float(signed.mean())
            # Non-strict: β·(r-v) >= 0 (per spec §7.2 distinction).
            frac_positive_signed = float((signed >= 0.0).mean())
            mean_abs_advantage = float(np.abs(advantages).mean())
            mean_d_eff = float(d_eff.mean())
            median_d_eff = float(np.median(d_eff))
            frac_d_eff_below_gamma = float((d_eff < self._gamma).mean())
            frac_d_eff_above_one = float((d_eff > 1.0).mean())
            mean_gamma_minus_d_eff = float((self._gamma - d_eff).mean())
            bellman_residual = float(np.abs(td_errors).mean())
            td_target_abs_max = float(np.max(np.abs(td_targets)))
        else:
            # Empty episode (shouldn't happen but be defensive).
            alignment_rate = 0.0
            mean_signed_alignment = 0.0
            frac_positive_signed = 0.0
            mean_abs_advantage = 0.0
            mean_d_eff = 0.0
            median_d_eff = 0.0
            frac_d_eff_below_gamma = 0.0
            frac_d_eff_above_one = 0.0
            mean_gamma_minus_d_eff = 0.0
            bellman_residual = 0.0
            td_target_abs_max = 0.0

        sched_diag = self._beta_schedule.diagnostics()

        return {
            "episode_index": int(self._current_episode),
            "beta_used": float(self._current_beta),
            # beta_raw / beta_deployed: post-update values intended for
            # the next episode (the schedule already advanced). Logging
            # both lets the runner record either current or next.
            "beta_raw": float(sched_diag["beta_raw"]),
            "beta_deployed": float(sched_diag["beta_used"]),
            "alignment_rate": alignment_rate,
            "mean_signed_alignment": mean_signed_alignment,
            "frac_positive_signed_alignment": frac_positive_signed,
            "mean_advantage": float(advantages.mean()) if advantages.size else 0.0,
            "mean_abs_advantage": mean_abs_advantage,
            "mean_d_eff": mean_d_eff,
            "median_d_eff": median_d_eff,
            "frac_d_eff_below_gamma": frac_d_eff_below_gamma,
            "frac_d_eff_above_one": frac_d_eff_above_one,
            "mean_gamma_minus_d_eff": mean_gamma_minus_d_eff,
            "bellman_residual": bellman_residual,
            "td_target_abs_max": td_target_abs_max,
            "q_abs_max": q_abs_max,
            "nan_count": nan_count,
            "divergence_event": bool(self._ep_divergence_event),
            "length": int(rewards.size),
        }

    # ------------------------------------------------------------------
    # Single TD-update code path (spec §16.2 invariant)
    # ------------------------------------------------------------------
    def _step_update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        absorbing: bool,
    ) -> Dict[str, Any]:
        """One TD update. EVERY method ID enters here.

        Bumping ``_step_update_call_counter`` first means the tests can
        introspect that a stepping codepath actually went through here
        regardless of where it errors out below.
        """
        self._step_update_call_counter += 1

        # Terminal v_next is 0 (spec §4.1). On absorbing transitions the
        # MDP has no successor and we must not read Q[next_state] which
        # could be a stale random value from the table.
        if absorbing:
            v_next = 0.0
        else:
            v_next = float(np.max(self._Q[next_state]))

        beta = self._current_beta
        td_target = g(beta, self._gamma, reward, v_next)

        # β=0 bit-identity guard (spec §16.1). The kernel already takes
        # the classical branch at |β| <= _EPS_BETA; assert it agrees
        # with the reference computation. Cheap and catches future
        # regressions in the operator module.
        if abs(beta) <= _EPS_BETA:
            classical = reward + self._gamma * v_next
            if td_target != classical:
                raise AssertionError(
                    f"β=0 bit-identity violated: kernel={td_target!r} != "
                    f"classical={classical!r} (β={beta!r}, γ={self._gamma!r}, "
                    f"r={reward!r}, v={v_next!r})"
                )

        td_error = td_target - self._Q[state, action]
        self._Q[state, action] += self._lr * td_error

        d_eff = effective_discount(beta, self._gamma, reward, v_next)
        # Strict alignment (spec §3.3 / §7.2 headline).
        signed = beta * (reward - v_next)
        aligned = signed > 0.0
        advantage = reward - v_next

        # Buffer per-step values for end_episode aggregation.
        self._ep_rewards.append(reward)
        self._ep_v_nexts.append(v_next)
        self._ep_aligned.append(bool(aligned))
        self._ep_d_eff.append(float(d_eff))
        self._ep_signed_align.append(float(signed))
        self._ep_advantages.append(float(advantage))
        self._ep_td_errors.append(float(td_error))
        self._ep_td_targets.append(float(td_target))

        # Per-step running divergence check. Cheaper than scanning all
        # of Q here: just inspect the cell we just wrote plus the
        # incoming target. NaN / overflow on this cell is sufficient
        # signal — the full Q-scan happens in end_episode anyway.
        new_q = self._Q[state, action]
        if not np.isfinite(new_q) or not np.isfinite(td_target):
            self._ep_divergence_event = True
        elif abs(new_q) > self._divergence_threshold:
            self._ep_divergence_event = True

        q_abs_max_running = float(np.max(np.abs(self._Q)))
        # ``bellman_residual`` per-step is the absolute TD error per
        # spec §7.3 (per-step) which differs from the per-episode mean
        # used in end_episode().
        return {
            "td_target": float(td_target),
            "td_error": float(td_error),
            "v_next": float(v_next),
            "advantage": float(advantage),
            "d_eff": float(d_eff),
            "aligned": bool(aligned),
            "signed_alignment": float(signed),
            "beta_used": float(beta),
            "bellman_residual": float(abs(td_error)),
            "q_abs_max_running": q_abs_max_running,
        }
