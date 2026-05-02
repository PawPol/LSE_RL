"""Strategic-learning agent baselines (Phase VIII M7.2).

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §6.3
patch-2026-05-01 §3 (lines 744–770) — promotes regret matching and
smoothed fictitious play from opponent (adversary) classes to AGENT
classes so they can serve as Phase VIII Stage 2 baselines. Pre-empts
the highest-probability reviewer attack ("weak baselines"): for a
strategic-learning paper the natural baselines include strategic-
learning agents themselves.

Wrapper contract
----------------
Both classes implement the same agent interface as
:class:`experiments.adaptive_beta.agents.AdaptiveBetaQAgent`:

- ``select_action(state, episode_index) -> int``
- ``begin_episode(episode_index) -> None``
- ``step(state, action, reward, next_state, absorbing, episode_index)
  -> Dict[str, Any]``
- ``end_episode(episode_index) -> Dict[str, Any]``

Per spec §6.3 patch §3 the wrappers:

- carry a ``beta_schedule = ZeroBetaSchedule`` that is unused (β not
  applicable; included only so the logger does not error on the
  missing field);
- emit per-episode metrics ``return``, ``length``, ``epsilon`` (= 0
  always), ``bellman_residual`` (= 0 always — non-Q-learning agents
  have no TD error), ``nan_count`` (= 0), ``divergence_event``
  (= False always); these are all pulled by the Stage 2 runner from
  the dict returned by :meth:`end_episode`;
- do NOT emit operator-mechanism metrics (``alignment_rate``,
  ``mean_d_eff``, ``beta_used``, etc.) — those are TAB-specific.

Payoff-matrix wiring
--------------------
The opponent classes (``RegretMatching``, ``SmoothedFictitiousPlay``)
take a ``payoff_opponent`` matrix at construction with the convention

    payoff_opponent[a_t, j]  --> "agent played a_t, opponent payoff for
                                  playing j" (rows = agent action, cols
                                  = opponent action; opponent regrets /
                                  best-responds over j).

When we wrap such a class as an AGENT, the role of "agent" shifts to
the env's adversary and the role of "opponent" shifts to us. We must
therefore pass a payoff matrix whose rows index the env-adversary's
action and whose cols index our (the wrapper's) action, with cell
values equal to OUR payoff. The natural source for this is the
agent-side payoff matrix, transposed:

    inner_payoff = payoff_agent.T

so that ``inner_payoff[env_adv_action, my_action] = my_payoff``. The
inner ``observe`` call is then ``observe(agent_action=env_adv_action,
opponent_action=my_action, ...)``, and the inner sampler over its
``n_actions = inner_payoff.shape[1] = my_n_actions`` returns *my*
action. See :meth:`_StrategicAgentBase.step` for the exact wiring.

DC-Long50 (and other non-matrix games) handling
-----------------------------------------------
Spec §10.3 declares strategic-learning agents on DC-Long50 to be
*"expected to fail (no value bootstrapping); this is a diagnostic
feature documenting the necessity of the TAB approach"*. We therefore
must NOT raise on non-matrix games. The fallback is a fixed
uniform-random policy: when no payoff matrix is available, the
wrapper draws actions uniformly from the agent action space and skips
all opponent-class state updates. The resulting AUC is whatever the
random policy earns, which is a non-NaN value the aggregator can use.

This is consistent with the spec: these agents are baselines for the
matrix-game cells (where they are well-defined); on DC-Long50 their
*lack* of value bootstrapping is the documented failure mode the
paper highlights.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from experiments.adaptive_beta.schedules import ZeroBetaSchedule
from experiments.adaptive_beta.strategic_games.adversaries.regret_matching import (
    RegretMatching,
)
from experiments.adaptive_beta.strategic_games.adversaries.smoothed_fictitious_play import (  # noqa: E501
    SmoothedFictitiousPlay,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


__all__ = [
    "RegretMatchingAgent",
    "SmoothedFictitiousPlayAgent",
]


def _as_int(x: Any) -> int:
    """Robust scalar extraction (lessons.md #28: ``flat[0]`` for state)."""
    return int(np.asarray(x).flat[0])


class _StrategicAgentBase:
    """Shared plumbing for strategic-learning agent wrappers.

    Subclasses customise ``_make_inner`` (constructs the underlying
    opponent class with ``payoff_agent.T`` as its inner
    ``payoff_opponent``) and inherit the lifecycle methods.

    The wrapper holds a reference to the env so :meth:`step` can read
    the just-realised opponent action from ``env.history`` (matrix
    games update the history in their own ``step`` BEFORE returning,
    and the runner calls ``agent.step`` immediately after
    ``env.step``). For non-matrix games (e.g. ``delayed_chain``) the
    history will reflect a 1-action passive opponent and the wrapper
    falls back to uniform-random action selection.

    Constructor parameters mirror :class:`AdaptiveBetaQAgent` for the
    fields the runner already passes (``n_states``, ``n_actions``,
    ``gamma``, ``learning_rate``, ``epsilon_schedule``, ``rng``,
    ``q_init``); the strategic-learning baselines do not USE these
    fields — they exist only for interface compatibility. The fields
    that ARE used are ``payoff_agent`` (the matrix-game agent-side
    payoff, ``None`` for non-matrix games), ``env_history_provider``
    (a zero-arg callable returning the env's running
    :class:`GameHistory`; allows the wrapper to peek at the latest
    realised opponent action in :meth:`step` without a hard env
    reference), and ``seed``.
    """

    #: Public marker so the runner / tests can identify the wrapper
    #: family without isinstance checks against private subclasses.
    is_strategic_learning_agent: bool = True

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_schedule: Callable[[int], float],
        rng: np.random.Generator,
        q_init: float = 0.0,
        payoff_agent: Optional[np.ndarray] = None,
        env_history_provider: Optional[Callable[[], GameHistory]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_states <= 0 or n_actions <= 0:
            raise ValueError(
                f"n_states ({n_states}) and n_actions ({n_actions}) must be > 0"
            )
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

        self._n_states: int = int(n_states)
        self._n_actions: int = int(n_actions)
        self._gamma: float = float(gamma)
        self._lr: float = float(learning_rate)
        # ``epsilon_schedule`` is accepted for interface parity with
        # AdaptiveBetaQAgent but never consulted: strategic-learning
        # baselines have no exploration parameter (their stochasticity
        # comes from the regret-matching / logit-FP policy itself).
        self._epsilon_schedule: Callable[[int], float] = epsilon_schedule
        self._rng: np.random.Generator = rng
        self._q_init: float = float(q_init)

        # Required-for-logger compatibility: the Stage 2 runner does not
        # query this, but the parent dispatcher's audit may.
        self.beta_schedule: ZeroBetaSchedule = ZeroBetaSchedule()

        # ``payoff_agent`` must be either None (non-matrix fallback) or
        # a 2-D array with rows = agent_actions, cols = opponent_actions
        # (the standard matrix-game convention exposed by every
        # ``games/<x>.py`` module).
        if payoff_agent is None:
            self._payoff_agent: Optional[np.ndarray] = None
            self._inner_payoff: Optional[np.ndarray] = None
            self._matrix_game: bool = False
        else:
            pa = np.asarray(payoff_agent, dtype=np.float64)
            if pa.ndim != 2:
                raise ValueError(
                    f"payoff_agent must be 2-D, got shape {pa.shape}"
                )
            if pa.shape[0] != self._n_actions:
                raise ValueError(
                    f"payoff_agent.shape[0] ({pa.shape[0]}) must equal "
                    f"agent n_actions ({self._n_actions})"
                )
            self._payoff_agent = pa
            # See module docstring "Payoff-matrix wiring": pass pa.T so
            # the inner class's regret/best-response dimension lines up
            # with our (the wrapper's) action space.
            self._inner_payoff = pa.T.copy()
            self._matrix_game = True

        self._env_history_provider: Optional[Callable[[], GameHistory]] = (
            env_history_provider
        )
        self._seed: Optional[int] = None if seed is None else int(seed)

        # Inner strategic-learning policy. Subclasses set this in their
        # own ``__init__`` via ``_make_inner()`` immediately after
        # ``super().__init__``.
        self._inner: Any = None

        # Per-episode lifecycle bookkeeping.
        self._current_episode: int = -1
        self._ep_return: float = 0.0
        self._ep_length: int = 0

        # Snapshot of ``len(env.history)`` at the start of the current
        # step. Used to identify the just-appended (agent_action,
        # opponent_action) pair the env recorded inside ``env.step``.
        self._history_len_before_step: int = 0

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    def _make_inner(self) -> Any:  # pragma: no cover — abstract
        raise NotImplementedError(
            "subclasses must implement _make_inner() to return the "
            "underlying opponent-class instance"
        )

    # ------------------------------------------------------------------
    # AdaptiveBetaQAgent-compatible interface
    # ------------------------------------------------------------------
    def select_action(self, state: Any, episode_index: int) -> int:
        """Return the next agent action.

        Matrix games: defer to the inner strategic policy's ``act``.
        Non-matrix fallback: uniform-random over ``n_actions``.

        ``state`` is accepted for interface parity but ignored — the
        strategic-learning policy is *history-conditioned*, not
        state-conditioned (action-set is the entire policy domain in
        the matrix-game model these classes were written for).
        """
        del state, episode_index  # explicit: ignored
        if self._matrix_game and self._inner is not None:
            history = self._current_history()
            return int(self._inner.act(history))
        # Fallback: uniform random over the wrapper's action space.
        return int(self._rng.integers(0, self._n_actions))

    def begin_episode(self, episode_index: int) -> None:
        """Clear per-episode buffers; do NOT reset inner policy.

        The strategic-learning policies maintain cross-episode state
        (regret tables, empirical-frequency belief) by design; that is
        the whole point of the baseline. We therefore do NOT call
        ``inner.reset()`` between episodes — only at construction.
        """
        self._current_episode = int(episode_index)
        self._ep_return = 0.0
        self._ep_length = 0
        self._history_len_before_step = 0

    def step(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        absorbing: bool,
        episode_index: int,
    ) -> Dict[str, Any]:
        """Update the inner policy from the just-realised step.

        Sequence: the runner has already called ``env.step(action)``,
        which (for matrix games) appended one entry to
        ``env.history``. We read that entry (agent + opponent action,
        agent reward) and forward it to the inner strategic-learning
        policy via ``observe``.

        For non-matrix games (``env.history`` reflects a 1-action
        passive opponent) the observe call is a no-op skip — there is
        no usable opponent action, and the inner policy is None
        anyway under the fallback path.
        """
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"step received episode_index={episode_index} but current "
                f"episode is {self._current_episode}; call begin_episode() "
                f"first"
            )
        del state, next_state  # not used by the strategic-learning policy
        a = _as_int(action)
        r = float(reward)

        self._ep_return += r
        self._ep_length += 1

        if self._matrix_game and self._inner is not None:
            history = self._current_history()
            # The env's most recent history entry is the (agent_action,
            # opponent_action) pair from this very step — env.step
            # appends it BEFORE returning, and the runner calls
            # agent.step immediately after env.step, so by construction
            # ``len(history) >= 1``.
            n = len(history)
            if n == 0:
                # Defensive: should not happen on matrix-game cells.
                pass
            else:
                env_agent_action = int(history.agent_actions[-1])
                env_opp_action = int(history.opponent_actions[-1])
                # Sanity guard: the action the runner just sent into
                # env.step should match the agent_action env recorded.
                # If they disagree, the inner policy state is corrupt;
                # fall back to skipping observe rather than poisoning
                # the regret table with a misaligned (a, b) pair.
                if env_agent_action == a:
                    # See module docstring "Payoff-matrix wiring":
                    # we wrap the inner class so that ITS "agent" axis
                    # is the env's adversary and ITS "opponent" axis is
                    # us. Therefore swap roles in the observe call.
                    self._inner.observe(
                        agent_action=env_opp_action,   # env's adversary plays here
                        opponent_action=a,             # we (wrapper) played a
                        agent_reward=0.0,              # inner class reads payoff_opp from matrix
                        opponent_reward=r,
                        info=None,
                    )

        return {
            "td_target": 0.0,
            "td_error": 0.0,
            "v_next": 0.0,
            "reward": r,
            "q_abs_max_running": 0.0,
        }

    def end_episode(self, episode_index: int) -> Dict[str, Any]:
        """Return per-episode aggregates the runner consumes.

        Strict subset per spec §6.3 patch §3: the wrapper has no β /
        no Q-table / no TD error, so the runner-required fields
        (``length``, ``q_abs_max``, ``nan_count``,
        ``divergence_event``) are returned as zeros / False; the
        runner builds the per-episode arrays from these plus its own
        running tally of ``return``, ``epsilon`` (= 0), and
        ``bellman_residual`` (= 0 from the per-step diag this class
        emits).
        """
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"end_episode received {episode_index} but current episode "
                f"is {self._current_episode}"
            )
        return {
            "episode_index": int(self._current_episode),
            "return": float(self._ep_return),
            "length": int(self._ep_length),
            "q_abs_max": 0.0,
            "nan_count": 0,
            "divergence_event": False,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _current_history(self) -> GameHistory:
        """Return the env's running history, or an empty one."""
        if self._env_history_provider is None:
            return GameHistory()
        h = self._env_history_provider()
        if h is None:
            return GameHistory()
        return h


# ---------------------------------------------------------------------------
# Concrete wrappers
# ---------------------------------------------------------------------------
class RegretMatchingAgent(_StrategicAgentBase):
    """Regret-matching strategic-learning agent.

    Wraps :class:`RegretMatching` (canonical Hart & Mas-Colell, 2000
    cumulative-regret recursion) as a Phase VIII agent baseline.

    Hyperparameters
    ---------------
    ``mode``  : ``"full_info"`` (default; needs the matrix) or
                ``"realized_payoff"``.
    ``value_lr`` : per-action payoff EMA learning rate; only used in
                   ``mode='realized_payoff'``.

    Non-matrix fallback
    -------------------
    When ``payoff_agent`` is ``None`` (e.g. ``delayed_chain``) the
    wrapper does NOT instantiate ``RegretMatching`` and falls back to
    a uniform-random policy. See module docstring "DC-Long50
    handling".
    """

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_schedule: Callable[[int], float],
        rng: np.random.Generator,
        q_init: float = 0.0,
        payoff_agent: Optional[np.ndarray] = None,
        env_history_provider: Optional[Callable[[], GameHistory]] = None,
        seed: Optional[int] = None,
        mode: str = "full_info",
        value_lr: float = 0.05,
    ) -> None:
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=epsilon_schedule,
            rng=rng,
            q_init=q_init,
            payoff_agent=payoff_agent,
            env_history_provider=env_history_provider,
            seed=seed,
        )
        self._mode: str = str(mode)
        self._value_lr: float = float(value_lr)
        if self._matrix_game:
            self._inner = self._make_inner()

    def _make_inner(self) -> RegretMatching:
        assert self._inner_payoff is not None  # type-narrowing
        return RegretMatching(
            payoff_opponent=self._inner_payoff,
            mode=self._mode,
            value_lr=self._value_lr,
            n_actions=self._n_actions,
            seed=self._seed,
        )


class SmoothedFictitiousPlayAgent(_StrategicAgentBase):
    """Smoothed (logit / quantal) fictitious-play strategic-learning agent.

    Wraps :class:`SmoothedFictitiousPlay`: best-respond to the empirical
    distribution of the env-adversary's recent actions under a logit
    policy ``π(j) ∝ exp(Q_br[j] / temperature)``.

    Hyperparameters
    ---------------
    ``temperature``  : logit temperature τ > 0; spec §7.5 grid {0.05,
                       0.2, 1.0}. Default 0.2 (the middle of the grid;
                       conservative best-response sharpness).
    ``memory_m``    : empirical-belief window. ``None`` (default) =
                       unbounded fictitious play.

    Non-matrix fallback
    -------------------
    Same as :class:`RegretMatchingAgent`: uniform-random when
    ``payoff_agent is None``.
    """

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_schedule: Callable[[int], float],
        rng: np.random.Generator,
        q_init: float = 0.0,
        payoff_agent: Optional[np.ndarray] = None,
        env_history_provider: Optional[Callable[[], GameHistory]] = None,
        seed: Optional[int] = None,
        temperature: float = 0.2,
        memory_m: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=epsilon_schedule,
            rng=rng,
            q_init=q_init,
            payoff_agent=payoff_agent,
            env_history_provider=env_history_provider,
            seed=seed,
        )
        self._temperature: float = float(temperature)
        self._memory_m: Optional[int] = (
            None if memory_m is None else int(memory_m)
        )
        if self._matrix_game:
            self._inner = self._make_inner()

    def _make_inner(self) -> SmoothedFictitiousPlay:
        assert self._inner_payoff is not None  # type-narrowing
        return SmoothedFictitiousPlay(
            payoff_opponent=self._inner_payoff,
            temperature=self._temperature,
            memory_m=self._memory_m,
            n_actions=self._n_actions,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Fast-path ``select_action``: bypass the inner class's ``act`` and
    # compute the logit best-response directly against the env's
    # adversary's empirical play.
    #
    # Why bypass the inner class: ``SmoothedFictitiousPlay.act()`` reads
    # ``history.empirical_agent_policy(n_actions=self._n_agent_actions)``
    # under the convention "agent = the player whose actions populate
    # empirical_agent_policy". In our wrapper role-swap, the empirical
    # distribution of interest is the env-adversary's actions
    # (``history.opponent_actions``). The naive way is to construct a
    # role-swapped history and pass it to the inner class — but
    # rebuilding the history container every step is O(|history|) and
    # makes long runs (10k episodes × 20 steps × growing history)
    # quadratic in wallclock. Instead, we read the empirical
    # distribution directly via ``history.empirical_opponent_policy``
    # (already O(memory_m)) and apply the same logit policy formula
    # the inner class uses — see ``SmoothedFictitiousPlay._logit_policy``.
    #
    # The math is identical to what the inner class would compute on a
    # role-swapped history:
    #
    #     belief[b]  = empirical-frequency of env adversary action b
    #                  over the last ``memory_m`` steps
    #     Q_br[a]    = E_b[ payoff_agent[a, b] ] = (payoff_agent @ belief)[a]
    #     π(a)       ∝ exp(Q_br[a] / temperature)         (logit BR)
    #
    # which is the logit best-response of the wrapper to the env
    # adversary's empirical play. The shifted-max softmax form keeps
    # the exponentiation numerically stable for large |Q_br| / small τ
    # without invoking ``np.log-sum-exp``, which spec §2.1 reserves for
    # the shared TAB / safe weighted log-sum-exp operator (test
    # ``test_log-sum-exp_only_in_smoothed_fictitious_play`` enforces).
    # ------------------------------------------------------------------
    def select_action(self, state: Any, episode_index: int) -> int:  # noqa: D401
        del state, episode_index
        if not (self._matrix_game and self._payoff_agent is not None):
            return int(self._rng.integers(0, self._n_actions))

        history = self._current_history()
        n_env_adv_actions = int(self._payoff_agent.shape[1])
        belief = history.empirical_opponent_policy(
            m=self._memory_m, n_actions=n_env_adv_actions
        )  # shape (n_env_adv_actions,)
        # Q_br[a] = E_b[ payoff_agent[a, b] ]
        q_br = self._payoff_agent @ belief  # shape (n_my_actions,)
        # Stable softmax: subtract max before exp, then normalize.
        # Mathematically identical to log-sum-exp-based softmax; avoids
        # spec §2.1 violation by not calling log-sum-exp directly here.
        logits = q_br / self._temperature
        logits_shifted = logits - float(logits.max())
        pi = np.exp(logits_shifted)
        s = float(pi.sum())
        if s <= 0.0 or not np.isfinite(s):
            return int(self._rng.integers(0, self._n_actions))
        pi = pi / s
        return int(self._rng.choice(self._n_actions, p=pi))
