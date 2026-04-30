"""Sign-switching-regime controller adversary (Phase VIII §5.7 / §10.5 / §M3).

Spec authority:
    docs/specs/phase_VIII_tab_six_games.md §5.7 (adversary catalogue),
    §6.6 (Oracle β reads ``info["regime"]``), §10.5 (Stage 4 sign-switching
    composite envs in M9), §M3 / §M9.

Role
----
This adversary is *the controller* for the hidden regime variable
ξ_t ∈ {"plus", "minus"} that drives the (G_+, G_-) sign-switching
composite environment to be wired in M9
(``tab_six_games/composites/sign_switching.py``). It exposes:

1. **A regime clock** in two modes:
   - ``mode="exogenous"`` flips ξ every ``dwell_episodes`` episodes,
     with ``dwell_episodes ∈ {100, 250, 500, 1000}`` (spec §5.7 / §10.5).
   - ``mode="endogenous"`` flips ξ when the rolling mean of the last
     ``trigger_window`` *agent rewards* crosses configurable thresholds:
     mean > ``trigger_threshold_high`` flips ``plus → minus`` (the game
     became favourable for the agent → switch to the harder one);
     mean < ``trigger_threshold_low`` flips ``minus → plus``
     (recovery: switch back to the easier one).
2. **An opponent action distribution** per regime. Each ξ-state has its
   own opponent policy callable so the M9 composite env can route
   ``opponent_action = adversary.act(history)`` and ``payoff[ξ]`` to a
   single integer per round. Defaults are uniform-random over the
   ``n_actions`` actions (callable-free, deterministic under ``self._rng``).

Regime exposure contract (spec §6.6)
------------------------------------
``info()["regime"]`` is the canonical hand-off to ``OracleBetaSchedule``.
The composite env will surface adversary metadata as
``step_info["adversary_info"]`` (the existing ``MatrixGameEnv``
convention; see ``MatrixGameEnv.step``) and the M9 composite is
expected to *also* lift ``regime`` to the top-level ``step_info``
itself. **Non-oracle schedules MUST NOT read ``info["regime"]``**;
the contract is enforced by the schedule factory, not by this
adversary (this module is read-only narrative metadata as per spec
§6.6 third paragraph).

Determinism
-----------
All sampling goes through ``self._rng`` (a ``numpy.random.Generator``
re-seeded by ``reset(seed=...)`` per the §5.2 base contract). Two
``SignSwitchingRegimeOpponent`` instances built with the same
constructor kwargs and seed produce a byte-identical action stream
(verified by the M3 W1.C smoke test).

Lessons consulted
-----------------
- ``tasks/lessons.md`` #1 (.venv / torch import discipline) — relevant
  only at run-time, not at module import.
- ``tasks/lessons.md`` #11 (numpy state extraction): all integer
  extractions go through ``int(np.asarray(x).flat[0])``. We do not
  call ``int(state)`` on raw inputs.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Legal regime labels (spec §5.7). Exposed publicly so test fixtures and
#: the M9 composite env can spell-check their lookup tables against this
#: tuple rather than re-typing the literals.
REGIMES: tuple = ("plus", "minus")

#: Legal switching modes.
_VALID_MODES: tuple = ("exogenous", "endogenous")

#: Allowed exogenous-dwell durations called out in spec §5.7 / §10.5.
#: We do NOT enforce membership: the spec lists ``{100, 250, 500, 1000}``
#: as the *grid* the M9 sweep is parameterised over, but unit tests and
#: dev runs use smaller values (e.g. dwell=3 in the M3 W1.C smoke).
ALLOWED_DWELL_GRID: tuple = (100, 250, 500, 1000)


# Type alias for a per-regime opponent policy. The callable receives the
# current ``GameHistory`` and must return an action index in
# ``[0, n_actions)``. Both ``act(history, agent_action=None)`` style and
# argument-less stub callables are not supported — keep the signature
# uniform with ``StrategicAdversary.act``.
OpponentPolicy = Callable[[GameHistory], int]


def _toggle_regime(regime: str) -> str:
    """Return the opposite regime label.

    Pure helper kept module-level so it can be referenced by tests and
    the M9 composite env without instantiating the adversary.
    """
    if regime == "plus":
        return "minus"
    if regime == "minus":
        return "plus"
    raise ValueError(
        f"unknown regime {regime!r}; expected one of {REGIMES}"
    )


# ---------------------------------------------------------------------------
# SignSwitchingRegimeOpponent
# ---------------------------------------------------------------------------

class SignSwitchingRegimeOpponent(StrategicAdversary):
    """Hidden-regime controller for the (G_+, G_-) composite env (M9).

    The adversary maintains a hidden ξ_t ∈ {"plus", "minus"} that
    determines which payoff matrix the composite env applies on each
    step. Per ``act`` call, the action is sampled from the opponent
    policy associated with the *currently active* regime.

    Parameters
    ----------
    n_actions
        Action-space cardinality. Both per-regime policies must return
        actions in ``[0, n_actions)``.
    mode
        ``"exogenous"`` (dwell-time clock) or ``"endogenous"`` (rolling
        agent-reward trigger). Default ``"exogenous"``.
    dwell_episodes
        Number of complete episodes between flips in exogenous mode.
        Default 250 (mid-grid). Spec §5.7 grid: ``{100, 250, 500, 1000}``.
    trigger_window
        Endogenous-mode rolling-window length over agent rewards.
        Default 50. Ignored in exogenous mode.
    trigger_threshold_high
        Endogenous-mode upper threshold: when the rolling mean exceeds
        this and the current regime is ``"plus"``, flip to ``"minus"``.
        Default ``+0.5``.
    trigger_threshold_low
        Endogenous-mode lower threshold: when the rolling mean falls
        below this and the current regime is ``"minus"``, flip back to
        ``"plus"``. Default ``-0.5``.
    initial_regime
        ``"plus"`` (default) or ``"minus"``.
    policy_plus_actions
        Optional ``Callable[[GameHistory], int]`` returning the
        opponent's action when ξ = "plus". Default ``None`` →
        uniform-random over the ``n_actions`` actions, sampled with
        ``self._rng``.
    policy_minus_actions
        Optional ``Callable[[GameHistory], int]`` for ξ = "minus".
        Default ``None`` → uniform-random.
    seed
        Integer seed for the action-sampling RNG (and for the
        per-regime defaults' uniform sampling).

    Notes
    -----
    The endogenous trigger consumes ONLY the rolling agent rewards
    pushed into ``observe(...)`` — *not* the contents of the supplied
    ``history`` argument to ``act``. This keeps the trigger consistent
    with what the M9 composite env actually emits to the adversary
    (one ``observe`` per realised step) and avoids double-counting
    when the agent provides a freshly-built ``GameHistory`` per call.
    """

    adversary_type: str = "sign_switching_regime"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        n_actions: int,
        mode: str = "exogenous",
        dwell_episodes: int = 250,
        trigger_window: int = 50,
        trigger_threshold_high: float = 0.5,
        trigger_threshold_low: float = -0.5,
        initial_regime: str = "plus",
        policy_plus_actions: Optional[OpponentPolicy] = None,
        policy_minus_actions: Optional[OpponentPolicy] = None,
        seed: Optional[int] = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {mode!r}"
            )
        if dwell_episodes <= 0:
            raise ValueError(
                f"dwell_episodes must be >= 1, got {dwell_episodes}"
            )
        if trigger_window <= 0:
            raise ValueError(
                f"trigger_window must be >= 1, got {trigger_window}"
            )
        if not np.isfinite(trigger_threshold_high) or not np.isfinite(
            trigger_threshold_low
        ):
            raise ValueError(
                "trigger thresholds must be finite, got "
                f"high={trigger_threshold_high}, low={trigger_threshold_low}"
            )
        if trigger_threshold_high < trigger_threshold_low:
            # Permitted in principle (asymmetric hysteresis is fine), but
            # a *strictly inverted* pair guarantees pathological flapping
            # because both branches fire simultaneously. Refuse loudly.
            raise ValueError(
                f"trigger_threshold_high ({trigger_threshold_high}) must be "
                f">= trigger_threshold_low ({trigger_threshold_low})"
            )
        if initial_regime not in REGIMES:
            raise ValueError(
                f"initial_regime must be one of {REGIMES}, got {initial_regime!r}"
            )

        super().__init__(n_actions=n_actions, seed=seed)

        # Configuration (immutable post-construction).
        self._mode: str = mode
        self._dwell_episodes: int = int(dwell_episodes)
        self._trigger_window: int = int(trigger_window)
        self._trigger_high: float = float(trigger_threshold_high)
        self._trigger_low: float = float(trigger_threshold_low)
        self._initial_regime: str = initial_regime

        # Per-regime opponent policies. ``None`` means "use the
        # built-in uniform-random fallback driven by self._rng".
        self._policy_plus: Optional[OpponentPolicy] = policy_plus_actions
        self._policy_minus: Optional[OpponentPolicy] = policy_minus_actions

        # Mutable state (re-initialised in reset()).
        self._regime: str = initial_regime
        self._episodes_in_current_regime: int = 0
        self._switches_so_far: int = 0
        # Deque of agent rewards over the rolling window. Bounded length
        # so old samples drop off automatically.
        self._reward_window: Deque[float] = deque(maxlen=self._trigger_window)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _uniform_action(self) -> int:
        """Draw a uniform random action in ``[0, n_actions)``.

        Uses ``self._rng`` so the action stream is deterministic under
        seed (per §5.2 contract).
        """
        return int(self._rng.integers(low=0, high=self.n_actions))

    def _sample_for_regime(self, history: GameHistory) -> int:
        """Dispatch action sampling to the regime-specific policy.

        Falls back to ``_uniform_action`` when the per-regime callable
        is ``None``. The returned integer is bounds-checked against
        ``self.n_actions``.
        """
        if self._regime == "plus":
            policy = self._policy_plus
        else:
            policy = self._policy_minus

        if policy is None:
            a = self._uniform_action()
        else:
            raw = policy(history)
            a = int(np.asarray(raw).flat[0])
        if not (0 <= a < self.n_actions):
            raise ValueError(
                f"opponent policy for regime {self._regime!r} returned "
                f"action {a} not in [0, {self.n_actions})"
            )
        return a

    def _rolling_reward_mean(self) -> float:
        """Mean of the last ``trigger_window`` agent rewards.

        Returns ``0.0`` when the window is empty (a neutral value that
        does not trigger either threshold, since by construction
        ``trigger_high >= trigger_low`` and the spec §5.7 default is
        ``+0.5 / -0.5``: 0.0 sits strictly between them).
        """
        if len(self._reward_window) == 0:
            return 0.0
        # Convert to ndarray once for a single dot-product mean.
        arr = np.asarray(self._reward_window, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return float("nan")
        return float(arr.mean())

    def _maybe_endogenous_flip(self) -> bool:
        """Apply the endogenous-trigger rule. Returns True iff flipped.

        The rule is anti-symmetric:
          - if regime == "plus" and rolling mean > high   → flip to "minus"
          - if regime == "minus" and rolling mean < low   → flip to "plus"
        Any other configuration leaves the regime unchanged. The window
        must be full (``len == trigger_window``) before the trigger can
        fire, otherwise a one-shot positive reward at episode-1 would
        cause an immediate flip.
        """
        if len(self._reward_window) < self._trigger_window:
            return False
        mean = self._rolling_reward_mean()
        if not np.isfinite(mean):
            return False  # Refuse to flip on NaN window (defensive).
        if self._regime == "plus" and mean > self._trigger_high:
            self._regime = _toggle_regime(self._regime)
            self._switches_so_far += 1
            self._episodes_in_current_regime = 0
            return True
        if self._regime == "minus" and mean < self._trigger_low:
            self._regime = _toggle_regime(self._regime)
            self._switches_so_far += 1
            self._episodes_in_current_regime = 0
            return True
        return False

    def _maybe_exogenous_flip(self) -> bool:
        """Apply the dwell-clock flip rule. Returns True iff flipped."""
        if self._episodes_in_current_regime >= self._dwell_episodes:
            self._regime = _toggle_regime(self._regime)
            self._switches_so_far += 1
            self._episodes_in_current_regime = 0
            return True
        return False

    # ------------------------------------------------------------------
    # Public hooks (called by the M9 composite env)
    # ------------------------------------------------------------------
    @property
    def regime(self) -> str:
        """Public view of the currently active ξ ∈ {"plus", "minus"}."""
        return self._regime

    def on_episode_end(self) -> None:
        """Advance the regime clock by one completed episode.

        Called by ``MatrixGameEnv`` (and the M9 composite that wraps it)
        at episode boundaries — i.e., immediately after a step with
        ``absorbing=True``. The episode counter is incremented FIRST
        (so the just-completed episode is "in" the current regime),
        then the mode-specific flip rule fires.
        """
        self._episodes_in_current_regime += 1
        if self._mode == "exogenous":
            self._maybe_exogenous_flip()
        else:
            self._maybe_endogenous_flip()

    # ------------------------------------------------------------------
    # ABC interface (spec §5.2)
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._regime = self._initial_regime
        self._episodes_in_current_regime = 0
        self._switches_so_far = 0
        # Replace the deque so its maxlen is honoured even if the
        # caller mutated _trigger_window between resets (we don't
        # advertise this, but be defensive).
        self._reward_window = deque(maxlen=self._trigger_window)

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        return self._sample_for_regime(history)

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Track rolling agent rewards for the endogenous-trigger rule
        # (also harmless to track in exogenous mode — keeps info()
        # output unified and lets the user reconfigure mode mid-run
        # in tests if they really want to).
        r = float(np.asarray(agent_reward).flat[0])
        self._reward_window.append(r)

    def info(self) -> Dict[str, Any]:
        # Spec §5.2 mandatory keys are filled by ``_build_info``; the
        # sign-switching-specific narrative fields are passed as
        # ``**extra``.
        return self._build_info(
            phase="sign_switching_regime",
            policy_entropy=None,  # depends on the per-regime policy
            regime=self._regime,
            mode=self._mode,
            switches_so_far=int(self._switches_so_far),
            episodes_in_current_regime=int(self._episodes_in_current_regime),
            rolling_reward=self._rolling_reward_mean(),
            dwell_episodes=int(self._dwell_episodes),
            trigger_window=int(self._trigger_window),
            trigger_threshold_high=float(self._trigger_high),
            trigger_threshold_low=float(self._trigger_low),
        )
