"""Hypothesis-testing adversary (spec §7.8 — PRIORITY).

State machine:

1. Hold a hypothesis ``h`` over the agent's action distribution.
2. Play a (smoothed) best response to ``h``.
3. Maintain a sliding test window of the last ``s`` agent actions.
4. If the empirical agent distribution over the window differs from
   ``h`` by more than tolerance ``τ`` (in total variation distance),
   reject the hypothesis.
5. Enter a search phase for ``search_len`` rounds, during which the
   adversary plays uniform-random.
6. Sample a new hypothesis (uniform on the simplex by default), reset
   ``model_rejected`` and ``search_phase`` flags, increment
   ``hypothesis_id``, and continue.

Logging contract (spec §7.8 last paragraph + §13):
- ``model_rejected``        : True for the single step where rejection fires.
- ``search_phase``          : True while in the post-rejection search phase.
- ``hypothesis_id``         : monotonically increasing counter.
- ``hypothesis_distance``   : TV distance between the latest empirical
                              window and the held hypothesis (extra key,
                              passed via ``_build_info(**extra)``).

Distance metric
---------------
Total variation:

    d_TV(p, h) = 0.5 * Σ_a |p[a] − h[a]|

This is the canonical choice for finite-action distributions and is
bounded in ``[0, 1]`` so the spec grid ``τ ∈ {0.025, 0.05, 0.10}``
makes immediate sense. The spec doesn't pin a metric explicitly; we
record this choice in the docstring (and emit ``"distance_metric":
"tv"`` in ``info()``) so it's auditable.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.adversaries.finite_memory_best_response import (
    _argmax_random,
    _softmax_stable,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two finite distributions."""
    return float(0.5 * np.abs(p - q).sum())


def _sample_simplex(rng: np.random.Generator, k: int) -> np.ndarray:
    """Uniform sample from the (k-1)-simplex via Dirichlet(1,...,1)."""
    if k <= 0:
        raise ValueError(f"_sample_simplex requires k >= 1, got {k}")
    return rng.dirichlet(np.ones(k, dtype=np.float64))


class HypothesisTestingAdversary(StrategicAdversary):
    """Hypothesis-testing opponent with rejection-triggered search phase.

    Parameters
    ----------
    payoff_opponent
        Shape ``(n_agent_actions, n_opponent_actions)``.
    test_window_s
        Window length ``s`` for the empirical agent distribution.
        Spec §7.8 grid: ``{100, 500, 1000}``.
    tolerance_tau
        TV-distance threshold ``τ`` for hypothesis rejection. Spec
        grid: ``{0.025, 0.05, 0.10}``.
    search_len
        Number of rounds spent in uniform-random search after a
        rejection. Spec grid: ``{10, 50, 100}``.
    temperature
        Logit temperature for the smoothed best response. Spec grid:
        ``{0.05, 0.2, 1.0}``. ``0.0`` → hard argmax with random
        tie-break.
    n_actions
        Opponent action-space cardinality.
    seed
        Integer seed for hypothesis sampling AND action sampling.
    initial_hypothesis
        Optional initial hypothesis over agent actions, shape
        ``(n_agent_actions,)``. Defaults to a uniform Dirichlet draw.
    """

    adversary_type: str = "hypothesis_testing"

    def __init__(
        self,
        payoff_opponent: np.ndarray,
        test_window_s: int,
        tolerance_tau: float,
        search_len: int,
        temperature: float = 0.05,
        n_actions: Optional[int] = None,
        seed: Optional[int] = None,
        initial_hypothesis: Optional[np.ndarray] = None,
    ) -> None:
        po = np.asarray(payoff_opponent, dtype=np.float64)
        if po.ndim != 2:
            raise ValueError(
                f"payoff_opponent must be 2-D, got shape {po.shape}"
            )
        if test_window_s <= 0:
            raise ValueError(f"test_window_s must be >= 1, got {test_window_s}")
        if not (0.0 < tolerance_tau <= 1.0):
            raise ValueError(
                f"tolerance_tau must lie in (0, 1], got {tolerance_tau}"
            )
        if search_len < 0:
            raise ValueError(f"search_len must be >= 0, got {search_len}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")

        self._payoff_opponent: np.ndarray = po
        self._n_agent_actions: int = int(po.shape[0])

        n = int(po.shape[1]) if n_actions is None else int(n_actions)
        if n != po.shape[1]:
            raise ValueError(
                f"n_actions={n} != payoff_opponent.shape[1]={po.shape[1]}"
            )
        super().__init__(n_actions=n, seed=seed)

        self._test_window_s: int = int(test_window_s)
        self._tolerance_tau: float = float(tolerance_tau)
        self._search_len: int = int(search_len)
        self._temperature: float = float(temperature)

        # Sliding window of the agent's most recent actions.
        self._window: Deque[int] = deque(maxlen=self._test_window_s)

        # Hypothesis state.
        if initial_hypothesis is not None:
            h0 = np.asarray(initial_hypothesis, dtype=np.float64)
            if h0.shape != (self._n_agent_actions,):
                raise ValueError(
                    f"initial_hypothesis shape must be ({self._n_agent_actions},), "
                    f"got {h0.shape}"
                )
            if np.any(h0 < 0.0) or abs(float(h0.sum()) - 1.0) > 1e-9:
                raise ValueError(
                    "initial_hypothesis must be a probability vector"
                )
            self._initial_hypothesis: Optional[np.ndarray] = h0 / h0.sum()
        else:
            self._initial_hypothesis = None

        # Mutable state populated by ``reset``.
        self._hypothesis: np.ndarray = np.full(
            self._n_agent_actions, 1.0 / self._n_agent_actions, dtype=np.float64
        )
        self._hypothesis_id: int = 0
        self._search_remaining: int = 0
        # Latches True for the single step where rejection fires; cleared
        # by the next ``act`` call.
        self._model_rejected: bool = False
        self._last_distance: float = 0.0
        self._last_policy: np.ndarray = np.full(
            self.n_actions, 1.0 / self.n_actions, dtype=np.float64
        )

    # ------------------------------------------------------------------
    # ABC interface
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._window.clear()
        if self._initial_hypothesis is not None:
            self._hypothesis = self._initial_hypothesis.copy()
        else:
            self._hypothesis = _sample_simplex(self._rng, self._n_agent_actions)
        self._hypothesis_id = 0
        self._search_remaining = 0
        self._model_rejected = False
        self._last_distance = 0.0
        self._last_policy = np.full(
            self.n_actions, 1.0 / self.n_actions, dtype=np.float64
        )

    @property
    def search_phase(self) -> bool:
        """True while still serving out the post-rejection search budget."""
        return self._search_remaining > 0

    def _resample_hypothesis(self) -> None:
        """Draw a fresh hypothesis and bump the id counter."""
        self._hypothesis = _sample_simplex(self._rng, self._n_agent_actions)
        self._hypothesis_id += 1

    def _br_policy(self) -> np.ndarray:
        """(Smoothed) best-response policy against the current hypothesis.

        Returns shape ``(n_actions,)``.
        """
        # Q_br[j] = E_a [payoff_opponent[a, j]] under the hypothesis.
        q_br = self._hypothesis @ self._payoff_opponent  # (n_opp_actions,)
        if self._temperature <= 0.0:
            chosen = _argmax_random(q_br, self._rng)
            policy = np.zeros(self.n_actions, dtype=np.float64)
            policy[chosen] = 1.0
            return policy
        logits = q_br / self._temperature
        return _softmax_stable(logits)

    def _uniform_policy(self) -> np.ndarray:
        return np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float64)

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        # Clear the one-step rejection flag at the start of each act —
        # the env logger reads info() AFTER act(), so the flag latched
        # in observe() must persist through this single act() call.
        # We achieve that by leaving the flag set if it was set during
        # the most recent observe(); the flag is cleared at the END
        # of act() AFTER the policy decision so info() between
        # observe() and the next observe() still reports True.
        in_search = self.search_phase
        if in_search:
            policy = self._uniform_policy()
            chosen = int(self._rng.choice(self.n_actions, p=policy))
            # Decrement the search budget for the round we are about
            # to play. Consistent with "search phase lasts search_len
            # rounds".
            self._search_remaining = max(0, self._search_remaining - 1)
        else:
            policy = self._br_policy()
            chosen = int(self._rng.choice(self.n_actions, p=policy))

        self._last_policy = policy
        return chosen

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        a = int(np.asarray(agent_action).flat[0])
        # Append to the agent-action sliding window.
        self._window.append(a)

        # Clear the previous-step rejection latch BEFORE testing again,
        # so model_rejected only ever holds for one round.
        self._model_rejected = False

        # Only test the hypothesis when the window is full — otherwise
        # the empirical estimate is too noisy to reject from. This is
        # consistent with the spec's "test window of length s".
        if (
            len(self._window) >= self._test_window_s
            and not self.search_phase
        ):
            window_arr = np.asarray(self._window, dtype=np.int64)
            counts = np.bincount(
                window_arr, minlength=self._n_agent_actions
            ).astype(np.float64)
            empirical = counts / counts.sum()  # shape (n_agent_actions,)
            d = _tv_distance(empirical, self._hypothesis)
            self._last_distance = d
            if d > self._tolerance_tau:
                # Reject hypothesis; enter search phase; resample.
                self._model_rejected = True
                self._search_remaining = self._search_len
                self._resample_hypothesis()
                # Optional: clear the window so post-rejection testing
                # uses fresh post-resample data. The spec doesn't
                # mandate this, but otherwise the very next test would
                # measure the just-resampled hypothesis against stale
                # window data and fire spuriously. We choose the
                # conservative interpretation and clear.
                self._window.clear()

    def info(self) -> Dict[str, Any]:
        # Convey hypothesis_distance as an extra key (the §5.2 mandatory
        # set carries the four state-machine flags / id / entropy).
        return self._build_info(
            phase="search" if self.search_phase else "exploit",
            memory_m=self._test_window_s,
            temperature=self._temperature,
            model_rejected=self._model_rejected,
            search_phase=self.search_phase,
            hypothesis_id=self._hypothesis_id,
            policy_entropy=self._entropy(self._last_policy),
            hypothesis_distance=float(self._last_distance),
            tolerance_tau=float(self._tolerance_tau),
            search_len=int(self._search_len),
            distance_metric="tv",
        )
