"""Compact game-history container for finite-memory adversaries.

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§5.3.

A ``GameHistory`` records the per-step trace of a two-player matrix game
from the agent's perspective. It is consumed by every adversary that
conditions on the past (finite-memory best response, fictitious play,
regret matching, hypothesis testing, ...).

The container is pure Python / NumPy and has zero dependence on
MushroomRL or the matrix-game environment — adversaries can be unit
tested against hand-crafted histories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class GameHistory:
    """Trace of a two-player repeated matrix game from the agent POV.

    Attributes
    ----------
    agent_actions
        List of integer action indices played by the learning agent, in
        chronological order. ``len(agent_actions) == t`` after ``t``
        completed steps.
    opponent_actions
        Integer action indices played by the strategic adversary,
        co-indexed with ``agent_actions``.
    agent_rewards
        Per-step scalar rewards earned by the learning agent.
    opponent_rewards
        Per-step scalar rewards earned by the adversary. ``None``
        entries are permitted for partial-information settings; they
        propagate as ``np.nan`` in ``rolling_return``.
    info
        Per-step dict of side-information emitted by the environment or
        adversary (e.g. phase id, model_rejected, hypothesis_id). Free
        schema; consumers should defensively ``.get(...)``.

    Notes
    -----
    The container does NOT enforce equal lengths across the lists at
    insertion time — the caller is expected to push exactly one entry
    per list per step. Methods that need a coherent prefix raise
    ``ValueError`` if lengths disagree.
    """

    agent_actions: List[int] = field(default_factory=list)
    opponent_actions: List[int] = field(default_factory=list)
    agent_rewards: List[float] = field(default_factory=list)
    opponent_rewards: List[Optional[float]] = field(default_factory=list)
    info: List[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.agent_actions)

    def append(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[dict] = None,
    ) -> None:
        """Append a single step to the trace.

        Parameters are normalised: scalar numpy types are coerced to
        Python ``int`` / ``float`` for downstream determinism (so that
        ``GameHistory`` snapshots round-trip cleanly through JSON).
        """
        a = int(np.asarray(agent_action).flat[0])
        o = int(np.asarray(opponent_action).flat[0])
        r = float(np.asarray(agent_reward).flat[0])
        if opponent_reward is None:
            r_opp: Optional[float] = None
        else:
            r_opp = float(np.asarray(opponent_reward).flat[0])

        self.agent_actions.append(a)
        self.opponent_actions.append(o)
        self.agent_rewards.append(r)
        self.opponent_rewards.append(r_opp)
        self.info.append({} if info is None else dict(info))

    # ------------------------------------------------------------------
    # Spec §5.3 API
    # ------------------------------------------------------------------
    def last(self, m: int) -> "GameHistory":
        """Return a view of the last ``m`` steps as a new ``GameHistory``.

        If ``m <= 0`` returns an empty history. If ``m > len(self)`` the
        full history is returned (no padding).
        """
        if m <= 0:
            return GameHistory()
        n = len(self)
        if m >= n:
            # Return a shallow copy so the caller cannot mutate ``self``.
            return GameHistory(
                agent_actions=list(self.agent_actions),
                opponent_actions=list(self.opponent_actions),
                agent_rewards=list(self.agent_rewards),
                opponent_rewards=list(self.opponent_rewards),
                info=[dict(d) for d in self.info],
            )
        sl = slice(n - m, n)
        return GameHistory(
            agent_actions=list(self.agent_actions[sl]),
            opponent_actions=list(self.opponent_actions[sl]),
            agent_rewards=list(self.agent_rewards[sl]),
            opponent_rewards=list(self.opponent_rewards[sl]),
            info=[dict(d) for d in self.info[sl]],
        )

    def empirical_agent_policy(
        self,
        m: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """Empirical agent action distribution over the last ``m`` steps.

        Parameters
        ----------
        m
            Window length. ``None`` uses the full history.
        n_actions
            Action-space cardinality. ``None`` infers it from
            ``max(agent_actions) + 1``; on an empty history the empty
            array is returned.

        Returns
        -------
        probs : np.ndarray, shape ``(n_actions,)``
            Normalised frequency vector. If the window is empty,
            returns the uniform distribution over ``n_actions`` (or
            an empty array if ``n_actions`` cannot be inferred).
        """
        return _empirical_policy(
            self.agent_actions, m=m, n_actions=n_actions
        )

    def empirical_opponent_policy(
        self,
        m: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """Empirical opponent action distribution over the last ``m`` steps.

        See ``empirical_agent_policy`` for the contract. Adversaries
        such as fictitious play that observe agent actions invoke
        ``empirical_agent_policy`` instead — this getter is mainly for
        adversary self-introspection and metric pipelines.
        """
        return _empirical_policy(
            self.opponent_actions, m=m, n_actions=n_actions
        )

    def rolling_return(self, m: int) -> float:
        """Sum of agent rewards over the last ``m`` steps.

        ``None`` opponent-rewards are ignored (they don't enter this
        computation); but if any of the agent rewards in the window is
        non-finite the function propagates ``np.nan`` (consistent with
        the §9 metric-aggregation contract).
        """
        if m <= 0 or len(self.agent_rewards) == 0:
            return 0.0
        n = len(self.agent_rewards)
        window = self.agent_rewards[max(0, n - m):]
        arr = np.asarray(window, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        if not np.all(np.isfinite(arr)):
            return float("nan")
        return float(arr.sum())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _empirical_policy(
    actions: List[int],
    *,
    m: Optional[int],
    n_actions: Optional[int],
) -> np.ndarray:
    """Empirical action distribution over the last ``m`` items.

    Returns shape ``(n_actions,)``. If the window is empty and
    ``n_actions`` is given, returns the uniform distribution; if both
    are absent, returns an empty array.
    """
    if n_actions is None:
        if len(actions) == 0:
            return np.zeros(0, dtype=np.float64)
        n_actions = int(max(actions)) + 1
    n_actions = int(n_actions)

    if len(actions) == 0:
        if n_actions == 0:
            return np.zeros(0, dtype=np.float64)
        return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)

    if m is None or m >= len(actions):
        window: List[int] = list(actions)
    elif m <= 0:
        return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
    else:
        window = list(actions[-m:])

    arr = np.asarray(window, dtype=np.int64)
    counts = np.bincount(arr, minlength=n_actions).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
    return counts / total
