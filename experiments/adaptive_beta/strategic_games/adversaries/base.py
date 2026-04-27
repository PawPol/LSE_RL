"""Abstract base class for strategic-learning adversaries.

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§5.2.

Every concrete adversary must:
- be deterministic under fixed seed (Phase VII-B branch invariant),
- expose the full ``info()`` metadata key set declared in
  ``ADVERSARY_INFO_KEYS`` (use ``None`` for inapplicable fields,
  never omit a key),
- implement ``reset / act / observe / info``.

The ``act`` signature accepts an optional ``agent_action``: most
adversaries only see the agent's prior actions through the history,
but a few information-rich variants (e.g. simultaneous-move counterfactuals
in a debug mode) may peek at the current-round agent action.
Production-mode adversaries must NOT exploit this for cheating
(they should rely on history only); the parameter is kept available so
that test fixtures can construct degenerate / oracle adversaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.history import GameHistory


# Mandatory metadata keys per spec §5.2. Every concrete adversary's
# ``info()`` MUST return a dict whose key set is a superset of this.
ADVERSARY_INFO_KEYS: FrozenSet[str] = frozenset(
    {
        "adversary_type",
        "phase",
        "memory_m",
        "inertia_lambda",
        "temperature",
        "model_rejected",
        "search_phase",
        "hypothesis_id",
        "policy_entropy",
    }
)


def _validate_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Assert that ``info`` covers ``ADVERSARY_INFO_KEYS``.

    Returns ``info`` (for fluent use). Raises ``KeyError`` on missing
    fields — never silently fills, so violations are loud at test time.
    """
    missing = ADVERSARY_INFO_KEYS - set(info.keys())
    if missing:
        raise KeyError(
            f"StrategicAdversary.info() missing required keys: "
            f"{sorted(missing)}; spec §5.2 mandates the full key set"
        )
    return info


class StrategicAdversary(ABC):
    """Abstract base class for all Phase VII-B strategic adversaries.

    Concrete subclasses set ``adversary_type`` (string id) and any
    parameter attributes (``memory_m``, ``inertia_lambda``,
    ``temperature``, ``tau``, etc.) used by their ``act`` and
    ``observe`` logic.

    Determinism contract
    --------------------
    All sampling must go through ``self._rng`` (a
    ``numpy.random.Generator`` re-seeded by ``reset(seed=...)``). Do
    NOT call ``np.random.*`` module-level routines.
    """

    #: Class-level identifier surfaced in ``info()`` (override per subclass).
    adversary_type: str = "abstract"

    def __init__(self, n_actions: int, seed: Optional[int] = None) -> None:
        if n_actions <= 0:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")
        self.n_actions: int = int(n_actions)
        self._seed: Optional[int] = None if seed is None else int(seed)
        self._rng: np.random.Generator = np.random.default_rng(self._seed)
        # Subclasses populate any rolling-state attributes in their own
        # ``__init__`` and reset them in ``reset()``.

    # ------------------------------------------------------------------
    # Spec §5.2 API
    # ------------------------------------------------------------------
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the adversary's internal state.

        If ``seed`` is provided, it overrides the constructor seed and
        re-seeds the ``self._rng`` generator. Subclasses MUST clear any
        rolling caches (regret tables, hypothesis state, etc.) here so
        that re-running an experiment with the same seed produces a
        byte-identical action stream.
        """
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        """Return the adversary's next action.

        Parameters
        ----------
        history
            Past play. Adversaries that condition on the prefix
            (finite-memory, fictitious play, regret matching,
            hypothesis testing) read from this; stationary adversaries
            ignore it.
        agent_action
            Optional current-round agent action, available only when
            the environment is configured to reveal simultaneous-move
            information to the adversary (debug / oracle mode). Most
            adversaries pass ``None`` and rely solely on ``history``.

        Returns
        -------
        action : int
            Integer action index in ``[0, n_actions)``.
        """
        raise NotImplementedError

    @abstractmethod
    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the adversary's internal state after a completed step.

        The matrix-game environment calls this once per ``step`` after
        the joint action and rewards are realised. Adversaries that
        maintain regret tables, empirical estimates, or hypothesis
        state update them here.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Return the §5.2 metadata block.

        Concrete subclasses MUST include every key in
        ``ADVERSARY_INFO_KEYS``; missing fields surface as a loud
        ``KeyError`` via ``_validate_info``. Inapplicable fields use
        ``None`` (e.g. a stationary opponent has ``memory_m=None``).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers shared by subclasses
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(probs: np.ndarray) -> float:
        """Shannon entropy in nats. Robust to zeros."""
        p = np.asarray(probs, dtype=np.float64)
        if p.size == 0:
            return 0.0
        # Mask out zeros to avoid 0*log(0) warnings.
        nz = p > 0.0
        if not np.any(nz):
            return 0.0
        return float(-np.sum(p[nz] * np.log(p[nz])))

    def _build_info(
        self,
        *,
        phase: Optional[str] = None,
        memory_m: Optional[int] = None,
        inertia_lambda: Optional[float] = None,
        temperature: Optional[float] = None,
        model_rejected: Optional[bool] = None,
        search_phase: Optional[bool] = None,
        hypothesis_id: Optional[int] = None,
        policy_entropy: Optional[float] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build a §5.2-compliant info dict with the mandatory key set.

        Subclasses may pass additional ``extra`` fields (e.g.
        ``hypothesis_distance`` for hypothesis-testing). The mandatory
        keys are filled in directly; ``model_rejected`` and
        ``search_phase`` default to ``False`` (rather than ``None``)
        because they are boolean state machines — a stationary opponent
        has ``model_rejected=False`` always.
        """
        info_dict: Dict[str, Any] = {
            "adversary_type": self.adversary_type,
            "phase": phase,
            "memory_m": memory_m,
            "inertia_lambda": inertia_lambda,
            "temperature": temperature,
            "model_rejected": (
                False if model_rejected is None else bool(model_rejected)
            ),
            "search_phase": (
                False if search_phase is None else bool(search_phase)
            ),
            "hypothesis_id": (
                None if hypothesis_id is None else int(hypothesis_id)
            ),
            "policy_entropy": (
                None if policy_entropy is None else float(policy_entropy)
            ),
        }
        info_dict.update(extra)
        return _validate_info(info_dict)
