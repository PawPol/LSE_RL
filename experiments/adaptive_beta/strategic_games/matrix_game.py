"""Generic matrix-game environment for Phase VII-B strategic learning.

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§5.1; parent ``docs/specs/phase_VII_adaptive_beta.md`` §22.2 (MushroomRL
``Environment`` subclass lock).

Design notes
------------
- MushroomRL's ``Environment`` exposes a ``info`` *property* that returns
  an ``MDPInfo`` object (used by every callback in
  ``mushroom_rl.core``). The Phase VII-B spec §5.1 also calls for an
  ``info()`` method on the env returning a metadata dict. Renaming the
  property would break Phase VII machinery; we therefore expose the
  game-level metadata via a ``game_info()`` method and keep the
  MushroomRL property intact. ``info`` (property) → ``MDPInfo``;
  ``game_info()`` (method) → spec §5.1 metadata dict.

  This deviation is recorded in the planner recap (``tasks/lessons.md``
  if a follow-up surfaces a callsite that explicitly expected the
  spec name).

- ``StateEncoder`` is a callable abstraction so that the same matrix
  game can be observed under different information regimes (single
  dummy state, last-action one-hot, rolling-statistics, etc.). The
  default encoder returns a single-state observation (analogous to
  the existing RPS env's hidden-phase mode), which keeps the
  observation-space cardinality at 1 and is compatible with the
  Phase VII tabular agent.

- Determinism contract:
  * the env's own RNG (``self._env_rng``) is re-seeded in ``reset``
    via ``SeedSequence([seed, episode_index])``, mirroring the
    existing RPS env;
  * the adversary's RNG is independent and is re-seeded once per
    construction unless ``reset(seed=...)`` propagates a new seed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


# A state encoder maps raw game context (current_step, history,
# adversary info, n_states declared) to a state vector AND declares a
# fixed observation cardinality up front (so the env can build its
# ``Discrete`` observation space at construction time).
StateEncoder = Callable[[Dict[str, Any]], np.ndarray]


def default_single_state_encoder(_ctx: Dict[str, Any]) -> np.ndarray:
    """Default encoder: collapse to a single dummy state ``[0]``.

    Compatible with the existing Phase VII RPS env (hidden-phase mode)
    and the tabular agent. ``Discrete(1)`` observation space.
    """
    return np.array([0], dtype=np.int64)


def make_default_state_encoder(
    horizon: int,
    n_actions: int,
) -> Tuple[StateEncoder, int]:
    """Build a default ``(timestep, prev_opponent_action)`` encoder.

    For ``H > 1`` repeated matrix games it is useful to expose the
    intra-episode timestep AND the previous opponent action to the
    tabular agent so that it can react within an episode. We encode the
    pair as a single flat ``int32`` index with cardinality
    ``horizon * (n_actions + 1)``: the ``+1`` slot accommodates the
    "no prior action yet" symbol used at step 0.

    Encoding (row-major, slot 0 = "no previous action"):

        s = step_in_episode * (n_actions + 1) + (prev_opp_action + 1)

    Parameters
    ----------
    horizon
        Episode length used when constructing the env.
    n_actions
        Number of opponent actions (i.e. the second dimension of the
        payoff matrices).

    Returns
    -------
    encoder, n_states
        ``encoder`` is a ``StateEncoder`` callable; ``n_states`` is the
        observation-space cardinality to pass to ``MatrixGameEnv``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_actions < 1:
        raise ValueError(f"n_actions must be >= 1, got {n_actions}")

    h = int(horizon)
    n = int(n_actions)
    n_states = h * (n + 1)
    no_action_slot = 0  # "no previous action" symbol for step 0

    def _encode(ctx: Dict[str, Any]) -> np.ndarray:
        # ``step_in_episode`` is the number of steps completed at the
        # time the encoder is invoked; ``reset`` sets it to 0 (post-reset)
        # and ``step`` calls the encoder after incrementing it. We clamp
        # to the absorbing-state cell so the index never exceeds n_states.
        step = int(ctx.get("step_in_episode", 0))
        if step < 0:
            step = 0
        if step >= h:
            step = h - 1
        prev = ctx.get("last_opponent_action", None)
        if prev is None:
            slot = no_action_slot
        else:
            prev_int = int(np.asarray(prev).flat[0])
            if not (0 <= prev_int < n):
                raise ValueError(
                    f"last_opponent_action {prev_int} out of range "
                    f"[0, {n}) for default state encoder"
                )
            slot = prev_int + 1
        idx = step * (n + 1) + slot
        return np.array([idx], dtype=np.int64)

    return _encode, n_states


class MatrixGameEnv(Environment):
    """MushroomRL-compatible repeated matrix-game environment.

    Parameters
    ----------
    payoff_agent
        Agent payoff matrix, shape ``(n_agent_actions, n_opponent_actions)``.
        ``payoff_agent[i, j]`` is the agent's reward when it plays
        ``i`` and the opponent plays ``j``.
    payoff_opponent
        Opponent payoff matrix, same shape. ``None`` is permitted for
        zero-sum games — in that case the env synthesises
        ``-payoff_agent`` internally.
    adversary
        ``StrategicAdversary`` instance. The env owns the adversary's
        lifecycle: it calls ``adversary.reset`` from ``env.reset``,
        ``adversary.act(...)`` from ``env.step``, and
        ``adversary.observe(...)`` after every step.
    horizon
        Episode length (number of agent actions per episode).
    state_encoder
        ``StateEncoder`` instance. Default = single-dummy-state encoder.
    n_states
        Observation-space cardinality (must match the encoder's image).
        Default = 1, consistent with the default encoder.
    seed
        Optional integer seed. Used to derive both the env's
        per-episode RNG (mirroring the RPS env's
        ``SeedSequence([seed, episode_index])`` scheme) and to seed
        the adversary if no adversary-level seed has been set.
    game_name
        Human-readable game id, surfaced in ``game_info()``.
    metadata
        Free-form dict of game-level metadata (e.g. action labels,
        canonical-sign flag). Merged into ``game_info()``.
    gamma
        Discount factor for the underlying ``MDPInfo``. Default 0.95
        to match the parent Phase VII RPS env.

    Information regime
    ------------------
    Default = payoff/action-observing but model-hidden:
    - the agent observes its own reward,
    - the agent observes the opponent's action via ``info["opponent_action"]``,
    - the agent does NOT observe the adversary's internal model state
      unless the encoder pulls fields out of ``info["adversary_info"]``.
    """

    # Default canonical-sign tag — overridden per-game in the registry.
    env_canonical_sign: Optional[str] = None

    def __init__(
        self,
        payoff_agent: np.ndarray,
        payoff_opponent: Optional[np.ndarray],
        adversary: StrategicAdversary,
        horizon: int,
        state_encoder: Optional[StateEncoder] = None,
        n_states: int = 1,
        seed: Optional[int] = None,
        game_name: str = "matrix_game",
        metadata: Optional[Dict[str, Any]] = None,
        gamma: float = 0.95,
    ) -> None:
        pa = np.asarray(payoff_agent, dtype=np.float64)
        if pa.ndim != 2:
            raise ValueError(
                f"payoff_agent must be 2-D, got shape {pa.shape}"
            )
        n_a, n_o = int(pa.shape[0]), int(pa.shape[1])

        if payoff_opponent is None:
            po = -pa  # zero-sum default
        else:
            po = np.asarray(payoff_opponent, dtype=np.float64)
            if po.shape != pa.shape:
                raise ValueError(
                    f"payoff_opponent shape {po.shape} != payoff_agent shape {pa.shape}"
                )

        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {n_states}")
        if adversary.n_actions != n_o:
            raise ValueError(
                f"adversary.n_actions={adversary.n_actions} != "
                f"payoff_opponent.shape[1]={n_o}"
            )

        self._payoff_agent: np.ndarray = pa  # shape (n_a, n_o)
        self._payoff_opponent: np.ndarray = po
        self._adversary: StrategicAdversary = adversary
        self._horizon: int = int(horizon)
        self._n_agent_actions: int = n_a
        self._n_opponent_actions: int = n_o
        self._state_encoder: StateEncoder = (
            state_encoder if state_encoder is not None else default_single_state_encoder
        )
        self._n_states: int = int(n_states)
        self._seed: Optional[int] = None if seed is None else int(seed)
        self._game_name: str = str(game_name)
        self._metadata: Dict[str, Any] = (
            {} if metadata is None else dict(metadata)
        )
        self._gamma: float = float(gamma)

        observation_space = spaces.Discrete(self._n_states)
        action_space = spaces.Discrete(self._n_agent_actions)
        mdp_info = MDPInfo(
            observation_space, action_space, self._gamma, self._horizon
        )
        super().__init__(mdp_info)

        # Mutable state.
        self._episode_index: int = 0
        self._step_in_episode: int = 0
        self._state: np.ndarray = np.array([0], dtype=np.int64)
        self._history: GameHistory = GameHistory()
        # Per-episode env RNG (used for any env-level stochasticity such
        # as trembles or hidden-type games; see games/rules_of_road.py).
        self._env_rng: np.random.Generator = np.random.default_rng()

        # Defer adversary reset until env reset is called for the first
        # time, so that the adversary picks up the propagated seed.

    # ------------------------------------------------------------------
    # Spec §5.1 surface (renamed `info` → `game_info` to avoid the
    # MushroomRL Environment.info @property collision; see module docstring).
    # ------------------------------------------------------------------
    def current_phase(self) -> str:
        """Adversary phase string for the current step.

        Falls back to ``"unknown"`` if the adversary's ``info()`` does
        not populate ``"phase"``.
        """
        adv_info = self._adversary.info()
        phase = adv_info.get("phase")
        return "unknown" if phase is None else str(phase)

    def game_info(self) -> Dict[str, Any]:
        """Spec §5.1 metadata dict (renamed from spec name ``info``).

        Returns a fresh dict on every call (safe to mutate at the call
        site).
        """
        out: Dict[str, Any] = {
            "game_name": self._game_name,
            "n_agent_actions": self._n_agent_actions,
            "n_opponent_actions": self._n_opponent_actions,
            "horizon": self._horizon,
            "gamma": self._gamma,
            "episode_index": self._episode_index,
            "step_in_episode": self._step_in_episode,
            "current_phase": self.current_phase(),
            "env_canonical_sign": self.env_canonical_sign,
            "adversary_type": self._adversary.adversary_type,
        }
        out.update(self._metadata)
        return out

    @property
    def adversary(self) -> StrategicAdversary:
        """Read-only access to the bound adversary."""
        return self._adversary

    @property
    def history(self) -> GameHistory:
        """Read-only access to the running game history."""
        return self._history

    # ------------------------------------------------------------------
    # MushroomRL Environment interface
    # ------------------------------------------------------------------
    def reset(
        self, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to the start of the next episode.

        Returns ``(state, info)`` per the Phase VII spec convention.
        ``state`` is a 1-D numpy ndarray; ``info`` is a dict.
        """
        # Re-seed the env-level RNG deterministically from the
        # (seed, episode_index) pair, mirroring the Phase VII RPS env.
        if self._seed is not None:
            ss = np.random.SeedSequence([self._seed, self._episode_index])
            self._env_rng = np.random.default_rng(ss)
        else:
            self._env_rng = np.random.default_rng()

        # Reset the adversary on the first episode and on any explicit
        # seed change. Subsequent resets do NOT clobber the adversary's
        # rolling state, because finite-memory / regret adversaries
        # accumulate across episodes (§7 spec convention).
        if self._episode_index == 0:
            adv_seed = self._seed
            self._adversary.reset(seed=adv_seed)
            self._history = GameHistory()

        self._step_in_episode = 0
        self._state = self._state_encoder(
            {
                "step_in_episode": 0,
                "episode_index": self._episode_index,
                "history": self._history,
                "adversary_info": self._adversary.info(),
                "n_states": self._n_states,
            }
        )

        info: Dict[str, Any] = {
            "phase": self.current_phase(),
            "is_shift_step": False,
            "catastrophe": False,
            "terminal_success": False,
            "opponent_action": None,
            "agent_action": None,
            "adversary_info": self._adversary.info(),
            "game_name": self._game_name,
            "episode_index": self._episode_index,
        }
        return self._state, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Normalise the agent action via the lessons.md pattern.
        agent_action: int = int(np.asarray(action).flat[0])
        if not (0 <= agent_action < self._n_agent_actions):
            raise ValueError(
                f"agent action must lie in [0, {self._n_agent_actions}), "
                f"got {agent_action}"
            )

        # Adversary acts on the current history (does NOT see this round's
        # agent action by default — model-hidden information regime).
        opponent_action_raw = self._adversary.act(
            self._history, agent_action=None
        )
        opponent_action: int = int(np.asarray(opponent_action_raw).flat[0])
        if not (0 <= opponent_action < self._n_opponent_actions):
            raise ValueError(
                f"adversary returned out-of-range action {opponent_action} "
                f"(n_opponent_actions={self._n_opponent_actions})"
            )

        # Realise rewards from the payoff matrices.
        agent_reward: float = float(self._payoff_agent[agent_action, opponent_action])
        opponent_reward: float = float(
            self._payoff_opponent[agent_action, opponent_action]
        )

        # Snapshot adversary info BEFORE observe(), since observe() may
        # mutate state-machine flags (e.g. hypothesis_testing.model_rejected).
        # We capture the info AFTER observe() so the env logger sees the
        # post-step adversary state, which is what spec §13 expects
        # ("adversary_info_json" reflects the just-completed transition).
        # Adversary updates its internal state from the realised step.
        self._adversary.observe(
            agent_action=agent_action,
            opponent_action=opponent_action,
            agent_reward=agent_reward,
            opponent_reward=opponent_reward,
            info=None,
        )
        adv_info_post = self._adversary.info()

        # If the adversary is a scripted-phase opponent in episode-mode,
        # forward end-of-episode hooks at the right time.
        self._history.append(
            agent_action=agent_action,
            opponent_action=opponent_action,
            agent_reward=agent_reward,
            opponent_reward=opponent_reward,
            info=adv_info_post,
        )

        self._step_in_episode += 1
        absorbing: bool = self._step_in_episode >= self._horizon

        info: Dict[str, Any] = {
            "phase": str(adv_info_post.get("phase") or "unknown"),
            "is_shift_step": False,
            "catastrophe": False,
            "terminal_success": False,
            "agent_action": agent_action,
            "opponent_action": opponent_action,
            "opponent_reward": opponent_reward,
            "adversary_info": adv_info_post,
            "game_name": self._game_name,
            "episode_index": self._episode_index,
        }

        # Update the cached state observation. The encoder sees the
        # full post-step context.
        self._state = self._state_encoder(
            {
                "step_in_episode": self._step_in_episode,
                "episode_index": self._episode_index,
                "history": self._history,
                "adversary_info": adv_info_post,
                "n_states": self._n_states,
                "last_agent_action": agent_action,
                "last_opponent_action": opponent_action,
            }
        )

        if absorbing:
            # Forward episode-end hook to phase-clock adversaries.
            on_ep_end = getattr(self._adversary, "on_episode_end", None)
            if callable(on_ep_end):
                on_ep_end()
            self._episode_index += 1

        return self._state, agent_reward, absorbing, info

    # ------------------------------------------------------------------
    # Optional MushroomRL hooks
    # ------------------------------------------------------------------
    def seed(self, seed: int) -> None:
        """Override the env seed (and propagate to the adversary).

        Resets the episode counter so the new seed takes effect from
        the next ``reset`` call.
        """
        self._seed = int(seed)
        self._episode_index = 0
        self._adversary.reset(seed=self._seed)
        self._history = GameHistory()

    def render(self, record: bool = False) -> None:
        # No-op: matrix games are not visual.
        return None
