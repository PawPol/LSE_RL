"""Phase VII-A regression: scripted-phase strategic vs original RPS env.

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§12.2 (regression: scripted RPS strategic implementation matches existing
RPS behavior within tolerance).

This is a TOLERANCE comparison, not byte-identity. The Phase VII-A RPS
env (``experiments/adaptive_beta/envs/rps.py``) re-seeds the opponent
RNG **per episode** via ``SeedSequence([seed, episode_index])``, while
the strategic ``ScriptedPhaseOpponent`` re-seeds once at adversary
``reset`` (per env-builder dispatch note). Across a 5000-step trace we
expect action-frequency statistics to agree within 5%.
"""

from __future__ import annotations

import numpy as np

from experiments.adaptive_beta.envs.rps import (
    PHASE_BIASED_EXPLOITABLE,
    PHASE_COUNTER_EXPLOIT,
    PHASE_UNIFORM_RANDOM,
    RPS,
    _OPPONENT_DIST,
)
from experiments.adaptive_beta.strategic_games.adversaries.scripted_phase import (
    ScriptedPhaseOpponent,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


def _drive_strategic_scripted(
    n_steps: int, *, switch_period: int, seed: int,
) -> np.ndarray:
    """Drive a ScriptedPhaseOpponent in step-mode against a fixed agent and
    return the opponent action stream.
    """
    adv = ScriptedPhaseOpponent(
        phase_policies=[
            _OPPONENT_DIST[PHASE_BIASED_EXPLOITABLE],
            _OPPONENT_DIST[PHASE_COUNTER_EXPLOIT],
            _OPPONENT_DIST[PHASE_UNIFORM_RANDOM],
        ],
        switch_period=switch_period,
        phase_names=[
            PHASE_BIASED_EXPLOITABLE,
            PHASE_COUNTER_EXPLOIT,
            PHASE_UNIFORM_RANDOM,
        ],
        mode="step",
        seed=seed,
    )
    h = GameHistory()
    actions = np.zeros(n_steps, dtype=np.int64)
    for t in range(n_steps):
        a = int(adv.act(h))
        actions[t] = a
        adv.observe(
            agent_action=0, opponent_action=a, agent_reward=0.0, opponent_reward=0.0,
        )
        h.append(agent_action=0, opponent_action=a, agent_reward=0.0)
    return actions


def _drive_rps_env(n_steps: int, *, horizon: int, seed: int) -> np.ndarray:
    """Drive the Phase VII-A RPS env and capture the opponent action stream."""
    env = RPS(horizon=horizon, switch_period_episodes=100, seed=seed)
    out = np.zeros(n_steps, dtype=np.int64)
    written = 0
    while written < n_steps:
        env.reset()
        for _ in range(horizon):
            _, _, done, info = env.step(0)
            out[written] = int(info["opponent_action"])
            written += 1
            if written >= n_steps:
                break
            if done:
                break
    return out


def test_scripted_strategic_action_freq_matches_rps_env_within_5pct() -> None:
    """`spec §12.2` — over 5000 steps, the empirical opponent-action
    distribution per phase agrees within 5% across the two implementations.

    Tolerance rationale (env-builder dispatch note): the Phase VII-A RPS
    env re-seeds the opponent RNG every episode via SeedSequence, while
    the strategic scripted_phase adversary re-seeds once at construction
    + once on reset. We therefore assert frequency alignment, NOT byte
    identity.
    """
    n_steps = 5000
    horizon = 20  # RPS default horizon
    # 5000 steps over horizon=20 → 250 episodes; switch_period_episodes=100
    # → 100 episodes per phase. To make the strategic step-mode adversary
    # cycle in the same way, we set switch_period = 100 * 20 = 2000 steps.
    rps_actions = _drive_rps_env(n_steps, horizon=horizon, seed=11)
    strat_actions = _drive_strategic_scripted(
        n_steps, switch_period=horizon * 100, seed=11,
    )
    assert rps_actions.shape == strat_actions.shape == (n_steps,)

    # Check empirical distribution per phase block (1 phase per 2000 steps).
    phase_block = horizon * 100  # = 2000 steps
    n_phases = n_steps // phase_block
    for p in range(n_phases):
        lo, hi = p * phase_block, (p + 1) * phase_block
        rps_freq = np.bincount(rps_actions[lo:hi], minlength=3) / float(hi - lo)
        strat_freq = np.bincount(strat_actions[lo:hi], minlength=3) / float(hi - lo)
        diff = np.max(np.abs(rps_freq - strat_freq))
        assert diff <= 0.05, (
            f"phase block {p}: max coordinate diff {diff:.3f} > 5% tolerance "
            f"(rps={rps_freq.tolist()}, strat={strat_freq.tolist()})"
        )


def test_scripted_strategic_phase_label_matches_rps_phase() -> None:
    """`spec §12.2` — phase labels emitted by ScriptedPhaseOpponent match
    those of the RPS env at every step block.
    """
    n_steps = 1200
    horizon = 20
    # Strategic: switch every horizon * switch_period_episodes steps.
    adv = ScriptedPhaseOpponent(
        phase_policies=[
            _OPPONENT_DIST[PHASE_BIASED_EXPLOITABLE],
            _OPPONENT_DIST[PHASE_COUNTER_EXPLOIT],
            _OPPONENT_DIST[PHASE_UNIFORM_RANDOM],
        ],
        switch_period=horizon * 100,
        phase_names=[
            PHASE_BIASED_EXPLOITABLE,
            PHASE_COUNTER_EXPLOIT,
            PHASE_UNIFORM_RANDOM,
        ],
        mode="step",
        seed=0,
    )
    env = RPS(horizon=horizon, switch_period_episodes=100, seed=0)
    h = GameHistory()
    written = 0
    while written < n_steps:
        env.reset()
        for _ in range(horizon):
            # Strategic adversary phase label at this step.
            strat_phase = adv.info()["phase"]
            rps_phase = env.current_phase
            assert strat_phase == rps_phase, (
                f"step {written}: strat phase {strat_phase!r} != rps phase {rps_phase!r}"
            )
            a = int(adv.act(h))
            adv.observe(
                agent_action=0, opponent_action=a, agent_reward=0.0,
                opponent_reward=0.0,
            )
            h.append(agent_action=0, opponent_action=a, agent_reward=0.0)
            env.step(0)
            written += 1
            if written >= n_steps:
                break
