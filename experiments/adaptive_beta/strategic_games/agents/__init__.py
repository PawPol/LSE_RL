"""Strategic-learning agent baselines (Phase VIII M7.2).

Public exports for the wrapper-agent classes that promote the existing
opponent classes (regret matching, smoothed fictitious play) to the
agent role for use as baselines in the Phase VIII Stage 2 dispatcher.

Spec authority
--------------
``docs/specs/phase_VIII_tab_six_games.md`` §6.3 patch-2026-05-01 §3
(strategic-learning agent baselines, promoted from M11 to M7).
"""

from __future__ import annotations

from experiments.adaptive_beta.strategic_games.agents.strategic_learning_agents import (
    RegretMatchingAgent,
    SmoothedFictitiousPlayAgent,
)

__all__ = [
    "RegretMatchingAgent",
    "SmoothedFictitiousPlayAgent",
]
