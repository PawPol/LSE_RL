"""Hazard-cell environment wrappers for Phase II catastrophe stress tasks.

Wrappers add one or more hazard cells/transitions that give immediate large
negative rewards and optionally terminate the episode.  The base MDP
transition matrix and action space are unchanged; hazard events are
injected at step time.

Implemented here:
  - ``GridHazardWrapper``  (spec S5.2.B)
"""

from __future__ import annotations

# Wrappers implemented by env-builder (task 8).

__all__: list[str] = [
    "GridHazardWrapper",
]
