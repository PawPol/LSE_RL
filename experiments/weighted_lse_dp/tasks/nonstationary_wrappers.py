"""Nonstationary environment wrappers for Phase II regime-shift stress tasks.

Wrappers trigger a configurable structural change (goal position, reward
sign, slip probability) after a fixed number of episodes or training steps.
The state and action spaces are never modified by the wrapper.

Implemented here:
  - ``ChainRegimeShiftWrapper``  (spec S5.1.D)
  - ``GridRegimeShiftWrapper``   (spec S5.2.C)
"""

from __future__ import annotations

# Wrappers implemented by env-builder (tasks 6, 9).

__all__: list[str] = [
    "ChainRegimeShiftWrapper",
    "GridRegimeShiftWrapper",
]
