"""Phase VIII tab-six-games runner namespace.

Each runner under this package is a CLI entry point that turns one
:mod:`experiments.adaptive_beta.tab_six_games.configs` YAML into a
fan-out of ``(game, subcase, method, seed)`` runs. Every run lands
under ``results/adaptive_beta/tab_six_games/raw/<run_id>/`` with the
Phase VIII run.json + metrics.npz contract (spec §8.1).

The package is intentionally empty so that tests / downstream consumers
can import it without triggering MushroomRL or NumPy work.
"""

from __future__ import annotations

__all__: list[str] = []
