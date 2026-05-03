"""Phase VIII Stage 4 composite-environment subpackage (spec §10.5 / M9).

This package hosts MushroomRL ``Environment`` subclasses that wrap a
pair of underlying matrix-game envs and route per-step transitions
through one of them based on a hidden regime variable ξ_t. The first
member is :mod:`sign_switching` — the (G_+, G_-) sign-switching
composite mandated by the M9 oracle-validation gate.
"""

from experiments.adaptive_beta.tab_six_games.composites.sign_switching import (
    SignSwitchingComposite,
)

__all__ = ["SignSwitchingComposite"]
