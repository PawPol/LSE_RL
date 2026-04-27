"""Phase VII-B strategic-games analysis subpackage.

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§§9.1, 10, 11.1, 15. Aggregator and event-aligned-panel builders only;
plotting is owned by ``plotter-analyst`` (§4 directory layout).
"""

from experiments.adaptive_beta.strategic_games.analysis.aggregate import (
    event_aligned_panel,
    load_run_summary,
    paired_diffs,
    promotion_gate,
)

__all__ = [
    "load_run_summary",
    "paired_diffs",
    "event_aligned_panel",
    "promotion_gate",
]
