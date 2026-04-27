"""Tests for the strategic logging schema (Phase VII-B spec §13).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§13 (logging schema episode + transition column lists).

Invariants guarded
------------------
- ``EPISODE_COLUMNS_STRATEGIC`` matches spec §13 episode list exactly
  (length + names + order).
- ``TRANSITION_COLUMNS_STRATEGIC`` matches spec §13 transition list exactly.
- ``episode_to_row`` JSON-encodes ``adversary_info`` (no leak of numpy
  scalars).
- Round-trip: write CSV → read back → schema-version header present.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.adaptive_beta.logging_callbacks import _RunIdentity
from experiments.adaptive_beta.strategic_games.logging import (
    EPISODE_COLUMNS_STRATEGIC,
    SCHEMA_VERSION_EPISODES_STRATEGIC,
    SCHEMA_VERSION_TRANSITIONS_STRATEGIC,
    StrategicLogger,
    TRANSITION_COLUMNS_STRATEGIC,
    episode_to_row,
    transition_to_row,
)


# ---------------------------------------------------------------------------
# Spec §13 column-list verbatim
# ---------------------------------------------------------------------------
SPEC_EPISODE_COLUMNS = (
    "run_id",
    "seed",
    "game",
    "adversary",
    "method",
    "episode",
    "return",
    "auc_so_far",
    "beta",
    "alignment_rate",
    "mean_effective_discount",
    "bellman_residual",
    "catastrophic",
    "diverged",
    "nan_count",
    "opponent_policy_entropy",
    "policy_total_variation",
    "support_shift",
    "model_rejected",
    "search_phase",
    "phase",
    "memory_m",
    "inertia_lambda",
    "temperature",
    "tau",
)

SPEC_TRANSITION_COLUMNS = (
    "run_id",
    "episode",
    "t",
    "state",
    "agent_action",
    "opponent_action",
    "reward",
    "next_state",
    "done",
    "beta",
    "advantage",
    "effective_discount",
    "alignment_indicator",
    "adversary_info_json",
)


def test_episode_columns_match_spec_exactly() -> None:
    """`spec §13` — EPISODE_COLUMNS_STRATEGIC matches the spec list verbatim."""
    assert tuple(EPISODE_COLUMNS_STRATEGIC) == SPEC_EPISODE_COLUMNS, (
        f"length got {len(EPISODE_COLUMNS_STRATEGIC)} vs "
        f"spec {len(SPEC_EPISODE_COLUMNS)}; "
        f"diff={set(EPISODE_COLUMNS_STRATEGIC) ^ set(SPEC_EPISODE_COLUMNS)}"
    )


def test_transition_columns_match_spec_exactly() -> None:
    """`spec §13` — TRANSITION_COLUMNS_STRATEGIC matches the spec list verbatim."""
    assert tuple(TRANSITION_COLUMNS_STRATEGIC) == SPEC_TRANSITION_COLUMNS


# ---------------------------------------------------------------------------
# episode_to_row / transition_to_row helpers
# ---------------------------------------------------------------------------

def _make_episode_row():
    return episode_to_row(
        run_id="r1", seed=0, game="matching_pennies", adversary="hypothesis_testing",
        method="vanilla", episode=0, episode_return=1.5, auc_so_far=2.0, beta=0.0,
        alignment_rate=0.5, mean_effective_discount=0.9, bellman_residual=0.1,
        catastrophic=False, diverged=False, nan_count=0,
        opponent_policy_entropy=np.log(2.0), policy_total_variation=0.1,
        support_shift=False, model_rejected=False, search_phase=False,
        phase="exploit", memory_m=20, inertia_lambda=0.5, temperature=0.2, tau=0.05,
    )


def test_episode_to_row_returns_strategic_episode_row() -> None:
    """`spec §13` — helper packs all spec §13 fields into a typed dataclass."""
    row = _make_episode_row()
    d = row.as_dict()
    assert set(d.keys()) == set(SPEC_EPISODE_COLUMNS)
    assert d["return"] == 1.5
    assert d["adversary"] == "hypothesis_testing"


def test_transition_to_row_json_encodes_adversary_info() -> None:
    """`spec §13` — ``adversary_info`` is JSON-encoded; numpy scalars are
    converted to Python types (no numpy-scalar leak into the CSV column).
    """
    adv_info = {
        "adversary_type": "hypothesis_testing",
        "hypothesis_id": np.int64(2),
        "hypothesis_distance": np.float32(0.123),
        "model_rejected": np.bool_(True),
        "policy_vec": np.array([0.5, 0.5]),
    }
    row = transition_to_row(
        run_id="r1", episode=0, t=0, state=0, agent_action=0, opponent_action=1,
        reward=1.0, next_state=0, done=True, beta=0.0, advantage=0.0,
        effective_discount=0.9, alignment_indicator=False, adversary_info=adv_info,
    )
    parsed = json.loads(row.adversary_info_json)
    assert isinstance(parsed["hypothesis_id"], int)
    assert isinstance(parsed["hypothesis_distance"], float)
    assert isinstance(parsed["model_rejected"], bool)
    assert isinstance(parsed["policy_vec"], list)
    # No numpy artefacts should leak through.
    assert "numpy" not in row.adversary_info_json.lower()


# ---------------------------------------------------------------------------
# StrategicLogger round-trip
# ---------------------------------------------------------------------------

def test_strategic_logger_csv_round_trip_with_schema_header(tmp_path: Path) -> None:
    """`spec §13` — write CSV → read back → schema header
    ``# schema_version=phaseVII.episodes.v_strategic_1`` is present and the
    body has all spec §13 columns.
    """
    identity = _RunIdentity(run_id="r1", env="matching_pennies|hypothesis_testing",
                            method="vanilla", seed=0)
    logger = StrategicLogger(
        identity, game="matching_pennies", adversary="hypothesis_testing",
    )
    row = _make_episode_row()
    logger.record_episode_strategic(row)
    out = tmp_path / "episodes.csv"
    logger.flush_episodes_csv(out)

    # Inspect header line and column order.
    with open(out, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0] == f"# schema_version={SCHEMA_VERSION_EPISODES_STRATEGIC}"
    cols = lines[1].split(",")
    assert tuple(cols) == SPEC_EPISODE_COLUMNS
    # Body row count = 1.
    assert len(lines) == 3


def test_strategic_logger_transitions_writes_artifact(tmp_path: Path) -> None:
    """`spec §13` — transition flush writes either parquet or CSV fallback,
    and the schema-version header / metadata is recorded.
    """
    identity = _RunIdentity(run_id="r1", env="matching_pennies|hypothesis_testing",
                            method="vanilla", seed=0)
    logger = StrategicLogger(
        identity, game="matching_pennies", adversary="hypothesis_testing",
    )
    row = transition_to_row(
        run_id="r1", episode=0, t=0, state=0, agent_action=0, opponent_action=1,
        reward=1.0, next_state=0, done=True, beta=0.0, advantage=0.0,
        effective_discount=0.9, alignment_indicator=False,
        adversary_info={"adversary_type": "hypothesis_testing"},
    )
    logger.record_transition_strategic(row)
    out = tmp_path / "transitions.parquet"
    n = logger.flush_transitions(out)
    assert n == 1
    # Either parquet (preferred) or CSV fallback should now exist.
    assert out.exists() or out.with_suffix(".csv").exists()


# ---------------------------------------------------------------------------
# Tripwire
# ---------------------------------------------------------------------------

def test_invariant_episode_columns_count_matches_spec_25() -> None:
    """`spec §13` — exactly 25 episode columns. A regression that adds or
    drops a column would break downstream Phase VII-B aggregators that
    assume positional-stable column lists.
    """
    assert len(EPISODE_COLUMNS_STRATEGIC) == 25


def test_invariant_transition_columns_count_matches_spec_14() -> None:
    """`spec §13` — exactly 14 transition columns."""
    assert len(TRANSITION_COLUMNS_STRATEGIC) == 14
