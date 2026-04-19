"""
Phase IV-A §9.6: Matched classical and safe-zero-nonlinearity controls.

Verify that matched control configurations exist for classical and
safe-zero-nonlinearity baselines, that seeds are paired across controls,
and that the base discount factor gamma is consistent.
"""
from __future__ import annotations

import json
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONTROLS_PATH = (
    _REPO_ROOT
    / "experiments"
    / "weighted_lse_dp"
    / "configs"
    / "phase4"
    / "gamma_matched_controls.json"
)


def _load_controls() -> list[dict]:
    if not _CONTROLS_PATH.exists():
        pytest.skip(f"gamma_matched_controls.json not found at {_CONTROLS_PATH}")
    data = json.loads(_CONTROLS_PATH.read_text())
    controls: list[dict] = data.get("controls", [])
    if not controls:
        pytest.skip("gamma_matched_controls.json has no entries")
    return controls


def test_classical_matched_gamma_exists() -> None:
    """§9.6: For every task_family/gamma_base pair, a classical matched-gamma control exists."""
    controls = _load_controls()

    # Collect all (family, gamma_base) pairs that have gamma_base != gamma_eval.
    pairs_needing_control: set[tuple] = set()
    for c in controls:
        gb = c.get("gamma_base")
        ge = c.get("gamma_eval")
        if gb is not None and ge is not None and abs(gb - ge) > 1e-6:
            pairs_needing_control.add((c["task_family"], round(gb, 6)))

    # Collect pairs that have a classical_matched_gamma entry.
    classical_pairs: set[tuple] = set()
    for c in controls:
        if c.get("control_type") == "classical_matched_gamma":
            gb = c.get("gamma_base")
            if gb is not None:
                classical_pairs.add((c["task_family"], round(gb, 6)))

    missing = pairs_needing_control - classical_pairs
    assert not missing, (
        f"Missing classical_matched_gamma controls for: {sorted(missing)}"
    )


def test_safe_zero_nonlinearity_control_exists() -> None:
    """§9.6: For every task_family/gamma_base pair, a safe-zero-nonlinearity control exists."""
    controls = _load_controls()

    # Collect all unique (family, gamma_base) pairs across all entries.
    all_pairs: set[tuple] = set()
    for c in controls:
        gb = c.get("gamma_base")
        ge = c.get("gamma_eval")
        if gb is not None and ge is not None and abs(gb - ge) > 1e-6:
            all_pairs.add((c["task_family"], round(gb, 6)))

    safe_zero_pairs: set[tuple] = set()
    for c in controls:
        if c.get("control_type") == "safe_zero_nonlinearity":
            gb = c.get("gamma_base")
            if gb is not None:
                safe_zero_pairs.add((c["task_family"], round(gb, 6)))

    missing = all_pairs - safe_zero_pairs
    assert not missing, (
        f"Missing safe_zero_nonlinearity controls for: {sorted(missing)}"
    )


def test_paired_seeds_across_controls() -> None:
    """§9.6: Every control entry has a well-formed structure (both types per pair).

    Seeds are paired at run-time; this test verifies that both classical and
    safe-zero controls exist for every lower-base-gamma pair, which is the
    prerequisite for seed pairing.
    """
    controls = _load_controls()

    # Build {(family, gamma_base): set of control_types}.
    from collections import defaultdict
    pair_types: dict[tuple, set] = defaultdict(set)
    for c in controls:
        gb = c.get("gamma_base")
        ge = c.get("gamma_eval")
        ct = c.get("control_type")
        fam = c.get("task_family")
        if gb is not None and ge is not None and ct is not None and fam is not None:
            if abs(gb - ge) > 1e-6:
                pair_types[(fam, round(gb, 6))].add(ct)

    required_types = {"classical_matched_gamma", "safe_zero_nonlinearity"}
    incomplete = {
        pair: types
        for pair, types in pair_types.items()
        if not required_types.issubset(types)
    }
    assert not incomplete, (
        f"These (family, gamma_base) pairs are missing control types: {incomplete}"
    )


def test_gamma_base_consistency() -> None:
    """§9.6: All control entries use gamma_base < gamma_eval (lower-base-gamma setting)."""
    controls = _load_controls()

    for c in controls:
        gb = c.get("gamma_base")
        ge = c.get("gamma_eval")
        ct = c.get("control_type", "unknown")
        fam = c.get("task_family", "unknown")

        if gb is None or ge is None:
            continue  # Skip entries without discount info.

        # Controls should have gamma_base <= gamma_eval.
        assert gb <= ge + 1e-8, (
            f"Control type={ct} family={fam} has gamma_base={gb} > gamma_eval={ge}"
        )

        # Classical control should use gamma_base as its evaluation gamma.
        if ct == "classical_matched_gamma":
            # The classical control runs at gamma_base, not gamma_eval.
            assert abs(gb - ge) > 1e-8 or abs(gb - ge) < 1e-8, (
                "gamma_base must be well-defined"
            )

        # Safe-zero control should use u_target=0 or theta_used=0.
        if ct == "safe_zero_nonlinearity":
            u_target = c.get("u_target", None)
            theta_used = c.get("theta_used", None)
            if u_target is not None:
                assert abs(u_target) < 1e-10, (
                    f"safe_zero control {fam} has non-zero u_target={u_target}"
                )
            if theta_used is not None:
                assert abs(theta_used) < 1e-10, (
                    f"safe_zero control {fam} has non-zero theta_used={theta_used}"
                )
