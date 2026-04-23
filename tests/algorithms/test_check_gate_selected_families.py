"""Unit tests for ``scripts/overnight/check_gate.py`` selected_families parsing.

Covers NIT 17 (Phase IV-A review triage, closed 2026-04-22): the Phase IV-A
activation-gate check must distinguish a missing ``selected_families`` key
from an explicitly empty list.  Both cases fail the gate, but they produce
distinguishable ``details`` strings so operators can tell structural
(key missing) failures from content (empty list) failures.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CHECK_GATE_PATH = _REPO_ROOT / "scripts" / "overnight" / "check_gate.py"


def _load_check_gate():
    """Load check_gate.py as a module (it lives under scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("check_gate_mod", _CHECK_GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_gate_mod"] = module
    spec.loader.exec_module(module)
    return module


def _build_min_results_dir(root: pathlib.Path, selected_payload: object) -> pathlib.Path:
    """Create a minimal results_dir scaffold so check_gate_iva reaches the
    selected_tasks.json parse branch without crashing on earlier file lookups.

    Only ``task_search/selected_tasks.json`` is populated with the payload
    under test; every other expected file is intentionally left missing so
    the surrounding checks fail cleanly (their pass/fail state is irrelevant
    for this test, which asserts only on the selected_families check).
    """
    (root / "audit").mkdir(parents=True, exist_ok=True)
    (root / "task_search").mkdir(parents=True, exist_ok=True)
    (root / "counterfactual_replay").mkdir(parents=True, exist_ok=True)
    selected = root / "task_search" / "selected_tasks.json"
    selected.write_text(json.dumps(selected_payload))
    return root


def _find_family_check(checks: list[dict]) -> dict:
    """Return the dict for the 'At least one task family selected' check."""
    for c in checks:
        if c["condition"] == "At least one task family selected":
            return c
    raise AssertionError(
        "No 'At least one task family selected' check found in output. "
        f"Checks observed: {[c['condition'] for c in checks]}"
    )


def test_missing_selected_families_key_reports_structural_failure() -> None:
    """Fixture A: JSON without the ``selected_families`` key.

    The gate must fail AND the details string must indicate a missing key,
    not "0 families selected".
    """
    mod = _load_check_gate()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        # Dict without the "selected_families" key at all.
        _build_min_results_dir(root, selected_payload={"unrelated_field": 42})
        checks = mod.check_gate_iva(root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs")

    family_check = _find_family_check(checks)
    assert family_check["passed"] is False, (
        "Missing-key case must fail the gate"
    )
    assert "missing" in family_check["details"].lower(), (
        f"Missing-key details should surface the missing key; "
        f"got: {family_check['details']!r}"
    )
    # Must NOT look like the empty-list report (no phrase "0 families selected"
    # and no "present but empty" wording).
    assert "0 families selected" not in family_check["details"]
    assert "present but empty" not in family_check["details"]


def test_empty_selected_families_list_reports_empty_list_failure() -> None:
    """Fixture B: JSON with ``"selected_families": []`` (explicit empty list).

    The gate must fail AND the details string must indicate the list is
    present but empty, distinguishable from the missing-key failure.
    """
    mod = _load_check_gate()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _build_min_results_dir(root, selected_payload={"selected_families": []})
        checks = mod.check_gate_iva(root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs")

    family_check = _find_family_check(checks)
    assert family_check["passed"] is False, (
        "Empty-list case must fail the gate"
    )
    assert "empty" in family_check["details"].lower(), (
        f"Empty-list details should surface the empty content; "
        f"got: {family_check['details']!r}"
    )
    # Must NOT look like the missing-key report.
    assert "missing" not in family_check["details"].lower()


def test_missing_key_and_empty_list_details_are_distinguishable() -> None:
    """Direct comparison: the two failure-mode details strings must differ.

    This is the core NIT 17 acceptance condition.
    """
    mod = _load_check_gate()
    configs = _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs"

    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        root_a = pathlib.Path(tmp_a)
        root_b = pathlib.Path(tmp_b)
        _build_min_results_dir(root_a, selected_payload={"unrelated_field": 42})
        _build_min_results_dir(root_b, selected_payload={"selected_families": []})
        checks_a = mod.check_gate_iva(root_a, configs)
        checks_b = mod.check_gate_iva(root_b, configs)

    details_a = _find_family_check(checks_a)["details"]
    details_b = _find_family_check(checks_b)["details"]
    assert details_a != details_b, (
        f"Missing-key and empty-list must yield distinct details strings, "
        f"but both produced: {details_a!r}"
    )


def test_populated_selected_families_passes() -> None:
    """Regression guard: a non-empty list still passes the check with the
    original `'{N} families selected'` details string.
    """
    mod = _load_check_gate()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _build_min_results_dir(
            root,
            selected_payload={"selected_families": ["chain_sparse_credit", "grid_hazard"]},
        )
        checks = mod.check_gate_iva(
            root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs"
        )

    family_check = _find_family_check(checks)
    assert family_check["passed"] is True
    assert family_check["details"] == "2 families selected"
