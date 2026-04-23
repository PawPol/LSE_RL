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


# ---------------------------------------------------------------------------
# R2 (2026-04-22) — accept the runner's canonical schemas without declaring
# structural failure.  ``run_phase4_activation_search.py::_write_selected_tasks``
# emits a top-level list of task dicts (each with a ``family`` field);
# ``_write_frozen_suite_config`` emits ``{"selected_tasks": [...]}``;
# some 4a2 curations use ``{"tasks": [...]}``.  All three MUST be accepted.
# ---------------------------------------------------------------------------

def test_canonical_runner_top_level_list_of_tasks_accepted() -> None:
    """Fixture matching the current runner's ``_write_selected_tasks`` output:
    a top-level list of task dicts with ``family`` keys.  Must PASS.
    """
    mod = _load_check_gate()
    payload = [
        {"family": "dense_chain_cost", "idx": 0, "cfg": {}},
        {"family": "grid_hazard", "idx": 1, "cfg": {}},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _build_min_results_dir(root, selected_payload=payload)
        checks = mod.check_gate_iva(
            root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs"
        )
    family_check = _find_family_check(checks)
    assert family_check["passed"] is True, family_check
    assert "2 families selected" in family_check["details"]


def test_canonical_selected_tasks_dict_accepted() -> None:
    """Fixture matching ``{"selected_tasks": [...]}`` — the frozen
    activation-suite schema and the schema the review flagged as being
    rejected by the previous NIT-17 fix.  Must PASS.
    """
    mod = _load_check_gate()
    payload = {
        "phase": "IV-A",
        "status": "frozen",
        "selected_tasks": [
            {"family": "dense_chain_cost", "cfg": {}},
            {"family": "grid_hazard", "cfg": {}},
            {"family": "dense_chain_cost", "cfg": {}},  # duplicate family
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _build_min_results_dir(root, selected_payload=payload)
        checks = mod.check_gate_iva(
            root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs"
        )
    family_check = _find_family_check(checks)
    assert family_check["passed"] is True, family_check
    # De-duplication happens on the ``family`` field.
    assert "2 families selected" in family_check["details"]


def test_canonical_tasks_key_accepted() -> None:
    """Fixture matching ``{"tasks": [...]}`` — the manually-curated 4a2
    selected_tasks shape observed on disk.  Must PASS.
    """
    mod = _load_check_gate()
    payload = {
        "schema_version": "phase4A2.selected_tasks.v1",
        "tasks": [
            {"family": "dense_chain_cost"},
            {"family": "grid_hazard"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _build_min_results_dir(root, selected_payload=payload)
        checks = mod.check_gate_iva(
            root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs"
        )
    family_check = _find_family_check(checks)
    assert family_check["passed"] is True, family_check
    assert "2 families selected" in family_check["details"]


# ---------------------------------------------------------------------------
# R1 (spec §13.1, amended 2026-04-22) — four-condition primary formal gate.
# The gate MUST check not only mean_abs_u and frac_informative_u_ge_5e3, but
# also mean_abs_delta_discount_informative >= 1e-3 AND
# target_gap_norm_informative >= 5e-3.
# ---------------------------------------------------------------------------

def _build_replay_results_dir(
    root: pathlib.Path,
    replay_task: dict,
    design_point_u_pred: float = 1.0e-2,
    family: str = "dense_chain_cost",
) -> None:
    """Set up a results directory populated enough for GATE 1-4 to run.

    Writes:
    * task_search/selected_tasks.json (top-level list with a single family)
    * task_search/candidate_scores.csv (one row driving GATE 1)
    * counterfactual_replay/all_replay_summaries.json (one task dict)
    """
    (root / "audit").mkdir(parents=True, exist_ok=True)
    (root / "task_search").mkdir(parents=True, exist_ok=True)
    (root / "counterfactual_replay").mkdir(parents=True, exist_ok=True)

    # Selected tasks (top-level list, matches current runner output).
    (root / "task_search" / "selected_tasks.json").write_text(
        json.dumps([{"family": family, "idx": 0, "cfg": {}}])
    )

    # Design-point prediction for GATE 1.
    (root / "task_search" / "candidate_scores.csv").write_text(
        "family,mean_abs_u_pred\n"
        f"{family},{design_point_u_pred}\n"
    )

    # Replay summary for GATE 2a/2b/3/4.
    replay_task.setdefault("family", family)
    replay_task.setdefault("tag", f"{family}_0")
    (root / "counterfactual_replay" / "all_replay_summaries.json").write_text(
        json.dumps({"tasks": [replay_task]})
    )


def _find_check(checks: list[dict], prefix: str) -> dict:
    for c in checks:
        if c["condition"].startswith(prefix):
            return c
    raise AssertionError(
        f"No check starting with {prefix!r}; observed: "
        f"{[c['condition'] for c in checks]}"
    )


def test_gate3_fails_when_delta_discount_informative_too_small() -> None:
    """mean_abs_u and frac pass, but mean_abs_delta_discount_informative = 0.

    The gate MUST fail with [GATE 3] and the details must mention the
    discount-gap metric.
    """
    mod = _load_check_gate()
    replay = {
        "mean_abs_u_replay_informative": 1.0e-2,
        "median_abs_u_replay_informative": 1.0e-2,
        "frac_informative_u_ge_5e3": 0.8,
        "mean_abs_delta_discount_informative": 0.0,    # fails C3
        "target_gap_norm_informative": 1.0e-2,         # passes C4
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _build_replay_results_dir(root, replay)
        checks = mod.check_gate_iva(
            root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs"
        )
    gate3 = _find_check(checks, "[GATE 3]")
    assert gate3["passed"] is False, gate3
    assert "mean_abs_delta_discount" in gate3["condition"]
    # Detailed numeric surfaced.
    assert "best=0.000000" in gate3["details"]

    # The sibling Condition 4 check should still pass.
    gate4 = _find_check(checks, "[GATE 4]")
    assert gate4["passed"] is True

    # The per-family eligibility record must flag the family as ineligible
    # with the discount-gap reason populated.
    elig_check = None
    for c in checks:
        if "_eligibility_records" in c:
            elig_check = c
            break
    assert elig_check is not None
    recs = elig_check["_eligibility_records"]
    assert any(
        r["family"] == "dense_chain_cost"
        and r["iv_b_eligible"] is False
        and "mean_abs_delta_discount" in r["eligibility_reason"]
        for r in recs
    ), recs


def test_gate4_fails_when_target_gap_norm_informative_too_small() -> None:
    """mean_abs_u, frac, and delta-discount all pass, but
    target_gap_norm_informative = 0.  GATE 4 MUST fail.
    """
    mod = _load_check_gate()
    replay = {
        "mean_abs_u_replay_informative": 1.0e-2,
        "median_abs_u_replay_informative": 1.0e-2,
        "frac_informative_u_ge_5e3": 0.8,
        "mean_abs_delta_discount_informative": 5.0e-3,  # passes C3
        "target_gap_norm_informative": 0.0,             # fails C4
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _build_replay_results_dir(root, replay)
        checks = mod.check_gate_iva(
            root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs"
        )
    gate4 = _find_check(checks, "[GATE 4]")
    assert gate4["passed"] is False, gate4
    assert "target_gap_norm" in gate4["condition"]
    assert "best=0.000000" in gate4["details"]

    # Condition 3 still passes.
    gate3 = _find_check(checks, "[GATE 3]")
    assert gate3["passed"] is True

    # Eligibility reason surfaces the target-gap shortfall.
    elig_check = next(c for c in checks if "_eligibility_records" in c)
    recs = elig_check["_eligibility_records"]
    assert any(
        r["family"] == "dense_chain_cost"
        and r["iv_b_eligible"] is False
        and "target_gap_norm" in r["eligibility_reason"]
        for r in recs
    ), recs


def test_all_four_gates_pass_and_iv_b_eligible() -> None:
    """Regression: a replay payload that clears all four spec-§13.1
    conditions MUST surface PASS for [GATE 2a], [GATE 2b], [GATE 3], and
    [GATE 4] and mark the family ``iv_b_eligible=True``.
    """
    mod = _load_check_gate()
    replay = {
        "mean_abs_u_replay_informative": 1.0e-2,
        "median_abs_u_replay_informative": 1.0e-2,
        "frac_informative_u_ge_5e3": 0.8,
        "mean_abs_delta_discount_informative": 5.0e-3,
        "target_gap_norm_informative": 1.0e-2,
        "n_informative_transitions": 1000,
        "frac_informative_transitions": 0.9,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _build_replay_results_dir(root, replay)
        checks = mod.check_gate_iva(
            root, _REPO_ROOT / "experiments" / "weighted_lse_dp" / "configs"
        )
    for prefix in ("[GATE 1]", "[GATE 2a]", "[GATE 2b]", "[GATE 3]", "[GATE 4]"):
        c = _find_check(checks, prefix)
        assert c["passed"] is True, (prefix, c)

    elig_check = next(c for c in checks if "_eligibility_records" in c)
    recs = elig_check["_eligibility_records"]
    fam_rec = next(r for r in recs if r["family"] == "dense_chain_cost")
    assert fam_rec["iv_b_eligible"] is True, fam_rec
    # All four informative gate_pass_* flags set.
    info = fam_rec["informative_replay"]
    assert info["gate_pass_mean_or_median"] is True
    assert info["gate_pass_frac"] is True
    assert info["gate_pass_delta_discount"] is True
    assert info["gate_pass_target_gap"] is True
