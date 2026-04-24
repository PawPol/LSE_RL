"""Phase V WP0 consistency gate.

This test enforces the spec section 7 WP0 fail-loud gates by re-running
the consistency audit (or consuming the latest
``results/audit/consistency_report.json``) and failing the suite on any
``severity == "BLOCKER"`` row that is owned by the Phase V pipeline.

Blocker semantics (task todo #11 remediation)
---------------------------------------------
A ``findings`` row is "blocking" if and only if **all** of the following
hold:

* ``severity == "BLOCKER"``.
* ``owner != "WP6"``. Rows with ``owner == "WP6"`` are paper-text vs
  tabled-number mismatches that the WP6 paper restructure will resolve;
  they are reported in the JSON/MD report but do NOT fail CI.

Rationale. The WP0 audit surfaces two distinct failure classes:

1. **Phase V pipeline bugs** -- e.g. the safe operator violates its
   certified stagewise rate on reachable mass. These MUST fail CI
   because they block every downstream WP (spec §10 execution order).
2. **Paper-text vs table mismatches** -- e.g. the paper quotes a 79%
   clip fraction but the recomputed value is 96.25%. These are not
   code defects; they are authorship errors the WP6 paper-restructure
   package will fix before submission. Failing CI on them would trap
   every pipeline PR behind an unrelated prose fix.

The splitting happens in the audit runner (see
``scripts/audit/run_consistency_audit.py::_tag_paper_cluster_ownership``).
This test is a thin consumer: it reads ``owner`` and the severity.

Behaviour
---------
1. If ``results/audit/consistency_report.json`` is absent or older than
   the LaTeX paper source, the test regenerates it by invoking
   ``scripts/audit/run_consistency_audit.py``. This guarantees CI runs
   the gate on every invocation rather than relying on a stale artifact.
2. The report is loaded. Rows with ``severity == "BLOCKER"`` and
   ``owner != "WP6"`` fail the suite. Paper-text blockers, MINORs, and
   INFOs do not block.
3. The ``schema_version`` string is asserted equal to ``"1.0.0"`` so the
   gate catches silent schema drift.
4. A reporting-only test surfaces the paper-text blocker count for
   visibility (but does not fail).

Environment
-----------
- ``LSE_AUDIT_FORCE_REGENERATE=1`` forces a regeneration even if the
  report exists and is fresh.
- ``LSE_AUDIT_SKIP_REGENERATE=1`` skips regeneration (use when running
  in a sandbox that cannot exec the audit script).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_AUDIT_DIR = _REPO_ROOT / "results" / "audit"
_REPORT_PATH = _AUDIT_DIR / "consistency_report.json"
_AUDIT_SCRIPT = _REPO_ROOT / "scripts" / "audit" / "run_consistency_audit.py"
_PAPER_TEX = _REPO_ROOT / "paper" / "neurips_selective_temporal_credit_assignment_positioned.tex"

_SCHEMA_VERSION = "1.0.0"

#: Owner tag used by the audit runner to flag rows handed off to the
#: WP6 paper-restructure work package. These BLOCKERs are reported but
#: non-blocking (see module docstring).
_PAPER_TEXT_OWNER = "WP6"


def _regenerate_report() -> None:
    """Invoke the audit script; propagate any non-zero exit code as a
    pytest failure."""
    cmd = [sys.executable, str(_AUDIT_SCRIPT)]
    env = os.environ.copy()
    # Make the subprocess tolerant: we want the report on disk even if
    # some sub-check errors out internally.
    r = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(
            "Audit script exited non-zero.\n"
            f"stdout:\n{r.stdout}\n\nstderr:\n{r.stderr}"
        )


@pytest.fixture(scope="module")
def consistency_report() -> dict:
    """Return the latest consistency report, regenerating if stale."""
    if os.environ.get("LSE_AUDIT_FORCE_REGENERATE") == "1":
        _regenerate_report()
    elif os.environ.get("LSE_AUDIT_SKIP_REGENERATE") == "1":
        pass
    elif not _REPORT_PATH.is_file():
        _regenerate_report()
    else:
        # Regenerate if paper source has been modified since the report.
        try:
            report_mtime = _REPORT_PATH.stat().st_mtime
            paper_mtime = _PAPER_TEX.stat().st_mtime \
                if _PAPER_TEX.is_file() else 0.0
            if paper_mtime > report_mtime:
                _regenerate_report()
        except OSError:
            _regenerate_report()

    assert _REPORT_PATH.is_file(), (
        f"consistency_report.json missing at {_REPORT_PATH} after "
        "regeneration attempt"
    )
    with open(_REPORT_PATH, "r") as fh:
        return json.load(fh)


def test_report_has_expected_schema(consistency_report: dict) -> None:
    assert consistency_report.get("schema_version") == _SCHEMA_VERSION, (
        f"schema_version drift: got "
        f"{consistency_report.get('schema_version')!r}, "
        f"expected {_SCHEMA_VERSION!r}"
    )
    assert "summary" in consistency_report, "report missing 'summary' block"
    assert "findings" in consistency_report, "report missing 'findings' block"
    summary = consistency_report["summary"]
    for k in ("blockers", "minors", "infos", "phases_completed"):
        assert k in summary, f"summary missing key '{k}'"
    # New keys added by the WP0 remediation pass (task #11).
    for k in ("phase_v_blockers", "paper_text_blockers"):
        assert k in summary, (
            f"summary missing key '{k}' (added by WP0 remediation)"
        )


def test_no_phase_v_blockers(consistency_report: dict) -> None:
    """Fail on any BLOCKER row not owned by the WP6 paper restructure.

    Paper-text BLOCKERs (``owner == "WP6"``) are reported by
    ``test_paper_text_blockers_reported_only`` below but do not fail CI.
    """
    phase_v_blockers = [
        r for r in consistency_report["findings"]
        if r.get("severity") == "BLOCKER"
        and r.get("owner", "") != _PAPER_TEXT_OWNER
    ]
    if not phase_v_blockers:
        return
    # Format a compact multi-line failure message
    msg_lines = [
        f"{len(phase_v_blockers)} Phase V BLOCKER finding(s) in "
        "consistency_report.json (paper-text BLOCKERs excluded; see "
        f"owner=={_PAPER_TEXT_OWNER!r}):",
        "",
    ]
    for r in phase_v_blockers[:20]:
        msg_lines.append(
            f"  [{r.get('id')}] {r.get('phase')}/{r.get('check')} "
            f"@ {r.get('artifact')}"
        )
        msg_lines.append(f"      expected: {r.get('expected')}")
        msg_lines.append(f"      actual:   {r.get('actual')}")
        if r.get("note"):
            msg_lines.append(f"      note:     {r.get('note')}")
    if len(phase_v_blockers) > 20:
        msg_lines.append(f"  ... ({len(phase_v_blockers) - 20} more)")
    pytest.fail("\n".join(msg_lines))


def test_paper_text_blockers_reported_only(consistency_report: dict) -> None:
    """Surface WP6 paper-text BLOCKERs for visibility (non-blocking).

    This test never fails; it only prints the paper-text BLOCKER count
    so ``pytest -v`` output records the handoff state. The WP6 paper
    restructure is expected to clear these.
    """
    paper_blockers = [
        r for r in consistency_report["findings"]
        if r.get("severity") == "BLOCKER"
        and r.get("owner", "") == _PAPER_TEXT_OWNER
    ]
    summary = consistency_report.get("summary", {})
    # Sanity: the summary count should match the row count.
    assert summary.get("paper_text_blockers", 0) == len(paper_blockers), (
        f"summary.paper_text_blockers = {summary.get('paper_text_blockers')}"
        f" but findings contain {len(paper_blockers)} rows with "
        f"owner=={_PAPER_TEXT_OWNER!r}"
    )
    # Report only -- do not fail.
    if paper_blockers:
        print(
            f"\n[audit-gate] {len(paper_blockers)} paper-text BLOCKER(s) "
            f"deferred to WP6:"
        )
        for r in paper_blockers:
            print(f"  [{r.get('id')}] {r.get('check')} @ {r.get('artifact')}")


def test_summary_counts_match_findings(consistency_report: dict) -> None:
    """Sanity: the summary block and the findings list agree."""
    summary = consistency_report["summary"]
    findings = consistency_report["findings"]
    counts = {"BLOCKER": 0, "MINOR": 0, "INFO": 0}
    phase_v_blockers = 0
    paper_text_blockers = 0
    for r in findings:
        sev = r.get("severity")
        if sev in counts:
            counts[sev] += 1
        if sev == "BLOCKER":
            if r.get("owner", "") == _PAPER_TEXT_OWNER:
                paper_text_blockers += 1
            else:
                phase_v_blockers += 1
    assert summary["blockers"] == counts["BLOCKER"], (
        f"summary.blockers={summary['blockers']} but findings contain "
        f"{counts['BLOCKER']} BLOCKERs"
    )
    assert summary["minors"] == counts["MINOR"], (
        f"summary.minors={summary['minors']} but findings contain "
        f"{counts['MINOR']} MINORs"
    )
    assert summary["infos"] == counts["INFO"], (
        f"summary.infos={summary['infos']} but findings contain "
        f"{counts['INFO']} INFOs"
    )
    assert summary.get("phase_v_blockers", 0) == phase_v_blockers, (
        f"summary.phase_v_blockers={summary.get('phase_v_blockers')} but "
        f"findings contain {phase_v_blockers} Phase V BLOCKERs"
    )
    assert summary.get("paper_text_blockers", 0) == paper_text_blockers, (
        f"summary.paper_text_blockers={summary.get('paper_text_blockers')}"
        f" but findings contain {paper_text_blockers} WP6 BLOCKERs"
    )


def test_remediation_summary_present(consistency_report: dict) -> None:
    """The WP0 remediation block should be present with expected keys.

    Added by task #11: the summary must expose the remediation counts so
    humans can reason about downgrade/retention state without re-reading
    every ``findings`` row.
    """
    summary = consistency_report["summary"]
    remediation = summary.get("remediation")
    assert isinstance(remediation, dict), (
        "summary.remediation missing (added by WP0 task #11)"
    )
    for key in (
        "flagged_dp_runs", "downgraded", "retained_blockers",
        "remediation_errors", "requires_phase_iii_fix",
    ):
        assert key in remediation, (
            f"summary.remediation missing key {key!r}"
        )
