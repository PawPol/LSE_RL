"""Phase V WP0 consistency gate.

This test enforces the §7 WP0 fail-loud gates by re-running the
consistency audit (or consuming the latest
``results/audit/consistency_report.json``) and failing the suite on any
``severity == "BLOCKER"`` row.

Behaviour
---------
1. If ``results/audit/consistency_report.json`` is absent or older than
   the LaTeX paper source, the test regenerates it by invoking
   ``scripts/audit/run_consistency_audit.py``. This guarantees CI runs
   the gate on every invocation rather than relying on a stale artifact.
2. The report is loaded and every finding with ``severity == "BLOCKER"``
   is surfaced as a pytest failure. MINORs and INFOs do not block.
3. The ``schema_version`` string is asserted equal to ``"1.0.0"`` so the
   gate catches silent schema drift.

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


def test_no_blockers(consistency_report: dict) -> None:
    """Fail loudly on any BLOCKER finding."""
    blockers = [
        r for r in consistency_report["findings"]
        if r.get("severity") == "BLOCKER"
    ]
    if not blockers:
        return
    # Format a compact multi-line failure message
    msg_lines = [
        f"{len(blockers)} WP0 BLOCKER finding(s) in consistency_report.json:",
        "",
    ]
    for r in blockers[:20]:
        msg_lines.append(
            f"  [{r.get('id')}] {r.get('phase')}/{r.get('check')} "
            f"@ {r.get('artifact')}"
        )
        msg_lines.append(f"      expected: {r.get('expected')}")
        msg_lines.append(f"      actual:   {r.get('actual')}")
    if len(blockers) > 20:
        msg_lines.append(f"  ... ({len(blockers) - 20} more)")
    pytest.fail("\n".join(msg_lines))


def test_summary_counts_match_findings(consistency_report: dict) -> None:
    """Sanity: the summary block and the findings list agree."""
    summary = consistency_report["summary"]
    findings = consistency_report["findings"]
    counts = {"BLOCKER": 0, "MINOR": 0, "INFO": 0}
    for r in findings:
        sev = r.get("severity")
        if sev in counts:
            counts[sev] += 1
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
