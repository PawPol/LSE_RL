#!/usr/bin/env python
"""Overnight run checkpoint state machine.

Manages persistent state for the /lse:overnight autonomous pipeline.
The checkpoint file tracks progress across Phase IV-A → IV-B → IV-C so
that a crashed session can resume from the last known-good state.

Usage from Claude Code subagents (via Bash tool):
    python scripts/overnight/checkpoint.py init
    python scripts/overnight/checkpoint.py get
    python scripts/overnight/checkpoint.py update --phase IV-A --status planning
    python scripts/overnight/checkpoint.py task-done --task "P4A-01"
    python scripts/overnight/checkpoint.py task-fail --task "P4A-01" --reason "verifier FAIL"
    python scripts/overnight/checkpoint.py gate --phase IV-A --result pass
    python scripts/overnight/checkpoint.py finish
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT_PATH = Path("tasks/overnight_checkpoint.json")
LOG_PATH = Path("tasks/overnight_log.md")

PHASES = ("IV-A", "IV-B", "IV-C")
VALID_STATUSES = (
    "pending", "planning", "implementing", "verifying",
    "reviewing", "gate_checking", "complete", "failed", "gate_failed",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load() -> dict:
    if not CHECKPOINT_PATH.exists():
        print("ERROR: No checkpoint file. Run 'init' first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(CHECKPOINT_PATH.read_text())


def _save(state: dict) -> None:
    state["last_checkpoint"] = _now()
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _log(message: str) -> None:
    """Append a timestamped line to the overnight log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _now()
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[{timestamp}] {message}\n")


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a fresh checkpoint."""
    start_phase = args.phase if hasattr(args, "phase") and args.phase else "IV-A"
    state = {
        "started_at": _now(),
        "current_phase": start_phase,
        "phase_status": {p: "pending" for p in PHASES},
        "task_queue": [],
        "completed_tasks": [],
        "failed_tasks": [],
        "failure_count": 0,
        "failure_budget": 3,
        "retry_counts": {},
        "gate_results": {},
        "codex_sessions": [],
        "auto_resolved_questions": [],
        "last_checkpoint": _now(),
    }
    if start_phase != "IV-A":
        # Mark earlier phases as assumed-complete
        for p in PHASES:
            if p == start_phase:
                break
            state["phase_status"][p] = "complete"
    _save(state)
    _log(f"Overnight run initialized. Starting phase: {start_phase}")
    # Initialize log file header
    with open(LOG_PATH, "w") as f:
        f.write(f"# Overnight Run — Phase IV\n")
        f.write(f"Started: {_now()}\n")
        f.write(f"Starting phase: {start_phase}\n")
        f.write(f"---\n")
    print(json.dumps(state, indent=2))


def cmd_get(args: argparse.Namespace) -> None:
    """Print current checkpoint state."""
    state = _load()
    print(json.dumps(state, indent=2))


def cmd_update(args: argparse.Namespace) -> None:
    """Update phase status."""
    state = _load()
    phase = args.phase
    status = args.status
    if phase not in PHASES:
        print(f"ERROR: Invalid phase '{phase}'. Must be one of {PHASES}", file=sys.stderr)
        sys.exit(1)
    if status not in VALID_STATUSES:
        print(f"ERROR: Invalid status '{status}'.", file=sys.stderr)
        sys.exit(1)
    state["phase_status"][phase] = status
    if status in ("planning", "implementing", "verifying", "reviewing", "gate_checking"):
        state["current_phase"] = phase
    _save(state)
    _log(f"Phase {phase} status → {status}")
    print(f"Phase {phase} → {status}")


def cmd_task_done(args: argparse.Namespace) -> None:
    """Mark a task as completed."""
    state = _load()
    task_id = args.task
    state["completed_tasks"].append({
        "task": task_id,
        "completed_at": _now(),
        "phase": state["current_phase"],
    })
    # Remove from queue if present
    state["task_queue"] = [t for t in state["task_queue"] if t.get("id") != task_id]
    _save(state)
    _log(f"Task completed: {task_id}")
    print(f"Task {task_id} marked complete")


def cmd_task_fail(args: argparse.Namespace) -> None:
    """Record a task failure."""
    state = _load()
    task_id = args.task
    reason = args.reason or "unknown"

    # Track retries
    retry_key = task_id
    retries = state["retry_counts"].get(retry_key, 0) + 1
    state["retry_counts"][retry_key] = retries

    state["failed_tasks"].append({
        "task": task_id,
        "failed_at": _now(),
        "phase": state["current_phase"],
        "reason": reason,
        "retry_number": retries,
    })
    state["failure_count"] += 1
    _save(state)
    _log(f"Task FAILED: {task_id} — {reason} (failure {state['failure_count']}/{state['failure_budget']}, retry #{retries})")

    budget_remaining = state["failure_budget"] - state["failure_count"]
    print(f"Task {task_id} failed. Budget remaining: {budget_remaining}")
    if budget_remaining <= 0:
        print("FAILURE BUDGET EXHAUSTED — phase must stop.")


def cmd_enqueue(args: argparse.Namespace) -> None:
    """Add tasks to the queue."""
    state = _load()
    # Expects JSON array on stdin or via --tasks
    tasks_json = args.tasks
    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError:
        # Treat as single task ID
        tasks = [{"id": tasks_json, "tag": "unknown", "deps": []}]
    state["task_queue"].extend(tasks)
    _save(state)
    _log(f"Enqueued {len(tasks)} tasks")
    print(f"Enqueued {len(tasks)} tasks. Queue size: {len(state['task_queue'])}")


def cmd_gate(args: argparse.Namespace) -> None:
    """Record a gate check result."""
    state = _load()
    phase = args.phase
    result = args.result.lower()
    detail = args.detail or ""

    state["gate_results"][phase] = {
        "result": result,
        "detail": detail,
        "checked_at": _now(),
    }

    if result == "pass":
        state["phase_status"][phase] = "complete"
        _log(f"Gate {phase}: PASS — {detail}")
        # Advance to next phase
        idx = PHASES.index(phase)
        if idx + 1 < len(PHASES):
            next_phase = PHASES[idx + 1]
            state["current_phase"] = next_phase
            _log(f"Advancing to {next_phase}")
            print(f"Gate PASS. Advancing to {next_phase}")
        else:
            _log("All phases complete!")
            print("Gate PASS. All phases complete!")
    else:
        state["phase_status"][phase] = "gate_failed"
        _log(f"Gate {phase}: FAIL — {detail}")
        print(f"Gate FAIL: {detail}. Pipeline stopped at {phase}.")

    _save(state)


def cmd_codex_session(args: argparse.Namespace) -> None:
    """Record a Codex session ID."""
    state = _load()
    state["codex_sessions"].append({
        "session_id": args.session_id,
        "type": args.type,
        "phase": state["current_phase"],
        "started_at": _now(),
    })
    _save(state)
    _log(f"Codex session started: {args.session_id} ({args.type})")
    print(f"Recorded Codex session {args.session_id}")


def cmd_auto_resolve(args: argparse.Namespace) -> None:
    """Record an auto-resolved open question."""
    state = _load()
    state["auto_resolved_questions"].append({
        "question": args.question,
        "resolution": args.resolution,
        "phase": state["current_phase"],
        "resolved_at": _now(),
    })
    _save(state)
    _log(f"Auto-resolved: {args.question} → {args.resolution}")
    print(f"Auto-resolved question logged")


def cmd_budget_check(args: argparse.Namespace) -> None:
    """Check if failure budget is exhausted."""
    state = _load()
    remaining = state["failure_budget"] - state["failure_count"]
    exhausted = remaining <= 0
    print(json.dumps({
        "failure_count": state["failure_count"],
        "failure_budget": state["failure_budget"],
        "remaining": remaining,
        "exhausted": exhausted,
    }))
    sys.exit(1 if exhausted else 0)


def cmd_finish(args: argparse.Namespace) -> None:
    """Finalize the overnight run with a summary report."""
    state = _load()
    state["finished_at"] = _now()
    _save(state)

    # Compute duration
    started = datetime.fromisoformat(state["started_at"])
    finished = datetime.fromisoformat(state["finished_at"])
    duration = finished - started
    hours = duration.total_seconds() / 3600

    # Build summary
    summary_lines = [
        "",
        "---",
        "",
        "## Final Report",
        f"Completed: {state['finished_at']}",
        f"Duration: {hours:.1f} hours",
        "",
    ]
    for p in PHASES:
        status = state["phase_status"].get(p, "pending")
        n_completed = sum(1 for t in state["completed_tasks"] if t["phase"] == p)
        n_failed = sum(1 for t in state["failed_tasks"] if t["phase"] == p)
        summary_lines.append(f"Phase {p}: {status} ({n_completed} tasks completed, {n_failed} failures)")

    summary_lines.append("")
    summary_lines.append("### Gate results")
    for p in PHASES:
        gr = state["gate_results"].get(p, {"result": "not_checked", "detail": ""})
        summary_lines.append(f"  {p}: {gr['result']} — {gr['detail']}")

    if state["auto_resolved_questions"]:
        summary_lines.append("")
        summary_lines.append(f"### Auto-resolved questions ({len(state['auto_resolved_questions'])})")
        for i, q in enumerate(state["auto_resolved_questions"], 1):
            summary_lines.append(f"  {i}. {q['question']} → {q['resolution']}")

    if state["failed_tasks"]:
        summary_lines.append("")
        summary_lines.append(f"### Unresolved failures ({len(state['failed_tasks'])})")
        for f in state["failed_tasks"]:
            summary_lines.append(f"  - [{f['phase']}] {f['task']} — {f['reason']}")

    summary_text = "\n".join(summary_lines)
    with open(LOG_PATH, "a") as logfile:
        logfile.write(summary_text + "\n")

    print(summary_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight checkpoint manager")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize checkpoint")
    p_init.add_argument("--phase", default="IV-A", choices=PHASES)

    sub.add_parser("get", help="Print current state")

    p_update = sub.add_parser("update", help="Update phase status")
    p_update.add_argument("--phase", required=True, choices=PHASES)
    p_update.add_argument("--status", required=True, choices=VALID_STATUSES)

    p_td = sub.add_parser("task-done", help="Mark task complete")
    p_td.add_argument("--task", required=True)

    p_tf = sub.add_parser("task-fail", help="Record task failure")
    p_tf.add_argument("--task", required=True)
    p_tf.add_argument("--reason", default="")

    p_eq = sub.add_parser("enqueue", help="Add tasks to queue")
    p_eq.add_argument("--tasks", required=True, help="JSON array or single task ID")

    p_gate = sub.add_parser("gate", help="Record gate result")
    p_gate.add_argument("--phase", required=True, choices=PHASES)
    p_gate.add_argument("--result", required=True, choices=["pass", "fail"])
    p_gate.add_argument("--detail", default="")

    p_cx = sub.add_parser("codex-session", help="Record Codex session")
    p_cx.add_argument("--session-id", required=True)
    p_cx.add_argument("--type", required=True, choices=["review", "adversarial"])

    p_ar = sub.add_parser("auto-resolve", help="Log auto-resolved question")
    p_ar.add_argument("--question", required=True)
    p_ar.add_argument("--resolution", required=True)

    sub.add_parser("budget-check", help="Check failure budget (exit 1 if exhausted)")

    sub.add_parser("finish", help="Finalize and write summary report")

    args = parser.parse_args()
    cmd_map = {
        "init": cmd_init,
        "get": cmd_get,
        "update": cmd_update,
        "task-done": cmd_task_done,
        "task-fail": cmd_task_fail,
        "enqueue": cmd_enqueue,
        "gate": cmd_gate,
        "codex-session": cmd_codex_session,
        "auto-resolve": cmd_auto_resolve,
        "budget-check": cmd_budget_check,
        "finish": cmd_finish,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
