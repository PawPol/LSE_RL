# Phase VIII Autonomous Harness Configuration

**Document version:** v1 (2026-04-30).
**Scope:** Phase VIII (Six-Game Safe TAB Experiment Suite), milestones M1–M12.
**Branch:** `phase-VIII-tab-six-games-2026-04-30`.
**HEAD at kickoff:** `5d887df9` (or whatever the current HEAD is — verify in step 0).
**Authority:** This file is an autonomy overlay on `AGENTS.md` for Phase VIII
only. `AGENTS.md` invariants remain in force; this file *narrows* autonomy
boundaries and *pins* model routing. Where this file conflicts with
`AGENTS.md`, this file wins for Phase VIII milestones; outside Phase VIII,
`AGENTS.md` is authoritative.

---

## 0. Purpose

This file defines the autonomy regime under which the orchestrator may
dispatch Phase VIII subagents without per-task user approval. It pins:

1. Tier-by-tier autonomy boundaries (which milestones run unattended,
   which require human gates).
2. Model routing per agent role.
3. Auto-resolution rules per task tag.
4. Failure budget, stop conditions, and hard human gates.
5. Checkpoint, logging, and resumption protocol.
6. The kickoff prompt the user pastes into Claude Code.

Outside this regime — i.e., for any work past M12 or any non-Phase-VIII work
— the orchestrator reverts to AGENTS.md interactive mode.

---

## 1. Tiered Autonomy Definition

Phase VIII milestones are partitioned into three tiers:

### 1.1 Tier 1 — Full Autonomy

**Milestones:** M1, M2, M3, M4, M5.
**Character:** Implementation tier — building modules, writing tests,
verifying existing infrastructure. Decisions are engineering-flavored, not
paper-strategy.

The orchestrator may dispatch any subagent whose tasks fall under Tier 1
without user approval, subject to the rules in §3 (auto-resolution), §4
(failure budget), and §5 (stop conditions).

Codex reviews fire automatically at:
- end of M4 if any operator or stable-infrastructure file was modified
  (per AGENTS.md and Phase VIII spec §11.2 operator focus)
- end of M5 verifier gate (schema parity check)

### 1.2 Tier 2 — Autonomous with Hard Human Gates

**Milestones:** M6, M7, M10. (M11 is gated entry — see Tier 3.)
**Character:** Experimental tier — running ablations, producing main-pass
results.

The orchestrator may run dev passes autonomously. Before any dev → main
escalation, the orchestrator MUST halt and request user authorization. Codex
review fires automatically at end of milestone if results are flagged for
main-paper inclusion.

Hard gates within Tier 2:

| Milestone | Gate                                                     |
| --------- | -------------------------------------------------------- |
| M6        | dev β-sweep complete → user authorizes main pass         |
| M6        | main pass complete → Codex review → user signs off       |
| M7        | dev pass complete → user authorizes main pass            |
| M7        | main pass complete → Codex review → user signs off       |
| M10       | dev pass complete → user authorizes main pass            |
| M10       | main pass complete → Codex adversarial review (mandatory)|

### 1.3 Tier 3 — No Autonomy

**Milestones:** M8, M9, M11, M12.
**Character:** Analytical and paper-strategy tier — sign-specialization
identification, oracle-composite validity, optional-method authorization,
final paper-update recommendation.

The orchestrator may PREPARE Tier 3 deliverables (analysis tables,
recommendation drafts, plots) but MAY NOT mark any Tier 3 milestone complete
without explicit user authorization at the milestone close.

Hard gates:

| Milestone | Gate                                                                           |
| --------- | ------------------------------------------------------------------------------ |
| M8        | sign-specialization table draft + recommendation memo → user accepts/rewrites  |
| M9        | oracle-composite validation → user accepts validity → adaptive comparison runs |
| M11       | user explicitly authorizes appendix-method work BEFORE any subagent dispatch   |
| M12       | final-recommendation memo + paper-update patches drafted → user signs off      |

---

## 2. Model Routing

### 2.1 Opus 4.7 (orchestration, planning, engineering)

Used for:

```text
planner
env-builder
algo-implementer
operator-theorist
calibration-engineer
experiment-runner
plotter-analyst
```

Rationale: these roles produce code, design specifications, and orchestrate
work. Code generation, mathematical derivation, and orchestration are
Opus-class strengths.

Invocation: agent dispatch via Claude Code's native subagent tool with
`model: opus` (or whatever the orchestrator's model-selection mechanism
takes for "opus-4-7"). The orchestrator itself MUST run on Opus 4.7.

### 2.2 Codex (highest available GPT, high reasoning)

Used for:

```text
test-author
verifier              (test-result interpretation + verification memos)
review-triage
codex:review
codex:adversarial-review
```

Rationale: testing and reviewing benefit from an *independent* model with
adversarial-style reasoning. Routing reviews through a different model
family than the implementer reduces the chance that a flawed implementation
gets rubber-stamped by a model with the same blind spots.

### 2.3 Codex CLI Configuration

The repository's `.codex/` directory carries the Codex CLI binding. Before
kickoff, the orchestrator MUST verify the following pin:

```yaml
# .codex/config (or whatever the equivalent is for your codex CLI version)
model:           "<TODO: pin highest-available reasoning model — check `codex --help`>"
reasoning_effort: "high"
background:      true        # for review/adversarial-review gates
output_format:   "markdown"
```

**Action item before kickoff:** the user (or the planner subagent on first
boot) must run `codex --help` and pin the exact `--model` flag value here.
If the model name turns out to require a different field (`--reasoning`,
`--effort`, etc.), update accordingly and record the resolution in
`tasks/lessons.md`.

Codex outputs land at:

```text
results/adaptive_beta/tab_six_games/codex_reviews/<milestone>_<gate>_<UTC-timestamp>.md
```

Filename convention:

```text
M4_operator_review_2026-05-02T03-14-15Z.md
M6_fixed_beta_review_2026-05-04T11-22-33Z.md
M9_adversarial_review_2026-05-08T19-44-55Z.md
M12_final_close_2026-05-15T07-08-09Z.md
```

---

## 3. Auto-Resolution Rules per Task Tag

Tasks in `tasks/todo.md` carry `[V|N|X|A|audit]` action tags and
`[scope-tag]` per AGENTS.md §4. Auto-resolution depends on both.

### 3.1 By `[scope-tag]`

| Scope tag        | Auto-resolution behavior                                                                |
| ---------------- | --------------------------------------------------------------------------------------- |
| `[spec-read]`    | Autonomous read; record in handoff.                                                     |
| `[infra]`        | Autonomous build; verifier gate.                                                        |
| `[env]`          | Autonomous build; verifier gate.                                                        |
| `[algo]`         | Autonomous build; verifier gate. If touches `tab_operator.py`, route to OPERATOR ALARM. |
| `[operator]`     | OPERATOR ALARM — see §11. NO autonomous edit. HALT.                                     |
| `[safety]`       | OPERATOR ALARM — see §11. NO autonomous edit to clipping/certification path. HALT.      |
| `[scheduler]`    | Autonomous build (new BetaSchedule subclasses); verifier gate.                          |
| `[stress-design]`| Tier 2: autonomous design + dev pass; gate before main pass.                            |
| `[ablation]`     | Tier 2: autonomous dev pass; gate before main pass; Codex review before paper claim.    |
| `[analysis]`     | Tier 1 milestones: autonomous. Tier 3 milestones: prepare only.                         |
| `[plot]`         | Autonomous build; smoke test required; no paper-claim auto-acceptance.                  |
| `[logging]`      | Autonomous build; verifier gate.                                                        |
| `[test]`         | Autonomous (test-author runs on Codex per §2.2); verifier gate.                         |
| `[audit]`        | Autonomous if Tier 1 verifier; mandatory at Codex review gates (§9).                    |

### 3.2 By Milestone

| Milestone | Tier | Autonomous within | User gate triggers              |
| --------- | ---- | ----------------- | ------------------------------- |
| M1        | 1    | all tasks         | OPERATOR ALARM only             |
| M2        | 1    | all tasks         | OPERATOR ALARM only             |
| M3        | 1    | all tasks         | OPERATOR ALARM only             |
| M4        | 1    | all tasks         | OPERATOR ALARM; mandatory Codex |
| M5        | 1    | all tasks         | OPERATOR ALARM only             |
| M6        | 2    | dev pass          | main-pass authorization         |
| M7        | 2    | dev pass          | main-pass authorization         |
| M8        | 3    | preparation only  | acceptance of recommendation    |
| M9        | 3    | oracle validation | acceptance of validity          |
| M10       | 2    | dev pass          | main-pass authorization         |
| M11       | 3    | nothing           | entry authorization             |
| M12       | 3    | preparation only  | acceptance of final memo        |

### 3.3 Tier 1 → Tier 2 Escalation Triggers

Even within Tier 1, the orchestrator must escalate to user gate if:

- Any subagent reports a HALT (failure budget exhausted; see §4).
- Any task touches `[operator]` or `[safety]` scope (see §11).
- Any test failure proposes a fix in `tab_operator.py` or
  `safe_weighted_common.py`.
- Any `divergence_event > 0` appears in a smoke run without an existing
  documented suppression.
- Any `nan_count > 0` in a smoke run.
- Any unresolved BLOCKER from a Codex review.
- Wall-clock or dispatch budget exceeded (see §12).

---

## 4. Failure Budget

Per AGENTS.md overnight-mode precedent (Phase IV ledger):

```text
retry_per_task:           2
escalate_after:           3rd consecutive failure → HALT, write memo
memo_path:                results/adaptive_beta/tab_six_games/halts/<task_id>_<UTC-ts>.md
memo_contents:
  - task_id, milestone, role, attempt_count
  - failing tool call (or test) verbatim
  - traceback or error
  - subagent's diagnosis of root cause
  - subagent's proposed fix (NO autonomous application)
  - clean rollback path (git reset / worktree drop)
```

When the orchestrator writes a halt memo, it MUST also:

1. Update the checkpoint (§7) to `status: halted`.
2. Append to `tasks/phase_VIII_autonomous_log.jsonl` with `event: HALT`.
3. STOP further dispatch. Wait for user.

Failure-budget reset rules:

- Budget resets per task on user-authorized retry.
- Budget does NOT reset across milestones (so a chronically flaky test
  doesn't burn 2 retries × 12 milestones).

---

## 5. Stop Conditions

The orchestrator MUST halt and surface to user if any of the following:

1. **OPERATOR ALARM** — any subagent proposes editing
   `src/lse_rl/operator/tab_operator.py` or
   `mushroom-rl-dev/.../safe_weighted_common.py`. (See §11.)
2. **BLOCKER** — any Codex review or adversarial review categorizes any
   finding as `BLOCKER` (per AGENTS.md review-triage taxonomy).
3. **Failure-budget exhaustion** — any task has 3 consecutive failed
   attempts.
4. **Time budget** — wall-clock exceeds 72h since kickoff.
5. **Dispatch budget** — 50 subagent dispatches have completed without a
   user-acknowledged checkpoint.
6. **Token budget** — see §12 (default: TBD, pin before kickoff).
7. **Tier transition** — about to enter a Tier 3 milestone without prior
   user authorization (§1.3).
8. **Tier 2 dev → main escalation** — about to launch a main pass.
9. **Inconsistent state on resume** — see §10.
10. **Unexplained `divergence_event` or `nan_count > 0`** in any smoke run.
11. **Spec drift detected** — observed code behaves contrary to
    `docs/specs/phase_VIII_tab_six_games.md`. Per CLAUDE.md §4 ("If a spec
    conflicts with observed code behavior, STOP and re-plan with the
    user").
12. **Lesson conflict** — proposed action contradicts an entry in
    `tasks/lessons.md`. STOP and surface the conflict.

---

## 6. Hard Human Gates (Phase VIII Phone-Home Schedule)

This is the comprehensive list of points where the orchestrator MUST stop
and wait for user input before proceeding. Each gate is a distinct ack
event — the orchestrator may not coalesce them.

```text
G0   — kickoff config confirmed (Codex model pinned, budgets pinned)
G1   — M1 verifier gate PASS / FAIL summary
G2   — M2 verifier gate PASS / FAIL summary
G3   — M3 verifier gate PASS / FAIL summary
G4a  — M4 implementation complete; before Codex review fires
G4b  — M4 Codex review result (FAIL → HALT; PASS → continue)
G5   — M5 verifier gate PASS / FAIL summary
G6a  — M6 dev β-sweep complete; AUTHORIZE main pass?
G6b  — M6 main pass complete; before Codex review fires
G6c  — M6 Codex review result (if results enter main paper)
G7a  — M7 dev pass complete; AUTHORIZE main pass?
G7b  — M7 main pass complete; before Codex review
G7c  — M7 Codex review result (if results enter main paper)
G8   — M8 sign-specialization analysis: ACCEPT recommendation? (Tier 3)
G9a  — M9 oracle-composite validation: composite VALID for adaptivity? (Tier 3)
G9b  — M9 adaptive comparison main pass complete; before Codex review
G9c  — M9 Codex/adversarial review result
G10a — M10 dev pass complete; AUTHORIZE main pass?
G10b — M10 main pass complete; before Codex adversarial review
G10c — M10 Codex adversarial review result
G11  — M11 ENTRY: authorize appendix-method work? (Tier 3)
G12a — M12 final-recommendation memo + paper patches drafted (Tier 3)
G12b — final Codex review + adversarial review result
G12c — review-triage close: BLOCKERs cleared?
```

Default behavior at any gate: orchestrator writes a phone-home memo
to `results/adaptive_beta/tab_six_games/gates/G<N>_<UTC-ts>.md` summarizing
the milestone state and the question, updates the checkpoint to
`status: awaiting_user`, and STOPS.

---

## 7. Checkpoint State File

**Path:** `tasks/phase_VIII_autonomous_checkpoint.json`
**Format:** single JSON object, atomic write-temp-and-rename.
**Updated:** at the start of every dispatch, after every dispatch returns,
on HALT, on phone-home, and on resume.

Schema:

```json
{
  "schema_version": "1.0.0",
  "config_path": "tasks/phase_VIII_autonomous_harness.md",
  "config_sha256": "<sha256 of this file at kickoff>",
  "kickoff_timestamp_utc": "<ISO 8601>",
  "kickoff_git_head": "5d887df9...",
  "current_branch": "phase-VIII-tab-six-games-2026-04-30",
  "current_git_head": "<HEAD SHA at last update>",
  "current_milestone": "M1" | "M2" | ... | "M12",
  "current_tier": 1 | 2 | 3,
  "milestone_status": {
    "M1": "pending" | "in_progress" | "verifier_pass" | "verifier_fail" | "complete",
    "M2": "...",
    ...
    "M12": "..."
  },
  "in_flight_dispatches": [
    {"task_id": "W1.B", "role": "experiment-runner", "model": "opus-4-7",
     "worktree": "phase-VIII/experiment-runner/phase8-run-roster",
     "started_utc": "...", "status": "running" | "succeeded" | "failed"}
  ],
  "completed_dispatches": [
    {"task_id": "W1.A", "role": "verifier", "model": "codex-<pinned>",
     "started_utc": "...", "ended_utc": "...", "status": "succeeded",
     "output_path": "results/.../W1.A_handoff.md", "retry_count": 0}
  ],
  "failure_log": [
    {"task_id": "...", "attempt": 1, "error": "...", "action": "retry"}
  ],
  "halts": [
    {"task_id": "...", "reason": "OPERATOR ALARM | BLOCKER | budget | ...",
     "memo_path": "results/.../halts/...md", "halted_utc": "..."}
  ],
  "gates_acked": ["G0", "G1", ...],
  "gates_pending": ["G2", ...],
  "budgets": {
    "wall_clock_elapsed_seconds": 0,
    "wall_clock_cap_seconds": 259200,
    "dispatch_count": 0,
    "dispatch_cap": 50,
    "token_spend_by_role": {"opus-4-7": 0, "codex-<pinned>": 0},
    "token_caps_by_role": {"opus-4-7": null, "codex-<pinned>": null}
  },
  "next_action": {
    "type": "dispatch" | "phone_home" | "halt" | "complete",
    "details": "..."
  }
}
```

If the checkpoint cannot be parsed or its `config_sha256` does not match
the current file's hash, the orchestrator MUST refuse to resume and surface
the inconsistency.

---

## 8. Logging Contract

**Per-dispatch log:** `tasks/phase_VIII_autonomous_log.jsonl`
**Format:** JSONL (one event per line).
**Atomic:** append-only with `O_APPEND`; safe under crash.

Event schema (all fields required):

```json
{
  "ts_utc": "<ISO 8601>",
  "event": "DISPATCH" | "HANDOFF" | "RETRY" | "HALT" | "GATE" | "RESUME" | "CHECKPOINT",
  "milestone": "M1" | ... | "M12",
  "task_id": "<wave-letter>.<index>",
  "role": "planner" | "env-builder" | ... | "review-triage",
  "model": "opus-4-7" | "codex-<pinned>",
  "worktree": "<branch>" | null,
  "git_head_before": "<SHA>",
  "git_head_after": "<SHA>" | null,
  "duration_seconds": <float> | null,
  "retry_count": <int>,
  "status": "running" | "succeeded" | "failed" | "halted" | "awaiting_user",
  "output_path": "<path>" | null,
  "notes": "<freeform short string>"
}
```

**Per-milestone summary:**
`results/adaptive_beta/tab_six_games/milestones/<M>_summary.md` — written
when a milestone reaches verifier-pass.

**Halt memo:**
`results/adaptive_beta/tab_six_games/halts/<task_id>_<UTC-ts>.md` — written
on any HALT (see §4).

**Gate phone-home memo:**
`results/adaptive_beta/tab_six_games/gates/G<N>_<UTC-ts>.md` — written at
each hard gate (see §6).

---

## 9. Codex Review Gate Schedule

Codex review and adversarial review fire at the following points
(per Phase VIII spec §11.2; this section is the autonomous-mode dispatch
table).

| Gate | Type                | Trigger                                                   | Required for advance? |
| ---- | ------------------- | --------------------------------------------------------- | --------------------- |
| G4b  | review              | M4 close, IF operator/stable infra was modified           | Yes (PASS to continue)|
| G4b  | review              | M4 close (always, light scope)                            | Yes                   |
| G6c  | review              | M6 close, IF results flagged for main paper               | Yes                   |
| G7c  | review              | M7 close, IF results flagged for main paper               | Yes                   |
| G9c  | review + adversarial| M9 close, IF adaptive-β claims proposed                   | Yes (both PASS)       |
| G10c | adversarial review  | M10 close (always — claim is "adaptive beats fixed signs")| Yes                   |
| G12b | review + adversarial| M12 final close (always)                                  | Yes (both PASS)       |

Adversarial-review focus strings are pinned in Phase VIII spec §11.2.
The orchestrator MUST quote them verbatim to Codex at dispatch.

Review output is routed through `review-triage` (Codex). Triage produces
categorized findings (BLOCKER / MAJOR / MINOR / NIT). Behavior:

- **BLOCKER:** halt, write halt memo, phone home.
- **MAJOR:** queue for next user gate, do NOT auto-advance milestone.
- **MINOR:** auto-fix attempt with one retry; if fix fails, queue as MAJOR.
- **NIT:** record in milestone summary; do not act.

---

## 10. Resumption Protocol

If the orchestrator is restarted (Claude Code session ends, machine
restarts, etc.), it must follow this protocol on first dispatch:

1. Read `tasks/phase_VIII_autonomous_checkpoint.json`.
2. Compute the SHA-256 of `tasks/phase_VIII_autonomous_harness.md`.
3. Compare against `config_sha256` in the checkpoint. If different,
   HALT with a config-drift memo. Do NOT proceed.
4. Run `git rev-parse HEAD`. Compare against `current_git_head`. If
   different, verify the divergent commit is benign (e.g., a manual
   user commit) by inspecting `git log <checkpoint_head>..HEAD`. If
   anything beyond benign, HALT.
5. Inspect `in_flight_dispatches`. For each entry with `status: running`:
   - If the worktree branch exists with completed work, mark
     `succeeded` (best-effort). Verify by reading the subagent's
     handoff memo at `output_path`.
   - If the worktree branch is partial or absent, mark `failed` and
     trigger one retry per the failure budget (§4).
6. Inspect `gates_pending`. If a gate was awaiting user, surface it.
7. Otherwise, dispatch `next_action`.

The orchestrator MUST log a `RESUME` event in the JSONL log.

---

## 11. Operator-Touch Alarm

Editing `src/lse_rl/operator/tab_operator.py` or
`mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py`
is a hard alarm condition.

**Prohibition:** No subagent may edit either file without explicit user
authorization, regardless of tier or scope tag.

**Detection:** Before any subagent's worktree is merged into the working
branch, the orchestrator MUST run:

```bash
git diff <merge_base> <worktree_head> -- \
  src/lse_rl/operator/tab_operator.py \
  mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py
```

If the diff is non-empty, HALT immediately. Write an OPERATOR ALARM memo
to `results/adaptive_beta/tab_six_games/halts/operator_alarm_<UTC-ts>.md`
including:

- the proposed diff
- the subagent's claimed justification
- the milestone and task tag
- explicit user-authorization request

**Exception:** if the user has pre-authorized an operator edit (recorded
in `tasks/lessons.md` under a specific lesson and referenced in the
worktree branch description), the orchestrator may proceed but MUST still
trigger an ad-hoc Codex review (G-AdHoc) before the merge lands.

---

## 12. Time / Dispatch / Token Budget

Defaults (tuneable in §13):

| Budget               | Default            | Behavior on overflow                           |
| -------------------- | ------------------ | ---------------------------------------------- |
| Wall-clock           | 72h since kickoff  | HALT, phone home (extend on user command)      |
| Subagent dispatches  | 50                 | HALT, mandatory checkpoint review              |
| Tokens — Opus 4.7    | TBD (see below)    | HALT, phone home                               |
| Tokens — Codex       | TBD                | HALT, phone home                               |

Token caps default to `null` (no cap) in v1 because token-spend tracking
across two model providers requires the orchestrator to instrument both
the Claude Agent dispatch and the Codex CLI invocation. If you want hard
token caps, pin them at kickoff; the orchestrator must read them from this
file's §13 and refuse to dispatch if overrun is imminent.

---

## 13. Defaults Tuneable Before Kickoff

Edit this section (and only this section) to override defaults before
running the kickoff prompt. Then re-hash the file (the SHA-256 lock in §7
is computed at kickoff, so changes after kickoff trigger a config-drift
HALT).

```yaml
# === Tier scope overrides ===
# To make ALL of Phase VIII autonomous (Tier 1+2+3 except OPERATOR ALARM),
# move M8/M9/M11/M12 to tier_2 here. NOT recommended for paper-track work.
tier_1_milestones: [M1, M2, M3, M4, M5]
tier_2_milestones: [M6, M7, M10]
tier_3_milestones: [M8, M9, M11, M12]

# === Codex model pin (REQUIRED — fill before kickoff) ===
codex_model:           "gpt-5.5"
codex_reasoning_effort: "xhigh"
codex_background:       true
# Probe (2026-04-30): codex-cli 0.128.0 + gpt-5.5 + xhigh round-tripped
# successfully (HTTP 200, ~20k tokens). codex-cli 0.121.0 was insufficient
# (HTTP 400 "requires a newer version of Codex"); user upgraded via
# `npm install -g @openai/codex@latest`.

# === Failure budget ===
retry_per_task:        2
halt_on_attempt:       3       # third consecutive failure halts

# === Wall-clock budget ===
wall_clock_cap_hours:  72

# === Dispatch budget ===
dispatch_cap:          50

# === Token budgets (null = uncapped; pin if you want hard caps) ===
token_cap_opus_4_7:    null
token_cap_codex:       null

# === Hard gates beyond §6 default ===
# add additional gate IDs here if you want extra phone-homes
extra_gates: []

# === Auto-fix MINOR findings? ===
auto_fix_minor:        true    # set false to queue MINORs for user too

# === Auto-fix MAJOR findings? ===
auto_fix_major:        false   # default false — MAJOR always phones home
```

---

## 14. Kickoff Prompt for Claude Code

Paste the following into the Claude Code terminal (in the project
directory) to start the autonomous run. Replace `<CODEX_MODEL>` with your
pinned model name from §13.

```
Phase VIII autonomous run kickoff.

Authority: @AGENTS.md is the orchestration protocol; @tasks/phase_VIII_
autonomous_harness.md is the autonomy overlay for Phase VIII (this run).
Where the harness file narrows or contradicts AGENTS.md for Phase VIII,
the harness wins. CLAUDE.md §7 stub claim about AGENTS.md is out of date
(harness §1 documents this).

Branch: phase-VIII-tab-six-games-2026-04-30 (verify HEAD before any work).
Spec: @docs/specs/phase_VIII_tab_six_games.md.
Todo: @tasks/todo.md (Phase VIII block + M1 dispatch plan already on
disk; commit them as a "chore(plan)" commit BEFORE first dispatch).

Operate under Tier 1 + Tier 2 autonomy with Tier 3 hard gates per the
harness file §1. Model routing per §2:
  - Opus 4.7 for orchestration/planning/engineering roles (§2.1)
  - Codex (model: <CODEX_MODEL>, reasoning_effort: high) for testing/
    reviewing roles (§2.2). Verify .codex/ config matches §2.3 before
    first Codex dispatch; if it doesn't, update .codex/ in a separate
    "chore(codex): pin model and reasoning effort" commit BEFORE any
    review dispatch.

Step 0 — config gate (G0):
  1. Compute SHA-256 of tasks/phase_VIII_autonomous_harness.md.
  2. Verify §13 defaults (especially codex_model is no longer
     "TODO_PIN_BEFORE_KICKOFF"). If still TODO, HALT and request pin.
  3. Initialize tasks/phase_VIII_autonomous_checkpoint.json per §7
     schema.
  4. Initialize tasks/phase_VIII_autonomous_log.jsonl (empty file).
  5. Verify git status clean modulo:
       - results/search/shortlist_VI_B_A.csv (pre-existing, ignore)
       - 280 orphan .npz files (correctly gitignored)
       - the new harness markdown file (must be committed before
         dispatch; see step 1 below)
  6. Commit the autonomous harness markdown:
       git add tasks/phase_VIII_autonomous_harness.md
       git commit -m "chore(harness): add Phase VIII autonomous harness config v1"

Step 1 — commit the M1 dispatch plan that's already on disk:
       git add tasks/todo.md
       git commit -m "chore(plan): append Phase VIII M1 dispatch plan"

Step 2 — begin Tier 1 execution:
  Dispatch W1.A (verifier baseline sweep) per the M1 dispatch plan
  in tasks/todo.md, with the constraints from harness §3 and §4:
    - role: verifier; model: codex-<CODEX_MODEL>
    - failure budget: 2 retries; HALT on 3rd
    - any failing test that proposes a fix in tab_operator.py or
      safe_weighted_common.py → OPERATOR ALARM (§11)
    - on success: emit "W1.A PASS" handoff, update checkpoint,
      proceed to W1.B + W1.C in parallel per the dispatch plan

Step 3 onward — autonomous loop:
  Follow the M1 dispatch plan through M5. Honor every Tier 1
  invariant from §3, §4, §5, §11. Honor every Codex review gate
  from §9. Update checkpoint and log on every event per §7 and §8.
  Phone home at every G-prefixed gate from §6.

Stop conditions (§5):
  Halt and phone home immediately on OPERATOR ALARM, BLOCKER,
  failure-budget exhaustion, time/dispatch budget overflow, Tier
  transition without authorization, spec drift, or lesson conflict.

Resume protocol (§10):
  If session is restarted, read checkpoint, verify config SHA, verify
  git HEAD, then resume next_action.

Pending decisions subsumed by this kickoff:
  (a) plan-as-written approval — implicit in step 1
  (b) commit plan now or defer — answered: commit now (step 1)
  (c) authorize W1.A dispatch — answered: authorized (step 2)

DO NOT dispatch any subagent in step 2 until step 0 and step 1 commits
land successfully.

Begin.
```

---

## 15. Two Pending Decisions Subsumed by Kickoff

The kickoff prompt above absorbs the three open questions from the prior
session turn:

- (a) M1 dispatch plan as written → APPROVED (step 1 commits it).
- (b) Commit plan now or defer → COMMIT NOW (step 1).
- (c) Authorize W1.A dispatch → AUTHORIZED (step 2 dispatches W1.A under
  Tier 1 autonomy).

The user does not need to answer (a)/(b)/(c) separately if they paste the
kickoff prompt — those answers are baked in.

---

## 16. Changelog

- v1 (2026-04-30): initial autonomous-harness configuration. Tier 1+2+3
  partition; Opus 4.7 for engineering; Codex (TODO-pinned model) for
  testing/reviewing; 72h wall-clock + 50-dispatch + null-token caps;
  full hard-gate schedule (G0–G12c); checkpoint and log schemas;
  operator-touch alarm; resumption protocol. Subsumes the three
  pending M0→M1 decisions.

---

## 17. Pre-Kickoff Checklist for the User

Before pasting the kickoff prompt, confirm:

- [ ] §13 `codex_model` is pinned (not `"TODO_PIN_BEFORE_KICKOFF"`).
- [ ] §13 wall-clock cap acceptable (default 72h).
- [ ] §13 dispatch cap acceptable (default 50).
- [ ] §13 token caps decision made (default null = uncapped).
- [ ] Tier scope acceptable: Tier 1 = [M1,M2,M3,M4,M5], Tier 2 = [M6,M7,M10],
      Tier 3 = [M8,M9,M11,M12].
- [ ] You have ~72h available to phone-home at gates G1, G2, G3, G4a,
      G4b, G5 (Tier 1 alone is ~6 phone-homes).
- [ ] `.codex/` directory is configured with the pinned model and high
      reasoning effort (orchestrator will verify in step 0; you can
      pre-update if you want).
- [ ] `/tmp/LSE_RL.pre-filterrepo-2026-04-30.bundle` (16 GB backup) is
      preserved at least until the first M1 commit lands.

If any item is unchecked, fix it before kickoff. If you change §13 after
kickoff, the SHA-256 lock in the checkpoint will trigger a config-drift
HALT on the next dispatch — by design.
