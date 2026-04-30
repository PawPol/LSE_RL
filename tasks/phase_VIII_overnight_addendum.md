# Phase VIII Overnight Autonomous Mode — Addendum

**Document version:** v1 (2026-04-30).
**Scope:** This addendum overlays `tasks/phase_VIII_autonomous_harness.md`
for a single overnight run starting after G1 ack on 2026-04-30. It LIFTS
the human gates of base harness §6 except for explicit safety halts, and
adds a bug-hunt / suspicious-results protocol so Codex GPT-5.5 xhigh
actively probes for unintuitive results before they propagate.
**Authority order (highest first):**
1. This addendum (overnight overrides for this run only).
2. `tasks/phase_VIII_autonomous_harness.md` (base harness).
3. `AGENTS.md` (orchestration protocol).
4. `CLAUDE.md` (project rules).
5. `docs/specs/phase_VIII_tab_six_games.md` (spec).
**Lifetime:** This addendum applies until either (a) all M1–M12 complete
PASS, (b) a halting stop condition fires, or (c) the user manually
revokes by editing this file's status to "REVOKED".

---

## 0. Why this addendum exists

The user is leaving the workstation and wants the orchestrator to:

1. Complete all coding work (M1–M5).
2. Run all designed experiments (M6–M11).
3. Produce final analysis and recommendation memos (M8, M9, M12).
4. Find bugs by detecting suspicious experimental results and probing
   them with Codex GPT-5.5 xhigh.
5. NOT halt at gates that exist only to phone-home a present user.

Trust delegation: the user explicitly authorizes Codex GPT-5.5 xhigh to
develop details, propose refinements, and identify bugs based on first
results. Opus 4.7 implementer subagents apply Codex's recommendations
under the auto-fix-MAJOR rule (§4.2 below).

The user does NOT authorize: editing operator code, editing the paper
.tex, ignoring divergence/NaN, hiding negative results.

---

## 1. Override of base harness §1 (tiered autonomy)

For this run only:

```text
Tier 1  (M1, M2, M3, M4, M5)        — unchanged: full autonomy.
Tier 2  (M6, M7, M10)               — UPGRADED to full autonomy with auto-promotion (see §4.1).
Tier 3  (M8, M9, M11, M12)          — UPGRADED to autonomous-prepare-and-decide (see §4.3).
```

Effective regime: **all Phase VIII milestones M1–M12 are autonomous**
subject to the safety halts in §6 below.

Tier 3 caveat: M12 produces paper-update *patches as markdown files only*.
The orchestrator MUST NOT execute `git apply` against any `.tex` file in
`paper/`. Patches are drafted; user reviews on return.

---

## 2. Override of base harness §6 (hard gates)

The base harness §6 defines gates G0–G12c. Disposition for this
overnight run:

| Gate  | Base harness behavior                       | Overnight behavior                                           |
| ----- | ------------------------------------------- | ------------------------------------------------------------ |
| G0    | config gate                                 | already cleared (codex_model = gpt-5.5 xhigh).               |
| G1    | M1 close phone-home                         | AUTO-PROCEED on PASS; HALT on FAIL or BLOCKER.               |
| G2    | M2 close phone-home                         | AUTO-PROCEED on PASS; HALT on FAIL or BLOCKER.               |
| G3    | M3 close phone-home                         | AUTO-PROCEED on PASS; HALT on FAIL or BLOCKER.               |
| G4a   | M4 impl complete, before Codex              | SKIP (proceed straight to G4b).                              |
| G4b   | M4 Codex review result                      | AUTO-PROCEED on PASS; HALT on BLOCKER. MAJOR → auto-fix loop.|
| G5    | M5 close phone-home                         | AUTO-PROCEED on PASS; HALT on FAIL or BLOCKER.               |
| G6a   | M6 dev → main authorize                     | AUTO-PROMOTE per §4.1 sanity criteria.                       |
| G6b   | M6 main → Codex review                      | AUTO-PROCEED (Codex always fires for M6 main).               |
| G6c   | M6 Codex review result                      | AUTO-PROCEED on PASS; HALT on BLOCKER. MAJOR → auto-fix loop.|
| G7a   | M7 dev → main authorize                     | AUTO-PROMOTE per §4.1.                                       |
| G7b   | M7 main → Codex review                      | AUTO-PROCEED.                                                |
| G7c   | M7 Codex review result                      | AUTO-PROCEED on PASS; HALT on BLOCKER.                       |
| G8    | M8 sign-specialization accept               | AUTO-DECIDE per §4.3.G8 (write memo, follow spec §10/M8).    |
| G9a   | M9 oracle composite validity                | AUTO-DECIDE per §4.3.G9a (write memo per spec §10/M9 gate).  |
| G9b   | M9 main → Codex                             | AUTO-PROCEED.                                                |
| G9c   | M9 Codex/adversarial review                 | AUTO-PROCEED on PASS; HALT on BLOCKER.                       |
| G10a  | M10 dev → main                              | AUTO-PROMOTE per §4.1.                                       |
| G10b  | M10 main → Codex adversarial                | AUTO-PROCEED (mandatory Codex adversarial fires).            |
| G10c  | M10 adversarial review result               | AUTO-PROCEED on PASS; HALT on BLOCKER.                       |
| G11   | M11 entry authorization                     | AUTO-ENTER iff M10 closed PASS; skip-with-memo otherwise.    |
| G12a  | M12 final memo + paper patches              | AUTO-DRAFT memo + patches as files; DO NOT git-apply to .tex.|
| G12b  | M12 Codex review + adversarial              | AUTO-PROCEED.                                                |
| G12c  | review-triage close                         | AUTO-CLOSE iff no unresolved BLOCKERs.                       |

Outcome: all G-gates are non-blocking under overnight mode. Only the
safety halts in §6 below stop the run.

---

## 3. Bug-hunt and Suspicious-Results Protocol

### 3.1 Suspicious-result triggers

After every milestone that produces experimental results (M6, M7, M9,
M10, M11), the orchestrator runs a deterministic suspicious-result
detector before declaring the milestone closed. Detector rules:

```text
T1 — implausible effect:        |Δ_AUC(method vs vanilla)| > 200%
T2 — implausibly tight:          σ_AUC(method, paired seeds) < 0.01 × |mean|
T3 — non-monotone β-grid:        sign of Δ_AUC across {-2,-1,-0.5,0,0.5,1,2}
                                 changes >2 times where spec predicts monotone
T4 — Phase VII inconsistency:    overlapping cells (shapley × FM-RM, shapley
                                 × HypTest) differ from
                                 results/adaptive_beta/strategic/stage_B2_main_summary.md
                                 by > 30% on AUC mean
T5 — bimodal seed dist:          Hartigan dip-test p < 0.01 on per-seed AUC
T6 — too-fast convergence:       median bellman_residual reaches 1e-3 in
                                 < 10 episodes (likely value bootstrap leak)
T7 — clip frequency anomaly:     beta_clipping_frequency > 0.5 for any seed
                                 (clip should be rare under normal operation)
T8 — divergence rate anomaly:    divergence_event > 0 in any non-wrong-sign
                                 method (wrong-sign expected to diverge)
T9 — oracle inferiority:         oracle_beta_AUC < max(fixed_pos, fixed_neg)
                                 in any sign-switching composite (oracle
                                 should beat both fixed signs for the
                                 composite to be a valid adaptivity benchmark
                                 per spec §10/M9)
T10 — sign accuracy collapse:    beta_sign_correct < 0.55 across episodes
                                 in adaptive methods (random would be 0.5)
```

When ANY trigger fires, the orchestrator dispatches a focused Codex
bug-hunt review (separate from the milestone-close Codex review):

```text
role:       codex:bug-hunt
model:      gpt-5.5 xhigh
focus:      "Trigger <Tn> fired on <milestone>/<game>/<subcase>/<method>.
            Investigate whether this is (a) a genuine experimental
            finding worth reporting, (b) a numerical bug
            (off-by-one, value bootstrap leak, seed contamination,
            indexing error, type coercion, log/exp underflow per
            lessons.md #27, etc.), (c) an environmental issue
            (bad initial Q, opponent reset bug, regime info leak),
            or (d) an analysis bug (aggregation key mismatch,
            paired-seed alignment off, AUC integration bounds).
            Cite specific lines in the implementation to support
            your verdict. Categorize: BLOCKER / MAJOR / MINOR / NIT.
            If MAJOR or BLOCKER, propose a concrete fix.
            If genuine finding, describe what makes it a credible
            result rather than a bug."
input:      raw run.json + metrics.npz for the offending cells,
            implementation files of the relevant agent / schedule /
            env, latest milestone summary, lessons.md, full Phase
            VII Stage B2 summary.
output:     results/adaptive_beta/tab_six_games/codex_reviews/
              bug_hunt_<trigger>_<milestone>_<UTC-ts>.md
```

Disposition of bug-hunt review:

- **BLOCKER:** halt with bug_hunt_blocker memo. Do NOT auto-fix.
- **MAJOR:** auto-fix loop per §4.2.
- **MINOR:** record in milestone summary; proceed.
- **NIT:** record; proceed.
- **Genuine finding (not a bug):** record in milestone summary as a
  flagged-but-validated result; proceed; tag the result for paper
  attention in the M12 final memo.

### 3.2 Default Codex milestone-review focus expansion

For overnight mode, Codex milestone reviews (G4b, G6c, G7c, G9c, G10c,
G12b) carry an EXPANDED focus that explicitly includes the trigger
list above plus this overnight-specific instruction appended verbatim
to the spec §11.2 focus strings:

```text
Additional overnight focus: actively look for the following bug
classes regardless of whether triggers fired:
  - Off-by-one in episode/transition indexing
  - Seed contamination across paired comparisons
  - Aggregation key drift (lessons.md #19, #20)
  - Default-root path drift (lessons.md #11) — verify all run paths
    fall under results/adaptive_beta/tab_six_games/
  - Numpy ≥ 2.0 int(state) regression on shape-(1,) arrays
    (lessons.md #28)
  - log/exp underflow on negative β tails (lessons.md #27)
  - Value bootstrap leakage in target-Q estimation
  - β=0 bit-identity guard regression (must equal classical Bellman
    target exactly under |β| ≤ 1e-8)
  - Single-update-path invariant for AdaptiveBetaQAgent (Phase VII
    spec §16.2; spec §6 reiterates)
  - operator/safety scope drift — confirm zero edits to
    src/lse_rl/operator/tab_operator.py and
    mushroom-rl-dev/.../safe_weighted_common.py
  - Wrong-sign method should reliably underperform; if it doesn't,
    that is itself a bug signal
  - oracle should beat both fixed signs in sign-switching composites;
    if it doesn't, the composite design is invalid (spec §10/M9)
```

### 3.3 Counter-intuitive results notebook

The orchestrator maintains
`results/adaptive_beta/tab_six_games/counter_intuitive_findings.md` as
an append-only log of every result that triggered a bug-hunt review
AND was confirmed as a genuine finding (not a bug). This is one of the
key deliverables of overnight mode — it surfaces results that warrant
the user's attention on return.

Schema per entry:

```text
## <UTC ts> — <milestone> <game> <subcase> <method>

**Trigger:** Tn (one-line description)
**Finding summary:** what was unintuitive
**Codex bug-hunt verdict:** GENUINE FINDING (link to bug-hunt memo)
**Numbers:** mean, std, paired diff vs vanilla, paired diff vs
            best fixed β, vs Phase VII baseline (where available),
            seed count, failure count
**Why it's credible:** Codex's reasoning excerpted verbatim
**Implications for paper:** brief — for user to triage
**Reproduction:** run-id list, configs, command-line invocation
```

---

## 4. Auto-progression and auto-fix rules

### 4.1 Tier 2 dev → main auto-promotion (G6a, G7a, G10a)

Auto-promote dev pass → main pass IFF ALL of:

```text
P1 — dev pass verifier PASS
P2 — Phase8RunRoster shows all dev cells completed (no failed/skipped)
P3 — no NaN, no divergence_event in any dev run
P4 — no suspicious-result trigger fired on dev pass (§3.1)
P5 — paired-seed t-test p > 0.05 vs vanilla on dev seed sample is NOT
     a hard requirement (we want to see signal); but if effect size is
     borderline (Cohen's d in [0.2, 0.5]), increase main-pass seeds
     from 10 to 20 to restore statistical power before promoting
P6 — wall-clock and dispatch budgets have headroom (>20% remaining
     OR enough for a main pass per §5.4 estimate)
```

If any of P1–P4 or P6 fails: HALT with auto_promote_blocked memo.
P5 is informational, never blocking.

### 4.2 Auto-fix MAJOR loop

Override base harness §13 `auto_fix_major` from `false` → `true`.

When a Codex review (milestone-close OR bug-hunt) categorizes a finding
as MAJOR with a proposed fix:

```text
1. Implementer subagent (Opus 4.7, role: algo-implementer or
   experiment-runner depending on scope) receives the MAJOR finding
   verbatim.
2. Implementer drafts the fix in a new worktree
   (phase-VIII/auto-fix/<milestone>_<finding-id>_<UTC-ts>).
3. Implementer runs the affected tests + a smoke regression of the
   nearest milestone tests.
4. If green, merge worktree into branch with commit message:
     phase-VIII(<milestone> auto-fix): <one-line>
     - Fixes Codex MAJOR finding <id>: <verbatim title>
     - Source review: <path to review memo>
   Re-run the original Codex review pass to confirm the finding is
   resolved (one extra dispatch).
5. If verifier or post-fix Codex re-review fails, escalate to BLOCKER
   → HALT.
6. Maximum auto-fix loops per milestone close: 3. After the third
   failed fix, HALT with auto_fix_exhausted memo.
```

NIT and MINOR findings are auto-fixed similarly but with a relaxed
loop (1 retry; no re-review dispatch — verifier-pass is sufficient).

### 4.3 Tier 3 auto-decide rules

#### G8 — sign-specialization

Spec §10/M8 gate: "If no credible G_+ exists, stop adaptive
sign-switching work and write a negative-result memo."

Auto-decide algorithm:

```text
1. Load Stage 1 + Stage 2 results from results/adaptive_beta/
   tab_six_games/processed/.
2. For each (game, subcase), compute paired-seed AUC differences:
     fixed_pos − vanilla, fixed_neg − vanilla, fixed_pos − fixed_neg.
3. G_plus candidate: paired-seed bootstrap 95% CI of (fixed_pos − vanilla)
   strictly > 0 AND CI of (fixed_pos − fixed_neg) > 0.
4. G_minus candidate: analogous for fixed_neg.
5. If at least one G_+ AND at least one G_- exist: proceed to M9
   with the top-ranked candidate of each sign.
6. If G_+ but no G_-: proceed to M9 with G_+ alone, document
   asymmetry in stage3_sign_specialization.md.
7. If neither: write
   results/adaptive_beta/tab_six_games/no_G_plus_found.md per spec
   §14, skip M9 + M10 (mark them stopped-by-gate in roster), proceed
   to M11/M12.
```

#### G9a — oracle composite validity

Spec §10/M9 gate: "oracle β must beat both fixed signs for a composite
to be a valid adaptivity benchmark. If oracle β does not beat fixed
signs, redesign once; if still failing, stop and report."

Auto-decide algorithm:

```text
1. Run oracle validation on the composite from §4.3 G8 candidates.
2. Compute paired-seed AUC: oracle vs fixed_pos, oracle vs fixed_neg.
3. If oracle bootstrap 95% CI > 0 vs BOTH fixed signs: composite VALID,
   proceed to M9 adaptive comparison.
4. If oracle does not beat both: redesign once
     - swap composite ingredients to second-ranked G_+, G_- candidates
     - re-run oracle validation
5. If still failing: write
   results/adaptive_beta/tab_six_games/oracle_composite_failed.md per
   spec §14, skip remaining M9 work, proceed to M10.
```

#### G11 — M11 entry

Auto-enter M11 IFF M10 closed PASS (i.e., M10 adversarial review
PASS, no unresolved MAJOR/BLOCKER, contraction-adaptive results
finite). Otherwise skip M11 with a stopped-by-gate memo.

Within M11, run only the methods explicitly listed in spec §6.2
optional set (hedge_beta, discounted_hedge_beta,
continuous_gradient_beta, bilevel_SOBBO_beta) — and only on the
validated composite from M9 (or the best Stage 1 cell if M9 was
skipped).

#### G12a — final recommendation

Auto-draft these memos to `results/adaptive_beta/tab_six_games/`:

```text
final_recommendation.md
paper_update/main_patch.md          (recommendation case 1)
paper_update/appendix_patch.md      (recommendation case 2)
paper_update/no_update.md           (recommendation case 3)
```

The orchestrator picks ONE of the three patches as the primary
recommendation per spec §15 criteria (strong/medium/no-update). Other
two are marked `Status: Inactive alternative` in their first line.

NO `git apply` against `paper/*.tex`. Files are markdown deliverables
only. User reviews on return.

---

## 5. Override of base harness §12 (budgets)

```yaml
wall_clock_cap_hours:  36          # was 72, but overnight expected ~8-16h;
                                   # 36 is a hard ceiling for runaway protection
dispatch_cap:          300         # was 50; overnight needs many more dispatches
token_cap_opus_4_7:    null        # uncapped; halts on dispatch_cap or wall_clock first
token_cap_codex:       null        # uncapped; halts on dispatch_cap or wall_clock first
auto_fix_minor:        true
auto_fix_major:        true        # overridden from base harness
auto_fix_nit:          true
```

Budget overflow behavior: HALT with budget_overflow memo. Do NOT
attempt to compress remaining work.

### 5.1 Per-milestone soft budget hint (advisory)

| Milestone | Estimated wall-clock | Estimated dispatches |
| --------- | -------------------- | -------------------- |
| M1        | 1–2 h                | 7–10                 |
| M2        | 1–2 h                | 4–6                  |
| M3        | 1 h                  | 4–6                  |
| M4        | 2–3 h                | 12–18                |
| M5        | 1–2 h                | 6–10                 |
| M6        | 4–8 h (compute-heavy)| 6–10                 |
| M7        | 2–4 h                | 4–6                  |
| M8        | <1 h (analysis only) | 2–3                  |
| M9        | 4–6 h                | 6–10                 |
| M10       | 2–4 h                | 4–8                  |
| M11       | 2–3 h (optional)     | 4–6                  |
| M12       | <1 h (memo drafting) | 3–5                  |
| **Total** | **20–36 h**          | **62–98**            |

If budgets are tracking 1.5× the upper estimate at any milestone close,
the orchestrator preemptively dispatches a budget_review subagent
(Opus, role: experiment-runner) that proposes scope reductions
(reduce seed counts, drop appendix methods, etc.) and HALTS for user
acknowledgment on return. Better to wake up to a halted run with a
plan than a run that imploded mid-M11.

---

## 6. Safety halts (retained verbatim from base harness §5, plus additions)

The following halt conditions are NEVER overridden by overnight mode:

1. **OPERATOR ALARM** — any subagent edits or proposes editing
   `src/lse_rl/operator/tab_operator.py` or
   `mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_common.py`.
2. **BLOCKER from any review** — milestone-close, bug-hunt, or
   adversarial.
3. **`nan_count > 0`** in any run.
4. **`divergence_event > 0`** in any non-wrong-sign method.
5. **Spec drift** — observed code conflicts with
   `docs/specs/phase_VIII_tab_six_games.md`.
6. **Lesson conflict** — proposed action contradicts `tasks/lessons.md`.
7. **Failure-budget exhaustion** — 3rd consecutive task failure.
8. **Budget overflow** — wall-clock or dispatch exceeded.
9. **Auto-fix loop exhausted** — 3 consecutive failed fix attempts on
   the same MAJOR finding.
10. **Bug-hunt BLOCKER** — Codex GPT-5.5 xhigh confirms a result is a
    bug rather than a finding.
11. **`.tex` edit attempt** — any subagent attempts to edit
    `paper/*.tex` (overnight prohibits this).
12. **Stopped-by-gate-with-cascade** — if M8 produces no_G_plus_found
    AND M9 + M10 are skipped, the orchestrator does NOT halt — that's
    a documented spec-§14 negative result. Proceeds to M11/M12 with
    the cascade documented. This is NOT a halt; it's a planned skip.

On any halt: write halt memo to
`results/adaptive_beta/tab_six_games/halts/<reason>_<UTC-ts>.md`,
update checkpoint, write the JSONL HALT event, STOP. Do NOT phone
home (user is away); the halt memo is the phone-home.

---

## 7. End-of-run synthesis

When the orchestrator either (a) completes M12 or (b) halts on any
condition above, it MUST dispatch a final synthesis subagent BEFORE
terminating:

```text
role:    synthesis (planner role on Opus 4.7)
inputs:
  - tasks/phase_VIII_autonomous_checkpoint.json
  - tasks/phase_VIII_autonomous_log.jsonl
  - all milestone summary files in results/adaptive_beta/
    tab_six_games/milestones/
  - all halt memos in results/adaptive_beta/tab_six_games/halts/
  - all Codex review memos in results/adaptive_beta/tab_six_games/
    codex_reviews/
  - results/adaptive_beta/tab_six_games/counter_intuitive_findings.md
  - results/adaptive_beta/tab_six_games/final_recommendation.md (if produced)
outputs:
  - results/adaptive_beta/tab_six_games/overnight_run_summary.md
  - results/adaptive_beta/tab_six_games/decisions_made_autonomously.md
  - tasks/lessons.md (appended with new lessons)
```

### 7.1 overnight_run_summary.md required sections

```text
1. Run wall-clock + dispatch + token totals
2. Milestones completed PASS / FAIL / skipped-with-reason
3. Most important findings (top 3, ranked by paper relevance)
4. Counter-intuitive results requiring user attention (link to log)
5. Halts and how they were handled (link to halt memos)
6. Codex review verdicts (BLOCKER/MAJOR/MINOR/NIT counts per gate)
7. Auto-fixes applied (count + summary)
8. M12 paper-update recommendation (main / appendix / no_update)
9. Outstanding decisions for the user on return
10. Suggested next session prompt
```

### 7.2 decisions_made_autonomously.md required sections

Every decision made without user input gets one entry. Schema:

```text
## <UTC ts> — <decision id>

**Context:** the choice point in spec or harness
**Options considered:** brief
**Decision:** what was chosen
**Rationale:** Codex or orchestrator reasoning, verbatim
**Affected artifacts:** files, runs, results
**Reversal cost:** how hard it would be to redo if user disagrees
```

This memo is the user's single read on return to understand what the
agent did with delegated authority.

---

## 8. Kickoff prompt for overnight mode

The user pastes this in the Claude Code terminal AT G1 ack time
(i.e., right now, after the M1 baseline phone-home):

```
Overnight mode engaged for Phase VIII.

Authority order (highest first):
  @tasks/phase_VIII_overnight_addendum.md   (this run only)
  @tasks/phase_VIII_autonomous_harness.md   (base harness)
  @AGENTS.md                                 (orchestration)
  @CLAUDE.md                                 (project rules)
  @docs/specs/phase_VIII_tab_six_games.md   (spec)

Status at engage time:
  - W1.A baseline PASS (506/506 tests, codex gpt-5.5 xhigh)
  - W1.A evidence committed
  - in flight: about to dispatch W1.B + W1.C parallel

Apply addendum overrides:
  - All G-gates G1..G12c are non-blocking (auto-decide rules in §2/§4).
  - Tier 2 (M6, M7, M10) auto-promote dev → main per §4.1.
  - Tier 3 (M8, M9, M11, M12) auto-decide per §4.3.
  - Auto-fix MAJOR enabled (§4.2).
  - Bug-hunt protocol active (§3): suspicious-result triggers T1–T10
    fire focused Codex bug-hunt reviews; counter-intuitive findings
    accumulate in counter_intuitive_findings.md.
  - Budgets: 36h wall-clock cap, 300 dispatch cap, no token cap.
  - Safety halts (§6) UNCHANGED — operator alarm, BLOCKER, NaN,
    divergence, spec drift, lesson conflict, .tex edit, budget
    overflow, auto-fix exhaustion all still HALT.

Continue from where you are: dispatch W1.B + W1.C in parallel per the
M1 dispatch plan and the constraints I sent in the prior turn (W1.B
defends its serialization choice; W1.C uses np.log not log1p; explicit
make_run_dir base; operator-touch diff check before each merge).

Then proceed autonomously through M1 → M2 → M3 → M4 → M5 → M6 → M7
→ M8 (auto-decide G8) → M9 (auto-decide G9a) → M10 → M11 (auto-enter
G11 iff M10 PASS) → M12 (auto-draft memos, no .tex edits).

Run end-of-run synthesis per §7 BEFORE terminating, regardless of
whether the run completes M12 or halts earlier.

I am leaving. The next time I look at this terminal, expect to see
either:
  (a) overnight_run_summary.md indicating M12 complete, or
  (b) a halt memo in results/adaptive_beta/tab_six_games/halts/
      with the cause and the synthesis memo for context.

Begin.
```

---

## 9. Pre-engagement checklist (run before pasting §8 kickoff)

- [ ] Confirm W1.A handoff is committed (per the prior turn's
      "evidence commit").
- [ ] Confirm `.codex/config` reflects model = gpt-5.5 xhigh and
      reasoning_effort = high.
- [ ] Confirm `tasks/phase_VIII_autonomous_checkpoint.json` reflects
      W1.A status: succeeded, gates_acked: [G0], current_milestone: M1,
      milestone_status.M1: in_progress.
- [ ] Confirm git working tree clean modulo: orphan .npz files,
      shortlist_VI_B_A.csv, this addendum file (uncommitted — will be
      committed by the agent in step 1 of the §8 kickoff).
- [ ] Confirm `/tmp/LSE_RL.pre-filterrepo-2026-04-30.bundle` still
      exists as off-tree backup.
- [ ] Confirm wall-clock and dispatch budget headroom acceptable
      given the 20–36h estimate in §5.1.
- [ ] You have at least 12 contiguous hours available before next
      check-in.

If any item is unchecked, fix before kickoff. If you are about to
leave the terminal RIGHT NOW and the items are clean, paste §8 and go.

---

## 10. Changelog

- v1 (2026-04-30): initial overnight addendum. Lifts G1–G12c to
  auto-proceed/auto-decide. Adds bug-hunt protocol with 10 triggers
  and Codex GPT-5.5 xhigh focused reviews. Adds counter-intuitive
  findings notebook. Defines auto-fix MAJOR loop with 3-attempt
  ceiling. Defines Tier 3 auto-decide rules per spec §10/§14/§15.
  Pins overnight budgets (36h, 300 dispatches). Mandates end-of-run
  synthesis. Retains all safety halts. Bans `.tex` edits.
