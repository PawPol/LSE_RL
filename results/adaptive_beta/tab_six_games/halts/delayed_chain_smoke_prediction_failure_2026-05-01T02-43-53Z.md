# HALT — Phase VIII delayed_chain smoke prediction failure

**Halt UTC:** 2026-05-01T02:43:53Z
**Triggered by:** `tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py::test_smoke_DC_Long50_AUC_ordering`
**Trigger class (per patch §11.7):** **T11** — sign-prediction failure on advance-only delayed_chain
**Severity (per addendum §6 + patch §11.7):** **paper-critical, BLOCKER semantics, halt for human review**
**Branch HEAD at halt:** `47d5353d` (M2 reopen wave 1 merged)
**Autonomous regime status:** STOPPED at this memo per addendum §6 protocol

---

## 1. What failed

```
$ .venv/bin/python -m pytest tests/ -q

1 failed, 1684 passed, 2 skipped, 37 warnings in 35.57s

FAILED tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py
       ::test_smoke_DC_Long50_AUC_ordering
```

The smoke test runs β ∈ {-1, 0, +1} on DC-Long50 with 3 seeds × 1k episodes via `AdaptiveBetaQAgent` + `PassiveOpponent`. The test asserts the falsifiable prediction from patch §11.3:

> **P-Sign:** AUC(+β) > AUC(0) > AUC(-β) on advance-only delayed_chain subcases (DC-Short10, DC-Medium20, DC-Long50), paired-bootstrap 95% CI strictly ordered, Cohen's d > 0.3 on each gap.

Observed result (per M2 reopen W1.B subagent's verification, confirmed by independent pytest run):

```
AUC(+1) = AUC(0) = AUC(-1) = 1000.0
(3 seeds each, all identical)
```

Cohen's d on either gap = 0/0 = undefined (zero variance, zero mean difference). The prediction P-Sign **cannot be tested** by this implementation in its current form.

---

## 2. Root cause

The advance-only delayed_chain subcases (DC-Short10, DC-Medium20, DC-Long50) have **action space = `Discrete(1)`** — a single advance action. The chain transition is deterministic: from state `s ∈ [0, L-1]`, action 0 (advance) → state `s+1`. There is no decision the agent can make differently under different β values; every policy is identical (the only valid policy is "always advance"). Every episode terminates at state L = goal in exactly L steps with reward +1. Episode return is **deterministic and constant at 1.0** regardless of β.

AUC = sum of episode returns over 1k episodes = 1000.0 for all three β values. β-induced differences would, in principle, manifest in **Q-value convergence rate**, not in returned reward — but the smoke test's primary axis is AUC of returns, which has zero variance across β.

---

## 3. Why this is paper-critical

The paper title is **"Selective Temporal Credit Assignment"**. Patch §11 was added explicitly to anchor that title in directly testable in-suite evidence. The headline cell DC-Long50 is the long-horizon credit-assignment cell whose AUC differentiation across β IS the paper's claim. If the implementation cannot demonstrate that differentiation on the return axis, the paper has no in-suite long-horizon evidence — only the matrix-game H ≤ 10 horizons remain, which patch §11.1 explicitly identified as insufficient.

Per patch §11.7 T11 design intent:

> "T11 is a sign-flip trigger and should fire with high priority — if the headline temporal-credit-assignment cell does not produce the predicted ordering, that is paper-critical and warrants halting M6 for human review per addendum §6 BLOCKER semantics."

The smoke test fires this trigger BEFORE M6 even starts, on implementation evidence alone, which is the strongest possible signal that the patch §11.3 prediction is mis-specified (or the implementation is) and proceeding with M6 main pass on this configuration would waste compute and produce inconclusive results.

---

## 4. Three remediation options (per M2 reopen W1.B subagent flag)

The implementing subagent flagged this concern at handoff time and proposed three remediation options. Listed here verbatim with my orchestrator-side analysis:

### Option (a) — Replace AUC with a Q-convergence diagnostic on advance-only subcases

**Mechanism:** Define a new metric `q_convergence_rate` = number of episodes until `|Q(s_0, a=advance) - V*(s_0)|/V*(s_0) < 0.05` where V*(s_0) = γ^L. Different β values converge at different rates under optimistic init; this is the actual mechanism the prediction was trying to capture.

**Implementation cost:** Low. Add `q_convergence_rate` to the metrics module + smoke test rewrite. Affects M6 aggregator: `q_convergence_rate` joins the §7.2 delta metrics list.

**Paper alignment:** GOOD — the metric directly measures "selective temporal credit assignment" speed, which is the paper's claim. Cleaner story than "AUC differentiation under deterministic returns".

**Risk:** Existing patch §11.3 wording (P-Sign as AUC ordering) becomes obsolete and needs re-derivation. Other predictions (P-Scaling, P-Branch, P-VII-Parity) need restatement against q_convergence_rate.

### Option (b) — Add a no-op action to advance-only subcases

**Mechanism:** Change action space from `Discrete(1)` to `Discrete(2)` where action 0 = advance, action 1 = no-op (stay in place, costs 1 unit of horizon without progress). Now β influences the rate at which Q-learning prefers action 0 over action 1; β=+ pushes faster preference for advance, β=- delays it. Episode return varies with β-induced exploration patterns.

**Implementation cost:** Medium. Modify `delayed_chain.py` advance-only subcases to support 2 actions; update tests; potentially break the "advance-only" semantic name.

**Paper alignment:** OK — preserves AUC axis as the primary metric, but introduces an artificial action that wasn't in the patch §11.2 contract. Reviewer attack vector: "the advance-only subcase isn't really advance-only".

**Risk:** Patch §11.2 explicitly says "Discrete(1)" for advance-only subcases. Changing this contradicts the patch as folded. Needs spec amendment.

### Option (c) — Scope the prediction to DC-Branching20 only

**Mechanism:** Acknowledge that advance-only subcases are degenerate (no decision = no AUC differentiation on the return axis) and restrict the falsifiable prediction P-Sign to DC-Branching20 only (which has Discrete(2) and a real choice between advance vs branch_wrong). Keep DC-Short10, DC-Medium20, DC-Long50 as Q-convergence diagnostics only (per option (a)) without an AUC-ordering prediction.

**Implementation cost:** Low. Mark the smoke test on advance-only as `@pytest.mark.skip(reason=...)` with clear documentation; update patch §11.3 wording in spec; keep the smoke test for DC-Branching20 only.

**Paper alignment:** WEAK — paper-headline cell DC-Long50 loses its AUC-ordering prediction; only DC-Branching20 (which has the trap-arm exploration penalty per patch §11.3 P-Branch) remains. The "long-horizon" axis is partially demoted from headline to diagnostic.

**Risk:** Reduces the paper's in-suite long-horizon evidence to a single cell (DC-Branching20), which is the WEAK end of patch §11.1's motivation.

---

## 5. Recommended disposition (orchestrator's view)

**Option (a) is the cleanest path forward.** It honors the paper's core mechanism claim (β changes Q-value propagation rate, which IS the temporal credit assignment story), provides a new metric (`q_convergence_rate`) that joins the spec's §7.2 inventory cleanly, and restates P-Sign in a form the implementation can actually produce. Options (b) and (c) are weaker on paper-alignment grounds.

However, this is a **research-design decision**, not an engineering decision. The patch §11 author (the user, per patch §0 + the "researcher critique" provenance in patch §9) needs to choose. Per CLAUDE.md §3 clarification protocol and addendum §6 halt-for-human-review semantics, the autonomous run STOPS here.

---

## 6. Repo state at halt

```
branch              : phase-VIII-tab-six-games-2026-04-30
HEAD                : 47d5353d
M0–M5               : COMPLETE (in tree + pushed to origin)
M6 wave 0           : COMPLETE — spec patch v2 folded in (commit c08126e0)
M2 reopen           : COMPLETE code-wise — RR-Sparse + delayed_chain + PassiveOpponent landed
                     (commits 8a0ef36e + 5b5cf905 + 47d5353d + test_registry bump)
M2 reopen close     : BLOCKED — smoke prediction failure routes to T11 halt
M4 reopen           : NOT STARTED (FP/RM agents still pending)
M6 waves 1+         : NOT STARTED
working tree        : clean
unmerged files      : none
pytest baseline     : 1684 PASSED + 2 SKIPPED + 1 FAILED (smoke prediction)
```

Test counts:
- Pre-amendment baseline: 1642 + 2 SKIP
- Post M6 wave 0 (spec fold-in only): 1642 + 2 SKIP (no test change)
- Post M2 reopen merges: **1684 + 2 SKIP + 1 FAIL** (+42 new tests = 7 RR-Sparse + 8 delayed_chain + 4 PassiveOpponent + 23 test_registry parametric coverage; minus 1 for the smoke test that fails)

The 1 failing test is `test_smoke_DC_Long50_AUC_ordering`. It is `@pytest.mark.smoke`-decorated but pytest is collecting and running it anyway (no `-m "not smoke"` filter configured). Subagent recommended skipping with `@pytest.mark.skip` until the research decision is made; orchestrator did NOT apply that skip (would mask the issue).

---

## 7. What the user should decide on return

1. Pick option **(a)**, **(b)**, or **(c)** from §4 above.
2. If (a): authorize a follow-up M2 reopen wave to add `q_convergence_rate` metric + update patch §11.3 wording in the spec.
3. If (b): authorize a Discrete(1) → Discrete(2) change in advance-only subcases + spec amendment.
4. If (c): authorize skipping the smoke test on advance-only + spec text restatement of P-Sign scope.
5. Optionally: alternative remediations the orchestrator did not enumerate.

After decision, the autonomous run can resume from M2 reopen close → M4 reopen → M6 waves per session 3 step plan.

---

## 8. Files modified in this session before halt

```
docs/specs/phase_VIII_tab_six_games.md         (M6 wave 0 fold-in; +364 lines)
experiments/adaptive_beta/strategic_games/games/potential.py    (lemma docstring)
experiments/adaptive_beta/strategic_games/games/rules_of_road.py (sparse_terminal flag)
experiments/adaptive_beta/strategic_games/games/delayed_chain.py (NEW; 4 subcases)
experiments/adaptive_beta/strategic_games/adversaries/passive.py (NEW; PassiveOpponent)
experiments/adaptive_beta/strategic_games/registry.py            (RR-Sparse + delayed_chain + passive registered)
tasks/todo.md                                                   (M2/M4 reopen tasks appended)
tests/adaptive_beta/strategic_games/test_rules_of_road_sparse.py (NEW; 7 PASS)
tests/adaptive_beta/strategic_games/test_delayed_chain.py        (NEW; 8 PASS)
tests/adaptive_beta/strategic_games/test_passive_opponent.py     (NEW; 4 PASS)
tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py (NEW; FAIL — this halt)
tests/adaptive_beta/strategic_games/test_registry.py             (count bump 7→8→9)
tests/adaptive_beta/strategic_games/conftest.py                  (smoke marker registered)
results/adaptive_beta/tab_six_games/M2_reopen_rr_sparse_handoff.md  (NEW; handoff)
results/adaptive_beta/tab_six_games/M2_reopen_delayed_chain_handoff.md (NEW; handoff with FLAG-1 explicit)
```

Commits landed (all on phase-VIII-tab-six-games-2026-04-30, all pushed):
```
c08126e0  phase-VIII(M6 wave 0): fold pre-M6 spec amendment v2 per researcher critique
8a0ef36e  phase-VIII(M2 reopen wave 1): merge RR-Sparse subcase per patch §1
db2fed8d  phase-VIII(M2 reopen wave 1): start of D.2 merge (auto-merged conflict pending)
47d5353d  phase-VIII(M2 reopen wave 1): merge delayed_chain + PassiveOpponent per patch §11
```

(Final push to origin has NOT occurred for `47d5353d` — the halt fires before the close commit. Orchestrator will push everything-up-to-halt after writing this memo.)

---

## 9. Clean rollback path (if user chooses)

If the user prefers to undo M2 reopen entirely and start over with a different patch §11 design:

```
git reset --hard c08126e0   # back to M6 wave 0 state (spec patch folded; no code yet)
```

This loses RR-Sparse (which is fine on its own merit per patch §1), so consider:

```
git reset --hard 8a0ef36e   # keep RR-Sparse; lose delayed_chain
```

OR keep everything and just add a follow-up commit per chosen option (a)/(b)/(c). The orchestrator recommends the latter — RR-Sparse and PassiveOpponent are both useful on their own merit; only the prediction test needs revision.
