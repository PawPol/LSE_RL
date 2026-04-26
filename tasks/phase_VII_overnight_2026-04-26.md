# Phase VII Overnight Run — 2026-04-26

**Authorization:** `tasks/phase_VII_confirmation_final.md` (full autonomous, M1 → final paper update if data supports it).
**Branch:** `phase-VII-overnight-2026-04-26`.
**Wall-clock budget:** 8–10 h. Stop and write status memo if unfinished.
**Failure policy:** retry transient failures up to 2×, then stop and memo. No-clip divergence = data, not failure.
**Stage gating:** auto-promote on quantitative bar (see §1 of authorization).

---

## Quantitative promotion bar (locked 2026-04-26)

**Stage A → Stage B promotion** triggers if **all** of:
1. `adaptive_beta` improves AUC over `vanilla` on paired seeds in **at least one env** (paired-mean diff > 0).
2. No increase in `catastrophic_episodes` for `adaptive_beta` vs `vanilla` beyond noise (paired diff ≤ +1 SE).
3. No `divergence_event` for `adaptive_beta` (clipped variant); `adaptive_beta_no_clip` may diverge — that's data.
4. Mechanism evidence on at least one non-bandit env: `alignment_rate > 0.5` on informative transitions, **or** `mean_d_eff < γ` on informative transitions, **or** `recovery_time(adaptive_beta) < recovery_time(vanilla)`.

**Stage B → Stage C promotion** triggers if:
- Headline `adaptive_beta` vs `vanilla` paired bootstrap 95% CI on AUC excludes 0 on at least one promoted env, AND
- Mechanism diagnostics confirm the predicted alignment / effective-discount mechanism.

If no env clears Stage-A bar → stop, write negative-result memo.
If no env clears Stage-B bar → skip Stage C, write Stage B summary, finalize report.

---

## Task ledger

| ID | Milestone | Status | Started | Completed | Commit | Notes |
|----|-----------|--------|---------|-----------|--------|-------|
| M0 | Spec + todo lock | DONE | 2026-04-26 | 2026-04-26 | 192a8793 | §22 resolutions committed; branch baseline. |
| M1.1 | Operator kernel extraction (`src/lse_rl/operator/tab_operator.py`) | DONE | 2026-04-26 | 2026-04-26 | 44f75a0e | 141 LOC stateless kernel; pre 345 / post 442 / full algorithms suite 788 green. |
| M1.2 | `SafeWeightedCommon` refactor + equivalence test | DONE | 2026-04-26 | 2026-04-26 | 44f75a0e | 1568 (β,γ,r,v) tuples pinned to ≤1e-12; instrumentation byte-equivalent; lessons.md updated. |
| M1.3 | Adaptive-β schedules + tests | DONE | 2026-04-26 | 2026-04-26 | afeacf9d | 8 schedule classes + factory; 24 tests green; forbidden-future spy enforces no leakage; sticky divergence flag per §13.5. |
| M1.4 | M1 verifier + `/codex:review` + `/codex:adversarial-review` | DONE | 2026-04-26 | 2026-04-26 | 61754a57 | verifier 1053 green; Codex 2 BLOCKERs + 2 MAJORs all fixed; 876 algo+adaptive tests green. |
| M2.1 | `envs/rps.py` (MushroomRL) | DONE | 2026-04-26 | 2026-04-26 | b09962a5 | 225 LOC; canonical sign = None; phase cycle + oracle verified. |
| M2.2 | `envs/switching_bandit.py` | DONE | 2026-04-26 | 2026-04-26 | b09962a5 | 165 LOC; canonical sign = None; §22.5 mechanism exclusion documented. |
| M2.3 | `envs/hazard_gridworld.py` | DONE | 2026-04-26 | 2026-04-26 | b09962a5 | 256 LOC; canonical sign = "-"; reshuffle + manhattan-greedy oracle. |
| M2.4 | `envs/delayed_chain.py` | DONE | 2026-04-26 | 2026-04-26 | b09962a5 | 122 LOC; canonical sign = "+"; deterministic. |
| M2.5 | M2 env tests + verifier | DONE | 2026-04-26 | 2026-04-26 | d7a2d292 | 52 env tests; 928 algorithms+adaptive total green. |
| M3.1 | Q-learning agent (one shared `_step_update`) | DONE | 2026-04-26 | 2026-04-26 | 0efa824e | 384 LOC; 15 tests; β=0 bit-id confirmed; one-code-path counter==5 for all 8 methods. Test 8 deviation flagged. |
| M3.2 | Logging callbacks (episode + transition schemas) | DONE | 2026-04-26 | 2026-04-26 | b4d39ae9 | 351 LOC; pyarrow schema pinned; 27-col episode + 21-col transition. |
| M3.3 | `run_experiment.py` + configs | DONE | 2026-04-26 | 2026-04-26 | b4d39ae9 | 588 LOC runner; 4 yaml configs; manifest split — Phase V untouched, Phase VII gets own list. Smoke 20/20 passed. |
| M3.4 | Reproducibility + same-code-path tests | DONE | 2026-04-26 | 2026-04-26 | 61bc2ffd | 26 new tests; 969 total. |
| M3.5 | M3 verifier | DONE | 2026-04-26 | 2026-04-26 | 61bc2ffd | 969 algo+adaptive green; runner CLI overrides + relative_to bugfix verified by smoke. |
| Stage A | 4 envs × 5 methods × 3 seeds × 1k eps = 60 runs | DONE | 2026-04-26 | 2026-04-26 | 0f19d19d | 60/60 completed in 16s; all artifacts; manifest clean. |
| Stage A summary | `results/adaptive_beta/stage_A_summary.md` + auto-gate | DONE | 2026-04-26 | 2026-04-26 | 8071465b | **Strict gate FAILED** on all 4 envs. Criterion 1 (AUC paired diff > 0) failed everywhere. Mechanism strong on RPS (align=0.81, d_eff=0.47) but final-return effect within noise at 3 seeds. Self-play NOT promoted to Stage B per §22.4. |
| Stage A extended (rps) | autonomous tie-breaker (10 seeds × 5k eps) | DONE | 2026-04-26 | 2026-04-26 | f9d7e6bd | 50 runs in 81s. AUC paired diff +1276 ± 120. Cleared strict gate; rps promoted. |
| Stage B (rps) | 10 seeds × 10k eps × 6 valid methods = 60 runs | DONE | 2026-04-26 | 2026-04-26 | eac654e3 | AUC paired diff +1791 ± 171. Mechanism active. 0 div eps for adaptive_β/no_clip. |
| Stage B summary | `stage_B_summary.md` | DONE | 2026-04-26 | 2026-04-26 | eac654e3 | Promotes rps to Stage C. |
| Stage C (rps) | 20 seeds × 10k eps × 5 core methods = 100 runs | DONE | 2026-04-26 | 2026-04-26 | eac654e3 | Headline: AUC paired-bootstrap CI [+1473, +1970]; align=0.792, d_eff=0.568 (γ=0.95). |
| Stage C summary | `stage_C_summary.md` | DONE | 2026-04-26 | 2026-04-26 | eac654e3 | 3 of 5 spec §0 predictions confirmed on rps. |
| Final report | `final_report.md` (spec §17) | DONE | 2026-04-26 | 2026-04-26 | eac654e3 | Full structure; appendix recommendation. |
| Final recommendation | `final_recommendation.md` | DONE | 2026-04-26 | 2026-04-26 | eac654e3 | Replaces provisional negative memo at ea2a9271. |
| Paper appendix draft | `paper/phase_VII_appendix_draft.tex` | DONE | 2026-04-26 | 2026-04-26 | eac654e3 | NOT linked into main paper input chain. User decides merge. |
| Stage B | promoted envs × 8 methods × 10 seeds × 10k eps | CONDITIONAL | | | | |
| Stage B summary | `stage_B_summary.md` + Stage C decision | CONDITIONAL | | | | |
| Stage B Codex gate | `/codex:review` + triage | CONDITIONAL | | | | |
| Stage C | headline env × core methods × 20 seeds × 10k eps | CONDITIONAL | | | | |
| Stage C summary | `stage_C_summary.md` | CONDITIONAL | | | | |
| Final analysis | figures + tables + processed/ + final_recommendation.md | CONDITIONAL | | | | |
| Paper update | revise §Experiments only if data supports | CONDITIONAL | | | | |
| Final Codex gate | `/codex:review` + `/codex:adversarial-review` + verifier | CONDITIONAL | | | | |

---

## Codex gate verdicts

### M1 gate (2026-04-26, HEAD = afeacf9d)

**verifier:** PASS — 1053 passed, 1 pre-existing skip, 0 failures (full repo minus tests/audit). Equivalence test file contributes 97 (β,γ,r,v) tuples; full algorithms suite 788 green.

**`/codex:review` (job review-mofh9hay-a83i2n):** 2 P2 findings.
- [P2] `tab_operator.py:122-123` — `rho_batch` / `effective_discount_batch` lose broadcasted shape on classical-collapse path (`np.full_like(r_arr, ...)` preserves `r`'s shape instead of broadcasted output shape).
- [P2] `schedules.py:152-158` — `beta_for_episode(e)` ignores `episode_index`, returns `last_beta_used` for every e. After any update, every past episode reports the latest β. Breaks any delayed logging downstream (`beta_deployed` schema column).

**`/codex:adversarial-review` (job review-mofhc24g-qi6sc7):** verdict NEEDS-ATTENTION. Same 2 HIGH-severity findings (confirms above) + 2 MEDIUM:
- [MAJOR-3] `build_schedule._resolved_hparams()` silently drops unknown keys (typos like `beta_caps`, `lamda_smooth` accepted; foot-gun for M3 agent configs).
- [MAJOR-4] Equivalence test does not cover `compute_safe_target_ev_batch()` (3D expectation form). Silent regression risk in scalar→batch→3D refactor.

**Triage:** all 4 findings are real and load-bearing for M3 agent. Fixing all 4 before proceeding to M2.

| ID | Severity | File | Owner | Status |
|----|----------|------|-------|--------|
| BLOCKER-1 | HIGH | `src/lse_rl/operator/tab_operator.py:110-141` | operator-theorist | DONE @ 61754a57 |
| BLOCKER-2 | HIGH | `experiments/adaptive_beta/schedules.py:152-158` | calibration-engineer | DONE @ 61754a57 |
| MAJOR-3 | MED  | `experiments/adaptive_beta/schedules.py:63-72` | calibration-engineer | DONE @ 61754a57 |
| MAJOR-4 | MED  | `tests/algorithms/test_phase_VII_operator_kernel_equivalence.py:86-123` | operator-theorist | DONE @ 61754a57 |

**M1 gate verdict:** PASS. 876 tests green (algorithms + adaptive_beta). Proceeding to M2.

### Final Codex gate (2026-04-26, HEAD = eac654e3) — 6 substantive findings + 2 bookkeeping FAILs

**`/codex:review` (job review-mofl3eyj-qyacyj):**
- [P1] Off-by-one β in `agents.py:269-278`: `end_episode()` reads `sched_diag` after `update_after_episode` advances the schedule; `beta_raw`/`beta_deployed` columns shifted by one episode in episodes.csv/metrics.npz.
- [P2] Regret on non-bandit envs computed against zero (just `−return`); meaningless on rps/hazard/chain.
- [P2] Manifest append not concurrent-safe (read-modify-write race).

**`/codex:adversarial-review` (job review-mofl5zsh-phuovg):**
- [HIGH] `_resolve_seed_assignment` uses different `agent_seed` per method ⇒ different ε-greedy RNG streams. Per spec §8.4 this is by design (env stream is paired; ε offset is method-specific), but it leaves a confound for the headline AUC claim. Need an exploration-stream-parity control run.
- [HIGH] Stage C paired-bootstrap CI [+1473, +1970] cited in memos is NOT computed by `analyze.py` — only mean+SE is persisted. Reproducibility gap.
- [MED] `analyze.py` skips `status != "completed"` records ⇒ silent survivorship if a run hard-fails before metrics.npz writes.

**Verifier audit FAILs:**
- §16.5: parquets emit `mean_alignment_rate_last_500` / `mean_d_eff_last_500` — spec requires `align_rate` / `mean_d_eff`. Cosmetic; data correct.
- §16.7: manifest has 297 entries (not the 270 expected). The 27 surplus on `dev` are **test-smoke runs** from `test_reproducibility.py` and `test_runner_smoke.py` invoking the runner via subprocess; the runner appends to the global manifest path regardless of `--out`. Pollution bug.

| ID | Severity | File | Owner | Status |
|----|----------|------|-------|--------|
| FINAL-BLOCKER-1 | HIGH | `experiments/adaptive_beta/agents.py:269-278` (off-by-one β) | algo-implementer | PENDING |
| FINAL-HIGH-2 | HIGH | `experiments/adaptive_beta/analyze.py` (paired-bootstrap CI not persisted) | plotter-analyst | PENDING |
| FINAL-MAJOR-3 | MED  | `experiments/adaptive_beta/run_experiment.py:187-197` (regret) | experiment-runner | PENDING |
| FINAL-MAJOR-4 | MED  | `experiments/adaptive_beta/analyze.py:816-823` (silent fail-skip) | plotter-analyst | PENDING |
| FINAL-MAJOR-5 | MED  | `experiments/adaptive_beta/run_experiment.py:243-247` (manifest concurrency, test pollution) | experiment-runner | PENDING |
| FINAL-§16.5 | LOW  | parquet column aliases | experiment-runner | PENDING |
| FINAL-confound | INFO | exploration-stream parity control run on rps Stage C | orchestrator | PENDING |

---

## Failures / retries / deviations

(to be filled in)

---

## Final status

(to be written at end-of-run)
