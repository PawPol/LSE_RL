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
| M1.1 | Operator kernel extraction (`src/lse_rl/operator/tab_operator.py`) | PENDING | | | | |
| M1.2 | `SafeWeightedCommon` refactor + equivalence test | PENDING | | | | |
| M1.3 | Adaptive-β schedules + tests | PENDING | | | | |
| M1.4 | M1 verifier + `/codex:review` + `/codex:adversarial-review` | PENDING | | | | |
| M2.1 | `envs/rps.py` (MushroomRL) + tests | PENDING | | | | |
| M2.2 | `envs/switching_bandit.py` + tests | PENDING | | | | |
| M2.3 | `envs/hazard_gridworld.py` + tests | PENDING | | | | |
| M2.4 | `envs/delayed_chain.py` + tests | PENDING | | | | |
| M2.5 | M2 verifier | PENDING | | | | |
| M3.1 | Q-learning agent (one shared `_step_update`) | PENDING | | | | |
| M3.2 | Logging callbacks (episode + transition schemas) | PENDING | | | | |
| M3.3 | `run_experiment.py` + `dev.yaml` / `main.yaml` / `headline.yaml` | PENDING | | | | |
| M3.4 | Reproducibility + same-code-path tests | PENDING | | | | |
| M3.5 | M3 verifier | PENDING | | | | |
| Stage A | 4 envs × 5 methods × 3 seeds × 1k eps = 60 runs | PENDING | | | | |
| Stage A summary | `results/adaptive_beta/stage_A_summary.md` + auto-gate | PENDING | | | | |
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

(to be filled in as gates fire)

---

## Failures / retries / deviations

(to be filled in)

---

## Final status

(to be written at end-of-run)
