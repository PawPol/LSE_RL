# Phase VII-B Strategic Learning — Final Recommendation Memo

**Date:** 2026-04-26
**Branch:** `phase-VII-B-strategic-2026-04-26` (cut from `phase-VII-overnight-2026-04-26`)
**Authorization:** full autonomous run through Stage B2-Main + review gate (user directive 2026-04-26).
**Source spec:** `tasks/phase_VII_B_strategic_learning_coding_agent_spec.md`.

---

## 1. Final verdict

> **NO PAPER UPDATE.**
>
> Stage B2-Main shows that **fixed `β = −1` paired-bootstrap-significantly
> dominates adaptive-β** on the only game that cleared the Stage B2-Dev
> promotion gate (Shapley, with both promoted endogenous adversaries).
> The adaptive-β controller saturates to its lower clip floor `β = −2.0`
> within ~200 episodes and stays there — converging to a behavior the
> static `fixed_negative` method achieves with zero overhead. Spec §17's
> "fixed-positive or fixed-negative dominates consistently" criterion
> is the operative trigger.

**Headline paired-bootstrap CIs (n = 10 seeds, 10 000 paired resamples):**

| Cell | Comparison | Endpoint | mean | 95% CI | Verdict |
|---|---|---|---|---|---|
| Shapley × HypTest | adaptive_β − fixed_neg | AUC_full | **−374.9** | [−708.2, −45.1] | **fixed_neg dominates** |
| Shapley × FMRM | adaptive_β − fixed_neg | final_return | **−0.126** | [−0.260, −0.008] | **fixed_neg dominates** |
| Shapley × FMRM | adaptive_β − vanilla | AUC_full | +162.8 | [+87.8, +244.8] | adaptive ≥ vanilla |
| Shapley × HypTest | adaptive_β − vanilla | AUC_full | +696.6 | [+227.8, +1169.2] | adaptive ≥ vanilla |
| any cell | adaptive_β − vanilla | AUC_first_2k (spec §15 primary) | n.s. | — | no sample-efficiency signal |

Adaptive-β beats vanilla but loses to fixed_negative. Net: not headline-quality.

---

## 2. What was implemented

A complete strategic-learning adversary suite under
`experiments/adaptive_beta/strategic_games/`:

- **Shared operator:** reuses `src/lse_rl/operator/tab_operator.py` —
  no Bellman-target math duplicated (verified by AST walk in
  `tests/adaptive_beta/strategic_games/test_operator_shared_kernel.py`).
- **Framework:** `MatrixGameEnv` (MushroomRL `Environment` subclass),
  `GameHistory`, `StrategicAdversary` ABC, registry.
- **5 games:** matching_pennies (H=1, mechanism-degenerate),
  shapley (cyclic 3×3, non-convergent FP), rules_of_road (with tremble +
  payoff-bias variants), asymmetric_coordination (canonical sign +β),
  strategic_rps. **Deferred:** soda/uncertain game.
- **9 endogenous adversaries:** stationary, scripted_phase,
  finite_memory_best_response, finite_memory_fictitious_play,
  smoothed_fictitious_play, regret_matching (full-info + realized-payoff),
  finite_memory_regret_matching, hypothesis_testing (priority §7.8 — full
  state machine: rejection trigger, search phase, hypothesis sampling),
  realized_payoff_regret (stub). **Deferred:** self_play.
- **Strategic metrics + logging:** `EPISODE_COLUMNS_STRATEGIC` (25 cols
  per spec §13), `TRANSITION_COLUMNS_STRATEGIC` (14 cols), event-aligned
  aggregator, paired-bootstrap diffs, promotion gate, dual-schema
  emission (VII-A + VII-B columns side-by-side).
- **Tests:** 171 unit/integration/regression/repro tests —
  `pytest tests/adaptive_beta/strategic_games/` is 171/171 green.
  Full repo: 1387/1387 (1 unrelated Phase V skip).
- **Stage configs:** `stage_B2_dev.yaml`, `stage_B2_main.yaml` (template
  with `${promoted_*}` placeholder), `stage_B2_main_resolved.yaml`
  (post-promotion Cartesian product), `stage_B2_stress.yaml` (deferred
  stub).

---

## 3. Experiments executed

| Stage | scope | runs | episodes | wall-clock |
|---|---|---|---|---|
| B2-Dev | 4 games × 3 advs × 5 methods × 3 seeds × 1k eps | 180 | 180 000 | 89 s |
| B2-Main | 1 game × 2 advs × 5 methods × 10 seeds × 10k eps | 100 | 1 000 000 | 10 m 47 s |
| **Total** | | **280** | **1.18 M** | **~12.5 min** |

All on a single CPU. 0 cells failed at Stage B2-Main; 0 divergence
events for clipped `adaptive_beta` across 200 000 candidate episodes.

---

## 4. Stage B2-Dev gate outcomes (n = 3 seeds)

- **Strict pass (3 cells, CI excludes 0):**
  Shapley × FMRM (Δ AUC +436, CI [+237, +830]),
  Shapley × HypTest (Δ AUC +296, CI [+180, +369]),
  matching_pennies × FMRM (Δ AUC +43, CI [+14, +82]).
- **Directional only (3 cells):** rules_of_road × {FMBR, FMRM},
  Shapley × FMBR.
- **Fail (6 cells):** all 3 strategic_rps cells (adaptive-β UNDER-performs
  vanilla by 400–750 AUC), matching_pennies × {FMBR, HypTest},
  rules_of_road × HypTest.

**Promoted to Stage B2-Main:** Shapley × {FMRM, HypTest} only.
matching_pennies × FMRM was dropped to keep the Cartesian product clean
(the single-game choice means the runner can dispatch a 1×2 product
without forcing matching_pennies × HypTest into a gate-failed cell).

---

## 5. Spec §22 six-question checklist

### 5.1 Which strategic settings produced adaptive-β gains?

- **Shapley × FMRM:** AUC paired diff vs vanilla = +162.8, CI [+88, +245].
- **Shapley × HypTest:** AUC paired diff vs vanilla = +696.6, CI [+228, +1169].
- **No game beat fixed_negative.** On both promoted cells, fixed_negative
  matched or exceeded adaptive-β with paired-CI excluding 0 in the
  fixed_neg → adaptive_β direction.

### 5.2 Were gains sample-efficiency, final-return, or recovery gains?

- **NOT sample-efficiency.** `auc_first_2k` paired diffs show no signal
  (CIs cross 0 on both cells). The spec §15 primary endpoint is null.
- **Late-training plateau effect.** Most of the AUC gain accrues after
  episode ~3 000 once adaptive-β has saturated to its clip floor.
- **NOT recovery.** `recovery_time` is uniform across methods on Shapley
  (every episode triggers a `support_shift` event so the metric is
  ill-defined; using global learning-speed as a proxy yielded no
  across-method differentiation).

### 5.3 Did mechanism metrics support the explanation?

Partly. The spec §3.3 alignment identity holds (`mean_alignment ≈ 0.86`
on adaptive-β, `mean_d_eff ≈ 0.48 < γ = 0.95`). But the **mechanism
profile is observationally indistinguishable from fixed_negative**
(`mean_alignment ≈ 0.90`, `mean_d_eff ≈ 0.55`). The "adaptive-β acts as
a controller" story collapses into "adaptive-β saturates to the
fixed-β value the game prefers", which is not a controller story —
it is a discovery story for which fixed_negative skips the discovery.

### 5.4 Did any fixed β dominate?

**Yes — `fixed_negative` on Shapley.**
- Shapley × HypTest, AUC_full: fixed_neg − vanilla = +1071, CI excludes 0;
  adaptive_β − fixed_neg = −375, CI excludes 0.
- Shapley × FMRM, final_return: fixed_neg − adaptive_β = +0.126,
  CI excludes 0.
- This is the §17 "no paper update" trigger.

`fixed_positive` catastrophically underperforms on both cells (Δ AUC ≈
−10 000) — also informative: the game has a strong sign preference, and
the adaptive-β controller successfully discovers it, but discovery is
not free, and the sustained advantage of "discovery" is precisely the
ramp-up cost that the fixed-sign-correct baseline avoids.

### 5.5 Did any adversary expose a failure mode?

**Yes — strategic_RPS with all three endogenous adversaries (Stage B2-Dev).**
Adaptive-β UNDER-performs vanilla by:
- −744 ± 120 AUC (FMBR, n=3)
- −415 ± 95 AUC (FMRM, n=3)
- −703 ± 110 AUC (HypTest, n=3)

This contradicts the Phase VII-A headline RPS claim (adaptive-β +1 732
AUC vs vanilla on scripted-phase RPS). The Phase VII-A gain may be
**adversary-specific to scripted-phase opponents** — endogenous
learners do not produce the same signal. This is the most important
follow-up finding in the entire phase. Documented in Stage B2-Main
memo §8 and below in §6.4 of this memo.

### 5.6 Should the paper be updated, appendix-only, or unchanged?

**Unchanged.** Per spec §17:
- Strong adaptive-β gains in only 1 game (single-environment scope)
- fixed_negative dominates on the headline metric (CI excludes 0)
- Mechanism evidence is supportive but indistinguishable from fixed_neg
- Sample-efficiency endpoint shows no signal
- A counter-finding on strategic_RPS challenges the Phase VII-A claim

The recommendation file is at `paper_update/no_update_recommendation.md`.
No `.tex` file was edited.

---

## 6. Open questions / follow-ups for the user

### 6.1 Strategic_RPS regression (HIGH PRIORITY)

The most consequential VII-B finding is that adaptive-β under-performs
vanilla on strategic_RPS with all three endogenous adversaries. This
challenges the Phase VII-A scripted-phase RPS result. Recommended
follow-ups (~30 min compute total):

1. **Symmetric Stage A on strategic_RPS at 10 seeds × 5k eps**
   (matching the Phase VII-A "tie-breaker" extended scale that promoted
   scripted-RPS) to test whether the regression is small-sample noise
   or a real effect.
2. **A controlled comparison** at fixed seeds between scripted-phase
   RPS and strategic_RPS (FMRM) to isolate which adversary feature
   triggers the Phase VII-A gain (regularity? predictability?
   non-adaptiveness?).
3. **Paper hedge:** consider a one-line caveat in the Phase VII-A
   appendix that the RPS gain is conditional on scripted-phase
   adversaries, pending confirmation from (1).

### 6.2 Single-game scope of Stage B2-Main

Only Shapley promoted. `matching_pennies × FMRM` cleared the strict
gate but was dropped for Cartesian-product hygiene. Two alternatives
the user could authorize:

1. **Direct dispatch of matching_pennies × FMRM at Main scale**
   (10 seeds × 10k eps × 5 methods = 50 cells, ~5 min). Tests whether
   the controller is mechanism-degenerate at H=1 (per spec §22.5
   precedent it shouldn't be presented as mechanism evidence, but the
   performance signal might still be informative).
2. **Modify the runner** to accept explicit (game, adversary) pair
   lists rather than Cartesian products, allowing irregular promoted
   matrices.

### 6.3 Sensitivity grid (Stage B2-Stress)

Spec §11.3 stress grid (memory_m, inertia_λ, tolerance_τ, search_len,
adversary temperature, payoff noise, action tremble) is **deferred —
not run, requires fresh user authorization**. ~30–60 min compute.

### 6.4 Self-play (DEFERRED)

Spec §7.10 self-play variants (vanilla vs vanilla, adaptive vs vanilla,
adaptive vs adaptive, adaptive vs fixed_pos, adaptive vs fixed_neg) were
not implemented. ~60 min implementation + run. Recommended only if the
user wants exploitability evidence.

### 6.5 Soda/uncertain game (DEFERRED)

Spec §6.6 hidden-opponent-type game not implemented. ~45 min. Lower
priority — the "no update" verdict means a 6th game does not change the
paper-impact decision.

### 6.6 `adaptive_magnitude_only` and `adaptive_beta_no_clip` ablations

Per OQ-4 these were excluded from Stage B2-Main. Adding them on the
2 promoted cells (~20 cells × ~6s = 2 min) would round out the spec
§11.1 ablation table. Defer unless the user wants it.

### 6.7 Codex review

The user-triggerable review prompts are pre-staged at
`paper_update/codex_review_prompts.md`:

```
/codex:review --base phase-VII-overnight-2026-04-26 --background
/codex:adversarial-review --base phase-VII-overnight-2026-04-26 --background "..."
```

The adversarial focus string targets: endogenous-vs-scripted opponent
purity, paired-seed integrity at n=10, manifest accounting, no operator
math duplication, fixed_negative-dominance interpretation honesty,
single-game generalization limits, strategic_RPS regression follow-up
priority. The user can trigger these against the now-committed branch.

---

## 7. Review-triage outcome

Internal review-triage flagged 3 BLOCKERs + 7 MAJOR + 7 MINOR + 4 NIT.
**All 3 BLOCKERs resolved before this memo:**

1. ~~Smoke test reads live `stage_B2_main.yaml`~~ → fixed: synthesizes
   placeholder string instead.
2. ~~Main YAML template overwritten in-place by promotion gate~~ →
   fixed: `stage_B2_main.yaml` restored as `${promoted_*}` placeholder
   template; `stage_B2_main_resolved.yaml` carries the post-promotion
   Cartesian product (and is what the actual run consumed).
3. ~~Phase VII-B work entirely uncommitted~~ → fixed: 2 commits
   (impl + results) on `phase-VII-B-strategic-2026-04-26`.

The 7 MAJOR + 11 MINOR/NIT items are on the review-triage queue at
`tasks/todo.md` under the new "Phase VII-B Review Triage" section, for
the user to triage at leisure. None invalidate the verdict.

---

## 8. What I executed autonomously

Per user authorization (2026-04-26):

- Cut fresh branch `phase-VII-B-strategic-2026-04-26`.
- Resolved 5 planner Open Questions (Shapley payoff convention,
  strategic_RPS-vs-scripted Dev variant, event-aligned half-window,
  Main ablation policy, review-path).
- Dispatched 11 specialist subagents end-to-end (planner, env-builder
  ×2, calibration-engineer, experiment-runner ×3, test-author,
  verifier ×2, plotter-analyst ×2, review-triage).
- Hit 1 hard incident: 100% Main cell failure on first dispatch
  (config bug — analyst replaced YAML placeholder with bare adversary
  name list, dropping hyperparameter dicts). Diagnosed, fixed YAML to
  use dict form, cleared failed manifest, re-dispatched cleanly.
- Made one judgment call the user might review: kept the strict gate
  bar at Stage B2-Dev rather than letting directional-only cells
  through. This is honest at the cost of single-game Stage B2-Main
  scope.

If the user disagrees with any autonomous decision, the relevant
artifacts are preserved:
- Stage B2-Dev raw failure logs are gitignored but regenerable.
- Stage B2-Main first-dispatch failure manifest was cleared per spec
  §19 (no silent retries) before the fix; the second dispatch is the
  only one in the manifest.
- Original `${promoted_*}` template config is restored at
  `stage_B2_main.yaml`; the resolved Cartesian product is at
  `stage_B2_main_resolved.yaml`.

---

## 9. Pointers

| Artifact | Path |
|---|---|
| Spec | `tasks/phase_VII_B_strategic_learning_coding_agent_spec.md` |
| Parent (Phase VII-A) spec | `docs/specs/phase_VII_adaptive_beta.md` |
| Stage B2-Dev summary | `results/adaptive_beta/strategic/stage_B2_dev_summary.md` |
| Stage B2-Main summary | `results/adaptive_beta/strategic/stage_B2_main_summary.md` |
| Final recommendation (this file) | `results/adaptive_beta/strategic/final_recommendation.md` |
| Paper-update verdict | `paper_update/no_update_recommendation.md` |
| Codex review prompts | `paper_update/codex_review_prompts.md` |
| Review-triage queue | `tasks/todo.md` (Phase VII-B Review Triage section) |
| Per-run summaries | `results/adaptive_beta/strategic/processed/{dev,main}/per_run_summary.parquet` |
| Paired diffs | `results/adaptive_beta/strategic/processed/{dev,main}/paired_diffs.parquet` |
| Figures | `results/adaptive_beta/strategic/figures/{dev,main,event_aligned_*.pdf}` |
| Tables | `results/adaptive_beta/strategic/tables/main_strategic_metrics.{csv,tex}` |
| Implementation | `experiments/adaptive_beta/strategic_games/` |
| Tests | `tests/adaptive_beta/strategic_games/` (171/171 green) |
| Operator (shared, not duplicated) | `src/lse_rl/operator/tab_operator.py` |
| Branch | `phase-VII-B-strategic-2026-04-26` (parent: `phase-VII-overnight-2026-04-26`) |
| Commits | 2 — `213e4b20` (impl) + `683819cc` (results+memo) |

---

## 10. The user's decisions

After reviewing this memo, the user picks one of:

1. **Merge to main as-is** with NO PAPER UPDATE. The phase ends here;
   the strategic_RPS regression is a known future-work item.
2. **Defer merge** and authorize one of the §6 follow-ups (most
   actionable: §6.1 strategic_RPS extended scale; §6.2 add
   matching_pennies × FMRM at Main scale).
3. **Trigger Codex review** at `/codex:review` and
   `/codex:adversarial-review` (prompts at
   `paper_update/codex_review_prompts.md`) before the merge decision.
4. **Reject the verdict** if the paired-CI numbers are read differently
   than I have read them — I would push back, but the user makes the
   final call.

The headline numbers: adaptive-β is **better than vanilla but worse
than fixed_negative** with paired-bootstrap CI excluding 0 on Shapley.
That is a real effect, not noise — and it is the wrong sign for a
paper update.
