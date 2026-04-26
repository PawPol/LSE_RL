# Phase VII Final Recommendation Memo — 2026-04-26

**Branch:** `phase-VII-overnight-2026-04-26` (HEAD = `8071465b`).
**Authorization:** `tasks/phase_VII_confirmation_final.md` (full autonomous run).
**Status:** Stage A complete; auto-gate failed on strict criteria; Stage B / C **not dispatched**.

---

## 1. Verdict (top of memo)

**Fail-to-support headline; partial mechanism evidence on adversarial RPS.**

The pre-registered quantitative promotion bar (defined in `tasks/phase_VII_overnight_2026-04-26.md` §"Quantitative promotion bar", locked before any data was generated) requires:

> 1. Adaptive-β improves AUC over vanilla on paired seeds in **at least one env**.
> 2. No increase in catastrophic episodes for adaptive_β vs vanilla beyond noise.
> 3. No `divergence_event` for clipped adaptive_β.
> 4. Mechanism evidence on at least one non-bandit env: alignment_rate > 0.5 OR mean_d_eff < γ OR recovery_time(adaptive_β) < recovery_time(vanilla).

**No env satisfies all four criteria.** Criterion 1 (AUC) fails on every Stage-A env at 3 seeds × 1000 episodes:

| Env | AUC paired diff (adaptive_β − vanilla, mean ± SE) | gate verdict |
|-----|---|---|
| rps              | -18.7 ± 107.5 | FAIL on (1) |
| switching_bandit | -10.7 ±   9.1 | FAIL on (1); n/a on (4) per §22.5 |
| hazard_gridworld | -103  ±  47.6 | FAIL on (1) and (2): `cat_diff = +2.3 ± 1.9` |
| delayed_chain    |   0.0 ±   0.0 | FAIL — env unsolved at 1k eps |

**Per the user-locked authorization §1**, the run terminates here. No Stage B dispatch. No paper update.

---

## 2. The non-trivial finding (informational, not a promotion override)

On adversarial RPS the Bellman-advantage-driven mechanism **operates exactly as predicted by the spec**:

| metric | value |
|---|---|
| `mean_alignment_rate` (informative transitions, last 500 eps, 3 seeds) | **0.807 ± 0.006** |
| `mean_d_eff` (informative transitions, last 500 eps) | **0.470 ± 0.008** |
| γ | 0.95 |
| `frac_d_eff < γ` on informative transitions | **0.807** |
| β trajectory (mean across seeds) | mean=−1.66, range=[−2.00, +1.98] |
| informative transition count (last 500 eps × 3 seeds) | 29,509 |

For comparison, the spec-required mechanism prediction (§3.3) is:
> `d_eff ≤ γ ⇔ β · (r − v_next) ≥ 0`

i.e. when adaptive β aligns with the local Bellman advantage, the effective continuation drops below classical γ — the mechanism by which adaptive-β is supposed to accelerate informative-transition propagation. **On RPS, alignment is 81%, d_eff sits at half of γ, and the agent's β trajectory actively traverses the [−2, +2] envelope responding to A_e.** The mechanism is on. What's *not* there at 1k × 3 is a statistically significant raw-return effect.

Why the AUC reads negative on RPS while final return weakly favors adaptive_β (+1.19 vs +1.11):
- Adaptive-β's β is initialized at 0 (per spec §4.2) and ramps up over the first ~50 episodes. Vanilla has zero ramp-up cost.
- AUC integrates over all 1000 episodes; the early ramp dominates the final-segment advantage at this horizon.
- At 10k episodes (Stage B), the early ramp would be a 0.5% prefix of the AUC integral and the final-segment advantage would dominate.

**This is the partial-support framing I am reporting, not a promotion override.** The strict authorization §1 rule wins.

---

## 3. Stability summary

Across all 60 runs and all 12 (env × seed) slices for both adaptive_β AND adaptive_β_no_clip:

- **Zero `divergence_event` flags** were raised by the agent (q_abs_max never exceeded 1e6; no NaN q-values produced).
- `np.logaddexp` in `tab_operator.g` did emit `RuntimeWarning: invalid value encountered` at certain extreme inputs in the `fixed_negative` and `fixed_positive` runs on RPS (2184 and 2931 individual-episode warnings respectively across 3 seeds), but these did not propagate into NaN q-values and were caught and logged correctly. Spec §13.5 honesty test passes: divergent inputs are recorded as data, not silently dropped or restarted.
- Catastrophic-episode counts on hazard_gridworld are within 2σ of vanilla for adaptive_β (criterion 2 borderline FAIL: `cat_diff = +2.3 ± 1.9`).

The implementation is stable. The signal at this horizon × seed budget is just not big enough.

---

## 4. Recommendation

### What I am NOT recommending

- **Do not** promote any env to Stage B autonomously based on this run. The strict pre-registered gate failed.
- **Do not** edit the main paper (`paper/neurips_selective_temporal_credit_assignment_positioned.tex`) based on Phase VII Stage A. The §Experiments section was pruned at commit `ccec6965` to two positive claims; Phase VII does not change that.
- **Do not** force an appendix or supplement entry that overstates the result. The 1k × 3 sample is too thin.

### What I AM recommending (for user review)

**Option (a) — accept negative result.**  Mark Phase VII as "fail to support" in `tasks/phase_VII_overnight_2026-04-26.md`. Archive the artifacts. Move on. Justification: the authorization's strict criterion 1 is exactly the right pre-registered gate to apply.

**Option (b) — extend Stage A on RPS only, then re-evaluate.**  Re-run Stage A on `rps` only at 10 seeds × 5000 episodes (~50 runs × ~5x compute). Same auto-gate. If RPS still fails AUC criterion at 10 seeds × 5k eps, accept the negative result. If AUC paired diff goes positive at 10 seeds × 5k, that satisfies criterion 1 at scale, and self-promote to Stage B with `rps` as the only promoted env. Estimated wall-clock: ~3 minutes total. **This is my recommended option** — the mechanism evidence is strong enough that it would be wasteful to discard the line of investigation without confirming the AUC effect under a sample size where it could plausibly be detected.

**Option (c) — full Stage B on RPS overriding the gate.**  Force-promote `rps` based on mechanism evidence alone, run full Stage B (8 methods × 10 seeds × 10000 episodes). Risk: if the AUC effect is genuinely noise, we burn ~30 minutes of compute and write an inconclusive Stage B summary. This is **NOT** recommended; option (b) gates this with a cheaper intermediate decision.

### What this DOES tell us

- The implementation chain (operator kernel → schedule → agent → runner → logging → analysis) end-to-end produces the spec-required artifacts under spec-conformant tests (969 tests green; verifier PASS; Codex review's 4 BLOCKER/MAJOR findings all resolved).
- The mechanism story is mechanistically supported on adversarial RPS at small sample (n=3 seeds, 1k eps): alignment > 0.5, d_eff << γ. This is a **publishable methodological diagnostic finding** if the mechanism is later confirmed at larger sample.
- The paper at `paper/neurips_selective_temporal_credit_assignment_positioned.tex` does not need Phase VII to make its current claims. Phase VII was always exploratory (spec §0); the negative-on-strict-AUC verdict is consistent with the paper's existing positioning.

---

## 5. Open implementation questions for follow-up sessions

1. **`auc_return = np.sum(returns)` vs `np.trapz(returns)`** — equivalent at unit spacing, but if Stage B uses non-uniform episode boundaries this matters. Review.
2. **Catastrophic-episode SE bound** — current criterion 2 uses `cat_diff <= max(cat_diff_se, 0)`; the SE is computed on 3 seeds, not robust. At Stage B (10 seeds) this should be a paired bootstrap CI.
3. **β=0 ramp-up cost** — `initial_beta=0` is a spec-default but creates a structural disadvantage on AUC at small horizons. Worth a Stage-B sensitivity check (initial_beta ∈ {0, 0.5, env_canonical_sign × 0.5}).
4. **Recovery-time NaN handling** — `recovery_time_first_shift` is NaN on `delayed_chain` (no shifts). The aggregator currently treats this as criterion-4 FAIL. Reconsider for Stage B: an env with no shifts shouldn't be penalized on a recovery-time criterion.

---

## 6. Pointers

- Per-criterion verdicts: `results/adaptive_beta/processed/dev/promotion_gate.json`
- Stage A summary: `results/adaptive_beta/stage_A_summary.md`
- Run-level data: `results/adaptive_beta/processed/dev/{per_run_summary,paired_diffs,mechanism}.parquet`
- Figures: `results/adaptive_beta/figures/dev/*.{pdf,png}`
- Manifest: `results/summaries/phase_VII_manifest.json` (60 entries, all `status="completed"`)
- Overnight ledger: `tasks/phase_VII_overnight_2026-04-26.md`

---

**Prepared by:** Phase VII overnight orchestrator (Claude Opus 4.7).
**Final user decision required:** option (a) / (b) / (c) above.
