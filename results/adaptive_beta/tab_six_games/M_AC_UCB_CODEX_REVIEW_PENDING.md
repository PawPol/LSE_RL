# M_AC_UCB pilot — Codex review pending (user-invoked)

- **Created**: 2026-05-03
- **Status**: Codex review NOT yet run; placeholder to remind Pawel
  to invoke `/codex:review` and `/codex:adversarial-review` after
  reading the falsification verdict.

## Why a placeholder

`/codex:review` and `/codex:adversarial-review` are billed,
user-triggered slash commands. The orchestrator cannot launch them
autonomously. Per the directive's Wave E2: "Codex review is
user-invoked; write a placeholder note pointing at the spec + impl
diff + figures so Pawel can run the slash commands manually."

## What to review

Suggested invocation order:

1. `/codex:review` — native working-tree review of:
   - The Stage 5 runner patch (`experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage5_adaptive.py`,
     +263/−5 LOC) including the AC-UCB method dispatch and the
     JSON-serialization sanitiser added to the run.json writer.
   - The pilot config (`experiments/adaptive_beta/tab_six_games/configs/M_AC_UCB_pilot.yaml`,
     method-name corrections to canonical Phase VIII naming).
   - The aggregator (`experiments/adaptive_beta/tab_six_games/analysis/aggregate_AC_UCB.py`,
     unchanged in this run, but worth a sanity pass).
   - The dispatch driver (`scripts/m_ac_ucb_pilot_dispatch.py`,
     15-shard parallel ProcessPool).

2. `/codex:adversarial-review` with focus text:
   `"adversarial: scrutinise spec §10 falsifiability conclusion. Are the
   diagnostics (median admissible_size = 1; corr(μ_cf, μ_op) ≈ −0.3 to
   −0.4) sufficient to claim AC-UCB is mechanism-broken at α=0.10, or is
   there a configuration/measurement bug masquerading as algorithm
   failure? Specifically test: did RollingEnvelope initialise with a
   pessimistic ⌃B_0 that locks the cap closed? Did the bucket key
   collapse to a single bucket (bucketing→global) and undermine the
   bucketed comparison?"`

## Pointers for Codex

- Spec authority: `docs/specs/phase_IX_AC_UCB.md` §1–§14; key
  sections: §4 (cap formula), §5 (bucketing), §9 (gate),
  §10 (falsifiability path).
- Pilot artifacts: `results/adaptive_beta/tab_six_games/processed/M_AC_UCB_pilot/`
- Falsifiability analysis: `processed/M_AC_UCB_pilot/FALSIFIABILITY.md`
- Final report: `M_AC_UCB_final_report.md`
- Verifier sign-off: `M_AC_UCB_VERIFIER_SIGNOFF.md`
- Implementation diff: `git diff <last-pre-IX-commit>...HEAD`
  (HEAD will be a fresh commit landing the Phase IX runner patch
  + falsification artifacts).

## Suggested adversarial questions to test

1. **Cap-init audit**: Is `RollingEnvelope.__init__` setting the
   initial bound to a pessimistic value that locks the admissible
   set to {β=0} from episode 0? Read
   `experiments/adaptive_beta/calibration/rolling_box.py`.
2. **Bucket collapse**: For matrix-game cells, does
   `matrix_opp_and_step` actually produce ≥ 2 distinct buckets in
   the pilot? If it always returns the same key, bucketed degenerates
   to global.
3. **Spec §4 cap formula vs paper Thm**: re-derive
   `\bar\beta_e = log(κ/[γ(1+γ−κ)]) / (R_max + ⌃B_glob)` from the
   audit doc §3.3 and verify the numerator stays positive. At γ=0.99
   and α=0.10: κ = 0.99 + 0.10·0.01 = 0.991; numerator =
   log(0.991/[0.99·(1.99−0.991)]) = log(0.991/0.989) ≈ 0.002. With
   denominator R_max + ⌃B_glob ≈ 1 + (some Q-magnitude), the cap is
   ~0.002 — well below the smallest non-zero arm magnitude (β=0.05).
   Quantitative confirmation that α=0.10 is the wrong default.
4. **`ac_ucb_tx` vs `ac_ucb_ep` sanity**: spec §1.2 claims `tx` and
   `ep` are equivalent on stationary-horizon cells. Empirically the
   pilot shows nearly identical alignment-rates between them; is the
   difference at within-noise level or evidence of a bug?

## Where to land Codex's verdict

If Codex agrees with the falsification → final report stays as-is.
If Codex disagrees and points to a specific bug → patch + re-pilot at
α = 0.25 per spec §10.1 escape valve, then re-evaluate the gate.

## Status

- Codex review: **NOT INVOKED.**
- Adversarial Codex review: **NOT INVOKED.**
- Both pending Pawel's manual invocation.
