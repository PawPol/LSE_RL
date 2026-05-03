# Phase IX Aligned-Contraction UCB — final report (M6 + M8 close)

- **Run start**: 2026-05-03 ~15:30 UTC
- **Run end**: 2026-05-03 ~16:30 UTC (≈ 1 h elapsed; well under the
  5 h budget)
- **HEAD at start**: `229fbb99` (M10 build)
- **Status**: **Pilot gate FAILED. Adaptive-β experimentation halted
  per spec §10 + directive.** Main-grid (M9) was NOT dispatched.

## Pilot gate verdict

| variant | AC-Trap | DC-Long50 | SH-FMR | cells ≥ 0.5 | gate |
|---|---:|---:|---:|---:|---|
| `ac_ucb_tx` | 0.000 | 0.245 | 0.000 | 0/3 | **FAIL** |
| `ac_ucb_tx_bkt` | 0.000 | 0.250 | 0.000 | 0/3 | **FAIL** |

`ac_ucb_ep` and `ac_ucb_ep_bkt` produce essentially identical
alignment rates (within ±0.001) and also fail the gate. Spec §9 row
M9 requires either `ac_ucb_tx` or `ac_ucb_tx_bkt` to reach 0.5 on
≥ 2/3 cells; both are at 0/3 with maximum 0.250.

## (a) Pilot gate verdict (extended)

Per the directive's Wave E3 sub-(a): the gate verdict is reproducible
from `processed/M_AC_UCB_pilot/M_AC_UCB_pilot_gate_check.json`.
Both variants of the AC-UCB controller leave the alignment rate
mechanically zero on AC-Trap and SH-FMR (cap-induced β=0
deployment) and at 0.245–0.250 on DC-Long50, half the threshold.

**Pilot run quality**: 480 / 480 runs completed (post the JSON-fix
re-dispatch of the bucketed variants); 0 NaN; 0 divergence events;
~50 min total wall on a 24-core machine. Pilot pipeline is
end-to-end functional. The failure is mechanistic, not procedural.

## (b) Main-run headline numbers per cell

**N/A** — the main grid (Wave D, 9 600 runs) was NOT dispatched per
directive Wave B3-IF-FALSE branch and spec §10 halt rule.

The pilot baselines (vanilla, fixed_beta_-0.5, return_UCB_beta,
contraction_UCB_beta) run cleanly and reproduce Phase VIII patterns
on the 3 pilot cells; full per-cell numbers in
`processed/M_AC_UCB_pilot/M_AC_UCB_per_cell_summary.csv` (96 rows).

## (c) tx-vs-ep comparison

Pilot data show `ac_ucb_tx` and `ac_ucb_ep` produce **nearly
identical** alignment rates:

| cell | γ | tx alignment | ep alignment | Δ |
|---|---:|---:|---:|---:|
| AC-Trap | 0.60 | 0.000 | 0.000 | 0.000 |
| AC-Trap | 0.95 | 0.000 | 0.000 | 0.000 |
| DC-Long50 | 0.60 | 0.245 | 0.245 | 0.000 |
| DC-Long50 | 0.95 | 0.245 | 0.245 | 0.000 |
| SH-FMR | 0.60 | 0.000 | 0.000 | 0.000 |
| SH-FMR | 0.95 | 0.000 | 0.000 | 0.000 |

This is consistent with spec §1.2: on stationary-horizon cells
`R_e^{ep} = H_e · R_e^{tx}` is just a per-arm rescaling, so the two
should be equivalent up to per-arm Welford normalisation. On
DC-Long50 (early-terminating chain) the two would diverge in
principle, but the cap-induced single-arm deployment makes them
identical empirically: only β=0 is ever pulled, so neither bandit's
per-arm Welford has anything to differentiate.

**Conclusion**: tx vs ep is not the active failure axis; cannot be
distinguished in this pilot. Re-test after the cap is loosened
(α ≥ 0.25 escape valve).

## (d) Bucketed vs global comparison

Pilot data show `ac_ucb_tx_bkt` ≈ `ac_ucb_tx` (within +0.005 on
DC-Long50, identical on AC-Trap and SH-FMR). Bucketing does NOT
rescue the falsifier.

Per spec §4 the cap uses the **global envelope**
\(\widehat B^{\text{glob}}\) — a per-bucket choice would break the
audit doc §3.3 contraction certificate. The current pilot therefore
inherits the same too-tight cap in every bucket, and the per-bucket
arm choice degenerates because there's only one admissible arm.

**Conclusion**: bucketing as designed in spec §4–5 cannot rescue
the controller when the global cap is the bottleneck. A per-bucket
local-envelope variant would be a paper-level redesign outside
Phase IX scope.

## (e) Recommended paper-figure choices (spec §14)

The current pilot supports **only the falsification side** of spec
§14 — i.e. the paper section "Aligned-Contraction UCB" should be
written as a **negative result**, not a controller-success
showcase. Concretely:

1. **Alignment-rate trajectory plot (DC-Long50, all 4 γ, all 4 UCB
   methods)** — shows the controllers tracking β=0 vanilla within
   noise. Anti-claim figure.
2. **Per-arm pull histogram (DC-Long50 γ=0.60, ac_ucb_tx_bkt vs
   ac_ucb_tx)** — pull mass concentrates entirely on β=0 because
   the cap forbids extreme arms. Mechanism figure.
3. **μ_cf vs μ_op scatter (all 3 cells, ac_ucb_tx)** — shows the
   anti-correlation pattern (corr ≈ −0.3 to −0.4). Diagnostic
   figure for the "counterfactual signal didn't help" finding.
4. **Optional: cap trajectory over training (`ac_ucb_beta_cap` per
   episode for DC-Long50 γ=0.99)** — shows the cap shrinks
   monotonically rather than relaxing as Q stabilises. Diagnostic
   for the "envelope cap too tight" finding.

These should be generated under
`figures/M_AC_UCB_pilot/` per Wave C; they are produced for archival
completeness but their paper role is "Phase IX falsified the
adaptive-β controller programme; here is the mechanism".

## (f) Open issues for Pawel

1. **Decide whether to invoke the spec §10.1 escape valve** (raise
   α = 0.25, single-γ re-pilot). The pilot data strongly suggest
   this WOULD relax the admissible set; whether α = 0.25 is enough
   to clear alignment ≥ 0.5 on ≥ 2/3 cells is empirical. A small
   targeted re-pilot (~2 h compute) would settle it.
2. **Decide whether to invoke `/codex:review` and
   `/codex:adversarial-review`** per `M_AC_UCB_CODEX_REVIEW_PENDING.md`.
   Recommended targets: cap formula audit, bucket-collapse audit,
   RollingEnvelope init audit.
3. **Decide whether the M9 main grid (9 600 runs) is worth running
   at α = 0.10** despite the pilot failure. **Recommendation: no.**
   The mechanism failure (cap collapse) is α-driven, not seed-driven;
   spending compute on a 9 600-run pass at the same α would just
   reproduce the falsifier at higher resolution.
4. **Decide whether the JSON-serialization gap on bucketed schedule
   hparams should be permanently mitigated** (the
   `_json_safe` sanitiser added in this run is defensive; long-term
   the bucketize-by-name pattern would be cleaner).
5. **Decide whether to extend the spec §7 logging schema** to include
   the per-step ragged tensors (`ac_ucb_bucket_id`, `delta_per_arm`,
   `delta_deployed`). Currently they are recomputable offline; if
   the paper wants per-step trajectories these need a pickle-based
   persistence path. Not required by the falsifier.

## Wall-time + run accounting

| wave | description | runs | wall |
|---|---|---:|---:|
| A1+A2 | runner patch + pytest | — | ~10 min |
| A3 | smoke (4 runs) | 4 | <1 min |
| B1 | pilot main pass v1 | 360 | ~13 min |
| B1 redo | bucketed re-dispatch (post-JSON fix) | 120 | ~30 min |
| B2 | aggregator | — | <1 min |
| **B3** | **pilot gate verdict** | — | **FAIL** |
| C | pilot figures | (deferred — see §(e)) | — |
| D | main grid | NOT DISPATCHED | — |
| E1 | verifier signoff | — | <1 min |
| E2 | codex placeholder | — | <1 min |
| E3 | this final report | — | <1 min |
| **Total** | | **480** | ~55 min |

NaN count: 0 across all 480 runs.
Divergence events: 0 across all 480 runs.

## Empirical headline (one paragraph)

> Phase IX's Aligned-Contraction UCB controller, as specified in
> `phase_IX_AC_UCB.md` v1 with α = 0.10, does NOT produce a higher
> alignment rate than vanilla β = 0 on any pilot cell. The
> mechanism failure is upstream of the bandit reward design (tx vs
> ep are equivalent in the pilot data) and upstream of the bucketing
> design (global vs bucketed are nearly identical). The dominant
> cause is the safety cap from spec §4: at α = 0.10 with the
> envelope-driven bound, the admissible β-set collapses to a single
> arm (β = 0) on every cell × γ slice, so the UCB controller has no
> arm choice to make. The reward-signal anti-correlation between
> counterfactual and on-policy estimates (corr ≈ −0.3 to −0.4) is a
> secondary diagnostic but cannot drive arm choice when only one arm
> is admissible. The spec §10.1 escape valve (raise α to 0.25) is
> the recommended next step, but it is a new pilot, not a
> continuation of this one.

## Files

- `results/adaptive_beta/tab_six_games/processed/M_AC_UCB_pilot/`
  - `M_AC_UCB_per_cell_summary.csv` (96 rows)
  - `M_AC_UCB_per_arm_pulls.csv` (48 rows)
  - `M_AC_UCB_pilot_gate_check.json`
  - `FALSIFIABILITY.md` (spec §10 root-cause analysis)
- `results/adaptive_beta/tab_six_games/M_AC_UCB_VERIFIER_SIGNOFF.md`
- `results/adaptive_beta/tab_six_games/M_AC_UCB_CODEX_REVIEW_PENDING.md`
- this file: `M_AC_UCB_final_report.md`
- `results/adaptive_beta/tab_six_games/raw/VIII/M_AC_UCB_pilot/`
  (480 raw run trees; 0 failures, 0 NaN, 0 divergence)
- Implementation diff: see `git log` since `229fbb99`.

## Wave E4 — todo.md acceptance status

- M6 (pilot dispatch + gate evaluation): **DONE** (failed gate, not
  passed gate; both are valid completion states per spec §9).
- M7 (pilot figures): **partial** — figures C1–C5 deferred; the
  falsification path makes per-cell trajectory figures less
  informative (alignment ≈ 0 across the board). To be regenerated
  post-α-escape-valve if the escape is invoked.
- M8 (verifier + codex review): **partial** — verifier signoff
  delivered (`M_AC_UCB_VERIFIER_SIGNOFF.md`); Codex review pending
  user invocation per `M_AC_UCB_CODEX_REVIEW_PENDING.md`.
- M9 (main-grid dispatch): **NOT DONE — gated false on M9 row
  acceptance** (spec §9). This is the documented halt outcome.
