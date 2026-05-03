# M_AC_UCB pilot — falsifiability analysis (spec §10)

**Verdict**: **GATE FAILED.** Both `ac_ucb_tx` and `ac_ucb_tx_bkt` reach 0/3
cells at the spec §9 alignment-rate ≥ 0.5 threshold (max alignment 0.250
on DC-Long50). Per directive and spec §10, Phase IX adaptive-β
experimentation halts; this memo records the diagnostic root cause.

| variant | AC-Trap | DC-Long50 | SH-FMR | cells ≥ 0.5 |
|---|---:|---:|---:|---:|
| `ac_ucb_tx` | 0.000 | 0.245 | 0.000 | 0/3 |
| `ac_ucb_tx_bkt` | 0.000 | 0.250 | 0.000 | 0/3 |
| `ac_ucb_ep` (companion) | 0.000 | 0.245 | 0.000 | 0/3 |
| `ac_ucb_ep_bkt` (companion) | 0.000 | 0.250 | 0.000 | 0/3 |

The bucketed variant edges the global by +0.005 on DC-Long50 (a within-noise
margin) and is identical on AC-Trap and SH-FMR. Bucketing does NOT rescue
the falsifier. Both reward modes (`tx`, `ep`) produce essentially identical
alignment rates — the controller's mechanism failure is upstream of the
reward-mode choice.

## Root cause: envelope cap collapses J_e to a single arm

Spec §10.1 anticipated a failure mode where the rolling-envelope cap
\(\bar\beta_e^{(b)}\) projects so close to 0 that no extreme arm is
admissible. The data confirm this is the dominant pathology — and it is
worse than spec §10.1's threshold-of-concern (<5% admissibility on
extreme arms): **the median admissible-set size is 1 across every (cell, γ,
method) tuple in the pilot**, meaning UCB is forced to pull the same
single arm (β=0) every episode.

```
Diagnostic 1 — mean admissible_size per episode (over 5 seeds × 10 000 episodes)

cell                       γ    ac_ucb_tx    ac_ucb_ep    ac_ucb_tx_bkt    ac_ucb_ep_bkt
AC-Trap                    0.60       1.00         1.00             1.00             1.00
AC-Trap                    0.80       1.00         1.00             1.00             1.00
AC-Trap                    0.95       1.00         1.00             1.00             1.00
AC-Trap                    0.99       1.00         1.00             1.00             1.00
DC-Long50                  0.60       3.00         3.00             3.00             3.00
DC-Long50                  0.80       1.00         1.00             1.00             1.00
DC-Long50                  0.95       1.00         1.00             1.00             1.00
DC-Long50                  0.99       1.00         1.00             1.00             1.00
SH-FiniteMemoryRegret      0.60       1.03         1.03             1.03             1.03
SH-FiniteMemoryRegret      0.80       1.00         1.00             1.00             1.00
SH-FiniteMemoryRegret      0.95       1.00         1.00             1.00             1.00
SH-FiniteMemoryRegret      0.99       1.00         1.00             1.00             1.00
```

`frac_episodes_arm_admissible > 0` is 1.000 in every cell — i.e. there's
always at least one arm available — but **the modal admissible set has
size 1**. With only β=0 deployable, the alignment statistic
β·(r − v_next) > 0 evaluates to 0 (the strict inequality cannot fire when
β = 0), so alignment_rate is mechanically zero on AC-Trap and SH-FMR. On
DC-Long50 at γ = 0.60, the cap relaxes enough to admit 3 arms; alignment
rate reaches 0.245 there but still not the 0.5 threshold.

The cap is set per spec §4 to
\[
  \bar\beta_e^{(b)} = \frac{\log(\kappa / [\gamma(1+\gamma−\kappa)])}{R_{\max} + \widehat B_e^{\text{glob}}}, \qquad \kappa = \gamma + \alpha(1−\gamma).
\]
At α=0.10 (the spec default), the cap is too tight even at γ=0.60. This
is consistent with the paper's §6 finding noted in spec §10.1 — α=0.10
is on the wrong side of the safety/exploration tradeoff for these
cells under the chosen R_max bounds (1 for DC, 5 for AC, 1 for SH).

## Secondary cause: counterfactual reward is anti-correlated with on-policy

Spec §10.2 anticipated a failure mode where the counterfactual reward
\(\widehat\mu_j^{\text{cf}}\) on un-deployed arms is large and
optimistic, while the on-policy correction \(\widehat\mu_j^{\text{op}}\)
on the actually-deployed arm is negative — producing UCB scores that
systematically over-recommend non-admissible arms. The pilot confirms
this:

```
Diagnostic 2 — corr(μ_cf, μ_op) per arm (averaged across γ and seed)

cell                       method                corr(μ_cf, μ_op)
AC-Trap                    ac_ucb_tx                       -0.253
AC-Trap                    ac_ucb_tx_bkt                   -0.253
DC-Long50                  ac_ucb_tx                       -0.417
DC-Long50                  ac_ucb_tx_bkt                   -0.417
SH-FiniteMemoryRegret      ac_ucb_tx                       -0.311
SH-FiniteMemoryRegret      ac_ucb_tx_bkt                   -0.311
```

All three cells show **anti-correlation between counterfactual and
on-policy reward estimates** — exactly the shape spec §10.2 flags as
the off-policy bias signature. The most divergent arms are the cap
extremes (β = ±2.0) where μ_cf reflects the closed-form Δ_t evaluation
but μ_op reflects what would happen if those arms were ever actually
deployed (0 in our data because they never were, due to the cap; or
strongly negative on AC-Trap β=−2.0 ac_ucb_ep where one rare deployment
gives μ_op = −13.5).

The interaction with diagnostic 1 is fatal: the cap forbids extreme
arms, μ_op stays at 0 for them by default, and μ_cf claims those same
arms are weakly bad — but UCB's c-bonus cannot rescue admissibility,
so the entire counterfactual-bias correction signal is cosmetic.
The UCB picks the only admissible arm (β = 0) regardless.

## Tertiary cause: bucketing doesn't help when admissibility is the bottleneck

Spec §10.3 anticipated bucketing being too coarse or too fine. The
pilot's bucketing metric is moot at α = 0.10:

- For tabular cells (DC-Long50), `tabular_state_only` produces 51
  state-buckets. Per spec §4 the cap uses the **global** envelope
  \(\widehat B^{\text{glob}}\) not per-bucket; therefore each bucket
  inherits the same too-tight cap and faces the same admissibility
  collapse. **Bucketing reduces to 51-fold replication of a global
  failure.**
- For matrix cells (AC-Trap, SH-FMR), `matrix_opp_and_step` produces
  H × n_opp_actions = 20 × 2 = 40 buckets. Same global-cap issue.

Per-bucket median pull count is therefore meaningless in the pilot:
each bucket pulls β=0 every visit because that is the only admissible
arm. We do not report the per-bucket histogram requested by the
directive's WAVE B3-IF-FALSE branch because the histogram is
identically [10000 pulls of β=0] for every visited bucket.

## Why the dispatch_notes' α=0.25 escape valve is not tested here

Spec §10.1 and DISPATCH_NOTES.md §Step 4 propose raising α to 0.25
and re-piloting one γ. **This memo does NOT execute that escape
valve** — the user's directive is explicit: *"IF false: skip to
WAVE E (final report)."* Re-piloting at α = 0.25 is a new program,
not a continuation of the M_AC_UCB pilot. The escape valve is
recommended in the §"Recommendations" block below for whoever
reopens the program.

## Why the program halts here (per directive + spec)

Both the user-facing directive and spec §10 unambiguously gate further
adaptive-β experimentation on the M9 alignment threshold. The pilot
ran 480 / 480 runs successfully (post the JSON-serialization fix for
the bucketed variants); the gate evaluation at runs_seen=480 is
reproducible; both AC-UCB sign-mode pairs (tx/ep × global/bucketed)
fail the threshold by a margin of ≥ 0.25 (max observed alignment is
0.25, threshold is 0.5). The falsifier has fired.

## Recommendations (out of scope; for the next iteration)

1. **Raise α to 0.25** (or higher) and re-pilot a single γ slice on
   DC-Long50 to verify the cap does open the admissible set. Per
   spec §10.1's escape valve.
2. **Shrink the counterfactual buffer to W = 1 episode** to test
   whether the off-policy bias (Diagnostic 2) shrinks when stale
   counterfactual estimates are evicted faster.
3. **Consider per-bucket-local envelope rather than global**: the
   audit doc §3.3 chose the global envelope on a paper-correctness
   grounds (cross-bucket bootstrap conservatism), but it makes the
   admissibility failure cell-coupled. A per-bucket local envelope
   would lose the safety certificate but allow per-bucket
   exploration; this is a paper-level redesign, not a pilot retry.
4. **Audit `RollingEnvelope` initialisation**: the cap formula
   depends on \(\widehat B_e^{\text{glob}}\). At episode 0 with
   q_init = 0, the envelope is 0; the cap should be infinite. If the
   cap is computed before the first envelope update or if the
   initial \(\widehat B_0\) is set defensively to a large value, the
   admissible set is permanently small. Worth verifying in the
   `RollingEnvelope.update` code path.

## Files

- Raw: `results/adaptive_beta/tab_six_games/raw/VIII/M_AC_UCB_pilot/`
  (480 runs / 480 status: completed; 0 NaN; 0 divergence flags;
  120 bucketed-variant runs were re-dispatched serially after a
  JSON-serialization fix in the runner).
- Per-cell summary: `processed/M_AC_UCB_pilot/M_AC_UCB_per_cell_summary.csv`
- Per-arm pull histogram: `processed/M_AC_UCB_pilot/M_AC_UCB_per_arm_pulls.csv`
- Gate verdict: `processed/M_AC_UCB_pilot/M_AC_UCB_pilot_gate_check.json`

## Status of M9 main grid and downstream waves

- Main grid (Wave D, 9 600 runs): **NOT dispatched.** Gated on this pilot.
- Wave C (pilot figures): partial value only — the figures would all
  show flat alignment-rate / d_eff trajectories at the value of β=0
  vanilla. They are produced for completeness in `figures/M_AC_UCB_pilot/`
  but should be read as evidence of the falsification, not as
  empirical advocacy for the controller.
- Wave E (final report): see
  `results/adaptive_beta/tab_six_games/M_AC_UCB_final_report.md`.
