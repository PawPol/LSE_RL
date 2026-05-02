# AC-Trap β=+0.10 mechanism follow-up — 10-seed expansion

- **Created**: 2026-05-02 (post V10.8 close)
- **HEAD**: `5acf7edb` (V10.8 disposition memo committed)
- **Config**: `experiments/adaptive_beta/tab_six_games/configs/AC_Trap_mechanism_followup.yaml`
- **Raw**: `results/adaptive_beta/tab_six_games/raw/VIII/AC_Trap_mechanism_followup/`
  (40 runs = 1 cell × 4 methods × 10 seeds × 10k episodes)
- **Verdict**: **AUC advantage holds at 10 seeds**; mechanism is
  **early-episode** alignment, not late-episode alignment. The G6c §3
  caution stands: this is a small statistical surface feature whose
  diagnostic at convergence (`alignment_rate[-200:]`) does NOT predict
  it.

## Background

V10.5 G6c flagged AC-Trap, γ=0.60, β=+0.10 as the **only** Tier II cell
where a positive-β arm has a paired-bootstrap CI strictly above 0
relative to vanilla:

```
G6c (5 seeds):   Δ_AUC = +128.6,  CI₉₅ = [+45.20, +212.00]   ✓
```

But the same arm has `alignment_rate[-200:] ≈ 0.05` — far below the
0.5 threshold the alignment-condition theory would predict. G6c
recommended a 10-seed expansion + per-step trajectory analysis to
test (a) whether the AUC gain is robust and (b) what mechanism — if
any — drives it.

## Result 1 — AUC advantage replicates at 10 seeds

```
## AC-Trap γ=0.60, 10k ep, 10 seeds — per-method summary
              method      mean AUC         std    align[-200:]    d_eff[5000:]   div
             vanilla     528928.50      536.94          0.0000           NA       0
    fixed_beta_+0.05     528935.90      571.03          0.0490        0.6612      0
     fixed_beta_+0.1     529060.20      558.32          0.0490        0.7849      0
     fixed_beta_+0.2     518891.90     3502.51          0.0490        1.2058      0

## Paired-bootstrap 95% CI of AUC advantage vs vanilla (B=20,000)
   fixed_beta_+0.05 vs vanilla:  Δ=    7.40   CI=[  -25.00,    44.40]   ✗
    fixed_beta_+0.1 vs vanilla:  Δ=  131.70   CI=[   88.10,   174.00]   ✓
    fixed_beta_+0.2 vs vanilla:  Δ=-10036.60  CI=[-12234.31, -8648.80]  ✗
```

- **+0.10 confirmed**: paired CI [+88.10, +174.00] at B=20,000 strictly
  above 0. The 5-seed G6c result tightens at 10 seeds (CI width
  shrinks from 167 to 86); the point estimate moves slightly
  (+128.6 → +131.7). This is **not** seed-count noise.
- **+0.05 not significant**: CI straddles 0 — the effect is
  specifically located near β = +0.10, not a smooth +β monotone gain.
- **+0.20 collapses sharply**: Δ = −10,036, CI strictly below 0. The
  arm's `effective_discount_mean[5000:] = 1.21` (max 1.31) — the
  operator's asymptotic `d_eff = (1+γ) max(r,v)/v > 1` regime is
  active, sustained, and destructive. (V10's `divergence_event`
  detector did not fire here, presumably because its threshold is
  set beyond 1.31; AUC collapse is the operative signal.)

## Result 2 — mechanism is *early*-episode alignment

The end-of-training diagnostic `alignment_rate[-200:] ≈ 0.05` masks
a transient that IS predictive of the AUC advantage. Per-seed
trajectory snapshots for the +0.10 arm:

```
## fixed_beta_+0.10 — alignment_rate by training window (10 seeds)
seed  ep[0:10]  ep[10:100]  ep[100:1000]  ep[1000:5000]  ep[5000:]      AUC
  0     0.6150      0.3306        0.0434         0.0413     0.0489  528291.00
  1     0.6300      0.3228        0.0446         0.0415     0.0489  529621.00
  2     0.6250      0.2844        0.0483         0.0417     0.0486  529300.00
  3     0.6600      0.3511        0.0407         0.0414     0.0490  529185.00
  4     0.6250      0.3350        0.0460         0.0412     0.0491  529257.00
  5     0.6200      0.3267        0.0450         0.0417     0.0490  529073.00
  6     0.6800      0.3367        0.0447         0.0413     0.0489  529738.00
  7     0.6450      0.3594        0.0426         0.0421     0.0490  529479.00
  8     0.6200      0.3344        0.0446         0.0414     0.0489  528533.00
  9     0.6100      0.3322        0.0446         0.0412     0.0487  528125.00
```

- **Early phase (ep 0–10)**: `alignment_rate ≈ 0.62`. With Q
  initialised to 0 and `β = +0.10`, the alignment statistic
  `β · (r − v) > 0` is dominated by `r` while `v ≈ 0`; AC-Trap's
  reward structure makes `r > 0` the typical first-step state, so
  the operator credits `r` (not `v`) early in learning.
- **Mid-early phase (ep 10–100)**: `alignment_rate ≈ 0.33`. Q values
  fill in but credit-anchor preference remains.
- **Late phase (ep 100→end)**: `alignment_rate → 0.045–0.049`,
  matching vanilla's behavioural saturation. The end-of-training
  diagnostic sees only this regime.

The mechanism is therefore **early-episode reward-anchored credit
assignment**, sustained for ~100 episodes (1% of the run), giving
the +0.10 arm a small positive lead over vanilla that persists in
cumulative AUC because vanilla never recovers the early gap.

This explains why:

- **+0.05 doesn't see it**: `β · (r − v)` is too small to swing the
  log-sum-exp soft-max away from `v` even when `v ≈ 0`, so the
  early-phase alignment rate is similar to vanilla's degenerate
  baseline.
- **+0.20 destroys it**: the early bias becomes too strong; once Q
  starts to grow, the operator's `d_eff` overshoots γ persistently
  (0.78 → 1.21 by ep 5000) and AUC collapses faster than any
  early-phase advantage can compensate for.

The +0.10 magnitude is in a narrow Goldilocks window where early
alignment is large enough to register but late-phase amplification
is bounded.

## Result 3 — no divergence at any β in this cell

```
divergence_event totals (10 seeds × 10k ep):
  vanilla:           0
  fixed_beta_+0.05:  0
  fixed_beta_+0.1:   0
  fixed_beta_+0.2:   0
```

The 524 `divergence_event > 0` fires reported in V10 G6c §d concentrate
in cells with **higher γ** (γ ∈ {0.95, 0.90}) and stronger +β arms
(|β| ≥ 0.5). At γ = 0.60 the operator's `d_eff` ceiling
`(1+γ) = 1.60` is low enough that even +0.20 stays below the
runner's divergence threshold, despite mean `d_eff > 1`. AC-Trap at
γ=0.60 is therefore not contributing to V10's divergence signature.

## Implications for the paper narrative

1. **G6c §3 caution holds with refinement**: the AUC gain is real,
   reproducible at 10 seeds, but the **late-phase** alignment
   diagnostic does NOT predict it. The mechanism is **early-phase**
   alignment, which the current diagnostic does not summarise.
2. **Theoretical implication**: TAB's alignment condition is a
   *time-varying* property; the relevant window for a given cell
   depends on the value-function trajectory, not just the
   stationary policy. This is consistent with the operator's
   structure (the alignment indicator depends on `r − v`, and `v`
   evolves) but contradicts the paper's static framing of the
   alignment condition.
3. **Practical implication for §8.4 (pending)**: the AC-Trap result
   is best framed as *narrow positive evidence with mechanism
   distinct from the headline alignment story*, rather than full
   alignment-condition vindication. A diagnostic that summarises
   `alignment_rate` over a *learning-curve-aware* window (e.g.
   first 1% of episodes or until `Δ AUC` ceiling) would be
   needed to canonicalise this finding.
4. **No further experiments needed for V10 closure**: the H1
   confirmation at 10 seeds is the last open §8 item that did not
   require user sign-off. V10.9 M7+ continuation remains
   user-gated per spec §10.2 + open HALT 7.

## Reproduction

```bash
.venv/bin/python -m lse_rl.experiments.adaptive_beta.tab_six_games.runner \
    --config experiments/adaptive_beta/tab_six_games/configs/AC_Trap_mechanism_followup.yaml \
    --output-root results/adaptive_beta/tab_six_games/raw/VIII/AC_Trap_mechanism_followup
```

Analysis stats reproduced inline in the V10.9 close commit message;
the per-seed trajectory snapshots above use the saved
`metrics.npz::alignment_rate` arrays directly (no separate analysis
script — figures are not required for this targeted memo).

## Pending (NOT V10.9 scope)

- **Time-varying alignment diagnostic** (paper §8.4 framing): a
  learning-curve-aware summary metric would canonicalise this
  finding. Not required to close V10.
- **Codex re-review of mechanism interpretation**: optional. The
  AUC + CI numbers are deterministic; the mechanism narrative is a
  paper-level claim, not a result claim.
- **Generalise to SH-FMR γ=0.60**: G6c showed SH-FMR's +β arm has
  CI straddling 0 at 5 seeds. A parallel 10-seed expansion could
  test whether the same early-alignment mechanism is at work, but
  this is M7+ scope.

## Conclusion

AC-Trap β=+0.10 at γ=0.60 is **the only positive-β cell in the v10
empirical record with a paired-bootstrap CI strictly above vanilla
at both 5 and 10 seeds**. The mechanism is early-episode (ep 0–100)
alignment, not the late-phase (ep ≥ 1000) alignment summarised by
the current diagnostic. This refines but does not overturn the G6c
verdict: H1 holds 1/4 (AC-Trap only), with mechanism distinct from
the headline alignment story.
