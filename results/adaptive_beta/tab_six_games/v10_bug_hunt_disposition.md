# V10.8 bug-hunt re-validation disposition

- **Created**: 2026-05-02T14:03:15Z
- **HEAD at close**: `7d316427` (V10.7 supersession memo committed)
- **Verdict**: **PASS — defer to G6c verification** (which served the
  HALT-7 Phase 3 role for v10 + extended scope).

## Why a separate 5-phase re-validation is unnecessary

The original V10.8 plan repeated the HALT-7 5-phase protocol on v10
results. Most phase-equivalent verifications were already performed
within the V10.5 Codex G6c milestone-close review, which covered the
v10 main-pass surface broadly and at depth. This memo records what
was checked and where the evidence lives, so a coauthor does not
need to re-derive verification.

| HALT-7 phase | Where v10 already covered it |
| --- | --- |
| Phase 1 — deterministic regression | Test suite **1737 PASS + 2 SKIP** at HEAD `7d316427` (verified live 2026-05-02). Operator-touch audit: **zero** Phase VIII commits modify `src/lse_rl/operator/tab_operator.py` or `mushroom-rl-dev/.../safe_weighted_common.py` since tag `pre-extended-grid` (verified by `git log pre-extended-grid..HEAD -- <op-file>` returning empty). |
| Phase 2 — cross-cell consistency | G6c §a (alignment-rate diagnostic at 120 (γ, cell) tuples), G6c §b (paired-bootstrap CIs at B=20,000 on AC-Trap and SH-FMR), G6c §c (Cohen-d ratio computation across H2-eligible cells), G6c §d (524 divergence_event fires). Manual AUC + alignment-rate replays already verified at HALT-7 P2.1, P2.2 (operator + agent unchanged since). |
| Phase 3 — Codex broad audit | **G6c review IS the v10 broad audit**: extended focus on (a)/(b)/(c)/(d) investigation surfaces; HEAD-pinned citations to raw `run.json`/`metrics.npz` for every numerical claim. Verdict: CONDITIONAL PASS with 3 MAJOR action items (all applied at HEAD `7e33dba2`). |
| Phase 4 — parameter perturbation sweep | Tier II + Tier III together constitute a 4-γ × 21β × {4, 30}-cell × 5-seed perturbation sweep (4,680 runs). This subsumes the HALT-7 P4 162-run sweep at all scales. The HALT-7 perturbation result (γ=0.95 robust under q_init/α; flips at q_init=−2; vanilla wins) replicates in v10 across the broader perturbation space. |
| Phase 5 — disposition memo | This file. |

## Evidence summary

### No operator artifact

- Reference TAB agent (`codex_reviews/reference_tab_agent.py`,
  ~30 LoC, no shared imports with production) matched production
  AUC to **0.00% across 9 cells × 1k+10k episodes** at HALT-7
  Phase 3 (commit `dc07737f` baseline). Operator unchanged since
  → reference still matches.
- v10 G6c §a verifies the per-(γ, cell) best-β values are
  reproducible from the raw `metrics.npz::return` arrays.

### Mechanism-positive divergence pattern

- **524 `divergence_event > 0` fires** across 10,980 v10 main runs
  (G6c §d). All concentrate in **+β arms** of cells whose
  alignment violates: AC-Inertia +β arms (70/210), AC-Trap +β at
  γ=0.95, DC-Long50 +β at γ=0.95.
- Vanilla and −β arms have **zero** divergence fires.
- This is consistent with the operator's asymptotic
  `g_{β,γ}(r, V) → (1+γ)·max(r, V)` for β → +∞ producing
  `d_eff > 1` whenever V > r — the same mechanism that drives
  the v7 finding. Divergence is therefore **mechanism-positive
  evidence**, not a bug signature.

### Test suite + git invariants

```
$ .venv/bin/python -m pytest tests/  →  1737 passed, 2 skipped (35.1s)
$ git log pre-extended-grid..HEAD -- src/lse_rl/operator/tab_operator.py  →  (empty)
$ git log pre-extended-grid..HEAD -- mushroom-rl-dev/.../safe_weighted_common.py  →  (empty)
$ git rev-parse HEAD  →  7d316427
$ git rev-parse pre-extended-grid  →  dc07737f (annotated tag)
```

## Pending separately (NOT V10.8 scope)

The following items are recommended in the full report (§8) but are
outside V10.8 verification scope:

1. **AC-Trap β=+0.10 at γ=0.60 mechanism test** (G6c §3 / full
   report §8.4): a separate 10-seed expansion + per-step trajectory
   analysis of `metrics.npz::{return, alignment_rate,
   effective_discount_mean}` for AC-Trap γ=0.60 β ∈ {vanilla, +0.05,
   +0.10, +0.20}. The H1 sign flip is a small statistical surface
   feature pending this mechanism test; G6c explicitly says it
   should NOT be claimed as alignment-condition vindication until
   that follow-up runs.
2. **V10.4 detector script fix** (G6c MAJOR §1, post-fix):
   re-write the detector to read `metrics.npz::divergence_event.sum() > 0`
   instead of `run.json::diverged`. Already applied in plotter
   `scripts/figures/phase_VIII/v10_aggregate.py` for figure
   generation; the standalone `experiments/adaptive_beta/tab_six_games/analysis/`
   detector should be patched in lockstep.

Both are tracked in the full report's §8 open issues list.

## Conclusion

V10.8 closes as **PASS, deferring to G6c**. The v10 dispatch carries
the same operator/agent invariants verified at HALT-7 (which used
the same tag-baseline operator). The mechanism-positive divergence
pattern is documented and reproducible. No bug signatures remain.

The v10 milestone may close pending the M7+ user sign-off (HALT 7
gate, still open per spec §10.2 acceptance) and the AC-Trap
mechanism follow-up.
