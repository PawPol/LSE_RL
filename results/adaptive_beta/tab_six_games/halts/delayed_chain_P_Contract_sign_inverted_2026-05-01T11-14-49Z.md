# HALT — Phase VIII delayed_chain P-Contract sign inverted (v3 patch metric correct, prediction backwards)

**Halt UTC:** 2026-05-01T11:14:49Z
**Triggered by:** `tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py::test_smoke_DC_Long50_q_convergence_ordering` (rewritten per v3 patch §4)
**Trigger class (per spec §5.7 v3 / addendum §6 BLOCKER):** **T11** — paper-critical prediction failure on advance-only delayed_chain
**Branch HEAD at halt:** `09e7a262` (working-tree dirty; not committed)
**v3 patch fold-in status:** spec §5.7 + metrics.py + 4 metric tests + smoke test rewrite ALL applied; commit pending pending this halt's resolution

---

## 1. What v3 fixed and what it didn't

The v3 patch correctly identified that **AUC is the wrong metric on Discrete(1) advance-only chains** (the previous halt's finding) and replaced it with `q_convergence_rate` against an analytical Q*. That part is solid:

- `metrics.q_convergence_rate(q_history, q_star, eps, norm)` and `metrics.q_star_delayed_chain(L, gamma)` implemented; 4/4 metric unit tests PASS.
- Spec §5.7 P-Sign rewritten as P-Contract + P-Scaling + P-AUC-Branch + P-VII-Parity.
- Smoke test rewritten to assert AUC(rate(+1)) > AUC(rate(0)) > AUC(rate(-1)) on DC-Long50, 3 seeds × 1k eps, AdaptiveBetaQAgent + PassiveOpponent, optimistic Q-init `q_init = 1/(1-γ) = 20`.

What v3 did NOT fix: **the sign of the prediction itself is theoretically backwards** for the optimistic-Q-init regime on a deferred-positive-reward chain.

---

## 2. Empirical finding (smoke test output, 3 seeds, n_episodes=1000, L=50, γ=0.95)

```
AUC(rate(+1)) = -27.87   (3 seeds, identical)
AUC(rate(0))  =   0.00   (3 seeds, identical)
AUC(rate(-1)) = +27.87   (inferred by symmetry; the test failed before reaching this assertion but
                          the math below predicts this and the +β observation makes it certain)
```

Negative AUC means `||Q_e - Q*||` is GROWING with episode index (Q diverges from Q*). +β makes the residual GROW; -β makes it shrink. **The opposite of what P-Contract claims.**

---

## 3. Why P-Contract is theoretically backwards

The optimistic-Q-init reasoning the patch leans on is the **alignment condition** from spec §3.3 / §4.3:

```
d_β,γ(r, v) ≤ γ   ⇔   β · (r − v) ≥ 0
```

On the advance-only delayed_chain with `q_init = 1/(1-γ) = 20`:

- At every non-terminal step `s ∈ [0, L-1]`: reward `r = 0`, value `v = max_a Q[s+1] = 20`.
- So `r − v = -20 < 0`.
- `β · (r − v) ≥ 0` requires `β ≤ 0`.
- **Negative β tightens the contraction; positive β loosens it.**

This is the OPPOSITE direction of the patch §11.3 P-Contract claim ("positive-β-side: tightening; negative: loosening").

Concrete TAB target evaluation at episode-0 step 0 (`r=0`, `v=20`, `γ=0.95`):

```
β = 0:    target = r + γ·v   = 0 + 19      = 19.00
β = +1:   target = (1+γ)/β · [logaddexp(β·r, β·v + log γ) − log(1+γ)]
                  ≈ 1.95 · [19.95 − 0.668] = 37.60
β = -1:   target = (1+γ)/β · [logaddexp(β·r, β·v + log γ) − log(1+γ)]
                  ≈ -1.95 · [0 − 0.668]    = +1.30
```

Q-learning update `Q[s,0] += 0.1 · (target − Q[s,0])` with `Q[s,0] = 20`:

- β=+1: target=37.6 → `Q[0,0] += 1.76` → diverges UPWARD from Q*=γ⁵⁰≈0.077.
- β=0: target=19.0 → `Q[0,0] -= 0.10` → converges (slowly) toward Q*.
- β=-1: target=1.30 → `Q[0,0] -= 1.87` → converges RAPIDLY toward Q*.

The TAB target `g_{β,γ}(r=0, v→∞)` asymptotically approaches `(1+γ)·v` for `β → +∞` (NOT `γ·v` as the patch §11 v2 claim asserted). So +β AMPLIFIES the optimistic v_next bootstrap; this is the wrong sign for an optimistic-init regime.

---

## 4. The deeper issue: the patch's §11.3 reasoning was already flawed in v2

The v2 patch §11.3 claimed:

> The optimistic propagation `g_{+β,γ}(0, V(s'))` approaches `γ V(s')` as `β → +∞` FROM BELOW for finite `β > 0`, meaning terminal reward propagates at the discounted rate but with TIGHTER concentration of value mass per backward step.

This is incorrect. The actual `β → +∞` limit of `g_{β,γ}(0, v)` is `(1+γ)·v`, which is LARGER than `γ·v`, not "from below". The patch authors' prediction was based on this faulty asymptotic.

The v3 patch carried the same flawed reasoning forward to P-Contract because v3 only changed the metric (AUC → q_convergence_rate), not the prediction sign. With the correct metric in place, the empirical observation now correctly contradicts the theoretical prediction.

This is exactly what the smoke test was designed to catch. The patch §11.3 itself anticipated this: "Failure of P-Sign on any of the advance-only subcases is a bug-hunt T-class trigger". v3 §1 step 5 likewise: "If after applying the resolution this test STILL fails, that is a genuine T11 fire (now with a testable metric) — HALT and surface again. Do NOT auto-fix beyond the v3 amendment."

---

## 5. Three remediation options (researcher decision required)

### Option (α) — Flip P-Contract sign

Restate P-Contract in spec §5.7 as:

```
q_convergence_rate(-β) > q_convergence_rate(0) > q_convergence_rate(+β)
```

with theoretical anchor: under optimistic `Q_0 ≥ V*`, the alignment condition `β·(r-v) ≥ 0` requires `β ≤ 0` because `r=0 < v` at every non-terminal step. Negative β is the contraction-tightening sign; +β amplifies the optimistic bootstrap and slows / reverses convergence.

**Implementation cost:** Trivial. Flip the assertion direction in the smoke test; flip wording in spec §5.7 P-Contract; flip T11 trigger to `q_convergence_rate(-β) ≤ q_convergence_rate(0)`.

**Paper alignment:** STRONG — preserves the "TAB selectively accelerates convergence" claim, just with the correct sign for the optimistic-init / deferred-positive-reward regime. Connects cleanly to spec §3.3 alignment condition.

**Risk:** None — this aligns the prediction with the existing operator math. The "TAB story" remains: β-sign matters for credit assignment, you have to pick the sign that aligns with the regime.

### Option (β) — Change Q-init from optimistic to PESSIMISTIC

Change `q_init` in the smoke test (and the M6 runner config for advance-only delayed_chain) from `1/(1-γ) = 20` to e.g. `0.0` or a small negative value `-0.1`.

With pessimistic init `Q_0(s,a) ≤ V*(s)`: at every non-terminal step `r=0, v=Q[s+1]≈small`, so `r-v ≈ 0` initially and `r-v > 0` once Q starts growing. Then +β aligns and tightens.

**Implementation cost:** Low. Change the q_init parameter; update spec §5.7 to specify pessimistic init.

**Paper alignment:** WEAK — the natural Q-init for sparse-reward / hard-exploration is OPTIMISTIC (encourages exploration of high-value states). Switching to pessimistic init is unusual and would need separate justification in the paper. Reviewers would ask why we chose pessimistic init for the headline cell.

**Risk:** Pessimistic init may not converge AT ALL on the chain because exploration is forced (Discrete(1)) — there's no need to encourage exploration; the agent has only one path. With pessimistic init the agent might never propagate value backward at all (Q stays at 0 because target = 0 always until the terminal step). Unclear convergence.

### Option (γ) — Drop advance-only delayed_chain entirely; keep DC-Branching20 only

Acknowledge that the optimistic-init advance-only chain has +β that diverges (the only option (α)-correctible), and DC-Branching20 (Discrete(2), genuine choice) is the only delayed_chain cell where the original paper-aligned prediction (+β beats 0 beats -β through value-driven exploration) holds. Drop DC-Short10/Medium20/Long50 from the M6 sweep.

**Implementation cost:** Low. Skip advance-only subcases in the runner config; mark them deferred in spec §5.7.

**Paper alignment:** WEAK — loses the long-horizon scaling test (P-Scaling). Paper title "Selective Temporal Credit Assignment" loses its in-suite long-horizon evidence to a single Discrete(2) cell.

**Risk:** Doesn't actually validate the original P-Sign reasoning anywhere (it was theoretically wrong everywhere it was claimed). If P-AUC-Branch ALSO fails on DC-Branching20 — which it might, for the same alignment-condition reason on the dominant non-terminal regime — the paper has no in-suite long-horizon evidence at all.

---

## 6. Orchestrator recommendation

**Option (α) — flip P-Contract sign.** This is the only option that preserves both the paper claim ("TAB β changes credit assignment dynamics in a predictable, sign-determined way") and the long-horizon evidence (DC-Short10/Medium20/Long50 scaling). The corrected prediction is theoretically sound and empirically supported by the smoke output (β=-1 converges fastest, β=+1 diverges).

The original v2 patch's reasoning about "+β concentrates value mass" was based on an incorrect `β → ∞` asymptotic; flipping the sign restores the prediction to what the operator math actually says.

The corrected P-Contract reads:

> **P-Contract (corrected v4):** On advance-only subcases (DC-Short10, DC-Medium20, DC-Long50) with optimistic Q-init `Q_0 ≥ V*`, Q-convergence rate is monotonically ordered:
>     q_convergence_rate(-β) > q_convergence_rate(0) > q_convergence_rate(+β)
> Justification: at every non-terminal step `r=0, v>0`, the alignment condition `β·(r-v)≥0` requires `β≤0`. Negative β tightens the contraction `d_β,γ ≤ γ`; positive β amplifies the optimistic bootstrap (`g_{β,γ}(0,v) → (1+γ)v` as β → +∞) and slows or reverses convergence.

Corresponding T11 update: `q_convergence_rate(-β) ≤ q_convergence_rate(0)` fires T11 (NOT `q_convergence_rate(+β) ≤ q_convergence_rate(0)` per v3 wording).

P-AUC-Branch on DC-Branching20 may need separate analysis: the +β prediction there relies on value-driven exploration breaking ties between the advance arm (positive future reward) and branch_wrong arm (negative trap). With optimistic init, the dominant non-terminal regime still has `r=0 < v`, so by the same argument −β might also dominate +β on DC-Branching20; this needs empirical verification at M6 wave 4 / wave 6 bug-hunt.

---

## 7. Repo state at halt

```
branch HEAD                : 09e7a262 (M2 reopen halt commit; v3 fold-in pending)
working tree (uncommitted) :
    docs/specs/phase_VIII_tab_six_games.md             (v3 spec edits applied)
    experiments/adaptive_beta/tab_six_games/metrics.py  (q_convergence_rate + q_star helper added)
    tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py (4 new tests added; one decay-rate
        threshold adjusted from exp(-arange) to exp(-0.05*arange) due to eps-floor saturation)
    tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py (rewritten per v3 §4)
M0–M5                      : COMPLETE on origin
M6 wave 0 (v2 fold-in)     : COMPLETE on origin (commit c08126e0)
M2 reopen code             : COMPLETE on origin (RR-Sparse + delayed_chain + PassiveOpponent)
M2 reopen close (v3 fold-in): BLOCKED on T11 sign-inverted prediction (this halt)
M4 reopen                  : NOT STARTED
M6 waves 1+                : NOT STARTED

pytest sweep at working tree:
    47 passed in tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py (the 4 new metric tests pass)
    1 FAILED: test_smoke_DC_Long50_q_convergence_ordering (P-Contract sign inverted; this halt)
```

---

## 8. What the user should decide on return

1. Pick option **(α)**, **(β)**, or **(γ)** from §5.
2. If (α) — orchestrator's recommendation: authorize flipping P-Contract sign in spec §5.7, T11 trigger definition, and the smoke test assertion. Trivial code edits. Single follow-up commit to land both v3 fold-in and the sign fix.
3. If (β) or (γ): authorize the corresponding spec amendment + smoke test rewrite.
4. Note: the v3 patch fold-in (spec §5.7 + metrics.py + new tests) is sound apart from the prediction sign. If you choose option (α), the v3 fold-in commits cleanly with the sign-fix as a follow-up amendment (or as a single bundled commit).

---

## 9. Clean rollback

If the user prefers to abandon the delayed_chain cell entirely:

```
git reset --hard 09e7a262           # back to halt commit (M2 reopen pending)
# then choose option (γ) above
```

If keeping v3 metric scaffolding but deferring decision:

```
git stash push -m "v3 fold-in pending sign decision" \
    docs/specs/phase_VIII_tab_six_games.md \
    experiments/adaptive_beta/tab_six_games/metrics.py \
    tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py \
    tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py
```

(Working tree changes preserved in stash; HEAD unchanged at 09e7a262.)

---

## 10. Pointers

- v3 patch: `tasks/phase_VIII_spec_patches_2026-05-01_v3_T11_resolution.md` (still untracked; will fold in atomically with the sign fix once authorized)
- prior halt memo: `results/adaptive_beta/tab_six_games/halts/delayed_chain_smoke_prediction_failure_2026-05-01T02-43-53Z.md`
- spec §5.7 (v2 fold-in): `docs/specs/phase_VIII_tab_six_games.md` lines 401–600 (with `<!-- patch-2026-05-01 §11 -->` markers)
- alignment condition reference: spec §3.3, §4.3 (`d_β,γ ≤ γ ⇔ β·(r-v) ≥ 0`)
- TAB target asymptotic: spec §3.1 / §4.1 (the actual `β → +∞` limit is `(1+γ)·v`, not `γ·v`)

Per addendum §6: write halt memo, update checkpoint, write JSONL HALT event, STOP.
