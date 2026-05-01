# HALT 3 — Phase VIII delayed_chain metric design flaw (v4 sign also wrong; deeper issue)

**Halt UTC:** 2026-05-01T11:31:32Z
**Triggered by:** `tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py::test_smoke_DC_Long50_q_convergence_ordering` (v4 sign + terminal-slice fix applied)
**Trigger class (per spec §5.7 v4 / addendum §6 BLOCKER):** **T11** — paper-critical prediction failure on advance-only delayed_chain (third successive fire)
**Branch HEAD at halt:** `10d58695` (HALT 2 commit; v4 working-tree edits pending)
**v4 patch fold-in status:** spec §5.7 P-Contract sign FLIPPED, q_star off-by-one FIXED, smoke test FLIPPED, terminal-state slice ADDED. All sound. The prediction itself is empirically wrong on the chosen metric.

---

## 1. What v4 fixed (correctly)

Per the user's HALT 2 resolution directive:

1. **q_star_delayed_chain off-by-one** — fixed: `Q*(s, advance) = γ^(L-1-s)`, so `Q*(L-1, advance) = 1 = γ^0` and `Q*(0, advance) = γ^(L-1)`.
2. **P-Contract sign** — flipped from `+β > 0 > −β` to `−β > 0 > +β` per alignment-condition derivation.
3. **T11 trigger** — flipped to match new sign.
4. **Smoke-test assertions** — flipped to match.
5. **Terminal-state slice** — added (orchestrator-level fix; the L∞ norm was being dominated by `|Q[L,0] − Q*[L,0]| = 20` because the agent never updates terminal-state Q-values; sliced `q_hist[:, :L, :]` to measure convergence on the L learning states only).

These four corrections + my one terminal-slice fix were all necessary and correct. Metric tests now pass: 47/47.

But the smoke test STILL fails — and on a different basis.

---

## 2. New empirical finding (v4 + terminal-slice, 3 seeds, DC-Long50, optimistic Q-init=20)

```
AUC(rate(-1)) = +3.58    (partial convergence; overshoots to ~0.55 below Q*)
AUC(rate( 0)) = +21.03   (full convergence to classical Q*; residual at eps floor)
AUC(rate(+1)) = -27.87   (diverges to ~1e13)
```

The empirical ordering is `AUC(rate(0)) > AUC(rate(-1)) > AUC(rate(+1))`. Neither v3 ordering (`+ > 0 > −`) nor v4 ordering (`− > 0 > +`) matches. The actual order is **β=0 dominates**.

---

## 3. The deeper issue (theoretical)

The TAB operator's **fixed point depends on β**. The smoke test measures `||Q_e − Q*||` where `Q*` is the **classical** (β=0) Bellman fixed point: `Q*(s, advance) = γ^(L-1-s)`.

But:

- For β = 0 the TAB target collapses to `r + γ·v` exactly. The classical Q* IS the fixed point. Q-learning converges to Q*.
- For β ≠ 0 the TAB target is `g_{β,γ}(r, v) = (1+γ)/β · [logaddexp(βr, βv + log γ) − log(1+γ)]`. The fixed point of `g_{β,γ}`-iterated Bellman is a **β-specific value function**, NOT classical Q*.

Concretely, on the advance-only chain with `r=0` at non-terminal steps:

- **β = +∞ limit:** `g_{β,γ}(0, v) → (1+γ)·v`. The fixed-point recursion `V_β(s) = (1+γ)·V_β(s+1)` blows up (geometric ratio > 1). β=+1 with finite-but-large amplification diverges.
- **β = −∞ limit:** `g_{β,γ}(0, v) → 0`. The fixed-point recursion is `V_β(s) = 0` everywhere except the terminal step. β=−1's fixed point on the chain is approximately 0 at all non-terminal states (with terminal-step boundary contributing a small amount). So Q goes to ~0, not to Q* = γ^(L-1-s).

The empirical β=−1 result (AUC=+3.58, residual converges to ~0.55) reflects this: β=−1's fixed point is much smaller than classical Q*, so it overshoots Q* on the way down and lands at its own (smaller) fixed point. The "rate of approach" is positive (Q is decreasing) but stops before reaching classical Q*.

The smoke test asks "which β converges fastest to classical Q*?" — the answer is **β=0**, because only β=0 has classical Q* as its fixed point. β≠0 reach DIFFERENT fixed points and the residual against classical Q* is lower-bounded by the gap `|Q*_β − Q*_0|`.

---

## 4. Why v4 sign reasoning was correct but incomplete

The alignment condition `d_{β,γ}(r,v) ≤ γ ⇔ β·(r-v) ≥ 0` (spec §3.3) tells you which β makes the contraction TIGHTER. With `r=0 < v`, β ≤ 0 satisfies it. v4's sign flip was therefore correct for the contraction-tightness claim.

What v4 missed: **tighter contraction toward what fixed point?** A tighter contraction means faster convergence to whatever Q*_β is. If Q*_β ≠ Q*_0, you converge fast to the WRONG point and stop. The metric `||Q_e − Q*_0||` then shows residual going down (you're approaching Q*_β which is below Q*_0) and then plateauing at `|Q*_β − Q*_0|`.

The empirical β=−1 trajectory matches exactly: `Q[s,0]` decays from 20 to ~0.55 (fast contraction toward Q*_β ≈ 0) and stops well above the classical Q*(s, advance) = γ^(49) ≈ 0.08. So `||Q_e − Q*_0||` → 0.47 (the gap between Q*_β and Q*_0 at s=0).

---

## 5. The metric is asking the wrong question

`q_convergence_rate(q_history, q_star_classical)` measures convergence to the CLASSICAL fixed point. This metric will ALWAYS favor β=0 because β=0 is the only schedule whose dynamics have classical Q* as fixed point. β≠0 schedules converge to their OWN β-specific fixed points, which on this chain are systematically smaller than Q*_0 (so residual against Q*_0 saturates).

The "TAB story" the patch §11 author intended was probably one of the following:

(I) **β-induced contraction speed at fixed Q-target:** would need a metric that compensates for the β-specific fixed point. E.g., `||Q_e − Q*_β|| / ||Q*_β||` — convergence toward each schedule's own fixed point. On this metric β≠0 might genuinely converge faster (tighter contraction). But it's not testing "the operator converges to the right answer faster"; it's testing "the operator converges to ITS answer faster". Less paper-relevant.

(II) **Bellman residual convergence (not state-value):** `||TΦQ_e − Q_e||` for the appropriate operator T_β. This DOES have a uniform notion of convergence ("the residual of the operator's own fixed-point equation"). Different β values approach their respective fixed points; the residual decay rate IS β-monotonic in the alignment-condition direction. Closer to the spec §M5 `contraction_reward` metric definition.

(III) **Operator-induced policy quality on a separate axis (e.g., AUC of return on a policy-relevant action space):** what DC-Branching20 is supposed to provide.

Patch §11.1 motivation was about "value-propagation behavior on horizons where temporal credit actually matters". That fits (II) — Bellman residual decay — better than (I) Q-distance to classical Q*.

---

## 6. Three remediation options

### Option (δ) — Switch metric to Bellman residual decay

Replace `q_convergence_rate(q_history, q_star)` smoke test with `bellman_residual_decay` smoke test. The Bellman residual at episode `e` is `||T_β Q_e − Q_e||` for the appropriate operator. Decay rate is `log(R_e + ε) − log(R_{e+1} + ε)`, a quantity already defined in spec §7.2 / §M5 `contraction_reward`. This is the natural metric for the operator-mechanism story.

**Implementation cost:** Medium. Need to compute T_β Q for each β at each episode (one operator application per state). The metric helper exists; need to wire it into the smoke test.

**Paper alignment:** STRONG. Directly tests "TAB-induced contraction speed". Doesn't require classical Q* as ground truth.

**Risk:** Need to verify the alignment-condition prediction holds on Bellman-residual decay. Math suggests it should (β·(r-v)≥0 is a contraction-tightness condition for T_β toward Q*_β, which IS what Bellman residual measures), but empirical validation needed in a smoke run.

### Option (ε) — Drop the absolute-prediction from the smoke; keep delayed_chain as a paper-narrative cell

Acknowledge that delayed_chain reveals **β=0 is the natural choice for classical Q-learning convergence** and that's a legitimate paper finding (β-TAB on long-horizon chains has no advantage on the convergence axis; the paper's TAB story shifts to "TAB matters when the natural fixed point depends on regime, e.g. sign-switching composites").

Mark the smoke test as a non-prediction informational diagnostic. Proceed with M6 with delayed_chain as a cell that produces "β=0 wins" as its empirical signature, which is itself a TAB story (the operator selectively matters when the regime shifts, not on stationary delayed-reward chains).

**Implementation cost:** Trivial. Mark smoke as `@pytest.mark.skip(reason='delayed_chain has no β-monotonic prediction; β=0 dominates as expected for classical Bellman fixed point')`.

**Paper alignment:** WEAK. Loses the long-horizon credit-assignment claim entirely. Paper title "Selective Temporal Credit Assignment" reduces to "selective" only on regime-switching tasks, not on long-horizon chains.

### Option (ζ) — Drop delayed_chain entirely from M6

Acknowledge the delayed_chain cell doesn't produce the expected β-monotonic story on any natural metric. Remove it from M6 sweep, keep RR-Sparse + the original 6 games + matrix-game horizons (H≤10).

**Implementation cost:** Trivial. Remove from M6 config; mark spec §5.7 as deferred.

**Paper alignment:** WEAK. Lose the long-horizon evidence to existing matrix-game horizons (H≤10). Paper title becomes hard to defend without out-of-suite evidence.

---

## 7. Orchestrator's view

After three successive halts on this cell, the diagnosis is converging on a deeper issue: **the falsifiable prediction P-Contract assumed a β-monotonic ordering that doesn't actually hold on the chosen metric, because the TAB operator's fixed point shifts with β**.

Option (δ) — switch to Bellman-residual decay — is the most paper-aligned fix. It would require:

1. New metric helper or reuse of `contraction_reward` from spec §M5.
2. Smoke-test rewrite computing `T_β Q` per episode per β.
3. Theoretical justification: under the alignment condition, `T_β` is a strict contraction toward `Q*_β`; the Bellman-residual decay rate is then bounded below by `1 − d_{β,γ}` and is monotonic in β-sign (in the alignment direction).
4. Empirical validation: ideally a quick smoke run in the orchestrator session before M6 dispatch.

But this is a research-design decision (what metric anchors the paper-headline temporal-credit-assignment claim), not an engineering decision. After the v2 / v3 / v4 sequence already produced two halts, I'm not going to silently auto-pick option (δ) — even though the v4 directive said "auto-fix not allowed beyond v4 amendment". Per CLAUDE.md §3 + addendum §6, I escalate to user.

The user's prior pattern (halt memos with options enumerated, user picks remediation) has worked well. Halting and waiting is the right call.

---

## 8. State at halt

```
branch HEAD            : 10d58695 (HALT 2 commit; pushed to origin)
working tree           : DIRTY with v4 corrections applied to:
                          - docs/specs/phase_VIII_tab_six_games.md (P-Contract sign flipped + T11 flipped)
                          - experiments/adaptive_beta/tab_six_games/metrics.py (q_star off-by-one fixed)
                          - tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py
                            (q_star test updated; decay-rate test slowed to avoid eps saturation)
                          - tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py
                            (assertions flipped + terminal-state slice added)
v4 metric scaffolding  : SOUND (47 metric tests PASS)
v4 P-Contract sign     : SOUND (alignment-condition derivation correct)
v4 smoke prediction    : EMPIRICALLY WRONG (β=0 dominates on classical-Q* metric)

pytest sweep at working tree:
  - tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py: 47 PASS
  - tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py: 1 FAIL
  - rest of suite: untouched (1684 PASS + 2 SKIP at HEAD before v4 edits)

v4 corrections NOT yet committed (per HALT 2 discipline: keep v3+v4 working-
tree edits pending researcher decision so they can land atomically with the
final fix). Working tree preserved across this halt as well.
```

---

## 9. Three options in §6 above; orchestrator recommends option (δ)

Option (δ) — Bellman-residual decay as the metric — preserves the paper-headline temporal-credit-assignment claim and is theoretically aligned with the operator. It is a research-design call (what's the natural metric for the TAB operator's contraction story on long-horizon chains) but the math is on solid footing.

If user accepts option (δ): I dispatch a small orchestrator session that adds `bellman_residual_decay_smoke` helper, rewrites the smoke test, and verifies empirically before committing the v3+v4+v5 atomically.

If user accepts option (ε) or (ζ): trivial commits + checkpoint update. Resume autonomous run from M2 reopen close → M4 reopen → M6 (without the delayed_chain cell on M6 if (ζ)).

---

## 10. Pointers

- v3 patch: `tasks/phase_VIII_spec_patches_2026-05-01_v3_T11_resolution.md`
- v2 patch (parent): `tasks/phase_VIII_spec_patches_2026-05-01.md`
- HALT 1 memo: `results/adaptive_beta/tab_six_games/halts/delayed_chain_smoke_prediction_failure_2026-05-01T02-43-53Z.md`
- HALT 2 memo: `results/adaptive_beta/tab_six_games/halts/delayed_chain_P_Contract_sign_inverted_2026-05-01T11-14-49Z.md`
- spec §5.7 (with v3+v4 fold-ins, marked `<!-- patch-2026-05-01-v3 -->` and `<!-- patch-2026-05-01-v4 -->`)
- alignment condition: spec §3.3, §4.3
- Bellman residual / contraction reward (spec §7.2 / §M5): existing metric `contraction_reward` could be the helper for option (δ)

Per addendum §6: write halt memo, update checkpoint, write JSONL HALT event, STOP.
